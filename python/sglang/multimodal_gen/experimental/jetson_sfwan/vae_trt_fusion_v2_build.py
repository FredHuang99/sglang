"""Build the isolated producer-side SFWan VAE INT8 fusion-v2 steady plan.

The command never rebuilds or replaces the Q/DQ-v5 initial plan.  It analyzes
and rewrites only the steady graph, verifies all 84 Conv call sites still use
audited INT8 tactics, and commits its timing cache only after the mixed-cache
ABI and plugin-layer audit have passed.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from .vae_trt_build import (
    _audit_tensorrt_tactics,
    _build_engine_bytes,
    _write_bytes,
)
from .vae_trt_fusion import sha256_file, write_json_atomic
from .vae_trt_fusion_build import (
    _environment,
    _load_analysis_contracts,
    _run_analysis,
    _target_names,
    _validate_io,
)
from .vae_trt_fusion_v2 import (
    BOUNDARY_PLUGIN,
    EXPECTED_INT8_CACHE_SLOTS,
    FUSION_V2_ANALYSIS_FILE,
    FUSION_V2_AUDIT_FILE,
    FUSION_V2_AUDIT_SCHEMA_VERSION,
    FUSION_V2_BUILD_STATE_FILE,
    FUSION_V2_ENGINE_FILE,
    FUSION_V2_INSPECTOR_FILE,
    FUSION_V2_MANIFEST_FILE,
    FUSION_V2_ONNX_FILE,
    FUSION_V2_PLUGIN_LIBRARY_FILE,
    FUSION_V2_PLUGIN_MANIFEST_FILE,
    FUSION_V2_SCHEMA_VERSION,
    FUSION_V2_SUBDIRECTORY,
    FUSION_V2_TIMING_CACHE_FILE,
    FUSION_V2_VARIANT,
    PLUGIN_CREATORS,
    PLUGIN_INIT_SYMBOL,
    PLUGIN_NAMESPACE,
    PLUGIN_VERSION,
    analyze_fusion_v2_steady_graph,
    load_fusion_v2_manifest,
    rewrite_fusion_v2_steady_graph,
    validate_fusion_v2_manifest,
)
from .vae_trt_qdq import EXPECTED_CALL_SITES, QDQ_SCHEMA_VERSION
from .vae_trt_runtime import load_trt_vae_manifest, validate_trt_vae_manifest

_SOURCE_FILE = "steady_int8_qdq_v5.onnx"


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _load_plugin(*, trt: Any, path: Path) -> Any:
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    try:
        library = ctypes.CDLL(str(path), mode=mode)
    except OSError as exc:
        raise RuntimeError(f"could not load fusion-v2 plugin: {path}") from exc
    try:
        initialize = getattr(library, PLUGIN_INIT_SYMBOL)
    except AttributeError as exc:
        raise RuntimeError(
            f"fusion-v2 plugin does not export {PLUGIN_INIT_SYMBOL}"
        ) from exc
    initialize.argtypes = []
    initialize.restype = ctypes.c_bool
    if not initialize():
        raise RuntimeError("fusion-v2 plugin registration failed")
    registry = trt.get_plugin_registry()
    if registry is None:
        raise RuntimeError("TensorRT plugin registry is unavailable")
    for name in PLUGIN_CREATORS:
        creator = None
        getter = getattr(registry, "get_plugin_creator", None)
        if callable(getter):
            creator = getter(name, PLUGIN_VERSION, PLUGIN_NAMESPACE)
        if creator is None:
            getter = getattr(registry, "get_creator", None)
            if callable(getter):
                creator = getter(name, PLUGIN_VERSION, PLUGIN_NAMESPACE)
        if creator is None:
            raise RuntimeError(f"fusion-v2 creator was not registered: {name}")
    return library


def _base_manifest_identity(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _expected_steady_io(
    *, base_manifest: Mapping[str, Any], selected_indices: set[int]
) -> dict[str, tuple[str, list[int], str]]:
    expected: dict[str, tuple[str, list[int], str]] = {
        "latent": ("input", [1, 16, 3, 60, 104], "float16"),
        "rgb": ("output", [1, 3, 12, 480, 832], "float16"),
    }
    cache = base_manifest.get("cache")
    bindings = cache.get("bindings") if isinstance(cache, Mapping) else None
    if not isinstance(bindings, list) or len(bindings) != 32:
        raise ValueError("base manifest has no complete cache bindings")
    for index, binding in enumerate(bindings):
        shape = [int(value) for value in binding["shape"]]
        dtype = "int8" if index in selected_indices else "float16"
        expected[f"cache_in_{index:03d}"] = ("input", shape, dtype)
        expected[f"cache_out_{index:03d}"] = ("output", shape, dtype)
    return expected


def _inspector_texts(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError("TensorRT returned invalid inspector JSON") from exc
    if isinstance(parsed, dict):
        records = parsed.get("Layers", parsed.get("layers", [parsed]))
    else:
        records = parsed
    if not isinstance(records, list):
        records = []
    return [
        json.dumps(record, sort_keys=True)
        for record in records
        if isinstance(record, dict)
    ]


def _plugin_layer_audit(inspector_json: str) -> dict[str, Any]:
    texts = _inspector_texts(inspector_json)
    counts = {
        "norm1_silu_pack_quant": sum(
            "fusion_v2/norm1_silu_quant_pack/" in text for text in texts
        ),
        "conv1_norm2_silu_pack_quant": sum(
            "fusion_v2/conv1_to_conv2_norm_silu_quant_pack/" in text
            for text in texts
        ),
        "conv2_residual_next_norm_pack_quant": sum(
            "fusion_v2/conv2_residual_next_norm_silu_quant_pack/" in text
            for text in texts
        ),
        "conv2_residual_tail": sum(
            "fusion_v2/conv2_residual_tail/" in text for text in texts
        ),
    }
    expected = {
        "norm1_silu_pack_quant": 3,
        "conv1_norm2_silu_pack_quant": 9,
        "conv2_residual_next_norm_pack_quant": 6,
        "conv2_residual_tail": 3,
    }
    return {
        "passed": counts == expected,
        "counts": counts,
        "expected": expected,
        "creator_present": any(
            BOUNDARY_PLUGIN in text or "fusion_v2/" in text for text in texts
        ),
    }


def _state(
    *, root: Path, stage: str, identity: Mapping[str, Any], details: Any = None
) -> None:
    write_json_atomic(
        root / FUSION_V2_BUILD_STATE_FILE,
        {
            "schema_version": FUSION_V2_SCHEMA_VERSION,
            "variant": FUSION_V2_VARIANT,
            "stage": stage,
            "identity": dict(identity),
            "details": details,
        },
    )


def build_fusion_v2(
    *,
    engine_dir: str | Path,
    stage: str,
    workspace_gib: float,
    resume: bool,
    device_index: int,
    plugin_library: str | Path | None,
) -> dict[str, Any]:
    if stage not in {"analyze", "build", "all"}:
        raise ValueError("--stage must be analyze, build, or all")
    if workspace_gib <= 0:
        raise ValueError("--workspace-gib must be positive")
    try:
        import onnx
        import tensorrt as trt
        import torch
    except ImportError as exc:
        raise RuntimeError("fusion-v2 build requires onnx, tensorrt, and torch") from exc

    base_root = Path(engine_dir).expanduser().resolve()
    fusion_root = base_root / FUSION_V2_SUBDIRECTORY
    fusion_root.mkdir(parents=True, exist_ok=True)
    base_manifest = load_trt_vae_manifest(base_root)
    base_validated = validate_trt_vae_manifest(
        base_manifest,
        engine_dir=base_root,
        precision="int8",
        model_path=None,
        verify_plan_hashes=True,
    )
    base_audit_path = base_root / base_manifest["int8_audit"]["report_file"]
    try:
        base_audit = json.loads(base_audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("base v5 audit report is unreadable") from exc
    if not isinstance(base_audit, dict) or base_audit.get("passed") is not True:
        raise ValueError("base v5 audit report did not pass")
    source_path = base_root / _SOURCE_FILE
    if not source_path.is_file():
        raise ValueError(f"v5 steady Q/DQ source does not exist: {source_path}")
    destination_plugin = fusion_root / FUSION_V2_PLUGIN_LIBRARY_FILE
    if plugin_library is not None:
        source_plugin = Path(plugin_library).expanduser().resolve()
        if not source_plugin.is_file():
            raise ValueError(f"fusion-v2 plugin does not exist: {source_plugin}")
        if source_plugin != destination_plugin:
            shutil.copy2(source_plugin, destination_plugin)
    if not destination_plugin.is_file():
        raise ValueError(
            f"install {FUSION_V2_PLUGIN_LIBRARY_FILE} into {fusion_root} first"
        )
    environment = _environment(
        trt=trt, torch=torch, onnx=onnx, device_index=device_index
    )
    plugin_manifest = {
        "schema_version": FUSION_V2_SCHEMA_VERSION,
        "variant": FUSION_V2_VARIANT,
        "file": FUSION_V2_PLUGIN_LIBRARY_FILE,
        "sha256": sha256_file(destination_plugin),
        "init_symbol": PLUGIN_INIT_SYMBOL,
        "creators": list(PLUGIN_CREATORS),
        "plugin_version": PLUGIN_VERSION,
        "plugin_namespace": PLUGIN_NAMESPACE,
        "build": environment,
    }
    write_json_atomic(
        fusion_root / FUSION_V2_PLUGIN_MANIFEST_FILE, plugin_manifest
    )
    library = _load_plugin(trt=trt, path=destination_plugin)
    del library

    target_names = _target_names(base_manifest)
    contracts = _load_analysis_contracts(
        base_root=base_root, base_manifest=base_manifest
    )
    base_analysis = _run_analysis(
        source_paths={
            "initial": base_root / "initial_int8_qdq_v5.onnx",
            "steady": source_path,
        },
        target_names=target_names,
        focus_module_prefix="decoder.up_blocks.3",
        analysis_contracts=contracts,
    )
    if base_analysis.get("passed") is not True:
        raise RuntimeError(f"base fusion analysis failed: {base_analysis['errors']}")
    analysis = analyze_fusion_v2_steady_graph(
        source_path=source_path,
        base_analysis=base_analysis["graphs"]["steady"],
    )
    write_json_atomic(fusion_root / FUSION_V2_ANALYSIS_FILE, analysis)
    identity = {
        "base_manifest_sha256": _base_manifest_identity(base_manifest),
        "source_sha256": sha256_file(source_path),
        "plugin_sha256": plugin_manifest["sha256"],
        "workspace_gib": float(workspace_gib),
        "environment": environment,
    }
    if analysis.get("passed") is not True:
        _state(root=fusion_root, stage="analysis_failed", identity=identity, details=analysis["errors"])
        raise RuntimeError(f"fusion-v2 analysis failed: {analysis['errors']}")
    _state(root=fusion_root, stage="analyzed", identity=identity)
    if stage == "analyze":
        return analysis

    if resume and (fusion_root / FUSION_V2_MANIFEST_FILE).is_file():
        existing = load_fusion_v2_manifest(base_root)
        if existing.get("identity") == identity:
            validate_fusion_v2_manifest(
                existing,
                engine_dir=base_root,
                base_manifest=base_manifest,
                verify_hashes=True,
            )
            return existing

    graph_record = rewrite_fusion_v2_steady_graph(
        source_path=source_path,
        destination_path=fusion_root / FUSION_V2_ONNX_FILE,
        analysis=analysis,
    )
    stable_cache = fusion_root / FUSION_V2_TIMING_CACHE_FILE
    candidate_cache = fusion_root / f"{FUSION_V2_TIMING_CACHE_FILE}.candidate"
    candidate_cache.unlink(missing_ok=True)
    if stable_cache.is_file():
        _write_bytes(candidate_cache, stable_cache.read_bytes())
    _state(root=fusion_root, stage="building", identity=identity)
    plan, io_contract, inspector_json, candidate_bytes = _build_engine_bytes(
        trt=trt,
        onnx_path=fusion_root / FUSION_V2_ONNX_FILE,
        workspace_gib=workspace_gib,
        profiling_verbosity="detailed",
        timing_cache_path=candidate_cache,
    )
    selected_indices = {int(entry["index"]) for entry in analysis["selected_cache_slots"]}
    _validate_io(
        records=io_contract,
        expected=_expected_steady_io(
            base_manifest=base_manifest, selected_indices=selected_indices
        ),
    )
    weight_encoding = base_manifest["quantization"]["weight_encoding"]
    all_call_sites = [
        record["call_site"]
        for record in base_analysis["graphs"]["steady"]["call_sites"]
    ]
    tactic = _audit_tensorrt_tactics(
        graph_kind="steady",
        call_site_names=all_call_sites,
        inspector_json=inspector_json,
        expected_call_sites=EXPECTED_CALL_SITES,
        weight_encoding=weight_encoding,
    )
    plugin_audit = _plugin_layer_audit(inspector_json)
    int8_bindings = sorted(
        record["name"]
        for record in io_contract
        if str(record.get("dtype", "")).lower() == "int8"
    )
    expected_int8_bindings = sorted(
        [f"cache_in_{index:03d}" for index in selected_indices]
        + [f"cache_out_{index:03d}" for index in selected_indices]
    )
    errors: list[str] = []
    if tactic.get("passed") is not True:
        errors.append("one or more Conv3d call sites lost their INT8 tactic")
    if plugin_audit["passed"] is not True or not plugin_audit["creator_present"]:
        errors.append("fusion-v2 physical plugin layer count is incomplete")
    if int8_bindings != expected_int8_bindings:
        errors.append("fusion-v2 mixed-cache binding dtype contract changed")
    plan_sha = hashlib.sha256(plan).hexdigest()
    audit = {
        "schema_version": FUSION_V2_AUDIT_SCHEMA_VERSION,
        "variant": FUSION_V2_VARIANT,
        "qdq_schema_version": QDQ_SCHEMA_VERSION,
        "weight_encoding": weight_encoding,
        "passed": not errors,
        "complete": not errors,
        "errors": errors,
        "steady_plan_sha256": plan_sha,
        "plan_sha256": {
            "initial": base_validated["engines"]["initial"]["sha256"],
            "steady": plan_sha,
        },
        "int8_target_conv_call_site_count": (
            EXPECTED_CALL_SITES if tactic.get("passed") is True else None
        ),
        "tactic": tactic,
        "tactics": {
            "initial": base_audit["tactics"]["initial"],
            "steady": tactic,
        },
        "plugin": plugin_audit,
        "cache": {
            "selected_int8_slot_count": len(selected_indices),
            "selected_indices": sorted(selected_indices),
            "int8_bindings": int8_bindings,
            "fp16_binding_count": sum(
                str(record.get("dtype", "")).lower() in {"float16", "half"}
                for record in io_contract
            ),
        },
    }
    write_json_atomic(fusion_root / FUSION_V2_AUDIT_FILE, audit)
    _write_bytes(fusion_root / FUSION_V2_ENGINE_FILE, plan)
    inspector_value = json.loads(inspector_json)
    write_json_atomic(fusion_root / FUSION_V2_INSPECTOR_FILE, inspector_value)
    if errors:
        _state(root=fusion_root, stage="audit_failed", identity=identity, details=errors)
        raise RuntimeError(f"fusion-v2 audit failed: {errors}")
    if candidate_bytes is not None:
        _write_bytes(candidate_cache, candidate_bytes)
    if candidate_cache.is_file():
        candidate_cache.replace(stable_cache)

    cache_slots = [
        {
            **dict(entry),
            "dtype": "int8",
            "format": "cdhw32",
        }
        for entry in analysis["selected_cache_slots"]
    ]
    manifest = {
        "schema_version": FUSION_V2_SCHEMA_VERSION,
        "variant": FUSION_V2_VARIANT,
        "identity": identity,
        "base_manifest_sha256": identity["base_manifest_sha256"],
        "initial_plan_kind": "raw_int8_v5",
        "initial_plan_sha256": base_validated["engines"]["initial"]["sha256"],
        "steady_engine": {
            "file": FUSION_V2_ENGINE_FILE,
            "sha256": plan_sha,
            "source_onnx_file": FUSION_V2_ONNX_FILE,
            "source_onnx_sha256": sha256_file(
                fusion_root / FUSION_V2_ONNX_FILE
            ),
            "inspector_file": FUSION_V2_INSPECTOR_FILE,
            "inspector_sha256": sha256_file(
                fusion_root / FUSION_V2_INSPECTOR_FILE
            ),
            "profiling_verbosity": "detailed",
            "io_tensors": io_contract,
        },
        "plugin": plugin_manifest,
        "cache": {
            "tensor_count": 32,
            "selected_slots": cache_slots,
            "selected_slot_count": EXPECTED_INT8_CACHE_SLOTS,
            "other_slot_count": 32 - EXPECTED_INT8_CACHE_SLOTS,
            "initial_dtype": "float16",
            "steady_mixed_dtype": True,
            "migration": "one_time_after_chunk_0",
        },
        "graph": graph_record,
        "audit": {
            "file": FUSION_V2_AUDIT_FILE,
            "sha256": sha256_file(fusion_root / FUSION_V2_AUDIT_FILE),
            "passed": True,
        },
        "build": environment,
    }
    write_json_atomic(fusion_root / FUSION_V2_MANIFEST_FILE, manifest)
    validate_fusion_v2_manifest(
        manifest,
        engine_dir=base_root,
        base_manifest=base_manifest,
        verify_hashes=True,
    )
    _state(root=fusion_root, stage="completed", identity=identity)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SFWan VAE TensorRT producer-side fusion-v2"
    )
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--stage", choices=("analyze", "build", "all"), default="all")
    parser.add_argument("--workspace-gib", type=float, default=8.0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--plugin-library")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_fusion_v2(
        engine_dir=args.engine_dir,
        stage=args.stage,
        workspace_gib=args.workspace_gib,
        resume=args.resume,
        device_index=args.device_index,
        plugin_library=args.plugin_library,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
