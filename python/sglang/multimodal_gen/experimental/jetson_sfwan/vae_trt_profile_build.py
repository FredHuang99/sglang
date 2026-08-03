"""Build isolated DETAILED FP16 plans for SFWan VAE layer profiling.

The command consumes an already completed TensorRT VAE engine directory.  It
builds only the FP16 control plans used by the opt-in layer profiler and
references (but never rebuilds) the audited INT8 Q/DQ v5 plans.  Every output
uses a layer-profile-specific name so a failed or interrupted diagnostic build
cannot replace production plans, manifests, audits, or timing caches.

This command must run on the target GPU.  TensorRT plans and timing caches are
specific to the CUDA, TensorRT, and compute-capability environment that creates
them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .vae_trt_build import (
    _audit_feature_cache_kind_io,
    _build_engine_bytes,
    _commit_timing_cache,
    _load_plan,
    _load_validated_fp16_onnx,
    _sha256_file,
    _write_bytes,
    _write_json,
)
from .vae_trt_profile import build_physical_layer_catalog
from .vae_trt_qdq import (
    EXPECTED_CALL_SITES,
    EXPECTED_LOGICAL_CONVS,
    QDQ_OPSET,
    QDQ_SCHEMA_VERSION,
    collect_fp16_target_call_sites,
)
from .vae_trt_runtime import (
    TRT_VAE_CACHE_BANK_BYTES,
    TRT_VAE_CACHE_COUNT,
    TRT_VAE_CACHE_TOTAL_ELEMENTS,
    TRT_VAE_LATENT_SHAPE,
    load_trt_vae_manifest,
    validate_trt_vae_manifest,
)

TRT_LAYER_PROFILE_MANIFEST_SCHEMA_VERSION = 1
TRT_LAYER_PROFILE_SCOPE = "vae_profile_only"
_KINDS = ("initial", "steady")
_SOURCE_FILES = {
    "initial": "initial_fp16_opset19.onnx",
    "steady": "steady_fp16_opset19.onnx",
}
_PROFILE_PLAN_FILES = {
    "initial": "initial_fp16_layer_profile.plan",
    "steady": "steady_fp16_layer_profile.plan",
}
_PROFILE_INSPECTOR_FILES = {
    "initial": "initial_fp16_layer_profile_inspector.json",
    "steady": "steady_fp16_layer_profile_inspector.json",
}
_PROFILE_MANIFEST_FILE = "trt_layer_profile_manifest.json"
_PROFILE_STATE_FILE = "trt_layer_profile_build_state.json"
_PROFILE_TIMING_CACHE_FILE = "trt_layer_profile_timing.cache"


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required JSON file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_inspector_json(path: Path) -> dict[str, Any] | list[Any]:
    if not path.is_file():
        raise ValueError(f"required inspector file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid inspector JSON file: {path}") from exc
    if not isinstance(value, (dict, list)):
        raise ValueError(f"{path} must contain a JSON object or array")
    return value


def _resolve_inside(root: Path, value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the engine directory") from exc
    return path


def _relative_file(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root))


def _canonical_dtype(value: Any) -> str:
    normalized = str(value).lower().split(".")[-1]
    return "float16" if normalized in {"half", "float16"} else normalized


def _production_cache_contract(
    manifest: dict[str, Any],
) -> tuple[list[tuple[int, ...]], list[str], list[str]]:
    cache = manifest.get("cache")
    if not isinstance(cache, dict):
        raise ValueError("production manifest has no cache contract")
    bindings = cache.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != TRT_VAE_CACHE_COUNT:
        raise ValueError("production manifest must contain 32 cache bindings")

    cache_shapes: list[tuple[int, ...]] = []
    input_names: list[str] = []
    output_names: list[str] = []
    for expected_index, binding in enumerate(bindings):
        if not isinstance(binding, dict) or binding.get("index") != expected_index:
            raise ValueError("production cache bindings must be ordered 0..31")
        expected_input_name = f"cache_in_{expected_index:03d}"
        expected_output_name = f"cache_out_{expected_index:03d}"
        input_name = binding.get("input_name")
        output_name = binding.get("output_name")
        if input_name != expected_input_name or output_name != expected_output_name:
            raise ValueError(
                "production cache binding names do not match the fixed TensorRT ABI: "
                f"{input_name!r}, {output_name!r}"
            )
        if binding.get("dtype") != "float16":
            raise ValueError("production feature-cache bindings must be float16")
        raw_shape = binding.get("shape")
        if (
            not isinstance(raw_shape, list)
            or len(raw_shape) != 5
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in raw_shape
            )
        ):
            raise ValueError(f"invalid cache shape at index {expected_index}")
        cache_shapes.append(tuple(raw_shape))
        input_names.append(input_name)
        output_names.append(output_name)

    if sum(math.prod(shape) for shape in cache_shapes) != (
        TRT_VAE_CACHE_TOTAL_ELEMENTS
    ):
        raise ValueError("production feature-cache element count is invalid")
    return cache_shapes, input_names, output_names


def _expected_io(
    *,
    kind: str,
    manifest: dict[str, Any],
    cache_shapes: list[tuple[int, ...]],
    cache_input_names: list[str],
    cache_output_names: list[str],
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    if kind not in _KINDS:
        raise ValueError(f"unsupported TensorRT VAE engine kind: {kind}")
    latent_shape = tuple(manifest.get("latent_shape", ()))
    if latent_shape != TRT_VAE_LATENT_SHAPE:
        raise ValueError(f"invalid production latent shape: {latent_shape}")
    height = int(manifest.get("height", -1))
    width = int(manifest.get("width", -1))
    rgb_frames = 9 if kind == "initial" else 12

    inputs: dict[str, tuple[int, ...]] = {"latent": latent_shape}
    if kind == "steady":
        inputs.update(dict(zip(cache_input_names, cache_shapes, strict=True)))
    outputs: dict[str, tuple[int, ...]] = {"rgb": (1, 3, rgb_frames, height, width)}
    outputs.update(dict(zip(cache_output_names, cache_shapes, strict=True)))
    return inputs, outputs


def _validate_engine_io(
    *,
    kind: str,
    records: list[dict[str, Any]],
    expected_inputs: dict[str, tuple[int, ...]],
    expected_outputs: dict[str, tuple[int, ...]],
) -> None:
    if not isinstance(records, list):
        raise ValueError(f"TensorRT {kind} engine I/O contract is not a list")
    by_name: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"TensorRT {kind} engine has an invalid I/O record")
        name = record.get("name")
        if not isinstance(name, str) or not name or name in by_name:
            raise ValueError(f"TensorRT {kind} engine has duplicate/invalid I/O names")
        by_name[name] = record

    expected_names = set(expected_inputs) | set(expected_outputs)
    if set(by_name) != expected_names:
        raise ValueError(
            f"TensorRT {kind} engine I/O names differ from the fixed ABI; "
            f"missing={sorted(expected_names - set(by_name))}, "
            f"extra={sorted(set(by_name) - expected_names)}"
        )
    for mode, expected in (("input", expected_inputs), ("output", expected_outputs)):
        for name, shape in expected.items():
            record = by_name[name]
            if str(record.get("mode", "")).lower() != mode:
                raise ValueError(f"TensorRT {kind} binding {name!r} must be an {mode}")
            if _canonical_dtype(record.get("dtype")) != "float16":
                raise ValueError(f"TensorRT {kind} binding {name!r} must be float16")
            if tuple(record.get("shape", ())) != shape:
                raise ValueError(
                    f"TensorRT {kind} binding {name!r} shape is "
                    f"{record.get('shape')}, expected {shape}"
                )

    cache_audit = _audit_feature_cache_kind_io(kind=kind, records=records)
    if cache_audit.get("passed") is not True:
        raise ValueError(
            f"TensorRT {kind} feature-cache I/O audit failed: "
            f"{cache_audit.get('errors')}"
        )


def _validate_source_record(
    *,
    root: Path,
    manifest: dict[str, Any],
    kind: str,
    source_path: Path,
) -> str:
    build = manifest.get("build")
    qdq_sources = build.get("qdq_source_onnx") if isinstance(build, dict) else None
    record = qdq_sources.get(kind) if isinstance(qdq_sources, dict) else None
    if not isinstance(record, dict):
        raise ValueError(f"production manifest has no Q/DQ source record for {kind}")
    if record.get("file") != _SOURCE_FILES[kind]:
        raise ValueError(f"production {kind} Q/DQ source filename is inconsistent")
    if record.get("opset") != QDQ_OPSET:
        raise ValueError(f"production {kind} Q/DQ source must use opset {QDQ_OPSET}")
    if source_path != (root / record["file"]).resolve() or not source_path.is_file():
        raise ValueError(f"required FP16 source ONNX does not exist: {source_path}")
    digest = _sha256_file(source_path)
    if record.get("sha256") != digest:
        raise ValueError(f"production {kind} Q/DQ source ONNX digest mismatch")
    return digest


def _protected_production_paths(*, root: Path, manifest: dict[str, Any]) -> set[Path]:
    protected = {(root / "manifest.json").resolve()}
    engines = manifest.get("engines")
    if isinstance(engines, dict):
        for precision_engines in engines.values():
            if not isinstance(precision_engines, dict):
                continue
            for record in precision_engines.values():
                if isinstance(record, dict) and isinstance(record.get("file"), str):
                    protected.add(_resolve_inside(root, record["file"], label="plan"))
    int8_audit = manifest.get("int8_audit")
    if isinstance(int8_audit, dict) and isinstance(int8_audit.get("report_file"), str):
        protected.add(
            _resolve_inside(root, int8_audit["report_file"], label="INT8 audit")
        )
    build = manifest.get("build")
    if isinstance(build, dict) and isinstance(build.get("timing_cache_file"), str):
        raw_path = Path(build["timing_cache_file"]).expanduser()
        if not raw_path.is_absolute():
            raw_path = root / raw_path
        protected.add(raw_path.resolve())
    return protected


def _validate_profile_outputs_are_isolated(
    *, root: Path, manifest: dict[str, Any], timing_cache_path: Path
) -> None:
    protected = _protected_production_paths(root=root, manifest=manifest)
    profile_outputs = {
        (root / _PROFILE_MANIFEST_FILE).resolve(),
        (root / _PROFILE_STATE_FILE).resolve(),
        timing_cache_path.resolve(),
        *((root / name).resolve() for name in _PROFILE_PLAN_FILES.values()),
        *((root / name).resolve() for name in _PROFILE_INSPECTOR_FILES.values()),
    }
    collisions = sorted(str(path) for path in profile_outputs & protected)
    if collisions:
        raise ValueError(
            "layer-profile output paths collide with production artifacts: "
            f"{collisions}"
        )


def _environment_identity(
    *,
    trt: Any,
    torch: Any,
    onnx: Any,
    device_index: int,
    workspace_gib: float,
    root: Path,
    manifest_path: Path,
    production_manifest: dict[str, Any],
    source_sha256: dict[str, str],
    int8_plan_sha256: dict[str, str],
    int8_audit_path: Path,
    timing_cache_path: Path,
) -> dict[str, Any]:
    capability = list(torch.cuda.get_device_capability(device_index))
    production_build = production_manifest.get("build")
    if not isinstance(production_build, dict):
        raise ValueError("production manifest has no build environment")
    expected_capability = production_build.get("compute_capability")
    if capability != expected_capability:
        raise ValueError(
            f"current compute capability {capability} differs from production "
            f"plans {expected_capability}"
        )
    current_trt = str(trt.__version__)
    current_cuda = str(torch.version.cuda)
    if current_trt != str(production_build.get("tensorrt_version")):
        raise ValueError("current TensorRT version differs from production plans")
    if current_cuda != str(production_build.get("cuda_version")):
        raise ValueError("current CUDA version differs from production plans")

    return {
        "schema_version": TRT_LAYER_PROFILE_MANIFEST_SCHEMA_VERSION,
        "scope": TRT_LAYER_PROFILE_SCOPE,
        "engine_dir": str(root),
        "production_manifest_sha256": _sha256_file(manifest_path),
        "model_id": production_manifest.get("model_id"),
        "source_onnx_sha256": source_sha256,
        "int8_plan_sha256": int8_plan_sha256,
        "int8_audit_sha256": _sha256_file(int8_audit_path),
        "compute_capability": capability,
        "device_name": torch.cuda.get_device_name(device_index),
        "device_index": device_index,
        "torch_version": str(torch.__version__),
        "cuda_version": current_cuda,
        "tensorrt_version": current_trt,
        "onnx_version": str(onnx.__version__),
        "workspace_gib": float(workspace_gib),
        "network_definition": ["explicit_batch", "strongly_typed"],
        "profiling_verbosity": "detailed",
        "timing_cache_file": _relative_file(root, timing_cache_path),
    }


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _load_json_object(path)


def _record_stage(
    *,
    state_path: Path,
    state: dict[str, Any],
    kind: str,
    status: str,
    source_sha256: str,
    plan_path: Path,
    inspector_path: Path,
    error: str | None = None,
) -> None:
    if status not in {"building", "failed", "completed"}:
        raise ValueError(f"invalid layer-profile build status: {status}")
    record: dict[str, Any] = {
        "status": status,
        "source_sha256": source_sha256,
        "plan_file": plan_path.name,
        "inspector_file": inspector_path.name,
        "profiling_verbosity": "detailed",
    }
    if plan_path.is_file():
        record["plan_sha256"] = _sha256_file(plan_path)
    if inspector_path.is_file():
        record["inspector_sha256"] = _sha256_file(inspector_path)
    if error is not None:
        record["error"] = error
    state.setdefault("stages", {})[kind] = record
    _write_json(state_path, state)


def _resume_stage(
    *,
    trt: Any,
    state: dict[str, Any],
    kind: str,
    source_sha256: str,
    plan_path: Path,
    inspector_path: Path,
    expected_inputs: dict[str, tuple[int, ...]],
    expected_outputs: dict[str, tuple[int, ...]],
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    record = state.get("stages", {}).get(kind)
    if not isinstance(record, dict) or record.get("status") != "completed":
        return None
    if (
        record.get("source_sha256") != source_sha256
        or record.get("profiling_verbosity") != "detailed"
        or record.get("plan_file") != plan_path.name
        or record.get("inspector_file") != inspector_path.name
        or not plan_path.is_file()
        or not inspector_path.is_file()
        or record.get("plan_sha256") != _sha256_file(plan_path)
        or record.get("inspector_sha256") != _sha256_file(inspector_path)
    ):
        return None
    try:
        inspector = _load_inspector_json(inspector_path)
        io_contract, live_inspector_json = _load_plan(trt=trt, plan_path=plan_path)
        live_inspector = json.loads(live_inspector_json)
        if not isinstance(live_inspector, (dict, list)):
            raise ValueError("TensorRT engine inspector returned invalid JSON")
        _validate_engine_io(
            kind=kind,
            records=io_contract,
            expected_inputs=expected_inputs,
            expected_outputs=expected_outputs,
        )
        if inspector != live_inspector:
            raise ValueError("saved and live TensorRT inspector records differ")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Layer-profile stage {kind} is not reusable: {exc}")
        return None
    print(f"Resuming completed layer-profile stage {kind}: {plan_path}")
    return io_contract, inspector


def build_layer_profile_plans(
    *,
    engine_dir: str | Path,
    workspace_gib: float = 8.0,
    device_index: int = 0,
    resume: bool = False,
    timing_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build isolated FP16 DETAILED plans and reference audited INT8 v5 plans."""

    if workspace_gib <= 0:
        raise ValueError("--workspace-gib must be positive")

    import onnx
    import tensorrt as trt
    import torch

    root = Path(engine_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"TensorRT VAE engine directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    production_manifest = load_trt_vae_manifest(root)
    model_id = production_manifest.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("production TensorRT VAE manifest has no model_id")
    validated_int8 = validate_trt_vae_manifest(
        production_manifest,
        engine_dir=root,
        precision="int8",
        model_path=model_id,
        verify_plan_hashes=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT layer-profile plans require CUDA")
    if device_index < 0 or device_index >= int(torch.cuda.device_count()):
        raise ValueError(f"invalid CUDA device index: {device_index}")
    torch.cuda.set_device(device_index)

    cache_shapes, cache_input_names, cache_output_names = _production_cache_contract(
        production_manifest
    )
    expected_io = {
        kind: _expected_io(
            kind=kind,
            manifest=production_manifest,
            cache_shapes=cache_shapes,
            cache_input_names=cache_input_names,
            cache_output_names=cache_output_names,
        )
        for kind in _KINDS
    }

    source_paths = {kind: (root / _SOURCE_FILES[kind]).resolve() for kind in _KINDS}
    source_sha256 = {
        kind: _validate_source_record(
            root=root,
            manifest=production_manifest,
            kind=kind,
            source_path=source_paths[kind],
        )
        for kind in _KINDS
    }
    quantization = production_manifest.get("quantization")
    raw_target_module_names = (
        quantization.get("target_module_names")
        if isinstance(quantization, dict)
        else None
    )
    if (
        not isinstance(raw_target_module_names, list)
        or len(raw_target_module_names) != EXPECTED_LOGICAL_CONVS
        or any(
            not isinstance(module_name, str) or not module_name
            for module_name in raw_target_module_names
        )
        or len(set(raw_target_module_names)) != EXPECTED_LOGICAL_CONVS
    ):
        raise ValueError(
            "production manifest must identify the 28 v5 target residual Convs"
        )
    target_module_names = tuple(raw_target_module_names)
    fp16_target_call_sites: dict[str, list[dict[str, Any]]] = {}
    for kind in _KINDS:
        inputs, outputs = expected_io[kind]
        source_model, _opset = _load_validated_fp16_onnx(
            path=source_paths[kind],
            expected_input_shapes=inputs,
            expected_output_shapes=outputs,
            required_opset=QDQ_OPSET,
        )
        del source_model
        fp16_target_call_sites[kind] = collect_fp16_target_call_sites(
            source_path=source_paths[kind],
            graph_kind=kind,
            target_module_names=target_module_names,
        )
        if len(fp16_target_call_sites[kind]) != EXPECTED_CALL_SITES:
            raise ValueError(
                f"FP16 {kind} source target map is incomplete: "
                f"{len(fp16_target_call_sites[kind])}"
            )

    int8_plans: dict[str, Path] = {}
    int8_io: dict[str, list[dict[str, Any]]] = {}
    int8_inspector_paths: dict[str, Path] = {}
    for kind in _KINDS:
        plan_path = Path(validated_int8["engines"][kind]["path"]).resolve()
        try:
            plan_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("audited INT8 plan escapes the engine directory") from exc
        io_contract, live_inspector_json = _load_plan(
            trt=trt,
            plan_path=plan_path,
        )
        live_inspector = json.loads(live_inspector_json)
        if not isinstance(live_inspector, (dict, list)):
            raise ValueError("TensorRT INT8 engine inspector returned invalid JSON")
        inspector_path = root / f"{kind}_int8_qdq_v5_inspector.json"
        saved_inspector = _load_inspector_json(inspector_path)
        if saved_inspector != live_inspector:
            raise ValueError(
                f"saved TensorRT INT8 {kind} inspector does not match its plan"
            )
        inputs, outputs = expected_io[kind]
        _validate_engine_io(
            kind=kind,
            records=io_contract,
            expected_inputs=inputs,
            expected_outputs=outputs,
        )
        int8_plans[kind] = plan_path
        int8_io[kind] = io_contract
        int8_inspector_paths[kind] = inspector_path

    int8_audit_record = production_manifest.get("int8_audit")
    if not isinstance(int8_audit_record, dict):
        raise ValueError("production manifest has no INT8 audit record")
    int8_audit_path = _resolve_inside(
        root,
        int8_audit_record.get("report_file", ""),
        label="INT8 audit",
    )
    int8_audit = _load_json_object(int8_audit_path)
    if (
        int8_audit.get("passed") is not True
        or int8_audit.get("complete") is not True
        or int8_audit.get("schema_version") != QDQ_SCHEMA_VERSION
    ):
        raise ValueError("production INT8 v5 audit is not complete and passing")

    resolved_timing_cache = _resolve_inside(
        root,
        timing_cache_path or _PROFILE_TIMING_CACHE_FILE,
        label="layer-profile timing cache",
    )
    resolved_timing_cache.parent.mkdir(parents=True, exist_ok=True)
    _validate_profile_outputs_are_isolated(
        root=root,
        manifest=production_manifest,
        timing_cache_path=resolved_timing_cache,
    )

    identity = _environment_identity(
        trt=trt,
        torch=torch,
        onnx=onnx,
        device_index=device_index,
        workspace_gib=workspace_gib,
        root=root,
        manifest_path=manifest_path,
        production_manifest=production_manifest,
        source_sha256=source_sha256,
        int8_plan_sha256={kind: _sha256_file(int8_plans[kind]) for kind in _KINDS},
        int8_audit_path=int8_audit_path,
        timing_cache_path=resolved_timing_cache,
    )
    state_path = root / _PROFILE_STATE_FILE
    state = _load_state(state_path) if resume else {}
    if state and state.get("identity") != identity:
        raise ValueError(
            f"{state_path} belongs to a different layer-profile build; rerun "
            "without --resume or use the original environment and arguments"
        )
    if not state:
        state = {
            "schema_version": TRT_LAYER_PROFILE_MANIFEST_SCHEMA_VERSION,
            "identity": identity,
            "stages": {},
        }
    _write_json(state_path, state)

    fp16_records: dict[str, dict[str, Any]] = {}
    for kind in _KINDS:
        source_path = source_paths[kind]
        plan_path = root / _PROFILE_PLAN_FILES[kind]
        inspector_path = root / _PROFILE_INSPECTOR_FILES[kind]
        inputs, outputs = expected_io[kind]
        resumed = (
            _resume_stage(
                trt=trt,
                state=state,
                kind=kind,
                source_sha256=source_sha256[kind],
                plan_path=plan_path,
                inspector_path=inspector_path,
                expected_inputs=inputs,
                expected_outputs=outputs,
            )
            if resume
            else None
        )
        if resumed is None:
            _record_stage(
                state_path=state_path,
                state=state,
                kind=kind,
                status="building",
                source_sha256=source_sha256[kind],
                plan_path=plan_path,
                inspector_path=inspector_path,
            )
            try:
                plan, io_contract, inspector_json, candidate_cache = (
                    _build_engine_bytes(
                        trt=trt,
                        onnx_path=source_path,
                        workspace_gib=workspace_gib,
                        profiling_verbosity="detailed",
                        timing_cache_path=resolved_timing_cache,
                    )
                )
                inspector = json.loads(inspector_json)
                if not isinstance(inspector, (dict, list)):
                    raise ValueError("TensorRT engine inspector returned invalid JSON")
                _validate_engine_io(
                    kind=kind,
                    records=io_contract,
                    expected_inputs=inputs,
                    expected_outputs=outputs,
                )
                catalog = build_physical_layer_catalog(
                    engine_kind=kind,
                    plan_sha256=hashlib.sha256(plan).hexdigest(),
                    inspector=inspector,
                    precision="fp16",
                    fp16_target_call_sites=fp16_target_call_sites[kind],
                )
                _write_bytes(plan_path, plan)
                _write_json(inspector_path, inspector)
                _commit_timing_cache(
                    timing_cache_path=resolved_timing_cache,
                    candidate=candidate_cache,
                )
                _record_stage(
                    state_path=state_path,
                    state=state,
                    kind=kind,
                    status="completed",
                    source_sha256=source_sha256[kind],
                    plan_path=plan_path,
                    inspector_path=inspector_path,
                )
            except BaseException as exc:
                _record_stage(
                    state_path=state_path,
                    state=state,
                    kind=kind,
                    status="failed",
                    source_sha256=source_sha256[kind],
                    plan_path=plan_path,
                    inspector_path=inspector_path,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
        else:
            io_contract, inspector = resumed

        catalog = build_physical_layer_catalog(
            engine_kind=kind,
            plan_sha256=_sha256_file(plan_path),
            inspector=inspector,
            precision="fp16",
            fp16_target_call_sites=fp16_target_call_sites[kind],
        )

        fp16_records[kind] = {
            "file": _relative_file(root, plan_path),
            "sha256": _sha256_file(plan_path),
            "source_onnx_file": _relative_file(root, source_path),
            "source_onnx_sha256": source_sha256[kind],
            "inspector_file": _relative_file(root, inspector_path),
            "inspector_sha256": _sha256_file(inspector_path),
            "profiling_verbosity": "detailed",
            "plan_kind": "same_source_fp16",
            "target_call_sites": fp16_target_call_sites[kind],
            "target_call_site_count": EXPECTED_CALL_SITES,
            "inspector_catalog_sha256": catalog["catalog_sha256"],
            "rgb_shape": list(outputs["rgb"]),
            "io_tensors": io_contract,
        }

    manifest = {
        "schema_version": TRT_LAYER_PROFILE_MANIFEST_SCHEMA_VERSION,
        "scope": TRT_LAYER_PROFILE_SCOPE,
        "model_id": production_manifest["model_id"],
        "batch_size": 1,
        "height": production_manifest["height"],
        "width": production_manifest["width"],
        "latent_shape": production_manifest["latent_shape"],
        "latent_dtype": "float16",
        "production_manifest": {
            "file": manifest_path.name,
            "sha256": _sha256_file(manifest_path),
            "schema_version": production_manifest.get("schema_version"),
        },
        "production_manifest_sha256": _sha256_file(manifest_path),
        "cache": {
            "tensor_count": TRT_VAE_CACHE_COUNT,
            "dtype": "float16",
            "total_elements": TRT_VAE_CACHE_TOTAL_ELEMENTS,
            "single_bank_bytes": TRT_VAE_CACHE_BANK_BYTES,
            "double_bank_bytes": TRT_VAE_CACHE_BANK_BYTES * 2,
            "bindings": production_manifest["cache"]["bindings"],
        },
        "engines": {
            "fp16": {
                "plan_kind": "same_source_fp16",
                **fp16_records,
            },
            "int8": {
                "plan_kind": "audited_int8_v5",
                **{
                    kind: {
                        "file": _relative_file(root, int8_plans[kind]),
                        "sha256": _sha256_file(int8_plans[kind]),
                        "inspector_file": _relative_file(
                            root,
                            int8_inspector_paths[kind],
                        ),
                        "inspector_sha256": _sha256_file(int8_inspector_paths[kind]),
                        "profiling_verbosity": "detailed",
                        "rgb_shape": [
                            1,
                            3,
                            9 if kind == "initial" else 12,
                            production_manifest["height"],
                            production_manifest["width"],
                        ],
                        "io_tensors": int8_io[kind],
                    }
                    for kind in _KINDS
                },
            },
        },
        "int8_audit": {
            "passed": True,
            "schema_version": QDQ_SCHEMA_VERSION,
            "file": _relative_file(root, int8_audit_path),
            "sha256": _sha256_file(int8_audit_path),
            "plan_sha256": {kind: _sha256_file(int8_plans[kind]) for kind in _KINDS},
            "weight_encoding": int8_audit.get("weight_encoding"),
        },
        "build": {
            **identity,
            "state_file": state_path.name,
            "state_sha256": _sha256_file(state_path),
            "timing_cache_sha256": (
                _sha256_file(resolved_timing_cache)
                if resolved_timing_cache.is_file()
                else None
            ),
            "resume_enabled": resume,
        },
    }
    profile_manifest_path = root / _PROFILE_MANIFEST_FILE
    _write_json(profile_manifest_path, manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine-dir",
        required=True,
        help="existing completed TensorRT VAE engine directory",
    )
    parser.add_argument("--workspace-gib", type=float, default=8.0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed stages from trt_layer_profile_build_state.json",
    )
    parser.add_argument(
        "--timing-cache",
        help=(
            "separate layer-profile timing cache inside ENGINE_DIR "
            f"(default: {_PROFILE_TIMING_CACHE_FILE})"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_layer_profile_plans(
        engine_dir=args.engine_dir,
        workspace_gib=args.workspace_gib,
        device_index=args.device_index,
        resume=args.resume,
        timing_cache_path=args.timing_cache,
    )
    root = Path(args.engine_dir).expanduser().resolve()
    print(
        json.dumps(
            {
                "engine_dir": str(root),
                "manifest": str(root / _PROFILE_MANIFEST_FILE),
                "manifest_schema": manifest["schema_version"],
                "scope": manifest["scope"],
                "fp16_profile_plan_sha256": {
                    kind: manifest["engines"]["fp16"][kind]["sha256"] for kind in _KINDS
                },
                "int8_plan_sha256": manifest["int8_audit"]["plan_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
