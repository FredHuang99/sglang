"""Build isolated P1/P2/P3 Native INT8 V2 SFWan VAE plans on Orin."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

from .vae_trt_build import _build_engine_bytes, _load_plan, _write_bytes
from .vae_trt_fusion import sha256_file, write_json_atomic
from .vae_trt_fusion_build import _environment, _validate_io
from .vae_trt_native_int8 import (
    EXPECTED_CALL_SITES_PER_GRAPH,
    EXPECTED_RESIDUAL_BLOCKS,
    EXPECTED_SIGNATURES,
    NATIVE_INT8_ANALYSIS_FILE,
    NATIVE_INT8_CUTLASS_COMMIT,
    NATIVE_INT8_ONNX_FILES,
    NATIVE_INT8_PACKED_WEIGHTS_FILE,
    NATIVE_INT8_SCALES_FILE,
    NATIVE_INT8_SUBDIRECTORY,
    NATIVE_INT8_TUNE_FILE,
    NATIVE_INT8_WEIGHTS_MANIFEST_FILE,
    load_native_int8_manifest,
    validate_native_int8_manifest,
)
from .vae_trt_native_int8_build import (
    _expected_io,
    _map_native_plugins,
    _native_cache_io_format_constraints,
    _timing_cache_supported,
    _validate_cutlass_checkout,
    _validate_native_cache_io_formats,
)
from .vae_trt_native_int8_v2 import (
    NATIVE_INT8_V2_CUTLASS_COMMIT,
    NATIVE_INT8_V2_LEVEL_OFFSETS,
    NATIVE_INT8_V2_LEVELS,
    NATIVE_INT8_V2_PLUGIN_INIT_SYMBOL,
    NATIVE_INT8_V2_PLUGIN_LIBRARY_FILE,
    NATIVE_INT8_V2_PLUGIN_NAME,
    NATIVE_INT8_V2_PLUGIN_NAMESPACE,
    NATIVE_INT8_V2_PLUGIN_VERSION,
    NATIVE_INT8_V2_P1_ALGORITHM,
    NATIVE_INT8_V2_P1_KERNEL_REVISION,
    NATIVE_INT8_V2_P1_SCHEMA_VERSION,
    NATIVE_INT8_V2_P2_ALGORITHM,
    NATIVE_INT8_V2_P2_KERNEL_REVISION,
    NATIVE_INT8_V2_P3_ALGORITHM,
    NATIVE_INT8_V2_P3_KERNEL_REVISION,
    NATIVE_INT8_V2_SCHEMA_VERSION,
    NATIVE_INT8_V2_SERIALIZATION_ABI_REVISION,
    NATIVE_INT8_V2_SUBDIRECTORY,
    NATIVE_INT8_V2_VARIANT,
    load_native_int8_v2_manifest,
    native_int8_v2_level_root,
    rewrite_native_int8_v2_graph,
    validate_native_int8_v2_manifest,
)
from .vae_trt_runtime import load_trt_vae_manifest, validate_trt_vae_manifest

_KINDS = ("initial", "steady")
_LEVEL_FILES = {
    "initial_onnx": "initial.onnx",
    "steady_onnx": "steady.onnx",
    "initial_plan": "initial.plan",
    "steady_plan": "steady.plan",
    "initial_inspector": "initial_inspector.json",
    "steady_inspector": "steady_inspector.json",
    "manifest": "manifest.json",
    "audit": "audit.json",
    "timing_cache": "timing.cache",
}


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _identity(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _copy_verified(source: Path, destination: Path) -> None:
    if destination.is_file() and sha256_file(destination) == sha256_file(source):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _load_state(root: Path, identity: Mapping[str, Any], resume: bool) -> dict[str, Any]:
    path = root / "build_state_windowed_v3.json"
    fresh = {
        "schema_version": NATIVE_INT8_V2_SCHEMA_VERSION,
        "variant": NATIVE_INT8_V2_VARIANT,
        "identity": dict(identity),
        "stages": {},
    }
    if not resume or not path.is_file():
        return fresh
    state = _load_json(path, label="Native INT8 V2 build state")
    if (
        state.get("schema_version") != NATIVE_INT8_V2_SCHEMA_VERSION
        or state.get("variant") != NATIVE_INT8_V2_VARIANT
    ):
        raise ValueError("Native INT8 V2 --resume build-state schema changed")
    previous = state.get("identity")
    if not isinstance(previous, Mapping):
        raise ValueError("Native INT8 V2 build state has no identity")
    old = dict(previous)
    new = dict(identity)
    old_plugin = old.pop("plugin_sha256", None)
    new_plugin = new.pop("plugin_sha256", None)
    if old != new:
        raise ValueError("Native INT8 V2 --resume identity changed")
    if old_plugin != new_plugin:
        preserved = {
            key: value
            for key, value in state.get("stages", {}).items()
            if key == "analyze" and value.get("status") == "completed"
        }
        fresh["stages"] = preserved
        return fresh
    return state


def _save_state(root: Path, state: Mapping[str, Any]) -> None:
    write_json_atomic(root / "build_state_windowed_v3.json", state)


def _mark(
    root: Path, state: dict[str, Any], stage: str, status: str, **details: Any
) -> None:
    state.setdefault("stages", {})[stage] = {
        "status": status,
        "updated_unix_time_ns": time.time_ns(),
        **details,
    }
    _save_state(root, state)


def _completed(state: Mapping[str, Any], stage: str) -> bool:
    value = state.get("stages", {}).get(stage)
    return isinstance(value, Mapping) and value.get("status") == "completed"


def _load_plugin(*, trt: Any, path: Path) -> Any:
    try:
        library = ctypes.CDLL(str(path), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    except OSError as exc:
        raise RuntimeError(f"could not load Native INT8 V2 plugin: {path}") from exc
    initialize = getattr(library, NATIVE_INT8_V2_PLUGIN_INIT_SYMBOL, None)
    if initialize is None:
        raise RuntimeError(
            f"Native INT8 V2 plugin lacks {NATIVE_INT8_V2_PLUGIN_INIT_SYMBOL}"
        )
    initialize.argtypes = []
    initialize.restype = ctypes.c_bool
    if not initialize():
        raise RuntimeError("Native INT8 V2 plugin registration failed")
    registry = trt.get_plugin_registry()
    creator = None
    for getter_name in ("get_plugin_creator", "get_creator"):
        getter = getattr(registry, getter_name, None)
        if callable(getter):
            creator = getter(
                NATIVE_INT8_V2_PLUGIN_NAME,
                NATIVE_INT8_V2_PLUGIN_VERSION,
                NATIVE_INT8_V2_PLUGIN_NAMESPACE,
            )
        if creator is not None:
            break
    if creator is None:
        raise RuntimeError("Native INT8 V2 TensorRT creator was not registered")
    commit = getattr(library, "sfwanNativeInt8V2CutlassCommit", None)
    if commit is None:
        raise RuntimeError("Native INT8 V2 plugin does not expose CUTLASS commit")
    commit.argtypes = []
    commit.restype = ctypes.c_char_p
    actual = commit().decode("ascii")
    if actual != NATIVE_INT8_V2_CUTLASS_COMMIT:
        raise RuntimeError(
            f"Native INT8 V2 CUTLASS ABI mismatch: {actual} != "
            f"{NATIVE_INT8_V2_CUTLASS_COMMIT}"
        )
    return library


def _p1_kernel_contract(library: Any) -> dict[str, Any]:
    algorithm = getattr(library, "sfwanNativeInt8V2P1Algorithm", None)
    contract = getattr(library, "sfwanNativeInt8V2P1KernelContract", None)
    if algorithm is None or contract is None:
        raise RuntimeError("Native INT8 V2 plugin lacks the P1 kernel contract")
    algorithm.argtypes = []
    algorithm.restype = ctypes.c_char_p
    actual_algorithm = algorithm().decode("ascii")
    values = (ctypes.c_uint64 * 5)()
    contract.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int32]
    contract.restype = ctypes.c_int32
    if int(contract(values, len(values))) != 0:
        raise RuntimeError("Native INT8 V2 plugin rejected the P1 kernel audit")
    result = {
        "tensor_core_int8": bool(values[0]),
        "accumulator2_global_store": bool(values[1]),
        "separate_residual_kernel": bool(values[2]),
        "direct_conv_kernel_used_by_p1": bool(values[3]),
        "serialization_abi_revision": int(values[4]),
    }
    expected = {
        "tensor_core_int8": True,
        "accumulator2_global_store": False,
        "separate_residual_kernel": False,
        "direct_conv_kernel_used_by_p1": False,
        "serialization_abi_revision": NATIVE_INT8_V2_SERIALIZATION_ABI_REVISION,
    }
    if actual_algorithm != NATIVE_INT8_V2_P1_ALGORITHM or result != expected:
        raise RuntimeError(
            "Native INT8 V2 P1 kernel contract mismatch: "
            f"algorithm={actual_algorithm!r}, contract={result!r}"
        )
    return {"algorithm": actual_algorithm, **result}


def _level_kernel_contract(library: Any, *, level: str) -> dict[str, Any]:
    if level == "p1":
        return _p1_kernel_contract(library)
    specifications = {
        "p2": {
            "algorithm_symbol": "sfwanNativeInt8V2P2Algorithm",
            "contract_symbol": "sfwanNativeInt8V2P2KernelContract",
            "algorithm": NATIVE_INT8_V2_P2_ALGORITHM,
            "revision": NATIVE_INT8_V2_P2_KERNEL_REVISION,
            "keys": (
                "tensor_core_int8",
                "direct_causal_iterator",
                "legacy_direct_causal_used",
                "temporal_window_materialized",
                "entry_value_global_loads",
                "mid_accumulator_global_loads",
                "accumulator1_global_store",
                "accumulator2_global_store",
                "conv2_fused_residual_epilogue",
            ),
            "expected": (True, False, False, True, 1, 1, True, False, True),
        },
        "p3": {
            "algorithm_symbol": "sfwanNativeInt8V2P3Algorithm",
            "contract_symbol": "sfwanNativeInt8V2P3KernelContract",
            "algorithm": NATIVE_INT8_V2_P3_ALGORITHM,
            "revision": NATIVE_INT8_V2_P3_KERNEL_REVISION,
            "keys": (
                "tensor_core_int8",
                "direct_causal_iterator",
                "legacy_persistent_wmma",
                "temporal_window_materialized",
                "entry_value_global_loads",
                "mid_value_global_loads",
                "accumulator1_global_store",
                "accumulator2_global_store",
                "conv1_output_dtype_fp16",
                "conv1_fused_dequant_bias",
                "conv2_fused_residual_epilogue",
                "persistent_block_count",
            ),
            "expected": (
                True, False, False, True, 1, 1, False, False, True, True,
                True, 0,
            ),
        },
    }[level]
    algorithm = getattr(library, specifications["algorithm_symbol"], None)
    contract = getattr(library, specifications["contract_symbol"], None)
    if algorithm is None or contract is None:
        raise RuntimeError(f"Native INT8 V2 plugin lacks the {level} CUTLASS contract")
    algorithm.argtypes = []
    algorithm.restype = ctypes.c_char_p
    actual_algorithm = algorithm().decode("ascii")
    values = (ctypes.c_uint64 * len(specifications["keys"]))()
    contract.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int32]
    contract.restype = ctypes.c_int32
    if int(contract(values, len(values))) != 0:
        raise RuntimeError(f"Native INT8 V2 plugin rejected the {level} contract")
    converted: list[Any] = []
    for key, value in zip(specifications["keys"], values):
        converted.append(
            int(value)
            if key.endswith(("_bytes", "_loads", "_count"))
            else bool(value)
        )
    result = dict(zip(specifications["keys"], converted))
    expected = dict(zip(specifications["keys"], specifications["expected"]))
    if actual_algorithm != specifications["algorithm"] or result != expected:
        raise RuntimeError(
            f"Native INT8 V2 {level} kernel contract mismatch: "
            f"algorithm={actual_algorithm!r}, contract={result!r}"
        )
    return {
        "algorithm": actual_algorithm,
        "kernel_revision": specifications["revision"],
        **result,
    }


class _BlockConfig(ctypes.Structure):
    _fields_ = [
        ("inputShape", ctypes.c_int32 * 5),
        ("shortcutShape", ctypes.c_int32 * 5),
        ("outputShape", ctypes.c_int32 * 5),
        ("cache1InputShape", ctypes.c_int32 * 5),
        ("cache2InputShape", ctypes.c_int32 * 5),
        ("cache1OutputShape", ctypes.c_int32 * 5),
        ("cache2OutputShape", ctypes.c_int32 * 5),
        ("weight1Shape", ctypes.c_int32 * 4),
        ("weight2Shape", ctypes.c_int32 * 4),
        ("conv1Params", ctypes.c_int32 * 6),
        ("conv2Params", ctypes.c_int32 * 6),
        ("inputIsInt8", ctypes.c_int32),
        ("shortcutIsInt8", ctypes.c_int32),
        ("outputIsInt8", ctypes.c_int32),
        ("hasCache1", ctypes.c_int32),
        ("hasCache2", ctypes.c_int32),
        ("tile1", ctypes.c_int32),
        ("tile2", ctypes.c_int32),
        ("profileId", ctypes.c_int32),
        ("inputScale", ctypes.c_float),
        ("conv1InputScale", ctypes.c_float),
        ("conv2InputScale", ctypes.c_float),
        ("outputScale", ctypes.c_float),
    ]


def _attribute_map(node: Any, helper: Any) -> dict[str, Any]:
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def _block_config(attrs: Mapping[str, Any]) -> _BlockConfig:
    config = _BlockConfig()
    arrays = {
        "inputShape": ("input_shape", 5),
        "shortcutShape": ("shortcut_shape", 5),
        "outputShape": ("output_shape", 5),
        "cache1InputShape": ("cache1_input_shape", 5),
        "cache2InputShape": ("cache2_input_shape", 5),
        "cache1OutputShape": ("cache1_output_shape", 5),
        "cache2OutputShape": ("cache2_output_shape", 5),
        "weight1Shape": ("weight1_shape", 4),
        "weight2Shape": ("weight2_shape", 4),
        "conv1Params": ("conv1_params", 6),
        "conv2Params": ("conv2_params", 6),
    }
    for field, (attribute, count) in arrays.items():
        values = [int(value) for value in attrs[attribute]]
        if len(values) != count:
            raise ValueError(f"Native INT8 V2 plugin attribute {attribute} is invalid")
        getattr(config, field)[:] = values
    integers = {
        "inputIsInt8": "input_is_int8",
        "shortcutIsInt8": "shortcut_is_int8",
        "outputIsInt8": "output_is_int8",
        "hasCache1": "has_cache1",
        "hasCache2": "has_cache2",
        "tile1": "tile1",
        "tile2": "tile2",
        "profileId": "profile_id",
    }
    for field, attribute in integers.items():
        setattr(config, field, int(attrs[attribute]))
    floats = {
        "inputScale": "input_scale",
        "conv1InputScale": "conv1_input_scale",
        "conv2InputScale": "conv2_input_scale",
        "outputScale": "output_scale",
    }
    for field, attribute in floats.items():
        setattr(config, field, float(attrs[attribute]))
    return config


def _workspace_audit(*, library: Any, graph_paths: Mapping[str, Path], level: str) -> dict[str, Any]:
    try:
        import onnx
        from onnx import helper
    except ImportError as exc:
        raise RuntimeError("Native INT8 V2 workspace audit requires ONNX") from exc
    workspace = library.sfwanNativeInt8V2WorkspaceContractV2
    workspace.argtypes = [
        ctypes.POINTER(_BlockConfig),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int32,
    ]
    workspace.restype = ctypes.c_int32
    records: list[dict[str, Any]] = []
    maxima = [0, 0, 0, 0, 0]
    for kind, path in graph_paths.items():
        model = onnx.load(str(path), load_external_data=False)
        for node in model.graph.node:
            if node.op_type != NATIVE_INT8_V2_PLUGIN_NAME:
                continue
            config = _block_config(_attribute_map(node, helper))
            values = (ctypes.c_uint64 * 5)()
            if int(workspace(ctypes.byref(config), values, 5)) != 0:
                raise RuntimeError(f"workspace audit failed for {node.name}")
            current = [int(value) for value in values]
            maxima = [max(old, value) for old, value in zip(maxima, current)]
            record: dict[str, Any] = {
                "kind": kind,
                "name": node.name,
                "accumulator1_workspace_bytes": current[0],
                "accumulator2_workspace_bytes": current[1],
                "temporal_window_bytes": current[2],
                "conv1_mid_workspace_bytes": current[3],
                "total_workspace_bytes": current[4],
            }
            records.append(record)
    return {
        "plugin_call_count": len(records),
        "accumulator1_workspace_bytes": maxima[0],
        "accumulator2_workspace_bytes": maxima[1],
        "temporal_window_bytes": maxima[2],
        "conv1_mid_workspace_bytes": maxima[3],
        "max_plugin_workspace_bytes": maxima[4],
        "persistent_block_count": 0,
        "records": records,
    }


def _rebind_existing_level(
    *,
    level: str,
    level_root: Path,
    base_root: Path,
    base_manifest: Mapping[str, Any],
    v1_manifest: Mapping[str, Any],
    plugin_manifest: Mapping[str, Any],
    library: Any,
) -> dict[str, Any]:
    """Atomically bind an unchanged P1/P2 plan to a compatible corrected DSO."""

    manifest_path = level_root / _LEVEL_FILES["manifest"]
    old = _load_json(
        manifest_path, label=f"existing Native INT8 V2 {level} manifest"
    )
    old_plugin = old.get("plugin")
    engines = old.get("engines")
    graph = old.get("graph")
    audit_record = old.get("audit")
    timing_cache = old.get("timing_cache")
    if not all(
        isinstance(value, Mapping)
        for value in (old_plugin, engines, graph, audit_record, timing_cache)
    ):
        raise RuntimeError(
            f"{level} compatible rebind rejected an incomplete old manifest"
        )
    if (
        old.get("schema_version")
        != (
            NATIVE_INT8_V2_P1_SCHEMA_VERSION
            if level == "p1"
            else NATIVE_INT8_V2_SCHEMA_VERSION
        )
        or old.get("variant") != NATIVE_INT8_V2_VARIANT
        or old.get("level") != level
        or old.get("base_native_int8_v1_identity") != _identity(v1_manifest)
    ):
        raise RuntimeError(
            f"{level} compatible rebind identity changed; rebuild is required"
        )
    abi_keys = (
        "file",
        "plugin_name",
        "plugin_version",
        "plugin_namespace",
        "init_symbol",
        "cutlass_commit",
    )
    if any(old_plugin.get(key) != plugin_manifest.get(key) for key in abi_keys):
        raise RuntimeError(
            f"{level} plugin creator/ABI changed; rebuild is required"
        )
    if old_plugin.get("sha256") == plugin_manifest.get("sha256"):
        raise RuntimeError(
            f"{level} manifest already references this DSO but lacks the corrected "
            "kernel contract; install the rebuilt plugin or perform a full rebuild"
        )

    old_audit_path = level_root / str(audit_record.get("file"))
    if (
        not old_audit_path.is_file()
        or sha256_file(old_audit_path) != audit_record.get("sha256")
    ):
        raise RuntimeError(f"{level} old audit SHA changed; rebuild is required")
    old_audit = _load_json(
        old_audit_path, label=f"existing Native INT8 V2 {level} audit"
    )
    if old_audit.get("passed") is not True or old_audit.get("errors") != []:
        raise RuntimeError(f"{level} old audit did not pass; rebuild is required")

    graph_paths: dict[str, Path] = {}
    inspector_mappings: dict[str, Any] = {}
    for kind in _KINDS:
        engine = engines.get(kind)
        graph_record = graph.get(kind)
        if not isinstance(engine, Mapping) or not isinstance(graph_record, Mapping):
            raise RuntimeError(f"{level} {kind} artifact record is incomplete")
        plan_path = level_root / str(engine.get("file"))
        onnx_path = level_root / str(engine.get("source_onnx_file"))
        inspector_path = level_root / str(engine.get("inspector_file"))
        checks = (
            (plan_path, engine.get("sha256"), "plan"),
            (onnx_path, engine.get("source_onnx_sha256"), "ONNX"),
            (inspector_path, engine.get("inspector_sha256"), "Inspector"),
        )
        for path, expected_sha, label in checks:
            if not path.is_file() or sha256_file(path) != expected_sha:
                raise RuntimeError(
                    f"{level} {kind} {label} changed; compatible rebind is unsafe"
                )
        if old_audit.get("plan_sha256", {}).get(kind) != engine.get("sha256"):
            raise RuntimeError(f"{level} {kind} audit/plan identity changed")
        if graph_record.get("destination_sha256") != sha256_file(onnx_path):
            raise RuntimeError(f"{level} {kind} serialized plugin graph changed")
        expected_names = [
            value["v2_name"] for value in graph_record.get("plugins", [])
        ]
        if len(expected_names) != EXPECTED_RESIDUAL_BLOCKS * 3:
            raise RuntimeError(f"{level} {kind} serialized plugin count changed")
        inspector = json.loads(inspector_path.read_text(encoding="utf-8"))
        inspector_mappings[kind] = _map_native_plugins(
            inspector, expected_names=expected_names
        )
        if inspector_mappings[kind]["passed"] is not True:
            raise RuntimeError(f"{level} {kind} Inspector mapping changed")
        graph_paths[kind] = onnx_path

    cache_path = level_root / str(timing_cache.get("file"))
    if not cache_path.is_file() or sha256_file(cache_path) != timing_cache.get(
        "sha256"
    ):
        raise RuntimeError(
            f"{level} timing cache changed; compatible rebind is unsafe"
        )
    if old.get("cache") != v1_manifest.get("cache"):
        raise RuntimeError(f"{level} mixed-cache ABI changed; rebuild is required")

    workspace = _workspace_audit(
        library=library, graph_paths=graph_paths, level=level
    )
    expected_plugins = EXPECTED_RESIDUAL_BLOCKS * 3 * 2
    if (
        workspace["plugin_call_count"] != expected_plugins
        or workspace["accumulator2_workspace_bytes"] != 0
    ):
        raise RuntimeError(f"{level} corrected DSO workspace contract did not pass")
    old_workspace = int(old_audit.get("max_plugin_workspace_bytes", 0))
    if old_workspace <= 0 or workspace["max_plugin_workspace_bytes"] > old_workspace:
        raise RuntimeError(
            f"{level} corrected DSO requires a larger plugin workspace; "
            "compatible plan rebind is unsafe"
        )
    kernel_contract = _level_kernel_contract(library, level=level)
    algorithm = str(kernel_contract.pop("algorithm"))
    kernel_revision = str(
        kernel_contract.pop(
            "kernel_revision",
            NATIVE_INT8_V2_P1_KERNEL_REVISION,
        )
    )
    plan_sha = {kind: str(engines[kind]["sha256"]) for kind in _KINDS}
    audit = {
        **old_audit,
        "passed": True,
        "complete": True,
        "errors": [],
        "accumulator1_workspace_bytes": workspace[
            "accumulator1_workspace_bytes"
        ],
        "accumulator2_workspace_bytes": workspace[
            "accumulator2_workspace_bytes"
        ],
        "temporal_window_bytes": workspace["temporal_window_bytes"],
        "conv1_mid_workspace_bytes": workspace["conv1_mid_workspace_bytes"],
        "max_plugin_workspace_bytes": workspace["max_plugin_workspace_bytes"],
        "residual_epilogue_kernel_present": False,
        "native_int8_v2_algorithm": algorithm,
        "native_int8_v2_kernel_revision": kernel_revision,
        "native_int8_v2_legacy_wmma_used": False,
        "native_int8_v2_kernel_contract": kernel_contract,
        "persistent_block_count": 0,
        "p1_algorithm": algorithm if level == "p1" else None,
        "p1_kernel_revision": kernel_revision if level == "p1" else None,
        "p1_kernel_contract": kernel_contract if level == "p1" else None,
        "plan_sha256": plan_sha,
        "inspector_plugin_mappings": inspector_mappings,
        "workspace_records": workspace["records"],
        "plan_reused": True,
        "plugin_rebound": True,
    }
    write_json_atomic(old_audit_path, audit)
    rebound = {
        **old,
        "plugin": dict(plugin_manifest),
        "audit": {
            "file": old_audit_path.name,
            "sha256": sha256_file(old_audit_path),
            "passed": True,
        },
        "plan_reused": True,
        "plugin_rebound": True,
        "native_int8_v2_algorithm": algorithm,
        "native_int8_v2_kernel_revision": kernel_revision,
        "native_int8_v2_legacy_wmma_used": False,
        "p1_kernel_revision": kernel_revision if level == "p1" else None,
    }
    write_json_atomic(manifest_path, rebound)
    validate_native_int8_v2_manifest(
        rebound,
        engine_dir=base_root,
        base_manifest=base_manifest,
        level=level,
        verify_hashes=True,
    )
    return rebound


def _archive_legacy_p3(level_root: Path) -> None:
    """Preserve the obsolete single-warp P3 artifacts before rebuilding."""

    names = (
        _LEVEL_FILES["manifest"],
        _LEVEL_FILES["audit"],
        _LEVEL_FILES["initial_plan"],
        _LEVEL_FILES["steady_plan"],
        _LEVEL_FILES["initial_inspector"],
        _LEVEL_FILES["steady_inspector"],
        _LEVEL_FILES["timing_cache"],
    )
    for name in names:
        source = level_root / name
        if not source.exists():
            continue
        destination = source.with_name(f"{source.name}.legacy_wmma")
        if destination.exists():
            destination = source.with_name(
                f"{source.name}.legacy_wmma.{time.time_ns()}"
            )
        source.replace(destination)


def _build_level(
    *,
    level: str,
    root: Path,
    base_root: Path,
    base_manifest: Mapping[str, Any],
    v1_manifest: Mapping[str, Any],
    v1_validated: Mapping[str, Any],
    plugin_manifest: Mapping[str, Any],
    library: Any,
    workspace_gib: float,
    trt: Any,
    resume: bool,
) -> dict[str, Any]:
    level_root = native_int8_v2_level_root(base_root, level=level)
    level_root.mkdir(parents=True, exist_ok=True)
    existing_manifest = level_root / _LEVEL_FILES["manifest"]
    if level == "p3" and existing_manifest.is_file() and not resume:
        _archive_legacy_p3(level_root)
    if resume and existing_manifest.is_file():
        manifest = load_native_int8_v2_manifest(base_root, level=level)
        try:
            validate_native_int8_v2_manifest(
                manifest,
                engine_dir=base_root,
                base_manifest=base_manifest,
                level=level,
                verify_hashes=True,
            )
        except ValueError:
            if level == "p1":
                return _rebind_existing_level(
                    level=level,
                    level_root=level_root,
                    base_root=base_root,
                    base_manifest=base_manifest,
                    v1_manifest=v1_manifest,
                    plugin_manifest=plugin_manifest,
                    library=library,
                )
            _archive_legacy_p3(level_root)
        else:
            return manifest

    v1_root = base_root / NATIVE_INT8_SUBDIRECTORY
    graph_records: dict[str, Any] = {}
    graph_paths: dict[str, Path] = {}
    for kind in _KINDS:
        source = v1_root / NATIVE_INT8_ONNX_FILES[kind]
        destination = level_root / _LEVEL_FILES[f"{kind}_onnx"]
        graph_records[kind] = rewrite_native_int8_v2_graph(
            source_path=source, destination_path=destination, level=level
        )
        graph_paths[kind] = destination

    int8_slots = {
        int(value["index"])
        for value in v1_validated["cache_bindings"]
        if value["dtype"] == "int8"
    }
    stable_cache = level_root / _LEVEL_FILES["timing_cache"]
    candidate_cache = level_root / "timing.cache.candidate"
    candidate_cache.unlink(missing_ok=True)
    use_timing_cache = _timing_cache_supported(trt)
    plan_records: dict[str, Any] = {}
    for kind in _KINDS:
        plan, io_contract, inspector_json, cache_bytes = _build_engine_bytes(
            trt=trt,
            onnx_path=graph_paths[kind],
            workspace_gib=workspace_gib,
            profiling_verbosity="detailed",
            timing_cache_path=candidate_cache if use_timing_cache else None,
            io_tensor_formats=_native_cache_io_format_constraints(
                trt=trt, kind=kind, int8_slots=int8_slots
            ),
        )
        _validate_native_cache_io_formats(
            records=io_contract, kind=kind, int8_slots=int8_slots
        )
        if use_timing_cache and cache_bytes is not None:
            _write_bytes(candidate_cache, cache_bytes)
        plan_path = level_root / _LEVEL_FILES[f"{kind}_plan"]
        inspector_path = level_root / _LEVEL_FILES[f"{kind}_inspector"]
        _write_bytes(plan_path, plan)
        inspector = json.loads(inspector_json)
        write_json_atomic(inspector_path, inspector)
        plan_records[kind] = {
            "file": plan_path.name,
            "sha256": hashlib.sha256(plan).hexdigest(),
            "source_onnx_file": graph_paths[kind].name,
            "source_onnx_sha256": sha256_file(graph_paths[kind]),
            "inspector_file": inspector_path.name,
            "inspector_sha256": sha256_file(inspector_path),
            "profiling_verbosity": "detailed",
            "io_tensors": io_contract,
        }

    workspace = _workspace_audit(
        library=library, graph_paths=graph_paths, level=level
    )
    errors: list[str] = []
    expected_plugins = EXPECTED_RESIDUAL_BLOCKS * 3 * 2
    if workspace["plugin_call_count"] != expected_plugins:
        errors.append("V2 did not audit all 84 residual-block calls")
    if workspace["accumulator2_workspace_bytes"] != 0:
        errors.append("Conv2 accumulator workspace is nonzero")
    if level in {"p2", "p3"} and workspace["temporal_window_bytes"] <= 0:
        errors.append("temporal-window workspace is missing")
    if level == "p3" and workspace["accumulator1_workspace_bytes"] != 0:
        errors.append("Conv1 accumulator workspace is nonzero")
    if level == "p2" and workspace["accumulator1_workspace_bytes"] <= 0:
        errors.append("P2 Conv1 accumulator workspace is missing")
    if level == "p3" and workspace["conv1_mid_workspace_bytes"] <= 0:
        errors.append("P3 FP16 Conv1-mid workspace is missing")
    if workspace["persistent_block_count"] != 0:
        errors.append("obsolete persistent WMMA block is still active")
    level_kernel_contract = _level_kernel_contract(library, level=level)
    algorithm = str(level_kernel_contract["algorithm"])
    kernel_revision = str(
        level_kernel_contract.get(
            "kernel_revision", NATIVE_INT8_V2_P1_KERNEL_REVISION
        )
    )

    inspector_mappings: dict[str, Any] = {}
    for kind in _KINDS:
        _validate_io(
            records=plan_records[kind]["io_tensors"],
            expected=_expected_io(
                base_manifest=base_manifest, kind=kind, int8_slots=int8_slots
            ),
        )
        expected_names = [
            value["v2_name"] for value in graph_records[kind]["plugins"]
        ]
        inspector = json.loads(
            (level_root / _LEVEL_FILES[f"{kind}_inspector"]).read_text(
                encoding="utf-8"
            )
        )
        inspector_mappings[kind] = _map_native_plugins(
            inspector, expected_names=expected_names
        )
        if inspector_mappings[kind]["passed"] is not True:
            errors.append(f"{kind} TensorRT inspector plugin mapping is incomplete")

    audit = {
        "schema_version": NATIVE_INT8_V2_SCHEMA_VERSION,
        "variant": NATIVE_INT8_V2_VARIANT,
        "level": level,
        "passed": not errors,
        "complete": not errors,
        "errors": errors,
        "initial_call_site_count": EXPECTED_CALL_SITES_PER_GRAPH,
        "steady_call_site_count": EXPECTED_CALL_SITES_PER_GRAPH,
        "signature_count": EXPECTED_SIGNATURES,
        "accumulator1_workspace_bytes": workspace[
            "accumulator1_workspace_bytes"
        ],
        "accumulator2_workspace_bytes": workspace[
            "accumulator2_workspace_bytes"
        ],
        "temporal_window_bytes": workspace["temporal_window_bytes"],
        "conv1_mid_workspace_bytes": workspace["conv1_mid_workspace_bytes"],
        "max_plugin_workspace_bytes": workspace["max_plugin_workspace_bytes"],
        "direct_causal_iterator": False,
        "persistent_block_count": workspace["persistent_block_count"],
        "residual_epilogue_kernel_present": False,
        "native_int8_v2_algorithm": algorithm,
        "native_int8_v2_kernel_revision": kernel_revision,
        "native_int8_v2_legacy_wmma_used": False,
        "native_int8_v2_kernel_contract": {
            key: value
            for key, value in level_kernel_contract.items()
            if key not in {"algorithm", "kernel_revision"}
        },
        "p1_algorithm": algorithm if level == "p1" else None,
        "p1_kernel_revision": (
            kernel_revision if level == "p1" else None
        ),
        "p1_kernel_contract": (
            {
                key: value
                for key, value in level_kernel_contract.items()
                if key not in {"algorithm", "kernel_revision"}
            }
            if level == "p1"
            else None
        ),
        "plan_reused": False,
        "plugin_rebound": False,
        "plan_sha256": {
            kind: plan_records[kind]["sha256"] for kind in _KINDS
        },
        "inspector_plugin_mappings": inspector_mappings,
        "workspace_records": workspace["records"],
    }
    audit_path = level_root / _LEVEL_FILES["audit"]
    write_json_atomic(audit_path, audit)
    if errors:
        raise RuntimeError(f"Native INT8 V2 {level} audit failed: {errors}")
    if use_timing_cache and candidate_cache.is_file():
        candidate_cache.replace(stable_cache)
    elif not stable_cache.is_file():
        _write_bytes(stable_cache, b"")

    profile_ids = {
        kind: [
            {
                **dict(record),
                "plugin_name": str(record["plugin_name"]).replace(
                    "native_int8/", f"native_int8_v2/{level}/", 1
                ),
            }
            for record in v1_validated["profile_ids"][kind]
        ]
        for kind in _KINDS
    }
    manifest = {
        "schema_version": (
            NATIVE_INT8_V2_P1_SCHEMA_VERSION
            if level == "p1"
            else NATIVE_INT8_V2_SCHEMA_VERSION
        ),
        "variant": NATIVE_INT8_V2_VARIANT,
        "level": level,
        "base_native_int8_v1_identity": _identity(v1_manifest),
        "plugin": dict(plugin_manifest),
        "engines": plan_records,
        "graph": graph_records,
        "cache": dict(v1_manifest["cache"]),
        "weights": {
            **dict(v1_manifest["weights"]),
            "file": (
                "../packed_weights.safetensors"
                if level == "p1"
                else "../../packed_weights.safetensors"
            ),
        },
        "scales": {
            **dict(v1_manifest["scales"]),
            "file": "../scales.json" if level == "p1" else "../../scales.json",
        },
        "kernel_catalog": {
            "file": (
                "../kernel_catalog.json"
                if level == "p1"
                else "../../kernel_catalog.json"
            ),
            "level_offset": NATIVE_INT8_V2_LEVEL_OFFSETS[level],
            "kernel_tile_map": {
                signature: int(tile) + NATIVE_INT8_V2_LEVEL_OFFSETS[level]
                for signature, tile in v1_manifest["tune"][
                    "kernel_tile_map"
                ].items()
            },
        },
        "kernel_profile_ids": profile_ids,
        "audit": {
            "file": audit_path.name,
            "sha256": sha256_file(audit_path),
            "passed": True,
        },
        "timing_cache": {
            "file": stable_cache.name,
            "sha256": sha256_file(stable_cache),
            "api_supported": use_timing_cache,
        },
        "build": dict(v1_manifest["build"]),
        "plan_reused": False,
        "plugin_rebound": False,
        "native_int8_v2_algorithm": algorithm,
        "native_int8_v2_kernel_revision": kernel_revision,
        "native_int8_v2_legacy_wmma_used": False,
        "p1_kernel_revision": (
            kernel_revision if level == "p1" else None
        ),
    }
    write_json_atomic(existing_manifest, manifest)
    validate_native_int8_v2_manifest(
        manifest,
        engine_dir=base_root,
        base_manifest=base_manifest,
        level=level,
        verify_hashes=True,
    )
    return manifest


def build_native_int8_v2(
    *,
    engine_dir: str | Path,
    stage: str,
    level: str,
    cutlass_dir: str | Path,
    workspace_gib: float,
    tune_warmup: int,
    tune_repeat: int,
    resume: bool,
    device_index: int,
    plugin_library: str | Path | None,
) -> dict[str, Any]:
    if stage not in {"analyze", "p1", "p2", "p3", "audit", "all"}:
        raise ValueError(f"unsupported Native INT8 V2 stage: {stage}")
    if level not in NATIVE_INT8_V2_LEVELS:
        raise ValueError(f"unsupported Native INT8 V2 level: {level}")
    if workspace_gib <= 0 or tune_warmup < 0 or tune_repeat <= 0:
        raise ValueError("workspace/tune parameters are invalid")
    try:
        import onnx
        import tensorrt as trt
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Native INT8 V2 build requires ONNX, TensorRT, and CUDA torch"
        ) from exc

    base_root = Path(engine_dir).expanduser().resolve()
    root = base_root / NATIVE_INT8_V2_SUBDIRECTORY
    root.mkdir(parents=True, exist_ok=True)
    base_manifest = load_trt_vae_manifest(base_root)
    validate_trt_vae_manifest(
        base_manifest,
        engine_dir=base_root,
        precision="int8",
        model_path=None,
        verify_plan_hashes=True,
    )
    v1_manifest = load_native_int8_manifest(base_root)
    v1_validated = validate_native_int8_manifest(
        v1_manifest,
        engine_dir=base_root,
        base_manifest=base_manifest,
        verify_hashes=True,
    )
    _cutlass, commit = _validate_cutlass_checkout(cutlass_dir)
    if commit != NATIVE_INT8_CUTLASS_COMMIT:
        raise ValueError("Native INT8 V2 CUTLASS checkout differs from V1")

    plugin_path = root / NATIVE_INT8_V2_PLUGIN_LIBRARY_FILE
    if plugin_library is not None:
        source = Path(plugin_library).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"Native INT8 V2 plugin does not exist: {source}")
        _copy_verified(source, plugin_path)
    plugin_sha = sha256_file(plugin_path) if plugin_path.is_file() else None
    identity = {
        "schema_version": NATIVE_INT8_V2_SCHEMA_VERSION,
        "base_manifest": _identity(base_manifest),
        "native_int8_v1_manifest": _identity(v1_manifest),
        "plugin_sha256": plugin_sha,
        "cutlass_commit": commit,
        "workspace_gib": float(workspace_gib),
        "tune_warmup": tune_warmup,
        "tune_repeat": tune_repeat,
    }
    state = _load_state(root, identity, resume)
    _save_state(root, state)

    v1_root = base_root / NATIVE_INT8_SUBDIRECTORY
    provenance = {
        "schema_version": NATIVE_INT8_V2_SCHEMA_VERSION,
        "variant": NATIVE_INT8_V2_VARIANT,
        "base_native_int8_v1_identity": _identity(v1_manifest),
        "source_sha256": {},
        "levels": {
            "p1": {
                "conv2_fused_residual_epilogue": True,
                "direct_causal_iterator": False,
                "persistent_two_conv_block": False,
            },
            "p2": {
                "conv2_fused_residual_epilogue": True,
                "direct_causal_iterator": False,
                "register_reuse_producer": True,
                "temporal_window_materialized": True,
                "persistent_two_conv_block": False,
            },
            "p3": {
                "conv2_fused_residual_epilogue": True,
                "direct_causal_iterator": False,
                "register_reuse_producer": True,
                "temporal_window_materialized": True,
                "persistent_two_conv_block": False,
                "conv1_fused_fp16_epilogue": True,
            },
        },
    }
    copies = {
        NATIVE_INT8_ANALYSIS_FILE: root / "analysis.json",
        NATIVE_INT8_SCALES_FILE: root / "scales.json",
        NATIVE_INT8_PACKED_WEIGHTS_FILE: root / "packed_weights.safetensors",
        NATIVE_INT8_WEIGHTS_MANIFEST_FILE: root / "weights_manifest.json",
        NATIVE_INT8_TUNE_FILE: root / "kernel_catalog.json",
    }
    for source_name, destination in copies.items():
        source = v1_root / source_name
        _copy_verified(source, destination)
        provenance["source_sha256"][destination.name] = sha256_file(destination)
    write_json_atomic(root / "analysis.json", {
        **_load_json(root / "analysis.json", label="V1 analysis"),
        "native_int8_v2": provenance,
    })
    if not _completed(state, "analyze"):
        _mark(
            root,
            state,
            "analyze",
            "completed",
            analysis_sha256=sha256_file(root / "analysis.json"),
        )
    if stage == "analyze":
        return _load_json(root / "analysis.json", label="Native INT8 V2 analysis")

    environment = _environment(
        trt=trt, torch=torch, onnx=onnx, device_index=device_index
    )
    if environment.get("compute_capability") != [8, 7]:
        raise RuntimeError("Native INT8 V2 requires Jetson Orin SM87")
    if not plugin_path.is_file():
        raise ValueError(
            f"install {NATIVE_INT8_V2_PLUGIN_LIBRARY_FILE} into {root} first"
        )
    library = _load_plugin(trt=trt, path=plugin_path)
    plugin_manifest = {
        "schema_version": NATIVE_INT8_V2_SCHEMA_VERSION,
        "variant": NATIVE_INT8_V2_VARIANT,
        "file": NATIVE_INT8_V2_PLUGIN_LIBRARY_FILE,
        "sha256": sha256_file(plugin_path),
        "plugin_name": NATIVE_INT8_V2_PLUGIN_NAME,
        "plugin_version": NATIVE_INT8_V2_PLUGIN_VERSION,
        "plugin_namespace": NATIVE_INT8_V2_PLUGIN_NAMESPACE,
        "init_symbol": NATIVE_INT8_V2_PLUGIN_INIT_SYMBOL,
        "cutlass_commit": commit,
        "build": environment,
    }

    if stage == "audit":
        manifest = load_native_int8_v2_manifest(base_root, level=level)
        validate_native_int8_v2_manifest(
            manifest,
            engine_dir=base_root,
            base_manifest=base_manifest,
            level=level,
            verify_hashes=True,
        )
        return manifest
    target = level if stage == "all" else stage
    target_index = NATIVE_INT8_V2_LEVELS.index(target)
    levels = NATIVE_INT8_V2_LEVELS[: target_index + 1] if stage == "all" else (target,)
    result: dict[str, Any] = {}
    for current in levels:
        current_index = NATIVE_INT8_V2_LEVELS.index(current)
        if current_index > 0:
            previous = NATIVE_INT8_V2_LEVELS[current_index - 1]
            previous_manifest = load_native_int8_v2_manifest(
                base_root, level=previous
            )
            validate_native_int8_v2_manifest(
                previous_manifest,
                engine_dir=base_root,
                base_manifest=base_manifest,
                level=previous,
                verify_hashes=True,
            )
        _mark(root, state, current, "running")
        try:
            result = _build_level(
                level=current,
                root=root,
                base_root=base_root,
                base_manifest=base_manifest,
                v1_manifest=v1_manifest,
                v1_validated=v1_validated,
                plugin_manifest=plugin_manifest,
                library=library,
                workspace_gib=workspace_gib,
                trt=trt,
                resume=resume,
            )
            _mark(
                root,
                state,
                current,
                "completed",
                manifest_sha256=sha256_file(
                    native_int8_v2_level_root(base_root, level=current)
                    / "manifest.json"
                ),
            )
        except BaseException as exc:
            _mark(root, state, current, "failed", error=f"{type(exc).__name__}: {exc}")
            raise
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SFWan VAE Native INT8 V2 P1/P2/P3 engines"
    )
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument(
        "--stage", choices=("analyze", "p1", "p2", "p3", "audit", "all"), required=True
    )
    parser.add_argument("--level", choices=NATIVE_INT8_V2_LEVELS, default="p3")
    parser.add_argument("--cutlass-dir", required=True)
    parser.add_argument("--workspace-gib", type=float, default=8.0)
    parser.add_argument("--tune-warmup", type=int, default=0)
    parser.add_argument("--tune-repeat", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--plugin-library")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_native_int8_v2(
        engine_dir=args.engine_dir,
        stage=args.stage,
        level=args.level,
        cutlass_dir=args.cutlass_dir,
        workspace_gib=args.workspace_gib,
        tune_warmup=args.tune_warmup,
        tune_repeat=args.tune_repeat,
        resume=args.resume,
        device_index=args.device_index,
        plugin_library=args.plugin_library,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
