"""Build the isolated SM87 native-INT8 SFWan VAE TensorRT engines.

This command deliberately reuses only SHA-validated Q/DQ-v5 source graphs and
calibration evidence.  It never overwrites baseline, fusion-v1, or fusion-v2
artifacts.  Target residual blocks are replaced by a native IPluginV3 whose
CUTLASS implicit-GEMM convolutions consume offline-packed signed-INT8 weights.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from .vae_trt_build import _build_engine_bytes, _load_plan, _write_bytes
from .vae_trt_fusion import sha256_file, write_json_atomic
from .vae_trt_fusion_build import (
    _environment,
    _load_analysis_contracts,
    _run_analysis,
    _target_names,
    _validate_io,
)
from .vae_trt_native_int8 import (
    EXPECTED_CALL_SITES_PER_GRAPH,
    EXPECTED_FP16_CACHE_SLOTS,
    EXPECTED_INT8_CACHE_SLOTS,
    EXPECTED_LOGICAL_CONVS,
    EXPECTED_RESIDUAL_BLOCKS,
    EXPECTED_SIGNATURES,
    EXPECTED_TOTAL_CACHE_SLOTS,
    NATIVE_INT8_ACTIVATION_LAYOUT,
    NATIVE_INT8_ALGORITHM,
    NATIVE_INT8_ANALYSIS_FILE,
    NATIVE_INT8_AUDIT_FILE,
    NATIVE_INT8_AUDIT_SCHEMA_VERSION,
    NATIVE_INT8_BUILD_STATE_FILE,
    NATIVE_INT8_CUTLASS_COMMIT,
    NATIVE_INT8_INSPECTOR_FILES,
    NATIVE_INT8_MANIFEST_FILE,
    NATIVE_INT8_ONNX_FILES,
    NATIVE_INT8_PACKED_WEIGHTS_FILE,
    NATIVE_INT8_PLAN_FILES,
    NATIVE_INT8_PLUGIN_CREATORS,
    NATIVE_INT8_PLUGIN_INIT_SYMBOL,
    NATIVE_INT8_PLUGIN_LIBRARY_FILE,
    NATIVE_INT8_PLUGIN_MANIFEST_FILE,
    NATIVE_INT8_PLUGIN_NAME,
    NATIVE_INT8_PLUGIN_NAMESPACE,
    NATIVE_INT8_PLUGIN_VERSION,
    NATIVE_INT8_SCALES_FILE,
    NATIVE_INT8_SCHEMA_VERSION,
    NATIVE_INT8_SUBDIRECTORY,
    NATIVE_INT8_TIMING_CACHE_FILE,
    NATIVE_INT8_TUNE_FILE,
    NATIVE_INT8_VARIANT,
    NATIVE_INT8_WEIGHT_LAYOUT,
    NATIVE_INT8_WEIGHTS_MANIFEST_FILE,
    analyze_native_int8_graphs,
    derive_native_static_scales,
    load_native_int8_manifest,
    rewrite_native_int8_graph,
    validate_native_int8_manifest,
)
from .vae_trt_runtime import load_trt_vae_manifest, validate_trt_vae_manifest

_KINDS = ("initial", "steady")
_SOURCE_FILES = {
    "initial": "initial_int8_qdq_v5.onnx",
    "steady": "steady_int8_qdq_v5.onnx",
}
_V5_SCALES_FILE = "quant_scales_v5.json"
_STAGE_ORDER = ("analyze", "calibrate", "pack", "tune", "build", "audit")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _identity_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _base_manifest_identity(manifest: Mapping[str, Any]) -> str:
    return _identity_hash(manifest)


def _state_path(root: Path) -> Path:
    return root / NATIVE_INT8_BUILD_STATE_FILE


def _new_state(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": NATIVE_INT8_SCHEMA_VERSION,
        "variant": NATIVE_INT8_VARIANT,
        "identity": dict(identity),
        "stages": {},
    }


def _load_state(
    *, root: Path, identity: Mapping[str, Any], resume: bool
) -> dict[str, Any]:
    if not resume or not _state_path(root).is_file():
        return _new_state(identity)
    state = _load_json(_state_path(root), label="native INT8 build state")
    if (
        state.get("schema_version") != NATIVE_INT8_SCHEMA_VERSION
        or state.get("variant") != NATIVE_INT8_VARIANT
        or state.get("identity") != dict(identity)
    ):
        raise ValueError(
            "native INT8 --resume identity differs from the preserved build state; "
            "use a new engine directory or omit --resume"
        )
    if not isinstance(state.get("stages"), dict):
        raise ValueError("native INT8 build state has no stage map")
    return state


def _mark_stage(
    *, root: Path, state: dict[str, Any], stage: str, status: str, **details: Any
) -> None:
    state["stages"][stage] = {
        "status": status,
        "updated_unix_time_ns": time.time_ns(),
        **details,
    }
    write_json_atomic(_state_path(root), state)


def _stage_complete(state: Mapping[str, Any], stage: str) -> bool:
    record = state.get("stages", {}).get(stage)
    return isinstance(record, Mapping) and record.get("status") == "completed"


def _load_plugin(*, trt: Any, path: Path) -> Any:
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    try:
        library = ctypes.CDLL(str(path), mode=mode)
    except OSError as exc:
        raise RuntimeError(f"could not load native INT8 plugin: {path}") from exc
    try:
        initialize = getattr(library, NATIVE_INT8_PLUGIN_INIT_SYMBOL)
    except AttributeError as exc:
        raise RuntimeError(
            f"native INT8 plugin does not export {NATIVE_INT8_PLUGIN_INIT_SYMBOL}"
        ) from exc
    initialize.argtypes = []
    initialize.restype = ctypes.c_bool
    if not initialize():
        raise RuntimeError("native INT8 plugin registration failed")
    registry = trt.get_plugin_registry()
    if registry is None:
        raise RuntimeError("TensorRT plugin registry is unavailable")
    for name in NATIVE_INT8_PLUGIN_CREATORS:
        creator = None
        getter = getattr(registry, "get_plugin_creator", None)
        if callable(getter):
            creator = getter(
                name, NATIVE_INT8_PLUGIN_VERSION, NATIVE_INT8_PLUGIN_NAMESPACE
            )
        if creator is None:
            getter = getattr(registry, "get_creator", None)
            if callable(getter):
                creator = getter(
                    name, NATIVE_INT8_PLUGIN_VERSION, NATIVE_INT8_PLUGIN_NAMESPACE
                )
        if creator is None:
            raise RuntimeError(f"native INT8 creator was not registered: {name}")
    return library


def _plugin_manifest(
    *,
    root: Path,
    path: Path,
    environment: Mapping[str, Any],
    cutlass_commit: str,
) -> dict[str, Any]:
    return {
        "schema_version": NATIVE_INT8_SCHEMA_VERSION,
        "variant": NATIVE_INT8_VARIANT,
        "file": _relative(root, path),
        "sha256": sha256_file(path),
        "init_symbol": NATIVE_INT8_PLUGIN_INIT_SYMBOL,
        "creators": list(NATIVE_INT8_PLUGIN_CREATORS),
        "plugin_name": NATIVE_INT8_PLUGIN_NAME,
        "plugin_version": NATIVE_INT8_PLUGIN_VERSION,
        "plugin_namespace": NATIVE_INT8_PLUGIN_NAMESPACE,
        "cutlass_commit": cutlass_commit,
        "build": dict(environment),
    }


def _validate_cutlass_checkout(path: str | Path | None) -> tuple[Path, str]:
    if path is None:
        raise ValueError(
            "--cutlass-dir is required and must be the pinned native INT8 checkout"
        )
    root = Path(path).expanduser().resolve()
    if not (root / "include/cutlass/cutlass.h").is_file():
        raise ValueError(f"--cutlass-dir is not a CUTLASS checkout: {root}")
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "--cutlass-dir must retain Git metadata so its exact ABI can be bound"
        ) from exc
    if commit != NATIVE_INT8_CUTLASS_COMMIT:
        raise ValueError(
            "native_int8_v1 requires CUTLASS commit "
            f"{NATIVE_INT8_CUTLASS_COMMIT}, got {commit}"
        )
    return root, commit


def _validate_plugin_cutlass_commit(library: Any, *, expected: str) -> None:
    try:
        function = library.sfwanNativeInt8CutlassCommit
    except AttributeError as exc:
        raise RuntimeError(
            "native INT8 plugin does not expose its CUTLASS ABI commit"
        ) from exc
    function.argtypes = []
    function.restype = ctypes.c_char_p
    raw = function()
    actual = raw.decode("ascii") if isinstance(raw, bytes) else None
    if actual != expected:
        raise RuntimeError(
            f"native INT8 plugin CUTLASS ABI mismatch: {actual!r} != {expected}"
        )


def _source_paths(base_root: Path) -> dict[str, Path]:
    paths = {kind: base_root / name for kind, name in _SOURCE_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"native INT8 v5 source ONNX files are missing: {missing}")
    return paths


def _analysis(
    *,
    base_root: Path,
    base_manifest: Mapping[str, Any],
    source_paths: Mapping[str, Path],
) -> dict[str, Any]:
    targets = _target_names(base_manifest)
    contracts = _load_analysis_contracts(
        base_root=base_root, base_manifest=base_manifest
    )
    base_analysis = _run_analysis(
        source_paths=source_paths,
        target_names=targets,
        focus_module_prefix="decoder",
        analysis_contracts=contracts,
    )
    if base_analysis.get("passed") is not True:
        raise RuntimeError(
            f"native INT8 base graph analysis failed: {base_analysis.get('errors')}"
        )
    result = analyze_native_int8_graphs(
        source_paths=source_paths,
        base_analysis=base_analysis,
        target_module_names=targets,
    )
    if result.get("passed") is not True:
        raise RuntimeError(
            f"native INT8 residual/cache analysis failed: {result.get('errors')}"
        )
    return result


def _load_v5_scales(base_root: Path) -> dict[str, Any]:
    path = base_root / _V5_SCALES_FILE
    scales = _load_json(path, label="Q/DQ-v5 calibration scales")
    if scales.get("schema_version") != 5:
        raise ValueError("native INT8 requires Q/DQ-v5 calibration scales")
    return scales


def _initializer_arrays(path: Path) -> tuple[Any, dict[str, Any]]:
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:
        raise RuntimeError("native INT8 packing requires onnx") from exc
    model = onnx.load(str(path), load_external_data=True)
    return model, {
        value.name: numpy_helper.to_array(value) for value in model.graph.initializer
    }


def _pack_weights(
    *,
    root: Path,
    analysis: Mapping[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    try:
        import numpy as np
        from safetensors.numpy import save_file
    except ImportError as exc:
        raise RuntimeError(
            "native INT8 weight packing requires numpy and safetensors"
        ) from exc
    _model, initializers = _initializer_arrays(source_path)
    first_call = analysis["graphs"]["initial"]["groups_by_call"][0]
    module_records: dict[str, dict[str, Any]] = {}
    signature_ids: dict[str, set[str]] = {}
    for kind in _KINDS:
        for call in analysis["graphs"][kind]["groups_by_call"]:
            for block in call["blocks"]:
                for conv_name in ("conv1", "conv2"):
                    record = block[conv_name]
                    signature_ids.setdefault(record["module_name"], set()).add(
                        record["signature_id"]
                    )
    for block in first_call["blocks"]:
        for conv_name in ("conv1", "conv2"):
            module_records[block[conv_name]["module_name"]] = block[conv_name]
    if len(module_records) != EXPECTED_LOGICAL_CONVS:
        raise ValueError("native INT8 pack stage did not find 28 unique weights")

    tensors: dict[str, Any] = {}
    records: dict[str, Any] = {}
    for module_name in sorted(module_records):
        record = module_records[module_name]
        source = initializers.get(record["source_weight_initializer"])
        scale_source = initializers.get(record["weight_scale_initializer"])
        bias_source = initializers.get(record["bias_initializer"])
        if source is None or scale_source is None or bias_source is None:
            raise ValueError(f"native INT8 weight constants are missing: {module_name}")
        source = np.asarray(source)
        original_shape = [int(value) for value in source.shape]
        if len(original_shape) != 5:
            raise ValueError(f"native INT8 weight is not rank five: {module_name}")
        if source.dtype == np.int8:
            quantized = source.copy()
            scales = np.asarray(scale_source, dtype=np.float32).reshape(-1)
        else:
            fp32 = source.astype(np.float32, copy=False)
            absmax = np.max(np.abs(fp32), axis=(1, 2, 3, 4))
            scales = np.maximum(absmax / 127.0, np.float32(1.0e-8)).astype(
                np.float32
            )
            quantized = np.clip(
                np.rint(fp32 / scales.reshape(-1, 1, 1, 1, 1)), -127, 127
            ).astype(np.int8)
        if scales.shape != (original_shape[0],) or not np.all(
            np.isfinite(scales) & (scales > 0)
        ):
            raise ValueError(f"native INT8 weight scale is invalid: {module_name}")
        # O,I,T,R,S -> K,R,S,T*C.  TensorNHWC filters are packed KRSC.
        packed = np.ascontiguousarray(
            quantized.transpose(0, 3, 4, 2, 1).reshape(
                original_shape[0],
                original_shape[3],
                original_shape[4],
                original_shape[2] * original_shape[1],
            )
        )
        bias = np.ascontiguousarray(np.asarray(bias_source, dtype=np.float32).reshape(-1))
        if bias.shape != (original_shape[0],):
            raise ValueError(f"native INT8 bias is invalid: {module_name}")
        packed_key = f"weights/{module_name}/krstc"
        scale_key = f"weights/{module_name}/scale"
        bias_key = f"weights/{module_name}/bias"
        tensors[packed_key] = packed
        tensors[scale_key] = np.ascontiguousarray(scales)
        tensors[bias_key] = bias
        records[module_name] = {
            "module_name": module_name,
            "source_kind": record["source_weight_kind"],
            "source_initializer": record["source_weight_initializer"],
            "original_shape": original_shape,
            "original_dtype": str(source.dtype),
            "packed_key": packed_key,
            "packed_shape": [int(value) for value in packed.shape],
            "packed_dtype": "int8",
            "packed_layout": NATIVE_INT8_WEIGHT_LAYOUT,
            "packed_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
            "scale_key": scale_key,
            "scale_shape": [int(value) for value in scales.shape],
            "scale_sha256": hashlib.sha256(scales.tobytes()).hexdigest(),
            "bias_key": bias_key,
            "bias_sha256": hashlib.sha256(bias.tobytes()).hexdigest(),
            "quantization": "signed_symmetric_per_output_channel",
            "axis": 0,
            "alignment_bytes": 16,
            "signature_ids": sorted(signature_ids[module_name]),
            "signature_id": sorted(signature_ids[module_name])[0],
        }

    destination = root / NATIVE_INT8_PACKED_WEIGHTS_FILE
    temporary = destination.with_name(f"{destination.name}.partial")
    save_file(tensors, str(temporary))
    temporary.replace(destination)
    manifest = {
        "schema_version": NATIVE_INT8_SCHEMA_VERSION,
        "variant": NATIVE_INT8_VARIANT,
        "algorithm": NATIVE_INT8_ALGORITHM,
        "tensor_file": NATIVE_INT8_PACKED_WEIGHTS_FILE,
        "tensor_file_sha256": sha256_file(destination),
        "logical_weight_count": len(records),
        "weight_layout": NATIVE_INT8_WEIGHT_LAYOUT,
        "runtime_repack": False,
        "weights": records,
    }
    write_json_atomic(root / NATIVE_INT8_WEIGHTS_MANIFEST_FILE, manifest)
    return manifest


def _find_signature_records(
    analysis: Mapping[str, Any], signature_id: str
) -> list[dict[str, Any]]:
    records = []
    for kind in _KINDS:
        for call in analysis["graphs"][kind]["groups_by_call"]:
            for block in call["blocks"]:
                for conv_name in ("conv1", "conv2"):
                    record = block[conv_name]
                    if record["signature_id"] == signature_id:
                        records.append(record)
    return records


def _configure_tuner(library: Any) -> Any:
    function = library.sfwanNativeInt8TuneConv
    pointer = ctypes.POINTER(ctypes.c_int32)
    function.argtypes = [
        pointer,
        pointer,
        pointer,
        pointer,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
    ]
    function.restype = ctypes.c_int32
    count = library.sfwanNativeInt8TileCount
    count.argtypes = []
    count.restype = ctypes.c_int32
    tile_count = int(count())
    if tile_count <= 0:
        raise RuntimeError("native INT8 plugin reports no kernel tiles")
    return function, tile_count


def _tune(
    *,
    analysis: Mapping[str, Any],
    weights_manifest: Mapping[str, Any],
    library: Any,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    function, tile_count = _configure_tuner(library)
    results: dict[str, Any] = {}
    errors: list[str] = []
    for signature_id in sorted(analysis["signatures"]):
        candidates = _find_signature_records(analysis, signature_id)
        if not candidates:
            errors.append(f"signature {signature_id} has no native call site")
            continue
        record = candidates[0]
        weight = weights_manifest["weights"][record["module_name"]]
        input_shape = [int(value) for value in record["current_shape"]]
        output_shape = [int(value) for value in record["conv_output_shape"]]
        weight_shape = [int(value) for value in weight["packed_shape"]]
        attrs = record["conv_attributes"]
        pads = [int(value) for value in record["pads"]]
        params = [
            pads[3],
            pads[4],
            int(attrs["strides"][1]),
            int(attrs["strides"][2]),
            int(attrs["dilations"][1]),
            int(attrs["dilations"][2]),
        ]
        arrays = [
            (ctypes.c_int32 * len(values))(*values)
            for values in (input_shape, weight_shape, output_shape, params)
        ]
        tile_results = []
        for tile_id in range(tile_count):
            elapsed = ctypes.c_float()
            status = int(
                function(
                    arrays[0],
                    arrays[1],
                    arrays[2],
                    arrays[3],
                    tile_id,
                    warmup,
                    repeat,
                    ctypes.byref(elapsed),
                )
            )
            valid = status == 0 and math.isfinite(elapsed.value) and elapsed.value >= 0
            tile_results.append(
                {
                    "tile_id": tile_id,
                    "status": status,
                    "valid": valid,
                    "elapsed_ms": float(elapsed.value) if valid else None,
                }
            )
        valid_results = [value for value in tile_results if value["valid"]]
        if not valid_results:
            errors.append(f"signature {signature_id} has no executable SM87 tile")
            selected = None
        else:
            selected = min(valid_results, key=lambda value: value["elapsed_ms"])
        results[signature_id] = {
            "signature_id": signature_id,
            "source_call_sites": analysis["signatures"][signature_id]["call_sites"],
            "representative": record["call_site"],
            "input_shape": input_shape,
            "weight_shape": weight_shape,
            "output_shape": output_shape,
            "conv_params": params,
            "candidates": tile_results,
            "selected_tile_id": selected["tile_id"] if selected else None,
            "selected_elapsed_ms": selected["elapsed_ms"] if selected else None,
            "passed": selected is not None,
        }
    return {
        "schema_version": NATIVE_INT8_SCHEMA_VERSION,
        "variant": NATIVE_INT8_VARIANT,
        "algorithm": NATIVE_INT8_ALGORITHM,
        "compute_capability": [8, 7],
        "warmup": warmup,
        "repeat": repeat,
        "signature_count": len(results),
        "tile_count": tile_count,
        "passed": not errors and len(results) == EXPECTED_SIGNATURES,
        "errors": errors,
        "signatures": results,
    }


def _timing_cache_supported(trt: Any) -> bool:
    config_type = getattr(trt, "IBuilderConfig", None)
    return config_type is not None and all(
        callable(getattr(config_type, name, None))
        for name in ("create_timing_cache", "set_timing_cache", "get_timing_cache")
    )


def _expected_io(
    *,
    base_manifest: Mapping[str, Any],
    kind: str,
    int8_slots: set[int],
) -> dict[str, tuple[str, list[int], str]]:
    expected: dict[str, tuple[str, list[int], str]] = {
        "latent": ("input", [1, 16, 3, 60, 104], "float16"),
        "rgb": (
            "output",
            [1, 3, 9 if kind == "initial" else 12, 480, 832],
            "float16",
        ),
    }
    bindings = base_manifest["cache"]["bindings"]
    for index, binding in enumerate(bindings):
        shape = [int(value) for value in binding["shape"]]
        dtype = "int8" if index in int8_slots else "float16"
        if kind == "steady":
            expected[f"cache_in_{index:03d}"] = ("input", shape, dtype)
        expected[f"cache_out_{index:03d}"] = ("output", shape, dtype)
    return expected


def _inspector_layers(inspector: Any) -> list[dict[str, Any]]:
    if isinstance(inspector, list):
        return [value for value in inspector if isinstance(value, dict)]
    if not isinstance(inspector, dict):
        return []
    for key in ("Layers", "layers"):
        value = inspector.get(key)
        if isinstance(value, list):
            return [record for record in value if isinstance(record, dict)]
    return [inspector]


def _native_plugin_present(inspector: Any) -> bool:
    return any(
        NATIVE_INT8_PLUGIN_NAME
        in json.dumps(record, sort_keys=True, separators=(",", ":"))
        or "native_int8/" in json.dumps(record, sort_keys=True, separators=(",", ":"))
        for record in _inspector_layers(inspector)
    )


def _map_native_plugins(
    inspector: Any, *, expected_names: list[str]
) -> dict[str, Any]:
    """Bind every logical residual plugin to physical Inspector evidence.

    TensorRT 10.3 may put the ONNX/plugin name in either ``Name`` or
    ``Metadata`` and may wrap the layer list in an object.  Matching the
    canonical full plugin name avoids accepting a plan that retained only one
    native node while silently lowering the other residual blocks.
    """

    layer_text = [
        json.dumps(record, sort_keys=True, separators=(",", ":"))
        for record in _inspector_layers(inspector)
    ]
    mapped = [
        name for name in expected_names if any(name in text for text in layer_text)
    ]
    missing = sorted(set(expected_names) - set(mapped))
    return {
        "expected_count": len(expected_names),
        "mapped_count": len(mapped),
        "mapped": mapped,
        "missing": missing,
        "passed": not missing,
    }


def _audit_graph(path: Path) -> dict[str, Any]:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("native INT8 graph audit requires onnx") from exc
    model = onnx.load(str(path), load_external_data=True)
    plugin_nodes = [
        node for node in model.graph.node if node.op_type == NATIVE_INT8_PLUGIN_NAME
    ]
    target_nodes = [
        node
        for node in model.graph.node
        if node.name.startswith("int8/")
        and node.op_type
        in {"Conv", "QuantizeLinear", "DequantizeLinear", "Cast"}
    ]
    return {
        "passed": len(plugin_nodes) == EXPECTED_RESIDUAL_BLOCKS * 3
        and not target_nodes,
        "plugin_node_count": len(plugin_nodes),
        "replaced_target_conv_call_site_count": len(plugin_nodes) * 2,
        "remaining_target_boundary_nodes": [node.name for node in target_nodes],
    }


def _cache_contract(
    *,
    base_manifest: Mapping[str, Any],
    analysis: Mapping[str, Any],
    scales: Mapping[str, Any],
) -> dict[str, Any]:
    selected_by_index = {
        int(entry["index"]): entry for entry in analysis["int8_cache_slots"]
    }
    scale_by_module = scales["cache"]
    bindings = []
    bank_bytes = 0
    for index, base in enumerate(base_manifest["cache"]["bindings"]):
        shape = [int(value) for value in base["shape"]]
        selected = selected_by_index.get(index)
        if selected is None:
            record = {
                "index": index,
                "input_name": f"cache_in_{index:03d}",
                "output_name": f"cache_out_{index:03d}",
                "shape": shape,
                "dtype": "float16",
                "format": "linear",
                "owner_module": base.get("module_name"),
                "scale": None,
            }
            element_bytes = 2
        else:
            record = {
                "index": index,
                "input_name": f"cache_in_{index:03d}",
                "output_name": f"cache_out_{index:03d}",
                "shape": shape,
                "dtype": "int8",
                "format": NATIVE_INT8_ACTIVATION_LAYOUT.lower(),
                "owner_module": selected["module_name"],
                "scale": float(scale_by_module[selected["module_name"]]["scale"]),
            }
            element_bytes = 1
        record["bank_offset"] = bank_bytes
        record["bytes"] = math.prod(shape) * element_bytes
        bank_bytes += int(record["bytes"])
        bindings.append(record)
    return {
        "tensor_count": len(bindings),
        "int8_slot_count": sum(value["dtype"] == "int8" for value in bindings),
        "fp16_slot_count": sum(value["dtype"] == "float16" for value in bindings),
        "mixed_dtype": True,
        "migration": "none",
        "single_bank_bytes": bank_bytes,
        "double_bank_bytes": bank_bytes * 2,
        "bindings": bindings,
    }


def _environment_matches(
    *, environment: Mapping[str, Any], base_manifest: Mapping[str, Any]
) -> None:
    if environment.get("compute_capability") != [8, 7]:
        raise RuntimeError("native_int8_v1 requires Jetson Orin SM87")
    base = base_manifest.get("build", {})
    for key in ("tensorrt_version", "cuda_version"):
        if str(environment.get(key)) != str(base.get(key)):
            raise RuntimeError(
                f"native/base build environment differs in {key}: "
                f"{environment.get(key)} != {base.get(key)}"
            )


def build_native_int8(
    *,
    engine_dir: str | Path,
    stage: str,
    cutlass_dir: str | Path | None,
    workspace_gib: float,
    tune_warmup: int,
    tune_repeat: int,
    resume: bool,
    preflight_only: bool,
    device_index: int,
    plugin_library: str | Path | None,
) -> dict[str, Any]:
    if stage not in {*_STAGE_ORDER, "all"}:
        raise ValueError(f"unsupported native INT8 build stage: {stage}")
    if workspace_gib <= 0 or tune_warmup < 0 or tune_repeat <= 0:
        raise ValueError("workspace/tune parameters are invalid")
    try:
        import onnx
        import tensorrt as trt
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "native INT8 build requires onnx, tensorrt, and CUDA torch"
        ) from exc

    base_root = Path(engine_dir).expanduser().resolve()
    root = base_root / NATIVE_INT8_SUBDIRECTORY
    root.mkdir(parents=True, exist_ok=True)
    base_manifest = load_trt_vae_manifest(base_root)
    validate_trt_vae_manifest(
        base_manifest,
        engine_dir=base_root,
        precision="int8",
        model_path=None,
        verify_plan_hashes=True,
    )
    sources = _source_paths(base_root)
    environment = _environment(
        trt=trt, torch=torch, onnx=onnx, device_index=device_index
    )
    _environment_matches(environment=environment, base_manifest=base_manifest)
    cutlass, cutlass_commit = _validate_cutlass_checkout(cutlass_dir)

    destination_plugin = root / NATIVE_INT8_PLUGIN_LIBRARY_FILE
    if plugin_library is not None:
        source_plugin = Path(plugin_library).expanduser().resolve()
        if not source_plugin.is_file():
            raise ValueError(f"native INT8 plugin does not exist: {source_plugin}")
        if source_plugin != destination_plugin:
            shutil.copy2(source_plugin, destination_plugin)
    plugin_sha = sha256_file(destination_plugin) if destination_plugin.is_file() else None
    identity = {
        "schema_version": NATIVE_INT8_SCHEMA_VERSION,
        "base_manifest_sha256": _base_manifest_identity(base_manifest),
        "source_sha256": {kind: sha256_file(path) for kind, path in sources.items()},
        "plugin_sha256": plugin_sha,
        "cutlass_dir": str(cutlass),
        "cutlass_commit": cutlass_commit,
        "workspace_gib": float(workspace_gib),
        "tune_warmup": tune_warmup,
        "tune_repeat": tune_repeat,
        "environment": environment,
    }
    state = _load_state(root=root, identity=identity, resume=resume)
    limit = "tune" if preflight_only else ("audit" if stage == "all" else stage)
    stages = _STAGE_ORDER[: _STAGE_ORDER.index(limit) + 1]
    plugin_manifest: dict[str, Any] | None = None
    library = None

    if resume and (root / NATIVE_INT8_MANIFEST_FILE).is_file() and limit == "audit":
        existing = load_native_int8_manifest(base_root)
        validate_native_int8_manifest(
            existing,
            engine_dir=base_root,
            base_manifest=base_manifest,
            verify_hashes=True,
        )
        return existing

    analysis_path = root / NATIVE_INT8_ANALYSIS_FILE
    if "analyze" in stages:
        if resume and _stage_complete(state, "analyze") and analysis_path.is_file():
            analysis = _load_json(analysis_path, label="native INT8 analysis")
        else:
            _mark_stage(root=root, state=state, stage="analyze", status="running")
            try:
                analysis = _analysis(
                    base_root=base_root,
                    base_manifest=base_manifest,
                    source_paths=sources,
                )
                write_json_atomic(analysis_path, analysis)
                _mark_stage(
                    root=root,
                    state=state,
                    stage="analyze",
                    status="completed",
                    artifact_sha256=sha256_file(analysis_path),
                )
            except BaseException as exc:
                _mark_stage(
                    root=root,
                    state=state,
                    stage="analyze",
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
    else:
        analysis = _load_json(analysis_path, label="native INT8 analysis")
    if analysis.get("passed") is not True:
        raise RuntimeError("native INT8 analysis artifact did not pass")
    if limit == "analyze":
        return analysis

    scales_path = root / NATIVE_INT8_SCALES_FILE
    if "calibrate" in stages:
        if resume and _stage_complete(state, "calibrate") and scales_path.is_file():
            scales = _load_json(scales_path, label="native INT8 scales")
        else:
            _mark_stage(root=root, state=state, stage="calibrate", status="running")
            try:
                scales = derive_native_static_scales(
                    analysis=analysis, v5_scales=_load_v5_scales(base_root)
                )
                write_json_atomic(scales_path, scales)
                _mark_stage(
                    root=root,
                    state=state,
                    stage="calibrate",
                    status="completed",
                    artifact_sha256=sha256_file(scales_path),
                )
            except BaseException as exc:
                _mark_stage(
                    root=root,
                    state=state,
                    stage="calibrate",
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
    else:
        scales = _load_json(scales_path, label="native INT8 scales")
    if limit == "calibrate":
        return scales

    weights_manifest_path = root / NATIVE_INT8_WEIGHTS_MANIFEST_FILE
    if "pack" in stages:
        packed_path = root / NATIVE_INT8_PACKED_WEIGHTS_FILE
        if (
            resume
            and _stage_complete(state, "pack")
            and packed_path.is_file()
            and weights_manifest_path.is_file()
        ):
            weights_manifest = _load_json(
                weights_manifest_path, label="native INT8 weights manifest"
            )
            if sha256_file(packed_path) != weights_manifest.get("tensor_file_sha256"):
                raise ValueError("native packed-weight SHA256 changed")
        else:
            _mark_stage(root=root, state=state, stage="pack", status="running")
            try:
                weights_manifest = _pack_weights(
                    root=root,
                    analysis=analysis,
                    source_path=sources["initial"],
                )
                _mark_stage(
                    root=root,
                    state=state,
                    stage="pack",
                    status="completed",
                    artifact_sha256=sha256_file(weights_manifest_path),
                    tensor_sha256=weights_manifest["tensor_file_sha256"],
                )
            except BaseException as exc:
                _mark_stage(
                    root=root,
                    state=state,
                    stage="pack",
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
    else:
        weights_manifest = _load_json(
            weights_manifest_path, label="native INT8 weights manifest"
        )
    if limit == "pack":
        return weights_manifest

    if not destination_plugin.is_file():
        raise ValueError(
            f"install {NATIVE_INT8_PLUGIN_LIBRARY_FILE} into {root} before "
            f"running stage {limit}"
        )
    plugin_manifest = _plugin_manifest(
        root=root,
        path=destination_plugin,
        environment=environment,
        cutlass_commit=cutlass_commit,
    )
    write_json_atomic(root / NATIVE_INT8_PLUGIN_MANIFEST_FILE, plugin_manifest)
    library = _load_plugin(trt=trt, path=destination_plugin)
    _validate_plugin_cutlass_commit(library, expected=cutlass_commit)

    tune_path = root / NATIVE_INT8_TUNE_FILE
    if "tune" in stages:
        if resume and _stage_complete(state, "tune") and tune_path.is_file():
            tune = _load_json(tune_path, label="native INT8 tune report")
        else:
            _mark_stage(root=root, state=state, stage="tune", status="running")
            try:
                tune = _tune(
                    analysis=analysis,
                    weights_manifest=weights_manifest,
                    library=library,
                    warmup=tune_warmup,
                    repeat=tune_repeat,
                )
                write_json_atomic(tune_path, tune)
                if tune.get("passed") is not True:
                    raise RuntimeError(
                        f"native INT8 SM87 tune failed: {tune.get('errors')}"
                    )
                _mark_stage(
                    root=root,
                    state=state,
                    stage="tune",
                    status="completed",
                    artifact_sha256=sha256_file(tune_path),
                )
            except BaseException as exc:
                _mark_stage(
                    root=root,
                    state=state,
                    stage="tune",
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
    else:
        tune = _load_json(tune_path, label="native INT8 tune report")
    if tune.get("passed") is not True:
        raise RuntimeError("native INT8 tune report did not pass")
    if limit == "tune":
        return tune

    try:
        from safetensors.numpy import load_file
    except ImportError as exc:
        raise RuntimeError("native INT8 graph build requires safetensors") from exc
    packed_tensors = load_file(str(root / NATIVE_INT8_PACKED_WEIGHTS_FILE))
    graph_records: dict[str, Any] = {}
    plan_records: dict[str, Any] = {}
    stable_cache = root / NATIVE_INT8_TIMING_CACHE_FILE
    candidate_cache = root / f"{NATIVE_INT8_TIMING_CACHE_FILE}.candidate"
    use_timing_cache = _timing_cache_supported(trt)
    reuse_built_plans = (
        resume
        and _stage_complete(state, "build")
        and all(
            (root / NATIVE_INT8_PLAN_FILES[kind]).is_file()
            and (root / NATIVE_INT8_ONNX_FILES[kind]).is_file()
            for kind in _KINDS
        )
    )
    if "build" in stages and not reuse_built_plans:
        _mark_stage(root=root, state=state, stage="build", status="running")
        try:
            candidate_cache.unlink(missing_ok=True)
            if use_timing_cache and stable_cache.is_file():
                _write_bytes(candidate_cache, stable_cache.read_bytes())
            for kind in _KINDS:
                graph_records[kind] = rewrite_native_int8_graph(
                    source_path=sources[kind],
                    destination_path=root / NATIVE_INT8_ONNX_FILES[kind],
                    graph_analysis=analysis["graphs"][kind],
                    packed_tensors=packed_tensors,
                    weights_manifest=weights_manifest,
                    static_scales=scales,
                    tune=tune,
                )
                plan, io_contract, inspector_json, candidate_bytes = (
                    _build_engine_bytes(
                        trt=trt,
                        onnx_path=root / NATIVE_INT8_ONNX_FILES[kind],
                        workspace_gib=workspace_gib,
                        profiling_verbosity="detailed",
                        timing_cache_path=candidate_cache if use_timing_cache else None,
                    )
                )
                if use_timing_cache and candidate_bytes is not None:
                    _write_bytes(candidate_cache, candidate_bytes)
                _write_bytes(root / NATIVE_INT8_PLAN_FILES[kind], plan)
                try:
                    inspector = json.loads(inspector_json)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"TensorRT returned invalid {kind} inspector JSON"
                    ) from exc
                write_json_atomic(root / NATIVE_INT8_INSPECTOR_FILES[kind], inspector)
                plan_records[kind] = {
                    "file": NATIVE_INT8_PLAN_FILES[kind],
                    "sha256": hashlib.sha256(plan).hexdigest(),
                    "source_onnx_file": NATIVE_INT8_ONNX_FILES[kind],
                    "source_onnx_sha256": sha256_file(
                        root / NATIVE_INT8_ONNX_FILES[kind]
                    ),
                    "inspector_file": NATIVE_INT8_INSPECTOR_FILES[kind],
                    "inspector_sha256": sha256_file(
                        root / NATIVE_INT8_INSPECTOR_FILES[kind]
                    ),
                    "profiling_verbosity": "detailed",
                    "io_tensors": io_contract,
                }
            _mark_stage(
                root=root,
                state=state,
                stage="build",
                status="completed",
                plan_sha256={kind: value["sha256"] for kind, value in plan_records.items()},
                timing_cache_supported=use_timing_cache,
            )
        except BaseException as exc:
            _mark_stage(
                root=root,
                state=state,
                stage="build",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
    else:
        for kind in _KINDS:
            plan_path = root / NATIVE_INT8_PLAN_FILES[kind]
            io_contract, inspector_json = _load_plan(trt=trt, plan_path=plan_path)
            inspector_path = root / NATIVE_INT8_INSPECTOR_FILES[kind]
            inspector = json.loads(inspector_json)
            write_json_atomic(inspector_path, inspector)
            graph_records[kind] = {
                "destination": str((root / NATIVE_INT8_ONNX_FILES[kind]).resolve()),
                "destination_sha256": sha256_file(root / NATIVE_INT8_ONNX_FILES[kind]),
                "plugin_node_count": EXPECTED_RESIDUAL_BLOCKS * 3,
                "replaced_target_conv_call_site_count": EXPECTED_CALL_SITES_PER_GRAPH,
                "selected_int8_cache_slots": [
                    int(value["index"]) for value in analysis["int8_cache_slots"]
                ],
                "profile_ids": [
                    value
                    for call in analysis["graphs"][kind]["groups_by_call"]
                    for block in call["blocks"]
                    for value in [
                        {
                            "profile_id": block["profile_id"],
                            "plugin_name": (
                                f"native_int8/{kind}/{block['prefix']}/"
                                f"call_{block['call_index']}"
                            ),
                            "block_prefix": block["prefix"],
                            "call_index": block["call_index"],
                            "conv1_signature": block["conv1"]["signature_id"],
                            "conv2_signature": block["conv2"]["signature_id"],
                        }
                    ]
                ],
            }
            plan_records[kind] = {
                "file": NATIVE_INT8_PLAN_FILES[kind],
                "sha256": sha256_file(plan_path),
                "source_onnx_file": NATIVE_INT8_ONNX_FILES[kind],
                "source_onnx_sha256": sha256_file(root / NATIVE_INT8_ONNX_FILES[kind]),
                "inspector_file": NATIVE_INT8_INSPECTOR_FILES[kind],
                "inspector_sha256": sha256_file(inspector_path),
                "profiling_verbosity": "detailed",
                "io_tensors": io_contract,
            }
    if limit == "build":
        return {"graphs": graph_records, "engines": plan_records}

    _mark_stage(root=root, state=state, stage="audit", status="running")
    try:
        cache = _cache_contract(
            base_manifest=base_manifest, analysis=analysis, scales=scales
        )
        int8_slots = {
            int(value["index"])
            for value in cache["bindings"]
            if value["dtype"] == "int8"
        }
        errors: list[str] = []
        graph_audits = {}
        inspector_presence = {}
        inspector_plugin_mappings = {}
        for kind in _KINDS:
            _validate_io(
                records=plan_records[kind]["io_tensors"],
                expected=_expected_io(
                    base_manifest=base_manifest, kind=kind, int8_slots=int8_slots
                ),
            )
            graph_audits[kind] = _audit_graph(root / NATIVE_INT8_ONNX_FILES[kind])
            inspector = json.loads(
                (root / NATIVE_INT8_INSPECTOR_FILES[kind]).read_text(encoding="utf-8")
            )
            inspector_presence[kind] = _native_plugin_present(inspector)
            expected_plugin_names = [
                str(record["plugin_name"])
                for record in graph_records[kind]["profile_ids"]
            ]
            inspector_plugin_mappings[kind] = _map_native_plugins(
                inspector, expected_names=expected_plugin_names
            )
            if graph_audits[kind]["passed"] is not True:
                errors.append(f"{kind} graph did not replace all target boundaries")
            if not inspector_presence[kind]:
                errors.append(f"{kind} plan has no native plugin evidence")
            if inspector_plugin_mappings[kind]["passed"] is not True:
                errors.append(
                    f"{kind} plan did not map all 42 native residual plugins: "
                    f"{inspector_plugin_mappings[kind]['missing']}"
                )
        if cache["int8_slot_count"] != EXPECTED_INT8_CACHE_SLOTS:
            errors.append("native INT8 cache-slot count changed")
        if cache["fp16_slot_count"] != EXPECTED_FP16_CACHE_SLOTS:
            errors.append("native FP16 cache-slot count changed")
        if tune.get("signature_count") != EXPECTED_SIGNATURES or not all(
            value.get("passed") is True for value in tune["signatures"].values()
        ):
            errors.append("native SM87 kernel signature coverage is incomplete")
        audit = {
            "schema_version": NATIVE_INT8_AUDIT_SCHEMA_VERSION,
            "variant": NATIVE_INT8_VARIANT,
            "algorithm": NATIVE_INT8_ALGORITHM,
            "passed": not errors,
            "complete": not errors,
            "errors": errors,
            "logical_residual_block_count": EXPECTED_RESIDUAL_BLOCKS,
            "logical_target_conv_count": EXPECTED_LOGICAL_CONVS,
            "initial_call_site_count": graph_audits["initial"][
                "replaced_target_conv_call_site_count"
            ],
            "steady_call_site_count": graph_audits["steady"][
                "replaced_target_conv_call_site_count"
            ],
            "signature_count": tune["signature_count"],
            "graph": graph_audits,
            "plugin_present": inspector_presence,
            "inspector_plugin_mappings": inspector_plugin_mappings,
            "plan_sha256": {
                kind: plan_records[kind]["sha256"] for kind in _KINDS
            },
            "packed_weights_sha256": weights_manifest["tensor_file_sha256"],
            "cache": {
                "int8_slot_count": cache["int8_slot_count"],
                "fp16_slot_count": cache["fp16_slot_count"],
                "migration": cache["migration"],
                "initial_steady_abi_identical": True,
            },
            "runtime_scale_mode": "static",
            "weight_mode": "offline_per_channel_packed",
            "forbidden_target_qdq_cast_reformat_count": 0,
        }
        write_json_atomic(root / NATIVE_INT8_AUDIT_FILE, audit)
        if errors:
            raise RuntimeError(f"native INT8 audit failed: {errors}")
        if use_timing_cache and candidate_cache.is_file():
            candidate_cache.replace(stable_cache)
        elif not stable_cache.is_file():
            _write_bytes(stable_cache, b"")
        manifest = {
            "schema_version": NATIVE_INT8_SCHEMA_VERSION,
            "variant": NATIVE_INT8_VARIANT,
            "algorithm": NATIVE_INT8_ALGORITHM,
            "identity": identity,
            "base_manifest_sha256": _base_manifest_identity(base_manifest),
            "plugin": plugin_manifest,
            "engines": plan_records,
            "graph": graph_records,
            "analysis": {
                "file": NATIVE_INT8_ANALYSIS_FILE,
                "sha256": sha256_file(analysis_path),
            },
            "scales": {
                "file": NATIVE_INT8_SCALES_FILE,
                "sha256": sha256_file(scales_path),
                "runtime_mode": "static",
            },
            "weights": {
                "file": NATIVE_INT8_PACKED_WEIGHTS_FILE,
                "sha256": weights_manifest["tensor_file_sha256"],
                "manifest_file": NATIVE_INT8_WEIGHTS_MANIFEST_FILE,
                "manifest_sha256": sha256_file(weights_manifest_path),
                "mode": "offline_per_channel_packed",
                "layout": NATIVE_INT8_WEIGHT_LAYOUT,
            },
            "tune": {
                "file": NATIVE_INT8_TUNE_FILE,
                "sha256": sha256_file(tune_path),
                "kernel_tile_map": {
                    signature_id: value["selected_tile_id"]
                    for signature_id, value in tune["signatures"].items()
                },
            },
            "cache": cache,
            "audit": {
                "file": NATIVE_INT8_AUDIT_FILE,
                "sha256": sha256_file(root / NATIVE_INT8_AUDIT_FILE),
                "passed": True,
            },
            "kernel_profile_ids": {
                kind: graph_records[kind]["profile_ids"] for kind in _KINDS
            },
            "timing_cache": {
                "file": NATIVE_INT8_TIMING_CACHE_FILE,
                "sha256": sha256_file(stable_cache),
                "api_supported": use_timing_cache,
            },
            "build": environment,
        }
        write_json_atomic(root / NATIVE_INT8_MANIFEST_FILE, manifest)
        validate_native_int8_manifest(
            manifest,
            engine_dir=base_root,
            base_manifest=base_manifest,
            verify_hashes=True,
        )
        _mark_stage(
            root=root,
            state=state,
            stage="audit",
            status="completed",
            manifest_sha256=sha256_file(root / NATIVE_INT8_MANIFEST_FILE),
        )
        return manifest
    except BaseException as exc:
        _mark_stage(
            root=root,
            state=state,
            stage="audit",
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SFWan VAE native SM87 INT8 residual-block engines"
    )
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument(
        "--stage", choices=(*_STAGE_ORDER, "all"), default="all"
    )
    parser.add_argument("--cutlass-dir")
    parser.add_argument("--workspace-gib", type=float, default=8.0)
    parser.add_argument("--tune-warmup", type=int, default=0)
    parser.add_argument("--tune-repeat", type=int, default=1)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--plugin-library")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_native_int8(
        engine_dir=args.engine_dir,
        stage=args.stage,
        cutlass_dir=args.cutlass_dir,
        workspace_gib=args.workspace_gib,
        tune_warmup=args.tune_warmup,
        tune_repeat=args.tune_repeat,
        resume=args.resume,
        preflight_only=args.preflight_only,
        device_index=args.device_index,
        plugin_library=args.plugin_library,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
