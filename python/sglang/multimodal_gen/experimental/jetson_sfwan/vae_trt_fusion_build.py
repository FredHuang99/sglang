"""Analyze, micro-probe, and build isolated SFWan VAE fusion-v1 plans.

The command consumes a completed and audited explicit-Q/DQ v5 engine
directory.  It never overwrites v5 artifacts.  The expensive full build is
allowed only after every required ``decoder.up_blocks.3`` signature passes a
real on-device micro-probe with an INT8 Conv tactic and no intervening
reformat between the pack/quant plugin and Conv.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

from .vae_trt_build import (
    _audit_feature_cache_kind_io,
    _audit_tensorrt_tactics,
    _build_engine_bytes,
    _write_bytes,
)
from .vae_trt_fusion import (
    CACHE_UPDATE_PLUGIN,
    EPILOGUE_PLUGIN,
    FUSION_AUDIT_FILE,
    FUSION_AUDIT_SCHEMA_VERSION,
    FUSION_ANALYSIS_SCHEMA_VERSION,
    FUSION_BUILD_STATE_FILE,
    FUSION_ENGINE_FILES,
    FUSION_INSPECTOR_FILES,
    FUSION_MANIFEST_FILE,
    FUSION_ONNX_FILES,
    FUSION_PLUGIN_LIBRARY_FILE,
    FUSION_PLUGIN_MANIFEST_FILE,
    FUSION_PROBE_FILE,
    FUSION_SCHEMA_VERSION,
    FUSION_SUBDIRECTORY,
    FUSION_TIMING_CACHE_FILE,
    FUSION_VARIANT,
    PACK_QUANT_PLUGIN,
    PLUGIN_CREATORS,
    PLUGIN_NAMESPACE,
    PLUGIN_VERSION,
    analyze_fusion_graph,
    rewrite_fusion_graph,
    sha256_file,
    validate_fusion_manifest,
    write_json_atomic,
)
from .vae_trt_qdq import EXPECTED_CALL_SITES, QDQ_SCHEMA_VERSION
from .vae_trt_runtime import (
    TRT_VAE_CACHE_BANK_BYTES,
    TRT_VAE_CACHE_COUNT,
    TRT_VAE_CACHE_TOTAL_ELEMENTS,
    load_trt_vae_manifest,
    validate_trt_vae_manifest,
)

_KINDS = ("initial", "steady")
_SOURCE_FILES = {
    "initial": "initial_int8_qdq_v5.onnx",
    "steady": "steady_int8_qdq_v5.onnx",
}
_ANALYSIS_FILE = "fusion_analysis_v2.json"
_PROBE_SUBDIRECTORY = "probes"
_SCHEMES = (
    "baseline",
    "input_pack_quant",
    "input_pack_quant_cache_dual",
    "input_pack_quant_cache_split",
    "full_boundary_dual",
    "full_boundary_split",
)


class FusionAuditError(RuntimeError):
    """A plan was built, but its physical tactic/plugin contract failed."""


class _ProbeInfrastructureError(RuntimeError):
    """A probe setup failure that cannot be repaired by another scheme."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _load_plugin(*, trt: Any, plugin_path: Path) -> Any:
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    try:
        library = ctypes.CDLL(str(plugin_path), mode=mode)
    except OSError as exc:
        raise RuntimeError(f"could not load fusion plugin: {plugin_path}") from exc
    try:
        initialize = library.initSfWanVaeTrtFusionPlugins
    except AttributeError as exc:
        raise RuntimeError(
            "fusion plugin does not export initSfWanVaeTrtFusionPlugins"
        ) from exc
    initialize.argtypes = []
    initialize.restype = ctypes.c_bool
    if not initialize():
        raise RuntimeError("fusion plugin creator registration failed")
    registry = trt.get_plugin_registry()
    if registry is None:
        raise RuntimeError("TensorRT plugin registry is unavailable")
    missing = []
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
            missing.append(name)
    if missing:
        raise RuntimeError(f"fusion plugin creators were not registered: {missing}")
    return library


def _environment(
    *, trt: Any, torch: Any, onnx: Any, device_index: int
) -> dict[str, Any]:
    capability = list(torch.cuda.get_device_capability(device_index))
    if capability != [8, 7]:
        raise RuntimeError(f"fusion-v1 requires SM87, got SM{capability}")
    return {
        "compute_capability": capability,
        "gpu_name": str(torch.cuda.get_device_name(device_index)),
        "tensorrt_version": str(trt.__version__),
        "cuda_version": str(torch.version.cuda),
        "torch_version": str(torch.__version__),
        "onnx_version": str(onnx.__version__),
        "device_index": device_index,
    }


def _plugin_manifest(
    *, fusion_root: Path, plugin_path: Path, environment: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": FUSION_SCHEMA_VERSION,
        "variant": FUSION_VARIANT,
        "file": _relative(fusion_root, plugin_path),
        "sha256": sha256_file(plugin_path),
        "namespace": PLUGIN_NAMESPACE,
        "version": PLUGIN_VERSION,
        "creators": list(PLUGIN_CREATORS),
        "build": dict(environment),
    }


def _identity(
    *,
    base_root: Path,
    base_manifest_path: Path,
    source_paths: Mapping[str, Path],
    plugin_path: Path,
    environment: Mapping[str, Any],
    focus_module_prefix: str,
    workspace_gib: float,
    probe_warmup: int,
    probe_repeat: int,
    int8_audit_sha256: str,
    analysis_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": FUSION_SCHEMA_VERSION,
        "analysis_schema_version": FUSION_ANALYSIS_SCHEMA_VERSION,
        "base_root": str(base_root),
        "base_manifest_sha256": sha256_file(base_manifest_path),
        "source_sha256": {kind: sha256_file(source_paths[kind]) for kind in _KINDS},
        "plugin_sha256": sha256_file(plugin_path),
        "environment": dict(environment),
        "focus_module_prefix": focus_module_prefix,
        "workspace_gib": float(workspace_gib),
        "probe_warmup": probe_warmup,
        "probe_repeat": probe_repeat,
        "int8_audit_sha256": int8_audit_sha256,
        "analysis_contract_sha256": analysis_contract_sha256,
    }


def _state_stage(
    *, state: dict[str, Any], state_path: Path, name: str, status: str, **values: Any
) -> None:
    stages = state.setdefault("stages", {})
    stages[name] = {
        "status": status,
        "updated_unix_time_ns": time.time_ns(),
        **values,
    }
    write_json_atomic(state_path, state)


def _target_names(base_manifest: Mapping[str, Any]) -> tuple[str, ...]:
    quantization = base_manifest.get("quantization")
    names = (
        quantization.get("target_module_names")
        if isinstance(quantization, dict)
        else None
    )
    if (
        not isinstance(names, list)
        or len(names) != 28
        or any(not isinstance(value, str) or not value for value in names)
        or len(set(names)) != 28
    ):
        raise ValueError("base manifest has no complete 28-Conv target list")
    return tuple(names)


def _load_analysis_contracts(
    *, base_root: Path, base_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Load the SHA-bound v5 shapes that ONNX value_info may omit.

    The v5 builder captured these Conv input/output shapes from the real Wan
    forward before export.  They are more authoritative than best-effort ONNX
    shape inference, while the manifest owns the public FP16 cache ABI.
    """

    audit_record = base_manifest.get("int8_audit")
    if not isinstance(audit_record, Mapping):
        raise ValueError("base manifest has no INT8 v5 audit record")
    report_file = audit_record.get("report_file")
    report_sha256 = audit_record.get("report_sha256")
    if (
        not isinstance(report_file, str)
        or not report_file
        or not isinstance(report_sha256, str)
        or len(report_sha256) != 64
    ):
        raise ValueError("base manifest INT8 v5 audit reference is invalid")
    report_path = (base_root / report_file).resolve()
    try:
        report_path.relative_to(base_root)
    except ValueError as exc:
        raise ValueError(
            "base INT8 v5 audit path escapes the engine directory"
        ) from exc
    if not report_path.is_file() or sha256_file(report_path) != report_sha256:
        raise ValueError("base INT8 v5 audit report SHA256 does not match")
    report = _load_json(report_path, label="base INT8 v5 audit report")
    if (
        report.get("schema_version") != QDQ_SCHEMA_VERSION
        or report.get("passed") is not True
        or report.get("complete") is not True
        or report.get("errors") != []
    ):
        raise ValueError("base INT8 v5 audit report is incomplete")

    structural = report.get("structural")
    if not isinstance(structural, Mapping):
        raise ValueError("base INT8 v5 structural audits are missing")
    graph_contracts: dict[str, dict[str, dict[str, Any]]] = {}
    for kind in _KINDS:
        graph = structural.get(kind)
        signatures = (
            graph.get("conv_signatures") if isinstance(graph, Mapping) else None
        )
        if not isinstance(signatures, list) or len(signatures) != EXPECTED_CALL_SITES:
            raise ValueError(
                f"base INT8 v5 {kind} audit has no complete Conv shape contracts"
            )
        contracts: dict[str, dict[str, Any]] = {}
        expected_prefix = f"int8/{kind}/"
        for signature in signatures:
            if not isinstance(signature, Mapping):
                raise ValueError(f"base INT8 v5 {kind} Conv contract is invalid")
            call_site = signature.get("call_site")
            input_shape = signature.get("input_shape")
            output_shape = signature.get("output_shape")
            if (
                not isinstance(call_site, str)
                or not call_site.startswith(expected_prefix)
                or call_site in contracts
            ):
                raise ValueError(f"base INT8 v5 {kind} Conv call site is invalid")
            for label, shape in (("input", input_shape), ("output", output_shape)):
                if (
                    not isinstance(shape, list)
                    or len(shape) != 5
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value <= 0
                        for value in shape
                    )
                ):
                    raise ValueError(
                        f"base INT8 v5 {call_site} {label} shape is invalid: {shape}"
                    )
            contracts[call_site] = {
                "input_shape": [int(value) for value in input_shape],
                "output_shape": [int(value) for value in output_shape],
            }
        graph_contracts[kind] = contracts

    cache = base_manifest.get("cache")
    bindings = cache.get("bindings") if isinstance(cache, Mapping) else None
    if not isinstance(bindings, list) or len(bindings) != TRT_VAE_CACHE_COUNT:
        raise ValueError("base manifest has no complete feature-cache bindings")
    cache_shapes = {kind: {} for kind in _KINDS}
    seen_inputs: set[str] = set()
    seen_outputs: set[str] = set()
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise ValueError("base manifest feature-cache binding is invalid")
        input_name = binding.get("input_name")
        output_name = binding.get("output_name")
        shape = binding.get("shape")
        if (
            not isinstance(input_name, str)
            or not input_name
            or input_name in seen_inputs
            or not isinstance(output_name, str)
            or not output_name
            or output_name in seen_outputs
            or not isinstance(shape, list)
            or len(shape) != 5
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in shape
            )
            or binding.get("dtype") != "float16"
        ):
            raise ValueError("base manifest feature-cache binding contract is invalid")
        seen_inputs.add(input_name)
        seen_outputs.add(output_name)
        normalized = [int(value) for value in shape]
        cache_shapes["initial"][output_name] = normalized
        cache_shapes["steady"][input_name] = normalized
        cache_shapes["steady"][output_name] = normalized

    payload = {
        "analysis_schema_version": FUSION_ANALYSIS_SCHEMA_VERSION,
        "int8_audit_sha256": report_sha256,
        "graphs": graph_contracts,
        "cache_shapes": cache_shapes,
    }
    contract_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **payload,
        "contract_sha256": contract_sha256,
        "int8_audit_path": str(report_path),
    }


def _analysis_signature(record: Mapping[str, Any]) -> str:
    payload = {
        key: record.get(key)
        for key in (
            "conv_kind",
            "current_shape",
            "cache_shape",
            "padded_shape",
            "pads",
            "epilogue_output_shape",
        )
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _run_analysis(
    *,
    source_paths: Mapping[str, Path],
    target_names: tuple[str, ...],
    focus_module_prefix: str,
    analysis_contracts: Mapping[str, Any],
) -> dict[str, Any]:
    analyses = {
        kind: analyze_fusion_graph(
            source_path=source_paths[kind],
            graph_kind=kind,
            target_module_names=target_names,
            focus_module_prefix=focus_module_prefix,
            shape_contracts=analysis_contracts["graphs"][kind],
            cache_shape_contracts=analysis_contracts["cache_shapes"][kind],
        )
        for kind in _KINDS
    }
    records = [
        record for analysis in analyses.values() for record in analysis["call_sites"]
    ]
    signatures: dict[str, dict[str, Any]] = {}
    for record in records:
        signature_id = _analysis_signature(record)
        entry = signatures.setdefault(
            signature_id,
            {
                "signature_id": signature_id,
                "representative": record["call_site"],
                "call_sites": {"initial": [], "steady": []},
                "focused": False,
                "eligible": True,
            },
        )
        kind = (
            "initial" if record["call_site"].startswith("int8/initial/") else "steady"
        )
        entry["call_sites"][kind].append(record["call_site"])
        entry["focused"] = bool(entry["focused"] or record["focused"])
        entry["eligible"] = bool(
            entry["eligible"]
            and record["eligible_input"]
            and record["eligible_cache_update"]
            and record["eligible_epilogue"]
        )
    errors = [
        error for analysis in analyses.values() for error in analysis.get("errors", [])
    ]
    return {
        "schema_version": FUSION_SCHEMA_VERSION,
        "analysis_schema_version": FUSION_ANALYSIS_SCHEMA_VERSION,
        "analysis_contract_sha256": analysis_contracts["contract_sha256"],
        "int8_audit_sha256": analysis_contracts["int8_audit_sha256"],
        "focus_module_prefix": focus_module_prefix,
        "passed": not errors,
        "errors": errors,
        "graphs": analyses,
        "signatures": signatures,
    }


def _find_record(analysis: Mapping[str, Any], call_site: str) -> dict[str, Any]:
    for graph in analysis["graphs"].values():
        for record in graph["call_sites"]:
            if record["call_site"] == call_site:
                return record
    raise KeyError(call_site)


def _probe_inputs(record: Mapping[str, Any], *, full_boundary: bool) -> list[str]:
    values = [record["current_tensor"]]
    cache = record.get("cache_tensor")
    if isinstance(cache, str) and cache:
        values.append(cache)
    if full_boundary and record.get("epilogue_mode") == "conv2_residual":
        values.append(record["epilogue_residual_tensor"])
    return list(dict.fromkeys(values))


def _normalize_probe_shape(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise _ProbeInfrastructureError(f"{label} is not a static tensor shape")
    shape = [int(dimension) for dimension in value]
    if any(dimension <= 0 for dimension in shape):
        raise _ProbeInfrastructureError(
            f"{label} contains a dynamic or non-positive dimension: {shape}"
        )
    return shape


def _probe_input_shapes(
    record: Mapping[str, Any], *, full_boundary: bool
) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}

    def _add(name: Any, shape: Any, *, label: str) -> None:
        if not isinstance(name, str) or not name:
            raise _ProbeInfrastructureError(f"{label} has no tensor name")
        normalized = _normalize_probe_shape(shape, label=label)
        existing = shapes.get(name)
        if existing is not None and existing != normalized:
            raise _ProbeInfrastructureError(
                f"probe input {name} has conflicting static shapes: "
                f"{existing} versus {normalized}"
            )
        shapes[name] = normalized

    _add(
        record.get("current_tensor"),
        record.get("current_shape"),
        label="current activation",
    )
    cache = record.get("cache_tensor")
    if isinstance(cache, str) and cache:
        _add(cache, record.get("cache_shape"), label="feature cache")
    if full_boundary and record.get("epilogue_mode") == "conv2_residual":
        _add(
            record.get("epilogue_residual_tensor"),
            record.get("epilogue_output_shape"),
            label="epilogue residual",
        )
    expected = set(_probe_inputs(record, full_boundary=full_boundary))
    if set(shapes) != expected:
        raise _ProbeInfrastructureError(
            "probe input-shape contract does not match the extracted inputs: "
            f"inputs={sorted(expected)}, shapes={sorted(shapes)}"
        )
    return shapes


def _specialize_probe_inputs(
    *, onnx: Any, model: Any, input_shapes: Mapping[str, list[int]]
) -> None:
    graph_inputs = {value.name: value for value in model.graph.input}
    if set(graph_inputs) != set(input_shapes):
        raise _ProbeInfrastructureError(
            "extracted probe inputs differ from the static analysis contract: "
            f"onnx={sorted(graph_inputs)}, analysis={sorted(input_shapes)}"
        )
    for name, shape in input_shapes.items():
        tensor_type = graph_inputs[name].type.tensor_type
        dimensions = tensor_type.shape.dim
        if len(dimensions) != len(shape):
            raise _ProbeInfrastructureError(
                f"probe input {name} rank differs from static analysis: "
                f"onnx={len(dimensions)}, analysis={len(shape)}"
            )
        for dimension, size in zip(dimensions, shape, strict=True):
            dimension.ClearField("dim_param")
            dimension.dim_value = int(size)
        if any(
            not dimension.HasField("dim_value") or int(dimension.dim_value) <= 0
            for dimension in dimensions
        ):
            raise _ProbeInfrastructureError(
                f"probe input {name} remained dynamic after specialization"
            )
    onnx.checker.check_model(model)


def _extract_probe(
    *,
    onnx: Any,
    source: Path,
    destination: Path,
    inputs: list[str],
    outputs: list[str],
    input_shapes: Mapping[str, list[int]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.stem}.partial.onnx")
    onnx.utils.extract_model(
        str(source),
        str(temporary),
        inputs,
        outputs,
        check_model=True,
    )
    model = onnx.load(str(temporary))
    _specialize_probe_inputs(
        onnx=onnx,
        model=model,
        input_shapes=input_shapes,
    )
    onnx.save(model, str(temporary))
    temporary.replace(destination)


def _inspector_layers(inspector_json: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(inspector_json)
    except json.JSONDecodeError:
        return []
    if isinstance(value, list):
        return [record for record in value if isinstance(record, dict)]
    if isinstance(value, dict):
        for key in ("Layers", "layers"):
            records = value.get(key)
            if isinstance(records, list):
                return [record for record in records if isinstance(record, dict)]
        return [value]
    return []


def _layer_text(record: Mapping[str, Any]) -> str:
    return " ".join(
        str(record.get(key, ""))
        for key in ("Name", "name", "LayerType", "layer_type", "Metadata", "metadata")
    )


def _pack_to_conv_has_reformat(*, inspector_json: str, call_site: str) -> bool:
    layers = _inspector_layers(inspector_json)
    pack_indices = [
        index
        for index, record in enumerate(layers)
        if (
            f"fusion/input_pack_quant/{call_site}" in _layer_text(record)
            or f"fusion/input_pack_quant_cache/{call_site}" in _layer_text(record)
            or (
                PACK_QUANT_PLUGIN in _layer_text(record)
                and call_site in _layer_text(record)
            )
        )
    ]
    conv_indices = [
        index
        for index, record in enumerate(layers)
        if call_site in _layer_text(record)
        and "convolution" in _layer_text(record).lower()
    ]
    if len(pack_indices) != 1 or len(conv_indices) != 1:
        return True
    low, high = sorted((pack_indices[0], conv_indices[0]))
    return any(
        "reformat" in _layer_text(record).lower() for record in layers[low + 1 : high]
    )


def _conv_to_epilogue_has_reformat(*, inspector_json: str, call_site: str) -> bool:
    layers = _inspector_layers(inspector_json)
    conv_indices = [
        index
        for index, record in enumerate(layers)
        if call_site in _layer_text(record)
        and "convolution" in _layer_text(record).lower()
    ]
    epilogue_indices = [
        index
        for index, record in enumerate(layers)
        if (
            f"fusion/conv1_norm_silu/{call_site}" in _layer_text(record)
            or f"fusion/conv2_residual/{call_site}" in _layer_text(record)
            or (
                EPILOGUE_PLUGIN in _layer_text(record)
                and call_site in _layer_text(record)
            )
        )
    ]
    if len(conv_indices) != 1 or len(epilogue_indices) != 1:
        return True
    low, high = sorted((conv_indices[0], epilogue_indices[0]))
    return any(
        "reformat" in _layer_text(record).lower() for record in layers[low + 1 : high]
    )


def _run_plan_timing(
    *,
    trt: Any,
    torch: Any,
    plan: bytes,
    device_index: int,
    warmup: int,
    repeat: int,
    cache_validation: Mapping[str, Any],
) -> dict[str, Any]:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("could not deserialize fusion micro-probe")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("could not create fusion micro-probe context")
    tensors: dict[str, Any] = {}
    outputs: list[Any] = []
    cuda_device = torch.device(f"cuda:{device_index}")
    for index in range(int(engine.num_io_tensors)):
        name = engine.get_tensor_name(index)
        shape = tuple(int(value) for value in engine.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)
        if dtype == trt.float16:
            torch_dtype = torch.float16
        elif dtype == trt.int8:
            torch_dtype = torch.int8
        else:
            raise ValueError(f"probe binding {name} has unsupported dtype {dtype}")
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            if torch_dtype == torch.int8:
                generator = torch.Generator(device=cuda_device)
                generator.manual_seed(
                    int(hashlib.sha256(name.encode()).hexdigest()[:16], 16)
                )
                tensor = torch.randint(
                    -8,
                    9,
                    shape,
                    device=cuda_device,
                    dtype=torch_dtype,
                    generator=generator,
                )
            else:
                generator = torch.Generator(device=cuda_device)
                generator.manual_seed(
                    int(hashlib.sha256(name.encode()).hexdigest()[:16], 16)
                )
                tensor = torch.randn(
                    shape,
                    device=cuda_device,
                    dtype=torch_dtype,
                    generator=generator,
                )
        else:
            tensor = torch.empty(shape, device=cuda_device, dtype=torch_dtype)
            outputs.append(tensor)
        tensors[name] = tensor
        if not context.set_tensor_address(name, int(tensor.data_ptr())):
            raise RuntimeError(f"TensorRT rejected probe binding {name}")
    stream = torch.cuda.current_stream(device=device_index)
    for _ in range(warmup):
        if not context.execute_async_v3(stream_handle=int(stream.cuda_stream)):
            raise RuntimeError("fusion micro-probe warmup failed")
    torch.cuda.synchronize(device_index)
    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        if not context.execute_async_v3(stream_handle=int(stream.cuda_stream)):
            raise RuntimeError("fusion micro-probe execution failed")
        end.record(stream)
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    if not outputs or any(
        not bool(torch.isfinite(value.float()).all()) for value in outputs
    ):
        raise RuntimeError("fusion micro-probe produced missing or non-finite outputs")
    current_name = cache_validation.get("current_tensor")
    cache_name = cache_validation.get("cache_tensor")
    output_name = cache_validation.get("cache_update_tensor")
    if current_name not in tensors or output_name not in tensors:
        raise RuntimeError("fusion micro-probe cache validation bindings are missing")
    source = tensors[current_name]
    if isinstance(cache_name, str) and cache_name:
        if cache_name not in tensors:
            raise RuntimeError("fusion micro-probe cache input binding is missing")
        source = torch.cat((tensors[cache_name], source), dim=2)
    cache_output = tensors[output_name]
    expected_cache = source[:, :, -int(cache_output.shape[2]) :, :, :]
    cache_error = float(
        (cache_output.float() - expected_cache.float()).abs().max().item()
    )
    cache_matches = bool(cache_error == 0.0)
    if not cache_matches:
        raise RuntimeError(
            "fusion micro-probe changed the FP16 causal cache update: "
            f"max_abs_error={cache_error}"
        )
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "warmup": warmup,
        "repeat": repeat,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[p95_index],
        "population_stddev_ms": statistics.pstdev(samples),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": samples,
        "cache_update_matches": cache_matches,
        "cache_update_max_abs_error": cache_error,
    }


def _build_probe(
    *,
    trt: Any,
    torch: Any,
    onnx: Any,
    source_path: Path,
    graph_analysis: Mapping[str, Any],
    record: Mapping[str, Any],
    scheme: str,
    probe_root: Path,
    workspace_gib: float,
    device_index: int,
    warmup: int,
    repeat: int,
    weight_encoding: str,
) -> dict[str, Any]:
    signature = _analysis_signature(record)
    safe_scheme = scheme.replace("/", "_")
    probe_onnx = probe_root / f"{signature}_{safe_scheme}.onnx"
    rewritten_full = probe_root / f"{signature}_{safe_scheme}_full.onnx"
    fuse_input = scheme != "baseline"
    fuse_cache = "cache_" in scheme or scheme.startswith("full_boundary_")
    fuse_epilogue = scheme.startswith("full_boundary_")
    cache_update_mode = "dual" if scheme.endswith("_dual") else "separate"
    if fuse_input:
        rewrite_fusion_graph(
            source_path=source_path,
            destination_path=rewritten_full,
            analysis=graph_analysis,
            selected_call_sites={record["call_site"]},
            fuse_cache_update=fuse_cache,
            fuse_epilogue=fuse_epilogue,
            cache_update_mode=cache_update_mode,
        )
        extraction_source = rewritten_full
    else:
        extraction_source = source_path
    # Every scheme exposes the same final epilogue and cache-update outputs.
    # Otherwise a faster candidate could simply be doing less work and the
    # threshold would be scientifically meaningless.
    inputs = _probe_inputs(record, full_boundary=True)
    input_shapes = _probe_input_shapes(record, full_boundary=True)
    outputs = [
        record["epilogue_output_tensor"],
        record["cache_update_tensor"],
    ]
    _extract_probe(
        onnx=onnx,
        source=extraction_source,
        destination=probe_onnx,
        inputs=inputs,
        outputs=list(dict.fromkeys(outputs)),
        input_shapes=input_shapes,
    )
    plan, io_contract, inspector_json, _candidate_cache = _build_engine_bytes(
        trt=trt,
        onnx_path=probe_onnx,
        workspace_gib=workspace_gib,
        profiling_verbosity="detailed",
        timing_cache_path=None,
    )
    tactic = _audit_tensorrt_tactics(
        graph_kind=f"probe_{signature}_{scheme}",
        call_site_names=[record["call_site"]],
        inspector_json=inspector_json,
        expected_call_sites=1,
        weight_encoding=weight_encoding,
    )
    reformat_between = (
        _pack_to_conv_has_reformat(
            inspector_json=inspector_json,
            call_site=record["call_site"],
        )
        if fuse_input
        else False
    )
    output_reformat_between = (
        _conv_to_epilogue_has_reformat(
            inspector_json=inspector_json,
            call_site=record["call_site"],
        )
        if fuse_epilogue
        else False
    )
    timing = _run_plan_timing(
        trt=trt,
        torch=torch,
        plan=plan,
        device_index=device_index,
        warmup=warmup,
        repeat=repeat,
        cache_validation=record,
    )
    return {
        "scheme": scheme,
        "source_onnx": str(probe_onnx),
        "source_sha256": sha256_file(probe_onnx),
        "plan_sha256": hashlib.sha256(plan).hexdigest(),
        "io_tensors": io_contract,
        "tactic": tactic,
        "reformat_between_pack_and_conv": reformat_between,
        "reformat_between_conv_and_epilogue": output_reformat_between,
        "timing": timing,
        "passed_correctness": bool(
            tactic["passed"]
            and not reformat_between
            and not output_reformat_between
            and timing["cache_update_matches"]
        ),
    }


def _run_probe_suite(
    *,
    trt: Any,
    torch: Any,
    onnx: Any,
    source_paths: Mapping[str, Path],
    analysis: Mapping[str, Any],
    fusion_root: Path,
    workspace_gib: float,
    device_index: int,
    warmup: int,
    repeat: int,
    weight_encoding: str,
) -> dict[str, Any]:
    probe_root = fusion_root / _PROBE_SUBDIRECTORY
    probe_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    errors: list[str] = []
    for signature_id, signature_record in analysis["signatures"].items():
        representative = _find_record(analysis, signature_record["representative"])
        kind = (
            "initial"
            if representative["call_site"].startswith("int8/initial/")
            else "steady"
        )
        if not signature_record["eligible"]:
            result = {
                **signature_record,
                "passed": False,
                "errors": ["signature did not pass static fusion analysis"],
                "schemes": {},
            }
            results[signature_id] = result
            if signature_record["focused"]:
                errors.append(f"required signature {signature_id} is ineligible")
            continue
        schemes: dict[str, Any] = {}
        try:
            for scheme in _SCHEMES:
                try:
                    schemes[scheme] = _build_probe(
                        trt=trt,
                        torch=torch,
                        onnx=onnx,
                        source_path=source_paths[kind],
                        graph_analysis=analysis["graphs"][kind],
                        record=representative,
                        scheme=scheme,
                        probe_root=probe_root,
                        workspace_gib=workspace_gib,
                        device_index=device_index,
                        warmup=warmup,
                        repeat=repeat,
                        weight_encoding=weight_encoding,
                    )
                except _ProbeInfrastructureError:
                    raise
                except BaseException as exc:
                    if scheme == "baseline":
                        raise _ProbeInfrastructureError(
                            "the unfused baseline probe could not be built or run; "
                            "trying other fusion schemes cannot repair this failure: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    schemes[scheme] = {
                        "scheme": scheme,
                        "passed_correctness": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            baseline = float(schemes["baseline"]["timing"]["mean_ms"])
            input_ms = float(schemes["input_pack_quant"]["timing"]["mean_ms"])
            input_gain = (baseline - input_ms) / baseline
            base_correctness = all(
                schemes[key].get("passed_correctness") is True
                for key in ("baseline", "input_pack_quant")
            )
            candidates: list[dict[str, Any]] = []
            for mode in ("dual", "split"):
                cache_key = f"input_pack_quant_cache_{mode}"
                full_key = f"full_boundary_{mode}"
                cache_record = schemes[cache_key]
                full_record = schemes[full_key]
                correctness = bool(
                    base_correctness
                    and cache_record.get("passed_correctness") is True
                    and full_record.get("passed_correctness") is True
                )
                if correctness:
                    cache_ms = float(cache_record["timing"]["mean_ms"])
                    full_ms = float(full_record["timing"]["mean_ms"])
                    cache_gain = (baseline - cache_ms) / baseline
                    epilogue_gain = (cache_ms - full_ms) / cache_ms
                else:
                    cache_ms = None
                    full_ms = None
                    cache_gain = None
                    epilogue_gain = None
                passed_thresholds = bool(
                    correctness
                    and input_gain >= 0.10
                    and cache_gain is not None
                    and cache_gain >= 0.10
                    and epilogue_gain is not None
                    and epilogue_gain >= 0.05
                )
                candidates.append(
                    {
                        "mode": mode,
                        "correctness": correctness,
                        "cache_ms": cache_ms,
                        "full_ms": full_ms,
                        "input_gain_fraction": input_gain,
                        "cache_gain_fraction": cache_gain,
                        "epilogue_incremental_gain_fraction": epilogue_gain,
                        "passed_thresholds": passed_thresholds,
                    }
                )
            passing = [
                value
                for value in candidates
                if value["passed_thresholds"] is True and value["full_ms"] is not None
            ]
            if not passing:
                raise RuntimeError(
                    "neither mixed-output nor split cache-update fusion passed "
                    f"correctness and performance thresholds: {candidates}"
                )
            selected = min(passing, key=lambda value: float(value["full_ms"]))
            selected_mode = str(selected["mode"])
            result = {
                **signature_record,
                "passed": True,
                "errors": [],
                "input_gain_fraction": input_gain,
                "cache_gain_fraction": selected["cache_gain_fraction"],
                "epilogue_incremental_gain_fraction": selected[
                    "epilogue_incremental_gain_fraction"
                ],
                "selected_cache_update_mode": selected_mode,
                "selected_cache_scheme": f"input_pack_quant_cache_{selected_mode}",
                "selected_full_scheme": f"full_boundary_{selected_mode}",
                "selection_candidates": candidates,
                "schemes": schemes,
            }
        except _ProbeInfrastructureError:
            raise
        except BaseException as exc:
            result = {
                **signature_record,
                "passed": False,
                "errors": [f"{type(exc).__name__}: {exc}"],
                "schemes": schemes,
            }
        results[signature_id] = result
        if signature_record["focused"] and not result["passed"]:
            errors.append(f"required signature {signature_id} failed micro-probe")
    return {
        "schema_version": FUSION_SCHEMA_VERSION,
        "variant": FUSION_VARIANT,
        "passed": not errors,
        "errors": errors,
        "warmup": warmup,
        "repeat": repeat,
        "signature_count": len(results),
        "required_signature_count": sum(
            bool(value["focused"]) for value in results.values()
        ),
        "signatures": results,
    }


def _selected_call_sites(
    *, analysis: Mapping[str, Any], probe: Mapping[str, Any]
) -> tuple[dict[str, set[str]], dict[str, dict[str, str]]]:
    result = {kind: set() for kind in _KINDS}
    modes: dict[str, dict[str, str]] = {kind: {} for kind in _KINDS}
    for signature_id, signature in probe["signatures"].items():
        if signature.get("passed") is not True:
            continue
        mode = signature.get("selected_cache_update_mode")
        if mode not in {"dual", "split"}:
            raise RuntimeError(
                f"passing signature {signature_id} has no cache-update mode"
            )
        for kind in _KINDS:
            call_sites = signature["call_sites"].get(kind, [])
            result[kind].update(call_sites)
            normalized_mode = "dual" if mode == "dual" else "separate"
            for call_site in call_sites:
                modes[kind][call_site] = normalized_mode
    focused = {
        kind: {
            record["call_site"]
            for record in analysis["graphs"][kind]["call_sites"]
            if record["focused"]
        }
        for kind in _KINDS
    }
    for kind in _KINDS:
        if not focused[kind] or not focused[kind] <= result[kind]:
            raise RuntimeError(f"probe did not approve every focused {kind} call site")
    return result, modes


def _expected_io_from_manifest(
    *, manifest: Mapping[str, Any], kind: str
) -> dict[str, tuple[str, list[int], str]]:
    expected = {
        "latent": ("input", list(manifest["latent_shape"]), "float16"),
        "rgb": (
            "output",
            [
                1,
                3,
                9 if kind == "initial" else 12,
                manifest["height"],
                manifest["width"],
            ],
            "float16",
        ),
    }
    for binding in manifest["cache"]["bindings"]:
        index = int(binding["index"])
        if kind == "steady":
            expected[f"cache_in_{index:03d}"] = (
                "input",
                list(binding["shape"]),
                "float16",
            )
        expected[f"cache_out_{index:03d}"] = (
            "output",
            list(binding["shape"]),
            "float16",
        )
    return expected


def _validate_io(
    *, records: list[dict[str, Any]], expected: Mapping[str, tuple[str, list[int], str]]
) -> None:
    by_name = {record.get("name"): record for record in records}
    if set(by_name) != set(expected):
        raise ValueError(
            f"fusion engine I/O mismatch: missing={sorted(set(expected) - set(by_name))}, "
            f"extra={sorted(set(by_name) - set(expected))}"
        )
    for name, (mode, shape, dtype) in expected.items():
        record = by_name[name]
        if (
            str(record.get("mode", "")).lower() != mode
            or record.get("shape") != shape
            or str(record.get("dtype", "")).lower()
            not in {dtype, "half" if dtype == "float16" else dtype}
        ):
            raise ValueError(
                f"fusion binding {name} differs from the base ABI: {record}"
            )


def _plugin_counts(inspector_json: str) -> dict[str, int]:
    texts = [_layer_text(record) for record in _inspector_layers(inspector_json)]
    dual_cache = sum("fusion/input_pack_quant_cache/" in text for text in texts)
    separate_cache = sum(
        CACHE_UPDATE_PLUGIN in text or "fusion/cache_update/" in text for text in texts
    )
    return {
        "input_pack_quant": sum(
            PACK_QUANT_PLUGIN in text
            or "fusion/input_pack_quant/" in text
            or "fusion/input_pack_quant_cache/" in text
            for text in texts
        ),
        "cache_update": dual_cache + separate_cache,
        "cache_update_dual": dual_cache,
        "cache_update_separate": separate_cache,
        "conv1_norm_silu": sum(
            (EPILOGUE_PLUGIN in text and "conv1_norm_silu" in text)
            or "fusion/conv1_norm_silu/" in text
            for text in texts
        ),
        "conv2_residual": sum(
            (EPILOGUE_PLUGIN in text and "conv2_residual" in text)
            or "fusion/conv2_residual/" in text
            for text in texts
        ),
    }


def _build_full(
    *,
    trt: Any,
    base_manifest: Mapping[str, Any],
    source_paths: Mapping[str, Path],
    analysis: Mapping[str, Any],
    probe: Mapping[str, Any],
    fusion_root: Path,
    workspace_gib: float,
    stable_timing_cache: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    quantization = base_manifest.get("quantization")
    weight_encoding = (
        quantization.get("weight_encoding")
        if isinstance(quantization, Mapping)
        else None
    )
    if not isinstance(weight_encoding, str) or not weight_encoding:
        raise ValueError("base INT8 manifest has no weight encoding")
    selected, cache_modes = _selected_call_sites(analysis=analysis, probe=probe)
    graph_records: dict[str, Any] = {}
    for kind in _KINDS:
        destination = fusion_root / FUSION_ONNX_FILES[kind]
        graph_records[kind] = rewrite_fusion_graph(
            source_path=source_paths[kind],
            destination_path=destination,
            analysis=analysis["graphs"][kind],
            selected_call_sites=selected[kind],
            fuse_cache_update=True,
            fuse_epilogue=True,
            cache_update_modes=cache_modes[kind],
        )

    candidate_cache = fusion_root / f"{FUSION_TIMING_CACHE_FILE}.candidate"
    candidate_cache.unlink(missing_ok=True)
    if stable_timing_cache.is_file():
        _write_bytes(candidate_cache, stable_timing_cache.read_bytes())
    engines: dict[str, Any] = {}
    tactic_audits: dict[str, Any] = {}
    errors: list[str] = []
    for kind in _KINDS:
        onnx_path = fusion_root / FUSION_ONNX_FILES[kind]
        plan, io_contract, inspector_json, candidate_bytes = _build_engine_bytes(
            trt=trt,
            onnx_path=onnx_path,
            workspace_gib=workspace_gib,
            profiling_verbosity="detailed",
            timing_cache_path=candidate_cache,
        )
        plan_path = fusion_root / FUSION_ENGINE_FILES[kind]
        inspector_path = fusion_root / FUSION_INSPECTOR_FILES[kind]
        expected_io = _expected_io_from_manifest(manifest=base_manifest, kind=kind)
        try:
            _validate_io(records=io_contract, expected=expected_io)
        except ValueError as exc:
            raise FusionAuditError(f"fusion {kind} I/O audit failed") from exc
        try:
            cache_audit = _audit_feature_cache_kind_io(kind=kind, records=io_contract)
        except (KeyError, TypeError, ValueError) as exc:
            raise FusionAuditError(f"fusion {kind} feature-cache audit failed") from exc
        if cache_audit["passed"] is not True or cache_audit["quantized_bindings"] != []:
            raise FusionAuditError(f"fusion {kind} changed the FP16 cache ABI")
        all_call_sites = [
            record["call_site"] for record in analysis["graphs"][kind]["call_sites"]
        ]
        try:
            tactic = _audit_tensorrt_tactics(
                graph_kind=kind,
                call_site_names=all_call_sites,
                inspector_json=inspector_json,
                expected_call_sites=EXPECTED_CALL_SITES,
                weight_encoding=weight_encoding,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FusionAuditError(
                f"fusion {kind} tactic audit could not be completed"
            ) from exc
        counts = _plugin_counts(inspector_json)
        required_count = len(selected[kind])
        input_reformats = sorted(
            call_site
            for call_site in selected[kind]
            if _pack_to_conv_has_reformat(
                inspector_json=inspector_json,
                call_site=call_site,
            )
        )
        output_reformats = sorted(
            call_site
            for call_site in selected[kind]
            if _conv_to_epilogue_has_reformat(
                inspector_json=inspector_json,
                call_site=call_site,
            )
        )
        if (
            tactic["passed"] is not True
            or counts["input_pack_quant"] != required_count
            or counts["cache_update"] != required_count
            or counts["conv1_norm_silu"] + counts["conv2_residual"] != required_count
            or input_reformats
            or output_reformats
        ):
            errors.append(f"{kind} tactic/plugin audit failed")
        plan_sha = hashlib.sha256(plan).hexdigest()
        tactic.update(
            {
                "plan_sha256": plan_sha,
                "target_conv_call_site_count": EXPECTED_CALL_SITES,
                "int8_target_conv_call_site_count": (
                    EXPECTED_CALL_SITES
                    if tactic["passed"]
                    else tactic["mapped_count"] - len(tactic["non_int8_call_sites"])
                ),
                "fp16_or_tf32_fallback_call_sites": sorted(
                    set(tactic["fp16_fallback_call_sites"])
                    | set(tactic["fp32_or_tf32_fallback_call_sites"])
                ),
                "plugin_counts": counts,
                "selected_cache_update_modes": dict(cache_modes[kind]),
                "input_reformat_call_sites": input_reformats,
                "output_reformat_call_sites": output_reformats,
                "cache": cache_audit,
            }
        )
        tactic_audits[kind] = tactic
        _write_bytes(plan_path, plan)
        try:
            inspector = json.loads(inspector_json)
        except json.JSONDecodeError as exc:
            raise FusionAuditError(
                "TensorRT returned invalid fusion inspector JSON"
            ) from exc
        write_json_atomic(inspector_path, inspector)
        if candidate_bytes is not None:
            _write_bytes(candidate_cache, candidate_bytes)
        engines[kind] = {
            "file": _relative(fusion_root, plan_path),
            "sha256": plan_sha,
            "source_onnx_file": _relative(fusion_root, onnx_path),
            "source_onnx_sha256": sha256_file(onnx_path),
            "inspector": {
                "file": _relative(fusion_root, inspector_path),
                "sha256": sha256_file(inspector_path),
            },
            "profiling_verbosity": "detailed",
            "rgb_shape": [
                1,
                3,
                9 if kind == "initial" else 12,
                base_manifest["height"],
                base_manifest["width"],
            ],
            "io_tensors": io_contract,
            "selected_fusion_call_sites": sorted(selected[kind]),
            "selected_cache_update_modes": dict(cache_modes[kind]),
            "fusion_counts": counts,
        }
    if errors:
        raise FusionAuditError(f"fusion full-plan audit failed: {errors}")
    if candidate_cache.is_file():
        candidate_cache.replace(stable_timing_cache)
    total_counts = {
        key: sum(engines[kind]["fusion_counts"][key] for kind in _KINDS)
        for key in (
            "input_pack_quant",
            "cache_update",
            "cache_update_dual",
            "cache_update_separate",
            "conv1_norm_silu",
            "conv2_residual",
        )
    }
    audit = {
        "schema_version": FUSION_AUDIT_SCHEMA_VERSION,
        "variant": FUSION_VARIANT,
        "qdq_schema_version": QDQ_SCHEMA_VERSION,
        "weight_encoding": weight_encoding,
        "passed": True,
        "complete": True,
        "errors": [],
        "plugin_sha256": None,
        "feature_cache_dtype": "float16",
        "feature_cache_tensor_count": TRT_VAE_CACHE_COUNT,
        "fusion_counts": total_counts,
        # Keep the v5 tactic-audit shape as a compatibility surface for the
        # existing physical-layer profiler.  These records describe the
        # newly built fusion plans, not the baseline v5 plans.
        "plan_sha256": {kind: engines[kind]["sha256"] for kind in _KINDS},
        "tactics": tactic_audits,
        "engines": tactic_audits,
    }
    return {
        "engines": engines,
        "fusion_counts": total_counts,
        "graphs": graph_records,
    }, audit


def build_fusion(
    *,
    engine_dir: str | Path,
    stage: str = "all",
    focus_module_prefix: str = "decoder.up_blocks.3",
    preflight_only: bool = False,
    resume: bool = False,
    workspace_gib: float = 8.0,
    probe_warmup: int = 20,
    probe_repeat: int = 100,
    device_index: int = 0,
    plugin_library: str | Path | None = None,
) -> dict[str, Any]:
    if stage not in {"analyze", "probe", "build", "all"}:
        raise ValueError("--stage must be analyze, probe, build, or all")
    if workspace_gib <= 0:
        raise ValueError("--workspace-gib must be positive")
    if probe_warmup < 0 or probe_repeat <= 0:
        raise ValueError("probe warmup/repeat are invalid")
    if focus_module_prefix != "decoder.up_blocks.3":
        raise ValueError(
            "fusion_v1 currently requires --focus-module-prefix decoder.up_blocks.3"
        )

    import onnx
    import tensorrt as trt
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("fusion-v1 build requires CUDA")
    if device_index < 0 or device_index >= int(torch.cuda.device_count()):
        raise ValueError(f"invalid CUDA device index: {device_index}")
    torch.cuda.set_device(device_index)

    base_root = Path(engine_dir).expanduser().resolve()
    base_manifest_path = base_root / "manifest.json"
    base_manifest = load_trt_vae_manifest(base_root)
    model_id = base_manifest.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("base TensorRT manifest has no model_id")
    validate_trt_vae_manifest(
        base_manifest,
        engine_dir=base_root,
        precision="int8",
        model_path=model_id,
        verify_plan_hashes=True,
    )
    analysis_contracts = _load_analysis_contracts(
        base_root=base_root,
        base_manifest=base_manifest,
    )
    source_paths = {kind: base_root / _SOURCE_FILES[kind] for kind in _KINDS}
    for kind, path in source_paths.items():
        if not path.is_file():
            raise ValueError(f"required Q/DQ v5 {kind} ONNX is missing: {path}")

    fusion_root = base_root / FUSION_SUBDIRECTORY
    fusion_root.mkdir(parents=True, exist_ok=True)
    if plugin_library is None:
        plugin_path = fusion_root / FUSION_PLUGIN_LIBRARY_FILE
    else:
        plugin_path = Path(plugin_library).expanduser().resolve()
    if not plugin_path.is_file():
        raise ValueError(f"compiled fusion plugin does not exist: {plugin_path}")
    environment = _environment(
        trt=trt,
        torch=torch,
        onnx=onnx,
        device_index=device_index,
    )
    if environment["tensorrt_version"] != str(
        base_manifest["build"]["tensorrt_version"]
    ):
        raise ValueError("fusion builder TensorRT version differs from the base plan")
    _load_plugin(trt=trt, plugin_path=plugin_path)
    plugin_manifest = _plugin_manifest(
        fusion_root=fusion_root,
        plugin_path=plugin_path,
        environment=environment,
    )
    identity = _identity(
        base_root=base_root,
        base_manifest_path=base_manifest_path,
        source_paths=source_paths,
        plugin_path=plugin_path,
        environment=environment,
        focus_module_prefix=focus_module_prefix,
        workspace_gib=workspace_gib,
        probe_warmup=probe_warmup,
        probe_repeat=probe_repeat,
        int8_audit_sha256=analysis_contracts["int8_audit_sha256"],
        analysis_contract_sha256=analysis_contracts["contract_sha256"],
    )
    state_path = fusion_root / FUSION_BUILD_STATE_FILE
    state = (
        _load_json(state_path, label="fusion build state")
        if resume and state_path.is_file()
        else {}
    )
    if state and state.get("identity") != identity:
        raise ValueError("fusion build state belongs to a different build identity")
    if not state:
        state = {
            "schema_version": FUSION_SCHEMA_VERSION,
            "identity": identity,
            "stages": {},
        }
        write_json_atomic(state_path, state)
    # Do not mutate an existing experiment before its SHA-bound identity has
    # been accepted.  In particular, a mismatched --resume must not replace
    # the plugin manifest referenced by a previously valid fusion plan.
    write_json_atomic(fusion_root / FUSION_PLUGIN_MANIFEST_FILE, plugin_manifest)

    target_names = _target_names(base_manifest)
    analysis_path = fusion_root / _ANALYSIS_FILE

    def _validate_analysis_provenance(value: Mapping[str, Any]) -> None:
        if value.get("identity") != identity:
            raise ValueError("saved fusion analysis identity differs")
        if (
            value.get("analysis_schema_version")
            != FUSION_ANALYSIS_SCHEMA_VERSION
            or value.get("analysis_contract_sha256")
            != analysis_contracts["contract_sha256"]
            or value.get("int8_audit_sha256")
            != analysis_contracts["int8_audit_sha256"]
        ):
            raise ValueError("saved fusion analysis provenance is invalid")
        graphs = value.get("graphs")
        if not isinstance(graphs, Mapping) or set(graphs) != set(_KINDS):
            raise ValueError("saved fusion analysis graph records are incomplete")
        for kind in _KINDS:
            graph = graphs[kind]
            if (
                not isinstance(graph, Mapping)
                or graph.get("analysis_schema_version")
                != FUSION_ANALYSIS_SCHEMA_VERSION
            ):
                raise ValueError(
                    f"saved fusion {kind} graph analysis schema is invalid"
                )

    analysis = None
    if (
        resume
        and state.get("stages", {}).get("analyze", {}).get("status") == "completed"
    ):
        analysis = _load_json(analysis_path, label="fusion analysis")
        _validate_analysis_provenance(analysis)
    if analysis is None and stage in {"analyze", "probe", "all"}:
        _state_stage(
            state=state, state_path=state_path, name="analyze", status="running"
        )
        try:
            analysis = _run_analysis(
                source_paths=source_paths,
                target_names=target_names,
                focus_module_prefix=focus_module_prefix,
                analysis_contracts=analysis_contracts,
            )
            analysis["identity"] = identity
            _validate_analysis_provenance(analysis)
            write_json_atomic(analysis_path, analysis)
            status = "completed" if analysis["passed"] else "analysis_failed"
            _state_stage(
                state=state,
                state_path=state_path,
                name="analyze",
                status=status,
                artifact=analysis_path.name,
                artifact_sha256=sha256_file(analysis_path),
                errors=analysis["errors"],
            )
            if not analysis["passed"]:
                raise RuntimeError(
                    f"fusion graph analysis failed: {analysis['errors']}"
                )
        except BaseException as exc:
            if state["stages"].get("analyze", {}).get("status") == "running":
                _state_stage(
                    state=state,
                    state_path=state_path,
                    name="analyze",
                    status="analysis_failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise
    if stage == "analyze":
        if analysis is None:
            analysis = _load_json(analysis_path, label="fusion analysis")
        return {"stage": "analyze", "analysis": analysis, "state": str(state_path)}
    if analysis is None:
        analysis = _load_json(analysis_path, label="fusion analysis")
    _validate_analysis_provenance(analysis)
    if analysis.get("passed") is not True:
        raise RuntimeError("fusion analysis did not pass")

    probe_path = fusion_root / FUSION_PROBE_FILE
    probe = None
    if resume and state.get("stages", {}).get("probe", {}).get("status") == "completed":
        probe = _load_json(probe_path, label="fusion probe report")
        if probe.get("identity") != identity or probe.get("passed") is not True:
            raise ValueError("saved fusion probe is not reusable")
    if probe is None and stage in {"probe", "all"}:
        _state_stage(state=state, state_path=state_path, name="probe", status="running")
        try:
            probe = _run_probe_suite(
                trt=trt,
                torch=torch,
                onnx=onnx,
                source_paths=source_paths,
                analysis=analysis,
                fusion_root=fusion_root,
                workspace_gib=workspace_gib,
                device_index=device_index,
                warmup=probe_warmup,
                repeat=probe_repeat,
                weight_encoding=base_manifest["quantization"]["weight_encoding"],
            )
            probe["identity"] = identity
            write_json_atomic(probe_path, probe)
            status = "completed" if probe["passed"] else "probe_failed"
            _state_stage(
                state=state,
                state_path=state_path,
                name="probe",
                status=status,
                artifact=probe_path.name,
                artifact_sha256=sha256_file(probe_path),
                errors=probe["errors"],
            )
            if not probe["passed"]:
                raise RuntimeError(f"fusion micro-probe failed: {probe['errors']}")
        except BaseException as exc:
            if state["stages"].get("probe", {}).get("status") == "running":
                _state_stage(
                    state=state,
                    state_path=state_path,
                    name="probe",
                    status="probe_failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise
    if stage == "probe" or preflight_only:
        if probe is None:
            probe = _load_json(probe_path, label="fusion probe report")
        return {"stage": "probe", "probe": probe, "state": str(state_path)}
    if probe is None:
        probe = _load_json(probe_path, label="fusion probe report")
    if probe.get("identity") != identity:
        raise ValueError("saved fusion probe identity differs")
    if probe.get("passed") is not True:
        raise RuntimeError("fusion micro-probe did not pass")

    manifest_path = fusion_root / FUSION_MANIFEST_FILE
    if resume and state.get("stages", {}).get("build", {}).get("status") == "completed":
        manifest = _load_json(manifest_path, label="fusion manifest")
        try:
            validate_fusion_manifest(
                manifest,
                engine_dir=base_root,
                base_manifest=base_manifest,
                verify_hashes=True,
            )
        except ValueError as exc:
            raise FusionAuditError(
                "fusion manifest failed final artifact audit"
            ) from exc
        return manifest

    _state_stage(state=state, state_path=state_path, name="build", status="running")
    try:
        build_result, audit = _build_full(
            trt=trt,
            base_manifest=base_manifest,
            source_paths=source_paths,
            analysis=analysis,
            probe=probe,
            fusion_root=fusion_root,
            workspace_gib=workspace_gib,
            stable_timing_cache=fusion_root / FUSION_TIMING_CACHE_FILE,
        )
        audit["plugin_sha256"] = plugin_manifest["sha256"]
        audit_path = fusion_root / FUSION_AUDIT_FILE
        write_json_atomic(audit_path, audit)
        per_engine_counts = {
            kind: build_result["engines"][kind]["fusion_counts"] for kind in _KINDS
        }
        focused_complete = all(
            set(
                record["call_site"]
                for record in analysis["graphs"][kind]["call_sites"]
                if record["focused"]
            )
            <= set(build_result["engines"][kind]["selected_fusion_call_sites"])
            for kind in _KINDS
        )
        manifest = {
            "schema_version": FUSION_SCHEMA_VERSION,
            "variant": FUSION_VARIANT,
            "model_id": base_manifest["model_id"],
            "base": {
                "manifest_file": "../manifest.json",
                "manifest_sha256": sha256_file(base_manifest_path),
                "qdq_schema_version": QDQ_SCHEMA_VERSION,
            },
            "build": environment,
            "plugin": {
                "file": plugin_manifest["file"],
                "sha256": plugin_manifest["sha256"],
                "namespace": PLUGIN_NAMESPACE,
                "version": PLUGIN_VERSION,
                "creators": list(PLUGIN_CREATORS),
                "manifest_file": FUSION_PLUGIN_MANIFEST_FILE,
                "manifest_sha256": sha256_file(
                    fusion_root / FUSION_PLUGIN_MANIFEST_FILE
                ),
            },
            "cache": {
                "tensor_count": TRT_VAE_CACHE_COUNT,
                "dtype": "float16",
                "total_elements": TRT_VAE_CACHE_TOTAL_ELEMENTS,
                "single_bank_bytes": TRT_VAE_CACHE_BANK_BYTES,
                "double_bank_bytes": TRT_VAE_CACHE_BANK_BYTES * 2,
                "bindings": base_manifest["cache"]["bindings"],
            },
            "engines": build_result["engines"],
            "audit": {
                "file": FUSION_AUDIT_FILE,
                "sha256": sha256_file(audit_path),
                "passed": True,
                "schema_version": FUSION_AUDIT_SCHEMA_VERSION,
            },
            "probe": {
                "file": FUSION_PROBE_FILE,
                "sha256": sha256_file(probe_path),
                "passed": True,
            },
            "required_focus_prefix": focus_module_prefix,
            "required_focus_complete": focused_complete,
            "fusion_counts": build_result["fusion_counts"],
            "per_engine_fusion_counts": per_engine_counts,
            "build_state_file": FUSION_BUILD_STATE_FILE,
            "timing_cache_file": FUSION_TIMING_CACHE_FILE,
            "timing_cache_sha256": (
                sha256_file(fusion_root / FUSION_TIMING_CACHE_FILE)
                if (fusion_root / FUSION_TIMING_CACHE_FILE).is_file()
                else None
            ),
        }
        write_json_atomic(manifest_path, manifest)
        validate_fusion_manifest(
            manifest,
            engine_dir=base_root,
            base_manifest=base_manifest,
            verify_hashes=True,
        )
        _state_stage(
            state=state,
            state_path=state_path,
            name="build",
            status="completed",
            manifest=manifest_path.name,
            manifest_sha256=sha256_file(manifest_path),
            audit=audit_path.name,
            audit_sha256=sha256_file(audit_path),
        )
        # State changes after manifest creation; update the manifest only with
        # the stable filename, not its digest, so there is no recursive hash.
        return manifest
    except BaseException as exc:
        _state_stage(
            state=state,
            state_path=state_path,
            name="build",
            status=(
                "audit_failed" if isinstance(exc, FusionAuditError) else "build_failed"
            ),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument(
        "--stage",
        choices=("analyze", "probe", "build", "all"),
        default="all",
    )
    parser.add_argument(
        "--focus-module-prefix",
        default="decoder.up_blocks.3",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workspace-gib", type=float, default=8.0)
    parser.add_argument("--probe-warmup", type=int, default=20)
    parser.add_argument("--probe-repeat", type=int, default=100)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--plugin-library")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_fusion(
        engine_dir=args.engine_dir,
        stage=args.stage,
        focus_module_prefix=args.focus_module_prefix,
        preflight_only=args.preflight_only,
        resume=args.resume,
        workspace_gib=args.workspace_gib,
        probe_warmup=args.probe_warmup,
        probe_repeat=args.probe_repeat,
        device_index=args.device_index,
        plugin_library=args.plugin_library,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
