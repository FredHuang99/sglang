"""Native SM87 INT8 residual-block graph contract for the SFWan VAE.

This module is deliberately import-light.  Runtime manifest validation needs
only the standard library; ONNX, NumPy, and safetensors are imported only by
the Jetson build commands.

The native implementation replaces each pair of residual ``3x3x3`` Conv3d
calls with one TensorRT plugin node.  The plugin folds the temporal kernel into
the channel dimension and runs a CUTLASS SM80-family INT8 Conv2d implicit-GEMM
for the spatial convolution.  This is mathematically the same cross-correlation
for the fixed one-latent-frame decoder calls, but avoids both explicit spatial
padding and im2col materialization.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .vae_trt_fusion import (
    _lazy_onnx,
    _node_maps,
    _topological_live_nodes,
    sha256_file,
    write_json_atomic,
)
from .vae_trt_fusion_v2 import _find_norm1_contract

NATIVE_INT8_VARIANT = "native_int8_v1"
NATIVE_INT8_SCHEMA_VERSION = 1
NATIVE_INT8_ANALYSIS_SCHEMA_VERSION = 1
NATIVE_INT8_AUDIT_SCHEMA_VERSION = 1
NATIVE_INT8_SUBDIRECTORY = "native_int8_v1"

NATIVE_INT8_ANALYSIS_FILE = "native_int8_analysis.json"
NATIVE_INT8_SCALES_FILE = "native_int8_scales.json"
NATIVE_INT8_PACKED_WEIGHTS_FILE = "native_int8_packed_weights.safetensors"
NATIVE_INT8_WEIGHTS_MANIFEST_FILE = "native_int8_weights_manifest.json"
NATIVE_INT8_TUNE_FILE = "native_int8_tune.json"
NATIVE_INT8_BUILD_STATE_FILE = "native_int8_build_state.json"
NATIVE_INT8_AUDIT_FILE = "native_int8_audit.json"
NATIVE_INT8_MANIFEST_FILE = "native_int8_manifest.json"
NATIVE_INT8_TIMING_CACHE_FILE = "native_int8_timing.cache"

NATIVE_INT8_PLUGIN_LIBRARY_FILE = "libsfwan_vae_native_int8_v1.so"
NATIVE_INT8_PLUGIN_MANIFEST_FILE = "plugin_manifest.json"
NATIVE_INT8_PLUGIN_NAME = "SfWanNativeInt8ResidualBlockPlugin"
NATIVE_INT8_PLUGIN_NAMESPACE = "sglang.sfwan"
NATIVE_INT8_PLUGIN_VERSION = "1"
NATIVE_INT8_PLUGIN_INIT_SYMBOL = "initSfWanVaeNativeInt8V1Plugins"
NATIVE_INT8_PLUGIN_CREATORS = (NATIVE_INT8_PLUGIN_NAME,)

NATIVE_INT8_ONNX_FILES = {
    "initial": "initial_int8_native_v1.onnx",
    "steady": "steady_int8_native_v1.onnx",
}
NATIVE_INT8_PLAN_FILES = {
    "initial": "initial_int8_native_v1.plan",
    "steady": "steady_int8_native_v1.plan",
}
NATIVE_INT8_INSPECTOR_FILES = {
    "initial": "initial_int8_native_v1_inspector.json",
    "steady": "steady_int8_native_v1_inspector.json",
}

EXPECTED_LOGICAL_CONVS = 28
EXPECTED_RESIDUAL_BLOCKS = 14
EXPECTED_CALLS_PER_MODULE = 3
EXPECTED_CALL_SITES_PER_GRAPH = 84
EXPECTED_SIGNATURES = 9
EXPECTED_TOTAL_CACHE_SLOTS = 32
EXPECTED_INT8_CACHE_SLOTS = 28
EXPECTED_FP16_CACHE_SLOTS = 4

NATIVE_INT8_ALGORITHM = "temporal_folded_cutlass_conv2d_implicit_gemm"
NATIVE_INT8_WEIGHT_LAYOUT = "KRSTC_FLAT"
NATIVE_INT8_ACTIVATION_LAYOUT = "CDHW32"
NATIVE_INT8_ACCUMULATOR_DTYPE = "int32"
NATIVE_INT8_CUTLASS_COMMIT = "57e3cfb47a2d9e0d46eb6335c3dc411498efa198"

_CACHE_INPUT_RE = re.compile(r"^cache_in_(?P<index>\d{3})$")
_CACHE_OUTPUT_RE = re.compile(r"^cache_out_(?P<index>\d{3})$")


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _load_json_value(path: Path, *, label: str) -> Any:
    """Load JSON whose top-level shape is owned by an external producer."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc


def _resolve_inside(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} file is missing")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes {root}: {value!r}") from exc
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _base_manifest_identity(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _static_shape_map(model: Any) -> dict[str, list[int]]:
    values: dict[str, list[int]] = {}
    try:
        import onnx

        inferred = onnx.shape_inference.infer_shapes(
            model, check_type=True, strict_mode=False, data_prop=True
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        inferred = model
    for value in [
        *inferred.graph.input,
        *inferred.graph.output,
        *inferred.graph.value_info,
    ]:
        shape: list[int] = []
        for dimension in value.type.tensor_type.shape.dim:
            if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
                shape = []
                break
            shape.append(int(dimension.dim_value))
        if shape:
            values[value.name] = shape
    for initializer in model.graph.initializer:
        values.setdefault(
            initializer.name, [int(value) for value in initializer.dims]
        )
    return values


def _record_map(graph_analysis: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = graph_analysis.get("call_sites")
    if not isinstance(records, list):
        raise ValueError("base fusion analysis has no call-site list")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        call_site = record.get("call_site")
        if isinstance(call_site, str):
            result[call_site] = dict(record)
    return result


def _module_records(graph_analysis: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_module: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in _record_map(graph_analysis).values():
        by_module[str(record["module_name"])].append(record)
    for records in by_module.values():
        records.sort(key=lambda value: int(value["call_index"]))
    return dict(by_module)


def _conv_attributes(node: Any, helper: Any) -> dict[str, Any]:
    attributes = {value.name: helper.get_attribute_value(value) for value in node.attribute}

    def values(name: str, default: list[int]) -> list[int]:
        raw = attributes.get(name, default)
        return [int(value) for value in raw]

    return {
        "kernel_shape": values("kernel_shape", [3, 3, 3]),
        "pads": values("pads", [0, 0, 0, 0, 0, 0]),
        "strides": values("strides", [1, 1, 1]),
        "dilations": values("dilations", [1, 1, 1]),
        "groups": int(attributes.get("group", 1)),
    }


def _weight_contract(
    *,
    conv: Any,
    producer: Mapping[str, Any],
    initializers: Mapping[str, Any],
    numpy_helper: Any,
    helper: Any,
) -> dict[str, Any]:
    if len(conv.input) < 3:
        raise ValueError(f"target Conv has no weight/bias inputs: {conv.name}")
    weight_dq = producer.get(conv.input[1])
    if weight_dq is None or weight_dq.op_type != "DequantizeLinear":
        raise ValueError(f"target Conv weight is not DQ-backed: {conv.name}")
    scale_name = weight_dq.input[1]
    zero_name = weight_dq.input[2] if len(weight_dq.input) > 2 else None
    scale = initializers.get(scale_name)
    zero = initializers.get(zero_name) if zero_name else None
    if scale is None or zero is None:
        raise ValueError(f"target Conv weight scale/zero point is not constant: {conv.name}")
    scale_array = numpy_helper.to_array(scale)
    zero_array = numpy_helper.to_array(zero)
    if scale_array.ndim != 1 or not all(
        math.isfinite(float(value)) and float(value) > 0
        for value in scale_array.reshape(-1)
    ):
        raise ValueError(f"target Conv weight scale is invalid: {conv.name}")
    if zero_array.dtype.name != "int8" or any(
        int(value) != 0 for value in zero_array.reshape(-1)
    ):
        raise ValueError(f"target Conv zero point is not signed INT8 zero: {conv.name}")

    quantized_name = weight_dq.input[0]
    quantized = initializers.get(quantized_name)
    source_name = quantized_name
    source_kind = "prequantized_int8"
    if quantized is None:
        quantize = producer.get(quantized_name)
        if quantize is None or quantize.op_type != "QuantizeLinear":
            raise ValueError(f"target Conv weight source is not constant Q/DQ: {conv.name}")
        source_name = quantize.input[0]
        quantized = initializers.get(source_name)
        source_kind = "fp32_qdq"
    if quantized is None:
        raise ValueError(f"target Conv source weight is not an initializer: {conv.name}")
    source_shape = [int(value) for value in quantized.dims]
    if len(source_shape) != 5:
        raise ValueError(f"target Conv weight must be OITRS rank five: {conv.name}")
    output_channels = source_shape[0]
    if int(scale_array.size) != output_channels:
        raise ValueError(f"target Conv per-channel scale length changed: {conv.name}")

    bias_name = conv.input[2]
    bias = initializers.get(bias_name)
    if bias is None:
        bias_producer = producer.get(bias_name)
        if bias_producer is None or bias_producer.op_type not in {"Cast", "Identity"}:
            raise ValueError(f"target Conv bias is not constant: {conv.name}")
        bias_name = bias_producer.input[0]
        bias = initializers.get(bias_name)
    if bias is None or int(numpy_helper.to_array(bias).size) != output_channels:
        raise ValueError(f"target Conv bias shape changed: {conv.name}")

    axis = 0
    for attribute in weight_dq.attribute:
        if attribute.name == "axis":
            axis = int(helper.get_attribute_value(attribute))
    if axis != 0:
        raise ValueError(f"target Conv weight DQ axis must be zero: {conv.name}")
    return {
        "weight_dq_node": weight_dq.name,
        "source_weight_initializer": source_name,
        "source_weight_kind": source_kind,
        "source_weight_shape": source_shape,
        "weight_scale_initializer": scale_name,
        "weight_zero_initializer": zero_name,
        "bias_initializer": bias_name,
        "weight_scale_count": int(scale_array.size),
    }


def _block_prefix(module_name: str) -> str:
    if module_name.endswith(".conv1"):
        return module_name[: -len(".conv1")]
    if module_name.endswith(".conv2"):
        return module_name[: -len(".conv2")]
    raise ValueError(f"target module is not a residual Conv pair: {module_name}")


def _block_sort_key(prefix: str) -> tuple[Any, ...]:
    return tuple(
        int(piece) if piece.isdigit() else piece
        for piece in re.split(r"(\d+)", prefix)
    )


def analyze_native_int8_graphs(
    *,
    source_paths: Mapping[str, Path],
    base_analysis: Mapping[str, Any],
    signature_contracts: Mapping[str, Mapping[str, Mapping[str, Any]]],
    target_module_names: tuple[str, ...],
) -> dict[str, Any]:
    """Prove all residual blocks, cache owners, groups, and weight contracts."""

    onnx, np, _TensorProto, helpers = _lazy_onnx()
    helper, numpy_helper = helpers
    target_set = set(target_module_names)
    expected_modules = {
        f"{prefix}.{suffix}"
        for prefix in {_block_prefix(name) for name in target_module_names}
        for suffix in ("conv1", "conv2")
    }
    errors: list[str] = []
    if target_set != expected_modules or len(expected_modules) != EXPECTED_LOGICAL_CONVS:
        errors.append("target list does not form 14 exact conv1/conv2 residual pairs")

    graphs: dict[str, Any] = {}
    public_cache_slots_by_module: dict[str, int] = {}
    signature_fields = (
        "input_shape",
        "weight_shape",
        "output_shape",
        "kernel_shape",
        "pads",
        "strides",
        "dilations",
        "group",
    )
    signature_by_call_site: dict[str, str] = {}
    signatures: dict[str, dict[str, Any]] = {}
    for kind in ("initial", "steady"):
        contracts = signature_contracts.get(kind)
        if not isinstance(contracts, Mapping):
            errors.append(f"{kind}: v5 Conv signature contracts are missing")
            continue
        for call_site, contract in contracts.items():
            if not isinstance(call_site, str) or not isinstance(contract, Mapping):
                errors.append(f"{kind}: invalid v5 Conv signature contract")
                continue
            signature_id = contract.get("signature_id")
            if not isinstance(signature_id, str):
                errors.append(f"{kind}:{call_site}: v5 Conv signature ID is missing")
                continue
            canonical = {field: contract.get(field) for field in signature_fields}
            entry = signatures.setdefault(
                signature_id,
                {
                    "signature_id": signature_id,
                    "signature": canonical,
                    "call_sites": {"initial": [], "steady": []},
                },
            )
            if entry["signature"] != canonical:
                errors.append(f"v5 Conv signature collision: {signature_id}")
                continue
            signature_by_call_site[call_site] = signature_id
            entry["call_sites"][kind].append(call_site)
    signature_ids = set(signatures)
    for kind in ("initial", "steady"):
        graph_analysis = base_analysis.get("graphs", {}).get(kind)
        if not isinstance(graph_analysis, Mapping):
            errors.append(f"{kind}: base analysis is missing")
            continue
        module_records = _module_records(graph_analysis)
        model = onnx.load(str(source_paths[kind]), load_external_data=True)
        onnx.checker.check_model(model, full_check=True)
        shapes = _static_shape_map(model)
        producer, consumers = _node_maps(model)
        nodes_by_name = {node.name: node for node in model.graph.node}
        initializers = {value.name: value for value in model.graph.initializer}
        graph_outputs = {value.name for value in model.graph.output}
        blocks_by_call: dict[int, list[dict[str, Any]]] = {
            index: [] for index in range(EXPECTED_CALLS_PER_MODULE)
        }

        for prefix in sorted({_block_prefix(name) for name in target_module_names}, key=_block_sort_key):
            conv1_records = module_records.get(f"{prefix}.conv1", [])
            conv2_records = module_records.get(f"{prefix}.conv2", [])
            if len(conv1_records) != 3 or len(conv2_records) != 3:
                errors.append(f"{kind}:{prefix}: Conv pair is not unrolled three times")
                continue
            for call_index in range(EXPECTED_CALLS_PER_MODULE):
                conv1 = dict(conv1_records[call_index])
                conv2 = dict(conv2_records[call_index])
                try:
                    if not all(
                        record.get(flag) is True
                        for record in (conv1, conv2)
                        for flag in (
                            "eligible_input",
                            "eligible_cache_update",
                            "eligible_epilogue",
                        )
                    ):
                        raise ValueError("base boundary proof is incomplete")
                    norm1 = _find_norm1_contract(
                        record=conv1,
                        peer_conv2=conv2,
                        producer=producer,
                        consumers=consumers,
                        graph_outputs=graph_outputs,
                        initializers=initializers,
                        helper=helper,
                        numpy_helper=numpy_helper,
                        np=np,
                        shapes=shapes,
                    )
                    node1 = nodes_by_name[conv1["call_site"]]
                    node2 = nodes_by_name[conv2["call_site"]]
                    signature1 = signature_by_call_site.get(conv1["call_site"])
                    signature2 = signature_by_call_site.get(conv2["call_site"])
                    if signature1 is None or signature2 is None:
                        raise ValueError("call site is not mapped to a proven signature")
                    weight1 = _weight_contract(
                        conv=node1,
                        producer=producer,
                        initializers=initializers,
                        numpy_helper=numpy_helper,
                        helper=helper,
                    )
                    weight2 = _weight_contract(
                        conv=node2,
                        producer=producer,
                        initializers=initializers,
                        numpy_helper=numpy_helper,
                        helper=helper,
                    )
                    block = {
                        "prefix": prefix,
                        "call_index": call_index,
                        "norm1": norm1,
                        "conv1": {
                            **conv1,
                            **weight1,
                            "signature_id": signature1,
                            "conv_attributes": _conv_attributes(node1, helper),
                        },
                        "conv2": {
                            **conv2,
                            **weight2,
                            "signature_id": signature2,
                            "conv_attributes": _conv_attributes(node2, helper),
                        },
                        "block_input_tensor": norm1["input_tensor"],
                        "shortcut_tensor": conv2["epilogue_residual_tensor"],
                        "block_output_tensor": conv2["epilogue_output_tensor"],
                        "block_output_shape": list(conv2["epilogue_output_shape"]),
                        "shortcut_is_identity": conv2["epilogue_residual_tensor"] == norm1["input_tensor"],
                    }
                    blocks_by_call[call_index].append(block)
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"{kind}:{prefix}:call_{call_index}: {exc}")

        groups: list[dict[str, Any]] = []
        public_slots: dict[int, dict[str, Any]] = {}
        for call_index, blocks in blocks_by_call.items():
            blocks.sort(key=lambda value: _block_sort_key(value["prefix"]))
            by_input: dict[str, list[dict[str, Any]]] = defaultdict(list)
            by_output = {block["block_output_tensor"]: block for block in blocks}
            for block in blocks:
                by_input[block["block_input_tensor"]].append(block)
            predecessor: dict[str, str] = {}
            successor: dict[str, str] = {}
            for block in blocks:
                candidates = by_input.get(block["block_output_tensor"], [])
                if len(candidates) == 1:
                    next_block = candidates[0]
                    predecessor[next_block["prefix"]] = block["prefix"]
                    successor[block["prefix"]] = next_block["prefix"]
                elif len(candidates) > 1:
                    errors.append(
                        f"{kind}:call_{call_index}:{block['prefix']}: ambiguous residual successor"
                    )
            block_map = {block["prefix"]: block for block in blocks}
            starts = [block for block in blocks if block["prefix"] not in predecessor]
            call_groups: list[dict[str, Any]] = []
            visited: set[str] = set()
            for group_index, start in enumerate(sorted(starts, key=lambda value: _block_sort_key(value["prefix"]))):
                chain: list[dict[str, Any]] = []
                current = start
                while current["prefix"] not in visited:
                    visited.add(current["prefix"])
                    chain.append(current)
                    next_prefix = successor.get(current["prefix"])
                    if next_prefix is None:
                        break
                    current = block_map[next_prefix]
                for index, block in enumerate(chain):
                    block["group_index"] = group_index
                    block["group_block_index"] = index
                    block["group_size"] = len(chain)
                    block["input_is_int8"] = index > 0
                    block["output_is_int8"] = index + 1 < len(chain)
                    block["shortcut_is_int8"] = bool(
                        block["shortcut_is_identity"] and index > 0
                    )
                    block["profile_id"] = (
                        (0 if kind == "initial" else 1000)
                        + call_index * EXPECTED_RESIDUAL_BLOCKS
                        + len([value for value in blocks if _block_sort_key(value["prefix"]) < _block_sort_key(block["prefix"])])
                    )
                call_groups.append(
                    {
                        "group_index": group_index,
                        "block_prefixes": [block["prefix"] for block in chain],
                    }
                )
            if len(visited) != len(blocks):
                errors.append(f"{kind}:call_{call_index}: residual chain contains a cycle")
            groups.append({"call_index": call_index, "groups": call_groups, "blocks": blocks})

            for block in blocks:
                for conv_name in ("conv1", "conv2"):
                    record = block[conv_name]
                    module_name = record["module_name"]
                    cache_in = record.get("cache_tensor")
                    cache_out = record.get("cache_update_tensor")
                    if kind == "steady" and call_index == 0:
                        match = _CACHE_INPUT_RE.match(cache_in or "")
                        if match is None:
                            errors.append(f"{kind}:{record['call_site']}: no public cache input")
                        else:
                            public_cache_slots_by_module[module_name] = int(match.group("index"))
                    if call_index == 2:
                        match = _CACHE_OUTPUT_RE.match(cache_out or "")
                        if match is None:
                            errors.append(f"{kind}:{record['call_site']}: no public cache output")
                        else:
                            slot = int(match.group("index"))
                            existing = public_cache_slots_by_module.get(module_name)
                            if existing is not None and existing != slot:
                                errors.append(f"{kind}:{module_name}: cache input/output slot changed")
                            public_slots[slot] = {
                                "index": slot,
                                "module_name": module_name,
                                "input_name": f"cache_in_{slot:03d}",
                                "output_name": f"cache_out_{slot:03d}",
                                "shape": list(record["cache_update_shape"]),
                                "dtype": "int8",
                                "format": NATIVE_INT8_ACTIVATION_LAYOUT,
                            }

        if sum(len(entry["blocks"]) for entry in groups) != 42:
            errors.append(f"{kind}: expected 42 unrolled residual blocks")
        if len(public_slots) != EXPECTED_INT8_CACHE_SLOTS:
            errors.append(
                f"{kind}: expected {EXPECTED_INT8_CACHE_SLOTS} public INT8 cache slots, got {len(public_slots)}"
            )
        graphs[kind] = {
            "source_file": str(source_paths[kind].resolve()),
            "source_sha256": sha256_file(source_paths[kind]),
            "groups_by_call": groups,
            "int8_cache_slots": [public_slots[index] for index in sorted(public_slots)],
            "residual_block_call_count": sum(len(entry["blocks"]) for entry in groups),
            "target_conv_call_site_count": sum(
                2 * len(entry["blocks"]) for entry in groups
            ),
        }

    initial_slots = {
        entry["index"]: entry for entry in graphs.get("initial", {}).get("int8_cache_slots", [])
    }
    steady_slots = {
        entry["index"]: entry for entry in graphs.get("steady", {}).get("int8_cache_slots", [])
    }
    if initial_slots.keys() != steady_slots.keys():
        errors.append("initial/steady INT8 cache slot sets differ")
    for index in initial_slots.keys() & steady_slots.keys():
        for key in ("module_name", "shape", "dtype", "format"):
            if initial_slots[index].get(key) != steady_slots[index].get(key):
                errors.append(f"initial/steady cache slot {index} differs in {key}")
    if len(signature_ids) != EXPECTED_SIGNATURES:
        errors.append(
            f"expected {EXPECTED_SIGNATURES} Conv signatures, got {len(signature_ids)}"
        )
    return {
        "schema_version": NATIVE_INT8_SCHEMA_VERSION,
        "analysis_schema_version": NATIVE_INT8_ANALYSIS_SCHEMA_VERSION,
        "variant": NATIVE_INT8_VARIANT,
        "algorithm": NATIVE_INT8_ALGORITHM,
        "passed": not errors,
        "errors": errors,
        "logical_residual_block_count": EXPECTED_RESIDUAL_BLOCKS,
        "logical_target_conv_count": EXPECTED_LOGICAL_CONVS,
        "target_call_site_count_per_graph": EXPECTED_CALL_SITES_PER_GRAPH,
        "signature_count": len(signature_ids),
        "signatures": signatures,
        "graphs": graphs,
        "int8_cache_slots": [steady_slots[index] for index in sorted(steady_slots)],
        "int8_cache_slot_count": len(steady_slots),
        "fp16_cache_slot_count": EXPECTED_TOTAL_CACHE_SLOTS - len(steady_slots),
    }


def derive_native_static_scales(
    *, analysis: Mapping[str, Any], v5_scales: Mapping[str, Any]
) -> dict[str, Any]:
    if analysis.get("passed") is not True:
        raise ValueError("native analysis did not pass")
    activation_input = v5_scales.get("activation_input")
    activation_output = v5_scales.get("activation_output")
    if not isinstance(activation_input, Mapping) or not isinstance(activation_output, Mapping):
        raise ValueError("v5 activation input/output scales are missing")
    graphs: dict[str, Any] = {}
    shared_input_scales = {
        module_name: max(
            float(activation_input["initial"][module_name]),
            float(activation_input["steady"][module_name]),
        )
        for module_name in {
            block[conv_name]["module_name"]
            for kind in ("initial", "steady")
            for call in analysis["graphs"][kind]["groups_by_call"]
            for block in call["blocks"]
            for conv_name in ("conv1", "conv2")
        }
    }
    if any(
        not math.isfinite(value) or value <= 0
        for value in shared_input_scales.values()
    ):
        raise ValueError("native shared initial/steady activation scale is invalid")
    for kind in ("initial", "steady"):
        kind_input = activation_input.get(kind)
        kind_output = activation_output.get(kind)
        if not isinstance(kind_input, Mapping) or not isinstance(kind_output, Mapping):
            raise ValueError(f"v5 {kind} activation scales are missing")
        blocks: dict[str, Any] = {}
        for call in analysis["graphs"][kind]["groups_by_call"]:
            for block in call["blocks"]:
                conv1_name = block["conv1"]["module_name"]
                conv2_name = block["conv2"]["module_name"]
                values = {
                    "block_input": shared_input_scales[conv1_name],
                    "conv1_input": shared_input_scales[conv1_name],
                    "conv1_output": float(kind_output[conv1_name]),
                    "conv2_input": shared_input_scales[conv2_name],
                    "conv2_output": float(kind_output[conv2_name]),
                    "block_output": (
                        float(kind_input[conv1_name])
                        if block["output_is_int8"]
                        else float(kind_output[conv2_name])
                    ),
                }
                # For an internal block the next block's conv1 scale is the
                # exact physical group-output scale.  Resolve it by tensor
                # connectivity instead of assuming adjacent module names.
                if block["output_is_int8"]:
                    next_block = next(
                        candidate
                        for candidate in call["blocks"]
                        if candidate["block_input_tensor"]
                        == block["block_output_tensor"]
                    )
                    values["block_output"] = float(
                        shared_input_scales[next_block["conv1"]["module_name"]]
                    )
                if any(not math.isfinite(value) or value <= 0 for value in values.values()):
                    raise ValueError(f"invalid static scale for {kind}:{block['prefix']}")
                key = f"{block['prefix']}/call_{block['call_index']}"
                values["identity_residual_multiplier"] = (
                    values["block_input"] / values["block_output"]
                )
                blocks[key] = values
        graphs[kind] = blocks
    cache = {
        entry["module_name"]: {
            "slot": int(entry["index"]),
            "scale": shared_input_scales[entry["module_name"]],
        }
        for entry in analysis["int8_cache_slots"]
    }
    return {
        "schema_version": NATIVE_INT8_SCHEMA_VERSION,
        "variant": NATIVE_INT8_VARIANT,
        "runtime_scale_mode": "static",
        "calculation": "max_abs_over_127_from_v5_dummy_calibration",
        "graphs": graphs,
        "cache": cache,
    }


def _replace_value_info_dtype(model: Any, name: str, tensor_type: int) -> None:
    for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
        if value.name == name:
            value.type.tensor_type.elem_type = tensor_type


def _ensure_value_info(
    *, model: Any, helper: Any, name: str, tensor_type: int, shape: list[int]
) -> None:
    existing = {
        value.name
        for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]
    }
    if name not in existing:
        model.graph.value_info.extend(
            [helper.make_tensor_value_info(name, tensor_type, shape)]
        )
    else:
        _replace_value_info_dtype(model, name, tensor_type)


def rewrite_native_int8_graph(
    *,
    source_path: str | Path,
    destination_path: str | Path,
    graph_analysis: Mapping[str, Any],
    packed_tensors: Mapping[str, Any],
    weights_manifest: Mapping[str, Any],
    static_scales: Mapping[str, Any],
    tune: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace all 42 unrolled residual blocks with native plugin nodes."""

    if graph_analysis.get("source_sha256") != sha256_file(source_path):
        raise ValueError("native analysis/source digest mismatch")
    onnx, np, TensorProto, helpers = _lazy_onnx()
    helper, numpy_helper = helpers
    model = onnx.load(str(source_path), load_external_data=True)
    nodes_by_name = {node.name: node for node in model.graph.node}
    producer, _consumers = _node_maps(model)
    remove_nodes: set[str] = set()
    inserted: list[Any] = []
    new_initializers: dict[str, Any] = {}
    dummy_cache_name = "native_int8/dummy_cache"
    new_initializers[dummy_cache_name] = numpy_helper.from_array(
        np.zeros((1,), dtype=np.int8), dummy_cache_name
    )

    weight_records = weights_manifest.get("weights")
    if not isinstance(weight_records, Mapping):
        raise ValueError("packed-weight manifest is incomplete")
    selected_slots: set[int] = set()
    profile_ids: list[dict[str, Any]] = []
    kind = "initial" if "initial" in Path(source_path).name else "steady"
    scale_records = static_scales.get("graphs", {}).get(kind)
    if not isinstance(scale_records, Mapping):
        raise ValueError(f"native {kind} static scales are missing")

    def add_tensor(key: str, name: str, dtype: Any) -> str:
        tensor_name = f"native_int8/constants/{name}"
        if tensor_name not in new_initializers:
            if key not in packed_tensors:
                raise ValueError(f"packed tensor is missing: {key}")
            new_initializers[tensor_name] = numpy_helper.from_array(
                np.asarray(packed_tensors[key], dtype=dtype), tensor_name
            )
        return tensor_name

    for call in graph_analysis["groups_by_call"]:
        for block in call["blocks"]:
            prefix = block["prefix"]
            call_index = int(block["call_index"])
            conv1 = block["conv1"]
            conv2 = block["conv2"]
            scale_key = f"{prefix}/call_{call_index}"
            scales = scale_records[scale_key]
            weights1 = weight_records[conv1["module_name"]]
            weights2 = weight_records[conv2["module_name"]]

            cache1 = conv1.get("cache_tensor") or dummy_cache_name
            cache2 = conv2.get("cache_tensor") or dummy_cache_name
            has_cache1 = int(bool(conv1.get("cache_tensor")))
            has_cache2 = int(bool(conv2.get("cache_tensor")))
            cache1_out = conv1["cache_update_tensor"]
            cache2_out = conv2["cache_update_tensor"]
            for record in (conv1, conv2):
                match = _CACHE_OUTPUT_RE.match(record["cache_update_tensor"])
                if match:
                    selected_slots.add(int(match.group("index")))

            packed_weight1 = add_tensor(
                weights1["packed_key"], f"{conv1['module_name']}/weight", np.int8
            )
            weight_scale1 = add_tensor(
                weights1["scale_key"], f"{conv1['module_name']}/scale", np.float32
            )
            bias1 = add_tensor(
                weights1["bias_key"], f"{conv1['module_name']}/bias", np.float32
            )
            packed_weight2 = add_tensor(
                weights2["packed_key"], f"{conv2['module_name']}/weight", np.int8
            )
            weight_scale2 = add_tensor(
                weights2["scale_key"], f"{conv2['module_name']}/scale", np.float32
            )
            bias2 = add_tensor(
                weights2["bias_key"], f"{conv2['module_name']}/bias", np.float32
            )
            signature1 = conv1.get("signature_id") or weights1["signature_id"]
            signature2 = conv2.get("signature_id") or weights2["signature_id"]
            tile1 = int(tune["signatures"][signature1]["selected_tile_id"])
            tile2 = int(tune["signatures"][signature2]["selected_tile_id"])
            output_type = TensorProto.INT8 if block["output_is_int8"] else TensorProto.FLOAT16

            inputs = [
                block["block_input_tensor"],
                block["shortcut_tensor"],
                cache1,
                cache2,
                block["norm1"]["gamma_tensor"],
                conv1["epilogue_gamma_tensor"],
                packed_weight1,
                weight_scale1,
                bias1,
                packed_weight2,
                weight_scale2,
                bias2,
            ]
            outputs = [block["block_output_tensor"], cache1_out, cache2_out]
            attrs1 = conv1["conv_attributes"]
            attrs2 = conv2["conv_attributes"]
            explicit_pads1 = [int(value) for value in conv1["pads"]]
            explicit_pads2 = [int(value) for value in conv2["pads"]]
            if len(explicit_pads1) != 10 or len(explicit_pads2) != 10:
                raise ValueError("native Conv requires static rank-five Pad extents")
            plugin = helper.make_node(
                NATIVE_INT8_PLUGIN_NAME,
                inputs,
                outputs,
                name=f"native_int8/{kind}/{prefix}/call_{call_index}",
                domain="com.sglang.sfwan",
                plugin_version=NATIVE_INT8_PLUGIN_VERSION,
                plugin_namespace=NATIVE_INT8_PLUGIN_NAMESPACE,
                input_shape=[int(value) for value in block["norm1"]["output_shape"]],
                shortcut_shape=[int(value) for value in block["block_output_shape"]],
                output_shape=[int(value) for value in block["block_output_shape"]],
                cache1_input_shape=[int(value) for value in (conv1.get("cache_shape") or [0] * 5)],
                cache2_input_shape=[int(value) for value in (conv2.get("cache_shape") or [0] * 5)],
                cache1_output_shape=[int(value) for value in conv1["cache_update_shape"]],
                cache2_output_shape=[int(value) for value in conv2["cache_update_shape"]],
                weight1_shape=[int(value) for value in weights1["packed_shape"]],
                weight2_shape=[int(value) for value in weights2["packed_shape"]],
                conv1_params=[
                    explicit_pads1[3], explicit_pads1[4],
                    int(attrs1["strides"][1]), int(attrs1["strides"][2]),
                    int(attrs1["dilations"][1]), int(attrs1["dilations"][2]),
                ],
                conv2_params=[
                    explicit_pads2[3], explicit_pads2[4],
                    int(attrs2["strides"][1]), int(attrs2["strides"][2]),
                    int(attrs2["dilations"][1]), int(attrs2["dilations"][2]),
                ],
                input_is_int8=int(block["input_is_int8"]),
                shortcut_is_int8=int(block["shortcut_is_int8"]),
                output_is_int8=int(block["output_is_int8"]),
                has_cache1=has_cache1,
                has_cache2=has_cache2,
                tile1=tile1,
                tile2=tile2,
                profile_id=int(block["profile_id"]),
                input_scale=float(scales["block_input"]),
                conv1_input_scale=float(scales["conv1_input"]),
                conv2_input_scale=float(scales["conv2_input"]),
                output_scale=float(scales["block_output"]),
            )
            inserted.append(plugin)
            profile_ids.append(
                {
                    "profile_id": int(block["profile_id"]),
                    "plugin_name": plugin.name,
                    "block_prefix": prefix,
                    "call_index": call_index,
                    "conv1_signature": signature1,
                    "conv2_signature": signature2,
                }
            )

            remove_nodes.update(
                {
                    conv1["call_site"],
                    conv2["call_site"],
                    conv1["activation_cast_node"],
                    conv1["activation_q_node"],
                    conv1["activation_dq_node"],
                    conv1["output_q_node"],
                    conv1["output_dq_node"],
                    conv1["output_cast_node"],
                    conv2["activation_cast_node"],
                    conv2["activation_q_node"],
                    conv2["activation_dq_node"],
                    conv2["output_q_node"],
                    conv2["output_dq_node"],
                    conv2["output_cast_node"],
                    conv1["pad_node"],
                    conv2["pad_node"],
                    *block["norm1"]["remove_nodes"],
                    *conv1.get("epilogue_remove_nodes", []),
                    *conv2.get("epilogue_remove_nodes", []),
                }
            )
            for record in (conv1, conv2):
                if record.get("concat_node"):
                    remove_nodes.add(record["concat_node"])
                old_cache_producer = producer.get(record["cache_update_tensor"])
                if old_cache_producer is not None:
                    remove_nodes.add(old_cache_producer.name)

            _ensure_value_info(
                model=model,
                helper=helper,
                name=block["block_output_tensor"],
                tensor_type=output_type,
                shape=[int(value) for value in block["block_output_shape"]],
            )
            for record in (conv1, conv2):
                _ensure_value_info(
                    model=model,
                    helper=helper,
                    name=record["cache_update_tensor"],
                    tensor_type=TensorProto.INT8,
                    shape=[int(value) for value in record["cache_update_shape"]],
                )
                if record.get("cache_tensor"):
                    _replace_value_info_dtype(model, record["cache_tensor"], TensorProto.INT8)

    original_names = {node.name for node in model.graph.node}
    if any(node.name in original_names for node in inserted):
        raise ValueError("native plugin node collides with source graph")
    kept = [node for node in model.graph.node if node.name not in remove_nodes]
    del model.graph.node[:]
    model.graph.node.extend([*kept, *inserted])
    model.graph.initializer.extend(new_initializers.values())
    ordered = _topological_live_nodes(model)
    del model.graph.node[:]
    model.graph.node.extend(ordered)
    used_initializers = {
        value for node in model.graph.node for value in node.input if value
    }
    retained = [
        value for value in model.graph.initializer if value.name in used_initializers
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
    if not any(item.domain == "com.sglang.sfwan" for item in model.opset_import):
        model.opset_import.extend(
            [helper.make_opsetid("com.sglang.sfwan", NATIVE_INT8_SCHEMA_VERSION)]
        )
    onnx.checker.check_model(model, full_check=True)
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    onnx.save(model, str(temporary))
    temporary.replace(destination)
    return {
        "schema_version": NATIVE_INT8_SCHEMA_VERSION,
        "kind": kind,
        "source_sha256": sha256_file(source_path),
        "destination": str(destination.resolve()),
        "destination_sha256": sha256_file(destination),
        "plugin_node_count": len(inserted),
        "replaced_target_conv_call_site_count": len(inserted) * 2,
        "selected_int8_cache_slots": sorted(selected_slots),
        "profile_ids": profile_ids,
        "removed_node_count": len(remove_nodes),
        "algorithm": NATIVE_INT8_ALGORITHM,
    }


def load_native_int8_manifest(engine_dir: str | Path) -> dict[str, Any]:
    root = Path(engine_dir).expanduser().resolve() / NATIVE_INT8_SUBDIRECTORY
    path = root / NATIVE_INT8_MANIFEST_FILE
    if not path.is_file():
        raise ValueError(f"native INT8 manifest does not exist: {path}")
    return _load_json_object(path, label="native INT8 manifest")


def validate_native_int8_manifest(
    manifest: Mapping[str, Any],
    *,
    engine_dir: str | Path,
    base_manifest: Mapping[str, Any],
    verify_hashes: bool,
) -> dict[str, Any]:
    root = Path(engine_dir).expanduser().resolve() / NATIVE_INT8_SUBDIRECTORY
    if manifest.get("schema_version") != NATIVE_INT8_SCHEMA_VERSION:
        raise ValueError("unsupported native INT8 manifest schema")
    if manifest.get("variant") != NATIVE_INT8_VARIANT:
        raise ValueError("manifest is not native_int8_v1")
    if manifest.get("algorithm") != NATIVE_INT8_ALGORITHM:
        raise ValueError("native INT8 algorithm contract changed")
    if manifest.get("base_manifest_sha256") != _base_manifest_identity(base_manifest):
        raise ValueError("native/base manifest identity mismatch")

    plugin = manifest.get("plugin")
    engines = manifest.get("engines")
    audit_record = manifest.get("audit")
    cache = manifest.get("cache")
    if not all(
        isinstance(value, Mapping)
        for value in (plugin, engines, audit_record, cache)
    ):
        raise ValueError("native INT8 manifest is incomplete")
    plugin_path = _resolve_inside(root, plugin.get("file"), label="native plugin")
    if (
        plugin.get("plugin_name") != NATIVE_INT8_PLUGIN_NAME
        or plugin.get("plugin_version") != NATIVE_INT8_PLUGIN_VERSION
        or plugin.get("plugin_namespace") != NATIVE_INT8_PLUGIN_NAMESPACE
        or plugin.get("cutlass_commit") != NATIVE_INT8_CUTLASS_COMMIT
    ):
        raise ValueError("native plugin ABI identity changed")
    if verify_hashes and sha256_file(plugin_path) != plugin.get("sha256"):
        raise ValueError("native plugin SHA256 mismatch")
    audit_path = _resolve_inside(root, audit_record.get("file"), label="native audit")
    if verify_hashes and sha256_file(audit_path) != audit_record.get("sha256"):
        raise ValueError("native audit SHA256 mismatch")
    audit = _load_json_object(audit_path, label="native audit")
    if (
        audit.get("schema_version") != NATIVE_INT8_AUDIT_SCHEMA_VERSION
        or audit.get("passed") is not True
        or audit.get("errors") != []
        or audit.get("initial_call_site_count") != EXPECTED_CALL_SITES_PER_GRAPH
        or audit.get("steady_call_site_count") != EXPECTED_CALL_SITES_PER_GRAPH
        or audit.get("signature_count") != EXPECTED_SIGNATURES
    ):
        raise ValueError("native INT8 audit did not pass")
    inspector_plugin_mappings = audit.get("inspector_plugin_mappings")
    if not isinstance(inspector_plugin_mappings, Mapping):
        raise ValueError("native INT8 audit has no per-engine plugin mapping")
    for kind in ("initial", "steady"):
        mapping = inspector_plugin_mappings.get(kind)
        if (
            not isinstance(mapping, Mapping)
            or mapping.get("expected_count") != EXPECTED_RESIDUAL_BLOCKS * 3
            or mapping.get("mapped_count") != EXPECTED_RESIDUAL_BLOCKS * 3
            or mapping.get("missing") != []
        ):
            raise ValueError(
                f"native {kind} audit did not map every residual-block plugin"
            )
    validated_engines: dict[str, Any] = {}
    for kind in ("initial", "steady"):
        record = engines.get(kind)
        if not isinstance(record, Mapping):
            raise ValueError(f"native {kind} engine record is missing")
        plan = _resolve_inside(root, record.get("file"), label=f"native {kind} plan")
        inspector = _resolve_inside(
            root, record.get("inspector_file"), label=f"native {kind} inspector"
        )
        if verify_hashes:
            if sha256_file(plan) != record.get("sha256"):
                raise ValueError(f"native {kind} plan SHA256 mismatch")
            if sha256_file(inspector) != record.get("inspector_sha256"):
                raise ValueError(f"native {kind} inspector SHA256 mismatch")
        if audit.get("plan_sha256", {}).get(kind) != record.get("sha256"):
            raise ValueError(f"native {kind} plan/audit identity mismatch")
        validated_engines[kind] = {
            **dict(record),
            "path": str(plan),
            # TensorRT 10.3 may serialize inspector information either as a
            # list of layers or as an object containing a layer list.  Keep
            # that native shape; the catalog builder already accepts both.
            "inspector": _load_json_value(
                inspector, label=f"native {kind} inspector"
            ),
        }
    slots = cache.get("bindings")
    if not isinstance(slots, list) or len(slots) != EXPECTED_TOTAL_CACHE_SLOTS:
        raise ValueError("native mixed-cache binding list is incomplete")
    if any(not isinstance(entry, Mapping) for entry in slots):
        raise ValueError("native mixed-cache binding entries must be objects")
    indices = [entry.get("index") for entry in slots]
    if indices != list(range(EXPECTED_TOTAL_CACHE_SLOTS)):
        raise ValueError("native mixed-cache binding indices are not canonical")
    calculated_bank_bytes = 0
    for entry in slots:
        shape = entry.get("shape")
        dtype = entry.get("dtype")
        if (
            not isinstance(shape, list)
            or len(shape) != 5
            or any(not isinstance(value, int) or value <= 0 for value in shape)
        ):
            raise ValueError(f"native cache slot {entry['index']} shape is invalid")
        if dtype == "int8":
            if entry.get("format") != NATIVE_INT8_ACTIVATION_LAYOUT.lower():
                raise ValueError("native INT8 cache layout changed")
            scale = entry.get("scale")
            if not isinstance(scale, (int, float)) or not math.isfinite(
                float(scale)
            ) or float(scale) <= 0:
                raise ValueError("native INT8 cache scale is invalid")
            element_bytes = 1
        elif dtype == "float16":
            if entry.get("format") != "linear" or entry.get("scale") is not None:
                raise ValueError("native FP16 cache contract changed")
            element_bytes = 2
        else:
            raise ValueError("native mixed-cache dtype is unsupported")
        expected_bytes = math.prod(shape) * element_bytes
        if entry.get("bank_offset") != calculated_bank_bytes:
            raise ValueError("native mixed-cache bank offsets are not contiguous")
        if entry.get("bytes") != expected_bytes:
            raise ValueError("native mixed-cache byte count is invalid")
        calculated_bank_bytes += expected_bytes
    int8_slots = [entry for entry in slots if entry.get("dtype") == "int8"]
    fp16_slots = [entry for entry in slots if entry.get("dtype") == "float16"]
    if len(int8_slots) != EXPECTED_INT8_CACHE_SLOTS or len(fp16_slots) != EXPECTED_FP16_CACHE_SLOTS:
        raise ValueError("native mixed-cache dtype counts changed")
    if cache.get("migration") != "none":
        raise ValueError("native INT8 must not depend on cache migration")
    if (
        cache.get("single_bank_bytes") != calculated_bank_bytes
        or cache.get("double_bank_bytes") != calculated_bank_bytes * 2
    ):
        raise ValueError("native mixed-cache aggregate byte count changed")
    profile_ids = manifest.get("kernel_profile_ids")
    if not isinstance(profile_ids, Mapping):
        raise ValueError("native kernel profile map is missing")
    for kind in ("initial", "steady"):
        records = profile_ids.get(kind)
        if not isinstance(records, list) or len(records) != EXPECTED_RESIDUAL_BLOCKS * 3:
            raise ValueError(f"native {kind} kernel profile map is incomplete")
        if any(not isinstance(record, Mapping) for record in records):
            raise ValueError(f"native {kind} kernel profile entries must be objects")
        ids = [record.get("profile_id") for record in records]
        if (
            any(
                isinstance(profile_id, bool)
                or not isinstance(profile_id, int)
                or not 0 <= profile_id < 2048
                for profile_id in ids
            )
            or len(set(ids)) != len(ids)
        ):
            raise ValueError(f"native {kind} kernel profile IDs are not unique")
        for record in records:
            plugin_name = record.get("plugin_name")
            if (
                not isinstance(plugin_name, str)
                or not plugin_name.startswith(f"native_int8/{kind}/")
                or not isinstance(record.get("block_prefix"), str)
                or record.get("call_index") not in (0, 1, 2)
                or not isinstance(record.get("conv1_signature"), str)
                or not isinstance(record.get("conv2_signature"), str)
            ):
                raise ValueError(
                    f"native {kind} kernel profile metadata is incomplete"
                )
    return {
        "root": root,
        "plugin_path": plugin_path,
        "engines": validated_engines,
        "audit": audit,
        "cache_bindings": [dict(entry) for entry in slots],
        "profile_ids": dict(profile_ids),
    }


__all__ = [
    "EXPECTED_CALL_SITES_PER_GRAPH",
    "EXPECTED_FP16_CACHE_SLOTS",
    "EXPECTED_INT8_CACHE_SLOTS",
    "EXPECTED_LOGICAL_CONVS",
    "EXPECTED_RESIDUAL_BLOCKS",
    "EXPECTED_SIGNATURES",
    "NATIVE_INT8_ACTIVATION_LAYOUT",
    "NATIVE_INT8_ALGORITHM",
    "NATIVE_INT8_ANALYSIS_FILE",
    "NATIVE_INT8_AUDIT_FILE",
    "NATIVE_INT8_AUDIT_SCHEMA_VERSION",
    "NATIVE_INT8_BUILD_STATE_FILE",
    "NATIVE_INT8_CUTLASS_COMMIT",
    "NATIVE_INT8_INSPECTOR_FILES",
    "NATIVE_INT8_MANIFEST_FILE",
    "NATIVE_INT8_ONNX_FILES",
    "NATIVE_INT8_PACKED_WEIGHTS_FILE",
    "NATIVE_INT8_PLAN_FILES",
    "NATIVE_INT8_PLUGIN_CREATORS",
    "NATIVE_INT8_PLUGIN_INIT_SYMBOL",
    "NATIVE_INT8_PLUGIN_LIBRARY_FILE",
    "NATIVE_INT8_PLUGIN_MANIFEST_FILE",
    "NATIVE_INT8_PLUGIN_NAME",
    "NATIVE_INT8_PLUGIN_NAMESPACE",
    "NATIVE_INT8_PLUGIN_VERSION",
    "NATIVE_INT8_SCALES_FILE",
    "NATIVE_INT8_SCHEMA_VERSION",
    "NATIVE_INT8_SUBDIRECTORY",
    "NATIVE_INT8_TIMING_CACHE_FILE",
    "NATIVE_INT8_TUNE_FILE",
    "NATIVE_INT8_VARIANT",
    "NATIVE_INT8_WEIGHT_LAYOUT",
    "NATIVE_INT8_WEIGHTS_MANIFEST_FILE",
    "analyze_native_int8_graphs",
    "derive_native_static_scales",
    "load_native_int8_manifest",
    "rewrite_native_int8_graph",
    "validate_native_int8_manifest",
]
