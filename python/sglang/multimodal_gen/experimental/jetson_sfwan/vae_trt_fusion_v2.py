"""Producer-side INT8 boundary fusion for the steady SFWan VAE graph.

Fusion-v2 deliberately reuses the audited TensorRT INT8 Conv3d kernels from
Q/DQ-v5.  It rewrites only ``decoder.up_blocks.3`` in the steady graph so the
producer of each Conv activation also performs RMSNorm, SiLU, quantization,
causal packing, and feature-cache maintenance.  The initial graph remains the
unaltered v5 plan and is migrated once into the mixed-cache steady ABI.

This module stays import-light: ONNX and NumPy are imported only by build-time
functions, while runtime manifest validation needs only the standard library.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from .vae_trt_fusion import (
    _collect_reverse_subgraph,
    _constant_array,
    _lazy_onnx,
    _node_maps,
    _require_removable_subgraph,
    _topological_live_nodes,
    sha256_file,
    write_json_atomic,
)

FUSION_V2_VARIANT = "fusion_v2"
FUSION_V2_SCHEMA_VERSION = 2
FUSION_V2_ANALYSIS_SCHEMA_VERSION = 1
FUSION_V2_AUDIT_SCHEMA_VERSION = 1
FUSION_V2_SUBDIRECTORY = "fusion_v2"
FUSION_V2_MANIFEST_FILE = "fusion_v2_manifest.json"
FUSION_V2_AUDIT_FILE = "fusion_v2_audit.json"
FUSION_V2_ANALYSIS_FILE = "fusion_v2_analysis.json"
FUSION_V2_BUILD_STATE_FILE = "fusion_v2_build_state.json"
FUSION_V2_ONNX_FILE = "steady_int8_fusion_v2.onnx"
FUSION_V2_ENGINE_FILE = "steady_int8_fusion_v2.plan"
FUSION_V2_INSPECTOR_FILE = "steady_int8_fusion_v2_inspector.json"
FUSION_V2_TIMING_CACHE_FILE = "fusion_v2_timing.cache"
FUSION_V2_PLUGIN_LIBRARY_FILE = "libsfwan_vae_trt_fusion_v2.so"
FUSION_V2_PLUGIN_MANIFEST_FILE = "plugin_manifest.json"

PLUGIN_NAMESPACE = "sglang.sfwan"
PLUGIN_VERSION = "2"
BOUNDARY_PLUGIN = "SfWanResidualBoundaryV2Plugin"
PLUGIN_CREATORS = (BOUNDARY_PLUGIN,)
PLUGIN_INIT_SYMBOL = "initSfWanVaeTrtFusionV2Plugins"

FOCUS_MODULE_PREFIX = "decoder.up_blocks.3"
EXPECTED_LOGICAL_CONVS = 6
EXPECTED_CALLS_PER_CONV = 3
EXPECTED_SELECTED_CALL_SITES = EXPECTED_LOGICAL_CONVS * EXPECTED_CALLS_PER_CONV
EXPECTED_INT8_CACHE_SLOTS = 6

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


def _record_map(base_analysis: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = base_analysis.get("call_sites")
    if not isinstance(records, list):
        raise ValueError("fusion-v1 steady analysis has no call-site records")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if isinstance(record, dict) and isinstance(record.get("call_site"), str):
            result[record["call_site"]] = dict(record)
    return result


def _focused_records(base_analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = [
        record
        for record in _record_map(base_analysis).values()
        if record.get("focused") is True
    ]
    records.sort(key=lambda value: (value["call_index"], value["module_name"]))
    if len(records) != EXPECTED_SELECTED_CALL_SITES:
        raise ValueError(
            "fusion-v2 expects exactly 18 steady up_blocks.3 Conv call sites, "
            f"got {len(records)}"
        )
    for record in records:
        if not (
            record.get("eligible_input")
            and record.get("eligible_cache_update")
            and record.get("eligible_epilogue")
        ):
            raise ValueError(f"fusion-v2 call site is not proven: {record['call_site']}")
    return records


def _find_norm1_contract(
    *,
    record: Mapping[str, Any],
    peer_conv2: Mapping[str, Any],
    producer: Mapping[str, Any],
    consumers: Mapping[str, list[Any]],
    graph_outputs: set[str],
    initializers: Mapping[str, Any],
    helper: Any,
    numpy_helper: Any,
    np: Any,
    shapes: Mapping[str, list[int]],
) -> dict[str, Any]:
    """Recover ``block_input -> norm1 -> SiLU -> conv1 current`` exactly."""

    source = peer_conv2.get("epilogue_residual_tensor")
    target = record.get("current_tensor")
    if not isinstance(source, str) or not isinstance(target, str):
        raise ValueError("residual block input/current tensor is unavailable")
    subgraph = _collect_reverse_subgraph(
        target=target,
        source=source,
        producer=producer,
    )
    allowed_ops = {
        "Abs",
        "Add",
        "Cast",
        "Clip",
        "Constant",
        "Div",
        "Expand",
        "Identity",
        "Max",
        "Mul",
        "Pow",
        "ReduceL2",
        "ReduceMean",
        "ReduceSum",
        "Reshape",
        "Shape",
        "Sigmoid",
        "Slice",
        "Sqrt",
        "Unsqueeze",
    }
    unexpected = sorted({node.op_type for node in subgraph} - allowed_ops)
    if unexpected:
        raise ValueError(f"unexpected norm1/SiLU ops: {unexpected}")
    output_shape = shapes.get(target) or record.get("current_shape")
    if not isinstance(output_shape, list) or len(output_shape) != 5:
        raise ValueError("norm1 output shape is unavailable")
    channels = int(output_shape[1])
    produced = {tensor for node in subgraph for tensor in node.output}
    external = {
        tensor
        for node in subgraph
        for tensor in node.input
        if tensor and tensor != source and tensor not in produced
    }
    gamma_candidates: list[str] = []
    for name in external:
        value = _constant_array(
            name=name,
            initializers=initializers,
            numpy_helper=numpy_helper,
            producer=producer,
            helper=helper,
            np_module=np,
            shapes=dict(shapes),
            ranks={},
            memo={},
        )
        if value is not None and int(np.asarray(value).size) == channels:
            gamma_candidates.append(name)
    if len(gamma_candidates) != 1:
        raise ValueError(f"norm1 gamma is ambiguous: {gamma_candidates}")
    _require_removable_subgraph(
        nodes=subgraph,
        replacement_output=target,
        consumers=consumers,
        graph_outputs=graph_outputs,
    )
    return {
        "input_tensor": source,
        "gamma_tensor": gamma_candidates[0],
        "output_tensor": target,
        "output_shape": [int(value) for value in output_shape],
        "remove_nodes": [node.name for node in subgraph],
    }


def analyze_fusion_v2_steady_graph(
    *,
    source_path: str | Path,
    base_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the producer-side residual chain and selected cache ABI."""

    if base_analysis.get("graph_kind") != "steady":
        raise ValueError("fusion-v2 accepts only the steady graph analysis")
    onnx, np, _TensorProto, helpers = _lazy_onnx()
    helper, numpy_helper = helpers
    model = onnx.load(str(source_path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    try:
        inferred = onnx.shape_inference.infer_shapes(
            model, check_type=True, strict_mode=False, data_prop=True
        )
    except (RuntimeError, TypeError, ValueError):
        inferred = model
    shapes: dict[str, list[int]] = {}
    for value in [
        *inferred.graph.input,
        *inferred.graph.output,
        *inferred.graph.value_info,
    ]:
        dims: list[int] = []
        for dimension in value.type.tensor_type.shape.dim:
            if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
                dims = []
                break
            dims.append(int(dimension.dim_value))
        if dims:
            shapes[value.name] = dims
    producer, consumers = _node_maps(model)
    graph_outputs = {value.name for value in model.graph.output}
    initializers = {value.name: value for value in model.graph.initializer}
    focused = _focused_records(base_analysis)
    module_scales: dict[str, set[float]] = {}
    for record in focused:
        module_scales.setdefault(record["module_name"], set()).add(
            float(record["activation_scale"])
        )
    by_module_call = {
        (record["module_name"], int(record["call_index"])): record
        for record in focused
    }
    chains: list[dict[str, Any]] = []
    errors: list[str] = []
    for module_name, scales in module_scales.items():
        if len(scales) != 1:
            errors.append(
                f"{module_name}: activation/cache scale differs across calls: "
                f"{sorted(scales)}"
            )
    cache_slots: dict[int, dict[str, Any]] = {}
    for call_index in range(EXPECTED_CALLS_PER_CONV):
        blocks: list[dict[str, Any]] = []
        for block_index in range(3):
            prefix = f"{FOCUS_MODULE_PREFIX}.resnets.{block_index}"
            conv1 = by_module_call.get((f"{prefix}.conv1", call_index))
            conv2 = by_module_call.get((f"{prefix}.conv2", call_index))
            if conv1 is None or conv2 is None:
                errors.append(f"call {call_index} block {block_index}: Conv pair missing")
                continue
            try:
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
                if conv2.get("epilogue_residual_tensor") != norm1["input_tensor"]:
                    raise ValueError("up_blocks.3 shortcut is not identity")
                for record in (conv1, conv2):
                    cache_name = record.get("cache_tensor")
                    cache_out = record.get("cache_update_tensor")
                    cache_shape = record.get("cache_update_shape")
                    if not isinstance(cache_name, str) or not isinstance(cache_out, str):
                        raise ValueError("steady selected Conv has no external cache")
                    match_in = _CACHE_INPUT_RE.match(cache_name)
                    match_out = _CACHE_OUTPUT_RE.match(cache_out)
                    if call_index == 0:
                        if match_in is None:
                            raise ValueError(
                                f"first steady call cache is not public: {cache_name}"
                            )
                        slot = int(match_in.group("index"))
                        entry = cache_slots.setdefault(
                            slot,
                            {
                                "index": slot,
                                "input_name": cache_name,
                                "output_name": None,
                                "shape": [int(value) for value in cache_shape],
                                "scale": float(record["activation_scale"]),
                                "module_name": record["module_name"],
                            },
                        )
                        if not math.isclose(
                            entry["scale"],
                            float(record["activation_scale"]),
                            rel_tol=0.0,
                            abs_tol=0.0,
                        ):
                            raise ValueError("selected cache scale changed within module")
                    if call_index == 2 and match_out is None:
                        raise ValueError(
                            f"last steady call cache output is not public: {cache_out}"
                        )
                    if call_index == 2:
                        slot = int(match_out.group("index"))
                        if slot not in cache_slots:
                            raise ValueError("cache input/output slot mapping changed")
                        cache_slots[slot]["output_name"] = cache_out
                blocks.append(
                    {
                        "block_index": block_index,
                        "prefix": prefix,
                        "norm1": norm1,
                        "conv1": conv1,
                        "conv2": conv2,
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"call {call_index} block {block_index}: {exc}")
        chains.append({"call_index": call_index, "blocks": blocks})

    if len(cache_slots) != EXPECTED_INT8_CACHE_SLOTS:
        errors.append(
            f"expected {EXPECTED_INT8_CACHE_SLOTS} selected cache slots, "
            f"got {len(cache_slots)}"
        )
    for entry in cache_slots.values():
        if entry["output_name"] is None:
            errors.append(f"cache slot {entry['index']} has no public output")
    return {
        "schema_version": FUSION_V2_SCHEMA_VERSION,
        "analysis_schema_version": FUSION_V2_ANALYSIS_SCHEMA_VERSION,
        "variant": FUSION_V2_VARIANT,
        "source_file": str(Path(source_path).resolve()),
        "source_sha256": sha256_file(source_path),
        "focus_module_prefix": FOCUS_MODULE_PREFIX,
        "passed": not errors,
        "errors": errors,
        "chains": chains,
        "selected_cache_slots": [cache_slots[index] for index in sorted(cache_slots)],
        "selected_call_sites": sorted(record["call_site"] for record in focused),
    }


def _replace_value_info_dtype(model: Any, name: str, tensor_type: int) -> None:
    for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
        if value.name == name:
            value.type.tensor_type.elem_type = tensor_type


def _plugin_node(
    *,
    helper: Any,
    name: str,
    mode: int,
    inputs: list[str],
    outputs: list[str],
    consume_scale: float,
    produce_scale: float,
    current_shape: list[int],
    cache_shape: list[int] | None,
    packed_shape: list[int] | None,
    pads: list[int] | None,
) -> Any:
    return helper.make_node(
        BOUNDARY_PLUGIN,
        inputs,
        outputs,
        name=name,
        domain="com.sglang.sfwan",
        plugin_version=PLUGIN_VERSION,
        plugin_namespace=PLUGIN_NAMESPACE,
        mode=int(mode),
        consume_scale=float(consume_scale),
        produce_scale=float(produce_scale),
        current_shape=[int(value) for value in current_shape],
        cache_shape=[int(value) for value in (cache_shape or [0] * 5)],
        packed_shape=[int(value) for value in (packed_shape or [0] * 5)],
        pads=[int(value) for value in (pads or [0] * 10)],
    )


def rewrite_fusion_v2_steady_graph(
    *,
    source_path: str | Path,
    destination_path: str | Path,
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Rewrite three unrolled residual chains into producer-side plugins."""

    if analysis.get("passed") is not True or analysis.get("errors") != []:
        raise ValueError("fusion-v2 analysis did not pass")
    if analysis.get("source_sha256") != sha256_file(source_path):
        raise ValueError("fusion-v2 analysis/source digest mismatch")
    onnx, _np, TensorProto, helpers = _lazy_onnx()
    helper, _numpy_helper = helpers
    model = onnx.load(str(source_path), load_external_data=True)
    nodes_by_name = {node.name: node for node in model.graph.node}
    producer, _consumers = _node_maps(model)
    remove_nodes: set[str] = set()
    inserted: list[Any] = []
    value_infos: list[Any] = []

    def remove_activation_boundary(record: Mapping[str, Any], packed: str) -> None:
        dq = nodes_by_name[record["activation_dq_node"]]
        dq.input[0] = packed
        remove_nodes.update(
            {record["activation_cast_node"], record["activation_q_node"]}
        )
        cache_target = record["cache_update_tensor"]
        old_cache_producer = producer.get(cache_target)
        if old_cache_producer is not None:
            remove_nodes.add(old_cache_producer.name)

    def remove_conv_output_boundary(record: Mapping[str, Any]) -> None:
        remove_nodes.update(
            {
                record["output_dq_node"],
                record["output_cast_node"],
                *record.get("epilogue_remove_nodes", []),
            }
        )

    plugin_counts = {
        "norm1_silu_pack_quant": 0,
        "conv1_norm2_silu_pack_quant": 0,
        "conv2_residual_next_norm_pack_quant": 0,
        "conv2_residual_tail": 0,
    }
    for chain in analysis["chains"]:
        call_index = int(chain["call_index"])
        blocks = chain["blocks"]
        if len(blocks) != 3:
            raise ValueError(f"fusion-v2 call {call_index} has an incomplete chain")
        for block_index, block in enumerate(blocks):
            conv1 = block["conv1"]
            conv2 = block["conv2"]
            norm1 = block["norm1"]
            conv1_pack = f"fusion_v2/c{call_index}/b{block_index}/conv1_packed"
            conv1_cache = conv1["cache_update_tensor"]
            if block_index == 0:
                inserted.append(
                    _plugin_node(
                        helper=helper,
                        name=(
                            "fusion_v2/norm1_silu_quant_pack/"
                            f"call_{call_index}/block_{block_index}"
                        ),
                        mode=0,
                        inputs=[
                            norm1["input_tensor"],
                            norm1["gamma_tensor"],
                            conv1["cache_tensor"],
                        ],
                        outputs=[conv1_pack, conv1_cache],
                        consume_scale=1.0,
                        produce_scale=float(conv1["activation_scale"]),
                        current_shape=norm1["output_shape"],
                        cache_shape=conv1["cache_update_shape"],
                        packed_shape=conv1["padded_shape"],
                        pads=conv1["pads"],
                    )
                )
                remove_nodes.update(norm1["remove_nodes"])
                plugin_counts["norm1_silu_pack_quant"] += 1
            # For block 1/2 the previous mode-2 plugin creates conv1_pack and
            # conv1_cache, so only the consumer rewiring happens here.
            value_infos.append(
                helper.make_tensor_value_info(
                    conv1_pack, TensorProto.INT8, conv1["padded_shape"]
                )
            )
            remove_activation_boundary(conv1, conv1_pack)

            conv2_pack = f"fusion_v2/c{call_index}/b{block_index}/conv2_packed"
            output_q1 = nodes_by_name[conv1["output_q_node"]]
            inserted.append(
                _plugin_node(
                    helper=helper,
                    name=(
                        "fusion_v2/conv1_to_conv2_norm_silu_quant_pack/"
                        f"call_{call_index}/block_{block_index}"
                    ),
                    mode=1,
                    inputs=[
                        output_q1.output[0],
                        conv1["epilogue_gamma_tensor"],
                        conv2["cache_tensor"],
                    ],
                    outputs=[conv2_pack, conv2["cache_update_tensor"]],
                    consume_scale=float(conv1["output_scale"]),
                    produce_scale=float(conv2["activation_scale"]),
                    current_shape=conv1["conv_output_shape"],
                    cache_shape=conv2["cache_update_shape"],
                    packed_shape=conv2["padded_shape"],
                    pads=conv2["pads"],
                )
            )
            value_infos.append(
                helper.make_tensor_value_info(
                    conv2_pack, TensorProto.INT8, conv2["padded_shape"]
                )
            )
            remove_conv_output_boundary(conv1)
            remove_activation_boundary(conv2, conv2_pack)
            plugin_counts["conv1_norm2_silu_pack_quant"] += 1

            output_q2 = nodes_by_name[conv2["output_q_node"]]
            residual_output = conv2["epilogue_output_tensor"]
            if block_index < 2:
                next_block = blocks[block_index + 1]
                next_conv1 = next_block["conv1"]
                next_norm1 = next_block["norm1"]
                next_pack = (
                    f"fusion_v2/c{call_index}/b{block_index + 1}/conv1_packed"
                )
                inserted.append(
                    _plugin_node(
                        helper=helper,
                        name=(
                            "fusion_v2/conv2_residual_next_norm_silu_quant_pack/"
                            f"call_{call_index}/block_{block_index}"
                        ),
                        mode=2,
                        inputs=[
                            output_q2.output[0],
                            conv2["epilogue_residual_tensor"],
                            next_norm1["gamma_tensor"],
                            next_conv1["cache_tensor"],
                        ],
                        outputs=[
                            residual_output,
                            next_pack,
                            next_conv1["cache_update_tensor"],
                        ],
                        consume_scale=float(conv2["output_scale"]),
                        produce_scale=float(next_conv1["activation_scale"]),
                        current_shape=conv2["conv_output_shape"],
                        cache_shape=next_conv1["cache_update_shape"],
                        packed_shape=next_conv1["padded_shape"],
                        pads=next_conv1["pads"],
                    )
                )
                remove_nodes.update(next_norm1["remove_nodes"])
                plugin_counts["conv2_residual_next_norm_pack_quant"] += 1
            else:
                inserted.append(
                    _plugin_node(
                        helper=helper,
                        name=(
                            "fusion_v2/conv2_residual_tail/"
                            f"call_{call_index}/block_{block_index}"
                        ),
                        mode=3,
                        inputs=[
                            output_q2.output[0],
                            conv2["epilogue_residual_tensor"],
                        ],
                        outputs=[residual_output],
                        consume_scale=float(conv2["output_scale"]),
                        produce_scale=1.0,
                        current_shape=conv2["conv_output_shape"],
                        cache_shape=None,
                        packed_shape=None,
                        pads=None,
                    )
                )
                plugin_counts["conv2_residual_tail"] += 1
            remove_conv_output_boundary(conv2)

    selected_names: set[str] = set()
    for entry in analysis["selected_cache_slots"]:
        selected_names.add(entry["input_name"])
        selected_names.add(entry["output_name"])
    for chain in analysis["chains"]:
        for block in chain["blocks"]:
            for record in (block["conv1"], block["conv2"]):
                selected_names.add(record["cache_tensor"])
                selected_names.add(record["cache_update_tensor"])
    for name in selected_names:
        _replace_value_info_dtype(model, name, TensorProto.INT8)

    original_names = {node.name for node in model.graph.node}
    if any(node.name in original_names for node in inserted):
        raise ValueError("fusion-v2 plugin node name collision")
    kept = [node for node in model.graph.node if node.name not in remove_nodes]
    del model.graph.node[:]
    model.graph.node.extend([*kept, *inserted])
    model.graph.value_info.extend(value_infos)
    ordered = _topological_live_nodes(model)
    del model.graph.node[:]
    model.graph.node.extend(ordered)
    if not any(item.domain == "com.sglang.sfwan" for item in model.opset_import):
        model.opset_import.extend(
            [helper.make_opsetid("com.sglang.sfwan", FUSION_V2_SCHEMA_VERSION)]
        )
    used_initializers = {
        tensor for node in model.graph.node for tensor in node.input if tensor
    }
    kept_initializers = [
        value for value in model.graph.initializer if value.name in used_initializers
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    onnx.checker.check_model(model, full_check=True)
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    onnx.save(model, str(temporary))
    temporary.replace(destination)
    return {
        "schema_version": FUSION_V2_SCHEMA_VERSION,
        "source_sha256": sha256_file(source_path),
        "destination": str(destination.resolve()),
        "destination_sha256": sha256_file(destination),
        "selected_call_sites": list(analysis["selected_call_sites"]),
        "selected_cache_slots": list(analysis["selected_cache_slots"]),
        "plugin_counts": plugin_counts,
        "plugin_node_count": len(inserted),
        "removed_node_count": len(remove_nodes),
    }


def load_fusion_v2_manifest(engine_dir: str | Path) -> dict[str, Any]:
    root = Path(engine_dir).expanduser().resolve() / FUSION_V2_SUBDIRECTORY
    path = root / FUSION_V2_MANIFEST_FILE
    if not path.is_file():
        raise ValueError(f"TensorRT fusion-v2 manifest does not exist: {path}")
    return _load_json_object(path, label="TensorRT fusion-v2 manifest")


def validate_fusion_v2_manifest(
    manifest: Mapping[str, Any],
    *,
    engine_dir: str | Path,
    base_manifest: Mapping[str, Any],
    verify_hashes: bool,
) -> dict[str, Any]:
    root = Path(engine_dir).expanduser().resolve() / FUSION_V2_SUBDIRECTORY
    if manifest.get("schema_version") != FUSION_V2_SCHEMA_VERSION:
        raise ValueError("unsupported fusion-v2 manifest schema")
    if manifest.get("variant") != FUSION_V2_VARIANT:
        raise ValueError("manifest is not fusion_v2")
    base_sha = manifest.get("base_manifest_sha256")
    actual_base_sha = hashlib.sha256(
        json.dumps(base_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if base_sha != actual_base_sha:
        raise ValueError("fusion-v2/base manifest identity mismatch")
    plugin = manifest.get("plugin")
    steady = manifest.get("steady_engine")
    audit = manifest.get("audit")
    if not all(isinstance(value, Mapping) for value in (plugin, steady, audit)):
        raise ValueError("fusion-v2 manifest is incomplete")
    plugin_path = _resolve_inside(root, plugin.get("file"), label="fusion-v2 plugin")
    plan_path = _resolve_inside(root, steady.get("file"), label="fusion-v2 plan")
    inspector_path = _resolve_inside(
        root, steady.get("inspector_file"), label="fusion-v2 inspector"
    )
    audit_path = _resolve_inside(root, audit.get("file"), label="fusion-v2 audit")
    for path, record, label in (
        (plugin_path, plugin, "plugin"),
        (plan_path, steady, "steady plan"),
        (inspector_path, {"sha256": steady.get("inspector_sha256")}, "inspector"),
        (audit_path, {"sha256": audit.get("sha256")}, "audit"),
    ):
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"fusion-v2 {label} SHA256 is invalid")
        if verify_hashes and sha256_file(path) != digest:
            raise ValueError(f"fusion-v2 {label} SHA256 mismatch")
    audit_value = _load_json_object(audit_path, label="fusion-v2 audit")
    if (
        audit_value.get("schema_version") != FUSION_V2_AUDIT_SCHEMA_VERSION
        or audit_value.get("passed") is not True
        or audit_value.get("errors") != []
        or audit_value.get("steady_plan_sha256") != steady.get("sha256")
    ):
        raise ValueError("fusion-v2 audit did not pass")
    cache = manifest.get("cache")
    slots = cache.get("selected_slots") if isinstance(cache, Mapping) else None
    if not isinstance(slots, list) or len(slots) != EXPECTED_INT8_CACHE_SLOTS:
        raise ValueError("fusion-v2 selected-cache contract is incomplete")
    indices = set()
    for entry in slots:
        if not isinstance(entry, Mapping):
            raise ValueError("fusion-v2 cache slot is invalid")
        index = entry.get("index")
        shape = entry.get("shape")
        scale = entry.get("scale")
        if (
            not isinstance(index, int)
            or index in indices
            or not isinstance(shape, list)
            or len(shape) != 5
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) <= 0
            or entry.get("dtype") != "int8"
        ):
            raise ValueError("fusion-v2 cache slot contract is invalid")
        indices.add(index)
    return {
        "root": root,
        "plugin_path": plugin_path,
        "steady_engine": {
            "path": plan_path,
            "sha256": steady["sha256"],
            "inspector": _load_json_object(
                inspector_path, label="fusion-v2 inspector"
            ),
        },
        "audit": audit_value,
        "selected_cache_slots": [dict(entry) for entry in slots],
        "plugin_init_symbol": str(plugin.get("init_symbol", PLUGIN_INIT_SYMBOL)),
    }


__all__ = [
    "BOUNDARY_PLUGIN",
    "EXPECTED_INT8_CACHE_SLOTS",
    "FUSION_V2_ANALYSIS_FILE",
    "FUSION_V2_ANALYSIS_SCHEMA_VERSION",
    "FUSION_V2_AUDIT_FILE",
    "FUSION_V2_AUDIT_SCHEMA_VERSION",
    "FUSION_V2_BUILD_STATE_FILE",
    "FUSION_V2_ENGINE_FILE",
    "FUSION_V2_INSPECTOR_FILE",
    "FUSION_V2_MANIFEST_FILE",
    "FUSION_V2_ONNX_FILE",
    "FUSION_V2_PLUGIN_LIBRARY_FILE",
    "FUSION_V2_PLUGIN_MANIFEST_FILE",
    "FUSION_V2_SCHEMA_VERSION",
    "FUSION_V2_SUBDIRECTORY",
    "FUSION_V2_TIMING_CACHE_FILE",
    "FUSION_V2_VARIANT",
    "PLUGIN_CREATORS",
    "PLUGIN_INIT_SYMBOL",
    "PLUGIN_NAMESPACE",
    "PLUGIN_VERSION",
    "analyze_fusion_v2_steady_graph",
    "load_fusion_v2_manifest",
    "rewrite_fusion_v2_steady_graph",
    "validate_fusion_v2_manifest",
    "write_json_atomic",
]
