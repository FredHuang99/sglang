"""Explicit-Q/DQ ONNX rewriting for the Jetson SFWan VAE.

The build tool exports an FP16 graph that calls the existing SGLang Wan
decoder.  This module rewrites only the 28 residual-block Conv3d weights and
their 84 unrolled call sites.  ONNX and NumPy are imported lazily so importing
the service does not require the TensorRT build environment.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

QDQ_SCHEMA_VERSION = 3
QDQ_OPSET = 19
INT8_MAX = 127.0
MIN_QUANT_SCALE = 1.0e-8
EXPECTED_LOGICAL_CONVS = 28
EXPECTED_CALLS_PER_LOGICAL_CONV = 3
EXPECTED_CALL_SITES = EXPECTED_LOGICAL_CONVS * EXPECTED_CALLS_PER_LOGICAL_CONV


def _lazy_onnx() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:  # pragma: no cover - exercised on the Orin builder
        raise RuntimeError(
            "TensorRT VAE graph rewriting requires the 'onnx' and 'numpy' packages"
        ) from exc
    return onnx, np, TensorProto, (helper, numpy_helper)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _initializer_map(model: Any) -> dict[str, Any]:
    return {initializer.name: initializer for initializer in model.graph.initializer}


def _resolve_weight_names(
    *,
    initializer_names: set[str],
    target_module_names: tuple[str, ...],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for module_name in target_module_names:
        suffix = f"{module_name}.weight"
        matches = sorted(
            name
            for name in initializer_names
            if name == suffix or name.endswith(f".{suffix}")
        )
        if len(matches) != 1:
            raise ValueError(
                f"expected one ONNX initializer for {suffix!r}, found {matches}"
            )
        resolved[module_name] = matches[0]
    return resolved


def _remove_unused_initializers(model: Any) -> None:
    used = {input_name for node in model.graph.node for input_name in node.input}
    retained = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name in used
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)


def _opset_version(model: Any) -> int:
    for opset in model.opset_import:
        if opset.domain in {"", "ai.onnx"}:
            return int(opset.version)
    raise ValueError("ONNX graph has no default-domain opset")


def _as_positive_scale(value: Any, *, module_name: str) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"activation scale for {module_name!r} is not numeric"
        ) from exc
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"activation scale for {module_name!r} must be positive and finite"
        )
    return max(scale, MIN_QUANT_SCALE)


def _attribute_map(node: Any, helper: Any) -> dict[str, Any]:
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def _tensor_shape_map(model: Any) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    values = [
        *model.graph.input,
        *model.graph.output,
        *model.graph.value_info,
    ]
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dimensions: list[int] = []
        valid = True
        for dimension in tensor_type.shape.dim:
            if not dimension.HasField("dim_value") or int(dimension.dim_value) <= 0:
                valid = False
                break
            dimensions.append(int(dimension.dim_value))
        if valid:
            shapes[value.name] = dimensions
    for initializer in model.graph.initializer:
        shapes[initializer.name] = [int(value) for value in initializer.dims]
    return shapes


def _canonical_signature(signature: dict[str, Any]) -> dict[str, Any]:
    return {
        key: signature[key]
        for key in (
            "input_shape",
            "weight_shape",
            "output_shape",
            "kernel_shape",
            "pads",
            "strides",
            "dilations",
            "group",
        )
    }


def conv_signature_id(signature: dict[str, Any]) -> str:
    encoded = json.dumps(
        _canonical_signature(signature),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _axis(node: Any, helper: Any) -> int | None:
    attributes = _attribute_map(node, helper)
    value = attributes.get("axis")
    return int(value) if value is not None else None


def _is_zero_int8(initializer: Any, *, TensorProto: Any, numpy_helper: Any) -> bool:
    if initializer is None or int(initializer.data_type) != int(TensorProto.INT8):
        return False
    value = numpy_helper.to_array(initializer)
    return bool((value == 0).all())


def _is_positive_finite(value: Any) -> bool:
    return bool((value > 0).all() and math.isfinite(float(value.max())))


def _make_qdq_call_site(
    *,
    helper: Any,
    numpy_helper: Any,
    np: Any,
    graph_kind: str,
    module_name: str,
    call_index: int,
    activation_input: str,
    weight_fp16: Any,
    weight_scale_fp16: Any,
    activation_scale_fp16: Any,
    bias_fp16: Any,
) -> tuple[list[Any], list[Any], str, str, str, dict[str, str]]:
    """Create the canonical v3 constant->Q->DQ topology for one Conv call."""

    safe_module = _safe_name(module_name)
    prefix = f"sfwan_{graph_kind}_{safe_module}_call_{call_index}"
    activation_scale_name = f"{prefix}_activation_scale"
    activation_zero_name = f"{prefix}_activation_zero"
    activation_int8 = f"{prefix}_activation_int8"
    activation_dq = f"{prefix}_activation_dequantized"
    weight_source_name = f"{prefix}_weight_fp16_source"
    weight_scale_name = f"{prefix}_weight_scale"
    weight_zero_name = f"{prefix}_weight_zero"
    weight_int8 = f"{prefix}_weight_int8"
    weight_dq = f"{prefix}_weight_dequantized"
    bias_name = f"{prefix}_bias_fp16"

    initializers = [
        numpy_helper.from_array(
            np.ascontiguousarray(weight_fp16, dtype=np.float16),
            name=weight_source_name,
        ),
        numpy_helper.from_array(
            np.ascontiguousarray(weight_scale_fp16, dtype=np.float16),
            name=weight_scale_name,
        ),
        numpy_helper.from_array(
            np.zeros(weight_scale_fp16.shape, dtype=np.int8),
            name=weight_zero_name,
        ),
        numpy_helper.from_array(
            np.asarray(activation_scale_fp16, dtype=np.float16),
            name=activation_scale_name,
        ),
        numpy_helper.from_array(
            np.asarray(0, dtype=np.int8),
            name=activation_zero_name,
        ),
        numpy_helper.from_array(
            np.ascontiguousarray(bias_fp16, dtype=np.float16),
            name=bias_name,
        ),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            [activation_input, activation_scale_name, activation_zero_name],
            [activation_int8],
            name=(
                f"qdq/{graph_kind}/activation/{module_name}/call_{call_index}/quantize"
            ),
        ),
        helper.make_node(
            "DequantizeLinear",
            [activation_int8, activation_scale_name, activation_zero_name],
            [activation_dq],
            name=(
                f"qdq/{graph_kind}/activation/{module_name}/"
                f"call_{call_index}/dequantize"
            ),
        ),
        helper.make_node(
            "QuantizeLinear",
            [weight_source_name, weight_scale_name, weight_zero_name],
            [weight_int8],
            name=(f"qdq/{graph_kind}/weight/{module_name}/call_{call_index}/quantize"),
            axis=0,
        ),
        helper.make_node(
            "DequantizeLinear",
            [weight_int8, weight_scale_name, weight_zero_name],
            [weight_dq],
            name=(
                f"qdq/{graph_kind}/weight/{module_name}/call_{call_index}/dequantize"
            ),
            axis=0,
        ),
    ]
    names = {
        "activation_scale": activation_scale_name,
        "activation_zero": activation_zero_name,
        "weight_source": weight_source_name,
        "weight_scale": weight_scale_name,
        "weight_zero": weight_zero_name,
        "bias": bias_name,
    }
    return nodes, initializers, activation_dq, weight_dq, bias_name, names


def audit_qdq_model(
    model: Any,
    *,
    graph_kind: str,
    target_module_names: tuple[str, ...],
) -> dict[str, Any]:
    """Validate the complete v3 structural INT8 contract after rewriting."""

    _onnx, np, TensorProto, helpers = _lazy_onnx()
    helper, numpy_helper = helpers
    target_call_prefix = f"int8/{graph_kind}/"
    target_convs = [
        node
        for node in model.graph.node
        if node.op_type == "Conv" and node.name.startswith(target_call_prefix)
    ]
    qlinear_convs = [
        node.name
        for node in model.graph.node
        if node.op_type in {"QLinearConv", "ConvInteger"}
    ]
    activation_q = [
        node
        for node in model.graph.node
        if node.op_type == "QuantizeLinear"
        and node.name.startswith(f"qdq/{graph_kind}/activation/")
    ]
    activation_dq = [
        node
        for node in model.graph.node
        if node.op_type == "DequantizeLinear"
        and node.name.startswith(f"qdq/{graph_kind}/activation/")
    ]
    weight_q = [
        node
        for node in model.graph.node
        if node.op_type == "QuantizeLinear"
        and node.name.startswith(f"qdq/{graph_kind}/weight/")
    ]
    weight_dq = [
        node
        for node in model.graph.node
        if node.op_type == "DequantizeLinear"
        and node.name.startswith(f"qdq/{graph_kind}/weight/")
    ]
    target_casts = [
        node.name
        for node in model.graph.node
        if node.op_type == "Cast" and node.name.startswith(f"qdq/{graph_kind}/")
    ]
    initializers = _initializer_map(model)
    producers = {
        output_name: node
        for node in model.graph.node
        for output_name in node.output
        if output_name
    }
    shapes = _tensor_shape_map(model)
    per_module_counts = {
        module_name: sum(
            node.name.startswith(f"{target_call_prefix}{module_name}/call_")
            for node in target_convs
        )
        for module_name in target_module_names
    }

    errors: list[str] = []
    if len(target_module_names) != EXPECTED_LOGICAL_CONVS:
        errors.append(
            f"logical target count is {len(target_module_names)}, "
            f"expected {EXPECTED_LOGICAL_CONVS}"
        )
    expected_counts = {
        "target Conv": len(target_convs),
        "activation QuantizeLinear": len(activation_q),
        "activation DequantizeLinear": len(activation_dq),
        "weight QuantizeLinear": len(weight_q),
        "weight DequantizeLinear": len(weight_dq),
    }
    for label, count in expected_counts.items():
        if count != EXPECTED_CALL_SITES:
            errors.append(f"{label} count is {count}, expected {EXPECTED_CALL_SITES}")
    invalid_modules = {
        name: count
        for name, count in per_module_counts.items()
        if count != EXPECTED_CALLS_PER_LOGICAL_CONV
    }
    if invalid_modules:
        errors.append(f"per-module call counts are invalid: {invalid_modules}")
    if qlinear_convs:
        errors.append(f"unsupported quantized Conv operators exist: {qlinear_convs}")
    if target_casts:
        errors.append(f"target Q/DQ paths contain Cast nodes: {target_casts}")

    invalid_bindings: list[str] = []
    weight_sources: set[str] = set()
    weight_q_outputs: set[str] = set()
    weight_dq_outputs: set[str] = set()
    activation_q_outputs: set[str] = set()
    activation_dq_outputs: set[str] = set()
    bias_sources: set[str] = set()
    signatures: list[dict[str, Any]] = []

    for conv in target_convs:
        if len(conv.input) < 3 or not conv.input[2]:
            invalid_bindings.append(f"{conv.name}:missing_bias")
            continue
        activation_dequantize = producers.get(conv.input[0])
        weight_dequantize = producers.get(conv.input[1])
        if (
            activation_dequantize is None
            or activation_dequantize.op_type != "DequantizeLinear"
        ):
            invalid_bindings.append(f"{conv.name}:activation_missing_dq")
            continue
        if weight_dequantize is None or weight_dequantize.op_type != "DequantizeLinear":
            invalid_bindings.append(f"{conv.name}:weight_missing_dq")
            continue
        activation_quantize = producers.get(activation_dequantize.input[0])
        weight_quantize = producers.get(weight_dequantize.input[0])
        if (
            activation_quantize is None
            or activation_quantize.op_type != "QuantizeLinear"
        ):
            invalid_bindings.append(f"{conv.name}:activation_missing_q")
            continue
        if weight_quantize is None or weight_quantize.op_type != "QuantizeLinear":
            invalid_bindings.append(f"{conv.name}:weight_missing_q")
            continue

        activation_q_outputs.add(activation_quantize.output[0])
        activation_dq_outputs.add(activation_dequantize.output[0])
        weight_q_outputs.add(weight_quantize.output[0])
        weight_dq_outputs.add(weight_dequantize.output[0])
        weight_source_name = weight_quantize.input[0]
        bias_name = conv.input[2]
        weight_sources.add(weight_source_name)
        bias_sources.add(bias_name)

        activation_scale = initializers.get(activation_quantize.input[1])
        activation_zero = initializers.get(activation_quantize.input[2])
        weight_source = initializers.get(weight_source_name)
        weight_scale = initializers.get(weight_quantize.input[1])
        weight_zero = initializers.get(weight_quantize.input[2])
        bias = initializers.get(bias_name)
        binding_errors: list[str] = []
        if list(activation_quantize.input[1:]) != list(activation_dequantize.input[1:]):
            binding_errors.append("activation_q_dq_parameters_differ")
        if list(weight_quantize.input[1:]) != list(weight_dequantize.input[1:]):
            binding_errors.append("weight_q_dq_parameters_differ")
        if _axis(weight_quantize, helper) != 0 or _axis(weight_dequantize, helper) != 0:
            binding_errors.append("weight_axis_not_zero")
        if activation_scale is None:
            binding_errors.append("activation_scale_missing")
        else:
            activation_scale_array = numpy_helper.to_array(activation_scale)
            if (
                int(activation_scale.data_type) != int(TensorProto.FLOAT16)
                or activation_scale_array.ndim != 0
                or not _is_positive_finite(activation_scale_array)
            ):
                binding_errors.append("activation_scale_invalid")
        if not _is_zero_int8(
            activation_zero,
            TensorProto=TensorProto,
            numpy_helper=numpy_helper,
        ):
            binding_errors.append("activation_zero_invalid")
        if weight_source is None:
            binding_errors.append("weight_source_missing")
            weight_shape: list[int] = []
        else:
            weight_shape = [int(value) for value in weight_source.dims]
            if (
                int(weight_source.data_type) != int(TensorProto.FLOAT16)
                or len(weight_shape) != 5
            ):
                binding_errors.append("weight_source_not_rank5_fp16")
        if weight_scale is None:
            binding_errors.append("weight_scale_missing")
        else:
            weight_scale_array = numpy_helper.to_array(weight_scale)
            expected_channels = weight_shape[0] if weight_shape else -1
            if (
                int(weight_scale.data_type) != int(TensorProto.FLOAT16)
                or weight_scale_array.ndim != 1
                or int(weight_scale_array.size) != expected_channels
                or not _is_positive_finite(weight_scale_array)
            ):
                binding_errors.append("weight_scale_invalid")
        if not _is_zero_int8(
            weight_zero,
            TensorProto=TensorProto,
            numpy_helper=numpy_helper,
        ):
            binding_errors.append("weight_zero_invalid")
        if (
            bias is None
            or int(bias.data_type) != int(TensorProto.FLOAT16)
            or len(bias.dims) != 1
            or (weight_shape and int(bias.dims[0]) != weight_shape[0])
        ):
            binding_errors.append("bias_invalid")
        if binding_errors:
            invalid_bindings.extend(f"{conv.name}:{value}" for value in binding_errors)
            continue

        attributes = _attribute_map(conv, helper)
        input_shape = shapes.get(activation_quantize.input[0])
        output_shape = shapes.get(conv.output[0])
        if input_shape is None or output_shape is None:
            invalid_bindings.append(f"{conv.name}:static_shape_missing")
            continue
        signature = {
            "call_site": conv.name,
            "input_shape": input_shape,
            "weight_shape": weight_shape,
            "output_shape": output_shape,
            "kernel_shape": [
                int(value) for value in attributes.get("kernel_shape", weight_shape[2:])
            ],
            "pads": [int(value) for value in attributes.get("pads", [0] * 6)],
            "strides": [int(value) for value in attributes.get("strides", [1] * 3)],
            "dilations": [int(value) for value in attributes.get("dilations", [1] * 3)],
            "group": int(attributes.get("group", 1)),
        }
        signature["signature_id"] = conv_signature_id(signature)
        signatures.append(signature)

    unique_requirements = {
        "weight sources": weight_sources,
        "weight Q outputs": weight_q_outputs,
        "weight DQ outputs": weight_dq_outputs,
        "activation Q outputs": activation_q_outputs,
        "activation DQ outputs": activation_dq_outputs,
        "bias sources": bias_sources,
    }
    for label, values in unique_requirements.items():
        if len(values) != EXPECTED_CALL_SITES:
            errors.append(
                f"{label} are shared or missing: {len(values)} unique, "
                f"expected {EXPECTED_CALL_SITES}"
            )
    if invalid_bindings:
        errors.append(f"target Conv Q/DQ bindings are invalid: {invalid_bindings}")
    if len(signatures) != EXPECTED_CALL_SITES:
        errors.append(
            f"Conv signature count is {len(signatures)}, expected {EXPECTED_CALL_SITES}"
        )

    return {
        "schema_version": QDQ_SCHEMA_VERSION,
        "graph_kind": graph_kind,
        "passed": not errors,
        "errors": errors,
        "logical_conv_count": len(target_module_names),
        "target_conv_call_site_count": len(target_convs),
        "activation_quantize_count": len(activation_q),
        "activation_dequantize_count": len(activation_dq),
        "weight_quantize_count": len(weight_q),
        "weight_dequantize_count": len(weight_dq),
        "unique_weight_source_count": len(weight_sources),
        "unique_weight_quantize_output_count": len(weight_q_outputs),
        "unique_weight_dequantize_output_count": len(weight_dq_outputs),
        "unique_bias_count": len(bias_sources),
        "call_sites_per_logical_conv": per_module_counts,
        "call_site_names": [node.name for node in target_convs],
        "conv_signatures": signatures,
        "signature_ids": sorted({value["signature_id"] for value in signatures}),
        "unsupported_quantized_conv_nodes": qlinear_convs,
        "target_cast_nodes": target_casts,
        "invalid_bindings": invalid_bindings,
    }


def rewrite_onnx_with_int8_qdq(
    *,
    source_path: str | Path,
    destination_path: str | Path,
    graph_kind: str,
    target_module_names: tuple[str, ...],
    activation_scales: dict[str, float],
) -> dict[str, Any]:
    """Rewrite selected Conv3d call sites with independent explicit INT8 Q/DQ."""

    if graph_kind not in {"initial", "steady"}:
        raise ValueError("graph_kind must be 'initial' or 'steady'")
    if len(target_module_names) != EXPECTED_LOGICAL_CONVS:
        raise ValueError(
            f"expected {EXPECTED_LOGICAL_CONVS} target modules, "
            f"got {len(target_module_names)}"
        )
    if len(set(target_module_names)) != len(target_module_names):
        raise ValueError("target module names must be unique")
    if set(activation_scales) != set(target_module_names):
        missing = sorted(set(target_module_names) - set(activation_scales))
        extra = sorted(set(activation_scales) - set(target_module_names))
        raise ValueError(
            f"activation scale keys do not match targets; missing={missing}, extra={extra}"
        )

    onnx, np, _TensorProto, helpers = _lazy_onnx()
    helper, numpy_helper = helpers
    source = Path(source_path)
    destination = Path(destination_path)
    model = onnx.load(str(source), load_external_data=True)
    source_opset = _opset_version(model)
    if source_opset != QDQ_OPSET:
        raise ValueError(
            f"explicit Q/DQ rewrite requires a real ONNX opset {QDQ_OPSET} "
            f"source graph, got opset {source_opset}; re-export the FP16 graph"
        )

    source_initializers = _initializer_map(model)
    resolved_weights = _resolve_weight_names(
        initializer_names=set(source_initializers),
        target_module_names=target_module_names,
    )
    module_by_weight = {
        weight_name: module_name
        for module_name, weight_name in resolved_weights.items()
    }
    calls_by_module = {module_name: 0 for module_name in target_module_names}
    module_weights: dict[str, Any] = {}
    module_weight_scales: dict[str, Any] = {}
    weight_scales: dict[str, dict[str, list[float]]] = {}
    for module_name, weight_name in resolved_weights.items():
        weight = numpy_helper.to_array(source_initializers[weight_name]).astype(
            np.float32
        )
        if weight.ndim != 5:
            raise ValueError(
                f"target {module_name!r} weight must be rank-5, got {weight.shape}"
            )
        channel_max = np.max(np.abs(weight), axis=(1, 2, 3, 4))
        raw_scale = np.maximum(channel_max / INT8_MAX, MIN_QUANT_SCALE).astype(
            np.float32
        )
        effective_scale = np.maximum(
            raw_scale,
            np.finfo(np.float16).tiny,
        ).astype(np.float16)
        module_weights[module_name] = weight.astype(np.float16)
        module_weight_scales[module_name] = effective_scale
        weight_scales[module_name] = {
            "raw_fp32": raw_scale.tolist(),
            "effective_fp16": effective_scale.tolist(),
        }

    new_nodes: list[Any] = []
    new_initializers: list[Any] = []
    weight_qdq_paths: dict[str, dict[str, str]] = {}
    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            new_nodes.append(copy.deepcopy(node))
            continue
        module_name = module_by_weight.get(node.input[1])
        if module_name is None:
            new_nodes.append(copy.deepcopy(node))
            continue
        if len(node.input) < 3 or not node.input[2]:
            raise ValueError(f"target Conv {node.name!r} has no static bias")
        bias_initializer = source_initializers.get(node.input[2])
        if bias_initializer is None:
            raise ValueError(
                f"target Conv bias initializer {node.input[2]!r} is missing"
            )
        bias_fp16 = numpy_helper.to_array(bias_initializer).astype(np.float16)
        call_index = calls_by_module[module_name]
        calls_by_module[module_name] += 1
        activation_scale = _as_positive_scale(
            activation_scales[module_name],
            module_name=module_name,
        )
        activation_scale_fp16 = np.asarray(
            max(activation_scale, float(np.finfo(np.float16).tiny)),
            dtype=np.float16,
        )
        (
            qdq_nodes,
            qdq_initializers,
            activation_dq,
            weight_dq,
            bias_name,
            names,
        ) = _make_qdq_call_site(
            helper=helper,
            numpy_helper=numpy_helper,
            np=np,
            graph_kind=graph_kind,
            module_name=module_name,
            call_index=call_index,
            activation_input=node.input[0],
            weight_fp16=module_weights[module_name],
            weight_scale_fp16=module_weight_scales[module_name],
            activation_scale_fp16=activation_scale_fp16,
            bias_fp16=bias_fp16,
        )
        new_nodes.extend(qdq_nodes)
        new_initializers.extend(qdq_initializers)
        rewritten = copy.deepcopy(node)
        rewritten.input[0] = activation_dq
        rewritten.input[1] = weight_dq
        rewritten.input[2] = bias_name
        rewritten.name = f"int8/{graph_kind}/{module_name}/call_{call_index}"
        new_nodes.append(rewritten)
        weight_qdq_paths[rewritten.name] = names

    invalid_counts = {
        module_name: count
        for module_name, count in calls_by_module.items()
        if count != EXPECTED_CALLS_PER_LOGICAL_CONV
    }
    if invalid_counts:
        raise ValueError(
            "FP16 ONNX did not unroll every target Conv exactly three times: "
            f"{invalid_counts}"
        )

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_initializers)
    _remove_unused_initializers(model)
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(
        model,
        check_type=True,
        strict_mode=True,
        data_prop=False,
    )
    onnx.checker.check_model(model, full_check=True)
    audit = audit_qdq_model(
        model,
        graph_kind=graph_kind,
        target_module_names=target_module_names,
    )
    if not audit["passed"]:
        raise ValueError(f"explicit Q/DQ structural audit failed: {audit['errors']}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    onnx.save(model, str(temporary))
    temporary.replace(destination)
    return {
        **audit,
        "source": str(source),
        "destination": str(destination),
        "opset": QDQ_OPSET,
        "weight_qdq_paths": weight_qdq_paths,
        "activation_scales": {
            name: {
                "raw_fp32": _as_positive_scale(value, module_name=name),
                "effective_fp16": float(
                    np.asarray(
                        max(
                            _as_positive_scale(value, module_name=name),
                            float(np.finfo(np.float16).tiny),
                        ),
                        dtype=np.float16,
                    )
                ),
            }
            for name, value in activation_scales.items()
        },
        "weight_scales": weight_scales,
        "weight_quantization": (
            "independent_fp16_constant_to_symmetric_signed_int8_"
            "per_output_channel_axis_0"
        ),
        "activation_quantization": "symmetric_signed_int8_per_tensor",
    }


def write_int8_conv3d_probe(
    *,
    path: str | Path,
    signature: dict[str, Any],
    signature_id: str,
) -> str:
    """Write one probe using the exact v3 Q/DQ topology and Conv signature."""

    onnx, np, TensorProto, helpers = _lazy_onnx()
    helper, numpy_helper = helpers
    if conv_signature_id(signature) != signature_id:
        raise ValueError("probe signature_id does not match the Conv signature")
    canonical = _canonical_signature(signature)
    weight_shape = tuple(canonical["weight_shape"])
    output_channels = int(weight_shape[0])
    weight_fp16 = np.full(weight_shape, 0.125, dtype=np.float16)
    weight_scale = np.full(
        (output_channels,),
        np.float16(0.125 / INT8_MAX),
        dtype=np.float16,
    )
    graph_kind = f"probe_{signature_id}"
    module_name = "conv"
    qdq_nodes, initializers, activation_dq, weight_dq, bias_name, _names = (
        _make_qdq_call_site(
            helper=helper,
            numpy_helper=numpy_helper,
            np=np,
            graph_kind=graph_kind,
            module_name=module_name,
            call_index=0,
            activation_input="activation",
            weight_fp16=weight_fp16,
            weight_scale_fp16=weight_scale,
            activation_scale_fp16=np.asarray(1.0 / INT8_MAX, dtype=np.float16),
            bias_fp16=np.zeros((output_channels,), dtype=np.float16),
        )
    )
    call_site = f"int8/probe/{signature_id}/call_0"
    conv_attributes = {
        "kernel_shape": canonical["kernel_shape"],
        "pads": canonical["pads"],
        "strides": canonical["strides"],
        "dilations": canonical["dilations"],
        "group": canonical["group"],
    }
    qdq_nodes.append(
        helper.make_node(
            "Conv",
            [activation_dq, weight_dq, bias_name],
            ["output"],
            name=call_site,
            **conv_attributes,
        )
    )
    graph = helper.make_graph(
        qdq_nodes,
        f"sfwan_int8_conv3d_probe_{signature_id}",
        [
            helper.make_tensor_value_info(
                "activation",
                TensorProto.FLOAT16,
                canonical["input_shape"],
            )
        ],
        [
            helper.make_tensor_value_info(
                "output",
                TensorProto.FLOAT16,
                canonical["output_shape"],
            )
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", QDQ_OPSET)],
        producer_name="sglang-jetson-sfwan-qdq-v3-probe",
    )
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    onnx.save(model, str(temporary))
    temporary.replace(destination)
    return call_site


def write_qdq_report(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
