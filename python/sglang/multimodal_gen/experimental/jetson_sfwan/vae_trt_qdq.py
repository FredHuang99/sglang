"""Explicit-Q/DQ ONNX rewriting for the Jetson SFWan VAE.

The build tool exports an FP16 graph that calls the existing SGLang Wan
decoder.  This module inserts input, weight, and trailing output Q/DQ around
only the 28 residual-block Conv3d modules and their 84 unrolled call sites.
ONNX and NumPy are imported lazily so importing the service does not require
the TensorRT build environment.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

QDQ_SCHEMA_VERSION = 5
QDQ_OPSET = 19
QDQ_TOPOLOGY = "fp16_cast_fp32_input_qdq_fp32_conv_output_qdq_fp32_cast_fp16"
WEIGHT_ENCODING_FP32_QDQ = "fp32_qdq"
WEIGHT_ENCODING_PREQUANTIZED_INT8_DQ = "prequantized_int8_dq"
WEIGHT_ENCODINGS = (
    WEIGHT_ENCODING_FP32_QDQ,
    WEIGHT_ENCODING_PREQUANTIZED_INT8_DQ,
)
INT8_MAX = 127.0
MIN_QUANT_SCALE = 1.0e-8
EXPECTED_LOGICAL_CONVS = 28
EXPECTED_CALLS_PER_LOGICAL_CONV = 3
EXPECTED_CALL_SITES = EXPECTED_LOGICAL_CONVS * EXPECTED_CALLS_PER_LOGICAL_CONV
EXPECTED_CONV_SIGNATURES = 9


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


def collect_fp16_target_call_sites(
    *,
    source_path: str | Path,
    graph_kind: str,
    target_module_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Identify the FP16 source nodes corresponding to the v5 INT8 targets.

    The Q/DQ rewriter identifies targets through their static weight
    initializer, not through fragile node-name heuristics.  The diagnostic
    FP16 plan needs the exact same 84-call-site identity so its physical
    TensorRT layers can be compared with the audited INT8 plan.  This helper
    intentionally performs only read-only graph inspection; it does not
    change the v5 topology or any initializer.
    """

    if graph_kind not in {"initial", "steady"}:
        raise ValueError("graph_kind must be 'initial' or 'steady'")
    if len(target_module_names) != EXPECTED_LOGICAL_CONVS:
        raise ValueError(
            f"expected {EXPECTED_LOGICAL_CONVS} target modules, "
            f"got {len(target_module_names)}"
        )
    if len(set(target_module_names)) != len(target_module_names):
        raise ValueError("target module names must be unique")

    onnx, _np, _TensorProto, _helpers = _lazy_onnx()
    source = Path(source_path)
    model = onnx.load(str(source), load_external_data=True)
    if _opset_version(model) != QDQ_OPSET:
        raise ValueError(
            f"FP16 target mapping requires ONNX opset {QDQ_OPSET}: {source}"
        )
    initializers = _initializer_map(model)
    resolved_weights = _resolve_weight_names(
        initializer_names=set(initializers),
        target_module_names=target_module_names,
    )
    module_by_weight = {
        weight_name: module_name
        for module_name, weight_name in resolved_weights.items()
    }
    calls_by_module = {module_name: 0 for module_name in target_module_names}
    call_sites: list[dict[str, Any]] = []
    source_node_names: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        module_name = module_by_weight.get(node.input[1])
        if module_name is None:
            continue
        if not node.name:
            raise ValueError(
                f"target FP16 Conv for {module_name!r} has no ONNX node name"
            )
        if node.name in source_node_names:
            raise ValueError(f"duplicate target FP16 ONNX node name: {node.name!r}")
        source_node_names.add(node.name)
        call_index = calls_by_module[module_name]
        calls_by_module[module_name] += 1
        call_sites.append(
            {
                # Reuse the stable v5 logical identifier so FP16 and INT8
                # catalogs can be compared without a name translation step.
                "logical_call_site": (
                    f"int8/{graph_kind}/{module_name}/call_{call_index}"
                ),
                "module_name": module_name,
                "call_index": call_index,
                "source_onnx_node_name": node.name,
                "source_weight_initializer": node.input[1],
            }
        )

    invalid_counts = {
        module_name: count
        for module_name, count in calls_by_module.items()
        if count != EXPECTED_CALLS_PER_LOGICAL_CONV
    }
    if invalid_counts or len(call_sites) != EXPECTED_CALL_SITES:
        raise ValueError(
            "FP16 ONNX target mapping did not find exactly three calls for "
            f"every residual Conv: counts={invalid_counts}, total={len(call_sites)}"
        )
    if len({record["logical_call_site"] for record in call_sites}) != len(call_sites):
        raise ValueError("FP16 target logical call-site identifiers are not unique")
    return call_sites


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


def _tensor_element_type_map(model: Any) -> dict[str, int]:
    element_types: dict[str, int] = {}
    values = [
        *model.graph.input,
        *model.graph.output,
        *model.graph.value_info,
    ]
    for value in values:
        tensor_type = value.type.tensor_type
        if tensor_type.HasField("elem_type") and int(tensor_type.elem_type) != 0:
            element_types[value.name] = int(tensor_type.elem_type)
    for initializer in model.graph.initializer:
        element_types[initializer.name] = int(initializer.data_type)
    return element_types


def _normalize_rank5_shape(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 5:
        return None
    if any(
        isinstance(dimension, bool) or not isinstance(dimension, int)
        for dimension in value
    ):
        return None
    normalized = [int(dimension) for dimension in value]
    return normalized if all(dimension > 0 for dimension in normalized) else None


def _normalize_call_site_shape_contracts(
    *,
    contracts: dict[str, list[dict[str, list[int]]]] | None,
    target_module_names: tuple[str, ...],
) -> dict[str, list[dict[str, list[int]]]] | None:
    if contracts is None:
        return None
    if set(contracts) != set(target_module_names):
        missing = sorted(set(target_module_names) - set(contracts))
        extra = sorted(set(contracts) - set(target_module_names))
        raise ValueError(
            "Conv shape contract keys do not match targets; "
            f"missing={missing}, extra={extra}"
        )

    normalized: dict[str, list[dict[str, list[int]]]] = {}
    for module_name in target_module_names:
        call_contracts = contracts[module_name]
        if not isinstance(call_contracts, list) or len(call_contracts) != 3:
            raise ValueError(
                f"Conv shape contract for {module_name!r} must contain three calls"
            )
        normalized_calls: list[dict[str, list[int]]] = []
        for call_index, contract in enumerate(call_contracts):
            if not isinstance(contract, dict) or set(contract) != {
                "input_shape",
                "output_shape",
            }:
                raise ValueError(
                    f"Conv shape contract for {module_name!r} call {call_index} "
                    "must contain only input_shape and output_shape"
                )
            input_shape = _normalize_rank5_shape(contract["input_shape"])
            output_shape = _normalize_rank5_shape(contract["output_shape"])
            if input_shape is None or output_shape is None:
                raise ValueError(
                    f"Conv shape contract for {module_name!r} call {call_index} "
                    "must contain positive rank-5 integer shapes"
                )
            normalized_calls.append(
                {
                    "input_shape": input_shape,
                    "output_shape": output_shape,
                }
            )
        normalized[module_name] = normalized_calls
    return normalized


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
    try:
        return bool(
            int(value.size) > 0
            and (value > 0).all()
            and math.isfinite(float(value.max()))
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _has_qdq_contract(node: Any) -> bool:
    return bool(
        len(node.input) == 3
        and all(node.input)
        and len(node.output) == 1
        and node.output[0]
    )


def _make_qdq_call_site(
    *,
    helper: Any,
    numpy_helper: Any,
    TensorProto: Any,
    np: Any,
    graph_kind: str,
    module_name: str,
    call_index: int,
    activation_input: str,
    weight_fp32: Any,
    weight_scale_fp32: Any,
    activation_input_scale_fp32: Any,
    activation_output_scale_fp32: Any,
    bias_fp32: Any,
    weight_encoding: str,
) -> tuple[
    list[Any],
    list[Any],
    list[Any],
    str,
    str,
    str,
    str,
    str,
    dict[str, str],
]:
    """Create one TensorRT-10.3-compatible explicit INT8 Conv call site.

    TensorRT 10.3 only supports FP32 data and scales for Q/DQ layers.  Keep
    the surrounding exported VAE graph in FP16, cast the activation to FP32
    before input Q/DQ, and put a second Q/DQ pair directly after Conv.  The
    trailing Q lets TensorRT fuse an INT8-output convolution instead of
    legally selecting a Float/TF32 Conv followed only by a Cast.
    """

    if weight_encoding not in WEIGHT_ENCODINGS:
        raise ValueError(f"unsupported INT8 weight encoding: {weight_encoding!r}")

    safe_module = _safe_name(module_name)
    prefix = f"sfwan_{graph_kind}_{safe_module}_call_{call_index}"
    activation_fp32 = f"{prefix}_activation_fp32"
    activation_scale_name = f"{prefix}_activation_input_scale"
    activation_zero_name = f"{prefix}_activation_input_zero"
    activation_int8 = f"{prefix}_activation_int8"
    activation_dq = f"{prefix}_activation_dequantized"
    weight_source_name = (
        f"{prefix}_weight_fp32_source"
        if weight_encoding == WEIGHT_ENCODING_FP32_QDQ
        else f"{prefix}_weight_int8_source"
    )
    weight_scale_name = f"{prefix}_weight_scale"
    weight_zero_name = f"{prefix}_weight_zero"
    weight_int8 = f"{prefix}_weight_int8"
    weight_dq = f"{prefix}_weight_dequantized"
    bias_name = f"{prefix}_bias_fp32"
    conv_output_fp32 = f"{prefix}_conv_output_fp32"
    output_scale_name = f"{prefix}_activation_output_scale"
    output_zero_name = f"{prefix}_activation_output_zero"
    output_int8 = f"{prefix}_conv_output_int8"
    output_dq = f"{prefix}_conv_output_dequantized"

    weight_source = np.ascontiguousarray(weight_fp32, dtype=np.float32)
    if weight_encoding == WEIGHT_ENCODING_PREQUANTIZED_INT8_DQ:
        scale_shape = (int(weight_scale_fp32.shape[0]),) + (1,) * (
            int(weight_source.ndim) - 1
        )
        weight_source = np.clip(
            np.rint(weight_source / weight_scale_fp32.reshape(scale_shape)),
            -int(INT8_MAX),
            int(INT8_MAX),
        ).astype(np.int8)

    initializers = [
        numpy_helper.from_array(
            weight_source,
            name=weight_source_name,
        ),
        numpy_helper.from_array(
            np.ascontiguousarray(weight_scale_fp32, dtype=np.float32),
            name=weight_scale_name,
        ),
        numpy_helper.from_array(
            np.zeros(weight_scale_fp32.shape, dtype=np.int8),
            name=weight_zero_name,
        ),
        numpy_helper.from_array(
            np.asarray(activation_input_scale_fp32, dtype=np.float32),
            name=activation_scale_name,
        ),
        numpy_helper.from_array(
            np.asarray(0, dtype=np.int8),
            name=activation_zero_name,
        ),
        numpy_helper.from_array(
            np.ascontiguousarray(bias_fp32, dtype=np.float32),
            name=bias_name,
        ),
        numpy_helper.from_array(
            np.asarray(activation_output_scale_fp32, dtype=np.float32),
            name=output_scale_name,
        ),
        numpy_helper.from_array(
            np.asarray(0, dtype=np.int8),
            name=output_zero_name,
        ),
    ]
    pre_nodes = [
        helper.make_node(
            "Cast",
            [activation_input],
            [activation_fp32],
            name=(
                f"qdq/{graph_kind}/activation/{module_name}/"
                f"call_{call_index}/cast_to_fp32"
            ),
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "QuantizeLinear",
            [activation_fp32, activation_scale_name, activation_zero_name],
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
    ]
    if weight_encoding == WEIGHT_ENCODING_FP32_QDQ:
        pre_nodes.append(
            helper.make_node(
                "QuantizeLinear",
                [weight_source_name, weight_scale_name, weight_zero_name],
                [weight_int8],
                name=(
                    f"qdq/{graph_kind}/weight/{module_name}/call_{call_index}/quantize"
                ),
                axis=0,
            )
        )
        weight_dq_input = weight_int8
    else:
        weight_dq_input = weight_source_name
    pre_nodes.append(
        helper.make_node(
            "DequantizeLinear",
            [weight_dq_input, weight_scale_name, weight_zero_name],
            [weight_dq],
            name=(
                f"qdq/{graph_kind}/weight/{module_name}/call_{call_index}/dequantize"
            ),
            axis=0,
        )
    )
    post_nodes = [
        helper.make_node(
            "QuantizeLinear",
            [conv_output_fp32, output_scale_name, output_zero_name],
            [output_int8],
            name=(f"qdq/{graph_kind}/output/{module_name}/call_{call_index}/quantize"),
        ),
        helper.make_node(
            "DequantizeLinear",
            [output_int8, output_scale_name, output_zero_name],
            [output_dq],
            name=(
                f"qdq/{graph_kind}/output/{module_name}/call_{call_index}/dequantize"
            ),
        ),
    ]
    names = {
        "activation_cast_output": activation_fp32,
        "activation_scale": activation_scale_name,
        "activation_zero": activation_zero_name,
        "weight_source": weight_source_name,
        "weight_scale": weight_scale_name,
        "weight_zero": weight_zero_name,
        "weight_encoding": weight_encoding,
        "bias": bias_name,
        "conv_output_fp32": conv_output_fp32,
        "output_scale": output_scale_name,
        "output_zero": output_zero_name,
        "output_quantized": output_int8,
        "output_dequantized": output_dq,
    }
    return (
        pre_nodes,
        post_nodes,
        initializers,
        activation_dq,
        weight_dq,
        bias_name,
        conv_output_fp32,
        output_dq,
        names,
    )


def audit_qdq_model(
    model: Any,
    *,
    graph_kind: str,
    target_module_names: tuple[str, ...],
    shape_overrides: dict[str, dict[str, list[int]]] | None = None,
    weight_encoding: str | None = None,
) -> dict[str, Any]:
    """Validate the complete TensorRT-10.3-compatible v5 Q/DQ contract."""

    _onnx, _np, TensorProto, helpers = _lazy_onnx()
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
    output_q = [
        node
        for node in model.graph.node
        if node.op_type == "QuantizeLinear"
        and node.name.startswith(f"qdq/{graph_kind}/output/")
    ]
    output_dq = [
        node
        for node in model.graph.node
        if node.op_type == "DequantizeLinear"
        and node.name.startswith(f"qdq/{graph_kind}/output/")
    ]
    activation_casts = [
        node
        for node in model.graph.node
        if node.op_type == "Cast"
        and node.name.startswith(f"qdq/{graph_kind}/activation/")
        and node.name.endswith("/cast_to_fp32")
    ]
    output_casts = [
        node
        for node in model.graph.node
        if node.op_type == "Cast"
        and node.name.startswith(f"qdq/{graph_kind}/output/")
        and node.name.endswith("/cast_to_fp16")
    ]
    all_target_casts = [
        node
        for node in model.graph.node
        if node.op_type == "Cast" and node.name.startswith(f"qdq/{graph_kind}/")
    ]
    expected_cast_names = {node.name for node in (*activation_casts, *output_casts)}
    unexpected_target_casts = sorted(
        node.name for node in all_target_casts if node.name not in expected_cast_names
    )
    initializers = _initializer_map(model)
    producers = {
        output_name: node
        for node in model.graph.node
        for output_name in node.output
        if output_name
    }
    consumers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append(node)
    shapes = _tensor_shape_map(model)
    element_types = _tensor_element_type_map(model)
    per_module_counts = {
        module_name: sum(
            node.name.startswith(f"{target_call_prefix}{module_name}/call_")
            for node in target_convs
        )
        for module_name in target_module_names
    }

    errors: list[str] = []
    detected_weight_encoding: str | None = None
    if len(weight_q) == EXPECTED_CALL_SITES:
        detected_weight_encoding = WEIGHT_ENCODING_FP32_QDQ
    elif len(weight_q) == 0:
        detected_weight_encoding = WEIGHT_ENCODING_PREQUANTIZED_INT8_DQ
    else:
        errors.append(
            "weight encoding is mixed or incomplete: "
            f"{len(weight_q)} QuantizeLinear nodes"
        )
    if weight_encoding is not None and weight_encoding not in WEIGHT_ENCODINGS:
        raise ValueError(f"unsupported INT8 weight encoding: {weight_encoding!r}")
    if (
        weight_encoding is not None
        and detected_weight_encoding is not None
        and weight_encoding != detected_weight_encoding
    ):
        errors.append(
            f"weight encoding is {detected_weight_encoding}, expected {weight_encoding}"
        )
    effective_weight_encoding = weight_encoding or detected_weight_encoding
    if len(target_module_names) != EXPECTED_LOGICAL_CONVS:
        errors.append(
            f"logical target count is {len(target_module_names)}, "
            f"expected {EXPECTED_LOGICAL_CONVS}"
        )
    expected_counts = {
        "target Conv": len(target_convs),
        "activation QuantizeLinear": len(activation_q),
        "activation DequantizeLinear": len(activation_dq),
        "activation FP16-to-FP32 Cast": len(activation_casts),
        "weight DequantizeLinear": len(weight_dq),
        "output QuantizeLinear": len(output_q),
        "output DequantizeLinear": len(output_dq),
        "output FP32-to-FP16 Cast": len(output_casts),
    }
    if effective_weight_encoding == WEIGHT_ENCODING_FP32_QDQ:
        expected_counts["weight QuantizeLinear"] = len(weight_q)
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
    if unexpected_target_casts:
        errors.append(
            "target Q/DQ paths contain unexpected Cast nodes: "
            f"{unexpected_target_casts}"
        )
    target_conv_names = {node.name for node in target_convs}
    if shape_overrides is not None and set(shape_overrides) != target_conv_names:
        missing = sorted(target_conv_names - set(shape_overrides))
        extra = sorted(set(shape_overrides) - target_conv_names)
        errors.append(
            "captured Conv shape call sites do not match rewritten targets; "
            f"missing={missing}, extra={extra}"
        )

    invalid_bindings: list[str] = []
    weight_sources: set[str] = set()
    weight_q_outputs: set[str] = set()
    weight_dq_outputs: set[str] = set()
    activation_q_outputs: set[str] = set()
    activation_dq_outputs: set[str] = set()
    activation_cast_outputs: set[str] = set()
    output_q_outputs: set[str] = set()
    output_dq_outputs: set[str] = set()
    output_cast_outputs: set[str] = set()
    activation_scale_sources: set[str] = set()
    output_scale_sources: set[str] = set()
    bias_sources: set[str] = set()
    signatures: list[dict[str, Any]] = []
    shape_sources: dict[str, str] = {}

    for conv in target_convs:
        if len(conv.input) < 3 or not conv.input[2]:
            invalid_bindings.append(f"{conv.name}:missing_bias")
            continue
        if len(conv.output) != 1 or not conv.output[0]:
            invalid_bindings.append(f"{conv.name}:invalid_output_count")
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
        if not _has_qdq_contract(activation_dequantize):
            invalid_bindings.append(f"{conv.name}:activation_dq_contract_invalid")
            continue
        if not _has_qdq_contract(weight_dequantize):
            invalid_bindings.append(f"{conv.name}:weight_dq_contract_invalid")
            continue
        activation_quantize = producers.get(activation_dequantize.input[0])
        if (
            activation_quantize is None
            or activation_quantize.op_type != "QuantizeLinear"
        ):
            invalid_bindings.append(f"{conv.name}:activation_missing_q")
            continue
        if not _has_qdq_contract(activation_quantize):
            invalid_bindings.append(f"{conv.name}:activation_q_contract_invalid")
            continue
        weight_quantize = producers.get(weight_dequantize.input[0])
        if effective_weight_encoding == WEIGHT_ENCODING_FP32_QDQ:
            if weight_quantize is None or weight_quantize.op_type != "QuantizeLinear":
                invalid_bindings.append(f"{conv.name}:weight_missing_q")
                continue
            if not _has_qdq_contract(weight_quantize):
                invalid_bindings.append(f"{conv.name}:weight_q_contract_invalid")
                continue
        elif effective_weight_encoding == WEIGHT_ENCODING_PREQUANTIZED_INT8_DQ:
            if weight_quantize is not None:
                invalid_bindings.append(f"{conv.name}:prequantized_weight_has_q")
                continue
        else:
            invalid_bindings.append(f"{conv.name}:weight_encoding_invalid")
            continue
        activation_cast = producers.get(activation_quantize.input[0])
        if activation_cast is None or activation_cast.op_type != "Cast":
            invalid_bindings.append(f"{conv.name}:activation_missing_fp32_cast")
            continue
        if (
            len(activation_cast.input) != 1
            or not activation_cast.input[0]
            or len(activation_cast.output) != 1
            or not activation_cast.output[0]
        ):
            invalid_bindings.append(f"{conv.name}:activation_cast_contract_invalid")
            continue
        conv_output_consumers = consumers.get(conv.output[0], [])
        if len(conv_output_consumers) != 1:
            invalid_bindings.append(
                f"{conv.name}:conv_output_consumer_count_{len(conv_output_consumers)}"
            )
            continue
        output_quantize = conv_output_consumers[0]
        if output_quantize.op_type != "QuantizeLinear":
            invalid_bindings.append(f"{conv.name}:output_missing_q")
            continue
        if not _has_qdq_contract(output_quantize):
            invalid_bindings.append(f"{conv.name}:output_q_contract_invalid")
            continue
        output_quantize_consumers = consumers.get(output_quantize.output[0], [])
        if len(output_quantize_consumers) != 1:
            invalid_bindings.append(
                f"{conv.name}:output_q_consumer_count_{len(output_quantize_consumers)}"
            )
            continue
        output_dequantize = output_quantize_consumers[0]
        if output_dequantize.op_type != "DequantizeLinear":
            invalid_bindings.append(f"{conv.name}:output_missing_dq")
            continue
        if not _has_qdq_contract(output_dequantize):
            invalid_bindings.append(f"{conv.name}:output_dq_contract_invalid")
            continue
        output_dequantize_consumers = consumers.get(output_dequantize.output[0], [])
        if len(output_dequantize_consumers) != 1:
            invalid_bindings.append(
                f"{conv.name}:output_dq_consumer_count_"
                f"{len(output_dequantize_consumers)}"
            )
            continue
        output_cast = output_dequantize_consumers[0]
        if output_cast.op_type != "Cast":
            invalid_bindings.append(f"{conv.name}:output_missing_fp16_cast")
            continue

        call_suffix = conv.name.removeprefix(target_call_prefix)
        expected_activation_cast_name = (
            f"qdq/{graph_kind}/activation/{call_suffix}/cast_to_fp32"
        )
        expected_activation_q_name = (
            f"qdq/{graph_kind}/activation/{call_suffix}/quantize"
        )
        expected_activation_dq_name = (
            f"qdq/{graph_kind}/activation/{call_suffix}/dequantize"
        )
        expected_weight_q_name = f"qdq/{graph_kind}/weight/{call_suffix}/quantize"
        expected_weight_dq_name = f"qdq/{graph_kind}/weight/{call_suffix}/dequantize"
        expected_output_q_name = f"qdq/{graph_kind}/output/{call_suffix}/quantize"
        expected_output_dq_name = f"qdq/{graph_kind}/output/{call_suffix}/dequantize"
        expected_output_cast_name = (
            f"qdq/{graph_kind}/output/{call_suffix}/cast_to_fp16"
        )
        if activation_cast.name != expected_activation_cast_name:
            invalid_bindings.append(f"{conv.name}:activation_cast_name_invalid")
        if output_cast.name != expected_output_cast_name:
            invalid_bindings.append(f"{conv.name}:output_cast_name_invalid")
        if activation_quantize.name != expected_activation_q_name:
            invalid_bindings.append(f"{conv.name}:activation_q_name_invalid")
        if activation_dequantize.name != expected_activation_dq_name:
            invalid_bindings.append(f"{conv.name}:activation_dq_name_invalid")
        if (
            effective_weight_encoding == WEIGHT_ENCODING_FP32_QDQ
            and weight_quantize.name != expected_weight_q_name
        ):
            invalid_bindings.append(f"{conv.name}:weight_q_name_invalid")
        if weight_dequantize.name != expected_weight_dq_name:
            invalid_bindings.append(f"{conv.name}:weight_dq_name_invalid")
        if output_quantize.name != expected_output_q_name:
            invalid_bindings.append(f"{conv.name}:output_q_name_invalid")
        if output_dequantize.name != expected_output_dq_name:
            invalid_bindings.append(f"{conv.name}:output_dq_name_invalid")
        if int(_attribute_map(activation_cast, helper).get("to", -1)) != int(
            TensorProto.FLOAT
        ):
            invalid_bindings.append(f"{conv.name}:activation_cast_not_fp32")
        if int(_attribute_map(output_cast, helper).get("to", -1)) != int(
            TensorProto.FLOAT16
        ):
            invalid_bindings.append(f"{conv.name}:output_cast_not_fp16")
        if (
            len(output_cast.input) != 1
            or output_cast.input[0] != output_dequantize.output[0]
            or len(output_cast.output) != 1
            or not output_cast.output[0]
        ):
            invalid_bindings.append(f"{conv.name}:output_cast_contract_invalid")
            continue

        activation_cast_outputs.add(activation_cast.output[0])
        activation_q_outputs.add(activation_quantize.output[0])
        activation_dq_outputs.add(activation_dequantize.output[0])
        if weight_quantize is not None:
            weight_q_outputs.add(weight_quantize.output[0])
        weight_dq_outputs.add(weight_dequantize.output[0])
        weight_source_name = (
            weight_quantize.input[0]
            if weight_quantize is not None
            else weight_dequantize.input[0]
        )
        bias_name = conv.input[2]
        weight_sources.add(weight_source_name)
        bias_sources.add(bias_name)
        output_q_outputs.add(output_quantize.output[0])
        output_dq_outputs.add(output_dequantize.output[0])
        output_cast_outputs.add(output_cast.output[0])
        activation_scale_sources.add(activation_quantize.input[1])
        output_scale_sources.add(output_quantize.input[1])

        activation_scale = initializers.get(activation_quantize.input[1])
        activation_zero = initializers.get(activation_quantize.input[2])
        weight_source = initializers.get(weight_source_name)
        weight_scale = initializers.get(weight_dequantize.input[1])
        weight_zero = initializers.get(weight_dequantize.input[2])
        output_scale = initializers.get(output_quantize.input[1])
        output_zero = initializers.get(output_quantize.input[2])
        bias = initializers.get(bias_name)
        binding_errors: list[str] = []
        expected_element_types = {
            activation_cast.input[0]: TensorProto.FLOAT16,
            activation_cast.output[0]: TensorProto.FLOAT,
            activation_quantize.output[0]: TensorProto.INT8,
            activation_dequantize.output[0]: TensorProto.FLOAT,
            weight_dequantize.output[0]: TensorProto.FLOAT,
            bias_name: TensorProto.FLOAT,
            conv.output[0]: TensorProto.FLOAT,
            output_quantize.output[0]: TensorProto.INT8,
            output_dequantize.output[0]: TensorProto.FLOAT,
            output_cast.output[0]: TensorProto.FLOAT16,
        }
        if weight_quantize is not None:
            expected_element_types[weight_quantize.input[0]] = TensorProto.FLOAT
            expected_element_types[weight_quantize.output[0]] = TensorProto.INT8
        else:
            expected_element_types[weight_dequantize.input[0]] = TensorProto.INT8
        for tensor_name, expected_type in expected_element_types.items():
            actual_type = element_types.get(tensor_name)
            if actual_type is not None and actual_type != int(expected_type):
                binding_errors.append(
                    f"tensor_type_invalid:{tensor_name}:"
                    f"actual={actual_type}:expected={int(expected_type)}"
                )
        if list(activation_quantize.input[1:]) != list(activation_dequantize.input[1:]):
            binding_errors.append("activation_q_dq_parameters_differ")
        if weight_quantize is not None and list(weight_quantize.input[1:]) != list(
            weight_dequantize.input[1:]
        ):
            binding_errors.append("weight_q_dq_parameters_differ")
        if (
            weight_quantize is not None and _axis(weight_quantize, helper) != 0
        ) or _axis(weight_dequantize, helper) != 0:
            binding_errors.append("weight_axis_not_zero")
        if list(output_quantize.input[1:]) != list(output_dequantize.input[1:]):
            binding_errors.append("output_q_dq_parameters_differ")
        if activation_scale is None:
            binding_errors.append("activation_scale_missing")
        else:
            activation_scale_array = numpy_helper.to_array(activation_scale)
            if (
                int(activation_scale.data_type) != int(TensorProto.FLOAT)
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
        elif numpy_helper.to_array(activation_zero).ndim != 0:
            binding_errors.append("activation_zero_not_scalar")
        if output_scale is None:
            binding_errors.append("output_scale_missing")
        else:
            output_scale_array = numpy_helper.to_array(output_scale)
            if (
                int(output_scale.data_type) != int(TensorProto.FLOAT)
                or output_scale_array.ndim != 0
                or not _is_positive_finite(output_scale_array)
            ):
                binding_errors.append("output_scale_invalid")
        if not _is_zero_int8(
            output_zero,
            TensorProto=TensorProto,
            numpy_helper=numpy_helper,
        ):
            binding_errors.append("output_zero_invalid")
        elif numpy_helper.to_array(output_zero).ndim != 0:
            binding_errors.append("output_zero_not_scalar")
        if weight_source is None:
            binding_errors.append("weight_source_missing")
            weight_shape: list[int] = []
        else:
            weight_shape = [int(value) for value in weight_source.dims]
            expected_weight_type = (
                TensorProto.FLOAT
                if effective_weight_encoding == WEIGHT_ENCODING_FP32_QDQ
                else TensorProto.INT8
            )
            if (
                int(weight_source.data_type) != int(expected_weight_type)
                or len(weight_shape) != 5
            ):
                binding_errors.append(
                    "weight_source_not_rank5_"
                    + (
                        "fp32"
                        if effective_weight_encoding == WEIGHT_ENCODING_FP32_QDQ
                        else "int8"
                    )
                )
        if weight_scale is None:
            binding_errors.append("weight_scale_missing")
        else:
            weight_scale_array = numpy_helper.to_array(weight_scale)
            expected_channels = weight_shape[0] if weight_shape else -1
            if (
                int(weight_scale.data_type) != int(TensorProto.FLOAT)
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
        elif weight_scale is not None and (
            numpy_helper.to_array(weight_zero).shape
            != numpy_helper.to_array(weight_scale).shape
        ):
            binding_errors.append("weight_zero_shape_invalid")
        if (
            bias is None
            or int(bias.data_type) != int(TensorProto.FLOAT)
            or len(bias.dims) != 1
            or (weight_shape and int(bias.dims[0]) != weight_shape[0])
        ):
            binding_errors.append("bias_invalid")
        if binding_errors:
            invalid_bindings.extend(f"{conv.name}:{value}" for value in binding_errors)
            continue

        attributes = _attribute_map(conv, helper)
        inferred_input_shape = shapes.get(activation_cast.input[0])
        inferred_output_shape = shapes.get(output_cast.output[0])
        override = (
            shape_overrides.get(conv.name) if shape_overrides is not None else None
        )
        captured_input_shape = (
            _normalize_rank5_shape(override.get("input_shape"))
            if isinstance(override, dict)
            else None
        )
        captured_output_shape = (
            _normalize_rank5_shape(override.get("output_shape"))
            if isinstance(override, dict)
            else None
        )
        if override is not None and (
            captured_input_shape is None or captured_output_shape is None
        ):
            invalid_bindings.append(f"{conv.name}:captured_shape_invalid")
        if (
            inferred_input_shape is not None
            and captured_input_shape is not None
            and inferred_input_shape != captured_input_shape
        ):
            invalid_bindings.append(
                f"{conv.name}:input_shape_mismatch:"
                f"onnx={inferred_input_shape}:captured={captured_input_shape}"
            )
        if (
            inferred_output_shape is not None
            and captured_output_shape is not None
            and inferred_output_shape != captured_output_shape
        ):
            invalid_bindings.append(
                f"{conv.name}:output_shape_mismatch:"
                f"onnx={inferred_output_shape}:captured={captured_output_shape}"
            )
        input_shape = captured_input_shape or inferred_input_shape
        output_shape = captured_output_shape or inferred_output_shape
        if input_shape is None or output_shape is None:
            invalid_bindings.append(f"{conv.name}:static_shape_missing")
            continue
        shape_sources[conv.name] = (
            "captured" if captured_input_shape is not None else "onnx_inferred"
        )
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
        "weight DQ outputs": weight_dq_outputs,
        "activation Q outputs": activation_q_outputs,
        "activation DQ outputs": activation_dq_outputs,
        "activation Cast outputs": activation_cast_outputs,
        "activation scale sources": activation_scale_sources,
        "output Q outputs": output_q_outputs,
        "output DQ outputs": output_dq_outputs,
        "output scale sources": output_scale_sources,
        "output Cast outputs": output_cast_outputs,
        "bias sources": bias_sources,
    }
    if effective_weight_encoding == WEIGHT_ENCODING_FP32_QDQ:
        unique_requirements["weight Q outputs"] = weight_q_outputs
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
        "qdq_topology": QDQ_TOPOLOGY,
        "weight_encoding": effective_weight_encoding,
        "graph_kind": graph_kind,
        "passed": not errors,
        "errors": errors,
        "logical_conv_count": len(target_module_names),
        "target_conv_call_site_count": len(target_convs),
        "activation_quantize_count": len(activation_q),
        "activation_dequantize_count": len(activation_dq),
        "activation_cast_count": len(activation_casts),
        "weight_quantize_count": len(weight_q),
        "weight_dequantize_count": len(weight_dq),
        "output_quantize_count": len(output_q),
        "output_dequantize_count": len(output_dq),
        "output_cast_count": len(output_casts),
        "unique_weight_source_count": len(weight_sources),
        "unique_weight_quantize_output_count": len(weight_q_outputs),
        "unique_weight_dequantize_output_count": len(weight_dq_outputs),
        "unique_output_quantize_output_count": len(output_q_outputs),
        "unique_output_dequantize_output_count": len(output_dq_outputs),
        "unique_bias_count": len(bias_sources),
        "call_sites_per_logical_conv": per_module_counts,
        "call_site_names": [node.name for node in target_convs],
        "conv_signatures": signatures,
        "shape_sources": shape_sources,
        "signature_ids": sorted({value["signature_id"] for value in signatures}),
        "unsupported_quantized_conv_nodes": qlinear_convs,
        "activation_cast_nodes": [node.name for node in activation_casts],
        "output_cast_nodes": [node.name for node in output_casts],
        "unexpected_target_cast_nodes": unexpected_target_casts,
        "invalid_bindings": invalid_bindings,
    }


def rewrite_onnx_with_int8_qdq(
    *,
    source_path: str | Path,
    destination_path: str | Path,
    graph_kind: str,
    target_module_names: tuple[str, ...],
    activation_input_scales: dict[str, float],
    activation_output_scales: dict[str, float],
    weight_encoding: str = WEIGHT_ENCODING_FP32_QDQ,
    call_site_shape_contracts: dict[str, list[dict[str, list[int]]]] | None = None,
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
    if weight_encoding not in WEIGHT_ENCODINGS:
        raise ValueError(f"unsupported INT8 weight encoding: {weight_encoding!r}")
    for scale_kind, scales in (
        ("input", activation_input_scales),
        ("output", activation_output_scales),
    ):
        if set(scales) != set(target_module_names):
            missing = sorted(set(target_module_names) - set(scales))
            extra = sorted(set(scales) - set(target_module_names))
            raise ValueError(
                f"activation {scale_kind} scale keys do not match targets; "
                f"missing={missing}, extra={extra}"
            )
    normalized_shape_contracts = _normalize_call_site_shape_contracts(
        contracts=call_site_shape_contracts,
        target_module_names=target_module_names,
    )

    onnx, np, TensorProto, helpers = _lazy_onnx()
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
        effective_scale = np.maximum(raw_scale, MIN_QUANT_SCALE).astype(np.float32)
        module_weights[module_name] = weight.astype(np.float32)
        module_weight_scales[module_name] = effective_scale
        weight_scales[module_name] = {
            "raw_fp32": raw_scale.tolist(),
            "effective_fp32": effective_scale.tolist(),
        }

    new_nodes: list[Any] = []
    new_initializers: list[Any] = []
    weight_qdq_paths: dict[str, dict[str, str]] = {}
    shape_overrides: dict[str, dict[str, list[int]]] = {}
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
        if len(node.output) != 1 or not node.output[0]:
            raise ValueError(f"target Conv {node.name!r} must have exactly one output")
        bias_initializer = source_initializers.get(node.input[2])
        if bias_initializer is None:
            raise ValueError(
                f"target Conv bias initializer {node.input[2]!r} is missing"
            )
        bias_fp32 = numpy_helper.to_array(bias_initializer).astype(np.float32)
        call_index = calls_by_module[module_name]
        calls_by_module[module_name] += 1
        activation_input_scale = _as_positive_scale(
            activation_input_scales[module_name],
            module_name=f"{module_name}:input",
        )
        activation_output_scale = _as_positive_scale(
            activation_output_scales[module_name],
            module_name=f"{module_name}:output",
        )
        activation_input_scale_fp32 = np.asarray(
            max(activation_input_scale, MIN_QUANT_SCALE),
            dtype=np.float32,
        )
        activation_output_scale_fp32 = np.asarray(
            max(activation_output_scale, MIN_QUANT_SCALE),
            dtype=np.float32,
        )
        (
            qdq_pre_nodes,
            qdq_post_nodes,
            qdq_initializers,
            activation_dq,
            weight_dq,
            bias_name,
            conv_output_fp32,
            output_dq,
            names,
        ) = _make_qdq_call_site(
            helper=helper,
            numpy_helper=numpy_helper,
            TensorProto=TensorProto,
            np=np,
            graph_kind=graph_kind,
            module_name=module_name,
            call_index=call_index,
            activation_input=node.input[0],
            weight_fp32=module_weights[module_name],
            weight_scale_fp32=module_weight_scales[module_name],
            activation_input_scale_fp32=activation_input_scale_fp32,
            activation_output_scale_fp32=activation_output_scale_fp32,
            bias_fp32=bias_fp32,
            weight_encoding=weight_encoding,
        )
        new_nodes.extend(qdq_pre_nodes)
        new_initializers.extend(qdq_initializers)
        rewritten = copy.deepcopy(node)
        rewritten.input[0] = activation_dq
        rewritten.input[1] = weight_dq
        rewritten.input[2] = bias_name
        rewritten.name = f"int8/{graph_kind}/{module_name}/call_{call_index}"
        original_output = rewritten.output[0]
        rewritten.output[0] = conv_output_fp32
        new_nodes.append(rewritten)
        new_nodes.extend(qdq_post_nodes)
        output_cast_name = (
            f"qdq/{graph_kind}/output/{module_name}/call_{call_index}/cast_to_fp16"
        )
        new_nodes.append(
            helper.make_node(
                "Cast",
                [output_dq],
                [original_output],
                name=output_cast_name,
                to=TensorProto.FLOAT16,
            )
        )
        names["output_cast"] = output_cast_name
        names["output_cast_output"] = original_output
        weight_qdq_paths[rewritten.name] = names
        if normalized_shape_contracts is not None:
            shape_overrides[rewritten.name] = copy.deepcopy(
                normalized_shape_contracts[module_name][call_index]
            )

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
        shape_overrides=shape_overrides or None,
        weight_encoding=weight_encoding,
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
        "activation_input_scales": {
            name: {
                "raw_fp32": _as_positive_scale(value, module_name=f"{name}:input"),
                "effective_fp32": float(
                    np.asarray(max(float(value), MIN_QUANT_SCALE), dtype=np.float32)
                ),
            }
            for name, value in activation_input_scales.items()
        },
        "activation_output_scales": {
            name: {
                "raw_fp32": _as_positive_scale(value, module_name=f"{name}:output"),
                "effective_fp32": float(
                    np.asarray(max(float(value), MIN_QUANT_SCALE), dtype=np.float32)
                ),
            }
            for name, value in activation_output_scales.items()
        },
        "weight_scales": weight_scales,
        "weight_encoding": weight_encoding,
        "weight_quantization": {
            WEIGHT_ENCODING_FP32_QDQ: (
                "independent_fp32_constant_qdq_to_symmetric_signed_int8_"
                "per_output_channel_axis_0"
            ),
            WEIGHT_ENCODING_PREQUANTIZED_INT8_DQ: (
                "independent_prequantized_int8_constant_dq_per_output_channel_axis_0"
            ),
        }[weight_encoding],
        "activation_quantization": (
            "fp16_cast_to_fp32_input_qdq_then_output_qdq_and_cast_to_fp16_per_tensor"
        ),
    }


def write_int8_conv3d_probe(
    *,
    path: str | Path,
    signature: dict[str, Any],
    signature_id: str,
    weight_encoding: str = WEIGHT_ENCODING_FP32_QDQ,
) -> str:
    """Write one probe using the exact v5 Q/DQ topology and Conv signature."""

    onnx, np, TensorProto, helpers = _lazy_onnx()
    helper, numpy_helper = helpers
    if conv_signature_id(signature) != signature_id:
        raise ValueError("probe signature_id does not match the Conv signature")
    canonical = _canonical_signature(signature)
    weight_shape = tuple(canonical["weight_shape"])
    output_channels = int(weight_shape[0])
    weight_fp32 = np.full(weight_shape, 0.125, dtype=np.float32)
    weight_scale = np.full(
        (output_channels,),
        np.float32(0.125 / INT8_MAX),
        dtype=np.float32,
    )
    graph_kind = f"probe_{signature_id}"
    module_name = "conv"
    (
        qdq_pre_nodes,
        qdq_post_nodes,
        initializers,
        activation_dq,
        weight_dq,
        bias_name,
        conv_output_fp32,
        output_dq,
        _names,
    ) = _make_qdq_call_site(
        helper=helper,
        numpy_helper=numpy_helper,
        TensorProto=TensorProto,
        np=np,
        graph_kind=graph_kind,
        module_name=module_name,
        call_index=0,
        activation_input="activation",
        weight_fp32=weight_fp32,
        weight_scale_fp32=weight_scale,
        activation_input_scale_fp32=np.asarray(1.0 / INT8_MAX, dtype=np.float32),
        activation_output_scale_fp32=np.asarray(1.0 / INT8_MAX, dtype=np.float32),
        bias_fp32=np.zeros((output_channels,), dtype=np.float32),
        weight_encoding=weight_encoding,
    )
    call_site = f"int8/probe/{signature_id}/call_0"
    conv_attributes = {
        "kernel_shape": canonical["kernel_shape"],
        "pads": canonical["pads"],
        "strides": canonical["strides"],
        "dilations": canonical["dilations"],
        "group": canonical["group"],
    }
    qdq_pre_nodes.append(
        helper.make_node(
            "Conv",
            [activation_dq, weight_dq, bias_name],
            [conv_output_fp32],
            name=call_site,
            **conv_attributes,
        )
    )
    qdq_pre_nodes.extend(qdq_post_nodes)
    qdq_pre_nodes.append(
        helper.make_node(
            "Cast",
            [output_dq],
            ["output"],
            name=f"qdq/{graph_kind}/output/{module_name}/call_0/cast_to_fp16",
            to=TensorProto.FLOAT16,
        )
    )
    graph = helper.make_graph(
        qdq_pre_nodes,
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
        producer_name="sglang-jetson-sfwan-qdq-v5-probe",
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
