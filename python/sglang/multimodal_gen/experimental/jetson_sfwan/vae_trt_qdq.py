"""Explicit-Q/DQ ONNX rewriting for the Jetson SFWan VAE.

The build tool exports an FP16 graph that calls the existing SGLang Wan
decoder.  This module rewrites only the 28 residual-block Conv3d weights and
their 84 unrolled call sites.  ONNX and NumPy are imported lazily so importing
the service does not require the TensorRT build environment.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

QDQ_SCHEMA_VERSION = 2
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


def _replace_initializer(model: Any, replacement: Any) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == replacement.name:
            model.graph.initializer[index].CopyFrom(replacement)
            return
    model.graph.initializer.append(replacement)


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


def audit_qdq_model(
    model: Any,
    *,
    graph_kind: str,
    target_module_names: tuple[str, ...],
) -> dict[str, Any]:
    """Validate the structural INT8 contract after rewriting."""

    _onnx, _np, TensorProto, _helpers = _lazy_onnx()

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
    weight_dq = [
        node
        for node in model.graph.node
        if node.op_type == "DequantizeLinear"
        and node.name.startswith(f"qdq/{graph_kind}/weight/")
    ]
    target_casts = [
        node.name
        for node in model.graph.node
        if node.op_type == "Cast"
        and node.name.startswith(f"qdq/{graph_kind}/activation/")
    ]
    initializers = _initializer_map(model)
    producers = {
        output_name: node
        for node in model.graph.node
        for output_name in node.output
        if output_name
    }
    per_module_counts: dict[str, int] = {}
    for module_name in target_module_names:
        prefix = f"{target_call_prefix}{module_name}/call_"
        per_module_counts[module_name] = sum(
            node.name.startswith(prefix) for node in target_convs
        )

    errors: list[str] = []
    if len(target_module_names) != EXPECTED_LOGICAL_CONVS:
        errors.append(
            f"logical target count is {len(target_module_names)}, "
            f"expected {EXPECTED_LOGICAL_CONVS}"
        )
    if len(target_convs) != EXPECTED_CALL_SITES:
        errors.append(
            f"target Conv count is {len(target_convs)}, expected {EXPECTED_CALL_SITES}"
        )
    if len(activation_q) != EXPECTED_CALL_SITES:
        errors.append(
            f"activation QuantizeLinear count is {len(activation_q)}, "
            f"expected {EXPECTED_CALL_SITES}"
        )
    if len(activation_dq) != EXPECTED_CALL_SITES:
        errors.append(
            f"activation DequantizeLinear count is {len(activation_dq)}, "
            f"expected {EXPECTED_CALL_SITES}"
        )
    if len(weight_dq) != EXPECTED_LOGICAL_CONVS:
        errors.append(
            f"weight DequantizeLinear count is {len(weight_dq)}, "
            f"expected {EXPECTED_LOGICAL_CONVS}"
        )
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
        errors.append(f"target Q/DQ paths contain FP32 Cast nodes: {target_casts}")

    invalid_fp16_bindings: list[str] = []
    for node in target_convs:
        for role, input_name in zip(
            ("activation", "weight"),
            node.input[:2],
            strict=True,
        ):
            producer = producers.get(input_name)
            if producer is None or producer.op_type != "DequantizeLinear":
                invalid_fp16_bindings.append(f"{node.name}:{role}:missing_dq")
                continue
            scale = initializers.get(producer.input[1])
            if scale is None or int(scale.data_type) != int(TensorProto.FLOAT16):
                invalid_fp16_bindings.append(f"{node.name}:{role}:scale_not_fp16")
        if len(node.input) >= 3 and node.input[2]:
            bias = initializers.get(node.input[2])
            if bias is None or int(bias.data_type) != int(TensorProto.FLOAT16):
                invalid_fp16_bindings.append(f"{node.name}:bias_not_fp16")
    if invalid_fp16_bindings:
        errors.append(
            f"target Conv bindings are not canonical FP16 Q/DQ: {invalid_fp16_bindings}"
        )

    call_site_names = [node.name for node in target_convs]
    return {
        "schema_version": QDQ_SCHEMA_VERSION,
        "graph_kind": graph_kind,
        "passed": not errors,
        "errors": errors,
        "logical_conv_count": len(target_module_names),
        "target_conv_call_site_count": len(target_convs),
        "activation_quantize_count": len(activation_q),
        "activation_dequantize_count": len(activation_dq),
        "weight_dequantize_count": len(weight_dq),
        "call_sites_per_logical_conv": per_module_counts,
        "call_site_names": call_site_names,
        "unsupported_quantized_conv_nodes": qlinear_convs,
        "target_cast_nodes": target_casts,
        "invalid_fp16_bindings": invalid_fp16_bindings,
    }


def rewrite_onnx_with_int8_qdq(
    *,
    source_path: str | Path,
    destination_path: str | Path,
    graph_kind: str,
    target_module_names: tuple[str, ...],
    activation_scales: dict[str, float],
) -> dict[str, Any]:
    """Rewrite selected FP16 Conv3d call sites with explicit signed INT8 Q/DQ."""

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

    onnx, np, TensorProto, helpers = _lazy_onnx()
    helper, numpy_helper = helpers
    source = Path(source_path)
    destination = Path(destination_path)
    model = onnx.load(str(source), load_external_data=True)
    source_opset = _opset_version(model)
    if source_opset != QDQ_OPSET:
        raise ValueError(
            f"explicit Q/DQ rewrite requires a real ONNX opset {QDQ_OPSET} "
            f"source graph, got opset {source_opset}; re-export the FP16 graph "
            "instead of changing only its opset_import version"
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
    new_nodes: list[Any] = []
    weight_dq_outputs: dict[str, str] = {}
    quantized_weight_names: dict[str, str] = {}
    weight_scales: dict[str, dict[str, list[float]]] = {}

    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            new_nodes.append(copy.deepcopy(node))
            continue
        module_name = module_by_weight.get(node.input[1])
        if module_name is None:
            new_nodes.append(copy.deepcopy(node))
            continue

        call_index = calls_by_module[module_name]
        calls_by_module[module_name] += 1
        safe_module = _safe_name(module_name)
        prefix = f"sfwan_{graph_kind}_{safe_module}_call_{call_index}"

        if module_name not in weight_dq_outputs:
            weight_initializer = initializers[resolved_weights[module_name]]
            weight = numpy_helper.to_array(weight_initializer).astype(np.float32)
            if weight.ndim != 5:
                raise ValueError(
                    f"target {module_name!r} weight must be rank-5 Conv3d, "
                    f"got shape {weight.shape}"
                )
            channel_max = np.max(np.abs(weight), axis=(1, 2, 3, 4))
            raw_channel_scale = np.maximum(
                channel_max / INT8_MAX,
                MIN_QUANT_SCALE,
            ).astype(np.float32)
            channel_scale = np.maximum(
                raw_channel_scale,
                np.finfo(np.float16).tiny,
            ).astype(np.float16)
            quantized_weight = np.clip(
                np.rint(
                    weight / channel_scale.astype(np.float32)[:, None, None, None, None]
                ),
                -127,
                127,
            ).astype(np.int8)
            weight_name = f"{prefix}_weight_int8"
            scale_name = f"{prefix}_weight_scale"
            zero_name = f"{prefix}_weight_zero"
            dq_output = f"{prefix}_weight_fp16"
            model.graph.initializer.extend(
                [
                    numpy_helper.from_array(quantized_weight, name=weight_name),
                    numpy_helper.from_array(channel_scale, name=scale_name),
                    numpy_helper.from_array(
                        np.zeros(channel_scale.shape, dtype=np.int8),
                        name=zero_name,
                    ),
                ]
            )
            new_nodes.append(
                helper.make_node(
                    "DequantizeLinear",
                    [weight_name, scale_name, zero_name],
                    [dq_output],
                    name=f"qdq/{graph_kind}/weight/{module_name}",
                    axis=0,
                )
            )
            weight_dq_outputs[module_name] = dq_output
            quantized_weight_names[module_name] = weight_name
            weight_scales[module_name] = {
                "raw_fp32": raw_channel_scale.tolist(),
                "effective_fp16": channel_scale.tolist(),
            }

        activation_scale_value = _as_positive_scale(
            activation_scales[module_name],
            module_name=module_name,
        )
        effective_activation_scale = np.asarray(
            max(activation_scale_value, float(np.finfo(np.float16).tiny)),
            dtype=np.float16,
        )
        activation_scale_name = f"{prefix}_activation_scale"
        activation_zero_name = f"{prefix}_activation_zero"
        activation_int8 = f"{prefix}_activation_int8"
        activation_dq = f"{prefix}_activation_dequantized"
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(
                    effective_activation_scale,
                    name=activation_scale_name,
                ),
                numpy_helper.from_array(
                    np.asarray(0, dtype=np.int8),
                    name=activation_zero_name,
                ),
            ]
        )
        new_nodes.extend(
            [
                helper.make_node(
                    "QuantizeLinear",
                    [
                        node.input[0],
                        activation_scale_name,
                        activation_zero_name,
                    ],
                    [activation_int8],
                    name=f"qdq/{graph_kind}/activation/{module_name}/"
                    f"call_{call_index}/quantize",
                ),
                helper.make_node(
                    "DequantizeLinear",
                    [
                        activation_int8,
                        activation_scale_name,
                        activation_zero_name,
                    ],
                    [activation_dq],
                    name=f"qdq/{graph_kind}/activation/{module_name}/"
                    f"call_{call_index}/dequantize",
                ),
            ]
        )

        rewritten = copy.deepcopy(node)
        rewritten.input[0] = activation_dq
        rewritten.input[1] = weight_dq_outputs[module_name]
        rewritten.name = f"int8/{graph_kind}/{module_name}/call_{call_index}"
        if len(rewritten.input) >= 3 and rewritten.input[2]:
            bias_name = rewritten.input[2]
            bias_initializer = initializers.get(bias_name)
            if bias_initializer is None:
                raise ValueError(
                    f"target Conv bias initializer {bias_name!r} is missing"
                )
            bias = numpy_helper.to_array(bias_initializer).astype(np.float16)
            _replace_initializer(
                model,
                numpy_helper.from_array(bias, name=bias_name),
            )
            initializers[bias_name] = _initializer_map(model)[bias_name]
        new_nodes.append(rewritten)

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
    report = {
        **audit,
        "source": str(source),
        "destination": str(destination),
        "opset": QDQ_OPSET,
        "quantized_weight_initializers": quantized_weight_names,
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
        "weight_quantization": "symmetric_signed_int8_per_output_channel_axis_0",
        "activation_quantization": "symmetric_signed_int8_per_tensor",
    }
    return report


def write_qdq_report(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
