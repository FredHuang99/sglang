"""Isolated TensorRT VAE fusion-v1 graph and artifact helpers.

This module deliberately has no import-time Torch, ONNX, CUDA, or TensorRT
dependency.  The existing explicit-Q/DQ v5 artifacts remain the source of
truth.  Fusion-v1 only replaces audited boundary subgraphs and fails closed
whenever the exact causal-cache or residual-block topology cannot be proven.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .vae_trt_qdq import EXPECTED_CALL_SITES, QDQ_SCHEMA_VERSION
from .vae_trt_runtime import (
    TRT_VAE_CACHE_BANK_BYTES,
    TRT_VAE_CACHE_COUNT,
    TRT_VAE_CACHE_TOTAL_ELEMENTS,
    TRT_VAE_HEIGHT,
    TRT_VAE_WIDTH,
)

FUSION_VARIANT = "fusion_v1"
FUSION_SCHEMA_VERSION = 1
FUSION_AUDIT_SCHEMA_VERSION = 1
FUSION_SUBDIRECTORY = "fusion_v1"
FUSION_MANIFEST_FILE = "fusion_manifest.json"
FUSION_AUDIT_FILE = "fusion_audit_v1.json"
FUSION_BUILD_STATE_FILE = "fusion_build_state.json"
FUSION_PROBE_FILE = "fusion_probe_v1.json"
FUSION_PLUGIN_MANIFEST_FILE = "plugin_manifest.json"
FUSION_PLUGIN_LIBRARY_FILE = "libsfwan_vae_trt_fusion.so"
FUSION_TIMING_CACHE_FILE = "fusion_timing.cache"

PLUGIN_NAMESPACE = "sglang.sfwan"
PLUGIN_VERSION = "1"
PACK_QUANT_PLUGIN = "SfWanCausalPackQuantPlugin"
CACHE_UPDATE_PLUGIN = "SfWanCacheUpdatePlugin"
EPILOGUE_PLUGIN = "SfWanInt8EpiloguePlugin"
PLUGIN_CREATORS = (PACK_QUANT_PLUGIN, CACHE_UPDATE_PLUGIN, EPILOGUE_PLUGIN)

FUSION_ENGINE_FILES = {
    "initial": "initial_int8_fusion_v1.plan",
    "steady": "steady_int8_fusion_v1.plan",
}
FUSION_ONNX_FILES = {
    "initial": "initial_int8_fusion_v1.onnx",
    "steady": "steady_int8_fusion_v1.onnx",
}
FUSION_INSPECTOR_FILES = {
    "initial": "initial_int8_fusion_v1_inspector.json",
    "steady": "steady_int8_fusion_v1_inspector.json",
}

_TARGET_CONV_RE = re.compile(
    r"^int8/(?P<kind>initial|steady)/(?P<module>.+)/call_(?P<call>\d+)$"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(destination)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _load_json_value(path: Path, *, label: str) -> Any:
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
        raise ValueError(f"{label} path escapes {root}: {value!r}") from exc
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def load_fusion_manifest(engine_dir: str | Path) -> dict[str, Any]:
    root = Path(engine_dir).expanduser().resolve() / FUSION_SUBDIRECTORY
    path = root / FUSION_MANIFEST_FILE
    if not path.is_file():
        raise ValueError(f"TensorRT fusion manifest does not exist: {path}")
    return _load_json_object(path, label="TensorRT fusion manifest")


def _validate_sha_record(
    *, root: Path, record: Mapping[str, Any], label: str, verify_hashes: bool
) -> tuple[Path, str]:
    path = _resolve_inside(root, record.get("file"), label=label)
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label} SHA256 is invalid")
    if verify_hashes and sha256_file(path) != digest:
        raise ValueError(f"{label} SHA256 does not match: {path}")
    return path, digest


def validate_fusion_manifest(
    manifest: dict[str, Any],
    *,
    engine_dir: str | Path,
    base_manifest: dict[str, Any],
    verify_hashes: bool,
) -> dict[str, Any]:
    """Validate fusion-v1 without importing CUDA/TensorRT.

    The returned paths are the only files the runtime may consume.  A fusion
    plan is never accepted merely because it deserializes: plugin identity,
    base-manifest identity, cache ABI, tactic audit, and every file digest are
    bound together here.
    """

    if manifest.get("schema_version") != FUSION_SCHEMA_VERSION:
        raise ValueError("unsupported TensorRT VAE fusion manifest schema")
    if manifest.get("variant") != FUSION_VARIANT:
        raise ValueError("TensorRT VAE fusion manifest variant is invalid")
    root = Path(engine_dir).expanduser().resolve()
    fusion_root = root / FUSION_SUBDIRECTORY
    base_manifest_path = root / "manifest.json"
    if not base_manifest_path.is_file():
        raise ValueError("base TensorRT VAE manifest is missing")
    base_record = manifest.get("base")
    if not isinstance(base_record, dict):
        raise ValueError("fusion manifest has no base artifact identity")
    if base_record.get("manifest_sha256") != sha256_file(base_manifest_path):
        raise ValueError("fusion manifest belongs to a different base manifest")
    if base_record.get("qdq_schema_version") != QDQ_SCHEMA_VERSION:
        raise ValueError("fusion-v1 must be derived from explicit Q/DQ v5")

    build = manifest.get("build")
    base_build = base_manifest.get("build")
    if not isinstance(build, dict) or not isinstance(base_build, dict):
        raise ValueError("fusion/base build environment is missing")
    for key in ("compute_capability", "tensorrt_version", "cuda_version"):
        if build.get(key) != base_build.get(key):
            raise ValueError(f"fusion build {key} differs from the base plan")
    if build.get("compute_capability") != [8, 7]:
        raise ValueError("fusion-v1 only supports Jetson Orin SM87")

    plugin = manifest.get("plugin")
    if not isinstance(plugin, dict):
        raise ValueError("fusion manifest has no plugin record")
    plugin_path, plugin_sha = _validate_sha_record(
        root=fusion_root,
        record=plugin,
        label="TensorRT VAE fusion plugin",
        verify_hashes=verify_hashes,
    )
    if plugin.get("namespace") != PLUGIN_NAMESPACE:
        raise ValueError("fusion plugin namespace is invalid")
    if plugin.get("version") != PLUGIN_VERSION:
        raise ValueError("fusion plugin version is invalid")
    if plugin.get("creators") != list(PLUGIN_CREATORS):
        raise ValueError("fusion plugin creator set is incomplete")
    plugin_manifest_path, plugin_manifest_sha = _validate_sha_record(
        root=fusion_root,
        record={
            "file": plugin.get("manifest_file"),
            "sha256": plugin.get("manifest_sha256"),
        },
        label="TensorRT VAE fusion plugin manifest",
        verify_hashes=verify_hashes,
    )
    plugin_manifest = _load_json_object(
        plugin_manifest_path, label="TensorRT VAE fusion plugin manifest"
    )
    if (
        plugin_manifest.get("variant") != FUSION_VARIANT
        or plugin_manifest.get("sha256") != plugin_sha
        or plugin_manifest.get("namespace") != PLUGIN_NAMESPACE
        or plugin_manifest.get("version") != PLUGIN_VERSION
        or plugin_manifest.get("creators") != list(PLUGIN_CREATORS)
    ):
        raise ValueError("fusion plugin manifest identity is invalid")

    cache = manifest.get("cache")
    base_cache = base_manifest.get("cache")
    if not isinstance(cache, dict) or not isinstance(base_cache, dict):
        raise ValueError("fusion/base feature-cache contract is missing")
    required_cache = {
        "tensor_count": TRT_VAE_CACHE_COUNT,
        "dtype": "float16",
        "total_elements": TRT_VAE_CACHE_TOTAL_ELEMENTS,
        "single_bank_bytes": TRT_VAE_CACHE_BANK_BYTES,
        "double_bank_bytes": TRT_VAE_CACHE_BANK_BYTES * 2,
    }
    for key, expected in required_cache.items():
        if cache.get(key) != expected:
            raise ValueError(f"fusion cache {key} is invalid")
    if cache.get("bindings") != base_cache.get("bindings"):
        raise ValueError("fusion-v1 changed the 32-binding FP16 cache ABI")

    audit_record = manifest.get("audit")
    if not isinstance(audit_record, dict) or audit_record.get("passed") is not True:
        raise ValueError("fusion-v1 has no passing audit")
    audit_path, audit_sha = _validate_sha_record(
        root=fusion_root,
        record=audit_record,
        label="TensorRT VAE fusion audit",
        verify_hashes=verify_hashes,
    )
    audit = _load_json_object(audit_path, label="TensorRT VAE fusion audit")
    if (
        audit.get("schema_version") != FUSION_AUDIT_SCHEMA_VERSION
        or audit.get("variant") != FUSION_VARIANT
        or audit.get("qdq_schema_version") != QDQ_SCHEMA_VERSION
        or audit.get("passed") is not True
        or audit.get("complete") is not True
        or audit.get("errors") != []
    ):
        raise ValueError("TensorRT VAE fusion audit is incomplete")
    quantization = base_manifest.get("quantization")
    weight_encoding = (
        quantization.get("weight_encoding") if isinstance(quantization, dict) else None
    )
    if audit.get("weight_encoding") != weight_encoding:
        raise ValueError("fusion audit weight encoding differs from Q/DQ v5")
    if audit.get("plugin_sha256") != plugin_sha:
        raise ValueError("fusion audit and plugin SHA256 disagree")

    engines = manifest.get("engines")
    if not isinstance(engines, dict):
        raise ValueError("fusion manifest has no engines")
    validated_engines: dict[str, dict[str, Any]] = {}
    for kind, frames in (("initial", 9), ("steady", 12)):
        record = engines.get(kind)
        if not isinstance(record, dict):
            raise ValueError(f"fusion manifest has no {kind} engine")
        plan_path, plan_sha = _validate_sha_record(
            root=fusion_root,
            record=record,
            label=f"TensorRT VAE fusion {kind} plan",
            verify_hashes=verify_hashes,
        )
        source_path, source_sha = _validate_sha_record(
            root=fusion_root,
            record={
                "file": record.get("source_onnx_file"),
                "sha256": record.get("source_onnx_sha256"),
            },
            label=f"TensorRT VAE fusion {kind} source ONNX",
            verify_hashes=verify_hashes,
        )
        inspector_record = record.get("inspector")
        if not isinstance(inspector_record, dict):
            raise ValueError(f"fusion {kind} inspector record is missing")
        inspector_path, _ = _validate_sha_record(
            root=fusion_root,
            record=inspector_record,
            label=f"TensorRT VAE fusion {kind} inspector",
            verify_hashes=verify_hashes,
        )
        if record.get("rgb_shape") != [1, 3, frames, TRT_VAE_HEIGHT, TRT_VAE_WIDTH]:
            raise ValueError(f"fusion {kind} RGB shape is invalid")
        record_counts = record.get("fusion_counts")
        if not isinstance(record_counts, dict):
            raise ValueError(f"fusion {kind} count record is missing")
        if record.get("profiling_verbosity") != "detailed":
            raise ValueError(f"fusion {kind} plan must retain detailed Inspector data")
        audit_engines = audit.get("engines")
        audit_tactics = audit.get("tactics")
        audit_plan_sha256 = audit.get("plan_sha256")
        if not all(
            isinstance(value, dict)
            for value in (audit_engines, audit_tactics, audit_plan_sha256)
        ):
            raise ValueError("fusion tactic audit records are malformed")
        plan_audit = audit_engines.get(kind)
        tactic_audit = audit_tactics.get(kind)
        audit_plan_sha = audit_plan_sha256.get(kind)
        if (
            not isinstance(plan_audit, dict)
            or tactic_audit != plan_audit
            or plan_audit.get("passed") is not True
            or plan_audit.get("plan_sha256") != plan_sha
            or audit_plan_sha != plan_sha
            or plan_audit.get("target_conv_call_site_count") != EXPECTED_CALL_SITES
            or plan_audit.get("int8_target_conv_call_site_count") != EXPECTED_CALL_SITES
            or plan_audit.get("non_int8_call_sites") != []
            or plan_audit.get("fp16_or_tf32_fallback_call_sites") != []
            or plan_audit.get("input_reformat_call_sites") != []
            or plan_audit.get("output_reformat_call_sites") != []
            or plan_audit.get("plugin_counts") != record_counts
        ):
            raise ValueError(f"fusion {kind} tactic audit is invalid")
        selected_call_sites = record.get("selected_fusion_call_sites")
        if (
            not isinstance(selected_call_sites, list)
            or len(selected_call_sites)
            != int(record_counts.get("input_pack_quant", -1))
            or len(set(selected_call_sites)) != len(selected_call_sites)
            or any(
                not isinstance(value, str) or not value for value in selected_call_sites
            )
        ):
            raise ValueError(f"fusion {kind} selected call-site record is invalid")
        selected_cache_modes = record.get("selected_cache_update_modes")
        if (
            not isinstance(selected_cache_modes, dict)
            or set(selected_cache_modes) != set(selected_call_sites)
            or any(
                value not in {"dual", "separate"}
                for value in selected_cache_modes.values()
            )
            or sum(value == "dual" for value in selected_cache_modes.values())
            != record_counts.get("cache_update_dual")
            or sum(value == "separate" for value in selected_cache_modes.values())
            != record_counts.get("cache_update_separate")
            or record_counts.get("cache_update")
            != record_counts.get("cache_update_dual", 0)
            + record_counts.get("cache_update_separate", 0)
        ):
            raise ValueError(f"fusion {kind} cache-update mode record is invalid")
        if plan_audit.get("selected_cache_update_modes") != selected_cache_modes:
            raise ValueError(f"fusion {kind} tactic/cache-mode audit disagrees")
        validated_engines[kind] = {
            **record,
            "path": str(plan_path),
            "source_onnx_path": str(source_path),
            "source_onnx_sha256": source_sha,
            "inspector_path": str(inspector_path),
            "inspector": _load_json_value(inspector_path, label="engine inspector"),
        }

    counts = manifest.get("fusion_counts")
    if not isinstance(counts, dict):
        raise ValueError("fusion manifest has no fusion counts")
    required_count_keys = (
        "input_pack_quant",
        "cache_update",
        "cache_update_dual",
        "cache_update_separate",
        "conv1_norm_silu",
        "conv2_residual",
    )
    if any(
        isinstance(counts.get(key), bool)
        or not isinstance(counts.get(key), int)
        or counts.get(key) < 0
        for key in required_count_keys
    ):
        raise ValueError("fusion manifest contains invalid fusion counts")
    if counts.get("input_pack_quant", 0) <= 0:
        raise ValueError("fusion-v1 contains no input pack/quant fusion")
    if counts.get("cache_update") != counts.get("cache_update_dual", 0) + counts.get(
        "cache_update_separate", 0
    ):
        raise ValueError("fusion aggregate cache-update counts are inconsistent")
    per_engine_counts = manifest.get("per_engine_fusion_counts")
    if not isinstance(per_engine_counts, dict):
        raise ValueError("fusion manifest has no per-engine fusion counts")
    for kind in ("initial", "steady"):
        kind_counts = per_engine_counts.get(kind)
        if not isinstance(kind_counts, dict) or kind_counts != validated_engines[
            kind
        ].get("fusion_counts"):
            raise ValueError(f"fusion {kind} count records disagree")
    if any(
        counts[key]
        != sum(int(per_engine_counts[kind][key]) for kind in ("initial", "steady"))
        for key in required_count_keys
    ):
        raise ValueError("fusion aggregate counts disagree with per-engine counts")
    if audit.get("fusion_counts") != counts:
        raise ValueError("fusion audit and manifest counts disagree")
    probe_record = manifest.get("probe")
    if not isinstance(probe_record, dict) or probe_record.get("passed") is not True:
        raise ValueError("fusion manifest has no passing micro-probe")
    probe_path, probe_sha = _validate_sha_record(
        root=fusion_root,
        record=probe_record,
        label="TensorRT VAE fusion probe",
        verify_hashes=verify_hashes,
    )
    probe = _load_json_object(probe_path, label="TensorRT VAE fusion probe")
    if (
        probe.get("schema_version") != FUSION_SCHEMA_VERSION
        or probe.get("variant") != FUSION_VARIANT
        or probe.get("passed") is not True
        or probe.get("errors") != []
    ):
        raise ValueError("TensorRT VAE fusion probe report did not pass")
    if manifest.get("required_focus_prefix") != "decoder.up_blocks.3":
        raise ValueError("fusion-v1 required focus prefix is invalid")
    if manifest.get("required_focus_complete") is not True:
        raise ValueError("fusion-v1 did not fuse every required focus call site")

    timing_cache_file = manifest.get("timing_cache_file")
    timing_cache_sha = manifest.get("timing_cache_sha256")
    if timing_cache_sha is not None:
        _validate_sha_record(
            root=fusion_root,
            record={"file": timing_cache_file, "sha256": timing_cache_sha},
            label="TensorRT VAE fusion timing cache",
            verify_hashes=verify_hashes,
        )

    return {
        "root": str(fusion_root),
        "plugin_path": str(plugin_path),
        "plugin_sha256": plugin_sha,
        "plugin_manifest_path": str(plugin_manifest_path),
        "plugin_manifest_sha256": plugin_manifest_sha,
        "audit_path": str(audit_path),
        "audit_sha256": audit_sha,
        "audit": audit,
        "engines": validated_engines,
        "fusion_counts": dict(counts),
        "per_engine_fusion_counts": {
            kind: dict(per_engine_counts[kind]) for kind in ("initial", "steady")
        },
        "probe_path": str(probe_path),
        "probe_sha256": probe_sha,
    }


def _lazy_onnx() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:  # pragma: no cover - Jetson build dependency
        raise RuntimeError("fusion graph analysis requires onnx and numpy") from exc
    return onnx, np, TensorProto, (helper, numpy_helper)


def _attribute_map(node: Any, helper: Any) -> dict[str, Any]:
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def _shape_map(model: Any) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    values = [*model.graph.input, *model.graph.output, *model.graph.value_info]
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        shape: list[int] = []
        valid = True
        for dimension in tensor_type.shape.dim:
            if not dimension.HasField("dim_value") or int(dimension.dim_value) <= 0:
                valid = False
                break
            shape.append(int(dimension.dim_value))
        if valid:
            result[value.name] = shape
    return result


def _node_maps(model: Any) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    producer: dict[str, Any] = {}
    consumers: dict[str, list[Any]] = defaultdict(list)
    for node in model.graph.node:
        for output in node.output:
            if output:
                if output in producer:
                    raise ValueError(f"ONNX tensor has multiple producers: {output}")
                producer[output] = node
        for value in node.input:
            if value:
                consumers[value].append(node)
    return producer, consumers


def _one_producer(producer: Mapping[str, Any], tensor: str, op_type: str) -> Any:
    node = producer.get(tensor)
    if node is None or node.op_type != op_type:
        raise ValueError(f"expected {tensor!r} to come from {op_type}")
    return node


def _one_consumer(consumers: Mapping[str, list[Any]], tensor: str, op_type: str) -> Any:
    matches = [node for node in consumers.get(tensor, []) if node.op_type == op_type]
    if len(matches) != 1:
        raise ValueError(f"expected one {op_type} consumer for {tensor!r}")
    return matches[0]


def _require_only_consumers(
    *,
    tensor: str,
    consumers: Mapping[str, list[Any]],
    allowed_node_names: set[str],
) -> None:
    unexpected = sorted(
        node.name
        for node in consumers.get(tensor, [])
        if node.name not in allowed_node_names
    )
    if unexpected:
        raise ValueError(f"tensor {tensor!r} has shared consumers: {unexpected}")


def _require_removable_subgraph(
    *,
    nodes: Iterable[Any],
    replacement_output: str,
    consumers: Mapping[str, list[Any]],
    graph_outputs: set[str],
) -> None:
    node_list = list(nodes)
    names = {node.name for node in node_list}
    for node in node_list:
        for output in node.output:
            if not output or output == replacement_output:
                continue
            if output in graph_outputs:
                raise ValueError(
                    f"removed intermediate {output!r} is a public graph output"
                )
            _require_only_consumers(
                tensor=output,
                consumers=consumers,
                allowed_node_names=names,
            )


def _constant_array(
    *,
    name: str,
    initializers: Mapping[str, Any],
    numpy_helper: Any,
    producer: Mapping[str, Any] | None = None,
    helper: Any | None = None,
    np_module: Any | None = None,
    shapes: Mapping[str, list[int]] | None = None,
    memo: dict[str, Any | None] | None = None,
    visiting: set[str] | None = None,
) -> Any | None:
    """Resolve a small, deterministic ONNX constant-expression subgraph.

    The legacy Torch ONNX exporter commonly supplies ``Pad`` extents through
    ``Constant -> Cast/Reshape/Slice`` nodes instead of graph initializers.
    Those values are still compile-time constants.  Only metadata operators
    with fully constant inputs are evaluated here; any dependency on a graph
    input returns ``None`` so fusion analysis continues to fail closed for
    genuinely dynamic padding.
    """

    if memo is None:
        memo = {}
    if name in memo:
        return memo[name]
    initializer = initializers.get(name)
    if initializer is not None:
        value = numpy_helper.to_array(initializer)
        memo[name] = value
        return value
    if producer is None or helper is None or np_module is None:
        memo[name] = None
        return None
    node = producer.get(name)
    if node is None:
        memo[name] = None
        return None
    if visiting is None:
        visiting = set()
    if name in visiting:
        raise ValueError(f"constant-expression cycle at {name!r}")
    visiting.add(name)

    def resolve(input_name: str) -> Any | None:
        if not input_name:
            return None
        return _constant_array(
            name=input_name,
            initializers=initializers,
            numpy_helper=numpy_helper,
            producer=producer,
            helper=helper,
            np_module=np_module,
            shapes=shapes,
            memo=memo,
            visiting=visiting,
        )

    attributes = _attribute_map(node, helper)
    result: Any | None = None
    try:
        if node.op_type == "Constant":
            if "value" in attributes:
                raw = attributes["value"]
                try:
                    result = numpy_helper.to_array(raw)
                except (AttributeError, TypeError, ValueError):
                    result = np_module.asarray(raw)
            else:
                for attribute_name in (
                    "value_ints",
                    "value_floats",
                    "value_int",
                    "value_float",
                ):
                    if attribute_name in attributes:
                        result = np_module.asarray(attributes[attribute_name])
                        break
        elif node.op_type == "Identity" and len(node.input) == 1:
            result = resolve(node.input[0])
        elif node.op_type == "Cast" and len(node.input) == 1:
            value = resolve(node.input[0])
            to = attributes.get("to")
            if value is not None and to is not None:
                dtype = helper.tensor_dtype_to_np_dtype(int(to))
                result = np_module.asarray(value).astype(dtype, copy=False)
        elif node.op_type == "Concat" and node.input:
            values = [resolve(input_name) for input_name in node.input]
            if all(value is not None for value in values):
                result = np_module.concatenate(
                    [np_module.asarray(value) for value in values],
                    axis=int(attributes.get("axis", 0)),
                )
        elif node.op_type == "Reshape" and len(node.input) >= 2:
            value = resolve(node.input[0])
            target = resolve(node.input[1])
            if value is not None and target is not None:
                source = np_module.asarray(value)
                target_shape = [int(item) for item in np_module.asarray(target).flat]
                if int(attributes.get("allowzero", 0)) == 0:
                    target_shape = [
                        source.shape[index] if dimension == 0 else dimension
                        for index, dimension in enumerate(target_shape)
                    ]
                result = source.reshape(target_shape)
        elif node.op_type == "Transpose" and len(node.input) == 1:
            value = resolve(node.input[0])
            if value is not None:
                source = np_module.asarray(value)
                permutation = attributes.get("perm")
                result = source.transpose(
                    None
                    if permutation is None
                    else tuple(int(item) for item in permutation)
                )
        elif node.op_type in {"Unsqueeze", "Squeeze"} and node.input:
            value = resolve(node.input[0])
            raw_axes = (
                resolve(node.input[1])
                if len(node.input) > 1 and node.input[1]
                else attributes.get("axes")
            )
            if value is not None and raw_axes is not None:
                axes = tuple(int(item) for item in np_module.asarray(raw_axes).flat)
                source = np_module.asarray(value)
                result = (
                    np_module.expand_dims(source, axis=axes)
                    if node.op_type == "Unsqueeze"
                    else np_module.squeeze(source, axis=axes)
                )
        elif node.op_type == "Slice" and node.input:
            value = resolve(node.input[0])
            starts = (
                resolve(node.input[1])
                if len(node.input) > 1
                else attributes.get("starts")
            )
            ends = (
                resolve(node.input[2])
                if len(node.input) > 2
                else attributes.get("ends")
            )
            axes = (
                resolve(node.input[3])
                if len(node.input) > 3 and node.input[3]
                else attributes.get("axes")
            )
            steps = (
                resolve(node.input[4])
                if len(node.input) > 4 and node.input[4]
                else attributes.get("steps")
            )
            if value is not None and starts is not None and ends is not None:
                source = np_module.asarray(value)
                starts_list = [int(item) for item in np_module.asarray(starts).flat]
                ends_list = [int(item) for item in np_module.asarray(ends).flat]
                axes_list = (
                    list(range(len(starts_list)))
                    if axes is None
                    else [int(item) for item in np_module.asarray(axes).flat]
                )
                steps_list = (
                    [1] * len(starts_list)
                    if steps is None
                    else [int(item) for item in np_module.asarray(steps).flat]
                )
                if not (
                    len(starts_list)
                    == len(ends_list)
                    == len(axes_list)
                    == len(steps_list)
                ):
                    raise ValueError("constant Slice metadata lengths differ")
                slices = [slice(None)] * source.ndim
                for start, end, axis, step in zip(
                    starts_list,
                    ends_list,
                    axes_list,
                    steps_list,
                    strict=True,
                ):
                    slices[axis] = slice(start, end, step)
                result = source[tuple(slices)]
        elif node.op_type == "ConstantOfShape" and len(node.input) == 1:
            raw_shape = resolve(node.input[0])
            if raw_shape is not None:
                target_shape = tuple(
                    int(item) for item in np_module.asarray(raw_shape).flat
                )
                fill = attributes.get("value")
                if fill is None:
                    result = np_module.zeros(target_shape, dtype=np_module.float32)
                else:
                    fill_array = numpy_helper.to_array(fill)
                    result = np_module.full(
                        target_shape,
                        np_module.asarray(fill_array).reshape(-1)[0],
                        dtype=np_module.asarray(fill_array).dtype,
                    )
        elif node.op_type == "Shape" and len(node.input) == 1:
            source_shape = shapes.get(node.input[0]) if shapes is not None else None
            if source_shape is None:
                source = resolve(node.input[0])
                if source is not None:
                    source_shape = list(np_module.asarray(source).shape)
            if source_shape is not None:
                start = int(attributes.get("start", 0))
                end = int(attributes.get("end", len(source_shape)))
                result = np_module.asarray(
                    source_shape[start:end], dtype=np_module.int64
                )
        elif node.op_type == "Size" and len(node.input) == 1:
            value = resolve(node.input[0])
            if value is not None:
                result = np_module.asarray(
                    np_module.asarray(value).size,
                    dtype=np_module.int64,
                )
        elif (
            node.op_type in {"Add", "Sub", "Mul", "Div", "Max", "Min"}
            and len(node.input) == 2
        ):
            left = resolve(node.input[0])
            right = resolve(node.input[1])
            if left is not None and right is not None:
                operation = {
                    "Add": np_module.add,
                    "Sub": np_module.subtract,
                    "Mul": np_module.multiply,
                    "Div": np_module.divide,
                    "Max": np_module.maximum,
                    "Min": np_module.minimum,
                }[node.op_type]
                result = operation(
                    np_module.asarray(left),
                    np_module.asarray(right),
                )
        elif node.op_type == "Neg" and len(node.input) == 1:
            value = resolve(node.input[0])
            if value is not None:
                result = np_module.negative(np_module.asarray(value))
        elif node.op_type == "Range" and len(node.input) == 3:
            start = resolve(node.input[0])
            limit = resolve(node.input[1])
            delta = resolve(node.input[2])
            if start is not None and limit is not None and delta is not None:
                result = np_module.arange(
                    np_module.asarray(start).reshape(-1)[0],
                    np_module.asarray(limit).reshape(-1)[0],
                    np_module.asarray(delta).reshape(-1)[0],
                )
        elif node.op_type == "Gather" and len(node.input) == 2:
            data = resolve(node.input[0])
            indices = resolve(node.input[1])
            if data is not None and indices is not None:
                result = np_module.take(
                    np_module.asarray(data),
                    np_module.asarray(indices),
                    axis=int(attributes.get("axis", 0)),
                )
    except (IndexError, TypeError, ValueError):
        result = None
    finally:
        visiting.remove(name)
    memo[name] = result
    return result


def _tensor_depends_on(
    tensor: str,
    source: str,
    producer: Mapping[str, Any],
    memo: dict[tuple[str, str], bool],
) -> bool:
    key = (tensor, source)
    if key in memo:
        return memo[key]
    if tensor == source:
        memo[key] = True
        return True
    node = producer.get(tensor)
    if node is None:
        memo[key] = False
        return False
    memo[key] = False
    value = any(
        _tensor_depends_on(parent, source, producer, memo)
        for parent in node.input
        if parent
    )
    memo[key] = value
    return value


def _collect_reverse_subgraph(
    *, target: str, source: str, producer: Mapping[str, Any], limit: int = 64
) -> list[Any]:
    """Collect the unique producer subgraph between source and target."""

    memo: dict[tuple[str, str], bool] = {}
    if not _tensor_depends_on(target, source, producer, memo):
        raise ValueError(f"{target!r} does not depend on {source!r}")
    result: list[Any] = []
    seen: set[str] = set()

    def visit(tensor: str) -> None:
        if tensor == source:
            return
        node = producer.get(tensor)
        if node is None or node.name in seen:
            return
        if not any(
            _tensor_depends_on(parent, source, producer, memo)
            for parent in node.input
            if parent
        ):
            return
        for parent in node.input:
            if parent and _tensor_depends_on(parent, source, producer, memo):
                visit(parent)
        seen.add(node.name)
        result.append(node)
        if len(result) > limit:
            raise ValueError("fusion boundary subgraph exceeds the safety limit")

    visit(target)
    return result


def _find_cache_output(
    *,
    current: str,
    cache: str | None,
    graph_outputs: Iterable[str],
    shapes: Mapping[str, list[int]],
    producer: Mapping[str, Any],
) -> str | None:
    current_shape = shapes.get(current)
    if current_shape is None or len(current_shape) != 5:
        return None
    expected = [*current_shape]
    expected[2] = min(
        2, current_shape[2] + (shapes.get(cache, [0, 0, 0])[2] if cache else 0)
    )
    memo: dict[tuple[str, str], bool] = {}
    matches = []
    for output in graph_outputs:
        if shapes.get(output) != expected:
            continue
        if not _tensor_depends_on(output, current, producer, memo):
            continue
        if cache is not None and not _tensor_depends_on(output, cache, producer, memo):
            continue
        matches.append(output)
    return matches[0] if len(matches) == 1 else None


def _find_add_after(
    *, tensor: str, producer: Mapping[str, Any], consumers: Mapping[str, list[Any]]
) -> tuple[Any, list[Any]] | None:
    del producer
    frontier = [(tensor, [])]
    visited = {tensor}
    allowed = {"Identity", "Cast", "Reshape", "Transpose", "Contiguous"}
    for _ in range(8):
        next_frontier: list[tuple[str, list[Any]]] = []
        for value, path in frontier:
            for node in consumers.get(value, []):
                if node.op_type == "Add" and len(node.input) == 2:
                    return node, path
                if node.op_type not in allowed or len(node.output) != 1:
                    continue
                output = node.output[0]
                if output not in visited:
                    visited.add(output)
                    next_frontier.append((output, [*path, node]))
        frontier = next_frontier
    return None


def _ordered_causal_concat_inputs(
    inputs: Iterable[str], shapes: Mapping[str, list[int]]
) -> tuple[str, str]:
    """Return ``(cache, current)`` for the exported causal Concat.

    Wan emits ``torch.cat([cache_x, x], dim=2)``.  Ordering, rather than a
    one-frame heuristic, is the only correct discriminator after temporal
    upsampling because ``x`` can contain four frames.
    """

    values = list(inputs)
    if len(values) != 2:
        raise ValueError("causal cache concat must have two inputs")
    cache, current = values
    cache_shape = shapes.get(cache)
    current_shape = shapes.get(current)
    if not all(
        isinstance(shape, list) and len(shape) == 5
        for shape in (cache_shape, current_shape)
    ):
        raise ValueError("causal concat inputs have no static rank-5 shape")
    if not (1 <= int(cache_shape[2]) <= 2):
        raise ValueError("causal concat first input is not a 1/2-frame cache")
    if int(current_shape[2]) <= 0:
        raise ValueError("causal concat current activation has no frames")
    if any(
        int(cache_shape[index]) != int(current_shape[index]) for index in (0, 1, 3, 4)
    ):
        raise ValueError("causal concat cache/current shapes are incompatible")
    return cache, current


def analyze_fusion_graph(
    *,
    source_path: str | Path,
    graph_kind: str,
    target_module_names: tuple[str, ...],
    focus_module_prefix: str = "decoder.up_blocks.3",
) -> dict[str, Any]:
    """Prove the exact v5 boundary patterns eligible for fusion.

    The analyzer records unsupported patterns instead of guessing.  The build
    command requires every focused call site to be eligible before it can
    produce a plan.
    """

    if graph_kind not in {"initial", "steady"}:
        raise ValueError("graph_kind must be initial or steady")
    onnx, np, _TensorProto, helpers = _lazy_onnx()
    helper, numpy_helper = helpers
    model = onnx.load(str(source_path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    producer, consumers = _node_maps(model)
    nodes_by_name = {node.name: node for node in model.graph.node}
    shapes = _shape_map(model)
    initializers = {value.name: value for value in model.graph.initializer}
    constant_memo: dict[str, Any | None] = {}
    graph_outputs = {value.name for value in model.graph.output}
    target_set = set(target_module_names)
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    for conv in model.graph.node:
        match = _TARGET_CONV_RE.match(conv.name)
        if match is None or match.group("kind") != graph_kind:
            continue
        module_name = match.group("module")
        if module_name not in target_set:
            continue
        record: dict[str, Any] = {
            "call_site": conv.name,
            "module_name": module_name,
            "call_index": int(match.group("call")),
            "conv_kind": ("conv1" if module_name.endswith(".conv1") else "conv2"),
            "focused": focus_module_prefix in module_name,
            "eligible_input": False,
            "eligible_cache_update": False,
            "eligible_epilogue": False,
            "errors": [],
        }
        try:
            activation_dq = _one_producer(producer, conv.input[0], "DequantizeLinear")
            activation_q = _one_producer(
                producer, activation_dq.input[0], "QuantizeLinear"
            )
            activation_cast = _one_producer(producer, activation_q.input[0], "Cast")
            _require_only_consumers(
                tensor=activation_cast.output[0],
                consumers=consumers,
                allowed_node_names={activation_q.name},
            )
            _require_only_consumers(
                tensor=activation_q.output[0],
                consumers=consumers,
                allowed_node_names={activation_dq.name},
            )
            output_q = _one_consumer(consumers, conv.output[0], "QuantizeLinear")
            output_dq = _one_consumer(consumers, output_q.output[0], "DequantizeLinear")
            output_cast = _one_consumer(consumers, output_dq.output[0], "Cast")
            scale_array = _constant_array(
                name=activation_q.input[1],
                initializers=initializers,
                numpy_helper=numpy_helper,
                producer=producer,
                helper=helper,
                np_module=np,
                shapes=shapes,
                memo=constant_memo,
            )
            output_scale_array = _constant_array(
                name=output_q.input[1],
                initializers=initializers,
                numpy_helper=numpy_helper,
                producer=producer,
                helper=helper,
                np_module=np,
                shapes=shapes,
                memo=constant_memo,
            )
            if scale_array is None or np.asarray(scale_array).size != 1:
                raise ValueError("activation quantization scale is not static scalar")
            if output_scale_array is None or np.asarray(output_scale_array).size != 1:
                raise ValueError("output quantization scale is not static scalar")
            scale = float(np.asarray(scale_array).reshape(-1)[0])
            output_scale = float(np.asarray(output_scale_array).reshape(-1)[0])
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("activation quantization scale is invalid")
            if not math.isfinite(output_scale) or output_scale <= 0:
                raise ValueError("output quantization scale is invalid")

            padded = activation_cast.input[0]
            pad = producer.get(padded)
            if pad is None or pad.op_type != "Pad":
                raise ValueError("activation Cast is not preceded by an explicit Pad")
            pads = _constant_array(
                name=pad.input[1],
                initializers=initializers,
                numpy_helper=numpy_helper,
                producer=producer,
                helper=helper,
                np_module=np,
                shapes=shapes,
                memo=constant_memo,
            )
            if pads is None:
                raise ValueError("Pad extents are not static")
            pads_list = [int(value) for value in np.asarray(pads).reshape(-1)]
            if len(pads_list) != 10:
                raise ValueError(f"rank-5 Pad has invalid extents: {pads_list}")
            pad_value = 0.0
            if len(pad.input) > 2 and pad.input[2]:
                raw_pad_value = _constant_array(
                    name=pad.input[2],
                    initializers=initializers,
                    numpy_helper=numpy_helper,
                    producer=producer,
                    helper=helper,
                    np_module=np,
                    shapes=shapes,
                    memo=constant_memo,
                )
                if raw_pad_value is None:
                    raise ValueError("Pad value is not static")
                pad_value = float(np.asarray(raw_pad_value).reshape(-1)[0])
            if pad_value != 0.0:
                raise ValueError("causal padding must use zero")

            cat_or_current = pad.input[0]
            cat = producer.get(cat_or_current)
            current = cat_or_current
            cache = None
            concat_axis = None
            if cat is not None and cat.op_type == "Concat":
                attributes = _attribute_map(cat, helper)
                concat_axis = int(attributes.get("axis", -1))
                if concat_axis != 2 or len(cat.input) != 2:
                    raise ValueError("causal cache concat must have two inputs on T")
                # ``causal_conv3d_cat_pad`` emits ``cat([cache_x, x], dim=2)``.
                # The current activation is not necessarily one frame: after
                # temporal upsampling (notably in up_blocks.3) it can contain
                # four frames.  Inferring roles from T==1 would therefore
                # reject the exact hotspot this experiment is meant to test.
                cache, current = _ordered_causal_concat_inputs(cat.input, shapes)
            current_shape = shapes.get(current)
            cache_shape = shapes.get(cache) if cache else None
            padded_shape = shapes.get(activation_dq.output[0]) or shapes.get(padded)
            if current_shape is None or len(current_shape) != 5:
                raise ValueError("current activation has no static rank-5 shape")
            if cache is not None and (cache_shape is None or len(cache_shape) != 5):
                raise ValueError("feature cache has no static rank-5 shape")
            if padded_shape is None or len(padded_shape) != 5:
                raise ValueError("padded activation has no static rank-5 shape")
            record.update(
                {
                    "activation_dq_node": activation_dq.name,
                    "activation_q_node": activation_q.name,
                    "activation_cast_node": activation_cast.name,
                    "activation_scale_name": activation_q.input[1],
                    "activation_scale": scale,
                    "output_q_node": output_q.name,
                    "output_dq_node": output_dq.name,
                    "output_cast_node": output_cast.name,
                    "output_scale_name": output_q.input[1],
                    "output_scale": output_scale,
                    "current_tensor": current,
                    "current_shape": current_shape,
                    "cache_tensor": cache,
                    "cache_shape": cache_shape,
                    "padded_tensor": padded,
                    "padded_shape": padded_shape,
                    "pad_node": pad.name,
                    "concat_node": cat.name if cache is not None else None,
                    "concat_axis": concat_axis,
                    "pads": pads_list,
                    "output_q_tensor": output_q.output[0],
                    "output_cast_tensor": output_cast.output[0],
                    "eligible_input": True,
                }
            )
        except (KeyError, StopIteration, TypeError, ValueError) as exc:
            record["errors"].append(f"input_boundary:{exc}")
        records.append(record)

    by_module: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_module[record["module_name"]].append(record)
    for module_records in by_module.values():
        module_records.sort(key=lambda value: value["call_index"])
        if [record["call_index"] for record in module_records] != [0, 1, 2]:
            for record in module_records:
                record["errors"].append("module is not unrolled exactly three times")
            continue
        for index, record in enumerate(module_records):
            if not record["eligible_input"]:
                continue
            target = (
                module_records[index + 1].get("cache_tensor")
                if index < 2
                else _find_cache_output(
                    current=record["current_tensor"],
                    cache=record.get("cache_tensor"),
                    graph_outputs=graph_outputs,
                    shapes=shapes,
                    producer=producer,
                )
            )
            if isinstance(target, str) and target:
                expected_shape = shapes.get(target)
                if expected_shape is not None and len(expected_shape) == 5:
                    record["cache_update_tensor"] = target
                    record["cache_update_shape"] = expected_shape
                    record["eligible_cache_update"] = True
            if not record["eligible_cache_update"]:
                record["errors"].append("cache_update:exact output was not proven")

    for record in records:
        if not record["eligible_input"]:
            continue
        try:
            output_cast_tensor = record["output_cast_tensor"]
            if record["conv_kind"] == "conv2":
                found = _find_add_after(
                    tensor=output_cast_tensor,
                    producer=producer,
                    consumers=consumers,
                )
                if found is None:
                    raise ValueError("residual Add was not found")
                add, path = found
                residual = next(
                    value
                    for value in add.input
                    if value != (path[-1].output[0] if path else output_cast_tensor)
                )
                record.update(
                    {
                        "epilogue_mode": "conv2_residual",
                        "epilogue_output_tensor": add.output[0],
                        "epilogue_output_shape": shapes.get(add.output[0]),
                        "epilogue_residual_tensor": residual,
                        "epilogue_remove_nodes": [node.name for node in [*path, add]],
                    }
                )
                output_dq_node = nodes_by_name.get(record["output_dq_node"])
                output_cast_node = nodes_by_name.get(record["output_cast_node"])
                if output_dq_node is None or output_cast_node is None:
                    raise ValueError("output DQ/Cast node chain is incomplete")
                _require_removable_subgraph(
                    nodes=[output_dq_node, output_cast_node, *path, add],
                    replacement_output=add.output[0],
                    consumers=consumers,
                    graph_outputs=graph_outputs,
                )
                record["eligible_epilogue"] = True
            else:
                block_prefix = record["module_name"].rsplit(".conv1", 1)[0]
                conv2_module = f"{block_prefix}.conv2"
                peer = next(
                    value
                    for value in by_module.get(conv2_module, [])
                    if value["call_index"] == record["call_index"]
                )
                target = peer.get("current_tensor")
                if not isinstance(target, str):
                    raise ValueError("conv2 current activation is unavailable")
                subgraph = _collect_reverse_subgraph(
                    target=target,
                    source=output_cast_tensor,
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
                    raise ValueError(f"unexpected norm/SiLU ops: {unexpected}")
                produced = {value for node in subgraph for value in node.output}
                external = {
                    value
                    for node in subgraph
                    for value in node.input
                    if value and value != output_cast_tensor and value not in produced
                }
                output_shape = shapes.get(target)
                if output_shape is None or len(output_shape) != 5:
                    raise ValueError("norm/SiLU output shape is unavailable")
                channels = int(output_shape[1])
                gamma_candidates = []
                for name in external:
                    value = _constant_array(
                        name=name,
                        initializers=initializers,
                        numpy_helper=numpy_helper,
                        producer=producer,
                        helper=helper,
                        np_module=np,
                        shapes=shapes,
                        memo=constant_memo,
                    )
                    if value is not None and int(np.asarray(value).size) == channels:
                        gamma_candidates.append(name)
                if len(gamma_candidates) != 1:
                    raise ValueError(f"RMSNorm gamma is ambiguous: {gamma_candidates}")
                record.update(
                    {
                        "epilogue_mode": "conv1_norm_silu",
                        "epilogue_output_tensor": target,
                        "epilogue_output_shape": output_shape,
                        "epilogue_gamma_tensor": gamma_candidates[0],
                        "epilogue_remove_nodes": [node.name for node in subgraph],
                    }
                )
                output_dq_node = nodes_by_name.get(record["output_dq_node"])
                output_cast_node = nodes_by_name.get(record["output_cast_node"])
                if output_dq_node is None or output_cast_node is None:
                    raise ValueError("output DQ/Cast node chain is incomplete")
                _require_removable_subgraph(
                    nodes=[output_dq_node, output_cast_node, *subgraph],
                    replacement_output=target,
                    consumers=consumers,
                    graph_outputs=graph_outputs,
                )
                record["eligible_epilogue"] = True
        except (KeyError, StopIteration, TypeError, ValueError) as exc:
            record["errors"].append(f"epilogue:{exc}")

    focused = [record for record in records if record["focused"]]
    if not focused:
        errors.append(f"no target call sites matched {focus_module_prefix!r}")
    for record in focused:
        if not (
            record["eligible_input"]
            and record["eligible_cache_update"]
            and record["eligible_epilogue"]
        ):
            errors.append(f"{record['call_site']}: {record['errors']}")
    return {
        "schema_version": FUSION_SCHEMA_VERSION,
        "graph_kind": graph_kind,
        "source_file": str(Path(source_path).resolve()),
        "source_sha256": sha256_file(source_path),
        "focus_module_prefix": focus_module_prefix,
        "target_call_site_count": len(records),
        "focused_call_site_count": len(focused),
        "focused_complete": not errors,
        "errors": errors,
        "call_sites": records,
    }


def _topological_live_nodes(model: Any) -> list[Any]:
    """Drop dead nodes and return a deterministic topological ordering."""

    required = {value.name for value in model.graph.output}
    producer = {
        output: node for node in model.graph.node for output in node.output if output
    }
    live_names: set[str] = set()
    stack = list(required)
    while stack:
        tensor = stack.pop()
        node = producer.get(tensor)
        if node is None or node.name in live_names:
            continue
        live_names.add(node.name)
        stack.extend(value for value in node.input if value)
    nodes = [node for node in model.graph.node if node.name in live_names]
    by_output = {output: node for node in nodes for output in node.output if output}
    indegree: dict[str, int] = {node.name: 0 for node in nodes}
    downstream: dict[str, list[Any]] = defaultdict(list)
    for node in nodes:
        dependencies = {
            by_output[value].name
            for value in node.input
            if value in by_output and by_output[value].name != node.name
        }
        indegree[node.name] = len(dependencies)
        for dependency in dependencies:
            downstream[dependency].append(node)
    order_index = {node.name: index for index, node in enumerate(model.graph.node)}
    ready = [node for node in nodes if indegree[node.name] == 0]
    ready.sort(key=lambda node: order_index[node.name])
    result: list[Any] = []
    while ready:
        node = ready.pop(0)
        result.append(node)
        for consumer in downstream[node.name]:
            indegree[consumer.name] -= 1
            if indegree[consumer.name] == 0:
                ready.append(consumer)
                ready.sort(key=lambda value: order_index.get(value.name, 10**9))
    if len(result) != len(nodes):
        raise ValueError("fusion rewrite introduced an ONNX dependency cycle")
    return result


def rewrite_fusion_graph(
    *,
    source_path: str | Path,
    destination_path: str | Path,
    analysis: Mapping[str, Any],
    selected_call_sites: set[str],
    fuse_cache_update: bool,
    fuse_epilogue: bool,
    cache_update_mode: str = "separate",
    cache_update_modes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Replace only proven call sites with fusion-v1 custom nodes."""

    onnx, _np, TensorProto, helpers = _lazy_onnx()
    helper, _numpy_helper = helpers
    model = onnx.load(str(source_path), load_external_data=True)
    if analysis.get("source_sha256") != sha256_file(source_path):
        raise ValueError("fusion analysis does not belong to the source ONNX")
    records = {
        record["call_site"]: record
        for record in analysis.get("call_sites", [])
        if isinstance(record, dict)
    }
    if not selected_call_sites or not selected_call_sites <= records.keys():
        raise ValueError("selected fusion call sites are empty or unknown")
    if cache_update_mode not in {"dual", "separate"}:
        raise ValueError("cache_update_mode must be dual or separate")
    per_call_site_modes = dict(cache_update_modes or {})
    if not set(per_call_site_modes).issubset(selected_call_sites):
        raise ValueError("cache_update_modes contains an unselected call site")
    if any(value not in {"dual", "separate"} for value in per_call_site_modes.values()):
        raise ValueError("cache_update_modes contains an unsupported mode")
    selected = [records[name] for name in sorted(selected_call_sites)]
    for record in selected:
        if not record.get("eligible_input"):
            raise ValueError(f"input fusion is unsafe for {record['call_site']}")
        if fuse_cache_update and not record.get("eligible_cache_update"):
            raise ValueError(f"cache fusion is unsafe for {record['call_site']}")
        if fuse_epilogue and not record.get("eligible_epilogue"):
            raise ValueError(f"epilogue fusion is unsafe for {record['call_site']}")

    remove_nodes: set[str] = set()
    inserted: list[Any] = []
    value_infos: list[Any] = []
    for record in selected:
        call_site = record["call_site"]
        suffix = hashlib.sha256(call_site.encode()).hexdigest()[:12]
        pack_output = f"fusion/{suffix}/packed_int8"
        selected_cache_mode = per_call_site_modes.get(call_site, cache_update_mode)
        dual_cache_output = bool(fuse_cache_update and selected_cache_mode == "dual")
        pack_inputs = [record["current_tensor"]]
        if record.get("cache_tensor"):
            pack_inputs.append(record["cache_tensor"])
        pack_attributes = {
            "plugin_version": PLUGIN_VERSION,
            "plugin_namespace": PLUGIN_NAMESPACE,
            "scale": float(record["activation_scale"]),
            "has_cache": int(bool(record.get("cache_tensor"))),
            "emit_cache": int(dual_cache_output),
            "pads": [int(value) for value in record["pads"]],
            "output_shape": [int(value) for value in record["padded_shape"]],
            "cache_output_shape": [
                int(value)
                for value in (
                    record["cache_update_shape"] if dual_cache_output else [0] * 5
                )
            ],
            "current_shape": [int(value) for value in record["current_shape"]],
            "cache_shape": [
                int(value) for value in (record.get("cache_shape") or [0] * 5)
            ],
        }
        pack_outputs = [pack_output]
        if dual_cache_output:
            pack_outputs.append(record["cache_update_tensor"])
        inserted.append(
            helper.make_node(
                PACK_QUANT_PLUGIN,
                pack_inputs,
                pack_outputs,
                name=(
                    f"fusion/input_pack_quant_cache/{call_site}"
                    if dual_cache_output
                    else f"fusion/input_pack_quant/{call_site}"
                ),
                domain="com.sglang.sfwan",
                **pack_attributes,
            )
        )
        value_infos.append(
            helper.make_tensor_value_info(
                pack_output,
                TensorProto.INT8,
                record["padded_shape"],
            )
        )
        activation_dq = next(
            node
            for node in model.graph.node
            if node.name == record["activation_dq_node"]
        )
        activation_dq.input[0] = pack_output
        remove_nodes.update(
            {record["activation_cast_node"], record["activation_q_node"]}
        )

        if fuse_cache_update:
            cache_target = record["cache_update_tensor"]
            # Preserve the original tensor name.  In particular, the final
            # update may be a public cache_out_XXX binding whose ABI must not
            # change.  Its old single-output producer is removed below.
            cache_output = cache_target
            cache_inputs = [record["current_tensor"]]
            if record.get("cache_tensor"):
                cache_inputs.append(record["cache_tensor"])
            if selected_cache_mode == "separate":
                inserted.append(
                    helper.make_node(
                        CACHE_UPDATE_PLUGIN,
                        cache_inputs,
                        [cache_output],
                        name=f"fusion/cache_update/{call_site}",
                        domain="com.sglang.sfwan",
                        plugin_version=PLUGIN_VERSION,
                        plugin_namespace=PLUGIN_NAMESPACE,
                        has_cache=int(bool(record.get("cache_tensor"))),
                        output_shape=[
                            int(value) for value in record["cache_update_shape"]
                        ],
                        current_shape=[int(value) for value in record["current_shape"]],
                        cache_shape=[
                            int(value)
                            for value in (record.get("cache_shape") or [0] * 5)
                        ],
                    )
                )
            # ``cache_output`` deliberately reuses an existing tensor name so
            # every downstream consumer and public cache binding keeps the v5
            # ABI. Its rank/dtype metadata already exists in the source graph;
            # adding a duplicate ValueInfoProto can make ONNX validation or
            # TensorRT parsing version-dependent.
            original_producer = next(
                (node for node in model.graph.node if cache_target in node.output),
                None,
            )
            if original_producer is not None:
                if len([value for value in original_producer.output if value]) != 1:
                    raise ValueError(
                        f"cache producer has multiple outputs: {original_producer.name}"
                    )
                remove_nodes.add(original_producer.name)
        if fuse_epilogue:
            output_q = next(
                node
                for node in model.graph.node
                if node.name == record["output_q_node"]
            )
            epilogue_inputs = [output_q.output[0]]
            mode = record["epilogue_mode"]
            if mode == "conv1_norm_silu":
                epilogue_inputs.append(record["epilogue_gamma_tensor"])
                mode_id = 0
            elif mode == "conv2_residual":
                epilogue_inputs.append(record["epilogue_residual_tensor"])
                mode_id = 1
            else:
                raise ValueError(f"unsupported fusion epilogue mode: {mode}")
            epilogue_output = record["epilogue_output_tensor"]
            old_producer = next(
                (node for node in model.graph.node if epilogue_output in node.output),
                None,
            )
            if old_producer is not None:
                remove_nodes.add(old_producer.name)
            inserted.append(
                helper.make_node(
                    EPILOGUE_PLUGIN,
                    epilogue_inputs,
                    [epilogue_output],
                    name=f"fusion/{mode}/{call_site}",
                    domain="com.sglang.sfwan",
                    plugin_version=PLUGIN_VERSION,
                    plugin_namespace=PLUGIN_NAMESPACE,
                    mode=mode_id,
                    scale=float(record["output_scale"]),
                    output_shape=[
                        int(value) for value in record["epilogue_output_shape"]
                    ],
                )
            )
            remove_nodes.update(
                {
                    record["output_dq_node"],
                    record["output_cast_node"],
                    *record.get("epilogue_remove_nodes", []),
                }
            )

    original_names = {node.name for node in model.graph.node}
    if any(node.name in original_names for node in inserted):
        raise ValueError("fusion plugin node name collides with the source graph")
    kept = [node for node in model.graph.node if node.name not in remove_nodes]
    del model.graph.node[:]
    model.graph.node.extend([*kept, *inserted])
    model.graph.value_info.extend(value_infos)
    ordered = _topological_live_nodes(model)
    del model.graph.node[:]
    model.graph.node.extend(ordered)

    if not any(item.domain == "com.sglang.sfwan" for item in model.opset_import):
        model.opset_import.extend(
            [helper.make_opsetid("com.sglang.sfwan", FUSION_SCHEMA_VERSION)]
        )

    used_initializers = {
        value for node in model.graph.node for value in node.input if value
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
    counts = {
        "input_pack_quant": len(selected),
        "cache_update": len(selected) if fuse_cache_update else 0,
        "cache_update_dual": sum(
            fuse_cache_update
            and per_call_site_modes.get(record["call_site"], cache_update_mode)
            == "dual"
            for record in selected
        ),
        "cache_update_separate": sum(
            fuse_cache_update
            and per_call_site_modes.get(record["call_site"], cache_update_mode)
            == "separate"
            for record in selected
        ),
        "conv1_norm_silu": sum(
            record.get("epilogue_mode") == "conv1_norm_silu" for record in selected
        )
        if fuse_epilogue
        else 0,
        "conv2_residual": sum(
            record.get("epilogue_mode") == "conv2_residual" for record in selected
        )
        if fuse_epilogue
        else 0,
    }
    return {
        "schema_version": FUSION_SCHEMA_VERSION,
        "source_sha256": sha256_file(source_path),
        "destination": str(destination.resolve()),
        "destination_sha256": sha256_file(destination),
        "selected_call_sites": sorted(selected_call_sites),
        "selected_cache_update_modes": {
            record["call_site"]: per_call_site_modes.get(
                record["call_site"], cache_update_mode
            )
            for record in selected
            if fuse_cache_update
        },
        "fusion_counts": counts,
        "plugin_nodes": [node.name for node in inserted],
    }


__all__ = [
    "CACHE_UPDATE_PLUGIN",
    "EPILOGUE_PLUGIN",
    "FUSION_AUDIT_FILE",
    "FUSION_AUDIT_SCHEMA_VERSION",
    "FUSION_BUILD_STATE_FILE",
    "FUSION_ENGINE_FILES",
    "FUSION_INSPECTOR_FILES",
    "FUSION_MANIFEST_FILE",
    "FUSION_ONNX_FILES",
    "FUSION_PLUGIN_LIBRARY_FILE",
    "FUSION_PLUGIN_MANIFEST_FILE",
    "FUSION_PROBE_FILE",
    "FUSION_SCHEMA_VERSION",
    "FUSION_SUBDIRECTORY",
    "FUSION_TIMING_CACHE_FILE",
    "FUSION_VARIANT",
    "PACK_QUANT_PLUGIN",
    "PLUGIN_CREATORS",
    "PLUGIN_NAMESPACE",
    "PLUGIN_VERSION",
    "analyze_fusion_graph",
    "load_fusion_manifest",
    "rewrite_fusion_graph",
    "sha256_file",
    "validate_fusion_manifest",
    "write_json_atomic",
]
