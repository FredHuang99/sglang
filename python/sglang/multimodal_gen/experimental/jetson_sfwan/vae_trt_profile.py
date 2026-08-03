"""Import-light helpers for opt-in TensorRT VAE layer profiling.

This module deliberately does not import TensorRT, Torch, ONNX, or NumPy at
module import time.  Production VAE execution must not acquire a profiler or
read any of the profiling artifacts unless the dedicated CLI switch is set.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .vae_trt_qdq import EXPECTED_CALL_SITES, QDQ_SCHEMA_VERSION

TRT_LAYER_PROFILE_SCHEMA_VERSION = 2
SUPPORTED_TRT_LAYER_PROFILE_SCHEMA_VERSIONS = frozenset({1, 2})
TRT_LAYER_PROFILE_MANIFEST_FILE = "trt_layer_profile_manifest.json"
TRT_LAYER_PROFILE_SCOPE = "vae_profile_only"

PROFILE_CATEGORIES = (
    "target_quantized_conv",
    "target_qdq_cast_reformat",
    "non_target_conv",
    "attention",
    "upsample_resample",
    "norm_activation_residual",
    "cache_layout_copy",
    "fused_input_pack_quant",
    "fused_cache_update",
    "fused_conv1_norm_silu",
    "fused_conv2_residual",
    "other",
)
LEGACY_PROFILE_CATEGORIES = tuple(
    category for category in PROFILE_CATEGORIES if not category.startswith("fused_")
)

# Layer callbacks and the outer CUDA event are different instrumentation
# mechanisms.  A 10% tolerance is intentionally diagnostic rather than a
# claim that either clock is the ground truth.
MIN_LAYER_SUM_OVER_ENGINE_RATIO = 0.90
MAX_LAYER_SUM_OVER_ENGINE_RATIO = 1.10
MAX_OTHER_CATEGORY_FRACTION = 0.10


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _resolve_artifact(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} file name is invalid")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the engine directory") from exc
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _validate_digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{label} SHA256 is invalid")
    return value.lower()


def load_trt_layer_profile_manifest(engine_dir: str | Path) -> dict[str, Any]:
    """Load the opt-in manifest without importing TensorRT or Torch."""

    root = Path(engine_dir).expanduser().resolve()
    path = root / TRT_LAYER_PROFILE_MANIFEST_FILE
    if not path.is_file():
        raise ValueError(f"TensorRT layer-profile manifest does not exist: {path}")
    return _read_json_object(path, label="TensorRT layer-profile manifest")


def validate_trt_layer_profile_manifest(
    manifest: Mapping[str, Any],
    *,
    engine_dir: str | Path,
    precision: str,
    production_manifest: Mapping[str, Any] | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Validate and resolve one precision's diagnostic plans.

    The profile manifest owns only diagnostic artifacts.  INT8 records must
    point at the already-audited v5 production plans; FP16 records must point
    at the separately built same-source DETAILED plans.
    """

    if not isinstance(manifest, Mapping):
        raise ValueError("TensorRT layer-profile manifest must be an object")
    if manifest.get("schema_version") not in (
        SUPPORTED_TRT_LAYER_PROFILE_SCHEMA_VERSIONS
    ):
        raise ValueError(
            "unsupported TensorRT layer-profile manifest schema: "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("scope") != TRT_LAYER_PROFILE_SCOPE:
        raise ValueError("TensorRT layer-profile manifest scope is invalid")
    if precision not in {"fp16", "int8"}:
        raise ValueError(f"unsupported TensorRT layer-profile precision: {precision}")

    root = Path(engine_dir).expanduser().resolve()
    engines = manifest.get("engines")
    if not isinstance(engines, Mapping):
        # Accept ``plans`` as an early builder spelling while still returning
        # one normalized contract to the runtime.
        engines = manifest.get("plans")
    if not isinstance(engines, Mapping):
        raise ValueError("TensorRT layer-profile manifest has no engine records")
    precision_records = engines.get(precision)
    if not isinstance(precision_records, Mapping):
        raise ValueError(f"layer-profile manifest has no {precision} plans")

    expected_plan_kind = (
        "same_source_fp16" if precision == "fp16" else "audited_int8_v5"
    )
    plan_kind = precision_records.get("plan_kind")
    if plan_kind is None:
        plan_kind = (
            manifest.get("plan_kind", {}).get(precision)
            if isinstance(manifest.get("plan_kind"), Mapping)
            else None
        )
    if plan_kind is not None and plan_kind != expected_plan_kind:
        raise ValueError(
            f"TensorRT {precision} layer-profile plan kind is {plan_kind!r}; "
            f"expected {expected_plan_kind!r}"
        )

    normalized: dict[str, Any] = {}
    for kind in ("initial", "steady"):
        record = precision_records.get(kind)
        if not isinstance(record, Mapping):
            raise ValueError(f"layer-profile manifest has no {precision}/{kind} plan")
        record_plan_kind = record.get("plan_kind", plan_kind)
        if record_plan_kind != expected_plan_kind:
            raise ValueError(
                f"TensorRT {precision}/{kind} layer-profile plan kind is "
                f"{record_plan_kind!r}; expected {expected_plan_kind!r}"
            )
        if str(record.get("profiling_verbosity", "")).lower() != "detailed":
            raise ValueError(
                f"TensorRT {precision}/{kind} layer-profile plan is not DETAILED"
            )
        plan_path = _resolve_artifact(
            root, record.get("file"), label=f"TensorRT {precision}/{kind} plan"
        )
        plan_digest = _validate_digest(
            record.get("sha256"), label=f"TensorRT {precision}/{kind} plan"
        )
        inspector_path = _resolve_artifact(
            root,
            record.get("inspector_file"),
            label=f"TensorRT {precision}/{kind} inspector",
        )
        inspector_digest = _validate_digest(
            record.get("inspector_sha256"),
            label=f"TensorRT {precision}/{kind} inspector",
        )
        if verify_hashes:
            if sha256_file(plan_path) != plan_digest:
                raise ValueError(
                    f"TensorRT {precision}/{kind} layer-profile plan digest mismatch"
                )
            if sha256_file(inspector_path) != inspector_digest:
                raise ValueError(
                    f"TensorRT {precision}/{kind} inspector digest mismatch"
                )
        inspector = _read_json_value(
            inspector_path, label=f"TensorRT {precision}/{kind} inspector"
        )
        normalized[kind] = {
            **dict(record),
            "path": str(plan_path),
            "sha256": plan_digest,
            "inspector_path": str(inspector_path),
            "inspector_sha256": inspector_digest,
            "inspector": inspector,
        }

    production_record = manifest.get("production_manifest")
    if not isinstance(production_record, Mapping):
        raise ValueError("layer-profile manifest has no production-manifest binding")
    production_path = _resolve_artifact(
        root,
        production_record.get("file"),
        label="production TensorRT VAE manifest",
    )
    production_digest = _validate_digest(
        production_record.get("sha256"),
        label="production TensorRT VAE manifest",
    )
    if verify_hashes and sha256_file(production_path) != production_digest:
        raise ValueError("production TensorRT VAE manifest digest mismatch")
    loaded_production_manifest = _read_json_object(
        production_path, label="production TensorRT VAE manifest"
    )
    if production_manifest is None:
        production_manifest = loaded_production_manifest
    elif dict(production_manifest) != loaded_production_manifest:
        raise ValueError("supplied production manifest differs from the bound file")

    if (
        manifest.get("batch_size") != 1
        or manifest.get("height") != 480
        or manifest.get("width") != 832
        or manifest.get("latent_shape") != [1, 16, 3, 60, 104]
        or manifest.get("latent_dtype") != "float16"
    ):
        raise ValueError("TensorRT layer-profile fixed model contract is invalid")
    cache = manifest.get("cache")
    if (
        not isinstance(cache, Mapping)
        or cache.get("tensor_count") != 32
        or cache.get("dtype") != "float16"
        or not isinstance(cache.get("bindings"), list)
        or len(cache["bindings"]) != 32
    ):
        raise ValueError("TensorRT layer-profile feature-cache contract is invalid")
    for index, binding in enumerate(cache["bindings"]):
        if (
            not isinstance(binding, Mapping)
            or binding.get("index") != index
            or binding.get("dtype") != "float16"
            or not isinstance(binding.get("shape"), list)
            or len(binding["shape"]) != 5
        ):
            raise ValueError(f"TensorRT layer-profile cache binding {index} is invalid")

    build = manifest.get("build")
    if (
        not isinstance(build, Mapping)
        or build.get("scope") != TRT_LAYER_PROFILE_SCOPE
        or build.get("compute_capability") != [8, 7]
        or str(build.get("profiling_verbosity", "")).lower() != "detailed"
        or not isinstance(build.get("tensorrt_version"), str)
        or not isinstance(build.get("cuda_version"), str)
    ):
        raise ValueError("TensorRT layer-profile build identity is invalid")

    if precision == "fp16":
        production_build = production_manifest.get("build")
        source_records = (
            production_build.get("qdq_source_onnx")
            if isinstance(production_build, Mapping)
            else None
        )
        if not isinstance(source_records, Mapping):
            raise ValueError("production manifest has no FP16 Q/DQ source binding")
        for kind in ("initial", "steady"):
            record = normalized[kind]
            source_path = _resolve_artifact(
                root,
                record.get("source_onnx_file"),
                label=f"TensorRT {kind} layer-profile source ONNX",
            )
            source_digest = _validate_digest(
                record.get("source_onnx_sha256"),
                label=f"TensorRT {kind} layer-profile source ONNX",
            )
            production_source = source_records.get(kind)
            if (
                not isinstance(production_source, Mapping)
                or production_source.get("file") != record.get("source_onnx_file")
                or production_source.get("sha256") != source_digest
            ):
                raise ValueError(
                    f"TensorRT {kind} profile plan is not bound to the production "
                    "opset-19 FP16 source"
                )
            if verify_hashes and sha256_file(source_path) != source_digest:
                raise ValueError(
                    f"TensorRT {kind} layer-profile source ONNX digest mismatch"
                )
            target_call_sites = _validate_fp16_target_call_sites(
                record.get("target_call_sites"), engine_kind=kind
            )
            if record.get("target_call_site_count") != EXPECTED_CALL_SITES:
                raise ValueError(
                    f"TensorRT {kind} FP16 target-call-site count is invalid"
                )
            expected_catalog_digest = _validate_digest(
                record.get("inspector_catalog_sha256"),
                label=f"TensorRT {kind} FP16 Inspector catalog",
            )
            catalog = build_physical_layer_catalog(
                engine_kind=kind,
                plan_sha256=record["sha256"],
                inspector=record["inspector"],
                precision="fp16",
                fp16_target_call_sites=target_call_sites,
            )
            if catalog["catalog_sha256"] != expected_catalog_digest:
                raise ValueError(
                    f"TensorRT {kind} FP16 Inspector target catalog changed"
                )
            record["target_call_sites"] = target_call_sites

    int8_audit: dict[str, Any] | None = None
    if precision == "int8":
        quantization = production_manifest.get("quantization")
        if (
            not isinstance(quantization, Mapping)
            or quantization.get("qdq_schema_version") != QDQ_SCHEMA_VERSION
        ):
            raise ValueError(
                f"INT8 layer profiling requires Q/DQ schema {QDQ_SCHEMA_VERSION}"
            )
        audit_record = production_manifest.get("int8_audit")
        if (
            not isinstance(audit_record, Mapping)
            or audit_record.get("passed") is not True
        ):
            raise ValueError("INT8 layer profiling requires a passing v5 audit")
        audit_path = _resolve_artifact(
            root, audit_record.get("report_file"), label="TensorRT INT8 v5 audit"
        )
        profile_audit = manifest.get("int8_audit")
        if not isinstance(profile_audit, Mapping):
            raise ValueError("layer-profile manifest has no INT8 audit binding")
        profile_audit_path = _resolve_artifact(
            root, profile_audit.get("file"), label="layer-profile INT8 v5 audit"
        )
        if profile_audit_path != audit_path:
            raise ValueError("layer-profile and production INT8 audit paths differ")
        profile_audit_digest = _validate_digest(
            profile_audit.get("sha256"), label="layer-profile INT8 v5 audit"
        )
        if verify_hashes and sha256_file(audit_path) != profile_audit_digest:
            raise ValueError("layer-profile INT8 v5 audit digest mismatch")
        int8_audit = _read_json_object(audit_path, label="TensorRT INT8 v5 audit")
        if (
            int8_audit.get("schema_version") != QDQ_SCHEMA_VERSION
            or int8_audit.get("passed") is not True
            or int8_audit.get("complete") is not True
            or int8_audit.get("errors") != []
        ):
            raise ValueError("TensorRT INT8 v5 audit is incomplete")
        audit_plan_sha = int8_audit.get("plan_sha256")
        if not isinstance(audit_plan_sha, Mapping):
            raise ValueError("TensorRT INT8 v5 audit has no plan digests")
        profile_plan_sha = profile_audit.get("plan_sha256")
        if not isinstance(profile_plan_sha, Mapping):
            raise ValueError("layer-profile INT8 audit has no plan digests")
        for kind in ("initial", "steady"):
            if (
                audit_plan_sha.get(kind) != normalized[kind]["sha256"]
                or profile_plan_sha.get(kind) != normalized[kind]["sha256"]
            ):
                raise ValueError(
                    f"TensorRT INT8 {kind} profile plan differs from audited plan"
                )

    return {
        "schema_version": TRT_LAYER_PROFILE_SCHEMA_VERSION,
        "scope": TRT_LAYER_PROFILE_SCOPE,
        "precision": precision,
        "plan_kind": expected_plan_kind,
        "engines": normalized,
        "int8_audit": int8_audit,
        "build": dict(build),
    }


def _read_json_value(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc


def probe_trt_layer_profile_api(
    *, trt: Any, contexts: Mapping[str, Any] | Sequence[Any]
) -> dict[str, Any]:
    """Fail closed unless every TensorRT profiler API used at runtime exists."""

    missing: list[str] = []
    if not hasattr(trt, "IProfiler"):
        missing.append("tensorrt.IProfiler")
    values = (
        list(contexts.items())
        if isinstance(contexts, Mapping)
        else [(str(index), context) for index, context in enumerate(contexts)]
    )
    if not values:
        missing.append("execution_context")
    for name, context in values:
        for attribute in ("profiler", "enqueue_emits_profile"):
            try:
                getattr(context, attribute)
            except (AttributeError, RuntimeError, TypeError):
                missing.append(f"{name}.{attribute}")
        try:
            reporter = getattr(context, "report_to_profiler")
        except (AttributeError, RuntimeError, TypeError):
            reporter = None
        if not callable(reporter):
            missing.append(f"{name}.report_to_profiler")
    if missing:
        raise RuntimeError(
            "TensorRT layer profiling is unavailable; missing APIs: "
            + ", ".join(sorted(set(missing)))
        )
    return {
        "available": True,
        "context_count": len(values),
        "required_apis": [
            "tensorrt.IProfiler",
            "context.profiler",
            "context.enqueue_emits_profile",
            "context.report_to_profiler",
        ],
    }


def create_trt_profiler_class(trt: Any) -> type[Any]:
    """Create the TensorRT subclass lazily for the installed runtime."""

    if not hasattr(trt, "IProfiler"):
        raise RuntimeError("TensorRT does not expose IProfiler")

    class _SFWanTensorRTProfiler(trt.IProfiler):  # type: ignore[misc, name-defined]
        def __init__(self, callback: Callable[[str, float], None]) -> None:
            trt.IProfiler.__init__(self)
            self._callback = callback

        def report_layer_time(self, layer_name: str, ms: float) -> None:
            self._callback(str(layer_name), float(ms))

    _SFWanTensorRTProfiler.__name__ = "SFWanTensorRTLayerProfiler"
    return _SFWanTensorRTProfiler


def _inspector_layers(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("TensorRT Inspector output is invalid JSON") from exc
    if isinstance(value, list):
        return [dict(record) for record in value if isinstance(record, Mapping)]
    if not isinstance(value, Mapping):
        raise ValueError("TensorRT Inspector output must be a list or object")
    for key in ("Layers", "layers"):
        records = value.get(key)
        if isinstance(records, list):
            return [dict(record) for record in records if isinstance(record, Mapping)]
    return [dict(value)]


def _first_text(record: Mapping[str, Any], *keys: str) -> str:
    normalized = {
        str(key).lower().replace("_", ""): value for key, value in record.items()
    }
    for key in keys:
        value = normalized.get(key.lower().replace("_", ""))
        if isinstance(value, str):
            return value
    return ""


def _audit_layer_mapping(
    *, int8_audit: Mapping[str, Any] | None, engine_kind: str
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    names: dict[str, set[str]] = defaultdict(set)
    metadata: dict[str, set[str]] = defaultdict(set)
    if not isinstance(int8_audit, Mapping):
        return names, metadata
    tactics = int8_audit.get("tactics")
    tactic = tactics.get(engine_kind) if isinstance(tactics, Mapping) else None
    matches = tactic.get("matches") if isinstance(tactic, Mapping) else None
    if not isinstance(matches, Mapping):
        return names, metadata
    for call_site, evidence in matches.items():
        if not isinstance(call_site, str) or not isinstance(evidence, Mapping):
            continue
        for name in evidence.get("layer_names", []):
            if isinstance(name, str) and name:
                names[name].add(call_site)
        for value in evidence.get("metadata", []):
            if isinstance(value, str) and value:
                metadata[value].add(call_site)
    return names, metadata


def _validate_fp16_target_call_sites(
    value: Any, *, engine_kind: str
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != EXPECTED_CALL_SITES:
        raise ValueError(
            f"FP16 {engine_kind} target map must contain "
            f"{EXPECTED_CALL_SITES} call sites"
        )
    normalized: list[dict[str, Any]] = []
    logical_ids: set[str] = set()
    source_nodes: set[str] = set()
    module_calls: dict[str, set[int]] = defaultdict(set)
    for record in value:
        if not isinstance(record, Mapping):
            raise ValueError("FP16 target-map entries must be objects")
        logical_id = record.get("logical_call_site")
        module_name = record.get("module_name")
        call_index = record.get("call_index")
        source_node = record.get("source_onnx_node_name")
        weight = record.get("source_weight_initializer")
        if (
            not isinstance(logical_id, str)
            or not logical_id.startswith(f"int8/{engine_kind}/")
            or not isinstance(module_name, str)
            or not module_name
            or isinstance(call_index, bool)
            or not isinstance(call_index, int)
            or call_index not in {0, 1, 2}
            or not isinstance(source_node, str)
            or not source_node
            or not isinstance(weight, str)
            or not weight
        ):
            raise ValueError(f"invalid FP16 {engine_kind} target-map entry: {record}")
        if logical_id in logical_ids or source_node in source_nodes:
            raise ValueError(
                "FP16 target-map identifiers and source nodes must be unique"
            )
        logical_ids.add(logical_id)
        source_nodes.add(source_node)
        module_calls[module_name].add(call_index)
        normalized.append(dict(record))
    if len(module_calls) * 3 != EXPECTED_CALL_SITES or any(
        calls != {0, 1, 2} for calls in module_calls.values()
    ):
        raise ValueError(
            "FP16 target map must contain call indices 0, 1, and 2 for every "
            "logical residual Conv"
        )
    return normalized


def _contains_onnx_node_marker(text: str, marker: str) -> bool:
    """Match a complete ONNX node marker, excluding prefix collisions.

    ONNX exports commonly contain names such as ``.../Conv`` and
    ``.../Conv_1``.  A plain substring search would incorrectly map both to
    the shorter marker.  TensorRT may surround a source node with fusion
    punctuation, so non-identifier delimiters remain valid boundaries.
    """

    start = 0
    while True:
        index = text.find(marker, start)
        if index < 0:
            return False
        before = text[index - 1] if index > 0 else ""
        end = index + len(marker)
        after = text[end] if end < len(text) else ""
        if (not before or not (before.isalnum() or before == "_")) and (
            not after or not (after.isalnum() or after == "_")
        ):
            return True
        start = index + 1


def _is_convolution_record(
    *, layer_type: str, parameter_type: str, tactic_name: str
) -> bool:
    evidence = " ".join((layer_type, parameter_type, tactic_name)).lower()
    return "conv" in evidence


def _classify_layer(
    *,
    name: str,
    layer_type: str,
    metadata: str,
    parameter_type: str,
    logical_call_sites: Sequence[str],
    logical_mapping_source: str | None = None,
) -> tuple[str, str]:
    haystack = " ".join((name, layer_type, metadata, parameter_type)).lower()
    type_text = " ".join((layer_type, parameter_type)).lower()
    if (
        "sfwancausalpackquant" in haystack
        or "sfwan_causal_pack_quant" in haystack
        or "fusion/input_pack_quant/" in haystack
        or "fusion/input_pack_quant_cache/" in haystack
    ):
        return "fused_input_pack_quant", "fusion_v1_plugin"
    if "sfwancacheupdate" in haystack or "sfwan_cache_update" in haystack:
        return "fused_cache_update", "fusion_v1_plugin"
    if "sfwanint8epilogue" in haystack or "sfwan_int8_epilogue" in haystack:
        if any(marker in haystack for marker in ("conv1", "norm_silu", "normsilu")):
            return "fused_conv1_norm_silu", "fusion_v1_plugin"
        if any(marker in haystack for marker in ("conv2", "residual")):
            return "fused_conv2_residual", "fusion_v1_plugin"
        return "other", "fusion_v1_plugin_unresolved"
    # A v5 audit mapping is stronger evidence than a fused TensorRT layer type:
    # fusion may rename the physical layer so it no longer says "Convolution".
    if logical_call_sites:
        return "target_quantized_conv", logical_mapping_source or "v5_audit"
    target_quantization_scope = any(
        marker in haystack
        for marker in (
            "qdq/initial/",
            "qdq/steady/",
            "int8/initial/",
            "int8/steady/",
        )
    )
    # The Q/DQ rewriter owns these namespaces.  TensorRT 10.3 may lower a
    # Cast/Q/DQ node to a zero-cost NoOp or Constant and retain only the scoped
    # ONNX name, so requiring the physical layer type to still say Cast or
    # Reformat would incorrectly leave valid quantization scaffolding in
    # ``other``.
    if target_quantization_scope:
        return "target_qdq_cast_reformat", "name_rule"
    if any(marker in haystack for marker in ("attention", "sdpa", "scaled_dot")):
        return "attention", "name_rule"
    if any(
        marker in haystack
        for marker in ("upsample", "resample", "interpolate", "resize")
    ):
        return "upsample_resample", "name_rule"

    # TensorRT 10.3 lowers the Wan RMSNorm path into ReduceL2/Reduce plus
    # fused elementwise/kgen layers.  Those physical layers commonly retain
    # only ``.../norm*/ReduceL2`` or ``.../nonlinearity*/Sigmoid`` in their
    # Inspector names, rather than the generic words previously checked here.
    # Attention is checked above so an attention reduction is not mislabeled
    # as decoder normalization.
    norm_or_activation = any(
        marker in haystack for marker in ("/norm", "norm_", "reducel2")
    ) or any(marker in haystack for marker in ("/nonlinearity", "sigmoid", "tanh"))
    compiler_elementwise = "kgen" in type_text and any(
        marker in haystack
        for marker in (
            "castcastaddcast",
            "castcastmulcast",
            "castcastsubcast",
            "castcastdivcast",
        )
    )
    residual_reformat = (
        "reformat" in type_text and "pwn(" in haystack and "/add" in haystack
    )
    if (
        any(
            marker in haystack
            for marker in (
                "rmsnorm",
                "layernorm",
                "normalization",
                "silu",
                "activation",
                "residual",
            )
        )
        or norm_or_activation
        or compiler_elementwise
        or residual_reformat
    ):
        return "norm_activation_residual", "name_rule"
    cache_scope = any(
        marker in haystack
        for marker in (
            "cache_",
            "cache/",
            "cachein",
            "cacheout",
            "feature_cache",
            "feature cache",
        )
    )
    cache_operation = any(
        marker in haystack
        for marker in (
            "concatenation",
            "concat",
            "slice",
            "shuffle",
            "reshape",
            "transpose",
            "permute",
            "copy",
        )
    )
    if cache_scope and cache_operation:
        return "cache_layout_copy", "name_rule"

    # Exported Wan causal state updates are static Slice operations.  The
    # TensorRT compiler may fuse a Slice and dtype/layout conversion into a
    # generated ``SlicCast`` kernel and erase the original cache tensor name.
    # Upsample/attention/norm paths were handled above; a remaining Slice or
    # SlicCast in these fixed initial/steady graphs is therefore the physical
    # feature-cache update/copy path.  Generic Reformat layers remain ``other``
    # unless they carry explicit cache evidence, avoiding a false claim that
    # every TensorRT layout conversion is feature-cache traffic.
    causal_cache_slice = (
        "slice" in type_text
        or "sliccast" in haystack
        or (
            "slice" in haystack
            and any(marker in type_text for marker in ("reformat", "shuffle", "kgen"))
        )
    )
    if causal_cache_slice:
        return "cache_layout_copy", "name_rule"
    if any(marker in type_text for marker in ("convolution", "conv")):
        return "non_target_conv", "inspector"
    return "other", "unknown"


def _refine_int8_boundary_kernels(entries: list[dict[str, Any]]) -> None:
    """Classify anonymous compiler layout kernels feeding audited INT8 Conv.

    TensorRT 10.3's compiler backend erases the ONNX Q/DQ name from several
    physical activation-boundary kernels.  On Orin these appear as short
    ``kgen`` runs such as ``TranReshSlic -> ReshTran`` immediately before a
    v5-audited INT8 convolution.  The ordered Inspector catalog and the audit
    mapping together provide stronger evidence than the generated name alone:
    the consumer is one of the exact 84 target call sites and its audited
    activation input is INT8.

    Only a contiguous run of at most three recognized layout kernels directly
    preceding such a consumer is reclassified.  Generic kgen/Reformat work
    elsewhere remains ``other``; this avoids turning proximity to an arbitrary
    convolution into quantization evidence.
    """

    layout_markers = (
        "tranreshslic",
        "tranresh",
        "reshtran",
        "slicresh",
        "__myl_tran_",
    )
    target_indices = [
        index
        for index, entry in enumerate(entries)
        if entry["category"] == "target_quantized_conv"
    ]
    for target_index in target_indices:
        for offset in range(1, 4):
            index = target_index - offset
            if index < 0:
                break
            entry = entries[index]
            if entry["category"] != "other":
                break
            type_text = " ".join(
                (str(entry.get("layer_type", "")), str(entry.get("parameter_type", "")))
            ).lower()
            evidence = " ".join(
                (str(entry.get("name", "")), str(entry.get("tactic_name", "")))
            ).lower()
            if "kgen" not in type_text or not any(
                marker in evidence for marker in layout_markers
            ):
                break
            entry["category"] = "target_qdq_cast_reformat"
            entry["classification_source"] = "int8_audited_boundary"


def build_physical_layer_catalog(
    *,
    engine_kind: str,
    plan_sha256: str,
    inspector: Any,
    precision: str,
    int8_audit: Mapping[str, Any] | None = None,
    fp16_target_call_sites: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an ordered Inspector catalog without assuming ONNX=TRT 1:1."""

    if engine_kind not in {"initial", "steady"}:
        raise ValueError(f"invalid TensorRT VAE engine kind: {engine_kind}")
    if precision not in {"fp16", "int8"}:
        raise ValueError(f"invalid TensorRT VAE precision: {precision}")
    if precision == "int8" and fp16_target_call_sites is not None:
        raise ValueError("FP16 source target map cannot be used with an INT8 plan")
    plan_sha256 = _validate_digest(plan_sha256, label="TensorRT profile plan")
    if precision == "int8":
        fusion_audit = (
            isinstance(int8_audit, Mapping) and int8_audit.get("variant") == "fusion_v1"
        )
        valid_schema = (
            int8_audit.get("schema_version") == 1
            and int8_audit.get("qdq_schema_version") == QDQ_SCHEMA_VERSION
            if fusion_audit
            else isinstance(int8_audit, Mapping)
            and int8_audit.get("schema_version") == QDQ_SCHEMA_VERSION
        )
        if (
            not isinstance(int8_audit, Mapping)
            or not valid_schema
            or int8_audit.get("passed") is not True
            or int8_audit.get("complete") is not True
            or int8_audit.get("errors") != []
        ):
            raise ValueError(
                "INT8 physical-layer catalog requires a passing v5 or fusion-v1 audit"
            )
        audit_plan_sha = int8_audit.get("plan_sha256")
        if (
            not isinstance(audit_plan_sha, Mapping)
            or audit_plan_sha.get(engine_kind) != plan_sha256
        ):
            raise ValueError("INT8 physical-layer catalog plan/audit digest mismatch")
        tactics = int8_audit.get("tactics")
        tactic = tactics.get(engine_kind) if isinstance(tactics, Mapping) else None
        if (
            not isinstance(tactic, Mapping)
            or tactic.get("passed") is not True
            or tactic.get("mapped_count") != EXPECTED_CALL_SITES
            or tactic.get("errors") != []
        ):
            raise ValueError("INT8 physical-layer catalog tactic audit is incomplete")
    audit_names, audit_metadata = _audit_layer_mapping(
        int8_audit=int8_audit, engine_kind=engine_kind
    )
    fp16_targets = (
        _validate_fp16_target_call_sites(
            list(fp16_target_call_sites), engine_kind=engine_kind
        )
        if fp16_target_call_sites is not None
        else []
    )
    occurrences: dict[str, int] = defaultdict(int)
    entries: list[dict[str, Any]] = []
    for record in _inspector_layers(inspector):
        name = _first_text(record, "Name", "LayerName")
        if not name:
            raise ValueError("TensorRT Inspector layer has no exact name")
        occurrence = occurrences[name]
        occurrences[name] += 1
        layer_type = _first_text(record, "LayerType", "Type")
        metadata = _first_text(record, "Metadata")
        parameter_type = _first_text(record, "ParameterType")
        tactic_name = _first_text(record, "TacticName")
        call_sites = set(audit_names.get(name, set()))
        call_sites.update(audit_metadata.get(metadata, set()))
        # Some TensorRT releases preserve the ONNX call-site only in Metadata.
        # This is still tied to the audited call-site set, not inferred from an
        # arbitrary substring such as "int8".
        if int8_audit is not None and metadata:
            all_audited = set().union(*audit_names.values(), *audit_metadata.values())
            call_sites.update(site for site in all_audited if site in metadata)
        mapping_source = (
            "fusion_v1_audit"
            if call_sites
            and isinstance(int8_audit, Mapping)
            and int8_audit.get("variant") == "fusion_v1"
            else "v5_audit"
            if call_sites
            else None
        )
        if fp16_targets and _is_convolution_record(
            layer_type=layer_type,
            parameter_type=parameter_type,
            tactic_name=tactic_name,
        ):
            for target in fp16_targets:
                source_node = str(target["source_onnx_node_name"])
                if _contains_onnx_node_marker(
                    name, source_node
                ) or _contains_onnx_node_marker(metadata, source_node):
                    call_sites.add(str(target["logical_call_site"]))
                    mapping_source = "source_onnx"
        category, source = _classify_layer(
            name=name,
            layer_type=layer_type,
            metadata=metadata,
            parameter_type=parameter_type,
            logical_call_sites=sorted(call_sites),
            logical_mapping_source=mapping_source,
        )
        entries.append(
            {
                "index": len(entries),
                "name": name,
                "occurrence_index": occurrence,
                "exact_key": f"{name}#{occurrence}",
                "layer_type": layer_type,
                "parameter_type": parameter_type,
                "tactic_name": tactic_name,
                "category": category,
                "logical_call_sites": sorted(call_sites),
                "classification_source": source,
            }
        )
    if not entries:
        raise ValueError("TensorRT Inspector returned no physical layers")
    if precision == "int8":
        _refine_int8_boundary_kernels(entries)
    if precision == "int8" or fp16_targets:
        mapped_call_sites = {
            call_site for entry in entries for call_site in entry["logical_call_sites"]
        }
        if len(mapped_call_sites) != EXPECTED_CALL_SITES:
            raise ValueError(
                f"{precision.upper()} physical-layer catalog maps "
                f"{len(mapped_call_sites)} target "
                f"call sites; expected {EXPECTED_CALL_SITES}"
            )
        expected_call_sites = {
            str(target["logical_call_site"]) for target in fp16_targets
        }
        if expected_call_sites and mapped_call_sites != expected_call_sites:
            raise ValueError(
                "FP16 physical-layer catalog mapped a different target-call-site set"
            )
    catalog = {
        "schema_version": TRT_LAYER_PROFILE_SCHEMA_VERSION,
        "engine_kind": engine_kind,
        "precision": precision,
        "plan_sha256": plan_sha256,
        "expected_target_call_site_count": (
            EXPECTED_CALL_SITES if precision == "int8" or fp16_targets else 0
        ),
        "layers": entries,
    }
    catalog["catalog_sha256"] = catalog_sha256(catalog)
    return catalog


def catalog_sha256(catalog: Mapping[str, Any]) -> str:
    value = dict(catalog)
    value.pop("catalog_sha256", None)
    return sha256_json(value)


def _runtime_catalog_from_callbacks(
    catalog: Mapping[str, Any], callback_keys: Sequence[tuple[str, int]]
) -> dict[str, Any]:
    candidates = {
        (entry.get("name"), entry.get("occurrence_index")): entry
        for entry in catalog.get("layers", [])
        if isinstance(entry, Mapping)
    }
    layers: list[dict[str, Any]] = []
    for index, key in enumerate(callback_keys):
        candidate = candidates.get(key)
        if candidate is None:
            raise RuntimeError(
                "TensorRT profiler callback cannot be mapped by exact name and "
                f"occurrence: {key[0]!r}#{key[1]}"
            )
        layers.append({**dict(candidate), "index": index})
    runtime = {
        "schema_version": TRT_LAYER_PROFILE_SCHEMA_VERSION,
        "engine_kind": catalog["engine_kind"],
        "precision": catalog["precision"],
        "plan_sha256": catalog["plan_sha256"],
        "expected_target_call_site_count": catalog.get(
            "expected_target_call_site_count", 0
        ),
        "layers": layers,
    }
    runtime["catalog_sha256"] = catalog_sha256(runtime)
    return runtime


class TrtLayerProfileCapture:
    """Own one IProfiler and enforce one enqueue/report/capture at a time."""

    def __init__(
        self,
        *,
        trt: Any,
        context: Any,
        engine_kind: str,
        catalog: Mapping[str, Any],
    ) -> None:
        if engine_kind not in {"initial", "steady"}:
            raise ValueError(f"invalid TensorRT VAE engine kind: {engine_kind}")
        if catalog.get("engine_kind") != engine_kind:
            raise ValueError("TensorRT profile catalog engine kind mismatch")
        if catalog_sha256(catalog) != catalog.get("catalog_sha256"):
            raise ValueError("TensorRT profile Inspector catalog digest mismatch")
        probe_trt_layer_profile_api(trt=trt, contexts={engine_kind: context})
        self.engine_kind = engine_kind
        self.context = context
        self._inspector_catalog = dict(catalog)
        self._runtime_catalog: dict[str, Any] | None = None
        self._stable_keys: tuple[tuple[str, int], ...] | None = None
        self._active = False
        self._enqueue_succeeded = False
        self._callback_error: BaseException | None = None
        self._names: list[str] = []
        self._times_ms: list[float] = []
        profiler_class = create_trt_profiler_class(trt)
        self.profiler = profiler_class(self._record_layer_time)
        try:
            context.profiler = self.profiler
            context.enqueue_emits_profile = False
        except (AttributeError, RuntimeError, TypeError) as exc:
            raise RuntimeError(
                f"could not attach TensorRT profiler to {engine_kind} context"
            ) from exc
        if context.profiler is None:
            raise RuntimeError("TensorRT context did not retain a profiler")
        if bool(context.enqueue_emits_profile):
            raise RuntimeError("TensorRT context did not disable enqueue profiling")

    @property
    def catalog(self) -> dict[str, Any] | None:
        return (
            dict(self._runtime_catalog) if self._runtime_catalog is not None else None
        )

    def begin_capture(self, chunk_index: int) -> None:
        if self._active:
            raise RuntimeError("nested TensorRT layer-profile capture is forbidden")
        if isinstance(chunk_index, bool) or not isinstance(chunk_index, int):
            raise ValueError("TensorRT profile chunk index must be an integer")
        if self.engine_kind == "initial" and chunk_index != 0:
            raise ValueError("initial TensorRT profiler only accepts chunk 0")
        if self.engine_kind == "steady" and chunk_index <= 0:
            raise ValueError("steady TensorRT profiler only accepts chunks after 0")
        self._active = True
        self._enqueue_succeeded = False
        self._callback_error = None
        self._names = []
        self._times_ms = []
        self._chunk_index = chunk_index

    def mark_enqueue_succeeded(self) -> None:
        if not self._active:
            raise RuntimeError("TensorRT enqueue completed outside an active capture")
        if self._enqueue_succeeded:
            raise RuntimeError("TensorRT enqueue was marked successful more than once")
        self._enqueue_succeeded = True

    def _record_layer_time(self, layer_name: str, ms: float) -> None:
        if not self._active:
            self._callback_error = RuntimeError(
                "TensorRT profiler callback arrived outside an active capture"
            )
            return
        if not math.isfinite(ms) or ms < 0:
            self._callback_error = RuntimeError(
                f"TensorRT profiler reported invalid time for {layer_name!r}: {ms}"
            )
            return
        self._names.append(layer_name)
        self._times_ms.append(ms)

    def report_and_finish(self, *, engine_event_ms: float) -> dict[str, Any]:
        if not self._active:
            raise RuntimeError("TensorRT profile report has no active capture")
        try:
            if not self._enqueue_succeeded:
                raise RuntimeError(
                    "TensorRT profile report must follow a successful enqueue"
                )
            if not math.isfinite(engine_event_ms) or engine_event_ms < 0:
                raise ValueError("TensorRT engine CUDA event time is invalid")
            result = self.context.report_to_profiler()
            if result is not True:
                raise RuntimeError(
                    f"TensorRT {self.engine_kind} report_to_profiler returned "
                    f"{result!r}"
                )
            if self._callback_error is not None:
                raise RuntimeError(
                    "TensorRT profiler callback failed"
                ) from self._callback_error
            if not self._names:
                raise RuntimeError(
                    f"TensorRT {self.engine_kind} profiler reported zero layers"
                )
            occurrences: dict[str, int] = defaultdict(int)
            keys: list[tuple[str, int]] = []
            for name in self._names:
                occurrence = occurrences[name]
                occurrences[name] += 1
                keys.append((name, occurrence))
            stable_keys = tuple(keys)
            if self._stable_keys is None:
                self._runtime_catalog = _runtime_catalog_from_callbacks(
                    self._inspector_catalog, stable_keys
                )
                self._stable_keys = stable_keys
            elif stable_keys != self._stable_keys:
                raise RuntimeError(
                    f"TensorRT {self.engine_kind} profiler catalog drifted "
                    "between chunks"
                )
            assert self._runtime_catalog is not None
            return make_compact_layer_profile_metrics(
                catalog=self._runtime_catalog,
                layer_times_ms=self._times_ms,
                engine_event_ms=engine_event_ms,
            )
        finally:
            self._active = False
            self._enqueue_succeeded = False
            self._callback_error = None
            self._names = []
            self._times_ms = []

    def abort_capture(self) -> None:
        self._active = False
        self._enqueue_succeeded = False
        self._callback_error = None
        self._names = []
        self._times_ms = []


def make_compact_layer_profile_metrics(
    *,
    catalog: Mapping[str, Any],
    layer_times_ms: Sequence[float],
    engine_event_ms: float,
) -> dict[str, Any]:
    layers = catalog.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("TensorRT runtime profile catalog has no layers")
    times = [float(value) for value in layer_times_ms]
    if len(times) != len(layers):
        raise ValueError(
            f"TensorRT layer time count {len(times)} differs from catalog "
            f"length {len(layers)}"
        )
    if any(not math.isfinite(value) or value < 0 for value in times):
        raise ValueError("TensorRT layer times must be finite and non-negative")
    if not math.isfinite(engine_event_ms) or engine_event_ms < 0:
        raise ValueError("TensorRT engine event time must be finite and non-negative")
    digest = catalog.get("catalog_sha256")
    if digest != catalog_sha256(catalog):
        raise ValueError("TensorRT runtime profile catalog digest mismatch")
    category_totals = {category: 0.0 for category in PROFILE_CATEGORIES}
    for layer, elapsed in zip(layers, times, strict=True):
        category = layer.get("category") if isinstance(layer, Mapping) else None
        if category not in category_totals:
            raise ValueError(f"invalid TensorRT layer profile category: {category!r}")
        category_totals[category] += elapsed
    # Derive the total from the category buckets so their sum is exactly the
    # recorded layer sum rather than merely close after a different add order.
    layer_sum = float(sum(category_totals.values()))
    ratio = layer_sum / engine_event_ms if engine_event_ms > 0 else None
    return {
        "schema_version": TRT_LAYER_PROFILE_SCHEMA_VERSION,
        "engine_kind": catalog.get("engine_kind"),
        "catalog_sha256": digest,
        "layer_times_ms": times,
        "layer_sum_ms": layer_sum,
        "engine_event_ms": float(engine_event_ms),
        "layer_sum_over_engine_ratio": ratio,
        "category_totals_ms": category_totals,
        "reported_layer_count": len(times),
    }


def validate_compact_layer_profile_metrics(
    value: Mapping[str, Any], *, catalog: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("TensorRT compact layer profile must be an object")
    if value.get("schema_version") not in (SUPPORTED_TRT_LAYER_PROFILE_SCHEMA_VERSIONS):
        raise ValueError("TensorRT compact layer profile schema is invalid")
    rebuilt = make_compact_layer_profile_metrics(
        catalog=catalog,
        layer_times_ms=value.get("layer_times_ms", []),
        engine_event_ms=float(value.get("engine_event_ms", math.nan)),
    )
    for key in (
        "engine_kind",
        "catalog_sha256",
        "reported_layer_count",
    ):
        if value.get(key) != rebuilt[key]:
            raise ValueError(f"TensorRT compact layer profile {key} mismatch")
    for key in ("layer_sum_ms", "layer_sum_over_engine_ratio"):
        left = value.get(key)
        right = rebuilt[key]
        if left is None or right is None:
            if left is not right:
                raise ValueError(f"TensorRT compact layer profile {key} mismatch")
        elif not math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError(f"TensorRT compact layer profile {key} mismatch")
    category = value.get("category_totals_ms")
    category_keys = set(category) if isinstance(category, Mapping) else set()
    if not isinstance(category, Mapping) or category_keys not in (
        set(PROFILE_CATEGORIES),
        set(LEGACY_PROFILE_CATEGORIES),
    ):
        raise ValueError("TensorRT compact profile categories are incomplete")
    for name in PROFILE_CATEGORIES:
        legacy_value = category.get(name, 0.0)
        if not math.isclose(
            float(legacy_value),
            rebuilt["category_totals_ms"][name],
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError(f"TensorRT category total mismatch: {name}")
    return rebuilt


def _stats(values: Sequence[float]) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    if not numeric:
        return {
            "mean_ms": None,
            "population_stddev_ms": None,
            "min_ms": None,
            "max_ms": None,
            "sample_count": 0,
        }
    return {
        "mean_ms": statistics.fmean(numeric),
        "population_stddev_ms": statistics.pstdev(numeric),
        "min_ms": min(numeric),
        "max_ms": max(numeric),
        "sample_count": len(numeric),
    }


def _aggregate_capture_group(
    captures: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    catalogs_by_digest: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    layer_sums = [float(value["layer_sum_ms"]) for _, value in captures]
    event_times = [float(value["engine_event_ms"]) for _, value in captures]
    ratios = [
        float(value["layer_sum_over_engine_ratio"])
        for _, value in captures
        if value.get("layer_sum_over_engine_ratio") is not None
    ]
    category_values = {
        category: [
            float(value["category_totals_ms"][category]) for _, value in captures
        ]
        for category in PROFILE_CATEGORIES
    }
    result = {
        "layer_sum": _stats(layer_sums),
        "engine_event": _stats(event_times),
        "layer_sum_over_engine_ratio": _stats(ratios),
        "categories": {
            category: _stats(values) for category, values in category_values.items()
        },
        "sample_count": len(captures),
    }
    per_layer: dict[str, list[float]] = defaultdict(list)
    layer_meta: dict[str, Mapping[str, Any]] = {}
    for _, capture in captures:
        digest = str(capture["catalog_sha256"])
        catalog = catalogs_by_digest[digest]
        for layer, elapsed in zip(
            catalog["layers"], capture["layer_times_ms"], strict=True
        ):
            key = f"{catalog['engine_kind']}:{layer['exact_key']}"
            per_layer[key].append(float(elapsed))
            layer_meta[key] = layer
    result["per_layer"] = [
        {
            "key": key,
            "engine_kind": key.split(":", 1)[0],
            "layer": dict(layer_meta[key]),
            **_stats(per_layer[key]),
        }
        for key in sorted(per_layer)
    ]
    return result


def _compact_group(value: Mapping[str, Any]) -> dict[str, Any]:
    """Drop physical-layer detail from the normal client summary."""

    return {key: item for key, item in value.items() if key != "per_layer"}


def aggregate_trt_layer_profile_iterations(
    iterations: Sequence[Mapping[str, Any]],
    *,
    catalogs: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any] | None = None,
    precision: str,
    plan_sha256: Mapping[str, str],
    qdq_schema_version: int | None = None,
    weight_encoding: str | None = None,
) -> dict[str, Any]:
    """Build detailed and compact artifacts using measured iterations only."""

    if precision not in {"fp16", "int8"}:
        raise ValueError(f"invalid TensorRT layer-profile precision: {precision}")
    catalog_values = (
        list(catalogs.values()) if isinstance(catalogs, Mapping) else list(catalogs)
    )
    catalogs_by_digest: dict[str, Mapping[str, Any]] = {}
    catalog_by_kind: dict[str, Mapping[str, Any]] = {}
    for catalog in catalog_values:
        digest = catalog.get("catalog_sha256")
        if digest != catalog_sha256(catalog):
            raise ValueError("TensorRT aggregation catalog digest mismatch")
        kind = catalog.get("engine_kind")
        if kind not in {"initial", "steady"}:
            raise ValueError("TensorRT aggregation catalog engine kind is invalid")
        if catalog.get("precision") != precision:
            raise ValueError("TensorRT aggregation catalog precision mismatch")
        if catalog.get("plan_sha256") != plan_sha256.get(kind):
            raise ValueError("TensorRT aggregation catalog plan digest mismatch")
        if kind in catalog_by_kind and catalog_by_kind[kind] != catalog:
            raise ValueError(f"multiple TensorRT {kind} catalogs were supplied")
        catalogs_by_digest[str(digest)] = catalog
        catalog_by_kind[str(kind)] = catalog
    if set(catalog_by_kind) != {"initial", "steady"}:
        raise ValueError("both initial and steady TensorRT catalogs are required")

    raw_iterations: list[dict[str, Any]] = []
    measured_captures: list[tuple[int, int, dict[str, Any]]] = []
    expected_chunk_indices: list[int] | None = None
    warnings: list[str] = []
    for item in iterations:
        iteration_index = item.get("iteration")
        warmup = item.get("warmup")
        if isinstance(iteration_index, bool) or not isinstance(iteration_index, int):
            raise ValueError("profile iteration index must be an integer")
        if not isinstance(warmup, bool):
            raise ValueError("profile warmup flag must be boolean")
        if item.get("state") != "completed" or item.get("error") is not None:
            raise ValueError(f"profile iteration {iteration_index} did not complete")
        execution = item.get("profile_execution")
        chunks = execution.get("chunks") if isinstance(execution, Mapping) else None
        if not isinstance(chunks, list) or not chunks:
            raise ValueError(f"profile iteration {iteration_index} has no chunks")
        chunk_indices: list[int] = []
        raw_chunks: list[dict[str, Any]] = []
        for fallback_index, chunk in enumerate(chunks):
            if not isinstance(chunk, Mapping):
                raise ValueError("profile execution chunk must be an object")
            chunk_index = chunk.get("chunk_index", fallback_index)
            if isinstance(chunk_index, bool) or not isinstance(chunk_index, int):
                raise ValueError("profile chunk index must be an integer")
            layer_profile = chunk.get("trt_layer_profile")
            if not isinstance(layer_profile, Mapping):
                raise ValueError(
                    f"profile iteration {iteration_index} chunk {chunk_index} "
                    "has no TensorRT layer profile"
                )
            kind = "initial" if chunk_index == 0 else "steady"
            digest = layer_profile.get("catalog_sha256")
            catalog = catalogs_by_digest.get(str(digest))
            if catalog is None or catalog.get("engine_kind") != kind:
                raise ValueError("TensorRT profile capture/catalog mismatch")
            validated = validate_compact_layer_profile_metrics(
                layer_profile, catalog=catalog
            )
            chunk_indices.append(chunk_index)
            raw_chunks.append({"chunk_index": chunk_index, **validated})
            if not warmup:
                measured_captures.append((iteration_index, chunk_index, validated))
        if chunk_indices != list(range(len(chunk_indices))):
            raise ValueError("TensorRT profile chunks must be contiguous from zero")
        if expected_chunk_indices is None:
            expected_chunk_indices = chunk_indices
        elif chunk_indices != expected_chunk_indices:
            raise ValueError(
                "TensorRT profile chunk topology drifted across iterations"
            )
        raw_iterations.append(
            {
                "iteration": iteration_index,
                "warmup": warmup,
                "request_id": item.get("request_id"),
                "chunks": raw_chunks,
            }
        )

    if not measured_captures:
        raise ValueError("TensorRT layer profile has no measured iterations")
    by_chunk: dict[int, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    by_iteration: dict[int, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for iteration, chunk_index, capture in measured_captures:
        by_chunk[chunk_index].append((iteration, capture))
        by_iteration[iteration].append((chunk_index, capture))

    per_chunk = {
        str(index): _aggregate_capture_group(
            values, catalogs_by_digest=catalogs_by_digest
        )
        for index, values in sorted(by_chunk.items())
    }
    initial = _aggregate_capture_group(
        by_chunk[0], catalogs_by_digest=catalogs_by_digest
    )
    steady_values = [
        value
        for index, values in sorted(by_chunk.items())
        if index > 0
        for value in values
    ]
    steady_pooled = _aggregate_capture_group(
        steady_values, catalogs_by_digest=catalogs_by_digest
    )

    whole_captures: list[tuple[int, dict[str, Any]]] = []
    for iteration, captures in sorted(by_iteration.items()):
        categories = {category: 0.0 for category in PROFILE_CATEGORIES}
        layer_sum = 0.0
        engine_event = 0.0
        for _, capture in captures:
            layer_sum += float(capture["layer_sum_ms"])
            engine_event += float(capture["engine_event_ms"])
            for category in PROFILE_CATEGORIES:
                categories[category] += float(capture["category_totals_ms"][category])
        whole_captures.append(
            (
                iteration,
                {
                    "layer_sum_ms": layer_sum,
                    "engine_event_ms": engine_event,
                    "layer_sum_over_engine_ratio": (
                        layer_sum / engine_event if engine_event > 0 else None
                    ),
                    "category_totals_ms": categories,
                    # Whole-request per-layer aggregation is built from its
                    # constituent captures below rather than this empty vector.
                    "catalog_sha256": "",
                    "layer_times_ms": [],
                },
            )
        )
    whole = {
        "layer_sum": _stats([value["layer_sum_ms"] for _, value in whole_captures]),
        "engine_event": _stats(
            [value["engine_event_ms"] for _, value in whole_captures]
        ),
        "layer_sum_over_engine_ratio": _stats(
            [
                value["layer_sum_over_engine_ratio"]
                for _, value in whole_captures
                if value["layer_sum_over_engine_ratio"] is not None
            ]
        ),
        "categories": {
            category: _stats(
                [value["category_totals_ms"][category] for _, value in whole_captures]
            )
            for category in PROFILE_CATEGORIES
        },
        "sample_count": len(whole_captures),
    }
    whole_mean = float(whole["layer_sum"]["mean_ms"] or 0.0)
    category_percentages = {
        category: (
            float(whole["categories"][category]["mean_ms"] or 0.0) / whole_mean * 100.0
            if whole_mean > 0
            else 0.0
        )
        for category in PROFILE_CATEGORIES
    }

    ratios = [
        float(capture["layer_sum_over_engine_ratio"])
        for _, _, capture in measured_captures
        if capture.get("layer_sum_over_engine_ratio") is not None
    ]
    coverage_complete = bool(ratios) and all(
        MIN_LAYER_SUM_OVER_ENGINE_RATIO <= ratio <= MAX_LAYER_SUM_OVER_ENGINE_RATIO
        for ratio in ratios
    )
    if not coverage_complete:
        warnings.append(
            "TensorRT layer callback sums differ materially from outer CUDA-event times"
        )
    other_fraction = category_percentages["other"] / 100.0
    classification_complete = other_fraction <= MAX_OTHER_CATEGORY_FRACTION
    if not classification_complete:
        warnings.append(
            "unclassified TensorRT layers exceed 10% of measured layer time"
        )

    target_mapping_complete = True
    for kind, catalog in catalog_by_kind.items():
        expected_target_count = int(catalog.get("expected_target_call_site_count", 0))
        mapped = {
            site
            for layer in catalog["layers"]
            for site in layer.get("logical_call_sites", [])
        }
        if (
            expected_target_count != EXPECTED_CALL_SITES
            or len(mapped) != EXPECTED_CALL_SITES
        ):
            target_mapping_complete = False
            warnings.append(
                f"{precision.upper()} {kind} catalog maps {len(mapped)} target "
                f"call sites; expected {EXPECTED_CALL_SITES}"
            )

    validation = {
        "valid_for_optimization_decision": (
            coverage_complete and classification_complete and target_mapping_complete
        ),
        "catalog_stable": True,
        "target_mapping_complete": target_mapping_complete,
        "layer_profile_complete": coverage_complete,
        "classification_complete": classification_complete,
        "warnings": warnings,
    }
    detailed = {
        "schema_version": TRT_LAYER_PROFILE_SCHEMA_VERSION,
        "environment": dict(environment or {}),
        "precision": precision,
        "plan_sha256": dict(plan_sha256),
        "qdq_schema_version": qdq_schema_version,
        "weight_encoding": weight_encoding,
        "catalogs": {
            kind: dict(catalog) for kind, catalog in sorted(catalog_by_kind.items())
        },
        "iterations": raw_iterations,
        "statistics": {
            "per_chunk": per_chunk,
            "initial": initial,
            "steady_pooled": steady_pooled,
            "whole_request": whole,
            "category_percentages": category_percentages,
        },
        "validation": validation,
    }
    summary = {
        "schema_version": TRT_LAYER_PROFILE_SCHEMA_VERSION,
        "per_chunk": {
            index: _compact_group(value) for index, value in per_chunk.items()
        },
        "initial": _compact_group(initial),
        "steady_pooled": _compact_group(steady_pooled),
        "whole_request": whole,
        "category_percentages": category_percentages,
        "target_conv_percentage": category_percentages["target_quantized_conv"],
        "qdq_cast_reformat_percentage": category_percentages[
            "target_qdq_cast_reformat"
        ],
        "cache_percentage": category_percentages["cache_layout_copy"],
        "fused_input_pack_quant_percentage": category_percentages[
            "fused_input_pack_quant"
        ],
        "fused_cache_update_percentage": category_percentages["fused_cache_update"],
        "fused_conv1_norm_silu_percentage": category_percentages[
            "fused_conv1_norm_silu"
        ],
        "fused_conv2_residual_percentage": category_percentages["fused_conv2_residual"],
        "other_percentage": category_percentages["other"],
        **validation,
    }
    return {"detailed": detailed, "summary": summary}


def _artifact_bytes(detailed: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            detailed,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def detailed_artifact_reference(
    *, path: str | Path, detailed: Mapping[str, Any]
) -> dict[str, str]:
    """Return the stable reference the client stores in its compact summary."""

    resolved = Path(path).expanduser().resolve()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(_artifact_bytes(detailed)).hexdigest(),
    }


def write_trt_layer_profile_artifact(
    *, path: str | Path, detailed: Mapping[str, Any]
) -> dict[str, str]:
    """Atomically persist detailed evidence and return its actual file digest."""

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    payload = _artifact_bytes(detailed)
    temporary.write_bytes(payload)
    temporary.replace(resolved)
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


__all__ = [
    "MAX_LAYER_SUM_OVER_ENGINE_RATIO",
    "MAX_OTHER_CATEGORY_FRACTION",
    "MIN_LAYER_SUM_OVER_ENGINE_RATIO",
    "PROFILE_CATEGORIES",
    "TRT_LAYER_PROFILE_MANIFEST_FILE",
    "TRT_LAYER_PROFILE_SCHEMA_VERSION",
    "TRT_LAYER_PROFILE_SCOPE",
    "TrtLayerProfileCapture",
    "aggregate_trt_layer_profile_iterations",
    "build_physical_layer_catalog",
    "catalog_sha256",
    "create_trt_profiler_class",
    "detailed_artifact_reference",
    "load_trt_layer_profile_manifest",
    "make_compact_layer_profile_metrics",
    "probe_trt_layer_profile_api",
    "sha256_file",
    "sha256_json",
    "validate_compact_layer_profile_metrics",
    "validate_trt_layer_profile_manifest",
    "write_trt_layer_profile_artifact",
]
