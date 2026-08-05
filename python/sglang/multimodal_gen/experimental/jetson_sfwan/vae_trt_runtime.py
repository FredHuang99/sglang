"""TensorRT runtime for the fixed-shape Jetson SFWan VAE engines."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

from .vae_trt_qdq import (
    EXPECTED_CALL_SITES,
    EXPECTED_CONV_SIGNATURES,
    QDQ_SCHEMA_VERSION,
    QDQ_TOPOLOGY,
    WEIGHT_ENCODINGS,
    WEIGHT_ENCODING_FP32_QDQ,
)

TRT_VAE_MANIFEST_SCHEMA_VERSION = 1
TRT_VAE_HEIGHT = 480
TRT_VAE_WIDTH = 832
TRT_VAE_LATENT_SHAPE = (1, 16, 3, 60, 104)
TRT_VAE_CACHE_COUNT = 32
TRT_VAE_CACHE_TOTAL_ELEMENTS = 944_286_720
TRT_VAE_CACHE_BANK_BYTES = TRT_VAE_CACHE_TOTAL_ELEMENTS * 2

TrtVaePrecision = Literal["fp16", "int8"]
TrtVaeEngineKind = Literal["initial", "steady"]
TrtVaeVariant = Literal[
    "baseline",
    "fusion_v1",
    "fusion_v2",
    "native_int8_v1",
    "native_int8_v2",
]
NativeInt8V2Level = Literal["p1", "p2", "p3"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version_prefix(version: str, fields: int = 2) -> tuple[int, ...]:
    values: list[int] = []
    for part in version.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        values.append(int(digits))
        if len(values) == fields:
            break
    return tuple(values)


def _create_trt_execution_stream(*, torch: Any, device: Any) -> Any:
    """Create the persistent non-default stream used by TensorRT execution."""

    stream = torch.cuda.Stream(device=device)
    default_stream = torch.cuda.default_stream(device=device)
    if int(stream.cuda_stream) == int(default_stream.cuda_stream):
        raise RuntimeError("TensorRT VAE did not receive a non-default CUDA stream")
    return stream


def _load_fusion_plugin_library(
    *,
    trt: Any,
    plugin_path: str | Path,
    creator_names: tuple[str, ...],
    plugin_version: str,
    plugin_namespace: str,
    init_symbol: str = "initSfWanVaeTrtFusionPlugins",
) -> Any:
    """Load and verify fusion creators before deserializing a plugin plan."""

    import ctypes

    path = Path(plugin_path)
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    try:
        library = ctypes.CDLL(str(path), mode=mode)
    except OSError as exc:
        raise RuntimeError(
            f"could not load TensorRT VAE fusion plugin: {path}"
        ) from exc
    try:
        initialize = getattr(library, init_symbol)
    except AttributeError as exc:
        raise RuntimeError(
            f"TensorRT VAE fusion plugin has no {init_symbol}"
        ) from exc
    initialize.argtypes = []
    initialize.restype = ctypes.c_bool
    if not initialize():
        raise RuntimeError("TensorRT VAE fusion plugin registration failed")
    registry = trt.get_plugin_registry()
    if registry is None:
        raise RuntimeError("TensorRT plugin registry is unavailable")
    missing: list[str] = []
    for name in creator_names:
        creator = None
        getter = getattr(registry, "get_plugin_creator", None)
        if callable(getter):
            creator = getter(name, plugin_version, plugin_namespace)
        if creator is None:
            getter = getattr(registry, "get_creator", None)
            if callable(getter):
                creator = getter(name, plugin_version, plugin_namespace)
        if creator is None:
            missing.append(name)
    if missing:
        raise RuntimeError(
            f"TensorRT VAE fusion plugin creators were not registered: {missing}"
        )
    return library


def _configure_context_nvtx(
    *,
    context: Any,
    trt: Any,
    precision: TrtVaePrecision,
    enable_nvtx: bool,
    detailed_plan: bool = False,
) -> str:
    if precision != "int8" and not detailed_plan:
        effective = getattr(context, "nvtx_verbosity", None)
        return (
            str(effective).split(".")[-1].lower()
            if effective is not None
            else "unavailable"
        )
    requested = (
        trt.ProfilingVerbosity.DETAILED if enable_nvtx else trt.ProfilingVerbosity.NONE
    )
    try:
        context.nvtx_verbosity = requested
        effective = context.nvtx_verbosity
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise RuntimeError(
            "TensorRT VAE detailed plan could not set execution-context NVTX verbosity"
        ) from exc
    if effective != requested:
        raise RuntimeError(
            f"TensorRT context NVTX verbosity is {effective}, expected {requested}"
        )
    return str(effective).split(".")[-1].lower()


def load_trt_vae_manifest(engine_dir: str | Path) -> dict[str, Any]:
    path = Path(engine_dir).expanduser().resolve() / "manifest.json"
    if not path.is_file():
        raise ValueError(f"TensorRT VAE manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"TensorRT VAE manifest is invalid: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("TensorRT VAE manifest must be a JSON object")
    return manifest


def validate_trt_vae_manifest(
    manifest: dict[str, Any],
    *,
    engine_dir: str | Path,
    precision: TrtVaePrecision,
    model_path: str | None,
    verify_plan_hashes: bool,
) -> dict[str, Any]:
    """Validate the static contract without importing Torch or TensorRT."""

    if manifest.get("schema_version") != TRT_VAE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported TensorRT VAE manifest schema: "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("batch_size") != 1:
        raise ValueError("TensorRT VAE manifest must use batch size one")
    if (manifest.get("height"), manifest.get("width")) != (
        TRT_VAE_HEIGHT,
        TRT_VAE_WIDTH,
    ):
        raise ValueError("TensorRT VAE manifest must target 480x832")
    if tuple(manifest.get("latent_shape", ())) != TRT_VAE_LATENT_SHAPE:
        raise ValueError(
            "TensorRT VAE latent shape must be "
            f"{TRT_VAE_LATENT_SHAPE}, got {manifest.get('latent_shape')}"
        )
    if manifest.get("latent_dtype") != "float16":
        raise ValueError("TensorRT VAE engine ingress must be float16")
    if model_path is not None and manifest.get("model_id") != model_path:
        raise ValueError(
            "TensorRT VAE manifest model_id does not match --model-path: "
            f"{manifest.get('model_id')!r} != {model_path!r}"
        )

    cache = manifest.get("cache")
    if not isinstance(cache, dict):
        raise ValueError("TensorRT VAE manifest has no cache contract")
    if cache.get("allocated_slot_count") != 33:
        raise ValueError("TensorRT VAE manifest must record 33 allocated cache slots")
    active_slot_indices = cache.get("active_slot_indices")
    if (
        not isinstance(active_slot_indices, list)
        or len(active_slot_indices) != TRT_VAE_CACHE_COUNT
        or len(set(active_slot_indices)) != TRT_VAE_CACHE_COUNT
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= 33
            for index in active_slot_indices
        )
    ):
        raise ValueError(
            "TensorRT VAE manifest must record 32 unique active source slots"
        )
    bindings = cache.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != TRT_VAE_CACHE_COUNT:
        raise ValueError(
            f"TensorRT VAE manifest must contain {TRT_VAE_CACHE_COUNT} cache bindings"
        )
    indices = [
        binding.get("index") for binding in bindings if isinstance(binding, dict)
    ]
    if indices != list(range(TRT_VAE_CACHE_COUNT)):
        raise ValueError("TensorRT VAE cache binding indices must be contiguous 0..31")
    shapes = []
    for binding in bindings:
        shape = binding.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 5
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in shape
            )
        ):
            raise ValueError(f"invalid TensorRT VAE cache shape: {shape!r}")
        if binding.get("dtype") != "float16":
            raise ValueError("TensorRT VAE feature cache must use float16")
        shapes.append(tuple(shape))
    total_elements = sum(math.prod(shape) for shape in shapes)
    if total_elements != TRT_VAE_CACHE_TOTAL_ELEMENTS:
        raise ValueError(
            f"feature cache has {total_elements} elements; "
            f"expected {TRT_VAE_CACHE_TOTAL_ELEMENTS}"
        )
    if cache.get("total_elements") != TRT_VAE_CACHE_TOTAL_ELEMENTS:
        raise ValueError("TensorRT VAE cache total_elements is invalid")
    if cache.get("single_bank_bytes") != TRT_VAE_CACHE_BANK_BYTES:
        raise ValueError("TensorRT VAE single cache-bank byte count is invalid")
    if cache.get("double_bank_bytes") != TRT_VAE_CACHE_BANK_BYTES * 2:
        raise ValueError("TensorRT VAE double cache-bank byte count is invalid")

    engines = manifest.get("engines")
    if not isinstance(engines, dict) or precision not in engines:
        raise ValueError(f"TensorRT VAE manifest has no {precision} engines")
    precision_engines = engines[precision]
    if not isinstance(precision_engines, dict):
        raise ValueError(f"TensorRT VAE {precision} engine contract is invalid")
    root = Path(engine_dir).expanduser().resolve()
    validated_engines: dict[str, Any] = {}
    for kind, expected_frames in (("initial", 9), ("steady", 12)):
        record = precision_engines.get(kind)
        if not isinstance(record, dict):
            raise ValueError(f"TensorRT VAE manifest has no {precision}/{kind} engine")
        if record.get("rgb_shape") != [1, 3, expected_frames, 480, 832]:
            raise ValueError(
                f"TensorRT VAE {precision}/{kind} RGB shape is invalid: "
                f"{record.get('rgb_shape')}"
            )
        relative_file = record.get("file")
        if not isinstance(relative_file, str) or not relative_file:
            raise ValueError(f"TensorRT VAE {precision}/{kind} file is invalid")
        plan_path = (root / relative_file).resolve()
        try:
            plan_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "TensorRT VAE plan path escapes the engine directory"
            ) from exc
        if not plan_path.is_file():
            raise ValueError(f"TensorRT VAE plan does not exist: {plan_path}")
        expected_digest = record.get("sha256")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise ValueError(f"TensorRT VAE {precision}/{kind} SHA256 is invalid")
        if verify_plan_hashes and _sha256_file(plan_path) != expected_digest:
            raise ValueError(f"TensorRT VAE plan digest does not match: {plan_path}")
        validated_engines[kind] = {**record, "path": str(plan_path)}

    if precision == "int8":
        quantization = manifest.get("quantization")
        if (
            not isinstance(quantization, dict)
            or quantization.get("qdq_schema_version") != QDQ_SCHEMA_VERSION
            or quantization.get("topology") != QDQ_TOPOLOGY
        ):
            raise ValueError(
                f"INT8 TensorRT VAE requires Q/DQ schema {QDQ_SCHEMA_VERSION}"
            )
        weight_encoding = quantization.get("weight_encoding")
        if weight_encoding not in WEIGHT_ENCODINGS:
            raise ValueError("INT8 TensorRT VAE weight encoding is invalid")
        audit = manifest.get("int8_audit")
        if not isinstance(audit, dict) or audit.get("passed") is not True:
            raise ValueError(
                "INT8 TensorRT VAE requires a passing structural and tactic audit"
            )
        if audit.get("schema_version") != QDQ_SCHEMA_VERSION:
            raise ValueError("INT8 TensorRT VAE manifest audit schema is invalid")
        if audit.get("weight_encoding") != weight_encoding:
            raise ValueError(
                "INT8 TensorRT VAE manifest weight encoding is inconsistent"
            )
        if audit.get("preflight_passed") is not True:
            raise ValueError("INT8 TensorRT VAE signature preflight did not pass")
        if audit.get("target_conv_call_sites_per_graph") != EXPECTED_CALL_SITES:
            raise ValueError("INT8 TensorRT VAE audit call-site count is invalid")
        report_file = audit.get("report_file")
        if not isinstance(report_file, str) or not report_file:
            raise ValueError("INT8 TensorRT VAE manifest has no audit report file")
        report_path = (root / report_file).resolve()
        try:
            report_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "TensorRT VAE audit path escapes the engine directory"
            ) from exc
        if not report_path.is_file():
            raise ValueError(
                f"TensorRT VAE INT8 audit report does not exist: {report_path}"
            )
        expected_report_digest = audit.get("report_sha256")
        if (
            not isinstance(expected_report_digest, str)
            or len(expected_report_digest) != 64
        ):
            raise ValueError("TensorRT VAE INT8 audit SHA256 is invalid")
        if verify_plan_hashes and _sha256_file(report_path) != expected_report_digest:
            raise ValueError(
                f"TensorRT VAE INT8 audit digest does not match: {report_path}"
            )
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"TensorRT VAE INT8 audit report is invalid: {report_path}"
            ) from exc
        if not isinstance(report, dict) or report.get("passed") is not True:
            raise ValueError("TensorRT VAE INT8 audit report did not pass")
        if (
            report.get("schema_version") != QDQ_SCHEMA_VERSION
            or report.get("weight_encoding") != weight_encoding
            or report.get("preflight_passed") is not True
            or report.get("complete") is not True
            or report.get("errors") != []
        ):
            raise ValueError("TensorRT VAE INT8 audit report is incomplete")
        if report.get("target_conv_call_sites_per_graph") != EXPECTED_CALL_SITES:
            raise ValueError(
                "TensorRT VAE INT8 audit report call-site count is invalid"
            )
        probe_suite = report.get("probe_suite")
        if not isinstance(probe_suite, dict) or probe_suite.get("passed") is not True:
            raise ValueError("TensorRT VAE INT8 signature probe suite did not pass")
        if (
            probe_suite.get("schema_version") != QDQ_SCHEMA_VERSION
            or probe_suite.get("selected_weight_encoding") != weight_encoding
            or probe_suite.get("weight_encoding") != weight_encoding
            or probe_suite.get("errors") != []
        ):
            raise ValueError("TensorRT VAE INT8 probe schema is invalid")
        source_counts = probe_suite.get("source_call_site_counts")
        if source_counts != {
            "initial": EXPECTED_CALL_SITES,
            "steady": EXPECTED_CALL_SITES,
        }:
            raise ValueError("TensorRT VAE INT8 probe coverage is incomplete")
        signatures = probe_suite.get("signatures")
        signature_count = probe_suite.get("signature_count")
        probed_signature_count = probe_suite.get("probed_signature_count")
        probe_empty_lists = (
            "unmapped_call_sites",
            "non_int8_call_sites",
            "activation_not_int8_call_sites",
            "output_not_int8_call_sites",
            "static_weight_not_int8_call_sites",
            "dynamic_filter_call_sites",
            "fp16_fallback_call_sites",
            "fp32_or_tf32_fallback_call_sites",
        )
        if (
            isinstance(signature_count, bool)
            or not isinstance(signature_count, int)
            or signature_count != EXPECTED_CONV_SIGNATURES
            or probed_signature_count != signature_count
            or not isinstance(signatures, dict)
            or len(signatures) != signature_count
        ):
            raise ValueError("TensorRT VAE INT8 signature probe records are invalid")
        probed_source_call_sites = {"initial": [], "steady": []}
        for signature_id, probe in signatures.items():
            if (
                not isinstance(probe, dict)
                or probe.get("signature_id") != signature_id
                or probe.get("passed") is not True
                or probe.get("mapped_count") != 1
                or probe.get("errors") != []
                or probe.get("weight_encoding") != weight_encoding
                or any(probe.get(key) != [] for key in probe_empty_lists)
            ):
                raise ValueError(
                    f"TensorRT VAE INT8 signature probe is invalid: {signature_id}"
                )
            probe_matches = probe.get("matches")
            if not isinstance(probe_matches, dict) or len(probe_matches) != 1:
                raise ValueError(
                    f"TensorRT VAE INT8 signature probe evidence is invalid: "
                    f"{signature_id}"
                )
            probe_evidence = next(iter(probe_matches.values()))
            if (
                not isinstance(probe_evidence, dict)
                or probe_evidence.get("passed") is not True
                or probe_evidence.get("activation_int8") is not True
                or probe_evidence.get("output_int8") is not True
                or probe_evidence.get("static_weight_int8") is not True
                or probe_evidence.get("dynamic_filter") is True
                or probe_evidence.get("int8_tactic") is not True
                or probe_evidence.get("fp16_fallback") is not False
                or probe_evidence.get("fp32_or_tf32_fallback") is not False
            ):
                raise ValueError(
                    f"TensorRT VAE INT8 signature probe tactic is invalid: "
                    f"{signature_id}"
                )
            source_call_sites = probe.get("source_call_sites")
            if not isinstance(source_call_sites, dict):
                raise ValueError(
                    f"TensorRT VAE INT8 signature coverage is invalid: {signature_id}"
                )
            for kind in ("initial", "steady"):
                call_sites = source_call_sites.get(kind)
                if not isinstance(call_sites, list) or any(
                    not isinstance(call_site, str) or not call_site
                    for call_site in call_sites
                ):
                    raise ValueError(
                        "TensorRT VAE INT8 signature source call sites are invalid: "
                        f"{signature_id}/{kind}"
                    )
                probed_source_call_sites[kind].extend(call_sites)
        for kind in ("initial", "steady"):
            call_sites = probed_source_call_sites[kind]
            if (
                len(call_sites) != EXPECTED_CALL_SITES
                or len(set(call_sites)) != EXPECTED_CALL_SITES
            ):
                raise ValueError(
                    f"TensorRT VAE INT8 {kind} signature coverage is invalid"
                )

        structural = report.get("structural")
        if not isinstance(structural, dict):
            raise ValueError("TensorRT VAE INT8 structural audits are missing")
        expected_weight_quantize_count = (
            EXPECTED_CALL_SITES if weight_encoding == WEIGHT_ENCODING_FP32_QDQ else 0
        )
        for kind in ("initial", "steady"):
            graph_audit = structural.get(kind)
            if (
                not isinstance(graph_audit, dict)
                or graph_audit.get("schema_version") != QDQ_SCHEMA_VERSION
                or graph_audit.get("qdq_topology") != QDQ_TOPOLOGY
                or graph_audit.get("weight_encoding") != weight_encoding
                or graph_audit.get("passed") is not True
                or graph_audit.get("errors") != []
                or graph_audit.get("target_conv_call_site_count") != EXPECTED_CALL_SITES
                or graph_audit.get("activation_cast_count") != EXPECTED_CALL_SITES
                or graph_audit.get("activation_quantize_count") != EXPECTED_CALL_SITES
                or graph_audit.get("activation_dequantize_count") != EXPECTED_CALL_SITES
                or graph_audit.get("weight_quantize_count")
                != expected_weight_quantize_count
                or graph_audit.get("weight_dequantize_count") != EXPECTED_CALL_SITES
                or graph_audit.get("output_quantize_count") != EXPECTED_CALL_SITES
                or graph_audit.get("output_dequantize_count") != EXPECTED_CALL_SITES
                or graph_audit.get("output_cast_count") != EXPECTED_CALL_SITES
                or graph_audit.get("unique_weight_source_count") != EXPECTED_CALL_SITES
                or graph_audit.get("unique_weight_quantize_output_count")
                != expected_weight_quantize_count
                or graph_audit.get("unique_weight_dequantize_output_count")
                != EXPECTED_CALL_SITES
                or graph_audit.get("unique_output_quantize_output_count")
                != EXPECTED_CALL_SITES
                or graph_audit.get("unique_output_dequantize_output_count")
                != EXPECTED_CALL_SITES
                or graph_audit.get("unique_bias_count") != EXPECTED_CALL_SITES
                or graph_audit.get("unexpected_target_cast_nodes") != []
            ):
                raise ValueError(
                    f"TensorRT VAE INT8 {kind} structural audit is invalid"
                )

        feature_cache = report.get("feature_cache")
        expected_cache_counts = {
            "initial": {"inputs": 0, "outputs": TRT_VAE_CACHE_COUNT},
            "steady": {
                "inputs": TRT_VAE_CACHE_COUNT,
                "outputs": TRT_VAE_CACHE_COUNT,
            },
        }
        if (
            not isinstance(feature_cache, dict)
            or feature_cache.get("passed") is not True
            or feature_cache.get("errors") != []
            or feature_cache.get("dtype") != "float16"
            or feature_cache.get("tensor_count") != TRT_VAE_CACHE_COUNT
            or feature_cache.get("binding_counts") != expected_cache_counts
            or feature_cache.get("single_bank_bytes") != TRT_VAE_CACHE_BANK_BYTES
            or feature_cache.get("double_bank_bytes") != TRT_VAE_CACHE_BANK_BYTES * 2
            or feature_cache.get("quantized_bindings") != []
            or feature_cache.get("physical_engine_io_checked") is not True
        ):
            raise ValueError("TensorRT VAE INT8 feature-cache audit is invalid")

        tactics = report.get("tactics")
        audit_plan_sha = audit.get("plan_sha256")
        report_plan_sha = report.get("plan_sha256")
        if not isinstance(tactics, dict):
            raise ValueError("TensorRT VAE INT8 tactic audits are missing")
        for kind in ("initial", "steady"):
            tactic = tactics.get(kind)
            plan_sha256 = validated_engines[kind]["sha256"]
            if not isinstance(tactic, dict) or tactic.get("passed") is not True:
                raise ValueError(f"TensorRT VAE INT8 {kind} tactic audit did not pass")
            if (
                tactic.get("mapped_count") != EXPECTED_CALL_SITES
                or tactic.get("errors") != []
            ):
                raise ValueError(
                    f"TensorRT VAE INT8 {kind} tactic coverage is incomplete"
                )
            for key in (
                "unmapped_call_sites",
                "non_int8_call_sites",
                "activation_not_int8_call_sites",
                "output_not_int8_call_sites",
                "static_weight_not_int8_call_sites",
                "dynamic_filter_call_sites",
                "fp16_fallback_call_sites",
                "fp32_or_tf32_fallback_call_sites",
            ):
                if tactic.get(key) != []:
                    raise ValueError(
                        f"TensorRT VAE INT8 {kind} audit has non-empty {key}"
                    )
            matches = tactic.get("matches")
            if not isinstance(matches, dict) or len(matches) != EXPECTED_CALL_SITES:
                raise ValueError(
                    f"TensorRT VAE INT8 {kind} tactic records are incomplete"
                )
            for call_site, evidence in matches.items():
                if (
                    not isinstance(call_site, str)
                    or not isinstance(evidence, dict)
                    or evidence.get("passed") is not True
                    or evidence.get("activation_int8") is not True
                    or evidence.get("output_int8") is not True
                    or evidence.get("static_weight_int8") is not True
                    or evidence.get("dynamic_filter") is True
                    or evidence.get("int8_tactic") is not True
                    or evidence.get("fp16_fallback") is not False
                    or evidence.get("fp32_or_tf32_fallback") is not False
                ):
                    raise ValueError(
                        f"TensorRT VAE INT8 {kind} tactic evidence is invalid: "
                        f"{call_site}"
                    )
            if tactic.get("build_profiling_verbosity") != "detailed":
                raise ValueError(
                    f"TensorRT VAE INT8 {kind} plan was not built as detailed"
                )
            if tactic.get("weight_encoding") != weight_encoding:
                raise ValueError(
                    f"TensorRT VAE INT8 {kind} weight encoding is inconsistent"
                )
            if validated_engines[kind].get("profiling_verbosity") != "detailed":
                raise ValueError(
                    f"TensorRT VAE INT8 {kind} manifest verbosity is invalid"
                )
            if validated_engines[kind].get("audit_passed") is not True:
                raise ValueError(
                    f"TensorRT VAE INT8 {kind} manifest is not audit-passed"
                )
            if (
                tactic.get("plan_sha256") != plan_sha256
                or not isinstance(audit_plan_sha, dict)
                or audit_plan_sha.get(kind) != plan_sha256
                or not isinstance(report_plan_sha, dict)
                or report_plan_sha.get(kind) != plan_sha256
            ):
                raise ValueError(f"TensorRT VAE INT8 {kind} plan/audit digest mismatch")

    return {
        "cache_shapes": shapes,
        "engines": validated_engines,
    }


class TensorRTVaeRuntime:
    """Execute one initial engine followed by steady engines for one request."""

    def __init__(
        self,
        *,
        engine_dir: str,
        precision: TrtVaePrecision,
        model_path: str,
        device: Any,
        enable_profile: bool,
        enable_nvtx: bool,
        enable_trt_layer_profile: bool = False,
        enable_native_int8_kernel_profile: bool = False,
        variant: TrtVaeVariant = "baseline",
        native_int8_v2_level: NativeInt8V2Level = "p1",
    ) -> None:
        try:
            import tensorrt as trt
            import torch
        except ImportError as exc:  # pragma: no cover - Jetson-only runtime
            raise RuntimeError(
                "TensorRT VAE execution requires both torch and tensorrt"
            ) from exc

        self.engine_dir = str(Path(engine_dir).expanduser().resolve())
        self.precision = precision
        self.device = device
        self.enable_profile = enable_profile
        self.enable_trt_layer_profile = enable_trt_layer_profile
        self.enable_native_int8_kernel_profile = (
            enable_native_int8_kernel_profile
        )
        self.enable_nvtx = enable_nvtx
        self.variant = variant
        self.native_int8_v2_level = native_int8_v2_level
        if variant not in {
            "baseline",
            "fusion_v1",
            "fusion_v2",
            "native_int8_v1",
            "native_int8_v2",
        }:
            raise ValueError(f"unsupported TensorRT VAE variant: {variant}")
        if variant in {
            "fusion_v1",
            "fusion_v2",
            "native_int8_v1",
            "native_int8_v2",
        } and precision != "int8":
            raise ValueError(f"{variant} requires TensorRT INT8 VAE precision")
        if enable_trt_layer_profile and not enable_profile:
            raise ValueError(
                "TensorRT layer profiling requires the regular profile timer"
            )
        if enable_native_int8_kernel_profile:
            if not enable_profile:
                raise ValueError(
                    "native INT8 kernel profiling requires the regular profile timer"
                )
            if variant not in {"native_int8_v1", "native_int8_v2"}:
                raise ValueError(
                    "native INT8 kernel profiling requires native_int8_v1 "
                    "or native_int8_v2"
                )
        if native_int8_v2_level not in {"p1", "p2", "p3"}:
            raise ValueError(
                f"unsupported Native INT8 V2 level: {native_int8_v2_level}"
            )
        self.manifest = load_trt_vae_manifest(self.engine_dir)
        validated = validate_trt_vae_manifest(
            self.manifest,
            engine_dir=self.engine_dir,
            precision=precision,
            model_path=model_path,
            verify_plan_hashes=True,
        )
        fusion_validated = None
        self._fusion_manifest: dict[str, Any] | None = None
        self._fusion_validated: dict[str, Any] | None = None
        self._fusion_plugin_library: Any | None = None
        native_validated = None
        self._native_manifest: dict[str, Any] | None = None
        self._native_validated: dict[str, Any] | None = None
        self._native_plugin_library: Any | None = None
        if variant == "fusion_v1":
            # Deliberately delayed: the baseline path never imports fusion
            # helpers, reads fusion artifacts, or loads a plugin library.
            from .vae_trt_fusion import (
                PLUGIN_CREATORS,
                PLUGIN_NAMESPACE,
                PLUGIN_VERSION,
                load_fusion_manifest,
                validate_fusion_manifest,
            )

            fusion_manifest = load_fusion_manifest(self.engine_dir)
            fusion_validated = validate_fusion_manifest(
                fusion_manifest,
                engine_dir=self.engine_dir,
                base_manifest=self.manifest,
                verify_hashes=True,
            )
            self._fusion_manifest = fusion_manifest
            self._fusion_validated = fusion_validated
            self._fusion_plugin_library = _load_fusion_plugin_library(
                trt=trt,
                plugin_path=fusion_validated["plugin_path"],
                creator_names=PLUGIN_CREATORS,
                plugin_version=PLUGIN_VERSION,
                plugin_namespace=PLUGIN_NAMESPACE,
            )
        elif variant == "fusion_v2":
            # Fusion-v2 is a different plugin ABI and artifact namespace.  It
            # deliberately reuses the raw v5 initial plan and selects only a
            # rewritten steady plan with six INT8 feature-cache slots.
            from .vae_trt_fusion_v2 import (
                PLUGIN_CREATORS,
                PLUGIN_INIT_SYMBOL,
                PLUGIN_NAMESPACE,
                PLUGIN_VERSION,
                load_fusion_v2_manifest,
                validate_fusion_v2_manifest,
            )

            fusion_manifest = load_fusion_v2_manifest(self.engine_dir)
            fusion_validated = validate_fusion_v2_manifest(
                fusion_manifest,
                engine_dir=self.engine_dir,
                base_manifest=self.manifest,
                verify_hashes=True,
            )
            self._fusion_manifest = fusion_manifest
            self._fusion_validated = fusion_validated
            self._fusion_plugin_library = _load_fusion_plugin_library(
                trt=trt,
                plugin_path=fusion_validated["plugin_path"],
                creator_names=PLUGIN_CREATORS,
                plugin_version=PLUGIN_VERSION,
                plugin_namespace=PLUGIN_NAMESPACE,
                init_symbol=PLUGIN_INIT_SYMBOL,
            )
            self._configure_fusion_v2_migration(
                library=self._fusion_plugin_library,
                ctypes_module=__import__("ctypes"),
            )
        elif variant == "native_int8_v1":
            # The native plugin and manifest live in a completely separate
            # namespace.  Existing variants never import or inspect them.
            from .vae_trt_native_int8 import (
                NATIVE_INT8_PLUGIN_CREATORS,
                NATIVE_INT8_PLUGIN_INIT_SYMBOL,
                NATIVE_INT8_PLUGIN_NAMESPACE,
                NATIVE_INT8_PLUGIN_VERSION,
                load_native_int8_manifest,
                validate_native_int8_manifest,
            )

            native_manifest = load_native_int8_manifest(self.engine_dir)
            native_validated = validate_native_int8_manifest(
                native_manifest,
                engine_dir=self.engine_dir,
                base_manifest=self.manifest,
                verify_hashes=True,
            )
            self._native_manifest = native_manifest
            self._native_validated = native_validated
            self._native_plugin_library = _load_fusion_plugin_library(
                trt=trt,
                plugin_path=native_validated["plugin_path"],
                creator_names=NATIVE_INT8_PLUGIN_CREATORS,
                plugin_version=NATIVE_INT8_PLUGIN_VERSION,
                plugin_namespace=NATIVE_INT8_PLUGIN_NAMESPACE,
                init_symbol=NATIVE_INT8_PLUGIN_INIT_SYMBOL,
            )
            self._configure_native_int8_kernel_profile(
                library=self._native_plugin_library,
                ctypes_module=__import__("ctypes"),
            )
        elif variant == "native_int8_v2":
            from .vae_trt_native_int8_v2 import (
                NATIVE_INT8_V2_PLUGIN_INIT_SYMBOL,
                NATIVE_INT8_V2_PLUGIN_NAME,
                NATIVE_INT8_V2_PLUGIN_NAMESPACE,
                NATIVE_INT8_V2_PLUGIN_VERSION,
                NATIVE_INT8_V2_P1_ALGORITHM,
                NATIVE_INT8_V2_SERIALIZATION_ABI_REVISION,
                load_native_int8_v2_manifest,
                validate_native_int8_v2_manifest,
            )

            native_manifest = load_native_int8_v2_manifest(
                self.engine_dir, level=native_int8_v2_level
            )
            native_validated = validate_native_int8_v2_manifest(
                native_manifest,
                engine_dir=self.engine_dir,
                base_manifest=self.manifest,
                level=native_int8_v2_level,
                verify_hashes=True,
            )
            self._native_manifest = native_manifest
            self._native_validated = native_validated
            self._native_plugin_library = _load_fusion_plugin_library(
                trt=trt,
                plugin_path=native_validated["plugin_path"],
                creator_names=(NATIVE_INT8_V2_PLUGIN_NAME,),
                plugin_version=NATIVE_INT8_V2_PLUGIN_VERSION,
                plugin_namespace=NATIVE_INT8_V2_PLUGIN_NAMESPACE,
                init_symbol=NATIVE_INT8_V2_PLUGIN_INIT_SYMBOL,
            )
            if native_int8_v2_level == "p1":
                self._validate_native_int8_v2_p1_kernel_contract(
                    library=self._native_plugin_library,
                    ctypes_module=__import__("ctypes"),
                    expected_algorithm=NATIVE_INT8_V2_P1_ALGORITHM,
                    expected_serialization_abi=(
                        NATIVE_INT8_V2_SERIALIZATION_ABI_REVISION
                    ),
                )
            self._configure_native_int8_kernel_profile(
                library=self._native_plugin_library,
                ctypes_module=__import__("ctypes"),
            )
        layer_profile_validated = None
        if enable_trt_layer_profile:
            # This import is intentionally gated.  Production runtime startup
            # neither reads diagnostic artifacts nor imports profiler helpers.
            from .vae_trt_profile import (
                load_trt_layer_profile_manifest,
                validate_trt_layer_profile_manifest,
            )

            if fusion_validated is not None and variant == "fusion_v1":
                layer_profile_validated = {
                    "schema_version": 2,
                    "scope": "vae_profile_only",
                    "precision": "int8",
                    "plan_kind": "audited_int8_fusion_v1",
                    "engines": fusion_validated["engines"],
                    "int8_audit": fusion_validated["audit"],
                    "build": dict(self._fusion_manifest.get("build", {})),
                }
                self._layer_profile_manifest = self._fusion_manifest
            elif fusion_validated is not None and variant == "fusion_v2":
                initial_inspector_path = (
                    Path(self.engine_dir) / "initial_int8_qdq_v5_inspector.json"
                )
                try:
                    initial_inspector = json.loads(
                        initial_inspector_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        "fusion-v2 layer profile requires the v5 initial inspector"
                    ) from exc
                initial_record = {
                    **validated["engines"]["initial"],
                    "inspector": initial_inspector,
                }
                layer_profile_validated = {
                    "schema_version": 2,
                    "scope": "vae_profile_only",
                    "precision": "int8",
                    "plan_kind": "raw_initial_plus_fusion_v2_steady",
                    "engines": {
                        "initial": initial_record,
                        "steady": fusion_validated["steady_engine"],
                    },
                    "int8_audit": fusion_validated["audit"],
                    "build": dict(self._fusion_manifest.get("build", {})),
                }
                self._layer_profile_manifest = self._fusion_manifest
            elif native_validated is not None and variant in {
                "native_int8_v1",
                "native_int8_v2",
            }:
                native_audit = dict(native_validated["audit"])
                native_audit["native_plugin_mappings"] = {
                    kind: [
                        {
                            **dict(record),
                            "logical_call_sites": [
                                f"int8/{kind}/{record['block_prefix']}.conv1/"
                                f"call_{record['call_index']}",
                                f"int8/{kind}/{record['block_prefix']}.conv2/"
                                f"call_{record['call_index']}",
                            ],
                        }
                        for record in native_validated["profile_ids"].get(kind, [])
                    ]
                    for kind in ("initial", "steady")
                }
                layer_profile_validated = {
                    "schema_version": 3,
                    "scope": "vae_profile_only",
                    "precision": "int8",
                    "plan_kind": (
                        "native_int8_v1_detailed"
                        if variant == "native_int8_v1"
                        else f"native_int8_v2_{native_int8_v2_level}_detailed"
                    ),
                    "engines": native_validated["engines"],
                    "int8_audit": native_audit,
                    "build": dict(self._native_manifest.get("build", {})),
                }
                self._layer_profile_manifest = self._native_manifest
            else:
                layer_profile_manifest = load_trt_layer_profile_manifest(
                    self.engine_dir
                )
                layer_profile_validated = validate_trt_layer_profile_manifest(
                    layer_profile_manifest,
                    engine_dir=self.engine_dir,
                    precision=precision,
                    production_manifest=self.manifest,
                    verify_hashes=True,
                )
                self._layer_profile_manifest = layer_profile_manifest
        else:
            self._layer_profile_manifest = None
        self._validate_environment(torch=torch, trt=trt)
        self._torch = torch
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._trt_runtime = trt.Runtime(self._logger)
        self._cache_shapes: list[tuple[int, ...]] = validated["cache_shapes"]
        self._engines: dict[str, Any] = {}
        self._contexts: dict[str, Any] = {}
        self._context_nvtx_verbosity: dict[str, str] = {}
        if layer_profile_validated is not None:
            selected_engines = layer_profile_validated["engines"]
        elif native_validated is not None:
            selected_engines = native_validated["engines"]
        elif variant == "fusion_v2" and fusion_validated is not None:
            selected_engines = {
                "initial": validated["engines"]["initial"],
                "steady": fusion_validated["steady_engine"],
            }
        elif fusion_validated is not None:
            selected_engines = fusion_validated["engines"]
        else:
            selected_engines = validated["engines"]
        self._active_plan_sha256 = {
            kind: str(selected_engines[kind]["sha256"])
            for kind in ("initial", "steady")
        }
        for kind in ("initial", "steady"):
            plan_path = Path(selected_engines[kind]["path"])
            engine = self._trt_runtime.deserialize_cuda_engine(plan_path.read_bytes())
            if engine is None:
                raise RuntimeError(f"could not deserialize TensorRT plan: {plan_path}")
            self._validate_engine_contract(kind=kind, engine=engine)
            context = engine.create_execution_context()
            if context is None:
                raise RuntimeError(
                    f"could not create TensorRT execution context: {plan_path}"
                )
            self._context_nvtx_verbosity[kind] = _configure_context_nvtx(
                context=context,
                trt=trt,
                precision=precision,
                enable_nvtx=enable_nvtx,
                detailed_plan=(
                    enable_trt_layer_profile
                    or variant == "fusion_v1"
                    or (variant == "fusion_v2" and kind == "steady")
                    or variant in {"native_int8_v1", "native_int8_v2"}
                ),
            )
            self._engines[kind] = engine
            self._contexts[kind] = context
        self._layer_profile_captures: dict[str, Any] = {}
        self._layer_profile_plan_sha256: dict[str, str] = {}
        self._layer_profile_plan_kind: str | None = None
        self._layer_profile_environment: dict[str, Any] = {}
        self._layer_profile_qdq_schema_version: int | None = None
        self._layer_profile_weight_encoding: str | None = None
        if layer_profile_validated is not None:
            from .vae_trt_profile import (
                TrtLayerProfileCapture,
                build_physical_layer_catalog,
                probe_trt_layer_profile_api,
            )

            probe_trt_layer_profile_api(trt=trt, contexts=self._contexts)
            self._layer_profile_plan_kind = layer_profile_validated["plan_kind"]
            self._layer_profile_environment = dict(
                layer_profile_validated.get("build", {})
            )
            audit = layer_profile_validated.get("int8_audit")
            if isinstance(audit, dict):
                self._layer_profile_qdq_schema_version = audit.get(
                    "qdq_schema_version", audit.get("schema_version")
                )
                self._layer_profile_weight_encoding = audit.get("weight_encoding")
            for kind in ("initial", "steady"):
                record = layer_profile_validated["engines"][kind]
                plan_sha256 = str(record["sha256"])
                catalog = build_physical_layer_catalog(
                    engine_kind=kind,
                    plan_sha256=plan_sha256,
                    inspector=record["inspector"],
                    precision=precision,
                    int8_audit=audit,
                    fp16_target_call_sites=(
                        record.get("target_call_sites") if precision == "fp16" else None
                    ),
                )
                self._layer_profile_plan_sha256[kind] = plan_sha256
                self._layer_profile_captures[kind] = TrtLayerProfileCapture(
                    trt=trt,
                    context=self._contexts[kind],
                    engine_kind=kind,
                    catalog=catalog,
                )
        self._fusion_v2_cache_slots: dict[int, dict[str, Any]] = {}
        if variant == "fusion_v2":
            self._fusion_v2_cache_slots = {
                int(entry["index"]): dict(entry)
                for entry in fusion_validated["selected_cache_slots"]
            }
        self._native_cache_slots: dict[int, dict[str, Any]] = {}
        if variant in {"native_int8_v1", "native_int8_v2"}:
            self._native_cache_slots = {
                int(entry["index"]): dict(entry)
                for entry in native_validated["cache_bindings"]
                if entry["dtype"] == "int8"
            }
            all_native_bindings = {
                int(entry["index"]): dict(entry)
                for entry in native_validated["cache_bindings"]
            }
            if set(all_native_bindings) != set(range(len(self._cache_shapes))):
                raise ValueError("native INT8 mixed-cache indices are incomplete")
            for index, shape in enumerate(self._cache_shapes):
                if tuple(all_native_bindings[index]["shape"]) != tuple(shape):
                    raise ValueError(
                        f"native INT8 cache slot {index} shape differs from base ABI"
                    )
        self._cache_dtypes = [
            torch.int8
            if index in self._fusion_v2_cache_slots
            or index in self._native_cache_slots
            else torch.float16
            for index in range(len(self._cache_shapes))
        ]
        self._cache_banks = [
            [
                torch.empty(shape, device=device, dtype=self._cache_dtypes[index])
                for index, shape in enumerate(self._cache_shapes)
            ]
            for _ in range(2)
        ]
        self._initial_cache_bank = (
            [
                torch.empty(shape, device=device, dtype=torch.float16)
                for shape in self._cache_shapes
            ]
            if variant == "fusion_v2"
            else None
        )
        self._cache_bank_bytes = sum(
            math.prod(shape) * (1 if dtype == torch.int8 else 2)
            for shape, dtype in zip(
                self._cache_shapes, self._cache_dtypes, strict=True
            )
        )
        self._rgb_outputs = {
            "initial": torch.empty(
                (1, 3, 9, TRT_VAE_HEIGHT, TRT_VAE_WIDTH),
                device=device,
                dtype=torch.float16,
            ),
            "steady": torch.empty(
                (1, 3, 12, TRT_VAE_HEIGHT, TRT_VAE_WIDTH),
                device=device,
                dtype=torch.float16,
            ),
        }
        self._execution_stream = _create_trt_execution_stream(
            torch=torch,
            device=device,
        )
        self._request_active = False
        self._next_chunk_index = 0
        self._read_bank_index: int | None = None
        self._pending_cache_migration_events: tuple[Any, Any] | None = None

    def _configure_fusion_v2_migration(
        self, *, library: Any, ctypes_module: Any
    ) -> None:
        try:
            function = library.sfwanFusionV2MigrateCache
        except AttributeError as exc:
            raise RuntimeError(
                "fusion-v2 plugin has no sfwanFusionV2MigrateCache launcher"
            ) from exc
        function.argtypes = [
            ctypes_module.c_void_p,
            ctypes_module.c_void_p,
            ctypes_module.c_float,
            ctypes_module.POINTER(ctypes_module.c_int32),
            ctypes_module.c_void_p,
        ]
        function.restype = ctypes_module.c_int32
        self._fusion_v2_migrate_cache = function
        self._fusion_v2_ctypes = ctypes_module

    def _configure_native_int8_kernel_profile(
        self, *, library: Any, ctypes_module: Any
    ) -> None:
        """Bind the opt-in native timing API without affecting other variants."""

        symbol_prefix = (
            "sfwanNativeInt8V2"
            if self.variant == "native_int8_v2"
            else "sfwanNativeInt8"
        )
        symbols = {}
        names = {
            "set": f"{symbol_prefix}SetProfiling",
            "reset": f"{symbol_prefix}ResetProfiles",
            "read": f"{symbol_prefix}ReadProfile",
        }
        for name in names.values():
            try:
                symbols[name] = getattr(library, name)
            except AttributeError as exc:
                raise RuntimeError(
                    f"native INT8 plugin has no required profiling symbol {name}"
                ) from exc
        symbols[names["set"]].argtypes = [ctypes_module.c_int32]
        symbols[names["set"]].restype = ctypes_module.c_int32
        symbols[names["reset"]].argtypes = []
        symbols[names["reset"]].restype = ctypes_module.c_int32
        symbols[names["read"]].argtypes = [
            ctypes_module.c_int32,
            ctypes_module.POINTER(ctypes_module.c_float),
            ctypes_module.c_int32,
            ctypes_module.POINTER(ctypes_module.c_int64),
        ]
        symbols[names["read"]].restype = ctypes_module.c_int32
        enabled = int(self.enable_native_int8_kernel_profile)
        if int(symbols[names["set"]](enabled)) != 0:
            raise RuntimeError("native INT8 plugin rejected profiling mode")
        self._native_profile_set = symbols[names["set"]]
        self._native_profile_reset = symbols[names["reset"]]
        self._native_profile_read = symbols[names["read"]]
        self._native_profile_ctypes = ctypes_module

    def _validate_native_int8_v2_p1_kernel_contract(
        self,
        *,
        library: Any,
        ctypes_module: Any,
        expected_algorithm: str,
        expected_serialization_abi: int,
    ) -> None:
        """Bind the loaded P1 plan only to the corrected compatible kernel."""

        try:
            algorithm = library.sfwanNativeInt8V2P1Algorithm
            contract = library.sfwanNativeInt8V2P1KernelContract
        except AttributeError as exc:
            raise RuntimeError(
                "Native INT8 V2 P1 plugin lacks its kernel contract"
            ) from exc
        algorithm.argtypes = []
        algorithm.restype = ctypes_module.c_char_p
        values = (ctypes_module.c_uint64 * 5)()
        contract.argtypes = [
            ctypes_module.POINTER(ctypes_module.c_uint64),
            ctypes_module.c_int32,
        ]
        contract.restype = ctypes_module.c_int32
        status = int(contract(values, len(values)))
        actual = algorithm().decode("ascii")
        expected = (1, 0, 0, 0, int(expected_serialization_abi))
        observed = tuple(int(value) for value in values)
        if status != 0 or actual != expected_algorithm or observed != expected:
            raise RuntimeError(
                "Native INT8 V2 P1 loaded DSO contract mismatch: "
                f"algorithm={actual!r}, values={observed!r}"
            )

    def _begin_native_int8_kernel_profile(self) -> None:
        if not self.enable_native_int8_kernel_profile:
            return
        if int(self._native_profile_reset()) != 0:
            raise RuntimeError("native INT8 plugin could not reset profile counters")

    def _read_native_int8_kernel_profile(
        self, *, kind: str, chunk_index: int
    ) -> dict[str, Any] | None:
        if not self.enable_native_int8_kernel_profile:
            return None
        if self.variant == "native_int8_v2" and self.native_int8_v2_level == "p1":
            stage_names = (
                "entry_norm_silu_quant_cache_write_ms",
                "conv1_mainloop_epilogue_ms",
                "mid_norm_silu_quant_cache_write_ms",
                "conv2_mma_and_fused_residual_epilogue_ms",
                "residual_epilogue_ms",
                "group_exit_ms",
                "residual_block_total_ms",
            )
        elif self.variant == "native_int8_v2" and self.native_int8_v2_level == "p2":
            stage_names = (
                "entry_norm_silu_quant_cache_ms",
                "direct_causal_iterator_conv1_ms",
                "mid_norm_silu_quant_cache_ms",
                "direct_causal_iterator_conv2_residual_ms",
                "fused_dequant_bias_residual_requant_ms",
                "group_exit_ms",
                "residual_block_total_ms",
            )
        elif self.variant == "native_int8_v2":
            stage_names = (
                "persistent_block_total_ms",
                "persistent_conv1_ms",
                "persistent_mid_norm_silu_quant_ms",
                "persistent_conv2_residual_ms",
                "persistent_cache_store_ms",
                "persistent_group_exit_ms",
                "residual_block_total_ms",
            )
        else:
            stage_names = (
                "entry_norm_silu_quant_cache_write_ms",
                "conv1_mainloop_epilogue_ms",
                "mid_norm_silu_quant_cache_write_ms",
                "conv2_mainloop_ms",
                "residual_epilogue_ms",
                "group_exit_ms",
                "residual_block_total_ms",
            )
        ctypes_module = self._native_profile_ctypes
        records = self._native_validated["profile_ids"].get(kind, [])
        per_block: list[dict[str, Any]] = []
        stage_totals = {name: 0.0 for name in stage_names}
        signature_totals: dict[str, dict[str, Any]] = {}
        total_calls = 0
        tile_map = (
            self._native_manifest["kernel_catalog"]["kernel_tile_map"]
            if self.variant == "native_int8_v2"
            else self._native_manifest["tune"]["kernel_tile_map"]
        )
        for record in records:
            values = (ctypes_module.c_float * len(stage_names))()
            calls = ctypes_module.c_int64()
            status = self._native_profile_read(
                ctypes_module.c_int32(int(record["profile_id"])),
                values,
                ctypes_module.c_int32(len(stage_names)),
                ctypes_module.byref(calls),
            )
            if int(status) != 0:
                raise RuntimeError(
                    f"native INT8 profile read failed for id {record['profile_id']}"
                )
            if int(calls.value) != 1:
                raise RuntimeError(
                    f"native INT8 profile id {record['profile_id']} reported "
                    f"{calls.value} calls; expected one"
                )
            stages = {
                name: float(values[index])
                for index, name in enumerate(stage_names)
            }
            for name, value in stages.items():
                stage_totals[name] += value
            total_calls += int(calls.value)
            signature_stage_keys = (
                (
                    "conv1_signature",
                    "direct_causal_iterator_conv1_ms",
                ),
                (
                    "conv2_signature",
                    "direct_causal_iterator_conv2_residual_ms",
                ),
            ) if self.variant == "native_int8_v2" and self.native_int8_v2_level == "p2" else (
                ("conv1_signature", stage_names[1]),
                ("conv2_signature", stage_names[3]),
            )
            for signature_key, stage_key in signature_stage_keys:
                signature = record.get(signature_key)
                if not isinstance(signature, str):
                    continue
                summary = signature_totals.setdefault(
                    signature,
                    {
                        "tile_id": tile_map.get(signature),
                        "call_count": 0,
                        "conv_cuda_ms": 0.0,
                    },
                )
                summary["call_count"] += 1
                summary["conv_cuda_ms"] += stages[stage_key]
            per_block.append(
                {
                    "profile_id": int(record["profile_id"]),
                    "block_prefix": record["block_prefix"],
                    "call_index": int(record["call_index"]),
                    "conv1_signature": record.get("conv1_signature"),
                    "conv2_signature": record.get("conv2_signature"),
                    "stages": stages,
                }
            )
        result = {
            "schema_version": 2 if self.variant == "native_int8_v2" else 1,
            "engine_kind": kind,
            "chunk_index": chunk_index,
            "native_int8_variant": self.variant,
            "native_int8_v2_level": (
                self.native_int8_v2_level
                if self.variant == "native_int8_v2"
                else None
            ),
            "profiled_block_call_count": total_calls,
            "stage_totals_ms": stage_totals,
            "per_signature": signature_totals,
            "blocks": per_block,
        }
        if self.variant == "native_int8_v2" and self.native_int8_v2_level == "p1":
            result["residual_epilogue_status"] = "fused_into_conv2"
        return result

    def _validate_engine_contract(self, *, kind: str, engine: Any) -> None:
        trt = self._trt
        expected: dict[str, tuple[Any, tuple[int, ...]]] = {
            "latent": (trt.TensorIOMode.INPUT, TRT_VAE_LATENT_SHAPE),
            "rgb": (
                trt.TensorIOMode.OUTPUT,
                (
                    1,
                    3,
                    9 if kind == "initial" else 12,
                    TRT_VAE_HEIGHT,
                    TRT_VAE_WIDTH,
                ),
            ),
        }
        for index, shape in enumerate(self._cache_shapes):
            if kind == "steady":
                expected[f"cache_in_{index:03d}"] = (
                    trt.TensorIOMode.INPUT,
                    shape,
                )
            expected[f"cache_out_{index:03d}"] = (
                trt.TensorIOMode.OUTPUT,
                shape,
            )
        actual_names = {
            engine.get_tensor_name(index) for index in range(int(engine.num_io_tensors))
        }
        if actual_names != set(expected):
            raise ValueError(
                f"TensorRT {kind} engine bindings differ from the manifest contract; "
                f"missing={sorted(set(expected) - actual_names)}, "
                f"extra={sorted(actual_names - set(expected))}"
            )
        selected_cache_indices = set(
            getattr(self, "_fusion_v2_cache_slots", {}).keys()
        )
        # During context creation the slot map is populated from the already
        # validated fusion-v2 manifest.  Initial remains the raw v5 FP16 ABI;
        # only selected steady cache bindings use INT8/CDHW32.
        if self.variant == "fusion_v2" and not selected_cache_indices:
            selected_cache_indices = {
                int(entry["index"])
                for entry in self._fusion_validated["selected_cache_slots"]
            }
        native_cache_indices = set(
            getattr(self, "_native_cache_slots", {}).keys()
        )
        if self.variant in {"native_int8_v1", "native_int8_v2"} and not native_cache_indices:
            native_cache_indices = {
                int(entry["index"])
                for entry in self._native_validated["cache_bindings"]
                if entry["dtype"] == "int8"
            }
        for name, (mode, shape) in expected.items():
            if engine.get_tensor_mode(name) != mode:
                raise ValueError(f"TensorRT binding {name!r} has the wrong I/O mode")
            if tuple(engine.get_tensor_shape(name)) != shape:
                raise ValueError(
                    f"TensorRT binding {name!r} shape is "
                    f"{tuple(engine.get_tensor_shape(name))}, expected {shape}"
                )
            expected_dtype = trt.float16
            match = None
            if name.startswith("cache_in_") or name.startswith("cache_out_"):
                try:
                    match = int(name.rsplit("_", 1)[1])
                except (IndexError, ValueError):
                    match = None
            if (
                self.variant == "fusion_v2"
                and kind == "steady"
                and match in selected_cache_indices
            ):
                expected_dtype = trt.int8
            if self.variant in {"native_int8_v1", "native_int8_v2"} and match in native_cache_indices:
                expected_dtype = trt.int8
            if engine.get_tensor_dtype(name) != expected_dtype:
                raise ValueError(
                    f"TensorRT binding {name!r} must expose {expected_dtype}, "
                    f"got {engine.get_tensor_dtype(name)}"
                )
            if (
                self.variant in {"native_int8_v1", "native_int8_v2"}
                and match in native_cache_indices
                and callable(getattr(engine, "get_tensor_format", None))
            ):
                expected_format = getattr(trt.TensorFormat, "CDHW32", None)
                if expected_format is None:
                    raise ValueError(
                        "TensorRT runtime has no CDHW32 tensor-format symbol"
                    )
                actual_format = engine.get_tensor_format(name)
                if actual_format != expected_format:
                    raise ValueError(
                        f"native INT8 cache binding {name!r} must expose "
                        f"CDHW32, got {actual_format}"
                    )

    def _validate_environment(self, *, torch: Any, trt: Any) -> None:
        build = self.manifest.get("build")
        if not isinstance(build, dict):
            raise ValueError("TensorRT VAE manifest has no build environment")
        capability = list(torch.cuda.get_device_capability(self.device))
        if capability != build.get("compute_capability"):
            raise ValueError(
                f"TensorRT VAE plan targets SM{build.get('compute_capability')}, "
                f"current device is SM{capability}"
            )
        if capability != [8, 7]:
            raise ValueError("Jetson SFWan TensorRT VAE V1 supports SM87 only")
        if str(trt.__version__) != str(build.get("tensorrt_version")):
            raise ValueError(
                "TensorRT VAE plan/runtime version mismatch: "
                f"{build.get('tensorrt_version')} != {trt.__version__}"
            )
        current_cuda = str(torch.version.cuda)
        built_cuda = str(build.get("cuda_version"))
        if _version_prefix(current_cuda) != _version_prefix(built_cuda):
            raise ValueError(
                f"TensorRT VAE CUDA major/minor mismatch: {built_cuda} != {current_cuda}"
            )

    @property
    def contract(self) -> dict[str, Any]:
        fusion_validated = getattr(self, "_fusion_validated", None)
        fusion_audit = (
            fusion_validated.get("audit")
            if isinstance(fusion_validated, dict)
            else None
        )
        native_validated = getattr(self, "_native_validated", None)
        native_audit = (
            native_validated.get("audit")
            if isinstance(native_validated, dict)
            else None
        )
        contract = {
            "vae_backend": "tensorrt",
            "vae_engine_dir": self.engine_dir,
            "vae_engine_precision": self.precision,
            "vae_engine_schema_version": self.manifest["schema_version"],
            "vae_engine_sm": self.manifest["build"]["compute_capability"],
            "vae_engine_tensorrt_version": self.manifest["build"]["tensorrt_version"],
            "vae_engine_cuda_version": self.manifest["build"]["cuda_version"],
            "vae_runtime_gpu_name": str(self._torch.cuda.get_device_name(self.device)),
            "vae_runtime_compute_capability": list(
                self._torch.cuda.get_device_capability(self.device)
            ),
            "vae_runtime_cuda_version": str(self._torch.version.cuda),
            "vae_runtime_torch_version": str(self._torch.__version__),
            "vae_int8_audit_passed": bool(
                native_audit.get("passed", False)
                if isinstance(native_audit, dict)
                else fusion_audit.get("passed", False)
                if isinstance(fusion_audit, dict)
                else self.manifest.get("int8_audit", {}).get("passed", False)
            ),
            "vae_trt_variant": self.variant,
            "vae_engine_plan_sha256": dict(self._active_plan_sha256),
            "vae_runtime_nvtx_verbosity": dict(self._context_nvtx_verbosity),
            "vae_cache_tensor_count": TRT_VAE_CACHE_COUNT,
            "vae_cache_bank_bytes": self._cache_bank_bytes,
            "vae_cache_double_bank_bytes": self._cache_bank_bytes * 2,
        }
        if self.variant == "fusion_v1":
            plugin = self._fusion_manifest["plugin"]
            contract.update(
                {
                    "vae_trt_plugin_sha256": plugin["sha256"],
                    "vae_trt_fusion_audit_passed": bool(
                        self._fusion_manifest["audit"]["passed"]
                    ),
                    "vae_trt_fusion_counts": dict(fusion_validated["fusion_counts"]),
                    "vae_trt_fusion_per_engine_counts": dict(
                        fusion_validated["per_engine_fusion_counts"]
                    ),
                }
            )
        elif self.variant == "fusion_v2":
            plugin = self._fusion_manifest["plugin"]
            selected = self._fusion_validated["selected_cache_slots"]
            int8_bytes = sum(
                math.prod(entry["shape"]) for entry in selected
            )
            selected_indices = {int(entry["index"]) for entry in selected}
            fp16_bytes = sum(
                math.prod(shape) * 2
                for index, shape in enumerate(self._cache_shapes)
                if index not in selected_indices
            )
            contract.update(
                {
                    "vae_trt_plugin_sha256": plugin["sha256"],
                    "vae_trt_fusion_audit_passed": True,
                    "vae_trt_fusion_counts": dict(
                        self._fusion_manifest["graph"]["plugin_counts"]
                    ),
                    "vae_trt_initial_variant": "raw_int8_v5",
                    "vae_trt_steady_variant": "fusion_v2",
                    "vae_cache_selected_int8_slots": sorted(selected_indices),
                    "vae_cache_selected_int8_slot_count": len(selected_indices),
                    "vae_cache_steady_mixed_bank_bytes": int8_bytes
                    + fp16_bytes,
                    "vae_cache_migration": "one_time_after_chunk_0",
                }
            )
        elif self.variant == "native_int8_v1":
            plugin = self._native_manifest["plugin"]
            cache = self._native_manifest["cache"]
            contract.update(
                {
                    "vae_trt_plugin_sha256": plugin["sha256"],
                    "native_int8_schema_version": self._native_manifest[
                        "schema_version"
                    ],
                    "native_int8_plugin_sha256": plugin["sha256"],
                    "native_int8_initial_plan_sha256": self._active_plan_sha256[
                        "initial"
                    ],
                    "native_int8_steady_plan_sha256": self._active_plan_sha256[
                        "steady"
                    ],
                    "native_int8_target_logical_conv_count": 28,
                    "native_int8_initial_call_site_count": 84,
                    "native_int8_steady_call_site_count": 84,
                    "native_int8_signature_count": 9,
                    "native_int8_cache_int8_slot_count": cache[
                        "int8_slot_count"
                    ],
                    "native_int8_cache_fp16_slot_count": cache[
                        "fp16_slot_count"
                    ],
                    "native_int8_cache_bank_bytes": cache[
                        "single_bank_bytes"
                    ],
                    "native_int8_runtime_scale_mode": "static",
                    "native_int8_weight_mode": "offline_per_channel_packed",
                    "native_int8_kernel_tile_map": dict(
                        self._native_manifest["tune"]["kernel_tile_map"]
                    ),
                    "native_int8_algorithm": self._native_manifest["algorithm"],
                    "native_int8_weight_layout": self._native_manifest["weights"][
                        "layout"
                    ],
                    "native_int8_cutlass_commit": plugin["cutlass_commit"],
                    "native_int8_audit_passed": bool(native_audit["passed"]),
                    "native_int8_kernel_profile_enabled": (
                        self.enable_native_int8_kernel_profile
                    ),
                    "vae_cache_migration": "none",
                }
            )
        elif self.variant == "native_int8_v2":
            plugin = self._native_manifest["plugin"]
            cache = self._native_manifest["cache"]
            audit = self._native_validated["audit"]
            contract.update(
                {
                    "vae_trt_plugin_sha256": plugin["sha256"],
                    "native_int8_v2_level": self.native_int8_v2_level,
                    "native_int8_v2_schema_version": self._native_manifest[
                        "schema_version"
                    ],
                    "native_int8_v2_plugin_sha256": plugin["sha256"],
                    "native_int8_v2_plan_sha256": dict(
                        self._active_plan_sha256
                    ),
                    "native_int8_v2_accumulator1_workspace_bytes": audit[
                        "accumulator1_workspace_bytes"
                    ],
                    "native_int8_v2_accumulator2_workspace_bytes": audit[
                        "accumulator2_workspace_bytes"
                    ],
                    "native_int8_v2_temporal_window_bytes": audit[
                        "temporal_window_bytes"
                    ],
                    "native_int8_v2_direct_causal_iterator": audit[
                        "direct_causal_iterator"
                    ],
                    "native_int8_v2_persistent_block_count": audit[
                        "persistent_block_count"
                    ],
                    "native_int8_v2_signature_kernel_map": dict(
                        self._native_manifest["kernel_catalog"][
                            "kernel_tile_map"
                        ]
                    ),
                    "native_int8_v2_audit_passed": bool(audit["passed"]),
                    "native_int8_v2_plan_reused": bool(
                        self._native_manifest.get("plan_reused", False)
                    ),
                    "native_int8_v2_plugin_rebound": bool(
                        self._native_manifest.get("plugin_rebound", False)
                    ),
                    "native_int8_v2_p1_algorithm": audit.get("p1_algorithm"),
                    "native_int8_v2_p1_kernel_revision": audit.get(
                        "p1_kernel_revision"
                    ),
                    "native_int8_v2_p1_kernel_contract": audit.get(
                        "p1_kernel_contract"
                    ),
                    "native_int8_cache_int8_slot_count": cache[
                        "int8_slot_count"
                    ],
                    "native_int8_cache_fp16_slot_count": cache[
                        "fp16_slot_count"
                    ],
                    "native_int8_cache_bank_bytes": cache[
                        "single_bank_bytes"
                    ],
                    "native_int8_runtime_scale_mode": "static",
                    "native_int8_weight_mode": "offline_per_channel_packed",
                    "native_int8_cutlass_commit": plugin["cutlass_commit"],
                    "native_int8_kernel_profile_enabled": (
                        self.enable_native_int8_kernel_profile
                    ),
                    "vae_cache_migration": "none",
                }
            )
        else:
            contract["native_int8_kernel_profile_enabled"] = False
        if self.enable_trt_layer_profile:
            from .vae_trt_profile import TRT_LAYER_PROFILE_SCHEMA_VERSION

            contract.update(
                {
                    "trt_layer_profile_enabled": True,
                    "trt_layer_profile_scope": "vae_profile_only",
                    "trt_layer_profile_schema_version": (
                        TRT_LAYER_PROFILE_SCHEMA_VERSION
                    ),
                    "trt_layer_profile_plan_kind": self._layer_profile_plan_kind,
                    "trt_layer_profile_plan_sha256": dict(
                        self._layer_profile_plan_sha256
                    ),
                }
            )
        else:
            contract["trt_layer_profile_enabled"] = False
        return contract

    @property
    def layer_profile_metadata(self) -> dict[str, Any] | None:
        if not self.enable_trt_layer_profile:
            return None
        catalogs = {
            kind: capture.catalog
            for kind, capture in self._layer_profile_captures.items()
        }
        if any(catalog is None for catalog in catalogs.values()):
            return None
        return {
            "environment": dict(self._layer_profile_environment),
            "precision": self.precision,
            "plan_kind": self._layer_profile_plan_kind,
            "plan_sha256": dict(self._layer_profile_plan_sha256),
            "qdq_schema_version": self._layer_profile_qdq_schema_version,
            "weight_encoding": self._layer_profile_weight_encoding,
            "catalogs": catalogs,
        }

    def _abort_layer_profile_captures(self) -> None:
        for capture in getattr(self, "_layer_profile_captures", {}).values():
            capture.abort_capture()

    def reset_request(self) -> None:
        self._abort_layer_profile_captures()
        if self.enable_native_int8_kernel_profile:
            self._begin_native_int8_kernel_profile()
        self._request_active = True
        self._next_chunk_index = 0
        self._read_bank_index = None
        self._pending_cache_migration_events = None

    def finish_request(self) -> None:
        self._abort_layer_profile_captures()
        self._request_active = False
        self._next_chunk_index = 0
        self._read_bank_index = None
        self._pending_cache_migration_events = None

    def _set_address(self, context: Any, name: str, tensor: Any) -> None:
        if not context.set_tensor_address(name, int(tensor.data_ptr())):
            raise RuntimeError(f"TensorRT rejected the address for binding {name!r}")

    def _migrate_fusion_v2_initial_cache(self, *, execution_stream: Any) -> None:
        """Convert six raw-v5 FP16 slots once and seed steady bank zero."""

        if self.variant != "fusion_v2" or self._initial_cache_bank is None:
            raise RuntimeError("fusion-v2 cache migration was requested incorrectly")
        torch = self._torch
        destination_bank = self._cache_banks[0]
        start = end = None
        if self.enable_profile:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(execution_stream)
        ctypes_module = self._fusion_v2_ctypes
        for index, (source, destination) in enumerate(
            zip(self._initial_cache_bank, destination_bank, strict=True)
        ):
            source.record_stream(execution_stream)
            destination.record_stream(execution_stream)
            slot = self._fusion_v2_cache_slots.get(index)
            if slot is None:
                with torch.cuda.stream(execution_stream):
                    destination.copy_(source, non_blocking=True)
                continue
            shape_values = [int(value) for value in slot["shape"]]
            shape_array = (ctypes_module.c_int32 * 5)(*shape_values)
            status = self._fusion_v2_migrate_cache(
                ctypes_module.c_void_p(int(source.data_ptr())),
                ctypes_module.c_void_p(int(destination.data_ptr())),
                ctypes_module.c_float(float(slot["scale"])),
                shape_array,
                ctypes_module.c_void_p(int(execution_stream.cuda_stream)),
            )
            if int(status) != 0:
                raise RuntimeError(
                    f"fusion-v2 cache migration failed for slot {index}"
                )
        if self.enable_profile:
            end.record(execution_stream)
            self._pending_cache_migration_events = (start, end)
        self._read_bank_index = 0

    def _execute(
        self,
        *,
        kind: TrtVaeEngineKind,
        latent: Any,
        chunk_index: int | None = None,
    ) -> Any:
        torch = self._torch
        context = self._contexts[kind]
        fusion_v2_initial = self.variant == "fusion_v2" and kind == "initial"
        output_bank_index = (
            None
            if fusion_v2_initial
            else 0 if self._read_bank_index is None else 1 - self._read_bank_index
        )
        output_bank = (
            self._initial_cache_bank
            if fusion_v2_initial
            else self._cache_banks[output_bank_index]
        )
        if output_bank is None:
            raise RuntimeError("TensorRT VAE cache output bank is unavailable")
        self._set_address(context, "latent", latent)
        if kind == "steady":
            if self._read_bank_index is None:
                raise RuntimeError("steady TensorRT VAE execution has no input cache")
            for index, tensor in enumerate(self._cache_banks[self._read_bank_index]):
                self._set_address(context, f"cache_in_{index:03d}", tensor)
        rgb = self._rgb_outputs[kind]
        self._set_address(context, "rgb", rgb)
        for index, tensor in enumerate(output_bank):
            self._set_address(context, f"cache_out_{index:03d}", tensor)
        caller_stream = torch.cuda.current_stream(device=self.device)
        execution_stream = self._execution_stream
        bound_tensors = [latent, rgb, *output_bank]
        if kind == "steady":
            bound_tensors.extend(self._cache_banks[self._read_bank_index])
        capture = getattr(self, "_layer_profile_captures", {}).get(kind)
        if capture is not None:
            if chunk_index is None:
                raise RuntimeError("TensorRT layer profiling requires a chunk index")
            capture.begin_capture(chunk_index)
        dependency_established = False
        try:
            try:
                execution_stream.wait_stream(caller_stream)
                dependency_established = True
                for tensor in bound_tensors:
                    tensor.record_stream(execution_stream)
                if not context.execute_async_v3(
                    stream_handle=int(execution_stream.cuda_stream)
                ):
                    raise RuntimeError(
                        f"TensorRT {kind} VAE execution returned failure"
                    )
                if capture is not None:
                    capture.mark_enqueue_succeeded()
            finally:
                if dependency_established:
                    caller_stream.wait_stream(execution_stream)
        except BaseException:
            if capture is not None:
                capture.abort_capture()
            raise
        if not fusion_v2_initial:
            self._read_bank_index = output_bank_index
        return rgb

    def decode_chunk(
        self,
        *,
        chunk_index: int,
        denormalized_fp32: Any,
    ) -> tuple[Any, dict[str, Any]]:
        torch = self._torch
        if not self._request_active:
            raise RuntimeError("reset_request() must precede TensorRT VAE execution")
        if chunk_index != self._next_chunk_index:
            raise ValueError(
                f"TensorRT VAE expected chunk {self._next_chunk_index}, "
                f"got {chunk_index}"
            )
        if tuple(denormalized_fp32.shape) != TRT_VAE_LATENT_SHAPE:
            raise ValueError(
                f"TensorRT VAE expects latent shape {TRT_VAE_LATENT_SHAPE}, "
                f"got {tuple(denormalized_fp32.shape)}"
            )
        if denormalized_fp32.dtype != torch.float32:
            raise ValueError("TensorRT VAE expects FP32 denormalized latent input")
        kind: TrtVaeEngineKind = "initial" if chunk_index == 0 else "steady"

        cast_ms = None
        engine_ms = None
        if self.enable_profile:
            cast_start = torch.cuda.Event(enable_timing=True)
            cast_end = torch.cuda.Event(enable_timing=True)
            cast_start.record()
        if self.enable_nvtx:
            torch.cuda.nvtx.range_push(f"sfwan.vae.trt.{kind}.input_cast")
        try:
            latent_fp16 = denormalized_fp32.to(dtype=torch.float16)
        finally:
            if self.enable_nvtx:
                torch.cuda.nvtx.range_pop()
        if self.enable_profile:
            cast_end.record()
            cast_end.synchronize()
            cast_ms = float(cast_start.elapsed_time(cast_end))

        self._begin_native_int8_kernel_profile()
        if self.enable_profile:
            engine_start = torch.cuda.Event(enable_timing=True)
            engine_end = torch.cuda.Event(enable_timing=True)
            engine_start.record()
        if self.enable_nvtx:
            torch.cuda.nvtx.range_push(f"sfwan.vae.trt.{kind}.execute")
        try:
            if getattr(self, "enable_trt_layer_profile", False):
                output = self._execute(
                    kind=kind,
                    latent=latent_fp16,
                    chunk_index=chunk_index,
                )
            else:
                output = self._execute(kind=kind, latent=latent_fp16)
        finally:
            if self.enable_nvtx:
                torch.cuda.nvtx.range_pop()
        if self.enable_profile:
            engine_end.record()
            engine_end.synchronize()
            engine_ms = float(engine_start.elapsed_time(engine_end))
        native_kernel_profile = self._read_native_int8_kernel_profile(
            kind=kind,
            chunk_index=chunk_index,
        )
        if self.variant == "fusion_v2" and kind == "initial":
            caller_stream = torch.cuda.current_stream(device=self.device)
            execution_stream = self._execution_stream
            execution_stream.wait_stream(caller_stream)
            self._migrate_fusion_v2_initial_cache(
                execution_stream=execution_stream
            )
            caller_stream.wait_stream(execution_stream)
        cache_migration_ms = None
        if self._pending_cache_migration_events is not None:
            migration_start, migration_end = self._pending_cache_migration_events
            migration_end.synchronize()
            cache_migration_ms = float(
                migration_start.elapsed_time(migration_end)
            )
            self._pending_cache_migration_events = None

        layer_profile_metrics = None
        if getattr(self, "enable_trt_layer_profile", False):
            if engine_ms is None:
                self._layer_profile_captures[kind].abort_capture()
                raise RuntimeError(
                    "TensorRT layer profiling requires a CUDA engine-event time"
                )
            try:
                layer_profile_metrics = self._layer_profile_captures[
                    kind
                ].report_and_finish(engine_event_ms=engine_ms)
            except BaseException:
                self._layer_profile_captures[kind].abort_capture()
                raise

        self._next_chunk_index += 1
        metrics: dict[str, Any] = {
            "trt_engine_kind": kind,
            "trt_precision": self.precision,
        }
        if self.enable_profile:
            metrics.update(
                {
                    "trt_input_cast_cuda_ms": cast_ms,
                    "trt_engine_cuda_ms": engine_ms,
                }
            )
            if cache_migration_ms is not None:
                metrics["fusion_v2_cache_migration_cuda_ms"] = (
                    cache_migration_ms
                )
        if layer_profile_metrics is not None:
            metrics["trt_layer_profile"] = layer_profile_metrics
        if native_kernel_profile is not None:
            metrics["native_int8_kernel_profile"] = native_kernel_profile
        return output, metrics

    def close(self) -> None:
        self.finish_request()
        if getattr(self, "_native_profile_set", None) is not None:
            self._native_profile_set(0)
        getattr(self, "_layer_profile_captures", {}).clear()
        self._contexts.clear()
        self._engines.clear()
        self._cache_banks.clear()
        if self._initial_cache_bank is not None:
            self._initial_cache_bank.clear()
        self._rgb_outputs.clear()
        self._fusion_plugin_library = None
        self._native_plugin_library = None
