"""TensorRT runtime for the fixed-shape Jetson SFWan VAE engines."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

from .vae_trt_qdq import EXPECTED_CALL_SITES, QDQ_SCHEMA_VERSION, QDQ_TOPOLOGY

TRT_VAE_MANIFEST_SCHEMA_VERSION = 1
TRT_VAE_HEIGHT = 480
TRT_VAE_WIDTH = 832
TRT_VAE_LATENT_SHAPE = (1, 16, 3, 60, 104)
TRT_VAE_CACHE_COUNT = 32
TRT_VAE_CACHE_TOTAL_ELEMENTS = 944_286_720
TRT_VAE_CACHE_BANK_BYTES = TRT_VAE_CACHE_TOTAL_ELEMENTS * 2

TrtVaePrecision = Literal["fp16", "int8"]
TrtVaeEngineKind = Literal["initial", "steady"]


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


def _configure_context_nvtx(
    *, context: Any, trt: Any, precision: TrtVaePrecision, enable_nvtx: bool
) -> str:
    if precision != "int8":
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
            "TensorRT INT8 VAE could not set execution-context NVTX verbosity"
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
        audit = manifest.get("int8_audit")
        if not isinstance(audit, dict) or audit.get("passed") is not True:
            raise ValueError(
                "INT8 TensorRT VAE requires a passing structural and tactic audit"
            )
        if audit.get("schema_version") != QDQ_SCHEMA_VERSION:
            raise ValueError("INT8 TensorRT VAE manifest audit schema is invalid")
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
            "static_weight_not_int8_call_sites",
            "dynamic_filter_call_sites",
            "fp16_fallback_call_sites",
            "fp32_or_tf32_fallback_call_sites",
        )
        if (
            isinstance(signature_count, bool)
            or not isinstance(signature_count, int)
            or signature_count <= 0
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
        for kind in ("initial", "steady"):
            graph_audit = structural.get(kind)
            if (
                not isinstance(graph_audit, dict)
                or graph_audit.get("schema_version") != QDQ_SCHEMA_VERSION
                or graph_audit.get("qdq_topology") != QDQ_TOPOLOGY
                or graph_audit.get("passed") is not True
                or graph_audit.get("errors") != []
                or graph_audit.get("target_conv_call_site_count") != EXPECTED_CALL_SITES
                or graph_audit.get("activation_cast_count") != EXPECTED_CALL_SITES
                or graph_audit.get("activation_quantize_count") != EXPECTED_CALL_SITES
                or graph_audit.get("activation_dequantize_count") != EXPECTED_CALL_SITES
                or graph_audit.get("weight_quantize_count") != EXPECTED_CALL_SITES
                or graph_audit.get("weight_dequantize_count") != EXPECTED_CALL_SITES
                or graph_audit.get("output_cast_count") != EXPECTED_CALL_SITES
                or graph_audit.get("unique_weight_source_count") != EXPECTED_CALL_SITES
                or graph_audit.get("unique_bias_count") != EXPECTED_CALL_SITES
                or graph_audit.get("unexpected_target_cast_nodes") != []
            ):
                raise ValueError(
                    f"TensorRT VAE INT8 {kind} structural audit is invalid"
                )

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
        self.enable_nvtx = enable_nvtx
        self.manifest = load_trt_vae_manifest(self.engine_dir)
        validated = validate_trt_vae_manifest(
            self.manifest,
            engine_dir=self.engine_dir,
            precision=precision,
            model_path=model_path,
            verify_plan_hashes=True,
        )
        self._validate_environment(torch=torch, trt=trt)
        self._torch = torch
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._trt_runtime = trt.Runtime(self._logger)
        self._cache_shapes: list[tuple[int, ...]] = validated["cache_shapes"]
        self._engines: dict[str, Any] = {}
        self._contexts: dict[str, Any] = {}
        self._context_nvtx_verbosity: dict[str, str] = {}
        for kind in ("initial", "steady"):
            plan_path = Path(validated["engines"][kind]["path"])
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
            )
            self._engines[kind] = engine
            self._contexts[kind] = context
        self._cache_banks = [
            [
                torch.empty(shape, device=device, dtype=torch.float16)
                for shape in self._cache_shapes
            ]
            for _ in range(2)
        ]
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
        self._request_active = False
        self._next_chunk_index = 0
        self._read_bank_index: int | None = None

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
        for name, (mode, shape) in expected.items():
            if engine.get_tensor_mode(name) != mode:
                raise ValueError(f"TensorRT binding {name!r} has the wrong I/O mode")
            if tuple(engine.get_tensor_shape(name)) != shape:
                raise ValueError(
                    f"TensorRT binding {name!r} shape is "
                    f"{tuple(engine.get_tensor_shape(name))}, expected {shape}"
                )
            if engine.get_tensor_dtype(name) != trt.float16:
                raise ValueError(
                    f"TensorRT binding {name!r} must expose float16, "
                    f"got {engine.get_tensor_dtype(name)}"
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
        return {
            "vae_backend": "tensorrt",
            "vae_engine_dir": self.engine_dir,
            "vae_engine_precision": self.precision,
            "vae_engine_schema_version": self.manifest["schema_version"],
            "vae_engine_sm": self.manifest["build"]["compute_capability"],
            "vae_engine_tensorrt_version": self.manifest["build"]["tensorrt_version"],
            "vae_int8_audit_passed": bool(
                self.manifest.get("int8_audit", {}).get("passed", False)
            ),
            "vae_runtime_nvtx_verbosity": dict(self._context_nvtx_verbosity),
            "vae_cache_tensor_count": TRT_VAE_CACHE_COUNT,
            "vae_cache_bank_bytes": TRT_VAE_CACHE_BANK_BYTES,
            "vae_cache_double_bank_bytes": TRT_VAE_CACHE_BANK_BYTES * 2,
        }

    def reset_request(self) -> None:
        self._request_active = True
        self._next_chunk_index = 0
        self._read_bank_index = None

    def finish_request(self) -> None:
        self._request_active = False
        self._next_chunk_index = 0
        self._read_bank_index = None

    def _set_address(self, context: Any, name: str, tensor: Any) -> None:
        if not context.set_tensor_address(name, int(tensor.data_ptr())):
            raise RuntimeError(f"TensorRT rejected the address for binding {name!r}")

    def _execute(self, *, kind: TrtVaeEngineKind, latent: Any) -> Any:
        torch = self._torch
        context = self._contexts[kind]
        output_bank_index = (
            0 if self._read_bank_index is None else 1 - self._read_bank_index
        )
        output_bank = self._cache_banks[output_bank_index]
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
        stream = torch.cuda.current_stream(device=self.device)
        if not context.execute_async_v3(stream_handle=int(stream.cuda_stream)):
            raise RuntimeError(f"TensorRT {kind} VAE execution returned failure")
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

        if self.enable_profile:
            engine_start = torch.cuda.Event(enable_timing=True)
            engine_end = torch.cuda.Event(enable_timing=True)
            engine_start.record()
        if self.enable_nvtx:
            torch.cuda.nvtx.range_push(f"sfwan.vae.trt.{kind}.execute")
        try:
            output = self._execute(kind=kind, latent=latent_fp16)
        finally:
            if self.enable_nvtx:
                torch.cuda.nvtx.range_pop()
        if self.enable_profile:
            engine_end.record()
            engine_end.synchronize()
            engine_ms = float(engine_start.elapsed_time(engine_end))

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
        return output, metrics

    def close(self) -> None:
        self.finish_request()
        self._contexts.clear()
        self._engines.clear()
        self._cache_banks.clear()
        self._rgb_outputs.clear()
