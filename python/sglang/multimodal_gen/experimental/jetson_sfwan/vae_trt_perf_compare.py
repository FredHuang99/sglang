"""Validate and compare production SFWan VAE profile summaries.

The tool is intentionally import-light.  It consumes client JSON artifacts and
never imports Torch, TensorRT, ONNX, or server code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPECTED_HEIGHT = 480
EXPECTED_WIDTH = 832
EXPECTED_FRAMES = 81
EXPECTED_CHUNKS = 7
EXPECTED_WARMUP = 10
EXPECTED_REPEAT = 50
NATIVE_PROFILE_COMPONENTS = (
    "target_int8_conv_ms",
    "norm_silu_quant_cache_write_ms",
    "residual_epilogue_ms",
    "group_exit_ms",
    "native_residual_blocks_ms",
    "remaining_trt_operators_ms",
    "engine_external_ms",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid profile summary JSON: {source}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"profile summary must be an object: {source}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_ms(value: Any, *, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return numeric


def _stats(values: Sequence[float]) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("cannot summarize an empty sample")
    return {
        "mean_ms": statistics.fmean(numeric),
        "population_stddev_ms": statistics.pstdev(numeric),
        "min_ms": min(numeric),
        "max_ms": max(numeric),
        "sample_count": len(numeric),
    }


def _require_context(
    summary: Mapping[str, Any],
    *,
    label: str,
    expected_warmup: int,
    expected_repeat: int,
) -> dict[str, Any]:
    if summary.get("mode") != "profile-vae":
        raise ValueError(f"{label}: mode must be profile-vae")
    if summary.get("warmup") != expected_warmup:
        raise ValueError(
            f"{label}: warmup must be {expected_warmup}, got {summary.get('warmup')}"
        )
    if summary.get("repeat") != expected_repeat:
        raise ValueError(
            f"{label}: repeat must be {expected_repeat}, got {summary.get('repeat')}"
        )
    context = summary.get("measurement_context")
    if not isinstance(context, dict) or context.get("schema_version") != 1:
        raise ValueError(
            f"{label}: missing measurement_context schema 1; rerun with the "
            "current client"
        )
    request = context.get("request")
    run = context.get("run")
    server = context.get("server")
    if not all(isinstance(value, dict) for value in (request, run, server)):
        raise ValueError(f"{label}: measurement_context is incomplete")
    expected_request = {
        "height": EXPECTED_HEIGHT,
        "width": EXPECTED_WIDTH,
        "num_frames": EXPECTED_FRAMES,
        "total_chunks": EXPECTED_CHUNKS,
    }
    for key, expected in expected_request.items():
        if request.get(key) != expected:
            raise ValueError(
                f"{label}: request {key} must be {expected}, got {request.get(key)}"
            )
    if run.get("warmup") != expected_warmup or run.get("repeat") != expected_repeat:
        raise ValueError(f"{label}: run metadata differs from summary metadata")
    if run.get("measured_iterations") != expected_repeat:
        raise ValueError(f"{label}: measured iteration count metadata is invalid")
    if server.get("role") != "vae":
        raise ValueError(f"{label}: production comparison requires a VAE-only server")
    if server.get("profile_enabled") is not True:
        raise ValueError(f"{label}: server profile timing was not enabled")
    if server.get("trt_layer_profile_enabled") is not False:
        raise ValueError(
            f"{label}: diagnostic TensorRT layer profiling must be disabled"
        )
    if bool(server.get("native_int8_kernel_profile_enabled", False)):
        raise ValueError(
            f"{label}: diagnostic native INT8 kernel profiling must be disabled"
        )
    return context


def _extract_samples(
    summary: Mapping[str, Any],
    *,
    label: str,
    expected_repeat: int,
) -> dict[str, Any]:
    measured = summary.get("measured")
    if not isinstance(measured, list) or len(measured) != expected_repeat:
        raise ValueError(
            f"{label}: expected {expected_repeat} measured iterations, "
            f"got {len(measured) if isinstance(measured, list) else 'invalid'}"
        )
    per_chunk: list[list[float]] = [[] for _ in range(EXPECTED_CHUNKS)]
    whole: list[float] = []
    engine: list[float] = []
    for iteration_index, iteration in enumerate(measured):
        if not isinstance(iteration, dict) or iteration.get("warmup") is not False:
            raise ValueError(
                f"{label}: measured iteration {iteration_index} is invalid"
            )
        if iteration.get("state") != "completed" or iteration.get("error") is not None:
            raise ValueError(f"{label}: measured iteration {iteration_index} failed")
        execution = iteration.get("profile_execution")
        chunks = execution.get("chunks") if isinstance(execution, dict) else None
        if not isinstance(chunks, list) or len(chunks) != EXPECTED_CHUNKS:
            raise ValueError(f"{label}: iteration {iteration_index} has invalid chunks")
        seen: set[int] = set()
        iteration_engine = 0.0
        has_engine = True
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise ValueError(f"{label}: chunk record is invalid")
            chunk_index = chunk.get("chunk_index")
            if (
                not isinstance(chunk_index, int)
                or not 0 <= chunk_index < EXPECTED_CHUNKS
            ):
                raise ValueError(f"{label}: chunk index is invalid")
            if chunk_index in seen:
                raise ValueError(f"{label}: duplicate chunk {chunk_index}")
            seen.add(chunk_index)
            per_chunk[chunk_index].append(
                _finite_ms(
                    chunk.get("chunk_execution_cuda_ms"),
                    label=f"{label} chunk {chunk_index} execution",
                )
            )
            if chunk.get("trt_engine_cuda_ms") is None:
                has_engine = False
            else:
                iteration_engine += _finite_ms(
                    chunk["trt_engine_cuda_ms"],
                    label=f"{label} chunk {chunk_index} TRT engine",
                )
        if seen != set(range(EXPECTED_CHUNKS)):
            raise ValueError(f"{label}: chunk sequence is incomplete")
        whole.append(
            _finite_ms(
                execution.get("vae_execution_cuda_ms"),
                label=f"{label} whole-request execution",
            )
        )
        if has_engine:
            engine.append(iteration_engine)
    result = {
        "per_chunk": [_stats(values) for values in per_chunk],
        "initial": _stats(per_chunk[0]),
        "steady_pooled": _stats(
            [value for values in per_chunk[1:] for value in values]
        ),
        "whole_request": _stats(whole),
    }
    if engine:
        result["trt_engine"] = _stats(engine)
    return result


def _speedup(reference_ms: float, candidate_ms: float) -> float:
    if candidate_ms <= 0:
        raise ValueError("candidate latency must be positive")
    return reference_ms / candidate_ms


def _extract_native_kernel_profile(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate diagnostic native-plugin stages without calling them production."""

    context = summary.get("measurement_context")
    server = context.get("server") if isinstance(context, Mapping) else None
    if (
        not isinstance(server, Mapping)
        or server.get("vae_trt_variant") not in {"native_int8_v1", "native_int8_v2"}
        or server.get("native_int8_kernel_profile_enabled") is not True
    ):
        raise ValueError(
            "native kernel-profile summary must come from an instrumented "
            "native_int8_v1 or native_int8_v2 server"
        )
    measured = summary.get("measured")
    if not isinstance(measured, list) or not measured:
        raise ValueError("native kernel-profile summary has no measured iterations")
    per_chunk = [
        {component: [] for component in NATIVE_PROFILE_COMPONENTS}
        for _ in range(EXPECTED_CHUNKS)
    ]
    whole = {component: [] for component in NATIVE_PROFILE_COMPONENTS}
    for iteration in measured:
        execution = iteration.get("profile_execution")
        chunks = execution.get("chunks") if isinstance(execution, Mapping) else None
        if not isinstance(chunks, list) or len(chunks) != EXPECTED_CHUNKS:
            raise ValueError("native kernel-profile iteration has invalid chunks")
        request_values = {component: 0.0 for component in NATIVE_PROFILE_COMPONENTS}
        for chunk in chunks:
            index = chunk.get("chunk_index")
            profile = chunk.get("native_int8_kernel_profile")
            stages = profile.get("stage_totals_ms") if isinstance(profile, Mapping) else None
            if not isinstance(index, int) or not 0 <= index < EXPECTED_CHUNKS:
                raise ValueError("native kernel-profile chunk index is invalid")
            if not isinstance(stages, Mapping):
                raise ValueError(f"native chunk {index} has no plugin stage totals")
            trt_ms = _finite_ms(
                chunk.get("trt_engine_cuda_ms"), label=f"native chunk {index} engine"
            )
            execution_ms = _finite_ms(
                chunk.get("chunk_execution_cuda_ms"),
                label=f"native chunk {index} execution",
            )
            native_total = _finite_ms(
                stages.get("residual_block_total_ms"),
                label=f"native chunk {index} residual blocks",
            )
            def stage(*names: str) -> float:
                for name in names:
                    if name in stages:
                        return _finite_ms(
                            stages[name], label=f"native chunk {index} {name}"
                        )
                return 0.0

            values = {
                "target_int8_conv_ms": stage(
                    "conv1_mainloop_epilogue_ms",
                    "direct_causal_iterator_conv1_ms",
                    "persistent_conv1_ms",
                )
                + stage(
                    "conv2_mainloop_ms",
                    "conv2_mma_and_fused_residual_epilogue_ms",
                    "direct_causal_iterator_conv2_residual_ms",
                    "persistent_conv2_residual_ms",
                ),
                "norm_silu_quant_cache_write_ms": stage(
                    "entry_norm_silu_quant_cache_write_ms",
                    "entry_norm_silu_quant_cache_ms",
                    "persistent_entry_ms",
                )
                + stage(
                    "mid_norm_silu_quant_cache_write_ms",
                    "mid_norm_silu_quant_cache_ms",
                    "persistent_mid_norm_silu_quant_ms",
                ),
                "residual_epilogue_ms": stage(
                    "residual_epilogue_ms",
                    "fused_dequant_bias_residual_requant_ms",
                ),
                "group_exit_ms": stage(
                    "group_exit_ms",
                    "persistent_group_exit_ms",
                ),
                "native_residual_blocks_ms": native_total,
                "remaining_trt_operators_ms": max(0.0, trt_ms - native_total),
                "engine_external_ms": max(0.0, execution_ms - trt_ms),
            }
            for component, value in values.items():
                per_chunk[index][component].append(value)
                request_values[component] += value
        for component, value in request_values.items():
            whole[component].append(value)
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "variant": server["vae_trt_variant"],
        "native_int8_v2_level": server.get("native_int8_v2_level"),
        "measured_iteration_count": len(measured),
        "per_chunk": [
            {component: _stats(values) for component, values in chunk.items()}
            for chunk in per_chunk
        ],
        "whole_request": {
            component: _stats(values) for component, values in whole.items()
        },
        "attribution_limit": (
            "cache write is fused into the producer kernels and is therefore "
            "reported together with Norm/SiLU/static quantization"
        ),
    }


def compare_profile_summaries(
    inputs: Mapping[str, tuple[str | Path, Mapping[str, Any]]],
    *,
    expected_warmup: int = EXPECTED_WARMUP,
    expected_repeat: int = EXPECTED_REPEAT,
) -> dict[str, Any]:
    required = {"fp32", "fp16_trt", "int8_v5"}
    allowed = {
        *required,
        "int8_fusion_v1",
        "int8_fusion_v2",
        "native_int8_v1",
        "native_int8_v2_p1",
        "native_int8_v2_p2",
        "native_int8_v2_p3",
    }
    if not required.issubset(inputs):
        raise ValueError(f"missing comparison inputs: {sorted(required - set(inputs))}")
    if not set(inputs).issubset(allowed):
        raise ValueError(f"unknown comparison inputs: {sorted(set(inputs) - allowed)}")
    contexts: dict[str, dict[str, Any]] = {}
    results: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for label, (source, summary) in inputs.items():
        path = Path(source).expanduser().resolve()
        contexts[label] = _require_context(
            summary,
            label=label,
            expected_warmup=expected_warmup,
            expected_repeat=expected_repeat,
        )
        results[label] = _extract_samples(
            summary,
            label=label,
            expected_repeat=expected_repeat,
        )
        artifacts[label] = {"path": str(path), "sha256": _sha256_file(path)}

    canonical_request = contexts["fp32"]["request"]
    for label, context in contexts.items():
        if context["request"] != canonical_request:
            raise ValueError(f"{label}: request context is not comparable")
    all_server_contexts = [context["server"] for context in contexts.values()]
    for key in (
        "vae_runtime_gpu_name",
        "vae_runtime_compute_capability",
        "vae_runtime_cuda_version",
    ):
        values = {
            json.dumps(context.get(key), sort_keys=True)
            for context in all_server_contexts
        }
        if len(values) != 1 or next(iter(values)) in {"null", '""'}:
            raise ValueError(f"comparison inputs disagree on {key}")
    trt_contexts = [contexts[label]["server"] for label in inputs if label != "fp32"]
    for key in ("vae_engine_sm", "vae_engine_tensorrt_version"):
        values = {
            json.dumps(context.get(key), sort_keys=True) for context in trt_contexts
        }
        if len(values) != 1 or next(iter(values)) in {"null", '""'}:
            raise ValueError(f"TensorRT inputs disagree on {key}")
    expected_backends = {
        "fp32": ("pytorch", "fp32"),
        "fp16_trt": ("tensorrt", "fp16"),
        "int8_v5": ("tensorrt", "int8"),
        "int8_fusion_v1": ("tensorrt", "int8"),
        "int8_fusion_v2": ("tensorrt", "int8"),
        "native_int8_v1": ("tensorrt", "int8"),
        "native_int8_v2_p1": ("tensorrt", "int8"),
        "native_int8_v2_p2": ("tensorrt", "int8"),
        "native_int8_v2_p3": ("tensorrt", "int8"),
    }
    for label, context in contexts.items():
        expected_backend, expected_precision = expected_backends[label]
        actual = (
            context["server"].get("vae_backend"),
            context["server"].get("vae_engine_precision"),
        )
        if actual != (expected_backend, expected_precision):
            raise ValueError(
                f"{label}: expected backend/precision "
                f"{expected_backend}/{expected_precision}, got {actual[0]}/{actual[1]}"
            )
    for label in ("fp16_trt", "int8_v5"):
        server = contexts[label]["server"]
        if server.get("vae_trt_variant", "baseline") != "baseline":
            raise ValueError(f"{label}: expected baseline TensorRT variant")
        plan_sha = server.get("vae_engine_plan_sha256")
        if not isinstance(plan_sha, dict) or set(plan_sha) != {"initial", "steady"}:
            raise ValueError(f"{label}: missing initial/steady plan SHA metadata")
    if "int8_fusion_v1" in contexts:
        server = contexts["int8_fusion_v1"]["server"]
        if server.get("vae_trt_variant") != "fusion_v1":
            raise ValueError("int8_fusion_v1 input does not use fusion_v1")
        if not server.get("vae_trt_plugin_sha256"):
            raise ValueError("int8_fusion_v1 input has no plugin SHA")
    if "int8_fusion_v2" in contexts:
        server = contexts["int8_fusion_v2"]["server"]
        if server.get("vae_trt_variant") != "fusion_v2":
            raise ValueError("int8_fusion_v2 input does not use fusion_v2")
        if not server.get("vae_trt_plugin_sha256"):
            raise ValueError("int8_fusion_v2 input has no plugin SHA")
        if server.get("vae_trt_initial_variant") != "raw_int8_v5":
            raise ValueError("int8_fusion_v2 did not use raw v5 for chunk 0")
        if server.get("vae_trt_steady_variant") != "fusion_v2":
            raise ValueError("int8_fusion_v2 did not use fusion_v2 for chunks 1-6")
        if server.get("vae_cache_selected_int8_slot_count") != 6:
            raise ValueError("int8_fusion_v2 did not use the six-slot INT8 cache ABI")
        if server.get("vae_cache_migration") != "one_time_after_chunk_0":
            raise ValueError("int8_fusion_v2 cache migration contract is missing")
    if "native_int8_v1" in contexts:
        server = contexts["native_int8_v1"]["server"]
        if server.get("vae_trt_variant") != "native_int8_v1":
            raise ValueError("native_int8_v1 input uses a different variant")
        required_native = {
            "native_int8_audit_passed": True,
            "native_int8_target_logical_conv_count": 28,
            "native_int8_initial_call_site_count": 84,
            "native_int8_steady_call_site_count": 84,
            "native_int8_signature_count": 9,
            "native_int8_cache_int8_slot_count": 28,
            "native_int8_cache_fp16_slot_count": 4,
            "native_int8_runtime_scale_mode": "static",
            "native_int8_weight_mode": "offline_per_channel_packed",
            "native_int8_algorithm": (
                "temporal_folded_cutlass_conv2d_implicit_gemm"
            ),
            "native_int8_weight_layout": "KRSTC_FLAT",
            "vae_cache_migration": "none",
        }
        for key, expected in required_native.items():
            if server.get(key) != expected:
                raise ValueError(
                    f"native_int8_v1 {key} must be {expected!r}, "
                    f"got {server.get(key)!r}"
                )
        if not server.get("native_int8_plugin_sha256"):
            raise ValueError("native_int8_v1 input has no plugin SHA")
        cutlass_commit = server.get("native_int8_cutlass_commit")
        if not isinstance(cutlass_commit, str) or len(cutlass_commit) != 40:
            raise ValueError("native_int8_v1 input has no pinned CUTLASS commit")
    for level in ("p1", "p2", "p3"):
        label = f"native_int8_v2_{level}"
        if label not in contexts:
            continue
        server = contexts[label]["server"]
        if server.get("vae_trt_variant") != "native_int8_v2":
            raise ValueError(f"{label} input uses a different variant")
        if server.get("native_int8_v2_level") != level:
            raise ValueError(f"{label} input uses a different V2 level")
        if server.get("native_int8_v2_audit_passed") is not True:
            raise ValueError(f"{label} audit did not pass")
        if server.get("native_int8_v2_accumulator2_workspace_bytes") != 0:
            raise ValueError(f"{label} retained Conv2 accumulator workspace")
        if level in {"p2", "p3"}:
            if server.get("native_int8_v2_temporal_window_bytes") != 0:
                raise ValueError(f"{label} retained temporal-window workspace")
            if server.get("native_int8_v2_direct_causal_iterator") is not True:
                raise ValueError(f"{label} did not enable direct causal iterator")
        if level == "p3":
            if server.get("native_int8_v2_accumulator1_workspace_bytes") != 0:
                raise ValueError(f"{label} retained Conv1 accumulator workspace")
            if server.get("native_int8_v2_persistent_block_count") != 84:
                raise ValueError(f"{label} persistent coverage is incomplete")
        if not server.get("native_int8_v2_plugin_sha256"):
            raise ValueError(f"{label} has no plugin SHA")

    fp32_total = results["fp32"]["whole_request"]["mean_ms"]
    fp16_total = results["fp16_trt"]["whole_request"]["mean_ms"]
    for label, value in results.items():
        total = value["whole_request"]["mean_ms"]
        value["speedup_vs_fp32"] = _speedup(fp32_total, total)
        value["speedup_vs_fp16_trt"] = _speedup(fp16_total, total)

    return {
        "schema_version": 1,
        "comparison_kind": "sfwan_vae_production",
        "measurement_context": {
            "request": canonical_request,
            "warmup": expected_warmup,
            "repeat": expected_repeat,
            "trt_layer_profile_enabled": False,
        },
        "artifacts": artifacts,
        "results": results,
    }


def render_markdown(comparison: Mapping[str, Any]) -> str:
    results = comparison["results"]
    ordered = [
        label
        for label in (
            "fp32",
            "fp16_trt",
            "int8_v5",
            "int8_fusion_v1",
            "int8_fusion_v2",
            "native_int8_v1",
            "native_int8_v2_p1",
            "native_int8_v2_p2",
            "native_int8_v2_p3",
        )
        if label in results
    ]
    columns = ["precision/variant", *[f"chunk {index}" for index in range(7)]]
    columns.extend(["whole request", "vs FP32", "vs FP16 TRT"])
    lines = [
        "# SFWan VAE production latency comparison",
        "",
        "All latency values are measured CUDA execution time in milliseconds; "
        "warmup samples are excluded.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |",
    ]
    for label in ordered:
        value = results[label]
        cells = [label]
        cells.extend(f"{item['mean_ms']:.3f}" for item in value["per_chunk"])
        cells.extend(
            [
                f"{value['whole_request']['mean_ms']:.3f}",
                f"{value['speedup_vs_fp32']:.3f}x",
                f"{value['speedup_vs_fp16_trt']:.3f}x",
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Statistical summary",
            "",
            "| variant | initial mean +/- sigma (ms) | "
            "steady pooled mean +/- sigma (ms) | samples |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for label in ordered:
        value = results[label]
        initial = value["initial"]
        steady = value["steady_pooled"]
        lines.append(
            f"| {label} | {initial['mean_ms']:.3f} +/- "
            f"{initial['population_stddev_ms']:.3f} | "
            f"{steady['mean_ms']:.3f} +/- "
            f"{steady['population_stddev_ms']:.3f} | "
            f"{value['whole_request']['sample_count']} requests |"
        )
    native_profiles = comparison.get("native_int8_kernel_profiles")
    if not isinstance(native_profiles, Mapping):
        legacy = comparison.get("native_int8_kernel_profile")
        native_profiles = {"native_int8": legacy} if isinstance(legacy, Mapping) else {}
    if native_profiles:
        lines.extend(
            [
                "",
                "## Native INT8 diagnostic component attribution",
                "",
                "These values come from the opt-in instrumented plugin and are not "
                "production latency. Cache writes are fused with Norm/SiLU/quant.",
                "",
                "| variant | component | 7-chunk mean (ms) | population sigma (ms) |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for label, native_profile in native_profiles.items():
            for component in NATIVE_PROFILE_COMPONENTS:
                stats = native_profile["whole_request"][component]
                lines.append(
                    f"| {label} | {component} | {stats['mean_ms']:.3f} | "
                    f"{stats['population_stddev_ms']:.3f} |"
                )
    return "\n".join(lines) + "\n"


def _write_text(path: str | Path, value: str) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(target)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-json", required=True)
    parser.add_argument("--fp16-trt-json", required=True)
    parser.add_argument("--int8-v5-json", required=True)
    parser.add_argument("--int8-fusion-v1-json")
    parser.add_argument("--int8-fusion-v2-json")
    parser.add_argument("--native-int8-v1-json")
    parser.add_argument("--native-int8-v2-p1-json")
    parser.add_argument("--native-int8-v2-p2-json")
    parser.add_argument("--native-int8-v2-p3-json")
    parser.add_argument("--native-int8-kernel-profile-json")
    parser.add_argument("--native-int8-v1-kernel-profile-json")
    parser.add_argument("--native-int8-v2-p1-kernel-profile-json")
    parser.add_argument("--native-int8-v2-p2-kernel-profile-json")
    parser.add_argument("--native-int8-v2-p3-kernel-profile-json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--expected-warmup", type=int, default=EXPECTED_WARMUP)
    parser.add_argument("--expected-repeat", type=int, default=EXPECTED_REPEAT)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    sources = {
        "fp32": args.fp32_json,
        "fp16_trt": args.fp16_trt_json,
        "int8_v5": args.int8_v5_json,
    }
    if args.int8_fusion_v1_json:
        sources["int8_fusion_v1"] = args.int8_fusion_v1_json
    if args.int8_fusion_v2_json:
        sources["int8_fusion_v2"] = args.int8_fusion_v2_json
    if args.native_int8_v1_json:
        sources["native_int8_v1"] = args.native_int8_v1_json
    if args.native_int8_v2_p1_json:
        sources["native_int8_v2_p1"] = args.native_int8_v2_p1_json
    if args.native_int8_v2_p2_json:
        sources["native_int8_v2_p2"] = args.native_int8_v2_p2_json
    if args.native_int8_v2_p3_json:
        sources["native_int8_v2_p3"] = args.native_int8_v2_p3_json
    inputs = {label: (path, _read_json(path)) for label, path in sources.items()}
    comparison = compare_profile_summaries(
        inputs,
        expected_warmup=args.expected_warmup,
        expected_repeat=args.expected_repeat,
    )
    if args.native_int8_kernel_profile_json:
        profile_path = Path(args.native_int8_kernel_profile_json).expanduser().resolve()
        comparison["native_int8_kernel_profile"] = _extract_native_kernel_profile(
            _read_json(profile_path)
        )
        comparison["artifacts"]["native_int8_kernel_profile"] = {
            "path": str(profile_path),
            "sha256": _sha256_file(profile_path),
        }
    profile_sources = {
        "native_int8_v1": args.native_int8_v1_kernel_profile_json,
        "native_int8_v2_p1": args.native_int8_v2_p1_kernel_profile_json,
        "native_int8_v2_p2": args.native_int8_v2_p2_kernel_profile_json,
        "native_int8_v2_p3": args.native_int8_v2_p3_kernel_profile_json,
    }
    native_profiles: dict[str, Any] = {}
    for label, source in profile_sources.items():
        if not source:
            continue
        profile_path = Path(source).expanduser().resolve()
        profile = _extract_native_kernel_profile(_read_json(profile_path))
        expected_variant = "native_int8_v1" if label == "native_int8_v1" else "native_int8_v2"
        expected_level = label.removeprefix("native_int8_v2_") if expected_variant == "native_int8_v2" else None
        if profile["variant"] != expected_variant or profile["native_int8_v2_level"] != expected_level:
            raise ValueError(f"{label} kernel profile identity does not match its CLI argument")
        native_profiles[label] = profile
        comparison["artifacts"][f"{label}_kernel_profile"] = {
            "path": str(profile_path),
            "sha256": _sha256_file(profile_path),
        }
    if native_profiles:
        comparison["native_int8_kernel_profiles"] = native_profiles
    _write_text(
        args.output_json,
        json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    markdown = render_markdown(comparison)
    _write_text(args.output_markdown, markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
