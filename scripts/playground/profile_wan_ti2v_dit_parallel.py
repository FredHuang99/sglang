#!/usr/bin/env python
"""Profile Wan2.2-TI2V-5B DiT under TP/FSDP/offload configurations.

This helper is intentionally independent from profile_wan_ti2v_stages.py so it
does not interfere with the SP-focused workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams  # noqa: E402
from sglang.multimodal_gen.registry import get_model_info  # noqa: E402
from sglang.multimodal_gen.runtime.distributed import (  # noqa: E402
    cleanup_dist_env_and_memory,
    get_local_torch_device,
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request  # noqa: E402
from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (  # noqa: E402
    SyncExecutor,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs  # noqa: E402
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (  # noqa: E402
    maybe_download_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile Wan2.2-TI2V-5B DiT TP/FSDP/offload modes"
    )
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["tp", "fsdp", "dit-cpu-offload", "dit-layerwise-offload"],
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--hsdp-replicate-dim", type=int, default=1)
    parser.add_argument("--hsdp-shard-dim", type=int, default=None)
    parser.add_argument("--attention-backend", type=str, default=None)
    parser.add_argument("--dit-offload-prefetch-size", type=float, default=0.0)
    parser.add_argument(
        "--sync-stage-profiling",
        choices=["off", "denoising", "all"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cinematic science-fiction city with reflective rain streets.",
    )
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-txt", type=str, default=None)
    parser.add_argument("--disable-autocast", action="store_true", default=False)
    parser.add_argument(
        "--list-legal-tp-sizes",
        action="store_true",
        default=False,
        help="Print legal TP sizes for this model and exit.",
    )
    return parser.parse_args()


def is_rank_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def validate_args(args: argparse.Namespace) -> None:
    if args.num_gpus < 1:
        raise ValueError(f"--num-gpus must be >= 1, got {args.num_gpus}")
    if args.warmup_iters < 0:
        raise ValueError(
            f"--warmup-iters must be >= 0, got {args.warmup_iters}"
        )
    if args.profile_iters < 1:
        raise ValueError(
            f"--profile-iters must be >= 1, got {args.profile_iters}"
        )
    if args.list_legal_tp_sizes and args.mode != "tp":
        raise ValueError("--list-legal-tp-sizes can only be used with --mode tp.")
    if args.mode == "tp" and args.num_gpus < 1:
        raise ValueError("TP profiling requires at least 1 GPU.")
    if args.mode == "fsdp" and args.num_gpus < 2:
        raise ValueError("FSDP profiling requires at least 2 GPUs.")
    if args.mode in ("dit-cpu-offload", "dit-layerwise-offload") and args.num_gpus != 1:
        raise ValueError(f"{args.mode} is profiled as a single-GPU mode; use --num-gpus 1.")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reduce_max_scalar(value: float, device: torch.device) -> float:
    if not dist.is_initialized():
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def positive_divisors(value: int) -> list[int]:
    divisors = set()
    for candidate in range(1, int(value**0.5) + 1):
        if value % candidate == 0:
            divisors.add(candidate)
            divisors.add(value // candidate)
    return sorted(divisors)


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    json_path = Path(args.output_json).resolve() if args.output_json else None
    txt_path = Path(args.output_txt).resolve() if args.output_txt else None

    if json_path is None and txt_path is None:
        return None, None

    if json_path is None and txt_path is not None:
        json_path = txt_path.with_suffix(".json")
    if txt_path is None and json_path is not None:
        txt_path = json_path.with_suffix(".txt")

    return json_path, txt_path


def write_outputs(
    result: dict[str, Any],
    summary: str,
    json_path: Path | None,
    txt_path: Path | None,
) -> None:
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if txt_path is not None:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(summary + "\n", encoding="utf-8")


def print_and_persist(
    result: dict[str, Any],
    summary: str,
    json_path: Path | None,
    txt_path: Path | None,
) -> None:
    if not is_rank_zero():
        return
    print(summary)
    if json_path is not None:
        print(f"JSON report: {json_path}")
    if txt_path is not None:
        print(f"Text report: {txt_path}")
    write_outputs(result, summary, json_path, txt_path)


def extract_output_shape(output: Any) -> list[int] | None:
    if isinstance(output, torch.Tensor):
        return list(output.shape)
    if isinstance(output, (list, tuple)) and output:
        first_item = output[0]
        if isinstance(first_item, torch.Tensor):
            return list(first_item.shape)
    return None


def get_stage_duration_ms(stage_timings: dict[str, float]) -> float:
    if "DenoisingStage" in stage_timings:
        return float(stage_timings["DenoisingStage"])
    if "denoising_stage" in stage_timings:
        return float(stage_timings["denoising_stage"])
    for name, duration in stage_timings.items():
        if "denoising" in name.lower():
            return float(duration)
    raise KeyError(
        "Unable to find a denoising stage in timings. "
        f"Available stages: {sorted(stage_timings)}"
    )


def resolve_model_path(model_path: str) -> str:
    return maybe_download_model(model_path)


def get_dit_arch(server_args: ServerArgs):
    return server_args.pipeline_config.dit_config.arch_config


def get_legal_tp_sizes(server_args: ServerArgs, max_gpus: int) -> list[int]:
    arch = get_dit_arch(server_args)
    common_sizes = set(positive_divisors(arch.num_attention_heads))
    common_sizes &= set(positive_divisors(arch.hidden_size))
    common_sizes &= set(positive_divisors(arch.ffn_dim))
    return sorted(size for size in common_sizes if size <= max_gpus)


def validate_tp_mode(server_args: ServerArgs) -> list[int]:
    tp_size = server_args.tp_size
    legal_sizes = get_legal_tp_sizes(server_args, server_args.num_gpus)
    if tp_size not in legal_sizes:
        raise ValueError(
            f"Illegal TP size {tp_size} for {server_args.model_path}. "
            f"Legal TP sizes up to {server_args.num_gpus} GPU(s): {legal_sizes}"
        )
    if server_args.num_gpus != tp_size:
        raise ValueError(
            "This helper profiles pure TP runs only; require num_gpus == tp_size."
        )
    return legal_sizes


def build_server_args(
    args: argparse.Namespace,
) -> tuple[ServerArgs, list[int] | None]:
    common_kwargs = dict(
        model_path=args.model_path,
        attention_backend=args.attention_backend,
        disable_autocast=args.disable_autocast,
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        warmup=False,
        sp_degree=1,
        ulysses_degree=1,
        ring_degree=1,
    )

    if args.mode == "tp":
        server_args = ServerArgs.from_kwargs(
            **common_kwargs,
            num_gpus=args.num_gpus,
            tp_size=args.tp_size or args.num_gpus,
            use_fsdp_inference=False,
            dit_cpu_offload=False,
            dit_layerwise_offload=False,
        )
        legal_tp_sizes = validate_tp_mode(server_args)
    elif args.mode == "fsdp":
        server_args = ServerArgs.from_kwargs(
            **common_kwargs,
            num_gpus=args.num_gpus,
            tp_size=1,
            use_fsdp_inference=True,
            hsdp_replicate_dim=args.hsdp_replicate_dim,
            hsdp_shard_dim=args.hsdp_shard_dim or args.num_gpus,
            dit_cpu_offload=False,
            dit_layerwise_offload=False,
        )
        legal_tp_sizes = None
    elif args.mode == "dit-cpu-offload":
        server_args = ServerArgs.from_kwargs(
            **common_kwargs,
            num_gpus=1,
            tp_size=1,
            use_fsdp_inference=False,
            dit_cpu_offload=True,
            dit_layerwise_offload=False,
        )
        legal_tp_sizes = None
    else:
        server_args = ServerArgs.from_kwargs(
            **common_kwargs,
            num_gpus=1,
            tp_size=1,
            use_fsdp_inference=False,
            dit_cpu_offload=False,
            dit_layerwise_offload=True,
            dit_offload_prefetch_size=args.dit_offload_prefetch_size,
        )
        legal_tp_sizes = None

    server_args.pipeline_config.vae_config.load_encoder = False
    server_args.pipeline_config.vae_config.load_decoder = True
    return server_args, legal_tp_sizes


def maybe_init_runtime_distributed(server_args: ServerArgs) -> None:
    if server_args.num_gpus <= 1:
        return
    if int(os.environ.get("WORLD_SIZE", "1")) != server_args.num_gpus:
        raise RuntimeError(
            "Multi-GPU profiling expects torchrun to provide WORLD_SIZE equal to "
            f"--num-gpus ({server_args.num_gpus})."
        )
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=server_args.tp_size,
        sp_size=server_args.sp_degree,
        enable_cfg_parallel=server_args.enable_cfg_parallel,
        ulysses_degree=server_args.ulysses_degree,
        ring_degree=server_args.ring_degree,
        dp_size=server_args.dp_size,
        distributed_init_method="env://",
        dist_timeout=server_args.dist_timeout,
    )


def summarize_result(result: dict[str, Any]) -> str:
    lines = [
        "DiT parallel profiling summary:",
        f"  mode: {result['mode']}",
        f"  num_gpus: {result['num_gpus']}",
        f"  timing_ms: avg={result['timing_ms']['avg']:.2f}, "
        f"min={result['timing_ms']['min']:.2f}, max={result['timing_ms']['max']:.2f}",
    ]
    lines.append(f"  parallelism: {result['parallelism']}")
    if result.get("offload"):
        lines.append(f"  offload: {result['offload']}")
    if result.get("output_shape") is not None:
        lines.append(f"  output_shape: {result['output_shape']}")
    peak_memory_mb = result.get("peak_memory_mb")
    if peak_memory_mb is not None:
        lines.append(f"  peak_memory_mb: {peak_memory_mb:.2f}")
    denoise = result.get("denoise_steps_ms")
    if denoise:
        lines.append(
            "  denoise_steps_ms: "
            f"count={denoise['count']}, avg={denoise['avg']:.2f}, "
            f"min={denoise['min']:.2f}, max={denoise['max']:.2f}"
        )
    if result.get("legal_tp_sizes"):
        lines.append(f"  legal_tp_sizes: {result['legal_tp_sizes']}")
    return "\n".join(lines)


def finalize_result(
    *,
    args: argparse.Namespace,
    server_args: ServerArgs,
    iteration_timings_ms: list[float],
    output_shape: list[int] | None,
    peak_memory_mb: float | None,
    denoise_iteration_avgs: list[float],
    denoise_iteration_mins: list[float],
    denoise_iteration_maxs: list[float],
    legal_tp_sizes: list[int] | None,
) -> dict[str, Any]:
    return {
        "mode": args.mode,
        "model_path": server_args.model_path,
        "num_gpus": server_args.num_gpus,
        "parallelism": {
            "tp_size": server_args.tp_size,
            "sp_degree": server_args.sp_degree,
            "hsdp_replicate_dim": server_args.hsdp_replicate_dim,
            "hsdp_shard_dim": server_args.hsdp_shard_dim,
        },
        "offload": {
            "use_fsdp_inference": server_args.use_fsdp_inference,
            "dit_cpu_offload": server_args.dit_cpu_offload,
            "dit_layerwise_offload": server_args.dit_layerwise_offload,
            "dit_offload_prefetch_size": server_args.dit_offload_prefetch_size,
        },
        "workload": {
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "seed": args.seed,
        },
        "iterations_ms": iteration_timings_ms,
        "timing_ms": {
            "avg": mean(iteration_timings_ms),
            "min": min(iteration_timings_ms),
            "max": max(iteration_timings_ms),
        },
        "denoise_steps_ms": {
            "count": args.num_inference_steps,
            "avg": mean(denoise_iteration_avgs) if denoise_iteration_avgs else 0.0,
            "min": mean(denoise_iteration_mins) if denoise_iteration_mins else 0.0,
            "max": mean(denoise_iteration_maxs) if denoise_iteration_maxs else 0.0,
        },
        "output_shape": output_shape,
        "peak_memory_mb": peak_memory_mb,
        "legal_tp_sizes": legal_tp_sizes,
    }


def maybe_collect_peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def emit_legal_tp_report(args: argparse.Namespace) -> int:
    server_args, legal_tp_sizes = build_server_args(args)
    payload = {
        "model_path": server_args.model_path,
        "available_gpu_budget": args.num_gpus,
        "legal_tp_sizes": legal_tp_sizes,
        "dit_arch": {
            "num_attention_heads": get_dit_arch(server_args).num_attention_heads,
            "hidden_size": get_dit_arch(server_args).hidden_size,
            "ffn_dim": get_dit_arch(server_args).ffn_dim,
        },
    }
    if is_rank_zero():
        print(json.dumps(payload, indent=2))
    return 0


def profile_dit_parallel_mode(
    args: argparse.Namespace,
    server_args: ServerArgs,
    legal_tp_sizes: list[int] | None,
) -> dict[str, Any]:
    model_info = get_model_info(server_args.model_path, backend=server_args.backend)
    pipeline = model_info.pipeline_cls(
        server_args.model_path,
        server_args,
        executor=SyncExecutor(server_args=server_args),
    )

    iteration_timings_ms: list[float] = []
    denoise_iteration_avgs: list[float] = []
    denoise_iteration_mins: list[float] = []
    denoise_iteration_maxs: list[float] = []
    output_shape: list[int] | None = None
    device = get_local_torch_device()

    for iteration in range(args.warmup_iters + args.profile_iters):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        sampling_params = SamplingParams.from_user_sampling_params_args(
            server_args.model_path,
            server_args=server_args,
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed + iteration,
            num_inference_steps=args.num_inference_steps,
            perf_dump_path="inline_profile.json",
            enable_sequence_shard=False,
            save_output=False,
        )
        req = prepare_request(server_args=server_args, sampling_params=sampling_params)
        req.suppress_logs = True

        synchronize_device(device)
        start = time.perf_counter()
        output_batch = pipeline.forward(req, server_args)
        synchronize_device(device)
        output_batch.timings.total_duration_ms = (time.perf_counter() - start) * 1000.0

        stage_duration_ms = get_stage_duration_ms(output_batch.timings.stages)
        stage_duration_ms = reduce_max_scalar(stage_duration_ms, device)

        denoise_steps = list(output_batch.timings.steps)
        if denoise_steps:
            avg_step_ms = reduce_max_scalar(mean(denoise_steps), device)
            min_step_ms = reduce_max_scalar(min(denoise_steps), device)
            max_step_ms = reduce_max_scalar(max(denoise_steps), device)
        else:
            avg_step_ms = min_step_ms = max_step_ms = 0.0

        if iteration >= args.warmup_iters:
            iteration_timings_ms.append(stage_duration_ms)
            denoise_iteration_avgs.append(avg_step_ms)
            denoise_iteration_mins.append(min_step_ms)
            denoise_iteration_maxs.append(max_step_ms)

        if output_shape is None and output_batch.output is not None:
            output_shape = extract_output_shape(output_batch.output)

    return finalize_result(
        args=args,
        server_args=server_args,
        iteration_timings_ms=iteration_timings_ms,
        output_shape=output_shape,
        peak_memory_mb=maybe_collect_peak_memory_mb(device),
        denoise_iteration_avgs=denoise_iteration_avgs,
        denoise_iteration_mins=denoise_iteration_mins,
        denoise_iteration_maxs=denoise_iteration_maxs,
        legal_tp_sizes=legal_tp_sizes,
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    os.environ["SGLANG_DIFFUSION_SYNC_STAGE_PROFILING"] = args.sync_stage_profiling

    server_args, legal_tp_sizes = build_server_args(args)
    if args.list_legal_tp_sizes:
        return emit_legal_tp_report(args)

    server_args.model_path = resolve_model_path(server_args.model_path)
    json_path, txt_path = resolve_output_paths(args)

    try:
        maybe_init_runtime_distributed(server_args)
        result = profile_dit_parallel_mode(args, server_args, legal_tp_sizes)
    finally:
        if dist.is_initialized():
            cleanup_dist_env_and_memory()

    summary = summarize_result(result)
    print_and_persist(result, summary, json_path, txt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
