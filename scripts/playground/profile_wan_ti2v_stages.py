#!/usr/bin/env python
"""Profile Wan2.2-TI2V-5B stages with reproducible defaults.

For multi-GPU runs, launch this helper with `torchrun --standalone --nproc_per_node N`.
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

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.distributed import (  # noqa: E402
    cleanup_dist_env_and_memory,
    get_local_torch_device,
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request  # noqa: E402
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (  # noqa: E402
    PipelineComponentLoader,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (  # noqa: E402
    SyncExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (  # noqa: E402
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args  # noqa: E402
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (  # noqa: E402
    maybe_download_model,
    verify_model_config_and_directory,
)
from sglang.multimodal_gen.runtime.utils.profile_matrix import (  # noqa: E402
    build_ulysses_ring_pairs,
    validate_sp_topology,
)
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Wan2.2-TI2V-5B stages")
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["encoder", "dit", "vae-encode", "vae-decode"],
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--sp-degree", type=int, default=None)
    parser.add_argument("--ulysses-degree", type=int, default=None)
    parser.add_argument("--ring-degree", type=int, default=None)
    parser.add_argument(
        "--encoder-parallel-mode",
        choices=["tp"],
        default="tp",
    )
    parser.add_argument("--attention-backend", type=str, default=None)
    parser.add_argument(
        "--sync-stage-profiling",
        choices=["off", "denoising", "all"],
        default="off",
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
    parser.add_argument(
        "--vae-encode-input-frames",
        type=int,
        default=1,
        help="Synthetic frame count used only for --stage vae-encode. "
        "For TI2V/I2V tasks, this defaults to 1 regardless of this value.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-txt", type=str, default=None)
    parser.add_argument("--disable-autocast", action="store_true", default=False)
    parser.add_argument(
        "--list-legal-dit-combos",
        action="store_true",
        default=False,
        help="Print legal (ulysses, ring) pairs for the requested SP degree and exit.",
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
    if args.stage == "dit" and args.device != "cuda":
        raise ValueError("DiT stage profiling is only supported on CUDA devices.")
    if args.stage == "dit" and (args.tp_size not in (None, 1)):
        raise ValueError(
            "DiT profiling matrix is limited to SP-only runs; use tp-size=1."
        )
    if args.device == "cpu" and args.num_gpus != 1:
        raise ValueError("Pure CPU profiling only supports --num-gpus 1.")
    if args.list_legal_dit_combos and args.stage != "dit":
        raise ValueError("--list-legal-dit-combos can only be used with --stage dit.")
    if args.num_frames < 1:
        raise ValueError(f"--num-frames must be >= 1, got {args.num_frames}")
    if args.vae_encode_input_frames < 1:
        raise ValueError(f"--vae-encode-input-frames must be >= 1, got {args.vae_encode_input_frames}")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reduce_max_scalar(value: float, device: torch.device) -> float:
    if not dist.is_initialized():
        return value
    tensor = torch.tensor(
        [value],
        dtype=torch.float64,
        device=device if device.type == "cuda" else torch.device("cpu"),
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def stage_device_from_arg(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    return get_local_torch_device()


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


def resolve_model_path(model_path: str) -> tuple[str, dict[str, Any]]:
    resolved_model_path = maybe_download_model(model_path)
    model_index = verify_model_config_and_directory(resolved_model_path)
    return resolved_model_path, dict(model_index)


def load_requested_components(
    *,
    model_path: str,
    model_index: dict[str, Any],
    component_names: list[str],
    server_args: ServerArgs,
) -> dict[str, Any]:
    filtered_index = {
        key: value
        for key, value in model_index.items()
        if key
        not in {"_class_name", "_diffusers_version", "boundary_ratio", "expand_timesteps"}
    }
    loaded: dict[str, Any] = {}
    for component_name in component_names:
        component_entry = filtered_index.get(component_name)
        if component_entry is None:
            raise KeyError(f"Component {component_name!r} not found in model_index.json")
        library_name, _ = component_entry
        component_model_path = os.path.join(model_path, component_name)
        component, _ = PipelineComponentLoader.load_component(
            component_name=component_name,
            component_model_path=component_model_path,
            transformers_or_diffusers=library_name,
            server_args=server_args,
        )
        loaded[component_name] = component
    return loaded


def get_dit_num_heads(server_args: ServerArgs) -> int:
    return server_args.pipeline_config.dit_config.arch_config.num_attention_heads


def get_legal_dit_pairs(server_args: ServerArgs, sp_degree: int) -> list[tuple[int, int]]:
    return build_ulysses_ring_pairs(
        sp_degree,
        num_heads=get_dit_num_heads(server_args),
    )


def extract_output_shape(output: Any) -> list[int] | None:
    if isinstance(output, torch.Tensor):
        return list(output.shape)
    if isinstance(output, (list, tuple)) and output:
        first_item = output[0]
        if isinstance(first_item, torch.Tensor):
            return list(first_item.shape)
    return None


def get_stage_duration_ms(
    stage_timings: dict[str, float],
    preferred_names: list[str],
) -> float:
    for name in preferred_names:
        if name in stage_timings:
            return float(stage_timings[name])
    lowered = {name.lower(): duration for name, duration in stage_timings.items()}
    for name in preferred_names:
        if name.lower() in lowered:
            return float(lowered[name.lower()])
    for name, duration in stage_timings.items():
        if "denoising" in name.lower():
            return float(duration)
    raise KeyError(
        "Unable to find a denoising stage in timings. "
        f"Available stages: {sorted(stage_timings)}"
    )


def build_server_args(args: argparse.Namespace) -> ServerArgs:
    if args.stage == "dit":
        num_gpus = args.num_gpus
        tp_size = 1
        sp_degree = args.sp_degree or num_gpus
        ulysses_degree = args.ulysses_degree
        ring_degree = args.ring_degree
    elif args.stage == "encoder":
        if args.device == "cpu":
            num_gpus = tp_size = sp_degree = ulysses_degree = ring_degree = 1
        else:
            num_gpus = args.num_gpus
            tp_size = args.tp_size or num_gpus
            sp_degree = ulysses_degree = ring_degree = 1
    else:
        if args.device == "cpu":
            num_gpus = tp_size = sp_degree = ulysses_degree = ring_degree = 1
        else:
            num_gpus = args.num_gpus
            tp_size = 1
            sp_degree = args.sp_degree or num_gpus
            ulysses_degree = sp_degree
            ring_degree = 1

    attention_backend = args.attention_backend

    server_args = ServerArgs.from_kwargs(
        model_path=args.model_path,
        num_gpus=num_gpus,
        tp_size=tp_size,
        sp_degree=sp_degree,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        attention_backend=attention_backend,
        disable_autocast=args.disable_autocast,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        warmup=False,
    )

    if args.stage == "dit":
        if server_args.sp_degree < 1:
            server_args.sp_degree = sp_degree
        legal_pairs = get_legal_dit_pairs(server_args, server_args.sp_degree)
        if not legal_pairs:
            raise ValueError(
                "No legal (ulysses, ring) pair for "
                f"sp_degree={server_args.sp_degree} with "
                f"{get_dit_num_heads(server_args)} attention heads."
            )
        if args.ulysses_degree is None and args.ring_degree is None:
            server_args.ulysses_degree, server_args.ring_degree = legal_pairs[0]
        else:
            validate_sp_topology(
                num_gpus=server_args.num_gpus,
                sp_degree=server_args.sp_degree,
                ulysses_degree=server_args.ulysses_degree,
                ring_degree=server_args.ring_degree,
                num_heads=get_dit_num_heads(server_args),
            )
        if server_args.ring_degree > 1 and server_args.attention_backend is None:
            server_args.attention_backend = "fa"

    if args.stage == "dit":
        server_args.pipeline_config.vae_config.load_encoder = False
        server_args.pipeline_config.vae_config.load_decoder = True

    return server_args


def maybe_init_runtime_distributed(server_args: ServerArgs, args: argparse.Namespace) -> None:
    if args.device != "cuda":
        return

    # Set up environment variables for single GPU case if not already set
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = str(server_args.num_gpus)
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(server_args.master_port)

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


def synthetic_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    return generator


def build_vae_input(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    generator = synthetic_generator(args.seed, device)
    return torch.rand(
        (1, 3, args.num_frames, args.height, args.width),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )


def resolve_vae_encode_input_frames(args, server_args) -> int:
    """Resolve the actual frame count to use for VAE encode.

    For I2V/TI2V tasks, always use 1 frame (the condition image).
    Otherwise, use the user-specified --vae-encode-input-frames value.
    """
    if server_args.pipeline_config.task_type in (ModelTaskType.I2V, ModelTaskType.TI2V):
        return 1
    return args.vae_encode_input_frames


def build_vae_latents(
    server_args: ServerArgs,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    batch = prepare_request(
        server_args=server_args,
        sampling_params=SamplingParams.from_user_sampling_params_args(
            server_args.model_path,
            server_args=server_args,
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
        ),
    )
    # Use effective video frame count (after request adjustment) instead of args.num_frames
    temporal_compression_ratio = server_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
    effective_video_num_frames = int(batch.num_frames)
    latent_num_frames = (effective_video_num_frames - 1) // temporal_compression_ratio + 1
    latent_shape = server_args.pipeline_config.prepare_latent_shape(
        batch,
        batch_size=1,
        num_frames=latent_num_frames,
    )
    generator = synthetic_generator(args.seed, device)
    latents = torch.randn(
        latent_shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    return latents, effective_video_num_frames, latent_num_frames


def summarize_result(result: dict[str, Any]) -> str:
    lines = [
        "Stage profiling summary:",
        f"  stage: {result['stage']}",
        f"  device: {result['device']}",
        f"  num_gpus: {result['num_gpus']}",
        f"  timing_ms: avg={result['timing_ms']['avg']:.2f}, "
        f"min={result['timing_ms']['min']:.2f}, max={result['timing_ms']['max']:.2f}",
    ]
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
    if result.get("legal_sp_combinations"):
        lines.append(
            f"  legal_sp_combinations: {result['legal_sp_combinations']}"
        )
    return "\n".join(lines)


def finalize_stage_result(
    *,
    args: argparse.Namespace,
    server_args: ServerArgs,
    iteration_timings_ms: list[float],
    output_shape: list[int] | None,
    peak_memory_mb: float | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "stage": args.stage,
        "device": args.device,
        "model_path": server_args.model_path,
        "num_gpus": server_args.num_gpus,
        "parallelism": {
            "tp_size": server_args.tp_size,
            "sp_degree": server_args.sp_degree,
            "ulysses_degree": server_args.ulysses_degree,
            "ring_degree": server_args.ring_degree,
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
        "output_shape": output_shape,
        "peak_memory_mb": peak_memory_mb,
    }
    if extra:
        result.update(extra)
    return result


def maybe_collect_peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def profile_encoder_stage(
    args: argparse.Namespace,
    server_args: ServerArgs,
    model_path: str,
    model_index: dict[str, Any],
) -> dict[str, Any]:
    stage_device = stage_device_from_arg(args.device)

    # For CPU mode, load text encoder directly without the complex distributed path
    if stage_device.type == "cpu":
        from transformers import T5EncoderModel, T5TokenizerFast

        text_encoder_path = os.path.join(model_path, "text_encoder")
        tokenizer_path = os.path.join(model_path, "tokenizer")

        text_encoder = T5EncoderModel.from_pretrained(text_encoder_path, dtype=torch.bfloat16)
        tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path)

        text_encoder = text_encoder.to(stage_device)
        text_encoder.eval()

        iteration_timings_ms: list[float] = []
        output_shape: list[int] | None = None

        for iteration in range(args.warmup_iters + args.profile_iters):
            synchronize_device(stage_device)
            start = time.perf_counter()

            # Simple encoding for CPU mode
            input_ids = tokenizer(args.prompt, return_tensors="pt", padding="max_length", max_length=512).input_ids
            input_ids = input_ids.to(stage_device)

            with torch.no_grad():
                output = text_encoder(input_ids)
                prompt_embeds = output.last_hidden_state

            synchronize_device(stage_device)
            duration_ms = (time.perf_counter() - start) * 1000.0
            if iteration >= args.warmup_iters:
                iteration_timings_ms.append(duration_ms)
            if output_shape is None:
                output_shape = list(prompt_embeds.shape)

        peak_memory_mb = maybe_collect_peak_memory_mb(stage_device)
        extra = {
            "encoder_parallel_mode": args.encoder_parallel_mode,
        }
        return finalize_stage_result(
            args=args,
            server_args=server_args,
            iteration_timings_ms=iteration_timings_ms,
            output_shape=output_shape,
            peak_memory_mb=peak_memory_mb,
            extra=extra,
        )

    # GPU mode - use the standard path
    components = load_requested_components(
        model_path=model_path,
        model_index=model_index,
        component_names=["text_encoder", "tokenizer"],
        server_args=server_args,
    )

    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]
    text_encoder = text_encoder.to(stage_device)

    from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import TextEncodingStage
    stage = TextEncodingStage(text_encoders=[text_encoder], tokenizers=[tokenizer])

    iteration_timings_ms: list[float] = []
    output_shape: list[int] | None = None

    for iteration in range(args.warmup_iters + args.profile_iters):
        if stage_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(stage_device)
        synchronize_device(stage_device)
        start = time.perf_counter()
        prompt_embeds_list, _, _ = stage.encode_text(
            args.prompt,
            server_args,
            encoder_index=[0],
            return_attention_mask=True,
            device=stage_device,
        )
        synchronize_device(stage_device)
        duration_ms = (time.perf_counter() - start) * 1000.0
        duration_ms = reduce_max_scalar(duration_ms, stage_device)
        if iteration >= args.warmup_iters:
            iteration_timings_ms.append(duration_ms)
        if output_shape is None:
            output_shape = list(prompt_embeds_list[0].shape)

    peak_memory_mb = maybe_collect_peak_memory_mb(stage_device)
    extra = {
        "encoder_parallel_mode": args.encoder_parallel_mode,
    }
    return finalize_stage_result(
        args=args,
        server_args=server_args,
        iteration_timings_ms=iteration_timings_ms,
        output_shape=output_shape,
        peak_memory_mb=peak_memory_mb,
        extra=extra,
    )


def profile_vae_encode_stage(
    args: argparse.Namespace,
    server_args: ServerArgs,
    model_path: str,
    model_index: dict[str, Any],
) -> dict[str, Any]:
    server_args.pipeline_config.vae_config.load_encoder = True
    server_args.pipeline_config.vae_config.load_decoder = False
    components = load_requested_components(
        model_path=model_path,
        model_index=model_index,
        component_names=["vae"],
        server_args=server_args,
    )
    vae = components["vae"]
    stage_device = stage_device_from_arg(args.device)
    vae = vae.to(stage_device)

    # Resolve actual encode input frames (for TI2V/I2V, always use 1)
    encode_input_frames = resolve_vae_encode_input_frames(args, server_args)
    # Create args with encode-specific frame count
    encode_args = argparse.Namespace(**vars(args))
    encode_args.num_frames = encode_input_frames
    sample = build_vae_input(encode_args, stage_device)
    vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
    autocast_enabled = vae_dtype != torch.float32 and not server_args.disable_autocast

    iteration_timings_ms: list[float] = []
    output_shape: list[int] | None = None

    for iteration in range(args.warmup_iters + args.profile_iters):
        if stage_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(stage_device)
        inputs = (sample * 2.0 - 1.0).clamp(-1, 1)
        synchronize_device(stage_device)
        start = time.perf_counter()
        with torch.autocast(
            device_type=stage_device.type,
            dtype=vae_dtype,
            enabled=autocast_enabled,
        ):
            current_inputs = inputs if autocast_enabled else inputs.to(vae_dtype)
            latents = vae.encode(current_inputs).mean
        synchronize_device(stage_device)
        duration_ms = (time.perf_counter() - start) * 1000.0
        duration_ms = reduce_max_scalar(duration_ms, stage_device)
        if iteration >= args.warmup_iters:
            iteration_timings_ms.append(duration_ms)
        if output_shape is None:
            output_shape = list(latents.shape)

    return finalize_stage_result(
        args=args,
        server_args=server_args,
        iteration_timings_ms=iteration_timings_ms,
        output_shape=output_shape,
        peak_memory_mb=maybe_collect_peak_memory_mb(stage_device),
        extra={"vae_encode_input_frames": encode_input_frames},
    )


def scale_and_shift_latents(
    latents: torch.Tensor,
    server_args: ServerArgs,
    vae,
) -> torch.Tensor:
    scaling_factor, shift_factor = server_args.pipeline_config.get_decode_scale_and_shift(
        latents.device,
        latents.dtype,
        vae,
    )
    if isinstance(scaling_factor, torch.Tensor):
        latents = latents / scaling_factor.to(latents.device, latents.dtype)
    else:
        latents = latents / scaling_factor
    if shift_factor is not None:
        if isinstance(shift_factor, torch.Tensor):
            latents = latents + shift_factor.to(latents.device, latents.dtype)
        else:
            latents = latents + shift_factor
    return latents


def profile_vae_decode_stage(
    args: argparse.Namespace,
    server_args: ServerArgs,
    model_path: str,
    model_index: dict[str, Any],
) -> dict[str, Any]:
    server_args.pipeline_config.vae_config.load_encoder = False
    server_args.pipeline_config.vae_config.load_decoder = True
    components = load_requested_components(
        model_path=model_path,
        model_index=model_index,
        component_names=["vae"],
        server_args=server_args,
    )
    vae = components["vae"]
    stage_device = stage_device_from_arg(args.device)
    vae = vae.to(stage_device)

    latents, effective_video_num_frames, latent_num_frames = build_vae_latents(
        server_args, args, stage_device
    )
    latents = scale_and_shift_latents(latents, server_args, vae)
    latents = server_args.pipeline_config.preprocess_decoding(
        latents,
        server_args=server_args,
        vae=vae,
    )
    vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
    autocast_enabled = vae_dtype != torch.float32 and not server_args.disable_autocast

    iteration_timings_ms: list[float] = []
    output_shape: list[int] | None = None

    for iteration in range(args.warmup_iters + args.profile_iters):
        if stage_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(stage_device)
        synchronize_device(stage_device)
        start = time.perf_counter()
        with torch.autocast(
            device_type=stage_device.type,
            dtype=vae_dtype,
            enabled=autocast_enabled,
        ):
            current_latents = latents if autocast_enabled else latents.to(vae_dtype)
            decoded = vae.decode(current_latents)
            if isinstance(decoded, tuple):
                decoded = decoded[0]
            elif hasattr(decoded, "sample"):
                decoded = decoded.sample
        synchronize_device(stage_device)
        duration_ms = (time.perf_counter() - start) * 1000.0
        duration_ms = reduce_max_scalar(duration_ms, stage_device)
        if iteration >= args.warmup_iters:
            iteration_timings_ms.append(duration_ms)
        if output_shape is None:
            output_shape = list(decoded.shape)

    return finalize_stage_result(
        args=args,
        server_args=server_args,
        iteration_timings_ms=iteration_timings_ms,
        output_shape=output_shape,
        peak_memory_mb=maybe_collect_peak_memory_mb(stage_device),
        extra={
            "effective_video_num_frames": effective_video_num_frames,
            "latent_num_frames": latent_num_frames,
        },
    )


def profile_dit_stage(
    args: argparse.Namespace,
    server_args: ServerArgs,
    model_path: str,
) -> dict[str, Any]:
    model_info = get_model_info(model_path, backend=server_args.backend)
    pipeline = model_info.pipeline_cls(
        model_path,
        server_args,
        executor=SyncExecutor(server_args=server_args),
    )

    iteration_timings_ms: list[float] = []
    denoise_iteration_avgs: list[float] = []
    denoise_iteration_mins: list[float] = []
    denoise_iteration_maxs: list[float] = []
    output_shape: list[int] | None = None
    stage_device = stage_device_from_arg("cuda")

    for iteration in range(args.warmup_iters + args.profile_iters):
        if stage_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(stage_device)
        sampling_params = SamplingParams.from_user_sampling_params_args(
            model_path,
            server_args=server_args,
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed + iteration,
            num_inference_steps=args.num_inference_steps,
            perf_dump_path="inline_profile.json",
            enable_sequence_shard=server_args.sp_degree > 1,
            save_output=False,
        )
        req = prepare_request(server_args=server_args, sampling_params=sampling_params)
        req.suppress_logs = True

        start = time.perf_counter()
        output_batch = pipeline.forward(req, server_args)
        synchronize_device(stage_device)
        output_batch.timings.total_duration_ms = (time.perf_counter() - start) * 1000.0

        stage_duration_ms = get_stage_duration_ms(
            output_batch.timings.stages,
            preferred_names=["DenoisingStage", "denoising_stage"],
        )
        stage_duration_ms = reduce_max_scalar(stage_duration_ms, stage_device)

        denoise_steps = list(output_batch.timings.steps)
        if denoise_steps:
            avg_step_ms = reduce_max_scalar(mean(denoise_steps), stage_device)
            min_step_ms = reduce_max_scalar(min(denoise_steps), stage_device)
            max_step_ms = reduce_max_scalar(max(denoise_steps), stage_device)
        else:
            avg_step_ms = min_step_ms = max_step_ms = 0.0

        if iteration >= args.warmup_iters:
            iteration_timings_ms.append(stage_duration_ms)
            denoise_iteration_avgs.append(avg_step_ms)
            denoise_iteration_mins.append(min_step_ms)
            denoise_iteration_maxs.append(max_step_ms)

        if output_shape is None and output_batch.output is not None:
            output_shape = extract_output_shape(output_batch.output)

    extra = {
        "denoise_steps_ms": {
            "count": args.num_inference_steps,
            "avg": mean(denoise_iteration_avgs) if denoise_iteration_avgs else 0.0,
            "min": mean(denoise_iteration_mins) if denoise_iteration_mins else 0.0,
            "max": mean(denoise_iteration_maxs) if denoise_iteration_maxs else 0.0,
        },
        "legal_sp_combinations": get_legal_dit_pairs(
            server_args,
            server_args.sp_degree,
        ),
    }

    return finalize_stage_result(
        args=args,
        server_args=server_args,
        iteration_timings_ms=iteration_timings_ms,
        output_shape=output_shape,
        peak_memory_mb=maybe_collect_peak_memory_mb(stage_device),
        extra=extra,
    )


def emit_legal_combo_report(args: argparse.Namespace) -> int:
    server_args = build_server_args(args)
    sp_degree = server_args.sp_degree
    payload = {
        "num_gpus": args.num_gpus,
        "sp_degree": sp_degree,
        "legal_pairs": [
            {"ulysses": ulysses, "ring": ring}
            for ulysses, ring in get_legal_dit_pairs(server_args, sp_degree)
        ],
    }
    if is_rank_zero():
        print(json.dumps(payload, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    validate_args(args)
    os.environ["SGLANG_DIFFUSION_SYNC_STAGE_PROFILING"] = args.sync_stage_profiling

    if args.list_legal_dit_combos:
        return emit_legal_combo_report(args)

    server_args = build_server_args(args)
    set_global_server_args(server_args)
    maybe_init_runtime_distributed(server_args, args)

    model_path, model_index = resolve_model_path(server_args.model_path)
    server_args.model_path = model_path
    json_path, txt_path = resolve_output_paths(args)

    try:
        if args.stage == "encoder":
            result = profile_encoder_stage(args, server_args, model_path, model_index)
        elif args.stage == "vae-encode":
            result = profile_vae_encode_stage(
                args,
                server_args,
                model_path,
                model_index,
            )
        elif args.stage == "vae-decode":
            result = profile_vae_decode_stage(
                args,
                server_args,
                model_path,
                model_index,
            )
        else:
            result = profile_dit_stage(args, server_args, model_path)
    finally:
        if dist.is_initialized():
            cleanup_dist_env_and_memory()

    summary = summarize_result(result)
    print_and_persist(result, summary, json_path, txt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
