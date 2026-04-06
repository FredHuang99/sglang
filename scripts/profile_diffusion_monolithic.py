#!/usr/bin/env python3
"""Preset-driven monolithic diffusion profiler.

This wraps the existing Wan-specific profiler and swaps only the model/task
specific pieces so we can reuse the same init/probe/summary pipeline across
multiple diffusion models. The default preset is Z-Image.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import aiohttp
import numpy as np

from sglang.multimodal_gen.benchmarks.bench_serving import (
    async_request_image_sglang,
    async_request_video_sglang,
)
from sglang.multimodal_gen.benchmarks.datasets import RequestFuncInput
from sglang.multimodal_gen.configs.sample.wan import (
    Wan2_2_TI2V_5B_SamplingParam,
    WanT2V_1_3B_SamplingParams,
)
from sglang.multimodal_gen.configs.sample.zimage import ZImageSamplingParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
LEGACY_SCRIPT_PATH = SCRIPT_PATH.with_name("profile_wan22_ti2v_5b_monolithic.py")


def load_legacy_module():
    spec = importlib.util.spec_from_file_location(
        "_sglang_legacy_monolithic_profile", LEGACY_SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load legacy profiler at {LEGACY_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_legacy_module()
RunConfig = legacy.RunConfig


@dataclass(frozen=True)
class ProfilePreset:
    name: str
    model_path: str
    model_id: str
    expected_task_type: str
    task_name: str
    request_kind: str
    requires_input_image: bool
    output_dir: str
    default_prompt: str
    default_input_image: str | None
    sampling_factory: Callable[[], Any]


def make_zimage_sampling() -> ZImageSamplingParams:
    return ZImageSamplingParams(width=1024, height=1024)


PRESETS: dict[str, ProfilePreset] = {
    "z-image": ProfilePreset(
        name="z-image",
        model_path="/home/heyang/models/Z_Image",
        model_id="Z-Image",
        expected_task_type="T2I",
        task_name="text-to-image",
        request_kind="image",
        requires_input_image=False,
        output_dir="/home/heyang/profile_output/z_image",
        default_prompt=(
            "cute anime style girl with massive fluffy fennec ears and a big fluffy "
            "tail blonde messy long hair blue eyes wearing a maid outfit with a long "
            "black gold leaf pattern dress and a white apron, it is a postcard held "
            'by a hand in front of a beautiful realistic city at sunset and there is '
            'cursive writing that says "ZImage, Now in ComfyUI"'
        ),
        default_input_image=None,
        sampling_factory=make_zimage_sampling,
    ),
    "wan2.2-ti2v-5b": ProfilePreset(
        name="wan2.2-ti2v-5b",
        model_path="/home/heyang/models/Wan2_2-TI2V-5B-Diffusers",
        model_id="Wan2.2-TI2V-5B-Diffusers",
        expected_task_type="TI2V",
        task_name="image-to-video",
        request_kind="video",
        requires_input_image=True,
        output_dir="/home/heyang/profile_output/wan2_2_ti2v_5b",
        default_prompt="The girl turn the body and spin around in place.",
        default_input_image=str(REPO_ROOT / "examples" / "assets" / "example_image.png"),
        sampling_factory=Wan2_2_TI2V_5B_SamplingParam,
    ),
    "wan2.1-t2v-1.3b": ProfilePreset(
        name="wan2.1-t2v-1.3b",
        model_path="/home/heyang/models/Wan2_1_T2V_1_3B",
        model_id="Wan2.1-T2V-1.3B-Diffusers",
        expected_task_type="T2V",
        task_name="text-to-video",
        request_kind="video",
        requires_input_image=False,
        output_dir="/home/heyang/profile_output/wan2_1_t2v_1_3b",
        default_prompt=(
            "A small corgi walks happily through a sunlit garden, gentle camera "
            "movement, soft cinematic lighting."
        ),
        default_input_image=None,
        sampling_factory=WanT2V_1_3B_SamplingParams,
    ),
}


ACTIVE_PRESET: ProfilePreset | None = None
ACTIVE_INPUT_IMAGE: Path | None = None
ACTIVE_REQUEST_FUNC: Callable[..., Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile monolithic diffusion serving with model presets."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="z-image",
        help="Preset to profile. Defaults to z-image.",
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--input-image", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--parallel-degrees", type=str, default="1,2,4")
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--online-rps", type=float, default=1.0)
    parser.add_argument("--probe-runs", type=int, default=3)
    parser.add_argument(
        "--run-serving-phases",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--wait-timeout", type=int, default=1800)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--keep-artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def resolve_input_image(args: argparse.Namespace, preset: ProfilePreset) -> Path | None:
    raw = args.input_image or preset.default_input_image
    if raw is None:
        return None
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    return path


def resolve_output_dir(args: argparse.Namespace, preset: ProfilePreset) -> Path:
    target = args.output_dir or preset.output_dir
    return Path(target).expanduser().resolve()


def normalize_model_id(model_id: str | None) -> str | None:
    if model_id is None:
        return None
    return model_id.rstrip("/").split("/")[-1]


def build_request_extra_body(sampling: Any) -> dict[str, Any]:
    raw = asdict(sampling)
    keys = (
        "seed",
        "guidance_scale",
        "guidance_scale_2",
        "true_cfg_scale",
        "guidance_rescale",
        "cfg_normalization",
        "boundary_ratio",
        "num_inference_steps",
        "negative_prompt",
        "enable_teacache",
        "teacache_params",
        "enable_sequence_shard",
    )
    extra_body: dict[str, Any] = {}
    for key in keys:
        value = raw.get(key)
        if value is not None:
            extra_body[key] = value
    return extra_body


def build_requests(
    *,
    dataset_path: Path,
    base_url: str,
    model_path: str,
    prompt: str,
    sampling: Any,
    num_prompts: int,
) -> list[RequestFuncInput]:
    del dataset_path
    preset = ACTIVE_PRESET
    if preset is None:
        raise RuntimeError("ACTIVE_PRESET is not initialized.")
    if preset.requires_input_image and ACTIVE_INPUT_IMAGE is None:
        raise ValueError(
            f"Preset {preset.name} requires --input-image or a preset default image."
        )

    api_url = (
        f"{base_url}/v1/images/generations"
        if preset.request_kind == "image"
        else f"{base_url}/v1/videos"
    )
    image_paths = [str(ACTIVE_INPUT_IMAGE)] if ACTIVE_INPUT_IMAGE is not None else None
    extra_body = build_request_extra_body(sampling)
    requests_list: list[RequestFuncInput] = []
    for _ in range(num_prompts):
        requests_list.append(
            RequestFuncInput(
                prompt=prompt,
                api_url=api_url,
                model=model_path,
                width=sampling.width,
                height=sampling.height,
                num_frames=getattr(sampling, "num_frames", None),
                fps=getattr(sampling, "fps", None),
                extra_body=dict(extra_body),
                image_paths=list(image_paths) if image_paths else None,
                num_inference_steps=getattr(sampling, "num_inference_steps", None),
            )
        )
    return requests_list


async def execute_requests(
    requests_list: list[RequestFuncInput],
    *,
    request_rate: float,
    max_concurrency: int,
    seed: int = 42,
    phase_name: str,
) -> tuple[list[Any], float]:
    request_func = ACTIVE_REQUEST_FUNC
    if request_func is None:
        raise RuntimeError("ACTIVE_REQUEST_FUNC is not initialized.")

    semaphore = asyncio.Semaphore(max_concurrency)
    rng = np.random.default_rng(seed)
    total_requests = len(requests_list)

    async def limited_request(request_index: int, req, session):
        async with semaphore:
            return request_index, await request_func(req, session)

    async with aiohttp.ClientSession() as session:
        tasks = []
        start_time = legacy.time.perf_counter()
        logger.info(
            "[%s] submitting %s request(s) with request_rate=%s and max_concurrency=%s",
            phase_name,
            total_requests,
            request_rate,
            max_concurrency,
        )
        for request_index, req in enumerate(requests_list, start=1):
            if request_rate != float("inf"):
                interval = rng.exponential(1.0 / request_rate)
                await asyncio.sleep(interval)
            logger.info(
                "[%s] enqueued request %s/%s",
                phase_name,
                request_index,
                total_requests,
            )
            tasks.append(asyncio.create_task(limited_request(request_index, req, session)))

        outputs: list[Any] = [None] * total_requests
        completed = 0
        for future in asyncio.as_completed(tasks):
            request_index, output = await future
            outputs[request_index - 1] = output
            completed += 1
            if total_requests <= 10 or completed in {1, total_requests} or completed % 5 == 0:
                logger.info(
                    "[%s] completed request %s/%s (latest_success=%s)",
                    phase_name,
                    completed,
                    total_requests,
                    getattr(output, "success", False),
                )
        total_duration = legacy.time.perf_counter() - start_time
        logger.info(
            "[%s] all %s request(s) finished in %.2fs",
            phase_name,
            total_requests,
            total_duration,
        )
    return outputs, total_duration


def build_metric_definitions() -> dict[str, Any]:
    payload = legacy.build_metric_definitions()
    return json.loads(
        json.dumps(payload).replace(
            "scripts/profile_wan22_ti2v_5b_monolithic.py",
            "scripts/profile_diffusion_monolithic.py",
        )
    )


def build_base_summary(
    *,
    args: argparse.Namespace,
    preset: ProfilePreset,
    input_image: Path | None,
    sampling: Any,
    parallel_degree: int,
    visible_gpu_count: int,
) -> dict[str, Any]:
    return {
        "generated_at": legacy.time.strftime("%Y-%m-%dT%H:%M:%S"),
        "preset": preset.name,
        "model": args.model_path,
        "model_id": args.model_id,
        "expected_task_type": preset.expected_task_type,
        "sampling_params": {
            "height": sampling.height,
            "width": sampling.width,
            "num_frames": getattr(sampling, "num_frames", None),
            "fps": getattr(sampling, "fps", None),
            "guidance_scale": getattr(sampling, "guidance_scale", None),
            "num_inference_steps": getattr(sampling, "num_inference_steps", None),
            "negative_prompt": getattr(sampling, "negative_prompt", None),
            "cfg_normalization": getattr(sampling, "cfg_normalization", None),
        },
        "request_setup": {
            "prompt": args.prompt,
            "input_image": str(input_image) if input_image is not None else None,
            "task_name": preset.task_name,
            "request_kind": preset.request_kind,
            "dataset_mode": "Repeated fixed request objects built directly in-script.",
            "num_requests": args.num_requests,
            "probe_runs": args.probe_runs,
            "online_rps": args.online_rps,
            "run_serving_phases": args.run_serving_phases,
            "runtime_backend": legacy.DEFAULT_RUNTIME_BACKEND,
            "runtime_backend_reason": legacy.DEFAULT_RUNTIME_BACKEND_REASON,
            "warmup_enabled": True,
            "warmup_stage_profiling_enabled": False,
            "warmup_resolutions": [f"{sampling.width}x{sampling.height}"],
            "warmup_steps": 1,
            "offload_policy": "All offload disabled explicitly: dit_cpu_offload=false, dit_layerwise_offload=false, text_encoder_cpu_offload=false, image_encoder_cpu_offload=false, vae_cpu_offload=false, pin_cpu_memory=false.",
            "probe_execution_mode": (
                "Strictly one-by-one. The script starts one probe request, waits "
                "for it to finish, then starts the next probe request."
            ),
        },
        "parallel_degree": parallel_degree,
        "graph_mode": "regular",
        "runtime_backend": legacy.DEFAULT_RUNTIME_BACKEND,
        "runtime_backend_reason": legacy.DEFAULT_RUNTIME_BACKEND_REASON,
        "visible_gpu_count": visible_gpu_count,
        "parallel_sweep_policy": {
            "explicit_only": "Only explicit tp*sp=P combinations are profiled; when sp>1, enumerate all ulysses*ring=sp pairs.",
            "deduplication": "Equivalent (tp, sp, ulysses, ring) combinations are merged so gpu=1 only runs once.",
        },
        "keep_artifacts": args.keep_artifacts,
        "metric_definitions": build_metric_definitions(),
        "human_summary": {},
        "runs": [],
        "skipped_configs": [],
    }


def refresh_human_summary(summary: dict[str, Any]) -> None:
    legacy.refresh_human_summary(summary)
    human_summary = summary.get("human_summary")
    if isinstance(human_summary, dict):
        human_summary["preset"] = summary.get("preset")
        request_setup = human_summary.get("request_setup")
        if isinstance(request_setup, dict):
            request_setup["task_name"] = (summary.get("request_setup") or {}).get(
                "task_name"
            )


def main() -> None:
    global ACTIVE_PRESET, ACTIVE_INPUT_IMAGE, ACTIVE_REQUEST_FUNC

    args = parse_args()
    preset = PRESETS[args.preset]
    args.model_path = args.model_path or preset.model_path
    args.model_id = normalize_model_id(args.model_id or preset.model_id)
    args.prompt = args.prompt or preset.default_prompt
    ACTIVE_PRESET = preset
    ACTIVE_INPUT_IMAGE = resolve_input_image(args, preset)
    ACTIVE_REQUEST_FUNC = (
        async_request_image_sglang
        if preset.request_kind == "image"
        else async_request_video_sglang
    )
    legacy.EXPECTED_TASK_TYPE = preset.expected_task_type
    legacy.execute_requests = execute_requests
    legacy.build_requests = build_requests

    parallel_degrees = legacy.parse_parallel_degrees(args)
    visible_gpu_count = legacy.resolve_visible_gpu_count()
    sampling = preset.sampling_factory()
    output_dir = resolve_output_dir(args, preset)
    output_dir.mkdir(parents=True, exist_ok=True)

    for parallel_degree in parallel_degrees:
        summary_path = output_dir / f"gpu{parallel_degree}_summary.json"
        degree_tmp_root = output_dir / ".tmp" / f"gpu{parallel_degree}"
        degree_tmp_root.mkdir(parents=True, exist_ok=True)
        summary = build_base_summary(
            args=args,
            preset=preset,
            input_image=ACTIVE_INPUT_IMAGE,
            sampling=sampling,
            parallel_degree=parallel_degree,
            visible_gpu_count=visible_gpu_count,
        )

        if visible_gpu_count < parallel_degree:
            summary["skipped_configs"].append(
                {
                    "phase": "precheck",
                    "requested_parallelism": {"num_gpus": parallel_degree},
                    "reason": (
                        f"parallel_degree={parallel_degree} requires at least "
                        f"{parallel_degree} visible GPUs, but only "
                        f"{visible_gpu_count} are visible."
                    ),
                }
            )
            refresh_human_summary(summary)
            legacy.save_json(summary_path, summary)
            if not args.keep_artifacts:
                legacy.safe_rmtree(degree_tmp_root)
            continue

        run_configs = legacy.build_run_configs(parallel_degree)
        logger.info(
            "preset=%s gpu=%s will profile %s unique tp*sp=P configuration(s)",
            preset.name,
            parallel_degree,
            len(run_configs),
        )
        for run_config in run_configs:
            logger.info(
                "Profiling preset=%s gpu=%s config=%s",
                preset.name,
                parallel_degree,
                run_config.name,
            )
            run_dir = degree_tmp_root / run_config.name
            request_dir = degree_tmp_root / "requests"
            request_dir.mkdir(parents=True, exist_ok=True)
            try:
                summary["runs"].append(
                    legacy.run_single_config(
                        model_path=args.model_path,
                        dataset_path=request_dir,
                        prompt=args.prompt,
                        sampling=sampling,
                        run_config=run_config,
                        args=args,
                        run_dir=run_dir,
                    )
                )
            except legacy.RunConfigError as exc:
                summary["skipped_configs"].append(
                    {
                        "phase": exc.phase,
                        "requested_parallelism": asdict(run_config),
                        "reason": str(exc),
                    }
                )
            except Exception as exc:
                summary["skipped_configs"].append(
                    {
                        "phase": "unknown",
                        "requested_parallelism": asdict(run_config),
                        "reason": str(exc),
                    }
                )
            finally:
                refresh_human_summary(summary)
                legacy.save_json(summary_path, summary)
                if not args.keep_artifacts:
                    legacy.safe_rmtree(run_dir)

        refresh_human_summary(summary)
        legacy.save_json(summary_path, summary)
        logger.info("Summary written to %s", summary_path.resolve())
        if not args.keep_artifacts:
            legacy.safe_rmtree(degree_tmp_root)


if __name__ == "__main__":
    main()
