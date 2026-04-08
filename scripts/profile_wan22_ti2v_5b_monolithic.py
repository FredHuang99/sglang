#!/usr/bin/env python3
"""Profile Wan2.2 TI2V 5B monolithic regular-graph serving runs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import site
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiohttp
import numpy as np
import requests
import torch

from sglang.multimodal_gen.benchmarks.bench_serving import (
    async_request_video_sglang,
    calculate_metrics,
)
from sglang.multimodal_gen.benchmarks.datasets import RequestFuncInput, VBenchDataset
from sglang.multimodal_gen.configs.sample.wan import Wan2_2_TI2V_5B_SamplingParam
from sglang.multimodal_gen.runtime.utils.common import kill_process_tree
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

DEFAULT_MODEL_PATH = "/home/heyang/models/Wan2_2-TI2V-5B-Diffusers"
DEFAULT_MODEL_ID = "Wan2.2-TI2V-5B-Diffusers"
EXPECTED_TASK_TYPE = "TI2V"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXAMPLE_IMAGE = REPO_ROOT / "examples" / "assets" / "example_image.png"
DEFAULT_TI2V_PROMPT = "The girl turn the body and spin around in place."
DEFAULT_RUNTIME_BACKEND = "native-fallback"
DEFAULT_RUNTIME_BACKEND_REASON = (
    "CuTeDSL/CUTLASS fused norm kernels are not reliably usable in the target "
    "environment. The observed failures include incomplete cutlass/cutlass.cute "
    "APIs such as missing Float16 and missing cute.jit, so diffusion fused norm "
    "ops are forced to use the native fallback path."
)
WAIT_LOG_INTERVAL_S = 15


@dataclass(frozen=True)
class RunConfig:
    name: str
    mode: str
    num_gpus: int
    tp_size: int | None
    sp_degree: int | None
    ulysses_degree: int | None
    ring_degree: int | None


class RunConfigError(RuntimeError):
    def __init__(self, phase: str, message: str):
        super().__init__(message)
        self.phase = phase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile Wan2.2 TI2V 5B monolithic regular-graph serving."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Model path or HF repo id.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=(
            "Explicit model-id override used by sglang registry to resolve the "
            "correct config for local model paths."
        ),
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default=str(DEFAULT_EXAMPLE_IMAGE),
        help="Path to a local reference image used for all requests.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_TI2V_PROMPT,
        help="Prompt used for probe/offline/online requests.",
    )
    parser.add_argument(
        "--parallel-degrees",
        type=str,
        default="1,2,4", #"1",
        help="Comma/space separated total GPU counts, e.g. '1,2,4,8'.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=20,
        help="Measured request count for offline and online phases.",
    )
    parser.add_argument(
        "--online-rps",
        type=float,
        default=1.0,
        help="Online request rate (Poisson arrivals).",
    )
    parser.add_argument(
        "--probe-runs",
        type=int,
        default=3,
        help="Number of probe requests to average for stage timing.",
    )
    parser.add_argument(
        "--run-serving-phases",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run offline/online serving benchmarks in addition to probe profiling.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the local server to.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=1800,
        help="Timeout in seconds for server readiness and phase completion.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass --trust-remote-code to the server.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep temporary logs/perf files instead of deleting them after each run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/heyang/profile_output/wan2_2_ti2v_5b",
        help="Directory for generated gpu*_summary.json files.",
    )
    return parser.parse_args()


def parse_parallel_degrees(args: argparse.Namespace) -> list[int]:
    cleaned = (
        args.parallel_degrees.replace("{", "")
        .replace("}", "")
        .replace("[", "")
        .replace("]", "")
    )
    tokens = [token for token in re.split(r"[\s,]+", cleaned.strip()) if token]
    if not tokens:
        raise ValueError("No valid values were found in --parallel-degrees.")

    degrees: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        value = int(token)
        if value <= 0:
            raise ValueError(
                f"parallel degree must be a positive integer, got {value}."
            )
        if value not in seen:
            seen.add(value)
            degrees.append(value)
    return degrees


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    return (
        Path.cwd()
        / ".codex_tmp"
        / f"wan22_ti2v_5b_monolithic_profile_{now_stamp()}"
    ).resolve()


def resolve_visible_gpu_count() -> int:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        devices = [d.strip() for d in cuda_visible_devices.split(",") if d.strip()]
        return len(devices)
    return torch.cuda.device_count()


def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def detect_cutlass_python_packages_dir() -> Path | None:
    candidates: list[Path] = []
    try:
        candidates.extend(Path(path) for path in site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            candidates.append(Path(user_site))
    except Exception:
        pass

    seen: set[Path] = set()
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)
        python_packages_dir = base / "nvidia_cutlass_dsl" / "python_packages"
        if (python_packages_dir / "cutlass").exists():
            return python_packages_dir.resolve()
    return None


def prepend_pythonpath(env: dict[str, str], path: Path) -> None:
    existing = env.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    path_str = str(path)
    if path_str not in parts:
        parts.insert(0, path_str)
    env["PYTHONPATH"] = os.pathsep.join(parts)


def tail_text(path: Path, lines: int = 200) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return "\n".join(text.splitlines()[-lines:])


def wait_for_server_ready(
    *,
    process: subprocess.Popen,
    base_url: str,
    init_profile_path: Path,
    server_log_path: Path,
    timeout_s: int,
    expected_task_type: str | None = None,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_log_time = 0.0
    health_ready = False
    init_profile_ready = False
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Server exited early with code {process.returncode}.\n"
                f"{tail_text(server_log_path)}"
            )
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            health_ready = resp.status_code == 200
            init_profile_ready = init_profile_path.exists()
            if health_ready and init_profile_ready:
                model_resp = requests.get(f"{base_url}/v1/models", timeout=2)
                if model_resp.status_code != 200:
                    time.sleep(1)
                    continue

                model_payload = model_resp.json()
                model_cards = model_payload.get("data") or []
                if not model_cards:
                    time.sleep(1)
                    continue

                model_card = model_cards[0]
                task_type = model_card.get("task_type")
                if expected_task_type and task_type != expected_task_type:
                    raise RuntimeError(
                        "Server became healthy but resolved the wrong task type. "
                        f"Expected {expected_task_type}, got {task_type}. "
                        f"Model card: {json.dumps(model_card, ensure_ascii=False)}"
                    )
                logger.info(
                    "Server ready: /health ok, init_profile.json dumped, /v1/models task_type=%s",
                    task_type,
                )
                return model_card
        except requests.RequestException:
            pass
        now = time.time()
        if now - last_log_time >= WAIT_LOG_INTERVAL_S:
            elapsed = timeout_s - max(deadline - now, 0)
            logger.info(
                "Waiting for server readiness / startup warmup... elapsed=%ss health=%s init_profile=%s",
                int(elapsed),
                health_ready,
                init_profile_ready,
            )
            last_log_time = now
        time.sleep(1)
    raise TimeoutError(
        f"Server did not become ready within {timeout_s}s.\n{tail_text(server_log_path)}"
    )


def factor_pairs(n: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(1, n + 1):
        if n % i == 0:
            pairs.append((i, n // i))
    return pairs


def build_run_configs(parallel_degree: int) -> list[RunConfig]:
    runs: list[RunConfig] = []
    seen: set[tuple[int, int, int, int]] = set()

    def add_run(run: RunConfig) -> None:
        key = (
            run.tp_size or 1,
            run.sp_degree or 1,
            run.ulysses_degree or 1,
            run.ring_degree or 1,
        )
        if key in seen:
            return
        seen.add(key)
        runs.append(run)

    divisors_desc = sorted(
        [d for d in range(1, parallel_degree + 1) if parallel_degree % d == 0],
        reverse=True,
    )
    for tp_size in divisors_desc:
        sp_degree = parallel_degree // tp_size
        if sp_degree == 1:
            add_run(
                RunConfig(
                    name=f"e_tp{tp_size}_sp{sp_degree}",
                    mode="explicit",
                    num_gpus=parallel_degree,
                    tp_size=tp_size,
                    sp_degree=sp_degree,
                    ulysses_degree=None,
                    ring_degree=None,
                )
            )
            continue
        for ulysses_degree, ring_degree in factor_pairs(sp_degree):
            add_run(
                RunConfig(
                    name=f"e_tp{tp_size}_sp{sp_degree}_u{ulysses_degree}_r{ring_degree}",
                    mode="explicit",
                    num_gpus=parallel_degree,
                    tp_size=tp_size,
                    sp_degree=sp_degree,
                    ulysses_degree=ulysses_degree,
                    ring_degree=ring_degree,
                )
            )
    return runs


def build_dataset_dir(input_image: Path, prompt: str, output_dir: Path) -> Path:
    dataset_root = output_dir / "dataset"
    origin_dir = dataset_root / "data" / "origin"
    origin_dir.mkdir(parents=True, exist_ok=True)
    copied_image = origin_dir / input_image.name
    shutil.copy2(input_image, copied_image)

    info_path = dataset_root / "data" / "i2v-bench-info.json"
    payload = [{"file_name": copied_image.name, "caption": prompt}]
    info_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")
    return dataset_root


def build_sampling_config() -> Wan2_2_TI2V_5B_SamplingParam:
    return Wan2_2_TI2V_5B_SamplingParam()


def build_request_extra_body(
    sampling: Wan2_2_TI2V_5B_SamplingParam,
) -> dict[str, Any]:
    extra_body: dict[str, Any] = {
        "guidance_scale": sampling.guidance_scale,
        "num_inference_steps": sampling.num_inference_steps,
    }
    if getattr(sampling, "negative_prompt", None):
        extra_body["negative_prompt"] = sampling.negative_prompt
    if getattr(sampling, "guidance_scale_2", None) is not None:
        extra_body["guidance_scale_2"] = sampling.guidance_scale_2
    return extra_body


def make_dataset_args(
    *,
    dataset_path: Path,
    num_prompts: int,
    sampling: Wan2_2_TI2V_5B_SamplingParam,
) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="image-to-video",
        dataset_path=str(dataset_path),
        num_prompts=num_prompts,
        width=sampling.width,
        height=sampling.height,
        num_frames=sampling.num_frames,
        fps=sampling.fps,
    )


def build_requests(
    *,
    dataset_path: Path,
    base_url: str,
    model_path: str,
    prompt: str,
    sampling: Wan2_2_TI2V_5B_SamplingParam,
    num_prompts: int,
) -> list[RequestFuncInput]:
    dataset_args = make_dataset_args(
        dataset_path=dataset_path,
        num_prompts=num_prompts,
        sampling=sampling,
    )
    dataset = VBenchDataset(
        dataset_args,
        api_url=f"{base_url}/v1/videos",
        model=model_path,
    )
    extra_body = build_request_extra_body(sampling)
    requests_list = []
    for req in dataset.get_requests():
        requests_list.append(
            replace(
                req,
                prompt=prompt,
                extra_body=dict(extra_body),
                num_inference_steps=sampling.num_inference_steps,
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
    semaphore = asyncio.Semaphore(max_concurrency)
    rng = np.random.default_rng(seed)
    total_requests = len(requests_list)

    async def limited_request(request_index: int, req, session):
        async with semaphore:
            return request_index, await async_request_video_sglang(req, session)

    async with aiohttp.ClientSession() as session:
        tasks = []
        start_time = time.perf_counter()
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
        total_duration = time.perf_counter() - start_time
        logger.info(
            "[%s] all %s request(s) finished in %.2fs",
            phase_name,
            total_requests,
            total_duration,
        )
    return outputs, total_duration


def clear_perf_log(perf_log_path: Path, phase_name: str) -> None:
    if perf_log_path.exists():
        perf_log_path.unlink()
        logger.info("[%s] cleared previous performance.log: %s", phase_name, perf_log_path)
    else:
        logger.info(
            "[%s] performance.log does not exist yet; starting fresh at %s",
            phase_name,
            perf_log_path,
        )


def read_perf_records(perf_log_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not perf_log_path.exists():
        return records
    with perf_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def wait_for_perf_records(
    perf_log_path: Path, expected_count: int, timeout_s: int, phase_name: str
) -> list[dict[str, Any]]:
    deadline = time.time() + timeout_s
    last_log_time = 0.0
    last_count = -1
    while time.time() < deadline:
        records = read_perf_records(perf_log_path)
        if len(records) >= expected_count:
            logger.info(
                "[%s] collected %s/%s perf record(s)",
                phase_name,
                len(records),
                expected_count,
            )
            return records
        now = time.time()
        if len(records) != last_count or now - last_log_time >= WAIT_LOG_INTERVAL_S:
            elapsed = timeout_s - max(deadline - now, 0)
            logger.info(
                "[%s] waiting for perf records... current=%s expected=%s elapsed=%ss",
                phase_name,
                len(records),
                expected_count,
                int(elapsed),
            )
            last_count = len(records)
            last_log_time = now
        time.sleep(1)
    records = read_perf_records(perf_log_path)
    raise TimeoutError(
        f"Expected at least {expected_count} perf records, got {len(records)} from "
        f"{perf_log_path}"
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def round_float(value: float) -> float:
    return round(float(value), 4)


def mb_to_gb(value: float) -> float:
    return round_float(float(value) / 1024.0)


def ms_to_s(value: float) -> float:
    return round_float(float(value) / 1000.0)


def summarize_series(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": round_float(float(np.mean(values))),
        "median": round_float(float(np.median(values))),
        "p99": round_float(percentile(values, 99)),
        "max": round_float(max(values)),
    }


def describe_memory_snapshot_capture_points(
    records: list[dict[str, Any]],
) -> dict[str, str]:
    snapshot_keys: set[str] = set()
    for record in records:
        snapshot_keys.update((record.get("memory_snapshots", {}) or {}).keys())

    descriptions: dict[str, str] = {}
    for key in sorted(snapshot_keys):
        if key == "before_forward":
            descriptions[key] = (
                "Captured immediately before the request enters pipeline.forward(). "
                "This is the pre-request baseline for the first profiled request."
            )
        elif key == "after_forward":
            descriptions[key] = (
                "Captured immediately after pipeline.forward() returns for the request."
            )
        elif key == "mem_analysis":
            descriptions[key] = (
                "Captured in GPUWorker.do_mem_analysis() after request execution has finished."
            )
        elif key.startswith("after_"):
            stage_name = key[len("after_") :]
            descriptions[key] = (
                f"Captured immediately after {stage_name} finishes."
            )
        else:
            descriptions[key] = "Captured at a runtime-defined profiling checkpoint."
    return descriptions


def aggregate_pre_forward_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    reserved_values: list[float] = []
    allocated_values: list[float] = []
    first_snapshot: dict[str, Any] | None = None

    for record in records:
        snapshot = (record.get("memory_snapshots", {}) or {}).get("before_forward")
        if not snapshot:
            continue
        if first_snapshot is None:
            first_snapshot = snapshot
        reserved_values.append(float(snapshot.get("reserved_mb", 0.0)))
        allocated_values.append(float(snapshot.get("allocated_mb", 0.0)))

    reserved_summary = summarize_series(reserved_values)
    allocated_summary = summarize_series(allocated_values)
    first_reserved_mb = (
        round_float(float(first_snapshot.get("reserved_mb", 0.0)))
        if first_snapshot
        else 0.0
    )
    first_allocated_mb = (
        round_float(float(first_snapshot.get("allocated_mb", 0.0)))
        if first_snapshot
        else 0.0
    )
    return {
        "request_baseline_reserved_mb_mean": reserved_summary["mean"],
        "request_baseline_reserved_mb_median": reserved_summary["median"],
        "request_baseline_reserved_mb_p99": reserved_summary["p99"],
        "request_baseline_reserved_mb_max": reserved_summary["max"],
        "request_baseline_reserved_gb_mean": mb_to_gb(reserved_summary["mean"]),
        "request_baseline_reserved_gb_median": mb_to_gb(reserved_summary["median"]),
        "request_baseline_reserved_gb_p99": mb_to_gb(reserved_summary["p99"]),
        "request_baseline_reserved_gb_max": mb_to_gb(reserved_summary["max"]),
        "request_baseline_allocated_mb_mean": allocated_summary["mean"],
        "request_baseline_allocated_mb_median": allocated_summary["median"],
        "request_baseline_allocated_mb_p99": allocated_summary["p99"],
        "request_baseline_allocated_mb_max": allocated_summary["max"],
        "request_baseline_allocated_gb_mean": mb_to_gb(allocated_summary["mean"]),
        "request_baseline_allocated_gb_median": mb_to_gb(
            allocated_summary["median"]
        ),
        "request_baseline_allocated_gb_p99": mb_to_gb(allocated_summary["p99"]),
        "request_baseline_allocated_gb_max": mb_to_gb(allocated_summary["max"]),
        "first_request_pre_forward_reserved_mb": first_reserved_mb,
        "first_request_pre_forward_reserved_gb": mb_to_gb(first_reserved_mb),
        "first_request_pre_forward_allocated_mb": first_allocated_mb,
        "first_request_pre_forward_allocated_gb": mb_to_gb(first_allocated_mb),
    }


def aggregate_denoiser_step_memory(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_request_peak_reserved: list[float] = []
    per_request_peak_allocated: list[float] = []
    per_request_step_counts: list[float] = []

    for record in records:
        snapshot_map = record.get("memory_snapshots", {}) or {}
        step_reserved: list[float] = []
        step_allocated: list[float] = []
        for checkpoint_name, snapshot in snapshot_map.items():
            if not checkpoint_name.startswith("after_denoising_step_"):
                continue
            step_reserved.append(float(snapshot.get("peak_reserved_mb", 0.0)))
            step_allocated.append(float(snapshot.get("peak_allocated_mb", 0.0)))

        if step_reserved:
            per_request_peak_reserved.append(max(step_reserved))
            per_request_peak_allocated.append(max(step_allocated))
            per_request_step_counts.append(float(len(step_reserved)))

    reserved_summary = summarize_series(per_request_peak_reserved)
    allocated_summary = summarize_series(per_request_peak_allocated)
    step_count_summary = summarize_series(per_request_step_counts)
    return {
        "requests_with_denoiser_step_memory": len(per_request_peak_reserved),
        "denoiser_step_snapshot_count_mean": step_count_summary["mean"],
        "denoiser_step_snapshot_count_median": step_count_summary["median"],
        "denoiser_step_snapshot_count_p99": step_count_summary["p99"],
        "denoiser_step_snapshot_count_max": step_count_summary["max"],
        "denoiser_step_peak_reserved_mb_mean": reserved_summary["mean"],
        "denoiser_step_peak_reserved_mb_median": reserved_summary["median"],
        "denoiser_step_peak_reserved_mb_p99": reserved_summary["p99"],
        "denoiser_step_peak_reserved_mb_max": reserved_summary["max"],
        "denoiser_step_peak_reserved_gb_mean": mb_to_gb(reserved_summary["mean"]),
        "denoiser_step_peak_reserved_gb_median": mb_to_gb(
            reserved_summary["median"]
        ),
        "denoiser_step_peak_reserved_gb_p99": mb_to_gb(reserved_summary["p99"]),
        "denoiser_step_peak_reserved_gb_max": mb_to_gb(reserved_summary["max"]),
        "denoiser_step_peak_allocated_mb_mean": allocated_summary["mean"],
        "denoiser_step_peak_allocated_mb_median": allocated_summary["median"],
        "denoiser_step_peak_allocated_mb_p99": allocated_summary["p99"],
        "denoiser_step_peak_allocated_mb_max": allocated_summary["max"],
        "denoiser_step_peak_allocated_gb_mean": mb_to_gb(
            allocated_summary["mean"]
        ),
        "denoiser_step_peak_allocated_gb_median": mb_to_gb(
            allocated_summary["median"]
        ),
        "denoiser_step_peak_allocated_gb_p99": mb_to_gb(
            allocated_summary["p99"]
        ),
        "denoiser_step_peak_allocated_gb_max": mb_to_gb(allocated_summary["max"]),
    }


def aggregate_stage_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    durations: dict[str, list[float]] = {}
    peak_reserved: dict[str, list[float]] = {}
    peak_allocated: dict[str, list[float]] = {}

    for record in records:
        snapshot_map = record.get("memory_snapshots", {}) or {}
        for stage in record.get("stages", []) or []:
            stage_name = stage.get("name")
            if not stage_name:
                continue
            durations.setdefault(stage_name, []).append(
                float(stage.get("execution_time_ms", 0.0))
            )
            snapshot = snapshot_map.get(f"after_{stage_name}")
            if snapshot:
                peak_reserved.setdefault(stage_name, []).append(
                    float(snapshot.get("peak_reserved_mb", 0.0))
                )
                peak_allocated.setdefault(stage_name, []).append(
                    float(snapshot.get("peak_allocated_mb", 0.0))
                )

    stage_duration_ms_mean: dict[str, float] = {}
    stage_duration_ms_median: dict[str, float] = {}
    stage_duration_ms_p99: dict[str, float] = {}
    stage_duration_ms_max: dict[str, float] = {}
    stage_duration_s_mean: dict[str, float] = {}
    stage_duration_s_median: dict[str, float] = {}
    stage_duration_s_p99: dict[str, float] = {}
    stage_duration_s_max: dict[str, float] = {}
    for stage_name, values in sorted(durations.items()):
        stage_duration_ms_mean[stage_name] = round_float(float(np.mean(values)))
        stage_duration_ms_median[stage_name] = round_float(float(np.median(values)))
        stage_duration_ms_p99[stage_name] = round_float(percentile(values, 99))
        stage_duration_ms_max[stage_name] = round_float(max(values))
        stage_duration_s_mean[stage_name] = ms_to_s(stage_duration_ms_mean[stage_name])
        stage_duration_s_median[stage_name] = ms_to_s(
            stage_duration_ms_median[stage_name]
        )
        stage_duration_s_p99[stage_name] = ms_to_s(stage_duration_ms_p99[stage_name])
        stage_duration_s_max[stage_name] = ms_to_s(stage_duration_ms_max[stage_name])

    stage_peak_reserved_mb_mean: dict[str, float] = {}
    stage_peak_reserved_mb_median: dict[str, float] = {}
    stage_peak_reserved_mb_max: dict[str, float] = {}
    stage_peak_reserved_gb_mean: dict[str, float] = {}
    stage_peak_reserved_gb_median: dict[str, float] = {}
    stage_peak_reserved_gb_max: dict[str, float] = {}
    for stage_name, values in sorted(peak_reserved.items()):
        stage_peak_reserved_mb_mean[stage_name] = round_float(float(np.mean(values)))
        stage_peak_reserved_mb_median[stage_name] = round_float(
            float(np.median(values))
        )
        stage_peak_reserved_mb_max[stage_name] = round_float(max(values))
        stage_peak_reserved_gb_mean[stage_name] = mb_to_gb(
            stage_peak_reserved_mb_mean[stage_name]
        )
        stage_peak_reserved_gb_median[stage_name] = mb_to_gb(
            stage_peak_reserved_mb_median[stage_name]
        )
        stage_peak_reserved_gb_max[stage_name] = mb_to_gb(
            stage_peak_reserved_mb_max[stage_name]
        )

    stage_peak_allocated_mb_mean: dict[str, float] = {}
    stage_peak_allocated_mb_median: dict[str, float] = {}
    stage_peak_allocated_mb_max: dict[str, float] = {}
    stage_peak_allocated_gb_mean: dict[str, float] = {}
    stage_peak_allocated_gb_median: dict[str, float] = {}
    stage_peak_allocated_gb_max: dict[str, float] = {}
    for stage_name, values in sorted(peak_allocated.items()):
        stage_peak_allocated_mb_mean[stage_name] = round_float(
            float(np.mean(values))
        )
        stage_peak_allocated_mb_median[stage_name] = round_float(
            float(np.median(values))
        )
        stage_peak_allocated_mb_max[stage_name] = round_float(max(values))
        stage_peak_allocated_gb_mean[stage_name] = mb_to_gb(
            stage_peak_allocated_mb_mean[stage_name]
        )
        stage_peak_allocated_gb_median[stage_name] = mb_to_gb(
            stage_peak_allocated_mb_median[stage_name]
        )
        stage_peak_allocated_gb_max[stage_name] = mb_to_gb(
            stage_peak_allocated_mb_max[stage_name]
        )

    return {
        "records_count": len(records),
        "memory_snapshot_capture_points": describe_memory_snapshot_capture_points(
            records
        ),
        "stage_duration_ms_mean": stage_duration_ms_mean,
        "stage_duration_ms_median": stage_duration_ms_median,
        "stage_duration_ms_p99": stage_duration_ms_p99,
        "stage_duration_ms_max": stage_duration_ms_max,
        "stage_duration_s_mean": stage_duration_s_mean,
        "stage_duration_s_median": stage_duration_s_median,
        "stage_duration_s_p99": stage_duration_s_p99,
        "stage_duration_s_max": stage_duration_s_max,
        "stage_peak_reserved_mb_mean": stage_peak_reserved_mb_mean,
        "stage_peak_reserved_mb_median": stage_peak_reserved_mb_median,
        "stage_peak_reserved_mb_max": stage_peak_reserved_mb_max,
        "stage_peak_reserved_gb_mean": stage_peak_reserved_gb_mean,
        "stage_peak_reserved_gb_median": stage_peak_reserved_gb_median,
        "stage_peak_reserved_gb_max": stage_peak_reserved_gb_max,
        "stage_peak_allocated_mb_mean": stage_peak_allocated_mb_mean,
        "stage_peak_allocated_mb_median": stage_peak_allocated_mb_median,
        "stage_peak_allocated_mb_max": stage_peak_allocated_mb_max,
        "stage_peak_allocated_gb_mean": stage_peak_allocated_gb_mean,
        "stage_peak_allocated_gb_median": stage_peak_allocated_gb_median,
        "stage_peak_allocated_gb_max": stage_peak_allocated_gb_max,
        **aggregate_denoiser_step_memory(records),
    }


def aggregate_probe(
    records: list[dict[str, Any]], probe_outputs: list[Any]
) -> dict[str, Any]:
    stage_values: dict[str, list[float]] = {}
    total_durations: list[float] = []
    request_peak_memory_values = [
        float(output.peak_memory_mb)
        for output in probe_outputs
        if getattr(output, "peak_memory_mb", 0.0) > 0
    ]

    for record in records:
        total_durations.append(float(record.get("total_duration_ms", 0.0)))
        for stage in record.get("stages", []) or []:
            stage_name = stage.get("name")
            if not stage_name:
                continue
            stage_values.setdefault(stage_name, []).append(
                float(stage.get("execution_time_ms", 0.0))
            )

    request_peak_summary = summarize_series(request_peak_memory_values)
    stage_time_mean_ms = {
        stage_name: round_float(float(np.mean(values)))
        for stage_name, values in sorted(stage_values.items())
    }
    return {
        "num_runs": len(records),
        "stage_time_mean_ms": stage_time_mean_ms,
        "stage_time_mean_s": {
            stage_name: ms_to_s(value) for stage_name, value in stage_time_mean_ms.items()
        },
        "total_duration_mean_ms": round_float(float(np.mean(total_durations)))
        if total_durations
        else 0.0,
        "total_duration_mean_s": ms_to_s(float(np.mean(total_durations)))
        if total_durations
        else 0.0,
        "request_peak_memory_mb_mean": request_peak_summary["mean"],
        "request_peak_memory_mb_median": request_peak_summary["median"],
        "request_peak_memory_mb_max": request_peak_summary["max"],
        "request_peak_memory_gb_mean": mb_to_gb(request_peak_summary["mean"]),
        "request_peak_memory_gb_median": mb_to_gb(request_peak_summary["median"]),
        "request_peak_memory_gb_max": mb_to_gb(request_peak_summary["max"]),
        **aggregate_pre_forward_baseline(records),
        **aggregate_stage_metrics(records),
        **aggregate_denoiser_step_memory(records),
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def augment_init_profile_units(init_profile: dict[str, Any]) -> dict[str, Any]:
    snapshot_keys = [
        "before_build_pipeline",
        "after_build_pipeline",
        "after_startup_warmup",
    ]
    for snapshot_key in snapshot_keys:
        snapshot = init_profile.get(snapshot_key)
        if not isinstance(snapshot, dict):
            continue
        for source_key, target_key in (
            ("allocated_mb", "allocated_gb"),
            ("reserved_mb", "reserved_gb"),
            ("peak_allocated_mb", "peak_allocated_gb"),
            ("peak_reserved_mb", "peak_reserved_gb"),
        ):
            if source_key in snapshot:
                snapshot[target_key] = mb_to_gb(float(snapshot[source_key]))

    for source_key, target_key in (
        ("parameter_reserved_mb", "parameter_reserved_gb"),
        ("parameter_allocated_mb", "parameter_allocated_gb"),
        ("warmup_persistent_reserved_mb", "warmup_persistent_reserved_gb"),
        ("warmup_peak_reserved_mb", "warmup_peak_reserved_gb"),
        ("warmup_peak_allocated_mb", "warmup_peak_allocated_gb"),
        ("runtime_transient_peak_reserved_mb", "runtime_transient_peak_reserved_gb"),
        (
            "runtime_transient_peak_allocated_mb",
            "runtime_transient_peak_allocated_gb",
        ),
    ):
        if source_key in init_profile:
            init_profile[target_key] = mb_to_gb(float(init_profile[source_key]))

    for mapping_key in (
        "component_loaded_weight_file_size_gb",
        "component_final_module_size_gb",
        "component_param_size_gb",
        "component_gpu_load_consumed_gb",
        "component_weight_profile_gb",
        "pipeline_memory_usages_gb",
    ):
        mapping = init_profile.get(mapping_key)
        if isinstance(mapping, dict):
            init_profile[mapping_key] = {
                name: (round_float(float(value)) if value is not None else None)
                for name, value in mapping.items()
            }
    return init_profile


def build_stage_component_size_maps(
    init_profile: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    stage_component_map = init_profile.get("stage_component_map", {}) or {}
    component_loaded_weight_file_size_gb = (
        init_profile.get("component_loaded_weight_file_size_gb", {}) or {}
    )
    component_final_module_size_gb = (
        init_profile.get("component_final_module_size_gb", {})
        or init_profile.get("component_param_size_gb", {})
        or {}
    )
    component_gpu_load_consumed_gb = (
        init_profile.get("component_gpu_load_consumed_gb", {}) or {}
    )

    stage_loaded_weight_file_size_gb: dict[str, float] = {}
    stage_final_module_size_gb: dict[str, float] = {}
    stage_gpu_load_consumed_gb: dict[str, float] = {}
    for stage_name, components in stage_component_map.items():
        if not isinstance(components, list):
            continue
        stage_loaded_weight_file_size_gb[stage_name] = round_float(
            sum(
                float(component_loaded_weight_file_size_gb.get(component, 0.0) or 0.0)
                for component in components
            )
        )
        stage_final_module_size_gb[stage_name] = round_float(
            sum(
                float(component_final_module_size_gb.get(component, 0.0) or 0.0)
                for component in components
            )
        )
        stage_gpu_load_consumed_gb[stage_name] = round_float(
            sum(
                float(component_gpu_load_consumed_gb.get(component, 0.0) or 0.0)
                for component in components
            )
        )
    return (
        stage_loaded_weight_file_size_gb,
        stage_final_module_size_gb,
        stage_gpu_load_consumed_gb,
    )


def attach_stage_runtime_heuristics(
    phase_summary: dict[str, Any], init_profile: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(phase_summary, dict):
        return phase_summary

    stage_peak_reserved_gb_max = phase_summary.get("stage_peak_reserved_gb_max", {}) or {}
    stage_peak_allocated_gb_max = phase_summary.get(
        "stage_peak_allocated_gb_max", {}
    ) or {}
    (
        stage_loaded_weight_file_size_gb,
        stage_final_module_size_gb,
        stage_gpu_load_consumed_gb,
    ) = build_stage_component_size_maps(init_profile)

    phase_summary["stage_component_loaded_weight_file_size_gb"] = (
        stage_loaded_weight_file_size_gb
    )
    phase_summary["stage_component_final_module_size_gb"] = (
        stage_final_module_size_gb
    )
    phase_summary["stage_component_param_size_gb"] = stage_final_module_size_gb
    phase_summary["stage_component_gpu_load_consumed_gb"] = stage_gpu_load_consumed_gb

    phase_summary["stage_rough_runtime_like_reserved_gb_max"] = {
        stage_name: round_float(
            max(
                float(peak_gb)
                - float(stage_final_module_size_gb.get(stage_name, 0.0)),
                0.0,
            )
        )
        for stage_name, peak_gb in stage_peak_reserved_gb_max.items()
    }
    phase_summary["stage_rough_runtime_like_allocated_gb_max"] = {
        stage_name: round_float(
            max(
                float(peak_gb)
                - float(stage_final_module_size_gb.get(stage_name, 0.0)),
                0.0,
            )
        )
        for stage_name, peak_gb in stage_peak_allocated_gb_max.items()
    }
    phase_summary["stage_rough_runtime_like_reserved_gb_max_from_loaded_files"] = {
        stage_name: round_float(
            max(
                float(peak_gb)
                - float(stage_loaded_weight_file_size_gb.get(stage_name, 0.0)),
                0.0,
            )
        )
        for stage_name, peak_gb in stage_peak_reserved_gb_max.items()
    }
    phase_summary["stage_rough_runtime_like_allocated_gb_max_from_loaded_files"] = {
        stage_name: round_float(
            max(
                float(peak_gb)
                - float(stage_loaded_weight_file_size_gb.get(stage_name, 0.0)),
                0.0,
            )
        )
        for stage_name, peak_gb in stage_peak_allocated_gb_max.items()
    }
    return phase_summary


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_builtin(payload), indent=2, ensure_ascii=False), "utf-8"
    )


def benchmark_outputs_to_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "duration_s": round_float(metrics.get("duration", 0.0)),
        "duration_ms": round_float(metrics.get("duration", 0.0) * 1000.0),
        "completed_requests": int(metrics.get("completed_requests", 0)),
        "failed_requests": int(metrics.get("failed_requests", 0)),
        "throughput_qps": round_float(metrics.get("throughput_qps", 0.0)),
        "latency_mean": round_float(metrics.get("latency_mean", 0.0)),
        "latency_median": round_float(metrics.get("latency_median", 0.0)),
        "latency_p99": round_float(metrics.get("latency_p99", 0.0)),
        "latency_mean_ms": round_float(metrics.get("latency_mean", 0.0) * 1000.0),
        "latency_median_ms": round_float(
            metrics.get("latency_median", 0.0) * 1000.0
        ),
        "latency_p99_ms": round_float(metrics.get("latency_p99", 0.0) * 1000.0),
        "latency_mean_s": round_float(metrics.get("latency_mean", 0.0)),
        "latency_median_s": round_float(metrics.get("latency_median", 0.0)),
        "latency_p99_s": round_float(metrics.get("latency_p99", 0.0)),
        "request_peak_memory_mb_max": round_float(
            metrics.get("peak_memory_mb_max", 0.0)
        ),
        "request_peak_memory_mb_mean": round_float(
            metrics.get("peak_memory_mb_mean", 0.0)
        ),
        "request_peak_memory_mb_median": round_float(
            metrics.get("peak_memory_mb_median", 0.0)
        ),
        "request_peak_memory_gb_max": mb_to_gb(
            metrics.get("peak_memory_mb_max", 0.0)
        ),
        "request_peak_memory_gb_mean": mb_to_gb(
            metrics.get("peak_memory_mb_mean", 0.0)
        ),
        "request_peak_memory_gb_median": mb_to_gb(
            metrics.get("peak_memory_mb_median", 0.0)
        ),
    }


def build_resolved_parallelism(init_profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_gpus": init_profile.get("num_gpus"),
        "tp_size": init_profile.get("tp_size"),
        "sp_degree": init_profile.get("sp_degree"),
        "ulysses_degree": init_profile.get("ulysses_degree"),
        "ring_degree": init_profile.get("ring_degree"),
        "dp_size": init_profile.get("dp_size"),
    }


def build_server_command(
    *,
    model_path: str,
    model_id: str | None,
    run_config: RunConfig,
    sampling: Wan2_2_TI2V_5B_SamplingParam,
    host: str,
    port: int,
    scheduler_port: int,
    master_port: int,
    output_dir: Path,
    trust_remote_code: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.runtime.entrypoints.cli.main",
        "serve",
        "--model-path",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--scheduler-port",
        str(scheduler_port),
        "--master-port",
        str(master_port),
        "--num-gpus",
        str(run_config.num_gpus),
        "--warmup",
        "--warmup-resolutions",
        f"{sampling.width}x{sampling.height}",
        "--warmup-steps",
        "1",
        "--output-path",
        str((output_dir / "outputs").resolve()),
        "--input-save-path",
        str((output_dir / "uploads").resolve()),
        "--log-level",
        "info",
        "--dit-cpu-offload",
        "false",
        "--dit-layerwise-offload",
        "false",
        "--text-encoder-cpu-offload",
        "false",
        "--image-encoder-cpu-offload",
        "false",
        "--vae-cpu-offload",
        "false",
        "--pin-cpu-memory",
        "false",
    ]
    if model_id:
        command.extend(["--model-id", model_id])
    if run_config.tp_size is not None:
        command.extend(["--tp-size", str(run_config.tp_size)])
    if run_config.sp_degree is not None:
        command.extend(["--sp-degree", str(run_config.sp_degree)])
    if run_config.ulysses_degree is not None:
        command.extend(["--ulysses-degree", str(run_config.ulysses_degree)])
    if run_config.ring_degree is not None:
        command.extend(["--ring-degree", str(run_config.ring_degree)])
    if trust_remote_code:
        command.append("--trust-remote-code")
    return command


def launch_server(
    *,
    model_path: str,
    model_id: str | None,
    run_config: RunConfig,
    sampling: Wan2_2_TI2V_5B_SamplingParam,
    run_dir: Path,
    host: str,
    timeout_s: int,
    trust_remote_code: bool,
) -> tuple[subprocess.Popen, str, Path, Path, dict[str, Any]]:
    port = find_free_port(host)
    scheduler_port = find_free_port(host)
    master_port = find_free_port(host)
    server_log_path = run_dir / "server.log"
    perf_dir = run_dir / "perf"
    perf_dir.mkdir(parents=True, exist_ok=True)
    init_profile_path = perf_dir / "init_profile.json"
    base_url = f"http://{host}:{port}"

    env = os.environ.copy()
    env["SGLANG_DIFFUSION_STAGE_LOGGING"] = "1"
    env["SGLANG_PERF_LOG_DIR"] = str(perf_dir.resolve())
    env["SGLANG_DIFFUSION_DUMP_INIT_PROFILE"] = "1"
    env["SGLANG_DIFFUSION_CAPTURE_STAGE_MEMORY"] = "1"
    cutlass_python_packages_dir = detect_cutlass_python_packages_dir()
    if cutlass_python_packages_dir is not None:
        prepend_pythonpath(env, cutlass_python_packages_dir)
        logger.info(
            "Added CUTLASS Python packages dir to PYTHONPATH: %s",
            cutlass_python_packages_dir,
        )
    else:
        logger.warning(
            "Could not find nvidia_cutlass_dsl/python_packages on sys.path roots. "
            "If `import cutlass` fails in the server, add that directory to PYTHONPATH."
        )

    command = build_server_command(
        model_path=model_path,
        model_id=model_id,
        run_config=run_config,
        sampling=sampling,
        host=host,
        port=port,
        scheduler_port=scheduler_port,
        master_port=master_port,
        output_dir=run_dir,
        trust_remote_code=trust_remote_code,
    )
    logger.info(
        "Launching %s [runtime_backend=%s]",
        " ".join(command),
        DEFAULT_RUNTIME_BACKEND,
    )
    log_fh = server_log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        model_card = wait_for_server_ready(
            process=process,
            base_url=base_url,
            init_profile_path=init_profile_path,
            server_log_path=server_log_path,
            timeout_s=timeout_s,
            expected_task_type=EXPECTED_TASK_TYPE,
        )
    except Exception:
        log_fh.flush()
        log_fh.close()
        kill_process_tree(process.pid)
        raise
    process._sgl_log_fh = log_fh  # type: ignore[attr-defined]
    return process, base_url, perf_dir, init_profile_path, model_card


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    try:
        kill_process_tree(process.pid)
    except Exception:
        process.send_signal(signal.SIGTERM)
    log_fh = getattr(process, "_sgl_log_fh", None)
    if log_fh is not None:
        try:
            log_fh.flush()
            log_fh.close()
        except Exception:
            pass


async def run_probe_phase(
    requests_list: list[RequestFuncInput],
    probe_runs: int,
    phase_name: str,
) -> tuple[list[Any], float]:
    probe_request = replace(requests_list[0])
    outputs: list[Any] = []
    total_duration = 0.0
    logger.info("[%s] probe requests will run strictly one-by-one", phase_name)
    for probe_index in range(1, probe_runs + 1):
        logger.info("[%s] starting probe request %s/%s", phase_name, probe_index, probe_runs)
        probe_outputs, probe_duration = await execute_requests(
            [replace(probe_request)],
            request_rate=float("inf"),
            max_concurrency=1,
            phase_name=f"{phase_name}:run{probe_index}",
        )
        outputs.extend(probe_outputs)
        total_duration += probe_duration
        logger.info(
            "[%s] finished probe request %s/%s in %.2fs",
            phase_name,
            probe_index,
            probe_runs,
            probe_duration,
        )
    return outputs, total_duration


async def run_serving_phase(
    requests_list: list[RequestFuncInput],
    request_rate: float,
    max_concurrency: int,
    seed: int,
    phase_name: str,
) -> tuple[list[Any], float]:
    return await execute_requests(
        requests_list,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        seed=seed,
        phase_name=phase_name,
    )


def ensure_successful_requests(
    phase_name: str, metrics: dict[str, Any], expected_requests: int
) -> None:
    completed_requests = int(metrics.get("completed_requests", 0))
    failed_requests = int(metrics.get("failed_requests", 0))
    if completed_requests != expected_requests or failed_requests != 0:
        raise RunConfigError(
            phase_name,
            f"{phase_name} expected {expected_requests} successful requests, got "
            f"{completed_requests} success / {failed_requests} failed."
        )


def ensure_probe_successful(probe_outputs: list[Any], expected_count: int) -> None:
    success_count = sum(
        1 for output in probe_outputs if getattr(output, "success", False)
    )
    if success_count != expected_count:
        errors = [
            getattr(output, "error", "")
            for output in probe_outputs
            if not getattr(output, "success", False)
        ]
        raise RunConfigError(
            "probe",
            f"probe expected {expected_count} successful requests, got "
            f"{success_count}. Errors: {errors}",
        )


def build_metric_definitions() -> dict[str, Any]:
    return {
        "runtime_backend": {
            "selected_backend": DEFAULT_RUNTIME_BACKEND,
            "reason": DEFAULT_RUNTIME_BACKEND_REASON,
            "meaning": (
                "Current profiling results should be interpreted under the "
                "native-fallback backend instead of the CuTeDSL/CUTLASS fused "
                "norm backend."
            ),
            "native_fallback_impl": {
                "fused_scale_residual_norm_scale_shift": (
                    "python/sglang/multimodal_gen/runtime/layers/layernorm.py::"
                    "_ScaleResidualNormScaleShift.forward_native"
                ),
                "fused_norm_scale_shift": (
                    "python/sglang/multimodal_gen/runtime/layers/layernorm.py::"
                    "_NormScaleShift.forward_native"
                ),
            },
            "composition": [
                "Residual/gate accumulation uses regular torch tensor ops on CUDA.",
                "Normalization falls back to existing non-CUTLASS implementations "
                "(sgl-kernel RMSNorm, Triton one-pass RMSNorm, or torch layer_norm "
                "depending on norm type and shape).",
                "Scale/shift modulation still uses the Triton fuse_scale_shift_kernel.",
            ],
            "difference_vs_cutedsl": [
                "More kernel launches instead of one fused CUTLASS kernel.",
                "More intermediate tensors and usually higher transient memory.",
                "Lower throughput and typically higher latency.",
            ],
        },
        "request_level": {
            "probe_execution_mode": {
                "source": "scripts/profile_wan22_ti2v_5b_monolithic.py::run_probe_phase()",
                "meaning": "Probe requests are executed strictly one-by-one. The script launches one probe request, waits for it to finish, and only then starts the next probe run.",
            },
            "throughput_qps": {
                "source": "python/sglang/multimodal_gen/benchmarks/bench_serving.py::calculate_metrics()['throughput_qps']",
                "meaning": "Completed requests divided by phase wall-clock duration.",
            },
            "latency_mean": {
                "source": "python/sglang/multimodal_gen/benchmarks/bench_serving.py::calculate_metrics()['latency_mean']",
                "meaning": "Mean end-to-end request latency in seconds on the client side.",
            },
            "latency_median": {
                "source": "python/sglang/multimodal_gen/benchmarks/bench_serving.py::calculate_metrics()['latency_median']",
                "meaning": "Median end-to-end request latency in seconds on the client side.",
            },
            "latency_p99": {
                "source": "python/sglang/multimodal_gen/benchmarks/bench_serving.py::calculate_metrics()['latency_p99']",
                "meaning": "P99 end-to-end request latency in seconds on the client side.",
            },
            "request_peak_memory_mb_*": {
                "source": "python/sglang/multimodal_gen/benchmarks/bench_serving.py::calculate_metrics()['peak_memory_mb_*']",
                "bottom_metric": "RequestFuncOutput.peak_memory_mb -> OutputBatch.peak_memory_mb -> GPUWorker.do_mem_analysis() -> torch.cuda.max_memory_reserved()/MiB",
                "meaning": "Per-request peak reserved GPU memory in MiB, aggregated across requests.",
            },
        },
        "init_profile": {
            "component_loaded_weight_file_size_gb": {
                "source": "Component-specific loader-resolved weight files summed by byte size during initialization.",
                "meaning": "Byte size of the actual checkpoint files selected by the loader for each component.",
            },
            "component_final_module_size_gb": {
                "source": "python/sglang/multimodal_gen/runtime/loader/utils.py::get_memory_usage_of_component(module)",
                "meaning": "Size of the final in-memory module object (parameters + buffers) after loading completes.",
            },
            "component_param_size_gb": {
                "source": "Alias of component_final_module_size_gb kept for backward compatibility.",
                "meaning": "Final in-memory component footprint in GB.",
            },
            "component_gpu_load_consumed_gb": {
                "source": "python/sglang/multimodal_gen/runtime/loader/component_loaders/component_loader.py::ComponentLoader.load()",
                "meaning": "GPU memory consumed while loading each component during init. This reflects the current offload/load-residency behavior, not the intrinsic parameter size.",
            },
            "component_weight_profile_gb": {
                "source": "Alias of component_gpu_load_consumed_gb kept for backward compatibility.",
                "meaning": "Same value as component_gpu_load_consumed_gb. Prefer component_param_size_gb when you need the intrinsic component weight size.",
            },
            "stage_component_map": {
                "source": "GPUWorker.finalize_init_profile_after_startup_warmup()",
                "meaning": "Per-stage component attribution map derived from the actual stage instances built by SGLang. Shared components can appear in multiple stages, so these stage mappings should not be summed directly.",
            },
            "stage_component_param_size_gb": {
                "source": "Derived in scripts/profile_wan22_ti2v_5b_monolithic.py from stage_component_map + component_final_module_size_gb",
                "meaning": "Heuristic per-stage final-module footprint sum in GB. Shared components can appear in multiple stages, so values across stages should not be summed.",
            },
            "stage_component_loaded_weight_file_size_gb": {
                "source": "Derived in scripts/profile_wan22_ti2v_5b_monolithic.py from stage_component_map + component_loaded_weight_file_size_gb",
                "meaning": "Heuristic per-stage sum of actual loaded checkpoint-file sizes in GB. Shared components can appear in multiple stages, so values across stages should not be summed.",
            },
            "parameter_reserved_mb": {
                "formula": "after_build_pipeline.reserved_mb - before_build_pipeline.reserved_mb",
                "meaning": "Reserved memory attributed to model/pipeline construction.",
            },
            "parameter_allocated_mb": {
                "formula": "after_build_pipeline.allocated_mb - before_build_pipeline.allocated_mb",
                "meaning": "Allocated memory attributed to model/pipeline construction.",
            },
            "warmup_persistent_reserved_mb": {
                "formula": "after_startup_warmup.reserved_mb - after_build_pipeline.reserved_mb",
                "meaning": "Persistent reserved memory added by startup warmup. In this diffusion runtime it is a warmup-time persistent delta, not an isolated graph-only allocator probe.",
            },
            "warmup_peak_reserved_mb": {
                "source": "after_startup_warmup.peak_reserved_mb",
                "meaning": "Peak reserved memory observed during startup warmup after reset_peak_memory_stats().",
            },
            "warmup_peak_allocated_mb": {
                "source": "after_startup_warmup.peak_allocated_mb",
                "meaning": "Peak allocated memory observed during startup warmup after reset_peak_memory_stats().",
            },
            "runtime_transient_peak_reserved_mb": {
                "formula": "after_startup_warmup.peak_reserved_mb - after_startup_warmup.reserved_mb",
                "meaning": "Transient reserved-memory headroom seen during startup warmup above the final persistent footprint.",
            },
            "runtime_transient_peak_allocated_mb": {
                "formula": "after_startup_warmup.peak_allocated_mb - after_startup_warmup.allocated_mb",
                "meaning": "Transient allocated-memory headroom seen during startup warmup above the final persistent footprint.",
            },
        },
        "stage_level": {
            "memory_snapshot_semantics": {
                "allocated_mb": "Sampled immediately after a stage finishes. This is current live tensor memory after the stage, not before the stage starts.",
                "reserved_mb": "Sampled immediately after a stage finishes. This is current allocator-reserved VRAM after the stage, not before the stage starts.",
                "peak_allocated_mb": "Peak allocated memory since the request began (after the request-level reset_peak_memory_stats()), sampled after the stage finishes. It is cumulative up to and including that stage.",
                "peak_reserved_mb": "Peak reserved memory since the request began (after the request-level reset_peak_memory_stats()), sampled after the stage finishes. It is cumulative up to and including that stage.",
                "can_subtract_allocated_from_reserved": "No. reserved_mb - allocated_mb is allocator slack/cache at the sampling point, not a clean decomposition of runtime+others.",
                "can_subtract_component_weight_from_stage_peak": "Only as a rough heuristic. Stage peak metrics already include request baseline plus cumulative runtime effects up to that stage, so subtraction does not isolate a pure runtime-only term.",
            },
            "before_forward_*": {
                "source": "performance.log -> memory_snapshots['before_forward']",
                "meaning": "Per-request baseline captured immediately before pipeline.forward() starts. The first-request values show the reserved/allocated state before the first profiled request is served.",
            },
            "stage_duration_ms_*": {
                "source": "performance.log -> stages[].execution_time_ms",
                "meaning": "High-level stage execution time in milliseconds, aggregated across requests.",
            },
            "stage_peak_reserved_mb_*": {
                "source": "performance.log -> memory_snapshots['after_<stage>'].peak_reserved_mb",
                "meaning": "Request-local cumulative peak reserved memory sampled immediately after each high-level stage. Because peak stats are request-local cumulative values, they represent peak observed up to and including that stage.",
            },
            "stage_peak_allocated_mb_*": {
                "source": "performance.log -> memory_snapshots['after_<stage>'].peak_allocated_mb",
                "meaning": "Request-local cumulative peak allocated memory sampled immediately after each high-level stage.",
            },
            "stage_rough_runtime_like_*": {
                "source": "Derived in scripts/profile_wan22_ti2v_5b_monolithic.py",
                "formula": "stage_peak_*_gb_max - stage_component_final_module_size_gb, clamped at 0",
                "meaning": "A heuristic runtime-like upper-bound indicator based on final in-memory module size. This is not a strict runtime-only measurement because stage peaks are cumulative and include allocator effects.",
            },
            "stage_rough_runtime_like_*_from_loaded_files": {
                "source": "Derived in scripts/profile_wan22_ti2v_5b_monolithic.py",
                "formula": "stage_peak_*_gb_max - stage_component_loaded_weight_file_size_gb, clamped at 0",
                "meaning": "A heuristic runtime-like upper-bound indicator using actual loaded checkpoint-file size as the weight estimate.",
            },
        },
        "denoiser_step_level": {
            "denoiser_step_peak_reserved_mb_*": {
                "source": "performance.log -> memory_snapshots['after_denoising_step_i'].peak_reserved_mb",
                "formula": "For each request, take max over all denoising-step snapshots; then aggregate across requests.",
                "meaning": "Peak reserved memory reached during the denoising loop.",
            },
            "denoiser_step_peak_allocated_mb_*": {
                "source": "performance.log -> memory_snapshots['after_denoising_step_i'].peak_allocated_mb",
                "formula": "For each request, take max over all denoising-step snapshots; then aggregate across requests.",
                "meaning": "Peak allocated memory reached during the denoising loop.",
            },
            "denoiser_step_snapshot_count_*": {
                "source": "Count of 'after_denoising_step_i' snapshots in performance.log for each request.",
                "meaning": "How many denoising-step snapshots were observed per request.",
            },
        },
    }


def safe_rmtree(path: Path) -> None:
    logger.info("Cleaning temporary artifacts: %s", path)
    shutil.rmtree(path, ignore_errors=True)


def build_base_summary(
    *,
    args: argparse.Namespace,
    input_image: Path,
    sampling: Wan2_2_TI2V_5B_SamplingParam,
    parallel_degree: int,
    visible_gpu_count: int,
) -> dict[str, Any]:
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model_path,
        "model_id": args.model_id,
        "expected_task_type": EXPECTED_TASK_TYPE,
        "sampling_params": {
            "height": sampling.height,
            "width": sampling.width,
            "num_frames": sampling.num_frames,
            "fps": sampling.fps,
            "guidance_scale": sampling.guidance_scale,
            "num_inference_steps": sampling.num_inference_steps,
            "negative_prompt": getattr(sampling, "negative_prompt", None),
        },
        "request_setup": {
            "prompt": args.prompt,
            "input_image": str(input_image),
            "dataset_mode": "VBench single-image directory repeated to num_requests",
            "num_requests": args.num_requests,
            "probe_runs": args.probe_runs,
            "online_rps": args.online_rps,
            "run_serving_phases": args.run_serving_phases,
            "runtime_backend": DEFAULT_RUNTIME_BACKEND,
            "runtime_backend_reason": DEFAULT_RUNTIME_BACKEND_REASON,
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
        "runtime_backend": DEFAULT_RUNTIME_BACKEND,
        "runtime_backend_reason": DEFAULT_RUNTIME_BACKEND_REASON,
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


def build_human_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    requested = run.get("requested_parallelism", {}) or {}
    resolved = run.get("resolved_parallelism", {}) or {}
    probe = run.get("probe", {}) or {}
    init_profile = run.get("init_profile", {}) or {}
    offline = run.get("offline_burst", {}) or {}
    online = run.get("online", {}) or {}

    human_run = {
        "name": requested.get("name"),
        "effective_pipeline": run.get("effective_pipeline", {}),
        "parallelism": {
            "tp": resolved.get("tp_size", requested.get("tp_size")),
            "sp": resolved.get("sp_degree", requested.get("sp_degree")),
            "ulysses": resolved.get(
                "ulysses_degree",
                requested.get("ulysses_degree") or 1,
            ),
            "ring": resolved.get("ring_degree", requested.get("ring_degree") or 1),
            "num_gpus": resolved.get("num_gpus", requested.get("num_gpus")),
        },
        "runtime_backend": run.get("runtime_backend"),
        "init_memory_gb": {
            "parameter_reserved_gb": init_profile.get("parameter_reserved_gb"),
            "parameter_allocated_gb": init_profile.get("parameter_allocated_gb"),
            "warmup_persistent_reserved_gb": init_profile.get(
                "warmup_persistent_reserved_gb"
            ),
            "warmup_peak_reserved_gb": init_profile.get("warmup_peak_reserved_gb"),
            "warmup_peak_allocated_gb": init_profile.get("warmup_peak_allocated_gb"),
            "runtime_transient_peak_reserved_gb": init_profile.get(
                "runtime_transient_peak_reserved_gb"
            ),
            "runtime_transient_peak_allocated_gb": init_profile.get(
                "runtime_transient_peak_allocated_gb"
            ),
        },
        "component_loaded_weight_file_size_gb": init_profile.get(
            "component_loaded_weight_file_size_gb", {}
        ),
        "component_final_module_size_gb": init_profile.get(
            "component_final_module_size_gb",
            init_profile.get("component_param_size_gb", {}),
        ),
        "component_param_size_gb": init_profile.get("component_param_size_gb", {}),
        "component_gpu_load_consumed_gb": init_profile.get(
            "component_gpu_load_consumed_gb",
            init_profile.get("component_weight_profile_gb", {}),
        ),
        "component_weight_profile_gb": init_profile.get("component_weight_profile_gb", {}),
        "stage_component_map": init_profile.get("stage_component_map", {}),
        "probe": {
            "num_runs": probe.get("num_runs"),
            "total_duration_mean_s": probe.get("total_duration_mean_s"),
            "request_peak_memory_gb_mean": probe.get("request_peak_memory_gb_mean"),
            "request_peak_memory_gb_median": probe.get(
                "request_peak_memory_gb_median"
            ),
            "request_peak_memory_gb_max": probe.get("request_peak_memory_gb_max"),
            "first_request_pre_forward_reserved_gb": probe.get(
                "first_request_pre_forward_reserved_gb"
            ),
            "first_request_pre_forward_allocated_gb": probe.get(
                "first_request_pre_forward_allocated_gb"
            ),
            "denoiser_step_peak_reserved_gb_max": probe.get(
                "denoiser_step_peak_reserved_gb_max"
            ),
            "denoiser_step_peak_allocated_gb_max": probe.get(
                "denoiser_step_peak_allocated_gb_max"
            ),
            "stage_time_mean_s": probe.get("stage_time_mean_s", {}),
            "stage_peak_reserved_gb_max": probe.get("stage_peak_reserved_gb_max", {}),
            "stage_peak_allocated_gb_max": probe.get(
                "stage_peak_allocated_gb_max", {}
            ),
            "stage_component_param_size_gb": probe.get(
                "stage_component_param_size_gb", {}
            ),
            "stage_component_loaded_weight_file_size_gb": probe.get(
                "stage_component_loaded_weight_file_size_gb", {}
            ),
            "stage_component_final_module_size_gb": probe.get(
                "stage_component_final_module_size_gb", {}
            ),
            "stage_rough_runtime_like_reserved_gb_max": probe.get(
                "stage_rough_runtime_like_reserved_gb_max", {}
            ),
            "stage_rough_runtime_like_allocated_gb_max": probe.get(
                "stage_rough_runtime_like_allocated_gb_max", {}
            ),
            "stage_rough_runtime_like_reserved_gb_max_from_loaded_files": probe.get(
                "stage_rough_runtime_like_reserved_gb_max_from_loaded_files", {}
            ),
            "stage_rough_runtime_like_allocated_gb_max_from_loaded_files": probe.get(
                "stage_rough_runtime_like_allocated_gb_max_from_loaded_files", {}
            ),
            "memory_snapshot_capture_points": probe.get(
                "memory_snapshot_capture_points", {}
            ),
            "memory_snapshot_semantics": {
                "sampling_point": "All stage memory snapshots are sampled immediately after the stage finishes.",
                "peak_scope": "peak_* values are cumulative from request start to the end of that stage.",
                "note": "reserved-allocated is allocator slack/cache, not a pure runtime+others term.",
            },
        },
    }

    if not offline.get("skipped", False):
        human_run["offline_burst"] = {
            "throughput_qps": offline.get("throughput_qps"),
            "latency_mean_s": offline.get("latency_mean_s", offline.get("latency_mean")),
            "latency_median_s": offline.get(
                "latency_median_s", offline.get("latency_median")
            ),
            "latency_p99_s": offline.get("latency_p99_s", offline.get("latency_p99")),
            "request_peak_memory_gb_max": offline.get("request_peak_memory_gb_max"),
        }
    else:
        human_run["offline_burst"] = {
            "skipped": True,
            "reason": offline.get("reason"),
        }

    if not online.get("skipped", False):
        human_run["online"] = {
            "throughput_qps": online.get("throughput_qps"),
            "latency_mean_s": online.get("latency_mean_s", online.get("latency_mean")),
            "latency_median_s": online.get(
                "latency_median_s", online.get("latency_median")
            ),
            "latency_p99_s": online.get("latency_p99_s", online.get("latency_p99")),
            "request_peak_memory_gb_max": online.get("request_peak_memory_gb_max"),
        }
    else:
        human_run["online"] = {
            "skipped": True,
            "reason": online.get("reason"),
        }

    return human_run


def refresh_human_summary(summary: dict[str, Any]) -> None:
    summary["human_summary"] = {
        "model": summary.get("model"),
        "model_id": summary.get("model_id"),
        "parallel_degree": summary.get("parallel_degree"),
        "graph_mode": summary.get("graph_mode"),
        "runtime_backend": summary.get("runtime_backend"),
        "request_setup": {
            "prompt": summary.get("request_setup", {}).get("prompt"),
            "input_image": summary.get("request_setup", {}).get("input_image"),
            "probe_runs": summary.get("request_setup", {}).get("probe_runs"),
            "run_serving_phases": summary.get("request_setup", {}).get(
                "run_serving_phases"
            ),
        },
        "completed_runs": [
            build_human_run_summary(run) for run in summary.get("runs", [])
        ],
        "skipped_configs": [
            {
                "phase": item.get("phase"),
                "name": (item.get("requested_parallelism") or {}).get("name"),
                "reason": item.get("reason"),
            }
            for item in summary.get("skipped_configs", [])
        ],
    }


def run_single_config(
    *,
    model_path: str,
    dataset_path: Path,
    prompt: str,
    sampling: Wan2_2_TI2V_5B_SamplingParam,
    run_config: RunConfig,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    process = None
    try:
        try:
            logger.info(
                "[%s] launching server and running startup warmup (stage profiling disabled during warmup)",
                run_config.name,
            )
            process, base_url, perf_dir, init_profile_path, served_model_card = launch_server(
                model_path=model_path,
                model_id=args.model_id,
                run_config=run_config,
                sampling=sampling,
                run_dir=run_dir,
                host=args.host,
                timeout_s=args.wait_timeout,
                trust_remote_code=args.trust_remote_code,
            )
        except Exception as exc:
            raise RunConfigError("launch", str(exc)) from exc
        logger.info(
            "[%s] server ready; startup warmup finished and init profile is available at %s",
            run_config.name,
            init_profile_path,
        )

        logger.info(
            "[%s] building repeated request set from dataset %s",
            run_config.name,
            dataset_path,
        )
        requests_list = build_requests(
            dataset_path=dataset_path,
            base_url=base_url,
            model_path=model_path,
            prompt=prompt,
            sampling=sampling,
            num_prompts=args.num_requests,
        )
        logger.info(
            "[%s] request set ready: probe_runs=%s measured_requests=%s",
            run_config.name,
            args.probe_runs,
            len(requests_list),
        )
        perf_log_path = perf_dir / "performance.log"

        try:
            logger.info(
                "[%s] starting probe phase: %s run(s), profiling enabled",
                run_config.name,
                args.probe_runs,
            )
            clear_perf_log(perf_log_path, f"{run_config.name}:probe")
            probe_outputs, _ = asyncio.run(
                run_probe_phase(
                    requests_list,
                    probe_runs=args.probe_runs,
                    phase_name=f"{run_config.name}:probe",
                )
            )
            ensure_probe_successful(probe_outputs, args.probe_runs)
            probe_records = wait_for_perf_records(
                perf_log_path,
                args.probe_runs,
                args.wait_timeout,
                f"{run_config.name}:probe",
            )
            probe_summary = aggregate_probe(probe_records, probe_outputs)
            logger.info(
                "[%s] first-request pre-forward baseline: reserved=%.2f GB allocated=%.2f GB",
                run_config.name,
                float(probe_summary.get("first_request_pre_forward_reserved_gb", 0.0)),
                float(
                    probe_summary.get("first_request_pre_forward_allocated_gb", 0.0)
                ),
            )
            logger.info("[%s] probe phase finished successfully", run_config.name)
        except RunConfigError:
            raise
        except Exception as exc:
            raise RunConfigError("probe", str(exc)) from exc

        offline_summary: dict[str, Any] = {
            "skipped": True,
            "reason": "Serving benchmarks are disabled by default. Pass --run-serving-phases to enable offline and online benchmarking.",
        }
        online_summary: dict[str, Any] = {
            "skipped": True,
            "reason": "Serving benchmarks are disabled by default. Pass --run-serving-phases to enable offline and online benchmarking.",
        }

        if args.run_serving_phases:
            try:
                logger.info(
                    "[%s] starting offline phase: %s request(s), request_rate=inf, max_concurrency=%s",
                    run_config.name,
                    args.num_requests,
                    args.num_requests,
                )
                clear_perf_log(perf_log_path, f"{run_config.name}:offline")
                offline_outputs, offline_duration = asyncio.run(
                    run_serving_phase(
                        requests_list,
                        request_rate=float("inf"),
                        max_concurrency=args.num_requests,
                        seed=42,
                        phase_name=f"{run_config.name}:offline",
                    )
                )
                offline_metrics = calculate_metrics(
                    offline_outputs,
                    offline_duration,
                    requests_list,
                    SimpleNamespace(slo_scale=3.0),
                    slo_enabled=False,
                )
                ensure_successful_requests(
                    "offline_burst", offline_metrics, args.num_requests
                )
                offline_records = wait_for_perf_records(
                    perf_log_path,
                    args.num_requests,
                    args.wait_timeout,
                    f"{run_config.name}:offline",
                )
                offline_summary = {
                    **benchmark_outputs_to_summary(offline_metrics),
                    **aggregate_stage_metrics(offline_records),
                }
                logger.info("[%s] offline phase finished successfully", run_config.name)
            except RunConfigError:
                raise
            except Exception as exc:
                raise RunConfigError("offline_burst", str(exc)) from exc

            try:
                logger.info(
                    "[%s] starting online phase: %s request(s), request_rate=%s, max_concurrency=%s",
                    run_config.name,
                    args.num_requests,
                    args.online_rps,
                    args.num_requests,
                )
                clear_perf_log(perf_log_path, f"{run_config.name}:online")
                online_outputs, online_duration = asyncio.run(
                    run_serving_phase(
                        requests_list,
                        request_rate=args.online_rps,
                        max_concurrency=args.num_requests,
                        seed=43,
                        phase_name=f"{run_config.name}:online",
                    )
                )
                online_metrics = calculate_metrics(
                    online_outputs,
                    online_duration,
                    requests_list,
                    SimpleNamespace(slo_scale=3.0),
                    slo_enabled=False,
                )
                ensure_successful_requests("online", online_metrics, args.num_requests)
                online_records = wait_for_perf_records(
                    perf_log_path,
                    args.num_requests,
                    args.wait_timeout,
                    f"{run_config.name}:online",
                )
                online_summary = {
                    **benchmark_outputs_to_summary(online_metrics),
                    **aggregate_stage_metrics(online_records),
                }
                logger.info("[%s] online phase finished successfully", run_config.name)
            except RunConfigError:
                raise
            except Exception as exc:
                raise RunConfigError("online", str(exc)) from exc
        else:
            logger.info(
                "[%s] skipping offline/online serving phases because --run-serving-phases is false",
                run_config.name,
            )

        try:
            logger.info(
                "[%s] loading init memory snapshot from %s",
                run_config.name,
                init_profile_path,
            )
            init_profile = augment_init_profile_units(load_json(init_profile_path))
            probe_summary = attach_stage_runtime_heuristics(probe_summary, init_profile)
            if not offline_summary.get("skipped", False):
                offline_summary = attach_stage_runtime_heuristics(
                    offline_summary, init_profile
                )
            if not online_summary.get("skipped", False):
                online_summary = attach_stage_runtime_heuristics(
                    online_summary, init_profile
                )
        except Exception as exc:
            raise RunConfigError("init_profile", str(exc)) from exc

        return {
            "runtime_backend": DEFAULT_RUNTIME_BACKEND,
            "runtime_backend_reason": DEFAULT_RUNTIME_BACKEND_REASON,
            "requested_parallelism": asdict(run_config),
            "resolved_parallelism": build_resolved_parallelism(init_profile),
            "effective_pipeline": {
                "pipeline_class": init_profile.get("pipeline_class"),
                "pipeline_name": init_profile.get("pipeline_name"),
                "task_type": init_profile.get("task_type"),
            },
            "served_model_card": served_model_card,
            "init_profile": init_profile,
            "probe": probe_summary,
            "offline_burst": offline_summary,
            "online": online_summary,
        }
    finally:
        stop_server(process)


def main() -> None:
    args = parse_args()
    input_image = Path(args.input_image).expanduser().resolve()
    if not input_image.exists():
        raise FileNotFoundError(f"Input image not found: {input_image}")

    parallel_degrees = parse_parallel_degrees(args)
    visible_gpu_count = resolve_visible_gpu_count()
    sampling = build_sampling_config()
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    for parallel_degree in parallel_degrees:
        summary_path = output_dir / f"gpu{parallel_degree}_summary.json"
        degree_tmp_root = output_dir / ".tmp" / f"gpu{parallel_degree}"
        degree_tmp_root.mkdir(parents=True, exist_ok=True)
        dataset_path = build_dataset_dir(input_image, args.prompt, degree_tmp_root)
        summary = build_base_summary(
            args=args,
            input_image=input_image,
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
            save_json(summary_path, summary)
            if not args.keep_artifacts:
                safe_rmtree(degree_tmp_root)
            continue

        run_configs = build_run_configs(parallel_degree)
        logger.info(
            "gpu=%s will profile %s unique tp*sp=P configuration(s)",
            parallel_degree,
            len(run_configs),
        )
        total_configs = len(run_configs)
        for config_idx, run_config in enumerate(run_configs, start=1):
            logger.info(
                "Profiling gpu=%s config=%s (%s/%s)",
                parallel_degree,
                run_config.name,
                config_idx,
                total_configs,
            )
            run_dir = degree_tmp_root / run_config.name
            try:
                summary["runs"].append(
                    run_single_config(
                        model_path=args.model_path,
                        dataset_path=dataset_path,
                        prompt=args.prompt,
                        sampling=sampling,
                        run_config=run_config,
                        args=args,
                        run_dir=run_dir,
                    )
                )
            except RunConfigError as exc:
                summary["skipped_configs"].append(
                    {
                        "phase": exc.phase,
                        "requested_parallelism": asdict(run_config),
                        "reason": str(exc),
                    }
                )
                logger.warning(
                    "Skipping gpu=%s config=%s (%s/%s) after %s failure and continuing to next config: %s",
                    parallel_degree,
                    run_config.name,
                    config_idx,
                    total_configs,
                    exc.phase,
                    exc,
                )
                continue
            except Exception as exc:
                summary["skipped_configs"].append(
                    {
                        "phase": "unknown",
                        "requested_parallelism": asdict(run_config),
                        "reason": str(exc),
                    }
                )
                logger.exception(
                    "Skipping gpu=%s config=%s (%s/%s) after unexpected failure and continuing to next config",
                    parallel_degree,
                    run_config.name,
                    config_idx,
                    total_configs,
                )
                continue
            finally:
                refresh_human_summary(summary)
                save_json(summary_path, summary)
                if not args.keep_artifacts:
                    safe_rmtree(run_dir)

        refresh_human_summary(summary)
        save_json(summary_path, summary)
        logger.info("Summary written to %s", summary_path.resolve())
        if not args.keep_artifacts:
            safe_rmtree(degree_tmp_root)


if __name__ == "__main__":
    main()
