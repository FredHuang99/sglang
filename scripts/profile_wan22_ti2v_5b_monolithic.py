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
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXAMPLE_IMAGE = REPO_ROOT / "examples" / "assets" / "example_image.png"
DEFAULT_TI2V_PROMPT = "The girl turn the body and spin around in place."


@dataclass(frozen=True)
class RunConfig:
    name: str
    mode: str
    num_gpus: int
    tp_size: int
    sp_degree: int
    ulysses_degree: int
    ring_degree: int


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
        default="1", #"1,2,4,8",
        help="Comma/space separated total GPU counts, e.g. '1,2,4,8'.",
    )
    parser.add_argument(
        "--parallel-degree",
        type=int,
        default=None,
        help="Deprecated single-value override for --parallel-degrees.",
    )
    parser.add_argument(
        "--include-hybrid-tp-sp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also try mixed configs where tp=P and sp=P; failures are skipped.",
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
    if args.parallel_degree is not None:
        return [args.parallel_degree]

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
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Server exited early with code {process.returncode}.\n"
                f"{tail_text(server_log_path)}"
            )
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200 and init_profile_path.exists():
                return
        except requests.RequestException:
            pass
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


def build_run_configs(
    parallel_degree: int, include_hybrid_tp_sp: bool
) -> list[RunConfig]:
    runs: list[RunConfig] = []
    seen: set[tuple[int, int, int, int]] = set()

    def add_run(run: RunConfig) -> None:
        key = (
            run.tp_size,
            run.sp_degree,
            run.ulysses_degree,
            run.ring_degree,
        )
        if key in seen:
            return
        seen.add(key)
        runs.append(run)

    add_run(
        RunConfig(
            name=f"tp{parallel_degree}_sp1_u1_r1",
            mode="tp",
            num_gpus=parallel_degree,
            tp_size=parallel_degree,
            sp_degree=1,
            ulysses_degree=1,
            ring_degree=1,
        )
    )
    for ulysses_degree, ring_degree in factor_pairs(parallel_degree):
        add_run(
            RunConfig(
                name=(
                    f"tp1_sp{parallel_degree}_u{ulysses_degree}_r{ring_degree}"
                ),
                mode="sp",
                num_gpus=parallel_degree,
                tp_size=1,
                sp_degree=parallel_degree,
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
            )
        )
    if include_hybrid_tp_sp:
        for ulysses_degree, ring_degree in factor_pairs(parallel_degree):
            add_run(
                RunConfig(
                    name=(
                        f"tp{parallel_degree}_sp{parallel_degree}_u"
                        f"{ulysses_degree}_r{ring_degree}"
                    ),
                    mode="hybrid",
                    num_gpus=parallel_degree,
                    tp_size=parallel_degree,
                    sp_degree=parallel_degree,
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
) -> tuple[list[Any], float]:
    semaphore = asyncio.Semaphore(max_concurrency)
    rng = np.random.default_rng(seed)

    async def limited_request(req, session):
        async with semaphore:
            return await async_request_video_sglang(req, session)

    async with aiohttp.ClientSession() as session:
        tasks = []
        start_time = time.perf_counter()
        for req in requests_list:
            if request_rate != float("inf"):
                interval = rng.exponential(1.0 / request_rate)
                await asyncio.sleep(interval)
            tasks.append(asyncio.create_task(limited_request(req, session)))
        outputs = await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - start_time
    return outputs, total_duration


def clear_perf_log(perf_log_path: Path) -> None:
    if perf_log_path.exists():
        perf_log_path.unlink()


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
    perf_log_path: Path, expected_count: int, timeout_s: int
) -> list[dict[str, Any]]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        records = read_perf_records(perf_log_path)
        if len(records) >= expected_count:
            return records
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


def summarize_series(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": round_float(float(np.mean(values))),
        "median": round_float(float(np.median(values))),
        "p99": round_float(percentile(values, 99)),
        "max": round_float(max(values)),
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
        "denoiser_step_peak_allocated_mb_mean": allocated_summary["mean"],
        "denoiser_step_peak_allocated_mb_median": allocated_summary["median"],
        "denoiser_step_peak_allocated_mb_p99": allocated_summary["p99"],
        "denoiser_step_peak_allocated_mb_max": allocated_summary["max"],
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
    for stage_name, values in sorted(durations.items()):
        stage_duration_ms_mean[stage_name] = round_float(float(np.mean(values)))
        stage_duration_ms_median[stage_name] = round_float(float(np.median(values)))
        stage_duration_ms_p99[stage_name] = round_float(percentile(values, 99))
        stage_duration_ms_max[stage_name] = round_float(max(values))

    stage_peak_reserved_mb_mean: dict[str, float] = {}
    stage_peak_reserved_mb_median: dict[str, float] = {}
    stage_peak_reserved_mb_max: dict[str, float] = {}
    for stage_name, values in sorted(peak_reserved.items()):
        stage_peak_reserved_mb_mean[stage_name] = round_float(float(np.mean(values)))
        stage_peak_reserved_mb_median[stage_name] = round_float(
            float(np.median(values))
        )
        stage_peak_reserved_mb_max[stage_name] = round_float(max(values))

    stage_peak_allocated_mb_mean: dict[str, float] = {}
    stage_peak_allocated_mb_median: dict[str, float] = {}
    stage_peak_allocated_mb_max: dict[str, float] = {}
    for stage_name, values in sorted(peak_allocated.items()):
        stage_peak_allocated_mb_mean[stage_name] = round_float(
            float(np.mean(values))
        )
        stage_peak_allocated_mb_median[stage_name] = round_float(
            float(np.median(values))
        )
        stage_peak_allocated_mb_max[stage_name] = round_float(max(values))

    return {
        "records_count": len(records),
        "stage_duration_ms_mean": stage_duration_ms_mean,
        "stage_duration_ms_median": stage_duration_ms_median,
        "stage_duration_ms_p99": stage_duration_ms_p99,
        "stage_duration_ms_max": stage_duration_ms_max,
        "stage_peak_reserved_mb_mean": stage_peak_reserved_mb_mean,
        "stage_peak_reserved_mb_median": stage_peak_reserved_mb_median,
        "stage_peak_reserved_mb_max": stage_peak_reserved_mb_max,
        "stage_peak_allocated_mb_mean": stage_peak_allocated_mb_mean,
        "stage_peak_allocated_mb_median": stage_peak_allocated_mb_median,
        "stage_peak_allocated_mb_max": stage_peak_allocated_mb_max,
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
    return {
        "num_runs": len(records),
        "stage_time_mean_ms": {
            stage_name: round_float(float(np.mean(values)))
            for stage_name, values in sorted(stage_values.items())
        },
        "total_duration_mean_ms": round_float(float(np.mean(total_durations)))
        if total_durations
        else 0.0,
        "request_peak_memory_mb_mean": request_peak_summary["mean"],
        "request_peak_memory_mb_median": request_peak_summary["median"],
        "request_peak_memory_mb_max": request_peak_summary["max"],
        **aggregate_denoiser_step_memory(records),
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        "completed_requests": int(metrics.get("completed_requests", 0)),
        "failed_requests": int(metrics.get("failed_requests", 0)),
        "throughput_qps": round_float(metrics.get("throughput_qps", 0.0)),
        "latency_mean": round_float(metrics.get("latency_mean", 0.0)),
        "latency_median": round_float(metrics.get("latency_median", 0.0)),
        "latency_p99": round_float(metrics.get("latency_p99", 0.0)),
        "request_peak_memory_mb_max": round_float(
            metrics.get("peak_memory_mb_max", 0.0)
        ),
        "request_peak_memory_mb_mean": round_float(
            metrics.get("peak_memory_mb_mean", 0.0)
        ),
        "request_peak_memory_mb_median": round_float(
            metrics.get("peak_memory_mb_median", 0.0)
        ),
    }


def build_server_command(
    *,
    model_path: str,
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
        "--tp-size",
        str(run_config.tp_size),
        "--sp-degree",
        str(run_config.sp_degree),
        "--ulysses-degree",
        str(run_config.ulysses_degree),
        "--ring-degree",
        str(run_config.ring_degree),
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
    ]
    if trust_remote_code:
        command.append("--trust-remote-code")
    return command


def launch_server(
    *,
    model_path: str,
    run_config: RunConfig,
    sampling: Wan2_2_TI2V_5B_SamplingParam,
    run_dir: Path,
    host: str,
    timeout_s: int,
    trust_remote_code: bool,
) -> tuple[subprocess.Popen, str, Path, Path]:
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

    command = build_server_command(
        model_path=model_path,
        run_config=run_config,
        sampling=sampling,
        host=host,
        port=port,
        scheduler_port=scheduler_port,
        master_port=master_port,
        output_dir=run_dir,
        trust_remote_code=trust_remote_code,
    )
    logger.info("Launching %s", " ".join(command))
    log_fh = server_log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        wait_for_server_ready(
            process=process,
            base_url=base_url,
            init_profile_path=init_profile_path,
            server_log_path=server_log_path,
            timeout_s=timeout_s,
        )
    except Exception:
        log_fh.flush()
        log_fh.close()
        kill_process_tree(process.pid)
        raise
    process._sgl_log_fh = log_fh  # type: ignore[attr-defined]
    return process, base_url, perf_dir, init_profile_path


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
) -> tuple[list[Any], float]:
    probe_requests = [replace(requests_list[0]) for _ in range(probe_runs)]
    return await execute_requests(
        probe_requests,
        request_rate=float("inf"),
        max_concurrency=1,
    )


async def run_serving_phase(
    requests_list: list[RequestFuncInput],
    request_rate: float,
    max_concurrency: int,
    seed: int,
) -> tuple[list[Any], float]:
    return await execute_requests(
        requests_list,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        seed=seed,
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
        "request_level": {
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
            "warmup_enabled": True,
            "warmup_resolutions": [f"{sampling.width}x{sampling.height}"],
            "warmup_steps": 1,
        },
        "parallel_degree": parallel_degree,
        "graph_mode": "regular",
        "visible_gpu_count": visible_gpu_count,
        "include_hybrid_tp_sp": args.include_hybrid_tp_sp,
        "keep_artifacts": args.keep_artifacts,
        "metric_definitions": build_metric_definitions(),
        "runs": [],
        "skipped_configs": [],
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
            process, base_url, perf_dir, init_profile_path = launch_server(
                model_path=model_path,
                run_config=run_config,
                sampling=sampling,
                run_dir=run_dir,
                host=args.host,
                timeout_s=args.wait_timeout,
                trust_remote_code=args.trust_remote_code,
            )
        except Exception as exc:
            raise RunConfigError("launch", str(exc)) from exc

        requests_list = build_requests(
            dataset_path=dataset_path,
            base_url=base_url,
            model_path=model_path,
            prompt=prompt,
            sampling=sampling,
            num_prompts=args.num_requests,
        )
        perf_log_path = perf_dir / "performance.log"

        try:
            clear_perf_log(perf_log_path)
            probe_outputs, _ = asyncio.run(
                run_probe_phase(requests_list, probe_runs=args.probe_runs)
            )
            ensure_probe_successful(probe_outputs, args.probe_runs)
            probe_records = wait_for_perf_records(
                perf_log_path, args.probe_runs, args.wait_timeout
            )
            probe_summary = aggregate_probe(probe_records, probe_outputs)
        except RunConfigError:
            raise
        except Exception as exc:
            raise RunConfigError("probe", str(exc)) from exc

        try:
            clear_perf_log(perf_log_path)
            offline_outputs, offline_duration = asyncio.run(
                run_serving_phase(
                    requests_list,
                    request_rate=float("inf"),
                    max_concurrency=args.num_requests,
                    seed=42,
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
                perf_log_path, args.num_requests, args.wait_timeout
            )
        except RunConfigError:
            raise
        except Exception as exc:
            raise RunConfigError("offline_burst", str(exc)) from exc

        try:
            clear_perf_log(perf_log_path)
            online_outputs, online_duration = asyncio.run(
                run_serving_phase(
                    requests_list,
                    request_rate=args.online_rps,
                    max_concurrency=args.num_requests,
                    seed=43,
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
                perf_log_path, args.num_requests, args.wait_timeout
            )
        except RunConfigError:
            raise
        except Exception as exc:
            raise RunConfigError("online", str(exc)) from exc

        try:
            init_profile = load_json(init_profile_path)
        except Exception as exc:
            raise RunConfigError("init_profile", str(exc)) from exc

        return {
            "parallelism": asdict(run_config),
            "init_profile": init_profile,
            "probe": probe_summary,
            "offline_burst": {
                **benchmark_outputs_to_summary(offline_metrics),
                **aggregate_stage_metrics(offline_records),
            },
            "online": {
                **benchmark_outputs_to_summary(online_metrics),
                **aggregate_stage_metrics(online_records),
            },
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
                    "parallelism": {"num_gpus": parallel_degree},
                    "reason": (
                        f"parallel_degree={parallel_degree} requires at least "
                        f"{parallel_degree} visible GPUs, but only "
                        f"{visible_gpu_count} are visible."
                    ),
                }
            )
            save_json(summary_path, summary)
            if not args.keep_artifacts:
                safe_rmtree(degree_tmp_root)
            continue

        run_configs = build_run_configs(parallel_degree, args.include_hybrid_tp_sp)
        for run_config in run_configs:
            logger.info("Profiling gpu=%s config=%s", parallel_degree, run_config.name)
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
                        "parallelism": asdict(run_config),
                        "reason": str(exc),
                    }
                )
            except Exception as exc:
                summary["skipped_configs"].append(
                    {
                        "phase": "unknown",
                        "parallelism": asdict(run_config),
                        "reason": str(exc),
                    }
                )
            finally:
                save_json(summary_path, summary)
                if not args.keep_artifacts:
                    safe_rmtree(run_dir)

        save_json(summary_path, summary)
        logger.info("Summary written to %s", summary_path.resolve())
        if not args.keep_artifacts:
            safe_rmtree(degree_tmp_root)


if __name__ == "__main__":
    main()
