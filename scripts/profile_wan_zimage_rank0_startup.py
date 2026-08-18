#!/usr/bin/env python3
"""Compare baseline and rank0-broadcast startup for Wan and Z-Image.

The official startup metric is measured from Popen to a validated /v1/models
response without launch-time weight profiling. A separate metrics pass checks
that rank0 is the only rank reading transformer and VAE weights.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import pprint
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import requests


SUPPORTED_GPU_COUNTS = (1, 2, 4, 8)
DEFAULT_NUM_RUNS = 5
DEFAULT_WARMUP_RUNS = 2
DEFAULT_METRICS_RUNS = 1
MAX_PORT_LAUNCH_ATTEMPTS = 5
SETUPS = ("baseline", "rank0_broadcast_pageable")
OPTIMIZED_FLAGS = (
    "--diffusion-weight-staging",
    "pageable",
    "--diffusion-weight-load-mode",
    "rank0-broadcast",
    "--diffusion-weight-broadcast-components",
    "transformer,vae",
)
PROFILE_COMPONENTS = ("transformer", "vae")

TIME_FIELDS_MS = (
    "weight_load:discover_files_ms",
    "weight_load:read_safetensors_ms",
    "weight_load:cpu_materialize_ms",
    "weight_load:pin_memory_ms",
    "weight_load:h2d_or_param_copy_ms",
    "weight_load:d2h_or_offload_ms",
    "weight_load:nccl_broadcast_ms",
    "weight_load:rank0_wait_ms",
)
BYTE_FIELDS = (
    "weight_load:total_bytes",
    "weight_load:pinned_bytes",
    "weight_load:broadcast_bytes",
    "weight_load:warm_pool_store_bytes",
)
COUNT_FIELDS = (
    "weight_load:staged_tensor_count",
    "weight_load:pinned_tensor_count",
    "weight_load:broadcast_tensor_count",
)
ERROR_FIELDS = (
    "error",
    "weight_load:pin_memory_error",
    "weight_load:broadcast_error",
    "weight_load:warm_pool_error",
)

COMMON_ENV_OVERRIDES = {
    "SGLANG_CACHE_DIT_ENABLED": "false",
    "SGLANG_DIFFUSION_SYNC_STAGE_PROFILING": "0",
    "SGLANG_DIFFUSION_CUDA_EVENT_STAGE_PROFILING": "0",
    "SGLANG_DIFFUSION_STAGE_LOGGING": "0",
}
SETUP_ENV_OVERRIDES = {
    "baseline": {"SGLANG_USE_RUNAI_MODEL_STREAMER": "true"},
    "rank0_broadcast_pageable": {"SGLANG_USE_RUNAI_MODEL_STREAMER": "true"},
}
RUNAI_STREAM_RE = re.compile(
    r"\[RunAI Streamer\].*?stream\s+([0-9.]+)\s+GiB.*?:\s+([0-9.]+)s"
)

HISTORICAL_H200_REFERENCE = {
    "wan22_ti2v_5b": {
        "baseline_s": {1: 24.447095, 2: 28.080947, 4: 40.251256, 8: 61.020383},
        "rank0_broadcast_pageable_s": {
            1: 24.47,
            2: 26.28,
            4: 29.69,
            8: 37.61,
        },
    },
    "z_image": {
        "baseline_s": {1: 22.216599, 2: 28.225255, 4: 32.482700, 8: 43.77},
        "rank0_broadcast_pageable_s": {
            1: 22.439025,
            2: 25.534809,
            4: 28.016483,
            8: 36.36,
        },
    },
}

APRIL_REPRODUCTION_POLICY = {
    "exact": [
        "entrypoint=sglang.multimodal_gen.runtime.entrypoints.cli.main serve",
        "backend=sglang",
        "tp_size=1",
        "sp_degree=num_gpus",
        "model-specific ulysses_degree/ring_degree",
        "baseline process environment enables RunAI",
        "optimized process environment enables RunAI for non-broadcast loaders",
        "optimized transformer rank0 locally disables RunAI to avoid deadlock",
        "enable_cfg_parallel=false",
        "all component offload disabled",
        "use_fsdp_inference=false",
        "enable_torch_compile=false",
        "strict_ports=true",
    ],
    "equivalent": {
        "performance_mode=manual": (
            "omitted because this branch has no performance auto-tuner; all "
            "relevant settings are explicit"
        ),
        "cfg_parallel_size=1": (
            "omitted because this branch has no cfg_parallel_size argument; "
            "enable_cfg_parallel=false makes the size inactive"
        ),
        "warmup_mode=off": "mapped to warmup=false",
    },
    "unavailable_and_disabled": [
        "enable_breakable_cuda_graph=false",
        "enable_layerwise_nvtx_marker=false",
    ],
    "not_applicable": [
        "custom all-reduce v2 is not used by the April diffusion startup setup "
        "and tp_size is 1"
    ],
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_id: str
    default_path: Path
    task_type: str
    pipeline_class: str
    parallelism: dict[int, tuple[int, int]]


WAN22 = ModelSpec(
    key="wan22_ti2v_5b",
    label="Wan2.2-TI2V-5B",
    model_id="Wan2.2-TI2V-5B-Diffusers",
    default_path=Path("/workspace/models/Wan2.2-TI2V-5B-Diffusers"),
    task_type="TI2V",
    pipeline_class="WanPipeline",
    parallelism={1: (1, 1), 2: (2, 1), 4: (4, 1), 8: (8, 1)},
)
WAN21 = ModelSpec(
    key="wan21_t2v_1_3b",
    label="Wan2.1-T2V-1.3B",
    model_id="Wan2.1-T2V-1.3B-Diffusers",
    default_path=Path("/workspace/models/Wan2.1-T2V-1.3B-Diffusers"),
    task_type="T2V",
    pipeline_class="WanPipeline",
    parallelism={1: (1, 1), 2: (2, 1), 4: (4, 1), 8: (4, 2)},
)
Z_IMAGE = ModelSpec(
    key="z_image",
    label="Z-Image",
    model_id="Z-Image",
    default_path=Path("/workspace/models/Z-Image"),
    task_type="T2I",
    pipeline_class="ZImagePipeline",
    parallelism={1: (1, 1), 2: (2, 1), 4: (2, 2), 8: (2, 4)},
)
ALL_MODEL_SPECS = (WAN22, WAN21, Z_IMAGE)


@dataclass
class LaunchedServer:
    process: subprocess.Popen[bytes]
    log_file: BinaryIO
    log_path: Path
    command: list[str]
    started_ns: int


def parse_args() -> argparse.Namespace:
    default_output = Path("/workspace/outputs/wan_zimage_rank0_startup") / (
        time.strftime("%Y%m%d_%H%M%S")
    )
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce April/H200 fresh startup for Wan2.2, Wan2.1, and "
            "Z-Image with baseline versus pageable rank0 broadcast."
        )
    )
    parser.add_argument("--wan22-model-path", type=Path, default=WAN22.default_path)
    parser.add_argument("--wan21-model-path", type=Path, default=WAN21.default_path)
    parser.add_argument("--z-image-model-path", type=Path, default=Z_IMAGE.default_path)
    parser.add_argument("--gpu-counts", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--metrics-runs", type=int, default=DEFAULT_METRICS_RUNS)
    parser.add_argument("--server-timeout-s", type=float, default=3600.0)
    parser.add_argument("--ready-poll-interval-s", type=float, default=0.05)
    parser.add_argument("--shutdown-timeout-s", type=float, default=60.0)
    parser.add_argument("--cooldown-s", type=float, default=2.0)
    parser.add_argument("--python", default=sys.executable or "python3")
    args = parser.parse_args()

    args.gpu_counts = normalize_gpu_counts(args.gpu_counts)
    if args.num_runs <= 0:
        parser.error("--num-runs must be positive")
    if args.warmup_runs < 0 or args.warmup_runs >= args.num_runs:
        parser.error("--warmup-runs must be >= 0 and smaller than --num-runs")
    if args.metrics_runs <= 0:
        parser.error("--metrics-runs must be positive for correctness validation")
    for name in (
        "server_timeout_s",
        "ready_poll_interval_s",
        "shutdown_timeout_s",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.cooldown_s < 0:
        parser.error("--cooldown-s must be non-negative")
    return args


def normalize_gpu_counts(values: Iterable[int]) -> list[int]:
    counts = list(dict.fromkeys(values))
    invalid = [value for value in counts if value not in SUPPORTED_GPU_COUNTS]
    if invalid:
        raise ValueError(
            f"GPU counts must be selected from {SUPPORTED_GPU_COUNTS}; got {invalid}"
        )
    if not counts:
        raise ValueError("At least one GPU count is required")
    return counts


def prepare_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {resolved}. Use a new run directory."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def validate_model_path(spec: ModelSpec, path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {resolved}")
    model_index_path = resolved / "model_index.json"
    if not model_index_path.is_file():
        raise FileNotFoundError(f"Missing model_index.json: {model_index_path}")
    model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
    class_name = model_index.get("_class_name")
    if class_name != spec.pipeline_class:
        raise ValueError(
            f"Expected _class_name={spec.pipeline_class!r} in {model_index_path}, "
            f"got {class_name!r}"
        )
    return resolved


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _capture_output(command: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip()
    return output if result.returncode == 0 and output else None


def collect_environment_provenance() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    torch_version = _package_version("torch")
    cuda_version = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except Exception:
        pass
    gpu_rows = _capture_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ]
    )
    return {
        "git_sha": _capture_output(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch_version,
        "torch_cuda_version": cuda_version,
        "runai_model_streamer_version": _package_version(
            "runai-model-streamer", "runai_model_streamer"
        ),
        "sglang_version": _package_version("sglang"),
        "gpu_rows": gpu_rows.splitlines() if gpu_rows else [],
    }


def require_runai_model_streamer() -> None:
    if importlib.util.find_spec("runai_model_streamer") is None:
        raise RuntimeError(
            "The April reproduction requires runai_model_streamer for the baseline"
        )
    try:
        __import__("runai_model_streamer")
    except Exception as exc:
        raise RuntimeError(
            "runai_model_streamer is installed but cannot be imported: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def write_python_summary(path: Path, payload: dict[str, Any]) -> None:
    text = "startup_results = " + pprint.pformat(
        payload, width=120, sort_dicts=False
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _bind_address(host: str) -> tuple[int, str]:
    if host in {"::", "[::]"}:
        return socket.AF_INET6, "::"
    # ServerArgs validates IPv4 ports by binding the wildcard address. Match
    # that behavior so a port bound on another local interface is not selected.
    return socket.AF_INET, ""


def _can_bind(host: str, port: int) -> bool:
    family, bind_host = _bind_address(host)
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((bind_host, port))
            sock.listen(1)
        return True
    except OSError:
        return False


def find_free_port(host: str, excluded: set[int] | None = None) -> int:
    excluded = excluded or set()
    family, bind_host = _bind_address(host)
    for _ in range(100):
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((bind_host, 0))
            sock.listen(1)
            port = int(sock.getsockname()[1])
        if port not in excluded:
            return port
    raise RuntimeError("Could not allocate a unique local port")


def allocate_ports(host: str) -> dict[str, int]:
    for _ in range(100):
        http_port = find_free_port(host)
        if http_port < 65535 and _can_bind(host, http_port + 1):
            break
    else:
        raise RuntimeError("Could not allocate adjacent HTTP and broker ports")

    excluded = {http_port, http_port + 1}
    scheduler_port = find_free_port(host, excluded)
    excluded.add(scheduler_port)
    master_port = find_free_port(host, excluded)
    return {
        "http": http_port,
        "broker": http_port + 1,
        "scheduler": scheduler_port,
        "master": master_port,
    }


def build_server_command(
    args: argparse.Namespace,
    spec: ModelSpec,
    model_path: Path,
    gpu_count: int,
    setup: str,
    ports: dict[str, int],
    server_dir: Path,
    *,
    profile_enabled: bool,
    profile_run_id: str | None = None,
    profile_output_dir: Path | None = None,
) -> list[str]:
    ulysses_degree, ring_degree = spec.parallelism[gpu_count]
    generated_dir = server_dir / "generated"
    uploaded_dir = server_dir / "uploaded"
    generated_dir.mkdir(parents=True, exist_ok=True)
    uploaded_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.python,
        "-m",
        "sglang.multimodal_gen.runtime.entrypoints.cli.main",
        "serve",
        "--model-path",
        str(model_path),
        "--model-id",
        spec.model_id,
        "--backend",
        "sglang",
        "--num-gpus",
        str(gpu_count),
        "--tp-size",
        "1",
        "--sp-degree",
        str(gpu_count),
        "--ulysses-degree",
        str(ulysses_degree),
        "--ring-degree",
        str(ring_degree),
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
        "--use-fsdp-inference",
        "false",
        "--enable-torch-compile",
        "false",
        "--warmup",
        "false",
        "--log-level",
        "info",
        "--host",
        args.host,
        "--port",
        str(ports["http"]),
        "--scheduler-port",
        str(ports["scheduler"]),
        "--master-port",
        str(ports["master"]),
        "--strict-ports",
        "true",
        "--output-path",
        str(generated_dir),
        "--input-save-path",
        str(uploaded_dir),
    ]
    if setup == "rank0_broadcast_pageable":
        command.extend(OPTIMIZED_FLAGS)
    elif setup != "baseline":
        raise ValueError(f"Unknown setup: {setup}")

    if profile_enabled:
        if profile_run_id is None or profile_output_dir is None:
            raise ValueError("Profile run id and output directory are required")
        command.extend(
            [
                "--profile-enabled",
                "true",
                "--profile-output-dir",
                str(profile_output_dir),
                "--profile-run-id",
                profile_run_id,
            ]
        )
    return command


def build_server_environment(
    server_dir: Path, setup: str, *, profile_enabled: bool
) -> tuple[dict[str, str], dict[str, str]]:
    if setup not in SETUP_ENV_OVERRIDES:
        raise ValueError(f"Unknown setup: {setup}")
    environment = os.environ.copy()
    environment.update(COMMON_ENV_OVERRIDES)
    environment.update(SETUP_ENV_OVERRIDES[setup])
    environment["SGLANG_PERF_LOG_DIR"] = str(server_dir / "performance_logs")
    for name in (
        "SGLANG_DIFFUSION_TORCH_PROFILER_DIR",
        "SGLANG_TORCH_PROFILER_DIR",
        "SGLANG_TEST_NUM_INFERENCE_STEPS",
        "SGLANG_LAUNCH_TASK_LOG_PATH",
    ):
        environment.pop(name, None)
    if profile_enabled:
        environment["SGLANG_LAUNCH_TASK_LOG_PATH"] = str(
            server_dir / "launch_tasks.jsonl"
        )
    effective_overrides = {
        **COMMON_ENV_OVERRIDES,
        **SETUP_ENV_OVERRIDES[setup],
        "SGLANG_PERF_LOG_DIR": environment["SGLANG_PERF_LOG_DIR"],
    }
    if profile_enabled:
        effective_overrides["SGLANG_LAUNCH_TASK_LOG_PATH"] = environment[
            "SGLANG_LAUNCH_TASK_LOG_PATH"
        ]
    return environment, effective_overrides


def launch_server(
    command: list[str], log_path: Path, environment: dict[str, str]
) -> LaunchedServer:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("wb")
    popen_kwargs: dict[str, Any] = {
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        "env": environment,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    elif os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    started_ns = time.perf_counter_ns()
    try:
        process = subprocess.Popen(command, **popen_kwargs)
    except Exception:
        log_file.close()
        raise
    return LaunchedServer(process, log_file, log_path, command, started_ns)


def stop_server(server: LaunchedServer | None, timeout_s: float) -> None:
    if server is None:
        return
    process = server.process
    try:
        if process.poll() is None:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                process.terminate()
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait(timeout=10)
    finally:
        server.log_file.close()


def read_log_tail(path: Path, line_count: int = 120) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def collect_runai_stream_info(path: Path) -> dict[str, Any]:
    events: list[dict[str, float]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = RUNAI_STREAM_RE.search(line)
            if match:
                events.append(
                    {
                        "gib": float(match.group(1)),
                        "elapsed_s": float(match.group(2)),
                    }
                )
    return {
        "stream_count": len(events),
        "streamed_gib_sum": round(sum(event["gib"] for event in events), 3),
        "stream_time_s_sum": round(
            sum(event["elapsed_s"] for event in events), 3
        ),
        "events": events,
    }


def load_launch_task_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _safe_metric_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_") or "unknown"


def summarize_launch_tasks(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    starts: list[float] = []
    ends: list[float] = []
    for record in records:
        try:
            elapsed_s = float(record.get("elapsed_s"))
        except (TypeError, ValueError):
            continue
        task = _safe_metric_name(str(record.get("task") or "unknown"))
        component = record.get("component")
        key = task if not component else f"{task}_{_safe_metric_name(str(component))}"
        grouped.setdefault(key, []).append(elapsed_s)
        try:
            end_s = float(record.get("timestamp_s"))
        except (TypeError, ValueError):
            continue
        ends.append(end_s)
        starts.append(end_s - elapsed_s)

    summary: dict[str, Any] = {"record_count": len(records)}
    for key, values in grouped.items():
        summary[key] = {
            "count": len(values),
            "mean_s": round(statistics.fmean(values), 6),
            "max_s": round(max(values), 6),
        }
    if starts and ends:
        summary["observed_span_s"] = round(max(ends) - min(starts), 6)
    return summary


def summarize_launch_task_runs(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-rank task maxima across metrics runs.

    These tasks overlap (for example worker_init_total contains component loads),
    so their SP deltas are diagnostic signals and must not be added together.
    """
    grouped: dict[str, list[float]] = {}
    for record in records:
        launch_tasks = record.get("launch_tasks", {})
        if not isinstance(launch_tasks, dict):
            continue
        for task, task_summary in launch_tasks.items():
            if not isinstance(task_summary, dict):
                continue
            try:
                max_s = float(task_summary["max_s"])
            except (KeyError, TypeError, ValueError):
                continue
            grouped.setdefault(task, []).append(max_s)

    return {
        task: {
            "runs": len(values),
            "rank_max_mean_s": round(statistics.fmean(values), 6),
            "rank_max_max_s": round(max(values), 6),
        }
        for task, values in sorted(grouped.items())
    }


def is_strict_port_collision(path: Path) -> bool:
    log_tail = read_log_tail(path)
    return (
        " port " in log_tail
        and "is unavailable and --strict-ports is enabled" in log_tail
    )


def server_base_url(host: str, port: int) -> str:
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::", "[::]"} else host
    if ":" in connect_host and not connect_host.startswith("["):
        connect_host = f"[{connect_host}]"
    return f"http://{connect_host}:{port}"


def wait_for_ready(
    server: LaunchedServer,
    base_url: str,
    spec: ModelSpec,
    model_path: Path,
    gpu_count: int,
    timeout_s: float,
    poll_interval_s: float,
) -> tuple[dict[str, Any], int]:
    deadline = time.perf_counter() + timeout_s
    last_error = ""
    expected_path = str(model_path)
    with requests.Session() as session:
        session.trust_env = False
        while time.perf_counter() < deadline:
            return_code = server.process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"Server exited with code {return_code} before becoming ready.\n"
                    f"{read_log_tail(server.log_path)}"
                )
            try:
                response = session.get(f"{base_url}/v1/models", timeout=1.0)
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                else:
                    payload = response.json()
                    cards = payload.get("data")
                    if not isinstance(cards, list) or len(cards) != 1:
                        last_error = f"unexpected model cards: {cards!r}"
                    else:
                        card = cards[0]
                        expected = {
                            "id": expected_path,
                            "pipeline_class": spec.pipeline_class,
                            "task_type": spec.task_type,
                            "num_gpus": gpu_count,
                        }
                        mismatches = {
                            key: {"expected": value, "actual": card.get(key)}
                            for key, value in expected.items()
                            if card.get(key) != value
                        }
                        if not mismatches:
                            return card, time.perf_counter_ns()
                        last_error = f"model card mismatches: {mismatches}"
            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
            time.sleep(poll_interval_s)
    raise TimeoutError(
        f"Timed out after {timeout_s}s waiting for {base_url}/v1/models: "
        f"{last_error}\n{read_log_tail(server.log_path)}"
    )


def load_server_args_from_log(path: Path) -> dict[str, Any]:
    prefix = "server_args: "
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker_index = line.find(prefix)
        if marker_index < 0:
            continue
        payload = line[marker_index + len(prefix) :].strip()
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"Could not find structured server_args in {path}")


def validate_effective_server_args(
    actual: dict[str, Any],
    spec: ModelSpec,
    model_path: Path,
    gpu_count: int,
    setup: str,
) -> None:
    ulysses_degree, ring_degree = spec.parallelism[gpu_count]
    expected = {
        "model_path": str(model_path),
        "model_id": spec.model_id,
        "backend": "sglang",
        "num_gpus": gpu_count,
        "tp_size": 1,
        "sp_degree": gpu_count,
        "ulysses_degree": ulysses_degree,
        "ring_degree": ring_degree,
        "enable_cfg_parallel": False,
        "dit_cpu_offload": False,
        "dit_layerwise_offload": False,
        "text_encoder_cpu_offload": False,
        "image_encoder_cpu_offload": False,
        "vae_cpu_offload": False,
        "pin_cpu_memory": False,
        "use_fsdp_inference": False,
        "enable_torch_compile": False,
        "warmup": False,
        "strict_ports": True,
    }
    if setup == "baseline":
        expected.update(
            {
                "diffusion_weight_staging": "none",
                "diffusion_weight_load_mode": "default",
                "diffusion_weight_broadcast_components": [],
            }
        )
    else:
        expected.update(
            {
                "diffusion_weight_staging": "pageable",
                "diffusion_weight_load_mode": "rank0-broadcast",
                "diffusion_weight_broadcast_components": ["transformer", "vae"],
            }
        )
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Effective server args do not match April reproduction setup: {mismatches}"
        )


def validate_setup_server_arg_parity(
    baseline_record: dict[str, Any], optimized_record: dict[str, Any]
) -> dict[str, Any]:
    """Ensure weight loading is the only semantic server-argument difference."""
    ignored = {
        "port",
        "scheduler_port",
        "master_port",
        "output_path",
        "input_save_path",
        "profile_output_dir",
        "profile_run_id",
        "diffusion_weight_staging",
        "diffusion_weight_load_mode",
        "diffusion_weight_broadcast_components",
    }
    baseline = baseline_record.get("effective_server_args", {})
    optimized = optimized_record.get("effective_server_args", {})
    keys = (set(baseline) | set(optimized)) - ignored
    differences = {
        key: {"baseline": baseline.get(key), "optimized": optimized.get(key)}
        for key in sorted(keys)
        if baseline.get(key) != optimized.get(key)
    }
    baseline_runai = baseline_record.get("setup_env_overrides", {}).get(
        "SGLANG_USE_RUNAI_MODEL_STREAMER"
    )
    optimized_runai = optimized_record.get("setup_env_overrides", {}).get(
        "SGLANG_USE_RUNAI_MODEL_STREAMER"
    )
    if baseline_runai != "true" or optimized_runai != "true":
        differences["SGLANG_USE_RUNAI_MODEL_STREAMER"] = {
            "baseline": baseline_runai,
            "optimized": optimized_runai,
        }
    if differences:
        raise ValueError(
            "Baseline and optimized setup drift beyond weight-load arguments: "
            f"{differences}"
        )
    return {
        "valid": True,
        "ignored_dynamic_or_weight_load_args": sorted(ignored),
    }


def execute_launch(
    args: argparse.Namespace,
    spec: ModelSpec,
    model_path: Path,
    gpu_count: int,
    setup: str,
    trial_dir: Path,
    *,
    profile_enabled: bool,
    environment_provenance: dict[str, Any],
    profile_run_id: str | None = None,
) -> dict[str, Any]:
    server_dir = trial_dir / "server"
    profile_output_dir = trial_dir / "profile" if profile_enabled else None
    environment, effective_env_overrides = build_server_environment(
        server_dir, setup, profile_enabled=profile_enabled
    )
    launch_task_path = (
        Path(environment["SGLANG_LAUNCH_TASK_LOG_PATH"])
        if profile_enabled
        else None
    )
    record: dict[str, Any] = {
        "status": "starting",
        "setup": setup,
        "profile_enabled": profile_enabled,
        "profile_run_id": profile_run_id,
        "port_allocation_attempts": [],
        "common_env_overrides": COMMON_ENV_OVERRIDES,
        "setup_env_overrides": SETUP_ENV_OVERRIDES[setup],
        "effective_env_overrides": effective_env_overrides,
        "environment": environment_provenance,
        "launch_task_log_path": str(launch_task_path) if launch_task_path else None,
        "server_log": str(trial_dir / "server.log"),
    }
    try:
        for port_attempt in range(1, MAX_PORT_LAUNCH_ATTEMPTS + 1):
            ports = allocate_ports(args.host)
            command = build_server_command(
                args,
                spec,
                model_path,
                gpu_count,
                setup,
                ports,
                server_dir,
                profile_enabled=profile_enabled,
                profile_run_id=profile_run_id,
                profile_output_dir=profile_output_dir,
            )
            log_path = trial_dir / "server.log"
            record.update(
                {
                    "status": "starting",
                    "port_allocation_attempt": port_attempt,
                    "ports": ports,
                    "command": command,
                }
            )
            save_json(trial_dir / "run_meta.json", record)

            server = None
            retry_port_collision = False
            try:
                if launch_task_path is not None:
                    launch_task_path.unlink(missing_ok=True)
                server = launch_server(
                    command,
                    log_path,
                    environment,
                )
                card, ready_ns = wait_for_ready(
                    server,
                    server_base_url(args.host, ports["http"]),
                    spec,
                    model_path,
                    gpu_count,
                    args.server_timeout_s,
                    args.ready_poll_interval_s,
                )
                startup_time_ms = (ready_ns - server.started_ns) / 1_000_000.0
                actual_server_args = load_server_args_from_log(server.log_path)
                validate_effective_server_args(
                    actual_server_args, spec, model_path, gpu_count, setup
                )
                record.update(
                    {
                        "status": "complete",
                        "startup_time_ms": startup_time_ms,
                        "ready_model_card": card,
                        "effective_server_args": actual_server_args,
                        "runai_streams": collect_runai_stream_info(log_path),
                    }
                )
                if launch_task_path is not None:
                    record["launch_tasks"] = summarize_launch_tasks(
                        load_launch_task_records(launch_task_path)
                    )
                return record
            except BaseException as exc:
                retry_port_collision = (
                    port_attempt < MAX_PORT_LAUNCH_ATTEMPTS
                    and is_strict_port_collision(log_path)
                )
                if retry_port_collision:
                    archived_log = trial_dir / (
                        f"server_port_collision_attempt_{port_attempt}.log"
                    )
                    record["port_allocation_attempts"].append(
                        {
                            "attempt": port_attempt,
                            "ports": ports,
                            "error": f"{type(exc).__name__}: {exc}",
                            "server_log": str(archived_log),
                        }
                    )
                    record["status"] = "retrying-port-allocation"
                    continue
                record.update(
                    {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                raise
            finally:
                stop_server(server, args.shutdown_timeout_s)
                if retry_port_collision and log_path.exists():
                    os.replace(
                        log_path,
                        trial_dir
                        / f"server_port_collision_attempt_{port_attempt}.log",
                    )
    finally:
        save_json(trial_dir / "run_meta.json", record)
        if args.cooldown_s:
            time.sleep(args.cooldown_s)


def load_profile_records(profile_dir: Path) -> list[dict[str, Any]]:
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"Weight profile directory was not created: {profile_dir}")
    records: list[dict[str, Any]] = []
    for path in sorted(profile_dir.glob("weight_load_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Profile root must be an object: {path}")
        value["_profile_path"] = str(path)
        records.append(value)
    if not records:
        raise ValueError(f"No weight profile JSON files found in {profile_dir}")
    return records


def _integer(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def validate_profile_records(
    records: list[dict[str, Any]], gpu_count: int, setup: str
) -> dict[str, Any]:
    expected_ranks = set(range(gpu_count))
    component_records: dict[str, list[dict[str, Any]]] = {}
    for component in PROFILE_COMPONENTS:
        selected = [record for record in records if record.get("component") == component]
        if not selected:
            raise ValueError(f"Missing {component} profile records")
        ranks = [_integer(record.get("sp_rank"), "sp_rank") for record in selected]
        if len(ranks) != len(set(ranks)):
            raise ValueError(
                f"Duplicate {component} rank records detected: {sorted(ranks)}"
            )
        if set(ranks) != expected_ranks:
            raise ValueError(
                f"{component} ranks mismatch: expected {sorted(expected_ranks)}, "
                f"got {sorted(ranks)}"
            )
        component_records[component] = selected

    for component, selected in component_records.items():
        by_rank = {
            _integer(record.get("sp_rank"), "sp_rank"): record
            for record in selected
        }
        for rank, record in by_rank.items():
            if record.get("status") != "success":
                raise ValueError(
                    f"{component} rank {rank} profile failed: {record.get('error')}"
                )
            if _integer(record.get("world_size"), "world_size") != gpu_count:
                raise ValueError(f"{component} rank {rank} world_size mismatch")
            errors = {
                field: record.get(field)
                for field in ERROR_FIELDS
                if record.get(field) not in (None, "", False)
            }
            if errors:
                raise ValueError(f"{component} rank {rank} errors: {errors}")

        if setup == "baseline":
            expected_backend = "runai" if component == "transformer" else "safetensors"
            expected_bytes = _number(
                by_rank[0].get("weight_load:total_bytes"), "total_bytes"
            )
            if expected_bytes <= 0:
                raise ValueError(f"Baseline {component} rank 0 read no bytes")
            for rank, record in by_rank.items():
                if record.get("weight_load:load_mode_effective") != "default":
                    raise ValueError(f"Baseline {component} rank {rank} is not default")
                total_bytes = _number(
                    record.get("weight_load:total_bytes"), "total_bytes"
                )
                if total_bytes != expected_bytes:
                    raise ValueError(
                        f"Baseline {component} rank {rank} did not read the full "
                        f"replica: expected {expected_bytes:.0f} bytes, got "
                        f"{total_bytes:.0f}"
                    )
                if record.get("weight_load:read_backend") != expected_backend:
                    raise ValueError(
                        f"Baseline {component} rank {rank} expected read backend "
                        f"{expected_backend!r}, got "
                        f"{record.get('weight_load:read_backend')!r}"
                    )
                if _number(
                    record.get("weight_load:broadcast_bytes"), "broadcast_bytes"
                ) != 0:
                    raise ValueError(
                        f"Baseline {component} rank {rank} unexpectedly broadcast"
                    )
            continue

        if gpu_count == 1:
            rank0 = by_rank[0]
            expected_backend = "runai" if component == "transformer" else "safetensors"
            if rank0.get("weight_load:load_mode_requested") != "rank0-broadcast":
                raise ValueError(f"SP1 {component} did not request rank0 broadcast")
            if rank0.get("weight_load:load_mode_effective") != "default":
                raise ValueError(f"SP1 {component} should fall back to default")
            if rank0.get("weight_load:read_backend") != expected_backend:
                raise ValueError(
                    f"SP1 {component} expected read backend {expected_backend!r}, "
                    f"got {rank0.get('weight_load:read_backend')!r}"
                )
            continue

        rank0 = by_rank[0]
        rank0_total = _number(rank0.get("weight_load:total_bytes"), "total_bytes")
        rank0_broadcast = _number(
            rank0.get("weight_load:broadcast_bytes"), "broadcast_bytes"
        )
        rank0_count = _integer(
            rank0.get("weight_load:broadcast_tensor_count"), "broadcast_tensor_count"
        )
        if rank0_total <= 0 or rank0_broadcast <= 0 or rank0_count <= 0:
            raise ValueError(f"Optimized {component} rank0 metrics are empty")
        expected_rank0_backend = (
            "rank0-broadcast-no-runai"
            if component == "transformer"
            else "safetensors"
        )
        if rank0.get("weight_load:read_backend") != expected_rank0_backend:
            raise ValueError(
                f"Optimized {component} rank0 expected read backend "
                f"{expected_rank0_backend!r}, got "
                f"{rank0.get('weight_load:read_backend')!r}"
            )

        for rank, record in by_rank.items():
            if record.get("weight_load:load_mode_effective") != "rank0-broadcast":
                raise ValueError(f"Optimized {component} rank {rank} did not broadcast")
            broadcast_bytes = _number(
                record.get("weight_load:broadcast_bytes"), "broadcast_bytes"
            )
            broadcast_count = _integer(
                record.get("weight_load:broadcast_tensor_count"),
                "broadcast_tensor_count",
            )
            if broadcast_bytes != rank0_broadcast or broadcast_count != rank0_count:
                raise ValueError(
                    f"Optimized {component} rank {rank} broadcast metadata mismatch"
                )
            if rank == 0:
                if record.get("weight_load:staging_effective") != "pageable":
                    raise ValueError(f"Optimized {component} rank0 is not pageable")
                continue
            if _number(record.get("weight_load:total_bytes"), "total_bytes") != 0:
                raise ValueError(f"Optimized {component} rank {rank} read checkpoint bytes")
            if record.get("weight_load:read_backend") != "rank0-broadcast-receive":
                raise ValueError(
                    f"Optimized {component} rank {rank} receive backend mismatch"
                )

    return {
        "valid": True,
        "expected_rank_count": gpu_count,
        "profile_record_count": len(records),
        "profile_components": sorted(
            {str(record.get("component")) for record in records}
        ),
        "rank0_broadcast_applicable": setup != "baseline" and gpu_count > 1,
    }


def normalized_profile_metrics(record: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rank": _integer(record.get("sp_rank"), "sp_rank"),
        "read_backend": record.get("weight_load:read_backend"),
        "load_mode_requested": record.get("weight_load:load_mode_requested"),
        "load_mode_effective": record.get("weight_load:load_mode_effective"),
        "staging_effective": record.get("weight_load:staging_effective"),
    }
    for field in TIME_FIELDS_MS:
        result[field.removeprefix("weight_load:").removesuffix("_ms") + "_s"] = round(
            _number(record.get(field, 0.0), field) / 1000.0, 6
        )
    for field in BYTE_FIELDS:
        result[field.removeprefix("weight_load:").removesuffix("_bytes") + "_gib"] = round(
            _number(record.get(field, 0), field) / (1024**3), 6
        )
    for field in COUNT_FIELDS:
        result[field.removeprefix("weight_load:")] = _integer(
            record.get(field, 0), field
        )
    return result


def aggregate_numeric_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    normalized = [normalized_profile_metrics(record) for record in records]
    numeric_keys = sorted(
        {
            key
            for item in normalized
            for key, value in item.items()
            if isinstance(value, (int, float)) and key != "rank"
        }
    )
    result: dict[str, Any] = {
        "rank_count": len(records),
        "read_backends": sorted({str(item["read_backend"]) for item in normalized}),
        "load_modes": sorted(
            {str(item["load_mode_effective"]) for item in normalized}
        ),
    }
    for key in numeric_keys:
        values = [float(item[key]) for item in normalized]
        result[f"{key}_avg"] = round(statistics.fmean(values), 6)
        result[f"{key}_max"] = round(max(values), 6)
    return result


def summarize_profile_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for component in PROFILE_COMPONENTS:
        selected = [record for record in records if record.get("component") == component]
        rank0 = next(
            record
            for record in selected
            if _integer(record.get("sp_rank"), "sp_rank") == 0
        )
        nonrank = [
            record
            for record in selected
            if _integer(record.get("sp_rank"), "sp_rank") != 0
        ]
        result[component] = {
            "cluster_total_gib": round(
                sum(
                    _number(record.get("weight_load:total_bytes"), "total_bytes")
                    for record in selected
                )
                / (1024**3),
                6,
            ),
            "rank0": normalized_profile_metrics(rank0),
            "nonrank": aggregate_numeric_records(nonrank),
        }
    return result


def summarize_timing_runs(
    records: list[dict[str, Any]], warmup_runs: int
) -> dict[str, Any]:
    complete = [record for record in records if record.get("status") == "complete"]
    if len(complete) != len(records):
        raise ValueError("Cannot summarize incomplete startup runs")
    measured = complete[warmup_runs:]
    values = [float(record["startup_time_ms"]) for record in measured]
    if not values:
        raise ValueError("No measured startup runs remain after warmup exclusion")
    stream_counts = [
        float(record.get("runai_streams", {}).get("stream_count", 0))
        for record in measured
    ]
    stream_times = [
        float(record.get("runai_streams", {}).get("stream_time_s_sum", 0.0))
        for record in measured
    ]
    return {
        "all_runs_ms": [round(float(record["startup_time_ms"]), 3) for record in complete],
        "measured_runs_ms": [round(value, 3) for value in values],
        "measured_mean_ms": round(statistics.fmean(values), 3),
        "measured_p50_ms": round(statistics.median(values), 3),
        "measured_min_ms": round(min(values), 3),
        "measured_max_ms": round(max(values), 3),
        "measured_std_ms": round(statistics.pstdev(values), 3),
        "runai_stream_count_mean": round(statistics.fmean(stream_counts), 3),
        "runai_stream_time_s_mean": round(statistics.fmean(stream_times), 3),
    }


def transformer_critical_path_summary(
    validations: list[dict[str, Any]],
) -> dict[str, Any]:
    time_keys = (
        "discover_files_s",
        "read_safetensors_s",
        "cpu_materialize_s",
        "pin_memory_s",
        "h2d_or_param_copy_s",
        "d2h_or_offload_s",
        "rank0_wait_s",
        "nccl_broadcast_s",
    )
    values: list[float] = []
    for validation in validations:
        transformer = validation.get("metrics", {}).get("transformer", {})
        rank0 = transformer.get("rank0", {})
        rank0_path = sum(float(rank0.get(key, 0.0)) for key in time_keys)
        nonrank = transformer.get("nonrank", {})
        nonrank_path = sum(
            float(nonrank.get(f"{key}_max", 0.0)) for key in time_keys
        )
        values.append(max(rank0_path, nonrank_path))
    if not values:
        return {}
    return {
        "runs": len(values),
        "mean_s": round(statistics.fmean(values), 6),
        "max_s": round(max(values), 6),
    }


def _scaling_ratio(values: dict[int, float], low: int, high: int) -> float | None:
    low_value = values.get(low)
    high_value = values.get(high)
    if low_value is None or high_value is None or low_value <= 0:
        return None
    return round(high_value / low_value, 4)


def _gain_pct(baseline_s: float, optimized_s: float) -> float:
    return round(((baseline_s - optimized_s) / baseline_s) * 100.0, 3)


def residual_scaling_attribution(
    model_summary: dict[str, Any],
    *,
    setup: str,
    low_sp: int = 1,
    high_sp: int = 8,
) -> dict[str, Any]:
    points = model_summary["points"]
    if low_sp not in points or high_sp not in points:
        return {}

    low_tasks = points[low_sp]["launch_task_diagnostics"].get(setup, {})
    high_tasks = points[high_sp]["launch_task_diagnostics"].get(setup, {})
    task_deltas: list[dict[str, Any]] = []
    for task in sorted(set(low_tasks) | set(high_tasks)):
        low_value = float(low_tasks.get(task, {}).get("rank_max_mean_s", 0.0))
        high_value = float(high_tasks.get(task, {}).get("rank_max_mean_s", 0.0))
        task_deltas.append(
            {
                "task": task,
                "sp1_rank_max_mean_s": round(low_value, 6),
                "sp8_rank_max_mean_s": round(high_value, 6),
                "sp8_minus_sp1_s": round(high_value - low_value, 6),
            }
        )
    task_deltas.sort(key=lambda item: item["sp8_minus_sp1_s"], reverse=True)

    startup_key = (
        "baseline_s" if setup == "baseline" else "rank0_broadcast_pageable_s"
    )
    startup_values = model_summary[startup_key]
    return {
        "setup": setup,
        "from_sp": low_sp,
        "to_sp": high_sp,
        "ready_wall_growth_s": round(
            float(startup_values[high_sp]) - float(startup_values[low_sp]), 6
        ),
        "overlapping_task_deltas": task_deltas,
        "note": (
            "Task durations overlap and are ranked as causal diagnostics; do not "
            "sum them as a wall-time decomposition."
        ),
    }


def build_summary(state: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "metadata": {
            "metric": "Popen to validated /v1/models readiness",
            "raw_timing_unit": "ms",
            "startup_dict_unit": "seconds",
            "num_runs": state["num_runs"],
            "warmup_runs": state["warmup_runs"],
            "metrics_runs": state["metrics_runs"],
            "common_env_overrides": COMMON_ENV_OVERRIDES,
            "setup_env_overrides": SETUP_ENV_OVERRIDES,
            "optimized_flags": list(OPTIMIZED_FLAGS),
            "april_reproduction_policy": APRIL_REPRODUCTION_POLICY,
            "environment": state["environment"],
            "historical_reference_note": (
                "H200 references are comparison targets, not absolute H100 gates"
            ),
        },
        "models": {},
    }
    for point in state["points"]:
        model_summary = summary["models"].setdefault(
            point["model"],
            {
                "baseline_s": {},
                "rank0_broadcast_pageable_s": {},
                "gain_pct": {},
                "points": {},
            },
        )
        baseline = summarize_timing_runs(
            point["timing_runs"]["baseline"], state["warmup_runs"]
        )
        optimized = summarize_timing_runs(
            point["timing_runs"]["rank0_broadcast_pageable"],
            state["warmup_runs"],
        )
        baseline_mean = baseline["measured_mean_ms"]
        optimized_mean = optimized["measured_mean_ms"]
        reduction_pct = (
            ((baseline_mean - optimized_mean) / baseline_mean) * 100.0
            if baseline_mean > 0
            else 0.0
        )
        speedup = baseline_mean / optimized_mean if optimized_mean > 0 else None
        gpu_count = int(point["gpu_count"])
        baseline_s = round(baseline_mean / 1000.0, 6)
        optimized_s = round(optimized_mean / 1000.0, 6)
        model_summary["baseline_s"][gpu_count] = baseline_s
        model_summary["rank0_broadcast_pageable_s"][gpu_count] = optimized_s
        model_summary["gain_pct"][gpu_count] = _gain_pct(baseline_s, optimized_s)
        model_summary["points"][gpu_count] = {
            "parallelism": point["parallelism"],
            "baseline": baseline,
            "rank0_broadcast_pageable": optimized,
            "reduction_pct": round(reduction_pct, 3),
            "speedup_x": round(speedup, 4) if speedup is not None else None,
            "server_arg_parity": point.get("server_arg_parity", {}),
            "metrics_validation": point["metrics_validation"],
            "transformer_critical_path": {
                setup: transformer_critical_path_summary(
                    point["metrics_validation"].get(setup, [])
                )
                for setup in SETUPS
            },
            "launch_task_diagnostics": {
                setup: summarize_launch_task_runs(
                    point["metrics_runs"].get(setup, [])
                )
                for setup in SETUPS
            },
        }

    benefit_targets = {
        "wan22_ti2v_5b": {4: 20.0, 8: 30.0},
        "z_image": {4: 10.0, 8: 15.0},
    }
    for model_key, model_summary in summary["models"].items():
        baseline_s = model_summary["baseline_s"]
        optimized_s = model_summary["rank0_broadcast_pageable_s"]
        critical_path_s = {
            gpu: (
                point["transformer_critical_path"]
                .get("rank0_broadcast_pageable", {})
                .get("mean_s")
            )
            for gpu, point in model_summary["points"].items()
        }
        critical_path_s = {
            gpu: float(value)
            for gpu, value in critical_path_s.items()
            if value is not None
        }
        checks: dict[str, Any] = {}
        if 1 in baseline_s and 1 in optimized_s:
            sp1_delta_pct = abs(_gain_pct(baseline_s[1], optimized_s[1]))
            checks["sp1_delta_within_5pct"] = {
                "value_pct": round(sp1_delta_pct, 3),
                "pass": sp1_delta_pct <= 5.0,
            }
        critical_ratio = _scaling_ratio(critical_path_s, 2, 8)
        if critical_ratio is not None:
            checks["optimized_transformer_sp8_over_sp2_at_most_1_25"] = {
                "value": critical_ratio,
                "pass": critical_ratio <= 1.25,
            }
        for gpu, target in benefit_targets.get(model_key, {}).items():
            if gpu in model_summary["gain_pct"]:
                gain = model_summary["gain_pct"][gpu]
                checks[f"sp{gpu}_gain_at_least_{int(target)}pct"] = {
                    "value_pct": gain,
                    "pass": gain >= target,
                }
        if model_key == "wan21_t2v_1_3b" and 8 in model_summary["gain_pct"]:
            checks["sp8_gain_positive"] = {
                "value_pct": model_summary["gain_pct"][8],
                "pass": model_summary["gain_pct"][8] > 0,
            }
        model_summary["scaling"] = {
            "baseline_sp8_over_sp1": _scaling_ratio(baseline_s, 1, 8),
            "optimized_sp8_over_sp1": _scaling_ratio(optimized_s, 1, 8),
            "optimized_transformer_sp8_over_sp2": critical_ratio,
        }
        model_summary["historical_h200_reference"] = HISTORICAL_H200_REFERENCE.get(
            model_key
        )
        model_summary["residual_scaling_attribution"] = {
            setup: residual_scaling_attribution(model_summary, setup=setup)
            for setup in SETUPS
        }
        model_summary["reproduction_checks"] = checks
    return summary


def model_metadata(spec: ModelSpec, model_path: Path) -> dict[str, Any]:
    return {
        "key": spec.key,
        "label": spec.label,
        "model_id": spec.model_id,
        "model_path": str(model_path),
        "pipeline_class": spec.pipeline_class,
        "task_type": spec.task_type,
        "parallelism": {
            str(gpu_count): {
                "ulysses_degree": values[0],
                "ring_degree": values[1],
            }
            for gpu_count, values in spec.parallelism.items()
        },
    }


def main() -> None:
    args = parse_args()
    require_runai_model_streamer()
    environment_provenance = collect_environment_provenance()
    requested_paths = {
        WAN22.key: args.wan22_model_path,
        WAN21.key: args.wan21_model_path,
        Z_IMAGE.key: args.z_image_model_path,
    }
    model_paths = {
        spec.key: validate_model_path(spec, requested_paths[spec.key])
        for spec in ALL_MODEL_SPECS
    }
    output_dir = prepare_output_dir(args.output_dir)
    state_path = output_dir / "results.json"
    state: dict[str, Any] = {
        "status": "running",
        "metric": "Popen to validated /v1/models readiness",
        "unit": "ms",
        "num_runs": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "metrics_runs": args.metrics_runs,
        "gpu_counts": args.gpu_counts,
        "setups": list(SETUPS),
        "common_env_overrides": COMMON_ENV_OVERRIDES,
        "setup_env_overrides": SETUP_ENV_OVERRIDES,
        "environment": environment_provenance,
        "april_reproduction_policy": APRIL_REPRODUCTION_POLICY,
        "models": [
            model_metadata(spec, model_paths[spec.key]) for spec in ALL_MODEL_SPECS
        ],
        "points": [],
    }
    save_json(state_path, state)

    try:
        for spec in ALL_MODEL_SPECS:
            model_path = model_paths[spec.key]
            for gpu_count in args.gpu_counts:
                ulysses_degree, ring_degree = spec.parallelism[gpu_count]
                point: dict[str, Any] = {
                    "model": spec.key,
                    "gpu_count": gpu_count,
                    "parallelism": {
                        "tp_size": 1,
                        "sp_degree": gpu_count,
                        "ulysses_degree": ulysses_degree,
                        "ring_degree": ring_degree,
                        "cfg_parallel": False,
                        "fsdp_inference": False,
                    },
                    "status": "running",
                    "timing_runs": {setup: [] for setup in SETUPS},
                    "metrics_runs": {setup: [] for setup in SETUPS},
                    "metrics_validation": {},
                }
                state["points"].append(point)
                save_json(state_path, state)

                for run_index in range(args.num_runs):
                    setup_order = SETUPS if run_index % 2 == 0 else tuple(reversed(SETUPS))
                    for setup in setup_order:
                        trial_dir = (
                            output_dir
                            / spec.key
                            / f"gpu_{gpu_count}"
                            / setup
                            / "timing"
                            / f"run_{run_index + 1:02d}"
                        )
                        print(
                            f"[timing] {spec.label} GPU={gpu_count} setup={setup} "
                            f"run={run_index + 1}/{args.num_runs}",
                            flush=True,
                        )
                        record = execute_launch(
                            args,
                            spec,
                            model_path,
                            gpu_count,
                            setup,
                            trial_dir,
                            profile_enabled=False,
                            environment_provenance=environment_provenance,
                        )
                        record["run"] = run_index + 1
                        record["warmup"] = run_index < args.warmup_runs
                        point["timing_runs"][setup].append(record)
                        save_json(trial_dir / "run_meta.json", record)
                        save_json(state_path, state)

                    point.setdefault("server_arg_parity", {})[
                        f"timing_run_{run_index + 1:02d}"
                    ] = validate_setup_server_arg_parity(
                        point["timing_runs"]["baseline"][-1],
                        point["timing_runs"]["rank0_broadcast_pageable"][-1],
                    )
                    save_json(state_path, state)

                for setup in SETUPS:
                    validations: list[dict[str, Any]] = []
                    for metrics_index in range(args.metrics_runs):
                        trial_dir = (
                            output_dir
                            / spec.key
                            / f"gpu_{gpu_count}"
                            / setup
                            / "metrics"
                            / f"run_{metrics_index + 1:02d}"
                        )
                        profile_run_id = (
                            f"{spec.key}_gpu{gpu_count}_{setup}_metrics_"
                            f"r{metrics_index + 1:02d}"
                        )
                        print(
                            f"[metrics] {spec.label} GPU={gpu_count} setup={setup} "
                            f"run={metrics_index + 1}/{args.metrics_runs}",
                            flush=True,
                        )
                        record = execute_launch(
                            args,
                            spec,
                            model_path,
                            gpu_count,
                            setup,
                            trial_dir,
                            profile_enabled=True,
                            environment_provenance=environment_provenance,
                            profile_run_id=profile_run_id,
                        )
                        profile_dir = (
                            trial_dir
                            / "profile"
                            / f"{profile_run_id}_launch_weight_load"
                        )
                        record["run"] = metrics_index + 1
                        record["profile_dir"] = str(profile_dir)
                        point["metrics_runs"][setup].append(record)
                        try:
                            records = load_profile_records(profile_dir)
                            validation = validate_profile_records(
                                records, gpu_count, setup
                            )
                            validation["profile_dir"] = str(profile_dir)
                            validation["metrics"] = summarize_profile_records(records)
                            validations.append(validation)
                            record["profile_validation"] = validation
                        except BaseException as exc:
                            record["profile_validation"] = {
                                "valid": False,
                                "error": f"{type(exc).__name__}: {exc}",
                                "profile_dir": str(profile_dir),
                            }
                            save_json(trial_dir / "run_meta.json", record)
                            save_json(state_path, state)
                            raise
                        save_json(trial_dir / "run_meta.json", record)
                        save_json(state_path, state)
                    point["metrics_validation"][setup] = validations

                for metrics_index in range(args.metrics_runs):
                    point.setdefault("server_arg_parity", {})[
                        f"metrics_run_{metrics_index + 1:02d}"
                    ] = validate_setup_server_arg_parity(
                        point["metrics_runs"]["baseline"][metrics_index],
                        point["metrics_runs"]["rank0_broadcast_pageable"][
                            metrics_index
                        ],
                    )

                point["status"] = "complete"
                save_json(state_path, state)

        summary = build_summary(state)
        state["status"] = "complete"
        state["summary"] = summary
        save_json(state_path, state)
        write_python_summary(output_dir / "summary.py", summary)
        print("\n" + pprint.pformat(summary, width=120, sort_dicts=False), flush=True)
        print(f"\nWrote results: {state_path}", flush=True)
        print(f"Wrote Python dict: {output_dir / 'summary.py'}", flush=True)
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        for point in reversed(state["points"]):
            if point.get("status") == "running":
                point["status"] = "failed"
                point["error"] = state["error"]
                break
        save_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
