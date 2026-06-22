#!/usr/bin/env python3
"""Measure fine-grained server launch breakdown for sglang and sglang-diffusion."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    import requests
except ModuleNotFoundError:
    requests = None

import profile_server_launch_time as launch_time

logger = launch_time.logger
PRESETS = launch_time.PRESETS
RunConfig = launch_time.RunConfig

POLL_INTERVAL_S = 0.2

EXPECTED_TASK_KEYS: dict[str, list[str]] = {
    "sglang": [
        "entrypoint_bootstrap",
        "engine_subprocess_setup",
        "parent_worker_spawn",
        "scheduler_spawn",
        "detokenizer_spawn",
        "tokenizer_manager_init",
        "parent_wait_workers_ready",
        "http_server_startup",
        "model_runner_initialize_total",
        "model_runner_pre_load_setup",
        "model_runner_post_load_setup",
        "torch_distributed_init",
        "load_weight",
        "kv_cache_dtype_config",
        "memory_pool_init",
        "cublas_init",
        "attention_backend_init",
        "kernel_warmup",
        "cuda_graph_capture",
        "piecewise_cuda_graph_capture",
        "symmetric_memory_pool_prealloc",
        "torch_compile",
    ],
    "sglang-diffusion": [
        "entrypoint_bootstrap",
        "parent_worker_spawn",
        "parent_wait_workers_ready",
        "http_server_startup",
        "scheduler_bind",
        "worker_init_total",
        "device_bootstrap",
        "distributed_init",
        "distributed_and_model_parallel_init_total",
        "pipeline_select",
        "build_pipeline_total",
        "pipeline_config_load",
        "executor_build",
        "component_load:text_encoder",
        "component_load:tokenizer",
        "component_load:vae",
        "component_load:transformer",
        "component_load:scheduler",
        "component_cpu_materialization:text_encoder",
        "component_cpu_materialization:tokenizer",
        "component_cpu_materialization:vae",
        "component_cpu_materialization:transformer",
        "component_cpu_materialization:scheduler",
        "component_load_stats:text_encoder",
        "component_load_stats:tokenizer",
        "component_load_stats:vae",
        "component_load_stats:transformer",
        "component_load_stats:scheduler",
        "role_launch_total:encoder",
        "role_launch_total:denoiser",
        "role_launch_total:decoder",
        "role_launch_total:shared",
        "pipeline_initialize",
        "pipeline_stage_create",
        "worker_ready_signal",
        "cuda_graph_capture",
        "torch_compile",
    ],
}


class LaunchWaitError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        health_ready_ns: int | None = None,
        models_ready_ns: int | None = None,
    ) -> None:
        super().__init__(message)
        self.health_ready_ns = health_ready_ns
        self.models_ready_ns = models_ready_ns


def normalized_family(preset: launch_time.LaunchPreset) -> str:
    return "sglang" if preset.family == "sglang" else "sglang-diffusion"


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure fine-grained server launch breakdown for sglang and "
            "sglang-diffusion presets."
        )
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(PRESETS.keys()),
        help=(
            "Comma-separated preset list. Defaults to all supported presets: "
            + ",".join(PRESETS.keys())
        ),
    )
    parser.add_argument(
        "--parallel-degrees",
        type=str,
        default="8,4,2,1",
        help="Comma/space separated GPU counts, e.g. '8,4,2,1'.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind servers to.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=1800,
        help="Timeout in seconds for each launch attempt.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output root. Defaults to "
            "/workspace/outputs/server_launch_breakdown/<run_id>."
        ),
    )
    parser.add_argument(
        "--keep-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep per-case logs and temporary launch directories.",
    )
    parser.add_argument(
        "--llm-setups",
        type=str,
        default="default",
        help=(
            "Comma-separated LLM setup keys for prompt-enhancer presets. "
            f"Supported setups: {','.join(launch_time.LLM_SETUPS.keys())}."
        ),
    )
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    return Path(f"/workspace/outputs/server_launch_breakdown/{now_stamp()}").resolve()


def build_diffusion_command(
    *,
    preset: launch_time.LaunchPreset,
    run_config: RunConfig,
    host: str,
    port: int,
    scheduler_port: int,
    master_port: int,
    case_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.runtime.entrypoints.cli.main",
        "serve",
        "--model-path",
        preset.model_path,
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
        "false",
        "--output-path",
        str((case_dir / "outputs").resolve()),
        "--input-save-path",
        str((case_dir / "uploads").resolve()),
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
        "--use-fsdp-inference",
        "false",
    ]
    if preset.model_id:
        command.extend(["--model-id", preset.model_id])
    if run_config.tp_size is not None:
        command.extend(["--tp-size", str(run_config.tp_size)])
    if run_config.sp_degree is not None:
        command.extend(["--sp-degree", str(run_config.sp_degree)])
    if run_config.ulysses_degree is not None:
        command.extend(["--ulysses-degree", str(run_config.ulysses_degree)])
    if run_config.ring_degree is not None:
        command.extend(["--ring-degree", str(run_config.ring_degree)])
    if preset.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def llm_setup_args(llm_setup: launch_time.LLMSetup) -> dict[str, Any]:
    return {
        "chunked_prefill_size": llm_setup.chunked_prefill_size,
        "max_running_requests": llm_setup.max_running_requests,
        "max_total_tokens": llm_setup.max_total_tokens,
        "cuda_graph_max_bs": llm_setup.cuda_graph_max_bs,
    }


def prepare_case_env(launch_tasks_path: Path, *, diffusion: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["SGLANG_LAUNCH_TASK_LOG_PATH"] = str(launch_tasks_path.resolve())
    if diffusion:
        cutlass_python_packages_dir = launch_time.detect_cutlass_python_packages_dir()
        if cutlass_python_packages_dir is not None:
            launch_time.prepend_pythonpath(env, cutlass_python_packages_dir)
            logger.info(
                "Added CUTLASS Python packages dir to PYTHONPATH: %s",
                cutlass_python_packages_dir,
            )
    return env


def extract_task_type(model_card: dict[str, Any] | None) -> str | None:
    if not isinstance(model_card, dict):
        return None
    data = model_card.get("data")
    if not isinstance(data, list) or not data:
        return None
    item = data[0]
    if not isinstance(item, dict):
        return None
    task_type = item.get("task_type")
    return str(task_type).upper() if task_type is not None else None


def wait_for_server_endpoints(
    *,
    process: subprocess.Popen[str],
    base_url: str,
    server_log_path: Path,
    timeout_s: int,
    expected_task_type: str | None,
) -> tuple[int | None, int, dict[str, Any]]:
    if requests is None:
        raise RuntimeError(
            "The requests package is required to wait for server readiness."
        )
    start_ns = time.perf_counter_ns()
    deadline = time.time() + timeout_s
    health_ready_ns: int | None = None
    models_ready_ns: int | None = None
    model_card: dict[str, Any] | None = None
    last_log_time = 0.0

    while True:
        if health_ready_ns is None:
            try:
                response = requests.get(f"{base_url}/health", timeout=1)
                if response.ok:
                    health_ready_ns = time.perf_counter_ns() - start_ns
            except Exception:
                pass

        try:
            response = requests.get(f"{base_url}/v1/models", timeout=1)
            if response.ok:
                payload = response.json()
                actual_task_type = extract_task_type(payload)
                if expected_task_type is None or actual_task_type == expected_task_type:
                    model_card = payload
                    models_ready_ns = time.perf_counter_ns() - start_ns
                    if health_ready_ns is None:
                        try:
                            health_response = requests.get(
                                f"{base_url}/health", timeout=1
                            )
                            if health_response.ok:
                                health_ready_ns = time.perf_counter_ns() - start_ns
                        except Exception:
                            pass
                    return health_ready_ns, models_ready_ns, model_card
        except Exception:
            pass

        if process.poll() is not None:
            raise LaunchWaitError(
                f"Server exited early with code {process.returncode}.\n"
                f"{launch_time.read_log_tail(server_log_path)}",
                health_ready_ns=health_ready_ns,
                models_ready_ns=models_ready_ns,
            )

        now = time.time()
        if now >= deadline:
            raise LaunchWaitError(
                f"Timed out waiting for readiness at {base_url} after {timeout_s}s.\n"
                f"{launch_time.read_log_tail(server_log_path)}",
                health_ready_ns=health_ready_ns,
                models_ready_ns=models_ready_ns,
            )

        if now - last_log_time >= launch_time.WAIT_LOG_INTERVAL_S:
            elapsed_s = int(timeout_s - max(deadline - now, 0))
            logger.info(
                "Waiting for readiness... elapsed=%ss health=%s models=%s expected_task_type=%s",
                elapsed_s,
                health_ready_ns is not None,
                models_ready_ns is not None,
                expected_task_type,
            )
            last_log_time = now
        time.sleep(POLL_INTERVAL_S)


def load_launch_task_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def build_task_key(event: dict[str, Any]) -> str:
    component = event.get("component")
    if component:
        return f"{event['task']}:{component}"
    return str(event["task"])


def summarize_launch_tasks(
    events: list[dict[str, Any]],
    *,
    family: str,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("phase") != "end":
            continue
        key = build_task_key(event)
        entry = grouped.setdefault(
            key,
            {
                "task": event.get("task"),
                "component": event.get("component"),
                "observed": True,
                "occurrences": 0,
                "elapsed_ms_sum": 0.0,
                "elapsed_ms_max": None,
                "elapsed_s_sum": 0.0,
                "elapsed_s_max": None,
                "statuses": {},
                "events": [],
            },
        )
        elapsed_ms = event.get("elapsed_ms")
        elapsed_s = event.get("elapsed_s")
        entry["occurrences"] += 1
        if isinstance(elapsed_ms, (int, float)):
            entry["elapsed_ms_sum"] = round(entry["elapsed_ms_sum"] + elapsed_ms, 3)
            entry["elapsed_ms_max"] = (
                elapsed_ms
                if entry["elapsed_ms_max"] is None
                else round(max(entry["elapsed_ms_max"], elapsed_ms), 3)
            )
        if isinstance(elapsed_s, (int, float)):
            entry["elapsed_s_sum"] = round(entry["elapsed_s_sum"] + elapsed_s, 3)
            entry["elapsed_s_max"] = (
                elapsed_s
                if entry["elapsed_s_max"] is None
                else round(max(entry["elapsed_s_max"], elapsed_s), 3)
            )
        status = str(event.get("status", "unknown"))
        entry["statuses"][status] = entry["statuses"].get(status, 0) + 1
        entry["events"].append(
            {
                "timestamp": event.get("timestamp"),
                "rank": event.get("rank"),
                "status": status,
                "elapsed_ms": elapsed_ms,
                "elapsed_s": elapsed_s,
                "extra": event.get("extra"),
                "error": event.get("error"),
            }
        )

    for key in EXPECTED_TASK_KEYS.get(family, []):
        grouped.setdefault(
            key,
            {
                "task": key.split(":", 1)[0],
                "component": key.split(":", 1)[1] if ":" in key else None,
                "observed": False,
                "occurrences": 0,
                "elapsed_ms_sum": None,
                "elapsed_ms_max": None,
                "elapsed_s_sum": None,
                "elapsed_s_max": None,
                "statuses": {},
                "events": [],
            },
        )
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def make_case_record(
    *,
    preset: launch_time.LaunchPreset,
    gpu_count: int,
    requested_parallelism: dict[str, Any],
    resolved_parallelism: dict[str, Any] | None,
    case_name: str,
    case_dir: Path,
    launch_command: list[str] | None,
    status: str,
    reason: str | None,
    health_ready_ns: int | None,
    models_ready_ns: int | None,
    model_card: dict[str, Any] | None,
    launch_tasks_path: Path,
    task_events: list[dict[str, Any]],
    task_summary: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "model_key": preset.key,
        "family": normalized_family(preset),
        "model_path": preset.model_path,
        "model_id": preset.model_id,
        "gpu_count": gpu_count,
        "case_name": case_name,
        "requested_parallelism": requested_parallelism,
        "resolved_parallelism": resolved_parallelism
        if resolved_parallelism is not None
        else requested_parallelism,
        "health_ready_ms": launch_time.ns_to_ms(health_ready_ns)
        if health_ready_ns is not None
        else None,
        "health_ready_s": launch_time.ns_to_s(health_ready_ns)
        if health_ready_ns is not None
        else None,
        "models_ready_ms": launch_time.ns_to_ms(models_ready_ns)
        if models_ready_ns is not None
        else None,
        "models_ready_s": launch_time.ns_to_s(models_ready_ns)
        if models_ready_ns is not None
        else None,
        "launch_time_ms": launch_time.ns_to_ms(models_ready_ns)
        if models_ready_ns is not None
        else None,
        "launch_time_s": launch_time.ns_to_s(models_ready_ns)
        if models_ready_ns is not None
        else None,
        "status": status,
        "reason": reason,
        "case_dir": str(case_dir),
        "server_log_path": str(case_dir / "server.log"),
        "launch_tasks_path": str(launch_tasks_path),
        "launch_breakdown_path": str(case_dir / "launch_breakdown.json"),
        "launch_command": launch_command,
        "served_model_card": model_card,
        "task_events_count": len(task_events),
        "tasks": task_summary,
    }
    if extra:
        record.update(extra)
    return record


def write_case_breakdown(case_dir: Path, record: dict[str, Any]) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "case_name": record.get("case_name"),
        "model_key": record.get("model_key"),
        "family": record.get("family"),
        "gpu_count": record.get("gpu_count"),
        "setup_name": record.get("setup_name"),
        "setup_args": record.get("setup_args"),
        "requested_parallelism": record.get("requested_parallelism"),
        "resolved_parallelism": record.get("resolved_parallelism"),
        "health_ready_ms": record.get("health_ready_ms"),
        "health_ready_s": record.get("health_ready_s"),
        "models_ready_ms": record.get("models_ready_ms"),
        "models_ready_s": record.get("models_ready_s"),
        "launch_time_ms": record.get("launch_time_ms"),
        "launch_time_s": record.get("launch_time_s"),
        "status": record.get("status"),
        "reason": record.get("reason"),
        "launch_command": record.get("launch_command"),
        "server_log_path": record.get("server_log_path"),
        "launch_tasks_path": record.get("launch_tasks_path"),
        "tasks": record.get("tasks"),
        "served_model_card": record.get("served_model_card"),
    }
    launch_time.save_json(case_dir / "launch_breakdown.json", payload)


def format_failure_reason(exc: Exception, server_log_path: Path) -> str:
    tail = launch_time.read_log_tail(server_log_path)
    if tail:
        return f"{exc}\n{tail}"
    return str(exc)


def start_process(
    *,
    command: list[str],
    server_log_path: Path,
    env: dict[str, str],
) -> tuple[subprocess.Popen[str], Any]:
    return launch_time.start_logged_process(command=command, log_path=server_log_path, env=env)


def measure_promptenhancer_case(
    *,
    preset: launch_time.LaunchPreset,
    gpu_count: int,
    llm_setup: launch_time.LLMSetup,
    host: str,
    timeout_s: int,
    case_dir: Path,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    port = launch_time.find_free_port(host)
    base_url = f"http://{host}:{port}"
    server_log_path = case_dir / "server.log"
    launch_tasks_path = case_dir / "launch_tasks.jsonl"
    env = prepare_case_env(launch_tasks_path, diffusion=False)
    command = launch_time.build_promptenhancer_command(
        preset=preset,
        tp_size=gpu_count,
        host=host,
        port=port,
        llm_setup=llm_setup,
    )

    process: subprocess.Popen[str] | None = None
    health_ready_ns: int | None = None
    models_ready_ns: int | None = None
    model_card: dict[str, Any] | None = None
    try:
        logger.info("Launching %s", " ".join(command))
        process, _ = start_process(
            command=command,
            server_log_path=server_log_path,
            env=env,
        )
        health_ready_ns, models_ready_ns, model_card = wait_for_server_endpoints(
            process=process,
            base_url=base_url,
            server_log_path=server_log_path,
            timeout_s=timeout_s,
            expected_task_type=None,
        )
        task_events = load_launch_task_events(launch_tasks_path)
        task_summary = summarize_launch_tasks(task_events, family="sglang")
        return make_case_record(
            preset=preset,
            gpu_count=gpu_count,
            requested_parallelism={"tp_size": gpu_count},
            resolved_parallelism={"tp_size": gpu_count},
            case_name=f"tp{gpu_count}__{llm_setup.key}",
            case_dir=case_dir,
            launch_command=command,
            status="completed",
            reason=None,
            health_ready_ns=health_ready_ns,
            models_ready_ns=models_ready_ns,
            model_card=model_card,
            launch_tasks_path=launch_tasks_path,
            task_events=task_events,
            task_summary=task_summary,
            extra={
                "setup_name": llm_setup.key,
                "setup_args": llm_setup_args(llm_setup),
                "ready_url_health": f"{base_url}/health",
                "ready_url_models": f"{base_url}/v1/models",
            },
        )
    except LaunchWaitError as exc:
        task_events = load_launch_task_events(launch_tasks_path)
        task_summary = summarize_launch_tasks(task_events, family="sglang")
        return make_case_record(
            preset=preset,
            gpu_count=gpu_count,
            requested_parallelism={"tp_size": gpu_count},
            resolved_parallelism={"tp_size": gpu_count},
            case_name=f"tp{gpu_count}__{llm_setup.key}",
            case_dir=case_dir,
            launch_command=command,
            status="failed",
            reason=str(exc),
            health_ready_ns=exc.health_ready_ns,
            models_ready_ns=exc.models_ready_ns,
            model_card=model_card,
            launch_tasks_path=launch_tasks_path,
            task_events=task_events,
            task_summary=task_summary,
            extra={
                "setup_name": llm_setup.key,
                "setup_args": llm_setup_args(llm_setup),
                "ready_url_health": f"{base_url}/health",
                "ready_url_models": f"{base_url}/v1/models",
            },
        )
    except Exception as exc:
        task_events = load_launch_task_events(launch_tasks_path)
        task_summary = summarize_launch_tasks(task_events, family="sglang")
        return make_case_record(
            preset=preset,
            gpu_count=gpu_count,
            requested_parallelism={"tp_size": gpu_count},
            resolved_parallelism={"tp_size": gpu_count},
            case_name=f"tp{gpu_count}__{llm_setup.key}",
            case_dir=case_dir,
            launch_command=command,
            status="failed",
            reason=format_failure_reason(exc, server_log_path),
            health_ready_ns=health_ready_ns,
            models_ready_ns=models_ready_ns,
            model_card=model_card,
            launch_tasks_path=launch_tasks_path,
            task_events=task_events,
            task_summary=task_summary,
            extra={
                "setup_name": llm_setup.key,
                "setup_args": llm_setup_args(llm_setup),
                "ready_url_health": f"{base_url}/health",
                "ready_url_models": f"{base_url}/v1/models",
            },
        )
    finally:
        launch_time.stop_server(process)


def measure_diffusion_case(
    *,
    preset: launch_time.LaunchPreset,
    gpu_count: int,
    run_config: RunConfig,
    host: str,
    timeout_s: int,
    case_dir: Path,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    port = launch_time.find_free_port(host)
    scheduler_port = launch_time.find_free_port(host)
    master_port = launch_time.find_free_port(host)
    base_url = f"http://{host}:{port}"
    server_log_path = case_dir / "server.log"
    launch_tasks_path = case_dir / "launch_tasks.jsonl"
    env = prepare_case_env(launch_tasks_path, diffusion=True)
    command = build_diffusion_command(
        preset=preset,
        run_config=run_config,
        host=host,
        port=port,
        scheduler_port=scheduler_port,
        master_port=master_port,
        case_dir=case_dir,
    )

    process: subprocess.Popen[str] | None = None
    health_ready_ns: int | None = None
    models_ready_ns: int | None = None
    model_card: dict[str, Any] | None = None
    try:
        logger.info("Launching %s", " ".join(command))
        process, _ = start_process(
            command=command,
            server_log_path=server_log_path,
            env=env,
        )
        health_ready_ns, models_ready_ns, model_card = wait_for_server_endpoints(
            process=process,
            base_url=base_url,
            server_log_path=server_log_path,
            timeout_s=timeout_s,
            expected_task_type=preset.expected_task_type,
        )
        task_events = load_launch_task_events(launch_tasks_path)
        task_summary = summarize_launch_tasks(task_events, family="sglang-diffusion")
        return make_case_record(
            preset=preset,
            gpu_count=gpu_count,
            requested_parallelism=asdict(run_config),
            resolved_parallelism=asdict(run_config),
            case_name=run_config.name,
            case_dir=case_dir,
            launch_command=command,
            status="completed",
            reason=None,
            health_ready_ns=health_ready_ns,
            models_ready_ns=models_ready_ns,
            model_card=model_card,
            launch_tasks_path=launch_tasks_path,
            task_events=task_events,
            task_summary=task_summary,
            extra={
                "ready_url_health": f"{base_url}/health",
                "ready_url_models": f"{base_url}/v1/models",
            },
        )
    except LaunchWaitError as exc:
        task_events = load_launch_task_events(launch_tasks_path)
        task_summary = summarize_launch_tasks(task_events, family="sglang-diffusion")
        return make_case_record(
            preset=preset,
            gpu_count=gpu_count,
            requested_parallelism=asdict(run_config),
            resolved_parallelism=asdict(run_config),
            case_name=run_config.name,
            case_dir=case_dir,
            launch_command=command,
            status="failed",
            reason=str(exc),
            health_ready_ns=exc.health_ready_ns,
            models_ready_ns=exc.models_ready_ns,
            model_card=model_card,
            launch_tasks_path=launch_tasks_path,
            task_events=task_events,
            task_summary=task_summary,
            extra={
                "ready_url_health": f"{base_url}/health",
                "ready_url_models": f"{base_url}/v1/models",
            },
        )
    except Exception as exc:
        task_events = load_launch_task_events(launch_tasks_path)
        task_summary = summarize_launch_tasks(task_events, family="sglang-diffusion")
        return make_case_record(
            preset=preset,
            gpu_count=gpu_count,
            requested_parallelism=asdict(run_config),
            resolved_parallelism=asdict(run_config),
            case_name=run_config.name,
            case_dir=case_dir,
            launch_command=command,
            status="failed",
            reason=format_failure_reason(exc, server_log_path),
            health_ready_ns=health_ready_ns,
            models_ready_ns=models_ready_ns,
            model_card=model_card,
            launch_tasks_path=launch_tasks_path,
            task_events=task_events,
            task_summary=task_summary,
            extra={
                "ready_url_health": f"{base_url}/health",
                "ready_url_models": f"{base_url}/v1/models",
            },
        )
    finally:
        launch_time.stop_server(process)


def refresh_human_summary(summary: dict[str, Any]) -> None:
    grouped: dict[str, dict[str, Any]] = {}
    for record in summary.get("cases", []):
        model_summary = grouped.setdefault(
            record["model_key"],
            {
                "family": record["family"],
                "completed": 0,
                "failed": 0,
                "skipped": 0,
                "completed_cases": [],
            },
        )
        status = record.get("status")
        if status == "completed":
            model_summary["completed"] += 1
            model_summary["completed_cases"].append(
                {
                    "case_name": record["case_name"],
                    "gpu_count": record["gpu_count"],
                    "setup_name": record.get("setup_name"),
                    "health_ready_ms": record["health_ready_ms"],
                    "models_ready_ms": record["models_ready_ms"],
                }
            )
        elif status == "skipped":
            model_summary["skipped"] += 1
        else:
            model_summary["failed"] += 1
    summary["human_summary"] = {
        "models": grouped,
        "total_cases": len(summary.get("cases", [])),
    }


def build_summary(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    visible_gpu_count: int,
) -> dict[str, Any]:
    selected_models = launch_time.parse_model_keys(args.models)
    selected_llm_setups = launch_time.parse_llm_setup_keys(args.llm_setups)
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_root": str(output_dir),
        "visible_gpu_count": visible_gpu_count,
        "parallel_degrees": launch_time.parse_parallel_degrees(args),
        "selected_models": selected_models,
        "selected_llm_setups": selected_llm_setups,
        "keep_artifacts": args.keep_artifacts,
        "launch_readiness_semantics": {
            "health_ready_ms": (
                "Measured from subprocess start until /health first returns HTTP 200."
            ),
            "models_ready_ms": (
                "Measured from subprocess start until /v1/models first returns HTTP 200. "
                "For diffusion presets, the returned task_type must match the expected preset task_type."
            ),
        },
        "task_breakdown_semantics": {
            "launch_tasks_jsonl": (
                "Structured begin/end task events emitted by the server runtime when "
                "SGLANG_LAUNCH_TASK_LOG_PATH is set."
            ),
            "envelope_vs_leaf_tasks": (
                "Some tasks are coarse envelope timers (for example worker_init_total, "
                "build_pipeline_total, model_runner_initialize_total, parent_wait_workers_ready) "
                "that intentionally overlap with smaller leaf tasks. Do not sum every task "
                "to estimate launch_time_ms."
            ),
            "elapsed_ms_sum": (
                "Sum of all observed end-event durations for the same task key."
            ),
            "elapsed_ms_max": (
                "Max single observed end-event duration for the same task key. "
                "This is often the most useful wall-clock-like number for per-rank tasks."
            ),
            "observed_false": (
                "The task was expected in the schema but was not observed in this launch."
            ),
            "diffusion_cuda_graph_capture": (
                "With diffusion startup warmup disabled, a separate launch-time CUDA graph "
                "capture step may not run at all. In that case the task remains "
                "observed=false instead of reporting a synthetic duration."
            ),
        },
        "cases": [],
        "human_summary": {},
    }


def main() -> None:
    args = parse_args()
    model_keys = launch_time.parse_model_keys(args.models)
    llm_setup_keys = launch_time.parse_llm_setup_keys(args.llm_setups)
    parallel_degrees = launch_time.parse_parallel_degrees(args)
    visible_gpu_count = launch_time.resolve_visible_gpu_count()
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "server_launch_breakdown_summary.json"
    summary = build_summary(
        args=args,
        output_dir=output_dir,
        visible_gpu_count=visible_gpu_count,
    )

    for model_key in model_keys:
        preset = PRESETS[model_key]
        logger.info("Starting launch-breakdown sweep for preset=%s", model_key)
        for gpu_count in parallel_degrees:
            if preset.family == "sglang":
                for setup_key in llm_setup_keys:
                    llm_setup = launch_time.LLM_SETUPS[setup_key]
                    case_dir = (
                        output_dir
                        / preset.output_subdir
                        / f"tp{gpu_count}"
                        / llm_setup.key
                    )
                    if visible_gpu_count < gpu_count:
                        record = make_case_record(
                            preset=preset,
                            gpu_count=gpu_count,
                            requested_parallelism={"tp_size": gpu_count},
                            resolved_parallelism={"tp_size": gpu_count},
                            case_name=f"tp{gpu_count}__{llm_setup.key}",
                            case_dir=case_dir,
                            launch_command=None,
                            status="skipped",
                            reason=(
                                f"Requested gpu_count={gpu_count}, but only {visible_gpu_count} "
                                f"visible GPU(s) are available."
                            ),
                            health_ready_ns=None,
                            models_ready_ns=None,
                            model_card=None,
                            launch_tasks_path=case_dir / "launch_tasks.jsonl",
                            task_events=[],
                            task_summary=summarize_launch_tasks([], family="sglang"),
                            extra={
                                "setup_name": llm_setup.key,
                                "setup_args": llm_setup_args(llm_setup),
                            },
                        )
                        summary["cases"].append(record)
                        write_case_breakdown(case_dir, record)
                        refresh_human_summary(summary)
                        launch_time.save_json(summary_path, summary)
                        launch_time.cleanup_case_dir(case_dir, args.keep_artifacts)
                        continue

                    logger.info(
                        "Measuring preset=%s family=sglang tp=%s setup=%s",
                        model_key,
                        gpu_count,
                        llm_setup.key,
                    )
                    record = measure_promptenhancer_case(
                        preset=preset,
                        gpu_count=gpu_count,
                        llm_setup=llm_setup,
                        host=args.host,
                        timeout_s=args.wait_timeout,
                        case_dir=case_dir,
                    )
                    summary["cases"].append(record)
                    write_case_breakdown(case_dir, record)
                    refresh_human_summary(summary)
                    launch_time.save_json(summary_path, summary)
                    launch_time.cleanup_case_dir(case_dir, args.keep_artifacts)
                continue

            if visible_gpu_count < gpu_count:
                case_dir = output_dir / preset.output_subdir / f"gpu{gpu_count}"
                record = make_case_record(
                    preset=preset,
                    gpu_count=gpu_count,
                    requested_parallelism={"num_gpus": gpu_count},
                    resolved_parallelism={"num_gpus": gpu_count},
                    case_name=f"gpu{gpu_count}",
                    case_dir=case_dir,
                    launch_command=None,
                    status="skipped",
                    reason=(
                        f"Requested gpu_count={gpu_count}, but only {visible_gpu_count} "
                        f"visible GPU(s) are available."
                    ),
                    health_ready_ns=None,
                    models_ready_ns=None,
                    model_card=None,
                    launch_tasks_path=case_dir / "launch_tasks.jsonl",
                    task_events=[],
                    task_summary=summarize_launch_tasks(
                        [], family=normalized_family(preset)
                    ),
                )
                summary["cases"].append(record)
                write_case_breakdown(case_dir, record)
                refresh_human_summary(summary)
                launch_time.save_json(summary_path, summary)
                launch_time.cleanup_case_dir(case_dir, args.keep_artifacts)
                continue

            run_configs = launch_time.build_run_configs(gpu_count)
            logger.info(
                "preset=%s gpu=%s will measure %s explicit tp*sp=P launch breakdown case(s)",
                model_key,
                gpu_count,
                len(run_configs),
            )
            total_configs = len(run_configs)
            for config_idx, run_config in enumerate(run_configs, start=1):
                case_dir = (
                    output_dir / preset.output_subdir / f"gpu{gpu_count}" / run_config.name
                )
                logger.info(
                    "Measuring preset=%s gpu=%s config=%s (%s/%s)",
                    model_key,
                    gpu_count,
                    run_config.name,
                    config_idx,
                    total_configs,
                )
                record = measure_diffusion_case(
                    preset=preset,
                    gpu_count=gpu_count,
                    run_config=run_config,
                    host=args.host,
                    timeout_s=args.wait_timeout,
                    case_dir=case_dir,
                )
                summary["cases"].append(record)
                write_case_breakdown(case_dir, record)
                refresh_human_summary(summary)
                launch_time.save_json(summary_path, summary)
                launch_time.cleanup_case_dir(case_dir, args.keep_artifacts)

    refresh_human_summary(summary)
    launch_time.save_json(summary_path, summary)
    logger.info("Launch-breakdown summary written to %s", summary_path)


if __name__ == "__main__":
    main()
