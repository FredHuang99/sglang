#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark diffusion cold-start weight loading setups.

This script launches the diffusion ``serve`` entrypoint repeatedly,
keeps every per-run weight-load JSON as backup, and writes a compact Markdown
and CSV summary. Phase5 warm-pool support is in-process; subprocess repeats are
still useful for launch baselines, but cross-process warm-pool hits require a
future shared-memory or daemon design.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import shlex
import signal
import statistics
import subprocess
import sys
import time
import threading
import urllib.error
import urllib.request
from queue import Empty, Queue
from pathlib import Path
from typing import Any


TIME_FIELDS_MS = {
    "weight_load:discover_files_ms": "discover_s",
    "weight_load:read_safetensors_ms": "read_s",
    "weight_load:cpu_materialize_ms": "cpu_materialize_s",
    "weight_load:pin_memory_ms": "pin_memory_s",
    "weight_load:h2d_or_param_copy_ms": "h2d_or_param_copy_s",
    "weight_load:d2h_or_offload_ms": "d2h_or_offload_s",
    "weight_load:nccl_broadcast_ms": "nccl_broadcast_s",
    "weight_load:rank0_wait_ms": "rank0_wait_s",
}
BYTE_FIELDS = {
    "weight_load:total_bytes": "total_gib",
    "weight_load:pinned_bytes": "pinned_gib",
    "weight_load:broadcast_bytes": "broadcast_gib",
    "weight_load:warm_pool_store_bytes": "warm_pool_store_gib",
}
COUNT_FIELDS = {
    "weight_load:staged_tensor_count": "staged_tensor_count",
    "weight_load:pinned_tensor_count": "pinned_tensor_count",
    "weight_load:broadcast_tensor_count": "broadcast_tensor_count",
}
BOOL_FIELDS = {
    "weight_load:warm_pool_hit": "warm_pool_hit",
}
ERROR_FIELDS = (
    "error",
    "weight_load:pin_memory_error",
    "weight_load:broadcast_error",
    "weight_load:warm_pool_error",
)
READ_BACKEND_FIELD = "weight_load:read_backend"
RUNAI_STREAM_RE = re.compile(
    r"\[RunAI Streamer\].*?stream\s+([0-9.]+)\s+GiB.*?:\s+([0-9.]+)s"
)

COMPAT_PROFILE_FLAGS = {
    "none": [],
    # Kept for backward CLI compatibility. The Z-Image launch-breakdown flags
    # are now part of the always-on reference template below.
    "zimage-launch-breakdown": [],
}

REFERENCE_TEMPLATE_FLAGS = [
    "--warmup",
    "false",
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
    "--tp-size",
    "1",
]

REFERENCE_IGNORED_ARGS = {
    "model-path",
    "port",
    "scheduler-port",
    "master-port",
    "output-path",
    "input-save-path",
    "profile-enabled",
    "profile-output-dir",
    "profile-run-id",
    "strict-ports",
}

SERVER_ARGS_EXPECTED = {
    "dit_cpu_offload": False,
    "dit_layerwise_offload": False,
    "text_encoder_cpu_offload": False,
    "image_encoder_cpu_offload": False,
    "vae_cpu_offload": False,
    "pin_cpu_memory": False,
    "use_fsdp_inference": False,
    "tp_size": 1,
}

SETUP_FLAGS = {
    "baseline": [],
    "none": [],
    "pageable": [
        "--diffusion-weight-staging",
        "pageable",
        "--diffusion-weight-load-mode",
        "rank0-broadcast",
        "--diffusion-weight-broadcast-components",
        "transformer,vae",
    ],
    "pinned": [
        "--diffusion-weight-staging",
        "pinned",
        "--diffusion-weight-load-mode",
        "rank0-broadcast",
        "--diffusion-weight-broadcast-components",
        "transformer,vae",
    ],
    "warm-pool": [
        "--diffusion-weight-staging",
        "pageable",
        "--diffusion-weight-load-mode",
        "rank0-broadcast",
        "--diffusion-weight-broadcast-components",
        "transformer,vae",
        "--diffusion-weight-warm-pool",
        "pageable",
        "--diffusion-weight-warm-pool-components",
        "transformer,vae",
    ],
}

COMMON_ENV_OVERRIDES = {
    "SGLANG_USE_RUNAI_MODEL_STREAMER": "false",
}
SETUP_ENV_OVERRIDES: dict[str, dict[str, str]] = {}

MEASUREMENT_PASS_CHOICES = ("timing", "metrics", "both")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run diffusion cold-start launch benchmarks and summarize JSON profiles."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--sp-degree", type=int, required=True)
    parser.add_argument("--ulysses-degree", type=int, required=True)
    parser.add_argument("--ring-degree", type=int, required=True)
    parser.add_argument(
        "--setups",
        default="baseline,pageable,pinned,warm-pool",
        help=(
            "Comma-separated setup names: baseline, none, pageable, pinned, "
            "warm-pool. All setups disable RunAI model streamer for a fair "
            "weight-loading comparison."
        ),
    )
    parser.add_argument("--repeat-k", type=int, default=3)
    parser.add_argument("--profile-output-dir", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--base-run-id", default=None)
    parser.add_argument(
        "--launch-cwd",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--python", default=sys.executable or "python3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--launch-entrypoint",
        choices=("module", "script"),
        default="module",
        help="Use the module serve entrypoint by default; script is legacy/debug only.",
    )
    parser.add_argument(
        "--attention-backend",
        default=None,
        help=(
            "Optional debug override. The reference-aligned Z-Image benchmark "
            "does not pass this flag by default."
        ),
    )
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--ready-mode",
        choices=("health", "models"),
        default="models",
        help="Official launch wall-time endpoint. models uses /v1/models.",
    )
    parser.add_argument(
        "--measurement-pass",
        choices=MEASUREMENT_PASS_CHOICES,
        default="timing",
        help=(
            "timing omits --profile-enabled for official wall time; metrics "
            "emits weight-load JSON; both runs timing then metrics and reports "
            "them separately."
        ),
    )
    parser.add_argument(
        "--compat-profile-preset",
        choices=tuple(COMPAT_PROFILE_FLAGS),
        default="none",
        help="Deprecated compatibility option; reference flags are always applied.",
    )
    parser.add_argument(
        "--reference-launch-summary",
        default=None,
        help="Path to server_launch_breakdown_summary_diffusion.json for command diff checks.",
    )
    parser.add_argument(
        "--diagnostic-module-profile",
        action="store_true",
        help="Also write launch module-load JSON; disabled for reference-aligned benchmark runs.",
    )
    parser.add_argument(
        "--fail-on-native-text-encoder-fallback",
        action="store_true",
        help="Mark a run failed if text_encoder falls back to native transformers.",
    )
    parser.add_argument(
        "--extra-launch-arg",
        action="append",
        default=[],
        help="Additional launch arg as key=value, e.g. port=30042.",
    )
    parser.add_argument(
        "--extra-launch-args",
        default="",
        help="Additional raw launch args, parsed with shlex.",
    )
    return parser.parse_args()


def split_setups(value: str) -> list[str]:
    setups = [item.strip() for item in value.replace(";", ",").split(",")]
    setups = [item for item in setups if item]
    unknown = [item for item in setups if item not in SETUP_FLAGS]
    if unknown:
        raise ValueError(f"Unknown setups {unknown}; choices={sorted(SETUP_FLAGS)}")
    return setups


def measurement_passes(value: str) -> list[str]:
    if value == "both":
        return ["timing", "metrics"]
    if value not in ("timing", "metrics"):
        raise ValueError(f"Unknown measurement pass {value!r}")
    return [value]


def setup_env_overrides(setup: str) -> dict[str, str]:
    values = dict(COMMON_ENV_OVERRIDES)
    values.update(SETUP_ENV_OVERRIDES.get(setup, {}))
    return values


def extra_launch_args(args: argparse.Namespace) -> list[str]:
    values: list[str] = []
    for item in args.extra_launch_arg:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--extra-launch-arg must be key=value, got {item!r}")
        values.extend([f"--{key.replace('_', '-')}", value])
    if args.extra_launch_args:
        values.extend(shlex.split(args.extra_launch_args))
    return values


def _run_benchmark_dir(args: argparse.Namespace, run_id: str) -> Path:
    return Path(args.profile_output_dir) / f"{run_id}_launch_benchmark"


def build_command(
    args: argparse.Namespace,
    setup: str,
    run_id: str,
    *,
    profile_enabled: bool,
) -> list[str]:
    ports = getattr(args, "_current_ports", None) or {}
    run_profile_dir = Path(
        getattr(args, "_current_run_profile_dir", _run_benchmark_dir(args, run_id))
    )
    if getattr(args, "launch_entrypoint", "module") == "module":
        command = [
            args.python,
            "-m",
            "sglang.multimodal_gen.runtime.entrypoints.cli.main",
            "serve",
        ]
    else:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "python"
            / "sglang"
            / "multimodal_gen"
            / "runtime"
            / "launch_server.py"
        )
        command = [args.python, str(script_path)]
    output_path = run_profile_dir / "outputs"
    input_save_path = run_profile_dir / "uploads"
    command.extend(
        [
            "--model-path",
            args.model_path,
            "--host",
            args.host,
            "--port",
            str(ports.get("port", find_free_port(args.host))),
            "--scheduler-port",
            str(ports.get("scheduler_port", find_free_port(args.host))),
            "--master-port",
            str(ports.get("master_port", find_free_port(args.host))),
            "--strict-ports",
            "true",
            "--num-gpus",
            str(args.num_gpus),
            "--warmup",
            "false",
            "--output-path",
            str(output_path),
            "--input-save-path",
            str(input_save_path),
        ]
    )
    command.extend(REFERENCE_TEMPLATE_FLAGS[2:])
    if args.model_id:
        command.extend(["--model-id", args.model_id])
    command.extend(
        [
            "--sp-degree",
            str(args.sp_degree),
            "--ulysses-degree",
            str(args.ulysses_degree),
            "--ring-degree",
            str(args.ring_degree),
        ]
    )
    if profile_enabled:
        command.extend(
            [
                "--profile-enabled",
                "--profile-output-dir",
                args.profile_output_dir,
                "--profile-run-id",
                run_id,
            ]
        )
    if args.attention_backend:
        command.extend(["--attention-backend", args.attention_backend])
    if getattr(args, "diagnostic_module_profile", False):
        command.append("--launch-module-profile-enabled")
    command.extend(COMPAT_PROFILE_FLAGS[args.compat_profile_preset])
    command.extend(SETUP_FLAGS[setup])
    command.extend(extra_launch_args(args))
    return command


def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=20)
    except Exception:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass


def _enqueue_stdout(stdout, queue: Queue[str], log_path: Path) -> None:
    try:
        with open(log_path, "w", encoding="utf-8") as log_fp:
            for line in iter(stdout.readline, ""):
                log_fp.write(line)
                log_fp.flush()
                queue.put(line)
    finally:
        try:
            stdout.close()
        except Exception:
            pass


def _probe_http(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:
            return 200 <= int(response.status) < 300
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        return False


def _extract_server_arg_tokens(command: list[str]) -> list[str]:
    if "serve" in command:
        return command[command.index("serve") + 1 :]
    for index, token in enumerate(command):
        if token.endswith("launch_server.py"):
            return command[index + 1 :]
    return command


def _parse_server_arg_map(tokens: list[str]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        key = token[2:]
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            value = tokens[index + 1]
            index += 2
        else:
            value = "true"
            index += 1
        parsed.setdefault(key, []).append(str(value))
    return parsed


def normalized_server_arg_map(command: list[str]) -> dict[str, list[str]]:
    parsed = _parse_server_arg_map(_extract_server_arg_tokens(command))
    return {
        key: values
        for key, values in parsed.items()
        if key not in REFERENCE_IGNORED_ARGS
        and not key.startswith("profile-")
        and not key.startswith("diffusion-weight-")
    }


def diff_reference_command(
    reference_command: list[str],
    command: list[str],
) -> list[str]:
    reference = normalized_server_arg_map(reference_command)
    current = normalized_server_arg_map(command)
    diffs: list[str] = []
    for key in sorted(set(reference) | set(current)):
        expected = reference.get(key)
        actual = current.get(key)
        if expected != actual:
            diffs.append(f"--{key}: expected={expected} actual={actual}")
    return diffs


def load_reference_case(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.reference_launch_summary:
        return None
    with open(args.reference_launch_summary, encoding="utf-8") as fp:
        data = json.load(fp)
    for case in data.get("cases", []):
        parallelism = case.get("resolved_parallelism", {})
        if (
            case.get("model_key") == "wan2.2-ti2v-5b"
            and int(case.get("gpu_count", -1)) == int(args.num_gpus)
            and int(parallelism.get("tp_size", -1)) == 1
            and int(parallelism.get("sp_degree", -1)) == int(args.sp_degree)
            and int(parallelism.get("ulysses_degree", -1))
            == int(args.ulysses_degree)
            and int(parallelism.get("ring_degree", -1)) == int(args.ring_degree)
        ):
            return case
    raise ValueError(
        "No matching z-image reference case found for "
        f"num_gpus={args.num_gpus}, tp_size=1, sp_degree={args.sp_degree}, "
        f"ulysses_degree={args.ulysses_degree}, ring_degree={args.ring_degree}"
    )


def _failed_run_result(
    *,
    args: argparse.Namespace,
    setup: str,
    run_id: str,
    command: list[str],
    ports: dict[str, int],
    run_meta_path: Path,
    server_log_path: Path,
    failure_reason: str,
    command_diff: list[str],
    reference_case: dict[str, Any] | None,
    measurement_pass: str,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    result = {
        "setup": setup,
        "measurement_pass": measurement_pass,
        "run_id": run_id,
        "launch_wall_s": 0.0,
        "ready": False,
        "http_ready": False,
        "health_ready_s": None,
        "models_ready_s": None,
        "log_ready_s": None,
        "exit_code": None,
        "failure_reason": failure_reason,
        "command": " ".join(shlex.quote(part) for part in command),
        "command_argv": command,
        "command_diff": command_diff,
        "reference_aligned": False,
        "reference_case_name": reference_case.get("case_name") if reference_case else None,
        "reference_launch_time_s": reference_case.get("launch_time_s")
        if reference_case
        else None,
        "ports": ports,
        "host": args.host,
        "ready_mode": args.ready_mode,
        "compat_profile_preset": args.compat_profile_preset,
        "start_epoch_s": time.time(),
        "server_log_path": str(server_log_path),
        "run_meta_path": str(run_meta_path),
        "launch_weight_load_dir": str(
            Path(args.profile_output_dir) / f"{run_id}_launch_weight_load"
        ),
        "launch_module_load_dir": str(
            Path(args.profile_output_dir) / f"{run_id}_launch_module_load"
        ),
        "launch_task_log_path": str(run_meta_path.parent / "launch_tasks.jsonl"),
        "env_overrides": env_overrides or {},
    }
    with open(run_meta_path, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, sort_keys=True)
        fp.write("\n")
    return result


def load_server_args_from_log(server_log_path: str | Path) -> dict[str, Any] | None:
    path = Path(server_log_path)
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "server_args:"
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            continue
    return None


def validate_server_args(server_args: dict[str, Any] | None) -> list[str]:
    if server_args is None:
        return ["server_args: missing"]
    errors: list[str] = []
    for key, expected in SERVER_ARGS_EXPECTED.items():
        actual = server_args.get(key)
        if actual != expected:
            errors.append(f"{key}: expected={expected!r} actual={actual!r}")
    return errors


def run_launch(
    args: argparse.Namespace,
    setup: str,
    run_id: str,
    *,
    measurement_pass: str,
) -> dict[str, Any]:
    profile_enabled = measurement_pass == "metrics"
    ports = {
        "port": find_free_port(args.host),
        "scheduler_port": find_free_port(args.host),
        "master_port": find_free_port(args.host),
    }
    run_profile_dir = _run_benchmark_dir(args, run_id)
    run_profile_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = run_profile_dir / "server.log"
    run_meta_path = run_profile_dir / "run_meta.json"
    launch_task_log_path = run_profile_dir / "launch_tasks.jsonl"
    args._current_ports = ports
    args._current_run_profile_dir = run_profile_dir
    command = build_command(
        args,
        setup,
        run_id,
        profile_enabled=profile_enabled,
    )
    delattr(args, "_current_ports")
    delattr(args, "_current_run_profile_dir")
    command_str = " ".join(shlex.quote(part) for part in command)
    reference_case = getattr(args, "_reference_case", None)
    command_diff = (
        diff_reference_command(reference_case["launch_command"], command)
        if reference_case is not None
        else []
    )
    if command_diff:
        return _failed_run_result(
            args=args,
            setup=setup,
            run_id=run_id,
            command=command,
            ports=ports,
            run_meta_path=run_meta_path,
            server_log_path=server_log_path,
            failure_reason="reference_command_diff",
            command_diff=command_diff,
            reference_case=reference_case,
            measurement_pass=measurement_pass,
            env_overrides=setup_env_overrides(setup),
        )
    health_url = f"http://{args.host}:{ports['port']}/health"
    models_url = f"http://{args.host}:{ports['port']}/v1/models"
    start = time.perf_counter()
    start_epoch_s = time.time()
    popen_kwargs: dict[str, Any] = {
        "cwd": args.launch_cwd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
    }
    if os.name == "posix":
        popen_kwargs["preexec_fn"] = os.setsid
    else:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    env = os.environ.copy()
    env["SGLANG_LAUNCH_TASK_LOG_PATH"] = str(launch_task_log_path)
    env.update(setup_env_overrides(setup))
    popen_kwargs["env"] = env

    proc = subprocess.Popen(command, **popen_kwargs)
    assert proc.stdout is not None
    output_queue: Queue[str] = Queue()
    reader_thread = threading.Thread(
        target=_enqueue_stdout,
        args=(proc.stdout, output_queue, server_log_path),
        daemon=True,
    )
    reader_thread.start()
    ready = False
    health_ready_s: float | None = None
    models_ready_s: float | None = None
    log_ready_s: float | None = None
    exit_code: int | None = None
    try:
        while True:
            try:
                line = output_queue.get(timeout=0.2)
                print(line, end="")
                if "Uvicorn running on" in line or "Application startup complete" in line:
                    log_ready_s = log_ready_s or (time.perf_counter() - start)
            except Empty:
                pass
            elapsed_s = time.perf_counter() - start
            if health_ready_s is None and _probe_http(health_url):
                health_ready_s = elapsed_s
            if models_ready_s is None and _probe_http(models_url):
                models_ready_s = elapsed_s
            if args.ready_mode == "health":
                ready = health_ready_s is not None
            else:
                ready = models_ready_s is not None
            if ready:
                break
            exit_code = proc.poll()
            if exit_code is not None:
                break
            if elapsed_s > args.timeout_s:
                break
    finally:
        elapsed_s = time.perf_counter() - start
        if args.ready_mode == "health" and health_ready_s is not None:
            wall_s = health_ready_s
        elif args.ready_mode == "models" and models_ready_s is not None:
            wall_s = models_ready_s
        else:
            wall_s = elapsed_s
        terminate_process(proc)
        reader_thread.join(timeout=5)
    if exit_code is None:
        exit_code = proc.poll()

    result = {
        "setup": setup,
        "measurement_pass": measurement_pass,
        "run_id": run_id,
        "launch_wall_s": wall_s,
        "ready": ready,
        "http_ready": health_ready_s is not None or models_ready_s is not None,
        "health_ready_s": health_ready_s,
        "models_ready_s": models_ready_s,
        "log_ready_s": log_ready_s,
        "exit_code": exit_code,
        "command": command_str,
        "command_argv": command,
        "command_diff": command_diff,
        "reference_aligned": not command_diff if reference_case is not None else None,
        "reference_case_name": reference_case.get("case_name")
        if reference_case
        else None,
        "reference_launch_time_s": reference_case.get("launch_time_s")
        if reference_case
        else None,
        "ports": ports,
        "host": args.host,
        "ready_mode": args.ready_mode,
        "compat_profile_preset": args.compat_profile_preset,
        "start_epoch_s": start_epoch_s,
        "server_log_path": str(server_log_path),
        "run_meta_path": str(run_meta_path),
        "launch_weight_load_dir": str(
            Path(args.profile_output_dir) / f"{run_id}_launch_weight_load"
        ),
        "launch_module_load_dir": str(
            Path(args.profile_output_dir) / f"{run_id}_launch_module_load"
        ),
        "launch_task_log_path": str(launch_task_log_path),
        "env_overrides": setup_env_overrides(setup),
    }
    with open(run_meta_path, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, sort_keys=True)
        fp.write("\n")
    return result


def load_profile_records(profile_output_dir: str, run_id: str) -> list[dict[str, Any]]:
    run_dir = Path(profile_output_dir) / f"{run_id}_launch_weight_load"
    records: list[dict[str, Any]] = []
    if not run_dir.exists():
        return records
    for path in sorted(run_dir.glob("weight_load_*.json")):
        with open(path, encoding="utf-8") as fp:
            record = json.load(fp)
        record["_profile_json"] = str(path)
        records.append(record)
    return records


def load_module_records(profile_output_dir: str, run_id: str) -> list[dict[str, Any]]:
    run_dir = Path(profile_output_dir) / f"{run_id}_launch_module_load"
    records: list[dict[str, Any]] = []
    if not run_dir.exists():
        return records
    for path in sorted(run_dir.glob("module_load_*.json")):
        with open(path, encoding="utf-8") as fp:
            record = json.load(fp)
        record["_profile_json"] = str(path)
        records.append(record)
    return records


def load_launch_task_records(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    task_path = Path(path)
    if not task_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in task_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _safe_metric_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_") or "unknown"


def summarize_launch_tasks(
    run_result: dict[str, Any],
    task_records: list[dict[str, Any]],
) -> dict[str, float]:
    summary: dict[str, float] = {}
    grouped: dict[str, list[float]] = {}
    starts: list[float] = []
    ends: list[float] = []
    for record in task_records:
        elapsed_s = _numeric(record.get("elapsed_s"))
        if elapsed_s is None:
            elapsed_ms = _numeric(record.get("elapsed_ms"))
            elapsed_s = elapsed_ms / 1000.0 if elapsed_ms is not None else None
        if elapsed_s is None:
            continue
        task = _safe_metric_name(str(record.get("task") or "unknown"))
        component = record.get("component")
        prefix = f"launch_task_{task}"
        if component:
            prefix = f"{prefix}_{_safe_metric_name(str(component))}"
        grouped.setdefault(prefix, []).append(elapsed_s)
        end_s = _numeric(record.get("timestamp_s"))
        if end_s is not None:
            ends.append(end_s)
            starts.append(end_s - elapsed_s)

    for prefix, values in grouped.items():
        summary[f"{prefix}_avg_s"] = statistics.mean(values)
        summary[f"{prefix}_max_s"] = max(values)
        summary[f"{prefix}_count"] = float(len(values))

    if starts and ends:
        observed_span_s = max(ends) - min(starts)
        summary["launch_task_observed_span_s"] = max(0.0, observed_span_s)
        wall_s = _numeric(run_result.get("launch_wall_s"))
        if wall_s is not None:
            summary["launch_task_unexplained_s"] = max(0.0, wall_s - observed_span_s)
    return summary


def collect_runai_stream_info(server_log_path: str | Path | None) -> dict[str, float]:
    if not server_log_path:
        return {}
    path = Path(server_log_path)
    if not path.exists():
        return {}
    result = {
        "runai_transformer_stream_count": 0.0,
        "runai_transformer_stream_time_s": 0.0,
        "runai_text_encoder_stream_count": 0.0,
        "runai_text_encoder_stream_time_s": 0.0,
    }
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = RUNAI_STREAM_RE.search(line)
        if not match:
            continue
        gib = float(match.group(1))
        elapsed_s = float(match.group(2))
        # Z-Image transformer shards total about 11.5 GiB. Text encoder shards
        # are smaller in the reference setup, so this threshold is sufficient
        # for diagnosis without adding intrusive runtime tags.
        bucket = "transformer" if gib >= 8.0 else "text_encoder"
        result[f"runai_{bucket}_stream_count"] += 1.0
        result[f"runai_{bucket}_stream_time_s"] += elapsed_s
    return result


def collect_text_encoder_source_info_from_log(
    server_log_path: str | Path | None,
) -> tuple[list[str], int]:
    if not server_log_path:
        return [], 0
    path = Path(server_log_path)
    if not path.exists():
        return [], 0
    sources: set[str] = set()
    fallback_count = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "component=text_encoder" in line and "ProfileModuleLoadDone" in line:
            match = re.search(r"\bsource=([^\s]+)", line)
            if match:
                sources.add(match.group(1))
        if "Loaded text_encoder:" in line:
            match = re.search(r"\(([^()]+)\s+version\)", line)
            if match:
                sources.add(match.group(1))
        if "Native component text_encoder" in line:
            sources.add("native")
            fallback_count += 1
        elif "falling back to native version" in line and "text_encoder" in line:
            fallback_count += 1
    return sorted(sources), fallback_count


def collect_text_encoder_source_info(
    module_records: list[dict[str, Any]],
) -> tuple[list[str], int]:
    text_encoder_records = [
        record
        for record in module_records
        if record.get("component") == "text_encoder"
    ]
    sources = sorted(
        {str(record.get("source") or "unknown") for record in text_encoder_records}
    )
    fallback_count = sum(
        1
        for record in text_encoder_records
        if record.get("source") == "native" or bool(record.get("fallback"))
    )
    return sources, fallback_count


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rank_bucket(record: dict[str, Any]) -> str:
    rank = str(record.get("sp_rank", record.get("rank", "unknown")))
    return "rank0" if rank == "0" else "nonrank"


def _metric_values(record: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for src, dst in TIME_FIELDS_MS.items():
        value = _numeric(record.get(src))
        if value is not None:
            values[dst] = value / 1000.0
    for src, dst in BYTE_FIELDS.items():
        value = _numeric(record.get(src))
        if value is not None:
            values[dst] = value / (1024**3)
    for src, dst in COUNT_FIELDS.items():
        value = _numeric(record.get(src))
        if value is not None:
            values[dst] = value
    for src, dst in BOOL_FIELDS.items():
        value = _numeric(record.get(src))
        if value is not None:
            values[dst] = value
    values["error_count"] = float(
        sum(1 for field in ERROR_FIELDS if record.get(field) not in (None, "", False))
    )
    return values


def summarize_run(
    run_result: dict[str, Any],
    records: list[dict[str, Any]],
    module_records: list[dict[str, Any]] | None = None,
    launch_task_records: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    summary: dict[str, float] = {
        "launch_wall_s": float(run_result["launch_wall_s"]),
        "ready": 1.0 if run_result["ready"] else 0.0,
    }
    measurement_pass = str(run_result.get("measurement_pass") or "")
    if measurement_pass in ("timing", "metrics"):
        pass_metric = (
            "wall_timing_s" if measurement_pass == "timing" else "wall_profiled_s"
        )
        summary[pass_metric] = float(run_result["launch_wall_s"])
    for field in ("health_ready_s", "models_ready_s", "log_ready_s"):
        value = _numeric(run_result.get(field))
        if value is not None:
            summary[field] = value
    for field in ("reference_launch_time_s", "reference_aligned"):
        value = _numeric(run_result.get(field))
        if value is not None:
            summary[field] = value
    reference_launch_time_s = _numeric(run_result.get("reference_launch_time_s"))
    if reference_launch_time_s is not None:
        summary["wall_delta_vs_reference_s"] = (
            float(run_result["launch_wall_s"]) - reference_launch_time_s
        )
    grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    backend_counts: dict[tuple[str, str, str], int] = {}
    for record in records:
        component = str(record.get("component", "unknown"))
        bucket = _rank_bucket(record)
        grouped.setdefault((component, bucket), []).append(_metric_values(record))
        backend = record.get(READ_BACKEND_FIELD)
        if backend:
            backend_counts[(component, bucket, str(backend))] = (
                backend_counts.get((component, bucket, str(backend)), 0) + 1
            )

    metric_names = sorted(
        {
            metric
            for values in grouped.values()
            for item in values
            for metric in item.keys()
        }
    )
    for (component, bucket), values in grouped.items():
        for metric in metric_names:
            metric_values = [item[metric] for item in values if metric in item]
            if not metric_values:
                continue
            if bucket == "rank0":
                summary[f"{component}_rank0_{metric}"] = statistics.mean(
                    metric_values
                )
            else:
                summary[f"{component}_nonrank_avg_{metric}"] = statistics.mean(
                    metric_values
                )
                summary[f"{component}_nonrank_max_{metric}"] = max(metric_values)
    for (component, bucket, backend), count in backend_counts.items():
        backend_key = _safe_metric_name(backend)
        if bucket == "rank0":
            summary[f"{component}_rank0_read_backend_{backend_key}_count"] = float(count)
        else:
            summary[f"{component}_nonrank_read_backend_{backend_key}_count"] = float(
                count
            )
    module_records = module_records or []
    module_grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    max_module_end_s: float | None = None
    for record in module_records:
        component = str(record.get("component", "unknown"))
        duration_s = float(record.get("duration_ms", 0.0)) / 1000.0
        started_at_s = _numeric(record.get("started_at_s"))
        if started_at_s is not None:
            end_s = started_at_s + duration_s
            max_module_end_s = (
                end_s if max_module_end_s is None else max(max_module_end_s, end_s)
            )
        module_grouped.setdefault((component, _rank_bucket(record)), []).append(
            {
                "module_duration_s": duration_s,
                "module_fallback_count": 1.0
                if bool(record.get("fallback"))
                else 0.0,
                "module_native_source_count": 1.0
                if record.get("source") == "native"
                else 0.0,
                "module_error_count": 1.0
                if record.get("status") != "success" or record.get("error")
                else 0.0,
            }
        )
    for (component, bucket), values in module_grouped.items():
        for metric in sorted({key for item in values for key in item}):
            metric_values = [item[metric] for item in values if metric in item]
            if bucket == "rank0":
                summary[f"{component}_rank0_{metric}"] = statistics.mean(
                    metric_values
                )
            else:
                summary[f"{component}_nonrank_avg_{metric}"] = statistics.mean(
                    metric_values
                )
                summary[f"{component}_nonrank_max_{metric}"] = max(metric_values)
    if max_module_end_s is not None and run_result.get("start_epoch_s") is not None:
        explained_s = max_module_end_s - float(run_result["start_epoch_s"])
        summary["wall_unexplained_s"] = max(0.0, summary["launch_wall_s"] - explained_s)
    runai_info = collect_runai_stream_info(run_result.get("server_log_path"))
    summary.update(runai_info)
    if launch_task_records is not None:
        summary.update(summarize_launch_tasks(run_result, launch_task_records))
    return summary


def summarize_combined_run(
    *,
    setup: str,
    timing_result: dict[str, Any] | None,
    metrics_result: dict[str, Any] | None,
    records: list[dict[str, Any]],
    module_records: list[dict[str, Any]],
) -> dict[str, float]:
    official_result = timing_result or metrics_result
    if official_result is None:
        return {"ready": 0.0}

    launch_task_records = load_launch_task_records(
        official_result.get("launch_task_log_path")
    )
    metrics_ready = bool(metrics_result and metrics_result.get("ready"))
    summary = summarize_run(
        official_result,
        records if metrics_ready else [],
        module_records if metrics_ready else [],
        launch_task_records=launch_task_records,
    )
    summary["ready"] = 1.0 if official_result.get("ready") else 0.0
    if timing_result is not None:
        summary["wall_timing_s"] = float(timing_result["launch_wall_s"])
        reference_launch_time_s = _numeric(timing_result.get("reference_launch_time_s"))
        if reference_launch_time_s is not None:
            summary["wall_delta_vs_reference_s"] = (
                float(timing_result["launch_wall_s"]) - reference_launch_time_s
            )
    if metrics_ready:
        summary["wall_profiled_s"] = float(metrics_result["launch_wall_s"])
    if timing_result is not None and metrics_ready:
        summary["profile_overhead_s"] = (
            float(metrics_result["launch_wall_s"])
            - float(timing_result["launch_wall_s"])
        )
        summary["profile_intrusive"] = (
            1.0 if abs(summary["profile_overhead_s"]) > 2.0 else 0.0
        )

        # Keep RunAI stream diagnosis from both passes visible. The official
        # timing pass explains production wall time; metrics pass explains JSONs.
        timing_runai = collect_runai_stream_info(timing_result.get("server_log_path"))
        metrics_runai = collect_runai_stream_info(metrics_result.get("server_log_path"))
        for key, value in timing_runai.items():
            summary[f"timing_{key}"] = value
        for key, value in metrics_runai.items():
            summary[f"metrics_{key}"] = value
    summary["setup_sample_count"] = 1.0
    return summary


def aggregate_by_setup(
    run_summaries: list[tuple[str, dict[str, float]]],
) -> tuple[list[str], list[dict[str, Any]]]:
    setup_to_rows: dict[str, list[dict[str, float]]] = {}
    for setup, summary in run_summaries:
        setup_to_rows.setdefault(setup, []).append(summary)

    metric_names = sorted(
        {
            metric
            for summaries in setup_to_rows.values()
            for summary in summaries
            for metric in summary.keys()
        }
    )
    columns = ["setup"]
    for metric in metric_names:
        columns.extend(
            [
                f"{metric}_avg",
                f"{metric}_p50",
                f"{metric}_min",
                f"{metric}_max",
                f"{metric}_std",
            ]
        )

    rows: list[dict[str, Any]] = []
    for setup, summaries in setup_to_rows.items():
        row: dict[str, Any] = {"setup": setup}
        for metric in metric_names:
            values = [summary[metric] for summary in summaries if metric in summary]
            if not values:
                continue
            row[f"{metric}_avg"] = statistics.mean(values)
            row[f"{metric}_p50"] = statistics.median(values)
            row[f"{metric}_min"] = min(values)
            row[f"{metric}_max"] = max(values)
            row[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return columns, rows


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_outputs(
    summary_output: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    run_results: list[dict[str, Any]],
) -> tuple[Path, Path]:
    output = Path(summary_output)
    if output.suffix.lower() == ".csv":
        csv_path = output
        md_path = output.with_suffix(".md")
    elif output.suffix.lower() == ".md":
        md_path = output
        csv_path = output.with_suffix(".csv")
    else:
        output.mkdir(parents=True, exist_ok=True)
        md_path = output / "summary.md"
        csv_path = output / "summary.csv"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})

    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write("# Diffusion Cold-Start Benchmark Summary\n\n")
        fp.write("## Setup Summary\n\n")
        fp.write("| " + " | ".join(columns) + " |\n")
        fp.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rows:
            fp.write(
                "| "
                + " | ".join(_format_cell(row.get(column)) for column in columns)
                + " |\n"
            )

        fp.write("\n## Runs\n\n")
        run_columns = [
            "setup",
            "measurement_pass",
            "run_id",
            "ready",
            "launch_wall_s",
            "health_ready_s",
            "models_ready_s",
            "log_ready_s",
            "ready_mode",
            "reference_case_name",
            "reference_launch_time_s",
            "reference_aligned",
            "failure_reason",
            "command_diff",
            "text_encoder_sources",
            "text_encoder_fallback_count",
            "launch_task_log_path",
            "env_overrides",
            "exit_code",
            "command",
        ]
        fp.write("| " + " | ".join(run_columns) + " |\n")
        fp.write("| " + " | ".join("---" for _ in run_columns) + " |\n")
        for result in run_results:
            values = {
                "setup": result["setup"],
                "measurement_pass": result.get("measurement_pass", ""),
                "run_id": result["run_id"],
                "ready": result["ready"],
                "launch_wall_s": f"{float(result['launch_wall_s']):.2f}",
                "health_ready_s": _format_cell(result.get("health_ready_s")),
                "models_ready_s": _format_cell(result.get("models_ready_s")),
                "log_ready_s": _format_cell(result.get("log_ready_s")),
                "ready_mode": result.get("ready_mode", ""),
                "reference_case_name": result.get("reference_case_name", ""),
                "reference_launch_time_s": _format_cell(
                    result.get("reference_launch_time_s")
                ),
                "reference_aligned": _format_cell(result.get("reference_aligned")),
                "failure_reason": result.get("failure_reason", ""),
                "command_diff": "<br>".join(result.get("command_diff", [])),
                "text_encoder_sources": ", ".join(
                    result.get("text_encoder_sources", [])
                ),
                "text_encoder_fallback_count": result.get(
                    "text_encoder_fallback_count", 0
                ),
                "launch_task_log_path": result.get("launch_task_log_path", ""),
                "env_overrides": json.dumps(
                    result.get("env_overrides", {}), sort_keys=True
                ),
                "exit_code": result["exit_code"],
                "command": f"`{str(result.get('command', '')).replace('|', '\\|')}`",
            }
            fp.write(
                "| "
                + " | ".join(
                    str(values.get(column, "")).replace("|", "\\|")
                    for column in run_columns
                )
                + " |\n"
            )

    return md_path, csv_path


def main() -> None:
    args = parse_args()
    setups = split_setups(args.setups)
    passes = measurement_passes(args.measurement_pass)
    if args.repeat_k <= 0:
        raise ValueError("--repeat-k must be positive")
    args._reference_case = load_reference_case(args)
    base_run_id = args.base_run_id or time.strftime("diffusion_cold_start_%Y%m%d_%H%M%S")

    run_results: list[dict[str, Any]] = []
    run_summaries: list[tuple[str, dict[str, float]]] = []
    for setup in setups:
        for index in range(args.repeat_k):
            per_pass_results: dict[str, dict[str, Any]] = {}
            for measurement_pass in passes:
                pass_suffix = measurement_pass if len(passes) > 1 else measurement_pass
                run_id = (
                    f"{base_run_id}_{setup.replace('-', '_')}_r{index + 1}"
                    f"_{pass_suffix}"
                )
                result = run_launch(
                    args,
                    setup,
                    run_id,
                    measurement_pass=measurement_pass,
                )
                if result.get("failure_reason") != "reference_command_diff":
                    resolved_server_args = load_server_args_from_log(
                        result["server_log_path"]
                    )
                    server_args_errors = validate_server_args(resolved_server_args)
                    result["server_args_validation_errors"] = server_args_errors
                    if server_args_errors:
                        result["ready"] = False
                        result["failure_reason"] = "reference_args_violation"

                module_records = load_module_records(args.profile_output_dir, run_id)
                text_encoder_sources, text_encoder_fallback_count = (
                    collect_text_encoder_source_info(module_records)
                )
                if not text_encoder_sources and text_encoder_fallback_count == 0:
                    text_encoder_sources, text_encoder_fallback_count = (
                        collect_text_encoder_source_info_from_log(
                            result.get("server_log_path")
                        )
                    )
                result["text_encoder_fallback_count"] = text_encoder_fallback_count
                result["text_encoder_sources"] = text_encoder_sources
                if (
                    args.fail_on_native_text_encoder_fallback
                    and text_encoder_fallback_count
                ):
                    result["ready"] = False
                    result["failure_reason"] = "native_text_encoder_fallback"
                run_results.append(result)
                per_pass_results[measurement_pass] = result
                with open(result["run_meta_path"], "w", encoding="utf-8") as fp:
                    json.dump(result, fp, indent=2, sort_keys=True)
                    fp.write("\n")

            timing_result = per_pass_results.get("timing")
            metrics_result = per_pass_results.get("metrics")
            metrics_run_id = (
                metrics_result.get("run_id")
                if metrics_result is not None
                else per_pass_results[passes[-1]].get("run_id")
            )
            records = load_profile_records(args.profile_output_dir, metrics_run_id)
            module_records = load_module_records(args.profile_output_dir, metrics_run_id)
            official = timing_result or metrics_result
            pair_complete = all(
                per_pass_results[measurement_pass].get("ready")
                for measurement_pass in passes
            )
            if official and official.get("ready") and pair_complete:
                run_summaries.append(
                    (
                        setup,
                        summarize_combined_run(
                            setup=setup,
                            timing_result=timing_result,
                            metrics_result=metrics_result,
                            records=records,
                            module_records=module_records,
                        ),
                    )
                )

    columns, rows = aggregate_by_setup(run_summaries)
    md_path, csv_path = write_outputs(args.summary_output, columns, rows, run_results)
    print(f"Wrote Markdown summary: {md_path}")
    print(f"Wrote CSV summary: {csv_path}")


if __name__ == "__main__":
    main()
