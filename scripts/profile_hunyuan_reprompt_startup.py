#!/usr/bin/env python3
"""Profile HunyuanImage-2.1 reprompt server startup time."""

from __future__ import annotations

import argparse
import json
import os
import pprint
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any

import requests


DEFAULT_SERVED_MODEL_NAME = "HunyuanImage-2.1-reprompt"
DEFAULT_MODEL_PATH = Path("/workspace/models/reprompt")
EXPECTED_ARCHITECTURE = "HunYuanDenseV1ForCausalLM"
EXPECTED_MODEL_TYPE = "hunyuan_v1_dense"
DEFAULT_TP_SIZES = (1, 2, 4, 8)
SUMMARY_TP_ORDER = (8, 4, 2, 1)
NUM_RUNS = 5
NUM_WARMUP_RUNS = 2
EXPECTED_MAX_MODEL_LEN = 32768
CUSTOM_ALL_REDUCE_V2_ENV = "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"
REMOVED_SERVER_ENVIRONMENT = ("SGLANG_USE_JIT_ALL_REDUCE", CUSTOM_ALL_REDUCE_V2_ENV)
ALL_REDUCE_MODES = ("legacy_v1", "custom_v2", "nccl")
ATTENTION_BACKEND_MODES = ("auto", "flashinfer", "fa3")
DECODE_CUDA_GRAPH_BACKENDS = ("full", "disabled")
LEGACY_CUSTOM_AR_GRAPH_PATTERN = re.compile(
    r"Registering \d+ cuda graph addresses"
)
CUSTOM_V2_INIT_MARKER = "All Reduce config: symmetric_memory ="
CUSTOM_V2_VMM_GRAPH_MARKER = " cuda graph addresses via "
SETUPS = {
    "non_optimized": (),
    "optimized": (
        "--chunked-prefill-size",
        "512",
        "--max-running-requests",
        "1",
        "--cuda-graph-max-bs-decode",
        "1",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile HunyuanImage-2.1 reprompt startup time for ordinary and "
            "optimized launch configurations. Each configuration runs two "
            "warmups and three measured launches."
        )
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--served-model-name",
        default=DEFAULT_SERVED_MODEL_NAME,
        help=(
            "Alias returned by SGLang /v1/models. The implementation class is "
            "selected separately from config.json architectures."
        ),
    )
    parser.add_argument(
        "--tp-sizes", nargs="+", type=int, default=list(DEFAULT_TP_SIZES)
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/outputs/hunyuan_reprompt_startup"),
    )
    parser.add_argument("--startup-timeout-s", type=float, default=1800.0)
    parser.add_argument("--ready-poll-interval-s", type=float, default=0.05)
    parser.add_argument("--cooldown-s", type=float, default=2.0)
    parser.add_argument(
        "--attention-backend",
        choices=ATTENTION_BACKEND_MODES,
        default="flashinfer",
        help="SGLang attention backend. Use fa3 on a compatible Hopper image.",
    )
    parser.add_argument(
        "--all-reduce-mode",
        choices=ALL_REDUCE_MODES,
        default="legacy_v1",
        help=(
            "legacy_v1 forces the legacy custom all-reduce, custom_v2 forces "
            "the JIT V2 path, and nccl disables custom all-reduce."
        ),
    )
    parser.add_argument(
        "--decode-cuda-graph-backend",
        choices=DECODE_CUDA_GRAPH_BACKENDS,
        default="full",
    )
    args = parser.parse_args()

    args.model_path = args.model_path.expanduser()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.tp_sizes = list(dict.fromkeys(args.tp_sizes))
    invalid_tp_sizes = [tp for tp in args.tp_sizes if tp not in DEFAULT_TP_SIZES]
    if invalid_tp_sizes:
        parser.error(f"--tp-sizes only supports 1, 2, 4, 8; got {invalid_tp_sizes}")
    if args.startup_timeout_s <= 0 or args.ready_poll_interval_s <= 0:
        parser.error("startup timeout and ready poll interval must be positive")
    if args.cooldown_s < 0:
        parser.error("--cooldown-s must be non-negative")
    return args


def resolve_model_path(model_path: Path) -> Path:
    candidate = model_path.resolve()
    direct_config = candidate / "config.json"
    nested_model_path = candidate / "reprompt"
    if direct_config.is_file():
        return candidate
    if (nested_model_path / "config.json").is_file():
        return nested_model_path
    raise FileNotFoundError(
        "Could not find the reprompt config. Expected either "
        f"{direct_config} or {nested_model_path / 'config.json'}. Pass the "
        "downloaded reprompt directory or its immediate parent."
    )


def validate_checkpoint_files(model_path: Path) -> None:
    required_files = (
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "hy.tiktoken",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenization_hy.py",
        "tokenizer_config.json",
    )
    missing_files = [name for name in required_files if not (model_path / name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Incomplete reprompt directory {model_path}; missing files: {missing_files}"
        )

    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid or empty weight_map in {index_path}")
    raw_shard_names = set(weight_map.values())
    invalid_shard_names = [
        name for name in raw_shard_names if not isinstance(name, str)
    ]
    if invalid_shard_names:
        raise ValueError(f"Invalid shard names in {index_path}: {invalid_shard_names}")
    shard_names = sorted(raw_shard_names)
    missing_shards = [name for name in shard_names if not (model_path / name).is_file()]
    if missing_shards:
        raise FileNotFoundError(
            f"Incomplete reprompt weights in {model_path}; missing shards: {missing_shards}"
        )


def load_model_identity(model_path: Path) -> tuple[str, str]:
    validate_checkpoint_files(model_path)
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or EXPECTED_ARCHITECTURE not in architectures:
        raise ValueError(
            f"Expected architecture {EXPECTED_ARCHITECTURE!r} in {config_path}, "
            f"got {architectures!r}"
        )
    model_type = config.get("model_type")
    if model_type != EXPECTED_MODEL_TYPE:
        raise ValueError(
            f"Expected model_type {EXPECTED_MODEL_TYPE!r} in {config_path}, "
            f"got {model_type!r}"
        )
    return EXPECTED_ARCHITECTURE, EXPECTED_MODEL_TYPE


def save_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary_path, path)


def prepare_output_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {path}")
        if next(path.iterdir(), None) is not None:
            raise FileExistsError(
                f"Output directory must be empty to avoid mixing profile runs: {path}"
            )
        return
    path.mkdir(parents=True)


def forced_server_environment(all_reduce_mode: str) -> dict[str, str]:
    if all_reduce_mode == "legacy_v1":
        return {CUSTOM_ALL_REDUCE_V2_ENV: "0"}
    if all_reduce_mode == "custom_v2":
        return {CUSTOM_ALL_REDUCE_V2_ENV: "1"}
    if all_reduce_mode == "nccl":
        return {}
    raise ValueError(
        f"Unsupported all-reduce mode {all_reduce_mode!r}; "
        f"expected one of {ALL_REDUCE_MODES}"
    )


def build_server_environment(all_reduce_mode: str = "legacy_v1") -> dict[str, str]:
    environment = os.environ.copy()
    for name in REMOVED_SERVER_ENVIRONMENT:
        environment.pop(name, None)
    environment.update(forced_server_environment(all_reduce_mode))
    return environment


def read_log_tail(path: Path, line_count: int = 100) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def find_free_port(host: str) -> int:
    bind_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((bind_host, 0))
        return int(sock.getsockname()[1])


def base_url(host: str, port: int) -> str:
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if ":" in connect_host and not connect_host.startswith("["):
        connect_host = f"[{connect_host}]"
    return f"http://{connect_host}:{port}"


def wait_for_port_release(host: str, port: int, timeout_s: float = 30.0) -> None:
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            with socket.create_connection((connect_host, port), timeout=0.2):
                pass
        except OSError:
            return
        time.sleep(0.1)
    raise TimeoutError(f"Port {connect_host}:{port} remained open after server shutdown")


def build_server_command(
    model_path: Path,
    served_model_name: str,
    host: str,
    port: int,
    tp_size: int,
    setup_name: str,
    all_reduce_mode: str = "legacy_v1",
    attention_backend: str = "flashinfer",
    decode_cuda_graph_backend: str = "full",
) -> list[str]:
    if all_reduce_mode not in ALL_REDUCE_MODES:
        raise ValueError(
            f"Unsupported all-reduce mode {all_reduce_mode!r}; "
            f"expected one of {ALL_REDUCE_MODES}"
        )
    if attention_backend not in ATTENTION_BACKEND_MODES:
        raise ValueError(
            f"Unsupported attention backend {attention_backend!r}; "
            f"expected one of {ATTENTION_BACKEND_MODES}"
        )
    if decode_cuda_graph_backend not in DECODE_CUDA_GRAPH_BACKENDS:
        raise ValueError(
            f"Unsupported decode CUDA graph backend {decode_cuda_graph_backend!r}; "
            f"expected one of {DECODE_CUDA_GRAPH_BACKENDS}"
        )
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--served-model-name",
        served_model_name,
        "--trust-remote-code",
        "--host",
        host,
        "--port",
        str(port),
        "--tp-size",
        str(tp_size),
        "--context-length",
        "32768",
        "--mem-fraction-static",
        "0.9",
        "--skip-server-warmup",
        "--cuda-graph-backend-decode",
        decode_cuda_graph_backend,
        "--cuda-graph-backend-prefill",
        "disabled",
    ]
    if attention_backend != "auto":
        command.extend(["--attention-backend", attention_backend])
    if all_reduce_mode == "nccl":
        command.append("--disable-custom-all-reduce")
    command.extend(SETUPS[setup_name])
    return command


def start_timed_server(
    command: list[str], log_path: Path, all_reduce_mode: str = "legacy_v1"
) -> tuple[subprocess.Popen[bytes], Any, int]:
    log_file = log_path.open("wb")
    kwargs: dict[str, Any] = {
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        "env": build_server_environment(all_reduce_mode),
    }
    if os.name == "posix":
        kwargs["start_new_session"] = True
    elif os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    start_ns = time.perf_counter_ns()
    try:
        process = subprocess.Popen(command, **kwargs)
    except Exception:
        log_file.close()
        raise
    return process, log_file, start_ns


def stop_server(process: subprocess.Popen[bytes] | None, log_file: Any) -> None:
    if process is not None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.terminate()

        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait(timeout=10)

    if log_file is not None:
        log_file.close()


def wait_for_ready(
    process: subprocess.Popen[bytes],
    session: requests.Session,
    url: str,
    served_model_name: str,
    timeout_s: float,
    poll_interval_s: float,
    log_path: Path,
) -> tuple[int, dict[str, Any]]:
    deadline = time.perf_counter() + timeout_s
    last_error = ""
    while time.perf_counter() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Server exited with code {return_code} before becoming ready.\n"
                f"{read_log_tail(log_path)}"
            )
        try:
            response = session.get(f"{url}/v1/models", timeout=1.0)
            if response.status_code == 200:
                payload = response.json()
                model_ids = [item.get("id") for item in payload.get("data", [])]
                if served_model_name in model_ids:
                    return time.perf_counter_ns(), payload
                last_error = f"ready endpoint returned model ids {model_ids!r}"
            else:
                last_error = f"ready endpoint returned HTTP {response.status_code}"
        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)
        time.sleep(poll_interval_s)
    raise TimeoutError(
        f"Timed out after {timeout_s}s waiting for {url}/v1/models: {last_error}\n"
        f"{read_log_tail(log_path)}"
    )


def validate_model_card(
    payload: dict[str, Any], served_model_name: str
) -> dict[str, Any]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"Invalid /v1/models payload: {payload!r}")
    matches = [item for item in data if item.get("id") == served_model_name]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one /v1/models entry for {served_model_name!r}, "
            f"got {matches!r}"
        )
    model_entry = matches[0]
    if model_entry.get("max_model_len") != EXPECTED_MAX_MODEL_LEN:
        raise RuntimeError(
            f"Expected max_model_len={EXPECTED_MAX_MODEL_LEN}, got "
            f"{model_entry.get('max_model_len')!r}"
        )
    return model_entry


def validate_server_log(
    log_path: Path,
    model_path: Path,
    tp_size: int,
    setup_name: str,
    all_reduce_mode: str = "legacy_v1",
    attention_backend: str = "flashinfer",
    decode_cuda_graph_backend: str = "full",
) -> dict[str, Any]:
    if all_reduce_mode not in ALL_REDUCE_MODES:
        raise ValueError(
            f"Unsupported all-reduce mode {all_reduce_mode!r}; "
            f"expected one of {ALL_REDUCE_MODES}"
        )
    if attention_backend not in ATTENTION_BACKEND_MODES:
        raise ValueError(
            f"Unsupported attention backend {attention_backend!r}; "
            f"expected one of {ATTENTION_BACKEND_MODES}"
        )
    if decode_cuda_graph_backend not in DECODE_CUDA_GRAPH_BACKENDS:
        raise ValueError(
            f"Unsupported decode CUDA graph backend {decode_cuda_graph_backend!r}; "
            f"expected one of {DECODE_CUDA_GRAPH_BACKENDS}"
        )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    resolved_attention_match = re.search(
        r"attention_backend='([^']+)'", log_text
    )
    if resolved_attention_match is None:
        raise RuntimeError(
            f"Could not resolve attention_backend from {log_path}\n"
            f"{read_log_tail(log_path)}"
        )
    resolved_attention_backend = resolved_attention_match.group(1)
    if (
        attention_backend != "auto"
        and resolved_attention_backend != attention_backend
    ):
        raise RuntimeError(
            f"Requested attention backend {attention_backend!r}, but the server "
            f"resolved {resolved_attention_backend!r}.\n{read_log_tail(log_path)}"
        )
    required_markers = [
        f"model_path='{model_path}'",
        f"tp_size={tp_size}",
        f"attention_backend='{resolved_attention_backend}'",
        f"cuda_graph_backend_decode='{decode_cuda_graph_backend}'",
        "cuda_graph_backend_prefill='disabled'",
        f"type={EXPECTED_ARCHITECTURE}",
        "Disable prefill CUDA graph because",
    ]
    if decode_cuda_graph_backend == "full":
        required_markers.append("Capture target decode CUDA graph end.")
    if all_reduce_mode in {"legacy_v1", "custom_v2"}:
        required_markers.append("disable_custom_all_reduce=False")
        if tp_size > 1:
            if all_reduce_mode == "custom_v2":
                required_markers.append(CUSTOM_V2_INIT_MARKER)
            elif decode_cuda_graph_backend == "full":
                required_markers.append(" cuda graph addresses")
    else:
        required_markers.append("disable_custom_all_reduce=True")
    if setup_name == "optimized":
        required_markers.extend(
            [
                "chunked_prefill_size=512",
                "max_running_requests=1",
                "cuda_graph_max_bs_decode=1",
            ]
        )

    forbidden_markers = [
        "Setup Custom allreduce failed",
        "Capture cuda graph failed",
    ]
    if all_reduce_mode == "legacy_v1":
        forbidden_markers.extend(
            [
                "All-reduce call path: NCCL (custom AR disabled)",
                "sgl_kernel_jit_cuda_ipc",
                "custom_all_reduce_v2",
                CUSTOM_V2_INIT_MARKER,
                CUSTOM_V2_VMM_GRAPH_MARKER,
            ]
        )
    elif all_reduce_mode == "custom_v2":
        forbidden_markers.extend(
            [
                "All-reduce call path: NCCL (custom AR disabled)",
                "CustomAllReduceV2 is disabled",
            ]
        )
    else:
        forbidden_markers.extend(
            [CUSTOM_V2_INIT_MARKER, CUSTOM_V2_VMM_GRAPH_MARKER]
        )
    if decode_cuda_graph_backend == "disabled":
        forbidden_markers.append("Capture target decode CUDA graph")
    missing = [marker for marker in required_markers if marker not in log_text]
    present_forbidden = [
        marker for marker in forbidden_markers if marker in log_text
    ]
    legacy_graph_registration_present = bool(
        LEGACY_CUSTOM_AR_GRAPH_PATTERN.search(log_text)
    )
    custom_v2_initialized = CUSTOM_V2_INIT_MARKER in log_text
    custom_v2_vmm_graph_registration_present = (
        CUSTOM_V2_VMM_GRAPH_MARKER in log_text
    )
    if (
        all_reduce_mode in {"custom_v2", "nccl"}
        and legacy_graph_registration_present
    ):
        present_forbidden.append("legacy custom all-reduce graph registration")
    if missing or present_forbidden:
        raise RuntimeError(
            f"Server log validation failed for TP={tp_size}, setup={setup_name}; "
            f"missing={missing}, forbidden={present_forbidden}.\n"
            f"{read_log_tail(log_path)}"
        )
    return {
        "required_markers": required_markers,
        "forbidden_markers_absent": forbidden_markers,
        "all_reduce_mode": all_reduce_mode,
        "requested_attention_backend": attention_backend,
        "resolved_attention_backend": resolved_attention_backend,
        "attention_backend": resolved_attention_backend,
        "decode_cuda_graph_backend": decode_cuda_graph_backend,
        "legacy_graph_registration_present": legacy_graph_registration_present,
        "custom_v2_initialized": custom_v2_initialized,
        "custom_v2_vmm_graph_registration_present": (
            custom_v2_vmm_graph_registration_present
        ),
        "prefill_cuda_graph_backend": "disabled",
    }


def write_summary(
    output_path: Path,
    aggregates: dict[tuple[str, int], float],
    tp_sizes: list[int],
) -> None:
    ordered_tp_sizes = [tp for tp in SUMMARY_TP_ORDER if tp in tp_sizes]
    optimized = {
        tp: aggregates[("optimized", tp)] for tp in ordered_tp_sizes
    }
    non_optimized = {
        tp: aggregates[("non_optimized", tp)] for tp in ordered_tp_sizes
    }
    content = (
        "hunyuan_reprompt_init_time_ms = "
        + pprint.pformat(optimized, sort_dicts=False, width=100)
        + "\n\n"
        + "hunyuan_reprompt_init_time_non_optimized_ms = "
        + pprint.pformat(non_optimized, sort_dicts=False, width=100)
        + "\n"
    )
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    os.replace(temporary_path, output_path)


def main() -> None:
    args = parse_args()
    args.model_path = resolve_model_path(args.model_path)
    architecture, model_type = load_model_identity(args.model_path)
    prepare_output_directory(args.output_dir)
    details_path = args.output_dir / "details.json"
    summary_path = args.output_dir / "summary.py"
    details: dict[str, Any] = {
        "status": "running",
        "model_path": str(args.model_path),
        "served_model_name": args.served_model_name,
        "architecture": architecture,
        "model_type": model_type,
        "tp_sizes": args.tp_sizes,
        "setups": {name: list(extra_args) for name, extra_args in SETUPS.items()},
        "num_runs": NUM_RUNS,
        "num_warmup_runs": NUM_WARMUP_RUNS,
        "server_environment": {
            "forced": forced_server_environment(args.all_reduce_mode),
            "removed": list(REMOVED_SERVER_ENVIRONMENT),
        },
        "attention_backend": args.attention_backend,
        "all_reduce_mode": args.all_reduce_mode,
        "decode_cuda_graph_backend": args.decode_cuda_graph_backend,
        "startup_time_semantics": (
            "time.perf_counter_ns immediately before Popen until /v1/models "
            "first returns HTTP 200 with the expected served model id. Log and "
            "configuration validation happens after the ready timestamp."
        ),
        "runs": [],
        "aggregates": [],
    }
    save_json(details_path, details)
    aggregates: dict[tuple[str, int], float] = {}

    try:
        for tp_size in args.tp_sizes:
            for setup_name in SETUPS:
                measured_times: list[float] = []
                for run_index in range(NUM_RUNS):
                    is_warmup = run_index < NUM_WARMUP_RUNS
                    port = find_free_port(args.host)
                    url = base_url(args.host, port)
                    log_path = args.output_dir / (
                        f"server_tp{tp_size}_{setup_name}_run{run_index + 1:02d}.log"
                    )
                    command = build_server_command(
                        args.model_path,
                        args.served_model_name,
                        args.host,
                        port,
                        tp_size,
                        setup_name,
                        all_reduce_mode=args.all_reduce_mode,
                        attention_backend=args.attention_backend,
                        decode_cuda_graph_backend=args.decode_cuda_graph_backend,
                    )
                    record: dict[str, Any] = {
                        "tp_size": tp_size,
                        "setup": setup_name,
                        "run": run_index + 1,
                        "warmup": is_warmup,
                        "status": "running",
                        "url": url,
                        "command": command,
                        "log_path": str(log_path),
                    }
                    details["runs"].append(record)
                    save_json(details_path, details)
                    print(
                        f"[startup] TP={tp_size} setup={setup_name} "
                        f"run={run_index + 1}/{NUM_RUNS} "
                        f"{'warmup' if is_warmup else 'measure'}",
                        flush=True,
                    )

                    process: subprocess.Popen[bytes] | None = None
                    log_file = None
                    session = requests.Session()
                    session.trust_env = False
                    try:
                        process, log_file, start_ns = start_timed_server(
                            command, log_path, args.all_reduce_mode
                        )
                        ready_ns, model_card = wait_for_ready(
                            process,
                            session,
                            url,
                            args.served_model_name,
                            args.startup_timeout_s,
                            args.ready_poll_interval_s,
                            log_path,
                        )
                        startup_time_ms = (ready_ns - start_ns) / 1_000_000.0
                        model_entry = validate_model_card(
                            model_card, args.served_model_name
                        )
                        validation = validate_server_log(
                            log_path,
                            args.model_path,
                            tp_size,
                            setup_name,
                            all_reduce_mode=args.all_reduce_mode,
                            attention_backend=args.attention_backend,
                            decode_cuda_graph_backend=args.decode_cuda_graph_backend,
                        )
                        record.update(
                            status="completed",
                            startup_time_ms=startup_time_ms,
                            model_card=model_card,
                            validated_model_entry=model_entry,
                            validation=validation,
                        )
                        if not is_warmup:
                            measured_times.append(startup_time_ms)
                    except Exception as exc:
                        record.update(status="failed", error=str(exc))
                        raise
                    finally:
                        session.close()
                        stop_server(process, log_file)
                        wait_for_port_release(args.host, port)
                        save_json(details_path, details)
                        if args.cooldown_s:
                            time.sleep(args.cooldown_s)

                if len(measured_times) != NUM_RUNS - NUM_WARMUP_RUNS:
                    raise RuntimeError("Measured launch count is incomplete")
                startup_time_ms = round(fmean(measured_times), 3)
                aggregates[(setup_name, tp_size)] = startup_time_ms
                details["aggregates"].append(
                    {
                        "tp_size": tp_size,
                        "setup": setup_name,
                        "startup_time_ms": startup_time_ms,
                    }
                )
                save_json(details_path, details)

        write_summary(summary_path, aggregates, args.tp_sizes)
        details["status"] = "completed"
        details["summary_path"] = str(summary_path)
        save_json(details_path, details)
        print(f"[done] summary: {summary_path}", flush=True)
        print(f"[done] details: {details_path}", flush=True)
    except BaseException as exc:
        details["status"] = "failed"
        details["error"] = str(exc)
        save_json(details_path, details)
        raise


if __name__ == "__main__":
    main()
