#!/usr/bin/env python3
"""Profile HunyuanImage-2.1 reprompt TTFT and TPOT with lightweight timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pprint
import random
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
NUM_RUNS = 5
NUM_WARMUP_RUNS = 2
EXPECTED_MAX_MODEL_LEN = 32768
FORCED_SERVER_ENVIRONMENT = {"SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": "0"}
REMOVED_SERVER_ENVIRONMENT = ("SGLANG_USE_JIT_ALL_REDUCE",)
ALL_REDUCE_MODES = ("legacy_v1", "nccl")
ATTENTION_BACKEND_MODES = ("auto", "flashinfer")
DECODE_CUDA_GRAPH_BACKENDS = ("full", "disabled")
PROMPT_TOKENS_PATTERN = re.compile(rb'"prompt_tokens"\s*:\s*(\d+)')
COMPLETION_TOKENS_PATTERN = re.compile(rb'"completion_tokens"\s*:\s*(\d+)')
DECODE_CUDA_GRAPH_STATE_PATTERN = re.compile(
    r"Decode batch[^\r\n]*cuda graph: (True|False)"
)
IO_MATRIX = {
    128: (512, 2048),
    256: (256, 1920),
    384: (128, 1792),
    512: (1664,),
    640: (1536,),
    768: (1408,),
    896: (1280,),
    1024: (1152,),
    1152: (1024,),
    1280: (896,),
    1408: (768,),
    1536: (640,),
    1664: (512,),
    1792: (384,),
    1920: (256,),
    2048: (128,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile HunyuanImage-2.1 reprompt TTFT/TPOT for the registry "
            "input/output matrix. Each point runs two warmups and three "
            "measured requests."
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
        default=Path("/workspace/outputs/hunyuan_reprompt_latency"),
    )
    parser.add_argument("--server-timeout-s", type=float, default=1800.0)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.model_path = args.model_path.expanduser()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.tp_sizes = list(dict.fromkeys(args.tp_sizes))
    invalid_tp_sizes = [tp for tp in args.tp_sizes if tp not in DEFAULT_TP_SIZES]
    if invalid_tp_sizes:
        parser.error(f"--tp-sizes only supports 1, 2, 4, 8; got {invalid_tp_sizes}")
    if args.server_timeout_s <= 0 or args.request_timeout_s <= 0:
        parser.error("timeouts must be positive")
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


def load_model_config(model_path: Path) -> tuple[int, str, str]:
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
    vocab_size = config.get("vocab_size")
    if not isinstance(vocab_size, int) or vocab_size <= 1:
        raise ValueError(f"Invalid vocab_size in {config_path}: {vocab_size!r}")
    return vocab_size, EXPECTED_ARCHITECTURE, EXPECTED_MODEL_TYPE


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


def build_server_environment(all_reduce_mode: str = "legacy_v1") -> dict[str, str]:
    if all_reduce_mode not in ALL_REDUCE_MODES:
        raise ValueError(
            f"Unsupported all-reduce mode {all_reduce_mode!r}; "
            f"expected one of {ALL_REDUCE_MODES}"
        )
    environment = os.environ.copy()
    for name in REMOVED_SERVER_ENVIRONMENT:
        environment.pop(name, None)
    if all_reduce_mode == "legacy_v1":
        environment.update(FORCED_SERVER_ENVIRONMENT)
    else:
        for name in FORCED_SERVER_ENVIRONMENT:
            environment.pop(name, None)
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


def build_server_command(
    model_path: Path,
    served_model_name: str,
    host: str,
    port: int,
    tp_size: int,
    incremental_streaming_output: bool = True,
    all_reduce_mode: str = "legacy_v1",
    attention_backend: str = "flashinfer",
    decode_cuda_graph_backend: str = "full",
    server_random_seed: int | None = None,
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
    if server_random_seed is not None:
        command.extend(["--random-seed", str(server_random_seed)])
    if incremental_streaming_output:
        command.append("--incremental-streaming-output")
    if all_reduce_mode == "nccl":
        command.append("--disable-custom-all-reduce")
    return command


def start_server(
    command: list[str], log_path: Path, all_reduce_mode: str = "legacy_v1"
) -> tuple[subprocess.Popen[bytes], Any]:
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
    try:
        process = subprocess.Popen(command, **kwargs)
    except Exception:
        log_file.close()
        raise
    return process, log_file


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
    url: str,
    served_model_name: str,
    timeout_s: float,
    log_path: Path,
) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_s
    last_error = ""
    with requests.Session() as session:
        session.trust_env = False
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
                        return payload
                    last_error = f"ready endpoint returned model ids {model_ids!r}"
                else:
                    last_error = (
                        f"ready endpoint returned HTTP {response.status_code}"
                    )
            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
            time.sleep(0.1)
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
    require_decode_execution: bool = False,
    incremental_streaming_output: bool = True,
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
        f"incremental_streaming_output={incremental_streaming_output}",
        f"attention_backend='{resolved_attention_backend}'",
        f"cuda_graph_backend_decode='{decode_cuda_graph_backend}'",
        "cuda_graph_backend_prefill='disabled'",
        f"type={EXPECTED_ARCHITECTURE}",
        "Disable prefill CUDA graph because",
    ]
    if decode_cuda_graph_backend == "full":
        required_markers.append("Capture target decode CUDA graph end.")
    if all_reduce_mode == "legacy_v1":
        required_markers.append("disable_custom_all_reduce=False")
        if tp_size > 1 and decode_cuda_graph_backend == "full":
            required_markers.append(" cuda graph addresses")
    else:
        required_markers.append("disable_custom_all_reduce=True")
    forbidden_markers = [
        "Setup Custom allreduce failed",
        "sgl_kernel_jit_cuda_ipc",
        "custom_all_reduce_v2",
        "Capture cuda graph failed",
    ]
    if all_reduce_mode == "legacy_v1":
        forbidden_markers.append("All-reduce call path: NCCL (custom AR disabled)")
    else:
        forbidden_markers.append(" cuda graph addresses")
    if decode_cuda_graph_backend == "disabled":
        forbidden_markers.append("Capture target decode CUDA graph")
    missing = [marker for marker in required_markers if marker not in log_text]
    present_forbidden = [
        marker for marker in forbidden_markers if marker in log_text
    ]
    if missing or present_forbidden:
        raise RuntimeError(
            f"Server log validation failed for TP={tp_size}; "
            f"missing={missing}, forbidden={present_forbidden}.\n"
            f"{read_log_tail(log_path)}"
        )
    decode_cuda_graph_states: list[bool] = []
    if require_decode_execution:
        decode_cuda_graph_states = [
            match.group(1) == "True"
            for match in DECODE_CUDA_GRAPH_STATE_PATTERN.finditer(log_text)
        ]
        expected_decode_cuda_graph = decode_cuda_graph_backend == "full"
        if not decode_cuda_graph_states or any(
            state != expected_decode_cuda_graph
            for state in decode_cuda_graph_states
        ):
            raise RuntimeError(
                f"Decode CUDA graph validation failed for TP={tp_size}; "
                f"expected={expected_decode_cuda_graph}, "
                f"observed={decode_cuda_graph_states}. Only Decode batch log "
                "lines are checked; eager prefill is expected because prefill "
                "CUDA graph is disabled.\n"
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
        "decode_cuda_graph_state_count": len(decode_cuda_graph_states),
        "prefill_cuda_graph_backend": "disabled",
    }


def random_input_ids(
    vocab_size: int, input_length: int, seed: int, first_token_id: int
) -> list[int]:
    if input_length <= 0:
        raise ValueError(f"input_length must be positive, got {input_length}")
    if not 0 <= first_token_id < vocab_size:
        raise ValueError(
            f"first_token_id must be in [0, {vocab_size}), got {first_token_id}"
        )
    generator = random.Random(seed)
    input_ids = [generator.randrange(vocab_size) for _ in range(input_length)]
    input_ids[0] = first_token_id
    return input_ids


def measure_request(
    session: requests.Session,
    url: str,
    input_ids: list[int],
    output_length: int,
    timeout_s: float,
    sampling_seed: int | None = None,
) -> dict[str, float | int]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_length,
            "ignore_eos": True,
        },
        "stream": True,
    }
    if sampling_seed is not None:
        payload["sampling_params"]["sampling_seed"] = sampling_seed
    first_token_ns: int | None = None
    last_token_ns: int | None = None
    prompt_tokens: int | None = None
    completion_tokens = 0
    sse_event_count = 0
    sse_bytes = 0
    token_update_event_count = 0
    start_ns = time.perf_counter_ns()

    with session.post(
        f"{url}/generate",
        json=payload,
        stream=True,
        timeout=(10.0, timeout_s),
    ) as response:
        if response.status_code != 200:
            raise RuntimeError(
                f"/generate returned HTTP {response.status_code}: {response.text}"
            )
        response.raw.decode_content = True
        while True:
            raw_line = response.raw.readline()
            line_received_ns = time.perf_counter_ns()
            if not raw_line:
                break
            sse_bytes += len(raw_line)
            line = raw_line.strip()
            if not line or not line.startswith(b"data:"):
                continue
            sse_event_count += 1
            body = line[len(b"data:") :].strip()
            if body == b"[DONE]":
                break
            prompt_match = PROMPT_TOKENS_PATTERN.search(body)
            if prompt_match is not None:
                event_prompt_tokens = int(prompt_match.group(1))
                if prompt_tokens is None:
                    prompt_tokens = event_prompt_tokens
                elif event_prompt_tokens != prompt_tokens:
                    raise RuntimeError(
                        "prompt_tokens changed during the streaming response"
                    )
            completion_match = COMPLETION_TOKENS_PATTERN.search(body)
            if completion_match is None:
                continue
            event_completion_tokens = int(completion_match.group(1))
            if event_completion_tokens < completion_tokens:
                raise RuntimeError(
                    "completion_tokens decreased during the streaming response"
                )
            if event_completion_tokens == completion_tokens:
                continue
            completion_tokens = event_completion_tokens
            token_update_event_count += 1
            if first_token_ns is None:
                first_token_ns = line_received_ns
            last_token_ns = line_received_ns

    expected_prompt_tokens = len(input_ids)
    if prompt_tokens != expected_prompt_tokens:
        raise RuntimeError(
            f"Expected {expected_prompt_tokens} input tokens, received "
            f"{prompt_tokens!r}"
        )
    if completion_tokens != output_length:
        raise RuntimeError(
            f"Expected {output_length} output tokens, received {completion_tokens}"
        )
    if first_token_ns is None or last_token_ns is None:
        raise RuntimeError("The streaming response contained no token event")

    ttft_ms = (first_token_ns - start_ns) / 1_000_000.0
    e2e_ms = (last_token_ns - start_ns) / 1_000_000.0
    tpot_ms = (last_token_ns - first_token_ns) / (output_length - 1) / 1_000_000.0
    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": e2e_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "sse_event_count": sse_event_count,
        "sse_bytes": sse_bytes,
        "token_update_event_count": token_update_event_count,
    }


def write_summary(
    output_path: Path,
    aggregates: dict[tuple[int, int, int], dict[str, float]],
    tp_sizes: list[int],
) -> None:
    ordered_tp_sizes = [tp for tp in DEFAULT_TP_SIZES if tp in tp_sizes]
    ttft = {
        input_length: {
            output_length: {
                tp: aggregates[(input_length, output_length, tp)]["ttft_ms"]
                for tp in ordered_tp_sizes
            }
            for output_length in output_lengths
        }
        for input_length, output_lengths in IO_MATRIX.items()
    }
    tpot = {
        input_length: {
            output_length: {
                tp: aggregates[(input_length, output_length, tp)]["tpot_ms"]
                for tp in ordered_tp_sizes
            }
            for output_length in output_lengths
        }
        for input_length, output_lengths in IO_MATRIX.items()
    }
    content = (
        "hunyuan_reprompt_ttft_ms = "
        + pprint.pformat(ttft, sort_dicts=False, width=100)
        + "\n\n"
        + "hunyuan_reprompt_tpot_ms = "
        + pprint.pformat(tpot, sort_dicts=False, width=100)
        + "\n"
    )
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    os.replace(temporary_path, output_path)


def main() -> None:
    args = parse_args()
    args.model_path = resolve_model_path(args.model_path)
    vocab_size, architecture, model_type = load_model_config(args.model_path)
    requests_per_tp = sum(len(lengths) for lengths in IO_MATRIX.values()) * NUM_RUNS
    if requests_per_tp > vocab_size:
        raise ValueError(
            f"Need {requests_per_tp} unique first tokens per TP, but vocab_size is "
            f"only {vocab_size}"
        )
    prepare_output_directory(args.output_dir)
    details_path = args.output_dir / "details.json"
    summary_path = args.output_dir / "summary.py"
    details: dict[str, Any] = {
        "status": "running",
        "model_path": str(args.model_path),
        "served_model_name": args.served_model_name,
        "architecture": architecture,
        "model_type": model_type,
        "vocab_size": vocab_size,
        "tp_sizes": args.tp_sizes,
        "io_matrix": {
            str(input_length): list(output_lengths)
            for input_length, output_lengths in IO_MATRIX.items()
        },
        "num_runs": NUM_RUNS,
        "num_warmup_runs": NUM_WARMUP_RUNS,
        "server_environment": {
            "forced": FORCED_SERVER_ENVIRONMENT,
            "removed": list(REMOVED_SERVER_ENVIRONMENT),
        },
        "timing_semantics": (
            "TTFT is request start to receipt of the first raw SSE token line; "
            "TPOT is (last token line receipt - first token line receipt) / "
            "(OSL - 1). Token counts are extracted directly from raw SSE bytes "
            "without materializing the response JSON."
        ),
        "radix_cache_control": (
            "Every request within a TP run has a distinct deterministic first "
            "input token, preventing shared-prefix Radix Cache hits."
        ),
        "runs": [],
        "aggregates": [],
    }
    save_json(details_path, details)
    aggregates: dict[tuple[int, int, int], dict[str, float]] = {}

    try:
        for tp_size in args.tp_sizes:
            port = find_free_port(args.host)
            url = base_url(args.host, port)
            log_path = args.output_dir / f"server_tp{tp_size}.log"
            command = build_server_command(
                args.model_path,
                args.served_model_name,
                args.host,
                port,
                tp_size,
            )
            process: subprocess.Popen[bytes] | None = None
            log_file = None
            request_session: requests.Session | None = None
            try:
                print(f"[server] launching TP={tp_size} on {url}", flush=True)
                process, log_file = start_server(command, log_path)
                model_card = wait_for_ready(
                    process,
                    url,
                    args.served_model_name,
                    args.server_timeout_s,
                    log_path,
                )
                model_entry = validate_model_card(
                    model_card, args.served_model_name
                )
                startup_validation = validate_server_log(
                    log_path, args.model_path, tp_size
                )
                server_record = {
                    "tp_size": tp_size,
                    "url": url,
                    "command": command,
                    "log_path": str(log_path),
                    "model_card": model_card,
                    "validated_model_entry": model_entry,
                    "startup_validation": startup_validation,
                }
                details.setdefault("servers", []).append(server_record)
                save_json(details_path, details)
                request_session = requests.Session()
                request_session.trust_env = False
                request_ordinal = 0

                for case_index, (input_length, output_lengths) in enumerate(
                    IO_MATRIX.items()
                ):
                    for output_length in output_lengths:
                        measured_ttft: list[float] = []
                        measured_tpot: list[float] = []
                        for run_index in range(NUM_RUNS):
                            is_warmup = run_index < NUM_WARMUP_RUNS
                            seed = (
                                args.seed
                                + tp_size * 1_000_000
                                + case_index * 10_000
                                + output_length * 10
                                + run_index
                            )
                            first_token_id = (
                                args.seed + tp_size * 1_000 + request_ordinal
                            ) % vocab_size
                            input_ids = random_input_ids(
                                vocab_size,
                                input_length,
                                seed,
                                first_token_id,
                            )
                            input_ids_sha256 = hashlib.sha256(
                                ",".join(str(token_id) for token_id in input_ids).encode(
                                    "ascii"
                                )
                            ).hexdigest()
                            request_ordinal += 1
                            record: dict[str, Any] = {
                                "tp_size": tp_size,
                                "input_length": input_length,
                                "output_length": output_length,
                                "run": run_index + 1,
                                "warmup": is_warmup,
                                "seed": seed,
                                "first_token_id": first_token_id,
                                "input_ids_sha256": input_ids_sha256,
                                "status": "running",
                            }
                            details["runs"].append(record)
                            save_json(details_path, details)
                            print(
                                f"[request] TP={tp_size} ISL={input_length} "
                                f"OSL={output_length} run={run_index + 1}/{NUM_RUNS} "
                                f"{'warmup' if is_warmup else 'measure'}",
                                flush=True,
                            )
                            try:
                                metrics = measure_request(
                                    request_session,
                                    url,
                                    input_ids,
                                    output_length,
                                    args.request_timeout_s,
                                )
                            except Exception as exc:
                                record.update(status="failed", error=str(exc))
                                save_json(details_path, details)
                                raise
                            record.update(
                                status="completed",
                                **metrics,
                            )
                            save_json(details_path, details)
                            if not is_warmup:
                                measured_ttft.append(float(metrics["ttft_ms"]))
                                measured_tpot.append(float(metrics["tpot_ms"]))

                        if len(measured_ttft) != NUM_RUNS - NUM_WARMUP_RUNS:
                            raise RuntimeError("Measured run count is incomplete")
                        aggregate = {
                            "ttft_ms": round(fmean(measured_ttft), 3),
                            "tpot_ms": round(fmean(measured_tpot), 3),
                        }
                        aggregates[(input_length, output_length, tp_size)] = aggregate
                        details["aggregates"].append(
                            {
                                "tp_size": tp_size,
                                "input_length": input_length,
                                "output_length": output_length,
                                **aggregate,
                            }
                        )
                        save_json(details_path, details)
                server_record["runtime_validation"] = validate_server_log(
                    log_path,
                    args.model_path,
                    tp_size,
                    require_decode_execution=True,
                )
                save_json(details_path, details)
            finally:
                if request_session is not None:
                    request_session.close()
                stop_server(process, log_file)

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
