#!/usr/bin/env bash
set -euo pipefail

# Launch sglang serve sequentially for multiple TP sizes, wait until /v1/models
# becomes ready, then stop the server and export one JSON file per run that
# records the init-time values SGLang computed/logged.

MODEL_PATH="${MODEL_PATH:-/workspace/models/Hunyuan_PromptEnhancer_32B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
CTX_LEN="${CTX_LEN:-32768}"
TP_SIZE="${TP_SIZE:-2 4}"
READY_CHECK_TIMEOUT="${READY_CHECK_TIMEOUT:-600}"
SERVER_BOOTSTRAP_GRACE_SEC="${SERVER_BOOTSTRAP_GRACE_SEC:-5}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/workspace/outputs/pe32b/init/${RUN_ID}}"

# Leave these empty by default so we can observe SGLang's own init-time search
# unless the caller explicitly overrides them.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}"

SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

print_server_log_tail() {
  local log_path="$1"

  if [[ -n "${log_path}" ]] && [[ -f "${log_path}" ]]; then
    echo "Last 200 lines of ${log_path}:" >&2
    tail -n 200 "${log_path}" >&2 || true
  fi
}

ensure_server_bootstrapped() {
  local grace_s="$1"
  local pid="${2:-}"
  local log_path="${3:-}"

  for ((sec = 1; sec <= grace_s; sec++)); do
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "Server process exited during bootstrap grace window." >&2
      print_server_log_tail "${log_path}"
      return 1
    fi
    sleep 1
  done

  if [[ -n "${pid}" ]] && ! kill -0 "${pid}" >/dev/null 2>&1; then
    echo "Server process exited before ready check." >&2
    print_server_log_tail "${log_path}"
    return 1
  fi

  return 0
}

wait_for_models_ready() {
  local base_url="$1"
  local timeout_s="$2"
  local pid="${3:-}"
  local log_path="${4:-}"
  local start_ts
  start_ts="$(date +%s)"

  while true; do
    if python - "$base_url" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

base_url = sys.argv[1].rstrip("/")
with urllib.request.urlopen(f"{base_url}/v1/models", timeout=2) as resp:
    if 200 <= resp.status < 300:
        raise SystemExit(0)
raise SystemExit(1)
PY
    then
      return 0
    fi

    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "Server process exited before /v1/models became ready." >&2
      print_server_log_tail "${log_path}"
      return 1
    fi

    if (( "$(date +%s)" - start_ts >= timeout_s )); then
      echo "Timed out waiting for ${base_url}/v1/models after ${timeout_s}s." >&2
      print_server_log_tail "${log_path}"
      return 1
    fi

    sleep 1
  done
}

write_profile_json() {
  local log_path="$1"
  local json_path="$2"
  local tp="$3"

  python - "$log_path" "$json_path" "$MODEL_PATH" "$HOST" "$PORT" "$tp" "$CTX_LEN" "$MEM_FRACTION_STATIC" "$CHUNKED_PREFILL_SIZE" "$CUDA_GRAPH_MAX_BS" "$MAX_PREFILL_TOKENS" "$MAX_TOTAL_TOKENS" "$MAX_RUNNING_REQUESTS" <<'PY'
import ast
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


def parse_optional_cli(value):
    value = value.strip()
    if value == "":
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def parse_server_arg(line, name):
    match = re.search(rf"{re.escape(name)}=(.*?)(?:, [A-Za-z_][A-Za-z0-9_]*=|\)$)", line)
    if not match:
        return None

    raw = match.group(1).strip()
    try:
        return ast.literal_eval(raw)
    except Exception:
        if raw == "None":
            return None
        if raw == "True":
            return True
        if raw == "False":
            return False
        return raw


def parse_float(pattern, text):
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1))


def parse_int(pattern, text):
    match = re.search(pattern, text)
    if not match:
        return None
    return int(match.group(1))


def mib_to_gb(value):
    if value is None:
        return None
    return round(value / 1024.0, 3)


def size_to_gb(value, unit):
    if value is None:
        return None
    if unit.upper() == "GB":
        return round(value, 3)
    if unit.upper() == "MB":
        return round(value / 1024.0, 3)
    return None


def last_line_containing(lines, needle):
    for line in reversed(lines):
        if needle in line:
            return line
    return ""


def parse_ranked_prefix(line):
    match = re.match(r"^\[(?P<timestamp>[^\]]+?) (?P<rank>TP\d+)\] (?P<body>.*)$", line)
    if not match:
        return None
    return match.groupdict()


def summarize_numeric_field(events, field):
    values = [event[field] for event in events if event.get(field) is not None]
    if not values:
        return None
    return {
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "avg": round(sum(values) / len(values), 3),
        "sum": round(sum(values), 3),
    }


def summarize_ranked_events(events, numeric_fields):
    summary = {"count": len(events)}
    for field in numeric_fields:
        stats = summarize_numeric_field(events, field)
        if stats is not None:
            summary[field] = stats
    return summary


def collect_ranked_memory_events(lines):
    grouped = {
        "torch_distributed_init": [],
        "load_weight_begin": [],
        "load_weight_end": [],
        "kv_cache_allocation": [],
        "memory_pool_end": [],
        "cuda_graph_begin": [],
        "cuda_graph_end": [],
    }

    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        parsed = parse_ranked_prefix(line)
        if parsed is None:
            continue

        rank = parsed["rank"]
        timestamp = parsed["timestamp"]
        body = parsed["body"]

        common = {
            "rank": rank,
            "timestamp": timestamp,
            "line_number": idx,
            "log_line": line,
        }

        if body.startswith("Init torch distributed ends."):
            grouped["torch_distributed_init"].append(
                {
                    **common,
                    "elapsed_s": parse_float(r"elapsed=([0-9.]+) s", body),
                    "mem_usage_gb": parse_float(r"mem usage=([0-9.]+) GB", body),
                }
            )
            continue

        if body.startswith("Load weight begin."):
            grouped["load_weight_begin"].append(
                {
                    **common,
                    "avail_mem_gb": parse_float(r"avail mem=([0-9.]+) GB", body),
                }
            )
            continue

        if body.startswith("Load weight end."):
            grouped["load_weight_end"].append(
                {
                    **common,
                    "elapsed_s": parse_float(r"elapsed=([0-9.]+) s", body),
                    "avail_mem_gb": parse_float(r"avail mem=([0-9.]+) GB", body),
                    "mem_usage_gb": parse_float(r"mem usage=([0-9.]+) GB", body),
                    "model_type": (
                        re.search(r"type=([^,]+)", body).group(1)
                        if re.search(r"type=([^,]+)", body)
                        else None
                    ),
                }
            )
            continue

        if body.startswith("KV Cache is allocated."):
            k_size_gb = parse_float(r"K size: ([0-9.]+) GB", body)
            v_size_gb = parse_float(r"V size: ([0-9.]+) GB", body)
            kv_size_gb = parse_float(r"KV size: ([0-9.]+) GB", body)
            if kv_size_gb is None and k_size_gb is not None and v_size_gb is not None:
                kv_size_gb = round(k_size_gb + v_size_gb, 3)
            grouped["kv_cache_allocation"].append(
                {
                    **common,
                    "num_tokens": parse_int(r"#tokens: (\d+)", body),
                    "k_size_gb": k_size_gb,
                    "v_size_gb": v_size_gb,
                    "kv_size_gb": kv_size_gb,
                }
            )
            continue

        if body.startswith("Memory pool end."):
            grouped["memory_pool_end"].append(
                {
                    **common,
                    "avail_mem_gb": parse_float(r"avail mem=([0-9.]+) GB", body),
                }
            )
            continue

        if body.startswith("Capture cuda graph begin."):
            grouped["cuda_graph_begin"].append(
                {
                    **common,
                    "avail_mem_gb": parse_float(r"avail mem=([0-9.]+) GB", body),
                }
            )
            continue

        if body.startswith("Capture cuda graph end."):
            grouped["cuda_graph_end"].append(
                {
                    **common,
                    "elapsed_s": parse_float(r"Time elapsed: ([0-9.]+) s", body),
                    "mem_usage_gb": parse_float(r"mem usage=([0-9.]+) GB", body),
                    "avail_mem_gb": parse_float(r"avail mem=([0-9.]+) GB", body),
                }
            )

    aggregates = {
        "torch_distributed_init": summarize_ranked_events(
            grouped["torch_distributed_init"], ["elapsed_s", "mem_usage_gb"]
        ),
        "load_weight_begin": summarize_ranked_events(
            grouped["load_weight_begin"], ["avail_mem_gb"]
        ),
        "load_weight_end": summarize_ranked_events(
            grouped["load_weight_end"], ["elapsed_s", "avail_mem_gb", "mem_usage_gb"]
        ),
        "kv_cache_allocation": summarize_ranked_events(
            grouped["kv_cache_allocation"], ["num_tokens", "k_size_gb", "v_size_gb", "kv_size_gb"]
        ),
        "memory_pool_end": summarize_ranked_events(
            grouped["memory_pool_end"], ["avail_mem_gb"]
        ),
        "cuda_graph_begin": summarize_ranked_events(
            grouped["cuda_graph_begin"], ["avail_mem_gb"]
        ),
        "cuda_graph_end": summarize_ranked_events(
            grouped["cuda_graph_end"], ["elapsed_s", "mem_usage_gb", "avail_mem_gb"]
        ),
    }

    return {"events": grouped, "aggregates": aggregates}


def collect_extra_memory_overheads(lines):
    parsed_items = []

    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        lowered = line.lower()

        if "routing experts device buffer allocated." in lowered:
            size_mb = parse_float(r"size: ([0-9.]+) MB", line)
            parsed_items.append(
                {
                    "name": "routing_experts_device_buffer",
                    "memory_kind": "device",
                    "line_number": idx,
                    "size_mb": size_mb,
                    "size_gb": mib_to_gb(size_mb),
                    "log_line": line,
                }
            )
            continue

        if "routing experts host buffer allocated." in lowered:
            size_gb = parse_float(r"size: ([0-9.]+) GB", line)
            parsed_items.append(
                {
                    "name": "routing_experts_host_buffer",
                    "memory_kind": "host",
                    "line_number": idx,
                    "size_gb": size_gb,
                    "log_line": line,
                }
            )
            continue

        if "mamba cache is allocated." in lowered:
            entry = {
                "name": "mamba_cache",
                "memory_kind": "device",
                "line_number": idx,
                "num_tokens": parse_int(r"#tokens: (\d+)", line),
                "conv_state_gb": parse_float(r"conv_state size: ([0-9.]+)GB", line),
                "ssm_state_gb": parse_float(r"ssm_state size: ([0-9.]+)GB", line),
                "intermediate_ssm_state_cache_gb": parse_float(
                    r"intermediate_ssm_state_cache size: ([0-9.]+)GB", line
                ),
                "intermediate_conv_window_cache_gb": parse_float(
                    r"intermediate_conv_window_cache size: ([0-9.]+)GB", line
                ),
                "log_line": line,
            }
            components = [
                entry["conv_state_gb"],
                entry["ssm_state_gb"],
                entry["intermediate_ssm_state_cache_gb"],
                entry["intermediate_conv_window_cache_gb"],
            ]
            components = [value for value in components if value is not None]
            if components:
                entry["total_gb"] = round(sum(components), 3)
            parsed_items.append(entry)
            continue

        # Generic catch-all for other size-bearing init-time overhead lines that
        # may come from communication/workspace/buffer subsystems. We keep the
        # original line so downstream analysis can interpret new backends.
        if (
            "size:" in lowered
            and "kv cache is allocated." not in lowered
            and "mamba cache is allocated." not in lowered
            and any(
            token in lowered for token in ("buffer", "workspace", "cache")
            )
        ):
            size_match = re.search(r"size: ([0-9.]+) (GB|MB)", line)
            if size_match:
                size_value = float(size_match.group(1))
                size_unit = size_match.group(2)
                memory_kind = "host" if "host" in lowered else "device"
                parsed_items.append(
                    {
                        "name": "generic_extra_overhead",
                        "memory_kind": memory_kind,
                        "line_number": idx,
                        "size_value": size_value,
                        "size_unit": size_unit,
                        "size_gb": size_to_gb(size_value, size_unit),
                        "log_line": line,
                    }
                )

    def _sum_items(kind):
        total = 0.0
        found = False
        for item in parsed_items:
            if item.get("memory_kind") != kind:
                continue
            size_gb = item.get("size_gb")
            if size_gb is None:
                size_gb = item.get("total_gb")
            if size_gb is None:
                continue
            total += size_gb
            found = True
        return round(total, 3) if found else None

    configured_hints = {
        "flashinfer_workspace_env_bytes": int(
            os.environ.get("SGLANG_FLASHINFER_WORKSPACE_SIZE", str(384 * 1024 * 1024))
        ),
        "flashinfer_workspace_env_gb": round(
            int(
                os.environ.get(
                    "SGLANG_FLASHINFER_WORKSPACE_SIZE", str(384 * 1024 * 1024)
                )
            )
            / (1024**3),
            3,
        ),
        "trtllm_mha_default_workspace_gb": round((512 * 1024 * 1024) / (1024**3), 3),
        "trtllm_mla_default_workspace_gb": round((150 * 1024 * 1024) / (1024**3), 3),
    }

    return {
        "parsed_items": parsed_items,
        "device_total_gb": _sum_items("device"),
        "host_total_gb": _sum_items("host"),
        "configured_hints": configured_hints,
    }


def compute_reserved_mem_heuristic_gb(resolved):
    chunked_prefill_size = resolved.get("chunked_prefill_size")
    max_prefill_tokens = resolved.get("max_prefill_tokens")
    cuda_graph_max_bs = resolved.get("cuda_graph_max_bs")
    tp_size = resolved.get("tp_size")
    pp_size = resolved.get("pp_size")
    disable_piecewise_cuda_graph = resolved.get("disable_piecewise_cuda_graph")
    enable_dp_attention = resolved.get("enable_dp_attention")
    dp_size = resolved.get("dp_size")
    speculative_algorithm = resolved.get("speculative_algorithm")

    if chunked_prefill_size is None or max_prefill_tokens is None or cuda_graph_max_bs is None:
        return None
    if tp_size is None or pp_size is None:
        return None

    reserved_mem_mib = 512.0

    if chunked_prefill_size > 0:
        reserved_mem_mib += max(chunked_prefill_size, 2048) * 1.5
    else:
        reserved_mem_mib += max(max_prefill_tokens, 2048) * 1.5

    reserved_mem_mib += cuda_graph_max_bs * 2
    reserved_mem_mib += tp_size * pp_size / 8 * 1024

    if enable_dp_attention and dp_size is not None:
        reserved_mem_mib += cuda_graph_max_bs * dp_size * 3
        if cuda_graph_max_bs > 300:
            reserved_mem_mib += cuda_graph_max_bs * dp_size * 1.5

    if speculative_algorithm is not None:
        if speculative_algorithm == "STANDALONE":
            reserved_mem_mib += 6 * 1024
        elif speculative_algorithm != "NGRAM":
            reserved_mem_mib += 4 * 1024

    if disable_piecewise_cuda_graph is False:
        # We intentionally do not reconstruct piecewise token lists from the log in
        # this script. The current intended mode is piecewise disabled.
        return None

    return round(reserved_mem_mib / 1024.0, 3)


(
    log_path,
    json_path,
    model_path_arg,
    host,
    port,
    tp_arg,
    ctx_len_arg,
    mem_fraction_arg,
    chunked_prefill_arg,
    cuda_graph_max_bs_arg,
    max_prefill_tokens_arg,
    max_total_tokens_arg,
    max_running_requests_arg,
) = sys.argv[1:]

log_path = Path(log_path)
json_path = Path(json_path)
text = log_path.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()
ranked_memory = collect_ranked_memory_events(lines)

server_args_line = last_line_containing(lines, "server_args=ServerArgs(")
scheduler_line = last_line_containing(lines, "max_total_num_tokens=")
load_weight_begin_line = last_line_containing(lines, "Load weight begin.")
load_weight_end_line = last_line_containing(lines, "Load weight end.")
kv_alloc_line = last_line_containing(lines, "KV Cache is allocated.")
memory_pool_end_line = last_line_containing(lines, "Memory pool end.")
cuda_graph_end_line = last_line_containing(lines, "Capture cuda graph end.")
piecewise_cuda_graph_end_line = last_line_containing(lines, "Capture piecewise CUDA graph end.")

server_args = {}
for key in [
    "model_path",
    "served_model_name",
    "context_length",
    "tp_size",
    "pp_size",
    "dp_size",
    "mem_fraction_static",
    "max_total_tokens",
    "max_running_requests",
    "chunked_prefill_size",
    "max_prefill_tokens",
    "cuda_graph_max_bs",
    "disable_cuda_graph",
    "disable_piecewise_cuda_graph",
    "enable_dp_attention",
    "speculative_algorithm",
    "attention_backend",
    "enable_flashinfer_allreduce_fusion",
    "enable_aiter_allreduce_fusion",
    "enable_symm_mem",
    "enable_mscclpp",
    "enable_torch_symm_mem",
    "disable_custom_all_reduce",
]:
    server_args[key] = parse_server_arg(server_args_line, key)

resolved_max_total_num_tokens = parse_int(r"max_total_num_tokens=(\d+)", scheduler_line)
resolved_chunked_prefill_size = parse_int(r"chunked_prefill_size=(\d+)", scheduler_line)
resolved_max_prefill_tokens = parse_int(r"max_prefill_tokens=(\d+)", scheduler_line)
resolved_max_running_requests = parse_int(r"max_running_requests=(\d+)", scheduler_line)
resolved_context_len = parse_int(r"context_len=(\d+)", scheduler_line)
available_gpu_mem_gb = parse_float(r"available_gpu_mem=([0-9.]+) GB", scheduler_line)

load_weight_begin_avail_mem_gb = parse_float(r"avail mem=([0-9.]+) GB", load_weight_begin_line)
load_weight_end_avail_mem_gb = parse_float(r"avail mem=([0-9.]+) GB", load_weight_end_line)
weight_load_mem_usage_gb = parse_float(r"mem usage=([0-9.]+) GB", load_weight_end_line)

kv_cache_tokens = parse_int(r"#tokens: (\d+)", kv_alloc_line)
kv_cache_k_gb = parse_float(r"K size: ([0-9.]+) GB", kv_alloc_line)
kv_cache_v_gb = parse_float(r"V size: ([0-9.]+) GB", kv_alloc_line)
kv_cache_total_gb = parse_float(r"KV size: ([0-9.]+) GB", kv_alloc_line)
if kv_cache_total_gb is None and kv_cache_k_gb is not None and kv_cache_v_gb is not None:
    kv_cache_total_gb = round(kv_cache_k_gb + kv_cache_v_gb, 3)

memory_pool_end_avail_mem_gb = parse_float(r"avail mem=([0-9.]+) GB", memory_pool_end_line)
cuda_graph_mem_usage_gb = parse_float(r"mem usage=([0-9.]+) GB", cuda_graph_end_line)
cuda_graph_end_avail_mem_gb = parse_float(r"avail mem=([0-9.]+) GB", cuda_graph_end_line)
piecewise_cuda_graph_mem_usage_gb = parse_float(r"mem usage=([0-9.]+) GB", piecewise_cuda_graph_end_line)
piecewise_cuda_graph_end_avail_mem_gb = parse_float(r"avail mem=([0-9.]+) GB", piecewise_cuda_graph_end_line)

resolved = {
    "model_path": server_args.get("model_path") or model_path_arg,
    "served_model_name": server_args.get("served_model_name"),
    "context_length": resolved_context_len or server_args.get("context_length") or parse_optional_cli(ctx_len_arg),
    "tp_size": server_args.get("tp_size") or parse_optional_cli(tp_arg),
    "pp_size": server_args.get("pp_size"),
    "dp_size": server_args.get("dp_size"),
    "chunked_prefill_size": resolved_chunked_prefill_size or server_args.get("chunked_prefill_size"),
    "max_prefill_tokens": resolved_max_prefill_tokens or server_args.get("max_prefill_tokens"),
    "max_total_tokens_user_cap": server_args.get("max_total_tokens"),
    "max_total_num_tokens": resolved_max_total_num_tokens,
    "max_running_requests": resolved_max_running_requests or server_args.get("max_running_requests"),
    "mem_fraction_static": server_args.get("mem_fraction_static"),
    "cuda_graph_enabled": (
        False if server_args.get("disable_cuda_graph") is True
        else True if server_args.get("disable_cuda_graph") is False
        else None
    ),
    "cuda_graph_max_bs": server_args.get("cuda_graph_max_bs"),
    "disable_cuda_graph": server_args.get("disable_cuda_graph"),
    "piecewise_cuda_graph_enabled": (
        False if server_args.get("disable_piecewise_cuda_graph") is True
        else True if server_args.get("disable_piecewise_cuda_graph") is False
        else None
    ),
    "disable_piecewise_cuda_graph": server_args.get("disable_piecewise_cuda_graph"),
    "attention_backend": server_args.get("attention_backend"),
    "enable_dp_attention": bool(server_args.get("enable_dp_attention")),
    "speculative_algorithm": server_args.get("speculative_algorithm"),
    "enable_flashinfer_allreduce_fusion": server_args.get(
        "enable_flashinfer_allreduce_fusion"
    ),
    "enable_aiter_allreduce_fusion": server_args.get(
        "enable_aiter_allreduce_fusion"
    ),
    "enable_symm_mem": server_args.get("enable_symm_mem"),
    "enable_mscclpp": server_args.get("enable_mscclpp"),
    "enable_torch_symm_mem": server_args.get("enable_torch_symm_mem"),
    "disable_custom_all_reduce": server_args.get("disable_custom_all_reduce"),
}

requested = {
    "host": host,
    "port": int(port),
    "tp_size": parse_optional_cli(tp_arg),
    "context_length": parse_optional_cli(ctx_len_arg),
    "mem_fraction_static": parse_optional_cli(mem_fraction_arg),
    "chunked_prefill_size": parse_optional_cli(chunked_prefill_arg),
    "cuda_graph_max_bs": parse_optional_cli(cuda_graph_max_bs_arg),
    "max_prefill_tokens": parse_optional_cli(max_prefill_tokens_arg),
    "max_total_tokens": parse_optional_cli(max_total_tokens_arg),
    "max_running_requests": parse_optional_cli(max_running_requests_arg),
    "disable_cuda_graph": False,
    "disable_piecewise_cuda_graph": True,
    "skip_server_warmup": True,
}

memory = {
    "available_gpu_mem_gb_at_scheduler": available_gpu_mem_gb,
    "load_weight_begin_avail_mem_gb": load_weight_begin_avail_mem_gb,
    "load_weight_end_avail_mem_gb": load_weight_end_avail_mem_gb,
    "weight_load_mem_usage_gb": weight_load_mem_usage_gb,
    "kv_cache_tokens": kv_cache_tokens,
    "kv_cache_k_gb": kv_cache_k_gb,
    "kv_cache_v_gb": kv_cache_v_gb,
    "kv_cache_total_gb": kv_cache_total_gb,
    "memory_pool_end_avail_mem_gb": memory_pool_end_avail_mem_gb,
    "cuda_graph_mem_usage_gb": cuda_graph_mem_usage_gb,
    "cuda_graph_end_avail_mem_gb": cuda_graph_end_avail_mem_gb,
    "piecewise_cuda_graph_mem_usage_gb": piecewise_cuda_graph_mem_usage_gb,
    "piecewise_cuda_graph_end_avail_mem_gb": piecewise_cuda_graph_end_avail_mem_gb,
    "token_pool_search_watermark": resolved.get("mem_fraction_static"),
    "reserved_mem_heuristic_gb": compute_reserved_mem_heuristic_gb(resolved),
    "per_rank_events": ranked_memory["events"],
    "aggregates": ranked_memory["aggregates"],
    "extra_overheads": collect_extra_memory_overheads(lines),
}

payload = {
    "schema_version": 2,
    "status": "ready",
    "timestamp": datetime.now().astimezone().isoformat(),
    "requested": requested,
    "resolved": resolved,
    "memory": memory,
    "artifacts": {
        "server_log": str(log_path),
        "ready_endpoint": f"http://{host}:{port}/v1/models",
    },
}

json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

trap cleanup EXIT INT TERM

mkdir -p "${RUN_ROOT}"
read -r -a TP_SIZE_LIST <<< "${TP_SIZE}"

if ! command -v python >/dev/null 2>&1; then
  echo "python not found in PATH." >&2
  exit 1
fi

echo "Run root: ${RUN_ROOT}"

for TP in "${TP_SIZE_LIST[@]}"; do
  TP_DIR="${RUN_ROOT}/tp${TP}"
  SERVER_LOG="${TP_DIR}/server.log"
  PROFILE_JSON="${TP_DIR}/serve_init_profile.json"
  mkdir -p "${TP_DIR}"

  SERVER_ARGS=(
    -m sglang.launch_server
    --model-path "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --trust-remote-code
    --tp-size "${TP}"
    --skip-server-warmup
    --disable-piecewise-cuda-graph
  )

  if [[ -n "${CTX_LEN}" ]]; then
    SERVER_ARGS+=(--context-length "${CTX_LEN}")
  fi

  if [[ -n "${MEM_FRACTION_STATIC}" ]]; then
    SERVER_ARGS+=(--mem-fraction-static "${MEM_FRACTION_STATIC}")
  fi

  if [[ -n "${CHUNKED_PREFILL_SIZE}" ]]; then
    SERVER_ARGS+=(--chunked-prefill-size "${CHUNKED_PREFILL_SIZE}")
  fi

  if [[ -n "${CUDA_GRAPH_MAX_BS}" ]]; then
    SERVER_ARGS+=(--cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}")
  fi

  if [[ -n "${MAX_PREFILL_TOKENS}" ]]; then
    SERVER_ARGS+=(--max-prefill-tokens "${MAX_PREFILL_TOKENS}")
  fi

  if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
    SERVER_ARGS+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
  fi

  if [[ -n "${MAX_RUNNING_REQUESTS}" ]]; then
    SERVER_ARGS+=(--max-running-requests "${MAX_RUNNING_REQUESTS}")
  fi

  echo
  echo "========================================"
  echo "Tensor parallel size: ${TP}"
  echo "Server log: ${SERVER_LOG}"
  echo "Profile JSON: ${PROFILE_JSON}"
  echo "========================================"

  python "${SERVER_ARGS[@]}" > "${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!

  echo "Server launched, waiting ${SERVER_BOOTSTRAP_GRACE_SEC}s bootstrap grace before /v1/models."

  if ! ensure_server_bootstrapped "${SERVER_BOOTSTRAP_GRACE_SEC}" "${SERVER_PID}" "${SERVER_LOG}"; then
    stop_server
    exit 1
  fi

  echo "Waiting for http://${HOST}:${PORT}/v1/models with timeout ${READY_CHECK_TIMEOUT}s"
  if ! wait_for_models_ready "http://${HOST}:${PORT}" "${READY_CHECK_TIMEOUT}" "${SERVER_PID}" "${SERVER_LOG}"; then
    stop_server
    exit 1
  fi

  # Give the logger a brief moment to flush final init-time lines before we stop.
  sleep 1
  stop_server

  write_profile_json "${SERVER_LOG}" "${PROFILE_JSON}" "${TP}"
  sleep 3
done

echo
echo "Done."
echo "Run root: ${RUN_ROOT}"
