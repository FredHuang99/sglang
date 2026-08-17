#!/usr/bin/env python3
"""Run short Hunyuan reprompt streaming and all-reduce latency A/B checks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import fmean, median
from typing import Any

import requests

import profile_hunyuan_reprompt_latency as latency


NUM_RUNS = 5
NUM_WARMUP_RUNS = 2
DEFAULT_INPUT_LENGTH = 128
DEFAULT_OUTPUT_LENGTH = 512
DECODE_THROUGHPUT_PATTERN = re.compile(
    r"Decode batch[^\r\n]*gen throughput \(token/s\): ([0-9.]+)"
)
EXPERIMENT_CONFIGS = {
    "streaming": (
        {
            "name": "tp1_default_stream",
            "tp_size": 1,
            "incremental_streaming_output": False,
            "all_reduce_mode": "legacy_v1",
        },
        {
            "name": "tp1_incremental_stream",
            "tp_size": 1,
            "incremental_streaming_output": True,
            "all_reduce_mode": "legacy_v1",
        },
    ),
    "allreduce": (
        {
            "name": "tp8_nccl",
            "tp_size": 8,
            "incremental_streaming_output": False,
            "all_reduce_mode": "nccl",
        },
        {
            "name": "tp8_legacy_v1",
            "tp_size": 8,
            "incremental_streaming_output": False,
            "all_reduce_mode": "legacy_v1",
        },
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one short 128/512 A/B experiment. The streaming experiment "
            "compares default versus incremental output on TP1. The allreduce "
            "experiment compares NCCL versus legacy Custom AllReduce V1 on TP8."
        )
    )
    parser.add_argument(
        "--experiment", choices=tuple(EXPERIMENT_CONFIGS), required=True
    )
    parser.add_argument(
        "--model-path", type=Path, default=latency.DEFAULT_MODEL_PATH
    )
    parser.add_argument(
        "--served-model-name", default=latency.DEFAULT_SERVED_MODEL_NAME
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--input-length", type=int, default=DEFAULT_INPUT_LENGTH)
    parser.add_argument("--output-length", type=int, default=DEFAULT_OUTPUT_LENGTH)
    parser.add_argument("--server-timeout-s", type=float, default=1800.0)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--cooldown-s", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/outputs/hunyuan_reprompt_ab"),
    )
    args = parser.parse_args()
    args.model_path = args.model_path.expanduser()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.input_length <= 0:
        parser.error("--input-length must be positive")
    if args.output_length <= 1:
        parser.error("--output-length must be greater than one")
    if args.server_timeout_s <= 0 or args.request_timeout_s <= 0:
        parser.error("timeouts must be positive")
    if args.cooldown_s < 0:
        parser.error("--cooldown-s must be non-negative")
    return args


def command_snapshot(command: list[str], timeout_s: float = 10.0) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": command,
            "return_code": result.returncode,
            "output": result.stdout.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def package_versions() -> dict[str, str | None]:
    names = (
        "torch",
        "sglang",
        "sglang-kernel",
        "flashinfer-python",
        "apache-tvm-ffi",
        "transformers",
        "requests",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_snapshot() -> dict[str, Any]:
    return command_snapshot(
        [
            "nvidia-smi",
            "--query-gpu=index,name,pstate,clocks.current.sm,"
            "clocks.current.memory,power.draw,power.limit,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )


def system_snapshot() -> dict[str, Any]:
    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version,
        "packages": package_versions(),
        "selected_environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": os.environ.get(
                "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"
            ),
            "SGLANG_USE_JIT_ALL_REDUCE": os.environ.get(
                "SGLANG_USE_JIT_ALL_REDUCE"
            ),
        },
        "git_revision": command_snapshot(["git", "rev-parse", "HEAD"]),
        "nvcc": command_snapshot(["nvcc", "--version"]),
        "gpus": gpu_snapshot(),
        "gpu_topology": command_snapshot(["nvidia-smi", "topo", "-m"]),
    }


def scheduler_decode_metrics(log_path: Path) -> dict[str, float | int]:
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    throughputs = [
        float(match.group(1))
        for match in DECODE_THROUGHPUT_PATTERN.finditer(log_text)
    ]
    if not throughputs or any(
        value <= 0 or not math.isfinite(value) for value in throughputs
    ):
        raise RuntimeError(
            f"No valid scheduler decode throughput was found in {log_path}"
        )
    scheduler_tpot_values = [1000.0 / value for value in throughputs]
    return {
        "decode_log_count": len(throughputs),
        "decode_throughput_tok_s_mean": fmean(throughputs),
        "decode_throughput_tok_s_median": median(throughputs),
        "decode_throughput_tok_s_min": min(throughputs),
        "decode_throughput_tok_s_max": max(throughputs),
        "scheduler_tpot_ms_mean": fmean(scheduler_tpot_values),
        "scheduler_tpot_ms_median": median(scheduler_tpot_values),
    }


def average_measurements(records: list[dict[str, Any]]) -> dict[str, float]:
    measured = [record for record in records if not record["warmup"]]
    if len(measured) != NUM_RUNS - NUM_WARMUP_RUNS:
        raise RuntimeError("Measured A/B run count is incomplete")
    fields = (
        "ttft_ms",
        "tpot_ms",
        "e2e_ms",
        "sse_event_count",
        "sse_bytes",
        "token_update_event_count",
    )
    return {
        f"{field}_mean": fmean(float(record[field]) for record in measured)
        for field in fields
    }


def comparison(
    experiment: str, aggregates: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if experiment == "streaming":
        baseline_name = "tp1_default_stream"
        candidate_name = "tp1_incremental_stream"
    else:
        baseline_name = "tp8_nccl"
        candidate_name = "tp8_legacy_v1"
    baseline = aggregates[baseline_name]
    candidate = aggregates[candidate_name]
    ratio_fields = (
        "ttft_ms_mean",
        "tpot_ms_mean",
        "e2e_ms_mean",
        "scheduler_tpot_ms_mean",
        "decode_throughput_tok_s_mean",
        "sse_bytes_mean",
    )
    ratios = {
        f"candidate_div_baseline_{field}": candidate[field] / baseline[field]
        for field in ratio_fields
    }
    return {
        "baseline": baseline_name,
        "candidate": candidate_name,
        **ratios,
    }


def main() -> None:
    args = parse_args()
    args.model_path = latency.resolve_model_path(args.model_path)
    vocab_size, architecture, model_type = latency.load_model_config(args.model_path)
    if NUM_RUNS > vocab_size:
        raise ValueError(f"vocab_size={vocab_size} is too small for unique prefixes")
    latency.prepare_output_directory(args.output_dir)
    details_path = args.output_dir / "details.json"
    summary_path = args.output_dir / "summary.json"
    details: dict[str, Any] = {
        "status": "running",
        "experiment": args.experiment,
        "model_path": str(args.model_path),
        "served_model_name": args.served_model_name,
        "architecture": architecture,
        "model_type": model_type,
        "vocab_size": vocab_size,
        "input_length": args.input_length,
        "output_length": args.output_length,
        "num_runs": NUM_RUNS,
        "num_warmup_runs": NUM_WARMUP_RUNS,
        "timing_semantics": (
            "Token timestamps are recorded immediately after raw SSE line receipt. "
            "Token counts are extracted directly from SSE bytes without full JSON parsing."
        ),
        "system_before": system_snapshot(),
        "configs": [],
    }
    latency.save_json(details_path, details)
    aggregates: dict[str, dict[str, Any]] = {}

    try:
        for config_index, config in enumerate(EXPERIMENT_CONFIGS[args.experiment]):
            name = str(config["name"])
            tp_size = int(config["tp_size"])
            incremental = bool(config["incremental_streaming_output"])
            all_reduce_mode = str(config["all_reduce_mode"])
            port = latency.find_free_port(args.host)
            url = latency.base_url(args.host, port)
            log_path = args.output_dir / f"server_{name}.log"
            command = latency.build_server_command(
                args.model_path,
                args.served_model_name,
                args.host,
                port,
                tp_size,
                incremental_streaming_output=incremental,
                all_reduce_mode=all_reduce_mode,
            )
            config_record: dict[str, Any] = {
                **config,
                "status": "running",
                "url": url,
                "command": command,
                "log_path": str(log_path),
                "runs": [],
            }
            details["configs"].append(config_record)
            latency.save_json(details_path, details)
            process: subprocess.Popen[bytes] | None = None
            log_file = None
            session: requests.Session | None = None
            try:
                print(f"[server] {name} launching on {url}", flush=True)
                process, log_file = latency.start_server(
                    command, log_path, all_reduce_mode=all_reduce_mode
                )
                model_card = latency.wait_for_ready(
                    process,
                    url,
                    args.served_model_name,
                    args.server_timeout_s,
                    log_path,
                )
                model_entry = latency.validate_model_card(
                    model_card, args.served_model_name
                )
                config_record["startup_validation"] = latency.validate_server_log(
                    log_path,
                    args.model_path,
                    tp_size,
                    incremental_streaming_output=incremental,
                    all_reduce_mode=all_reduce_mode,
                )
                config_record["model_card"] = model_card
                config_record["validated_model_entry"] = model_entry
                latency.save_json(details_path, details)
                session = requests.Session()
                session.trust_env = False

                for run_index in range(NUM_RUNS):
                    is_warmup = run_index < NUM_WARMUP_RUNS
                    seed = args.seed + config_index * 10_000 + run_index
                    first_token_id = (
                        args.seed + config_index * 1_000 + run_index
                    ) % vocab_size
                    input_ids = latency.random_input_ids(
                        vocab_size,
                        args.input_length,
                        seed,
                        first_token_id,
                    )
                    input_ids_sha256 = hashlib.sha256(
                        ",".join(str(token_id) for token_id in input_ids).encode(
                            "ascii"
                        )
                    ).hexdigest()
                    run_record: dict[str, Any] = {
                        "run": run_index + 1,
                        "warmup": is_warmup,
                        "seed": seed,
                        "first_token_id": first_token_id,
                        "input_ids_sha256": input_ids_sha256,
                        "status": "running",
                    }
                    config_record["runs"].append(run_record)
                    latency.save_json(details_path, details)
                    print(
                        f"[request] {name} run={run_index + 1}/{NUM_RUNS} "
                        f"{'warmup' if is_warmup else 'measure'}",
                        flush=True,
                    )
                    try:
                        metrics = latency.measure_request(
                            session,
                            url,
                            input_ids,
                            args.output_length,
                            args.request_timeout_s,
                        )
                    except Exception as exc:
                        run_record.update(status="failed", error=str(exc))
                        latency.save_json(details_path, details)
                        raise
                    run_record.update(status="completed", **metrics)
                    latency.save_json(details_path, details)

                time.sleep(0.2)
                config_record["runtime_validation"] = latency.validate_server_log(
                    log_path,
                    args.model_path,
                    tp_size,
                    require_decode_execution=True,
                    incremental_streaming_output=incremental,
                    all_reduce_mode=all_reduce_mode,
                )
                aggregate = {
                    "name": name,
                    "tp_size": tp_size,
                    "incremental_streaming_output": incremental,
                    "all_reduce_mode": all_reduce_mode,
                    **average_measurements(config_record["runs"]),
                    **scheduler_decode_metrics(log_path),
                }
                aggregates[name] = aggregate
                config_record.update(
                    status="completed",
                    aggregate=aggregate,
                    gpu_after_requests=gpu_snapshot(),
                )
                latency.save_json(details_path, details)
            except Exception as exc:
                config_record.update(status="failed", error=str(exc))
                raise
            finally:
                if session is not None:
                    session.close()
                latency.stop_server(process, log_file)
                latency.save_json(details_path, details)
                if args.cooldown_s:
                    time.sleep(args.cooldown_s)

        summary = {
            "experiment": args.experiment,
            "input_length": args.input_length,
            "output_length": args.output_length,
            "aggregates": aggregates,
            "comparison": comparison(args.experiment, aggregates),
        }
        latency.save_json(summary_path, summary)
        details["system_after"] = system_snapshot()
        details["status"] = "completed"
        details["summary_path"] = str(summary_path)
        details["summary"] = summary
        latency.save_json(details_path, details)
        print("[done] A/B summary", flush=True)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    except BaseException as exc:
        details["system_after"] = system_snapshot()
        details["status"] = "failed"
        details["error"] = str(exc)
        latency.save_json(details_path, details)
        raise


if __name__ == "__main__":
    main()
