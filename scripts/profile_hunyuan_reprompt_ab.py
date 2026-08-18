#!/usr/bin/env python3
"""Run controlled Hunyuan reprompt latency root-cause A/B experiments."""

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
DEFAULT_SERVER_RANDOM_SEED = 20260818
DECODE_STATUS_PATTERN = re.compile(
    r"Decode batch[^\r\n]*cuda graph: (True|False), "
    r"gen throughput \(token/s\): ([0-9.]+)"
)


def experiment_config(
    name: str,
    tp_size: int,
    attention_backend: str = "flashinfer",
    decode_cuda_graph_backend: str = "full",
    all_reduce_mode: str = "legacy_v1",
    incremental_streaming_output: bool = True,
) -> dict[str, Any]:
    return {
        "name": name,
        "tp_size": tp_size,
        "attention_backend": attention_backend,
        "decode_cuda_graph_backend": decode_cuda_graph_backend,
        "all_reduce_mode": all_reduce_mode,
        "incremental_streaming_output": incremental_streaming_output,
    }


EXPERIMENT_CONFIGS = {
    "streaming": (
        experiment_config(
            "tp1_default_stream", 1, incremental_streaming_output=False
        ),
        experiment_config("tp1_incremental_stream", 1),
    ),
    "allreduce": (
        experiment_config("tp8_nccl", 8, all_reduce_mode="nccl"),
        experiment_config("tp8_legacy_v1", 8),
    ),
    "backend_graph": (
        experiment_config("tp1_auto_full", 1, attention_backend="auto"),
        experiment_config("tp1_flashinfer_full", 1),
        experiment_config("tp1_fa3_full", 1, attention_backend="fa3"),
        experiment_config(
            "tp1_flashinfer_eager", 1, decode_cuda_graph_backend="disabled"
        ),
        experiment_config(
            "tp1_fa3_eager",
            1,
            attention_backend="fa3",
            decode_cuda_graph_backend="disabled",
        ),
    ),
    "tp_stack": (
        experiment_config("tp1_flashinfer_full", 1),
        experiment_config("tp1_fa3_full", 1, attention_backend="fa3"),
        experiment_config("tp8_flashinfer_full_v1", 8),
        experiment_config(
            "tp8_flashinfer_full_nccl", 8, all_reduce_mode="nccl"
        ),
        experiment_config(
            "tp8_flashinfer_eager_v1",
            8,
            decode_cuda_graph_backend="disabled",
        ),
        experiment_config(
            "tp8_fa3_full_v1", 8, attention_backend="fa3"
        ),
        experiment_config(
            "tp8_fa3_full_nccl",
            8,
            attention_backend="fa3",
            all_reduce_mode="nccl",
        ),
        experiment_config(
            "tp8_fa3_eager_v1",
            8,
            attention_backend="fa3",
            decode_cuda_graph_backend="disabled",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a matched-input Hunyuan reprompt latency A/B experiment. "
            "backend_graph isolates TP1 attention and CUDA Graph behavior; "
            "tp_stack isolates TP8 attention, graph, and all-reduce behavior."
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
    parser.add_argument("--scheduler-log-timeout-s", type=float, default=5.0)
    parser.add_argument("--scheduler-log-quiet-s", type=float, default=0.25)
    parser.add_argument("--cooldown-s", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--server-random-seed", type=int, default=DEFAULT_SERVER_RANDOM_SEED
    )
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
    if args.scheduler_log_timeout_s <= 0:
        parser.error("--scheduler-log-timeout-s must be positive")
    if args.scheduler_log_quiet_s <= 0:
        parser.error("--scheduler-log-quiet-s must be positive")
    if args.scheduler_log_quiet_s >= args.scheduler_log_timeout_s:
        parser.error(
            "--scheduler-log-quiet-s must be smaller than "
            "--scheduler-log-timeout-s"
        )
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


def input_ids_sha256(input_ids: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(token_id) for token_id in input_ids).encode("ascii")
    ).hexdigest()


def build_matched_inputs(
    vocab_size: int,
    input_length: int,
    seed: int,
) -> list[dict[str, Any]]:
    matched_inputs: list[dict[str, Any]] = []
    for run_index in range(NUM_RUNS):
        input_seed = seed + run_index
        first_token_id = (seed + run_index) % vocab_size
        input_ids = latency.random_input_ids(
            vocab_size,
            input_length,
            input_seed,
            first_token_id,
        )
        matched_inputs.append(
            {
                "run": run_index + 1,
                "seed": input_seed,
                "sampling_seed": seed + 100_000 + run_index,
                "first_token_id": first_token_id,
                "input_ids_sha256": input_ids_sha256(input_ids),
                "input_ids": input_ids,
            }
        )
    return matched_inputs


def read_log_bytes(log_path: Path, start_offset: int, end_offset: int) -> str:
    with log_path.open("rb") as log_file:
        log_file.seek(start_offset)
        return log_file.read(end_offset - start_offset).decode(
            "utf-8", errors="replace"
        )


def wait_for_request_log_segment(
    log_path: Path,
    start_offset: int,
    timeout_s: float,
    quiet_s: float,
) -> tuple[str, int]:
    deadline = time.monotonic() + timeout_s
    last_size = start_offset
    last_change = time.monotonic()
    while time.monotonic() < deadline:
        current_size = log_path.stat().st_size
        if current_size != last_size:
            last_size = current_size
            last_change = time.monotonic()
        elif (
            current_size > start_offset
            and time.monotonic() - last_change >= quiet_s
        ):
            segment = read_log_bytes(log_path, start_offset, current_size)
            if DECODE_STATUS_PATTERN.search(segment) is not None:
                return segment, current_size
        time.sleep(0.05)
    segment = read_log_bytes(log_path, start_offset, last_size)
    raise TimeoutError(
        f"No stable per-request decode log was found in {log_path} within "
        f"{timeout_s}s. Captured segment:\n{segment[-4000:]}"
    )


def scheduler_decode_metrics(
    log_segment: str,
    expected_cuda_graph: bool,
) -> dict[str, Any]:
    matches = list(DECODE_STATUS_PATTERN.finditer(log_segment))
    throughputs = [float(match.group(2)) for match in matches]
    graph_states = [match.group(1) == "True" for match in matches]
    if not throughputs or any(
        value <= 0 or not math.isfinite(value) for value in throughputs
    ):
        raise RuntimeError(
            "No valid scheduler decode throughput was found in the request log segment"
        )
    unexpected_graph_states = [
        state for state in graph_states if state != expected_cuda_graph
    ]
    if unexpected_graph_states:
        raise RuntimeError(
            f"Expected cuda graph={expected_cuda_graph} for every decode sample, "
            f"got {graph_states}"
        )
    throughput_median = median(throughputs)
    return {
        "decode_log_count": len(throughputs),
        "decode_throughput_tok_s_samples": throughputs,
        "decode_throughput_tok_s_median": throughput_median,
        "decode_throughput_tok_s_min": min(throughputs),
        "decode_throughput_tok_s_max": max(throughputs),
        "scheduler_tpot_ms_from_median": 1000.0 / throughput_median,
        "cuda_graph": expected_cuda_graph,
    }


def aggregate_measurements(records: list[dict[str, Any]]) -> dict[str, float | int]:
    measured = [record for record in records if not record["warmup"]]
    if len(measured) != NUM_RUNS - NUM_WARMUP_RUNS:
        raise RuntimeError("Measured A/B run count is incomplete")
    client_fields = (
        "ttft_ms",
        "tpot_ms",
        "e2e_ms",
        "sse_event_count",
        "sse_bytes",
        "token_update_event_count",
    )
    aggregate: dict[str, float | int] = {
        f"{field}_mean": fmean(float(record[field]) for record in measured)
        for field in client_fields
    }
    request_medians = [
        float(record["scheduler"]["decode_throughput_tok_s_median"])
        for record in measured
    ]
    all_samples = [
        float(sample)
        for record in measured
        for sample in record["scheduler"]["decode_throughput_tok_s_samples"]
    ]
    if not all_samples:
        raise RuntimeError("Measured requests contain no scheduler samples")
    combined_median = median(all_samples)
    aggregate.update(
        {
            "scheduler_decode_log_count_measured": len(all_samples),
            "scheduler_decode_throughput_tok_s_request_median_mean": fmean(
                request_medians
            ),
            "scheduler_decode_throughput_tok_s_combined_median": combined_median,
            "scheduler_tpot_ms_request_median_mean": fmean(
                1000.0 / value for value in request_medians
            ),
            "scheduler_tpot_ms_combined_median": 1000.0 / combined_median,
        }
    )
    aggregate["client_div_scheduler_tpot"] = (
        float(aggregate["tpot_ms_mean"])
        / float(aggregate["scheduler_tpot_ms_combined_median"])
    )
    return aggregate


COMPARISON_FIELDS = (
    "ttft_ms_mean",
    "tpot_ms_mean",
    "e2e_ms_mean",
    "scheduler_tpot_ms_combined_median",
    "scheduler_decode_throughput_tok_s_combined_median",
    "sse_bytes_mean",
)


def pairwise_comparison(
    name: str,
    baseline_name: str,
    candidate_name: str,
    aggregates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline = aggregates[baseline_name]
    candidate = aggregates[candidate_name]
    ratios = {
        f"candidate_div_baseline_{field}": candidate[field] / baseline[field]
        for field in COMPARISON_FIELDS
    }
    return {
        "name": name,
        "baseline": baseline_name,
        "candidate": candidate_name,
        **ratios,
        "client_tpot_speedup_baseline_div_candidate": (
            baseline["tpot_ms_mean"] / candidate["tpot_ms_mean"]
        ),
        "scheduler_tpot_speedup_baseline_div_candidate": (
            baseline["scheduler_tpot_ms_combined_median"]
            / candidate["scheduler_tpot_ms_combined_median"]
        ),
    }


def comparison_specs(experiment: str) -> tuple[tuple[str, str, str], ...]:
    if experiment == "streaming":
        return (
            (
                "incremental_vs_default_stream",
                "tp1_default_stream",
                "tp1_incremental_stream",
            ),
        )
    if experiment == "allreduce":
        return (("legacy_v1_vs_nccl", "tp8_nccl", "tp8_legacy_v1"),)
    if experiment == "backend_graph":
        return (
            (
                "explicit_flashinfer_vs_auto",
                "tp1_auto_full",
                "tp1_flashinfer_full",
            ),
            (
                "fa3_vs_flashinfer_full",
                "tp1_flashinfer_full",
                "tp1_fa3_full",
            ),
            (
                "flashinfer_full_vs_eager",
                "tp1_flashinfer_eager",
                "tp1_flashinfer_full",
            ),
            (
                "fa3_full_vs_eager",
                "tp1_fa3_eager",
                "tp1_fa3_full",
            ),
        )
    return (
        (
            "flashinfer_tp8_v1_vs_tp1",
            "tp1_flashinfer_full",
            "tp8_flashinfer_full_v1",
        ),
        (
            "fa3_tp8_v1_vs_tp1",
            "tp1_fa3_full",
            "tp8_fa3_full_v1",
        ),
        (
            "flashinfer_v1_vs_nccl",
            "tp8_flashinfer_full_nccl",
            "tp8_flashinfer_full_v1",
        ),
        (
            "fa3_v1_vs_nccl",
            "tp8_fa3_full_nccl",
            "tp8_fa3_full_v1",
        ),
        (
            "flashinfer_full_vs_eager_tp8_v1",
            "tp8_flashinfer_eager_v1",
            "tp8_flashinfer_full_v1",
        ),
        (
            "fa3_full_vs_eager_tp8_v1",
            "tp8_fa3_eager_v1",
            "tp8_fa3_full_v1",
        ),
    )


def comparisons(
    experiment: str,
    aggregates: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        pairwise_comparison(name, baseline, candidate, aggregates)
        for name, baseline, candidate in comparison_specs(experiment)
    ]


def validate_matched_input_hashes(
    configs: list[dict[str, Any]], expected_hashes: list[str]
) -> None:
    for config in configs:
        actual_hashes = [run["input_ids_sha256"] for run in config["runs"]]
        if actual_hashes != expected_hashes:
            raise RuntimeError(
                f"Matched-input validation failed for {config['name']}: "
                f"expected={expected_hashes}, actual={actual_hashes}"
            )


def print_human_summary(
    aggregates: dict[str, dict[str, Any]],
    pairwise: list[dict[str, Any]],
) -> None:
    print("[summary] per-configuration means", flush=True)
    print(
        f"{'name':32} {'TP':>2} {'attention':>17} {'graph':>8} "
        f"{'allreduce':>10} {'TTFT ms':>9} {'TPOT ms':>9} "
        f"{'sched ms':>9} {'SSE KiB':>9}",
        flush=True,
    )
    for aggregate in aggregates.values():
        requested_attention = aggregate["attention_backend"]
        resolved_attention = aggregate["resolved_attention_backend"]
        attention_label = (
            f"auto->{resolved_attention}"
            if requested_attention == "auto"
            else resolved_attention
        )
        print(
            f"{aggregate['name']:32} {aggregate['tp_size']:>2} "
            f"{attention_label:>17} "
            f"{aggregate['decode_cuda_graph_backend']:>8} "
            f"{aggregate['all_reduce_mode']:>10} "
            f"{aggregate['ttft_ms_mean']:>9.3f} "
            f"{aggregate['tpot_ms_mean']:>9.3f} "
            f"{aggregate['scheduler_tpot_ms_combined_median']:>9.3f} "
            f"{aggregate['sse_bytes_mean'] / 1024.0:>9.1f}",
            flush=True,
        )
    print("[summary] pairwise scheduler TPOT speedups", flush=True)
    for item in pairwise:
        print(
            f"{item['name']}: {item['baseline']} / {item['candidate']} = "
            f"{item['scheduler_tpot_speedup_baseline_div_candidate']:.4f}x",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    args.model_path = latency.resolve_model_path(args.model_path)
    vocab_size, architecture, model_type = latency.load_model_config(args.model_path)
    if NUM_RUNS > vocab_size:
        raise ValueError(f"vocab_size={vocab_size} is too small for unique prefixes")
    latency.prepare_output_directory(args.output_dir)
    details_path = args.output_dir / "details.json"
    summary_path = args.output_dir / "summary.json"
    matched_inputs = build_matched_inputs(vocab_size, args.input_length, args.seed)
    matched_input_metadata = [
        {key: value for key, value in item.items() if key != "input_ids"}
        for item in matched_inputs
    ]
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
        "server_random_seed": args.server_random_seed,
        "matched_inputs": matched_input_metadata,
        "timing_semantics": (
            "Client token timestamps are recorded immediately after raw SSE line "
            "receipt. Every configuration reuses the same five input_ids and "
            "sampling seeds. Scheduler decode samples are sliced from the server "
            "log per request and only the final three requests are aggregated."
        ),
        "system_before": system_snapshot(),
        "configs": [],
    }
    latency.save_json(details_path, details)
    aggregates: dict[str, dict[str, Any]] = {}

    try:
        for config in EXPERIMENT_CONFIGS[args.experiment]:
            name = str(config["name"])
            tp_size = int(config["tp_size"])
            incremental = bool(config["incremental_streaming_output"])
            all_reduce_mode = str(config["all_reduce_mode"])
            attention_backend = str(config["attention_backend"])
            decode_cuda_graph_backend = str(config["decode_cuda_graph_backend"])
            expected_cuda_graph = decode_cuda_graph_backend == "full"
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
                attention_backend=attention_backend,
                decode_cuda_graph_backend=decode_cuda_graph_backend,
                server_random_seed=args.server_random_seed,
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
                startup_validation = latency.validate_server_log(
                    log_path,
                    args.model_path,
                    tp_size,
                    incremental_streaming_output=incremental,
                    all_reduce_mode=all_reduce_mode,
                    attention_backend=attention_backend,
                    decode_cuda_graph_backend=decode_cuda_graph_backend,
                )
                config_record["startup_validation"] = startup_validation
                config_record["model_card"] = model_card
                config_record["validated_model_entry"] = model_entry
                latency.save_json(details_path, details)
                session = requests.Session()
                session.trust_env = False

                for run_index, matched_input in enumerate(matched_inputs):
                    is_warmup = run_index < NUM_WARMUP_RUNS
                    run_record: dict[str, Any] = {
                        **{
                            key: value
                            for key, value in matched_input.items()
                            if key != "input_ids"
                        },
                        "warmup": is_warmup,
                        "status": "running",
                    }
                    config_record["runs"].append(run_record)
                    latency.save_json(details_path, details)
                    print(
                        f"[request] {name} run={run_index + 1}/{NUM_RUNS} "
                        f"{'warmup' if is_warmup else 'measure'}",
                        flush=True,
                    )
                    log_start_offset = log_path.stat().st_size
                    run_record["scheduler_log_start_offset"] = log_start_offset
                    try:
                        metrics = latency.measure_request(
                            session,
                            url,
                            matched_input["input_ids"],
                            args.output_length,
                            args.request_timeout_s,
                            sampling_seed=int(matched_input["sampling_seed"]),
                        )
                        log_segment, log_end_offset = wait_for_request_log_segment(
                            log_path,
                            log_start_offset,
                            args.scheduler_log_timeout_s,
                            args.scheduler_log_quiet_s,
                        )
                        scheduler_metrics = scheduler_decode_metrics(
                            log_segment, expected_cuda_graph
                        )
                    except Exception as exc:
                        time.sleep(0.2)
                        server_returncode = process.poll()
                        failure = (
                            f"{type(exc).__name__}: {exc}\n"
                            f"server_process_returncode={server_returncode!r}\n"
                            f"server log tail:\n{latency.read_log_tail(log_path)}"
                        )
                        run_record.update(
                            status="failed",
                            error=failure,
                            server_process_returncode=server_returncode,
                        )
                        latency.save_json(details_path, details)
                        raise RuntimeError(failure) from exc
                    run_record.update(
                        status="completed",
                        scheduler_log_end_offset=log_end_offset,
                        scheduler=scheduler_metrics,
                        **metrics,
                    )
                    latency.save_json(details_path, details)

                runtime_validation = latency.validate_server_log(
                    log_path,
                    args.model_path,
                    tp_size,
                    require_decode_execution=True,
                    incremental_streaming_output=incremental,
                    all_reduce_mode=all_reduce_mode,
                    attention_backend=attention_backend,
                    decode_cuda_graph_backend=decode_cuda_graph_backend,
                )
                aggregate = {
                    "name": name,
                    "tp_size": tp_size,
                    "incremental_streaming_output": incremental,
                    "all_reduce_mode": all_reduce_mode,
                    "attention_backend": attention_backend,
                    "resolved_attention_backend": runtime_validation[
                        "resolved_attention_backend"
                    ],
                    "decode_cuda_graph_backend": decode_cuda_graph_backend,
                    **aggregate_measurements(config_record["runs"]),
                }
                aggregates[name] = aggregate
                config_record.update(
                    status="completed",
                    runtime_validation=runtime_validation,
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

        expected_hashes = [
            item["input_ids_sha256"] for item in matched_input_metadata
        ]
        validate_matched_input_hashes(details["configs"], expected_hashes)
        pairwise = comparisons(args.experiment, aggregates)
        summary = {
            "experiment": args.experiment,
            "input_length": args.input_length,
            "output_length": args.output_length,
            "matched_input_hashes": expected_hashes,
            "aggregates": aggregates,
            "comparisons": pairwise,
        }
        latency.save_json(summary_path, summary)
        details["system_after"] = system_snapshot()
        details["status"] = "completed"
        details["summary_path"] = str(summary_path)
        details["summary"] = summary
        latency.save_json(details_path, details)
        print_human_summary(aggregates, pairwise)
        print("[done] A/B summary JSON", flush=True)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    except BaseException as exc:
        details["system_after"] = system_snapshot()
        details["status"] = "failed"
        details["error"] = str(exc)
        latency.save_json(details_path, details)
        raise


if __name__ == "__main__":
    main()
