#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark diffusion cold-start weight loading setups.

This script launches ``multimodal_gen/runtime/launch_server.py`` repeatedly,
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
import shlex
import signal
import statistics
import subprocess
import sys
import time
import threading
from queue import Empty, Queue
from pathlib import Path
from typing import Any


TIME_FIELDS_MS = {
    "weight_load:discover_files_ms": "discover_s",
    "weight_load:read_safetensors_ms": "read_s",
    "weight_load:cpu_materialize_ms": "cpu_materialize_s",
    "weight_load:pin_memory_ms": "pin_memory_s",
    "weight_load:h2d_or_param_copy_ms": "h2d_or_param_copy_s",
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
        help="Comma-separated setup names: baseline, none, pageable, pinned, warm-pool.",
    )
    parser.add_argument("--repeat-k", type=int, default=3)
    parser.add_argument("--profile-output-dir", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--base-run-id", default=None)
    parser.add_argument(
        "--launch-cwd",
        default=str(
            Path(__file__).resolve().parents[1]
            / "python"
            / "sglang"
            / "multimodal_gen"
            / "runtime"
        ),
    )
    parser.add_argument("--python", default=sys.executable or "python3")
    parser.add_argument("--attention-backend", default="fa")
    parser.add_argument("--timeout-s", type=float, default=900.0)
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


def build_command(args: argparse.Namespace, setup: str, run_id: str) -> list[str]:
    command = [
        args.python,
        "launch_server.py",
        "--model-path",
        args.model_path,
        "--profile-enabled",
        "--profile-output-dir",
        args.profile_output_dir,
        "--profile-run-id",
        run_id,
        "--num-gpus",
        str(args.num_gpus),
        "--sp-degree",
        str(args.sp_degree),
        "--ulysses-degree",
        str(args.ulysses_degree),
        "--ring-degree",
        str(args.ring_degree),
        "--attention-backend",
        args.attention_backend,
    ]
    if args.model_id:
        command.extend(["--model-id", args.model_id])
    command.extend(SETUP_FLAGS[setup])
    command.extend(extra_launch_args(args))
    return command


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


def _enqueue_stdout(stdout, queue: Queue[str]) -> None:
    try:
        for line in iter(stdout.readline, ""):
            queue.put(line)
    finally:
        try:
            stdout.close()
        except Exception:
            pass


def run_launch(args: argparse.Namespace, setup: str, run_id: str) -> dict[str, Any]:
    command = build_command(args, setup, run_id)
    start = time.perf_counter()
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

    proc = subprocess.Popen(command, **popen_kwargs)
    assert proc.stdout is not None
    output_queue: Queue[str] = Queue()
    reader_thread = threading.Thread(
        target=_enqueue_stdout,
        args=(proc.stdout, output_queue),
        daemon=True,
    )
    reader_thread.start()
    ready = False
    exit_code: int | None = None
    try:
        while True:
            try:
                line = output_queue.get(timeout=0.2)
                print(line, end="")
                if "Uvicorn running on" in line or "Application startup complete" in line:
                    ready = True
                    break
            except Empty:
                pass
            exit_code = proc.poll()
            if exit_code is not None:
                break
            if time.perf_counter() - start > args.timeout_s:
                break
    finally:
        wall_s = time.perf_counter() - start
        terminate_process(proc)
    if exit_code is None:
        exit_code = proc.poll()

    return {
        "setup": setup,
        "run_id": run_id,
        "launch_wall_s": wall_s,
        "ready": ready,
        "exit_code": exit_code,
        "command": " ".join(shlex.quote(part) for part in command),
    }


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
) -> dict[str, float]:
    summary: dict[str, float] = {
        "launch_wall_s": float(run_result["launch_wall_s"]),
        "ready": 1.0 if run_result["ready"] else 0.0,
    }
    grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    for record in records:
        component = str(record.get("component", "unknown"))
        grouped.setdefault((component, _rank_bucket(record)), []).append(
            _metric_values(record)
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
        columns.extend([f"{metric}_avg", f"{metric}_max"])

    rows: list[dict[str, Any]] = []
    for setup, summaries in setup_to_rows.items():
        row: dict[str, Any] = {"setup": setup}
        for metric in metric_names:
            values = [summary[metric] for summary in summaries if metric in summary]
            if not values:
                continue
            row[f"{metric}_avg"] = statistics.mean(values)
            row[f"{metric}_max"] = max(values)
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
        fp.write("| setup | run_id | ready | launch_wall_s | exit_code |\n")
        fp.write("| --- | --- | --- | --- | --- |\n")
        for result in run_results:
            fp.write(
                "| {setup} | {run_id} | {ready} | {wall:.2f} | {exit_code} |\n".format(
                    setup=result["setup"],
                    run_id=result["run_id"],
                    ready=result["ready"],
                    wall=float(result["launch_wall_s"]),
                    exit_code=result["exit_code"],
                )
            )

    return md_path, csv_path


def main() -> None:
    args = parse_args()
    setups = split_setups(args.setups)
    if args.repeat_k <= 0:
        raise ValueError("--repeat-k must be positive")
    base_run_id = args.base_run_id or time.strftime("diffusion_cold_start_%Y%m%d_%H%M%S")

    run_results: list[dict[str, Any]] = []
    run_summaries: list[tuple[str, dict[str, float]]] = []
    for setup in setups:
        for index in range(args.repeat_k):
            run_id = f"{base_run_id}_{setup.replace('-', '_')}_r{index + 1}"
            result = run_launch(args, setup, run_id)
            run_results.append(result)
            records = load_profile_records(args.profile_output_dir, run_id)
            run_summaries.append((setup, summarize_run(result, records)))

    columns, rows = aggregate_by_setup(run_summaries)
    md_path, csv_path = write_outputs(args.summary_output, columns, rows, run_results)
    print(f"Wrote Markdown summary: {md_path}")
    print(f"Wrote CSV summary: {csv_path}")


if __name__ == "__main__":
    main()
