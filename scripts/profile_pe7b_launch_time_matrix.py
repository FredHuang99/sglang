#!/usr/bin/env python3
"""Profile PE7B launch time for TP 1/2/4/8 with two setup groups."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import profile_server_launch_breakdown as breakdown
import profile_server_launch_time as launch_time

DEFAULT_GPU_NUMS = [1, 2, 4, 8]
PE7B_PRESET_KEY = "promptenhancer-7b"

PE7B_SETUPS: dict[str, launch_time.LLMSetup] = {
    "unconstrained": launch_time.LLMSetup(
        key="unconstrained",
        description=(
            "PE7B preset baseline: context_length=32768 and "
            "mem_fraction_static=0.90, with batch/KV/chunk/cuda graph limits left "
            "for SGLang to infer."
        ),
    ),
    "constrained_4096": launch_time.LLM_SETUPS["constrained_4096"],
}

CSV_COLUMNS = [
    "row_name",
    "status",
    "tp_size",
    "llm_setup",
    "e2e_launch_time_s",
    "weight_load_time_s",
    "cuda_graph_capture_time_s",
    "mem_fraction_static",
    "context_length",
    "max_running_requests",
    "max_total_tokens",
    "cuda_graph_max_bs",
    "chunked_prefill_size",
]

DETAIL_COLUMNS = [
    "row_name",
    "status",
    "reason",
    "tp_size",
    "llm_setup",
    "case_dir",
    "server_log_path",
    "launch_tasks_path",
    "launch_breakdown_path",
    "command_json",
]


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_int_list(raw_values: list[str] | None, *, default: list[int]) -> list[int]:
    if not raw_values:
        return list(default)
    tokens: list[str] = []
    for raw in raw_values:
        tokens.extend(part for part in raw.replace(",", " ").split() if part)
    values: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        value = int(token)
        if value <= 0:
            raise ValueError(f"GPU number must be positive, got {value}.")
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile PromptEnhancer-7B launch time for TP 1/2/4/8 under "
            "unconstrained and constrained launch settings."
        )
    )
    parser.add_argument(
        "--gpu-nums",
        nargs="*",
        default=None,
        help="GPU/TP sizes to run. Defaults to 1 2 4 8.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override the PE7B preset model path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to /workspace/outputs/pe7b_launch_time/<stamp>.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Main summary CSV path. Defaults under --output-dir.",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=1800,
        help="Per-launch readiness timeout in seconds.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write the commands and CSV skeleton; do not launch servers.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed/skipped case.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep per-case logs and breakdown JSON.",
    )
    args = parser.parse_args(argv)
    args.gpu_nums = parse_int_list(args.gpu_nums, default=DEFAULT_GPU_NUMS)
    return args


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    return Path(f"/workspace/outputs/pe7b_launch_time/{now_stamp()}").resolve()


def get_preset(model_path: str | None = None) -> launch_time.LaunchPreset:
    preset = launch_time.PRESETS[PE7B_PRESET_KEY]
    if model_path:
        preset = replace(preset, model_path=model_path)
    return preset


def task_elapsed_s(record: dict[str, Any], task_key: str) -> float | None:
    task = (record.get("tasks") or {}).get(task_key)
    if not isinstance(task, dict) or not task.get("observed"):
        return None
    value = task.get("elapsed_s_max")
    return float(value) if isinstance(value, (int, float)) else None


def csv_number(value: Any) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def param_value(value: Any) -> str:
    return "auto" if value is None else str(value)


def make_record(
    *,
    preset: launch_time.LaunchPreset,
    gpu_num: int,
    setup: launch_time.LLMSetup,
    case_dir: Path,
    status: str,
    reason: str | None,
    command: list[str] | None,
) -> dict[str, Any]:
    task_summary = breakdown.summarize_launch_tasks([], family="sglang")
    return breakdown.make_case_record(
        preset=preset,
        gpu_count=gpu_num,
        requested_parallelism={"tp_size": gpu_num},
        resolved_parallelism={"tp_size": gpu_num},
        case_name=f"tp{gpu_num}__{setup.key}",
        case_dir=case_dir,
        launch_command=command,
        status=status,
        reason=reason,
        health_ready_ns=None,
        models_ready_ns=None,
        model_card=None,
        launch_tasks_path=case_dir / "launch_tasks.jsonl",
        task_events=[],
        task_summary=task_summary,
        extra={
            "setup_name": setup.key,
            "setup_args": breakdown.llm_setup_args(setup),
        },
    )


def build_csv_row(
    record: dict[str, Any],
    *,
    preset: launch_time.LaunchPreset,
    setup: launch_time.LLMSetup,
    gpu_num: int,
) -> dict[str, str]:
    return {
        "row_name": f"tp{gpu_num}_{setup.key}",
        "status": str(record.get("status") or ""),
        "tp_size": str(gpu_num),
        "llm_setup": setup.key,
        "e2e_launch_time_s": csv_number(record.get("launch_time_s")),
        "weight_load_time_s": csv_number(task_elapsed_s(record, "load_weight")),
        "cuda_graph_capture_time_s": csv_number(
            task_elapsed_s(record, "cuda_graph_capture")
        ),
        "mem_fraction_static": param_value(preset.mem_fraction_static),
        "context_length": param_value(preset.context_length),
        "max_running_requests": param_value(setup.max_running_requests),
        "max_total_tokens": param_value(setup.max_total_tokens),
        "cuda_graph_max_bs": param_value(setup.cuda_graph_max_bs),
        "chunked_prefill_size": param_value(setup.chunked_prefill_size),
    }


def build_detail_row(
    record: dict[str, Any],
    *,
    setup: launch_time.LLMSetup,
    gpu_num: int,
) -> dict[str, str]:
    return {
        "row_name": f"tp{gpu_num}_{setup.key}",
        "status": str(record.get("status") or ""),
        "reason": str(record.get("reason") or ""),
        "tp_size": str(gpu_num),
        "llm_setup": setup.key,
        "case_dir": str(record.get("case_dir") or ""),
        "server_log_path": str(record.get("server_log_path") or ""),
        "launch_tasks_path": str(record.get("launch_tasks_path") or ""),
        "launch_breakdown_path": str(record.get("launch_breakdown_path") or ""),
        "command_json": json.dumps(record.get("launch_command") or [], ensure_ascii=False),
    }


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    *,
    records: list[dict[str, Any]],
    csv_rows: list[dict[str, str]],
    detail_rows: list[dict[str, str]],
    csv_path: Path,
    details_csv_path: Path,
    details_json_path: Path,
) -> None:
    write_csv(csv_path, csv_rows, CSV_COLUMNS)
    write_csv(details_csv_path, detail_rows, DETAIL_COLUMNS)
    launch_time.save_json(
        details_json_path,
        {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "records": records,
        },
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    preset = get_preset(args.model_path)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (
        Path(args.csv_path).expanduser().resolve()
        if args.csv_path
        else output_dir / "pe7b_launch_time_summary.csv"
    )
    details_csv_path = output_dir / "pe7b_launch_time_details.csv"
    details_json_path = output_dir / "pe7b_launch_time_details.json"
    visible_gpu_count = launch_time.resolve_visible_gpu_count()

    records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, str]] = []
    detail_rows: list[dict[str, str]] = []

    for gpu_num in args.gpu_nums:
        for setup in PE7B_SETUPS.values():
            case_dir = output_dir / preset.output_subdir / f"tp{gpu_num}" / setup.key
            command = launch_time.build_promptenhancer_command(
                preset=preset,
                tp_size=gpu_num,
                host=args.host,
                port=30000,
                llm_setup=setup,
            )
            if args.dry_run:
                record = make_record(
                    preset=preset,
                    gpu_num=gpu_num,
                    setup=setup,
                    case_dir=case_dir,
                    status="dry_run",
                    reason=None,
                    command=command,
                )
            elif visible_gpu_count < gpu_num:
                record = make_record(
                    preset=preset,
                    gpu_num=gpu_num,
                    setup=setup,
                    case_dir=case_dir,
                    status="skipped",
                    reason=(
                        f"Requested gpu_num={gpu_num}, but only "
                        f"{visible_gpu_count} visible GPU(s) are available."
                    ),
                    command=command,
                )
            else:
                record = breakdown.measure_promptenhancer_case(
                    preset=preset,
                    gpu_count=gpu_num,
                    llm_setup=setup,
                    host=args.host,
                    timeout_s=args.timeout_s,
                    case_dir=case_dir,
                )

            records.append(record)
            breakdown.write_case_breakdown(case_dir, record)
            csv_rows.append(
                build_csv_row(record, preset=preset, setup=setup, gpu_num=gpu_num)
            )
            detail_rows.append(build_detail_row(record, setup=setup, gpu_num=gpu_num))
            write_outputs(
                records=records,
                csv_rows=csv_rows,
                detail_rows=detail_rows,
                csv_path=csv_path,
                details_csv_path=details_csv_path,
                details_json_path=details_json_path,
            )
            launch_time.cleanup_case_dir(case_dir, args.keep_artifacts)

            if args.fail_fast and record.get("status") not in {"completed", "dry_run"}:
                raise SystemExit(f"Stopping after {record.get('status')}: {record.get('reason')}")

    launch_time.logger.info("PE7B launch summary CSV written to %s", csv_path)


if __name__ == "__main__":
    main()
