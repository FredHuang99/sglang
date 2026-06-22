#!/usr/bin/env python3
"""Profile sglang-diffusion launch time with one TP=1/SP=GPU case per GPU count."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import profile_server_launch_breakdown as breakdown
import profile_server_launch_time as launch_time

try:
    from sglang.multimodal_gen.runtime.disaggregation.roles import (
        RoleType,
        get_module_role,
    )
except Exception:
    from enum import Enum

    class RoleType(str, Enum):
        ENCODER = "encoder"
        DENOISER = "denoiser"
        DECODER = "decoder"

    def get_module_role(module_name: str) -> RoleType | None:
        encoder_prefixes = (
            "text_encoder",
            "tokenizer",
            "image_encoder",
            "image_processor",
            "processor",
            "connectors",
            "vision_language_encoder",
        )
        if any(
            module_name == p or module_name.startswith(p + "_")
            for p in encoder_prefixes
        ):
            return RoleType.ENCODER
        if module_name in {"hy3dshape_conditioner", "hy3dshape_image_processor"}:
            return RoleType.ENCODER

        denoising_prefixes = (
            "transformer",
            "video_dit",
            "audio_dit",
            "dual_tower_bridge",
        )
        if any(
            module_name == p or module_name.startswith(p + "_")
            for p in denoising_prefixes
        ):
            return RoleType.DENOISER
        if module_name == "hy3dshape_model":
            return RoleType.DENOISER

        decoder_prefixes = ("vae", "audio_vae", "video_vae", "vocoder")
        if any(
            module_name == p or module_name.startswith(p + "_")
            for p in decoder_prefixes
        ):
            return RoleType.DECODER
        if module_name == "hy3dshape_vae":
            return RoleType.DECODER
        return None

DEFAULT_GPU_NUMS = [1, 2, 4, 8]
DIFFUSION_MODELS = ["wan2.2-ti2v-5b", "wan2.1-t2v-1.3b", "z-image"]

ROLE_TO_COLUMN_PREFIX = {
    RoleType.ENCODER: "text_encoder",
    RoleType.DENOISER: "denoiser",
    RoleType.DECODER: "decoder",
}

CSV_COLUMNS = [
    "row_name",
    "status",
    "model",
    "gpu_num",
    "tp_size",
    "sp_degree",
    "ulysses_degree",
    "ring_degree",
    "e2e_launch_time_s",
    "text_encoder_e2e_launch_time_s",
    "denoiser_e2e_launch_time_s",
    "decoder_e2e_launch_time_s",
    "text_encoder_weight_size_gb",
    "text_encoder_weight_load_time_s",
    "text_encoder_cpu_materialization_time_s",
    "denoiser_weight_size_gb",
    "denoiser_weight_load_time_s",
    "denoiser_cpu_materialization_time_s",
    "decoder_weight_size_gb",
    "decoder_weight_load_time_s",
    "decoder_cpu_materialization_time_s",
]

DETAIL_COLUMNS = [
    "row_name",
    "status",
    "reason",
    "model",
    "gpu_num",
    "tp_size",
    "sp_degree",
    "ulysses_degree",
    "ring_degree",
    "shared_component_launch_time_s",
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
            "Profile launch-time summary for one sglang-diffusion model with "
            "exactly one TP=1/SP=GPU case per requested GPU count."
        )
    )
    parser.add_argument("--model", required=True, choices=DIFFUSION_MODELS)
    parser.add_argument(
        "--gpu-nums",
        nargs="*",
        default=None,
        help="GPU counts to run. Defaults to 1 2 4 8.",
    )
    parser.add_argument("--model-path", default=None, help="Override preset model path.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to /workspace/outputs/diffusion_launch_time/<model>/<stamp>.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Main summary CSV path. Defaults under --output-dir.",
    )
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
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
    return Path(
        f"/workspace/outputs/diffusion_launch_time/{args.model}/{now_stamp()}"
    ).resolve()


def get_preset(model_key: str, model_path: str | None = None) -> launch_time.LaunchPreset:
    preset = launch_time.PRESETS[model_key]
    if preset.family != "diffusion":
        raise ValueError(f"{model_key} is not a diffusion preset.")
    if model_path:
        preset = replace(preset, model_path=model_path)
    return preset


def resolve_diffusion_run_config(model_key: str, gpu_num: int) -> launch_time.RunConfig:
    sp_degree = gpu_num
    if model_key == "wan2.2-ti2v-5b":
        ulysses_degree = sp_degree
    elif model_key == "wan2.1-t2v-1.3b":
        ulysses_degree = 4 if sp_degree == 8 else sp_degree
    elif model_key == "z-image":
        ulysses_degree = 1 if sp_degree == 1 else 2
    else:
        raise ValueError(f"Unsupported diffusion preset: {model_key}")

    if sp_degree % ulysses_degree != 0:
        raise ValueError(
            f"Invalid reduced SP policy for {model_key}: sp={sp_degree}, "
            f"ulysses={ulysses_degree}."
        )
    ring_degree = sp_degree // ulysses_degree
    return launch_time.RunConfig(
        name=f"tp1_sp{sp_degree}_u{ulysses_degree}_r{ring_degree}",
        mode="tp1_sp_gpu",
        num_gpus=gpu_num,
        tp_size=1,
        sp_degree=sp_degree,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
    )


def validate_run_config(run_config: launch_time.RunConfig, gpu_num: int) -> None:
    if run_config.tp_size != 1:
        raise ValueError(f"Diffusion summary only supports tp_size=1, got {run_config.tp_size}.")
    if run_config.sp_degree != gpu_num:
        raise ValueError(
            f"Diffusion summary requires sp_degree=gpu_num={gpu_num}, "
            f"got {run_config.sp_degree}."
        )
    if run_config.ulysses_degree is None or run_config.ring_degree is None:
        raise ValueError("ulysses_degree and ring_degree must be explicit.")
    if run_config.ulysses_degree * run_config.ring_degree != run_config.sp_degree:
        raise ValueError(
            "sp_degree must equal ulysses_degree * ring_degree, got "
            f"{run_config.sp_degree} vs "
            f"{run_config.ulysses_degree} * {run_config.ring_degree}."
        )


def command_value(command: list[str], flag: str) -> str | None:
    try:
        index = command.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def validate_command_parallelism(command: list[str], gpu_num: int) -> None:
    if command_value(command, "--tp-size") != "1":
        raise ValueError(f"Command must contain --tp-size 1: {command}")
    if command_value(command, "--sp-degree") != str(gpu_num):
        raise ValueError(f"Command must contain --sp-degree {gpu_num}: {command}")


def build_command(
    *,
    preset: launch_time.LaunchPreset,
    run_config: launch_time.RunConfig,
    host: str,
    case_dir: Path,
) -> list[str]:
    command = breakdown.build_diffusion_command(
        preset=preset,
        run_config=run_config,
        host=host,
        port=30000,
        scheduler_port=30001,
        master_port=30002,
        case_dir=case_dir,
    )
    validate_command_parallelism(command, run_config.num_gpus)
    return command


def task_elapsed_s(record: dict[str, Any], task_key: str) -> float | None:
    task = (record.get("tasks") or {}).get(task_key)
    if not isinstance(task, dict) or not task.get("observed"):
        return None
    value = task.get("elapsed_s_max")
    return float(value) if isinstance(value, (int, float)) else None


def task_extra_values(record: dict[str, Any], task_key: str, field: str) -> list[float]:
    task = (record.get("tasks") or {}).get(task_key)
    if not isinstance(task, dict) or not task.get("observed"):
        return []
    values: list[float] = []
    for event in task.get("events", []) or []:
        extra = event.get("extra") if isinstance(event, dict) else None
        if not isinstance(extra, dict):
            continue
        value = extra.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def components_for_role(record: dict[str, Any], role: RoleType) -> list[str]:
    components: set[str] = set()
    tasks = record.get("tasks") or {}
    for key in tasks:
        if ":" not in key:
            continue
        task_name, component = key.split(":", 1)
        if task_name not in {
            "component_load",
            "component_cpu_materialization",
            "component_load_stats",
        }:
            continue
        if get_module_role(component) == role:
            components.add(component)
    return sorted(components)


def sum_or_none(values: list[float | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    if not present:
        return None
    return sum(present)


def component_elapsed(record: dict[str, Any], task_name: str, component: str) -> float | None:
    return task_elapsed_s(record, f"{task_name}:{component}")


def component_weight_size(record: dict[str, Any], component: str) -> float | None:
    values = task_extra_values(
        record, f"component_load_stats:{component}", "loaded_weight_file_size_gb"
    )
    if values:
        return max(values)
    values = task_extra_values(
        record, f"component_load_stats:{component}", "final_module_size_gb"
    )
    if values:
        return max(values)
    return None


def role_weight_size(record: dict[str, Any], role: RoleType) -> float | None:
    return sum_or_none(
        [component_weight_size(record, component) for component in components_for_role(record, role)]
    )


def role_component_elapsed(
    record: dict[str, Any], role: RoleType, task_name: str
) -> float | None:
    return sum_or_none(
        [
            component_elapsed(record, task_name, component)
            for component in components_for_role(record, role)
        ]
    )


def role_e2e_elapsed(record: dict[str, Any], role: RoleType) -> float | None:
    return task_elapsed_s(record, f"role_launch_total:{role.value}")


def csv_number(value: Any) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def make_record(
    *,
    preset: launch_time.LaunchPreset,
    run_config: launch_time.RunConfig,
    case_dir: Path,
    status: str,
    reason: str | None,
    command: list[str] | None,
) -> dict[str, Any]:
    task_summary = breakdown.summarize_launch_tasks([], family="sglang-diffusion")
    return breakdown.make_case_record(
        preset=preset,
        gpu_count=run_config.num_gpus,
        requested_parallelism=asdict(run_config),
        resolved_parallelism=asdict(run_config),
        case_name=run_config.name,
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
    )


def build_csv_row(
    record: dict[str, Any],
    *,
    model_key: str,
    run_config: launch_time.RunConfig,
) -> dict[str, str]:
    row: dict[str, str] = {
        "row_name": f"gpu{run_config.num_gpus}_tp1_sp{run_config.sp_degree}",
        "status": str(record.get("status") or ""),
        "model": model_key,
        "gpu_num": str(run_config.num_gpus),
        "tp_size": str(run_config.tp_size),
        "sp_degree": str(run_config.sp_degree),
        "ulysses_degree": str(run_config.ulysses_degree),
        "ring_degree": str(run_config.ring_degree),
        "e2e_launch_time_s": csv_number(record.get("launch_time_s")),
    }
    for role, prefix in ROLE_TO_COLUMN_PREFIX.items():
        row[f"{prefix}_e2e_launch_time_s"] = csv_number(role_e2e_elapsed(record, role))
        row[f"{prefix}_weight_size_gb"] = csv_number(role_weight_size(record, role))
        row[f"{prefix}_weight_load_time_s"] = csv_number(
            role_component_elapsed(record, role, "component_load")
        )
        row[f"{prefix}_cpu_materialization_time_s"] = csv_number(
            role_component_elapsed(record, role, "component_cpu_materialization")
        )
    return row


def build_detail_row(
    record: dict[str, Any],
    *,
    model_key: str,
    run_config: launch_time.RunConfig,
) -> dict[str, str]:
    return {
        "row_name": f"gpu{run_config.num_gpus}_tp1_sp{run_config.sp_degree}",
        "status": str(record.get("status") or ""),
        "reason": str(record.get("reason") or ""),
        "model": model_key,
        "gpu_num": str(run_config.num_gpus),
        "tp_size": str(run_config.tp_size),
        "sp_degree": str(run_config.sp_degree),
        "ulysses_degree": str(run_config.ulysses_degree),
        "ring_degree": str(run_config.ring_degree),
        "shared_component_launch_time_s": csv_number(
            task_elapsed_s(record, "role_launch_total:shared")
        ),
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
    preset = get_preset(args.model, args.model_path)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (
        Path(args.csv_path).expanduser().resolve()
        if args.csv_path
        else output_dir / "diffusion_launch_time_summary.csv"
    )
    details_csv_path = output_dir / "diffusion_launch_time_details.csv"
    details_json_path = output_dir / "diffusion_launch_time_details.json"
    visible_gpu_count = launch_time.resolve_visible_gpu_count()

    records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, str]] = []
    detail_rows: list[dict[str, str]] = []

    for gpu_num in args.gpu_nums:
        run_config = resolve_diffusion_run_config(args.model, gpu_num)
        case_dir = output_dir / preset.output_subdir / f"gpu{gpu_num}" / run_config.name
        try:
            validate_run_config(run_config, gpu_num)
            command = build_command(
                preset=preset,
                run_config=run_config,
                host=args.host,
                case_dir=case_dir,
            )
        except Exception as exc:
            record = make_record(
                preset=preset,
                run_config=run_config,
                case_dir=case_dir,
                status="failed",
                reason=str(exc),
                command=None,
            )
        else:
            if args.dry_run:
                record = make_record(
                    preset=preset,
                    run_config=run_config,
                    case_dir=case_dir,
                    status="dry_run",
                    reason=None,
                    command=command,
                )
            elif visible_gpu_count < gpu_num:
                record = make_record(
                    preset=preset,
                    run_config=run_config,
                    case_dir=case_dir,
                    status="skipped",
                    reason=(
                        f"Requested gpu_num={gpu_num}, but only "
                        f"{visible_gpu_count} visible GPU(s) are available."
                    ),
                    command=command,
                )
            else:
                record = breakdown.measure_diffusion_case(
                    preset=preset,
                    gpu_count=gpu_num,
                    run_config=run_config,
                    host=args.host,
                    timeout_s=args.timeout_s,
                    case_dir=case_dir,
                )
                launch_command = record.get("launch_command")
                if isinstance(launch_command, list):
                    validate_command_parallelism(launch_command, gpu_num)

        records.append(record)
        breakdown.write_case_breakdown(case_dir, record)
        csv_rows.append(build_csv_row(record, model_key=args.model, run_config=run_config))
        detail_rows.append(
            build_detail_row(record, model_key=args.model, run_config=run_config)
        )
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

    launch_time.logger.info("Diffusion launch summary CSV written to %s", csv_path)


if __name__ == "__main__":
    main()
