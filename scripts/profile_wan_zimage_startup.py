#!/usr/bin/env python3
"""Profile fresh server startup time for Wan and Z-Image models."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from profile_diffusion_common import (
    ALL_MODEL_SPECS,
    NUM_RUNS,
    NUM_WARMUP_RUNS,
    WAN21,
    WAN22,
    Z_IMAGE,
    allocate_ports,
    build_server_command,
    build_server_environment,
    launch_server,
    measured_mean,
    model_metadata,
    normalize_attention_backend,
    normalize_gpu_counts,
    prepare_output_dir,
    save_json,
    server_base_url,
    stop_server,
    validate_cuda_compat_lib_dir,
    validate_model_path,
    wait_for_ready,
    write_python_summary,
)


SUMMARY_VARIABLES = {
    WAN22.key: "wan22_ti2v_5b_startup_time_ms",
    WAN21.key: "wan21_t2v_1_3b_startup_time_ms",
    Z_IMAGE.key: "z_image_startup_time_ms",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure fresh-process startup time for Wan2.2, Wan2.1, and Z-Image. "
            "Each point uses two warmups and three measured starts."
        )
    )
    parser.add_argument("--wan22-model-path", type=Path, default=WAN22.default_path)
    parser.add_argument("--wan21-model-path", type=Path, default=WAN21.default_path)
    parser.add_argument("--z-image-model-path", type=Path, default=Z_IMAGE.default_path)
    parser.add_argument("--gpu-counts", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cuda-compat-lib-dir", type=Path)
    parser.add_argument("--attention-backend")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/outputs/wan_zimage_startup"),
    )
    parser.add_argument("--server-timeout-s", type=float, default=3600.0)
    parser.add_argument("--ready-poll-interval-s", type=float, default=0.05)
    parser.add_argument("--shutdown-timeout-s", type=float, default=60.0)
    parser.add_argument("--cooldown-s", type=float, default=2.0)
    args = parser.parse_args()
    try:
        args.gpu_counts = normalize_gpu_counts(args.gpu_counts)
        args.attention_backend = normalize_attention_backend(args.attention_backend)
        args.cuda_compat_lib_dir = validate_cuda_compat_lib_dir(
            args.cuda_compat_lib_dir
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    for name in (
        "server_timeout_s",
        "ready_poll_interval_s",
        "shutdown_timeout_s",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.cooldown_s < 0:
        parser.error("--cooldown-s must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    requested_paths = {
        WAN22.key: args.wan22_model_path,
        WAN21.key: args.wan21_model_path,
        Z_IMAGE.key: args.z_image_model_path,
    }
    model_paths = {
        spec.key: validate_model_path(spec, requested_paths[spec.key])
        for spec in ALL_MODEL_SPECS
    }
    output_dir = prepare_output_dir(args.output_dir)
    state_path = output_dir / "results.json"
    state: dict[str, Any] = {
        "status": "running",
        "metric": "Popen to validated /v1/models readiness",
        "unit": "ms",
        "generation_stage_profiling": False,
        "runs_per_point": NUM_RUNS,
        "warmup_runs": NUM_WARMUP_RUNS,
        "gpu_counts": args.gpu_counts,
        "runtime_overrides": {
            "attention_backend": args.attention_backend,
            "cuda_compat_lib_dir": (
                str(args.cuda_compat_lib_dir)
                if args.cuda_compat_lib_dir is not None
                else None
            ),
        },
        "models": [
            model_metadata(spec, model_paths[spec.key]) for spec in ALL_MODEL_SPECS
        ],
        "points": [],
    }
    save_json(state_path, state)
    summaries: dict[str, list[list[float | int]]] = {
        spec.key: [] for spec in ALL_MODEL_SPECS
    }

    for spec in ALL_MODEL_SPECS:
        model_path = model_paths[spec.key]
        for gpu_count in args.gpu_counts:
            point: dict[str, Any] = {
                "model": spec.key,
                "gpu_count": gpu_count,
                "sp_degree": gpu_count,
                "tp_size": 1,
                "ulysses_degree": spec.parallelism[gpu_count][0],
                "ring_degree": spec.parallelism[gpu_count][1],
                "status": "running",
                "runs": [],
            }
            state["points"].append(point)
            save_json(state_path, state)

            for run_index in range(NUM_RUNS):
                trial_dir = (
                    output_dir
                    / spec.key
                    / f"gpu_{gpu_count}"
                    / f"run_{run_index + 1:02d}"
                )
                server_dir = trial_dir / "server"
                ports = allocate_ports(args.host)
                command = build_server_command(
                    spec,
                    model_path,
                    gpu_count,
                    args.host,
                    ports,
                    server_dir,
                    attention_backend=args.attention_backend,
                )
                record: dict[str, Any] = {
                    "run": run_index + 1,
                    "warmup": run_index < NUM_WARMUP_RUNS,
                    "ports": ports,
                    "command": command,
                    "status": "starting",
                }
                point["runs"].append(record)
                save_json(state_path, state)
                server = None
                print(
                    f"[startup] {spec.label} GPU={gpu_count} "
                    f"run={run_index + 1}/{NUM_RUNS}",
                    flush=True,
                )
                try:
                    server = launch_server(
                        command,
                        trial_dir / "server.log",
                        build_server_environment(
                            server_dir,
                            enable_cuda_event_stage_profiling=False,
                            cuda_compat_lib_dir=args.cuda_compat_lib_dir,
                        ),
                    )
                    base_url = server_base_url(args.host, ports["http"])
                    card, ready_ns = wait_for_ready(
                        server,
                        base_url,
                        spec,
                        model_path,
                        gpu_count,
                        args.server_timeout_s,
                        args.ready_poll_interval_s,
                    )
                    record["startup_time_ms"] = (
                        ready_ns - server.started_ns
                    ) / 1_000_000.0
                    record["ready_model_card"] = card
                    record["status"] = "complete"
                    save_json(state_path, state)
                except BaseException as exc:
                    record["status"] = "failed"
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    point["status"] = "failed"
                    state["status"] = "failed"
                    save_json(state_path, state)
                    raise
                finally:
                    stop_server(server, args.shutdown_timeout_s)
                    if args.cooldown_s:
                        time.sleep(args.cooldown_s)

            average_ms = measured_mean(point["runs"], "startup_time_ms")
            point["measured_average_ms"] = average_ms
            point["status"] = "complete"
            summaries[spec.key].append([gpu_count, average_ms])
            save_json(state_path, state)

    summary_values = [
        (SUMMARY_VARIABLES[spec.key], summaries[spec.key]) for spec in ALL_MODEL_SPECS
    ]
    summary_text = write_python_summary(output_dir / "summary.py", summary_values)
    state["status"] = "complete"
    state["summary"] = {
        SUMMARY_VARIABLES[spec.key]: summaries[spec.key] for spec in ALL_MODEL_SPECS
    }
    save_json(state_path, state)
    print("\n" + summary_text, end="", flush=True)


if __name__ == "__main__":
    main()
