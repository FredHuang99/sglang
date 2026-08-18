#!/usr/bin/env python3
"""Profile native Wan and Z-Image execution time with lightweight server metrics."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from profile_diffusion_common import (
    ALL_MODEL_SPECS,
    DEFAULT_REFERENCE_IMAGE,
    NUM_RUNS,
    NUM_WARMUP_RUNS,
    WAN21,
    WAN22,
    Z_IMAGE,
    allocate_ports,
    build_server_command,
    build_server_environment,
    launch_server,
    materialize_reference_image,
    measured_mean,
    module_durations_ms,
    model_metadata,
    normalize_gpu_counts,
    prepare_output_dir,
    read_perf_dump,
    repository_commit,
    response_request_id,
    save_json,
    send_generation_request,
    server_base_url,
    stop_server,
    total_duration_ms,
    validate_model_path,
    wait_for_ready,
    write_python_summary,
)


SUMMARY_VARIABLES = {
    WAN22.key: "wan22_ti2v_5b_execution_time_ms",
    WAN21.key: "wan21_t2v_1_3b_execution_time_ms",
    Z_IMAGE.key: "z_image_execution_time_ms",
}

MODULE_SUMMARY_VARIABLES = {
    spec.key: {
        module: f"{spec.key}_{module}_duration_ms"
        for module in ("encoder", "denoiser", "decoder")
    }
    for spec in ALL_MODEL_SPECS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure SGLang-native pipeline execution time for Wan2.2, Wan2.1, "
            "and Z-Image. Each point uses two warmups and three measured runs."
        )
    )
    parser.add_argument("--wan22-model-path", type=Path, default=WAN22.default_path)
    parser.add_argument("--wan21-model-path", type=Path, default=WAN21.default_path)
    parser.add_argument("--z-image-model-path", type=Path, default=Z_IMAGE.default_path)
    parser.add_argument("--gpu-counts", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--wan22-reference-image", default=DEFAULT_REFERENCE_IMAGE
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/outputs/wan_zimage_execution"),
    )
    parser.add_argument("--server-timeout-s", type=float, default=3600.0)
    parser.add_argument("--request-timeout-s", type=float, default=3600.0)
    parser.add_argument("--reference-download-timeout-s", type=float, default=300.0)
    parser.add_argument("--perf-timeout-s", type=float, default=30.0)
    parser.add_argument("--ready-poll-interval-s", type=float, default=0.1)
    parser.add_argument("--video-poll-interval-s", type=float, default=1.0)
    parser.add_argument("--shutdown-timeout-s", type=float, default=60.0)
    parser.add_argument("--cooldown-s", type=float, default=2.0)
    args = parser.parse_args()
    try:
        args.gpu_counts = normalize_gpu_counts(args.gpu_counts)
    except ValueError as exc:
        parser.error(str(exc))
    for name in (
        "server_timeout_s",
        "request_timeout_s",
        "reference_download_timeout_s",
        "perf_timeout_s",
        "ready_poll_interval_s",
        "video_poll_interval_s",
        "shutdown_timeout_s",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.cooldown_s < 0:
        parser.error("--cooldown-s must be non-negative")
    return args


def response_metadata(response: dict[str, Any]) -> dict[str, Any]:
    return {
        key: response.get(key)
        for key in (
            "id",
            "status",
            "file_path",
            "file_paths",
            "num_outputs",
            "inference_time_s",
        )
        if response.get(key) is not None
    }


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
    profile_commit = repository_commit()
    reference_image = materialize_reference_image(
        args.wan22_reference_image,
        output_dir,
        args.reference_download_timeout_s,
    )

    state_path = output_dir / "results.json"
    state: dict[str, Any] = {
        "status": "running",
        "metric": "perf_dump.total_duration_ms",
        "module_metric": "perf_dump.steps[stage].duration_ms",
        "module_timing_method": "cuda_event",
        "unit": "ms",
        "commit_hash": profile_commit,
        "runs_per_point": NUM_RUNS,
        "warmup_runs": NUM_WARMUP_RUNS,
        "gpu_counts": args.gpu_counts,
        "reference_image": str(reference_image),
        "models": [
            model_metadata(spec, model_paths[spec.key]) for spec in ALL_MODEL_SPECS
        ],
        "points": [],
    }
    save_json(state_path, state)
    summaries: dict[str, list[list[float | int]]] = {
        spec.key: [] for spec in ALL_MODEL_SPECS
    }
    module_summaries: dict[str, dict[str, list[list[float | int]]]] = {
        spec.key: {module: [] for module in ("encoder", "denoiser", "decoder")}
        for spec in ALL_MODEL_SPECS
    }

    for spec in ALL_MODEL_SPECS:
        model_path = model_paths[spec.key]
        for gpu_count in args.gpu_counts:
            point_dir = output_dir / spec.key / f"gpu_{gpu_count}"
            server_dir = point_dir / "server"
            perf_dir = point_dir / "perf"
            ports = allocate_ports(args.host)
            command = build_server_command(
                spec, model_path, gpu_count, args.host, ports, server_dir
            )
            point: dict[str, Any] = {
                "model": spec.key,
                "gpu_count": gpu_count,
                "sp_degree": gpu_count,
                "tp_size": 1,
                "ulysses_degree": spec.parallelism[gpu_count][0],
                "ring_degree": spec.parallelism[gpu_count][1],
                "ports": ports,
                "command": command,
                "status": "starting",
                "runs": [],
            }
            state["points"].append(point)
            save_json(state_path, state)
            server = None
            print(
                f"[server] launching {spec.label} on {gpu_count} GPU(s) "
                f"at {server_base_url(args.host, ports['http'])}",
                flush=True,
            )
            try:
                server = launch_server(
                    command,
                    point_dir / "server.log",
                    build_server_environment(
                        server_dir, enable_cuda_event_stage_profiling=True
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
                point["status"] = "running"
                point["ready_model_card"] = card
                point["ready_after_launch_ms"] = (
                    ready_ns - server.started_ns
                ) / 1_000_000.0
                save_json(state_path, state)

                for run_index in range(NUM_RUNS):
                    perf_path = perf_dir / f"run_{run_index + 1:02d}.json"
                    print(
                        f"[request] {spec.label} GPU={gpu_count} "
                        f"run={run_index + 1}/{NUM_RUNS}",
                        flush=True,
                    )
                    response = send_generation_request(
                        base_url,
                        spec,
                        spec.prompt,
                        perf_path,
                        reference_image if spec.reference_image else None,
                        args.request_timeout_s,
                        args.video_poll_interval_s,
                    )
                    request_id = response_request_id(response)
                    perf_dump = read_perf_dump(
                        perf_path,
                        args.perf_timeout_s,
                        expected_request_id=request_id,
                        expected_commit_hash=profile_commit,
                        expected_model_path=model_path,
                        expected_world_size=gpu_count,
                    )
                    module_times = module_durations_ms(perf_dump)
                    record = {
                        "run": run_index + 1,
                        "warmup": run_index < NUM_WARMUP_RUNS,
                        "total_duration_ms": total_duration_ms(perf_dump),
                        "module_duration_ms": module_times,
                        "perf_dump_path": str(perf_path),
                        "response": response_metadata(response),
                    }
                    point["runs"].append(record)
                    save_json(state_path, state)

                average_ms = measured_mean(point["runs"], "total_duration_ms")
                point["measured_average_ms"] = average_ms
                module_average_ms = {
                    module: measured_mean(
                        [
                            {
                                f"{module}_duration_ms": run["module_duration_ms"][
                                    module
                                ]
                            }
                            for run in point["runs"]
                        ],
                        f"{module}_duration_ms",
                    )
                    for module in ("encoder", "denoiser", "decoder")
                }
                point["measured_module_average_ms"] = module_average_ms
                point["status"] = "complete"
                summaries[spec.key].append([gpu_count, average_ms])
                for module, module_average in module_average_ms.items():
                    module_summaries[spec.key][module].append(
                        [gpu_count, module_average]
                    )
                save_json(state_path, state)
            except BaseException as exc:
                point["status"] = "failed"
                point["error"] = f"{type(exc).__name__}: {exc}"
                state["status"] = "failed"
                save_json(state_path, state)
                raise
            finally:
                stop_server(server, args.shutdown_timeout_s)
                if args.cooldown_s:
                    time.sleep(args.cooldown_s)

    summary_values = [
        (SUMMARY_VARIABLES[spec.key], summaries[spec.key]) for spec in ALL_MODEL_SPECS
    ]
    summary_text = write_python_summary(output_dir / "summary.py", summary_values)
    module_summary_values = [
        (
            MODULE_SUMMARY_VARIABLES[spec.key][module],
            module_summaries[spec.key][module],
        )
        for spec in ALL_MODEL_SPECS
        for module in ("encoder", "denoiser", "decoder")
    ]
    module_summary_text = write_python_summary(
        output_dir / "module_summary.py", module_summary_values
    )
    state["status"] = "complete"
    state["summary"] = {
        SUMMARY_VARIABLES[spec.key]: summaries[spec.key] for spec in ALL_MODEL_SPECS
    }
    state["module_summary"] = {
        MODULE_SUMMARY_VARIABLES[spec.key][module]: module_summaries[spec.key][module]
        for spec in ALL_MODEL_SPECS
        for module in ("encoder", "denoiser", "decoder")
    }
    save_json(state_path, state)
    print("\n" + summary_text + "\n" + module_summary_text, end="", flush=True)


if __name__ == "__main__":
    main()
