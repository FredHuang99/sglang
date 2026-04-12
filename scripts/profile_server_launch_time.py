#!/usr/bin/env python3
"""Measure server launch time for sglang and sglang-diffusion presets."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from sglang.multimodal_gen.configs.sample.wan import (
    Wan2_2_TI2V_5B_SamplingParam,
    WanT2V_1_3B_SamplingParams,
)
from sglang.multimodal_gen.configs.sample.zimage import ZImageSamplingParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

SCRIPT_PATH = Path(__file__).resolve()
LEGACY_SCRIPT_PATH = SCRIPT_PATH.with_name("profile_wan22_ti2v_5b_monolithic.py")


def load_legacy_module():
    spec = importlib.util.spec_from_file_location(
        "_sglang_legacy_monolithic_profile_for_launch_time",
        LEGACY_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load legacy profiler at {LEGACY_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_legacy_module()
RunConfig = legacy.RunConfig


@dataclass(frozen=True)
class LaunchPreset:
    key: str
    family: str
    model_path: str
    model_id: str
    output_subdir: str
    trust_remote_code: bool = False
    expected_task_type: str | None = None
    sampling_factory: Callable[[], Any] | None = None
    context_length: int | None = None
    mem_fraction_static: float | None = None
    chunked_prefill_size: int | None = None
    cuda_graph_max_bs: int | None = None
    skip_server_warmup: bool = False
    disable_piecewise_cuda_graph: bool = False


def make_zimage_sampling() -> ZImageSamplingParams:
    return ZImageSamplingParams(width=1024, height=1024)


PRESETS: dict[str, LaunchPreset] = {
    "promptenhancer-7b": LaunchPreset(
        key="promptenhancer-7b",
        family="sglang",
        model_path="/workspace/models/Hunyuan_PromptEnhancer_7B",
        model_id="Hunyuan_PromptEnhancer_7B",
        output_subdir="pe7b/launch",
        trust_remote_code=True,
        context_length=32768,
        mem_fraction_static=0.90,
        skip_server_warmup=True,
        disable_piecewise_cuda_graph=True,
    ),
    "promptenhancer-32b": LaunchPreset(
        key="promptenhancer-32b",
        family="sglang",
        model_path="/workspace/models/Hunyuan_PromptEnhancer_32B",
        model_id="Hunyuan_PromptEnhancer_32B",
        output_subdir="pe32b/launch",
        trust_remote_code=True,
        context_length=32768,
        mem_fraction_static=0.90,
        skip_server_warmup=True,
        disable_piecewise_cuda_graph=True,
    ),
    "wan2.2-ti2v-5b": LaunchPreset(
        key="wan2.2-ti2v-5b",
        family="diffusion",
        model_path="/workspace/models/Wan2_2_TI2V_5B",
        model_id="Wan2.2-TI2V-5B-Diffusers",
        output_subdir="wan2_2_ti2v_5b/launch",
        expected_task_type="TI2V",
        sampling_factory=Wan2_2_TI2V_5B_SamplingParam,
    ),
    "wan2.1-t2v-1.3b": LaunchPreset(
        key="wan2.1-t2v-1.3b",
        family="diffusion",
        model_path="/workspace/models/Wan2_1_T2V_1_3B",
        model_id="Wan2.1-T2V-1.3B-Diffusers",
        output_subdir="wan2_1_t2v_1_3b/launch",
        expected_task_type="T2V",
        sampling_factory=WanT2V_1_3B_SamplingParams,
    ),
    "z-image": LaunchPreset(
        key="z-image",
        family="diffusion",
        model_path="/workspace/models/Z_Image",
        model_id="Z-Image",
        output_subdir="z_image/launch",
        expected_task_type="T2I",
        sampling_factory=make_zimage_sampling,
    ),
}


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure launch time for sglang and sglang-diffusion server presets."
        )
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(PRESETS.keys()),
        help=(
            "Comma-separated preset list. Defaults to all supported presets: "
            + ",".join(PRESETS.keys())
        ),
    )
    parser.add_argument(
        "--parallel-degrees",
        type=str,
        default="8,4,2,1",
        help="Comma/space separated GPU counts, e.g. '8,4,2,1'.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind servers to.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=1800,
        help="Timeout in seconds for each launch attempt.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root. Defaults to /workspace/outputs/server_launch_time/<run_id>.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep per-case logs and temporary launch directories.",
    )
    return parser.parse_args()


def parse_model_keys(raw: str) -> list[str]:
    keys = [token.strip() for token in raw.replace(";", ",").split(",") if token.strip()]
    if not keys:
        raise ValueError("No valid presets were found in --models.")
    invalid = [key for key in keys if key not in PRESETS]
    if invalid:
        raise ValueError(
            f"Unsupported preset(s): {invalid}. Supported presets: {sorted(PRESETS)}"
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    return Path(f"/workspace/outputs/server_launch_time/{now_stamp()}").resolve()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_log_tail(log_path: Path, lines: int = 120) -> str:
    if not log_path.exists():
        return ""
    content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def build_promptenhancer_command(
    *,
    preset: LaunchPreset,
    tp_size: int,
    host: str,
    port: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        preset.model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--tp-size",
        str(tp_size),
    ]
    if preset.trust_remote_code:
        command.append("--trust-remote-code")
    if preset.context_length is not None:
        command.extend(["--context-length", str(preset.context_length)])
    if preset.mem_fraction_static is not None:
        command.extend(["--mem-fraction-static", str(preset.mem_fraction_static)])
    if preset.skip_server_warmup:
        command.append("--skip-server-warmup")
    if preset.disable_piecewise_cuda_graph:
        command.append("--disable-piecewise-cuda-graph")
    if preset.chunked_prefill_size is not None:
        command.extend(["--chunked-prefill-size", str(preset.chunked_prefill_size)])
    if preset.cuda_graph_max_bs is not None:
        command.extend(["--cuda-graph-max-bs", str(preset.cuda_graph_max_bs)])
    return command


def build_diffusion_command_preview(
    *,
    preset: LaunchPreset,
    run_config: RunConfig,
    sampling: Any,
    host: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.runtime.entrypoints.cli.main",
        "serve",
        "--model-path",
        preset.model_path,
        "--host",
        host,
        "--port",
        "<dynamic-port>",
        "--scheduler-port",
        "<dynamic-port>",
        "--master-port",
        "<dynamic-port>",
        "--num-gpus",
        str(run_config.num_gpus),
        "--warmup",
        "--warmup-resolutions",
        f"{sampling.width}x{sampling.height}",
        "--warmup-steps",
        "1",
        "--output-path",
        "<case-dir>/outputs",
        "--input-save-path",
        "<case-dir>/uploads",
        "--log-level",
        "info",
        "--dit-cpu-offload",
        "false",
        "--dit-layerwise-offload",
        "false",
        "--text-encoder-cpu-offload",
        "false",
        "--image-encoder-cpu-offload",
        "false",
        "--vae-cpu-offload",
        "false",
        "--pin-cpu-memory",
        "false",
    ]
    if preset.model_id:
        command.extend(["--model-id", preset.model_id])
    if run_config.tp_size is not None:
        command.extend(["--tp-size", str(run_config.tp_size)])
    if run_config.sp_degree is not None:
        command.extend(["--sp-degree", str(run_config.sp_degree)])
    if run_config.ulysses_degree is not None:
        command.extend(["--ulysses-degree", str(run_config.ulysses_degree)])
    if run_config.ring_degree is not None:
        command.extend(["--ring-degree", str(run_config.ring_degree)])
    if preset.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def wait_for_promptenhancer_ready(
    *,
    process: subprocess.Popen[str],
    base_url: str,
    server_log_path: Path,
    timeout_s: int,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    last_log_at = -1.0
    while True:
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=2)
            if response.ok:
                return response.json()
        except Exception:
            pass

        if process.poll() is not None:
            raise RuntimeError(
                f"Server exited early with code {process.returncode}.\n"
                f"{read_log_tail(server_log_path)}"
            )

        elapsed = time.perf_counter() - start_time
        if elapsed >= timeout_s:
            raise TimeoutError(
                f"Timed out waiting for {base_url}/v1/models after {timeout_s}s.\n"
                f"{read_log_tail(server_log_path)}"
            )
        if elapsed - last_log_at >= legacy.WAIT_LOG_INTERVAL_S:
            logger.info(
                "Waiting for prompt-enhancer server readiness... elapsed=%ss ready=False",
                int(elapsed),
            )
            last_log_at = elapsed
        time.sleep(1)


def cleanup_case_dir(case_dir: Path, keep_artifacts: bool) -> None:
    if keep_artifacts:
        return
    legacy.safe_rmtree(case_dir)


def ns_to_ms(value_ns: int) -> float:
    return round(value_ns / 1_000_000.0, 3)


def ns_to_s(value_ns: int) -> float:
    return round(value_ns / 1_000_000_000.0, 3)


def make_case_record(
    *,
    preset: LaunchPreset,
    gpu_count: int,
    requested_parallelism: dict[str, Any],
    case_name: str,
    case_dir: Path,
    launch_command: list[str] | None,
    launch_time_ns: int | None,
    status: str,
    reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "model_key": preset.key,
        "family": preset.family,
        "model_path": preset.model_path,
        "model_id": preset.model_id,
        "gpu_count": gpu_count,
        "case_name": case_name,
        "requested_parallelism": requested_parallelism,
        "launch_time_ms": ns_to_ms(launch_time_ns) if launch_time_ns is not None else None,
        "launch_time_s": ns_to_s(launch_time_ns) if launch_time_ns is not None else None,
        "status": status,
        "reason": reason,
        "case_dir": str(case_dir),
        "server_log_path": str(case_dir / "server.log"),
        "launch_command": launch_command,
    }
    if extra:
        record.update(extra)
    return record


def measure_promptenhancer_case(
    *,
    preset: LaunchPreset,
    gpu_count: int,
    host: str,
    timeout_s: int,
    case_dir: Path,
) -> dict[str, Any]:
    server_log_path = case_dir / "server.log"
    case_dir.mkdir(parents=True, exist_ok=True)
    port = legacy.find_free_port(host)
    base_url = f"http://{host}:{port}"
    command = build_promptenhancer_command(
        preset=preset,
        tp_size=gpu_count,
        host=host,
        port=port,
    )

    process: subprocess.Popen[str] | None = None
    log_fh = None
    start_ns = time.perf_counter_ns()
    try:
        logger.info("Launching %s", " ".join(command))
        log_fh = server_log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        process._sgl_log_fh = log_fh  # type: ignore[attr-defined]
        model_card = wait_for_promptenhancer_ready(
            process=process,
            base_url=base_url,
            server_log_path=server_log_path,
            timeout_s=timeout_s,
        )
        launch_time_ns = time.perf_counter_ns() - start_ns
        return make_case_record(
            preset=preset,
            gpu_count=gpu_count,
            requested_parallelism={"tp_size": gpu_count},
            case_name=f"tp{gpu_count}",
            case_dir=case_dir,
            launch_command=command,
            launch_time_ns=launch_time_ns,
            status="completed",
            extra={
                "ready_url": f"{base_url}/v1/models",
                "served_model_card": model_card,
            },
        )
    except Exception as exc:
        return make_case_record(
            preset=preset,
            gpu_count=gpu_count,
            requested_parallelism={"tp_size": gpu_count},
            case_name=f"tp{gpu_count}",
            case_dir=case_dir,
            launch_command=command,
            launch_time_ns=None,
            status="failed",
            reason=str(exc),
        )
    finally:
        legacy.stop_server(process)
        if process is None and log_fh is not None:
            try:
                log_fh.flush()
                log_fh.close()
            except Exception:
                pass


def measure_diffusion_case(
    *,
    preset: LaunchPreset,
    gpu_count: int,
    run_config: RunConfig,
    host: str,
    timeout_s: int,
    case_dir: Path,
) -> dict[str, Any]:
    if preset.sampling_factory is None or preset.expected_task_type is None:
        raise ValueError(f"Preset {preset.key} is missing diffusion metadata.")

    sampling = preset.sampling_factory()
    launch_time_ns: int | None = None
    process = None
    legacy.EXPECTED_TASK_TYPE = preset.expected_task_type
    start_ns = time.perf_counter_ns()
    command_preview = build_diffusion_command_preview(
        preset=preset,
        run_config=run_config,
        sampling=sampling,
        host=host,
    )
    try:
        process, base_url, perf_dir, init_profile_path, model_card = legacy.launch_server(
            model_path=preset.model_path,
            model_id=preset.model_id,
            run_config=run_config,
            sampling=sampling,
            run_dir=case_dir,
            host=host,
            timeout_s=timeout_s,
            trust_remote_code=preset.trust_remote_code,
            enable_text_encoder_offload=False,
        )
        launch_time_ns = time.perf_counter_ns() - start_ns
        return make_case_record(
            preset=preset,
            gpu_count=gpu_count,
            requested_parallelism=asdict(run_config),
            case_name=run_config.name,
            case_dir=case_dir,
            launch_command=command_preview,
            launch_time_ns=launch_time_ns,
            status="completed",
            extra={
                "ready_url": f"{base_url}/v1/models",
                "perf_dir": str(perf_dir),
                "init_profile_path": str(init_profile_path),
                "served_model_card": model_card,
            },
        )
    except Exception as exc:
        return make_case_record(
            preset=preset,
            gpu_count=gpu_count,
            requested_parallelism=asdict(run_config),
            case_name=run_config.name,
            case_dir=case_dir,
            launch_command=command_preview,
            launch_time_ns=None,
            status="failed",
            reason=str(exc),
        )
    finally:
        legacy.stop_server(process)


def refresh_human_summary(summary: dict[str, Any]) -> None:
    grouped: dict[str, dict[str, Any]] = {}
    for record in summary.get("cases", []):
        model_key = record["model_key"]
        model_summary = grouped.setdefault(
            model_key,
            {
                "family": record["family"],
                "completed": 0,
                "failed": 0,
                "skipped": 0,
                "completed_cases": [],
            },
        )
        status = record.get("status")
        if status == "completed":
            model_summary["completed"] += 1
            model_summary["completed_cases"].append(
                {
                    "case_name": record["case_name"],
                    "gpu_count": record["gpu_count"],
                    "launch_time_ms": record["launch_time_ms"],
                    "launch_time_s": record["launch_time_s"],
                }
            )
        elif status == "skipped":
            model_summary["skipped"] += 1
        else:
            model_summary["failed"] += 1
    summary["human_summary"] = {
        "models": grouped,
        "total_cases": len(summary.get("cases", [])),
    }


def build_summary(*, args: argparse.Namespace, output_dir: Path, visible_gpu_count: int) -> dict[str, Any]:
    selected_models = parse_model_keys(args.models)
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_root": str(output_dir),
        "visible_gpu_count": visible_gpu_count,
        "parallel_degrees": legacy.parse_parallel_degrees(args),
        "selected_models": selected_models,
        "keep_artifacts": args.keep_artifacts,
        "launch_time_semantics": {
            "unit_ms": "Measured with time.perf_counter_ns(); reported in milliseconds with 3 decimal places.",
            "unit_s": "Same duration also reported in seconds with 3 decimal places.",
            "sglang": "From launch start until /v1/models first returns HTTP 200.",
            "sglang_diffusion": "From launch start until /health is ready, init_profile.json is dumped, and /v1/models reports the expected task_type.",
        },
        "cases": [],
        "human_summary": {},
    }


def main() -> None:
    args = parse_args()
    model_keys = parse_model_keys(args.models)
    parallel_degrees = legacy.parse_parallel_degrees(args)
    visible_gpu_count = legacy.resolve_visible_gpu_count()
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "server_launch_time_summary.json"
    summary = build_summary(
        args=args,
        output_dir=output_dir,
        visible_gpu_count=visible_gpu_count,
    )

    for model_key in model_keys:
        preset = PRESETS[model_key]
        logger.info("Starting launch-time sweep for preset=%s", model_key)
        for gpu_count in parallel_degrees:
            if visible_gpu_count < gpu_count:
                record = make_case_record(
                    preset=preset,
                    gpu_count=gpu_count,
                    requested_parallelism={"num_gpus": gpu_count},
                    case_name=f"gpu{gpu_count}",
                    case_dir=output_dir / preset.output_subdir / f"gpu{gpu_count}",
                    launch_command=None,
                    launch_time_ns=None,
                    status="skipped",
                    reason=(
                        f"Requested gpu_count={gpu_count}, but only {visible_gpu_count} "
                        f"visible GPU(s) are available."
                    ),
                )
                summary["cases"].append(record)
                refresh_human_summary(summary)
                save_json(summary_path, summary)
                continue

            if preset.family == "sglang":
                case_dir = output_dir / preset.output_subdir / f"tp{gpu_count}"
                logger.info(
                    "Measuring preset=%s family=sglang tp=%s",
                    model_key,
                    gpu_count,
                )
                record = measure_promptenhancer_case(
                    preset=preset,
                    gpu_count=gpu_count,
                    host=args.host,
                    timeout_s=args.wait_timeout,
                    case_dir=case_dir,
                )
                summary["cases"].append(record)
                refresh_human_summary(summary)
                save_json(summary_path, summary)
                cleanup_case_dir(case_dir, args.keep_artifacts)
                continue

            run_configs = legacy.build_run_configs(gpu_count)
            logger.info(
                "preset=%s gpu=%s will measure %s explicit tp*sp=P launch case(s)",
                model_key,
                gpu_count,
                len(run_configs),
            )
            total_configs = len(run_configs)
            for config_idx, run_config in enumerate(run_configs, start=1):
                case_dir = output_dir / preset.output_subdir / f"gpu{gpu_count}" / run_config.name
                logger.info(
                    "Measuring preset=%s gpu=%s config=%s (%s/%s)",
                    model_key,
                    gpu_count,
                    run_config.name,
                    config_idx,
                    total_configs,
                )
                record = measure_diffusion_case(
                    preset=preset,
                    gpu_count=gpu_count,
                    run_config=run_config,
                    host=args.host,
                    timeout_s=args.wait_timeout,
                    case_dir=case_dir,
                )
                summary["cases"].append(record)
                refresh_human_summary(summary)
                save_json(summary_path, summary)
                cleanup_case_dir(case_dir, args.keep_artifacts)

    refresh_human_summary(summary)
    save_json(summary_path, summary)
    logger.info("Launch-time summary written to %s", summary_path)


if __name__ == "__main__":
    main()
