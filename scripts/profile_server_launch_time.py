#!/usr/bin/env python3
"""Measure server launch time for sglang and sglang-diffusion presets."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
import shutil
import signal
import site
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

try:
    import requests
except ModuleNotFoundError:
    requests = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_PATH = Path(__file__).resolve()
LEGACY_SCRIPT_PATH = SCRIPT_PATH.with_name("profile_wan22_ti2v_5b_monolithic.py")
WAIT_LOG_INTERVAL_S = 15


@lru_cache(maxsize=1)
def get_diffusion_legacy():
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


@dataclass(frozen=True)
class RunConfig:
    name: str
    mode: str
    num_gpus: int
    tp_size: int | None
    sp_degree: int | None
    ulysses_degree: int | None
    ring_degree: int | None


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


@dataclass(frozen=True)
class LLMSetup:
    key: str
    description: str
    chunked_prefill_size: int | None = None
    max_running_requests: int | None = None
    max_total_tokens: int | None = None
    cuda_graph_max_bs: int | None = None


def make_wan22_ti2v_sampling() -> Any:
    from sglang.multimodal_gen.configs.sample.wan import Wan2_2_TI2V_5B_SamplingParam

    return Wan2_2_TI2V_5B_SamplingParam()


def make_want2v_13b_sampling() -> Any:
    from sglang.multimodal_gen.configs.sample.wan import WanT2V_1_3B_SamplingParams

    return WanT2V_1_3B_SamplingParams()


def make_zimage_sampling() -> Any:
    from sglang.multimodal_gen.configs.sample.zimage import ZImageSamplingParams

    return ZImageSamplingParams(width=1024, height=1024)


LLM_SETUPS: dict[str, LLMSetup] = {
    "default": LLMSetup(
        key="default",
        description="Preset default prompt-enhancer launch parameters.",
    ),
    "cp512_req1_cg1": LLMSetup(
        key="cp512_req1_cg1",
        description=(
            "Use chunked prefill size 512, max_running_requests 1, "
            "and cuda_graph_max_bs 1."
        ),
        chunked_prefill_size=512,
        max_running_requests=1,
        cuda_graph_max_bs=1,
    ),
    "constrained_4096": LLMSetup(
        key="constrained_4096",
        description=(
            "Use max_running_requests 1, max_total_tokens 4096, "
            "chunked_prefill_size 4096, and cuda_graph_max_bs 1."
        ),
        chunked_prefill_size=4096,
        max_running_requests=1,
        max_total_tokens=4096,
        cuda_graph_max_bs=1,
    ),
}


DIFFUSION_REGISTRY_HF_PATHS: dict[str, str] = {
    "wan2.2-ti2v-5b": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "wan2.1-t2v-1.3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "z-image": "Tongyi-MAI/Z-Image",
}

DIFFUSION_REGISTRY_MODEL_IDS: dict[str, str] = {
    key: value.rsplit("/", 1)[-1]
    for key, value in DIFFUSION_REGISTRY_HF_PATHS.items()
}


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
        model_id=DIFFUSION_REGISTRY_MODEL_IDS["wan2.2-ti2v-5b"],
        output_subdir="wan2_2_ti2v_5b/launch",
        expected_task_type="TI2V",
        sampling_factory=make_wan22_ti2v_sampling,
    ),
    "wan2.1-t2v-1.3b": LaunchPreset(
        key="wan2.1-t2v-1.3b",
        family="diffusion",
        model_path="/workspace/models/Wan2_1_T2V_1_3B",
        model_id=DIFFUSION_REGISTRY_MODEL_IDS["wan2.1-t2v-1.3b"],
        output_subdir="wan2_1_t2v_1_3b/launch",
        expected_task_type="T2V",
        sampling_factory=make_want2v_13b_sampling,
    ),
    "z-image": LaunchPreset(
        key="z-image",
        family="diffusion",
        model_path="/workspace/models/Z_Image",
        model_id=DIFFUSION_REGISTRY_MODEL_IDS["z-image"],
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


def parse_llm_setup_keys(raw: str | None) -> list[str]:
    if raw is None:
        return ["default"]
    keys = [token.strip() for token in raw.replace(";", ",").split(",") if token.strip()]
    if not keys:
        return ["default"]
    invalid = [key for key in keys if key not in LLM_SETUPS]
    if invalid:
        raise ValueError(
            f"Unsupported llm setup(s): {invalid}. Supported setups: {sorted(LLM_SETUPS)}"
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def parse_parallel_degrees(args: argparse.Namespace) -> list[int]:
    cleaned = (
        args.parallel_degrees.replace("{", "")
        .replace("}", "")
        .replace("[", "")
        .replace("]", "")
    )
    tokens = [token for token in re.split(r"[\s,]+", cleaned.strip()) if token]
    if not tokens:
        raise ValueError("No valid values were found in --parallel-degrees.")

    degrees: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        value = int(token)
        if value <= 0:
            raise ValueError(
                f"parallel degree must be a positive integer, got {value}."
            )
        if value not in seen:
            seen.add(value)
            degrees.append(value)
    return degrees


def resolve_visible_gpu_count() -> int:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        devices = [d.strip() for d in cuda_visible_devices.split(",") if d.strip()]
        return len(devices)
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def detect_cutlass_python_packages_dir() -> Path | None:
    candidates: list[Path] = []
    try:
        candidates.extend(Path(path) for path in site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            candidates.append(Path(user_site))
    except Exception:
        pass

    seen: set[Path] = set()
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)
        python_packages_dir = base / "nvidia_cutlass_dsl" / "python_packages"
        if (python_packages_dir / "cutlass").exists():
            return python_packages_dir.resolve()
    return None


def prepend_pythonpath(env: dict[str, str], path: Path) -> None:
    existing = env.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    path_str = str(path)
    if path_str not in parts:
        parts.insert(0, path_str)
    env["PYTHONPATH"] = os.pathsep.join(parts)


def safe_rmtree(path: Path) -> None:
    logger.info("Cleaning temporary artifacts: %s", path)
    shutil.rmtree(path, ignore_errors=True)


def factor_pairs(n: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(1, n + 1):
        if n % i == 0:
            pairs.append((i, n // i))
    return pairs


def build_run_configs(parallel_degree: int) -> list[RunConfig]:
    runs: list[RunConfig] = []
    seen: set[tuple[int, int, int, int]] = set()

    def add_run(run: RunConfig) -> None:
        key = (
            run.tp_size or 1,
            run.sp_degree or 1,
            run.ulysses_degree or 1,
            run.ring_degree or 1,
        )
        if key in seen:
            return
        seen.add(key)
        runs.append(run)

    divisors_desc = sorted(
        [d for d in range(1, parallel_degree + 1) if parallel_degree % d == 0],
        reverse=True,
    )
    for tp_size in divisors_desc:
        sp_degree = parallel_degree // tp_size
        if sp_degree == 1:
            add_run(
                RunConfig(
                    name=f"e_tp{tp_size}_sp{sp_degree}",
                    mode="explicit",
                    num_gpus=parallel_degree,
                    tp_size=tp_size,
                    sp_degree=sp_degree,
                    ulysses_degree=None,
                    ring_degree=None,
                )
            )
            continue
        for ulysses_degree, ring_degree in factor_pairs(sp_degree):
            add_run(
                RunConfig(
                    name=f"e_tp{tp_size}_sp{sp_degree}_u{ulysses_degree}_r{ring_degree}",
                    mode="explicit",
                    num_gpus=parallel_degree,
                    tp_size=tp_size,
                    sp_degree=sp_degree,
                    ulysses_degree=ulysses_degree,
                    ring_degree=ring_degree,
                )
            )
    return runs


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


def start_logged_process(
    *,
    command: list[str],
    log_path: Path,
    env: dict[str, str] | None = None,
) -> tuple[subprocess.Popen[str], Any]:
    log_fh = log_path.open("w", encoding="utf-8")
    popen_kwargs: dict[str, Any] = {
        "stdout": log_fh,
        "stderr": subprocess.STDOUT,
        "text": True,
        "env": env,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        popen_kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(command, **popen_kwargs)
    except Exception:
        log_fh.close()
        raise
    process._sgl_log_fh = log_fh  # type: ignore[attr-defined]
    return process, log_fh


def stop_server(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return

    try:
        if process.poll() is None:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
    except Exception:
        pass

    try:
        if process.poll() is None:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    except Exception:
        pass

    log_fh = getattr(process, "_sgl_log_fh", None)
    if log_fh is not None:
        try:
            log_fh.flush()
            log_fh.close()
        except Exception:
            pass


def build_promptenhancer_command(
    *,
    preset: LaunchPreset,
    tp_size: int,
    host: str,
    port: int,
    llm_setup: LLMSetup | None = None,
) -> list[str]:
    effective_chunked_prefill_size = (
        llm_setup.chunked_prefill_size
        if llm_setup is not None and llm_setup.chunked_prefill_size is not None
        else preset.chunked_prefill_size
    )
    effective_max_running_requests = (
        llm_setup.max_running_requests
        if llm_setup is not None and llm_setup.max_running_requests is not None
        else None
    )
    effective_max_total_tokens = (
        llm_setup.max_total_tokens
        if llm_setup is not None and llm_setup.max_total_tokens is not None
        else None
    )
    effective_cuda_graph_max_bs = (
        llm_setup.cuda_graph_max_bs
        if llm_setup is not None and llm_setup.cuda_graph_max_bs is not None
        else preset.cuda_graph_max_bs
    )

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
    if preset.model_id:
        command.extend(["--served-model-name", preset.model_id])
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
    if effective_chunked_prefill_size is not None:
        command.extend(["--chunked-prefill-size", str(effective_chunked_prefill_size)])
    if effective_max_running_requests is not None:
        command.extend(["--max-running-requests", str(effective_max_running_requests)])
    if effective_max_total_tokens is not None:
        command.extend(["--max-total-tokens", str(effective_max_total_tokens)])
    if effective_cuda_graph_max_bs is not None:
        command.extend(["--cuda-graph-max-bs", str(effective_cuda_graph_max_bs)])
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
        "--use-fsdp-inference",
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
    if requests is None:
        raise RuntimeError(
            "The requests package is required to wait for server readiness."
        )
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
        if elapsed - last_log_at >= WAIT_LOG_INTERVAL_S:
            logger.info(
                "Waiting for prompt-enhancer server readiness... elapsed=%ss ready=False",
                int(elapsed),
            )
            last_log_at = elapsed
        time.sleep(1)


def cleanup_case_dir(case_dir: Path, keep_artifacts: bool) -> None:
    if keep_artifacts:
        return
    safe_rmtree(case_dir)


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
    port = find_free_port(host)
    base_url = f"http://{host}:{port}"
    command = build_promptenhancer_command(
        preset=preset,
        tp_size=gpu_count,
        host=host,
        port=port,
    )

    process: subprocess.Popen[str] | None = None
    start_ns = time.perf_counter_ns()
    try:
        logger.info("Launching %s", " ".join(command))
        process, _ = start_logged_process(command=command, log_path=server_log_path)
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
        stop_server(process)


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
    diffusion_legacy = get_diffusion_legacy()
    diffusion_legacy.EXPECTED_TASK_TYPE = preset.expected_task_type
    start_ns = time.perf_counter_ns()
    command_preview = build_diffusion_command_preview(
        preset=preset,
        run_config=run_config,
        sampling=sampling,
        host=host,
    )
    try:
        process, base_url, perf_dir, init_profile_path, model_card = diffusion_legacy.launch_server(
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
        diffusion_legacy.stop_server(process)


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
        "parallel_degrees": parse_parallel_degrees(args),
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
    parallel_degrees = parse_parallel_degrees(args)
    visible_gpu_count = resolve_visible_gpu_count()
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

            run_configs = build_run_configs(gpu_count)
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
