"""Profile SGLang Diffusion pipeline stage durations.

This helper launches one isolated ``sglang generate`` process per
``(gpu_num, resolution)`` case and extracts the lightweight stage timings from
``--perf-dump-path``. It intentionally avoids torch profiler, forced CUDA
sync, and memory capture knobs so the measured request path stays low-intrusion.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ALL_RESOLUTIONS = ("144p", "240p", "360p", "480p", "720p", "1k", "2k")
DEFAULT_GPU_NUMS = (1, 2, 4, 8)
DEFAULT_NUM_RUNS = 5
DEFAULT_NUM_WARMUP_RUNS = 2
WAN22_DEFAULT_IMAGE_RELATIVE_PATH = Path("examples") / "assets" / "example_image.png"

CANONICAL_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "144p": (256, 144),
    "240p": (432, 240),
    "360p": (640, 360),
    "480p": (832, 480),
    "720p": (1280, 720),
    "1k": (1024, 576),
    "2k": (2048, 1152),
}

STAGE_NAME_TO_COLUMN = {
    "TextEncodingStage": "text_encoder",
    "DenoisingStage": "denoising",
    "DecodingStage": "decoder",
}

CSV_COLUMNS = ("row_name", "text_encoder", "denoising", "decoder")
STAGE_COLUMNS = ("text_encoder", "denoising", "decoder")
DETAIL_COLUMNS = (
    "row_name",
    "model",
    "model_path",
    "gpu_num",
    "resolution",
    "width",
    "height",
    "ulysses_degree",
    "ring_degree",
    "status",
    "returncode",
    "elapsed_s",
    "num_runs",
    "num_warmup_runs",
    "measured_runs",
    "successful_measured_runs",
    "run_statuses",
    "text_encoder",
    "denoising",
    "decoder",
    "perf_path",
    "case_dir",
    "error_tail",
    "command",
)


@dataclasses.dataclass(frozen=True)
class ModelPreset:
    name: str
    hf_path: str
    model_id: str
    size_multiple: int
    attention_heads: int
    default_prompt: str
    default_image_paths: tuple[str, ...] = ()


MODEL_PRESETS: dict[str, ModelPreset] = {
    "wan2.2-ti2v-5b": ModelPreset(
        name="wan2.2-ti2v-5b",
        hf_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        model_id="Wan2.2-TI2V-5B-Diffusers",
        size_multiple=32,
        attention_heads=40,
        default_prompt=(
            "A cinematic landscape video with soft morning light and clear details"
        ),
        default_image_paths=(str(WAN22_DEFAULT_IMAGE_RELATIVE_PATH),),
    ),
    "wan2.1-t2v-1.3b": ModelPreset(
        name="wan2.1-t2v-1.3b",
        hf_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        model_id="Wan2.1-T2V-1.3B-Diffusers",
        size_multiple=16,
        attention_heads=40,
        default_prompt=(
            "A cinematic landscape video with soft morning light and clear details"
        ),
    ),
    "z-image": ModelPreset(
        name="z-image",
        hf_path="Tongyi-MAI/Z-Image",
        model_id="Z-Image",
        size_multiple=16,
        attention_heads=30,
        default_prompt="A detailed studio photograph of a futuristic glass sculpture",
    ),
}


@dataclasses.dataclass(frozen=True)
class ParallelConfig:
    sp_degree: int
    ulysses_degree: int
    ring_degree: int
    valid: bool = True
    reason: str = "auto"
    error: str = ""

    @property
    def is_fallback(self) -> bool:
        return self.reason == "head_conflict_fallback"


@dataclasses.dataclass
class RunOutput:
    returncode: int | None
    elapsed_s: float
    timed_out: bool
    stdout_tail: str
    stderr_tail: str


@dataclasses.dataclass
class CaseResult:
    row_name: str
    model: str
    model_path: str
    gpu_num: int
    resolution: str
    width: int
    height: int
    ulysses_degree: int | None
    ring_degree: int | None
    status: str
    returncode: int | None
    elapsed_s: float | None
    durations_s: dict[str, float | None]
    perf_path: str
    case_dir: str
    error_tail: str
    command: list[str]
    num_runs: int = 1
    num_warmup_runs: int = 0
    measured_runs: int = 1
    successful_measured_runs: int = 0
    run_statuses: list[str] = dataclasses.field(default_factory=list)
    run_results: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def to_csv_row(self) -> dict[str, str]:
        return {
            "row_name": self.row_name,
            "text_encoder": format_duration(self.durations_s.get("text_encoder")),
            "denoising": format_duration(self.durations_s.get("denoising")),
            "decoder": format_duration(self.durations_s.get("decoder")),
        }

    def to_detail_row(self) -> dict[str, str]:
        row = {
            "row_name": self.row_name,
            "model": self.model,
            "model_path": self.model_path,
            "gpu_num": str(self.gpu_num),
            "resolution": self.resolution,
            "width": str(self.width),
            "height": str(self.height),
            "ulysses_degree": none_to_empty(self.ulysses_degree),
            "ring_degree": none_to_empty(self.ring_degree),
            "status": self.status,
            "returncode": none_to_empty(self.returncode),
            "elapsed_s": format_duration(self.elapsed_s),
            "num_runs": str(self.num_runs),
            "num_warmup_runs": str(self.num_warmup_runs),
            "measured_runs": str(self.measured_runs),
            "successful_measured_runs": str(self.successful_measured_runs),
            "run_statuses": "|".join(self.run_statuses),
            "text_encoder": format_duration(self.durations_s.get("text_encoder")),
            "denoising": format_duration(self.durations_s.get("denoising")),
            "decoder": format_duration(self.durations_s.get("decoder")),
            "perf_path": self.perf_path,
            "case_dir": self.case_dir,
            "error_tail": self.error_tail.replace("\r", "\\r").replace("\n", "\\n"),
            "command": format_command(self.command),
        }
        return row


def none_to_empty(value: Any) -> str:
    return "" if value is None else str(value)


def format_duration(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value:.6f}"


def average_values(values: Sequence[float | None]) -> float | None:
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return sum(valid_values) / len(valid_values)


def round_down_to_multiple(value: int, multiple: int) -> int:
    rounded = value - (value % multiple)
    if rounded <= 0:
        raise ValueError(f"Cannot round {value} down to a positive multiple of {multiple}")
    return rounded


def resolve_resolution(model_name: str, label: str) -> tuple[int, int]:
    if label not in CANONICAL_RESOLUTIONS:
        raise ValueError(
            f"Unknown resolution '{label}'. Expected one of: {', '.join(ALL_RESOLUTIONS)}"
        )

    if model_name == "wan2.2-ti2v-5b" and label == "720p":
        return 1280, 704

    preset = MODEL_PRESETS[model_name]
    width, height = CANONICAL_RESOLUTIONS[label]
    return (
        round_down_to_multiple(width, preset.size_multiple),
        round_down_to_multiple(height, preset.size_multiple),
    )


def detect_attention_heads(model_path: str, fallback: int) -> int:
    """Best-effort local config reader; HF repo IDs use the preset fallback."""
    path = Path(model_path).expanduser()
    if not path.exists() or not path.is_dir():
        return fallback

    candidates = [
        path / "transformer" / "config.json",
        path / "transformer_2" / "config.json",
        path / "dit" / "config.json",
        path / "config.json",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        heads = first_int_for_keys(
            config,
            keys=(
                "num_attention_heads",
                "attention_heads",
                "num_heads",
                "n_heads",
                "n_kv_heads",
            ),
        )
        if heads is not None and heads > 0:
            return heads
    return fallback


def first_int_for_keys(obj: Any, keys: Sequence[str]) -> int | None:
    if isinstance(obj, Mapping):
        for key in keys:
            value = obj.get(key)
            if isinstance(value, int):
                return value
        for value in obj.values():
            nested = first_int_for_keys(value, keys)
            if nested is not None:
                return nested
    elif isinstance(obj, list):
        for value in obj:
            nested = first_int_for_keys(value, keys)
            if nested is not None:
                return nested
    return None


def parse_auto_or_int(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("degree must be a positive integer or 'auto'")
    return parsed


def resolve_image_paths(
    preset: ModelPreset,
    image_paths: Sequence[str] | None,
) -> list[str] | None:
    if image_paths:
        return [str(path) for path in image_paths]
    if preset.default_image_paths:
        return [resolve_repo_relative_path(path) for path in preset.default_image_paths]
    return None


def resolve_repo_relative_path(path: str) -> str:
    maybe_path = Path(path)
    if maybe_path.is_absolute():
        return str(maybe_path)
    return str(repo_root_from_file() / maybe_path)


def resolve_parallelism(
    gpu_num: int,
    attention_heads: int | None,
    ulysses_degree: int | None,
    ring_degree: int | None,
) -> ParallelConfig:
    sp_degree = gpu_num
    user_set_degree = ulysses_degree is not None or ring_degree is not None

    if user_set_degree:
        if ulysses_degree is None:
            if sp_degree % ring_degree != 0:
                return invalid_parallelism(
                    sp_degree,
                    ulysses_degree,
                    ring_degree,
                    "sp_degree must be divisible by explicit ring_degree",
                )
            ulysses_degree = sp_degree // ring_degree
        if ring_degree is None:
            if sp_degree % ulysses_degree != 0:
                return invalid_parallelism(
                    sp_degree,
                    ulysses_degree,
                    ring_degree,
                    "sp_degree must be divisible by explicit ulysses_degree",
                )
            ring_degree = sp_degree // ulysses_degree
        if ulysses_degree * ring_degree != sp_degree:
            return invalid_parallelism(
                sp_degree,
                ulysses_degree,
                ring_degree,
                "sp_degree must equal ring_degree * ulysses_degree",
            )
        return ParallelConfig(
            sp_degree=sp_degree,
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            reason="user",
        )

    candidate = ParallelConfig(
        sp_degree=sp_degree,
        ulysses_degree=sp_degree,
        ring_degree=1,
        reason="auto",
    )
    if attention_heads is None or attention_heads % candidate.ulysses_degree == 0:
        return candidate

    fallback = fallback_parallelism(gpu_num, reason="head_conflict_fallback")
    if fallback.valid and attention_heads % fallback.ulysses_degree == 0:
        return fallback
    return candidate


def invalid_parallelism(
    sp_degree: int,
    ulysses_degree: int | None,
    ring_degree: int | None,
    error: str,
) -> ParallelConfig:
    return ParallelConfig(
        sp_degree=sp_degree,
        ulysses_degree=ulysses_degree or 0,
        ring_degree=ring_degree or 0,
        valid=False,
        reason="invalid",
        error=error,
    )


def fallback_parallelism(
    gpu_num: int,
    reason: str = "head_conflict_fallback",
) -> ParallelConfig:
    if gpu_num == 1:
        return ParallelConfig(
            sp_degree=1,
            ulysses_degree=1,
            ring_degree=1,
            reason=reason,
        )
    if gpu_num % 2 != 0:
        return ParallelConfig(
            sp_degree=gpu_num,
            ulysses_degree=0,
            ring_degree=0,
            valid=False,
            reason=reason,
            error="head-conflict fallback requires an even gpu_num",
        )
    return ParallelConfig(
        sp_degree=gpu_num,
        ulysses_degree=2,
        ring_degree=gpu_num // 2,
        reason=reason,
    )


def should_retry_with_head_fallback(output: RunOutput) -> bool:
    text = f"{output.stdout_tail}\n{output.stderr_tail}".lower()
    head_terms = ("head", "heads", "attention", "ulysses")
    conflict_terms = ("divisible", "divide", "mod", "assert", "invalid")
    return any(term in text for term in head_terms) and any(
        term in text for term in conflict_terms
    )


def extract_stage_durations(report: Mapping[str, Any]) -> dict[str, float | None]:
    durations_ms: dict[str, float] = {}

    for item in report.get("steps", []) or []:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        if name not in STAGE_NAME_TO_COLUMN:
            continue
        duration_ms = item.get("duration_ms", item.get("execution_time_ms"))
        if isinstance(duration_ms, (int, float)):
            durations_ms[STAGE_NAME_TO_COLUMN[name]] = float(duration_ms)

    stages = report.get("stages")
    if isinstance(stages, Mapping):
        for name, duration_ms in stages.items():
            if name in STAGE_NAME_TO_COLUMN and isinstance(duration_ms, (int, float)):
                durations_ms[STAGE_NAME_TO_COLUMN[name]] = float(duration_ms)
    elif isinstance(stages, list):
        for item in stages:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name")
            if name not in STAGE_NAME_TO_COLUMN:
                continue
            duration_ms = item.get("duration_ms", item.get("execution_time_ms"))
            if isinstance(duration_ms, (int, float)):
                durations_ms[STAGE_NAME_TO_COLUMN[name]] = float(duration_ms)

    if "denoising" not in durations_ms:
        denoise_steps = report.get("denoise_steps_ms", []) or []
        denoise_total = 0.0
        found_step = False
        for item in denoise_steps:
            if isinstance(item, Mapping):
                duration_ms = item.get("duration_ms", item.get("execution_time_ms"))
            else:
                duration_ms = item
            if isinstance(duration_ms, (int, float)):
                denoise_total += float(duration_ms)
                found_step = True
        if found_step:
            durations_ms["denoising"] = denoise_total

    return {
        "text_encoder": ms_to_seconds(durations_ms.get("text_encoder")),
        "denoising": ms_to_seconds(durations_ms.get("denoising")),
        "decoder": ms_to_seconds(durations_ms.get("decoder")),
    }


def ms_to_seconds(value_ms: float | None) -> float | None:
    if value_ms is None:
        return None
    return value_ms / 1000.0


def parse_perf_dump(perf_path: Path) -> dict[str, float | None]:
    with perf_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    return extract_stage_durations(report)


def build_generate_command(
    *,
    preset: ModelPreset,
    model_path: str,
    prompt: str,
    image_paths: Sequence[str] | None,
    gpu_num: int,
    width: int,
    height: int,
    parallel: ParallelConfig,
    case_dir: Path,
    base_gpu_id: int | None,
) -> list[str]:
    perf_path = case_dir / "perf.json"
    output_file_name = f"{preset.name}_{gpu_num}gpu_{width}x{height}"

    command = [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.runtime.entrypoints.cli.main",
        "generate",
        "--model-path",
        model_path,
        "--model-id",
        preset.model_id,
        "--num-gpus",
        str(gpu_num),
        "--tp-size",
        "1",
        "--sp-degree",
        str(parallel.sp_degree),
        "--ulysses-degree",
        str(parallel.ulysses_degree),
        "--ring-degree",
        str(parallel.ring_degree),
        "--warmup",
        "false",
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
        "--output-path",
        str(case_dir / "outputs"),
        "--input-save-path",
        str(case_dir / "inputs"),
        "--log-level",
        "info",
        "--width",
        str(width),
        "--height",
        str(height),
        "--prompt",
        prompt,
        "--output-file-name",
        output_file_name,
        "--perf-dump-path",
        str(perf_path),
    ]
    if base_gpu_id is not None:
        command.extend(["--base-gpu-id", str(base_gpu_id)])
    if image_paths:
        command.append("--image-path")
        command.extend(str(path) for path in image_paths)
    return command


def build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    python_root = Path(__file__).resolve().parents[3]
    current_pythonpath = env.get("PYTHONPATH")
    if current_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join([str(python_root), current_pythonpath])
    else:
        env["PYTHONPATH"] = str(python_root)

    env["SGLANG_DIFFUSION_SYNC_STAGE_PROFILING"] = "0"
    env["SGLANG_DIFFUSION_CAPTURE_STAGE_MEMORY"] = "0"
    env["SGLANG_DIFFUSION_STAGE_LOGGING"] = "0"
    return env


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    case_dir: Path,
    timeout_s: int,
) -> RunOutput:
    case_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = case_dir / "stdout.log"
    stderr_path = case_dir / "stderr.log"
    start = time.perf_counter()
    popen_kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "env": build_subprocess_env(),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(command, **popen_kwargs)
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
        timed_out = False
    except subprocess.TimeoutExpired:
        terminate_process_tree(process)
        stdout, stderr = process.communicate()
        timed_out = True

    elapsed_s = time.perf_counter() - start
    stdout_path.write_text(stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(stderr or "", encoding="utf-8", errors="replace")
    return RunOutput(
        returncode=process.returncode,
        elapsed_s=elapsed_s,
        timed_out=timed_out,
        stdout_tail=tail_text(stdout_path),
        stderr_tail=tail_text(stderr_path),
    )


def terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)


def tail_text(path: Path, max_bytes: int = 8000) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(max(0, size - max_bytes))
        return f.read().decode("utf-8", errors="replace")


def format_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(list(command))


def available_gpu_count() -> int | None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        devices = [item.strip() for item in visible_devices.split(",") if item.strip()]
        if devices == ["-1"]:
            return 0
        return len(devices)

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return sum(1 for line in result.stdout.splitlines() if line.strip())


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[4]


def default_output_dir() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path.cwd() / "outputs" / "profile_stage_breakdown" / timestamp


def write_csv_outputs(
    results: Sequence[CaseResult],
    *,
    csv_path: Path,
    details_csv_path: Path,
    details_json_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_csv_row())

    with details_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_detail_row())

    details_payload = []
    for result in results:
        payload = dataclasses.asdict(result)
        payload["command_text"] = format_command(result.command)
        details_payload.append(payload)
    details_json_path.write_text(
        json.dumps(details_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def make_run_detail(
    result: CaseResult,
    *,
    run_index: int,
    is_warmup: bool,
) -> dict[str, Any]:
    return {
        "run_index": run_index,
        "is_warmup": is_warmup,
        "status": result.status,
        "returncode": result.returncode,
        "elapsed_s": result.elapsed_s,
        "durations_s": result.durations_s,
        "perf_path": result.perf_path,
        "case_dir": result.case_dir,
        "error_tail": result.error_tail,
        "command": result.command,
        "command_text": format_command(result.command),
        "ulysses_degree": result.ulysses_degree,
        "ring_degree": result.ring_degree,
    }


def average_measured_durations(
    run_results: Sequence[CaseResult],
    *,
    num_warmup_runs: int,
) -> dict[str, float | None]:
    measured_results = run_results[num_warmup_runs:]
    return {
        stage: average_values(
            [result.durations_s.get(stage) for result in measured_results]
        )
        for stage in STAGE_COLUMNS
    }


def aggregate_repeated_results(
    run_results: Sequence[CaseResult],
    *,
    num_warmup_runs: int,
    case_dir: Path,
) -> CaseResult:
    if not run_results:
        raise ValueError("run_results must not be empty")

    first_result = run_results[0]
    measured_results = list(run_results[num_warmup_runs:])
    durations = average_measured_durations(
        run_results, num_warmup_runs=num_warmup_runs
    )
    measured_statuses = [result.status for result in measured_results]
    successful_measured_runs = sum(status == "ok" for status in measured_statuses)

    if all(result.status == "dry_run" for result in run_results):
        status = "dry_run"
    elif measured_results and successful_measured_runs == len(measured_results):
        status = "ok"
    elif any(value is not None for value in durations.values()):
        status = "partial_repeats"
    else:
        status = "failed_repeats"

    elapsed_values = [result.elapsed_s for result in run_results if result.elapsed_s]
    returncode = 0 if all(result.returncode == 0 for result in run_results) else None
    if any(result.returncode not in (0, None) for result in run_results):
        returncode = next(
            result.returncode
            for result in run_results
            if result.returncode not in (0, None)
        )

    run_details = [
        make_run_detail(
            result,
            run_index=index + 1,
            is_warmup=index < num_warmup_runs,
        )
        for index, result in enumerate(run_results)
    ]
    measured_perf_paths = [
        detail["perf_path"] for detail in run_details if not detail["is_warmup"]
    ]
    error_tail = "\n".join(
        f"run_{index + 1}: {result.error_tail}"
        for index, result in enumerate(run_results)
        if result.error_tail
    )

    return CaseResult(
        row_name=first_result.row_name,
        model=first_result.model,
        model_path=first_result.model_path,
        gpu_num=first_result.gpu_num,
        resolution=first_result.resolution,
        width=first_result.width,
        height=first_result.height,
        ulysses_degree=first_result.ulysses_degree,
        ring_degree=first_result.ring_degree,
        status=status,
        returncode=returncode,
        elapsed_s=sum(elapsed_values) if elapsed_values else None,
        durations_s=durations,
        perf_path=";".join(measured_perf_paths),
        case_dir=str(case_dir),
        error_tail=error_tail,
        command=first_result.command,
        num_runs=len(run_results),
        num_warmup_runs=num_warmup_runs,
        measured_runs=len(measured_results),
        successful_measured_runs=successful_measured_runs,
        run_statuses=[result.status for result in run_results],
        run_results=run_details,
    )


def make_failure_result(
    *,
    row_name: str,
    preset: ModelPreset,
    model_path: str,
    gpu_num: int,
    resolution: str,
    width: int,
    height: int,
    parallel: ParallelConfig | None,
    status: str,
    case_dir: Path,
    error_tail: str,
    command: Sequence[str] | None = None,
    returncode: int | None = None,
    elapsed_s: float | None = None,
) -> CaseResult:
    return CaseResult(
        row_name=row_name,
        model=preset.name,
        model_path=model_path,
        gpu_num=gpu_num,
        resolution=resolution,
        width=width,
        height=height,
        ulysses_degree=parallel.ulysses_degree if parallel else None,
        ring_degree=parallel.ring_degree if parallel else None,
        status=status,
        returncode=returncode,
        elapsed_s=elapsed_s,
        durations_s={"text_encoder": None, "denoising": None, "decoder": None},
        perf_path=str(case_dir / "perf.json"),
        case_dir=str(case_dir),
        error_tail=error_tail,
        command=list(command or []),
    )


def run_case(
    *,
    preset: ModelPreset,
    model_path: str,
    prompt: str,
    image_paths: Sequence[str] | None,
    gpu_num: int,
    resolution: str,
    width: int,
    height: int,
    parallel: ParallelConfig,
    case_dir: Path,
    timeout_s: int,
    dry_run: bool,
    base_gpu_id: int | None,
) -> CaseResult:
    row_name = f"{gpu_num}_{resolution}"
    command = build_generate_command(
        preset=preset,
        model_path=model_path,
        prompt=prompt,
        image_paths=image_paths,
        gpu_num=gpu_num,
        width=width,
        height=height,
        parallel=parallel,
        case_dir=case_dir,
        base_gpu_id=base_gpu_id,
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "command.txt").write_text(format_command(command), encoding="utf-8")

    if dry_run:
        return make_failure_result(
            row_name=row_name,
            preset=preset,
            model_path=model_path,
            gpu_num=gpu_num,
            resolution=resolution,
            width=width,
            height=height,
            parallel=parallel,
            status="dry_run",
            case_dir=case_dir,
            error_tail="",
            command=command,
        )

    output = run_command(
        command,
        cwd=repo_root_from_file(),
        case_dir=case_dir,
        timeout_s=timeout_s,
    )
    if output.timed_out:
        return make_failure_result(
            row_name=row_name,
            preset=preset,
            model_path=model_path,
            gpu_num=gpu_num,
            resolution=resolution,
            width=width,
            height=height,
            parallel=parallel,
            status="timeout",
            case_dir=case_dir,
            error_tail=output.stderr_tail or output.stdout_tail,
            command=command,
            returncode=output.returncode,
            elapsed_s=output.elapsed_s,
        )
    if output.returncode != 0:
        return make_failure_result(
            row_name=row_name,
            preset=preset,
            model_path=model_path,
            gpu_num=gpu_num,
            resolution=resolution,
            width=width,
            height=height,
            parallel=parallel,
            status="failed",
            case_dir=case_dir,
            error_tail=output.stderr_tail or output.stdout_tail,
            command=command,
            returncode=output.returncode,
            elapsed_s=output.elapsed_s,
        )

    perf_path = case_dir / "perf.json"
    if not perf_path.exists():
        return make_failure_result(
            row_name=row_name,
            preset=preset,
            model_path=model_path,
            gpu_num=gpu_num,
            resolution=resolution,
            width=width,
            height=height,
            parallel=parallel,
            status="missing_perf",
            case_dir=case_dir,
            error_tail="Process succeeded but perf.json was not created.",
            command=command,
            returncode=output.returncode,
            elapsed_s=output.elapsed_s,
        )

    try:
        durations = parse_perf_dump(perf_path)
    except (OSError, json.JSONDecodeError) as exc:
        return make_failure_result(
            row_name=row_name,
            preset=preset,
            model_path=model_path,
            gpu_num=gpu_num,
            resolution=resolution,
            width=width,
            height=height,
            parallel=parallel,
            status="bad_perf",
            case_dir=case_dir,
            error_tail=f"Could not parse perf dump: {exc}",
            command=command,
            returncode=output.returncode,
            elapsed_s=output.elapsed_s,
        )

    status = "ok"
    if any(value is None for value in durations.values()):
        status = "partial_perf"
    return CaseResult(
        row_name=row_name,
        model=preset.name,
        model_path=model_path,
        gpu_num=gpu_num,
        resolution=resolution,
        width=width,
        height=height,
        ulysses_degree=parallel.ulysses_degree,
        ring_degree=parallel.ring_degree,
        status=status,
        returncode=output.returncode,
        elapsed_s=output.elapsed_s,
        durations_s=durations,
        perf_path=str(perf_path),
        case_dir=str(case_dir),
        error_tail="" if status == "ok" else "Perf dump is missing one or more stages.",
        command=command,
    )


def run_case_with_optional_retry(
    *,
    preset: ModelPreset,
    model_path: str,
    prompt: str,
    image_paths: Sequence[str] | None,
    gpu_num: int,
    resolution: str,
    width: int,
    height: int,
    parallel: ParallelConfig,
    case_dir: Path,
    timeout_s: int,
    dry_run: bool,
    base_gpu_id: int | None,
) -> CaseResult:
    result = run_case(
        preset=preset,
        model_path=model_path,
        prompt=prompt,
        image_paths=image_paths,
        gpu_num=gpu_num,
        resolution=resolution,
        width=width,
        height=height,
        parallel=parallel,
        case_dir=case_dir,
        timeout_s=timeout_s,
        dry_run=dry_run,
        base_gpu_id=base_gpu_id,
    )
    if (
        dry_run
        or result.status not in {"failed", "timeout"}
        or parallel.is_fallback
        or not should_retry_with_head_fallback(
            RunOutput(
                returncode=result.returncode,
                elapsed_s=result.elapsed_s or 0.0,
                timed_out=result.status == "timeout",
                stdout_tail="",
                stderr_tail=result.error_tail,
            )
        )
    ):
        return result

    fallback = fallback_parallelism(gpu_num)
    if not fallback.valid or (
        fallback.ulysses_degree == parallel.ulysses_degree
        and fallback.ring_degree == parallel.ring_degree
    ):
        return result

    retry_result = run_case(
        preset=preset,
        model_path=model_path,
        prompt=prompt,
        image_paths=image_paths,
        gpu_num=gpu_num,
        resolution=resolution,
        width=width,
        height=height,
        parallel=fallback,
        case_dir=case_dir / "fallback_u2",
        timeout_s=timeout_s,
        dry_run=dry_run,
        base_gpu_id=base_gpu_id,
    )
    retry_result.error_tail = (
        "Retried after initial launch failure that looked like an attention-head "
        f"parallelism conflict. Initial error tail: {result.error_tail}"
        if retry_result.status == "ok"
        else retry_result.error_tail
    )
    return retry_result


def run_repeated_case_with_optional_retry(
    *,
    preset: ModelPreset,
    model_path: str,
    prompt: str,
    image_paths: Sequence[str] | None,
    gpu_num: int,
    resolution: str,
    width: int,
    height: int,
    parallel: ParallelConfig,
    case_dir: Path,
    timeout_s: int,
    dry_run: bool,
    base_gpu_id: int | None,
    num_runs: int,
    num_warmup_runs: int,
) -> CaseResult:
    run_results: list[CaseResult] = []
    for run_index in range(1, num_runs + 1):
        run_kind = "warmup" if run_index <= num_warmup_runs else "measure"
        run_case_dir = case_dir / f"run_{run_index:02d}_{run_kind}"
        print(
            f"  [request] {gpu_num}_{resolution} "
            f"run {run_index}/{num_runs} ({run_kind})"
        )
        result = run_case_with_optional_retry(
            preset=preset,
            model_path=model_path,
            prompt=prompt,
            image_paths=image_paths,
            gpu_num=gpu_num,
            resolution=resolution,
            width=width,
            height=height,
            parallel=parallel,
            case_dir=run_case_dir,
            timeout_s=timeout_s,
            dry_run=dry_run,
            base_gpu_id=base_gpu_id,
        )
        run_results.append(result)

    return aggregate_repeated_results(
        run_results,
        num_warmup_runs=num_warmup_runs,
        case_dir=case_dir,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile text_encoder/denoising/decoder stage times for one "
            "SGLang diffusion model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--resolutions", nargs="+", default=list(ALL_RESOLUTIONS))
    parser.add_argument("--gpu-nums", nargs="+", type=int, default=list(DEFAULT_GPU_NUMS))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--image-path", nargs="+", default=None)
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument(
        "--num-warmup-runs",
        type=int,
        default=DEFAULT_NUM_WARMUP_RUNS,
        help="Number of leading runs to exclude from the averaged CSV result.",
    )
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--base-gpu-id", type=int, default=None)
    parser.add_argument("--ulysses-degree", type=parse_auto_or_int, default=None)
    parser.add_argument("--ring-degree", type=parse_auto_or_int, default=None)
    parser.add_argument(
        "--no-gpu-availability-check",
        action="store_true",
        help="Do not skip cases whose requested GPU count exceeds detected visible GPUs.",
    )
    args = parser.parse_args(argv)

    unknown_resolutions = [r for r in args.resolutions if r not in ALL_RESOLUTIONS]
    if unknown_resolutions:
        raise SystemExit(
            f"Unknown resolution(s): {unknown_resolutions}. Expected: {list(ALL_RESOLUTIONS)}"
        )
    if any(gpu_num <= 0 for gpu_num in args.gpu_nums):
        raise SystemExit("--gpu-nums values must be positive integers")
    if args.timeout_s <= 0:
        raise SystemExit("--timeout-s must be positive")
    if args.num_runs <= 0:
        raise SystemExit("--num-runs must be positive")
    if args.num_warmup_runs < 0:
        raise SystemExit("--num-warmup-runs must be >= 0")
    if args.num_warmup_runs >= args.num_runs:
        raise SystemExit("--num-warmup-runs must be smaller than --num-runs")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preset = MODEL_PRESETS[args.model]
    model_path = args.model_path or preset.hf_path
    prompt = args.prompt or preset.default_prompt
    image_paths = resolve_image_paths(preset, args.image_path)
    output_dir = (args.output_dir or default_output_dir()).resolve()
    csv_path = (args.csv_path or output_dir / "stage_breakdown.csv").resolve()
    details_csv_path = output_dir / "stage_breakdown_details.csv"
    details_json_path = output_dir / "stage_breakdown_details.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    attention_heads = detect_attention_heads(model_path, preset.attention_heads)
    detected_gpus = None if args.no_gpu_availability_check else available_gpu_count()
    results: list[CaseResult] = []

    for gpu_num in args.gpu_nums:
        for resolution in args.resolutions:
            width, height = resolve_resolution(preset.name, resolution)
            case_dir = output_dir / "cases" / f"{preset.name}_{gpu_num}gpu_{resolution}"
            row_name = f"{gpu_num}_{resolution}"

            if detected_gpus is not None and gpu_num > detected_gpus:
                result = make_failure_result(
                    row_name=row_name,
                    preset=preset,
                    model_path=model_path,
                    gpu_num=gpu_num,
                    resolution=resolution,
                    width=width,
                    height=height,
                    parallel=None,
                    status="skipped_unavailable_gpus",
                    case_dir=case_dir,
                    error_tail=(
                        f"Requested {gpu_num} GPU(s), but only {detected_gpus} "
                        "visible GPU(s) were detected."
                    ),
                )
                results.append(result)
                print(f"[skip] {row_name}: {result.error_tail}")
                if args.fail_fast:
                    write_csv_outputs(
                        results,
                        csv_path=csv_path,
                        details_csv_path=details_csv_path,
                        details_json_path=details_json_path,
                    )
                    return 1
                continue

            parallel = resolve_parallelism(
                gpu_num=gpu_num,
                attention_heads=attention_heads,
                ulysses_degree=args.ulysses_degree,
                ring_degree=args.ring_degree,
            )
            if not parallel.valid:
                result = make_failure_result(
                    row_name=row_name,
                    preset=preset,
                    model_path=model_path,
                    gpu_num=gpu_num,
                    resolution=resolution,
                    width=width,
                    height=height,
                    parallel=parallel,
                    status="invalid_parallelism",
                    case_dir=case_dir,
                    error_tail=parallel.error,
                )
                results.append(result)
                print(f"[invalid] {row_name}: {parallel.error}")
                if args.fail_fast:
                    write_csv_outputs(
                        results,
                        csv_path=csv_path,
                        details_csv_path=details_csv_path,
                        details_json_path=details_json_path,
                    )
                    return 1
                continue

            print(
                "[run] "
                f"{row_name}: {width}x{height}, gpu={gpu_num}, "
                f"ulysses={parallel.ulysses_degree}, ring={parallel.ring_degree}"
            )
            result = run_repeated_case_with_optional_retry(
                preset=preset,
                model_path=model_path,
                prompt=prompt,
                image_paths=image_paths,
                gpu_num=gpu_num,
                resolution=resolution,
                width=width,
                height=height,
                parallel=parallel,
                case_dir=case_dir,
                timeout_s=args.timeout_s,
                dry_run=args.dry_run,
                base_gpu_id=args.base_gpu_id,
                num_runs=args.num_runs,
                num_warmup_runs=args.num_warmup_runs,
            )
            results.append(result)
            print(f"[{result.status}] {row_name} -> {result.to_csv_row()}")
            if args.fail_fast and result.status not in {"ok", "dry_run"}:
                write_csv_outputs(
                    results,
                    csv_path=csv_path,
                    details_csv_path=details_csv_path,
                    details_json_path=details_json_path,
                )
                return 1

            write_csv_outputs(
                results,
                csv_path=csv_path,
                details_csv_path=details_csv_path,
                details_json_path=details_json_path,
            )

    write_csv_outputs(
        results,
        csv_path=csv_path,
        details_csv_path=details_csv_path,
        details_json_path=details_json_path,
    )
    print(f"Wrote stage CSV: {csv_path}")
    print(f"Wrote details CSV: {details_csv_path}")
    print(f"Wrote details JSON: {details_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
