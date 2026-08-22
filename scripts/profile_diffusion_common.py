#!/usr/bin/env python3
"""Shared helpers for lightweight Wan and Z-Image profiling scripts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import fmean
from typing import Any, BinaryIO, Iterable
from urllib.parse import urlparse

import requests


NUM_RUNS = 5
NUM_WARMUP_RUNS = 2
MAX_PORT_LAUNCH_ATTEMPTS = 5
EXPECTED_DENOISE_STEPS = 50
SUPPORTED_GPU_COUNTS = (1, 2, 4, 8)
REQUIRED_STAGE_NAMES = (
    "TextEncodingStage",
    "DenoisingStage",
    "DecodingStage",
)
DEFAULT_REFERENCE_IMAGE = str(
    Path(__file__).resolve().parents[1]
    / "examples"
    / "frontend_language"
    / "quick_start"
    / "images"
    / "cat.jpeg"
)

# Match the April H200 Z-Image baseline. Z-Image has 30 attention heads, so
# these points use the largest Ulysses degree that divides both SP and 30;
# the remaining SP dimension is assigned to Ring Attention.
Z_IMAGE_APRIL_PARALLELISM = {
    1: (1, 1),
    2: (2, 1),
    4: (2, 2),
    8: (2, 4),
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_id: str
    default_path: Path
    task_type: str
    pipeline_class: str
    prompt: str
    size: str
    num_frames: int
    fps: int
    parallelism: dict[int, tuple[int, int]]
    reference_image: bool = False


WAN22 = ModelSpec(
    key="wan22_ti2v_5b",
    label="Wan2.2-TI2V-5B",
    model_id="Wan2.2-TI2V-5B-Diffusers",
    default_path=Path("/workspace/models/Wan2.2-TI2V-5B-Diffusers"),
    task_type="TI2V",
    pipeline_class="WanPipeline",
    prompt="The cat starts walking slowly towards the camera.",
    size="1280x704",
    num_frames=121,
    fps=24,
    parallelism={1: (1, 1), 2: (2, 1), 4: (4, 1), 8: (8, 1)},
    reference_image=True,
)

WAN21 = ModelSpec(
    key="wan21_t2v_1_3b",
    label="Wan2.1-T2V-1.3B",
    model_id="Wan2.1-T2V-1.3B-Diffusers",
    default_path=Path("/workspace/models/Wan2.1-T2V-1.3B-Diffusers"),
    task_type="T2V",
    pipeline_class="WanPipeline",
    prompt="A curious raccoon",
    size="832x480",
    num_frames=81,
    fps=16,
    parallelism={1: (1, 1), 2: (2, 1), 4: (4, 1), 8: (4, 2)},
)

Z_IMAGE = ModelSpec(
    key="z_image",
    label="Z-Image",
    model_id="Z-Image",
    default_path=Path("/workspace/models/Z-Image"),
    task_type="T2I",
    pipeline_class="ZImagePipeline",
    prompt="Doraemon is eating dorayaki",
    size="1024x1024",
    num_frames=1,
    fps=24,
    parallelism=Z_IMAGE_APRIL_PARALLELISM,
)

ALL_MODEL_SPECS = (WAN22, WAN21, Z_IMAGE)
WAN_MODEL_SPECS = (WAN22, WAN21)


@dataclass
class LaunchedServer:
    process: subprocess.Popen[bytes]
    log_file: BinaryIO
    log_path: Path
    command: list[str]
    started_ns: int


@dataclass
class ReadyServer:
    server: LaunchedServer
    ports: dict[str, int]
    command: list[str]
    model_card: dict[str, Any]
    ready_ns: int
    launch_attempts: list[dict[str, Any]]


class StrictPortUnavailableError(RuntimeError):
    pass


def normalize_gpu_counts(values: Iterable[int]) -> list[int]:
    counts = list(dict.fromkeys(values))
    invalid = [value for value in counts if value not in SUPPORTED_GPU_COUNTS]
    if invalid:
        raise ValueError(
            f"GPU counts must be selected from {SUPPORTED_GPU_COUNTS}; got {invalid}"
        )
    if not counts:
        raise ValueError("At least one GPU count is required")
    return counts


def normalize_attention_backend(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("Attention backend must not be empty")
    return normalized


def validate_cuda_compat_lib_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"CUDA compatibility library directory not found: {resolved}"
        )
    if not any(
        (resolved / library).is_file()
        for library in ("libcuda.so.1", "libcuda.so")
    ):
        raise FileNotFoundError(
            f"CUDA compatibility library directory does not contain libcuda.so.1 "
            f"or libcuda.so: {resolved}"
        )
    return resolved


def prepare_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {resolved}. Use a new run directory."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def validate_model_path(spec: ModelSpec, path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {resolved}")
    model_index_path = resolved / "model_index.json"
    if not model_index_path.is_file():
        raise FileNotFoundError(f"Missing model_index.json: {model_index_path}")
    model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
    class_name = model_index.get("_class_name")
    if class_name != spec.pipeline_class:
        raise ValueError(
            f"Expected _class_name={spec.pipeline_class!r} in {model_index_path}, "
            f"got {class_name!r}"
        )
    return resolved


def model_metadata(spec: ModelSpec, model_path: Path) -> dict[str, Any]:
    return {
        "key": spec.key,
        "label": spec.label,
        "model_id": spec.model_id,
        "model_path": str(model_path),
        "pipeline_class": spec.pipeline_class,
        "task_type": spec.task_type,
        "prompt": spec.prompt,
        "size": spec.size,
        "num_frames": spec.num_frames,
        "fps": spec.fps,
        "parallelism": {
            str(gpu_count): {"ulysses_degree": values[0], "ring_degree": values[1]}
            for gpu_count, values in spec.parallelism.items()
        },
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary_path, path)


@lru_cache(maxsize=1)
def repository_commit() -> str:
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit_hash = result.stdout.strip()
    if len(commit_hash) != 40 or any(
        character not in "0123456789abcdefABCDEF" for character in commit_hash
    ):
        raise RuntimeError(f"Invalid Git commit hash: {commit_hash!r}")
    return commit_hash.lower()


def read_log_tail(path: Path, line_count: int = 120) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def _reservation_bind_address(host: str) -> tuple[int, str]:
    if host in {"::", "[::]"}:
        return socket.AF_INET6, "::"
    return socket.AF_INET, ""


@lru_cache(maxsize=1)
def _profile_port_ranges() -> tuple[tuple[int, int], ...]:
    minimum = 10_000
    maximum = 65_534
    ephemeral_low = 32_768
    ephemeral_high = 60_999
    linux_ephemeral_range = Path("/proc/sys/net/ipv4/ip_local_port_range")
    try:
        values = linux_ephemeral_range.read_text(encoding="ascii").split()
        if len(values) == 2:
            ephemeral_low, ephemeral_high = (int(value) for value in values)
    except (OSError, ValueError):
        pass

    ranges = []
    if minimum <= ephemeral_low - 1:
        ranges.append((minimum, min(maximum, ephemeral_low - 1)))
    if ephemeral_high + 1 <= maximum:
        ranges.append((max(minimum, ephemeral_high + 1), maximum))
    if not ranges:
        ranges.append((minimum, maximum))
    return tuple(ranges)


def _random_profile_port(excluded: set[int], *, adjacent: bool = False) -> int:
    ranges = [
        (lower, upper - int(adjacent))
        for lower, upper in _profile_port_ranges()
        if lower <= upper - int(adjacent)
    ]
    for _ in range(1_000):
        lower, upper = random.choice(ranges)
        port = random.randint(lower, upper)
        required = {port, port + 1} if adjacent else {port}
        if required.isdisjoint(excluded):
            return port
    raise RuntimeError("Could not select a unique non-ephemeral profile port")


def _reserve_port(host: str, port: int) -> socket.socket:
    family, bind_host = _reservation_bind_address(host)
    reservation = socket.socket(family, socket.SOCK_STREAM)
    try:
        if family == socket.AF_INET6:
            reservation.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        reservation.bind((bind_host, port))
        reservation.listen(1)
        return reservation
    except BaseException:
        reservation.close()
        raise


def allocate_ports(host: str) -> dict[str, int]:
    for _ in range(500):
        reservations: list[socket.socket] = []
        try:
            excluded: set[int] = set()
            http_port = _random_profile_port(excluded, adjacent=True)
            reservations.append(_reserve_port(host, http_port))
            reservations.append(_reserve_port(host, http_port + 1))
            excluded.update((http_port, http_port + 1))

            scheduler_port = _random_profile_port(excluded)
            reservations.append(_reserve_port(host, scheduler_port))
            excluded.add(scheduler_port)

            master_port = _random_profile_port(excluded)
            reservations.append(_reserve_port(host, master_port))

            ports = {
                "http": http_port,
                "broker": http_port + 1,
                "scheduler": scheduler_port,
                "master": master_port,
            }
        except OSError:
            continue
        finally:
            for reservation in reservations:
                reservation.close()
        return ports
    raise RuntimeError(
        "Could not reserve unique HTTP, broker, scheduler, and master ports"
    )


def server_base_url(host: str, port: int) -> str:
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::", "[::]"} else host
    if ":" in connect_host and not connect_host.startswith("["):
        connect_host = f"[{connect_host}]"
    return f"http://{connect_host}:{port}"


def build_server_command(
    spec: ModelSpec,
    model_path: Path,
    gpu_count: int,
    host: str,
    ports: dict[str, int],
    server_dir: Path,
    *,
    attention_backend: str | None = None,
    text_encoder_cpu_offload: bool = False,
) -> list[str]:
    ulysses_degree, ring_degree = spec.parallelism[gpu_count]
    generated_dir = server_dir / "generated"
    uploaded_dir = server_dir / "uploaded"
    generated_dir.mkdir(parents=True, exist_ok=True)
    uploaded_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.runtime.launch_server",
        "--model-path",
        str(model_path),
        "--model-id",
        spec.model_id,
        "--backend",
        "sglang",
        "--performance-mode",
        "manual",
        "--num-gpus",
        str(gpu_count),
        "--tp-size",
        "1",
        "--sp-degree",
        str(gpu_count),
        "--ulysses-degree",
        str(ulysses_degree),
        "--ring-degree",
        str(ring_degree),
        "--enable-cfg-parallel",
        "false",
        "--cfg-parallel-size",
        "1",
        "--dit-cpu-offload",
        "false",
        "--dit-layerwise-offload",
        "false",
        "--text-encoder-cpu-offload",
        str(text_encoder_cpu_offload).lower(),
        "--image-encoder-cpu-offload",
        "false",
        "--vae-cpu-offload",
        "false",
        "--pin-cpu-memory",
        "false",
        "--use-fsdp-inference",
        "false",
        "--enable-torch-compile",
        "false",
        "--enable-breakable-cuda-graph",
        "false",
        "--enable-layerwise-nvtx-marker",
        "false",
        "--warmup-mode",
        "off",
        "--host",
        host,
        "--port",
        str(ports["http"]),
        "--scheduler-port",
        str(ports["scheduler"]),
        "--master-port",
        str(ports["master"]),
        "--strict-ports",
        "true",
        "--output-path",
        str(generated_dir),
        "--input-save-path",
        str(uploaded_dir),
    ]
    if attention_backend is not None:
        command.extend(["--attention-backend", attention_backend])
    return command


def build_server_environment(
    server_dir: Path,
    *,
    enable_cuda_event_stage_profiling: bool = False,
    flush_offloaded_text_encoder_after_encoding: bool = False,
    cuda_compat_lib_dir: Path | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    if cuda_compat_lib_dir is not None:
        compat_dir = str(cuda_compat_lib_dir)
        existing = environment.get("LD_LIBRARY_PATH", "")
        library_paths = [compat_dir]
        library_paths.extend(path for path in existing.split(os.pathsep) if path)
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            dict.fromkeys(library_paths)
        )
    environment["SGLANG_CACHE_DIT_ENABLED"] = "false"
    environment["SGLANG_DIFFUSION_SYNC_STAGE_PROFILING"] = "0"
    environment["SGLANG_DIFFUSION_CUDA_EVENT_STAGE_PROFILING"] = (
        "1" if enable_cuda_event_stage_profiling else "0"
    )
    environment["SGLANG_DIFFUSION_STAGE_LOGGING"] = "0"
    environment.pop(
        "SGLANG_PROFILE_FLUSH_OFFLOADED_TEXT_ENCODER_AFTER_ENCODING", None
    )
    if flush_offloaded_text_encoder_after_encoding:
        environment[
            "SGLANG_PROFILE_FLUSH_OFFLOADED_TEXT_ENCODER_AFTER_ENCODING"
        ] = "1"
    environment["SGLANG_PERF_LOG_DIR"] = str(server_dir / "performance_logs")
    if enable_cuda_event_stage_profiling:
        environment["SGLANG_GIT_COMMIT"] = repository_commit()
    environment.pop("SGLANG_DIFFUSION_TORCH_PROFILER_DIR", None)
    environment.pop("SGLANG_TORCH_PROFILER_DIR", None)
    environment.pop("SGLANG_TEST_NUM_INFERENCE_STEPS", None)
    return environment


def launch_server(command: list[str], log_path: Path, environment: dict[str, str]) -> LaunchedServer:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("wb")
    popen_kwargs: dict[str, Any] = {
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        "env": environment,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    elif os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    started_ns = time.perf_counter_ns()
    try:
        process = subprocess.Popen(command, **popen_kwargs)
    except Exception:
        log_file.close()
        raise
    return LaunchedServer(process, log_file, log_path, command, started_ns)


def stop_server(server: LaunchedServer | None, timeout_s: float) -> None:
    if server is None:
        return
    process = server.process
    try:
        if process.poll() is None:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                process.terminate()
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait(timeout=10)
    finally:
        server.log_file.close()


def wait_for_ready(
    server: LaunchedServer,
    base_url: str,
    spec: ModelSpec,
    model_path: Path,
    gpu_count: int,
    timeout_s: float,
    poll_interval_s: float,
) -> tuple[dict[str, Any], int]:
    deadline = time.perf_counter() + timeout_s
    last_error = ""
    expected_path = str(model_path)
    with requests.Session() as session:
        session.trust_env = False
        while time.perf_counter() < deadline:
            return_code = server.process.poll()
            if return_code is not None:
                log_tail = read_log_tail(server.log_path)
                error_type = (
                    StrictPortUnavailableError
                    if "is unavailable and --strict-ports is enabled" in log_tail
                    else RuntimeError
                )
                raise error_type(
                    f"Server exited with code {return_code} before becoming ready.\n"
                    f"{log_tail}"
                )
            try:
                response = session.get(f"{base_url}/v1/models", timeout=1.0)
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                else:
                    payload = response.json()
                    cards = payload.get("data")
                    if not isinstance(cards, list) or len(cards) != 1:
                        last_error = f"unexpected model cards: {cards!r}"
                    else:
                        card = cards[0]
                        expected = {
                            "id": expected_path,
                            "pipeline_class": spec.pipeline_class,
                            "task_type": spec.task_type,
                            "num_gpus": gpu_count,
                        }
                        mismatches = {
                            key: {"expected": value, "actual": card.get(key)}
                            for key, value in expected.items()
                            if card.get(key) != value
                        }
                        if not mismatches:
                            return card, time.perf_counter_ns()
                        last_error = f"model card mismatches: {mismatches}"
            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
            time.sleep(poll_interval_s)
    raise TimeoutError(
        f"Timed out after {timeout_s}s waiting for {base_url}/v1/models: "
        f"{last_error}\n{read_log_tail(server.log_path)}"
    )


def launch_server_with_port_retries(
    spec: ModelSpec,
    model_path: Path,
    gpu_count: int,
    host: str,
    server_dir: Path,
    log_path: Path,
    *,
    attention_backend: str | None,
    enable_cuda_event_stage_profiling: bool,
    cuda_compat_lib_dir: Path | None,
    server_timeout_s: float,
    ready_poll_interval_s: float,
    shutdown_timeout_s: float,
    text_encoder_cpu_offload: bool = False,
    flush_offloaded_text_encoder_after_encoding: bool = False,
) -> ReadyServer:
    attempts_path = log_path.with_name(f"{log_path.stem}_launch_attempts.json")
    launch_attempts: list[dict[str, Any]] = []
    for attempt_number in range(1, MAX_PORT_LAUNCH_ATTEMPTS + 1):
        ports = allocate_ports(host)
        command = build_server_command(
            spec,
            model_path,
            gpu_count,
            host,
            ports,
            server_dir,
            attention_backend=attention_backend,
            text_encoder_cpu_offload=text_encoder_cpu_offload,
        )
        attempt_log_path = (
            log_path
            if attempt_number == 1
            else log_path.with_name(
                f"{log_path.stem}_port_retry_{attempt_number:02d}{log_path.suffix}"
            )
        )
        attempt: dict[str, Any] = {
            "attempt": attempt_number,
            "ports": ports,
            "command": command,
            "log_path": str(attempt_log_path),
            "status": "starting",
        }
        launch_attempts.append(attempt)
        save_json(attempts_path, launch_attempts)
        server = None
        try:
            server = launch_server(
                command,
                attempt_log_path,
                build_server_environment(
                    server_dir,
                    enable_cuda_event_stage_profiling=(
                        enable_cuda_event_stage_profiling
                    ),
                    flush_offloaded_text_encoder_after_encoding=(
                        flush_offloaded_text_encoder_after_encoding
                    ),
                    cuda_compat_lib_dir=cuda_compat_lib_dir,
                ),
            )
            base_url = server_base_url(host, ports["http"])
            model_card, ready_ns = wait_for_ready(
                server,
                base_url,
                spec,
                model_path,
                gpu_count,
                server_timeout_s,
                ready_poll_interval_s,
            )
        except StrictPortUnavailableError as exc:
            attempt["status"] = "strict_port_unavailable"
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            save_json(attempts_path, launch_attempts)
            stop_server(server, shutdown_timeout_s)
            server = None
            if attempt_number == MAX_PORT_LAUNCH_ATTEMPTS:
                raise
            print(
                f"[server] strict port collision on launch attempt "
                f"{attempt_number}/{MAX_PORT_LAUNCH_ATTEMPTS}; retrying",
                flush=True,
            )
            continue
        except BaseException as exc:
            attempt["status"] = "failed"
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            save_json(attempts_path, launch_attempts)
            stop_server(server, shutdown_timeout_s)
            raise

        attempt["status"] = "ready"
        attempt["ready_after_launch_ms"] = (
            ready_ns - server.started_ns
        ) / 1_000_000.0
        save_json(attempts_path, launch_attempts)
        return ReadyServer(
            server=server,
            ports=ports,
            command=command,
            model_card=model_card,
            ready_ns=ready_ns,
            launch_attempts=launch_attempts,
        )
    raise AssertionError("Port launch retry loop exited unexpectedly")


def materialize_reference_image(source: str, output_dir: Path, timeout_s: float) -> Path:
    target = output_dir / "inputs" / "wan22_reference.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        temporary = target.with_suffix(target.suffix + ".tmp")
        with requests.get(source, stream=True, timeout=(10.0, timeout_s)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        os.replace(temporary, target)
    else:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Reference image does not exist: {source_path}")
        shutil.copy2(source_path, target)
    if target.stat().st_size == 0:
        raise ValueError(f"Reference image is empty: {target}")
    return target


def _request_json(
    spec: ModelSpec,
    prompt: str,
    perf_dump_path: Path,
    reference_image: Path | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": spec.model_id,
        "prompt": prompt,
        "size": spec.size,
        "n": 1,
        "perf_dump_path": str(perf_dump_path),
    }
    if spec.task_type in {"T2V", "TI2V"}:
        payload["num_frames"] = spec.num_frames
        payload["fps"] = spec.fps
    if spec.reference_image:
        if reference_image is None:
            raise ValueError(f"{spec.label} requires a reference image")
        payload["input_reference"] = str(reference_image)
    return payload


def send_generation_request(
    base_url: str,
    spec: ModelSpec,
    prompt: str,
    perf_dump_path: Path,
    reference_image: Path | None,
    timeout_s: float,
    video_poll_interval_s: float,
) -> dict[str, Any]:
    perf_dump_path.parent.mkdir(parents=True, exist_ok=True)
    if perf_dump_path.exists():
        raise FileExistsError(f"Refusing to overwrite perf dump: {perf_dump_path}")
    payload = _request_json(spec, prompt, perf_dump_path, reference_image)
    with requests.Session() as session:
        session.trust_env = False
        if spec.task_type == "T2I":
            payload["response_format"] = "url"
            response = session.post(
                f"{base_url}/v1/images/generations",
                json=payload,
                timeout=(10.0, timeout_s),
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Image request failed with HTTP {response.status_code}: "
                    f"{response.text[:1000]}"
                )
            result = response.json()
            if not result.get("data"):
                raise RuntimeError(f"Image response has no output data: {result}")
            return result

        submit_started = time.perf_counter()
        response = session.post(
            f"{base_url}/v1/videos",
            json=payload,
            timeout=(10.0, min(timeout_s, 60.0)),
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Video submit failed with HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )
        submitted = response.json()
        job_id = submitted.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError(f"Video submit response has no job id: {submitted}")
        poll_url = f"{base_url}/v1/videos/{job_id}"
        while True:
            elapsed = time.perf_counter() - submit_started
            if elapsed >= timeout_s:
                raise TimeoutError(
                    f"Video request {job_id} timed out after {timeout_s}s"
                )
            time.sleep(video_poll_interval_s)
            poll = session.get(
                poll_url,
                timeout=(10.0, min(30.0, max(1.0, timeout_s - elapsed))),
            )
            if poll.status_code != 200:
                raise RuntimeError(
                    f"Video poll failed with HTTP {poll.status_code}: "
                    f"{poll.text[:1000]}"
                )
            result = poll.json()
            status = result.get("status")
            if status == "completed":
                if not result.get("file_path") and not result.get("file_paths"):
                    raise RuntimeError(f"Completed video response has no output: {result}")
                return result
            if status == "failed":
                raise RuntimeError(f"Video generation failed: {result}")


def response_request_id(response: dict[str, Any]) -> str:
    request_id = response.get("id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError(f"Generation response has no valid request id: {response}")
    return request_id


def _validate_perf_dump(
    payload: dict[str, Any],
    *,
    expected_request_id: str,
    expected_commit_hash: str,
    expected_model_path: Path,
    expected_world_size: int,
    expected_denoise_steps: int,
) -> None:
    if payload.get("schema_version") != 2:
        raise ValueError(
            f"perf dump schema_version must be 2, got {payload.get('schema_version')!r}"
        )
    if payload.get("tag") != "server_perf_dump":
        raise ValueError(f"Unexpected perf dump tag: {payload.get('tag')!r}")
    if payload.get("request_id") != expected_request_id:
        raise ValueError(
            "perf dump request_id does not match the completed request: "
            f"{payload.get('request_id')!r} != {expected_request_id!r}"
        )
    actual_commit = payload.get("commit_hash")
    if not isinstance(actual_commit, str) or (
        actual_commit.lower() != expected_commit_hash.lower()
    ):
        raise ValueError(
            "perf dump commit does not match the profiling script checkout: "
            f"{actual_commit!r} != {expected_commit_hash!r}"
        )
    if payload.get("stage_timing_method") != "cuda_event":
        raise ValueError(
            "perf dump did not use CUDA-event stage timing: "
            f"{payload.get('stage_timing_method')!r}"
        )
    cuda_event_stage_names = payload.get("cuda_event_stage_names")
    if not isinstance(cuda_event_stage_names, list) or (
        set(cuda_event_stage_names) != set(REQUIRED_STAGE_NAMES)
    ):
        raise ValueError(
            "CUDA-event timing must cover exactly the required module stages: "
            f"{cuda_event_stage_names!r}"
        )

    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("perf dump meta must be an object")
    if meta.get("rank") != 0:
        raise ValueError(
            f"perf dump must be published by rank 0, got {meta.get('rank')!r}"
        )
    if meta.get("world_size") != expected_world_size:
        raise ValueError(
            "perf dump world_size mismatch: "
            f"{meta.get('world_size')!r} != {expected_world_size}"
        )
    actual_model = meta.get("model")
    if not isinstance(actual_model, str) or (
        Path(actual_model).expanduser().resolve() != expected_model_path.resolve()
    ):
        raise ValueError(
            "perf dump model path mismatch: "
            f"{actual_model!r} != {str(expected_model_path)!r}"
        )

    total_duration_ms(payload)
    stages = payload.get("steps")
    if not isinstance(stages, list):
        raise ValueError("perf dump does not contain a steps list")
    stage_counts: dict[str, int] = {}
    for stage in stages:
        if not isinstance(stage, dict) or not isinstance(stage.get("name"), str):
            raise ValueError(f"Invalid stage entry: {stage!r}")
        stage_name = stage["name"]
        stage_counts[stage_name] = stage_counts.get(stage_name, 0) + 1
        require_nonnegative_ms(stage.get("duration_ms"), f"{stage_name}.duration_ms")
    invalid_stage_counts = {
        name: stage_counts.get(name, 0)
        for name in REQUIRED_STAGE_NAMES
        if stage_counts.get(name, 0) != 1
    }
    if invalid_stage_counts:
        raise ValueError(
            "Expected exactly one entry for each required stage; got "
            f"{invalid_stage_counts}"
        )
    for stage_name in REQUIRED_STAGE_NAMES:
        stage_duration_ms(payload, stage_name)

    denoise_steps = payload.get("denoise_steps_ms")
    if not isinstance(denoise_steps, list) or (
        len(denoise_steps) != expected_denoise_steps
    ):
        count = len(denoise_steps) if isinstance(denoise_steps, list) else None
        raise ValueError(
            f"Expected {expected_denoise_steps} denoise steps, got {count}"
        )
    for index, step in enumerate(denoise_steps):
        if not isinstance(step, dict) or step.get("step") != index:
            raise ValueError(f"Invalid denoise step {index}: {step!r}")
        require_positive_ms(step.get("duration_ms"), f"denoise step {index}")


def read_perf_dump(
    path: Path,
    timeout_s: float,
    *,
    expected_request_id: str,
    expected_commit_hash: str,
    expected_model_path: Path,
    expected_world_size: int,
    expected_denoise_steps: int = EXPECTED_DENOISE_STEPS,
) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_s
    last_error = "file was not created"
    stable_signature: tuple[int, int, str] | None = None
    stable_observations = 0
    while time.perf_counter() < deadline:
        if path.is_file():
            try:
                raw = path.read_bytes()
                stat = path.stat()
                signature = (
                    len(raw),
                    stat.st_mtime_ns,
                    hashlib.sha256(raw).hexdigest(),
                )
                if signature == stable_signature:
                    stable_observations += 1
                else:
                    stable_signature = signature
                    stable_observations = 1
                if stable_observations < 3:
                    time.sleep(0.1)
                    continue
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("perf dump root is not an object")
                _validate_perf_dump(
                    payload,
                    expected_request_id=expected_request_id,
                    expected_commit_hash=expected_commit_hash,
                    expected_model_path=expected_model_path,
                    expected_world_size=expected_world_size,
                    expected_denoise_steps=expected_denoise_steps,
                )
                return payload
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                last_error = str(exc)
        time.sleep(0.1)
    raise TimeoutError(f"Could not read perf dump {path}: {last_error}")


def require_positive_ms(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field_name} must be finite and positive, got {value!r}")
    return result


def require_nonnegative_ms(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(
            f"{field_name} must be finite and non-negative, got {value!r}"
        )
    return result


def total_duration_ms(perf_dump: dict[str, Any]) -> float:
    return require_positive_ms(perf_dump.get("total_duration_ms"), "total_duration_ms")


def denoising_stage_ms(perf_dump: dict[str, Any]) -> float:
    return stage_duration_ms(perf_dump, "DenoisingStage")


def stage_duration_ms(perf_dump: dict[str, Any], stage_name: str) -> float:
    stages = perf_dump.get("steps")
    if not isinstance(stages, list):
        raise ValueError("perf dump does not contain a steps list")
    matches = [
        stage
        for stage in stages
        if isinstance(stage, dict) and stage.get("name") == stage_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {stage_name} entry, found {len(matches)}"
        )
    return require_positive_ms(
        matches[0].get("duration_ms"), f"{stage_name}.duration_ms"
    )


def module_durations_ms(perf_dump: dict[str, Any]) -> dict[str, float]:
    return {
        "encoder": stage_duration_ms(perf_dump, "TextEncodingStage"),
        "denoiser": stage_duration_ms(perf_dump, "DenoisingStage"),
        "decoder": stage_duration_ms(perf_dump, "DecodingStage"),
    }


def measured_mean(records: list[dict[str, Any]], metric_name: str) -> float:
    if len(records) != NUM_RUNS:
        raise ValueError(f"Expected {NUM_RUNS} records, got {len(records)}")
    measured = records[NUM_WARMUP_RUNS:]
    values = [
        require_positive_ms(record.get(metric_name), metric_name)
        for record in measured
    ]
    return float(fmean(values))


def write_python_summary(
    path: Path, variable_values: list[tuple[str, list[list[float | int]]]]
) -> str:
    sections = []
    for variable_name, pairs in variable_values:
        pair_text = ", ".join(
            f"[{int(gpu_count)}, {float(value):.3f}]" for gpu_count, value in pairs
        )
        sections.append(f"{variable_name} = [{pair_text}]")
    text = "\n\n".join(sections) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def prompt_fingerprint(prompt: str) -> dict[str, Any]:
    return {
        "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "characters": len(prompt),
        "preview": prompt[:120],
    }
