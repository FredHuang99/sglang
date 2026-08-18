#!/usr/bin/env python3
"""Benchmark Wan and Z-Image text encoders with a CPU-only lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable


CPU_ENVIRONMENT = {
    "SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "cpu",
    "CUDA_VISIBLE_DEVICES": "",
    "SGLANG_USE_RUNAI_MODEL_STREAMER": "false",
}
THREAD_ENVIRONMENT_VARIABLES = (
    "SGLANG_CPU_OMP_THREADS_BIND",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "KMP_AFFINITY",
    "KMP_BLOCKTIME",
)
MODEL_KEYS = ("wan22_ti2v_5b", "wan21_t2v_1_3b", "z_image")


@dataclass(frozen=True)
class WorkerModel:
    key: str
    label: str
    model_id: str
    model_path: Path
    prompt: str
    width: int
    height: int
    num_frames: int
    fps: int
    reference_image: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Wan2.2, Wan2.1, and Z-Image text encoders entirely on CPU. "
            "Each model runs in an isolated subprocess."
        )
    )
    parser.add_argument(
        "--wan22-model-path",
        type=Path,
        default=Path("/workspace/models/Wan2.2-TI2V-5B-Diffusers"),
    )
    parser.add_argument(
        "--wan21-model-path",
        type=Path,
        default=Path("/workspace/models/Wan2.1-T2V-1.3B-Diffusers"),
    )
    parser.add_argument(
        "--z-image-model-path",
        type=Path,
        default=Path("/workspace/models/Z-Image"),
    )
    parser.add_argument(
        "--wan22-reference-image",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "examples"
            / "frontend_language"
            / "quick_start"
            / "images"
            / "cat.jpeg"
        ),
    )
    parser.add_argument(
        "--cpu-core-bind",
        default="auto",
        help=(
            "Use 'auto' for the SGLang TP1 NUMA/SNC policy, or pass a CPU list "
            "such as 0-63. The value is forwarded as SGLANG_CPU_OMP_THREADS_BIND."
        ),
    )
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/outputs/wan_zimage_cpu_encoder"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the worker plan without importing SGLang or loading weights.",
    )

    # Internal worker arguments. They are deliberately hidden from --help.
    parser.add_argument("--_worker-model", choices=MODEL_KEYS, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_master-port", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.num_runs != 5 or args.warmup_runs != 2:
        parser.error(
            "this benchmark uses a fixed 5-run protocol: --num-runs 5 and "
            "--warmup-runs 2"
        )
    if not args.cpu_core_bind.strip():
        parser.error("--cpu-core-bind must be 'auto' or a non-empty CPU list")
    if args._worker_model and (
        args._worker_result is None or args._master_port is None
    ):
        parser.error(
            "internal worker mode requires --_worker-result and --_master-port"
        )
    return args


def model_definitions(args: argparse.Namespace) -> dict[str, WorkerModel]:
    return {
        "wan22_ti2v_5b": WorkerModel(
            key="wan22_ti2v_5b",
            label="Wan2.2-TI2V-5B",
            model_id="Wan2.2-TI2V-5B-Diffusers",
            model_path=args.wan22_model_path,
            prompt="The cat starts walking slowly towards the camera.",
            width=1280,
            height=704,
            num_frames=121,
            fps=24,
            reference_image=args.wan22_reference_image,
        ),
        "wan21_t2v_1_3b": WorkerModel(
            key="wan21_t2v_1_3b",
            label="Wan2.1-T2V-1.3B",
            model_id="Wan2.1-T2V-1.3B-Diffusers",
            model_path=args.wan21_model_path,
            prompt="A curious raccoon",
            width=832,
            height=480,
            num_frames=81,
            fps=16,
            reference_image=None,
        ),
        "z_image": WorkerModel(
            key="z_image",
            label="Z-Image",
            model_id="Z-Image",
            model_path=args.z_image_model_path,
            prompt="Doraemon is eating dorayaki",
            width=1024,
            height=1024,
            num_frames=1,
            fps=24,
            reference_image=None,
        ),
    }


def resolve_worker_model(args: argparse.Namespace) -> WorkerModel:
    model = model_definitions(args)[args._worker_model]
    return WorkerModel(
        **{
            **asdict(model),
            "model_path": model.model_path.expanduser().resolve(),
            "reference_image": (
                model.reference_image.expanduser().resolve()
                if model.reference_image is not None
                else None
            ),
        }
    )


def prepare_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {resolved}. Use a new run directory."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def validate_inputs(models: Iterable[WorkerModel]) -> None:
    for model in models:
        if not model.model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model.model_path}")
        model_index = model.model_path / "model_index.json"
        if not model_index.is_file():
            raise FileNotFoundError(f"Missing model_index.json: {model_index}")
        if model.reference_image is not None and not model.reference_image.is_file():
            raise FileNotFoundError(
                f"Wan2.2 reference image not found: {model.reference_image}"
            )


def available_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def repository_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def worker_command(
    args: argparse.Namespace,
    model: WorkerModel,
    result_path: Path,
    master_port: int,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--wan22-model-path",
        str(args.wan22_model_path),
        "--wan21-model-path",
        str(args.wan21_model_path),
        "--z-image-model-path",
        str(args.z_image_model_path),
        "--wan22-reference-image",
        str(args.wan22_reference_image),
        "--cpu-core-bind",
        args.cpu_core_bind,
        "--num-runs",
        str(args.num_runs),
        "--warmup-runs",
        str(args.warmup_runs),
        "--_worker-model",
        model.key,
        "--_worker-result",
        str(result_path),
        "--_master-port",
        str(master_port),
    ]
    return command


def worker_environment(cpu_core_bind: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(CPU_ENVIRONMENT)
    if cpu_core_bind == "auto":
        env.pop("SGLANG_CPU_OMP_THREADS_BIND", None)
    else:
        env["SGLANG_CPU_OMP_THREADS_BIND"] = cpu_core_bind
    return env


def summarize_samples(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        raise ValueError("At least one measured sample is required")
    return {
        "avg_ms": statistics.fmean(values_ms),
        "p50_ms": statistics.median(values_ms),
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
        "std_ms": statistics.pstdev(values_ms),
    }


def execute_workers(args: argparse.Namespace) -> None:
    output_dir = prepare_output_dir(args.output_dir)
    models = list(model_definitions(args).values())
    models = [
        WorkerModel(
            **{
                **asdict(model),
                "model_path": model.model_path.expanduser().resolve(),
                "reference_image": (
                    model.reference_image.expanduser().resolve()
                    if model.reference_image is not None
                    else None
                ),
            }
        )
        for model in models
    ]
    validate_inputs(models)

    aggregate: dict[str, Any] = {
        "status": "running",
        "metric": "TextEncodingStage.forward wall time",
        "unit": "ms",
        "commit_hash": repository_commit(),
        "runs_per_model": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.num_runs - args.warmup_runs,
        "requested_cpu_core_bind": args.cpu_core_bind,
        "required_environment": CPU_ENVIRONMENT,
        "models": {},
    }
    result_path = output_dir / "cpu_encoder_results.json"
    save_json(result_path, aggregate)

    for model in models:
        model_dir = output_dir / model.key
        model_dir.mkdir(parents=True, exist_ok=True)
        worker_result = model_dir / "result.json"
        log_path = model_dir / "worker.log"
        command = worker_command(args, model, worker_result, available_local_port())
        print(f"[cpu-encoder] {model.label}", flush=True)
        with log_path.open("wb") as log_fp:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=worker_environment(args.cpu_core_bind),
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            aggregate["status"] = "failed"
            aggregate["failure"] = {
                "model": model.key,
                "returncode": completed.returncode,
                "worker_log": str(log_path),
            }
            save_json(result_path, aggregate)
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
            raise RuntimeError(
                f"{model.label} CPU encoder worker failed with code "
                f"{completed.returncode}. See {log_path}.\n{tail}"
            )
        if not worker_result.is_file():
            raise RuntimeError(f"Worker did not create result file: {worker_result}")
        aggregate["models"][model.key] = json.loads(
            worker_result.read_text(encoding="utf-8")
        )
        save_json(result_path, aggregate)

    aggregate["status"] = "success"
    save_json(result_path, aggregate)
    write_python_summary(output_dir / "summary.py", aggregate)
    write_markdown_summary(output_dir / "summary.md", aggregate)
    print(f"Results: {result_path}", flush=True)
    print(f"Summary: {output_dir / 'summary.md'}", flush=True)


def dry_run(args: argparse.Namespace) -> None:
    plans = []
    for model in model_definitions(args).values():
        result_path = args.output_dir / model.key / "result.json"
        plans.append(
            {
                "model": model.key,
                "environment": worker_environment(args.cpu_core_bind),
                "command": worker_command(args, model, result_path, 29500),
            }
        )
    # Keep the dry-run readable by showing only environment values controlled here.
    for plan in plans:
        plan["environment"] = {
            key: plan["environment"].get(key)
            for key in (*CPU_ENVIRONMENT, "SGLANG_CPU_OMP_THREADS_BIND")
        }
    print(json.dumps(plans, indent=2, ensure_ascii=True))


def write_python_summary(path: Path, aggregate: dict[str, Any]) -> None:
    summary: dict[str, Any] = {}
    lines = ["# Generated by profile_wan_zimage_cpu_encoder.py", ""]
    for key, result in aggregate["models"].items():
        samples = result["samples"]
        warmup = [sample["duration_ms"] for sample in samples if sample["warmup"]]
        measured = [sample["duration_ms"] for sample in samples if not sample["warmup"]]
        value = {
            "warmup_ms": warmup,
            "measured_ms": measured,
            **summarize_samples(measured),
        }
        summary[key] = value
        variable = f"{key}_encoder_cpu_duration_ms"
        lines.append(f"{variable} = {value!r}")
        lines.append("")
    lines.append(f"cpu_encoder_results_ms = {summary!r}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_summary(path: Path, aggregate: dict[str, Any]) -> None:
    lines = [
        "# Wan / Z-Image CPU Encoder Benchmark",
        "",
        "The timed region is only `TextEncodingStage.forward()`. Model loading and "
        "request/input preparation are reported separately.",
        "",
        "| Model | Load (s) | Measured avg (ms) | P50 (ms) | Min (ms) | Max (ms) | Std (ms) | Threads | Policy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in aggregate["models"].values():
        measured = [
            sample["duration_ms"]
            for sample in result["samples"]
            if not sample["warmup"]
        ]
        stats = summarize_samples(measured)
        thread = result["environment"]["thread_policy"]
        lines.append(
            "| {label} | {load:.2f} | {avg:.2f} | {p50:.2f} | {minimum:.2f} | "
            "{maximum:.2f} | {std:.2f} | {threads} | {policy} |".format(
                label=result["model"]["label"],
                load=result["load_duration_ms"] / 1000.0,
                avg=stats["avg_ms"],
                p50=stats["p50_ms"],
                minimum=stats["min_ms"],
                maximum=stats["max_ms"],
                std=stats["std_ms"],
                threads=thread["torch_num_threads_after"],
                policy=thread["effective_policy"],
            )
        )

    lines.extend(["", "## Samples", ""])
    for result in aggregate["models"].values():
        warmup = [
            sample["duration_ms"] for sample in result["samples"] if sample["warmup"]
        ]
        measured = [
            sample["duration_ms"]
            for sample in result["samples"]
            if not sample["warmup"]
        ]
        lines.extend(
            [
                f"### {result['model']['label']}",
                "",
                f"- Warmup (ms): `{[round(value, 3) for value in warmup]}`",
                f"- Measured (ms): `{[round(value, 3) for value in measured]}`",
                f"- Input preparation avg (ms): `{result['input_preparation_avg_ms']:.3f}`",
                f"- Tensor outputs: `{result['output_tensor_summary']}`",
                "",
            ]
        )

    lines.extend(["## Thread And NUMA Policy", ""])
    for result in aggregate["models"].values():
        env = result["environment"]
        thread = env["thread_policy"]
        lines.extend(
            [
                f"### {result['model']['label']}",
                "",
                f"- Effective policy: `{thread['effective_policy']}`",
                f"- AMX available: `{thread['amx_available']}`",
                f"- Requested binding: `{thread['requested_binding']}`",
                f"- Resolved binding: `{thread['resolved_binding']}`",
                f"- Actual affinity: `{env['cpu']['affinity_after']}`",
                f"- NUMA/SNC physical cores: `{env['cpu']['physical_cpu_ids_by_node']}`",
                f"- PyTorch intra-op / inter-op: `{thread['torch_num_threads_after']}` / "
                f"`{thread['torch_num_interop_threads']}`",
                f"- Warning: `{thread['warning']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tensor_descriptors(value: Any, prefix: str = "value") -> list[dict[str, Any]]:
    import torch

    found: list[dict[str, Any]] = []
    seen: set[int] = set()

    def visit(item: Any, name: str) -> None:
        if isinstance(item, torch.Tensor):
            found.append(
                {
                    "name": name,
                    "device": str(item.device),
                    "shape": list(item.shape),
                    "dtype": str(item.dtype),
                }
            )
            return
        if item is None or isinstance(item, (str, bytes, int, float, bool, Path)):
            return
        item_id = id(item)
        if item_id in seen:
            return
        seen.add(item_id)
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{name}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{name}[{index}]")
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name), f"{name}.{field.name}")

    visit(value, prefix)
    return found


def assert_cpu_tensors(descriptors: list[dict[str, Any]], context: str) -> None:
    non_cpu = [item for item in descriptors if item["device"] != "cpu"]
    if non_cpu:
        raise RuntimeError(f"{context} contains non-CPU tensors: {non_cpu[:10]}")


def parse_cpu_ids(value: str) -> set[int]:
    cpu_ids: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending CPU range: {token}")
            cpu_ids.update(range(start, end + 1))
        else:
            cpu_ids.add(int(token))
    if not cpu_ids:
        raise ValueError(f"CPU binding resolved to an empty set: {value!r}")
    return cpu_ids


def module_tensor_descriptors(modules: dict[str, Any]) -> list[dict[str, Any]]:
    import torch

    found: list[dict[str, Any]] = []
    for module_name, module in modules.items():
        if not module_name.startswith("text_encoder"):
            continue
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"{module_name} is not a torch.nn.Module: {type(module)}")
        for name, parameter in module.named_parameters(recurse=True):
            found.append(
                {
                    "name": f"{module_name}.parameter.{name}",
                    "device": str(parameter.device),
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                }
            )
        for name, buffer in module.named_buffers(recurse=True):
            found.append(
                {
                    "name": f"{module_name}.buffer.{name}",
                    "device": str(buffer.device),
                    "shape": list(buffer.shape),
                    "dtype": str(buffer.dtype),
                }
            )
    if not found:
        raise RuntimeError("No text encoder parameters or buffers were found")
    return found


def install_encoder_cpu_io_guards(modules: dict[str, Any]):
    import torch

    handles = []
    observed = {"input_tensor_count": 0, "output_tensor_count": 0}
    for module_name, module in modules.items():
        if not module_name.startswith("text_encoder"):
            continue
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"{module_name} is not a torch.nn.Module: {type(module)}")

        def pre_hook(_module, args, kwargs, name=module_name):
            descriptors = tensor_descriptors(
                {"args": args, "kwargs": kwargs}, f"{name}.forward_input"
            )
            assert_cpu_tensors(descriptors, f"{name} forward inputs")
            observed["input_tensor_count"] += len(descriptors)

        def post_hook(_module, args, kwargs, output, name=module_name):
            descriptors = tensor_descriptors(output, f"{name}.forward_output")
            if not descriptors:
                raise RuntimeError(f"{name} forward produced no tensor outputs")
            assert_cpu_tensors(descriptors, f"{name} forward outputs")
            observed["output_tensor_count"] += len(descriptors)

        handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(module.register_forward_hook(post_hook, with_kwargs=True))
    if not handles:
        raise RuntimeError("No text encoder modules were available for CPU I/O guards")
    return handles, observed


def configure_cpu_threads(requested_binding: str) -> dict[str, Any]:
    import psutil
    import torch

    from sglang.srt.utils import cpu_has_amx_support, get_cpu_ids_by_node
    from sglang.srt.utils.numa_utils import init_threads_binding

    try:
        import sgl_kernel  # noqa: F401
    except ImportError:
        pass

    amx_available = bool(cpu_has_amx_support())
    physical_cpu_ids_by_node = get_cpu_ids_by_node()
    init_op = getattr(torch.ops.sgl_kernel, "init_cpu_threads_env", None)
    initialize_op = getattr(torch.ops.sgl_kernel, "initialize", None)
    before = torch.get_num_threads()
    resolved_binding: str | None = None
    warning: str | None = None
    effective_policy = "pytorch_runtime_default"

    if amx_available and init_op is not None:
        resolved_binding = init_threads_binding(tp_rank=0, tp_size=1)
        init_op(resolved_binding)
        os.environ["LOCAL_SIZE"] = "1"
        if initialize_op is not None:
            initialize_op(1, 0)
        effective_policy = (
            "sglang_auto" if requested_binding == "auto" else "sglang_explicit"
        )
        expected_cpu_ids = parse_cpu_ids(resolved_binding)
        actual_cpu_ids = set(psutil.Process().cpu_affinity())
        if actual_cpu_ids != expected_cpu_ids:
            raise RuntimeError(
                "SGLang CPU binding did not produce the requested affinity: "
                f"expected={sorted(expected_cpu_ids)}, actual={sorted(actual_cpu_ids)}"
            )
        if torch.get_num_threads() != len(expected_cpu_ids):
            raise RuntimeError(
                "SGLang CPU binding did not align PyTorch intra-op threads with "
                f"the selected cores: threads={torch.get_num_threads()}, "
                f"cores={len(expected_cpu_ids)}"
            )
    else:
        missing = []
        if not amx_available:
            missing.append("Intel AMX")
        if init_op is None:
            missing.append("torch.ops.sgl_kernel.init_cpu_threads_env")
        warning = (
            "SGLang native CPU thread binding was not applied because "
            + " and ".join(missing)
            + " is unavailable; preserving PyTorch/runtime defaults."
        )
        print(f"WARNING: {warning}", flush=True)

    return {
        "requested_binding": requested_binding,
        "effective_policy": effective_policy,
        "resolved_binding": resolved_binding,
        "amx_available": amx_available,
        "init_cpu_threads_env_available": init_op is not None,
        "sgl_kernel_initialize_available": initialize_op is not None,
        "torch_num_threads_before": before,
        "torch_num_threads_after": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "physical_cpu_ids_by_node": physical_cpu_ids_by_node,
        "warning": warning,
    }


def collect_environment(
    thread_policy: dict[str, Any], affinity_before: list[int]
) -> dict[str, Any]:
    import psutil
    import torch

    parallel_info = torch.__config__.parallel_info()
    config_text = torch.__config__.show()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cpu": {
            "logical_count": psutil.cpu_count(logical=True),
            "physical_count": psutil.cpu_count(logical=False),
            "numa_snc_count": len(thread_policy["physical_cpu_ids_by_node"]),
            "physical_cpu_ids_by_node": thread_policy["physical_cpu_ids_by_node"],
            "affinity_before": affinity_before,
            "affinity_after": psutil.Process().cpu_affinity(),
        },
        "thread_policy": thread_policy,
        "libraries": {
            "openmp_available": "OpenMP" in parallel_info or "OpenMP" in config_text,
            "mkl_available": "MKL" in parallel_info or "MKL" in config_text,
            "mkldnn_onednn_available": bool(torch.backends.mkldnn.is_available()),
            "parallel_info": parallel_info,
        },
        "environment_variables": {
            name: os.environ.get(name)
            for name in (*CPU_ENVIRONMENT, *THREAD_ENVIRONMENT_VARIABLES)
        },
    }


def build_server_args(model: WorkerModel):
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    return ServerArgs.from_kwargs(
        model_path=str(model.model_path),
        model_id=model.model_id,
        backend="sglang",
        num_gpus=1,
        tp_size=1,
        sp_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        dp_size=1,
        disagg_role="encoder",
        warmup=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        pin_cpu_memory=False,
        use_fsdp_inference=False,
        enable_cfg_parallel=False,
        enable_torch_compile=False,
    )


def build_sampling_params(model: WorkerModel, server_args: Any, run_index: int):
    from sglang.multimodal_gen.configs.sample import SamplingParams

    kwargs: dict[str, Any] = {
        "prompt": model.prompt,
        "width": model.width,
        "height": model.height,
        "num_frames": model.num_frames,
        "fps": model.fps,
        "request_id": f"cpu-encoder-{model.key}-{run_index + 1}",
    }
    if model.reference_image is not None:
        kwargs["image_path"] = str(model.reference_image)
    return SamplingParams.from_user_sampling_params_args(
        str(model.model_path), server_args=server_args, **kwargs
    )


def find_unique_stage(stages: list[Any], stage_type: type) -> Any:
    matches = [stage for stage in stages if isinstance(stage, stage_type)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {stage_type.__name__}, found {len(matches)}"
        )
    return matches[0]


def run_worker(args: argparse.Namespace) -> None:
    for name, value in CPU_ENVIRONMENT.items():
        if os.environ.get(name) != value:
            raise RuntimeError(
                f"Worker must set {name}={value!r} before importing SGLang; "
                f"got {os.environ.get(name)!r}"
            )

    import psutil

    affinity_before = psutil.Process().cpu_affinity()
    requested_binding = args.cpu_core_bind
    thread_policy = configure_cpu_threads(requested_binding)
    environment = collect_environment(thread_policy, affinity_before)

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
    from sglang.multimodal_gen.runtime.pipelines_core import build_pipeline
    from sglang.multimodal_gen.runtime.pipelines_core.stages import (
        InputValidationStage,
        TextEncodingStage,
    )
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args

    model = resolve_worker_model(args)
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(args._master_port),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        }
    )
    server_args = build_server_args(model)
    set_global_server_args(server_args)
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=1,
        cfg_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        dp_size=1,
        distributed_init_method="env://",
        dist_timeout=300,
    )

    try:
        load_started_ns = time.perf_counter_ns()
        pipeline = build_pipeline(server_args)
        load_duration_ms = (time.perf_counter_ns() - load_started_ns) / 1_000_000.0

        expected_required_modules = {
            "text_encoder",
            "tokenizer",
            "scheduler",
        }
        if set(pipeline.required_config_modules) != expected_required_modules:
            raise RuntimeError(
                "Encoder-only pipeline must load exactly text_encoder, tokenizer, "
                f"and scheduler; got {pipeline.required_config_modules}"
            )

        module_tensors = module_tensor_descriptors(pipeline.modules)
        assert_cpu_tensors(module_tensors, "text encoder parameters/buffers")
        io_guard_handles, observed_encoder_io = install_encoder_cpu_io_guards(
            pipeline.modules
        )
        input_stage = find_unique_stage(pipeline.stages, InputValidationStage)
        text_stage = find_unique_stage(pipeline.stages, TextEncodingStage)

        samples: list[dict[str, Any]] = []
        output_summary: list[dict[str, Any]] = []
        try:
            for run_index in range(args.num_runs):
                sampling_started_ns = time.perf_counter_ns()
                sampling_params = build_sampling_params(model, server_args, run_index)
                request = prepare_request(server_args, sampling_params)
                request = input_stage.forward(request, server_args)
                input_preparation_ms = (
                    time.perf_counter_ns() - sampling_started_ns
                ) / 1_000_000.0

                input_tensors = tensor_descriptors(
                    getattr(request, "condition_image", None),
                    "request.condition_image",
                )
                assert_cpu_tensors(input_tensors, "encoder request inputs")

                started_ns = time.perf_counter_ns()
                request = text_stage.forward(request, server_args)
                duration_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
                if run_index == 0:
                    for handle in io_guard_handles:
                        handle.remove()
                    io_guard_handles.clear()

                output_tensors: list[dict[str, Any]] = []
                for field in TextEncodingStage.deduplicated_output_fields:
                    output_tensors.extend(
                        tensor_descriptors(
                            getattr(request, field, None), f"request.{field}"
                        )
                    )
                if not output_tensors:
                    raise RuntimeError("TextEncodingStage produced no tensor outputs")
                assert_cpu_tensors(output_tensors, "text encoder outputs")
                output_summary = output_tensors
                samples.append(
                    {
                        "run": run_index + 1,
                        "warmup": run_index < args.warmup_runs,
                        "duration_ms": duration_ms,
                        "input_preparation_ms": input_preparation_ms,
                        "input_tensors": input_tensors,
                        "output_tensors": output_tensors,
                    }
                )
                print(
                    f"[{model.key}] run={run_index + 1}/{args.num_runs} "
                    f"warmup={run_index < args.warmup_runs} "
                    f"duration_ms={duration_ms:.3f}",
                    flush=True,
                )
        finally:
            for handle in io_guard_handles:
                handle.remove()

        if (
            observed_encoder_io["input_tensor_count"] == 0
            or observed_encoder_io["output_tensor_count"] == 0
        ):
            raise RuntimeError(
                "CPU I/O guards did not observe text encoder input and output tensors"
            )

        measured = [sample for sample in samples if not sample["warmup"]]
        result = {
            "status": "success",
            "model": {
                "key": model.key,
                "label": model.label,
                "model_id": model.model_id,
                "model_path": str(model.model_path),
                "prompt": model.prompt,
                "width": model.width,
                "height": model.height,
                "num_frames": model.num_frames,
                "fps": model.fps,
                "reference_image": (
                    str(model.reference_image)
                    if model.reference_image is not None
                    else None
                ),
            },
            "metric": "TextEncodingStage.forward wall time",
            "load_duration_ms": load_duration_ms,
            "input_preparation_avg_ms": statistics.fmean(
                sample["input_preparation_ms"] for sample in measured
            ),
            "samples": samples,
            "measured_statistics": summarize_samples(
                [sample["duration_ms"] for sample in measured]
            ),
            "loaded_modules": sorted(pipeline.modules),
            "required_config_modules": pipeline.required_config_modules,
            "encoder_parameter_buffer_tensor_count": len(module_tensors),
            "encoder_io_guard_observations": observed_encoder_io,
            "encoder_io_guard_validation_run": 1,
            "output_tensor_summary": output_summary,
            "environment": environment,
        }
        save_json(args._worker_result, result)
    finally:
        cleanup_dist_env_and_memory()


def main() -> None:
    args = parse_args()
    if args._worker_model:
        run_worker(args)
    elif args.dry_run:
        dry_run(args)
    else:
        execute_workers(args)


if __name__ == "__main__":
    main()
