# SPDX-License-Identifier: Apache-2.0
"""Launch a single-node two-instance DDiT disaggregated Wan T2V service."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from typing import Any

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.launch_server import (
    launch_disagg_role,
    launch_disagg_server,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


DDIT_POLICIES = (
    "forced_switch",
    "hungry_first",
    "fixed_baseline",
    "naive",
    "naive_greedy",
    "wsjf",
    "wsjf_scale_up",
)


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", default="wan2.1-t2v-1.3b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--scheduler-port", type=int, default=5555)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated physical GPU ids reused by encoder and ddit_worker.",
    )
    parser.add_argument("--encoder-tp", type=int, default=None)
    parser.add_argument("--ddit-worker-sp", type=int, default=None)
    parser.add_argument("--ddit-worker-ulysses", type=int, default=None)
    parser.add_argument("--ddit-worker-ring", type=int, default=1)
    parser.add_argument(
        "--ddit-schedule-policy", choices=DDIT_POLICIES, default="hungry_first"
    )
    parser.add_argument("--ddit-baseline-gpus", type=int, default=1)
    parser.add_argument("--ddit-window-size", type=int, default=8)
    parser.add_argument("--ddit-profile-path", default=None)
    parser.add_argument("--ddit-profile-model-id", default=None)
    parser.add_argument("--ddit-sp-degree-map", default=None)
    parser.add_argument(
        "--ddit-dynamic-sp-prebuild-mode",
        choices=("off", "auto", "plan", "canonical", "all"),
        default="auto",
        help=(
            "Dynamic SP prebuild candidate policy. 'all' is debug-only and "
            "may create many NCCL communicators."
        ),
    )
    parser.add_argument(
        "--ddit-prebuild-sp-groups",
        type=_parse_bool,
        default=None,
        help=(
            "Whether to prebuild DDiT dynamic SP groups. Defaults to false for "
            "forced_switch correctness and true for E2E scheduling policies."
        ),
    )
    parser.add_argument("--ddit-allowed-gpu-counts", default="1,2,4,8")
    parser.add_argument("--ddit-log-dir", default=None)
    parser.add_argument("--disagg-transfer-backend", default="auto")
    parser.add_argument(
        "--disagg-max-slots-per-instance",
        type=int,
        default=None,
        help=(
            "Prepared-payload admission slots per encoder/ddit_worker instance. "
            "Defaults to 1 for non-window policies and --ddit-window-size for "
            "WSJF policies."
        ),
    )
    parser.add_argument(
        "--disagg-transfer-pool-size",
        type=int,
        default=512 * 1024 * 1024,
    )
    parser.add_argument("--disagg-transfer-redundancy", type=float, default=1.25)
    parser.add_argument(
        "--disagg-transfer-pin-memory",
        choices=["auto", "off", "required"],
        default="auto",
    )
    parser.add_argument(
        "--disagg-warmup",
        action="store_true",
        help="Run startup disagg transfer calibration warmup requests.",
    )
    parser.add_argument(
        "--disagg-warmup-resolutions",
        default=None,
        help="Comma-separated WxH warmup resolutions, e.g. 1280x720.",
    )
    parser.add_argument("--disagg-warmup-steps", type=int, default=1)
    parser.add_argument("--disagg-timeout", type=int, default=3600)
    parser.add_argument("--disagg-downstream-wait-timeout", type=int, default=1800)
    parser.add_argument("--log-level", default="info")
    return parser


def _parse_gpu_ids(value: str | None, num_gpus: int) -> list[int]:
    if value is None:
        return list(range(num_gpus))
    tokens = [part for part in value.replace(",", " ").split() if part]
    if not tokens:
        raise ValueError("--gpu-ids must contain at least one id when set.")
    gpu_ids = [int(token) for token in tokens]
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"--gpu-ids contains duplicate ids: {gpu_ids}")
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(f"--gpu-ids must be non-negative: {gpu_ids}")
    return gpu_ids


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def _resolve_disagg_max_slots(args: argparse.Namespace) -> int:
    if args.disagg_max_slots_per_instance is not None:
        return max(1, int(args.disagg_max_slots_per_instance))
    if args.ddit_schedule_policy in ("wsjf", "wsjf_scale_up"):
        return max(1, int(args.ddit_window_size))
    return 1


def _resolve_ddit_prebuild_sp_groups(args: argparse.Namespace) -> bool:
    if args.ddit_prebuild_sp_groups is not None:
        return bool(args.ddit_prebuild_sp_groups)
    if args.ddit_dynamic_sp_prebuild_mode == "off":
        return False
    if args.ddit_dynamic_sp_prebuild_mode in {"all", "canonical", "plan"}:
        return True
    return args.ddit_schedule_policy != "forced_switch"


def _common_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    max_slots = _resolve_disagg_max_slots(args)
    kwargs: dict[str, Any] = {
        "model_path": args.model_path,
        "model_id": args.model_id,
        "host": args.host,
        "log_level": args.log_level,
        "disagg_max_slots_per_instance": max_slots,
        "disagg_transfer_backend": args.disagg_transfer_backend,
        "disagg_transfer_pool_size": args.disagg_transfer_pool_size,
        "disagg_transfer_redundancy": args.disagg_transfer_redundancy,
        "disagg_transfer_pin_memory": args.disagg_transfer_pin_memory,
        "disagg_timeout": args.disagg_timeout,
        "disagg_downstream_wait_timeout": args.disagg_downstream_wait_timeout,
        "warmup": args.disagg_warmup,
        "warmup_resolutions": _parse_csv(args.disagg_warmup_resolutions),
        "warmup_steps": args.disagg_warmup_steps,
        "dit_cpu_offload": False,
        "dit_layerwise_offload": False,
        "text_encoder_cpu_offload": False,
        "image_encoder_cpu_offload": False,
        "vae_cpu_offload": False,
        "pin_cpu_memory": False,
        "enable_ddit": True,
        "ddit_schedule_policy": args.ddit_schedule_policy,
        "ddit_baseline_gpus": args.ddit_baseline_gpus,
        "ddit_window_size": args.ddit_window_size,
        "ddit_allowed_gpu_counts": args.ddit_allowed_gpu_counts,
        "ddit_prebuild_sp_groups": _resolve_ddit_prebuild_sp_groups(args),
        "ddit_dynamic_sp_prebuild_mode": args.ddit_dynamic_sp_prebuild_mode,
        "ddit_log_dir": args.ddit_log_dir,
        "ddit_profile_path": args.ddit_profile_path,
        "ddit_profile_model_id": args.ddit_profile_model_id or args.model_id,
        "ddit_sp_degree_map": args.ddit_sp_degree_map,
    }
    return kwargs


def _make_encoder_args(
    args: argparse.Namespace,
    *,
    gpu_ids: list[int],
    head_endpoint: str,
    work_port: int,
) -> ServerArgs:
    kwargs = _common_kwargs(args)
    kwargs.update(
        {
            "disagg_role": RoleType.ENCODER,
            "disagg_mode": True,
            "disagg_server_addr": head_endpoint,
            "disagg_instance_id": 0,
            "scheduler_port": work_port,
            "port": args.port + 10,
            "num_gpus": len(gpu_ids),
            "gpu_ids": gpu_ids,
            "encoder_tp": args.encoder_tp or len(gpu_ids),
        }
    )
    return ServerArgs.from_kwargs(**kwargs)


def _make_ddit_worker_args(
    args: argparse.Namespace,
    *,
    gpu_ids: list[int],
    head_endpoint: str,
    work_port: int,
) -> ServerArgs:
    full_sp = args.ddit_worker_sp or len(gpu_ids)
    kwargs = _common_kwargs(args)
    kwargs.update(
        {
            "disagg_role": RoleType.DDIT_WORKER,
            "disagg_mode": True,
            "disagg_server_addr": head_endpoint,
            "disagg_instance_id": 0,
            "scheduler_port": work_port,
            "port": args.port + 20,
            "num_gpus": len(gpu_ids),
            "gpu_ids": gpu_ids,
            "denoiser_tp": 1,
            "denoiser_sp": full_sp,
            "denoiser_ulysses": args.ddit_worker_ulysses or full_sp,
            "denoiser_ring": args.ddit_worker_ring,
        }
    )
    return ServerArgs.from_kwargs(**kwargs)


def _make_head_args(
    args: argparse.Namespace,
    *,
    encoder_work_endpoint: str,
    ddit_worker_work_endpoint: str,
) -> ServerArgs:
    kwargs = _common_kwargs(args)
    kwargs.update(
        {
            "disagg_role": RoleType.SERVER,
            "disagg_mode": True,
            "host": args.host,
            "port": args.port,
            "scheduler_port": args.scheduler_port,
            "encoder_urls": encoder_work_endpoint,
            "ddit_worker_urls": ddit_worker_work_endpoint,
        }
    )
    return ServerArgs.from_kwargs(**kwargs)


def _tcp_endpoint(host: str, port: int) -> str:
    return f"tcp://{host}:{port}"


def _assert_no_disagg_port_collisions(
    head_args: ServerArgs,
    encoder_args: ServerArgs,
    ddit_worker_args: ServerArgs,
) -> None:
    port_labels = [
        (head_args.scheduler_port, "head frontend"),
        (head_args.scheduler_port + 1, "encoder result"),
        (head_args.scheduler_port + 2, "denoiser result"),
        (head_args.scheduler_port + 3, "decoder result"),
        (head_args.scheduler_port + 4, "ddit_worker result"),
        (encoder_args.scheduler_port, "encoder work"),
        (encoder_args.scheduler_port + 1, "encoder control"),
        (ddit_worker_args.scheduler_port, "ddit_worker work"),
        (ddit_worker_args.scheduler_port + 1, "ddit_worker control"),
    ]
    seen: dict[int, str] = {}
    collisions: list[str] = []
    for port, label in port_labels:
        previous = seen.get(port)
        if previous is not None:
            collisions.append(f"{port}: {previous} vs {label}")
        else:
            seen[port] = label
    if collisions:
        raise ValueError(
            "DDiT disagg port collision detected after ServerArgs port settling: "
            + "; ".join(collisions)
        )


def _resolve_launch_args(
    args: argparse.Namespace, gpu_ids: list[int]
) -> tuple[ServerArgs, ServerArgs, ServerArgs]:
    provisional_encoder_work_port = args.scheduler_port + 10
    provisional_ddit_worker_work_port = args.scheduler_port + 20
    head_args = _make_head_args(
        args,
        encoder_work_endpoint=_tcp_endpoint(args.host, provisional_encoder_work_port),
        ddit_worker_work_endpoint=_tcp_endpoint(
            args.host, provisional_ddit_worker_work_port
        ),
    )

    actual_head_endpoint = _tcp_endpoint(head_args.host, head_args.scheduler_port)
    encoder_args = _make_encoder_args(
        args,
        gpu_ids=gpu_ids,
        head_endpoint=actual_head_endpoint,
        work_port=head_args.scheduler_port + 10,
    )
    ddit_worker_args = _make_ddit_worker_args(
        args,
        gpu_ids=gpu_ids,
        head_endpoint=actual_head_endpoint,
        work_port=head_args.scheduler_port + 20,
    )
    head_args.encoder_urls = _tcp_endpoint(args.host, encoder_args.scheduler_port)
    head_args.ddit_worker_urls = _tcp_endpoint(
        args.host, ddit_worker_args.scheduler_port
    )
    _assert_no_disagg_port_collisions(head_args, encoder_args, ddit_worker_args)

    return encoder_args, ddit_worker_args, head_args


def _terminate(processes: list[mp.Process], timeout_s: float = 5.0) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.time() + timeout_s
    for process in processes:
        remaining = max(0.0, deadline - time.time())
        process.join(timeout=remaining)


def main() -> None:
    args = build_parser().parse_args()
    gpu_ids = _parse_gpu_ids(args.gpu_ids, args.num_gpus)
    encoder_args, ddit_worker_args, head_args = _resolve_launch_args(args, gpu_ids)

    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(
            target=launch_disagg_role,
            args=(encoder_args,),
            name="sglang-ddit-encoder-instance",
            daemon=False,
        ),
        ctx.Process(
            target=launch_disagg_role,
            args=(ddit_worker_args,),
            name="sglang-ddit-worker-instance",
            daemon=False,
        ),
    ]

    for process in processes:
        process.start()

    try:
        launch_disagg_server(head_args)
    finally:
        _terminate(processes)


if __name__ == "__main__":
    main()
