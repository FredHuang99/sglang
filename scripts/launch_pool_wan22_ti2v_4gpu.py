"""Launch pooled or split 4-GPU disaggregated Wan2.2 TI2V deployments."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from typing import Any

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.launch_server import (
    kill_process_tree,
    launch_disagg_role,
    launch_disagg_server,
    launch_pool_disagg_server,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.disagg_launcher_utils import (
    resolve_disagg_ib_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/data/Wan2_2_TI2V_5B")
    parser.add_argument("--model-id", default="Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--deployment-layout", choices=["single_host", "split_two_hosts"], default="single_host")
    parser.add_argument("--node-role", choices=["all_in_one", "machine_a", "machine_b"], default="all_in_one")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30010)
    parser.add_argument("--scheduler-port", type=int, default=30020)
    parser.add_argument("--machine-a-host", default="10.3.4.3")
    parser.add_argument("--machine-b-host", default="10.3.4.2")
    parser.add_argument("--encoder-scheduler-port", type=int, default=None)
    parser.add_argument("--denoiser-scheduler-port", type=int, default=None)
    parser.add_argument("--decoder-scheduler-port", type=int, default=None)
    parser.add_argument("--encoder-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--encoder-base-gpu-id", type=int, default=4)
    parser.add_argument("--encoder-num-gpus", type=int, default=4)
    parser.add_argument(
        "--encoder-gpu-ids",
        nargs="+",
        default=None,
        help=(
            "Physical encoder GPU IDs, e.g. --encoder-gpu-ids 0 1 6 7 "
            "or --encoder-gpu-ids 0,1,6,7. Overrides encoder base/num."
        ),
    )
    parser.add_argument("--denoiser-base-gpu-id", type=int, default=4)
    parser.add_argument("--denoiser-num-gpus", type=int, default=4)
    parser.add_argument(
        "--denoiser-gpu-ids",
        nargs="+",
        default=None,
        help=(
            "Physical denoiser GPU IDs, e.g. --denoiser-gpu-ids 0 1 6 7 "
            "or --denoiser-gpu-ids 0,1,6,7. Overrides denoiser base/num."
        ),
    )
    parser.add_argument("--decoder-base-gpu-id", type=int, default=4)
    parser.add_argument("--decoder-num-gpus", type=int, default=4)
    parser.add_argument(
        "--decoder-gpu-ids",
        nargs="+",
        default=None,
        help=(
            "Physical decoder GPU IDs, e.g. --decoder-gpu-ids 0 1 6 7 "
            "or --decoder-gpu-ids 0,1,6,7. Overrides decoder base/num."
        ),
    )
    parser.add_argument("--encoder-tp", type=int, default=None)
    parser.add_argument("--denoiser-sp", type=int, default=None)
    parser.add_argument("--denoiser-ulysses", type=int, default=None)
    parser.add_argument("--denoiser-ring", type=int, default=1)
    parser.add_argument("--decoder-sp", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument("--disagg-role-device", type=str, default="cuda")
    parser.add_argument("--disagg-transfer-backend", type=str, default=None)
    parser.add_argument("--encoder-transfer-backend", type=str, default=None)
    parser.add_argument("--denoiser-transfer-backend", type=str, default=None)
    parser.add_argument("--decoder-transfer-backend", type=str, default=None)
    parser.add_argument("--encoder-ib-device", type=str, default="auto")
    parser.add_argument("--denoiser-ib-device", type=str, default="auto")
    parser.add_argument("--decoder-ib-device", type=str, default="auto")
    parser.add_argument("--disagg-dispatch-policy", type=str, default="round_robin")
    parser.add_argument("--disagg-max-slots-per-instance", type=int, default=8)
    parser.add_argument("--disagg-timeout", type=int, default=3600)
    parser.add_argument("--disagg-downstream-wait-timeout", type=int, default=3600)
    parser.add_argument("--warmup", action="store_true", default=False)
    parser.add_argument("--disable-warmup", action="store_true")
    parser.add_argument("--profile-enabled", action="store_true")
    parser.add_argument("--profile-output-dir", type=str, default="/data/profile")
    parser.add_argument("--profile-run-id", type=str, default="disagg_wan2_2_ti2v_5b_4gpu")
    parser.add_argument(
        "--text-encoder-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--image-encoder-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--vae-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--dit-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--dit-layerwise-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--pin-cpu-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _parse_gpu_ids(values: list[str] | str | None, *, flag_name: str) -> list[int] | None:
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]

    tokens: list[str] = []
    for value in values:
        tokens.extend(part for part in str(value).replace(",", " ").split() if part)
    if not tokens:
        raise ValueError(f"{flag_name} requires at least one GPU id.")

    gpu_ids: list[int] = []
    for token in tokens:
        try:
            gpu_id = int(token)
        except ValueError as exc:
            raise ValueError(f"{flag_name} contains a non-integer GPU id: {token}") from exc
        if gpu_id < 0:
            raise ValueError(f"{flag_name} GPU ids must be non-negative: {gpu_id}")
        gpu_ids.append(gpu_id)

    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"{flag_name} contains duplicate GPU ids: {gpu_ids}")
    return gpu_ids


def _build_gpu_group(base_gpu_id: int, num_gpus: int) -> list[int]:
    if num_gpus <= 0:
        return []
    return list(range(base_gpu_id, base_gpu_id + num_gpus))


def _resolve_role_gpu_ids(args: argparse.Namespace, role_name: str) -> list[int]:
    explicit_gpu_ids = _parse_gpu_ids(
        getattr(args, f"{role_name}_gpu_ids", None),
        flag_name=f"--{role_name}-gpu-ids",
    )
    if explicit_gpu_ids is not None:
        return explicit_gpu_ids
    return _build_gpu_group(
        getattr(args, f"{role_name}_base_gpu_id", 0),
        getattr(args, f"{role_name}_num_gpus", 0),
    )


def _resolve_encoder_gpu_ids(args: argparse.Namespace) -> list[int]:
    explicit_gpu_ids = _parse_gpu_ids(
        getattr(args, "encoder_gpu_ids", None), flag_name="--encoder-gpu-ids"
    )
    if args.encoder_device == "cpu":
        if explicit_gpu_ids is not None:
            raise ValueError("--encoder-gpu-ids cannot be used with encoder-device=cpu.")
        return []
    if explicit_gpu_ids is not None:
        return explicit_gpu_ids
    return _build_gpu_group(
        getattr(args, "encoder_base_gpu_id", 0),
        getattr(args, "encoder_num_gpus", 0),
    )


def _resolve_role_gpu_count(args: argparse.Namespace, role_name: str) -> int:
    if role_name == "encoder":
        return len(_resolve_encoder_gpu_ids(args))
    return len(_resolve_role_gpu_ids(args, role_name))


def _resolve_bind_host(args: argparse.Namespace) -> str:
    if args.deployment_layout == "split_two_hosts" and args.host == "127.0.0.1":
        return "0.0.0.0"
    return args.host


def _resolve_encoder_tp(args: argparse.Namespace) -> int:
    if args.encoder_device == "cpu":
        if args.encoder_tp not in (None, 1):
            raise ValueError("encoder_tp must be 1 when encoder-device=cpu.")
        _resolve_encoder_gpu_ids(args)
        return 1
    return args.encoder_tp or max(1, _resolve_role_gpu_count(args, "encoder"))


def _resolve_denoiser_sp(args: argparse.Namespace) -> int:
    return args.denoiser_sp or max(1, _resolve_role_gpu_count(args, "denoiser"))


def _resolve_denoiser_ulysses(args: argparse.Namespace) -> int:
    return args.denoiser_ulysses or _resolve_denoiser_sp(args)


def _resolve_decoder_sp(args: argparse.Namespace) -> int:
    return args.decoder_sp or max(1, _resolve_role_gpu_count(args, "decoder"))


def _resolve_role_port(explicit_port: int | None, scheduler_port: int, offset: int) -> int:
    return explicit_port if explicit_port is not None else scheduler_port + offset


def _resolve_role_transfer_backend(args: argparse.Namespace, role_name: str) -> str:
    explicit = getattr(args, f"{role_name}_transfer_backend")
    if explicit is not None:
        return explicit
    if args.disagg_transfer_backend is not None:
        return args.disagg_transfer_backend
    if args.deployment_layout == "single_host":
        return "mock"
    return "mock" if role_name == "encoder" else "auto"


def _resolve_role_ib_device(
    requested: str | None,
    *,
    host: str,
) -> str | None:
    return resolve_disagg_ib_device(requested, host=host)


def _resolve_pool_transfer_backend(args: argparse.Namespace) -> str:
    if args.disagg_transfer_backend is not None:
        return args.disagg_transfer_backend
    for role_name in ("encoder", "denoiser", "decoder"):
        explicit = getattr(args, f"{role_name}_transfer_backend")
        if explicit is not None:
            return explicit
    return "mock"


def _resolve_pool_ib_device(args: argparse.Namespace, *, host: str) -> str | None:
    for role_name in ("denoiser", "decoder", "encoder"):
        explicit = getattr(args, f"{role_name}_ib_device")
        resolved = _resolve_role_ib_device(explicit, host=host)
        if resolved is not None:
            return resolved
    return None


def _build_common_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_path": args.model_path,
        "model_id": args.model_id,
        "log_level": args.log_level,
        "encoder_tp": _resolve_encoder_tp(args),
        "denoiser_sp": _resolve_denoiser_sp(args),
        "denoiser_ulysses": _resolve_denoiser_ulysses(args),
        "denoiser_ring": args.denoiser_ring,
        "decoder_sp": _resolve_decoder_sp(args),
        "text_encoder_cpu_offload": args.text_encoder_cpu_offload,
        "image_encoder_cpu_offload": args.image_encoder_cpu_offload,
        "vae_cpu_offload": args.vae_cpu_offload,
        "dit_layerwise_offload": args.dit_layerwise_offload,
        "dit_cpu_offload": args.dit_cpu_offload,
        "pin_cpu_memory": args.pin_cpu_memory,
        "disagg_dispatch_policy": args.disagg_dispatch_policy,
        "disagg_max_slots_per_instance": args.disagg_max_slots_per_instance,
        "disagg_timeout": args.disagg_timeout,
        "disagg_downstream_wait_timeout": args.disagg_downstream_wait_timeout,
        "warmup": args.warmup,
        "profile_enabled": args.profile_enabled,
        "profile_output_dir": args.profile_output_dir,
        "profile_run_id": args.profile_run_id,
    }


def _build_pool_server_args(args: argparse.Namespace) -> ServerArgs:
    bind_host = _resolve_bind_host(args)
    kwargs = _build_common_kwargs(args)
    kwargs.update(
        {
            "host": bind_host,
            "port": args.port,
            "scheduler_port": args.scheduler_port,
            "disagg_role_device": args.disagg_role_device,
            "disagg_transfer_backend": _resolve_pool_transfer_backend(args),
            "disagg_p2p_hostname": bind_host,
            "disagg_ib_device": _resolve_pool_ib_device(args, host=bind_host),
        }
    )
    return ServerArgs.from_kwargs(**kwargs)


def _build_role_server_args(
    args: argparse.Namespace,
    *,
    role: RoleType,
    host: str,
    scheduler_port: int,
    gpu_ids: list[int],
    disagg_role_device: str,
    disagg_server_addr: str,
    transfer_backend: str,
    ib_device: str | None,
) -> ServerArgs:
    is_cpu_role = role == RoleType.ENCODER and disagg_role_device == "cpu"
    kwargs = _build_common_kwargs(args)
    kwargs.update(
        {
            "host": host,
            "scheduler_port": scheduler_port,
            "num_gpus": 0 if is_cpu_role else len(gpu_ids),
            "base_gpu_id": gpu_ids[0] if gpu_ids else 0,
            "gpu_ids": None if is_cpu_role else gpu_ids,
            "disagg_role": role,
            "disagg_server_addr": disagg_server_addr,
            "disagg_role_device": disagg_role_device,
            "disagg_transfer_backend": transfer_backend,
            "disagg_p2p_hostname": host,
            "disagg_ib_device": ib_device,
        }
    )
    return ServerArgs.from_kwargs(**kwargs)


def _build_head_server_args(args: argparse.Namespace) -> ServerArgs:
    bind_host = _resolve_bind_host(args)
    encoder_port = _resolve_role_port(args.encoder_scheduler_port, args.scheduler_port, 100)
    denoiser_port = _resolve_role_port(args.denoiser_scheduler_port, args.scheduler_port, 200)
    decoder_port = _resolve_role_port(args.decoder_scheduler_port, args.scheduler_port, 300)
    kwargs = _build_common_kwargs(args)
    kwargs.update(
        {
            "host": bind_host,
            "port": args.port,
            "scheduler_port": args.scheduler_port,
            "disagg_role": RoleType.SERVER,
            "encoder_urls": f"tcp://{args.machine_a_host}:{encoder_port}",
            "denoiser_urls": f"tcp://{args.machine_a_host}:{denoiser_port}",
            "decoder_urls": f"tcp://{args.machine_b_host}:{decoder_port}",
        }
    )
    return ServerArgs.from_kwargs(**kwargs)


def _spawn_role_process(server_args: ServerArgs, name: str) -> mp.Process:
    ctx = mp.get_context("spawn")
    process = ctx.Process(
        target=launch_disagg_role,
        args=(server_args,),
        name=name,
        daemon=False,
    )
    process.start()
    return process


def _cleanup_processes(processes: list[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            try:
                kill_process_tree(process.pid)
            except Exception:
                process.terminate()
    for process in processes:
        try:
            process.join(timeout=5)
        except Exception:
            pass


def _launch_single_host(args: argparse.Namespace) -> None:
    server_args = _build_pool_server_args(args)
    encoder_gpus = (
        [[]]
        if args.encoder_device == "cpu"
        else [_resolve_encoder_gpu_ids(args)]
    )
    denoiser_gpus = [_resolve_role_gpu_ids(args, "denoiser")]
    decoder_gpus = [_resolve_role_gpu_ids(args, "decoder")]

    print("Launching pooled disaggregated Wan2.2 TI2V server")
    print(f"  model_path      : {args.model_path}")
    print(f"  model_id        : {args.model_id}")
    print(f"  host/http_port  : {_resolve_bind_host(args)}:{args.port}")
    print(f"  scheduler_port  : {args.scheduler_port}")
    print(f"  encoder_device  : {args.encoder_device}")
    print(f"  encoder_gpus    : {encoder_gpus}")
    print(f"  denoiser_gpus   : {denoiser_gpus}")
    print(f"  decoder_gpus    : {decoder_gpus}")
    print(
        "  topology        : encoder "
        f"{'cpu' if args.encoder_device == 'cpu' else f'tp={_resolve_encoder_tp(args)}'}"
        f" | denoiser sp={_resolve_denoiser_sp(args)} ulysses={_resolve_denoiser_ulysses(args)} ring={args.denoiser_ring}"
        f" | decoder sp={_resolve_decoder_sp(args)}"
    )
    if args.profile_enabled:
        print(f"  profile_output  : {args.profile_output_dir}")
        print(f"  profile_run_id  : {args.profile_run_id}")
    print(f"  server_warmup   : {args.warmup}")
    if not args.warmup:
        print("  benchmark warmup: use benchmark --num-warmup-requests")

    launch_pool_disagg_server(
        server_args,
        encoder_gpus=encoder_gpus,
        denoiser_gpus=denoiser_gpus,
        decoder_gpus=decoder_gpus,
    )


def _launch_split_machine_a(args: argparse.Namespace) -> None:
    encoder_port = _resolve_role_port(args.encoder_scheduler_port, args.scheduler_port, 100)
    denoiser_port = _resolve_role_port(args.denoiser_scheduler_port, args.scheduler_port, 200)
    server_addr = f"tcp://{args.machine_a_host}:{args.scheduler_port}"
    encoder_gpu_ids = _resolve_encoder_gpu_ids(args)
    denoiser_gpu_ids = _resolve_role_gpu_ids(args, "denoiser")

    encoder_args = _build_role_server_args(
        args,
        role=RoleType.ENCODER,
        host=args.machine_a_host,
        scheduler_port=encoder_port,
        gpu_ids=encoder_gpu_ids,
        disagg_role_device=args.encoder_device,
        disagg_server_addr=server_addr,
        transfer_backend=_resolve_role_transfer_backend(args, "encoder"),
        ib_device=_resolve_role_ib_device(args.encoder_ib_device, host=args.machine_a_host),
    )
    denoiser_args = _build_role_server_args(
        args,
        role=RoleType.DENOISER,
        host=args.machine_a_host,
        scheduler_port=denoiser_port,
        gpu_ids=denoiser_gpu_ids,
        disagg_role_device=args.disagg_role_device,
        disagg_server_addr=server_addr,
        transfer_backend=_resolve_role_transfer_backend(args, "denoiser"),
        ib_device=_resolve_role_ib_device(args.denoiser_ib_device, host=args.machine_a_host),
    )
    head_args = _build_head_server_args(args)

    role_processes = [
        _spawn_role_process(encoder_args, "wan-disagg-encoder-a"),
        _spawn_role_process(denoiser_args, "wan-disagg-denoiser-a"),
    ]

    print("Launching split two-host Wan2.2 TI2V deployment (machine_a)")
    print(f"  machine_a_host  : {args.machine_a_host}")
    print(f"  machine_b_host  : {args.machine_b_host}")
    print(f"  bind_host       : {_resolve_bind_host(args)}")
    print(f"  http_port       : {args.port}")
    print(f"  server_addr     : {server_addr}")
    print(f"  encoder_port    : {encoder_port} ({args.encoder_device})")
    print(f"  denoiser_port   : {denoiser_port}")
    print(f"  encoder_gpus    : {encoder_gpu_ids if encoder_gpu_ids else 'cpu'}")
    print(f"  denoiser_gpus   : {denoiser_gpu_ids}")
    print(
        "  encoder         : "
        + ("cpu" if args.encoder_device == "cpu" else f"tp={_resolve_encoder_tp(args)}")
    )
    print(
        "  denoiser        : "
        f"sp={_resolve_denoiser_sp(args)} ulysses={_resolve_denoiser_ulysses(args)} ring={args.denoiser_ring}"
    )
    print(f"  decoder_remote  : tcp://{args.machine_b_host}:{_resolve_role_port(args.decoder_scheduler_port, args.scheduler_port, 300)}")
    print(f"  server_warmup   : {args.warmup}")
    if not args.warmup:
        print("  benchmark warmup: use benchmark --num-warmup-requests")

    try:
        launch_disagg_server(head_args)
    finally:
        _cleanup_processes(role_processes)


def _launch_split_machine_b(args: argparse.Namespace) -> None:
    decoder_port = _resolve_role_port(args.decoder_scheduler_port, args.scheduler_port, 300)
    decoder_gpu_ids = _resolve_role_gpu_ids(args, "decoder")
    decoder_args = _build_role_server_args(
        args,
        role=RoleType.DECODER,
        host=args.machine_b_host,
        scheduler_port=decoder_port,
        gpu_ids=decoder_gpu_ids,
        disagg_role_device=args.disagg_role_device,
        disagg_server_addr=f"tcp://{args.machine_a_host}:{args.scheduler_port}",
        transfer_backend=_resolve_role_transfer_backend(args, "decoder"),
        ib_device=_resolve_role_ib_device(args.decoder_ib_device, host=args.machine_b_host),
    )

    print("Launching split two-host Wan2.2 TI2V deployment (machine_b)")
    print(f"  machine_a_host  : {args.machine_a_host}")
    print(f"  machine_b_host  : {args.machine_b_host}")
    print(f"  decoder_port    : {decoder_port}")
    print(f"  decoder         : sp={_resolve_decoder_sp(args)}")
    print(f"  decoder_gpus    : {decoder_gpu_ids}")
    print(f"  server_warmup   : {args.warmup}")

    launch_disagg_role(decoder_args)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.disable_warmup:
        args.warmup = False

    if args.deployment_layout == "single_host":
        if args.node_role != "all_in_one":
            raise ValueError("single_host layout only supports --node-role all_in_one.")
        _launch_single_host(args)
        return

    if args.node_role == "machine_a":
        _launch_split_machine_a(args)
        return
    if args.node_role == "machine_b":
        _launch_split_machine_b(args)
        return
    raise ValueError(
        "split_two_hosts layout requires --node-role machine_a or machine_b."
    )


if __name__ == "__main__":
    main()
