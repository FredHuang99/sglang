"""Launch a 4-GPU monolithic Wan2.2 TI2V server."""

from __future__ import annotations

import argparse
import os
import sys


ZIMAGE_NUM_ATTENTION_HEADS = 30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/data/Wan2_2_TI2V_5B")
    parser.add_argument("--model-id", default="Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30010)
    parser.add_argument("--scheduler-port", type=int, default=30020)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--base-gpu-id", type=int, default=4)
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        default=None,
        help=(
            "Physical GPU IDs to expose, e.g. --gpu-ids 0 1 6 7 or "
            "--gpu-ids 0,1,6,7. Overrides --base-gpu-id/--num-gpus."
        ),
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--sp-degree", type=int, default=4)
    parser.add_argument("--ulysses-degree", type=int, default=4)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument("--warmup", action="store_true", default=False)
    parser.add_argument("--disable-warmup", action="store_true")
    parser.add_argument("--profile-enabled", action="store_true")
    parser.add_argument("--profile-output-dir", type=str, default="/data/profile")
    parser.add_argument("--profile-run-id", type=str, default="mono_wan2_2_ti2v_5b_4gpu")
    parser.add_argument(
        "--text-encoder-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--image-encoder-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--vae-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
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


def _resolve_gpu_group(args: argparse.Namespace) -> list[int]:
    explicit_gpu_ids = _parse_gpu_ids(args.gpu_ids, flag_name="--gpu-ids")
    if explicit_gpu_ids is not None:
        return explicit_gpu_ids
    if args.num_gpus <= 0:
        raise ValueError("--num-gpus must be positive.")
    return list(range(args.base_gpu_id, args.base_gpu_id + args.num_gpus))


def _arg_was_provided(argv: list[str], flag_name: str) -> bool:
    return any(arg == flag_name or arg.startswith(f"{flag_name}=") for arg in argv)


def _is_zimage_model(args: argparse.Namespace) -> bool:
    candidate = " ".join(
        part for part in (args.model_id, args.model_path) if part
    ).lower()
    normalized = candidate.replace("_", "-")
    return "z-image" in normalized or "zimage" in normalized


def _preferred_zimage_ulysses_degree(sp_degree: int) -> int:
    for candidate in range(sp_degree, 0, -1):
        if sp_degree % candidate == 0 and ZIMAGE_NUM_ATTENTION_HEADS % candidate == 0:
            return candidate
    return 1


def _apply_zimage_topology_defaults(
    args: argparse.Namespace, raw_argv: list[str]
) -> None:
    topology_flags = ("--sp-degree", "--ulysses-degree", "--ring-degree")
    if any(_arg_was_provided(raw_argv, flag) for flag in topology_flags):
        return

    # Z-Image has 30 attention heads, so the 4-GPU Wan default
    # ulysses=4 is invalid. Keep SP at 4 GPUs but split it as ring=2 x ulysses=2.
    args.ulysses_degree = _preferred_zimage_ulysses_degree(args.sp_degree)
    args.ring_degree = args.sp_degree // args.ulysses_degree


def _validate_zimage_topology(args: argparse.Namespace) -> None:
    if args.ulysses_degree <= 0:
        raise ValueError("--ulysses-degree must be positive.")
    if ZIMAGE_NUM_ATTENTION_HEADS % args.ulysses_degree != 0:
        raise ValueError(
            "Invalid Z-Image parallelism: "
            f"Z-Image has {ZIMAGE_NUM_ATTENTION_HEADS} attention heads, so "
            f"--ulysses-degree must divide {ZIMAGE_NUM_ATTENTION_HEADS}. "
            f"Got --ulysses-degree {args.ulysses_degree}. "
            "For 4 GPUs use --sp-degree 4 --ulysses-degree 2 --ring-degree 2."
        )


def main() -> None:
    parser = build_parser()
    raw_argv = sys.argv[1:]
    args = parser.parse_args()
    #if args.num_gpus != 4:
    #    raise ValueError("This launcher is fixed to a 4-GPU topology.")

    if args.disable_warmup:
        args.warmup = False

    if _is_zimage_model(args):
        _apply_zimage_topology_defaults(args, raw_argv)
        _validate_zimage_topology(args)

    gpu_group = _resolve_gpu_group(args)
    args.num_gpus = len(gpu_group)

    dit_parallel_size = args.tp_size * args.sp_degree
    if args.num_gpus < dit_parallel_size:
        raise ValueError(
            "Invalid monolithic parallelism: "
            f"num_gpus={args.num_gpus} but tp_size*sp_degree={dit_parallel_size}. "
            "Use --tp-size 1 --sp-degree 4 for 4-GPU SP, or "
            "--tp-size 4 --sp-degree 1 for 4-GPU TP."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpu_group)

    from sglang.multimodal_gen.runtime.launch_server import launch_server
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    server_args = ServerArgs.from_kwargs(
        model_path=args.model_path,
        model_id=args.model_id,
        host=args.host,
        port=args.port,
        scheduler_port=args.scheduler_port,
        num_gpus=args.num_gpus,
        tp_size=args.tp_size,
        sp_degree=args.sp_degree,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        warmup=args.warmup,
        log_level=args.log_level,
        profile_enabled=args.profile_enabled,
        profile_output_dir=args.profile_output_dir,
        profile_run_id=args.profile_run_id,
        text_encoder_cpu_offload=args.text_encoder_cpu_offload,
        image_encoder_cpu_offload=args.image_encoder_cpu_offload,
        vae_cpu_offload=args.vae_cpu_offload,
        dit_cpu_offload=args.dit_cpu_offload,
        dit_layerwise_offload=args.dit_layerwise_offload,
        pin_cpu_memory=args.pin_cpu_memory,
    )

    print("Launching monolithic Wan2.2 TI2V server")
    print(f"  model_path      : {args.model_path}")
    print(f"  model_id        : {args.model_id}")
    print(f"  host/http_port  : {args.host}:{args.port}")
    print(f"  scheduler_port  : {args.scheduler_port}")
    print(f"  physical_gpus   : {gpu_group}")
    print(f"  logical_ranks   : {list(range(args.num_gpus))}")
    print(
        "  topology        : "
        f"monolithic tp={args.tp_size} sp={args.sp_degree} "
        f"ulysses={args.ulysses_degree} ring={args.ring_degree}"
    )
    print(f"  server_warmup   : {args.warmup}")
    if not args.warmup:
        print("  benchmark warmup: use benchmark --num-warmup-requests")
    print(f"  text_offload    : {args.text_encoder_cpu_offload}")
    print(f"  image_offload   : {args.image_encoder_cpu_offload}")
    print(f"  vae_offload     : {args.vae_cpu_offload}")
    print(f"  dit_offload     : {args.dit_cpu_offload}")
    print(f"  dit_layerwise   : {args.dit_layerwise_offload}")
    if args.profile_enabled:
        print(f"  profile_output  : {args.profile_output_dir}")
        print(f"  profile_run_id  : {args.profile_run_id}")

    launch_server(server_args)


if __name__ == "__main__":
    main()
