"""Launch a 4-GPU monolithic Wan2.2 TI2V server."""

from __future__ import annotations

import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/data/Wan2_2_TI2V_5B")
    parser.add_argument("--model-id", default="Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30010)
    parser.add_argument("--scheduler-port", type=int, default=30020)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--base-gpu-id", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--sp-degree", type=int, default=4)
    parser.add_argument("--ulysses-degree", type=int, default=4)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument("--warmup", action="store_true", default=True)
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    #if args.num_gpus != 4:
    #    raise ValueError("This launcher is fixed to a 4-GPU topology.")

    if args.disable_warmup:
        args.warmup = False

    gpu_group = list(range(args.base_gpu_id, args.base_gpu_id + args.num_gpus))
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
    print(f"  visible_gpus    : {gpu_group}")
    print("  topology        : monolithic sp=4 ulysses=4 ring=1")
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
