"""Command line entrypoint for ShiftServe validation tooling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sglang.shiftserve.client import (
    build_request_specs,
    dominant_csv_to_traffic_json,
)
from sglang.shiftserve.config import (
    BinConfig,
    load_deployment_config,
    load_profile_config,
    load_traffic_config,
)
from sglang.shiftserve.docs import generate_cli_markdown, generate_cookbook_markdown
from sglang.shiftserve.launcher import (
    LaunchCommandBuilder,
    LaunchDefaults,
    iter_launch_table,
)
from sglang.shiftserve.router import ShiftServeRequest, ShiftServeRouter
from sglang.shiftserve.scheduler import SchedulerMode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ShiftServe PE/diffusion flip tooling")
    sub = parser.add_subparsers(dest="cmd", required=True)

    validate = sub.add_parser("validate-config")
    validate.add_argument("--deployment-json", required=True)
    validate.add_argument("--profile-json", required=True)
    validate.add_argument("--traffic-json", required=True)

    launch = sub.add_parser("launch-plan")
    launch.add_argument("--deployment-json", required=True)
    launch.add_argument("--pe-model-path", required=True)
    launch.add_argument("--diffusion-model-path", required=True)
    launch.add_argument("--server-addr", required=True)
    launch.add_argument("--weighted-schedule", action=argparse.BooleanOptionalAction, default=False)
    _add_common_launch_args(launch)

    simulate = sub.add_parser("simulate")
    simulate.add_argument("--traffic-json", required=True)
    simulate.add_argument("--mode", choices=["no_flip", "can_flip", "manual_flip"], default="no_flip")
    simulate.add_argument("--weighted-schedule", action=argparse.BooleanOptionalAction, default=False)
    simulate.add_argument("--window-size", type=int, default=16)
    simulate.add_argument("--margin-enabled", action=argparse.BooleanOptionalAction, default=False)
    simulate.add_argument("--margin-ratio", type=float, default=0.0)
    simulate.add_argument("--request-rate", type=float)
    simulate.add_argument("--out-dir", required=True)

    convert = sub.add_parser("traffic-from-csv")
    convert.add_argument("--dominant-intervals-csv", required=True)
    convert.add_argument("--out-json", required=True)
    convert.add_argument("--duration-min", type=float)
    convert.add_argument("--default-rate-per-min", type=float, default=1.0)

    docs = sub.add_parser("write-docs")
    docs.add_argument("--output-dir", default=str(Path.home() / "Desktop"))
    return parser


def _add_common_launch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rank0-broadcast",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--transfer-pool-size",
        type=int,
        default=LaunchDefaults.transfer_pool_size,
    )
    parser.add_argument(
        "--transfer-pin-memory",
        choices=["auto", "off", "required"],
        default=LaunchDefaults.transfer_pin_memory,
    )
    parser.add_argument(
        "--max-slots-per-instance",
        type=int,
        default=LaunchDefaults.max_slots_per_instance,
    )
    parser.add_argument("--disagg-timeout", type=int, default=LaunchDefaults.disagg_timeout)
    parser.add_argument("--disagg-downstream-timeout", type=int, default=LaunchDefaults.disagg_downstream_timeout)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "validate-config":
        deployment = load_deployment_config(args.deployment_json)
        profile = load_profile_config(args.profile_json)
        traffic = load_traffic_config(args.traffic_json)
        print(
            json.dumps(
                {
                    "nodes": len(deployment.nodes),
                    "instances": len(deployment.instances),
                    "profile_keys": sorted(profile.raw.keys()),
                    "traffic_intervals": len(traffic.intervals),
                    "bins": {
                        "short": deployment.bins.short,
                        "long": deployment.bins.long,
                    },
                },
                indent=2,
            )
        )
        return

    if args.cmd == "launch-plan":
        deployment = load_deployment_config(args.deployment_json)
        defaults = LaunchDefaults(
            transfer_pool_size=args.transfer_pool_size,
            max_slots_per_instance=args.max_slots_per_instance,
            transfer_pin_memory=args.transfer_pin_memory,
            disagg_timeout=args.disagg_timeout,
            disagg_downstream_timeout=args.disagg_downstream_timeout,
            rank0_broadcast=args.rank0_broadcast,
        )
        builder = LaunchCommandBuilder(defaults)
        commands = builder.build_launch_plan(
            deployment,
            pe_model_path=args.pe_model_path,
            diffusion_model_path=args.diffusion_model_path,
            server_addr=args.server_addr,
            weighted_schedule=args.weighted_schedule,
        )
        for instance_id, command in iter_launch_table(commands):
            print(f"[{instance_id}] {command}")
        return

    if args.cmd == "simulate":
        traffic = load_traffic_config(args.traffic_json)
        specs = build_request_specs(traffic, bins=BinConfig())
        mode = SchedulerMode.from_weighted_flag(args.weighted_schedule)
        router = ShiftServeRouter.for_dry_run(out_dir=args.out_dir, mode=mode)
        for spec in specs:
            router.submit(
                ShiftServeRequest(
                    request_id=spec.request_id,
                    input_tokens=spec.input_tokens,
                    output_tokens=spec.output_tokens,
                    bin_name=spec.bin_name,
                )
            )
        router.metrics.write_events()
        print(json.dumps(router.metrics.write_summary(), indent=2))
        return

    if args.cmd == "traffic-from-csv":
        dominant_csv_to_traffic_json(
            args.dominant_intervals_csv,
            args.out_json,
            duration_min=args.duration_min,
            default_rate_per_min=args.default_rate_per_min,
        )
        return

    if args.cmd == "write-docs":
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        generate_cli_markdown(out / "shiftserve_cli.md")
        generate_cookbook_markdown(out / "shiftserve_cookbook.md")
        print(out)
        return


if __name__ == "__main__":
    main()
