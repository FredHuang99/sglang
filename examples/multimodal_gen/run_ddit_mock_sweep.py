"""Run a matrix of DDiT mock simulator cases and aggregate summary metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from ddit_mock_simulator import (
    DEFAULT_POLICIES,
    DEFAULT_PROFILE_PATH,
    SimulationConfig,
    parse_csv,
    parse_ratios,
    run_simulation,
)


DEFAULT_RATES = ("0.2", "0.6", "1.0", "2.0", "burst")
SUMMARY_METRICS = (
    "p50",
    "p90",
    "p99",
    "mean",
    "max",
    "slo5",
    "slo10",
    "throughput",
    "makespan",
    "completed_count",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-path", default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--model-id", default="z-image")
    parser.add_argument("--num-nodes", type=int, default=2)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_POLICIES),
        help="Comma-separated policies or 'all'.",
    )
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--resolutions", default="720p,2k")
    parser.add_argument("--ratios", default="0.75,0.25")
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--rates", default=",".join(DEFAULT_RATES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--te-resource-mode", choices=("lane", "exclusive"), default="lane"
    )
    parser.add_argument("--out-dir", required=True)
    return parser


def _parse_policies(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(DEFAULT_POLICIES)
    policies = parse_csv(raw)
    unknown = [policy for policy in policies if policy not in DEFAULT_POLICIES]
    if unknown:
        raise ValueError(f"Unsupported policies: {unknown}")
    return policies


def _case_dir(base: Path, rate: str, policy: str) -> Path:
    safe_rate = str(rate).replace(".", "p")
    return base / f"rr_{safe_rate}" / policy


def _write_metric_csv(path: Path, metric: str, rates: list[str], policies: list[str], table: dict[tuple[str, str], dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["request_rate", *policies])
        for rate in rates:
            row = [rate]
            for policy in policies:
                value = table.get((rate, policy), {}).get(metric, "")
                row.append(value)
            writer.writerow(row)


def _write_summary_xlsx_or_csv(
    out_dir: Path,
    rates: list[str],
    policies: list[str],
    table: dict[tuple[str, str], dict[str, Any]],
) -> None:
    try:
        from openpyxl import Workbook
    except Exception:
        for metric in SUMMARY_METRICS:
            _write_metric_csv(
                out_dir / f"summary_{metric}.csv", metric, rates, policies, table
            )
        return

    workbook = Workbook()
    default = workbook.active
    workbook.remove(default)
    for metric in SUMMARY_METRICS:
        sheet = workbook.create_sheet(title=metric[:31])
        sheet.append(["request_rate", *policies])
        for rate in rates:
            sheet.append(
                [
                    rate,
                    *[
                        table.get((rate, policy), {}).get(metric, "")
                        for policy in policies
                    ],
                ]
            )
    workbook.save(out_dir / "summary.xlsx")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    policies = _parse_policies(args.policies)
    rates = parse_csv(args.rates)
    resolutions = tuple(parse_csv(args.resolutions))
    ratios = tuple(parse_ratios(args.ratios))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table: dict[tuple[str, str], dict[str, Any]] = {}
    for rate in rates:
        for policy in policies:
            case_out = _case_dir(out_dir, rate, policy)
            config = SimulationConfig(
                profile_path=args.profile_path,
                model_id=args.model_id,
                policy=policy,
                num_nodes=args.num_nodes,
                gpus_per_node=args.gpus_per_node,
                window_size=args.window_size,
                resolutions=resolutions,
                ratios=ratios,
                num_requests=args.num_requests,
                rate=rate,
                seed=args.seed,
                te_resource_mode=args.te_resource_mode,
                out_dir=str(case_out),
            )
            result = run_simulation(config, write=True)
            table[(rate, policy)] = result["summary"]
            print(
                "case done: "
                f"rate={rate} policy={policy} "
                f"completed={result['summary']['completed_count']} "
                f"p50={result['summary']['p50']:.3f}s "
                f"p99={result['summary']['p99']:.3f}s"
            )

    _write_summary_xlsx_or_csv(out_dir, rates, policies, table)
    print(f"sweep completed: out_dir={out_dir}")


if __name__ == "__main__":
    main()
