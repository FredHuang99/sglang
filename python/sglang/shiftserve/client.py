"""Traffic generation utilities for ShiftServe validation runs."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from sglang.shiftserve.config import BinConfig, TrafficConfig, load_traffic_config


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    scheduled_time_s: float
    bin_name: str
    output_tokens: int
    input_tokens: int


def build_request_specs(
    traffic: TrafficConfig,
    *,
    bins: BinConfig | None = None,
    default_input_tokens: int = 128,
) -> list[RequestSpec]:
    bins = bins or BinConfig()
    specs: list[RequestSpec] = []
    intervals = traffic.intervals
    if not intervals:
        intervals = [
            type("_Interval", (), {
                "start_min": 0.0,
                "end_min": traffic.duration_min,
                "rate_per_min": traffic.default_rate_per_min,
                "bin": "short",
                "input_tokens": None,
            })()
        ]
    counter = 0
    for interval in intervals:
        duration_s = max(0.0, (interval.end_min - interval.start_min) * 60.0)
        if interval.rate_per_min <= 0 or duration_s <= 0:
            continue
        gap_s = 60.0 / interval.rate_per_min
        n = int(duration_s / gap_s)
        for idx in range(n):
            bin_name = interval.bin
            output_tokens = bins.long if bin_name == "long" else bins.short
            specs.append(
                RequestSpec(
                    request_id=f"shiftserve-{counter:08d}",
                    scheduled_time_s=interval.start_min * 60.0 + idx * gap_s,
                    bin_name=bin_name,
                    output_tokens=output_tokens,
                    input_tokens=interval.input_tokens or default_input_tokens,
                )
            )
            counter += 1
    return specs


def dominant_csv_to_traffic_json(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    duration_min: float | None = None,
    default_rate_per_min: float = 1.0,
) -> None:
    intervals = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = float(row.get("start_min") or row.get("minute") or row.get("start") or 0)
            end = float(row.get("end_min") or row.get("end") or (start + 1))
            label = str(row.get("bin") or row.get("dominant") or row.get("type") or "short")
            rate = float(row.get("rate_per_min") or row.get("rate") or default_rate_per_min)
            intervals.append(
                {
                    "start_min": start,
                    "end_min": end,
                    "rate_per_min": rate,
                    "bin": "long" if "long" in label.lower() or "2048" in label else "short",
                }
            )
    payload = {
        "duration_min": duration_min
        if duration_min is not None
        else max((item["end_min"] for item in intervals), default=1.0),
        "default_rate_per_min": default_rate_per_min,
        "intervals": intervals,
    }
    with Path(out_path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def send_requests(specs: list[RequestSpec], server_url: str) -> None:
    started = time.monotonic()
    for spec in specs:
        wait_s = spec.scheduled_time_s - (time.monotonic() - started)
        if wait_s > 0:
            time.sleep(wait_s)
        body = json.dumps(asdict(spec)).encode("utf-8")
        req = urllib.request.Request(
            server_url.rstrip("/") + "/v1/shiftserve/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3600) as resp:
            resp.read()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ShiftServe traffic client")
    parser.add_argument("--traffic-json", required=True)
    parser.add_argument("--server-url")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-json")
    args = parser.parse_args(argv)

    traffic = load_traffic_config(args.traffic_json)
    specs = build_request_specs(traffic)
    if args.out_json:
        with Path(args.out_json).open("w", encoding="utf-8") as f:
            json.dump([asdict(spec) for spec in specs], f, indent=2)
    if args.dry_run or not args.server_url:
        print(json.dumps([asdict(spec) for spec in specs], indent=2))
        return
    send_requests(specs, args.server_url)


if __name__ == "__main__":
    main()
