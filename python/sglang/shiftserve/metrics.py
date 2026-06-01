"""Request event logging and summary metrics for ShiftServe."""

from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RequestEvent:
    request_id: str
    event: str
    timestamp_s: float
    stage: str | None = None
    instance_id: str | None = None
    bundle_id: str | None = None
    dit_slot_id: int | None = None
    vae_slot_id: int | None = None
    attempt_id: int = 0
    generated_tokens: int | None = None
    completed_steps: int | None = None
    reason: str | None = None


class MetricsRecorder:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._events: list[RequestEvent] = []

    def mark(
        self,
        request_id: str,
        event: str,
        *,
        stage: str | None = None,
        instance_id: str | None = None,
        bundle_id: str | None = None,
        dit_slot_id: int | None = None,
        vae_slot_id: int | None = None,
        attempt_id: int = 0,
        generated_tokens: int | None = None,
        completed_steps: int | None = None,
        reason: str | None = None,
        timestamp_s: float | None = None,
    ) -> RequestEvent:
        evt = RequestEvent(
            request_id=request_id,
            event=event,
            timestamp_s=time.time() if timestamp_s is None else float(timestamp_s),
            stage=stage,
            instance_id=instance_id,
            bundle_id=bundle_id,
            dit_slot_id=dit_slot_id,
            vae_slot_id=vae_slot_id,
            attempt_id=attempt_id,
            generated_tokens=generated_tokens,
            completed_steps=completed_steps,
            reason=reason,
        )
        self._events.append(evt)
        return evt

    @property
    def events(self) -> list[RequestEvent]:
        return list(self._events)

    def write_events(self) -> None:
        jsonl_path = self.out_dir / "request_events.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

        csv_path = self.out_dir / "request_events.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self._events[0]).keys()) if self._events else [
                "request_id",
                "event",
                "timestamp_s",
                "stage",
                "instance_id",
                "bundle_id",
                "dit_slot_id",
                "vae_slot_id",
                "attempt_id",
                "generated_tokens",
                "completed_steps",
                "reason",
            ])
            writer.writeheader()
            for event in self._events:
                writer.writerow(asdict(event))

    def write_summary(self, *, slo_unit_s: float | None = None) -> dict[str, Any]:
        summary = summarize_events(self._events, slo_unit_s=slo_unit_s)
        path = self.out_dir / "run_summary.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)
        return summary


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def summarize_events(
    events: list[RequestEvent],
    *,
    slo_unit_s: float | None = None,
) -> dict[str, Any]:
    by_request: dict[str, dict[str, float]] = {}
    for event in events:
        item = by_request.setdefault(event.request_id, {})
        if event.event == "system_enter":
            item["start"] = event.timestamp_s
        elif event.event in {"request_done", "finish", "vae_end"}:
            item["end"] = max(item.get("end", event.timestamp_s), event.timestamp_s)

    latencies = [
        item["end"] - item["start"]
        for item in by_request.values()
        if "start" in item and "end" in item and item["end"] >= item["start"]
    ]
    if events:
        lifespan = max(e.timestamp_s for e in events) - min(e.timestamp_s for e in events)
    else:
        lifespan = 0.0
    throughput = len(latencies) / lifespan if lifespan > 0 else 0.0
    slo10 = 0.0
    slo5 = 0.0
    if slo_unit_s and latencies:
        slo10 = sum(1 for v in latencies if v <= slo_unit_s * 10) / len(latencies)
        slo5 = sum(1 for v in latencies if v <= slo_unit_s * 5) / len(latencies)

    return {
        "p50": statistics.median(latencies) if latencies else 0.0,
        "p90": _percentile(latencies, 90),
        "p99": _percentile(latencies, 99),
        "throughput": throughput,
        "lifespan": lifespan,
        "slo10_attainment": slo10,
        "slo5_attainment": slo5,
    }
