# SPDX-License-Identifier: Apache-2.0
"""Structured profiling helpers for Wan TI2V benchmarking."""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

DEFAULT_PROFILE_OUTPUT_DIR = "/data/profile"
LOGICAL_STAGE_NAMES = ("encoder", "denoiser", "decoder")


def ensure_dir(path: str | os.PathLike[str]) -> str:
    abs_path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id:
        return str(run_id)
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_profile_dir(
    output_dir: str | None,
    run_id: str | None = None,
    deployment_mode: str | None = None,
    traffic_mode: str | None = None,
) -> str:
    base_dir = output_dir or DEFAULT_PROFILE_OUTPUT_DIR
    base_dir = ensure_dir(base_dir)

    suffix_parts = [resolve_run_id(run_id)]
    if deployment_mode:
        suffix_parts.append(str(deployment_mode))
    if traffic_mode:
        suffix_parts.append(str(traffic_mode))

    return ensure_dir(os.path.join(base_dir, "_".join(suffix_parts)))


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(float(v) for v in values)
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[low]
    low_value = sorted_values[low]
    high_value = sorted_values[high]
    return low_value + (high_value - low_value) * (rank - low)


def summarize_percentiles(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
    }


def format_metric_value(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"


def build_summary_lines(
    *,
    e2e_latency_ms: list[float],
    logical_stage_duration_ms: dict[str, list[float]],
    throughput_rps: float | None,
) -> list[str]:
    lines: list[str] = []
    for label, values in [
        ("e2e_latency_ms", e2e_latency_ms),
        ("encoder_duration_ms", logical_stage_duration_ms.get("encoder", [])),
        ("denoiser_duration_ms", logical_stage_duration_ms.get("denoiser", [])),
        ("decoder_duration_ms", logical_stage_duration_ms.get("decoder", [])),
    ]:
        stats = summarize_percentiles(values)
        lines.append(f"{label}.P50={format_metric_value(stats['p50'])}")
        lines.append(f"{label}.P90={format_metric_value(stats['p90'])}")
        lines.append(f"{label}.P99={format_metric_value(stats['p99'])}")
    lines.append(f"throughput_rps={format_metric_value(throughput_rps)}")
    return lines


def summarize_throughput(
    request_count: int,
    *,
    first_arrival_time_s: float | None,
    last_finish_time_s: float | None,
) -> float | None:
    if (
        request_count <= 0
        or first_arrival_time_s is None
        or last_finish_time_s is None
        or last_finish_time_s <= first_arrival_time_s
    ):
        return None
    return request_count / (last_finish_time_s - first_arrival_time_s)


def _coerce_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


class CsvProfileWriter:
    """Append rows to CSV while allowing later rows to introduce new columns."""

    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(os.path.expanduser(file_path))
        self._lock = threading.Lock()
        self._fieldnames: list[str] = []

    @property
    def fieldnames(self) -> list[str]:
        return list(self._fieldnames)

    def write_row(self, row: dict[str, Any]) -> None:
        with self._lock:
            normalized = {key: _coerce_csv_value(value) for key, value in row.items()}
            new_fields = [
                fieldname
                for fieldname in normalized
                if fieldname not in self._fieldnames
            ]
            if new_fields:
                self._fieldnames.extend(new_fields)
                self._rewrite_with_existing_rows(normalized)
                return

            self._append(normalized)

    def _append(self, row: dict[str, str]) -> None:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        needs_header = not os.path.exists(self.file_path) or os.path.getsize(
            self.file_path
        ) == 0
        with open(self.file_path, "a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=self._fieldnames)
            if needs_header:
                writer.writeheader()
            writer.writerow({name: row.get(name, "") for name in self._fieldnames})

    def _rewrite_with_existing_rows(self, row: dict[str, str]) -> None:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        existing_rows: list[dict[str, str]] = []
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            with open(self.file_path, "r", newline="", encoding="utf-8") as fp:
                existing_rows = list(csv.DictReader(fp))

        with open(self.file_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=self._fieldnames)
            writer.writeheader()
            for existing_row in existing_rows:
                writer.writerow(
                    {name: existing_row.get(name, "") for name in self._fieldnames}
                )
            writer.writerow({name: row.get(name, "") for name in self._fieldnames})


class RequestCsvProfiler:
    """Accumulate per-request fields and flush them to CSV."""

    def __init__(self, file_path: str, finalized_cache_size: int = 65536):
        self._writer = CsvProfileWriter(file_path)
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}
        self._finalized_request_ids: OrderedDict[str, None] = OrderedDict()
        self._finalized_cache_size = max(0, int(finalized_cache_size))

    @property
    def file_path(self) -> str:
        return self._writer.file_path

    def update(self, request_id: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            if request_id in self._finalized_request_ids:
                return {"request_id": request_id}
            record = self._records.setdefault(
                "{}".format(request_id), {"request_id": request_id}
            )
            record.update(fields)
            return dict(record)

    def _remember_finalized(self, request_id: str) -> None:
        if self._finalized_cache_size <= 0:
            return
        self._finalized_request_ids[request_id] = None
        self._finalized_request_ids.move_to_end(request_id)
        while len(self._finalized_request_ids) > self._finalized_cache_size:
            self._finalized_request_ids.popitem(last=False)

    def finalize(
        self,
        request_id: str,
        *,
        status: str,
        error: str | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            if request_id in self._finalized_request_ids:
                return
            record = self._records.pop(
                "{}".format(request_id), {"request_id": request_id}
            )
            self._remember_finalized(request_id)
        record["status"] = status
        if error is not None:
            record["error"] = error
        if extra_fields:
            record.update(extra_fields)
        self._writer.write_row(record)

    def write_row(self, row: dict[str, Any]) -> None:
        self._writer.write_row(row)


def flatten_request_metrics(metrics) -> dict[str, Any]:
    if metrics is None:
        return {}

    arrival_time_s = getattr(metrics, "arrival_time_s", None)
    start_time_s = getattr(metrics, "start_time_s", None)
    finish_time_s = getattr(metrics, "finish_time_s", None)
    total_duration_ms = getattr(metrics, "total_duration_ms", None)
    row: dict[str, Any] = {
        "request_id": getattr(metrics, "request_id", ""),
        "arrival_time_s": arrival_time_s,
        "start_time_s": start_time_s,
        "finish_time_s": finish_time_s,
        "total_duration_ms": total_duration_ms,
    }
    if (
        arrival_time_s is not None
        and start_time_s is not None
        and start_time_s >= arrival_time_s
    ):
        row["queue_duration_ms"] = (start_time_s - arrival_time_s) * 1000.0
    if (
        arrival_time_s is not None
        and finish_time_s is not None
        and finish_time_s >= arrival_time_s
    ):
        row["e2e_duration_ms"] = (finish_time_s - arrival_time_s) * 1000.0

    logical = aggregate_logical_stage_durations(metrics)
    for stage_name, duration_ms in logical.items():
        row[f"logical_{stage_name}_duration_ms"] = duration_ms
    total_stage_duration_ms = sum(logical.values())
    if total_duration_ms is not None:
        row["unattributed_duration_ms"] = max(
            0.0, float(total_duration_ms) - total_stage_duration_ms
        )

    for stage_name, duration_ms in getattr(metrics, "stages", {}).items():
        row[f"raw_stage_{stage_name}_duration_ms"] = duration_ms

    for index, duration_ms in enumerate(getattr(metrics, "steps", [])):
        row[f"raw_denoising_step_{index}_duration_ms"] = duration_ms

    return row


def aggregate_logical_stage_durations(metrics) -> dict[str, float]:
    stages = dict(getattr(metrics, "stages", {}) or {})
    steps = list(getattr(metrics, "steps", []) or [])

    decoder_ms = 0.0
    denoiser_ms = 0.0
    encoder_ms = 0.0

    for stage_name, duration_ms in stages.items():
        stage_key = stage_name.lower()
        duration_ms = float(duration_ms)
        if "decod" in stage_key or "vae" in stage_key:
            decoder_ms += duration_ms
        elif "denois" in stage_key:
            denoiser_ms += duration_ms
        else:
            encoder_ms += duration_ms

    if denoiser_ms <= 0.0 and steps:
        denoiser_ms = sum(float(step_ms) for step_ms in steps)

    return {
        "encoder": encoder_ms,
        "denoiser": denoiser_ms,
        "decoder": decoder_ms,
    }
