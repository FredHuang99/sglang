"""Lifecycle and rank-switch logging for DDiT experiments."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist

from .config import resolve_resolution_key

LIFECYCLE_COLUMNS = [
    "request_id",
    "resolution",
    "add_time",
    "dit_start_time",
    "dit_end_time",
    "vae_start_time",
    "vae_end_time",
    "status",
    "error",
]
LIFECYCLE_SUMMARY_COLUMNS = ["metric", "value"]
SUMMARY_ROW_NAMES = ("p50", "p90", "p99")


def _is_rank_zero() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _now() -> float:
    return time.time()


def _percentile_nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, int((percentile / 100.0) * len(ordered) + 0.999999999))
    return ordered[min(rank, len(ordered)) - 1]


@dataclass
class DDiTLogPaths:
    log_dir: str
    lifecycle_csv: str
    lifecycle_summary_csv: str
    rank_switch_jsonl: str
    op_trace_jsonl: str


class LifecycleCsvLogger:
    """One-row-per-request lifecycle CSV logger."""

    def __init__(self, path: str, summary_path: str | None = None):
        self.path = path
        self.summary_path = summary_path or self._default_summary_path(path)
        self._lock = threading.Lock()
        self._rows: dict[str, dict[str, Any]] = {}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                request_id = row.get("request_id")
                if request_id in SUMMARY_ROW_NAMES:
                    continue
                if request_id:
                    self._rows[request_id] = dict(row)

    def record(
        self,
        *,
        request_id: str,
        resolution: str,
        event: str,
        timestamp: float | None = None,
        status: str | None = None,
        error: str | None = None,
    ) -> None:
        timestamp = _now() if timestamp is None else timestamp
        event_to_column = {
            "add": "add_time",
            "dit_start": "dit_start_time",
            "dit_end": "dit_end_time",
            "vae_start": "vae_start_time",
            "vae_end": "vae_end_time",
        }
        if event not in event_to_column:
            raise ValueError(f"Unsupported DDiT lifecycle event: {event}")

        with self._lock:
            row = self._rows.setdefault(
                request_id,
                {
                    "request_id": request_id,
                    "resolution": resolution,
                    "add_time": "",
                    "dit_start_time": "",
                    "dit_end_time": "",
                    "vae_start_time": "",
                    "vae_end_time": "",
                    "status": "running",
                    "error": "",
                },
            )
            row["resolution"] = resolution or row.get("resolution") or "unknown"
            row[event_to_column[event]] = f"{timestamp:.6f}"
            if status:
                row["status"] = status
            elif event == "vae_end":
                row["status"] = "completed"
            if error:
                row["error"] = error
                row["status"] = "failed"
            self._flush_locked()

    def _flush_locked(self) -> None:
        directory = os.path.dirname(self.path)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".ddit_lifecycle_", suffix=".csv", dir=directory
        )
        os.close(fd)
        try:
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=LIFECYCLE_COLUMNS)
                writer.writeheader()
                for request_id in sorted(self._rows):
                    writer.writerow(
                        {
                            column: self._rows[request_id].get(column, "")
                            for column in LIFECYCLE_COLUMNS
                        }
                    )
            os.replace(tmp_path, self.path)
            self._flush_summary_locked()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _flush_summary_locked(self) -> None:
        summary_path = self.summary_path
        directory = os.path.dirname(summary_path)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".ddit_lifecycle_summary_", suffix=".csv", dir=directory
        )
        os.close(fd)
        try:
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=LIFECYCLE_SUMMARY_COLUMNS)
                writer.writeheader()
                for name, value in self._lifespan_summary_rows():
                    writer.writerow({"metric": name, "value": f"{value:.6f}"})
            os.replace(tmp_path, summary_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def _default_summary_path(path: str) -> str:
        root, ext = os.path.splitext(path)
        return f"{root}_summary{ext or '.csv'}"

    def _lifespan_summary_rows(self) -> list[tuple[str, float]]:
        lifespans = []
        for row in self._rows.values():
            try:
                add_time = float(row.get("add_time") or "")
                vae_end_time = float(row.get("vae_end_time") or "")
            except (TypeError, ValueError):
                continue
            if vae_end_time >= add_time:
                lifespans.append(vae_end_time - add_time)

        if not lifespans:
            return []
        return [
            ("p50", _percentile_nearest_rank(lifespans, 50)),
            ("p90", _percentile_nearest_rank(lifespans, 90)),
            ("p99", _percentile_nearest_rank(lifespans, 99)),
        ]


class RankSwitchJsonlLogger:
    """Append-only rank transition logger."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def record(
        self,
        *,
        request_id: str,
        resolution: str,
        node_id: str,
        stage: str,
        step: int | None,
        old_ranks: tuple[int, ...] | list[int],
        new_ranks: tuple[int, ...] | list[int],
        reason: str,
        policy: str,
        timestamp: float | None = None,
    ) -> None:
        payload = {
            "timestamp": _now() if timestamp is None else timestamp,
            "request_id": request_id,
            "resolution": resolution,
            "node_id": node_id,
            "stage": stage,
            "step": step,
            "old_ranks": list(old_ranks),
            "new_ranks": list(new_ranks),
            "reason": reason,
            "policy": policy,
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=True) + "\n")


class OpTraceJsonlLogger:
    """Append-only rank operation trace logger written by rank 0."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def record_many(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, sort_keys=True) + "\n")


_LOGGER_CACHE: dict[
    str, tuple[LifecycleCsvLogger, RankSwitchJsonlLogger, OpTraceJsonlLogger]
] = {}


def resolve_log_paths(server_args: Any) -> DDiTLogPaths:
    log_dir = getattr(server_args, "ddit_log_dir", None) or os.path.join(
        getattr(server_args, "output_path", None) or "outputs", "ddit_logs"
    )
    lifecycle_csv = os.path.join(log_dir, "ddit_lifecycle.csv")
    lifecycle_summary_csv = os.path.join(log_dir, "ddit_lifecycle_summary.csv")
    rank_switch_jsonl = os.path.join(log_dir, "ddit_rank_switch.jsonl")
    op_trace_jsonl = os.path.join(log_dir, "ddit_op_trace.jsonl")
    return DDiTLogPaths(
        log_dir=log_dir,
        lifecycle_csv=lifecycle_csv,
        lifecycle_summary_csv=lifecycle_summary_csv,
        rank_switch_jsonl=rank_switch_jsonl,
        op_trace_jsonl=op_trace_jsonl,
    )


def get_loggers(
    server_args: Any,
) -> tuple[LifecycleCsvLogger, RankSwitchJsonlLogger, OpTraceJsonlLogger]:
    paths = resolve_log_paths(server_args)
    cached = _LOGGER_CACHE.get(paths.log_dir)
    if cached is not None:
        return cached
    lifecycle = LifecycleCsvLogger(paths.lifecycle_csv, paths.lifecycle_summary_csv)
    switches = RankSwitchJsonlLogger(paths.rank_switch_jsonl)
    op_trace = OpTraceJsonlLogger(paths.op_trace_jsonl)
    _LOGGER_CACHE[paths.log_dir] = (lifecycle, switches, op_trace)
    return lifecycle, switches, op_trace


def record_lifecycle(
    server_args: Any,
    batch: Any,
    event: str,
    *,
    timestamp: float | None = None,
    status: str | None = None,
    error: str | None = None,
) -> None:
    if not getattr(server_args, "enable_ddit", False) or not _is_rank_zero():
        return
    request_id = str(getattr(batch, "request_id", None) or "unknown")
    resolution = resolve_resolution_key(batch)
    lifecycle, _, _ = get_loggers(server_args)
    lifecycle.record(
        request_id=request_id,
        resolution=resolution,
        event=event,
        timestamp=timestamp,
        status=status,
        error=error,
    )


def record_rank_switch(
    server_args: Any,
    batch: Any,
    *,
    stage: str,
    step: int | None,
    old_ranks: tuple[int, ...] | list[int],
    new_ranks: tuple[int, ...] | list[int],
    reason: str,
    policy: str,
) -> None:
    if not getattr(server_args, "enable_ddit", False) or not _is_rank_zero():
        return
    _, switch_logger, _ = get_loggers(server_args)
    switch_logger.record(
        request_id=str(getattr(batch, "request_id", None) or "unknown"),
        resolution=resolve_resolution_key(batch),
        node_id=str(getattr(server_args, "ddit_node_id", "node0")),
        stage=stage,
        step=step,
        old_ranks=old_ranks,
        new_ranks=new_ranks,
        reason=reason,
        policy=policy,
    )


def record_op_trace_rows(server_args: Any, rows: list[dict[str, Any]]) -> None:
    if not getattr(server_args, "enable_ddit", False) or not _is_rank_zero():
        return
    _, _, op_trace = get_loggers(server_args)
    op_trace.record_many(rows)
