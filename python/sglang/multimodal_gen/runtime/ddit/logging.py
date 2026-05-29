"""Lifecycle and rank-switch logging for DDiT experiments."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch.distributed as dist

from .config import resolve_resolution_key
from .profile import DDiTProfile, ProfileStore

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
SUMMARY_ROW_NAMES = ("p50", "p90", "p99", "slo10", "slo5")


class _LifecycleStatusPriority(IntEnum):
    EMPTY = 0
    QUEUED = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4


_STATUS_PRIORITY = {
    "": _LifecycleStatusPriority.EMPTY,
    "queued": _LifecycleStatusPriority.QUEUED,
    "running": _LifecycleStatusPriority.RUNNING,
    "completed": _LifecycleStatusPriority.COMPLETED,
    "failed": _LifecycleStatusPriority.FAILED,
}


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


@contextmanager
def _interprocess_file_lock(path: str):
    lock_path = f"{path}.lock"
    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)
    with open(lock_path, "a+b") as lock_file:
        lock_file.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _empty_lifecycle_row(request_id: str) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "resolution": "",
        "add_time": "",
        "dit_start_time": "",
        "dit_end_time": "",
        "vae_start_time": "",
        "vae_end_time": "",
        "status": "",
        "error": "",
    }


def _merge_timestamp(
    existing: Any,
    incoming: Any,
    *,
    prefer: str,
) -> str:
    existing_s = str(existing or "")
    incoming_s = str(incoming or "")
    if not existing_s:
        return incoming_s
    if not incoming_s:
        return existing_s
    try:
        existing_f = float(existing_s)
        incoming_f = float(incoming_s)
    except (TypeError, ValueError):
        return incoming_s or existing_s
    selected = (
        min(existing_f, incoming_f)
        if prefer == "earliest"
        else max(existing_f, incoming_f)
    )
    return f"{selected:.6f}"


def _merge_status(existing: Any, incoming: Any) -> str:
    existing_s = str(existing or "")
    incoming_s = str(incoming or "")
    if _STATUS_PRIORITY.get(
        incoming_s, _LifecycleStatusPriority.EMPTY
    ) >= _STATUS_PRIORITY.get(
        existing_s,
        _LifecycleStatusPriority.EMPTY,
    ):
        return incoming_s
    return existing_s


def _merge_lifecycle_rows(
    existing_rows: dict[str, dict[str, Any]],
    incoming_rows: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {
        request_id: {
            column: row.get(column, "") for column in LIFECYCLE_COLUMNS
        }
        for request_id, row in existing_rows.items()
    }
    for request_id, incoming in incoming_rows.items():
        row = merged.setdefault(request_id, _empty_lifecycle_row(request_id))
        row["request_id"] = request_id
        incoming_resolution = str(incoming.get("resolution") or "")
        if incoming_resolution and not str(row.get("resolution") or ""):
            row["resolution"] = incoming_resolution
        elif incoming_resolution and str(row.get("resolution") or "") == "unknown":
            row["resolution"] = incoming_resolution
        for column in ("add_time", "dit_start_time", "vae_start_time"):
            row[column] = _merge_timestamp(
                row.get(column), incoming.get(column), prefer="earliest"
            )
        for column in ("dit_end_time", "vae_end_time"):
            row[column] = _merge_timestamp(
                row.get(column), incoming.get(column), prefer="latest"
            )
        row["status"] = _merge_status(row.get("status"), incoming.get("status"))
        incoming_error = str(incoming.get("error") or "")
        if incoming_error:
            row["error"] = incoming_error
        elif not row.get("error"):
            row["error"] = ""
    return merged


@dataclass
class DDiTLogPaths:
    log_dir: str
    lifecycle_csv: str
    lifecycle_summary_csv: str
    rank_switch_jsonl: str
    op_trace_jsonl: str


class LifecycleCsvLogger:
    """One-row-per-request lifecycle CSV logger."""

    def __init__(
        self,
        path: str,
        summary_path: str | None = None,
        profile: DDiTProfile | None = None,
    ):
        self.path = path
        self.summary_path = summary_path or self._default_summary_path(path)
        self.profile = profile
        self._lock = threading.Lock()
        self._rows: dict[str, dict[str, Any]] = {}
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._load()

    def _load(self) -> None:
        self._rows = self._read_rows_from_disk()

    def _read_rows_from_disk(self) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        if not os.path.exists(self.path):
            return rows
        with open(self.path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                request_id = row.get("request_id")
                if request_id in SUMMARY_ROW_NAMES:
                    continue
                if request_id:
                    rows[request_id] = dict(row)
        return rows

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
        with _interprocess_file_lock(self.path):
            self._rows = _merge_lifecycle_rows(
                self._read_rows_from_disk(), self._rows
            )
            directory = os.path.dirname(self.path) or "."
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
        directory = os.path.dirname(summary_path) or "."
        if directory != ".":
            os.makedirs(directory, exist_ok=True)
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
        completed: list[tuple[dict[str, Any], float]] = []
        for row in self._rows.values():
            try:
                add_time = float(row.get("add_time") or "")
                vae_end_time = float(row.get("vae_end_time") or "")
            except (TypeError, ValueError):
                continue
            if vae_end_time >= add_time:
                completed.append((row, vae_end_time - add_time))

        lifespans = [latency for _row, latency in completed]
        if not lifespans:
            return []
        rows = [
            ("p50", _percentile_nearest_rank(lifespans, 50)),
            ("p90", _percentile_nearest_rank(lifespans, 90)),
            ("p99", _percentile_nearest_rank(lifespans, 99)),
        ]
        rows.extend(self._slo_attainment_rows(completed))
        return rows

    def _slo_attainment_rows(
        self, completed: list[tuple[dict[str, Any], float]]
    ) -> list[tuple[str, float]]:
        if self.profile is None:
            return []

        denominators = {"slo10": 0, "slo5": 0}
        attained = {"slo10": 0, "slo5": 0}
        for row, latency in completed:
            resolution = str(row.get("resolution") or "")
            unit_slo = self.profile.unit_slo(resolution, gpu_count=8)
            if unit_slo is None:
                continue
            thresholds = {"slo10": unit_slo * 10.0, "slo5": unit_slo * 5.0}
            for name, threshold in thresholds.items():
                denominators[name] += 1
                if latency <= threshold:
                    attained[name] += 1

        return [
            (name, attained[name] / denominators[name])
            for name in ("slo10", "slo5")
            if denominators[name] > 0
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
    try:
        profile = ProfileStore.load(server_args)
    except Exception:
        profile = None
    lifecycle = LifecycleCsvLogger(
        paths.lifecycle_csv, paths.lifecycle_summary_csv, profile=profile
    )
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
