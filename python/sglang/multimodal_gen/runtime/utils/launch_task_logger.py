# SPDX-License-Identifier: Apache-2.0
"""Low-overhead launch task timing for diffusion cold-start benchmarks."""

from __future__ import annotations

import datetime as _datetime
import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Generator

_ENV_KEY = "SGLANG_LAUNCH_TASK_LOG_PATH"
_LOCK = threading.Lock()
_MARKS: dict[str, float] = {}


def enabled() -> bool:
    return bool(os.environ.get(_ENV_KEY))


def _log_path() -> str | None:
    path = os.environ.get(_ENV_KEY)
    return path or None


def _default_rank() -> int | None:
    try:
        return int(os.environ["RANK"])
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def mark_task_start(key: str) -> None:
    if not enabled():
        return
    _MARKS[key] = time.perf_counter()


def finish_marked_task(
    key: str,
    task: str,
    *,
    component: str | None = None,
    rank: int | None = None,
    status: str = "ok",
    extra: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    start_perf = _MARKS.pop(key, None)
    record_task(
        task,
        start_perf=start_perf,
        component=component,
        rank=rank,
        status=status,
        extra=extra,
        error=error,
    )


def record_task(
    task: str,
    *,
    start_perf: float | None = None,
    component: str | None = None,
    rank: int | None = None,
    status: str = "ok",
    extra: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    path = _log_path()
    if not path:
        return

    now_perf = time.perf_counter()
    elapsed_ms = 0.0 if start_perf is None else (now_perf - start_perf) * 1000.0
    timestamp_s = time.time()
    record = {
        "timestamp": _datetime.datetime.fromtimestamp(timestamp_s).isoformat(
            timespec="milliseconds"
        ),
        "timestamp_s": timestamp_s,
        "task": task,
        "component": component,
        "rank": _default_rank() if rank is None else rank,
        "status": status,
        "elapsed_ms": max(0.0, elapsed_ms),
        "elapsed_s": max(0.0, elapsed_ms) / 1000.0,
        "extra": _json_safe(extra or {}),
        "error": error,
        "pid": os.getpid(),
    }
    line = json.dumps(record, sort_keys=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with _LOCK:
        with open(path, "a", encoding="utf-8") as fp:
            fp.write(line)
            fp.write("\n")


@contextmanager
def launch_task(
    task: str,
    *,
    component: str | None = None,
    rank: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Generator[None, None, None]:
    if not enabled():
        yield
        return
    start_perf = time.perf_counter()
    try:
        yield
    except Exception as exc:
        record_task(
            task,
            start_perf=start_perf,
            component=component,
            rank=rank,
            status="error",
            extra=extra,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    else:
        record_task(
            task,
            start_perf=start_perf,
            component=component,
            rank=rank,
            status="ok",
            extra=extra,
        )
