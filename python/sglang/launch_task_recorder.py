from __future__ import annotations

import datetime as _dt
import json
import os
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator

_LOG_PATH_ENV = "SGLANG_LAUNCH_TASK_LOG_PATH"
_HTTP_STARTUP_TASKS: dict[tuple[str, str, int | str | None], dict[str, Any]] = {}
_HTTP_STARTUP_LOCK = threading.Lock()


def _log_path() -> Path | None:
    raw = os.environ.get(_LOG_PATH_ENV)
    if not raw:
        return None
    return Path(raw)


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="milliseconds")


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return repr(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _default_rank(rank: int | str | None) -> int | str | None:
    if rank is not None:
        return rank
    for key in ("RANK", "LOCAL_RANK"):
        raw = os.environ.get(key)
        if not raw:
            continue
        try:
            return int(raw)
        except ValueError:
            return raw
    return None


def emit_launch_task_event(
    *,
    task: str,
    phase: str,
    family: str,
    component: str | None = None,
    rank: int | str | None = None,
    elapsed_ms: float | None = None,
    status: str | None = None,
    extra: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    log_path = _log_path()
    if log_path is None:
        return

    payload: dict[str, Any] = {
        "timestamp": _now_iso(),
        "family": family,
        "task": task,
        "phase": phase,
    }
    resolved_rank = _default_rank(rank)
    if resolved_rank is not None:
        payload["rank"] = resolved_rank
    if component is not None:
        payload["component"] = component
    if elapsed_ms is not None:
        payload["elapsed_ms"] = round(float(elapsed_ms), 3)
        payload["elapsed_s"] = round(float(elapsed_ms) / 1000.0, 3)
    if status is not None:
        payload["status"] = status
    if extra:
        payload["extra"] = _to_builtin(extra)
    if error:
        payload["error"] = error

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


@contextmanager
def record_launch_task(
    *,
    task: str,
    family: str,
    component: str | None = None,
    rank: int | str | None = None,
    extra: dict[str, Any] | None = None,
) -> Iterator[None]:
    start_ns = time.perf_counter_ns()
    emit_launch_task_event(
        task=task,
        phase="begin",
        family=family,
        component=component,
        rank=rank,
        extra=extra,
    )
    try:
        yield
    except Exception as exc:
        emit_launch_task_event(
            task=task,
            phase="end",
            family=family,
            component=component,
            rank=rank,
            elapsed_ms=(time.perf_counter_ns() - start_ns) / 1_000_000.0,
            status="error",
            extra=extra,
            error=str(exc),
        )
        raise
    else:
        emit_launch_task_event(
            task=task,
            phase="end",
            family=family,
            component=component,
            rank=rank,
            elapsed_ms=(time.perf_counter_ns() - start_ns) / 1_000_000.0,
            status="ok",
            extra=extra,
        )


def record_launch_task_timing(
    *,
    task: str,
    family: str,
    elapsed_ms: float,
    component: str | None = None,
    rank: int | str | None = None,
    extra: dict[str, Any] | None = None,
    status: str = "ok",
) -> None:
    emit_launch_task_event(
        task=task,
        phase="end",
        family=family,
        component=component,
        rank=rank,
        elapsed_ms=elapsed_ms,
        status=status,
        extra=extra,
    )


def _http_startup_key(
    *,
    task: str,
    family: str,
    rank: int | str | None,
) -> tuple[str, str, int | str | None]:
    return (family, task, _default_rank(rank))


def _finish_http_startup_task(
    *,
    task: str,
    family: str,
    rank: int | str | None = None,
    status: str,
    error: str | None = None,
) -> None:
    key = _http_startup_key(task=task, family=family, rank=rank)
    with _HTTP_STARTUP_LOCK:
        state = _HTTP_STARTUP_TASKS.pop(key, None)
    if state is None:
        return

    emit_launch_task_event(
        task=task,
        phase="end",
        family=family,
        rank=state["rank"],
        elapsed_ms=(time.perf_counter_ns() - state["start_ns"]) / 1_000_000.0,
        status=status,
        extra=state["extra"],
        error=error,
    )


def start_http_startup_probe(
    *,
    family: str,
    url: str,
    task: str = "http_server_startup",
    rank: int | str | None = None,
    extra: dict[str, Any] | None = None,
):
    log_path = _log_path()
    if log_path is None:
        return lambda: None

    resolved_rank = _default_rank(rank)
    payload_extra = dict(extra or {})
    payload_extra.setdefault("url", url)
    key = _http_startup_key(task=task, family=family, rank=resolved_rank)

    with _HTTP_STARTUP_LOCK:
        if key not in _HTTP_STARTUP_TASKS:
            _HTTP_STARTUP_TASKS[key] = {
                "start_ns": time.perf_counter_ns(),
                "rank": resolved_rank,
                "extra": payload_extra,
            }
            emit_launch_task_event(
                task=task,
                phase="begin",
                family=family,
                rank=resolved_rank,
                extra=payload_extra,
            )

    def _cancel() -> None:
        _finish_http_startup_task(
            task=task,
            family=family,
            rank=resolved_rank,
            status="cancelled",
            error=f"Cancelled before {url} became ready",
        )

    return _cancel


def mark_http_startup_ready(
    *,
    family: str,
    task: str = "http_server_startup",
    rank: int | str | None = None,
) -> None:
    _finish_http_startup_task(
        task=task,
        family=family,
        rank=rank,
        status="ok",
    )


def profile_launch_task(
    *,
    task: str,
    family: str,
    component: str | Callable[..., str | None] | None = None,
    rank: int | str | Callable[..., int | str | None] | None = None,
    extra: dict[str, Any] | Callable[..., dict[str, Any] | None] | None = None,
):
    def _resolve(value, *args, **kwargs):
        return value(*args, **kwargs) if callable(value) else value

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            resolved_component = _resolve(component, *args, **kwargs)
            resolved_rank = _resolve(rank, *args, **kwargs)
            resolved_extra = _resolve(extra, *args, **kwargs)
            with record_launch_task(
                task=task,
                family=family,
                component=resolved_component,
                rank=resolved_rank,
                extra=resolved_extra,
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator
