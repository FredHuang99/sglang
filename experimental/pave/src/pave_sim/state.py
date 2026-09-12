# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime state and analytic token/step boundaries adapted from the reference simulator."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .profiles import PEMatch
from .timing import rounded_ratio, seconds, ticks

STAGES = ("PE", "TE", "DiT", "VAE")


@dataclass
class Request:
    id: int
    arrival_s: float
    input_tokens: int
    output_tokens: int  # Execution truth; never passed to a scheduling policy.
    kind: str
    stage_index: int = 0
    generated: int = 0
    steps: int = 0
    finished_s: float | None = None
    owner: str | None = None

    @property
    def stage(self) -> str | None:
        return STAGES[self.stage_index] if self.stage_index < len(STAGES) else None


@dataclass(frozen=True)
class Progress:
    """The policy-visible part of a request; deliberately excludes final PE length."""
    request_id: int
    generated: int
    steps: int


@dataclass(init=False)
class Execution:
    request_id: int
    attempt_id: int
    stage: str
    start_tick: int
    duration_ticks: int
    initial_progress: int = 0
    target_progress: int = 0
    pe: PEMatch | None = None
    full_dit_ticks: int = 0
    end_tick: int | None = None
    completed: bool = False

    def __init__(self, request_id, attempt_id, stage, start_s, duration_s,
                 initial_progress=0, target_progress=0, pe=None, step_s=0.0,
                 end_s=None, completed=False, *, full_dit_s=None):
        self.request_id, self.attempt_id, self.stage = request_id, attempt_id, stage
        self.initial_progress, self.target_progress = initial_progress, target_progress
        self.start_tick = ticks(start_s)
        self.pe, self.completed = pe, completed
        self.end_tick = None if end_s is None else ticks(end_s)
        self.full_dit_ticks = 0
        count = target_progress - initial_progress
        if stage == "PE":
            if pe is None or count <= 0:
                raise ValueError("PE execution requires a match and remaining tokens")
            self.ttft_ticks = ticks(pe.ttft_s, positive=True)
            self.tpot_ticks = ticks(pe.tpot_s, positive=True)
            self.duration_ticks = self.ttft_ticks + (count - 1) * self.tpot_ticks
        elif stage == "DiT":
            ticks(step_s, positive=True)
            self.full_dit_ticks = ticks(full_dit_s if full_dit_s is not None else step_s * 50, positive=True)
            if count <= 0 or self.full_dit_ticks < 50:
                raise ValueError("DiT steps must each resolve to a positive tick interval")
            self.duration_ticks = rounded_ratio(count * self.full_dit_ticks, 50)
        else:
            self.duration_ticks = ticks(duration_s, positive=True)

    @property
    def start_s(self):
        return seconds(self.start_tick)

    @property
    def duration_s(self):
        return seconds(self.duration_ticks)

    @property
    def step_s(self):
        return seconds(self.full_dit_ticks) / 50

    @property
    def end_s(self):
        return None if self.end_tick is None else seconds(self.end_tick)

    @end_s.setter
    def end_s(self, value):
        self.end_tick = None if value is None else ticks(value)

    @property
    def finish_tick(self):
        return self.start_tick + self.duration_ticks

    @property
    def finish_s(self) -> float:
        return seconds(self.finish_tick)

    def _step_offset(self, count: int) -> int:
        return rounded_ratio(count * self.full_dit_ticks, 50)

    def progress_at(self, time_s: float) -> int:
        stop = min(ticks(time_s), self.finish_tick, self.end_tick if self.end_tick is not None else self.finish_tick)
        elapsed = stop - self.start_tick
        if elapsed < 0:
            return self.initial_progress
        if self.stage == "PE":
            if elapsed < self.ttft_ticks:
                return self.initial_progress
            increment = 1 + (elapsed - self.ttft_ticks) // self.tpot_ticks
        elif self.stage == "DiT":
            lo, hi = 0, self.target_progress - self.initial_progress
            while lo < hi:
                middle = (lo + hi + 1) // 2
                if self._step_offset(middle) <= elapsed:
                    lo = middle
                else:
                    hi = middle - 1
            increment = lo
        else:
            increment = int(stop >= self.finish_tick)
        return min(self.target_progress, self.initial_progress + increment)

    def boundary(self, now_s: float) -> float:
        return seconds(self.boundary_tick(ticks(now_s)))

    def boundary_tick(self, now: int) -> int:
        if self.stage not in ("PE", "DiT"):
            return self.finish_tick
        if self.stage == "PE":
            origin = self.start_tick + self.ttft_ticks
            if now <= origin:
                return origin
            count = (now - origin + self.tpot_ticks - 1) // self.tpot_ticks
            return min(self.finish_tick, origin + count * self.tpot_ticks)
        else:
            lo, hi = 0, self.target_progress - self.initial_progress
            while lo < hi:
                middle = (lo + hi) // 2
                if self.start_tick + self._step_offset(middle) < now:
                    lo = middle + 1
                else:
                    hi = middle
            return self.start_tick + self._step_offset(lo)

    def busy_overlap(self, lo: float, hi: float) -> float:
        end = min(ticks(hi), self.end_tick if self.end_tick is not None else self.finish_tick)
        return seconds(max(0, end - max(ticks(lo), self.start_tick)))

    def work_between(self, lo: float, hi: float) -> float:
        if ticks(hi) < self.start_tick or ticks(lo) >= (self.end_tick if self.end_tick is not None else self.finish_tick):
            return 0.0
        before = self.initial_progress if ticks(lo) < self.start_tick else self.progress_at(lo)
        units = max(0, self.progress_at(hi) - before)
        return units / 50 if self.stage == "DiT" else float(units)


@dataclass
class Lane:
    uid: str
    physical_uid: str
    stage: str
    order: int
    raw: dict
    state: str = "ready"
    created_s: float = 0.0
    queue: deque[int] = field(default_factory=deque)
    recovery: deque[int] = field(default_factory=deque)
    current: Execution | None = None
    version: int = 0
    history: list[Execution] = field(default_factory=list)
    last_rate: float | None = None
    last_rate_time_s: float | None = None

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def waiting(self) -> tuple[int, ...]:
        return tuple(self.recovery) + tuple(self.queue)


@dataclass
class Physical:
    uid: str
    raw: dict
    epoch: int
    lanes: list[str]
    state: str
    created_s: float
    ready_s: float | None = None
    retired_s: float | None = None


@dataclass
class ConversionGroup:
    id: str
    original: dict
    targets: list[dict]
    sources: list[str]
    launched: list[str] = field(default_factory=list)
    launch_started: bool = False
    finished: bool = False


@dataclass
class Transition:
    id: int
    target_bin: int
    groups: list[ConversionGroup]
    record: dict
