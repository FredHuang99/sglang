"""Completed-length observations and service-capacity estimates without length oracles."""

from __future__ import annotations

import math
from collections import deque

from .profiles import SimulationProfiles
from .state import Lane, Progress
from .timing import seconds, ticks


class Monitor:
    def __init__(self, window_s: float, input_tokens: int, bins: tuple[int, int], profiles: SimulationProfiles):
        self.window_s, self.input_tokens, self.bins = window_s, input_tokens, bins
        self.profiles = profiles
        self.lengths: deque[tuple[int, str, int, int]] = deque()

    @property
    def window_s(self):
        return seconds(self.window_ticks)

    @window_s.setter
    def window_s(self, value):
        self.window_ticks = ticks(value, positive=True)

    def complete_pe(self, time: float, lane: Lane, request_id: int, output: int) -> None:
        self.lengths.append((ticks(time), lane.uid, request_id, output))

    def length(self, time: float, lane: Lane | None = None) -> dict:
        now = ticks(time)
        lo = now - self.window_ticks
        samples = [sample for sample in self.lengths if lo < sample[0] <= now and (lane is None or sample[1] == lane.uid)]
        return {
            "samples": len(samples),
            "mean": math.fsum(s[3] for s in samples) / len(samples) if samples else None,
            "window_start_exclusive_s": seconds(lo),
            "window_end_inclusive_s": seconds(now),
            "covered_s": seconds(max(0, now - max(lo, ticks(lane.created_s) if lane else 0))),
            "first_sample_s": seconds(samples[0][0]) if samples else None,
            "last_sample_s": seconds(samples[-1][0]) if samples else None,
        }

    def estimated_length(self, lane: Lane, time: float, active_bin: int) -> dict:
        local = self.length(time, lane)
        if local["samples"]:
            return {**local, "source": "instance_completed_window"}
        system = self.length(time)
        if system["samples"]:
            return {**system, "source": "system_completed_window"}
        return {**local, "mean": float(active_bin), "source": "active_bin"}

    def remaining_tokens(self, lane: Lane, generated: int, time: float, active_bin: int) -> float:
        estimate = self.estimated_length(lane, time, active_bin)["mean"]
        if generated > estimate:
            estimate = next((b for b in self.bins if b >= generated), self.bins[-1])
        return max(0.0, estimate - generated)

    def rate(self, lane: Lane, time: float, active_bin: int) -> dict:
        """Read the current window without changing observations or fallback history."""
        now = ticks(time)
        lo_tick = now - self.window_ticks
        lo, time = seconds(lo_tick), seconds(now)
        # History stores closed segments plus the current segment. Clip at the query
        # time, so pending executions never contribute future work or future time.
        busy = math.fsum(run.busy_overlap(lo, time) for run in lane.history)
        samples = [
            run for run in lane.history
            if run.completed and run.end_tick is not None and lo_tick < run.end_tick <= now
        ]
        if lane.stage in ("PE", "DiT"):
            work = math.fsum(run.work_between(lo, time) for run in lane.history)
            numerator, denominator = work, busy
        else:
            work = float(len(samples))
            numerator = work
            denominator = seconds(sum(run.end_tick - run.start_tick for run in samples))
        measured = numerator / denominator if numerator > 0 and denominator > 0 else None
        if measured is not None and math.isfinite(measured):
            rate, source = measured, "busy_time_window"
        elif lane.last_rate is not None:
            rate, source = lane.last_rate, "last_valid_current_lifecycle"
        else:
            output = active_bin
            if lane.stage == "PE":
                average = self.estimated_length(lane, time, active_bin)["mean"]
                output = max(1, int(math.floor(average + 0.5)))
            latency = self.profiles.latency(lane.raw, lane.stage, self.input_tokens, output)
            rate = output / latency if lane.stage == "PE" else 1 / latency
            source = "same_hardware_parallelism_profile"
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"No positive capacity estimate for {lane.uid}")
        return {
            "rate": rate, "source": source, "unit": "token/s" if lane.stage == "PE" else "req/s",
            "window_s": self.window_s, "work_in_window": work, "busy_s_in_window": busy,
            "window_start_exclusive_s": lo, "window_end_inclusive_s": time,
            "covered_s": seconds(max(0, now - max(lo_tick, ticks(lane.created_s)))),
            "completed_service_samples": len(samples),
            "last_valid_estimate_s": lane.last_rate_time_s,
            "cached_rate": lane.last_rate,
            "measured_rate": measured if measured is not None and math.isfinite(measured) else None,
            "estimator_service_s": denominator,
            "realized_rate_per_window_s": work / self.window_s,
        }

    def sample(self, lane: Lane, time: float, active_bin: int) -> dict:
        """Commit one periodic observation, never a profile or fallback value."""
        observation = self.rate(lane, time, active_bin)
        if observation["measured_rate"] is not None:
            lane.last_rate = observation["measured_rate"]
            lane.last_rate_time_s = seconds(ticks(time))
        return {**observation, "cached_rate": lane.last_rate,
                "last_valid_estimate_s": lane.last_rate_time_s,
                "cache_updated": observation["measured_rate"] is not None}

    def maintain(self, time: float) -> None:
        """Only the common monitoring clock retires old length samples."""
        lo = ticks(time) - self.window_ticks
        while self.lengths and self.lengths[0][0] <= lo:
            self.lengths.popleft()


class WorkEstimator:
    def __init__(self, monitor: Monitor):
        self.monitor = monitor

    def units(self, lane: Lane, request: Progress, now: float, active_bin: int) -> float:
        if lane.stage == "PE":
            return self.monitor.remaining_tokens(lane, request.generated, now, active_bin)
        if lane.stage == "DiT":
            return max(0, 50 - request.steps) / 50
        return 1.0
