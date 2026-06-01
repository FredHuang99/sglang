"""Deterministic ShiftServe scheduling and flip-decision logic."""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable


class StageKind(str, enum.Enum):
    PE = "pe"
    TE = "te"
    DIT = "dit"
    VAE = "vae"
    DIT_VAE = "dit_vae"


class SchedulerMode(str, enum.Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"

    @classmethod
    def from_weighted_flag(cls, enabled: bool) -> "SchedulerMode":
        return cls.WEIGHTED if enabled else cls.ROUND_ROBIN


@dataclass(frozen=True)
class StageCostProfile:
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    te_ms: float = 0.0
    dit_steps: int = 1
    dit_per_step_ms: float = 0.0
    vae_ms: float = 0.0


@dataclass
class RunningRequest:
    request_id: str
    stage: StageKind
    bin_tokens: int = 0
    generated_tokens: int = 0
    remaining_steps: int = 0


@dataclass
class InstanceRuntimeState:
    instance_id: str
    kind: StageKind
    node_id: str
    active: bool = True
    ready: bool = True
    draining: bool = False
    launching: bool = False
    queue: deque[str] = field(default_factory=deque)
    running: RunningRequest | None = None
    completed_output_tokens: deque[int] = field(default_factory=lambda: deque(maxlen=16))
    node_group_id: str | None = None
    pipeline_group_id: str | None = None
    max_slots: int = 1

    def is_eligible(self) -> bool:
        return self.active and self.ready and not self.draining and not self.launching

    def queue_work_items(self) -> int:
        return len(self.queue)


@dataclass(frozen=True)
class RequestEstimate:
    request_id: str
    stage: StageKind
    bin_tokens: int = 512
    input_tokens: int = 128
    generated_tokens: int = 0
    remaining_steps: int | None = None
    target_node_id: str | None = None
    pipeline_group_id: str | None = None
    insert_at_head: bool = False


@dataclass(frozen=True)
class SelectionResult:
    instance_id: str
    estimated_work_ms: float
    mode: SchedulerMode
    fallback_reason: str | None = None


class WindowedTokenEstimator:
    """Maps recent PE output lengths to short/long bins per instance."""

    def __init__(self, short_bin: int = 512, long_bin: int = 2048, window_size: int = 16):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if short_bin <= 0 or long_bin <= short_bin:
            raise ValueError("bins must satisfy 0 < short_bin < long_bin")
        self.short_bin = short_bin
        self.long_bin = long_bin
        self.midpoint = (short_bin + long_bin) / 2
        self._windows: dict[str, deque[int]] = {}
        self._window_size = window_size

    def record(self, instance_id: str, output_tokens: int) -> None:
        window = self._windows.setdefault(instance_id, deque(maxlen=self._window_size))
        window.append(int(output_tokens))

    def expected_bin(self, instance_id: str, default_bin: int | None = None) -> int:
        window = self._windows.get(instance_id)
        if not window:
            return default_bin or self.short_bin
        avg = sum(window) / len(window)
        return self.long_bin if avg > self.midpoint else self.short_bin


class HysteresisFlipMonitor:
    """Global PE-window flip detector with optional margin hysteresis."""

    def __init__(
        self,
        short_bin: int = 512,
        long_bin: int = 2048,
        window_size: int = 16,
        margin_enabled: bool = False,
        margin_ratio: float = 0.0,
        initial_direction: str = "short",
    ):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not 0.0 <= margin_ratio <= 1.0:
            raise ValueError("margin_ratio must be in [0, 1]")
        self.short_bin = short_bin
        self.long_bin = long_bin
        self.midpoint = (short_bin + long_bin) / 2
        delta = (long_bin - short_bin) * margin_ratio if margin_enabled else 0.0
        self.high_threshold = self.midpoint + delta
        self.low_threshold = self.midpoint - delta
        self.direction = initial_direction
        self._window: deque[int] = deque(maxlen=window_size)

    def record_completion(self, output_tokens: int) -> str | None:
        self._window.append(int(output_tokens))
        avg = sum(self._window) / len(self._window)
        if self.direction == "short" and avg > self.high_threshold:
            self.direction = "long"
            return "short_to_long"
        if self.direction == "long" and avg < self.low_threshold:
            self.direction = "short"
            return "long_to_short"
        return None


class RoundRobinCursor:
    def __init__(self) -> None:
        self._next: dict[StageKind, int] = {}

    def choose(
        self,
        stage: StageKind,
        candidates: list[InstanceRuntimeState],
    ) -> InstanceRuntimeState:
        if not candidates:
            raise ValueError("Round-robin selection requires at least one candidate")
        start = self._next.get(stage, 0) % len(candidates)
        chosen = candidates[start]
        self._next[stage] = (start + 1) % len(candidates)
        return chosen


class ShiftServeScheduler:
    """Selects PE/TE/DiT/VAE instances for validation and full flip experiments."""

    def __init__(
        self,
        *,
        mode: SchedulerMode = SchedulerMode.ROUND_ROBIN,
        cost_profile: StageCostProfile | None = None,
        short_bin: int = 512,
        long_bin: int = 2048,
        pe_window_size: int = 16,
    ):
        self.mode = mode
        self.cost_profile = cost_profile or StageCostProfile()
        self.token_estimator = WindowedTokenEstimator(
            short_bin=short_bin,
            long_bin=long_bin,
            window_size=pe_window_size,
        )
        self._rr = RoundRobinCursor()

    def eligible_instances(
        self,
        instances: Iterable[InstanceRuntimeState],
        request: RequestEstimate,
    ) -> list[InstanceRuntimeState]:
        candidates = [
            instance
            for instance in instances
            if instance.kind == request.stage and instance.is_eligible()
        ]
        if request.target_node_id is not None:
            candidates = [
                instance
                for instance in candidates
                if instance.node_id == request.target_node_id
            ]
        if request.pipeline_group_id is not None:
            candidates = [
                instance
                for instance in candidates
                if instance.pipeline_group_id == request.pipeline_group_id
            ]
        return sorted(candidates, key=lambda instance: instance.instance_id)

    def select(
        self,
        instances: Iterable[InstanceRuntimeState],
        request: RequestEstimate,
    ) -> SelectionResult:
        candidates = self.eligible_instances(instances, request)
        fallback_reason = None
        if not candidates and request.target_node_id is not None:
            relaxed = RequestEstimate(
                **{
                    **request.__dict__,
                    "target_node_id": None,
                    "pipeline_group_id": None,
                }
            )
            candidates = self.eligible_instances(instances, relaxed)
            fallback_reason = "same_node_or_pipeline_group_unavailable"
        if not candidates:
            raise ValueError(f"No eligible instances for stage {request.stage.value}")

        if self.mode == SchedulerMode.ROUND_ROBIN:
            chosen = self._rr.choose(request.stage, candidates)
            return SelectionResult(
                instance_id=chosen.instance_id,
                estimated_work_ms=0.0,
                mode=self.mode,
                fallback_reason=fallback_reason,
            )

        scored = [
            (self.estimate_work_ms(instance, request), instance.instance_id, instance)
            for instance in candidates
        ]
        work_ms, _, chosen = min(scored, key=lambda item: (item[0], item[1]))
        return SelectionResult(
            instance_id=chosen.instance_id,
            estimated_work_ms=work_ms,
            mode=self.mode,
            fallback_reason=fallback_reason,
        )

    def enqueue(
        self,
        instance: InstanceRuntimeState,
        request_id: str,
        *,
        head: bool = False,
    ) -> None:
        if head:
            instance.queue.appendleft(request_id)
        else:
            instance.queue.append(request_id)

    def estimate_work_ms(
        self,
        instance: InstanceRuntimeState,
        request: RequestEstimate,
    ) -> float:
        running = self._estimate_running_ms(instance)
        queued = instance.queue_work_items() * self._estimate_new_request_ms(
            instance.kind, request, instance
        )
        new = self._estimate_new_request_ms(instance.kind, request, instance)
        return running + queued + new

    def _estimate_running_ms(self, instance: InstanceRuntimeState) -> float:
        running = instance.running
        if running is None:
            return 0.0
        profile = self.cost_profile
        if running.stage == StageKind.PE:
            expected_bin = self.token_estimator.expected_bin(
                instance.instance_id,
                default_bin=running.bin_tokens,
            )
            if running.generated_tokens <= 0:
                return profile.ttft_ms + max(expected_bin - 1, 0) * profile.tpot_ms
            if running.generated_tokens > expected_bin:
                expected_bin = max(expected_bin, running.generated_tokens)
            return max(expected_bin - running.generated_tokens, 0) * profile.tpot_ms
        if running.stage in (StageKind.DIT, StageKind.DIT_VAE):
            steps = running.remaining_steps or profile.dit_steps
            return max(steps, 0) * profile.dit_per_step_ms
        if running.stage == StageKind.TE:
            return profile.te_ms
        if running.stage == StageKind.VAE:
            return profile.vae_ms
        return 0.0

    def _estimate_new_request_ms(
        self,
        stage: StageKind,
        request: RequestEstimate,
        instance: InstanceRuntimeState,
    ) -> float:
        profile = self.cost_profile
        if stage == StageKind.PE:
            bin_tokens = self.token_estimator.expected_bin(
                instance.instance_id,
                default_bin=request.bin_tokens,
            )
            return profile.ttft_ms + max(bin_tokens - 1, 0) * profile.tpot_ms
        if stage == StageKind.TE:
            return profile.te_ms
        if stage == StageKind.DIT:
            steps = request.remaining_steps or profile.dit_steps
            return max(steps, 0) * profile.dit_per_step_ms
        if stage == StageKind.DIT_VAE:
            steps = request.remaining_steps or profile.dit_steps
            return max(steps, 0) * profile.dit_per_step_ms + profile.vae_ms
        if stage == StageKind.VAE:
            return profile.vae_ms
        return 0.0
