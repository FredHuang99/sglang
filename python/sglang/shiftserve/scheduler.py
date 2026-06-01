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
class DITVAEStageRequest:
    """A request inside one DIT_VAE bundle's internal stage pipeline."""

    request_id: str
    remaining_steps: int = 0
    migration_priority: int = 0
    dit_started_at_s: float | None = None
    dit_done_at_s: float | None = None
    vae_started_at_s: float | None = None
    vae_done_at_s: float | None = None


@dataclass
class DITVAEBundleState:
    """One launched DIT_VAE instance with independent DiT and VAE stage slots."""

    bundle_id: str
    node_id: str
    paired_te_id: str | None = None
    active: bool = True
    ready: bool = True
    draining: bool = False
    launching: bool = False
    dit_bs: int = 1
    vae_bs: int = 1
    dit_queue: deque[DITVAEStageRequest] = field(default_factory=deque)
    vae_queue: deque[DITVAEStageRequest] = field(default_factory=deque)
    dit_running: dict[int, DITVAEStageRequest] = field(default_factory=dict)
    vae_running: dict[int, DITVAEStageRequest] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dit_bs < 1 or self.vae_bs < 1:
            raise ValueError("DIT_VAE stage slots must satisfy dit_bs>=1 and vae_bs>=1")

    def is_eligible(self) -> bool:
        return self.active and self.ready and not self.draining and not self.launching

    def dit_free_slots(self) -> int:
        return max(self.dit_bs - len(self.dit_running), 0)

    def vae_free_slots(self) -> int:
        return max(self.vae_bs - len(self.vae_running), 0)

    def enqueue_dit(self, request: DITVAEStageRequest, *, head: bool = False) -> None:
        if head:
            self.dit_queue.appendleft(request)
        else:
            self.dit_queue.append(request)

    def enqueue_vae(self, request: DITVAEStageRequest, *, head: bool = False) -> None:
        if head:
            self.vae_queue.appendleft(request)
        else:
            self.vae_queue.append(request)

    def start_next_dit(
        self, *, now_s: float | None = None
    ) -> tuple[int, DITVAEStageRequest] | None:
        slot = self._first_free_slot(self.dit_running, self.dit_bs)
        if slot is None or not self.dit_queue:
            return None
        request = self.dit_queue.popleft()
        request.dit_started_at_s = now_s
        self.dit_running[slot] = request
        return slot, request

    def finish_dit(
        self, request_id: str, *, now_s: float | None = None
    ) -> DITVAEStageRequest:
        slot = self._find_running_slot(self.dit_running, request_id, stage_name="DiT")
        request = self.dit_running.pop(slot)
        request.dit_done_at_s = now_s
        self.vae_queue.append(request)
        return request

    def start_next_vae(
        self, *, now_s: float | None = None
    ) -> tuple[int, DITVAEStageRequest] | None:
        slot = self._first_free_slot(self.vae_running, self.vae_bs)
        if slot is None or not self.vae_queue:
            return None
        request = self.vae_queue.popleft()
        request.vae_started_at_s = now_s
        self.vae_running[slot] = request
        return slot, request

    def finish_vae(
        self, request_id: str, *, now_s: float | None = None
    ) -> DITVAEStageRequest:
        slot = self._find_running_slot(self.vae_running, request_id, stage_name="VAE")
        request = self.vae_running.pop(slot)
        request.vae_done_at_s = now_s
        return request

    def work_items(self) -> int:
        return (
            len(self.dit_queue)
            + len(self.vae_queue)
            + len(self.dit_running)
            + len(self.vae_running)
        )

    def estimate_candidate_finish_ms(
        self,
        profile: StageCostProfile,
        request: "RequestEstimate",
    ) -> float:
        """Estimate finish time through a two-stage tandem DiT->VAE pipeline."""

        candidate_dit_ms = self._dit_ms(request.remaining_steps, profile)
        dit_ahead_ms = sum(
            self._dit_ms(r.remaining_steps, profile)
            for r in self.dit_running.values()
        )
        dit_ahead_count = len(self.dit_running)
        if not request.insert_at_head:
            dit_ahead_ms += sum(
                self._dit_ms(r.remaining_steps, profile) for r in self.dit_queue
            )
            dit_ahead_count += len(self.dit_queue)

        vae_available_ms = len(self.vae_running) * profile.vae_ms
        vae_available_ms += len(self.vae_queue) * profile.vae_ms
        candidate_dit_finish_ms = dit_ahead_ms + candidate_dit_ms
        candidate_vae_ahead_ms = vae_available_ms + dit_ahead_count * profile.vae_ms
        return max(candidate_dit_finish_ms, candidate_vae_ahead_ms) + profile.vae_ms

    @staticmethod
    def _dit_ms(remaining_steps: int | None, profile: StageCostProfile) -> float:
        steps = profile.dit_steps if remaining_steps is None else remaining_steps
        return max(steps, 0) * profile.dit_per_step_ms

    @staticmethod
    def _first_free_slot(
        running: dict[int, DITVAEStageRequest], capacity: int
    ) -> int | None:
        for slot in range(capacity):
            if slot not in running:
                return slot
        return None

    @staticmethod
    def _find_running_slot(
        running: dict[int, DITVAEStageRequest],
        request_id: str,
        *,
        stage_name: str,
    ) -> int:
        for slot, request in running.items():
            if request.request_id == request_id:
                return slot
        raise KeyError(f"{request_id} is not running in DIT_VAE {stage_name} slot")


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
    paired_te_id: str | None = None
    stage_slots: dict[str, int] = field(default_factory=dict)
    dit_vae_bundle: DITVAEBundleState | None = None

    def __post_init__(self) -> None:
        if self.kind == StageKind.DIT_VAE and self.dit_vae_bundle is None:
            self.dit_vae_bundle = DITVAEBundleState(
                bundle_id=self.instance_id,
                node_id=self.node_id,
                paired_te_id=self.paired_te_id,
                active=self.active,
                ready=self.ready,
                draining=self.draining,
                launching=self.launching,
                dit_bs=int(self.stage_slots.get("dit_bs", 1)),
                vae_bs=int(self.stage_slots.get("vae_bs", 1)),
            )

    def is_eligible(self) -> bool:
        if self.kind == StageKind.DIT_VAE and self.dit_vae_bundle is not None:
            self._sync_bundle_status()
            return self.dit_vae_bundle.is_eligible()
        return self.active and self.ready and not self.draining and not self.launching

    def queue_work_items(self) -> int:
        if self.kind == StageKind.DIT_VAE and self.dit_vae_bundle is not None:
            return self.dit_vae_bundle.work_items()
        return len(self.queue)

    def _sync_bundle_status(self) -> None:
        if self.dit_vae_bundle is None:
            return
        self.dit_vae_bundle.active = self.active
        self.dit_vae_bundle.ready = self.ready
        self.dit_vae_bundle.draining = self.draining
        self.dit_vae_bundle.launching = self.launching
        self.dit_vae_bundle.node_id = self.node_id
        self.dit_vae_bundle.paired_te_id = self.paired_te_id


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
        remaining_steps: int | None = None,
    ) -> None:
        if instance.kind == StageKind.DIT_VAE and instance.dit_vae_bundle is not None:
            instance.dit_vae_bundle.enqueue_dit(
                DITVAEStageRequest(
                    request_id=request_id,
                    remaining_steps=(
                        self.cost_profile.dit_steps
                        if remaining_steps is None
                        else remaining_steps
                    ),
                ),
                head=head,
            )
            return
        if head:
            instance.queue.appendleft(request_id)
        else:
            instance.queue.append(request_id)

    def estimate_work_ms(
        self,
        instance: InstanceRuntimeState,
        request: RequestEstimate,
    ) -> float:
        if instance.kind == StageKind.DIT_VAE and instance.dit_vae_bundle is not None:
            return instance.dit_vae_bundle.estimate_candidate_finish_ms(
                self.cost_profile,
                request,
            )
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
                expected_bin = self._promote_expected_bin(
                    expected_bin,
                    running.generated_tokens,
                )
            return max(expected_bin - running.generated_tokens, 0) * profile.tpot_ms
        if running.stage in (StageKind.DIT, StageKind.DIT_VAE):
            steps = (
                profile.dit_steps
                if running.remaining_steps is None
                else running.remaining_steps
            )
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
            steps = (
                profile.dit_steps
                if request.remaining_steps is None
                else request.remaining_steps
            )
            return max(steps, 0) * profile.dit_per_step_ms
        if stage == StageKind.DIT_VAE:
            if instance.dit_vae_bundle is not None:
                return instance.dit_vae_bundle.estimate_candidate_finish_ms(
                    profile,
                    request,
                )
            steps = (
                profile.dit_steps
                if request.remaining_steps is None
                else request.remaining_steps
            )
            return max(steps, 0) * profile.dit_per_step_ms + profile.vae_ms
        if stage == StageKind.VAE:
            return profile.vae_ms
        return 0.0

    def _promote_expected_bin(self, expected_bin: int, generated_tokens: int) -> int:
        if expected_bin < self.token_estimator.long_bin:
            return self.token_estimator.long_bin
        return max(expected_bin, generated_tokens)
