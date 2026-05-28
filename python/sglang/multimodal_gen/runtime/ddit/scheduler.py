"""Hungry-first DDiT scheduling primitives.

This module is intentionally independent from the serving event loop so the
policy can be unit tested without a CUDA runtime.
"""

from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import (
    is_power_of_two,
    parse_allowed_gpu_counts,
    parse_local_ranks,
    select_preferred_rank_tuple,
)
from .profile import DDiTProfile, ProfileStore


class RequestPhase(str, Enum):
    TEXT_ENCODER_PENDING = "text_encoder_pending"
    DIT_WAITING = "dit_waiting"
    WAITING = "waiting"
    DIT = "dit"
    VAE = "vae"
    DONE = "done"


@dataclass
class DDiTRequestState:
    request_id: str
    resolution: str
    total_steps: int
    arrival_time: float = field(default_factory=time.time)
    phase: RequestPhase = RequestPhase.WAITING
    ranks: tuple[int, ...] = ()
    cur_step: int = 0
    last_scheduled_step: int = 0
    vae_k: int = 1


@dataclass
class DDiTSchedulerConfig:
    local_ranks: tuple[int, ...] = tuple(range(8))
    opt_gpus_num: dict[str, int] = field(
        default_factory=lambda: {"144p": 1, "240p": 1, "360p": 2, "720p": 4}
    )
    dit_step_times: dict[str, dict[int, float]] = field(
        default_factory=lambda: {
            "144p": {1: 4.45, 2: 3.67, 4: 3.69, 8: 3.98},
            "240p": {1: 8.0, 2: 5.0, 4: 4.5, 8: 4.8},
            "360p": {1: 16.0, 2: 9.0, 4: 7.0, 8: 7.5},
            "720p": {1: 112.76, 2: 48.64, 4: 25.08, 8: 13.39},
        }
    )
    allowed_gpu_counts: tuple[int, ...] = (1, 2, 4, 8)


@dataclass
class FixedBaselineSchedulerConfig:
    local_ranks: tuple[int, ...] = tuple(range(8))
    baseline_gpus: int = 1
    allowed_gpu_counts: tuple[int, ...] = (1, 2, 4, 8)


class FixedBaselineScheduler:
    """Fixed-k DiT/VAE rank allocator with full-node text encoder semantics."""

    vae_same_as_dit = True
    policy_name = "fixed_baseline"

    def __init__(self, config: FixedBaselineSchedulerConfig | None = None):
        self.config = config or FixedBaselineSchedulerConfig()
        if self.config.baseline_gpus not in self.config.allowed_gpu_counts:
            raise ValueError(
                f"baseline_gpus must be one of {self.config.allowed_gpu_counts}, "
                f"got {self.config.baseline_gpus}"
            )
        if not is_power_of_two(self.config.baseline_gpus):
            raise ValueError(
                f"baseline_gpus must be a power of two, got {self.config.baseline_gpus}"
            )
        if self.config.baseline_gpus > len(self.config.local_ranks):
            raise ValueError(
                f"baseline_gpus={self.config.baseline_gpus} exceeds local ranks "
                f"{self.config.local_ranks}"
            )

        self.requests: dict[str, DDiTRequestState] = {}
        self.text_encoder_queue: deque[str] = deque()
        self.dit_waiting: deque[str] = deque()
        self.gpu_owner: dict[int, str | None] = {
            rank: None for rank in self.config.local_ranks
        }

    def add_request(self, request: DDiTRequestState) -> None:
        if request.request_id in self.requests:
            raise ValueError(f"Duplicate DDiT request id: {request.request_id}")
        request.phase = RequestPhase.TEXT_ENCODER_PENDING
        self.requests[request.request_id] = request
        self.text_encoder_queue.append(request.request_id)

    def mark_text_encoder_done(self, request_id: str) -> None:
        req = self.requests[request_id]
        req.phase = RequestPhase.DIT_WAITING
        self.dit_waiting.append(request_id)

    def has_waiting_requests(self) -> bool:
        return bool(self.dit_waiting)

    def update_cur_step(self, request_id: str, cur_step: int) -> None:
        self.requests[request_id].cur_step = int(cur_step)

    def complete_request(self, request_id: str) -> None:
        req = self.requests[request_id]
        for rank in req.ranks:
            self.gpu_owner[rank] = None
        req.ranks = ()
        req.phase = RequestPhase.DONE

    def _free_ranks(self) -> tuple[int, ...]:
        return tuple(rank for rank, owner in self.gpu_owner.items() if owner is None)

    def _assign(self, request_id: str, ranks: tuple[int, ...]) -> None:
        req = self.requests[request_id]
        for rank in ranks:
            if self.gpu_owner[rank] is not None:
                raise RuntimeError(f"Rank {rank} is already owned")
            self.gpu_owner[rank] = request_id
        req.ranks = tuple(sorted(ranks))
        req.phase = RequestPhase.DIT

    def schedule(self) -> list[dict[str, Any]]:
        decisions: list[dict[str, Any]] = []
        while self.dit_waiting:
            free = self._free_ranks()
            if len(free) < self.config.baseline_gpus:
                break
            request_id = self.dit_waiting.popleft()
            ranks = select_preferred_rank_tuple(free, self.config.baseline_gpus)
            self._assign(request_id, ranks)
            decisions.append(
                {
                    "request_id": request_id,
                    "stage": "baseline",
                    "old_ranks": (),
                    "new_ranks": ranks,
                    "reason": "fixed_baseline",
                }
            )
        return decisions


class HungryFirstScheduler:
    """Resource policy matching the FlexDiT hungry-first behavior."""

    vae_same_as_dit = False
    policy_name = "hungry_first"

    def __init__(self, config: DDiTSchedulerConfig | None = None):
        self.config = config or DDiTSchedulerConfig()
        self.requests: dict[str, DDiTRequestState] = {}
        self.waiting: deque[str] = deque()
        self.gpu_owner: dict[int, str | None] = {
            rank: None for rank in self.config.local_ranks
        }

    def add_request(self, request: DDiTRequestState) -> None:
        if request.request_id in self.requests:
            raise ValueError(f"Duplicate DDiT request id: {request.request_id}")
        self.requests[request.request_id] = request
        self.waiting.append(request.request_id)

    def update_cur_step(self, request_id: str, cur_step: int) -> None:
        self.requests[request_id].cur_step = int(cur_step)

    def has_waiting_requests(self) -> bool:
        return bool(self.waiting)

    def complete_dit(
        self, request_id: str, *, vae_k: int | None = None
    ) -> tuple[int, ...]:
        req = self.requests[request_id]
        req.phase = RequestPhase.VAE
        if vae_k is not None:
            req.vae_k = int(vae_k)
        keep = req.ranks[: req.vae_k]
        for rank in req.ranks:
            if rank not in keep:
                self.gpu_owner[rank] = None
        req.ranks = keep
        return keep

    def transition_to_vae(
        self, request_id: str, vae_ranks: tuple[int, ...]
    ) -> tuple[int, ...]:
        req = self.requests[request_id]
        vae_ranks = tuple(sorted(int(rank) for rank in vae_ranks))
        old_ranks = set(req.ranks)
        new_ranks = set(vae_ranks)
        for rank in old_ranks - new_ranks:
            self.gpu_owner[rank] = None
        for rank in new_ranks - old_ranks:
            if self.gpu_owner.get(rank) not in (None, request_id):
                raise RuntimeError(
                    f"Rank {rank} is already owned by {self.gpu_owner[rank]}"
                )
            self.gpu_owner[rank] = request_id
        for rank in new_ranks & old_ranks:
            self.gpu_owner[rank] = request_id
        req.ranks = vae_ranks
        req.phase = RequestPhase.VAE
        return vae_ranks

    def complete_vae(self, request_id: str) -> None:
        req = self.requests[request_id]
        for rank in req.ranks:
            self.gpu_owner[rank] = None
        req.ranks = ()
        req.phase = RequestPhase.DONE

    def _free_ranks(self) -> list[int]:
        return [rank for rank, owner in self.gpu_owner.items() if owner is None]

    def _take_preferred_free_ranks(
        self, free: list[int], count: int
    ) -> tuple[tuple[int, ...], list[int]]:
        selected = select_preferred_rank_tuple(tuple(free), count)
        selected_set = set(selected)
        remaining = [rank for rank in free if rank not in selected_set]
        return selected, remaining

    def _assign(self, request_id: str, ranks: tuple[int, ...]) -> None:
        req = self.requests[request_id]
        old_ranks = set(req.ranks)
        new_ranks = set(ranks)
        for rank in old_ranks - new_ranks:
            self.gpu_owner[rank] = None
        for rank in new_ranks - old_ranks:
            if self.gpu_owner[rank] not in (None, request_id):
                raise RuntimeError(
                    f"Rank {rank} is already owned by {self.gpu_owner[rank]}"
                )
            self.gpu_owner[rank] = request_id
        req.ranks = tuple(sorted(ranks))
        req.phase = RequestPhase.DIT
        req.last_scheduled_step = req.cur_step

    def _opt_gpu_count(self, resolution: str) -> int:
        opt = self.config.opt_gpus_num.get(resolution, 1)
        return max(1, min(opt, max(self.config.allowed_gpu_counts)))

    def _floor_allowed_power_of_two(self, count: int) -> int:
        allowed = [
            value
            for value in self.config.allowed_gpu_counts
            if value <= count and is_power_of_two(value)
        ]
        return max(allowed) if allowed else 1

    def _starvation_score(self, req: DDiTRequestState) -> float:
        current_k = max(1, len(req.ranks))
        opt_k = self._opt_gpu_count(req.resolution)
        if current_k >= opt_k:
            return 0.0
        profile = self.config.dit_step_times.get(req.resolution, {})
        current_t = profile.get(current_k, profile.get(1, 0.0))
        opt_t = profile.get(opt_k, current_t)
        lag_time = max(0.0, current_t - opt_t)
        lag_steps = max(0, req.cur_step - req.last_scheduled_step)
        return lag_steps * lag_time

    def schedule(self) -> list[dict[str, Any]]:
        """Run one scheduling pass and return rank-change decisions."""
        decisions: list[dict[str, Any]] = []
        free = self._free_ranks()

        hungry_heap: list[tuple[float, str]] = []
        for req in self.requests.values():
            if req.phase != RequestPhase.DIT:
                continue
            score = self._starvation_score(req)
            if score > 0:
                heapq.heappush(hungry_heap, (-score, req.request_id))

        while hungry_heap and free:
            _neg_score, request_id = heapq.heappop(hungry_heap)
            req = self.requests[request_id]
            opt_k = self._opt_gpu_count(req.resolution)
            target_k = self._floor_allowed_power_of_two(
                min(opt_k, len(req.ranks) + len(free))
            )
            if target_k <= len(req.ranks):
                continue
            add_count = target_k - len(req.ranks)
            selected, free = self._take_preferred_free_ranks(free, add_count)
            new_ranks = tuple(sorted(req.ranks + selected))
            old_ranks = req.ranks
            self._assign(request_id, new_ranks)
            decisions.append(
                {
                    "request_id": request_id,
                    "stage": "dit",
                    "old_ranks": old_ranks,
                    "new_ranks": new_ranks,
                    "reason": "hungry_first",
                }
            )

        while self.waiting and free:
            request_id = self.waiting[0]
            req = self.requests[request_id]
            opt_k = self._opt_gpu_count(req.resolution)
            target_k = self._floor_allowed_power_of_two(min(opt_k, len(free)))
            if target_k <= 0:
                break
            self.waiting.popleft()
            new_ranks, free = self._take_preferred_free_ranks(free, target_k)
            self._assign(request_id, new_ranks)
            decisions.append(
                {
                    "request_id": request_id,
                    "stage": "dit_start",
                    "old_ranks": (),
                    "new_ranks": new_ranks,
                    "reason": "waiting_queue",
                }
            )

        return decisions


@dataclass
class ProfileSchedulerConfig:
    local_ranks: tuple[int, ...]
    allowed_gpu_counts: tuple[int, ...]
    profile: DDiTProfile
    window_size: int = 8


class ProfileBackedScheduler:
    """Base class for profile-backed concurrent DDiT policies."""

    vae_same_as_dit = True
    policy_name = "profile"

    def __init__(self, config: ProfileSchedulerConfig):
        self.config = config
        self.requests: dict[str, DDiTRequestState] = {}
        self.waiting: deque[str] = deque()
        self.gpu_owner: dict[int, str | None] = {
            rank: None for rank in self.config.local_ranks
        }

    def add_request(self, request: DDiTRequestState) -> None:
        if request.request_id in self.requests:
            raise ValueError(f"Duplicate DDiT request id: {request.request_id}")
        request.phase = RequestPhase.WAITING
        self.requests[request.request_id] = request
        self.waiting.append(request.request_id)

    def has_waiting_requests(self) -> bool:
        return bool(self.waiting)

    def update_cur_step(self, request_id: str, cur_step: int) -> None:
        self.requests[request_id].cur_step = int(cur_step)

    def complete_request(self, request_id: str) -> None:
        req = self.requests[request_id]
        for rank in req.ranks:
            self.gpu_owner[rank] = None
        req.ranks = ()
        req.phase = RequestPhase.DONE

    def _free_ranks(self) -> list[int]:
        return [rank for rank, owner in self.gpu_owner.items() if owner is None]

    def _take_preferred_free_ranks(
        self, free: list[int], count: int
    ) -> tuple[tuple[int, ...], list[int]]:
        selected = select_preferred_rank_tuple(tuple(free), count)
        selected_set = set(selected)
        remaining = [rank for rank in free if rank not in selected_set]
        return selected, remaining

    def _assign(self, request_id: str, ranks: tuple[int, ...]) -> None:
        req = self.requests[request_id]
        old_ranks = set(req.ranks)
        new_ranks = set(ranks)
        for rank in old_ranks - new_ranks:
            self.gpu_owner[rank] = None
        for rank in new_ranks - old_ranks:
            if self.gpu_owner[rank] not in (None, request_id):
                raise RuntimeError(
                    f"Rank {rank} is already owned by {self.gpu_owner[rank]}"
                )
            self.gpu_owner[rank] = request_id
        req.ranks = tuple(sorted(ranks))
        req.phase = RequestPhase.DIT
        req.last_scheduled_step = req.cur_step

    def _floor_allowed_power_of_two(self, count: int) -> int:
        allowed = [
            value
            for value in self.config.allowed_gpu_counts
            if value <= count and is_power_of_two(value)
        ]
        return max(allowed) if allowed else 0

    def _opt_gpu_count(self, resolution: str) -> int:
        return max(1, int(self.config.profile.opt_gpus_num.get(resolution, 1)))

    def _decision(
        self,
        request_id: str,
        old_ranks: tuple[int, ...],
        new_ranks: tuple[int, ...],
        reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "stage": "dit",
            "old_ranks": old_ranks,
            "new_ranks": new_ranks,
            "reason": reason or self.policy_name,
            "policy": self.policy_name,
        }

    def _fixed_start_decision(
        self, request_id: str, free: list[int], target_k: int
    ) -> tuple[dict[str, Any], list[int]]:
        ranks, free = self._take_preferred_free_ranks(free, target_k)
        self._assign(request_id, ranks)
        return self._decision(request_id, (), ranks), free

    def _target_k_for_free(self, req: DDiTRequestState, free_count: int) -> int:
        return self._floor_allowed_power_of_two(
            min(self._opt_gpu_count(req.resolution), free_count)
        )

    def _starvation_score(self, req: DDiTRequestState) -> float:
        current_k = max(1, len(req.ranks))
        opt_k = self._opt_gpu_count(req.resolution)
        if current_k >= opt_k:
            return 0.0
        current_t = self.config.profile.per_step_time(req.resolution, current_k)
        opt_t = self.config.profile.per_step_time(req.resolution, opt_k)
        lag_time = max(0.0, current_t - opt_t)
        lag_steps = max(0, req.cur_step - req.last_scheduled_step)
        return lag_steps * lag_time


class NaiveScheduler(ProfileBackedScheduler):
    policy_name = "naive"

    def schedule(self) -> list[dict[str, Any]]:
        decisions: list[dict[str, Any]] = []
        while self.waiting:
            free = self._free_ranks()
            request_id = self.waiting[0]
            req = self.requests[request_id]
            target_k = self._opt_gpu_count(req.resolution)
            if (
                target_k not in self.config.allowed_gpu_counts
                or not is_power_of_two(target_k)
                or len(free) < target_k
            ):
                break
            self.waiting.popleft()
            decision, _free = self._fixed_start_decision(request_id, free, target_k)
            decisions.append(decision)
        return decisions


class NaiveGreedyScheduler(ProfileBackedScheduler):
    policy_name = "naive_greedy"

    def schedule(self) -> list[dict[str, Any]]:
        decisions: list[dict[str, Any]] = []
        while self.waiting:
            free = self._free_ranks()
            if not free:
                break
            request_id = self.waiting[0]
            req = self.requests[request_id]
            target_k = self._target_k_for_free(req, len(free))
            if target_k <= 0:
                break
            self.waiting.popleft()
            decision, _free = self._fixed_start_decision(request_id, free, target_k)
            decisions.append(decision)
        return decisions


class WSJFScheduler(ProfileBackedScheduler):
    policy_name = "wsjf"

    def __init__(self, config: ProfileSchedulerConfig):
        super().__init__(config)
        self.window: deque[str] = deque()

    def has_waiting_requests(self) -> bool:
        return bool(self.waiting or self.window)

    def _fill_window(self) -> None:
        while self.waiting and len(self.window) < self.config.window_size:
            self.window.append(self.waiting.popleft())

    def _pick_shortest_job(
        self, free_count: int
    ) -> tuple[str | None, int, float | None]:
        best_request_id = None
        best_k = 0
        best_time = None
        for request_id in self.window:
            req = self.requests[request_id]
            target_k = self._target_k_for_free(req, free_count)
            if target_k <= 0:
                continue
            estimate = self.config.profile.estimate_remaining_time(req, target_k)
            if best_time is None or estimate < best_time:
                best_request_id = request_id
                best_k = target_k
                best_time = estimate
        return best_request_id, best_k, best_time

    def schedule(self) -> list[dict[str, Any]]:
        decisions: list[dict[str, Any]] = []
        while True:
            self._fill_window()
            free = self._free_ranks()
            if not free or not self.window:
                break
            request_id, target_k, _estimate = self._pick_shortest_job(len(free))
            if request_id is None or target_k <= 0:
                break
            self.window.remove(request_id)
            decision, _free = self._fixed_start_decision(request_id, free, target_k)
            decisions.append(decision)
        return decisions


class WSJFScaleUpScheduler(WSJFScheduler):
    policy_name = "wsjf_scale_up"

    def schedule(self) -> list[dict[str, Any]]:
        decisions: list[dict[str, Any]] = []
        free = self._free_ranks()
        hungry_heap: list[tuple[float, str]] = []
        for req in self.requests.values():
            if req.phase != RequestPhase.DIT:
                continue
            score = self._starvation_score(req)
            if score > 0:
                heapq.heappush(hungry_heap, (-score, req.request_id))

        while hungry_heap and free:
            _neg_score, request_id = heapq.heappop(hungry_heap)
            req = self.requests[request_id]
            opt_k = self._opt_gpu_count(req.resolution)
            target_k = self._floor_allowed_power_of_two(
                min(opt_k, len(req.ranks) + len(free))
            )
            if target_k <= len(req.ranks):
                continue
            add_count = target_k - len(req.ranks)
            selected, free = self._take_preferred_free_ranks(free, add_count)
            old_ranks = req.ranks
            new_ranks = tuple(sorted(req.ranks + selected))
            self._assign(request_id, new_ranks)
            decisions.append(
                self._decision(
                    request_id,
                    old_ranks,
                    new_ranks,
                    reason="wsjf_scale_up",
                )
            )

        while True:
            self._fill_window()
            if not free or not self.window:
                break
            request_id, target_k, _estimate = self._pick_shortest_job(len(free))
            if request_id is None or target_k <= 0:
                break
            self.window.remove(request_id)
            decision, free = self._fixed_start_decision(request_id, free, target_k)
            decisions.append(decision)
        return decisions


def build_hungry_scheduler_config(server_args: Any, world_size: int) -> DDiTSchedulerConfig:
    local_ranks = parse_local_ranks(
        getattr(server_args, "ddit_local_ranks", None), world_size
    )
    allowed_gpu_counts = parse_allowed_gpu_counts(
        getattr(server_args, "ddit_allowed_gpu_counts", None), len(local_ranks)
    )
    config = DDiTSchedulerConfig(
        local_ranks=local_ranks,
        allowed_gpu_counts=allowed_gpu_counts,
    )
    profile = ProfileStore.load(server_args)
    config.opt_gpus_num = dict(profile.opt_gpus_num)
    config.dit_step_times = {
        resolution: dict(times) for resolution, times in profile.dit_step_times.items()
    }
    return config


def build_fixed_baseline_scheduler_config(
    server_args: Any, world_size: int
) -> FixedBaselineSchedulerConfig:
    local_ranks = parse_local_ranks(
        getattr(server_args, "ddit_local_ranks", None), world_size
    )
    allowed_gpu_counts = parse_allowed_gpu_counts(
        getattr(server_args, "ddit_allowed_gpu_counts", None), len(local_ranks)
    )
    baseline_gpus = int(getattr(server_args, "ddit_baseline_gpus", 1))
    return FixedBaselineSchedulerConfig(
        local_ranks=local_ranks,
        baseline_gpus=baseline_gpus,
        allowed_gpu_counts=allowed_gpu_counts,
    )


def build_profile_scheduler_config(
    server_args: Any, world_size: int
) -> ProfileSchedulerConfig:
    local_ranks = parse_local_ranks(
        getattr(server_args, "ddit_local_ranks", None), world_size
    )
    allowed_gpu_counts = parse_allowed_gpu_counts(
        getattr(server_args, "ddit_allowed_gpu_counts", None), len(local_ranks)
    )
    window_size = max(1, int(getattr(server_args, "ddit_window_size", 8)))
    return ProfileSchedulerConfig(
        local_ranks=local_ranks,
        allowed_gpu_counts=allowed_gpu_counts,
        profile=ProfileStore.load(server_args),
        window_size=window_size,
    )


def build_profile_scheduler(
    schedule_policy: str, server_args: Any, world_size: int
) -> ProfileBackedScheduler:
    config = build_profile_scheduler_config(server_args, world_size)
    if schedule_policy == "naive":
        return NaiveScheduler(config)
    if schedule_policy == "naive_greedy":
        return NaiveGreedyScheduler(config)
    if schedule_policy == "wsjf":
        return WSJFScheduler(config)
    if schedule_policy == "wsjf_scale_up":
        return WSJFScaleUpScheduler(config)
    raise ValueError(f"Unsupported profile-backed DDiT policy: {schedule_policy}")
