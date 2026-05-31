"""Discrete-event simulator for two-node DDiT scheduling experiments."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_PROFILE_PATH = "examples/multimodal_gen/ddit_profile_a100_data.json"
DEFAULT_POLICIES = (
    "naive",
    "naive_greedy",
    "hungry_first",
    "wsjf",
    "wsjf_scale_up",
)
ALLOWED_GPU_COUNTS = (1, 2, 4, 8)


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_ratios(value: str) -> list[float]:
    ratios = [float(part) for part in parse_csv(value)]
    if not ratios:
        raise ValueError("--ratios cannot be empty")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("--ratios cannot contain negative values")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"--ratios must sum to 1.0, got {total}")
    return ratios


def counts_from_ratios(num_requests: int, ratios: list[float]) -> list[int]:
    if num_requests <= 0:
        raise ValueError("--num-requests must be positive")
    counts = [round(num_requests * ratio) for ratio in ratios[:-1]]
    last = num_requests - sum(counts)
    if last < 0:
        counts.append(0)
        over = -last
        for idx in sorted(range(len(counts) - 1), key=lambda i: counts[i], reverse=True):
            take = min(over, counts[idx])
            counts[idx] -= take
            over -= take
            if over == 0:
                break
        counts[-1] = num_requests - sum(counts[:-1])
    else:
        counts.append(last)
    return counts


def parse_rate(value: str) -> float | None:
    if str(value).lower() == "burst":
        return None
    rate = float(value)
    if rate <= 0:
        raise ValueError("--rate must be positive or 'burst'")
    return rate


def normalize_model_id(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-")
    if "wan" in normalized and "2.1" in normalized and "1.3" in normalized:
        return "wan2.1-t2v-1.3b"
    if "z-image" in normalized or "zimage" in normalized:
        return "z-image"
    return normalized


def repo_root() -> Path:
    script = Path(__file__).resolve()
    for candidate in (script.parent, *script.parents):
        if (candidate / "python" / "sglang").exists() or (candidate / ".git").exists():
            return candidate
    return Path.cwd()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root() / candidate).resolve()


@dataclass(frozen=True)
class DDiTProfile:
    model_id: str
    opt_gpus_num: dict[str, int]
    opt_vae_k: dict[str, int]
    dit_step_times: dict[str, dict[int, float]]
    vae_times: dict[str, dict[int, float]]
    text_encoder_times: dict[str, float]
    dit_step_num: int

    @classmethod
    def load(cls, profile_path: str | Path, model_id: str) -> "DDiTProfile":
        path = resolve_path(profile_path)
        if not path.exists():
            raise FileNotFoundError(f"DDiT profile path does not exist: {path}")
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)

        normalized = normalize_model_id(model_id)
        if "models" in payload:
            models = payload.get("models") or {}
            if model_id in models:
                model_payload = models[model_id]
                resolved_id = model_id
            elif normalized in models:
                model_payload = models[normalized]
                resolved_id = normalized
            else:
                raise ValueError(
                    f"Model id {model_id!r} was not found in DDiT profile {path}"
                )
        else:
            model_payload = payload
            resolved_id = normalized

        dit_step_num = int(model_payload.get("dit_step_num", 50))
        return cls(
            model_id=resolved_id,
            opt_gpus_num={
                str(resolution): int(count)
                for resolution, count in (model_payload.get("opt_gpus_num") or {}).items()
            },
            opt_vae_k={
                str(resolution): int(count)
                for resolution, count in (model_payload.get("opt_vae_k") or {}).items()
            },
            dit_step_times={
                str(resolution): {int(k): float(v) for k, v in times.items()}
                for resolution, times in (model_payload.get("dit_step_times") or {}).items()
            },
            vae_times={
                str(resolution): {int(k): float(v) for k, v in times.items()}
                for resolution, times in (model_payload.get("vae_times") or {}).items()
            },
            text_encoder_times={
                str(resolution): float(value)
                for resolution, value in (
                    model_payload.get("text_encoder_times") or {}
                ).items()
            },
            dit_step_num=dit_step_num,
        )

    def opt_gpu_count(self, resolution: str, allowed: tuple[int, ...]) -> int:
        opt = int(self.opt_gpus_num.get(resolution, 1))
        return max(1, min(opt, max(allowed)))

    def opt_vae_count(self, resolution: str, allowed: tuple[int, ...]) -> int:
        opt = int(self.opt_vae_k.get(resolution, 1))
        return max(1, min(opt, max(allowed)))

    def per_step_time(self, resolution: str, gpu_count: int) -> float:
        profile = self.dit_step_times.get(resolution, {})
        if gpu_count in profile:
            return profile[gpu_count]
        if profile:
            smaller = [k for k in profile if k <= gpu_count]
            if smaller:
                return profile[max(smaller)]
            return profile[min(profile)]
        return 1.0

    def text_time(self, resolution: str) -> float:
        return float(self.text_encoder_times.get(resolution, 0.0))

    def vae_time(self, resolution: str, gpu_count: int) -> float:
        profile = self.vae_times.get(resolution, {})
        if gpu_count in profile:
            return profile[gpu_count]
        if profile:
            smaller = [k for k in profile if k <= gpu_count]
            if smaller:
                return profile[max(smaller)]
            return profile[min(profile)]
        return 0.0

    def unit_slo(self, resolution: str, *, gpu_count: int = 8) -> float | None:
        text = self.text_encoder_times.get(resolution)
        dit = self.dit_step_times.get(resolution, {}).get(gpu_count)
        vae = self.vae_times.get(resolution, {}).get(gpu_count)
        if text is None or dit is None or vae is None:
            return None
        return float(text) + float(vae) + float(dit) * int(self.dit_step_num)


@dataclass
class SimRequest:
    request_id: str
    resolution: str
    arrival_time: float
    total_steps: int
    cur_step: int = 0
    node_id: int | None = None
    phase: str = "created"
    ranks: tuple[int, ...] = ()
    final_dit_ranks: tuple[int, ...] = ()
    vae_ranks: tuple[int, ...] = ()
    last_scheduled_step: int = 0
    arrival_index: int = 0
    times: dict[str, float] = field(default_factory=dict)


@dataclass
class NodeState:
    node_id: int
    ranks: tuple[int, ...]
    te_busy_request: str | None = None
    owners: dict[int, str | None] = field(default_factory=dict)
    dit_waiting: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.owners:
            self.owners = {rank: None for rank in self.ranks}

    def free_ranks(self, *, te_resource_mode: str = "lane") -> list[int]:
        if te_resource_mode == "exclusive" and self.te_busy_request is not None:
            return []
        return [rank for rank in self.ranks if self.owners[rank] is None]

    def owned_ranks(self, request_id: str) -> tuple[int, ...]:
        return tuple(rank for rank, owner in self.owners.items() if owner == request_id)


@dataclass(frozen=True)
class SimulationConfig:
    profile_path: str
    model_id: str = "z-image"
    policy: str = "hungry_first"
    num_nodes: int = 2
    gpus_per_node: int = 8
    allowed_gpu_counts: tuple[int, ...] = ALLOWED_GPU_COUNTS
    window_size: int = 8
    resolutions: tuple[str, ...] = ("720p", "2k")
    ratios: tuple[float, ...] = (0.75, 0.25)
    num_requests: int = 128
    rate: str = "1.0"
    seed: int = 42
    te_resource_mode: str = "lane"
    out_dir: str = "outputs/ddit_mock_sim"


class DDiTMockSimulator:
    def __init__(self, config: SimulationConfig):
        if config.policy not in DEFAULT_POLICIES:
            raise ValueError(f"Unsupported policy: {config.policy}")
        if config.te_resource_mode not in ("lane", "exclusive"):
            raise ValueError("--te-resource-mode must be 'lane' or 'exclusive'")
        if len(config.resolutions) != len(config.ratios):
            raise ValueError("resolutions and ratios length mismatch")

        self.config = config
        self.profile = DDiTProfile.load(config.profile_path, config.model_id)
        self.nodes = [
            NodeState(
                node_id=node_id,
                ranks=tuple(
                    range(
                        node_id * config.gpus_per_node,
                        (node_id + 1) * config.gpus_per_node,
                    )
                ),
            )
            for node_id in range(config.num_nodes)
        ]
        self.requests = self._build_workload()
        self.pending_te: list[str] = []
        self.events: list[tuple[float, int, str, str]] = []
        self.event_seq = 0
        self.rank_switch_records: list[dict[str, Any]] = []
        self.op_trace_records: list[dict[str, Any]] = []
        self.completed: set[str] = set()
        self.dit_step_pending: set[str] = set()
        for req in self.requests:
            self._push_event(req.arrival_time, "arrival", req.request_id)

    def _build_workload(self) -> list[SimRequest]:
        counts = counts_from_ratios(self.config.num_requests, list(self.config.ratios))
        requests: list[SimRequest] = []
        rate = parse_rate(self.config.rate)
        next_arrival = 0.0
        local_index = 0
        for resolution, count in zip(self.config.resolutions, counts):
            for idx in range(count):
                requests.append(
                    SimRequest(
                        request_id=f"ddit_{resolution}_{idx:05d}_{self.config.seed}",
                        resolution=resolution,
                        arrival_time=next_arrival,
                        total_steps=self.profile.dit_step_num,
                        arrival_index=local_index,
                    )
                )
                local_index += 1
                if rate is not None:
                    next_arrival += 1.0 / rate
        rng = random.Random(self.config.seed)
        rng.shuffle(requests)
        for idx, req in enumerate(requests):
            req.arrival_index = idx
            req.arrival_time = 0.0 if rate is None else idx * (1.0 / rate)
            req.times["add_time"] = req.arrival_time
            req.phase = "pending_te"
        return requests

    def _push_event(self, when: float, event_type: str, request_id: str) -> None:
        self.event_seq += 1
        heapq.heappush(self.events, (when, self.event_seq, event_type, request_id))

    def _opt_k(self, req: SimRequest) -> int:
        return self.profile.opt_gpu_count(req.resolution, self.config.allowed_gpu_counts)

    def _floor_allowed_power_of_two(self, count: int) -> int:
        allowed = [
            value
            for value in self.config.allowed_gpu_counts
            if value <= count and value > 0 and value & (value - 1) == 0
        ]
        return max(allowed) if allowed else 0

    @staticmethod
    def _select_preferred_rank_tuple(free_ranks: list[int] | tuple[int, ...], count: int) -> tuple[int, ...]:
        ranks = tuple(sorted(int(rank) for rank in free_ranks))
        if count <= 0 or count > len(ranks):
            return ()
        rank_set = set(ranks)
        for start in ranks:
            candidate = tuple(range(start, start + count))
            if all(rank in rank_set for rank in candidate):
                return candidate
        return tuple(ranks[:count])

    def _candidate_start_k(self, req: SimRequest, free_count: int) -> int:
        if self.config.policy == "naive":
            opt = self._opt_k(req)
            return opt if free_count >= opt else 0
        return self._floor_allowed_power_of_two(min(self._opt_k(req), free_count))

    def _node_sort_key(self, req: SimRequest, node: NodeState) -> tuple[float, int, int]:
        free = node.free_ranks(te_resource_mode=self.config.te_resource_mode)
        target_k = self._candidate_start_k(req, len(free))
        if target_k <= 0:
            return (math.inf, -len(free), node.node_id)
        estimate = self.profile.per_step_time(req.resolution, target_k) * req.total_steps
        return (estimate, -len(free), node.node_id)

    def _eligible_nodes_for_te(self, req: SimRequest) -> list[NodeState]:
        eligible = []
        for node in self.nodes:
            if node.te_busy_request is not None:
                continue
            free = node.free_ranks(te_resource_mode=self.config.te_resource_mode)
            if self._candidate_start_k(req, len(free)) > 0:
                eligible.append(node)
        return eligible

    def _start_te(self, now: float, req: SimRequest, node: NodeState) -> None:
        req.node_id = node.node_id
        req.phase = "text_encoder"
        req.times["text_start"] = now
        node.te_busy_request = req.request_id
        duration = self.profile.text_time(req.resolution)
        self.op_trace_records.append(
            {
                "time": now,
                "event": "text_encoder_start",
                "request_id": req.request_id,
                "resolution": req.resolution,
                "node_id": node.node_id,
                "duration_s": duration,
            }
        )
        self._push_event(now + duration, "te_done", req.request_id)

    def _schedule_te(self, now: float) -> bool:
        if not self.pending_te:
            return False
        if self.config.policy in ("wsjf", "wsjf_scale_up"):
            return self._schedule_te_wsjf(now)
        changed = False
        while self.pending_te:
            req = self._request(self.pending_te[0])
            eligible = self._eligible_nodes_for_te(req)
            if not eligible:
                break
            node = min(eligible, key=lambda item: self._node_sort_key(req, item))
            self.pending_te.pop(0)
            self._start_te(now, req, node)
            changed = True
        return changed

    def _schedule_te_wsjf(self, now: float) -> bool:
        changed = False
        while self.pending_te:
            window = self.pending_te[: self.config.window_size]
            best: tuple[float, int, str, NodeState] | None = None
            for order, request_id in enumerate(window):
                req = self._request(request_id)
                for node in self._eligible_nodes_for_te(req):
                    free = node.free_ranks(te_resource_mode=self.config.te_resource_mode)
                    target_k = self._candidate_start_k(req, len(free))
                    if target_k <= 0:
                        continue
                    estimate = req.total_steps * self.profile.per_step_time(
                        req.resolution, target_k
                    )
                    key = (estimate, order, request_id, node)
                    if best is None or key[:3] < best[:3]:
                        best = key
            if best is None:
                break
            _estimate, _order, request_id, node = best
            self.pending_te.remove(request_id)
            self._start_te(now, self._request(request_id), node)
            changed = True
        return changed

    def _te_done(self, now: float, req: SimRequest) -> None:
        node = self._node(req.node_id)
        node.te_busy_request = None
        req.phase = "dit_waiting"
        req.times["text_end"] = now
        node.dit_waiting.append(req.request_id)
        self.op_trace_records.append(
            {
                "time": now,
                "event": "text_encoder_done",
                "request_id": req.request_id,
                "resolution": req.resolution,
                "node_id": node.node_id,
            }
        )

    def _assign_ranks(self, req: SimRequest, new_ranks: tuple[int, ...], now: float, reason: str) -> None:
        node = self._node(req.node_id)
        old_ranks = req.ranks
        old_set = set(old_ranks)
        new_set = set(new_ranks)
        for rank in old_set - new_set:
            node.owners[rank] = None
        for rank in new_set - old_set:
            if node.owners[rank] not in (None, req.request_id):
                raise RuntimeError(f"Rank {rank} is already owned by {node.owners[rank]}")
            node.owners[rank] = req.request_id
        req.ranks = tuple(sorted(new_ranks))
        req.last_scheduled_step = req.cur_step
        req.phase = "dit"
        self.rank_switch_records.append(
            {
                "time": now,
                "request_id": req.request_id,
                "resolution": req.resolution,
                "node_id": node.node_id,
                "stage": "dit",
                "step": req.cur_step,
                "old_ranks": list(old_ranks),
                "new_ranks": list(req.ranks),
                "old_k": len(old_ranks),
                "new_k": len(req.ranks),
                "reason": reason,
                "policy": self.config.policy,
            }
        )

    def _schedule_dit_starts(self, now: float) -> bool:
        if self.config.policy in ("wsjf", "wsjf_scale_up"):
            return self._schedule_dit_starts_wsjf(now)
        changed = False
        for node in self.nodes:
            while node.dit_waiting:
                req = self._request(node.dit_waiting[0])
                free = node.free_ranks(te_resource_mode=self.config.te_resource_mode)
                target_k = self._candidate_start_k(req, len(free))
                if target_k <= 0:
                    break
                ranks = self._select_preferred_rank_tuple(free, target_k)
                if not ranks:
                    break
                node.dit_waiting.pop(0)
                req.times.setdefault("dit_start", now)
                self._assign_ranks(req, ranks, now, "waiting_queue")
                changed = True
        return changed

    def _schedule_dit_starts_wsjf(self, now: float) -> bool:
        changed = False
        for node in self.nodes:
            while node.dit_waiting:
                free = node.free_ranks(te_resource_mode=self.config.te_resource_mode)
                if not free:
                    break
                best: tuple[float, int, str, int] | None = None
                for idx, request_id in enumerate(node.dit_waiting[: self.config.window_size]):
                    req = self._request(request_id)
                    target_k = self._candidate_start_k(req, len(free))
                    if target_k <= 0:
                        continue
                    estimate = (req.total_steps - req.cur_step) * self.profile.per_step_time(
                        req.resolution, target_k
                    )
                    key = (estimate, idx, request_id, target_k)
                    if best is None or key[:3] < best[:3]:
                        best = key
                if best is None:
                    break
                _estimate, _idx, request_id, target_k = best
                node.dit_waiting.remove(request_id)
                req = self._request(request_id)
                ranks = self._select_preferred_rank_tuple(free, target_k)
                req.times.setdefault("dit_start", now)
                self._assign_ranks(req, ranks, now, "wsjf")
                changed = True
        return changed

    def _starvation_score(self, req: SimRequest) -> float:
        current_k = max(1, len(req.ranks))
        opt_k = self._opt_k(req)
        if current_k >= opt_k:
            return 0.0
        current_t = self.profile.per_step_time(req.resolution, current_k)
        opt_t = self.profile.per_step_time(req.resolution, opt_k)
        lag_time = max(0.0, current_t - opt_t)
        lag_steps = max(0, req.cur_step - req.last_scheduled_step)
        return lag_steps * lag_time

    def _schedule_scale_up(self, now: float) -> bool:
        if self.config.policy not in ("hungry_first", "wsjf_scale_up"):
            return False
        changed = False
        while True:
            best: tuple[float, int, str, SimRequest, list[int]] | None = None
            for req in self.requests:
                if req.phase != "dit":
                    continue
                node = self._node(req.node_id)
                free = node.free_ranks(te_resource_mode=self.config.te_resource_mode)
                if not free:
                    continue
                opt_k = self._opt_k(req)
                target_k = self._floor_allowed_power_of_two(
                    min(opt_k, len(req.ranks) + len(free))
                )
                if target_k <= len(req.ranks):
                    continue
                score = self._starvation_score(req)
                if score <= 0:
                    continue
                key = (-score, req.arrival_index, req.request_id, req, free)
                if best is None or key[:3] < best[:3]:
                    best = key
            if best is None:
                break
            _neg_score, _arrival_index, _request_id, req, free = best
            opt_k = self._opt_k(req)
            target_k = self._floor_allowed_power_of_two(
                min(opt_k, len(req.ranks) + len(free))
            )
            add_count = target_k - len(req.ranks)
            add_ranks = self._select_preferred_rank_tuple(free, add_count)
            if not add_ranks:
                break
            reason = "hungry_first" if self.config.policy == "hungry_first" else "wsjf_scale_up"
            self._assign_ranks(req, tuple(sorted(req.ranks + add_ranks)), now, reason)
            changed = True
        return changed

    def _schedule_dit_steps(self, now: float) -> bool:
        changed = False
        for req in self.requests:
            if req.phase != "dit" or req.request_id in self.dit_step_pending:
                continue
            if req.cur_step >= req.total_steps:
                continue
            duration = self.profile.per_step_time(req.resolution, len(req.ranks))
            step = req.cur_step
            self.dit_step_pending.add(req.request_id)
            self.op_trace_records.append(
                {
                    "time": now,
                    "event": "dit_step_start",
                    "request_id": req.request_id,
                    "resolution": req.resolution,
                    "node_id": req.node_id,
                    "step": step,
                    "ranks": list(req.ranks),
                    "k": len(req.ranks),
                    "duration_s": duration,
                }
            )
            self._push_event(now + duration, "dit_step_done", req.request_id)
            changed = True
        return changed

    def _dit_step_done(self, now: float, req: SimRequest) -> None:
        self.dit_step_pending.discard(req.request_id)
        step = req.cur_step
        req.cur_step += 1
        self.op_trace_records.append(
            {
                "time": now,
                "event": "dit_step_done",
                "request_id": req.request_id,
                "resolution": req.resolution,
                "node_id": req.node_id,
                "step": step,
                "cur_step": req.cur_step,
                "ranks": list(req.ranks),
                "k": len(req.ranks),
            }
        )
        if req.cur_step >= req.total_steps:
            req.times["dit_end"] = now
            self._transition_to_vae(now, req)

    def _transition_to_vae(self, now: float, req: SimRequest) -> None:
        node = self._node(req.node_id)
        final_dit_ranks = req.ranks
        req.final_dit_ranks = final_dit_ranks
        if self.config.policy == "hungry_first":
            vae_k = self.profile.opt_vae_count(req.resolution, self.config.allowed_gpu_counts)
            vae_ranks = tuple(sorted(final_dit_ranks))[:vae_k]
        else:
            vae_ranks = final_dit_ranks
            vae_k = len(vae_ranks)
        for rank in set(final_dit_ranks) - set(vae_ranks):
            node.owners[rank] = None
        req.ranks = tuple(sorted(vae_ranks))
        req.vae_ranks = req.ranks
        req.phase = "vae"
        req.times["vae_start"] = now
        self.rank_switch_records.append(
            {
                "time": now,
                "request_id": req.request_id,
                "resolution": req.resolution,
                "node_id": req.node_id,
                "stage": "vae",
                "step": req.cur_step,
                "old_ranks": list(final_dit_ranks),
                "new_ranks": list(req.ranks),
                "old_k": len(final_dit_ranks),
                "new_k": len(req.ranks),
                "reason": (
                    f"{self.config.policy}_same_ranks"
                    if self.config.policy != "hungry_first"
                    else "dit_to_vae"
                ),
                "policy": self.config.policy if self.config.policy != "hungry_first" else "ddit_vae_gpus",
            }
        )
        duration = self.profile.vae_time(req.resolution, vae_k)
        self.op_trace_records.append(
            {
                "time": now,
                "event": "vae_start",
                "request_id": req.request_id,
                "resolution": req.resolution,
                "node_id": req.node_id,
                "ranks": list(req.ranks),
                "k": vae_k,
                "duration_s": duration,
            }
        )
        self._push_event(now + duration, "vae_done", req.request_id)

    def _vae_done(self, now: float, req: SimRequest) -> None:
        node = self._node(req.node_id)
        for rank in req.ranks:
            node.owners[rank] = None
        req.phase = "done"
        req.times["vae_end"] = now
        req.ranks = ()
        self.completed.add(req.request_id)
        self.op_trace_records.append(
            {
                "time": now,
                "event": "vae_done",
                "request_id": req.request_id,
                "resolution": req.resolution,
                "node_id": req.node_id,
            }
        )

    def _request(self, request_id: str) -> SimRequest:
        for req in self.requests:
            if req.request_id == request_id:
                return req
        raise KeyError(request_id)

    def _node(self, node_id: int | None) -> NodeState:
        if node_id is None:
            raise RuntimeError("Request is not assigned to a node")
        return self.nodes[int(node_id)]

    def _process_event(self, now: float, event_type: str, request_id: str) -> None:
        req = self._request(request_id)
        if event_type == "arrival":
            req.phase = "pending_te"
            self.pending_te.append(req.request_id)
            self.op_trace_records.append(
                {
                    "time": now,
                    "event": "arrival",
                    "request_id": req.request_id,
                    "resolution": req.resolution,
                }
            )
        elif event_type == "te_done":
            self._te_done(now, req)
        elif event_type == "dit_step_done":
            self._dit_step_done(now, req)
        elif event_type == "vae_done":
            self._vae_done(now, req)
        else:
            raise ValueError(f"Unknown event type: {event_type}")

    def _schedule_until_idle(self, now: float) -> None:
        while True:
            changed = False
            changed = self._schedule_scale_up(now) or changed
            changed = self._schedule_dit_starts(now) or changed
            changed = self._schedule_te(now) or changed
            changed = self._schedule_dit_steps(now) or changed
            if not changed:
                break

    def run(self) -> dict[str, Any]:
        now = 0.0
        while self.events:
            now, _seq, event_type, request_id = heapq.heappop(self.events)
            same_time = [(event_type, request_id)]
            while self.events and abs(self.events[0][0] - now) < 1e-12:
                _time, _seq, event_type, request_id = heapq.heappop(self.events)
                same_time.append((event_type, request_id))
            for event_type, request_id in same_time:
                self._process_event(now, event_type, request_id)
            self._schedule_until_idle(now)
            if len(self.completed) == len(self.requests):
                break
        if len(self.completed) != len(self.requests):
            pending = [req.request_id for req in self.requests if req.phase != "done"]
            raise RuntimeError(
                f"Simulation did not complete all requests: "
                f"{len(self.completed)}/{len(self.requests)} done, pending={pending[:8]}"
            )
        return self._build_result()

    def _build_result(self) -> dict[str, Any]:
        lifecycle = [self._lifecycle_row(req) for req in self.requests]
        summary = summarize_lifecycle(lifecycle, self.profile)
        summary.update(
            {
                "policy": self.config.policy,
                "rate": self.config.rate,
                "model_id": self.profile.model_id,
                "num_requests": len(self.requests),
                "num_nodes": self.config.num_nodes,
                "gpus_per_node": self.config.gpus_per_node,
                "te_resource_mode": self.config.te_resource_mode,
            }
        )
        return {
            "lifecycle": lifecycle,
            "rank_switch": self.rank_switch_records,
            "op_trace": self.op_trace_records,
            "summary": summary,
        }

    def _lifecycle_row(self, req: SimRequest) -> dict[str, Any]:
        add_time = req.times.get("add_time", req.arrival_time)
        text_start = req.times.get("text_start")
        text_end = req.times.get("text_end")
        dit_start = req.times.get("dit_start")
        dit_end = req.times.get("dit_end")
        vae_start = req.times.get("vae_start")
        vae_end = req.times.get("vae_end")
        return {
            "request_id": req.request_id,
            "resolution": req.resolution,
            "policy": self.config.policy,
            "rate": self.config.rate,
            "node_id": req.node_id,
            "add_time": add_time,
            "text_start": text_start,
            "text_end": text_end,
            "dit_start": dit_start,
            "dit_end": dit_end,
            "vae_start": vae_start,
            "vae_end": vae_end,
            "total_latency_s": None if vae_end is None else vae_end - add_time,
            "text_latency_s": _duration(text_start, text_end),
            "dit_latency_s": _duration(dit_start, dit_end),
            "vae_latency_s": _duration(vae_start, vae_end),
            "queue_to_text_s": None if text_start is None else text_start - add_time,
            "queue_to_dit_s": _duration(text_end, dit_start),
            "steps": req.cur_step,
            "final_dit_k": len(req.final_dit_ranks),
            "vae_k": len(req.vae_ranks),
            "final_dit_ranks": json.dumps(list(req.final_dit_ranks)),
            "vae_ranks": json.dumps(list(req.vae_ranks)),
        }


def _duration(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return end - start


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def summarize_lifecycle(rows: list[dict[str, Any]], profile: DDiTProfile) -> dict[str, Any]:
    latencies = [float(row["total_latency_s"]) for row in rows if row["total_latency_s"] is not None]
    starts = [float(row["add_time"]) for row in rows]
    ends = [float(row["vae_end"]) for row in rows if row["vae_end"] is not None]
    slo5_hits = 0
    slo10_hits = 0
    slo_total = 0
    for row in rows:
        latency = row["total_latency_s"]
        if latency is None:
            continue
        unit_slo = profile.unit_slo(str(row["resolution"]), gpu_count=8)
        if unit_slo is None:
            continue
        slo_total += 1
        if latency <= unit_slo * 5:
            slo5_hits += 1
        if latency <= unit_slo * 10:
            slo10_hits += 1
    makespan = (max(ends) - min(starts)) if starts and ends else 0.0
    return {
        "completed_count": len(latencies),
        "p50": percentile(latencies, 0.50),
        "p90": percentile(latencies, 0.90),
        "p99": percentile(latencies, 0.99),
        "mean": sum(latencies) / len(latencies) if latencies else 0.0,
        "max": max(latencies) if latencies else 0.0,
        "makespan": makespan,
        "throughput": len(latencies) / makespan if makespan > 0 else 0.0,
        "slo5": slo5_hits / slo_total if slo_total else 0.0,
        "slo10": slo10_hits / slo_total if slo_total else 0.0,
    }


def write_outputs(result: dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "ddit_lifecycle.csv", result["lifecycle"])
    _write_jsonl(out / "ddit_rank_switch.jsonl", result["rank_switch"])
    _write_jsonl(out / "ddit_op_trace.jsonl", result["op_trace"])
    _write_csv(out / "ddit_lifecycle_summary.csv", [result["summary"]])
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2, sort_keys=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def run_simulation(config: SimulationConfig, *, write: bool = True) -> dict[str, Any]:
    result = DDiTMockSimulator(config).run()
    if write:
        write_outputs(result, config.out_dir)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-path", default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--model-id", default="z-image")
    parser.add_argument("--num-nodes", type=int, default=2)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--policy", choices=DEFAULT_POLICIES, default="hungry_first")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--resolutions", default="720p,2k")
    parser.add_argument("--ratios", default="0.75,0.25")
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--rate", default="1.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--te-resource-mode", choices=("lane", "exclusive"), default="lane")
    parser.add_argument("--out-dir", required=True)
    return parser


def config_from_args(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        profile_path=args.profile_path,
        model_id=args.model_id,
        policy=args.policy,
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        window_size=args.window_size,
        resolutions=tuple(parse_csv(args.resolutions)),
        ratios=tuple(parse_ratios(args.ratios)),
        num_requests=args.num_requests,
        rate=str(args.rate),
        seed=args.seed,
        te_resource_mode=args.te_resource_mode,
        out_dir=args.out_dir,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    result = run_simulation(config, write=True)
    print(
        "DDiT mock simulation completed: "
        f"policy={config.policy} rate={config.rate} "
        f"completed={result['summary']['completed_count']} "
        f"p50={result['summary']['p50']:.3f}s "
        f"p99={result['summary']['p99']:.3f}s "
        f"out_dir={config.out_dir}"
    )


if __name__ == "__main__":
    main()
