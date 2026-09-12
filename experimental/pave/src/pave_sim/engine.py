# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic stage-safe event engine adapted from NVIDIA's reference simulator.

Execution sees request truth. Routing and source selection receive only progress
observations. GPU-group launch barriers are independent, while bin commits are global.
"""

from __future__ import annotations

import copy
import heapq
import math
import random
from collections import Counter, deque
from dataclasses import asdict

from .config import RunSpec, Settings
from .inputs import Case, Trace, template_counts
from .monitor import Monitor, WorkEstimator
from .records import Journal
from .scheduling import Scheduler
from .selection import choose_forward, score_sources
from .state import STAGES, ConversionGroup, Execution, Lane, Physical, Progress, Transition
from .timing import SEMANTICS, elapsed, seconds, ticks

PRIORITIES = {"completion": 0, "boundary": 1, "group_launch": 2, "ready": 2, "monitor": 3, "arrival": 4}


class Simulator:
    def __init__(self, settings: Settings, spec: RunSpec, case: Case, trace: Trace, journal: Journal,
                 *, run_id: str | None = None, max_events: int | None = None):
        spec.validate()
        if max_events is not None and (type(max_events) is not int or max_events <= 0):
            raise ValueError("max_events must be a positive integer")
        self.run_id, self.max_events = run_id or spec.run_id, max_events
        self.settings, self.spec, self.case, self.journal = settings, spec, case, journal
        self.requests = {r.id: r for r in trace.requests(settings, spec.rate_per_min)}
        self.monitor = Monitor(spec.window_s, settings.input_tokens, (settings.short_output_tokens, settings.long_output_tokens), case.profiles)
        self.estimator = WorkEstimator(self.monitor)
        self.scheduler = Scheduler(spec.scheduler, self.estimator, spec.seed)
        self.source_selection_rng = random.Random(spec.source_selection_seed)
        self.clock_tick, self.sequence, self.completed = 0, 0, 0
        self.period_ticks = ticks(self.period, positive=True)
        self.monitor_index = 1
        self.events, self.attempts, self.flips = [], [], []
        self.pending: deque[tuple[int, bool, str]] = deque()
        self.arrived: set[int] = set()
        self.attempt_for: dict[int, int] = {}
        self.physicals: dict[str, Physical] = {}
        self.live: dict[str, Physical] = {}
        self.lanes: dict[str, Lane] = {}
        self.epochs: Counter = Counter()
        initial = case.source if spec.initial_deployment == "source" else case.target
        self.active_bin = initial["output_tokens"]
        self.transition: Transition | None = None
        self.lineage: list[tuple[dict, list[str]]] = []
        self.base_pe = {raw["id"] for raw in case.source["instances"] if "PE" in raw["stages"]}
        for raw in initial["instances"]:
            self.create_physical(raw, 0, ready=True)
        for index in range(initial["cpu_te"]["instances"]):
            raw = {"id": f"cpu-te-{index + 1:03d}", "node": "CPU", "gpu_ids": [], "hardware": "CPU", "template": "TE_CPU", "bundle_size": 0, "stages": {"TE": {"latency_s": initial["cpu_te"]["latency_s"]}}}
            self.create_physical(raw, 0, ready=True)
        for request in self.requests.values():
            self.schedule(request.arrival_s, "arrival", {"request": request.id})
        self.schedule_tick(self.period_ticks, "monitor", {})

    @property
    def clock(self) -> float:
        return seconds(self.clock_tick)

    @clock.setter
    def clock(self, value: float) -> None:
        self.clock_tick = ticks(value)

    @property
    def period(self) -> float:
        return self.settings.monitor_period_s or self.spec.window_s

    def emit(self, event: str, **fields) -> None:
        self.journal.emit(event, **fields)

    def schedule(self, time: float, kind: str, payload: dict) -> None:
        self.schedule_tick(ticks(time), kind, payload)

    def schedule_tick(self, time: int, kind: str, payload: dict) -> None:
        if self.max_events is not None and self.sequence >= self.max_events:
            raise RuntimeError(f"Simulation event budget exceeded: {self.max_events}")
        if type(time) is not int or time < self.clock_tick:
            raise RuntimeError(f"Invalid event time for {kind}: {time}")
        heapq.heappush(self.events, (time, PRIORITIES[kind], self.sequence, kind, payload))
        self.sequence += 1

    def create_physical(self, raw: dict, now: float, *, ready: bool) -> Physical:
        if raw["id"] in self.live:
            raise RuntimeError(f"Physical placement already live: {raw['id']}")
        raw = copy.deepcopy(raw)
        self.epochs[raw["id"]] += 1
        epoch = self.epochs[raw["id"]]
        uid = f"{raw['id']}@{epoch}"
        state = "ready" if ready else "launching"
        lane_ids = []
        for stage in STAGES:
            if stage in raw["stages"]:
                lane_id = f"{uid}/{stage}"
                self.lanes[lane_id] = Lane(
                    lane_id, uid, stage, len(self.lanes), raw,
                    state=state, created_s=now,
                )
                lane_ids.append(lane_id)
        physical = Physical(uid, raw, epoch, lane_ids, state, now, now if ready else None)
        self.live[raw["id"]] = self.physicals[uid] = physical
        self.emit("instance_created", time_s=now, instance_uid=uid, placement=raw, state=state)
        return physical

    def observe(self, request_id: int, time: float) -> Progress:
        request = self.requests[request_id]
        generated, steps = request.generated, request.steps
        lane = self.lanes.get(request.owner)
        if lane and lane.current and lane.current.request_id == request_id:
            if lane.stage == "PE":
                generated = lane.current.progress_at(time)
            elif lane.stage == "DiT":
                steps = lane.current.progress_at(time)
        return Progress(request_id, generated, steps)

    def advance(self, time: float) -> None:
        self.advance_tick(ticks(time))

    def advance_tick(self, time: int) -> None:
        if time < self.clock_tick:
            raise RuntimeError("Clock moved backward")
        for lane in self.lanes.values():
            if lane.current:
                request = self.requests[lane.current.request_id]
                progress = self.observe(request.id, seconds(time))
                if progress.generated < request.generated or progress.steps < request.steps:
                    raise RuntimeError("Execution progress moved backward")
                request.generated, request.steps = progress.generated, progress.steps
        self.clock_tick = time

    def route(self, request_id: int, recovery: bool, reason: str) -> None:
        request = self.requests[request_id]
        if request.owner is not None or request.stage is None:
            raise RuntimeError("Routing an owned or completed request")
        candidates = [lane for lane in self.lanes.values() if lane.stage == request.stage and lane.ready]
        lane, scores = self.scheduler.choose(candidates, self.observe(request_id, self.clock), self.clock, self.active_bin, self.observe)
        self.emit("dispatch", time_s=self.clock, request_id=request_id, stage=request.stage, scheduler=self.spec.scheduler, candidates=scores, selected_lane=lane.uid, insertion="recovery_fifo" if recovery else "tail", reason=reason)
        attempt = {
            "attempt_id": len(self.attempts), "request_id": request_id, "stage": lane.stage,
            "instance_uid": lane.physical_uid, "placement_id": lane.raw["id"], "lane_uid": lane.uid,
            "node": lane.raw["node"], "gpu_ids": lane.raw["gpu_ids"], "hardware": lane.raw["hardware"],
            "template": lane.raw["template"], "parallelism": lane.raw["bundle_size"],
            "enter_s": self.clock, "start_s": None, "exit_s": None, "exit_reason": None,
            "queue_insertion": "recovery_fifo" if recovery else "tail", "executed_work": 0.0,
            "work_unit": "token" if lane.stage == "PE" else "step" if lane.stage == "DiT" else "request",
            "generated_on_enter": request.generated, "steps_on_enter": request.steps,
        }
        self.attempts.append(attempt)
        self.attempt_for[request_id] = attempt["attempt_id"]
        request.owner = lane.uid
        (lane.recovery if recovery else lane.queue).append(request_id)
        self.emit("queue_enter", time_s=self.clock, **attempt)
        self.start(lane)

    def start(self, lane: Lane) -> None:
        if not lane.ready or lane.current or not lane.waiting():
            return
        request_id = (lane.recovery if lane.recovery else lane.queue).popleft()
        request = self.requests[request_id]
        if request.owner != lane.uid or request.stage != lane.stage:
            raise RuntimeError("Queue ownership/stage mismatch")
        attempt = self.attempts[self.attempt_for[request_id]]
        if lane.stage == "PE":
            remaining = request.output_tokens - request.generated
            if remaining <= 0:
                raise RuntimeError("A completed PE request was enqueued again")
            match = self.case.profiles.pe(lane.raw["hardware"], lane.raw["bundle_size"], request.input_tokens + request.generated, remaining)
            run = Execution(
                request_id=request_id,
                attempt_id=attempt["attempt_id"],
                stage=lane.stage,
                start_s=self.clock,
                duration_s=match.duration_s,
                initial_progress=request.generated,
                target_progress=request.output_tokens,
                pe=match,
            )
            attempt["pe_profile"] = match.record()
        elif lane.stage == "DiT":
            if not 0 <= request.steps < 50:
                raise RuntimeError("A completed DiT request was enqueued again")
            latency = self.case.profiles.latency(lane.raw, "DiT", request.input_tokens, self.active_bin)
            run = Execution(
                request_id=request_id,
                attempt_id=attempt["attempt_id"],
                stage=lane.stage,
                start_s=self.clock,
                duration_s=(50 - request.steps) * latency / 50,
                initial_progress=request.steps,
                target_progress=50,
                step_s=latency / 50,
                full_dit_s=latency,
            )
        else:
            duration = self.case.profiles.latency(lane.raw, lane.stage, request.input_tokens, self.active_bin)
            run = Execution(request_id, attempt["attempt_id"], lane.stage, self.clock, duration, 0, 1)
        if run.duration_s <= 0:
            raise RuntimeError("Nonpositive execution duration")
        lane.current = run
        lane.history.append(run)
        lane.version += 1
        attempt.update(start_s=self.clock, planned_service_s=run.duration_s, initial_progress=run.initial_progress)
        self.schedule_tick(run.finish_tick, "completion", {"lane": lane.uid, "version": lane.version, "attempt": run.attempt_id})
        self.emit("queue_exit", time_s=self.clock, request_id=request_id, lane_uid=lane.uid, reason="execution_start", attempt_id=run.attempt_id)
        self.emit("execution_start", time_s=self.clock, **attempt)

    def close_execution(self, lane: Lane, completed: bool) -> int:
        run = lane.current
        if run is None:
            raise RuntimeError("Closing an idle execution")
        request = self.requests[run.request_id]
        progress = self.observe(request.id, self.clock)
        request.generated, request.steps = progress.generated, progress.steps
        run.end_s, run.completed = self.clock, completed
        final_progress = run.progress_at(self.clock)
        attempt = self.attempts[run.attempt_id]
        attempt.update(
            exit_s=self.clock,
            exit_reason="completed" if completed else "migrated_running",
            executed_service_s=seconds(self.clock_tick - run.start_tick),
            final_progress=final_progress,
            executed_work=final_progress - run.initial_progress,
            generated_on_exit=request.generated,
            steps_on_exit=request.steps,
        )
        self.emit("execution_end", time_s=self.clock, **attempt)
        lane.current = None
        lane.version += 1
        request.owner = None
        del self.attempt_for[request.id]
        return request.id

    def complete(self, payload: dict) -> None:
        lane = self.lanes[payload["lane"]]
        if lane.version != payload["version"] or lane.current is None or lane.current.attempt_id != payload["attempt"]:
            self.emit("stale_event_ignored", time_s=self.clock, kind="completion", **payload)
            return
        request_id = self.close_execution(lane, True)
        request = self.requests[request_id]
        if lane.stage == "PE":
            if request.generated != request.output_tokens:
                raise RuntimeError("PE completion does not account for every token")
            self.monitor.complete_pe(self.clock, lane, request.id, request.output_tokens)
        elif lane.stage == "DiT" and request.steps != 50:
            raise RuntimeError("DiT completion does not account for every step")
        request.stage_index += 1
        if request.stage is None:
            request.finished_s = self.clock
            self.completed += 1
            self.emit("request_exit", time_s=self.clock, request_id=request.id, latency_s=elapsed(self.clock, request.arrival_s))
        else:
            self.pending.append((request_id, False, "stage_completion"))
        self.record_safe_lane(lane, request_id, migrated=False)
        self.check_groups()

    def drain_boundary(self, payload: dict) -> None:
        lane = self.lanes[payload["lane"]]
        if lane.version != payload["version"] or lane.current is None:
            self.emit("stale_event_ignored", time_s=self.clock, kind="boundary", **payload)
            return
        if lane.current.finish_tick <= self.clock_tick:
            self.complete({**payload, "attempt": lane.current.attempt_id})
            return
        request_id = self.close_execution(lane, False)
        self.pending.append((request_id, True, "running_migration"))
        self.record_safe_lane(lane, request_id, migrated=True)
        self.check_groups()

    def record_safe_lane(self, lane: Lane, request_id: int, *, migrated: bool) -> None:
        if self.transition is None or lane.state != "draining":
            return
        group = next(g for g in self.transition.groups if lane.physical_uid in g.sources)
        record = self.group_record(group)
        record["actual_safe_s"][lane.uid] = self.clock
        if migrated:
            record["migrated_running"][lane.stage].append(request_id)
        self.emit(
            "safe_boundary", time_s=self.clock, group_id=group.id,
            lane_uid=lane.uid, request_id=request_id,
            disposition="migrated_running" if migrated else "module_completed",
        )

    def monitor_tick(self) -> None:
        if self.completed == len(self.requests):
            return
        self.monitor.maintain(self.clock)
        observation = self.monitor.length(self.clock)
        middle = (self.settings.short_output_tokens + self.settings.long_output_tokens) / 2
        delta = self.spec.margin * (self.settings.long_output_tokens - self.settings.short_output_tokens)
        desired = self.active_bin
        if observation["mean"] is not None:
            if self.active_bin == self.settings.short_output_tokens and observation["mean"] > middle + delta:
                desired = self.settings.long_output_tokens
            elif self.active_bin == self.settings.long_output_tokens and observation["mean"] < middle - delta:
                desired = self.settings.short_output_tokens
        estimates = []
        for physical in self.live.values():
            for uid in physical.lanes:
                lane = self.lanes[uid]
                estimates.append({"lane_uid": uid, "state": lane.state, "capacity": self.monitor.sample(lane, self.clock, self.active_bin), "length": self.monitor.estimated_length(lane, self.clock, self.active_bin) if lane.stage == "PE" else None})
        self.emit("monitor", time_s=self.clock, active_bin=self.active_bin, desired_bin=desired, transition_in_progress=self.transition is not None, lower=middle - delta, upper=middle + delta, length=observation, instances=estimates)
        if self.spec.can_flip and self.transition is None and desired != self.active_bin:
            self.begin_flip(desired, observation)
        self.monitor_index = self.clock_tick // self.period_ticks + 1
        self.schedule_tick(self.monitor_index * self.period_ticks, "monitor", {})

    def begin_flip(self, target_bin: int, observation: dict) -> None:
        if not self.case.recipes:
            self.emit("flip_not_required", time_s=self.clock, reason="restricted_endpoint_has_no_role_changes")
            self.active_bin = target_bin
            return
        groups = []
        if target_bin == self.settings.long_output_tokens:
            selected, score = choose_forward(
                self.case, list(self.live.values()), self.lanes, self.estimator,
                self.clock, self.active_bin, self.observe, self.emit,
                policy=self.spec.source_selection, rng=self.source_selection_rng,
            )
            for index, (original, targets, uid) in enumerate(selected):
                groups.append(ConversionGroup(f"flip-{len(self.flips)}-group-{index}", copy.deepcopy(original), targets, [uid]))
        else:
            sources = [self.physicals[uid] for _, uids in self.lineage for uid in uids]
            score = score_sources(sources, list(self.live.values()), self.lanes, self.estimator, self.clock, self.active_bin, self.observe)
            if not score["feasible"]:
                raise RuntimeError("Reverse restricted flip would remove the last ready stage")
            for index, (original, uids) in enumerate(self.lineage):
                groups.append(ConversionGroup(f"flip-{len(self.flips)}-group-{index}", copy.deepcopy(original), [copy.deepcopy(original)], list(uids)))
        record = {"flip_id": len(self.flips), "direction": "short_to_long" if target_bin == self.settings.long_output_tokens else "long_to_short", "detected_s": self.clock, "target_bin": target_bin, "length_observation": observation, "selection": score, "groups": [], "completed_s": None}
        self.flips.append(record)
        self.transition = Transition(record["flip_id"], target_bin, groups, record)
        # Mark the entire set before any migration; no selected source may receive work.
        for group in groups:
            group_record = {
                "group_id": group.id,
                "node": group.original["node"],
                "gpu_ids": group.original["gpu_ids"],
                "source_uids": group.sources,
                "target_templates": [t["template"] for t in group.targets],
                "safe_boundaries_s": {},
                "actual_safe_s": {},
                "migrated_waiting": {stage: [] for stage in STAGES},
                "migrated_running": {stage: [] for stage in STAGES},
                "launch_started_s": None,
                "targets": [],
                "ready_s": None,
            }
            record["groups"].append(group_record)
            for uid in group.sources:
                physical = self.physicals[uid]
                if physical.raw["id"] in self.base_pe or physical.state != "ready":
                    raise RuntimeError("Attempt to convert original PE or a non-ready source")
                physical.state = "draining"
                for lane_id in physical.lanes:
                    lane = self.lanes[lane_id]
                    lane.state = "draining"
                    group_record["safe_boundaries_s"][lane_id] = lane.current.boundary(self.clock) if lane.current else self.clock
                    if lane.current is None:
                        group_record["actual_safe_s"][lane_id] = self.clock
        self.emit("flip_begin", time_s=self.clock, **record)
        for group in groups:
            for uid in group.sources:
                for lane_id in self.physicals[uid].lanes:
                    lane = self.lanes[lane_id]
                    while lane.waiting():
                        request_id = (lane.recovery if lane.recovery else lane.queue).popleft()
                        request = self.requests[request_id]
                        attempt = self.attempts[self.attempt_for.pop(request_id)]
                        attempt.update(exit_s=self.clock, exit_reason="migrated_waiting", executed_service_s=0.0, generated_on_exit=request.generated, steps_on_exit=request.steps)
                        self.emit("queue_exit", time_s=self.clock, **attempt)
                        self.group_record(group)["migrated_waiting"][lane.stage].append(request_id)
                        request.owner = None
                        self.pending.append((request_id, False, "waiting_migration"))
                    if lane.current and lane.stage in ("PE", "DiT"):
                        self.schedule_tick(lane.current.boundary_tick(self.clock_tick), "boundary", {"lane": lane.uid, "version": lane.version})
        self.check_groups()

    def group_record(self, group: ConversionGroup) -> dict:
        assert self.transition is not None
        return next(r for r in self.transition.record["groups"] if r["group_id"] == group.id)

    def check_groups(self) -> None:
        if self.transition is None:
            return
        for group in self.transition.groups:
            source_lanes = [
                self.lanes[lane_id]
                for uid in group.sources
                for lane_id in self.physicals[uid].lanes
            ]
            safe = all(lane.current is None and not lane.waiting() for lane in source_lanes)
            if not group.launch_started and safe:
                group.launch_started = True
                self.schedule_tick(self.clock_tick, "group_launch", {"group": group.id, "flip": self.transition.id})

    def launch_group(self, payload: dict) -> None:
        if self.transition is None or self.transition.id != payload["flip"]:
            return
        group = next(g for g in self.transition.groups if g.id == payload["group"])
        record = self.group_record(group)
        record["launch_started_s"] = self.clock
        if set(record["actual_safe_s"]) != set(record["safe_boundaries_s"]):
            raise RuntimeError("Launching a group without every module's actual safe time")
        for uid in group.sources:
            physical = self.physicals[uid]
            for lane_id in physical.lanes:
                lane = self.lanes[lane_id]
                if lane.current or lane.waiting():
                    raise RuntimeError("A GPU group launched before all its modules were safe")
                lane.state = "retired"
                lane.version += 1
            physical.state, physical.retired_s = "retired", self.clock
            del self.live[physical.raw["id"]]
            self.emit("instance_retired", time_s=self.clock, instance_uid=uid)
        for raw in group.targets:
            physical = self.create_physical(raw, self.clock, ready=False)
            group.launched.append(physical.uid)
            optimized = self.spec.startup_optimized(raw["stages"])
            delay = self.case.profiles.startup(raw, optimized)
            delay_ticks = ticks(delay, positive=delay != 0)
            delay = seconds(delay_ticks)
            target_record = {"instance_uid": physical.uid, "placement_id": raw["id"], "startup_s": delay, "launch_started_s": self.clock, "ready_s": None, "startup_profile": raw["startup"], "startup_mode": self.spec.startup_mode, "optimized": optimized}
            record["targets"].append(target_record)
            self.emit("launch_start", time_s=self.clock, group_id=group.id, **target_record)
            self.schedule_tick(self.clock_tick + delay_ticks, "ready", {"physical": physical.uid, "group": group.id, "flip": self.transition.id})

    def ready(self, payload: dict) -> None:
        if self.transition is None or self.transition.id != payload["flip"]:
            return
        group = next(g for g in self.transition.groups if g.id == payload["group"])
        physical = self.physicals[payload["physical"]]
        if physical.state != "launching":
            raise RuntimeError("Duplicate launch completion")
        physical.state, physical.ready_s = "ready", self.clock
        for lane_id in physical.lanes:
            self.lanes[lane_id].state = "ready"
        record = self.group_record(group)
        next(r for r in record["targets"] if r["instance_uid"] == physical.uid)["ready_s"] = self.clock
        self.emit("instance_ready", time_s=self.clock, instance_uid=physical.uid, group_id=group.id)
        if all(self.physicals[uid].state == "ready" for uid in group.launched):
            group.finished = True
            record["ready_s"] = self.clock
        if all(g.finished for g in self.transition.groups):
            transition = self.transition
            self.active_bin = transition.target_bin
            self.lineage = [(copy.deepcopy(g.original), list(g.launched)) for g in transition.groups] if self.active_bin == self.settings.long_output_tokens else []
            transition.record["completed_s"] = self.clock
            self.emit("flip_complete", time_s=self.clock, flip_id=transition.id, active_bin=self.active_bin)
            self.transition = None
            self.check_endpoint()

    def check_endpoint(self) -> None:
        expected = self.case.source if self.active_bin == self.settings.short_output_tokens else self.case.target
        actual = [p.raw for p in self.live.values() if p.raw["hardware"] != "CPU"]
        if template_counts(actual) != template_counts(expected["instances"]):
            raise RuntimeError("Restricted runtime endpoint has incorrect hardware/template counts")
        occupied = {(i["node"], g) for i in actual for g in i["gpu_ids"]}
        target_occupied = {(i["node"], g) for i in self.case.source["instances"] for g in i["gpu_ids"]}
        if occupied != target_occupied:
            raise RuntimeError("Restricted runtime endpoint changed the original GPU allocation")
        for stage in ("PE", "DiT", "VAE"):
            capacity = math.fsum(1 / self.case.profiles.latency(raw, stage, self.settings.input_tokens, self.active_bin) for raw in actual if stage in raw["stages"])
            if not math.isclose(capacity, expected["stage_capacities_req_s"][stage], rel_tol=1e-8, abs_tol=1e-9):
                raise RuntimeError(f"Runtime endpoint capacity differs for {stage}")

    def check_state(self) -> None:
        owners, occupied = {}, set()
        for physical in self.live.values():
            for gpu in physical.raw["gpu_ids"]:
                location = physical.raw["node"], gpu
                if location in occupied:
                    raise RuntimeError("Two live physical instances own the same GPU")
                occupied.add(location)
            for uid in physical.lanes:
                lane = self.lanes[uid]
                ids = lane.waiting() + ((lane.current.request_id,) if lane.current else ())
                for request_id in ids:
                    if request_id in owners or self.requests[request_id].owner != uid:
                        raise RuntimeError("Request has duplicate or inconsistent ownership")
                    if self.requests[request_id].stage != lane.stage:
                        raise RuntimeError("Request resides in the wrong module")
                    owners[request_id] = uid
        if not self.base_pe <= set(self.live):
            raise RuntimeError("Restricted flip modified original PE")
        for request_id in self.arrived:
            request = self.requests[request_id]
            if (request_id in owners) == (request.finished_s is not None):
                raise RuntimeError("Arrived request is lost or both owned and complete")
            if not 0 <= request.generated <= request.output_tokens or not 0 <= request.steps <= 50:
                raise RuntimeError("Request progress is outside valid bounds")
        if self.completed < len(self.arrived) and not any(lane.current for lane in self.lanes.values()) and self.transition is None:
            raise RuntimeError("Unfinished requests cannot make execution progress")

    def run(self) -> dict:
        self.check_endpoint()
        while self.events:
            time = self.events[0][0]
            self.advance_tick(time)
            # Finish all same-time control events before routing. Newly scheduled
            # same-time boundaries/launches re-enter this heap with their priority.
            while self.events and self.events[0][0] == time:
                _, _, _, kind, payload = heapq.heappop(self.events)
                if kind == "arrival":
                    request_id = payload["request"]
                    self.arrived.add(request_id)
                    self.pending.append((request_id, False, "arrival"))
                    self.emit("request_enter", time_s=self.clock, request_id=request_id)
                elif kind == "completion":
                    self.complete(payload)
                elif kind == "boundary":
                    self.drain_boundary(payload)
                elif kind == "monitor":
                    self.monitor_tick()
                elif kind == "group_launch":
                    self.launch_group(payload)
                elif kind == "ready":
                    self.ready(payload)
            while self.pending:
                self.route(*self.pending.popleft())
            for lane in self.lanes.values():
                self.start(lane)
            self.check_state()
            if self.completed == len(self.requests) and self.transition is None:
                self.events.clear()
                break
        if self.completed != len(self.requests):
            raise RuntimeError("Event queue ended with incomplete requests")
        if self.transition is not None:
            raise RuntimeError("Event queue ended before an in-flight conversion completed")
        return {
            "schema_version": 1, "run_id": self.run_id, "spec": asdict(self.spec),
            "requests": [asdict(r) for r in self.requests.values()], "attempts": self.attempts,
            "flips": self.flips, "instances": [asdict(p) for p in self.physicals.values()],
            "slo_baselines": self.case.slo_baselines(), "final_control_time_s": self.clock,
            "final_active_bin": self.active_bin,
            "simulation_semantics": SEMANTICS.copy(),
        }

    def diagnostic(self) -> dict:
        return {
            "time_s": self.clock,
            "active_bin": self.active_bin,
            "completed": self.completed,
            "total_requests": len(self.requests),
            "pending_routes": list(self.pending),
            "transition": self.transition.record if self.transition else None,
            "lanes": [
                {
                    "lane": lane.uid,
                    "state": lane.state,
                    "waiting": list(lane.waiting()),
                    "running": lane.current.request_id if lane.current else None,
                }
                for lane in self.lanes.values()
                if lane.state != "retired"
            ],
        }
