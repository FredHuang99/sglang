"""Three routing policies. All input requests are progress-only observations."""

from __future__ import annotations

import random
from collections.abc import Callable

from .monitor import WorkEstimator
from .state import Lane, Progress


class Scheduler:
    def __init__(self, name: str, estimator: WorkEstimator, seed: int):
        self.name, self.estimator, self.random = name, estimator, random.Random(seed)

    def choose(
        self, candidates: list[Lane], incoming: Progress, now: float, active_bin: int,
        observe: Callable[[int, float], Progress],
    ) -> tuple[Lane, list[dict]]:
        ready = sorted((lane for lane in candidates if lane.ready), key=lambda lane: lane.order)
        if not ready:
            raise RuntimeError("No ready, non-draining instance can accept this stage")
        records = []
        for lane in ready:
            if self.name == "least_waiting":
                record = {"lane": lane.uid, "score": len(lane.waiting())}
            elif self.name == "capacity_weighted":
                latency = self.estimator.monitor.profiles.latency(
                    lane.raw, lane.stage, self.estimator.monitor.input_tokens, active_bin
                )
                record = {"lane": lane.uid, "capacity_req_s": 1 / latency}
            elif self.name == "estimated_completion":
                running = self.estimator.units(lane, observe(lane.current.request_id, now), now, active_bin) if lane.current else 0.0
                waiting = sum(self.estimator.units(lane, observe(r, now), now, active_bin) for r in lane.waiting())
                added = self.estimator.units(lane, incoming, now, active_bin)
                capacity = self.estimator.monitor.rate(lane, now, active_bin)
                record = {"lane": lane.uid, "score": (running + waiting + added) / capacity["rate"], "running_units": running, "waiting_units": waiting, "incoming_units": added, "capacity": capacity}
                if lane.stage == "PE":
                    record["length"] = self.estimator.monitor.estimated_length(lane, now, active_bin)
            else:
                raise ValueError(f"Unknown scheduler: {self.name}")
            records.append(record)
        if self.name == "capacity_weighted":
            total = sum(r["capacity_req_s"] for r in records)
            for record in records:
                record["probability"] = record["capacity_req_s"] / total
            chosen = self.random.choices(ready, weights=[r["probability"] for r in records], k=1)[0]
        else:
            index = min(range(len(records)), key=lambda i: (records[i]["score"], ready[i].order))
            chosen = ready[index]
        return chosen, records
