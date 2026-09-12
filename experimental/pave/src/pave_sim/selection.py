"""Exact source-set scoring; startup delay is diagnostic, never an objective."""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable

from .inputs import Case
from .monitor import WorkEstimator
from .state import STAGES, Lane, Physical, Progress


def score_sources(
    sources: list[Physical], live: list[Physical], lanes: dict[str, Lane],
    estimator: WorkEstimator, now: float, active_bin: int,
    observe: Callable[[int, float], Progress],
) -> dict:
    excluded = {uid for physical in sources for uid in physical.lanes}
    units = dict.fromkeys(STAGES, 0.0)
    boundaries = {}
    for physical in sources:
        for uid in physical.lanes:
            lane = lanes[uid]
            units[lane.stage] += sum(estimator.units(lane, observe(r, now), now, active_bin) for r in lane.waiting())
            if lane.current:
                boundary = lane.current.boundary(now)
                boundaries[uid] = boundary
                if lane.stage in ("PE", "DiT"):
                    # Project only to the next token/step. Do not give the policy the
                    # request's final PE length, even if execution knows its EOS.
                    units[lane.stage] += estimator.units(lane, observe(lane.current.request_id, boundary), now, active_bin)
            else:
                boundaries[uid] = now
    capacities = dict.fromkeys(STAGES, 0.0)
    observations = []
    for physical in live:
        for uid in physical.lanes:
            lane = lanes[uid]
            if uid not in excluded and lane.ready:
                rate = estimator.monitor.rate(lane, now, active_bin)
                capacities[lane.stage] += rate["rate"]
                observations.append({"lane": uid, **rate})
    missing = [stage for stage in STAGES if capacities[stage] <= 0]
    if missing:
        return {"feasible": False, "reason": "no_remaining_ready_stage", "stages": missing, "score_s": None}
    added = {stage: units[stage] / capacities[stage] for stage in STAGES}
    return {
        "feasible": True, "score_s": max(added.values()), "work_units": units,
        "remaining_capacity": capacities, "added_work_s": added,
        "safe_boundary_s": boundaries, "remaining_capacity_observations": observations,
    }


def choose_forward(
    case: Case, live: list[Physical], lanes: dict[str, Lane], estimator: WorkEstimator,
    now: float, active_bin: int, observe: Callable[[int, float], Progress], emit: Callable,
    *, policy: str = "min_disruption", rng: random.Random | None = None,
) -> tuple[list[tuple[dict, list[dict], str]], dict]:
    if policy not in ("min_disruption", "random_feasible"):
        raise ValueError("Unknown source selection policy")
    if policy == "random_feasible" and rng is None:
        raise ValueError("Random source selection requires its own RNG")
    pools = []
    for recipe in case.recipes:
        candidates = sorted(
            (physical for physical in live if physical.state == "ready"
             and physical.raw["hardware"] == recipe.hardware
             and physical.raw["template"] == recipe.source_template
             and physical.raw["bundle_size"] == recipe.width
             and all(lanes[uid].ready for uid in physical.lanes)),
            key=lambda physical: physical.raw["id"],
        )
        if len(candidates) < recipe.quantity:
            raise RuntimeError(f"Not enough ready sources for restricted recipe {recipe.id}")
        pools.append((recipe, candidates))

    def assignments(index: int, chosen: list, used: set[str]):
        if index == len(pools):
            yield chosen
            return
        recipe, candidates = pools[index]
        available = [p for p in candidates if p.uid not in used]
        for group in itertools.combinations(available, recipe.quantity):
            yield from assignments(index + 1, chosen + [(p, recipe) for p in group], used | {p.uid for p in group})

    best, best_score, best_key = None, None, None
    feasible = []
    candidate_count = 0
    for selected in assignments(0, [], set()):
        candidate_count += 1
        sources = [p for p, _ in selected]
        score = score_sources(sources, live, lanes, estimator, now, active_bin, observe)
        key_ids = tuple(sorted((p.raw["id"], recipe.id) for p, recipe in selected))
        emit("flip_candidate", time_s=now, source_ids=[p.uid for p in sources], assignments=key_ids, **score)
        if score["feasible"]:
            feasible.append((selected, score))
            key = score["score_s"], key_ids
            if best_key is None or key < best_key:
                best, best_score, best_key = selected, score, key
    if best is None:
        raise RuntimeError("No restricted source set leaves all stages ready during conversion")
    minimum = best_score["score_s"]
    if policy == "random_feasible":
        # One uniform draw over complete feasible assignments, independent of
        # scores and of every other random stream in the simulator.
        best, best_score = feasible[rng.randrange(len(feasible))]
    emit("source_selection", time_s=now, policy=policy,
         candidate_count=candidate_count, feasible_count=len(feasible),
         source_ids=[p.uid for p, _ in best], selected_score_s=best_score["score_s"],
         minimum_score_s=minimum,
         selected_lanes=[{
             "lane_uid": uid, "stage": lanes[uid].stage,
             "running_request": lanes[uid].current.request_id if lanes[uid].current else None,
             "waiting_requests": list(lanes[uid].waiting()),
         } for p, _ in best for uid in p.lanes])
    return [(p.raw, recipe.replacements(p.raw), p.uid) for p, recipe in best], best_score
