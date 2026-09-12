"""Recompute request metrics, physical visits and allocation from recorded attempts."""

from __future__ import annotations

import math
from collections import defaultdict

from .config import RunSpec
from .records import write_csv, write_json, write_jsonl
from .timing import elapsed


def time_difference(result, end, start):
    # Rebuilding legacy records must retain their original clock semantics.
    return elapsed(end, start) if result.get("simulation_semantics", {}).get("version", 1) >= 2 else end - start

METRICS = ("throughput_req_s", "p50_s", "p99_s", "slo5", "slo10")
LABELS = {"throughput_req_s": "吞吐 req/s", "p50_s": "p50 秒", "p99_s": "p99 秒", "slo5": "SLO5", "slo10": "SLO10"}


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile_value / 100
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def request_rows(result: dict) -> list[dict]:
    rows = []
    for raw in result["requests"]:
        if raw["finished_s"] is None:
            raise ValueError("Cannot report a complete run with unfinished requests")
        latency = time_difference(result, raw["finished_s"], raw["arrival_s"])
        if latency < 0:
            raise ValueError("Negative request latency")
        baseline = result["slo_baselines"][raw["kind"]]["latency_s"]
        rows.append({**raw, "run_id": result["run_id"], "latency_s": latency, "slo_baseline_s": baseline, "slo5_met": latency <= 5 * baseline, "slo10_met": latency <= 10 * baseline})
    return rows


def summary(result: dict) -> dict:
    rows = request_rows(result)
    if not rows:
        raise ValueError("No requests in completed simulation")
    first, last = min(r["arrival_s"] for r in rows), max(r["finished_s"] for r in rows)
    if last <= first:
        raise ValueError("Request makespan must be positive")
    spec = RunSpec(**result["spec"])
    makespan = time_difference(result, last, first)
    values = {
        "schema_version": 1, "run_id": result["run_id"], "case_id": spec.case_id,
        **result["spec"], "scheduler": spec.scheduler, "custom_scheduler": spec.scheduler_override is not None,
        "requests": len(rows), "first_arrival_s": first, "last_request_finish_s": last,
        "makespan_s": makespan, "throughput_req_s": len(rows) / makespan,
        "p50_s": percentile([r["latency_s"] for r in rows], 50),
        "p99_s": percentile([r["latency_s"] for r in rows], 99),
        "slo5": sum(r["slo5_met"] for r in rows) / len(rows),
        "slo10": sum(r["slo10_met"] for r in rows) / len(rows),
        "flip_triggered": len(result["flips"]),
        "flip_completed": sum(f["completed_s"] is not None for f in result["flips"]),
        "converted_groups": sum(len(f["groups"]) for f in result["flips"]),
        "final_control_time_s": result["final_control_time_s"],
        "final_active_bin": result["final_active_bin"], "slo_baselines": result["slo_baselines"],
    }
    for kind in ("Short", "Long"):
        subset = [r for r in rows if r["kind"] == kind]
        values[kind.lower()] = {
            "requests": len(subset),
            "p50_s": percentile([r["latency_s"] for r in subset], 50),
            "p99_s": percentile([r["latency_s"] for r in subset], 99),
            "slo5": sum(r["slo5_met"] for r in subset) / len(subset) if subset else None,
            "slo10": sum(r["slo10_met"] for r in subset) / len(subset) if subset else None,
        }
    return values


def allocations(result: dict) -> list[dict]:
    groups = {}
    # Include idle placements with zero work; otherwise a routing report hides
    # precisely the instances whose under-utilization it needs to explain.
    for physical in result["instances"]:
        raw = physical["raw"]
        for stage in raw["stages"]:
            key = stage, raw["id"]
            if key not in groups:
                groups[key] = {
                    "run_id": result["run_id"], "stage": stage,
                    "placement_id": raw["id"], "node": raw["node"],
                    "hardware": raw["hardware"], "template": raw["template"],
                    "gpu_ids": raw["gpu_ids"], "received": 0, "completed": 0,
                    "executed_work": 0.0, "service_s": 0.0,
                    "work_unit": "token" if stage == "PE" else "step" if stage == "DiT" else "request",
                    "lifecycles": set(),
                }
            groups[key]["lifecycles"].add(physical["uid"])
    for attempt in result["attempts"]:
        key = attempt["stage"], attempt["placement_id"]
        if key not in groups:
            raise ValueError(f"Execution attempt refers to an unrecorded placement: {key}")
        row = groups[key]
        row["received"] += 1
        row["completed"] += attempt["exit_reason"] == "completed"
        row["executed_work"] += attempt["executed_work"]
        row["service_s"] += attempt.get("executed_service_s", 0.0)
        row["lifecycles"].add(attempt["instance_uid"])
    totals = defaultdict(lambda: {"received": 0, "completed": 0, "executed_work": 0.0})
    for row in groups.values():
        for field in totals[row["stage"]]:
            totals[row["stage"]][field] += row[field]
    for row in groups.values():
        row["lifecycles"] = sorted(row["lifecycles"])
        for field, denominator in totals[row["stage"]].items():
            row[f"{field}_share"] = row[field] / denominator if denominator else 0.0
    return [groups[key] for key in sorted(groups)]


def physical_visits(result: dict) -> list[dict]:
    """Merge adjacent same-process stages, so DiT/VAE do not invent extra physical visits."""
    by_request = defaultdict(list)
    for attempt in result["attempts"]:
        by_request[attempt["request_id"]].append(attempt)
    visits = []
    for request_id, attempts in sorted(by_request.items()):
        current = None
        for attempt in sorted(attempts, key=lambda a: (a["enter_s"], a["attempt_id"])):
            if current is not None and current["instance_uid"] == attempt["instance_uid"] and current["exit_s"] == attempt["enter_s"] and current["exit_reason"] == "completed":
                current["exit_s"] = attempt["exit_s"]
                current["exit_reason"] = attempt["exit_reason"]
                current["attempt_ids"].append(attempt["attempt_id"])
                current["stages"].append(attempt["stage"])
            else:
                current = {"run_id": result["run_id"], "request_id": request_id, "instance_uid": attempt["instance_uid"], "placement_id": attempt["placement_id"], "enter_s": attempt["enter_s"], "exit_s": attempt["exit_s"], "exit_reason": attempt["exit_reason"], "attempt_ids": [attempt["attempt_id"]], "stages": [attempt["stage"]]}
                visits.append(current)
    return visits


def flip_rows(result: dict) -> list[dict]:
    rows = []
    for flip in result["flips"]:
        for group in flip["groups"]:
            rows.append({
                "run_id": result["run_id"], "flip_id": flip["flip_id"], "direction": flip["direction"],
                "detected_s": flip["detected_s"], "selection_added_work_s": flip["selection"]["score_s"],
                **group,
                "migrated_waiting_count": sum(len(ids) for ids in group["migrated_waiting"].values()),
                "migrated_running_count": sum(len(ids) for ids in group["migrated_running"].values()),
                "safe_wait_s": time_difference(result, group["launch_started_s"], flip["detected_s"]) if group["launch_started_s"] is not None else None,
                "group_unavailable_s": time_difference(result, group["ready_s"], flip["detected_s"]) if group["ready_s"] is not None else None,
            })
    return rows


def write_run(directory, result: dict, metadata: dict) -> dict:
    values = summary(result)
    requests = request_rows(result)
    for name, rows in (("requests", requests), ("attempts", result["attempts"]), ("instances", result["instances"]), ("visits", physical_visits(result))):
        write_jsonl(directory / f"{name}.jsonl", rows)
    write_json(directory / "flips.json", result["flips"])
    write_json(directory / "summary.json", values)
    write_csv(directory / "requests.csv", requests)
    write_csv(directory / "attempts.csv", result["attempts"])
    write_csv(directory / "allocations.csv", allocations(result))
    write_csv(directory / "flip_groups.csv", flip_rows(result))
    write_json(directory / "run.json", {
        "schema_version": 1, "status": "complete", "run_id": result["run_id"],
        "spec": result["spec"], "slo_baselines": result["slo_baselines"],
        "final_control_time_s": result["final_control_time_s"], "final_active_bin": result["final_active_bin"],
        **({"simulation_semantics": result["simulation_semantics"]} if "simulation_semantics" in result else {}),
        **metadata,
    })
    return values
