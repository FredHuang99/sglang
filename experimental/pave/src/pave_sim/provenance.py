"""Comparison identity is derived from run facts, never directory names or a matrix."""
from __future__ import annotations

import copy
import math

from pave_ilp.profiles import digest

from .config import RunSpec
from .timing import SEMANTICS, seconds, ticks


def required(data: dict, path: str, run_id: str):
    value = data
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value or value[key] is None:
            raise ValueError(f"Incomplete input provenance for run {run_id}: missing {path}")
        value = value[key]
    return value


def fingerprint(data: dict, path: str, run_id: str) -> str:
    value = required(data, path, run_id)
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"Invalid input provenance for run {run_id}: {path} must be SHA-256")
    return value.lower()


def comparison_context(result: dict) -> dict:
    run_id = result["run_id"]
    raw_spec = required(result, "spec", run_id)
    for key in ("generator", "scenario", "strategy", "rate_per_min", "window_s", "margin", "seed"):
        required(raw_spec, key, run_id)
    spec = RunSpec(**raw_spec)
    spec.validate()
    if spec.initial_deployment != "source" or spec.startup_mode_override is not None or "evaluation_context" in result:
        raise ValueError("Evaluation runs require the evaluation report interface")
    if type(spec.seed) is not int:
        raise ValueError(f"Invalid seed in run {run_id}")
    semantics = result.get("simulation_semantics")
    if semantics is None:
        # Schema-1 records predate the explicit semantics field. Their source
        # digest still isolates them; do not infer that they use the new model.
        semantics = {"version": 1, "clock": "legacy_float", "capacity_cache_sampling": "legacy_query"}
    elif semantics != SEMANTICS:
        raise ValueError(f"Unsupported simulation_semantics in run {run_id}: {semantics}")
    cfg = required(result, "configuration", run_id)
    shared = {
        "generator": spec.generator, "scenario": spec.scenario, "seed": spec.seed,
        "source_deployment_sha256": fingerprint(result, "input_case.source.sha256", run_id),
        "restricted_flip_sha256": fingerprint(result, "input_case.flip.sha256", run_id),
        "profile_values_sha256": fingerprint(result, "profile_values_sha256", run_id),
        "trace_sha256": fingerprint(result, "trace_sha256", run_id),
        "source_code_sha256": fingerprint(result, "environment.source_sha256", run_id),
        "simulation_semantics": copy.deepcopy(semantics),
        "arrival_mode": "equally_spaced",
        "slo_baselines": copy.deepcopy(required(result, "slo_baselines", run_id)),
    }
    for key in ("input_tokens", "short_output_tokens", "long_output_tokens"):
        value = required(result, f"configuration.{key}", run_id)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Invalid configuration.{key} in run {run_id}")
        shared[key] = value
    if shared["short_output_tokens"] >= shared["long_output_tokens"]:
        raise ValueError(f"Invalid output bins in run {run_id}")
    tie = required(result, "configuration.initial_tie", run_id)
    if tie not in ("Short", "Long"):
        raise ValueError(f"Invalid configuration.initial_tie in run {run_id}")
    shared["initial_tie"] = tie
    def canonical_time(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {name} in run {run_id}")
        return seconds(ticks(value, positive=True)) if semantics["version"] >= 2 else float(value)
    shared["duration_s"] = canonical_time(required(result, "configuration.duration_s", run_id), "duration_s")
    if "monitor_period_s" not in cfg:
        raise ValueError(f"Incomplete input provenance for run {run_id}: missing configuration.monitor_period_s")
    period = cfg["monitor_period_s"] if cfg["monitor_period_s"] is not None else spec.window_s
    conditions = {"rate_per_min": float(spec.rate_per_min),
                  "window_s": canonical_time(spec.window_s, "window_s"),
                  "effective_monitor_period_s": canonical_time(period, "monitor_period_s")}
    return {"schema_version": 1, "shared": shared, "shared_sha256": digest(shared),
            "conditions": conditions, "context_sha256": digest({"shared": shared, "conditions": conditions})}


def validate_contexts(results: list[dict]) -> dict[str, dict]:
    contexts, cases, windows, setups = {}, {}, {}, {}
    for result in results:
        run_id = result["run_id"]
        context = comparison_context(result)
        contexts[run_id] = context
        shared, condition = context["shared"], context["conditions"]
        case = (shared["generator"], shared["scenario"])
        if case in cases:
            old_id, old = cases[case]
            conflict = [key for key in shared if shared[key] != old[key]]
            if conflict:
                raise ValueError(f"Conflicting input provenance for runs {old_id} and {run_id}: {', '.join(conflict)}")
        else:
            cases[case] = run_id, shared
        spec = result["spec"]
        if spec.get("scheduler_override") is not None or spec.get("startup_mode_override") is not None:
            continue
        # Different rates/windows are intended matrix axes, but monitor cadence
        # for the same rate/window is fixed across B--E, not an ablation factor.
        if spec["strategy"] != "A":
            key = (*case, condition["rate_per_min"], condition["window_s"])
            period = condition["effective_monitor_period_s"]
            if key in windows and windows[key][1] != period:
                raise ValueError(f"Conflicting effective_monitor_period_s for runs {windows[key][0]} and {run_id}")
            windows[key] = run_id, period
        key = (*case, spec["strategy"], condition["rate_per_min"],
               None if spec["strategy"] == "A" else condition["window_s"],
               spec["margin"] if spec["strategy"] in ("D", "E") else None)
        if key in setups:
            raise ValueError(f"Ambiguous comparison baseline/setup: runs {setups[key]} and {run_id}")
        setups[key] = run_id
    return contexts
