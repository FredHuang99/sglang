"""Offline placement matching and whole-generator-bundle to PE conversions."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from itertools import combinations_with_replacement

from .deployment import (
    SCHEMA_VERSION,
    capacities,
    evaluate,
    instance,
    layout_signature,
    legal_gpu_groups,
    refresh,
    signature,
    validate_deployment,
)
from .profiles import STAGES, ProfileCatalog
from .solver import CAPACITY_TOL, MILP
from .templates import Node, Template


def match_target(
    source: dict,
    target: dict,
    templates: dict[str, tuple[Template, ...]],
    catalog: ProfileCatalog,
    time_limit_s: float,
) -> dict:
    """Relabel target node layouts and place their bundles to maximize exact retention.

    Preserving each target node's (template, origin) multiset also preserves the
    fragment-fill policy and prevents spreading fillers to new, core-empty nodes.
    """
    model = MILP()
    nodes = tuple(Node(**n) for n in target["nodes"])
    old = {signature(i) for i in source["instances"]}
    old_gpus = {(i["node"], g) for i in source["instances"] for g in i["gpu_ids"]}
    patterns = defaultdict(Counter)
    for node in nodes:
        pattern = tuple(
            sorted(
                (i["template"], i["origin"]) for i in target["instances"] if i["node"] == node.name
            )
        )
        patterns[node.hardware][pattern] += 1
    ys, placements = {}, []
    for node in nodes:
        for pattern in sorted(patterns[node.hardware]):
            ys[node.name, pattern] = model.variable(1)
        model.constrain({j: 1 for (name, _), j in ys.items() if name == node.name}, 1, 1)
        needed = {name for pattern in patterns[node.hardware] for name, _ in pattern}
        for template in templates[node.hardware]:
            if template.name not in needed:
                continue
            indices = []
            for ids in legal_gpu_groups(node.gpu_count, template.width):
                idx = model.variable(1)
                placements.append((idx, node, ids, template))
                indices.append(idx)
            row = dict.fromkeys(indices, 1)
            row.update(
                {
                    j: -sum(name == template.name for name, _ in p)
                    for (node_name, p), j in ys.items()
                    if node_name == node.name
                }
            )
            model.constrain(row, 0, 0)
        for gpu in range(node.gpu_count):
            model.constrain(
                {j: 1 for j, n, ids, _ in placements if n.name == node.name and gpu in ids}, upper=1
            )
    for hw, counts in patterns.items():
        hw_nodes = {n.name for n in nodes if n.hardware == hw}
        for pattern, count in counts.items():
            model.constrain(
                {j: 1 for (n, p), j in ys.items() if n in hw_nodes and p == pattern}, count, count
            )
    retention = {j: -1 for j, n, ids, t in placements if (n.name, ids, t.name) in old}
    x = model.solve(retention, time_limit_s, "placement_max_retained_instances")
    optimum = round(sum(value * x[j] for j, value in retention.items()))
    model.constrain(retention, optimum, optimum)
    changed_gpus = {
        j: sum((n.name, g) not in old_gpus for g in ids)
        - (t.width if (n.name, ids, t.name) in old else 0)
        for j, n, ids, t in placements
    }
    x = model.solve(changed_gpus, time_limit_s, "placement_min_changed_gpus")
    result = copy.deepcopy(target)
    result["instances"] = []
    origins = {}
    for (node_name, pattern), j in ys.items():
        if x[j] > 0.5:
            for name in {name for name, _ in pattern}:
                origins[node_name, name] = sorted(
                    (origin for t, origin in pattern if t == name),
                    key=lambda origin: (origin != "ilp", origin),
                )
    for j, node, ids, template in placements:
        if x[j] > 0.5:
            origin = origins[node.name, template.name].pop(0)
            result["instances"].append(instance(node, ids, template, origin))
    result["instances"].sort(key=signature)
    result["core_instances"] = [
        copy.deepcopy(i) for i in result["instances"] if i["origin"] == "ilp"
    ]
    result["fragment_actions"] = [
        {
            "node": i["node"],
            "gpu_ids": i["gpu_ids"],
            "template": i["template"],
            "rule": i["origin"].removeprefix("fragment_"),
        }
        for i in result["instances"]
        if i["origin"] != "ilp"
    ]
    result["placement_solver"] = model.history
    refresh(result, catalog)
    validate_deployment(result, catalog)
    return result


def diff_actions(source: dict, target: dict) -> list[dict]:
    old = {signature(i): i for i in source["instances"]}
    new = {signature(i): i for i in target["instances"]}
    removed = [i for key, i in old.items() if key not in new]
    added = [i for key, i in new.items() if key not in old]
    actions = []
    for node in sorted({i["node"] for i in removed + added}):
        lhs = [i for i in removed if i["node"] == node]
        rhs = [i for i in added if i["node"] == node]
        actions.append(
            {
                "node": node,
                "gpu_ids": sorted({g for i in lhs + rhs for g in i["gpu_ids"]}),
                "remove_instance_ids": sorted(i["id"] for i in lhs),
                "add_instances": rhs,
            }
        )
    return actions


def restricted_target(
    source: dict,
    target: dict,
    templates: dict[str, tuple[Template, ...]],
    catalog: ProfileCatalog,
    time_limit_s: float,
) -> tuple[dict, list[dict]]:
    base = evaluate(source, catalog, target["output_tokens"])
    # Group identical source bundles; the MILP chooses conversion counts, not layouts.
    groups = defaultdict(list)
    for item in base["instances"]:
        if "PE" not in item["stages"]:
            groups[item["hardware"], item["template"]].append(item)
    model = MILP()
    lam = model.variable(float("inf"), integer=False)
    options = []
    for key, members in sorted(groups.items()):
        hw, _ = key
        width = members[0]["bundle_size"]
        pe = sorted(
            (t for t in templates[hw] if tuple(t.stages) == ("PE",) and t.width <= width),
            key=lambda t: t.width,
        )
        indices = []
        for count in range(1, width + 1):
            for replacements in combinations_with_replacement(pe, count):
                if sum(t.width for t in replacements) != width:
                    continue
                idx = model.variable(len(members))
                indices.append(idx)
                delta = {
                    s: sum(t.caps[s] for t in replacements)
                    - members[0]["stages"].get(s, {}).get("capacity_req_s", 0)
                    for s in STAGES
                }
                options.append(
                    (idx, key, tuple(sorted(replacements, key=lambda t: (-t.width, t.name))), delta)
                )
        model.constrain(dict.fromkeys(indices, 1), upper=len(members))
    base_caps = capacities(base["instances"])
    for stage in STAGES:
        model.constrain(
            {lam: 1, **{j: -60 * delta[stage] for j, _, _, delta in options}},
            upper=60 * base_caps[stage],
        )
    x = model.solve({lam: -1}, time_limit_s, "restricted_max_throughput")
    optimum = float(x[lam]) / 60
    model.constrain({lam: 1}, lower=max(0, optimum - CAPACITY_TOL) * 60)
    gpu_cost = {j: groups[key][0]["bundle_size"] for j, key, _, _ in options}
    x = model.solve(gpu_cost, time_limit_s, "restricted_min_changed_gpus")
    cost = round(sum(x[j] * c for j, c in gpu_cost.items()))
    model.constrain(gpu_cost, cost, cost)
    x = model.solve({j: 1 for j, _, _, _ in options}, time_limit_s, "restricted_min_source_bundles")
    result = copy.deepcopy(base)
    result.pop("evaluation_only", None)
    # This is a reachable transition result, not a fresh fragment-fill pass.
    result.pop("core_instances", None)
    result.pop("core_throughput_req_s", None)
    result["fragment_actions"] = []
    result["solver"] = {
        "backend": "scipy.optimize.milp/HiGHS",
        "optimal_throughput_req_s": optimum,
        "capacity_tolerance_req_s": CAPACITY_TOL,
        "phases": model.history,
    }
    result["transition_kind"] = "restricted_generator_to_pe"
    actions = []
    removed = set()
    nodes = {n["name"]: Node(**n) for n in source["nodes"]}
    for j, key, replacements, _ in options:
        for _ in range(int(x[j])):
            donor = groups[key].pop(0)
            removed.add(donor["id"])
            start = min(donor["gpu_ids"])
            added = []
            for template in replacements:
                ids = tuple(range(start, start + template.width))
                added.append(instance(nodes[donor["node"]], ids, template, "restricted_flip"))
                start += template.width
            result["instances"].extend(added)
            actions.append(
                {
                    "node": donor["node"],
                    "gpu_ids": donor["gpu_ids"],
                    "remove_instance_ids": [donor["id"]],
                    "add_instances": added,
                }
            )
    result["instances"] = sorted(
        (i for i in result["instances"] if i["id"] not in removed), key=signature
    )
    refresh(result, catalog)
    validate_deployment(result, catalog)
    return result, actions


def benefit(no_flip: float, changed: float) -> dict:
    delta = changed - no_flip
    if abs(delta) <= CAPACITY_TOL:
        delta = 0.0
    return {
        "no_flip_req_s": no_flip,
        "with_flip_req_s": changed,
        "delta_req_s": delta,
        "improvement_percent": 100 * delta / no_flip if no_flip > 0 else None,
        "extra_requests_per_minute": 60 * delta,
        "ideal_extra_requests_formula": "delta_req_s * long_phase_duration_s",
        "assumption": "Sufficient queued demand and zero transition cost; not a simulation result.",
    }


def build_flip_file(
    source: dict,
    target: dict,
    templates: dict[str, tuple[Template, ...]],
    catalog: ProfileCatalog,
    time_limit_s: float,
) -> dict:
    no_flip = evaluate(source, catalog, target["output_tokens"])
    restricted, restricted_actions = restricted_target(
        source, target, templates, catalog, time_limit_s
    )
    gap = target["throughput_req_s"] - restricted["throughput_req_s"]
    if gap < -2 * CAPACITY_TOL:
        raise ValueError("Restricted throughput exceeds the independently optimized full target")
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "flip",
        "generator": source["generator"],
        "scenario": source["scenario"],
        "source_output_tokens": source["output_tokens"],
        "target_output_tokens": target["output_tokens"],
        "source_layout_sha256": source["layout_sha256"],
        "target_layout_sha256": target["layout_sha256"],
        "profile_values_sha256": catalog.snapshot["values_sha256"],
        "no_flip": {
            "throughput_req_s": no_flip["throughput_req_s"],
            "stage_capacities_req_s": no_flip["stage_capacities_req_s"],
            "cpu_instances": no_flip["cpu_te"]["instances"],
        },
        "full": {
            "actions": diff_actions(source, target),
            "cpu_instances_before": source["cpu_te"]["instances"],
            "cpu_instances_after": target["cpu_te"]["instances"],
            "benefit": benefit(no_flip["throughput_req_s"], target["throughput_req_s"]),
        },
        "restricted": {
            "actions": restricted_actions,
            "deployment": restricted,
            "cpu_instances_before": source["cpu_te"]["instances"],
            "cpu_instances_after": restricted["cpu_te"]["instances"],
            "benefit": benefit(no_flip["throughput_req_s"], restricted["throughput_req_s"]),
            "gap_to_full_req_s": max(0, gap),
            "reaches_full_layout": layout_signature(restricted) == layout_signature(target),
            "reaches_full_template_counts": Counter(
                (i["hardware"], i["template"]) for i in restricted["instances"]
            )
            == Counter((i["hardware"], i["template"]) for i in target["instances"]),
        },
    }
    validate_flip(source, target, result, catalog)
    return result


def validate_flip(source: dict, target: dict, flip: dict, catalog: ProfileCatalog) -> None:
    if flip.get("schema_version") != SCHEMA_VERSION or flip.get("kind") != "flip":
        raise ValueError("Unsupported flip schema")
    if (
        flip["source_layout_sha256"] != source["layout_sha256"]
        or flip["target_layout_sha256"] != target["layout_sha256"]
        or flip["profile_values_sha256"] != catalog.snapshot["values_sha256"]
    ):
        raise ValueError("Flip provenance mismatch")
    baseline = evaluate(source, catalog, target["output_tokens"])
    expected_baseline = {
        "throughput_req_s": baseline["throughput_req_s"],
        "stage_capacities_req_s": baseline["stage_capacities_req_s"],
        "cpu_instances": baseline["cpu_te"]["instances"],
    }
    if flip["no_flip"] != expected_baseline:
        raise ValueError("Incorrect no-flip baseline")
    restricted = flip["restricted"]["deployment"]
    expected_gap = max(0, target["throughput_req_s"] - restricted["throughput_req_s"])
    if flip["restricted"]["gap_to_full_req_s"] != expected_gap:
        raise ValueError("Incorrect restricted throughput gap")
    if flip["restricted"]["reaches_full_layout"] != (
        layout_signature(restricted) == layout_signature(target)
    ):
        raise ValueError("Incorrect restricted layout equivalence")

    def counter(deployment):
        return Counter((i["hardware"], i["template"]) for i in deployment["instances"])

    if flip["restricted"]["reaches_full_template_counts"] != (
        counter(restricted) == counter(target)
    ):
        raise ValueError("Incorrect restricted template equivalence")
    for kind, expected in (("full", target), ("restricted", flip["restricted"]["deployment"])):
        validate_deployment(expected, catalog)
        if (
            flip[kind]["cpu_instances_before"] != source["cpu_te"]["instances"]
            or flip[kind]["cpu_instances_after"] != expected["cpu_te"]["instances"]
        ):
            raise ValueError("Incorrect flip CPU replica transition")
        expected_instances = {i["id"]: i for i in expected["instances"]}
        remaining = {i["id"]: copy.deepcopy(i) for i in source["instances"]}
        for action in flip[kind]["actions"]:
            lhs = []
            for instance_id in action["remove_instance_ids"]:
                if instance_id not in remaining:
                    raise ValueError("Flip removes an absent or already-removed instance")
                lhs.append(remaining.pop(instance_id))
            rhs = action["add_instances"]
            if any(i["node"] != action["node"] for i in lhs + rhs):
                raise ValueError("Flip crosses nodes")
            if set(action["gpu_ids"]) != {g for i in lhs + rhs for g in i["gpu_ids"]}:
                raise ValueError("Flip action GPU coverage is incorrect")
            if kind == "restricted":
                if (
                    len(lhs) != 1
                    or "PE" in lhs[0]["stages"]
                    or any(tuple(i["stages"]) != ("PE",) for i in rhs)
                ):
                    raise ValueError(
                        "Restricted flip must convert one whole generator bundle to PE"
                    )
                if {g for i in lhs for g in i["gpu_ids"]} != {g for i in rhs for g in i["gpu_ids"]}:
                    raise ValueError("Restricted flip changes its source GPU set")
            for item in rhs:
                if item != expected_instances.get(item["id"]):
                    raise ValueError("Flip replacement metadata does not match the target instance")
                if item["id"] in remaining:
                    raise ValueError("Flip adds an existing instance ID")
                if any(
                    i["node"] == item["node"] and set(i["gpu_ids"]) & set(item["gpu_ids"])
                    for i in remaining.values()
                ):
                    raise ValueError("Flip overlaps a retained instance")
                remaining[item["id"]] = copy.deepcopy(item)
        reconstructed = copy.deepcopy(expected)
        reconstructed.pop("core_instances", None)
        reconstructed["instances"] = list(remaining.values())
        reconstructed["cpu_te"]["instances"] = flip[kind]["cpu_instances_after"]
        reconstructed = evaluate(reconstructed, catalog, target["output_tokens"])
        validate_deployment(reconstructed, catalog)
        if reconstructed["layout_sha256"] != expected["layout_sha256"]:
            raise ValueError("Applying flip actions does not reproduce the target layout")
        if abs(reconstructed["throughput_req_s"] - expected["throughput_req_s"]) > CAPACITY_TOL:
            raise ValueError("Applying flip actions does not reproduce target capacity")
        if flip[kind]["benefit"] != benefit(
            baseline["throughput_req_s"], expected["throughput_req_s"]
        ):
            raise ValueError("Incorrect flip benefit")
