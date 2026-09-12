"""Physical instances, fragment filling, and independent deployment evaluation."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict

from .profiles import CAPACITY_MODEL, STAGES, ProfileCatalog, digest
from .templates import Node, Template, cluster, compile_templates

SCHEMA_VERSION = 1


def signature(instance: dict) -> tuple:
    return instance["node"], tuple(instance["gpu_ids"]), instance["template"]


def layout_signature(deployment: dict) -> str:
    return digest(sorted(signature(i) for i in deployment["instances"]))


def instance(node: Node, gpu_ids: tuple[int, ...], template: Template, origin: str = "ilp") -> dict:
    gpu_ids = tuple(sorted(gpu_ids))
    label = "_".join(map(str, gpu_ids))
    return {
        "id": f"{node.name}:gpu-{label}:{template.name}",
        "node": node.name,
        "gpu_ids": list(gpu_ids),
        "hardware": node.hardware,
        "template": template.name,
        "bundle_size": template.width,
        "memory_per_gpu_gb": template.memory,
        "stages": copy.deepcopy(template.stages),
        "startup": copy.deepcopy(template.startup),
        "origin": origin,
    }


def capacities(instances: list[dict]) -> dict[str, float]:
    return {
        s: math.fsum(i["stages"].get(s, {}).get("capacity_req_s", 0) for i in instances)
        for s in STAGES
    }


def cpu_count(throughput: float, latency: float) -> int:
    demand = throughput * latency
    nearest = round(demand)
    # Remove floating-point roundoff only, not a genuinely positive residual demand.
    if abs(demand - nearest) <= 4 * math.ulp(demand):
        return max(0, nearest)
    return max(0, math.ceil(demand))


def refresh(deployment: dict, catalog: ProfileCatalog, cpu_instances: int | None = None) -> dict:
    caps = capacities(deployment["instances"])
    gpu_throughput = min(caps.values())
    cpu = catalog.cpu(deployment["generator"])
    cpu["instances"] = (
        cpu_count(gpu_throughput, cpu["latency_s"]) if cpu_instances is None else cpu_instances
    )
    cpu["aggregate_capacity_req_s"] = cpu["instances"] * cpu["capacity_per_instance_req_s"]
    caps["TE"] = cpu["aggregate_capacity_req_s"]
    deployment["cpu_te"] = cpu
    deployment["stage_capacities_req_s"] = caps
    deployment["throughput_req_s"] = min(caps.values())
    deployment["bottlenecks"] = [
        s
        for s, c in caps.items()
        if math.isclose(c, min(caps.values()), abs_tol=1e-9, rel_tol=1e-8)
    ]
    deployment["layout_sha256"] = layout_signature(deployment)
    return deployment


def make_deployment(
    catalog: ProfileCatalog,
    generator: str,
    scenario: str,
    input_tokens: int,
    output_tokens: int,
    kv_cache_tokens: int,
    placements: list,
    solver: dict,
    rejected: list[dict],
    nodes: tuple[Node, ...] | None = None,
) -> dict:
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "deployment",
        "generator": generator,
        "pe_model": "pe7b",
        "scenario": scenario,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "kv_cache_tokens": kv_cache_tokens,
        "denoising_steps_profiled": 50,
        "capacity_model": CAPACITY_MODEL,
        "network_assumption": "NVLink bundles within a node; inter-node transfer is not a bottleneck",
        "profile_values_sha256": catalog.snapshot["values_sha256"],
        "nodes": [asdict(n) for n in (nodes or cluster(scenario))],
        "instances": [instance(n, ids, t) for n, ids, t in placements],
        "solver": solver,
        "rejected_templates": rejected,
        "fragment_policy": "same_template_first",
        "fragment_actions": [],
    }
    return refresh(result, catalog)


def legal_gpu_groups(gpu_count: int, width: int) -> tuple[tuple[int, ...], ...]:
    # NVLink profile assumes interchangeable ranks. Canonical aligned groups cover all
    # partitions of 1/2/4/8 GPUs and make retained groups unambiguous during flips.
    return tuple(
        tuple(range(start, start + width)) for start in range(0, gpu_count - width + 1, width)
    )


def fill_fragments(
    deployment: dict,
    templates: dict[str, tuple[Template, ...]],
    catalog: ProfileCatalog,
    strategy: str = "same_template_first",
) -> dict:
    if strategy not in ("same_template_first", "minimum_excess"):
        raise ValueError("Unknown fragment filling strategy")
    result = copy.deepcopy(deployment)
    result["core_instances"] = copy.deepcopy(deployment["instances"])
    result["core_throughput_req_s"] = deployment["throughput_req_s"]
    result["fragment_policy"] = strategy
    for raw in result["nodes"]:
        node = Node(**raw)
        existing = [i for i in result["instances"] if i["node"] == node.name]
        if not existing:
            continue
        used = {g for i in existing for g in i["gpu_ids"]}
        free = set(range(node.gpu_count)) - used
        choices = templates[node.hardware]
        lookup = {t.name: t for t in choices}
        if strategy == "same_template_first":
            same = sorted(
                {i["template"] for i in existing}, key=lambda name: (-lookup[name].width, name)
            )
            while free:
                placed = False
                for name in same:
                    template = lookup[name]
                    groups = [
                        ids
                        for ids in legal_gpu_groups(node.gpu_count, template.width)
                        if set(ids) <= free
                    ]
                    if groups:
                        ids = groups[0]
                        result["instances"].append(
                            instance(node, ids, template, "fragment_same_template")
                        )
                        result["fragment_actions"].append(
                            {
                                "node": node.name,
                                "gpu_ids": list(ids),
                                "template": name,
                                "rule": "same_template",
                            }
                        )
                        free.difference_update(ids)
                        placed = True
                        break
                if not placed:
                    break
        if not free:
            continue
        # Enumerate only the small fragment of an already-active <=8-GPU node.
        current_caps = capacities(result["instances"])
        best = None
        best_score = None

        def visit(remaining: set[int], selected: list[tuple], increments: dict) -> None:
            nonlocal best, best_score
            if not remaining:
                final = {s: current_caps[s] + increments[s] for s in STAGES}
                throughput = min(final.values())
                excess = sum(c - throughput for c in final.values())
                score = (
                    -round(throughput, 12),
                    round(excess, 12),
                    len(selected),
                    tuple((t.name, ids) for ids, t in selected),
                )
                if best_score is None or score < best_score:
                    best_score, best = score, list(selected)
                return
            first = min(remaining)
            for t in choices:
                for ids in legal_gpu_groups(node.gpu_count, t.width):
                    if first in ids and set(ids) <= remaining:
                        visit(
                            remaining - set(ids),
                            selected + [(ids, t)],
                            {s: increments[s] + t.caps[s] for s in STAGES},
                        )

        visit(free, [], dict.fromkeys(STAGES, 0.0))
        for ids, t in best or []:
            result["instances"].append(instance(node, ids, t, "fragment_minimum_excess"))
            result["fragment_actions"].append(
                {
                    "node": node.name,
                    "gpu_ids": list(ids),
                    "template": t.name,
                    "rule": "minimum_excess",
                }
            )
    result["instances"].sort(key=signature)
    return refresh(result, catalog)


def evaluate(deployment: dict, catalog: ProfileCatalog, output_tokens: int) -> dict:
    """Reprofile a FIXED physical layout and preserve its existing CPU replica count."""
    result = copy.deepcopy(deployment)
    result["output_tokens"] = output_tokens
    result["evaluation_only"] = True
    for item in result["instances"]:
        item["stages"] = {
            s: catalog.stage(
                result["generator"],
                item["hardware"],
                s,
                item["bundle_size"],
                result["input_tokens"],
                output_tokens,
                result["kv_cache_tokens"],
            )
            for s in item["stages"]
        }
        item["memory_per_gpu_gb"] = sum(p["memory_per_gpu_gb"] for p in item["stages"].values())
    return refresh(result, catalog, deployment["cpu_te"]["instances"])


def validate_deployment(deployment: dict, catalog: ProfileCatalog) -> None:
    if deployment.get("schema_version") != SCHEMA_VERSION or deployment.get("kind") != "deployment":
        raise ValueError("Unsupported deployment schema")
    if deployment["profile_values_sha256"] != catalog.snapshot["values_sha256"]:
        raise ValueError("Deployment/profile fingerprint mismatch")
    nodes = tuple(Node(**n) for n in deployment["nodes"])
    if len({n.name for n in nodes}) != len(nodes):
        raise ValueError("Duplicate node ID")
    compiled, _ = compile_templates(
        catalog,
        deployment["generator"],
        deployment["scenario"],
        deployment["input_tokens"],
        deployment["output_tokens"],
        deployment["kv_cache_tokens"],
        nodes,
    )
    node_map = {n.name: n for n in nodes}
    lookup = {(hw, t.name): t for hw, ts in compiled.items() for t in ts}
    seen_ids, occupied = set(), set()
    for item in deployment["instances"]:
        if item["id"] in seen_ids or item["node"] not in node_map:
            raise ValueError("Duplicate instance ID or unknown node")
        seen_ids.add(item["id"])
        node = node_map[item["node"]]
        if node.hardware != item["hardware"]:
            raise ValueError("Instance hardware does not match node")
        try:
            template = lookup[item["hardware"], item["template"]]
        except KeyError as exc:
            raise ValueError("Instance uses an illegal template") from exc
        ids = tuple(item["gpu_ids"])
        if (
            ids not in legal_gpu_groups(node.gpu_count, template.width)
            or item["bundle_size"] != template.width
        ):
            raise ValueError("Invalid bundle GPU IDs or width")
        expected = instance(node, ids, template)
        if (
            item["id"] != expected["id"]
            or item["stages"] != expected["stages"]
            or item["startup"] != expected["startup"]
        ):
            raise ValueError("Incorrect instance ID, profile coefficients, or startup metadata")
        if not math.isclose(item["memory_per_gpu_gb"], template.memory, abs_tol=1e-9):
            raise ValueError("Incorrect per-GPU memory total")
        for gpu in ids:
            key = node.name, gpu
            if key in occupied:
                raise ValueError("Two instances occupy the same GPU")
            occupied.add(key)
    expected = evaluate(deployment, catalog, deployment["output_tokens"])
    if expected["cpu_te"] != deployment["cpu_te"]:
        raise ValueError("CPU profile or aggregate capacity is incorrect")
    for stage, capacity in expected["stage_capacities_req_s"].items():
        if not math.isclose(
            capacity, deployment["stage_capacities_req_s"][stage], abs_tol=1e-10, rel_tol=1e-9
        ):
            raise ValueError("Stage capacity total is incorrect")
    if not math.isclose(
        expected["throughput_req_s"], deployment["throughput_req_s"], abs_tol=1e-10, rel_tol=1e-9
    ):
        raise ValueError("System throughput total is incorrect")
    if deployment["layout_sha256"] != layout_signature(deployment):
        raise ValueError("Deployment layout checksum mismatch")
    n_cpu = deployment["cpu_te"]["instances"]
    if isinstance(n_cpu, bool) or not isinstance(n_cpu, int) or n_cpu < 0:
        raise ValueError("CPU replica count must be a nonnegative integer")
    minimum = cpu_count(
        min(capacities(deployment["instances"]).values()), expected["cpu_te"]["latency_s"]
    )
    if n_cpu < minimum:
        raise ValueError("CPU TE does not provide the required capacity")
    if "core_instances" in deployment:
        core = {signature(i) for i in deployment["core_instances"]}
        final = {signature(i) for i in deployment["instances"]}
        if not core <= final:
            raise ValueError("Fragment filling removed a core instance")
        if {i["node"] for i in deployment["instances"]} != {
            i["node"] for i in deployment["core_instances"]
        }:
            raise ValueError("Fragment filling activated an idle node")
