"""Compile policies into immutable coefficients before invoking any solver."""

from __future__ import annotations

from dataclasses import dataclass

from .profiles import STAGES, WIDTHS, ProfileCatalog

SCENARIOS = ("cluster1", "cluster2", "clustersimu")


@dataclass(frozen=True)
class Node:
    name: str
    hardware: str
    gpu_count: int = 8
    hbm_per_gpu_gb: float = 80


@dataclass(frozen=True)
class Template:
    name: str
    hardware: str
    width: int
    stages: dict
    startup: dict

    @property
    def memory(self) -> float:
        return sum(p["memory_per_gpu_gb"] for p in self.stages.values())

    @property
    def caps(self) -> dict[str, float]:
        return {stage: self.stages.get(stage, {}).get("capacity_req_s", 0.0) for stage in STAGES}


def cluster(scenario: str) -> tuple[Node, ...]:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    counts = (
        {"a100_80g": 1, "h100": 1}
        if scenario != "clustersimu"
        else {"a800": 8, "a100_40g": 12, "h100": 4}
    )
    return tuple(
        Node(f"{hw}-{i:02d}", hw, hbm_per_gpu_gb=40 if hw == "a100_40g" else 80)
        for hw, count in counts.items()
        for i in range(1, count + 1)
    )


def policy(scenario: str) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
    if scenario == "cluster1":
        specs = [("PE", WIDTHS), ("DiT", WIDTHS), ("VAE", WIDTHS), ("DiT_VAE_Comb", WIDTHS)]
    elif scenario == "cluster2":
        specs = [("PE", (1, 2)), ("DiT_VAE_Comb", (2, 4))]
    elif scenario == "clustersimu":
        specs = [("PE", (2,)), ("DiT_VAE_Comb", (8,))]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return tuple(
        (
            f"{kind}_{'TP' if kind == 'PE' else 'SP'}{width}_Only",
            width,
            ("DiT", "VAE") if kind == "DiT_VAE_Comb" else (kind,),
        )
        for kind, widths in specs
        for width in widths
    )


def compile_templates(
    catalog: ProfileCatalog,
    generator: str,
    scenario: str,
    input_tokens: int,
    output_tokens: int,
    kv_cache_tokens: int,
    nodes: tuple[Node, ...] | None = None,
) -> tuple[dict[str, tuple[Template, ...]], list[dict]]:
    if min(input_tokens, output_tokens, kv_cache_tokens) <= 0:
        raise ValueError("Token lengths must be positive")
    if input_tokens + output_tokens > kv_cache_tokens:
        raise ValueError("KV pool must cover the input and output of one profiled request")
    nodes = nodes or cluster(scenario)
    hardware = {}
    for node in nodes:
        signature = (node.gpu_count, node.hbm_per_gpu_gb)
        if node.hardware in hardware and hardware[node.hardware] != signature:
            raise ValueError("A hardware type must have a uniform GPU count and per-GPU HBM")
        hardware[node.hardware] = signature
    legal, rejected = {}, []
    for hw, (gpu_count, hbm) in sorted(hardware.items()):
        candidates = []
        for name, width, members in policy(scenario):
            if width > gpu_count:
                continue
            stages = {
                stage: catalog.stage(
                    generator, hw, stage, width, input_tokens, output_tokens, kv_cache_tokens
                )
                for stage in members
            }
            template = Template(
                name, hw, width, stages, catalog.startup(generator, hw, width, members == ("PE",))
            )
            if template.memory <= hbm + 1e-9:
                candidates.append(template)
            else:
                rejected.append(
                    {
                        "hardware": hw,
                        "template": name,
                        "memory_per_gpu_gb": template.memory,
                        "hbm_per_gpu_gb": hbm,
                        "reason": "per_gpu_hbm_exceeded",
                    }
                )
        legal[hw] = tuple(candidates)
    return legal, rejected


def partitions(total: int, widths: tuple[int, ...] = WIDTHS) -> tuple[tuple[int, ...], ...]:
    result = []

    def visit(left: int, maximum: int, current: tuple[int, ...]) -> None:
        if left == 0:
            result.append(current)
        for width in sorted(widths, reverse=True):
            if width <= left and width <= maximum:
                visit(left - width, width, current + (width,))

    visit(total, total, ())
    return tuple(result)
