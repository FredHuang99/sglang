"""Config loaders for ShiftServe flip experiments.

The config layer intentionally accepts a small, explicit JSON contract while
preserving unknown metadata fields for experiment notes and future launch
options. This keeps the scheduler deterministic without making the validation
configs painful to iterate on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


INSTANCE_KIND_ALIASES = {
    "prompt_enhancer": "pe",
    "llm": "pe",
    "encoder": "te",
    "text_encoder": "te",
    "denoiser": "dit",
    "den": "dit",
    "decoder": "vae",
    "dec": "vae",
    "dit_vae_worker": "dit_vae",
}
VALID_INSTANCE_KINDS = {"pe", "te", "dit", "vae", "dit_vae"}


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _as_int_list(value: Any, *, field_name: str) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        tokens = [part for part in value.replace(",", " ").split() if part]
    else:
        tokens = list(value)
    result: list[int] = []
    for token in tokens:
        try:
            parsed = int(token)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} contains a non-integer value: {token}") from exc
        if parsed < 0:
            raise ValueError(f"{field_name} values must be non-negative: {parsed}")
        result.append(parsed)
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} contains duplicate values: {result}")
    return result


def normalize_instance_kind(kind: str) -> str:
    normalized = INSTANCE_KIND_ALIASES.get(kind.lower(), kind.lower())
    if normalized not in VALID_INSTANCE_KINDS:
        raise ValueError(
            f"Invalid ShiftServe instance kind '{kind}'. "
            f"Expected one of {sorted(VALID_INSTANCE_KINDS)}"
        )
    return normalized


@dataclass(frozen=True)
class NodeConfig:
    node_id: str
    host: str = "127.0.0.1"
    role_host: str | None = None
    node_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "NodeConfig":
        node_id = str(raw.get("node_id") or raw.get("id") or "")
        if not node_id:
            raise ValueError("Every node must define node_id")
        known = {"node_id", "id", "host", "role_host", "node_type"}
        return cls(
            node_id=node_id,
            host=str(raw.get("host", "127.0.0.1")),
            role_host=raw.get("role_host"),
            node_type=raw.get("node_type"),
            metadata={k: v for k, v in raw.items() if k not in known},
        )


@dataclass(frozen=True)
class InstanceConfig:
    id: str
    kind: str
    node_id: str
    device: str = "cuda"
    gpu_ids: list[int] = field(default_factory=list)
    ranks: int = 1
    ports: dict[str, int] = field(default_factory=dict)
    source_active: bool = False
    target_active: bool = False
    node_group_id: str | None = None
    pipeline_group_id: str | None = None
    model_id: str | None = None
    node_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "InstanceConfig":
        instance_id = str(raw.get("id") or raw.get("instance_id") or "")
        if not instance_id:
            raise ValueError("Every instance must define id")
        kind = normalize_instance_kind(str(raw.get("kind") or raw.get("role") or ""))
        node_id = str(raw.get("node_id") or "")
        if not node_id:
            raise ValueError(f"Instance {instance_id} must define node_id")
        ports = raw.get("ports", {})
        if not isinstance(ports, dict):
            raise ValueError(f"Instance {instance_id} ports must be an object")
        known = {
            "id",
            "instance_id",
            "kind",
            "role",
            "node_id",
            "device",
            "gpu_ids",
            "ranks",
            "ports",
            "source_active",
            "target_active",
            "node_group_id",
            "pipeline_group_id",
            "model_id",
            "node_type",
        }
        return cls(
            id=instance_id,
            kind=kind,
            node_id=node_id,
            device=str(raw.get("device", "cuda")),
            gpu_ids=_as_int_list(raw.get("gpu_ids"), field_name=f"{instance_id}.gpu_ids"),
            ranks=int(raw.get("ranks", 1)),
            ports={str(k): int(v) for k, v in ports.items()},
            source_active=bool(raw.get("source_active", False)),
            target_active=bool(raw.get("target_active", False)),
            node_group_id=raw.get("node_group_id"),
            pipeline_group_id=raw.get("pipeline_group_id"),
            model_id=raw.get("model_id"),
            node_type=raw.get("node_type"),
            metadata={k: v for k, v in raw.items() if k not in known},
        )


@dataclass(frozen=True)
class FlipPlanConfig:
    direction: str
    sources: list[str]
    targets: list[str]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FlipPlanConfig":
        direction = str(raw.get("direction") or "")
        if not direction:
            raise ValueError("Every flip_plan entry must define direction")
        return cls(
            direction=direction,
            sources=[str(v) for v in raw.get("sources", [])],
            targets=[str(v) for v in raw.get("targets", [])],
        )


@dataclass(frozen=True)
class BinConfig:
    short: int = 512
    long: int = 2048

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "BinConfig":
        raw = raw or {}
        short = int(raw.get("short", 512))
        long = int(raw.get("long", 2048))
        if short <= 0 or long <= 0 or short >= long:
            raise ValueError("bins must satisfy 0 < short < long")
        return cls(short=short, long=long)


@dataclass(frozen=True)
class DeploymentConfig:
    nodes: dict[str, NodeConfig]
    instances: dict[str, InstanceConfig]
    flip_plan: list[FlipPlanConfig] = field(default_factory=list)
    bins: BinConfig = field(default_factory=BinConfig)
    port_base: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DeploymentConfig":
        nodes = [NodeConfig.from_dict(item) for item in raw.get("nodes", [])]
        instances = [InstanceConfig.from_dict(item) for item in raw.get("instances", [])]
        node_map = {node.node_id: node for node in nodes}
        instance_map = {instance.id: instance for instance in instances}
        if len(node_map) != len(nodes):
            raise ValueError("Duplicate node_id in deployment config")
        if len(instance_map) != len(instances):
            raise ValueError("Duplicate instance id in deployment config")
        missing_nodes = sorted({i.node_id for i in instances} - set(node_map))
        if missing_nodes:
            raise ValueError(f"Instances reference unknown nodes: {missing_nodes}")
        for plan in raw.get("flip_plan", []):
            for field_name in ("sources", "targets"):
                missing = sorted(set(plan.get(field_name, [])) - set(instance_map))
                if missing:
                    raise ValueError(
                        f"flip_plan direction={plan.get('direction')} references "
                        f"unknown {field_name}: {missing}"
                    )
        known = {"nodes", "instances", "flip_plan", "bins", "port_base"}
        return cls(
            nodes=node_map,
            instances=instance_map,
            flip_plan=[FlipPlanConfig.from_dict(item) for item in raw.get("flip_plan", [])],
            bins=BinConfig.from_dict(raw.get("bins")),
            port_base=(None if raw.get("port_base") is None else int(raw["port_base"])),
            metadata={k: v for k, v in raw.items() if k not in known},
        )

    def active_instances(self, *, target: bool = False) -> list[InstanceConfig]:
        return [
            instance
            for instance in self.instances.values()
            if (instance.target_active if target else instance.source_active)
        ]


@dataclass(frozen=True)
class ProfileConfig:
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ProfileConfig":
        return cls(raw=raw)

    def lookup(self, *path: str, default: Any = None) -> Any:
        current: Any = self.raw
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current


@dataclass(frozen=True)
class TrafficIntervalConfig:
    start_min: float
    end_min: float
    rate_per_min: float
    bin: str = "short"
    input_tokens: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrafficIntervalConfig":
        return cls(
            start_min=float(raw["start_min"]),
            end_min=float(raw["end_min"]),
            rate_per_min=float(raw["rate_per_min"]),
            bin=str(raw.get("bin", "short")),
            input_tokens=(
                None if raw.get("input_tokens") is None else int(raw["input_tokens"])
            ),
        )


@dataclass(frozen=True)
class TrafficConfig:
    duration_min: float
    default_rate_per_min: float
    intervals: list[TrafficIntervalConfig] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrafficConfig":
        known = {"duration_min", "default_rate_per_min", "intervals"}
        return cls(
            duration_min=float(raw.get("duration_min", 1.0)),
            default_rate_per_min=float(raw.get("default_rate_per_min", 1.0)),
            intervals=[
                TrafficIntervalConfig.from_dict(item) for item in raw.get("intervals", [])
            ],
            metadata={k: v for k, v in raw.items() if k not in known},
        )


def load_deployment_config(path: str | Path) -> DeploymentConfig:
    return DeploymentConfig.from_dict(_read_json(path))


def load_profile_config(path: str | Path) -> ProfileConfig:
    return ProfileConfig.from_dict(_read_json(path))


def load_traffic_config(path: str | Path) -> TrafficConfig:
    return TrafficConfig.from_dict(_read_json(path))
