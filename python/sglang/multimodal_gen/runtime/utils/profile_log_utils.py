# SPDX-License-Identifier: Apache-2.0
"""Small helpers for stable, grep-friendly profiling log context."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProfileLogContext:
    role: str
    instance_id: str
    rank: str
    physical_rank: str
    world_size: str
    device: str
    mem_kind: str


def _stringify(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    return str(getattr(value, "value", value))


def get_profile_log_context(
    server_args: Any | None,
    *,
    rank: Any | None = None,
    physical_rank: Any | None = None,
) -> ProfileLogContext:
    """Build stable role/rank labels for profile logs.

    In pooled disagg launch, LOCAL_RANK is the physical GPU id for CUDA roles.
    For CPU roles it is the worker id, which is still the best per-process
    identifier for matching logs to a role instance.
    """

    role = _stringify(getattr(server_args, "disagg_role", None))
    instance_id = _stringify(getattr(server_args, "disagg_instance_id", None))

    if rank is None:
        rank = os.environ.get("RANK")
    if physical_rank is None:
        physical_rank = os.environ.get("LOCAL_RANK")

    world_size = os.environ.get("WORLD_SIZE")
    if world_size is None and server_args is not None:
        world_size = getattr(server_args, "num_gpus", None)

    device = "unknown"
    if server_args is not None:
        if hasattr(server_args, "resolved_role_device"):
            device = server_args.resolved_role_device()
        else:
            device = getattr(server_args, "disagg_role_device", "unknown")

    return ProfileLogContext(
        role=role,
        instance_id=instance_id,
        rank=_stringify(rank),
        physical_rank=_stringify(physical_rank),
        world_size=_stringify(world_size),
        device=_stringify(device),
        mem_kind="host" if device == "cpu" else "device",
    )


def bytes_to_gib(size: int | float) -> float:
    return float(size) / float(1024**3)
