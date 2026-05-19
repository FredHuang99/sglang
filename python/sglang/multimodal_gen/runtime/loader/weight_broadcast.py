# SPDX-License-Identifier: Apache-2.0
"""Rank0 SP broadcast helpers for diffusion checkpoint loading."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.weight_load_profiler import (
    WEIGHT_LOAD_NCCL_BROADCAST_MS,
    WEIGHT_LOAD_RANK0_WAIT_MS,
    DiffusionWeightLoadProfiler,
)

logger = init_logger(__name__)

DIFFUSION_WEIGHT_LOAD_MODE_CHOICES: tuple[str, ...] = (
    "default",
    "rank0-broadcast",
)
SUPPORTED_BROADCAST_COMPONENTS: tuple[str, ...] = ("transformer",)


@dataclass(frozen=True)
class Rank0BroadcastDecision:
    enabled: bool
    requested_mode: str
    effective_mode: str
    reason: str
    sp_rank: int
    sp_world_size: int
    sp_group: Any | None = None


def normalize_weight_load_mode(mode: str | None) -> str:
    normalized = "default" if mode is None else str(mode).lower()
    if normalized not in DIFFUSION_WEIGHT_LOAD_MODE_CHOICES:
        raise ValueError(
            "Invalid diffusion weight load mode: "
            f"{mode}. Must be one of {DIFFUSION_WEIGHT_LOAD_MODE_CHOICES}."
        )
    return normalized


def normalize_broadcast_components(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.replace(";", ",").split(",")
    else:
        raw_items = list(value)
    return [str(item).strip().lower() for item in raw_items if str(item).strip()]


def resolve_rank0_broadcast_decision(
    *,
    load_mode: str | None,
    broadcast_components: str | Iterable[str] | None,
    component_name: str | None,
    tp_size: int | None,
    fsdp_inference: bool,
) -> Rank0BroadcastDecision:
    requested_mode = normalize_weight_load_mode(load_mode)
    component = (component_name or "").lower()
    components = normalize_broadcast_components(broadcast_components)

    if requested_mode != "rank0-broadcast":
        return Rank0BroadcastDecision(
            enabled=False,
            requested_mode=requested_mode,
            effective_mode="default",
            reason="load_mode_default",
            sp_rank=0,
            sp_world_size=1,
        )

    if component not in components:
        return Rank0BroadcastDecision(
            enabled=False,
            requested_mode=requested_mode,
            effective_mode="default",
            reason=f"component_not_enabled:{component or 'unknown'}",
            sp_rank=0,
            sp_world_size=1,
        )

    if component not in SUPPORTED_BROADCAST_COMPONENTS:
        logger.warning(
            "Diffusion rank0-broadcast does not support component=%s; "
            "falling back to default loader.",
            component,
        )
        return Rank0BroadcastDecision(
            enabled=False,
            requested_mode=requested_mode,
            effective_mode="default",
            reason=f"component_unsupported:{component}",
            sp_rank=0,
            sp_world_size=1,
        )

    if int(tp_size or 1) != 1:
        logger.warning(
            "Diffusion rank0-broadcast requires tp_size=1, got tp_size=%s; "
            "falling back to default loader.",
            tp_size,
        )
        return Rank0BroadcastDecision(
            enabled=False,
            requested_mode=requested_mode,
            effective_mode="default",
            reason=f"tp_size_not_supported:{tp_size}",
            sp_rank=0,
            sp_world_size=1,
        )

    if fsdp_inference:
        logger.warning(
            "Diffusion rank0-broadcast is disabled when FSDP inference is enabled; "
            "falling back to default loader."
        )
        return Rank0BroadcastDecision(
            enabled=False,
            requested_mode=requested_mode,
            effective_mode="default",
            reason="fsdp_inference_enabled",
            sp_rank=0,
            sp_world_size=1,
        )

    try:
        from sglang.multimodal_gen.runtime.distributed import parallel_state

        if not parallel_state.model_parallel_is_initialized():
            return Rank0BroadcastDecision(
                enabled=False,
                requested_mode=requested_mode,
                effective_mode="default",
                reason="model_parallel_uninitialized",
                sp_rank=0,
                sp_world_size=1,
            )
        sp_group = parallel_state.get_sp_group()
        sp_rank = int(parallel_state.get_sp_parallel_rank())
        sp_world_size = int(parallel_state.get_sp_world_size())
    except Exception as exc:
        logger.warning(
            "Could not resolve SP group for diffusion rank0-broadcast; "
            "falling back to default loader. error=%s",
            exc,
        )
        return Rank0BroadcastDecision(
            enabled=False,
            requested_mode=requested_mode,
            effective_mode="default",
            reason=f"sp_group_error:{type(exc).__name__}",
            sp_rank=0,
            sp_world_size=1,
        )

    if sp_world_size <= 1:
        return Rank0BroadcastDecision(
            enabled=False,
            requested_mode=requested_mode,
            effective_mode="default",
            reason="sp_world_size_le_1",
            sp_rank=sp_rank,
            sp_world_size=sp_world_size,
            sp_group=sp_group,
        )

    return Rank0BroadcastDecision(
        enabled=True,
        requested_mode=requested_mode,
        effective_mode="rank0-broadcast",
        reason="enabled",
        sp_rank=sp_rank,
        sp_world_size=sp_world_size,
        sp_group=sp_group,
    )


def set_profile_load_mode(
    profile: DiffusionWeightLoadProfiler | None,
    decision: Rank0BroadcastDecision,
) -> None:
    if profile is None:
        return
    profile.set_load_mode_requested(decision.requested_mode)
    profile.set_load_mode_effective(decision.effective_mode)


def materialize_empty_model_state_dict(
    model: nn.Module,
    *,
    device: torch.device,
    strict: bool,
) -> Any:
    """Materialize a meta-initialized module with empty tensors on device."""
    empty_state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if not isinstance(tensor, torch.Tensor):
            continue
        empty_state_dict[name] = torch.empty(
            tuple(tensor.shape), dtype=tensor.dtype, device=device
        )
    result = model.load_state_dict(empty_state_dict, strict=strict, assign=True)
    model.reverse_param_names_mapping = {}
    return result


def offload_model_tensors_to_cpu(model: nn.Module) -> None:
    model.to(torch.device("cpu"))


def iter_module_tensors(model: nn.Module) -> list[tuple[str, torch.Tensor, str]]:
    entries: list[tuple[str, torch.Tensor, str]] = []
    seen: set[int] = set()
    for name, param in model.named_parameters():
        tensor = param.data
        ident = id(tensor)
        if ident not in seen:
            entries.append((name, tensor, "parameter"))
            seen.add(ident)
    for name, buffer in model.named_buffers():
        tensor = buffer
        ident = id(tensor)
        if ident not in seen:
            entries.append((name, tensor, "buffer"))
            seen.add(ident)
    entries.sort(key=lambda item: item[0])
    return entries


def metadata_for_entries(
    entries: Iterable[tuple[str, torch.Tensor, str]]
) -> list[tuple[str, tuple[int, ...], str, str]]:
    return [
        (name, tuple(tensor.shape), str(tensor.dtype), kind)
        for name, tensor, kind in entries
    ]


def _collective_device(
    entries: list[tuple[str, torch.Tensor, str]], sp_group
) -> torch.device:
    if entries:
        return entries[0][1].device
    device = getattr(sp_group, "device", torch.device("cpu"))
    return torch.device(device)


def validate_broadcast_metadata(
    entries: list[tuple[str, torch.Tensor, str]],
    sp_group,
    *,
    weight_load_profile: DiffusionWeightLoadProfiler | None = None,
) -> None:
    local_metadata = metadata_for_entries(entries)
    rank0_metadata = sp_group.broadcast_object(
        local_metadata if sp_group.rank_in_group == 0 else None,
        src=0,
    )
    local_ok = local_metadata == rank0_metadata

    device = _collective_device(entries, sp_group)
    ok_tensor = torch.tensor(
        [1 if local_ok else 0],
        device=device,
        dtype=torch.int32,
    )
    sp_group.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
    if int(ok_tensor.item()) != 1:
        error = "rank0-broadcast metadata mismatch before tensor broadcast"
        if weight_load_profile is not None:
            weight_load_profile.set_broadcast_error(error)
        raise RuntimeError(error)


def wait_for_rank0_ready(
    entries: list[tuple[str, torch.Tensor, str]],
    sp_group,
    *,
    weight_load_profile: DiffusionWeightLoadProfiler | None = None,
) -> None:
    device = _collective_device(entries, sp_group)
    ready = torch.ones(1, device=device, dtype=torch.int32)
    start = time.perf_counter()
    sp_group.broadcast(ready, src=0)
    if weight_load_profile is not None:
        weight_load_profile.add_ms(
            WEIGHT_LOAD_RANK0_WAIT_MS,
            (time.perf_counter() - start) * 1000.0,
        )


def broadcast_module_tensors(
    model: nn.Module,
    sp_group,
    *,
    weight_load_profile: DiffusionWeightLoadProfiler | None = None,
) -> None:
    entries = iter_module_tensors(model)
    wait_for_rank0_ready(
        entries,
        sp_group,
        weight_load_profile=weight_load_profile,
    )
    validate_broadcast_metadata(
        entries,
        sp_group,
        weight_load_profile=weight_load_profile,
    )

    start = time.perf_counter()
    try:
        for _name, tensor, _kind in entries:
            if tensor.numel() == 0:
                continue
            sp_group.broadcast(tensor, src=0)
            if weight_load_profile is not None:
                weight_load_profile.add_broadcast_tensor(tensor)
    except Exception as exc:
        if weight_load_profile is not None:
            weight_load_profile.set_broadcast_error(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if weight_load_profile is not None:
            weight_load_profile.add_ms(
                WEIGHT_LOAD_NCCL_BROADCAST_MS,
                (time.perf_counter() - start) * 1000.0,
            )
