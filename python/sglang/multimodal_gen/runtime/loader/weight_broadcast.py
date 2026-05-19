# SPDX-License-Identifier: Apache-2.0
"""Rank0 SP broadcast helpers for diffusion checkpoint loading."""

from __future__ import annotations

import os
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
_BROADCAST_PROGRESS_BYTES = 1 << 30


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


def _group_rank(sp_group) -> int | str:
    return getattr(sp_group, "rank", "unknown")


def _sp_rank(sp_group) -> int | str:
    return getattr(sp_group, "rank_in_group", "unknown")


def _sp_world_size(sp_group) -> int | str:
    return getattr(sp_group, "world_size", "unknown")


def log_broadcast_stage(
    stage: str,
    sp_group=None,
    *,
    component_name: str | None = None,
    detail: str | None = None,
    weight_load_profile: DiffusionWeightLoadProfiler | None = None,
) -> None:
    logger.info(
        "DiffusionRank0BroadcastStage stage=%s component=%s rank=%s "
        "sp_rank=%s sp_world_size=%s pid=%s%s",
        stage,
        component_name or "unknown",
        _group_rank(sp_group),
        _sp_rank(sp_group),
        _sp_world_size(sp_group),
        os.getpid(),
        f" {detail}" if detail else "",
        main_process_only=False,
        local_main_process_only=False,
    )
    if weight_load_profile is not None:
        weight_load_profile.record_stage(stage, detail=detail)


def _cpu_control_all_reduce_min(local_ok: bool, sp_group) -> bool:
    ok_tensor = torch.tensor([1 if local_ok else 0], dtype=torch.int32)
    if getattr(sp_group, "world_size", 1) <= 1:
        return bool(ok_tensor.item())

    cpu_group = getattr(sp_group, "cpu_group", None)
    if cpu_group is not None:
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN, group=cpu_group)
    else:
        sp_group.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
    return bool(ok_tensor.item())


def confirm_rank0_broadcast_entry(
    sp_group,
    *,
    component_name: str | None = None,
    weight_load_profile: DiffusionWeightLoadProfiler | None = None,
) -> None:
    """Confirm every SP rank has entered the rank0-broadcast branch."""
    log_broadcast_stage(
        "entry_confirm_enter",
        sp_group,
        component_name=component_name,
        weight_load_profile=weight_load_profile,
    )
    local_metadata = (
        component_name or "unknown",
        int(getattr(sp_group, "world_size", 1)),
    )
    rank0_metadata = sp_group.broadcast_object(
        local_metadata if getattr(sp_group, "rank_in_group", 0) == 0 else None,
        src=0,
    )
    local_ok = local_metadata == rank0_metadata
    if not _cpu_control_all_reduce_min(local_ok, sp_group):
        error = (
            "rank0-broadcast entry mismatch before weight read: "
            f"local={local_metadata} rank0={rank0_metadata}"
        )
        if weight_load_profile is not None:
            weight_load_profile.set_broadcast_error(error)
        log_broadcast_stage(
            "entry_confirm_error",
            sp_group,
            component_name=component_name,
            detail=error,
            weight_load_profile=weight_load_profile,
        )
        raise RuntimeError(error)
    log_broadcast_stage(
        "entry_confirm_exit",
        sp_group,
        component_name=component_name,
        weight_load_profile=weight_load_profile,
    )


def broadcast_rank0_load_status(
    sp_group,
    *,
    rank0_status: dict[str, Any] | None,
    component_name: str | None = None,
    weight_load_profile: DiffusionWeightLoadProfiler | None = None,
) -> dict[str, Any]:
    log_broadcast_stage(
        "rank0_status_broadcast_enter",
        sp_group,
        component_name=component_name,
        weight_load_profile=weight_load_profile,
    )
    start = time.perf_counter()
    status = sp_group.broadcast_object(
        rank0_status if getattr(sp_group, "rank_in_group", 0) == 0 else None,
        src=0,
    )
    if weight_load_profile is not None:
        weight_load_profile.add_ms(
            WEIGHT_LOAD_RANK0_WAIT_MS,
            (time.perf_counter() - start) * 1000.0,
        )
    log_broadcast_stage(
        "rank0_status_broadcast_exit",
        sp_group,
        component_name=component_name,
        detail=f"ok={bool(status and status.get('ok'))}",
        weight_load_profile=weight_load_profile,
    )
    if not isinstance(status, dict):
        error = f"rank0 load status must be dict, got {type(status).__name__}"
        if weight_load_profile is not None:
            weight_load_profile.set_broadcast_error(error)
        raise RuntimeError(error)
    return status


def confirm_tensor_broadcast_ready(
    sp_group,
    *,
    local_ok: bool,
    component_name: str | None = None,
    local_error: str | None = None,
    weight_load_profile: DiffusionWeightLoadProfiler | None = None,
) -> None:
    log_broadcast_stage(
        "tensor_ready_confirm_enter",
        sp_group,
        component_name=component_name,
        detail=f"local_ok={local_ok}",
        weight_load_profile=weight_load_profile,
    )
    if _cpu_control_all_reduce_min(local_ok, sp_group):
        log_broadcast_stage(
            "tensor_ready_confirm_exit",
            sp_group,
            component_name=component_name,
            detail="ok=True",
            weight_load_profile=weight_load_profile,
        )
        return

    error = local_error or "some SP rank failed before tensor broadcast"
    if weight_load_profile is not None:
        weight_load_profile.set_broadcast_error(error)
    log_broadcast_stage(
        "tensor_ready_confirm_error",
        sp_group,
        component_name=component_name,
        detail=error,
        weight_load_profile=weight_load_profile,
    )
    raise RuntimeError(error)


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


def validate_broadcast_metadata(
    entries: list[tuple[str, torch.Tensor, str]],
    sp_group,
    *,
    component_name: str | None = None,
    weight_load_profile: DiffusionWeightLoadProfiler | None = None,
) -> None:
    log_broadcast_stage(
        "metadata_broadcast_enter",
        sp_group,
        component_name=component_name,
        detail=f"tensor_count={len(entries)}",
        weight_load_profile=weight_load_profile,
    )
    local_metadata = metadata_for_entries(entries)
    rank0_metadata = sp_group.broadcast_object(
        local_metadata if sp_group.rank_in_group == 0 else None,
        src=0,
    )
    local_ok = local_metadata == rank0_metadata

    if not _cpu_control_all_reduce_min(local_ok, sp_group):
        error = "rank0-broadcast metadata mismatch before tensor broadcast"
        if weight_load_profile is not None:
            weight_load_profile.set_broadcast_error(error)
        log_broadcast_stage(
            "metadata_broadcast_error",
            sp_group,
            component_name=component_name,
            detail=error,
            weight_load_profile=weight_load_profile,
        )
        raise RuntimeError(error)
    log_broadcast_stage(
        "metadata_broadcast_exit",
        sp_group,
        component_name=component_name,
        detail=f"tensor_count={len(entries)}",
        weight_load_profile=weight_load_profile,
    )

def broadcast_module_tensors(
    model: nn.Module,
    sp_group,
    *,
    component_name: str | None = None,
    weight_load_profile: DiffusionWeightLoadProfiler | None = None,
) -> None:
    entries = iter_module_tensors(model)
    validate_broadcast_metadata(
        entries,
        sp_group,
        component_name=component_name,
        weight_load_profile=weight_load_profile,
    )

    log_broadcast_stage(
        "tensor_broadcast_start",
        sp_group,
        component_name=component_name,
        detail=f"tensor_count={len(entries)}",
        weight_load_profile=weight_load_profile,
    )
    start = time.perf_counter()
    broadcast_bytes = 0
    next_progress_bytes = _BROADCAST_PROGRESS_BYTES
    try:
        for name, tensor, _kind in entries:
            if tensor.numel() == 0:
                continue
            sp_group.broadcast(tensor, src=0)
            if weight_load_profile is not None:
                weight_load_profile.add_broadcast_tensor(tensor)
            broadcast_bytes += int(tensor.numel() * tensor.element_size())
            if broadcast_bytes >= next_progress_bytes:
                log_broadcast_stage(
                    "tensor_broadcast_progress",
                    sp_group,
                    component_name=component_name,
                    detail=(
                        f"bytes={broadcast_bytes} last_tensor={name} "
                        f"tensor_count={len(entries)}"
                    ),
                    weight_load_profile=weight_load_profile,
                )
                next_progress_bytes += _BROADCAST_PROGRESS_BYTES
    except Exception as exc:
        if weight_load_profile is not None:
            weight_load_profile.set_broadcast_error(f"{type(exc).__name__}: {exc}")
        log_broadcast_stage(
            "tensor_broadcast_error",
            sp_group,
            component_name=component_name,
            detail=f"{type(exc).__name__}: {exc}",
            weight_load_profile=weight_load_profile,
        )
        raise
    finally:
        if weight_load_profile is not None:
            weight_load_profile.add_ms(
                WEIGHT_LOAD_NCCL_BROADCAST_MS,
                (time.perf_counter() - start) * 1000.0,
            )
    log_broadcast_stage(
        "tensor_broadcast_done",
        sp_group,
        component_name=component_name,
        detail=f"bytes={broadcast_bytes} tensor_count={len(entries)}",
        weight_load_profile=weight_load_profile,
    )
