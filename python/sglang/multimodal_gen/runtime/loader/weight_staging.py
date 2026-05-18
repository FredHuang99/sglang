# SPDX-License-Identifier: Apache-2.0
"""Optional CPU staging for diffusion checkpoint tensors."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Literal

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.weight_load_profiler import (
    WEIGHT_LOAD_PIN_MEMORY_MS,
    DiffusionWeightLoadProfiler,
)

logger = init_logger(__name__)

DiffusionWeightStagingMode = Literal["none", "pageable", "pinned", "auto"]
DIFFUSION_WEIGHT_STAGING_CHOICES: tuple[str, ...] = (
    "none",
    "pageable",
    "pinned",
    "auto",
)


def normalize_weight_staging_mode(mode: str | None) -> str:
    normalized = "none" if mode is None else str(mode).lower()
    if normalized not in DIFFUSION_WEIGHT_STAGING_CHOICES:
        raise ValueError(
            "Invalid diffusion weight staging mode: "
            f"{mode}. Must be one of {DIFFUSION_WEIGHT_STAGING_CHOICES}."
        )
    return normalized


def is_weight_staging_rank0() -> bool:
    """Return True for SP rank0, or for single-process/uninitialized launches."""
    try:
        from sglang.multimodal_gen.runtime.distributed import parallel_state
    except Exception:
        return True

    try:
        if parallel_state.model_parallel_is_initialized():
            return int(parallel_state.get_sp_parallel_rank()) == 0
    except Exception as exc:
        logger.warning(
            "Could not determine SP rank for diffusion weight staging; "
            "disabling staging on this process. error=%s",
            exc,
        )
        return False
    return True


def should_stage_weights_on_current_rank(mode: str | None) -> bool:
    return normalize_weight_staging_mode(mode) != "none" and is_weight_staging_rank0()


def maybe_stage_weight_iterator(
    iterator: Iterable[tuple[str, torch.Tensor]],
    *,
    staging_mode: str | None,
    weight_load_profile: DiffusionWeightLoadProfiler | None = None,
    pin_tensor: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Iterable[tuple[str, torch.Tensor]]:
    mode = normalize_weight_staging_mode(staging_mode)
    if mode == "none" or not is_weight_staging_rank0():
        return iterator
    return stage_weight_iterator(
        iterator,
        staging_mode=mode,
        weight_load_profile=weight_load_profile,
        pin_tensor=pin_tensor,
    )


def stage_weight_iterator(
    iterator: Iterable[tuple[str, torch.Tensor]],
    *,
    staging_mode: str,
    weight_load_profile: DiffusionWeightLoadProfiler | None = None,
    pin_tensor: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Iterable[tuple[str, torch.Tensor]]:
    mode = normalize_weight_staging_mode(staging_mode)
    if mode == "none":
        yield from iterator
        return

    if weight_load_profile is not None:
        weight_load_profile.set_staging_requested(mode)
        weight_load_profile.set_staging_effective(
            "pageable" if mode == "pageable" else "pinned"
        )

    pin_fn = pin_tensor or (lambda tensor: tensor.pin_memory())
    pin_disabled = mode == "pageable"
    warned_pin_failure = False

    for name, tensor in iterator:
        staged_tensor = tensor
        if weight_load_profile is not None:
            weight_load_profile.add_staged_tensor(tensor)

        if not pin_disabled:
            start = time.perf_counter()
            try:
                staged_tensor = pin_fn(tensor)
                if not isinstance(staged_tensor, torch.Tensor):
                    raise TypeError(
                        f"pin_memory returned {type(staged_tensor)!r}, expected Tensor"
                    )
                if weight_load_profile is not None:
                    weight_load_profile.add_pinned_tensor(staged_tensor)
            except Exception as exc:
                pin_disabled = True
                staged_tensor = tensor
                error = f"{type(exc).__name__}: {exc}"
                if weight_load_profile is not None:
                    weight_load_profile.set_staging_effective("pageable")
                    weight_load_profile.set_pin_memory_error(error)
                if not warned_pin_failure:
                    logger.warning(
                        "Diffusion weight pinned staging failed; "
                        "falling back to pageable CPU staging. error=%s",
                        error,
                    )
                    warned_pin_failure = True
            finally:
                if weight_load_profile is not None:
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    weight_load_profile.add_ms(WEIGHT_LOAD_PIN_MEMORY_MS, elapsed_ms)

        yield name, staged_tensor
