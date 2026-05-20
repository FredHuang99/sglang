# SPDX-License-Identifier: Apache-2.0
"""In-process pageable CPU warm pool for diffusion checkpoint tensors."""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

DIFFUSION_WEIGHT_WARM_POOL_CHOICES: tuple[str, ...] = ("disabled", "pageable")
DIFFUSION_WEIGHT_WARM_POOL_COMPONENT_CHOICES: tuple[str, ...] = (
    "transformer",
    "vae",
)


@dataclass(frozen=True)
class WeightWarmPoolKey:
    component: str
    model_path: str
    file_fingerprint: tuple[tuple[str, int, int], ...]
    dtype: str
    component_class: str


@dataclass
class WeightWarmPoolEntry:
    tensors: dict[str, torch.Tensor]
    bytes: int
    created_at_s: float
    reverse_param_names_mapping: dict[str, Any] | None = None


class WeightWarmPool:
    """A tiny LRU cache for materialized CPU tensors within one process."""

    def __init__(self) -> None:
        self._entries: OrderedDict[WeightWarmPoolKey, WeightWarmPoolEntry] = (
            OrderedDict()
        )
        self._total_bytes = 0

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def clear(self) -> None:
        self._entries.clear()
        self._total_bytes = 0

    def get(self, key: WeightWarmPoolKey) -> WeightWarmPoolEntry | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry

    def put(
        self,
        key: WeightWarmPoolKey,
        entry: WeightWarmPoolEntry,
        *,
        max_bytes: int = 0,
    ) -> bool:
        if max_bytes > 0 and entry.bytes > max_bytes:
            logger.warning(
                "Skipping diffusion warm-pool store for component=%s because "
                "entry size %.2f GiB exceeds max %.2f GiB.",
                key.component,
                entry.bytes / (1024**3),
                max_bytes / (1024**3),
            )
            return False

        old = self._entries.pop(key, None)
        if old is not None:
            self._total_bytes -= old.bytes

        while max_bytes > 0 and self._total_bytes + entry.bytes > max_bytes:
            _old_key, old_entry = self._entries.popitem(last=False)
            self._total_bytes -= old_entry.bytes

        self._entries[key] = entry
        self._total_bytes += entry.bytes
        return True


_GLOBAL_WEIGHT_WARM_POOL = WeightWarmPool()


def get_global_weight_warm_pool() -> WeightWarmPool:
    return _GLOBAL_WEIGHT_WARM_POOL


def normalize_weight_warm_pool_mode(mode: str | None) -> str:
    normalized = "disabled" if mode is None else str(mode).lower()
    if normalized not in DIFFUSION_WEIGHT_WARM_POOL_CHOICES:
        raise ValueError(
            "Invalid diffusion weight warm-pool mode: "
            f"{mode}. Must be one of {DIFFUSION_WEIGHT_WARM_POOL_CHOICES}."
        )
    return normalized


def normalize_weight_warm_pool_components(
    value: str | Iterable[str] | None,
) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.replace(";", ",").split(",")
    else:
        raw_items = list(value)
    return [str(item).strip().lower() for item in raw_items if str(item).strip()]


def is_weight_warm_pool_enabled(
    *,
    mode: str | None,
    components: str | Iterable[str] | None,
    component: str | None,
) -> bool:
    normalized_mode = normalize_weight_warm_pool_mode(mode)
    if normalized_mode == "disabled":
        return False
    normalized_component = (component or "").lower()
    return normalized_component in normalize_weight_warm_pool_components(components)


def warm_pool_max_bytes(max_gb: float | int | None) -> int:
    if max_gb is None:
        return 0
    value = float(max_gb)
    if value <= 0:
        return 0
    return int(value * (1024**3))


def tensor_mapping_nbytes(tensors: Mapping[str, torch.Tensor]) -> int:
    total = 0
    for tensor in tensors.values():
        if isinstance(tensor, torch.Tensor):
            total += int(tensor.numel() * tensor.element_size())
    return total


def make_weight_warm_pool_entry(
    tensors: Mapping[str, torch.Tensor],
    *,
    reverse_param_names_mapping: Mapping[str, Any] | None = None,
) -> WeightWarmPoolEntry:
    stored_tensors = dict(tensors)
    return WeightWarmPoolEntry(
        tensors=stored_tensors,
        bytes=tensor_mapping_nbytes(stored_tensors),
        created_at_s=time.time(),
        reverse_param_names_mapping=(
            dict(reverse_param_names_mapping)
            if reverse_param_names_mapping is not None
            else None
        ),
    )


def build_weight_warm_pool_key(
    *,
    component: str,
    model_path: str,
    safetensors_files: Iterable[str],
    dtype: torch.dtype | str | None,
    component_class: str,
) -> WeightWarmPoolKey:
    fingerprint: list[tuple[str, int, int]] = []
    for path in sorted(str(item) for item in safetensors_files):
        stat = os.stat(path)
        mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))
        fingerprint.append((os.path.abspath(path), int(stat.st_size), int(mtime_ns)))

    return WeightWarmPoolKey(
        component=component.lower(),
        model_path=os.path.abspath(model_path),
        file_fingerprint=tuple(fingerprint),
        dtype=str(dtype) if dtype is not None else "none",
        component_class=component_class,
    )


def get_weight_warm_pool_entry(
    key: WeightWarmPoolKey,
) -> WeightWarmPoolEntry | None:
    return get_global_weight_warm_pool().get(key)


def put_weight_warm_pool_entry(
    key: WeightWarmPoolKey,
    entry: WeightWarmPoolEntry,
    *,
    max_gb: float | int | None = 0,
) -> bool:
    return get_global_weight_warm_pool().put(
        key,
        entry,
        max_bytes=warm_pool_max_bytes(max_gb),
    )
