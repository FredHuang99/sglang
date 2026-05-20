# SPDX-License-Identifier: Apache-2.0
"""Structured launch-time profiling for diffusion weight loading."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Any

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.profile_log_utils import (
    get_profile_log_context,
)
from sglang.multimodal_gen.runtime.utils.request_profiling import resolve_profile_dir

logger = init_logger(__name__)

WEIGHT_LOAD_DISCOVER_FILES_MS = "weight_load:discover_files_ms"
WEIGHT_LOAD_READ_SAFETENSORS_MS = "weight_load:read_safetensors_ms"
WEIGHT_LOAD_CPU_MATERIALIZE_MS = "weight_load:cpu_materialize_ms"
WEIGHT_LOAD_PIN_MEMORY_MS = "weight_load:pin_memory_ms"
WEIGHT_LOAD_H2D_OR_PARAM_COPY_MS = "weight_load:h2d_or_param_copy_ms"
WEIGHT_LOAD_D2H_OR_OFFLOAD_MS = "weight_load:d2h_or_offload_ms"
WEIGHT_LOAD_NCCL_BROADCAST_MS = "weight_load:nccl_broadcast_ms"
WEIGHT_LOAD_RANK0_WAIT_MS = "weight_load:rank0_wait_ms"
WEIGHT_LOAD_TOTAL_BYTES = "weight_load:total_bytes"
WEIGHT_LOAD_STAGING_REQUESTED = "weight_load:staging_requested"
WEIGHT_LOAD_STAGING_EFFECTIVE = "weight_load:staging_effective"
WEIGHT_LOAD_STAGED_TENSOR_COUNT = "weight_load:staged_tensor_count"
WEIGHT_LOAD_PINNED_TENSOR_COUNT = "weight_load:pinned_tensor_count"
WEIGHT_LOAD_PINNED_BYTES = "weight_load:pinned_bytes"
WEIGHT_LOAD_PIN_MEMORY_ERROR = "weight_load:pin_memory_error"
WEIGHT_LOAD_MODE_REQUESTED = "weight_load:load_mode_requested"
WEIGHT_LOAD_MODE_EFFECTIVE = "weight_load:load_mode_effective"
WEIGHT_LOAD_BROADCAST_TENSOR_COUNT = "weight_load:broadcast_tensor_count"
WEIGHT_LOAD_BROADCAST_BYTES = "weight_load:broadcast_bytes"
WEIGHT_LOAD_BROADCAST_ERROR = "weight_load:broadcast_error"
WEIGHT_LOAD_WARM_POOL_REQUESTED = "weight_load:warm_pool_requested"
WEIGHT_LOAD_WARM_POOL_EFFECTIVE = "weight_load:warm_pool_effective"
WEIGHT_LOAD_WARM_POOL_HIT = "weight_load:warm_pool_hit"
WEIGHT_LOAD_WARM_POOL_STORE_BYTES = "weight_load:warm_pool_store_bytes"
WEIGHT_LOAD_WARM_POOL_ERROR = "weight_load:warm_pool_error"

WEIGHT_LOAD_TIMING_FIELDS = (
    WEIGHT_LOAD_DISCOVER_FILES_MS,
    WEIGHT_LOAD_READ_SAFETENSORS_MS,
    WEIGHT_LOAD_CPU_MATERIALIZE_MS,
    WEIGHT_LOAD_PIN_MEMORY_MS,
    WEIGHT_LOAD_H2D_OR_PARAM_COPY_MS,
    WEIGHT_LOAD_D2H_OR_OFFLOAD_MS,
    WEIGHT_LOAD_NCCL_BROADCAST_MS,
    WEIGHT_LOAD_RANK0_WAIT_MS,
)


def _safe_parallel_rank(fn_name: str) -> str:
    try:
        from sglang.multimodal_gen.runtime.distributed import parallel_state

        if parallel_state.model_parallel_is_initialized():
            return str(getattr(parallel_state, fn_name)())
    except Exception:
        pass
    return "unknown"


def _safe_tensor_nbytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.numel() * tensor.element_size())
    except Exception:
        return 0


def _sanitize_filename_part(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _resolve_launch_weight_profile_dir(server_args: Any | None) -> str:
    if server_args is None:
        return resolve_profile_dir(None, deployment_mode="launch_weight_load")

    cached = getattr(server_args, "_launch_weight_load_profile_dir", None)
    if cached:
        return cached

    profile_dir = resolve_profile_dir(
        getattr(server_args, "profile_output_dir", None),
        getattr(server_args, "profile_run_id", None),
        deployment_mode="launch_weight_load",
    )
    try:
        setattr(server_args, "_launch_weight_load_profile_dir", profile_dir)
    except Exception:
        pass
    return profile_dir


class DiffusionWeightLoadProfiler:
    """Accumulates per-component launch-time weight loading metrics."""

    def __init__(
        self,
        *,
        component: str,
        server_args: Any | None = None,
        enabled: bool = False,
    ) -> None:
        self.component = component
        self.server_args = server_args
        self.enabled = bool(enabled)
        self._timings_ms = {field: 0.0 for field in WEIGHT_LOAD_TIMING_FIELDS}
        self._total_bytes = 0
        self._staging_fields: dict[str, Any] = {
            WEIGHT_LOAD_STAGING_REQUESTED: "none",
            WEIGHT_LOAD_STAGING_EFFECTIVE: "none",
            WEIGHT_LOAD_STAGED_TENSOR_COUNT: 0,
            WEIGHT_LOAD_PINNED_TENSOR_COUNT: 0,
            WEIGHT_LOAD_PINNED_BYTES: 0,
            WEIGHT_LOAD_PIN_MEMORY_ERROR: None,
        }
        self._load_mode_fields: dict[str, Any] = {
            WEIGHT_LOAD_MODE_REQUESTED: "default",
            WEIGHT_LOAD_MODE_EFFECTIVE: "default",
            WEIGHT_LOAD_BROADCAST_TENSOR_COUNT: 0,
            WEIGHT_LOAD_BROADCAST_BYTES: 0,
            WEIGHT_LOAD_BROADCAST_ERROR: None,
        }
        self._warm_pool_fields: dict[str, Any] = {
            WEIGHT_LOAD_WARM_POOL_REQUESTED: "disabled",
            WEIGHT_LOAD_WARM_POOL_EFFECTIVE: "disabled",
            WEIGHT_LOAD_WARM_POOL_HIT: False,
            WEIGHT_LOAD_WARM_POOL_STORE_BYTES: 0,
            WEIGHT_LOAD_WARM_POOL_ERROR: None,
        }
        self._status = "running"
        self._error: str | None = None
        self._finalized = False
        self._started_at_s = time.time()

        ctx = get_profile_log_context(server_args)
        self._context = {
            "role": ctx.role,
            "instance_id": ctx.instance_id,
            "rank": ctx.rank,
            "physical_rank": ctx.physical_rank,
            "world_size": ctx.world_size,
            "device": ctx.device,
            "mem_kind": ctx.mem_kind,
            "sp_rank": _safe_parallel_rank("get_sp_parallel_rank"),
            "tp_rank": _safe_parallel_rank("get_tp_rank"),
        }

    @property
    def active(self) -> bool:
        return self.enabled or logger.isEnabledFor(10)

    @classmethod
    def from_server_args(
        cls, server_args: Any | None, component: str
    ) -> "DiffusionWeightLoadProfiler":
        return cls(
            component=component,
            server_args=server_args,
            enabled=bool(getattr(server_args, "profile_enabled", False)),
        )

    @contextmanager
    def timing_scope(self, field: str) -> Generator[None, None, None]:
        if not self.active:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add_ms(field, (time.perf_counter() - start) * 1000.0)

    def add_ms(self, field: str, value_ms: float) -> None:
        if not self.active:
            return
        if field not in self._timings_ms:
            self._timings_ms[field] = 0.0
        self._timings_ms[field] += max(0.0, float(value_ms))

    def get_ms(self, field: str) -> float:
        return float(self._timings_ms.get(field, 0.0))

    def add_tensor_bytes(self, tensor: torch.Tensor) -> None:
        if not self.active:
            return
        self._total_bytes += _safe_tensor_nbytes(tensor)

    def add_tensor_collection_bytes(self, tensors: Iterable[torch.Tensor]) -> None:
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                self.add_tensor_bytes(tensor)

    def set_staging_requested(self, mode: str) -> None:
        if not self.active:
            return
        self._staging_fields[WEIGHT_LOAD_STAGING_REQUESTED] = mode

    def set_staging_effective(self, mode: str) -> None:
        if not self.active:
            return
        self._staging_fields[WEIGHT_LOAD_STAGING_EFFECTIVE] = mode

    def add_staged_tensor(self, tensor: torch.Tensor) -> None:
        if not self.active:
            return
        self._staging_fields[WEIGHT_LOAD_STAGED_TENSOR_COUNT] += 1

    def add_pinned_tensor(self, tensor: torch.Tensor) -> None:
        if not self.active:
            return
        self._staging_fields[WEIGHT_LOAD_PINNED_TENSOR_COUNT] += 1
        self._staging_fields[WEIGHT_LOAD_PINNED_BYTES] += _safe_tensor_nbytes(tensor)

    def set_pin_memory_error(self, error: str | None) -> None:
        if not self.active:
            return
        self._staging_fields[WEIGHT_LOAD_PIN_MEMORY_ERROR] = error

    def set_load_mode_requested(self, mode: str) -> None:
        if not self.active:
            return
        self._load_mode_fields[WEIGHT_LOAD_MODE_REQUESTED] = mode

    def set_load_mode_effective(self, mode: str) -> None:
        if not self.active:
            return
        self._load_mode_fields[WEIGHT_LOAD_MODE_EFFECTIVE] = mode

    def add_broadcast_tensor(self, tensor: torch.Tensor) -> None:
        if not self.active:
            return
        self._load_mode_fields[WEIGHT_LOAD_BROADCAST_TENSOR_COUNT] += 1
        self._load_mode_fields[WEIGHT_LOAD_BROADCAST_BYTES] += _safe_tensor_nbytes(
            tensor
        )

    def set_broadcast_error(self, error: str | None) -> None:
        if not self.active:
            return
        self._load_mode_fields[WEIGHT_LOAD_BROADCAST_ERROR] = error

    def set_warm_pool_requested(self, mode: str) -> None:
        if not self.active:
            return
        self._warm_pool_fields[WEIGHT_LOAD_WARM_POOL_REQUESTED] = mode

    def set_warm_pool_effective(self, mode: str) -> None:
        if not self.active:
            return
        self._warm_pool_fields[WEIGHT_LOAD_WARM_POOL_EFFECTIVE] = mode

    def set_warm_pool_hit(self, hit: bool) -> None:
        if not self.active:
            return
        self._warm_pool_fields[WEIGHT_LOAD_WARM_POOL_HIT] = bool(hit)

    def set_warm_pool_store_bytes(self, value: int) -> None:
        if not self.active:
            return
        self._warm_pool_fields[WEIGHT_LOAD_WARM_POOL_STORE_BYTES] = int(value)

    def set_warm_pool_error(self, error: str | None) -> None:
        if not self.active:
            return
        self._warm_pool_fields[WEIGHT_LOAD_WARM_POOL_ERROR] = error

    def profile_safetensors_iterator(
        self, iterator: Iterable[tuple[str, torch.Tensor]]
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        if not self.active:
            yield from iterator
            return

        iterator = iter(iterator)
        while True:
            start = time.perf_counter()
            try:
                name, tensor = next(iterator)
            except StopIteration:
                break
            except Exception:
                self.add_ms(
                    WEIGHT_LOAD_READ_SAFETENSORS_MS,
                    (time.perf_counter() - start) * 1000.0,
                )
                raise
            self.add_ms(
                WEIGHT_LOAD_READ_SAFETENSORS_MS,
                (time.perf_counter() - start) * 1000.0,
            )
            self.add_tensor_bytes(tensor)
            yield name, tensor

    def as_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": 1,
            "component": self.component,
            "status": self._status,
            "error": self._error,
            "started_at_s": self._started_at_s,
            **self._context,
            **self._timings_ms,
            WEIGHT_LOAD_TOTAL_BYTES: self._total_bytes,
            **self._staging_fields,
            **self._load_mode_fields,
            **self._warm_pool_fields,
        }
        return record

    def finalize(self, *, status: str = "success", error: str | None = None) -> str | None:
        if self._finalized:
            return None

        self._status = status
        self._error = error
        self._finalized = True
        record = self.as_dict()

        if self.active:
            logger.info("ProfileWeightLoadDone %s", json.dumps(record, sort_keys=True))

        if not self.enabled:
            return None

        profile_dir = _resolve_launch_weight_profile_dir(self.server_args)
        component = _sanitize_filename_part(self.component)
        rank = _sanitize_filename_part(record["rank"])
        physical_rank = _sanitize_filename_part(record["physical_rank"])
        path = os.path.join(
            profile_dir,
            f"weight_load_{component}_rank{rank}_local{physical_rank}_pid{os.getpid()}.json",
        )
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fp:
            json.dump(record, fp, indent=2, sort_keys=True)
            fp.write("\n")
        os.replace(tmp_path, path)
        logger.info("ProfileWeightLoadJSON path=%s", path)
        return path
