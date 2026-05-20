# SPDX-License-Identifier: Apache-2.0
"""Structured launch-time profiling for diffusion module loading."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.profile_log_utils import (
    get_profile_log_context,
)
from sglang.multimodal_gen.runtime.utils.request_profiling import resolve_profile_dir

logger = init_logger(__name__)


def _sanitize_filename_part(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _safe_parallel_rank(fn_name: str) -> str:
    try:
        from sglang.multimodal_gen.runtime.distributed import parallel_state

        if parallel_state.model_parallel_is_initialized():
            return str(getattr(parallel_state, fn_name)())
    except Exception:
        pass
    return "unknown"


def _resolve_launch_module_profile_dir(server_args: Any | None) -> str:
    if server_args is None:
        return resolve_profile_dir(None, deployment_mode="launch_module_load")

    cached = getattr(server_args, "_launch_module_load_profile_dir", None)
    if cached:
        return cached

    profile_dir = resolve_profile_dir(
        getattr(server_args, "profile_output_dir", None),
        getattr(server_args, "profile_run_id", None),
        deployment_mode="launch_module_load",
    )
    try:
        setattr(server_args, "_launch_module_load_profile_dir", profile_dir)
    except Exception:
        pass
    return profile_dir


class DiffusionModuleLoadProfiler:
    """Writes one JSON record for each loaded pipeline module."""

    def __init__(
        self,
        *,
        component: str,
        component_path: str,
        server_args: Any | None,
        enabled: bool,
        available_before_gb: float | None = None,
    ) -> None:
        self.component = component
        self.component_path = component_path
        self.server_args = server_args
        self.enabled = bool(enabled)
        self.available_before_gb = available_before_gb
        self._started_at_s = time.time()
        self._start = time.perf_counter()
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

    @classmethod
    def from_server_args(
        cls,
        server_args: Any | None,
        *,
        component: str,
        component_path: str,
        available_before_gb: float | None = None,
    ) -> "DiffusionModuleLoadProfiler":
        return cls(
            component=component,
            component_path=component_path,
            server_args=server_args,
            enabled=bool(getattr(server_args, "launch_module_profile_enabled", False)),
            available_before_gb=available_before_gb,
        )

    def finalize(
        self,
        *,
        status: str,
        source: str | None,
        component_class: str | None,
        model_size_gb: Any | None,
        available_after_gb: float | None,
        consumed_gb: float | None,
        error: str | None = None,
        fallback: bool = False,
        fallback_reason: str | None = None,
        customized_error_type: str | None = None,
        transformers_or_diffusers: str | None = None,
    ) -> str | None:
        duration_ms = (time.perf_counter() - self._start) * 1000.0
        record: dict[str, Any] = {
            "schema_version": 1,
            "component": self.component,
            "component_path": self.component_path,
            "component_class": component_class,
            "library": transformers_or_diffusers,
            "source": source,
            "fallback": bool(fallback),
            "fallback_source": "native" if fallback else None,
            "fallback_reason": fallback_reason,
            "customized_error_type": customized_error_type,
            "status": status,
            "error": error,
            "started_at_s": self._started_at_s,
            "duration_ms": duration_ms,
            "available_before_gb": self.available_before_gb,
            "available_after_gb": available_after_gb,
            "consumed_gb": consumed_gb,
            "model_size_gb": model_size_gb,
            "dit_cpu_offload": getattr(self.server_args, "dit_cpu_offload", None),
            "text_encoder_cpu_offload": getattr(
                self.server_args, "text_encoder_cpu_offload", None
            ),
            "vae_cpu_offload": getattr(self.server_args, "vae_cpu_offload", None),
            "pin_cpu_memory": getattr(self.server_args, "pin_cpu_memory", None),
            **self._context,
        }

        logger.info("ProfileModuleLoadDoneJSON %s", json.dumps(record, sort_keys=True))
        if not self.enabled:
            return None

        profile_dir = _resolve_launch_module_profile_dir(self.server_args)
        component = _sanitize_filename_part(self.component)
        rank = _sanitize_filename_part(record["rank"])
        physical_rank = _sanitize_filename_part(record["physical_rank"])
        path = os.path.join(
            profile_dir,
            f"module_load_{component}_rank{rank}_local{physical_rank}_pid{os.getpid()}.json",
        )
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fp:
            json.dump(record, fp, indent=2, sort_keys=True)
            fp.write("\n")
        os.replace(tmp_path, path)
        logger.info("ProfileModuleLoadJSON path=%s", path)
        return path
