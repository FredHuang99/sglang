"""DDiT latency profile loading and lookup."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


def _placeholder_profile(model_id: str) -> dict[str, Any]:
    if model_id == "wan2.1-t2v-1.3b":
        return {
            "opt_gpus_num": {"144p": 1, "240p": 1, "360p": 2, "720p": 4},
            "dit_step_times": {
                "144p": {"1": 1.0, "2": 0.85, "4": 0.9, "8": 1.05},
                "240p": {"1": 2.0, "2": 1.2, "4": 1.0, "8": 1.1},
                "360p": {"1": 4.0, "2": 2.1, "4": 1.8, "8": 1.9},
                "720p": {"1": 16.0, "2": 8.2, "4": 4.4, "8": 4.8},
            },
        }
    return {
        "opt_gpus_num": {"512p": 2, "768p": 4, "1024p": 4, "144p": 1, "720p": 4},
        "dit_step_times": {
            "144p": {"1": 1.0, "2": 0.9, "4": 1.0, "8": 1.2},
            "512p": {"1": 5.0, "2": 2.8, "4": 2.2, "8": 2.4},
            "720p": {"1": 9.0, "2": 4.8, "4": 3.0, "8": 3.2},
            "768p": {"1": 10.0, "2": 5.2, "4": 3.1, "8": 3.3},
            "1024p": {"1": 16.0, "2": 8.4, "4": 4.9, "8": 5.2},
        },
    }


BUILTIN_PROFILE_MODEL_IDS = ("z-image", "wan2.1-t2v-1.3b")


@dataclass
class DDiTProfile:
    model_id: str
    opt_gpus_num: dict[str, int] = field(default_factory=dict)
    dit_step_times: dict[str, dict[int, float]] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, model_id: str, payload: dict[str, Any]) -> "DDiTProfile":
        opt_gpus_num = {
            str(resolution): int(count)
            for resolution, count in payload.get("opt_gpus_num", {}).items()
        }
        dit_step_times = {
            str(resolution): {int(k): float(v) for k, v in times.items()}
            for resolution, times in payload.get("dit_step_times", {}).items()
        }
        return cls(
            model_id=model_id,
            opt_gpus_num=opt_gpus_num,
            dit_step_times=dit_step_times,
        )

    def opt_gpu_count(self, resolution: str, allowed_gpu_counts: tuple[int, ...]) -> int:
        opt = int(self.opt_gpus_num.get(resolution, 1))
        return max(1, min(opt, max(allowed_gpu_counts)))

    def per_step_time(self, resolution: str, gpu_count: int) -> float:
        profile = self.dit_step_times.get(resolution, {})
        if gpu_count in profile:
            return profile[gpu_count]
        if profile:
            smaller = [k for k in profile if k <= gpu_count]
            if smaller:
                return profile[max(smaller)]
            return profile[min(profile)]
        return 1.0

    def estimate_remaining_time(
        self, request: Any, gpu_count: int, *, current_step: int | None = None
    ) -> float:
        cur_step = request.cur_step if current_step is None else int(current_step)
        remaining_steps = max(0, int(request.total_steps) - int(cur_step))
        return remaining_steps * self.per_step_time(request.resolution, gpu_count)


def resolve_profile_model_id(server_args: Any) -> str:
    explicit = getattr(server_args, "ddit_profile_model_id", None)
    if explicit:
        return str(explicit).lower()
    model_id = getattr(server_args, "model_id", None)
    if model_id:
        return _normalize_model_id(str(model_id))
    model_path = getattr(server_args, "model_path", None)
    if model_path:
        return _normalize_model_id(os.path.basename(str(model_path)))
    return "z-image"


def _normalize_model_id(value: str) -> str:
    normalized = value.lower().replace("_", "-")
    if "wan" in normalized and "2.1" in normalized and "1.3" in normalized:
        return "wan2.1-t2v-1.3b"
    if "z-image" in normalized or "zimage" in normalized:
        return "z-image"
    return normalized


class ProfileStore:
    """Load profile data from JSON or built-in deterministic placeholders."""

    @classmethod
    def load(cls, server_args: Any) -> DDiTProfile:
        model_id = resolve_profile_model_id(server_args)
        profile_path = getattr(server_args, "ddit_profile_path", None)
        if profile_path:
            with open(profile_path, encoding="utf-8") as f:
                payload = json.load(f)
            return cls._from_file_payload(model_id, payload)
        builtin_id = model_id if model_id in BUILTIN_PROFILE_MODEL_IDS else "z-image"
        return DDiTProfile.from_payload(builtin_id, _placeholder_profile(builtin_id))

    @classmethod
    def _from_file_payload(cls, model_id: str, payload: dict[str, Any]) -> DDiTProfile:
        if "models" not in payload:
            return DDiTProfile.from_payload(model_id, payload)
        models = payload.get("models") or {}
        if model_id in models:
            return DDiTProfile.from_payload(model_id, models[model_id])
        normalized = _normalize_model_id(model_id)
        if normalized in models:
            return DDiTProfile.from_payload(normalized, models[normalized])
        if not models:
            return DDiTProfile.from_payload(model_id, {})
        first_model_id = sorted(models)[0]
        return DDiTProfile.from_payload(first_model_id, models[first_model_id])
