"""Parse literal profiling data without executing the source Python file."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from importlib.resources import files
from pathlib import Path

HARDWARE = ("a100_40g", "a100_80g", "a800", "h100")
GENERATORS = {"wan2.2-ti2v-5b": "wan22_ti2v_5b", "wan2.1-t2v-1.3b": "wan21_t2v_1_3b"}
WIDTHS = (1, 2, 4, 8)
STAGES = ("PE", "DiT", "VAE")
CAPACITY_MODEL = "independent_stage_capacity_no_colocation_interference"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_profile_source(path: Path) -> dict:
    raw = path.read_bytes()
    if path.suffix.lower() == ".json":
        payload = json.loads(raw)
        if payload.get("schema_version") != 1 or "values" not in payload:
            raise ValueError("Expected a version-1 PAVE profile snapshot")
        if digest(payload["values"]) != payload.get("values_sha256"):
            raise ValueError("Profile snapshot checksum mismatch")
        return payload
    if path.suffix.lower() != ".py":
        raise ValueError("Profile source must be a literal .py table or a PAVE .json snapshot")
    values = {}
    for statement in ast.parse(raw.decode("utf-8-sig")).body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue  # Module docstring.
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            raise ValueError("Profile Python files may contain only literal assignments")
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or target.id in values:
            raise ValueError("Profile fields must have unique simple names")
        try:
            value = ast.literal_eval(statement.value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Profile field {target.id} is not a literal") from exc
        if target.id.startswith(("pe7b_", "wan22_ti2v_5b_", "wan21_t2v_1_3b_", "wan22_t2v_1_3b_")):
            values[target.id] = value
    # JSON normalization makes int keys consistent across raw imports and bundled snapshots.
    values = json.loads(json.dumps(values))
    return {
        "schema_version": 1,
        "source_name": path.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "values_sha256": digest(values),
        "values": values,
    }


class ProfileCatalog:
    def __init__(self, snapshot: dict):
        if (
            snapshot.get("schema_version") != 1
            or digest(snapshot["values"]) != snapshot["values_sha256"]
        ):
            raise ValueError("Invalid profile snapshot or checksum")
        self.snapshot = snapshot
        self.values = snapshot["values"]
        self.used_fields: set[str] = set()

    @classmethod
    def load(cls, path: Path | None = None) -> ProfileCatalog:
        if path is not None:
            return cls(read_profile_source(path))
        return cls(json.loads(files("pave_ilp").joinpath("data/profiles.json").read_text("utf-8")))

    def number(self, field: str, *keys: int | str) -> float:
        try:
            value = self.values[field]
            for key in keys:
                value = value[str(key)]
            if isinstance(value, bool):
                raise ValueError("Boolean measurement")
            result = float(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing or invalid profile: {field}{keys}") from exc
        if not math.isfinite(result) or result < 0:
            raise ValueError(f"Nonfinite or negative profile: {field}{keys}")
        self.used_fields.add(field)
        return result

    def stage(
        self,
        generator: str,
        hardware: str,
        stage: str,
        width: int,
        input_tokens: int,
        output_tokens: int,
        kv_cache_tokens: int,
    ) -> dict:
        prefix = GENERATORS[generator]
        fields = []
        if stage == "PE":
            ttft_field = f"pe7b_{hardware}_nvlink_ttft_ms"
            tpot_field = f"pe7b_{hardware}_nvlink_tpot_ms"
            ttft = self.number(ttft_field, input_tokens, output_tokens, width)
            tpot = self.number(tpot_field, input_tokens, output_tokens, width)
            latency = (ttft + (output_tokens - 1) * tpot) / 1000
            memory_hardware = "a100_40g" if hardware == "a100_40g" else "a800"
            memory_field = f"pe7b_{memory_hardware}_nvlink_memory_overhead_gb"
            components = {
                c: self.number(memory_field, c, width) for c in ("weights", "cudagraph", "others")
            }
            components["kv_cache"] = (
                self.number("pe7b_kvcache_per_token_gb", width) * kv_cache_tokens
            )
            fields = [ttft_field, tpot_field, memory_field, "pe7b_kvcache_per_token_gb"]
        else:
            suffix = {"TE": "encoder", "DiT": "denoiser", "VAE": "decoder"}[stage]
            duration_field = f"{prefix}_{hardware}_nvlink_{suffix}_duration_ms"
            memory_field = f"{prefix}_a800_nvlink_{suffix}_memory_overhead_gb"
            latency = self.number(duration_field, width) / 1000
            components = {c: self.number(memory_field, c, width) for c in ("weights", "runtime")}
            fields = [duration_field, memory_field]
        if latency <= 0:
            raise ValueError(f"Nonpositive service time: {generator}/{hardware}/{stage}/{width}")
        return {
            "latency_s": latency,
            "capacity_req_s": 1 / latency,
            "memory_per_gpu_gb": sum(components.values()),
            "memory_components_per_gpu_gb": components,
            "profile_fields": fields,
        }

    def cpu(self, generator: str) -> dict:
        canonical = f"{GENERATORS[generator]}_aws_h100_cpu_encoder_duration_ms"
        field = canonical
        alias = "wan22_t2v_1_3b_aws_h100_cpu_encoder_duration_ms"
        if canonical not in self.values and generator == "wan2.1-t2v-1.3b" and alias in self.values:
            field = alias
        latency = self.number(field) / 1000
        if latency <= 0:
            raise ValueError("CPU TE latency must be positive")
        return {
            "template": "TE_CPU",
            "latency_s": latency,
            "capacity_per_instance_req_s": 1 / latency,
            "profile_field": field,
            "canonical_field": canonical,
            "hardware_reuse": list(HARDWARE),
            "memory_per_instance_gb": None,
            "provisioning_assumption": "Sufficient CPU replicas are available; CPU memory is not profiled.",
        }

    def startup(self, generator: str, hardware: str, width: int, pe: bool) -> dict:
        prefix = "pe7b_a800_nvlink" if pe else f"{GENERATORS[generator]}_{hardware}_nvlink"
        unit = "old_s" if pe else "ms"
        scale = 1 if pe else 1000
        fields = {
            name: f"{prefix}_init_time_{name}_{unit}" for name in ("optimized", "non_optimized")
        }
        return {
            **{f"{name}_s": self.number(field, width) / scale for name, field in fields.items()},
            "profile_fields": fields,
            "measurement_scope": "PE" if pe else "generator_pipeline",
            "used_in_steady_state_objective": False,
        }
