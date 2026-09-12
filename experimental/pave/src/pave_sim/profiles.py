"""Simulation coefficients, including input-first nearest-neighbor PE lookup."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from pave_ilp.profiles import ProfileCatalog


@dataclass(frozen=True)
class PEMatch:
    hardware: str
    parallelism: int
    actual_input: int
    actual_remaining: int
    matched_input: int
    matched_output: int
    ttft_s: float
    tpot_s: float

    @property
    def duration_s(self) -> float:
        return self.ttft_s + (self.actual_remaining - 1) * self.tpot_s

    def record(self) -> dict:
        return {
            **asdict(self),
            "input_error": self.matched_input - self.actual_input,
            "output_error": self.matched_output - self.actual_remaining,
            "duration_s": self.duration_s,
            "lookup_rule": "nearest_input_then_nearest_output_smaller_key_ties",
        }


class SimulationProfiles:
    def __init__(self, catalog: ProfileCatalog, generator: str, kv_tokens: int):
        self.catalog = catalog
        self.generator = generator
        self.kv_tokens = kv_tokens
        self._checked: set[tuple[str, int]] = set()

    def _validate_pe_grid(self, hardware: str, width: int) -> None:
        key = hardware, width
        if key in self._checked:
            return
        fields = [f"pe7b_{hardware}_nvlink_{kind}_ms" for kind in ("ttft", "tpot")]
        tables = [self.catalog.values.get(field) for field in fields]
        if not all(isinstance(t, dict) and t for t in tables):
            raise ValueError(f"Missing PE profile tables for {hardware}")
        ttft, tpot = tables
        if ttft.keys() != tpot.keys():
            raise ValueError(f"PE input keys differ between TTFT/TPOT for {hardware}")
        for inp, outputs in ttft.items():
            if not isinstance(outputs, dict) or not outputs or not isinstance(tpot[inp], dict):
                raise ValueError(f"Invalid PE output table at {hardware}/{inp}")
            if int(inp) <= 0 or outputs.keys() != tpot[inp].keys():
                raise ValueError(f"PE output keys differ at {hardware}/{inp}")
            for out, ranks in outputs.items():
                if not isinstance(ranks, dict) or not ranks or not isinstance(tpot[inp][out], dict):
                    raise ValueError(f"Invalid PE parallelism table at {hardware}/{inp}/{out}")
                if int(out) <= 0 or ranks.keys() != tpot[inp][out].keys():
                    raise ValueError(f"PE parallelism keys differ at {hardware}/{inp}/{out}")
                for field in fields:
                    value = self.catalog.number(field, inp, out, width)
                    if value <= 0:
                        raise ValueError(f"Nonpositive PE coefficient: {field}/{inp}/{out}/{width}")
        self._checked.add(key)

    def pe(self, hardware: str, width: int, actual_input: int, remaining: int) -> PEMatch:
        if actual_input <= 0 or remaining <= 0:
            raise ValueError("PE lookup requires positive input and remaining output")
        if actual_input + remaining > self.kv_tokens:
            raise ValueError("PE request exceeds the fixed KV pool")
        self._validate_pe_grid(hardware, width)
        prefix = f"pe7b_{hardware}_nvlink"
        table = self.catalog.values[f"{prefix}_ttft_ms"]
        # Input is authoritative. Never reconsider it to improve output proximity.
        inp = min(map(int, table), key=lambda x: (abs(x - actual_input), x))
        out = min(map(int, table[str(inp)]), key=lambda x: (abs(x - remaining), x))
        return PEMatch(
            hardware, width, actual_input, remaining, inp, out,
            self.catalog.number(f"{prefix}_ttft_ms", inp, out, width) / 1000,
            self.catalog.number(f"{prefix}_tpot_ms", inp, out, width) / 1000,
        )

    def latency(self, raw: dict, stage: str, input_tokens: int, output_tokens: int) -> float:
        if stage == "PE":
            return self.pe(raw["hardware"], raw["bundle_size"], input_tokens, output_tokens).duration_s
        if stage == "TE":
            return self.catalog.cpu(self.generator)["latency_s"]
        return self.catalog.stage(
            self.generator, raw["hardware"], stage, raw["bundle_size"],
            input_tokens, output_tokens, self.kv_tokens,
        )["latency_s"]

    def startup(self, raw: dict, optimized: bool) -> float:
        result = self.catalog.startup(
            self.generator, raw["hardware"], raw["bundle_size"], tuple(raw["stages"]) == ("PE",)
        )
        value = result["optimized_s" if optimized else "non_optimized_s"]
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid startup duration")
        return value
