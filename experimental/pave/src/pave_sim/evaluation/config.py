"""Approved experiment defaults and compilation into the shared simulator."""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json

from pave_ilp.profiles import GENERATORS
from ..config import RunSpec, Settings

CLUSTERS = ("cluster1", "cluster2", "clustersimu")
POLICIES = ("E", "Static-512", "Static-2048", "E-Least", "E-Weighted", "E-NoOpt", "E-PEOnly")
DEFAULT_RATES = {
    "wan2.2-ti2v-5b": {"cluster1": [6, 7, 8], "cluster2": [5.5, 6.5, 7.5], "clustersimu": [44, 52, 60]},
    "wan2.1-t2v-1.3b": {"cluster1": [12, 14, 16], "cluster2": [10, 12, 14], "clustersimu": [66, 78, 90]},
}
EXPECTED_STATS = {"15": [240, 202, 45], "30": [120, 102, 26], "60": [60, 52, 16]}


@dataclass
class EvaluationConfig:
    ilp_dir: str = "~/Desktop/PAVE_ILP"
    raw_trace: str = "~/Downloads/AzureLMMInferenceTrace_multimodal/AzureLMMInferenceTrace_multimodal.csv"
    hour_start_utc: str = "2024-10-18T06:00:00Z"
    duration_s: float = 3600
    input_tokens: int = 128
    short_output_tokens: int = 512
    long_output_tokens: int = 2048
    rates: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_RATES))
    main_clusters: list = field(default_factory=lambda: ["cluster1", "cluster2"])
    startup_clusters: list = field(default_factory=lambda: ["cluster2", "clustersimu"])
    startup_policies: list = field(default_factory=lambda: ["E", "E-PEOnly", "E-NoOpt"])
    windows_s: list = field(default_factory=lambda: [10, 20, 30, 60])
    margins: list = field(default_factory=lambda: [0, 0.05, 0.10, 0.15])
    monitor_period_s: float = 10
    request_seeds: list = field(default_factory=lambda: list(range(5)))
    scheduler_seeds: list = field(default_factory=lambda: list(range(5)))
    initial_tie: str = "Short"
    max_events: int = 2000000
    synthetic: bool = False
    expected_raw_sha256: str | None = "eeaba4bae383eeb3724a4fc804ab49f160e918b6dbe356250111bc3ab50d4a95"
    expected_p50: float = 98
    expected_stats: dict = field(default_factory=lambda: copy.deepcopy(EXPECTED_STATS))

    @classmethod
    def load(cls, path: Path):
        raw = json.loads(path.read_text("utf-8-sig"))
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown evaluation configuration: {sorted(unknown)}")
        obj = cls(**raw)
        for key in ("ilp_dir", "raw_trace"):
            value = Path(getattr(obj, key)).expanduser()
            setattr(obj, key, str((path.parent / value).resolve() if not value.is_absolute() else value.resolve()))
        obj.validate()
        return obj

    def validate(self):
        if not self.rates or not set(self.rates) <= set(GENERATORS):
            raise ValueError("Unsupported or empty model set")
        for names in (self.main_clusters, self.startup_clusters):
            if not names or len(names) != len(set(names)) or not set(names) <= set(CLUSTERS):
                raise ValueError("Invalid experiment cluster set")
        if not {'cluster1', 'cluster2'} <= set(self.main_clusters):
            raise ValueError("Main experiments must include both cluster candidates")
        if (not self.startup_policies or len(self.startup_policies) != len(set(self.startup_policies))
                or not set(self.startup_policies) <= {'E', 'E-PEOnly', 'E-NoOpt'} or 'E' not in self.startup_policies):
            raise ValueError("Startup comparisons require E and supported unique policies")
        for model, groups in self.rates.items():
            if set(groups) != set(CLUSTERS):
                raise ValueError(f"All three clusters must be configured for {model}")
            for rates in groups.values():
                if not rates or len(rates) != len(set(rates)):
                    raise ValueError("Rates must be nonempty and unique within a case")
                for rate in rates:
                    if isinstance(rate, bool) or not math.isfinite(rate) or rate <= 0:
                        raise ValueError("Rates must be finite and positive")
        if type(self.synthetic) is not bool or type(self.max_events) is not int or self.max_events <= 0:
            raise ValueError("Invalid synthetic flag or event budget")
        for seeds in (self.request_seeds, self.scheduler_seeds):
            if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int for s in seeds):
                raise ValueError("Seeds must be nonempty unique integers")
        self.settings().validate()
        start = datetime.fromisoformat(self.hour_start_utc.replace("Z", "+00:00"))
        if start.tzinfo is None or start.utcoffset().total_seconds() != 0:
            raise ValueError("hour_start_utc must explicitly use UTC")
        if self.duration_s % 60 != 0:
            raise ValueError("Trace duration must contain whole minutes")
        if not math.isfinite(self.expected_p50) or self.expected_p50 < 0:
            raise ValueError("Invalid expected median")
        if self.expected_raw_sha256 is not None and (len(self.expected_raw_sha256) != 64 or
                any(c not in '0123456789abcdef' for c in self.expected_raw_sha256)):
            raise ValueError("Expected trace digest must be a lowercase SHA256")

    def settings(self, ilp_dir=None):
        return Settings(ilp_dir=str(ilp_dir or self.ilp_dir), generators=tuple(self.rates),
            scenarios=CLUSTERS, input_tokens=self.input_tokens, short_output_tokens=self.short_output_tokens,
            long_output_tokens=self.long_output_tokens, duration_s=self.duration_s,
            windows_s=tuple(self.windows_s), margins=tuple(self.margins),
            monitor_period_s=self.monitor_period_s, initial_tie=self.initial_tie)

    def semantic(self):
        data = asdict(self)
        for key in ("ilp_dir", "raw_trace", "expected_raw_sha256", "expected_p50", "expected_stats"):
            data.pop(key)
        return data


def compile_policy(slot: dict, config: EvaluationConfig) -> RunSpec:
    policy = slot["policy"]
    if policy not in POLICIES:
        raise ValueError(f"Unknown evaluation policy {policy}")
    static = policy.startswith("Static-")
    strategy = "A" if static else "D" if policy == "E-NoOpt" else "E"
    scheduler = "least_waiting" if static or policy == "E-Least" else "capacity_weighted" if policy == "E-Weighted" else None
    return RunSpec(slot["generator"], slot["scenario"], strategy, slot["rate_per_min"],
                   30.0 if static else slot["window_s"], 0.0 if static else slot["margin"],
                   slot["scheduler_seed"] if policy == "E-Weighted" else 0,
                   scheduler, "target" if policy == "Static-2048" else "source",
                   "pe_only_optimized" if policy == "E-PEOnly" else None)
