"""Explicit experiment settings and deduplicated A--E run specifications."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from pave_ilp.profiles import GENERATORS, digest
from pave_ilp.templates import SCENARIOS

SCHEDULERS = ("least_waiting", "estimated_completion", "capacity_weighted")
SOURCE_SELECTIONS = ("min_disruption", "random_feasible")
VARIANTS = {
    "A": (False, "least_waiting", False, False),
    "B": (True, "least_waiting", False, False),
    "C": (True, "estimated_completion", False, False),
    "D": (True, "estimated_completion", True, False),
    "E": (True, "estimated_completion", True, True),
}


def positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class Settings:
    ilp_dir: str = "~/Desktop/PAVE_ILP"
    profile_data: str | None = None
    trace: str | None = None  # None selects the bundled one-hour trace.
    generators: tuple[str, ...] = tuple(GENERATORS)
    scenarios: tuple[str, ...] = tuple(SCENARIOS)
    input_tokens: int = 128
    short_output_tokens: int = 512
    long_output_tokens: int = 2048
    request_rates_per_min: tuple[float, ...] = (4.0, 6.0, 8.0)
    windows_s: tuple[float, ...] = (10.0, 20.0, 30.0, 60.0)
    margins: tuple[float, ...] = (0.05, 0.10, 0.15)
    monitor_period_s: float | None = None
    duration_s: float = 3600.0
    initial_tie: str = "Short"
    seed: int = 0

    def validate(self) -> None:
        if not self.generators or not set(self.generators) <= set(GENERATORS):
            raise ValueError("Select supported Wan generators")
        if not self.scenarios or not set(self.scenarios) <= set(SCENARIOS):
            raise ValueError("Select supported cluster scenarios")
        for key in ("input_tokens", "short_output_tokens", "long_output_tokens"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.short_output_tokens >= self.long_output_tokens:
            raise ValueError("short_output_tokens must be below long_output_tokens")
        for key in ("request_rates_per_min", "windows_s", "margins"):
            values = getattr(self, key)
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{key} must be nonempty with no duplicates")
            for value in values:
                if key == "margins":
                    if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value < 0.5:
                        raise ValueError("margin must be in [0, 0.5)")
                else:
                    positive(value, key)
        if any(m >= 0.5 for m in self.margins):
            raise ValueError("Margin ratios must be below 0.5")
        positive(self.duration_s, "duration_s")
        if self.monitor_period_s is not None:
            positive(self.monitor_period_s, "monitor_period_s")
        if self.initial_tie not in ("Short", "Long"):
            raise ValueError("initial_tie must be Short or Long")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if len(set(self.generators)) != len(self.generators) or len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError("Duplicate generator or scenario")

    @classmethod
    def load(cls, path: Path | None, overrides: dict) -> Settings:
        values = json.loads(path.read_text("utf-8")) if path else {}
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        # Paths supplied in a file are relative to that file, not to the caller's cwd.
        for key in ("ilp_dir", "profile_data", "trace"):
            if path and values.get(key):
                location = Path(values[key]).expanduser()
                values[key] = str(location if location.is_absolute() else (path.parent / location).resolve())
        values.update({k: v for k, v in overrides.items() if v is not None})
        for key in ("generators", "scenarios", "request_rates_per_min", "windows_s", "margins"):
            if key in values:
                if key in ("generators", "scenarios"):
                    values[key] = tuple(values[key])
                else:
                    if any(isinstance(value, bool) for value in values[key]):
                        raise ValueError(f"Boolean value in {key}")
                    values[key] = tuple(float(value) for value in values[key])
        result = cls(**values)
        result.validate()
        return result


@dataclass(frozen=True)
class RunSpec:
    generator: str
    scenario: str
    strategy: str
    rate_per_min: float
    window_s: float
    margin: float
    seed: int
    scheduler_override: str | None = None
    initial_deployment: str = "source"
    startup_mode_override: str | None = None
    source_selection: str = "min_disruption"
    source_selection_seed: int = 0

    @property
    def can_flip(self) -> bool:
        return VARIANTS[self.strategy][0]

    @property
    def scheduler(self) -> str:
        return self.scheduler_override or VARIANTS[self.strategy][1]

    @property
    def optimized(self) -> bool:
        return VARIANTS[self.strategy][3]

    @property
    def startup_mode(self) -> str:
        return self.startup_mode_override or ("all_optimized" if self.optimized else "none_optimized")

    def startup_optimized(self, stages) -> bool:
        stages = set(stages)
        if not stages or not stages <= {"PE", "DiT", "VAE"} or ("PE" in stages and len(stages) != 1):
            raise ValueError("Unsupported startup target modules")
        return self.startup_mode == "all_optimized" or (self.startup_mode == "pe_only_optimized" and stages == {"PE"})

    @property
    def case_id(self) -> str:
        return f"{GENERATORS[self.generator]}_{self.scenario}"

    @property
    def run_id(self) -> str:
        return f"{self.case_id}_{self.strategy}_{digest(asdict(self))[:12]}"

    def validate(self) -> None:
        if self.strategy not in VARIANTS or self.scheduler not in SCHEDULERS:
            raise ValueError("Unknown strategy or scheduler")
        if self.source_selection not in SOURCE_SELECTIONS:
            raise ValueError("Unknown source selection policy")
        if type(self.source_selection_seed) is not int:
            raise ValueError("source_selection_seed must be an integer")
        if self.source_selection != "min_disruption" and not self.can_flip:
            raise ValueError("Source selection override requires flip")
        if self.startup_mode_override not in (None, "all_optimized", "pe_only_optimized", "none_optimized"):
            raise ValueError("Unknown startup mode")
        if self.startup_mode_override is not None and not self.can_flip:
            raise ValueError("Startup override requires flip")
        if self.initial_deployment not in ("source", "target"):
            raise ValueError("Unknown initial deployment")
        if self.initial_deployment == "target" and self.can_flip:
            raise ValueError("Starting at target is supported only for fixed deployments")
        positive(self.rate_per_min, "rate_per_min")
        positive(self.window_s, "window_s")
        if not math.isfinite(self.margin) or not 0 <= self.margin < 0.5:
            raise ValueError("margin must be in [0, 0.5)")
        if not VARIANTS[self.strategy][2] and self.margin != 0:
            raise ValueError("A/B/C must have margin disabled")


def sweep_specs(settings: Settings):
    """99 unique runs per case with the defaults; display slots are handled later."""
    for generator in settings.generators:
        for scenario in settings.scenarios:
            for rate in settings.request_rates_per_min:
                yield RunSpec(generator, scenario, "A", rate, settings.windows_s[0], 0, settings.seed)
                for window in settings.windows_s:
                    for strategy in ("B", "C"):
                        yield RunSpec(generator, scenario, strategy, rate, window, 0, settings.seed)
                    for margin in settings.margins:
                        for strategy in ("D", "E"):
                            yield RunSpec(generator, scenario, strategy, rate, window, margin, settings.seed)
