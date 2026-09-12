"""Read-only adaptation of PAVE deployment files and the dominant-interval trace."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
from collections import Counter
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from fractions import Fraction

from pave_ilp.deployment import evaluate, validate_deployment
from pave_ilp.profiles import CAPACITY_MODEL, GENERATORS, ProfileCatalog, digest

from .config import Settings
from .profiles import SimulationProfiles
from .state import Request
from .timing import TICKS_PER_SECOND, rounded_ratio, seconds, ticks


@dataclass(frozen=True)
class Interval:
    start_s: float
    end_s: float
    source: str
    resolved: str


@dataclass(frozen=True)
class Trace:
    intervals: tuple[Interval, ...]
    raw: bytes
    sha256: str

    @classmethod
    def load(cls, settings: Settings) -> Trace:
        raw = (
            Path(settings.trace).expanduser().read_bytes() if settings.trace
            else files("pave_sim").joinpath("data/dominant_intervals.csv").read_bytes()
        )
        rows = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        previous, end, intervals = settings.initial_tie, 0, []
        for row in rows:
            start, stop = float(row["start_minute"]) * 60, float(row["end_minute"]) * 60
            if not all(math.isfinite(v) for v in (start, stop)) or stop <= start:
                raise ValueError("Trace interval has invalid bounds")
            start_tick, stop_tick = ticks(start), ticks(stop)
            if stop_tick <= start_tick:
                raise ValueError("Trace interval is empty at clock resolution")
            if start_tick != end:
                raise ValueError("Trace must be ordered, contiguous, non-overlapping, and start at zero")
            source = row["dominant"].strip().title()
            resolved = previous if source == "Tie" else source
            if resolved not in ("Short", "Long"):
                raise ValueError(f"Unsupported trace label: {source}")
            intervals.append(Interval(seconds(start_tick), seconds(stop_tick), source, resolved))
            previous, end = resolved, stop_tick
        if not intervals or end < ticks(settings.duration_s, positive=True):
            raise ValueError("Trace does not cover duration_s")
        return cls(tuple(intervals), raw, hashlib.sha256(raw).hexdigest())

    def requests(self, settings: Settings, rate: float) -> list[Request]:
        result, interval_index, index = [], 0, 0
        arrival_period = Fraction(60 * TICKS_PER_SECOND) / Fraction(str(rate))
        if arrival_period < 1:
            raise ValueError("Arrival interval must resolve to at least one nanosecond")
        duration = ticks(settings.duration_s, positive=True)
        while True:
            time_tick = rounded_ratio(index * arrival_period.numerator, arrival_period.denominator)
            if time_tick >= duration:
                break
            while time_tick >= ticks(self.intervals[interval_index].end_s):
                interval_index += 1
            interval = self.intervals[interval_index]
            output = settings.short_output_tokens if interval.resolved == "Short" else settings.long_output_tokens
            result.append(Request(index, seconds(time_tick), settings.input_tokens, output, interval.resolved))
            index += 1
        return result


@dataclass(frozen=True)
class Recipe:
    id: str
    hardware: str
    source_template: str
    width: int
    quantity: int
    # Each target contains a validated exemplar and ranks relative to its source group.
    children: tuple[tuple[dict, tuple[int, ...]], ...]

    def replacements(self, source: dict) -> list[dict]:
        result = []
        for exemplar, ranks in self.children:
            target = copy.deepcopy(exemplar)
            target["node"] = source["node"]
            target["gpu_ids"] = [source["gpu_ids"][rank] for rank in ranks]
            target["id"] = (
                f"{target['node']}:gpu-{'_'.join(map(str, target['gpu_ids']))}:{target['template']}"
            )
            result.append(target)
        return result


def template_counts(items: list[dict]) -> Counter:
    return Counter((raw["hardware"], raw["template"]) for raw in items)


@dataclass
class Case:
    id: str
    source: dict
    target: dict
    recipes: tuple[Recipe, ...]
    profiles: SimulationProfiles
    input_files: dict[str, dict]

    @classmethod
    def load(cls, settings: Settings, generator: str, scenario: str) -> Case:
        root = Path(settings.ilp_dir).expanduser().resolve()
        case_id = f"{GENERATORS[generator]}_{scenario}"
        source_path = root / "deployments" / f"{case_id}_{settings.short_output_tokens}.json"
        flip_path = root / "flips" / f"{case_id}.json"
        profile_path = Path(settings.profile_data).expanduser() if settings.profile_data else root / "profiles.json"
        paths = {"source": source_path, "flip": flip_path, "profile": profile_path}
        blobs = {name: path.read_bytes() for name, path in paths.items()}
        source, flip = json.loads(blobs["source"]), json.loads(blobs["flip"])
        catalog = ProfileCatalog.load(profile_path)
        if flip.get("schema_version") != 1 or "restricted" not in flip:
            raise ValueError("Expected a version-1 restricted flip file")
        target = flip["restricted"]["deployment"]
        for deployment, output in ((source, settings.short_output_tokens), (target, settings.long_output_tokens)):
            validate_deployment(deployment, catalog)  # Coefficient validation, no solving.
            if (deployment["generator"], deployment["scenario"], deployment["input_tokens"], deployment["output_tokens"]) != (
                generator, scenario, settings.input_tokens, output
            ):
                raise ValueError("Deployment workload/model/scenario does not match the simulation")
            if deployment.get("denoising_steps_profiled") != 50 or deployment.get("capacity_model") != CAPACITY_MODEL:
                raise ValueError("Simulation requires the approved independent-capacity 50-step profiles")
        if source["nodes"] != target["nodes"] or source["kv_cache_tokens"] != target["kv_cache_tokens"]:
            raise ValueError("Restricted endpoints must use the same nodes and KV pool")
        if source["cpu_te"]["instances"] != target["cpu_te"]["instances"]:
            raise ValueError("CPU replica transitions are not supported")
        if (flip["generator"], flip["scenario"], flip["source_output_tokens"], flip["target_output_tokens"]) != (
            generator, scenario, settings.short_output_tokens, settings.long_output_tokens
        ):
            raise ValueError("Flip metadata does not match its deployments")
        if flip["source_layout_sha256"] != source["layout_sha256"] or flip["profile_values_sha256"] != catalog.snapshot["values_sha256"]:
            raise ValueError("Flip source/profile fingerprint mismatch")
        restricted = flip["restricted"]
        if (restricted["cpu_instances_before"], restricted["cpu_instances_after"]) != (
            source["cpu_te"]["instances"], target["cpu_te"]["instances"]
        ):
            raise ValueError("Restricted CPU metadata mismatch")
        remaining = {i["id"]: copy.deepcopy(i) for i in source["instances"]}
        expected = {i["id"]: i for i in target["instances"]}
        grouped = {}
        for action in restricted["actions"]:
            removed = action["remove_instance_ids"]
            if len(removed) != 1 or removed[0] not in remaining:
                raise ValueError("Restricted action must remove exactly one existing generator instance")
            donor = remaining.pop(removed[0])
            children = action["add_instances"]
            if "PE" in donor["stages"] or not children:
                raise ValueError("Restricted flip may not alter original PE")
            if action["node"] != donor["node"] or action["gpu_ids"] != donor["gpu_ids"]:
                raise ValueError("Restricted source GPU group mismatch")
            used, converted = [], []
            for child in sorted(children, key=lambda c: tuple(c["gpu_ids"])):
                if child != expected.get(child["id"]) or tuple(child["stages"]) != ("PE",):
                    raise ValueError("Restricted target must be a validated target PE instance")
                if child["node"] != donor["node"] or child["hardware"] != donor["hardware"]:
                    raise ValueError("Restricted flip cannot cross nodes/hardware")
                if child["id"] in remaining:
                    raise ValueError("Restricted flip duplicates a retained instance")
                ranks = tuple(donor["gpu_ids"].index(g) for g in child["gpu_ids"])
                used.extend(child["gpu_ids"])
                remaining[child["id"]] = copy.deepcopy(child)
                converted.append((copy.deepcopy(child), ranks))
            if sorted(used) != donor["gpu_ids"]:
                raise ValueError("PE subdivisions must exactly cover the original GPU group once")
            key = (donor["hardware"], donor["template"], donor["bundle_size"], tuple((c["template"], ranks) for c, ranks in converted))
            if key not in grouped:
                grouped[key] = [0, tuple(converted)]
            grouped[key][0] += 1
        reconstructed = copy.deepcopy(target)
        reconstructed.pop("core_instances", None)
        reconstructed["instances"] = list(remaining.values())
        reconstructed = evaluate(reconstructed, catalog, settings.long_output_tokens)
        validate_deployment(reconstructed, catalog)
        if reconstructed["layout_sha256"] != target["layout_sha256"]:
            raise ValueError("Restricted actions do not reproduce their declared endpoint")
        recipes = tuple(
            Recipe(digest(key)[:12], key[0], key[1], key[2], value[0], value[1])
            for key, value in sorted(grouped.items())
        )
        metadata = {
            name: {"path": str(paths[name].resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in blobs.items()
        }
        profiles = SimulationProfiles(catalog, generator, source["kv_cache_tokens"])
        # Validate every required PE grid before event processing, including the
        # sparse input rows that a later migration may query.
        for deployment in (source, target):
            for raw in deployment["instances"]:
                if "PE" in raw["stages"]:
                    profiles.pe(
                        raw["hardware"], raw["bundle_size"],
                        settings.input_tokens, settings.long_output_tokens,
                    )
        return cls(case_id, source, target, recipes, profiles, metadata)

    def slo_baselines(self) -> dict[str, dict]:
        result = {}
        for kind, deployment in (("Short", self.source), ("Long", self.target)):
            stages = {stage: [] for stage in ("PE", "DiT", "VAE")}
            for raw in deployment["instances"]:
                for stage in raw["stages"]:
                    stages[stage].append(self.profiles.latency(raw, stage, deployment["input_tokens"], deployment["output_tokens"]))
            details = {stage: {"instances": len(values), "mean_service_s": math.fsum(values) / len(values)} for stage, values in stages.items()}
            details["TE"] = {"instances": deployment["cpu_te"]["instances"], "mean_service_s": deployment["cpu_te"]["latency_s"]}
            result[kind] = {"latency_s": math.fsum(v["mean_service_s"] for v in details.values()), "stages": details, "reference_layout_sha256": deployment["layout_sha256"]}
        return result
