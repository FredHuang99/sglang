"""Explicit foreground entry points; run/sweep are never invoked during import."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from pave_ilp.profiles import digest

from . import __version__
from .config import SCHEDULERS, RunSpec, Settings, sweep_specs
from .engine import Simulator
from .inputs import Case, Trace
from .metrics import write_run
from .records import Journal, write_json
from .reports import generate_reports
from .timing import SEMANTICS


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_hash() -> str:
    accumulator = hashlib.sha256()
    root = Path(__file__).resolve().parent.parent
    for package in ("pave_ilp", "pave_sim"):
        for path in sorted((root / package).rglob("*.py")):
            accumulator.update(path.relative_to(root).as_posix().encode())
            accumulator.update(path.read_bytes())
    return accumulator.hexdigest()


def new_output(path: Path, *, create: bool = True) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Output directory must be new or empty; existing results are never overwritten")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def archive_inputs(output: Path, cases: dict[tuple[str, str], Case], trace: Trace) -> dict:
    archived = output / "inputs" / "ilp"
    (archived / "deployments").mkdir(parents=True)
    (archived / "flips").mkdir()
    snapshots = {case.profiles.catalog.snapshot["values_sha256"] for case in cases.values()}
    if len(snapshots) != 1:
        raise ValueError("A run collection must use one consistent profile snapshot")
    first = next(iter(cases.values()))
    write_json(archived / "profiles.json", first.profiles.catalog.snapshot)
    profile_original = Path(first.input_files["profile"]["path"])
    shutil.copyfile(profile_original, output / "inputs" / f"profile_source{profile_original.suffix}")
    (output / "inputs" / "dominant_intervals.csv").write_bytes(trace.raw)
    sources = {}
    for case in cases.values():
        source_name = f"{case.id}_{case.source['output_tokens']}.json"
        destinations = {"source": archived / "deployments" / source_name, "flip": archived / "flips" / f"{case.id}.json"}
        sources[case.id] = {}
        for name, destination in destinations.items():
            shutil.copyfile(case.input_files[name]["path"], destination)
            sources[case.id][name] = {**case.input_files[name], "archived_path": destination.relative_to(output).as_posix()}
    return {"cases": sources, "trace_sha256": trace.sha256, "profile_values_sha256": next(iter(snapshots)), "replay_ilp_dir": "inputs/ilp", "replay_trace": "inputs/dominant_intervals.csv"}


def execute(settings: Settings, specs: list[RunSpec], destination: Path) -> dict:
    # These validations parse data and compute coefficients; they never invoke MILP.
    settings.validate()
    for spec in specs:
        spec.validate()
    trace = Trace.load(settings)
    cases = {(spec.generator, spec.scenario): None for spec in specs}
    for generator, scenario in cases:
        cases[generator, scenario] = Case.load(settings, generator, scenario)
    output = new_output(destination)
    inputs = archive_inputs(output, cases, trace)
    (output / "runs").mkdir()
    manifest = {
        "schema_version": 1, "kind": "simulation_collection", "status": "running",
        "started_utc": timestamp(), "configuration": asdict(settings),
        "configuration_sha256": digest(asdict(settings)), "inputs": inputs,
        "simulation_semantics": SEMANTICS.copy(),
        "environment": {"pave_sim": __version__, "python": platform.python_version(), "platform": platform.platform(), "source_sha256": source_hash()},
        "planned_runs": [{"run_id": spec.run_id, **asdict(spec)} for spec in specs],
        "completed_run_ids": [], "failed_runs": [],
    }
    write_json(output / "manifest.json", manifest)
    try:
        for spec in specs:
            directory = output / "runs" / spec.run_id
            directory.mkdir()
            run_metadata = {"configuration": asdict(settings), "input_case": inputs["cases"][spec.case_id], "profile_values_sha256": inputs["profile_values_sha256"], "trace_sha256": trace.sha256, "environment": manifest["environment"], "simulation_semantics": manifest["simulation_semantics"]}
            write_json(directory / "run.json", {"schema_version": 1, "run_id": spec.run_id, "status": "running", "spec": asdict(spec), **run_metadata})
            journal = Journal(directory / "events.jsonl", spec.run_id)
            engine = None
            try:
                engine = Simulator(settings, spec, cases[spec.generator, spec.scenario], trace, journal)
                result = engine.run()
                write_run(directory, result, run_metadata)
                manifest["completed_run_ids"].append(spec.run_id)
            except BaseException as error:
                failure = {"run_id": spec.run_id, "error": str(error), "error_type": type(error).__name__}
                manifest["failed_runs"].append(failure)
                status = "failed" if isinstance(error, Exception) else "interrupted"
                write_json(directory / "run.json", {"schema_version": 1, "status": status, "spec": asdict(spec), **failure, **run_metadata})
                write_json(directory / "diagnostic.json", engine.diagnostic() if engine else failure)
                if not isinstance(error, Exception):
                    raise
            finally:
                journal.close()
            write_json(output / "manifest.json", manifest)
        if manifest["completed_run_ids"]:
            generate_reports(output, output / "reports", asdict(settings))
        manifest["status"] = "failed" if manifest["failed_runs"] else "complete"
    finally:
        if manifest["status"] == "running":
            manifest["status"] = "interrupted"
        manifest["finished_utc"] = timestamp()
        write_json(output / "manifest.json", manifest)
    return {"status": manifest["status"], "output_dir": str(output), "completed_runs": len(manifest["completed_run_ids"]), "failed_runs": len(manifest["failed_runs"])}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="pave-sim", description="Profile-driven CPU simulation using restricted PAVE flips")
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("evaluation", help="Prepare and execute the versioned evaluation campaign")
    for command in ("run", "sweep"):
        child = sub.add_parser(command)
        child.add_argument("--config", type=Path)
        child.add_argument("--ilp-dir")
        child.add_argument("--profile-data")
        child.add_argument("--trace")
        child.add_argument("--generator", dest="generators", action="append")
        child.add_argument("--scenario", dest="scenarios", action="append")
        child.add_argument("--input-tokens", type=int)
        child.add_argument("--short-output-tokens", type=int)
        child.add_argument("--long-output-tokens", type=int)
        child.add_argument("--duration-s", type=float)
        child.add_argument("--monitor-period-s", type=float)
        child.add_argument("--initial-tie", choices=("Short", "Long"))
        child.add_argument("--seed", type=int)
        child.add_argument("--output-dir", required=True, type=Path)
        if command == "run":
            child.add_argument("--strategy", choices=tuple("ABCDE"), default="C")
            child.add_argument("--rate-per-min", type=float)
            child.add_argument("--window-s", type=float)
            child.add_argument("--margin", type=float)
            child.add_argument("--scheduler", choices=SCHEDULERS)
        else:
            child.add_argument("--request-rates-per-min", nargs="+", type=float)
            child.add_argument("--windows-s", nargs="+", type=float)
            child.add_argument("--margins", nargs="+", type=float)
    report = sub.add_parser("report", help="Recompute reports from existing records; never simulate")
    report.add_argument("--input-dir", required=True, type=Path)
    report.add_argument("--output-dir", required=True, type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    actual = list(sys.argv[1:] if argv is None else argv)
    if actual and actual[0] == "evaluation":
        from .evaluation.cli import main as evaluation_main
        return evaluation_main(actual[1:])
    args = parser().parse_args(argv)
    try:
        if args.command == "report":
            root = args.input_dir.expanduser().resolve()
            manifest = json.loads((root / "manifest.json").read_text("utf-8"))
            if manifest.get("schema_version") != 1 or manifest.get("kind") != "simulation_collection":
                raise ValueError("Expected a version-1 simulation collection manifest")
            output = new_output(args.output_dir, create=False)
            report = generate_reports(root, output, manifest["configuration"])
            print(json.dumps({"output_dir": str(output), **report}, ensure_ascii=False))
            return 0
        supplied = vars(args)
        config_keys = set(Settings.__dataclass_fields__)
        settings = Settings.load(args.config, {key: value for key, value in supplied.items() if key in config_keys})
        if args.command == "run":
            if len(settings.generators) != 1 or len(settings.scenarios) != 1:
                raise ValueError("run requires one --generator and one --scenario (or a single-case config)")
            rate = args.rate_per_min if args.rate_per_min is not None else settings.request_rates_per_min[0]
            window = args.window_s if args.window_s is not None else settings.windows_s[0]
            margin = args.margin if args.margin is not None else settings.margins[0] if args.strategy in ("D", "E") else 0.0
            spec = RunSpec(settings.generators[0], settings.scenarios[0], args.strategy, rate, window, margin, settings.seed, args.scheduler)
            # Report slots must reflect the actual single-run parameters as well.
            configuration = asdict(settings)
            configuration.update(request_rates_per_min=(rate,), windows_s=(window,))
            if margin > 0:
                configuration["margins"] = (margin,)
            settings = Settings(**configuration)
            specs = [spec]
        else:
            specs = list(sweep_specs(settings))
        outcome = execute(settings, specs, args.output_dir)
        print(json.dumps(outcome, ensure_ascii=False))
        return 0 if outcome["status"] == "complete" else 1
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f"pave-sim: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
