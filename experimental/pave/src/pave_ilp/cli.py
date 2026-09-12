"""CPU-only command-line entry point for the approved six-case experiment matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .deployment import fill_fragments, make_deployment, validate_deployment
from .flips import build_flip_file, match_target, validate_flip
from .profiles import GENERATORS, ProfileCatalog
from .reports import (
    AUDIT,
    case_name,
    render_case,
    render_commands,
    render_summary,
    render_table3,
    summary_records,
)
from .solver import SolveError, solve_counts
from .templates import SCENARIOS, cluster, compile_templates


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def source_digest() -> str:
    root = Path(__file__).parent
    sha = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        sha.update(path.relative_to(root).as_posix().encode())
        sha.update(path.read_bytes())
    return sha.hexdigest()


def environment() -> dict:
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pave_ilp": __version__,
        "code_sha256": source_digest(),
    }
    for package in ("numpy", "scipy", "pytest"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def run_plan(args: argparse.Namespace) -> Path:
    generators = list(dict.fromkeys(args.generator or GENERATORS))
    scenarios = list(dict.fromkeys(args.scenario or SCENARIOS))
    if args.source_output_tokens >= args.target_output_tokens:
        raise ValueError(
            "This experiment expects source output length strictly below target length"
        )
    if args.long_phase_duration_s is not None and args.long_phase_duration_s <= 0:
        raise ValueError("Long-phase duration must be positive")
    output = args.output_dir.resolve()
    parameters = {
        "generators": generators,
        "scenarios": scenarios,
        "input_tokens": args.input_tokens,
        "source_output_tokens": args.source_output_tokens,
        "target_output_tokens": args.target_output_tokens,
        "kv_cache_tokens": args.kv_cache_tokens,
        "mip_time_limit_s": args.mip_time_limit_s,
        "long_phase_duration_s": args.long_phase_duration_s,
        "fragment_policy": "same_template_first",
    }
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise ValueError(
                "Output directory is not empty; choose a new directory or use --overwrite"
            )
        old_manifest = output / "run_manifest.json"
        if not old_manifest.exists():
            raise ValueError("Only an existing PAVE run may be overwritten")
        old = json.loads(old_manifest.read_text("utf-8"))["parameters"]
        for key in ("generators", "scenarios", "source_output_tokens", "target_output_tokens"):
            if old[key] != parameters[key]:
                raise ValueError("Use a new directory when changing the experiment matrix")
    output.mkdir(parents=True, exist_ok=True)
    catalog = ProfileCatalog.load(args.profile_data)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters,
        "environment": environment(),
        "profile_values_sha256": catalog.snapshot["values_sha256"],
        "completed_cases": [],
        "errors": [],
        "invocation_argv": list(sys.argv),
    }
    write_json(output / "run_manifest.json", manifest)
    write_json(output / "profiles.json", catalog.snapshot)
    cases = []
    for generator in generators:
        for scenario in scenarios:
            name = case_name(generator, scenario)
            print(
                f"[{name}] solving {args.source_output_tokens} -> {args.target_output_tokens}",
                flush=True,
            )
            try:
                deployments = []
                target_templates = None
                for tokens in (args.source_output_tokens, args.target_output_tokens):
                    templates, rejected = compile_templates(
                        catalog,
                        generator,
                        scenario,
                        args.input_tokens,
                        tokens,
                        args.kv_cache_tokens,
                    )
                    placements, solver = solve_counts(
                        cluster(scenario), templates, args.mip_time_limit_s
                    )
                    deployment = make_deployment(
                        catalog,
                        generator,
                        scenario,
                        args.input_tokens,
                        tokens,
                        args.kv_cache_tokens,
                        placements,
                        solver,
                        rejected,
                    )
                    deployment = fill_fragments(deployment, templates, catalog)
                    validate_deployment(deployment, catalog)
                    deployments.append(deployment)
                    target_templates = templates
                source, target = deployments
                target = match_target(
                    source, target, target_templates, catalog, args.mip_time_limit_s
                )
                flip = build_flip_file(
                    source, target, target_templates, catalog, args.mip_time_limit_s
                )
                for deployment in (source, target):
                    write_json(
                        output / "deployments" / f"{name}_{deployment['output_tokens']}.json",
                        deployment,
                    )
                write_json(output / "flips" / f"{name}.json", flip)
                report = output / "reports" / f"{name}.md"
                report.parent.mkdir(exist_ok=True)
                report.write_text(
                    render_case(source, target, flip, args.long_phase_duration_s), encoding="utf-8"
                )
                cases.append((source, target, flip))
                manifest["completed_cases"].append(name)
                print(
                    f"[{name}] source={source['throughput_req_s']:.6f}; no-flip={flip['no_flip']['throughput_req_s']:.6f}; "
                    f"target={target['throughput_req_s']:.6f}; restricted={flip['restricted']['deployment']['throughput_req_s']:.6f}",
                    flush=True,
                )
            except (SolveError, ValueError) as exc:
                error = {"case": name, "error": str(exc)}
                if isinstance(exc, SolveError):
                    error["solver"] = exc.metadata
                manifest["errors"].append(error)
                print(f"[{name}] FAILED: {exc}", file=sys.stderr, flush=True)
            write_json(output / "run_manifest.json", manifest)
    manifest["status"] = "failed" if manifest["errors"] else "complete"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["used_profile_fields"] = sorted(catalog.used_fields)
    write_json(output / "run_manifest.json", manifest)
    write_json(output / "summary.json", summary_records(cases))
    for name, text in (
        ("summary.md", render_summary(cases)),
        ("table3.md", render_table3(cases)),
        ("audit.md", AUDIT),
        ("commands.md", render_commands(manifest)),
    ):
        (output / name).write_text(text, encoding="utf-8")
    if manifest["errors"]:
        raise ValueError(f"{len(manifest['errors'])} cases failed; inspect run_manifest.json")
    validate_output(output)
    return output


def validate_output(output: Path) -> dict:
    manifest = json.loads((output / "run_manifest.json").read_text("utf-8"))
    if manifest["schema_version"] != 1 or manifest["status"] != "complete":
        raise ValueError("Run is not a complete version-1 experiment")
    catalog = ProfileCatalog.load(output / "profiles.json")
    if manifest["profile_values_sha256"] != catalog.snapshot["values_sha256"]:
        raise ValueError("Run manifest/profile fingerprint mismatch")
    parameters = manifest["parameters"]
    expected_case_names = [
        case_name(g, s) for g in parameters["generators"] for s in parameters["scenarios"]
    ]
    if manifest["completed_cases"] != expected_case_names or manifest["errors"]:
        raise ValueError("Manifest case list is incomplete or contains errors")
    cases = []
    for generator in parameters["generators"]:
        for scenario in parameters["scenarios"]:
            name = case_name(generator, scenario)
            deployments = [
                json.loads(
                    (output / "deployments" / f"{name}_{parameters[key]}.json").read_text("utf-8")
                )
                for key in ("source_output_tokens", "target_output_tokens")
            ]
            source, target = deployments
            for deployment, tokens in zip(
                deployments,
                (parameters["source_output_tokens"], parameters["target_output_tokens"]),
            ):
                if (
                    deployment["generator"] != generator
                    or deployment["scenario"] != scenario
                    or deployment["input_tokens"] != parameters["input_tokens"]
                    or deployment["kv_cache_tokens"] != parameters["kv_cache_tokens"]
                    or deployment["output_tokens"] != tokens
                ):
                    raise ValueError("Deployment parameters do not match the run manifest")
                validate_deployment(deployment, catalog)
            flip = json.loads((output / "flips" / f"{name}.json").read_text("utf-8"))
            validate_flip(source, target, flip, catalog)
            if (
                target["throughput_req_s"] + 2e-8
                < flip["restricted"]["deployment"]["throughput_req_s"]
            ):
                raise ValueError("Restricted result exceeds full optimum")
            cases.append((source, target, flip))
            expected_report = render_case(source, target, flip, parameters["long_phase_duration_s"])
            if (output / "reports" / f"{name}.md").read_text("utf-8") != expected_report:
                raise ValueError(f"Report disagrees with structured data: {name}")
    if json.loads((output / "summary.json").read_text("utf-8")) != summary_records(cases):
        raise ValueError("Summary disagrees with deployment-derived capacities")
    if (output / "summary.md").read_text("utf-8") != render_summary(cases):
        raise ValueError("Summary Markdown disagrees with structured data")
    if (output / "table3.md").read_text("utf-8") != render_table3(cases):
        raise ValueError("Table 3 disagrees with structured data")
    return {
        "cases": len(cases),
        "deployments": 2 * len(cases),
        "flip_files": len(cases),
        "status": "validated",
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="PAVE CPU-only offline deployment and static flip planning"
    )
    sub = result.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Solve deployments and both static flip strategies")
    plan.add_argument("--generator", action="append", choices=tuple(GENERATORS))
    plan.add_argument("--scenario", action="append", choices=SCENARIOS)
    plan.add_argument("--profile-data", type=Path)
    plan.add_argument("--input-tokens", type=int, default=128)
    plan.add_argument("--source-output-tokens", type=int, default=512)
    plan.add_argument("--target-output-tokens", type=int, default=2048)
    plan.add_argument("--kv-cache-tokens", type=int, default=4096)
    plan.add_argument("--mip-time-limit-s", type=float, default=120)
    plan.add_argument("--long-phase-duration-s", type=float)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--overwrite", action="store_true")
    validate = sub.add_parser(
        "validate", help="Recompute capacities and verify physical flip actions"
    )
    validate.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        if args.command == "plan":
            print(run_plan(args))
        else:
            print(json.dumps(validate_output(args.output_dir), ensure_ascii=False))
    except (ValueError, OSError, KeyError) as exc:
        print(f"PAVE error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
