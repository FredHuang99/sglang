#!/usr/bin/env python3
"""Profile PE7B request execution time for TP/input/output matrices."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import profile_server_launch_time as launch_time

DEFAULT_GPU_NUMS = [1, 2, 4, 8]
DEFAULT_INPUT_LENS = [
    128,
    256,
    384,
    512,
    640,
    768,
    896,
    1024,
    1152,
    1280,
    1408,
    1536,
    1664,
    1792,
    1920,
    2048,
]
DEFAULT_OUTPUT_LENS = [
    [384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1920, 2048],
    [256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1792, 1920],
    [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1664, 1792],
    [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1536, 1664],
    [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1408, 1536],
    [128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1408],
    [128, 256, 384, 512, 640, 768, 896, 1152, 1280],
    [128, 256, 384, 512, 640, 768, 1024, 1152],
    [128, 256, 384, 512, 640, 896, 1024],
    [128, 256, 384, 512, 768, 896],
    [128, 256, 384, 640, 768],
    [128, 256, 512, 640],
    [128, 384, 512],
    [256, 384],
    [128, 256],
    [128],
]
DEFAULT_FLAT_INPUT_LENS = [128]
DEFAULT_FLAT_OUTPUT_LENS = DEFAULT_OUTPUT_LENS[0]
DEFAULT_NUM_RUNS = 5
DEFAULT_NUM_WARMUP_RUNS = 2
PE7B_PRESET_KEY = "promptenhancer-7b"
PE7B_SETUP = launch_time.LLM_SETUPS["constrained_4096"]

CSV_COLUMNS = [
    "row_name",
    "status",
    "tp_size",
    "input_length",
    "output_length",
    "ttft_ms",
    "tpot_ms",
]

DETAIL_COLUMNS = [
    "row_name",
    "status",
    "reason",
    "tp_size",
    "input_length",
    "output_length",
    "num_runs",
    "num_warmup_runs",
    "measured_runs",
    "successful_measured_runs",
    "run_statuses",
    "ttft_ms",
    "tpot_ms",
    "case_dir",
    "server_log_path",
    "server_command_json",
    "run_details_json",
]


@dataclass
class BenchRun:
    run_index: int
    is_warmup: bool
    status: str
    reason: str
    returncode: int | None
    elapsed_s: float | None
    ttft_ms: float | None
    tpot_ms: float | None
    bench_jsonl_path: str
    client_log_path: str
    command: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index,
            "is_warmup": self.is_warmup,
            "status": self.status,
            "reason": self.reason,
            "returncode": self.returncode,
            "elapsed_s": self.elapsed_s,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "bench_jsonl_path": self.bench_jsonl_path,
            "client_log_path": self.client_log_path,
            "command": self.command,
            "command_text": format_command(self.command),
        }


@dataclass
class CaseResult:
    row_name: str
    status: str
    reason: str
    tp_size: int
    input_length: int
    output_length: int
    ttft_ms: float | None
    tpot_ms: float | None
    case_dir: str
    server_log_path: str
    server_command: list[str]
    run_results: list[BenchRun]

    @property
    def measured_runs(self) -> list[BenchRun]:
        return [run for run in self.run_results if not run.is_warmup]

    def to_csv_row(self) -> dict[str, str]:
        return {
            "row_name": self.row_name,
            "status": self.status,
            "tp_size": str(self.tp_size),
            "input_length": str(self.input_length),
            "output_length": str(self.output_length),
            "ttft_ms": format_number(self.ttft_ms),
            "tpot_ms": format_number(self.tpot_ms),
        }

    def to_detail_row(self) -> dict[str, str]:
        measured = self.measured_runs
        successful_measured_runs = sum(run.status == "ok" for run in measured)
        return {
            "row_name": self.row_name,
            "status": self.status,
            "reason": self.reason,
            "tp_size": str(self.tp_size),
            "input_length": str(self.input_length),
            "output_length": str(self.output_length),
            "num_runs": str(len(self.run_results)),
            "num_warmup_runs": str(len(self.run_results) - len(measured)),
            "measured_runs": str(len(measured)),
            "successful_measured_runs": str(successful_measured_runs),
            "run_statuses": "|".join(run.status for run in self.run_results),
            "ttft_ms": format_number(self.ttft_ms),
            "tpot_ms": format_number(self.tpot_ms),
            "case_dir": self.case_dir,
            "server_log_path": self.server_log_path,
            "server_command_json": json.dumps(
                self.server_command, ensure_ascii=False
            ),
            "run_details_json": json.dumps(
                [run.to_json() for run in self.run_results], ensure_ascii=False
            ),
        }


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def format_command(command: Sequence[str]) -> str:
    import shlex

    return shlex.join(str(part) for part in command)


def format_number(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value:.6f}"


def average(values: Sequence[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def parse_int_list(raw_values: list[str] | None, *, default: list[int]) -> list[int]:
    if not raw_values:
        return list(default)
    tokens: list[str] = []
    for raw in raw_values:
        tokens.extend(part for part in raw.replace(",", " ").split() if part)
    values: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        value = int(token)
        if value <= 0:
            raise ValueError(f"Value must be positive, got {value}.")
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def default_io_cases() -> list[tuple[int, int]]:
    if len(DEFAULT_INPUT_LENS) != len(DEFAULT_OUTPUT_LENS):
        raise ValueError(
            "DEFAULT_INPUT_LENS and DEFAULT_OUTPUT_LENS must have the same length."
        )
    return [
        (input_len, output_len)
        for input_len, output_lens in zip(DEFAULT_INPUT_LENS, DEFAULT_OUTPUT_LENS)
        for output_len in output_lens
    ]


def parse_io_matrix(raw_values: list[str] | None) -> list[tuple[int, int]]:
    if not raw_values:
        return []
    tokens: list[str] = []
    for raw in raw_values:
        tokens.extend(part for part in raw.split() if part)

    cases: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for token in tokens:
        if ":" not in token:
            raise ValueError(
                f"Invalid --io-matrix item {token!r}; expected INPUT:OUT1,OUT2."
            )
        input_token, output_token = token.split(":", 1)
        input_len = int(input_token)
        if input_len <= 0:
            raise ValueError(f"Input length must be positive, got {input_len}.")
        output_lens = parse_int_list([output_token], default=[])
        if not output_lens:
            raise ValueError(f"No output lengths found in --io-matrix item {token!r}.")
        for output_len in output_lens:
            pair = (input_len, output_len)
            if pair not in seen:
                seen.add(pair)
                cases.append(pair)
    return cases


def resolve_io_cases(
    *,
    input_lens: list[str] | None,
    output_lens: list[str] | None,
    io_matrix: list[str] | None,
) -> list[tuple[int, int]]:
    if io_matrix:
        if input_lens or output_lens:
            raise ValueError(
                "--io-matrix cannot be combined with --input-lens or --output-lens."
            )
        return parse_io_matrix(io_matrix)

    if not input_lens and not output_lens:
        return default_io_cases()

    resolved_input_lens = parse_int_list(
        input_lens, default=DEFAULT_FLAT_INPUT_LENS
    )
    resolved_output_lens = parse_int_list(
        output_lens, default=DEFAULT_FLAT_OUTPUT_LENS
    )
    return [
        (input_len, output_len)
        for input_len in resolved_input_lens
        for output_len in resolved_output_lens
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile PromptEnhancer-7B execution TTFT/TPOT for TP/input/output "
            "matrices. Each case runs 5 requests by default: 2 warmup and 3 measured."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gpu-nums", nargs="*", default=None)
    parser.add_argument("--input-lens", nargs="*", default=None)
    parser.add_argument("--output-lens", nargs="*", default=None)
    parser.add_argument(
        "--io-matrix",
        nargs="*",
        default=None,
        help=(
            "Structured input/output matrix, e.g. "
            "'128:384,512,640' '256:256,384'. Cannot be combined with "
            "--input-lens or --output-lens."
        ),
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to /workspace/outputs/pe7b_execution_time/<stamp>.",
    )
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--keep-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep per-case server/client logs and JSONL artifacts.",
    )
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument("--num-warmup-runs", type=int, default=DEFAULT_NUM_WARMUP_RUNS)
    args = parser.parse_args(argv)

    args.gpu_nums = parse_int_list(args.gpu_nums, default=DEFAULT_GPU_NUMS)
    try:
        args.io_cases = resolve_io_cases(
            input_lens=args.input_lens,
            output_lens=args.output_lens,
            io_matrix=args.io_matrix,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.timeout_s <= 0:
        raise SystemExit("--timeout-s must be positive")
    if args.num_runs <= 0:
        raise SystemExit("--num-runs must be positive")
    if args.num_warmup_runs < 0:
        raise SystemExit("--num-warmup-runs must be >= 0")
    if args.num_warmup_runs >= args.num_runs:
        raise SystemExit("--num-warmup-runs must be smaller than --num-runs")
    return args


def get_preset(model_path: str | None = None) -> launch_time.LaunchPreset:
    preset = launch_time.PRESETS[PE7B_PRESET_KEY]
    if model_path:
        preset = replace(preset, model_path=model_path)
    return preset


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    return Path(f"/workspace/outputs/pe7b_execution_time/{now_stamp()}").resolve()


def build_server_command(
    *,
    preset: launch_time.LaunchPreset,
    tp_size: int,
    host: str,
    port: int,
) -> list[str]:
    return launch_time.build_promptenhancer_command(
        preset=preset,
        tp_size=tp_size,
        host=host,
        port=port,
        llm_setup=PE7B_SETUP,
    )


def build_bench_command(
    *,
    preset: launch_time.LaunchPreset,
    host: str,
    port: int,
    input_len: int,
    output_len: int,
    output_file: Path,
    run_tag: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        preset.model_path,
        "--served-model-name",
        preset.model_id,
        "--tokenizer",
        preset.model_path,
        "--dataset-name",
        "random-ids",
        "--num-prompts",
        "1",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--random-range-ratio",
        "1.0",
        "--request-rate",
        "inf",
        "--max-concurrency",
        "1",
        "--ready-check-timeout-sec",
        "0",
        "--warmup-requests",
        "0",
        "--disable-tqdm",
        "--seed",
        "42",
        "--tag",
        run_tag,
        "--output-file",
        str(output_file),
    ]


def parse_bench_jsonl(path: Path) -> tuple[float | None, float | None, str]:
    if not path.exists():
        return None, None, "bench output JSONL was not created"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        return None, None, "bench output JSONL is empty"
    try:
        row = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return None, None, f"could not parse bench JSONL: {exc}"

    ttft = row.get("mean_ttft_ms")
    tpot = row.get("mean_tpot_ms")
    reason_parts = []
    if not isinstance(ttft, (int, float)):
        reason_parts.append("mean_ttft_ms missing")
        ttft = None
    if not isinstance(tpot, (int, float)):
        reason_parts.append("mean_tpot_ms missing")
        tpot = None
    return (
        float(ttft) if ttft is not None else None,
        float(tpot) if tpot is not None else None,
        "; ".join(reason_parts),
    )


def read_tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def run_bench_once(
    *,
    preset: launch_time.LaunchPreset,
    host: str,
    port: int,
    input_len: int,
    output_len: int,
    run_dir: Path,
    run_index: int,
    is_warmup: bool,
    timeout_s: int,
    dry_run: bool,
) -> BenchRun:
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "bench_serving.jsonl"
    client_log_path = run_dir / "client.log"
    run_tag = (
        f"tp_bench,isl={input_len},osl={output_len},"
        f"run={run_index},warmup={is_warmup}"
    )
    command = build_bench_command(
        preset=preset,
        host=host,
        port=port,
        input_len=input_len,
        output_len=output_len,
        output_file=jsonl_path,
        run_tag=run_tag,
    )
    (run_dir / "command.txt").write_text(format_command(command), encoding="utf-8")

    if dry_run:
        return BenchRun(
            run_index=run_index,
            is_warmup=is_warmup,
            status="dry_run",
            reason="",
            returncode=None,
            elapsed_s=None,
            ttft_ms=None,
            tpot_ms=None,
            bench_jsonl_path=str(jsonl_path),
            client_log_path=str(client_log_path),
            command=command,
        )

    start = time.perf_counter()
    with client_log_path.open("w", encoding="utf-8") as log_fh:
        try:
            result = subprocess.run(
                command,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            elapsed_s = time.perf_counter() - start
        except subprocess.TimeoutExpired:
            elapsed_s = time.perf_counter() - start
            return BenchRun(
                run_index=run_index,
                is_warmup=is_warmup,
                status="timeout",
                reason=read_tail(client_log_path),
                returncode=None,
                elapsed_s=elapsed_s,
                ttft_ms=None,
                tpot_ms=None,
                bench_jsonl_path=str(jsonl_path),
                client_log_path=str(client_log_path),
                command=command,
            )

    ttft_ms, tpot_ms, parse_reason = parse_bench_jsonl(jsonl_path)
    if result.returncode != 0:
        status = "failed"
        reason = read_tail(client_log_path) or parse_reason
    elif ttft_ms is None or tpot_ms is None:
        status = "bad_result"
        reason = parse_reason
    else:
        status = "ok"
        reason = ""

    return BenchRun(
        run_index=run_index,
        is_warmup=is_warmup,
        status=status,
        reason=reason,
        returncode=result.returncode,
        elapsed_s=elapsed_s,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        bench_jsonl_path=str(jsonl_path),
        client_log_path=str(client_log_path),
        command=command,
    )


def aggregate_case(
    *,
    tp_size: int,
    input_len: int,
    output_len: int,
    case_dir: Path,
    server_log_path: Path,
    server_command: list[str],
    run_results: list[BenchRun],
) -> CaseResult:
    measured = [run for run in run_results if not run.is_warmup]
    ok_measured = [run for run in measured if run.status == "ok"]
    ttft_ms = average([run.ttft_ms for run in ok_measured])
    tpot_ms = average([run.tpot_ms for run in ok_measured])

    if all(run.status == "dry_run" for run in run_results):
        status = "dry_run"
        reason = ""
    elif measured and len(ok_measured) == len(measured):
        status = "ok"
        reason = ""
    elif ttft_ms is not None or tpot_ms is not None:
        status = "partial"
        reason = "; ".join(run.reason for run in measured if run.reason)
    else:
        status = "failed"
        reason = "; ".join(run.reason for run in measured if run.reason)

    return CaseResult(
        row_name=f"tp{tp_size}_in{input_len}_out{output_len}",
        status=status,
        reason=reason,
        tp_size=tp_size,
        input_length=input_len,
        output_length=output_len,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        case_dir=str(case_dir),
        server_log_path=str(server_log_path),
        server_command=server_command,
        run_results=run_results,
    )


def make_case_failure(
    *,
    tp_size: int,
    input_len: int,
    output_len: int,
    case_dir: Path,
    server_log_path: Path,
    server_command: list[str],
    status: str,
    reason: str,
    num_runs: int,
    num_warmup_runs: int,
) -> CaseResult:
    run_results = [
        BenchRun(
            run_index=index,
            is_warmup=index <= num_warmup_runs,
            status=status,
            reason=reason,
            returncode=None,
            elapsed_s=None,
            ttft_ms=None,
            tpot_ms=None,
            bench_jsonl_path=str(case_dir / f"run_{index:02d}" / "bench_serving.jsonl"),
            client_log_path=str(case_dir / f"run_{index:02d}" / "client.log"),
            command=[],
        )
        for index in range(1, num_runs + 1)
    ]
    return CaseResult(
        row_name=f"tp{tp_size}_in{input_len}_out{output_len}",
        status=status,
        reason=reason,
        tp_size=tp_size,
        input_length=input_len,
        output_length=output_len,
        ttft_ms=None,
        tpot_ms=None,
        case_dir=str(case_dir),
        server_log_path=str(server_log_path),
        server_command=server_command,
        run_results=run_results,
    )


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    *,
    results: list[CaseResult],
    csv_path: Path,
    details_csv_path: Path,
    details_json_path: Path,
) -> None:
    write_csv(csv_path, [result.to_csv_row() for result in results], CSV_COLUMNS)
    write_csv(
        details_csv_path,
        [result.to_detail_row() for result in results],
        DETAIL_COLUMNS,
    )
    launch_time.save_json(
        details_json_path,
        {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "results": [
                {
                    "row_name": result.row_name,
                    "status": result.status,
                    "reason": result.reason,
                    "tp_size": result.tp_size,
                    "input_length": result.input_length,
                    "output_length": result.output_length,
                    "ttft_ms": result.ttft_ms,
                    "tpot_ms": result.tpot_ms,
                    "case_dir": result.case_dir,
                    "server_log_path": result.server_log_path,
                    "server_command": result.server_command,
                    "server_command_text": format_command(result.server_command),
                    "run_results": [run.to_json() for run in result.run_results],
                }
                for result in results
            ],
        },
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    preset = get_preset(args.model_path)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (
        Path(args.csv_path).expanduser().resolve()
        if args.csv_path
        else output_dir / "pe7b_execution_time_summary.csv"
    )
    details_csv_path = output_dir / "pe7b_execution_time_details.csv"
    details_json_path = output_dir / "pe7b_execution_time_details.json"
    visible_gpu_count = launch_time.resolve_visible_gpu_count()

    results: list[CaseResult] = []

    for tp_size in args.gpu_nums:
        tp_dir = output_dir / preset.output_subdir / f"tp{tp_size}"
        server_log_path = tp_dir / "server.log"
        port = launch_time.find_free_port(args.host)
        base_url = f"http://{args.host}:{port}"
        server_command = build_server_command(
            preset=preset,
            tp_size=tp_size,
            host=args.host,
            port=port,
        )
        tp_dir.mkdir(parents=True, exist_ok=True)
        (tp_dir / "server_command.txt").write_text(
            format_command(server_command), encoding="utf-8"
        )

        server_process: subprocess.Popen[str] | None = None
        server_ready = args.dry_run
        server_error = ""

        if not args.dry_run and visible_gpu_count < tp_size:
            server_error = (
                f"Requested tp_size={tp_size}, but only {visible_gpu_count} "
                "visible GPU(s) are available."
            )
        elif not args.dry_run:
            try:
                launch_time.logger.info("Launching PE7B TP%s server", tp_size)
                server_process, _ = launch_time.start_logged_process(
                    command=server_command,
                    log_path=server_log_path,
                )
                launch_time.wait_for_promptenhancer_ready(
                    process=server_process,
                    base_url=base_url,
                    server_log_path=server_log_path,
                    timeout_s=args.timeout_s,
                )
                server_ready = True
            except Exception as exc:
                server_error = str(exc)

        try:
            for input_len, output_len in args.io_cases:
                case_dir = tp_dir / f"input{input_len}_output{output_len}"
                if not server_ready:
                    result = make_case_failure(
                        tp_size=tp_size,
                        input_len=input_len,
                        output_len=output_len,
                        case_dir=case_dir,
                        server_log_path=server_log_path,
                        server_command=server_command,
                        status="skipped" if server_error else "dry_run",
                        reason=server_error,
                        num_runs=args.num_runs,
                        num_warmup_runs=args.num_warmup_runs,
                    )
                else:
                    run_results: list[BenchRun] = []
                    for run_index in range(1, args.num_runs + 1):
                        is_warmup = run_index <= args.num_warmup_runs
                        run_kind = "warmup" if is_warmup else "measure"
                        run_dir = case_dir / f"run_{run_index:02d}_{run_kind}"
                        print(
                            f"[request] tp{tp_size} input={input_len} "
                            f"output={output_len} run {run_index}/{args.num_runs} "
                            f"({run_kind})"
                        )
                        run = run_bench_once(
                            preset=preset,
                            host=args.host,
                            port=port,
                            input_len=input_len,
                            output_len=output_len,
                            run_dir=run_dir,
                            run_index=run_index,
                            is_warmup=is_warmup,
                            timeout_s=args.timeout_s,
                            dry_run=args.dry_run,
                        )
                        run_results.append(run)

                    result = aggregate_case(
                        tp_size=tp_size,
                        input_len=input_len,
                        output_len=output_len,
                        case_dir=case_dir,
                        server_log_path=server_log_path,
                        server_command=server_command,
                        run_results=run_results,
                    )

                results.append(result)
                print(f"[{result.status}] {result.row_name} -> {result.to_csv_row()}")
                write_outputs(
                    results=results,
                    csv_path=csv_path,
                    details_csv_path=details_csv_path,
                    details_json_path=details_json_path,
                )
                if args.fail_fast and result.status not in {"ok", "dry_run"}:
                    raise SystemExit(
                        f"Stopping after {result.status}: {result.reason}"
                    )
        finally:
            launch_time.stop_server(server_process)

        if not args.keep_artifacts:
            launch_time.cleanup_case_dir(tp_dir, keep_artifacts=False)

    write_outputs(
        results=results,
        csv_path=csv_path,
        details_csv_path=details_csv_path,
        details_json_path=details_json_path,
    )
    launch_time.logger.info("PE7B execution summary CSV written to %s", csv_path)


if __name__ == "__main__":
    main()
