"""Evaluation commands are explicit; report/select never execute a workload."""
import argparse
import json
import sys
from pathlib import Path

from .config import EvaluationConfig


def parser():
    result = argparse.ArgumentParser(prog='pave-sim evaluation')
    sub = result.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Read/validate inputs and freeze slots; no simulation')
    prepare.add_argument('--config', type=Path, required=True)
    prepare.add_argument('--output-dir', type=Path, required=True)
    run = sub.add_parser('run', help='Run one foreground phase; stops on the first failure')
    run.add_argument('--campaign', type=Path, required=True)
    run.add_argument('--phase', choices=['2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.6-low', '2.6-rest'], required=True)
    run.add_argument('--retry-failed', action='store_true')
    selection = sub.add_parser('select', help='Freeze selection from complete archived results')
    selection.add_argument('--campaign', type=Path, required=True)
    selection.add_argument('--target', choices=['parameters', 'cluster'], required=True)
    selection.add_argument('--output-file', type=Path, help='Recompute a review receipt without changing frozen selections')
    report = sub.add_parser('report', help='Reconstruct raw tables; no simulation')
    report.add_argument('--campaign', type=Path, required=True)
    report.add_argument('--experiment', choices=list('abcdef') + ['all'], default='all')
    report.add_argument('--output-dir', type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        from .campaign import prepare, run_phase
        from .analysis import report, select
        if args.command == 'prepare':
            outcome = prepare(EvaluationConfig.load(args.config.expanduser().resolve()), args.output_dir.expanduser().resolve())
        elif args.command == 'run':
            outcome = run_phase(args.campaign.expanduser().resolve(), args.phase, retry_failed=args.retry_failed)
        elif args.command == 'select':
            outcome = select(args.campaign.expanduser().resolve(), args.target,
                             args.output_file.expanduser().resolve() if args.output_file else None)
        else:
            outcome = report(args.campaign.expanduser().resolve(), args.output_dir.expanduser().resolve(), args.experiment)
        print(json.dumps(outcome, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f'pave-sim evaluation: {error}', file=sys.stderr)
        return 1
