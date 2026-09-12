"""Read raw completed-run records to count flips and hardware work shares."""
import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    return json.loads(path.read_text('utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inspect(campaign, run_id):
    root = Path(campaign).resolve()
    manifest = read(root / 'manifest.json')
    run = manifest['runs'][run_id]
    if run['status'] != 'complete':
        raise ValueError('A completed run is required')
    attempts = [a for a in run['attempts'] if a['status'] == 'complete']
    if len(attempts) != 1:
        raise ValueError('Ambiguous completed attempts')
    saved = attempts[0]
    directory = (root / saved['path']).resolve()
    if not directory.is_relative_to(root):
        raise ValueError('Unsafe run directory')
    for name, checksum in saved['files'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or sha(path) != checksum:
            raise ValueError(f'Raw record checksum mismatch: {name}')
    flips = read(directory / 'flips.json')
    completed_flips = sum(f.get('completed_s') is not None for f in flips)
    grouped = defaultdict(lambda: dict(attempts=0, completed=0, work=0, service_s=0.0))
    opener = gzip.open if (directory / 'attempts.jsonl.gz').exists() else open
    name = 'attempts.jsonl.gz' if opener is gzip.open else 'attempts.jsonl'
    with opener(directory / name, 'rt', encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line)
            key = r['stage'], r['hardware'], r['work_unit']
            row = grouped[key]
            row['attempts'] += 1
            row['completed'] += r['exit_reason'] == 'completed'
            row['work'] += r['executed_work']
            row['service_s'] += r['executed_service_s']
    rows = []
    for (stage, hardware, unit), row in sorted(grouped.items()):
        total = sum(v['work'] for (s, _, u), v in grouped.items() if s == stage and u == unit)
        count = sum(v['completed'] for (s, _, _), v in grouped.items() if s == stage)
        rows.append(dict(stage=stage, hardware=hardware, work_unit=unit, **row,
                         work_share=row['work']/total if total else None,
                         completion_share=row['completed']/count if count else None))
    summary = read(directory / 'summary.json')
    if len(flips) != summary['flip_triggered'] or completed_flips != summary['flip_completed']:
        raise ValueError('Raw flip count differs from summary')
    return dict(run_id=run_id, raw_directory=str(directory), verified_files=len(saved['files']),
                flips_triggered=len(flips), flips_completed=completed_flips,
                flip_directions=dict(Counter(f['direction'] for f in flips)),
                hardware_work=rows,
                counting_note='Attempts include migrations; completed counts module finishes. '
                              'DiT work shares count actual steps across all attempts, including migrated work.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-file', type=Path)
    args = parser.parse_args(argv)
    result = inspect(args.campaign, args.run_id)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_file:
        target = args.output_file.expanduser().resolve()
        if target.is_relative_to(args.campaign.resolve()):
            parser.error('Write inspection output outside the campaign')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8') as stream:
            stream.write(text + '\n')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
