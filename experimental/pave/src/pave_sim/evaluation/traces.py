"""Exact arrival/quota construction, independent of routing and execution state."""
from __future__ import annotations

import csv
import hashlib
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from pave_ilp.profiles import digest
from ..state import Request
from ..timing import TICKS_PER_SECOND, rounded_ratio, seconds, ticks


def file_sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build_trace(config):
    path = Path(config.raw_trace).expanduser()
    checksum = file_sha(path)
    if config.expected_raw_sha256 and checksum != config.expected_raw_sha256:
        raise ValueError(f"Raw trace SHA256 mismatch: expected {config.expected_raw_sha256}, actual {checksum}")
    histogram, selected = Counter(), []
    start = datetime.fromisoformat(config.hour_start_utc.replace('Z', '+00:00'))
    duration = ticks(config.duration_s, positive=True)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'TIMESTAMP', 'GeneratedTokens'} <= set(reader.fieldnames or []):
            raise ValueError('Raw trace requires TIMESTAMP and GeneratedTokens')
        for index, row in enumerate(reader, 2):
            try:
                generated = int(row['GeneratedTokens'])
                when = datetime.fromisoformat(row['TIMESTAMP'].replace('Z', '+00:00'))
                if generated < 0 or when.tzinfo is None:
                    raise ValueError('Negative tokens or timezone missing')
                offset = when - start
                offset_tick = ((offset.days * 86400 + offset.seconds) * 1000000 + offset.microseconds) * 1000
            except (ValueError, TypeError) as error:
                raise ValueError(f'Invalid raw trace row {index}: {error}') from error
            histogram[generated] += 1
            if 0 <= offset_tick < duration:
                selected.append({'arrival_tick': offset_tick, 'generated_tokens': generated, 'timestamp': row['TIMESTAMP']})
    count = sum(histogram.values())
    if not count or not selected:
        raise ValueError('Empty raw trace or selected interval')
    ranks, medians, cumulative = ((count - 1) // 2, count // 2), [], 0
    for value, number in sorted(histogram.items()):
        medians.extend(value for rank in ranks if cumulative <= rank < cumulative + number)
        cumulative += number
    median = sum(medians) / 2
    if median != config.expected_p50:
        raise ValueError(f'Global p50 mismatch: expected {config.expected_p50}, actual {median}')
    bins, stats = {}, {}
    for width in (15, 30, 60):
        rows = [{'start_tick': ticks(i * width), 'end_tick': ticks((i + 1) * width),
                 'short': 0, 'long': 0} for i in range(int(config.duration_s / width))]
        for record in selected:
            row = rows[record['arrival_tick'] // ticks(width)]
            row['long' if record['generated_tokens'] >= median else 'short'] += 1
        previous = config.initial_tie
        for row in rows:
            row['source'] = 'Long' if row['long'] > row['short'] else 'Short' if row['short'] > row['long'] else 'Tie'
            row['resolved'] = previous if row['source'] == 'Tie' else row['source']
            previous = row['resolved']
        long_count = sum(row['resolved'] == 'Long' for row in rows)
        changes = sum(left['resolved'] != right['resolved'] for left, right in zip(rows, rows[1:]))
        stats[str(width)] = {'intervals': len(rows), 'long_intervals': long_count,
                             'long_fraction': long_count / len(rows), 'type_changes': changes}
        expected = config.expected_stats.get(str(width))
        if expected and expected != [len(rows), long_count, changes]:
            raise ValueError(f'Trace statistics mismatch for {width}s: {stats[str(width)]}; expected {expected}')
        bins[str(width)] = rows
    if any(row['short'] + row['long'] == 0 for row in bins['60']):
        raise ValueError('Real-proportion trace contains an empty minute; proportion is undefined')
    return {'schema_version': 1, 'raw_sha256': checksum, 'raw_rows': count, 'p50': median,
            'histogram': {str(k): v for k, v in sorted(histogram.items())}, 'hour_start_utc': config.hour_start_utc,
            'duration_s': config.duration_s, 'selected_rows': selected,
            'raw_long_fraction': sum(r['generated_tokens'] >= median for r in selected) / len(selected),
            'minute_mean_long_fraction': sum(r['long'] / (r['long'] + r['short']) for r in bins['60']) / len(bins['60']),
            'bins': bins, 'statistics': stats}


def arrivals(rate, duration_s):
    period = Fraction(60 * TICKS_PER_SECOND) / Fraction(str(rate))
    if period < 1:
        raise ValueError('Arrival period must be at least one tick')
    result, index, end = [], 0, ticks(duration_s, positive=True)
    while True:
        time = rounded_ratio(index * period.numerator, period.denominator)
        if time >= end:
            return result
        result.append(time)
        index += 1


def construct_requests(config, trace, name, rate, seed=None):
    times = arrivals(rate, config.duration_s)
    if name not in ('majority15', 'majority30', 'majority60', 'mixture'):
        raise ValueError(f'Unknown trace kind: {name}')
    labels, quotas = [], []
    if name == 'mixture':
        if type(seed) is not int:
            raise ValueError('Mixture requires an integer request seed')
        rng, cumulative, assigned = random.Random(seed), Fraction(), 0
        for row in trace['bins']['60']:
            n = sum(row['start_tick'] <= time < row['end_tick'] for time in times)
            denominator = row['short'] + row['long']
            if not denominator:
                raise ValueError('Undefined real proportion in empty minute')
            cumulative += n * Fraction(row['long'], denominator)
            target = rounded_ratio(cumulative.numerator, cumulative.denominator)
            long_n = target - assigned
            if not 0 <= long_n <= n:
                raise RuntimeError('Invalid cumulative quota')
            kinds = ['Long'] * long_n + ['Short'] * (n - long_n)
            rng.shuffle(kinds)
            labels.extend(kinds)
            assigned = target
            quotas.append({'start_tick': row['start_tick'], 'requests': n, 'long': long_n,
                           'cumulative_long': assigned, 'expected_numerator': cumulative.numerator,
                           'expected_denominator': cumulative.denominator})
    else:
        if seed is not None:
            raise ValueError('Majority traces do not use a request seed')
        width = int(name.removeprefix('majority'))
        labels = [trace['bins'][str(width)][time // ticks(width)]['resolved'] for time in times]
    records = [{'id': i, 'arrival_tick': time, 'input_tokens': config.input_tokens,
                'output_tokens': config.long_output_tokens if kind == 'Long' else config.short_output_tokens,
                'kind': kind} for i, (time, kind) in enumerate(zip(times, labels))]
    return {'schema_version': 1, 'trace': name, 'rate_per_min': float(rate), 'request_seed': seed,
            'duration_s': float(config.duration_s), 'records': records, 'quotas': quotas,
            'requests_sha256': digest(records), 'long_fraction': labels.count('Long') / len(labels) if labels else None,
            'type_changes': sum(a != b for a, b in zip(labels, labels[1:]))}


@dataclass(frozen=True)
class RequestPlan:
    """Primitive immutable rows; each invocation returns fresh mutable Requests."""
    rows: tuple[tuple, ...]
    rate: float
    duration_s: float

    @classmethod
    def load(cls, payload, settings, rate):
        records = payload['records']
        if payload.get('schema_version') != 1 or digest(records) != payload.get('requests_sha256'):
            raise ValueError('Request list schema/digest mismatch')
        if Fraction(str(payload['rate_per_min'])) != Fraction(str(rate)) or ticks(payload['duration_s']) != ticks(settings.duration_s):
            raise ValueError('Request workload mismatch')
        expected = arrivals(rate, settings.duration_s)
        if len(records) != len(expected):
            raise ValueError('Request count differs from arrival definition')
        rows = []
        for i, (record, time) in enumerate(zip(records, expected)):
            kind = record['kind']
            output = settings.short_output_tokens if kind == 'Short' else settings.long_output_tokens
            if (kind not in ('Short', 'Long') or any(type(record[k]) is not int for k in ('id', 'arrival_tick', 'input_tokens', 'output_tokens'))
                    or record['id'] != i or record['arrival_tick'] != time or record['input_tokens'] != settings.input_tokens
                    or record['output_tokens'] != output):
                raise ValueError(f'Invalid request record {i}')
            rows.append((i, time, record['input_tokens'], output, kind))
        return cls(tuple(rows), float(rate), float(settings.duration_s))

    def requests(self, settings, rate):
        if Fraction(str(rate)) != Fraction(str(self.rate)) or ticks(settings.duration_s) != ticks(self.duration_s):
            raise ValueError('Request plan reused for a different workload')
        return [Request(i, seconds(time), inp, output, kind) for i, time, inp, output, kind in self.rows]
