"""Reconstruct metrics and selections from archived facts, never run simulations."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from pave_ilp.profiles import digest
from ..metrics import allocations, percentile, request_rows, summary
from ..records import write_csv, write_json
from ..timing import elapsed, seconds, ticks
from .campaign import fingerprint_sources, load_campaign, load_result, now, write_manifest
from .matrix import references

METRIC_COLUMNS = ('throughput_req_s', 'p50_s', 'p99_s', 'slo5', 'slo10', 'observed_throughput_req_s',
                  'observed_completed', 'end_backlog', 'drain_s', 'last_request_finish_s',
                  'flip_triggered', 'flip_completed', 'converted_groups', 'gpu_unavailable_s',
                  'pe_prefill_s', 'pe_reprefill_s', 'total_wait_s', 'total_service_s',
                  'long_fraction', 'type_changes', 'requests')


def measure(result, binding):
    values = summary(result)
    effective = binding['context']['effective']
    if result['run_id'] != binding['run_id']:
        raise ValueError('Result identity mismatch')
    values.update(effective)
    values.update(slot_id=binding['bound']['slot_id'], case_id=binding['bound']['case_id'],
                  diagnostic_window_s=result['spec']['window_s'],
                  identity_sha256=binding['identity_sha256'],
                  requests_sha256=binding['context']['requests_sha256'])
    values['trace_statistics'] = binding['context'].get('trace_statistics', {})
    rows = request_rows(result)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate completed request')
    end = ticks(effective['duration_s'])
    observed = sum(ticks(r['finished_s']) <= end for r in rows)
    values.update(observed_completed=observed, observed_throughput_req_s=observed / effective['duration_s'],
                  end_backlog=sum(ticks(r['arrival_s']) < end < ticks(r['finished_s']) for r in rows),
                  drain_s=seconds(max(0, ticks(values['last_request_finish_s']) - end)),
                  long_fraction=sum(r['kind'] == 'Long' for r in rows) / len(rows),
                  type_changes=sum(a['kind'] != b['kind'] for a, b in zip(rows, rows[1:])))
    stages = {s: {'waiting_s': 0.0, 'service_s': 0.0, 'executed_work': 0, 'completed': 0} for s in ('PE', 'TE', 'DiT', 'VAE')}
    waiting, service, prefill, reprefill = 0.0, 0.0, 0.0, 0.0
    per_request = defaultdict(lambda: {'waiting_s': 0.0, 'service_s': 0.0})
    for a in result['attempts']:
        if a['exit_s'] is None:
            raise ValueError('Unclosed execution/queue attempt')
        q = elapsed(a['exit_s'] if a['start_s'] is None else a['start_s'], a['enter_s'])
        busy = a.get('executed_service_s', 0.0)
        if q < 0 or busy < 0:
            raise ValueError('Negative wait/service time')
        stages[a['stage']]['waiting_s'] += q
        stages[a['stage']]['service_s'] += busy
        stages[a['stage']]['executed_work'] += a['executed_work']
        stages[a['stage']]['completed'] += a['exit_reason'] == 'completed'
        per_request[a['request_id']]['waiting_s'] += q
        per_request[a['request_id']]['service_s'] += busy
        waiting += q
        service += busy
        if a['stage'] == 'PE' and a['start_s'] is not None:
            amount = min(busy, seconds(ticks(a['pe_profile']['ttft_s'], positive=True)))
            prefill += amount
            if a['initial_progress'] > 0:
                reprefill += amount
    if stages['PE']['executed_work'] != sum(r['output_tokens'] for r in rows) or stages['DiT']['executed_work'] != 50 * len(rows):
        raise ValueError('PE token or DiT step conservation failed')
    for s in stages:
        if stages[s]['completed'] != len(rows):
            raise ValueError(f'Duplicate/missing completed stage: {s}')
    for r in rows:
        cost = per_request[r['id']]
        if not math.isclose(cost['waiting_s'] + cost['service_s'], r['latency_s'], rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError(f'Request time accounting failed: {r["id"]}')
    directions = {s: {'triggered': 0, 'completed': 0, 'groups': 0, 'startup_instance_s': 0.0,
                       'gpu_unavailable_s': 0.0, 'safe_wait_s': 0.0} for s in ('short_to_long', 'long_to_short')}
    for flip in result['flips']:
        d = directions[flip['direction']]
        d['triggered'] += 1
        d['completed'] += flip['completed_s'] is not None
        for group in flip['groups']:
            if group['ready_s'] is None:
                raise ValueError('Incomplete conversion in a completed run')
            d['groups'] += 1
            d['safe_wait_s'] += elapsed(group['launch_started_s'], flip['detected_s'])
            # Split PE children can become ready independently. Account GPU
            # unavailability per child instead of charging every GPU until last ready.
            physicals = {p['uid']: p for p in result['instances']}
            for child in group['targets']:
                d['startup_instance_s'] += child['startup_s']
                gpu_count = len(physicals[child['instance_uid']]['raw']['gpu_ids'])
                d['gpu_unavailable_s'] += gpu_count * elapsed(child['ready_s'], flip['detected_s'])
    values.update(stages=stages, directions=directions, total_wait_s=waiting, total_service_s=service,
                  pe_prefill_s=prefill, pe_reprefill_s=reprefill,
                  gpu_unavailable_s=sum(d['gpu_unavailable_s'] for d in directions.values()),
                  instance_allocations=allocations(result))
    return values


def rank_parameters(rows, config):
    expected = {(g, c, float(r)) for g, groups in config.rates.items() for c in config.main_clusters for r in groups[c]}
    groups = {}
    for row in rows:
        condition = (row['generator'], row['scenario'], row['rate_per_min'])
        key = (row['window_s'], row['margin'], condition)
        if key in groups or row['policy'] != 'E' or row['trace'] != 'majority60':
            raise ValueError('Duplicate or invalid grid result')
        if any(not math.isfinite(row[k]) or row[k] <= 0 for k in ('throughput_req_s', 'p99_s')) or not 0 <= row['slo5'] <= 1:
            raise ValueError('Invalid selection metric')
        groups[key] = row
    wanted = {(w, m, c) for w in config.windows_s for m in config.margins for c in expected}
    if set(groups) != wanted:
        raise ValueError('Parameter selection requires the exact complete grid')
    max_t = {c: max(groups[w, m, c]['throughput_req_s'] for w in config.windows_s for m in config.margins) for c in expected}
    min_p = {c: min(groups[w, m, c]['p99_s'] for w in config.windows_s for m in config.margins) for c in expected}
    ranking = []
    for w in config.windows_s:
        for m in config.margins:
            selected = [groups[w, m, c] for c in sorted(expected)]
            ranking.append({'window_s': w, 'margin': m,
                'throughput_score': math.exp(math.fsum(math.log(groups[w, m, c]['throughput_req_s'] / max_t[c]) for c in sorted(expected)) / len(expected)),
                'slo5_score': math.fsum(r['slo5'] for r in selected) / len(selected),
                'p99_score': math.exp(math.fsum(math.log(groups[w, m, c]['p99_s'] / min_p[c]) for c in sorted(expected)) / len(expected)),
                'run_ids': [r['run_id'] for r in selected]})
    best = max(r['throughput_score'] for r in ranking)
    eligible = [r for r in ranking if r['throughput_score'] >= best * 0.995]
    chosen = min(eligible, key=lambda r: (-r['slo5_score'], r['p99_score'], r['window_s'], r['margin']))
    for row in ranking:
        row['throughput_eligible'] = row['throughput_score'] >= best * 0.995
    return {'window_s': chosen['window_s'], 'margin': chosen['margin']}, sorted(ranking, key=lambda r: (-r['throughput_score'], -r['slo5_score'], r['p99_score'], r['window_s'], r['margin']))


def rank_cluster(rows, config):
    lookup = {}
    for row in rows:
        key = (row['generator'], row['scenario'], row['rate_per_min'], row['policy'])
        if key in lookup:
            raise ValueError('Duplicate cluster comparison result')
        lookup[key] = row
    ranking = []
    for cluster in ('cluster1', 'cluster2'):
        pairs = []
        for model in config.rates:
            for rate in config.rates[model][cluster]:
                try:
                    e, static = lookup[model, cluster, rate, 'E'], lookup[model, cluster, rate, 'Static-2048']
                except KeyError as error:
                    raise ValueError('Incomplete cluster comparisons') from error
                if static['throughput_req_s'] <= 0:
                    raise ValueError('Nonpositive cluster baseline')
                pairs.append({'generator': model, 'rate_per_min': rate,
                    'relative_gain': e['throughput_req_s'] / static['throughput_req_s'] - 1,
                    'run_ids': [e['run_id'], static['run_id']]})
        ranking.append({'cluster': cluster, 'score': math.fsum(p['relative_gain'] for p in pairs) / len(pairs), 'pairs': pairs})
    ranking.sort(key=lambda r: (-r['score'], r['cluster']))
    return ranking[0]['cluster'], ranking


def select(root: Path, target, output_file: Path | None = None):
    manifest, config = load_campaign(root)
    if (root / 'execution.lock').exists():
        raise ValueError('Cannot change selections while execution is active')
    if target in manifest['selections'] and output_file is None:
        raise ValueError('Selection already frozen; do not silently replace its dependent runs')
    if output_file is not None and output_file.exists():
        raise ValueError('Selection review output must be a new file')
    if target == 'parameters':
        chosen_slots = [s['slot_id'] for s in manifest['slots'] if s['batch'] == 'grid']
    elif target == 'cluster':
        if 'parameters' not in manifest['selections']:
            raise ValueError('Cluster selection requires parameter selection')
        chosen_slots = references(config, manifest['slots'], manifest['selections'])['a']
    else:
        raise ValueError('Selection target must be parameters or cluster')
    rows = [measure(load_result(root, manifest, manifest['bindings'][s]), manifest['bindings'][s]) for s in chosen_slots]
    chosen, ranking = rank_parameters(rows, config) if target == 'parameters' else rank_cluster(rows, config)
    receipt = {'chosen': chosen, 'ranking': ranking, 'created_utc': now(), 'rule_version': 2,
               'analysis_sha256': fingerprint_sources()['analysis_sha256'],
               'input_identities': {r['run_id']: r['identity_sha256'] for r in rows}}
    receipt['selection_sha256'] = digest(receipt)
    if output_file is not None:
        write_json(output_file, receipt)
        return {'target': target, 'chosen': chosen, 'source_runs': len(rows), 'committed': False, 'output_file': str(output_file)}
    manifest['selections'][target] = receipt
    write_manifest(root, manifest, config)
    write_json(root / f'selection_{target}.json', receipt)
    return {'target': target, 'chosen': chosen, 'source_runs': len(rows), 'committed': True}


def flatten(data, prefix=''):
    result = {}
    for key, value in data.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(flatten(value, name + '.'))
        elif key != 'instance_allocations':
            result[name] = value
    return result


def aggregate(rows):
    groups = defaultdict(list)
    keys = ('generator', 'scenario', 'rate_per_min', 'policy', 'trace', 'window_s', 'margin')
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, values in groups.items():
        output = dict(zip(keys, key))
        output.update(repetitions=len(values), run_ids=[r['run_id'] for r in values])
        # Each run's p99/SLO is computed before averaging. Never pool requests.
        flattened = [flatten(r) for r in values]
        nested = sorted({k for r in flattened for k in r if k.startswith(('short.', 'long.', 'stages.', 'directions.', 'trace_statistics.'))})
        for metric in (*METRIC_COLUMNS, 'final_control_time_s', *nested):
            numbers = [r[metric] for r in flattened if r.get(metric) is not None]
            output[metric + '_samples'] = len(numbers)
            output[metric + '_mean'] = math.fsum(numbers) / len(numbers) if numbers else None
            output[metric + '_min'] = min(numbers) if numbers else None
            output[metric + '_max'] = max(numbers) if numbers else None
        result.append(output)
    return result


def comparisons(name, rows, parameters):
    """Explicit paired contrasts; deterministic references retain one run ID."""
    if name == 'd':
        return []
    groups = defaultdict(list)
    for row in rows:
        groups[row['generator'], row['scenario'], row['rate_per_min']].append(row)
    output = []
    for group in groups.values():
        selected = [r for r in group if r['policy'] == 'E' and r['window_s'] == parameters['window_s'] and r['margin'] == parameters['margin']]
        if len(selected) != 1:
            raise ValueError('Missing or ambiguous comparison reference')
        ours = selected[0]
        pairs = [(variant, ours) if name in ('b', 'c') else (ours, variant)
                 for variant in group if variant['run_id'] != ours['run_id']]
        if name == 'f':
            policies = {r['policy']: r for r in group}
            if len(policies) != len(group):
                raise ValueError('Duplicate startup comparison policy')
            if {'E-PEOnly', 'E-NoOpt'} <= policies.keys():
                pairs.append((policies['E-PEOnly'], policies['E-NoOpt']))
        for new, base in pairs:
            if new['requests_sha256'] != base['requests_sha256']:
                raise ValueError('Comparison requests differ outside the declared experimental axis')
            item = {'new_run_id': new['run_id'], 'base_run_id': base['run_id'],
                    'generator': new['generator'], 'scenario': new['scenario'], 'rate_per_min': new['rate_per_min'],
                    'new_policy': new['policy'], 'base_policy': base['policy'],
                    'new_window_s': new['window_s'], 'base_window_s': base['window_s'],
                    'new_margin': new['margin'], 'base_margin': base['margin'],
                    'new_scheduler_seed': new['scheduler_seed'], 'base_scheduler_seed': base['scheduler_seed']}
            for metric in ('throughput_req_s', 'p50_s', 'p99_s', 'slo5', 'slo10'):
                delta = new[metric] - base[metric]
                improvement = -delta if metric in ('p50_s', 'p99_s') else delta
                item[metric + '_absolute_change'] = delta
                item[metric + '_improvement_percent'] = improvement / base[metric] * 100 if base[metric] else None
                if metric in ('slo5', 'slo10'):
                    item[metric + '_percentage_points'] = delta * 100
            output.append(item)
    return output


def startup_tables(results):
    """Physical startup facts; concurrent children are never summed into group latency."""
    targets, groups, periods = [], [], []
    for result in results:
        effective = result['evaluation_context']['effective']
        end = ticks(effective['duration_s'])
        physicals = {p['uid']: p['raw'] for p in result['instances']}
        common = {k: effective[k] for k in ('generator', 'scenario', 'policy', 'rate_per_min', 'startup_mode')}
        common['run_id'] = result['run_id']
        bins = {(period, direction): dict(common, period=period, direction=direction,
                detected=0, completed=0, converted_groups=0, startup_gpu_s=0., unavailable_gpu_s=0.)
                for period in ('observation', 'drain') for direction in ('short_to_long', 'long_to_short')}
        def duration_in_period(start, stop, period):
            return seconds(max(0, min(stop, end) - start) if period == 'observation' else max(0, stop - max(start, end)))
        for flip in result['flips']:
            detected, completed = ticks(flip['detected_s']), ticks(flip['completed_s'])
            direction = flip['direction']
            bins['observation' if detected <= end else 'drain', direction]['detected'] += 1
            bins['observation' if completed <= end else 'drain', direction]['completed'] += 1
            for group in flip['groups']:
                launch, ready = ticks(group['launch_started_s']), ticks(group['ready_s'])
                bins['observation' if detected <= end else 'drain', direction]['converted_groups'] += 1
                base = dict(common, flip_id=flip['flip_id'], group_id=group['group_id'], direction=direction,
                            node=group['node'], detected_s=flip['detected_s'])
                groups.append(dict(base, gpu_ids=group['gpu_ids'], target_templates=group['target_templates'],
                    safe_wait_s=seconds(launch-detected), launch_started_s=group['launch_started_s'],
                    ready_s=group['ready_s'], startup_elapsed_s=seconds(ready-launch)))
                for target in group['targets']:
                    raw = physicals[target['instance_uid']]
                    pe = set(raw['stages']) == {'PE'}
                    opt = effective['startup_mode'] == 'all_optimized' or (effective['startup_mode'] == 'pe_only_optimized' and pe)
                    expected = raw['startup']['optimized_s' if opt else 'non_optimized_s']
                    if target['optimized'] != opt or ticks(target['startup_s']) != ticks(expected) or target['startup_mode'] != effective['startup_mode']:
                        raise ValueError('Startup record/profile/mode mismatch')
                    finish = ticks(target['ready_s'])
                    targets.append(dict(base, instance_uid=target['instance_uid'], template=raw['template'],
                        hardware=raw['hardware'], gpu_ids=raw['gpu_ids'], optimized=opt,
                        optimized_profile_s=raw['startup']['optimized_s'], non_optimized_profile_s=raw['startup']['non_optimized_s'],
                        actual_startup_s=target['startup_s'], launch_started_s=target['launch_started_s'], ready_s=target['ready_s']))
                    for period in ('observation', 'drain'):
                        bins[period, direction]['startup_gpu_s'] += len(raw['gpu_ids']) * duration_in_period(launch, finish, period)
                        bins[period, direction]['unavailable_gpu_s'] += len(raw['gpu_ids']) * duration_in_period(detected, finish, period)
        periods.extend(bins.values())
    return {'startups': targets, 'groups': groups, 'direction_periods': periods}


def report(root: Path, destination: Path, experiment='all'):
    manifest, config = load_campaign(root)
    names = list('abcdef') if experiment == 'all' else [experiment]
    if any(name not in 'abcdef' or len(name) != 1 for name in names):
        raise ValueError('Unknown experiment')
    if any(n in ('d', 'e') for n in names) and 'cluster' not in manifest['selections']:
        raise ValueError('Trace/scheduler reports require cluster selection')
    refs = references(config, manifest['slots'], manifest['selections'])
    selected = set(s for n in names for s in refs[n])
    # Validate the complete requested dataset before creating any report files.
    rows = {s: measure(load_result(root, manifest, manifest['bindings'][s]), manifest['bindings'][s]) for s in sorted(selected)}
    tables = {}
    for name in names:
        data = [rows[s] for s in refs[name]]
        tables[name] = (data, aggregate(data), comparisons(name, data, manifest['selections']['parameters']['chosen']))
    startup = startup_tables([load_result(root, manifest, manifest['bindings'][s]) for s in refs['f']]) if 'f' in names else None
    if destination.exists():
        raise ValueError('Report destination must be new')
    destination.mkdir(parents=True)
    counts = {}
    if startup is not None:
        for kind, records in startup.items():
            write_json(destination / f'f_{kind}.json', records)
            write_csv(destination / f'f_{kind}.csv', records)
    for name in names:
        data, combined, paired = tables[name]
        write_json(destination / f'{name}_runs.json', data)
        write_csv(destination / f'{name}_runs.csv', [flatten(row) for row in data])
        write_csv(destination / f'{name}_summary.csv', combined)
        write_csv(destination / f'{name}_allocations.csv', [entry for r in data for entry in r['instance_allocations']])
        if name != 'd':
            write_csv(destination / f'{name}_comparisons.csv', paired)
        counts[name] = {'runs': len(data), 'summary_rows': len(combined)}
    write_json(destination / 'report.json', {'schema_version': 1, 'kind': 'evaluation_tables',
               'created_utc': now(), 'analysis_sha256': fingerprint_sources()['analysis_sha256'],
               'configuration_sha256': manifest['configuration_sha256'], 'counts': counts,
               'selections': manifest['selections'], 'd_raw_metrics_only': True,
               'source_run_ids': sorted({r['run_id'] for r in rows.values()})})
    lines = ['# Evaluation 数据表', '', '全部数值从原始请求、执行和转换记录重算。', '',
             '| 实验 | 原始运行数 | 汇总行数 |', '|---|---:|---:|']
    lines += [f'| {name} | {value["runs"]} | {value["summary_rows"]} |' for name, value in counts.items()]
    lines += ['', 'summary中的mean/min/max针对逐运行指标；不是合并请求计算的分位数。',
              'd仅提供原始单位，未归一化。完整setup、选择依据和来源见report.json及各实验runs文件。', '']
    for name in names:
        lines += [f'## 实验 {name}', '', '| 模型 / 集群 | req/min | 策略 | Trace | W | M | 吞吐 req/s | p50 s | p99 s | SLO5 | 重复数 |',
                  '|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in tables[name][1]:
            cells = [f"{row['generator']} / {row['scenario']}", row['rate_per_min'], row['policy'], row['trace'],
                     row['window_s'], row['margin'], *[row[k + '_mean'] for k in ('throughput_req_s', 'p50_s', 'p99_s', 'slo5')], row['repetitions']]
            lines.append('| ' + ' | '.join('—' if x is None else f'{x:.6g}' if isinstance(x, float) else str(x) for x in cells) + ' |')
        lines.append('')
    (destination / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return {'output_dir': str(destination), 'counts': counts, 'unique_runs': len(rows)}
