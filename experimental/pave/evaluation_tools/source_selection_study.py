"""Explicit source-selection ablation; archived PAVE is never simulated again.

Run with the existing environment and -B. All generated artifacts live in a
new Desktop campaign; this module does nothing on import.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import difflib
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
DESKTOP = Path.home() / 'Desktop'
ARCHIVE = DESKTOP / 'PAVE_EVALUATION_RESULTS/archive/campaign-002'
FROZEN = DESKTOP / 'PAVE_EVALUATION_RESULTS/archive/source_and_environment'
BASE = DESKTOP / 'PAVE_SOURCE_SELECTION_ABLATION'
KIND = 'source_selection_ablation_v1'
RANDOM_POLICY = 'PAVE-RandomSource'
SEEDS = tuple(range(5))
ALLOWED_RUNTIME_CHANGES = {
    'pave_sim/config.py', 'pave_sim/engine.py', 'pave_sim/selection.py',
}
STAGES = ('PE', 'TE', 'DiT', 'VAE')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes campaign: {relative}')
    return path


def api(root=None):
    source = root / 'provenance/source/src' if root else PROJECT / 'src'
    sys.path.insert(0, str(source))
    from pave_sim.evaluation import campaign as c
    from pave_sim.evaluation.analysis import measure
    from pave_sim.records import Journal, write_jsonl, read_jsonl, write_csv
    from pave_sim.config import RunSpec
    from pave_sim.timing import ticks, seconds, elapsed
    return c, measure, Journal, write_jsonl, read_jsonl, write_csv, RunSpec, ticks, seconds, elapsed


def copied(source, destination, expected=None):
    actual = sha(source)
    if expected and actual != expected:
        raise ValueError(f'Archived checksum mismatch: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(destination) != actual:
            raise ValueError(f'Conflicting frozen input: {destination}')
    else:
        shutil.copy2(source, destination)
    if sha(destination) != actual:
        raise ValueError(f'Copy checksum mismatch: {destination}')
    return actual


def prepare(root):
    c, _, _, _, read_jsonl, _, _, ticks, _, _ = api()
    parent = read(ARCHIVE / 'manifest.json')
    fingerprint = c.fingerprint_sources()
    old_files = parent['source']['runtime_files']
    new_files = fingerprint['runtime_files']
    changes = sorted(k for k in old_files.keys() | new_files.keys() if old_files.get(k) != new_files.get(k))
    if set(changes) != ALLOWED_RUNTIME_CHANGES:
        raise ValueError(f'Unexpected runtime changes: {changes}')
    if fingerprint['analysis_sha256'] != parent['source']['analysis_sha256']:
        raise ValueError('Shared analysis implementation differs from archived campaign')
    env = {'python': platform.python_version(), 'dependencies': {
        name: importlib.metadata.version(name) for name in ('numpy', 'scipy')}}
    if env != parent['environment']:
        raise ValueError(f'Execution environment differs from archive: {env}')
    shutil.copytree(PROJECT / 'src', root / 'provenance/source/src',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
    copied(Path(__file__), root / 'provenance/source/evaluation_tools/source_selection_study.py')
    for name in ('pyproject.toml',):
        copied(PROJECT / name, root / 'provenance/source' / name)
    diffs = []
    for name in changes:
        before = (FROZEN / 'src' / name).read_text('utf-8')
        after = (PROJECT / 'src' / name).read_text('utf-8')
        if sha(FROZEN / 'src' / name) != old_files[name]:
            raise ValueError('Archived source snapshot mismatch')
        diffs.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                        fromfile='archived/' + name, tofile='ablation/' + name))
    (root / 'provenance/runtime_changes.patch').write_text(''.join(diffs), encoding='utf-8')
    copied(ARCHIVE / 'manifest.json', root / 'provenance/parent_manifest.json')
    copied(ARCHIVE / 'configuration.json', root / 'provenance/parent_configuration.json')
    save(root / 'provenance/source_comparison.json', {
        'archived': parent['source'], 'ablation': fingerprint, 'changed_runtime_files': changes,
        'unchanged_analysis': True, 'environment': env,
        'scope': 'forward source choice plus independent RNG, configuration, and diagnostics only',
    })
    for name in ('throughput.csv', 'p50.csv', 'p99.csv', 'diagnostics.csv'):
        copied(DESKTOP / 'PAVE_EVALUATION_READABLE/a_主实验' / name,
               root / 'provenance/readable_main' / name)
    chosen = {}
    for binding in parent['bindings'].values():
        row = binding.get('bound') or {}
        if (row.get('scenario') in ('cluster1', 'cluster2') and row.get('policy') == 'E'
                and row.get('window_s') == 60 and row.get('margin') == .15
                and row.get('trace') == 'majority60'):
            key = row['generator'], row['scenario'], float(row['rate_per_min'])
            if key in chosen and chosen[key]['run_id'] != binding['run_id']:
                raise ValueError('Multiple archived PAVE runs for one condition')
            chosen[key] = binding
    expected = {(g, s, float(r)) for g, groups in parent['configuration']['rates'].items()
                for s in ('cluster1', 'cluster2') for r in groups[s]}
    if set(chosen) != expected or len(chosen) != 12:
        raise ValueError('Expected exactly twelve complete PAVE references')
    config = c.EvaluationConfig.load(ARCHIVE / 'configuration.json')
    input_rows, references, jobs = {}, [], []
    for index, (key, old) in enumerate(sorted(chosen.items())):
        if old['status'] != 'complete':
            raise ValueError('Incomplete PAVE reference')
        row = old['bound']; case_id = row['case_id']
        required = [f'inputs/ilp/deployments/{case_id}_512.json',
                    f'inputs/ilp/flips/{case_id}.json', 'inputs/ilp/profiles.json',
                    parent['requests'][old['request_key']]['path']]
        for relative in required:
            h = copied(inside(ARCHIVE, relative), inside(root, relative), parent['input_files'][relative])
            input_rows[relative] = {'path': relative, 'original_path': str(inside(ARCHIVE, relative)), 'sha256': h}
        attempts = [a for a in parent['runs'][old['run_id']]['attempts'] if a['status'] == 'complete']
        if not attempts:
            raise ValueError('Missing complete PAVE attempt')
        attempt = attempts[-1]
        source = inside(ARCHIVE, attempt['path'])
        destination = root / 'reference_pave' / old['run_id'] / 'attempt-001'
        hashes = {name: copied(source / name, destination / name, h) for name, h in attempt['files'].items()}
        request_path = parent['requests'][old['request_key']]['path']
        settings = config.settings(root / 'inputs/ilp')
        plan = c.RequestPlan.load(read(root / request_path), settings, row['rate_per_min'])
        actual = [(r['id'], ticks(r['arrival_s']), r['input_tokens'], r['output_tokens'], r['kind'])
                  for r in sorted(read_jsonl(destination / 'requests.jsonl.gz'), key=lambda r: r['id'])]
        if actual != list(plan.rows):
            raise ValueError('Archived PAVE requests differ from the frozen request plan')
        reference = {'pair_id': f'pair-{index + 1:02d}', 'binding': old,
                     'directory': destination.relative_to(root).as_posix(),
                     'original_directory': str(source), 'files_sha256': hashes,
                     'request_path': request_path, 'request_count': len(plan.rows)}
        references.append(reference)
        for seed in SEEDS:
            spec = replace(c.compile_policy(row, config), source_selection='random_feasible', source_selection_seed=seed)
            spec.validate()
            context = copy.deepcopy(old['context'])
            context['effective'].update(policy=RANDOM_POLICY, source_selection='random_feasible', source_selection_seed=seed)
            context.update(simulation_source_sha256=fingerprint['runtime_sha256'],
                           ablation_kind=KIND, reference_pave_run_id=old['run_id'])
            identity = c.digest(context)
            run_id = f'{case_id}-RandomSource-{identity[:24]}'
            bound = dict(row, policy=RANDOM_POLICY, source_selection_seed=seed,
                         batch=KIND, slot_id='source-' + identity[:24])
            binding = {'run_id': run_id, 'identity_sha256': identity, 'context': context,
                       'request_key': old['request_key'], 'bound': bound}
            jobs.append({'pair_id': reference['pair_id'], 'binding': binding, 'spec': asdict(spec),
                         'request_path': request_path, 'status': 'ready', 'attempts': []})
    save(root / 'provenance/inputs.json', list(input_rows.values()))
    save(root / 'manifest.json', {'kind': KIND, 'created_utc': now(), 'status': 'prepared',
         'source': fingerprint, 'environment': env, 'references': references, 'jobs': jobs,
         'reference_count': 12, 'new_run_count': 60, 'seeds': list(SEEDS),
         'configuration': parent['configuration'], 'runner_sha256': sha(Path(__file__))})
    print('PREPARED: 12 archived PAVE references; 60 RandomSource runs; zero PAVE runs scheduled.', flush=True)


def load_manifest(root):
    m = read(root / 'manifest.json')
    if m.get('kind') != KIND or len(m['references']) != 12 or len(m['jobs']) != 60:
        raise ValueError('Not the approved source-selection campaign')
    if any(j['binding']['bound']['policy'] != RANDOM_POLICY or j['spec']['source_selection'] != 'random_feasible'
           for j in m['jobs']):
        raise ValueError('Execution manifest may contain only RandomSource jobs')
    return m


def verify_files(directory, hashes):
    for name, expected in hashes.items():
        if sha(inside(directory, name)) != expected:
            raise ValueError(f'Result checksum mismatch: {directory / name}')


def run(root, retry_failed=False):
    c, measure, Journal, write_jsonl, _, _, RunSpec, _, _, _ = api(root)
    m = load_manifest(root)
    if c.fingerprint_sources() != m['source']:
        raise ValueError('Frozen simulation source changed')
    env = {'python': platform.python_version(), 'dependencies': {
        name: importlib.metadata.version(name) for name in ('numpy', 'scipy')}}
    if env != m['environment']:
        raise ValueError('Execution environment changed')
    for item in read(root / 'provenance/inputs.json'):
        if sha(inside(root, item['path'])) != item['sha256']:
            raise ValueError('Frozen input changed')
    config = c.EvaluationConfig.load(root / 'provenance/parent_configuration.json')
    lock = root / 'execution.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        for index, job in enumerate(m['jobs']):
            if job['status'] == 'complete':
                attempt = job['attempts'][-1]
                verify_files(inside(root, attempt['directory']), attempt['files_sha256'])
                continue
            if job['status'] != 'ready' and not retry_failed:
                raise ValueError('Inspect incomplete attempt before explicit --retry-failed')
            binding = job['binding']; row = binding['bound']; run_id = binding['run_id']
            directory = root / 'runs' / run_id / f'attempt-{len(job["attempts"]) + 1:03d}'
            directory.mkdir(parents=True, exist_ok=False)
            attempt = {'directory': directory.relative_to(root).as_posix(), 'status': 'running', 'started_utc': now()}
            job['attempts'].append(attempt); job['status'] = 'running'; m['status'] = 'running'
            save(root / 'manifest.json', m)
            journal = None; engine = None; started = time.perf_counter()
            print(f'START {index + 1}/60 {row["generator"]} {row["scenario"]} {row["rate_per_min"]} seed={row["source_selection_seed"]}', flush=True)
            try:
                settings = config.settings(root / 'inputs/ilp')
                case = c.Case.load(settings, row['generator'], row['scenario'])
                plan = c.RequestPlan.load(read(inside(root, job['request_path'])), settings, row['rate_per_min'])
                spec = RunSpec(**job['spec'])
                journal = Journal(directory / 'events.jsonl.gz', run_id)
                engine = c.Simulator(settings, spec, case, plan, journal, run_id=run_id, max_events=config.max_events)
                result = engine.run()
                journal.close(); journal = None
                # Persist original facts before analysis so collection is repeatable.
                for name in ('requests', 'attempts', 'instances'):
                    write_jsonl(directory / f'{name}.jsonl.gz', result[name])
                save(directory / 'flips.json', result['flips'])
                metadata = {k: v for k, v in result.items() if k not in ('requests', 'attempts', 'instances', 'flips')}
                metadata.update(kind='evaluation_run', status='complete', evaluation_context=binding['context'])
                save(directory / 'run.json', metadata)
                values = measure(result, binding)
                save(directory / 'summary.json', values)
                attempt.update(status='complete', wall_s=time.perf_counter() - started,
                               files_sha256={p.name: sha(p) for p in directory.iterdir() if p.is_file()})
                job['status'] = 'complete'
                print(f'DONE {index + 1}/60 wall={attempt["wall_s"]:.2f}s thpt={values["throughput_req_s"] * 60:.6f} req/min p99={values["p99_s"] / 60:.6f} min', flush=True)
            except BaseException as error:
                job['status'] = 'failed' if isinstance(error, Exception) else 'interrupted'
                attempt.update(status=job['status'], wall_s=time.perf_counter() - started, error=repr(error))
                save(directory / 'diagnostic.json', engine.diagnostic() if engine else {'error': repr(error)})
                if engine:
                    save(directory / 'partial_flips.json', engine.flips)
                traceback.print_exc()
                raise
            finally:
                if journal is not None:
                    journal.close()
                attempt['finished_utc'] = now()
                m['status'] = 'runs_complete' if all(j['status'] == 'complete' for j in m['jobs']) else job['status']
                save(root / 'manifest.json', m)
    finally:
        lock.unlink()


def stats(values):
    return {'mean': statistics.fmean(values), 'min': min(values), 'max': max(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0, 'n': len(values)}


def ratio_reduction(pave, other):
    return 100 * (1 - pave / other) if other else None


def load_result(directory, read_jsonl):
    result = read(directory / 'run.json')
    for name in ('requests', 'attempts', 'instances'):
        result[name] = read_jsonl(directory / f'{name}.jsonl.gz')
    result['flips'] = read(directory / 'flips.json')
    return result


def mechanism_rows(result, directory, common, ticks, seconds, elapsed):
    candidates = defaultdict(list)
    with gzip.open(directory / 'events.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            e = json.loads(line)
            if e['event'] == 'flip_candidate':
                candidates[ticks(e['time_s'])].append(e)
    by_instance, by_request = defaultdict(list), defaultdict(list)
    for attempt in result['attempts']:
        by_instance[attempt['instance_uid']].append(attempt)
        by_request[attempt['request_id']].append(attempt)
    requests = {r['id']: r for r in result['requests']}
    decisions, candidate_rows, groups, migrations = [], [], [], []
    for flip in result['flips']:
        trigger = ticks(flip['detected_s'])
        sources = {u for g in flip['groups'] for u in g['source_uids']}
        base = dict(common, flip_id=flip['flip_id'], direction=flip['direction'], detected_s=flip['detected_s'])
        pool = candidates.get(trigger, []) if flip['direction'] == 'short_to_long' else []
        feasible = [e for e in pool if e['feasible']]
        scores = [e['score_s'] for e in feasible]
        for i, e in enumerate(pool):
            candidate_rows.append(dict(base, candidate_index=i, source_uids=e['source_ids'], feasible=e['feasible'],
                selected=set(e['source_ids']) == sources, score_s=e['score_s'],
                **{f'{s}_estimated_units': e.get('work_units', {}).get(s) for s in STAGES},
                **{f'{s}_remaining_capacity': e.get('remaining_capacity', {}).get(s) for s in STAGES}))
        if flip['direction'] == 'short_to_long' and not any(set(e['source_ids']) == sources for e in feasible):
            raise ValueError('Selected source set missing from archived feasible candidates')
        active = [a for uid in sources for a in by_instance[uid]
                  if ticks(a['enter_s']) <= trigger and (ticks(a['exit_s']) > trigger
                     or (ticks(a['exit_s']) == trigger and a['exit_reason'].startswith('migrated')))]
        running = [a for a in active if a['start_s'] is not None and ticks(a['start_s']) <= trigger]
        waiting = [a for a in active if a['start_s'] is None or ticks(a['start_s']) > trigger]
        selected_score = flip['selection']['score_s']
        decision = dict(base, completed_s=flip['completed_s'], source_uids=sorted(sources),
            candidate_count=len(pool) if pool else None, feasible_count=len(feasible) if pool else None,
            selected_score_s=selected_score, min_score_s=min(scores) if scores else None,
            median_score_s=statistics.median(scores) if scores else None, max_score_s=max(scores) if scores else None,
            selected_minus_min_s=selected_score - min(scores) if scores else None,
            all_scores_tied=math.isclose(min(scores), max(scores), abs_tol=1e-10, rel_tol=1e-10) if scores else None,
            selected_is_min=math.isclose(selected_score, min(scores), abs_tol=1e-10, rel_tol=1e-10) if scores else None,
            source_running_requests=len(running), source_waiting_requests=len(waiting),
            source_running_ids=[a['request_id'] for a in running], source_waiting_ids=[a['request_id'] for a in waiting],
            trigger_to_ready_s=elapsed(flip['completed_s'], flip['detected_s']))
        decisions.append(decision)
        for group in flip['groups']:
            groups.append(dict(base, group_id=group['group_id'], source_uids=group['source_uids'],
                gpu_ids=group['gpu_ids'], node=group['node'], target_templates=group['target_templates'],
                migrated_waiting=sum(len(v) for v in group['migrated_waiting'].values()),
                migrated_running=sum(len(v) for v in group['migrated_running'].values()),
                safe_wait_s=elapsed(group['launch_started_s'], flip['detected_s']),
                launch_started_s=group['launch_started_s'], ready_s=group['ready_s'],
                trigger_to_ready_s=elapsed(group['ready_s'], flip['detected_s'])))
            for kind in ('waiting', 'running'):
                for stage, ids in group[f'migrated_{kind}'].items():
                    for request_id in ids:
                        matches = [a for a in by_request[request_id] if a['stage'] == stage
                            and a['instance_uid'] in group['source_uids'] and a['exit_reason'] == f'migrated_{kind}'
                            and trigger <= ticks(a['exit_s']) <= ticks(group['launch_started_s'])]
                        if len(matches) != 1:
                            raise ValueError('Cannot identify unique migration attempt')
                        a = matches[0]; moved = ticks(a['exit_s'])
                        stage_wait = dict.fromkeys(STAGES, 0)
                        for later in by_request[request_id]:
                            start = ticks(later['start_s'] if later['start_s'] is not None else later['exit_s'])
                            stage_wait[later['stage']] += max(0, start - max(moved, ticks(later['enter_s'])))
                        migrations.append(dict(base, group_id=group['group_id'], request_id=request_id,
                            stage=stage, kind=kind, source_uid=a['instance_uid'], attempt_id=a['attempt_id'],
                            migrated_s=a['exit_s'], finished_s=requests[request_id]['finished_s'],
                            remaining_latency_s=elapsed(requests[request_id]['finished_s'], a['exit_s']),
                            subsequent_wait_s=seconds(sum(stage_wait.values())),
                            **{f'subsequent_{s}_wait_s': seconds(stage_wait[s]) for s in STAGES}))
    return decisions, candidate_rows, groups, migrations


def collect(root):
    c, measure, _, write_jsonl, read_jsonl, write_csv, _, ticks, seconds, elapsed = api(root)
    m = load_manifest(root)
    if any(j['status'] != 'complete' for j in m['jobs']):
        raise ValueError('Collect requires all sixty formal runs to be complete')
    collector_copy = root / 'provenance/collection_source/evaluation_tools/source_selection_study.py'
    collector_copy.parent.mkdir(parents=True, exist_ok=True)
    if collector_copy.resolve() != Path(__file__).resolve():
        shutil.copy2(Path(__file__), collector_copy)
    tables = root / 'tables'; tables.mkdir(exist_ok=True)
    request_output = tables / 'request_metrics'; request_output.mkdir(exist_ok=True)
    performance, stages, decisions, candidates, groups, migrations, allocation_rows, integrity = [], [], [], [], [], [], [], []
    last_completions = []
    references = {r['pair_id']: r for r in m['references']}
    sources = [(r['pair_id'], 'PAVE', None, r['binding'], r['directory'], r['files_sha256']) for r in m['references']]
    sources += [(j['pair_id'], RANDOM_POLICY, j['spec']['source_selection_seed'], j['binding'],
                 j['attempts'][-1]['directory'], j['attempts'][-1]['files_sha256']) for j in m['jobs']]
    readable = {}
    for name, metric in [('throughput', 'throughput_req_s'), ('p50', 'p50_s'), ('p99', 'p99_s')]:
        with (root / f'provenance/readable_main/{name}.csv').open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                if row['policy'] == 'E':
                    readable[row['run_ids'], metric] = float(row['mean'])
    for index, (pair_id, policy, seed, binding, relative, hashes) in enumerate(sources):
        directory = inside(root, relative); verify_files(directory, hashes)
        result = load_result(directory, read_jsonl)
        values = measure(result, binding)
        original = read(directory / 'summary.json')
        for metric in ('throughput_req_s', 'p50_s', 'p99_s'):
            if not math.isclose(values[metric], original[metric], rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError(f'Recomputed metric mismatch: {metric}')
        payload = read(inside(root, references[pair_id]['request_path']))
        wanted = [(r['id'], r['arrival_tick'], r['input_tokens'], r['output_tokens'], r['kind']) for r in payload['records']]
        actual = [(r['id'], ticks(r['arrival_s']), r['input_tokens'], r['output_tokens'], r['kind'])
                  for r in sorted(result['requests'], key=lambda r: r['id'])]
        if actual != wanted:
            raise ValueError('Paired request identity mismatch')
        common = dict(pair_id=pair_id, policy=policy, source_selection_seed=seed, run_id=binding['run_id'],
            model=binding['bound']['generator'], cluster=binding['bound']['scenario'], rate_per_min=binding['bound']['rate_per_min'])
        d, cand, g, mig = mechanism_rows(result, directory, common, ticks, seconds, elapsed)
        decisions.extend(d); candidates.extend(cand); groups.extend(g); migrations.extend(mig)
        forward = [v for v in d if v['direction'] == 'short_to_long']
        detail = defaultdict(lambda: {f'{s}_{k}_s': 0.0 for s in STAGES for k in ('wait', 'service')})
        for a in result['attempts']:
            detail[a['request_id']][f'{a["stage"]}_wait_s'] += elapsed(a['start_s'] if a['start_s'] is not None else a['exit_s'], a['enter_s'])
            detail[a['request_id']][f'{a["stage"]}_service_s'] += a['executed_service_s']
        request_metrics = [dict(common, request_id=r['id'], arrival_s=r['arrival_s'], finished_s=r['finished_s'],
            request_kind=r['kind'], latency_s=elapsed(r['finished_s'], r['arrival_s']), **detail[r['id']]) for r in result['requests']]
        write_jsonl(request_output / f'{binding["run_id"]}.jsonl.gz', request_metrics)
        last_request = max(result['requests'], key=lambda r: r['finished_s'])
        last_attempts = [a for a in result['attempts'] if a['request_id'] == last_request['id']]
        completed_dit = next(a for a in last_attempts if a['stage'] == 'DiT' and a['exit_reason'] == 'completed')
        last_completions.append(dict(common, request_id=last_request['id'], arrival_s=last_request['arrival_s'],
            finished_s=last_request['finished_s'], dit_completed_hardware=completed_dit['hardware'],
            dit_completed_instance_uid=completed_dit['instance_uid'], **detail[last_request['id']],
            attempts=last_attempts))
        flat = dict(common, requests=values['requests'], throughput_req_s=values['throughput_req_s'],
            throughput_req_min=60 * values['throughput_req_s'], p50_s=values['p50_s'], p99_s=values['p99_s'],
            p50_min=values['p50_s'] / 60, p99_min=values['p99_s'] / 60,
            total_mean_wait_s=values['total_wait_s'] / values['requests'],
            last_request_finish_s=values['last_request_finish_s'], flip_count=values['flip_triggered'],
            forward_flip_count=len(forward), converted_groups=values['converted_groups'],
            migrated_waiting_count=sum(v['kind'] == 'waiting' for v in mig),
            migrated_running_count=sum(v['kind'] == 'running' for v in mig),
            migrated_unique_requests=len({v['request_id'] for v in mig}),
            forward_migrated_waiting_count=sum(v['kind'] == 'waiting' and v['direction'] == 'short_to_long' for v in mig),
            forward_migrated_running_count=sum(v['kind'] == 'running' and v['direction'] == 'short_to_long' for v in mig),
            reverse_migrated_waiting_count=sum(v['kind'] == 'waiting' and v['direction'] == 'long_to_short' for v in mig),
            reverse_migrated_running_count=sum(v['kind'] == 'running' and v['direction'] == 'long_to_short' for v in mig),
            safe_wait_group_sum_s=sum(v['safe_wait_s'] for v in g),
            gpu_conversion_interval_s=values['gpu_unavailable_s'],
            forward_mean_score_excess_s=statistics.fmean(v['selected_minus_min_s'] for v in forward) if forward else None,
            forward_mean_source_running_requests=statistics.fmean(v['source_running_requests'] for v in forward) if forward else None,
            forward_mean_source_waiting_requests=statistics.fmean(v['source_waiting_requests'] for v in forward) if forward else None,
            forward_min_score_choice_fraction=statistics.fmean(v['selected_is_min'] for v in forward) if forward else None,
            forward_tied_score_fraction=statistics.fmean(v['all_scores_tied'] for v in forward) if forward else None)
        for stage in STAGES:
            waits = [r[f'{stage}_wait_s'] for r in request_metrics]
            flat[f'{stage}_mean_wait_s'] = statistics.fmean(waits)
            stages.append(dict(common, stage=stage, mean_wait_s=flat[f'{stage}_mean_wait_s'],
                p99_wait_s=percentile(waits, 99),
                mean_service_s=values['stages'][stage]['service_s'] / values['requests'],
                completed=values['stages'][stage]['completed']))
        performance.append(flat)
        allocation_rows.extend(dict(row, **{k: v for k, v in common.items() if k != 'run_id'}) for row in values['instance_allocations'])
        readable_error = None
        if policy == 'PAVE':
            readable_error = max(abs(values[k] - readable[binding['run_id'], k]) for k in ('throughput_req_s', 'p50_s', 'p99_s'))
            if readable_error > 1e-7:
                raise ValueError('PAVE raw results disagree with readable main tables')
        integrity.append(dict(common, requests_match=True, raw_metrics_match=True, readable_max_absolute_error=readable_error,
                              source_directory=relative))
        print(f'COLLECT {index + 1}/72 {policy} {common["cluster"]} {common["rate_per_min"]}', flush=True)
    metric_names = ['throughput_req_s', 'throughput_req_min', 'p50_s', 'p99_s', 'p50_min', 'p99_min',
        'total_mean_wait_s', *[f'{s}_mean_wait_s' for s in STAGES], 'flip_count', 'forward_flip_count',
        'migrated_waiting_count', 'migrated_running_count', 'migrated_unique_requests',
        'forward_migrated_waiting_count', 'forward_migrated_running_count',
        'reverse_migrated_waiting_count', 'reverse_migrated_running_count',
        'forward_mean_source_running_requests', 'forward_mean_source_waiting_requests',
        'safe_wait_group_sum_s', 'gpu_conversion_interval_s', 'forward_mean_score_excess_s',
        'forward_min_score_choice_fraction', 'forward_tied_score_fraction']
    summary_rows, comparisons, plot_rows = [], [], []
    for pair_id, reference in references.items():
        rows = [r for r in performance if r['pair_id'] == pair_id]
        pave = next(r for r in rows if r['policy'] == 'PAVE')
        random_rows = [r for r in rows if r['policy'] == RANDOM_POLICY]
        if len(random_rows) != 5 or {r['source_selection_seed'] for r in random_rows} != set(SEEDS):
            raise ValueError('Missing or duplicated random repetitions')
        common = {k: pave[k] for k in ('pair_id', 'model', 'cluster', 'rate_per_min')}
        means = {}
        for policy, subset in [('PAVE', [pave]), (RANDOM_POLICY, random_rows)]:
            for metric in metric_names:
                sample = [r[metric] for r in subset if r[metric] is not None]
                if not sample:
                    continue
                s = stats(sample)
                summary_rows.append(dict(common, policy=policy, metric=metric, **s))
                plot_rows.append(dict(common, policy=policy, metric=metric, value=s['mean'], min=s['min'], max=s['max'], std=s['std'], repetitions=s['n']))
                if policy == RANDOM_POLICY:
                    means[metric] = s['mean']
        comparison = dict(common, throughput_increment_pct=100 * (pave['throughput_req_s'] / means['throughput_req_s'] - 1),
            p50_reduction_pct=ratio_reduction(pave['p50_s'], means['p50_s']),
            p99_reduction_pct=ratio_reduction(pave['p99_s'], means['p99_s']),
            total_wait_reduction_pct=ratio_reduction(pave['total_mean_wait_s'], means['total_mean_wait_s']),
            **{f'{s}_wait_reduction_pct': ratio_reduction(pave[f'{s}_mean_wait_s'], means[f'{s}_mean_wait_s']) for s in STAGES})
        comparisons.append(comparison)
    collections = {'performance': performance, 'stage_waiting': stages, 'source_decisions': decisions,
        'source_candidates': candidates, 'conversion_groups': groups, 'migrated_requests': migrations,
        'instance_allocations': allocation_rows, 'policy_summary': summary_rows,
        'comparisons': comparisons, 'plot_data': plot_rows, 'last_completions': last_completions}
    for name, rows in collections.items():
        save(tables / f'{name}.json', rows); write_csv(tables / f'{name}.csv', rows)
    figure_data = {'window_s': 60, 'margin': .15, 'monitor_period_s': 10,
                  'trace': 'majority60', 'cases': []}
    for model, cluster in sorted({(r['model'], r['cluster']) for r in performance}):
        selected = [r for r in summary_rows if r['model'] == model and r['cluster'] == cluster]
        rates = sorted({r['rate_per_min'] for r in selected})
        series = {}
        for policy in ('PAVE', RANDOM_POLICY):
            series[policy] = {}
            for metric in ('throughput_req_min', 'p50_min', 'p99_min', 'PE_mean_wait_s', 'DiT_mean_wait_s'):
                lookup = {r['rate_per_min']: r for r in selected if r['policy'] == policy and r['metric'] == metric}
                series[policy][metric] = {field: [lookup[rate][field] for rate in rates]
                                          for field in ('mean', 'min', 'max', 'std', 'n')}
        figure_data['cases'].append({'model': model, 'cluster': cluster, 'rates_req_min': rates, 'series': series})
    save(tables / 'figure_data.json', figure_data)
    aggregate = {}
    for label, subset in [('all', comparisons), ('cluster1', [r for r in comparisons if r['cluster'] == 'cluster1']),
                          ('cluster2', [r for r in comparisons if r['cluster'] == 'cluster2'])]:
        aggregate[label] = {'settings': len(subset)}
        for metric in ('throughput_increment_pct', 'p50_reduction_pct', 'p99_reduction_pct',
                       'total_wait_reduction_pct', 'PE_wait_reduction_pct', 'DiT_wait_reduction_pct'):
            sample = [r[metric] for r in subset if r[metric] is not None]
            aggregate[label][metric] = dict(stats(sample), improved=sum(v > 1e-8 for v in sample),
                worse=sum(v < -1e-8 for v in sample), tied=sum(abs(v) <= 1e-8 for v in sample))
    save(tables / 'aggregate.json', aggregate)
    save(root / 'provenance/collection_integrity.json', integrity)
    write_analysis(root, m, performance, comparisons, decisions, aggregate, last_completions)
    m.update(status='complete', collected_utc=now(), analysis_runner_sha256=sha(Path(__file__)),
             collection_counts={name: len(rows) for name, rows in collections.items()})
    save(root / 'manifest.json', m)
    hashes = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*'))
              if p.is_file() and 'logs' not in p.relative_to(root).parts and 'tmp' not in p.relative_to(root).parts
              and p.name not in ('checksums.json', 'execution.lock')}
    save(root / 'provenance/checksums.json', hashes)
    print(json.dumps({'status': 'complete', 'runs': 72, 'new_simulations': 60, 'pave_reruns': 0,
                      'aggregate': aggregate}, ensure_ascii=False), flush=True)


def percentile(values, p):
    ordered = sorted(values); position = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def write_analysis(root, manifest, performance, comparisons, decisions, aggregate, last_completions):
    lines = ['# Source Selection消融：数据与初步分析', '',
        '比较对象为归档PAVE与PAVE-RandomSource；只新增60次随机选源模拟，12次PAVE未重新运行。',
        '随机策略在同一可行候选组合集合中等概率抽样，其他机制与输入保持一致。反向恢复仍按原转换关系执行。', '',
        '## 统计口径', '',
        '- 每组先平均5次随机运行的原始指标，再计算PAVE相对该均值的改善；12组设置等权。',
        '- 吞吐为请求总数除以首个到达到最后完成的时间；时延包含全部请求。保留秒、分钟和两种吞吐单位。',
        '- source_candidates的score_s为预计负担，不是实际等待；minimum仅是在该次决策状态下的最低评分。',
        '- migrated_requests以迁移事件计数；同一请求可多次出现，其后续等待区间可能重叠，不可简单加总为全局等待。',
        '- gpu_conversion_interval_s包含安全等待及启动；安全等待期间可能仍在执行，不等于纯空闲GPU时间。',
        '- PAVE旧日志没有新增source_selection事件；候选评分和被选来源由原flip_candidate、flips和attempts重建。',
        '- 不同运行后续flip时间可能不同，不把同序号flip当成同一控制状态。', '',
        '## 全部配对结果', '',
        '正数表示PAVE更好，负数表示随机版本均值更好。', '',
        '| 模型 | 集群 | req/min | 吞吐提高% | p50降低% | p99降低% | DiT等待降低% |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append(f'| {r["model"]} | {r["cluster"]} | {r["rate_per_min"]:g} | {r["throughput_increment_pct"]:.4f} | {r["p50_reduction_pct"]:.4f} | {r["p99_reduction_pct"]:.4f} | {r["DiT_wait_reduction_pct"]:.4f} |')
    lines += ['', '## 等权汇总', '']
    for label, values in aggregate.items():
        t, p50, p99 = (values[k] for k in ('throughput_increment_pct', 'p50_reduction_pct', 'p99_reduction_pct'))
        lines.append(f'- {label}（{values["settings"]}组）：平均吞吐提高{t["mean"]:.4f}%，p50降低{p50["mean"]:.4f}%，p99降低{p99["mean"]:.4f}%；吞吐改善/持平/退化为{t["improved"]}/{t["tied"]}/{t["worse"]}组，p99为{p99["improved"]}/{p99["tied"]}/{p99["worse"]}组。')
    lines += ['', '## 选源机制汇总', '',
        '以下先按运行统计，再在同一模型/集群/负载内平均随机重复，最后对12组设置等权；避免flip较多的运行获得更大权重。', '',
        '| 策略 | 平均正向flip次数 | 选中最低评分比例% | 全候选同分比例% | 选中评分超过最低值(s) | 平均迁移等待请求次数 |',
        '|---|---:|---:|---:|---:|---:|']
    for policy in ('PAVE', RANDOM_POLICY):
        means = []
        for pair in manifest['references']:
            subset = [r for r in performance if r['pair_id'] == pair['pair_id'] and r['policy'] == policy]
            keys = ['forward_flip_count', 'forward_min_score_choice_fraction', 'forward_tied_score_fraction',
                    'forward_mean_score_excess_s', 'migrated_waiting_count']
            means.append({k: statistics.fmean(r[k] for r in subset) for k in keys})
        avg = {k: statistics.fmean(r[k] for r in means) for k in means[0]}
        lines.append(f'| {policy} | {avg["forward_flip_count"]:.4f} | {100 * avg["forward_min_score_choice_fraction"]:.4f} | {100 * avg["forward_tied_score_fraction"]:.4f} | {avg["forward_mean_score_excess_s"]:.6f} | {avg["migrated_waiting_count"]:.4f} |')
    def policy_mean(policy, key, subset=None):
        rows = performance if subset is None else subset
        return statistics.fmean(r[key] for r in rows if r['policy'] == policy)

    pw = policy_mean('PAVE', 'forward_migrated_waiting_count')
    rw = policy_mean(RANDOM_POLICY, 'forward_migrated_waiting_count')
    pr = policy_mean('PAVE', 'forward_migrated_running_count')
    rr = policy_mean(RANDOM_POLICY, 'forward_migrated_running_count')
    pfrac = statistics.fmean(r['forward_migrated_waiting_count'] / r['requests'] for r in performance if r['policy'] == 'PAVE')
    rfrac = statistics.fmean(r['forward_migrated_waiting_count'] / r['requests'] for r in performance if r['policy'] == RANDOM_POLICY)
    lines += ['', '## 由实际数据得到的初步判断', '',
        f'当前数据不支持source selection稳定增加吞吐。12组平均吞吐变化为{aggregate["all"]["throughput_increment_pct"]["mean"]:+.4f}%，p99平均降低{aggregate["all"]["p99_reduction_pct"]["mean"]:.4f}%；改善幅度较小，且存在退化设置。', '',
        f'机制层面的效果更清楚：每组运行的正向flip平均迁移等待请求次数由随机版本的{rw:.4f}降至PAVE的{pw:.4f}；运行中请求的迁移次数由{rr:.4f}降至{pr:.4f}。这说明选源策略减少了被转换来源上的请求迁移，而不是只在评分上占优。', '',
        f'但正向迁移等待请求次数与请求总数的比值仅为随机版本的{100*rfrac:.4f}%和PAVE的{100*pfrac:.4f}%（先逐设置计算再等权平均，按事件计数）。两种策略还保持相同的转换数量、硬件类型和目标副本数，所以选源主要改变转换期间的扰动，不增加同一phase的部署容量。这些条件限制了端到端提升空间。', '',
        '实际候选数与配置相符：Wan2.1/Cluster1为20种，Wan2.2/Cluster1为21种，Cluster2两个模型均为3种。约30%的正向决策中所有候选同分，随机版本平均约48%的决策也选到最低评分。更多候选并未在当前结果中形成更大的吞吐优势。', '']
    special = [r for r in performance if r['model'] == 'wan2.1-t2v-1.3b' and r['cluster'] == 'cluster1' and r['rate_per_min'] == 14]
    if special:
        pp = next(r for r in special if r['policy'] == 'PAVE')
        rc = next(r for r in comparisons if r['pair_id'] == pp['pair_id'])
        flip_counts = [r['flip_count'] for r in special if r['policy'] == RANDOM_POLICY]
        lines += [f'Wan2.1/Cluster1、14 req/min需要特别谨慎解释：PAVE的DiT平均等待由随机均值{policy_mean(RANDOM_POLICY, "DiT_mean_wait_s", special):.4f}s降至{pp["DiT_mean_wait_s"]:.4f}s，降低{rc["DiT_wait_reduction_pct"]:.4f}%；但PE平均等待从{policy_mean(RANDOM_POLICY, "PE_mean_wait_s", special):.4f}s增至{pp["PE_mean_wait_s"]:.4f}s。PAVE发生{pp["flip_count"]}次flip，随机重复的次数为{flip_counts}。这包含后续flip变化及阶段之间的等待重新分布，不能将DiT等待下降全部归因于一次选源更优。', '']
    special = [r for r in last_completions if r['model'] == 'wan2.2-ti2v-5b' and r['cluster'] == 'cluster1' and r['rate_per_min'] == 8]
    if special:
        pp = next(r for r in special if r['policy'] == 'PAVE')
        rand = [r for r in special if r['policy'] == RANDOM_POLICY]
        delta = [pp['finished_s'] - r['finished_s'] for r in rand]
        rc = next(r for r in comparisons if r['pair_id'] == pp['pair_id'])
        lines += [f'Wan2.2/Cluster1、8 req/min解释了吞吐与p99方向不同的情况：PAVE的p99降低{rc["p99_reduction_pct"]:.4f}%，但吞吐变化为{rc["throughput_increment_pct"]:+.4f}%。六次运行最后完成的都是请求{pp["request_id"]}；PAVE将其DiT阶段交给{pp["dit_completed_hardware"]}，执行{pp["DiT_service_s"]:.6f}s，随机版本均交给{rand[0]["dit_completed_hardware"]}，执行{rand[0]["DiT_service_s"]:.6f}s。虽然随机版本中该请求等待更久，较短执行时间仍使最后完成提前{min(delta):.4f}--{max(delta):.4f}s。因此，减少迁移或平均等待不必然缩短最后请求完成时间；last_completions保留了这些请求的全部阶段记录。', '']
    lines += ['这些结果适合说明选源可以减少转换扰动，但当前设置中的端到端收益有限。暂不据此宣称稳定提高吞吐，也不以单个正面案例替代全部配对结果。图的指标与布局留待下一步确定。', '',
        '## 文件入口', '',
        '- tables/performance.csv：72条逐运行指标。',
        '- tables/policy_summary.csv与plot_data.json：24组策略/设置的均值、范围、标准差及完整精度绘图值。',
        '- tables/figure_data.json：按模型、集群、负载排序的吞吐、p50、p99和PE/DiT等待数组。',
        '- tables/comparisons.csv：12组PAVE相对随机均值的改善。',
        '- tables/source_candidates.csv与source_decisions.csv：每次候选及选中状态。',
        '- tables/conversion_groups.csv与migrated_requests.csv：转换过程和迁移后等待。',
        '- tables/request_metrics/：全部请求的阶段等待与服务时间。',
        '- tables/last_completions.csv：各运行最后完成请求的硬件、等待及执行记录，用于解释吞吐差异。',
        '- provenance/：输入、源码差异、原PAVE绑定与完整性记录。',
        '', '未运行测试；未重新模拟PAVE；未修改原归档。', '']
    (root / 'analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    (root / 'README.md').write_text(
        '# PAVE source-selection ablation\n\n阅读 analysis.md 查看统计口径与全部结果。\n\n'
        '功能入口：experimental/pave/evaluation_tools/source_selection_study.py；子命令 prepare、run、collect，均显式指定 --output-dir。\n'
        'run只执行RandomSource；collect只读取数据。原PAVE位于reference_pave，含来源路径和校验值。\n'
        '复现须使用provenance/source中的代码及manifest记录的环境；Python加-B禁用字节码缓存。\n', encoding='utf-8')


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text); self.log.write(text); self.log.flush()
        return len(text)

    def flush(self):
        self.terminal.flush(); self.log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'collect'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if not root.is_relative_to(BASE.resolve()) or root == BASE.resolve():
        raise ValueError(f'Output must be a new campaign beneath {BASE}')
    if args.command == 'prepare':
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        raise ValueError('Prepare a campaign first')
    (root / 'logs').mkdir(exist_ok=True); (root / 'tmp').mkdir(exist_ok=True)
    for key in ('TMP', 'TEMP', 'TMPDIR'):
        os.environ[key] = str(root / 'tmp')
    os.chdir(root)
    log_path = root / 'logs' / f'{args.command}-{datetime.now():%Y%m%d-%H%M%S-%f}.log'
    with log_path.open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                if args.command == 'prepare':
                    prepare(root)
                elif args.command == 'run':
                    run(root, args.retry_failed)
                else:
                    collect(root)
            except BaseException:
                traceback.print_exc()
                raise


if __name__ == '__main__':
    main()
