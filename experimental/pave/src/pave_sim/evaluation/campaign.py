"""Content-addressed preparation and foreground whole-run retry orchestration."""
from __future__ import annotations

import copy
import json
import importlib.metadata
import os
import platform
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from pave_ilp.profiles import GENERATORS, digest
from ..engine import Simulator
from ..inputs import Case
from ..records import Journal, read_jsonl, write_csv, write_json, write_jsonl
from ..timing import SEMANTICS, ticks
from .config import EvaluationConfig, compile_policy
from .matrix import bind, in_phase, references, slots
from .traces import RequestPlan, build_trace, construct_requests, file_sha

SCHEMA = 2


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text('utf-8'))


def atomic_json(path, data):
    temporary = path.with_name(path.name + '.tmp')
    write_json(temporary, data)
    # Windows indexers can briefly hold the old manifest without delete sharing.
    # Retry only sharing/access errors; never remove the old file or weaken checks.
    for attempt in range(7):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 6:
                raise
            time.sleep(0.02 * 2 ** attempt)


def fingerprint_sources():
    root = Path(__file__).resolve().parents[1]
    analysis_names = {'metrics.py', 'reports.py', 'provenance.py', 'cli.py',
                      'evaluation/analysis.py', 'evaluation/cli.py'}
    runtime, analysis = {}, {}
    for path in sorted(root.rglob('*.py')):
        name = path.relative_to(root).as_posix()
        (analysis if name in analysis_names else runtime)['pave_sim/' + name] = file_sha(path)
    for path in sorted((root.parent / 'pave_ilp').rglob('*.py')):
        runtime['pave_ilp/' + path.relative_to(root.parent / 'pave_ilp').as_posix()] = file_sha(path)
    return {'runtime_sha256': digest(runtime), 'analysis_sha256': digest(analysis),
            'runtime_files': runtime, 'analysis_files': analysis}


def request_key(name, rate, seed):
    return digest([name, float(rate), seed])[:24]


def make_context(row, config, manifest):
    key = request_key(row['trace'], row['rate_per_min'], row['request_seed'])
    case = manifest['cases'][row['case_id']]
    static = row['policy'].startswith('Static-')
    effective = {k: row[k] for k in ('generator', 'scenario', 'policy', 'rate_per_min', 'trace', 'request_seed')}
    effective.update(window_s=None if static else float(row['window_s']), margin=None if static else float(row['margin']),
                     scheduler_seed=row['scheduler_seed'] if row['policy'] == 'E-Weighted' else None,
                     monitor_period_s=None if static else float(config.monitor_period_s),
                     input_tokens=config.input_tokens, short_output_tokens=config.short_output_tokens,
                     long_output_tokens=config.long_output_tokens, duration_s=float(config.duration_s),
                     initial_tie=config.initial_tie, arrival_mode='equally_spaced', synthetic=config.synthetic)
    effective['startup_mode'] = None if static else compile_policy(row, config).startup_mode
    context = {'schema_version': SCHEMA, 'effective': effective,
               'requests_sha256': manifest['requests'][key]['requests_sha256'],
               'trace_definition_sha256': manifest['trace_definition_sha256'],
               'source_sha256': case['source_sha256'], 'flip_sha256': case['flip_sha256'],
               'profile_sha256': case['profile_sha256'], 'profile_values_sha256': case['profile_values_sha256'],
               'simulation_source_sha256': manifest['source']['runtime_sha256'],
               'simulation_semantics': SEMANTICS.copy(), 'slo_baselines': case['slo_baselines'],
               'execution_environment': manifest['environment'],
               'trace_statistics': manifest['trace_metadata'][row['trace']]}
    return context, key


def resolve(manifest, config):
    result = {}
    for slot in manifest['slots']:
        row = bind(slot, config, manifest['selections'])
        if row is None:
            result[slot['slot_id']] = {'status': 'waiting_selection'}
            continue
        context, key = make_context(row, config, manifest)
        identity = digest(context)
        run_id = f"{row['case_id']}-{row['policy']}-{identity[:24]}"
        result[slot['slot_id']] = {'status': manifest['runs'].get(run_id, {}).get('status', 'ready'),
                                  'run_id': run_id, 'identity_sha256': identity,
                                  'context': context, 'request_key': key, 'bound': row}
    return result


def write_manifest(root, manifest, config):
    manifest['bindings'] = resolve(manifest, config)
    manifest['status_counts'] = dict(Counter(b['status'] for b in manifest['bindings'].values()))
    manifest['updated_utc'] = now()
    if all(k in manifest['selections'] for k in ('parameters', 'cluster')):
        manifest['references'] = references(config, manifest['slots'], manifest['selections'])
    atomic_json(root / 'manifest.json', manifest)


def definition_hash(manifest):
    return digest({k: manifest[k] for k in ('configuration', 'configuration_sha256', 'source',
                   'simulation_semantics', 'environment', 'slots', 'cases', 'requests', 'input_files', 'trace_definition_sha256', 'trace_metadata')})


def prepare(config: EvaluationConfig, root: Path):
    config.validate()
    if root.exists():
        raise ValueError('Prepare requires a new campaign directory')
    logical = slots(config)
    cases = {(g, c): Case.load(config.settings(), g, c) for g in config.rates for c in config.rates[g]}
    trace = build_trace(config)
    all_rates = sorted({float(r) for groups in config.rates.values() for rates in groups.values() for r in rates})
    ablation_rates = sorted({float(r) for groups in config.rates.values() for c in ('cluster1', 'cluster2') for r in groups[c]})
    payloads = {}
    for rate in all_rates:
        payloads[request_key('majority60', rate, None)] = construct_requests(config, trace, 'majority60', rate)
    for rate in ablation_rates:
        for name, seed in [('majority15', None), ('majority30', None)] + [('mixture', seed) for seed in config.request_seeds]:
            payloads[request_key(name, rate, seed)] = construct_requests(config, trace, name, rate, seed)
    # All expensive validation precedes creation of the official output.
    root.mkdir(parents=True)
    (root / 'inputs/ilp/deployments').mkdir(parents=True)
    (root / 'inputs/ilp/flips').mkdir()
    (root / 'inputs/requests').mkdir()
    (root / 'runs').mkdir()
    manifest = {'schema_version': SCHEMA, 'kind': 'pave_evaluation', 'created_utc': now(),
                'configuration': asdict(config), 'configuration_sha256': digest(config.semantic()),
                'source': fingerprint_sources(), 'simulation_semantics': SEMANTICS.copy(),
                'environment': {'python': platform.python_version(),
                    'dependencies': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy')}},
                'host_platform': platform.platform(),
                'slots': logical, 'selections': {}, 'runs': {}, 'cases': {}, 'requests': {}, 'input_files': {}}
    def track(path):
        manifest['input_files'][path.relative_to(root).as_posix()] = file_sha(path)
    for (generator, scenario), case in cases.items():
        for role, relative in (('source', f'deployments/{case.id}_{config.short_output_tokens}.json'),
                               ('flip', f'flips/{case.id}.json'), ('profile', 'profiles.json')):
            destination = root / 'inputs/ilp' / relative
            shutil.copyfile(case.input_files[role]['path'], destination)
            if file_sha(destination) != case.input_files[role]['sha256']:
                raise ValueError('Input changed while preparing campaign')
            track(destination)
        manifest['cases'][case.id] = {'generator': generator, 'scenario': scenario,
            'source_sha256': case.input_files['source']['sha256'], 'flip_sha256': case.input_files['flip']['sha256'],
            'profile_sha256': case.input_files['profile']['sha256'],
            'profile_values_sha256': case.profiles.catalog.snapshot['values_sha256'],
            'slo_baselines': case.slo_baselines(), 'source_gpu_instances': len(case.source['instances']),
            'target_gpu_instances': len(case.target['instances']), 'cpu_instances': case.source['cpu_te']['instances']}
    definition = {k: v for k, v in trace.items() if k != 'selected_rows'}
    write_json(root / 'inputs/trace.json', definition)
    track(root / 'inputs/trace.json')
    manifest['trace_definition_sha256'] = digest(definition)
    manifest['trace_metadata'] = {f'majority{width}': dict(stats) for width, stats in trace['statistics'].items()}
    manifest['trace_metadata']['mixture'] = {'raw_long_fraction': trace['raw_long_fraction'],
                                             'minute_mean_long_fraction': trace['minute_mean_long_fraction']}
    write_csv(root / 'inputs/hour.csv', trace['selected_rows'])
    track(root / 'inputs/hour.csv')
    for width, rows in trace['bins'].items():
        write_csv(root / f'inputs/majority{width}.csv', rows)
        track(root / f'inputs/majority{width}.csv')
    for key, payload in payloads.items():
        destination = root / 'inputs/requests' / f'{key}.json'
        write_json(destination, payload)
        track(destination)
        manifest['requests'][key] = {k: payload[k] for k in ('trace', 'rate_per_min', 'request_seed', 'requests_sha256', 'long_fraction', 'type_changes')}
        manifest['requests'][key].update(path=destination.relative_to(root).as_posix(), count=len(payload['records']))
    write_json(root / 'configuration.json', asdict(config))
    track(root / 'configuration.json')
    manifest['definition_sha256'] = definition_hash(manifest)
    write_manifest(root, manifest, config)
    write_csv(root / 'slots.csv', [{**s, 'status': manifest['bindings'][s['slot_id']]['status']} for s in logical])
    return {'campaign': str(root), 'slots': len(logical), 'batches': dict(Counter(s['batch'] for s in logical)),
            'status_counts': manifest['status_counts'], 'request_lists': len(payloads), 'formal_runs': 0,
            'trace_statistics': trace['statistics']}


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Archive path escapes campaign: {relative}')
    return path


def load_campaign(root: Path, *, execution=False):
    manifest = read(root / 'manifest.json')
    if manifest.get('schema_version') != SCHEMA or manifest.get('kind') != 'pave_evaluation':
        raise ValueError('Expected a version-2 PAVE evaluation campaign; keep older campaigns with their frozen source')
    if definition_hash(manifest) != manifest.get('definition_sha256'):
        raise ValueError('Frozen campaign definition checksum mismatch')
    config = EvaluationConfig(**manifest['configuration'])
    config.validate()
    if digest(config.semantic()) != manifest['configuration_sha256'] or manifest['slots'] != slots(config):
        raise ValueError('Frozen campaign configuration/slots changed')
    if manifest['simulation_semantics'] != SEMANTICS:
        raise ValueError('Unsupported simulation semantics')
    if execution and fingerprint_sources()['runtime_sha256'] != manifest['source']['runtime_sha256']:
        raise ValueError('Simulation source changed; create a new campaign instead of mixing runs')
    environment = {'python': platform.python_version(),
                   'dependencies': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy')}}
    if execution and environment != manifest['environment']:
        raise ValueError('Execution environment changed; use the frozen versions or a new campaign')
    for name, checksum in manifest['input_files'].items():
        if file_sha(inside(root, name)) != checksum:
            raise ValueError(f'Archived input checksum mismatch: {name}')
    if digest(read(root / 'inputs/trace.json')) != manifest['trace_definition_sha256']:
        raise ValueError('Trace definition digest mismatch')
    for selection in manifest['selections'].values():
        if digest({k: v for k, v in selection.items() if k != 'selection_sha256'}) != selection['selection_sha256']:
            raise ValueError('Selection receipt checksum mismatch')
    # Recompute, never trust persisted binding rows or display references.
    expected = resolve(manifest, config)
    if manifest['bindings'] != expected:
        raise ValueError('Campaign bindings inconsistent with frozen definition')
    return manifest, config


def load_result(root, manifest, binding):
    run_id = binding['run_id']
    run = manifest['runs'].get(run_id)
    if not run or run['status'] != 'complete':
        raise ValueError(f'Missing complete result: {run_id}')
    attempt = run['attempts'][-1]
    directory = inside(root, attempt['path'])
    for name, checksum in attempt['files'].items():
        if file_sha(inside(directory, name)) != checksum:
            raise ValueError(f'Result checksum mismatch: {run_id}/{name}')
    required = {'run.json', 'requests.jsonl.gz', 'attempts.jsonl.gz', 'instances.jsonl.gz', 'flips.json', 'events.jsonl.gz'}
    if not required <= set(attempt['files']):
        raise ValueError(f'Incomplete result files: {run_id}')
    result = read(directory / 'run.json')
    if (result.get('kind') != 'evaluation_run' or result.get('status') != 'complete'
            or result.get('evaluation_context') != binding['context'] or result['run_id'] != run_id):
        raise ValueError(f'Run provenance mismatch: {run_id}')
    cfg = EvaluationConfig(**manifest['configuration'])
    if result['spec'] != asdict(compile_policy(binding['bound'], cfg)):
        raise ValueError(f'Run policy mismatch: {run_id}')
    if result['slo_baselines'] != binding['context']['slo_baselines'] or result['simulation_semantics'] != SEMANTICS:
        raise ValueError(f'Run SLO/semantics mismatch: {run_id}')
    for name in ('requests', 'attempts', 'instances'):
        result[name] = read_jsonl(directory / f'{name}.jsonl.gz')
    result['flips'] = read(directory / 'flips.json')
    request_records = [{'id': r['id'], 'arrival_tick': ticks(r['arrival_s']),
                        'input_tokens': r['input_tokens'], 'output_tokens': r['output_tokens'], 'kind': r['kind']} for r in result['requests']]
    if digest(request_records) != binding['context']['requests_sha256']:
        raise ValueError(f'Executed request list mismatch: {run_id}')
    return result


def run_phase(root: Path, phase: str, *, retry_failed=False):
    if phase not in ('2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.6-low', '2.6-rest'):
        raise ValueError('Unsupported execution phase')
    manifest, config = load_campaign(root, execution=True)
    selected = [s for s in manifest['slots'] if in_phase(s, phase, config)]
    if any(manifest['bindings'][s['slot_id']]['status'] == 'waiting_selection' for s in selected):
        raise ValueError(f'Phase {phase} is waiting for parameter/cluster selection')
    if phase == '2.2':
        for s in manifest['slots']:
            if in_phase(s, '2.1', config):
                load_result(root, manifest, manifest['bindings'][s['slot_id']])
    lock = root / 'execution.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ValueError('Campaign has an execution lock; do not run concurrent writers') from error
    os.close(descriptor)
    executed, reused = [], []
    try:
        for logical in selected:
            binding = manifest['bindings'][logical['slot_id']]
            run_id = binding['run_id']
            previous = manifest['runs'].get(run_id)
            if previous and previous['status'] == 'complete':
                load_result(root, manifest, binding)
                reused.append(run_id)
                continue
            if previous and not retry_failed:
                raise ValueError(f'Run {run_id} has a previous incomplete attempt; use --retry-failed')
            attempts = previous['attempts'] if previous else []
            directory = root / 'runs' / run_id / f'attempt-{len(attempts) + 1:03d}'
            directory.mkdir(parents=True)
            attempt = {'path': directory.relative_to(root).as_posix(), 'started_utc': now(), 'status': 'running'}
            attempts.append(attempt)
            manifest['runs'][run_id] = {'status': 'running', 'attempts': attempts}
            write_manifest(root, manifest, config)
            journal, engine = None, None
            clock = time.perf_counter()
            print(f"START {len(executed) + len(reused) + 1}/{len(selected)} {run_id} {binding['bound']['rate_per_min']} req/min W={binding['bound']['window_s']} M={binding['bound']['margin']}", file=sys.stderr, flush=True)
            try:
                row = binding['bound']
                settings = config.settings(root / 'inputs/ilp')
                case = Case.load(settings, row['generator'], row['scenario'])
                payload = read(inside(root, manifest['requests'][binding['request_key']]['path']))
                request_plan = RequestPlan.load(payload, settings, row['rate_per_min'])
                journal = Journal(directory / 'events.jsonl.gz', run_id)
                engine = Simulator(settings, compile_policy(row, config), case, request_plan, journal,
                                   run_id=run_id, max_events=config.max_events)
                result = engine.run()
                journal.close()
                journal = None
                for name in ('requests', 'attempts', 'instances'):
                    write_jsonl(directory / f'{name}.jsonl.gz', result[name])
                write_json(directory / 'flips.json', result['flips'])
                metadata = {k: v for k, v in result.items() if k not in ('requests', 'attempts', 'instances', 'flips')}
                metadata.update(kind='evaluation_run', status='complete', evaluation_context=binding['context'])
                write_json(directory / 'run.json', metadata)
                from .analysis import measure
                write_json(directory / 'summary.json', measure(result, binding))
                attempt.update(status='complete', files={p.name: file_sha(p) for p in directory.iterdir() if p.is_file()})
                attempt['wall_s'] = time.perf_counter() - clock
                manifest['runs'][run_id]['status'] = 'complete'
                executed.append(run_id)
                print(f"DONE {run_id} {attempt['wall_s']:.3f}s", file=sys.stderr, flush=True)
            except BaseException as error:
                status = 'failed' if isinstance(error, Exception) else 'interrupted'
                attempt.update(status=status, error=str(error), error_type=type(error).__name__)
                manifest['runs'][run_id]['status'] = status
                write_json(directory / 'diagnostic.json', engine.diagnostic() if engine else {'error': str(error)})
                write_json(directory / 'run.json', {'kind': 'evaluation_run', 'status': status,
                           'run_id': run_id, 'evaluation_context': binding['context'], 'error': str(error)})
                raise
            finally:
                if journal is not None:
                    journal.close()
                attempt['finished_utc'] = now()
                write_manifest(root, manifest, config)
    finally:
        lock.unlink()
    return {'phase': phase, 'executed': len(executed), 'reused': len(reused), 'run_ids': executed + reused}
