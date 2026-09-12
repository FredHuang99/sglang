"""Selective whole-run orchestration, using the frozen simulator unchanged.

The per-attempt sequence intentionally follows pave_sim.evaluation.campaign.run_phase.
It is kept here because changing that frozen module invalidates existing run identities.
"""
import os
import sys
import time
from pathlib import Path

from pave_sim.evaluation import campaign as c
from pave_sim.evaluation.analysis import measure
from pave_sim.records import Journal, write_json, write_jsonl
from .selection import classification, validate_plan


def run_selected(root, slot_ids, *, plan=None, retry_failed=False):
    root = Path(root)
    if len(slot_ids) != len(set(slot_ids)) or not slot_ids:
        raise ValueError('An explicit nonempty unique slot list is required')
    lock = root / 'execution.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    executed, reused = [], []
    try:
        manifest, config = c.load_campaign(root, execution=True)
        if plan is not None:
            if plan.get('schema_version')==2:
                from .remaining_plan import validate_plan as validate_closeout
                validate_closeout(plan,manifest,config)
            else:
                validate_plan(plan, manifest, config)
        by_slot = {s['slot_id']: s for s in manifest['slots']}
        if any(s not in by_slot for s in slot_ids):
            raise ValueError('Unknown execution slot')
        selected = [by_slot[s] for s in slot_ids]
        classified={r['slot_id']:r['classification'] for r in plan['rows']} if plan else {}
        if any(classified.get(s['slot_id'],classification(s)) != 'paper_required' for s in selected):
            raise ValueError('Historical/cancelled policy cannot be executed by follow-up tools')
        if any(manifest['bindings'][s['slot_id']]['status'] == 'waiting_selection' for s in selected):
            raise ValueError('Execution is waiting for selection')
        for logical in selected:
            binding = manifest['bindings'][logical['slot_id']]
            run_id = binding['run_id']
            previous = manifest['runs'].get(run_id)
            if previous and previous['status'] == 'complete':
                c.load_result(root, manifest, binding)
                reused.append(run_id)
                continue
            if previous and not retry_failed:
                raise ValueError(f'Run {run_id} has an incomplete attempt; use --retry-failed')
            attempts = previous['attempts'] if previous else []
            directory = root / 'runs' / run_id / f'attempt-{len(attempts)+1:03d}'
            directory.mkdir(parents=True)
            attempt = {'path': directory.relative_to(root).as_posix(), 'started_utc': c.now(), 'status': 'running'}
            attempts.append(attempt)
            manifest['runs'][run_id] = {'status': 'running', 'attempts': attempts}
            c.write_manifest(root, manifest, config)
            journal, engine = None, None
            clock = time.perf_counter()
            print(f'START {len(executed)+len(reused)+1}/{len(selected)} {run_id}', file=sys.stderr, flush=True)
            try:
                row = binding['bound']
                settings = config.settings(root / 'inputs/ilp')
                case = c.Case.load(settings, row['generator'], row['scenario'])
                payload = c.read(c.inside(root, manifest['requests'][binding['request_key']]['path']))
                request_plan = c.RequestPlan.load(payload, settings, row['rate_per_min'])
                journal = Journal(directory / 'events.jsonl.gz', run_id)
                engine = c.Simulator(settings, c.compile_policy(row, config), case, request_plan, journal,
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
                write_json(directory / 'summary.json', measure(result, binding))
                attempt.update(status='complete', files={p.name: c.file_sha(p) for p in directory.iterdir() if p.is_file()})
                attempt['wall_s'] = time.perf_counter()-clock
                manifest['runs'][run_id]['status'] = 'complete'
                executed.append(run_id)
                print(f'DONE {run_id} {attempt["wall_s"]:.3f}s', file=sys.stderr, flush=True)
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
                attempt['finished_utc'] = c.now()
                c.write_manifest(root, manifest, config)
    finally:
        lock.unlink()
    return {'executed': len(executed), 'reused': len(reused), 'run_ids': executed+reused}


def run_batch(root, batch, plan_path, *, retry_failed=False):
    root = Path(root).resolve()
    if batch not in ('f-low','f-rest','d','e'):
        raise ValueError('Use an explicit batch: f-low, f-rest, d or e')
    if any((p / 'provenance/checksums.json').is_file() for p in root.parents):
        raise ValueError('Read-only archive: prepare a separate working campaign for new execution')
    manifest, config = c.load_campaign(root, execution=True)
    plan = c.read(Path(plan_path))
    if plan.get('schema_version')==2:
        from .remaining_plan import validate_plan as validate_closeout, batch_ids
        validate_closeout(plan,manifest,config)
        return run_selected(root,batch_ids(plan,batch),plan=plan,retry_failed=retry_failed)
    if batch in ('remaining','f-rest'):
        raise ValueError('Use the revised closeout-v2 plan; legacy f-rest includes cancelled PEOnly slots')
    validate_plan(plan, manifest, config)
    phases = {'f-low': '2.6-low', 'f-rest': '2.6-rest', 'e': '2.5'}
    if batch in phases:
        return c.run_phase(root, phases[batch], retry_failed=retry_failed)
    if batch != 'd':
        raise ValueError('Unknown follow-up batch')
    selected = [r['slot_id'] for r in plan['rows'] if r['batch'] == 'trace' and r['classification'] == 'paper_required']
    return run_selected(root, selected, plan=plan, retry_failed=retry_failed)
