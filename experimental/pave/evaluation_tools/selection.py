"""Audited selection revisions and an external publication/execution plan."""
from __future__ import annotations

import copy
import math
import os
from collections import Counter
from pathlib import Path

from pave_ilp.profiles import digest
from pave_sim.evaluation import campaign as c
from pave_sim.evaluation.analysis import measure
from pave_sim.evaluation.matrix import references
from pave_sim.records import write_csv, write_json


def tool_identity():
    return {p.name: c.file_sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def rank_without_slo(rows, config):
    conditions = {(g, s, float(r)) for g, groups in config.rates.items()
                  for s in config.main_clusters for r in groups[s]}
    indexed = {}
    for r in rows:
        key = (r['window_s'], r['margin'], (r['generator'], r['scenario'], r['rate_per_min']))
        if key in indexed or r['policy'] != 'E' or r['trace'] != 'majority60':
            raise ValueError('Duplicate or invalid grid row')
        if any(not math.isfinite(r[k]) or r[k] <= 0 for k in ('throughput_req_s', 'p99_s')):
            raise ValueError('Invalid grid performance')
        indexed[key] = r
    expected = {(w, m, x) for w in config.windows_s for m in config.margins for x in conditions}
    if set(indexed) != expected:
        raise ValueError('Complete grid required')
    max_t = {x: max(indexed[w, m, x]['throughput_req_s'] for w in config.windows_s for m in config.margins) for x in conditions}
    min_p = {x: min(indexed[w, m, x]['p99_s'] for w in config.windows_s for m in config.margins) for x in conditions}
    ranking = []
    for w in config.windows_s:
        for m in config.margins:
            ranking.append({'window_s': w, 'margin': m,
                'throughput_score': math.exp(math.fsum(math.log(indexed[w, m, x]['throughput_req_s']/max_t[x]) for x in sorted(conditions))/len(conditions)),
                'p99_score': math.exp(math.fsum(math.log(indexed[w, m, x]['p99_s']/min_p[x]) for x in sorted(conditions))/len(conditions)),
                'run_ids': [indexed[w, m, x]['run_id'] for x in sorted(conditions)]})
    best = max(r['throughput_score'] for r in ranking)
    for r in ranking:
        r['eligible'] = r['throughput_score'] >= best * .995
    chosen = min((r for r in ranking if r['eligible']), key=lambda r: (r['p99_score'], r['window_s'], r['margin']))
    return {k: chosen[k] for k in ('window_s', 'margin')}, sorted(ranking, key=lambda r: (-r['throughput_score'], r['p99_score'], r['window_s'], r['margin']))


def rank_cluster_512(rows, config):
    lookup = {}
    for r in rows:
        if r['policy'] not in ('E', 'Static-512'):
            continue
        key = (r['generator'], r['scenario'], r['rate_per_min'], r['policy'])
        if key in lookup:
            raise ValueError('Duplicate cluster result')
        lookup[key] = r
    ranking = []
    for cluster in ('cluster1', 'cluster2'):
        pairs = []
        for model in config.rates:
            for rate in config.rates[model][cluster]:
                try:
                    e, b = (lookup[model, cluster, float(rate), p] for p in ('E', 'Static-512'))
                except KeyError as error:
                    raise ValueError('Incomplete cluster comparison') from error
                if any(not math.isfinite(r['throughput_req_s']) or r['throughput_req_s'] <= 0 for r in (e, b)):
                    raise ValueError('Invalid cluster throughput')
                if e.get('requests_sha256') != b.get('requests_sha256'):
                    raise ValueError('Cluster comparison requests differ')
                pairs.append({'generator': model, 'rate_per_min': rate,
                              'relative_gain': e['throughput_req_s']/b['throughput_req_s']-1,
                              'run_ids': [e['run_id'], b['run_id']]})
        ranking.append({'cluster': cluster, 'score': math.fsum(p['relative_gain'] for p in pairs)/len(pairs), 'pairs': pairs})
    ranking.sort(key=lambda r: (-r['score'], r['cluster']))
    return ranking[0]['cluster'], ranking


def classification(slot):
    if slot['policy'] != 'Static-2048':
        return 'paper_required'
    return 'historical_only' if slot['batch'] == 'static' else 'cancelled'


def paper_references(manifest, config):
    raw = references(config, manifest['slots'], manifest['selections'])
    allowed = {s['slot_id'] for s in manifest['slots'] if classification(s) == 'paper_required'}
    return {name: [s for s in ids if s in allowed] for name, ids in raw.items()}


def make_plan(manifest, config):
    rows = []
    for slot in manifest['slots']:
        b = manifest['bindings'][slot['slot_id']]
        if 'run_id' not in b:
            raise ValueError('Selections must be committed before making plan')
        rows.append({'slot_id': slot['slot_id'], 'run_id': b['run_id'],
                     'classification': classification(slot), 'batch': slot['batch'],
                     'bound': b['bound']})
    plan = {'schema_version': 1, 'definition_sha256': manifest['definition_sha256'],
            'runtime_sha256': manifest['source']['runtime_sha256'],
            'selections': {k: v['selection_sha256'] for k, v in manifest['selections'].items()},
            'rows': rows, 'references': paper_references(manifest, config),
            'rule': 'Static-512 publication; d/e selected against Static-512; cancel unrun Static-2048 trace slots'}
    plan['plan_sha256'] = digest(plan)
    return plan


def validate_plan(plan, manifest, config):
    if plan != make_plan(manifest, config):
        raise ValueError('Execution plan is stale, altered, or from a different campaign')


def write_index(path, plan, manifest):
    rows = []
    for r in plan['rows']:
        b = manifest['bindings'][r['slot_id']]
        rows.append({**r['bound'], 'run_id': r['run_id'], 'classification': r['classification'],
                     'status': 'cancelled' if r['classification'] == 'cancelled' else b['status'],
                     'runtime_status': b['status'],
                     'experiments': ','.join(k for k, v in plan['references'].items() if r['slot_id'] in v)})
    write_csv(path, rows)
    return dict(Counter((r['classification'], r['status']) for r in rows))


def revise_cluster(root, evidence):
    root, evidence = Path(root), Path(evidence)
    if evidence.exists():
        raise ValueError('Revision evidence directory already exists; inspect prior receipt')
    lock = root / 'execution.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        m, cfg = c.load_campaign(root, execution=True)
        dependent = [m['bindings'][s['slot_id']] for s in m['slots'] if s['batch'] in ('trace', 'scheduler')]
        if any(b['run_id'] in m['runs'] for b in dependent):
            raise ValueError('Cannot revise cluster after a dependent attempt has started')
        if m['selections']['cluster'].get('baseline') == 'Static-512':
            raise ValueError('Cluster already revised')
        old = copy.deepcopy(m)
        cached = {}
        def metric(slot_id):
            b = m['bindings'][slot_id]
            if b['run_id'] not in cached:
                cached[b['run_id']] = measure(c.load_result(root, m, b), b)
            return cached[b['run_id']]
        grid = [metric(s['slot_id']) for s in m['slots'] if s['batch'] == 'grid']
        parameters, p_ranking = rank_without_slo(grid, cfg)
        if parameters != m['selections']['parameters']['chosen']:
            raise ValueError('No-SLO selection differs; stop before mutation')
        main = [metric(s) for s in references(cfg, m['slots'], m['selections'])['a']]
        chosen, ranking = rank_cluster_512(main, cfg)
        if not cfg.synthetic and chosen != 'cluster1':
            raise ValueError('Observed ranking differs from the approved Cluster1 decision')
        receipt = {'chosen': chosen, 'ranking': ranking, 'baseline': 'Static-512',
                   'created_utc': c.now(), 'rule_version': 'publication-cluster-v1',
                   'previous_selection_sha256': m['selections']['cluster']['selection_sha256'],
                   'tool_sources': tool_identity(),
                   'input_identities': {r['run_id']: r['identity_sha256'] for r in main if r['policy'] != 'Static-2048'}}
        receipt['selection_sha256'] = digest(receipt)
        m['selections']['cluster'] = receipt
        new_bindings = c.resolve(m, cfg)
        for slot in m['slots']:
            sid = slot['slot_id']
            if slot['batch'] not in ('trace', 'scheduler') and new_bindings[sid] != old['bindings'][sid]:
                raise ValueError('Revision changes an unrelated binding')
        if m['runs'] != old['runs']:
            raise ValueError('Revision changes existing runs')
        evidence.mkdir(parents=True)
        write_json(evidence / 'manifest_before.json', old)
        write_json(evidence / 'selection_cluster_before.json', old['selections']['cluster'])
        write_json(evidence / 'references_before.json', old['references'])
        write_json(evidence / 'parameters_without_slo.json', {'chosen': parameters, 'ranking': p_ranking,
                   'rule': 'throughput >= .995 * best; then normalized p99; then smaller W/M'})
        write_json(evidence / 'selection_cluster_after.json', receipt)
        # Manifest is authoritative. Roll back the two current receipt files on a write failure.
        try:
            c.write_manifest(root, m, cfg)
            c.atomic_json(root / 'selection_cluster.json', receipt)
            checked, _ = c.load_campaign(root, execution=True)
        except BaseException:
            c.atomic_json(root / 'manifest.json', old)
            c.atomic_json(root / 'selection_cluster.json', old['selections']['cluster'])
            raise
        plan = make_plan(checked, cfg)
        write_json(evidence / 'execution_plan.json', plan)
        write_index(evidence / 'run_index.csv', plan, checked)
        write_json(evidence / 'revision_receipt.json', {'status': 'complete', 'chosen': chosen,
                   'completed_runs_unchanged': len(m['runs']), 'startup_bindings_unchanged': True,
                   'definition_sha256_unchanged': old['definition_sha256'] == checked['definition_sha256'],
                   'reference_counts': {k: len(v) for k, v in plan['references'].items()}})
        return {'chosen': chosen, 'plan': str(evidence / 'execution_plan.json'), 'runs_unchanged': len(m['runs'])}
    finally:
        lock.unlink()
