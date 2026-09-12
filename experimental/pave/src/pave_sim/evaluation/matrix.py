"""Logical slots are distinct from bound, content-addressed executions."""
from __future__ import annotations

import copy
from pave_ilp.profiles import GENERATORS, digest
from .config import CLUSTERS


def slots(config):
    result = []
    def add(batch, generator, scenario, rate_index, policy, trace='majority60', window=None,
            margin=None, request_seed=None, scheduler_seed=None):
        row = dict(batch=batch, generator=generator, scenario=scenario, rate_index=rate_index,
                   policy=policy, trace=trace, window_s=window, margin=margin,
                   request_seed=request_seed, scheduler_seed=scheduler_seed)
        row['slot_id'] = 'slot-' + digest(row)[:20]
        result.append(row)
    for model, clusters in config.rates.items():
        for cluster in config.main_clusters:
            rates = clusters[cluster]
            for index in range(len(rates)):
                for window in config.windows_s:
                    for margin in config.margins:
                        add('grid', model, cluster, index, 'E', window=window, margin=margin)
                for policy in ('Static-512', 'Static-2048'):
                    add('static', model, cluster, index, policy)
        # Both candidate clusters must expose the same number of load tiers.
        if len(clusters['cluster1']) != len(clusters['cluster2']):
            raise ValueError('Candidate clusters must have equal numbers of load tiers')
        for index in range(len(clusters['cluster1'])):
            traces = [('majority15', None), ('majority30', None)] + [('mixture', s) for s in config.request_seeds]
            for trace, seed in traces:
                for policy in ('E', 'Static-512', 'Static-2048'):
                    add('trace', model, '$cluster', index, policy, trace, '$window' if policy == 'E' else None,
                        '$margin' if policy == 'E' else None, seed)
            add('scheduler', model, '$cluster', index, 'E-Least', window='$window', margin='$margin')
            for seed in config.scheduler_seeds:
                add('scheduler', model, '$cluster', index, 'E-Weighted', window='$window', margin='$margin', scheduler_seed=seed)
        for cluster in config.startup_clusters:
            for index in range(len(clusters[cluster])):
                for policy in config.startup_policies:
                    if policy == 'E' and cluster in config.main_clusters:
                        continue
                    add('startup', model, cluster, index, policy, window='$window', margin='$margin')
    return result


def bind(slot, config, selections):
    row = copy.deepcopy(slot)
    parameters = selections.get('parameters', {}).get('chosen')
    cluster = selections.get('cluster', {}).get('chosen')
    # Trace and scheduler batches deliberately wait for both selections, including
    # their static members, so a batch has one coherent context.
    if row['batch'] in ('trace', 'scheduler') and (parameters is None or cluster is None):
        return None
    if row['batch'] == 'startup' and parameters is None:
        return None
    if row['scenario'] == '$cluster':
        if cluster not in ('cluster1', 'cluster2'):
            raise ValueError('Invalid selected cluster')
        row['scenario'] = cluster
    for key, placeholder in (('window_s', '$window'), ('margin', '$margin')):
        if row[key] == placeholder:
            row[key] = parameters[key]
    row['rate_per_min'] = float(config.rates[row['generator']][row['scenario']][row['rate_index']])
    row['case_id'] = f"{GENERATORS[row['generator']]}_{row['scenario']}"
    if not row['policy'].startswith('Static-') and (row['window_s'] not in config.windows_s or row['margin'] not in config.margins):
        raise ValueError('Selected parameters are outside the frozen grid')
    return row


def in_phase(row, phase, config):
    pilot_window = 30 if 30 in config.windows_s else config.windows_s[0]
    pilot_margin = 0.10 if 0.10 in config.margins else config.margins[0]
    pilot = row['batch'] == 'grid' and row['window_s'] == pilot_window and row['margin'] == pilot_margin and row['rate_index'] in (0, len(config.rates[row['generator']][row['scenario']]) - 1)
    if phase == '2.1':
        return pilot
    if phase == '2.2':
        return row['batch'] == 'grid' and not pilot
    if phase in ('2.6-low', '2.6-rest'):
        return row['batch'] == 'startup' and ((row['rate_index'] == 0) if phase == '2.6-low' else (row['rate_index'] > 0))
    return row['batch'] == {'2.3': 'static', '2.4': 'trace', '2.5': 'scheduler', '2.6': 'startup'}.get(phase)


def references(config, logical, selections):
    if 'parameters' not in selections:
        raise ValueError('Dataset references require parameter selection')
    p, c = selections['parameters']['chosen'], selections.get('cluster', {}).get('chosen')
    output = {name: [] for name in 'abcdef'}
    for source in logical:
        row = bind(source, config, selections)
        if row is None:
            continue
        selected_e = row['batch'] == 'grid' and row['window_s'] == p['window_s'] and row['margin'] == p['margin']
        main = selected_e or row['batch'] == 'static'
        if main:
            output['a'].append(row['slot_id'])
        if row['batch'] == 'grid' and row['scenario'] in ('cluster1', 'cluster2'):
            if row['window_s'] == p['window_s']:
                output['b'].append(row['slot_id'])
            if row['margin'] == p['margin']:
                output['c'].append(row['slot_id'])
        if row['batch'] == 'trace' or (main and row['scenario'] == c):
            output['d'].append(row['slot_id'])
        if row['batch'] == 'scheduler' or (selected_e and row['scenario'] == c):
            output['e'].append(row['slot_id'])
        if row['batch'] == 'startup' or (selected_e and row['scenario'] in config.startup_clusters):
            output['f'].append(row['slot_id'])
    return output
