"""Read-only campaign audit and pilot tables. Never invoke a simulation or select parameters."""
import gzip
import json
import math
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from pave_sim.evaluation.analysis import flatten, measure
from pave_sim.evaluation.campaign import fingerprint_sources, load_campaign, load_result
from pave_sim.evaluation.traces import file_sha
from pave_sim.records import write_csv, write_json
from pave_sim.timing import ticks, seconds

STAGES = ('PE', 'TE', 'DiT', 'VAE')


def check(value, message):
    if not value:
        raise ValueError(message)


def near(a, b, label):
    check(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8), f'{label}: {a} != {b}')


def quantile(values, percent):
    ordered = sorted(values)
    at = (len(ordered) - 1) * percent / 100
    low = math.floor(at)
    high = math.ceil(at)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def audit_startup(policy, raw, record):
    """Independent expected mode from declared policy and physical modules."""
    modes = {'E': 'all_optimized', 'E-PEOnly': 'pe_only_optimized', 'E-NoOpt': 'none_optimized'}
    check(policy in modes, 'Unsupported startup audit policy')
    opt = policy == 'E' or (policy == 'E-PEOnly' and set(raw['stages']) == {'PE'})
    check(record['startup_mode'] == modes[policy] and record['optimized'] == opt, 'Startup mode/profile selection')
    check(ticks(record['startup_s']) == ticks(raw['startup']['optimized_s' if opt else 'non_optimized_s']), 'Archived startup latency')
    if record.get('ready_s') is not None:
        check(ticks(record['ready_s']) - ticks(record['launch_started_s']) == ticks(record['startup_s']), 'Startup ready duration')


def audit_run(result, row, directory, expected, campaign):
    rid = result['run_id']
    run = result['spec']
    window = run['window_s']
    delta = run['margin'] * 1536
    scheduler = 'least_waiting' if row['policy'].startswith('Static-') else 'estimated_completion'
    trace = json.loads((Path(campaign) / 'inputs/trace.json').read_text('utf-8'))
    intervals = trace['bins']['60']
    def input_type(t):
        return intervals[t // ticks(60)]['resolved'] if t < ticks(3600) else None
    requests = {r['id']: r for r in result['requests']}
    check(len(requests) == len(result['requests']) == expected['requests'], 'Request count/uniqueness')
    check(all(r['finished_s'] is not None and r['stage_index'] == 4 and r['owner'] is None
              and r['generated'] == r['output_tokens'] and r['steps'] == 50 for r in requests.values()), 'Final request state')
    check(all(0 <= ticks(r['arrival_s']) < ticks(3600) for r in requests.values()), 'Arrival bounds')
    check(row['observed_completed'] + row['end_backlog'] == len(requests), 'Observation/backlog conservation')
    latencies = [seconds(ticks(r['finished_s']) - ticks(r['arrival_s'])) for r in requests.values()]
    near(row['p50_s'], quantile(latencies, 50), 'Independent p50')
    near(row['p99_s'], quantile(latencies, 99), 'Independent p99')
    near(row['throughput_req_s'], len(requests) / (max(r['finished_s'] for r in requests.values()) - min(r['arrival_s'] for r in requests.values())), 'Independent throughput')
    for factor in (5, 10):
        attainment = sum((ticks(r['finished_s']) - ticks(r['arrival_s'])) <= ticks(factor * result['slo_baselines'][r['kind']]['latency_s']) for r in requests.values()) / len(requests)
        near(row[f'slo{factor}'], attainment, f'Independent SLO{factor}')
    per_req_stage = defaultdict(list)
    physical_by_uid = {p['uid']: p for p in result['instances']}
    for a in result['attempts']:
        per_req_stage[a['request_id'], a['stage']].append(a)
        check(a['exit_s'] is not None, 'Unclosed attempt')
        if a['start_s'] is not None:
            check(ticks(a['enter_s']) <= ticks(a['start_s']) <= ticks(a['exit_s']), 'Attempt time ordering')
            check(ticks(a['executed_service_s']) == ticks(a['exit_s']) - ticks(a['start_s']), 'Attempt service time')
            if a['stage'] == 'PE':
                p, g = a['pe_profile'], a['initial_progress']
                remaining = requests[a['request_id']]['output_tokens'] - g
                check(p['actual_input'] == 128 + g and p['actual_remaining'] == remaining, 'PE reprefill request')
                duration = ticks(p['ttft_s']) + (remaining - 1) * ticks(p['tpot_s'])
                check(ticks(a['planned_service_s']) == duration, 'PE remaining-token service formula')
                if a['exit_reason'] == 'completed':
                    check(ticks(a['executed_service_s']) == duration, 'Completed PE service')
                else:
                    busy = ticks(a['executed_service_s'])
                    check(busy >= ticks(p['ttft_s']) and (busy - ticks(p['ttft_s'])) % ticks(p['tpot_s']) == 0, 'Interrupted PE token boundary')
                check(a['executed_work'] == 1 + (ticks(a['executed_service_s']) - ticks(p['ttft_s'])) // ticks(p['tpot_s']), 'PE token/service conservation')
            elif a['stage'] == 'DiT':
                full = ticks(physical_by_uid[a['instance_uid']]['raw']['stages']['DiT']['latency_s'])
                check(ticks(a['planned_service_s']) == round(Fraction((50-a['initial_progress'])*full,50)), 'DiT remaining-step duration')
                check(ticks(a['executed_service_s']) == round(Fraction(int(a['executed_work'])*full,50)), 'DiT step-safe execution time')
            else:
                check(a['exit_reason'] == 'completed', 'TE/VAE interrupted during execution')
    for request in requests.values():
        for stage in STAGES:
            attempts = sorted(per_req_stage[request['id'], stage], key=lambda a: a['attempt_id'])
            check(sum(a['exit_reason'] == 'completed' for a in attempts) == 1, 'Stage completed once')
            if stage in ('PE', 'DiT'):
                target = request['output_tokens'] if stage == 'PE' else 50
                check(sum(a['executed_work'] for a in attempts) == target, 'Per-request token/step conservation')
                name = 'generated' if stage == 'PE' else 'steps'
                before = 0
                for a in attempts:
                    check(a[name + '_on_enter'] == before, 'Migration progress continuity')
                    after = a[name + '_on_exit']
                    check(before <= after <= target, 'Progress monotonicity')
                    before = after

    # Independent stream replay: ownership, GPU occupancy, BS=1, observed lengths.
    queue, running, owner = {}, {}, {}
    physicals, occupied, launch_records, ready_records = {}, {}, {}, {}
    original_pe = {p['uid'] for p in result['instances'] if p['created_s'] == 0 and 'PE' in p['raw']['stages']}
    entered, exited, ended = Counter(), Counter(), Counter()
    counts, pe_completed, monitor_rows = Counter(), deque(), []
    wait_area, busy_area = Counter(), Counter()
    observed_wait_area, observed_busy_area = Counter(), Counter()
    last, number, active, transition = 0, 0, (2048 if run['initial_deployment'] == 'target' else 512), None
    flip_begin, flip_end = {}, {}
    max_wait = Counter()
    comb_overlap_ticks = 0
    active_bin_ticks, observed_bin_ticks = Counter(), Counter()
    transition_ticks, observed_transition_ticks = 0, 0
    with gzip.open(directory / 'events.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            event = json.loads(line)
            check(event['run_id'] == rid and event['sequence'] == number, 'Event identity/sequence')
            now = ticks(event['time_s'])
            check(now >= last, 'Event chronology')
            dt, observed_dt = now - last, max(0, min(now, ticks(3600)) - min(last, ticks(3600)))
            active_bin_ticks[str(active)] += dt
            observed_bin_ticks[str(active)] += observed_dt
            if transition is not None:
                transition_ticks += dt
                observed_transition_ticks += observed_dt
            running_by_physical = defaultdict(set)
            for a in running.values():
                running_by_physical[a['instance_uid']].add(a['stage'])
            comb_overlap_ticks += sum({'DiT','VAE'} <= s for s in running_by_physical.values()) * dt
            waiting_counts = Counter(a['stage'] for a in queue.values())
            busy_counts = Counter(a['stage'] for a in running.values())
            for stage in STAGES:
                wait_area[stage] += waiting_counts[stage] * dt
                busy_area[stage] += busy_counts[stage] * dt
                observed_wait_area[stage] += waiting_counts[stage] * observed_dt
                observed_busy_area[stage] += busy_counts[stage] * observed_dt
                max_wait[stage] = max(max_wait[stage], waiting_counts[stage])
            last, number = now, number + 1
            kind = event['event']; counts[kind] += 1
            if kind == 'instance_created':
                uid, raw = event['instance_uid'], event['placement']
                check(uid not in physicals, 'Unique physical lifecycle')
                physicals[uid] = {'raw': raw, 'state': event['state']}
                for gpu in raw['gpu_ids']:
                    key = raw['node'], gpu
                    check(key not in occupied, 'GPU overlap')
                    occupied[key] = uid
            elif kind == 'instance_retired':
                uid = event['instance_uid']; p = physicals[uid]
                check(not any(a['instance_uid'] == uid for a in list(queue.values()) + list(running.values())), 'Retire with owned work')
                check(uid not in original_pe, 'Original PE retired')
                for gpu in p['raw']['gpu_ids']:
                    check(occupied.pop((p['raw']['node'], gpu)) == uid, 'GPU retirement ownership')
                p['state'] = 'retired'
            elif kind == 'instance_ready':
                p = physicals[event['instance_uid']]
                check(p['state'] == 'launching', 'Ready lifecycle state')
                launched = launch_records[event['instance_uid']]
                check(now == ticks(launched['launch_started_s']) + ticks(launched['startup_s']), 'Ready event/profile time')
                ready_records[event['instance_uid']] = now
                p['state'] = 'ready'
            elif kind == 'launch_start':
                audit_startup(row['policy'], physicals[event['instance_uid']]['raw'], event)
                check(now == ticks(event['launch_started_s']) and event['instance_uid'] not in launch_records, 'One launch per physical lifecycle')
                launch_records[event['instance_uid']] = event
            elif kind == 'request_enter':
                entered[event['request_id']] += 1
            elif kind == 'queue_enter':
                qid = event['request_id']; aid = event['attempt_id']
                check(qid not in owner and aid not in queue, 'Duplicate queued owner')
                queue[aid] = event; owner[qid] = ('queue', aid)
            elif kind == 'queue_exit':
                aid = event['attempt_id']; a = queue.pop(aid)
                check(owner.pop(a['request_id']) == ('queue', aid), 'Queue exit ownership')
            elif kind == 'execution_start':
                lane, qid = event['lane_uid'], event['request_id']
                check(lane not in running and qid not in owner, 'BS=1/execution ownership')
                check(physicals[event['instance_uid']]['state'] == 'ready', 'Start on unavailable instance')
                running[lane] = event; owner[qid] = ('running', lane)
            elif kind == 'execution_end':
                lane, qid = event['lane_uid'], event['request_id']
                check(running.pop(lane)['attempt_id'] == event['attempt_id'], 'Execution attempt pairing')
                check(owner.pop(qid) == ('running', lane), 'Execution exit ownership')
                ended[event['attempt_id']] += 1
                if event['stage'] == 'PE' and event['exit_reason'] == 'completed':
                    pe_completed.append((now, requests[qid]['output_tokens']))
            elif kind == 'request_exit':
                qid = event['request_id']
                check(qid not in owner and now == ticks(requests[qid]['finished_s']), 'Final completion ownership/time')
                exited[qid] += 1
            elif kind == 'dispatch':
                uid = event['selected_lane'].rsplit('/', 1)[0]
                check(physicals[uid]['state'] == 'ready', 'Dispatch to unavailable instance')
                check(event['scheduler'] == scheduler, 'Declared scheduler')
            elif kind == 'monitor':
                check(now % ticks(10) == 0, 'Monitor period')
                while pe_completed and pe_completed[0][0] <= now - ticks(window):
                    pe_completed.popleft()
                mean = sum(x[1] for x in pe_completed) / len(pe_completed) if pe_completed else None
                check(event['length']['samples'] == len(pe_completed), 'PE completion window sample count')
                if mean is None:
                    check(event['length']['mean'] is None, 'Empty window length')
                else:
                    near(mean, event['length']['mean'], 'Observed length mean')
                check(event['active_bin'] == active and event['transition_in_progress'] == (transition is not None), 'Monitor transition state')
                desired = 2048 if active == 512 and mean is not None and mean > 1280 + delta else 512 if active == 2048 and mean is not None and mean < 1280 - delta else active
                check(event['desired_bin'] == desired, 'Margin threshold decision')
                monitor_rows.append({'run_id': rid, 'time_s': event['time_s'], 'active_bin': active, 'desired_bin': desired,
                    'transition': transition, 'pe_samples': len(pe_completed), 'observed_mean_output': mean,
                    'arrival_type_at_this_time': input_type(now),
                    **{s + '_waiting': waiting_counts[s] for s in STAGES}, **{s + '_running': busy_counts[s] for s in STAGES}})
            elif kind == 'flip_begin':
                check(transition is None, 'Overlapping flip')
                transition = event['flip_id']; flip_begin[transition] = now
                for group in event['groups']:
                    for uid in group['source_uids']:
                        check(physicals[uid]['state'] == 'ready', 'Flip source state')
                        physicals[uid]['state'] = 'draining'
            elif kind == 'flip_complete':
                check(transition == event['flip_id'], 'Flip commit identity')
                active, transition = event['active_bin'], None
                flip_end[event['flip_id']] = now
    check(not queue and not running and not owner and transition is None, 'Final ownership/drain state')
    check(entered == exited == Counter({i: 1 for i in requests}), 'Event request completion uniqueness')
    check(all(v == 1 for v in ended.values()), 'Execution attempt ended twice')
    for stage in STAGES:
        near(seconds(wait_area[stage]), row['stages'][stage]['waiting_s'], 'Independent queue area ' + stage)
        near(seconds(busy_area[stage]), row['stages'][stage]['service_s'], 'Independent busy area ' + stage)

    groups, flips = [], []
    physical = {p['uid']: p for p in result['instances']}
    for f in result['flips']:
        detected, completed = ticks(f['detected_s']), ticks(f['completed_s'])
        check(flip_begin[f['flip_id']] == detected and flip_end[f['flip_id']] == completed, 'Flip record/event agreement')
        ready_times = []
        for g in f['groups']:
            launch, ready = ticks(g['launch_started_s']), ticks(g['ready_s'])
            check(detected <= launch <= ready <= completed, 'Group time ordering')
            check(set(g['actual_safe_s']) == set(g['safe_boundaries_s']), 'Safe lane coverage')
            for lane, safe in g['actual_safe_s'].items():
                check(ticks(safe) == ticks(g['safe_boundaries_s'][lane]) <= launch, 'Expected/actual safe boundary')
            check(max(map(ticks, g['actual_safe_s'].values())) == launch, 'Group starts at own safe time')
            source_gpus = {(physical[u]['raw']['node'], gpu) for u in g['source_uids'] for gpu in physical[u]['raw']['gpu_ids']}
            check(source_gpus == {(g['node'], gpu) for gpu in g['gpu_ids']}, 'Reported conversion GPU group')
            target_gpus = set()
            for target in g['targets']:
                p = physical[target['instance_uid']]
                audit_startup(row['policy'], p['raw'], target)
                check(ready_records[target['instance_uid']] == ticks(target['ready_s']), 'Ready record/event agreement')
                check(ticks(target['launch_started_s']) == launch and ticks(target['ready_s']) == launch + ticks(target['startup_s']), 'Target startup timing')
                for gpu in p['raw']['gpu_ids']:
                    location = p['raw']['node'], gpu
                    check(location not in target_gpus, 'Split target GPU duplication')
                    target_gpus.add(location)
            check(source_gpus == target_gpus, 'Conversion group GPU conservation')
            check(ready == max(ticks(t['ready_s']) for t in g['targets']), 'Independent child ready/group ready')
            ready_times.append(ready)
            groups.append({'run_id': rid, 'flip_id': f['flip_id'], 'direction': f['direction'], 'detected_s': f['detected_s'],
                'group_id': g['group_id'], 'node': g['node'], 'gpu_ids': g['gpu_ids'], 'source_uids': g['source_uids'],
                'target_templates': g['target_templates'], 'safe_wait_s': seconds(launch-detected),
                'launch_started_s': g['launch_started_s'], 'ready_s': g['ready_s'],
                'migrated_waiting': sum(map(len, g['migrated_waiting'].values())),
                'migrated_running': sum(map(len, g['migrated_running'].values()))})
        check(max(ready_times) == completed, 'Flip commit after all groups ready')
        flips.append({'run_id': rid, 'flip_id': f['flip_id'], 'direction': f['direction'],
            'detected_s': f['detected_s'], 'completed_s': f['completed_s'], 'duration_s': seconds(completed-detected),
            'mean_output': f['length_observation']['mean'], 'pe_samples': f['length_observation']['samples'],
            'groups': len(f['groups']), 'in_drain': detected > ticks(3600),
            'arrival_type_at_detection': input_type(detected)})
    return {'run_id': rid, 'checks_passed': True, 'requests': len(requests), 'event_log_rows': number,
            'comb_dit_vae_overlap_instance_s': seconds(comb_overlap_ticks),
            'logical_bin_residence_s': {k:seconds(v) for k,v in active_bin_ticks.items()},
            'observed_logical_bin_residence_s': {k:seconds(v) for k,v in observed_bin_ticks.items()},
            'transition_wall_s':seconds(transition_ticks),'observed_transition_wall_s':seconds(observed_transition_ticks),
            'event_types': dict(counts), 'maximum_waiting': dict(max_wait),
            'observed_queue_area_s': {s: seconds(observed_wait_area[s]) for s in STAGES},
            'observed_busy_area_s': {s: seconds(observed_busy_area[s]) for s in STAGES}}, groups, flips, monitor_rows

