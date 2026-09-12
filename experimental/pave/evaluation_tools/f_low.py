"""Explicit 12-run startup batch analysis, independent of the full 36-row report."""
import argparse
import copy
from collections import Counter, defaultdict
from pathlib import Path

from pave_sim.evaluation import campaign as c
from pave_sim.evaluation.analysis import comparisons, flatten, measure, startup_tables
from pave_sim.records import write_csv, write_json
from pave_sim.timing import seconds, ticks
from .f_low_audit import audit_run, check
from .publication import remove_slo
from .selection import paper_references, tool_identity

POLICIES = ('E', 'E-PEOnly', 'E-NoOpt')

def low_bindings(m, cfg):
    out = [m['bindings'][sid] for sid in paper_references(m, cfg)['f']
           if m['bindings'][sid]['bound']['rate_index'] == 0]
    validate_pairs(out)
    return out

def validate_pairs(bindings):
    groups = defaultdict(list)
    if len(bindings) != 12 or len({b['run_id'] for b in bindings}) != 12:
        raise ValueError('f-low requires exactly 12 unique declared runs')
    for b in bindings:
        e = b['context']['effective']
        groups[e['generator'], e['scenario'], e['rate_per_min']].append(b)
    if len(groups) != 4:
        raise ValueError('f-low requires four conditions')
    for bs in groups.values():
        if Counter(b['context']['effective']['policy'] for b in bs) != Counter(POLICIES):
            raise ValueError('Missing/duplicate startup policy')
        fixed = []
        for b in bs:
            ctx = copy.deepcopy(b['context'])
            ctx['effective'].pop('policy')
            ctx['effective'].pop('startup_mode')
            fixed.append(ctx)
        if any(x != fixed[0] for x in fixed[1:]):
            raise ValueError('Comparison fixed context differs')

def freeze_scope(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('Scope already exists')
    m, cfg = c.load_campaign(root, execution=True)
    bs = low_bindings(m, cfg)
    expected = {('wan2.2-ti2v-5b','cluster2'): (5.5,330),
                ('wan2.2-ti2v-5b','clustersimu'): (44,2640),
                ('wan2.1-t2v-1.3b','cluster2'): (10,600),
                ('wan2.1-t2v-1.3b','clustersimu'): (66,3960)}
    rows = []
    for b in bs:
        e = b['context']['effective']
        rate, count = expected[e['generator'], e['scenario']]
        check((e['rate_per_min'],e['window_s'],e['margin'],e['monitor_period_s']) == (rate,60,.15,10), 'Approved setup mismatch')
        payload = c.read(c.inside(root, m['requests'][b['request_key']]['path']))
        check(len(payload['records']) == count, 'Frozen request count')
        reused = b['run_id'] in m['runs']
        if reused:
            c.load_result(root, m, b)
        else:
            check(b['status'] == 'ready', 'New slot is not ready')
        rows.append({'run_id': b['run_id'], 'slot_id': b['bound']['slot_id'],
                     'requests': count, 'reused': reused, 'context': b['context']})
    check(sum(r['reused'] for r in rows) == 2, 'Expected two reused E runs')
    check(sum(r['requests'] for r in rows if not r['reused']) == 21660, 'New request total')
    write_json(output, {'kind':'f_low_scope', 'runtime_sha256':m['source']['runtime_sha256'], 'rows':rows})
    return {'new':10, 'reused':2, 'new_requests':21660}

def enrich_periods(startup, results):
    by_run = {r['run_id']: r for r in results}
    for row in startup['direction_periods']:
        row.update(safe_wait_group_s=0., group_startup_elapsed_sum_s=0.)
        end = ticks(by_run[row['run_id']]['evaluation_context']['effective']['duration_s'])
        for g in startup['groups']:
            if (g['run_id'],g['direction']) != (row['run_id'],row['direction']):
                continue
            def intersection(a,b):
                a,b=ticks(a),ticks(b)
                return seconds(max(0,min(b,end)-a) if row['period']=='observation' else max(0,b-max(a,end)))
            row['safe_wait_group_s'] += intersection(g['detected_s'],g['launch_started_s'])
            row['group_startup_elapsed_sum_s'] += intersection(g['launch_started_s'],g['ready_s'])
    return startup

def representative_timelines(result):
    flips = result['flips']
    records = []
    chosen = []
    for direction in ('short_to_long','long_to_short'):
        first = next((f for f in flips if f['direction']==direction), None)
        chosen.append(('first_'+direction, first, None))
    all_groups = [(f,g) for f in flips for g in f['groups']]
    longest = max(all_groups, key=lambda fg: ticks(fg[1]['ready_s'])-ticks(fg[0]['detected_s']), default=None)
    chosen.append(('longest_unavailable_group', longest[0] if longest else None, longest[1]['group_id'] if longest else None))
    for tag,f,gid in chosen:
        if f is None:
            records.append({'run_id':result['run_id'],'example':tag,'occurred':False})
        else:
            records.append({'run_id':result['run_id'],'example':tag,'occurred':True,
                'flip_id':f['flip_id'],'direction':f['direction'],'detected_s':f['detected_s'],
                'completed_s':f['completed_s'],'length_observation':f['length_observation'],
                'groups':[g for g in f['groups'] if gid is None or g['group_id']==gid]})
    return records

def analyze(root, scope_file, output, *, audit=True):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('Analysis output must be new')
    m,cfg = c.load_campaign(root)
    declared = c.read(Path(scope_file))
    bs = low_bindings(m,cfg)
    by_id = {b['run_id']:b for b in bs}
    check(set(by_id) == {r['run_id'] for r in declared['rows']}, 'Declared scope mismatch')
    check(m['source']['runtime_sha256'] == declared['runtime_sha256'], 'Scope version mismatch')
    data, results, audits, groups, flips, monitors, timelines, costs, sources = [],[],[],[],[],[],[],[],[]
    for spec in declared['rows']:
        rid=spec['run_id']; b=by_id[rid]
        check(b['context']==spec['context'], 'Scoped context changed')
        result = c.load_result(root,m,b)
        row = measure(result,b)
        case = c.Case.load(cfg.settings(root/'inputs/ilp'),b['bound']['generator'],b['bound']['scenario'])
        for physical in result['instances']:
            raw=physical['raw']
            if raw['gpu_ids']:
                expected=case.profiles.catalog.startup(b['bound']['generator'],raw['hardware'],
                    len(raw['gpu_ids']),set(raw['stages'])=={'PE'})
                check(raw['startup']==expected,'Physical startup differs from frozen profile')
        attempt=m['runs'][rid]['attempts'][-1]
        directory=c.inside(root,attempt['path'])
        if audit:
            a,g,f,mon = audit_run(result,row,directory,spec,root)
            audits.append(a); groups.extend(g); flips.extend(f); monitors.extend(mon)
            print('AUDIT '+rid+' PASS events='+str(a['event_log_rows']),flush=True)
        else:
            a={}
        costs.append({'run_id':rid,'reused':spec['reused'],'wall_s':attempt['wall_s'],
            'events':a.get('event_log_rows'), 'bytes':sum(p.stat().st_size for p in directory.iterdir() if p.is_file()),
            'status':attempt['status']})
        sources.append({'experiment':'f-low','slot_id':b['bound']['slot_id'],'run_id':rid,
            'identity_sha256':b['identity_sha256'],'requests_sha256':b['context']['requests_sha256'],
            'raw_relative_directory':'archive/campaign-002/'+attempt['path'],'raw_files_sha256':attempt['files']})
        timelines.extend(representative_timelines(result))
        data.append(row); results.append(result)
    paired=comparisons('f',data,m['selections']['parameters']['chosen'])
    check(len(paired)==12,'Expected 12 comparisons')
    indexed={r['run_id']:r for r in data}
    raw_by_id={r['run_id']:r for r in results}
    paired_behavior=[]
    for pair in paired:
        for metric in ('throughput_req_s','p50_s','p99_s'):
            pair[metric+'_new']=indexed[pair['new_run_id']][metric]
            pair[metric+'_base']=indexed[pair['base_run_id']][metric]
        new,base=(raw_by_id[pair[k]] for k in ('new_run_id','base_run_id'))
        base_requests={r['id']:r for r in base['requests']}
        deltas=[ticks(r['finished_s'])-ticks(base_requests[r['id']]['finished_s']) for r in new['requests']]
        schedule=lambda r:[(f['direction'],ticks(f['detected_s'])) for f in r['flips']]
        paired_behavior.append({k:pair[k] for k in ('new_run_id','base_run_id','generator','scenario','rate_per_min','new_policy','base_policy')} | {
            'requests_earlier':sum(d<0 for d in deltas),'requests_same':sum(d==0 for d in deltas),
            'requests_later':sum(d>0 for d in deltas),'mean_finish_change_s':seconds(sum(deltas))/len(deltas),
            'same_flip_detection_schedule':schedule(new)==schedule(base),
            'new_flip_schedule':schedule(new),'base_flip_schedule':schedule(base)})
    startup=enrich_periods(startup_tables(results),results)
    output.mkdir(parents=True)
    def save(name,rows,clean=False):
        rows=remove_slo(rows) if clean else rows
        write_json(output/(name+'.json'),rows)
        write_csv(output/(name+'.csv'),[flatten(r) for r in rows])
    save('full_metrics',data)
    save('f_low_runs',data,True); save('f_low_comparisons',paired,True)
    save('f_low_paired_behavior',paired_behavior)
    fields=('run_id','generator','scenario','policy','rate_per_min','requests','p50_s','p99_s',
            'throughput_req_s','observed_throughput_req_s','observed_completed','end_backlog','drain_s',
            'last_request_finish_s','final_control_time_s','long_fraction','flip_triggered','converted_groups','pe_reprefill_s')
    save('f_low_system',[{k:r[k] for k in fields} for r in data])
    save('f_low_stage_costs',[dict(run_id=r['run_id'],generator=r['generator'],scenario=r['scenario'],
        policy=r['policy'],stage=s,**v,mean_wait_s=v['waiting_s']/r['requests'],mean_service_s=v['service_s']/r['requests'])
        for r in data for s,v in r['stages'].items()])
    save('f_low_allocations',[x for r in data for x in r['instance_allocations']])
    for name, rows in startup.items(): save('f_low_'+name,rows)
    save('f_low_audits',audits); save('f_low_group_audit',groups); save('f_low_flips',flips)
    save('f_low_monitors',monitors); save('f_low_timelines',timelines); save('f_low_costs',costs)
    save('f_low_sources',sources)
    write_json(output/'analysis_version.json', {'tool_sources':tool_identity(),
        'runtime_sha256':m['source']['runtime_sha256'],'run_ids':[r['run_id'] for r in data],
        'scope_sha256':c.file_sha(Path(scope_file)),'audit':audit})
    return {'runs':len(data),'comparisons':len(paired),'audited':len(audits), 'requests':sum(r['requests'] for r in data)}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['scope','analyze'])
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--scope',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    print(freeze_scope(args.campaign,args.scope) if args.command=='scope' else analyze(args.campaign,args.scope,args.output_dir))
