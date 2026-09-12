"""Versioned closeout policy layered over the unchanged 408-slot campaign."""
from collections import Counter
from pathlib import Path
from pave_ilp.profiles import digest
from pave_sim.evaluation import campaign as c
from pave_sim.records import write_json
from .selection import make_plan as legacy_plan, write_index

def make_plan(m,cfg):
    p=legacy_plan(m,cfg)
    p.pop('plan_sha256')
    p.update(schema_version=2,rule='closeout-v2: f E vs E-NoOpt; retire PEOnly; retain history; no simulation changes')
    for row in p['rows']:
        if row['bound']['policy']=='E-PEOnly':
            row['classification']='historical_only' if row['bound']['rate_index']==0 else 'cancelled'
    allowed={r['slot_id'] for r in p['rows'] if r['classification']=='paper_required'}
    p['references']={k:[s for s in ids if s in allowed] for k,ids in p['references'].items()}
    p['plan_sha256']=digest(p)
    return p

def validate_plan(plan,m,cfg):
    if plan!=make_plan(m,cfg):raise ValueError('Closeout execution plan is stale or altered')

def batch_ids(plan,batch):
    if batch not in ('f-rest','d','e'):raise ValueError('Unknown closeout batch')
    return [r['slot_id'] for r in plan['rows'] if r['classification']=='paper_required' and
        ((batch=='f-rest' and r['batch']=='startup' and r['bound']['rate_index']>0) or
         (batch=='d' and r['batch']=='trace') or (batch=='e' and r['batch']=='scheduler'))]

def revise(root,output):
    root,output=Path(root),Path(output)
    if output.exists():raise ValueError('Revision evidence exists')
    if (root/'execution.lock').exists():raise ValueError('Campaign is running')
    m,cfg=c.load_campaign(root,execution=True)
    plan=make_plan(m,cfg)
    if not cfg.synthetic:
        assert {k:len(batch_ids(plan,k)) for k in ('f-rest','d','e')}=={'f-rest':12,'d':84,'e':36}
        assert Counter(r['classification'] for r in plan['rows'])=={'paper_required':342,'historical_only':16,'cancelled':50}
        for r in plan['rows']:
            if r['classification']=='cancelled':
                assert r['run_id'] not in m['runs'],'Cancelled slot already has attempt'
            elif r['classification']=='historical_only':
                assert m['runs'][r['run_id']]['status']=='complete'
    output.mkdir(parents=True)
    write_json(output/'execution_plan_before.json',legacy_plan(m,cfg))
    write_json(output/'execution_plan.json',plan)
    write_index(output/'run_index.csv',plan,m)
    write_json(output/'revision_receipt.json',{'status':'complete','campaign_unchanged':True,
        'run_ids_unchanged':True,'source_unchanged':True,'classes':dict(Counter(r['classification'] for r in plan['rows'])),
        'references':{k:len(v) for k,v in plan['references'].items()},
        'batches':{k:len(batch_ids(plan,k)) for k in ('f-rest','d','e')}})
    return {'plan':str(output/'execution_plan.json'),'new_runs':sum(len(batch_ids(plan,k)) for k in ('f-rest','d','e'))}
