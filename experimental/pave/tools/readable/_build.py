"""Read-only analysis of archived PAVE results; writes a caller-selected new package."""
from pathlib import Path
import collections, csv, gzip, hashlib, json, math, statistics, sys, os

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(REBUILD_CONTEXT['work'])
ROOT = Path(REBUILD_CONTEXT['archive'])
OUT = Path(REBUILD_CONTEXT['output'])
CAM = 'archive/campaign-002'
FOLDERS = dict(a='a_主实验', b='b_margin', c='c_window', d='d_trace', e='e_调度器', f='f_启动优化', grid='附录_完整参数网格')
MODELS = ['wan2.2-ti2v-5b', 'wan2.1-t2v-1.3b']
CLUSTERS = ['cluster1','cluster2','clustersimu']
LABEL = {MODELS[0]:'Wan2.2', MODELS[1]:'Wan2.1', 'cluster1':'Cluster1','cluster2':'Cluster2','clustersimu':'ClusterSimu'}
METRICS = {'p50':('p50_s','p50','s',2),'p99':('p99_s','p99','s',2),'throughput':('throughput_req_s','排空吞吐','req/s',6)}
TRACE = {'majority15':'15秒多数票','majority30':'30秒多数票','majority60':'60秒多数票','mixture':'真实比例混合'}
inputs, csv_specs, md_checks, provenance = {}, [], [], []

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def load(rel):
    p=ROOT/rel
    inputs[str(rel).replace('\\','/')]={'sha256':sha(p),'bytes':p.stat().st_size}
    return json.loads(p.read_text(encoding='utf-8'))

def write(rel,text):
    p=OUT/rel; p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8',newline='\n')

def table(headers,rows):
    esc=lambda x:str(x).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(esc,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(esc,r))+' |' for r in rows])+'\n'

def sortkey(r):
    return MODELS.index(r['generator']),CLUSTERS.index(r['scenario']),r['rate_per_min'],r.get('window_s') or 0,r.get('margin') or 0

def cond(r): return r['generator'],r['scenario'],r['rate_per_min']
def label(k): return f'{LABEL[k[0]]}／{LABEL[k[1]]}、{k[2]:g} req/min'
def condcells(k): return [LABEL[k[0]],LABEL[k[1]],f'{k[2]:g}',f'{k[2]/60:.6f}']
def avg(xs): return statistics.fmean(xs)
def near(a,b): return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10)
def gain(new,base,field):
    return ((new/base-1) if field=='throughput_req_s' else (1-new/base))*100 if base else None
def val(r,field): return r[field] if field in r else r[field+'_mean']
def group(rows,key):
    g=collections.defaultdict(list)
    for r in rows:g[key(r)].append(r)
    return g
def choose(rows,**kw):
    found=[r for r in rows if all(r[k]==v for k,v in kw.items())]
    assert len(found)==1,(kw,len(found));return found[0]
def fmt(x,n=2):return f'{x:.{n}f}'
def range_text(xs,n=2):return f'{min(xs):.{n}f}–{max(xs):.{n}f}'
def direction_counts(xs):return (sum(x>1e-9 for x in xs),sum(abs(x)<=1e-9 for x in xs),sum(x< -1e-9 for x in xs))
def export_csv(rel,records):
    assert records,rel
    headers=list(records[0]);assert all(list(r)==headers for r in records),rel
    csv_specs.append({'path':rel,'matrix':[headers]+[[r[h] for h in headers] for r in records]})

def condition_groups(rows,exp):
    key=lambda r:cond(r)+((r['trace'],) if exp=='d' else (r['window_s'],) if exp=='grid' else ())
    groups=group(rows,key)
    return sorted(groups.items(),key=lambda kv:(MODELS.index(kv[0][0]),CLUSTERS.index(kv[0][1]),kv[0][2],list(TRACE).index(kv[0][3]) if exp=='d' else kv[0][3] if exp=='grid' else 0))

checks=load('provenance/checksums.json')
assert not OUT.exists() or (HERE/'build_evidence.json').is_file(),'Refuse to overwrite an unrelated package.'
missing=[n for n in checks if not (ROOT/n).is_file()]
assert not missing,missing
for n,d in checks.items():
    assert (ROOT/n).stat().st_size==d['bytes'] and sha(ROOT/n)==d['sha256'],n
print(f'Original archive verified: {len(checks)} files',flush=True)
manifest=load(f'{CAM}/manifest.json')
selection_path=next((ROOT/'provenance/selection_history').rglob('parameters_without_slo.json')).relative_to(ROOT).as_posix()
selection=load(selection_path)
assert selection['chosen']=={'window_s':60,'margin':0.15}
assert sum(r['eligible'] for r in selection['ranking'])==1
runsets={x:load(f'paper_tables/{x}_runs.json') for x in 'abcdef'}
summaries={x:load(f'paper_tables/{x}_summary.json') for x in 'abcdef'}
expected=dict(a=24,b=48,c=48,d=96,e=42,f=24)
assert {k:len(v) for k,v in runsets.items()}==expected
assert len(summaries['d'])==48 and len(summaries['e'])==18
assert all(r['policy'] in {'E','Static-512','E-NoOpt','E-Least','E-Weighted'} for rs in runsets.values() for r in rs)
bindings={b['run_id']:b for b in manifest['bindings'].values() if b.get('run_id')}
paths={rid:f"{CAM}/{next(a for a in m['attempts'] if a['status']=='complete')['path']}" for rid,m in manifest['runs'].items() if m['status']=='complete'}
grid=[]
for s in manifest['slots']:
    if s['batch']=='grid':
        rid=manifest['bindings'][s['slot_id']]['run_id'];r=load(paths[rid]+'/summary.json');grid.append(r)
assert len(grid)==192
runsets['grid']=grid
all_runs={r['run_id']:r for rs in runsets.values() for r in rs}
assert len(all_runs)==342

# Independent, small read-only reconstruction from all contributing request records.
raw_check_count=0
def quantile(xs,q):
    x=sorted(xs);pos=(len(x)-1)*q;lo=math.floor(pos);hi=math.ceil(pos)
    return x[lo]+(x[hi]-x[lo])*(pos-lo)
for rid,r in all_runs.items():
    rel=paths[rid]+'/requests.jsonl.gz';p=ROOT/rel
    inputs[rel]={'sha256':sha(p),'bytes':p.stat().st_size}
    with gzip.open(p,'rt',encoding='utf-8') as f:rr=[json.loads(line) for line in f]
    assert len(rr)==r['requests'] and len({x['id'] for x in rr})==len(rr)
    assert all(x['generated']==x['output_tokens'] and x['steps']==50 and x['finished_s'] is not None for x in rr)
    lat=[x['finished_s']-x['arrival_s'] for x in rr]
    assert near(quantile(lat,.5),r['p50_s']) and near(quantile(lat,.99),r['p99_s']),rid
    assert near(len(rr)/(max(x['finished_s'] for x in rr)-min(x['arrival_s'] for x in rr)),r['throughput_req_s']),rid
    assert sum(x['finished_s']<=3600 for x in rr)+r['end_backlog']==len(rr)
    ordered=sorted(rr,key=lambda x:(x['arrival_s'],x['id']))
    assert near(sum(x['kind']=='Long' for x in ordered)/len(ordered),r['long_fraction']),rid
    assert sum(a['kind']!=b['kind'] for a,b in zip(ordered,ordered[1:]))==r['type_changes'],rid
    raw_check_count+=len(rr)

# Rebuild each summary independently from the exact run membership.
for exp,ss in summaries.items():
    byid={r['run_id']:r for r in runsets[exp]}
    assert set(byid)=={rid for s in ss for rid in s['run_ids']}
    for s in ss:
        rs=[byid[rid] for rid in s['run_ids']];assert len(rs)==s['repetitions']
        for field,_,_,_ in METRICS.values():
            xs=[r[field] for r in rs]
            for suffix,fn in [('mean',avg),('min',min),('max',max)]:assert near(fn(xs),s[field+'_'+suffix])
print(f'Independent request reconstruction: {len(all_runs)} runs / {raw_check_count} requests',flush=True)

def series_for(exp):
    if exp in ('b','grid'):return 'margin',[0,.05,.1,.15],['M=0','M=0.05','M=0.10','M=0.15']
    if exp=='c':return 'window_s',[10,20,30,60],['W=10秒','W=20秒','W=30秒','W=60秒']
    ps={'a':['E','Static-512'],'d':['E','Static-512'],'e':['E','E-Least','E-Weighted'],'f':['E','E-NoOpt']}[exp]
    return 'policy',ps,ps

def metric_tables(exp):
    rs=runsets[exp]; groups=condition_groups(rs,exp);field_series,series,series_labels=series_for(exp)
    result=[]
    for metric,(field,title,unit,precision) in METRICS.items():
        rows,records=[],[]
        for i,(k,rr) in enumerate(groups,1):
            rowid=f'{exp}-{i:02d}';left=[rowid]+condcells(k)
            if exp=='d':left.append(TRACE[k[3]])
            if exp=='grid':left.append(f'{k[3]:g}')
            cells=[]
            for v,sl in zip(series,series_labels):
                subset=sorted([r for r in rr if r[field_series]==v],key=lambda r:(r.get('request_seed') or 0,r.get('scheduler_seed') or 0))
                assert len(subset) in (1,5),(exp,k,v)
                xs=[r[field] for r in subset];mean=avg(xs)
                cell=fmt(mean,precision) if len(xs)==1 else f'{fmt(mean,precision)} [{fmt(min(xs),precision)}, {fmt(max(xs),precision)}]'
                cells.append(cell)
                records.append(dict(table_id=f'{exp}-{metric}',row_id=rowid,model=k[0],cluster=k[1],rate_per_min=k[2],qps=k[2]/60,series=sl,policy=subset[0]['policy'],trace=subset[0]['trace'],window_s=subset[0].get('window_s'),margin=subset[0].get('margin'),metric=metric,unit=unit,repetitions=len(xs),mean=mean,min=min(xs),max=max(xs),run_ids=';'.join(r['run_id'] for r in subset)))
                if metric=='p50':
                    for r in subset:
                        rid=r['run_id'];ctx=bindings[rid]['context']
                        summary_rel=paths[rid]+'/summary.json'
                        if summary_rel not in inputs:load(summary_rel)
                        provenance.append(dict(experiment=exp,row_id=rowid,series=sl,model=k[0],cluster=k[1],rate_per_min=k[2],qps=k[2]/60,policy=r['policy'],trace=r['trace'],window_s=r.get('window_s'),margin=r.get('margin'),request_seed=r.get('request_seed'),scheduler_seed=r.get('scheduler_seed'),run_id=rid,summary_archive_relative=summary_rel,summary_sha256=inputs[summary_rel]['sha256'],raw_directory=os.path.relpath(ROOT/paths[rid],OUT).replace('\\','/'),requests_sha256=ctx['requests_sha256'],deployment_sha256=ctx['source_sha256'],flip_sha256=ctx['flip_sha256'],profile_sha256=ctx['profile_sha256'],simulation_sha256=ctx['simulation_source_sha256']))
            rows.append(left+cells)
        headers=['条件ID','模型','集群','req/min','QPS']+(['Trace'] if exp=='d' else ['W（秒）'] if exp=='grid' else [])+series_labels
        md=table(headers,rows)
        md_checks.append(dict(path=FOLDERS[exp]+('/完整网格.md' if exp=='grid' else '/数据与分析.md'),table_id=f'{exp}-{metric}',text=md,rows=len(rows)))
        export_csv(FOLDERS[exp]+f'/{metric}.csv',records)
        result.append(f'### 表 {exp}-{metric}：{title}（{unit}）\n\n'+md+f'\n取数：[未舍入CSV]({metric}.csv)。`row_id`对应条件ID，`series`对应结果列；数值列为`mean/min/max`，`repetitions`表示重复次数。'+('本表混合trace或加权调度的区间为五seed观测范围。' if exp in ('d','e') else '本表均为单次确定性运行。')+'\n')
    return '\n'.join(result)

TABLES={x:metric_tables(x) for x in 'abcdef'}
TABLES['grid']=metric_tables('grid')

for exp,seedfield,policy in [('d','request_seed',None),('e','scheduler_seed','E-Weighted')]:
    rs=[r for r in runsets[exp] if r.get(seedfield) is not None]
    assert len(rs)==(60 if exp=='d' else 30)
    detailed=[]
    for r in sorted(rs,key=lambda r:sortkey(r)+(r[seedfield],r['policy'])):
        detailed.append(dict(model=r['generator'],cluster=r['scenario'],rate_per_min=r['rate_per_min'],qps=r['rate_per_min']/60,policy=r['policy'],trace=r['trace'],request_seed=r.get('request_seed'),scheduler_seed=r.get('scheduler_seed'),p50_s=r['p50_s'],p99_s=r['p99_s'],throughput_req_s=r['throughput_req_s'],run_id=r['run_id']))
    export_csv(FOLDERS[exp]+'/seed_details.csv',detailed)
    text=f'# 实验{exp}：五seed逐运行明细\n\n每个值对应一次独立运行。均值、最小值、最大值见[主表](数据与分析.md)；这里不合并请求重新计算分位数。精确数值及run ID见[seed_details.csv](seed_details.csv)。\n'
    for metric,(field,title,unit,prec) in METRICS.items():
        rows=[]
        for k,rr in condition_groups(rs,exp):
            if exp=='d':
                for seed in range(5):rows.append(condcells(k)+[seed]+[fmt(choose(rr,request_seed=seed,policy=p)[field],prec) for p in ['E','Static-512']])
            else:rows.append(condcells(k)+[fmt(choose(rr,scheduler_seed=seed)[field],prec) for seed in range(5)])
        headers=['模型','集群','req/min','QPS']+(['排列seed','E','Static-512'] if exp=='d' else [f'调度seed={s}' for s in range(5)])
        block=table(headers,rows);md_checks.append(dict(path=FOLDERS[exp]+'/五seed明细.md',table_id=f'{exp}-seed-{metric}',text=block,rows=len(rows)))
        text+=f'\n## 表 {exp}-seed-{metric}：{title}（{unit}）\n\n'+block
    write(FOLDERS[exp]+'/五seed明细.md',text)

COMMON='输入128 token，Short/Long输出512/2048 token，DiT为50步；2024-10-18 06:00–07:00 UTC，一小时等间隔到达后完整排空。监控周期固定10秒。除本实验主动改变的参数外，E采用W=60秒、M=0.15、预计完成时间调度、restricted正反转换与全部启动优化。初始实例已就绪。'
SETUP={
'a':'两模型×Cluster1/2×三负载。E与Static-512使用同一请求清单、初始bin1及profile。Static-512采用least_waiting（仅计等待请求）、禁止flip，window/margin不适用于其业务策略。调度和转换同时变化，因此本实验衡量完整系统效果。',
'b':'两模型×Cluster1/2×三负载，只比较E；固定W=60秒，改变M=0/0.05/0.10/0.15。分析中的参照为M=0；全局采用M=0.15并不意味着每个条件或metric都以它为最优。',
'c':'两模型×Cluster1/2×三负载，只比较E；固定M=0.15，改变W=10/20/30/60秒。分析中的参照为W=10秒。所有配置仍每10秒监控一次，因此变化的是观测历史范围，不是监控周期。',
'd':'两个模型均用Cluster1，三个负载，四类trace，E与Static-512。15/30/60秒多数票各为一次确定性运行；真实比例混合按分钟累计配额、半偶舍入并在分钟内排列，排列seed=0–4。两策略在同seed下读取相同请求清单。',
'e':'两个模型均用Cluster1，三个负载，60秒多数票。E、E-Least、E-Weighted分别使用estimated_completion、least_waiting和capacity_weighted；全部保留相同选源规则、flip和优化启动。前两者确定性各一次，加权调度使用seed=0–4；请求清单不随调度seed改变。',
'f':'两模型×Cluster2/ClusterSimu×三负载，60秒多数票，只比较E与E-NoOpt。两者均采用estimated_completion与restricted转换；E的PE和生成模型启动全部优化，E-NoOpt全部使用非优化启动。该实验不分离PE与生成模型优化的独立贡献。'}
PURPOSE={
'a':'完整PAVE在四个模型/集群场景和三档负载下，能否改善固定预配置的端到端表现？',
'b':'margin能否减少不必要的状态切换，且这种变化是否转化为吞吐和时延改善？',
'c':'较长的观测窗口能否改善系统决策，收益是否跨模型、集群和负载一致？',
'd':'多数票聚合粒度变化，以及按真实比例混合请求，分别如何改变输入与系统表现？',
'e':'在相同转换机制下，按预计完成时间调度能否改善吞吐与尾时延，又有何中位时延取舍？',
'f':'profile层面的启动时延减少，能否在闭环系统中体现为更好的吞吐与端到端时延？'}
CONCLUSIONS={}
ANALYSIS={}
TAKE={}

# Supporting diagnostics are separate from the three complete metric tables.
DIAG_FIELDS=['requests','long_fraction','type_changes','observed_completed','observed_throughput_req_s','end_backlog','drain_s','flip_triggered','converted_groups','gpu_unavailable_s','pe_reprefill_s']
for exp in 'abcdef':
    rows=[]
    for r in sorted(runsets[exp],key=lambda r:sortkey(r)+(r['policy'],r['trace'],r.get('request_seed') or 0,r.get('scheduler_seed') or 0)):
        row={k:r.get(k) for k in ['generator','scenario','rate_per_min','policy','trace','window_s','margin','request_seed','scheduler_seed','run_id']}
        row.update({k:r.get(k) for k in DIAG_FIELDS})
        for stage in ['PE','TE','DiT','VAE']:
            row[stage+'_waiting_s_per_request']=r['stages'][stage]['waiting_s']/r['requests']
            row[stage+'_executed_work']=r['stages'][stage]['executed_work']
        rows.append(row)
    export_csv(FOLDERS[exp]+'/diagnostics.csv',rows)

def pair_stats(exp,base_policy):
    pairs=[]
    for k,rr in condition_groups(runsets[exp],exp):pairs.append((k,choose(rr,policy='E'),choose(rr,policy=base_policy)))
    return pairs
ap=pair_stats('a','Static-512')
ag={m:[gain(e[field],b[field],field) for k,e,b in ap] for m,(field,*_) in METRICS.items()}
CONCLUSIONS['a']=f"12个条件中，E全部提高排空吞吐并降低p50、p99。吞吐改善{range_text(ag['throughput'])}%，条件等权算术平均{avg(ag['throughput']):.2f}%；p50降低{range_text(ag['p50'])}%，p99降低{range_text(ag['p99'])}%。优势跨四个场景出现，但大小并不一致，高负载下仍有明显积压。"
a_para=['### 从整体优势到场景差异\n\n上述改善逐条件计算，再对12条件等权汇总；没有将不同模型的原始吞吐混在一起计算所谓总体容量。按模型/集群分组，可以看到以下区别：\n']
for key,pp in sorted(group(ap,lambda p:p[0][:2]).items(),key=lambda kv:(MODELS.index(kv[0][0]),CLUSTERS.index(kv[0][1]))):
    tp=[gain(e['throughput_req_s'],b['throughput_req_s'],'throughput_req_s') for k,e,b in pp]
    lo,hi=pp[0][1],pp[-1][1];stage=max(hi['stages'],key=lambda s:hi['stages'][s]['waiting_s']);share=hi['stages'][stage]['waiting_s']/hi['total_wait_s']*100
    a_para.append(f"**{LABEL[key[0]]}／{LABEL[key[1]]}。** 三负载吞吐改善为{range_text(tp)}%。E从{lo['rate_per_min']:g}增至{hi['rate_per_min']:g} req/min时，p99由{lo['p99_s']:.2f}升至{hi['p99_s']:.2f}秒，积压由{lo['end_backlog']}增至{hi['end_backlog']}。最高负载累计阶段等待中，{stage}占{share:.2f}%。这表明获益与拥塞可以同时存在。\n")
amax=max(ap,key=lambda p:gain(p[1]['throughput_req_s'],p[2]['throughput_req_s'],'throughput_req_s'))
amin=min(ap,key=lambda p:gain(p[1]['throughput_req_s'],p[2]['throughput_req_s'],'throughput_req_s'))
mid=choose(runsets['a'],generator=MODELS[1],scenario='cluster1',rate_per_min=14.,policy='E');high=choose(runsets['a'],generator=MODELS[1],scenario='cluster1',rate_per_min=16.,policy='E')
a_para+=['### 三个有解释价值的条件\n',f"**最大相对收益：{label(amax[0])}。** E吞吐{amax[1]['throughput_req_s']:.6f}，Static-512为{amax[2]['throughput_req_s']:.6f} req/s。固定配置在Long占主导的合成请求中累积队列，E的完整策略改善了此配置失配；本对照不能分解其中调度和转换的贡献。\n",f"**最小收益：{label(amin[0])}。** 吞吐仍改善{gain(amin[1]['throughput_req_s'],amin[2]['throughput_req_s'],'throughput_req_s'):.2f}%，但显著小于另外一些场景。论文应保留这个范围下界，避免只引用最大值。\n",f"**负载趋势例外：Wan2.1／Cluster1。** 从14到16 req/min，排空吞吐由{mid['throughput_req_s']:.6f}降至{high['throughput_req_s']:.6f}，而观察期吞吐为{mid['observed_throughput_req_s']:.6f}和{high['observed_throughput_req_s']:.6f} req/s。额外排空由{mid['drain_s']:.2f}升至{high['drain_s']:.2f}秒，说明排空分母扩大与吞吐下降同时发生。记录支持持续积压和PE等待的解释，不能据此认定一小时内处理能力同比下降。\n"]
ANALYSIS['a']='\n'.join(a_para)
TAKE['a']='可写：在本小时构造、所选负载和四场景下，完整PAVE改善全部12条件的三个指标；给范围和条件等权均值。应同时说明高负载仍有队列和长尾。不能写成纯flip贡献，或对任意trace均成立。取数用a-p50/a-p99/a-throughput的a-01至a-12；四场景均保留，每场景按三负载连接。'

def parameter_analysis(exp,field,values,base,chosen):
    groups=condition_groups(runsets[exp],exp);scores=[]
    for v in values:
        rr=[choose(x,**{field:v}) for _,x in groups]
        tp=[r['throughput_req_s']/max(t['throughput_req_s'] for t in x) for r,(_,x) in zip(rr,groups)]
        best=sum(near(r['throughput_req_s'],max(t['throughput_req_s'] for t in x)) for r,(_,x) in zip(rr,groups))
        scores.append([f'{v:g}',fmt(math.exp(avg([math.log(t) for t in tp])),6),best,fmt(avg([r['flip_triggered'] for r in rr]))])
    comparisons=[]
    for k,rr in groups:
        b=choose(rr,**{field:base});n=choose(rr,**{field:chosen})
        comparisons.append((k,n,b,{m:gain(n[f],b[f],f) for m,(f,*_) in METRICS.items()}))
    lines=[]
    for m,(_,title,*_) in METRICS.items():
        xs=[p[3][m] for p in comparisons];up,tie,down=direction_counts(xs)
        lines.append(f'{title}改善范围{range_text(xs)}%，改善/数值持平/退化条件数为{up}/{tie}/{down}。')
    # Deterministic example selection: largest TP gain, worst TP gain, strongest p50 tradeoff.
    examples=[max(comparisons,key=lambda p:p[3]['throughput']),min(comparisons,key=lambda p:p[3]['throughput']),min(comparisons,key=lambda p:p[3]['p50'])]
    seen=set();details=[]
    for k,n,b,g in examples:
        if k in seen:continue
        seen.add(k)
        details.append(f"**{label(k)}。** 参照→统一参数：吞吐{b['throughput_req_s']:.6f}→{n['throughput_req_s']:.6f} req/s，p50 {b['p50_s']:.2f}→{n['p50_s']:.2f}秒，p99 {b['p99_s']:.2f}→{n['p99_s']:.2f}秒，flip {b['flip_triggered']}→{n['flip_triggered']}次。吞吐改善{g['throughput']:.2f}%，p50降低{g['p50']:.2f}%，p99降低{g['p99']:.2f}%；负值表示退化。")
    return lines,scores,details,comparisons

for exp,field,vs,bv,cv in [('b','margin',[0,.05,.1,.15],0,.15),('c','window_s',[10,20,30,60],10,60)]:
    lines,scores,details,cmp=parameter_analysis(exp,field,vs,bv,cv)
    CONCLUSIONS[exp]=('M=0.15在固定60秒窗口的切片上取得更高的综合吞吐评分并减少平均flip，但并非逐条件、逐指标单调改善。' if exp=='b' else 'W=60秒的综合吞吐优势主要来自Wan2.1／Cluster1中高负载；其他场景常有短窗口更优的条件，因此统一长窗口不是逐场景最优。')+' '.join(lines)
    name='margin' if exp=='b' else 'window（秒）'
    analysis='### 整体趋势与个体差异\n\n'+table([name,'切片吞吐几何平均评分','吞吐最优或并列条件数','平均flip次数'],scores)+'\n评分先在每个条件内除以该切片最高吞吐，再取12条件几何平均；这里用于说明切片趋势，与完整16组合的选参评分不同。\n\n'
    for model in MODELS:
        mm=[x for x in cmp if x[0][0]==model]
        analysis+=f"{LABEL[model]}六条件的吞吐相对参照改善为{range_text([p[3]['throughput'] for p in mm])}%。"
    analysis+='\n\n### 代表条件及例外\n\n'+'\n\n'.join(details)+'\n\n### 机制与解释边界\n\n'
    if exp=='b':
        analysis+='margin增大使Short/Long切换阈值相距更远，日志显示平均转换减少；但减少一次转换可能节省启动成本，也可能推迟有利的部署变化。吞吐、p50、p99并非始终同向，不能只凭flip减少判断改进。当前数据支持M=0.15作为统一折中，不支持margin越大越好。'
        TAKE[exp]='正文可以先给M=0.15相对无margin的整体结果，再给一个退化或非单调条件。使用b-01至b-12和四个M列；三个metric均有完整证据。W固定60秒，不能据此声称任意window下相同结论。完整参数交互见附录，不重新挑选有利参数。'
    else:
        analysis+='所有运行的监控周期均为10秒。更长window改变完成PE样本的历史覆盖范围，也改变能力估计和动态调度所用信息；平滑与响应滞后同时可能发生。平均flip下降是观测事实，尚不能将性能变化归结为其中一个单独机制。12条件联合调参说明统一选择，不能替代跨trace泛化验证。'
        TAKE[exp]='使用c-01至c-12、四个W列，重点说明统一参数收益集中与场景差异。不能把横轴window当作监控周期，也不能逐模型替换成其最优W后仍称全局参数。需要讨论W/M交互时引用完整192网格，而不是仅用两个切片推断。'
    ANALYSIS[exp]=analysis

ds=summaries['d']
CONCLUSIONS['d']='多数票的Long区间占比接近，但类型变化次数明显不同，性能也并非处处不敏感。真实比例混合降低了Long工作量占比并打散请求顺序，因而应作为另一种负载构造检验，不能当作只改变聚合频率的对照。'
dtext='### 先核对输入，再解释性能\n\n'+table(['构造','Long区间占比','区间类型变化','含义'],[['15秒多数票','84.1667%','45','桶内采用单一类型'],['30秒多数票','85.0000%','26','桶内采用单一类型'],['60秒多数票','86.6667%','16','主实验构造'],['真实比例混合','不适用','逐seed请求顺序决定','按分钟真实比例分配并排列']])+'\n多数票表中的占比是时间桶占比。实际到达清单的Long比例、相邻请求类型变化和原始trace的桶变化分别记录；请求数较少时，一些短桶可能没有到达请求。\n\n'
inputrows=[]
for k,rr in condition_groups(runsets['d'],'d'):
    rr=[r for r in rr if r['policy']=='E']
    inputrows.append(condcells(k)+[TRACE[k[3]],rr[0]['requests'],f"{avg([r['long_fraction'] for r in rr])*100:.4f}%",range_text([r['type_changes'] for r in rr],0)])
dtext+=table(['模型','集群','req/min','QPS','Trace','请求数','实际Long比例','请求类型变化范围'],inputrows)
dtext+='\n### 多数票内部的比较\n\n'
for model in MODELS:
    lower=sorted({r['rate_per_min'] for r in ds if r['generator']==model})[:2]
    dtext+=f"{LABEL[model]}的低、中负载分别观察："
    fragments=[]
    for rate in lower:
        rr=[r for r in ds if r['generator']==model and r['rate_per_min']==rate and r['policy']=='E' and r['trace'].startswith('majority')]
        fragments.append(f"{rate:g} req/min下吞吐{range_text([r['throughput_req_s_mean'] for r in rr],6)} req/s、p99 {range_text([r['p99_s_mean'] for r in rr])}秒")
    dtext+='；'.join(fragments)+'。吞吐区间接近不意味着尾时延也相同，应分别读取三个metric。\n\n'
for model in MODELS:
    # Highest load is the predeclared stress condition, not a favorable selection.
    rate=max(r['rate_per_min'] for r in ds if r['generator']==model)
    rr=[r for r in ds if r['generator']==model and r['rate_per_min']==rate and r['policy']=='E' and r['trace'].startswith('majority')]
    dtext+=f"**{LABEL[model]}／Cluster1、{rate:g} req/min。** 三个多数票构造的吞吐范围为{range_text([r['throughput_req_s_mean'] for r in rr],6)} req/s，p99范围为{range_text([r['p99_s_mean'] for r in rr])}秒，平均flip范围为{range_text([r['flip_triggered_mean'] for r in rr],0)}次。"
    if model==MODELS[1]:dtext+='这一最高负载条件是“聚合粒度影响很小”说法的重要反例。'
    dtext+='\n\n'
mix=[r for r in ds if r['trace']=='mixture' and r['policy']=='E']
dtext+='### 真实比例混合单独解释\n\n'
dtext+=f"混合构造实际Long比例在不同负载下为{range_text([r['long_fraction_mean']*100 for r in mix],4)}%，明显低于多数票的时间占比。此处变化包含总token工作量、请求聚集方式和排列随机性，不能从混合与多数票的差异反推出纯flip频次效应。\n\n"
hi=choose(ds,generator=MODELS[1],scenario='cluster1',rate_per_min=16.,policy='E',trace='mixture')
dtext+=f"**Wan2.1／Cluster1、16 req/min的混合结果。** 五seed吞吐均值{hi['throughput_req_s_mean']:.6f}，范围[{hi['throughput_req_s_min']:.6f}, {hi['throughput_req_s_max']:.6f}] req/s；p99均值{hi['p99_s_mean']:.2f}，范围[{hi['p99_s_min']:.2f}, {hi['p99_s_max']:.2f}]秒。范围体现同配额不同顺序的差异，不是置信区间。Static-512也读取相同清单，可用于检查工作量构造改变后固定部署的绝对表现。\n\n"
static_mix=choose(ds,generator=MODELS[1],scenario='cluster1',rate_per_min=16.,policy='Static-512',trace='mixture')
dtext+=f"同一混合条件的Static-512吞吐均值为{static_mix['throughput_req_s_mean']:.6f} req/s，p50为{static_mix['p50_s_mean']:.2f}秒，p99为{static_mix['p99_s_mean']:.2f}秒；对应E的p50为{hi['p50_s_mean']:.2f}秒。这些是相同输入下的原始值对照，不能与另一个trace的结果直接归因为单一构造因素。\n\n"
dtext+='细粒度诊断见diagnostics.csv，按run ID可进一步查看阶段等待与排空。当前表保留全部96条运行的汇总或seed入口，没有对不同trace归一化或计算相对收益。'
ANALYSIS['d']=dtext
TAKE['d']='可写：Long时间占比接近不保证聚合粒度性能不变；多数票与真实比例需分开解释。引用d-01至d-24的原始指标和五seed范围。真实比例使用主表汇总时必须标注5 seeds；核查个例使用五seed明细。不能声称仅改变聚合频率，或将区间切换数当实际flip数。'

es=summaries['e'];ep=[]
for k,rr in condition_groups(es,'e'):ep.append((k,choose(rr,policy='E'),choose(rr,policy='E-Least'),choose(rr,policy='E-Weighted')))
eg_l=[gain(e['throughput_req_s_mean'],l['throughput_req_s_mean'],'throughput_req_s') for k,e,l,w in ep]
eg_w=[gain(e['throughput_req_s_mean'],w['throughput_req_s_mean'],'throughput_req_s') for k,e,l,w in ep]
assert all(e['throughput_req_s_mean']>max(l['throughput_req_s_mean'],w['throughput_req_s_mean']) and e['p99_s_mean']<min(l['p99_s_mean'],w['p99_s_mean']) for k,e,l,w in ep)
eworse=sum(e['p50_s_mean']>l['p50_s_mean'] for k,e,l,w in ep)
CONCLUSIONS['e']=f"按策略均值比较，E在六条件中均取得最高吞吐和最低p99；相对E-Least吞吐改善{range_text(eg_l)}%，相对E-Weighted均值改善{range_text(eg_w)}%。但E的p50在{eworse}/6条件下高于E-Least，且不能称吞吐超过每一个随机seed。"
etext='### 总体排序不等于所有metric全面占优\n\n预计完成时间评分考虑运行剩余工作、等待任务和本实例估计能力，least_waiting只比较等待请求数。结果支持当前配置下的吞吐与尾时延优势，同时暴露中位时延取舍。capacity_weighted按静态容量随机分配，不直接使用实时队列；五seed差异应与均值一起保留。\n\n### 三个代表条件\n\n'
selected=[max(ep,key=lambda p:gain(p[1]['throughput_req_s_mean'],p[2]['throughput_req_s_mean'],'throughput_req_s')),min(ep,key=lambda p:gain(p[1]['p50_s_mean'],p[2]['p50_s_mean'],'p50_s')),next(p for p in ep if p[0][0]==MODELS[1] and p[0][2]==16)]
for k,e,l,w in selected:
    etext+=f"**{label(k)}。** E/E-Least的吞吐为{e['throughput_req_s_mean']:.6f}/{l['throughput_req_s_mean']:.6f} req/s，p50为{e['p50_s_mean']:.2f}/{l['p50_s_mean']:.2f}秒，p99为{e['p99_s_mean']:.2f}/{l['p99_s_mean']:.2f}秒。加权调度吞吐均值{w['throughput_req_s_mean']:.6f}、最大值{w['throughput_req_s_max']:.6f} req/s。"
    if w['throughput_req_s_max']>e['throughput_req_s_mean']:etext+='其中至少一个加权seed高于E；均值排序不能扩展为逐seed支配。'
    etext+='\n\n'
alloc=load('paper_tables/e_allocations.json');eid={r['run_id']:r for r in runsets['e']}
alloc_export=[];hardware=collections.defaultdict(lambda:collections.defaultdict(float))
for r in alloc:
    if r['stage'] not in ('PE','DiT'):continue
    parent=eid[r['run_id']]
    alloc_export.append(dict(model=parent['generator'],cluster=parent['scenario'],rate_per_min=parent['rate_per_min'],policy=parent['policy'],scheduler_seed=parent.get('scheduler_seed'),**{k:r[k] for k in ['run_id','stage','placement_id','hardware','template','received','completed','executed_work','service_s','work_unit','received_share','completed_share','executed_work_share']}))
    hardware[(r['run_id'],r['stage'])][r['hardware']]+=r['executed_work']
export_csv(FOLDERS['e']+'/instance_work.csv',alloc_export)
hwrows=[]
for k,e,l,w in ep:
    for s in [e,l,w]:
        vals=[]
        for rid in s['run_ids']:
            hh=hardware[rid,'DiT'];vals.append(hh.get('h100',0)/sum(hh.values())*100)
        pe_wait=s['stages.PE.waiting_s_mean']/s['requests_mean'];dit_wait=s['stages.DiT.waiting_s_mean']/s['requests_mean']
        hwrows.append([LABEL[k[0]],f'{k[2]:g}',s['policy'],fmt(avg(vals)) if len(vals)==1 else f'{avg(vals):.2f} [{min(vals):.2f}, {max(vals):.2f}]',fmt(pe_wait),fmt(dit_wait),fmt(s['flip_triggered_mean'])])
etext+='### 工作分配、等待与转换的联动\n\n'+table(['模型','req/min','策略','H100承担DiT steps %','PE等待秒/请求','DiT等待秒/请求','平均flip'],hwrows)
etext+='\n这里每个seed先计算工作占比，再作均值和范围汇总；没有把不同seed请求拼在一起。接收次数包含迁移重试，completed是模块完成数量，executed_work是实际token或step。逐物理位置数据见instance_work.csv，零任务实例保留。H100工作占比变化、等待与尾时延之间的对应关系支持异构调度解释，但比例本身不等于负载均衡或容量利用率。\n\n调度改变了队列，进而可能改变选源集合和flip轨迹，因此这是替换调度器对完整PAVE的影响，不能称为固定部署微基准，也不能将其收益与其他消融收益直接相加。'
ANALYSIS['e']=etext
TAKE['e']='首要可取结论是吞吐/尾时延改善伴随p50取舍。用e-01至e-06三个策略列；随机加权用mean及min/max，保留有利seed例外。若论文只展示吞吐，应在相邻文字或表中交代p50并非全面更优。讨论异构分配时引用instance_work.csv，明确使用实际steps而非接收次数。'

fp=pair_stats('f','E-NoOpt');fg={m:[gain(e[field],b[field],field) for k,e,b in fp] for m,(field,*_) in METRICS.items()}
up,tie,down=direction_counts(fg['throughput'])
CONCLUSIONS['f']=f"全部启动优化的吞吐变化为{range_text(fg['throughput'])}%，12条件中改善{up}项、数值持平{tie}项、退化{down}项。Cluster2吞吐差异很小，ClusterSimu的变化更大且存在时延取舍；缩短启动时间并不保证系统吞吐单调提高。"
for metric in ('p50','p99'):
    up,tie,down=direction_counts(fg[metric])
    CONCLUSIONS['f']+=f"{metric}降低比例为{range_text(fg[metric])}%，改善/数值持平/退化为{up}/{tie}/{down}。"
ftext='### 从profile到实际系统：应核对哪一层\n\n所有目标实例按策略选用对应优化或非优化profile。profile差值描述一次启动的时间减少；整次运行的资源损失还取决于启动次数、安全等待和转换轨迹。初始部署已就绪，此实验不包含初始冷启动。\n\n'
startups=load('paper_tables/f_startups.json');groups=load('paper_tables/f_groups.json');periods=load('paper_tables/f_direction_periods.json')
startup_profiles={}
for s in startups:startup_profiles[(s['generator'],s['scenario'],s['hardware'],s['template'])]=(s['non_optimized_profile_s'],s['optimized_profile_s'])
ftext+=table(['模型','集群','目标硬件/模板','非优化秒','优化秒','每次减少秒'],[[LABEL[k[0]],LABEL[k[1]],k[2]+'/'+k[3],fmt(v[0],6),fmt(v[1],6),fmt(v[0]-v[1],6)] for k,v in sorted(startup_profiles.items())])
profile_rows=[]
for s in startups:profile_rows.append({k:s[k] for k in ['generator','scenario','policy','rate_per_min','run_id','flip_id','group_id','direction','hardware','template','optimized','optimized_profile_s','non_optimized_profile_s','actual_startup_s','launch_started_s','ready_s']})
export_csv(FOLDERS['f']+'/startup_instances.csv',profile_rows)
export_csv(FOLDERS['f']+'/direction_periods.csv',periods)
grows=[]
for g in groups:grows.append({**{k:g[k] for k in ['generator','scenario','policy','rate_per_min','run_id','flip_id','group_id','direction','node','safe_wait_s','launch_started_s','ready_s','startup_elapsed_s']},'gpu_ids':';'.join(map(str,g['gpu_ids'])),'gpu_count':len(g['gpu_ids']),'unavailable_gpu_s':len(g['gpu_ids'])*(g['safe_wait_s']+g['startup_elapsed_s'])})
export_csv(FOLDERS['f']+'/startup_groups.csv',grows)
ftext+='\n每个Comb是一条物理实例启动记录；PE子实例可以并行就绪。`startup_instances.csv`记录每实例时间，`startup_groups.csv`记录组等待与GPU·s，`direction_periods.csv`按观察期/排空期分组。不能将实例时间之和当并行组启动时长，或将ClusterSimu全部192张卡当参与转换的卡。\n\n### 按集群与负载看系统响应\n\n'
for model in MODELS:
    for cl in ['cluster2','clustersimu']:
        pp=[p for p in fp if p[0][:2]==(model,cl)]
        vals=[gain(e['throughput_req_s'],b['throughput_req_s'],'throughput_req_s') for k,e,b in pp]
        ftext+=f"**{LABEL[model]}／{LABEL[cl]}。** 低、中、高负载吞吐改善依次为"+'、'.join(f'{x:.4f}%' for x in vals)+'。'
        if cl=='cluster2':ftext+='吞吐差异接近零时，应检查尾时延及阶段队列，而不是直接写启动优化没有执行。'
        else:ftext+='需要结合中位时延、尾时延与排空分别解释，不能仅取最有利负载。'
        ftext+='\n\n'
ftext+='### 三个机制案例\n\n'
f_cases=[next(p for p in fp if p[0]==(MODELS[0],'cluster2',5.5)),max(fp,key=lambda p:gain(p[1]['throughput_req_s'],p[2]['throughput_req_s'],'throughput_req_s')),min(fp,key=lambda p:gain(p[1]['throughput_req_s'],p[2]['throughput_req_s'],'throughput_req_s'))]
for k,e,b in f_cases:
    mainstage=max(e['stages'],key=lambda s:e['stages'][s]['waiting_s'])
    ftext+=f"**{label(k)}，E-NoOpt→E。** flip {b['flip_triggered']}→{e['flip_triggered']}次；累计不可用{b['gpu_unavailable_s']:.2f}→{e['gpu_unavailable_s']:.2f} GPU·s；{mainstage}平均等待{b['stages'][mainstage]['waiting_s']/b['requests']:.2f}→{e['stages'][mainstage]['waiting_s']/e['requests']:.2f}秒/请求。排空吞吐{b['throughput_req_s']:.6f}→{e['throughput_req_s']:.6f}，观察期吞吐{b['observed_throughput_req_s']:.6f}→{e['observed_throughput_req_s']:.6f} req/s，额外排空{b['drain_s']:.2f}→{e['drain_s']:.2f}秒；p50降低{gain(e['p50_s'],b['p50_s'],'p50_s'):.2f}%，p99降低{gain(e['p99_s'],b['p99_s'],'p99_s'):.2f}%。\n\n"
ftext+='**记录直接显示**：单次启动更快与整次运行的吞吐、p50、p99并非必然同向。**与记录一致的解释**：容量恢复会改变后续排队、调度和转换，可能改善最后完成时间，却提高某些中间请求的等待；也可能观察期稍好而排空更差。**现有实验不能验证**：每条因果路径的独立贡献，因为没有固定队列或转换轨迹的控制运行。'
ANALYSIS['f']=ftext
TAKE['f']='取f-01至f-12两策略列，保留全部负载。可写“启动优化的系统收益依赖场景，profile节省不等同于端到端获益”，并分别给持平、改善和退化条件。不能从E/E-NoOpt推断DiT单独无效，不能将跨集群差异归结为SP并行度，不能与调度收益直接相加。'

for exp in 'abcdef':
    title={'a':'主实验','b':'Margin敏感性','c':'Window敏感性','d':'Trace构造','e':'请求调度器','f':'启动优化'}[exp]
    text=f'# 实验{exp}：{title}\n\n[返回总览](../阅读指南与结论总览.md)\n\n## 1. 实验目的与比较设计\n\n{PURPOSE[exp]}\n\n{SETUP[exp]}\n\n{COMMON}\n\n## 2. 先给结论\n\n{CONCLUSIONS[exp]}\n\n## 3. 完整原始指标\n\n吞吐越高越好，p50/p99越低越好；均为完整排空后的请求集合。QPS仅作负载显示，原请求生成使用req/min。'+('区间记为“均值 [最小值, 最大值]”；[五seed明细](五seed明细.md)保留每次运行。' if exp in ('d','e') else '')+'\n\n'+TABLES[exp]+f'\n## 4. 自顶向下分析\n\n{ANALYSIS[exp]}\n\n## 5. 论文取材与结论范围\n\n{TAKE[exp]}\n\n## 6. 来源与核对\n\n'
    text+=f'本实验引用{expected[exp]}条运行记录'+(f'，形成{len(summaries[exp])}条策略/条件汇总' if exp in ('d','e') else '')+'。来源为原归档`paper_tables/'+exp+'_runs.json`与对应summary；主要metric已从请求记录独立重算。\n\n'
    text+='[数据来源索引](../provenance/数据来源索引.csv)以实验、条件ID、结果列和seed定位原run ID、摘要及原始记录目录；同一运行跨实验复用不产生新记录。[诊断CSV](diagnostics.csv)含观察期吞吐、积压、排空、flip及四阶段等待。精确CSV中的mean/min/max保留数值精度，主表不用于计算。\n\n'
    text+='以上为已批准simulation模型内部的结果：零迁移传输成本、Comb模块无干扰等假设仍适用；没有真实GPU性能验证，也没有独立小时泛化验证。\n'
    write(FOLDERS[exp]+'/数据与分析.md',text)

write(FOLDERS['grid']+'/完整网格.md','# 完整W×M参数网格\n\n[返回总览](../阅读指南与结论总览.md) · [选参依据](选参依据.md)\n\n两个模型×Cluster1/2×三负载×四window×四margin，共192次E运行。每个metric表48行，每行四个margin；监控周期固定10秒。b/c是本数据的固定参数切片。三个表使用相同条件ID，不在此重新调参。\n\n'+TABLES['grid'])
rank=selection['ranking'];cut=max(r['throughput_score'] for r in rank)*.995
# Independently confirm, do not mutate selection receipt.
gridgroups=group(grid,cond)
for r in rank:
    rr=[x for x in grid if x['window_s']==r['window_s'] and x['margin']==r['margin']]
    score=math.exp(avg([math.log(x['throughput_req_s']/max(y['throughput_req_s'] for y in gridgroups[cond(x)])) for x in rr]))
    p99score=math.exp(avg([math.log(x['p99_s']/min(y['p99_s'] for y in gridgroups[cond(x)])) for x in rr]))
    assert near(score,r['throughput_score']) and near(p99score,r['p99_score'])
ranking=table(['排名','W秒','M','吞吐几何平均评分','归一化p99评分','进入99.5%范围'],[[i,r['window_s'],r['margin'],fmt(r['throughput_score'],9),fmt(r['p99_score'],9),'是' if r['eligible'] else '否'] for i,r in enumerate(rank,1)])
write(FOLDERS['grid']+'/选参依据.md',f'''# 全局参数的已有选择依据

[完整数据](完整网格.md) · [返回总览](../阅读指南与结论总览.md)

现有选择为W=60秒、M=0.15。此处重建已归档的无SLO复核，只校验已有结论，不改变选择文件或实验身份。

每个模型/集群/负载条件内，将16组合吞吐除以本条件最大吞吐，再对12条件取几何平均。候选门槛为最高评分的99.5%，本次为{cut:.9f}。只剩60秒/0.15一项，所以p99及较小W/M的后续消歧规则没有改变结果。

{ranking}
归一化p99的分母为本条件16组合的最低p99，分数越低越好。吞吐归一化仅用于统一参数选择，不应用于trace实验的交付表。

该选择对12条件等权，不按请求数或GPU数加权。b/c所示四值切片的评分与完整16组合评分分母不同，不能直接混为一张排行榜。

主trace同时用于调参与展示，应称目标工作负载上的统一调优结果；不能称独立泛化验证。d/e采用按主实验相对固定基准收益筛选的Cluster1，也应交代案例选择。

来源：原完整归档`{selection_path}`；逐run ID见数据来源索引中experiment=grid。未使用SLO进行本次分析。
''')

export_csv('provenance/数据来源索引.csv',provenance)
overview_rows=[]
for x in 'abcdef':
    exception={'a':'高负载仍可能积压且排空吞吐下降','b':'M较大并非逐条件单调更好','c':'收益集中，短窗口在一些场景更优','d':'Long占比接近也不能推断性能不变','e':'p50取舍及有利加权seed必须保留','f':'启动更快仍可能吞吐或时延退化'}[x]
    short={'a':'12条件的吞吐、p50、p99均改善','b':'统一M=0.15减少平均flip，存在例外','c':'统一W=60秒的优势主要来自特定高负载','d':'多数票粒度与真实比例需分开解释','e':'E的吞吐和p99在策略均值上占优','f':'系统收益依赖负载和集群'}[x]
    overview_rows.append([x,PURPOSE[x],short,exception,f'[数据与分析]({FOLDERS[x]}/数据与分析.md)'])
guide='''# PAVE实验数据阅读指南与结论总览

本包将已经完成的a–f及完整192次参数网格按实验整理。建议先读各实验“先给结论”，再看对应metric完整表；选择论文论点时再读分组、例外和机制解释。

每个实验的三张主要表分别是p50、p99和排空吞吐。取精确数据请使用同目录CSV，不手工抄录Markdown中的舍入数值。所有实验已完成，本次没有新增simulation。

## 1. 实验分别回答什么

'''+table(['实验','研究问题','主要发现','必须保留的限定','入口'],overview_rows)+'''
## 2. 方法、单位与阅读规则

两个模型为Wan2.2 TI2V 5B与Wan2.1 T2V 1.3B。Cluster1/2均为8×A100 80GB与8×H100 80GB，模板集合不同；ClusterSimu为192张异构GPU，仅用于启动消融。E从512部署启动，使用预计完成时间调度、restricted正反转换和全部启动优化。Static-512固定512部署，按等待请求数调度，禁止flip。

主小时为2024-10-18 06:00–07:00 UTC。原始generated tokens的全文件p50为98，Short<98、Long≥98；512与2048是仿真采用的代表性输出长度，不是原始数据中位数/最大值。多数票Tie继承前桶，首次Tie用Short。输入128 token，DiT为50步。

主小时一分钟多数票的Long区间占比86.6667%，类型切换16次。请求等间隔在[0,3600秒)进入，随后完整排空；窗口监控及转换在排空期间仍可发生。

'''
rates=manifest['configuration']['rates']
guide+=table(['模型','集群','三档req/min','对应QPS'],[[LABEL[m],LABEL[c],'、'.join(f'{r:g}' for r in rates[m][c]),'、'.join(f'{r/60:.6f}' for r in rates[m][c])] for m in MODELS for c in CLUSTERS])
guide+='''
主实验a、参数b/c仅用Cluster1/2；d/e用Cluster1；f用Cluster2/ClusterSimu。除主动变化的参数外，W=60秒、M=0.15、监控周期10秒。Static的W/M不适用，跨参数表复用原运行，不伪造新的运行配置。

```text
端到端时延 = 请求完成时刻 − 到达时刻（含排队、执行、迁移后的恢复与等待）
排空吞吐 = 完成请求数 /（最后请求完成 − 首次到达）
观察期吞吐 = 3600秒内完成数 / 3600
结束积压 = 3600秒尚未完成的请求数
额外排空 = max(0, 最后请求完成 − 3600)
吞吐改善 =（新值 / 基准 − 1）×100%
时延降低 =（1 − 新值 / 基准）×100%
```

p50/p99单位秒，吞吐单位req/s；QPS显示由req/min除60得到，不反过来用舍入QPS构造请求。原始性能表不含收益百分比。d仅使用原始指标与seed范围，不使用相对收益或归一化展示。

确定性策略每条件一次；五seed单元格为“均值 [最小值, 最大值]”。每个seed先算p50/p99，再汇总指标；范围不是置信区间。主表只做显示舍入，CSV保留数值精度。数值持平分类仅容纳计算舍入误差，不代表统计等效；没有重复随机实验的条件不声称统计显著性。

## 3. 如何从表格选择论文材料

先确定要写的论断，再定位实验和metric。例如“调度改善吞吐”查e-throughput；同时看e-p50/e-p99是否需要交代取舍。不能把不同实验中最佳点拼成同一个配置，也不能相加多项消融的百分比收益。

每个CSV为长表：一行是一个条件和结果列。`table_id`标识metric，`row_id`与Markdown条件ID一致，`series`标识曲线/对照，`mean/min/max`为数值列，`repetitions`标识1或5，`run_ids`关联原始证据。d/e的seed_details.csv进一步记录每个seed原始值。列名和单位在各表后说明。

'''
guide+=table(['实验','主要表条件行数','运行引用数','汇总单元数/metric','取数方法'],[['a',12,24,24,'12条件×E/Static-512'],['b',12,48,48,'12条件×四margin'],['c',12,48,48,'12条件×四window'],['d',24,96,48,'24条件×两策略；混合用五seed汇总'],['e',6,42,18,'6条件×三策略；加权用五seed汇总'],['f',12,24,24,'12条件×E/E-NoOpt'],['完整网格',48,192,192,'48条件行×四margin']])
guide+='''
a–f合计282条运行引用，含跨实验复用，不能称282次唯一simulation。把完整网格加入后，共覆盖342条论文所需唯一运行。旧归档另保留历史策略和探索记录，共370条历史成功运行。

[完整参数网格](附录_完整参数网格/完整网格.md)保存所有192组合；[选参依据](附录_完整参数网格/选参依据.md)说明唯一候选。无需从b/c的切片猜测未展示的组合。

## 4. 解释证据与局限

每章按整体→模型/集群→负载→例外组织。“记录直接显示”是可复算事实；“与记录一致的解释”是由队列、工作分配和转换支持的机制解释；没有控制实验时不写成已证明的独立因果贡献。最大收益、最差条件和metric冲突均保留，不仅选择有利结果。

这里验证的是既定simulation模型下的表现。零传输开销、Comb模块互不干扰等假设仍在，不能据此宣称真实GPU性能已验证。统一调参与展示使用同一目标小时，d/e集群经过主实验收益筛选，不是独立泛化评估。

## 5. 来源与保存范围

新包包含全部当前论文数据、逐seed和全网格；不修改原完整归档。当前正文不汇报SLO、Static-2048或PE-Only性能对照，这些历史数据仍在旧目录。

[数据来源索引](provenance/数据来源索引.csv)提供条件→run ID→SHA-256→原始记录目录；其中原始目录相对本包根目录解释。原始目录由 --archive-root 指定，含请求、执行尝试、实例、flip与压缩事件日志。来源索引的相对路径以本包根目录为基准；移动数据时应保持两目录的相对位置，或重新导出索引。

[数据核对记录](provenance/数据核对记录.md)说明计数、逐请求重算、CSV及Markdown核对。新包的数据表可独立读取，深入核查时再使用原日志。
'''
write('阅读指南与结论总览.md',guide)

HERE.mkdir(exist_ok=True)
(HERE/'csv_specs.json').write_text(json.dumps(csv_specs,ensure_ascii=False,allow_nan=False),encoding='utf-8')
(HERE/'md_checks.json').write_text(json.dumps(md_checks,ensure_ascii=False),encoding='utf-8')
evidence=dict(archive_files_verified=len(checks),archive_bytes=sum(x['bytes'] for x in checks.values()),experiment_run_references=expected,summary_counts={x:len(summaries[x]) for x in 'abcdef'},grid_runs=len(grid),unique_paper_runs=len(all_runs),request_records_recomputed=raw_check_count,inputs=inputs,source_index_rows=len(provenance),csv_files=len(csv_specs),metric_table_checks=len(md_checks),analysis_runtime='Python standard library; no Simulator import',frozen_runtime_sha256=manifest['source']['runtime_sha256'])
(HERE/'build_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in evidence.items() if k!='inputs'},ensure_ascii=False,indent=2))
