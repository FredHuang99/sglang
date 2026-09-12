"""Complete f/d/e datasets, independent raw-event audit and publication prose."""
import copy
import math
from collections import defaultdict
from pathlib import Path
from pave_sim.evaluation import campaign as c
from pave_sim.evaluation.analysis import measure,aggregate,comparisons,flatten,startup_tables
from pave_sim.records import write_json,write_csv
from .remaining_plan import make_plan,validate_plan
from .remaining_audit import audit_run
from .f_low import enrich_periods,representative_timelines
from .publication import remove_slo,table,label as old_label
from .selection import tool_identity

def label(row):return old_label(row).replace('Clustersimu','ClusterSimu')

def validate_contexts(bindings,name):
    grouped=defaultdict(list)
    for b in bindings:
        ctx=copy.deepcopy(b['context']);e=ctx['effective']
        grouped[e['generator'],e['scenario'],e['rate_per_min']].append(ctx)
    for contexts in grouped.values():
        if name=='d':
            dynamic={(x['effective']['window_s'],x['effective']['margin'],x['effective']['monitor_period_s']) for x in contexts if x['effective']['policy']=='E'}
            if len(dynamic)!=1:raise ValueError('Dynamic trace comparison changed fixed window/margin/period context')
            shared=defaultdict(set)
            for x in contexts:shared[x['effective']['trace'],x['effective']['request_seed']].add(x['requests_sha256'])
            if any(len(v)!=1 for v in shared.values()):raise ValueError('Strategies do not share identical trace requests context')
        normalized=[]
        for ctx in contexts:
            e=ctx['effective'];e.pop('policy');e.pop('startup_mode')
            if name=='e':e.pop('scheduler_seed')
            if name=='d':
                for k in ('trace','request_seed','window_s','margin','monitor_period_s'):e.pop(k)
                ctx.pop('requests_sha256');ctx.pop('trace_statistics',None)
            normalized.append(ctx)
        if any(ctx!=normalized[0] for ctx in normalized[1:]):
            raise ValueError('Fixed comparison context differs for '+name)

def export(root,plan_file,name,out):
    root,out=Path(root),Path(out)
    if out.exists():raise ValueError('Report destination must be new')
    if name not in ('f','d','e'):raise ValueError('Unknown closeout report')
    m,cfg=c.load_campaign(root)
    plan=c.read(Path(plan_file));validate_plan(plan,m,cfg)
    bs=[m['bindings'][s] for s in plan['references'][name]]
    validate_contexts(bs,name)
    if not cfg.synthetic:
        assert len(bs)=={'f':24,'d':96,'e':42}[name]
    rows,audits,monitors,flips,groups,sources,costs,allocations,timelines=[],[],[],[],[],[],[],[],[]
    startup={'startups':[],'groups':[],'direction_periods':[]}
    for index,b in enumerate(bs,1):
        rid=b['run_id'];result=c.load_result(root,m,b);row=measure(result,b)
        attempt=m['runs'][rid]['attempts'][-1];directory=c.inside(root,attempt['path'])
        payload=c.read(c.inside(root,m['requests'][b['request_key']]['path']))
        settings=cfg.settings(root/'inputs/ilp')
        c.RequestPlan.load(payload,settings,b['bound']['rate_per_min'])
        expected={r['id']:r for r in payload['records']}
        from pave_sim.timing import ticks
        assert all((ticks(r['arrival_s']),r['input_tokens'],r['output_tokens'],r['kind'])==
            (expected[r['id']]['arrival_tick'],expected[r['id']]['input_tokens'],expected[r['id']]['output_tokens'],expected[r['id']]['kind']) for r in result['requests'])
        case=c.Case.load(settings,b['bound']['generator'],b['bound']['scenario'])
        for p in result['instances']:
            raw=p['raw']
            if raw['gpu_ids']:
                assert raw['startup']==case.profiles.catalog.startup(b['bound']['generator'],raw['hardware'],len(raw['gpu_ids']),set(raw['stages'])=={'PE'})
        a,g,f,mon=audit_run(result,row,directory,{'requests':len(payload['records'])},root)
        rows.append(row);audits.append(a);groups.extend(g);flips.extend(f);monitors.extend(mon)
        allocations.extend(row['instance_allocations']);timelines.extend(representative_timelines(result))
        current=enrich_periods(startup_tables([result]),[result])
        for key in startup:startup[key].extend(current[key])
        sources.append({'experiment':name,'slot_id':b['bound']['slot_id'],'run_id':rid,
            'identity_sha256':b['identity_sha256'],'requests_sha256':b['context']['requests_sha256'],
            'raw_relative_directory':'archive/campaign-002/'+attempt['path'],'raw_files_sha256':attempt['files']})
        costs.append({'run_id':rid,'wall_s':attempt['wall_s'],'events':a['event_log_rows'],
            'bytes':sum(p.stat().st_size for p in directory.iterdir() if p.is_file()),'status':attempt['status']})
        print(f'AUDIT {name} {index}/{len(bs)} {rid} PASS',flush=True)
    summarized=aggregate(rows)
    pairs=comparisons(name,rows,m['selections']['parameters']['chosen'])
    if name=='f':
        assert len(pairs)==len(rows)//2 and all((p['new_policy'],p['base_policy'])==('E','E-NoOpt') for p in pairs)
    by_id={r['run_id']:r for r in rows}
    for p in pairs:
        for metric in ('p50_s','p99_s','throughput_req_s'):
            p[metric+'_new']=by_id[p['new_run_id']][metric]
            p[metric+'_base']=by_id[p['base_run_id']][metric]
    out.mkdir(parents=True)
    def save(file,data,clean=False):
        data=remove_slo(data) if clean else data
        write_json(out/(file+'.json'),data);write_csv(out/(file+'.csv'),[flatten(r) for r in data])
    save(name+'_full_metrics',rows)
    save(name+'_runs',rows,True);save(name+'_summary',summarized,True)
    if name!='d':save(name+'_comparisons',pairs,True)
    for suffix,data in [('audits',audits),('monitors',monitors),('flips',flips),('group_audit',groups),
                        ('sources',sources),('costs',costs),('allocations',allocations),('timelines',timelines)]:
        save(name+'_'+suffix,data)
    for key,data in startup.items():save(name+'_'+key,data)
    write_json(out/'analysis_version.json',{'experiment':name,'tool_sources':tool_identity(),
        'runtime_sha256':m['source']['runtime_sha256'],'plan_sha256':plan['plan_sha256'],
        'counts':{'runs':len(rows),'summary':len(summarized),'comparisons':len(pairs)},
        'source_ids':[r['run_id'] for r in rows],'audit_event_rows':sum(a['event_log_rows'] for a in audits)})
    (out/'section.md').write_text(render(name,rows,summarized,pairs,startup),encoding='utf-8')
    (out/'report.md').write_text('# 实验'+name+'：完整结果与分析\n\n'+render(name,rows,summarized,pairs,startup)+
        '\n## 验证与执行成本\n\n'+f'{len(rows)}条记录、{sum(a["event_log_rows"] for a in audits):,}条事件通过独立审计。请求清单、进度、阶段完成、时间记账、GPU占用、调度类型、窗口及启动profile均已核对。\n\n'+
        '\n'.join(table(['场景','策略','req/min','wall s','事件行','MiB'],[[label(by_id[x['run_id']]),by_id[x['run_id']]['policy'],by_id[x['run_id']]['rate_per_min'],x['wall_s'],x['events'],x['bytes']/1024**2] for x in costs]))+
        '\n成本含复用记录原始运行时间，不代表本批新增执行时间。完整来源与原始日志见sources及campaign归档。\n',encoding='utf-8')
    return {'experiment':name,'runs':len(rows),'summary':len(summarized),'comparisons':len(pairs),'events':sum(a['event_log_rows'] for a in audits)}

def render(name,rows,summary,pairs,startup):
    lines=[{'f':'### f：Startup Optimization（全优化与全部不优化）',
            'd':'### d：Sensitivity to Trace Construction',
            'e':'### e：Scheduling Ablation'}[name],'',
        '**共同方法。** 两个模型；W=60秒、margin=0.15、监控周期10秒；输入128，输出512/2048，DiT 50步；一小时等间隔到达后完整排空。p50/p99单位为秒，吞吐为req/s。对照固定输入和仿真版本，仅改变本实验声明的因素。','']
    if name=='f':
        lines+=['**研究问题与setup。** 在Cluster2和ClusterSimu的低、中、高三档负载，比较E与E-NoOpt。E优化全部PE及生成模型启动；E-NoOpt全部使用非优化启动。两者均从bin1开始，使用预计完成时间调度与restricted转换。24条结果形成12组配对，衡量启动优化的整体闭环效果。','',
            *table(['场景','req/min','策略','p50 s','p99 s','排空吞吐','积压','额外排空 s'],
                [[label(r),r['rate_per_min'],r['policy'],r['p50_s'],r['p99_s'],r['throughput_req_s'],r['end_backlog'],r['drain_s']] for r in rows]),
            *table(['场景','req/min','吞吐改善 %','p50降低 %','p99降低 %'],
                [[label(p),p['rate_per_min'],p['throughput_req_s_improvement_percent'],p['p50_s_improvement_percent'],p['p99_s_improvement_percent']] for p in pairs])]
        for metric,title in [('throughput_req_s','吞吐'),('p50_s','p50'),('p99_s','p99')]:
            values=[p[metric+'_improvement_percent'] for p in pairs]
            lines += [f'{title}改善范围为{min(values):.2f}%–{max(values):.2f}%；12条件中改善{sum(v>1e-9 for v in values)}项、持平{sum(abs(v)<=1e-9 for v in values)}项、退化{sum(v < -1e-9 for v in values)}项。负改善完整保留。','']
        byid={r['run_id']:r for r in rows}
        lines+=['**机制解释。** 每条件同时检查资源不可用成本和排队；组次数是实际转换次数，不能把ClusterSimu全部192张GPU都计为参与者。','']
        for p in pairs:
            e,n=byid[p['new_run_id']],byid[p['base_run_id']]
            stage=max(e['stages'],key=lambda s:e['stages'][s]['waiting_s'])
            lines += [f"{label(e)}、{e['rate_per_min']:g} req/min：无优化→全优化，flip为{n['flip_triggered']}→{e['flip_triggered']}次，转换组为{n['converted_groups']}→{e['converted_groups']}次，累计不可用成本为{n['gpu_unavailable_s']:.2f}→{e['gpu_unavailable_s']:.2f} GPU·s；主要排队阶段{stage}的平均等待为{n['stages'][stage]['waiting_s']/n['requests']:.2f}→{e['stages'][stage]['waiting_s']/e['requests']:.2f}秒。",'']
        profiles={}
        for s in startup['startups']:
            key=s['generator'],s['scenario'],s['hardware'],s['template']
            profiles[key]=[label(s),s['hardware']+'/'+s['template'],s['non_optimized_profile_s'],s['optimized_profile_s']]
        lines+=table(['场景','实际启动硬件/模板','非优化 s','优化 s'],[v for _,v in sorted(profiles.items())])
        lines+=['完整分期转换、safe/launch/ready时间线与物理启动次数见原始表。Comb按一次物理启动计数；PE子实例并行就绪，实例时间之和不是组等待时间。','',
            '**结论范围。** 本对照测量全部启动优化的整体效果，不能分别归因于PE或DiT优化。启动时延可改变后续调度与flip轨迹；成本减少不保证每个系统指标改善。跨集群还改变硬件和部署，不是单独并行度的因果对照。','']
    elif name=='d':
        lines+=['**研究问题与setup。** 在Cluster1的三档负载比较15/30/60秒多数票和真实比例混合trace，策略为E与Static-512。三个多数票各一次，混合trace五个排列seed；两策略读取同一个seed对应的请求清单。Static采用least_waiting且不flip。','',
            '96条逐运行记录汇总为48条原始指标记录；混合trace先逐运行求分位数，再汇总均值、最小值和最大值。下表展示均值；精确范围和每个seed保存在summary和runs文件。无归一化或相对收益列。','',
            *table(['场景','req/min','trace','策略','seed数','实际Long比例','p50均值 s','p99均值 s','吞吐均值','flip均值'],
                [[label(r),r['rate_per_min'],r['trace'],r['policy'],r['repetitions'],r['long_fraction_mean'],r['p50_s_mean'],r['p99_s_mean'],r['throughput_req_s_mean'],r['flip_triggered_mean']] for r in summary]),
            '**解释。** 多数票三个粒度的Long区间比例接近，但输入变化次数不同，不能据此预先认定性能不敏感。真实比例版本改变了总Long工作量和分钟内交错方式；该对照同时检验输入构造的影响，不能将全部变化归于聚合频率。','']
        bycase=defaultdict(list)
        for r in summary:bycase[r['generator'],r['rate_per_min'],r['policy']].append(r)
        for key,values in bycase.items():
            lookup={r['trace']:r for r in values};mix=lookup['mixture'];base=lookup['majority60']
            majors=[lookup['majority'+str(w)] for w in (15,30,60)]
            lines += [f"{label(base)}、{base['rate_per_min']:g} req/min、{base['policy']}：多数票吞吐均值范围{min(r['throughput_req_s_mean'] for r in majors):.4f}–{max(r['throughput_req_s_mean'] for r in majors):.4f} req/s，p99范围{min(r['p99_s_mean'] for r in majors):.2f}–{max(r['p99_s_mean'] for r in majors):.2f}秒；真实比例吞吐{mix['throughput_req_s_mean']:.4f} req/s、p99 {mix['p99_s_mean']:.2f}秒，实际Long比例为{mix['long_fraction_mean']:.4f}。",'']
        lines+=['**结论范围。** 本实验使用同一小时的四种映射，不构成跨小时泛化验证；Cluster1经过主实验案例筛选。完整阶段排队、积压及实际转换次数用于检查输入改变后的瓶颈。','']
    else:
        lines+=['**研究问题与setup。** 两个模型在Cluster1三档负载和主trace下，比较estimated_completion、least_waiting和capacity_weighted。三者均保留restricted flip与全部启动优化，只有请求调度器及有效调度随机种子变化。E和E-Least确定性各一次，E-Weighted五seed。','',
            *table(['场景','req/min','策略','seed数','p50均值 s','p99均值 s','吞吐均值','吞吐min','吞吐max','flip均值'],
                [[label(r),r['rate_per_min'],r['policy'],r['repetitions'],r['p50_s_mean'],r['p99_s_mean'],r['throughput_req_s_mean'],r['throughput_req_s_min'],r['throughput_req_s_max'],r['flip_triggered_mean']] for r in summary]),
            '**解释与工作分配。** 42条逐运行记录形成18条汇总。实例接收次数包含迁移重试，不能当作完成请求数；completed和实际token/step工作量分别保存。下表按硬件汇总实际PE/DiT工作占比，各seed分别记录，不把实例个数当作容量占比。','']
        for model in sorted({r['generator'] for r in rows}):
            for rate in sorted({r['rate_per_min'] for r in rows if r['generator']==model}):
                rs=[r for r in summary if r['generator']==model and r['rate_per_min']==rate]
                top=max(r['throughput_req_s_mean'] for r in rs)
                winners=[r for r in rs if abs(r['throughput_req_s_mean']-top)<=1e-12]
                best=winners[0]
                lines += [f"{label(best)}、{rate:g} req/min：按各策略逐运行吞吐的均值，最高者为{'、'.join(r['policy'] for r in winners)}（{top:.4f} req/s）。完整数据保留并列及随机范围，不按某个最有利seed选结果。",'']
        shares=[]
        for r in rows:
            grouped=defaultdict(float)
            for a in r['instance_allocations']:
                if a['stage'] in ('PE','DiT'):grouped[a['stage'],a['hardware']]+=a['executed_work']
            for (stage,hw),work in grouped.items():
                total=sum(w for (s,_),w in grouped.items() if s==stage)
                shares.append([label(r),r['rate_per_min'],r['policy'],r['scheduler_seed'],stage,hw,work/total if total else 0])
        lines+=table(['场景','req/min','策略','调度seed','模块','硬件','实际工作占比'],shares)
        lines+=['**结论范围。** 这是调度替换对完整PAVE的影响。队列和转换轨迹可以随调度改变，因此不解释为固定部署上的孤立调度微基准。五seed提供范围，不代表已完成统计显著性检验。','']
    lines+=['全部结果基于已批准的CPU离散事件模型与同一目标小时，不等同实测GPU性能。每行精确值和来源run ID见对应原始表；队列、排空和转换成本为解释证据。','']
    return '\n'.join(lines)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--experiment',choices=['f','d','e'],required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();print(export(a.campaign,a.plan,a.experiment,a.output_dir))
