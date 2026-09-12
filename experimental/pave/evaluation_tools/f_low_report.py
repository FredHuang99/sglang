"""Readable partial-f tables and data-grounded narrative; no execution entry point."""
from collections import defaultdict
from pathlib import Path
from pave_sim.evaluation import campaign as c
from .publication import label, table

def render(directory):
    directory=Path(directory)
    rows=c.read(directory/'f_low_runs.json')
    pairs=c.read(directory/'f_low_comparisons.json')
    starts=c.read(directory/'f_low_startups.json')
    periods=c.read(directory/'f_low_direction_periods.json')
    groups=c.read(directory/'f_low_groups.json')
    flips=c.read(directory/'f_low_flips.json')
    audits={r['run_id']:r for r in c.read(directory/'f_low_audits.json')}
    costs=c.read(directory/'f_low_costs.json')
    timelines=c.read(directory/'f_low_timelines.json')
    paired_behavior=c.read(directory/'f_low_paired_behavior.json')
    order={'E-NoOpt':0,'E-PEOnly':1,'E':2}
    rows=sorted(rows,key=lambda r:(r['generator'],r['scenario'],order[r['policy']]))
    lines=['### f：Startup Optimization（最低负载，已完成12/36条）','',
        '**研究问题与setup。** 本批在两个Wan模型的Cluster2和ClusterSimu上，分别比较E-NoOpt、E-PEOnly和E。四个条件的负载分别为Wan2.2的5.5/44 req/min和Wan2.1的10/66 req/min。其余条件相同：主小时60秒多数票、W=60秒、margin=0.15、10秒监控、预计完成时间调度、restricted正反转换、初始bin1已就绪。到达持续一小时，随后完整排空。每条件确定性运行一次；本批新增10次，复用两条Cluster2 E结果。','',
        '三种策略分别使用全部非优化启动、仅PE优化启动、全部优化启动。该对照保持profile以外的执行模型不变，但允许后续调度、队列和转换轨迹随启动时延变化。两项增量百分比不能相加，三组条件也不足以估计完整两因素交互效应。','',
        '**端到端结果。** 吞吐单位为req/s，时延单位为秒；吞吐采用全部请求完成数除以从首次到达到最后请求完成的时间。','',
        *table(['场景','req/min','策略','请求数','排空吞吐','p50 s','p99 s'],
            [[label(r),r['rate_per_min'],r['policy'],r['requests'],r['throughput_req_s'],r['p50_s'],r['p99_s']] for r in rows]),
        '下表的吞吐改善为(New/Base−1)×100%，时延降低为(1−New/Base)×100%；负值表示退化。精确原始值、绝对变化和双方run ID保存于比较表。','',
        *table(['场景','New / Base','吞吐改善 %','p50降低 %','p99降低 %'],
            [[label(p),p['new_policy']+' / '+p['base_policy'],p['throughput_req_s_improvement_percent'],p['p50_s_improvement_percent'],p['p99_s_improvement_percent']] for p in pairs]),
        '**观察期与排空。** 观察期完成包含3600秒边界；额外排空时间仅以最后请求完成时刻计算，控制事件收尾时刻单独归档。','',
        *table(['场景','策略','观察期吞吐','结束积压','额外排空 s','PE平均等待 s','DiT平均等待 s'],
            [[label(r),r['policy'],r['observed_throughput_req_s'],r['end_backlog'],r['drain_s'],r['stages']['PE']['waiting_s']/r['requests'],r['stages']['DiT']['waiting_s']/r['requests']] for r in rows])]
    by_case=defaultdict(dict)
    for r in rows:by_case[r['generator'],r['scenario']][r['policy']]=r
    for case,policies in by_case.items():
        lines += [f"**{label(policies['E'])}。**",'']
        values=[]
        for policy in ('E-NoOpt','E-PEOnly','E'):
            r=policies[policy];rid=r['run_id']
            ds=r['directions'];g=[x for x in groups if x['run_id']==rid]
            dominant=max(r['stages'],key=lambda s:r['stages'][s]['waiting_s'])
            values.append([policy,ds['short_to_long']['triggered'],ds['long_to_short']['triggered'],
                len(g),sum(x['safe_wait_s'] for x in g),sum(x['startup_elapsed_s'] for x in g),r['gpu_unavailable_s'],dominant])
        lines+=table(['策略','正向flip','反向flip','转换组次数','安全等待组·s','启动等待组·s','不可用GPU·s','主要排队阶段'],values)
        no,pe,e=(policies[p] for p in ('E-NoOpt','E-PEOnly','E'))
        delta=lambda a,b,k:(a[k]/b[k]-1)*100
        lines += [f"仅开启PE优化后，排空吞吐相对全部不优化变化{delta(pe,no,'throughput_req_s'):+.2f}%；进一步开启生成模型优化后变化{delta(e,pe,'throughput_req_s'):+.2f}%。全优化相对不优化的p99从{no['p99_s']:.2f}变为{e['p99_s']:.2f}秒，积压从{no['end_backlog']}变为{e['end_backlog']}。这些是本条件的闭环结果。启动profile减少不能直接替代系统指标；应同时查看转换次数、安全等待、阶段排队和后续恢复时机。",'']
        behavior=next(x for x in paired_behavior if x['new_run_id']==e['run_id'] and x['base_run_id']==pe['run_id'])
        major=max(e['stages'],key=lambda s:e['stages'][s]['waiting_s'])
        lines += [f"进一步开启生成模型优化后，累计不可用GPU·秒由{pe['gpu_unavailable_s']:.2f}变为{e['gpu_unavailable_s']:.2f}；主要排队阶段{major}的平均等待由{pe['stages'][major]['waiting_s']/pe['requests']:.2f}变为{e['stages'][major]['waiting_s']/e['requests']:.2f}秒。逐请求比较中，{behavior['requests_earlier']}个更早完成、{behavior['requests_same']}个完成时刻相同、{behavior['requests_later']}个更晚完成。两者的flip方向及检测时刻序列{'相同' if behavior['same_flip_detection_schedule'] else '不同'}。因此{'该条件可以直接观察到profile成本下降与所报告系统指标的解耦' if e['throughput_req_s']==pe['throughput_req_s'] else '系统表现同时包含启动成本改变和闭环后续状态的影响'}；没有固定转换轨迹的额外对照，不能将两者进一步拆成独立因果贡献。",'']
    lines+=['**启动机制。** 以下按实际启动过的硬件和物理模板列出profile与次数。每个Comb计一次启动；同组PE子实例可以并行就绪。实例启动时间之和、组启动等待时间之和以及GPU·秒各有不同含义。','']
    by_profile=defaultdict(list)
    for s in starts:by_profile[s['generator'],s['scenario'],s['policy'],s['hardware'],s['template']].append(s)
    lines+=table(['场景','策略','硬件/模板','次数','非优化 s','优化 s','实际启动 s范围'],
        [[label(v[0]),key[2],key[3]+'/'+key[4],len(v),v[0]['non_optimized_profile_s'],v[0]['optimized_profile_s'],
          f"{min(x['actual_startup_s'] for x in v):.6f}–{max(x['actual_startup_s'] for x in v):.6f}"] for key,v in sorted(by_profile.items())])
    lines+=['**转换发生在什么时候。** 检测、完成及转换组次数按各自事件时刻分期；成本按区间与观察期/排空期的交集分摊。处于边界上的事件计入观察期。因此同一行中的检测数与完成数可能不同。','',
        *table(['场景','策略','时段','方向','检测/完成','组次数','启动GPU·s','不可用GPU·s'],
            [[label(r),r['policy'],r['period'],r['direction'],f"{r['detected']}/{r['completed']}",r['converted_groups'],r['startup_gpu_s'],r['unavailable_gpu_s']] for r in periods])]
    lines += ['**代表性时间线。** 每个运行固定提取首次正向、首次反向和最长不可用GPU组，均保留全部目标子实例及safe/launch/ready细节。下表仅展示检测和完成时间，完整事件见`f_low_timelines.json`。最长组按detected到最后子实例ready计，不用于代替逐子实例GPU·秒。','',
        *table(['场景/策略','案例','发生','检测 s','完成 s','GPU组'],[
            [next(label(r)+'/'+r['policy'] for r in rows if r['run_id']==t['run_id']),t['example'],t['occurred'],
             t.get('detected_s','未发生'),t.get('completed_s','未发生'),
             '; '.join(str(g['node'])+':'+','.join(map(str,g['gpu_ids'])) for g in t.get('groups',[]))] for t in timelines]),
        '**证据范围。** 当前仅覆盖四个最低负载条件，不能提前推断中高负载结果。Cluster2与ClusterSimu同时改变硬件、部署和并行规模，跨集群差异不是单独SP并行度的因果效应。输入只有一个偏Long小时；结果来自约定的CPU离散事件模型，不是实测GPU性能。','']
    return '\n'.join(lines)

def write_report(directory):
    directory=Path(directory)
    content=render(directory)
    starts=c.read(directory/'f_low_startups.json')
    profile_rows={}
    for s in starts:
        key=(s['generator'],s['scenario'],s['hardware'],s['template'])
        profile_rows[key]=[label(s),s['hardware']+'/'+s['template'],s['non_optimized_profile_s'],
            s['optimized_profile_s'],s['non_optimized_profile_s']-s['optimized_profile_s']]
    paper=content.split('**启动机制。**')[0]+'\n'.join([
        '**机制证据与结论范围。** 实际使用的物理模板及两套启动profile如下；逐策略次数、安全等待、GPU·秒和观察期/排空期分解见本批专门报告。','',
        *table(['场景','硬件/模板','非优化 s','优化 s','单实例减少 s'],[v for _,v in sorted(profile_rows.items())]),
        '四个最低负载条件显示，profile层面的启动缩短与端到端收益并不等价。较小集群的部分指标持平；大集群需要同时解释排队、反馈和转换轨迹。当前结果不能证明启动优化在所有负载下都会提高吞吐，也不能用跨集群差异单独推断SP并行度效应。', '',
        '完整的分期转换表、物理启动表、36条代表性时间线和执行成本，见同目录[f最低负载分析.md](f最低负载分析.md)及`paper_tables/f_low_*`。f中高负载尚未执行。',''])
    (directory/'startup_section.md').write_text(paper,encoding='utf-8')
    costs=c.read(directory/'f_low_costs.json')
    rows=c.read(directory/'f_low_runs.json');by_id={r['run_id']:r for r in rows}
    audits=c.read(directory/'f_low_audits.json')
    total_new=sum(r['wall_s'] for r in costs if not r['reused'])
    report=['# f最低负载批次：执行、审计与分析','',
        f"10次新增仿真全部完成，复用2条已有E结果；新增请求21,660个，12条对照记录共22,590个请求。新增仿真墙钟时间合计{total_new:.2f}秒。",'',
        f"独立审计通过{len(audits)}条运行、{sum(r['event_log_rows'] for r in audits):,}条事件：核对请求/阶段完成、token/step守恒、等待与执行时间、原始流中的请求归属、GPU占用、就绪状态、完成样本窗口、阈值、safe边界及启动profile。",'',
        content,'## 执行成本','',*table(['场景','策略','复用','墙钟 s','事件条数','产物 MiB'],[
            [label(by_id[r['run_id']]),by_id[r['run_id']]['policy'],r['reused'],r['wall_s'],r['events'],r['bytes']/1024**2] for r in costs]),
        '成本表中复用记录的墙钟时间来自其原始运行，不计入本批新增成本。完整raw_metrics保留历史指标；论文正文与paper_tables只展示批准的性能指标。','',
        '本批完成后停止。f中高负载尚需新增20次，d尚需84次，e尚需36次；未自动执行后续批次。','']
    (directory/'report.md').write_text('\n'.join(report),encoding='utf-8')
    return {'report':str(directory/'report.md'),'new_wall_s':total_new}

if __name__=='__main__':
    import sys
    print(write_report(sys.argv[1]))
