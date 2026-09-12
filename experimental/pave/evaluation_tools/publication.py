"""Rebuild publication tables and prose without changing or executing simulations."""
import math
from collections import defaultdict
from pathlib import Path

from pave_sim.evaluation import campaign as c
from pave_sim.evaluation.analysis import aggregate, comparisons, flatten, measure, startup_tables
from pave_sim.records import write_csv, write_json
from .selection import make_plan, paper_references, tool_identity, write_index


def remove_slo(value):
    if isinstance(value, dict):
        return {k: remove_slo(v) for k, v in value.items()
                if not any(part == 'slo' or part.startswith(('slo5', 'slo10', 'slo_'))
                           for part in k.lower().split('.'))}
    if isinstance(value, list):
        return [remove_slo(v) for v in value]
    return value


def fmt(v):
    return f'{v:.4f}' if isinstance(v, float) else str(v)


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|'] + [
        '| ' + ' | '.join(fmt(x) for x in row) + ' |' for row in rows] + ['']


def label(row):
    return ('Wan2.2' if row['generator'].startswith('wan2.2') else 'Wan2.1') + '/' + row['scenario'].replace('cluster', 'Cluster')


def render_analysis(datasets, manifest, config):
    p = manifest['selections']['parameters']['chosen']
    cluster = manifest['selections']['cluster']['chosen'].replace('cluster', 'Cluster')
    lines = ['# PAVE Evaluation：实验结果与分析', '',
        '## 1. Experimental Methodology', '',
        '本节评估PAVE在输出长度随时间变化的负载下，相对固定资源预配置的端到端表现，并分析监控参数、启动和请求调度机制。本入口重建a/b/c；完整d/e/f通过remaining_report或可读导出入口读取，不在本报告重复汇总。', '',
        '### 1.1 模型、硬件与工作负载', '',
        '主实验包含Wan2.2 TI2V 5B和Wan2.1 T2V 1.3B，各在Cluster1和Cluster2上运行。两集群均为一个8卡A100 80GB节点与一个8卡H100 80GB节点，区别在于允许的部署模板。Cluster1允许PE、独立DiT/VAE及Comb的多种并行度；Cluster2限制为PE TP1/TP2和Comb SP2/SP4。', '',
        *table(['模型', 'Cluster1 req/min', 'Cluster2 req/min'],
               [[g, '、'.join(map(str, r['cluster1'])), '、'.join(map(str, r['cluster2']))] for g, r in config.rates.items()]),
        '请求以等间隔到达，输入长度固定128，输出长度为512或2048，DiT执行50步。主trace取Azure数据2024-10-18 06:00–07:00 UTC，以全文件generated tokens中位数98划分Short/Long，再按每分钟多数票决定该分钟发送的请求类型。512和2048是simulation采用的代表性输出长度，不是原始trace的中位数和最大值。多数票版本有52个Long分钟，占86.67%，输入类型变化16次；该变化次数不等于实际flip次数。', '',
        '在[0,3600秒)发送请求，之后继续排空。排空期间监控与转换仍可发生；最后请求完成后已启动转换正常收尾。主结果采用完整请求集合，不截掉排队较长的请求。', '',
        '### 1.2 对照与指标', '',
        'Static-512按代表性输出长度512预配置GPU资源并始终保留该部署，使用least_waiting调度：只比较等待请求数，运行任务不进入主评分，并列按稳定实例顺序选择。PAVE E从相同部署开始，使用预计完成时间调度、restricted正反转换和全部启动优化。两者读取相同请求清单。该对照衡量完整PAVE的效果，调度与部署转换同时变化。', '',
        '报告p50、p99端到端时延及排空吞吐。端到端时延包含排队、四阶段执行、恢复prefill和转换造成的等待；分位数采用线性插值。排空吞吐定义如下：', '',
        '```text', '排空吞吐 = 完成请求数 /（最后请求完成时间 − 首次到达时间）',
        '相对吞吐改善 =（PAVE吞吐 / 基准吞吐 − 1）×100%',
        '时延降低比例 =（1 − PAVE时延 / 基准时延）×100%', '```', '',
        '观察期吞吐、3600秒积压和额外排空时间用于区分一小时内的处理能力与尾部完成效应。吞吐单位为req/s；负载计算使用req/min原值。', '',
        '### 1.3 统一参数与案例选择', '',
        f"在两模型×两集群×三负载的12条件上，对4个window和4个margin完成192次网格运行。按条件最大吞吐归一化后取几何平均，保留最高分99.5%范围，再按归一化p99及较小参数消除并列。本次仅window={p['window_s']}秒、margin={p['margin']}进入候选范围。监控周期固定10秒，与window独立。所有后续实验使用同一组参数。", '',
        f'Trace和调度消融采用{cluster}：该集群在两模型×三负载上，相对固定基准的平均吞吐收益较高。这是按主实验结果筛选的案例；参数调优和主结果也使用同一个目标小时，因此本节不将其称为独立泛化验证。', '',
        '## 2. End-to-End Performance', '']
    if 'a' in datasets:
        grouped = defaultdict(dict)
        for r in datasets['a']:
            grouped[r['generator'], r['scenario'], r['rate_per_min']][r['policy']] = r
        trows, lrows, extra, improvements = [], [], [], []
        for _, v in sorted(grouped.items()):
            e, b = v['E'], v['Static-512']
            gain = (e['throughput_req_s']/b['throughput_req_s']-1)*100
            p50 = (1-e['p50_s']/b['p50_s'])*100
            p99 = (1-e['p99_s']/b['p99_s'])*100
            improvements.append((gain, p50, p99))
            trows.append([label(e), e['rate_per_min'], e['throughput_req_s'], b['throughput_req_s'], gain])
            lrows.append([label(e), e['rate_per_min'], e['p50_s'], b['p50_s'], e['p99_s'], b['p99_s']])
            extra.append([label(e), e['rate_per_min'], e['observed_throughput_req_s'], b['observed_throughput_req_s'],
                          e['end_backlog'], b['end_backlog'], e['drain_s'], b['drain_s']])
        gain, med, tail = (list(x) for x in zip(*improvements))
        lines += [f'在全部12个条件中，PAVE均提高排空吞吐并降低p50和p99。吞吐提升范围为{min(gain):.2f}%–{max(gain):.2f}%，条件等权算术平均{sum(gain)/len(gain):.2f}%；p50降低{min(med):.2f}%–{max(med):.2f}%，p99降低{min(tail):.2f}%–{max(tail):.2f}%。这些平均值对每个模型、集群和负载条件赋予相同权重，不按请求数量加权。', '',
                  *table(['场景', 'req/min', 'PAVE吞吐', '固定基准吞吐', '提升 %'], trows),
                  *table(['场景', 'req/min', 'PAVE p50 秒', '基准 p50 秒', 'PAVE p99 秒', '基准 p99 秒'], lrows),
                  '### 2.1 负载、积压与排空', '',
                  *table(['场景', 'req/min', 'PAVE观察期吞吐', '基准观察期吞吐', 'PAVE积压', '基准积压', 'PAVE额外排空秒', '基准额外排空秒'], extra)]
        for model, scenario in sorted({(r['generator'], r['scenario']) for r in datasets['a']}):
            es = sorted([r for r in datasets['a'] if r['policy']=='E' and (r['generator'], r['scenario'])==(model, scenario)], key=lambda r: r['rate_per_min'])
            lo, hi = es[0], es[-1]
            wait = max(hi['stages'], key=lambda s: hi['stages'][s]['waiting_s'])
            share = hi['stages'][wait]['waiting_s']/hi['total_wait_s']*100 if hi['total_wait_s'] else 0
            lines += [f"**{label(hi)}。** 从{lo['rate_per_min']:g}增至{hi['rate_per_min']:g} req/min，PAVE排空吞吐从{lo['throughput_req_s']:.4f}变为{hi['throughput_req_s']:.4f} req/s，p99从{lo['p99_s']:.2f}增至{hi['p99_s']:.2f}秒，结束积压从{lo['end_backlog']}增至{hi['end_backlog']}。最高负载下，{wait}贡献{share:.2f}%的阶段等待时间，说明端到端表现仍显著受排队影响。", '']
        lines += ['Wan2.1／Cluster1最高档的排空吞吐下降与观察期吞吐接近饱和同时出现，不能解释为一小时内完全失去处理能力。已有原始事件复核记录了该条件排空期继续触发转换；完成PE样本驱动的反馈会滞后于到达类型。该解释与队列和转换时间线一致，但本批没有单独固定转换轨迹的对照，不能量化每一项因素的独立因果贡献。', '']
    for name, axis, title in [('b', 'margin', '3. Sensitivity to Margin'), ('c', 'window_s', '4. Sensitivity to Window')]:
        lines += [f'## {title}', '']
        if name not in datasets:
            lines += ['本批未导出该实验，见后续数据版本。', '']
            continue
        group = defaultdict(dict)
        for r in datasets[name]:
            group[r['generator'], r['scenario'], r['rate_per_min']][r[axis]] = r
        levels = sorted({r[axis] for r in datasets[name]})
        lines += [f"固定{'window=60秒' if name=='b' else 'margin=0.15'}，在12个条件上改变{axis}。下表给出吞吐，完整p50/p99见同目录paper_tables中的逐运行与汇总表。", '',
                  *table(['场景', 'req/min']+[str(v) for v in levels],
                         [[label(next(iter(v.values()))), key[2]]+[v[x]['throughput_req_s'] for x in levels] for key,v in sorted(group.items())])]
        diagnostics = []
        for level in levels:
            rs = [g[level] for g in group.values()]
            scores = [g[level]['throughput_req_s']/max(r['throughput_req_s'] for r in g.values()) for g in group.values()]
            diagnostics.append([level, math.exp(math.fsum(map(math.log,scores))/len(scores)),
                sum(math.isclose(g[level]['throughput_req_s'], max(r['throughput_req_s'] for r in g.values()), rel_tol=1e-12) for g in group.values()),
                sum(r['flip_triggered'] for r in rs)/len(rs)])
        lines += table([axis, '切片归一化吞吐GM', '最优或并列条件数', '平均flip次数'], diagnostics)
        lines += [('margin从0增至0.15时，平均flip由17.50降至13.17次；总体吞吐评分提高，但并非逐条件单调改善。例如Wan2.2／Cluster1最高负载在0.10下吞吐更高。因此结果支持统一折中参数，不支持margin越大越好。'
                  if name=='b' else 'window从10秒增至60秒时，平均flip由22.33降至13.17次。较长窗口的主要吞吐优势来自Wan2.1／Cluster1的中高负载，多个其他条件在较短窗口下更好。窗口长度同时影响平滑、反馈时机与调度能力估计，不能仅以flip减少解释全部收益。'), '']
    lines += ['## 5. Startup、Trace与Scheduling Ablations', '']
    for name, description in [('f', '启动优化：Cluster2/ClusterSimu，E与全部不优化'),
                              ('d', f'Trace：{cluster}，15/30/60秒多数票与真实比例，PAVE及固定基准'),
                              ('e', f'调度：{cluster}，estimated_completion、least_waiting与capacity_weighted')]:
        lines += [f'### {name}：{description}', '']
        if name in datasets:
            groups = remove_slo(aggregate(datasets[name]))
            lines += table(['模型/集群', 'req/min', '策略', 'trace', '重复数', '吞吐均值', 'p50均值', 'p99均值'],
                           [[label(r), r['rate_per_min'], r['policy'], r['trace'], r['repetitions'], r['throughput_req_s_mean'], r['p50_s_mean'], r['p99_s_mean']] for r in groups])
            lines += ['本节表格来自已完成批次；机制解释须结合对应批次的转换与工作分配记录补充，不能仅凭表格判定原因。', '']
        else:
            lines += ['本入口不导出该实验。使用 evaluation_tools.remaining_report 与最终执行清单读取已有完整结果。', '']
    lines += ['## 6. Discussion and Limitations', '',
        '本节收益反映完整PAVE相对代表性长度固定预配置的结果，包含请求调度、部署和转换共同作用。主trace经过多数票和两种输出长度映射，偏Long且只有一个小时；不能将结果直接推广到任意真实输出分布。', '',
        '高负载中的端到端时延由队列显著影响。排空吞吐使用最后请求完成时间，观察期吞吐只统计截至3600秒的完成请求，两者回答不同问题；不能将前者的提升直接称为无限持续稳态容量增加。', '',
        '结果来自已批准的CPU离散事件模型，采用零迁移传输成本、Comb模块互不干扰等假设，不是GPU实测性能。启动消融比较全优化与全部不优化；闭环转换轨迹可以变化，不能拆分为已识别的PE和DiT独立贡献。', '',
        '全部逐条件值及来源run ID见paper_tables；输入、运行日志和版本身份见归档。历史选择与报告口径保存在provenance中。', '']
    return '\n'.join(lines)


def export_paper(root, destination, experiments=('a','b','c')):
    root, destination = Path(root), Path(destination)
    if any(name not in ('a', 'b', 'c') for name in experiments):
        raise ValueError('Use evaluation_tools.remaining_report with the revised plan for d/e/f')
    if destination.exists():
        raise ValueError('Paper output directory must be new')
    manifest, config = c.load_campaign(root)
    if manifest['selections']['cluster'].get('baseline') != 'Static-512':
        raise ValueError('Publication cluster revision is required')
    refs = paper_references(manifest, config)
    if not experiments or any(n not in refs for n in experiments) or len(set(experiments)) != len(experiments):
        raise ValueError('Invalid experiment list')
    cache, datasets, source_rows, full_results = {}, {}, [], {}
    for name in experiments:
        datasets[name] = []
        for sid in refs[name]:
            binding = manifest['bindings'][sid]
            rid = binding['run_id']
            if rid not in cache:
                result = c.load_result(root, manifest, binding)
                cache[rid] = measure(result, binding)
                if name == 'f':
                    full_results[rid] = result
            datasets[name].append(cache[rid])
            attempt = manifest['runs'][rid]['attempts'][-1]
            source_rows.append({'experiment': name, 'slot_id': sid, 'run_id': rid,
                'identity_sha256': binding['identity_sha256'], 'requests_sha256': binding['context']['requests_sha256'],
                'raw_relative_directory': 'archive/campaign-002/' + attempt['path'],
                'raw_files_sha256': attempt['files']})
    # Validate comparisons/aggregation for every dataset before writing any table.
    derived = {n: (aggregate(rs), comparisons(n, rs, manifest['selections']['parameters']['chosen'])) for n,rs in datasets.items()}
    destination.mkdir(parents=True)
    tables = destination / 'paper_tables'
    tables.mkdir()
    for name, rows in datasets.items():
        clean = remove_slo(rows)
        write_json(tables / f'{name}_runs.json', clean)
        write_csv(tables / f'{name}_runs.csv', [flatten(r) for r in clean])
        for suffix, data in zip(('summary','comparisons'), derived[name]):
            if name == 'd' and suffix == 'comparisons':
                continue
            write_json(tables / f'{name}_{suffix}.json', remove_slo(data))
            write_csv(tables / f'{name}_{suffix}.csv', remove_slo(data))
    if 'f' in datasets:
        # Cached main E records may have been loaded while processing a first.
        for sid in refs['f']:
            b = manifest['bindings'][sid]
            if b['run_id'] not in full_results:
                full_results[b['run_id']] = c.load_result(root, manifest, b)
        for kind, records in startup_tables(list(full_results.values())).items():
            write_json(tables / f'f_{kind}.json', records)
            write_csv(tables / f'f_{kind}.csv', records)
    provenance = destination / 'provenance'
    provenance.mkdir()
    write_json(provenance / 'report_sources.json', source_rows)
    write_csv(provenance / 'report_sources.csv', source_rows)
    write_json(provenance / 'execution_plan.json', make_plan(manifest, config))
    write_index(provenance / 'run_index.csv', make_plan(manifest, config), manifest)
    write_json(provenance / 'analysis_version.json', {'tool_sources': tool_identity(),
               'frozen_analysis_sha256': c.fingerprint_sources()['analysis_sha256'],
               'runtime_sha256': manifest['source']['runtime_sha256'], 'experiments': list(experiments)})
    (destination / 'Evaluation分析.md').write_text(render_analysis(datasets, manifest, config), encoding='utf-8')
    return {'rows': {k:len(v) for k,v in datasets.items()}, 'source_runs':len(cache), 'output':str(destination)}
