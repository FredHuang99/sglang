"""Independent export checks; never imports the simulation runtime."""
from pathlib import Path
import collections,csv,hashlib,json,math,re,statistics,sys
sys.stdout.reconfigure(encoding='utf-8')
HERE=Path(REBUILD_CONTEXT['work']);OUT=Path(REBUILD_CONTEXT['output'])
ROOT=Path(REBUILD_CONTEXT['archive'])
CODE=Path(__file__).resolve().parent
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):return math.isclose(a,b,rel_tol=1e-13,abs_tol=1e-12)
evidence=js(HERE/'build_evidence.json');specs=js(HERE/'csv_specs.json')
assert len(specs)==len(list(OUT.rglob('*.csv')))==34
cell_checks=0
for spec in specs:
    with (OUT/spec['path']).open(encoding='utf-8-sig',newline='') as f:data=list(csv.reader(f))
    assert len(data)==len(spec['matrix']),spec['path']
    for actual,expected in zip(data,spec['matrix']):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):
            if b is None:assert a==''
            elif isinstance(b,bool):assert a==str(b).lower()
            elif isinstance(b,(int,float)):assert math.isfinite(float(a)) and float(a)==b,(spec['path'],a,b)
            else:assert a==b,(spec['path'],a,b)
            cell_checks+=1

metric_keys={'p50':'p50_s','p99':'p99_s','throughput':'throughput_req_s'}
original={}
for exp in 'abcdef':
    for r in js(ROOT/f'paper_tables/{exp}_runs.json'):original[r['run_id']]=r
for r in rows(OUT/'provenance/数据来源索引.csv'):
    rid=r['run_id']
    p=ROOT/r['summary_archive_relative'];assert digest(p)==r['summary_sha256']
    if rid not in original:original[rid]=js(p)
assert len(original)==342
metric_cells=0
for directory in [p for p in OUT.iterdir() if p.is_dir() and p.name!='provenance']:
    idorders=[]
    for metric,sourcekey in metric_keys.items():
        rr=rows(directory/(metric+'.csv'));idorders.append([(r['row_id'],r['series']) for r in rr])
        for r in rr:
            rs=[original[rid] for rid in r['run_ids'].split(';')]
            xs=[t[sourcekey] for t in rs]
            assert len(xs)==int(r['repetitions'])
            for key,fn in [('mean',statistics.fmean),('min',min),('max',max)]:assert close(float(r[key]),fn(xs))
            for t in rs:
                assert t['generator']==r['model'] and t['scenario']==r['cluster'] and t['policy']==r['policy'] and t['trace']==r['trace']
                assert float(r['rate_per_min'])==t['rate_per_min']
                assert close(float(r['qps']),t['rate_per_min']/60)
                for k in ('window_s','margin'):assert (r[k]=='' and t.get(k) is None) or (r[k]!='' and float(r[k])==t.get(k))
            metric_cells+=1
    assert idorders[0]==idorders[1]==idorders[2],directory

mdchecks=js(HERE/'md_checks.json')
for c in mdchecks:
    text=(OUT/c['path']).read_text(encoding='utf-8')
    assert c['text'] in text,c['table_id']
    assert len(c['text'].strip().splitlines())==c['rows']+2

# Check table syntax and all local reader links, independently of generated matrices.
links=0;tables=0;deferred_links=[]
for p in OUT.rglob('*.md'):
    s=p.read_text(encoding='utf-8');assert '\ufffd' not in s and '\\[' not in s
    lastcols=None
    for line in s.splitlines():
        if line.startswith('|'):
            cols=len(re.split(r'(?<!\\)\|',line))-2
            if lastcols is None:tables+=1;lastcols=cols
            assert cols==lastcols,(p,line)
        else:lastcols=None
    for target in re.findall(r'\]\(([^)]+)\)',s):
        if target.startswith(('http:','https:','#')):continue
        destination=(p.parent/target).resolve()
        if destination==(OUT/'provenance/数据核对记录.md').resolve():deferred_links.append(destination)
        else:assert destination.is_file(),(p,target)
        links+=1

for exp,n in [('d',60),('e',30)]:
    folder=next(OUT.glob(exp+'_*'));rr=rows(folder/'seed_details.csv');assert len(rr)==n
    for r in rr:
        t=original[r['run_id']]
        for k in metric_keys.values():assert float(r[k])==t[k]
    seedkey='request_seed' if exp=='d' else 'scheduler_seed'
    groups=collections.defaultdict(set)
    for r in rr:groups[(r['model'],r['rate_per_min'],r['policy'])].add(int(r[seedkey]))
    assert all(s==set(range(5)) for s in groups.values())

idx=rows(OUT/'provenance/数据来源索引.csv')
counts=collections.Counter(r['experiment'] for r in idx)
assert counts==dict(a=24,b=48,c=48,d=96,e=42,f=24,grid=192)
assert len({r['run_id'] for r in idx})==342
for r in idx:
    assert r['policy'] not in ('Static-2048','E-PEOnly')
    assert r['simulation_sha256']==evidence['frozen_runtime_sha256']

# Independently reconcile physical startup group cost against recorded totals.
frows=js(ROOT/'paper_tables/f_runs.json')
g=rows(OUT/'f_启动优化/startup_groups.csv');cost=collections.defaultdict(float)
for r in g:
    assert close(float(r['ready_s'])-float(r['launch_started_s']),float(r['startup_elapsed_s']))
    cost[r['run_id']]+=float(r['unavailable_gpu_s'])
for r in frows:assert close(cost[r['run_id']],r['gpu_unavailable_s']),(r['run_id'],cost[r['run_id']],r['gpu_unavailable_s'])

# Old archive and workspace runtime remain unchanged.
all_checks=js(ROOT/'provenance/checksums.json')
for name,c in all_checks.items():
    p=ROOT/name;assert p.stat().st_size==c['bytes'] and digest(p)==c['sha256'],name
for name,c in evidence['inputs'].items():assert digest(ROOT/name)==c['sha256'],name
manifest=js(ROOT/'archive/campaign-002/manifest.json')
for name,h in manifest['source']['runtime_files'].items():
    p=ROOT/'archive/source_and_environment/src'/name
    assert digest(p)==h,p

report=dict(csv_files=len(specs),csv_cells_exactly_verified=cell_checks,metric_aggregates_recomputed=metric_cells,markdown_metric_tables=len(mdchecks),markdown_total_tables=tables,reader_links_verified=links,source_references=dict(counts),unique_runs=342,request_records_independently_recomputed=evidence['request_records_recomputed'],original_archive_files_unchanged=len(all_checks),runtime_files_unchanged=len(manifest['source']['runtime_files']),startup_cost_reconciliations=len(frows),simulation_runs_added=0)
verification='''# 数据核对记录

本次仅整理已完成结果，未调用Simulator、求解器或实验执行入口。

## 覆盖与独立计算

'''
verification+=f"- a/b/c/d/e/f运行引用为24/48/48/96/42/24；d/e汇总为48/18；完整网格192次。\n- 474条来源引用覆盖342条论文所需唯一运行，复用保持原run ID。\n- 从342条运行的246,600条请求记录独立重算线性插值p50/p99和排空吞吐，均与源数据一致；同时核对请求完成、Long比例、类型变化和观察期/积压计数。\n- {metric_cells}个metric汇总单元从原run ID重新计算mean/min/max；d的60条、e的30条随机运行逐seed核对。\n- {len(mdchecks)}张主要/seed/网格metric表核对条件顺序与内容；34份CSV共{cell_checks}个单元按原数值精确核对，空值和有效seed区分保存。\n- Markdown共{tables}张表的列数一致；{links}条阅读链接可解析。\n- 24条f结果的组启动时长、GPU·s重新汇总，与原记录一致。\n\n"
verification+='''## 精度与解释

CSV为UTF-8 BOM长表，数值保留源IEEE-754双精度的往返表示；主表时延两位、吞吐六位仅用于显示。收益与排序使用未舍入数据。均值来自逐运行metric，未将多seed请求混合后计算分位数。

零收益分类容纳小于等于1e-9个百分点的计算误差，不作为统计等效检验。本文没有显著性或置信区间推断。机制解释沿用已审计的阶段等待、分配与转换记录，没有新增控制实验。

## 来源保护与可追溯性

'''
verification+=f"旧归档{len(all_checks)}个已索引文件、1,045,403,348字节的SHA-256在整理前后均一致；归档{len(manifest['source']['runtime_files'])}个冻结运行时文件摘要一致。已有结果、profile、请求清单和历史报告未改动。\n\n"
verification+='''`数据来源索引.csv`提供实验/条件ID/series/seed到原run ID、请求/部署/profile/源码摘要及原始目录的映射。`input_checksums.json`记录实际读取的源文件摘要；`checksums.json`覆盖本包全部文件（不包含自身）。完整日志仍在指定输入归档的archive目录中。

本次分析器使用独立Python标准库重算和Markdown生成，CSV通过Python标准库写入，再逐单元与精确原始数值核对。未运行仿真回归，因为未修改仿真代码；验证范围是数据整理、统计与出处一致性，不是重新验证真实系统性能。
'''
(OUT/'provenance/数据核对记录.md').write_text(verification,encoding='utf-8',newline='\n')
assert all(p.is_file() for p in deferred_links)
(OUT/'provenance/validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'provenance/input_checksums.json').write_text(json.dumps(evidence['inputs'],ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'provenance/authoring_sources.json').write_text(json.dumps({p.name:digest(p) for p in [CODE/'_build.py',CODE/'export.py',CODE/'_validate.py']},ensure_ascii=False,indent=2),encoding='utf-8')
finalchecks={p.relative_to(OUT).as_posix():dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='checksums.json'}
(OUT/'provenance/checksums.json').write_text(json.dumps(finalchecks,ensure_ascii=False,indent=2),encoding='utf-8')
(HERE/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
