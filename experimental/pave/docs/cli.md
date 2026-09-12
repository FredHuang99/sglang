# PAVE CLI 使用说明

## 1. 环境和路径

以下 PowerShell 命令从 `experimental/pave` 执行。Linux 将 `& $py` 换成环境中的 `python`，并修改路径。所有输出目录使用新位置，历史归档不继续追加实验。

```powershell
$py = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$archive = 'C:/Users/woshi/Desktop/PAVE_EVALUATION_RESULTS'
$campaign = "$archive/archive/campaign-002"
$plan = "$archive/provenance/execution_plan.json"
$workRoot = 'C:/Users/woshi/Desktop/PAVE_NEW_WORK'
```

已有CPU环境：Python3.12.10、NumPy2.5.2、SciPy1.18.1。新环境使用 `python -m venv .venv`，再用该Python执行 `-m pip install -e .`。`pave-ilp`、`pave-sim`为安装入口；根目录的 `evaluation_tools`、`tools` 用 `python -m` 从本目录调用。不依赖Node或共享node_modules。

## 2. ILP搜索与校验

`plan`调用CPU MILP求解器；`validate`只读取和检查输出。

```powershell
& $py -B -m pave_ilp plan --generator wan2.1-t2v-1.3b --scenario cluster1 --output-dir "$workRoot/ilp"
& $py -B -m pave_ilp validate --output-dir 'C:/Users/woshi/Desktop/PAVE_ILP'
```

模型可选 `wan2.2-ti2v-5b`、`wan2.1-t2v-1.3b`；场景可选 `cluster1`、`cluster2`、`clustersimu`，可重复指定。未指定模型/场景时覆盖全部。默认输入128、源输出512、目标输出2048、KV pool4096、求解时限120秒。覆盖参数：`--input-tokens`、`--source-output-tokens`、`--target-output-tokens`、`--kv-cache-tokens`、`--mip-time-limit-s`、`--profile-data`。`--overwrite`仅用于明确允许覆盖的新工作目录。

包内profile在 `src/pave_ilp/data/profiles.json`。输出保留物理实例/GPU、逐卡显存、阶段容量，以及full/restricted两种flip。Simulation使用restricted实际目标。

## 3. 传统simulation：run / sweep / report

此入口保留A–E模式。`examples/simulation.json`的4/6/8 req/min等是传统样例，不是论文负载。

```powershell
& $py -B -m pave_sim run --config examples/simulation.json --generator wan2.1-t2v-1.3b --scenario cluster1 --strategy E --rate-per-min 12 --window-s 60 --margin 0.15 --monitor-period-s 10 --output-dir "$workRoot/single"
& $py -B -m pave_sim sweep --config examples/simulation.json --output-dir "$workRoot/legacy-sweep"
& $py -B -m pave_sim report --input-dir "$workRoot/single" --output-dir "$workRoot/single-report"
```

`run/sweep`执行simulation；`report`只读已有 `simulation_collection`。单次run必须指定一个模型/集群。`--scheduler`可选 `least_waiting`、`estimated_completion`、`capacity_weighted`。`--trace`指定dominant intervals CSV；默认 `src/pave_sim/data/dominant_intervals.csv`不是论文新小时的请求清单。

发送结束后排空。PE按最近输入key、再最近输出key查表；首次与恢复都用 `TTFT + (实际剩余输出−1)×TPOT`。DiT继续剩余steps，Comb双模块独立执行。参数详见各入口 `--help`。

## 4. Evaluation：新实验编排

`examples/evaluation.json`包含六组负载、UTC主小时、四window、四margin、10秒监控周期。当前通用默认定义有408个逻辑槽位，保留历史策略，**不是498次，也不等于最终342次论文子集**。更换原始trace时显式调整摘要、p50和统计期望，不跳过输入检查。

```powershell
# 先将样例复制到自己的工作目录，编辑输入路径和设置。
& $py -B -m pave_sim evaluation prepare --config "$workRoot/evaluation.json" --output-dir "$workRoot/campaign-new"
& $py -B -m pave_sim evaluation run --campaign "$workRoot/campaign-new" --phase 2.1
& $py -B -m pave_sim evaluation select --campaign "$workRoot/campaign-new" --target parameters
& $py -B -m pave_sim evaluation select --campaign "$workRoot/campaign-new" --target cluster
& $py -B -m pave_sim evaluation report --campaign "$workRoot/campaign-new" --experiment a --output-dir "$workRoot/report-new"
```

这些命令说明各入口，不表示只跑2.1便可选参。`prepare`仅冻结输入；`run`前台逐次执行、首错停止，`--retry-failed`显式重试。支持phase 2.1/2.2/2.3/2.4/2.5/2.6/2.6-low/2.6-rest。`select`要求依赖结果完整；传统选择规则保留在冻结代码，论文无SLO复核在归档凭据中。

论文固定W=60秒、margin=0.15：a/b/c为Cluster1/2，d/e为Cluster1，f为Cluster2/Simu。论文所需342次；campaign-002另存12条Static-2048与4条PE-Only，总358次成功；campaign-001探索12次单独保留；原408槽位取消50条。Static-512为least_waiting、无flip，f只比较E与E-NoOpt。

**历史桌面campaign只读，不对其执行run、select或选择修订。** 新实验另行prepare。旧 `remaining` 自动执行/发布入口已移除；显式清单执行器仍可用于经批准的新工作campaign：

```powershell
& $py -B -m evaluation_tools run --campaign "$workRoot/campaign-new" --plan "$workRoot/approved-plan.json" --batch d
```

batch可选f-low/f-rest/d/e，清单必须匹配campaign；不可照搬旧run ID，取消项不能执行。外部执行入口拒绝写入带归档校验清单的历史目录，不自动连跑或发布。

## 5. 重建已完成论文数据

以下仅读取原记录、写入新外部目录，不执行simulation，也不依赖已删除的review/evaluation工作副本。

```powershell
& $py -B -m evaluation_tools paper --campaign $campaign --experiments a b c --output-dir "$workRoot/abc"
& $py -B -m evaluation_tools.remaining_report --campaign $campaign --plan $plan --experiment f --output-dir "$workRoot/f"
# remaining_report还支持d/e，分别选择新输出目录。
& $py -B -m tools.readable.export --archive-root $archive --output-dir "$workRoot/readable"
```

d/e/f使用最终清单，不用通用矩阵的历史引用。可读工具严格验证已完成论文布局：342个唯一运行，a/b/c/d/e/f引用24/48/48/96/42/24，附192网格与逐seed。它不为任意新矩阵自动选参。先校验完整归档，再逐请求重算p50/p99/吞吐，核对CSV和Markdown；无需Node。禁止 `python -O`，失败目录带 `INCOMPLETE.txt`，不能视为完整交付。

## 6. 查询flip与DiT硬件占比

用阅读包 `provenance/数据来源索引.csv` 或完整归档 `paper_tables/*_runs.json`找到run ID。示例为Wan2.2/Cluster1/E/6 req/min：

```powershell
& $py -B -m evaluation_tools.inspect_run --campaign $campaign --run-id wan22_ti2v_5b_cluster1-E-05047fb1a59a6b39ef9fdc95 --output-file "$workRoot/run-inspection.json"
```

命令核验运行原始文件摘要，给出flip方向计数及各阶段/硬件的执行尝试数、完成数、实际工作量、服务时间和占比。`work_share`、`completion_share`为0–1比例，硬件型号不混并。

| 问题 | 底层数据与口径 |
|---|---|
| flip多少次 | `flips.json`：方向、检测和完成时间；观察期/排空期按对应时间划分 |
| DiT在H100/A100的工作比例 | `attempts.jsonl.gz`中stage=DiT，按hardware求和executed_work（steps）再除以总steps；包含迁移前执行部分 |
| DiT完整任务由谁完成 | stage=DiT且exit_reason=completed按hardware计数，与steps占比不同 |
| PE实际工作分配 | stage=PE的executed_work为token；reprefill时间不算新token |
| 队列/调度/边界/迁移 | `events.jsonl.gz`与attempts的enter/start/exit、queue_insertion和进度；接收/尝试次数包含迁移重试 |
| 物理GPU/生命周期/零任务实例 | `instances.jsonl.gz`和`run.json`；Comb按一个物理实例、两个模块处理 |

排空吞吐=完成请求数/(最后完成时间−首次到达时间)。p50/p99为单次运行所有端到端时延的线性插值；五seed先各自计算，再给均值/最小/最大，范围不是置信区间。原始数据保留SLO与历史策略，当前论文导出不展示它们。

## 7. 归档与校验

```powershell
& $py -B -m evaluation_tools verify-archive --directory $archive
& $py -B -m evaluation_tools archive --campaign "$workRoot/campaign-new" --output-dir "$workRoot/archive-new"
```

`archive`只打包指定campaign、自带输入、匹配的运行时源码及当前功能工具/文档，不隐式复制邻接campaign或review。可选 `--source-dir`指定匹配源码包，`--evidence`包含额外证据，`--original-trace`包含全量原始trace并核对摘要；不指定后者时只承诺保留冻结请求输入。输出必须在源码/输入之外且不存在。目录链接被拒绝，复制后核对SHA-256。

`publish --staged <新归档> --destination <新发布位置>`复制经校验归档，遇同名不同内容拒绝覆盖。历史三个桌面PAVE目录保持只读。一次性执行/发布流程、测试和修复证据在历史归档与清理备份中可恢复。
