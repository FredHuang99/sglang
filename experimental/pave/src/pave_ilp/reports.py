"""Compact Chinese reports built from the same structured deployment artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict

from .profiles import GENERATORS
from .templates import policy


def case_name(generator: str, scenario: str) -> str:
    return f"{GENERATORS[generator]}_{scenario}"


def n(value: float) -> str:
    return f"{value:.6f}"


def ranges(values) -> str:
    values = sorted(set(values))
    pieces = []
    for value in values:
        if pieces and value == pieces[-1][-1] + 1:
            pieces[-1].append(value)
        else:
            pieces.append([value])
    return ",".join(str(p[0]) if len(p) == 1 else f"{p[0]}–{p[-1]}" for p in pieces)


def compact_layout(deployment: dict) -> str:
    counts = Counter((i["hardware"], i["template"]) for i in deployment["instances"])
    return "; ".join(
        f"{hw}: {count}×{template}" for (hw, template), count in sorted(counts.items())
    )


def table(headers: list[str], rows: list[list[str]]) -> str:
    def row(values):
        return "| " + " | ".join(str(v).replace("|", "\\|") for v in values) + " |"

    return "\n".join([row(headers), row(["---"] * len(headers)), *(row(r) for r in rows)])


def deployment_section(deployment: dict) -> str:
    grouped = defaultdict(list)
    for item in deployment["instances"]:
        grouped[item["hardware"], item["template"]].append(item)
    rows = []
    for (hw, template), members in sorted(grouped.items()):
        first = members[0]
        per_instance = "; ".join(
            f"{s} {n(p['capacity_req_s'])}" for s, p in first["stages"].items()
        )
        aggregate = "; ".join(
            f"{s} {n(p['capacity_req_s'] * len(members))}" for s, p in first["stages"].items()
        )
        node_ids = sorted({i["node"] for i in members})
        location = (
            node_ids[0]
            if len(node_ids) == 1
            else f"{node_ids[0]} … {node_ids[-1]}（{len(node_ids)}节点）"
        )
        rows.append(
            [
                hw,
                template,
                str(len(members)),
                location,
                per_instance,
                aggregate,
                f"{first['memory_per_gpu_gb']:.4f}",
            ]
        )
    cpu = deployment["cpu_te"]
    rows.append(
        [
            "CPU逻辑池",
            "TE_CPU",
            str(cpu["instances"]),
            "按论文式(5)配置",
            f"TE {n(cpu['capacity_per_instance_req_s'])}",
            f"TE {n(cpu['aggregate_capacity_req_s'])}",
            "未测量",
        ]
    )
    capacities = deployment["stage_capacities_req_s"]
    lines = [
        f"### 输出长度 {deployment['output_tokens']}",
        "",
        table(
            [
                "硬件",
                "模板",
                "实例数",
                "所在节点",
                "单实例容量 req/s",
                "合计贡献 req/s",
                "逐卡显存 GB",
            ],
            rows,
        ),
        "",
        "阶段合计："
        + "；".join(f"{s} **{n(capacities[s])}**" for s in ("PE", "TE", "DiT", "VAE"))
        + " req/s。",
        f"系统吞吐 **{n(deployment['throughput_req_s'])} req/s**；瓶颈：{', '.join(deployment['bottlenecks'])}。",
        f"GPU使用 {sum(i['bundle_size'] for i in deployment['instances'])}/{sum(nd['gpu_count'] for nd in deployment['nodes'])}；"
        f"核心实例 {len(deployment.get('core_instances', deployment['instances']))}，填充新增 {len(deployment.get('fragment_actions', []))} 个。",
        "",
    ]
    return "\n".join(lines)


def actions_table(source: dict, actions: list[dict]) -> str:
    old = {i["id"]: i for i in source["instances"]}
    if not actions:
        return "无需转换 GPU 实例。"
    grouped = defaultdict(list)
    for action in actions:
        lhs = Counter(old[idx]["template"] for idx in action["remove_instance_ids"])
        rhs = Counter(i["template"] for i in action["add_instances"])
        hw = next(n["hardware"] for n in source["nodes"] if n["name"] == action["node"])
        grouped[
            hw, tuple(sorted(lhs.items())), tuple(sorted(rhs.items())), tuple(action["gpu_ids"])
        ].append(action["node"])
    rows = []
    for (hw, lhs, rhs, gpu_ids), names in sorted(grouped.items()):
        label = names[0] if len(names) == 1 else f"{names[0]} … {names[-1]}（{len(names)}节点）"

        def fmt(counts):
            return " + ".join(f"{count}×{name}" for name, count in counts) or "空闲"

        rows.append([hw, label, ranges(gpu_ids), fmt(lhs), fmt(rhs)])
    return table(["硬件", "节点", "GPU ID", "每节点移除", "每节点新增"], rows)


def render_case(source: dict, target: dict, flip: dict, duration_s: float | None = None) -> str:
    full = flip["full"]["benefit"]
    restricted = flip["restricted"]["benefit"]
    name = case_name(source["generator"], source["scenario"])
    lines = [
        f"# {source['generator']} · {source['scenario']}",
        "",
        f"输入 {source['input_tokens']} tokens；输出 {source['output_tokens']} → {target['output_tokens']} tokens；"
        f"KV pool {source['kv_cache_tokens']} tokens。以下均为离线稳态容量，单位 req/s。",
        "",
        f"**完整 flip 增益：{n(full['delta_req_s'])} req/s（{full['improvement_percent']:.2f}%）；"
        f"restricted 增益：{n(restricted['delta_req_s'])} req/s（{restricted['improvement_percent']:.2f}%）。**",
        "",
        "## 部署",
        "",
        deployment_section(source),
        deployment_section(target),
        "## 相同目标 workload 下的收益",
        "",
        table(
            ["模式", "系统吞吐 req/s", "相对 no-flip 增加 req/s", "提升", "每分钟理想额外处理量"],
            [
                ["保留原部署，不 flip", n(full["no_flip_req_s"]), "0", "0%", "0"],
                [
                    "完整目标 flip",
                    n(full["with_flip_req_s"]),
                    n(full["delta_req_s"]),
                    f"{full['improvement_percent']:.2f}%",
                    n(full["extra_requests_per_minute"]),
                ],
                [
                    "restricted flip",
                    n(restricted["with_flip_req_s"]),
                    n(restricted["delta_req_s"]),
                    f"{restricted['improvement_percent']:.2f}%",
                    n(restricted["extra_requests_per_minute"]),
                ],
            ],
        ),
        "",
        "no-flip 阶段容量："
        + "；".join(
            f"{s} {n(flip['no_flip']['stage_capacities_req_s'][s])}"
            for s in ("PE", "TE", "DiT", "VAE")
        )
        + "。",
        f"restricted 距完整目标 {n(flip['restricted']['gap_to_full_req_s'])} req/s；"
        f"模板计数相同：{'是' if flip['restricted']['reaches_full_template_counts'] else '否'}；"
        f"具体GPU布局相同：{'是' if flip['restricted']['reaches_full_layout'] else '否'}。",
        "",
        f"2048阶段持续 T 秒时，完整 flip 的理想额外处理量为 **{n(full['delta_req_s'])} × T**；"
        f"restricted 为 **{n(restricted['delta_req_s'])} × T**。"
        "该估计要求有足够待处理请求，忽略检测、迁移和启动成本，不代表 simulation 实测收益。",
        "",
        "## 转换方案",
        "",
        "### 完整目标",
        "",
        actions_table(source, flip["full"]["actions"]),
        "",
        f"CPU TE 实例：{source['cpu_te']['instances']} → {target['cpu_te']['instances']}。",
        "",
        "### Restricted",
        "",
        actions_table(source, flip["restricted"]["actions"]),
        "",
        f"CPU TE 实例：{source['cpu_te']['instances']} → {flip['restricted']['cpu_instances_after']}。",
        "",
        "## 模板与求解口径",
        "",
        ", ".join(f"`{t}`" for t, _, _ in policy(source["scenario"])) + "；另配 `TE_CPU`。",
        "",
        "fragment filling：节点内同模板优先，无法继续放置同模板时才按最少多余容量填充；只填已启用节点。",
        "Comb 是一个物理实例，DiT/VAE 分别按独立无干扰容量计入。PE 与生成器分离。",
        f"CPU TE 单实例 {source['cpu_te']['latency_s']:.5f}s；实例数按 ceil(λ_GPU × T_TE) 计算。",
        "全部容量求解及放置匹配均要求求解器返回 optimal；容差与每阶段统计见配置 JSON。",
        "",
        f"配置：[源](../deployments/{name}_{source['output_tokens']}.json)、"
        f"[目标](../deployments/{name}_{target['output_tokens']}.json)、[两种flip](../flips/{name}.json)。",
        "",
    ]
    if duration_s is not None:
        lines.insert(0, "")
        lines.extend(
            [
                f"指定目标阶段 {duration_s:g}s：完整 flip 理想增加 {full['delta_req_s'] * duration_s:.6f} 条请求，"
                f"restricted 理想增加 {restricted['delta_req_s'] * duration_s:.6f} 条。",
                "",
            ]
        )
    return "\n".join(lines)


def summary_records(cases: list[tuple]) -> list[dict]:
    return [
        {
            "case": case_name(s["generator"], s["scenario"]),
            "generator": s["generator"],
            "scenario": s["scenario"],
            "source_req_s": s["throughput_req_s"],
            "no_flip_req_s": f["no_flip"]["throughput_req_s"],
            "target_req_s": t["throughput_req_s"],
            **{f"full_{k}": v for k, v in f["full"]["benefit"].items()},
            "restricted_req_s": f["restricted"]["deployment"]["throughput_req_s"],
            "restricted_delta_req_s": f["restricted"]["benefit"]["delta_req_s"],
            "restricted_improvement_percent": f["restricted"]["benefit"]["improvement_percent"],
        }
        for s, t, f in cases
    ]


def render_summary(cases: list[tuple]) -> str:
    rows = []
    for r in summary_records(cases):
        rows.append(
            [
                f"[{r['generator']} · {r['scenario']}](reports/{r['case']}.md)",
                n(r["source_req_s"]),
                n(r["no_flip_req_s"]),
                n(r["target_req_s"]),
                n(r["full_delta_req_s"]),
                f"{r['full_improvement_percent']:.2f}%",
                n(r["restricted_req_s"]),
                f"{r['restricted_improvement_percent']:.2f}%",
            ]
        )
    return "\n".join(
        [
            "# PAVE 离线部署与 flip 收益",
            "",
            "两个 Wan 模型各运行 Cluster1、Cluster2、ClusterSimu。所有部署均采用节点内同模板优先的 fragment filling。",
            "",
            table(
                [
                    "实验",
                    "源bin容量",
                    "目标bin no-flip",
                    "完整目标容量",
                    "完整增量",
                    "完整提升",
                    "Restricted容量",
                    "Restricted提升",
                ],
                rows,
            ),
            "",
            "吞吐与增量单位均为 req/s。提升分母是保留源部署运行目标 workload 的吞吐。",
            "",
            "完整目标与 restricted 可能得到相同吞吐而保留不同的物理布局；详见每组转换表与 JSON。",
            "该表为稳态容量比较，不包含在线调度、迁移和冷启动损失。",
            "",
            "- [Table 3 部署素材](table3.md)",
            "- [审查与论文对齐](audit.md)",
            "- [Linux复现命令](commands.md)",
            "- [运行参数与环境](run_manifest.json)",
            "",
        ]
    )


def render_table3(cases: list[tuple]) -> str:
    rows = []
    for source, target, flip in cases:
        old = {i["id"]: i for i in source["instances"]}
        lhs, rhs = Counter(), Counter()
        for action in flip["full"]["actions"]:
            lhs.update(
                (old[idx]["hardware"], old[idx]["template"])
                for idx in action["remove_instance_ids"]
            )
            rhs.update((i["hardware"], i["template"]) for i in action["add_instances"])

        def fmt(counts):
            return (
                "; ".join(f"{hw}: {count}×{t}" for (hw, t), count in sorted(counts.items())) or "无"
            )

        rows.append(
            [
                f"{source['generator']} / {source['scenario']}",
                compact_layout(source),
                compact_layout(target),
                f"{fmt(lhs)} → {fmt(rhs)}",
                f"{source['cpu_te']['instances']} → {target['cpu_te']['instances']}",
            ]
        )
    return (
        "# Table 3 更新素材\n\n以下为新数据和已确认模板白名单下重新求得的部署，不复用论文旧数值。\n\n"
        + table(["设置", "Source", "Target", "完整Flip", "CPU TE数量"], rows)
        + "\n"
    )


AUDIT = """# ILP 代码审查与论文对齐

本次依据 PAVE 论文 §4、§8、aiconfigurator 工作区代码、新 profile 及用户确认的数据口径，迁移离线部署搜索。未迁移或运行 simulation。

| 检查项 | 旧实现或论文现状 | 本次处理及影响 |
| --- | --- | --- |
| 外部profile | 适配器加载文件后仍返回内置旧数据 | 解析字面量并实际使用输入数据；JSON快照及字段摘要可追溯 |
| GPU显存 | 逐卡测量值与bundle总HBM比较 | 改为每卡检查；Comb逐卡求和，TP/SP不再额外放大HBM额度 |
| 硬件映射 | A100型号未完整区分，部分H100采用旧H200数据 | 四类硬件使用已确认的新时延和显存映射 |
| CPU TE | 旧代码6.769752s，论文2.48s | Wan2.2为3.12839s，Wan2.1为3.18908s；按式(5)重算数量 |
| 模型字段误名 | 提示文档误用了Wan2.2的Wan2.1启动时延；早期数据CPU字段误名 | 启动时延按模型身份读取；兼容已确认CPU别名并记录来源 |
| 共驻模型 | 旧ILP和模拟器给DiT/VAE独立容量 | 按用户确认保留此模型；Comb按一个物理实例计数，PE不与生成器共驻 |
| 模板展开 | 旧代码已在求解前构建TemplateProfile | 保留预展开方法；所有求解阶段读取固定容量系数 |
| 矩阵ILP | y为节点分区数，z为模板bundle数，最大化最小阶段容量 | 与§4.2式(2)至式(4)一致；使用稀疏矩阵 |
| 同优解 | 极小权重混在吞吐目标中 | 分阶段最大吞吐、最少物理实例、最少启用节点，记录容差 |
| 实例标识 | 不同大小bundle可能使用相同标签 | 节点、GPU ID、模板共同确定实例ID；Comb不按阶段重复计数 |
| Fragment filling | 旧报告同时输出两套，论文未详细说明 | 只交付同模板优先主部署；最少多余容量作为兜底 |
| Flip | 聚合计数差异和restricted结果不等于具体物理映射 | 完整目标先匹配保留实例，再减少改动GPU；restricted独立优化并报告差距 |
| §8.2收益比较 | 源与目标在各自bin下的容量差不是直接flip收益 | 固定2048 workload，比较目标部署与源部署重估容量 |
| 搜索规模 | §4.3中35/105变量等统计对应旧裁剪集合 | 输出实际变量及约束数，不强行复现旧统计 |
| 网络与SLO | §4.4提出max-flow扩展；论文描述效率/SLO裁剪 | 本次网络非瓶颈，采用显式白名单；不声称实现带宽需求或数值SLO校验 |
| 启动开销 | profile含PE与generator pipeline启动测量 | 归档两种启动时延，不将整条generator启动测量冒充独立VAE启动时延 |

## 解释与边界

在本次容量模型下，Comb用同一组GPU提供DiT和VAE的独立容量，可以节省单独部署VAE的GPU；因此偏向Comb有明确的建模原因。同优解又优先减少实例数量。实际结果仍由容量与逐卡显存约束共同决定，不强制所有设置选Comb。

论文的CPU实例公式、矩阵ILP和网络非瓶颈目标得到保留。代码对物理布局、fragment filling、完整与restricted转换、求解状态和结果校验的描述更加具体；这些内容可补充到论文，但不应写成在线机制已在本次验证。

原始ILP核心解与fragment-filled部署分开留存在JSON。节点内同模板优先只使用已启用节点空闲GPU，无法继续复制已有模板时才比较合法剩余填充组合。填充不减少核心实例，不强制启动空闲节点。

CPU数量是所需容量推导，依赖CPU可提供足够副本的论文假设；缺少CPU峰值内存测量，因此不提供虚假的DRAM可行性认证。去噪时延固定为完整50步测量。所有收益均为充分负载、零转换损失下的稳态容量差。

修改后的模型、数据和模板限制可能改变Table 3实例数量及§8.2数字。旧实验结果用于解释差异，不作为新求解器的数值金标准。
"""


def render_commands(manifest: dict) -> str:
    p = manifest["parameters"]
    versions = manifest["environment"]
    pytest_requirement = f"pytest=={versions['pytest']}" if versions["pytest"] else "pytest>=8,<10"
    generators = " ".join(f"--generator {g}" for g in p["generators"])
    scenarios = " ".join(f"--scenario {s}" for s in p["scenarios"])
    optional = (
        f" --long-phase-duration-s {p['long_phase_duration_s']}"
        if p["long_phase_duration_s"] is not None
        else ""
    )
    return f'''# Linux Bash 或 zsh 复现命令

本次实际运行环境为 {versions["platform"]}，Python {versions["python"]}，NumPy {versions["numpy"]}，SciPy {versions["scipy"]}。下列命令使用相同参数和同一份profile快照；Linux命令未在本次Windows运行中实际执行。

## 准备

将包含pave_ilp实现的仓库和本交付目录复制到Linux。只需要CPU，不需要安装SGLang、PyTorch、CUDA或模型权重。将以下两个路径改成实际位置；Python建议使用与本次一致的3.12系列。

```bash
REPO="$HOME/sglang"
ARTIFACTS="$HOME/Desktop/PAVE_ILP"
cd "$REPO/experimental/pave"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install "numpy=={versions["numpy"]}" "scipy=={versions["scipy"]}" "{pytest_requirement}"
export PYTHONPATH="$PWD/src"
python -m pytest tests -q
```

## 重现本次全部实验

输出使用新目录，避免覆盖原交付物。可将OUT设置为任意新目录。

```bash
OUT="$HOME/Desktop/PAVE_ILP_reproduced"
python -m pave_ilp plan \\
  {generators} \\
  {scenarios} \\
  --profile-data "$ARTIFACTS/profiles.json" \\
  --input-tokens {p["input_tokens"]} \\
  --source-output-tokens {p["source_output_tokens"]} \\
  --target-output-tokens {p["target_output_tokens"]} \\
  --kv-cache-tokens {p["kv_cache_tokens"]} \\
  --mip-time-limit-s {p["mip_time_limit_s"]:g} \\
  --output-dir "$OUT"{optional}
python -m pave_ilp validate --output-dir "$OUT"
```

省略`--profile-data`时使用包内快照。也可传入原始`data_new.py`；导入只允许字面量赋值，不执行Python代码。已有非空输出目录默认拒绝覆盖；确认是自己的旧输出时才加`--overwrite`，且必须保持相同实验矩阵。

## 只运行一组

```bash
python -m pave_ilp plan \\
  --generator wan2.2-ti2v-5b --scenario cluster2 \\
  --profile-data "$ARTIFACTS/profiles.json" \\
  --input-tokens 128 --source-output-tokens 512 --target-output-tokens 2048 \\
  --kv-cache-tokens 4096 --mip-time-limit-s 120 \\
  --output-dir "$HOME/Desktop/PAVE_ILP_wan22_cluster2"
```

## 参数含义

| 参数 | 含义与设置 |
| --- | --- |
| `--generator` | 可重复；支持wan2.2-ti2v-5b、wan2.1-t2v-1.3b；省略则两者均运行 |
| `--scenario` | 可重复；cluster1、cluster2、clustersimu；省略则三种均运行 |
| `--input-tokens` | PE输入长度，默认128，必须存在完整对应profile |
| `--source-output-tokens` | 初始bin输出长度，默认512 |
| `--target-output-tokens` | 目标bin输出长度，默认2048 |
| `--kv-cache-tokens` | 固定KV pool长度，默认4096，至少覆盖单请求输入加输出；不随bin自动变化 |
| `--mip-time-limit-s` | 每个MILP优化阶段的时间上限，默认120秒；超时会保存诊断并使运行失败，不冒充最优解 |
| `--profile-data` | 原始字面量Python表或本工具导出的JSON快照 |
| `--long-phase-duration-s` | 可选，将容量差乘以指定秒数；仅为理想额外处理量估计，不运行simulation |
| `--output-dir` | 输出目录，含部署、flip、profile、报告与运行清单 |
| `--overwrite` | 显式覆盖同一实验矩阵已有输出；默认不覆盖 |

模板白名单见各组报告，fragment filling固定为同模板优先。启动时延仅归档，去噪固定使用50步总耗时，CPU TE按式(5)派生。单组JSON可在`deployments/`与`flips/`读取，汇总数值在`summary.json`。

验证命令检查数据摘要、GPU占用、白名单、逐卡显存、容量总计、CPU数量和两种flip动作，并从部署重新计算收益。不同平台或求解器版本可能在同优解中选择不同布局；本次固定数值依赖版本并记录了代码摘要。
'''
