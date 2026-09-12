# PAVE 离线工具

CPU-only 的 ILP 部署搜索、离散事件 simulation，以及实验执行、原始数据审计和报告工具。无需导入 SGLang 推理运行时或启动 GPU 服务。

完整命令、配置区别及底层数据查询见 [CLI 使用说明](docs/cli.md)。

```powershell
cd 'C:/Users/woshi/Desktop/sglang diffusion/experimental/pave'
& '.\.venv\Scripts\python.exe' -B -m pave_ilp --help
& '.\.venv\Scripts\python.exe' -B -m pave_sim --help
& '.\.venv\Scripts\python.exe' -B -m evaluation_tools --help
```

已有 `.venv` 保留。新机器安装：Python 3.10+，`python -m venv .venv`，使用该环境的 Python 执行 `-m pip install -e .`。实验环境为 Python 3.12.10、NumPy 2.5.2、SciPy 1.18.1；准确的历史环境见每个 campaign 的 manifest。重跑可比较实验须匹配冻结环境和运行时源码。

| 目录 | 功能 |
|---|---|
| `src/pave_ilp` | profile、模板、MILP、物理放置、静态 flip、报告 |
| `src/pave_sim` | 请求、事件引擎、监控、调度、迁移、指标、campaign |
| `evaluation_tools` | 指定批次执行、完整记录审计、论文数据和归档 |
| `tools/readable` | 从完整论文归档重建人类可读数据与分析 |
| `examples` | 通用 CLI 配置样例，不等于最终论文执行清单 |

工作区只保留功能代码、必要输入和使用说明。实验结果在桌面 `PAVE_EVALUATION_RESULTS`；阅读版在 `PAVE_EVALUATION_READABLE`；部署在 `PAVE_ILP`。新实验和报告使用独立输出目录。原测试、审查和清理恢复资料保存于桌面归档及 `SGLANG_WORKSPACE_CLEANUP_BACKUP`。

ILP/simulation 源码与实验冻结版本保持一致。外部分析、归档工具单独记录版本，不改写历史 run ID。版权说明见 [LICENSE](LICENSE)、[NOTICE](NOTICE)。
