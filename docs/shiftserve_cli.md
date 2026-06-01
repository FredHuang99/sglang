# ShiftServe CLI 手册

本文件和 `shiftserve_cookbook.md` 一一对应：这里每个会影响实验分支的 CLI 参数，都会在 cookbook 中标注它影响的 class/function 和代码逻辑。

## 公共输入

- `--profile-json`：PE/TE/DiT/VAE 的 profile。调度器通过 `StageCostProfile` 和 config loader 读取 timing。
- `--deployment-json`：node、instance、port、rank、flip plan。启动前由 `PortAllocator` 做显式端口冲突检查。
- `--traffic-json`：duration、request rate、short/long interval。`build_request_specs` 会把它转成确定性的 request id 和发送时间。
- `--out-dir`：输出目录，包含 `request_events.csv`、`request_events.jsonl`、`run_summary.csv`。

## Validation 命令

```bash
python -m sglang.shiftserve.cli validate-config \
  --deployment-json ./configs/deployment.json \
  --profile-json ./configs/profile.json \
  --traffic-json ./configs/traffic.json

python -m sglang.shiftserve.cli launch-plan \
  --deployment-json ./configs/deployment.json \
  --pe-model-path /mnt/models/pe \
  --diffusion-model-path /mnt/models/wan \
  --server-addr tcp://10.0.0.2:5555 \
  --weighted-schedule false

python -m sglang.shiftserve.cli simulate \
  --traffic-json ./configs/traffic.json \
  --mode manual_flip \
  --weighted-schedule false \
  --out-dir ./outputs/validation_round_robin
```

## Baseline 和 Ablation

```bash
# no-flip + round-robin baseline
python -m sglang.shiftserve.cli simulate \
  --traffic-json ./configs/traffic.json \
  --mode no_flip \
  --weighted-schedule false \
  --out-dir ./outputs/no_flip_round_robin

# can-flip + round-robin baseline
python -m sglang.shiftserve.cli simulate \
  --traffic-json ./configs/traffic.json \
  --mode can_flip \
  --weighted-schedule false \
  --out-dir ./outputs/can_flip_round_robin

# optimized weighted scheduler
python -m sglang.shiftserve.cli simulate \
  --traffic-json ./configs/traffic.json \
  --mode can_flip \
  --weighted-schedule true \
  --margin-enabled \
  --margin-ratio 0.1 \
  --out-dir ./outputs/can_flip_weighted_margin

# optimized launch plan：rank0 broadcast + fixed 2GiB transfer buffer
python -m sglang.shiftserve.cli launch-plan \
  --deployment-json ./configs/deployment.json \
  --pe-model-path /mnt/models/pe \
  --diffusion-model-path /mnt/models/wan \
  --server-addr tcp://10.0.0.2:5555 \
  --weighted-schedule true \
  --rank0-broadcast \
  --transfer-pool-size 2147483648 \
  --transfer-pin-memory auto \
  --max-slots-per-instance 1
```

## Trace CSV 转 Traffic JSON

```bash
python -m sglang.shiftserve.cli traffic-from-csv \
  --dominant-intervals-csv /home/ubuntu/traces/window1_hour1_dominant_intervals_thr98.csv \
  --out-json ./configs/traffic.json \
  --default-rate-per-min 4
```

## 启动参数默认值

`LaunchCommandBuilder.build_pe_command` 生成的 PE/LLM 命令默认包含：

- `--disable-piecewise-cuda-graph`
- `--cuda-graph-max-bs 1`
- `--chunked-prefill-size 512`
- `--max-running-requests 1`
- `--max-total-tokens 4096`
- `--skip-server-warmup`
- 如通过 `LaunchDefaults` 配置更大 KV cache，可额外启用 `--mem-fraction-static 0.9`

`LaunchCommandBuilder.build_diffusion_role_command` 生成的 diffusion role 命令默认包含：

- `--warmup false`
- `--dit-cpu-offload false`
- `--dit-layerwise-offload false`
- `--text-encoder-cpu-offload false`
- `--image-encoder-cpu-offload false`
- `--vae-cpu-offload false`
- `--pin-cpu-memory false`
- `--disagg-transfer-pool-size 2147483648`
- `--disagg-transfer-calibration-mode fixed`
- `--disagg-max-slots-per-instance 1`
- `--disagg-transfer-pin-memory auto`

注意：`--pin-cpu-memory false` 是模型侧 pin memory；`--disagg-transfer-pin-memory auto` 是 transfer buffer pinning，两者是独立开关。

## 参数表

| 参数 | 默认值 | 影响的代码路径 |
| --- | --- | --- |
| `--mode` | `no_flip` | `cli.py` 中选择 validation/manual/can-flip 实验模式 |
| `--weighted-schedule` | `false` | `SchedulerMode.from_weighted_flag`；`ShiftServeScheduler.select` 选择 round-robin 或 weighted 分支 |
| `--window-size` | `16` | PE bin estimator 和 `HysteresisFlipMonitor` 的窗口大小 |
| `--margin-enabled` | `false` | 是否启用 high/low hysteresis threshold |
| `--margin-ratio` | `0.0` | 根据 midpoint 计算 high/low threshold |
| `--rank0-broadcast` | `false` | diffusion launch command 是否追加 `--diffusion-weight-load-mode rank0-broadcast --diffusion-weight-broadcast-components transformer,vae` |
| `--transfer-pool-size` | `2147483648` | fixed disagg transfer buffer size |
| `--transfer-pin-memory` | `auto` | 只影响 transfer buffer pinning，不影响模型侧 `--pin-cpu-memory false` |
| `--max-slots-per-instance` | `1` | capacity tracking；默认保持 1 来匹配 bs=1 |
| `--disagg-timeout` | `3600` | diffusion role/server timeout |
| `--disagg-downstream-timeout` | `1800` | encoder->DiT 和 DiT->VAE downstream wait timeout |
| `--disagg-transfer-calibration-mode` | `fixed` | generated diffusion launch args 跳过 warmup buffer resizing |

## 输出文件

- `request_events.csv`：每个 lifecycle event 一行，包含 stage、instance、generated tokens、completed steps、migration/fallback reason。
- `request_events.jsonl`：和 CSV 相同的事件，方便脚本解析。
- `run_summary.csv`：固定 7 个指标：`p50`、`p90`、`p99`、`throughput`、`lifespan`、`slo10_attainment`、`slo5_attainment`。
