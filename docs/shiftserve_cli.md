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
  --max-slots-per-instance 1 \
  --dit-vae-dit-bs 1 \
  --dit-vae-vae-bs 1 \
  --dit-vae-stage-concurrency pipeline
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
| `--dit-vae-dit-bs` | `1` | `InstanceRuntimeState.dit_vae_bundle` 和 runtime DIT_VAE 内部 DiT stage slot 容量 |
| `--dit-vae-vae-bs` | `1` | `InstanceRuntimeState.dit_vae_bundle` 和 runtime DIT_VAE 内部 VAE stage slot 容量 |
| `--dit-vae-stage-concurrency` | `pipeline` | `scheduler_mixin._disagg_dit_vae_compute` 选择 split-stage pipeline 或 `serial_debug` 整体 forward |

## 输出文件

- `request_events.csv`：每个 lifecycle event 一行，包含 stage、instance、generated tokens、completed steps、migration/fallback reason。
- `request_events.jsonl`：和 CSV 相同的事件，方便脚本解析。
- `run_summary.csv`：固定 7 个指标：`p50`、`p90`、`p99`、`throughput`、`lifespan`、`slo10_attainment`、`slo5_attainment`。

## JSON 示例与字段说明

### `deployment.json`

```json
{
  "nodes": [
    {"node_id": "node-a", "host": "10.0.0.1", "role_host": "10.0.0.1", "node_type": "A100"},
    {"node_id": "node-b", "host": "10.0.0.2", "role_host": "10.0.0.2", "node_type": "H100"}
  ],
  "instances": [
    {
      "id": "pe-a0",
      "kind": "pe",
      "node_id": "node-a",
      "device": "cuda",
      "gpu_ids": [0],
      "ranks": 1,
      "ports": {"http": 30000},
      "source_active": true,
      "target_active": false
    },
    {
      "id": "te-b0",
      "kind": "te",
      "node_id": "node-b",
      "device": "cpu",
      "ports": {"work": 31000, "control": 31001},
      "source_active": true,
      "target_active": true
    },
    {
      "id": "ditvae-b0",
      "kind": "dit_vae",
      "node_id": "node-b",
      "device": "cuda",
      "gpu_ids": [0],
      "ranks": 1,
      "ports": {"work": 32000, "control": 32001},
      "paired_te_id": "te-b0",
      "stage_slots": {"dit_bs": 1, "vae_bs": 1},
      "source_active": true,
      "target_active": false
    }
  ],
  "flip_plan": [
    {"direction": "short_to_long", "sources": ["pe-a0"], "targets": ["ditvae-b0"]}
  ],
  "bins": {"short": 512, "long": 2048},
  "port_base": 30000
}
```

字段要求：
- `nodes[].node_id` 必填、全局唯一；`host` 是本机 bind/连接地址；`role_host` 是对其他节点可达的 role 地址；`node_type` 只用于实验标注。
- `instances[].id` 必填、全局唯一；`kind` 可填 `pe|te|dit|vae|dit_vae`，也支持 alias：`llm`、`encoder`、`denoiser`、`decoder`、`ddit_worker`。
- `ports` 必须包含该 role 的端口：`pe` 需要 `http`，`te/dit/vae/dit_vae` 需要 `work` 和 `control`；同一 `node_id` 上不能冲突。
- `paired_te_id` 用于 `dit_vae` bundle 绑定同 node TE。
- `stage_slots.dit_bs` 和 `stage_slots.vae_bs` 是同一个 `dit_vae` instance 内部的 stage capacity，默认都是 1；它们不是启动两个实例，也不是把 `--max-slots-per-instance` 改成 2。

### `profile.json`

```json
{
  "model_id": "wan-validation",
  "node_type": "H100",
  "ranks": 1,
  "tp": 1,
  "sp": 1,
  "pe": {"ttft_ms": 120.0, "tpot_ms": 8.5, "init_s": 42.0},
  "diffusion": {
    "te_s": 0.12,
    "dit_steps": 30,
    "dit_per_step_s": 0.18,
    "vae_s": 0.45,
    "init_s": 55.0
  },
  "rank0_broadcast": {"enabled": true, "init_s": 34.0}
}
```

字段要求：时间单位按 key 写明，`*_ms` 是毫秒、`*_s` 是秒；`dit_steps` 必须是正整数；`rank0_broadcast` 可选，用于 launch ablation 记录。

### `traffic.json`

```json
{
  "duration_min": 10,
  "default_rate_per_min": 4,
  "intervals": [
    {"start_min": 0, "end_min": 5, "rate_per_min": 4, "bin": "short", "input_tokens": 128},
    {"start_min": 5, "end_min": 10, "rate_per_min": 2, "bin": "long", "input_tokens": 256}
  ]
}
```

字段要求：`duration_min`、`default_rate_per_min` 是数字；`intervals[].start_min/end_min` 用分钟，必须落在实验时长内；`bin` 填 `short` 或 `long`，对应 `deployment.json` 的 `bins`；`input_tokens` 可选，不填时使用默认值。
