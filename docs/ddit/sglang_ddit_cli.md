# SGLang Diffusion DDiT 实验 CLI 说明

本文档给出单机 DDiT 实验的闭包命令：forced-switch 正确性验证，以及 `hungry_first`、`naive`、`naive_greedy`、`wsjf`、`wsjf_scale_up` 的 mixed workload E2E。命令只保留对当前 setup 有影响的参数。

统一约定：

- `<MODEL_PATH>`：Wan2.1 或当前要测的模型路径。
- `<PROFILE_JSON>`：真实 profile 文件路径；格式见 `Profile Data`。
- `<LOG_DIR>`：实验日志目录，例如 `C:\Users\woshi\Desktop\ddit_logs\hungry_first`.
- server 端 full ranks 由 `--num-gpus`、`--sp-degree`、`--ulysses-degree`、`--ring-degree` 决定，不写死只能是 8 卡；下面以 8 卡为例。
- mixed workload client 默认发送 T2V payload；TI2V / image 模型的图片路径说明见 `Mixed Client Notes`。

## Forced Switch Correctness

用途：验证单请求在 step 15/30/45 后执行 `1->2->4->8` rank switch，且 VAE 默认 1 卡。

Server:

```powershell
sglang serve `
  --model-path <MODEL_PATH> `
  --num-gpus 8 `
  --sp-degree 8 `
  --ulysses-degree 8 `
  --ring-degree 1 `
  --enable-ddit `
  --ddit-schedule-policy forced_switch `
  --ddit-log-dir <LOG_DIR>
```

Request:

```powershell
python examples\multimodal_gen\ddit_forced_switch_wan.py `
  --server-url http://127.0.0.1:30000 `
  --request-id forced_720p_001 `
  --size 1280x720 `
  --resolution-key 720p `
  --num-inference-steps 50 `
  --initial-ranks 0 `
  --switch-plan "15:1->2;30:2->4;45:4->8" `
  --ddit-vae-k 1
```

验收：

- `ddit_rank_switch.jsonl` 出现 step 15、30、45 的 `stage="dit"` switch。
- `ddit_rank_switch.jsonl` 出现 `stage="vae"` event。
- `ddit_lifecycle.csv` 中该 request 有 add/dit_start/dit_end/vae_start/vae_end。
- latent 不重新初始化，switch 发生在 completed denoising step boundary。

## Hungry First E2E

用途：验证 running DiT request 可按 hungry-first 饥饿度扩容；VAE 默认从 final DiT ranks 中选 1 卡。

Server:

```powershell
sglang serve `
  --model-path <MODEL_PATH> `
  --num-gpus 8 `
  --sp-degree 8 `
  --ulysses-degree 8 `
  --ring-degree 1 `
  --enable-ddit `
  --ddit-schedule-policy hungry_first `
  --ddit-profile-path <PROFILE_JSON> `
  --ddit-profile-model-id wan2.1-t2v-1.3b `
  --ddit-log-dir <LOG_DIR>
```

Client:

```powershell
python examples\multimodal_gen\ddit_mixed_workload_client.py `
  --server-url http://127.0.0.1:30000 `
  --num-requests 16 `
  --resolutions "144p,720p" `
  --ratios "0.5,0.5" `
  --rate burst `
  --seed 42
```

验收：

- DiT start event 的 `reason` 为 `waiting_queue`。
- 扩容 event 的 `reason` 为 `hungry_first`。
- VAE event 的 `reason` 为 `dit_to_vae`。
- `ddit_op_trace.jsonl` 中同一 `wave_id` 可出现不同 request 的 rank-disjoint `dit_step`。

备注：默认 VAE 是 1 卡，所以 server 命令不写 `--ddit-vae-gpus`。只有要专门验证 hungry-first 的多卡 VAE 时，才加 `--ddit-vae-gpus 2`、`4` 或 `8`。

## Naive E2E

用途：验证 FCFS head-of-line；队首 request 必须拿到 profile 中的 opt GPU 数才启动 DiT，不跳过队首。

Server:

```powershell
sglang serve `
  --model-path <MODEL_PATH> `
  --num-gpus 8 `
  --sp-degree 8 `
  --ulysses-degree 8 `
  --ring-degree 1 `
  --enable-ddit `
  --ddit-schedule-policy naive `
  --ddit-profile-path <PROFILE_JSON> `
  --ddit-profile-model-id wan2.1-t2v-1.3b `
  --ddit-log-dir <LOG_DIR>
```

Client:

```powershell
python examples\multimodal_gen\ddit_mixed_workload_client.py `
  --server-url http://127.0.0.1:30000 `
  --num-requests 16 `
  --resolutions "144p,720p" `
  --ratios "0.5,0.5" `
  --rate burst `
  --seed 42
```

验收：

- DiT start event 的 `reason` 为 `naive`。
- 如果队首 request 的 `opt_gpus_num` 大于当前 free ranks，后续 request 不会越过队首启动。
- VAE event 的 `reason` 为 `naive_same_ranks`，`old_ranks` 与 `new_ranks` 相同。

## Naive Greedy E2E

用途：验证 FCFS head-of-line；资源不足时降级到 `floor_power_of_two(min(opt_k, free_count))` 并固定 ranks 跑完 DiT/VAE。

Server:

```powershell
sglang serve `
  --model-path <MODEL_PATH> `
  --num-gpus 8 `
  --sp-degree 8 `
  --ulysses-degree 8 `
  --ring-degree 1 `
  --enable-ddit `
  --ddit-schedule-policy naive_greedy `
  --ddit-profile-path <PROFILE_JSON> `
  --ddit-profile-model-id wan2.1-t2v-1.3b `
  --ddit-log-dir <LOG_DIR>
```

Client:

```powershell
python examples\multimodal_gen\ddit_mixed_workload_client.py `
  --server-url http://127.0.0.1:30000 `
  --num-requests 16 `
  --resolutions "144p,720p" `
  --ratios "0.5,0.5" `
  --rate burst `
  --seed 42
```

验收：

- DiT start event 的 `reason` 为 `naive_greedy`。
- 启动时 `new_ranks` 的数量是 power-of-two，且不超过 `opt_k` 和当时 free ranks。
- VAE event 的 `reason` 为 `naive_greedy_same_ranks`。

## WSJF E2E

用途：验证 prepared waiting window 内按 estimated remaining DiT time 选择最短任务。

Server:

```powershell
sglang serve `
  --model-path <MODEL_PATH> `
  --num-gpus 8 `
  --sp-degree 8 `
  --ulysses-degree 8 `
  --ring-degree 1 `
  --enable-ddit `
  --ddit-schedule-policy wsjf `
  --ddit-profile-path <PROFILE_JSON> `
  --ddit-profile-model-id wan2.1-t2v-1.3b `
  --ddit-window-size 8 `
  --ddit-log-dir <LOG_DIR>
```

Client:

```powershell
python examples\multimodal_gen\ddit_mixed_workload_client.py `
  --server-url http://127.0.0.1:30000 `
  --num-requests 24 `
  --resolutions "144p,360p,720p" `
  --ratios "0.5,0.25,0.25" `
  --rate burst `
  --seed 42
```

验收：

- DiT start event 的 `reason` 为 `wsjf`。
- 窗口内 request 的启动顺序应符合 profile 估计时间的短作业优先。
- VAE event 的 `reason` 为 `wsjf_same_ranks`。

## WSJF Scale-Up E2E

用途：验证 waiting 选择使用 WSJF，running request 优先按 hungry-first 规则扩容。

Server:

```powershell
sglang serve `
  --model-path <MODEL_PATH> `
  --num-gpus 8 `
  --sp-degree 8 `
  --ulysses-degree 8 `
  --ring-degree 1 `
  --enable-ddit `
  --ddit-schedule-policy wsjf_scale_up `
  --ddit-profile-path <PROFILE_JSON> `
  --ddit-profile-model-id wan2.1-t2v-1.3b `
  --ddit-window-size 8 `
  --ddit-log-dir <LOG_DIR>
```

Client:

```powershell
python examples\multimodal_gen\ddit_mixed_workload_client.py `
  --server-url http://127.0.0.1:30000 `
  --num-requests 24 `
  --resolutions "144p,360p,720p" `
  --ratios "0.5,0.25,0.25" `
  --rate burst `
  --seed 42
```

验收：

- Waiting request 的 DiT start event `reason` 为 `wsjf_scale_up`。
- Running request 扩容 event `reason` 为 `wsjf_scale_up`，且 `old_ranks` 非空。
- VAE event 的 `reason` 为 `wsjf_scale_up_same_ranks`。

## Profile Data

`--ddit-profile-path` 的优先级高于内置 placeholder。传入 multi-model profile 时，用 `--ddit-profile-model-id` 选择具体模型。当前内置 placeholder 支持 `z-image` 和 `wan2.1-t2v-1.3b`；真实实验建议显式传 `<PROFILE_JSON>`。

Flat schema:

```json
{
  "opt_gpus_num": {
    "144p": 1,
    "360p": 2,
    "720p": 4
  },
  "dit_step_times": {
    "144p": {"1": 1.0, "2": 0.9, "4": 1.0, "8": 1.2},
    "360p": {"1": 4.0, "2": 2.1, "4": 1.8, "8": 1.9},
    "720p": {"1": 16.0, "2": 8.2, "4": 4.4, "8": 4.8}
  }
}
```

Multi-model schema:

```json
{
  "models": {
    "wan2.1-t2v-1.3b": {
      "opt_gpus_num": {
        "144p": 1,
        "360p": 2,
        "720p": 4
      },
      "dit_step_times": {
        "144p": {"1": 1.0, "2": 0.9, "4": 1.0, "8": 1.2},
        "360p": {"1": 4.0, "2": 2.1, "4": 1.8, "8": 1.9},
        "720p": {"1": 16.0, "2": 8.2, "4": 4.4, "8": 4.8}
      }
    },
    "z-image": {
      "opt_gpus_num": {
        "512p": 2,
        "1024p": 4
      },
      "dit_step_times": {
        "512p": {"1": 5.0, "2": 2.8, "4": 2.2, "8": 2.4},
        "1024p": {"1": 16.0, "2": 8.4, "4": 4.9, "8": 5.2}
      }
    }
  }
}
```

字段含义：

- `opt_gpus_num[resolution]`：该分辨率 DiT 阶段的目标 GPU 数。
- `dit_step_times[resolution][k]`：该分辨率用 k 张 GPU 跑一个 denoising step 的 profile latency。
- 若某 resolution 缺失，runtime fallback 到 `opt=1`、`step_time=1.0`，只保证可跑，不代表真实实验数据。

## Mixed Client Notes

请求数量按比例生成：

- 前 `n-1` 类数量为 `round(num_requests * ratio_i)`。
- 最后一类数量为 `num_requests - sum(previous_counts)`，保证总数严格等于 `num_requests`。
- 生成后按 `--seed` shuffle。

发送速率：

- `--rate burst`：循环内不 sleep，模拟同时到达。
- `--rate 1`：每秒 1 个请求。
- `--rate 0.2`：每 5 秒 1 个请求。
- `--rate 2`：每秒 2 个请求。
- `--rate 1.5`：每秒 1.5 个请求。

TI2V / image 模型：

- 默认图片路径由脚本从项目根目录解析到 `examples/assets/example_image.png`。
- 如果要指定图片，加 `--image-path <IMAGE_PATH>`。
- 如果要禁用图片字段，加 `--image-path ""`。
- 相对图片路径按 project root 解析，不按当前 shell 的 cwd 解析。

## Logs

生命周期 CSV：`<LOG_DIR>\ddit_lifecycle.csv`

列：

- `request_id`
- `resolution`
- `add_time`
- `dit_start_time`
- `dit_end_time`
- `vae_start_time`
- `vae_end_time`
- `status`
- `error`

常用指标：

- queue latency：`dit_start_time - add_time`
- DiT latency：`dit_end_time - dit_start_time`
- VAE latency：`vae_end_time - vae_start_time`
- E2E latency：`vae_end_time - add_time`
- CSV 最后三行是 `p50,<seconds>`、`p90,<seconds>`、`p99,<seconds>`，统计 completed requests 的 lifespan time。

Rank switch JSONL：`<LOG_DIR>\ddit_rank_switch.jsonl`

字段：`timestamp,request_id,resolution,node_id,stage,step,old_ranks,new_ranks,reason,policy`

策略闭包：

- forced：`stage="dit"` 的 step switch，加 `stage="vae"` 的 VAE event。
- hungry_first：`reason="waiting_queue"`、`reason="hungry_first"`、`reason="dit_to_vae"`。
- naive：`reason="naive"`、`reason="naive_same_ranks"`。
- naive_greedy：`reason="naive_greedy"`、`reason="naive_greedy_same_ranks"`。
- wsjf：`reason="wsjf"`、`reason="wsjf_same_ranks"`。
- wsjf_scale_up：DiT start 或 scale-up 的 `reason="wsjf_scale_up"`，VAE 的 `reason="wsjf_scale_up_same_ranks"`。

并发 trace：`<LOG_DIR>\ddit_op_trace.jsonl`

字段：`timestamp_start,timestamp_end,request_id,stage,step,ranks,wave_id,op_id,rank,action,status,error`

真并发验收：

- 同一 `wave_id` 中出现不同 `request_id` 的 `dit_step` 或 `vae_run`。
- 这些 op 的 `ranks` 不重叠。
- 这些 op 的 `[timestamp_start, timestamp_end]` 区间重叠。
