# SGLang Diffusion DDiT 实验 CLI 说明

本文档说明单机 DDiT/FlexDiT 实验、hungry schedule、fixed-k baseline、mixed workload E2E、正确性实验以及日志解析方式。

## 1. DDiT 服务启动

```powershell
sglang serve `
  --model-path <WAN_MODEL_PATH> `
  --num-gpus 8 `
  --sp-degree 8 `
  --ulysses-degree 8 `
  --ring-degree 1 `
  --enable-ddit `
  --ddit-node-id node0 `
  --ddit-local-ranks 0,1,2,3,4,5,6,7 `
  --ddit-allowed-gpu-counts 1,2,4,8 `
  --ddit-schedule-policy forced_switch `
  --ddit-initial-gpus 1 `
  --ddit-vae-gpus 1 `
  --ddit-log-dir C:\Users\woshi\Desktop\ddit_logs
```

关键参数：

- `--enable-ddit`：启用 DDiT 动态 SP、VAE k 卡控制和实验日志。
- `--ddit-schedule-policy`：调度策略，支持 `forced_switch`、`hungry_first`、`fixed_baseline`。
- `--ddit-initial-gpus`：forced-switch/DDiT 请求进入 DiT 时默认使用的卡数；不用于 fixed baseline。
- `--ddit-switch-plan`：forced switch 计划，例如 `15:1->2;30:2->4;45:4->8`。
- `--ddit-vae-gpus`：非 fixed-baseline 模式下的默认 VAE 卡数，默认 `1`。
- `--ddit-sp-degree-map`：可选 Ulysses/Ring 映射，例如 `1=1x1,2=2x1,4=2x2,8=4x2`。
- `--ddit-log-dir`：输出 `ddit_lifecycle.csv` 和 `ddit_rank_switch.jsonl` 的目录。

full ranks 由当前服务的 `--num-gpus` / `--sp-degree` / `--ulysses-degree` / `--ring-degree` / `--ddit-local-ranks` 决定，可以是 1/2/4/8，不写死为 8 卡。

## 2. Hungry Schedule

Hungry schedule 现在走单机真并发 DDiT runtime：rank0 只做 coordinator，不再把同一个 DiT step broadcast 给所有 ranks。每个 wave 内可以同时下发多个 rank-disjoint ops，例如 rank `[0]` 跑 144p request 的 step，同时 ranks `[1,2,3,4]` 跑 720p request 的 step。

```powershell
sglang serve `
  --model-path <WAN_MODEL_PATH> `
  --num-gpus 8 `
  --sp-degree 8 `
  --ulysses-degree 8 `
  --ring-degree 1 `
  --enable-ddit `
  --ddit-schedule-policy hungry_first `
  --ddit-local-ranks 0,1,2,3,4,5,6,7 `
  --ddit-allowed-gpu-counts 1,2,4,8 `
  --ddit-vae-gpus 1 `
  --ddit-profile-path <OPTIONAL_PROFILE_JSON> `
  --ddit-log-dir C:\Users\woshi\Desktop\ddit_logs
```

语义：

- text encoder 阶段始终使用当前服务 full ranks TP，不写死 8 卡。
- DiT/VAE 阶段由 coordinator 在 step boundary 生成 `CommandWave`，不同请求只要 ranks 不重叠就能同 wave 并发执行。
- 调度策略先扩容最饥饿的 running DiT 请求，再启动 waiting 请求。
- 饥饿度为 `(cur_step - last_scheduled_step) * (t(current_k) - t(opt_k))`。
- rank switch 时 latent 通过 subgroup gather + P2P send/recv 迁移，不走 world device broadcast。
- DiT 结束后，VAE 默认从 final DiT ranks 中选择 `--ddit-vae-gpus` 张卡执行。
- profile JSON 可包含 `opt_gpus_num` 和 `dit_step_times`；不传时使用占位默认表。

mixed workload 验收建议：

```powershell
python examples\multimodal_gen\ddit_mixed_workload_client.py `
  --server-url http://127.0.0.1:30000 `
  --num-requests 16 `
  --resolutions "144p,720p" `
  --ratios "0.5,0.5" `
  --rate burst `
  --seed 42
```

验收点：

- `ddit_lifecycle.csv` 能还原每个请求的 queue/DiT/VAE/E2E latency。
- `ddit_rank_switch.jsonl` 能看到 `waiting_queue`、`hungry_first`、`dit_to_vae` 事件。
- 当 profile 中 720p `opt_gpus_num=8` 且资源释放后，能看到 `reason=hungry_first` 的 DiT 扩容事件。
- `stage=vae` 事件记录 DiT->VAE rank 选择。
- `ddit_op_trace.jsonl` 中同一个 `wave_id` 应出现不同 request、不同 ranks 的重叠执行区间。

## 3. Fixed-k Baseline

Fixed baseline 不是 `--sp-degree k`。服务仍然用 full ranks 初始化 text encoder TP，`--ddit-baseline-gpus k` 只控制 DiT/VAE 阶段每个请求固定占用 k 张卡。

```powershell
sglang serve `
  --model-path <WAN_MODEL_PATH> `
  --num-gpus 8 `
  --sp-degree 8 `
  --ulysses-degree 8 `
  --ring-degree 1 `
  --enable-ddit `
  --ddit-schedule-policy fixed_baseline `
  --ddit-baseline-gpus 4 `
  --ddit-log-dir C:\Users\woshi\Desktop\ddit_logs
```

语义：

- text encoder 永远运行在当前服务 full ranks TP group 上。
- `--ddit-baseline-gpus k` 只控制 DiT/VAE 阶段固定使用 k 张卡。
- DiT 和 VAE 不分离：同一个请求的 VAE 复用该请求最终 DiT ranks。
- 多个 baseline 请求只要 rank group 不重叠，就会在 DiT step 和 VAE 阶段真并发。
- fixed baseline 下 `--ddit-vae-gpus`、请求级 `ddit_vae_k`、`ddit_vae_ranks` 不改变 VAE ranks。
- `--ddit-initial-gpus` 不参与 fixed baseline。
- rank 选择优先连续 free block；如果没有连续 block，则从 sorted free ranks 中稳定选择 k 个非连续 ranks。

## 4. Forced Switch 正确性脚本

```powershell
python examples\multimodal_gen\ddit_forced_switch_wan.py `
  --server-url http://127.0.0.1:30000 `
  --request-id forced_720p_001 `
  --resolution-key 720p `
  --size 1280x720 `
  --num-inference-steps 50 `
  --initial-ranks 0 `
  --switch-plan "15:1->2;30:2->4;45:4->8" `
  --ddit-vae-k 1
```

验收点：

- `ddit_rank_switch.jsonl` 中有 step 15、30、45 的 DiT rank 扩容。
- `ddit_rank_switch.jsonl` 中有 `stage=vae` 的 DiT->VAE rank 记录。
- `ddit_lifecycle.csv` 中该请求有完整 add/DiT/VAE 时间戳。
- 扩容时使用当前 latent gather/broadcast，不重新初始化 latent。

## 5. Mixed Workload E2E

```powershell
python examples\multimodal_gen\ddit_mixed_workload_client.py `
  --server-url http://127.0.0.1:30000 `
  --num-requests 128 `
  --resolutions "144p,720p" `
  --ratios "0.75,0.25" `
  --rate 1 `
  --seed 42
```

请求生成：

- `--num-requests x`：总请求数。
- `--resolutions "144p,360p,240p"`：分辨率标签列表。
- `--ratios "0.5,0.3,0.2"`：比例列表，和必须为 `1.0`。
- 前 `n-1` 类数量为 `round(x * ratio_i)`；最后一类为 `x - sum(previous_counts)`。
- 生成后用 `--seed` 固定 shuffle，保证可复现。

发送速率：

- `--rate 1`：每秒 1 个请求。
- `--rate 0.2`：每 5 秒 1 个请求。
- `--rate 2`：每秒 2 个请求。
- `--rate 1.5`：每秒 1.5 个请求。
- `--rate burst`：循环内不 sleep，模拟同时到达。

图片路径：

- 默认图片为项目根目录下的 `examples/assets/example_image.png`。
- client 会从脚本文件位置自动探测项目根目录，不依赖当前 shell 的 `cwd`。
- `--project-root` 可显式指定项目根。
- `--image-path` 显式传入时优先级最高；相对路径按 project root 解析。
- `--image-path ""` 可禁用图片字段。

## 6. 日志格式

生命周期 CSV：`<ddit-log-dir>\ddit_lifecycle.csv`

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

rank 切换 JSONL：`<ddit-log-dir>\ddit_rank_switch.jsonl`

字段：

- `timestamp`
- `request_id`
- `resolution`
- `node_id`
- `stage`：`dit`、`vae` 或 `baseline`
- `step`
- `old_ranks`
- `new_ranks`
- `reason`
- `policy`

示例：

```json
{"stage":"baseline","step":null,"old_ranks":[],"new_ranks":[0,1,2,3],"reason":"fixed_baseline","policy":"fixed_baseline"}
```

Hungry schedule 示例：

```json
{"stage":"dit","step":8,"old_ranks":[1,2,3,4],"new_ranks":[0,1,2,3,4,5,6,7],"reason":"hungry_first","policy":"hungry_first"}
```

并发执行 trace：`<ddit-log-dir>\ddit_op_trace.jsonl`

字段：`timestamp_start,timestamp_end,request_id,stage,step,ranks,wave_id,op_id,rank,action,status,error`。真并发验收时，检查同一个 `wave_id` 内是否有不同 `request_id` 的 `dit_step` 或 `vae_run`，且 `ranks` 不重叠、时间区间重叠。

## 7. 多机后续开发计划

第一版仍以单节点内 SP 为边界。后续要补齐：

- worker 注册协议：每台机器向 global scheduler 汇报 `node_id/host/local_ranks/health`。
- 节点级资源池：新请求先选择一个节点，后续 DiT/VAE 只在该节点内扩缩容。
- 多机 phase-aware worker-pool：把当前单机 per-rank command wave 扩展为 per-node command wave，并继续禁止跨节点 SP。
- 故障处理：worker 断连、请求失败、rank 释放和日志补全。
- 多机实验脚本：启动 head scheduler 和多个 node worker，汇总各节点 CSV/JSONL。
- 验收标准：同一请求的 DiT/VAE ranks 必须来自同一节点；不同请求可以分布在不同节点。
