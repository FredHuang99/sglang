# SGLang Diffusion DDiT 实验 CLI 说明

本文档说明单机 8 卡 DDiT/FlexDiT 风格实验的启动方式、正确性实验、mixed workload E2E 实验，以及日志怎么看。

## 1. 启动服务

示例：

```powershell
sglang serve `
  --model-path <WAN2.1_MODEL_PATH> `
  --num-gpus 8 `
  --sp-degree 8 `
  --ulysses-degree 8 `
  --ring-degree 1 `
  --enable-ddit `
  --ddit-node-id node0 `
  --ddit-local-ranks 0,1,2,3,4,5,6,7 `
  --ddit-allowed-gpu-counts 1,2,4,8 `
  --ddit-initial-gpus 1 `
  --ddit-vae-gpus 1 `
  --ddit-log-dir C:\Users\woshi\Desktop\ddit_logs
```

关键参数：

- `--enable-ddit`：启用 DDiT rank 切换、VAE k 卡控制和实验日志。
- `--ddit-node-id`：写入 rank switch JSONL 的节点名。
- `--ddit-local-ranks`：本节点可用于 DDiT 的 global ranks。单机 8 卡通常是 `0,1,2,3,4,5,6,7`。
- `--ddit-allowed-gpu-counts`：允许的 SP 卡数，默认 `1,2,4,8`。
- `--ddit-initial-gpus`：请求进入 DiT 时默认使用几张卡，默认 `1`。
- `--ddit-switch-plan`：服务级默认 forced switch 计划，例如 `15:1->2;30:2->4;45:4->8`。
- `--ddit-vae-gpus`：VAE 默认卡数，默认 `1`，支持 `1/2/4/8`。
- `--ddit-sp-degree-map`：可选的 Ulysses/Ring 映射，例如 `1=1x1,2=2x1,4=2x2,8=4x2`。不设置时默认 Ulysses-only。
- `--ddit-log-dir`：输出 `ddit_lifecycle.csv` 和 `ddit_rank_switch.jsonl` 的目录。

请求级参数优先级：

- `ddit_vae_ranks` 优先级最高，显式指定 VAE ranks。
- `ddit_vae_k` 次之，指定 VAE 使用几张卡。
- 否则使用服务级 `--ddit-vae-gpus`，默认 `1`。

## 2. Forced Switch 正确性实验

脚本路径：

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

参数说明：

- `--num-inference-steps 50`：强制 50 个 denoising steps，保证 15/30/45 都有效。
- `--initial-ranks 0`：DiT 起始只用 rank 0。
- `--switch-plan`：`after_step:old_k->new_k` 或 `after_step:rank,rank,...` 格式。
- `--ddit-vae-k`：VAE 使用几张卡；可改为 `2` 或 `4` 验证多卡 VAE。
- `--ddit-vae-ranks`：可选，例如 `0,1`，显式覆盖 `ddit_vae_k`。

验收点：

- `ddit_rank_switch.jsonl` 中能看到 step 15、30、45 的 DiT rank 扩容。
- `ddit_rank_switch.jsonl` 中能看到 `stage=vae` 的 DiT->VAE rank 切换。
- `ddit_lifecycle.csv` 中该请求有完整的 add/DiT/VAE 时间戳。
- 服务端不应重新初始化 latent；扩容时使用当前 latent gather/broadcast。

## 3. Mixed Workload E2E 实验

脚本路径：

```powershell
python examples\multimodal_gen\ddit_mixed_workload_client.py `
  --server-url http://127.0.0.1:30000 `
  --num-requests 128 `
  --resolutions "144p,720p" `
  --ratios "0.75,0.25" `
  --rate 1 `
  --seed 42
```

请求生成逻辑：

- `--num-requests x`：总请求数。
- `--resolutions "144p,360p,240p"`：一共有 n 种分辨率。
- `--ratios "0.5,0.3,0.2"`：比例总和必须为 1。
- 前 `n-1` 类数量为 `round(x * ratio_i)`。
- 最后一类数量为 `x - sum(previous_counts)`，保证总数严格等于 `x`。
- 生成后用 `--seed` 固定 shuffle，实验可复现。

发送速率：

- `--rate 1`：每秒 1 个请求。
- `--rate 0.2`：每 5 秒 1 个请求。
- `--rate 2`：每秒 2 个请求。
- `--rate 1.5`：每秒 1.5 个请求。
- `--rate burst`：循环内不 sleep，模拟同时到达。

分辨率到 size 的默认映射：

- `144p -> 256x144`
- `240p -> 426x240`
- `360p -> 640x360`
- `480p -> 854x480`
- `720p -> 1280x720`
- `1080p -> 1920x1080`

可以用 `--size-map-json '{"144p":"256x144","720p":"1280x720"}'` 覆盖。

## 4. 日志格式

生命周期 CSV：

路径：`<ddit-log-dir>\ddit_lifecycle.csv`

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

- 排队时间：`dit_start_time - add_time`
- DiT 时间：`dit_end_time - dit_start_time`
- VAE 时间：`vae_end_time - vae_start_time`
- E2E 时间：`vae_end_time - add_time`
- CSV 最后三行自动追加 lifespan time 汇总：`p50,<seconds>`、`p90,<seconds>`、`p99,<seconds>`。统计只使用已完成且同时具备 `add_time` 与 `vae_end_time` 的请求。

rank 切换 JSONL：

路径：`<ddit-log-dir>\ddit_rank_switch.jsonl`

字段：

- `timestamp`
- `request_id`
- `resolution`
- `node_id`
- `stage`：`dit` 或 `vae`
- `step`：DiT 切换时为 completed denoising step；VAE 切换时为 DiT 结束 step。
- `old_ranks`
- `new_ranks`
- `reason`
- `policy`

示例含义：

```json
{"stage":"dit","step":30,"old_ranks":[0,1],"new_ranks":[0,1,2,3],"reason":"switch_plan","policy":"forced_switch_plan"}
```

表示该请求在完成第 30 个 denoising step 后，从 2 卡扩到 4 卡。

## 5. 多机后续开发计划

当前第一版保证控制面参数和本节点 rank 约束，但不做跨节点 SP。

后续还需要：

- worker 注册协议：每台机器启动后向 global scheduler 汇报 `node_id/host/local_ranks/health`。
- 节点级资源池：新请求先选择节点，后续 DiT/VAE 只在该节点内扩缩。
- 多节点启动脚本：一键启动 head scheduler 和多个 node workers。
- 跨节点故障处理：节点断连、请求失败、rank 释放、日志补全。
- 多机日志汇总：按 `request_id` 合并各节点 CSV/JSONL。
- 多机验收：同一请求的 DiT/VAE ranks 必须来自同一节点；不同请求可分布在不同节点。
