# SGLang Diffusion DDiT 实验 CLI 说明

本文给出本轮 DDiT 开发的闭包命令：forced switch 正确性验证，以及 fixed baseline / hungry_first / naive / naive_greedy / wsjf / wsjf_scale_up 的 E2E 实验。所有命令都使用 `encoder + ddit_worker` two-instance launcher，这样 text encoder、transfer buffer、ddit_worker dynamic SP、DiT/VAE 和 output path 都会被覆盖。

统一占位：

- `<MODEL_PATH>`：Wan2.1 T2V 1.3B 或当前实验模型路径。
- `<PROFILE_JSON>`：真实 profile JSON，格式见 `Profile Data`。
- `<LOG_DIR>`：实验日志目录，例如 `C:\Users\woshi\Desktop\ddit_logs\hungry_first`。
- 示例以单机 8 卡为例；full ranks 由 `--num-gpus` 和 `--gpu-ids` 决定，不写死 8。
- disagg warmup 使用 `WxH`，例如 `1280x720`，不是 `720p` 这类 resolution key。

## Common Buffer Settings

下列参数在每个 server 命令里都显式给出，因为它们直接影响 encoder -> ddit_worker 的 prepared payload admission 和 transfer buffer sizing：

- `--disagg-transfer-backend auto`
- `--disagg-transfer-pin-memory auto`
- `--disagg-transfer-pool-size 536870912`
- `--disagg-transfer-redundancy 1.25`
- `--disagg-warmup`
- `--disagg-warmup-resolutions 1280x720`
- `--disagg-warmup-steps 1`

`536870912` 是 512MiB configured lower bound。启动后的 disagg calibration warmup 会测量实际 prepared payload，并按 `round_allocation_size(measured_payload_bytes) * disagg_max_slots_per_instance * redundancy` 自动放大实际 transfer buffer。mixed workload 如果包含比 720p 更大的请求，应把 `--disagg-warmup-resolutions` 设成 workload 中最大 payload 的尺寸。

## Forced Switch Correctness

用途：单请求全路径正确性验证。默认 50 denoising steps，在 completed step 15/30/45 后触发 `1->2->4->8` 的 DiT SP 扩容，VAE 默认 1 卡。

Server:

```powershell
python scripts\launch_ddit_disagg_wan_t2v.py `
  --model-path <MODEL_PATH> `
  --host 127.0.0.1 `
  --port 30000 `
  --scheduler-port 5555 `
  --num-gpus 8 `
  --gpu-ids 0,1,2,3,4,5,6,7 `
  --ddit-schedule-policy forced_switch `
  --ddit-profile-model-id wan2.1-t2v-1.3b `
  --ddit-sp-degree-map shortpath `
  --ddit-log-dir <LOG_DIR> `
  --disagg-max-slots-per-instance 1 `
  --disagg-transfer-backend auto `
  --disagg-transfer-pool-size 536870912 `
  --disagg-transfer-redundancy 1.25 `
  --disagg-transfer-pin-memory auto `
  --disagg-warmup `
  --disagg-warmup-resolutions 1280x720 `
  --disagg-warmup-steps 1
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

验收点：`ddit_rank_switch.jsonl` 出现 step 15/30/45 的 `stage="dit"` switch 和 `stage="vae"` event；`ddit_lifecycle.csv` 中该 request 有 add、dit_start、dit_end、vae_start、vae_end；warmup request 不进入 DDiT 实验日志。

## Fixed Baseline E2E

语义：text encoder full ranks；DiT/VAE 使用同一组固定 k ranks；VAE 结束才释放 ranks。

Server:

```powershell
python scripts\launch_ddit_disagg_wan_t2v.py `
  --model-path <MODEL_PATH> `
  --host 127.0.0.1 `
  --port 30000 `
  --scheduler-port 5555 `
  --num-gpus 8 `
  --gpu-ids 0,1,2,3,4,5,6,7 `
  --ddit-schedule-policy fixed_baseline `
  --ddit-baseline-gpus 4 `
  --ddit-profile-model-id wan2.1-t2v-1.3b `
  --ddit-sp-degree-map shortpath `
  --ddit-log-dir <LOG_DIR> `
  --disagg-max-slots-per-instance 1 `
  --disagg-transfer-backend auto `
  --disagg-transfer-pool-size 536870912 `
  --disagg-transfer-redundancy 1.25 `
  --disagg-transfer-pin-memory auto `
  --disagg-warmup `
  --disagg-warmup-resolutions 1280x720 `
  --disagg-warmup-steps 1
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

验收点：两个 request 可分别在 `[0,1,2,3]` 和 `[4,5,6,7]` 上并发推进；VAE 与 DiT 使用同一组 ranks；VAE reason 为 `fixed_baseline_same_ranks`。

## Hungry First E2E

语义：waiting request 按 opt GPU 或可用 power-of-two ranks 启动；running request 按 starvation score 尝试扩容；VAE 可与 DiT rank 分离。

Server:

```powershell
python scripts\launch_ddit_disagg_wan_t2v.py `
  --model-path <MODEL_PATH> `
  --host 127.0.0.1 `
  --port 30000 `
  --scheduler-port 5555 `
  --num-gpus 8 `
  --gpu-ids 0,1,2,3,4,5,6,7 `
  --ddit-schedule-policy hungry_first `
  --ddit-profile-path <PROFILE_JSON> `
  --ddit-profile-model-id wan2.1-t2v-1.3b `
  --ddit-sp-degree-map shortpath `
  --ddit-log-dir <LOG_DIR> `
  --disagg-max-slots-per-instance 1 `
  --disagg-transfer-backend auto `
  --disagg-transfer-pool-size 536870912 `
  --disagg-transfer-redundancy 1.25 `
  --disagg-transfer-pin-memory auto `
  --disagg-warmup `
  --disagg-warmup-resolutions 1280x720 `
  --disagg-warmup-steps 1
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

验收点：rank switch 中出现 `reason="waiting_queue"`、`reason="hungry_first"`、`reason="dit_to_vae"`；op trace 中同一 `wave_id` 可出现不同 request 的 rank-disjoint `dit_step`。

## Naive E2E

语义：FCFS head-of-line；队首 request 必须拿到 profile `opt_gpus_num` 才启动；启动后 DiT/VAE ranks 固定。

Server:

```powershell
python scripts\launch_ddit_disagg_wan_t2v.py `
  --model-path <MODEL_PATH> `
  --host 127.0.0.1 `
  --port 30000 `
  --scheduler-port 5555 `
  --num-gpus 8 `
  --gpu-ids 0,1,2,3,4,5,6,7 `
  --ddit-schedule-policy naive `
  --ddit-profile-path <PROFILE_JSON> `
  --ddit-profile-model-id wan2.1-t2v-1.3b `
  --ddit-sp-degree-map shortpath `
  --ddit-log-dir <LOG_DIR> `
  --disagg-max-slots-per-instance 1 `
  --disagg-transfer-backend auto `
  --disagg-transfer-pool-size 536870912 `
  --disagg-transfer-redundancy 1.25 `
  --disagg-transfer-pin-memory auto `
  --disagg-warmup `
  --disagg-warmup-resolutions 1280x720 `
  --disagg-warmup-steps 1
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

验收点：资源不足时不跳过队首 request；VAE reason 为 `naive_same_ranks`。

## Naive Greedy E2E

语义：FCFS head-of-line；若 free ranks 少于 opt k，则降级到 `floor_power_of_two(min(opt_k, free_count))`；启动后 DiT/VAE ranks 固定。

Server:

```powershell
python scripts\launch_ddit_disagg_wan_t2v.py `
  --model-path <MODEL_PATH> `
  --host 127.0.0.1 `
  --port 30000 `
  --scheduler-port 5555 `
  --num-gpus 8 `
  --gpu-ids 0,1,2,3,4,5,6,7 `
  --ddit-schedule-policy naive_greedy `
  --ddit-profile-path <PROFILE_JSON> `
  --ddit-profile-model-id wan2.1-t2v-1.3b `
  --ddit-sp-degree-map shortpath `
  --ddit-log-dir <LOG_DIR> `
  --disagg-max-slots-per-instance 1 `
  --disagg-transfer-backend auto `
  --disagg-transfer-pool-size 536870912 `
  --disagg-transfer-redundancy 1.25 `
  --disagg-transfer-pin-memory auto `
  --disagg-warmup `
  --disagg-warmup-resolutions 1280x720 `
  --disagg-warmup-steps 1
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

验收点：资源不足时可降级启动；VAE reason 为 `naive_greedy_same_ranks`。

## WSJF E2E

语义：维护 prepared waiting window，窗口内按 profile estimated remaining DiT time 从短到长选择；VAE same ranks。

Server:

```powershell
python scripts\launch_ddit_disagg_wan_t2v.py `
  --model-path <MODEL_PATH> `
  --host 127.0.0.1 `
  --port 30000 `
  --scheduler-port 5555 `
  --num-gpus 8 `
  --gpu-ids 0,1,2,3,4,5,6,7 `
  --ddit-schedule-policy wsjf `
  --ddit-profile-path <PROFILE_JSON> `
  --ddit-profile-model-id wan2.1-t2v-1.3b `
  --ddit-sp-degree-map shortpath `
  --ddit-window-size 8 `
  --ddit-log-dir <LOG_DIR> `
  --disagg-max-slots-per-instance 8 `
  --disagg-transfer-backend auto `
  --disagg-transfer-pool-size 536870912 `
  --disagg-transfer-redundancy 1.25 `
  --disagg-transfer-pin-memory auto `
  --disagg-warmup `
  --disagg-warmup-resolutions 1280x720 `
  --disagg-warmup-steps 1
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

验收点：prepared waiting window 内短作业优先；VAE reason 为 `wsjf_same_ranks`。

## WSJF Scale-Up E2E

语义：waiting 选择与 WSJF 一样；running DiT request 先按 hungry-first starvation score 尝试扩容，再启动 waiting request；VAE same ranks。

Server:

```powershell
python scripts\launch_ddit_disagg_wan_t2v.py `
  --model-path <MODEL_PATH> `
  --host 127.0.0.1 `
  --port 30000 `
  --scheduler-port 5555 `
  --num-gpus 8 `
  --gpu-ids 0,1,2,3,4,5,6,7 `
  --ddit-schedule-policy wsjf_scale_up `
  --ddit-profile-path <PROFILE_JSON> `
  --ddit-profile-model-id wan2.1-t2v-1.3b `
  --ddit-sp-degree-map shortpath `
  --ddit-window-size 8 `
  --ddit-log-dir <LOG_DIR> `
  --disagg-max-slots-per-instance 8 `
  --disagg-transfer-backend auto `
  --disagg-transfer-pool-size 536870912 `
  --disagg-transfer-redundancy 1.25 `
  --disagg-transfer-pin-memory auto `
  --disagg-warmup `
  --disagg-warmup-resolutions 1280x720 `
  --disagg-warmup-steps 1
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

验收点：running scale-up 优先于 waiting start；扩容 event 的 `old_ranks` 非空且 reason 为 `wsjf_scale_up`；VAE reason 为 `wsjf_scale_up_same_ranks`。

## Profile Data

`--ddit-profile-path` 优先级高于内置 placeholder。传入 multi-model profile 时，用 `--ddit-profile-model-id` 选择模型。当前内置 placeholder 支持 `z-image` 和 `wan2.1-t2v-1.3b`；真实实验建议显式传 `<PROFILE_JSON>`。

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
      "opt_gpus_num": {"144p": 1, "360p": 2, "720p": 4},
      "dit_step_times": {
        "144p": {"1": 1.0, "2": 0.9, "4": 1.0, "8": 1.2},
        "360p": {"1": 4.0, "2": 2.1, "4": 1.8, "8": 1.9},
        "720p": {"1": 16.0, "2": 8.2, "4": 4.4, "8": 4.8}
      }
    },
    "z-image": {
      "opt_gpus_num": {"512p": 2, "1024p": 4},
      "dit_step_times": {
        "512p": {"1": 5.0, "2": 2.8, "4": 2.2, "8": 2.4},
        "1024p": {"1": 16.0, "2": 8.4, "4": 4.9, "8": 5.2}
      }
    }
  }
}
```

## Shortpath SP Degree

`--ddit-sp-degree-map shortpath` 是 DDiT-only 的内置 degree 表，不读取 attention head 数，也不做运行时猜测；它只按 `--ddit-profile-model-id` / `--model-id` / `--model-path` 推断出的 model id 和当前 rank 数查表。当前支持：

| Model id | 1 rank | 2 ranks | 4 ranks | 8 ranks |
| --- | --- | --- | --- | --- |
| `wan2.1-t2v-1.3b` | `1x1` | `2x1` | `4x1` | `2x4` |
| `z-image` | `1x1` | `2x1` | `2x2` | `2x4` |

如果 model id 不是上述两个，或者当前 rank 数不在表内，server 会在启动校验或 dynamic SP group 创建时直接报错。普通 map 语法仍然保留，例如 `--ddit-sp-degree-map "1=1x1,2=2x1,4=2x2,8=2x4"`。

## Logs

Lifecycle CSV：`<LOG_DIR>\ddit_lifecycle.csv`

列：`request_id,resolution,add_time,dit_start_time,dit_end_time,vae_start_time,vae_end_time,status,error`

常用指标：

- queue latency：`dit_start_time - add_time`
- DiT latency：`dit_end_time - dit_start_time`
- VAE latency：`vae_end_time - vae_start_time`
- E2E latency：`vae_end_time - add_time`
- CSV 最后三行是 `p50,<seconds>`、`p90,<seconds>`、`p99,<seconds>`，统计 completed requests 的 lifespan time。

Rank switch JSONL：`<LOG_DIR>\ddit_rank_switch.jsonl`

字段：`timestamp,request_id,resolution,node_id,stage,step,old_ranks,new_ranks,reason,policy`

Op trace JSONL：`<LOG_DIR>\ddit_op_trace.jsonl`

字段：`timestamp_start,timestamp_end,request_id,stage,step,ranks,wave_id,op_id,rank,action,status,error`

真并发验收：同一 `wave_id` 中出现不同 `request_id` 的 `dit_step` 或 `vae_run`，这些 op 的 `ranks` 不重叠，且时间区间重叠。

## Args Appendix

| Args | 含义 |
| --- | --- |
| `--enable-ddit` | 启用 DDiT dynamic SP、并发 runtime、VAE rank 控制和实验日志。 |
| `--ddit-schedule-policy` | 调度策略：`forced_switch`、`hungry_first`、`fixed_baseline`、`naive`、`naive_greedy`、`wsjf`、`wsjf_scale_up`。 |
| `--ddit-baseline-gpus` | `fixed_baseline` 下每个请求 DiT/VAE 固定占用的 GPU 数。 |
| `--ddit-window-size` | `wsjf` 和 `wsjf_scale_up` 的 prepared waiting window 大小。 |
| `--ddit-profile-path` | DDiT profile JSON 路径；优先级高于内置 placeholder profile。 |
| `--ddit-profile-model-id` | multi-model profile 或内置 profile 的 model id，当前支持 `z-image`、`wan2.1-t2v-1.3b`。 |
| `--ddit-allowed-gpu-counts` | DDiT 允许分配的 GPU 数，默认 `1,2,4,8`，会按当前 `--num-gpus` 自动裁剪。 |
| `--ddit-log-dir` | DDiT 日志输出目录，包含 lifecycle CSV、rank switch JSONL 和 op trace JSONL。 |
| `--ddit-vae-gpus` | `hungry_first` / forced path 的默认 VAE GPU 数；profile-backed 策略和 fixed baseline 中 VAE same ranks，不使用该值。 |
| `--ddit-initial-gpus` | forced-switch 默认 DiT 起始 GPU 数；请求级 ranks 或 forced 脚本 `--initial-ranks` 会覆盖它。 |
| `--ddit-initial-ranks` | forced-switch 默认 DiT 起始 ranks；通常由 request/脚本侧指定。 |
| `--ddit-switch-plan` | forced-switch 默认 step 切换计划，例如 `15:1->2;30:2->4;45:4->8`；通常由 request/脚本侧指定。 |
| `--ddit-local-ranks` | 当前节点可用于 DiT/VAE 的 rank 列表；不传时使用当前 instance 的所有 local ranks。 |
| `--ddit-node-id` | 当前节点 id，写入 rank switch 日志；单机默认 `node0`。 |
| `--ddit-sp-degree-map` | 为不同 GPU 数配置 Ulysses/Ring degree，例如 `1=1x1,2=2x1,4=2x2,8=4x2`；也可设为 `shortpath`，对 `wan2.1-t2v-1.3b` 使用 `1x1/2x1/4x1/2x4`，对 `z-image` 使用 `1x1/2x1/2x2/2x4`。 |
| `--ddit-prebuild-sp-groups` | 是否启动时预建 dynamic SP process groups。 |
| `--ddit-debug-cpu-backup` | debug 开关；开启后额外保留 CPU 侧状态备份，用于排查 latent/state 迁移。 |
| `--disagg-role ddit_worker` | 启动 two-instance DDiT 的 DiT+VAE worker role；该 role 不加载 text/image encoder。 |
| `--ddit-worker-urls` | DiffusionServer/head 连接 DDiT worker pool 的 work endpoint 列表。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --model-path` | one-shot launcher 的模型路径。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --host` | head、encoder、ddit_worker 使用的 host。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --port` | HTTP server 端口。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --scheduler-port` | DiffusionServer frontend/control 基准端口；encoder work 使用 `+10`，ddit_worker work 使用 `+20`。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --num-gpus` | encoder 和 ddit_worker instance 的 local world size。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --gpu-ids` | encoder 和 ddit_worker 复用的物理 GPU ids，例如 `0,1,2,3,4,5,6,7`。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --encoder-tp` | encoder instance 的 text encoder TP degree；不传时等于 GPU 数。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --ddit-worker-sp` | ddit_worker full SP degree；不传时等于 GPU 数。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --ddit-worker-ulysses` | ddit_worker full Ulysses degree；不传时等于 worker SP degree。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --ddit-worker-ring` | ddit_worker full Ring degree；默认 `1`。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --ddit-sp-degree-map` | 透传到 DDiT dynamic SP resolver；Wan/Z-image 实验推荐 `shortpath`。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --disagg-max-slots-per-instance` | 每个 encoder/ddit_worker instance 的 prepared payload admission slots；forced/fixed/naive/greedy/hungry 推荐 `1`，WSJF/Scale-Up 推荐等于 window size。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --disagg-transfer-backend` | encoder 到 ddit_worker 的 transfer backend，默认 `auto`。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --disagg-transfer-pool-size` | transfer data buffer configured lower bound，单位 bytes；模板用 `536870912`。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --disagg-transfer-redundancy` | warmup measured payload 自动扩容时使用的冗余系数；模板用 `1.25`。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --disagg-transfer-pin-memory` | transfer buffer 是否使用 pinned host memory：`auto`、`off`、`required`。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --disagg-warmup` | 启动后通过 head 发送 disagg calibration request，用于测量 transfer payload 并触发 buffer resize；不是普通实验请求。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --disagg-warmup-resolutions` | disagg calibration warmup 尺寸，逗号分隔 `WxH`，例如 `1280x720`。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --disagg-warmup-steps` | disagg calibration warmup 的 denoising steps；模板用 `1`，只为测 buffer。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --disagg-timeout` | disagg request 总 timeout 秒数。 |
| `scripts\launch_ddit_disagg_wan_t2v.py --disagg-downstream-wait-timeout` | 等待下游 worker credit/slot 的 timeout 秒数。 |
| forced script `--server-url` | SGLang Diffusion server 地址。 |
| forced script `--request-id` | 当前 request id，写入 lifecycle、rank switch 和 op trace。 |
| forced script `--size` | 视频尺寸，例如 `1280x720`。 |
| forced script `--resolution-key` | profile/log 使用的分辨率标签，例如 `720p`。 |
| forced script `--num-inference-steps` | denoising steps 数；默认 50。 |
| forced script `--initial-ranks` | forced-switch DiT 初始 ranks，例如 `0` 或 `0,1`。 |
| forced script `--switch-plan` | forced-switch step 切换计划，例如 `15:1->2;30:2->4;45:4->8`。 |
| forced script `--ddit-vae-k` | forced correctness request 的 VAE GPU 数；默认 1。 |
| forced script `--ddit-vae-ranks` | forced correctness request 的显式 VAE ranks；优先级高于 `--ddit-vae-k`。 |
| mixed client `--num-requests` | 总请求数。 |
| mixed client `--resolutions` | 分辨率标签列表，例如 `144p,360p,720p`。 |
| mixed client `--ratios` | 各分辨率比例，float 列表且总和为 1，例如 `0.5,0.25,0.25`。 |
| mixed client `--rate` | 请求发送速率；可为数值 requests/s，也可为 `burst`。 |
| mixed client `--seed` | workload shuffle 和 request id 生成的随机种子。 |
| mixed client `--image-path` | TI2V / image 模型输入图片路径；相对路径按 project root 解析。 |
| mixed client `--project-root` | 显式指定项目根目录；未传时脚本从自身位置自动探测。 |
| mixed client `--size-map-json` | 覆盖 resolution 到 size 的映射 JSON，例如 `{"720p":"1280x720"}`。 |
| mixed client `--extra-json` | 合并到每个 request payload 的额外 JSON 字段。 |
| mixed client `--dry-run` | 只打印将要发送的 payload，不实际请求 server。 |
