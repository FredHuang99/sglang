# DIT Cold-Start Weight Loading Cookbook

This document covers the Phase1-Phase5 diffusion cold-start work for Z-Image
style DIT/VAE loading. It focuses on launch-time weight profile, rank0 SP
broadcast, pinned-memory ablation, and the Phase5 pageable host warm pool.

## Concepts

`staging` is not pinned memory by itself. In this codebase it means every
checkpoint tensor passes through a shared intermediate iterator before the
loader consumes it. That layer can count tensors and bytes, collect profile
fields, optionally pin tensors, or feed later features such as rank0 broadcast
and warm pool.

`pageable staging` is normal DRAM staging. Tensors remain in regular pageable
CPU memory. There is no page lock and no extra pin/register cost. Its value is a
stable loading path and profile point that later phases can reuse.

`pinned staging` is an explicit experimental mode. It calls
`tensor.pin_memory()` and stores tensors in page-locked CPU memory before the
H2D copy. This can help when the same pinned buffers are reused many times or
when request-time transfers are frequent. It is not recommended as the default
single cold-start path because pinning 11+ GiB every launch can cost more than
it saves.

`warm pool` is an in-process pageable CPU cache. On a miss, rank0 reads and
materializes CPU tensors as usual, then stores the materialized state dict. On a
hit, rank0 skips safetensors iteration and CPU state-dict construction, then
still performs the final device load and SP broadcast. Phase5 v1 does not share
memory across OS processes.

## Phase Summary

| Phase | Purpose | Main Flag |
| --- | --- | --- |
| Phase1 | Observe per-rank read/materialize/H2D cost | `--profile-enabled` |
| Phase2 | Add rank0 staging iterator and pinned ablation | `--diffusion-weight-staging` |
| Phase3 | Rank0 reads transformer, SP ranks receive via broadcast | `--diffusion-weight-load-mode rank0-broadcast` |
| Phase4 | Extend rank0 broadcast to VAE/decoder | `--diffusion-weight-broadcast-components transformer,vae` |
| Phase5 | Reuse pageable CPU tensors in one process | `--diffusion-weight-warm-pool pageable` |

## Parameter Cookbook

Baseline, no cold-start optimization:

```bash
python3 launch_server.py \
  --model-path /data/Z_Image \
  --model-id Z-Image \
  --profile-enabled \
  --profile-output-dir /data/profile \
  --profile-run-id zimage_sp4_baseline \
  --num-gpus 4 \
  --sp-degree 4 \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --attention-backend fa
```

Recommended Phase4/Phase5 mainline without warm-pool reuse:

```bash
python3 launch_server.py \
  --model-path /data/Z_Image \
  --model-id Z-Image \
  --profile-enabled \
  --profile-output-dir /data/profile \
  --profile-run-id zimage_sp4_pageable_broadcast \
  --diffusion-weight-staging pageable \
  --diffusion-weight-load-mode rank0-broadcast \
  --diffusion-weight-broadcast-components transformer,vae \
  --num-gpus 4 \
  --sp-degree 4 \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --attention-backend fa
```

Pinned ablation only:

```bash
python3 launch_server.py \
  --model-path /data/Z_Image \
  --model-id Z-Image \
  --profile-enabled \
  --profile-output-dir /data/profile \
  --profile-run-id zimage_sp4_pinned_ablation \
  --diffusion-weight-staging pinned \
  --diffusion-weight-load-mode rank0-broadcast \
  --diffusion-weight-broadcast-components transformer,vae \
  --num-gpus 4 \
  --sp-degree 4 \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --attention-backend fa
```

Pageable in-process warm pool:

```bash
python3 launch_server.py \
  --model-path /data/Z_Image \
  --model-id Z-Image \
  --profile-enabled \
  --profile-output-dir /data/profile \
  --profile-run-id zimage_sp4_warm_pool \
  --diffusion-weight-staging pageable \
  --diffusion-weight-load-mode rank0-broadcast \
  --diffusion-weight-broadcast-components transformer,vae \
  --diffusion-weight-warm-pool pageable \
  --diffusion-weight-warm-pool-components transformer,vae \
  --diffusion-weight-warm-pool-max-gb 0 \
  --num-gpus 4 \
  --sp-degree 4 \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --attention-backend fa
```

`--diffusion-weight-staging auto` is accepted for compatibility and resolves to
`pageable`. It no longer attempts pinned staging.

## Profile Files

Each profiled component writes one JSON file per process:

```text
<profile-output-dir>/<profile-run-id>_launch_weight_load/
  weight_load_transformer_rank0_local0_pid12345.json
  weight_load_vae_rank0_local0_pid12345.json
```

Core fields:

```text
component, rank, physical_rank, world_size, sp_rank, tp_rank
status, error, device, mem_kind
weight_load:discover_files_ms
weight_load:read_safetensors_ms
weight_load:cpu_materialize_ms
weight_load:h2d_or_param_copy_ms
weight_load:total_bytes
```

Staging and pinned ablation fields:

```text
weight_load:staging_requested
weight_load:staging_effective
weight_load:staged_tensor_count
weight_load:pin_memory_ms
weight_load:pinned_tensor_count
weight_load:pinned_bytes
weight_load:pin_memory_error
```

Broadcast fields:

```text
weight_load:load_mode_requested
weight_load:load_mode_effective
weight_load:rank0_wait_ms
weight_load:nccl_broadcast_ms
weight_load:broadcast_tensor_count
weight_load:broadcast_bytes
weight_load:broadcast_error
```

Warm-pool fields:

```text
weight_load:warm_pool_requested
weight_load:warm_pool_effective
weight_load:warm_pool_hit
weight_load:warm_pool_store_bytes
weight_load:warm_pool_error
```

Expected rank0-broadcast shape:

| Rank | Read Bytes | Broadcast Bytes | Meaning |
| --- | --- | --- | --- |
| SP rank0 | full component size | full component size | rank0 reads and sends |
| non-rank0 | 0 or near 0 | full component size | rank waits and receives |

## Code Walkthrough

Profiler:

`python/sglang/multimodal_gen/runtime/utils/weight_load_profiler.py`

`DiffusionWeightLoadProfiler` owns the per-component JSON schema. Loader code
only calls timing scopes and setter methods at coarse boundaries, so profile I/O
does not sit inside the hot tensor broadcast loop.

Staging:

`python/sglang/multimodal_gen/runtime/loader/weight_staging.py`

`maybe_stage_weight_iterator(iterator, staging_mode, weight_load_profile)`
returns the original iterator for `none` or non-rank0, and otherwise routes
rank0 tensors through `stage_weight_iterator`. `pageable` records tensor count
and bytes without pinning. `pinned` explicitly attempts `pin_memory()`. `auto`
normalizes to `pageable`.

Transformer load:

`python/sglang/multimodal_gen/runtime/loader/fsdp_load.py`

`maybe_load_fsdp_model` builds a meta-initialized transformer, resolves whether
rank0-broadcast is legal, and chooses one of two paths:

1. Default path: every rank iterates safetensors and loads weights.
2. Broadcast path: rank0 loads real weights, non-rank0 materializes empty
   tensors, then `broadcast_module_tensors` sends parameters and buffers from
   rank0.

`load_model_from_full_model_state_dict` now accepts optional preloaded CPU state
dict data. This is the Phase5 warm-pool hit path; the normal iterator path is
unchanged for misses and for default loading.

VAE load:

`python/sglang/multimodal_gen/runtime/loader/component_loaders/vae_loader.py`

The standard ModelRegistry VAE path uses the same rank0-broadcast control flow:
rank0 loads the state dict, non-rank0 keeps empty tensors, and SP broadcast
fills device tensors. Custom `auto_map` VAE classes keep the native path.

Broadcast helpers:

`python/sglang/multimodal_gen/runtime/loader/weight_broadcast.py`

The control plane uses CPU-safe collectives for entry/status/ready checks. The
data plane uses SP group tensor broadcast. Metadata is compared before large
tensor broadcast to avoid rank divergence. Progress logging is debug-only and
does not write sidecar files.

Warm pool:

`python/sglang/multimodal_gen/runtime/loader/weight_warm_pool.py`

The warm-pool key includes component, component path, safetensors path/size/mtime
fingerprint, dtype, and component class. The global LRU stores CPU tensors only
inside the current process. `--diffusion-weight-warm-pool-max-gb 0` means no
limit; positive values evict old entries.

## Benchmark Script

Use `scripts/benchmark_diffusion_cold_start.py` to run repeated launches and
summarize the JSON profile outputs:

```bash
python3 scripts/benchmark_diffusion_cold_start.py \
  --model-path /data/Z_Image \
  --model-id Z-Image \
  --num-gpus 4 \
  --sp-degree 4 \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --setups baseline,pageable,pinned,warm-pool \
  --repeat-k 3 \
  --profile-output-dir /data/profile \
  --summary-output /data/profile/zimage_cold_start_summary.md \
  --python python3
```

The script writes Markdown and CSV summaries. Rows are setup names. Columns are
`avg` and `max` for launch wall time and valid JSON metrics. Rank-aware metrics
are split into `rank0_*`, `nonrank_avg_*`, and `nonrank_max_*`.

Because Phase5 v1 is in-process only, subprocess repeat runs do not share a
warm pool across OS process boundaries. A warm-pool hit in real flip deployment
requires the loader to be reused inside the same long-lived worker/coordinator
process, or a future shared-memory warm-pool daemon.

## Interpretation

For the current Z-Image profile shape, the recommended production-like path is
pageable staging plus rank0-broadcast for `transformer,vae`. Pinned staging is
kept as an explicit experiment to measure pin/register overhead and to support
future request-time transfer-buffer experiments. It should not be enabled by
default for a one-shot cold start or for launch cadence around one to two
minutes unless the same pinned buffers are reused across many transfers.

Phase5 warm pool can reduce repeated in-process flips by skipping rank0
safetensors iteration and CPU materialization. It cannot remove the final
device load or SP broadcast, and it does not accelerate the very first load.
