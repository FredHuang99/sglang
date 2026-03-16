# Wan2.2-TI2V-5B DiT TP/FSDP/Offload Profiling

This note is intentionally separate from `wan2_2_ti2v_5b_profiling.md`.

Use this file only for DiT parallel-mode profiling:

- TP
- FSDP
- `dit-cpu-offload`
- `dit-layerwise-offload`

It uses a separate helper and a separate output directory so it does not interfere with the SP-focused workflow.

## Scope

Fixed workload:

- model: `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
- task: T2V path only
- image input: disabled
- resolution: `704x1280`
- frames: `121`
- inference steps: `50`

What is measured:

- `DenoisingStage` wall time
- per-denoising-step timing summary
- peak CUDA memory

What is not measured here:

- DiT SP
- encoder
- VAE

## Helper

Use the standalone helper:

```bash
scripts/playground/profile_wan_ti2v_dit_parallel.py
```

Supported modes:

- `tp`
- `fsdp`
- `dit-cpu-offload`
- `dit-layerwise-offload`

The helper writes:

- stdout summary
- JSON report
- TXT report

## Output Directory

Use a separate output root from the SP playbook:

```bash
export MODEL="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
export PROMPT="A cinematic science-fiction city with reflective rain streets."
export DIT_PAR_OUT="./wan22_ti2v5b_dit_parallel_modes"
mkdir -p "${DIT_PAR_OUT}"
```

## Quick Notes Before Running

### TP

- This helper profiles pure TP only.
- It requires `num_gpus == tp_size`.
- For Wan2.2-TI2V-5B, the legal TP sizes on an 8-GPU box are `1, 2, 4, 8`.

You can confirm that with:

```bash
python scripts/playground/profile_wan_ti2v_dit_parallel.py \
  --model-path "${MODEL}" \
  --mode tp \
  --num-gpus 8 \
  --list-legal-tp-sizes
```

### FSDP

- This helper profiles FSDP without SP and without TP.
- Default topology is `hsdp_replicate_dim=1`, `hsdp_shard_dim=num_gpus`.

### `dit-cpu-offload`

- This is treated here as a 1-GPU mode.
- It is not pure CPU compute.
- The model is loaded/offloaded through the DiT CPU-offload path, but execution still happens on GPU.

### `dit-layerwise-offload`

- This is also treated here as a 1-GPU mode.
- It supports async H2D prefetch via `--dit-offload-prefetch-size`.
- `0.0` means "prefetch 1 layer", which is the lowest-memory default.

## Recommended Run Order

If you want to split work by day:

Today:

- TP: `1 / 2 / 4 / 8` GPUs

Tomorrow:

- FSDP: `2 / 4 / 8` GPUs
- `dit-cpu-offload`: `1` GPU
- `dit-layerwise-offload`: `1` GPU

## Common Flags

All commands below use:

- `--sync-stage-profiling all`
- `--warmup-iters 1`
- `--profile-iters 3`

## TP Matrix

### TP on 1 GPU

```bash
python scripts/playground/profile_wan_ti2v_dit_parallel.py \
  --model-path "${MODEL}" \
  --mode tp \
  --num-gpus 1 \
  --tp-size 1 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${DIT_PAR_OUT}/tp/gpu1/profile.json" \
  --output-txt "${DIT_PAR_OUT}/tp/gpu1/profile.txt"
```

### TP on 2 GPUs

```bash
torchrun --standalone --nproc_per_node 2 scripts/playground/profile_wan_ti2v_dit_parallel.py \
  --model-path "${MODEL}" \
  --mode tp \
  --num-gpus 2 \
  --tp-size 2 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${DIT_PAR_OUT}/tp/gpu2/profile.json" \
  --output-txt "${DIT_PAR_OUT}/tp/gpu2/profile.txt"
```

### TP on 4 GPUs

```bash
torchrun --standalone --nproc_per_node 4 scripts/playground/profile_wan_ti2v_dit_parallel.py \
  --model-path "${MODEL}" \
  --mode tp \
  --num-gpus 4 \
  --tp-size 4 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${DIT_PAR_OUT}/tp/gpu4/profile.json" \
  --output-txt "${DIT_PAR_OUT}/tp/gpu4/profile.txt"
```

### TP on 8 GPUs

```bash
torchrun --standalone --nproc_per_node 8 scripts/playground/profile_wan_ti2v_dit_parallel.py \
  --model-path "${MODEL}" \
  --mode tp \
  --num-gpus 8 \
  --tp-size 8 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${DIT_PAR_OUT}/tp/gpu8/profile.json" \
  --output-txt "${DIT_PAR_OUT}/tp/gpu8/profile.txt"
```

## FSDP Matrix

### FSDP on 2 GPUs

```bash
torchrun --standalone --nproc_per_node 2 scripts/playground/profile_wan_ti2v_dit_parallel.py \
  --model-path "${MODEL}" \
  --mode fsdp \
  --num-gpus 2 \
  --hsdp-replicate-dim 1 \
  --hsdp-shard-dim 2 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${DIT_PAR_OUT}/fsdp/gpu2/profile.json" \
  --output-txt "${DIT_PAR_OUT}/fsdp/gpu2/profile.txt"
```

### FSDP on 4 GPUs

```bash
torchrun --standalone --nproc_per_node 4 scripts/playground/profile_wan_ti2v_dit_parallel.py \
  --model-path "${MODEL}" \
  --mode fsdp \
  --num-gpus 4 \
  --hsdp-replicate-dim 1 \
  --hsdp-shard-dim 4 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${DIT_PAR_OUT}/fsdp/gpu4/profile.json" \
  --output-txt "${DIT_PAR_OUT}/fsdp/gpu4/profile.txt"
```

### FSDP on 8 GPUs

```bash
torchrun --standalone --nproc_per_node 8 scripts/playground/profile_wan_ti2v_dit_parallel.py \
  --model-path "${MODEL}" \
  --mode fsdp \
  --num-gpus 8 \
  --hsdp-replicate-dim 1 \
  --hsdp-shard-dim 8 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${DIT_PAR_OUT}/fsdp/gpu8/profile.json" \
  --output-txt "${DIT_PAR_OUT}/fsdp/gpu8/profile.txt"
```

## `dit-cpu-offload` Matrix

### `dit-cpu-offload` on 1 GPU

```bash
python scripts/playground/profile_wan_ti2v_dit_parallel.py \
  --model-path "${MODEL}" \
  --mode dit-cpu-offload \
  --num-gpus 1 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${DIT_PAR_OUT}/dit_cpu_offload/gpu1/profile.json" \
  --output-txt "${DIT_PAR_OUT}/dit_cpu_offload/gpu1/profile.txt"
```

## `dit-layerwise-offload` Matrix

### `dit-layerwise-offload` on 1 GPU, lowest-memory setting

`0.0` means prefetch 1 layer.

```bash
python scripts/playground/profile_wan_ti2v_dit_parallel.py \
  --model-path "${MODEL}" \
  --mode dit-layerwise-offload \
  --num-gpus 1 \
  --dit-offload-prefetch-size 0.0 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${DIT_PAR_OUT}/dit_layerwise_offload/gpu1_prefetch0/profile.json" \
  --output-txt "${DIT_PAR_OUT}/dit_layerwise_offload/gpu1_prefetch0/profile.txt"
```

### `dit-layerwise-offload` on 1 GPU, moderate prefetch

`0.25` means roughly one quarter of layers are prefetched ahead.

```bash
python scripts/playground/profile_wan_ti2v_dit_parallel.py \
  --model-path "${MODEL}" \
  --mode dit-layerwise-offload \
  --num-gpus 1 \
  --dit-offload-prefetch-size 0.25 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${DIT_PAR_OUT}/dit_layerwise_offload/gpu1_prefetch025/profile.json" \
  --output-txt "${DIT_PAR_OUT}/dit_layerwise_offload/gpu1_prefetch025/profile.txt"
```

## Output Interpretation

The helper JSON reports:

- `mode`
- `parallelism`
- `offload`
- `iterations_ms`
- aggregated `timing_ms`
- `denoise_steps_ms`
- `peak_memory_mb`

For TP runs it also reports:

- `legal_tp_sizes`

## Practical Advice

If you want the fastest path to a first result today:

1. Run TP `1 / 2 / 4 / 8`.
2. Skip FSDP and offload until tomorrow.

If you want the lowest-risk order tomorrow:

1. FSDP `2`, then `4`, then `8`.
2. `dit-cpu-offload` `1` GPU.
3. `dit-layerwise-offload` `1` GPU with `0.0`, then optionally `0.25`.
