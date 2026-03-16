# Wan2.2-TI2V-5B Profiling Playbook

This playbook is for profiling `Wan-AI/Wan2.2-TI2V-5B-Diffusers` on SGLang-Diffusion with the following fixed workload:

- task: T2V path only
- image input: disabled
- prompt: arbitrary text
- resolution: `704x1280`
- frames: `121`
- GPUs: `8x A800 80G NVLink`

The examples below assume:

```bash
export MODEL="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
export PROMPT="A cinematic science-fiction city with reflective rain streets."
export OUT_DIR="./wan22_ti2v5b_profile"
mkdir -p "${OUT_DIR}"
```

## 1. What The Current Code Actually Supports

### Encoder

- Supported GPU parallel mode in the current SGLang path: TP.
- Supported memory-saving mode: `--text-encoder-cpu-offload`.
- Not wired as a separate profiling target: plain "FSDP-only encoder mode" without CPU offload.

What TP means here:

- TP shards encoder weights, not input tokens.
- Each rank still receives the full `input_ids` tensor.
- With the default Wan T5 path, the tokenizer truncates to `512` tokens and the postprocessed embedding shape for a single prompt is `[1, 512, 4096]`.
- That output is logically complete on every rank after the TP collectives.

Concrete example for one prompt:

- raw text input: `prompt`
- tokenized shape: `[1, 512]`
- encoder output after `t5_postprocess_text`: `[1, 512, 4096]`
- on TP=4: weights are sharded, but each rank still ends the encoder with the full `[1, 512, 4096]`

What `--text-encoder-cpu-offload` means:

- It does not mean the encoder computes entirely on CPU.
- It means encoder weights are managed by the FSDP CPU offload path and materialized back to GPU for execution.
- There is no separate explicit layerwise prefetch knob for the text encoder like the DiT layerwise offload path.

How to get pure CPU encoder compute time:

- Do not use `--text-encoder-cpu-offload` as a proxy.
- Use the helper script in this repo with `--stage encoder --device cpu`.

### DiT

- The profiling matrix in this playbook focuses on SP only.
- Codebase features outside the profiling matrix still exist: TP, FSDP-style offload, `--dit-cpu-offload`, and `--dit-layerwise-offload`.

How SP works in the current Wan DiT path:

- The model input is sequence-sharded first.
- With the fixed workload here, the latent shape is `[1, 16, 31, 44, 80]`.
- Wan patchifies with patch size `(1, 2, 2)`, so the DiT token count is `31 * 22 * 40 = 27280`.
- If `27280` is not divisible by `sp_degree`, the model pads sequence tokens before sharding and trims after the final gather.

Ulysses:

- Input to attention starts as sequence-local QKV, conceptually `[B, S_local, H, D]`.
- Ulysses all-to-all changes that to `[B, S_local * U, H / U, D]` inside each Ulysses subgroup.
- So Ulysses is effectively trading sequence shards for head shards.

Ring Attention:

- Ring keeps the head-sharded layout from Ulysses or local attention.
- Attention is then computed across the ring subgroup on the sequence dimension.
- Inside a block the output stays sequence-local.

When Ulysses and Ring are both enabled:

- First split: model input is already sequence-sharded across all `sp_degree` ranks.
- Then attention does Ulysses all-to-all inside each Ulysses subgroup.
- Ring attention runs inside each Ring subgroup.
- After attention, Ulysses all-to-all restores the original sequence-local layout.
- After the last block, the model does a sequence all-gather and restores the full hidden state before output projection and unpatchify.

Are outputs complete?

- Inside the blocks: no, each rank holds sequence-local activations.
- At the end of the DiT forward: yes, the model all-gathers back to the full sequence.

6-GPU support for Wan2.2-TI2V-5B:

- Yes, `sp_degree=6` is possible for this workload.
- Because Wan2.2-TI2V-5B uses `40` attention heads, `ulysses_degree` must divide `40`.
- Legal 6-GPU pairs are:
  - `(ulysses, ring) = (2, 3)`
  - `(ulysses, ring) = (1, 6)`
- Illegal 6-GPU pairs are:
  - `(6, 1)` because `40 % 6 != 0`
  - `(3, 2)` because `40 % 3 != 0`

### VAE

- Supported multi-GPU mode in the current Wan VAE path: SP.
- Supported memory-saving mode: `--vae-cpu-offload`.
- Not supported: VAE TP.

How VAE SP works:

- Wan VAE splits along the height dimension.
- Distributed convolution uses halo exchange to preserve correctness at shard boundaries.
- Encode and decode gather the height dimension back before returning the public output.

Concrete shapes for the fixed workload:

- VAE encode input: `[1, 3, 121, 704, 1280]`
- VAE encode output latent: `[1, 16, 31, 44, 80]`
- VAE decode input latent: `[1, 16, 31, 44, 80]`
- VAE decode output: `[1, 3, 121, 704, 1280]`

What `--vae-cpu-offload` means:

- It does not mean the VAE computes entirely on CPU.
- The current encode/decode stages explicitly move the VAE back to the local CUDA device before compute and move it back to CPU after the stage.
- There is no VAE layerwise prefetch path in the current code.

How to get pure CPU VAE compute time:

- Do not use `--vae-cpu-offload` as a proxy.
- Use the helper script with `--stage vae-encode --device cpu` or `--stage vae-decode --device cpu`.

## 2. What Existing `sglang generate` Profiling Can And Cannot Measure

Use `sglang generate` when you want:

- PyTorch profiler traces via `--profile`
- full-pipeline kernel traces via `--profile --profile-all-stages`
- lightweight JSON timing dumps via `--perf-dump-path`

The repo now supports two useful additions for this path:

- `--perf-dump-path` still writes JSON, and also prints a concise timing summary to stdout.
- `SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=all` can force CUDA synchronization for every profiled stage, which makes stage wall times much closer to what you actually want for benchmarking.

Recommended rule:

- For kernel analysis: use `--profile --profile-all-stages`.
- For human-readable per-stage timing: use `--perf-dump-path` and set `SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=all`.

Limitation:

- `sglang generate` cannot tell you "pure CPU encoder time" or "pure CPU VAE time" from offload flags alone, because those offload flags are still GPU execution paths.

## 3. End-To-End Baseline Command

Use this first to sanity-check the environment and collect the standard SGLang timing dump:

```bash
SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=all \
sglang generate \
  --model-path "${MODEL}" \
  --prompt "${PROMPT}" \
  --height 704 \
  --width 1280 \
  --num-frames 121 \
  --seed 0 \
  --num-inference-steps 50 \
  --profile \
  --profile-all-stages \
  --perf-dump-path "${OUT_DIR}/end_to_end/perf.json"
```

Artifacts:

- torch trace: under the normal SGLang profiler output directory
- stage/step JSON: `${OUT_DIR}/end_to_end/perf.json`
- console summary: printed automatically by the CLI

Important note:

- `stages_ms` in the JSON corresponds to stage class names such as `TextEncodingStage`, `DenoisingStage`, and `DecodingStage`.

## 4. Stage Helper

Use the helper added in this repo when you want direct per-stage timing or pure CPU timing:

```bash
scripts/playground/profile_wan_ti2v_stages.py
```

Features:

- stages: `encoder`, `dit`, `vae-encode`, `vae-decode`
- devices: `cpu`, `cuda`
- outputs: stdout summary + optional JSON + optional TXT
- default workload: exactly the workload described at the top of this page

Common flags used below:

- `--warmup-iters 1`
- `--profile-iters 3`
- `--sync-stage-profiling all` for accurate CUDA wall times

## 5. Encoder Commands

### Encoder On CPU

```bash
python scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage encoder \
  --device cpu \
  --prompt "${PROMPT}" \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/encoder/cpu/profile.json" \
  --output-txt "${OUT_DIR}/encoder/cpu/profile.txt"
```

### Encoder On 1 GPU

```bash
torchrun --standalone --nproc_per_node 1 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage encoder \
  --device cuda \
  --num-gpus 1 \
  --tp-size 1 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/encoder/gpu1/profile.json" \
  --output-txt "${OUT_DIR}/encoder/gpu1/profile.txt"
```

### Encoder On 2 GPUs

```bash
torchrun --standalone --nproc_per_node 2 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage encoder \
  --device cuda \
  --num-gpus 2 \
  --tp-size 2 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/encoder/gpu2/profile.json" \
  --output-txt "${OUT_DIR}/encoder/gpu2/profile.txt"
```

### Encoder On 4 GPUs

```bash
torchrun --standalone --nproc_per_node 4 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage encoder \
  --device cuda \
  --num-gpus 4 \
  --tp-size 4 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/encoder/gpu4/profile.json" \
  --output-txt "${OUT_DIR}/encoder/gpu4/profile.txt"
```

### Encoder On 8 GPUs

```bash
torchrun --standalone --nproc_per_node 8 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage encoder \
  --device cuda \
  --num-gpus 8 \
  --tp-size 8 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/encoder/gpu8/profile.json" \
  --output-txt "${OUT_DIR}/encoder/gpu8/profile.txt"
```

## 6. DiT Commands

### Ask The Helper Which `(ulysses, ring)` Pairs Are Legal

8 GPUs:

```bash
python scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 8 \
  --sp-degree 8 \
  --list-legal-dit-combos
```

6 GPUs:

```bash
python scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 6 \
  --sp-degree 6 \
  --list-legal-dit-combos
```

### DiT On 1 GPU

```bash
torchrun --standalone --nproc_per_node 1 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/dit/gpu1_u1_r1/profile.json" \
  --output-txt "${OUT_DIR}/dit/gpu1_u1_r1/profile.txt"
```

### DiT On 2 GPUs

`(ulysses, ring) = (2, 1)`

```bash
torchrun --standalone --nproc_per_node 2 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/dit/gpu2_u2_r1/profile.json" \
  --output-txt "${OUT_DIR}/dit/gpu2_u2_r1/profile.txt"
```

`(ulysses, ring) = (1, 2)`

```bash
torchrun --standalone --nproc_per_node 2 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 1 \
  --ring-degree 2 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/dit/gpu2_u1_r2/profile.json" \
  --output-txt "${OUT_DIR}/dit/gpu2_u1_r2/profile.txt"
```

### DiT On 4 GPUs

`(ulysses, ring) = (4, 1)`

```bash
torchrun --standalone --nproc_per_node 4 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 4 \
  --sp-degree 4 \
  --ulysses-degree 4 \
  --ring-degree 1 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/dit/gpu4_u4_r1/profile.json" \
  --output-txt "${OUT_DIR}/dit/gpu4_u4_r1/profile.txt"
```

`(ulysses, ring) = (2, 2)`

```bash
torchrun --standalone --nproc_per_node 4 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 4 \
  --sp-degree 4 \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/dit/gpu4_u2_r2/profile.json" \
  --output-txt "${OUT_DIR}/dit/gpu4_u2_r2/profile.txt"
```

`(ulysses, ring) = (1, 4)`

```bash
torchrun --standalone --nproc_per_node 4 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 4 \
  --sp-degree 4 \
  --ulysses-degree 1 \
  --ring-degree 4 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/dit/gpu4_u1_r4/profile.json" \
  --output-txt "${OUT_DIR}/dit/gpu4_u1_r4/profile.txt"
```

### DiT On 6 GPUs

`(ulysses, ring) = (2, 3)`

```bash
torchrun --standalone --nproc_per_node 6 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 6 \
  --sp-degree 6 \
  --ulysses-degree 2 \
  --ring-degree 3 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/dit/gpu6_u2_r3/profile.json" \
  --output-txt "${OUT_DIR}/dit/gpu6_u2_r3/profile.txt"
```

`(ulysses, ring) = (1, 6)`

```bash
torchrun --standalone --nproc_per_node 6 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 6 \
  --sp-degree 6 \
  --ulysses-degree 1 \
  --ring-degree 6 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/dit/gpu6_u1_r6/profile.json" \
  --output-txt "${OUT_DIR}/dit/gpu6_u1_r6/profile.txt"
```

### DiT On 8 GPUs

`(ulysses, ring) = (8, 1)`

```bash
torchrun --standalone --nproc_per_node 8 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 8 \
  --sp-degree 8 \
  --ulysses-degree 8 \
  --ring-degree 1 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/dit/gpu8_u8_r1/profile.json" \
  --output-txt "${OUT_DIR}/dit/gpu8_u8_r1/profile.txt"
```

`(ulysses, ring) = (4, 2)`

```bash
torchrun --standalone --nproc_per_node 8 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 8 \
  --sp-degree 8 \
  --ulysses-degree 4 \
  --ring-degree 2 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/dit/gpu8_u4_r2/profile.json" \
  --output-txt "${OUT_DIR}/dit/gpu8_u4_r2/profile.txt"
```

`(ulysses, ring) = (2, 4)`

```bash
torchrun --standalone --nproc_per_node 8 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 8 \
  --sp-degree 8 \
  --ulysses-degree 2 \
  --ring-degree 4 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/dit/gpu8_u2_r4/profile.json" \
  --output-txt "${OUT_DIR}/dit/gpu8_u2_r4/profile.txt"
```

`(ulysses, ring) = (1, 8)`

```bash
torchrun --standalone --nproc_per_node 8 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 8 \
  --sp-degree 8 \
  --ulysses-degree 1 \
  --ring-degree 8 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/dit/gpu8_u1_r8/profile.json" \
  --output-txt "${OUT_DIR}/dit/gpu8_u1_r8/profile.txt"
```

## 7. VAE Commands

The helper exposes VAE encode and decode separately.

### VAE Encode On CPU

```bash
python scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-encode \
  --device cpu \
  --prompt "${PROMPT}" \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/vae_encode/cpu/profile.json" \
  --output-txt "${OUT_DIR}/vae_encode/cpu/profile.txt"
```

### VAE Decode On CPU

```bash
python scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-decode \
  --device cpu \
  --prompt "${PROMPT}" \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/vae_decode/cpu/profile.json" \
  --output-txt "${OUT_DIR}/vae_decode/cpu/profile.txt"
```

### VAE Encode On 1 GPU

```bash
torchrun --standalone --nproc_per_node 1 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-encode \
  --device cuda \
  --num-gpus 1 \
  --sp-degree 1 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/vae_encode/gpu1/profile.json" \
  --output-txt "${OUT_DIR}/vae_encode/gpu1/profile.txt"
```

### VAE Decode On 1 GPU

```bash
torchrun --standalone --nproc_per_node 1 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-decode \
  --device cuda \
  --num-gpus 1 \
  --sp-degree 1 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/vae_decode/gpu1/profile.json" \
  --output-txt "${OUT_DIR}/vae_decode/gpu1/profile.txt"
```

### VAE Encode On 2 GPUs

```bash
torchrun --standalone --nproc_per_node 2 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-encode \
  --device cuda \
  --num-gpus 2 \
  --sp-degree 2 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/vae_encode/gpu2/profile.json" \
  --output-txt "${OUT_DIR}/vae_encode/gpu2/profile.txt"
```

### VAE Decode On 2 GPUs

```bash
torchrun --standalone --nproc_per_node 2 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-decode \
  --device cuda \
  --num-gpus 2 \
  --sp-degree 2 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/vae_decode/gpu2/profile.json" \
  --output-txt "${OUT_DIR}/vae_decode/gpu2/profile.txt"
```

### VAE Encode On 4 GPUs

```bash
torchrun --standalone --nproc_per_node 4 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-encode \
  --device cuda \
  --num-gpus 4 \
  --sp-degree 4 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/vae_encode/gpu4/profile.json" \
  --output-txt "${OUT_DIR}/vae_encode/gpu4/profile.txt"
```

### VAE Decode On 4 GPUs

```bash
torchrun --standalone --nproc_per_node 4 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-decode \
  --device cuda \
  --num-gpus 4 \
  --sp-degree 4 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/vae_decode/gpu4/profile.json" \
  --output-txt "${OUT_DIR}/vae_decode/gpu4/profile.txt"
```

### VAE Encode On 8 GPUs

```bash
torchrun --standalone --nproc_per_node 8 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-encode \
  --device cuda \
  --num-gpus 8 \
  --sp-degree 8 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/vae_encode/gpu8/profile.json" \
  --output-txt "${OUT_DIR}/vae_encode/gpu8/profile.txt"
```

### VAE Decode On 8 GPUs

```bash
torchrun --standalone --nproc_per_node 8 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-decode \
  --device cuda \
  --num-gpus 8 \
  --sp-degree 8 \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/vae_decode/gpu8/profile.json" \
  --output-txt "${OUT_DIR}/vae_decode/gpu8/profile.txt"
```

## 8. Output Files

### CLI `--perf-dump-path`

The JSON contains:

- `total_duration_ms`
- `steps` for backward compatibility
- `stages_ms` as a readable stage list
- `denoise_steps_ms` with per-step timings

### Stage Helper JSON

The helper writes:

- workload metadata
- parallelism metadata
- per-iteration timings
- aggregated `avg/min/max`
- output shape
- peak CUDA memory when applicable
- legal DiT `(ulysses, ring)` combinations for the chosen `sp_degree`

## 9. Minimal Validation Matrix

If you want a short smoke test before running the full matrix, run:

- encoder CPU
- encoder 1 GPU
- DiT 2 GPUs with `(2,1)`
- DiT 2 GPUs with `(1,2)`
- VAE encode CPU
- VAE decode CPU
- VAE encode 1 GPU
- VAE decode 1 GPU
- one end-to-end `sglang generate --profile --profile-all-stages --perf-dump-path ...`

That is enough to verify:

- the helper can run on CPU and CUDA
- multi-GPU SP initialization works
- ring attention works
- JSON and TXT outputs are written
- the standard SGLang profiler path still produces trace plus timing dump
