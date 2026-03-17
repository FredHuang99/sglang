# Wan2.2-TI2V-5B Profiling Playbook

This playbook is for profiling `Wan-AI/Wan2.2-TI2V-5B-Diffusers` on SGLang-Diffusion with the following fixed workload:

- task: TI2V (Text + Image to Video)
- prompt: arbitrary text
- resolution: `704x1280` (or supported resolutions)
- frames: `121` (effective video frames)
- GPUs: `8x H200 140GB NVLink`

The examples below assume:

```bash
export MODEL="/workspace/models/Wan2_2-TI2V-5B-Diffusers"
export PROMPT="A cinematic science-fiction city with reflective rain streets."
export OUT_DIR="/workspace/wan22_ti2v5b_profile"
mkdir -p "${OUT_DIR}"
```

## 1. Key Parameters

### 1.1 Attention Backend

For GPU testing, it is strongly recommended to use Flash Attention for optimal performance:

```bash
--attention-backend fa  # flashattention
```

### 1.2 VAE Frame Parameters

TI2V task VAE testing requires special attention to frame parameters:

- **`--num-frames`**: Target video frames (e.g., 121 frames), used for VAE Decoder
- **`--vae-encode-input-frames`**: VAE Encoder input frame count. For TI2V tasks, this should be 1 (only encoding the first frame condition image)

> **Note**: The script automatically sets VAE Encoder input frames to 1 for I2V/TI2V tasks based on task type, no manual setting required.

### 1.3 Resolution Requirements

VAE testing must use supported resolutions:
- `480x832` (portrait)
- `832x480` (landscape)

## 2. What The Current Code Actually Supports

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

How to get pure CPU encoder time:

- Do not use `--text-encoder-cpu-offload` as a proxy.
- Use the helper script in this repo with `--stage encoder --device cpu`.

### DiT

- The profiling matrix in this playbook focuses on SP only.
- Codebase features outside the profiling matrix still exist: TP, FSDP-style offload, `--dit-cpu-offload`, and `--dit-layerwise-offload`.

How SP works in the current Wan DiT path:

- The model input is sequence-sharded first.
- With the fixed workload here, the latent shape is `[1, 48, 31, 44, 80]` (note: z_dim=48 for TI2V).
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

### TI2V Specific Notes

For TI2V (Text + Image to Video) tasks:

- **VAE Encoder**: Only encodes 1 frame (the condition/first frame image), not the full video
  - Input: `[1, 3, 1, H, W]` (single frame)
  - Output latent: `[1, 48, 1, H/vae_stride[1], W/vae_stride[2]]`
- **VAE Decoder**: Decodes the full latent frames to video
  - Input: `[1, 48, latent_frames, H/vae_stride[1], W/vae_stride[2]]`
  - Output: `[1, 3, effective_frames, H, W]`

Latent frame calculation:
- `latent_frames = (effective_frames - 1) / temporal_compression_ratio + 1`
- For 121 frames with temporal_compression_ratio=4: `(121-1)/4 + 1 = 31` latent frames

Concrete shapes for the fixed workload:

- VAE encode input: `[1, 3, 1, 480, 832]` (TI2V uses 1 frame for encode)
- VAE encode output latent: `[1, 48, 1, 44, 80]`
- VAE decode input latent: `[1, 48, 31, 44, 80]` (for 121 effective frames)
- VAE decode output: `[1, 3, 121, 480, 832]`

What `--vae-cpu-offload` means:

- It does not mean the VAE computes entirely on CPU.
- The current encode/decode stages explicitly move the VAE back to the local CUDA device before compute and move it back to CPU after the stage.
- There is no VAE layerwise prefetch path in the current code.

How to get pure CPU VAE compute time:

- Do not use `--vae-cpu-offload` as a proxy.
- Use the helper script with `--stage vae-encode --device cpu` or `--stage vae-decode --device cpu`.

## 3. What Existing `sglang generate` Profiling Can And Cannot Measure

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

## 4. End-To-End Baseline Command

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

## 5. Stage Helper

Use the helper added in this repo when you want direct per-stage timing or pure CPU timing:

```bash
scripts/playground/profile_wan_ti2v_stages.py
```

Features:

- stages: `encoder`, `dit`, `vae-encode`, `vae-decode`
- devices: `cpu`, `cuda`
- outputs: stdout summary + optional JSON + optional TXT
- default workload: exactly the workload described at the top of this page
- attention backend: `--attention-backend fa` for Flash Attention

Common flags used below:

- `--warmup-iters 1`
- `--profile-iters 3`
- `--sync-stage-profiling all` for accurate CUDA wall times

## 6. Encoder Commands

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
  --attention-backend fa \
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
  --attention-backend fa \
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
  --attention-backend fa \
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
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/encoder/gpu8/profile.json" \
  --output-txt "${OUT_DIR}/encoder/gpu8/profile.txt"
```

## 7. DiT Commands

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
  --attention-backend fa \
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
  --attention-backend fa \
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
  --attention-backend fa \
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
  --attention-backend fa \
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
  --attention-backend fa \
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
  --attention-backend fa \
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
  --attention-backend fa \
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
  --attention-backend fa \
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
  --attention-backend fa \
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
  --attention-backend fa \
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
  --attention-backend fa \
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
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/dit/gpu8_u1_r8/profile.json" \
  --output-txt "${OUT_DIR}/dit/gpu8_u1_r8/profile.txt"
```

## 8. VAE Commands

The helper exposes VAE encode and decode separately.

> **Important**: For TI2V tasks:
> - VAE Encoder automatically uses 1 frame (the condition image)
> - Use `--num-frames 1` or `--vae-encode-input-frames 1` for VAE Encode
> - VAE Decoder uses the actual target frames

### VAE Encode On CPU

```bash
python scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-encode \
  --device cpu \
  --prompt "${PROMPT}" \
  --num-frames 1 \
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
  --num-frames 121 \
  --height 704 \
  --width 1280 \
  --num-inference-steps 50 \
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
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --num-frames 1 \
  --vae-encode-input-frames 1 \
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
  --ulysses-degree 1 \
  --ring-degree 1 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --num-frames 121 \
  --height 704 \
  --width 1280 \
  --num-inference-steps 50 \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/vae_decoder/gpu1/profile.json" \
  --output-txt "${OUT_DIR}/vae_decoder/gpu1/profile.txt"
```

### VAE Encode On 2 GPUs

```bash
torchrun --standalone --nproc_per_node 2 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-encode \
  --device cuda \
  --num-gpus 2 \
  --sp-degree 2 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --num-frames 1 \
  --vae-encode-input-frames 1 \
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
  --ulysses-degree 2 \
  --ring-degree 1 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --num-frames 121 \
  --height 704 \
  --width 1280 \
  --num-inference-steps 50 \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/vae_decoder/gpu2/profile.json" \
  --output-txt "${OUT_DIR}/vae_decoder/gpu2/profile.txt"
```

### VAE Encode On 4 GPUs

```bash
torchrun --standalone --nproc_per_node 4 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-encode \
  --device cuda \
  --num-gpus 4 \
  --sp-degree 4 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --num-frames 1 \
  --vae-encode-input-frames 1 \
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
  --ulysses-degree 4 \
  --ring-degree 1 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --num-frames 121 \
  --height 704 \
  --width 1280 \
  --num-inference-steps 50 \
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
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --num-frames 1 \
  --vae-encode-input-frames 1 \
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
  --ulysses-degree 8 \
  --ring-degree 1 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --num-frames 121 \
  --height 704 \
  --width 1280 \
  --num-inference-steps 50 \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/vae_decode/gpu8/profile.json" \
  --output-txt "${OUT_DIR}/vae_decode/gpu8/profile.txt"
```

## 9. Output Files

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
- extra metadata:
  - `vae_encode_input_frames`: actual frames used for VAE encode
  - `effective_video_num_frames`: effective video frames after request adjustment
  - `latent_num_frames`: latent frame count for VAE decode

## 10. Performance Summary

This section summarizes the profiling results for Wan2.2-TI2V-5B on H200 GPUs.

### 10.1 Test Configuration

- Model: Wan2.2-TI2V-5B-Diffusers
- Resolution: 1280x704
- Frames: 121 (effective video frames)
- Inference Steps: 50
- Prompt: "A cinematic science-fiction city with reflective rain streets."

### 10.2 Encoder (Text Encoder)

| GPUs | Time (s) | Speedup | Memory (GB) |
|------|----------|---------|-------------|
| CPU  | 2.48    | 1.0x    | N/A         |
| 1    | 0.048   | 51.7x   | 21.4        |
| 2    | 0.062   | 40.0x   | 10.8        |
| 4    | 0.062   | 40.0x   | 5.5         |
| 8    | 0.065   | 38.2x   | 2.9         |

**Note**: Encoder is very fast on GPU. 1 GPU is sufficient for most use cases.

### 10.3 VAE Encoder

| GPUs | Time (s) | Speedup | Memory (GB) |
|------|----------|---------|-------------|
| CPU  | 2.22    | 1.0x    | N/A         |
| 1    | 0.035   | 63.4x   | 14.1        |
| 2    | 0.024   | 92.5x   | 7.6         |
| 4    | 0.017   | 130.6x  | 4.2         |
| 8    | 0.016   | 138.8x  | 2.8         |

**Note**: VAE Encoder benefits significantly from multi-GPU parallelism. 4-8 GPUs provide optimal performance.

### 10.4 DiT (Denoising)

| GPUs | sp  | ulysses | ring | Time (s) | Speedup | Efficiency | Memory (GB) |
|------|-----|----------|------|-----------|---------|------------|-------------|
| 1    | 1   | 1        | 1    | 97.35     | 1.00x  | 100%       | 53.2        |
| 2    | 2   | 2        | 1    | 59.99     | 1.62x  | 81%        | 45.7        |
| 2    | 2   | 1        | 2    | 62.48     | 1.56x  | 78%        | 45.7        |
| 4    | 4   | 4        | 1    | 42.68     | 2.28x  | 57%        | 40.5        |
| 4    | 4   | 2        | 2    | 38.83     | 2.51x  | 63%        | 40.5        |
| 4    | 4   | 1        | 4    | 55.95     | 1.74x  | 44%        | 40.6        |
| 8    | 8   | 8        | 1    | 17.82     | 5.47x  | 68%        | 38.2        |
| 8    | 8   | 4        | 2    | 22.20     | 4.38x  | 55%        | 38.2        |
| 8    | 8   | 2        | 4    | 23.39     | 4.16x  | 52%        | 38.2        |
| 8    | 8   | 1        | 8    | 26.42     | 3.69x  | 46%        | 38.2        |

**Key Findings**:
- **Best config for 2 GPUs**: ulysses=2, ring=1 (1.62x speedup)
- **Best config for 4 GPUs**: ulysses=2, ring=2 (2.51x speedup)
- **Best config for 8 GPUs**: ulysses=8, ring=1 (5.47x speedup)
- Ulysses strategy outperforms Ring strategy at higher parallelism
- Memory decreases with more GPUs (53.2GB → 38.2GB)

### 10.5 VAE Decoder

| GPUs | sp  | ulysses | ring | Time (s) | Speedup | Efficiency | Memory (GB) |
|------|-----|----------|------|-----------|---------|------------|-------------|
| 1    | 1   | 1        | 1    | 9.41      | 1.00x  | 100%       | 53.2        |
| 2    | 2   | 2        | 1    | 5.24      | 1.80x  | 90%        | 45.7        |
| 4    | 4   | 4        | 1    | 2.81      | 3.35x  | 84%        | 40.6        |

**Key Findings**:
- VAE Decoder has excellent scaling efficiency:
  - 2 GPUs: 1.80x speedup (90% efficiency)
  - 4 GPUs: 3.35x speedup (84% efficiency)
- Memory decreases with more GPUs (53.2GB → 40.6GB)
- VAE Decoder scales better than DiT due to lower communication overhead

### 10.6 Recommended Configurations

| Scenario | Configuration | Expected Time |
|----------|---------------|---------------|
| Cost-optimized | 1 GPU DiT | ~97s |
| Balanced | 4 GPU (u=2, r=2) DiT | ~39s |
| High-performance | 8 GPU (u=8, r=1) DiT | ~18s |

### 10.7 Component Resource Requirements

| Component | 1 GPU | 4 GPU | 8 GPU |
|----------|-------|-------|-------|
| Encoder (TP) | 21.4 GB | 5.5 GB | 2.9 GB |
| VAE Encoder | 14.1 GB | 4.2 GB | 2.8 GB |
| VAE Decoder | 53.2 GB | 40.6 GB | - |
| DiT | 53.2 GB | 40.5 GB | 38.2 GB |

## 10. Minimal Validation Matrix

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
