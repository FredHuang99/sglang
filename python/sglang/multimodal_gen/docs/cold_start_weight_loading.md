# Diffusion Cold-Start Weight Loading Profile

This document describes the Phase 1 instrumentation for diffusion weight-load
cold-start analysis. Phase 1 is observation-only: it does not change model
loading behavior.

## Enable Profile Output

Use the existing diffusion profiling flags:

```bash
python -m sglang.multimodal_gen.runtime.launch_server \
  --model-path <Z-Image model path> \
  --profile-enabled \
  --profile-output-dir /data/profile \
  --profile-run-id zimage-sp4-baseline \
  <other launch args>
```

Each profiled component writes one JSON file per process under:

```text
<profile-output-dir>/<profile-run-id>_launch_weight_load/
```

The file name includes component name, global rank, local rank, and pid:

```text
weight_load_transformer_rank0_local0_pid12345.json
weight_load_vae_rank0_local0_pid12345.json
```

## JSON Fields

Every file uses the same schema so ranks can be compared directly:

```text
component
role
instance_id
rank
physical_rank
world_size
sp_rank
tp_rank
device
mem_kind
status
error
weight_load:discover_files_ms
weight_load:read_safetensors_ms
weight_load:cpu_materialize_ms
weight_load:pin_memory_ms
weight_load:h2d_or_param_copy_ms
weight_load:nccl_broadcast_ms
weight_load:rank0_wait_ms
weight_load:total_bytes
weight_load:staging_requested
weight_load:staging_effective
weight_load:staged_tensor_count
weight_load:pinned_tensor_count
weight_load:pinned_bytes
weight_load:pin_memory_error
weight_load:load_mode_requested
weight_load:load_mode_effective
weight_load:broadcast_tensor_count
weight_load:broadcast_bytes
weight_load:broadcast_error
```

Phase 1 fills `discover_files`, `read_safetensors`, `cpu_materialize`,
`h2d_or_param_copy`, and `total_bytes`. The `pin_memory`, `nccl_broadcast`, and
`rank0_wait` fields are reserved for later phases and remain zero in default
loading.

Phase 2 fills the staging fields only when `--diffusion-weight-staging` is not
`none` and the current process is SP rank0. Non-rank0 processes keep the default
loading path so Phase 2 can be compared with the Phase 1 baseline before
broadcast is introduced.

Phase 3 fills the load-mode and broadcast fields when
`--diffusion-weight-load-mode rank0-broadcast` is requested. In fallback cases,
`load_mode_requested` stays `rank0-broadcast` while `load_mode_effective` is
`default`.

## Phase 2 Rank0 Staging

Phase 2 adds an optional rank0 staging layer for transformer and standard
ModelRegistry VAE checkpoints. It does not implement NCCL broadcast and does
not skip non-rank0 safetensors reads.

Supported modes:

```text
none      default; current loader behavior
pageable  use the shared staging iterator without pinning CPU tensors
pinned    try tensor.pin_memory(), falling back to pageable on failure
auto      same fallback behavior as pinned; intended for benchmark scripts
```

Example commands:

```bash
# Baseline: no staging.
python -m sglang.multimodal_gen.runtime.launch_server \
  --model-path <Z-Image model path> \
  --profile-enabled \
  --profile-output-dir /data/profile \
  --profile-run-id zimage-sp4-staging-none \
  --diffusion-weight-staging none \
  <other launch args>

# Pageable rank0 staging.
python -m sglang.multimodal_gen.runtime.launch_server \
  --model-path <Z-Image model path> \
  --profile-enabled \
  --profile-output-dir /data/profile \
  --profile-run-id zimage-sp4-staging-pageable \
  --diffusion-weight-staging pageable \
  <other launch args>

# Pinned rank0 staging with pageable fallback.
python -m sglang.multimodal_gen.runtime.launch_server \
  --model-path <Z-Image model path> \
  --profile-enabled \
  --profile-output-dir /data/profile \
  --profile-run-id zimage-sp4-staging-pinned \
  --diffusion-weight-staging pinned \
  <other launch args>

# Auto mode for repeated experiments.
python -m sglang.multimodal_gen.runtime.launch_server \
  --model-path <Z-Image model path> \
  --profile-enabled \
  --profile-output-dir /data/profile \
  --profile-run-id zimage-sp4-staging-auto \
  --diffusion-weight-staging auto \
  <other launch args>
```

For Phase 2 result tables, add these columns to the Phase 1 summary:

```text
weight_load:staging_requested
weight_load:staging_effective
weight_load:staged_tensor_count
weight_load:pinned_tensor_count
weight_load:pinned_bytes
weight_load:pin_memory_error
```

Interpretation:

- `staging_requested=none` on non-rank0 is expected even if the launch flag is
  non-`none`; Phase 2 deliberately gates staging to SP rank0.
- `staging_effective=pageable` with a non-empty `pin_memory_error` means pinned
  memory was unavailable and the loader fell back without changing correctness.
- SP multi-card launch wall time may not improve in Phase 2 because non-rank0
  still performs the duplicated default read. The useful signal is whether
  rank0 produces stable staged tensors and meaningful pin/H2D timings.

## Phase 3 Transformer Rank0 Broadcast

Phase 3 adds an opt-in load mode for Z-Image transformer/denoiser weights under
`tp=1, sp=N`. Rank0 reads and materializes the checkpoint, non-rank0 ranks
materialize empty tensors with the same model structure, and the SP group then
broadcasts parameters and buffers from rank0.

Supported component:

```text
transformer
```

Unsupported components such as `vae` are ignored with a warning and keep the
default loader. VAE/decoder broadcast is reserved for Phase 4.

Example command:

```bash
python -m sglang.multimodal_gen.runtime.launch_server \
  --model-path <Z-Image model path> \
  --model-id Z-Image \
  --profile-enabled \
  --profile-output-dir /data/profile \
  --profile-run-id zimage-sp2-rank0-broadcast-pageable \
  --diffusion-weight-staging pageable \
  --diffusion-weight-load-mode rank0-broadcast \
  --diffusion-weight-broadcast-components transformer \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --attention-backend fa
```

Preconditions and fallback:

- `tp_size` must be 1.
- SP model-parallel state must be initialized and `sp_world_size > 1`.
- FSDP inference is not supported in Phase 3.
- Only `transformer` enters the broadcast path.
- If a precondition fails before collectives start, the loader falls back to
  default and records `load_mode_effective=default`.
- If a collective fails after broadcast has started, the error is recorded in
  `weight_load:broadcast_error` and the launch raises. Automatic fallback at
  that point is intentionally avoided because ranks could otherwise diverge and
  hang.

Profile checks:

```text
rank0 transformer:
  weight_load:total_bytes ~= full transformer size
  weight_load:load_mode_effective = rank0-broadcast
  weight_load:broadcast_bytes ~= full transformer size

non-rank0 transformer:
  weight_load:total_bytes = 0
  weight_load:read_safetensors_ms ~= 0
  weight_load:load_mode_effective = rank0-broadcast
  weight_load:rank0_wait_ms captures waiting for rank0 read/materialize
  weight_load:broadcast_bytes ~= full transformer size
```

For the first functional run, use `--diffusion-weight-staging pageable` so
pinned-memory overhead does not hide the broadcast effect. After correctness is
stable, repeat with `none`, `pinned`, and `auto`.

Benefit tests:

1. SP2 smoke: compare default vs rank0-broadcast and confirm cluster
   safetensors read bytes drop from two full transformer copies to one.
2. SP4/SP8 scaling: run at least three trials per mode and compare max
   transformer load time, cluster read bytes, rank0 wait, NCCL broadcast time,
   and launch wall time.
3. Use SP4/SP8 for paper scaling claims; SP2 is only a minimal functional and
   benefit smoke test.

## Baseline Procedure

Run the same Z-Image deployment with `tp=1` and `sp=1/2/4/8`. For each run:

1. Set a unique `--profile-run-id`, for example `zimage-sp1-default`.
2. Keep model path, dtype, GPU set, and warmup settings identical.
3. Collect all `weight_load_*.json` files from the launch profile directory.
4. For each component, compare max rank time and rank variance.

The first baseline table should report:

```text
sp_degree
component
max(weight_load:read_safetensors_ms)
max(weight_load:cpu_materialize_ms)
max(weight_load:h2d_or_param_copy_ms)
max(weight_load:total_bytes)
rank_count
```

If non-rank0 processes show the same `total_bytes` and similar read/materialize
time as rank0, the profile confirms duplicated per-rank loading. If
`h2d_or_param_copy_ms` grows with SP degree, PCIe/GPU copy contention is also
part of the launch bottleneck.

## CPU/Mock Functional Test

The helper-level unit test uses tiny safetensors on CPU. It validates that
profile fields exist, timings are non-negative, rank labels are present, and
`total_bytes` matches the tensors read from the checkpoint.
