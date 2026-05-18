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
```

Phase 1 fills `discover_files`, `read_safetensors`, `cpu_materialize`,
`h2d_or_param_copy`, and `total_bytes`. The `pin_memory`, `nccl_broadcast`, and
`rank0_wait` fields are reserved for later phases and remain zero in default
loading.

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
