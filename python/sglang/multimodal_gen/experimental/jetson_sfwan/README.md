# Minimal SFWan2.1 realtime service

This package is a deliberately small, single-model research service for
`wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers`. Its numerical contract follows
FastVideo's SFWan2.1 path; SGLang supplies only its component loader, causal Wan
Transformer/VAE implementations, causal caches, and low-level forward context.
It does not use SGLang's generic Pipeline, Stage, Req, ForwardBatch, scheduler
preparation, or decoding preprocessing abstractions.

The three process roles are:

- `monolithic`: tokenizer/T5, causal DiT, and causal VAE in one process.
- `dit`: tokenizer/T5 and causal DiT, with optional HTTP or same-host SHM
  latent transport to a VAE process.
- `vae`: a stateful causal VAE. One complete video request owns the only
  running slot until all of its chunks have been decoded.

Every role has a request-level FIFO waiting queue and exactly one running slot.
Batch size is fixed at one.

For a class-by-class audit, request lifecycles, buffer ownership, and the exact
meaning of every transport/profile parameter, see [CODE_GUIDE.md](CODE_GUIDE.md).

## FastVideo-aligned numerical contract

The reference request is 480x832, 81 frames, 16 FPS, seed 1024:

- T5 output is masked and zero-padded to `[1, 512, 4096]` FP32.
- The complete initial latent is generated in FP32 with one seeded CPU
  `torch.Generator`.
- Each DiT chunk is `[1, 16, 3, H/8, W/8]` BF16.
- Scheduler construction calls `set_timesteps` once. Generation never calls it.
- Raw DMD indices `[1000, 750, 500, 250]` are mapped through the loaded
  1000-step, shift-5 schedule. The model therefore receives the mapped FP32
  timesteps (approximately `[1000, 937.5, 833.33, 625]`).
- Every chunk performs four Transformer DMD forwards, three CPU BF16 re-noise
  draws, then one integer-zero clean-KV forward.
- `pred_noise_to_pred_video` uses FP64 intermediates and returns BF16. The
  causal path does not call `scheduler.step` or `scale_model_input`.
- CFG is intentionally absent: no negative prompt is encoded and no
  `guidance_scale` or `num_inference_steps` parameter is exposed.
- A latent chunk becomes visible to the monolithic VAE or a transport only
  after its clean-KV forward has completed.

The DiT clean BF16 value is written into the FP32 full-latent result, matching
FastVideo. Every VAE path then applies the same boundary conversion:

```text
normalized BF16 -> FP32 -> z * latents_std_fp32 + latents_mean_fp32
```

For each three-latent chunk, the VAE calls `post_quant_conv` once and calls the
decoder three times with `[1, 16, 1, h, w]`. Wan's native causal convolution
cache supplies startup history:

- first latent: two zero-history slots, producing one RGB frame;
- second latent: one real plus one zero-history slot, producing four frames;
- third and later latents: two real history slots, producing four frames.

The first chunk therefore yields 9 RGB frames and later chunks yield 12. Pixel
conversion is `(decoded / 2 + 0.5).clamp(0, 1) * 255`, followed directly by
`uint8` conversion (truncation, not rounding).

Valid physical frame counts satisfy:

```text
latent_frames = (video_frames - 1) / 4 + 1
latent_frames % 3 == 0
video_frames % 12 == 9
```

Examples are 81, 93, and 105 frames. Values above 81 are shape-compatible, but
the checkpoint's long-video visual quality is not guaranteed.

## Mode and resource isolation

| Mode | Latent path | Transfer resources |
|---|---|---|
| monolithic generation | GPU tensor passed directly from DiT to VAE | no latent D2H/H2D, sender, safetensors, or SHM |
| disaggregated HTTP | BF16 GPU -> reusable pinned D2H -> safetensors bytes -> header-only body view -> reusable pinned H2D -> GPU | one body-data-to-pinned CPU copy on VAE; bounded depth, default 2 |
| disaggregated SHM | BF16 GPU -> CUDA-registered POSIX SHM -> GPU | one request mapping; no safetensors payload or second staging copy |
| DiT profile | all chunks execute prompt/noise/DMD/clean-KV, then discard | no VAE registration, D2H, or sender |
| VAE profile | all deterministic dummy chunks use normal HTTP ingress | transfer executes normally but is excluded from execution metrics |

The VAE starts a one-chunk look-ahead ingress task while it decodes the current
chunk. This lets H2D overlap decoder compute while preserving request-level
FCFS and feature-cache ownership. Out-of-order chunks may wait in CPU memory or
SHM, but are always decoded in increasing chunk order.

For HTTP ingress, FastAPI first provides one pageable Python `bytes` body. The
VAE validates only the safetensors header and keeps a zero-copy view of the
tensor-data range while the chunk waits. Once that FCFS job needs the chunk, it
copies those raw BF16 bytes directly into a reusable pinned slot and launches
non-blocking H2D. It does not materialize an intermediate pageable Torch
tensor. This removes one full application-level CPU copy; it is not
network-stack-to-GPU zero-copy.

Detailed timing is disabled by default. Start a server with `--enable-profile`
to collect low-level execution and transfer metrics. Profile requests are
rejected unless that flag is present. Normal servers retain only inexpensive
lifecycle data such as accepted/started/completed timestamps, queue wait, total
wall time, chunk IDs, and frame counts. `--enable-nvtx` is independent: it may
emit NVTX ranges without enabling CUDA Event timing or per-call synchronization.

## CPU offload controls

CPU offload is selected when each server starts:

| Flag | Default | Effective roles | Behavior |
|---|---:|---|---|
| `--text-encoder-cpu-offload [true\|false]` | `true` | monolithic, dit | FSDP CPU offload with full C10d; layerwise offload on the strict Jetson local backend |
| `--dit-cpu-offload [true\|false]` | `false` | monolithic, dit | single-GPU FSDP CPU offload with full C10d; layerwise offload on the strict Jetson local backend |
| `--vae-cpu-offload [true\|false]` | `false` | monolithic, vae | move the complete VAE to GPU before each three-latent chunk and back to CPU afterward |

The flag alone means `true`; for example, `--dit-cpu-offload` enables DiT
offload, while `--text-encoder-cpu-offload false` keeps T5 GPU-resident.
Irrelevant role flags are accepted but load no component and allocate no
offload resource. The internal SGLang loader runs in `performance_mode=manual`,
so its automatic tuner does not replace these explicit choices. PyTorch builds
without C10d are accepted only at world size one with every parallel degree
equal to one; that path creates identity local groups and never constructs
NCCL, TCPStore, ProcessGroup, or FSDP state.

VAE offload moves only registered model parameters and buffers. The
request-scoped feature cache remains on GPU across chunks and is reset only at
the request boundary. With monolithic VAE offload enabled, the exact order is:

```text
DiT chunk i -> VAE weights H2D -> VAE decode chunk i -> RGB CPU
            -> VAE weights D2H -> DiT chunk i+1
```

VAE weight swaps are outside `profile_execution`, but remain inside
`decode_wall_ms`, request wall time, and `monolithic_total_ms`. T5 offload is
inside text-encoding wall time; DiT FSDP or layerwise block movement occurs
inside each Transformer forward and is therefore included in its forward
metric. `GET /v1/engine` exposes `contract.distributed_backend`,
`contract.cpu_offload_requested`, and `contract.cpu_offload_effective`.
The legacy role-specific `contract.cpu_offload` booleans remain available.

## Jetson Orin TensorRT VAE

The optional TensorRT backend is deliberately isolated from the reference
PyTorch decoder:

| `--vae-precision` | VAE implementation | PyTorch VAE weights loaded |
|---|---|---:|
| `fp32` | existing causal Wan VAE, reference precision | yes |
| `fp16` | existing causal Wan VAE, non-reference precision | yes |
| `fp16_trt` | fixed-shape TensorRT FP16 plans | no |
| `int8_trt` | fixed-shape explicit-Q/DQ TensorRT plans | no |

V1 TensorRT plans are batch-one, SM87-only, and fixed to a normalized latent
chunk of `[1,16,3,60,104]` for 480x832 output. They must be built on the same
Jetson software stack that will execute them. The build produces separate
`initial` and `steady` engines:

- `initial`: latent input only, 9 RGB frames and 32 cache outputs;
- `steady`: latent plus 32 cache inputs, 12 RGB frames and 32 updated cache
  outputs.

The runtime owns two FP16 cache banks and alternates `A -> B -> A`. One bank is
1,888,573,440 bytes (about 1.759 GiB), so both banks occupy about 3.518 GiB.
They are allocated once when the server starts and are reused across requests.
TensorRT VAE mode does not permit `--vae-cpu-offload`.

Run the fail-fast signature preflight on the Orin before starting the long
initial/steady builds:

```bash
export SFWAN_MODEL=wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers
export SFWAN_TRT_DIR=/workspace/engines/sfwan-vae-trt-sm87

python -m sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_build \
  --model-path "$SFWAN_MODEL" \
  --output-dir "$SFWAN_TRT_DIR" \
  --height 480 \
  --width 832 \
  --seed 1024 \
  --workspace-gib 8 \
  --profiling-verbosity none \
  --resume \
  --preflight-only \
  --timing-cache "$SFWAN_TRT_DIR/tensorrt_timing_qdq_v4.cache"
```

Only after that command returns `preflight_passed: true`, run the full build
with the same identity and `--resume`:

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_build \
  --model-path "$SFWAN_MODEL" \
  --output-dir "$SFWAN_TRT_DIR" \
  --height 480 \
  --width 832 \
  --seed 1024 \
  --workspace-gib 8 \
  --profiling-verbosity none \
  --resume \
  --timing-cache "$SFWAN_TRT_DIR/tensorrt_timing_qdq_v4.cache"
```

The command reuses the loaded SGLang Wan VAE to collect one deterministic
speed-only scale set, exports ONNX opset 19, and inserts explicit signed INT8
Q/DQ around only the 28 residual-block 3x3x3 Conv3d modules. Each graph has 84
unrolled calls. The [TensorRT 10.3 release
notes](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1030/pdf/TensorRT-Release-Notes.pdf)
document the FP32 Q/DQ data-and-scale restriction, so Q/DQ v4 casts each FP16
activation to FP32 before its independent Q/DQ pair, gives every call its own
FP32 weight constant, FP32 per-channel scale, INT8 zero point, weight Q/DQ, and
FP32 bias, then casts the FP32 Conv output back to FP16. No weight/DQ output is
shared across calls. DQ remains directly adjacent to Conv so TensorRT can fuse
the quantized input and constant weight instead of retaining separate Half
reformats that select an `f16f16` Conv tactic.
Conv input/output shapes for the preflight signatures are captured from the
real initial and steady dummy decoder forwards, including causal padding. If
ONNX shape inference also provides a target shape, the two sources must match;
missing internal ONNX `value_info` alone is not treated as a model error.

Q/DQ rewriting accepts only a graph that was genuinely exported as opset 19;
it never changes an older graph's `opset_import` label. On `--resume`, legacy
`initial_fp16.onnx` and `steady_fp16.onnx` files remain available for their
already-built FP16 plans. If they use an older opset, the builder preserves
them and atomically exports `initial_fp16_opset19.onnx` and
`steady_fp16_opset19.onnx` as the separate Q/DQ sources. A failed export leaves
only a `.partial` file and cannot replace a previously validated source.

The fail-closed build order is:

1. preserve any validated legacy FP16 graphs and validate or atomically export
   the two real-opset-19 Q/DQ source graphs;
2. extract every unique Conv signature from all 168 initial/steady call sites
   and build one v4 Q/DQ probe per signature;
3. stop before any full VAE plan if one signature lacks a static INT8 tactic;
4. build `initial_int8_qdq_v4.plan` with detailed Inspector data and audit all
   84 target calls;
5. repeat for `steady_int8_qdq_v4.plan`;
6. use those same audited detailed plans at runtime, and build/reuse only the
   two FP16 control plans. INT8 is never rebuilt as an unauditable NONE plan.

`--resume` records `built`, `audit_failed`, and `audit_passed` separately in
`build_state_v4.json`. A restart verifies the source ONNX hash, plan hash, Q/DQ
schema, and profiling verbosity before re-auditing or reusing a stage. Existing
legacy FP16 ONNX/plans are reused only when their legacy build identity matches
the model path, resolved checkpoint, shape, and seed; they are not relabelled
or used as Q/DQ input. Old INT8 plans are never adopted because their Q/DQ graph
may differ. V4 INT8 ONNX, plans, state, audit, scales, probes, and timing cache
all use new names and never overwrite v2/v3 artifacts. Validated v3 activation
scales and unquantized opset-19 source graphs may be adopted only when the model,
resolved checkpoint, shape, seed, SM, and opset identities match. A build returns only a
candidate timing cache. The candidate is atomically committed through
TensorRT's `IBuilderConfig` API only after that probe or full plan passes tactic
audit; a failed build or FP16/dynamic-filter audit leaves the stable cache
untouched.

A build is accepted only if:

- each graph has 28 logical modules, 84 Conv calls, 84 activation FP16-to-FP32
  casts and Q/DQ pairs, 84 independent constant-weight Q/DQ pairs, and 84 Conv
  output FP32-to-FP16 casts;
- all Q/DQ axes, FP32 scale/weight/bias dtypes, rank-5 weight shapes, positive
  scales, zero points, call-site shapes, Cast directions, and Conv attributes
  pass structural audit;
- every unique initial/steady Conv signature passes preflight;
- every full-plan call maps through Inspector `Name` or `Metadata` and proves
  INT8 activation, no dynamic filter (`HasDynamicFilter` is normally `0`),
  non-empty static INT8 weights, and an
  INT8/IMMA/i8 tactic with no `f16f16`, FP32, or TF32 fallback.

Inspect the fail-closed audit:

```bash
jq '{
  passed,
  errors,
  probe_suite: .probe_suite | {
    passed, signature_count, probed_signature_count,
    source_call_site_counts
  },
  initial: .tactics.initial | {
    passed, mapped_count, unmapped_call_sites, non_int8_call_sites,
    dynamic_filter_call_sites, fp16_fallback_call_sites, plan_sha256
  },
  steady: .tactics.steady | {
    passed, mapped_count, unmapped_call_sites, non_int8_call_sites,
    dynamic_filter_call_sites, fp16_fallback_call_sites, plan_sha256
  }
}' "$SFWAN_TRT_DIR/int8_audit_v4.json"

sha256sum "$SFWAN_TRT_DIR"/*.plan
```

Detailed audit plans and Inspector JSON are retained in the same directory;
there is no second full rebuild:

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine="$SFWAN_TRT_DIR/initial_int8_qdq_v4.plan" \
  --dumpLayerInfo \
  --profilingVerbosity=detailed

/usr/src/tensorrt/bin/trtexec \
  --loadEngine="$SFWAN_TRT_DIR/steady_int8_qdq_v4.plan" \
  --dumpLayerInfo \
  --profilingVerbosity=detailed
```

The INT8 plans retain detailed Inspector metadata, but the runtime lowers each
execution context to NVTX `NONE` unless the server is started with
`--enable-nvtx`. `/v1/engine` reports the effective value in
`vae_runtime_nvtx_verbosity`.

Start a VAE-only TensorRT server by choosing one of the plan precisions:

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.server \
  --role vae \
  --model-path "$SFWAN_MODEL" \
  --host 0.0.0.0 \
  --port 30001 \
  --latent-transport http \
  --vae-precision int8_trt \
  --vae-engine-dir "$SFWAN_TRT_DIR" \
  --vae-cpu-offload false \
  --enable-profile \
  --output-dir /workspace/results/sfwan-vae-int8
```

Use `--vae-precision fp16_trt` with the same directory for the TensorRT FP16
control. The existing VAE profile client still uploads all seven chunks:

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.client profile-vae \
  --server-url http://127.0.0.1:30001 \
  --height 480 \
  --width 832 \
  --num-frames 81 \
  --seed 1024 \
  --warmup 10 \
  --repeat 50 \
  --summary-json /workspace/results/vae-int8-trt-480x832-81.json
```

With `--enable-profile`, each chunk adds
`trt_input_cast_cuda_ms`, `trt_engine_cuda_ms`,
`trt_output_finalize_cuda_ms`, `trt_engine_kind`, and `trt_precision` under
`profile_execution.chunks[]`. `trt_engine_cuda_ms` covers only
`execute_async_v3`; `chunk_execution_cuda_ms` also includes common FP32
denormalization, the FP16 ingress cast, and FP32 output clamp. HTTP/H2D, queue,
RGB D2H, and MP4 remain outside `profile_execution`.

For monolithic execution, pass the same `--vae-precision` and
`--vae-engine-dir` to a `--role monolithic` server. Its order remains strictly
`DiT chunk i -> clean-KV -> TRT VAE chunk i -> RGB CPU -> DiT chunk i+1`;
there is no latent D2H/H2D or DiT/VAE overlap.

## 5090 monolithic

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.server \
  --role monolithic \
  --model-path wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers \
  --port 30000 \
  --output-dir ./sfwan_outputs

python -m sglang.multimodal_gen.experimental.jetson_sfwan.client generate \
  --server-url http://127.0.0.1:30000 \
  --num-requests 1 \
  --arrival-mode burst
```

The process creates one component-loading context and injects it into the DiT
and VAE roles; it does not load the model twice.

## 5090 DiT plus Jetson Orin VAE over HTTP

On the Jetson:

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.server \
  --role vae \
  --model-path wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers \
  --host 0.0.0.0 \
  --port 30001 \
  --public-url http://JETSON_IP:30001 \
  --latent-transport http \
  --vae-precision fp32 \
  --output-dir ./sfwan_vae_outputs
```

On the 5090:

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.server \
  --role dit \
  --model-path wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers \
  --host 0.0.0.0 \
  --port 30000 \
  --public-url http://DIT_IP:30000 \
  --vae-url http://JETSON_IP:30001 \
  --latent-transport http \
  --transfer-depth 2
```

Submit an asynchronous Poisson workload:

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.client generate \
  --server-url http://DIT_IP:30000 \
  --num-requests 4 \
  --arrival-mode poisson \
  --poisson-lambda 0.2 \
  --output-dir ./downloaded_videos \
  --summary-json ./results/poisson.json
```

HTTP sends are safetensors-only. Repeated PUTs with identical content are
idempotent; a conflicting digest is rejected.

## Same Linux CUDA host, separate DiT/VAE processes over SHM

SHM is explicit and fail-closed. Both processes must run on the same Linux CUDA
host; CUDA events are local to a process and are never passed across processes.

Start the VAE:

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.server \
  --role vae \
  --model-path wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers \
  --host 127.0.0.1 \
  --port 30001 \
  --latent-transport shm \
  --vae-precision fp32
```

Start the DiT:

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.server \
  --role dit \
  --model-path wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers \
  --host 127.0.0.1 \
  --port 30000 \
  --vae-url http://127.0.0.1:30001 \
  --latent-transport shm \
  --transfer-depth 2
```

The VAE creates a page-aligned request-level POSIX shared-memory mapping. Both
processes independently call `cudaHostRegister` on their mapping. DiT records a
local D2H event, sends a ready control message only after that event, and the
VAE creates a different local H2D event. The decoder stream waits on only the
VAE-side event.

## DiT-only profile

A DiT server may start without `--vae-url`:

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.server \
  --role dit \
  --model-path wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers \
  --enable-profile \
  --port 30000

python -m sglang.multimodal_gen.experimental.jetson_sfwan.client profile-dit \
  --server-url http://127.0.0.1:30000 \
  --warmup 1 \
  --repeat 3 \
  --summary-json ./results/dit.json
```

Each iteration resets RNG and caches and executes tokenizer/T5, full FP32 noise
creation, four DMD steps per chunk, re-noise, and clean-KV for the complete
request. The default 81-frame profile covers seven chunks with IDs 0 through 6;
each chunk reports its own four DMD calls and clean-KV call. It does not allocate
latent transfer buffers. Client submission RTT and FCFS queue wait are not part
of `profile_execution`.

## VAE-only profile

Start the VAE server in profile mode:

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.server \
  --role vae \
  --model-path wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers \
  --latent-transport http \
  --vae-precision fp32 \
  --enable-profile \
  --port 30001
```

The client intentionally uses normal HTTP ingress so the VAE receives the same
pageable request-body-to-pinned and H2D path as a remote DiT. Those preparation
steps still execute, but `profile_execution` contains only VAE model execution:

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.client profile-vae \
  --server-url http://JETSON_IP:30001 \
  --warmup 1 \
  --repeat 3 \
  --summary-json ./results/jetson_vae.json
```

The client generates one complete deterministic normalized FP32 dummy latent on
CPU, converts it once to BF16, then slices three-latent chunks. Decoder output,
RGB D2H, and MP4 are omitted by default. Add `--save-video` to include them.
Downloaded artifact paths then appear under top-level `saved_outputs`, outside
the `measured` and `all_iterations` execution records.
Every warmup/repeat is one complete, independent VAE request. For the default
81 frames, it uploads and decodes chunk IDs 0 through 6. Chunk 0 exercises the
request-startup `1/4/4` decoder behavior and produces 9 frames; chunks 1 through
6 each exercise steady-state `4/4/4` behavior and produce 12 frames. The result
reports every chunk separately and sums their execution time. Upload RTT,
safetensors parsing, pinned copies, H2D, queue wait, RGB D2H, and MP4 are not
included in that sum.

## Client arrival modes

`generate` supports:

- `burst`: all POSTs are scheduled immediately;
- `fixed --fixed-interval-seconds S`;
- `poisson --poisson-lambda L`: exponential inter-arrival times with arrival
  rate lambda `L` requests/s and mean interval `1/L` seconds.

POSTs are independent tasks and never wait for a previous generation to
finish. A JSONL workload can override prompt, height, width, frames/duration,
FPS, and seed per request; its non-empty row count must equal `--num-requests`.

## 93-frame hardware check

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.client generate \
  --server-url http://127.0.0.1:30000 \
  --num-requests 1 \
  --num-frames 93
```

This creates 24 latent frames, eight chunks, and exactly 93 RGB frames. The job
also records the unvalidated-long-video warning.

## API and metrics

- `POST /v1/generations`
- `POST /v1/dit-profiles`
- `POST /v1/latent-jobs`
- `PUT /v1/latent-jobs/{id}/chunks/{index}` for HTTP safetensors
- `POST /v1/latent-jobs/{id}/chunks/{index}/ready` for SHM control
- `GET /v1/jobs/{id}` and `GET /v1/jobs/{id}/result`
- `GET /v1/engine` and `GET /health`

Lifecycle metrics (timestamps, queue wait, total wall time, model load, chunk
IDs, frame counts) remain inexpensive and always available. With
`--enable-profile`, `profile_execution` contains model execution only, while raw
job diagnostics may additionally contain D2H/event/serialization/HTTP RTT,
HTTP header parse/body-to-pinned/H2D, SHM control, queue/service intervals, RGB
D2H, and MP4 encoding. The existing `safetensors_parse_ms` key now measures
header-only validation, while `pageable_to_pinned_ms` measures the single raw
body-data-to-pinned copy. These categories must not be summed together.
Cross-machine comparisons use host-local durations and sender-side RTT, not
synchronized wall clocks.

## Scope

This is a trusted-LAN research service without TLS or authentication. It does
not include batching, multiple models, TP/SP, multiple VAE workers, RGB
streaming, WebSockets, CUDA IPC, GPUDirect, Mooncake, ETCD, DLA, dynamic-shape
TensorRT, or a hand-written Jetson Conv3d kernel. FP32 is the
FastVideo-compatible VAE mode; FP16 and TensorRT modes are explicitly
non-reference profiling options. The TensorRT INT8 path quantizes only the 28
main residual Conv3d modules and does not claim visual accuracy.
Terminal job metadata has no TTL in V1 and is retained until server restart;
GPU tensors, caches, pinned slots, and SHM mappings are still released at
terminal state.
