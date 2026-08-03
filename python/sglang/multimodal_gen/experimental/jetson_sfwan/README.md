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

TensorRT layer profiling is a second, more intrusive diagnostic mode. It is
also disabled by default and is never implied by `--enable-profile` or
`--enable-nvtx`. See [TensorRT fine-grained layer profiling](#tensorrt-fine-grained-layer-profiling)
for its dedicated plans, server restrictions, result files, and validity
checks. Use that mode to attribute time to physical TensorRT layers; use it
*disabled* when reporting end-to-end engine latency.

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

The INT8 builder is fail-closed and uses Q/DQ schema v5. Run its short
signature preflight on the Orin before starting the long initial/steady
builds:

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
  --timing-cache "$SFWAN_TRT_DIR/tensorrt_timing_qdq_v5.cache"
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
  --timing-cache "$SFWAN_TRT_DIR/tensorrt_timing_qdq_v5.cache"
```

The command reuses the loaded SGLang Wan VAE to collect deterministic,
speed-only input and output activation scales, exports ONNX opset 19, and
rewrites only the 28 residual-block 3x3x3 Conv3d modules. Each initial/steady
graph has 84 unrolled target calls. TensorRT 10.3 requires FP32 Q/DQ data and
scales, so every v5 call site is:

```text
FP16 activation
  -> Cast FP32 -> input Q(INT8)/DQ(FP32)
  -> Conv(FP32 ONNX semantics, FP32 bias)
  -> output Q(INT8)/DQ(FP32) -> Cast FP16
```

The trailing output Q/DQ is essential: it permits TensorRT to fuse an
INT8-input, INT8-weight, INT8-output Conv instead of legally choosing the
Float/TF32 tactic observed with v4. Feature-cache graph bindings remain FP16
and never pass through Q/DQ.

The primary weight encoding gives every call its own FP32 rank-5 constant,
FP32 per-output-channel scale, signed-INT8 zero point, and Q/DQ. If preflight
shows that a mapped static filter remained non-INT8, the builder retries all
nine signatures with independent, offline-quantized INT8 weight constants
followed by FP32 DQ. Initial and steady must use the same selected encoding;
the manifest records either `fp32_qdq` or `prequantized_int8_dq`.

No weight, bias, scale, or Q/DQ output is shared between call sites.
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
2. extract exactly nine unique Conv signatures from all 168 initial/steady
   call sites and build one complete v5 input/weight/output-Q/DQ probe per
   signature;
3. if the primary weight form fails only to become a static INT8 filter,
   retry all nine signatures using prequantized INT8-weight DQ; otherwise
   stop before any full VAE plan;
4. build `initial_int8_qdq_v5.plan` with detailed Inspector data and audit all
   84 target calls;
5. repeat for `steady_int8_qdq_v5.plan`;
6. use those same audited detailed plans at runtime, and build/reuse only the
   two FP16 control plans. INT8 is never rebuilt as an unauditable NONE plan.

`--resume` records `built`, `audit_failed`, and `audit_passed` separately in
`build_state_v5.json`. A restart verifies the source ONNX hash, plan hash, Q/DQ
schema, and profiling verbosity before re-auditing or reusing a stage. Existing
legacy FP16 ONNX/plans are reused only when their legacy build identity matches
the model path, resolved checkpoint, shape, and seed; they are not relabelled
or used as Q/DQ input. Old INT8 plans are never adopted because their Q/DQ graph
may differ. V5 INT8 ONNX, plans, state, audit, scales, probes, and timing cache
all use new names and never overwrite v2/v3/v4 artifacts. Matching v4 input
activation scales and unquantized opset-19 source graphs may be adopted, but
v5 output activation scales are newly collected. Reuse requires matching model,
resolved checkpoint, shape, seed, SM, and opset identities. A build returns only
a candidate timing cache. The candidate is atomically committed through
TensorRT's `IBuilderConfig` API only after that probe or full plan passes tactic
and cache-binding audit; a failed build leaves the stable cache untouched.

A build is accepted only if:

- each graph has 28 logical modules and 84 Conv calls, with 84 independent
  input Q/DQ paths, 84 weight Q/DQ (or INT8-weight DQ) paths, 84 output Q/DQ
  paths, and 84 final FP32-to-FP16 casts;
- all Q/DQ axes, FP32 scale/weight/bias dtypes, rank-5 weight shapes, positive
  scales, zero points, call-site shapes, Cast directions, and Conv attributes
  pass structural audit;
- all nine unique initial/steady Conv signatures pass preflight;
- every full-plan call maps through Inspector `Name` or `Metadata` and proves
  INT8 activation and output, no dynamic filter (`HasDynamicFilter=0`),
  non-empty static INT8 weights, and an
  INT8/IMMA/i8 tactic with no `f16f16`, FP32, or TF32 fallback.
- initial exposes 32 FP16 cache outputs; steady exposes 32 FP16 cache inputs
  and 32 FP16 outputs. Any INT8 cache binding fails the build.

Inspect the fail-closed audit:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["SFWAN_TRT_DIR"]) / "int8_audit_v5.json"
audit = json.loads(path.read_text())
suite = audit["probe_suite"]
print("passed:", audit["passed"])
print("weight encoding:", audit["weight_encoding"])
print("signatures:", suite["probed_signature_count"], "/", suite["signature_count"])
print("source calls:", suite["source_call_site_counts"])
print("feature cache:", audit["feature_cache"])
for kind in ("initial", "steady"):
    tactic = audit.get("tactics", {}).get(kind)
    print(kind, None if tactic is None else {
        "passed": tactic["passed"],
        "mapped": tactic["mapped_count"],
        "non_int8": tactic["non_int8_call_sites"],
        "output_not_int8": tactic["output_not_int8_call_sites"],
        "plan_sha256": tactic["plan_sha256"],
    })
PY

sha256sum "$SFWAN_TRT_DIR"/*.plan
```

Detailed audit plans and Inspector JSON are retained in the same directory;
there is no second full rebuild:

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine="$SFWAN_TRT_DIR/initial_int8_qdq_v5.plan" \
  --dumpLayerInfo \
  --profilingVerbosity=detailed

/usr/src/tensorrt/bin/trtexec \
  --loadEngine="$SFWAN_TRT_DIR/steady_int8_qdq_v5.plan" \
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

## TensorRT fine-grained layer profiling

This feature is an opt-in, temporary performance-attribution path. It answers
which *physical TensorRT layers* consume time; it does not change the Q/DQ v5
topology, the 32 FP16 feature-cache bindings, the two cache banks, FCFS, HTTP
ingress, or chunk order. Its callbacks and explicit profile reporting perturb
execution, so its timings are diagnostic. For publication-quality latency,
restart the same server without `--enable-trt-layer-profile` and use
`trt_engine_cuda_ms`.

### Profile-plan artifacts

`vae_trt_profile_build.py` creates an isolated, resumable profile-plan family
inside an existing validated engine directory:

```text
initial_fp16_layer_profile.plan
steady_fp16_layer_profile.plan
initial_fp16_layer_profile_inspector.json
steady_fp16_layer_profile_inspector.json
trt_layer_profile_manifest.json
trt_layer_profile_build_state.json
trt_layer_profile_timing.cache
```

The FP16 controls are built with `ProfilingVerbosity.DETAILED` directly from
`initial_fp16_opset19.onnx` and `steady_fp16_opset19.onnx`: the same unquantized
opset-19 sources that precede the v5 Q/DQ rewrite. They are therefore suitable
controls for the initial/steady graph and cache ABI. The INT8 profile path does
not build a second INT8 engine; it references the already audited detailed v5
plans and verifies their plan hashes against `manifest.json` and
`int8_audit_v5.json`.

The builder resolves the same 28 residual Conv weights in each FP16 source
graph that the v5 rewriter used, records their 84 unrolled ONNX node names, and
binds that map to the FP16 Inspector catalog. Startup fails if all 84 cannot be
mapped. Consequently `target_quantized_conv` is a cross-precision category:
for INT8 it means the audited quantized physical layers; for FP16 it means the
exact unquantized counterparts selected for v5, not every Conv in the graph.

The profile builder never overwrites production plans, production manifests,
the v5 audit, or a production timing cache. Its build-state identity binds the
source-ONNX hashes, plan hashes, TensorRT/CUDA/SM versions, and build options.
`--resume` reuses only a completed, identity-matching stage. A timing-cache
candidate is committed atomically only after plan and I/O validation, so an
interrupted or failed build cannot poison the stable profile cache. Both plans
must expose the original FP16 latent/RGB contract and exactly 32 FP16 cache
bindings.

The conceptual build workflow is:

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_profile_build \
  --engine-dir "$SFWAN_TRT_DIR" \
  --workspace-gib 8 \
  --device-index 0 \
  --timing-cache "$SFWAN_TRT_DIR/trt_layer_profile_timing.cache" \
  --resume
```

Run this on the target Orin/TensorRT stack. The command builds only the missing
FP16 profile stages; existing audited INT8 v5 plans are validated, not rebuilt.

### Dedicated profile server

Fine-grained collection is deliberately restricted to a VAE-only profile
server:

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.server \
  --role vae \
  --model-path "$SFWAN_MODEL" \
  --port 30001 \
  --latent-transport http \
  --vae-precision int8_trt \
  --vae-engine-dir "$SFWAN_TRT_DIR" \
  --enable-profile \
  --enable-trt-layer-profile \
  --output-dir /workspace/results/sfwan-vae-int8-layer-profile
```

The switch is valid only with all of the following:

- `--enable-profile` is also present;
- `--role vae` is used;
- `--vae-precision` is `fp16_trt` or `int8_trt`;
- a validated `--vae-engine-dir` contains
  `trt_layer_profile_manifest.json`;
- TensorRT provides `IProfiler`, context `profiler`,
  `enqueue_emits_profile`, and `report_to_profiler`.

Startup fails if any requirement is missing. While the switch is enabled, the
server accepts only latent jobs whose `source` is `profile`; it rejects ordinary
disaggregated VAE generation so instrumented and uninstrumented measurements
cannot be mixed. It is not supported for monolithic or DiT roles. With the
switch absent, the runtime does not read the profile manifest, import the
layer-profiler helper, attach callbacks, or call `report_to_profiler()`.

For each chunk, the enabled runtime follows:

```text
begin_capture(chunk_index)
  -> execute_async_v3(current PyTorch CUDA stream)
  -> context.report_to_profiler()
  -> report_layer_time(name, milliseconds) callbacks
  -> finish_capture()
```

`context.enqueue_emits_profile` is set to `False`, so the existing asynchronous
enqueue remains explicit and the immediately following report belongs to that
same successful context execution. Chunk 0 must use `initial`; chunks 1--6 must
use `steady`. A false report result, zero callbacks, nested capture, wrong chunk
kind, changed layer order, or changed layer count fails closed and preserves the
diagnostic evidence. TensorRT documents this callback/report workflow in the
[TensorRT 10.3 Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1030/pdf/TensorRT-Developer-Guide.pdf).

### Result files and interpretation

The client requires a separate detailed output path when the server exposes
layer profiling:

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.client profile-vae \
  --server-url http://127.0.0.1:30001 \
  --height 480 \
  --width 832 \
  --num-frames 81 \
  --seed 1024 \
  --warmup 10 \
  --repeat 50 \
  --summary-json /workspace/results/vae-int8-layer-summary.json \
  --trt-layer-profile-json /workspace/results/vae-int8-layer-detail.json
```

The client first checks `GET /v1/engine`. Supplying
`--trt-layer-profile-json` to a normal server, or omitting it for an instrumented
server, is an error before the first request is sent. The detailed path must also
be different from `--summary-json`; otherwise the client fails before submission
instead of allowing the compact summary to overwrite the layer evidence.

The detailed JSON stores environment identity, plan and catalog hashes,
initial/steady physical-layer catalogs, every warmup and measured per-layer
array, per-layer mean/population-standard-deviation/min/max/count, per-chunk
statistics, the pooled steady result, whole-request totals, category totals,
and validation warnings. Warmup arrays are retained for diagnosis but excluded
from all reported means. `--summary-json` stores only compact per-chunk,
initial, pooled-steady, whole-request, and category summaries plus the absolute
path and SHA256 of the detailed artifact.

Physical layers are classified into these fixed categories:

- `target_quantized_conv`;
- `target_qdq_cast_reformat`;
- `non_target_conv`;
- `attention`;
- `upsample_resample`;
- `norm_activation_residual`;
- `cache_layout_copy`;
- `other`.

Classification uses Inspector `Name`, `Metadata`, layer type, the source-ONNX
target map for FP16, and the v5 tactic audit for INT8. Node matching uses
complete-name boundaries so names such as `Conv` and `Conv_1` cannot collide.
A TensorRT 10.3 `ReduceL2`/`Reduce` or fused `Sigmoid`/`Tanh` layer under the
decoder norm/nonlinearity path is classified as norm/activation.  After
attention, upsample, and norm paths have been excluded, exported causal-state
`Slice` and compiler-fused `SlicCast` layers are classified as cache/layout
work.  A generic Reformat still remains `other` unless it has explicit cache
evidence, so ordinary layout conversion is not mislabeled as feature-cache
traffic.
For the audited INT8 v5 plan, TensorRT 10.3 may erase the ONNX Q/DQ name and
emit anonymous `kgen` layout kernels immediately before a target Conv. The
catalog classifies only a contiguous run of at most three recognized
transpose/reshape/slice kernels feeding an audit-mapped INT8 Conv as
`target_qdq_cast_reformat`; the same generated names elsewhere remain
`other`. Compiler `CastCastAddCast`/`CastCastMulCast` kernels and Reformat
nodes tied to a `PWN(.../Add)` residual are classified as
`norm_activation_residual`. This ordered, audit-backed rule is needed for
Orin's compiler backend and does not infer precision from an `int8` substring.
A fused physical layer is timed once even when it maps to several
logical call sites; its time is never divided among those sites. Unreliable
mappings remain `other` instead of being guessed from a name. Each catalog is
bound to the plan SHA and to a hash of the ordered physical-layer records.

Each chunk's compact record contains `layer_times_ms`, `layer_sum_ms`, the
existing event interval as `engine_event_ms`, their coverage ratio, category
totals, layer count, engine kind, and catalog hash. The ratio is a coverage
diagnostic, not a decomposition of the difference: TensorRT callbacks, fused
subgraphs, graph optimization, and profiler perturbation can all make
`layer_sum_ms` differ from the CUDA-event interval.

The result is valid for an optimization decision only when plan/catalog hashes
match, catalogs remain stable, callbacks and arrays are complete, and all 84
target call sites map to corresponding physical layers in both FP16 and INT8
runs. INT8 mappings additionally require the v5 tactic audit. A materially non-unit
coverage ratio (outside `[0.90, 1.10]`) marks coverage incomplete. More than
10% in `other` marks the
classification insufficient for deciding which operator family to optimize.
The detailed JSON is written before a validation failure is returned so the
failed run remains inspectable.

Use the category percentages as decision gates:

- Q/DQ/Cast/Reformat at least 10%: reduce or fuse quantization boundaries;
- unquantized Conv at least 10%: expand Conv coverage after signature probes;
- steady cache work at least 20% and DRAM-bound: evaluate an INT8 cache ABI;
- attention or upsample dominant: optimize that kernel family instead.

Finally, restart without `--enable-trt-layer-profile` and repeat the same
warmup/repeat matrix before reporting FP16-vs-INT8 latency. Layer-profile time
and uninstrumented `trt_engine_cuda_ms` answer different questions and must not
be put in the same latency table.

## Experimental INT8 boundary fusion (`fusion_v1`)

The complete Jetson command sequence, including production baselines, plugin
build, foreground probes, resumable nohup build, artifact audit, layer profile,
and final comparison, is in [FUSION_V1_JETSON_RUNBOOK.md](FUSION_V1_JETSON_RUNBOOK.md).

`fusion_v1` is an opt-in Orin experiment. The default remains
`--vae-trt-variant baseline`; that branch neither imports the fusion helpers,
reads `fusion_v1/`, nor loads a plugin shared library. The experiment preserves
the Q/DQ-v5 INT8 Conv tactics and the external 32-binding FP16 feature-cache
ABI, while replacing expensive boundary work around probe-approved call sites:

```text
FP16 current + optional FP16 history
  -> SfWanCausalPackQuantPlugin
       causal concat + pad + INT8 quantize + CDHW32 pack
       optional second FP16 output: cache update
  -> existing audited INT8 Conv3d
  -> SfWanInt8EpiloguePlugin
       conv1: dequantize + RMSNorm + SiLU -> FP16
       conv2: dequantize + residual add -> FP16

If TensorRT rejects the mixed INT8/FP16 outputs:
  FP16 current + optional FP16 history
    -> SfWanCacheUpdatePlugin -> unchanged FP16 cache_out binding
```

The probe tries the dual-output pack/cache plugin first and independently tests
the split cache-update plugin as a TensorRT 10.3 fallback. It selects the
fastest candidate that passes correctness and tactic checks for each signature.
With the dual-output form, cache work is co-resident in the physical pack layer,
and one CUDA kernel writes both outputs. The layer profiler therefore
attributes that time to `fused_input_pack_quant` rather
than counting it twice as `fused_cache_update`. Cache storage remains FP16 and
the two runtime banks still alternate `A -> B -> A`. No new Conv is quantized.

### Isolated artifacts and fail-closed stages

All experimental files live under `$SFWAN_TRT_DIR/fusion_v1/`; v5 ONNX, plans,
audit, manifest, build state, and timing cache are never overwritten:

```text
fusion_v1/
  libsfwan_vae_trt_fusion.so
  plugin_manifest.json
  fusion_analysis_v1.json
  fusion_probe_v1.json
  initial_int8_fusion_v1.onnx
  steady_int8_fusion_v1.onnx
  initial_int8_fusion_v1.plan
  steady_int8_fusion_v1.plan
  initial_int8_fusion_v1_inspector.json
  steady_int8_fusion_v1_inspector.json
  fusion_audit_v1.json
  fusion_manifest.json
  fusion_build_state.json
  fusion_timing.cache
```

The builder exposes `analyze`, `probe`, and `build`. It requires every
`decoder.up_blocks.3` signature to pass. Other signatures are fused only when
their probe passes; otherwise they retain the v5 boundary. The six probe
schemes compare the same final Conv, epilogue, and cache-update outputs:

1. Q/DQ-v5 baseline;
2. input concat/pad/quantize/CDHW32 pack fusion;
3. input fusion with a mixed INT8/FP16 dual cache output;
4. input fusion plus a separate cache-update plugin fallback;
5. full input/cache/epilogue fusion using the dual-output plugin;
6. full input/cache/epilogue fusion using the split fallback.

Each signature must retain a static INT8 weight and a real INT8/IMMA tactic,
must have no Reformat between pack and Conv or Conv and epilogue, and must
preserve the FP16 cache update exactly. Input and input+cache candidates must
be at least 10% faster than the baseline; the epilogue must add at least 5%
relative improvement. A failed required probe prevents a full build.

Build the SM87 plugin inside the Jetson container:

```bash
export SGLANG_SRC=/workspace/sglang
export SFWAN_TRT_DIR=/workspace/engines/sfwan-vae-trt-sm87-iofix
export SFWAN_FUSION_DIR="$SFWAN_TRT_DIR/fusion_v1"
export SFWAN_FUSION_BUILD=/workspace/build/sfwan-vae-trt-fusion-sm87

cmake \
  -S "$SGLANG_SRC/python/sglang/multimodal_gen/experimental/jetson_sfwan/trt_plugins" \
  -B "$SFWAN_FUSION_BUILD" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSFWAN_CUDA_ARCHITECTURES=87 \
  -DCMAKE_INSTALL_PREFIX="$SFWAN_FUSION_DIR"
cmake --build "$SFWAN_FUSION_BUILD" --parallel 4
cmake --install "$SFWAN_FUSION_BUILD"
```

Run graph analysis and the required on-device micro-probes in the foreground:

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_fusion_build \
  --engine-dir "$SFWAN_TRT_DIR" \
  --stage analyze \
  --focus-module-prefix decoder.up_blocks.3 \
  --resume

python -m sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_fusion_build \
  --engine-dir "$SFWAN_TRT_DIR" \
  --stage probe \
  --focus-module-prefix decoder.up_blocks.3 \
  --probe-warmup 20 \
  --probe-repeat 100 \
  --workspace-gib 8 \
  --resume \
  --preflight-only
```

Only after `fusion_probe_v1.json` reports `passed: true`, run the expensive
initial/steady build. `--resume` uses SHA-bound stage state; the stable timing
cache is atomically replaced only after both full-plan audits pass:

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_fusion_build \
  --engine-dir "$SFWAN_TRT_DIR" \
  --stage build \
  --focus-module-prefix decoder.up_blocks.3 \
  --probe-warmup 20 \
  --probe-repeat 100 \
  --workspace-gib 8 \
  --resume
```

Start the experiment only by adding `--vae-trt-variant fusion_v1`:

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.server \
  --role vae \
  --model-path "$SFWAN_MODEL" \
  --vae-precision int8_trt \
  --vae-engine-dir "$SFWAN_TRT_DIR" \
  --vae-trt-variant fusion_v1 \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-profile
```

`GET /v1/engine` reports the variant, plugin SHA, plan SHAs, per-engine fusion
counts, cache-bank bytes, and fusion-audit state. Missing or mismatched plugin,
creator, plan, ONNX, Inspector, probe, audit, timing-cache, or manifest hashes
cause startup to fail; there is no silent baseline fallback.

### Production comparison

Every `profile-vae` summary now contains `measurement_context`: shape, seed,
warmup/repeat, measured count, precision/variant, plan and plugin SHA, layer
profile state, GPU/SM, CUDA, TensorRT, and Torch versions. The offline tool
rejects different requests, devices, TensorRT stacks, diagnostic-profile runs,
or mislabeled backends:

```bash
python -m sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_perf_compare \
  --fp32-json /workspace/results/fp32-production.json \
  --fp16-trt-json /workspace/results/fp16-trt-production.json \
  --int8-v5-json /workspace/results/int8-v5-production.json \
  --int8-fusion-v1-json /workspace/results/int8-fusion-v1-production.json \
  --output-json /workspace/results/vae-production-comparison.json \
  --output-markdown /workspace/results/vae-production-comparison.md
```

Fine-grained profile schema v2 adds `fused_input_pack_quant`,
`fused_cache_update`, `fused_conv1_norm_silu`, and
`fused_conv2_residual`. The profiler accepts the already-audited fusion plan;
it does not build another INT8 plan. As before, layer callbacks are diagnostic.
Restart without `--enable-trt-layer-profile` for production latency.

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
