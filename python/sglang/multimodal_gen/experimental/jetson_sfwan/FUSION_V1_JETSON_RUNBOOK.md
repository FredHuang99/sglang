# SFWan VAE INT8 fusion_v1 Jetson 执行手册

本文只适用于 Jetson AGX Orin SM87、TensorRT 10.3、CUDA 12.9、
480x832、batch size 1。`fusion_v1` 是隔离实验；不带
`--vae-trt-variant fusion_v1` 时仍使用既有 baseline，且不会加载插件。

下面假定：

- 源码位于 `/workspace/sglang`；
- 已验证的 Q/DQ v5 engine 目录为
  `/workspace/engines/sfwan-vae-trt-sm87-iofix`；
- `/workspace/sfwan_env.sh` 能激活当前 Python 环境；
- 模型可由 Hugging Face ID 或本地路径加载。

## 0. 固定环境与公共参数

每个新终端先执行：

```bash
source /workspace/sfwan_env.sh
cd /workspace/sglang

export SGLANG_SRC=/workspace/sglang
export SFWAN_MODEL="${SFWAN_MODEL:-wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers}"
export SFWAN_TRT_DIR=/workspace/engines/sfwan-vae-trt-sm87-iofix
export SFWAN_FUSION_DIR="$SFWAN_TRT_DIR/fusion_v1"
export SFWAN_FUSION_BUILD=/workspace/build/sfwan-vae-trt-fusion-sm87
export SFWAN_RESULTS=/workspace/results/sfwan-fusion-v1
export SFWAN_LOG_DIR=/workspace/logs

mkdir -p \
  "$SFWAN_FUSION_DIR" \
  "$SFWAN_FUSION_BUILD" \
  "$SFWAN_RESULTS" \
  "$SFWAN_LOG_DIR"

python3 - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("SM:", torch.cuda.get_device_capability(0))
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (8, 7)
PY
```

如果模型已经下载到本地，显式覆盖，例如：

```bash
export SFWAN_MODEL=/workspace/models/SFWan2.1-T2V-1.3B-Diffusers
```

## 1. 重测三组 production baseline

三组测试均使用 `warmup=10`、`repeat=50`，必须保持相同功耗模式、频率、
温度条件与后台负载。Server 只开 `--enable-profile`，**不能**加入
`--enable-trt-layer-profile`。每次只运行一个 server；测试完成后在 server
终端按 `Ctrl-C`，再启动下一组。

### 1.1 PyTorch FP32

Server 终端：

```bash
source /workspace/sfwan_env.sh
cd /workspace/sglang
export SFWAN_MODEL="${SFWAN_MODEL:-wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers}"
export SFWAN_RESULTS=/workspace/results/sfwan-fusion-v1

python3 -m sglang.multimodal_gen.experimental.jetson_sfwan.server \
  --role vae \
  --model-path "$SFWAN_MODEL" \
  --vae-precision fp32 \
  --host 0.0.0.0 \
  --port 30000 \
  --output-dir "$SFWAN_RESULTS/fp32-server" \
  --enable-profile
```

Client 终端：

```bash
source /workspace/sfwan_env.sh
cd /workspace/sglang
mkdir -p /workspace/results/sfwan-fusion-v1

python3 -m sglang.multimodal_gen.experimental.jetson_sfwan.client \
  profile-vae \
  --server-url http://127.0.0.1:30000 \
  --height 480 \
  --width 832 \
  --num-frames 81 \
  --fps 16 \
  --seed 1024 \
  --warmup 10 \
  --repeat 50 \
  --summary-json \
    /workspace/results/sfwan-fusion-v1/fp32-production.json
```

### 1.2 TensorRT FP16 baseline

Server 终端：

```bash
source /workspace/sfwan_env.sh
cd /workspace/sglang
export SFWAN_MODEL="${SFWAN_MODEL:-wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers}"
export SFWAN_TRT_DIR=/workspace/engines/sfwan-vae-trt-sm87-iofix
export SFWAN_RESULTS=/workspace/results/sfwan-fusion-v1

python3 -m sglang.multimodal_gen.experimental.jetson_sfwan.server \
  --role vae \
  --model-path "$SFWAN_MODEL" \
  --vae-precision fp16_trt \
  --vae-engine-dir "$SFWAN_TRT_DIR" \
  --vae-trt-variant baseline \
  --host 0.0.0.0 \
  --port 30000 \
  --output-dir "$SFWAN_RESULTS/fp16-trt-server" \
  --enable-profile
```

Client 终端：

```bash
source /workspace/sfwan_env.sh
cd /workspace/sglang

python3 -m sglang.multimodal_gen.experimental.jetson_sfwan.client \
  profile-vae \
  --server-url http://127.0.0.1:30000 \
  --height 480 \
  --width 832 \
  --num-frames 81 \
  --fps 16 \
  --seed 1024 \
  --warmup 10 \
  --repeat 50 \
  --summary-json \
    /workspace/results/sfwan-fusion-v1/fp16-trt-production.json
```

### 1.3 TensorRT INT8 Q/DQ v5 baseline

Server 终端：

```bash
source /workspace/sfwan_env.sh
cd /workspace/sglang
export SFWAN_MODEL="${SFWAN_MODEL:-wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers}"
export SFWAN_TRT_DIR=/workspace/engines/sfwan-vae-trt-sm87-iofix
export SFWAN_RESULTS=/workspace/results/sfwan-fusion-v1

python3 -m sglang.multimodal_gen.experimental.jetson_sfwan.server \
  --role vae \
  --model-path "$SFWAN_MODEL" \
  --vae-precision int8_trt \
  --vae-engine-dir "$SFWAN_TRT_DIR" \
  --vae-trt-variant baseline \
  --host 0.0.0.0 \
  --port 30000 \
  --output-dir "$SFWAN_RESULTS/int8-v5-server" \
  --enable-profile
```

Client 终端：

```bash
source /workspace/sfwan_env.sh
cd /workspace/sglang

python3 -m sglang.multimodal_gen.experimental.jetson_sfwan.client \
  profile-vae \
  --server-url http://127.0.0.1:30000 \
  --height 480 \
  --width 832 \
  --num-frames 81 \
  --fps 16 \
  --seed 1024 \
  --warmup 10 \
  --repeat 50 \
  --summary-json \
    /workspace/results/sfwan-fusion-v1/int8-v5-production.json
```

## 2. 编译并安装 SM87 TensorRT 插件

在 Jetson 容器内执行；不要在 host 执行：

```bash
source /workspace/sfwan_env.sh
cd /workspace/sglang

export SGLANG_SRC=/workspace/sglang
export SFWAN_TRT_DIR=/workspace/engines/sfwan-vae-trt-sm87-iofix
export SFWAN_FUSION_DIR="$SFWAN_TRT_DIR/fusion_v1"
export SFWAN_FUSION_BUILD=/workspace/build/sfwan-vae-trt-fusion-sm87

mkdir -p "$SFWAN_FUSION_DIR" "$SFWAN_FUSION_BUILD"

cmake \
  -S "$SGLANG_SRC/python/sglang/multimodal_gen/experimental/jetson_sfwan/trt_plugins" \
  -B "$SFWAN_FUSION_BUILD" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSFWAN_CUDA_ARCHITECTURES=87 \
  -DCMAKE_INSTALL_PREFIX="$SFWAN_FUSION_DIR"

cmake --build "$SFWAN_FUSION_BUILD" --parallel 4
cmake --install "$SFWAN_FUSION_BUILD"

test -s "$SFWAN_FUSION_DIR/libsfwan_vae_trt_fusion.so"
sha256sum "$SFWAN_FUSION_DIR/libsfwan_vae_trt_fusion.so"
file "$SFWAN_FUSION_DIR/libsfwan_vae_trt_fusion.so"
```

## 3. 前台 analyze 与 micro-probe

三个 stage 的 identity 包含 workspace、warmup、repeat 和 focus prefix；后续命令
必须保持这些参数一致。`fusion_v1` 当前严格要求
`decoder.up_blocks.3`，其他 prefix 会在昂贵构建前直接报错。

### 3.1 图分析

分析器 v2 会使用 SHA 校验后的 `int8_audit_v5.json` 中真实 forward 捕获的
Conv shape，以及 `manifest.json` 中 32 个 FP16 cache binding；ONNX shape
inference 只负责补充信息。它不会重新编译或重新安装上一节的插件。

如果已经运行过旧版分析器，先保留现场。第一次切换到分析器 v2 时不要传
`--resume`，因为新的 identity 明确绑定 analysis schema、v5 audit 和 shape
contract：

```bash
source /workspace/sfwan_env.sh
cd /workspace/sglang
export SFWAN_TRT_DIR=/workspace/engines/sfwan-vae-trt-sm87-iofix
export SFWAN_FUSION_DIR="$SFWAN_TRT_DIR/fusion_v1"

stamp="$(date +%Y%m%d-%H%M%S)"
for artifact in \
  fusion_build_state.json \
  fusion_analysis_v1.json \
  fusion_analysis_v2.json \
  fusion_probe_v1.json
do
  if test -f "$SFWAN_FUSION_DIR/$artifact"; then
    cp -a \
      "$SFWAN_FUSION_DIR/$artifact" \
      "$SFWAN_FUSION_DIR/$artifact.pre-analysis-v2-$stamp"
  fi
done

python3 -m sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_fusion_build \
  --engine-dir "$SFWAN_TRT_DIR" \
  --stage analyze \
  --focus-module-prefix decoder.up_blocks.3 \
  --workspace-gib 8 \
  --probe-warmup 20 \
  --probe-repeat 100
```

检查 v2 图分析结论。initial 和 steady 各应覆盖 `up_blocks.3` 的 18 个
call site，且每个 call site 的 Pad、current/cache shape、cache update 和
epilogue 都必须有证据：

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path(
    "/workspace/engines/sfwan-vae-trt-sm87-iofix/"
    "fusion_v1/fusion_analysis_v2.json"
)
report = json.loads(path.read_text())
print("analysis schema:", report["analysis_schema_version"])
print("passed:", report["passed"])
print("errors:", report["errors"])
print("contract SHA:", report["analysis_contract_sha256"])
print("v5 audit SHA:", report["int8_audit_sha256"])

assert report["analysis_schema_version"] == 2
assert report["passed"] is True, report["errors"]
for kind in ("initial", "steady"):
    graph = report["graphs"][kind]
    focused = [item for item in graph["call_sites"] if item["focused"]]
    print(
        kind,
        "focused=", len(focused),
        "shape inference warning=", graph.get("shape_inference_error"),
    )
    assert graph["analysis_schema_version"] == 2
    assert graph["focused_complete"] is True, graph["errors"]
    assert len(focused) == 18
    for item in focused:
        assert item["pad_resolution"]["resolved"] is True, item
        assert item["eligible_input"] is True, item
        assert item["eligible_cache_update"] is True, item
        assert item["eligible_epilogue"] is True, item
        assert len(item["current_shape"]) == 5, item
        assert len(item["cache_update_shape"]) == 5, item

print("fusion analysis v2: PASS")
PY
```

### 3.2 必须前台通过的 micro-probe

3.1 通过后，后续命令恢复使用 `--resume`；此时它只复用新的 v2 identity：

```bash
python3 -m sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_fusion_build \
  --engine-dir "$SFWAN_TRT_DIR" \
  --stage probe \
  --focus-module-prefix decoder.up_blocks.3 \
  --workspace-gib 8 \
  --probe-warmup 20 \
  --probe-repeat 100 \
  --resume \
  --preflight-only
```

检查 probe 结论：

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path(
    "/workspace/engines/sfwan-vae-trt-sm87-iofix/"
    "fusion_v1/fusion_probe_v1.json"
)
report = json.loads(path.read_text())
print("passed:", report["passed"])
print("errors:", report["errors"])
print("signature count:", report["signature_count"])
print("required signature count:", report["required_signature_count"])
print("selected schemes:")
for signature, result in sorted(report["signatures"].items()):
    print(
        signature,
        result.get("selected_full_scheme"),
        result.get("selected_cache_update_mode"),
        "input gain=", result.get("input_gain_fraction"),
        "cache gain=", result.get("cache_gain_fraction"),
        "epilogue gain=", result.get("epilogue_incremental_gain_fraction"),
    )
assert report["passed"] is True, report["errors"]
PY
```

只有上述断言通过，才进入完整 initial/steady engine 构建。

## 4. nohup 完整构建与断点恢复

### 4.1 启动

```bash
export SFWAN_FUSION_LOG=/workspace/logs/sfwan-fusion-v1-build.log
export SFWAN_FUSION_STATUS=/workspace/logs/sfwan-fusion-v1-build.status
export SFWAN_FUSION_PID=/workspace/logs/sfwan-fusion-v1-build.pid

mkdir -p /workspace/logs
rm -f "$SFWAN_FUSION_STATUS"

nohup bash -lc '
set -o pipefail
source /workspace/sfwan_env.sh
cd /workspace/sglang
python3 -m sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_fusion_build \
  --engine-dir /workspace/engines/sfwan-vae-trt-sm87-iofix \
  --stage build \
  --focus-module-prefix decoder.up_blocks.3 \
  --workspace-gib 8 \
  --probe-warmup 20 \
  --probe-repeat 100 \
  --resume
rc=$?
printf "exit_code=%s\n" "$rc" \
  > /workspace/logs/sfwan-fusion-v1-build.status
exit "$rc"
' > "$SFWAN_FUSION_LOG" 2>&1 < /dev/null &

echo $! | tee "$SFWAN_FUSION_PID"
```

### 4.2 查看进度

```bash
cat /workspace/logs/sfwan-fusion-v1-build.pid
ps -fp "$(cat /workspace/logs/sfwan-fusion-v1-build.pid)" || true
tail -n 100 /workspace/logs/sfwan-fusion-v1-build.log
cat /workspace/logs/sfwan-fusion-v1-build.status 2>/dev/null || true
```

需要实时观察时：

```bash
tail -f /workspace/logs/sfwan-fusion-v1-build.log
```

`tail -f` 只负责显示，按 `Ctrl-C` 不会终止 nohup 构建。

### 4.3 中断后的恢复

先确认旧 PID 已退出：

```bash
ps -fp "$(cat /workspace/logs/sfwan-fusion-v1-build.pid)" || true
```

然后原样重跑 4.1。`--resume` 只复用 SHA identity 完全一致且已经完成的
stage；失败 plan 和未通过 audit 的 timing cache不会被提交为稳定产物。

## 5. 完整 artifact、SHA、tactic 与 cache ABI 校验

构建退出码必须为零：

```bash
cat /workspace/logs/sfwan-fusion-v1-build.status
grep -Fx 'exit_code=0' /workspace/logs/sfwan-fusion-v1-build.status
```

运行代码内同一套 fail-closed manifest 校验：

```bash
source /workspace/sfwan_env.sh
cd /workspace/sglang

python3 - <<'PY'
from pathlib import Path

from sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_fusion import (
    load_fusion_manifest,
    validate_fusion_manifest,
)
from sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_runtime import (
    load_trt_vae_manifest,
)

root = Path("/workspace/engines/sfwan-vae-trt-sm87-iofix")
base = load_trt_vae_manifest(root)
fusion = load_fusion_manifest(root)
validated = validate_fusion_manifest(
    fusion,
    engine_dir=root,
    base_manifest=base,
    verify_hashes=True,
)

assert fusion["required_focus_prefix"] == "decoder.up_blocks.3"
assert fusion["required_focus_complete"] is True
assert fusion["cache"]["tensor_count"] == 32
assert fusion["cache"]["dtype"] == "float16"
assert fusion["audit"]["passed"] is True
assert fusion["probe"]["passed"] is True

print("plugin:", validated["plugin_path"])
print("plugin sha256:", validated["plugin_sha256"])
print(
    "plan sha256:",
    {
        kind: fusion["engines"][kind]["sha256"]
        for kind in ("initial", "steady")
    },
)
print("fusion counts:", fusion["fusion_counts"])
print("per-engine counts:", fusion["per_engine_fusion_counts"])
print("fusion_v1 manifest/cache/tactic audit: PASS")
PY
```

另看状态与 audit 摘要：

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path(
    "/workspace/engines/sfwan-vae-trt-sm87-iofix/fusion_v1"
)
state = json.loads((root / "fusion_build_state.json").read_text())
audit = json.loads((root / "fusion_audit_v1.json").read_text())
print("build stage:", state["stages"]["build"])
print("audit passed:", audit["passed"])
print("audit errors:", audit["errors"])
for kind in ("initial", "steady"):
    record = audit["engines"][kind]
    print(
        kind,
        "INT8 targets=",
        record["int8_target_conv_call_site_count"],
        "input reformat=",
        record["input_reformat_call_sites"],
        "output reformat=",
        record["output_reformat_call_sites"],
    )
    assert record["int8_target_conv_call_site_count"] == 84
    assert record["non_int8_call_sites"] == []
    assert record["fp16_or_tf32_fallback_call_sites"] == []
    assert record["input_reformat_call_sites"] == []
    assert record["output_reformat_call_sites"] == []
PY
```

## 6. fusion_v1 细粒度 profile

这是诊断测量，会扰动延迟。`fusion_v1` 直接使用已审计且带 DETAILED
Inspector 信息的 plan，不另建 INT8 plan。

Server 终端：

```bash
source /workspace/sfwan_env.sh
cd /workspace/sglang
export SFWAN_MODEL="${SFWAN_MODEL:-wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers}"
export SFWAN_TRT_DIR=/workspace/engines/sfwan-vae-trt-sm87-iofix
export SFWAN_RESULTS=/workspace/results/sfwan-fusion-v1

python3 -m sglang.multimodal_gen.experimental.jetson_sfwan.server \
  --role vae \
  --model-path "$SFWAN_MODEL" \
  --vae-precision int8_trt \
  --vae-engine-dir "$SFWAN_TRT_DIR" \
  --vae-trt-variant fusion_v1 \
  --host 0.0.0.0 \
  --port 30000 \
  --output-dir "$SFWAN_RESULTS/int8-fusion-layer-server" \
  --enable-profile \
  --enable-trt-layer-profile
```

确认加载的是 fusion variant：

```bash
curl -fsS http://127.0.0.1:30000/v1/engine | python3 -m json.tool
```

Client 终端：

```bash
source /workspace/sfwan_env.sh
cd /workspace/sglang

python3 -m sglang.multimodal_gen.experimental.jetson_sfwan.client \
  profile-vae \
  --server-url http://127.0.0.1:30000 \
  --height 480 \
  --width 832 \
  --num-frames 81 \
  --fps 16 \
  --seed 1024 \
  --warmup 10 \
  --repeat 20 \
  --summary-json \
    /workspace/results/sfwan-fusion-v1/int8-fusion-layer-summary.json \
  --trt-layer-profile-json \
    /workspace/results/sfwan-fusion-v1/int8-fusion-layer-detail.json
```

校验 profile v2 的有效性与类别闭合：

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path(
    "/workspace/results/sfwan-fusion-v1/"
    "int8-fusion-layer-detail.json"
)
detail = json.loads(path.read_text())
validation = detail["validation"]
print(json.dumps(validation, indent=2))
assert detail["schema_version"] == 2
assert validation["valid_for_optimization_decision"] is True
assert validation["catalog_stable"] is True
assert validation["layer_profile_complete"] is True

print("whole request categories (ms):")
print(
    json.dumps(
        {
            name: value["mean_ms"]
            for name, value in detail["statistics"]["whole_request"][
                "categories"
            ].items()
        },
        indent=2,
    )
)
PY
```

FP16 baseline、INT8 v5 baseline 的细粒度复测使用同样
`warmup=10/repeat=20`，分别以 `fp16_trt/baseline`、
`int8_trt/baseline` 启动 server，并各自提供独立的
`--summary-json` 与 `--trt-layer-profile-json`。不要把任何 layer-profile
summary 交给 production 比较工具。

## 7. fusion_v1 无插桩 production latency

先停止第 6 节 server，再启动不带 `--enable-trt-layer-profile` 的专用
production server。

Server 终端：

```bash
source /workspace/sfwan_env.sh
cd /workspace/sglang
export SFWAN_MODEL="${SFWAN_MODEL:-wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers}"
export SFWAN_TRT_DIR=/workspace/engines/sfwan-vae-trt-sm87-iofix
export SFWAN_RESULTS=/workspace/results/sfwan-fusion-v1

python3 -m sglang.multimodal_gen.experimental.jetson_sfwan.server \
  --role vae \
  --model-path "$SFWAN_MODEL" \
  --vae-precision int8_trt \
  --vae-engine-dir "$SFWAN_TRT_DIR" \
  --vae-trt-variant fusion_v1 \
  --host 0.0.0.0 \
  --port 30000 \
  --output-dir "$SFWAN_RESULTS/int8-fusion-production-server" \
  --enable-profile
```

Client 终端：

```bash
source /workspace/sfwan_env.sh
cd /workspace/sglang

python3 -m sglang.multimodal_gen.experimental.jetson_sfwan.client \
  profile-vae \
  --server-url http://127.0.0.1:30000 \
  --height 480 \
  --width 832 \
  --num-frames 81 \
  --fps 16 \
  --seed 1024 \
  --warmup 10 \
  --repeat 50 \
  --summary-json \
    /workspace/results/sfwan-fusion-v1/int8-fusion-v1-production.json
```

## 8. 自动生成四组 production 对比

```bash
source /workspace/sfwan_env.sh
cd /workspace/sglang

python3 -m sglang.multimodal_gen.experimental.jetson_sfwan.vae_trt_perf_compare \
  --fp32-json \
    /workspace/results/sfwan-fusion-v1/fp32-production.json \
  --fp16-trt-json \
    /workspace/results/sfwan-fusion-v1/fp16-trt-production.json \
  --int8-v5-json \
    /workspace/results/sfwan-fusion-v1/int8-v5-production.json \
  --int8-fusion-v1-json \
    /workspace/results/sfwan-fusion-v1/int8-fusion-v1-production.json \
  --output-json \
    /workspace/results/sfwan-fusion-v1/vae-production-comparison.json \
  --output-markdown \
    /workspace/results/sfwan-fusion-v1/vae-production-comparison.md \
  --expected-warmup 10 \
  --expected-repeat 50

cat /workspace/results/sfwan-fusion-v1/vae-production-comparison.md
```

比较器会拒绝以下不可比数据：不同 shape/seed/warmup/repeat、不同 GPU 或
CUDA/TensorRT、非 VAE-only server、开启了细粒度 layer profile、错误的
precision/variant，以及缺少 initial/steady plan SHA 的 TRT 结果。

## 9. 决策门槛

同时查看第 6 节细粒度结果和第 8 节 production 结果：

- `fusion_v1` 比 INT8 v5 总时间降低至少 10%：实验候选通过；
- `fusion_v1` 快于 FP16 TRT：可作为后续主线候选；
- 剩余 Q/DQ/Cast/Reformat 大于等于 10%：继续扩大边界融合；
- 未量化 Conv 大于等于 10%：再评估其余 Conv 量化；
- steady cache 相关大于等于 20% 且 Nsight 证明 DRAM-bound：再开发
  INT8 cache ABI；
- attention、upsample 或 norm占主导：转向对应算子，不盲目量化 cache。

无论结果如何，`baseline` 都保持默认且不加载插件；失败实验不能覆盖
Q/DQ v5 的 ONNX、plan、audit、manifest或timing cache。
