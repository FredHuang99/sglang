# Wan2.2 Disaggregation 放置组合测试报告

## 1. 概述

本报告基于 `wan_memory_calculator.py` 计算器和 `test_wan_placement.sh` 测试脚本，对 Wan2.2 视频生成模型的 disaggregation（分离式部署）架构进行详细的放置组合测试。

### 1.1 测试目标
- 评估不同组件放置方案下的传输延迟
- 识别最优放置配置
- 为生产环境部署提供参考依据

### 1.2 测试约束
- **DiT (Denoiser)**: 必须放置在 GPU 上
- **Encoder**: 可放置在 CPU 或 GPU
- **Decoder (VAE)**: 可放置在 CPU 或 GPU

### 1.3 测试范围说明

> **重要声明**：本测试聚焦于 **跨组件数据传输开销**，即 disaggregation 架构中各组件之间的数据传输延迟。

**本测试包含的内容**：
- Encoder → Denoiser 传输: prompt_embeds + latents (+ image_latent 仅 TI2V)
- Denoiser → Decoder 传输: denoised_latents
- 不同放置位置（CPU/GPU/跨节点）对传输延迟的影响

**本测试不包含的内容**：
- DiT 内部的计算开销（如 Transformer 层计算）
- DiT 内部的 Sequence Parallel (SP) AllReduce 通信开销
- VAE 内部的编解码计算开销
- 各组件自身的显存占用和计算时间

**原因**：DiT 内部的 SP 并行通信是 DiT 模型的内部实现细节，无论 DiT 部署在何处，这部分开销都已包含在 DiT 的计算时间内。本测试的目标是比较不同放置方案下的**传输效率差异**，为部署决策提供依据。

---

## 2. 模型配置

### 2.1 Wan2.2-TI2V-5B

| 参数 | 值 |
|------|-----|
| 模型类型 | Text + Image to Video |
| DiT dim | 3072 (24 heads × 128) |
| DiT num_layers | 30 |
| DiT num_heads | 24 |
| DiT ffn_dim | 14336 |
| VAE z_dim | 48 |
| VAE stride | (4, 16, 16) |
| text_seq_length | 512 |
| text_dim | 4096 |

### 2.2 Wan2.2-T2V-A14B

| 参数 | 值 |
|------|-----|
| 模型类型 | Text to Video |
| DiT dim | 5120 (40 heads × 128) |
| DiT num_layers | 40 |
| DiT num_heads | 40 |
| DiT ffn_dim | 13824 |
| VAE z_dim | 16 |
| VAE stride | (4, 8, 8) |
| text_seq_length | 512 |
| text_dim | 4096 |

---

## 3. 带宽配置

| 互联类型 | 带宽 | 说明 |
|----------|------|------|
| NVLink | 450 GB/s | GPU 间高速互联 (单向) |
| PCIe 5.0 | 64 GB/s | CPU-GPU 互联 (单工) |
| RDMA | 20 GB/s | 跨节点高速网络 |

### 带宽选择逻辑
- **跨节点**: RDMA (20 GB/s)
- **同节点 GPU→GPU**: NVLink (450 GB/s)
- **同节点涉及 CPU**: PCIe (64 GB/s)

---

## 4. 测试配置

### 4.1 输入参数
| 参数 | 值 |
|------|-----|
| 批量大小 | 1 |
| 视频帧数 | 81 |
| 分辨率 | 720 × 1280 |

### 4.2 GPU 分配
| Role | GPU 数量 |
|------|----------|
| Encoder | 1 |
| DiT (Denoiser) | 6 |
| Decoder (VAE) | 1 |
| **总计** | **8** |

---

## 5. 数据传输分析

### 5.1 不同配置的数据传输量对比

| 配置 | latents 大小 | Enc→Den 总大小 | Den→Dec 大小 | Latent Tokens |
|------|-------------|----------------|--------------|--------------|
| 81帧, 720×1280, TI2V | 0.0017 GB | 0.0107 GB | 0.0017 GB | 18,480 |
| 121帧, 720×1280, TI2V | 0.0024 GB | 0.0114 GB | 0.0024 GB | 27,280 |
| 81帧, 720×1280, T2V | 0.0023 GB | 0.0112 GB | 0.0023 GB | 75,600 |
| 81帧, 480×832, T2V | 0.0010 GB | 0.0099 GB | 0.0010 GB | 32,760 |

**结论**: 不同配置下传输数据量差异显著，最大相差约 **2.4 倍**。

### 5.2 固定配置 (81帧, 720×1280) 的详细数据

#### Wan2.2-TI2V-5B

| Role | 传入数据 | 传出数据 |
|------|----------|----------|
| Encoder | 0.0001 GB | 0.0107 GB |
| Denoiser | 0.0062 GB | 0.0017 GB |
| Decoder | 0.0017 GB | 0.0000 GB |

**传输内容**:
- Encoder→Denoiser: prompt_embeds + latents + image_latent + overhead (≈10.7 MB)
- Denoiser→Decoder: denoised_latents (≈1.7 MB)

#### Wan2.2-T2V-A14B

| Role | 传入数据 | 传出数据 |
|------|----------|----------|
| Encoder | 0.0001 GB | 0.0112 GB |
| Denoiser | 0.0067 GB | 0.0023 GB |
| Decoder | 0.0023 GB | 0.0000 GB |

### 5.3 Latent Token 计算公式

```
latent_frames = num_frames // vae_stride_t + 1
latent_h = height // vae_stride_h
latent_w = width // vae_stride_w
tokens_per_frame = (latent_h // patch_h) × (latent_w // patch_w)
total_tokens = latent_frames × tokens_per_frame
大小 = total_tokens × vae_z_dim × 2 / 1024³
```

#### 示例计算

**81帧, 720×1280, TI2V**:
```
latent_frames = 81 // 4 + 1 = 21
latent_h = 720 // 16 = 45
latent_w = 1280 // 16 = 80
tokens = 21 × (45//2) × (80//2) × 48 = 21 × 22 × 40 × 48 = 18,480 tokens
大小 = 18,480 × 48 × 2 / 1024³ ≈ 0.0017 GB
```

**121帧, 720×1280, TI2V**:
```
latent_frames = 121 // 4 + 1 = 31
latent_h = 720 // 16 = 45
latent_w = 1280 // 16 = 80
tokens = 31 × (45//2) × (80//2) × 48 = 31 × 22 × 40 × 48 = 27,280 tokens
大小 = 27,280 × 48 × 2 / 1024³ ≈ 0.0024 GB
```

**81帧, 720×1280, T2V**:
```
latent_frames = 81 // 4 + 1 = 21
latent_h = 720 // 8 = 90
latent_w = 1280 // 8 = 160
tokens = 21 × (90//2) × (160//2) × 16 = 21 × 45 × 80 × 16 = 75,600 tokens
大小 = 75,600 × 16 × 2 / 1024³ ≈ 0.0023 GB
```

**81帧, 480×832, T2V**:
```
latent_frames = 81 // 4 + 1 = 21
latent_h = 480 // 8 = 60
latent_w = 832 // 8 = 104
tokens = 21 × (60//2) × (104//2) × 16 = 21 × 30 × 52 × 16 = 32,760 tokens
大小 = 32,760 × 16 × 2 / 1024³ ≈ 0.0010 GB
```

---

## 6. 单机八卡测试结果

### 6.1 测试场景说明
所有组件在同一节点内（Node 0），Encoder 和 Decoder 可选 CPU 或 GPU，DiT 必须在 GPU。

### 6.2 测试结果

| 组合 | Encoder | DiT | VAE | Enc→Den 路径 | Enc→Den 延迟 | Den→Dec 延迟 | 总延迟 |
|------|---------|-----|-----|--------------|--------------|--------------|--------|
| T1 | CPU | GPU | CPU | 同节点 PCIe | 0.1665 ms | 0.0258 ms | **0.1923 ms** |
| T2 | CPU | GPU | GPU | 同节点 PCIe | 0.1665 ms | 0.0037 ms | **0.1702 ms** |
| T3 | GPU | GPU | CPU | 同节点 NVLink | 0.0237 ms | 0.0258 ms | **0.0495 ms** |
| T4 | GPU | GPU | GPU | 同节点 NVLink | 0.0237 ms | 0.0037 ms | **0.0274 ms** |

### 6.3 路径类型分析

| 路径 | 传输类型 | 带宽 | 触发条件 |
|------|----------|------|----------|
| Enc→Den (CPU→GPU) | PCIe | 64 GB/s | Encoder 在 CPU |
| Enc→Den (GPU→GPU) | NVLink | 450 GB/s | Encoder 在 GPU |
| Den→Dec (GPU→CPU) | PCIe | 64 GB/s | VAE 在 CPU |
| Den→Dec (GPU→GPU) | NVLink | 450 GB/s | VAE 在 GPU |

### 6.4 延迟分解

| 组合 | Enc→Den | Den→Dec | 总延迟 |
|------|---------|---------|--------|
| T4 (全 GPU) | 0.0237 ms | 0.0037 ms | **0.0274 ms** |
| T3 (Enc GPU, Dec CPU) | 0.0237 ms | 0.0258 ms | **0.0495 ms** |
| T2 (Enc CPU, Dec GPU) | 0.1665 ms | 0.0037 ms | **0.1702 ms** |
| T1 (全 CPU for Enc/Dec) | 0.1665 ms | 0.0258 ms | **0.1923 ms** |

### 6.5 最优配置

**推荐: T4 (Encoder GPU + DiT GPU + Decoder GPU)**

- 总延迟: 0.0274 ms
- 比 T1 快 7 倍
- 比 T2 快 6 倍
- 比 T3 快 1.8 倍

---

## 7. 双机跨节点测试结果

### 7.1 场景分类

| 场景 | Encoder 节点 | DiT 节点 | VAE 节点 | 描述 |
|------|-------------|----------|----------|------|
| S1 | Node 1 | Node 0 | Node 0 | DiT/VAE 在 Node 0, Encoder 在 Node 1 |
| S2 | Node 0 | Node 0 | Node 1 | Encoder/DiT 在 Node 0, VAE 在 Node 1 |
| S3 | Node 0 | Node 1 | Node 0 | Encoder/VAE 在 Node 0, DiT 在 Node 1 |
| S4 | Node 0 | Node 1 | Node 1 | Encoder 在 Node 0, DiT/VAE 在 Node 1 |
| S5 | Node 1 | Node 0 | Node 1 | Encoder 在 Node 1, DiT 在 Node 0, VAE 在 Node 1 |
| S6 | Node 1 | Node 1 | Node 0 | Encoder/DiT 在 Node 1, VAE 在 Node 0 |

### 7.2 详细测试结果

#### S1: Enc@Node1, DiT@Node0, VAE@Node0

| 组合 | Enc | VAE | Enc→Den 路径 | Enc→Den 延迟 | Den→Dec 延迟 | 总延迟 |
|------|-----|-----|--------------|--------------|--------------|--------|
| S1-1 | CPU | CPU | 跨节点 RDMA | 0.5329 ms | 0.0258 ms | **0.5587 ms** |
| S1-2 | CPU | GPU | 跨节点 RDMA | 0.5329 ms | 0.0037 ms | **0.5366 ms** |
| S1-3 | GPU | CPU | 跨节点 RDMA | 0.5329 ms | 0.0258 ms | **0.5587 ms** |
| S1-4 | GPU | GPU | 跨节点 RDMA | 0.5329 ms | 0.0037 ms | **0.5366 ms** |

#### S2: Enc@Node0, DiT@Node0, VAE@Node1

| 组合 | Enc | VAE | Enc→Den 路径 | Enc→Den 延迟 | Den→Dec 延迟 | 总延迟 |
|------|-----|-----|--------------|--------------|--------------|--------|
| S2-5 | CPU | CPU | 同节点 PCIe | 0.1665 ms | 0.0826 ms | **0.2491 ms** |
| S2-6 | CPU | GPU | 同节点 PCIe | 0.1665 ms | 0.0826 ms | **0.2491 ms** |
| S2-7 | GPU | CPU | 同节点 NVLink | 0.0237 ms | 0.0826 ms | **0.1063 ms** |
| S2-8 | GPU | GPU | 同节点 NVLink | 0.0237 ms | 0.0826 ms | **0.1063 ms** |

#### S3: Enc@Node0, DiT@Node1, VAE@Node0

| 组合 | Enc | VAE | Enc→Den 路径 | Enc→Den 延迟 | Den→Dec 延迟 | 总延迟 |
|------|-----|-----|--------------|--------------|--------------|--------|
| S3-9 | CPU | CPU | 跨节点 RDMA | 0.5329 ms | 0.0826 ms | **0.6155 ms** |
| S3-10 | CPU | GPU | 跨节点 RDMA | 0.5329 ms | 0.0826 ms | **0.6155 ms** |
| S3-11 | GPU | CPU | 跨节点 RDMA | 0.5329 ms | 0.0826 ms | **0.6155 ms** |
| S3-12 | GPU | GPU | 跨节点 RDMA | 0.5329 ms | 0.0826 ms | **0.6155 ms** |

#### S4: Enc@Node0, DiT@Node1, VAE@Node1

| 组合 | Enc | VAE | Enc→Den 路径 | Enc→Den 延迟 | Den→Dec 延迟 | 总延迟 |
|------|-----|-----|--------------|--------------|--------------|--------|
| S4-13 | CPU | CPU | 跨节点 RDMA | 0.5329 ms | 0.0258 ms | **0.5587 ms** |
| S4-14 | CPU | GPU | 跨节点 RDMA | 0.5329 ms | 0.0037 ms | **0.5366 ms** |
| S4-15 | GPU | CPU | 跨节点 RDMA | 0.5329 ms | 0.0258 ms | **0.5587 ms** |
| S4-16 | GPU | GPU | 跨节点 RDMA | 0.5329 ms | 0.0037 ms | **0.5366 ms** |

#### S5: Enc@Node1, DiT@Node0, VAE@Node1

| 组合 | Enc | VAE | Enc→Den 路径 | Enc→Den 延迟 | Den→Dec 延迟 | 总延迟 |
|------|-----|-----|--------------|--------------|--------------|--------|
| S5-17 | CPU | CPU | 跨节点 RDMA | 0.5329 ms | 0.0826 ms | **0.6155 ms** |
| S5-18 | CPU | GPU | 跨节点 RDMA | 0.5329 ms | 0.0826 ms | **0.6155 ms** |
| S5-19 | GPU | CPU | 跨节点 RDMA | 0.5329 ms | 0.0826 ms | **0.6155 ms** |
| S5-20 | GPU | GPU | 跨节点 RDMA | 0.5329 ms | 0.0826 ms | **0.6155 ms** |

#### S6: Enc@Node1, DiT@Node1, VAE@Node0

| 组合 | Enc | VAE | Enc→Den 路径 | Enc→Den 延迟 | Den→Dec 延迟 | 总延迟 |
|------|-----|-----|--------------|--------------|--------------|--------|
| S6-21 | CPU | CPU | 同节点 PCIe | 0.1665 ms | 0.0826 ms | **0.2491 ms** |
| S6-22 | CPU | GPU | 同节点 PCIe | 0.1665 ms | 0.0826 ms | **0.2491 ms** |
| S6-23 | GPU | CPU | 同节点 NVLink | 0.0237 ms | 0.0826 ms | **0.1063 ms** |
| S6-24 | GPU | GPU | 同节点 NVLink | 0.0237 ms | 0.0826 ms | **0.1063 ms** |

---

## 8. 性能排名

### 8.1 单机场景排名

| 排名 | 组合 | 配置 | 总延迟 |
|------|------|------|--------|
| 🥇 1 | T4 | Enc GPU, DiT GPU, Dec GPU | 0.0274 ms |
| 🥈 2 | T3 | Enc GPU, DiT GPU, Dec CPU | 0.0495 ms |
| 🥉 3 | T2 | Enc CPU, DiT GPU, Dec GPU | 0.1702 ms |
| 4 | T1 | Enc CPU, DiT GPU, Dec CPU | 0.1923 ms |

### 8.2 双机场景排名

| 排名 | 组合 | 配置 | 总延迟 |
|------|------|------|--------|
| 🥇 1 | S2-7/8, S6-23/24 | Enc GPU, Dec CPU | 0.1063 ms |
| 🥈 2 | S2-5/6, S6-21/22 | Enc CPU | 0.2491 ms |
| 🥉 3 | S1-2/4, S4-14/16 | 跨节点 Enc→Den, Dec GPU | 0.5366 ms |
| 4 | S1-1/3, S4-13/15 | 跨节点 Enc→Den, Dec CPU | 0.5587 ms |
| 5 | S3-x, S5-x | 跨节点 Den→Dec | 0.6155 ms |

---

## 9. 结论与建议

### 9.1 关键发现

1. **传输开销占比极小**
   - 单机最优配置传输延迟仅 0.0274 ms
   - 双机跨节点最大延迟 0.6155 ms
   - 相比 97s 的 DiT 计算时间，传输开销可忽略不计

2. **带宽影响显著**
   - NVLink (450 GB/s): ~0.024 ms
   - PCIe (64 GB/s): ~0.167 ms (~7 倍差异)
   - RDMA (20 GB/s): ~0.533 ms (~22 倍差异)

3. **放置策略建议**
   - 优先保证 Encoder 和 Decoder 在 GPU 上使用 NVLink
   - 跨节点传输应尽量避免，必须跨节点时选择 GPU→GPU 路径

### 9.2 部署建议

| 场景 | 推荐配置 | 理由 |
|------|----------|------|
| 单机 8 GPU | T4 (全 GPU) | 最低延迟 0.0274 ms |
| 单机资源受限 | T2 (Enc CPU, Dec GPU) | 延迟可接受 0.1702 ms |
| 双机优先 | S2-7/8 或 S6-23/24 | Den→Dec 同节点，延迟仅 0.1063 ms |
| 双机 Budget | S2-5/6 或 S6-21/22 | Enc→Den 同节点，延迟 0.2491 ms |

### 9.3 传输缓冲池规划 (基于不同配置)

**注意**: 传输数据量随配置变化，缓冲池规划需根据实际业务场景选择合适的池大小。

| 配置 | Role 对 | 数据量 | 池大小 | 并发数 |
|------|---------|--------|--------|--------|
| 81帧, 720×1280, TI2V | Enc→Den | 10.91 MB | 256 MB | 23 |
| 81帧, 720×1280, TI2V | Den→Dec | 1.69 MB | 256 MB | 151 |
| 121帧, 720×1280, TI2V | Enc→Den | 11.72 MB | 256 MB | 21 |
| 121帧, 720×1280, TI2V | Den→Dec | 2.50 MB | 256 MB | 102 |
| 81帧, 720×1280, T2V | Enc→Den | 11.43 MB | 256 MB | 22 |
| 81帧, 720×1280, T2V | Den→Dec | 2.31 MB | 256 MB | 110 |
| 81帧, 480×832, T2V | Enc→Den | 10.12 MB | 256 MB | 25 |
| 81帧, 480×832, T2V | Den→Dec | 1.00 MB | 256 MB | 256 |

**建议**: 池大小 256 MB 可满足大多数场景，81帧配置下可支持 20+ 并发请求。

---

## 10. 附录

### 10.1 测试命令

```bash
# 单机测试
bash test_wan_placement.sh

# 计算器详细输出
python wan_memory_calculator.py --model all --disagg
```

### 10.2 传输数据格式

**固定大小数据**:
- prompt_embeds: [1, 512, 4096] ≈ 4 MB (固定)
- overhead: ~5 MB (固定)

**变化数据**:
- latents: 取决于帧数、分辨率、模型
- image_latent: ~0.1 MB (仅 TI2V)

**Encoder → Denoiser**:
| 配置 | latents | 总大小 |
|------|---------|--------|
| 81帧, 720×1280, TI2V | 1.7 MB | 10.7 MB |
| 121帧, 720×1280, TI2V | 2.4 MB | 11.4 MB |
| 81帧, 720×1280, T2V | 2.3 MB | 11.2 MB |
| 81帧, 480×832, T2V | 1.0 MB | 9.9 MB |

**Denoiser → Decoder**:
- denoised_latents 大小与输入 latents 相同

### 10.3 时间占比 (来自 Profile 实测)

| Role | 时间 | 占比 |
|------|------|------|
| Encoder | 2.48 s | 2.3% |
| Denoiser (DiT) | 97.35 s | 89.1% |
| Decoder (VAE) | 9.41 s | 8.6% |
