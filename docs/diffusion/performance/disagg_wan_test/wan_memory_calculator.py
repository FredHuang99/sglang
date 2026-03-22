#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Wan2.2 视频生成模型 - 中间激活值显存与传输时延计算器
================================================================================

功能说明:
    1. 计算 Wan2.2 系列模型视频生成推理过程中关键中间 tensor 的显存占用
    2. 计算在不同硬件互联方式下的理论传输时延
    3. 支持单体 (Monolithic) 和 disaggregation (分离式) 两种部署架构

支持的模型:
    - Wan2.2-TI2V-5B: Text + Image to Video, 5B 参数, TI2V 专用 VAE (z_dim=48)
    - Wan2.2-T2V-A14B: Text to Video, 14B 参数, T2V VAE (z_dim=16)

适用场景:
    - 评估不同分辨率/帧数下的显存需求
    - 规划 disaggregation 架构的 GPU 分配
    - 评估跨节点传输的带宽需求
    - 优化 transfer buffer pool 大小

使用示例:
    # 默认单体模式 (所有组件在同一 GPU)
    python wan_memory_calculator.py

    # Disaggregation 模式 (组件在不同 GPU)
    python wan_memory_calculator.py --disagg

    # 自定义 GPU 分配
    python wan_memory_calculator.py --disagg --encoder-gpus 1 --denoiser-gpus 4 --decoder-gpus 1

    # Wan2.2-TI2V-5B 配置 (121帧, 704x1280)
    python wan_memory_calculator.py --disagg --num-frames 121 --height 704 --width 1280 --denoiser-gpus 8

    # Wan2.2-T2V-A14B 配置
    python wan_memory_calculator.py --model A14B --disagg --num-frames 81 --height 480 --width 832

"""

# Wan2.2 模型配置对比

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# 带宽配置常量 (GB/s)
# =============================================================================
# 说明:
#   - NVLink: NVIDIA GPU 间高速互联，单向 450 GB/s，双向 900 GB/s
#   - PCIe 5.0 x16: CPU-GPU 互联，64 GB/s 单向
#   - RDMA: 200Gbps 网络，实际可用带宽约 20 GB/s (考虑协议开销)
#   - Ethernet: 25Gbps 网络，实际可用约 0.4 GB/s
#
#            CPU/GPU 架构示意图:
#            ─────────────────
#
#            ┌─────────────┐         ┌─────────────┐
#            │    GPU 0    │◄──NVLink──►│    GPU 1   │
#            │   (450 GB/s)│         │   (450 GB/s)│
#            └──────┬──────┘         └──────┬──────┘
#                   │                        │
#                   │ PCIe                   │ PCIe
#                   │ (64 GB/s)             │ (64 GB/s)
#                   │                        │
#            ┌──────▼────────────────────────▼──────┐
#            │              CPU / Host                │
#            │                                          │
#            │         ┌──────────────┐               │
#            │         │  Network     │◄─── RDMA ────►│
#            │         │  (20 GB/s)  │               │
#            │         └──────────────┘               │
#            └─────────────────────────────────────────┘
#                               │
#                               │ Ethernet (0.4 GB/s)
#                               ▼
#                        ┌─────────────┐
#                        │   另一台    │
#                        │   服务器    │
#                        └─────────────┘

@dataclass
class BandwidthConfig:
    """
    网络/互联带宽配置 (单位: GB/s)

    带宽等级:
        - NVLink: GPU 间高速互联 (~450 GB/s)
        - PCIe: CPU-GPU 互联 (~64 GB/s)
        - RDMA: 跨节点高速网络 (~20 GB/s)
        - Ethernet: 普通网络 (~0.4 GB/s)
    """
    NVLINK_UNIDIRECTIONAL: float = 450.0   # NVLink 单向带宽
    NVLINK_BIDIRECTIONAL: float = 900.0   # NVLink 双向带宽 (参考值)
    PCIE5: float = 64.0                   # PCIe 5.0 x16 单向带宽
    RDMA: float = 20.0                    # 200Gbps RDMA (考虑协议开销)
    NETWORK: float = 0.4                   # 25Gbps 以太网 (实际可用带宽)


@dataclass
class PlacementConfig:
    """
    组件放置位置配置

    用于 disaggregation 模式，指定每个组件放在 CPU 还是 GPU 上，以及在哪台机器上
    - GPU: 组件在 GPU 上，组件间传输使用 NVLink
    - CPU: 组件在 CPU 上，GPU-CPU 传输使用 PCIe
    - 节点编号: 用于判断是否跨节点传输 (跨节点使用 RDMA)
    """
    encoder: str = "gpu"       # "cpu" or "gpu"
    dit: str = "gpu"           # "cpu" or "gpu"
    vae: str = "gpu"           # "cpu" or "gpu"
    encoder_node: int = 0      # Encoder 所在的节点编号
    dit_node: int = 0         # DiT 所在的节点编号
    vae_node: int = 0         # VAE 所在的节点编号


# =============================================================================
# 模型配置定义
# =============================================================================
# 说明:
#   Wan2.2 有两个主要版本:
#   - Wan2.2-TI2V-5B: Text + Image to Video, 5B DiT 参数, TI2V 专用 VAE
#   - Wan2.2-T2V-A14B: Text to Video, 14B DiT 参数, T2V VAE
#
#   DiT (Diffusion Transformer) 配置:
#   ─────────────────────────────────────────────
#   参数          │  Wan2.2-TI2V-5B   │  Wan2.2-T2V-A14B
#   ──────────────┼────────────────────┼────────────────────
#   dim           │     3072 (24×128)  │     5120 (40×128)
#   num_layers    │      30            │      40
#   num_heads     │      24            │      40
#   ffn_dim       │     14336         │     13824
#   patch_size    │   (1, 2, 2)      │   (1, 2, 2)
#
#   VAE 配置 (不同):
#   ────────────────
#   参数          │  Wan2.2-TI2V-5B   │  Wan2.2-T2V-A14B
#   ──────────────┼────────────────────┼────────────────────
#   z_dim         │      48          │      16
#   stride        │  (4, 16, 16)     │  (4, 8, 8)
#
#   显存占用 (来自 profile_wan_ti2v_stages.py 实测):
#   ─────────────────────────────────────────────
#   组件          │  1 GPU    │  4 GPU   │  8 GPU
#   ──────────────┼───────────┼──────────┼──────────
#   Encoder (TP) │  21.4 GB  │  5.5 GB  │  2.9 GB
#   VAE Encoder   │  14.1 GB  │  4.2 GB  │  2.8 GB
#   DiT           │  53.2 GB  │  40.5 GB │  38.2 GB
#   VAE Decoder   │  53.2 GB  │  40.6 GB │  -

@dataclass
class WanModelConfig:
    """
    Wan2.2 模型配置

    属性说明:
        name: 模型名称 (显示用)
        model_type: 模型类型 "TI2V" 或 "T2V"
        dim: Hidden size / 模型维度
        ffn_dim: Feed-Forward Network 维度
        num_layers: Transformer 层数
        num_heads: Attention 头数
        patch_size: (时序, 高度, 宽度) patch 大小
        vae_z_dim: VAE latent 通道数 (TI2V=48, T2V=16)
        vae_stride: VAE 压缩比
        *_memory_gb: 各组件的 GPU 显存占用 (来自实际 profiling)
    """
    name: str                                    # 模型显示名称
    model_type: str = "TI2V"                   # 模型类型
    dim: int = 5120                            # Hidden size (40 heads × 128 dim)
    ffn_dim: int = 13824                      # FFN 维度
    num_layers: int = 40                        # Transformer 层数
    num_heads: int = 40                         # Attention 头数
    patch_size: tuple[int, int, int] = (1, 2, 2)  # DiT patch 大小
    vae_z_dim: int = 48                       # VAE latent 通道数
    vae_stride: tuple[int, int, int] = (4, 16, 16)  # VAE 压缩比 (TI2V)
    # 以下为各组件的 GPU 显存占用 (来自 profile_wan_ti2v_stages.py 的实测数据, 1 GPU)
    encoder_memory_gb: float = 21.4           # Text Encoder ~21GB
    vae_encoder_memory_gb: float = 14.1        # VAE Encoder 显存
    dit_memory_gb: float = 53.2                 # DiT (Transformer) 显存
    vae_decoder_memory_gb: float = 53.2       # VAE Decoder 显存


# Wan2.2-TI2V-5B 配置
# ─────────────────────
# 来源: python/sglang/multimodal_gen/configs/models/dits/wanvideo.py
#       python/sglang/multimodal_gen/configs/pipeline_configs/wan.py
# 适用场景: 文本+图像生成视频, 需要条件图像输入
# 实际模型参数:
#   dim=3072, ffn_dim=14336, num_heads=24, num_layers=30, in_dim=48, out_dim=48
WAN_TI2V_5B_CONFIG = WanModelConfig(
    name="Wan2.2-TI2V-5B",
    model_type="TI2V",
    dim=3072,           # 24 heads × 128 dim
    ffn_dim=14336,
    num_layers=30,
    num_heads=24,
    patch_size=(1, 2, 2),
    vae_z_dim=48,      # TI2V 专用, 更大的 latent dim
    vae_stride=(4, 16, 16),  # TI2V VAE stride
    encoder_memory_gb=21.4,   # 实测数据
    vae_encoder_memory_gb=14.1,
    dit_memory_gb=53.2,       # DiT 最大
    vae_decoder_memory_gb=53.2,
)

# Wan2.2-T2V-A14B 配置
# ─────────────────────
# 来源: python/sglang/multimodal_gen/configs/pipeline_configs/wan.py
# 适用场景: 纯文本生成视频, 无需条件图像
WAN_T2V_A14B_CONFIG = WanModelConfig(
    name="Wan2.2-T2V-A14B",
    model_type="T2V",
    dim=5120,           # 相同
    ffn_dim=13824,      # 相同
    num_layers=40,       # 相同
    num_heads=40,        # 相同
    patch_size=(1, 2, 2),
    vae_z_dim=16,      # T2V VAE, 较小的 latent dim
    vae_stride=(4, 8, 8),  # T2V VAE stride
    encoder_memory_gb=21.4,   # 相同 (Text Encoder 相同)
    vae_encoder_memory_gb=14.1,
    dit_memory_gb=53.2,       # 相同 (DiT 相同)
    vae_decoder_memory_gb=53.2,
)


# =============================================================================
# 通用参数常量
# =============================================================================
# 说明:
#   这些是 Wan2.2 模型的通用参数
#
#   Text Encoder (T5 XXL):
#   - T5 是一个 text-to-text transformer 模型
#   - 输出固定长度 512 个 token
#   - 每个 token 是 4096 维向量
#
#   VAE (Variational Autoencoder):
#   - 用于压缩视频到 latent space
#   - 时序压缩比: 4
#   - 空间压缩比: TI2V=16, T2V=8
#   - Latent channel: TI2V=48, T2V=16

TEXT_SEQ_LENGTH = 512       # T5 XXL 输出 token 数 (固定)
TEXT_DIM = 4096            # T5 hidden size (每个 token 的向量维度)
DTYPE_BYTES = 2             # bf16/fp16 = 2 bytes per element


# =============================================================================
# Disaggregation 配置 (分离式部署)
# =============================================================================
# 说明:
#   Disaggregation 将流水线拆分成三个独立 Role，每个可以运行在不同 GPU 上
#
#   Role 划分:
#   ──────────
#   Role       │ 组件                          │ 时间占比  │ 显存占用
#   ───────────┼───────────────────────────────┼──────────┼──────────
#   Encoder    │ Text Encoder,                 │  ~2.3%   │  ~21 GB
#              │ VAE Encoder                   │           │
#              │ (* Image Encoder 仅 TI2V)     │           │
#   Denoiser  │ DiT (Transformer)             │  ~89.1%  │  ~53 GB  ← 最耗时!
#   Decoder    │ VAE Decoder                   │  ~8.6%   │  ~53 GB
#
#   数据传输:
#   ──────────
#   Encoder → Denoiser: ~7 MB (TI2V) / ~6 MB (T2V)
#                  含: prompt_embeds + latents (+ image_latent 仅 TI2V)
#   Denoiser → Decoder: ~2 MB (denoised latents)
#
#   GPU 分配示例:
#   ────────────
#   方案 A: 1-1-1 (3 GPU)
#   ┌─────────┐  ┌─────────┐  ┌─────────┐
#   │ Encoder │  │ Denoiser│  │ Decoder │
#   │  GPU 0  │  │  GPU 1  │  │  GPU 2  │
#   └─────────┘  └─────────┘  └─────────┘
#
#   方案 B: 1-4-1 (6 GPU)
#   ┌─────────┐  ┌─────────────────┐  ┌─────────┐
#   │ Encoder │  │    Denoiser     │  │ Decoder │
#   │  GPU 0  │  │  GPU 1-4 (SP)  │  │  GPU 5  │
#   └─────────┘  └─────────────────┘  └─────────┘

@dataclass
class DisaggConfig:
    """
    Disaggregation 分离式部署配置

    说明:
        每个 Role 可以有独立的 GPU 数量和并行策略
        Denoiser 通常需要最多 GPU (因为它最耗时)
        Encoder 和 Decoder 通常 1 GPU 就够

    属性:
        enabled: 是否启用 disaggregation 模式
        encoder_gpus: Encoder Role 的 GPU 数量
        denoiser_gpus: Denoiser Role 的 GPU 数量
        decoder_gpus: Decoder Role 的 GPU 数量
        nodes: 物理节点数量 (用于计算跨节点传输)
        intra_node_bw: 节点内互联类型 (nvlink/pcie)
        inter_node_bw: 节点间互联类型 (rdma/network)
        transfer_pool_size_mb: 每个 Role 的传输缓冲区大小
    """
    enabled: bool = False                         # 是否启用 disaggregation
    encoder_gpus: int = 1                        # Encoder Role 的 GPU 数
    denoiser_gpus: int = 1                       # Denoiser Role 的 GPU 数 (通常最多)
    decoder_gpus: int = 1                         # Decoder Role 的 GPU 数
    total_gpus: int = 3                           # 总 GPU 需求
    nodes: int = 1                                # 物理节点数
    intra_node_bw: str = "nvlink"                 # 节点内互联: nvlink 或 pcie
    inter_node_bw: str = "rdma"                   # 节点间互联: rdma 或 network
    transfer_pool_size_mb: int = 256              # 传输缓冲区池大小 (MB)


# =============================================================================
# 用户输入配置
# =============================================================================
# 说明:
#   这些参数决定视频的分辨率和长度
#
#   帧数计算规则:
#   ─────────────
#   Wan2.2 要求帧数是 4k+1 (k 为整数)
#   例如: 1, 5, 9, 17, 21, 41, 81, 121, 161...
#
#   分辨率约束:
#   ──────────
#   - 高度和宽度必须是 16 的倍数 (VAE stride 8 × Patch size 2 = 16)
#   - 最大面积有限制 (根据模型配置)

@dataclass
class InputConfig:
    """
    用户输入配置 - 视频生成参数

    属性:
        batch_size: 批量大小 (当前版本只支持 1)
        num_frames: 视频帧数 (必须是 4k+1, 如 81, 121)
        pixel_height: 帧高度 (必须是 16 的倍数)
        pixel_width: 帧宽度 (必须是 16 的倍数)

    示例:
        Wan2.2-TI2V-5B (1280x704, 121帧):
            batch_size=1, num_frames=121, pixel_height=704, pixel_width=1280
    """
    batch_size: int = 1                          # 批量大小
    num_frames: int = 81                         # 帧数 (4k+1 格式)
    pixel_height: int = 720                      # 帧高
    pixel_width: int = 1280                       # 帧宽


# 默认配置
DEFAULT_INPUT_CONFIG = InputConfig()


# =============================================================================
# 计算函数
# =============================================================================

def calculate_tensor_size_gb(shape: tuple[int, ...], dtype_bytes: int = DTYPE_BYTES) -> float:
    """
    计算 tensor 的显存占用 (GB)

    计算公式:
    ─────────
        显存(GB) = 元素总数 × 每个元素字节数 ÷ (1024³)

    参数:
        shape: tensor 的维度元组, 例如 (1, 512, 4096)
        dtype_bytes: 每个元素的字节数 (默认 2 for bf16/fp16)

    示例:
        Shape: (1, 512, 4096)
        元素总数: 1 × 512 × 4096 = 2,097,152
        字节数: 2,097,152 × 2 = 4,194,304 bytes
        显存: 4,194,304 ÷ (1024³) ≈ 0.0039 GB

    返回:
        显存大小 (GB)
    """
    total_elements = math.prod(shape)              # 计算总元素数
    bytes_total = total_elements * dtype_bytes    # 总字节数
    return bytes_total / (1024 ** 3)              # 转换为 GB (使用 1024 进制)


def calculate_transfer_latency(size_gb: float, bandwidth_gb_s: float) -> float:
    """
    计算数据传输时延 (毫秒)

    计算公式:
    ─────────
        时延(ms) = 数据大小(GB) ÷ 带宽(GB/s) × 1000

    参数:
        size_gb: 数据大小 (GB)
        bandwidth_gb_s: 带宽 (GB/s)

    示例:
        传输 0.0073 GB 数据, 带宽 450 GB/s (NVLink):
        时延 = 0.0073 ÷ 450 × 1000 = 0.016 ms

    返回:
        传输时延 (毫秒)
    """
    if bandwidth_gb_s <= 0:
        return float('inf')
    return (size_gb / bandwidth_gb_s) * 1000


def get_transfer_bandwidth(
    source_place: str,
    dest_place: str,
    source_node: int,
    dest_node: int,
    bandwidth: BandwidthConfig,
    use_inter_node_bw: bool = False,
) -> float:
    """
    根据源组件和目标组件的放置位置选择传输带宽

    参数:
        source_place: 源组件位置 ("cpu" 或 "gpu")
        dest_place: 目标组件位置 ("cpu" 或 "gpu")
        source_node: 源组件所在的节点编号
        dest_node: 目标组件所在的节点编号
        bandwidth: 带宽配置
        use_inter_node_bw: 是否使用节点间带宽 (True=RDMA, False=根据节点判断)

    返回:
        传输带宽 (GB/s)

    说明:
        - 同节点 GPU → GPU: NVLink (最高速)
        - 同节点涉及 CPU: PCIe (中等速度)
        - 跨节点: RDMA
    """
    if source_node != dest_node or use_inter_node_bw:
        # 跨节点使用 RDMA
        return bandwidth.RDMA

    # 同节点内传输
    if source_place == "gpu" and dest_place == "gpu":
        return bandwidth.NVLINK_UNIDIRECTIONAL
    else:
        return bandwidth.PCIE5


def is_inter_node_transfer(
    source_place: str,
    dest_place: str,
    source_node: int,
    dest_node: int,
) -> bool:
    """
    判断是否为跨节点传输
    """
    return source_node != dest_node


def calculate_latent_tokens(
    num_frames: int,
    pixel_height: int,
    pixel_width: int,
    vae_stride: tuple[int, int, int] = (4, 16, 16),
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> int:
    """
    计算 DiT Token 数量 (VAE 编码 + Patchify 后的序列长度)

    计算步骤图解:
    ─────────────

    Step 1: VAE 时序压缩
    ┌────────────────────────────────────────────────────────┐
    │  输入: num_frames (原始帧数)                             │
    │                                                        │
    │  latent_frames = num_frames // 4 + 1                   │
    │                                                        │
    │  例: num_frames=121 → latent_frames=121//4+1=31       │
    │  例: num_frames=81 → latent_frames=81//4+1=21        │
    └────────────────────────────────────────────────────────┘
                          │
                          ▼
    Step 2: VAE 空间压缩
    ┌────────────────────────────────────────────────────────┐
    │  输入: pixel_height, pixel_width (原始像素尺寸)          │
    │                                                        │
    │  latent_h = pixel_height // vae_stride_h               │
    │  latent_w = pixel_width // vae_stride_w                 │
    │                                                        │
    │  TI2V (stride=16): 704//16=44, 1280//16=80          │
    │  T2V  (stride=8):  720//8=90,  1280//8=160         │
    └────────────────────────────────────────────────────────┘
                          │
                          ▼
    Step 3: DiT Patchify
    ┌────────────────────────────────────────────────────────┐
    │  输入: latent (h, w, t) 和 patch_size=(1,2,2)        │
    │                                                        │
    │  tokens_t = latent_frames // 1 = 31                   │
    │  tokens_h = latent_h // 2                            │
    │  tokens_w = latent_w // 2                            │
    │                                                        │
    │  total_tokens = tokens_t × tokens_h × tokens_w         │
    └────────────────────────────────────────────────────────┘

    参数:
        num_frames: 视频帧数
        pixel_height: 帧高度 (像素)
        pixel_width: 帧宽度 (像素)
        vae_stride: VAE 压缩比 (TI2V=(4,16,16), T2V=(4,8,8))
        patch_size: DiT patch 大小 (默认 (1, 2, 2))

    返回:
        DiT token 总数
    """
    vae_t, vae_h, vae_w = vae_stride           # 解包 VAE 压缩比
    patch_t, patch_h, patch_w = patch_size     # 解包 patch 大小

    # Step 1: VAE 时序压缩
    latent_frames = num_frames // 4 + 1

    # Step 2: VAE 空间压缩 (高度和宽度)
    latent_h = pixel_height // vae_h
    latent_w = pixel_width // vae_w

    # Step 3: DiT Patchify (将空间tokens化)
    tokens_t = latent_frames // patch_t          # 时序 token 数
    tokens_h = latent_h // patch_h              # 高度 token 数
    tokens_w = latent_w // patch_w              # 宽度 token 数

    # Step 4: 总 token 数
    total_tokens = tokens_t * tokens_h * tokens_w

    return total_tokens


def calculate_text_context_size(batch_size: int = 1) -> float:
    """
    计算 Text Context (Text Encoder 输出) 的显存大小

    详解:
    ─────
    Text Context 是 T5 Encoder 对输入文本的编码结果

    Shape: [batch_size, text_seq_length, text_dim]
           [    1     ,      512      ,     4096    ]

    Tensor 图解:
    ┌─────────────────────────────────────────────────────┐
    │                  Text Context Tensor                  │
    │  Shape: [1, 512, 4096]                               │
    │                                                      │
    │  ┌─────────────────────────────────────────────┐    │
    │  │  batch_size=1 (只有 1 个 prompt)             │    │
    │  │  512 个 token, 每个是 4096 维向量             │    │
    │  │                                              │    │
    │  │  总元素数: 1 × 512 × 4096 = 2,097,152       │    │
    │  │  显存: 2,097,152 × 2 bytes ≈ 0.0039 GB      │    │
    │  └─────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────┘

    参数:
        batch_size: 批量大小 (默认 1)

    返回:
        Text Context 显存大小 (GB)
    """
    shape = (batch_size, TEXT_SEQ_LENGTH, TEXT_DIM)
    return calculate_tensor_size_gb(shape)


def calculate_dit_input_latent_size(
    batch_size: int = 1,
    num_frames: int = 81,
    pixel_height: int = 720,
    pixel_width: int = 1280,
    vae_z_dim: int = 48,
    vae_stride: tuple[int, int, int] = (4, 16, 16),
) -> tuple[float, dict]:
    """
    计算 DiT Input Latent (VAE 编码 + Patchify 后) 的显存大小

    详解:
    ─────
    DiT Input Latent 是 VAE Encoder 输出的 latent tensor,
    经过 patchify 后作为 DiT (Diffusion Transformer) 的输入

    这是去噪过程的起点, 从纯噪声开始

    Shape: [batch_size, total_tokens, vae_z_dim]
           [    1     ,    27,280    ,       48      ]  (TI2V)
           [    1     ,    28,392    ,       16      ]  (T2V)

    参数:
        batch_size: 批量大小
        num_frames: 视频帧数
        pixel_height: 帧高度
        pixel_width: 帧宽度
        vae_z_dim: VAE latent 通道数 (TI2V=48, T2V=16)
        vae_stride: VAE 压缩比 (TI2V=(4,16,16), T2V=(4,8,8))

    返回:
        tuple: (size_gb, detail_dict)
            - size_gb: 显存大小 (GB)
            - detail_dict: 包含 tokens 数和 shape 的详细信息
    """
    tokens = calculate_latent_tokens(num_frames, pixel_height, pixel_width, vae_stride)
    shape = (batch_size, tokens, vae_z_dim)
    size_gb = calculate_tensor_size_gb(shape)

    detail = {
        "tokens": tokens,
        "shape": shape,
    }
    return size_gb, detail


def calculate_dit_output_latent_size(
    batch_size: int = 1,
    num_frames: int = 81,
    pixel_height: int = 720,
    pixel_width: int = 1280,
    vae_z_dim: int = 48,
    vae_stride: tuple[int, int, int] = (4, 16, 16),
) -> tuple[float, dict]:
    """
    计算 DiT Output Latent 的显存大小

    详解:
    ─────
    DiT Output Latent 与 Input Latent shape 完全相同,
    因为 Diffusion 过程是: noisy → denoised (维度不变)

    区别在于内容: Input 是带噪声的, Output 是去噪后的

    参数: 同 calculate_dit_input_latent_size

    返回: 同 calculate_dit_input_latent_size
    """
    return calculate_dit_input_latent_size(batch_size, num_frames, pixel_height, pixel_width, vae_z_dim, vae_stride)


def calculate_sp_shard_size(size_gb: float, num_gpus: int) -> float:
    """
    计算 Sequence Parallel 模式下每个 GPU 的分片大小

    详解:
    ─────
    Sequence Parallel (SP) 将序列分成多个 shard, 每个 GPU 处理一部分

    SP 图解:
    ───────
    Full Sequence (75,600 tokens)
    ┌────────────────────────────────────────────────────────────────┐
    │  GPU 0   │   GPU 1   │   GPU 2   │   GPU 3   │   ...        │
    │  9,450   │   9,450   │   9,450   │   9,450   │              │
    │  tokens  │   tokens  │   tokens  │   tokens  │              │
    └────────────────────────────────────────────────────────────────┘
    <─────────────── 每个 GPU 处理 75,600 / num_gpus tokens ──────────>

    参数:
        size_gb: 完整 tensor 的显存大小 (GB)
        num_gpus: GPU 数量

    返回:
        每个 GPU 分片的显存大小 (GB)
    """
    return size_gb / num_gpus


# =============================================================================
# Disaggregation 相关计算函数
# =============================================================================

def get_transfer_size_for_role(role: str, text_size_gb: float, latent_size_gb: float,
                                image_size_gb: float = 0.0001) -> float:
    """
    计算传入到指定 Role 的数据大小 (GB)

    数据流图解:
    ───────────

    ┌─────────┐  text_embeds      ┌─────────┐  denoised_latents  ┌─────────┐
    │         │ + latents         │         │                    │         │
    │ Encoder │ + image_latent*   │ Denoiser│                    │ Decoder │
    │         │──────────────────►│         │───────────────────►│         │
    │ Role    │   ~7 MB          │ Role    │   ~2 MB           │ Role    │
    └─────────┘                  └─────────┘                   └─────────┘
    * 仅 TI2V 模型需要 image_latent，T2V 模型无此数据

    Role 数据传输详情:
    ─────────────────
    1. Encoder:
       - 接收: prompt text (~0.1 MB, 很小)
       - 发送: prompt_embeds + latents (+ image_latent 仅 TI2V)
       - TI2V: ~7 MB, T2V: ~6 MB

    2. Denoiser:
       - 接收: prompt_embeds + latents (+ image_latent 仅 TI2V)
       - 发送: denoised_latents (~2 MB)

    3. Decoder:
       - 接收: denoised latents (~2 MB)
       - 发送: 最终视频 (不算传输开销)

    参数:
        role: 角色名称 ("encoder", "denoiser", "decoder")
        text_size_gb: text_context 显存大小
        latent_size_gb: latent tensor 显存大小
        image_size_gb: image latent 大小 (TI2V=0.0001 GB, T2V=0 GB)

    返回:
        传入该 Role 的数据大小 (GB)
    """
    if role == "encoder":
        # Encoder 接收文本 prompt, 发送编码后的 embeddings
        # 输入很小 (~0.1 MB), 输出包含 text_embeds + initial_latents (+ image_latent 仅 TI2V)
        return text_size_gb * 0.001 + 0.0001
    elif role == "denoiser":
        # Denoiser 接收: text_embeds + latents (+ image_latent 仅 TI2V)
        return text_size_gb + latent_size_gb + image_size_gb + 0.0005
    elif role == "decoder":
        # Decoder 接收: denoised latents
        return latent_size_gb
    return 0.0


def get_bandwidth_for_path(src_role: str, dst_role: str, disagg: DisaggConfig,
                            bandwidth: BandwidthConfig) -> float:
    """
    获取两个 Role 之间传输的带宽

    带宽判断逻辑:
    ─────────────

    1. 判断是否跨节点:
       - 假设每个节点 8 GPU (典型的 H100 SXM 配置)
       - 根据 GPU 累计数量判断 src/dst 在哪个节点

    2. 节点内 vs 节点间:
       - 同一节点: 使用 NVLink 或 PCIe
       - 不同节点: 使用 RDMA 或 Ethernet

    参数:
        src_role: 源 Role
        dst_role: 目标 Role
        disagg: Disaggregation 配置
        bandwidth: 带宽配置

    返回:
        有效带宽 (GB/s)
    """
    gpus_per_node = 8  # 每个节点典型 GPU 数

    # 计算 src/dst 所在节点
    # 假设 Role 按顺序排列: Encoder → Denoiser → Decoder
    def get_role_node(role_name: str, cfg: DisaggConfig) -> int:
        gpu_count = 0
        if role_name == "encoder":
            return 0
        elif role_name == "denoiser":
            return (cfg.encoder_gpus - 1) // gpus_per_node
        elif role_name == "decoder":
            return (cfg.encoder_gpus + cfg.denoiser_gpus - 1) // gpus_per_node
        return 0

    src_node = get_role_node(src_role, disagg)
    dst_node = get_role_node(dst_role, disagg)

    # 同一节点用 NVLink/PCIe, 不同节点用 RDMA/Network
    if src_node == dst_node:
        if disagg.intra_node_bw == "nvlink":
            return bandwidth.NVLINK_UNIDIRECTIONAL
        else:  # pcie
            return bandwidth.PCIE5
    else:
        if disagg.inter_node_bw == "rdma":
            return bandwidth.RDMA
        else:  # network
            return bandwidth.NETWORK


# =============================================================================
# 输出格式化函数
# =============================================================================

def format_size(size_gb: float, decimals: int = 4) -> str:
    """格式化显存大小, 显示指定小数位数"""
    return f"{size_gb:.{decimals}f}"


def format_latency(latency_ms: float, decimals: int = 4) -> str:
    """格式化时延, 显示指定小数位数"""
    if latency_ms == float('inf'):
        return "inf"
    return f"{latency_ms:.{decimals}f}"


def print_separator(char="=", width=100):
    """打印分隔线"""
    print(char * width)


# =============================================================================
# 主表格打印函数
# =============================================================================

def print_architecture_diagram():
    """
    打印架构图解 (ASCII Art)

    包含两种模式:
    1. Monolithic (单体模式): 所有组件在同一 GPU
    2. Disaggregation (分离模式): 组件在不同 GPU
    """
    print(ARCHITECTURE_DIAGRAM)


def print_memory_table_monolithic(
    model_configs: list,
    input_config: InputConfig,
    bandwidth: BandwidthConfig,
):
    """
    打印单体模式 (Monolithic) 的表格

    单体模式特点:
    - 所有组件 (Text Encoder, DiT, VAE) 在同一 GPU 上运行
    - 不需要组件间的数据传输
    - 显存需求 = 所有组件显存之和
    """
    print_separator()
    print("Wan2.2 视频生成 - 显存与传输时延计算器 (单体模式)")
    print_separator()
    print()

    # ==========================================================================
    # 输入配置
    # ==========================================================================
    print("### 输入配置 ###")
    print(f"  批量大小:      {input_config.batch_size}")
    print(f"  视频帧数:      {input_config.num_frames} (必须是 4k+1, 如 81, 121)")
    print(f"  分辨率:        {input_config.pixel_height} x {input_config.pixel_width}")
    print()

    # ==========================================================================
    # 表 1: Tensor 显存占用 (单体模式)
    # ==========================================================================
    print("### 表 1: Tensor 显存占用 (所有组件在同一 GPU) ###")
    print("-" * 85)
    header = f"{'模型':<20} {'组件':<25} {'Shape':<25} {'显存 (GB)':<12}"
    print(header)
    print("-" * 85)

    memory_data = []

    for model_config in model_configs:
        model_name = model_config.name

        # Text Context
        text_size = calculate_text_context_size(input_config.batch_size)
        text_shape = f"[{input_config.batch_size},{TEXT_SEQ_LENGTH},{TEXT_DIM}]"
        memory_data.append({
            "model": model_name,
            "component": "Text Context",
            "shape": text_shape,
            "size_gb": text_size,
        })

        # DiT Input Latent (使用模型-specific VAE 配置)
        latent_size, latent_detail = calculate_dit_input_latent_size(
            input_config.batch_size,
            input_config.num_frames,
            input_config.pixel_height,
            input_config.pixel_width,
            model_config.vae_z_dim,
            model_config.vae_stride,
        )
        latent_shape = f"[{input_config.batch_size},{latent_detail['tokens']},{model_config.vae_z_dim}]"
        memory_data.append({
            "model": model_name,
            "component": "DiT Input Latent",
            "shape": latent_shape,
            "size_gb": latent_size,
        })

        # DiT Output Latent (与 Input shape 相同)
        memory_data.append({
            "model": model_name,
            "component": "DiT Output Latent",
            "shape": latent_shape,
            "size_gb": latent_size,
        })

    for row in memory_data:
        print(f"{row['model']:<20} {row['component']:<25} {row['shape']:<25} {format_size(row['size_gb']):<12}")

    print("-" * 85)
    print()

    # ==========================================================================
    # 表 2: 传输时延 (单体模式 = 无需传输)
    # ==========================================================================
    print("### 表 2: 传输时延 (单体模式无需传输) ###")
    print("  所有组件在同一 GPU 上运行, 无需组件间数据传输")
    print()

    # ==========================================================================
    # 表 3: 峰值显存估算
    # ==========================================================================
    print("### 表 3: 估算峰值 GPU 显存 (模型权重 + 中间激活) ###")
    print("-" * 60)
    header = f"{'模型':<20} {'估算峰值显存 (GB)':<30}"
    print(header)
    print("-" * 60)

    for model_config in model_configs:
        # 峰值显存 ≈ DiT 显存 + Text Context + Latent (近似)
        peak_mem = model_config.dit_memory_gb + memory_data[0]["size_gb"] + memory_data[1]["size_gb"]
        print(f"{model_config.name:<20} {peak_mem:<30.2f}")

    print("-" * 60)
    print()


def print_memory_table_disaggregated(
    model_configs: list,
    input_config: InputConfig,
    bandwidth: BandwidthConfig,
    disagg: DisaggConfig,
    placement: PlacementConfig,
):
    """
    打印 Disaggregation 模式 (分离式) 的表格

    Disaggregation 模式特点:
    - 组件分布在不同 GPU 上 (Encoder, Denoiser, Decoder 三个 Role)
    - Role 之间需要传输数据
    - 每个 Role 有独立的显存和传输需求
    """
    print_separator()
    print("Wan2.2 视频生成 - 显存与传输时延计算器 (Disaggregation 分离式模式)")
    print_separator()
    print()

    # ==========================================================================
    # 输入配置
    # ==========================================================================
    print("### 输入配置 ###")
    print(f"  批量大小:      {input_config.batch_size}")
    print(f"  视频帧数:      {input_config.num_frames}")
    print(f"  分辨率:        {input_config.pixel_height} x {input_config.pixel_width}")
    print()

    # ==========================================================================
    # Disaggregation 配置
    # ==========================================================================
    print("### Disaggregation 架构配置 ###")

    # 架构图示
    print("""
    架构图示:
    ┌─────────┐     传输 1 (~7 MB)     ┌─────────┐     传输 2 (~2 MB)     ┌─────────┐
    │ Encoder │ ─────────────────────►│ Denoiser│ ─────────────────────►│ Decoder │
    │  Role   │   NVLink: 0.02ms     │  Role   │   NVLink: 0.005ms    │  Role   │
    │         │   RDMA:   0.36ms     │         │   RDMA:   0.11ms     │         │
    │  GPU:1  │                      │  GPU:N  │                      │  GPU:1  │
    └─────────┘                      └─────────┘                      └─────────┘
    """)

    print(f"  Encoder GPUs:  {disagg.encoder_gpus} 个  (Text Encoder + Image Encoder + VAE Encoder)")
    print(f"  Denoiser GPUs: {disagg.denoiser_gpus} 个  (DiT - 占总时间 ~89%)")
    print(f"  Decoder GPUs:  {disagg.decoder_gpus} 个  (VAE Decoder)")
    print(f"  总 GPU 数:     {disagg.total_gpus}")
    print(f"  物理节点数:    {disagg.nodes}")
    print(f"  节点内互联:    {disagg.intra_node_bw.upper()}")
    print(f"  节点间互联:    {disagg.inter_node_bw.upper()}")
    print(f"  传输缓冲池:    {disagg.transfer_pool_size_mb} MB / Role")
    print()

    # ==========================================================================
    # 计算公共数据
    # ==========================================================================
    text_size = calculate_text_context_size(input_config.batch_size)

    # ==========================================================================
    # 表 1: 每个 Role 的显存占用和传输量
    # ==========================================================================
    print("### 表 1: 每个 Role 的显存占用与传输数据量 ###")
    print("-" * 95)
    header = f"{'模型':<16} {'Role':<12} {'组件显存 (GB)':<20} {'传入数据 (GB)':<18} {'传出数据 (GB)':<18}"
    print(header)
    print("-" * 95)

    for model_config in model_configs:
        model_name = model_config.name

        # 计算该模型的 latent_size (使用模型-specific VAE 配置)
        latent_size, latent_detail = calculate_dit_input_latent_size(
            input_config.batch_size,
            input_config.num_frames,
            input_config.pixel_height,
            input_config.pixel_width,
            model_config.vae_z_dim,
            model_config.vae_stride,
        )

        # image_size: TI2V 有条件图 (~0.1 MB), T2V 无条件图
        image_size = 0.0001 if model_config.model_type == "TI2V" else 0.0

        # Encoder Role
        enc_transfer_in = get_transfer_size_for_role("encoder", text_size, latent_size, image_size)
        enc_transfer_out = text_size + latent_size + image_size + 0.005  # prompt_embeds + latents + negative_prompt_embeds + pooled_embeds + timesteps + masks (+ image_latent 仅 TI2V)
        print(f"{model_name:<16} {'Encoder':<12} {model_config.encoder_memory_gb:<20.2f} "
              f"{format_size(enc_transfer_in):<18} {format_size(enc_transfer_out):<18}")

        # Denoiser Role
        deno_transfer_in = get_transfer_size_for_role("denoiser", text_size, latent_size, image_size)
        deno_transfer_out = latent_size  # Send denoised latents to decoder
        print(f"{model_name:<16} {'Denoiser':<12} {model_config.dit_memory_gb:<20.2f} "
              f"{format_size(deno_transfer_in):<18} {format_size(deno_transfer_out):<18}")

        # Decoder Role
        dec_transfer_in = get_transfer_size_for_role("decoder", text_size, latent_size, image_size)
        dec_transfer_out = 0  # Final output to client
        print(f"{model_name:<16} {'Decoder':<12} {model_config.vae_decoder_memory_gb:<20.2f} "
              f"{format_size(dec_transfer_in):<18} {format_size(dec_transfer_out):<18}")

    print("-" * 95)
    print()

    # ==========================================================================
    # 表 2: Role 间传输时延
    # ==========================================================================
    print("### 表 2: Role 间传输时延 (毫秒) ###")
    print("-" * 110)
    header = f"{'模型':<14} {'传输路径':<35} {'数据量':<10} {'路径类型':<12} {'实际 (ms)':<14} {'NVLink':<12} {'PCIe':<12} {'RDMA':<12}"
    print(header)
    print("-" * 110)

    for model_config in model_configs:
        model_name = model_config.name

        # 计算该模型的 latent_size (使用模型-specific VAE 配置)
        latent_size, latent_detail = calculate_dit_input_latent_size(
            input_config.batch_size,
            input_config.num_frames,
            input_config.pixel_height,
            input_config.pixel_width,
            model_config.vae_z_dim,
            model_config.vae_stride,
        )

        # image_size: TI2V 有条件图 (~0.1 MB), T2V 无条件图
        image_size = 0.0001 if model_config.model_type == "TI2V" else 0.0

        # Encoder → Denoiser
        enc_to_deno_size = text_size + latent_size + image_size + 0.005

        # 根据组件位置和节点选择带宽
        enc_to_deno_bw = get_transfer_bandwidth(
            placement.encoder, placement.dit,
            placement.encoder_node, placement.dit_node,
            bandwidth
        )
        path1_is_inter = is_inter_node_transfer(
            placement.encoder, placement.dit,
            placement.encoder_node, placement.dit_node
        )
        path1_lat = calculate_transfer_latency(enc_to_deno_size, enc_to_deno_bw)
        path1_lat_nvlink = calculate_transfer_latency(enc_to_deno_size, bandwidth.NVLINK_UNIDIRECTIONAL)
        path1_lat_pcie = calculate_transfer_latency(enc_to_deno_size, bandwidth.PCIE5)
        path1_lat_rdma = calculate_transfer_latency(enc_to_deno_size, bandwidth.RDMA)

        path1_node_type = "跨节点" if path1_is_inter else "同节点"
        path1_label = f"Encoder→Denoiser (Node{placement.encoder_node}→Node{placement.dit_node})"
        print(f"{model_name:<14} {path1_label:<35} {format_size(enc_to_deno_size):<10} {path1_node_type:<12} {format_latency(path1_lat):<14} {format_latency(path1_lat_nvlink):<12} {format_latency(path1_lat_pcie):<12} {format_latency(path1_lat_rdma):<12}")

        # Denoiser → Decoder
        deno_to_dec_size = latent_size
        deno_to_dec_bw = get_transfer_bandwidth(
            placement.dit, placement.vae,
            placement.dit_node, placement.vae_node,
            bandwidth
        )
        path2_is_inter = is_inter_node_transfer(
            placement.dit, placement.vae,
            placement.dit_node, placement.vae_node
        )
        path2_lat = calculate_transfer_latency(deno_to_dec_size, deno_to_dec_bw)
        path2_lat_nvlink = calculate_transfer_latency(deno_to_dec_size, bandwidth.NVLINK_UNIDIRECTIONAL)
        path2_lat_pcie = calculate_transfer_latency(deno_to_dec_size, bandwidth.PCIE5)
        path2_lat_rdma = calculate_transfer_latency(deno_to_dec_size, bandwidth.RDMA)

        path2_node_type = "跨节点" if path2_is_inter else "同节点"
        path2_label = f"Denoiser→Decoder (Node{placement.dit_node}→Node{placement.vae_node})"
        print(f"{model_name:<14} {path2_label:<35} {format_size(deno_to_dec_size):<10} {path2_node_type:<12} {format_latency(path2_lat):<14} {format_latency(path2_lat_nvlink):<12} {format_latency(path2_lat_pcie):<12} {format_latency(path2_lat_rdma):<12}")

    print("-" * 110)
    print()

    # ==========================================================================
    # 表 3: 传输缓冲池容量规划
    # ==========================================================================
    print("### 表 3: 传输缓冲池 (Transfer Buffer Pool) 容量规划 ###")
    print("-" * 85)
    header = f"{'模型':<18} {'Role 对':<20} {'数据量 (MB)':<18} {'池大小 (MB)':<18} {'并发数':<12}"
    print(header)
    print("-" * 85)

    for model_config in model_configs:
        model_name = model_config.name

        # 计算该模型的 latent_size
        latent_size, _ = calculate_dit_input_latent_size(
            input_config.batch_size,
            input_config.num_frames,
            input_config.pixel_height,
            input_config.pixel_width,
            model_config.vae_z_dim,
            model_config.vae_stride,
        )

        # Encoder → Denoiser
        enc_to_deno_size_mb = (text_size + latent_size + image_size + 0.005) * 1024
        concurrent_enc = max(1, int(disagg.transfer_pool_size_mb / max(0.001, enc_to_deno_size_mb)))
        print(f"{model_name:<18} {'Encoder→Denoiser':<20} {enc_to_deno_size_mb:<18.2f} {disagg.transfer_pool_size_mb:<18} {concurrent_enc:<12}")

        # Denoiser → Decoder
        deno_to_dec_size_mb = latent_size * 1024
        concurrent_deno = max(1, int(disagg.transfer_pool_size_mb / max(0.001, deno_to_dec_size_mb)))
        print(f"{model_name:<18} {'Denoiser→Decoder':<20} {deno_to_dec_size_mb:<18.2f} {disagg.transfer_pool_size_mb:<18} {concurrent_deno:<12}")

    print("-" * 85)
    print()
    print("  说明: 并发数表示每个传输缓冲池能同时缓存的请求数")
    print()

    # ==========================================================================
    # 表 4: Role 时间分解 (参考 profiling 数据)
    # ==========================================================================
    print("### 表 4: Role 时间分解 (参考: Wan2.2-TI2V-5B Profile 实测) ###")
    print("-" * 80)
    header = f"{'模型':<16} {'Role':<12} {'时间 (s)':<12} {'占比':<15} {'说明':<25}"
    print(header)
    print("-" * 80)

    # 参考时间 (来自 profile_wan_ti2v_stages.py 的实测数据)
    ref_total = 97.35 + 2.48 + 9.41  # DiT + Encoder + VAE Decode
    ref_encoder = 2.48
    ref_dit = 97.35
    ref_vae_decode = 9.41

    for model_config in model_configs:
        model_name = model_config.name
        pct_encoder = (ref_encoder / ref_total) * 100
        pct_dit = (ref_dit / ref_total) * 100
        pct_decoder = (ref_vae_decode / ref_total) * 100

        print(f"{model_name:<16} {'Encoder':<12} {ref_encoder:<12.2f} {pct_encoder:<15.1f} {'Text + Image encoding':<25}")
        print(f"{model_name:<16} {'Denoiser':<12} {ref_dit:<12.2f} {pct_dit:<15.1f} {'DiT 去噪 (50 steps)':<25}")
        print(f"{model_name:<16} {'Decoder':<12} {ref_vae_decode:<12.2f} {pct_decoder:<15.1f} {'VAE decode':<25}")

    print("-" * 80)
    print()

    # ==========================================================================
    # 表 5: Denoiser SP 分片分析 (如果多 GPU)
    # ==========================================================================
    if disagg.denoiser_gpus > 1:
        print("### 表 5: Denoiser Sequence Parallel (SP) 分片分析 ###")
        print("-" * 90)

        for model_config in model_configs:
            model_name = model_config.name

            # 计算该模型的 latent_size
            latent_size, latent_detail = calculate_dit_input_latent_size(
                input_config.batch_size,
                input_config.num_frames,
                input_config.pixel_height,
                input_config.pixel_width,
                model_config.vae_z_dim,
                model_config.vae_stride,
            )

            print(f"\n{model_name} DiT Latent: {latent_detail['tokens']:,} tokens, shape={latent_detail['shape']}")
            print("SP 图解:")
            tokens_per_gpu = latent_detail['tokens'] // disagg.denoiser_gpus
            print(f"┌─────────────────────────────────────────────────────────────────────────────┐")
            print(f"│  Full Sequence ({latent_detail['tokens']:,} tokens)                                          │")
            print(f"│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐                   │")
            print(f"│  │  GPU 0   │  GPU 1   │  GPU 2   │  GPU 3   │  ...    │                   │")
            print(f"│  │  {tokens_per_gpu:,} │  {tokens_per_gpu:,} │  {tokens_per_gpu:,} │  {tokens_per_gpu:,} │          │                   │")
            print(f"│  │  tokens  │  tokens  │  tokens  │  tokens  │          │                   │")
            print(f"│  └──────────┴──────────┴──────────┴──────────┴──────────┘                   │")
            print(f"│  ◄──────────────── 每个 GPU 处理 {latent_detail['tokens']:,} / N tokens ───────────────────────►  │")
            print(f"└─────────────────────────────────────────────────────────────────────────────┘")
            print()

            header = f"{'GPU 数':<10} {'每卡显存 (GB)':<18} {'NVLink 时延 (ms)':<20} {'PCIe 时延 (ms)':<18}"
            print(header)
            print("-" * 90)

            for num_gpus in [2, 4, 8]:
                shard_size_gb = calculate_sp_shard_size(latent_size, num_gpus)
                nvlink_lat = calculate_transfer_latency(shard_size_gb, bandwidth.NVLINK_UNIDIRECTIONAL)
                pcie_lat = calculate_transfer_latency(shard_size_gb, bandwidth.PCIE5)

                print(f"{num_gpus:<10} {format_size(shard_size_gb, 6):<18} {format_latency(nvlink_lat):<20} {format_latency(pcie_lat):<18}")

            print("-" * 90)
        print()

    # ==========================================================================
    # 瓶颈分析
    # ==========================================================================
    print("### 瓶颈分析 (Bottleneck Analysis) ###")
    print()

    # 根据节点配置选择带宽
    if disagg.nodes > 1:
        inter_bw = bandwidth.RDMA
    else:
        inter_bw = bandwidth.NVLINK_UNIDIRECTIONAL

    for model_config in model_configs:
        model_name = model_config.name

        # 计算该模型的 latent_size
        latent_size, latent_detail = calculate_dit_input_latent_size(
            input_config.batch_size,
            input_config.num_frames,
            input_config.pixel_height,
            input_config.pixel_width,
            model_config.vae_z_dim,
            model_config.vae_stride,
        )

        enc_to_deno_lat = calculate_transfer_latency(text_size + latent_size + image_size + 0.005, inter_bw)
        deno_to_dec_lat = calculate_transfer_latency(latent_size, inter_bw)
        total_transfer_lat = enc_to_deno_lat + deno_to_dec_lat

        print(f"  [{model_name}] 传输开销估算:")
        print(f"    - Encoder→Denoiser: {format_latency(enc_to_deno_lat)} ms")
        print(f"    - Denoiser→Decoder: {format_latency(deno_to_dec_lat)} ms")
        print(f"    - 总传输时延: {format_latency(total_transfer_lat)} ms")
        print()

    print("  与计算时间对比:")
    print(f"    - Denoiser (DiT): ~97s (占 ~89%)")
    print(f"    - 传输开销: 取决于模型和带宽，通常 < 1ms (可忽略)")
    print()

    print("  关键结论:")
    print("    ┌───────────────────────────────────────────────────────────────────────┐")
    print("    │  1. Disaggregation 的传输开销极小 (< 1ms vs 97s 计算)                   │")
    print("    │  2. 分离式部署的优势:                                                   │")
    print("    │     - 每个 Role 可独立扩展 GPU 数量                                      │")
    print("    │     - 可以为不同 Role 选择不同的并行策略 (TP/SP)                        │")
    print("    │     - 更容易实现负载均衡                                                 │")
    print("    │  3. 注意事项:                                                          │")
    print("    │     - 需要配置足够的传输缓冲池 (建议 256MB+)                             │")
    print("    │     - 跨节点传输时 RDMA 带宽至关重要                                    │")
    print("    └───────────────────────────────────────────────────────────────────────┘")
    print()

    # ==========================================================================
    # 计算验证
    # ==========================================================================
    print("### 计算验证 (Calculation Verification) ###")

    for model_config in model_configs:
        model_name = model_config.name

        # 计算该模型的 latent_size
        latent_size, latent_detail = calculate_dit_input_latent_size(
            input_config.batch_size,
            input_config.num_frames,
            input_config.pixel_height,
            input_config.pixel_width,
            model_config.vae_z_dim,
            model_config.vae_stride,
        )

        vae_stride_t = model_config.vae_stride[0]
        vae_stride_h = model_config.vae_stride[1]
        vae_stride_w = model_config.vae_stride[2]
        print(f"\n  [{model_name}] Latent tokens: {latent_detail['tokens']:,}")
        print(f"    - latent_frames = {input_config.num_frames} // {vae_stride_t} + 1 = {input_config.num_frames // vae_stride_t + 1}")
        print(f"    - latent_h = {input_config.pixel_height} // {vae_stride_h} = {input_config.pixel_height // vae_stride_h}")
        print(f"    - latent_w = {input_config.pixel_width} // {vae_stride_w} = {input_config.pixel_width // vae_stride_w}")
        print(f"    - vae_z_dim = {model_config.vae_z_dim}")
        print(f"    - patch_size = {model_config.patch_size}")
        print(f"    - tokens = {input_config.num_frames // vae_stride_t + 1} × {input_config.pixel_height // vae_stride_h // model_config.patch_size[1]} × {input_config.pixel_width // vae_stride_w // model_config.patch_size[2]} × {model_config.vae_z_dim}")
        print(f"    - tokens = {latent_detail['tokens']:,}")
    print()


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="Wan2.2 视频生成模型 - 显存与传输时延计算器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 默认单体模式 (所有组件在同一 GPU)
  python wan_memory_calculator.py

  # Disaggregation 模式 (组件分离到不同 GPU)
  python wan_memory_calculator.py --disagg

  # 自定义 GPU 分配
  python wan_memory_calculator.py --disagg --encoder-gpus 1 --denoiser-gpus 4 --decoder-gpus 1

  # 多节点场景 (跨机器)
  python wan_memory_calculator.py --disagg --nodes 2 --inter-node-bw rdma

  # 查看架构图解
  python wan_memory_calculator.py --diagram

详细说明:
  Disaggregation 模式将流水线分为三个 Role:
    - Encoder: Text Encoder + Image Encoder + VAE Encoder
    - Denoiser: DiT (主要计算, 约 89% 时间)
    - Decoder: VAE Decoder

  每个 Role 可以有独立的 GPU 数量, 通过高速网络互联.
        """
    )

    # 输入参数
    parser.add_argument("--batch-size", type=int, default=1, help="批量大小 (默认 1)")
    parser.add_argument("--num-frames", type=int, default=81, help="视频帧数 (必须是 4k+1, 如 81, 121)")
    parser.add_argument("--height", type=int, default=720, help="帧高度 (必须是 16 的倍数)")
    parser.add_argument("--width", type=int, default=1280, help="帧宽度 (必须是 16 的倍数)")

    # Disaggregation 选项
    parser.add_argument("--disagg", action="store_true",
                       help="启用 Disaggregation 分离式部署模式")
    parser.add_argument("--encoder-gpus", type=int, default=1,
                       help="Encoder Role 的 GPU 数量 (默认 1)")
    parser.add_argument("--denoiser-gpus", type=int, default=1,
                       help="Denoiser Role 的 GPU 数量 (默认 1, 建议 4-8)")
    parser.add_argument("--decoder-gpus", type=int, default=1,
                       help="Decoder Role 的 GPU 数量 (默认 1)")
    parser.add_argument("--nodes", type=int, default=1,
                       help="物理节点数量 (用于跨节点传输)")
    parser.add_argument("--intra-node-bw", choices=["nvlink", "pcie"], default="nvlink",
                       help="节点内互联类型 (默认 nvlink)")
    parser.add_argument("--inter-node-bw", choices=["rdma", "network"], default="rdma",
                       help="节点间互联类型 (默认 rdma)")
    parser.add_argument("--transfer-pool-size", type=int, default=256,
                       help="传输缓冲池大小 MB/Role (默认 256)")

    # 组件放置位置 (用于 disaggregation 模式)
    parser.add_argument("--encoder-place", choices=["cpu", "gpu"], default="gpu",
                       help="Encoder 放置位置: cpu 或 gpu (默认 gpu)")
    parser.add_argument("--dit-place", choices=["cpu", "gpu"], default="gpu",
                       help="DiT (Denoiser) 放置位置: cpu 或 gpu (默认 gpu)")
    parser.add_argument("--vae-place", choices=["cpu", "gpu"], default="gpu",
                       help="VAE (Decoder) 放置位置: cpu 或 gpu (默认 gpu)")

    # 组件所在节点 (用于多机配置)
    parser.add_argument("--encoder-node", type=int, default=0,
                       help="Encoder 所在的节点编号 (默认 0)")
    parser.add_argument("--dit-node", type=int, default=0,
                       help="DiT (Denoiser) 所在的节点编号 (默认 0)")
    parser.add_argument("--vae-node", type=int, default=0,
                       help="VAE (Decoder) 所在的节点编号 (默认 0)")

    # 模型选择
    parser.add_argument("--model", choices=["5B", "14B", "all"], default="all",
                       help="模型大小 (默认 all)")

    # 架构图解
    parser.add_argument("--diagram", action="store_true",
                       help="只打印架构图解")

    args = parser.parse_args()

    # 如果只打印图解
    if args.diagram:
        print_architecture_diagram()
        return

    # 构建配置
    input_config = InputConfig(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        pixel_height=args.height,
        pixel_width=args.width,
    )

    bandwidth = BandwidthConfig()

    placement = PlacementConfig(
        encoder=args.encoder_place,
        dit=args.dit_place,
        vae=args.vae_place,
        encoder_node=args.encoder_node,
        dit_node=args.dit_node,
        vae_node=args.vae_node,
    )

    disagg = DisaggConfig(
        enabled=args.disagg,
        encoder_gpus=args.encoder_gpus,
        denoiser_gpus=args.denoiser_gpus,
        decoder_gpus=args.decoder_gpus,
        total_gpus=args.encoder_gpus + args.denoiser_gpus + args.decoder_gpus,
        nodes=args.nodes,
        intra_node_bw=args.intra_node_bw,
        inter_node_bw=args.inter_node_bw,
        transfer_pool_size_mb=args.transfer_pool_size,
    )

    # 选择模型
    if args.model == "5B":
        model_configs = [WAN_TI2V_5B_CONFIG]
    elif args.model == "14B":
        model_configs = [WAN_T2V_A14B_CONFIG]
    else:
        model_configs = [WAN_TI2V_5B_CONFIG, WAN_T2V_A14B_CONFIG]

    # 打印表格
    if disagg.enabled:
        print_memory_table_disaggregated(model_configs, input_config, bandwidth, disagg, placement)
    else:
        print_memory_table_monolithic(model_configs, input_config, bandwidth)


if __name__ == "__main__":
    main()
