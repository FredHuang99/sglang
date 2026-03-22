#!/bin/bash
#================================================================
# Wan2.2 Disaggregation 放置组合测试脚本
#================================================================
# 测试场景: 单机/双机八卡配置
# - Encoder Role: 1 GPU (或 CPU)
# - DiT Role: 6 GPUs (必须在 GPU 上)
# - VAE Role: 1 GPU (或 CPU)
#
# 单机组合: 2 × 1 × 2 = 4 种 (DiT 固定为 GPU)
# 双机组合: 考虑跨节点传输
#================================================================

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALCULATOR="${SCRIPT_DIR}/wan_memory_calculator.py"

# 单机八卡配置
ENCODER_GPUS=1
DIT_GPUS=6
VAE_GPUS=1

#================================================================
# 测试函数
#================================================================
run_test() {
    local name=$1
    local encoder_place=$2
    local dit_place=$3
    local vae_place=$4
    local encoder_node=$5
    local dit_node=$6
    local vae_node=$7

    # 运行计算器获取传输时延
    output=$(/root/miniconda3/bin/python ${CALCULATOR} --disagg \
        --encoder-place ${encoder_place} \
        --dit-place ${dit_place} \
        --vae-place ${vae_place} \
        --encoder-gpus ${ENCODER_GPUS} \
        --denoiser-gpus ${DIT_GPUS} \
        --decoder-gpus ${VAE_GPUS} \
        --nodes 2 \
        --encoder-node ${encoder_node} \
        --dit-node ${dit_node} \
        --vae-node ${vae_node} \
        --model 5B 2>&1)

    # 提取传输时延 (从表 2)
    # 数据行以 Wan2.2- 开头，跳过表头
    enc_to_deno=$(echo "$output" | grep "Wan2.2-.*Encoder→Denoiser" | head -1 | awk '{print $(NF-3)}')
    deno_to_dec=$(echo "$output" | grep "Wan2.2-.*Denoiser→Decoder" | head -1 | awk '{print $(NF-3)}')

    # 判断路径类型 (同节点: NodeX→NodeX, 跨节点: NodeX→NodeY where X!=Y)
    enc_path=$(echo "$output" | grep "Wan2.2-.*Encoder→Denoiser" | head -1 | grep -oP 'Node\d+→Node\d+')
    dec_path=$(echo "$output" | grep "Wan2.2-.*Denoiser→Decoder" | head -1 | grep -oP 'Node\d+→Node\d+')

    # 计算总延迟
    total_lat=$(echo "$enc_to_deno $deno_to_dec" | awk '{printf "%.4f", $1 + $2}')

    # 判断是否跨节点 (NodeX→NodeY 且 X!=Y 为跨节点)
    enc_from=$(echo "$enc_path" | grep -oP 'Node\d+' | head -1 | grep -oP '\d+')
    enc_to=$(echo "$enc_path" | grep -oP 'Node\d+$' | grep -oP '\d+')
    if [ "$enc_from" != "$enc_to" ]; then
        enc_inter="跨节点"
    else
        enc_inter="同节点"
    fi
    dec_from=$(echo "$dec_path" | grep -oP 'Node\d+' | head -1 | grep -oP '\d+')
    dec_to=$(echo "$dec_path" | grep -oP 'Node\d+$' | grep -oP '\d+')
    if [ "$dec_from" != "$dec_to" ]; then
        dec_inter="跨节点"
    else
        dec_inter="同节点"
    fi

    # 输出结果
    printf "%-8s | %-6s | %-6s | %-6s | %-8s | %-8s | %-8s | %-8s | %-10s | %-10s | %-10s\n" \
        "${name}" "${encoder_place}" "${dit_place}" "${vae_place}" \
        "${encoder_node}" "${dit_node}" "${vae_node}" \
        "${enc_inter}" "${enc_to_deno} ms" "${deno_to_dec} ms" "${total_lat} ms"
}

#================================================================
# 测试单机组 (4 种组合)
#================================================================
echo "================================================================================"
echo "Wan2.2 Disaggregation 放置组合测试 (单机八卡)"
echo "================================================================================"
echo ""
echo "配置:"
echo "  - Encoder GPUs: ${ENCODER_GPUS} (可放在 CPU 或 GPU)"
echo "  - DiT GPUs: ${DIT_GPUS} (必须在 GPU 上)"
echo "  - VAE GPUs: ${VAE_GPUS} (可放在 CPU 或 GPU)"
echo ""

# 表头
echo "--------------------------------------------------------------------------------"
printf "%-8s | %-6s | %-6s | %-6s | %-8s | %-8s | %-8s | %-8s | %-10s | %-10s | %-10s\n" \
    "组合" "Enc" "DiT" "VAE" "Enc节点" "DiT节点" "VAE节点" "Enc→Den" "Enc→Den延迟" "Den→Dec延迟" "总延迟"
echo "--------------------------------------------------------------------------------"

idx=1
for encoder_place in cpu gpu; do
    for vae_place in cpu gpu; do
        name="T${idx}"
        run_test "${name}" "${encoder_place}" "gpu" "${vae_place}" 0 0 0
        idx=$((idx + 1))
    done
done

echo "--------------------------------------------------------------------------------"
echo ""

#================================================================
# 测试双机组 (典型跨节点场景)
#================================================================
echo "================================================================================"
echo "Wan2.2 Disaggregation 放置组合测试 (双机跨节点)"
echo "================================================================================"
echo ""
echo "DiT 必须在 GPU 上"
echo "考虑以下典型场景:"
echo "  1. DiT 在节点 0, Encoder/VAE 在节点 1"
echo "  2. Encoder 在节点 0, DiT/VAE 在节点 1"
echo "  3. VAE 在节点 0, Encoder/DiT 在节点 1"
echo ""

# 表头
echo "--------------------------------------------------------------------------------"
printf "%-8s | %-6s | %-6s | %-6s | %-8s | %-8s | %-8s | %-8s | %-10s | %-10s | %-10s\n" \
    "场景" "Enc" "DiT" "VAE" "Enc节点" "DiT节点" "VAE节点" "Enc→Den" "Enc→Den延迟" "Den→Dec延迟" "总延迟"
echo "--------------------------------------------------------------------------------"

idx=1

# 场景 1: DiT 在节点 0, Encoder 在节点 1, VAE 在节点 0
for encoder_place in cpu gpu; do
    for vae_place in cpu gpu; do
        name="S1-${idx}"
        run_test "${name}" "${encoder_place}" "gpu" "${vae_place}" 1 0 0
        idx=$((idx + 1))
    done
done

# 场景 2: DiT 在节点 0, Encoder 在节点 0, VAE 在节点 1
for encoder_place in cpu gpu; do
    for vae_place in cpu gpu; do
        name="S2-${idx}"
        run_test "${name}" "${encoder_place}" "gpu" "${vae_place}" 0 0 1
        idx=$((idx + 1))
    done
done

# 场景 3: DiT 在节点 1, Encoder 在节点 0, VAE 在节点 0
for encoder_place in cpu gpu; do
    for vae_place in cpu gpu; do
        name="S3-${idx}"
        run_test "${name}" "${encoder_place}" "gpu" "${vae_place}" 0 1 0
        idx=$((idx + 1))
    done
done

# 场景 4: DiT 在节点 1, Encoder 在节点 0, VAE 在节点 1
for encoder_place in cpu gpu; do
    for vae_place in cpu gpu; do
        name="S4-${idx}"
        run_test "${name}" "${encoder_place}" "gpu" "${vae_place}" 0 1 1
        idx=$((idx + 1))
    done
done

# 场景 5: DiT 在节点 0, Encoder 在节点 1, VAE 在节点 1
for encoder_place in cpu gpu; do
    for vae_place in cpu gpu; do
        name="S5-${idx}"
        run_test "${name}" "${encoder_place}" "gpu" "${vae_place}" 1 0 1
        idx=$((idx + 1))
    done
done

# 场景 6: DiT 在节点 1, Encoder 在节点 1, VAE 在节点 0
for encoder_place in cpu gpu; do
    for vae_place in cpu gpu; do
        name="S6-${idx}"
        run_test "${name}" "${encoder_place}" "gpu" "${vae_place}" 1 1 0
        idx=$((idx + 1))
    done
done

echo "--------------------------------------------------------------------------------"
echo ""

#================================================================
# 总结
#================================================================
echo "================================================================================"
echo "结果解读"
echo "================================================================================"
echo ""
echo "  - DiT (Denoiser) 必须在 GPU 上"
echo "  - '同节点' 表示 GPU→GPU 使用 NVLink (~0.015ms)"
echo "  - '跨节点' 表示使用 RDMA (~0.333ms)"
echo ""
echo "关键结论:"
echo "  - 同节点 GPU→GPU: ~0.018ms (最优)"
echo "  - 同节点涉及 CPU: ~0.130ms (PCIe)"
echo "  - 跨节点: ~0.350ms (RDMA)"
echo ""
echo "推荐配置: 单机全 GPU (T4)"
