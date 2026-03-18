#!/bin/bash
# Script to profile Encoder and VAE Encoder stages on 8 GPUs
# Combines encoder_8gpu_test.sh and vae_encoder_8gpu_test.sh

set -e

export MODEL="/workspace/models/Wan2_2-TI2V-5B-Diffusers"
export PROMPT="A cinematic science-fiction city with reflective rain streets."
export OUT_DIR="/workspace/wan22_ti2v5b_profile"
export LOG_DIR="/workspace/sglang/logs"
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

# Log file with timestamp
LOG_FILE="${LOG_DIR}/encoder_vae_encoder_8gpu_test_$(date +%Y%m%d_%H%M%S).log"

# Function to run command with logging
run_with_log() {
    "$@" 2>&1 | tee -a "${LOG_FILE}"
    return ${PIPESTATUS[0]}
}

echo "========================================"
echo "Encoder + VAE Encoder Profiling Tests"
echo "========================================"
echo "Model: ${MODEL}"
echo "Output: ${OUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "GPUs: CUDA 0,1,2,3,4,5,6,7 (8 GPUs)"
echo "Warmup: 3, Profile: 10"
echo "========================================"
echo ""

# Check available GPUs
echo ">>> Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv -i 0,1,2,3,4,5,6,7 2>/dev/null || echo "Warning: Cannot query all 8 GPUs"
echo ""

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ========================================
# Section 1: Encoder 8 GPU Test
# ========================================
echo ""
echo ">>>========================================"
echo ">>> Section 1: Encoder 8 GPU Test"
echo ">>>========================================"

mkdir -p "${OUT_DIR}/encoder/gpu8"
echo ""
echo ">>> Testing Encoder On 8 GPUs (CUDA 0-7)..."
run_with_log torchrun --standalone --nproc_per_node 8 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage encoder \
  --device cuda \
  --num-gpus 8 \
  --tp-size 8 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 3 \
  --profile-iters 10 \
  --output-json "${OUT_DIR}/encoder/gpu8/profile.json" \
  --output-txt "${OUT_DIR}/encoder/gpu8/profile.txt"

echo ""
echo "✓ Encoder 8 GPU test completed"

# ========================================
# Section 2: VAE Encoder 8 GPU Test
# ========================================
echo ""
echo ">>>========================================"
echo ">>> Section 2: VAE Encoder 8 GPU Test"
echo ">>>========================================"
echo ">>> Note: TI2V uses 1 frame for VAE encode (condition image)"

mkdir -p "${OUT_DIR}/vae_encoder/gpu8"
echo ""
echo ">>> Testing VAE Encoder On 8 GPUs (CUDA 0-7)..."
run_with_log torchrun --standalone --nproc_per_node 8 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-encode \
  --device cuda \
  --num-gpus 8 \
  --tp-size 8 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --vae-encode-input-frames 1 \
  --sync-stage-profiling all \
  --warmup-iters 3 \
  --profile-iters 10 \
  --output-json "${OUT_DIR}/vae_encoder/gpu8/profile.json" \
  --output-txt "${OUT_DIR}/vae_encoder/gpu8/profile.txt"

echo ""
echo "✓ VAE Encoder 8 GPU test completed"

# ========================================
# Summary
# ========================================
echo ""
echo "========================================"
echo "All 8-GPU Tests Completed!"
echo "========================================"
echo ""
echo "Results summary:"
echo ""
echo "1. Encoder (8 GPUs):"
echo "   Directory: ${OUT_DIR}/encoder/gpu8/"
echo "   - JSON: ${OUT_DIR}/encoder/gpu8/profile.json"
echo "   - TXT:  ${OUT_DIR}/encoder/gpu8/profile.txt"
echo ""
echo "2. VAE Encoder (8 GPUs):"
echo "   Directory: ${OUT_DIR}/vae_encoder/gpu8/"
echo "   - JSON: ${OUT_DIR}/vae_encoder/gpu8/profile.json"
echo "   - TXT:  ${OUT_DIR}/vae_encoder/gpu8/profile.txt"
echo ""
echo "Log file: ${LOG_FILE}"
echo "========================================"
