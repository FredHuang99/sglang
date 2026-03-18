#!/bin/bash
# Script to profile VAE Encoder stage with 3 warmup and 10 iterations
# Results saved to /workspace/wan22_ti2v5b_profile/vae_encoder2

set -e

export MODEL="/workspace/models/Wan2_2-TI2V-5B-Diffusers"
export PROMPT="A cinematic science-fiction city with reflective rain streets."
export OUT_DIR="/workspace/wan22_ti2v5b_profile/vae_encoder2"
export LOG_DIR="/workspace/sglang/logs"
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

# Log file with timestamp
LOG_FILE="${LOG_DIR}/vae_encoder_test2_$(date +%Y%m%d_%H%M%S).log"

# Function to run command with logging
run_with_log() {
    "$@" 2>&1 | tee -a "${LOG_FILE}"
    return ${PIPESTATUS[0]}
}

echo "========================================"
echo "VAE Encoder Profiling Tests (3 warmup, 10 iterations)"
echo "========================================"
echo "Log file: ${LOG_FILE}"
echo ""

# VAE Encoder On 4 GPUs (GPU 4-7)
echo ""
echo ">>> Testing VAE Encoder On 4 GPUs (CUDA 4-7)..."
export CUDA_VISIBLE_DEVICES=4,5,6,7
mkdir -p "${OUT_DIR}/gpu4"
run_with_log torchrun --standalone --nproc_per_node 4 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-encode \
  --device cuda \
  --num-gpus 4 \
  --tp-size 4 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --vae-encode-input-frames 1 \
  --sync-stage-profiling all \
  --warmup-iters 3 \
  --profile-iters 10 \
  --output-json "${OUT_DIR}/gpu4/profile.json" \
  --output-txt "${OUT_DIR}/gpu4/profile.txt"

# VAE Encoder On 2 GPUs (GPU 6-7)
echo ""
echo ">>> Testing VAE Encoder On 2 GPUs (CUDA 6-7)..."
export CUDA_VISIBLE_DEVICES=6,7
mkdir -p "${OUT_DIR}/gpu2"
run_with_log torchrun --standalone --nproc_per_node 2 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-encode \
  --device cuda \
  --num-gpus 2 \
  --tp-size 2 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --vae-encode-input-frames 1 \
  --sync-stage-profiling all \
  --warmup-iters 3 \
  --profile-iters 10 \
  --output-json "${OUT_DIR}/gpu2/profile.json" \
  --output-txt "${OUT_DIR}/gpu2/profile.txt"

# VAE Encoder On 1 GPU (GPU 7)
echo ""
echo ">>> Testing VAE Encoder On 1 GPU (CUDA 7)..."
export CUDA_VISIBLE_DEVICES=7
mkdir -p "${OUT_DIR}/gpu1"
run_with_log torchrun --standalone --nproc_per_node 1 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-encode \
  --device cuda \
  --num-gpus 1 \
  --tp-size 1 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --vae-encode-input-frames 1 \
  --sync-stage-profiling all \
  --warmup-iters 3 \
  --profile-iters 10 \
  --output-json "${OUT_DIR}/gpu1/profile.json" \
  --output-txt "${OUT_DIR}/gpu1/profile.txt"

# VAE Encoder On CPU
echo ""
echo ">>> Testing VAE Encoder On CPU..."
unset CUDA_VISIBLE_DEVICES
mkdir -p "${OUT_DIR}/cpu"
run_with_log python scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-encode \
  --device cpu \
  --prompt "${PROMPT}" \
  --vae-encode-input-frames 1 \
  --warmup-iters 3 \
  --profile-iters 10 \
  --output-json "${OUT_DIR}/cpu/profile.json" \
  --output-txt "${OUT_DIR}/cpu/profile.txt"

echo ""
echo "========================================"
echo "All VAE Encoder Tests Completed!"
echo "========================================"
echo ""
echo "Results summary:"
echo "  GPU4 (4-7): ${OUT_DIR}/gpu4/profile.json"
echo "  GPU2 (6-7): ${OUT_DIR}/gpu2/profile.json"
echo "  GPU1 (7):   ${OUT_DIR}/gpu1/profile.json"
echo "  CPU:        ${OUT_DIR}/cpu/profile.json"
