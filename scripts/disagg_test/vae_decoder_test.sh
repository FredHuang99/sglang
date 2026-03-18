#!/bin/bash
# Script to profile VAE Decoder stage with different GPU configurations
# Uses the same method as DiT profiling (runs full pipeline, measures DecodingStage only)
# Results saved to /workspace/wan22_ti2v5b_profile/vae_decoder

set -e

export MODEL="/workspace/models/Wan2_2-TI2V-5B-Diffusers"
export PROMPT="A cinematic science-fiction city with reflective rain streets."
export OUT_DIR="/workspace/wan22_ti2v5b_profile/vae_decoder"
export LOG_DIR="/workspace/sglang/logs"
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

# Log file with timestamp
LOG_FILE="${LOG_DIR}/vae_decoder_test_$(date +%Y%m%d_%H%M%S).log"

# Function to run command with logging
run_with_log() {
    "$@" 2>&1 | tee -a "${LOG_FILE}"
    return ${PIPESTATUS[0]}
}

echo "========================================"
echo "VAE Decoder Profiling Tests"
echo "========================================"
echo "Log file: ${LOG_FILE}"
echo ""

# ========================================
# VAE Decoder On 8 GPUs (GPU 0-7)
# ========================================
echo ""
echo ">>> Testing VAE Decoder On 8 GPUs (CUDA 0-7)..."
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p "${OUT_DIR}/gpu8"
run_with_log torchrun --standalone --nproc_per_node 8 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-decode \
  --device cuda \
  --num-gpus 8 \
  --sp-degree 8 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --num-frames 121 \
  --height 704 \
  --width 1280 \
  --num-inference-steps 50 \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/gpu8/profile.json" \
  --output-txt "${OUT_DIR}/gpu8/profile.txt"

# ========================================
# VAE Decoder On 4 GPUs (GPU 4-7)
# ========================================
echo ""
echo ">>> Testing VAE Decoder On 4 GPUs (CUDA 4-7)..."
export CUDA_VISIBLE_DEVICES=4,5,6,7
mkdir -p "${OUT_DIR}/gpu4"
run_with_log torchrun --standalone --nproc_per_node 4 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-decode \
  --device cuda \
  --num-gpus 4 \
  --sp-degree 4 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --num-frames 121 \
  --height 704 \
  --width 1280 \
  --num-inference-steps 50 \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/gpu4/profile.json" \
  --output-txt "${OUT_DIR}/gpu4/profile.txt"

# ========================================
# VAE Decoder On 2 GPUs (GPU 6-7)
# ========================================
echo ""
echo ">>> Testing VAE Decoder On 2 GPUs (CUDA 6-7)..."
export CUDA_VISIBLE_DEVICES=6,7
mkdir -p "${OUT_DIR}/gpu2"
run_with_log torchrun --standalone --nproc_per_node 2 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-decode \
  --device cuda \
  --num-gpus 2 \
  --sp-degree 2 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --num-frames 121 \
  --height 704 \
  --width 1280 \
  --num-inference-steps 50 \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/gpu2/profile.json" \
  --output-txt "${OUT_DIR}/gpu2/profile.txt"

# ========================================
# VAE Decoder On 1 GPU (GPU 7)
# ========================================
echo ""
echo ">>> Testing VAE Decoder On 1 GPU (CUDA 7)..."
export CUDA_VISIBLE_DEVICES=7
mkdir -p "${OUT_DIR}/gpu1"
run_with_log torchrun --standalone --nproc_per_node 1 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-decode \
  --device cuda \
  --num-gpus 1 \
  --sp-degree 1 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --num-frames 121 \
  --height 704 \
  --width 1280 \
  --num-inference-steps 50 \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/gpu1/profile.json" \
  --output-txt "${OUT_DIR}/gpu1/profile.txt"

# ========================================
# VAE Decoder On CPU
# ========================================
echo ""
echo ">>> Testing VAE Decoder On CPU..."
unset CUDA_VISIBLE_DEVICES
mkdir -p "${OUT_DIR}/cpu"
run_with_log python scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-decode \
  --device cpu \
  --prompt "${PROMPT}" \
  --num-frames 121 \
  --height 704 \
  --width 1280 \
  --num-inference-steps 50 \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/cpu/profile.json" \
  --output-txt "${OUT_DIR}/cpu/profile.txt"

echo ""
echo "========================================"
echo "All VAE Decoder Tests Completed!"
echo "========================================"
echo ""
echo "Results summary:"
echo "  GPU8 (0-7): ${OUT_DIR}/gpu8/profile.json"
echo "  GPU4 (4-7): ${OUT_DIR}/gpu4/profile.json"
echo "  GPU2 (6-7): ${OUT_DIR}/gpu2/profile.json"
echo "  GPU1 (7):   ${OUT_DIR}/gpu1/profile.json"
echo "  CPU:        ${OUT_DIR}/cpu/profile.json"
