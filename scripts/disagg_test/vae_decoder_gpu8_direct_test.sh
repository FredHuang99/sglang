#!/bin/bash
# Script to profile VAE Decoder on 8 GPUs (direct execution, no GPU waiting)
# Results saved to /workspace/wan22_ti2v5b_profile/vae_decoder_gpu8

set -e

export MODEL="/workspace/models/Wan2_2-TI2V-5B-Diffusers"
export PROMPT="A cinematic science-fiction city with reflective rain streets."
export OUT_DIR="/workspace/wan22_ti2v5b_profile/vae_decoder_gpu8"
export LOG_DIR="/workspace/sglang/logs"
export GPU_IDS="0,1,2,3,4,5,6,7"

mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

# Log file with timestamp
LOG_FILE="${LOG_DIR}/vae_decoder_gpu8_direct_test_$(date +%Y%m%d_%H%M%S).log"

# Function to run command with logging
run_with_log() {
    "$@" 2>&1 | tee -a "${LOG_FILE}"
    return ${PIPESTATUS[0]}
}

echo "========================================"
echo "VAE Decoder 8-GPU Profiling Test"
echo "========================================"
echo "Model: ${MODEL}"
echo "GPUs: CUDA ${GPU_IDS}"
echo "Output: ${OUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "========================================"
echo ""

# Check GPU status (informational only, no waiting)
echo ">>> Current GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv -i "$GPU_IDS" 2>/dev/null || echo "Warning: Cannot query all 8 GPUs"
echo ""

# Run the test directly
echo ">>> Testing VAE Decoder On 8 GPUs (CUDA ${GPU_IDS})..."
export CUDA_VISIBLE_DEVICES=$GPU_IDS
mkdir -p "${OUT_DIR}/gpu8"

# Record start time
START_TIME=$(date +%s)

run_with_log torchrun --standalone --nproc_per_node 8 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage vae-decode \
  --device cuda \
  --num-gpus 8 \
  --sp-degree 8 \
  --ulysses-degree 8 \
  --ring-degree 1 \
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

# Record end time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo "VAE Decoder 8-GPU Test Completed!"
echo "========================================"
echo "Total time: ${TOTAL_TIME} seconds"
echo "Results: ${OUT_DIR}/gpu8/profile.json"
echo "Log file: ${LOG_FILE}"
echo "========================================"
