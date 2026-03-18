#!/bin/bash
# Script to profile DiT stage on 6 GPUs
# Tests all legal (ulysses, ring) combinations for 6-GPU configuration
# Order: 6 GPUs (all configs in descending ulysses order)

set -e

export MODEL="/workspace/models/Wan2_2-TI2V-5B-Diffusers"
export PROMPT="A cinematic science-fiction city with reflective rain streets."
export OUT_DIR="/workspace/wan22_ti2v5b_profile/dit"
export LOG_DIR="/workspace/sglang/logs"
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/dit_6gpu_test_$(date +%Y%m%d_%H%M%S).log"

run_with_log() {
    "$@" 2>&1 | tee -a "${LOG_FILE}"
    return ${PIPESTATUS[0]}
}

echo "========================================"
echo "DiT Profiling Tests on 6 GPUs"
echo "========================================"
echo "Model: ${MODEL}"
echo "Output: ${OUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Warmup: 1, Profile: 3"
echo "GPUs: CUDA 0,1,2,3,4,5"
echo "========================================"
echo ""

# Check available GPUs
echo ">>> Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv -i 0,1,2,3,4,5 2>/dev/null || echo "Warning: Cannot query all 6 GPUs"
echo ""

# ========================================
# Section 1: 6 GPU Tests (All Configurations)
# ========================================
echo ">>>========================================"
echo ">>> Section 1: 6 GPU Tests"
echo ">>>========================================"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# Configuration 1.1: (ulysses, ring) = (2, 3)
echo ""
echo ">>> Test 1.1: DiT on 6 GPUs (ulysses=2, ring=3)..."
echo ">>> Description: Hybrid parallelism - 2-way Ulysses + 3-way Ring"
mkdir -p "${OUT_DIR}/gpu6_u2_r3"
run_with_log torchrun --standalone --nproc_per_node 6 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 6 \
  --sp-degree 6 \
  --ulysses-degree 2 \
  --ring-degree 3 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/gpu6_u2_r3/profile.json" \
  --output-txt "${OUT_DIR}/gpu6_u2_r3/profile.txt"

echo ""
echo "✓ Test 1.1 completed: gpu6_u2_r3"

# Configuration 1.2: (ulysses, ring) = (1, 6)
echo ""
echo ">>> Test 1.2: DiT on 6 GPUs (ulysses=1, ring=6)..."
echo ">>> Description: Full Ring attention, no Ulysses"
mkdir -p "${OUT_DIR}/gpu6_u1_r6"
run_with_log torchrun --standalone --nproc_per_node 6 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 6 \
  --sp-degree 6 \
  --ulysses-degree 1 \
  --ring-degree 6 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/gpu6_u1_r6/profile.json" \
  --output-txt "${OUT_DIR}/gpu6_u1_r6/profile.txt"

echo ""
echo "✓ Test 1.2 completed: gpu6_u1_r6"

# ========================================
# Summary
# ========================================
echo ""
echo "========================================"
echo "All DiT 6-GPU Tests Completed!"
echo "========================================"
echo ""
echo "Results directory: ${OUT_DIR}"
echo ""
echo "Configurations tested (2 total):"
echo ""
echo "Test 1.1: (ulysses=2, ring=3) -> ${OUT_DIR}/gpu6_u2_r3/"
echo "         Hybrid: 2-way Ulysses + 3-way Ring"
echo ""
echo "Test 1.2: (ulysses=1, ring=6) -> ${OUT_DIR}/gpu6_u1_r6/"
echo "         Full Ring parallelism across 6 GPUs"
echo ""
echo "Log file: ${LOG_FILE}"
echo "========================================"
echo ""
echo "Note: Both configurations are legal for Wan2.2-TI2V-5B"
echo "      because the model has 40 attention heads, and"
echo "      both [2, 1] divide 40 evenly."
