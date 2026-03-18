#!/bin/bash
# Script to profile DiT stage on 8 GPUs
# Tests all legal (ulysses, ring) combinations for 8-GPU configuration
# Order: 8 GPUs (all configs in descending ulysses order)

set -e

export MODEL="/workspace/models/Wan2_2-TI2V-5B-Diffusers"
export PROMPT="A cinematic science-fiction city with reflective rain streets."
export OUT_DIR="/workspace/wan22_ti2v5b_profile/dit"
export LOG_DIR="/workspace/sglang/logs"
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/dit_8gpu_test_$(date +%Y%m%d_%H%M%S).log"

run_with_log() {
    "$@" 2>&1 | tee -a "${LOG_FILE}"
    return ${PIPESTATUS[0]}
}

echo "========================================"
echo "DiT Profiling Tests on 8 GPUs"
echo "========================================"
echo "Model: ${MODEL}"
echo "Output: ${OUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Warmup: 1, Profile: 3"
echo "GPUs: CUDA 0,1,2,3,4,5,6,7"
echo "========================================"
echo ""

# Check available GPUs
echo ">>> Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv -i 0,1,2,3,4,5,6,7 2>/dev/null || echo "Warning: Cannot query all 8 GPUs"
echo ""

# ========================================
# Section 1: 8 GPU Tests (All Configurations)
# ========================================
echo ">>>========================================"
echo ">>> Section 1: 8 GPU Tests"
echo ">>>========================================"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Configuration 1.1: (ulysses, ring) = (8, 1)
echo ""
echo ">>> Test 1.1: DiT on 8 GPUs (ulysses=8, ring=1)..."
echo ">>> Description: Full Ulysses attention, no ring"
mkdir -p "${OUT_DIR}/gpu8_u8_r1"
run_with_log torchrun --standalone --nproc_per_node 8 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 8 \
  --sp-degree 8 \
  --ulysses-degree 8 \
  --ring-degree 1 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/gpu8_u8_r1/profile.json" \
  --output-txt "${OUT_DIR}/gpu8_u8_r1/profile.txt"

echo ""
echo "✓ Test 1.1 completed: gpu8_u8_r1"

# Configuration 1.2: (ulysses, ring) = (4, 2)
echo ""
echo ">>> Test 1.2: DiT on 8 GPUs (ulysses=4, ring=2)..."
echo ">>> Description: Hybrid parallelism - 4-way Ulysses + 2-way Ring"
mkdir -p "${OUT_DIR}/gpu8_u4_r2"
run_with_log torchrun --standalone --nproc_per_node 8 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 8 \
  --sp-degree 8 \
  --ulysses-degree 4 \
  --ring-degree 2 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/gpu8_u4_r2/profile.json" \
  --output-txt "${OUT_DIR}/gpu8_u4_r2/profile.txt"

echo ""
echo "✓ Test 1.2 completed: gpu8_u4_r2"

# Configuration 1.3: (ulysses, ring) = (2, 4)
echo ""
echo ">>> Test 1.3: DiT on 8 GPUs (ulysses=2, ring=4)..."
echo ">>> Description: Hybrid parallelism - 2-way Ulysses + 4-way Ring"
mkdir -p "${OUT_DIR}/gpu8_u2_r4"
run_with_log torchrun --standalone --nproc_per_node 8 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 8 \
  --sp-degree 8 \
  --ulysses-degree 2 \
  --ring-degree 4 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/gpu8_u2_r4/profile.json" \
  --output-txt "${OUT_DIR}/gpu8_u2_r4/profile.txt"

echo ""
echo "✓ Test 1.3 completed: gpu8_u2_r4"

# Configuration 1.4: (ulysses, ring) = (1, 8)
echo ""
echo ">>> Test 1.4: DiT on 8 GPUs (ulysses=1, ring=8)..."
echo ">>> Description: Full Ring attention, no Ulysses"
mkdir -p "${OUT_DIR}/gpu8_u1_r8"
run_with_log torchrun --standalone --nproc_per_node 8 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 8 \
  --sp-degree 8 \
  --ulysses-degree 1 \
  --ring-degree 8 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/gpu8_u1_r8/profile.json" \
  --output-txt "${OUT_DIR}/gpu8_u1_r8/profile.txt"

echo ""
echo "✓ Test 1.4 completed: gpu8_u1_r8"

# ========================================
# Summary
# ========================================
echo ""
echo "========================================"
echo "All DiT 8-GPU Tests Completed!"
echo "========================================"
echo ""
echo "Results directory: ${OUT_DIR}"
echo ""
echo "Configurations tested (4 total):"
echo ""
echo "Test 1.1: (ulysses=8, ring=1) -> ${OUT_DIR}/gpu8_u8_r1/"
echo "         Full Ulysses parallelism across 8 GPUs"
echo ""
echo "Test 1.2: (ulysses=4, ring=2) -> ${OUT_DIR}/gpu8_u4_r2/"
echo "         Hybrid: 4-way Ulysses + 2-way Ring"
echo ""
echo "Test 1.3: (ulysses=2, ring=4) -> ${OUT_DIR}/gpu8_u2_r4/"
echo "         Hybrid: 2-way Ulysses + 4-way Ring"
echo ""
echo "Test 1.4: (ulysses=1, ring=8) -> ${OUT_DIR}/gpu8_u1_r8/"
echo "         Full Ring parallelism across 8 GPUs"
echo ""
echo "Log file: ${LOG_FILE}"
echo "========================================"
echo ""
echo "Note: All configurations are legal for Wan2.2-TI2V-5B"
echo "      because the model has 40 attention heads, and"
echo "      [8, 4, 2, 1] all divide 40 evenly."
