#!/bin/bash
# Script to profile DiT stage on 4/2/1 GPUs (descending order)
# Tests all legal (ulysses, ring) combinations
# Order: 4 GPUs -> 2 GPUs -> 1 GPU

set -e

export MODEL="/workspace/models/Wan2_2-TI2V-5B-Diffusers"
export PROMPT="A cinematic science-fiction city with reflective rain streets."
export OUT_DIR="/workspace/wan22_ti2v5b_profile/dit"
export LOG_DIR="/workspace/sglang/logs"
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/dit_$(date +%Y%m%d_%H%M%S).log"

run_with_log() {
    "$@" 2>&1 | tee -a "${LOG_FILE}"
    return ${PIPESTATUS[0]}
}

echo "========================================"
echo "DiT Profiling Tests on 4/2/1 GPUs"
echo "========================================"
echo "Order: 4 GPUs -> 2 GPUs -> 1 GPU"
echo "Model: ${MODEL}"
echo "Output: ${OUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Warmup: 1, Profile: 3"
echo "GPUs: CUDA 4,5,6,7"
echo "========================================"
echo ""

# ========================================
# Section 1: 4 GPU Tests (First)
# ========================================
echo ""
echo ">>>========================================"
echo ">>> Section 1: 4 GPU Tests"
echo ">>>========================================"
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Configuration 1.1: (ulysses, ring) = (4, 1)
echo ""
echo ">>> Test 1.1: DiT on 4 GPUs (ulysses=4, ring=1)..."
mkdir -p "${OUT_DIR}/gpu4_u4_r1"
run_with_log torchrun --standalone --nproc_per_node 4 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 4 \
  --sp-degree 4 \
  --ulysses-degree 4 \
  --ring-degree 1 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/gpu4_u4_r1/profile.json" \
  --output-txt "${OUT_DIR}/gpu4_u4_r1/profile.txt"

echo ""
echo "✓ Test 1.1 completed: gpu4_u4_r1"

# Configuration 1.2: (ulysses, ring) = (2, 2)
echo ""
echo ">>> Test 1.2: DiT on 4 GPUs (ulysses=2, ring=2)..."
mkdir -p "${OUT_DIR}/gpu4_u2_r2"
run_with_log torchrun --standalone --nproc_per_node 4 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 4 \
  --sp-degree 4 \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/gpu4_u2_r2/profile.json" \
  --output-txt "${OUT_DIR}/gpu4_u2_r2/profile.txt"

echo ""
echo "✓ Test 1.2 completed: gpu4_u2_r2"

# Configuration 1.3: (ulysses, ring) = (1, 4)
echo ""
echo ">>> Test 1.3: DiT on 4 GPUs (ulysses=1, ring=4)..."
mkdir -p "${OUT_DIR}/gpu4_u1_r4"
run_with_log torchrun --standalone --nproc_per_node 4 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 4 \
  --sp-degree 4 \
  --ulysses-degree 1 \
  --ring-degree 4 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/gpu4_u1_r4/profile.json" \
  --output-txt "${OUT_DIR}/gpu4_u1_r4/profile.txt"

echo ""
echo "✓ Test 1.3 completed: gpu4_u1_r4"

# ========================================
# Section 2: 2 GPU Tests (Second)
# ========================================
echo ""
echo ">>>========================================"
echo ">>> Section 2: 2 GPU Tests"
echo ">>>========================================"
export CUDA_VISIBLE_DEVICES=4,5

# Configuration 2.1: (ulysses, ring) = (2, 1)
echo ""
echo ">>> Test 2.1: DiT on 2 GPUs (ulysses=2, ring=1)..."
mkdir -p "${OUT_DIR}/gpu2_u2_r1"
run_with_log torchrun --standalone --nproc_per_node 2 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/gpu2_u2_r1/profile.json" \
  --output-txt "${OUT_DIR}/gpu2_u2_r1/profile.txt"

echo ""
echo "✓ Test 2.1 completed: gpu2_u2_r1"

# Configuration 2.2: (ulysses, ring) = (1, 2)
echo ""
echo ">>> Test 2.2: DiT on 2 GPUs (ulysses=1, ring=2)..."
mkdir -p "${OUT_DIR}/gpu2_u1_r2"
run_with_log torchrun --standalone --nproc_per_node 2 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 1 \
  --ring-degree 2 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/gpu2_u1_r2/profile.json" \
  --output-txt "${OUT_DIR}/gpu2_u1_r2/profile.txt"

echo ""
echo "✓ Test 2.2 completed: gpu2_u1_r2"

# ========================================
# Section 3: 1 GPU Test (Last)
# ========================================
echo ""
echo ">>>========================================"
echo ">>> Section 3: 1 GPU Test"
echo ">>>========================================"
export CUDA_VISIBLE_DEVICES=4

# Configuration: (ulysses, ring) = (1, 1)
echo ""
echo ">>> Test 3.1: DiT on 1 GPU (ulysses=1, ring=1)..."
mkdir -p "${OUT_DIR}/gpu1_u1_r1"
run_with_log torchrun --standalone --nproc_per_node 1 scripts/playground/profile_wan_ti2v_stages.py \
  --model-path "${MODEL}" \
  --stage dit \
  --device cuda \
  --num-gpus 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --attention-backend fa \
  --prompt "${PROMPT}" \
  --sync-stage-profiling all \
  --warmup-iters 1 \
  --profile-iters 3 \
  --output-json "${OUT_DIR}/gpu1_u1_r1/profile.json" \
  --output-txt "${OUT_DIR}/gpu1_u1_r1/profile.txt"

echo ""
echo "✓ Test 3.1 completed: gpu1_u1_r1"

# ========================================
# Summary
# ========================================
echo ""
echo "========================================"
echo "All DiT Multi-GPU Tests Completed!"
echo "========================================"
echo ""
echo "Execution order: 4 GPUs -> 2 GPUs -> 1 GPU"
echo ""
echo "Results directory: ${OUT_DIR}"
echo ""
echo "Configurations tested:"
echo ""
echo "4 GPUs (Section 1):"
echo "  - (ulysses=4, ring=1) -> ${OUT_DIR}/gpu4_u4_r1/"
echo "  - (ulysses=2, ring=2) -> ${OUT_DIR}/gpu4_u2_r2/"
echo "  - (ulysses=1, ring=4) -> ${OUT_DIR}/gpu4_u1_r4/"
echo ""
echo "2 GPUs (Section 2):"
echo "  - (ulysses=2, ring=1) -> ${OUT_DIR}/gpu2_u2_r1/"
echo "  - (ulysses=1, ring=2) -> ${OUT_DIR}/gpu2_u1_r2/"
echo ""
echo "1 GPU (Section 3):"
echo "  - (ulysses=1, ring=1) -> ${OUT_DIR}/gpu1_u1_r1/"
echo ""
echo "Log file: ${LOG_FILE}"
echo "========================================"
