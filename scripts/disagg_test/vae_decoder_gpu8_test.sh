#!/bin/bash
# Script to profile VAE Decoder on 8 GPUs with GPU availability check
# Waits until all 8 GPUs are free before starting the test
# Results saved to /workspace/wan22_ti2v5b_profile/vae_decoder_gpu8

set -e

export MODEL="/workspace/models/Wan2_2-TI2V-5B-Diffusers"
export PROMPT="A cinematic science-fiction city with reflective rain streets."
export OUT_DIR="/workspace/wan22_ti2v5b_profile/vae_decoder_gpu8"
export LOG_DIR="/workspace/sglang/logs"
export GPU_IDS="0,1,2,3,4,5,6,7"
export REQUIRED_GPUS=8

mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

# Log file with timestamp
LOG_FILE="${LOG_DIR}/vae_decoder_gpu8_test_$(date +%Y%m%d_%H%M%S).log"

# Function to check if GPUs are free
# Returns 0 if all required GPUs are free, 1 otherwise
check_gpus_free() {
    local gpu_ids=($(echo "$1" | tr ',' ' '))
    local required=$2

    for gpu_id in "${gpu_ids[@]}"; do
        # Check GPU memory usage and running processes
        # Using nvidia-smi to check if GPU is occupied
        gpu_info=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "$gpu_id,0,0")

        memory_used=$(echo "$gpu_info" | awk -F',' '{print $2}' | tr -d ' ')
        utilization=$(echo "$gpu_info" | awk -F',' '{print $3}' | tr -d ' ')

        # Consider GPU occupied if memory used > 100MB or utilization > 5%
        if [ "$memory_used" -gt 100 ] || [ "$utilization" -gt 5 ]; then
            return 1
        fi
    done

    return 0
}

# Function to wait for GPUs to be free
wait_for_gpus() {
    local gpu_ids="$1"
    local required="$2"
    local max_wait_time=86400  # Maximum wait time in seconds (24 hours)
    local start_time=$(date +%s)
    local check_interval=30   # Check every 60 seconds (1 minute)

    echo "Waiting for GPUs $gpu_ids to be free..."
    while ! check_gpus_free "$gpu_ids" "$required"; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))

        if [ $elapsed -ge $max_wait_time ]; then
            echo "Maximum wait time reached ($max_wait_time seconds). Aborting."
            exit 1
        fi

        # Show GPU status
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for GPUs... (elapsed: ${elapsed}s)"
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu_ids"

        sleep $check_interval
    done

    echo "All GPUs are now free! Starting test..."
}

# Function to run command with logging
run_with_log() {
    "$@" 2>&1 | tee -a "${LOG_FILE}"
    return ${PIPESTATUS[0]}
}

echo "========================================"
echo "VAE Decoder 8-GPU Profiling Test"
echo "========================================"
echo "Log file: ${LOG_FILE}"
echo "Required GPUs: ${GPU_IDS}"
echo ""

# Wait for GPUs to be free
wait_for_gpus "$GPU_IDS" "$REQUIRED_GPUS"

# Run the test
echo ""
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
