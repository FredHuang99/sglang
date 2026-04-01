#!/usr/bin/env bash
set -euo pipefail

# Benchmark PromptEnhancer/PromptEnhancer-32B with sglang.bench_serving.
# The script launches one server per TP size, runs burst benchmarks over
# input/output/batch-size sweeps, and stores logs / JSONL / profiler traces in a
# unique run directory so repeated runs do not overwrite each other.

# /home/heyang/models/promptenhancer-32b
MODEL_PATH="${MODEL_PATH:-/home/heyang/models/promptenhancer-7b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"

INPUT_LEN="${INPUT_LEN:-128 256}"
OUTPUT_LEN="${OUTPUT_LEN:-256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048}"
#CTX_LEN="${CTX_LEN:-128000}"
CTX_LEN="${CTX_LEN:-32768}" 
WARMUP_REQUESTS="${WARMUP_REQUESTS:-5}"
SEED="${SEED:-42}"

# Tune these to maximize KV pool and let runtime split large batches.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
#CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
#CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-1}"
#DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
#MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-}"

#TP_SIZE="${TP_SIZE:-1 2 4 8}"
TP_SIZE="${TP_SIZE:-1}"
BS_LIST_STRING="${BS_LIST:-1 2 4 8 16 32}"

ENABLE_PROFILE="${ENABLE_PROFILE:-1}"
PROFILE_ACTIVITIES="${PROFILE_ACTIVITIES:-CPU GPU}"
PROFILE_NUM_STEPS="${PROFILE_NUM_STEPS:-}"
PROFILE_BY_STAGE="${PROFILE_BY_STAGE:-0}"
PROFILE_STAGES="${PROFILE_STAGES:-}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/home/heyang/profile_output/${RUN_ID}}"
RESULT_JSONL="${RUN_ROOT}/bench_serving_results.jsonl"

SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}

wait_server_ready() {
  local url="$1"
  local timeout_s="${2:-600}"
  local pid="${3:-}"
  local log_path="${4:-}"
  local start_ts
  start_ts="$(date +%s)"

  while true; do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "Server process exited before becoming ready." >&2
      if [[ -n "${log_path}" ]] && [[ -f "${log_path}" ]]; then
        echo "Last 200 lines of ${log_path}:" >&2
        tail -n 200 "${log_path}" >&2 || true
      fi
      return 1
    fi
    if (( "$(date +%s)" - start_ts >= timeout_s )); then
      if [[ -n "${log_path}" ]] && [[ -f "${log_path}" ]]; then
        echo "Timed out waiting for server. Last 200 lines of ${log_path}:" >&2
        tail -n 200 "${log_path}" >&2 || true
      fi
      return 1
    fi
    sleep 1
  done
}

trap cleanup EXIT INT TERM

mkdir -p "${RUN_ROOT}"
: > "${RESULT_JSONL}"

read -r -a INPUT_LEN_LIST <<< "${INPUT_LEN}"
read -r -a OUTPUT_LEN_LIST <<< "${OUTPUT_LEN}"
read -r -a TP_SIZE_LIST <<< "${TP_SIZE}"
read -r -a BS_LIST <<< "${BS_LIST_STRING}"
read -r -a PROFILE_ACTIVITY_ARR <<< "${PROFILE_ACTIVITIES}"
if [[ -n "${PROFILE_STAGES}" ]]; then
  read -r -a PROFILE_STAGE_ARR <<< "${PROFILE_STAGES}"
else
  PROFILE_STAGE_ARR=()
fi

echo "Run root: ${RUN_ROOT}"
echo "Summary JSONL: ${RESULT_JSONL}"

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
}

for TP in "${TP_SIZE_LIST[@]}"; do
  TP_DIR="${RUN_ROOT}/tp${TP}"
  SERVER_LOG="${TP_DIR}/server.log"
  mkdir -p "${TP_DIR}"

  export SGLANG_TORCH_PROFILER_DIR="${TP_DIR}/profiles_root"

  SERVER_ARGS=(
    -m sglang.launch_server
    --model-path "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --trust-remote-code
    --context-length "${CTX_LEN}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --tp-size "${TP}"
    --disable-piecewise-cuda-graph
    --cuda-graph-max-bs 32
  )

  #if [[ -n "${CHUNKED_PREFILL_SIZE:-}" ]]; then
  #  SERVER_ARGS+=(--chunked-prefill-size "${CHUNKED_PREFILL_SIZE}")
  #fi
  #
  #if [[ -n "${CUDA_GRAPH_MAX_BS:-}" ]]; then
  #  SERVER_ARGS+=(--cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}")
  #fi
  #
  #if [[ "${DISABLE_CUDA_GRAPH}" == "1" ]]; then
  #  SERVER_ARGS+=(--disable-cuda-graph)
  #fi
  #
  #if [[ -n "${MAX_PREFILL_TOKENS}" ]]; then
  #  SERVER_ARGS+=(--max-prefill-tokens "${MAX_PREFILL_TOKENS}")
  #fi

  echo
  echo "========================================"
  echo "Tensor parallel size: ${TP}"
  echo "Server log: ${SERVER_LOG}"
  echo "========================================"

  python "${SERVER_ARGS[@]}" > "${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!

  if ! wait_server_ready "http://${HOST}:${PORT}/get_server_info" 600 "${SERVER_PID}" "${SERVER_LOG}"; then
    echo "Server failed to become ready. See ${SERVER_LOG}" >&2
    stop_server
    exit 1
  fi

  for IL in "${INPUT_LEN_LIST[@]}"; do
    for OL in "${OUTPUT_LEN_LIST[@]}"; do
      for BS in "${BS_LIST[@]}"; do
        COMBO_DIR="${TP_DIR}/input${IL}_output${OL}_bs${BS}"
        PROFILE_ROOT="${COMBO_DIR}/profiles"
        CLIENT_LOG="${COMBO_DIR}/client.log"
        PROFILE_PREFIX="promptenhancer_tp${TP}_in${IL}_out${OL}_bs${BS}_${RUN_ID}"

        mkdir -p "${COMBO_DIR}" "${PROFILE_ROOT}"

        echo
        echo "----------------------------------------"
        echo "TP: ${TP}, input: ${IL}, output: ${OL}, bs: ${BS}"
        echo "Client log: ${CLIENT_LOG}"
        echo "Profile root: ${PROFILE_ROOT}"
        echo "----------------------------------------"

        CLIENT_ARGS=(
          -m sglang.bench_serving
          --backend sglang
          --host "${HOST}"
          --port "${PORT}"
          --model "${MODEL_PATH}"
          --dataset-name random
          --num-prompts "${BS}"
          --random-input-len "${IL}"
          --random-output-len "${OL}"
          --request-rate inf
          --max-concurrency "${BS}"
          --warmup-requests "${WARMUP_REQUESTS}"
          --seed "${SEED}"
          --output-file "${RESULT_JSONL}"
        )

        if [[ "${ENABLE_PROFILE}" == "1" ]]; then
          CLIENT_ARGS+=(
            --profile
            --profile-output-dir "${PROFILE_ROOT}"
            --profile-prefix "${PROFILE_PREFIX}"
            --profile-activities "${PROFILE_ACTIVITY_ARR[@]}"
          )

          if [[ -n "${PROFILE_NUM_STEPS}" ]]; then
            CLIENT_ARGS+=(--profile-num-steps "${PROFILE_NUM_STEPS}")
          fi

          if [[ "${PROFILE_BY_STAGE}" == "1" ]]; then
            CLIENT_ARGS+=(--profile-by-stage)
          fi

          if [[ "${#PROFILE_STAGE_ARR[@]}" -gt 0 ]]; then
            CLIENT_ARGS+=(--profile-stages "${PROFILE_STAGE_ARR[@]}")
          fi
        fi

        python "${CLIENT_ARGS[@]}" | tee "${CLIENT_LOG}"
      done
    done
  done

  stop_server
  sleep 3
done

echo
echo "Done."
echo "Run root: ${RUN_ROOT}"
echo "Summary JSONL: ${RESULT_JSONL}"
