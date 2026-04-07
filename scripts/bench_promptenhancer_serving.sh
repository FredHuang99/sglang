#!/usr/bin/env bash
set -euo pipefail

# Benchmark PromptEnhancer/PromptEnhancer-32B with sglang.bench_serving.
# The script launches one server per TP size, runs burst benchmarks over
# input/output/batch-size sweeps, and stores logs / JSONL / profiler traces in a
# unique run directory so repeated runs do not overwrite each other.

MODEL_PATH="${MODEL_PATH:-/workspace/models/Hunyuan_PromptEnhancer_32B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
DATASET_NAME="${DATASET_NAME:-random-ids}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-1.0}"

INPUT_LEN="${INPUT_LEN:-128 256}"
# 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1920 2048 2176
OUTPUT_LEN="${OUTPUT_LEN:-384 512 640 768 896 1024 1152}"
CTX_LEN="${CTX_LEN:-32768}" 
WARMUP_REQUESTS="${WARMUP_REQUESTS:-5}"
READY_CHECK_TIMEOUT="${READY_CHECK_TIMEOUT:-600}"
SERVER_BOOTSTRAP_GRACE_SEC="${SERVER_BOOTSTRAP_GRACE_SEC:-5}"
SEED="${SEED:-42}"

# Tune these to maximize KV pool and let runtime split large batches.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
#CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
#CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-1}"
#DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
#MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-}"

#TP_SIZE="${TP_SIZE:-1 2 4 8}"
TP_SIZE="${TP_SIZE:-2 4}"
BS_LIST_STRING="${BS_LIST:-1 2 4 8 16 32}"

ENABLE_PROFILE="${ENABLE_PROFILE:-0}"
DISABLE_TQDM="${DISABLE_TQDM:-0}"
PROFILE_ACTIVITIES="${PROFILE_ACTIVITIES:-CPU GPU}"
PROFILE_NUM_STEPS="${PROFILE_NUM_STEPS:-}"
PROFILE_BY_STAGE="${PROFILE_BY_STAGE:-0}"
PROFILE_STAGES="${PROFILE_STAGES:-}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/workspace/outputs/pe32b/run/${RUN_ID}}"
RESULT_JSONL="${RUN_ROOT}/bench_serving_results.jsonl"
SUMMARY_CSV="${RUN_ROOT}/bench_serving_summary.csv"

SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}

print_server_log_tail() {
  local log_path="$1"

  if [[ -n "${log_path}" ]] && [[ -f "${log_path}" ]]; then
    echo "Last 200 lines of ${log_path}:" >&2
    tail -n 200 "${log_path}" >&2 || true
  fi
}

ensure_server_bootstrapped() {
  local grace_s="$1"
  local pid="${2:-}"
  local log_path="${3:-}"

  for ((sec = 1; sec <= grace_s; sec++)); do
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "Server process exited during bootstrap grace window." >&2
      print_server_log_tail "${log_path}"
      return 1
    fi
    sleep 1
  done

  if [[ -n "${pid}" ]] && ! kill -0 "${pid}" >/dev/null 2>&1; then
    echo "Server process exited before benchmark handoff." >&2
    print_server_log_tail "${log_path}"
    return 1
  fi

  return 0
}

trap cleanup EXIT INT TERM

mkdir -p "${RUN_ROOT}"
: > "${RESULT_JSONL}"
printf '%s\n' 'tp,isl,osl,bs,completed,request_throughput_req_s,input_throughput_tok_s,output_throughput_tok_s,total_throughput_tok_s,mean_ttft_ms,median_ttft_ms,min_ttft_ms,max_ttft_ms,mean_tpot_ms,median_tpot_ms,min_tpot_ms,max_tpot_ms,main_prefill_batch_count,main_prefill_total_new_seq,main_prefill_new_seq_per_batch,main_prefill_new_token_per_batch' > "${SUMMARY_CSV}"

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

append_summary_csv_row() {
  local jsonl_path="$1"
  local csv_path="$2"
  local server_log_path="$3"
  local server_log_offset="$4"
  local tp="$5"
  local isl="$6"
  local osl="$7"
  local bs="$8"

  python -c '
import csv
import json
import pathlib
import re
import sys

jsonl_path, csv_path, server_log_path, server_log_offset, tp, isl, osl, bs = sys.argv[1:]
lines = pathlib.Path(jsonl_path).read_text(encoding="utf-8").splitlines()
if not lines:
    raise SystemExit("No benchmark results found in JSONL file.")
row = json.loads(lines[-1])

server_log = pathlib.Path(server_log_path)
offset = int(server_log_offset)
if server_log.exists():
    with server_log.open("rb") as f:
        f.seek(offset)
        segment = f.read().decode("utf-8", errors="replace")
else:
    segment = ""

segment_lines = segment.splitlines()
flush_idx = -1
for idx, line in enumerate(segment_lines):
    if "/flush_cache" in line and "200" in line:
        flush_idx = idx

main_lines = segment_lines[flush_idx + 1 :] if flush_idx >= 0 else segment_lines
prefill_pattern = re.compile(
    r"Prefill batch(?: \[\d+\])?,\s+#new-seq:\s*(?P<new_seq>\d+),\s+#new-token:\s*(?P<new_token>\d+)"
)

prefill_new_seq = []
prefill_new_token = []
for line in main_lines:
    match = prefill_pattern.search(line)
    if match:
        prefill_new_seq.append(int(match.group("new_seq")))
        prefill_new_token.append(int(match.group("new_token")))

main_prefill_batch_count = len(prefill_new_seq)
main_prefill_total_new_seq = sum(prefill_new_seq)
prefill_new_seq_str = "|".join(str(x) for x in prefill_new_seq)
prefill_new_token_str = "|".join(str(x) for x in prefill_new_token)

values = [
    tp,
    isl,
    osl,
    bs,
    row.get("completed", ""),
    row.get("request_throughput", ""),
    row.get("input_throughput", ""),
    row.get("output_throughput", ""),
    row.get("total_throughput", ""),
    row.get("mean_ttft_ms", ""),
    row.get("median_ttft_ms", ""),
    row.get("min_ttft_ms", ""),
    row.get("max_ttft_ms", ""),
    row.get("mean_tpot_ms", ""),
    row.get("median_tpot_ms", ""),
    row.get("min_tpot_ms", ""),
    row.get("max_tpot_ms", ""),
    main_prefill_batch_count,
    main_prefill_total_new_seq,
    prefill_new_seq_str,
    prefill_new_token_str,
]
with open(csv_path, "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(values)

def fmt_float(value):
    if value in (None, ""):
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)

request_throughput = fmt_float(row.get("request_throughput"))
input_throughput = fmt_float(row.get("input_throughput"))
output_throughput = fmt_float(row.get("output_throughput"))
total_throughput = fmt_float(row.get("total_throughput"))

print(
    "Summary: "
    f"req/s={request_throughput}, "
    f"input tok/s={input_throughput}, "
    f"output tok/s={output_throughput}, "
    f"total tok/s={total_throughput}, "
    f"prefill batches={main_prefill_batch_count}, "
    f"new_seq_per_batch={prefill_new_seq or []}"
)
' "${jsonl_path}" "${csv_path}" "${server_log_path}" "${server_log_offset}" "${tp}" "${isl}" "${osl}" "${bs}"
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
    --skip-server-warmup
    #--disable-cuda-graph
    --disable-piecewise-cuda-graph
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

  echo "Server launched, handing off readiness to bench_serving after ${SERVER_BOOTSTRAP_GRACE_SEC}s bootstrap grace."

  if ! ensure_server_bootstrapped "${SERVER_BOOTSTRAP_GRACE_SEC}" "${SERVER_PID}" "${SERVER_LOG}"; then
    echo "Server failed during bootstrap grace window. See ${SERVER_LOG}" >&2
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
        echo "Dataset: ${DATASET_NAME}"
        echo "Random range ratio: ${RANDOM_RANGE_RATIO}"
        echo "Client log: ${CLIENT_LOG}"
        echo "Profile root: ${PROFILE_ROOT}"
        echo "bench_serving is waiting for /v1/models with timeout ${READY_CHECK_TIMEOUT}s"
        echo "----------------------------------------"

        SERVER_LOG_OFFSET_BEFORE=0
        if [[ -f "${SERVER_LOG}" ]]; then
          SERVER_LOG_OFFSET_BEFORE=$(wc -c < "${SERVER_LOG}")
        fi

        CLIENT_ARGS=(
          -m sglang.bench_serving
          --backend sglang
          --host "${HOST}"
          --port "${PORT}"
          --model "${MODEL_PATH}"
          --dataset-name "${DATASET_NAME}"
          --num-prompts "${BS}"
          --random-input-len "${IL}"
          --random-output-len "${OL}"
          --random-range-ratio "${RANDOM_RANGE_RATIO}"
          --request-rate inf
          --max-concurrency "${BS}"
          --ready-check-timeout-sec "${READY_CHECK_TIMEOUT}"
          --warmup-requests "${WARMUP_REQUESTS}"
          --flush-cache
          --seed "${SEED}"
          --tag "tp=${TP},isl=${IL},osl=${OL},bs=${BS}"
          --output-file "${RESULT_JSONL}"
        )

        if [[ "${DISABLE_TQDM}" == "1" ]]; then
          CLIENT_ARGS+=(--disable-tqdm)
        fi

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
        sleep 1
        append_summary_csv_row "${RESULT_JSONL}" "${SUMMARY_CSV}" "${SERVER_LOG}" "${SERVER_LOG_OFFSET_BEFORE}" "${TP}" "${IL}" "${OL}" "${BS}"
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
echo "Summary CSV: ${SUMMARY_CSV}"
