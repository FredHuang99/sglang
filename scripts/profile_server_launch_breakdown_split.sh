#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PARALLEL_DEGREES="${PARALLEL_DEGREES:-8,4,2,1}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
HOST="${HOST:-127.0.0.1}"
KEEP_ARTIFACTS=1
WORKER_MODE=0
RUN_ROOT=""
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/profile_server_launch_breakdown_split.sh [options]

Options:
  --parallel-degrees <list>   Comma-separated GPU counts. Default: 8,4,2,1
  --wait-timeout <seconds>    Per-case readiness timeout. Default: 1800
  --host <host>               Bind host for launched servers. Default: 127.0.0.1
  --keep-artifacts            Keep per-case logs and launch artifacts. Default: enabled
  --no-keep-artifacts         Remove per-case artifacts after aggregation

This script self-backgrounds with nohup. It first profiles diffusion presets in the
sglang-diffusion conda env, then switches to the sglang env and profiles the two
PromptEnhancer presets with llm setups: default and cp512_req1_cg1.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --worker)
      WORKER_MODE=1
      shift
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --parallel-degrees)
      PARALLEL_DEGREES="$2"
      shift 2
      ;;
    --wait-timeout)
      WAIT_TIMEOUT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --keep-artifacts)
      KEEP_ARTIFACTS=1
      shift
      ;;
    --no-keep-artifacts)
      KEEP_ARTIFACTS=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${WORKER_MODE}" -ne 1 ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
  RUN_ROOT="${RUN_ROOT:-/workspace/outputs/server_launch_breakdown_split/${RUN_ID}}"
  mkdir -p "${RUN_ROOT}"
  DRIVER_LOG="${RUN_ROOT}/driver.log"

  KEEP_FLAG="--keep-artifacts"
  if [[ "${KEEP_ARTIFACTS}" -eq 0 ]]; then
    KEEP_FLAG="--no-keep-artifacts"
  fi

  nohup bash "$0" \
    --worker \
    --run-root "${RUN_ROOT}" \
    --parallel-degrees "${PARALLEL_DEGREES}" \
    --wait-timeout "${WAIT_TIMEOUT}" \
    --host "${HOST}" \
    "${KEEP_FLAG}" \
    > "${DRIVER_LOG}" 2>&1 &
  PID=$!

  echo "Background launch-breakdown run started."
  echo "PID: ${PID}"
  echo "Run root: ${RUN_ROOT}"
  echo "Driver log: ${DRIVER_LOG}"
  echo "Tail with: tail -f ${DRIVER_LOG}"
  exit 0
fi

if [[ -z "${RUN_ROOT}" ]]; then
  echo "--run-root is required in worker mode." >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}"
DIFFUSION_OUT="${RUN_ROOT}/diffusion"
LLM_OUT="${RUN_ROOT}/llm"
MANIFEST_PATH="${RUN_ROOT}/run_manifest.json"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*"
}

write_manifest() {
  local diffusion_status="$1"
  local llm_status="$2"
  local diffusion_exit_code="$3"
  local llm_exit_code="$4"
  local overall_exit_code="$5"

  python3 - <<PY
import json
from pathlib import Path

payload = {
    "generated_at": "$(date '+%Y-%m-%dT%H:%M:%S')",
    "run_root": "${RUN_ROOT}",
    "parallel_degrees": "${PARALLEL_DEGREES}",
    "wait_timeout": ${WAIT_TIMEOUT},
    "host": "${HOST}",
    "keep_artifacts": ${KEEP_ARTIFACTS},
    "phase_order": ["diffusion", "llm"],
    "llm_setup_note": "The requested max_batch_size=1 variant is implemented with --max-running-requests 1 because the LLM server CLI does not expose a top-level --max-batch-size flag.",
    "diffusion": {
        "status": "${diffusion_status}",
        "exit_code": ${diffusion_exit_code},
        "output_dir": "${DIFFUSION_OUT}",
        "models": ["wan2.2-ti2v-5b", "wan2.1-t2v-1.3b", "z-image"],
    },
    "llm": {
        "status": "${llm_status}",
        "exit_code": ${llm_exit_code},
        "output_dir": "${LLM_OUT}",
        "models": ["promptenhancer-7b", "promptenhancer-32b"],
        "llm_setups": ["default", "cp512_req1_cg1"],
    },
    "overall_exit_code": ${overall_exit_code},
}

Path("${MANIFEST_PATH}").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
PY
}

if [[ ! -f "${CONDA_SH}" ]]; then
  log "conda init script not found: ${CONDA_SH}"
  write_manifest "failed_to_activate_env" "not_started" 97 98 1
  exit 1
fi

# shellcheck source=/dev/null
source "${CONDA_SH}"

KEEP_ARGS=()
if [[ "${KEEP_ARTIFACTS}" -eq 0 ]]; then
  KEEP_ARGS+=(--no-keep-artifacts)
else
  KEEP_ARGS+=(--keep-artifacts)
fi

DIFFUSION_STATUS="not_started"
LLM_STATUS="not_started"
DIFFUSION_EXIT_CODE=99
LLM_EXIT_CODE=99

log "Run root: ${RUN_ROOT}"
log "Starting diffusion phase in conda env sglang-diffusion"
if conda activate sglang-diffusion; then
  set +e
  python3 "${SCRIPT_DIR}/profile_server_launch_breakdown.py" \
    --models "wan2.2-ti2v-5b,wan2.1-t2v-1.3b,z-image" \
    --parallel-degrees "${PARALLEL_DEGREES}" \
    --wait-timeout "${WAIT_TIMEOUT}" \
    --host "${HOST}" \
    --output-dir "${DIFFUSION_OUT}" \
    "${KEEP_ARGS[@]}"
  DIFFUSION_EXIT_CODE=$?
  set -e
  if [[ "${DIFFUSION_EXIT_CODE}" -eq 0 ]]; then
    DIFFUSION_STATUS="completed"
  else
    DIFFUSION_STATUS="failed"
  fi
else
  DIFFUSION_EXIT_CODE=97
  DIFFUSION_STATUS="failed_to_activate_env"
fi
log "Diffusion phase finished with status=${DIFFUSION_STATUS} exit_code=${DIFFUSION_EXIT_CODE}"

log "Starting LLM phase in conda env sglang"
if conda activate sglang; then
  set +e
  python3 "${SCRIPT_DIR}/profile_server_launch_breakdown.py" \
    --models "promptenhancer-7b,promptenhancer-32b" \
    --parallel-degrees "${PARALLEL_DEGREES}" \
    --wait-timeout "${WAIT_TIMEOUT}" \
    --host "${HOST}" \
    --llm-setups "default,cp512_req1_cg1" \
    --output-dir "${LLM_OUT}" \
    "${KEEP_ARGS[@]}"
  LLM_EXIT_CODE=$?
  set -e
  if [[ "${LLM_EXIT_CODE}" -eq 0 ]]; then
    LLM_STATUS="completed"
  else
    LLM_STATUS="failed"
  fi
else
  LLM_EXIT_CODE=98
  LLM_STATUS="failed_to_activate_env"
fi
log "LLM phase finished with status=${LLM_STATUS} exit_code=${LLM_EXIT_CODE}"

OVERALL_EXIT_CODE=0
if [[ "${DIFFUSION_EXIT_CODE}" -ne 0 || "${LLM_EXIT_CODE}" -ne 0 ]]; then
  OVERALL_EXIT_CODE=1
fi

write_manifest \
  "${DIFFUSION_STATUS}" \
  "${LLM_STATUS}" \
  "${DIFFUSION_EXIT_CODE}" \
  "${LLM_EXIT_CODE}" \
  "${OVERALL_EXIT_CODE}"

log "Manifest written to ${MANIFEST_PATH}"
log "All phases finished with overall_exit_code=${OVERALL_EXIT_CODE}"
exit "${OVERALL_EXIT_CODE}"
