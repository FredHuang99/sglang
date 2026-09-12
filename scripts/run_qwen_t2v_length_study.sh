#!/usr/bin/env bash
# Only start the study. Environment setup and downloads remain manual.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONOPTIMIZE SGLANG_MAX_NEW_TOKENS_LIMIT

# Explicit CLI arguments following this wrapper override the defaults below.
# Paths may be given as flags or as STUDY_ROOT/MODEL_DIR/DATA_DIR/RESULT_DIR.
exec "$PYTHON_BIN" -u -B "$SCRIPT_DIR/qwen_t2v_length_study.py" run \
  --concurrency "${CONCURRENCY:-64}" \
  --mem-fraction "${MEM_FRACTION:-0.85}" \
  --base-port "${BASE_PORT:-31000}" \
  --startup-timeout "${STARTUP_TIMEOUT:-1800}" \
  "$@"
