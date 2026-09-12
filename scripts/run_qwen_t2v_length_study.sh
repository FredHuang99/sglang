#!/usr/bin/env bash
# Only start the study. Environment setup and downloads remain manual.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$(command -v python)"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONOPTIMIZE SGLANG_MAX_NEW_TOKENS_LIMIT

# Explicit CLI arguments following this wrapper override the defaults below.
# Uses the activated environment. Pass --model-dir, --data-dir and --output-dir.
exec "$PYTHON_BIN" -u -B "$SCRIPT_DIR/qwen_t2v_length_study.py" run \
  --concurrency "${CONCURRENCY:-64}" \
  --mem-fraction "${MEM_FRACTION:-0.85}" \
  --base-port "${BASE_PORT:-31000}" \
  --startup-timeout "${STARTUP_TIMEOUT:-1800}" \
  "$@"
