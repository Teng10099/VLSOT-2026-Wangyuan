#!/usr/bin/env bash
# Portable wrapper for the legacy diagnostic only; not the v2 candidate pipeline.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if (( $# == 0 )); then
    echo "Legacy only: supply the full tools/smoke_infer.py arguments. See README.md for current v2 inference." >&2
    exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$PYTHON_BIN" -B "$PROJECT_DIR/tools/smoke_infer.py" "$@"
