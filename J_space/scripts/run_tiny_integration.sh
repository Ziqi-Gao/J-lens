#!/usr/bin/env bash
# Fit a real tiny Jacobian lens and run its spectral analysis.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
OUTPUT_FILE="${OUTPUT_FILE:-${REPO_ROOT}/artifacts/smoke/tiny_gpt2_lens.pt}"
DEVICE="${DEVICE:-cpu}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "error: Python environment not found; run: uv sync --extra dev --extra llm" >&2
  exit 2
fi

cd "${REPO_ROOT}"
exec "${PYTHON}" scripts/smoke_jlens.py \
  --output "${OUTPUT_FILE}" \
  --device "${DEVICE}" \
  "$@"
