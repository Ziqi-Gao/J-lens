#!/usr/bin/env bash
# Real tiny-model integration. The checkpoint must be cached when offline flags are set.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts/smoke/concept_pipeline}"
DEVICE="${DEVICE:-cpu}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "error: Python environment not found; run: uv sync --extra dev --extra llm" >&2
  exit 2
fi

cd "${REPO_ROOT}"
exec "${PYTHON}" scripts/smoke_concept_pipeline.py \
  --output "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  "$@"
