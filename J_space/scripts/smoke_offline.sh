#!/usr/bin/env bash
# Validate matrix configs and local contracts without network access.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CLI="${JLENS_WORKSPACE_CLI:-${REPO_ROOT}/.venv/bin/jlens-workspace}"
PYTEST="${REPO_ROOT}/.venv/bin/pytest"

if [[ ! -x "${CLI}" ]]; then
  echo "error: jlens-workspace CLI is not executable at ${CLI}" >&2
  echo "run: uv sync --extra dev" >&2
  exit 2
fi
if [[ ! -x "${PYTEST}" ]]; then
  echo "error: pytest is not executable at ${PYTEST}; run: uv sync --extra dev" >&2
  exit 2
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "${REPO_ROOT}"
"${CLI}" config validate J_space/configs/qwen35_4b.yaml
"${CLI}" config validate J_space/configs/qwen35_4b_centered.yaml
"${CLI}" config validate J_space/configs/qwen35_4b_row_normalized.yaml
"${PYTEST}" -q tests/test_config.py tests/test_matrix.py tests/test_matrix_workflow.py
