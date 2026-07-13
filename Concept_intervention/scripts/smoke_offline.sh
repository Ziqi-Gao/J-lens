#!/usr/bin/env bash
# Validate the concept direction without a model download, network, or GPU.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CLI="${JLENS_WORKSPACE_CLI:-${REPO_ROOT}/.venv/bin/jlens-workspace}"
PYTEST="${REPO_ROOT}/.venv/bin/pytest"
CONFIG="${CONCEPT_CONFIG:-${REPO_ROOT}/Concept_intervention/configs/qwen35_4b.yaml}"

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
"${CLI}" config validate "${CONFIG}"
"${CLI}" data validate \
  "${REPO_ROOT}/Concept_intervention/data/builtin_abstract_concepts.jsonl" \
  --min-per-label-per-split 2
"${PYTEST}" -q \
  tests/test_config.py \
  tests/test_data.py \
  tests/test_concept_workflow.py \
  tests/test_alignment_workflow.py \
  tests/test_batched_alignment_workflow.py
