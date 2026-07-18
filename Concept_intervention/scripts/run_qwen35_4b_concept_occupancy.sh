#!/usr/bin/env bash
# Concept-vector J-space occupancy pilot inside an existing GPU allocation.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CLI="${JLENS_WORKSPACE_CLI:-${REPO_ROOT}/.venv/bin/jlens-workspace}"
CONFIG="${1:-${OCCUPANCY_CONFIG:-${REPO_ROOT}/Concept_intervention/configs/qwen35_4b_concept_occupancy_pilot_v1.yaml}}"
PROBES="${OCCUPANCY_PROBES:-${REPO_ROOT}/artifacts/concept_intervention/qwen35_4b_go_emotions_7concept_full_ovr_v2/probes}"

if [[ ! -x "${CLI}" ]]; then
  echo "error: jlens-workspace CLI is not executable at ${CLI}" >&2
  echo "run: uv sync --extra dev --extra llm" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "error: occupancy config does not exist: ${CONFIG}" >&2
  exit 2
fi
if [[ ! -d "${PROBES}" ]]; then
  echo "error: probe artifact directory does not exist: ${PROBES}" >&2
  exit 2
fi

# This branch's src may differ from the venv's editable install target;
# resolve the package from this checkout explicitly and record it.
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"

cd "${REPO_ROOT}"
"${CLI}" doctor --require-llm
"${REPO_ROOT}/.venv/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; x=torch.ones((16,16), device="cuda", dtype=torch.float64); assert (x@x).sum().item() > 0; print(f"cuda={torch.cuda.get_device_name(0)} torch={torch.__version__} runtime={torch.version.cuda}")'
"${CLI}" config validate "${CONFIG}"
exec "${CLI}" occupancy concepts "${CONFIG}" --probes "${PROBES}"
