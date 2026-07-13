#!/usr/bin/env bash
# Run Qwen J-space spectral analysis in an existing GPU allocation.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CLI="${JLENS_WORKSPACE_CLI:-${REPO_ROOT}/.venv/bin/jlens-workspace}"
CONFIG="${1:-${JSPACE_CONFIG:-${REPO_ROOT}/J_space/configs/qwen35_4b_centered.yaml}}"

if [[ ! -x "${CLI}" ]]; then
  echo "error: jlens-workspace CLI is not executable at ${CLI}" >&2
  echo "run: uv sync --extra dev --extra llm" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "error: matrix config does not exist: ${CONFIG}" >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"

cd "${REPO_ROOT}"
"${CLI}" doctor --require-llm
"${REPO_ROOT}/.venv/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; x=torch.ones((16,16), device="cuda", dtype=torch.bfloat16); assert (x@x).sum().item() > 0; print(f"cuda={torch.cuda.get_device_name(0)} torch={torch.__version__} runtime={torch.version.cuda}")'
"${CLI}" config validate "${CONFIG}"
"${CLI}" lens fit "${CONFIG}"
exec "${CLI}" matrix run "${CONFIG}"
