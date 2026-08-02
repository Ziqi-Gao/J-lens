#!/usr/bin/env bash
# Create the J-lens-owned Python environment without writing to user/global paths.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/three_method_local_paths.sh"

BOOTSTRAP_PYTHON="${JLENS_BOOTSTRAP_PYTHON:-/usr/bin/python3.12}"
UV_TOOL_ENV="${JLENS_UV_TOOL_ENV:-${JLENS_LOCAL_SCRATCH_ROOT}/tools/uv}"
UV_BIN="${UV_TOOL_ENV}/bin/uv"

_jlens_local_require_under JLENS_UV_TOOL_ENV \
  "${UV_TOOL_ENV}" "${JLENS_ALLOWED_SCRATCH_ROOT}"
if [[ ! -x "${BOOTSTRAP_PYTHON}" ]]; then
  echo "error: Python 3.12 bootstrap interpreter is unavailable: ${BOOTSTRAP_PYTHON}" >&2
  exit 2
fi

mkdir -p \
  "${RUN_ROOT}" "${JLENS_LOCAL_DATA_ROOT}/datasets/raw" \
  "${JLENS_LOCAL_DATA_ROOT}/datasets/processed" \
  "${JLENS_LOCAL_DATA_ROOT}/models" "${JLENS_LOCAL_DATA_ROOT}/results" \
  "${JLENS_LOCAL_DATA_ROOT}/exports" "${JLENS_LOCAL_SCRATCH_ROOT}/envs" \
  "${JLENS_LOCAL_SCRATCH_ROOT}/tools" "${JLENS_LOCAL_RUNTIME_ROOT}" \
  "${JLENS_LOCAL_LOG_ROOT}" "${JLENS_LOCAL_TMP_ROOT}" "${UV_CACHE_DIR}" \
  "${PIP_CACHE_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" "${CUDA_CACHE_PATH}"

if [[ ! -x "${UV_BIN}" ]]; then
  "${BOOTSTRAP_PYTHON}" -m venv "${UV_TOOL_ENV}"
  "${UV_TOOL_ENV}/bin/python" -m pip install \
    --disable-pip-version-check "uv>=0.8,<1"
fi

export UV_PROJECT_ENVIRONMENT="${JLENS_ENV_ROOT}"
"${UV_BIN}" sync \
  --project "${CODE_ROOT}" \
  --frozen \
  --extra dev \
  --extra llm \
  --python "${BOOTSTRAP_PYTHON}"

if [[ ! -x "${JLENS_PYTHON}" ]]; then
  echo "error: uv completed without creating ${JLENS_PYTHON}" >&2
  exit 2
fi

MANIFEST="${JLENS_LOCAL_RUNTIME_ROOT}/environment.txt"
{
  printf 'code_root=%s\n' "${CODE_ROOT}"
  printf 'run_root=%s\n' "${RUN_ROOT}"
  printf 'scratch_root=%s\n' "${JLENS_LOCAL_SCRATCH_ROOT}"
  printf 'python=%s\n' "${JLENS_PYTHON}"
  printf 'python_version=%s\n' "$("${JLENS_PYTHON}" --version 2>&1)"
  printf 'uv=%s\n' "${UV_BIN}"
  printf 'uv_version=%s\n' "$("${UV_BIN}" --version)"
  printf 'lock_sha256=%s\n' "$(sha256sum "${CODE_ROOT}/uv.lock" | awk '{print $1}')"
} >"${MANIFEST}"

"${JLENS_PYTHON}" -c \
  'import jlens_workspace, numpy, sklearn, scipy; print("local_environment_ready=true")'
printf 'environment_manifest=%s\n' "${MANIFEST}"
