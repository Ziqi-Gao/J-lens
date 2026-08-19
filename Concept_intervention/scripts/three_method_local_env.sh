#!/usr/bin/env bash
# Activate the immutable three-method environment on the local non-Slurm server.
set -euo pipefail

if [[ "${THREE_METHOD_LOCAL_ENV_LOADED:-0}" == "1" ]]; then
  return 0 2>/dev/null || exit 0
fi
THREE_METHOD_LOCAL_ENV_LOADED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/three_method_local_paths.sh"

if [[ ! -d "${CODE_ROOT}/.git" ]]; then
  echo "error: J-lens Git checkout is missing: ${CODE_ROOT}" >&2
  return 2 2>/dev/null || exit 2
fi
if [[ ! -x "${JLENS_PYTHON}" ]]; then
  echo "error: local J-lens environment is absent: ${JLENS_PYTHON}" >&2
  echo "run ${SCRIPT_DIR}/setup_three_method_local.sh first" >&2
  return 2 2>/dev/null || exit 2
fi

mkdir -p \
  "${RUN_ROOT}" "${JLENS_LOCAL_RUNTIME_ROOT}" "${JLENS_LOCAL_LOG_ROOT}" \
  "${JLENS_LOCAL_TMP_ROOT}" "${HF_HOME}" "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" \
  "${TORCH_HOME}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" \
  "${CUDA_CACHE_PATH}"

JLENS_GIT_COMMIT="${JLENS_GIT_COMMIT:-$(git -C "${CODE_ROOT}" rev-parse HEAD)}"
export JLENS_GIT_COMMIT
source "${SCRIPT_DIR}/three_method_env.sh"
