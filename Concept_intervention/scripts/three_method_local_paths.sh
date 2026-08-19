#!/usr/bin/env bash
# J-lens-owned paths for the non-Slurm server. This file does not activate Python.
set -euo pipefail

if [[ "${THREE_METHOD_LOCAL_PATHS_LOADED:-0}" == "1" ]]; then
  return 0 2>/dev/null || exit 0
fi
THREE_METHOD_LOCAL_PATHS_LOADED=1

JLENS_ALLOWED_CODE_ROOT="/home/del6500/projects/J-lens"
JLENS_ALLOWED_DATA_ROOT="/data/del6500/J-lens"
JLENS_ALLOWED_SCRATCH_ROOT="/scr/del6500/J-lens"

CODE_ROOT="${CODE_ROOT:-${JLENS_ALLOWED_CODE_ROOT}}"
JLENS_LOCAL_DATA_ROOT="${JLENS_LOCAL_DATA_ROOT:-${JLENS_ALLOWED_DATA_ROOT}}"
JLENS_LOCAL_SCRATCH_ROOT="${JLENS_LOCAL_SCRATCH_ROOT:-${JLENS_ALLOWED_SCRATCH_ROOT}}"
RUN_ROOT="${RUN_ROOT:-${JLENS_LOCAL_DATA_ROOT}/runs/qwen35_4b_three_method_intervention_v1}"
JLENS_ENV_ROOT="${JLENS_ENV_ROOT:-${JLENS_LOCAL_SCRATCH_ROOT}/envs/three-method}"
JLENS_PYTHON="${JLENS_PYTHON:-${JLENS_ENV_ROOT}/bin/python}"

JLENS_CACHE_ROOT="${JLENS_CACHE_ROOT:-${JLENS_LOCAL_SCRATCH_ROOT}/cache}"
JLENS_LOCAL_RUNTIME_ROOT="${JLENS_LOCAL_RUNTIME_ROOT:-${JLENS_LOCAL_SCRATCH_ROOT}/runtime/three-method}"
JLENS_LOCAL_LOG_ROOT="${JLENS_LOCAL_LOG_ROOT:-${JLENS_LOCAL_SCRATCH_ROOT}/logs/three-method}"
JLENS_LOCAL_TMP_ROOT="${JLENS_LOCAL_TMP_ROOT:-${JLENS_LOCAL_SCRATCH_ROOT}/tmp/three-method}"

HF_HOME="${HF_HOME:-${JLENS_CACHE_ROOT}/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${JLENS_CACHE_ROOT}/uv}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${JLENS_CACHE_ROOT}/pip}"
TORCH_HOME="${TORCH_HOME:-${JLENS_CACHE_ROOT}/torch}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${JLENS_CACHE_ROOT}/triton}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${JLENS_CACHE_ROOT}/xdg}"
CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${JLENS_CACHE_ROOT}/cuda}"
TMPDIR="${TMPDIR:-${JLENS_LOCAL_TMP_ROOT}}"

_jlens_local_resolve() {
  realpath -m -- "$1"
}

_jlens_local_require_exact() {
  local label="$1"
  local candidate="$2"
  local expected="$3"
  if [[ "$(_jlens_local_resolve "${candidate}")" != "$(_jlens_local_resolve "${expected}")" ]]; then
    echo "error: ${label} must be ${expected}, got ${candidate}" >&2
    return 2
  fi
}

_jlens_local_require_under() {
  local label="$1"
  local candidate="$2"
  local root="$3"
  local resolved_candidate resolved_root
  resolved_candidate="$(_jlens_local_resolve "${candidate}")"
  resolved_root="$(_jlens_local_resolve "${root}")"
  case "${resolved_candidate}" in
    "${resolved_root}"|"${resolved_root}"/*) ;;
    *)
      echo "error: ${label} must stay under ${resolved_root}, got ${resolved_candidate}" >&2
      return 2
      ;;
  esac
}

_jlens_local_require_exact CODE_ROOT "${CODE_ROOT}" "${JLENS_ALLOWED_CODE_ROOT}"
_jlens_local_require_exact JLENS_LOCAL_DATA_ROOT \
  "${JLENS_LOCAL_DATA_ROOT}" "${JLENS_ALLOWED_DATA_ROOT}"
_jlens_local_require_exact JLENS_LOCAL_SCRATCH_ROOT \
  "${JLENS_LOCAL_SCRATCH_ROOT}" "${JLENS_ALLOWED_SCRATCH_ROOT}"
_jlens_local_require_under RUN_ROOT "${RUN_ROOT}" "${JLENS_ALLOWED_DATA_ROOT}"
for path_name in \
  JLENS_ENV_ROOT JLENS_CACHE_ROOT JLENS_LOCAL_RUNTIME_ROOT \
  JLENS_LOCAL_LOG_ROOT JLENS_LOCAL_TMP_ROOT HF_HOME HF_HUB_CACHE \
  HF_DATASETS_CACHE UV_CACHE_DIR PIP_CACHE_DIR TORCH_HOME \
  TRITON_CACHE_DIR XDG_CACHE_HOME CUDA_CACHE_PATH TMPDIR; do
  _jlens_local_require_under \
    "${path_name}" "${!path_name}" "${JLENS_ALLOWED_SCRATCH_ROOT}"
done

export CODE_ROOT RUN_ROOT JLENS_LOCAL_DATA_ROOT JLENS_LOCAL_SCRATCH_ROOT
export JLENS_ENV_ROOT JLENS_PYTHON JLENS_CACHE_ROOT
export JLENS_LOCAL_RUNTIME_ROOT JLENS_LOCAL_LOG_ROOT JLENS_LOCAL_TMP_ROOT
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE
export UV_CACHE_DIR PIP_CACHE_DIR TORCH_HOME TRITON_CACHE_DIR
export XDG_CACHE_HOME CUDA_CACHE_PATH TMPDIR
