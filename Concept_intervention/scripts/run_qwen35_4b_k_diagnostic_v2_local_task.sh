#!/usr/bin/env bash
# Exact K-diagnostic v2 command bodies; execution is allowed only inside the local DAG.
set -euo pipefail

PROFILE="${1:-}"
ACTION="${2:-}"
STAGE="${3:-}"
INSTANCE="${4:-singleton}"
if [[ "${JLENS_LOCAL_KDIAG_DAG_ACTIVE:-0}" != "1" ]]; then
  echo "error: K-diagnostic tasks must be launched by the top-level local DAG" >&2
  exit 2
fi

case "${STAGE}" in
  pilot) CONFIG_NAME="qwen35_4b_k_diagnostic_v2_pilot.yaml" ;;
  full) CONFIG_NAME="qwen35_4b_k_diagnostic_v2_full.yaml" ;;
  transformed) CONFIG_NAME="qwen35_4b_k_diagnostic_v2_transformed.yaml" ;;
  *) echo "error: unknown K-diagnostic stage: ${STAGE}" >&2; exit 2 ;;
esac
case "${ACTION}:${PROFILE}" in
  validate:kdiag-control|resource-preflight:kdiag-control|index:kdiag-control|report:kdiag-control) ;;
  prepare-targets:kdiag-targets|build-bases:kdiag-targets) ;;
  rotation-cache-preflight:kdiag-rotations|prepare-rotations:kdiag-rotations) ;;
  microbenchmark:kdiag-bundle|bundle:kdiag-bundle) ;;
  *) echo "error: action/profile mismatch: ${ACTION}/${PROFILE}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="/data/del6500/J-lens/runs/qwen35_4b_k_diagnostic_v2"
JLENS_LOCAL_RUNTIME_ROOT="/scr/del6500/J-lens/runtime/three-method"
export RUN_ROOT JLENS_LOCAL_RUNTIME_ROOT
source "${SCRIPT_DIR}/three_method_local_env.sh"
CONFIG="${CODE_ROOT}/Concept_intervention/configs/${CONFIG_NAME}"

case "${ACTION}" in
  validate) jlens k-diagnostic validate "${CONFIG}" ;;
  prepare-targets) jlens k-diagnostic prepare-targets "${CONFIG}" ;;
  build-bases) jlens k-diagnostic build-bases "${CONFIG}" ;;
  rotation-cache-preflight) jlens k-diagnostic rotation-cache-preflight "${CONFIG}" ;;
  prepare-rotations) jlens k-diagnostic prepare-rotations "${CONFIG}" ;;
  resource-preflight) jlens k-diagnostic resource-preflight "${CONFIG}" ;;
  index) jlens k-diagnostic index "${CONFIG}" ;;
  report) jlens k-diagnostic report "${CONFIG}" ;;
  microbenchmark)
    [[ "${INSTANCE}" =~ ^[0-9]+$ ]] || { echo "error: invalid bundle index" >&2; exit 2; }
    jlens k-diagnostic run "${CONFIG}" --bundle-index "${INSTANCE}" --microbenchmark
    ;;
  bundle)
    [[ "${INSTANCE}" =~ ^[0-9]+$ ]] || { echo "error: invalid bundle index" >&2; exit 2; }
    jlens k-diagnostic run "${CONFIG}" --bundle-index "${INSTANCE}"
    ;;
esac
