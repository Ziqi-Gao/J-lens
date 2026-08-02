#!/usr/bin/env bash
# Online bootstrap for the local layout; scientific execution is offline afterward.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/three_method_local_paths.sh"

if [[ ! -x "${JLENS_PYTHON}" ]]; then
  echo "error: run ${SCRIPT_DIR}/setup_three_method_local.sh first" >&2
  exit 2
fi
if [[ -n "$(git -C "${CODE_ROOT}" status --short)" ]]; then
  echo "error: bootstrap requires a clean immutable J-lens checkout" >&2
  exit 2
fi

mkdir -p \
  "${RUN_ROOT}" "${JLENS_LOCAL_RUNTIME_ROOT}" "${JLENS_LOCAL_LOG_ROOT}" \
  "${JLENS_LOCAL_TMP_ROOT}" "${HF_HOME}" "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}"

JLENS_GIT_COMMIT="$(git -C "${CODE_ROOT}" rev-parse HEAD)"
export JLENS_GIT_COMMIT
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
exec "${SCRIPT_DIR}/bootstrap_three_method_server.sh"
