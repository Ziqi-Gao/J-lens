#!/usr/bin/env bash
# Detach the local top-level DAG into an isolated GNU Screen session.
set -euo pipefail

MODE="${1:-}"
case "${MODE}" in
  bootstrap|smoke|full|all) ;;
  *) echo "usage: $0 {bootstrap|smoke|full|all}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/three_method_local_paths.sh"
if ! command -v screen >/dev/null; then
  echo "error: GNU Screen is required for detached local execution" >&2
  exit 2
fi
if [[ -n "$(git -C "${CODE_ROOT}" status --short)" ]]; then
  echo "error: Screen launch requires a clean immutable J-lens checkout" >&2
  exit 2
fi

mkdir -p "${JLENS_LOCAL_LOG_ROOT}/screen"
SESSION_NAME="jlens-three-method-${MODE}"
SCREEN_LOG="${JLENS_LOCAL_LOG_ROOT}/screen/${SESSION_NAME}.log"
if screen -list | rg -q "[.]${SESSION_NAME}[[:space:]]"; then
  echo "error: Screen session already exists: ${SESSION_NAME}" >&2
  exit 3
fi

screen -DmS "${SESSION_NAME}" -L -Logfile "${SCREEN_LOG}" \
  "${SCRIPT_DIR}/run_three_method_local.sh" "${MODE}"
printf 'screen_session=%s\nscreen_log=%s\n' "${SESSION_NAME}" "${SCREEN_LOG}"
