#!/usr/bin/env bash
# Run one J-lens command through the capacity-aware local GPU scheduler.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/three_method_local_paths.sh"

SCHEDULER_PYTHON="${JLENS_LOCAL_SCHEDULER_PYTHON:-/usr/bin/python3.12}"
SCHEDULER_MODULE="${SCRIPT_DIR}/../../src/jlens_workspace/local_scheduler.py"
if [[ ! -x "${SCHEDULER_PYTHON}" ]]; then
  echo "error: local scheduler Python is unavailable: ${SCHEDULER_PYTHON}" >&2
  exit 2
fi

if [[ "${1:-}" == "--describe-policy" ]]; then
  exec "${SCHEDULER_PYTHON}" "${SCHEDULER_MODULE}" describe "${2:-}"
fi
if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 COMMAND TASK [ARG ...]" >&2
  exit 2
fi

TASK_NAME="$2"
exec "${SCHEDULER_PYTHON}" "${SCHEDULER_MODULE}" run \
  --kind gpu --task "${TASK_NAME}" -- "$@"
