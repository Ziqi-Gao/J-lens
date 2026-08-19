#!/usr/bin/env bash
# Run one CPU task through the same host CPU/RAM lease broker used by GPU work.
set -euo pipefail

KIND="${1:-}"
TASK_NAME="${2:-}"
shift 2 || true
if [[ "${KIND}" != "cpu" ]]; then
  echo "error: three_method_local_resources.sh accepts CPU tasks only" >&2
  exit 2
fi
if [[ -z "${TASK_NAME}" || "$#" -eq 0 ]]; then
  echo "usage: $0 cpu TASK COMMAND [ARG ...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/three_method_local_paths.sh"
SCHEDULER_PYTHON="${JLENS_LOCAL_SCHEDULER_PYTHON:-/usr/bin/python3.12}"
SCHEDULER_MODULE="${SCRIPT_DIR}/../../src/jlens_workspace/local_scheduler.py"
if [[ ! -x "${SCHEDULER_PYTHON}" ]]; then
  echo "error: local scheduler Python is unavailable: ${SCHEDULER_PYTHON}" >&2
  exit 2
fi
exec "${SCHEDULER_PYTHON}" "${SCHEDULER_MODULE}" run \
  --kind cpu --task "${TASK_NAME}" -- "$@"
