#!/usr/bin/env bash
# Execute and checkpoint one CPU or GPU task in the local DAG.
set -euo pipefail

KIND="${1:-}"
TASK_NAME="${2:-}"
TASK_INDEX="${3:-}"
if [[ "${KIND}" != "cpu" && "${KIND}" != "gpu" ]]; then
  echo "error: worker kind must be cpu or gpu" >&2
  exit 2
fi
if [[ "${JLENS_LOCAL_DAG_ACTIVE:-0}" != "1" ]]; then
  echo "error: local worker must be launched by run_three_method_local.sh" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/three_method_local_paths.sh"

COMMIT="$(git -C "${CODE_ROOT}" rev-parse HEAD)"
TASK_KEY="${TASK_NAME}${TASK_INDEX:+_${TASK_INDEX}}"
STATE_DIR="${JLENS_LOCAL_RUNTIME_ROOT}/state/${COMMIT}"
LOG_DIR="${JLENS_LOCAL_LOG_ROOT}/${COMMIT}"
MARKER="${STATE_DIR}/${TASK_KEY}.done"
LOG_PATH="${LOG_DIR}/${TASK_KEY}.log"
mkdir -p "${STATE_DIR}" "${LOG_DIR}"

if [[ -f "${MARKER}" ]]; then
  printf 'skip_complete task=%s marker=%s\n' "${TASK_KEY}" "${MARKER}"
  exit 0
fi
if [[ "${JLENS_LOCAL_DRY_RUN:-0}" == "1" ]]; then
  "${SCRIPT_DIR}/run_three_method_local_task.sh" --decode \
    "${TASK_NAME}" ${TASK_INDEX:+"${TASK_INDEX}"}
  exit 0
fi
if [[ -n "$(git -C "${CODE_ROOT}" status --short)" ]]; then
  echo "error: local scientific tasks require a clean J-lens checkout" >&2
  exit 2
fi

printf 'start task=%s kind=%s log=%s\n' "${TASK_KEY}" "${KIND}" "${LOG_PATH}"
set +e
if [[ "${KIND}" == "gpu" ]]; then
  "${SCRIPT_DIR}/three_method_local_gpu.sh" \
    "${SCRIPT_DIR}/run_three_method_local_task.sh" \
    "${TASK_NAME}" ${TASK_INDEX:+"${TASK_INDEX}"} >"${LOG_PATH}" 2>&1
else
  "${SCRIPT_DIR}/run_three_method_local_task.sh" \
    "${TASK_NAME}" ${TASK_INDEX:+"${TASK_INDEX}"} >"${LOG_PATH}" 2>&1
fi
STATUS="$?"
set -e
if [[ "${STATUS}" != "0" ]]; then
  echo "error: task ${TASK_KEY} failed with status ${STATUS}; log=${LOG_PATH}" >&2
  tail -n 40 "${LOG_PATH}" >&2 || true
  exit "${STATUS}"
fi

MARKER_TMP="${MARKER}.tmp.$$"
{
  printf 'complete=true\n'
  printf 'task=%s\n' "${TASK_NAME}"
  printf 'task_index=%s\n' "${TASK_INDEX}"
  printf 'kind=%s\n' "${KIND}"
  printf 'git_commit=%s\n' "${COMMIT}"
  printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'log=%s\n' "${LOG_PATH}"
} >"${MARKER_TMP}"
mv "${MARKER_TMP}" "${MARKER}"
printf 'complete task=%s marker=%s\n' "${TASK_KEY}" "${MARKER}"
