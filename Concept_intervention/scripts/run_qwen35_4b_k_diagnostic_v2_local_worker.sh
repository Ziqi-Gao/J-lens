#!/usr/bin/env bash
# Execute and checkpoint one K-diagnostic task through the shared local scheduler.
set -euo pipefail

KIND="${1:-}"
PROFILE="${2:-}"
ACTION="${3:-}"
STAGE="${4:-}"
INSTANCE="${5:-singleton}"
if [[ "${KIND}" != "cpu" && "${KIND}" != "gpu" ]]; then
  echo "error: K-diagnostic worker kind must be cpu or gpu" >&2
  exit 2
fi
if [[ "${JLENS_LOCAL_KDIAG_DAG_ACTIVE:-0}" != "1" ]]; then
  echo "error: K-diagnostic worker must be launched by the top-level local DAG" >&2
  exit 2
fi
if [[ ! "${INSTANCE}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: unsafe K-diagnostic task instance: ${INSTANCE}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="/data/del6500/J-lens/runs/qwen35_4b_k_diagnostic_v2"
JLENS_LOCAL_RUNTIME_ROOT="/scr/del6500/J-lens/runtime/three-method"
export RUN_ROOT JLENS_LOCAL_RUNTIME_ROOT
source "${SCRIPT_DIR}/three_method_local_paths.sh"

COMMIT="$(git -C "${CODE_ROOT}" rev-parse HEAD)"
if [[ "${COMMIT}" != "${JLENS_LOCAL_KDIAG_COMMIT:-}" ]]; then
  echo "error: K-diagnostic checkout commit changed during the DAG" >&2
  exit 2
fi
if [[ -n "$(git -C "${CODE_ROOT}" status --short)" ]]; then
  echo "error: local scientific tasks require a clean J-lens checkout" >&2
  exit 2
fi

TASK_KEY="${ACTION}_${STAGE}_${INSTANCE}"
STATE_DIR="${JLENS_LOCAL_RUNTIME_ROOT}/k-diagnostic-v2/state/${COMMIT}"
LOG_DIR="${JLENS_LOCAL_LOG_ROOT}/k-diagnostic-v2/${COMMIT}"
MARKER="${STATE_DIR}/${TASK_KEY}.done"
LOG_PATH="${LOG_DIR}/${TASK_KEY}.log"
mkdir -p "${STATE_DIR}" "${LOG_DIR}"
if [[ -f "${MARKER}" ]]; then
  printf 'skip_complete task=%s marker=%s\n' "${TASK_KEY}" "${MARKER}"
  exit 0
fi

case "${PROFILE}" in
  kdiag-rotations) JLENS_CPUS_PER_TASK=16 ;;
  *) JLENS_CPUS_PER_TASK=4 ;;
esac
export JLENS_CPUS_PER_TASK
TASK_RUNNER="${SCRIPT_DIR}/run_qwen35_4b_k_diagnostic_v2_local_task.sh"
printf 'start task=%s kind=%s profile=%s log=%s\n' \
  "${TASK_KEY}" "${KIND}" "${PROFILE}" "${LOG_PATH}"
set +e
if [[ "${KIND}" == "gpu" ]]; then
  "${SCRIPT_DIR}/three_method_local_gpu.sh" \
    "${TASK_RUNNER}" "${PROFILE}" "${ACTION}" "${STAGE}" "${INSTANCE}" \
    >"${LOG_PATH}" 2>&1
else
  "${SCRIPT_DIR}/three_method_local_resources.sh" cpu "${PROFILE}" \
    "${TASK_RUNNER}" "${PROFILE}" "${ACTION}" "${STAGE}" "${INSTANCE}" \
    >"${LOG_PATH}" 2>&1
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
  printf 'task=%s\n' "${TASK_KEY}"
  printf 'kind=%s\nprofile=%s\n' "${KIND}" "${PROFILE}"
  printf 'git_commit=%s\n' "${COMMIT}"
  printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'log=%s\n' "${LOG_PATH}"
} >"${MARKER_TMP}"
mv "${MARKER_TMP}" "${MARKER}"
printf 'complete task=%s marker=%s\n' "${TASK_KEY}" "${MARKER}"
