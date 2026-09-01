#!/usr/bin/env bash
# ServerScheduler-only transport for the frozen K-diagnostic v2 task catalog.
set -euo pipefail

if [[ "${JLENS_SERVER_SCHEDULER_TASK_ACTIVE:-0}" != "1" ]]; then
  echo "error: task must be launched by the ServerScheduler adapter" >&2
  exit 2
fi

PROFILE="${1:-}"
ACTION="${2:-}"
STAGE="${3:-}"
INSTANCE="${4:-singleton}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
PYTHON="/scr/del6500/J-lens/envs/three-method/bin/python"
RUN_ROOT="/data/del6500/J-lens/runs/qwen35_4b_k_diagnostic_v2"

if [[ ! -x "${PYTHON}" ]]; then
  echo "error: registered J-lens Python is unavailable: ${PYTHON}" >&2
  exit 2
fi

case "${ACTION}" in
  validate)
    if [[ "${PROFILE}" != "kdiag-control" || "${INSTANCE}" != "singleton" ]]; then
      echo "error: invalid ServerScheduler K-diagnostic validation identity" >&2
      exit 2
    fi
    case "${STAGE}" in
      pilot|full|transformed) ;;
      *) echo "error: invalid ServerScheduler K-diagnostic stage" >&2; exit 2 ;;
    esac
    CONFIG="${CODE_ROOT}/Concept_intervention/configs/qwen35_4b_k_diagnostic_v2_${STAGE}.yaml"
    exec "${PYTHON}" -m jlens_workspace.cli k-diagnostic validate "${CONFIG}"
    ;;
  report) ;;
  *)
    # The minimal pilot registration never exposes these sealed catalog tasks.
    export JLENS_LOCAL_KDIAG_DAG_ACTIVE=1
    exec /usr/bin/bash "${SCRIPT_DIR}/run_qwen35_4b_k_diagnostic_v2_local_task.sh" "$@"
    ;;
esac

if [[ "${PROFILE}" != "kdiag-control" || "${STAGE}" != "transformed" || \
      "${INSTANCE}" != "singleton" ]]; then
  echo "error: invalid ServerScheduler K-diagnostic report identity" >&2
  exit 2
fi

CONFIG="${CODE_ROOT}/Concept_intervention/configs/qwen35_4b_k_diagnostic_v2_transformed.yaml"
exec "${PYTHON}" -m jlens_workspace.cli k-diagnostic report "${CONFIG}" \
  --output "${RUN_ROOT}/derivations/revision-r1/report"
