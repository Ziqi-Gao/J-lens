#!/usr/bin/env bash
# Fail-closed local launcher for the independent K diagnostic.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${CODE_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-/data/del6500/J-lens/runs/qwen35_4b_k_diagnostic_v1}"
PYTHON="${JLENS_PYTHON:-/scr/del6500/J-lens/envs/three-method/bin/python}"
WORKERS="${K_DIAGNOSTIC_WORKERS:-1}"
COMMAND="${1:-}"

case "${COMMAND}" in
  pilot|full|transformed|report|all) ;;
  -h|--help|"")
    echo "usage: $0 {pilot|full|transformed|report|all}"
    exit 0
    ;;
  *) echo "error: unknown command ${COMMAND}" >&2; exit 2 ;;
esac
if [[ ! -x "${PYTHON}" ]]; then
  echo "error: J-lens Python is not executable: ${PYTHON}" >&2
  exit 2
fi
if [[ ! "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: K_DIAGNOSTIC_WORKERS must be a positive integer" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42

run_stage() {
  local config_name="$1" action="$2" stage_directory="$3"
  local config="${CODE_ROOT}/Concept_intervention/configs/${config_name}"
  local grid="${RUN_ROOT}/artifacts/concept_intervention/qwen35_4b_k_diagnostic_v1/${stage_directory}/grid.json"
  cd "${RUN_ROOT}"
  "${PYTHON}" -m jlens_workspace.cli k-diagnostic validate "${config}"
  "${PYTHON}" -m jlens_workspace.cli k-diagnostic "${action}" "${config}"
  if [[ ! -f "${grid}" ]]; then
    echo "error: sealed grid was not created: ${grid}" >&2
    exit 2
  fi
  local count maximum
  count="$("${PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["count"]==len(p["shards"]); print(p["count"])' "${grid}")"
  if [[ ! "${count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: invalid grid count ${count}" >&2
    exit 2
  fi
  maximum=$((count - 1))
  seq 0 "${maximum}" | xargs -n 1 -P "${WORKERS}" \
    "${PYTHON}" -m jlens_workspace.cli k-diagnostic run "${config}" --grid-index
  "${PYTHON}" -m jlens_workspace.cli k-diagnostic index "${config}"
}

run_pilot() {
  run_stage qwen35_4b_k_diagnostic_v1_pilot.yaml prepare-targets pilot
}
run_full() {
  run_stage qwen35_4b_k_diagnostic_v1_full.yaml prepare-targets raw_metric_full
}
run_transformed() {
  run_stage qwen35_4b_k_diagnostic_v1_transformed.yaml build-bases transformed_metric
}
run_report() {
  cd "${RUN_ROOT}"
  "${PYTHON}" -m jlens_workspace.cli k-diagnostic report \
    "${CODE_ROOT}/Concept_intervention/configs/qwen35_4b_k_diagnostic_v1_transformed.yaml"
}

case "${COMMAND}" in
  pilot) run_pilot ;;
  full) run_full ;;
  transformed) run_transformed ;;
  report) run_report ;;
  all) run_pilot; run_full; run_transformed; run_report ;;
esac
