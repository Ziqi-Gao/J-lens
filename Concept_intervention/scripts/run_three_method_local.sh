#!/usr/bin/env bash
# Top-level fail-closed DAG for the local non-Slurm server.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_three_method_local.sh [--dry-run] COMMAND

Commands:
  bootstrap  Download and verify pinned model/data/RAPTOR inputs.
  smoke      Run every producer plus the four registered smoke shards and gate.
  full       Run remaining grids, method indexes, and final comparison.
  all        Run bootstrap, smoke, and full in order.
  status     Report local task markers and the final completion marker.
  plan       Print the registered local DAG without executing it.

Run setup_three_method_local.sh once before bootstrap. Component task scripts
are internal and intentionally reject direct execution.
EOF
}

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
COMMAND="${1:-}"
case "${COMMAND}" in
  bootstrap|smoke|full|all|status|plan) ;;
  -h|--help|"") usage; exit 0 ;;
  *) echo "error: unknown command: ${COMMAND}" >&2; usage >&2; exit 2 ;;
esac

print_plan() {
  cat <<'EOF'
bootstrap (online, pinned inputs)
  -> preflight
  -> J-lens refit + shared residual capture
  -> shared RAPTOR probes / Top-6 layer selection
  -> bootstrap probes
  -> J occupancy + ITI head capture / direction fit
  -> J[6] + RAPTOR[8] + ITI[6,251] smoke
  -> smoke-check gate
  -> J[0-391 except 6]
  -> RAPTOR[0-62 except 8]
  -> ITI[0-3086 except 6,251]
  -> three method indexes
  -> intervention comparison index
EOF
}
if [[ "${COMMAND}" == "plan" ]]; then
  print_plan
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/three_method_local_paths.sh"
COMMIT="$(git -C "${CODE_ROOT}" rev-parse HEAD)"
STATE_DIR="${JLENS_LOCAL_RUNTIME_ROOT}/state/${COMMIT}"
FINAL_INDEX="${RUN_ROOT}/artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/intervention_comparison/index.json"

if [[ "${COMMAND}" == "status" ]]; then
  if [[ -d "${STATE_DIR}" ]]; then
    printf 'commit=%s\ncompleted_local_tasks=%s\n' \
      "${COMMIT}" "$(find "${STATE_DIR}" -maxdepth 1 -type f -name '*.done' | wc -l)"
  else
    printf 'commit=%s\ncompleted_local_tasks=0\n' "${COMMIT}"
  fi
  printf 'final_index=%s\n' "${FINAL_INDEX}"
  if [[ -f "${FINAL_INDEX}" ]]; then
    /usr/bin/python3.12 -c \
      'import json,sys; payload=json.load(open(sys.argv[1])); print(f"complete={payload.get(\"complete\") is True}")' \
      "${FINAL_INDEX}"
  else
    printf 'complete=false\n'
  fi
  exit 0
fi

if [[ "${DRY_RUN}" == "0" && -n "$(git -C "${CODE_ROOT}" status --short)" ]]; then
  echo "error: local DAG requires a clean immutable J-lens checkout" >&2
  exit 2
fi
mkdir -p "${JLENS_LOCAL_RUNTIME_ROOT}" "${JLENS_LOCAL_LOG_ROOT}"
exec 8>"${JLENS_LOCAL_RUNTIME_ROOT}/dag.lock"
if ! flock -n 8; then
  echo "error: another J-lens local DAG controller is active" >&2
  exit 3
fi

JLENS_LOCAL_DAG_ACTIVE=1
JLENS_LOCAL_DRY_RUN="${DRY_RUN}"
export JLENS_LOCAL_DAG_ACTIVE JLENS_LOCAL_DRY_RUN
WORKER="${SCRIPT_DIR}/three_method_local_worker.sh"
CPU_WORKERS="${JLENS_LOCAL_CPU_WORKERS:-4}"
GPU_WORKERS="${JLENS_LOCAL_GPU_WORKERS:-4}"
for pair in "JLENS_LOCAL_CPU_WORKERS:${CPU_WORKERS}" "JLENS_LOCAL_GPU_WORKERS:${GPU_WORKERS}"; do
  if [[ ! "${pair#*:}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: ${pair%%:*} must be a positive integer" >&2
    exit 2
  fi
done

run_one() {
  local kind="$1" task="$2" index="${3:-}"
  "${WORKER}" "${kind}" "${task}" ${index:+"${index}"}
}

run_range() {
  local kind="$1" task="$2" maximum="$3" workers="$4"
  shift 4
  local excluded=("$@")
  local index skip value
  for ((index = 0; index <= maximum; index++)); do
    skip=0
    for value in "${excluded[@]}"; do
      [[ "${index}" == "${value}" ]] && skip=1
    done
    [[ "${skip}" == "0" ]] && printf '%s\n' "${index}"
  done | xargs -r -n 1 -P "${workers}" "${WORKER}" "${kind}" "${task}"
}


run_bootstrap() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'task=bootstrap online=true\n'
    return
  fi
  "${SCRIPT_DIR}/bootstrap_three_method_local.sh"
}

run_smoke() {
  run_one cpu preflight
  printf '%s\n' lens capture |
    xargs -r -n 1 -P "${GPU_WORKERS}" "${WORKER}" gpu
  run_one cpu layer-selection
  run_range cpu bootstrap-probes 6 "${CPU_WORKERS}"
  run_one cpu bootstrap-index
  run_one gpu iti-capture
  run_one cpu iti-fit
  run_range gpu occupancy 34 "${GPU_WORKERS}"
  run_one cpu occupancy-index
  printf '%s\n' \
    'j-grid 6' \
    'raptor-grid 8' \
    'iti-grid 6' \
    'iti-grid 251' |
    xargs -r -n 2 -P "${GPU_WORKERS}" "${WORKER}" gpu
  run_one cpu smoke-check
}

run_full() {
  local smoke_marker="${STATE_DIR}/smoke-check.done"
  if [[ "${DRY_RUN}" == "0" && ! -f "${smoke_marker}" ]]; then
    echo "error: smoke gate marker is absent: ${smoke_marker}" >&2
    exit 2
  fi
  run_range gpu j-grid 391 "${GPU_WORKERS}" 6
  run_range gpu raptor-grid 62 "${GPU_WORKERS}" 8
  run_range gpu iti-grid 3086 "${GPU_WORKERS}" 6 251
  run_range cpu method-index 2 "${CPU_WORKERS}"
  run_one cpu comparison-index
}

case "${COMMAND}" in
  bootstrap) run_bootstrap ;;
  smoke) run_smoke ;;
  full) run_full ;;
  all) run_bootstrap; run_smoke; run_full ;;
esac

if [[ "${DRY_RUN}" == "0" && "${COMMAND}" =~ ^(full|all)$ ]]; then
  /usr/bin/python3.12 -c \
    'import json,sys; payload=json.load(open(sys.argv[1])); assert payload.get("complete") is True, payload; print("three_method_complete=true")' \
    "${FINAL_INDEX}"
fi
