#!/usr/bin/env bash
# Top-level fail-closed FSM DAG for the registered K-diagnostic v2 experiment.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_qwen35_4b_k_diagnostic_v2.sh COMMAND

Commands:
  plan            Print the scheduler-backed DAG without executing it.
  status          Report commit-scoped local markers and scientific gates.
  microbenchmark  Prepare pilot inputs/cache, run only its worst bundle, then preflight.
  pilot           Complete and index the pilot after its approved microbenchmark.
  full            Complete and index raw_metric_full after the pilot index.
  transformed     Complete and index transformed_metric after the raw index.
  report          Generate the report after all three stage indexes are complete.

This entrypoint is for the FSM local scheduler. It never invokes sbatch and
never launches a CUDA process outside a J-lens CPU/GPU lease.
EOF
}

COMMAND="${1:-}"
case "${COMMAND}" in
  plan|status|microbenchmark|pilot|full|transformed|report) ;;
  -h|--help|"") usage; exit 0 ;;
  *) echo "error: unknown command: ${COMMAND}" >&2; usage >&2; exit 2 ;;
esac

print_plan() {
  cat <<'EOF'
microbenchmark
  -> CPU lease: validate pilot
  -> kdiag-targets exclusive GPU lease: prepare-targets pilot
  -> kdiag-rotations CPU lease: rotation-cache-preflight + prepare-rotations
  -> kdiag-bundle shared GPU lease: worst pilot bundle only
  -> CPU lease: resource-preflight pilot

pilot/full/transformed
  -> current-stage preparation and approved worst-bundle microbenchmark
  -> bounded waves; xargs launches scheduler workers, never Python directly
  -> CPU resource-preflight after each wave
  -> CPU stage index

report
  -> require all three complete stage indexes
  -> CPU lease: registered report
EOF
}
if [[ "${COMMAND}" == "plan" ]]; then
  print_plan
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="/data/del6500/J-lens/runs/qwen35_4b_k_diagnostic_v2"
JLENS_LOCAL_RUNTIME_ROOT="/scr/del6500/J-lens/runtime/three-method"
export RUN_ROOT JLENS_LOCAL_RUNTIME_ROOT
source "${SCRIPT_DIR}/three_method_local_paths.sh"

COMMIT="$(git -C "${CODE_ROOT}" rev-parse HEAD)"
KDIAG_RUNTIME_ROOT="${JLENS_LOCAL_RUNTIME_ROOT}/k-diagnostic-v2"
STATE_DIR="${KDIAG_RUNTIME_ROOT}/state/${COMMIT}"
LOG_DIR="${JLENS_LOCAL_LOG_ROOT}/k-diagnostic-v2/${COMMIT}"
ARTIFACT_ROOT="${RUN_ROOT}/artifacts/concept_intervention/qwen35_4b_k_diagnostic_v2"
WORKER="${SCRIPT_DIR}/run_qwen35_4b_k_diagnostic_v2_local_worker.sh"

stage_directory() {
  case "$1" in
    pilot) printf 'pilot\n' ;;
    full) printf 'raw_metric_full\n' ;;
    transformed) printf 'transformed_metric\n' ;;
    *) echo "error: unknown K-diagnostic stage: $1" >&2; return 2 ;;
  esac
}

index_path() {
  printf '%s/%s/index.json\n' "${ARTIFACT_ROOT}" "$(stage_directory "$1")"
}

if [[ "${COMMAND}" == "status" ]]; then
  printf 'commit=%s\nrun_root=%s\nscheduler_root=%s\n' \
    "${COMMIT}" "${RUN_ROOT}" "${JLENS_LOCAL_RUNTIME_ROOT}/scheduler"
  if [[ -d "${STATE_DIR}" ]]; then
    printf 'completed_local_tasks=%s\n' \
      "$(find "${STATE_DIR}" -maxdepth 1 -type f -name '*.done' | wc -l)"
  else
    printf 'completed_local_tasks=0\n'
  fi
  for stage in pilot full transformed; do
    path="$(index_path "${stage}")"
    printf '%s_index=%s\n' "${stage}" "${path}"
    [[ -f "${path}" ]] && rg -n '"complete"' "${path}" || true
  done
  exit 0
fi

if [[ -n "$(git -C "${CODE_ROOT}" status --short)" ]]; then
  echo "error: local K-diagnostic DAG requires a clean immutable checkout" >&2
  exit 2
fi
if [[ ! -x "${JLENS_PYTHON}" ]]; then
  echo "error: registered J-lens Python is unavailable: ${JLENS_PYTHON}" >&2
  exit 2
fi
mkdir -p "${KDIAG_RUNTIME_ROOT}" "${STATE_DIR}" "${LOG_DIR}"
exec 8>"${KDIAG_RUNTIME_ROOT}/dag.lock"
if ! flock -n 8; then
  echo "error: another K-diagnostic local DAG controller is active" >&2
  exit 3
fi

JLENS_LOCAL_KDIAG_DAG_ACTIVE=1
JLENS_LOCAL_KDIAG_COMMIT="${COMMIT}"
export JLENS_LOCAL_KDIAG_DAG_ACTIVE JLENS_LOCAL_KDIAG_COMMIT
WORKERS="${K_DIAGNOSTIC_WORKERS:-10}"
WAVE_SIZE="${K_DIAGNOSTIC_WAVE_SIZE:-20}"
for pair in "K_DIAGNOSTIC_WORKERS:${WORKERS}" "K_DIAGNOSTIC_WAVE_SIZE:${WAVE_SIZE}"; do
  if [[ ! "${pair#*:}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: ${pair%%:*} must be a positive integer" >&2
    exit 2
  fi
done

run_one() {
  local kind="$1" profile="$2" action="$3" stage="$4" instance="${5:-singleton}"
  "${WORKER}" "${kind}" "${profile}" "${action}" "${stage}" "${instance}"
}

prepare_stage() {
  local stage="$1" prepare_action="prepare-targets"
  [[ "${stage}" == "transformed" ]] && prepare_action="build-bases"
  run_one cpu kdiag-control validate "${stage}"
  run_one gpu kdiag-targets "${prepare_action}" "${stage}"
  run_one cpu kdiag-rotations rotation-cache-preflight "${stage}"
  run_one cpu kdiag-rotations prepare-rotations "${stage}"
}

bundle_plan() {
  local stage="$1" directory bundles
  directory="$(stage_directory "${stage}")"
  bundles="${ARTIFACT_ROOT}/${directory}/bundles.json"
  "${JLENS_PYTHON}" -c \
    'import json,sys; rows=json.load(open(sys.argv[1]))["bundles"]; print(len(rows), max(range(len(rows)), key=lambda i:int(rows[i]["conservative_cost_units"])))' \
    "${bundles}"
}

run_microbenchmark() {
  local stage="$1" count worst
  prepare_stage "${stage}"
  read -r count worst < <(bundle_plan "${stage}")
  run_one gpu kdiag-bundle microbenchmark "${stage}" "${worst}"
  run_one cpu kdiag-control resource-preflight "${stage}" post_microbenchmark
  printf 'stage=%s physical_bundles=%s worst_bundle_index=%s\n' \
    "${stage}" "${count}" "${worst}"
}

require_complete_index() {
  local stage="$1" path
  path="$(index_path "${stage}")"
  "${JLENS_PYTHON}" -c \
    'import json,sys; payload=json.load(open(sys.argv[1])); assert payload.get("complete") is True, payload' \
    "${path}"
}

run_remaining_bundles() {
  local stage="$1" count worst offset end
  local -a pending wave
  read -r count worst < <(bundle_plan "${stage}")
  mapfile -t pending < <(seq 0 $((count - 1)) | awk -v skip="${worst}" '$1 != skip')
  for ((offset = 0; offset < ${#pending[@]}; offset += WAVE_SIZE)); do
    end=$((offset + WAVE_SIZE))
    (( end > ${#pending[@]} )) && end=${#pending[@]}
    wave=("${pending[@]:offset:end-offset}")
    printf '%s\n' "${wave[@]}" | xargs -r -n 1 -P "${WORKERS}" \
      "${WORKER}" gpu kdiag-bundle bundle "${stage}"
    run_one cpu kdiag-control resource-preflight "${stage}" "wave_${offset}_${end}"
  done
  run_one cpu kdiag-control index "${stage}"
}

case "${COMMAND}" in
  microbenchmark)
    run_microbenchmark pilot
    ;;
  pilot)
    run_microbenchmark pilot
    run_remaining_bundles pilot
    ;;
  full)
    require_complete_index pilot
    run_microbenchmark full
    run_remaining_bundles full
    ;;
  transformed)
    require_complete_index full
    run_microbenchmark transformed
    run_remaining_bundles transformed
    ;;
  report)
    require_complete_index pilot
    require_complete_index full
    require_complete_index transformed
    run_one cpu kdiag-control report transformed
    ;;
esac
