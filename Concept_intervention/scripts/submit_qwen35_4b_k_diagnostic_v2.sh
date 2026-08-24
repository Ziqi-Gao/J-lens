#!/usr/bin/env bash
# Prepare, benchmark-gate, chunk physical-bundle arrays, then index.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${CODE_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-/data/del6500/J-lens/runs/qwen35_4b_k_diagnostic_v2}"
PYTHON="${JLENS_PYTHON:-/scr/del6500/J-lens/envs/three-method/bin/python}"
CONCURRENCY="${K_DIAGNOSTIC_SLURM_LIMIT:-20}"
STAGE="${1:-}"
case "${STAGE}" in
  pilot)
    CONFIG_NAME=qwen35_4b_k_diagnostic_v2_pilot.yaml
    ACTION=prepare-targets
    STAGE_DIRECTORY=pilot
    ;;
  full)
    CONFIG_NAME=qwen35_4b_k_diagnostic_v2_full.yaml
    ACTION=prepare-targets
    STAGE_DIRECTORY=raw_metric_full
    ;;
  transformed)
    CONFIG_NAME=qwen35_4b_k_diagnostic_v2_transformed.yaml
    ACTION=build-bases
    STAGE_DIRECTORY=transformed_metric
    ;;
  -h|--help|"") echo "usage: $0 {pilot|full|transformed}"; exit 0 ;;
  *) echo "error: unknown stage ${STAGE}" >&2; exit 2 ;;
esac
[[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || { echo "error: concurrency must be positive" >&2; exit 2; }
CONFIG="${CODE_ROOT}/Concept_intervention/configs/${CONFIG_NAME}"
BUNDLES="${RUN_ROOT}/artifacts/concept_intervention/qwen35_4b_k_diagnostic_v2/${STAGE_DIRECTORY}/bundles.json"
STAGE_ROOT="${RUN_ROOT}/artifacts/concept_intervention/qwen35_4b_k_diagnostic_v2/${STAGE_DIRECTORY}"
BENCHMARK="${STAGE_ROOT}/microbenchmark.json"
mkdir -p "${RUN_ROOT}/slurm"
export CODE_ROOT RUN_ROOT PYTHONPATH="${CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${RUN_ROOT}"

sbatch --wait --export="ALL,K_DIAGNOSTIC_CONFIG=${CONFIG},K_DIAGNOSTIC_ACTION=${ACTION}" \
  "${SCRIPT_DIR}/run_qwen35_4b_k_diagnostic_v2_stage.slurm"
sbatch --wait --export="ALL,K_DIAGNOSTIC_CONFIG=${CONFIG}" \
  "${SCRIPT_DIR}/run_qwen35_4b_k_diagnostic_v2_rotations.slurm"
read -r COUNT WORST REGISTERED_CHUNK < <("${PYTHON}" -c \
  'import json,sys; from jlens_workspace.config import load_experiment_config; p=json.load(open(sys.argv[1])); b=p["bundles"]; c=load_experiment_config(sys.argv[2]).k_diagnostic; print(len(b), max(range(len(b)), key=lambda i:int(b[i]["conservative_cost_units"])), c.slurm_array_chunk_size)' \
  "${BUNDLES}" "${CONFIG}")
CHUNK="${K_DIAGNOSTIC_ARRAY_CHUNK:-${REGISTERED_CHUNK}}"
[[ "${CHUNK}" =~ ^[1-9][0-9]*$ ]] || { echo "error: array chunk must be positive" >&2; exit 2; }
if (( CHUNK > REGISTERED_CHUNK )); then
  echo "error: requested chunk ${CHUNK} exceeds registered maximum ${REGISTERED_CHUNK}" >&2
  exit 2
fi

if [[ ! -f "${BENCHMARK}" ]]; then
  sbatch --wait --array=0-0 \
    --export="ALL,K_DIAGNOSTIC_CONFIG=${CONFIG},K_DIAGNOSTIC_BUNDLE_OFFSET=${WORST},K_DIAGNOSTIC_MICROBENCHMARK=1" \
    "${SCRIPT_DIR}/run_qwen35_4b_k_diagnostic_v2_grid.slurm"
fi
"${PYTHON}" -m jlens_workspace.cli k-diagnostic resource-preflight "${CONFIG}"

GRID_JOBS=()
for ((OFFSET=0; OFFSET<COUNT; OFFSET+=CHUNK)); do
  REMAINING=$((COUNT - OFFSET))
  SIZE="${CHUNK}"
  if (( REMAINING < SIZE )); then SIZE="${REMAINING}"; fi
  LAST=$((SIZE - 1))
  ARRAY_SPEC="0-${LAST}"
  if (( WORST >= OFFSET && WORST <= OFFSET + LAST )); then
    LOCAL_WORST=$((WORST - OFFSET))
    if (( LAST == 0 )); then
      continue
    fi
    if (( LOCAL_WORST == 0 )); then
      ARRAY_SPEC="1-${LAST}"
    elif (( LOCAL_WORST == LAST )); then
      ARRAY_SPEC="0-$((LAST - 1))"
    else
      ARRAY_SPEC="0-$((LOCAL_WORST - 1)),$((LOCAL_WORST + 1))-${LAST}"
    fi
  fi
  if [[ -z "${ARRAY_SPEC}" ]]; then
    continue
  fi
  JOB_ID="$(sbatch --wait --parsable --array="${ARRAY_SPEC}%${CONCURRENCY}" \
    --export="ALL,K_DIAGNOSTIC_CONFIG=${CONFIG},K_DIAGNOSTIC_BUNDLE_OFFSET=${OFFSET},K_DIAGNOSTIC_MICROBENCHMARK=0" \
    "${SCRIPT_DIR}/run_qwen35_4b_k_diagnostic_v2_grid.slurm")"
  GRID_JOBS+=("${JOB_ID}")
  "${PYTHON}" -m jlens_workspace.cli k-diagnostic resource-preflight "${CONFIG}"
done
DEPENDENCY="$(IFS=:; echo "${GRID_JOBS[*]}")"
INDEX_JOB="$(sbatch --wait --parsable \
  --export="ALL,K_DIAGNOSTIC_CONFIG=${CONFIG}" \
  "${SCRIPT_DIR}/run_qwen35_4b_k_diagnostic_v2_index.slurm")"
printf 'stage=%s\nlogical_grid_is_not_arrayed=true\nphysical_bundles=%s\nchunk=%s\ngrid_jobs=%s\nindex_job=%s\n' \
  "${STAGE}" "${COUNT}" "${CHUNK}" "${DEPENDENCY}" "${INDEX_JOB}"
