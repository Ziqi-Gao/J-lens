#!/usr/bin/env bash
# Prepare one stage synchronously, then submit its sealed grid and index.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${CODE_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-/data/del6500/J-lens/runs/qwen35_4b_k_diagnostic_v1}"
LIMIT="${K_DIAGNOSTIC_SLURM_LIMIT:-20}"
STAGE="${1:-}"
case "${STAGE}" in
  pilot)
    CONFIG_NAME=qwen35_4b_k_diagnostic_v1_pilot.yaml
    ACTION=prepare-targets
    STAGE_DIRECTORY=pilot
    ;;
  full)
    CONFIG_NAME=qwen35_4b_k_diagnostic_v1_full.yaml
    ACTION=prepare-targets
    STAGE_DIRECTORY=raw_metric_full
    ;;
  transformed)
    CONFIG_NAME=qwen35_4b_k_diagnostic_v1_transformed.yaml
    ACTION=build-bases
    STAGE_DIRECTORY=transformed_metric
    ;;
  -h|--help|"")
    echo "usage: $0 {pilot|full|transformed}"
    exit 0
    ;;
  *) echo "error: unknown stage ${STAGE}" >&2; exit 2 ;;
esac
if [[ ! "${LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: K_DIAGNOSTIC_SLURM_LIMIT must be positive" >&2
  exit 2
fi
CONFIG="${CODE_ROOT}/Concept_intervention/configs/${CONFIG_NAME}"
GRID="${RUN_ROOT}/artifacts/concept_intervention/qwen35_4b_k_diagnostic_v1/${STAGE_DIRECTORY}/grid.json"
PYTHON="${JLENS_PYTHON:-/scr/del6500/J-lens/envs/three-method/bin/python}"
mkdir -p "${RUN_ROOT}/slurm"
export CODE_ROOT RUN_ROOT
cd "${RUN_ROOT}"
sbatch --wait --export="ALL,K_DIAGNOSTIC_CONFIG=${CONFIG},K_DIAGNOSTIC_ACTION=${ACTION}" \
  "${SCRIPT_DIR}/run_qwen35_4b_k_diagnostic_v1_stage.slurm"
COUNT="$("${PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["count"]==len(p["shards"]); print(p["count"])' "${GRID}")"
if [[ ! "${COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: invalid sealed grid count ${COUNT}" >&2
  exit 2
fi
MAXIMUM=$((COUNT - 1))
GRID_JOB="$(sbatch --parsable --array="0-${MAXIMUM}%${LIMIT}" \
  --export="ALL,K_DIAGNOSTIC_CONFIG=${CONFIG}" \
  "${SCRIPT_DIR}/run_qwen35_4b_k_diagnostic_v1_grid.slurm")"
INDEX_JOB="$(sbatch --parsable --dependency="afterok:${GRID_JOB}" \
  --export="ALL,K_DIAGNOSTIC_CONFIG=${CONFIG}" \
  "${SCRIPT_DIR}/run_qwen35_4b_k_diagnostic_v1_index.slurm")"
printf 'stage=%s\ngrid_count=%s\ngrid_job=%s\nindex_job=%s\n' \
  "${STAGE}" "${COUNT}" "${GRID_JOB}" "${INDEX_JOB}"
