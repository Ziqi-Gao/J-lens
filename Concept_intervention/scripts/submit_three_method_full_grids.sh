#!/usr/bin/env bash
# Validate completed GPU smokes, then submit the exhaustive grids and indexes.
set -euo pipefail

source "${CODE_ROOT:?}/Concept_intervention/scripts/three_method_env.sh"
SCRIPTS="${CODE_ROOT}/Concept_intervention/scripts"
source "${SCRIPTS}/three_method_submit_env.sh"
COMPARISON_ROOT="artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/intervention_comparison"

jlens intervention smoke-check \
  --j-output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/j_component_intervention \
  --iti-output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/iti_intervention \
  --raptor-output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/raptor_intervention \
  --output "${COMPARISON_ROOT}/smoke_gate.json"

# The four completed smoke tasks are omitted from the exhaustive arrays.
J_GRID="$(
  three_method_sbatch_gpu --parsable --array="0-5,7-391%${SLURM_J_LIMIT}" \
    "${SCRIPTS}/run_j_component_grid.slurm"
)"
RAPTOR_GRID="$(
  three_method_sbatch_gpu --parsable --array="0-7,9-62%${SLURM_RAPTOR_LIMIT}" \
    "${SCRIPTS}/run_raptor_intervention_grid.slurm"
)"
ITI_GRID="$(
  three_method_sbatch_gpu --parsable --array="0-5,7-250,252-3086%${SLURM_ITI_LIMIT}" \
    "${SCRIPTS}/run_iti_intervention_grid.slurm"
)"
METHOD_INDEX="$(
  three_method_sbatch_cpu --parsable --dependency="afterok:${J_GRID}:${RAPTOR_GRID}:${ITI_GRID}" \
    "${SCRIPTS}/run_three_method_index.slurm"
)"
COMPARISON="$(
  three_method_sbatch_cpu --parsable --dependency="afterok:${METHOD_INDEX}" \
    "${SCRIPTS}/run_three_method_comparison_index.slurm"
)"

printf '%s\n' \
  "smoke_gate=${COMPARISON_ROOT}/smoke_gate.json" \
  "j_grid=${J_GRID}" \
  "raptor_grid=${RAPTOR_GRID}" \
  "iti_grid=${ITI_GRID}" \
  "method_index=${METHOD_INDEX}" \
  "comparison=${COMPARISON}"
