#!/usr/bin/env bash
# Validate completed GPU smokes, then submit the exhaustive grids and indexes.
set -euo pipefail

source "${CODE_ROOT:?}/Concept_intervention/scripts/three_method_env.sh"
SCRIPTS="${CODE_ROOT}/Concept_intervention/scripts"
COMPARISON_ROOT="artifacts/concept_intervention/intervention_comparison"

jlens intervention smoke-check \
  --j-output artifacts/concept_intervention/j_component_intervention \
  --iti-output artifacts/concept_intervention/iti_intervention \
  --raptor-output artifacts/concept_intervention/raptor_intervention \
  --output "${COMPARISON_ROOT}/smoke_gate.json"

# The four completed smoke tasks are omitted from the exhaustive arrays.
J_GRID="$(
  sbatch --parsable --array=0-5,7-391%16 \
    "${SCRIPTS}/run_j_component_grid.slurm"
)"
RAPTOR_GRID="$(
  sbatch --parsable --array=0-7,9-62%16 \
    "${SCRIPTS}/run_raptor_intervention_grid.slurm"
)"
ITI_GRID="$(
  sbatch --parsable --array=0-5,7-250,252-3086%24 \
    "${SCRIPTS}/run_iti_intervention_grid.slurm"
)"
METHOD_INDEX="$(
  sbatch --parsable --dependency="afterok:${J_GRID}:${RAPTOR_GRID}:${ITI_GRID}" \
    "${SCRIPTS}/run_three_method_index.slurm"
)"
COMPARISON="$(
  sbatch --parsable --dependency="afterok:${METHOD_INDEX}" \
    "${SCRIPTS}/run_three_method_comparison_index.slurm"
)"

printf '%s\n' \
  "smoke_gate=${COMPARISON_ROOT}/smoke_gate.json" \
  "j_grid=${J_GRID}" \
  "raptor_grid=${RAPTOR_GRID}" \
  "iti_grid=${ITI_GRID}" \
  "method_index=${METHOD_INDEX}" \
  "comparison=${COMPARISON}"
