#!/usr/bin/env bash
# Submit fail-closed shared preparation -> smoke -> full three-method grids.
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(git rev-parse --show-toplevel)}"
RUN_ROOT="${RUN_ROOT:-/gpfs/projects/p32737/del6500_home/J_lens}"
RAPTOR_ROOT="${RUN_ROOT}/third_party_external/RAPTOR"
RAPTOR_COMMIT="cf7405899174af39f3970e093e4b86bf0972ff87"

if [[ -n "$(git -C "${CODE_ROOT}" status --short)" ]]; then
  echo "error: three-method interventions require a clean immutable checkout" >&2
  exit 2
fi
JLENS_GIT_COMMIT="$(git -C "${CODE_ROOT}" rev-parse HEAD)"

if [[ ! -d "${RAPTOR_ROOT}/.git" ]]; then
  mkdir -p "${RUN_ROOT}/third_party_external"
  git clone https://github.com/Ziqi-Gao/RAPTOR.git "${RAPTOR_ROOT}"
fi
git -C "${RAPTOR_ROOT}" fetch origin "${RAPTOR_COMMIT}"
git -C "${RAPTOR_ROOT}" switch --detach "${RAPTOR_COMMIT}"
if [[ "$(git -C "${RAPTOR_ROOT}" rev-parse HEAD)" != "${RAPTOR_COMMIT}" ]]; then
  echo "error: failed to prepare pinned RAPTOR checkout" >&2
  exit 2
fi

mkdir -p \
  "${RUN_ROOT}/artifacts/concept_intervention/shared_intervention_protocol/slurm" \
  "${RUN_ROOT}/artifacts/concept_intervention/j_component_intervention/slurm" \
  "${RUN_ROOT}/artifacts/concept_intervention/iti_intervention/slurm" \
  "${RUN_ROOT}/artifacts/concept_intervention/raptor_intervention/slurm" \
  "${RUN_ROOT}/artifacts/concept_intervention/intervention_comparison/slurm"

export CODE_ROOT RUN_ROOT JLENS_GIT_COMMIT
cd "${RUN_ROOT}"
SCRIPTS="${CODE_ROOT}/Concept_intervention/scripts"

PREFLIGHT="$(sbatch --parsable "${SCRIPTS}/run_three_method_preflight.slurm")"
LENS="$(sbatch --parsable --dependency="afterok:${PREFLIGHT}" "${SCRIPTS}/run_shared_intervention_lens.slurm")"
CAPTURE="$(sbatch --parsable --dependency="afterok:${PREFLIGHT}" "${SCRIPTS}/run_shared_intervention_capture.slurm")"
LAYERS="$(sbatch --parsable --dependency="afterok:${CAPTURE}" "${SCRIPTS}/run_shared_layer_selection.slurm")"
BOOTSTRAP="$(sbatch --parsable --dependency="afterok:${LAYERS}" "${SCRIPTS}/run_shared_probe_bootstrap.slurm")"
BOOTSTRAP_INDEX="$(sbatch --parsable --dependency="afterok:${BOOTSTRAP}" "${SCRIPTS}/run_shared_probe_bootstrap_index.slurm")"
OCCUPANCY="$(sbatch --parsable --dependency="afterok:${LENS}:${BOOTSTRAP_INDEX}" "${SCRIPTS}/run_shared_j_occupancy.slurm")"
OCCUPANCY_INDEX="$(sbatch --parsable --dependency="afterok:${OCCUPANCY}" "${SCRIPTS}/run_shared_j_occupancy_index.slurm")"

ITI_CAPTURE="$(sbatch --parsable --dependency="afterok:${CAPTURE}" "${SCRIPTS}/run_iti_intervention_capture.slurm")"
ITI_FIT="$(sbatch --parsable --dependency="afterok:${ITI_CAPTURE}:${LAYERS}" "${SCRIPTS}/run_iti_intervention_fit.slurm")"

J_SMOKE="$(sbatch --parsable --array=6 --dependency="afterok:${OCCUPANCY_INDEX}" "${SCRIPTS}/run_j_component_grid.slurm")"
RAPTOR_SMOKE="$(sbatch --parsable --array=8 --dependency="afterok:${LAYERS}" "${SCRIPTS}/run_raptor_intervention_grid.slurm")"
ITI_SMOKE="$(sbatch --parsable --array=6,251 --dependency="afterok:${ITI_FIT}" "${SCRIPTS}/run_iti_intervention_grid.slurm")"
SMOKE_DEPENDENCY="afterok:${J_SMOKE}:${RAPTOR_SMOKE}:${ITI_SMOKE}"

FULL_SUBMISSION_GATE="$(
  sbatch --parsable --dependency="${SMOKE_DEPENDENCY}" \
    "${SCRIPTS}/run_three_method_full_submit.slurm"
)"

printf '%s\n' \
  "commit=${JLENS_GIT_COMMIT}" \
  "preflight=${PREFLIGHT}" \
  "lens=${LENS}" \
  "capture=${CAPTURE}" \
  "layers=${LAYERS}" \
  "bootstrap=${BOOTSTRAP}" \
  "bootstrap_index=${BOOTSTRAP_INDEX}" \
  "occupancy=${OCCUPANCY}" \
  "occupancy_index=${OCCUPANCY_INDEX}" \
  "iti_capture=${ITI_CAPTURE}" \
  "iti_fit=${ITI_FIT}" \
  "j_smoke=${J_SMOKE}" \
  "raptor_smoke=${RAPTOR_SMOKE}" \
  "iti_smoke=${ITI_SMOKE}" \
  "full_submission_gate=${FULL_SUBMISSION_GATE}"
