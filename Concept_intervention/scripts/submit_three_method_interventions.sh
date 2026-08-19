#!/usr/bin/env bash
# Submit fail-closed shared preparation -> smoke -> full three-method grids.
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(git rev-parse --show-toplevel)}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT to a persistent experiment workspace}"
RAPTOR_ROOT="${RUN_ROOT}/third_party_external/RAPTOR"
RAPTOR_COMMIT="cf7405899174af39f3970e093e4b86bf0972ff87"
source "${CODE_ROOT}/Concept_intervention/scripts/three_method_submit_env.sh"

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
  "${RUN_ROOT}/artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/shared_intervention_protocol/slurm" \
  "${RUN_ROOT}/artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/j_component_intervention/slurm" \
  "${RUN_ROOT}/artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/iti_intervention/slurm" \
  "${RUN_ROOT}/artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/raptor_intervention/slurm" \
  "${RUN_ROOT}/artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/intervention_comparison/slurm"

export CODE_ROOT RUN_ROOT JLENS_GIT_COMMIT JLENS_PYTHON HF_HOME HF_HUB_CACHE
cd "${RUN_ROOT}"
SCRIPTS="${CODE_ROOT}/Concept_intervention/scripts"

PREFLIGHT="$(three_method_sbatch_cpu --parsable "${SCRIPTS}/run_three_method_preflight.slurm")"
LENS="$(three_method_sbatch_gpu --parsable --dependency="afterok:${PREFLIGHT}" "${SCRIPTS}/run_shared_intervention_lens.slurm")"
CAPTURE="$(three_method_sbatch_gpu --parsable --dependency="afterok:${PREFLIGHT}" "${SCRIPTS}/run_shared_intervention_capture.slurm")"
LAYERS="$(three_method_sbatch_cpu --parsable --dependency="afterok:${CAPTURE}" "${SCRIPTS}/run_shared_layer_selection.slurm")"
BOOTSTRAP="$(three_method_sbatch_cpu --parsable --array="0-6%${SLURM_BOOTSTRAP_LIMIT}" --dependency="afterok:${LAYERS}" "${SCRIPTS}/run_shared_probe_bootstrap.slurm")"
BOOTSTRAP_INDEX="$(three_method_sbatch_cpu --parsable --dependency="afterok:${BOOTSTRAP}" "${SCRIPTS}/run_shared_probe_bootstrap_index.slurm")"
OCCUPANCY="$(three_method_sbatch_gpu --parsable --array="0-34%${SLURM_OCCUPANCY_LIMIT}" --dependency="afterok:${LENS}:${BOOTSTRAP_INDEX}" "${SCRIPTS}/run_shared_j_occupancy.slurm")"
OCCUPANCY_INDEX="$(three_method_sbatch_cpu --parsable --dependency="afterok:${OCCUPANCY}" "${SCRIPTS}/run_shared_j_occupancy_index.slurm")"

ITI_CAPTURE="$(three_method_sbatch_gpu --parsable --dependency="afterok:${CAPTURE}" "${SCRIPTS}/run_iti_intervention_capture.slurm")"
ITI_FIT="$(three_method_sbatch_cpu --parsable --dependency="afterok:${ITI_CAPTURE}:${LAYERS}" "${SCRIPTS}/run_iti_intervention_fit.slurm")"

J_SMOKE="$(three_method_sbatch_gpu --parsable --array=6 --dependency="afterok:${OCCUPANCY_INDEX}" "${SCRIPTS}/run_j_component_grid.slurm")"
RAPTOR_SMOKE="$(three_method_sbatch_gpu --parsable --array=8 --dependency="afterok:${LAYERS}" "${SCRIPTS}/run_raptor_intervention_grid.slurm")"
ITI_SMOKE="$(three_method_sbatch_gpu --parsable --array=6,251 --dependency="afterok:${ITI_FIT}" "${SCRIPTS}/run_iti_intervention_grid.slurm")"
SMOKE_DEPENDENCY="afterok:${J_SMOKE}:${RAPTOR_SMOKE}:${ITI_SMOKE}"

FULL_SUBMISSION_GATE="$(
  three_method_sbatch_cpu --parsable --dependency="${SMOKE_DEPENDENCY}" \
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
