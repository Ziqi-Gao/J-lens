#!/usr/bin/env bash
# Submit capture -> direction fit -> intervention -> fail-closed J comparison.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
OUTPUT="${REPO_ROOT}/artifacts/concept_intervention/qwen35_4b_iti_comparison_v1"
mkdir -p "${OUTPUT}/slurm"

if [[ -n "$(git -C "${REPO_ROOT}" status --short --untracked-files=no)" ]]; then
  echo "error: ITI must be submitted from a clean immutable checkout" >&2
  exit 2
fi

JLENS_GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
export REPO_ROOT JLENS_GIT_COMMIT
CAPTURE_JOB="$(
  sbatch --parsable \
    "${REPO_ROOT}/Concept_intervention/scripts/run_qwen35_4b_iti_capture_v1.slurm"
)"
FIT_JOB="$(
  sbatch --parsable --dependency="afterok:${CAPTURE_JOB}" \
    "${REPO_ROOT}/Concept_intervention/scripts/run_qwen35_4b_iti_fit_v1.slurm"
)"
PILOT_JOB="$(
  sbatch --parsable --array=5 --dependency="afterok:${FIT_JOB}" \
    "${REPO_ROOT}/Concept_intervention/scripts/run_qwen35_4b_iti_intervention_v1.slurm"
)"
INTERVENTION_JOB="$(
  sbatch --parsable --array=0-4,6%6 --dependency="afterok:${PILOT_JOB}" \
    "${REPO_ROOT}/Concept_intervention/scripts/run_qwen35_4b_iti_intervention_v1.slurm"
)"
INDEX_DEPENDENCY="afterok:${INTERVENTION_JOB}"
if [[ -n "${J_INTERVENTION_JOB_ID:-}" ]]; then
  INDEX_DEPENDENCY="${INDEX_DEPENDENCY}:${J_INTERVENTION_JOB_ID}"
fi
INDEX_JOB="$(
  sbatch --parsable --dependency="${INDEX_DEPENDENCY}" \
    "${REPO_ROOT}/Concept_intervention/scripts/run_qwen35_4b_iti_index_v1.slurm"
)"

printf 'capture=%s\nfit=%s\npilot=%s\nintervention=%s\nindex=%s\ncommit=%s\n' \
  "${CAPTURE_JOB}" "${FIT_JOB}" "${PILOT_JOB}" "${INTERVENTION_JOB}" \
  "${INDEX_JOB}" "${JLENS_GIT_COMMIT}"
