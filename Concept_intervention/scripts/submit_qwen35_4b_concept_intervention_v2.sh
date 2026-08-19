#!/usr/bin/env bash
# Submit the registered intervention array and its fail-closed final index.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
OUTPUT="${REPO_ROOT}/artifacts/concept_intervention/qwen35_4b_concept_intervention_v2"
mkdir -p "${OUTPUT}/slurm"

if [[ -n "$(git -C "${REPO_ROOT}" status --short --untracked-files=no)" ]]; then
  echo "error: intervention must be submitted from a clean immutable checkout" >&2
  exit 2
fi

JLENS_GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
export REPO_ROOT JLENS_GIT_COMMIT
PILOT_JOB="$(
  sbatch --parsable --array=5 \
    "${REPO_ROOT}/Concept_intervention/scripts/run_qwen35_4b_concept_intervention_v2.slurm"
)"
INTERVENTION_JOB="$(
  sbatch --parsable --array=0-4,6%6 --dependency="afterok:${PILOT_JOB}" \
    "${REPO_ROOT}/Concept_intervention/scripts/run_qwen35_4b_concept_intervention_v2.slurm"
)"
INDEX_JOB="$(
  sbatch --parsable --dependency="afterok:${INTERVENTION_JOB}" \
    "${REPO_ROOT}/Concept_intervention/scripts/run_qwen35_4b_concept_intervention_v2_index.slurm"
)"

printf 'pilot=%s\nintervention=%s\nindex=%s\ncommit=%s\n' \
  "${PILOT_JOB}" "${INTERVENTION_JOB}" "${INDEX_JOB}" "${JLENS_GIT_COMMIT}"
