#!/usr/bin/env bash
# Submit the complete v2 dependency chain from an immutable run checkout.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
OUTPUT="${REPO_ROOT}/artifacts/concept_intervention/qwen35_4b_concept_occupancy_v2"
mkdir -p "${OUTPUT}/slurm"

if [[ -n "$(git -C "${REPO_ROOT}" status --short)" ]]; then
  echo "error: v2 must be submitted from a clean immutable checkout" >&2
  exit 2
fi

export REPO_ROOT
BOOTSTRAP_JOB="$(
  sbatch --parsable \
    "${REPO_ROOT}/Concept_intervention/scripts/run_qwen35_4b_concept_occupancy_v2_bootstrap.slurm"
)"
BOOTSTRAP_INDEX_JOB="$(
  sbatch --parsable --dependency="afterok:${BOOTSTRAP_JOB}" \
    "${REPO_ROOT}/Concept_intervention/scripts/run_qwen35_4b_concept_occupancy_v2_bootstrap_index.slurm"
)"
OCCUPANCY_JOB="$(
  sbatch --parsable --dependency="afterok:${BOOTSTRAP_INDEX_JOB}" \
    "${REPO_ROOT}/Concept_intervention/scripts/run_qwen35_4b_concept_occupancy_v2.slurm"
)"
INDEX_JOB="$(
  sbatch --parsable --dependency="afterok:${OCCUPANCY_JOB}" \
    "${REPO_ROOT}/Concept_intervention/scripts/run_qwen35_4b_concept_occupancy_v2_index.slurm"
)"

printf 'bootstrap=%s\nbootstrap_index=%s\noccupancy=%s\nindex=%s\n' \
  "${BOOTSTRAP_JOB}" "${BOOTSTRAP_INDEX_JOB}" "${OCCUPANCY_JOB}" "${INDEX_JOB}"
