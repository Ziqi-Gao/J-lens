#!/usr/bin/env bash
#SBATCH --job-name=jlens-concept-int-v2-index
#SBATCH --account=p32737
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/concept_intervention/qwen35_4b_concept_intervention_v2/slurm/%x-%j.out
#SBATCH --error=artifacts/concept_intervention/qwen35_4b_concept_intervention_v2/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?submit with REPO_ROOT set to the immutable run checkout}"
JLENS_GIT_COMMIT="${JLENS_GIT_COMMIT:?submit with the frozen Git commit exported}"
CONFIG="${REPO_ROOT}/Concept_intervention/configs/qwen35_4b_concept_intervention_v2.yaml"
CLI="${JLENS_WORKSPACE_CLI:-${REPO_ROOT}/.venv/bin/jlens-workspace}"
OUTPUT="${REPO_ROOT}/artifacts/concept_intervention/qwen35_4b_concept_intervention_v2"

export JLENS_GIT_COMMIT
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"
exec "${CLI}" intervention index "${CONFIG}" --output "${OUTPUT}"
