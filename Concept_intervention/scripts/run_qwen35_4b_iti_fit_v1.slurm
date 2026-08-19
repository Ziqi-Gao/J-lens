#!/usr/bin/env bash
#SBATCH --job-name=jlens-iti-fit-v1
#SBATCH --account=p32737
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-6%7
#SBATCH --output=artifacts/concept_intervention/qwen35_4b_iti_comparison_v1/slurm/%x-%A_%a.out
#SBATCH --error=artifacts/concept_intervention/qwen35_4b_iti_comparison_v1/slurm/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?submit with REPO_ROOT set to the immutable run checkout}"
JLENS_GIT_COMMIT="${JLENS_GIT_COMMIT:?submit with the frozen Git commit exported}"
CONFIG="${REPO_ROOT}/Concept_intervention/configs/qwen35_4b_iti_comparison_v1.yaml"
CLI="${JLENS_WORKSPACE_CLI:-${REPO_ROOT}/.venv/bin/jlens-workspace}"
CONCEPTS=(
  "goemotions:admiration"
  "goemotions:approval"
  "goemotions:curiosity"
  "goemotions:disapproval"
  "goemotions:gratitude"
  "goemotions:love"
  "goemotions:optimism"
)
CONCEPT_ID="${CONCEPTS[${SLURM_ARRAY_TASK_ID}]}"

export JLENS_GIT_COMMIT PYTHONUNBUFFERED=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"
"${CLI}" config validate "${CONFIG}"
exec "${CLI}" iti fit "${CONFIG}" --concept-id "${CONCEPT_ID}"
