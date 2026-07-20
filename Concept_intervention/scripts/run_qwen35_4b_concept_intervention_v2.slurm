#!/usr/bin/env bash
#SBATCH --job-name=jlens-concept-int-v2
#SBATCH --account=p32737
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --array=0-6%7
#SBATCH --output=artifacts/concept_intervention/qwen35_4b_concept_intervention_v2/slurm/%x-%A_%a.out
#SBATCH --error=artifacts/concept_intervention/qwen35_4b_concept_intervention_v2/slurm/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?submit with REPO_ROOT set to the immutable run checkout}"
JLENS_GIT_COMMIT="${JLENS_GIT_COMMIT:?submit with the frozen Git commit exported}"
CONFIG="${REPO_ROOT}/Concept_intervention/configs/qwen35_4b_concept_intervention_v2.yaml"
CLI="${JLENS_WORKSPACE_CLI:-${REPO_ROOT}/.venv/bin/jlens-workspace}"
OUTPUT="${REPO_ROOT}/artifacts/concept_intervention/qwen35_4b_concept_intervention_v2"
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

export JLENS_GIT_COMMIT
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"
echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} concept=${CONCEPT_ID} commit=${JLENS_GIT_COMMIT}"
nvidia-smi
"${CLI}" doctor --require-llm
"${CLI}" config validate "${CONFIG}"
exec "${CLI}" intervention concepts-v2 "${CONFIG}" \
  --output "${OUTPUT}" \
  --concept-id "${CONCEPT_ID}"
