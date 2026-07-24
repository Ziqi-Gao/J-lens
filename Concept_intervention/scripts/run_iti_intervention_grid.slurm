#!/usr/bin/env bash
#SBATCH --job-name=jlens-iti-grid
#SBATCH --account=p32737
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --array=0-3086%24
#SBATCH --output=artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/iti_intervention/slurm/%x-%A_%a.out
#SBATCH --error=artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/iti_intervention/slurm/%x-%A_%a.err

set -euo pipefail
source "${CODE_ROOT:?}/Concept_intervention/scripts/three_method_env.sh"

CONCEPTS=(
  goemotions:admiration goemotions:approval goemotions:curiosity
  goemotions:disapproval goemotions:gratitude goemotions:love
  goemotions:optimism
)
TASK="${SLURM_ARRAY_TASK_ID}"
CONCEPT="${CONCEPTS[$((TASK / 441))]}"
GRID_INDEX="$((TASK % 441))"
nvidia-smi
jlens iti run "${ITI_CONFIG}" \
  --concept-id "${CONCEPT}" \
  --grid-index "${GRID_INDEX}"
