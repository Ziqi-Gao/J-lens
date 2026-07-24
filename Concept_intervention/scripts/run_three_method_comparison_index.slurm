#!/usr/bin/env bash
#SBATCH --job-name=jlens-3method-index
#SBATCH --account=p32737
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/intervention_comparison/slurm/%x-%j.out
#SBATCH --error=artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/intervention_comparison/slurm/%x-%j.err

set -euo pipefail
source "${CODE_ROOT:?}/Concept_intervention/scripts/three_method_env.sh"

jlens intervention compare-index \
  --shared-layer-selection artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/shared_intervention_protocol/selection/layer_selection.json \
  --j-output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/j_component_intervention \
  --iti-output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/iti_intervention \
  --raptor-output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/raptor_intervention \
  --output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/intervention_comparison \
  --concept-id goemotions:admiration \
  --concept-id goemotions:approval \
  --concept-id goemotions:curiosity \
  --concept-id goemotions:disapproval \
  --concept-id goemotions:gratitude \
  --concept-id goemotions:love \
  --concept-id goemotions:optimism
