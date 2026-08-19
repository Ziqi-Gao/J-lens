#!/usr/bin/env bash
#SBATCH --job-name=jlens-method-index
#SBATCH --account=p32737
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/intervention_comparison/slurm/%x-%A_%a.out
#SBATCH --error=artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/intervention_comparison/slurm/%x-%A_%a.err

set -euo pipefail
source "${CODE_ROOT:?}/Concept_intervention/scripts/three_method_env.sh"

case "${SLURM_ARRAY_TASK_ID}" in
  0) jlens intervention j-component-index "${J_CONFIG}" ;;
  1) jlens iti index "${ITI_CONFIG}" ;;
  2) jlens intervention raptor-index "${RAPTOR_CONFIG}" ;;
  *) echo "invalid index task" >&2; exit 2 ;;
esac
