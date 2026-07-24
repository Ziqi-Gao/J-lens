#!/usr/bin/env bash
#SBATCH --job-name=jlens-3method-submit
#SBATCH --account=p32737
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/intervention_comparison/slurm/%x-%j.out
#SBATCH --error=artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/intervention_comparison/slurm/%x-%j.err

set -euo pipefail
bash "${CODE_ROOT:?}/Concept_intervention/scripts/submit_three_method_full_grids.sh"
