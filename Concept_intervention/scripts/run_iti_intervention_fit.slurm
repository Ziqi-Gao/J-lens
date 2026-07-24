#!/usr/bin/env bash
#SBATCH --job-name=jlens-iti-fit
#SBATCH --account=p32737
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/iti_intervention/slurm/%x-%j.out
#SBATCH --error=artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/iti_intervention/slurm/%x-%j.err

set -euo pipefail
source "${CODE_ROOT:?}/Concept_intervention/scripts/three_method_env.sh"
jlens iti fit "${ITI_CONFIG}"
