#!/usr/bin/env bash
#SBATCH --job-name=jlens-shared-capture
#SBATCH --account=p32737
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/shared_intervention_protocol/slurm/%x-%j.out
#SBATCH --error=artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/shared_intervention_protocol/slurm/%x-%j.err

set -euo pipefail
source "${CODE_ROOT:?}/Concept_intervention/scripts/three_method_env.sh"
nvidia-smi
jlens concept capture "${SHARED_CONFIG}"
