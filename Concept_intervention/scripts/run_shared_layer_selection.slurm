#!/usr/bin/env bash
#SBATCH --job-name=jlens-shared-layers
#SBATCH --account=p32737
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/concept_intervention/shared_intervention_protocol/slurm/%x-%j.out
#SBATCH --error=artifacts/concept_intervention/shared_intervention_protocol/slurm/%x-%j.err

set -euo pipefail
source "${CODE_ROOT:?}/Concept_intervention/scripts/three_method_env.sh"
jlens concept layer-select "${SHARED_CONFIG}"
