#!/usr/bin/env bash
#SBATCH --job-name=jlens-shared-occupancy-index
#SBATCH --account=p32737
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/concept_intervention/shared_intervention_protocol/slurm/%x-%j.out
#SBATCH --error=artifacts/concept_intervention/shared_intervention_protocol/slurm/%x-%j.err

set -euo pipefail
source "${CODE_ROOT:?}/Concept_intervention/scripts/three_method_env.sh"
jlens occupancy index "${SHARED_CONFIG}" \
  --output artifacts/concept_intervention/shared_intervention_protocol/j_occupancy
