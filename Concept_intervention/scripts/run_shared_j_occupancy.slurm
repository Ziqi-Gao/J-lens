#!/usr/bin/env bash
#SBATCH --job-name=jlens-shared-occupancy
#SBATCH --account=p32737
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --array=0-34%10
#SBATCH --output=artifacts/concept_intervention/shared_intervention_protocol/slurm/%x-%A_%a.out
#SBATCH --error=artifacts/concept_intervention/shared_intervention_protocol/slurm/%x-%A_%a.err

set -euo pipefail
source "${CODE_ROOT:?}/Concept_intervention/scripts/three_method_env.sh"

LAYERS=(3 7 11 15 19 23 27)
REPLICATES=(primary bootstrap_1101 bootstrap_2202 bootstrap_3303 bootstrap_4404)
TASK="${SLURM_ARRAY_TASK_ID}"
LAYER="${LAYERS[$((TASK % 7))]}"
REPLICATE="${REPLICATES[$((TASK / 7))]}"
nvidia-smi
jlens occupancy concepts "${SHARED_CONFIG}" \
  --probes artifacts/concept_intervention/shared_intervention_protocol/selection/raptor_probes \
  --probe-replicates artifacts/concept_intervention/shared_intervention_protocol/selection/probe_replicates \
  --output artifacts/concept_intervention/shared_intervention_protocol/j_occupancy \
  --layer "${LAYER}" \
  --convention rmsnorm_weighted \
  --replicate-id "${REPLICATE}" \
  --k-max 64
