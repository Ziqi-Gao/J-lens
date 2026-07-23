#!/usr/bin/env bash
#SBATCH --job-name=jlens-shared-bootstrap
#SBATCH --account=p32737
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-6%7
#SBATCH --output=artifacts/concept_intervention/shared_intervention_protocol/slurm/%x-%A_%a.out
#SBATCH --error=artifacts/concept_intervention/shared_intervention_protocol/slurm/%x-%A_%a.err

set -euo pipefail
source "${CODE_ROOT:?}/Concept_intervention/scripts/three_method_env.sh"

LAYERS=(3 7 11 15 19 23 27)
LAYER="${LAYERS[${SLURM_ARRAY_TASK_ID}]}"
jlens concept bootstrap-probes "${SHARED_CONFIG}" \
  --activations artifacts/concept_intervention/shared_intervention_protocol/activations \
  --primary-probes artifacts/concept_intervention/shared_intervention_protocol/selection/raptor_probes \
  --output artifacts/concept_intervention/shared_intervention_protocol/selection/probe_replicates \
  --layer "${LAYER}"
