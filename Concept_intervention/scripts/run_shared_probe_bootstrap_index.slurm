#!/usr/bin/env bash
#SBATCH --job-name=jlens-shared-bootstrap-index
#SBATCH --account=p32737
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/shared_intervention_protocol/slurm/%x-%j.out
#SBATCH --error=artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/shared_intervention_protocol/slurm/%x-%j.err

set -euo pipefail
source "${CODE_ROOT:?}/Concept_intervention/scripts/three_method_env.sh"
jlens concept bootstrap-probes-index "${SHARED_CONFIG}" \
  --activations artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/shared_intervention_protocol/activations \
  --primary-probes artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/shared_intervention_protocol/selection/raptor_probes \
  --output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/shared_intervention_protocol/selection/probe_replicates
