#!/usr/bin/env bash
#SBATCH --job-name=jlens-3method-preflight
#SBATCH --account=p32737
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/intervention_comparison/slurm/%x-%j.out
#SBATCH --error=artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/intervention_comparison/slurm/%x-%j.err

set -euo pipefail
source "${CODE_ROOT:?}/Concept_intervention/scripts/three_method_env.sh"

jlens doctor --require-llm
for config in "${SHARED_CONFIG}" "${J_CONFIG}" "${ITI_CONFIG}" "${RAPTOR_CONFIG}"; do
  jlens config validate "${config}"
done
jlens data validate --config "${SHARED_CONFIG}"
"${PYTHON}" -c "from jlens_workspace.concept_intervention.raptor import verify_raptor_adaptive_epsilon_parity, verify_raptor_checkout; root=verify_raptor_checkout('third_party_external/RAPTOR'); verify_raptor_adaptive_epsilon_parity(root)"
