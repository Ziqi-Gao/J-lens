#!/usr/bin/env bash
#SBATCH --job-name=jlens-concept-occ-v2
#SBATCH --account=p32737
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --array=0-59%8
#SBATCH --output=artifacts/concept_intervention/qwen35_4b_concept_occupancy_v2/slurm/%x-%A_%a.out
#SBATCH --error=artifacts/concept_intervention/qwen35_4b_concept_occupancy_v2/slurm/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?submit with REPO_ROOT set to the immutable run checkout}"
CONFIG="${REPO_ROOT}/Concept_intervention/configs/qwen35_4b_concept_occupancy_v2.yaml"
CLI="${JLENS_WORKSPACE_CLI:-${REPO_ROOT}/.venv/bin/jlens-workspace}"
PRIMARY_PROBES="${OCCUPANCY_PROBES:-${REPO_ROOT}/artifacts/concept_intervention/qwen35_4b_go_emotions_7concept_full_ovr_v2/probes}"
PROBE_REPLICATES="${REPO_ROOT}/artifacts/concept_intervention/qwen35_4b_concept_occupancy_v2/probe_replicates"
OUTPUT="${REPO_ROOT}/artifacts/concept_intervention/qwen35_4b_concept_occupancy_v2"

LAYERS=(8 12 16 20 24 28)
CONVENTIONS=(raw rmsnorm_weighted)
REPLICATES=(primary bootstrap_1101 bootstrap_2202 bootstrap_3303 bootstrap_4404)
TASK="${SLURM_ARRAY_TASK_ID}"
LAYER="${LAYERS[$((TASK % 6))]}"
CONVENTION="${CONVENTIONS[$(((TASK / 6) % 2))]}"
REPLICATE="${REPLICATES[$((TASK / 12))]}"
if [[ "${REPLICATE}" == "primary" ]]; then
  K_MAX=64
else
  K_MAX=25
fi

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"
echo "job=${SLURM_JOB_ID} task=${TASK} layer=${LAYER} convention=${CONVENTION} replicate=${REPLICATE} k_max=${K_MAX} commit=$(git rev-parse HEAD)"
nvidia-smi
"${CLI}" doctor --require-llm
"${CLI}" config validate "${CONFIG}"
exec "${CLI}" occupancy concepts "${CONFIG}" \
  --probes "${PRIMARY_PROBES}" \
  --probe-replicates "${PROBE_REPLICATES}" \
  --output "${OUTPUT}" \
  --layer "${LAYER}" \
  --convention "${CONVENTION}" \
  --replicate-id "${REPLICATE}" \
  --k-max "${K_MAX}"
