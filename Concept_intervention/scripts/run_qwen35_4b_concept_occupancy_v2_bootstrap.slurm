#!/usr/bin/env bash
#SBATCH --job-name=jlens-occ-v2-probes
#SBATCH --account=p32737
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=artifacts/concept_intervention/qwen35_4b_concept_occupancy_v2/slurm/%x-%A_%a.out
#SBATCH --error=artifacts/concept_intervention/qwen35_4b_concept_occupancy_v2/slurm/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?submit with REPO_ROOT set to the immutable run checkout}"
CONFIG="${REPO_ROOT}/Concept_intervention/configs/qwen35_4b_concept_occupancy_v2.yaml"
CLI="${JLENS_WORKSPACE_CLI:-${REPO_ROOT}/.venv/bin/jlens-workspace}"
ACTIVATIONS="${OCCUPANCY_ACTIVATIONS:-${REPO_ROOT}/artifacts/concept_intervention/qwen35_4b_go_emotions_7concept_full_ovr_v2/activations}"
PRIMARY_PROBES="${OCCUPANCY_PROBES:-${REPO_ROOT}/artifacts/concept_intervention/qwen35_4b_go_emotions_7concept_full_ovr_v2/probes}"
OUTPUT="${REPO_ROOT}/artifacts/concept_intervention/qwen35_4b_concept_occupancy_v2/probe_replicates"
LAYERS=(8 12 16 20 24 28)
LAYER="${LAYERS[${SLURM_ARRAY_TASK_ID}]}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=42
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

cd "${REPO_ROOT}"
echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} layer=${LAYER} commit=$(git rev-parse HEAD)"
"${CLI}" config validate "${CONFIG}"
exec "${CLI}" concept bootstrap-probes "${CONFIG}" \
  --activations "${ACTIVATIONS}" \
  --primary-probes "${PRIMARY_PROBES}" \
  --output "${OUTPUT}" \
  --layer "${LAYER}"
