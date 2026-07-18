#!/usr/bin/env bash
#SBATCH --job-name=jlens-concept-occ-pilot
#SBATCH --account=p32737
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "error: submit from the repository root or export REPO_ROOT" >&2
  exit 2
fi
CONFIG="${OCCUPANCY_CONFIG:-${REPO_ROOT}/Concept_intervention/configs/qwen35_4b_concept_occupancy_pilot_v1.yaml}"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

cd "${REPO_ROOT}"
echo "job=${SLURM_JOB_ID:-interactive} host=$(hostname) config=${CONFIG} hf_home=${HF_HOME}"
nvidia-smi
srun --ntasks=1 --unbuffered \
  bash Concept_intervention/scripts/run_qwen35_4b_concept_occupancy.sh "${CONFIG}"
