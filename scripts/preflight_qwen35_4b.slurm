#!/usr/bin/env bash
#SBATCH --job-name=jlens-qwen4b-preflight
#SBATCH --account=p32737
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "error: submit from the repository root or export REPO_ROOT" >&2
  exit 2
fi
CONFIG="${PREFLIGHT_CONFIG:-${REPO_ROOT}/J_space/configs/qwen35_4b_centered.yaml}"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${REPO_ROOT}"
echo "job=${SLURM_JOB_ID:-interactive} host=$(hostname) config=${CONFIG}"
nvidia-smi
srun --ntasks=1 --unbuffered .venv/bin/python scripts/preflight_qwen35_4b.py "${CONFIG}"
