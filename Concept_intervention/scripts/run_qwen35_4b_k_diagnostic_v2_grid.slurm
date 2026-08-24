#!/usr/bin/env bash
#SBATCH --job-name=jlens-kdiag-v2-bundle
#SBATCH --account=p32737
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err

set -euo pipefail
: "${CODE_ROOT:?CODE_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${K_DIAGNOSTIC_CONFIG:?K_DIAGNOSTIC_CONFIG is required}"
: "${K_DIAGNOSTIC_BUNDLE_OFFSET:?K_DIAGNOSTIC_BUNDLE_OFFSET is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
PYTHON="${JLENS_PYTHON:-/scr/del6500/J-lens/envs/three-method/bin/python}"
export PYTHONPATH="${CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
BUNDLE_INDEX=$((K_DIAGNOSTIC_BUNDLE_OFFSET + SLURM_ARRAY_TASK_ID))
EXTRA_ARGUMENTS=()
if [[ "${K_DIAGNOSTIC_MICROBENCHMARK:-0}" == 1 ]]; then
  EXTRA_ARGUMENTS+=(--microbenchmark)
fi
cd "${RUN_ROOT}"
exec "${PYTHON}" -m jlens_workspace.cli k-diagnostic run \
  "${K_DIAGNOSTIC_CONFIG}" --bundle-index "${BUNDLE_INDEX}" "${EXTRA_ARGUMENTS[@]}"
