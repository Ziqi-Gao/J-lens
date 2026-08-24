#!/usr/bin/env bash
#SBATCH --job-name=jlens-kdiag-stage
#SBATCH --account=p32737
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail
: "${CODE_ROOT:?CODE_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${K_DIAGNOSTIC_CONFIG:?K_DIAGNOSTIC_CONFIG is required}"
: "${K_DIAGNOSTIC_ACTION:?K_DIAGNOSTIC_ACTION is required}"
case "${K_DIAGNOSTIC_ACTION}" in
  prepare-targets|build-bases) ;;
  *) echo "error: invalid stage action ${K_DIAGNOSTIC_ACTION}" >&2; exit 2 ;;
esac
PYTHON="${JLENS_PYTHON:-/scr/del6500/J-lens/envs/three-method/bin/python}"
export PYTHONPATH="${CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
cd "${RUN_ROOT}"
nvidia-smi
"${PYTHON}" -m jlens_workspace.cli k-diagnostic validate "${K_DIAGNOSTIC_CONFIG}"
exec "${PYTHON}" -m jlens_workspace.cli k-diagnostic \
  "${K_DIAGNOSTIC_ACTION}" "${K_DIAGNOSTIC_CONFIG}"
