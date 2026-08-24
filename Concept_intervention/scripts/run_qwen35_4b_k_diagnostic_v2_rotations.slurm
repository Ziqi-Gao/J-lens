#!/usr/bin/env bash
#SBATCH --job-name=jlens-kdiag-v2-haar
#SBATCH --account=p32737
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail
: "${CODE_ROOT:?CODE_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${K_DIAGNOSTIC_CONFIG:?K_DIAGNOSTIC_CONFIG is required}"
PYTHON="${JLENS_PYTHON:-/scr/del6500/J-lens/envs/three-method/bin/python}"
export PYTHONPATH="${CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=42
cd "${RUN_ROOT}"
"${PYTHON}" -m jlens_workspace.cli k-diagnostic \
  rotation-cache-preflight "${K_DIAGNOSTIC_CONFIG}"
exec "${PYTHON}" -m jlens_workspace.cli k-diagnostic \
  prepare-rotations "${K_DIAGNOSTIC_CONFIG}"
