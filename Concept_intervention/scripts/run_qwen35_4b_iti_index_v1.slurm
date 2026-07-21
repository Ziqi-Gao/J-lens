#!/usr/bin/env bash
#SBATCH --job-name=jlens-iti-idx-v1
#SBATCH --account=p32737
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/concept_intervention/qwen35_4b_iti_comparison_v1/slurm/%x-%j.out
#SBATCH --error=artifacts/concept_intervention/qwen35_4b_iti_comparison_v1/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?submit with REPO_ROOT set to the immutable run checkout}"
JLENS_GIT_COMMIT="${JLENS_GIT_COMMIT:?submit with the frozen Git commit exported}"
CONFIG="${REPO_ROOT}/Concept_intervention/configs/qwen35_4b_iti_comparison_v1.yaml"
CLI="${JLENS_WORKSPACE_CLI:-${REPO_ROOT}/.venv/bin/jlens-workspace}"

export JLENS_GIT_COMMIT PYTHONUNBUFFERED=1 PYTHONHASHSEED=42
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"
"${CLI}" config validate "${CONFIG}"
exec "${CLI}" iti index "${CONFIG}"
