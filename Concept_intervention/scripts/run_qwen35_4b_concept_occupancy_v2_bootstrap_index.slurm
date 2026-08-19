#!/usr/bin/env bash
#SBATCH --job-name=jlens-occ-v2-probe-index
#SBATCH --account=p32737
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/concept_intervention/qwen35_4b_concept_occupancy_v2/slurm/%x-%j.out
#SBATCH --error=artifacts/concept_intervention/qwen35_4b_concept_occupancy_v2/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?submit with REPO_ROOT set to the immutable run checkout}"
CONFIG="${REPO_ROOT}/Concept_intervention/configs/qwen35_4b_concept_occupancy_v2.yaml"
CLI="${JLENS_WORKSPACE_CLI:-${REPO_ROOT}/.venv/bin/jlens-workspace}"
ACTIVATIONS="${OCCUPANCY_ACTIVATIONS:-${REPO_ROOT}/artifacts/concept_intervention/qwen35_4b_go_emotions_7concept_full_ovr_v2/activations}"
PRIMARY_PROBES="${OCCUPANCY_PROBES:-${REPO_ROOT}/artifacts/concept_intervention/qwen35_4b_go_emotions_7concept_full_ovr_v2/probes}"
OUTPUT="${REPO_ROOT}/artifacts/concept_intervention/qwen35_4b_concept_occupancy_v2/probe_replicates"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"
exec "${CLI}" concept bootstrap-probes-index "${CONFIG}" \
  --activations "${ACTIVATIONS}" \
  --primary-probes "${PRIMARY_PROBES}" \
  --output "${OUTPUT}"
