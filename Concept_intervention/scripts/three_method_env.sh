#!/usr/bin/env bash
# Shared immutable execution environment for the three intervention methods.
set -euo pipefail

CODE_ROOT="${CODE_ROOT:?submit with CODE_ROOT set to the immutable code checkout}"
RUN_ROOT="${RUN_ROOT:?submit with RUN_ROOT set to the artifact/data checkout}"
JLENS_GIT_COMMIT="${JLENS_GIT_COMMIT:?submit with JLENS_GIT_COMMIT frozen}"
PYTHON="${JLENS_PYTHON:-${CODE_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "error: experiment Python is not executable: ${PYTHON}" >&2
  exit 2
fi

JLENS_REPOSITORY_ROOT="${CODE_ROOT}"
export CODE_ROOT RUN_ROOT JLENS_GIT_COMMIT JLENS_REPOSITORY_ROOT
export PYTHONPATH="${CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
OBSERVED_COMMIT="$(
  "${PYTHON}" -c \
    'import sys; from jlens_workspace.artifacts import git_head_commit; print(git_head_commit(sys.argv[1]))' \
    "${CODE_ROOT}"
)"
if [[ "${OBSERVED_COMMIT}" != "${JLENS_GIT_COMMIT}" ]]; then
  echo "error: code checkout moved: ${OBSERVED_COMMIT} != ${JLENS_GIT_COMMIT}" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-${RUN_ROOT}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONHASHSEED=42
EXECUTION_THREADS="${SLURM_CPUS_PER_TASK:-${JLENS_CPUS_PER_TASK:-1}}"
export OMP_NUM_THREADS="${EXECUTION_THREADS}"
export MKL_NUM_THREADS="${EXECUTION_THREADS}"
export OPENBLAS_NUM_THREADS="${EXECUTION_THREADS}"

SHARED_CONFIG="${CODE_ROOT}/Concept_intervention/configs/qwen35_4b_three_method_intervention_v1_shared.yaml"
J_CONFIG="${CODE_ROOT}/Concept_intervention/configs/qwen35_4b_three_method_intervention_v1_j_component.yaml"
ITI_CONFIG="${CODE_ROOT}/Concept_intervention/configs/qwen35_4b_three_method_intervention_v1_iti.yaml"
RAPTOR_CONFIG="${CODE_ROOT}/Concept_intervention/configs/qwen35_4b_three_method_intervention_v1_raptor.yaml"
export SHARED_CONFIG J_CONFIG ITI_CONFIG RAPTOR_CONFIG

cd "${RUN_ROOT}"

jlens() {
  "${PYTHON}" -m jlens_workspace.cli "$@"
}
