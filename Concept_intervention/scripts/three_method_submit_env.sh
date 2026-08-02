#!/usr/bin/env bash
# Portable sbatch overrides for the registered three-method experiment.

if [[ "${THREE_METHOD_SUBMIT_ENV_LOADED:-0}" == "1" ]]; then
  return 0 2>/dev/null || exit 0
fi
THREE_METHOD_SUBMIT_ENV_LOADED=1

SLURM_ACCOUNT="${SLURM_ACCOUNT:-p32737}"
SLURM_CPU_PARTITION="${SLURM_CPU_PARTITION:-normal}"
SLURM_GPU_PARTITION="${SLURM_GPU_PARTITION:-gengpu}"
SLURM_GPU_GRES="${SLURM_GPU_GRES:-gpu:a100:1}"
SLURM_EXCLUDE="${SLURM_EXCLUDE:-}"
SLURM_BOOTSTRAP_LIMIT="${SLURM_BOOTSTRAP_LIMIT:-7}"
SLURM_OCCUPANCY_LIMIT="${SLURM_OCCUPANCY_LIMIT:-10}"
SLURM_J_LIMIT="${SLURM_J_LIMIT:-16}"
SLURM_RAPTOR_LIMIT="${SLURM_RAPTOR_LIMIT:-16}"
SLURM_ITI_LIMIT="${SLURM_ITI_LIMIT:-24}"

for limit_name in \
  SLURM_BOOTSTRAP_LIMIT SLURM_OCCUPANCY_LIMIT \
  SLURM_J_LIMIT SLURM_RAPTOR_LIMIT SLURM_ITI_LIMIT; do
  limit_value="${!limit_name}"
  if [[ ! "${limit_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: ${limit_name} must be a positive integer" >&2
    return 2 2>/dev/null || exit 2
  fi
done

SBATCH_CPU_ARGS=(
  --account="${SLURM_ACCOUNT}"
  --partition="${SLURM_CPU_PARTITION}"
)
SBATCH_GPU_ARGS=(
  --account="${SLURM_ACCOUNT}"
  --partition="${SLURM_GPU_PARTITION}"
  --gres="${SLURM_GPU_GRES}"
)
if [[ -n "${SLURM_EXCLUDE}" ]]; then
  SBATCH_GPU_ARGS+=(--exclude="${SLURM_EXCLUDE}")
fi

export SLURM_ACCOUNT SLURM_CPU_PARTITION SLURM_GPU_PARTITION
export SLURM_GPU_GRES SLURM_EXCLUDE
export SLURM_BOOTSTRAP_LIMIT SLURM_OCCUPANCY_LIMIT
export SLURM_J_LIMIT SLURM_RAPTOR_LIMIT SLURM_ITI_LIMIT

three_method_sbatch_cpu() {
  sbatch "${SBATCH_CPU_ARGS[@]}" "$@"
}

three_method_sbatch_gpu() {
  sbatch "${SBATCH_GPU_ARGS[@]}" "$@"
}
