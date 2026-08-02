#!/usr/bin/env bash
# Run one J-lens command on a capacity-checked GPU without touching foreign work.
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 COMMAND [ARG ...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/three_method_local_paths.sh"

GPU_IDS="${JLENS_LOCAL_GPU_IDS:-0,1,2,3}"
MIN_FREE_MIB="${JLENS_LOCAL_GPU_MIN_FREE_MIB:-65536}"
MAX_UTILIZATION="${JLENS_LOCAL_GPU_MAX_UTILIZATION:-70}"
ALLOW_OVERLAY="${JLENS_LOCAL_GPU_ALLOW_OVERLAY:-1}"
STABILITY_SAMPLES="${JLENS_LOCAL_GPU_STABILITY_SAMPLES:-3}"
SAMPLE_INTERVAL="${JLENS_LOCAL_GPU_SAMPLE_INTERVAL_SECONDS:-2}"
POLL_SECONDS="${JLENS_LOCAL_GPU_POLL_SECONDS:-30}"
WAIT_TIMEOUT="${JLENS_LOCAL_GPU_WAIT_TIMEOUT_SECONDS:-0}"

for pair in \
  "MIN_FREE_MIB:${MIN_FREE_MIB}" "MAX_UTILIZATION:${MAX_UTILIZATION}" \
  "STABILITY_SAMPLES:${STABILITY_SAMPLES}" "SAMPLE_INTERVAL:${SAMPLE_INTERVAL}" \
  "POLL_SECONDS:${POLL_SECONDS}" "WAIT_TIMEOUT:${WAIT_TIMEOUT}"; do
  value="${pair#*:}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "error: ${pair%%:*} must be a non-negative integer" >&2
    exit 2
  fi
done
case "${ALLOW_OVERLAY}" in
  0|1) ;;
  *) echo "error: JLENS_LOCAL_GPU_ALLOW_OVERLAY must be 0 or 1" >&2; exit 2 ;;
esac
if ! command -v nvidia-smi >/dev/null || ! command -v flock >/dev/null; then
  echo "error: local GPU execution requires nvidia-smi and flock" >&2
  exit 2
fi

mkdir -p "${JLENS_LOCAL_RUNTIME_ROOT}/gpu-locks"
IFS=',' read -r -a CANDIDATE_GPUS <<<"${GPU_IDS}"
START_SECONDS="${SECONDS}"

while true; do
  for gpu in "${CANDIDATE_GPUS[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
      echo "error: invalid GPU id in JLENS_LOCAL_GPU_IDS: ${gpu}" >&2
      exit 2
    fi
    LOCK_PATH="${JLENS_LOCAL_RUNTIME_ROOT}/gpu-locks/gpu_${gpu}.lock"
    exec 9>"${LOCK_PATH}"
    if ! flock -n 9; then
      exec 9>&-
      continue
    fi

    CAPACITY_OK=1
    for ((sample = 0; sample < STABILITY_SAMPLES; sample++)); do
      QUERY="$(nvidia-smi -i "${gpu}" \
        --query-gpu=memory.free,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null || true)"
      if [[ -z "${QUERY}" ]]; then
        CAPACITY_OK=0
        break
      fi
      IFS=',' read -r FREE_MIB UTILIZATION <<<"${QUERY}"
      FREE_MIB="${FREE_MIB//[[:space:]]/}"
      UTILIZATION="${UTILIZATION//[[:space:]]/}"
      if [[ ! "${FREE_MIB}" =~ ^[0-9]+$ || ! "${UTILIZATION}" =~ ^[0-9]+$ ]]; then
        CAPACITY_OK=0
        break
      fi
      if (( FREE_MIB < MIN_FREE_MIB || UTILIZATION > MAX_UTILIZATION )); then
        CAPACITY_OK=0
        break
      fi
      if (( sample + 1 < STABILITY_SAMPLES )); then
        sleep "${SAMPLE_INTERVAL}"
      fi
    done

    PROCESS_SNAPSHOT="$(nvidia-smi -i "${gpu}" \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ "${ALLOW_OVERLAY}" == "0" && -n "${PROCESS_SNAPSHOT}" ]]; then
      CAPACITY_OK=0
    fi
    if [[ "${CAPACITY_OK}" == "1" ]]; then
      OVERLAY=false
      [[ -n "${PROCESS_SNAPSHOT}" ]] && OVERLAY=true
      printf 'jlens_gpu_selected=%s free_mib=%s utilization=%s overlay=%s\n' \
        "${gpu}" "${FREE_MIB}" "${UTILIZATION}" "${OVERLAY}"
      if [[ -n "${PROCESS_SNAPSHOT}" ]]; then
        printf 'preexisting_compute_processes:\n%s\n' "${PROCESS_SNAPSHOT}"
      fi
      export CUDA_DEVICE_ORDER=PCI_BUS_ID
      export CUDA_VISIBLE_DEVICES="${gpu}"
      set +e
      "$@"
      STATUS="$?"
      set -e
      flock -u 9
      exec 9>&-
      exit "${STATUS}"
    fi
    flock -u 9
    exec 9>&-
  done

  if (( WAIT_TIMEOUT > 0 && SECONDS - START_SECONDS >= WAIT_TIMEOUT )); then
    echo "error: no GPU met J-lens capacity thresholds before timeout" >&2
    exit 3
  fi
  printf 'waiting_for_jlens_gpu_capacity min_free_mib=%s max_utilization=%s\n' \
    "${MIN_FREE_MIB}" "${MAX_UTILIZATION}" >&2
  sleep "${POLL_SECONDS}"
done
