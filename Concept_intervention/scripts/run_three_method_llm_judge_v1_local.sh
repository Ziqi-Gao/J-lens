#!/usr/bin/env bash
# Persistent CPU/API controller for the frozen three-method LLM-judge study.
set -euo pipefail

readonly BASE_BATCH_LIMIT=16

usage() {
  cat <<'EOF'
Usage: run_three_method_llm_judge_v1_local.sh [--dry-run]

Dry-run is local and secret-free. A real run is accepted only from an immutable
snapshot under the dedicated three-method-v1 runtime and only after the exact
config/manifest-bound formal budget approval file exists.
EOF
}

DRY_RUN=0
case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  -h|--help) usage; exit 0 ;;
  "") ;;
  *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(realpath -m -- "${SCRIPT_DIR}/../..")"
PYTHON="${JLENS_JUDGE_PYTHON:-/scr/del6500/J-lens/envs/three-method/bin/python}"
CONFIG="${JLENS_JUDGE_CONFIG:-${BUNDLE_ROOT}/Concept_intervention/configs/qwen35_4b_three_method_llm_judge_v1.yaml}"
SECRET_FILE="${JLENS_JUDGE_SECRET_FILE:-/scr/del6500/J-lens/runtime/llm-judge/secrets.env}"
RUNTIME_ROOT="/scr/del6500/J-lens/runtime/llm-judge/three-method-v1"
APPROVAL_FILE="${RUNTIME_ROOT}/formal_budget_approval.json"
OUTPUT_ROOT="/data/del6500/J-lens/runs/qwen35_4b_three_method_intervention_v1/artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/llm_judge_three_method_v1"
RUN_LOCK="${RUNTIME_ROOT}/controller.lock"
STATUS_FILE="${RUNTIME_ROOT}/controller.status"

export PYTHONPATH="${BUNDLE_ROOT}/src"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=42
export PYTHONDONTWRITEBYTECODE=1
umask 077

print_plan() {
  cat <<EOF
experiment=qwen35_4b_three_method_llm_judge_v1
bundle_root=${BUNDLE_ROOT}
config=${CONFIG}
output_root=${OUTPUT_ROOT}
runtime_root=${RUNTIME_ROOT}
gpu_devices=disabled
base_batch_limit=${BASE_BATCH_LIMIT}
expert_batch_limit=1
jobs=2
validation=2 judges x (4480 pointwise + 2688 pairwise)
calibration=fail-closed before test
test=2 judges x (4480 pointwise + 2688 pairwise)
review=selective arbitration then expert review
final=human audit export then three-method aggregate
formal_budget_approval=${APPROVAL_FILE}
EOF
}

if [[ "${DRY_RUN}" == "1" ]]; then
  print_plan
  exit 0
fi

case "${BUNDLE_ROOT}" in
  "${RUNTIME_ROOT}/snapshots/"*) ;;
  *)
    echo "error: full controller must run from an immutable runtime snapshot" >&2
    exit 2
    ;;
esac
if [[ ! -x "${PYTHON}" ]]; then
  echo "error: judge Python is unavailable: ${PYTHON}" >&2
  exit 2
fi
for path in "${CONFIG}" "${SECRET_FILE}" "${APPROVAL_FILE}"; do
  if [[ ! -f "${path}" || -L "${path}" ]]; then
    echo "error: required plain file is unavailable: ${path}" >&2
    exit 2
  fi
done
if [[ "$(stat -c '%a' "${APPROVAL_FILE}")" != "600" ]]; then
  echo "error: formal budget approval must have mode 0600" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${SECRET_FILE}"
set +a
if [[ -z "${JLENS_JUDGE_API_KEY:-}" ]]; then
  echo "error: JLENS_JUDGE_API_KEY is not set" >&2
  exit 2
fi

mkdir -p -- "${RUNTIME_ROOT}"
exec 9>"${RUN_LOCK}"
if ! flock -n 9; then
  echo "error: another three-method judge controller holds ${RUN_LOCK}" >&2
  exit 3
fi
cd -- "${BUNDLE_ROOT}"

CURRENT_STAGE=preflight
write_status() {
  local state="$1" stage="$2" temporary
  temporary="${STATUS_FILE}.$$"
  printf 'state=%s\nstage=%s\npid=%s\nupdated_at=%s\nbundle_root=%s\n' \
    "${state}" "${stage}" "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${BUNDLE_ROOT}" >"${temporary}"
  mv -- "${temporary}" "${STATUS_FILE}"
}
finish_trap() {
  local code=$?
  trap - EXIT
  if [[ "${code}" -ne 0 ]]; then
    write_status failed "${CURRENT_STAGE}"
  fi
  exit "${code}"
}
trap finish_trap EXIT
write_status running "${CURRENT_STAGE}"

"${PYTHON}" - "${CONFIG}" "${OUTPUT_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from jlens_workspace.concept_intervention.judge.amendment import (
    validate_format_amendment,
)

config_path = Path(sys.argv[1])
root = Path(sys.argv[2])
manifest_path = root / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit("prepared manifest is missing")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
if manifest.get("config_sha256") != digest:
    raise SystemExit("runtime config differs from prepared registration")
frozen = Path(manifest["frozen_config"])
if hashlib.sha256(frozen.read_bytes()).hexdigest() != digest:
    raise SystemExit("frozen config seal mismatch")
amendment = validate_format_amendment(
    config_path,
    root=root,
    experiment_name=manifest["experiment_name"],
    verify_response_seals=True,
)
for record in manifest.get("task_artifacts", {}).values():
    for path_key, hash_key in (
        ("path", "sha256"),
        ("private_map", "private_map_sha256"),
    ):
        path = Path(record[path_key])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record[hash_key]:
            raise SystemExit(f"prepared task seal mismatch: {path}")
print(
    f"three_method_judge_preflight=true amendment={amendment['amendment_id']}",
    flush=True,
)
PY

jlens() {
  "${PYTHON}" -m jlens_workspace.cli "$@"
}
index_state() {
  local index_path="$1"
  "${PYTHON}" - "${index_path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("0 false")
else:
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(int(payload.get("completed_tasks", 0)), str(payload.get("complete") is True).lower())
PY
}
model_for_role() {
  case "$1" in
    primary) printf '%s\n' mistralai/mistral-small-2603 ;;
    secondary) printf '%s\n' anthropic/claude-haiku-4.5 ;;
    arbitration) printf '%s\n' google/gemini-3.6-flash ;;
    expert_review) printf '%s\n' anthropic/claude-sonnet-5 ;;
    *) echo "error: unsupported judge role: $1" >&2; return 2 ;;
  esac
}
run_until_complete() {
  local role="$1" task_set="$2" split="$3" limit="$4" jobs="$5"
  local model model_dir result_set index_path before after complete
  local return_code=0 stagnant=0 iterations=0
  local -a command
  model="$(model_for_role "${role}")"
  model_dir="${model//\//__}"
  result_set="${task_set}"
  if [[ -n "${split}" ]]; then
    result_set="${task_set}_${split}"
  fi
  index_path="${OUTPUT_ROOT}/responses/${model_dir}/${result_set}/index.json"
  while true; do
    read -r before complete < <(index_state "${index_path}")
    if [[ "${complete}" == "true" ]]; then
      echo "judge_set_complete role=${role} set=${result_set} completed=${before}"
      return 0
    fi
    iterations=$((iterations + 1))
    if [[ "${iterations}" -gt 600 ]]; then
      echo "error: judge set exceeded 600 bounded batches: ${result_set}" >&2
      return 2
    fi
    command=(judge run --config "${CONFIG}" --role "${role}" \
      --task-set "${task_set}" --limit "${limit}" --jobs "${jobs}" --json)
    if [[ -n "${split}" ]]; then
      command+=(--split "${split}")
    fi
    set +e
    jlens "${command[@]}"
    return_code=$?
    set -e
    read -r after complete < <(index_state "${index_path}")
    if [[ "${complete}" == "true" ]]; then
      echo "judge_set_complete role=${role} set=${result_set} completed=${after}"
      return 0
    fi
    if [[ "${after}" -gt "${before}" ]]; then
      stagnant=0
    else
      stagnant=$((stagnant + 1))
    fi
    if [[ "${stagnant}" -ge 3 ]]; then
      echo "error: no progress in three consecutive batches: ${result_set}" >&2
      return 2
    fi
    if [[ "${return_code}" -ne 0 ]]; then
      sleep 30
    else
      sleep 1
    fi
  done
}

CURRENT_STAGE=validate
write_status running "${CURRENT_STAGE}"
jlens judge validate --config "${CONFIG}" --json

CURRENT_STAGE=validation
write_status running "${CURRENT_STAGE}"
for role in primary secondary; do
  run_until_complete "${role}" pointwise validation "${BASE_BATCH_LIMIT}" 2
  run_until_complete "${role}" pairwise validation "${BASE_BATCH_LIMIT}" 2
done

CURRENT_STAGE=calibration
write_status running "${CURRENT_STAGE}"
set +e
jlens judge calibrate --config "${CONFIG}" --json
CALIBRATION_CODE=$?
set -e
if [[ "${CALIBRATION_CODE}" -ne 0 ]]; then
  trap - EXIT
  write_status blocked calibration
  echo "judge_pipeline_blocked=calibration" >&2
  exit "${CALIBRATION_CODE}"
fi

CURRENT_STAGE=test
write_status running "${CURRENT_STAGE}"
for role in primary secondary; do
  run_until_complete "${role}" pointwise test "${BASE_BATCH_LIMIT}" 2
  run_until_complete "${role}" pairwise test "${BASE_BATCH_LIMIT}" 2
done

CURRENT_STAGE=arbitration
write_status running "${CURRENT_STAGE}"
jlens judge review-prepare --config "${CONFIG}" --tier arbitration --json
run_until_complete arbitration arbitration "" "${BASE_BATCH_LIMIT}" 2

CURRENT_STAGE=expert_review
write_status running "${CURRENT_STAGE}"
jlens judge review-prepare --config "${CONFIG}" --tier expert_review --json
run_until_complete expert_review expert_review "" 1 1

CURRENT_STAGE=audit
write_status running "${CURRENT_STAGE}"
jlens judge audit-export --config "${CONFIG}" --json

CURRENT_STAGE=analysis
write_status running "${CURRENT_STAGE}"
jlens judge aggregate --config "${CONFIG}" --json

trap - EXIT
CURRENT_STAGE=complete
write_status complete "${CURRENT_STAGE}"
echo "three_method_judge_pipeline_complete=true"
