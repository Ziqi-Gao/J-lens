#!/usr/bin/env bash
# Run the registered J-component/RAPTOR LLM-judge pilot without requesting a GPU.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_llm_judge_pilot_v9_local.sh [--dry-run|--pairwise-validation-only]

Runs the frozen validation -> calibration -> test -> review -> aggregation
protocol. The caller should place this script and src/ in an immutable runtime
snapshot and launch it through the checked-in local CPU/RAM lease broker.
EOF
}

DRY_RUN=0
RUN_MODE=full
case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  --pairwise-validation-only) RUN_MODE=pairwise_validation_only ;;
  -h|--help) usage; exit 0 ;;
  "") ;;
  *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(realpath -m -- "${SCRIPT_DIR}/../..")"
PYTHON="${JLENS_JUDGE_PYTHON:-/scr/del6500/J-lens/envs/three-method/bin/python}"
CONFIG="${JLENS_JUDGE_CONFIG:-${BUNDLE_ROOT}/Concept_intervention/configs/qwen35_4b_j_raptor_llm_judge_pilot_v9.yaml}"
SECRET_FILE="${JLENS_JUDGE_SECRET_FILE:-/scr/del6500/J-lens/runtime/llm-judge/secrets.env}"
RUNTIME_ROOT="${JLENS_JUDGE_RUNTIME_ROOT:-/scr/del6500/J-lens/runtime/llm-judge/v9}"
OUTPUT_ROOT="/data/del6500/J-lens/runs/qwen35_4b_three_method_intervention_v1/artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/llm_judge_pilot_v9"
RUN_LOCK="${RUNTIME_ROOT}/full-run.lock"
STATUS_FILE="${RUNTIME_ROOT}/full-run.status"

require_under() {
  local label="$1" candidate="$2" allowed="$3" resolved
  resolved="$(realpath -m -- "${candidate}")"
  case "${resolved}" in
    "${allowed}"|"${allowed}"/*) ;;
    *) echo "error: ${label} must remain under ${allowed}: ${resolved}" >&2; exit 2 ;;
  esac
}

require_under BUNDLE_ROOT "${BUNDLE_ROOT}" /scr/del6500/J-lens
require_under RUNTIME_ROOT "${RUNTIME_ROOT}" /scr/del6500/J-lens
require_under OUTPUT_ROOT "${OUTPUT_ROOT}" /data/del6500/J-lens
if [[ ! -x "${PYTHON}" ]]; then
  echo "error: judge Python is unavailable: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" || -L "${CONFIG}" ]]; then
  echo "error: registered config must be a plain file: ${CONFIG}" >&2
  exit 2
fi

export PYTHONPATH="${BUNDLE_ROOT}/src"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=42
export PYTHONDONTWRITEBYTECODE=1
umask 077
cd -- "${BUNDLE_ROOT}"

print_plan() {
  cat <<EOF
bundle_root=${BUNDLE_ROOT}
run_mode=${RUN_MODE}
config=${CONFIG}
output_root=${OUTPUT_ROOT}
gpu_devices=disabled
base_batch_limit=10
expert_batch_limit=5
jobs=2
validation=primary,secondary x pointwise,pairwise
gate=calibration
test=primary,secondary x pointwise,pairwise
review=arbitration,expert_review
final=human_audit,aggregate
EOF
}
if [[ "${DRY_RUN}" == "1" ]]; then
  print_plan
  exit 0
fi

if [[ ! -f "${SECRET_FILE}" || -L "${SECRET_FILE}" ]]; then
  echo "error: judge secret file must be a plain file: ${SECRET_FILE}" >&2
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
  echo "error: another full LLM-judge controller holds ${RUN_LOCK}" >&2
  exit 3
fi

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

from jlens_workspace.concept_intervention.judge.prompts import rubric_hash
from jlens_workspace.concept_intervention.judge.workflow import _canonical_hash

config_path = Path(sys.argv[1])
root = Path(sys.argv[2])
manifest_path = root / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit("prepared manifest is missing")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
if manifest.get("config_sha256") != config_hash:
    raise SystemExit("prepared manifest config hash differs from runtime config")
if manifest.get("rubric_sha256") != rubric_hash():
    raise SystemExit("prepared manifest rubric hash differs from runtime rubric")
implementation_root = Path(__import__(
    "jlens_workspace.concept_intervention.judge", fromlist=["__file__"]
).__file__).parent
implementation_files = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(implementation_root.glob("*.py"))
}
if manifest.get("implementation", {}).get("combined_sha256") != _canonical_hash(
    implementation_files
):
    raise SystemExit("prepared manifest implementation hash differs from runtime code")
for record in manifest.get("task_artifacts", {}).values():
    path = Path(record["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise SystemExit(f"prepared task hash changed: {path}")
print("judge_preflight_hashes_valid=true", flush=True)
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
    primary) printf '%s\n' google/gemini-3.5-flash ;;
    secondary) printf '%s\n' anthropic/claude-haiku-4.5 ;;
    arbitration) printf '%s\n' google/gemini-3.6-flash ;;
    expert_review) printf '%s\n' anthropic/claude-opus-5 ;;
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
      echo "judge_set_complete role=${role} task_set=${task_set} split=${split:-none} completed=${before}"
      return 0
    fi
    iterations=$((iterations + 1))
    if [[ "${iterations}" -gt 500 ]]; then
      echo "error: judge set exceeded 500 bounded batches: ${result_set}" >&2
      return 2
    fi
    command=(judge run --config "${CONFIG}" --role "${role}" --task-set "${task_set}" \
      --limit "${limit}" --jobs "${jobs}" --json)
    if [[ -n "${split}" ]]; then
      command+=(--split "${split}")
    fi
    set +e
    jlens "${command[@]}"
    return_code=$?
    set -e
    read -r after complete < <(index_state "${index_path}")
    if [[ "${complete}" == "true" ]]; then
      echo "judge_set_complete role=${role} task_set=${task_set} split=${split:-none} completed=${after}"
      return 0
    fi
    if [[ "${after}" -gt "${before}" ]]; then
      stagnant=0
    else
      stagnant=$((stagnant + 1))
    fi
    if [[ "${stagnant}" -ge 3 ]]; then
      echo "error: judge set made no progress in three consecutive batches: ${result_set}" >&2
      return 2
    fi
    if [[ "${return_code}" -ne 0 ]]; then
      echo "judge_batch_retry role=${role} task_set=${task_set} split=${split:-none} rc=${return_code} stagnant=${stagnant}" >&2
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
if [[ "${RUN_MODE}" == "pairwise_validation_only" ]]; then
  for role in primary secondary; do
    run_until_complete "${role}" pairwise validation 10 2
  done
  trap - EXIT
  CURRENT_STAGE=pairwise_validation_complete
  write_status partial_complete "${CURRENT_STAGE}"
  echo "judge_pairwise_validation_complete=true"
  exit 0
fi

for role in primary secondary; do
  run_until_complete "${role}" pointwise validation 10 2
  run_until_complete "${role}" pairwise validation 10 2
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
  run_until_complete "${role}" pointwise test 10 2
  run_until_complete "${role}" pairwise test 10 2
done

CURRENT_STAGE=arbitration
write_status running "${CURRENT_STAGE}"
jlens judge review-prepare --config "${CONFIG}" --tier arbitration --json
run_until_complete arbitration arbitration "" 10 2

CURRENT_STAGE=expert_review
write_status running "${CURRENT_STAGE}"
jlens judge review-prepare --config "${CONFIG}" --tier expert_review --json
run_until_complete expert_review expert_review "" 5 2

CURRENT_STAGE=audit
write_status running "${CURRENT_STAGE}"
jlens judge audit-export --config "${CONFIG}" --json

CURRENT_STAGE=analysis
write_status running "${CURRENT_STAGE}"
jlens judge aggregate --config "${CONFIG}" --json

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index_path = root / "analysis" / "index.json"
if not index_path.is_file():
    raise SystemExit("analysis/index.json is missing after aggregation")
payload = json.loads(index_path.read_text(encoding="utf-8"))
if payload.get("complete") is not True:
    raise SystemExit("analysis/index.json does not report complete=true")
print("judge_analysis_complete=true", flush=True)
PY

trap - EXIT
CURRENT_STAGE=complete
write_status complete "${CURRENT_STAGE}"
echo "judge_pipeline_complete=true"
