#!/usr/bin/env bash
# Exact local equivalents of the registered Slurm task bodies.
set -euo pipefail

DECODE_ONLY=0
if [[ "${1:-}" == "--decode" ]]; then
  DECODE_ONLY=1
  shift
fi
TASK_NAME="${1:-}"
TASK_INDEX="${2:-}"

CONCEPTS=(
  goemotions:admiration goemotions:approval goemotions:curiosity
  goemotions:disapproval goemotions:gratitude goemotions:love
  goemotions:optimism
)
LAYERS=(3 7 11 15 19 23 27)
REPLICATES=(primary bootstrap_1101 bootstrap_2202 bootstrap_3303 bootstrap_4404)

require_index() {
  local maximum="$1"
  if [[ ! "${TASK_INDEX}" =~ ^[0-9]+$ ]] || (( TASK_INDEX > maximum )); then
    echo "error: ${TASK_NAME} requires an index in [0, ${maximum}]" >&2
    exit 2
  fi
}

CONCEPT=""
GRID_INDEX=""
LAYER=""
REPLICATE=""
case "${TASK_NAME}" in
  preflight|lens|capture|layer-selection|bootstrap-index|occupancy-index|iti-capture|iti-fit|smoke-check|comparison-index)
    if [[ -n "${TASK_INDEX}" ]]; then
      echo "error: ${TASK_NAME} does not take an index" >&2
      exit 2
    fi
    ;;
  bootstrap-probes)
    require_index 6
    LAYER="${LAYERS[${TASK_INDEX}]}"
    ;;
  occupancy)
    require_index 34
    LAYER="${LAYERS[$((TASK_INDEX % 7))]}"
    REPLICATE="${REPLICATES[$((TASK_INDEX / 7))]}"
    ;;
  j-grid)
    require_index 391
    CONCEPT="${CONCEPTS[$((TASK_INDEX / 56))]}"
    GRID_INDEX="$((TASK_INDEX % 56))"
    ;;
  raptor-grid)
    require_index 62
    CONCEPT="${CONCEPTS[$((TASK_INDEX / 9))]}"
    GRID_INDEX="$((TASK_INDEX % 9))"
    ;;
  iti-grid)
    require_index 3086
    CONCEPT="${CONCEPTS[$((TASK_INDEX / 441))]}"
    GRID_INDEX="$((TASK_INDEX % 441))"
    ;;
  method-index)
    require_index 2
    ;;
  *)
    echo "error: unknown local task: ${TASK_NAME}" >&2
    exit 2
    ;;
esac

if [[ "${DECODE_ONLY}" == "1" ]]; then
  printf 'task=%s\n' "${TASK_NAME}"
  [[ -n "${TASK_INDEX}" ]] && printf 'task_index=%s\n' "${TASK_INDEX}"
  [[ -n "${CONCEPT}" ]] && printf 'concept_id=%s\n' "${CONCEPT}"
  [[ -n "${GRID_INDEX}" ]] && printf 'grid_index=%s\n' "${GRID_INDEX}"
  [[ -n "${LAYER}" ]] && printf 'layer=%s\n' "${LAYER}"
  [[ -n "${REPLICATE}" ]] && printf 'replicate_id=%s\n' "${REPLICATE}"
  exit 0
fi
if [[ "${JLENS_LOCAL_DAG_ACTIVE:-0}" != "1" ]]; then
  echo "error: component tasks must be launched by run_three_method_local.sh" >&2
  exit 2
fi

case "${TASK_NAME}" in
  layer-selection|iti-fit) JLENS_CPUS_PER_TASK="${JLENS_CPUS_PER_TASK:-16}" ;;
  bootstrap-probes) JLENS_CPUS_PER_TASK="${JLENS_CPUS_PER_TASK:-8}" ;;
  *) JLENS_CPUS_PER_TASK="${JLENS_CPUS_PER_TASK:-4}" ;;
esac
export JLENS_CPUS_PER_TASK

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/three_method_local_env.sh"

SHARED_ROOT="artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/shared_intervention_protocol"
COMPARISON_ROOT="artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/intervention_comparison"

case "${TASK_NAME}" in
  preflight)
    jlens doctor --require-llm
    for config in "${SHARED_CONFIG}" "${J_CONFIG}" "${ITI_CONFIG}" "${RAPTOR_CONFIG}"; do
      jlens config validate "${config}"
    done
    jlens data validate --config "${SHARED_CONFIG}"
    "${PYTHON}" -c "from jlens_workspace.concept_intervention.raptor import verify_raptor_adaptive_epsilon_parity, verify_raptor_checkout; root=verify_raptor_checkout('third_party_external/RAPTOR'); verify_raptor_adaptive_epsilon_parity(root)"
    ;;
  lens) jlens lens fit "${SHARED_CONFIG}" ;;
  capture) jlens concept capture "${SHARED_CONFIG}" ;;
  layer-selection) jlens concept layer-select "${SHARED_CONFIG}" ;;
  bootstrap-probes)
    jlens concept bootstrap-probes "${SHARED_CONFIG}" \
      --activations "${SHARED_ROOT}/activations" \
      --primary-probes "${SHARED_ROOT}/selection/raptor_probes" \
      --output "${SHARED_ROOT}/selection/probe_replicates" \
      --layer "${LAYER}"
    ;;
  bootstrap-index)
    jlens concept bootstrap-probes-index "${SHARED_CONFIG}" \
      --activations "${SHARED_ROOT}/activations" \
      --primary-probes "${SHARED_ROOT}/selection/raptor_probes" \
      --output "${SHARED_ROOT}/selection/probe_replicates"
    ;;
  occupancy)
    OVERWRITE_ARGS=()
    case "${OCCUPANCY_OVERWRITE:-0}" in
      0) ;;
      1) OVERWRITE_ARGS+=(--overwrite) ;;
      *) echo "error: OCCUPANCY_OVERWRITE must be 0 or 1" >&2; exit 2 ;;
    esac
    jlens occupancy concepts "${SHARED_CONFIG}" \
      --probes "${SHARED_ROOT}/selection/raptor_probes" \
      --probe-replicates "${SHARED_ROOT}/selection/probe_replicates" \
      --output "${SHARED_ROOT}/j_occupancy" \
      --layer "${LAYER}" --convention rmsnorm_weighted \
      --replicate-id "${REPLICATE}" --k-max 64 "${OVERWRITE_ARGS[@]}"
    ;;
  occupancy-index)
    jlens occupancy index "${SHARED_CONFIG}" --output "${SHARED_ROOT}/j_occupancy"
    ;;
  iti-capture) jlens iti capture "${ITI_CONFIG}" ;;
  iti-fit) jlens iti fit "${ITI_CONFIG}" ;;
  j-grid)
    jlens intervention j-component "${J_CONFIG}" \
      --concept-id "${CONCEPT}" --grid-index "${GRID_INDEX}"
    ;;
  raptor-grid)
    jlens intervention raptor "${RAPTOR_CONFIG}" \
      --concept-id "${CONCEPT}" --grid-index "${GRID_INDEX}"
    ;;
  iti-grid)
    jlens iti run "${ITI_CONFIG}" \
      --concept-id "${CONCEPT}" --grid-index "${GRID_INDEX}"
    ;;
  smoke-check)
    jlens intervention smoke-check \
      --j-output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/j_component_intervention \
      --iti-output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/iti_intervention \
      --raptor-output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/raptor_intervention \
      --output "${COMPARISON_ROOT}/smoke_gate.json"
    ;;
  method-index)
    case "${TASK_INDEX}" in
      0) jlens intervention j-component-index "${J_CONFIG}" ;;
      1) jlens iti index "${ITI_CONFIG}" ;;
      2) jlens intervention raptor-index "${RAPTOR_CONFIG}" ;;
    esac
    ;;
  comparison-index)
    jlens intervention compare-index \
      --shared-layer-selection "${SHARED_ROOT}/selection/layer_selection.json" \
      --j-output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/j_component_intervention \
      --iti-output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/iti_intervention \
      --raptor-output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/raptor_intervention \
      --output "${COMPARISON_ROOT}" \
      --concept-id goemotions:admiration \
      --concept-id goemotions:approval \
      --concept-id goemotions:curiosity \
      --concept-id goemotions:disapproval \
      --concept-id goemotions:gratitude \
      --concept-id goemotions:love \
      --concept-id goemotions:optimism
    ;;
esac
