#!/usr/bin/env bash
# ServerScheduler-only transport for the frozen three-method task catalog.
set -euo pipefail

if [[ "${JLENS_SERVER_SCHEDULER_TASK_ACTIVE:-0}" != "1" ]]; then
  echo "error: task must be launched by the ServerScheduler adapter" >&2
  exit 2
fi

TASK_NAME="${1:-}"
TASK_INDEX="${2:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

case "${TASK_NAME}" in
  candidate-rescore)
    if [[ ! "${TASK_INDEX}" =~ ^[0-2]$ ]]; then
      echo "error: candidate-rescore requires an index in [0, 2]" >&2
      exit 2
    fi
    ;;
  comparison-rescore-index)
    if [[ -n "${TASK_INDEX}" ]]; then
      echo "error: comparison-rescore-index does not take an index" >&2
      exit 2
    fi
    ;;
  *)
    # Sealed tasks fail in the Python adapter before this point. Delegation
    # preserves the historical task decoding for the read-only preflight.
    export JLENS_LOCAL_DAG_ACTIVE=1
    exec /usr/bin/bash "${SCRIPT_DIR}/run_three_method_local_task.sh" "$@"
    ;;
esac

source "${SCRIPT_DIR}/three_method_local_env.sh"

SHARED_ROOT="artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/shared_intervention_protocol"
RESCORE_ROOT="derivations/revision-r2/candidate_score_rescore_v1"
RESCORE_COMPARISON_ROOT="derivations/revision-r2/intervention_comparison"

case "${TASK_NAME}" in
  candidate-rescore)
    case "${TASK_INDEX}" in
      0)
        jlens intervention candidate-rescore "${J_CONFIG}" \
          --method j_component_intervention \
          --source-output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/j_component_intervention \
          --output "${RESCORE_ROOT}/j_component_intervention"
        ;;
      1)
        jlens intervention candidate-rescore "${ITI_CONFIG}" \
          --method iti_intervention \
          --source-output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/iti_intervention \
          --output "${RESCORE_ROOT}/iti_intervention"
        ;;
      2)
        jlens intervention candidate-rescore "${RAPTOR_CONFIG}" \
          --method raptor_intervention \
          --source-output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/raptor_intervention \
          --output "${RESCORE_ROOT}/raptor_intervention"
        ;;
    esac
    ;;
  comparison-rescore-index)
    jlens intervention compare-index \
      --shared-layer-selection "${SHARED_ROOT}/selection/layer_selection.json" \
      --j-output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/j_component_intervention \
      --iti-output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/iti_intervention \
      --raptor-output artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/raptor_intervention \
      --j-candidate-rescore "${RESCORE_ROOT}/j_component_intervention" \
      --iti-candidate-rescore "${RESCORE_ROOT}/iti_intervention" \
      --raptor-candidate-rescore "${RESCORE_ROOT}/raptor_intervention" \
      --output "${RESCORE_COMPARISON_ROOT}" \
      --concept-id goemotions:admiration \
      --concept-id goemotions:approval \
      --concept-id goemotions:curiosity \
      --concept-id goemotions:disapproval \
      --concept-id goemotions:gratitude \
      --concept-id goemotions:love \
      --concept-id goemotions:optimism
    ;;
esac
