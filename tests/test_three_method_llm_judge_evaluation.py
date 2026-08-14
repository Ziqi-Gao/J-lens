from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from jlens_workspace.concept_intervention.judge.client import OpenRouterClient
from jlens_workspace.concept_intervention.judge.three_method_aggregation import (
    PRIMARY_CONTRASTS,
    _pairwise_analysis,
    _pointwise_analysis,
)
from jlens_workspace.concept_intervention.judge.three_method_config import (
    JudgeEvaluationConfig,
    OpenRouterConfig,
)
from jlens_workspace.concept_intervention.judge.three_method_prepare import (
    METHOD_PAIRS,
    PRIMARY_ROLES,
    SECONDARY_ROLES,
)
from jlens_workspace.concept_intervention.judge.workflow import (
    JudgeWorkflowError,
    run_judge_tasks,
)


def _config_payload(tmp_path: Path) -> dict[str, Any]:
    models = {
        "primary": "mistralai/mistral-small-2603",
        "secondary": "anthropic/claude-haiku-4.5",
        "arbitration": "google/gemini-3.6-flash",
        "expert_review": "anthropic/claude-sonnet-5",
    }
    return {
        "schema_version": 1,
        "protocol_version": "concept_intervention_llm_judge_v3",
        "study_design_version": "three_method_llm_judge_v1",
        "experiment_name": "qwen35_4b_three_method_llm_judge_v1",
        "output_dir": str(tmp_path / "llm_judge_three_method_v1"),
        "seed": 42,
        "source": {
            "comparison_index": str(tmp_path / "comparison-index.json"),
            "comparison_index_sha256": "a" * 64,
            "comparison_sha256": "b" * 64,
            "j_component_root": str(tmp_path / "j"),
            "raptor_root": str(tmp_path / "raptor"),
            "iti_root": str(tmp_path / "iti"),
            "candidate_rescore_root": str(tmp_path / "rescore"),
            "concept_definitions_path": str(tmp_path / "concepts.json"),
            "concept_ids": ["goemotions:optimism"],
        },
        "selection": {
            "tuning_split": "validation",
            "evaluation_splits": ["validation", "test"],
            "prompt_family": "open",
            "decodings": ["greedy", "sample"],
            "sample_seeds": [1001, 2002, 3003],
            "j_positive_strengths_only": True,
            "raptor_probability_above": 0.5,
            "matched_random_seed": 101,
        },
        "judges": models,
        "openrouter": {
            "max_retries": 1,
            "max_tokens": 400,
            "temperature": 0.0,
            "provider_order_by_model": {
                "anthropic/claude-sonnet-5": ["provider-a", "provider-b"],
            },
        },
        "budget": {
            "formal_budget_approval_file": str(tmp_path / "approval.json"),
            "max_tasks_per_invocation": 25,
            "max_smoke_tasks_per_invocation": 1,
            "max_completion_tokens_per_request": 1024,
            "max_estimated_total_tokens_per_invocation": 275000,
            "max_cost_usd_per_invocation": 2.0,
            "max_experiment_cost_usd": 10.0,
            "reserve_openrouter_credit_usd": 0.0,
        },
        "review": {
            "expert_provider_filter_escalation": True,
            "human_audit_per_concept": 1,
        },
        "analysis": {
            "bootstrap_samples": 1000,
            "permutation_samples": 1000,
            "primary_contrasts": list(PRIMARY_CONTRASTS),
            "descriptive_only": True,
        },
        "pricing": {
            "input_per_million": {model: 1.0 for model in models.values()},
            "output_per_million": {model: 2.0 for model in models.values()},
        },
    }


def _public_point_task() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rubric_version": "concept_intervention_judge_rubric_v4",
        "task_id": "a" * 64,
        "task_type": "pointwise",
        "split": "validation",
        "concept_id": "goemotions:optimism",
        "concept_name": "optimism",
        "concept_definition": "Positive expectations about the future.",
        "user_prompt": "What happens next?",
        "response": "I expect a good outcome.",
    }


def test_three_method_config_is_fixed_and_isolated(tmp_path: Path) -> None:
    config = JudgeEvaluationConfig.model_validate(_config_payload(tmp_path))
    assert config.analysis.primary_contrasts == list(PRIMARY_CONTRASTS)
    assert config.selection.sample_seeds == [1001, 2002, 3003]
    assert Path(config.output_dir).name == "llm_judge_three_method_v1"
    payload = _config_payload(tmp_path)
    payload["analysis"]["primary_contrasts"] = ["j_minus_raptor"]
    with pytest.raises(ValueError, match="primary_contrasts"):
        JudgeEvaluationConfig.model_validate(payload)


def test_registered_config_uses_cost_bounded_independent_judges() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "Concept_intervention/configs/qwen35_4b_three_method_llm_judge_v1.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert payload["judges"] == {
        "primary": "mistralai/mistral-small-2603",
        "secondary": "anthropic/claude-haiku-4.5",
        "arbitration": "google/gemini-3.6-flash",
        "expert_review": "anthropic/claude-sonnet-5",
    }
    assert payload["openrouter"]["provider_order_by_model"][
        "mistralai/mistral-small-2603"
    ] == ["mistral"]
    assert payload["budget"]["max_experiment_cost_usd"] == 50.0
    assert payload["budget"]["reserve_openrouter_credit_usd"] == 25.0
    assert payload["pricing"]["input_per_million"][
        "mistralai/mistral-small-2603"
    ] == 0.15
    assert payload["pricing"]["output_per_million"][
        "mistralai/mistral-small-2603"
    ] == 0.60


def test_client_retains_secret_safe_raw_audit() -> None:
    observed_headers: dict[str, str] = {}

    def transport(
        endpoint: str,
        headers: dict[str, str],
        body: bytes,
        timeout: float,
    ) -> dict[str, Any]:
        del endpoint, timeout
        observed_headers.update(headers)
        request = json.loads(body)
        assert "Authorization" not in request
        return {
            "id": "response-1",
            "model": request["model"],
            "provider": "provider-a",
            "created": 1,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "target_expression": 80,
                                "coherence": 90,
                                "relevance": 90,
                                "refusal": False,
                                "invalid": False,
                                "evidence": ["expect a good outcome"],
                                "rationale": "Clear optimism.",
                            }
                        )
                    },
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "cost": 0.001},
        }

    config = OpenRouterConfig(
        max_retries=0,
        provider_order_by_model={"mistralai/mistral-small-2603": ["provider-a"]},
    )
    client = OpenRouterClient(config, transport=transport)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("JLENS_JUDGE_API_KEY", "unit-test-secret")
        response = client.judge(
            model="mistralai/mistral-small-2603",
            task=_public_point_task(),
        )
    assert observed_headers["Authorization"] == "Bearer unit-test-secret"
    assert response.raw_provider_response["id"] == "response-1"
    assert response.request_payload["provider"]["order"] == ["provider-a"]
    assert response.attempt_records[0]["outcome"] == "validated"
    assert response.request_started_at
    assert response.response_received_at
    assert "unit-test-secret" not in json.dumps(response.request_payload)


def test_pointwise_has_all_three_prompt_clustered_primary_contrasts(
    tmp_path: Path,
) -> None:
    config = JudgeEvaluationConfig.model_validate(_config_payload(tmp_path))
    role_scores = {
        "common_baseline": 10,
        "j_component": 70,
        "raptor": 50,
        "iti_native": 40,
        "j_full": 60,
        "j_non_j": 20,
        "j_matched_random": 15,
        "iti_layer_matched": 35,
        "iti_probe_weight": 30,
        "iti_matched_random": 12,
    }
    rows: list[dict[str, Any]] = []
    for role in (*PRIMARY_ROLES, *SECONDARY_ROLES):
        for prompt_index in range(2):
            for decoding, seed in (("greedy", None), ("sample", 1001)):
                rows.append(
                    {
                        "concept_id": "goemotions:optimism",
                        "condition_role": role,
                        "prompt_id": f"open_{prompt_index}",
                        "decoding": decoding,
                        "seed": seed,
                        "target_expression": role_scores[role],
                        "quality": 90.0,
                        "refusal": 0.0,
                    }
                )
    concepts, macro = _pointwise_analysis(rows, config=config)
    assert set(concepts["goemotions:optimism"]).issuperset(PRIMARY_CONTRASTS)
    assert macro["j_minus_raptor"]["target_expression"]["estimate"] == 20.0
    assert macro["j_minus_iti_native"]["target_expression"]["estimate"] == 30.0
    assert macro["raptor_minus_iti_native"]["target_expression"]["estimate"] == 10.0


def test_pairwise_covers_three_pairs_and_both_orders() -> None:
    private: list[dict[str, Any]] = []
    results: dict[str, dict[str, dict[str, Any]]] = {
        "primary": {},
        "secondary": {},
    }
    for pair_index, (left, right) in enumerate(METHOD_PAIRS):
        method_pair = f"{left}_vs_{right}"
        for order in range(2):
            task_id = f"pair-{pair_index}-{order}"
            left_first = order == 0
            private.append(
                {
                    "task_id": task_id,
                    "pair_group_id": f"group-{pair_index}",
                    "method_pair": method_pair,
                    "concept_id": "goemotions:optimism",
                    "prompt_id": f"open_{pair_index}",
                    "decoding": "greedy",
                    "seed": None,
                    "response_a_role": left if left_first else right,
                    "response_b_role": right if left_first else left,
                }
            )
            judgment = {
                "target_preference": "A" if left_first else "B",
                "quality_preference": "tie",
            }
            for role in results:
                results[role][task_id] = {"judgment": judgment}
    rows, summary = _pairwise_analysis(private, results)
    assert len(rows) == 3
    assert set(summary["by_method_pair"]) == {
        f"{left}_vs_{right}" for left, right in METHOD_PAIRS
    }
    assert summary["position_bias_rate"] == {"primary": 0.0, "secondary": 0.0}
    assert summary["cross_judge_agreement_among_stable"] == 1.0


def test_formal_requests_locked_but_smoke_scope_is_narrow(tmp_path: Path) -> None:
    payload = _config_payload(tmp_path)
    config_path = tmp_path / "judge.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    root = Path(payload["output_dir"])
    root.mkdir(parents=True)
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(JudgeWorkflowError, match="formal remote judging is locked"):
        run_judge_tasks(
            config_path,
            role="primary",
            task_set="pointwise",
            split="validation",
            limit=1,
        )
    with pytest.raises(JudgeWorkflowError, match="authorized smoke scope"):
        run_judge_tasks(
            config_path,
            role="primary",
            task_set="pairwise",
            split="validation",
            limit=1,
            smoke=True,
        )
