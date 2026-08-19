from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest
import yaml

from jlens_workspace import cli
from jlens_workspace.concept_intervention.judge.client import (
    JudgeResponse,
    OpenRouterClient,
    OpenRouterContentFilterError,
    OpenRouterError,
)
from jlens_workspace.concept_intervention.judge.config import (
    JudgeEvaluationConfig,
    OpenRouterConfig,
)
from jlens_workspace.concept_intervention.judge.prompts import (
    render_messages,
    response_schema,
    rubric_hash,
    validate_judgment,
)
from jlens_workspace.concept_intervention.judge.statistics import (
    agreement_metrics,
    benjamini_hochberg,
    paired_cluster_summary,
)
from jlens_workspace.concept_intervention.judge.workflow import (
    JudgeWorkflowError,
    _canonical_hash,
    _result_directory,
    _select_registered_conditions,
    _Shard,
    aggregate_evaluation,
    calibrate_judges,
    export_human_audit,
    prepare_review_tasks,
    run_judge_tasks,
)


def _config_payload(tmp_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_version": "concept_intervention_llm_judge_v1",
        "experiment_name": "judge-test",
        "output_dir": str(tmp_path / "judge"),
        "seed": 42,
        "source": {
            "j_component_root": str(tmp_path / "j"),
            "raptor_root": str(tmp_path / "raptor"),
            "concept_definitions_path": str(tmp_path / "definitions.json"),
            "concept_ids": ["goemotions:optimism"],
        },
        "selection": {
            "evaluation_splits": ["validation", "test"],
            "decodings": ["greedy"],
            "include_j_controls": ["full", "j", "non_j", "random"],
            "random_control_seeds": [101],
        },
        "judges": {
            "primary": "anthropic/claude-sonnet-5",
            "secondary": "google/gemini-3.5-flash",
            "arbitration": "x-ai/grok-4.5",
            "expert_review": "anthropic/claude-opus-5",
        },
        "pricing": {
            "input_per_million": {
                "anthropic/claude-sonnet-5": 2.0,
                "google/gemini-3.5-flash": 1.5,
                "x-ai/grok-4.5": 2.0,
                "anthropic/claude-opus-5": 5.0,
            },
            "output_per_million": {
                "anthropic/claude-sonnet-5": 10.0,
                "google/gemini-3.5-flash": 9.0,
                "x-ai/grok-4.5": 6.0,
                "anthropic/claude-opus-5": 25.0,
            },
        },
        "analysis": {
            "bootstrap_samples": 1000,
            "permutation_samples": 1000,
        },
    }


def _point_task() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rubric_version": "concept_intervention_judge_rubric_v3",
        "task_id": "a" * 64,
        "task_type": "pointwise",
        "split": "validation",
        "concept_id": "goemotions:optimism",
        "concept_name": "optimism",
        "concept_definition": "Expectation that future outcomes will be favorable.",
        "user_prompt": "What might happen next?",
        "response": "I expect the plan to work well.",
    }


def _pair_task() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rubric_version": "concept_intervention_judge_rubric_v4",
        "task_id": "b" * 64,
        "task_type": "pairwise",
        "split": "validation",
        "concept_id": "goemotions:optimism",
        "concept_name": "optimism",
        "concept_definition": "Expectation that future outcomes will be favorable.",
        "user_prompt": "What might happen next?",
        "response_a": "Great! Great! Great! The future will be wonderful!",
        "response_b": "The plan is clear and well structured, and it might work.",
    }


def test_config_rejects_openai_and_router_aliases(tmp_path: Path) -> None:
    payload = _config_payload(tmp_path)
    JudgeEvaluationConfig.model_validate(payload)
    for invalid in ("openai/gpt-5", "openrouter/auto", "anthropic/model:free"):
        changed = json.loads(json.dumps(payload))
        changed["judges"]["primary"] = invalid
        with pytest.raises(ValueError, match="exact, concrete, non-OpenAI"):
            JudgeEvaluationConfig.model_validate(changed)


def test_prompt_separates_scores_and_sanitizes_only_optional_evidence() -> None:
    task = _point_task()
    messages = render_messages(task)
    rendered = "\n".join(message["content"] for message in messages)
    assert "independently" in rendered
    assert "longer response is not inherently better" in rendered
    assert "return an empty evidence array" in rendered
    judgment = {
        "target_expression": 70,
        "coherence": 95,
        "relevance": 90,
        "refusal": False,
        "invalid": False,
        "evidence": ["expect the plan to work well"],
        "rationale": "The response clearly expects a favorable result.",
    }
    assert validate_judgment(task, judgment) == judgment
    warnings: list[str] = []
    sanitized = validate_judgment(
        task, judgment | {"evidence": ["not an exact quote"]}, warnings=warnings
    )
    assert sanitized["evidence"] == []
    assert warnings == ["evidence: dropped 1 non-exact or out-of-bounds item(s)"]


def test_pairwise_prompt_keeps_target_expression_orthogonal_to_quality() -> None:
    messages = render_messages(_pair_task())
    rendered = "\n".join(message["content"] for message in messages)
    assert "same TARGET_EXPRESSION construct and 0--100 anchors" in rendered
    assert "must not\ndepend on coherence, relevance" in rendered
    assert "low-quality response can therefore have greater target expression" in rendered
    assert "Never use quality to break a target tie or reverse the target winner" in rendered
    assert "explain the target comparison and quality comparison" in rendered
    wire = response_schema("pairwise")["json_schema"]
    assert wire["name"] == "jlens_pairwise_judgment_v4"
    target_description = wire["schema"]["properties"]["target_preference"]["description"]
    quality_description = wire["schema"]["properties"]["quality_preference"]["description"]
    assert "quality dimensions must not affect" in target_description
    assert "target intensity must not affect" in quality_description


def test_wire_schema_is_provider_portable_and_local_validation_is_strict() -> None:
    unsupported = {
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
    }

    def assert_portable(value: Any) -> None:
        if isinstance(value, dict):
            assert not unsupported.intersection(value)
            for child in value.values():
                assert_portable(child)
        elif isinstance(value, list):
            for child in value:
                assert_portable(child)

    assert_portable(response_schema("pointwise"))
    assert_portable(response_schema("pairwise"))

    task = _point_task()
    valid = {
        "target_expression": 70,
        "coherence": 95,
        "relevance": 90,
        "refusal": False,
        "invalid": False,
        "evidence": [],
        "rationale": "Concise rationale.",
    }
    with pytest.raises(ValueError, match="target_expression"):
        validate_judgment(task, valid | {"target_expression": 101})
    warnings: list[str] = []
    sanitized = validate_judgment(
        task, valid | {"evidence": ["", "", "", ""]}, warnings=warnings
    )
    assert sanitized["evidence"] == []
    assert warnings == ["evidence: dropped 4 non-exact or out-of-bounds item(s)"]
    with pytest.raises(ValueError, match="evidence must be an array"):
        validate_judgment(task, valid | {"evidence": "expect the plan"})
    warnings = []
    sanitized = validate_judgment(
        task, valid | {"rationale": "x" * 601}, warnings=warnings
    )
    assert sanitized["rationale"] == "x" * 600
    assert warnings == ["rationale: truncated from 601 to 600 characters"]
    with pytest.raises(ValueError, match="rationale"):
        validate_judgment(task, valid | {"rationale": 1})


def test_openrouter_client_pins_model_and_privacy_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _point_task()
    captured: dict[str, Any] = {}

    def transport(
        endpoint: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> dict[str, Any]:
        captured.update(
            endpoint=endpoint,
            headers=headers,
            body=json.loads(body),
            timeout=timeout,
        )
        judgment = {
            "target_expression": 70,
            "coherence": 95,
            "relevance": 90,
            "refusal": False,
            "invalid": False,
            "evidence": ["expect the plan"],
            "rationale": "Clear favorable expectation.",
        }
        return {
            "id": "request-1",
            "model": captured["body"]["model"],
            "provider": "Anthropic",
            "created": 1,
            "choices": [
                {
                    "message": {"content": json.dumps(judgment)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30},
        }

    monkeypatch.setenv("JLENS_JUDGE_API_KEY", "unit-test-secret")
    client = OpenRouterClient(
        OpenRouterConfig(
            max_retries=0,
            temperature_unsupported_models=["anthropic/claude-sonnet-5"],
            reasoning_effort_by_model={
                "anthropic/claude-sonnet-5": "low",
                "google/gemini-3.5-flash": "minimal",
            },
        ),
        transport=transport,
    )
    result = client.judge(model="anthropic/claude-sonnet-5", task=task)
    assert result.returned_model == "anthropic/claude-sonnet-5"
    assert result.validation_warnings == ()
    assert captured["body"]["provider"] == {
        "data_collection": "deny",
        "require_parameters": True,
        "allow_fallbacks": False,
    }
    assert captured["body"]["response_format"]["type"] == "json_schema"
    assert "temperature" not in captured["body"]
    assert captured["body"]["reasoning"] == {"effort": "low", "exclude": True}
    assert "unit-test-secret" not in json.dumps(captured["body"])
    client.judge(model="google/gemini-3.5-flash", task=task)
    assert captured["body"]["temperature"] == 0.0
    assert captured["body"]["reasoning"] == {"effort": "minimal", "exclude": True}


@pytest.mark.parametrize("invalid_content", [None, '{"target_expression":70'])
def test_openrouter_client_retries_invalid_structured_response(
    monkeypatch: pytest.MonkeyPatch,
    invalid_content: str | None,
) -> None:
    calls = 0
    request_bodies: list[dict[str, Any]] = []
    task = _point_task()
    judgment = {
        "target_expression": 70,
        "coherence": 95,
        "relevance": 90,
        "refusal": False,
        "invalid": False,
        "evidence": ["expect the plan"],
        "rationale": "Clear favorable expectation.",
    }

    def transport(
        endpoint: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> dict[str, Any]:
        nonlocal calls
        request_bodies.append(json.loads(body))
        calls += 1
        content = invalid_content if calls == 1 else json.dumps(judgment)
        return {
            "id": f"request-{calls}",
            "model": "anthropic/claude-sonnet-5",
            "choices": [
                {
                    "message": {
                        "content": content,
                        "reasoning": "hidden" if calls == 1 else None,
                    },
                    "finish_reason": "length" if calls == 1 else "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30},
        }

    monkeypatch.setenv("JLENS_JUDGE_API_KEY", "unit-test-secret")
    client = OpenRouterClient(
        OpenRouterConfig(
            max_retries=1,
            provider_order_by_model={
                "anthropic/claude-sonnet-5": ["anthropic", "amazon-bedrock"]
            },
        ),
        transport=transport,
        sleeper=lambda _: None,
    )
    response = client.judge(model="anthropic/claude-sonnet-5", task=task)
    assert response.attempts == 2
    assert request_bodies[0]["provider"]["order"] == [
        "anthropic",
        "amazon-bedrock",
    ]
    assert request_bodies[1]["provider"]["order"] == [
        "amazon-bedrock", "anthropic"
    ]
    assert response.judgment == judgment
    assert calls == 2


@pytest.mark.parametrize(
    ("finish_reasons", "expected_type"),
    [
        (["content_filter", "content_filter"], OpenRouterContentFilterError),
        (["length", "content_filter"], OpenRouterError),
    ],
)
def test_content_filter_error_requires_every_registered_attempt(
    monkeypatch: pytest.MonkeyPatch,
    finish_reasons: list[str],
    expected_type: type[OpenRouterError],
) -> None:
    calls = 0

    def transport(
        endpoint: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> dict[str, Any]:
        nonlocal calls
        finish_reason = finish_reasons[calls]
        calls += 1
        return {
            "id": f"filtered-{calls}",
            "model": "anthropic/claude-sonnet-5",
            "choices": [
                {
                    "message": {"content": None},
                    "finish_reason": finish_reason,
                }
            ],
        }

    monkeypatch.setenv("JLENS_JUDGE_API_KEY", "unit-test-secret")
    client = OpenRouterClient(
        OpenRouterConfig(
            max_retries=1,
            provider_order_by_model={
                "anthropic/claude-sonnet-5": ["anthropic", "amazon-bedrock"]
            },
        ),
        transport=transport,
        sleeper=lambda _: None,
    )
    with pytest.raises(OpenRouterError) as captured:
        client.judge(model="anthropic/claude-sonnet-5", task=_point_task())
    assert type(captured.value) is expected_type
    assert calls == 2


def test_openrouter_key_status_is_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def key_transport(
        endpoint: str,
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        captured.update(endpoint=endpoint, headers=headers, timeout=timeout)
        return {"data": {"limit_remaining": 98.5, "usage": 1.5}}

    monkeypatch.setenv("JLENS_JUDGE_API_KEY", "unit-test-secret")
    client = OpenRouterClient(
        OpenRouterConfig(max_retries=0),
        key_transport=key_transport,
    )
    status = client.key_status()
    assert status["limit_remaining"] == 98.5
    assert captured["endpoint"] == "https://openrouter.ai/api/v1/key"
    assert captured["headers"]["Authorization"] == "Bearer unit-test-secret"


def test_nonretryable_http_error_reports_actual_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def transport(
        endpoint: str,
        headers: dict[str, str],
        body: bytes,
        timeout: float,
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(endpoint, 404, "not found", None, None)

    monkeypatch.setenv("JLENS_JUDGE_API_KEY", "unit-test-secret")
    client = OpenRouterClient(
        OpenRouterConfig(max_retries=5),
        transport=transport,
        sleeper=lambda _: None,
    )
    with pytest.raises(OpenRouterError, match=r"after 1 attempt\(s\): HTTP 404"):
        client.judge(model="anthropic/claude-sonnet-5", task=_point_task())
    assert calls == 1


def _shard(
    tmp_path: Path,
    *,
    method: str,
    condition: dict[str, Any],
    margin: float,
) -> _Shard:
    return _Shard(
        method=method,
        concept_id="goemotions:optimism",
        summary_path=tmp_path / f"{method}-{len(condition)}.json",
        summary={"grid_condition": condition},
        candidate_rows=({"evaluation_split": "validation", "target_margin": margin},),
    )


def test_selection_uses_validation_only_and_registered_tie_breaks(tmp_path: Path) -> None:
    j_shards = [
        _shard(tmp_path, method="j", condition={"condition_id": "full", "strength": 0.0}, margin=0),
        _shard(
            tmp_path, method="j", condition={"condition_id": "full", "strength": 0.25}, margin=0
        ),
        _shard(tmp_path, method="j", condition={"condition_id": "j", "strength": 0.125}, margin=2),
        _shard(tmp_path, method="j", condition={"condition_id": "j", "strength": 0.25}, margin=2),
        _shard(
            tmp_path, method="j", condition={"condition_id": "non_j", "strength": 0.25}, margin=0
        ),
        _shard(
            tmp_path,
            method="j",
            condition={"condition_id": "random_101", "strength": 0.25},
            margin=0,
        ),
    ]
    # The smaller tied J strength is selected, so add its matched controls.
    j_shards.extend(
        [
            _shard(
                tmp_path,
                method="j",
                condition={"condition_id": "full", "strength": 0.125},
                margin=0,
            ),
            _shard(
                tmp_path,
                method="j",
                condition={"condition_id": "non_j", "strength": 0.125},
                margin=0,
            ),
            _shard(
                tmp_path,
                method="j",
                condition={"condition_id": "random_101", "strength": 0.125},
                margin=0,
            ),
        ]
    )
    raptor_shards = [
        _shard(tmp_path, method="raptor", condition={"condition_id": "no_hook"}, margin=0),
        _shard(
            tmp_path,
            method="raptor",
            condition={"condition_id": "p90", "target_probability": 0.9},
            margin=3,
        ),
        _shard(
            tmp_path,
            method="raptor",
            condition={"condition_id": "p99", "target_probability": 0.99},
            margin=3,
        ),
    ]
    config = JudgeEvaluationConfig.model_validate(_config_payload(tmp_path))
    selected, record = _select_registered_conditions(j_shards, raptor_shards, config=config)
    assert record["j"]["strength"] == 0.125
    assert record["raptor"]["target_probability"] == 0.9
    assert selected["baseline"].condition["strength"] == 0.0
    assert {"full", "j", "non_j", "random_101", "raptor"}.issubset(selected)


def test_clustered_statistics_and_bh_are_deterministic() -> None:
    records = [
        {"prompt_id": "p1", "delta": 1.0},
        {"prompt_id": "p1", "delta": 3.0},
        {"prompt_id": "p2", "delta": 4.0},
    ]
    first = paired_cluster_summary(
        records,
        seed=7,
        bootstrap_samples=1000,
        permutation_samples=1000,
        confidence_level=0.95,
    )
    second = paired_cluster_summary(
        records,
        seed=7,
        bootstrap_samples=1000,
        permutation_samples=1000,
        confidence_level=0.95,
    )
    assert first == second
    assert first["estimate"] == 3.0
    assert first["prompt_clusters"] == 2
    assert agreement_metrics([0, 50, 100], [0, 40, 100])["spearman"] == 1.0
    adjusted = benjamini_hochberg({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.04, "c": 0.04})


def test_run_is_atomic_and_resumable_without_network(tmp_path: Path) -> None:
    payload = _config_payload(tmp_path)
    config_path = tmp_path / "judge.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    root = Path(payload["output_dir"])
    (root / "tasks").mkdir(parents=True)
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    tasks = [
        _point_task(),
        _point_task() | {"task_id": "b" * 64},
    ]
    (root / "tasks" / "pointwise_validation.jsonl").write_text(
        "".join(json.dumps(task) + "\n" for task in tasks),
        encoding="utf-8",
    )

    class FakeClient:
        calls = 0

        def key_status(self) -> dict[str, float]:
            return {"limit_remaining": 100.0}

        def judge(self, *, model: str, task: dict[str, Any]) -> JudgeResponse:
            self.calls += 1
            return JudgeResponse(
                requested_model=model,
                returned_model=model,
                response_id="fake",
                provider="test",
                created=1,
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                judgment={
                    "target_expression": 70,
                    "coherence": 95,
                    "relevance": 90,
                    "refusal": False,
                    "invalid": False,
                    "evidence": ["expect the plan"],
                    "rationale": "test",
                },
                attempts=1,
            )

    fake = FakeClient()
    first = run_judge_tasks(
        config_path,
        role="primary",
        task_set="pointwise",
        split="validation",
        client=fake,
        limit=1,
    )
    second = run_judge_tasks(
        config_path,
        role="primary",
        task_set="pointwise",
        split="validation",
        client=fake,
        limit=1,
    )
    third = run_judge_tasks(
        config_path,
        role="primary",
        task_set="pointwise",
        split="validation",
        client=fake,
        limit=1,
    )
    assert first["complete"] is False
    assert first["invocation_succeeded"] is True
    assert first["remaining_tasks"] == 1
    assert second["complete"] is True
    assert second["new_tasks"] == 1
    assert third["complete"] is True
    assert third["new_tasks"] == 0
    assert fake.calls == 2


def test_expert_provider_filter_escalates_and_resumes(tmp_path: Path) -> None:
    payload = _config_payload(tmp_path)
    payload["review"] = {
        "expert_provider_filter_escalation": True,
        "human_audit_per_concept": 1,
    }
    payload["openrouter"] = {
        "max_retries": 1,
        "provider_order_by_model": {
            "anthropic/claude-opus-5": ["provider-a", "provider-b"]
        },
    }
    config_path = tmp_path / "judge.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    root = Path(payload["output_dir"])
    (root / "tasks").mkdir(parents=True)
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    task = _point_task() | {"task_id": "c" * 64, "split": "test"}
    for name, rows in (
        ("expert_review.jsonl", [task]),
        ("pointwise_test.jsonl", [task]),
        ("pairwise_test.jsonl", []),
    ):
        (root / "tasks" / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    class FilteredClient:
        calls = 0

        def key_status(self) -> dict[str, float]:
            return {"limit_remaining": 100.0}

        def judge(self, *, model: str, task: dict[str, Any]) -> JudgeResponse:
            self.calls += 1
            raise OpenRouterContentFilterError(
                "all registered provider attempts returned content_filter: 2 attempt(s)"
            )

    fake = FilteredClient()
    first = run_judge_tasks(
        config_path,
        role="expert_review",
        task_set="expert_review",
        client=fake,
        limit=1,
    )
    second = run_judge_tasks(
        config_path,
        role="expert_review",
        task_set="expert_review",
        client=fake,
        limit=1,
    )
    assert first["complete"] is True
    assert first["invocation_succeeded"] is True
    assert first["completed_tasks"] == 1
    assert first["machine_completed_tasks"] == 0
    assert first["provider_filtered_tasks"] == 1
    assert first["invocation_provider_filtered_tasks"] == 1
    assert first["failures"] == []
    assert second["new_tasks"] == 0
    assert fake.calls == 1
    exclusion_path = Path(first["provider_filtered_exclusions_path"])
    assert [json.loads(row) for row in exclusion_path.read_text().splitlines()] == [
        {
            "task_id": task["task_id"],
            "reason": "provider_content_filter_across_registered_routes",
            "provider_order": ["provider-a", "provider-b"],
            "attempts": 2,
        }
    ]
    audit = export_human_audit(config_path)
    assert audit["provider_filtered_expert_tasks_forced"] == 1


def test_budget_reserve_blocks_before_task_transmission(tmp_path: Path) -> None:
    payload = _config_payload(tmp_path)
    payload["budget"] = {"reserve_openrouter_credit_usd": 100.0}
    config_path = tmp_path / "judge.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    root = Path(payload["output_dir"])
    (root / "tasks").mkdir(parents=True)
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    task = _point_task()
    (root / "tasks" / "pointwise_validation.jsonl").write_text(
        json.dumps(task) + "\n",
        encoding="utf-8",
    )

    class FakeClient:
        calls = 0

        def key_status(self) -> dict[str, float]:
            return {"limit_remaining": 100.0}

        def judge(self, *, model: str, task: dict[str, Any]) -> JudgeResponse:
            self.calls += 1
            raise AssertionError("budget gate must run before judge()")

    fake = FakeClient()
    with pytest.raises(JudgeWorkflowError, match="reserve_openrouter_credit_usd"):
        run_judge_tasks(
            config_path,
            role="primary",
            task_set="pointwise",
            split="validation",
            client=fake,
            limit=1,
        )
    assert fake.calls == 0


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_fake_results(
    root: Path,
    *,
    model: str,
    role: str,
    task_set: str,
    split: str,
    tasks: list[dict[str, Any]],
    scores: dict[str, int] | None = None,
    pair_preferences: dict[str, str] | None = None,
) -> None:
    destination = _result_directory(
        root,
        model=model,
        task_set=task_set,
        split=split,
    )
    destination.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        if task["task_type"] == "pointwise":
            target = int((scores or {})[task["task_id"]])
            judgment = {
                "target_expression": target,
                "coherence": 90,
                "relevance": 90,
                "refusal": False,
                "invalid": False,
                "evidence": [],
                "rationale": "synthetic",
            }
        else:
            judgment = {
                "target_preference": (pair_preferences or {})[task["task_id"]],
                "quality_preference": "tie",
                "target_strength": "moderate",
                "confidence": 90,
                "evidence_a": [],
                "evidence_b": [],
                "rationale": "synthetic",
            }
        payload = {
            "task_id": task["task_id"],
            "task_sha256": _canonical_hash(task),
            "rubric_sha256": rubric_hash(),
            "requested_model": model,
            "returned_model": model,
            "judge_role": role,
            "judgment": judgment,
        }
        (destination / f"{task['task_id']}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_expert_review_excludes_resolved_and_provider_filtered_cases(tmp_path: Path) -> None:
    resolved_id = "1" * 64
    excluded_id = "2" * 64
    unresolved_id = "3" * 64
    payload = _config_payload(tmp_path)
    payload["review"] = {
        "pointwise_disagreement_threshold": 25,
        "expert_disagreement_threshold": 20,
        "expert_audit_fraction": 0.0,
        "expert_provider_filtered_task_ids": [excluded_id],
        "human_audit_per_concept": 1,
    }
    config_path = tmp_path / "judge.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = JudgeEvaluationConfig.model_validate(payload)
    root = Path(config.output_dir)

    tasks = [
        _point_task() | {"task_id": resolved_id, "split": "test"},
        _point_task() | {"task_id": excluded_id, "split": "test"},
        _point_task() | {"task_id": unresolved_id, "split": "test"},
    ]
    _write_jsonl(root / "tasks" / "pointwise_test.jsonl", tasks)
    _write_jsonl(root / "tasks" / "pairwise_test.jsonl", [])
    _write_jsonl(root / "private" / "pairwise_test_map.jsonl", [])
    for role, model, scores in (
        (
            "primary",
            config.judges.primary,
            {resolved_id: 0, excluded_id: 0, unresolved_id: 0},
        ),
        (
            "secondary",
            config.judges.secondary,
            {resolved_id: 100, excluded_id: 100, unresolved_id: 100},
        ),
    ):
        _write_fake_results(
            root,
            model=model,
            role=role,
            task_set="pointwise",
            split="test",
            tasks=tasks,
            scores=scores,
        )

    arbitration_manifest = prepare_review_tasks(config_path, tier="arbitration")
    assert arbitration_manifest["tasks"] == 3
    _write_fake_results(
        root,
        model=config.judges.arbitration,
        role="arbitration",
        task_set="arbitration",
        split=None,
        tasks=tasks,
        scores={resolved_id: 50, excluded_id: 80, unresolved_id: 80},
    )

    expert_manifest = prepare_review_tasks(config_path, tier="expert_review")
    assert expert_manifest["tasks"] == 1
    assert expert_manifest["excluded_tasks"] == 1
    selected = [
        json.loads(row) for row in Path(expert_manifest["task_path"]).read_text().splitlines()
    ]
    assert [task["task_id"] for task in selected] == [unresolved_id]
    private_rows = [
        json.loads(row) for row in Path(expert_manifest["private_map"]).read_text().splitlines()
    ]
    assert private_rows == [
        {"task_id": unresolved_id, "review_reasons": ["arbitration_remains_discordant"]}
    ]
    exclusion_rows = [
        json.loads(row)
        for row in Path(expert_manifest["exclusions_path"]).read_text().splitlines()
    ]
    assert exclusion_rows == [
        {
            "task_id": excluded_id,
            "reason": "provider_content_filter_across_registered_routes",
        }
    ]

    audit_manifest = export_human_audit(config_path)
    audit_tasks = [
        json.loads(row) for row in Path(audit_manifest["tasks"]).read_text().splitlines()
    ]
    assert excluded_id in {task["task_id"] for task in audit_tasks}
    assert audit_manifest["provider_filtered_expert_tasks_forced"] == 1


def test_calibration_and_aggregation_end_to_end_offline(tmp_path: Path) -> None:
    payload = _config_payload(tmp_path)
    payload["analysis"]["report_after_invalid_guardrail_failure"] = True
    config_path = tmp_path / "judge.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = JudgeEvaluationConfig.model_validate(payload)
    root = Path(config.output_dir)
    (root / "private").mkdir(parents=True)

    validation_point = [_point_task() | {"task_id": f"val-point-{index}"} for index in range(3)]
    validation_pair = [
        {
            "task_id": f"val-pair-{index}",
            "task_type": "pairwise",
            "split": "validation",
            "concept_id": "goemotions:optimism",
        }
        for index in range(2)
    ]
    validation_pair_private = [
        {
            "task_id": "val-pair-0",
            "pair_group_id": "val-group",
            "response_a_role": "j",
            "response_b_role": "raptor",
        },
        {
            "task_id": "val-pair-1",
            "pair_group_id": "val-group",
            "response_a_role": "raptor",
            "response_b_role": "j",
        },
    ]
    _write_jsonl(root / "tasks" / "pointwise_validation.jsonl", validation_point)
    _write_jsonl(root / "tasks" / "pairwise_validation.jsonl", validation_pair)
    _write_jsonl(root / "private" / "pairwise_validation_map.jsonl", validation_pair_private)
    for role, model, scores in (
        ("primary", config.judges.primary, [10, 50, 90]),
        ("secondary", config.judges.secondary, [15, 55, 85]),
    ):
        _write_fake_results(
            root,
            model=model,
            role=role,
            task_set="pointwise",
            split="validation",
            tasks=validation_point,
            scores={
                task["task_id"]: score for task, score in zip(validation_point, scores, strict=True)
            },
        )
        _write_fake_results(
            root,
            model=model,
            role=role,
            task_set="pairwise",
            split="validation",
            tasks=validation_pair,
            pair_preferences={"val-pair-0": "A", "val-pair-1": "B"},
        )
    calibration = calibrate_judges(config_path)
    assert calibration["passed"] is True

    role_scores = {
        "baseline": 10,
        "full": 40,
        "j": 70,
        "non_j": 20,
        "random_101": 15,
        "raptor": 50,
    }
    test_point: list[dict[str, Any]] = []
    test_private: list[dict[str, Any]] = []
    test_scores: dict[str, int] = {}
    for prompt_index in range(2):
        for condition_role, target in role_scores.items():
            task_id = f"test-point-{prompt_index}-{condition_role}"
            test_point.append(_point_task() | {"task_id": task_id, "split": "test"})
            test_private.append(
                {
                    "task_id": task_id,
                    "concept_id": "goemotions:optimism",
                    "condition_role": condition_role,
                    "method": "synthetic",
                    "condition_id": condition_role,
                    "grid_point": {},
                    "prompt_id": f"prompt-{prompt_index}",
                    "decoding": "greedy",
                    "seed": None,
                }
            )
            test_scores[task_id] = target
    test_pair: list[dict[str, Any]] = []
    test_pair_private: list[dict[str, Any]] = []
    pair_preferences: dict[str, str] = {}
    for prompt_index in range(2):
        group = f"test-group-{prompt_index}"
        for order in range(2):
            task_id = f"test-pair-{prompt_index}-{order}"
            test_pair.append(
                {
                    "task_id": task_id,
                    "task_type": "pairwise",
                    "split": "test",
                    "concept_id": "goemotions:optimism",
                }
            )
            j_first = order == 0
            test_pair_private.append(
                {
                    "task_id": task_id,
                    "pair_group_id": group,
                    "concept_id": "goemotions:optimism",
                    "prompt_id": f"prompt-{prompt_index}",
                    "decoding": "greedy",
                    "seed": None,
                    "response_a_role": "j" if j_first else "raptor",
                    "response_b_role": "raptor" if j_first else "j",
                }
            )
            pair_preferences[task_id] = "A" if j_first else "B"
    _write_jsonl(root / "tasks" / "pointwise_test.jsonl", test_point)
    _write_jsonl(root / "tasks" / "pairwise_test.jsonl", test_pair)
    _write_jsonl(root / "private" / "pointwise_test_map.jsonl", test_private)
    _write_jsonl(root / "private" / "pairwise_test_map.jsonl", test_pair_private)
    for role, model in (
        ("primary", config.judges.primary),
        ("secondary", config.judges.secondary),
    ):
        _write_fake_results(
            root,
            model=model,
            role=role,
            task_set="pointwise",
            split="test",
            tasks=test_point,
            scores=test_scores,
        )
        _write_fake_results(
            root,
            model=model,
            role=role,
            task_set="pairwise",
            split="test",
            tasks=test_pair,
            pair_preferences=pair_preferences,
        )
    invalid_result_path = (
        _result_directory(
            root,
            model=config.judges.primary,
            task_set="pointwise",
            split="test",
        )
        / f"{test_point[0]['task_id']}.json"
    )
    invalid_result = json.loads(invalid_result_path.read_text(encoding="utf-8"))
    invalid_result["judgment"]["invalid"] = True
    invalid_result_path.write_text(json.dumps(invalid_result), encoding="utf-8")

    strict_payload = json.loads(json.dumps(payload))
    strict_payload["analysis"].pop("report_after_invalid_guardrail_failure")
    strict_config_path = tmp_path / "judge-strict.yaml"
    strict_config_path.write_text(yaml.safe_dump(strict_payload), encoding="utf-8")
    with pytest.raises(JudgeWorkflowError, match="invalid judgment rate"):
        aggregate_evaluation(strict_config_path)

    index = aggregate_evaluation(config_path)
    report = json.loads(Path(index["report"]).read_text(encoding="utf-8"))
    assert report["primary_result"]["target_expression"]["estimate"] == 20.0
    assert report["confirmatory_interpretation_allowed"] is False
    assert report["result_status"] == (
        "descriptive_only_after_registered_invalid_guardrail_failure"
    )
    assert report["pointwise"]["invalid_guardrail"]["passed"] is False
    assert index["descriptive_only"] is True
    assert report["pairwise"]["macro"]["consensus_counts"]["j"] == 2
    assert report["pairwise"]["position_bias_rate"] == {
        "primary": 0.0,
        "secondary": 0.0,
    }


def test_cli_exposes_judge_workflow() -> None:
    parser = cli.build_parser()
    for argv in (
        ["judge", "validate", "judge.yaml"],
        ["judge", "prepare", "judge.yaml"],
        [
            "judge",
            "run",
            "judge.yaml",
            "--role",
            "primary",
            "--task-set",
            "pointwise",
            "--split",
            "validation",
        ],
        ["judge", "calibrate", "judge.yaml"],
        ["judge", "aggregate", "judge.yaml"],
        ["judge", "audit-export", "judge.yaml"],
    ):
        assert callable(parser.parse_args(argv).handler)
