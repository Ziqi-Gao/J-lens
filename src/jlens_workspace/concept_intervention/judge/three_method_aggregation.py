"""Integrity-gated aggregation for the post-hoc three-method judge study."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from jlens_workspace.artifacts import atomic_write_json, sha256_file
from jlens_workspace.concept_intervention.evaluation import atomic_write_jsonl
from jlens_workspace.concept_intervention.judge.amendment import (
    validate_format_amendment,
)
from jlens_workspace.concept_intervention.judge.prompts import rubric_hash
from jlens_workspace.concept_intervention.judge.statistics import (
    agreement_metrics,
    benjamini_hochberg,
    hierarchical_macro_summary,
    paired_cluster_summary,
)
from jlens_workspace.concept_intervention.judge.three_method_config import (
    JudgeEvaluationConfig,
    load_judge_config,
)
from jlens_workspace.concept_intervention.judge.three_method_prepare import (
    PRIMARY_ROLES,
    SECONDARY_ROLES,
    validate_evaluation,
)
from jlens_workspace.concept_intervention.judge.workflow import (
    JudgeWorkflowError,
    _canonical_hash,
    _load_results,
    _normalized_pair_preference,
    _read_jsonl,
    _result_directory,
)

PRIMARY_CONTRASTS: dict[str, tuple[str, str]] = {
    "j_minus_raptor": ("j_component", "raptor"),
    "j_minus_iti_native": ("j_component", "iti_native"),
    "raptor_minus_iti_native": ("raptor", "iti_native"),
}
SENSITIVITY_CONTRASTS: dict[str, tuple[str, str]] = {
    "j_component_minus_common_baseline": ("j_component", "common_baseline"),
    "raptor_minus_common_baseline": ("raptor", "common_baseline"),
    "iti_native_minus_common_baseline": ("iti_native", "common_baseline"),
    "j_full_minus_common_baseline": ("j_full", "common_baseline"),
    "j_non_j_minus_common_baseline": ("j_non_j", "common_baseline"),
    "j_matched_random_minus_common_baseline": (
        "j_matched_random",
        "common_baseline",
    ),
    "iti_layer_matched_minus_common_baseline": (
        "iti_layer_matched",
        "common_baseline",
    ),
    "iti_probe_weight_minus_common_baseline": (
        "iti_probe_weight",
        "common_baseline",
    ),
    "iti_matched_random_minus_common_baseline": (
        "iti_matched_random",
        "common_baseline",
    ),
}
METRICS = ("target_expression", "quality", "refusal")


def _assert_complete_results(
    tasks: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    *,
    model: str,
    role: str,
    protocol_amendment: Mapping[str, Any],
) -> None:
    if set(results) != set(tasks):
        raise JudgeWorkflowError(
            f"incomplete {role} results: "
            f"missing={len(set(tasks) - set(results))}, "
            f"extra={len(set(results) - set(tasks))}"
        )
    expected_rubric = rubric_hash()
    for task_id, task in tasks.items():
        result = results[task_id]
        if (
            result.get("task_sha256") != _canonical_hash(task)
            or result.get("rubric_sha256") != expected_rubric
            or result.get("protocol_amendment") != protocol_amendment
            or result.get("requested_model") != model
            or result.get("returned_model") != model
            or result.get("judge_role") != role
            or not isinstance(result.get("request_payload"), Mapping)
            or not isinstance(result.get("raw_provider_response"), Mapping)
            or not isinstance(result.get("attempt_records"), list)
            or not result.get("attempt_records")
            or not result.get("request_started_at")
            or not result.get("response_received_at")
        ):
            raise JudgeWorkflowError(f"result identity/audit mismatch for {task_id}")


def _paired_records(
    treatment: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> list[dict[str, Any]]:
    def key(row: Mapping[str, Any]) -> tuple[str, str, int | None]:
        seed = row.get("seed")
        return (
            str(row["prompt_id"]),
            str(row["decoding"]),
            None if seed is None else int(seed),
        )

    treatment_by_key = {key(row): row for row in treatment}
    reference_by_key = {key(row): row for row in reference}
    if (
        len(treatment_by_key) != len(treatment)
        or len(reference_by_key) != len(reference)
        or set(treatment_by_key) != set(reference_by_key)
        or not treatment_by_key
    ):
        raise JudgeWorkflowError(f"unmatched paired rows for metric={metric}")
    return [
        {
            "prompt_id": prompt_id,
            "decoding": decoding,
            "seed": seed,
            "delta": float(treatment_by_key[(prompt_id, decoding, seed)][metric])
            - float(reference_by_key[(prompt_id, decoding, seed)][metric]),
        }
        for prompt_id, decoding, seed in sorted(
            treatment_by_key,
            key=lambda item: (
                item[0],
                item[1],
                -1 if item[2] is None else item[2],
            ),
        )
    ]


def _pointwise_rows(
    tasks: Mapping[str, Mapping[str, Any]],
    private_rows: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    private = {str(row["task_id"]): row for row in private_rows}
    if set(private) != set(tasks):
        raise JudgeWorkflowError("pointwise private map differs from public tasks")
    output: list[dict[str, Any]] = []
    for task_id in sorted(tasks):
        judgments = {
            role: role_results[task_id]["judgment"]
            for role, role_results in results.items()
        }
        target = float(
            np.mean([judgment["target_expression"] for judgment in judgments.values()])
        )
        coherence = float(
            np.mean([judgment["coherence"] for judgment in judgments.values()])
        )
        relevance = float(
            np.mean([judgment["relevance"] for judgment in judgments.values()])
        )
        row = private[task_id]
        output.append(
            {
                "task_id": task_id,
                "concept_id": row["concept_id"],
                "analysis_tier": row["analysis_tier"],
                "condition_role": row["condition_role"],
                "method": row["method"],
                "condition_id": row["condition_id"],
                "grid_point": row["grid_point"],
                "prompt_id": row["prompt_id"],
                "decoding": row["decoding"],
                "seed": row.get("seed"),
                "target_expression": target,
                "coherence": coherence,
                "relevance": relevance,
                "quality": (coherence + relevance) / 2.0,
                "refusal": float(
                    np.mean([float(item["refusal"]) for item in judgments.values()])
                ),
                "invalid": float(
                    np.mean([float(item["invalid"]) for item in judgments.values()])
                ),
                "judge_scores": {
                    role: {
                        key: judgment[key]
                        for key in (
                            "target_expression",
                            "coherence",
                            "relevance",
                            "refusal",
                            "invalid",
                        )
                    }
                    for role, judgment in judgments.items()
                },
            }
        )
    return output


def _pointwise_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: JudgeEvaluationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_concept_role: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_concept_role[str(row["concept_id"])][str(row["condition_role"])].append(row)
    expected_roles = set(PRIMARY_ROLES).union(SECONDARY_ROLES)
    for concept_id in config.source.concept_ids:
        if set(by_concept_role[concept_id]) != expected_roles:
            raise JudgeWorkflowError(f"{concept_id}: incomplete pointwise role matrix")

    contrast_specs = {**PRIMARY_CONTRASTS, **SENSITIVITY_CONTRASTS}
    concept_effects: dict[str, Any] = {}
    macro_inputs: dict[str, dict[str, Mapping[str, float]]] = defaultdict(dict)
    primary_p_values: dict[str, float] = {}
    for concept_index, concept_id in enumerate(config.source.concept_ids):
        concept_effects[concept_id] = {}
        roles = by_concept_role[concept_id]
        for contrast_index, (contrast, (treatment, reference)) in enumerate(
            contrast_specs.items()
        ):
            effects: dict[str, Any] = {
                "treatment_role": treatment,
                "reference_role": reference,
                "analysis_tier": (
                    "primary" if contrast in PRIMARY_CONTRASTS else "sensitivity"
                ),
            }
            for metric_index, metric in enumerate(METRICS):
                records = _paired_records(
                    roles[treatment],
                    roles[reference],
                    metric=metric,
                )
                summary = paired_cluster_summary(
                    records,
                    seed=(
                        config.seed
                        + 100_000 * concept_index
                        + 1_000 * contrast_index
                        + metric_index
                    ),
                    bootstrap_samples=config.analysis.bootstrap_samples,
                    permutation_samples=config.analysis.permutation_samples,
                    confidence_level=config.analysis.confidence_level,
                )
                effects[metric] = summary
                macro_inputs[f"{contrast}:{metric}"][concept_id] = summary[
                    "cluster_means"
                ]
                if contrast in PRIMARY_CONTRASTS and metric == "target_expression":
                    primary_p_values[f"{concept_id}:{contrast}"] = summary[
                        "p_value_two_sided"
                    ]
            effects["quality_guardrail_passed"] = (
                effects["quality"]["ci_lower"]
                >= config.analysis.quality_noninferiority_margin
            )
            effects["refusal_guardrail_passed"] = (
                effects["refusal"]["ci_upper"]
                <= config.analysis.refusal_rate_margin
            )
            concept_effects[concept_id][contrast] = effects

    for key, q_value in benjamini_hochberg(primary_p_values).items():
        concept_id, contrast = key.rsplit(":", 1)
        concept_effects[concept_id][contrast]["target_expression"][
            "fdr_q_value"
        ] = q_value

    macro: dict[str, Any] = {}
    for contrast_index, contrast in enumerate(contrast_specs):
        macro[contrast] = {
            "treatment_role": contrast_specs[contrast][0],
            "reference_role": contrast_specs[contrast][1],
            "analysis_tier": (
                "primary" if contrast in PRIMARY_CONTRASTS else "sensitivity"
            ),
        }
        for metric_index, metric in enumerate(METRICS):
            macro[contrast][metric] = hierarchical_macro_summary(
                macro_inputs[f"{contrast}:{metric}"],
                seed=config.seed + 1_000_000 + 10 * contrast_index + metric_index,
                bootstrap_samples=config.analysis.bootstrap_samples,
                confidence_level=config.analysis.confidence_level,
            )
        macro[contrast]["quality_guardrail_passed"] = (
            macro[contrast]["quality"]["ci_lower"]
            >= config.analysis.quality_noninferiority_margin
        )
        macro[contrast]["refusal_guardrail_passed"] = (
            macro[contrast]["refusal"]["ci_upper"]
            <= config.analysis.refusal_rate_margin
        )
    return concept_effects, macro


def _pairwise_analysis(
    private_rows: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    private = {str(row["task_id"]): row for row in private_rows}
    groups: dict[str, list[str]] = defaultdict(list)
    for task_id, row in private.items():
        groups[str(row["pair_group_id"])].append(task_id)
    rows: list[dict[str, Any]] = []
    consistency: dict[str, list[bool]] = defaultdict(list)
    quality_consistency: dict[str, list[bool]] = defaultdict(list)
    cross_judge: list[bool] = []
    for group_id in sorted(groups):
        task_ids = sorted(groups[group_id])
        if len(task_ids) != 2:
            raise JudgeWorkflowError(f"pair group lacks two orders: {group_id}")
        metadata = private[task_ids[0]]
        stable: dict[str, str | None] = {}
        stable_quality: dict[str, str | None] = {}
        orders: dict[str, list[str]] = {}
        quality_orders: dict[str, list[str]] = {}
        for role, role_results in results.items():
            preferences = [
                _normalized_pair_preference(role_results[task_id], private[task_id])
                for task_id in task_ids
            ]
            quality_preferences = [
                _normalized_pair_preference(
                    role_results[task_id],
                    private[task_id],
                    "quality_preference",
                )
                for task_id in task_ids
            ]
            orders[role] = preferences
            quality_orders[role] = quality_preferences
            is_stable = preferences[0] == preferences[1]
            quality_is_stable = quality_preferences[0] == quality_preferences[1]
            consistency[role].append(is_stable)
            quality_consistency[role].append(quality_is_stable)
            stable[role] = preferences[0] if is_stable else None
            stable_quality[role] = (
                quality_preferences[0] if quality_is_stable else None
            )
        stable_values = list(stable.values())
        consensus = (
            stable_values[0]
            if all(value is not None for value in stable_values)
            and len(set(stable_values)) == 1
            else "discordant"
        )
        stable_quality_values = list(stable_quality.values())
        quality_consensus = (
            stable_quality_values[0]
            if all(value is not None for value in stable_quality_values)
            and len(set(stable_quality_values)) == 1
            else "discordant"
        )
        if all(value is not None for value in stable_values):
            cross_judge.append(len(set(stable_values)) == 1)
        rows.append(
            {
                "pair_group_id": group_id,
                "method_pair": metadata["method_pair"],
                "concept_id": metadata["concept_id"],
                "prompt_id": metadata["prompt_id"],
                "decoding": metadata["decoding"],
                "seed": metadata.get("seed"),
                "order_preferences": orders,
                "stable_preferences": stable,
                "consensus": consensus,
                "quality_order_preferences": quality_orders,
                "stable_quality_preferences": stable_quality,
                "quality_consensus": quality_consensus,
            }
        )

    by_pair: dict[str, Any] = {}
    for method_pair in sorted({str(row["method_pair"]) for row in rows}):
        pair_rows = [row for row in rows if row["method_pair"] == method_pair]
        left, right = method_pair.split("_vs_", 1)
        labels = (left, right, "tie", "discordant")
        counts = {
            label: sum(row["consensus"] == label for row in pair_rows)
            for label in labels
        }
        quality_counts = {
            label: sum(row["quality_consensus"] == label for row in pair_rows)
            for label in labels
        }
        decisive = counts[left] + counts[right]
        by_pair[method_pair] = {
            "pairs": len(pair_rows),
            "consensus_counts": counts,
            "quality_consensus_counts": quality_counts,
            "left_preference_rate_among_decisive": (
                counts[left] / decisive if decisive else None
            ),
            "two_sided_binomial_p": (
                float(binomtest(counts[left], decisive, 0.5).pvalue)
                if decisive
                else None
            ),
        }
    summary = {
        "pairs": len(rows),
        "by_method_pair": by_pair,
        "order_consistency": {
            role: float(np.mean(values)) if values else 0.0
            for role, values in consistency.items()
        },
        "position_bias_rate": {
            role: 1.0 - float(np.mean(values)) if values else 1.0
            for role, values in consistency.items()
        },
        "quality_order_consistency": {
            role: float(np.mean(values)) if values else 0.0
            for role, values in quality_consistency.items()
        },
        "cross_judge_agreement_among_stable": (
            float(np.mean(cross_judge)) if cross_judge else 0.0
        ),
        "cross_judge_stable_groups": len(cross_judge),
    }
    return rows, summary


def _response_seals(
    root: Path,
    *,
    config: JudgeEvaluationConfig,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    models = {"primary": config.judges.primary, "secondary": config.judges.secondary}
    for role, model in models.items():
        for split in ("validation", "test"):
            for task_set in ("pointwise", "pairwise"):
                destination = _result_directory(
                    root,
                    model=model,
                    task_set=task_set,
                    split=split,
                )
                index_path = destination / "index.json"
                if not index_path.is_file():
                    raise JudgeWorkflowError(f"missing response index: {index_path}")
                index = json.loads(index_path.read_text(encoding="utf-8"))
                if not index.get("complete"):
                    raise JudgeWorkflowError(f"incomplete response index: {index_path}")
                files = {
                    path.name: sha256_file(path)
                    for path in sorted(destination.glob("*.json"))
                    if path.name != "index.json"
                }
                output[f"{role}:{task_set}:{split}"] = {
                    "index": str(index_path),
                    "index_sha256": sha256_file(index_path),
                    "responses": len(files),
                    "response_file_hashes_sha256": _canonical_hash(files),
                }
    for tier, model in (
        ("arbitration", config.judges.arbitration),
        ("expert_review", config.judges.expert_review),
    ):
        destination = _result_directory(
            root,
            model=model,
            task_set=tier,
            split=None,
        )
        index_path = destination / "index.json"
        if not index_path.is_file():
            raise JudgeWorkflowError(f"missing response index: {index_path}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if not index.get("complete"):
            raise JudgeWorkflowError(f"incomplete response index: {index_path}")
        files = {
            path.name: sha256_file(path)
            for path in sorted(destination.glob("*.json"))
            if path.name != "index.json"
        }
        output[tier] = {
            "index": str(index_path),
            "index_sha256": sha256_file(index_path),
            "responses": len(files),
            "response_file_hashes_sha256": _canonical_hash(files),
        }
    return output


def _verify_frozen_registration(
    config_path: Path,
    *,
    config: JudgeEvaluationConfig,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen_path = root / "frozen_config.yaml"
    if (
        manifest.get("config_sha256") != sha256_file(config_path)
        or not frozen_path.is_file()
        or sha256_file(frozen_path) != manifest.get("config_sha256")
    ):
        raise JudgeWorkflowError("frozen config seal mismatch")
    protocol_amendment = validate_format_amendment(
        config_path,
        root=root,
        experiment_name=config.experiment_name,
        verify_response_seals=True,
    )
    for artifact in manifest.get("task_artifacts", {}).values():
        for path_key, hash_key in (
            ("path", "sha256"),
            ("private_map", "private_map_sha256"),
        ):
            path = Path(artifact[path_key])
            if sha256_file(path) != artifact[hash_key]:
                raise JudgeWorkflowError(f"task artifact seal mismatch: {path}")
    source_validation = validate_evaluation(config_path)
    return manifest, source_validation, protocol_amendment


def aggregate_evaluation(config_path: str | Path) -> dict[str, Any]:
    """Produce a complete-but-always-descriptive three-method audit report."""

    registration = Path(config_path)
    config = load_judge_config(registration)
    root = Path(config.output_dir)
    manifest, source_validation, protocol_amendment = _verify_frozen_registration(
        registration,
        config=config,
        root=root,
    )
    calibration_path = root / "calibration.json"
    if not calibration_path.is_file():
        raise JudgeWorkflowError("calibration artifact is missing")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not calibration.get("passed"):
        raise JudgeWorkflowError("cannot aggregate a study that failed calibration")

    models = {"primary": config.judges.primary, "secondary": config.judges.secondary}
    point_tasks = {
        str(task["task_id"]): task
        for task in _read_jsonl(root / "tasks" / "pointwise_test.jsonl")
    }
    pair_tasks = {
        str(task["task_id"]): task
        for task in _read_jsonl(root / "tasks" / "pairwise_test.jsonl")
    }
    point_results: dict[str, dict[str, dict[str, Any]]] = {}
    pair_results: dict[str, dict[str, dict[str, Any]]] = {}
    for role, model in models.items():
        point_results[role] = _load_results(
            _result_directory(
                root,
                model=model,
                task_set="pointwise",
                split="test",
            )
        )
        pair_results[role] = _load_results(
            _result_directory(
                root,
                model=model,
                task_set="pairwise",
                split="test",
            )
        )
        _assert_complete_results(
            point_tasks,
            point_results[role],
            model=model,
            role=role,
            protocol_amendment=protocol_amendment,
        )
        _assert_complete_results(
            pair_tasks,
            pair_results[role],
            model=model,
            role=role,
            protocol_amendment=protocol_amendment,
        )

    point_rows = _pointwise_rows(
        point_tasks,
        _read_jsonl(root / "private" / "pointwise_test_map.jsonl"),
        point_results,
    )
    concept_effects, macro_effects = _pointwise_analysis(point_rows, config=config)
    pair_rows, pairwise = _pairwise_analysis(
        _read_jsonl(root / "private" / "pairwise_test_map.jsonl"),
        pair_results,
    )
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    point_path = analysis_dir / "pointwise_unblinded.jsonl"
    pair_path = analysis_dir / "pairwise_unblinded.jsonl"
    atomic_write_jsonl(point_path, point_rows)
    atomic_write_jsonl(pair_path, pair_rows)

    invalid_rate = float(np.mean([row["invalid"] for row in point_rows]))
    invalid_maximum = 1.0 - config.calibration.min_completion_rate
    invalid_passed = invalid_rate <= invalid_maximum
    invalid_by_judge = {
        role: float(
            np.mean(
                [float(row["judge_scores"][role]["invalid"]) for row in point_rows]
            )
        )
        for role in models
    }
    refusal_rate = float(np.mean([row["refusal"] for row in point_rows]))
    point_agreement = {
        metric: agreement_metrics(
            [
                float(point_results["primary"][task_id]["judgment"][metric])
                for task_id in sorted(point_tasks)
            ],
            [
                float(point_results["secondary"][task_id]["judgment"][metric])
                for task_id in sorted(point_tasks)
            ],
        )
        for metric in ("target_expression", "coherence", "relevance")
    }
    point_agreement["refusal_exact"] = float(
        np.mean(
            [
                point_results["primary"][task_id]["judgment"]["refusal"]
                == point_results["secondary"][task_id]["judgment"]["refusal"]
                for task_id in sorted(point_tasks)
            ]
        )
    )
    point_agreement["invalid_exact"] = float(
        np.mean(
            [
                point_results["primary"][task_id]["judgment"]["invalid"]
                == point_results["secondary"][task_id]["judgment"]["invalid"]
                for task_id in sorted(point_tasks)
            ]
        )
    )

    primary_guardrails = {
        contrast: {
            "quality_passed": macro_effects[contrast]["quality_guardrail_passed"],
            "refusal_passed": macro_effects[contrast]["refusal_guardrail_passed"],
        }
        for contrast in config.analysis.primary_contrasts
    }
    all_primary_guardrails = all(
        item["quality_passed"] and item["refusal_passed"]
        for item in primary_guardrails.values()
    )
    analysis_valid = invalid_passed and all_primary_guardrails

    review: dict[str, Any] = {}
    provider_filtered_total = 0
    for tier, model in (
        ("arbitration", config.judges.arbitration),
        ("expert_review", config.judges.expert_review),
    ):
        manifest_path = root / f"{tier}_manifest.json"
        if not manifest_path.is_file():
            raise JudgeWorkflowError(f"missing selective review manifest: {manifest_path}")
        tier_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tasks = _read_jsonl(root / "tasks" / f"{tier}.jsonl")
        results = _load_results(
            _result_directory(root, model=model, task_set=tier, split=None)
        )
        filtered = 0
        if tier == "expert_review":
            runtime_path = (
                root / "private" / "expert_review_runtime_exclusions.jsonl"
            )
            if runtime_path.is_file():
                filtered = len(_read_jsonl(runtime_path))
        if len(results) + filtered != len(tasks):
            raise JudgeWorkflowError(f"unresolved selective review tier: {tier}")
        provider_filtered_total += filtered
        review[tier] = {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "registered_tasks": len(tasks),
            "completed_tasks": len(results),
            "provider_filtered_tasks": filtered,
            "included_in_primary_estimator": False,
            "model": model,
            "frozen_task_sha256": tier_manifest.get("task_sha256"),
        }
    human_manifest_path = root / "human_audit" / "manifest.json"
    if not human_manifest_path.is_file():
        raise JudgeWorkflowError("human audit template/manifest is missing")
    response_seals = _response_seals(root, config=config)

    report = {
        "schema_version": 1,
        "protocol_version": config.protocol_version,
        "study_design_version": config.study_design_version,
        "experiment_name": config.experiment_name,
        "result_status": (
            "post_hoc_descriptive_analysis_valid"
            if analysis_valid
            else "post_hoc_descriptive_guardrail_failure"
        ),
        "complete": True,
        "llm_as_judge_run": True,
        "descriptive_only": True,
        "confirmatory_interpretation_allowed": False,
        "analysis_valid": analysis_valid,
        "methods": manifest["methods"],
        "protocol_amendment": protocol_amendment,
        "primary_contrasts": config.analysis.primary_contrasts,
        "primary_estimand": (
            "equal-concept macro mean of prompt-clustered paired differences, "
            "averaging registered judges and decodings within prompt"
        ),
        "important_interpretation": (
            "Target expression is separate from transparent coherence/relevance "
            "quality and refusal guardrails; no composite primary score is used."
        ),
        "calibration": {
            "passed": True,
            "path": str(calibration_path),
            "sha256": sha256_file(calibration_path),
            "gates": calibration["gates"],
        },
        "pointwise": {
            "judge_models": models,
            "rows": len(point_rows),
            "concept_effects": concept_effects,
            "macro_effects": macro_effects,
            "cross_judge_agreement": point_agreement,
            "invalid": {
                "count_mean_across_judges": sum(row["invalid"] for row in point_rows),
                "rate": invalid_rate,
                "rate_by_judge": invalid_by_judge,
                "maximum_registered_rate": invalid_maximum,
                "guardrail_passed": invalid_passed,
            },
            "refusal": {
                "count_mean_across_judges": sum(row["refusal"] for row in point_rows),
                "rate": refusal_rate,
            },
            "unblinded_rows": str(point_path),
            "unblinded_rows_sha256": sha256_file(point_path),
        },
        "pairwise": {
            **pairwise,
            "unblinded_rows": str(pair_path),
            "unblinded_rows_sha256": sha256_file(pair_path),
        },
        "primary_guardrails": primary_guardrails,
        "all_primary_guardrails_passed": all_primary_guardrails,
        "selective_review_sensitivity": review,
        "missing_tasks": 0,
        "provider_filtered_tasks": provider_filtered_total,
        "analysis_policy": {
            "unit_of_resampling": "prompt_id cluster",
            "decodings_nested_within_prompt": True,
            "concept_aggregation": "equal-weight macro average",
            "bootstrap_samples": config.analysis.bootstrap_samples,
            "permutation_samples": config.analysis.permutation_samples,
            "multiple_comparison_correction": (
                config.analysis.multiple_comparison_correction
            ),
            "post_hoc_due_to_prior_test_exposure": True,
            "test_data_used_for_method_selection": False,
        },
    }
    report_path = analysis_dir / "report.json"
    atomic_write_json(report_path, report)
    task_hashes = {
        name: {
            "public_sha256": artifact["sha256"],
            "private_sha256": artifact["private_map_sha256"],
        }
        for name, artifact in manifest["task_artifacts"].items()
    }
    index = {
        "schema_version": 1,
        "complete": True,
        "llm_as_judge_run": True,
        "descriptive_only": True,
        "analysis_valid": analysis_valid,
        "experiment_name": config.experiment_name,
        "methods": manifest["methods"],
        "primary_contrasts": config.analysis.primary_contrasts,
        "calibration_passed": True,
        "guardrails": {
            "invalid_passed": invalid_passed,
            "primary": primary_guardrails,
            "all_primary_passed": all_primary_guardrails,
        },
        "counts": {
            "pointwise_test": len(point_rows),
            "pairwise_order_groups_test": len(pair_rows),
            "missing": 0,
            "invalid_mean_across_judges": sum(
                row["invalid"] for row in point_rows
            ),
            "refusal_mean_across_judges": sum(
                row["refusal"] for row in point_rows
            ),
            "provider_filtered": provider_filtered_total,
        },
        "position_bias_rate": pairwise["position_bias_rate"],
        "cross_judge_agreement": {
            "pointwise": point_agreement,
            "pairwise_among_stable": pairwise[
                "cross_judge_agreement_among_stable"
            ],
        },
        "hashes": {
            "config_sha256": manifest["config_sha256"],
            "rubric_sha256": protocol_amendment["amended_rubric_sha256"],
            "implementation_combined_sha256": protocol_amendment[
                "amended_implementation_combined_sha256"
            ],
            "parent_rubric_sha256": manifest["rubric_sha256"],
            "parent_implementation_combined_sha256": manifest["implementation"][
                "combined_sha256"
            ],
            "protocol_amendment_sha256": protocol_amendment["sha256"],
            "manifest_sha256": sha256_file(root / "manifest.json"),
            "source_registry_sha256": _canonical_hash(
                manifest["source_registry"]
            ),
            "source_validation_sha256": _canonical_hash(source_validation),
            "task_hashes": task_hashes,
            "response_artifacts": response_seals,
            "response_artifacts_sha256": _canonical_hash(response_seals),
            "calibration_sha256": sha256_file(calibration_path),
            "human_audit_manifest_sha256": sha256_file(human_manifest_path),
            "report_sha256": sha256_file(report_path),
            "pointwise_rows_sha256": sha256_file(point_path),
            "pairwise_rows_sha256": sha256_file(pair_path),
        },
        "report": str(report_path),
    }
    atomic_write_json(analysis_dir / "index.json", index)
    atomic_write_json(root / "index.json", index)
    return index
