"""Unblind and aggregate the fixed two-judge primary estimator."""

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
from jlens_workspace.concept_intervention.judge.config import load_judge_config
from jlens_workspace.concept_intervention.judge.prompts import rubric_hash
from jlens_workspace.concept_intervention.judge.statistics import (
    benjamini_hochberg,
    hierarchical_macro_summary,
    paired_cluster_summary,
)
from jlens_workspace.concept_intervention.judge.workflow import (
    JudgeWorkflowError,
    _canonical_hash,
    _load_results,
    _normalized_pair_preference,
    _read_jsonl,
    _result_directory,
)


def _assert_complete_results(
    tasks: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    *,
    model: str,
    role: str,
) -> None:
    if set(results) != set(tasks):
        missing = sorted(set(tasks) - set(results))
        extra = sorted(set(results) - set(tasks))
        raise JudgeWorkflowError(
            f"incomplete {role} results: missing={len(missing)}, extra={len(extra)}"
        )
    expected_rubric = rubric_hash()
    for task_id, task in tasks.items():
        result = results[task_id]
        if (
            result.get("task_sha256") != _canonical_hash(task)
            or result.get("rubric_sha256") != expected_rubric
            or result.get("requested_model") != model
            or result.get("returned_model") != model
            or result.get("judge_role") != role
        ):
            raise JudgeWorkflowError(f"result identity mismatch for {task_id}")


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
            key=lambda value: (value[0], value[1], -1 if value[2] is None else value[2]),
        )
    ]


def _effect_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    config: Any,
) -> dict[str, Any]:
    return paired_cluster_summary(
        records,
        seed=seed,
        bootstrap_samples=config.analysis.bootstrap_samples,
        permutation_samples=config.analysis.permutation_samples,
        confidence_level=config.analysis.confidence_level,
    )


def _pairwise_analysis(
    private_rows: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    concept_ids: Sequence[str],
) -> dict[str, Any]:
    private_by_id = {str(row["task_id"]): row for row in private_rows}
    grouped: dict[str, list[str]] = defaultdict(list)
    group_metadata: dict[str, Mapping[str, Any]] = {}
    for task_id, private in private_by_id.items():
        group = str(private["pair_group_id"])
        grouped[group].append(task_id)
        group_metadata[group] = private
    rows: list[dict[str, Any]] = []
    per_judge_consistency: dict[str, list[bool]] = defaultdict(list)
    per_judge_quality_consistency: dict[str, list[bool]] = defaultdict(list)
    for group in sorted(grouped):
        task_ids = grouped[group]
        if len(task_ids) != 2:
            raise JudgeWorkflowError(f"pair group does not contain two orders: {group}")
        stable_preferences: dict[str, str | None] = {}
        order_preferences: dict[str, list[str]] = {}
        stable_quality_preferences: dict[str, str | None] = {}
        quality_order_preferences: dict[str, list[str]] = {}
        for role, role_results in results.items():
            preferences = [
                _normalized_pair_preference(role_results[task_id], private_by_id[task_id])
                for task_id in task_ids
            ]
            order_preferences[role] = preferences
            consistent = preferences[0] == preferences[1]
            per_judge_consistency[role].append(consistent)
            stable_preferences[role] = preferences[0] if consistent else None
            quality_preferences = [
                _normalized_pair_preference(
                    role_results[task_id], private_by_id[task_id], "quality_preference"
                )
                for task_id in task_ids
            ]
            quality_order_preferences[role] = quality_preferences
            quality_consistent = quality_preferences[0] == quality_preferences[1]
            per_judge_quality_consistency[role].append(quality_consistent)
            stable_quality_preferences[role] = (
                quality_preferences[0] if quality_consistent else None
            )
        stable_values = [value for value in stable_preferences.values() if value is not None]
        consensus = (
            stable_values[0]
            if len(stable_values) == len(results) and len(set(stable_values)) == 1
            else "discordant"
        )
        stable_quality_values = [
            value for value in stable_quality_preferences.values() if value is not None
        ]
        quality_consensus = (
            stable_quality_values[0]
            if len(stable_quality_values) == len(results) and len(set(stable_quality_values)) == 1
            else "discordant"
        )
        metadata = group_metadata[group]
        rows.append(
            {
                "pair_group_id": group,
                "concept_id": metadata["concept_id"],
                "prompt_id": metadata["prompt_id"],
                "decoding": metadata["decoding"],
                "seed": metadata.get("seed"),
                "order_preferences": order_preferences,
                "stable_preferences": stable_preferences,
                "consensus": consensus,
                "quality_order_preferences": quality_order_preferences,
                "stable_quality_preferences": stable_quality_preferences,
                "quality_consensus": quality_consensus,
            }
        )
    by_concept: dict[str, Any] = {}
    for concept_id in concept_ids:
        concept_rows = [row for row in rows if row["concept_id"] == concept_id]
        counts = {
            label: sum(row["consensus"] == label for row in concept_rows)
            for label in ("j", "raptor", "tie", "discordant")
        }
        quality_counts = {
            label: sum(row["quality_consensus"] == label for row in concept_rows)
            for label in ("j", "raptor", "tie", "discordant")
        }
        decisive = counts["j"] + counts["raptor"]
        by_concept[concept_id] = {
            "pairs": len(concept_rows),
            "consensus_counts": counts,
            "quality_consensus_counts": quality_counts,
            "j_preference_rate_among_decisive": (counts["j"] / decisive if decisive else None),
            "two_sided_binomial_p": (
                float(binomtest(counts["j"], decisive, 0.5).pvalue) if decisive else None
            ),
        }
    macro_counts = {
        label: sum(row["consensus"] == label for row in rows)
        for label in ("j", "raptor", "tie", "discordant")
    }
    macro_quality_counts = {
        label: sum(row["quality_consensus"] == label for row in rows)
        for label in ("j", "raptor", "tie", "discordant")
    }
    decisive = macro_counts["j"] + macro_counts["raptor"]
    return {
        "pair_rows": rows,
        "order_consistency": {
            role: float(np.mean(values)) if values else 0.0
            for role, values in per_judge_consistency.items()
        },
        "position_bias_rate": {
            role: 1.0 - float(np.mean(values)) if values else 1.0
            for role, values in per_judge_consistency.items()
        },
        "quality_order_consistency": {
            role: float(np.mean(values)) if values else 0.0
            for role, values in per_judge_quality_consistency.items()
        },
        "by_concept": by_concept,
        "macro": {
            "pairs": len(rows),
            "consensus_counts": macro_counts,
            "quality_consensus_counts": macro_quality_counts,
            "j_preference_rate_among_decisive": (
                macro_counts["j"] / decisive if decisive else None
            ),
            "two_sided_binomial_p": (
                float(binomtest(macro_counts["j"], decisive, 0.5).pvalue) if decisive else None
            ),
        },
    }


def aggregate_evaluation(config_path: str | Path) -> dict[str, Any]:
    """Compute pre-registered pointwise effects and order-swapped pairwise checks."""

    config = load_judge_config(config_path)
    root = Path(config.output_dir)
    calibration_path = root / "calibration.json"
    if not calibration_path.is_file():
        raise JudgeWorkflowError("calibration artifact is missing")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not calibration.get("passed"):
        raise JudgeWorkflowError("cannot aggregate a study that failed calibration")
    point_tasks = {
        str(task["task_id"]): task for task in _read_jsonl(root / "tasks" / "pointwise_test.jsonl")
    }
    pair_tasks = {
        str(task["task_id"]): task for task in _read_jsonl(root / "tasks" / "pairwise_test.jsonl")
    }
    models = {
        "primary": config.judges.primary,
        "secondary": config.judges.secondary,
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
        _assert_complete_results(point_tasks, point_results[role], model=model, role=role)
        _assert_complete_results(pair_tasks, pair_results[role], model=model, role=role)
    private_point = {
        str(row["task_id"]): row
        for row in _read_jsonl(root / "private" / "pointwise_test_map.jsonl")
    }
    if set(private_point) != set(point_tasks):
        raise JudgeWorkflowError("pointwise private map differs from public tasks")
    unblinded: list[dict[str, Any]] = []
    for task_id in sorted(point_tasks):
        judgments = {role: point_results[role][task_id]["judgment"] for role in models}
        target = float(np.mean([judgment["target_expression"] for judgment in judgments.values()]))
        coherence = float(np.mean([judgment["coherence"] for judgment in judgments.values()]))
        relevance = float(np.mean([judgment["relevance"] for judgment in judgments.values()]))
        refusal = float(np.mean([float(judgment["refusal"]) for judgment in judgments.values()]))
        invalid = float(np.mean([float(judgment["invalid"]) for judgment in judgments.values()]))
        private = private_point[task_id]
        unblinded.append(
            {
                "task_id": task_id,
                "concept_id": private["concept_id"],
                "condition_role": private["condition_role"],
                "method": private["method"],
                "condition_id": private["condition_id"],
                "grid_point": private["grid_point"],
                "prompt_id": private["prompt_id"],
                "decoding": private["decoding"],
                "seed": private.get("seed"),
                "target_expression": target,
                "coherence": coherence,
                "relevance": relevance,
                "quality": (coherence + relevance) / 2.0,
                "refusal": refusal,
                "invalid": invalid,
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
    invalid_rate = float(np.mean([row["invalid"] for row in unblinded]))
    invalid_tolerance = 1.0 - config.calibration.min_completion_rate
    invalid_guardrail_passed = invalid_rate <= invalid_tolerance
    invalid_rate_by_judge = {
        role: float(
            np.mean(
                [float(row["judge_scores"][role]["invalid"]) for row in unblinded]
            )
        )
        for role in models
    }
    any_judge_invalid_rate = float(
        np.mean(
            [
                any(bool(row["judge_scores"][role]["invalid"]) for role in models)
                for row in unblinded
            ]
        )
    )
    consensus_invalid_rate = float(
        np.mean(
            [
                all(bool(row["judge_scores"][role]["invalid"]) for role in models)
                for row in unblinded
            ]
        )
    )
    if (
        not invalid_guardrail_passed
        and not config.analysis.report_after_invalid_guardrail_failure
    ):
        raise JudgeWorkflowError(
            f"invalid judgment rate {invalid_rate:.4f} exceeds the registered tolerance"
        )
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    unblinded_path = analysis_dir / "pointwise_unblinded.jsonl"
    atomic_write_jsonl(unblinded_path, unblinded)
    by_concept_role: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in unblinded:
        by_concept_role[str(row["concept_id"])][str(row["condition_role"])].append(row)
    random_roles = [f"random_{seed}" for seed in config.selection.random_control_seeds]
    baseline_contrasts = ["full", "j", "non_j", *random_roles, "raptor"]
    contrast_specs = {
        **{role: (role, "baseline") for role in baseline_contrasts},
        "j_minus_raptor": ("j", "raptor"),
    }
    concept_effects: dict[str, Any] = {}
    macro_inputs: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    p_values: dict[str, float] = {}
    for concept_index, concept_id in enumerate(config.source.concept_ids):
        concept_effects[concept_id] = {}
        roles = by_concept_role[concept_id]
        for contrast_index, (contrast, (treatment, reference)) in enumerate(contrast_specs.items()):
            metrics: dict[str, Any] = {}
            for metric_index, metric in enumerate(("target_expression", "quality", "refusal")):
                records = _paired_records(roles[treatment], roles[reference], metric=metric)
                summary = _effect_summary(
                    records,
                    seed=(
                        config.seed
                        + 100_000 * concept_index
                        + 1_000 * contrast_index
                        + metric_index
                    ),
                    config=config,
                )
                metrics[metric] = summary
                macro_inputs[f"{contrast}:{metric}"][concept_id] = summary["cluster_means"]
                if metric == "target_expression":
                    p_values[f"{concept_id}:{contrast}"] = summary["p_value_two_sided"]
            metrics["quality_guardrail_passed"] = (
                metrics["quality"]["ci_lower"] >= config.analysis.quality_noninferiority_margin
            )
            metrics["refusal_guardrail_passed"] = (
                metrics["refusal"]["ci_upper"] <= config.analysis.refusal_rate_margin
            )
            concept_effects[concept_id][contrast] = metrics
    adjusted = benjamini_hochberg(
        {key: value for key, value in p_values.items() if not key.endswith(":j_minus_raptor")}
    )
    for key, q_value in adjusted.items():
        concept_id, contrast = key.rsplit(":", 1)
        concept_effects[concept_id][contrast]["target_expression"]["fdr_q_value"] = q_value
    macro: dict[str, Any] = {}
    for contrast_index, contrast in enumerate(contrast_specs):
        macro[contrast] = {}
        for metric_index, metric in enumerate(("target_expression", "quality", "refusal")):
            macro[contrast][metric] = hierarchical_macro_summary(
                macro_inputs[f"{contrast}:{metric}"],
                seed=config.seed + 1_000_000 + 10 * contrast_index + metric_index,
                bootstrap_samples=config.analysis.bootstrap_samples,
                confidence_level=config.analysis.confidence_level,
            )
        macro[contrast]["quality_guardrail_passed"] = (
            macro[contrast]["quality"]["ci_lower"] >= config.analysis.quality_noninferiority_margin
        )
        macro[contrast]["refusal_guardrail_passed"] = (
            macro[contrast]["refusal"]["ci_upper"] <= config.analysis.refusal_rate_margin
        )
    pair_private = _read_jsonl(root / "private" / "pairwise_test_map.jsonl")
    pairwise = _pairwise_analysis(
        pair_private,
        pair_results,
        concept_ids=config.source.concept_ids,
    )
    pair_rows = pairwise.pop("pair_rows")
    pair_path = analysis_dir / "pairwise_unblinded.jsonl"
    atomic_write_jsonl(pair_path, pair_rows)
    review = {}
    for tier, model in (
        ("arbitration", config.judges.arbitration),
        ("expert_review", config.judges.expert_review),
    ):
        results = _load_results(
            _result_directory(
                root,
                model=model,
                task_set=tier,
                split=None,
            )
        )
        review_task_path = root / "tasks" / f"{tier}.jsonl"
        registered_tasks = len(_read_jsonl(review_task_path)) if review_task_path.is_file() else 0
        provider_filtered = 0
        if tier == "expert_review":
            runtime_path = root / "private" / "expert_review_runtime_exclusions.jsonl"
            if runtime_path.is_file():
                provider_filtered = len(_read_jsonl(runtime_path))
        if len(results) + provider_filtered != registered_tasks:
            raise JudgeWorkflowError(
                "selective review is unresolved: "
                f"{tier} machine={len(results)} filtered={provider_filtered} "
                f"registered={registered_tasks}"
            )
        review[tier] = {
            "completed_tasks": len(results),
            "registered_tasks": registered_tasks,
            "provider_filtered_tasks": provider_filtered,
            "resolved_tasks": len(results) + provider_filtered,
            "model": model,
            "included_in_primary_estimator": False,
        }
    primary = macro[config.analysis.primary_contrast]
    report = {
        "schema_version": 1,
        "protocol_version": config.protocol_version,
        "experiment_name": config.experiment_name,
        "calibration": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "primary_estimand": (
            "macro mean paired J minus RAPTOR target-expression score on held-out "
            "open test prompts, averaged across the two registered judges"
        ),
        "result_status": (
            "confirmatory"
            if invalid_guardrail_passed
            else "descriptive_only_after_registered_invalid_guardrail_failure"
        ),
        "confirmatory_interpretation_allowed": invalid_guardrail_passed,
        "primary_result": primary,
        "primary_quality_guardrails": {
            "quality_noninferiority_margin": config.analysis.quality_noninferiority_margin,
            "quality_passed": primary["quality_guardrail_passed"],
            "refusal_rate_margin": config.analysis.refusal_rate_margin,
            "refusal_passed": primary["refusal_guardrail_passed"],
        },
        "important_interpretation": (
            "target expression and task quality are reported separately; no opaque "
            "multiplicative or composite primary score is used"
        ),
        "pointwise": {
            "judge_models": models,
            "invalid_rate": invalid_rate,
            "invalid_guardrail": {
                "passed": invalid_guardrail_passed,
                "mean_judge_invalid_rate": invalid_rate,
                "maximum_registered_rate": invalid_tolerance,
                "any_judge_invalid_rate": any_judge_invalid_rate,
                "consensus_invalid_rate": consensus_invalid_rate,
                "invalid_rate_by_judge": invalid_rate_by_judge,
                "report_after_failure_enabled": (
                    config.analysis.report_after_invalid_guardrail_failure
                ),
            },
            "concept_effects": concept_effects,
            "macro_effects": macro,
            "unblinded_rows": str(unblinded_path),
            "unblinded_rows_sha256": sha256_file(unblinded_path),
        },
        "pairwise": {
            **pairwise,
            "unblinded_rows": str(pair_path),
            "unblinded_rows_sha256": sha256_file(pair_path),
        },
        "selective_review_sensitivity": review,
        "analysis_policy": {
            "unit_of_resampling": "prompt_id cluster",
            "decodings_nested_within_prompt": True,
            "concept_aggregation": "equal-weight macro average",
            "bootstrap_samples": config.analysis.bootstrap_samples,
            "permutation_samples": config.analysis.permutation_samples,
            "multiple_comparison_correction": config.analysis.multiple_comparison_correction,
            "test_data_used_for_selection": False,
        },
    }
    report_path = analysis_dir / "report.json"
    atomic_write_json(report_path, report)
    index = {
        "schema_version": 1,
        "complete": True,
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "pointwise_rows": str(unblinded_path),
        "pointwise_rows_sha256": sha256_file(unblinded_path),
        "pairwise_rows": str(pair_path),
        "pairwise_rows_sha256": sha256_file(pair_path),
        "llm_as_judge_run": True,
        "primary_contrast": config.analysis.primary_contrast,
        "registered_invalid_guardrail_passed": invalid_guardrail_passed,
        "confirmatory_interpretation_allowed": invalid_guardrail_passed,
        "descriptive_only": not invalid_guardrail_passed,
    }
    atomic_write_json(analysis_dir / "index.json", index)
    return index
