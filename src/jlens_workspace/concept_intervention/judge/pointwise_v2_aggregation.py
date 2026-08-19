"""Pointwise-only aggregation for the post-calibration three-method v2 study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from jlens_workspace.artifacts import atomic_write_json, sha256_file
from jlens_workspace.concept_intervention.evaluation import atomic_write_jsonl
from jlens_workspace.concept_intervention.judge.prompts import rubric_hash
from jlens_workspace.concept_intervention.judge.statistics import agreement_metrics
from jlens_workspace.concept_intervention.judge.three_method_aggregation import (
    _assert_complete_results,
    _pointwise_analysis,
    _pointwise_rows,
)
from jlens_workspace.concept_intervention.judge.three_method_config import (
    JudgeEvaluationConfig,
    load_judge_config,
)
from jlens_workspace.concept_intervention.judge.three_method_prepare import (
    validate_evaluation,
)
from jlens_workspace.concept_intervention.judge.workflow import (
    JudgeWorkflowError,
    _canonical_hash,
    _load_results,
    _read_jsonl,
    _result_directory,
)


def _verify_registration(
    config_path: Path,
    *,
    config: JudgeEvaluationConfig,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the frozen registration, implementation, sources, and parent failure."""

    manifest_path = root / "manifest.json"
    frozen_path = root / "frozen_config.yaml"
    if not manifest_path.is_file() or not frozen_path.is_file():
        raise JudgeWorkflowError("v2 frozen registration is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("config_sha256") != sha256_file(config_path)
        or manifest.get("frozen_config_sha256") != sha256_file(frozen_path)
        or sha256_file(frozen_path) != sha256_file(config_path)
        or manifest.get("task_sets") != ["pointwise"]
        or set(manifest.get("task_artifacts", {}))
        != {"pointwise_validation", "pointwise_test"}
    ):
        raise JudgeWorkflowError("v2 frozen config or task-set identity mismatch")
    implementation_root = Path(__file__).parent
    implementation_files = {
        path.name: sha256_file(path)
        for path in sorted(implementation_root.glob("*.py"))
    }
    if (
        manifest.get("implementation", {}).get("files") != implementation_files
        or manifest.get("implementation", {}).get("combined_sha256")
        != _canonical_hash(implementation_files)
        or manifest.get("rubric_sha256") != rubric_hash()
    ):
        raise JudgeWorkflowError("v2 implementation or rubric changed after preparation")
    for artifact in manifest["task_artifacts"].values():
        for path_key, hash_key in (
            ("path", "sha256"),
            ("private_map", "private_map_sha256"),
        ):
            path = Path(artifact[path_key])
            if not path.is_file() or sha256_file(path) != artifact[hash_key]:
                raise JudgeWorkflowError(f"v2 task artifact seal mismatch: {path}")
    source_validation = validate_evaluation(config_path)
    if (
        manifest.get("adaptation") != source_validation.get("adaptation")
        or source_validation.get("task_sets") != ["pointwise"]
    ):
        raise JudgeWorkflowError("v2 parent-adaptation seal changed")
    return manifest, source_validation


def _response_seals(
    root: Path,
    *,
    config: JudgeEvaluationConfig,
) -> dict[str, Any]:
    """Seal exactly the two judges' validation and test pointwise responses."""

    output: dict[str, Any] = {}
    for role, model in (
        ("primary", config.judges.primary),
        ("secondary", config.judges.secondary),
    ):
        for split in ("validation", "test"):
            destination = _result_directory(
                root,
                model=model,
                task_set="pointwise",
                split=split,
            )
            index_path = destination / "index.json"
            if not index_path.is_file():
                raise JudgeWorkflowError(f"missing v2 response index: {index_path}")
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if index.get("complete") is not True:
                raise JudgeWorkflowError(f"incomplete v2 response index: {index_path}")
            files = {
                path.name: sha256_file(path)
                for path in sorted(destination.glob("*.json"))
                if path.name != "index.json"
            }
            output[f"{role}:pointwise:{split}"] = {
                "index": str(index_path),
                "index_sha256": sha256_file(index_path),
                "responses": len(files),
                "response_file_hashes_sha256": _canonical_hash(files),
            }
    return output


def aggregate_evaluation(config_path: str | Path) -> dict[str, Any]:
    """Produce a descriptive pointwise-only report with no pairwise inference."""

    registration = Path(config_path)
    config = load_judge_config(registration)
    if config.study_design_version != "three_method_pointwise_judge_v2":
        raise JudgeWorkflowError("pointwise v2 aggregation requires the v2 registration")
    root = Path(config.output_dir)
    manifest, source_validation = _verify_registration(
        registration,
        config=config,
        root=root,
    )
    calibration_path = root / "calibration.json"
    if not calibration_path.is_file():
        raise JudgeWorkflowError("v2 calibration artifact is missing")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if (
        calibration.get("passed") is not True
        or calibration.get("study_design_version")
        != "three_method_pointwise_judge_v2"
        or calibration.get("not_applicable_gates")
        != [
            "pairwise_order_consistency",
            "pairwise_cross_judge_agreement",
        ]
    ):
        raise JudgeWorkflowError("v2 pointwise calibration did not pass")

    models = {
        "primary": config.judges.primary,
        "secondary": config.judges.secondary,
    }
    tasks = {
        str(task["task_id"]): task
        for task in _read_jsonl(root / "tasks" / "pointwise_test.jsonl")
    }
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for role, model in models.items():
        results[role] = _load_results(
            _result_directory(
                root,
                model=model,
                task_set="pointwise",
                split="test",
            )
        )
        _assert_complete_results(
            tasks,
            results[role],
            model=model,
            role=role,
            protocol_amendment=None,
        )
    point_rows = _pointwise_rows(
        tasks,
        _read_jsonl(root / "private" / "pointwise_test_map.jsonl"),
        results,
    )
    concept_effects, macro_effects = _pointwise_analysis(point_rows, config=config)
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    point_path = analysis_dir / "pointwise_unblinded.jsonl"
    atomic_write_jsonl(point_path, point_rows)

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
    agreement = {
        metric: agreement_metrics(
            [
                float(results["primary"][task_id]["judgment"][metric])
                for task_id in sorted(tasks)
            ],
            [
                float(results["secondary"][task_id]["judgment"][metric])
                for task_id in sorted(tasks)
            ],
        )
        for metric in ("target_expression", "coherence", "relevance")
    }
    agreement["refusal_exact"] = float(
        np.mean(
            [
                results["primary"][task_id]["judgment"]["refusal"]
                == results["secondary"][task_id]["judgment"]["refusal"]
                for task_id in sorted(tasks)
            ]
        )
    )
    agreement["invalid_exact"] = float(
        np.mean(
            [
                results["primary"][task_id]["judgment"]["invalid"]
                == results["secondary"][task_id]["judgment"]["invalid"]
                for task_id in sorted(tasks)
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
        gate["quality_passed"] and gate["refusal_passed"]
        for gate in primary_guardrails.values()
    )
    analysis_valid = invalid_passed and all_primary_guardrails
    human_manifest_path = root / "human_audit" / "manifest.json"
    if not human_manifest_path.is_file():
        raise JudgeWorkflowError("v2 human audit template/manifest is missing")
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
        "adaptation": manifest["adaptation"],
        "response_reuse": False,
        "primary_contrasts": config.analysis.primary_contrasts,
        "primary_estimand": (
            "equal-concept macro mean of prompt-clustered paired pointwise "
            "differences, averaging registered judges and decodings within prompt"
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
            "cross_judge_agreement": agreement,
            "invalid": {
                "rate": invalid_rate,
                "rate_by_judge": invalid_by_judge,
                "maximum_registered_rate": invalid_maximum,
                "guardrail_passed": invalid_passed,
            },
            "refusal": {"rate": refusal_rate},
            "unblinded_rows": str(point_path),
            "unblinded_rows_sha256": sha256_file(point_path),
        },
        "pairwise": {
            "in_scope": False,
            "reason": "excluded by the post-calibration pointwise-only v2 design",
        },
        "selective_review_sensitivity": {
            "in_scope": False,
            "reason": "no arbitration or expert API tasks are registered in v2",
        },
        "primary_guardrails": primary_guardrails,
        "all_primary_guardrails_passed": all_primary_guardrails,
        "missing_tasks": 0,
        "provider_filtered_tasks": 0,
        "analysis_policy": {
            "unit_of_resampling": "prompt_id cluster",
            "decodings_nested_within_prompt": True,
            "concept_aggregation": "equal-weight macro average",
            "bootstrap_samples": config.analysis.bootstrap_samples,
            "permutation_samples": config.analysis.permutation_samples,
            "multiple_comparison_correction": (
                config.analysis.multiple_comparison_correction
            ),
            "post_hoc_due_to_prior_validation_exposure": True,
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
        "task_sets": ["pointwise"],
        "primary_contrasts": config.analysis.primary_contrasts,
        "calibration_passed": True,
        "guardrails": {
            "invalid_passed": invalid_passed,
            "primary": primary_guardrails,
            "all_primary_passed": all_primary_guardrails,
        },
        "counts": {
            "pointwise_test": len(point_rows),
            "pairwise_test": 0,
            "review_tasks": 0,
            "missing": 0,
            "provider_filtered": 0,
        },
        "cross_judge_agreement": {"pointwise": agreement},
        "hashes": {
            "config_sha256": manifest["config_sha256"],
            "rubric_sha256": rubric_hash(),
            "implementation_combined_sha256": manifest["implementation"][
                "combined_sha256"
            ],
            "manifest_sha256": sha256_file(root / "manifest.json"),
            "source_registry_sha256": _canonical_hash(
                manifest["source_registry"]
            ),
            "source_validation_sha256": _canonical_hash(source_validation),
            "adaptation_sha256": _canonical_hash(manifest["adaptation"]),
            "task_hashes": task_hashes,
            "response_artifacts": response_seals,
            "response_artifacts_sha256": _canonical_hash(response_seals),
            "calibration_sha256": sha256_file(calibration_path),
            "human_audit_manifest_sha256": sha256_file(human_manifest_path),
            "report_sha256": sha256_file(report_path),
            "pointwise_rows_sha256": sha256_file(point_path),
        },
        "report": str(report_path),
    }
    atomic_write_json(analysis_dir / "index.json", index)
    atomic_write_json(root / "index.json", index)
    return index
