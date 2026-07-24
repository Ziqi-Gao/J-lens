from __future__ import annotations

import json
from pathlib import Path

import pytest

from jlens_workspace.artifacts import sha256_file
from jlens_workspace.concept_intervention.comparison import (
    InterventionComparisonError,
    rebuild_intervention_comparison,
    validate_three_method_smokes,
)
from jlens_workspace.concept_intervention.generation import (
    build_target_artifact_seal,
    prompt_ids_sha256,
    write_generation_artifacts,
)

_GENERATION_SETTINGS = {
    "sample_seeds": [1001, 2002, 3003],
    "max_new_tokens": 128,
    "temperature": 0.75,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
}
_SYNTHETIC_LABELS = {
    f"concept:{letter}": letter for letter in "abcdefg"
}
_GOEMOTION_LABELS = {
    "goemotions:admiration": "admiration",
    "goemotions:approval": "approval",
    "goemotions:curiosity": "curiosity",
    "goemotions:disapproval": "rejection",
    "goemotions:gratitude": "gratitude",
    "goemotions:love": "love",
    "goemotions:optimism": "optimism",
}
_GENERATION_CONTRACT = {
    "schema_version": 1,
    "prompt_ids": ["classify_0"],
    "prompt_ids_sha256": prompt_ids_sha256(["classify_0"]),
    "prompt_count": 1,
    "candidate_prompt_ids": ["classify_0"],
    "candidate_prompt_splits": {"classify_0": "test"},
    "candidate_labels": _SYNTHETIC_LABELS,
    "open_prompt_ids": [],
    "sample_seeds": [1001, 2002, 3003],
    "decodings_per_prompt": 4,
    "expected_rows": 4,
    "generation_settings": _GENERATION_SETTINGS,
}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_iti_direction_metrics(
    path: Path,
    *,
    selected_layers_sha256: str,
    row_manifest_sha256: str,
) -> Path:
    direction_file = path.parent / "iti_native_mass_mean.npz"
    probe_file = path.parent / "iti_native_probe_weight.npz"
    development_file = path.parent / "iti_development_indices.npz"
    direction_file.write_bytes(b"mass mean")
    probe_file.write_bytes(b"probe weight")
    development_file.write_bytes(b"indices")
    files = {
        "variants": {
            "native": {
                "mass_mean": direction_file.name,
                "probe_weight": probe_file.name,
            }
        },
        "random": {},
        "development_indices": development_file.name,
    }
    path.write_text(
        json.dumps(
            {
                "method": "honest_llama_mass_mean_qwen_full_attention",
                "selected_layers_sha256": selected_layers_sha256,
                "row_manifest_sha256": row_manifest_sha256,
                "files": files,
                "files_sha256": {
                    "variants": {
                        "native": {
                            "mass_mean": sha256_file(direction_file),
                            "probe_weight": sha256_file(probe_file),
                        }
                    },
                    "random": {},
                    "development_indices": sha256_file(development_file),
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _score(
    prompt: str,
    split: str,
    margin: float,
    **condition: object,
) -> dict[str, object]:
    return {
        "prompt_id": prompt,
        "evaluation_split": split,
        "target_margin": margin,
        "candidate_log_probabilities": {"concept:a": -1.0, "other": -2.0},
        **condition,
    }


def _write_baseline_generation(
    root: Path,
    *,
    method: str,
    condition_id: str,
    grid_condition: dict[str, object],
) -> dict[str, object]:
    shard = root / "shards" / condition_id
    shard.mkdir(parents=True)
    rows = [
        {
            "generation_id": f"{method}-{decoding}-{seed}",
            "blind_id": f"blind-{method}-{decoding}-{seed}",
            "method": method,
            "concept_id": "concept:a",
            "condition_id": condition_id,
            "grid_point": grid_condition,
            "prompt_id": "classify_0",
            "prompt_text": "Prompt.",
            "decoding": decoding,
            "seed": seed,
            "generated_token_ids": [1],
            "generated_text": "Result.",
            "token_log_probabilities": [-0.5],
            "telemetry": [],
            "injected_norm_by_layer": {},
            "total_injected_norm": 0.0,
            "generation_settings": _GENERATION_SETTINGS,
        }
        for decoding, seed in (
            ("greedy", None),
            ("sample", 1001),
            ("sample", 2002),
            ("sample", 3003),
        )
    ]
    generation_files = write_generation_artifacts(shard, rows)
    summary_path = shard / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "method": method,
                "target_concept_id": "concept:a",
                "grid_condition": grid_condition,
                "generation_files": generation_files,
                "generation_contract": _GENERATION_CONTRACT,
            }
        ),
        encoding="utf-8",
    )
    return {
        "grid_condition": grid_condition,
        "summary": str(summary_path.relative_to(root.parents[1])),
        "summary_sha256": sha256_file(summary_path),
    }


def _write_smoke_shard(
    root: Path,
    *,
    method: str,
    grid_index: int,
    condition: dict[str, object],
) -> None:
    shard = (
        root
        / "targets"
        / "goemotions%3Aadmiration"
        / "shards"
        / f"grid_{grid_index:04d}"
    )
    shard.mkdir(parents=True)
    candidate = shard / "candidate_scores.jsonl"
    candidate_condition = {
        ("mode" if key == "condition" else key): value
        for key, value in condition.items()
    }
    score_condition = {
        ("condition" if key == "mode" else key): value
        for key, value in candidate_condition.items()
    }
    candidate.write_text(
        json.dumps(
            {
                "prompt_id": "classify_0",
                "method": method,
                "target_concept_id": "goemotions:admiration",
                "evaluation_split": "test",
                "candidate_log_probabilities": {
                    "goemotions:admiration": -1.0,
                    **{
                        concept_id: -2.0
                        for concept_id in _GOEMOTION_LABELS
                        if concept_id != "goemotions:admiration"
                    },
                },
                "candidate_probabilities_normalized": {
                    "goemotions:admiration": 0.4,
                    **{
                        concept_id: 0.1
                        for concept_id in _GOEMOTION_LABELS
                        if concept_id != "goemotions:admiration"
                    },
                },
                "target_log_probability": -1.0,
                "target_candidate_probability": 0.4,
                "target_margin": 1.0,
                "target_rank": 1,
                **score_condition,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = []
    layers = [3, 7, 11, 15, 19, 23]
    telemetry = [
        {
            "layer": layer,
            "forward_call": 0,
            "generation_step": 0,
            "batch_size": 1,
            "sequence_length": 1,
            "injected_norm": 1.0,
            **(
                {
                    "active_positions": 1,
                    "strength": float(condition["strength"]),
                    "residual_norm": 2.0,
                    "kind": "addition",
                }
                if method == "j_component_intervention"
                else (
                    {
                        "active_positions": 1,
                        "multiplier": float(condition["strength"]),
                    }
                    if method == "iti_intervention"
                    else {
                        "target_probability": float(
                            condition["target_probability"]
                        ),
                        "pre_intervention_logit": 0.0,
                        "pre_intervention_probability": 0.5,
                        "epsilon": 0.25,
                        "steered": True,
                    }
                )
            ),
        }
        for layer in layers
    ]
    for decoding, seed in (
        ("greedy", None),
        ("sample", 1001),
        ("sample", 2002),
        ("sample", 3003),
    ):
        suffix = f"{decoding}-{seed}"
        rows.append(
            {
                "generation_id": f"generation-{suffix}",
                "blind_id": f"blind-{suffix}",
                "method": method,
                "concept_id": "goemotions:admiration",
                "condition_id": str(condition["condition_id"]),
                "grid_point": condition,
                "intervention_metadata": {
                    "selected_layers": layers,
                    "active_layers": layers,
                    "selected_heads": [
                        {"layer": layer, "head": 0}
                        for layer in layers
                    ],
                },
                "prompt_id": "classify_0",
                "prompt_text": "Prompt.",
                "decoding": decoding,
                "seed": seed,
                "generated_token_ids": [1],
                "generated_text": "Result.",
                "token_log_probabilities": [-0.5],
                "telemetry": telemetry,
                "injected_norm_by_layer": {
                    str(layer): 1.0 for layer in layers
                },
                "total_injected_norm": 6.0,
                "generation_settings": _GENERATION_SETTINGS,
            }
        )
    generation_hashes = write_generation_artifacts(shard, rows)
    (shard / "summary.json").write_text(
        json.dumps(
            {
                "method": method,
                "target_concept_id": "goemotions:admiration",
                "grid_index": grid_index,
                "grid_condition": condition,
                "selected_layers": layers,
                "candidate_scores_sha256": sha256_file(candidate),
                "generation_files": generation_hashes,
                "generation_contract": {
                    **_GENERATION_CONTRACT,
                    "candidate_labels": _GOEMOTION_LABELS,
                },
            }
        ),
        encoding="utf-8",
    )


def test_smoke_gate_validates_registered_shards_before_full_submission(
    tmp_path: Path,
) -> None:
    j_root = tmp_path / "j"
    iti_root = tmp_path / "iti"
    raptor_root = tmp_path / "raptor"
    _write_smoke_shard(
        j_root,
        method="j_component_intervention",
        grid_index=6,
        condition={"condition_id": "full", "strength": 0.5},
    )
    _write_smoke_shard(
        raptor_root,
        method="raptor_intervention",
        grid_index=8,
        condition={
            "condition_id": "target_probability_0.9999",
            "target_probability": 0.9999,
        },
    )
    _write_smoke_shard(
        iti_root,
        method="iti_intervention",
        grid_index=6,
        condition={
            "variant": "native",
            "mode": "mass_mean",
            "random_seed": None,
            "condition_id": "iti_native_mass_mean",
            "top_k": 4,
            "strength": 20.0,
        },
    )
    _write_smoke_shard(
        iti_root,
        method="iti_intervention",
        grid_index=251,
        condition={
            "variant": "layer_matched",
            "mode": "mass_mean",
            "random_seed": None,
            "condition_id": "iti_layer_matched_mass_mean",
            "top_k": 8,
            "strength": 20.0,
        },
    )

    output = tmp_path / "smoke_gate.json"
    result = validate_three_method_smokes(
        output_path=output,
        j_root=j_root,
        iti_root=iti_root,
        raptor_root=raptor_root,
    )

    assert result["complete"] is True
    assert len(result["entries"]) == 4
    assert output.is_file()


def test_comparison_selects_on_validation_and_reports_paired_test_effect(
    tmp_path: Path,
) -> None:
    row_manifest = tmp_path / "row_manifest.json"
    row_manifest.write_text("{}", encoding="utf-8")
    row_manifest_hash = sha256_file(row_manifest)
    shared = tmp_path / "layer_selection.json"
    shared.write_text(
        json.dumps(
            {
                "row_manifest": "row_manifest.json",
                "row_manifest_sha256": row_manifest_hash,
            }
        ),
        encoding="utf-8",
    )
    selection_hash = sha256_file(shared)
    roots = {
        method: tmp_path / method
        for method in (
            "j_component_intervention",
            "iti_intervention",
            "raptor_intervention",
        )
    }
    for root in roots.values():
        root.mkdir()
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "model_id": "model",
                    "model_revision": "revision",
                    "tokenizer_id": "model",
                    "tokenizer_revision": "revision",
                    "lens_source": "local:lens.pt",
                    "dataset_source": "dataset",
                    "dataset_revision": "data-revision",
                    "dataset_hash": "sha256:dataset",
                    "git_commit": "a" * 40,
                    "notes": {
                        "generation": {
                            **_GENERATION_SETTINGS,
                            "candidate_prompts_sha256": "c" * 64,
                            "open_prompts_sha256": "d" * 64,
                            "candidate_labels": _SYNTHETIC_LABELS,
                            "candidate_prompt_splits": {
                                "classify_0": "test"
                            },
                            "candidate_prompt_count": 1,
                            "open_prompt_count": 0,
                            "prompt_count": 1,
                            "prompt_ids_sha256": prompt_ids_sha256(
                                ["classify_0"]
                            ),
                            "expected_rows_per_grid_point": 4,
                        },
                        "lens_sha256": "b" * 64,
                        "selected_layers_sha256": selection_hash,
                        "row_manifest_sha256": row_manifest_hash,
                    },
                }
            ),
            encoding="utf-8",
        )
    target_dirs = {method: root / "targets/concept%3Aa" for method, root in roots.items()}
    for directory in target_dirs.values():
        directory.mkdir(parents=True)
    baseline_shards = {
        "j_component_intervention": _write_baseline_generation(
            target_dirs["j_component_intervention"],
            method="j_component_intervention",
            condition_id="grid_j_zero",
            grid_condition={"condition_id": "full", "strength": 0.0},
        ),
        "raptor_intervention": _write_baseline_generation(
            target_dirs["raptor_intervention"],
            method="raptor_intervention",
            condition_id="grid_raptor_zero",
            grid_condition={"condition_id": "no_hook", "target_probability": None},
        ),
        "iti_intervention": _write_baseline_generation(
            target_dirs["iti_intervention"],
            method="iti_intervention",
            condition_id="grid_iti_zero",
            grid_condition={
                "variant": "native",
                "condition_id": "iti_native_mass_mean",
                "top_k": 4,
                "strength": 0.0,
            },
        ),
    }

    common_summary = {
        "selected_layers": [3, 7, 11, 15, 19, 23],
        "generation_contract": _GENERATION_CONTRACT,
    }
    (target_dirs["j_component_intervention"] / "summary.json").write_text(
        json.dumps(
            {
                **common_summary,
                "source_provenance": {
                    "selected_layers_sha256": selection_hash,
                    "row_manifest_sha256": row_manifest_hash,
                },
                "shards": [baseline_shards["j_component_intervention"]],
            }
        ),
        encoding="utf-8",
    )
    (target_dirs["raptor_intervention"] / "summary.json").write_text(
        json.dumps(
            {
                **common_summary,
                "source_provenance": {
                    "selected_layers_sha256": selection_hash,
                    "row_manifest_sha256": row_manifest_hash,
                },
                "shards": [baseline_shards["raptor_intervention"]],
            }
        ),
        encoding="utf-8",
    )
    direction_metrics = _write_iti_direction_metrics(
        tmp_path / "iti_direction_metrics.json",
        selected_layers_sha256=selection_hash,
        row_manifest_sha256=row_manifest_hash,
    )
    (target_dirs["iti_intervention"] / "summary.json").write_text(
        json.dumps(
            {
                **common_summary,
                "direction_metrics": str(direction_metrics),
                "direction_metrics_sha256": sha256_file(direction_metrics),
                "native_top_k_grid": [4, 8],
                "shards": [baseline_shards["iti_intervention"]],
                "selection": {
                    "native": {"selected": {"top_k": 8, "strength": 5.0}},
                    "layer_matched": {
                        "selected": {"top_k": 8, "strength": 10.0}
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    _write_jsonl(
        target_dirs["j_component_intervention"] / "candidate_scores.jsonl",
        [
            _score("v", "validation", 1.0, condition_id="j", strength=0.125),
            _score("v", "validation", 2.0, condition_id="j", strength=0.25),
            _score("t", "test", 1.0, condition_id="j", strength=0.0),
            _score("t", "test", 1.0, condition_id="full", strength=0.0),
            _score("t", "test", 4.0, condition_id="j", strength=0.25),
        ],
    )
    _write_jsonl(
        target_dirs["raptor_intervention"] / "candidate_scores.jsonl",
        [
            _score("v", "validation", 1.0, target_probability=0.9),
            _score("v", "validation", 2.0, target_probability=0.99),
            _score("t", "test", 2.0, condition_id="no_hook", target_probability=None),
            _score(
                "t",
                "test",
                6.0,
                condition_id="target_probability_0.99",
                target_probability=0.99,
            ),
        ],
    )
    iti_rows = []
    iti_rows.append(
        _score(
            "t",
            "test",
            3.0,
            variant="native",
            condition_id="iti_native_mass_mean",
            top_k=4,
            strength=0.0,
        )
    )
    for variant, strength, baseline, treatment in (
        ("native", 5.0, 3.0, 8.0),
        ("layer_matched", 10.0, 4.0, 10.0),
    ):
        condition = f"iti_{variant}_mass_mean"
        iti_rows.extend(
            [
                _score(
                    "t",
                    "test",
                    baseline,
                    variant=variant,
                    condition_id=condition,
                    top_k=8,
                    strength=0.0,
                ),
                _score(
                    "t",
                    "test",
                    treatment,
                    variant=variant,
                    condition_id=condition,
                    top_k=8,
                    strength=strength,
                ),
            ]
        )
    _write_jsonl(
        target_dirs["iti_intervention"] / "candidate_scores.jsonl", iti_rows
    )
    for method, root in roots.items():
        summary_path = target_dirs[method] / "summary.json"
        (root / "index.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "method": method,
                    "complete": True,
                    "expected_concepts": ["concept:a"],
                    "observed_concepts": ["concept:a"],
                    "manifest_sha256": sha256_file(root / "manifest.json"),
                    "entries": [
                        {
                            "concept_id": "concept:a",
                            "summary": str(summary_path.relative_to(root)),
                            "summary_sha256": sha256_file(summary_path),
                            "artifact_seal": build_target_artifact_seal(
                                root, summary_path
                            ),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    index = rebuild_intervention_comparison(
        output_dir=tmp_path / "intervention_comparison",
        shared_layer_selection=shared,
        j_root=roots["j_component_intervention"],
        iti_root=roots["iti_intervention"],
        raptor_root=roots["raptor_intervention"],
        concept_ids=["concept:a"],
    )
    comparison = json.loads(
        (tmp_path / "intervention_comparison/comparison.json").read_text(
            encoding="utf-8"
        )
    )
    entry = comparison["entries"][0]
    assert index["complete"] is True
    assert entry["j_component"]["held_out_target_margin_effect"] == pytest.approx(3.0)
    assert entry["raptor"]["held_out_target_margin_effect"] == pytest.approx(4.0)
    assert entry["iti_native"]["held_out_target_margin_effect"] == pytest.approx(5.0)
    assert entry["iti_layer_matched"]["held_out_target_margin_effect"] == pytest.approx(6.0)
    assert comparison["llm_as_judge_run"] is False

    direction_file = tmp_path / "iti_native_mass_mean.npz"
    original_direction = direction_file.read_bytes()
    direction_file.write_bytes(b"changed direction")
    with pytest.raises(InterventionComparisonError, match="direction artifact"):
        rebuild_intervention_comparison(
            output_dir=tmp_path / "tampered_direction_comparison",
            shared_layer_selection=shared,
            j_root=roots["j_component_intervention"],
            iti_root=roots["iti_intervention"],
            raptor_root=roots["raptor_intervention"],
            concept_ids=["concept:a"],
        )
    direction_file.write_bytes(original_direction)

    j_scores = (
        target_dirs["j_component_intervention"] / "candidate_scores.jsonl"
    )
    tampered = [
        json.loads(line)
        for line in j_scores.read_text(encoding="utf-8").splitlines()
    ]
    tampered[0]["target_margin"] = 999.0
    _write_jsonl(j_scores, tampered)

    with pytest.raises(InterventionComparisonError, match="artifact"):
        rebuild_intervention_comparison(
            output_dir=tmp_path / "tampered_comparison",
            shared_layer_selection=shared,
            j_root=roots["j_component_intervention"],
            iti_root=roots["iti_intervention"],
            raptor_root=roots["raptor_intervention"],
            concept_ids=["concept:a"],
        )
