from __future__ import annotations

import json
from pathlib import Path

import pytest

from jlens_workspace.artifacts import sha256_file
from jlens_workspace.concept_intervention.comparison import (
    rebuild_intervention_comparison,
    validate_three_method_smokes,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


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
        **condition,
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
    candidate.write_text("{}\n", encoding="utf-8")
    generation_hashes = {}
    for filename, key in (
        ("generations.jsonl", "generations_sha256"),
        ("judge_blind_generations.jsonl", "blind_generations_sha256"),
        ("judge_blind_map.jsonl", "blind_map_sha256"),
    ):
        path = shard / filename
        path.write_text("", encoding="utf-8")
        generation_hashes[key] = sha256_file(path)
    (shard / "summary.json").write_text(
        json.dumps(
            {
                "method": method,
                "target_concept_id": "goemotions:admiration",
                "grid_index": grid_index,
                "grid_condition": condition,
                "candidate_scores_sha256": sha256_file(candidate),
                "generation_files": generation_hashes,
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
    shared = tmp_path / "layer_selection.json"
    shared.write_text("{}", encoding="utf-8")
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
                        "generation": {"max_new_tokens": 128},
                        "lens_sha256": "b" * 64,
                        "selected_layers_sha256": selection_hash,
                    },
                }
            ),
            encoding="utf-8",
        )
        (root / "index.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "manifest_sha256": sha256_file(manifest),
                }
            ),
            encoding="utf-8",
        )
    target_dirs = {method: root / "targets/concept%3Aa" for method, root in roots.items()}
    for directory in target_dirs.values():
        directory.mkdir(parents=True)

    common_summary = {"selected_layers": [3, 7, 11, 15, 19, 23]}
    (target_dirs["j_component_intervention"] / "summary.json").write_text(
        json.dumps(
            {
                **common_summary,
                "source_provenance": {
                    "selected_layers_sha256": selection_hash
                },
            }
        ),
        encoding="utf-8",
    )
    (target_dirs["raptor_intervention"] / "summary.json").write_text(
        json.dumps(
            {
                **common_summary,
                "source_provenance": {
                    "selected_layers_sha256": selection_hash
                },
            }
        ),
        encoding="utf-8",
    )
    direction_metrics = tmp_path / "iti_direction_metrics.json"
    direction_metrics.write_text(
        json.dumps({"selected_layers_sha256": selection_hash}),
        encoding="utf-8",
    )
    (target_dirs["iti_intervention"] / "summary.json").write_text(
        json.dumps(
            {
                **common_summary,
                "direction_metrics": str(direction_metrics),
                "direction_metrics_sha256": sha256_file(direction_metrics),
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
