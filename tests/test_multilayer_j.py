from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

from jlens_workspace.artifacts import atomic_write_json, sha256_file
from jlens_workspace.concept_intervention.generation import (
    write_generation_artifacts,
)
from jlens_workspace.concept_intervention.j_component.multilayer import (
    aggregate_layer_k,
    rebuild_multilayer_j_index,
)


def _write_metrics(
    root: Path,
    *,
    concept_id: str,
    layer: int,
    replicate_id: str,
    measured: int,
    right_censored: bool,
) -> None:
    path = (
        root
        / "occupancy"
        / "rmsnorm_weighted"
        / "positive_cosine"
        / f"layer_{layer:02d}"
        / quote(concept_id, safe="")
        / "pos"
        / quote(replicate_id, safe="")
        / "metrics.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "method": "concept_occupancy_method",
                "solver_method": "nonnegative_gradient_pursuit_standard",
                "layer": layer,
                "concept_id": concept_id,
                "replicate_id": replicate_id,
                "convention": "rmsnorm_weighted",
                "selection_mode": "positive_cosine",
                "sign": "+",
                "k_max": 64,
                "primary_occupancy": {
                    "k_selected_before_crossing": measured,
                    "crossing_k": None if right_censored else measured + 1,
                    "right_censored": right_censored,
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("values", "expected_measured", "floor", "censored"),
    [
        ([0, 2, 4, 6, 64], 4, False, False),
        ([0, 0, 0, 2, 4], 0, True, False),
        ([64, 64, 64, 4, 2], 64, False, True),
    ],
)
def test_replicate_median_k_records_floor_and_right_censor(
    tmp_path: Path,
    values: list[int],
    expected_measured: int,
    floor: bool,
    censored: bool,
) -> None:
    replicates = ["primary", "bootstrap_1", "bootstrap_2", "bootstrap_3", "bootstrap_4"]
    for replicate, value in zip(replicates, values, strict=True):
        _write_metrics(
            tmp_path,
            concept_id="concept:a",
            layer=7,
            replicate_id=replicate,
            measured=value,
            right_censored=value == 64,
        )

    result = aggregate_layer_k(
        occupancy_dir=tmp_path,
        concept_id="concept:a",
        layer=7,
        replicate_ids=replicates,
        k_max=64,
    )

    assert result["K_measured"] == expected_measured
    assert result["K_used"] == max(1, expected_measured)
    assert result["k_floor_applied"] is floor
    assert result["statistically_supported"] is (not floor)
    assert result["right_censored"] is censored


def test_j_shard_index_requires_complete_grid_and_checks_zero_consistency(
    tmp_path: Path,
) -> None:
    target = tmp_path / "targets/concept%3Aa"
    provenance = {"layer_k": [{"layer": 3, "K_used": 4}]}
    for grid_index, condition_id in enumerate(("full", "j", "non_j")):
        shard = target / "shards" / f"grid_{grid_index:04d}"
        shard.mkdir(parents=True)
        candidate_path = shard / "candidate_scores.jsonl"
        candidate_path.write_text(
            json.dumps(
                {
                    "prompt_id": "prompt",
                    "target_log_probability": -1.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        generation_files = write_generation_artifacts(shard, [])
        atomic_write_json(
            shard / "summary.json",
            {
                "method": "j_component_intervention",
                "target_concept_id": "concept:a",
                "selected_layers": [3],
                "coordinate": "resid_post",
                "position": "last_token_each_forward_call",
                "normalization": "test",
                "mean_residual_norms": {"3": 2.0},
                "source_provenance": provenance,
                "candidate_scores_sha256": sha256_file(candidate_path),
                "generation_files": generation_files,
                "grid_index": grid_index,
                "grid_size": 3,
                "grid_condition": {
                    "condition_id": condition_id,
                    "strength": 0.0,
                },
            },
        )

    index = rebuild_multilayer_j_index(
        tmp_path,
        concept_ids=["concept:a"],
        strengths=[0.0],
        random_control_seeds=[],
    )

    assert index["complete"] is True
    summary = json.loads(
        (target / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["grid_size"] == 3
    assert summary["generation_artifacts_sharded"] is True
