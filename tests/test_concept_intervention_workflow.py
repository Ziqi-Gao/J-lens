from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import numpy as np

from jlens_workspace.artifacts import sha256_file
from jlens_workspace.config import load_experiment_config
from jlens_workspace.workflows.concept_intervention import (
    _candidate_token_ids,
    _curve_summary,
    _load_prompt_bank,
    load_registered_directions,
)


class _FakeTokenizer:
    chat_template = None

    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [sum(value.encode("utf-8"))]


def test_prompt_bank_counterbalances_and_candidate_tokens_are_unique(
    tmp_path: Path,
) -> None:
    prompts = tmp_path / "prompts.json"
    prompts.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "prompts": [
                    {"prompt_id": "a", "label_rotation": 0, "text": "{labels} is"},
                    {"prompt_id": "b", "label_rotation": 1, "text": "{labels} is"},
                ],
            }
        ),
        encoding="utf-8",
    )
    labels = {"concept:a": "alpha", "concept:b": "beta"}
    tokenizer = _FakeTokenizer()
    loaded = _load_prompt_bank(prompts, tokenizer=tokenizer, candidate_labels=labels)
    assert loaded[0].label_order == ("concept:a", "concept:b")
    assert loaded[1].label_order == ("concept:b", "concept:a")
    assert len(set(_candidate_token_ids(tokenizer, labels).values())) == 2


def test_registered_directions_are_identity_checked_and_reconstruct(
    tmp_path: Path,
) -> None:
    concept_id = "concept:love"
    occupancy = tmp_path / "occupancy_artifact"
    probes = tmp_path / "probes"
    index_path = occupancy / "occupancy" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(
        json.dumps(
            {
                "complete": True,
                "observed_combinations": 840,
            }
        ),
        encoding="utf-8",
    )
    commit = "a" * 40
    (occupancy / "provenance_audit.json").write_text(
        json.dumps({"source_git_commit": commit}), encoding="utf-8"
    )
    probe_dir = probes / "layer_28" / f"concept_{quote(concept_id, safe='')}"
    probe_dir.mkdir(parents=True)
    full = np.asarray([2.0, 1.0, 0.5])
    j_direction = np.asarray([1.0, 0.0, 0.0])
    non_j = full - j_direction
    np.save(probe_dir / "probe_vector.npy", full, allow_pickle=False)

    combo = (
        occupancy
        / "occupancy/rmsnorm_weighted/positive_cosine/layer_28"
        / quote(concept_id, safe="")
        / "pos/primary"
    )
    combo.mkdir(parents=True)
    np.save(combo / "w_J_k02.npy", j_direction, allow_pickle=False)
    np.save(combo / "w_nonJ_k02.npy", non_j, allow_pickle=False)
    np.save(combo / "errors.npy", np.asarray([1.0, 0.9, 0.8]), allow_pickle=False)
    metrics = {
        "method": "concept_occupancy_method_v2",
        "solver_method": "nonnegative_gradient_pursuit_v2",
        "layer": 28,
        "concept_id": concept_id,
        "sign": "+",
        "replicate_id": "primary",
        "convention": "rmsnorm_weighted",
        "selection_mode": "positive_cosine",
        "probe_vector_sha256": sha256_file(probe_dir / "probe_vector.npy"),
        "primary_occupancy": {
            "k_selected_before_crossing": 2,
            "w_j_file": "w_J_k02.npy",
            "w_nonj_file": "w_nonJ_k02.npy",
        },
        "decoded_tokens": [" love", " like"],
    }
    (combo / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    directions, provenance = load_registered_directions(
        occupancy_dir=occupancy,
        probes_dir=probes,
        layer=28,
        concept_id=concept_id,
        convention="rmsnorm_weighted",
        random_seeds=[11, 22],
        expected_occupancy_index_sha256=sha256_file(index_path),
        expected_occupancy_git_commit=commit,
    )
    assert [record.condition_id for record in directions] == [
        "full",
        "j",
        "non_j",
        "random_11",
        "random_22",
    ]
    np.testing.assert_allclose(directions[1].direction + directions[2].direction, full)
    assert provenance["selected_k"] == 2
    assert provenance["reconstruction_max_abs_error"] == 0.0


def test_curve_summary_compares_j_full_and_random_slopes() -> None:
    rows = []
    slopes = {
        "full": 2.0,
        "j": 1.0,
        "non_j": 0.5,
        "random_1": 0.1,
        "random_2": -0.1,
    }
    for condition_id, slope in slopes.items():
        for strength in (-1.0, 0.0, 1.0):
            for prompt_id in ("a", "b"):
                rows.append(
                    {
                        "condition_id": condition_id,
                        "strength": strength,
                        "target_log_probability": slope * strength,
                        "target_candidate_probability": 0.25,
                        "target_margin": slope * strength,
                        "target_rank": 1,
                        "prompt_id": prompt_id,
                    }
                )
    summary = _curve_summary(rows)
    assert summary["j_full_curve_correlation"] == 1.0
    assert summary["j_to_full_slope_ratio"] == 0.5
    assert summary["j_slope_exceeds_all_random_controls"] is True


def test_formal_intervention_config_validates() -> None:
    config = load_experiment_config(
        "Concept_intervention/configs/qwen35_4b_concept_intervention_v2.yaml"
    )
    assert config.intervention is not None
    assert config.intervention.method == "concept_j_component_intervention_v2"
    assert config.intervention.layer == 28
