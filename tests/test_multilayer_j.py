from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

from jlens_workspace.artifacts import atomic_write_json, sha256_file
from jlens_workspace.concept_intervention.generation import (
    prompt_ids_sha256,
    write_generation_artifacts,
)
from jlens_workspace.concept_intervention.j_component.multilayer import (
    MultiLayerJError,
    aggregate_layer_k,
    rebuild_multilayer_j_index,
)

_CANDIDATE_LABELS = {
    f"concept:{letter}": letter for letter in "abcdefg"
}
_CANDIDATE_LOGPROBS = {
    concept_id: (-1.0 if concept_id == "concept:a" else -2.0)
    for concept_id in _CANDIDATE_LABELS
}
_CANDIDATE_PROBABILITIES = {
    concept_id: (0.4 if concept_id == "concept:a" else 0.1)
    for concept_id in _CANDIDATE_LABELS
}


def _write_generation_manifest(
    root: Path,
    *,
    settings: dict[str, object],
    prompt_id: str = "classify_0",
) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "experiment_name": "j_component_intervention",
        "seed": 42,
        "model_id": "model",
        "model_revision": "model-revision",
        "tokenizer_id": "model",
        "tokenizer_revision": "model-revision",
        "lens_source": "local:lens",
        "lens_revision": None,
        "dataset_source": "dataset",
        "dataset_revision": "dataset-revision",
        "dataset_hash": "dataset-hash",
        "git_commit": "a" * 40,
        "python": "3.11.0",
        "platform": "test",
        "packages": {},
        "notes": {
            "direction": "concept_intervention",
            "coordinate": "resid_post/block_output",
            "config_sha256": "b" * 64,
            "force_bos": False,
            "workflow": "j_component_intervention",
            "selected_layers_sha256": "c" * 64,
            "row_manifest_sha256": "d" * 64,
            "generation": {
                **settings,
                "candidate_labels": _CANDIDATE_LABELS,
                "candidate_prompt_splits": {prompt_id: "test"},
                "candidate_prompt_count": 1,
                "open_prompt_count": 0,
                "prompt_count": 1,
                "prompt_ids_sha256": prompt_ids_sha256([prompt_id]),
                "expected_rows_per_grid_point": 1
                * (1 + len(settings["sample_seeds"])),
            },
        },
    }
    atomic_write_json(root / "manifest.json", payload)
    return payload


def _write_grid_manifest(
    root: Path,
    *,
    grid_index: int,
    manifest: dict[str, object],
) -> None:
    atomic_write_json(
        root
        / "manifests"
        / "concept%3Aa"
        / f"grid_{grid_index:04d}.json",
        manifest,
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
    settings = {
        "sample_seeds": [1001],
        "max_new_tokens": 8,
        "temperature": 0.75,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
    }
    manifest = _write_generation_manifest(tmp_path, settings=settings)
    for grid_index, condition_id in enumerate(("full", "j", "non_j")):
        _write_grid_manifest(
            tmp_path,
            grid_index=grid_index,
            manifest=manifest,
        )
        shard = target / "shards" / f"grid_{grid_index:04d}"
        shard.mkdir(parents=True)
        candidate_path = shard / "candidate_scores.jsonl"
        candidate_path.write_text(
            json.dumps(
                {
                    "prompt_id": "classify_0",
                    "method": "j_component_intervention",
                    "condition_id": condition_id,
                    "strength": 0.0,
                    "evaluation_split": "test",
                    "target_concept_id": "concept:a",
                    "candidate_log_probabilities": _CANDIDATE_LOGPROBS,
                    "candidate_probabilities_normalized": (
                        _CANDIDATE_PROBABILITIES
                    ),
                    "target_log_probability": -1.0,
                    "target_candidate_probability": 0.4,
                    "target_margin": 1.0,
                    "target_rank": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        generation_rows = [
            {
                "generation_id": f"generation-{grid_index}-{decoding}",
                "blind_id": f"blind-{grid_index}-{decoding}",
                "method": "j_component_intervention",
                "concept_id": "concept:a",
                "condition_id": condition_id,
                "grid_point": {"strength": 0.0},
                "intervention_metadata": {"selected_layers": [3]},
                "prompt_id": "classify_0",
                "prompt_text": "Prompt.",
                "decoding": decoding,
                "seed": seed,
                "generated_token_ids": [1],
                "generated_text": "Result.",
                "token_log_probabilities": [-0.5],
                "telemetry": [
                    {
                        "layer": 3,
                        "forward_call": 0,
                        "generation_step": 0,
                        "batch_size": 1,
                        "sequence_length": 1,
                        "active_positions": 1,
                        "strength": 0.0,
                        "residual_norm": 2.0,
                        "kind": "addition",
                        "injected_norm": 0.0,
                    }
                ],
                "injected_norm_by_layer": {"3": 0.0},
                "total_injected_norm": 0.0,
                "generation_settings": settings,
            }
            for decoding, seed in (("greedy", None), ("sample", 1001))
        ]
        generation_files = write_generation_artifacts(shard, generation_rows)
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
                "candidate_score_rows": 1,
                "candidate_scores_sha256": sha256_file(candidate_path),
                "generation_rows": 2,
                "generation_files": generation_files,
                "generation_contract": {
                    "schema_version": 1,
                    "prompt_ids": ["classify_0"],
                    "prompt_ids_sha256": prompt_ids_sha256(["classify_0"]),
                    "prompt_count": 1,
                    "candidate_prompt_ids": ["classify_0"],
                    "candidate_prompt_splits": {"classify_0": "test"},
                    "candidate_labels": _CANDIDATE_LABELS,
                    "open_prompt_ids": [],
                    "sample_seeds": [1001],
                    "decodings_per_prompt": 2,
                    "expected_rows": 2,
                    "generation_settings": settings,
                },
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


def test_j_index_rejects_mixed_contract_and_empty_nonzero_scores(
    tmp_path: Path,
) -> None:
    target = tmp_path / "targets/concept%3Aa"
    provenance = {"layer_k": [{"layer": 3, "K_used": 4}]}
    settings = {
        "sample_seeds": [1001],
        "max_new_tokens": 8,
        "temperature": 0.75,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
    }
    grid = [
        (condition_id, strength)
        for condition_id in ("full", "j", "non_j")
        for strength in (0.0, 0.5)
    ]
    manifest = _write_generation_manifest(tmp_path, settings=settings)
    for grid_index, (condition_id, strength) in enumerate(grid):
        _write_grid_manifest(
            tmp_path,
            grid_index=grid_index,
            manifest=manifest,
        )
        shard = target / "shards" / f"grid_{grid_index:04d}"
        shard.mkdir(parents=True)
        prompt_id = "other-prompt" if grid_index == 3 else "classify_0"
        candidate_path = shard / "candidate_scores.jsonl"
        candidate_rows = (
            []
            if grid_index == 5
            else [
                {
                    "prompt_id": prompt_id,
                    "method": "j_component_intervention",
                    "condition_id": condition_id,
                    "strength": strength,
                    "evaluation_split": "test",
                    "target_concept_id": "concept:a",
                    "candidate_log_probabilities": _CANDIDATE_LOGPROBS,
                    "candidate_probabilities_normalized": (
                        _CANDIDATE_PROBABILITIES
                    ),
                    "target_log_probability": -1.0,
                    "target_candidate_probability": 0.4,
                    "target_margin": 1.0,
                    "target_rank": 1,
                }
            ]
        )
        candidate_path.write_text(
            "".join(json.dumps(row) + "\n" for row in candidate_rows),
            encoding="utf-8",
        )
        injected_norm = 0.0 if strength == 0.0 else 1.0
        telemetry = [
            {
                "layer": 3,
                "forward_call": 0,
                "generation_step": 0,
                "batch_size": 1,
                "sequence_length": 1,
                "active_positions": 1,
                "strength": strength,
                "residual_norm": 2.0,
                "kind": "addition",
                "injected_norm": injected_norm,
            }
        ]
        generation_rows = [
            {
                "generation_id": f"generation-{grid_index}-{decoding}",
                "blind_id": f"blind-{grid_index}-{decoding}",
                "method": "j_component_intervention",
                "concept_id": "concept:a",
                "condition_id": condition_id,
                "grid_point": {"strength": strength},
                "intervention_metadata": {"selected_layers": [3]},
                "prompt_id": prompt_id,
                "prompt_text": "Prompt.",
                "decoding": decoding,
                "seed": seed,
                "generated_token_ids": [1],
                "generated_text": "Result.",
                "token_log_probabilities": [-0.5],
                "telemetry": telemetry,
                "injected_norm_by_layer": {"3": injected_norm},
                "total_injected_norm": injected_norm,
                "generation_settings": settings,
            }
            for decoding, seed in (("greedy", None), ("sample", 1001))
        ]
        generation_files = write_generation_artifacts(shard, generation_rows)
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
                "candidate_score_rows": len(candidate_rows),
                "candidate_scores_sha256": sha256_file(candidate_path),
                "generation_rows": len(generation_rows),
                "generation_files": generation_files,
                "generation_contract": {
                    "schema_version": 1,
                    "prompt_ids": [prompt_id],
                    "prompt_ids_sha256": prompt_ids_sha256([prompt_id]),
                    "prompt_count": 1,
                    "candidate_prompt_ids": [prompt_id],
                    "candidate_prompt_splits": {prompt_id: "test"},
                    "candidate_labels": _CANDIDATE_LABELS,
                    "open_prompt_ids": [],
                    "sample_seeds": [1001],
                    "decodings_per_prompt": 2,
                    "expected_rows": 2,
                    "generation_settings": settings,
                },
                "grid_index": grid_index,
                "grid_size": len(grid),
                "grid_condition": {
                    "condition_id": condition_id,
                    "strength": strength,
                },
            },
        )

    with pytest.raises(
        MultiLayerJError,
        match=r"output contract|generation contracts|candidate-score",
    ):
        rebuild_multilayer_j_index(
            tmp_path,
            concept_ids=["concept:a"],
            strengths=[0.0, 0.5],
            random_control_seeds=[],
        )
