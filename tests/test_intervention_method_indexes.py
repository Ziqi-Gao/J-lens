from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jlens_workspace.artifacts import atomic_write_json, sha256_file
from jlens_workspace.concept_intervention.generation import (
    prompt_ids_sha256,
    write_generation_artifacts,
)
from jlens_workspace.concept_intervention.iti.experiment import (
    iti_experiment_grid,
    rebuild_iti_experiment_index,
)
from jlens_workspace.concept_intervention.iti.workflow import ITIWorkflowError
from jlens_workspace.concept_intervention.raptor.intervention import RaptorError
from jlens_workspace.concept_intervention.raptor.workflow import (
    rebuild_raptor_index,
)

_SETTINGS = {
    "sample_seeds": [1001],
    "max_new_tokens": 8,
    "temperature": 0.75,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
}


def _contract(prompt_ids: list[str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "prompt_ids": prompt_ids,
        "prompt_ids_sha256": prompt_ids_sha256(prompt_ids),
        "prompt_count": len(prompt_ids),
        "candidate_prompt_ids": prompt_ids,
        "open_prompt_ids": [],
        "sample_seeds": [1001],
        "decodings_per_prompt": 2,
        "expected_rows": len(prompt_ids) * 2,
        "generation_settings": _SETTINGS,
    }


def _manifest(method: str, prompt_ids: list[str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_name": method,
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
            "workflow": method,
            "selected_layers_sha256": "c" * 64,
            "row_manifest_sha256": "d" * 64,
            "generation": {
                **_SETTINGS,
                "candidate_prompt_count": len(prompt_ids),
                "open_prompt_count": 0,
                "prompt_count": len(prompt_ids),
                "prompt_ids_sha256": prompt_ids_sha256(prompt_ids),
                "expected_rows_per_grid_point": len(prompt_ids) * 2,
            },
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _candidate_row(
    *,
    method: str,
    prompt_id: str,
    split: str,
    condition: dict[str, object],
) -> dict[str, object]:
    row_condition = {
        ("condition" if key == "mode" else key): value
        for key, value in condition.items()
    }
    return {
        "prompt_id": prompt_id,
        "method": method,
        "target_concept_id": "concept:a",
        "evaluation_split": split,
        "candidate_log_probabilities": {
            "concept:a": -1.0,
            "other": -2.0,
        },
        "candidate_probabilities_normalized": {
            "concept:a": 0.7,
            "other": 0.3,
        },
        "target_log_probability": -1.0,
        "target_candidate_probability": 0.7,
        "target_margin": 1.0,
        "target_rank": 1,
        **row_condition,
    }


def _generation_rows(
    *,
    method: str,
    grid_index: int,
    condition: dict[str, object],
    prompt_ids: list[str],
) -> list[dict[str, object]]:
    enabled = (
        condition.get("target_probability") is not None
        if method == "raptor_intervention"
        else float(condition["strength"]) != 0.0
    )
    telemetry = (
        [
            {
                "layer": 3,
                "forward_call": 0,
                "generation_step": 0,
                "batch_size": 1,
                "sequence_length": 1,
                "injected_norm": 1.0,
            }
        ]
        if enabled
        else []
    )
    grid_point = (
        {"target_probability": condition.get("target_probability")}
        if method == "raptor_intervention"
        else {
            "variant": condition["variant"],
            "top_k": condition["top_k"],
            "strength": condition["strength"],
        }
    )
    return [
        {
            "generation_id": (
                f"generation-{grid_index}-{prompt_id}-{decoding}-{seed}"
            ),
            "blind_id": f"blind-{grid_index}-{prompt_id}-{decoding}-{seed}",
            "method": method,
            "concept_id": "concept:a",
            "condition_id": condition["condition_id"],
            "grid_point": grid_point,
            "intervention_metadata": {
                "selected_layers": [3],
                "active_layers": [3],
            },
            "prompt_id": prompt_id,
            "prompt_text": "Prompt.",
            "decoding": decoding,
            "seed": seed,
            "generated_token_ids": [1],
            "generated_text": "Result.",
            "token_log_probabilities": [-0.5],
            "telemetry": telemetry,
            "injected_norm_by_layer": {"3": 1.0} if enabled else {},
            "total_injected_norm": 1.0 if enabled else 0.0,
            "generation_settings": _SETTINGS,
        }
        for prompt_id in prompt_ids
        for decoding, seed in (("greedy", None), ("sample", 1001))
    ]


def _write_shard_manifest(
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


def test_raptor_index_rejects_mixed_shard_provenance(tmp_path: Path) -> None:
    method = "raptor_intervention"
    prompt_ids = ["prompt"]
    manifest = _manifest(method, prompt_ids)
    atomic_write_json(tmp_path / "manifest.json", manifest)
    grid = [
        {"condition_id": "no_hook", "target_probability": None},
        {
            "condition_id": "target_probability_0.9",
            "target_probability": 0.9,
        },
    ]
    summaries = []
    for grid_index, condition in enumerate(grid):
        shard = (
            tmp_path
            / "targets"
            / "concept%3Aa"
            / "shards"
            / f"grid_{grid_index:04d}"
        )
        shard.mkdir(parents=True)
        candidate_rows = [
            _candidate_row(
                method=method,
                prompt_id="prompt",
                split="test",
                condition=condition,
            )
        ]
        candidate_path = shard / "candidate_scores.jsonl"
        _write_jsonl(candidate_path, candidate_rows)
        generation_rows = _generation_rows(
            method=method,
            grid_index=grid_index,
            condition=condition,
            prompt_ids=prompt_ids,
        )
        generation_files = write_generation_artifacts(shard, generation_rows)
        summary = {
            "schema_version": 1,
            "method": method,
            "target_concept_id": "concept:a",
            "selected_layers": [3],
            "coordinate": "resid_post",
            "position": "last_token_each_forward_call",
            "normalization": "unit probe",
            "source_provenance": {"probe": "shared"},
            "upstream": {"commit": "pinned"},
            "candidate_score_rows": len(candidate_rows),
            "candidate_scores_sha256": sha256_file(candidate_path),
            "generation_rows": len(generation_rows),
            "generation_files": generation_files,
            "generation_contract": _contract(prompt_ids),
            "grid_index": grid_index,
            "grid_size": len(grid),
            "grid_condition": condition,
        }
        atomic_write_json(shard / "summary.json", summary)
        _write_shard_manifest(
            tmp_path,
            grid_index=grid_index,
            manifest=manifest,
        )
        summaries.append((shard / "summary.json", summary))

    index = rebuild_raptor_index(
        tmp_path,
        concept_ids=["concept:a"],
        target_probabilities=[0.9],
    )
    assert index["complete"] is True

    summary_path, original = summaries[1]
    wrong_grid = {
        **original,
        "grid_condition": {
            "condition_id": "target_probability_0.8",
            "target_probability": 0.8,
        },
    }
    atomic_write_json(summary_path, wrong_grid)
    with pytest.raises(RaptorError, match="grid identity"):
        rebuild_raptor_index(
            tmp_path,
            concept_ids=["concept:a"],
            target_probabilities=[0.9],
        )

    mixed = {**original, "source_provenance": {"probe": "another"}}
    atomic_write_json(summary_path, mixed)
    with pytest.raises(RaptorError, match="provenance"):
        rebuild_raptor_index(
            tmp_path,
            concept_ids=["concept:a"],
            target_probabilities=[0.9],
        )


def test_iti_index_rejects_mixed_shard_provenance(tmp_path: Path) -> None:
    method = "iti_intervention"
    prompt_ids = ["v", "t"]
    manifest = _manifest(method, prompt_ids)
    atomic_write_json(tmp_path / "manifest.json", manifest)
    config = SimpleNamespace(
        variants=["native", "layer_matched"],
        top_k_grid=[1],
        layer_matched_top_k_grid=[1],
        random_control_seeds=[],
        strengths=[0.0, 1.0],
    )
    grid = iti_experiment_grid(config)
    direction_metrics = tmp_path / "directions/concept%3Aa/metrics.json"
    atomic_write_json(direction_metrics, {"method": "iti"})
    summaries = []
    for grid_index, condition in enumerate(grid):
        shard = (
            tmp_path
            / "targets"
            / "concept%3Aa"
            / "shards"
            / f"grid_{grid_index:04d}"
        )
        shard.mkdir(parents=True)
        candidate_rows = [
            _candidate_row(
                method=method,
                prompt_id=prompt_id,
                split=split,
                condition=condition,
            )
            for prompt_id, split in (("v", "validation"), ("t", "test"))
        ]
        candidate_path = shard / "candidate_scores.jsonl"
        _write_jsonl(candidate_path, candidate_rows)
        generation_rows = _generation_rows(
            method=method,
            grid_index=grid_index,
            condition=condition,
            prompt_ids=prompt_ids,
        )
        generation_files = write_generation_artifacts(shard, generation_rows)
        summary = {
            "schema_version": 1,
            "method": method,
            "target_concept_id": "concept:a",
            "variants": list(config.variants),
            "selected_layers": [3],
            "coordinate": "attention_head_output_pre_o_proj",
            "position": "last_token_each_forward_call",
            "normalization": "unit_direction_times_projection_std_times_alpha",
            "native_top_k_grid": list(config.top_k_grid),
            "layer_matched_top_k_grid": list(
                config.layer_matched_top_k_grid
            ),
            "strengths": list(config.strengths),
            "direction_metrics": str(direction_metrics),
            "direction_metrics_sha256": sha256_file(direction_metrics),
            "candidate_score_rows": len(candidate_rows),
            "candidate_scores_sha256": sha256_file(candidate_path),
            "generation_rows": len(generation_rows),
            "generation_files": generation_files,
            "generation_contract": _contract(prompt_ids),
            "grid_index": grid_index,
            "grid_size": len(grid),
            "grid_condition": condition,
        }
        atomic_write_json(shard / "summary.json", summary)
        _write_shard_manifest(
            tmp_path,
            grid_index=grid_index,
            manifest=manifest,
        )
        summaries.append((shard / "summary.json", summary))

    index = rebuild_iti_experiment_index(
        tmp_path,
        concept_ids=["concept:a"],
        config=config,
    )
    assert index["complete"] is True

    summary_path, mixed = summaries[-1]
    mixed["direction_metrics_sha256"] = "0" * 64
    atomic_write_json(summary_path, mixed)
    with pytest.raises(ITIWorkflowError, match="provenance"):
        rebuild_iti_experiment_index(
            tmp_path,
            concept_ids=["concept:a"],
            config=config,
        )
