from __future__ import annotations

import json
from pathlib import Path

import pytest

from jlens_workspace.artifacts import sha256_file
from jlens_workspace.concept_intervention.candidate_rescore import (
    CANONICAL_SCORE_CONTRACT,
    CandidateRescoreError,
    rebuild_candidate_rescore_index,
)

CONCEPT = "concept:a"
METHOD = "j_component_intervention"
LABELS = {f"concept:{letter}": letter for letter in "abcdefg"}
CONTRACT = {
    "candidate_labels": LABELS,
    "candidate_prompt_ids": ["classify_0"],
    "candidate_prompt_splits": {"classify_0": "test"},
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(*, strength: float = 0.0, target_log_prob: float = -1.0) -> dict[str, object]:
    log_probs = {
        concept_id: target_log_prob if concept_id == CONCEPT else -2.0
        for concept_id in LABELS
    }
    probabilities = {concept_id: 1.0 / 7.0 for concept_id in LABELS}
    off_target = [value for key, value in log_probs.items() if key != CONCEPT]
    return {
        "prompt_id": "classify_0",
        "evaluation_split": "test",
        "target_concept_id": CONCEPT,
        "strength": strength,
        "method": METHOD,
        "condition_id": "full",
        "condition": "full",
        "selected_layers": [7],
        "candidate_log_probabilities": log_probs,
        "candidate_probabilities_normalized": probabilities,
        "target_log_probability": target_log_prob,
        "target_candidate_probability": probabilities[CONCEPT],
        "target_margin": target_log_prob - sum(off_target) / len(off_target),
        "target_rank": 1,
    }


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    rescore = tmp_path / "rescore"
    source_manifest = source / "manifest.json"
    _write_json(source_manifest, {"method": METHOD})
    source_candidate = (
        source
        / "targets"
        / "concept%3Aa"
        / "shards"
        / "grid_0000"
        / "candidate_scores.jsonl"
    )
    _write_jsonl(source_candidate, [_row()])
    source_summary = source / "targets" / "concept%3Aa" / "summary.json"
    _write_json(
        source_summary,
        {
            "method": METHOD,
            "target_concept_id": CONCEPT,
            "selected_layers": [7],
            "generation_contract": CONTRACT,
        },
    )
    source_index = source / "index.json"
    _write_json(
        source_index,
        {
            "complete": True,
            "method": METHOD,
            "manifest_sha256": sha256_file(source_manifest),
            "entries": [
                {
                    "concept_id": CONCEPT,
                    "summary": "targets/concept%3Aa/summary.json",
                    "summary_sha256": sha256_file(source_summary),
                    "artifact_seal": {
                        "files": [
                            {
                                "path": (
                                    "targets/concept%3Aa/shards/grid_0000/"
                                    "candidate_scores.jsonl"
                                ),
                                "sha256": sha256_file(source_candidate),
                            }
                        ]
                    },
                }
            ],
        },
    )

    candidate = rescore / "targets" / "concept%3Aa" / "candidate_scores.jsonl"
    _write_jsonl(candidate, [_row(target_log_prob=-0.5)])
    summary = rescore / "targets" / "concept%3Aa" / "summary.json"
    _write_json(
        summary,
        {
            "artifact_kind": "candidate_score_rescore",
            "method": METHOD,
            "target_concept_id": CONCEPT,
            "selected_layers": [7],
            "generation_contract": CONTRACT,
            "generation_rows": 0,
            "generation_files": None,
            "candidate_score_contract": CANONICAL_SCORE_CONTRACT,
            "candidate_scores_sha256": sha256_file(candidate),
        },
    )
    _write_json(
        rescore / "manifest.json",
        {
            "notes": {
                "candidate_rescore": {
                    "schema_version": 1,
                    "method": METHOD,
                    "scoring_contract": CANONICAL_SCORE_CONTRACT,
                    "source_method_index_sha256": sha256_file(source_index),
                    "source_method_manifest_sha256": sha256_file(source_manifest),
                }
            }
        },
    )
    return source, rescore


def test_candidate_rescore_index_preserves_source_grid_identity(
    tmp_path: Path,
) -> None:
    source, rescore = _artifacts(tmp_path)

    index = rebuild_candidate_rescore_index(
        rescore,
        source_method_root=source,
        method=METHOD,
        concept_ids=[CONCEPT],
        index_builder={"git_commit": "builder"},
    )

    assert index["complete"] is True
    assert index["scoring_contract"] == CANONICAL_SCORE_CONTRACT
    assert index["entries"][0]["candidate_score_rows"] == 1
    assert index["entries"][0]["source_candidate_artifacts"] == 1


def test_candidate_rescore_rejects_changed_condition_identity(
    tmp_path: Path,
) -> None:
    source, rescore = _artifacts(tmp_path)
    candidate = rescore / "targets" / "concept%3Aa" / "candidate_scores.jsonl"
    _write_jsonl(candidate, [_row(strength=1.0, target_log_prob=-0.5)])
    summary = rescore / "targets" / "concept%3Aa" / "summary.json"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["candidate_scores_sha256"] = sha256_file(candidate)
    _write_json(summary, payload)

    with pytest.raises(CandidateRescoreError, match="grid/prompt identity"):
        rebuild_candidate_rescore_index(
            rescore,
            source_method_root=source,
            method=METHOD,
            concept_ids=[CONCEPT],
        )
