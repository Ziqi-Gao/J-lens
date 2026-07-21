from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from jlens_workspace.artifacts import atomic_write_json, sha256_file
from jlens_workspace.workflows.iti import _best_validation_setting, rebuild_iti_index


def test_validation_selection_prefers_margin_then_smaller_k_and_strength() -> None:
    rows = []
    for top_k, strength, margin in (
        (4, 5.0, 1.0),
        (8, 5.0, 2.0),
        (4, 10.0, 2.0),
        (8, 10.0, 2.0),
    ):
        rows.append({"top_k": top_k, "strength": strength, "target_margin": margin})
    selected = _best_validation_setting(rows)["selected"]
    assert selected["top_k"] == 4
    assert selected["strength"] == 10.0


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_iti_index_builds_identity_checked_held_out_j_comparison(
    tmp_path: Path,
) -> None:
    concept = "concept:a"
    encoded = quote(concept, safe="")
    iti_root = tmp_path / "iti"
    j_root = tmp_path / "j"
    residuals = tmp_path / "residuals"
    atomic_write_json(residuals / "metadata.json", {"example_hash": "example-sha"})
    (residuals / "labels.npy").write_bytes(b"registered-labels")
    j_config_path = tmp_path / "j_config.yaml"
    j_config_path.write_text(
        json.dumps(
            {"intervention": {"source_activations_dir": str(residuals)}}
        ),
        encoding="utf-8",
    )
    identity = {
        "model_id": "org/model",
        "model_revision": "model-sha",
        "tokenizer_id": "org/model",
        "tokenizer_revision": "model-sha",
        "dataset_source": "org/data",
        "dataset_revision": "data-sha",
    }
    atomic_write_json(
        iti_root / "manifest.json",
        identity
        | {
            "notes": {
                "prompts_sha256": "b" * 64,
                "source_residual_activations_dir": str(residuals),
                "source_residual_metadata_sha256": sha256_file(
                    residuals / "metadata.json"
                ),
                "source_residual_labels_sha256": sha256_file(residuals / "labels.npy"),
            }
        },
    )
    atomic_write_json(
        j_root / "manifest.json",
        identity
        | {
            "notes": {
                "prompts_sha256": "b" * 64,
                "config_path": str(j_config_path),
                "config_sha256": sha256_file(j_config_path),
            }
        },
    )
    atomic_write_json(j_root / "index.json", {"complete": True})
    iti_summary = {
        "method": "honest_llama_mass_mean_qwen_full_attention_v1",
        "target_concept_id": concept,
        "selection": {"selected": {"top_k": 4, "strength": 5.0}},
        "analysis": {},
        "run_metadata": {"random_control_seeds": [101]},
    }
    atomic_write_json(iti_root / "targets" / encoded / "summary.json", iti_summary)
    iti_rows = []
    for condition, effect in (("mass_mean", 2.0), ("random_101", 0.25)):
        for strength in (0.0, 5.0):
            iti_rows.append(
                {
                    "prompt_id": "classify_0",
                    "condition_id": condition,
                    "strength": strength,
                    "target_margin": effect if strength else 0.0,
                }
            )
    _write_jsonl(iti_root / "targets" / encoded / "test_scores.jsonl", iti_rows)

    j_rows = []
    for prompt_id in ("choose_0", "classify_0"):
        for condition, effect in (("j", 1.0), ("random_101", 0.1)):
            for strength in (0.0, 0.5):
                j_rows.append(
                    {
                        "prompt_id": prompt_id,
                        "condition_id": condition,
                        "strength": strength,
                        "target_margin": effect if strength else 0.0,
                    }
                )
    _write_jsonl(j_root / "targets" / encoded / "scores.jsonl", j_rows)

    index = rebuild_iti_index(
        iti_root,
        expected_concepts=[concept],
        reference_j_intervention_dir=j_root,
        validation_prompt_prefixes=["choose"],
        test_prompt_prefixes=["classify"],
    )
    assert index["complete"] is True
    comparison = json.loads((iti_root / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["same_model_tokenizer_dataset"] is True
    assert comparison["entries"][0]["iti"]["held_out_target_margin_effect"] == 2.0
    assert comparison["entries"][0]["j_component"]["held_out_target_margin_effect"] == 1.0
