from __future__ import annotations

import json
from pathlib import Path

import pytest

from jlens_workspace.artifacts import sha256_file
from jlens_workspace.concept_intervention.generation import (
    InterventionGenerationError,
    load_open_prompt_bank,
    prompt_ids_sha256,
    validate_generation_artifacts,
    write_generation_artifacts,
)


class _Tokenizer:
    chat_template = None


def test_frozen_open_prompts_are_exactly_balanced() -> None:
    root = Path(__file__).parents[1]
    prompts = load_open_prompt_bank(
        root / "Concept_intervention/data/open_intervention_prompts.json",
        tokenizer=_Tokenizer(),
    )
    assert len(prompts) == 32
    assert sum(prompt.split == "validation" for prompt in prompts) == 16
    assert sum(prompt.split == "test" for prompt in prompts) == 16
    assert len({prompt.prompt_id for prompt in prompts}) == 32


def test_blind_export_hides_method_but_keeps_private_mapping(tmp_path: Path) -> None:
    rows = [
        {
            "blind_id": "blind-a",
            "generation_id": "generation-a",
            "method": "j_component_intervention",
            "concept_id": "concept:a",
            "condition_id": "j",
            "grid_point": {"strength": 0.25},
            "prompt_id": "open_test_1",
            "prompt_text": "Write something.",
            "decoding": "greedy",
            "generated_text": "A response.",
        }
    ]
    paths = write_generation_artifacts(tmp_path, rows)
    blind = json.loads(
        Path(paths["blind_generations"]).read_text(encoding="utf-8").strip()
    )
    mapping = json.loads(Path(paths["blind_map"]).read_text(encoding="utf-8").strip())

    assert "method" not in blind
    assert "condition_id" not in blind
    assert mapping["method"] == "j_component_intervention"
    assert mapping["condition_id"] == "j"


def test_generation_validator_rejects_hash_valid_empty_files(
    tmp_path: Path,
) -> None:
    files = {}
    for filename, hash_key in (
        ("generations.jsonl", "generations_sha256"),
        ("judge_blind_generations.jsonl", "blind_generations_sha256"),
        ("judge_blind_map.jsonl", "blind_map_sha256"),
    ):
        path = tmp_path / filename
        path.write_text("", encoding="utf-8")
        files[hash_key] = sha256_file(path)

    with pytest.raises(InterventionGenerationError, match="row count"):
        validate_generation_artifacts(
            tmp_path,
            files,
            contract={
                "schema_version": 1,
                "prompt_ids": ["prompt-a"],
                "prompt_ids_sha256": prompt_ids_sha256(["prompt-a"]),
                "prompt_count": 1,
                "candidate_prompt_ids": ["prompt-a"],
                "open_prompt_ids": [],
                "sample_seeds": [1001, 2002, 3003],
                "decodings_per_prompt": 4,
                "generation_settings": {
                    "sample_seeds": [1001, 2002, 3003],
                    "max_new_tokens": 128,
                    "temperature": 0.75,
                    "top_p": 0.95,
                    "repetition_penalty": 1.1,
                    "no_repeat_ngram_size": 3,
                },
                "expected_rows": 4,
            },
        )


def test_nonzero_intervention_requires_hook_telemetry(tmp_path: Path) -> None:
    settings = {
        "sample_seeds": [1001],
        "max_new_tokens": 8,
        "temperature": 0.75,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
    }
    rows = [
        {
            "generation_id": f"generation-{decoding}",
            "blind_id": f"blind-{decoding}",
            "method": "j_component_intervention",
            "concept_id": "concept:a",
            "condition_id": "j",
            "grid_point": {"strength": 0.5},
            "intervention_metadata": {"selected_layers": [3]},
            "prompt_id": "prompt-a",
            "prompt_text": "Prompt.",
            "decoding": decoding,
            "seed": seed,
            "generated_token_ids": [1],
            "generated_text": "Result.",
            "token_log_probabilities": [-0.5],
            "telemetry": [],
            "injected_norm_by_layer": {},
            "total_injected_norm": 0.0,
            "generation_settings": settings,
        }
        for decoding, seed in (("greedy", None), ("sample", 1001))
    ]
    files = write_generation_artifacts(tmp_path, rows)

    with pytest.raises(InterventionGenerationError, match="telemetry"):
        validate_generation_artifacts(
            tmp_path,
            files,
            contract={
                "schema_version": 1,
                "prompt_ids": ["prompt-a"],
                "prompt_ids_sha256": prompt_ids_sha256(["prompt-a"]),
                "prompt_count": 1,
                "candidate_prompt_ids": ["prompt-a"],
                "open_prompt_ids": [],
                "sample_seeds": [1001],
                "decodings_per_prompt": 2,
                "expected_rows": 2,
                "generation_settings": settings,
            },
        )


def test_generation_validator_recomputes_layer_and_total_norms(
    tmp_path: Path,
) -> None:
    settings = {
        "sample_seeds": [1001],
        "max_new_tokens": 8,
        "temperature": 0.75,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
    }
    telemetry = [
        {
            "layer": 3,
            "forward_call": 0,
            "generation_step": 0,
            "batch_size": 1,
            "sequence_length": 1,
            "injected_norm": 1.0,
        }
    ]
    rows = [
        {
            "generation_id": f"generation-{decoding}",
            "blind_id": f"blind-{decoding}",
            "method": "j_component_intervention",
            "concept_id": "concept:a",
            "condition_id": "j",
            "grid_point": {"strength": 0.5},
            "intervention_metadata": {"selected_layers": [3]},
            "prompt_id": "prompt-a",
            "prompt_text": "Prompt.",
            "decoding": decoding,
            "seed": seed,
            "generated_token_ids": [1],
            "generated_text": "Result.",
            "token_log_probabilities": [-0.5],
            "telemetry": telemetry,
            "injected_norm_by_layer": {"3": 2.0},
            "total_injected_norm": 2.0,
            "generation_settings": settings,
        }
        for decoding, seed in (("greedy", None), ("sample", 1001))
    ]
    files = write_generation_artifacts(tmp_path, rows)

    with pytest.raises(InterventionGenerationError, match="norm mismatch"):
        validate_generation_artifacts(
            tmp_path,
            files,
            contract={
                "schema_version": 1,
                "prompt_ids": ["prompt-a"],
                "prompt_ids_sha256": prompt_ids_sha256(["prompt-a"]),
                "prompt_count": 1,
                "candidate_prompt_ids": ["prompt-a"],
                "open_prompt_ids": [],
                "sample_seeds": [1001],
                "decodings_per_prompt": 2,
                "expected_rows": 2,
                "generation_settings": settings,
            },
        )
