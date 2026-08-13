from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from jlens_workspace import cli
from jlens_workspace.artifacts import sha256_file
from jlens_workspace.concept_intervention.evaluation import load_prompt_bank
from jlens_workspace.concept_intervention.generation import (
    InterventionGenerationError,
    load_open_prompt_bank,
    prompt_ids_sha256,
    validate_candidate_score_artifact,
    validate_generation_artifacts,
    validate_shard_manifest_identity,
    write_generation_artifacts,
)
from jlens_workspace.config import GenerationConfig


class _Tokenizer:
    chat_template = None


_SETTINGS = {
    "sample_seeds": [1001],
    "max_new_tokens": 8,
    "temperature": 0.75,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
}
_CANDIDATE_LABELS = {
    f"concept:{letter}": letter for letter in "abcdefg"
}


def _contract(prompt_id: str = "choose_0") -> dict[str, object]:
    return {
        "schema_version": 1,
        "prompt_ids": [prompt_id],
        "prompt_ids_sha256": prompt_ids_sha256([prompt_id]),
        "prompt_count": 1,
        "candidate_prompt_ids": [prompt_id],
        "candidate_prompt_splits": {prompt_id: "validation"},
        "candidate_labels": _CANDIDATE_LABELS,
        "open_prompt_ids": [],
        "sample_seeds": [1001],
        "decodings_per_prompt": 2,
        "expected_rows": 2,
        "generation_settings": _SETTINGS,
    }


def _generation_rows(
    *,
    method: str,
    grid_point: dict[str, object],
    intervention_metadata: dict[str, object],
    telemetry: list[dict[str, object]],
    token_ids: list[int] | None = None,
) -> list[dict[str, object]]:
    generated = token_ids or [1]
    injected_by_layer: dict[str, float] = {}
    for event in telemetry:
        layer = str(event["layer"])
        injected_by_layer[layer] = injected_by_layer.get(layer, 0.0) + float(
            event["injected_norm"]
        )
    return [
        {
            "generation_id": f"generation-{method}-{decoding}",
            "blind_id": f"blind-{method}-{decoding}",
            "method": method,
            "concept_id": "concept:a",
            "condition_id": "condition",
            "grid_point": grid_point,
            "intervention_metadata": intervention_metadata,
            "prompt_id": "choose_0",
            "prompt_text": "Prompt.",
            "decoding": decoding,
            "seed": seed,
            "generated_token_ids": generated,
            "generated_text": "Result.",
            "token_log_probabilities": [-0.5] * len(generated),
            "telemetry": telemetry,
            "injected_norm_by_layer": injected_by_layer,
            "total_injected_norm": sum(injected_by_layer.values()),
            "generation_settings": _SETTINGS,
        }
        for decoding, seed in (("greedy", None), ("sample", 1001))
    ]


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


def test_generation_resources_resolve_from_registered_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "code"
    run_root = tmp_path / "run"
    data_root = repository_root / "Concept_intervention" / "data"
    data_root.mkdir(parents=True)
    run_root.mkdir()
    candidate_path = data_root / "candidates.json"
    open_path = data_root / "open.json"
    candidate_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "prompts": [
                    {
                        "prompt_id": "choose_0",
                        "label_rotation": 0,
                        "text": "Choose from {labels}.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    open_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "prompts": [
                    {
                        "prompt_id": "open_validation_0",
                        "split": "validation",
                        "text": "Write something.",
                    },
                    {
                        "prompt_id": "open_test_0",
                        "split": "test",
                        "text": "Write something else.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(run_root)
    monkeypatch.setenv("JLENS_REPOSITORY_ROOT", str(repository_root))
    candidate_relative = "Concept_intervention/data/candidates.json"
    open_relative = "Concept_intervention/data/open.json"

    candidate_prompts = load_prompt_bank(
        candidate_relative,
        tokenizer=_Tokenizer(),
        candidate_labels=_CANDIDATE_LABELS,
    )
    open_prompts = load_open_prompt_bank(open_relative, tokenizer=_Tokenizer())
    identity = cli._generation_identity(
        GenerationConfig(
            candidate_prompts_path=candidate_relative,
            open_prompts_path=open_relative,
            sample_seeds=[1001],
        ),
        _CANDIDATE_LABELS,
    )

    assert [prompt.prompt_id for prompt in candidate_prompts] == ["choose_0"]
    assert [prompt.prompt_id for prompt in open_prompts] == [
        "open_validation_0",
        "open_test_0",
    ]
    assert identity["candidate_prompts_sha256"] == sha256_file(candidate_path)
    assert identity["open_prompts_sha256"] == sha256_file(open_path)


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
            "active_positions": 1,
            "strength": 0.5,
            "residual_norm": 2.0,
            "kind": "addition",
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


def test_candidate_scores_require_all_frozen_labels_and_family_split(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate_scores.jsonl"
    base = {
        "prompt_id": "choose_0",
        "method": "j_component_intervention",
        "target_concept_id": "concept:a",
        "evaluation_split": "validation",
        "condition_id": "j",
        "strength": 0.5,
        "candidate_log_probabilities": {
            "concept:a": -1.0,
            "concept:b": -2.0,
        },
        "candidate_probabilities_normalized": {
            "concept:a": 0.7,
            "concept:b": 0.3,
        },
        "target_log_probability": -1.0,
        "target_candidate_probability": 0.7,
        "target_margin": 1.0,
        "target_rank": 1,
    }
    path.write_text(json.dumps(base) + "\n", encoding="utf-8")
    with pytest.raises(InterventionGenerationError, match="candidate labels"):
        validate_candidate_score_artifact(
            path,
            expected_sha256=sha256_file(path),
            contract=_contract(),
            expected_method="j_component_intervention",
            expected_concept_id="concept:a",
            expected_grid_condition={"condition_id": "j", "strength": 0.5},
        )

    log_probabilities = {
        concept_id: -float(index + 1)
        for index, concept_id in enumerate(_CANDIDATE_LABELS)
    }
    denominator = sum(
        math.exp(value) for value in log_probabilities.values()
    )
    probabilities = {
        concept_id: math.exp(value) / denominator
        for concept_id, value in log_probabilities.items()
    }
    off_target = [
        value
        for concept_id, value in log_probabilities.items()
        if concept_id != "concept:a"
    ]
    wrong_split = {
        **base,
        "evaluation_split": "test",
        "candidate_log_probabilities": log_probabilities,
        "candidate_probabilities_normalized": probabilities,
        "target_candidate_probability": probabilities["concept:a"],
        "target_margin": -1.0 - sum(off_target) / len(off_target),
    }
    path.write_text(json.dumps(wrong_split) + "\n", encoding="utf-8")
    with pytest.raises(InterventionGenerationError, match="prompt family"):
        validate_candidate_score_artifact(
            path,
            expected_sha256=sha256_file(path),
            contract=_contract(),
            expected_method="j_component_intervention",
            expected_concept_id="concept:a",
            expected_grid_condition={"condition_id": "j", "strength": 0.5},
        )

    tampered_contract = {
        **_contract(),
        "candidate_prompt_splits": {"choose_0": "test"},
    }
    with pytest.raises(InterventionGenerationError, match="fixed prompt splits"):
        validate_candidate_score_artifact(
            path,
            expected_sha256=sha256_file(path),
            contract=tampered_contract,
            expected_method="j_component_intervention",
            expected_concept_id="concept:a",
            expected_grid_condition={"condition_id": "j", "strength": 0.5},
        )


def test_telemetry_covers_every_generated_step_at_every_layer(
    tmp_path: Path,
) -> None:
    telemetry = [
        {
            "layer": 3,
            "forward_call": 0,
            "generation_step": 0,
            "batch_size": 1,
            "sequence_length": 4,
            "active_positions": 1,
            "strength": 0.5,
            "residual_norm": 2.0,
            "kind": "addition",
            "injected_norm": 1.0,
        }
    ]
    rows = _generation_rows(
        method="j_component_intervention",
        grid_point={"strength": 0.5},
        intervention_metadata={"selected_layers": [3]},
        telemetry=telemetry,
        token_ids=[1, 2],
    )
    files = write_generation_artifacts(tmp_path, rows)

    with pytest.raises(InterventionGenerationError, match="generated-token coverage"):
        validate_generation_artifacts(
            tmp_path,
            files,
            contract=_contract(),
        )


@pytest.mark.parametrize(
    ("method", "grid_point", "metadata"),
    [
        (
            "j_component_intervention",
            {"strength": 0.5},
            {"selected_layers": [3]},
        ),
        (
            "iti_intervention",
            {"variant": "native", "top_k": 4, "strength": 5.0},
            {"active_layers": [3]},
        ),
        (
            "raptor_intervention",
            {"target_probability": 0.9},
            {"selected_layers": [3]},
        ),
    ],
)
def test_telemetry_requires_method_specific_fields(
    tmp_path: Path,
    method: str,
    grid_point: dict[str, object],
    metadata: dict[str, object],
) -> None:
    telemetry = [
        {
            "layer": 3,
            "forward_call": 0,
            "generation_step": 0,
            "batch_size": 1,
            "sequence_length": 4,
            "injected_norm": 1.0,
        }
    ]
    rows = _generation_rows(
        method=method,
        grid_point=grid_point,
        intervention_metadata=metadata,
        telemetry=telemetry,
    )
    files = write_generation_artifacts(tmp_path, rows)

    with pytest.raises(InterventionGenerationError, match="method-specific"):
        validate_generation_artifacts(
            tmp_path,
            files,
            contract=_contract(),
        )


def _scientific_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_name": "experiment",
        "seed": 42,
        "model_id": "model",
        "model_revision": "model-revision",
        "tokenizer_id": "tokenizer",
        "tokenizer_revision": "tokenizer-revision",
        "lens_source": "local:lens",
        "lens_revision": None,
        "dataset_source": "dataset",
        "dataset_revision": "dataset-revision",
        "dataset_hash": "dataset-hash",
        "git_commit": "a" * 40,
        "python": "3.12.13",
        "platform": "Linux-old",
        "packages": {"torch": "2.7.1"},
        "notes": {
            "direction": "concept_intervention",
            "coordinate": "resid_post/block_output",
            "config_sha256": "b" * 64,
            "force_bos": False,
            "workflow": "j_component_intervention",
            "generation": {"prompt_ids_sha256": "c" * 64},
            "selected_layers_sha256": "d" * 64,
            "row_manifest_sha256": "e" * 64,
        },
    }


def test_shard_manifest_identity_accepts_identical_and_mixed_platforms() -> None:
    root = _scientific_manifest()
    validate_shard_manifest_identity(root, copy.deepcopy(root))

    mixed_platform = copy.deepcopy(root)
    mixed_platform["platform"] = "Linux-new"
    validate_shard_manifest_identity(root, mixed_platform)


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("git_commit",), "f" * 40),
        (("model_revision",), "other-model"),
        (("dataset_hash",), "other-data"),
        (("python",), "3.13.0"),
        (("packages",), {"torch": "different"}),
        (("notes", "config_sha256"), "0" * 64),
        (("notes", "generation"), {"prompt_ids_sha256": "0" * 64}),
    ],
)
def test_shard_manifest_identity_rejects_scientific_or_runtime_changes(
    field_path: tuple[str, ...],
    replacement: object,
) -> None:
    root = _scientific_manifest()
    shard = copy.deepcopy(root)
    if len(field_path) == 1:
        shard[field_path[0]] = replacement
    else:
        notes = shard["notes"]
        assert isinstance(notes, dict)
        notes[field_path[1]] = replacement

    with pytest.raises(InterventionGenerationError, match="scientific identity"):
        validate_shard_manifest_identity(root, shard)


@pytest.mark.parametrize("platform_value", [None, "", 7])
def test_shard_manifest_identity_rejects_missing_or_malformed_platform(
    platform_value: object,
) -> None:
    root = _scientific_manifest()
    shard = copy.deepcopy(root)
    if platform_value is None:
        shard.pop("platform")
    else:
        shard["platform"] = platform_value

    with pytest.raises(InterventionGenerationError, match="platform"):
        validate_shard_manifest_identity(root, shard)
