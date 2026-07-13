from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from jlens_workspace.data import (
    AxBenchAllowlist,
    AxBenchConcept,
    AxBenchPreparationError,
    Concept500Allowlist,
    Concept500Concept,
    Concept500PreparationError,
    ConceptExample,
    DatasetValidationError,
    GoEmotionsAllowlist,
    GoEmotionsConcept,
    adapt_axbench_rows,
    adapt_concept500_rows,
    deterministic_group_split,
    discover_axbench_concepts,
    load_axbench_allowlist,
    load_concept500_allowlist,
    load_go_emotions_allowlist,
    load_jsonl,
    load_jsonl_directory,
    prepare_concept500_rows,
    prepare_examples,
    select_go_emotions,
    validate_examples,
)

ROOT = Path(__file__).resolve().parents[1]
BUILTIN_DATA = ROOT / "Concept_intervention/data/builtin_abstract_concepts.jsonl"
AXBENCH_ALLOWLIST = ROOT / "Concept_intervention/data/axbench_abstract_allowlist.json"
CONCEPT500_ALLOWLIST = (
    ROOT / "Concept_intervention/data/axbench_concept500_abstract_allowlist.json"
)
GO_EMOTIONS_ALLOWLIST = (
    ROOT / "Concept_intervention/data/go_emotions_7concept_allowlist.json"
)
GO_EMOTIONS_FULL_ALLOWLIST = (
    ROOT / "Concept_intervention/data/go_emotions_7concept_full_allowlist.json"
)


def _record(**changes: object) -> ConceptExample:
    values: dict[str, object] = {
        "concept_id": "test:care",
        "concept_name": "Care",
        "definition": "Attentive concern for the needs and well-being of others.",
        "abstractness": "abstract",
        "label": 1,
        "text": "She notices the visitor is cold and offers a blanket.",
        "prompt": "What does the host do?",
        "response": "She notices the visitor is cold and offers a blanket.",
        "split": "train",
        "group_id": "care:train:positive:1",
        "source": "unit-test",
        "license": "CC0-1.0",
    }
    values.update(changes)
    return ConceptExample(**values)  # type: ignore[arg-type]


def _axbench_row(
    index: int,
    *,
    category: str,
    output_concept: str = "EEEEE",
    concept_id: int = -1,
) -> dict[str, object]:
    return {
        "input": f"Prompt {index}",
        "output": f"Substantive response {index}",
        "output_concept": output_concept,
        "concept_genre": "text",
        "category": category,
        "dataset_category": "instruction",
        "concept_id": concept_id,
    }


def _concept500_row(
    index: int,
    *,
    category: str,
    output_concept: str,
    concept_id: int,
) -> dict[str, object]:
    return {
        "input": f"Concept500 prompt {index}",
        "output": f"Concept500 response {index}",
        "output_concept": output_concept,
        "concept_genre": "text",
        "category": category,
        "dataset_category": "instruction",
        "concept_id": concept_id,
    }


def test_builtin_dataset_is_balanced_abstract_and_leak_free() -> None:
    examples = load_jsonl(BUILTIN_DATA, min_per_label_per_split=2)
    statistics = validate_examples(examples, min_per_label_per_split=2)

    assert len(examples) == 96
    assert statistics.total_concepts == 6
    assert statistics.total_groups == 48
    assert statistics.counts_by_split == {"train": 48, "validation": 24, "test": 24}
    assert statistics.counts_by_label == {0: 48, 1: 48}
    assert {example.abstractness for example in examples} == {"abstract"}

    for concept in statistics.counts_by_concept.values():
        assert concept["total"] == 16
        assert concept["splits"] == {
            "train": {0: 4, 1: 4},
            "validation": {0: 2, 1: 2},
            "test": {0: 2, 1: 2},
        }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"label": 2}, "binary"),
        ({"label": True}, "booleans"),
        ({"abstractness": "concrete"}, "exact string 'abstract'"),
        ({"text": "   "}, "text must be non-empty"),
        ({"split": "dev"}, "split must be one of"),
    ],
)
def test_schema_rejects_invalid_fields(change: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _record(**change)


def test_schema_rejects_missing_and_unknown_mapping_fields() -> None:
    raw = _record().to_dict()
    del raw["definition"]
    raw["typo"] = "value"

    with pytest.raises(ValueError, match="missing fields: definition; unknown fields: typo"):
        ConceptExample.from_dict(raw)


def test_schema_allows_empty_prompt_and_response_for_labeled_text() -> None:
    example = _record(prompt="", response="")
    assert example.text
    assert example.prompt == example.response == ""


def test_jsonl_loader_reports_file_and_line(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(json.dumps(_record().to_dict()) + "\n{not-json}\n", encoding="utf-8")

    with pytest.raises(DatasetValidationError) as error:
        load_jsonl(invalid, validate=False)
    assert f"{invalid}:2" in str(error.value)
    assert "invalid JSON" in str(error.value)


def test_validator_detects_group_leakage_duplicates_and_missing_class() -> None:
    examples = load_jsonl(BUILTIN_DATA)
    duplicate_across_split = replace(examples[0], split="test")

    with pytest.raises(DatasetValidationError) as error:
        validate_examples([*examples, duplicate_across_split])
    message = str(error.value)
    assert "crosses splits" in message
    assert "duplicate normalized content" in message

    positives_only = [example for example in examples if example.label == 1]
    with pytest.raises(DatasetValidationError, match="label 0 has 0 examples"):
        validate_examples(positives_only)


def test_deterministic_split_is_order_independent_and_stratified() -> None:
    examples = load_jsonl(BUILTIN_DATA)
    split_a = deterministic_group_split(examples, seed=1729)
    split_b = deterministic_group_split(reversed(examples), seed=1729)

    assignment_a = {example.group_id: example.split for example in split_a}
    assignment_b = {example.group_id: example.split for example in split_b}
    assert assignment_a == assignment_b

    statistics = validate_examples(split_a, min_per_label_per_split=2)
    for concept in statistics.counts_by_concept.values():
        assert concept["splits"] == {
            "train": {0: 4, 1: 4},
            "validation": {0: 2, 1: 2},
            "test": {0: 2, 1: 2},
        }

    skewed = deterministic_group_split(
        examples,
        seed=1729,
        ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
    )
    skewed_statistics = validate_examples(skewed)
    for concept in skewed_statistics.counts_by_concept.values():
        assert concept["splits"] == {
            "train": {0: 6, 1: 6},
            "validation": {0: 1, 1: 1},
            "test": {0: 1, 1: 1},
        }

    paired_group = [
        _record(group_id="mixed"),
        _record(
            label=0,
            text="The host closes the door.",
            response="The host closes the door.",
            group_id="mixed",
        ),
    ]
    paired_split = deterministic_group_split(
        paired_group,
        ratios={"train": 1.0, "validation": 0.0, "test": 0.0},
    )
    assert {record.split for record in paired_split} == {"train"}

    mixed_concept_group = [
        _record(group_id="mixed-concepts"),
        _record(
            concept_id="responsibility",
            concept_name="Responsibility",
            definition="Owning duties and consequences.",
            group_id="mixed-concepts",
        ),
    ]
    with pytest.raises(DatasetValidationError, match="mixes concepts"):
        deterministic_group_split(mixed_concept_group)


def test_prepare_writes_splits_statistics_manifest_and_stable_fingerprint(
    tmp_path: Path,
) -> None:
    examples = load_jsonl(BUILTIN_DATA)
    prepared = prepare_examples(examples, output_dir=tmp_path, seed=7)
    reversed_prepared = prepare_examples(reversed(examples), seed=7)

    assert prepared.fingerprint == reversed_prepared.fingerprint
    assert {path.name for path in tmp_path.iterdir()} == {
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "statistics.json",
        "manifest.json",
    }
    reloaded = load_jsonl_directory(tmp_path, min_per_label_per_split=2)
    assert len(reloaded) == 96
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    statistics = json.loads((tmp_path / "statistics.json").read_text(encoding="utf-8"))
    assert manifest["fingerprint"] == prepared.fingerprint
    assert statistics["total_examples"] == 96

    # Auxiliary J-lens prompts are not part of the declared probe dataset.
    (tmp_path / "fit_prompts.jsonl").write_text(
        '{"source_id":"fit-1","text":"independent prompt"}\n',
        encoding="utf-8",
    )
    assert len(load_jsonl_directory(tmp_path, min_per_label_per_split=2)) == 96


def test_minimum_per_concept_is_a_hard_validation_constraint() -> None:
    examples = load_jsonl(BUILTIN_DATA)
    with pytest.raises(DatasetValidationError, match="requires at least 17"):
        validate_examples(examples, min_per_concept=17)


def test_checked_in_go_emotions_allowlist_guarantees_2000_per_concept() -> None:
    allowlist = load_go_emotions_allowlist(GO_EMOTIONS_ALLOWLIST)
    assert allowlist.revision == "add492243ff905527e67aeb8b80c082af02207c3"
    assert len(allowlist.concepts) == 7
    assert sum(allowlist.positive_per_split.values()) == 1000
    assert sum(allowlist.negative_per_split.values()) == 1000


def test_checked_in_full_go_emotions_allowlist_uses_every_multilabel_row() -> None:
    allowlist = load_go_emotions_allowlist(GO_EMOTIONS_FULL_ALLOWLIST)
    assert allowlist.selection_mode == "all_one_vs_rest"
    assert allowlist.include_multilabel is True
    assert allowlist.positive_per_split is None
    assert allowlist.negative_per_split is None
    assert allowlist.fit_prompts_disjoint is False

    small = replace(
        allowlist,
        lens_fit_prompt_count=1,
        lens_fit_min_words=2,
        concepts=allowlist.concepts[:2],
    )
    rows = {
        split: [
            {
                "id": f"{split}-both",
                "text": f"{split} both labels",
                "labels": [small.concepts[0].label_id, small.concepts[1].label_id],
            },
            {
                "id": f"{split}-neither",
                "text": f"{split} neither label",
                "labels": [27],
            },
        ]
        for split in ("train", "validation", "test")
    }
    selected = select_go_emotions(rows, small, seed=3)
    assert len(selected.examples) == 2 * 2 * 3
    both = [
        example for example in selected.examples if example.group_id.endswith("-both")
    ]
    assert len(both) == 2 * 3
    assert {example.label for example in both} == {1}
    assert selected.fit_prompts[0]["source_id"] in {"train-both", "train-neither"}


def test_go_emotions_selection_is_binary_single_label_and_fit_disjoint() -> None:
    allowlist = GoEmotionsAllowlist(
        dataset_id="google-research-datasets/go_emotions",
        revision="test-revision",
        config="simplified",
        license="Apache-2.0",
        file_sha256={split: "0" * 64 for split in ("train", "validation", "test")},
        positive_per_split={split: 1 for split in ("train", "validation", "test")},
        negative_per_split={split: 1 for split in ("train", "validation", "test")},
        lens_fit_prompt_count=2,
        lens_fit_min_words=3,
        concepts=(
            GoEmotionsConcept(
                label_id=0,
                concept_name="admiration",
                definition="Regard for someone or something as worthy of esteem.",
            ),
        ),
    )
    rows: dict[str, list[dict[str, object]]] = {}
    for split in ("train", "validation", "test"):
        rows[split] = [
            {"id": f"{split}-positive", "text": f"{split} worthy praise", "labels": [0]},
            {"id": f"{split}-negative", "text": f"{split} unrelated neutral", "labels": [27]},
        ]
    rows["train"].extend(
        [
            {"id": "fit-a", "text": "long independent fitting prompt alpha", "labels": [4]},
            {"id": "fit-b", "text": "long independent fitting prompt beta", "labels": [7]},
            {"id": "multi", "text": "multi label row excluded", "labels": [0, 4]},
        ]
    )

    selected = select_go_emotions(rows, allowlist, seed=3)
    assert len(selected.examples) == 6
    assert {example.label for example in selected.examples} == {0, 1}
    assert all(example.prompt == example.response == "" for example in selected.examples)
    probe_ids = {example.group_id.removeprefix("goemotions:") for example in selected.examples}
    fit_ids = {prompt["source_id"] for prompt in selected.fit_prompts}
    assert len(fit_ids) == 2
    assert probe_ids.isdisjoint(fit_ids)


def test_checked_in_axbench_allowlist_fails_before_remote_loading() -> None:
    allowlist = load_axbench_allowlist(AXBENCH_ALLOWLIST)
    assert allowlist.dataset_id == "pyvene/axbench-concept16k"

    with pytest.raises(AxBenchPreparationError, match="not verified"):
        allowlist.ensure_verified()


def test_axbench_adapter_accepts_generator_and_uses_observed_remote_id() -> None:
    revision = "test-revision"
    allowlist = AxBenchAllowlist(
        dataset_id="pyvene/axbench-concept16k",
        revision=revision,
        variant="2b/l20",
        license="CC-BY-4.0",
        concepts=(
            AxBenchConcept(
                concept_name="Integrity",
                definition="Consistency between stated principles and conduct.",
                search_terms=("integrity",),
                remote_output_concept="conduct aligned with stated moral principles",
                verified_against_revision=revision,
            ),
        ),
    )
    rows = [
        *[_axbench_row(index, category="negative") for index in range(3)],
        *[
            _axbench_row(
                index + 3,
                category="positive",
                output_concept="conduct aligned with stated moral principles",
                concept_id=314,
            )
            for index in range(3)
        ],
    ]

    examples = adapt_axbench_rows((row for row in rows), allowlist, per_label=3)
    assert len(examples) == 6
    assert {example.concept_id for example in examples} == {"axbench:2b-l20:314"}
    assert {example.label for example in examples} == {0, 1}
    assert all(example.source.endswith("2b/l20/train/data.parquet") for example in examples)
    prepared = prepare_examples(examples, seed=11)
    assert prepared.statistics.counts_by_split == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }


def test_axbench_discovery_returns_only_observed_positive_candidates() -> None:
    rows = [
        _axbench_row(0, category="negative"),
        _axbench_row(
            1,
            category="positive",
            output_concept="language expressing epistemic uncertainty",
            concept_id=19,
        ),
    ]
    discoveries = discover_axbench_concepts(rows, ["uncertainty", "honesty"])

    assert discoveries == {
        "uncertainty": [
            {
                "concept_id": 19,
                "output_concept": "language expressing epistemic uncertainty",
            }
        ],
        "honesty": [],
    }


def test_checked_in_concept500_allowlist_is_pinned_and_abstract() -> None:
    allowlist = load_concept500_allowlist(CONCEPT500_ALLOWLIST)

    assert allowlist.dataset_id == "pyvene/axbench-concept500"
    assert allowlist.revision == "ad8a5d60c4616b599c24dd6689f05f696ec610f3"
    assert allowlist.variant == "9b/l20"
    assert len(allowlist.concepts) == 7
    assert {concept.remote_concept_id for concept in allowlist.concepts} == {
        68,
        82,
        114,
        178,
        195,
        212,
        361,
    }


def test_concept500_adapter_preserves_official_test_and_builds_validation(
    tmp_path: Path,
) -> None:
    target_name = "questions and statements challenging beliefs or assumptions"
    allowlist = Concept500Allowlist(
        dataset_id="pyvene/axbench-concept500",
        revision="test-revision",
        variant="9b/l20",
        license="CC-BY-4.0",
        train_sha256="0" * 64,
        test_sha256="1" * 64,
        concepts=(
            Concept500Concept(
                remote_concept_id=178,
                concept_name=target_name,
                definition="Language that challenges an assumption.",
            ),
        ),
    )
    train_rows = [
        *[
            _concept500_row(
                index,
                category="negative",
                output_concept="EEEEE",
                concept_id=-1,
            )
            for index in range(4)
        ],
        *[
            _concept500_row(
                index + 100,
                category="positive",
                output_concept=target_name,
                concept_id=178,
            )
            for index in range(4)
        ],
    ]
    test_rows = [
        *[
            _concept500_row(
                index + 200,
                category="positive",
                output_concept=target_name,
                concept_id=178,
            )
            for index in range(2)
        ],
        *[
            _concept500_row(
                index + 300,
                category="negative",
                output_concept=target_name,
                concept_id=178,
            )
            for index in range(2)
        ],
        _concept500_row(
            400,
            category="hard negative",
            output_concept="question//a request for information",
            concept_id=178,
        ),
    ]
    # Prompt groups may contain multiple labels/responses and must remain atomic.
    train_rows[0]["input"] = train_rows[4]["input"]
    test_rows[2]["input"] = test_rows[0]["input"]

    examples = adapt_concept500_rows(train_rows, test_rows, allowlist, seed=7)
    statistics = validate_examples(examples)
    assert statistics.total_examples == 13
    assert statistics.counts_by_split == {"train": 6, "validation": 2, "test": 5}
    assert statistics.counts_by_concept[
        "axbench-concept500:9b-l20:178"
    ]["splits"] == {
        "train": {0: 3, 1: 3},
        "validation": {0: 1, 1: 1},
        "test": {0: 3, 1: 2},
    }
    assert all("@test-revision/9b/l20/" in example.source for example in examples)
    for shared_prompt in (train_rows[0]["input"], test_rows[0]["input"]):
        matched = [example for example in examples if example.prompt == shared_prompt]
        assert len(matched) == 2
        assert len({example.group_id for example in matched}) == 1
        assert len({example.split for example in matched}) == 1

    prepared = prepare_concept500_rows(
        train_rows,
        test_rows,
        allowlist,
        tmp_path,
        seed=7,
        min_per_label_per_split=1,
    )
    assert prepared.statistics.total_examples == 13
    assert len(load_jsonl_directory(tmp_path)) == 13


def test_concept500_adapter_rejects_id_name_drift() -> None:
    allowlist = Concept500Allowlist(
        dataset_id="pyvene/axbench-concept500",
        revision="test-revision",
        variant="9b/l20",
        license="CC-BY-4.0",
        train_sha256="0" * 64,
        test_sha256="1" * 64,
        concepts=(
            Concept500Concept(
                remote_concept_id=5,
                concept_name="expected abstract concept",
                definition="An expected concept.",
            ),
        ),
    )
    train_rows = [
        _concept500_row(0, category="negative", output_concept="EEEEE", concept_id=-1),
        _concept500_row(
            1,
            category="positive",
            output_concept="a different concept",
            concept_id=5,
        ),
    ]

    with pytest.raises(Concept500PreparationError, match="does not match pinned name"):
        adapt_concept500_rows(train_rows, [], allowlist)
