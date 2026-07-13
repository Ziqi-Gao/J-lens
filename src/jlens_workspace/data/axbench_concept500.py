"""Strict preparation for the official AxBench Concept500 train/test files.

Concept500 provides 72 positive training examples per concept and a shared pool
of generic training negatives.  Its test file provides 36 positive and 36
ordinary negative examples per concept, plus a small number of polysemantic
hard negatives.  This adapter preserves the official test split, creates a
held-out validation split from training data, and reuses the shared negative
pool independently for each binary concept task.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prepare import PreparedDataset, prepare_examples
from .schema import ConceptExample

CONCEPT500_DATASET_ID = "pyvene/axbench-concept500"
CONCEPT500_REQUIRED_FIELDS = frozenset(
    {
        "input",
        "output",
        "output_concept",
        "concept_genre",
        "category",
        "dataset_category",
        "concept_id",
    }
)


class Concept500PreparationError(ValueError):
    """Raised when pinned Concept500 inputs do not satisfy the expected contract."""


@dataclass(frozen=True, slots=True)
class Concept500Concept:
    remote_concept_id: int
    concept_name: str
    definition: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Concept500Concept:
        expected = {"remote_concept_id", "concept_name", "definition"}
        if set(raw) != expected:
            raise Concept500PreparationError(
                f"Concept500 concept keys must be exactly {sorted(expected)}, "
                f"got {sorted(raw)}"
            )
        remote_id = raw["remote_concept_id"]
        if isinstance(remote_id, bool) or not isinstance(remote_id, int) or remote_id < 0:
            raise Concept500PreparationError("remote_concept_id must be a non-negative integer")
        for field_name in ("concept_name", "definition"):
            value = raw[field_name]
            if not isinstance(value, str) or not value.strip():
                raise Concept500PreparationError(f"{field_name} must be a non-empty string")
        return cls(
            remote_concept_id=remote_id,
            concept_name=raw["concept_name"],
            definition=raw["definition"],
        )


@dataclass(frozen=True, slots=True)
class Concept500Allowlist:
    dataset_id: str
    revision: str
    variant: str
    license: str
    train_sha256: str
    test_sha256: str
    concepts: tuple[Concept500Concept, ...]


def _sha256_value(value: str, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise Concept500PreparationError(f"{field_name} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise Concept500PreparationError(f"{field_name} must be hexadecimal") from error
    return value.casefold()


def load_concept500_allowlist(path: str | Path) -> Concept500Allowlist:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Concept500PreparationError(f"cannot read Concept500 allowlist {source}: {error}") from error
    if not isinstance(raw, dict):
        raise Concept500PreparationError("Concept500 allowlist must be a JSON object")
    expected = {
        "schema_version",
        "dataset_id",
        "revision",
        "variant",
        "license",
        "train_sha256",
        "test_sha256",
        "concepts",
    }
    if set(raw) != expected:
        raise Concept500PreparationError(
            f"Concept500 allowlist keys must be exactly {sorted(expected)}, got {sorted(raw)}"
        )
    if raw["schema_version"] != 1:
        raise Concept500PreparationError("unsupported Concept500 allowlist schema_version")
    if raw["dataset_id"] != CONCEPT500_DATASET_ID:
        raise Concept500PreparationError(
            f"dataset_id must be the supported upstream {CONCEPT500_DATASET_ID!r}"
        )
    for field_name in ("revision", "variant", "license"):
        value = raw[field_name]
        if not isinstance(value, str) or not value.strip():
            raise Concept500PreparationError(f"{field_name} must be a non-empty string")
    if raw["variant"] != "9b/l20":
        raise Concept500PreparationError("the formal abstract-concept allowlist requires variant '9b/l20'")
    raw_concepts = raw["concepts"]
    if not isinstance(raw_concepts, list) or not raw_concepts:
        raise Concept500PreparationError("concepts must be a non-empty list")
    concepts = tuple(Concept500Concept.from_dict(item) for item in raw_concepts)
    ids = [concept.remote_concept_id for concept in concepts]
    names = [concept.concept_name.casefold() for concept in concepts]
    if len(ids) != len(set(ids)):
        raise Concept500PreparationError("remote_concept_id values must be unique")
    if len(names) != len(set(names)):
        raise Concept500PreparationError("concept_name values must be unique")
    return Concept500Allowlist(
        dataset_id=raw["dataset_id"],
        revision=raw["revision"],
        variant=raw["variant"],
        license=raw["license"],
        train_sha256=_sha256_value(raw["train_sha256"], "train_sha256"),
        test_sha256=_sha256_value(raw["test_sha256"], "test_sha256"),
        concepts=concepts,
    )


def _checked_rows(
    rows: Iterable[Mapping[str, Any]], *, split: str
) -> list[tuple[int, Mapping[str, Any]]]:
    checked: list[tuple[int, Mapping[str, Any]]] = []
    for row_index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise Concept500PreparationError(
                f"Concept500 {split} row {row_index} is not a mapping"
            )
        missing = sorted(CONCEPT500_REQUIRED_FIELDS.difference(raw))
        if missing:
            raise Concept500PreparationError(
                f"Concept500 {split} row {row_index} lacks fields: {missing}"
            )
        if raw["dataset_category"] != "instruction":
            raise Concept500PreparationError(
                f"Concept500 {split} row {row_index} has unexpected dataset_category "
                f"{raw['dataset_category']!r}"
            )
        for field_name in ("input", "output", "output_concept"):
            value = raw[field_name]
            if not isinstance(value, str):
                raise Concept500PreparationError(
                    f"Concept500 {split} row {row_index} field {field_name} is not a string"
                )
        checked.append((row_index, raw))
    return checked


def _stable_digest(*values: object) -> bytes:
    return hashlib.sha256("\0".join(str(value) for value in values).encode()).digest()


_WHITESPACE = re.compile(r"\s+")


def _normalized_prompt(value: str) -> str:
    return _WHITESPACE.sub(" ", value.strip()).casefold()


def _split_train_groups(
    positives: Sequence[tuple[int, Mapping[str, Any]]],
    negatives: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    concept_id: int,
    seed: int,
    validation_fraction: float,
) -> dict[str, str]:
    """Assign complete prompt groups while hitting exact per-label targets.

    A small two-dimensional subset-sum is sufficient because the target is 18
    validation examples per label in the formal dataset.  It also handles the
    rare prompt shared by positive/negative or repeated positive responses.
    """

    if min(len(positives), len(negatives)) < 2:
        raise Concept500PreparationError("at least two rows per label are required")
    targets = (
        min(len(negatives) - 1, max(1, round(len(negatives) * validation_fraction))),
        min(len(positives) - 1, max(1, round(len(positives) * validation_fraction))),
    )
    groups: dict[str, list[int]] = defaultdict(list)
    for label, rows in ((1, positives), (0, negatives)):
        for _, row in rows:
            groups[_normalized_prompt(str(row["input"]))].append(label)
    ranked_groups = sorted(
        groups,
        key=lambda prompt: _stable_digest(seed, concept_id, "validation-group", prompt),
    )
    states: dict[tuple[int, int], tuple[str, ...]] = {(0, 0): ()}
    for prompt in ranked_groups:
        labels = groups[prompt]
        delta = (labels.count(0), labels.count(1))
        additions: dict[tuple[int, int], tuple[str, ...]] = {}
        for state, selected in states.items():
            candidate = (state[0] + delta[0], state[1] + delta[1])
            if candidate[0] <= targets[0] and candidate[1] <= targets[1]:
                additions.setdefault(candidate, (*selected, prompt))
        for state, selected in additions.items():
            states.setdefault(state, selected)
    if targets not in states:
        raise Concept500PreparationError(
            f"cannot form a prompt-group-safe validation split with label counts {targets} "
            f"for concept ID {concept_id}"
        )
    validation_prompts = set(states[targets])
    return {
        prompt: "validation" if prompt in validation_prompts else "train"
        for prompt in groups
    }


def _group_id(
    variant: str,
    source_split: str,
    concept_id: int,
    row: Mapping[str, Any],
) -> str:
    digest = _stable_digest(
        variant,
        source_split,
        concept_id,
        _normalized_prompt(str(row["input"])),
    ).hex()[:20]
    return (
        f"axbench-concept500:{variant.replace('/', '-')}:{source_split}:"
        f"{concept_id}:prompt:{digest}"
    )


def _make_example(
    *,
    allowlist: Concept500Allowlist,
    concept: Concept500Concept,
    row_index: int,
    row: Mapping[str, Any],
    source_split: str,
    split: str,
    label: int,
) -> ConceptExample:
    prompt = row["input"].strip()
    response = row["output"].strip()
    if not prompt or not response:
        raise Concept500PreparationError(
            f"selected {source_split} row {row_index} has an empty prompt or response"
        )
    source = (
        f"hf://datasets/{allowlist.dataset_id}@{allowlist.revision}/"
        f"{allowlist.variant}/{source_split}/data.parquet"
    )
    return ConceptExample(
        concept_id=(
            f"axbench-concept500:{allowlist.variant.replace('/', '-')}:"
            f"{concept.remote_concept_id}"
        ),
        concept_name=concept.concept_name,
        definition=concept.definition,
        abstractness="abstract",
        label=label,
        text=response,
        prompt=prompt,
        response=response,
        split=split,
        group_id=_group_id(
            allowlist.variant,
            source_split,
            concept.remote_concept_id,
            row,
        ),
        source=source,
        license=allowlist.license,
    )


def adapt_concept500_rows(
    train_rows: Iterable[Mapping[str, Any]],
    test_rows: Iterable[Mapping[str, Any]],
    allowlist: Concept500Allowlist,
    *,
    seed: int = 42,
    validation_fraction: float = 0.25,
) -> list[ConceptExample]:
    """Build binary tasks using all selected positives and official test rows.

    Each concept receives as many shared-pool training negatives as it has
    training positives.  The generic pool is sampled independently per concept;
    this matches the fact that these are separate one-vs-rest probing tasks.
    """

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    checked_train = _checked_rows(train_rows, split="train")
    checked_test = _checked_rows(test_rows, split="test")
    generic_negatives = [
        item
        for item in checked_train
        if item[1]["category"] == "negative"
        and item[1]["concept_id"] == -1
        and item[1]["output_concept"] == "EEEEE"
    ]
    output: list[ConceptExample] = []

    for concept in allowlist.concepts:
        positives = [
            item
            for item in checked_train
            if item[1]["category"] == "positive"
            and item[1]["concept_id"] == concept.remote_concept_id
        ]
        mismatched_names = [
            item[1]["output_concept"]
            for item in positives
            if item[1]["output_concept"] != concept.concept_name
        ]
        if mismatched_names:
            raise Concept500PreparationError(
                f"concept ID {concept.remote_concept_id} does not match pinned name "
                f"{concept.concept_name!r}"
            )
        if not positives:
            raise Concept500PreparationError(
                f"no training positives found for concept ID {concept.remote_concept_id}"
            )
        if len(generic_negatives) < len(positives):
            raise Concept500PreparationError(
                f"only {len(generic_negatives)} generic negatives are available for "
                f"{len(positives)} positives"
            )
        negatives = sorted(
            generic_negatives,
            key=lambda item: _stable_digest(
                seed,
                concept.remote_concept_id,
                "negative-selection",
                item[1]["input"],
                item[1]["output"],
            ),
        )[: len(positives)]
        train_assignment = _split_train_groups(
            positives,
            negatives,
            concept_id=concept.remote_concept_id,
            seed=seed,
            validation_fraction=validation_fraction,
        )
        for label, selected in ((1, positives), (0, negatives)):
            output.extend(
                _make_example(
                    allowlist=allowlist,
                    concept=concept,
                    row_index=row_index,
                    row=row,
                    source_split="train",
                    split=train_assignment[_normalized_prompt(str(row["input"]))],
                    label=label,
                )
                for row_index, row in selected
            )

        selected_test = [
            item
            for item in checked_test
            if item[1]["concept_id"] == concept.remote_concept_id
            and item[1]["category"] in {"positive", "negative", "hard negative"}
        ]
        test_categories = {str(item[1]["category"]) for item in selected_test}
        if not {"positive", "negative"}.issubset(test_categories):
            raise Concept500PreparationError(
                f"test rows for concept ID {concept.remote_concept_id} lack positive/negative coverage"
            )
        train_prompts = {
            _normalized_prompt(str(row["input"]))
            for _, row in (*positives, *negatives)
        }
        test_prompts = {
            _normalized_prompt(str(row["input"])) for _, row in selected_test
        }
        prompt_overlap = train_prompts.intersection(test_prompts)
        if prompt_overlap:
            raise Concept500PreparationError(
                f"concept ID {concept.remote_concept_id} has {len(prompt_overlap)} normalized "
                "prompts crossing the official train/test boundary"
            )
        for row_index, row in selected_test:
            category = str(row["category"])
            if category in {"positive", "negative"} and row["output_concept"] != concept.concept_name:
                raise Concept500PreparationError(
                    f"test {category} row for ID {concept.remote_concept_id} has mismatched name"
                )
            output.append(
                _make_example(
                    allowlist=allowlist,
                    concept=concept,
                    row_index=row_index,
                    row=row,
                    source_split="test",
                    split="test",
                    label=int(category == "positive"),
                )
            )
    return output


def prepare_concept500_rows(
    train_rows: Iterable[Mapping[str, Any]],
    test_rows: Iterable[Mapping[str, Any]],
    allowlist: Concept500Allowlist,
    output_dir: str | Path,
    *,
    seed: int = 42,
    validation_fraction: float = 0.25,
    min_per_label_per_split: int = 18,
) -> PreparedDataset:
    examples = adapt_concept500_rows(
        train_rows,
        test_rows,
        allowlist,
        seed=seed,
        validation_fraction=validation_fraction,
    )
    return prepare_examples(
        examples,
        output_dir=output_dir,
        seed=seed,
        ratios={"train": 0.5, "validation": 0.25, "test": 0.25},
        resplit=False,
        min_per_label_per_split=min_per_label_per_split,
    )
