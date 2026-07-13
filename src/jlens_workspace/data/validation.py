"""Dataset-level validation and compact, JSON-serializable statistics."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .schema import CANONICAL_SPLITS, ConceptExample


class DatasetValidationError(ValueError):
    """Raised with all independently detectable dataset problems."""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("dataset validation failed:\n- " + "\n- ".join(self.issues))


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    """Counts emitted after successful validation."""

    total_examples: int
    total_concepts: int
    total_groups: int
    counts_by_split: Mapping[str, int]
    counts_by_label: Mapping[int, int]
    counts_by_concept: Mapping[str, Mapping[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_examples": self.total_examples,
            "total_concepts": self.total_concepts,
            "total_groups": self.total_groups,
            "counts_by_split": dict(self.counts_by_split),
            "counts_by_label": {str(label): count for label, count in self.counts_by_label.items()},
            "counts_by_concept": {
                concept_id: {
                    "concept_name": entry["concept_name"],
                    "total": entry["total"],
                    "splits": {
                        split: {str(label): count for label, count in label_counts.items()}
                        for split, label_counts in entry["splits"].items()
                    },
                }
                for concept_id, entry in self.counts_by_concept.items()
            },
        }


_WHITESPACE = re.compile(r"\s+")


def _normalized(value: str) -> str:
    return _WHITESPACE.sub(" ", value.strip()).casefold()


def validate_examples(
    examples: Iterable[ConceptExample],
    *,
    expected_splits: Sequence[str] = CANONICAL_SPLITS,
    min_per_label_per_split: int = 1,
    min_per_concept: int = 1,
) -> DatasetStatistics:
    """Validate binary coverage, leakage, duplicates, and metadata consistency.

    ``group_id`` is global rather than concept-local: reusing it in two splits
    is always an error.  Exact normalized content is checked within a concept,
    while a generic negative may legitimately be reused for another concept.
    """

    records = list(examples)
    issues: list[str] = []
    expected_splits = tuple(expected_splits)
    if not expected_splits:
        raise ValueError("expected_splits must contain at least one split")
    unknown_expected = sorted(set(expected_splits) - set(CANONICAL_SPLITS))
    if unknown_expected:
        raise ValueError(f"unknown expected splits: {unknown_expected}")
    if min_per_label_per_split < 1:
        raise ValueError("min_per_label_per_split must be at least 1")
    if min_per_concept < 1:
        raise ValueError("min_per_concept must be at least 1")
    if not records:
        raise DatasetValidationError(("dataset is empty",))

    counts: dict[str, dict[str, Counter[int]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    concept_metadata: dict[str, tuple[str, str, str]] = {}
    name_to_id: dict[str, str] = {}
    group_to_split: dict[str, str] = {}
    seen_content: dict[tuple[str, str, str, str], int] = {}

    for index, record in enumerate(records, start=1):
        # Dataclass construction performs field-level type/non-empty checks.
        counts[record.concept_id][record.split][record.label] += 1

        metadata = (record.concept_name, record.definition, record.abstractness)
        prior_metadata = concept_metadata.setdefault(record.concept_id, metadata)
        if prior_metadata != metadata:
            issues.append(
                f"concept {record.concept_id!r} has inconsistent name/definition/abstractness"
            )

        normalized_name = _normalized(record.concept_name)
        prior_id = name_to_id.setdefault(normalized_name, record.concept_id)
        if prior_id != record.concept_id:
            issues.append(
                f"concept name {record.concept_name!r} maps to both {prior_id!r} "
                f"and {record.concept_id!r}"
            )

        prior_split = group_to_split.setdefault(record.group_id, record.split)
        if prior_split != record.split:
            issues.append(
                f"group {record.group_id!r} crosses splits {prior_split!r} and {record.split!r}"
            )

        content_key = (
            record.concept_id,
            _normalized(record.text),
            _normalized(record.prompt),
            _normalized(record.response),
        )
        prior_index = seen_content.setdefault(content_key, index)
        if prior_index != index:
            issues.append(
                f"duplicate normalized content for concept {record.concept_id!r} "
                f"at records {prior_index} and {index}"
            )

    expected_set = set(expected_splits)
    for concept_id in sorted(counts):
        concept_total = sum(
            count
            for split_counts_for_concept in counts[concept_id].values()
            for count in split_counts_for_concept.values()
        )
        if concept_total < min_per_concept:
            issues.append(
                f"concept {concept_id!r} has {concept_total} examples; "
                f"requires at least {min_per_concept}"
            )
        observed_splits = set(counts[concept_id])
        unexpected = sorted(observed_splits - expected_set)
        if unexpected:
            issues.append(f"concept {concept_id!r} has unexpected splits: {unexpected}")
        for split in expected_splits:
            label_counts = counts[concept_id][split]
            for label in (0, 1):
                count = label_counts[label]
                if count < min_per_label_per_split:
                    issues.append(
                        f"concept {concept_id!r} split {split!r} label {label} has {count} "
                        f"examples; requires at least {min_per_label_per_split}"
                    )

    if issues:
        # Preserve first occurrence while avoiding repeated metadata/leakage noise.
        raise DatasetValidationError(tuple(dict.fromkeys(issues)))

    split_counts = Counter(record.split for record in records)
    label_counts = Counter(record.label for record in records)
    concept_report: dict[str, Mapping[str, object]] = {}
    for concept_id in sorted(counts):
        concept_report[concept_id] = {
            "concept_name": concept_metadata[concept_id][0],
            "total": sum(
                count
                for split_counts_for_concept in counts[concept_id].values()
                for count in split_counts_for_concept.values()
            ),
            "splits": {
                split: {label: counts[concept_id][split][label] for label in (0, 1)}
                for split in expected_splits
            },
        }

    return DatasetStatistics(
        total_examples=len(records),
        total_concepts=len(counts),
        total_groups=len(group_to_split),
        counts_by_split={split: split_counts[split] for split in expected_splits},
        counts_by_label={label: label_counts[label] for label in (0, 1)},
        counts_by_concept=concept_report,
    )
