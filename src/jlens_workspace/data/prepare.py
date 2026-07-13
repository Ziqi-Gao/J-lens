"""Deterministic, group-safe splitting and preparation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from math import floor, isfinite
from pathlib import Path

from .io import load_jsonl, write_jsonl, write_statistics
from .schema import CANONICAL_SPLITS, ConceptExample
from .validation import DatasetStatistics, DatasetValidationError, validate_examples

DEFAULT_SPLIT_RATIOS: Mapping[str, float] = {
    "train": 0.5,
    "validation": 0.25,
    "test": 0.25,
}


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    examples: tuple[ConceptExample, ...]
    statistics: DatasetStatistics
    fingerprint: str
    seed: int
    split_ratios: Mapping[str, float]


def _normalized_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    unknown = sorted(set(ratios) - set(CANONICAL_SPLITS))
    if unknown:
        raise ValueError(f"unknown split names in ratios: {unknown}")
    values = {split: float(ratios.get(split, 0.0)) for split in CANONICAL_SPLITS}
    if any(not isfinite(value) for value in values.values()):
        raise ValueError("split ratios must be finite")
    if any(value < 0 for value in values.values()):
        raise ValueError("split ratios cannot be negative")
    total = sum(values.values())
    if total <= 0:
        raise ValueError("at least one split ratio must be positive")
    return {split: value / total for split, value in values.items() if value > 0}


def _allocation(group_count: int, ratios: Mapping[str, float]) -> dict[str, int]:
    splits = tuple(ratios)
    if group_count < len(splits):
        raise DatasetValidationError(
            (
                f"cannot cover {len(splits)} splits with only {group_count} groups; "
                "add independent groups or set a zero ratio for unused splits",
            )
        )
    ideal = {split: group_count * ratios[split] for split in splits}
    allocation = {split: max(1, floor(ideal[split])) for split in splits}
    while sum(allocation.values()) > group_count:
        candidates = [split for split in splits if allocation[split] > 1]
        split = max(
            candidates,
            key=lambda candidate: (
                allocation[candidate] - ideal[candidate],
                allocation[candidate],
                candidate,
            ),
        )
        allocation[split] -= 1
    while sum(allocation.values()) < group_count:
        split = max(
            splits,
            key=lambda candidate: (
                ideal[candidate] - allocation[candidate],
                ratios[candidate],
                candidate,
            ),
        )
        allocation[split] += 1
    return allocation


def _stable_group_key(
    seed: int, concept_id: str, label_counts: tuple[int, int], group_id: str
) -> bytes:
    value = f"{seed}\0{concept_id}\0{label_counts[0]}:{label_counts[1]}\0{group_id}".encode()
    return hashlib.sha256(value).digest()


def deterministic_group_split(
    examples: Iterable[ConceptExample],
    *,
    seed: int = 42,
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
) -> list[ConceptExample]:
    """Stratify by concept and group label composition while keeping groups atomic.

    A group must belong to one concept, but may deliberately contain paired
    positive/negative variants of the same source prompt. Groups with the same
    ``(#negative, #positive)`` composition form a stratum, preventing paired
    prompt leakage while preserving split coverage.
    """

    records = list(examples)
    normalized_ratios = _normalized_ratios(ratios)
    groups: dict[str, list[tuple[int, ConceptExample]]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record.group_id].append((index, record))

    strata: dict[tuple[str, tuple[int, int]], list[str]] = defaultdict(list)
    for group_id, members in groups.items():
        concept_ids = {record.concept_id for _, record in members}
        if len(concept_ids) != 1:
            raise DatasetValidationError(
                (
                    f"group {group_id!r} mixes concepts {sorted(concept_ids)}; "
                    "a source group must belong to one concept",
                )
            )
        label_counts = (
            sum(record.label == 0 for _, record in members),
            sum(record.label == 1 for _, record in members),
        )
        strata[(next(iter(concept_ids)), label_counts)].append(group_id)

    assignment: dict[str, str] = {}
    for (concept_id, label_counts), group_ids in sorted(strata.items()):
        group_ids.sort(
            key=lambda value: _stable_group_key(seed, concept_id, label_counts, value)
        )
        allocation = _allocation(len(group_ids), normalized_ratios)
        cursor = 0
        for split in normalized_ratios:
            next_cursor = cursor + allocation[split]
            for group_id in group_ids[cursor:next_cursor]:
                assignment[group_id] = split
            cursor = next_cursor

    return [replace(record, split=assignment[record.group_id]) for record in records]


def dataset_fingerprint(examples: Iterable[ConceptExample]) -> str:
    canonical_lines = [
        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in examples
    ]
    digest = hashlib.sha256()
    for line in sorted(canonical_lines):
        digest.update(line.encode())
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def prepare_examples(
    examples: Iterable[ConceptExample],
    *,
    output_dir: str | Path | None = None,
    seed: int = 42,
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
    resplit: bool = True,
    min_per_label_per_split: int = 1,
    min_per_concept: int = 1,
) -> PreparedDataset:
    records = list(examples)
    normalized_ratios = _normalized_ratios(ratios)
    if resplit:
        records = deterministic_group_split(records, seed=seed, ratios=normalized_ratios)
    expected_splits = tuple(normalized_ratios)
    statistics = validate_examples(
        records,
        expected_splits=expected_splits,
        min_per_label_per_split=min_per_label_per_split,
        min_per_concept=min_per_concept,
    )
    fingerprint = dataset_fingerprint(records)
    prepared = PreparedDataset(
        examples=tuple(records),
        statistics=statistics,
        fingerprint=fingerprint,
        seed=seed,
        split_ratios=normalized_ratios,
    )

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for split in expected_splits:
            split_records = sorted(
                (record for record in records if record.split == split),
                key=lambda record: (record.concept_id, record.label, record.group_id),
            )
            write_jsonl(destination / f"{split}.jsonl", split_records)
        write_statistics(destination / "statistics.json", statistics)
        manifest = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "seed": seed,
            "split_ratios": normalized_ratios,
            "files": {split: f"{split}.jsonl" for split in expected_splits},
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return prepared


def prepare_jsonl(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
) -> PreparedDataset:
    records = load_jsonl(input_path, validate=False)
    return prepare_examples(records, output_dir=output_dir, seed=seed, ratios=ratios)
