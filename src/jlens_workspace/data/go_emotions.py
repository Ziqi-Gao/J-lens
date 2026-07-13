"""Pinned GoEmotions preparation for large binary abstract-concept probes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jlens_workspace.artifacts import atomic_write_json

from .prepare import PreparedDataset, prepare_examples
from .schema import CANONICAL_SPLITS, ConceptExample

GO_EMOTIONS_DATASET_ID = "google-research-datasets/go_emotions"
GO_EMOTIONS_CONFIG = "simplified"
GO_EMOTIONS_LABELS = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
)


class GoEmotionsPreparationError(ValueError):
    """Raised when pinned GoEmotions inputs violate the preparation contract."""


@dataclass(frozen=True, slots=True)
class GoEmotionsConcept:
    label_id: int
    concept_name: str
    definition: str


@dataclass(frozen=True, slots=True)
class GoEmotionsAllowlist:
    dataset_id: str
    revision: str
    config: str
    license: str
    file_sha256: Mapping[str, str]
    positive_per_split: Mapping[str, int] | None
    negative_per_split: Mapping[str, int] | None
    lens_fit_prompt_count: int
    lens_fit_min_words: int
    concepts: tuple[GoEmotionsConcept, ...]
    selection_mode: str = "balanced_single_label"
    include_multilabel: bool = False
    fit_prompts_disjoint: bool = True


@dataclass(frozen=True, slots=True)
class GoEmotionsSelection:
    examples: tuple[ConceptExample, ...]
    fit_prompts: tuple[Mapping[str, str], ...]
    excluded_cross_split_texts: int
    raw_rows_by_split: Mapping[str, int]
    retained_source_rows_by_split: Mapping[str, int]
    excluded_within_split_duplicate_rows: Mapping[str, int]
    excluded_cross_split_rows: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class PreparedGoEmotions:
    dataset: PreparedDataset
    fit_prompts_path: Path


def _required_text(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GoEmotionsPreparationError(f"{key} must be a non-empty string")
    return value


def _sha256(value: Any, key: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise GoEmotionsPreparationError(f"{key} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise GoEmotionsPreparationError(f"{key} must be hexadecimal") from error
    return value.casefold()


def _split_counts(raw: Any, key: str) -> dict[str, int]:
    if not isinstance(raw, Mapping) or set(raw) != set(CANONICAL_SPLITS):
        raise GoEmotionsPreparationError(
            f"{key} must define exactly {list(CANONICAL_SPLITS)}"
        )
    output: dict[str, int] = {}
    for split in CANONICAL_SPLITS:
        value = raw[split]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise GoEmotionsPreparationError(f"{key}.{split} must be a positive integer")
        output[split] = value
    return output


def load_go_emotions_allowlist(path: str | Path) -> GoEmotionsAllowlist:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GoEmotionsPreparationError(f"cannot read GoEmotions allowlist {source}: {error}") from error
    expected = {
        "schema_version",
        "dataset_id",
        "revision",
        "config",
        "license",
        "file_sha256",
        "selection",
        "concepts",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        keys = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        raise GoEmotionsPreparationError(
            f"GoEmotions allowlist keys must be exactly {sorted(expected)}, got {keys}"
        )
    schema_version = raw["schema_version"]
    if schema_version not in (1, 2):
        raise GoEmotionsPreparationError("unsupported GoEmotions allowlist schema_version")
    if raw["dataset_id"] != GO_EMOTIONS_DATASET_ID:
        raise GoEmotionsPreparationError(f"dataset_id must be {GO_EMOTIONS_DATASET_ID!r}")
    if raw["config"] != GO_EMOTIONS_CONFIG:
        raise GoEmotionsPreparationError(f"config must be {GO_EMOTIONS_CONFIG!r}")
    revision = _required_text(raw, "revision")
    license_name = _required_text(raw, "license")

    file_hashes = raw["file_sha256"]
    if not isinstance(file_hashes, Mapping) or set(file_hashes) != set(CANONICAL_SPLITS):
        raise GoEmotionsPreparationError(
            f"file_sha256 must define exactly {list(CANONICAL_SPLITS)}"
        )
    checked_hashes = {
        split: _sha256(file_hashes[split], f"file_sha256.{split}")
        for split in CANONICAL_SPLITS
    }

    selection = raw["selection"]
    selection_keys = (
        {
            "single_label_only",
            "positive_per_split",
            "negative_per_split",
            "lens_fit_prompt_count",
            "lens_fit_min_words",
        }
        if schema_version == 1
        else {
            "mode",
            "include_multilabel",
            "lens_fit_prompt_count",
            "lens_fit_min_words",
        }
    )
    if not isinstance(selection, Mapping) or set(selection) != selection_keys:
        raise GoEmotionsPreparationError(
            f"selection keys must be exactly {sorted(selection_keys)}"
        )
    if schema_version == 1:
        if selection["single_label_only"] is not True:
            raise GoEmotionsPreparationError(
                "balanced GoEmotions probes require single_label_only=true"
            )
        positive: Mapping[str, int] | None = _split_counts(
            selection["positive_per_split"], "positive_per_split"
        )
        negative: Mapping[str, int] | None = _split_counts(
            selection["negative_per_split"], "negative_per_split"
        )
        selection_mode = "balanced_single_label"
        include_multilabel = False
        fit_prompts_disjoint = True
    else:
        if selection["mode"] != "all_one_vs_rest":
            raise GoEmotionsPreparationError(
                "schema_version 2 selection.mode must be 'all_one_vs_rest'"
            )
        if selection["include_multilabel"] is not True:
            raise GoEmotionsPreparationError(
                "all_one_vs_rest requires include_multilabel=true"
            )
        positive = None
        negative = None
        selection_mode = "all_one_vs_rest"
        include_multilabel = True
        fit_prompts_disjoint = False
    for key in ("lens_fit_prompt_count", "lens_fit_min_words"):
        value = selection[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise GoEmotionsPreparationError(f"selection.{key} must be a positive integer")

    raw_concepts = raw["concepts"]
    if not isinstance(raw_concepts, list) or not raw_concepts:
        raise GoEmotionsPreparationError("concepts must be a non-empty list")
    concepts: list[GoEmotionsConcept] = []
    for item in raw_concepts:
        if not isinstance(item, Mapping) or set(item) != {
            "label_id",
            "concept_name",
            "definition",
        }:
            raise GoEmotionsPreparationError(
                "each concept requires exactly label_id, concept_name, and definition"
            )
        label_id = item["label_id"]
        if (
            isinstance(label_id, bool)
            or not isinstance(label_id, int)
            or not 0 <= label_id < len(GO_EMOTIONS_LABELS)
        ):
            raise GoEmotionsPreparationError("concept label_id is out of range")
        concept_name = _required_text(item, "concept_name")
        definition = _required_text(item, "definition")
        if concept_name.casefold() != GO_EMOTIONS_LABELS[label_id]:
            raise GoEmotionsPreparationError(
                f"label {label_id} is {GO_EMOTIONS_LABELS[label_id]!r}, not {concept_name!r}"
            )
        concepts.append(GoEmotionsConcept(label_id, concept_name, definition))
    if len({item.label_id for item in concepts}) != len(concepts):
        raise GoEmotionsPreparationError("concept label_id values must be unique")
    if positive is not None and negative is not None:
        per_concept = sum(positive.values()) + sum(negative.values())
        if per_concept < 1000:
            raise GoEmotionsPreparationError(
                f"each concept would contain only {per_concept} examples; at least 1000 are required"
            )
    return GoEmotionsAllowlist(
        dataset_id=raw["dataset_id"],
        revision=revision,
        config=raw["config"],
        license=license_name,
        file_sha256=checked_hashes,
        positive_per_split=positive,
        negative_per_split=negative,
        lens_fit_prompt_count=selection["lens_fit_prompt_count"],
        lens_fit_min_words=selection["lens_fit_min_words"],
        concepts=tuple(concepts),
        selection_mode=selection_mode,
        include_multilabel=include_multilabel,
        fit_prompts_disjoint=fit_prompts_disjoint,
    )


_WHITESPACE = re.compile(r"\s+")


def _normalized_text(value: str) -> str:
    return _WHITESPACE.sub(" ", value.strip()).casefold()


def _stable_digest(*values: object) -> bytes:
    return hashlib.sha256("\0".join(str(value) for value in values).encode()).digest()


def _checked_rows(
    rows_by_split: Mapping[str, Iterable[Mapping[str, Any]]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    int,
    dict[str, int],
    dict[str, int],
    dict[str, int],
]:
    if set(rows_by_split) != set(CANONICAL_SPLITS):
        raise GoEmotionsPreparationError(
            f"rows_by_split must define exactly {list(CANONICAL_SPLITS)}"
        )
    occurrences: dict[str, set[str]] = defaultdict(set)
    checked: dict[str, list[dict[str, Any]]] = {}
    raw_counts: dict[str, int] = {}
    within_split_duplicates: dict[str, int] = {}
    seen_ids: set[str] = set()
    for split in CANONICAL_SPLITS:
        by_text: dict[str, dict[str, Any]] = {}
        raw_count = 0
        for row_number, raw in enumerate(rows_by_split[split]):
            raw_count += 1
            if not isinstance(raw, Mapping):
                raise GoEmotionsPreparationError(f"{split} row {row_number} is not a mapping")
            text = raw.get("text")
            source_id = raw.get("id")
            labels = raw.get("labels")
            if not isinstance(text, str) or not text.strip():
                raise GoEmotionsPreparationError(f"{split} row {row_number} has invalid text")
            if not isinstance(source_id, str) or not source_id.strip():
                raise GoEmotionsPreparationError(f"{split} row {row_number} has invalid id")
            if source_id in seen_ids:
                raise GoEmotionsPreparationError(f"source id {source_id!r} occurs more than once")
            seen_ids.add(source_id)
            if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
                raise GoEmotionsPreparationError(f"{split} row {row_number} has invalid labels")
            parsed_labels = tuple(int(label) for label in labels)
            if not parsed_labels or any(
                label < 0 or label >= len(GO_EMOTIONS_LABELS) for label in parsed_labels
            ):
                raise GoEmotionsPreparationError(f"{split} row {row_number} has invalid label ids")
            normalized = _normalized_text(text)
            candidate = {
                "text": text.strip(),
                "id": source_id,
                "labels": parsed_labels,
                "normalized": normalized,
            }
            prior = by_text.get(normalized)
            if prior is None or source_id < prior["id"]:
                by_text[normalized] = candidate
            occurrences[normalized].add(split)
        checked[split] = list(by_text.values())
        raw_counts[split] = raw_count
        within_split_duplicates[split] = raw_counts[split] - len(by_text)
    cross_split = {
        normalized for normalized, splits in occurrences.items() if len(splits) > 1
    }
    retained = {
        split: [row for row in checked[split] if row["normalized"] not in cross_split]
        for split in CANONICAL_SPLITS
    }
    cross_split_rows = {
        split: len(checked[split]) - len(retained[split]) for split in CANONICAL_SPLITS
    }
    return (
        retained,
        len(cross_split),
        raw_counts,
        within_split_duplicates,
        cross_split_rows,
    )


def _take(
    candidates: Iterable[dict[str, Any]],
    count: int,
    *,
    seed: int,
    concept: str,
    split: str,
    label: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda row: _stable_digest(
            seed, concept, split, label, row["id"], row["normalized"]
        ),
    )
    if len(ranked) < count:
        raise GoEmotionsPreparationError(
            f"concept {concept!r} split {split!r} label {label} has "
            f"{len(ranked)} eligible rows; requires {count}"
        )
    return ranked[:count]


def select_go_emotions(
    rows_by_split: Mapping[str, Iterable[Mapping[str, Any]]],
    allowlist: GoEmotionsAllowlist,
    *,
    seed: int = 42,
) -> GoEmotionsSelection:
    """Build pinned binary tasks in either balanced-control or full OVR mode."""

    (
        checked,
        excluded_cross_split,
        raw_rows_by_split,
        excluded_within_split_duplicate_rows,
        excluded_cross_split_rows,
    ) = _checked_rows(rows_by_split)
    examples: list[ConceptExample] = []
    probe_ids: set[str] = set()

    def append_example(
        concept: GoEmotionsConcept, row: Mapping[str, Any], split: str, label: int
    ) -> None:
        probe_ids.add(str(row["id"]))
        examples.append(
            ConceptExample(
                concept_id=f"goemotions:{concept.concept_name.casefold()}",
                concept_name=concept.concept_name,
                definition=concept.definition,
                abstractness="abstract",
                label=label,
                text=str(row["text"]),
                prompt="",
                response="",
                split=split,
                group_id=f"goemotions:{row['id']}",
                source=(
                    f"hf://datasets/{allowlist.dataset_id}@{allowlist.revision}/"
                    f"{allowlist.config}/{split}"
                ),
                license=allowlist.license,
            )
        )

    if allowlist.selection_mode == "all_one_vs_rest":
        for split in CANONICAL_SPLITS:
            for row in checked[split]:
                row_labels = set(row["labels"])
                for concept in allowlist.concepts:
                    append_example(
                        concept, row, split, int(concept.label_id in row_labels)
                    )
    else:
        if allowlist.positive_per_split is None or allowlist.negative_per_split is None:
            raise GoEmotionsPreparationError(
                "balanced selection requires explicit positive/negative counts"
            )
        for concept in allowlist.concepts:
            for split in CANONICAL_SPLITS:
                eligible = [row for row in checked[split] if len(row["labels"]) == 1]
                positives = _take(
                    (row for row in eligible if row["labels"][0] == concept.label_id),
                    allowlist.positive_per_split[split],
                    seed=seed,
                    concept=concept.concept_name,
                    split=split,
                    label=1,
                )
                negatives = _take(
                    (row for row in eligible if row["labels"][0] != concept.label_id),
                    allowlist.negative_per_split[split],
                    seed=seed,
                    concept=concept.concept_name,
                    split=split,
                    label=0,
                )
                for label, rows in ((1, positives), (0, negatives)):
                    for row in rows:
                        append_example(concept, row, split, label)
    fit_candidates = [
        row
        for row in checked["train"]
        if (not allowlist.fit_prompts_disjoint or row["id"] not in probe_ids)
        and len(row["text"].split()) >= allowlist.lens_fit_min_words
    ]
    fit_candidates.sort(
        key=lambda row: _stable_digest(seed, "jlens-fit", row["id"], row["normalized"])
    )
    if len(fit_candidates) < allowlist.lens_fit_prompt_count:
        raise GoEmotionsPreparationError(
            f"only {len(fit_candidates)} lens-fit prompts satisfy the "
            f"minimum length; requires {allowlist.lens_fit_prompt_count}"
        )
    fit_rows = fit_candidates[: allowlist.lens_fit_prompt_count]
    prompts = tuple({"source_id": row["id"], "text": row["text"]} for row in fit_rows)
    return GoEmotionsSelection(
        tuple(examples),
        prompts,
        excluded_cross_split,
        raw_rows_by_split,
        {split: len(checked[split]) for split in CANONICAL_SPLITS},
        excluded_within_split_duplicate_rows,
        excluded_cross_split_rows,
    )


def _write_prompt_jsonl(path: Path, prompts: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            for prompt in prompts:
                handle.write(json.dumps(dict(prompt), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def prepare_go_emotions(
    rows_by_split: Mapping[str, Iterable[Mapping[str, Any]]],
    allowlist: GoEmotionsAllowlist,
    output_dir: str | Path,
    *,
    seed: int = 42,
) -> PreparedGoEmotions:
    selection = select_go_emotions(rows_by_split, allowlist, seed=seed)
    if allowlist.positive_per_split is None or allowlist.negative_per_split is None:
        minimum = 1
        minimum_per_concept = len(selection.examples) // len(allowlist.concepts)
    else:
        minimum = min(
            *allowlist.positive_per_split.values(),
            *allowlist.negative_per_split.values(),
        )
        minimum_per_concept = sum(allowlist.positive_per_split.values()) + sum(
            allowlist.negative_per_split.values()
        )
    prepared = prepare_examples(
        selection.examples,
        output_dir=output_dir,
        seed=seed,
        ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
        resplit=False,
        min_per_label_per_split=minimum,
        min_per_concept=minimum_per_concept,
    )
    destination = Path(output_dir)
    prompts_path = destination / "fit_prompts.jsonl"
    _write_prompt_jsonl(prompts_path, selection.fit_prompts)
    atomic_write_json(
        destination / "source_manifest.json",
        {
            "schema_version": (
                2 if allowlist.selection_mode == "all_one_vs_rest" else 1
            ),
            "dataset_id": allowlist.dataset_id,
            "revision": allowlist.revision,
            "config": allowlist.config,
            "license": allowlist.license,
            "seed": seed,
            "file_sha256": dict(allowlist.file_sha256),
            "selection_mode": allowlist.selection_mode,
            "include_multilabel": allowlist.include_multilabel,
            "concepts": [
                {
                    "label_id": concept.label_id,
                    "concept_name": concept.concept_name,
                    "definition": concept.definition,
                }
                for concept in allowlist.concepts
            ],
            "positive_per_split": (
                None
                if allowlist.positive_per_split is None
                else dict(allowlist.positive_per_split)
            ),
            "negative_per_split": (
                None
                if allowlist.negative_per_split is None
                else dict(allowlist.negative_per_split)
            ),
            "dataset_fingerprint": prepared.fingerprint,
            "statistics": prepared.statistics.to_dict(),
            "excluded_cross_split_normalized_texts": (
                selection.excluded_cross_split_texts
            ),
            "source_row_accounting": {
                "raw_rows_by_split": dict(selection.raw_rows_by_split),
                "retained_rows_by_split": dict(
                    selection.retained_source_rows_by_split
                ),
                "excluded_within_split_duplicate_rows": dict(
                    selection.excluded_within_split_duplicate_rows
                ),
                "excluded_cross_split_rows": dict(
                    selection.excluded_cross_split_rows
                ),
            },
            "fit_prompts": {
                "file": prompts_path.name,
                "count": len(selection.fit_prompts),
                "minimum_words": allowlist.lens_fit_min_words,
                "disjoint_from_probe_examples": allowlist.fit_prompts_disjoint,
                "may_overlap_probe_examples": not allowlist.fit_prompts_disjoint,
            },
        },
    )
    return PreparedGoEmotions(prepared, prompts_path)
