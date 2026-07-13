"""Optional adapter for ``pyvene/axbench-concept16k``.

The upstream negative rows are a generic pool (``concept_id == -1`` and
``output_concept == "EEEEE"``), so this adapter assigns disjoint negatives to
each selected positive concept.  Remote concept IDs are always discovered from
positive rows; they are never guessed from a local name.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prepare import DEFAULT_SPLIT_RATIOS, PreparedDataset, prepare_examples
from .schema import ConceptExample

AXBENCH_DATASET_ID = "pyvene/axbench-concept16k"
AXBENCH_REQUIRED_FIELDS = frozenset(
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


class AxBenchPreparationError(ValueError):
    """Raised before emitting data when upstream identity/schema is uncertain."""


@dataclass(frozen=True, slots=True)
class AxBenchConcept:
    concept_name: str
    definition: str
    search_terms: tuple[str, ...]
    remote_output_concept: str | None = None
    verified_against_revision: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> AxBenchConcept:
        expected = {
            "concept_name",
            "definition",
            "search_terms",
            "remote_output_concept",
            "verified_against_revision",
        }
        if set(raw) != expected:
            raise AxBenchPreparationError(
                f"AxBench concept keys must be exactly {sorted(expected)}, got {sorted(raw)}"
            )
        search_terms = raw["search_terms"]
        if (
            not isinstance(search_terms, list)
            or not search_terms
            or any(not isinstance(term, str) or not term.strip() for term in search_terms)
        ):
            raise AxBenchPreparationError("search_terms must be a non-empty list of strings")
        for field_name in ("concept_name", "definition"):
            if not isinstance(raw[field_name], str) or not raw[field_name].strip():
                raise AxBenchPreparationError(f"{field_name} must be a non-empty string")
        for field_name in ("remote_output_concept", "verified_against_revision"):
            value = raw[field_name]
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise AxBenchPreparationError(f"{field_name} must be null or a non-empty string")
        return cls(
            concept_name=raw["concept_name"],
            definition=raw["definition"],
            search_terms=tuple(search_terms),
            remote_output_concept=raw["remote_output_concept"],
            verified_against_revision=raw["verified_against_revision"],
        )


@dataclass(frozen=True, slots=True)
class AxBenchAllowlist:
    dataset_id: str
    revision: str
    variant: str
    license: str
    concepts: tuple[AxBenchConcept, ...]

    def ensure_verified(self) -> None:
        """Require exact names verified against the pinned remote revision."""

        unverified = [
            concept.concept_name
            for concept in self.concepts
            if not concept.remote_output_concept
            or concept.verified_against_revision != self.revision
        ]
        if unverified:
            raise AxBenchPreparationError(
                "AxBench allowlist is not verified for the pinned revision "
                f"{self.revision}: {', '.join(unverified)}. Run concept discovery, inspect the "
                "returned positive rows, then set remote_output_concept and "
                "verified_against_revision. No remote concept IDs will be inferred from names."
            )


def load_axbench_allowlist(path: str | Path) -> AxBenchAllowlist:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AxBenchPreparationError(f"cannot read AxBench allowlist {source}: {error}") from error
    if not isinstance(raw, dict):
        raise AxBenchPreparationError("AxBench allowlist must be a JSON object")
    expected = {
        "schema_version",
        "dataset_id",
        "revision",
        "variant",
        "license",
        "concepts",
    }
    if set(raw) != expected:
        raise AxBenchPreparationError(
            f"AxBench allowlist keys must be exactly {sorted(expected)}, got {sorted(raw)}"
        )
    if raw["schema_version"] != 1:
        raise AxBenchPreparationError("unsupported AxBench allowlist schema_version")
    if raw["dataset_id"] != AXBENCH_DATASET_ID:
        raise AxBenchPreparationError(
            f"dataset_id must be the supported upstream {AXBENCH_DATASET_ID!r}"
        )
    for field_name in ("revision", "variant", "license"):
        if not isinstance(raw[field_name], str) or not raw[field_name].strip():
            raise AxBenchPreparationError(f"{field_name} must be a non-empty string")
    if raw["variant"] not in {"2b/l20", "9b/l20"}:
        raise AxBenchPreparationError("variant must be '2b/l20' or '9b/l20'")
    raw_concepts = raw["concepts"]
    if not isinstance(raw_concepts, list) or not raw_concepts:
        raise AxBenchPreparationError("concepts must be a non-empty list")
    concepts = tuple(AxBenchConcept.from_dict(item) for item in raw_concepts)
    names = [concept.concept_name.casefold() for concept in concepts]
    if len(names) != len(set(names)):
        raise AxBenchPreparationError("concept_name values must be unique")
    return AxBenchAllowlist(
        dataset_id=raw["dataset_id"],
        revision=raw["revision"],
        variant=raw["variant"],
        license=raw["license"],
        concepts=concepts,
    )


def load_axbench_source(
    allowlist: AxBenchAllowlist,
    *,
    streaming: bool = True,
    split: str = "train",
) -> Iterable[Mapping[str, Any]]:
    """Load one model/layer parquet via the optional Hugging Face dependency.

    This low-level read is also available for concept discovery. The mutating
    ``prepare_axbench`` path separately requires a verified allowlist before it
    calls this function.
    """

    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise AxBenchPreparationError(
            "AxBench loading requires the optional 'datasets' package; install the llm extra"
        ) from error
    data_file = f"{allowlist.variant}/{split}/data.parquet"
    try:
        return load_dataset(
            allowlist.dataset_id,
            data_files={split: data_file},
            split=split,
            revision=allowlist.revision,
            streaming=streaming,
        )
    except Exception as error:  # pragma: no cover - remote library exception types vary
        raise AxBenchPreparationError(
            f"failed to load {allowlist.dataset_id}@{allowlist.revision} file {data_file}: {error}"
        ) from error


def _checked_row(raw: Mapping[str, Any], row_index: int) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise AxBenchPreparationError(
            f"AxBench row {row_index} is not a mapping: {type(raw).__name__}"
        )
    missing = sorted(AXBENCH_REQUIRED_FIELDS - set(raw))
    if missing:
        raise AxBenchPreparationError(f"AxBench row {row_index} lacks fields: {missing}")
    return raw


def discover_axbench_concepts(
    rows: Iterable[Mapping[str, Any]],
    search_terms: Sequence[str],
    *,
    max_rows: int | None = None,
) -> dict[str, list[dict[str, str | int]]]:
    """Find positive remote names/IDs containing terms without asserting identity."""

    if not search_terms:
        raise ValueError("search_terms must not be empty")
    discoveries: dict[str, dict[tuple[int, str], None]] = {
        term: {} for term in search_terms
    }
    for row_index, raw in enumerate(rows):
        if max_rows is not None and row_index >= max_rows:
            break
        row = _checked_row(raw, row_index)
        if row["category"] != "positive":
            continue
        remote_name = row["output_concept"]
        remote_id = row["concept_id"]
        if not isinstance(remote_name, str) or not isinstance(remote_id, int) or remote_id < 0:
            raise AxBenchPreparationError(
                f"positive AxBench row {row_index} has invalid concept name/ID"
            )
        folded = remote_name.casefold()
        for term in search_terms:
            if term.casefold() in folded:
                discoveries[term][(remote_id, remote_name)] = None
    return {
        term: [
            {"concept_id": concept_id, "output_concept": output_concept}
            for concept_id, output_concept in sorted(matches)
        ]
        for term, matches in discoveries.items()
    }


def _row_group_id(
    variant: str, target_id: int, label: int, row_index: int, prompt: str, response: str
) -> str:
    digest = hashlib.sha256(
        f"{variant}\0{target_id}\0{label}\0{row_index}\0{prompt}\0{response}".encode()
    ).hexdigest()[:20]
    return f"axbench:{variant.replace('/', '-')}:{target_id}:{label}:{digest}"


def adapt_axbench_rows(
    rows: Iterable[Mapping[str, Any]],
    allowlist: AxBenchAllowlist,
    *,
    per_label: int = 8,
    max_rows: int | None = None,
) -> list[ConceptExample]:
    """Convert streamed or materialized AxBench rows to strict binary records."""

    allowlist.ensure_verified()
    if per_label < 3:
        raise ValueError("per_label must be at least 3 to cover train/validation/test")

    by_remote_name = {
        concept.remote_output_concept.casefold(): concept
        for concept in allowlist.concepts
        if concept.remote_output_concept is not None
    }
    positives: dict[str, list[tuple[int, Mapping[str, Any]]]] = {
        concept.concept_name: [] for concept in allowlist.concepts
    }
    discovered_ids: dict[str, int] = {}
    negatives: list[tuple[int, Mapping[str, Any]]] = []
    required_negatives = per_label * len(allowlist.concepts)

    for row_index, raw in enumerate(rows):
        if max_rows is not None and row_index >= max_rows:
            break
        row = _checked_row(raw, row_index)
        category = row["category"]
        prompt = row["input"]
        response = row["output"]
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        if not isinstance(response, str) or not response.strip():
            continue

        if category == "negative":
            if row["output_concept"] != "EEEEE" or row["concept_id"] != -1:
                raise AxBenchPreparationError(
                    f"negative AxBench row {row_index} violates EEEEE/-1 sentinel schema"
                )
            if len(negatives) < required_negatives:
                negatives.append((row_index, row))
        elif category == "positive":
            remote_name = row["output_concept"]
            remote_id = row["concept_id"]
            if not isinstance(remote_name, str) or not isinstance(remote_id, int) or remote_id < 0:
                raise AxBenchPreparationError(
                    f"positive AxBench row {row_index} has invalid concept name/ID"
                )
            concept = by_remote_name.get(remote_name.casefold())
            if concept is not None:
                prior_id = discovered_ids.setdefault(concept.concept_name, remote_id)
                if prior_id != remote_id:
                    raise AxBenchPreparationError(
                        f"remote concept {remote_name!r} maps to IDs {prior_id} and {remote_id}"
                    )
                if len(positives[concept.concept_name]) < per_label:
                    positives[concept.concept_name].append((row_index, row))
        else:
            raise AxBenchPreparationError(
                f"AxBench row {row_index} has unexpected category {category!r}"
            )

        if len(negatives) >= required_negatives and all(
            len(items) >= per_label for items in positives.values()
        ):
            break

    missing = [
        f"{concept.concept_name} ({len(positives[concept.concept_name])}/{per_label})"
        for concept in allowlist.concepts
        if len(positives[concept.concept_name]) < per_label
    ]
    if missing:
        raise AxBenchPreparationError(
            "verified allowlist concepts were absent or under-sampled in the scanned source: "
            + ", ".join(missing)
            + ". The adapter will not invent concept IDs."
        )
    if len(negatives) < required_negatives:
        raise AxBenchPreparationError(
            f"source yielded only {len(negatives)} usable generic negatives; "
            f"{required_negatives} are required"
        )

    source = (
        f"hf://datasets/{allowlist.dataset_id}@{allowlist.revision}/"
        f"{allowlist.variant}/train/data.parquet"
    )
    output: list[ConceptExample] = []
    negative_cursor = 0
    for concept in allowlist.concepts:
        target_id = discovered_ids[concept.concept_name]
        local_id = f"axbench:{allowlist.variant.replace('/', '-')}:{target_id}"
        for label, selected_rows in (
            (1, positives[concept.concept_name]),
            (0, negatives[negative_cursor : negative_cursor + per_label]),
        ):
            for row_index, row in selected_rows:
                prompt = row["input"]
                response = row["output"]
                output.append(
                    ConceptExample(
                        concept_id=local_id,
                        concept_name=concept.concept_name,
                        definition=concept.definition,
                        abstractness="abstract",
                        label=label,
                        text=response,
                        prompt=prompt,
                        response=response,
                        split="train",
                        group_id=_row_group_id(
                            allowlist.variant,
                            target_id,
                            label,
                            row_index,
                            prompt,
                            response,
                        ),
                        source=source,
                        license=allowlist.license,
                    )
                )
        negative_cursor += per_label
    return output


def prepare_axbench(
    allowlist_path: str | Path,
    output_dir: str | Path,
    *,
    streaming: bool = True,
    per_label: int = 8,
    max_rows: int | None = None,
    seed: int = 42,
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
) -> PreparedDataset:
    """Load, strictly adapt, deterministically split, and report AxBench data."""

    allowlist = load_axbench_allowlist(allowlist_path)
    # This happens before importing datasets or making a network request.
    allowlist.ensure_verified()
    rows = load_axbench_source(allowlist, streaming=streaming)
    examples = adapt_axbench_rows(rows, allowlist, per_label=per_label, max_rows=max_rows)
    return prepare_examples(
        examples,
        output_dir=output_dir,
        seed=seed,
        ratios=ratios,
        resplit=True,
    )
