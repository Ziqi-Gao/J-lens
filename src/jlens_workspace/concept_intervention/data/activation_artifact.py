"""Neutral readers for immutable Concept Intervention activation artifacts.

This module validates versioned ``resid_post`` activation artifacts and exposes
split, label, and group views without depending on probe fitting or steering.
Activation arrays have shape ``[n_examples, d_residual]``; labels have shape
``[n_examples]`` for schema v1 and ``[n_examples, n_concepts]`` for schema v2.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from jlens_workspace.data import CANONICAL_SPLITS
from jlens_workspace.foundation.activations import load_activation_layer
from jlens_workspace.foundation.artifacts import sha256_file, stable_hash

_LAYER_FILE = re.compile(r"layer_(\d+)\.npy")

_REQUIRED_EXPANDED_ROW_FIELDS = frozenset(
    {"row", "concept_id", "concept_name", "label", "split", "group_id"}
)

_REQUIRED_SHARED_ROW_FIELDS = frozenset({"row", "split", "group_id", "source", "text_sha256"})


class ConceptWorkflowError(ValueError):
    """Raised when an activation artifact violates the workflow contract."""


@dataclass(frozen=True, slots=True)
class ActivationArtifact:
    path: Path
    metadata: Mapping[str, Any]
    labels: NDArray[np.integer[Any]]
    rows: tuple[Mapping[str, Any], ...]
    layers: tuple[int, ...]
    artifact_hash: str
    concept_names: Mapping[str, str]
    concept_columns: Mapping[str, int] | None


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConceptWorkflowError(f"missing activation artifact file: {path.name}") from error
    except json.JSONDecodeError as error:
        raise ConceptWorkflowError(f"invalid JSON in {path}: {error.msg}") from error
    if not isinstance(raw, dict):
        raise ConceptWorkflowError(f"{path.name} must contain one JSON object")
    return raw


def _discover_layers(path: Path) -> dict[int, Path]:
    discovered: dict[int, Path] = {}
    candidates = sorted(path.glob("layer_*.npy"))
    if not candidates:
        raise ConceptWorkflowError(f"no layer_XX.npy files found in {path}")
    for candidate in candidates:
        match = _LAYER_FILE.fullmatch(candidate.name)
        if match is None:
            raise ConceptWorkflowError(f"invalid layer filename: {candidate.name}")
        layer = int(match.group(1))
        canonical_name = f"layer_{layer:02d}.npy"
        if candidate.name != canonical_name:
            raise ConceptWorkflowError(
                f"layer filename must use capture format {canonical_name}, got {candidate.name}"
            )
        if layer in discovered:
            raise ConceptWorkflowError(
                f"multiple files encode layer {layer}: "
                f"{discovered[layer].name} and {candidate.name}"
            )
        discovered[layer] = candidate
    return discovered


def _metadata_layers(metadata: Mapping[str, Any]) -> tuple[int, ...]:
    raw_layers = metadata.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ConceptWorkflowError("metadata.layers must be a non-empty integer list")
    layers: list[int] = []
    for raw_layer in raw_layers:
        if isinstance(raw_layer, bool) or not isinstance(raw_layer, int) or raw_layer < 0:
            raise ConceptWorkflowError("metadata.layers must contain non-negative integers")
        layers.append(raw_layer)
    if len(set(layers)) != len(layers):
        raise ConceptWorkflowError("metadata.layers contains duplicate layers")
    return tuple(sorted(layers))


def _load_labels(path: Path, *, schema_version: int) -> NDArray[np.integer[Any]]:
    try:
        labels = np.load(path, allow_pickle=False)
    except FileNotFoundError as error:
        raise ConceptWorkflowError("missing activation artifact file: labels.npy") from error
    except (OSError, ValueError) as error:
        raise ConceptWorkflowError(f"cannot load labels.npy: {error}") from error
    expected_ndim = 1 if schema_version == 1 else 2
    if labels.ndim != expected_ndim or labels.size == 0:
        shape_description = "[n_examples]" if schema_version == 1 else "[n_examples, n_concepts]"
        raise ConceptWorkflowError(
            f"labels.npy must have non-empty shape {shape_description}, got {labels.shape}"
        )
    if not np.issubdtype(labels.dtype, np.integer) or np.issubdtype(labels.dtype, np.bool_):
        raise ConceptWorkflowError("labels.npy must contain integer binary labels")
    permitted = (0, 1) if schema_version == 1 else (-1, 0, 1)
    if not np.isin(labels, permitted).all():
        raise ConceptWorkflowError(f"labels.npy contains values outside {set(permitted)}")
    return labels


def _read_rows(path: Path, *, schema_version: int) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ConceptWorkflowError("missing activation artifact file: rows.jsonl") from error
    if not lines:
        raise ConceptWorkflowError("rows.jsonl is empty")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ConceptWorkflowError(f"rows.jsonl:{line_number}: blank lines are not allowed")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ConceptWorkflowError(
                f"rows.jsonl:{line_number}: invalid JSON: {error.msg}"
            ) from error
        if not isinstance(row, dict):
            raise ConceptWorkflowError(f"rows.jsonl:{line_number}: row must be a JSON object")
        required = (
            _REQUIRED_EXPANDED_ROW_FIELDS if schema_version == 1 else _REQUIRED_SHARED_ROW_FIELDS
        )
        missing = sorted(required.difference(row))
        if missing:
            raise ConceptWorkflowError(
                f"rows.jsonl:{line_number}: missing fields: {', '.join(missing)}"
            )
        rows.append(row)
    return tuple(rows)


def _read_shared_concepts(path: Path, *, label_width: int) -> tuple[dict[str, str], dict[str, int]]:
    payload = _read_json_object(path)
    if set(payload) != {"schema_version", "concepts"} or payload["schema_version"] != 1:
        raise ConceptWorkflowError("concepts.json must contain schema_version=1 and concepts")
    concepts = payload["concepts"]
    if not isinstance(concepts, list) or len(concepts) != label_width:
        raise ConceptWorkflowError(f"concepts.json must contain {label_width} column definitions")
    names: dict[str, str] = {}
    columns: dict[str, int] = {}
    normalized_names: dict[str, str] = {}
    for expected_column, concept in enumerate(concepts):
        if not isinstance(concept, Mapping):
            raise ConceptWorkflowError("each concepts.json entry must be an object")
        required = {"column", "concept_id", "concept_name", "definition"}
        if set(concept) != required:
            raise ConceptWorkflowError(
                f"each concepts.json entry must contain exactly {sorted(required)}"
            )
        if concept["column"] != expected_column:
            raise ConceptWorkflowError(
                "concepts.json columns must be contiguous and match list order"
            )
        concept_id = concept["concept_id"]
        concept_name = concept["concept_name"]
        if not isinstance(concept_id, str) or not concept_id.strip():
            raise ConceptWorkflowError("concepts.json concept_id must be non-empty")
        if not isinstance(concept_name, str) or not concept_name.strip():
            raise ConceptWorkflowError("concepts.json concept_name must be non-empty")
        if concept_id in names:
            raise ConceptWorkflowError(f"duplicate concept_id {concept_id!r}")
        normalized = " ".join(concept_name.split()).casefold()
        prior_id = normalized_names.setdefault(normalized, concept_id)
        if prior_id != concept_id:
            raise ConceptWorkflowError(
                f"concept name {concept_name!r} maps to both {prior_id!r} and {concept_id!r}"
            )
        names[concept_id] = concept_name
        columns[concept_id] = expected_column
    return names, columns


def _nonempty_string(row: Mapping[str, Any], field: str, *, line_number: int) -> str:
    value = row[field]
    if not isinstance(value, str) or not value.strip():
        raise ConceptWorkflowError(f"rows.jsonl:{line_number}: {field} must be a non-empty string")
    return value


def _validate_rows(rows: tuple[Mapping[str, Any], ...], labels: NDArray[np.integer[Any]]) -> None:
    if len(rows) != labels.shape[0]:
        raise ConceptWorkflowError(
            f"rows/labels length mismatch: {len(rows)} rows versus {labels.shape[0]} labels"
        )

    concept_names: dict[str, str] = {}
    normalized_names: dict[str, str] = {}
    group_splits: dict[str, str] = {}
    coverage: dict[tuple[str, str], Counter[int]] = {}
    for index, row in enumerate(rows):
        line_number = index + 1
        row_index = row["row"]
        if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index != index:
            raise ConceptWorkflowError(
                f"rows.jsonl:{line_number}: row must equal its zero-based array index {index}"
            )
        label = row["label"]
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ConceptWorkflowError(
                f"rows.jsonl:{line_number}: label must be the integer 0 or 1"
            )
        if label != int(labels[index]):
            raise ConceptWorkflowError(
                f"labels/rows mismatch at row {index}: labels.npy={int(labels[index])}, "
                f"rows.jsonl={label}"
            )

        concept_id = _nonempty_string(row, "concept_id", line_number=line_number)
        concept_name = _nonempty_string(row, "concept_name", line_number=line_number)
        group_id = _nonempty_string(row, "group_id", line_number=line_number)
        split = _nonempty_string(row, "split", line_number=line_number)
        if split not in CANONICAL_SPLITS:
            raise ConceptWorkflowError(
                f"rows.jsonl:{line_number}: split must be one of {CANONICAL_SPLITS}"
            )

        prior_name = concept_names.setdefault(concept_id, concept_name)
        if prior_name != concept_name:
            raise ConceptWorkflowError(
                f"concept {concept_id!r} has inconsistent names {prior_name!r} and {concept_name!r}"
            )
        normalized_name = " ".join(concept_name.split()).casefold()
        prior_id = normalized_names.setdefault(normalized_name, concept_id)
        if prior_id != concept_id:
            raise ConceptWorkflowError(
                f"concept name {concept_name!r} maps to both {prior_id!r} and {concept_id!r}"
            )

        prior_split = group_splits.setdefault(group_id, split)
        if prior_split != split:
            raise ConceptWorkflowError(
                f"group {group_id!r} crosses splits {prior_split!r} and {split!r}"
            )
        coverage.setdefault((concept_id, split), Counter())[label] += 1

    for concept_id in sorted(concept_names):
        for split in CANONICAL_SPLITS:
            label_counts = coverage.get((concept_id, split), Counter())
            missing_labels = [label for label in (0, 1) if label_counts[label] == 0]
            if missing_labels:
                raise ConceptWorkflowError(
                    f"concept {concept_id!r} split {split!r} is missing labels {missing_labels}"
                )


def _validate_shared_rows(
    rows: tuple[Mapping[str, Any], ...],
    labels: NDArray[np.integer[Any]],
    concept_columns: Mapping[str, int],
) -> None:
    if len(rows) != labels.shape[0]:
        raise ConceptWorkflowError(
            f"rows/labels length mismatch: {len(rows)} rows versus {labels.shape[0]} labels"
        )
    if labels.shape[1] != len(concept_columns):
        raise ConceptWorkflowError(
            "labels/concepts width mismatch: "
            f"{labels.shape[1]} columns versus {len(concept_columns)} concepts"
        )

    group_splits: dict[str, str] = {}
    seen_hashes: dict[str, tuple[str, int]] = {}
    for index, row in enumerate(rows):
        line_number = index + 1
        row_index = row["row"]
        if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index != index:
            raise ConceptWorkflowError(
                f"rows.jsonl:{line_number}: row must equal its zero-based array index {index}"
            )
        group_id = _nonempty_string(row, "group_id", line_number=line_number)
        split = _nonempty_string(row, "split", line_number=line_number)
        _nonempty_string(row, "source", line_number=line_number)
        text_sha256 = _nonempty_string(row, "text_sha256", line_number=line_number)
        if len(text_sha256) != 64:
            raise ConceptWorkflowError(
                f"rows.jsonl:{line_number}: text_sha256 must have 64 characters"
            )
        try:
            int(text_sha256, 16)
        except ValueError as error:
            raise ConceptWorkflowError(
                f"rows.jsonl:{line_number}: text_sha256 must be hexadecimal"
            ) from error
        if split not in CANONICAL_SPLITS:
            raise ConceptWorkflowError(
                f"rows.jsonl:{line_number}: split must be one of {CANONICAL_SPLITS}"
            )
        prior_split = group_splits.setdefault(group_id, split)
        if prior_split != split:
            raise ConceptWorkflowError(
                f"group {group_id!r} crosses splits {prior_split!r} and {split!r}"
            )
        prior_hash = seen_hashes.setdefault(text_sha256, (split, line_number))
        if prior_hash != (split, line_number):
            raise ConceptWorkflowError(
                f"duplicate source text hash at rows {prior_hash[1]} and {line_number}"
            )

    for concept_id, column in concept_columns.items():
        for split in CANONICAL_SPLITS:
            split_mask = np.asarray([row["split"] == split for row in rows])
            split_labels = labels[split_mask, column]
            observed = set(int(value) for value in split_labels if value != -1)
            if observed != {0, 1}:
                raise ConceptWorkflowError(
                    f"concept {concept_id!r} split {split!r} must contain labels 0 and 1; "
                    f"observed {sorted(observed)}"
                )


def _artifact_digest(path: Path, layers: Mapping[int, Path], *, schema_version: int) -> str:
    files = [path / "metadata.json", path / "labels.npy", path / "rows.jsonl"]
    if schema_version == 2:
        files.append(path / "concepts.json")
    files.extend(layers[layer] for layer in sorted(layers))
    return stable_hash(f"{file_path.name}:{sha256_file(file_path)}" for file_path in files)


def _concept_names(rows: tuple[Mapping[str, Any], ...]) -> dict[str, str]:
    return {str(row["concept_id"]): str(row["concept_name"]) for row in rows}


def load_activation_artifact(path: str | Path) -> ActivationArtifact:
    source = Path(path)
    if not source.is_dir():
        raise ConceptWorkflowError(f"activation artifact is not a directory: {source}")
    metadata = _read_json_object(source / "metadata.json")
    schema_version = metadata.get("schema_version")
    if schema_version not in (1, 2):
        raise ConceptWorkflowError("metadata.schema_version must equal 1 or 2")
    if metadata.get("coordinate") != "resid_post":
        raise ConceptWorkflowError("activation coordinate must be 'resid_post'")
    if metadata.get("representation") != "last_non_padding_token":
        raise ConceptWorkflowError(
            "activation representation must be 'last_non_padding_token'; "
            "pooling artifacts are not valid probe inputs"
        )

    discovered_layers = _discover_layers(source)
    declared_layers = _metadata_layers(metadata)
    if set(declared_layers) != set(discovered_layers):
        raise ConceptWorkflowError(
            "metadata/layer-file mismatch: declared "
            f"{list(declared_layers)}, found {sorted(discovered_layers)}"
        )

    labels = _load_labels(source / "labels.npy", schema_version=schema_version)
    rows = _read_rows(source / "rows.jsonl", schema_version=schema_version)
    if schema_version == 1:
        _validate_rows(rows, labels)
        concept_names = _concept_names(rows)
        concept_columns = None
    else:
        concept_names, concept_columns = _read_shared_concepts(
            source / "concepts.json", label_width=labels.shape[1]
        )
        _validate_shared_rows(rows, labels, concept_columns)
    n_examples = metadata.get("n_examples")
    if isinstance(n_examples, bool) or not isinstance(n_examples, int):
        raise ConceptWorkflowError("metadata.n_examples must be an integer")
    if n_examples != labels.shape[0]:
        raise ConceptWorkflowError(
            f"metadata.n_examples={n_examples} does not match labels length {labels.shape[0]}"
        )

    expected_example_hash = stable_hash(json.dumps(row, sort_keys=True) for row in rows)
    if metadata.get("example_hash") != expected_example_hash:
        raise ConceptWorkflowError("metadata.example_hash does not match rows.jsonl")

    residual_width: int | None = None
    for layer in declared_layers:
        try:
            activations = load_activation_layer(source, layer)
        except (OSError, ValueError) as error:
            raise ConceptWorkflowError(
                f"cannot load {discovered_layers[layer].name}: {error}"
            ) from error
        if activations.ndim != 2 or activations.shape[0] != labels.shape[0]:
            raise ConceptWorkflowError(
                f"{discovered_layers[layer].name} must have shape "
                f"[{labels.shape[0]}, d_residual], got {activations.shape}"
            )
        if activations.shape[1] == 0:
            raise ConceptWorkflowError(f"{discovered_layers[layer].name} has zero residual width")
        if not np.issubdtype(activations.dtype, np.number):
            raise ConceptWorkflowError(f"{discovered_layers[layer].name} must have a numeric dtype")
        if residual_width is None:
            residual_width = int(activations.shape[1])
        elif activations.shape[1] != residual_width:
            raise ConceptWorkflowError("activation layers have different residual widths")

    return ActivationArtifact(
        path=source,
        metadata=metadata,
        labels=labels,
        rows=rows,
        layers=declared_layers,
        artifact_hash=_artifact_digest(source, discovered_layers, schema_version=schema_version),
        concept_names=concept_names,
        concept_columns=concept_columns,
    )


def activation_indices_by_split(
    artifact: ActivationArtifact, concept_id: str
) -> dict[str, NDArray[np.int64]]:
    column = None if artifact.concept_columns is None else artifact.concept_columns[concept_id]
    return {
        split: np.asarray(
            [
                index
                for index, row in enumerate(artifact.rows)
                if row["split"] == split
                and (
                    row.get("concept_id") == concept_id
                    if column is None
                    else int(artifact.labels[index, column]) != -1
                )
            ],
            dtype=np.int64,
        )
        for split in CANONICAL_SPLITS
    }


def activation_concept_labels(
    artifact: ActivationArtifact, concept_id: str
) -> NDArray[np.integer[Any]]:
    if artifact.concept_columns is None:
        return artifact.labels
    return artifact.labels[:, artifact.concept_columns[concept_id]]


def activation_groups(
    rows: tuple[Mapping[str, Any], ...], indices: NDArray[np.int64]
) -> NDArray[np.str_]:
    return np.asarray([str(rows[int(index)]["group_id"]) for index in indices])


__all__ = [
    "ActivationArtifact",
    "ConceptWorkflowError",
    "activation_concept_labels",
    "activation_groups",
    "activation_indices_by_split",
    "load_activation_artifact",
]
