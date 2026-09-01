"""Frozen row selection and shared artifact contracts for steering methods.

This module owns deterministic row balance, pinned protocol constants, and
pure readers/identity checks for the inputs shared by J-component, RAPTOR, and
ITI. It performs no probe fitting, hyperparameter selection, or intervention.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jlens_workspace.foundation.artifacts import (
    sha256_file,
)

RAPTOR_REPOSITORY = "https://github.com/Ziqi-Gao/RAPTOR.git"
RAPTOR_COMMIT = "cf7405899174af39f3970e093e4b86bf0972ff87"
RAPTOR_C_GRID = tuple(np.logspace(-4, 2, 100, dtype=np.float64))


class SharedProtocolError(ValueError):
    """Raised when a common input would differ across intervention methods."""


def _stable_group_key(*, seed: int, concept_id: str, split: str, group: str) -> str:
    value = f"{seed}\0{concept_id}\0{split}\0{group}".encode()
    return hashlib.sha256(value).hexdigest()


def deterministic_balanced_indices(
    labels: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    concept_id: str,
    split: str,
    seed: int,
) -> np.ndarray:
    """Select an exact class-balanced set while keeping whole groups together.

    Groups are ordered by a stable SHA-256 key.  The largest common cumulative
    row count between the two class-specific group prefixes is used.  This is
    exact for the production one-row groups and fails closed when unusual group
    sizes cannot produce a non-empty exact balance.
    """

    values = np.asarray(labels)
    if values.ndim != 1 or values.shape[0] != len(rows):
        raise SharedProtocolError("labels and rows must share shape [N]")
    rows_by_group: dict[str, list[int]] = defaultdict(list)
    for index, (label, row) in enumerate(zip(values, rows, strict=True)):
        if str(row.get("split")) != split:
            continue
        numeric = int(label)
        if numeric not in {0, 1}:
            continue
        group = str(row.get("group_id", ""))
        if not group:
            raise SharedProtocolError("every selected row needs a non-empty group_id")
        rows_by_group[group].append(index)

    grouped: dict[int, dict[str, list[int]]] = {
        0: defaultdict(list),
        1: defaultdict(list),
    }
    for group, indices in rows_by_group.items():
        group_labels = {int(values[index]) for index in indices}
        if len(group_labels) != 1:
            raise SharedProtocolError(
                f"{concept_id} split {split!r} group {group!r} spans both labels"
            )
        grouped[group_labels.pop()][group].extend(indices)

    ordered: dict[int, list[tuple[str, list[int]]]] = {}
    cumulative: dict[int, dict[int, int]] = {}
    for label in (0, 1):
        if not grouped[label]:
            raise SharedProtocolError(f"{concept_id} split {split!r} does not contain both labels")
        ordered[label] = sorted(
            grouped[label].items(),
            key=lambda item: _stable_group_key(
                seed=seed,
                concept_id=concept_id,
                split=split,
                group=item[0],
            ),
        )
        running = 0
        cumulative[label] = {}
        for prefix, (_, indices) in enumerate(ordered[label], start=1):
            running += len(indices)
            cumulative[label][running] = prefix

    common_counts = sorted(set(cumulative[0]).intersection(cumulative[1]))
    if not common_counts:
        raise SharedProtocolError(
            f"{concept_id} split {split!r} cannot be exactly row-balanced without breaking groups"
        )
    count = common_counts[-1]
    selected: list[int] = []
    for label in (0, 1):
        prefix = cumulative[label][count]
        for _, indices in ordered[label][:prefix]:
            selected.extend(indices)
    selected.sort(
        key=lambda index: hashlib.sha256(
            f"{seed}\0{concept_id}\0{split}\0{index}".encode()
        ).hexdigest()
    )
    output = np.asarray(selected, dtype=np.int64)
    selected_labels = values[output]
    if int(np.sum(selected_labels == 0)) != int(np.sum(selected_labels == 1)):
        raise SharedProtocolError("internal error: balanced manifest is not balanced")
    return output


def load_selected_layers(path: str | Path, concept_id: str) -> tuple[int, ...]:
    """Load one immutable per-concept selected-layer tuple."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("method") != "raptor_validation_accuracy":
        raise SharedProtocolError("unsupported shared layer-selection artifact")
    matches = [
        row for row in payload.get("concepts", []) if str(row.get("concept_id")) == concept_id
    ]
    if len(matches) != 1:
        raise SharedProtocolError(f"layer selection has {len(matches)} rows for {concept_id!r}")
    layers = tuple(int(value) for value in matches[0]["selected_layers"])
    expected = int(payload["selected_layer_count"])
    if len(layers) != expected or tuple(sorted(set(layers))) != layers:
        raise SharedProtocolError("selected layers are not sorted, unique, or complete")
    return layers


def load_balanced_indices(path: str | Path, concept_id: str, split: str) -> np.ndarray:
    """Load activation indices from the common row manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        rows = payload["concepts"][concept_id][split]["rows"]
    except (KeyError, TypeError) as error:
        raise SharedProtocolError(f"row manifest lacks {concept_id!r}/{split!r}") from error
    indices = np.asarray([int(row["activation_index"]) for row in rows], dtype=np.int64)
    if indices.size == 0 or len(set(indices.tolist())) != indices.size:
        raise SharedProtocolError("balanced row indices must be non-empty and unique")
    return indices


def validate_shared_identity(
    layer_selection_path: str | Path,
    *,
    row_manifest_path: str | Path,
    activation_artifact_hash: str,
) -> dict[str, Any]:
    """Fail closed when a method does not reference the registered shared inputs."""

    selection = json.loads(Path(layer_selection_path).read_text(encoding="utf-8"))
    if selection.get("activation_artifact_hash") != activation_artifact_hash:
        raise SharedProtocolError("shared activation artifact hash mismatch")
    observed = sha256_file(row_manifest_path)
    if selection.get("row_manifest_sha256") != observed:
        raise SharedProtocolError("shared row manifest SHA-256 mismatch")
    return selection
