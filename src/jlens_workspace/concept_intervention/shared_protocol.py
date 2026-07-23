"""Shared rows, RAPTOR-style probes, and validation-only layer selection.

This module owns the inputs that must be byte-identical across J-component,
RAPTOR, and ITI.  It contains no intervention hooks.  Probe coefficients and
intercepts are always mapped back from standardized features to the original
block-output ``resid_post`` coordinates.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from jlens_workspace.activations import load_activation_layer
from jlens_workspace.artifacts import atomic_write_json, sha256_file, stable_hash
from jlens_workspace.workflows.concept import (
    _concept_labels,
    _groups,
    _indices_by_split,
    _load_activation_artifact,
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
            raise SharedProtocolError(
                f"{concept_id} split {split!r} does not contain both labels"
            )
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
            f"{concept_id} split {split!r} cannot be exactly row-balanced "
            "without breaking groups"
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


def _row_identity(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {
        "activation_index": int(index),
        "group_id": str(row["group_id"]),
        "text_sha256": row.get("text_sha256"),
        "source_row_id": row.get("source_row_id", row.get("row_id")),
    }


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _raw_probe(
    pipeline: Pipeline, *, positive_label: int = 1
) -> tuple[np.ndarray, float]:
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]
    classes = np.asarray(classifier.classes_)
    positive_index = int(np.flatnonzero(classes == positive_label)[0])
    sign = 1.0 if positive_index == 1 else -1.0
    coefficient = sign * np.asarray(classifier.coef_[0], dtype=np.float64)
    intercept = sign * float(classifier.intercept_[0])
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    raw = coefficient / np.where(scale == 0.0, 1.0, scale)
    return raw, intercept - float(raw @ mean)


def _new_pipeline(*, max_iter: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    penalty="l2",
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )


def load_upstream_raptor_tuning(path: str | Path) -> Any:
    """Load the pinned author's probe-tuning module without copying its source."""

    root = Path(path).resolve()
    if not (root / ".git").exists() or not (root / "src/raptor").is_dir():
        raise SharedProtocolError(f"RAPTOR checkout is incomplete: {root}")
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip() != RAPTOR_COMMIT:
        raise SharedProtocolError(
            f"RAPTOR checkout must be pinned at {RAPTOR_COMMIT}"
        )
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    module = importlib.import_module("raptor.probes.tuning")
    module_path = Path(module.__file__).resolve()
    if root not in module_path.parents:
        raise SharedProtocolError(
            f"loaded RAPTOR probe tuning from another checkout: {module_path}"
        )
    return module


def _grouped_raptor_c_scores(
    activations: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    c_grid: Sequence[float],
    cv_folds: int,
    max_iter: int,
    seed: int,
    upstream: Any,
) -> np.ndarray:
    """Call the author's full C sweep inside every group-safe train fold."""

    expected_grid = np.asarray(c_grid, dtype=np.float64)
    observed_grid = np.asarray(upstream.RAPTOR_C_GRID, dtype=np.float64)
    if not np.allclose(expected_grid, observed_grid, rtol=1e-12, atol=0.0):
        raise SharedProtocolError("configured C grid differs from pinned RAPTOR")
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fold_scores: list[list[float]] = []
    for train_fold, validation_fold in cv.split(activations, labels, groups):
        scaler = StandardScaler().fit(activations[train_fold])
        train_values = scaler.transform(activations[train_fold])
        validation_values = scaler.transform(activations[validation_fold])
        captured: list[float] = []
        original_accuracy = upstream.accuracy_score

        def record_accuracy(
            truth: Any,
            prediction: Any,
            *,
            metric: Any = original_accuracy,
            output: list[float] = captured,
        ) -> float:
            value = float(metric(truth, prediction))
            output.append(value)
            return value

        upstream.accuracy_score = record_accuracy
        try:
            upstream.tune_raptor_c(
                train_values,
                labels[train_fold],
                validation_values,
                labels[validation_fold],
                max_iter=max_iter,
            )
        finally:
            upstream.accuracy_score = original_accuracy
        if len(captured) != len(expected_grid):
            raise SharedProtocolError(
                "pinned RAPTOR tuning did not evaluate the complete C grid"
            )
        fold_scores.append(captured)
    return np.asarray(fold_scores, dtype=np.float64)


def _fit_one_layer(
    activations: np.ndarray,
    labels: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    indices: Mapping[str, np.ndarray],
    *,
    c_grid: Sequence[float],
    cv_folds: int,
    max_iter: int,
    seed: int,
    upstream: Any,
) -> dict[str, Any]:
    train = indices["train"]
    validation = indices["validation"]
    test = indices["test"]
    fold_scores = _grouped_raptor_c_scores(
        np.asarray(activations[train], dtype=np.float64),
        labels[train],
        _groups(tuple(rows), train),
        c_grid=c_grid,
        cv_folds=cv_folds,
        max_iter=max_iter,
        seed=seed,
        upstream=upstream,
    )
    mean_scores = np.mean(fold_scores, axis=0)
    chosen_index = int(np.argmax(mean_scores))
    chosen_c = float(c_grid[chosen_index])

    train_pipeline = _new_pipeline(max_iter=max_iter, seed=seed)
    train_pipeline.set_params(classifier__C=chosen_c)
    train_pipeline.fit(activations[train], labels[train])
    validation_prob = train_pipeline.predict_proba(activations[validation])[
        :, int(np.flatnonzero(train_pipeline.classes_ == 1)[0])
    ]
    validation_prediction = train_pipeline.predict(activations[validation])

    fit_indices = np.concatenate((train, validation))
    final_pipeline = _new_pipeline(max_iter=max_iter, seed=seed)
    final_pipeline.set_params(classifier__C=chosen_c)
    final_pipeline.fit(activations[fit_indices], labels[fit_indices])
    raw, intercept = _raw_probe(final_pipeline)
    test_prob = final_pipeline.predict_proba(activations[test])[
        :, int(np.flatnonzero(final_pipeline.classes_ == 1)[0])
    ]
    test_prediction = final_pipeline.predict(activations[test])
    return {
        "chosen_C": chosen_c,
        "cv_best_accuracy": float(mean_scores[chosen_index]),
        "cv_fold_accuracies_at_chosen_C": [
            float(value) for value in fold_scores[:, chosen_index]
        ],
        "validation_accuracy": float(
            accuracy_score(labels[validation], validation_prediction)
        ),
        "validation_roc_auc": float(roc_auc_score(labels[validation], validation_prob)),
        "test_accuracy": float(accuracy_score(labels[test], test_prediction)),
        "test_roc_auc": float(roc_auc_score(labels[test], test_prob)),
        "coef_raw": raw,
        "intercept_raw": float(intercept),
        "n_iter": int(np.max(final_pipeline.named_steps["classifier"].n_iter_)),
    }


def run_shared_layer_selection(
    *,
    activation_artifact: str | Path,
    output_dir: str | Path,
    candidate_layers: Sequence[int],
    selected_layer_count: int,
    concept_ids: Sequence[str],
    c_grid: Sequence[float] = RAPTOR_C_GRID,
    cv_folds: int = 5,
    max_iter: int = 5000,
    seed: int = 42,
    raptor_upstream_checkout: str | Path = "third_party_external/RAPTOR",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create balanced rows, RAPTOR probes, and per-concept Top-M layers."""

    artifact = _load_activation_artifact(activation_artifact)
    layers = tuple(int(layer) for layer in candidate_layers)
    if tuple(sorted(set(layers))) != layers:
        raise SharedProtocolError("candidate layers must be sorted and unique")
    missing = sorted(set(layers) - set(artifact.layers))
    if missing:
        raise SharedProtocolError(f"activation artifact lacks candidate layers {missing}")
    if not 0 < selected_layer_count < len(layers):
        raise SharedProtocolError("selected_layer_count must lie in [1, layers-1]")
    concepts = tuple(str(value) for value in concept_ids)
    unknown = sorted(set(concepts) - set(artifact.concept_names))
    if unknown:
        raise SharedProtocolError(f"activation artifact lacks concepts {unknown}")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(f"shared protocol output exists: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    upstream_tuning = load_upstream_raptor_tuning(raptor_upstream_checkout)

    row_manifest: dict[str, Any] = {
        "schema_version": 1,
        "method": "deterministic_group_safe_exact_row_balance",
        "seed": seed,
        "activation_artifact": str(artifact.path),
        "activation_artifact_hash": artifact.artifact_hash,
        "concepts": {},
    }
    balanced_indices: dict[str, dict[str, np.ndarray]] = {}
    for concept_id in concepts:
        labels = np.asarray(_concept_labels(artifact, concept_id), dtype=np.int8)
        available = _indices_by_split(artifact, concept_id)
        split_indices: dict[str, np.ndarray] = {}
        split_payload: dict[str, Any] = {}
        selected_groups: dict[str, set[str]] = {}
        for split in ("train", "validation", "test"):
            eligible = set(int(value) for value in available[split])
            selected = deterministic_balanced_indices(
                labels,
                artifact.rows,
                concept_id=concept_id,
                split=split,
                seed=seed,
            )
            if not set(int(value) for value in selected).issubset(eligible):
                raise SharedProtocolError("balanced row selector escaped eligible rows")
            split_indices[split] = selected
            identities = [
                _row_identity(artifact.rows[int(index)], int(index))
                for index in selected
            ]
            selected_groups[split] = {
                str(identity["group_id"]) for identity in identities
            }
            split_payload[split] = {
                "count": int(selected.size),
                "label_0": int(np.sum(labels[selected] == 0)),
                "label_1": int(np.sum(labels[selected] == 1)),
                "rows": identities,
                "rows_hash": stable_hash(
                    json.dumps(row, sort_keys=True) for row in identities
                ),
            }
        for first, second in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            overlap = selected_groups[first].intersection(selected_groups[second])
            if overlap:
                raise SharedProtocolError(
                    f"{concept_id} groups overlap {first}/{second}: "
                    f"{sorted(overlap)[:5]}"
                )
        balanced_indices[concept_id] = split_indices
        row_manifest["concepts"][concept_id] = split_payload
    atomic_write_json(destination / "row_manifest.json", row_manifest)

    selection_entries: list[dict[str, Any]] = []
    activation_provenance = {
        "artifact_hash": artifact.artifact_hash,
        "coordinate": artifact.metadata["coordinate"],
        "representation": artifact.metadata["representation"],
        "add_special_tokens": artifact.metadata.get("add_special_tokens"),
        "manifest": artifact.metadata.get("manifest"),
    }
    probe_entries: list[dict[str, Any]] = []
    for concept_index, concept_id in enumerate(concepts):
        labels = np.asarray(_concept_labels(artifact, concept_id), dtype=np.int8)
        layer_metrics: list[dict[str, Any]] = []
        for layer in layers:
            activations = load_activation_layer(artifact.path, layer)
            result = _fit_one_layer(
                activations,
                labels,
                artifact.rows,
                balanced_indices[concept_id],
                c_grid=c_grid,
                cv_folds=cv_folds,
                max_iter=max_iter,
                seed=seed + concept_index * 1009 + layer,
                upstream=upstream_tuning,
            )
            concept_directory = (
                destination
                / "raptor_probes"
                / f"layer_{layer:02d}"
                / f"concept_{quote(concept_id, safe='')}"
            )
            vector_path = concept_directory / "probe_vector.npy"
            _atomic_save_npy(vector_path, result.pop("coef_raw"))
            metrics = {
                "schema_version": 1,
                "method": "raptor_resid_post_linear_probe",
                "layer": layer,
                "concept_id": concept_id,
                "concept_name": artifact.concept_names[concept_id],
                "artifact_hash": artifact.artifact_hash,
                "activation": activation_provenance,
                **result,
                "solver": "lbfgs",
                "penalty": "l2",
                "standardize": True,
                "C_selection": "train_only_stratified_group_cv_accuracy",
                "C_tuning_core": "pinned_raptor.probes.tuning.tune_raptor_c",
                "validation_use": "layer_selection_only",
                "final_fit_split": "train+validation",
                "test_use": "held_out_report_only",
                "probe_vector": str(vector_path.relative_to(destination)),
                "probe_vector_sha256": sha256_file(vector_path),
                "row_manifest_sha256": sha256_file(
                    destination / "row_manifest.json"
                ),
                "intercept_raw": result["intercept_raw"],
                "probe": {
                    "vector_file": vector_path.name,
                    "vector_sha256": sha256_file(vector_path),
                    "coordinate": "resid_post",
                    "dimension": int(
                        np.load(vector_path, mmap_mode="r").shape[0]
                    ),
                    "dtype": "float64",
                    "intercept_raw": result["intercept_raw"],
                    "positive_label": 1,
                    "negative_label": 0,
                    "standardize": True,
                    "fit_split": "train+validation",
                },
            }
            metrics_path = concept_directory / "metrics.json"
            atomic_write_json(metrics_path, metrics)
            probe_entries.append(
                {
                    "layer": layer,
                    "concept_id": concept_id,
                    "vector_file": str(
                        vector_path.relative_to(destination / "raptor_probes")
                    ),
                    "vector_sha256": sha256_file(vector_path),
                    "metrics_file": str(
                        metrics_path.relative_to(destination / "raptor_probes")
                    ),
                }
            )
            layer_metrics.append(metrics)
        ranked = sorted(
            layer_metrics,
            key=lambda row: (-float(row["validation_accuracy"]), int(row["layer"])),
        )
        selected = [int(row["layer"]) for row in ranked[:selected_layer_count]]
        selection_entries.append(
            {
                "concept_id": concept_id,
                "candidate_layers": list(layers),
                "selected_layers": sorted(selected),
                "ranked_layers": [
                    {
                        "rank": rank,
                        "layer": int(row["layer"]),
                        "validation_accuracy": float(row["validation_accuracy"]),
                        "chosen_C": float(row["chosen_C"]),
                    }
                    for rank, row in enumerate(ranked, start=1)
                ],
                "tie_break": "smaller_layer_id",
            }
        )

    selection = {
        "schema_version": 1,
        "method": "raptor_validation_accuracy",
        "candidate_layers": list(layers),
        "selected_layer_count": selected_layer_count,
        "concepts": selection_entries,
        "test_examples_used_for_C_or_layer_selection": 0,
        "row_manifest": "row_manifest.json",
        "row_manifest_sha256": sha256_file(destination / "row_manifest.json"),
        "activation_artifact": str(artifact.path),
        "activation_artifact_hash": artifact.artifact_hash,
        "raptor_upstream": {
            "repository": RAPTOR_REPOSITORY,
            "commit": RAPTOR_COMMIT,
            "license_in_upstream_root": None,
            "external_checkout": str(Path(raptor_upstream_checkout).resolve()),
            "probe_core_called_directly": "raptor.probes.tuning.tune_raptor_c",
            "declared_adaptation": (
                "RAPTOR scaler/L2-LBFGS/100-C probe with C selected by "
                "group-safe train-only CV; validation accuracy selects layers"
            ),
        },
    }
    atomic_write_json(destination / "layer_selection.json", selection)
    atomic_write_json(
        destination / "raptor_probes" / "manifest.json",
        {
            "schema_version": 1,
            "workflow": "shared_raptor_probe_fitting",
            "activation_artifact": str(artifact.path),
            "activation_artifact_hash": artifact.artifact_hash,
            "activation": activation_provenance,
            "layers": list(layers),
            "concept_ids": list(concepts),
            "probe_coordinate": "resid_post",
            "selection": {
                "C_grid": [float(value) for value in c_grid],
                "cv_folds": cv_folds,
                "scoring": "accuracy",
                "standardize": True,
                "solver": "lbfgs",
                "random_state": seed,
                "max_iter": max_iter,
            },
            "probes": probe_entries,
        },
    )
    manifest = {
        "schema_version": 1,
        "workflow": "shared_intervention_protocol",
        "complete": True,
        "layer_selection_sha256": sha256_file(
            destination / "layer_selection.json"
        ),
        "row_manifest_sha256": sha256_file(destination / "row_manifest.json"),
        "probe_count": len(concepts) * len(layers),
    }
    atomic_write_json(destination / "index.json", manifest)
    return selection


def load_selected_layers(path: str | Path, concept_id: str) -> tuple[int, ...]:
    """Load one immutable per-concept selected-layer tuple."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("method") != "raptor_validation_accuracy":
        raise SharedProtocolError("unsupported shared layer-selection artifact")
    matches = [
        row
        for row in payload.get("concepts", [])
        if str(row.get("concept_id")) == concept_id
    ]
    if len(matches) != 1:
        raise SharedProtocolError(
            f"layer selection has {len(matches)} rows for {concept_id!r}"
        )
    layers = tuple(int(value) for value in matches[0]["selected_layers"])
    expected = int(payload["selected_layer_count"])
    if len(layers) != expected or tuple(sorted(set(layers))) != layers:
        raise SharedProtocolError("selected layers are not sorted, unique, or complete")
    return layers


def load_balanced_indices(
    path: str | Path, concept_id: str, split: str
) -> np.ndarray:
    """Load activation indices from the common row manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        rows = payload["concepts"][concept_id][split]["rows"]
    except (KeyError, TypeError) as error:
        raise SharedProtocolError(
            f"row manifest lacks {concept_id!r}/{split!r}"
        ) from error
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
