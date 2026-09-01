"""End-to-end probe fitting from a residual-activation artifact.

The workflow enforces the statistical sequence used by concept steering:

1. choose ``C`` by grouped cross-validation on ``train`` only;
2. report ``validation`` as a diagnostic without using it for selection;
3. refit the chosen ``C`` on ``train + validation``;
4. evaluate ``test`` exactly once.

Only raw-coordinate probe vectors and JSON metrics are persisted.  Fitted
scikit-learn estimators are intentionally never pickled.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from numpy.typing import NDArray

from jlens_workspace.concept_intervention.data.activation_artifact import (
    ActivationArtifact,
    ConceptWorkflowError,
)
from jlens_workspace.concept_intervention.data.activation_artifact import (
    activation_concept_labels as _concept_labels,
)
from jlens_workspace.concept_intervention.data.activation_artifact import (
    activation_groups as _groups,
)
from jlens_workspace.concept_intervention.data.activation_artifact import (
    activation_indices_by_split as _indices_by_split,
)
from jlens_workspace.concept_intervention.data.activation_artifact import (
    load_activation_artifact as _load_activation_artifact,
)
from jlens_workspace.concept_intervention.probing.logistic import (
    CVScore,
    HeldOutMetrics,
    fit_logistic_probe,
)
from jlens_workspace.data import CANONICAL_SPLITS
from jlens_workspace.foundation.activations import load_activation_layer
from jlens_workspace.foundation.artifacts import atomic_write_json, sha256_file

_ActivationArtifact = ActivationArtifact


@dataclass(frozen=True, slots=True)
class ConceptProbeOutput:
    """Paths and identity for one fitted layer/concept probe."""

    layer: int
    concept_id: str
    concept_name: str
    chosen_C: float
    probe_vector_path: Path
    metrics_path: Path


@dataclass(frozen=True, slots=True)
class ConceptWorkflowResult:
    """All immutable outputs produced from one activation artifact."""

    activation_artifact: Path
    output_dir: Path
    artifact_hash: str
    probes: tuple[ConceptProbeOutput, ...]


def _choose_layers(available: tuple[int, ...], requested: Sequence[int] | None) -> tuple[int, ...]:
    if requested is None:
        return available
    selected: list[int] = []
    for layer in requested:
        if isinstance(layer, bool) or not isinstance(layer, int | np.integer):
            raise TypeError("layers must contain integers")
        selected.append(int(layer))
    if not selected:
        raise ValueError("layers must be non-empty when supplied")
    if len(set(selected)) != len(selected):
        raise ValueError("layers contains duplicates")
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ConceptWorkflowError(f"requested layers are absent from artifact: {unknown}")
    return tuple(sorted(selected))


def _choose_concepts(
    available: Mapping[str, str], requested: Sequence[str] | None
) -> tuple[str, ...]:
    if requested is None:
        return tuple(sorted(available))
    selected = list(requested)
    if not selected:
        raise ValueError("concept_ids must be non-empty when supplied")
    if any(not isinstance(concept_id, str) or not concept_id for concept_id in selected):
        raise TypeError("concept_ids must contain non-empty strings")
    if len(set(selected)) != len(selected):
        raise ValueError("concept_ids contains duplicates")
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ConceptWorkflowError(f"requested concepts are absent from artifact: {unknown}")
    return tuple(sorted(selected))


def _split_counts(
    rows: tuple[Mapping[str, Any], ...],
    concept_labels: NDArray[np.integer[Any]],
    indices: NDArray[np.int64],
) -> dict[str, int]:
    split_labels = concept_labels[indices]
    return {
        "total": int(indices.size),
        "label_0": int(np.sum(split_labels == 0)),
        "label_1": int(np.sum(split_labels == 1)),
        "groups": int(np.unique(_groups(rows, indices)).size),
    }


def _metrics_payload(metrics: HeldOutMetrics) -> dict[str, float]:
    return {
        "roc_auc": metrics.roc_auc,
        "average_precision": metrics.average_precision,
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
    }


def _cv_payload(score: CVScore) -> dict[str, Any]:
    return {
        "C": score.C,
        "mean_auc": score.mean_auc,
        "std_auc": score.std_auc,
        "fold_auc": list(score.fold_auc),
    }


def _atomic_save_npy(path: Path, array: NDArray[np.float64]) -> None:
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


def _prepare_output_dir(source: Path, output_dir: str | Path, *, overwrite: bool) -> Path:
    destination = Path(output_dir)
    source_resolved = source.resolve()
    destination_resolved = destination.resolve()
    paths_overlap = (
        destination_resolved == source_resolved
        or source_resolved in destination_resolved.parents
        or destination_resolved in source_resolved.parents
    )
    if paths_overlap:
        raise ValueError("output_dir and activation artifact must not overlap")
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"workflow output is not a directory: {destination}")
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(f"workflow output already exists: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def run_concept_probe_workflow(
    activation_artifact: str | Path,
    output_dir: str | Path,
    *,
    layers: Sequence[int] | None = None,
    concept_ids: Sequence[str] | None = None,
    C_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    cv_splits: int = 5,
    standardize: bool = True,
    class_weight: str | dict[Any, float] | None = "balanced",
    random_state: int = 0,
    max_iter: int = 2_000,
    n_jobs: int = 1,
    overwrite: bool = False,
) -> ConceptWorkflowResult:
    """Fit and persist leakage-safe probes for every selected layer/concept.

    The activation artifact must be the directory written by
    :func:`capture_residual_activations`.  ``probe_vector.npy`` is float64 in
    the original ``resid_post`` coordinates.  The adjacent ``metrics.json``
    records selection, validation/test metrics, counts, intercept, and source
    artifact hash; no estimator serialization is produced.
    """

    artifact = _load_activation_artifact(activation_artifact)
    selected_layers = _choose_layers(artifact.layers, layers)
    available_concepts = artifact.concept_names
    selected_concepts = _choose_concepts(available_concepts, concept_ids)
    destination = _prepare_output_dir(artifact.path, output_dir, overwrite=overwrite)
    activation_provenance = {
        "artifact_hash": artifact.artifact_hash,
        "coordinate": artifact.metadata["coordinate"],
        "representation": artifact.metadata["representation"],
        "add_special_tokens": artifact.metadata.get("add_special_tokens"),
        "manifest": artifact.metadata.get("manifest"),
    }

    outputs: list[ConceptProbeOutput] = []
    for layer in selected_layers:
        activations = load_activation_layer(artifact.path, layer)
        if not np.isfinite(activations).all():
            raise ConceptWorkflowError(f"layer_{layer:02d}.npy contains NaN or infinity")
        for concept_id in selected_concepts:
            concept_labels = _concept_labels(artifact, concept_id)
            split_indices = _indices_by_split(artifact, concept_id)
            train_indices = split_indices["train"]
            validation_indices = split_indices["validation"]
            test_indices = split_indices["test"]

            diagnostic = fit_logistic_probe(
                activations[train_indices],
                concept_labels[train_indices],
                activations[validation_indices],
                concept_labels[validation_indices],
                C_grid=C_grid,
                cv_splits=cv_splits,
                groups=_groups(artifact.rows, train_indices),
                positive_label=1,
                standardize=standardize,
                class_weight=class_weight,
                random_state=random_state,
                max_iter=max_iter,
                n_jobs=n_jobs,
            )

            train_validation_indices = np.concatenate((train_indices, validation_indices))
            # A one-value C grid cannot retune on validation.  This second use
            # of the public probe API only refits on train+validation and
            # computes the single test evaluation.
            final = fit_logistic_probe(
                activations[train_validation_indices],
                concept_labels[train_validation_indices],
                activations[test_indices],
                concept_labels[test_indices],
                C_grid=(diagnostic.chosen_C,),
                cv_splits=2,
                groups=_groups(artifact.rows, train_validation_indices),
                positive_label=1,
                standardize=standardize,
                class_weight=class_weight,
                random_state=random_state,
                max_iter=max_iter,
                n_jobs=n_jobs,
            )

            concept_directory = (
                destination / f"layer_{layer:02d}" / f"concept_{quote(concept_id, safe='')}"
            )
            vector_path = concept_directory / "probe_vector.npy"
            metrics_path = concept_directory / "metrics.json"
            _atomic_save_npy(vector_path, np.asarray(final.coef_raw, dtype=np.float64))
            vector_sha256 = sha256_file(vector_path)
            payload = {
                "schema_version": 1,
                "layer": layer,
                "concept_id": concept_id,
                "concept_name": available_concepts[concept_id],
                "artifact_hash": artifact.artifact_hash,
                "activation": activation_provenance,
                "chosen_C": diagnostic.chosen_C,
                "cv_strategy": diagnostic.cv_strategy,
                "cv_splits": cv_splits,
                "cv_scores": [_cv_payload(score) for score in diagnostic.cv_scores],
                "validation": _metrics_payload(diagnostic.heldout),
                "test": _metrics_payload(final.heldout),
                "counts": {
                    split: _split_counts(artifact.rows, concept_labels, split_indices[split])
                    for split in CANONICAL_SPLITS
                }
                | {
                    "train_validation": _split_counts(
                        artifact.rows,
                        concept_labels,
                        train_validation_indices,
                    )
                },
                "probe": {
                    "vector_file": vector_path.name,
                    "vector_sha256": vector_sha256,
                    "coordinate": "resid_post",
                    "dimension": int(final.coef_raw.shape[0]),
                    "dtype": "float64",
                    "intercept_raw": final.intercept_raw,
                    "positive_label": 1,
                    "negative_label": 0,
                    "standardize": standardize,
                    "fit_split": "train+validation",
                },
            }
            atomic_write_json(metrics_path, payload)
            outputs.append(
                ConceptProbeOutput(
                    layer=layer,
                    concept_id=concept_id,
                    concept_name=available_concepts[concept_id],
                    chosen_C=diagnostic.chosen_C,
                    probe_vector_path=vector_path,
                    metrics_path=metrics_path,
                )
            )

    atomic_write_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "workflow": "concept_probe_fitting",
            "activation_artifact": str(artifact.path),
            "activation_artifact_hash": artifact.artifact_hash,
            "activation": activation_provenance,
            "layers": list(selected_layers),
            "concept_ids": list(selected_concepts),
            "probe_coordinate": "resid_post",
            "selection": {
                "C_grid": [float(value) for value in C_grid],
                "cv_splits": cv_splits,
                "standardize": standardize,
                "class_weight": class_weight,
                "random_state": random_state,
                "max_iter": max_iter,
            },
            "probes": [
                {
                    "layer": output.layer,
                    "concept_id": output.concept_id,
                    "vector_file": str(output.probe_vector_path.relative_to(destination)),
                    "vector_sha256": sha256_file(output.probe_vector_path),
                    "metrics_file": str(output.metrics_path.relative_to(destination)),
                }
                for output in outputs
            ],
        },
    )

    return ConceptWorkflowResult(
        activation_artifact=artifact.path,
        output_dir=destination,
        artifact_hash=artifact.artifact_hash,
        probes=tuple(outputs),
    )


# Concise aliases for programmatic experiment drivers.
run_concept_workflow = run_concept_probe_workflow
fit_concept_probes_from_artifact = run_concept_probe_workflow
