"""Group-bootstrap stability replicates for fixed-C concept probes.

The primary probe workflow selects ``C`` on train only and fits its final
direction on train+validation. This module holds that selected ``C`` fixed,
bootstraps train+validation groups within class, and refits only the direction.
The test split is never loaded into a fit or used for replicate selection.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote

import numpy as np

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
    fit_fixed_logistic_direction,
)
from jlens_workspace.concept_intervention.protocol.shared import (
    load_balanced_indices,
)
from jlens_workspace.foundation.activations import load_activation_layer
from jlens_workspace.foundation.artifacts import atomic_write_json, sha256_file


class ProbeReplicateError(ValueError):
    """Raised when bootstrap probe inputs violate the artifact contract."""


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


def _primary_metrics_path(probes: Path, *, layer: int, concept_id: str) -> Path:
    return probes / f"layer_{layer:02d}" / f"concept_{quote(concept_id, safe='')}" / "metrics.json"


def _bootstrap_group_weights(
    groups: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Return stratified group-bootstrap multiplicities per row."""

    if groups.shape != labels.shape or groups.ndim != 1:
        raise ProbeReplicateError("groups and labels must share shape [N]")
    group_labels: dict[object, int] = {}
    for group, raw_label in zip(groups, labels, strict=True):
        label = int(raw_label)
        prior = group_labels.setdefault(group, label)
        if prior != label:
            raise ProbeReplicateError(f"group {group!r} has mixed labels within one concept")

    rng = np.random.Generator(np.random.Philox(seed))
    counts: Counter[object] = Counter()
    for label in (0, 1):
        class_groups = np.asarray(
            sorted(
                (group for group, value in group_labels.items() if value == label),
                key=str,
            ),
            dtype=object,
        )
        if class_groups.size < 2:
            raise ProbeReplicateError(f"class {label} needs at least two groups for bootstrap")
        sampled = rng.choice(class_groups, size=class_groups.size, replace=True)
        counts.update(sampled.tolist())
    return np.asarray([counts[group] for group in groups], dtype=np.float64)


def build_probe_bootstrap_replicates(
    *,
    activation_artifact: str | Path,
    primary_probes: str | Path,
    output_dir: str | Path,
    layers: Sequence[int],
    concept_ids: Sequence[str],
    seeds: Sequence[int],
    standardize: bool = True,
    class_weight: str | None = "balanced",
    max_iter: int = 5_000,
    row_manifest_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Fit fixed-C group-bootstrap directions and write an immutable manifest."""

    artifact = _load_activation_artifact(activation_artifact)
    probes = Path(primary_probes)
    destination = Path(output_dir)
    selected_layers = tuple(sorted(set(int(layer) for layer in layers)))
    selected_concepts = tuple(dict.fromkeys(str(value) for value in concept_ids))
    selected_seeds = tuple(int(seed) for seed in seeds)
    if not selected_layers or not selected_concepts or not selected_seeds:
        raise ProbeReplicateError("layers, concept_ids, and seeds must be non-empty")
    if len(set(selected_seeds)) != len(selected_seeds) or any(seed < 0 for seed in selected_seeds):
        raise ProbeReplicateError("bootstrap seeds must be unique and non-negative")
    unknown_layers = sorted(set(selected_layers) - set(artifact.layers))
    if unknown_layers:
        raise ProbeReplicateError(f"activation layers unavailable: {unknown_layers}")
    unknown_concepts = sorted(set(selected_concepts) - set(artifact.concept_names))
    if unknown_concepts:
        raise ProbeReplicateError(f"activation concepts unavailable: {unknown_concepts}")

    entries: list[dict[str, object]] = []
    for layer in selected_layers:
        activations = load_activation_layer(artifact.path, layer)
        for concept_id in selected_concepts:
            primary_path = _primary_metrics_path(probes, layer=layer, concept_id=concept_id)
            try:
                primary = json.loads(primary_path.read_text(encoding="utf-8"))
            except FileNotFoundError as error:
                raise ProbeReplicateError(
                    f"missing primary probe metrics: {primary_path}"
                ) from error
            if primary.get("artifact_hash") != artifact.artifact_hash:
                raise ProbeReplicateError(
                    f"primary probe activation identity mismatch: {primary_path}"
                )
            chosen_c = float(primary["chosen_C"])
            labels = _concept_labels(artifact, concept_id)
            if row_manifest_path is None:
                split_indices = _indices_by_split(artifact, concept_id)
                fit_indices = np.concatenate((split_indices["train"], split_indices["validation"]))
            else:
                fit_indices = np.concatenate(
                    [
                        load_balanced_indices(row_manifest_path, concept_id, split)
                        for split in ("train", "validation")
                    ]
                )
            fit_groups = np.asarray(_groups(artifact.rows, fit_indices), dtype=object)
            fit_labels = np.asarray(labels[fit_indices], dtype=np.int64)
            fit_activations = activations[fit_indices]

            for seed in selected_seeds:
                replicate_id = f"bootstrap_{seed}"
                replicate_dir = (
                    destination
                    / f"layer_{layer:02d}"
                    / f"concept_{quote(concept_id, safe='')}"
                    / replicate_id
                )
                vector_path = replicate_dir / "probe_vector.npy"
                metrics_path = replicate_dir / "metrics.json"
                if metrics_path.is_file() and vector_path.is_file() and not overwrite:
                    entries.append(
                        {
                            "layer": layer,
                            "concept_id": concept_id,
                            "replicate_id": replicate_id,
                            "vector_file": str(vector_path.relative_to(destination)),
                            "vector_sha256": sha256_file(vector_path),
                            "status": "already_complete",
                        }
                    )
                    continue
                weights = _bootstrap_group_weights(fit_groups, fit_labels, seed=seed)
                fitted = fit_fixed_logistic_direction(
                    fit_activations,
                    fit_labels,
                    C=chosen_c,
                    sample_weight=weights,
                    positive_label=1,
                    standardize=standardize,
                    class_weight=class_weight,
                    random_state=seed,
                    max_iter=max_iter,
                    solver=str(primary.get("solver", "liblinear")),
                )
                _atomic_save_npy(vector_path, fitted.coef_raw)
                vector_sha = sha256_file(vector_path)
                payload = {
                    "schema_version": 1,
                    "workflow": "fixed_c_group_bootstrap_probe_v1",
                    "layer": layer,
                    "concept_id": concept_id,
                    "concept_name": artifact.concept_names[concept_id],
                    "replicate_id": replicate_id,
                    "bootstrap_seed": seed,
                    "bootstrap_unit": "group",
                    "bootstrap_stratified_by_label": True,
                    "fit_split": "train+validation",
                    "test_accessed": False,
                    "row_manifest_path": (
                        None if row_manifest_path is None else str(row_manifest_path)
                    ),
                    "row_manifest_sha256": (
                        None if row_manifest_path is None else sha256_file(row_manifest_path)
                    ),
                    "chosen_C_from_primary_train_cv": chosen_c,
                    "standardize": standardize,
                    "class_weight": class_weight,
                    "solver": str(primary.get("solver", "liblinear")),
                    "activation_artifact_hash": artifact.artifact_hash,
                    "primary_probe_metrics": str(primary_path),
                    "positive_group_draws": int(weights[fit_labels == 1].sum()),
                    "negative_group_draws": int(weights[fit_labels == 0].sum()),
                    "vector_file": vector_path.name,
                    "vector_sha256": vector_sha,
                    "coordinate": "resid_post",
                    "dimension": int(fitted.coef_raw.size),
                    "dtype": "float64",
                }
                atomic_write_json(metrics_path, payload)
                entries.append(
                    {
                        "layer": layer,
                        "concept_id": concept_id,
                        "replicate_id": replicate_id,
                        "vector_file": str(vector_path.relative_to(destination)),
                        "vector_sha256": vector_sha,
                        "status": "completed",
                    }
                )

    manifest = {
        "schema_version": 1,
        "workflow": "fixed_c_group_bootstrap_probe_v1",
        "activation_artifact": str(artifact.path),
        "activation_artifact_hash": artifact.artifact_hash,
        "primary_probes": str(probes),
        "primary_probes_manifest_sha256": sha256_file(probes / "manifest.json"),
        "layers": list(selected_layers),
        "concept_ids": list(selected_concepts),
        "replicate_ids": [f"bootstrap_{seed}" for seed in selected_seeds],
        "seeds": list(selected_seeds),
        "test_accessed": False,
        "row_manifest_path": (None if row_manifest_path is None else str(row_manifest_path)),
        "row_manifest_sha256": (
            None if row_manifest_path is None else sha256_file(row_manifest_path)
        ),
        "entries": entries,
    }
    destination.mkdir(parents=True, exist_ok=True)
    layer_slug = "-".join(f"{layer:02d}" for layer in selected_layers)
    atomic_write_json(destination / "manifests" / f"layers_{layer_slug}.json", manifest)
    return {
        "output": str(destination),
        "completed": sum(entry["status"] == "completed" for entry in entries),
        "skipped": sum(entry["status"] == "already_complete" for entry in entries),
        "entries": len(entries),
    }


def rebuild_probe_replicate_manifest(
    output_dir: str | Path,
    *,
    activation_artifact: str | Path,
    primary_probes: str | Path,
    expected_entries: int | None = None,
) -> dict[str, object]:
    """Scan bootstrap metrics and atomically write the shared manifest."""

    destination = Path(output_dir)
    entries: list[dict[str, object]] = []
    identities: set[tuple[int, str, str]] = set()
    activation_hashes: set[str] = set()
    for metrics_path in sorted(destination.rglob("bootstrap_*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("workflow") != "fixed_c_group_bootstrap_probe_v1":
            raise ProbeReplicateError(f"unsupported replicate metrics: {metrics_path}")
        identity = (
            int(payload["layer"]),
            str(payload["concept_id"]),
            str(payload["replicate_id"]),
        )
        if identity in identities:
            raise ProbeReplicateError(
                f"duplicate bootstrap probe identity at {metrics_path}: {identity}"
            )
        identities.add(identity)
        vector_path = metrics_path.parent / str(payload["vector_file"])
        observed_hash = sha256_file(vector_path)
        if observed_hash != payload["vector_sha256"]:
            raise ProbeReplicateError(f"bootstrap vector SHA-256 mismatch: {vector_path}")
        activation_hashes.add(str(payload["activation_artifact_hash"]))
        entries.append(
            {
                "layer": identity[0],
                "concept_id": identity[1],
                "replicate_id": identity[2],
                "vector_file": str(vector_path.relative_to(destination)),
                "vector_sha256": observed_hash,
            }
        )
    if len(activation_hashes) > 1:
        raise ProbeReplicateError(f"mixed activation artifact hashes: {sorted(activation_hashes)}")
    observed = len(entries)
    if expected_entries is not None and observed > expected_entries:
        raise ProbeReplicateError(
            f"found {observed} replicates but expected at most {expected_entries}"
        )
    missing = None if expected_entries is None else max(0, expected_entries - observed)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "workflow": "fixed_c_group_bootstrap_probe_v1",
        "activation_artifact": str(activation_artifact),
        "activation_artifact_hash": (next(iter(activation_hashes)) if activation_hashes else None),
        "primary_probes": str(primary_probes),
        "primary_probes_manifest_sha256": sha256_file(Path(primary_probes) / "manifest.json"),
        "observed_entries": observed,
        "expected_entries": expected_entries,
        "missing_entries": missing,
        "complete": missing == 0 if missing is not None else None,
        "test_accessed": False,
        "entries": entries,
    }
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination / "manifest.json", manifest)
    return manifest


def load_bootstrap_probe_vectors(
    output_dir: str | Path,
    *,
    layer: int,
    replicate_id: str,
) -> dict[str, tuple[np.ndarray, str, dict[str, object]]]:
    """Load one bootstrap replicate across concepts with hash validation."""

    destination = Path(output_dir)
    vectors: dict[str, tuple[np.ndarray, str, dict[str, object]]] = {}
    pattern = f"layer_{layer:02d}/concept_*/{replicate_id}/metrics.json"
    for metrics_path in sorted(destination.glob(pattern)):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        concept_id = str(payload["concept_id"])
        vector_path = metrics_path.parent / str(payload["vector_file"])
        vector_hash = sha256_file(vector_path)
        if vector_hash != payload["vector_sha256"]:
            raise ProbeReplicateError(f"bootstrap vector SHA-256 mismatch: {vector_path}")
        vector = np.load(vector_path, allow_pickle=False)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ProbeReplicateError(
                f"bootstrap vector must be finite and one-dimensional: {vector_path}"
            )
        vectors[concept_id] = (vector, vector_hash, payload)
    if not vectors:
        raise ProbeReplicateError(
            f"no bootstrap probes for layer={layer}, replicate={replicate_id}"
        )
    return vectors
