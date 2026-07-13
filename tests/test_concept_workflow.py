from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import jlens_workspace.workflows.concept as concept_workflow_module
from jlens_workspace.artifacts import atomic_write_json, stable_hash
from jlens_workspace.workflows import ConceptWorkflowError, run_concept_probe_workflow


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    (path / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    metadata["example_hash"] = stable_hash(
        json.dumps(row, sort_keys=True) for row in rows
    )
    atomic_write_json(path / "metadata.json", metadata)


def _synthetic_activation_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "activations"
    artifact.mkdir()
    rng = np.random.default_rng(20260712)
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    layer_0: list[np.ndarray] = []
    layer_2: list[np.ndarray] = []
    split_groups = {"train": 3, "validation": 1, "test": 1}
    concepts = (("alpha", "Alpha"), ("beta", "Beta"))

    for concept_number, (concept_id, concept_name) in enumerate(concepts):
        direction_0 = np.eye(4)[concept_number]
        direction_2 = np.eye(4)[concept_number + 2]
        for split, group_count in split_groups.items():
            for label in (0, 1):
                for group_number in range(group_count):
                    row_index = len(rows)
                    group_id = f"{concept_id}-{split}-{label}-{group_number}"
                    rows.append(
                        {
                            "row": row_index,
                            "concept_id": concept_id,
                            "concept_name": concept_name,
                            "label": label,
                            "split": split,
                            "group_id": group_id,
                            "source": "synthetic",
                        }
                    )
                    labels.append(label)
                    sign = 1.0 if label else -1.0
                    layer_0.append(
                        sign * 3.0 * direction_0 + rng.normal(scale=0.08, size=4)
                    )
                    layer_2.append(
                        sign * 2.5 * direction_2 + rng.normal(scale=0.08, size=4)
                    )

    np.save(artifact / "labels.npy", np.asarray(labels, dtype=np.int8), allow_pickle=False)
    np.save(
        artifact / "layer_00.npy", np.asarray(layer_0, dtype=np.float32), allow_pickle=False
    )
    np.save(
        artifact / "layer_02.npy", np.asarray(layer_2, dtype=np.float32), allow_pickle=False
    )
    atomic_write_json(
        artifact / "metadata.json",
        {
            "schema_version": 1,
            "coordinate": "resid_post",
            "representation": "last_non_padding_token",
            "layers": [0, 2],
            "n_examples": len(rows),
            "example_hash": "filled-by-_write_rows",
            "manifest": None,
        },
    )
    _write_rows(artifact, rows)
    return artifact


def _synthetic_shared_activation_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "shared-activations"
    artifact.mkdir()
    rng = np.random.default_rng(20260713)
    rows: list[dict[str, Any]] = []
    labels: list[list[int]] = []
    activations: list[np.ndarray] = []
    split_repeats = {"train": 2, "validation": 1, "test": 1}
    for split, repeats in split_repeats.items():
        for repeat in range(repeats):
            for alpha, beta in ((0, 0), (0, 1), (1, 0), (1, 1)):
                index = len(rows)
                text = f"{split}-{repeat}-{alpha}-{beta}"
                rows.append(
                    {
                        "row": index,
                        "split": split,
                        "group_id": f"shared-{text}",
                        "source": "synthetic",
                        "license": "CC0-1.0",
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    }
                )
                labels.append([alpha, beta])
                activations.append(
                    np.asarray(
                        [
                            3.0 if alpha else -3.0,
                            3.0 if beta else -3.0,
                            0.0,
                            0.0,
                        ]
                    )
                    + rng.normal(scale=0.05, size=4)
                )

    np.save(
        artifact / "labels.npy", np.asarray(labels, dtype=np.int8), allow_pickle=False
    )
    np.save(
        artifact / "layer_00.npy",
        np.asarray(activations, dtype=np.float32),
        allow_pickle=False,
    )
    concepts = [
        {
            "column": 0,
            "concept_id": "alpha",
            "concept_name": "Alpha",
            "definition": "Synthetic alpha concept.",
        },
        {
            "column": 1,
            "concept_id": "beta",
            "concept_name": "Beta",
            "definition": "Synthetic beta concept.",
        },
    ]
    atomic_write_json(
        artifact / "concepts.json", {"schema_version": 1, "concepts": concepts}
    )
    atomic_write_json(
        artifact / "metadata.json",
        {
            "schema_version": 2,
            "coordinate": "resid_post",
            "representation": "last_non_padding_token",
            "layers": [0],
            "n_examples": len(rows),
            "example_hash": "filled-by-_write_rows",
            "manifest": None,
        },
    )
    _write_rows(artifact, rows)
    return artifact


def test_workflow_preserves_selection_refit_and_test_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _synthetic_activation_artifact(tmp_path)
    calls: list[dict[str, Any]] = []
    real_fit = concept_workflow_module.fit_logistic_probe

    def recording_fit(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_heldout: np.ndarray,
        y_heldout: np.ndarray,
        **kwargs: Any,
    ) -> Any:
        calls.append(
            {
                "train_count": X_train.shape[0],
                "heldout_count": X_heldout.shape[0],
                "train_labels": np.asarray(y_train).copy(),
                "heldout_labels": np.asarray(y_heldout).copy(),
                "C_grid": tuple(kwargs["C_grid"]),
                "groups": np.asarray(kwargs["groups"]).copy(),
            }
        )
        return real_fit(X_train, y_train, X_heldout, y_heldout, **kwargs)

    monkeypatch.setattr(concept_workflow_module, "fit_logistic_probe", recording_fit)
    output_dir = tmp_path / "probes"
    result = run_concept_probe_workflow(
        artifact,
        output_dir,
        C_grid=(0.01, 0.1),
        cv_splits=3,
        random_state=19,
    )

    assert len(result.probes) == 4
    assert len(calls) == 8
    for probe_number, probe in enumerate(result.probes):
        diagnostic_call = calls[2 * probe_number]
        final_call = calls[2 * probe_number + 1]
        assert diagnostic_call["train_count"] == 6
        assert diagnostic_call["heldout_count"] == 2
        assert diagnostic_call["C_grid"] == (0.01, 0.1)
        assert all("-train-" in group for group in diagnostic_call["groups"])
        assert final_call["train_count"] == 8
        assert final_call["heldout_count"] == 2
        assert final_call["C_grid"] == (probe.chosen_C,)
        assert any("-validation-" in group for group in final_call["groups"])
        assert all("-test-" not in group for group in final_call["groups"])

        vector = np.load(probe.probe_vector_path, allow_pickle=False)
        assert vector.shape == (4,)
        assert vector.dtype == np.float64
        payload = json.loads(probe.metrics_path.read_text(encoding="utf-8"))
        assert payload["artifact_hash"] == result.artifact_hash
        assert payload["layer"] == probe.layer
        assert payload["concept_id"] == probe.concept_id
        assert payload["chosen_C"] == probe.chosen_C
        assert len(payload["cv_scores"]) == 2
        assert payload["counts"]["train"] == {
            "groups": 6,
            "label_0": 3,
            "label_1": 3,
            "total": 6,
        }
        assert payload["counts"]["validation"]["total"] == 2
        assert payload["counts"]["test"]["total"] == 2
        assert payload["validation"]["roc_auc"] == pytest.approx(1.0)
        assert payload["test"]["roc_auc"] == pytest.approx(1.0)
        assert payload["probe"]["coordinate"] == "resid_post"
        assert payload["probe"]["fit_split"] == "train+validation"

    assert not list(output_dir.rglob("*.pkl"))
    assert not list(output_dir.rglob("*.pickle"))


def test_workflow_rejects_labels_rows_mismatch(tmp_path: Path) -> None:
    artifact = _synthetic_activation_artifact(tmp_path)
    labels = np.load(artifact / "labels.npy", allow_pickle=False)
    labels[0] = 1 - labels[0]
    np.save(artifact / "labels.npy", labels, allow_pickle=False)

    with pytest.raises(ConceptWorkflowError, match="labels/rows mismatch"):
        run_concept_probe_workflow(artifact, tmp_path / "output")


def test_workflow_rejects_group_crossing_splits(tmp_path: Path) -> None:
    artifact = _synthetic_activation_artifact(tmp_path)
    rows = [
        json.loads(line)
        for line in (artifact / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_group = next(
        row["group_id"]
        for row in rows
        if row["concept_id"] == "alpha" and row["split"] == "train" and row["label"] == 0
    )
    test_row = next(
        row
        for row in rows
        if row["concept_id"] == "alpha" and row["split"] == "test" and row["label"] == 0
    )
    test_row["group_id"] = train_group
    _write_rows(artifact, rows)

    with pytest.raises(ConceptWorkflowError, match="crosses splits"):
        run_concept_probe_workflow(artifact, tmp_path / "output")


def test_workflow_fits_columns_from_shared_source_artifact(tmp_path: Path) -> None:
    artifact = _synthetic_shared_activation_artifact(tmp_path)
    result = run_concept_probe_workflow(
        artifact,
        tmp_path / "shared-probes",
        C_grid=(0.1,),
        cv_splits=3,
        random_state=7,
    )

    assert {(probe.layer, probe.concept_id) for probe in result.probes} == {
        (0, "alpha"),
        (0, "beta"),
    }
    for probe in result.probes:
        payload = json.loads(probe.metrics_path.read_text(encoding="utf-8"))
        assert payload["counts"]["train"]["total"] == 8
        assert payload["counts"]["train"]["label_0"] == 4
        assert payload["counts"]["train"]["label_1"] == 4
        assert payload["validation"]["average_precision"] == pytest.approx(1.0)
        assert payload["test"]["roc_auc"] == pytest.approx(1.0)
