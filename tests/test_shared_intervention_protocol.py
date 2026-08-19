from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jlens_workspace.artifacts import sha256_file
from jlens_workspace.concept_intervention.shared_protocol import (
    SharedProtocolError,
    _grouped_raptor_c_scores,
    deterministic_balanced_indices,
    load_balanced_indices,
    load_selected_layers,
    validate_shared_identity,
)


def _rows(groups: list[tuple[str, int]]) -> tuple[np.ndarray, list[dict[str, str]]]:
    labels = np.asarray([label for _group, label in groups], dtype=np.int8)
    rows = [
        {"split": "train", "group_id": group, "text": f"row-{index}"}
        for index, (group, _label) in enumerate(groups)
    ]
    return labels, rows


def test_balanced_rows_are_deterministic_exact_and_group_atomic() -> None:
    labels, rows = _rows(
        [
            ("negative-pair", 0),
            ("negative-pair", 0),
            ("negative-single", 0),
            ("positive-pair", 1),
            ("positive-pair", 1),
            ("positive-extra-a", 1),
            ("positive-extra-b", 1),
        ]
    )
    first = deterministic_balanced_indices(
        labels, rows, concept_id="concept:a", split="train", seed=42
    )
    second = deterministic_balanced_indices(
        labels, rows, concept_id="concept:a", split="train", seed=42
    )

    np.testing.assert_array_equal(first, second)
    assert int(np.sum(labels[first] == 0)) == int(np.sum(labels[first] == 1))
    selected = set(first.tolist())
    for group in {row["group_id"] for row in rows}:
        group_rows = {
            index for index, row in enumerate(rows) if row["group_id"] == group
        }
        assert not selected.intersection(group_rows) or group_rows <= selected


def test_balanced_rows_reject_group_that_spans_labels() -> None:
    labels, rows = _rows([("mixed", 0), ("mixed", 1)])
    with pytest.raises(SharedProtocolError, match="spans both labels"):
        deterministic_balanced_indices(
            labels, rows, concept_id="concept:a", split="train", seed=42
        )


def test_grouped_c_tuning_calls_pinned_raptor_sweep_inside_each_fold() -> None:
    class FakeRaptorTuning:
        RAPTOR_C_GRID = np.asarray([0.1, 1.0])

        def __init__(self) -> None:
            self.calls = 0
            self.accuracy_score = (
                lambda truth, prediction: float(np.mean(truth == prediction))
            )

        def tune_raptor_c(
            self,
            _train_x: np.ndarray,
            _train_y: np.ndarray,
            _validation_x: np.ndarray,
            validation_y: np.ndarray,
            *,
            max_iter: int,
        ) -> tuple[float, float]:
            assert max_iter == 100
            self.calls += 1
            scores = [
                self.accuracy_score(
                    validation_y, np.zeros_like(validation_y)
                ),
                self.accuracy_score(validation_y, validation_y),
            ]
            best = int(np.argmax(scores))
            return float(self.RAPTOR_C_GRID[best]), float(scores[best])

    upstream = FakeRaptorTuning()
    labels = np.asarray([0, 0, 1, 1] * 3, dtype=np.int8)
    groups = np.asarray([f"group-{index}" for index in range(labels.size)])
    activations = np.column_stack((labels, np.arange(labels.size)))
    scores = _grouped_raptor_c_scores(
        activations,
        labels,
        groups,
        c_grid=[0.1, 1.0],
        cv_folds=3,
        max_iter=100,
        seed=42,
        upstream=upstream,
    )

    assert upstream.calls == 3
    assert scores.shape == (3, 2)
    np.testing.assert_allclose(scores[:, 1], 1.0)


def test_shared_artifact_loaders_fail_closed_on_identity(tmp_path: Path) -> None:
    row_manifest = tmp_path / "row_manifest.json"
    row_manifest.write_text(
        json.dumps(
            {
                "concepts": {
                    "concept:a": {
                        "train": {
                            "rows": [
                                {"activation_index": 3},
                                {"activation_index": 7},
                            ]
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "layer_selection.json"
    selection.write_text(
        json.dumps(
            {
                "method": "raptor_validation_accuracy",
                "selected_layer_count": 2,
                "activation_artifact_hash": "activation-hash",
                "row_manifest_sha256": sha256_file(row_manifest),
                "concepts": [
                    {
                        "concept_id": "concept:a",
                        "selected_layers": [3, 7],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert load_selected_layers(selection, "concept:a") == (3, 7)
    np.testing.assert_array_equal(
        load_balanced_indices(row_manifest, "concept:a", "train"), [3, 7]
    )
    validate_shared_identity(
        selection,
        row_manifest_path=row_manifest,
        activation_artifact_hash="activation-hash",
    )
    with pytest.raises(SharedProtocolError, match="activation artifact hash"):
        validate_shared_identity(
            selection,
            row_manifest_path=row_manifest,
            activation_artifact_hash="other",
        )
