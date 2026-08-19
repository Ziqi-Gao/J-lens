from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jlens_workspace.artifacts import sha256_file
from jlens_workspace.workflows.probe_replicates import (
    _bootstrap_group_weights,
    load_bootstrap_probe_vectors,
    rebuild_probe_replicate_manifest,
)


def test_group_bootstrap_is_stratified_deterministic_and_group_constant() -> None:
    groups = np.repeat(np.asarray(["a", "b", "c", "d"], dtype=object), 3)
    labels = np.repeat(np.asarray([0, 0, 1, 1]), 3)
    first = _bootstrap_group_weights(groups, labels, seed=1101)
    second = _bootstrap_group_weights(groups, labels, seed=1101)

    np.testing.assert_array_equal(first, second)
    for group in np.unique(groups):
        assert np.unique(first[groups == group]).size == 1
    assert sum(first[::3][labels[::3] == 0]) == 2
    assert sum(first[::3][labels[::3] == 1]) == 2


def test_replicate_manifest_rebuild_validates_and_loads_vectors(
    tmp_path: Path,
) -> None:
    output = tmp_path / "replicates"
    primary = tmp_path / "primary"
    primary.mkdir()
    (primary / "manifest.json").write_text("{}\n", encoding="utf-8")

    for concept_index, concept_id in enumerate(("a", "b")):
        directory = (
            output
            / "layer_08"
            / f"concept_{concept_id}"
            / "bootstrap_1101"
        )
        directory.mkdir(parents=True)
        vector = np.asarray([1.0, concept_index + 2.0], dtype=np.float64)
        np.save(directory / "probe_vector.npy", vector, allow_pickle=False)
        payload = {
            "schema_version": 1,
            "workflow": "fixed_c_group_bootstrap_probe_v1",
            "layer": 8,
            "concept_id": concept_id,
            "replicate_id": "bootstrap_1101",
            "activation_artifact_hash": "activation-hash",
            "vector_file": "probe_vector.npy",
            "vector_sha256": sha256_file(directory / "probe_vector.npy"),
            "dimension": 2,
        }
        (directory / "metrics.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    manifest = rebuild_probe_replicate_manifest(
        output,
        activation_artifact=tmp_path / "activations",
        primary_probes=primary,
        expected_entries=2,
    )
    assert manifest["complete"] is True
    assert manifest["observed_entries"] == 2

    loaded = load_bootstrap_probe_vectors(
        output, layer=8, replicate_id="bootstrap_1101"
    )
    assert set(loaded) == {"a", "b"}
    np.testing.assert_array_equal(loaded["a"][0], np.asarray([1.0, 2.0]))
