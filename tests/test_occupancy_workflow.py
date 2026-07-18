from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from jlens_workspace.config import ExperimentConfig
from jlens_workspace.pursuit import DenseDictionary
from jlens_workspace.workflows.occupancy import (
    ConceptTarget,
    OccupancyWorkflowError,
    load_combo_metrics,
    run_concept_occupancy,
    verify_lens_artifact_sha256,
)

LENS_SHA = "cdb356a078f28f5cfc9bc2f78ac90988abb78be9b05f24624b7f7b3bb5f29446"


def _atoms(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    return np.vstack([q.T * rng.uniform(0.5, 2.0, size=(12, 1)),
                      rng.normal(size=(6, 12)) * 0.8])


def _targets(atoms: np.ndarray) -> list[ConceptTarget]:
    vector = 1.2 * atoms[3] + 0.4 * atoms[7]
    return [
        ConceptTarget(
            layer=8,
            concept_id="goemotions:gratitude",
            vector=vector,
            vector_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),
        )
    ]


def _run(tmp_path: Path, atoms: np.ndarray, **overrides: object) -> dict[str, object]:
    parameters: dict[str, object] = {
        "output_dir": tmp_path,
        "dictionary_factory": lambda layer, convention: DenseDictionary(atoms),
        "targets": _targets(atoms),
        "conventions": ["rmsnorm_weighted"],
        "selection_modes": ["positive_cosine"],
        "signs": ["+", "-"],
        "k_max": 8,
        "report_grid": [1, 2, 4, 8],
        "random_seeds": [101, 202, 303],
        "chunk_size": 5,
    }
    parameters.update(overrides)
    return run_concept_occupancy(**parameters)  # type: ignore[arg-type]


def test_workflow_writes_complete_combo_artifacts(tmp_path: Path) -> None:
    atoms = _atoms()
    summary = _run(tmp_path, atoms)
    assert summary["completed"] == 2  # +w and -w
    combo = (
        tmp_path
        / "occupancy/rmsnorm_weighted/positive_cosine/layer_08"
        / "goemotions%3Agratitude/pos/primary"
    )
    for name in ("errors.npy", "gains.npy", "control_errors.npy", "control_gains.npy",
                 "support_token_ids.npy", "metrics.json"):
        assert (combo / name).is_file(), name
    for grid_k in (1, 2, 4, 8):
        assert (combo / f"w_J_k{grid_k:02d}.npy").is_file()
        assert (combo / f"w_nonJ_k{grid_k:02d}.npy").is_file()

    metrics = load_combo_metrics(combo / "metrics.json")
    assert metrics["method"] == "concept_occupancy_method_v1"
    assert set(metrics["occupancy"]) == {
        "first_nonexceed_v1",
        "consecutive3_nonexceed_v1",
    }
    control_errors = np.load(combo / "control_errors.npy", allow_pickle=False)
    assert control_errors.shape == (3, 9)
    # w_J + w_nonJ must reconstruct the signed target exactly.
    errors = np.load(combo / "errors.npy", allow_pickle=False)
    target = _targets(atoms)[0].vector
    w_j = np.load(combo / "w_J_k08.npy", allow_pickle=False)
    w_nonj = np.load(combo / "w_nonJ_k08.npy", allow_pickle=False)
    np.testing.assert_allclose(w_j + w_nonj, target, rtol=0, atol=1e-12)
    # The frozen error path makes errors[8] valid even under early stopping.
    np.testing.assert_allclose(
        np.linalg.norm(w_nonj) ** 2 / np.linalg.norm(target) ** 2,
        errors[8],
        rtol=1e-10,
    )

    negative = (
        tmp_path
        / "occupancy/rmsnorm_weighted/positive_cosine/layer_08"
        / "goemotions%3Agratitude/neg/primary"
    )
    assert (negative / "metrics.json").is_file()
    negative_metrics = load_combo_metrics(negative / "metrics.json")
    assert negative_metrics["sign"] == "-"
    assert negative_metrics["support_token_ids"] != metrics["support_token_ids"] or (
        negative_metrics["coefficients_per_k"] != metrics["coefficients_per_k"]
    )


def test_resume_skips_and_does_not_change_completed_outputs(tmp_path: Path) -> None:
    atoms = _atoms()
    first = _run(tmp_path, atoms)
    assert first["completed"] == 2 and first["skipped"] == 0
    combo_metrics = sorted(tmp_path.rglob("metrics.json"))
    before = {path: path.read_bytes() for path in combo_metrics}

    second = _run(tmp_path, atoms)
    assert second["completed"] == 0
    assert second["skipped"] == 2
    for path, payload in before.items():
        assert path.read_bytes() == payload, f"{path} changed on resume"


def test_lens_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    lens = tmp_path / "lens.pt"
    lens.write_bytes(b"not-a-real-lens")
    with pytest.raises(OccupancyWorkflowError, match="SHA-256 mismatch"):
        verify_lens_artifact_sha256(lens, LENS_SHA)
    with pytest.raises(OccupancyWorkflowError, match="64-char"):
        verify_lens_artifact_sha256(lens, "abc")


def test_probe_identity_mismatch_fails(tmp_path: Path) -> None:
    from jlens_workspace.cli import _load_probe_vectors

    probes = tmp_path / "probes"
    layer_dir = probes / "layer_08" / "concept_test"
    layer_dir.mkdir(parents=True)
    vector = np.arange(4, dtype=np.float64)
    np.save(layer_dir / "probe_vector.npy", vector, allow_pickle=False)
    vector_sha = hashlib.sha256((layer_dir / "probe_vector.npy").read_bytes()).hexdigest()
    activation = {
        "coordinate": "resid_post",
        "representation": "last_non_padding_token",
        "add_special_tokens": True,
        "manifest": {
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": "correct-revision",
            "tokenizer_id": "Qwen/Qwen3.5-4B",
            "tokenizer_revision": "correct-revision",
            "dataset_source": "google-research-datasets/go_emotions",
            "dataset_revision": "rev",
            "notes": {"force_bos": False, "config_sha256": "irrelevant"},
        },
    }
    (probes / "manifest.json").write_text(
        json.dumps(
            {
                "activation": activation,
                "activation_artifact_hash": "hash",
                "probes": [],
            }
        )
    )
    (layer_dir / "metrics.json").write_text(
        json.dumps(
            {
                "layer": 8,
                "concept_id": "test",
                "artifact_hash": "hash",
                "activation": activation,
                "probe": {
                    "vector_file": "probe_vector.npy",
                    "vector_sha256": vector_sha,
                    "coordinate": "resid_post",
                    "dimension": 4,
                },
            }
        )
    )

    matching_identity = {
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": "correct-revision",
        "force_bos": False,
    }
    vectors = _load_probe_vectors(probes, 8, expected_identity=matching_identity)
    assert set(vectors) == {"test"}

    with pytest.raises(ValueError, match="model_revision"):
        _load_probe_vectors(
            probes,
            8,
            expected_identity={**matching_identity, "model_revision": "WRONG"},
        )
    with pytest.raises(ValueError, match="no concept probes"):
        _load_probe_vectors(probes, 12, expected_identity=matching_identity)


def test_j_space_direction_rejects_occupancy_section() -> None:
    with pytest.raises(Exception, match="occupancy"):
        ExperimentConfig.model_validate(
            {
                "schema_version": 1,
                "direction": "j_space",
                "experiment_name": "bad",
                "output_dir": "out",
                "model": {"model_id": "tiny"},
                "lens": {"source": "local", "path_or_repo": "lens.pt", "layers": [0]},
                "matrix": {"layers": [0]},
                "occupancy": {"layers": [0], "expected_lens_sha256": LENS_SHA},
            }
        )
