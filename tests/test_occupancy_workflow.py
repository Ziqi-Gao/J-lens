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
    rebuild_occupancy_index,
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
    primary = metrics["primary_occupancy"]
    assert primary["k_selected_before_crossing"] == max(
        0, (primary["crossing_k"] or 9) - 1
    )
    assert (combo / primary["w_j_file"]).is_file()
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


def test_v2_index_is_complete_and_preserves_solver_identity(tmp_path: Path) -> None:
    atoms = _atoms()
    _run(
        tmp_path,
        atoms,
        method="concept_occupancy_method_v2",
        solver_method="nonnegative_gradient_pursuit_v2",
    )
    index = rebuild_occupancy_index(
        tmp_path,
        method="concept_occupancy_method_v2",
        expected_combinations=2,
    )
    assert index["complete"] is True
    assert index["observed_combinations"] == 2
    assert len(index["entries"]) == 2
    metrics = load_combo_metrics(
        tmp_path
        / "occupancy/rmsnorm_weighted/positive_cosine/layer_08"
        / "goemotions%3Agratitude/pos/primary/metrics.json"
    )
    assert metrics["method"] == "concept_occupancy_method_v2"
    assert metrics["solver_method"] == "nonnegative_gradient_pursuit_v2"


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


def test_control_subsample_is_deterministic_and_sized() -> None:
    from jlens_workspace.workflows.occupancy import subsample_control_norms

    norms = np.linspace(0.1, 3.0, 40)
    half_a = subsample_control_norms(norms, 0.5)
    half_b = subsample_control_norms(norms, 0.5)
    np.testing.assert_array_equal(half_a, half_b)
    assert half_a.size == 20
    # Values are a genuine sub-multiset of the real norms.
    assert set(np.round(half_a, 12)).issubset(set(np.round(norms, 12)))
    quarter = subsample_control_norms(norms, 0.25)
    assert quarter.size == 10 and not np.array_equal(quarter, half_a[:10])
    # Fraction 1.0 is the identity (primary control unchanged).
    assert subsample_control_norms(norms, 1.0) is norms


def test_single_fraction_keeps_legacy_schema(tmp_path: Path) -> None:
    atoms = _atoms()
    _run(tmp_path, atoms)  # default control_atom_fractions=(1.0,)
    combo = (
        tmp_path
        / "occupancy/rmsnorm_weighted/positive_cosine/layer_08"
        / "goemotions%3Agratitude/pos/primary"
    )
    metrics = json.loads((combo / "metrics.json").read_text())
    assert "occupancy_by_control_fraction" not in metrics
    assert "fractions" not in metrics["random_controls"]
    assert not list(combo.glob("control_errors_f*.npy"))


def test_multi_fraction_controls_and_occupancy(tmp_path: Path) -> None:
    atoms = _atoms()
    summary = _run(tmp_path, atoms, control_atom_fractions=(1.0, 0.5))
    assert summary["completed"] == 2
    combo = (
        tmp_path
        / "occupancy/rmsnorm_weighted/positive_cosine/layer_08"
        / "goemotions%3Agratitude/pos/primary"
    )
    reduced_errors = np.load(combo / "control_errors_f0p5.npy", allow_pickle=False)
    reduced_gains = np.load(combo / "control_gains_f0p5.npy", allow_pickle=False)
    assert reduced_errors.shape == (3, 9) and reduced_gains.shape == (3, 8)

    metrics = json.loads((combo / "metrics.json").read_text())
    by_fraction = metrics["occupancy_by_control_fraction"]
    assert set(by_fraction) == {"1", "0.5"}
    for entry in by_fraction.values():
        assert set(entry) == {"first_nonexceed_v1", "consecutive3_nonexceed_v1"}
    # Top-level occupancy stays the fraction-1.0 result.
    assert metrics["occupancy"] == by_fraction["1"]
    fractions_meta = metrics["random_controls"]["fractions"]
    assert [f["fraction"] for f in fractions_meta] == [1.0, 0.5]
    assert fractions_meta[1]["method"] == "matched_gaussian_atom_norms_subsampled_v1"
    assert fractions_meta[1]["n_atoms"] == 9  # ceil(0.5 * 18 atoms)
