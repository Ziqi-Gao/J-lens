from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jlens_workspace.jacobian import (  # noqa: E402
    JLensMetadata,
    ManagedJacobianLens,
    build_effective_unembedding,
)
from jlens_workspace.workflows.matrix import (  # noqa: E402
    MatrixWorkflowOptions,
    run_matrix_layers,
)


class _SyntheticLens:
    def __init__(self, jacobians):
        self.jacobians = jacobians
        self.source_layers = sorted(jacobians)
        self.d_model = next(iter(jacobians.values())).shape[0]
        self.n_prompts = 12


def _managed_lens():
    jacobians = {
        0: torch.diag(torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64)),
        1: torch.tensor(
            [[1.0, 0.2, 0.0], [0.0, 0.75, -0.1], [0.3, 0.0, 1.25]],
            dtype=torch.float64,
        ),
    }
    raw = _SyntheticLens(jacobians)
    metadata = JLensMetadata(
        model_id="synthetic/model",
        model_revision="model-commit",
        tokenizer_id="synthetic/tokenizer",
        tokenizer_revision="tokenizer-commit",
        d_model=3,
        source_layers=(0, 1),
        target_layer=2,
        n_prompts=12,
    )
    return ManagedJacobianLens(raw, metadata)


def test_matrix_workflow_matches_explicit_svd_and_rejects_conflict(tmp_path):
    torch.manual_seed(19)
    unembedding_weight = torch.randn(7, 3, dtype=torch.float64)
    effective = build_effective_unembedding(
        unembedding_weight,
        convention="raw",
        model_id="synthetic/model",
        model_revision="model-commit",
    )
    lens = _managed_lens()
    output_dir = tmp_path / "matrix-output"
    options = MatrixWorkflowOptions(
        centered=True,
        block_size=2,
        compute_device="cpu",
        compute_dtype="float64",
        energy_thresholds=(0.8, 0.95),
    )
    result = run_matrix_layers(lens, effective, [0, 1], output_dir, options)
    assert result.metrics_path == output_dir / "metrics.json"
    assert [output.layer for output in result.layers] == [0, 1]

    layer_output = result.layers[0]
    singular = np.load(layer_output.singular_values_path, allow_pickle=False)
    eigenvalues = np.load(layer_output.eigenvalues_path, allow_pickle=False)
    numerical_rank_basis = np.load(
        layer_output.numerical_rank_basis_path, allow_pickle=False
    )
    explicit = unembedding_weight @ lens.jacobians[0]
    explicit = explicit - explicit.mean(dim=0, keepdim=True)
    expected = torch.linalg.svdvals(explicit).numpy()
    np.testing.assert_allclose(singular[: expected.size], expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(eigenvalues, singular**2, rtol=1e-12, atol=1e-12)
    assert layer_output.numerical_rank == 2
    assert numerical_rank_basis.shape == (3, 2)
    np.testing.assert_allclose(
        numerical_rank_basis.T @ numerical_rank_basis,
        np.eye(2),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        explicit.numpy()
        @ (np.eye(3) - numerical_rank_basis @ numerical_rank_basis.T),
        0.0,
        rtol=1e-10,
        atol=1e-10,
    )

    metrics = json.loads(layer_output.metrics_path.read_text())
    assert metrics["matrix"]["definition"] == "A_l = U_eff J_l"
    assert metrics["matrix"]["convention"] == "raw"
    assert metrics["matrix"]["centered"] is True
    assert metrics["dtypes"]["gram_accumulation"] == "torch.float64"
    assert metrics["operator"]["provenance"]["model_revision"] == "model-commit"
    assert metrics["spectrum"]["eigenvalues_file"] == "eigenvalues.npy"
    assert "Gram matrix" in metrics["spectrum"]["eigenvalues_definition"]
    assert (
        metrics["spectrum"]["numerical_rank_basis_file"]
        == "basis_numerical_rank.npy"
    )
    assert metrics["spectrum"]["numerical_rank_basis_shape"] == [3, 2]
    assert (
        metrics["provenance"]["effective_unembedding_model_identity"]["status"]
        == "verified"
    )
    assert metrics["energy_bases"]["0.8"]["n_components"] >= 1

    root_metrics = json.loads(result.metrics_path.read_text())
    assert root_metrics["layers"][0]["eigenvalues_file"].endswith(
        "eigenvalues.npy"
    )
    assert root_metrics["layers"][0]["numerical_rank_basis_file"].endswith(
        "basis_numerical_rank.npy"
    )
    assert (
        root_metrics["provenance"]["effective_unembedding_model_identity"]["status"]
        == "verified"
    )
    assert (
        result.provenance["effective_unembedding_model_identity"]["verified"]
        is True
    )

    for threshold, basis_path in layer_output.basis_paths.items():
        basis = np.load(basis_path, allow_pickle=False)
        np.testing.assert_allclose(
            basis.T @ basis, np.eye(basis.shape[1]), rtol=1e-10, atol=1e-10
        )
        assert threshold in {0.8, 0.95}

    with pytest.raises(FileExistsError, match="already exists"):
        run_matrix_layers(lens, effective, [0], output_dir, options)


def test_matrix_workflow_rejects_mismatched_unembedding_identity_before_output(
    tmp_path,
):
    weight = torch.randn(5, 3, dtype=torch.float64)
    effective = build_effective_unembedding(
        weight,
        model_id="synthetic/model",
        model_revision="different-commit",
    )
    output_dir = tmp_path / "mismatch"
    with pytest.raises(ValueError, match="model_revision"):
        run_matrix_layers(_managed_lens(), effective, [0], output_dir)
    assert not output_dir.exists()


def test_matrix_workflow_marks_missing_unembedding_identity_unverified(tmp_path):
    effective = build_effective_unembedding(
        torch.randn(5, 3, dtype=torch.float64), convention="raw"
    )
    result = run_matrix_layers(
        _managed_lens(),
        effective,
        [0],
        tmp_path / "unverified",
        MatrixWorkflowOptions(compute_device="cpu", compute_dtype="float64"),
    )
    provenance = result.provenance["effective_unembedding_model_identity"]
    assert provenance["status"] == "unverified"
    assert provenance["verified"] is False
    assert provenance["missing_fields"] == ["model_id", "model_revision"]
    root_metrics = json.loads(result.metrics_path.read_text())
    layer_metrics = json.loads(result.layers[0].metrics_path.read_text())
    assert (
        root_metrics["provenance"]["effective_unembedding_model_identity"]["status"]
        == "unverified"
    )
    assert (
        layer_metrics["provenance"]["effective_unembedding_model_identity"]["status"]
        == "unverified"
    )
