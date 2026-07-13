from __future__ import annotations

import numpy as np
import pytest

from jlens_workspace.concepts import (
    activation_covariance,
    chunked_cosine_topk,
    chunked_token_j_topk,
    materialize_token_j_rows,
    sparse_nonnegative_decomposition,
    top_r_basis_concept_coverage,
)


def test_activation_covariance_matches_numpy_reference() -> None:
    activations = np.array(
        [[1.0, 2.0, -1.0], [3.0, 0.0, 2.0], [2.0, 4.0, 1.0], [0.0, 1.0, 3.0]]
    )
    observed = activation_covariance(activations, ridge=0.25)
    expected = np.cov(activations, rowvar=False) + 0.25 * np.eye(3)
    np.testing.assert_allclose(observed, expected, rtol=1e-14, atol=1e-14)


def test_chunked_token_j_search_matches_materialised_rows() -> None:
    unembedding = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [0.2, -0.5]]
    )
    jacobian = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, -1.0]])
    token_ids = np.array([101, 205, 309, 412, 518])
    query = unembedding[2] @ jacobian

    chunked = chunked_token_j_topk(
        query,
        unembedding,
        jacobian,
        k=4,
        chunk_size=2,
        token_ids=token_ids,
    )
    materialised = chunked_cosine_topk(
        query,
        unembedding @ jacobian,
        k=4,
        chunk_size=3,
        row_ids=token_ids,
    )

    assert chunked.indices[0] == 309
    assert chunked.scores[0] == pytest.approx(1.0)
    np.testing.assert_array_equal(chunked.indices, materialised.indices)
    np.testing.assert_allclose(chunked.scores, materialised.scores, atol=1e-15)
    np.testing.assert_allclose(
        materialize_token_j_rows(unembedding, jacobian, np.array([2, 4])),
        (unembedding @ jacobian)[[2, 4]],
    )


def test_covariance_cosine_changes_ranking_and_ties_use_lower_id() -> None:
    query = np.array([1.0, 1.0])
    rows = np.array([[0.0, 1.0], [1.0, 0.0]])

    euclidean = chunked_cosine_topk(query, rows, k=2, chunk_size=1)
    covariance = chunked_cosine_topk(
        query, rows, k=2, chunk_size=1, covariance=np.array([100.0, 1.0])
    )

    assert euclidean.indices.tolist() == [0, 1]
    assert covariance.indices.tolist() == [1, 0]
    assert covariance.metric == "covariance_cosine"


def test_absolute_topk_returns_signed_scores() -> None:
    result = chunked_cosine_topk(
        np.array([1.0, 0.0]),
        np.array([[0.1, 1.0], [-1.0, 0.0], [0.8, 0.0]]),
        k=2,
        chunk_size=1,
        absolute=True,
    )
    assert result.indices.tolist() == [1, 2]
    assert result.scores[0] == pytest.approx(-1.0)
    assert result.scores[1] == pytest.approx(1.0)


def test_sparse_nonnegative_decomposition_recovers_exact_j_component() -> None:
    rows = np.eye(4)
    target = np.array([2.0, 0.0, 0.5, 0.0])
    result = sparse_nonnegative_decomposition(
        target,
        rows,
        candidate_indices=np.array([10, 11, 12, 13]),
        max_nonzero=2,
    )

    np.testing.assert_allclose(result.coefficients, [2.0, 0.0, 0.5, 0.0])
    assert result.selected_candidate_indices.tolist() == [10, 12]
    np.testing.assert_allclose(result.j_component, target)
    np.testing.assert_allclose(result.non_j_component, np.zeros(4), atol=1e-14)
    assert result.relative_residual_norm == pytest.approx(0.0, abs=1e-14)
    assert result.explained_energy == pytest.approx(1.0)


def test_decomposition_exposes_non_j_residual() -> None:
    rows = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = np.array([2.0, 0.0, 3.0])
    result = sparse_nonnegative_decomposition(target, rows)

    np.testing.assert_allclose(result.j_component, [2.0, 0.0, 0.0])
    np.testing.assert_allclose(result.non_j_component, [0.0, 0.0, 3.0])
    assert result.residual_norm == pytest.approx(3.0)
    assert result.explained_energy == pytest.approx(4.0 / 13.0)


def test_top_r_basis_coverage_uses_the_span_not_basis_norms() -> None:
    concepts = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    non_unit_basis = np.array([[4.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 8.0]])

    result = top_r_basis_concept_coverage(concepts, non_unit_basis, top_r=2)

    assert result.basis_rank == 2
    np.testing.assert_allclose(result.per_concept, [1.0, 0.0, 0.0], atol=1e-15)
    assert result.mean_coverage == pytest.approx(1.0 / 3.0)


def test_alignment_accepts_torch_and_returns_numpy() -> None:
    torch = pytest.importorskip("torch")
    unembedding = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    jacobian = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    query = unembedding[2] @ jacobian

    result = chunked_token_j_topk(
        query, unembedding, jacobian, k=2, chunk_size=1
    )

    assert isinstance(result.indices, np.ndarray)
    assert isinstance(result.scores, np.ndarray)
    assert result.indices[0] == 2
