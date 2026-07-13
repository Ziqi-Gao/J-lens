from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from jlens_workspace.jacobian import build_effective_unembedding  # noqa: E402
from jlens_workspace.matrix import (  # noqa: E402
    TokenFrameOperator,
    basis_coverage,
    decompose_gram,
    gram_energy_coverage,
    minimum_energy_basis,
    principal_angles,
    streaming_gram,
)


def _operator(*, block_size=3):
    torch.manual_seed(11)
    unembedding = torch.randn(9, 4, dtype=torch.float64)
    jacobian = torch.randn(4, 4, dtype=torch.float64)
    effective = build_effective_unembedding(unembedding, convention="raw")
    operator = TokenFrameOperator(
        jacobian,
        effective,
        layer=2,
        block_size=block_size,
        compute_device="cpu",
        compute_dtype=torch.float64,
    )
    return operator, unembedding @ jacobian


def test_operator_streams_rows_equal_to_explicit_matrix():
    operator, explicit = _operator(block_size=2)
    blocks = list(operator.iter_rows())
    assert [(block.start, block.stop) for block in blocks] == [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
        (8, 9),
    ]
    streamed = torch.cat([block.rows for block in blocks])
    torch.testing.assert_close(streamed, explicit, rtol=1e-12, atol=1e-12)
    assert operator.metadata.convention == "raw"
    assert operator.metadata.layer == 2


@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("row_normalized", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_streaming_gram_matches_explicit_weighted_pca(
    centered, row_normalized, weighted
):
    operator, explicit = _operator(block_size=2)
    weights = torch.linspace(0.25, 2.25, explicit.shape[0], dtype=torch.float64)
    rows = explicit.clone()
    if row_normalized:
        rows = rows / torch.linalg.vector_norm(rows, dim=1, keepdim=True)
    active_weights = weights if weighted else torch.ones_like(weights)
    mean = (rows * active_weights[:, None]).sum(0) / active_weights.sum()
    if centered:
        rows = rows - mean
    weighted_rows = rows * torch.sqrt(active_weights)[:, None]
    expected_gram = weighted_rows.T @ weighted_rows

    result = streaming_gram(
        operator,
        centered=centered,
        row_normalized=row_normalized,
        row_weights=weights if weighted else None,
    )
    torch.testing.assert_close(result.gram, expected_gram, rtol=1e-11, atol=1e-11)
    if centered:
        torch.testing.assert_close(result.mean, mean, rtol=1e-12, atol=1e-12)
    else:
        assert result.mean is None

    spectrum = decompose_gram(result)
    expected_singular = torch.linalg.svdvals(weighted_rows)
    torch.testing.assert_close(
        spectrum.singular_values[: expected_singular.numel()],
        expected_singular,
        rtol=1e-10,
        atol=1e-10,
    )
    assert spectrum.numerical_rank == int(torch.linalg.matrix_rank(weighted_rows))


def test_spectrum_energy_basis_and_rank_statistics():
    diagonal = torch.diag(torch.tensor([4.0, 2.0, 0.0], dtype=torch.float64))
    operator = TokenFrameOperator(
        torch.eye(3, dtype=torch.float64),
        build_effective_unembedding(diagonal),
        compute_dtype=torch.float64,
        block_size=1,
    )
    gram = streaming_gram(operator)
    spectrum = decompose_gram(gram)
    torch.testing.assert_close(
        spectrum.singular_values,
        torch.tensor([4.0, 2.0, 0.0], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert spectrum.numerical_rank == 2
    assert spectrum.stable_rank == pytest.approx(1.25)
    assert spectrum.participation_ratio == pytest.approx(400.0 / 272.0)
    basis = minimum_energy_basis(spectrum, 0.79)
    assert basis.n_components == 1
    assert basis.captured_energy == pytest.approx(0.8)
    basis = spectrum.minimum_energy_basis(0.81)
    assert basis.n_components == 2
    torch.testing.assert_close(
        basis.basis.T @ basis.basis, torch.eye(2, dtype=torch.float64)
    )


def test_row_normalization_zero_policy_is_explicit():
    unembedding = torch.tensor(
        [[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]], dtype=torch.float64
    )
    operator = TokenFrameOperator(
        torch.eye(2, dtype=torch.float64),
        build_effective_unembedding(unembedding),
        compute_dtype=torch.float64,
    )
    with pytest.raises(RuntimeError, match="zero rows"):
        streaming_gram(operator, row_normalized=True)
    result = streaming_gram(
        operator, row_normalized=True, zero_row_policy="skip"
    )
    assert result.zero_rows == 1
    torch.testing.assert_close(result.gram, torch.eye(2, dtype=torch.float64))


def test_principal_angles_directional_coverage_and_gram_energy():
    e = torch.eye(4, dtype=torch.float64)
    a = e[:, :2]
    b = e[:, :3]
    angles = principal_angles(a, b)
    torch.testing.assert_close(angles, torch.zeros(2, dtype=torch.float64))
    coverage = basis_coverage(a, b)
    assert coverage.a_covered_by_b == pytest.approx(1.0)
    assert coverage.b_covered_by_a == pytest.approx(2.0 / 3.0)
    assert coverage.shared_dimension == pytest.approx(2.0)
    gram = torch.diag(torch.tensor([5.0, 3.0, 1.0, 1.0], dtype=torch.float64))
    assert gram_energy_coverage(gram, a) == pytest.approx(0.8)
