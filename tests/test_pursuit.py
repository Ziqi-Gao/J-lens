from __future__ import annotations

import numpy as np
import pytest

from jlens_workspace.concepts import sparse_nonnegative_decomposition
from jlens_workspace.pursuit import (
    DenseDictionary,
    DictionaryError,
    MatchedNormRandomDictionary,
    PursuitSolverError,
    absolute_threshold_ks,
    crossing_k,
    k_90_attainable,
    streaming_nonnegative_pursuit,
)


def _orthogonal_dictionary(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    orthogonal = q.T * rng.uniform(0.5, 3.0, size=(16, 1))
    distractors = rng.normal(size=(8, 16)) * 0.9
    return np.vstack([orthogonal, distractors])


def test_known_nonnegative_sparse_target_is_recovered() -> None:
    atoms = _orthogonal_dictionary()
    target = 1.7 * atoms[5] + 0.6 * atoms[11] + 0.25 * atoms[2]
    result = streaming_nonnegative_pursuit(
        DenseDictionary(atoms), target[None, :], k_max=6
    )[0]

    assert set(result.support[:3].tolist()) == {5, 11, 2}
    assert result.errors[3] < 1e-20
    coefficients = dict(
        zip(result.support[:3].tolist(), result.coefficients_at(3), strict=True)
    )
    assert coefficients[5] == pytest.approx(1.7, abs=1e-9)
    assert coefficients[11] == pytest.approx(0.6, abs=1e-9)
    assert coefficients[2] == pytest.approx(0.25, abs=1e-9)


@pytest.mark.parametrize("mode", ["positive_cosine", "raw_positive_dot"])
def test_error_is_monotonically_nonincreasing_and_coefficients_nonnegative(
    mode: str,
) -> None:
    rng = np.random.default_rng(11)
    atoms = rng.normal(size=(40, 8))
    target = rng.normal(size=8)
    result = streaming_nonnegative_pursuit(
        DenseDictionary(atoms), target[None, :], k_max=12, selection_modes=mode
    )[0]

    assert result.errors[0] == 1.0
    assert np.all(np.diff(result.errors) <= 1e-12)
    assert np.all(result.gains >= -1e-12)
    for coefficients in result.coefficients_per_k:
        assert np.all(coefficients >= 0.0)


def test_selection_uses_the_updated_residual_not_initial_ranking() -> None:
    # Atom 0 has the best initial cosine; once it is selected and refit, the
    # residual is dominated by e2, so the correct second pick is atom 2 even
    # though atom 1 outranks it in the INITIAL cosine ordering.
    atoms = np.asarray(
        [
            [1.0, 0.8, 0.0],
            [1.0, 0.4, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    target = np.asarray([1.0, 1.0, 0.0])
    initial_cosines = (atoms @ target) / np.linalg.norm(atoms, axis=1)
    assert np.argsort(-initial_cosines).tolist()[:2] == [0, 1]

    result = streaming_nonnegative_pursuit(
        DenseDictionary(atoms), target[None, :], k_max=2
    )[0]
    assert result.support.tolist() == [0, 2]
    assert result.errors[2] < 1e-24


def test_candidate_pool_approximation_differs_from_exact_mode() -> None:
    # The legacy pool-based path (concepts.sparse_nonnegative_decomposition on
    # the top-2 initial-cosine pool) cannot select atom 2 and therefore cannot
    # reach the exact solver's reconstruction: the two modes are demonstrably
    # different algorithms, so the pool mode may only ever be an ablation.
    atoms = np.asarray(
        [
            [1.0, 0.8, 0.0],
            [1.0, 0.4, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    target = np.asarray([1.0, 1.0, 0.0])
    initial_cosines = (atoms @ target) / np.linalg.norm(atoms, axis=1)
    pool = np.argsort(-initial_cosines)[:2]
    pooled = sparse_nonnegative_decomposition(
        target, atoms[pool], candidate_indices=pool, max_nonzero=2
    )
    exact = streaming_nonnegative_pursuit(
        DenseDictionary(atoms), target[None, :], k_max=2
    )[0]

    assert 2 not in set(pool.tolist())
    assert set(exact.support.tolist()) != set(
        pooled.selected_candidate_indices.tolist()
    )
    assert exact.errors[2] < pooled.relative_residual_norm**2 - 1e-9


def test_matrix_orientation_scores_and_reconstruction() -> None:
    rng = np.random.default_rng(3)
    atoms = rng.normal(size=(9, 5))
    dictionary = DenseDictionary(atoms)
    vector = rng.normal(size=5)

    # A @ w: one score per vocabulary atom.
    np.testing.assert_allclose(dictionary.dots(vector), atoms @ vector)

    # A.T @ alpha: a residual-space reconstruction.
    target = np.abs(rng.normal(size=5))
    result = streaming_nonnegative_pursuit(
        dictionary, target[None, :], k_max=4
    )[0]
    k = int(result.support.size)
    alpha_full = np.zeros(atoms.shape[0])
    alpha_full[result.support] = result.coefficients_at(k)
    reconstruction = atoms.T @ alpha_full
    assert reconstruction.shape == (5,)
    np.testing.assert_allclose(
        np.linalg.norm(target - reconstruction),
        result.residual_norms[k],
        rtol=1e-10,
        atol=1e-12,
    )


def test_centered_dictionaries_are_rejected() -> None:
    atoms = np.eye(4)
    with pytest.raises(DictionaryError, match="centered"):
        DenseDictionary(atoms, centered=True)
    with pytest.raises(DictionaryError, match="centered"):
        MatchedNormRandomDictionary(
            np.ones(4), seed=1, d_model=4, centered=True
        )


def test_tie_break_is_deterministic_lowest_id() -> None:
    atom = np.asarray([1.0, 0.0])
    atoms = np.vstack([np.asarray([[0.2, 0.9]]), atom, np.asarray([[0.1, -0.4]]), atom])
    target = np.asarray([1.0, 0.0])
    result = streaming_nonnegative_pursuit(
        DenseDictionary(atoms), target[None, :], k_max=1
    )[0]
    assert result.support.tolist() == [1]


def test_random_dictionary_is_deterministic_and_norm_matched() -> None:
    norms = np.asarray([0.5, 2.0, 0.0, 3.25, 1.0])
    first = MatchedNormRandomDictionary(norms, seed=202, d_model=6, chunk_size=2)
    second = MatchedNormRandomDictionary(norms, seed=202, d_model=6, chunk_size=2)
    other_seed = MatchedNormRandomDictionary(norms, seed=303, d_model=6, chunk_size=2)

    ids = np.arange(5)
    np.testing.assert_array_equal(first.materialize(ids), second.materialize(ids))
    assert not np.allclose(first.materialize(ids), other_seed.materialize(ids))

    np.testing.assert_allclose(
        np.linalg.norm(first.materialize(ids), axis=1), norms, rtol=1e-12
    )
    residual = np.asarray([1.0, -2.0, 0.5, 0.0, 3.0, -1.0])
    np.testing.assert_allclose(
        first.dots(residual), first.materialize(ids) @ residual, rtol=1e-12
    )


def test_random_dictionary_chunking_does_not_change_atoms() -> None:
    norms = np.linspace(0.5, 2.0, 7)
    coarse = MatchedNormRandomDictionary(norms, seed=101, d_model=4, chunk_size=7)
    # A different chunk size is a DIFFERENT deterministic dictionary version;
    # within one chunk size the atoms must be stable regardless of query order.
    ids_forward = np.arange(7)
    ids_shuffled = np.asarray([6, 0, 3, 1, 5, 2, 4])
    forward = coarse.materialize(ids_forward)
    shuffled = coarse.materialize(ids_shuffled)
    np.testing.assert_array_equal(forward[ids_shuffled], shuffled)


def test_crossing_rules_on_hand_constructed_curves() -> None:
    real = np.asarray([0.5, 0.009, 0.3, 0.004, 0.003, 0.002])
    controls = [np.full(6, 0.01), np.full(6, 0.012), np.full(6, 0.008)]

    rule_a = crossing_k(real, controls, rule="first_nonexceed_v1")
    assert rule_a == {"rule": "first_nonexceed_v1", "k": 2, "right_censored": False}

    # Rule B ignores the isolated dip at k=2 and fires at the start of the
    # first run of three consecutive non-exceedances (k=4,5,6).
    rule_b = crossing_k(real, controls, rule="consecutive3_nonexceed_v1")
    assert rule_b == {
        "rule": "consecutive3_nonexceed_v1",
        "k": 4,
        "right_censored": False,
    }


def test_no_crossing_is_right_censored_not_kmax() -> None:
    real = np.full(8, 0.5)
    controls = [np.full(8, 0.01)] * 5
    for rule in ("first_nonexceed_v1", "consecutive3_nonexceed_v1"):
        outcome = crossing_k(real, controls, rule=rule)
        assert outcome["right_censored"] is True
        assert outcome["k"] is None
        assert outcome["k_max"] == 8


def test_reconstruction_summaries() -> None:
    errors = np.asarray([1.0, 0.6, 0.35, 0.2, 0.15, 0.12, 0.1])
    # Total benefit 0.9; 90% of it (0.81) is first reached at k=4
    # (benefit(3)=0.8 < 0.81 <= benefit(4)=0.85).
    assert k_90_attainable(errors) == 4
    thresholds = dict(absolute_threshold_ks(errors, (0.01, 0.5, 0.85, 0.95)))
    assert thresholds["0.01"] == 1
    assert thresholds["0.5"] == 2
    assert thresholds["0.85"] == 4
    assert thresholds["0.95"] is None


def test_solver_input_validation() -> None:
    dictionary = DenseDictionary(np.eye(3))
    with pytest.raises(PursuitSolverError, match="non-zero"):
        streaming_nonnegative_pursuit(dictionary, np.zeros((1, 3)), k_max=2)
    with pytest.raises(PursuitSolverError, match="selection modes"):
        streaming_nonnegative_pursuit(
            dictionary, np.ones((1, 3)), k_max=2, selection_modes="cosine"
        )
    with pytest.raises(PursuitSolverError, match="k_max"):
        streaming_nonnegative_pursuit(dictionary, np.ones((1, 3)), k_max=0)


def test_early_stop_freezes_curves_when_no_admissible_atom() -> None:
    # Target is orthogonal to the single admissible direction after one pick:
    # with all remaining scores <= 0 the solver freezes and repeats values.
    atoms = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]])
    target = np.asarray([2.0, 3.0])
    result = streaming_nonnegative_pursuit(
        DenseDictionary(atoms), target[None, :], k_max=5
    )[0]
    assert result.support.tolist() == [0]
    assert result.stopped_early_at == 1
    np.testing.assert_allclose(result.errors[1:], result.errors[1])
    np.testing.assert_allclose(result.gains[1:], 0.0)


def test_streaming_token_frame_matches_explicit_matrix() -> None:
    torch = pytest.importorskip("torch")

    from jlens_workspace.jacobian import build_effective_unembedding
    from jlens_workspace.pursuit import TokenFrameDictionary

    torch.manual_seed(5)
    unembedding = torch.randn(11, 6, dtype=torch.float64)
    gamma = torch.tensor([0.5, 2.0, 1.5, 0.75, 1.25, 0.9], dtype=torch.float64)
    jacobian = torch.randn(6, 6, dtype=torch.float64)
    effective = build_effective_unembedding(
        unembedding,
        convention="rmsnorm_weighted",
        norm=type("ToyRMSNorm", (), {"weight": gamma, "variance_epsilon": 1e-6})(),
    )
    explicit = ((unembedding * gamma) @ jacobian).numpy()

    streamed = TokenFrameDictionary(effective, jacobian, chunk_size=3)
    dense = DenseDictionary(explicit)
    np.testing.assert_allclose(streamed.atom_norms(), dense.atom_norms(), rtol=1e-6)

    rng = np.random.default_rng(9)
    residuals = rng.normal(size=(2, 6))
    np.testing.assert_allclose(
        streamed.dots_batch(residuals), dense.dots_batch(residuals), rtol=1e-6
    )
    ids = np.asarray([0, 4, 10])
    np.testing.assert_allclose(
        streamed.materialize(ids), dense.materialize(ids), rtol=1e-6
    )

    target = np.abs(rng.normal(size=6)) + 0.1
    exact = streaming_nonnegative_pursuit(dense, target[None, :], k_max=4)[0]
    from_stream = streaming_nonnegative_pursuit(streamed, target[None, :], k_max=4)[0]
    assert exact.support.tolist() == from_stream.support.tolist()
    np.testing.assert_allclose(exact.errors, from_stream.errors, rtol=1e-9, atol=1e-12)


def test_token_frame_rejects_centering_and_convention_mismatch() -> None:
    torch = pytest.importorskip("torch")

    from jlens_workspace.jacobian import build_effective_unembedding
    from jlens_workspace.pursuit import (
        TokenFrameDictionary,
        build_token_frame_dictionary,
    )

    effective = build_effective_unembedding(
        torch.eye(4, dtype=torch.float64), convention="raw"
    )
    jacobian = torch.eye(4, dtype=torch.float64)
    with pytest.raises(DictionaryError, match="centered"):
        TokenFrameDictionary(effective, jacobian, centered=True)
    with pytest.raises(DictionaryError, match="convention"):
        build_token_frame_dictionary(
            effective, jacobian, convention="rmsnorm_weighted"
        )
