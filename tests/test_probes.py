from __future__ import annotations

import numpy as np
import pytest

from jlens_workspace.probes import fit_logistic_probe


def _synthetic_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260712)
    scales = np.array([0.05, 8.0, 0.7, 3.0, 1.5])
    features = rng.normal(size=(360, scales.size)) * scales
    raw_direction = np.array([20.0, -0.22, 1.3, 0.0, 0.35])
    latent_score = features @ raw_direction + 0.35 * rng.normal(size=features.shape[0])
    labels = (latent_score > np.median(latent_score)).astype(np.int64)
    permutation = rng.permutation(features.shape[0])
    train_indices = permutation[:260]
    heldout_indices = permutation[260:]
    return (
        features[train_indices],
        labels[train_indices],
        features[heldout_indices],
        labels[heldout_indices],
    )


def test_probe_tunes_on_train_and_exports_raw_coordinates() -> None:
    X_train, y_train, X_heldout, y_heldout = _synthetic_split()
    result = fit_logistic_probe(
        X_train,
        y_train,
        X_heldout,
        y_heldout,
        C_grid=(0.01, 0.1, 1.0),
        cv_splits=4,
        random_state=17,
    )

    assert result.chosen_C in {0.01, 0.1, 1.0}
    assert [score.C for score in result.cv_scores] == [0.01, 0.1, 1.0]
    assert all(len(score.fold_auc) == 4 for score in result.cv_scores)
    assert result.heldout.roc_auc > 0.95
    assert result.heldout.balanced_accuracy > 0.85

    pipeline_score = result.pipeline.decision_function(X_heldout)
    np.testing.assert_allclose(
        result.decision_function(X_heldout), pipeline_score, rtol=1e-12, atol=1e-12
    )
    positive_index = int(
        np.flatnonzero(result.pipeline.classes_ == result.positive_label)[0]
    )
    np.testing.assert_allclose(
        result.positive_probability(X_heldout),
        result.pipeline.predict_proba(X_heldout)[:, positive_index],
        rtol=1e-12,
        atol=1e-12,
    )


def test_positive_label_zero_flips_the_raw_score() -> None:
    X_train, y_train, X_heldout, y_heldout = _synthetic_split()
    result = fit_logistic_probe(
        X_train,
        y_train,
        X_heldout,
        y_heldout,
        C_grid=(0.1,),
        cv_splits=3,
        positive_label=0,
    )

    np.testing.assert_allclose(
        result.decision_function(X_heldout),
        -result.pipeline.decision_function(X_heldout),
        rtol=1e-12,
        atol=1e-12,
    )
    zero_index = int(np.flatnonzero(result.pipeline.classes_ == 0)[0])
    np.testing.assert_allclose(
        result.positive_probability(X_heldout),
        result.pipeline.predict_proba(X_heldout)[:, zero_index],
        rtol=1e-12,
        atol=1e-12,
    )


def test_heldout_values_cannot_change_selection_or_fit() -> None:
    X_train, y_train, X_heldout, y_heldout = _synthetic_split()
    first = fit_logistic_probe(
        X_train,
        y_train,
        X_heldout,
        y_heldout,
        C_grid=(0.001, 0.01, 0.1, 1.0),
        cv_splits=4,
        random_state=9,
    )
    # Extreme held-out values would visibly alter a scaler fitted before the
    # split.  They may alter evaluation, but never CV or the fitted probe.
    shifted_heldout = X_heldout * 1_000.0 + 50_000.0
    second = fit_logistic_probe(
        X_train,
        y_train,
        shifted_heldout,
        y_heldout,
        C_grid=(0.001, 0.01, 0.1, 1.0),
        cv_splits=4,
        random_state=9,
    )

    assert first.chosen_C == second.chosen_C
    assert first.cv_scores == second.cv_scores
    np.testing.assert_array_equal(first.coef_raw, second.coef_raw)
    assert first.intercept_raw == second.intercept_raw


def test_grouped_cv_is_deterministic_and_preserves_group_mode() -> None:
    rng = np.random.default_rng(88)
    groups = np.repeat(np.arange(20), 4)
    labels = np.repeat(np.arange(20) % 2, 4)
    train = rng.normal(size=(groups.size, 4))
    train[:, 0] += (2 * labels - 1) * 2.0
    heldout_labels = np.tile(np.array([0, 1]), 20)
    heldout = rng.normal(size=(heldout_labels.size, 4))
    heldout[:, 0] += (2 * heldout_labels - 1) * 2.0

    kwargs = {
        "C_grid": (0.01, 0.1, 1.0),
        "cv_splits": 5,
        "groups": groups,
        "random_state": 123,
    }
    first = fit_logistic_probe(train, labels, heldout, heldout_labels, **kwargs)
    second = fit_logistic_probe(train, labels, heldout, heldout_labels, **kwargs)

    assert first.cv_strategy == "stratified_group"
    assert first.chosen_C == second.chosen_C
    assert first.cv_scores == second.cv_scores
    np.testing.assert_array_equal(first.coef_raw, second.coef_raw)


def test_torch_tensors_are_an_explicit_optional_boundary() -> None:
    torch = pytest.importorskip("torch")
    X_train, y_train, X_heldout, y_heldout = _synthetic_split()
    result = fit_logistic_probe(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train),
        torch.tensor(X_heldout, dtype=torch.float32),
        torch.tensor(y_heldout),
        C_grid=(0.1,),
        cv_splits=3,
    )

    assert isinstance(result.coef_raw, np.ndarray)
    assert result.heldout.roc_auc > 0.95
