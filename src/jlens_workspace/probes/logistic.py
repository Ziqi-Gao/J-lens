"""Leakage-safe binary logistic probes for residual-stream activations.

The public entry point, :func:`fit_logistic_probe`, deliberately requires an
explicit train/held-out split.  Hyperparameter selection only sees the
training split, and any feature standardisation lives inside the
cross-validation ``Pipeline``.

Inputs may be NumPy arrays or PyTorch tensors.  PyTorch inputs are detached,
moved to CPU, and converted to NumPy at the API boundary.  Results are NumPy
arrays and fitted scikit-learn objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FeatureArray = object
LabelArray = object


def _torch_module_for(value: object) -> Any | None:
    """Return torch for a torch tensor without making torch a core dependency."""

    value_type = type(value)
    if not value_type.__module__.startswith("torch"):
        return None
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - impossible for real tensors
        raise TypeError("received a PyTorch tensor but PyTorch is not installed") from error
    return torch if torch.is_tensor(value) else None


def _features_to_numpy(value: FeatureArray, *, name: str) -> NDArray[np.float64]:
    """Convert a two-dimensional feature array to finite CPU float64 NumPy."""

    torch = _torch_module_for(value)
    if torch is not None:
        value = value.detach().cpu().numpy()
    elif not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray or torch.Tensor")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [n_samples, d_model], got {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return np.ascontiguousarray(array)


def _labels_to_numpy(value: LabelArray, *, name: str) -> NDArray[Any]:
    """Convert a one-dimensional label/group array to CPU NumPy."""

    torch = _torch_module_for(value)
    if torch is not None:
        value = value.detach().cpu().numpy()
    elif not isinstance(value, np.ndarray):
        value = np.asarray(value)
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape [n_samples], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")
    return array


@dataclass(frozen=True)
class CVScore:
    """Cross-validated ROC-AUC summary for one L2 penalty strength."""

    C: float
    mean_auc: float
    std_auc: float
    fold_auc: tuple[float, ...]


@dataclass(frozen=True)
class HeldOutMetrics:
    """Metrics computed exactly once on the explicit held-out split."""

    roc_auc: float
    average_precision: float
    accuracy: float
    balanced_accuracy: float


@dataclass(frozen=True)
class LogisticProbeResult:
    """A fitted binary concept probe and its selection/evaluation record.

    ``coef_raw`` and ``intercept_raw`` are expressed in the original residual
    coordinate system.  Their score is positive for ``positive_label``, even
    when that label is scikit-learn's lexicographically first class.
    """

    pipeline: Pipeline
    chosen_C: float
    cv_scores: tuple[CVScore, ...]
    heldout: HeldOutMetrics
    coef_raw: NDArray[np.float64]
    intercept_raw: float
    positive_label: Any
    negative_label: Any
    classes: NDArray[Any]
    cv_strategy: str
    standardize: bool

    def decision_function(self, X: FeatureArray) -> NDArray[np.float64]:
        """Return the positive-label log-odds in raw residual coordinates."""

        features = _features_to_numpy(X, name="X")
        if features.shape[1] != self.coef_raw.shape[0]:
            raise ValueError(
                f"X has d_model={features.shape[1]}, expected {self.coef_raw.shape[0]}"
            )
        return features @ self.coef_raw + self.intercept_raw

    def positive_probability(self, X: FeatureArray) -> NDArray[np.float64]:
        """Return ``P(y == positive_label)`` from the raw-coordinate score."""

        return expit(self.decision_function(X))


def _validate_binary_split(
    X_train: NDArray[np.float64],
    y_train: NDArray[Any],
    X_heldout: NDArray[np.float64],
    y_heldout: NDArray[Any],
) -> NDArray[Any]:
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train have different sample counts")
    if X_heldout.shape[0] != y_heldout.shape[0]:
        raise ValueError("X_heldout and y_heldout have different sample counts")
    if X_train.shape[1] != X_heldout.shape[1]:
        raise ValueError("train and held-out residual widths differ")

    classes = np.unique(y_train)
    if classes.size != 2:
        raise ValueError(f"binary probe requires exactly two train classes, got {classes}")
    heldout_classes = np.unique(y_heldout)
    if heldout_classes.size != 2 or not np.array_equal(heldout_classes, classes):
        raise ValueError(
            "held-out labels must contain the same two classes as training labels"
        )
    return classes


def _raw_coordinates(
    pipeline: Pipeline, *, positive_is_class_one: bool
) -> tuple[NDArray[np.float64], float]:
    """Map the fitted standardised classifier back to input coordinates."""

    classifier = pipeline.named_steps["classifier"]
    if not isinstance(classifier, LogisticRegression):  # defensive for callers
        raise TypeError("pipeline classifier is not LogisticRegression")
    coef_scaled = np.asarray(classifier.coef_[0], dtype=np.float64)
    intercept_scaled = float(classifier.intercept_[0])

    scaler = pipeline.named_steps["scaler"]
    if scaler == "passthrough" or scaler is None:
        mean = np.zeros_like(coef_scaled)
        scale = np.ones_like(coef_scaled)
    elif isinstance(scaler, StandardScaler):
        mean = (
            np.asarray(scaler.mean_, dtype=np.float64)
            if scaler.with_mean
            else np.zeros_like(coef_scaled)
        )
        scale = (
            np.asarray(scaler.scale_, dtype=np.float64)
            if scaler.with_std
            else np.ones_like(coef_scaled)
        )
    else:  # pragma: no cover - the construction below controls this step
        raise TypeError(f"unsupported scaler type: {type(scaler)!r}")

    coef_raw = coef_scaled / scale
    intercept_raw = intercept_scaled - float(coef_raw @ mean)
    if not positive_is_class_one:
        coef_raw = -coef_raw
        intercept_raw = -intercept_raw
    return coef_raw, intercept_raw


def fit_logistic_probe(
    X_train: FeatureArray,
    y_train: LabelArray,
    X_heldout: FeatureArray,
    y_heldout: LabelArray,
    *,
    C_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    cv_splits: int = 5,
    groups: LabelArray | None = None,
    positive_label: Any | None = None,
    standardize: bool = True,
    class_weight: str | dict[Any, float] | None = None,
    random_state: int = 0,
    max_iter: int = 2_000,
    n_jobs: int = 1,
) -> LogisticProbeResult:
    """Tune and evaluate a binary L2-logistic residual-stream probe.

    ``C`` is selected by mean training-split CV ROC-AUC.  Without ``groups``,
    shuffled :class:`StratifiedKFold` is used.  With ``groups``, shuffled
    :class:`StratifiedGroupKFold` keeps every group wholly within a fold.
    Smaller ``C`` wins deterministic ties because the validated grid is sorted.

    The held-out split is never passed to ``GridSearchCV`` and therefore cannot
    influence scaling, model fitting, or hyperparameter selection.
    """

    train = _features_to_numpy(X_train, name="X_train")
    train_labels = _labels_to_numpy(y_train, name="y_train")
    heldout = _features_to_numpy(X_heldout, name="X_heldout")
    heldout_labels = _labels_to_numpy(y_heldout, name="y_heldout")
    classes = _validate_binary_split(train, train_labels, heldout, heldout_labels)

    if positive_label is None:
        positive_label = classes[1]
    positive_matches = np.flatnonzero(classes == positive_label)
    if positive_matches.size != 1:
        raise ValueError(f"positive_label={positive_label!r} is not a training class")
    positive_index = int(positive_matches[0])
    negative_label = classes[1 - positive_index]

    if not isinstance(cv_splits, int) or cv_splits < 2:
        raise ValueError("cv_splits must be an integer >= 2")
    c_values = tuple(sorted({float(value) for value in C_grid}))
    if not c_values or not all(np.isfinite(value) and value > 0 for value in c_values):
        raise ValueError("C_grid must contain at least one finite positive value")

    fit_groups: NDArray[Any] | None = None
    if groups is None:
        counts = np.unique(train_labels, return_counts=True)[1]
        if int(counts.min()) < cv_splits:
            raise ValueError("each class must have at least cv_splits training samples")
        cv = StratifiedKFold(
            n_splits=cv_splits, shuffle=True, random_state=random_state
        )
        cv_strategy = "stratified"
    else:
        fit_groups = _labels_to_numpy(groups, name="groups")
        if fit_groups.shape[0] != train.shape[0]:
            raise ValueError("groups and X_train have different sample counts")
        if np.unique(fit_groups).size < cv_splits:
            raise ValueError("groups must contain at least cv_splits distinct groups")
        for label in classes:
            class_groups = np.unique(fit_groups[train_labels == label])
            if class_groups.size < cv_splits:
                raise ValueError(
                    "each class must occur in at least cv_splits distinct groups"
                )
        cv = StratifiedGroupKFold(
            n_splits=cv_splits, shuffle=True, random_state=random_state
        )
        cv_strategy = "stratified_group"

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler() if standardize else "passthrough"),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    class_weight=class_weight,
                    random_state=random_state,
                    max_iter=max_iter,
                ),
            ),
        ]
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"classifier__C": c_values},
        scoring="roc_auc",
        cv=cv,
        refit=True,
        n_jobs=n_jobs,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(train, train_labels, groups=fit_groups)
    fitted = search.best_estimator_

    split_columns = sorted(
        key
        for key in search.cv_results_
        if key.startswith("split") and key.endswith("_test_score")
    )
    cv_scores: list[CVScore] = []
    for row_index, params in enumerate(search.cv_results_["params"]):
        folds = tuple(
            float(search.cv_results_[column][row_index]) for column in split_columns
        )
        cv_scores.append(
            CVScore(
                C=float(params["classifier__C"]),
                mean_auc=float(search.cv_results_["mean_test_score"][row_index]),
                std_auc=float(search.cv_results_["std_test_score"][row_index]),
                fold_auc=folds,
            )
        )

    fitted_classes = np.asarray(fitted.named_steps["classifier"].classes_)
    fitted_positive_index = int(np.flatnonzero(fitted_classes == positive_label)[0])
    probabilities = fitted.predict_proba(heldout)[:, fitted_positive_index]
    predictions = fitted.predict(heldout)
    heldout_metrics = HeldOutMetrics(
        roc_auc=float(
            roc_auc_score(heldout_labels == positive_label, probabilities)
        ),
        average_precision=float(
            average_precision_score(heldout_labels == positive_label, probabilities)
        ),
        accuracy=float(accuracy_score(heldout_labels, predictions)),
        balanced_accuracy=float(
            balanced_accuracy_score(heldout_labels, predictions)
        ),
    )

    coef_raw, intercept_raw = _raw_coordinates(
        fitted, positive_is_class_one=(fitted_positive_index == 1)
    )
    return LogisticProbeResult(
        pipeline=fitted,
        chosen_C=float(search.best_params_["classifier__C"]),
        cv_scores=tuple(cv_scores),
        heldout=heldout_metrics,
        coef_raw=coef_raw,
        intercept_raw=intercept_raw,
        positive_label=positive_label,
        negative_label=negative_label,
        classes=fitted_classes,
        cv_strategy=cv_strategy,
        standardize=standardize,
    )
