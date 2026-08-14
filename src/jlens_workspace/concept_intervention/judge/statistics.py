"""Prompt-clustered inference helpers for the registered judge study."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import spearmanr


class JudgeStatisticsError(ValueError):
    """Raised when paired inference inputs violate the registered design."""


def agreement_metrics(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    """Report rank and absolute agreement for two aligned pointwise judges."""

    if not left or len(left) != len(right):
        raise JudgeStatisticsError("agreement arrays must be non-empty and aligned")
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    correlation = float(spearmanr(first, second).statistic)
    if not np.isfinite(correlation):
        correlation = 0.0
    difference = np.abs(first - second)
    return {
        "n": int(first.size),
        "spearman": correlation,
        "mae": float(np.mean(difference)),
        "within_10": float(np.mean(difference <= 10.0)),
        "within_20": float(np.mean(difference <= 20.0)),
    }


def paired_cluster_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap_samples: int,
    permutation_samples: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Estimate a paired effect after averaging decodings/judges within prompt."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        value = float(row["delta"])
        if not np.isfinite(value):
            raise JudgeStatisticsError("paired deltas must be finite")
        grouped[str(row["prompt_id"])].append(value)
    if not grouped:
        raise JudgeStatisticsError("paired cluster summary requires records")
    cluster_ids = sorted(grouped)
    values = np.asarray(
        [float(np.mean(grouped[cluster])) for cluster in cluster_ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
    bootstrap = np.mean(values[indices], axis=1)
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])
    observed = float(np.mean(values))
    signs = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(permutation_samples, len(values)),
        replace=True,
    )
    null = np.mean(signs * values, axis=1)
    p_value = float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (1 + len(null)))
    return {
        "estimate": observed,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence_level": confidence_level,
        "p_value_two_sided": p_value,
        "prompt_clusters": len(values),
        "observations": len(records),
        "cluster_means": {
            cluster: float(value) for cluster, value in zip(cluster_ids, values, strict=True)
        },
    }


def hierarchical_macro_summary(
    by_concept: Mapping[str, Mapping[str, float]],
    *,
    seed: int,
    bootstrap_samples: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Macro-average concepts while resampling prompt clusters within each concept."""

    if not by_concept:
        raise JudgeStatisticsError("macro summary requires concepts")
    concepts = sorted(by_concept)
    values: list[np.ndarray] = []
    for concept in concepts:
        cluster_values = np.asarray(
            [float(value) for value in by_concept[concept].values()],
            dtype=np.float64,
        )
        if cluster_values.size == 0 or not np.all(np.isfinite(cluster_values)):
            raise JudgeStatisticsError(f"invalid prompt clusters for {concept}")
        values.append(cluster_values)
    rng = np.random.default_rng(seed)
    bootstrap_concepts = []
    for cluster_values in values:
        indices = rng.integers(
            0,
            len(cluster_values),
            size=(bootstrap_samples, len(cluster_values)),
        )
        bootstrap_concepts.append(np.mean(cluster_values[indices], axis=1))
    bootstrap = np.mean(np.stack(bootstrap_concepts, axis=1), axis=1)
    concept_estimates = [float(np.mean(value)) for value in values]
    estimate = float(np.mean(concept_estimates))
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "estimate": estimate,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence_level": confidence_level,
        "concept_count": len(concepts),
        "concept_estimates": dict(zip(concepts, concept_estimates, strict=True)),
    }


def benjamini_hochberg(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return monotone Benjamini-Hochberg adjusted p-values by hypothesis ID."""

    if not p_values:
        return {}
    ordered = sorted(p_values, key=p_values.get)
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for rank, key in reversed(list(enumerate(ordered, start=1))):
        value = float(p_values[key])
        if not 0.0 <= value <= 1.0:
            raise JudgeStatisticsError("p-values must lie in [0, 1]")
        running = min(running, value * count / rank)
        adjusted[key] = float(min(1.0, running))
    return adjusted
