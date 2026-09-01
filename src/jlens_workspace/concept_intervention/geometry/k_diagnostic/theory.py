"""Isotropic high-dimensional extreme-value references."""

from __future__ import annotations

import math

from scipy.stats import beta


class TheoryError(ValueError):
    """An isotropic sphere reference received invalid parameters."""


def sphere_max_quantile(dimension: int, cardinality: int, quantile: float) -> float:
    """Exact quantile of the maximum one-sided cosine on ``S^(r-1)``.

    For one cosine ``C``, ``C^2 ~ Beta(1/2, (r-1)/2)`` and symmetry gives
    ``F_C(c) = (1 + F_Beta(c^2))/2`` for ``c >= 0``.  Inverting
    ``F_C(c)^V = quantile`` produces the reported positive-cosine maximum.
    """

    if dimension < 2 or cardinality < 1 or not 0.0 < quantile < 1.0:
        raise TheoryError("dimension>=2, cardinality>=1, and quantile in (0,1) required")
    single_cdf = quantile ** (1.0 / cardinality)
    beta_cdf = 2.0 * single_cdf - 1.0
    if beta_cdf <= 0.0:
        # This branch describes a negative maximum; registered cardinalities
        # are >=64 and never enter it for q05/q50/q95.
        return -math.sqrt(float(beta.ppf(1.0 - 2.0 * single_cdf, 0.5, (dimension - 1) / 2)))
    squared = float(beta.ppf(min(beta_cdf, 1.0), 0.5, (dimension - 1) / 2))
    return math.sqrt(max(0.0, squared))


def isotropic_sphere_reference(dimension: int, cardinality: int) -> dict[str, float | int | str]:
    """Return the preregistered approximation and exact order statistics."""

    if dimension < 2 or cardinality < 1:
        raise TheoryError("dimension>=2 and cardinality>=1 required")
    logarithm = math.log(cardinality)
    return {
        "reference": "isotropic reference",
        "dimension": int(dimension),
        "cardinality": int(cardinality),
        "single_cosine_std": 1.0 / math.sqrt(dimension),
        "approx_expected_max": math.sqrt(2.0 * logarithm / dimension),
        "exact_max_median": sphere_max_quantile(dimension, cardinality, 0.5),
        "exact_max_q05": sphere_max_quantile(dimension, cardinality, 0.05),
        "exact_max_q95": sphere_max_quantile(dimension, cardinality, 0.95),
        "approx_first_step_explained": 2.0 * logarithm / dimension,
    }
