"""Versioned occupancy crossing rules and reconstruction summaries.

All functions are pure NumPy over precomputed gain/error curves so they are
fully unit-testable offline. Both crossing rules compare the real-dictionary
marginal gains against the per-k MEDIAN across random-control seeds:

- ``first_nonexceed_v1``: the first k (1-indexed) with
  ``G_J(k) <= median_seeds G_random(k)``;
- ``consecutive3_nonexceed_v1``: the first k opening a run of three
  consecutive ks that all satisfy the condition; the reported value is the
  first k of that run.

When no crossing occurs up to ``k_max`` the occupancy is right-censored:
``k`` is ``None`` and ``right_censored`` is true — it must never be recorded
as ``k_max`` itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

CROSSING_RULES = ("first_nonexceed_v1", "consecutive3_nonexceed_v1")


class OccupancyRuleError(ValueError):
    """Raised for malformed curves or unknown rule identifiers."""


def _validate_gains(gains: object, name: str) -> FloatArray:
    array = np.asarray(gains, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise OccupancyRuleError(f"{name} must be a non-empty 1-D gain curve")
    if not np.isfinite(array).all():
        raise OccupancyRuleError(f"{name} must be finite")
    return array


def median_control_gains(control_gains: Sequence[object]) -> FloatArray:
    """Per-k median across random-control seeds' gain curves."""

    curves = [_validate_gains(curve, "control gain curve") for curve in control_gains]
    if not curves:
        raise OccupancyRuleError("at least one control gain curve is required")
    lengths = {curve.size for curve in curves}
    if len(lengths) != 1:
        raise OccupancyRuleError("control gain curves must share a length")
    return np.median(np.stack(curves), axis=0)


def crossing_k(
    real_gains: object,
    control_gains: Sequence[object],
    *,
    rule: str,
) -> dict[str, object]:
    """Apply one versioned crossing rule; k is 1-indexed over gains."""

    if rule not in CROSSING_RULES:
        raise OccupancyRuleError(f"unknown crossing rule {rule!r}")
    gains = _validate_gains(real_gains, "real gain curve")
    control = median_control_gains(control_gains)
    if control.size != gains.size:
        raise OccupancyRuleError("real and control gain curves must share a length")
    nonexceed = gains <= control
    k_max = int(gains.size)

    if rule == "first_nonexceed_v1":
        hits = np.flatnonzero(nonexceed)
        if hits.size:
            return {"rule": rule, "k": int(hits[0]) + 1, "right_censored": False}
        return {"rule": rule, "k": None, "right_censored": True, "k_max": k_max}

    run = 0
    for index, flag in enumerate(nonexceed):
        run = run + 1 if flag else 0
        if run == 3:
            return {"rule": rule, "k": int(index) - 1, "right_censored": False}
    return {"rule": rule, "k": None, "right_censored": True, "k_max": k_max}


def k_90_attainable(errors: object) -> int:
    """Smallest k with ``E(0) - E(k) >= 0.9 * (E(0) - E(k_max))``.

    Defined relative to the total reconstruction benefit actually attained by
    ``k_max`` (not an absolute 90% reconstruction). Returns 0 when no benefit
    was attained at all.
    """

    curve = np.asarray(errors, dtype=np.float64)
    if curve.ndim != 1 or curve.size < 2:
        raise OccupancyRuleError("errors must contain E(0)..E(k_max)")
    total_benefit = float(curve[0] - curve[-1])
    if total_benefit <= 0.0:
        return 0
    benefit = curve[0] - curve
    hits = np.flatnonzero(benefit >= 0.9 * total_benefit - 1e-15)
    return int(hits[0])


def absolute_threshold_ks(
    errors: object,
    thresholds: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
) -> Mapping[str, int | None]:
    """Smallest k whose explained fraction ``1 - E(k)`` reaches each threshold.

    Unreached thresholds map to ``None`` (recorded as ``not_reached``), never
    silently clamped.
    """

    curve = np.asarray(errors, dtype=np.float64)
    if curve.ndim != 1 or curve.size < 1:
        raise OccupancyRuleError("errors must be a non-empty curve")
    explained = 1.0 - curve
    output: dict[str, int | None] = {}
    for threshold in thresholds:
        value = float(threshold)
        if not 0 < value < 1:
            raise OccupancyRuleError("thresholds must lie in (0, 1)")
        hits = np.flatnonzero(explained >= value - 1e-15)
        output[f"{value:g}"] = int(hits[0]) if hits.size else None
    return output
