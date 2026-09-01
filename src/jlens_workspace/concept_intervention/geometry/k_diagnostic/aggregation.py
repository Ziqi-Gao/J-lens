"""Null-calibrated K and fixed-budget summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from jlens_workspace.concept_intervention.geometry.sparse_pursuit import crossing_k


class AggregationError(ValueError):
    """Saved gain/error curves cannot support the registered statistics."""


def observed_crossing_summary(
    values: Sequence[int | None], censored: Sequence[bool], *, k_max: int
) -> dict[str, Any]:
    """Summarize observed crossings without treating censoring as missing at random."""

    if len(values) != len(censored) or not values:
        raise AggregationError("crossings and censor flags must align and be non-empty")
    observed = np.asarray(
        [int(value) for value, flag in zip(values, censored, strict=True) if not flag],
        dtype=np.int64,
    )
    if any((value is None) != flag for value, flag in zip(values, censored, strict=True)):
        raise AggregationError("right-censored crossings must be represented by None")
    return {
        "sample_count": len(values),
        "observed_crossing_count": int(observed.size),
        "right_censored_count": int(sum(censored)),
        "right_censored_fraction": float(np.mean(censored)),
        "observed_crossing_median": (
            None if observed.size == 0 else float(np.median(observed))
        ),
        "observed_crossing_q25": (
            None if observed.size == 0 else float(np.quantile(observed, 0.25))
        ),
        "observed_crossing_q75": (
            None if observed.size == 0 else float(np.quantile(observed, 0.75))
        ),
        "censoring_point": int(k_max),
        "population_median_reported": False,
    }


def hierarchical_bootstrap_mean(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    samples: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap layer -> source/concept -> template/target; never null seeds."""

    if samples < 1000 or not 0.0 < confidence_level < 1.0:
        raise AggregationError("hierarchical bootstrap settings are invalid")
    prepared: list[tuple[int, str, str, float]] = []
    for row in rows:
        value = row.get(value_key)
        if value is None or not np.isfinite(float(value)):
            continue
        prepared.append(
            (
                int(row["layer"]),
                str(row["source_id"]),
                str(row.get("resampling_unit", row["target_id"])),
                float(value),
            )
        )
    if not prepared:
        raise AggregationError("hierarchical bootstrap has no finite target-level rows")
    by_layer: dict[int, dict[str, dict[str, list[float]]]] = {}
    for layer, source, unit, value in prepared:
        by_layer.setdefault(layer, {}).setdefault(source, {}).setdefault(unit, []).append(
            value
        )
    layers = sorted(by_layer)
    rng = np.random.Generator(np.random.Philox(seed))
    estimates = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        draw: list[float] = []
        for layer in rng.choice(layers, size=len(layers), replace=True):
            sources = sorted(by_layer[int(layer)])
            for source in rng.choice(sources, size=len(sources), replace=True):
                units = sorted(by_layer[int(layer)][str(source)])
                for unit in rng.choice(units, size=len(units), replace=True):
                    values = by_layer[int(layer)][str(source)][str(unit)]
                    draw.append(float(values[int(rng.integers(0, len(values)))]))
        estimates[sample_index] = float(np.mean(draw))
    alpha = 1.0 - confidence_level
    estimate = float(np.mean([entry[3] for entry in prepared]))
    return {
        "effect": estimate,
        "ci_low": float(np.quantile(estimates, alpha / 2.0)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
        "bootstrap_standard_error": float(np.std(estimates, ddof=1)),
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "resampling_hierarchy": ["layer", "concept_or_source", "template_or_target"],
        "target_level_n": len(prepared),
        "layer_n": len(layers),
        "source_n": len({entry[1] for entry in prepared}),
        "bootstrap_effects": estimates.tolist(),
    }


def hierarchical_group_contrast(
    numerator_rows: Sequence[Mapping[str, Any]],
    denominator_rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    samples: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap a nonpaired numerator-minus-denominator contrast by layer.

    Layers are sampled as the top-level clusters. Within each selected layer,
    numerator and denominator source/target units are sampled independently.
    Null seeds must already have been collapsed into each target-level value.
    """

    if samples < 1000 or not 0.0 < confidence_level < 1.0:
        raise AggregationError("hierarchical group bootstrap settings are invalid")

    def prepare(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, list[float]]]:
        output: dict[int, dict[str, list[float]]] = {}
        for row in rows:
            value = row.get(value_key)
            if value is None or not np.isfinite(float(value)):
                continue
            layer = int(row["layer"])
            source = str(row["source_id"])
            output.setdefault(layer, {}).setdefault(source, []).append(float(value))
        return output

    numerator = prepare(numerator_rows)
    denominator = prepare(denominator_rows)
    layers = sorted(set(numerator) & set(denominator))
    if not layers:
        raise AggregationError("group contrast has no common layer clusters")
    if set(numerator) != set(layers) or set(denominator) != set(layers):
        raise AggregationError("group contrast layers do not align exactly")

    def draw_group(
        rng: np.random.Generator, group: dict[str, list[float]], layer: int
    ) -> float:
        sources = sorted(group[layer])
        sampled = rng.choice(sources, size=len(sources), replace=True)
        values = [
            group[layer][str(source)][
                int(rng.integers(0, len(group[layer][str(source)])))
            ]
            for source in sampled
        ]
        return float(np.mean(values))

    rng = np.random.Generator(np.random.Philox(seed))
    estimates = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        effects = []
        for layer in rng.choice(layers, size=len(layers), replace=True):
            selected_layer = int(layer)
            effects.append(
                draw_group(rng, numerator, selected_layer)
                - draw_group(rng, denominator, selected_layer)
            )
        estimates[sample_index] = float(np.mean(effects))

    layer_effects = [
        np.mean([value for values in numerator[layer].values() for value in values])
        - np.mean([value for values in denominator[layer].values() for value in values])
        for layer in layers
    ]
    alpha = 1.0 - confidence_level
    return {
        "effect": float(np.mean(layer_effects)),
        "ci_low": float(np.quantile(estimates, alpha / 2.0)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
        "bootstrap_standard_error": float(np.std(estimates, ddof=1)),
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "resampling_hierarchy": [
            "layer",
            "independent_numerator_and_denominator_source_or_target",
        ],
        "layer_n": len(layers),
        "numerator_target_n": sum(
            len(values) for layer in numerator.values() for values in layer.values()
        ),
        "denominator_target_n": sum(
            len(values) for layer in denominator.values() for values in layer.values()
        ),
        "numerator_source_n": len(
            {source for layer in numerator.values() for source in layer}
        ),
        "denominator_source_n": len(
            {source for layer in denominator.values() for source in layer}
        ),
        "bootstrap_effects": estimates.tolist(),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values in original order."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise AggregationError("Holm p-values must be finite in [0,1]")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = values.size
    for rank, index in enumerate(order):
        running = max(running, float((total - rank) * values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def classify_registered_effect(
    *,
    ci_low: float,
    ci_high: float,
    minimum_effect: float,
    equivalence_margin: float,
    adjusted_p_value: float,
    alpha: float = 0.05,
) -> str:
    """Apply the preregistered positive-direction decision thresholds."""

    if ci_low >= minimum_effect and adjusted_p_value <= alpha:
        return "Supported"
    if ci_high <= equivalence_margin:
        return "Not supported"
    return "Inconclusive"


def empirical_upper_tail(value: float, null_values: object) -> float:
    values = np.asarray(null_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise AggregationError("empirical null values must be finite and one-dimensional")
    return float((1 + np.sum(values >= value)) / (values.size + 1))


def empirical_percentile(value: float, null_values: object) -> float:
    """Finite-sample lower-tail percentile with the same plus-one convention."""

    values = np.asarray(null_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise AggregationError("empirical null values must be finite and one-dimensional")
    return float((1 + np.sum(values <= value)) / (values.size + 1))


def summarize_null_calibrated_curve(
    real_errors: object,
    null_errors: object,
    *,
    report_grid: Sequence[int] = (1, 2, 4, 8, 16, 25, 32, 64),
) -> dict[str, Any]:
    """Report all registered occupancy definitions without conflating budgets."""

    real = np.asarray(real_errors, dtype=np.float64)
    null = np.asarray(null_errors, dtype=np.float64)
    if real.ndim != 1 or real.size < 2 or null.ndim != 2:
        raise AggregationError("errors require shapes [K+1] and [B,K+1]")
    if null.shape[1] != real.size or null.shape[0] < 1:
        raise AggregationError("real and null curves must share K_max")
    if not np.isfinite(real).all() or not np.isfinite(null).all():
        raise AggregationError("error curves must be finite")
    if np.any(np.diff(real) > 1e-10) or np.any(np.diff(null, axis=1) > 1e-10):
        raise AggregationError("error curves must be non-increasing")
    real_gains = real[:-1] - real[1:]
    null_gains = null[:, :-1] - null[:, 1:]
    control_curves = [row for row in null_gains]
    first = crossing_k(real_gains, control_curves, rule="first_nonexceed_v1")
    consecutive = crossing_k(
        real_gains, control_curves, rule="consecutive3_nonexceed_v1"
    )

    def supported(outcome: dict[str, object]) -> int | None:
        crossing = outcome["k"]
        return None if crossing is None else max(0, int(crossing) - 1)

    median_gain = np.median(null_gains, axis=0)
    cumulative = np.cumsum(real_gains - median_gain)
    peak = int(np.argmax(cumulative)) + 1
    k_max = real.size - 1
    fixed: dict[str, Any] = {}
    for requested in report_grid:
        if requested > k_max:
            continue
        null_explained = 1.0 - null[:, requested]
        real_explained = float(1.0 - real[requested])
        fixed[str(requested)] = {
            "explained_fraction": real_explained,
            "null_median_explained_fraction": float(
                np.median(null_explained)
            ),
            "real_minus_null": float(
                np.median(null[:, requested]) - real[requested]
            ),
            "empirical_upper_tail_p_value": empirical_upper_tail(
                real_explained, null_explained
            ),
            "empirical_percentile": empirical_percentile(
                real_explained, null_explained
            ),
            "semantics": "fixed reconstruction budget, not estimated occupancy",
        }
    cumulative_at = {
        str(k): float(cumulative[min(k, k_max) - 1])
        for k in (4, 16, 25, 64)
        if min(k, k_max) >= 1
    }
    marginal = [
        {
            "K": int(k),
            "real_marginal_gain": float(real_gains[k - 1]),
            "null_median_marginal_gain": float(np.median(null_gains[:, k - 1])),
            "empirical_upper_tail_p_value": empirical_upper_tail(
                float(real_gains[k - 1]), null_gains[:, k - 1]
            ),
            "empirical_percentile": empirical_percentile(
                float(real_gains[k - 1]), null_gains[:, k - 1]
            ),
        }
        for k in range(1, k_max + 1)
    ]
    null_maxima = np.empty(null.shape[0], dtype=np.float64)
    for null_index in range(null.shape[0]):
        if null.shape[0] > 1:
            comparison = np.delete(null_gains, null_index, axis=0)
            comparison_median = np.median(comparison, axis=0)
        else:
            comparison_median = np.zeros(k_max, dtype=np.float64)
        null_maxima[null_index] = float(
            np.max(np.cumsum(null_gains[null_index] - comparison_median))
        )
    fixed_empirical = {
        budget: {
            "observed_real_minus_null_median": values["real_minus_null"],
            "empirical_upper_tail_p_value": values[
                "empirical_upper_tail_p_value"
            ],
            "empirical_percentile": values["empirical_percentile"],
        }
        for budget, values in fixed.items()
    }
    empirical = {
        "schema_version": 1,
        "null_seed_count": int(null.shape[0]),
        "resolution": float(1.0 / (null.shape[0] + 1)),
        "plus_one_correction": True,
        "marginal_gain_by_k": marginal,
        "fixed_k4": fixed_empirical.get("4"),
        "fixed_reconstruction_benefit": fixed_empirical,
        "cumulative_benefit": {
            "statistic": "max cumulative real gain minus within-target null median gain",
            "observed": float(cumulative[peak - 1]),
            "empirical_upper_tail_p_value": empirical_upper_tail(
                float(cumulative[peak - 1]), null_maxima
            ),
            "empirical_percentile": empirical_percentile(
                float(cumulative[peak - 1]), null_maxima
            ),
            "null_leave_one_out_statistics": null_maxima.tolist(),
        },
    }
    return {
        "K_first": supported(first),
        "K_first_right_censored": bool(first["right_censored"]),
        "K_consecutive3": supported(consecutive),
        "K_consecutive3_right_censored": bool(consecutive["right_censored"]),
        "K_peak": peak,
        "max_cumulative_excess": float(cumulative[peak - 1]),
        "cumulative_excess": cumulative.tolist(),
        "cumulative_excess_at": cumulative_at,
        "fixed_budget": fixed,
        "empirical_diagnostics": empirical,
        "null_seed_count": int(null.shape[0]),
        "null_percentile_resolution": float(1.0 / (null.shape[0] + 1)),
    }
