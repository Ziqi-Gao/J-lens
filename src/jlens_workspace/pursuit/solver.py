"""Exact full-vocabulary streaming non-negative pursuit (method v1).

Solves, per target ``w`` and dictionary ``{a_t}``,

    min_{alpha >= 0, ||alpha||_0 <= k}  ||w - sum_t alpha_t a_t||_2^2

with a greedy pursuit whose every iteration (1) re-scans the FULL dictionary
against the CURRENT residual, (2) selects one new admissible atom, (3) refits
all selected coefficients with non-negative least squares on the RAW atoms,
and (4) records the normalized error path. A fixed candidate pool chosen from
the initial target similarity is deliberately not this algorithm; that
approximation lives elsewhere and is labeled as such.

Selection scores (``method_v1``):

- ``positive_cosine`` (primary): maximize ``<a_t, r> / ||a_t||`` over atoms
  with strictly positive score (the common ``||r||`` factor is dropped);
- ``raw_positive_dot`` (sensitivity): maximize the raw inner product
  ``<a_t, r>`` over strictly positive scores.

Errors are reported target-normalized: ``E(k) = ||w - w_J(k)||^2 / ||w||^2``
with ``E(0) = 1``; marginal gains are ``G(k) = E(k-1) - E(k)``. The target is
never unit-normalized before solving.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SELECTION_MODES = ("positive_cosine", "raw_positive_dot")
SOLVER_METHOD = "concept_occupancy_method_v1"

# Guard for the mathematically guaranteed monotonicity of NNLS-over-superset;
# violations beyond float64 roundoff indicate an implementation defect.
_MONOTONICITY_SLACK = 1e-10


class PursuitSolverError(RuntimeError):
    """Raised when pursuit inputs or numerical invariants are violated."""


@dataclass(frozen=True)
class PursuitResult:
    """Full per-integer-k pursuit path for one target.

    ``errors[k]`` is the normalized squared error after the NNLS refit at
    support size ``k`` (``errors[0] == 1``). Frozen entries repeat once the
    solver stops early (no admissible atom). ``coefficients_per_k[k]`` aligns
    with ``support[:k]`` in selection order.
    """

    target_norm: float
    support: IntArray
    errors: FloatArray
    gains: FloatArray
    coefficients_per_k: tuple[FloatArray, ...]
    residual_norms: FloatArray
    j_component_norms: FloatArray
    cosine_with_target: FloatArray
    condition_numbers: FloatArray
    selected_atom_norms: FloatArray
    stopped_early_at: int | None
    selection_mode: str

    @property
    def k_max(self) -> int:
        return int(self.errors.size - 1)

    def coefficients_at(self, k: int) -> FloatArray:
        if not 0 <= k <= self.k_max:
            raise ValueError(f"k must lie in [0, {self.k_max}]")
        effective = min(k, self.support.size)
        return self.coefficients_per_k[effective]


def _validate_targets(targets: object, d_model: int) -> FloatArray:
    matrix = np.asarray(targets, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != d_model:
        raise PursuitSolverError(f"targets must have shape [B, {d_model}]")
    if matrix.shape[0] == 0:
        raise PursuitSolverError("at least one target is required")
    if not np.isfinite(matrix).all():
        raise PursuitSolverError("targets must be finite")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise PursuitSolverError("targets must be non-zero")
    return matrix


def streaming_nonnegative_pursuit(
    dictionary: object,
    targets: object,
    *,
    k_max: int,
    selection_modes: Sequence[str] | str = "positive_cosine",
    zero_norm_policy: str = "skip",
) -> list[PursuitResult]:
    """Run batched exact pursuit; every target keeps its own support/residual.

    All targets in the batch share each iteration's single full-dictionary
    scan (``dots_batch``), which is what keeps the streamed random-control
    dictionaries affordable. ``selection_modes`` may be one mode for the whole
    batch or one per target.
    """

    if not isinstance(k_max, int) or isinstance(k_max, bool) or k_max < 1:
        raise PursuitSolverError("k_max must be a positive integer")
    d_model = int(dictionary.d_model)
    n_atoms = int(dictionary.n_atoms)
    matrix = _validate_targets(targets, d_model)
    batch = matrix.shape[0]
    if isinstance(selection_modes, str):
        modes = [selection_modes] * batch
    else:
        modes = [str(mode) for mode in selection_modes]
    if len(modes) != batch:
        raise PursuitSolverError("selection_modes must match the target batch")
    unknown = sorted(set(modes) - set(SELECTION_MODES))
    if unknown:
        raise PursuitSolverError(f"unknown selection modes: {unknown}")
    if zero_norm_policy != "skip":
        raise PursuitSolverError("zero_norm_policy must be 'skip'")

    atom_norms = np.asarray(dictionary.atom_norms(), dtype=np.float64)
    if atom_norms.shape != (n_atoms,):
        raise PursuitSolverError("dictionary atom_norms shape mismatch")
    zero_norm_mask = atom_norms == 0.0

    target_norms = np.linalg.norm(matrix, axis=1)
    squared_target_norms = target_norms**2

    residuals = matrix.copy()
    supports: list[list[int]] = [[] for _ in range(batch)]
    support_atoms: list[FloatArray] = [
        np.empty((0, d_model), dtype=np.float64) for _ in range(batch)
    ]
    coefficients_paths: list[list[FloatArray]] = [
        [np.empty(0, dtype=np.float64)] for _ in range(batch)
    ]
    errors = np.ones((batch, k_max + 1), dtype=np.float64)
    residual_norms = np.empty((batch, k_max + 1), dtype=np.float64)
    residual_norms[:, 0] = target_norms
    j_norms = np.zeros((batch, k_max + 1), dtype=np.float64)
    cosines = np.full((batch, k_max + 1), np.nan, dtype=np.float64)
    conditions = np.full((batch, k_max + 1), np.nan, dtype=np.float64)
    active = np.ones(batch, dtype=bool)
    stopped_at: list[int | None] = [None] * batch

    for step in range(1, k_max + 1):
        if active.any():
            active_indices = np.flatnonzero(active)
            dots = dictionary.dots_batch(residuals[active_indices])
            new_ids: dict[int, int] = {}
            for column, target_index in enumerate(active_indices):
                scores = dots[:, column].copy()
                if modes[target_index] == "positive_cosine":
                    with np.errstate(divide="ignore", invalid="ignore"):
                        scores = np.where(
                            zero_norm_mask, -np.inf, scores / atom_norms
                        )
                if supports[target_index]:
                    scores[np.asarray(supports[target_index], dtype=np.int64)] = -np.inf
                best = int(np.argmax(scores))
                if not scores[best] > 0.0:
                    active[target_index] = False
                    stopped_at[target_index] = len(supports[target_index])
                    continue
                new_ids[target_index] = best

            if new_ids:
                unique_ids = np.asarray(sorted(set(new_ids.values())), dtype=np.int64)
                materialized = dictionary.materialize(unique_ids)
                lookup = {
                    int(atom): row
                    for atom, row in zip(unique_ids, materialized, strict=True)
                }
                for target_index, atom_id in new_ids.items():
                    supports[target_index].append(int(atom_id))
                    support_atoms[target_index] = np.vstack(
                        (support_atoms[target_index], lookup[atom_id][None, :])
                    )
                    atoms = support_atoms[target_index]
                    coefficients, _ = nnls(atoms.T, matrix[target_index])
                    coefficients = np.asarray(coefficients, dtype=np.float64)
                    coefficients_paths[target_index].append(coefficients)
                    reconstruction = coefficients @ atoms
                    residual = matrix[target_index] - reconstruction
                    residuals[target_index] = residual
                    error = float(
                        (residual @ residual) / squared_target_norms[target_index]
                    )
                    previous = errors[target_index, step - 1]
                    if error > previous + _MONOTONICITY_SLACK:
                        raise PursuitSolverError(
                            f"error increased at k={step}: {previous} -> {error}"
                        )
                    errors[target_index, step] = min(error, previous)
                    residual_norms[target_index, step] = float(np.linalg.norm(residual))
                    j_norms[target_index, step] = float(np.linalg.norm(reconstruction))
                    denominator = (
                        np.linalg.norm(reconstruction) * target_norms[target_index]
                    )
                    if denominator > 0:
                        cosines[target_index, step] = float(
                            reconstruction @ matrix[target_index] / denominator
                        )
                    singular = np.linalg.svd(atoms, compute_uv=False)
                    smallest = float(singular.min())
                    conditions[target_index, step] = (
                        float(singular.max() / smallest) if smallest > 0 else np.inf
                    )

        # Frozen (or not-yet-updated) targets carry their previous path values
        # forward so every array is defined for all integer k up to k_max.
        for target_index in range(batch):
            if len(supports[target_index]) < step:
                errors[target_index, step] = errors[target_index, step - 1]
                residual_norms[target_index, step] = residual_norms[
                    target_index, step - 1
                ]
                j_norms[target_index, step] = j_norms[target_index, step - 1]
                cosines[target_index, step] = cosines[target_index, step - 1]
                conditions[target_index, step] = conditions[target_index, step - 1]

    results: list[PursuitResult] = []
    for target_index in range(batch):
        support = np.asarray(supports[target_index], dtype=np.int64)
        error_path = errors[target_index]
        results.append(
            PursuitResult(
                target_norm=float(target_norms[target_index]),
                support=support,
                errors=error_path,
                gains=error_path[:-1] - error_path[1:],
                coefficients_per_k=tuple(coefficients_paths[target_index]),
                residual_norms=residual_norms[target_index],
                j_component_norms=j_norms[target_index],
                cosine_with_target=cosines[target_index],
                condition_numbers=conditions[target_index],
                selected_atom_norms=atom_norms[support]
                if support.size
                else np.empty(0, dtype=np.float64),
                stopped_early_at=stopped_at[target_index],
                selection_mode=modes[target_index],
            )
        )
    return results
