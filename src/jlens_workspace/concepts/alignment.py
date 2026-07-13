"""Concept-direction alignment and J-space decomposition utilities.

The large object in this module is the token-by-residual matrix
``A = W_U @ J`` with shape ``[vocabulary, d_residual]``.  Public helpers either
consume already materialised candidate rows or iterate vocabulary chunks; the
full production matrix is never constructed.

NumPy is the numerical interchange format.  PyTorch is imported lazily only
when a tensor is passed, and tensor chunks are detached and returned to CPU at
the API boundary.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


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


def _to_numpy(value: object, *, name: str, ndim: int) -> FloatArray:
    torch = _torch_module_for(value)
    if torch is not None:
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be array-like or a PyTorch tensor") from error
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {array.shape}")
    if 0 in array.shape:
        raise ValueError(f"{name} must be non-empty, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return np.ascontiguousarray(array)


def _shape_2d(value: object, *, name: str) -> tuple[int, int]:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(f"{name} must be two-dimensional")
    rows, columns = int(shape[0]), int(shape[1])
    if rows <= 0 or columns <= 0:
        raise ValueError(f"{name} must be non-empty, got shape {(rows, columns)}")
    return rows, columns


def _indices_to_numpy(value: object, *, name: str) -> IntArray:
    torch = _torch_module_for(value)
    if torch is not None:
        value = value.detach().cpu().numpy()
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {raw.shape}")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    return np.ascontiguousarray(raw, dtype=np.int64)


def _validate_chunk_size(chunk_size: int) -> None:
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")


def activation_covariance(
    activations: object,
    *,
    center: bool = True,
    ddof: int = 1,
    ridge: float = 0.0,
) -> FloatArray:
    """Estimate a residual-coordinate covariance matrix in float64.

    Args:
        activations: Matrix with shape ``[n_samples, d_residual]``.
        center: Subtract the sample mean before forming the Gram matrix.
        ddof: Divisor correction; the denominator is ``n_samples - ddof``.
        ridge: Optional non-negative value added to the covariance diagonal.
    """

    values = _to_numpy(activations, name="activations", ndim=2)
    if not isinstance(ddof, int) or isinstance(ddof, bool) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer")
    denominator = values.shape[0] - ddof
    if denominator <= 0:
        raise ValueError("n_samples must be greater than ddof")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and non-negative")

    centered = values - values.mean(axis=0, keepdims=True) if center else values
    covariance = centered.T @ centered / denominator
    if ridge:
        covariance.flat[:: covariance.shape[0] + 1] += ridge
    # Suppress roundoff asymmetry before this matrix is used as a metric.
    return np.ascontiguousarray((covariance + covariance.T) * 0.5)


@dataclass(frozen=True)
class TopKAlignment:
    """Highest-scoring row identifiers and their signed cosine scores."""

    indices: IntArray
    scores: FloatArray
    metric: str
    absolute: bool


def _metric_state(
    query: FloatArray, covariance: object | None
) -> tuple[FloatArray | None, FloatArray | None, float, str]:
    dimension = query.shape[0]
    if covariance is None:
        metric_array = None
        query_metric = None
        query_squared_norm = float(query @ query)
        metric_name = "cosine"
    else:
        covariance_shape = getattr(covariance, "shape", None)
        if covariance_shape is None or len(covariance_shape) not in (1, 2):
            raise ValueError("covariance must be a diagonal vector or square matrix")
        metric_array = _to_numpy(
            covariance, name="covariance", ndim=len(covariance_shape)
        )
        if metric_array.ndim == 1:
            if metric_array.shape != (dimension,):
                raise ValueError(
                    f"diagonal covariance has shape {metric_array.shape}, expected {(dimension,)}"
                )
            if np.any(metric_array < 0):
                raise ValueError("diagonal covariance entries must be non-negative")
            query_metric = query * metric_array
        elif metric_array.ndim == 2:
            if metric_array.shape != (dimension, dimension):
                raise ValueError(
                    "covariance must have shape "
                    f"{(dimension, dimension)}, got {metric_array.shape}"
                )
            if not np.allclose(metric_array, metric_array.T, rtol=1e-10, atol=1e-12):
                raise ValueError("covariance must be symmetric")
            query_metric = metric_array @ query
        else:  # pragma: no cover - _to_numpy enforces the requested ndim
            raise ValueError("covariance must be a diagonal vector or square matrix")
        query_squared_norm = float(query @ query_metric)
        metric_name = "covariance_cosine"

    tolerance = np.finfo(np.float64).eps * max(1.0, float(query @ query)) * dimension
    if query_squared_norm <= tolerance:
        raise ValueError("query has zero norm under the selected metric")
    return metric_array, query_metric, query_squared_norm, metric_name


def _cosine_chunk(
    query: FloatArray,
    rows: FloatArray,
    *,
    covariance: FloatArray | None,
    query_metric: FloatArray | None,
    query_squared_norm: float,
) -> FloatArray:
    if covariance is None:
        numerators = rows @ query
        row_squared_norms = np.einsum("ij,ij->i", rows, rows)
    else:
        if covariance.ndim == 1:
            row_squared_norms = np.einsum("ij,j,ij->i", rows, covariance, rows)
        else:
            row_squared_norms = np.einsum(
                "ij,ij->i", rows @ covariance, rows, optimize=True
            )
        assert query_metric is not None
        numerators = rows @ query_metric

    numerical_tolerance = np.finfo(np.float64).eps * np.maximum(
        1.0, np.einsum("ij,ij->i", rows, rows)
    )
    invalid_negative = row_squared_norms < -numerical_tolerance
    if np.any(invalid_negative):
        raise ValueError("covariance is not positive semidefinite on the candidate rows")
    row_squared_norms = np.maximum(row_squared_norms, 0.0)
    denominators = np.sqrt(row_squared_norms * query_squared_norm)
    scores = np.full(rows.shape[0], np.nan, dtype=np.float64)
    np.divide(numerators, denominators, out=scores, where=denominators > 0)
    # A true inner-product cosine lies in [-1, 1]; only clip floating error.
    return np.clip(scores, -1.0, 1.0)


def _merge_topk(
    current_ids: IntArray,
    current_scores: FloatArray,
    new_ids: IntArray,
    new_scores: FloatArray,
    *,
    k: int,
    absolute: bool,
) -> tuple[IntArray, FloatArray]:
    valid = np.isfinite(new_scores)
    ids = np.concatenate((current_ids, new_ids[valid]))
    scores = np.concatenate((current_scores, new_scores[valid]))
    if ids.size == 0:
        return ids, scores
    ranking_scores = np.abs(scores) if absolute else scores
    # lexsort uses its last key first: descending score, then lower identifier.
    order = np.lexsort((ids, -ranking_scores))[:k]
    return ids[order], scores[order]


def _topk_from_chunks(
    query: FloatArray,
    chunks: Iterator[tuple[IntArray, FloatArray]],
    *,
    k: int,
    absolute: bool,
    covariance: object | None,
) -> TopKAlignment:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    metric_array, query_metric, query_squared_norm, metric_name = _metric_state(
        query, covariance
    )
    best_ids = np.empty(0, dtype=np.int64)
    best_scores = np.empty(0, dtype=np.float64)
    for identifiers, rows in chunks:
        if rows.shape[1] != query.shape[0]:
            raise ValueError(
                f"candidate rows have width {rows.shape[1]}, expected {query.shape[0]}"
            )
        scores = _cosine_chunk(
            query,
            rows,
            covariance=metric_array,
            query_metric=query_metric,
            query_squared_norm=query_squared_norm,
        )
        best_ids, best_scores = _merge_topk(
            best_ids,
            best_scores,
            identifiers,
            scores,
            k=k,
            absolute=absolute,
        )
    return TopKAlignment(
        indices=best_ids, scores=best_scores, metric=metric_name, absolute=absolute
    )


def _row_chunks(
    rows: object, identifiers: IntArray, *, chunk_size: int
) -> Iterator[tuple[IntArray, FloatArray]]:
    row_count, _ = _shape_2d(rows, name="rows")
    for start in range(0, row_count, chunk_size):
        stop = min(start + chunk_size, row_count)
        yield identifiers[start:stop], _to_numpy(
            rows[start:stop], name="row chunk", ndim=2  # type: ignore[index]
        )


def _row_identifiers(row_count: int, row_ids: object | None) -> IntArray:
    if row_ids is None:
        return np.arange(row_count, dtype=np.int64)
    identifiers = _indices_to_numpy(row_ids, name="row_ids")
    if identifiers.shape[0] != row_count:
        raise ValueError(
            f"row_ids must have shape {(row_count,)}, got {identifiers.shape}"
        )
    if np.unique(identifiers).size != row_count:
        raise ValueError("row_ids must be unique")
    return identifiers


def chunked_cosine_topk(
    query: object,
    rows: object,
    *,
    k: int = 10,
    chunk_size: int = 4_096,
    absolute: bool = False,
    covariance: object | None = None,
    row_ids: object | None = None,
) -> TopKAlignment:
    """Find top cosine-aligned rows without copying all rows at once.

    With ``covariance=Sigma``, cosine is computed under the activation metric
    ``<x, y>_Sigma = x.T @ Sigma @ y``.  A length-``d`` covariance vector is
    interpreted as a diagonal matrix.  Zero-norm rows are omitted.  Ties are
    resolved by lower ``row_ids`` values.
    """

    _validate_chunk_size(chunk_size)
    query_array = _to_numpy(query, name="query", ndim=1)
    row_count, width = _shape_2d(rows, name="rows")
    if width != query_array.shape[0]:
        raise ValueError(f"rows have width {width}, expected {query_array.shape[0]}")
    identifiers = _row_identifiers(row_count, row_ids)
    return _topk_from_chunks(
        query_array,
        _row_chunks(rows, identifiers, chunk_size=chunk_size),
        k=k,
        absolute=absolute,
        covariance=covariance,
    )


def _token_j_chunks(
    unembedding: object,
    jacobian: object,
    identifiers: IntArray,
    *,
    chunk_size: int,
) -> Iterator[tuple[IntArray, FloatArray]]:
    vocabulary, _ = _shape_2d(unembedding, name="unembedding")
    torch_w = _torch_module_for(unembedding)
    torch_j = _torch_module_for(jacobian)
    if torch_w is not None and torch_j is not None:
        if unembedding.device != jacobian.device:  # type: ignore[union-attr]
            raise ValueError("unembedding and jacobian tensors must be on the same device")
        compute_dtype = (
            torch_w.float64
            if unembedding.dtype == torch_w.float64 or jacobian.dtype == torch_w.float64
            else torch_w.float32
        )
        jacobian_compute = jacobian.detach().to(dtype=compute_dtype)  # type: ignore[union-attr]
        for start in range(0, vocabulary, chunk_size):
            stop = min(start + chunk_size, vocabulary)
            transformed = unembedding[start:stop].detach().to(dtype=compute_dtype) @ jacobian_compute  # type: ignore[index,union-attr]
            yield identifiers[start:stop], np.asarray(
                transformed.cpu().numpy(), dtype=np.float64
            )
        return

    jacobian_array = _to_numpy(jacobian, name="jacobian", ndim=2)
    for start in range(0, vocabulary, chunk_size):
        stop = min(start + chunk_size, vocabulary)
        unembedding_chunk = _to_numpy(
            unembedding[start:stop], name="unembedding chunk", ndim=2  # type: ignore[index]
        )
        yield identifiers[start:stop], unembedding_chunk @ jacobian_array


def chunked_token_j_topk(
    query: object,
    unembedding: object,
    jacobian: object,
    *,
    k: int = 10,
    chunk_size: int = 4_096,
    absolute: bool = False,
    covariance: object | None = None,
    token_ids: object | None = None,
) -> TopKAlignment:
    """Find token J-directions most aligned with a residual-space query.

    ``unembedding`` has shape ``[vocabulary, d_output]`` and ``jacobian`` has
    shape ``[d_output, d_residual]``.  Multiplication occurs one vocabulary
    chunk at a time, including when both inputs are GPU tensors.
    """

    _validate_chunk_size(chunk_size)
    query_array = _to_numpy(query, name="query", ndim=1)
    vocabulary, output_width = _shape_2d(unembedding, name="unembedding")
    jacobian_rows, residual_width = _shape_2d(jacobian, name="jacobian")
    if output_width != jacobian_rows:
        raise ValueError(
            "unembedding width must equal jacobian row count, got "
            f"{output_width} and {jacobian_rows}"
        )
    if residual_width != query_array.shape[0]:
        raise ValueError(
            f"query has width {query_array.shape[0]}, expected {residual_width}"
        )
    identifiers = _row_identifiers(vocabulary, token_ids)
    return _topk_from_chunks(
        query_array,
        _token_j_chunks(
            unembedding, jacobian, identifiers, chunk_size=chunk_size
        ),
        k=k,
        absolute=absolute,
        covariance=covariance,
    )


def materialize_token_j_rows(
    unembedding: object, jacobian: object, token_indices: object
) -> FloatArray:
    """Materialise ``W_U[token_indices] @ J`` for a small candidate set only."""

    vocabulary, output_width = _shape_2d(unembedding, name="unembedding")
    jacobian_rows, _ = _shape_2d(jacobian, name="jacobian")
    if output_width != jacobian_rows:
        raise ValueError("unembedding width must equal jacobian row count")
    indices = _indices_to_numpy(token_indices, name="token_indices")
    if indices.size == 0:
        raise ValueError("token_indices must be non-empty")
    if np.any(indices < 0) or np.any(indices >= vocabulary):
        raise IndexError("token_indices contain an out-of-range token")

    torch_w = _torch_module_for(unembedding)
    torch_j = _torch_module_for(jacobian)
    if torch_w is not None and torch_j is not None:
        if unembedding.device != jacobian.device:  # type: ignore[union-attr]
            raise ValueError("unembedding and jacobian tensors must be on the same device")
        index_tensor = torch_w.as_tensor(indices, device=unembedding.device)
        selected = torch_w.index_select(unembedding, 0, index_tensor)
        compute_dtype = (
            torch_w.float64
            if selected.dtype == torch_w.float64 or jacobian.dtype == torch_w.float64
            else torch_w.float32
        )
        return np.asarray(
            (selected.to(dtype=compute_dtype) @ jacobian.detach().to(dtype=compute_dtype))
            .cpu()
            .numpy(),
            dtype=np.float64,
        )

    selected_numpy = _to_numpy(
        unembedding[indices], name="selected unembedding rows", ndim=2  # type: ignore[index]
    )
    jacobian_numpy = _to_numpy(jacobian, name="jacobian", ndim=2)
    return selected_numpy @ jacobian_numpy


# A shorter synonym useful when candidate rows are being prepared for NNLS.
compute_token_j_rows = materialize_token_j_rows


@dataclass(frozen=True)
class JSpaceDecomposition:
    """A non-negative candidate-row reconstruction and its non-J residual."""

    coefficients: FloatArray
    candidate_indices: IntArray
    selected_candidate_indices: IntArray
    j_component: FloatArray
    non_j_component: FloatArray
    residual_norm: float
    relative_residual_norm: float
    explained_energy: float


def sparse_nonnegative_decomposition(
    target: object,
    candidate_rows: object,
    *,
    candidate_indices: object | None = None,
    max_nonzero: int | None = None,
    coefficient_tol: float = 1e-10,
) -> JSpaceDecomposition:
    """Decompose ``target`` into sparse non-negative J rows plus a residual.

    The dense case uses non-negative least squares directly.  With
    ``max_nonzero``, a deterministic non-negative orthogonal matching pursuit
    selects rows by positive residual correlation and refits NNLS after every
    selection.  Coefficients always correspond to the unnormalised input rows.
    """

    target_array = _to_numpy(target, name="target", ndim=1)
    rows = _to_numpy(candidate_rows, name="candidate_rows", ndim=2)
    candidate_count, width = rows.shape
    if width != target_array.shape[0]:
        raise ValueError(
            f"candidate rows have width {width}, expected {target_array.shape[0]}"
        )
    identifiers = _row_identifiers(candidate_count, candidate_indices)
    if not np.isfinite(coefficient_tol) or coefficient_tol < 0:
        raise ValueError("coefficient_tol must be finite and non-negative")
    if max_nonzero is not None and (
        not isinstance(max_nonzero, int)
        or isinstance(max_nonzero, bool)
        or max_nonzero <= 0
    ):
        raise ValueError("max_nonzero must be a positive integer or None")

    coefficients = np.zeros(candidate_count, dtype=np.float64)
    if max_nonzero is None or max_nonzero >= candidate_count:
        coefficients, _ = nnls(rows.T, target_array)
        coefficients[coefficients <= coefficient_tol] = 0.0
    else:
        row_norms = np.linalg.norm(rows, axis=1)
        available = row_norms > 0
        active: list[int] = []
        residual = target_array.copy()
        # Rejected zero-coefficient atoms cannot help again without another
        # active-set change, so each loop either grows the set or removes one.
        for _ in range(candidate_count):
            if len(active) >= max_nonzero:
                break
            correlations = np.full(candidate_count, -np.inf, dtype=np.float64)
            correlations[available] = (rows[available] @ residual) / row_norms[available]
            positive = np.flatnonzero(correlations > coefficient_tol)
            if positive.size == 0:
                break
            order = np.lexsort((identifiers[positive], -correlations[positive]))
            chosen = int(positive[order[0]])
            trial_active = [*active, chosen]
            trial_coefficients, _ = nnls(rows[trial_active].T, target_array)
            keep = trial_coefficients > coefficient_tol
            active = [
                index
                for index, keep_it in zip(trial_active, keep, strict=True)
                if keep_it
            ]
            active_coefficients = trial_coefficients[keep]
            available[chosen] = False
            if active:
                residual = target_array - active_coefficients @ rows[active]
            else:
                residual = target_array.copy()
            # Active atoms remain unavailable so max_nonzero counts unique rows.
            available[active] = False
        if active:
            active_coefficients, _ = nnls(rows[active].T, target_array)
            coefficients[np.asarray(active, dtype=np.int64)] = active_coefficients
            coefficients[coefficients <= coefficient_tol] = 0.0

    j_component = coefficients @ rows
    non_j_component = target_array - j_component
    residual_norm = float(np.linalg.norm(non_j_component))
    target_norm = float(np.linalg.norm(target_array))
    if target_norm == 0:
        relative_residual_norm = 0.0
        explained_energy = 1.0
    else:
        relative_residual_norm = residual_norm / target_norm
        explained_energy = float(
            np.clip(1.0 - relative_residual_norm**2, 0.0, 1.0)
        )
    selected = np.flatnonzero(coefficients > coefficient_tol)
    return JSpaceDecomposition(
        coefficients=coefficients,
        candidate_indices=identifiers,
        selected_candidate_indices=identifiers[selected],
        j_component=np.asarray(j_component, dtype=np.float64),
        non_j_component=np.asarray(non_j_component, dtype=np.float64),
        residual_norm=residual_norm,
        relative_residual_norm=relative_residual_norm,
        explained_energy=explained_energy,
    )


# Domain-oriented synonym for callers that do not need the algorithm name.
decompose_j_space = sparse_nonnegative_decomposition


@dataclass(frozen=True)
class BasisCoverage:
    """Per-concept energy captured by the span of the first ``top_r`` rows."""

    top_r: int
    basis_rank: int
    per_concept: FloatArray
    mean_coverage: float


def top_r_basis_concept_coverage(
    concept_vectors: object, basis_vectors: object, *, top_r: int
) -> BasisCoverage:
    """Measure concept energy captured by an ordered top-r row basis.

    ``concept_vectors`` has shape ``[n_concepts, d_residual]`` (or ``[d]``)
    and ``basis_vectors`` has shape ``[n_basis, d_residual]`` with vectors in
    descending importance order.  The selected rows are orthonormalised, so
    duplicate or non-unit basis vectors cannot inflate coverage.
    """

    concept_shape = getattr(concept_vectors, "shape", None)
    if concept_shape is None:
        concept_shape = np.shape(concept_vectors)
    if len(concept_shape) == 1:
        concept = _to_numpy(concept_vectors, name="concept_vectors", ndim=1)
        concepts = concept[None, :]
    elif len(concept_shape) == 2:
        concepts = _to_numpy(concept_vectors, name="concept_vectors", ndim=2)
    else:
        raise ValueError(
            "concept_vectors must be a vector or a matrix, got shape "
            f"{tuple(concept_shape)}"
        )
    basis = _to_numpy(basis_vectors, name="basis_vectors", ndim=2)
    if concepts.shape[1] != basis.shape[1]:
        raise ValueError("concept and basis residual widths differ")
    if (
        not isinstance(top_r, int)
        or isinstance(top_r, bool)
        or top_r <= 0
        or top_r > basis.shape[0]
    ):
        raise ValueError(f"top_r must be between 1 and {basis.shape[0]}")

    _, singular_values, right_vectors = np.linalg.svd(
        basis[:top_r], full_matrices=False
    )
    if singular_values.size == 0:
        basis_rank = 0
    else:
        tolerance = (
            max(basis[:top_r].shape)
            * np.finfo(np.float64).eps
            * singular_values[0]
        )
        basis_rank = int(np.sum(singular_values > tolerance))
    orthonormal_rows = right_vectors[:basis_rank]
    concept_energy = np.einsum("ij,ij->i", concepts, concepts)
    if basis_rank:
        projection = concepts @ orthonormal_rows.T
        captured_energy = np.einsum("ij,ij->i", projection, projection)
    else:
        captured_energy = np.zeros(concepts.shape[0], dtype=np.float64)
    coverage = np.zeros(concepts.shape[0], dtype=np.float64)
    np.divide(
        captured_energy,
        concept_energy,
        out=coverage,
        where=concept_energy > 0,
    )
    coverage = np.clip(coverage, 0.0, 1.0)
    return BasisCoverage(
        top_r=top_r,
        basis_rank=basis_rank,
        per_concept=coverage,
        mean_coverage=float(coverage.mean()),
    )


basis_concept_coverage = top_r_basis_concept_coverage
