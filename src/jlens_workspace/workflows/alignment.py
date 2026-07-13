"""Probe-to-J token alignment and sparse J/non-J decomposition workflow."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from numpy.typing import NDArray

from jlens_workspace.artifacts import atomic_write_json
from jlens_workspace.concepts import (
    activation_covariance,
    chunked_token_j_topk,
    materialize_token_j_rows,
    sparse_nonnegative_decomposition,
)


def _atomic_save_npy(path: Path, value: NDArray[np.float64]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        with open(temporary_name, "wb") as handle:
            np.save(handle, np.asarray(value, dtype=np.float64), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _token_label(tokenizer: Any | None, token_id: int) -> dict[str, Any]:
    result: dict[str, Any] = {"token_id": int(token_id)}
    if tokenizer is None:
        return result
    try:
        result["token"] = tokenizer.convert_ids_to_tokens(int(token_id))
        result["decoded"] = tokenizer.decode([int(token_id)])
    except (AttributeError, TypeError, ValueError):
        result["token"] = None
        result["decoded"] = None
    return result


def run_probe_j_alignment(
    *,
    probe_vector: object,
    unembedding: object,
    jacobian: object,
    output_dir: str | Path,
    tokenizer: Any | None = None,
    activations: object | None = None,
    top_k: int = 50,
    candidate_pool_size: int = 512,
    sparse_components: int = 16,
    chunk_size: int = 4096,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Search token-J rows, then decompose a probe over the positive candidates.

    ``unembedding`` and ``jacobian`` may be NumPy or same-device Torch tensors.
    For the RMSNorm-weighted convention pass ``diag(gamma) @ J`` as
    ``jacobian``; this avoids allocating a second ``V x D`` unembedding.
    """

    query = np.asarray(probe_vector, dtype=np.float64)
    if query.ndim != 1 or not np.isfinite(query).all() or not np.linalg.norm(query):
        raise ValueError("probe_vector must be a finite non-zero [d_model] vector")
    if candidate_pool_size < top_k:
        raise ValueError("candidate_pool_size must be at least top_k")
    if sparse_components > candidate_pool_size:
        raise ValueError("sparse_components cannot exceed candidate_pool_size")

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"alignment output already exists: {destination}")
    if destination.exists() and overwrite:
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    ordinary = chunked_token_j_topk(
        query,
        unembedding,
        jacobian,
        k=candidate_pool_size,
        chunk_size=chunk_size,
    )
    negative = chunked_token_j_topk(
        -query,
        unembedding,
        jacobian,
        k=top_k,
        chunk_size=chunk_size,
    )
    covariance = None
    covariance_alignment = None
    if activations is not None:
        covariance = activation_covariance(activations, center=True, ridge=1e-8)
        covariance_alignment = chunked_token_j_topk(
            query,
            unembedding,
            jacobian,
            k=top_k,
            chunk_size=chunk_size,
            covariance=covariance,
        )

    candidate_ids = ordinary.indices
    candidate_rows = materialize_token_j_rows(unembedding, jacobian, candidate_ids)
    decomposition = sparse_nonnegative_decomposition(
        query,
        candidate_rows,
        candidate_indices=candidate_ids,
        max_nonzero=sparse_components,
    )
    _atomic_save_npy(destination / "j_component.npy", decomposition.j_component)
    _atomic_save_npy(destination / "non_j_component.npy", decomposition.non_j_component)
    _atomic_save_npy(destination / "coefficients.npy", decomposition.coefficients)
    if covariance is not None:
        _atomic_save_npy(destination / "activation_covariance.npy", covariance)

    coefficient_by_token = {
        int(token_id): float(coefficient)
        for token_id, coefficient in zip(
            decomposition.candidate_indices,
            decomposition.coefficients,
            strict=True,
        )
        if coefficient > 0
    }

    def alignment_payload(alignment: Any, *, negate_score: bool = False) -> list[dict[str, Any]]:
        rows = []
        for token_id, score in zip(alignment.indices, alignment.scores, strict=True):
            token = _token_label(tokenizer, int(token_id))
            token["score"] = -float(score) if negate_score else float(score)
            if int(token_id) in coefficient_by_token:
                token["nnls_coefficient"] = coefficient_by_token[int(token_id)]
            rows.append(token)
        return rows

    payload = {
        "schema_version": 1,
        "metadata": metadata or {},
        "probe_norm": float(np.linalg.norm(query)),
        "top_positive": alignment_payload(ordinary)[:top_k],
        "top_negative": alignment_payload(negative, negate_score=True),
        "top_covariance": (
            None if covariance_alignment is None else alignment_payload(covariance_alignment)
        ),
        "decomposition": {
            "candidate_pool_size": int(candidate_ids.size),
            "max_nonzero": sparse_components,
            "selected_token_ids": [
                int(value) for value in decomposition.selected_candidate_indices
            ],
            "selected_count": int(decomposition.selected_candidate_indices.size),
            "residual_norm": decomposition.residual_norm,
            "relative_residual_norm": decomposition.relative_residual_norm,
            "explained_energy": decomposition.explained_energy,
            "j_component_file": "j_component.npy",
            "non_j_component_file": "non_j_component.npy",
            "coefficients_file": "coefficients.npy",
        },
    }
    atomic_write_json(destination / "alignment.json", payload)
    return payload


run_alignment_workflow = run_probe_j_alignment


def _merge_topk(
    old_ids: NDArray[np.int64],
    old_scores: NDArray[np.float64],
    new_ids: NDArray[np.int64],
    new_scores: NDArray[np.float64],
    k: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    ids = np.concatenate((old_ids, new_ids))
    scores = np.concatenate((old_scores, new_scores))
    finite = np.isfinite(scores)
    ids, scores = ids[finite], scores[finite]
    order = np.lexsort((ids, -scores))[:k]
    return ids[order], scores[order]


def _operator_candidate_rows(operator: Any, token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
    """Materialize only selected token rows from a TokenFrameOperator."""

    import torch

    effective = operator.unembedding
    index = torch.as_tensor(token_ids, dtype=torch.long, device=effective.weight.device)
    rows = torch.index_select(effective.weight, 0, index).to(
        device=operator.compute_device, dtype=operator.compute_dtype
    )
    if effective.column_scale is not None:
        scale = effective.column_scale.to(device=rows.device, dtype=rows.dtype)
        rows = rows * scale.unsqueeze(0)
    jacobian = operator.jacobian.to(device=rows.device, dtype=rows.dtype)
    return rows.matmul(jacobian).detach().to(device="cpu", dtype=torch.float64).numpy()


def run_batched_probe_j_alignment(
    *,
    probe_vectors: Mapping[str, object],
    operator: Any,
    output_dir: str | Path,
    tokenizer: Any | None = None,
    top_k: int = 50,
    candidate_pool_size: int = 512,
    sparse_components: int = 16,
    decompose: bool = True,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Align many probes with one vocabulary pass for a single layer.

    This is the production path: token-frame rows are formed once per chunk,
    scored against all probes as a matrix, and discarded. Only the union of
    final candidate rows is materialized only when sparse decomposition is
    explicitly enabled. With ``decompose=False``, the output is strictly the
    positive/negative cosine top-k requested for semantic inspection.
    """

    if not probe_vectors:
        raise ValueError("probe_vectors must not be empty")
    if decompose and (
        candidate_pool_size < top_k or sparse_components > candidate_pool_size
    ):
        raise ValueError("require sparse_components <= candidate_pool_size and top_k")
    names = tuple(sorted(probe_vectors))
    queries = np.stack([np.asarray(probe_vectors[name], dtype=np.float64) for name in names])
    if queries.ndim != 2 or queries.shape[1] != operator.d_model:
        raise ValueError(f"probe vectors must have width d_model={operator.d_model}")
    query_norms = np.linalg.norm(queries, axis=1)
    if not np.isfinite(queries).all() or np.any(query_norms == 0):
        raise ValueError("probe vectors must be finite and non-zero")

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"alignment output already exists: {destination}")
    if destination.exists() and overwrite:
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    empty_ids = np.empty(0, dtype=np.int64)
    empty_scores = np.empty(0, dtype=np.float64)
    positive = {name: (empty_ids.copy(), empty_scores.copy()) for name in names}
    negative = {name: (empty_ids.copy(), empty_scores.copy()) for name in names}
    for block in operator.iter_rows():
        rows = block.rows.detach().to(device="cpu").double().numpy()
        row_norms = np.linalg.norm(rows, axis=1)
        scores = rows @ queries.T
        denominator = row_norms[:, None] * query_norms[None, :]
        np.divide(scores, denominator, out=scores, where=denominator > 0)
        scores[denominator == 0] = -np.inf
        ids = np.arange(block.start, block.stop, dtype=np.int64)
        for column, name in enumerate(names):
            positive[name] = _merge_topk(
                *positive[name],
                ids,
                scores[:, column],
                candidate_pool_size if decompose else top_k,
            )
            negative[name] = _merge_topk(
                *negative[name], ids, -scores[:, column], top_k
            )

    union_lookup: dict[int, NDArray[np.float64]] = {}
    if decompose:
        union_ids = np.unique(np.concatenate([positive[name][0] for name in names]))
        union_rows = _operator_candidate_rows(operator, union_ids)
        union_lookup = {
            int(token_id): row
            for token_id, row in zip(union_ids, union_rows, strict=True)
        }
    results: dict[str, Any] = {}
    for index, name in enumerate(names):
        candidate_ids, positive_scores = positive[name]
        concept_dir = destination / quote(name, safe="")
        decomposition_payload = None
        coefficients: dict[int, float] = {}
        if decompose:
            candidate_rows = np.stack(
                [union_lookup[int(token_id)] for token_id in candidate_ids]
            )
            decomposition = sparse_nonnegative_decomposition(
                queries[index],
                candidate_rows,
                candidate_indices=candidate_ids,
                max_nonzero=sparse_components,
            )
            _atomic_save_npy(concept_dir / "j_component.npy", decomposition.j_component)
            _atomic_save_npy(
                concept_dir / "non_j_component.npy", decomposition.non_j_component
            )
            _atomic_save_npy(concept_dir / "coefficients.npy", decomposition.coefficients)
            coefficients = {
                int(token_id): float(coefficient)
                for token_id, coefficient in zip(
                    decomposition.candidate_indices,
                    decomposition.coefficients,
                    strict=True,
                )
                if coefficient > 0
            }
            decomposition_payload = {
                "candidate_pool_size": int(candidate_ids.size),
                "max_nonzero": sparse_components,
                "selected_token_ids": [
                    int(value) for value in decomposition.selected_candidate_indices
                ],
                "selected_count": int(decomposition.selected_candidate_indices.size),
                "residual_norm": decomposition.residual_norm,
                "relative_residual_norm": decomposition.relative_residual_norm,
                "explained_energy": decomposition.explained_energy,
                "j_component_file": "j_component.npy",
                "non_j_component_file": "non_j_component.npy",
                "coefficients_file": "coefficients.npy",
            }

        def tokens(
            ids: NDArray[np.int64],
            scores: NDArray[np.float64],
            coefficients: dict[int, float] = coefficients,
        ) -> list[dict[str, Any]]:
            output = []
            for token_id, score in zip(ids, scores, strict=True):
                row = _token_label(tokenizer, int(token_id))
                row["score"] = float(score)
                if int(token_id) in coefficients:
                    row["nnls_coefficient"] = coefficients[int(token_id)]
                output.append(row)
            return output

        negative_ids, negative_magnitudes = negative[name]
        payload = {
            "schema_version": 1,
            "probe_id": name,
            "probe_norm": float(query_norms[index]),
            "metadata": metadata or {},
            "operator": operator.metadata.to_dict(),
            "top_positive": tokens(candidate_ids[:top_k], positive_scores[:top_k]),
            "top_negative": tokens(negative_ids, -negative_magnitudes),
            "decomposition": decomposition_payload,
        }
        atomic_write_json(concept_dir / "alignment.json", payload)
        results[name] = payload

    root = {
        "schema_version": 1,
        "workflow": "batched_probe_j_alignment",
        "probe_ids": list(names),
        "vocabulary_passes": 1,
        "decomposition_enabled": decompose,
        "operator": operator.metadata.to_dict(),
        "metadata": metadata or {},
        "results": {name: f"{quote(name, safe='')}/alignment.json" for name in names},
    }
    atomic_write_json(destination / "alignment.json", root)
    return results
