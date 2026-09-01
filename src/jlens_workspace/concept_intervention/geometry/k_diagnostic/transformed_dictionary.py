"""Streaming linear metric wrapper for uncentered token J-atoms.

For ``T`` with shape ``[R, D]`` and a wrapped dictionary whose atoms are rows
of ``A`` with shape ``[V, D]``, this adapter exposes the transformed atoms
``A @ T.T`` without materializing either production-sized matrix.  Selected
coefficients can still be applied to the original atoms so every transformed
solve reports its raw ``resid_post`` reconstruction as well.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class TransformedDictionaryError(ValueError):
    """A transform or wrapped dictionary violates the coordinate contract."""


class LinearTransformedDictionary:
    """Expose ``T a_t`` as streamed ``R``-dimensional dictionary atoms."""

    def __init__(
        self,
        dictionary: object,
        transform: object,
        *,
        norm_chunk_size: int = 4096,
    ) -> None:
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.ndim != 2 or 0 in matrix.shape:
            raise TransformedDictionaryError("transform must have non-empty shape [R, D]")
        if matrix.shape[1] != int(dictionary.d_model):
            raise TransformedDictionaryError(
                f"transform width {matrix.shape[1]} != raw d_model {dictionary.d_model}"
            )
        if not np.isfinite(matrix).all():
            raise TransformedDictionaryError("transform must be finite")
        if norm_chunk_size < 1:
            raise TransformedDictionaryError("norm_chunk_size must be positive")
        self._dictionary = dictionary
        self._transform = np.ascontiguousarray(matrix)
        self._norm_chunk_size = int(norm_chunk_size)
        self._norms: FloatArray | None = None

    @property
    def n_atoms(self) -> int:
        return int(self._dictionary.n_atoms)

    @property
    def d_model(self) -> int:
        """Transformed metric dimension ``R``."""

        return int(self._transform.shape[0])

    @property
    def raw_d_model(self) -> int:
        return int(self._transform.shape[1])

    @property
    def transform(self) -> FloatArray:
        return self._transform.copy()

    @property
    def transform_sha256(self) -> str:
        return hashlib.sha256(self._transform.tobytes(order="C")).hexdigest()

    def transform_target(self, target: object) -> FloatArray:
        vector = np.asarray(target, dtype=np.float64)
        if vector.shape != (self.raw_d_model,):
            raise TransformedDictionaryError(
                f"raw target must have shape [{self.raw_d_model}]"
            )
        return self._transform @ vector

    def atom_norms(self) -> FloatArray:
        if self._norms is None:
            norms = np.empty(self.n_atoms, dtype=np.float64)
            for start in range(0, self.n_atoms, self._norm_chunk_size):
                stop = min(start + self._norm_chunk_size, self.n_atoms)
                raw = np.asarray(
                    self._dictionary.materialize(np.arange(start, stop, dtype=np.int64)),
                    dtype=np.float64,
                )
                transformed = raw @ self._transform.T
                norms[start:stop] = np.linalg.norm(transformed, axis=1)
            self._norms = norms
        return self._norms.copy()

    def dots(self, residual: object) -> FloatArray:
        return self.dots_batch(np.asarray(residual, dtype=np.float64)[None, :])[:, 0]

    def dots_batch(self, residuals: object) -> FloatArray:
        values = np.asarray(residuals, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.d_model:
            raise TransformedDictionaryError(
                f"transformed residuals must have shape [B, {self.d_model}]"
            )
        # (A T^T) R^T = A (R T)^T.
        lifted = values @ self._transform
        return np.asarray(self._dictionary.dots_batch(lifted), dtype=np.float64)

    def materialize(self, ids: object) -> FloatArray:
        raw = np.asarray(self._dictionary.materialize(ids), dtype=np.float64)
        return raw @ self._transform.T

    def raw_materialize(self, ids: object) -> FloatArray:
        """Return selected untransformed ``resid_post`` atoms only."""

        return np.asarray(self._dictionary.materialize(ids), dtype=np.float64)

    def raw_reconstruction(self, ids: object, coefficients: object) -> FloatArray:
        atoms = self.raw_materialize(ids)
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (atoms.shape[0],):
            raise TransformedDictionaryError("coefficients must align with selected atoms")
        return values @ atoms
