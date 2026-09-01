"""Memory-bounded dictionary adapters for streaming sparse pursuit.

Every adapter exposes the same three operations over ``V`` atoms living in
``R^D`` (atoms are residual-space directions; for the token frame, atom ``t``
is row ``t`` of the UN-CENTERED ``A_l = U_eff J_l``):

- ``atom_norms()``: the ``[V]`` Euclidean atom norms (computed once, streamed);
- ``dots(residual)``: raw inner products ``<a_t, r>`` for every atom as ``[V]``;
- ``materialize(ids)``: the selected atoms as a dense ``[len(ids), D]`` array.

The production ``V x D`` matrix is never materialized: the token-frame adapter
computes ``A r`` as ``U_eff (J r)`` and norms/atoms one vocabulary chunk at a
time; the random control regenerates its Gaussian directions chunk-by-chunk
from counter-based NumPy Philox streams, so a fixed seed reproduces the exact
dictionary on any platform without storing it.

Centering is structurally impossible here and explicitly rejected: a centered
row is no longer the actual J-direction of its token.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

RANDOM_CONTROL_METHOD = "matched_gaussian_atom_norms_v1"
RANDOM_CONTROL_RNG = "numpy_philox_counter_v1"


class DictionaryError(ValueError):
    """Raised when a dictionary adapter is constructed or used incorrectly."""


def _reject_centering(kwargs: dict[str, Any], name: str) -> None:
    if kwargs.pop("centered", False):
        raise DictionaryError(
            f"{name} refuses centered atoms: a centered row is no longer the "
            "actual token J-direction; centering is only valid for spectral "
            "analysis"
        )
    if kwargs:
        raise DictionaryError(f"unknown {name} options: {sorted(kwargs)}")


class DenseDictionary:
    """Explicit ``[V, D]`` atom matrix for tests and tiny problems."""

    def __init__(self, atoms: object, **kwargs: Any) -> None:
        _reject_centering(dict(kwargs), type(self).__name__)
        matrix = np.asarray(atoms, dtype=np.float64)
        if matrix.ndim != 2 or 0 in matrix.shape:
            raise DictionaryError("atoms must be a non-empty [V, D] matrix")
        if not np.isfinite(matrix).all():
            raise DictionaryError("atoms must be finite")
        self._atoms = np.ascontiguousarray(matrix)

    @property
    def n_atoms(self) -> int:
        return int(self._atoms.shape[0])

    @property
    def d_model(self) -> int:
        return int(self._atoms.shape[1])

    def atom_norms(self) -> FloatArray:
        return np.linalg.norm(self._atoms, axis=1)

    def dots(self, residual: object) -> FloatArray:
        return self.dots_batch(np.asarray(residual, dtype=np.float64)[None, :])[:, 0]

    def dots_batch(self, residuals: object) -> FloatArray:
        batch = np.asarray(residuals, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != self.d_model:
            raise DictionaryError(f"residuals must have shape [B, {self.d_model}]")
        return self._atoms @ batch.T

    def materialize(self, ids: object) -> FloatArray:
        indices = np.asarray(ids, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise DictionaryError("ids must be a non-empty 1-D integer array")
        if indices.min() < 0 or indices.max() >= self.n_atoms:
            raise DictionaryError("ids out of range")
        return np.array(self._atoms[indices], dtype=np.float64)


class UnitNormDictionary:
    """Scale every non-zero atom of another dictionary to unit Euclidean norm.

    The wrapper preserves streaming behavior: full atom matrices are never
    materialized, and only the selected rows are divided during
    :meth:`materialize`.
    """

    def __init__(self, dictionary: Any) -> None:
        self._dictionary = dictionary
        norms = np.asarray(dictionary.atom_norms(), dtype=np.float64)
        if norms.shape != (int(dictionary.n_atoms),):
            raise DictionaryError("wrapped dictionary atom_norms shape mismatch")
        if not np.isfinite(norms).all() or np.any(norms < 0):
            raise DictionaryError("wrapped dictionary atom norms are invalid")
        self._norms = norms

    @property
    def n_atoms(self) -> int:
        return int(self._dictionary.n_atoms)

    @property
    def d_model(self) -> int:
        return int(self._dictionary.d_model)

    def atom_norms(self) -> FloatArray:
        return np.where(self._norms > 0.0, 1.0, 0.0)

    def dots(self, residual: object) -> FloatArray:
        return self.dots_batch(np.asarray(residual, dtype=np.float64)[None, :])[:, 0]

    def dots_batch(self, residuals: object) -> FloatArray:
        dots = np.asarray(self._dictionary.dots_batch(residuals), dtype=np.float64)
        return np.divide(
            dots,
            self._norms[:, None],
            out=np.zeros_like(dots),
            where=self._norms[:, None] > 0.0,
        )

    def materialize(self, ids: object) -> FloatArray:
        indices = np.asarray(ids, dtype=np.int64)
        atoms = np.asarray(self._dictionary.materialize(indices), dtype=np.float64)
        return atoms / self._norms[indices, None]


class TokenFrameDictionary:
    """Atoms are the UN-CENTERED rows of ``A_l = U_eff J_l`` (Torch-backed).

    ``effective`` is an :class:`~jlens_workspace.foundation.jacobian.EffectiveUnembedding`
    already restricted to real tokenizer rows; ``jacobian`` is the fitted
    ``[D, D]`` layer matrix. ``dots`` uses the associativity trick
    ``A r = U_eff (J r)`` so no ``V x D`` product is ever formed; norms and
    materialization stream vocabulary chunks.
    """

    def __init__(
        self,
        effective: Any,
        jacobian: Any,
        *,
        chunk_size: int = 4096,
        compute_device: str | None = None,
        **kwargs: Any,
    ) -> None:
        _reject_centering(dict(kwargs), type(self).__name__)
        import torch

        if chunk_size <= 0:
            raise DictionaryError("chunk_size must be positive")
        matrix = jacobian.detach() if hasattr(jacobian, "detach") else torch.as_tensor(jacobian)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise DictionaryError("jacobian must have shape [d_model, d_model]")
        if int(effective.d_model) != int(matrix.shape[0]):
            raise DictionaryError(
                f"unembedding d_model={effective.d_model} does not match "
                f"jacobian d_model={int(matrix.shape[0])}"
            )
        self._torch = torch
        self._effective = effective
        self._device = torch.device(compute_device) if compute_device else matrix.device
        self._jacobian = matrix.to(device=self._device, dtype=torch.float64)
        self._chunk_size = int(chunk_size)
        self._norms: FloatArray | None = None

    @property
    def n_atoms(self) -> int:
        return int(self._effective.vocab_size)

    @property
    def d_model(self) -> int:
        return int(self._jacobian.shape[0])

    def atom_norms(self) -> FloatArray:
        if self._norms is None:
            torch = self._torch
            norms = np.empty(self.n_atoms, dtype=np.float64)
            with torch.inference_mode():
                for start in range(0, self.n_atoms, self._chunk_size):
                    stop = min(start + self._chunk_size, self.n_atoms)
                    rows = self._effective.rows(
                        start, stop, device=self._device, dtype=torch.float64
                    )
                    block = rows @ self._jacobian
                    norms[start:stop] = (
                        torch.linalg.vector_norm(block, dim=1).cpu().numpy()
                    )
            self._norms = norms
        return self._norms

    def dots(self, residual: object) -> FloatArray:
        return self.dots_batch(np.asarray(residual, dtype=np.float64)[None, :])[:, 0]

    def dots_batch(self, residuals: object) -> FloatArray:
        torch = self._torch
        batch = np.asarray(residuals, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != self.d_model:
            raise DictionaryError(f"residuals must have shape [B, {self.d_model}]")
        with torch.inference_mode():
            matrix = torch.as_tensor(batch, device=self._device, dtype=torch.float64)
            # A R^T = U_eff (J R^T): the [V, D] product is never formed.
            projected = self._jacobian @ matrix.T
            outputs = np.empty((self.n_atoms, batch.shape[0]), dtype=np.float64)
            for start in range(0, self.n_atoms, self._chunk_size):
                stop = min(start + self._chunk_size, self.n_atoms)
                rows = self._effective.rows(
                    start, stop, device=self._device, dtype=torch.float64
                )
                outputs[start:stop] = (rows @ projected).cpu().numpy()
        return outputs

    def materialize(self, ids: object) -> FloatArray:
        torch = self._torch
        indices = np.asarray(ids, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise DictionaryError("ids must be a non-empty 1-D integer array")
        if indices.min() < 0 or indices.max() >= self.n_atoms:
            raise DictionaryError("ids out of range")
        with torch.inference_mode():
            index = torch.as_tensor(indices, device=self._effective.weight.device)
            rows = torch.index_select(self._effective.weight, 0, index).to(
                device=self._device, dtype=torch.float64
            )
            if self._effective.column_scale is not None:
                scale = self._effective.column_scale.to(
                    device=self._device, dtype=torch.float64
                )
                rows = rows * scale.unsqueeze(0)
            atoms = (rows @ self._jacobian).cpu().numpy()
        return np.asarray(atoms, dtype=np.float64)


def build_token_frame_dictionary(
    effective: Any,
    jacobian: Any,
    *,
    convention: str,
    chunk_size: int = 4096,
    compute_device: str | None = None,
) -> TokenFrameDictionary:
    """Construct the dictionary while re-checking the convention identity.

    ``effective.metadata.convention`` must equal the requested convention so a
    raw unembedding can never be silently combined with an RMSNorm-weighted
    analysis label (or vice versa).
    """

    observed = getattr(getattr(effective, "metadata", None), "convention", None)
    if observed != convention:
        raise DictionaryError(
            f"effective unembedding convention {observed!r} does not match the "
            f"requested dictionary convention {convention!r}"
        )
    return TokenFrameDictionary(
        effective,
        jacobian,
        chunk_size=chunk_size,
        compute_device=compute_device,
    )


class MatchedNormRandomDictionary:
    """``matched_gaussian_atom_norms_v1`` random control dictionary.

    Atom ``t`` is ``n_t * g_t / ||g_t||`` where ``n_t`` is the matched real
    atom norm and ``g_t ~ N(0, I_D)``. Directions are generated one vocabulary
    chunk at a time from ``numpy.random.Philox`` keyed by
    ``(seed, chunk_index)``: the dictionary is a pure function of
    ``(seed, d_model, chunk_size, atom order)``, identical across replays and
    platforms, and is never stored.
    """

    def __init__(
        self,
        atom_norms: object,
        *,
        seed: int,
        d_model: int,
        chunk_size: int = 4096,
        **kwargs: Any,
    ) -> None:
        _reject_centering(dict(kwargs), type(self).__name__)
        norms = np.asarray(atom_norms, dtype=np.float64)
        if norms.ndim != 1 or norms.size == 0:
            raise DictionaryError("atom_norms must be a non-empty [V] vector")
        if not np.isfinite(norms).all() or np.any(norms < 0):
            raise DictionaryError("atom_norms must be finite and non-negative")
        if seed < 0:
            raise DictionaryError("seed must be non-negative")
        if d_model <= 0 or chunk_size <= 0:
            raise DictionaryError("d_model and chunk_size must be positive")
        self._norms = norms
        self._seed = int(seed)
        self._d_model = int(d_model)
        self._chunk_size = int(chunk_size)

    @property
    def n_atoms(self) -> int:
        return int(self._norms.size)

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def seed(self) -> int:
        return self._seed

    def _unit_chunk(self, chunk_index: int, start: int, stop: int) -> FloatArray:
        generator = np.random.Generator(
            np.random.Philox(key=[self._seed, chunk_index])
        )
        gaussians = generator.standard_normal(
            size=(stop - start, self._d_model), dtype=np.float64
        )
        lengths = np.linalg.norm(gaussians, axis=1, keepdims=True)
        if np.any(lengths == 0):  # pragma: no cover - probability zero
            lengths[lengths == 0] = 1.0
        return gaussians / lengths

    def atom_norms(self) -> FloatArray:
        return self._norms.copy()

    def dots(self, residual: object) -> FloatArray:
        return self.dots_batch(np.asarray(residual, dtype=np.float64)[None, :])[:, 0]

    def dots_batch(self, residuals: object) -> FloatArray:
        batch = np.asarray(residuals, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != self._d_model:
            raise DictionaryError(f"residuals must have shape [B, {self._d_model}]")
        outputs = np.empty((self.n_atoms, batch.shape[0]), dtype=np.float64)
        for chunk_index, start in enumerate(range(0, self.n_atoms, self._chunk_size)):
            stop = min(start + self._chunk_size, self.n_atoms)
            units = self._unit_chunk(chunk_index, start, stop)
            outputs[start:stop] = (units @ batch.T) * self._norms[start:stop, None]
        return outputs

    def materialize(self, ids: object) -> FloatArray:
        indices = np.asarray(ids, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise DictionaryError("ids must be a non-empty 1-D integer array")
        if indices.min() < 0 or indices.max() >= self.n_atoms:
            raise DictionaryError("ids out of range")
        atoms = np.empty((indices.size, self._d_model), dtype=np.float64)
        order = np.argsort(indices, kind="stable")
        sorted_ids = indices[order]
        position = 0
        for chunk_index, start in enumerate(range(0, self.n_atoms, self._chunk_size)):
            stop = min(start + self._chunk_size, self.n_atoms)
            local: list[int] = []
            while position < sorted_ids.size and sorted_ids[position] < stop:
                local.append(int(sorted_ids[position]) - start)
                position += 1
            if local:
                units = self._unit_chunk(chunk_index, start, stop)
                rows = units[np.asarray(local)] * self._norms[
                    np.asarray(local) + start, None
                ]
                atoms[order[position - len(local) : position]] = rows
            if position == sorted_ids.size:
                break
        return atoms
