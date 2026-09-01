"""Explicit null families for the registered K diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from jlens_workspace.concept_intervention.geometry.sparse_pursuit import (
    MatchedNormRandomDictionary,
)
from jlens_workspace.foundation.artifacts import atomic_write_json, sha256_file

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class NullModelError(ValueError):
    """A null construction would violate its preregistered semantics."""


HAAR_CACHE_ALGORITHM_VERSION = "gaussian_qr_haar_cached_v3_shared_dimension_seed"
SHARED_DIRECTION_PROVIDER_VERSION = "philox_shared_directions_v1"
SHARED_DIRECTION_STORAGE = "ephemeral_bundle_seed_ram"
# Execution-only safety cap: the production exact-float64 provider peaks at
# 5,248,421,888 bytes (4.888 GiB). Six GiB admits it while remaining far below
# the registered 64 GiB Slurm job memory; this is not a scientific/resource budget.
SHARED_DIRECTION_MAX_PEAK_BYTES = 6 * 2**30
SHARED_DIRECTION_MIN_AVAILABLE_RESERVE_BYTES = 8 * 2**30


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def haar_cache_identity(
    *, metric: str, transform_sha256: str | None, dimension: int, seed: int
) -> dict[str, object]:
    if not metric or dimension < 1 or seed < 0:
        raise NullModelError("invalid Haar cache identity")
    if transform_sha256 is not None and len(transform_sha256) != 64:
        raise NullModelError("Haar transform identity must be a SHA-256 digest")
    return {
        "algorithm_version": HAAR_CACHE_ALGORITHM_VERSION,
        "dimension": int(dimension),
        "seed": int(seed),
    }


def haar_cache_id(*, metric: str, transform_sha256: str | None, dimension: int, seed: int) -> str:
    identity = haar_cache_identity(
        metric=metric,
        transform_sha256=transform_sha256,
        dimension=dimension,
        seed=seed,
    )
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def _atomic_save_npy(path: Path, values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as handle:
            np.save(handle, np.asarray(values, dtype=np.float64), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_cached_haar_rotation(
    cache_root: str | Path,
    *,
    metric: str,
    transform_sha256: str | None,
    dimension: int,
    seed: int,
    create: bool = False,
    constructor: Any = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Create once or mmap an exact Haar matrix with immutable identity checks."""

    identity = haar_cache_identity(
        metric=metric,
        transform_sha256=transform_sha256,
        dimension=dimension,
        seed=seed,
    )
    cache_id = haar_cache_id(
        metric=metric,
        transform_sha256=transform_sha256,
        dimension=dimension,
        seed=seed,
    )
    root = Path(cache_root)
    directory = root / "matrices" / cache_id
    matrix_path = directory / "matrix.npy"
    manifest_path = directory / "manifest.json"
    complete_path = directory / "complete.json"

    def load() -> tuple[np.ndarray, dict[str, object]]:
        if not (manifest_path.is_file() and matrix_path.is_file() and complete_path.is_file()):
            raise NullModelError(f"partial Haar cache is not reusable: {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity or manifest.get("cache_id") != cache_id:
            raise NullModelError(f"Haar cache identity mismatch: {directory}")
        observed_hash = sha256_file(matrix_path)
        if (
            complete.get("complete") is not True
            or complete.get("cache_id") != cache_id
            or complete.get("matrix_sha256") != observed_hash
            or manifest.get("matrix_sha256") != observed_hash
        ):
            raise NullModelError(f"Haar cache hash mismatch: {directory}")
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        if matrix.shape != (dimension, dimension) or matrix.dtype != np.float64:
            raise NullModelError(f"Haar cache matrix shape/dtype mismatch: {directory}")
        return matrix, manifest

    if directory.exists():
        return load()
    if not create:
        raise NullModelError(f"registered Haar cache is missing: {directory}")

    lock_root = root / "locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock = lock_root / f"{cache_id}.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise NullModelError(f"Haar cache is already locked: {cache_id}") from error
    os.close(descriptor)
    try:
        if directory.exists():
            raise NullModelError(f"Haar cache appeared after lock: {directory}")
        directory.mkdir(parents=True, exist_ok=False)
        build = constructor or haar_orthogonal
        started = time.perf_counter()
        matrix, numerical_metadata = build(dimension, seed=seed)
        build_wall_seconds = time.perf_counter() - started
        _atomic_save_npy(matrix_path, matrix)
        matrix_hash = sha256_file(matrix_path)
        manifest = {
            "schema_version": 2,
            "cache_id": cache_id,
            "identity": identity,
            "matrix_sha256": matrix_hash,
            "build_wall_seconds": float(build_wall_seconds),
            "numerical_validation": numerical_metadata,
        }
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(
            complete_path,
            {
                "schema_version": 2,
                "complete": True,
                "cache_id": cache_id,
                "matrix_sha256": matrix_hash,
            },
        )
        return load()
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def haar_orthogonal(dimension: int, *, seed: int) -> tuple[FloatArray, dict[str, object]]:
    """Generate a deterministic Haar orthogonal matrix by Gaussian QR."""

    if dimension < 1 or seed < 0:
        raise NullModelError("dimension must be positive and seed non-negative")
    rng = np.random.Generator(np.random.Philox(seed))
    gaussian = rng.standard_normal((dimension, dimension), dtype=np.float64)
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q = np.ascontiguousarray(q * signs[None, :])
    identity = np.eye(dimension, dtype=np.float64)
    orthogonality_error = float(np.linalg.norm(q.T @ q - identity) / dimension)
    if orthogonality_error >= 1e-5:
        raise NullModelError(f"orthogonality error {orthogonality_error:.3e} exceeds 1e-5")
    return q, {
        "method": "gaussian_qr_haar_v1",
        "seed": int(seed),
        "dimension": int(dimension),
        "matrix_sha256": hashlib.sha256(q.tobytes(order="C")).hexdigest(),
        "normalized_frobenius_orthogonality_error": orthogonality_error,
    }


def inverse_rotate_target(target: object, rotation: object) -> tuple[FloatArray, float]:
    """Return ``Q.T @ w`` and its relative norm-preservation error."""

    vector = np.asarray(target, dtype=np.float64)
    q = np.asarray(rotation, dtype=np.float64)
    if q.shape != (vector.size, vector.size) or vector.ndim != 1:
        raise NullModelError("rotation/target shapes do not agree")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or norm == 0.0:
        raise NullModelError("rotation target must be finite and non-zero")
    rotated = q.T @ vector
    error = abs(float(np.linalg.norm(rotated)) - norm) / norm
    if error >= 1e-6:
        raise NullModelError(f"target norm preservation error {error:.3e} exceeds 1e-6")
    return rotated, error


def cardinality_for_fraction(vocabulary_size: int, fraction: float) -> int:
    if vocabulary_size < 1 or not 0.0 < fraction <= 1.0:
        raise NullModelError("invalid vocabulary size or dictionary fraction")
    return min(vocabulary_size, max(64, int(np.floor(fraction * vocabulary_size))))


def matched_norm_subset(
    atom_norms: object, *, fraction: float, seed: int
) -> tuple[FloatArray, dict[str, object]]:
    """Deterministically sample the cardinality-sweep norm multiset."""

    norms = np.asarray(atom_norms, dtype=np.float64)
    if norms.ndim != 1 or norms.size == 0 or not np.isfinite(norms).all():
        raise NullModelError("atom_norms must be a finite non-empty vector")
    if np.any(norms < 0) or seed < 0:
        raise NullModelError("atom norms and seed must be non-negative")
    count = cardinality_for_fraction(int(norms.size), float(fraction))
    if count == norms.size:
        indices = np.arange(norms.size, dtype=np.int64)
    else:
        rng = np.random.Generator(np.random.Philox(key=[int(seed), round(fraction * 2**32)]))
        indices = rng.choice(norms.size, size=count, replace=False)
    selected = np.ascontiguousarray(norms[indices])
    return selected, {
        "fraction": float(fraction),
        "actual_cardinality": int(count),
        "norm_subset_sha256": hashlib.sha256(selected.tobytes(order="C")).hexdigest(),
        "selection_seed": int(seed),
    }


def _available_memory_bytes() -> int:
    """Return Linux MemAvailable for a fail-closed allocation decision."""

    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, raw = line.split(":", maxsplit=1)
            fields[name] = int(raw.strip().split()[0]) * 1024
        available = int(fields["MemAvailable"])
    except (OSError, KeyError, ValueError, IndexError) as error:
        raise NullModelError("cannot determine available RAM for shared directions") from error
    if available <= 0:
        raise NullModelError("available RAM for shared directions is not positive")
    return available


def shared_direction_memory_plan(
    *,
    n_atoms: int,
    d_model: int,
    chunk_size: int,
    max_peak_bytes: int = SHARED_DIRECTION_MAX_PEAK_BYTES,
    available_bytes: int | None = None,
) -> dict[str, object]:
    """Gate the exact ephemeral [V,D] float64 direction provider.

    The peak includes the retained direction matrix plus one Gaussian chunk,
    one normalized chunk, and its row norms. This is an execution-only v2
    exception to the ordinary no-V-by-D-materialization rule: the matrix is
    never persisted as a scientific artifact and exists for one bundle/seed.
    """

    if n_atoms < 1 or d_model < 1 or chunk_size < 1 or max_peak_bytes < 1:
        raise NullModelError("shared-direction dimensions and budget must be positive")
    available = _available_memory_bytes() if available_bytes is None else int(available_bytes)
    if available < 1:
        raise NullModelError("shared-direction available RAM must be positive")
    rows = min(int(n_atoms), int(chunk_size))
    storage_bytes = int(n_atoms) * int(d_model) * np.dtype(np.float64).itemsize
    gaussian_chunk_bytes = rows * int(d_model) * np.dtype(np.float64).itemsize
    normalized_chunk_bytes = gaussian_chunk_bytes
    row_norm_bytes = rows * np.dtype(np.float64).itemsize
    peak_bytes = storage_bytes + gaussian_chunk_bytes + normalized_chunk_bytes + row_norm_bytes
    available_budget = max(0, available - SHARED_DIRECTION_MIN_AVAILABLE_RESERVE_BYTES)
    checks = {
        "within_hard_peak_budget": peak_bytes <= int(max_peak_bytes),
        "preserves_available_ram_reserve": peak_bytes <= available_budget,
    }
    return {
        "schema_version": 1,
        "provider_version": SHARED_DIRECTION_PROVIDER_VERSION,
        "storage": SHARED_DIRECTION_STORAGE,
        "n_atoms": int(n_atoms),
        "d_model": int(d_model),
        "chunk_size": int(chunk_size),
        "float64_direction_storage_bytes": storage_bytes,
        "generation_workspace_bytes": (
            gaussian_chunk_bytes + normalized_chunk_bytes + row_norm_bytes
        ),
        "peak_bytes": peak_bytes,
        "hard_peak_budget_bytes": int(max_peak_bytes),
        "observed_available_bytes": available,
        "reserved_available_bytes": SHARED_DIRECTION_MIN_AVAILABLE_RESERVE_BYTES,
        "effective_available_budget_bytes": available_budget,
        "checks": checks,
        "approved": all(checks.values()),
    }


class SharedDirectionProvider:
    """One exact Philox unit-direction matrix for one bundle/IID seed."""

    def __init__(
        self,
        *,
        n_atoms: int,
        d_model: int,
        seed: int,
        chunk_size: int,
        max_peak_bytes: int = SHARED_DIRECTION_MAX_PEAK_BYTES,
        available_bytes: int | None = None,
    ) -> None:
        if seed < 0:
            raise NullModelError("shared-direction seed must be non-negative")
        self.memory_plan = shared_direction_memory_plan(
            n_atoms=n_atoms,
            d_model=d_model,
            chunk_size=chunk_size,
            max_peak_bytes=max_peak_bytes,
            available_bytes=available_bytes,
        )
        if self.memory_plan["approved"] is not True:
            failed = sorted(
                name for name, passed in self.memory_plan["checks"].items() if not passed
            )
            raise NullModelError(f"shared-direction RAM preflight rejected: {failed}")
        self.n_atoms = int(n_atoms)
        self.d_model = int(d_model)
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        self._closed = False
        self._directions = np.empty((self.n_atoms, self.d_model), dtype=np.float64, order="C")
        digest = hashlib.sha256()
        self._generation_counts: dict[int, int] = {}
        for chunk_index, start in enumerate(range(0, self.n_atoms, self.chunk_size)):
            stop = min(start + self.chunk_size, self.n_atoms)
            generator = np.random.Generator(np.random.Philox(key=[self.seed, chunk_index]))
            gaussians = generator.standard_normal(
                size=(stop - start, self.d_model), dtype=np.float64
            )
            lengths = np.linalg.norm(gaussians, axis=1, keepdims=True)
            if np.any(lengths == 0):  # pragma: no cover - probability zero
                lengths[lengths == 0] = 1.0
            units = gaussians / lengths
            self._directions[start:stop] = units
            digest.update(np.ascontiguousarray(units).tobytes(order="C"))
            self._generation_counts[chunk_index] = 1
        self.raw_float64_sha256 = digest.hexdigest()

    @property
    def generation_counts(self) -> dict[int, int]:
        return dict(self._generation_counts)

    @property
    def generation_count(self) -> int:
        return sum(self._generation_counts.values())

    def unit_chunk(self, chunk_index: int, start: int, stop: int) -> FloatArray:
        if self._closed:
            raise NullModelError("shared-direction provider is closed")
        expected_start = int(chunk_index) * self.chunk_size
        if start != expected_start or not start < stop <= min(
            start + self.chunk_size, self.n_atoms
        ):
            raise NullModelError("shared-direction chunk request is not canonical")
        return self._directions[start:stop]

    def close(self) -> None:
        self._directions = np.empty((0, 0), dtype=np.float64)
        self._closed = True

    def __enter__(self) -> SharedDirectionProvider:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class SharedDirectionRandomDictionary:
    """Matched-norm random dictionary backed by one shared direction prefix."""

    def __init__(self, provider: SharedDirectionProvider, atom_norms: object) -> None:
        norms = np.asarray(atom_norms, dtype=np.float64)
        if norms.ndim != 1 or norms.size == 0 or norms.size > provider.n_atoms:
            raise NullModelError("shared random atom norms must be a provider prefix")
        if not np.isfinite(norms).all() or np.any(norms < 0):
            raise NullModelError("shared random atom norms must be finite and non-negative")
        self._provider = provider
        self._norms = np.ascontiguousarray(norms)

    @property
    def n_atoms(self) -> int:
        return int(self._norms.size)

    @property
    def d_model(self) -> int:
        return self._provider.d_model

    @property
    def seed(self) -> int:
        return self._provider.seed

    def atom_norms(self) -> FloatArray:
        return self._norms.copy()

    def dots(self, residual: object) -> FloatArray:
        return self.dots_batch(np.asarray(residual, dtype=np.float64)[None, :])[:, 0]

    def dots_batch(self, residuals: object) -> FloatArray:
        batch = np.asarray(residuals, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != self.d_model:
            raise NullModelError(f"shared random residuals must have shape [B, {self.d_model}]")
        outputs = np.empty((self.n_atoms, batch.shape[0]), dtype=np.float64)
        for chunk_index, start in enumerate(range(0, self.n_atoms, self._provider.chunk_size)):
            stop = min(start + self._provider.chunk_size, self.n_atoms)
            units = self._provider.unit_chunk(chunk_index, start, stop)
            outputs[start:stop] = (units @ batch.T) * self._norms[start:stop, None]
        return outputs

    def materialize(self, ids: object) -> FloatArray:
        indices = np.asarray(ids, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise NullModelError("shared random IDs must be a non-empty 1-D array")
        if indices.min() < 0 or indices.max() >= self.n_atoms:
            raise NullModelError("shared random IDs are out of range")
        atoms = np.empty((indices.size, self.d_model), dtype=np.float64)
        order = np.argsort(indices, kind="stable")
        sorted_ids = indices[order]
        position = 0
        for chunk_index, start in enumerate(range(0, self.n_atoms, self._provider.chunk_size)):
            stop = min(start + self._provider.chunk_size, self.n_atoms)
            local: list[int] = []
            while position < sorted_ids.size and sorted_ids[position] < stop:
                local.append(int(sorted_ids[position]) - start)
                position += 1
            if local:
                units = self._provider.unit_chunk(chunk_index, start, stop)
                local_array = np.asarray(local, dtype=np.int64)
                rows = units[local_array] * self._norms[local_array + start, None]
                atoms[order[position - len(local) : position]] = rows
            if position == sorted_ids.size:
                break
        return atoms


def random_prefix_atoms(
    atom_norms: object,
    *,
    k_max: int,
    seed: int,
    d_model: int,
) -> tuple[FloatArray, dict[str, object]]:
    """Construct exactly ``K_max`` matched-norm atoms, with no best-atom scan."""

    norms = np.asarray(atom_norms, dtype=np.float64)
    if k_max < 1 or k_max > norms.size:
        raise NullModelError("k_max must lie in [1, number of empirical norms]")
    rng = np.random.Generator(np.random.Philox(key=[seed, 991]))
    indices = rng.choice(norms.size, size=k_max, replace=False)
    selected = np.ascontiguousarray(norms[indices])
    dictionary = MatchedNormRandomDictionary(selected, seed=seed, d_model=d_model, chunk_size=k_max)
    atoms = dictionary.materialize(np.arange(k_max, dtype=np.int64))
    return atoms, {
        "null_family": "random_prefix_same_k",
        "statement": "This is not the full-cardinality random-search null.",
        "seed": int(seed),
        "k_max": int(k_max),
        "norm_subset_sha256": hashlib.sha256(selected.tobytes(order="C")).hexdigest(),
    }


@dataclass(frozen=True)
class PrefixPursuitResult:
    """Nested fixed-prefix coefficient path; support ``k`` is exactly ``0:k``."""

    errors: FloatArray
    gains: FloatArray
    coefficients_per_k: tuple[FloatArray, ...]
    reconstructions: FloatArray
    support_prefixes: tuple[IntArray, ...]


def _standard_nonnegative_update(
    atoms: FloatArray,
    target: FloatArray,
    previous: FloatArray,
) -> FloatArray:
    coefficients = np.concatenate((previous, np.zeros(1, dtype=np.float64)))
    residual = target - coefficients @ atoms
    gradient = atoms @ residual
    direction = gradient @ atoms
    denominator = float(direction @ direction)
    if denominator <= 0.0:
        return coefficients
    step = float(gradient @ gradient) / denominator
    projected = np.maximum(0.0, coefficients + step * gradient)
    delta = projected - coefficients
    reconstructed_delta = delta @ atoms
    delta_norm = float(reconstructed_delta @ reconstructed_delta)
    if delta_norm <= 0.0:
        return coefficients
    feasible_step = float(residual @ reconstructed_delta) / delta_norm
    return coefficients + min(1.0, max(0.0, feasible_step)) * delta


def random_prefix_pursuit(atoms: object, target: object) -> PrefixPursuitResult:
    """Add one predetermined prefix atom per step using the standard update."""

    matrix = np.asarray(atoms, dtype=np.float64)
    vector = np.asarray(target, dtype=np.float64)
    if matrix.ndim != 2 or vector.shape != (matrix.shape[1],) or matrix.shape[0] == 0:
        raise NullModelError("atoms/target must have shapes [K,D] and [D]")
    if not np.isfinite(matrix).all() or not np.isfinite(vector).all():
        raise NullModelError("atoms and target must be finite")
    squared_norm = float(vector @ vector)
    if squared_norm == 0.0:
        raise NullModelError("target must be non-zero")
    k_max = int(matrix.shape[0])
    errors = np.ones(k_max + 1, dtype=np.float64)
    reconstructions = np.zeros((k_max + 1, matrix.shape[1]), dtype=np.float64)
    paths: list[FloatArray] = [np.empty(0, dtype=np.float64)]
    supports: list[IntArray] = [np.empty(0, dtype=np.int64)]
    coefficients = paths[0]
    for k in range(1, k_max + 1):
        coefficients = _standard_nonnegative_update(matrix[:k], vector, coefficients)
        reconstruction = coefficients @ matrix[:k]
        error = float(np.sum((vector - reconstruction) ** 2) / squared_norm)
        if error > errors[k - 1] + 1e-10:
            raise NullModelError("random-prefix error increased")
        errors[k] = min(error, errors[k - 1])
        reconstructions[k] = reconstruction
        paths.append(coefficients.copy())
        supports.append(np.arange(k, dtype=np.int64))
    return PrefixPursuitResult(
        errors=errors,
        gains=errors[:-1] - errors[1:],
        coefficients_per_k=tuple(paths),
        reconstructions=reconstructions,
        support_prefixes=tuple(supports),
    )


def group_safe_permutation(
    labels: object,
    groups: object,
    *,
    seed: int,
    return_metadata: bool = False,
) -> IntArray | tuple[IntArray, dict[str, object]]:
    """Permute homogeneous group labels only among equally sized groups.

    The function operates on one already-selected split.  Callers invoke it
    independently for train and validation, so no test labels are accepted or
    inspected by this API.
    """

    values = np.asarray(labels)
    group_values = np.asarray(groups)
    if values.ndim != 1 or group_values.shape != values.shape or values.size == 0:
        raise NullModelError("labels and groups must share non-empty shape [N]")
    if set(np.unique(values).tolist()) != {0, 1}:
        raise NullModelError("group permutation requires binary labels")
    members: dict[object, NDArray[np.int64]] = {}
    for group in np.unique(group_values):
        indices = np.flatnonzero(group_values == group).astype(np.int64)
        if np.unique(values[indices]).size != 1:
            raise NullModelError(f"group {group!r} spans both labels")
        members[group.item() if hasattr(group, "item") else group] = indices
    by_size: dict[int, list[object]] = defaultdict(list)
    for group, indices in members.items():
        by_size[int(indices.size)].append(group)
    output = np.asarray(values, dtype=np.int64).copy()
    effective_groups = 0
    permutable_strata = 0
    for size, bucket in sorted(by_size.items()):
        ordered = sorted(bucket, key=str)
        bucket_labels = np.asarray([int(values[members[group][0]]) for group in ordered])
        if np.unique(bucket_labels).size < 2:
            continue
        permutable_strata += 1
        effective_groups += len(ordered)
        rng = np.random.Generator(np.random.Philox(key=[seed, size]))
        assigned = bucket_labels[rng.permutation(len(ordered))]
        for group, label in zip(ordered, assigned, strict=True):
            output[members[group]] = int(label)
    if int(output.sum()) != int(np.asarray(values, dtype=np.int64).sum()):
        raise NullModelError("group permutation changed the row-level class count")
    hamming = int(np.count_nonzero(output != values))
    if effective_groups == 0 or hamming == 0:
        raise NullModelError(
            "group permutation is identity; no effective equally-sized mixed-label stratum"
        )
    metadata: dict[str, object] = {
        "seed": int(seed),
        "row_count": int(values.size),
        "group_count": len(members),
        "effective_permutable_groups": int(effective_groups),
        "permutable_size_strata": int(permutable_strata),
        "hamming_distance": hamming,
        "non_identity_rate": float(hamming / values.size),
    }
    return (output, metadata) if return_metadata else output


def fit_fixed_c_permutation_probe(
    train_activations: object,
    train_labels: object,
    train_groups: object,
    validation_activations: object,
    validation_labels: object,
    validation_groups: object,
    *,
    C: float,
    seed: int,
    standardize: bool = True,
    class_weight: str | dict[Any, float] | None = "balanced",
    solver: str = "lbfgs",
    penalty: str = "l2",
    max_iter: int = 5000,
) -> tuple[FloatArray, dict[str, object]]:
    """Fit the fixed-C conditional null without accepting any test inputs."""

    from jlens_workspace.concept_intervention.probing import (
        fit_fixed_logistic_direction,
    )

    train_x = np.asarray(train_activations, dtype=np.float64)
    validation_x = np.asarray(validation_activations, dtype=np.float64)
    if train_x.ndim != 2 or validation_x.ndim != 2:
        raise NullModelError("train and validation activations must be [N,D]")
    if train_x.shape[1] != validation_x.shape[1]:
        raise NullModelError("train and validation residual widths differ")
    train_permuted, train_metadata = group_safe_permutation(
        train_labels, train_groups, seed=seed, return_metadata=True
    )
    validation_permuted, validation_metadata = group_safe_permutation(
        validation_labels,
        validation_groups,
        seed=seed + 1,
        return_metadata=True,
    )
    labels = np.concatenate((train_permuted, validation_permuted))
    activations = np.concatenate((train_x, validation_x), axis=0)
    if not standardize or class_weight != "balanced" or solver != "lbfgs" or penalty != "l2":
        raise NullModelError("permutation probe must preserve standardize/balanced/L2-LBFGS parity")
    fitted = fit_fixed_logistic_direction(
        activations,
        labels,
        C=C,
        positive_label=1,
        standardize=standardize,
        class_weight=class_weight,
        random_state=seed,
        max_iter=max_iter,
        solver=solver,
        penalty=penalty,
    )
    vector = np.asarray(fitted.coef_raw, dtype=np.float64)
    return vector, {
        "null_family": "label_permutation_probe",
        "conditional_test": "fixed-C conditional null",
        "C": float(C),
        "seed": int(seed),
        "standardize": standardize,
        "class_weight": class_weight,
        "solver": solver,
        "penalty": penalty,
        "max_iter": int(max_iter),
        "pipeline_parity_validated": True,
        "fit_split": "permuted train+validation",
        "test_labels_accessed": False,
        "train_row_class_count": int(train_permuted.sum()),
        "validation_row_class_count": int(validation_permuted.sum()),
        "train_permutation": train_metadata,
        "validation_permutation": validation_metadata,
    }


def permuted_class_mean_difference(
    train_activations: object,
    train_labels: object,
    train_groups: object,
    validation_activations: object,
    validation_labels: object,
    validation_groups: object,
    *,
    seed: int,
) -> tuple[FloatArray, dict[str, object]]:
    """Recompute a positive-minus-negative mean after group-safe permutation."""

    train_x = np.asarray(train_activations, dtype=np.float64)
    validation_x = np.asarray(validation_activations, dtype=np.float64)
    if train_x.ndim != 2 or validation_x.ndim != 2 or train_x.shape[1] != validation_x.shape[1]:
        raise NullModelError("train and validation activations must share shape [N,D]")
    train_y, train_metadata = group_safe_permutation(
        train_labels, train_groups, seed=seed, return_metadata=True
    )
    validation_y, validation_metadata = group_safe_permutation(
        validation_labels,
        validation_groups,
        seed=seed + 1,
        return_metadata=True,
    )
    activations = np.concatenate((train_x, validation_x), axis=0)
    labels = np.concatenate((train_y, validation_y))
    positive = np.mean(activations[labels == 1], axis=0, dtype=np.float64)
    negative = np.mean(activations[labels == 0], axis=0, dtype=np.float64)
    vector = np.asarray(positive - negative, dtype=np.float64)
    if not np.isfinite(vector).all() or float(vector @ vector) == 0.0:
        raise NullModelError("permuted class-mean direction is zero or non-finite")
    return vector, {
        "null_family": "label_permutation_probe",
        "conditional_test": "group-safe recomputed class-mean null",
        "seed": int(seed),
        "fit_split": "permuted train+validation",
        "test_labels_accessed": False,
        "train_permutation": train_metadata,
        "validation_permutation": validation_metadata,
    }
