"""Atomic, provenance-rich workflow for layerwise J-lens matrix spectra."""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from jlens_workspace.artifacts import atomic_write_json
from jlens_workspace.jacobian import EffectiveUnembedding, ManagedJacobianLens
from jlens_workspace.jacobian._optional import torch
from jlens_workspace.matrix import (
    TokenFrameOperator,
    basis_coverage,
    decompose_gram,
    minimum_energy_basis,
    streaming_gram,
)


class MatrixWorkflowError(RuntimeError):
    """Layerwise matrix analysis could not produce a valid artifact."""


@dataclass(frozen=True)
class MatrixWorkflowOptions:
    """Numerical and storage choices for :func:`run_matrix_layers`."""

    centered: bool = True
    row_normalized: bool = False
    row_weights: Any | None = field(default=None, repr=False, compare=False)
    normalize_by_total_weight: bool = False
    zero_row_policy: Literal["error", "skip"] = "error"
    block_size: int = 4096
    compute_device: str = "auto"
    compute_dtype: Literal["float32", "float64", "float16", "bfloat16"] = (
        "float32"
    )
    accumulation_device: str = "cpu"
    decomposition_device: str = "cpu"
    cpu_fallback: bool = True
    energy_thresholds: tuple[float, ...] = (0.9, 0.95, 0.99)
    rank_atol: float = 0.0
    rank_rtol: float | None = None
    rank_rtol_sweep: tuple[float, ...] = (1e-5, 1e-6, 1e-7, 1e-8)

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        thresholds = tuple(float(value) for value in self.energy_thresholds)
        if not thresholds or any(not 0 < value <= 1 for value in thresholds):
            raise ValueError("energy_thresholds must be non-empty and lie in (0, 1]")
        if len(set(thresholds)) != len(thresholds):
            raise ValueError("energy_thresholds must be unique")
        object.__setattr__(self, "energy_thresholds", tuple(sorted(thresholds)))
        if self.rank_atol < 0 or (
            self.rank_rtol is not None and self.rank_rtol < 0
        ):
            raise ValueError("rank tolerances must be non-negative")
        sweep = tuple(float(value) for value in self.rank_rtol_sweep)
        if not sweep or any(value <= 0 for value in sweep) or len(set(sweep)) != len(sweep):
            raise ValueError("rank_rtol_sweep must contain unique positive values")
        if self.rank_rtol is not None and self.rank_rtol not in sweep:
            raise ValueError("rank_rtol_sweep must include rank_rtol")
        object.__setattr__(self, "rank_rtol_sweep", tuple(sorted(sweep, reverse=True)))


@dataclass(frozen=True)
class MatrixLayerOutput:
    layer: int
    directory: Path
    metrics_path: Path
    singular_values_path: Path
    eigenvalues_path: Path
    numerical_rank_basis_path: Path
    basis_paths: Mapping[float, Path]
    numerical_rank: int
    entropy_effective_rank: float
    participation_ratio: float


@dataclass(frozen=True)
class MatrixWorkflowResult:
    output_dir: Path
    metrics_path: Path
    subspace_comparisons_path: Path
    layers: tuple[MatrixLayerOutput, ...]
    provenance: Mapping[str, object]


def _atomic_save_npy(path: Path, array: NDArray[np.float64]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _torch_dtype(name: str):
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype {name!r}") from exc


def _choose_layers(
    lens: ManagedJacobianLens, layers: Sequence[int]
) -> tuple[int, ...]:
    selected = tuple(int(layer) for layer in layers)
    if not selected:
        raise ValueError("layers must not be empty")
    if len(set(selected)) != len(selected):
        raise ValueError("layers must be unique")
    unknown = sorted(set(selected) - set(lens.source_layers))
    if unknown:
        raise ValueError(
            f"layers {unknown} are not fitted; available layers are {lens.source_layers}"
        )
    return selected


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.8f}".rstrip("0").rstrip(".")


def _threshold_slug(threshold: float) -> str:
    return _threshold_key(threshold).replace(".", "p")


def _rank_sensitivity(
    singular_values: NDArray[np.float64], options: MatrixWorkflowOptions
) -> dict[str, dict[str, float | int]]:
    largest = float(singular_values[0]) if singular_values.size else 0.0
    return {
        f"{rtol:.0e}": {
            "relative_tolerance": rtol,
            "absolute_tolerance": max(options.rank_atol, rtol * largest),
            "numerical_rank": int(
                np.sum(singular_values > max(options.rank_atol, rtol * largest))
            ),
        }
        for rtol in options.rank_rtol_sweep
    }


def _weights_metadata(weights: Any | None, vocab_size: int) -> dict[str, object]:
    if weights is None:
        return {"provided": False, "sha256_float64": None, "shape": None}
    tensor = torch.as_tensor(weights, dtype=torch.float64, device="cpu")
    if tensor.ndim != 1 or int(tensor.shape[0]) != vocab_size:
        # ``streaming_gram`` would also reject this, but fail before creating
        # any layer files and provide provenance-specific context.
        raise ValueError(f"row_weights must have shape [{vocab_size}]")
    contiguous = tensor.contiguous().numpy()
    return {
        "provided": True,
        "sha256_float64": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        "shape": [vocab_size],
        "dtype_for_hash": "float64",
    }


def _unembedding_model_provenance(
    lens: ManagedJacobianLens, unembedding: EffectiveUnembedding
) -> dict[str, object]:
    """Validate any available unembedding identity against the fitted lens."""

    observed = {
        "model_id": unembedding.metadata.model_id,
        "model_revision": unembedding.metadata.model_revision,
    }
    expected = {
        "model_id": lens.metadata.model_id,
        "model_revision": lens.metadata.model_revision,
    }
    missing = [name for name, value in observed.items() if value is None]
    for name, value in observed.items():
        if value is not None and value != expected[name]:
            raise ValueError(
                f"effective unembedding {name}={value!r} differs from "
                f"lens {name}={expected[name]!r}"
            )
    verified = not missing
    return {
        "status": "verified" if verified else "unverified",
        "verified": verified,
        "model_id": observed["model_id"],
        "model_revision": observed["model_revision"],
        "expected_model_id": expected["model_id"],
        "expected_model_revision": expected["model_revision"],
        "missing_fields": missing,
    }


def _layer_metrics(
    *,
    layer: int,
    operator: TokenFrameOperator,
    gram: Any,
    spectrum: Any,
    options: MatrixWorkflowOptions,
    singular_values_file: str,
    eigenvalues_file: str,
    numerical_rank_basis_file: str,
    mean_file: str | None,
    energy: Mapping[str, Mapping[str, object]],
    rank_sensitivity: Mapping[str, Mapping[str, float | int]],
    weights_metadata: Mapping[str, object],
    unembedding_provenance: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "layer": layer,
        "coordinate": "resid_post/block_output",
        "matrix": {
            "definition": "A_l = U_eff J_l",
            "shape": [operator.vocab_size, operator.d_model],
            "convention": operator.unembedding.metadata.convention,
            "centered": options.centered,
            "row_normalized": options.row_normalized,
            "weighted": gram.weighted,
            "normalize_by_total_weight": options.normalize_by_total_weight,
            "zero_row_policy": options.zero_row_policy,
        },
        "dtypes": {
            "jacobian": str(operator.jacobian.dtype),
            "unembedding": str(operator.unembedding.weight.dtype),
            "operator_compute": str(operator.compute_dtype),
            "gram_accumulation": gram.accumulation_dtype,
            "eigendecomposition": spectrum.decomposition_dtype,
            "saved_arrays": "float64",
        },
        # Gram construction may fall back per vocabulary block after the
        # operator is created, so persist its dynamic execution metadata rather
        # than only the static requested device.
        "operator": dict(gram.operator_metadata),
        "provenance": {
            "effective_unembedding_model_identity": dict(unembedding_provenance)
        },
        "gram": {
            "n_rows": gram.n_rows,
            "n_positive_weight_rows": gram.n_positive_weight_rows,
            "total_weight": gram.total_weight,
            "zero_rows": gram.zero_rows,
            "accumulation_device": gram.accumulation_device,
            "mean_file": mean_file,
            "row_weights": dict(weights_metadata),
        },
        "spectrum": {
            "singular_values_file": singular_values_file,
            "eigenvalues_file": eigenvalues_file,
            "eigenvalues_definition": (
                "eigenvalues of the analyzed Gram matrix A_variant.T @ A_variant"
            ),
            "numerical_rank": spectrum.numerical_rank,
            "rank_tolerance": spectrum.rank_tolerance,
            "rank_sensitivity": dict(rank_sensitivity),
            "numerical_rank_basis_file": numerical_rank_basis_file,
            "numerical_rank_basis_definition": (
                "right-singular row-space basis retained above rank_tolerance"
            ),
            "numerical_rank_basis_shape": [
                operator.d_model,
                spectrum.numerical_rank,
            ],
            "entropy_effective_rank": spectrum.entropy_effective_rank,
            "participation_ratio": spectrum.participation_ratio,
            "stable_rank": spectrum.stable_rank,
            "total_energy": spectrum.total_energy,
            "decomposition_device": spectrum.decomposition_device,
            "used_cpu_fallback": spectrum.used_cpu_fallback,
        },
        "energy_bases": dict(energy),
    }


def run_matrix_layers(
    managed_lens: ManagedJacobianLens,
    effective_unembedding: EffectiveUnembedding,
    layers: Sequence[int],
    output_dir: str | Path,
    options: MatrixWorkflowOptions | Mapping[str, Any] | None = None,
) -> MatrixWorkflowResult:
    """Analyze and atomically persist spectra for selected fitted layers.

    The destination must not already exist.  Each layer contains
    ``metrics.json``, singular/eigenvalue arrays, the complete numerical row-space
    basis at the recorded rank tolerance, an optional ``center_mean.npy``, and the
    minimum orthonormal basis for every requested energy threshold. Arrays are
    plain float64 NumPy files written with ``allow_pickle=False``.
    """

    if not isinstance(managed_lens, ManagedJacobianLens):
        raise TypeError(
            "managed_lens must couple official matrices to validated model/tokenizer metadata"
        )
    if options is None:
        options = MatrixWorkflowOptions()
    elif isinstance(options, Mapping):
        options = MatrixWorkflowOptions(**options)
    elif not isinstance(options, MatrixWorkflowOptions):
        raise TypeError("options must be MatrixWorkflowOptions, a mapping, or None")
    selected_layers = _choose_layers(managed_lens, layers)
    if effective_unembedding.d_model != managed_lens.d_model:
        raise ValueError(
            f"effective unembedding d_model={effective_unembedding.d_model} differs "
            f"from lens d_model={managed_lens.d_model}"
        )
    unembedding_provenance = _unembedding_model_provenance(
        managed_lens, effective_unembedding
    )
    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"matrix workflow output already exists: {destination}") from exc

    outputs: list[MatrixLayerOutput] = []
    energy_basis_arrays: dict[tuple[int, float], NDArray[np.float64]] = {}
    try:
        weights_metadata = _weights_metadata(
            options.row_weights, effective_unembedding.vocab_size
        )
        compute_dtype = _torch_dtype(options.compute_dtype)
        for layer in selected_layers:
            layer_directory = destination / f"layer_{layer:03d}"
            layer_directory.mkdir()
            operator = TokenFrameOperator.from_lens(
                managed_lens,
                layer,
                effective_unembedding,
                block_size=options.block_size,
                compute_device=options.compute_device,
                compute_dtype=compute_dtype,
                cpu_fallback=options.cpu_fallback,
            )
            gram = streaming_gram(
                operator,
                centered=options.centered,
                row_normalized=options.row_normalized,
                row_weights=options.row_weights,
                normalize_by_total_weight=options.normalize_by_total_weight,
                zero_row_policy=options.zero_row_policy,
                accumulation_device=options.accumulation_device,
                cpu_fallback=options.cpu_fallback,
            )
            spectrum = decompose_gram(
                gram,
                decomposition_device=options.decomposition_device,
                cpu_fallback=options.cpu_fallback,
                atol=options.rank_atol,
                rtol=options.rank_rtol,
            )
            singular_values_path = layer_directory / "singular_values.npy"
            _atomic_save_npy(
                singular_values_path,
                spectrum.singular_values.detach().cpu().numpy().astype(
                    np.float64, copy=False
                ),
            )
            singular_values = np.asarray(
                spectrum.singular_values.detach().cpu().numpy(), dtype=np.float64
            )
            eigenvalues_path = layer_directory / "eigenvalues.npy"
            _atomic_save_npy(
                eigenvalues_path,
                spectrum.eigenvalues.detach().cpu().numpy().astype(
                    np.float64, copy=False
                ),
            )
            numerical_rank_basis_path = (
                layer_directory / "basis_numerical_rank.npy"
            )
            _atomic_save_npy(
                numerical_rank_basis_path,
                spectrum.right_basis[:, : spectrum.numerical_rank]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False),
            )
            mean_path = None
            if gram.mean is not None:
                mean_path = layer_directory / "center_mean.npy"
                _atomic_save_npy(
                    mean_path,
                    gram.mean.detach().cpu().numpy().astype(np.float64, copy=False),
                )

            basis_paths: dict[float, Path] = {}
            energy_metrics: dict[str, Mapping[str, object]] = {}
            for threshold in options.energy_thresholds:
                energy_basis = minimum_energy_basis(spectrum, threshold)
                basis_path = (
                    layer_directory / f"basis_energy_{_threshold_slug(threshold)}.npy"
                )
                _atomic_save_npy(
                    basis_path,
                    energy_basis.basis.detach().cpu().numpy().astype(
                        np.float64, copy=False
                    ),
                )
                energy_basis_arrays[(layer, threshold)] = np.asarray(
                    energy_basis.basis.detach().cpu().numpy(), dtype=np.float64
                )
                basis_paths[threshold] = basis_path
                energy_metrics[_threshold_key(threshold)] = {
                    "basis_file": basis_path.name,
                    "n_components": energy_basis.n_components,
                    "requested_energy": energy_basis.requested_energy,
                    "captured_energy": energy_basis.captured_energy,
                }
            metrics_path = layer_directory / "metrics.json"
            atomic_write_json(
                metrics_path,
                _layer_metrics(
                    layer=layer,
                    operator=operator,
                    gram=gram,
                    spectrum=spectrum,
                    options=options,
                    singular_values_file=singular_values_path.name,
                    eigenvalues_file=eigenvalues_path.name,
                    numerical_rank_basis_file=numerical_rank_basis_path.name,
                    mean_file=mean_path.name if mean_path is not None else None,
                    energy=energy_metrics,
                    rank_sensitivity=_rank_sensitivity(singular_values, options),
                    weights_metadata=weights_metadata,
                    unembedding_provenance=unembedding_provenance,
                ),
            )
            outputs.append(
                MatrixLayerOutput(
                    layer=layer,
                    directory=layer_directory,
                    metrics_path=metrics_path,
                    singular_values_path=singular_values_path,
                    eigenvalues_path=eigenvalues_path,
                    numerical_rank_basis_path=numerical_rank_basis_path,
                    basis_paths=basis_paths,
                    numerical_rank=spectrum.numerical_rank,
                    entropy_effective_rank=spectrum.entropy_effective_rank,
                    participation_ratio=spectrum.participation_ratio,
                )
            )

        comparisons_path = destination / "subspace_comparisons.json"
        comparison_pairs: list[dict[str, object]] = []
        for layer_a, layer_b in itertools.combinations(selected_layers, 2):
            thresholds_payload: dict[str, object] = {}
            for threshold in options.energy_thresholds:
                comparison = basis_coverage(
                    torch.from_numpy(energy_basis_arrays[(layer_a, threshold)]),
                    torch.from_numpy(energy_basis_arrays[(layer_b, threshold)]),
                    assume_orthonormal=True,
                )
                thresholds_payload[_threshold_key(threshold)] = {
                    "dim_a": comparison.dim_a,
                    "dim_b": comparison.dim_b,
                    "a_covered_by_b": comparison.a_covered_by_b,
                    "b_covered_by_a": comparison.b_covered_by_a,
                    "shared_dimension": comparison.shared_dimension,
                    "principal_angles_degrees": [
                        math.degrees(float(value))
                        for value in comparison.angles_radians.cpu().numpy()
                    ],
                }
            comparison_pairs.append(
                {
                    "layer_a": layer_a,
                    "layer_b": layer_b,
                    "energy_bases": thresholds_payload,
                }
            )
        atomic_write_json(
            comparisons_path,
            {
                "schema_version": 1,
                "workflow": "matrix_layer_subspace_comparisons",
                "coordinate": "resid_post/block_output",
                "pairs": comparison_pairs,
            },
        )

        root_metrics_path = destination / "metrics.json"
        atomic_write_json(
            root_metrics_path,
            {
                "schema_version": 1,
                "workflow": "matrix_layers",
                "subspace_comparisons_file": comparisons_path.name,
                "layers": [
                    {
                        "layer": output.layer,
                        "metrics_file": str(output.metrics_path.relative_to(destination)),
                        "singular_values_file": str(
                            output.singular_values_path.relative_to(destination)
                        ),
                        "eigenvalues_file": str(
                            output.eigenvalues_path.relative_to(destination)
                        ),
                        "numerical_rank_basis_file": str(
                            output.numerical_rank_basis_path.relative_to(destination)
                        ),
                    }
                    for output in outputs
                ],
                "lens_metadata": managed_lens.metadata.to_dict(),
                "effective_unembedding": effective_unembedding.metadata.to_dict(),
                "provenance": {
                    "effective_unembedding_model_identity": dict(
                        unembedding_provenance
                    )
                },
                "options": {
                    "centered": options.centered,
                    "row_normalized": options.row_normalized,
                    "normalize_by_total_weight": options.normalize_by_total_weight,
                    "zero_row_policy": options.zero_row_policy,
                    "block_size": options.block_size,
                    "compute_device": options.compute_device,
                    "compute_dtype": options.compute_dtype,
                    "accumulation_device": options.accumulation_device,
                    "decomposition_device": options.decomposition_device,
                    "cpu_fallback": options.cpu_fallback,
                    "energy_thresholds": list(options.energy_thresholds),
                    "rank_atol": options.rank_atol,
                    "rank_rtol": options.rank_rtol,
                    "rank_rtol_sweep": list(options.rank_rtol_sweep),
                    "row_weights": weights_metadata,
                },
            },
        )
        return MatrixWorkflowResult(
            output_dir=destination,
            metrics_path=root_metrics_path,
            subspace_comparisons_path=comparisons_path,
            layers=tuple(outputs),
            provenance={
                "effective_unembedding_model_identity": dict(
                    unembedding_provenance
                )
            },
        )
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
