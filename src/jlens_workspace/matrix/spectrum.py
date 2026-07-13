"""Streaming float64 Gram construction and spectral summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from jlens_workspace.jacobian._optional import torch

from .operator import MatrixDeviceError, TokenFrameOperator, _is_cuda_oom


class SpectralAnalysisError(RuntimeError):
    """A Gram matrix or eigendecomposition failed numerical validation."""


@dataclass(frozen=True)
class GramResult:
    gram: torch.Tensor
    mean: torch.Tensor | None
    n_rows: int
    n_positive_weight_rows: int
    total_weight: float
    centered: bool
    row_normalized: bool
    weighted: bool
    normalized_by_total_weight: bool
    zero_row_policy: str
    zero_rows: int
    accumulation_device: str
    accumulation_dtype: str
    operator_metadata: dict[str, object]

    @property
    def d_model(self) -> int:
        return int(self.gram.shape[0])


@dataclass(frozen=True)
class SpectrumResult:
    """Descending eigensystem of ``A.T A`` and rank statistics."""

    eigenvalues: torch.Tensor
    singular_values: torch.Tensor
    right_basis: torch.Tensor
    numerical_rank: int
    rank_tolerance: float
    entropy_effective_rank: float
    participation_ratio: float
    stable_rank: float
    total_energy: float
    n_rows: int
    centered: bool
    row_normalized: bool
    weighted: bool
    normalized_by_total_weight: bool
    decomposition_device: str
    decomposition_dtype: str
    used_cpu_fallback: bool

    @property
    def effective_rank(self) -> float:
        """Entropy effective rank, with participation ratio kept separate."""

        return self.entropy_effective_rank

    def minimum_energy_basis(self, threshold: float) -> EnergyBasis:
        return minimum_energy_basis(self, threshold)


@dataclass(frozen=True)
class EnergyBasis:
    basis: torch.Tensor
    n_components: int
    requested_energy: float
    captured_energy: float
    total_energy: float


def _prepare_weights(
    row_weights: torch.Tensor | Sequence[float] | None, vocab_size: int
) -> torch.Tensor | None:
    if row_weights is None:
        return None
    weights = torch.as_tensor(row_weights, dtype=torch.float64, device="cpu")
    if weights.ndim != 1 or int(weights.shape[0]) != vocab_size:
        raise ValueError(f"row_weights must have shape [{vocab_size}]")
    if not torch.isfinite(weights).all():
        raise ValueError("row_weights must be finite")
    if bool((weights < 0).any()):
        raise ValueError("row_weights must be non-negative")
    if float(weights.sum()) <= 0:
        raise ValueError("row_weights must have positive total weight")
    return weights


def _accumulation_device(
    requested: torch.device | str,
    *,
    cpu_fallback: bool,
) -> tuple[torch.device, bool]:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        if not cpu_fallback:
            raise MatrixDeviceError(
                f"requested accumulation device {device}, but CUDA is unavailable"
            )
        return torch.device("cpu"), True
    return device, False


def _normalized_rows(
    rows: torch.Tensor,
    weights: torch.Tensor,
    *,
    row_normalized: bool,
    zero_row_policy: Literal["error", "skip"],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not row_normalized:
        return rows, weights, 0
    norms = torch.linalg.vector_norm(rows, dim=1)
    zeros = norms == 0
    positive_weight_zeros = zeros & (weights > 0)
    n_zeros = int(positive_weight_zeros.sum().item())
    if n_zeros and zero_row_policy == "error":
        raise SpectralAnalysisError(
            f"encountered {n_zeros} positive-weight zero rows during normalization"
        )
    if n_zeros:
        weights = weights.clone()
        weights[positive_weight_zeros] = 0
    safe_norms = torch.where(zeros, torch.ones_like(norms), norms)
    return rows / safe_norms.unsqueeze(1), weights, n_zeros


def streaming_gram(
    operator: TokenFrameOperator,
    *,
    centered: bool = False,
    row_normalized: bool = False,
    row_weights: torch.Tensor | Sequence[float] | None = None,
    normalize_by_total_weight: bool = False,
    zero_row_policy: Literal["error", "skip"] = "error",
    accumulation_device: torch.device | str = "cpu",
    cpu_fallback: bool = True,
) -> GramResult:
    """Compute a weighted Gram/covariance matrix with float64 accumulation.

    The default is the uncentered, unnormalized ``A.T @ A`` whose eigenvalue
    square roots match the singular values of explicit ``A``.  With weights,
    this is ``A.T diag(w) A``.  Centering uses the weighted mean.  Setting
    ``normalize_by_total_weight`` divides the final matrix by ``sum(w)`` and is
    recorded in the result because it rescales singular values.
    """

    if zero_row_policy not in {"error", "skip"}:
        raise ValueError("zero_row_policy must be 'error' or 'skip'")
    weights_all = _prepare_weights(row_weights, operator.vocab_size)
    acc_device, initial_fallback = _accumulation_device(
        accumulation_device, cpu_fallback=cpu_fallback
    )
    stream_cpu_fallback = False

    def block_weights(start: int, stop: int, device: torch.device) -> torch.Tensor:
        if weights_all is None:
            return torch.ones(stop - start, dtype=torch.float64, device=device)
        return weights_all[start:stop].to(device=device, dtype=torch.float64)

    mean = None
    total_weight = 0.0
    positive_rows = 0
    zero_rows = 0
    if centered:
        weighted_sum = torch.zeros(
            operator.d_model, dtype=torch.float64, device=acc_device
        )
        try:
            for block in operator.iter_rows():
                stream_cpu_fallback = (
                    stream_cpu_fallback or block.used_cpu_fallback
                )
                rows = block.rows.to(device=acc_device, dtype=torch.float64)
                weights = block_weights(block.start, block.stop, acc_device)
                rows, weights, zeros = _normalized_rows(
                    rows,
                    weights,
                    row_normalized=row_normalized,
                    zero_row_policy=zero_row_policy,
                )
                zero_rows += zeros
                weighted_sum += (rows * weights.unsqueeze(1)).sum(dim=0)
                total_weight += float(weights.sum().item())
                positive_rows += int((weights > 0).sum().item())
        except RuntimeError as exc:
            if not (cpu_fallback and acc_device.type == "cuda" and _is_cuda_oom(exc)):
                raise
            # Restart the pass on CPU; partial sums must not be reused.
            torch.cuda.empty_cache()
            return streaming_gram(
                operator,
                centered=centered,
                row_normalized=row_normalized,
                row_weights=weights_all,
                normalize_by_total_weight=normalize_by_total_weight,
                zero_row_policy=zero_row_policy,
                accumulation_device="cpu",
                cpu_fallback=False,
            )
        if total_weight <= 0:
            raise SpectralAnalysisError("no positive-weight rows remain")
        mean = weighted_sum / total_weight

    gram = torch.zeros(
        (operator.d_model, operator.d_model),
        dtype=torch.float64,
        device=acc_device,
    )
    if not centered:
        total_weight = 0.0
        positive_rows = 0
        zero_rows = 0
    try:
        for block in operator.iter_rows():
            stream_cpu_fallback = stream_cpu_fallback or block.used_cpu_fallback
            rows = block.rows.to(device=acc_device, dtype=torch.float64)
            weights = block_weights(block.start, block.stop, acc_device)
            rows, weights, zeros = _normalized_rows(
                rows,
                weights,
                row_normalized=row_normalized,
                zero_row_policy=zero_row_policy,
            )
            if not centered:
                zero_rows += zeros
                total_weight += float(weights.sum().item())
                positive_rows += int((weights > 0).sum().item())
            if mean is not None:
                rows = rows - mean.unsqueeze(0)
            weighted_rows = rows * torch.sqrt(weights).unsqueeze(1)
            gram.addmm_(weighted_rows.T, weighted_rows)
    except RuntimeError as exc:
        if not (cpu_fallback and acc_device.type == "cuda" and _is_cuda_oom(exc)):
            raise
        torch.cuda.empty_cache()
        return streaming_gram(
            operator,
            centered=centered,
            row_normalized=row_normalized,
            row_weights=weights_all,
            normalize_by_total_weight=normalize_by_total_weight,
            zero_row_policy=zero_row_policy,
            accumulation_device="cpu",
            cpu_fallback=False,
        )
    if total_weight <= 0:
        raise SpectralAnalysisError("no positive-weight rows remain")
    # Remove insignificant asymmetry introduced by blockwise floating point.
    gram = (gram + gram.T) * 0.5
    if normalize_by_total_weight:
        gram = gram / total_weight
    return GramResult(
        gram=gram,
        mean=mean,
        n_rows=operator.vocab_size,
        n_positive_weight_rows=positive_rows,
        total_weight=total_weight,
        centered=centered,
        row_normalized=row_normalized,
        weighted=weights_all is not None,
        normalized_by_total_weight=normalize_by_total_weight,
        zero_row_policy=zero_row_policy,
        zero_rows=zero_rows,
        accumulation_device=str(acc_device),
        accumulation_dtype=str(torch.float64),
        operator_metadata=operator.metadata.to_dict()
        | {
            "accumulation_initial_cpu_fallback": initial_fallback,
            "stream_cpu_fallback_used": stream_cpu_fallback,
        },
    )


def decompose_gram(
    result: GramResult,
    *,
    decomposition_device: torch.device | str = "cpu",
    cpu_fallback: bool = True,
    atol: float = 0.0,
    rtol: float | None = None,
    negative_eigenvalue_tolerance: float | None = None,
) -> SpectrumResult:
    """Use symmetric ``eigh`` to obtain singular values and right directions."""

    device, initial_fallback = _accumulation_device(
        decomposition_device, cpu_fallback=cpu_fallback
    )
    used_fallback = initial_fallback
    gram = result.gram.to(device=device, dtype=torch.float64)
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    except RuntimeError as exc:
        if not (cpu_fallback and device.type == "cuda" and _is_cuda_oom(exc)):
            raise
        torch.cuda.empty_cache()
        device = torch.device("cpu")
        used_fallback = True
        eigenvalues, eigenvectors = torch.linalg.eigh(
            result.gram.to(device="cpu", dtype=torch.float64)
        )
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)
    scale = max(1.0, float(eigenvalues.abs().max().item()))
    if negative_eigenvalue_tolerance is None:
        negative_eigenvalue_tolerance = (
            100.0 * torch.finfo(torch.float64).eps * result.d_model * scale
        )
    minimum = float(eigenvalues.min().item())
    if minimum < -negative_eigenvalue_tolerance:
        raise SpectralAnalysisError(
            f"Gram matrix has eigenvalue {minimum:.3e} below allowed numerical "
            f"tolerance {-negative_eigenvalue_tolerance:.3e}"
        )
    eigenvalues = eigenvalues.clamp_min(0)
    singular_values = torch.sqrt(eigenvalues)
    largest = float(singular_values[0].item()) if singular_values.numel() else 0.0
    if rtol is None:
        # Forming A.T @ A squares the condition number.  Its eigenvalue
        # roundoff is O(eps * sigma_max**2), so a singular-value tolerance of
        # O(sqrt(eps) * sigma_max) is the resolvable default; the much smaller
        # direct-SVD tolerance would report roundoff eigenvalues as full rank.
        rtol = math.sqrt(
            max(result.n_rows, result.d_model) * torch.finfo(torch.float64).eps
        )
    if rtol < 0 or atol < 0:
        raise ValueError("atol and rtol must be non-negative")
    rank_tolerance = max(float(atol), float(rtol) * largest)
    numerical_rank = int((singular_values > rank_tolerance).sum().item())
    total_energy_tensor = eigenvalues.sum()
    total_energy = float(total_energy_tensor.item())
    if total_energy > 0:
        probabilities = eigenvalues / total_energy_tensor
        positive = probabilities > 0
        entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
        entropy_effective_rank = float(torch.exp(entropy).item())
        participation_ratio = float(
            (total_energy_tensor.square() / eigenvalues.square().sum()).item()
        )
        stable_rank = float((total_energy_tensor / eigenvalues[0]).item())
    else:
        entropy_effective_rank = 0.0
        participation_ratio = 0.0
        stable_rank = 0.0
    return SpectrumResult(
        eigenvalues=eigenvalues,
        singular_values=singular_values,
        right_basis=eigenvectors,
        numerical_rank=numerical_rank,
        rank_tolerance=rank_tolerance,
        entropy_effective_rank=entropy_effective_rank,
        participation_ratio=participation_ratio,
        stable_rank=stable_rank,
        total_energy=total_energy,
        n_rows=result.n_rows,
        centered=result.centered,
        row_normalized=result.row_normalized,
        weighted=result.weighted,
        normalized_by_total_weight=result.normalized_by_total_weight,
        decomposition_device=str(device),
        decomposition_dtype=str(torch.float64),
        used_cpu_fallback=used_fallback,
    )


def minimum_energy_basis(spectrum: SpectrumResult, threshold: float) -> EnergyBasis:
    """Return the smallest leading orthonormal basis capturing ``threshold``."""

    if not 0 < threshold <= 1:
        raise ValueError("threshold must lie in (0, 1]")
    if spectrum.total_energy == 0:
        return EnergyBasis(
            basis=spectrum.right_basis[:, :0],
            n_components=0,
            requested_energy=float(threshold),
            captured_energy=1.0,
            total_energy=0.0,
        )
    cumulative = torch.cumsum(spectrum.eigenvalues, dim=0) / spectrum.total_energy
    target = torch.tensor(threshold, dtype=cumulative.dtype, device=cumulative.device)
    n_components = int(torch.searchsorted(cumulative, target, right=False).item()) + 1
    n_components = min(n_components, int(cumulative.numel()))
    captured = float(cumulative[n_components - 1].item())
    return EnergyBasis(
        basis=spectrum.right_basis[:, :n_components],
        n_components=n_components,
        requested_energy=float(threshold),
        captured_energy=captured,
        total_energy=spectrum.total_energy,
    )


def analyze_token_frame(
    operator: TokenFrameOperator,
    **gram_kwargs: object,
) -> tuple[GramResult, SpectrumResult]:
    """Convenience wrapper for the default CPU eigendecomposition."""

    gram = streaming_gram(operator, **gram_kwargs)
    return gram, decompose_gram(gram)
