"""Comparisons between orthogonal bases and Gram-weighted subspaces."""

from __future__ import annotations

from dataclasses import dataclass

from jlens_workspace.jacobian._optional import torch


@dataclass(frozen=True)
class BasisCoverage:
    angles_radians: torch.Tensor
    cosines: torch.Tensor
    a_covered_by_b: float
    b_covered_by_a: float
    shared_dimension: float
    dim_a: int
    dim_b: int


def orthonormalize_basis(
    basis: torch.Tensor,
    *,
    atol: float = 0.0,
    rtol: float | None = None,
) -> torch.Tensor:
    """Return an orthonormal basis for the input columns using an SVD rank test."""

    if basis.ndim != 2:
        raise ValueError("basis must have shape [ambient_dim, n_vectors]")
    if basis.shape[1] == 0:
        return basis.to(dtype=torch.float64)
    work = basis.to(dtype=torch.float64)
    left, singular, _ = torch.linalg.svd(work, full_matrices=False)
    largest = float(singular[0].item()) if singular.numel() else 0.0
    if rtol is None:
        rtol = max(int(basis.shape[0]), int(basis.shape[1])) * torch.finfo(
            torch.float64
        ).eps
    tolerance = max(float(atol), float(rtol) * largest)
    rank = int((singular > tolerance).sum().item())
    return left[:, :rank]


def principal_angles(
    basis_a: torch.Tensor,
    basis_b: torch.Tensor,
    *,
    assume_orthonormal: bool = False,
) -> torch.Tensor:
    """Principal angles in ascending order, in radians."""

    if basis_a.ndim != 2 or basis_b.ndim != 2:
        raise ValueError("bases must be two-dimensional")
    if basis_a.shape[0] != basis_b.shape[0]:
        raise ValueError("bases must share an ambient dimension")
    if assume_orthonormal:
        qa = basis_a.to(dtype=torch.float64)
        qb = basis_b.to(dtype=torch.float64)
    else:
        qa = orthonormalize_basis(basis_a)
        qb = orthonormalize_basis(basis_b)
    if qa.shape[1] == 0 or qb.shape[1] == 0:
        return torch.empty(0, dtype=torch.float64, device=qa.device)
    qb = qb.to(device=qa.device)
    cosines = torch.linalg.svdvals(qa.T @ qb).clamp(0, 1)
    return torch.acos(cosines)


def basis_coverage(
    basis_a: torch.Tensor,
    basis_b: torch.Tensor,
    *,
    assume_orthonormal: bool = False,
) -> BasisCoverage:
    """Directional coverage and principal angles for two subspaces.

    ``a_covered_by_b`` is the mean squared projection of an orthonormal basis
    for A onto B.  It equals one exactly when A is contained in B.  The reverse
    quantity uses B's dimensionality and is intentionally not forced equal.
    """

    if assume_orthonormal:
        qa = basis_a.to(dtype=torch.float64)
        qb = basis_b.to(dtype=torch.float64, device=qa.device)
    else:
        qa = orthonormalize_basis(basis_a)
        qb = orthonormalize_basis(basis_b).to(device=qa.device)
    dim_a, dim_b = int(qa.shape[1]), int(qb.shape[1])
    if dim_a == 0 or dim_b == 0:
        cosines = torch.empty(0, dtype=torch.float64, device=qa.device)
        shared = 0.0
    else:
        cosines = torch.linalg.svdvals(qa.T @ qb).clamp(0, 1)
        shared = float(cosines.square().sum().item())
    angles = torch.acos(cosines)
    return BasisCoverage(
        angles_radians=angles,
        cosines=cosines,
        a_covered_by_b=1.0 if dim_a == 0 else shared / dim_a,
        b_covered_by_a=1.0 if dim_b == 0 else shared / dim_b,
        shared_dimension=shared,
        dim_a=dim_a,
        dim_b=dim_b,
    )


def gram_energy_coverage(
    gram: torch.Tensor,
    basis: torch.Tensor,
    *,
    assume_orthonormal: bool = False,
) -> float:
    """Fraction of Gram trace captured by projection onto ``basis``."""

    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be square")
    if basis.ndim != 2 or basis.shape[0] != gram.shape[0]:
        raise ValueError("basis ambient dimension must match gram")
    q = (
        basis.to(dtype=torch.float64)
        if assume_orthonormal
        else orthonormalize_basis(basis)
    )
    gram64 = gram.to(device=q.device, dtype=torch.float64)
    total = torch.trace(gram64)
    if float(total.item()) < 0:
        raise ValueError("gram must have non-negative trace")
    if float(total.item()) == 0:
        return 1.0
    if q.shape[1] == 0:
        return 0.0
    captured = torch.trace(q.T @ gram64 @ q)
    value = float((captured / total).item())
    # Protect report serialization from sub-ulp excursions.
    return min(1.0, max(0.0, value))
