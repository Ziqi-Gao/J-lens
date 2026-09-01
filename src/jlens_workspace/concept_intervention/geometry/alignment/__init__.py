"""Probe-to-token-J alignment and decomposition utilities."""

from .core import (
    BasisCoverage,
    JSpaceDecomposition,
    TopKAlignment,
    activation_covariance,
    basis_concept_coverage,
    chunked_cosine_topk,
    chunked_token_j_topk,
    compute_token_j_rows,
    decompose_j_space,
    materialize_token_j_rows,
    sparse_nonnegative_decomposition,
    top_r_basis_concept_coverage,
)

__all__ = [
    "BasisCoverage",
    "JSpaceDecomposition",
    "TopKAlignment",
    "activation_covariance",
    "basis_concept_coverage",
    "chunked_cosine_topk",
    "chunked_token_j_topk",
    "compute_token_j_rows",
    "decompose_j_space",
    "materialize_token_j_rows",
    "sparse_nonnegative_decomposition",
    "top_r_basis_concept_coverage",
]
