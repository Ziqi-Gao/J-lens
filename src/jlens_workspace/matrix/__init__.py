"""Memory-bounded analysis of Jacobian-lens token frames."""

from .geometry import (
    BasisCoverage,
    basis_coverage,
    gram_energy_coverage,
    orthonormalize_basis,
    principal_angles,
)
from .operator import (
    ALOperator,
    MatrixDeviceError,
    TokenFrameOperator,
    TokenFrameOperatorMetadata,
    TokenRowBlock,
)
from .spectrum import (
    EnergyBasis,
    GramResult,
    SpectralAnalysisError,
    SpectrumResult,
    analyze_token_frame,
    decompose_gram,
    minimum_energy_basis,
    streaming_gram,
)

__all__ = [
    "ALOperator",
    "BasisCoverage",
    "EnergyBasis",
    "GramResult",
    "MatrixDeviceError",
    "SpectralAnalysisError",
    "SpectrumResult",
    "TokenFrameOperator",
    "TokenFrameOperatorMetadata",
    "TokenRowBlock",
    "analyze_token_frame",
    "basis_coverage",
    "decompose_gram",
    "gram_energy_coverage",
    "minimum_energy_basis",
    "orthonormalize_basis",
    "principal_angles",
    "streaming_gram",
]
