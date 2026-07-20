"""Streaming sparse non-negative pursuit over token J-direction dictionaries.

Implements versioned full-vocabulary re-search-per-step solvers, a norm-matched
Gaussian random dictionary control (``matched_gaussian_atom_norms_v1``), and
versioned occupancy crossing rules. Solver cores are NumPy; Torch appears only
in the token-frame dictionary adapter.
"""

from .dictionaries import (
    DenseDictionary,
    DictionaryError,
    MatchedNormRandomDictionary,
    TokenFrameDictionary,
    build_token_frame_dictionary,
)
from .occupancy import (
    CROSSING_RULES,
    absolute_threshold_ks,
    crossing_k,
    k_90_attainable,
)
from .solver import (
    SELECTION_MODES,
    SOLVER_METHODS,
    PursuitResult,
    PursuitSolverError,
    streaming_nonnegative_pursuit,
)

__all__ = [
    "CROSSING_RULES",
    "SELECTION_MODES",
    "SOLVER_METHODS",
    "DenseDictionary",
    "DictionaryError",
    "MatchedNormRandomDictionary",
    "PursuitResult",
    "PursuitSolverError",
    "TokenFrameDictionary",
    "absolute_threshold_ks",
    "build_token_frame_dictionary",
    "crossing_k",
    "k_90_attainable",
    "streaming_nonnegative_pursuit",
]
