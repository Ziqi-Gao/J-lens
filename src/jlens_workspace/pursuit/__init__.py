"""Streaming sparse non-negative pursuit over token J-direction dictionaries.

Implements ``concept_occupancy_method_v1``: an exact full-vocabulary
re-search-per-step non-negative pursuit, a norm-matched Gaussian random
dictionary control (``matched_gaussian_atom_norms_v1``), and versioned
occupancy crossing rules. Solver cores are NumPy; Torch appears only in the
token-frame dictionary adapter.
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
    PursuitResult,
    PursuitSolverError,
    streaming_nonnegative_pursuit,
)

__all__ = [
    "CROSSING_RULES",
    "SELECTION_MODES",
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
