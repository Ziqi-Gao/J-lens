"""Streaming non-negative pursuit over token J-direction dictionaries."""

from .dictionaries import (
    DenseDictionary,
    DictionaryError,
    MatchedNormRandomDictionary,
    TokenFrameDictionary,
    UnitNormDictionary,
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
    "UnitNormDictionary",
    "absolute_threshold_ks",
    "build_token_frame_dictionary",
    "crossing_k",
    "k_90_attainable",
    "streaming_nonnegative_pursuit",
]
