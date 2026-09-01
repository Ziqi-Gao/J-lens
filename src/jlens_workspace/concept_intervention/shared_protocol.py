"""Compatibility facade for the former combined shared-protocol module.

Canonical row/artifact contracts live in :mod:`.protocol.shared`; probe
fitting and validation-only layer selection live in
:mod:`.probing.layer_selection`. Imported objects are re-exported unchanged.
"""

from .probing import layer_selection as _layer_selection
from .probing.layer_selection import (
    load_upstream_raptor_tuning,
    run_shared_layer_selection,
)
from .protocol import shared as _shared
from .protocol.shared import (
    RAPTOR_C_GRID,
    RAPTOR_COMMIT,
    RAPTOR_REPOSITORY,
    SharedProtocolError,
    deterministic_balanced_indices,
    load_balanced_indices,
    load_selected_layers,
    validate_shared_identity,
)

_atomic_save_npy = _layer_selection._atomic_save_npy
_fit_one_layer = _layer_selection._fit_one_layer
_grouped_raptor_c_scores = _layer_selection._grouped_raptor_c_scores
_new_pipeline = _layer_selection._new_pipeline
_raw_probe = _layer_selection._raw_probe
_row_identity = _layer_selection._row_identity
_stable_group_key = _shared._stable_group_key

__all__ = [
    "RAPTOR_COMMIT",
    "RAPTOR_C_GRID",
    "RAPTOR_REPOSITORY",
    "SharedProtocolError",
    "deterministic_balanced_indices",
    "load_balanced_indices",
    "load_selected_layers",
    "load_upstream_raptor_tuning",
    "run_shared_layer_selection",
    "validate_shared_identity",
]
