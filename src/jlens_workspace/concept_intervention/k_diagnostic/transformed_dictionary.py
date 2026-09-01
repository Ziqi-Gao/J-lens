"""Compatibility alias for transformed K-diagnostic dictionaries."""

import sys as _sys
from importlib import import_module as _import_module

_sys.modules[__name__] = _import_module(
    "jlens_workspace.concept_intervention.geometry.k_diagnostic.transformed_dictionary"
)
