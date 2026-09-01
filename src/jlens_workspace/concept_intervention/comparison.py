"""Compatibility alias for three-method output comparison."""

import sys as _sys
from importlib import import_module as _import_module

_sys.modules[__name__] = _import_module(
    "jlens_workspace.concept_intervention.evaluation.comparison"
)
