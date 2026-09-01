"""Compatibility interface for Concept Intervention K diagnostics."""

from importlib import import_module as _import_module

_canonical = _import_module(
    "jlens_workspace.concept_intervention.geometry.k_diagnostic"
)
__all__ = _canonical.__all__
globals().update({name: getattr(_canonical, name) for name in __all__})
