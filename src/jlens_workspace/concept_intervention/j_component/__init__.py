"""Compatibility interface for residual-space J-component steering."""

from importlib import import_module as _import_module

_canonical = _import_module(
    "jlens_workspace.concept_intervention.steering.j_component"
)
__all__ = _canonical.__all__
globals().update({name: getattr(_canonical, name) for name in __all__})
