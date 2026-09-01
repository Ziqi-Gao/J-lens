"""Compatibility import for :mod:`jlens_workspace.foundation.activations`."""

from jlens_workspace.foundation import activations as _implementation
from jlens_workspace.foundation.activations import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(dir(_implementation)))
