"""Compatibility import for :mod:`jlens_workspace.j_space`."""

from jlens_workspace import j_space as _implementation
from jlens_workspace.j_space import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(dir(_implementation)))
