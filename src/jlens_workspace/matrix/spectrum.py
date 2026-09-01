"""Compatibility import for :mod:`jlens_workspace.j_space.spectrum`."""

from jlens_workspace.j_space import spectrum as _implementation
from jlens_workspace.j_space.spectrum import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(dir(_implementation)))
