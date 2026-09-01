"""Compatibility import for :mod:`jlens_workspace.j_space.workflow`."""

from jlens_workspace.j_space import workflow as _implementation
from jlens_workspace.j_space.workflow import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(dir(_implementation)))
