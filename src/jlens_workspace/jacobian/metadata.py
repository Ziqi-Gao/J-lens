"""Compatibility import for foundation Jacobian metadata."""

from jlens_workspace.foundation.jacobian import metadata as _implementation
from jlens_workspace.foundation.jacobian.metadata import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(dir(_implementation)))
