"""Compatibility import for the foundation lazy-dependency proxy."""

from jlens_workspace.foundation.jacobian import _optional as _implementation
from jlens_workspace.foundation.jacobian._optional import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(dir(_implementation)))
