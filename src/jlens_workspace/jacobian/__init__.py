"""Compatibility import for :mod:`jlens_workspace.foundation.jacobian`."""

from jlens_workspace.foundation import jacobian as _implementation
from jlens_workspace.foundation.jacobian import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(dir(_implementation)))
