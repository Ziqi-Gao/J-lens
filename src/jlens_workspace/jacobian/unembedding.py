"""Compatibility import for foundation effective-unembedding conventions."""

from jlens_workspace.foundation.jacobian import unembedding as _implementation
from jlens_workspace.foundation.jacobian.unembedding import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(dir(_implementation)))
