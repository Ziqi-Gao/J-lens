"""Compatibility import for the foundation Jacobian-lens adapter."""

from jlens_workspace.foundation.jacobian import adapter as _implementation
from jlens_workspace.foundation.jacobian.adapter import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(dir(_implementation)))
