"""Compatibility import for :mod:`jlens_workspace.foundation.config`."""

from jlens_workspace.foundation import config as _implementation
from jlens_workspace.foundation.config import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(dir(_implementation)))
