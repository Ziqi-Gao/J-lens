"""Shared infrastructure for the two J-lens research directions."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jlens-workspace")
except PackageNotFoundError:  # pragma: no cover - editable source tree
    __version__ = "0.1.0"

__all__ = ["__version__"]
