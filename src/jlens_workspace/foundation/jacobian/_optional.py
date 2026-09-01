"""Tiny lazy-module proxies for optional heavy runtime dependencies."""

from __future__ import annotations

import importlib
from types import ModuleType


class LazyModule:
    """Resolve a module only when one of its attributes is first requested."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._module: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._module is None:
            self._module = importlib.import_module(self._name)
        return self._module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)


torch = LazyModule("torch")
