"""Artifact IO with atomic metadata writes and reproducibility manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(items: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for item in items:
        encoded = item.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_commit(cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass(frozen=True)
class RunManifest:
    schema_version: int = 1
    experiment_name: str = ""
    seed: int = 42
    model_id: str | None = None
    model_revision: str | None = None
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    lens_source: str | None = None
    lens_revision: str | None = None
    dataset_source: str | None = None
    dataset_revision: str | None = None
    dataset_hash: str | None = None
    git_commit: str | None = None
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    packages: dict[str, str | None] = field(
        default_factory=lambda: {
            name: _package_version(name)
            for name in (
                "jlens-workspace",
                "jlens",
                "torch",
                "transformers",
                "datasets",
                "numpy",
                "scikit-learn",
            )
        }
    )
    notes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_workspace(cls, workspace: str | Path, **kwargs: Any) -> RunManifest:
        return cls(git_commit=_git_commit(Path(workspace)), **kwargs)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Write JSON by rename so interrupted jobs do not leave valid-looking partial files."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = asdict(payload) if hasattr(payload, "__dataclass_fields__") else payload
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
