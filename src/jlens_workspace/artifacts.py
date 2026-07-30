"""Artifact IO with atomic metadata writes and reproducibility manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


class GitIdentityError(ValueError):
    """Raised when a checkout's current commit cannot be resolved safely."""


def _git_directories(checkout: Path) -> tuple[Path, Path]:
    marker = checkout / ".git"
    if marker.is_dir():
        git_dir = marker.resolve()
    elif marker.is_file():
        value = marker.read_text(encoding="utf-8").strip()
        prefix = "gitdir:"
        if not value.casefold().startswith(prefix):
            raise GitIdentityError(f"invalid linked-worktree marker: {marker}")
        target = value[len(prefix) :].strip()
        git_dir = Path(target)
        if not git_dir.is_absolute():
            git_dir = marker.parent / git_dir
        git_dir = git_dir.resolve()
        if not git_dir.is_dir():
            raise GitIdentityError(f"linked-worktree git directory is missing: {git_dir}")
    else:
        raise GitIdentityError(f"checkout lacks a .git marker: {checkout}")

    common_dir = git_dir
    common_marker = git_dir / "commondir"
    if common_marker.is_file():
        target = common_marker.read_text(encoding="utf-8").strip()
        common_dir = Path(target)
        if not common_dir.is_absolute():
            common_dir = git_dir / common_dir
        common_dir = common_dir.resolve()
        if not common_dir.is_dir():
            raise GitIdentityError(f"Git common directory is missing: {common_dir}")
    return git_dir, common_dir


def _git_object_id(value: str, *, source: Path) -> str:
    candidate = value.strip().casefold()
    if len(candidate) not in {40, 64}:
        raise GitIdentityError(f"invalid Git object ID in {source}")
    try:
        int(candidate, 16)
    except ValueError as error:
        raise GitIdentityError(f"invalid Git object ID in {source}") from error
    return candidate


def _safe_git_ref(root: Path, reference: str) -> Path:
    if (
        not reference.startswith("refs/")
        or reference.startswith("/")
        or any(part in {"", ".", ".."} for part in reference.split("/"))
    ):
        raise GitIdentityError(f"unsafe Git reference: {reference!r}")
    path = (root / reference).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise GitIdentityError(f"Git reference escapes metadata root: {reference}") from error
    return path


def _packed_ref(root: Path, reference: str) -> str | None:
    path = root / "packed-refs"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "^")):
            continue
        fields = stripped.split(" ", 1)
        if len(fields) == 2 and fields[1] == reference:
            return _git_object_id(fields[0], source=path)
    return None


def git_head_commit(checkout: str | Path) -> str:
    """Resolve HEAD from Git metadata without requiring the Git executable."""

    root = Path(checkout).resolve()
    git_dir, common_dir = _git_directories(root)
    head = git_dir / "HEAD"
    if not head.is_file():
        raise GitIdentityError(f"Git HEAD is missing: {head}")
    value = head.read_text(encoding="utf-8").strip()
    for _depth in range(8):
        if not value.startswith("ref:"):
            return _git_object_id(value, source=head)
        reference = value.removeprefix("ref:").strip()
        for metadata_root in (git_dir, common_dir):
            ref_path = _safe_git_ref(metadata_root, reference)
            if ref_path.is_file():
                value = ref_path.read_text(encoding="utf-8").strip()
                head = ref_path
                break
        else:
            for metadata_root in (git_dir, common_dir):
                packed = _packed_ref(metadata_root, reference)
                if packed is not None:
                    return packed
            raise GitIdentityError(
                f"cannot resolve Git reference {reference!r} for {root}"
            )
    raise GitIdentityError(f"Git symbolic-reference chain is too deep: {root}")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repository_resource(path: str | Path) -> Path:
    """Resolve a read-only resource when code and run roots are separate.

    Relative artifact paths keep resolving against the current working
    directory. If a relative path is absent there, ``JLENS_REPOSITORY_ROOT``
    provides an explicit fallback for files committed with the immutable code
    checkout, such as prompt banks. The fallback may never escape that root.
    """

    source = Path(path)
    if source.is_absolute() or source.exists():
        return source
    root_value = os.environ.get("JLENS_REPOSITORY_ROOT")
    if not root_value:
        return source
    root = Path(root_value).resolve()
    candidate = (root / source).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"repository resource escapes JLENS_REPOSITORY_ROOT: {source}"
        ) from error
    return candidate if candidate.is_file() else source


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
    recorded = os.environ.get("JLENS_GIT_COMMIT")
    if recorded is not None:
        candidate = recorded.strip().casefold()
        if len(candidate) not in {40, 64}:
            raise ValueError("JLENS_GIT_COMMIT must be a 40- or 64-character hex digest")
        int(candidate, 16)
        return candidate
    try:
        return git_head_commit(cwd)
    except GitIdentityError:
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
