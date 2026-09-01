"""Content-addressed ITI direction-artifact contract.

This validator is intentionally independent of ITI fitting and hook code so
evaluation can authenticate a frozen method output without importing the
steering implementation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jlens_workspace.foundation.artifacts import sha256_file

ITI_METHOD = "honest_llama_mass_mean_qwen_full_attention_v1"
ITI_METHOD_UNVERSIONED = "honest_llama_mass_mean_qwen_full_attention"


class ITIError(ValueError):
    """An ITI input violates the registered coordinate or artifact contract."""


def _resolve_registered_file(root: Path, filename: str) -> Path:
    path = (root / filename).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ITIError(f"ITI direction file escapes its artifact root: {path}") from error
    return path


def _hash_registered_files(root: Path, files: Any) -> Any:
    """Mirror a nested filename registry with SHA-256 content identities."""

    if isinstance(files, str):
        path = _resolve_registered_file(root, files)
        if not path.is_file():
            raise ITIError(f"registered ITI direction file is missing: {path}")
        return sha256_file(path)
    if isinstance(files, Mapping):
        return {str(key): _hash_registered_files(root, value) for key, value in files.items()}
    raise ITIError("ITI file registry must contain only mappings and filenames")


def _validate_registered_files(root: Path, files: Any, hashes: Any) -> int:
    if isinstance(files, str):
        path = _resolve_registered_file(root, files)
        if not isinstance(hashes, str) or not path.is_file() or sha256_file(path) != hashes:
            raise ITIError(f"ITI direction file identity mismatch: {path}")
        return 1
    if isinstance(files, Mapping):
        if not isinstance(hashes, Mapping) or set(files) != set(hashes):
            raise ITIError("ITI direction file and hash registries differ")
        return sum(_validate_registered_files(root, files[key], hashes[key]) for key in files)
    raise ITIError("ITI file registry must contain only mappings and filenames")


def validate_iti_direction_artifact(metrics_path: str | Path) -> dict[str, Any]:
    """Verify the metrics identity and every registered ITI direction file."""

    path = Path(metrics_path)
    if not path.is_file():
        raise ITIError(f"ITI direction metrics are missing: {path}")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if metrics.get("method") not in {ITI_METHOD, ITI_METHOD_UNVERSIONED}:
        raise ITIError("unsupported ITI direction artifact")
    files = metrics.get("files")
    file_hashes = metrics.get("files_sha256")
    if not isinstance(files, Mapping) or not isinstance(file_hashes, Mapping):
        raise ITIError("ITI direction metrics lack content-addressed files")
    if metrics["method"] == ITI_METHOD:
        if set(files) != {"mass_mean", "probe_weight", "random", "development_indices"}:
            raise ITIError("legacy ITI direction file registry is incomplete")
    else:
        if set(files) != {"variants", "random", "development_indices"}:
            raise ITIError("shared ITI direction file registry is incomplete")
        variants = files["variants"]
        if (
            not isinstance(variants, Mapping)
            or not variants
            or any(
                not isinstance(value, Mapping)
                or set(value) != {"mass_mean", "probe_weight"}
                for value in variants.values()
            )
        ):
            raise ITIError("shared ITI variant registry is incomplete")
    if _validate_registered_files(path.parent, files, file_hashes) < 1:
        raise ITIError("ITI direction artifact has no registered files")
    return metrics


__all__ = [
    "ITI_METHOD",
    "ITI_METHOD_UNVERSIONED",
    "ITIError",
    "validate_iti_direction_artifact",
]
