"""Lazy adapter around Anthropic's official :mod:`jlens` package."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .metadata import (
    ARTIFACT_FORMAT,
    FORMAT_KEY,
    METADATA_KEY,
    JLensMetadata,
    JLensMetadataError,
)


class OfficialJLensUnavailableError(ImportError):
    """The optional official ``jlens`` package is not installed."""


def import_official_jlens() -> Any:
    """Import ``jlens`` on first use, never at package-import time."""

    try:
        return importlib.import_module("jlens")
    except ImportError as exc:
        raise OfficialJLensUnavailableError(
            "Anthropic's optional `jlens` package is required for this operation; "
            "install https://github.com/anthropics/jacobian-lens"
        ) from exc


def _official_version() -> str | None:
    try:
        return importlib.metadata.version("jlens")
    except importlib.metadata.PackageNotFoundError:
        module = import_official_jlens()
        value = getattr(module, "__version__", None)
        return str(value) if value is not None else None


def _nonempty_attribute(obj: Any, *names: str) -> str | None:
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _runtime_identity(model: Any) -> dict[str, str]:
    """Best-effort immutable identity exposed by HF model/tokenizer objects."""

    hf_model = getattr(model, "_hf_model", model)
    config = getattr(hf_model, "config", None)
    tokenizer = getattr(model, "tokenizer", None)
    model_id = _nonempty_attribute(hf_model, "name_or_path")
    if model_id is None and config is not None:
        model_id = _nonempty_attribute(config, "_name_or_path", "name_or_path")
    model_revision = _nonempty_attribute(hf_model, "_commit_hash")
    if model_revision is None and config is not None:
        model_revision = _nonempty_attribute(config, "_commit_hash")
    tokenizer_id = _nonempty_attribute(tokenizer, "name_or_path")
    tokenizer_revision = _nonempty_attribute(tokenizer, "_commit_hash")
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if tokenizer_revision is None and isinstance(init_kwargs, Mapping):
        value = init_kwargs.get("_commit_hash")
        if isinstance(value, str) and value.strip():
            tokenizer_revision = value.strip()
    return {
        key: value
        for key, value in {
            "model_id": model_id,
            "model_revision": model_revision,
            "tokenizer_id": tokenizer_id,
            "tokenizer_revision": tokenizer_revision,
        }.items()
        if value is not None
    }


def _validate_runtime_identity(model: Any, metadata: JLensMetadata) -> None:
    observed = _runtime_identity(model)
    metadata.validate_expected(observed)
    d_model = getattr(model, "d_model", None)
    if d_model is not None and int(d_model) != metadata.d_model:
        raise JLensMetadataError(
            f"runtime d_model={d_model} differs from metadata d_model={metadata.d_model}"
        )


@dataclass(frozen=True)
class ManagedJacobianLens:
    """An official lens coupled to the metadata needed for safe reuse."""

    lens: Any
    metadata: JLensMetadata

    def __post_init__(self) -> None:
        self.metadata.validate_lens(self.lens)

    @property
    def jacobians(self) -> Mapping[int, Any]:
        return self.lens.jacobians

    @property
    def source_layers(self) -> tuple[int, ...]:
        return self.metadata.source_layers

    @property
    def d_model(self) -> int:
        return self.metadata.d_model

    @property
    def n_prompts(self) -> int:
        return int(self.lens.n_prompts)


def _torch() -> Any:
    # Torch is a required runtime dependency of the project, but keeping this
    # import local makes metadata inspection and documentation builds cheap.
    return importlib.import_module("torch")


def _resolve_local_file(name_or_path: str | os.PathLike[str], filename: str) -> Path | None:
    path = Path(name_or_path)
    if path.is_file():
        return path
    if path.is_dir():
        return path / filename
    return None


def _read_embedded_metadata(path: Path) -> JLensMetadata | None:
    torch = _torch()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise JLensMetadataError(f"{path} is not a mapping checkpoint")
    value = state.get(METADATA_KEY)
    if value is None:
        return None
    if state.get(FORMAT_KEY) != ARTIFACT_FORMAT:
        raise JLensMetadataError(
            f"unsupported artifact format {state.get(FORMAT_KEY)!r}"
        )
    if not isinstance(value, Mapping):
        raise JLensMetadataError("embedded Jacobian metadata is not a mapping")
    return JLensMetadata.from_dict(value)


class OfficialJLensAdapter:
    """Load, save and fit official lenses while enforcing artifact identity."""

    @staticmethod
    def from_hf(hf_model: Any, tokenizer: Any, **kwargs: Any) -> Any:
        return import_official_jlens().from_hf(hf_model, tokenizer, **kwargs)

    @classmethod
    def fit(
        cls,
        model: Any,
        prompts: Sequence[str],
        *,
        metadata: JLensMetadata,
        **fit_kwargs: Any,
    ) -> ManagedJacobianLens:
        official = import_official_jlens()
        _validate_runtime_identity(model, metadata)
        n_layers = getattr(model, "n_layers", None)
        if n_layers is not None:
            n_layers = int(n_layers)
            raw_target = fit_kwargs.get("target_layer")
            target_layer = n_layers - 1 if raw_target is None else int(raw_target)
            if target_layer < 0:
                target_layer += n_layers
            if metadata.target_layer is None:
                metadata = replace(metadata, target_layer=target_layer)
            elif metadata.target_layer != target_layer:
                raise JLensMetadataError(
                    f"fit target_layer={target_layer} differs from metadata "
                    f"target_layer={metadata.target_layer}"
                )
            raw_sources = fit_kwargs.get("source_layers")
            if raw_sources is None:
                fit_sources = tuple(range(target_layer))
            else:
                fit_sources = tuple(
                    sorted(
                        {
                            int(layer) + n_layers if int(layer) < 0 else int(layer)
                            for layer in raw_sources
                        }
                    )
                )
            if fit_sources != metadata.source_layers:
                raise JLensMetadataError(
                    f"fit source_layers={fit_sources} differ from metadata "
                    f"source_layers={metadata.source_layers}"
                )
        lens = official.fit(model, prompts=prompts, **fit_kwargs)
        # Fill values that are outcomes of fitting, while refusing to change
        # caller-declared identity or layer information.
        fitted_prompts = int(lens.n_prompts)
        if metadata.n_prompts is None:
            metadata = replace(metadata, n_prompts=fitted_prompts)
        if metadata.upstream_version is None:
            metadata = replace(metadata, upstream_version=_official_version())
        metadata.validate_lens(lens)
        return ManagedJacobianLens(lens=lens, metadata=metadata)

    @classmethod
    def load(
        cls,
        name_or_path: str | os.PathLike[str],
        *,
        filename: str = "lens.pt",
        revision: str | None = None,
        metadata: JLensMetadata | None = None,
        expected: JLensMetadata | Mapping[str, Any] | None = None,
    ) -> ManagedJacobianLens:
        """Load a local or Hub lens and validate all requested metadata.

        Workspace artifacts embed metadata in the official-compatible ``.pt``
        file.  Legacy/upstream files do not; loading one therefore requires an
        explicit ``metadata=`` argument.  No model or tokenizer revision is
        guessed from a mutable Hub branch.
        """

        official = import_official_jlens()
        local_file = _resolve_local_file(name_or_path, filename)
        embedded = None
        if local_file is not None:
            if not local_file.is_file():
                raise FileNotFoundError(local_file)
            embedded = _read_embedded_metadata(local_file)
            lens = official.JacobianLens.load(str(local_file))
        else:
            lens = official.JacobianLens.from_pretrained(
                str(name_or_path), filename=filename, revision=revision
            )
        if embedded is not None and metadata is not None and embedded != metadata:
            raise JLensMetadataError(
                "explicit metadata differs from metadata embedded in the artifact"
            )
        resolved = embedded if embedded is not None else metadata
        if resolved is None:
            raise JLensMetadataError(
                "the lens has no embedded model/tokenizer metadata; pass "
                "metadata= explicitly for an upstream or legacy artifact"
            )
        resolved.validate_lens(lens)
        resolved.validate_expected(expected)
        return ManagedJacobianLens(lens=lens, metadata=resolved)

    @classmethod
    def save(
        cls,
        lens: ManagedJacobianLens | Any,
        path: str | os.PathLike[str],
        *,
        metadata: JLensMetadata | None = None,
        dtype: Any | None = None,
    ) -> Path:
        """Save an official-compatible checkpoint with embedded metadata.

        Unlike upstream ``JacobianLens.save``, this defaults to fp32 to retain
        the small singular-value tail needed for numerical-rank analysis.
        """

        torch = _torch()
        if isinstance(lens, ManagedJacobianLens):
            if metadata is not None and metadata != lens.metadata:
                raise JLensMetadataError(
                    "explicit metadata differs from ManagedJacobianLens metadata"
                )
            metadata = lens.metadata
            raw_lens = lens.lens
        else:
            raw_lens = lens
        if metadata is None:
            raise JLensMetadataError("metadata is required when saving a raw lens")
        if metadata.n_prompts is None:
            metadata = replace(metadata, n_prompts=int(raw_lens.n_prompts))
        metadata.validate_lens(raw_lens)
        dtype = torch.float32 if dtype is None else dtype
        if isinstance(dtype, str):
            try:
                dtype = getattr(torch, dtype)
            except AttributeError as exc:
                raise ValueError(f"unknown torch dtype {dtype!r}") from exc
        state = {
            "J": {
                int(layer): matrix.detach().to(device="cpu", dtype=dtype)
                for layer, matrix in raw_lens.jacobians.items()
            },
            "n_prompts": int(raw_lens.n_prompts),
            "source_layers": list(int(x) for x in raw_lens.source_layers),
            "d_model": int(raw_lens.d_model),
            FORMAT_KEY: ARTIFACT_FORMAT,
            METADATA_KEY: metadata.to_dict(),
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
        try:
            torch.save(state, temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination


def load_lens(*args: Any, **kwargs: Any) -> ManagedJacobianLens:
    return OfficialJLensAdapter.load(*args, **kwargs)


def save_lens(*args: Any, **kwargs: Any) -> Path:
    return OfficialJLensAdapter.save(*args, **kwargs)


def fit_lens(*args: Any, **kwargs: Any) -> ManagedJacobianLens:
    return OfficialJLensAdapter.fit(*args, **kwargs)
