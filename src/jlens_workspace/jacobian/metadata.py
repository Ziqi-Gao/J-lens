"""Metadata and validation for fitted Jacobian-lens artifacts.

The upstream :mod:`jlens` checkpoint intentionally contains only the matrices,
their layer indices, ``d_model`` and prompt count.  That is enough for readout,
but not enough to safely compare spectra or apply a lens to a newly loaded
model.  This module defines the additional identity information required by the
workspace experiments.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

ARTIFACT_FORMAT = "jlens-workspace.jacobian.v1"
METADATA_KEY = "jlens_workspace_metadata"
FORMAT_KEY = "jlens_workspace_format"


class JLensMetadataError(ValueError):
    """Base class for missing or malformed Jacobian-lens metadata."""


class JLensMetadataMismatchError(JLensMetadataError):
    """Raised when an artifact does not match the requested identity."""


def _required_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JLensMetadataError(f"{name} must be a non-empty string")
    return value.strip()


def _canonical_layers(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise JLensMetadataError("source_layers must be an iterable of integers")
    try:
        layers = tuple(int(layer) for layer in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise JLensMetadataError(
            "source_layers must be an iterable of integers"
        ) from exc
    if not layers:
        raise JLensMetadataError("source_layers must not be empty")
    if any(layer < 0 for layer in layers):
        raise JLensMetadataError("source_layers must use resolved, non-negative indices")
    if tuple(sorted(set(layers))) != layers:
        raise JLensMetadataError("source_layers must be sorted and unique")
    return layers


@dataclass(frozen=True)
class JLensMetadata:
    """Identity and fitting metadata associated with a set of Jacobians.

    Model and tokenizer revisions are mandatory.  Callers should use immutable
    commit hashes when loading from the Hugging Face Hub; branch names such as
    ``main`` are accepted but are deliberately compared literally on load.

    ``norm_convention`` records how token directions derived from the matrices
    are interpreted.  It does not alter the stored ``J_l`` matrices.
    """

    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    d_model: int
    source_layers: tuple[int, ...]
    target_layer: int | None = None
    norm_convention: str = "raw"
    n_prompts: int | None = None
    upstream_package: str = "jlens"
    upstream_version: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _required_text("model_id", self.model_id))
        object.__setattr__(
            self,
            "model_revision",
            _required_text("model_revision", self.model_revision),
        )
        object.__setattr__(
            self, "tokenizer_id", _required_text("tokenizer_id", self.tokenizer_id)
        )
        object.__setattr__(
            self,
            "tokenizer_revision",
            _required_text("tokenizer_revision", self.tokenizer_revision),
        )
        if int(self.d_model) <= 0:
            raise JLensMetadataError("d_model must be positive")
        object.__setattr__(self, "d_model", int(self.d_model))
        object.__setattr__(
            self, "source_layers", _canonical_layers(self.source_layers)
        )
        if self.target_layer is not None:
            target_layer = int(self.target_layer)
            if target_layer < 0:
                raise JLensMetadataError(
                    "target_layer must be resolved and non-negative"
                )
            if target_layer <= self.source_layers[-1]:
                raise JLensMetadataError(
                    "target_layer must be greater than every source layer"
                )
            object.__setattr__(self, "target_layer", target_layer)
        object.__setattr__(
            self,
            "norm_convention",
            _required_text("norm_convention", self.norm_convention),
        )
        if self.n_prompts is not None and int(self.n_prompts) < 0:
            raise JLensMetadataError("n_prompts must be non-negative")
        if self.n_prompts is not None:
            object.__setattr__(self, "n_prompts", int(self.n_prompts))
        extra = dict(self.extra)
        try:
            json.dumps(extra, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise JLensMetadataError("extra metadata must be JSON-serializable") from exc
        object.__setattr__(self, "extra", extra)

    @property
    def layers(self) -> tuple[int, ...]:
        """Alias used by matrix-analysis code and serialized reports."""

        return self.source_layers

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "d_model": self.d_model,
            "source_layers": list(self.source_layers),
            "target_layer": self.target_layer,
            "norm_convention": self.norm_convention,
            "n_prompts": self.n_prompts,
            "upstream_package": self.upstream_package,
            "upstream_version": self.upstream_version,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JLensMetadata:
        data = dict(value)
        if "source_layers" not in data and "layers" in data:
            data["source_layers"] = data.pop("layers")
        try:
            return cls(**data)
        except TypeError as exc:
            raise JLensMetadataError(f"invalid metadata fields: {exc}") from exc

    def validate_lens(self, lens: object) -> None:
        """Validate matrix shape, layer set and prompt count against ``lens``."""

        d_model = getattr(lens, "d_model", None)
        try:
            resolved_d_model = int(d_model)
        except (TypeError, ValueError) as exc:
            raise JLensMetadataMismatchError("lens has no valid d_model") from exc
        if resolved_d_model != self.d_model:
            raise JLensMetadataMismatchError(
                f"d_model mismatch: artifact metadata={self.d_model}, lens={d_model}"
            )
        try:
            layers = tuple(int(layer) for layer in lens.source_layers)
        except (TypeError, ValueError) as exc:
            raise JLensMetadataMismatchError(
                "lens has no valid source_layers"
            ) from exc
        if layers != self.source_layers:
            raise JLensMetadataMismatchError(
                "source_layers mismatch: "
                f"artifact metadata={self.source_layers}, lens={layers}"
            )
        jacobians = getattr(lens, "jacobians", None)
        if not isinstance(jacobians, Mapping):
            raise JLensMetadataMismatchError("lens has no jacobians mapping")
        if set(int(layer) for layer in jacobians) != set(self.source_layers):
            raise JLensMetadataMismatchError(
                "jacobian keys do not match metadata source_layers"
            )
        expected_shape = (self.d_model, self.d_model)
        for layer in self.source_layers:
            matrix = jacobians[layer]
            shape = tuple(int(dim) for dim in getattr(matrix, "shape", ()))
            if shape != expected_shape:
                raise JLensMetadataMismatchError(
                    f"J[{layer}] has shape {shape}, expected {expected_shape}"
                )
        lens_n_prompts = getattr(lens, "n_prompts", None)
        if (
            self.n_prompts is not None
            and lens_n_prompts is not None
            and int(lens_n_prompts) != self.n_prompts
        ):
            raise JLensMetadataMismatchError(
                "n_prompts mismatch: "
                f"artifact metadata={self.n_prompts}, lens={lens_n_prompts}"
            )

    def validate_expected(
        self, expected: JLensMetadata | Mapping[str, Any] | None
    ) -> None:
        """Compare requested identity fields without silently ignoring ``None``.

        A mapping can specify a subset.  A :class:`JLensMetadata` compares all
        identity and fitting fields except ``extra``.
        """

        if expected is None:
            return
        if isinstance(expected, JLensMetadata):
            expected_values: Mapping[str, Any] = expected.to_dict()
        else:
            expected_values = expected
        aliases = {"layers": "source_layers"}
        for raw_key, expected_value in expected_values.items():
            if raw_key == "extra":
                continue
            key = aliases.get(raw_key, raw_key)
            if not hasattr(self, key):
                raise JLensMetadataError(f"unknown expected metadata field {raw_key!r}")
            actual = getattr(self, key)
            if key == "source_layers":
                expected_value = tuple(int(x) for x in expected_value)
            if actual != expected_value:
                raise JLensMetadataMismatchError(
                    f"{key} mismatch: expected {expected_value!r}, found {actual!r}"
                )
