"""Explicit effective-unembedding conventions for J-lens token directions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from ._optional import torch


class UnsupportedNormalizationError(ValueError):
    """The requested linearization is not valid for the supplied final norm."""


class UnembeddingConvention(StrEnum):
    """Linear token-frame convention used to construct ``A_l``.

    ``RAW`` is exactly ``W_U J_l``.  ``RMSNORM_WEIGHTED`` folds only RMSNorm's
    learned per-channel scale into the unembedding, producing
    ``W_U diag(gamma) J_l``.  The activation-dependent reciprocal RMS is a
    scalar and is intentionally omitted from both conventions.
    """

    RAW = "raw"
    RMSNORM_WEIGHTED = "rmsnorm_weighted"

    @classmethod
    def parse(cls, value: UnembeddingConvention | str) -> UnembeddingConvention:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"unsupported unembedding convention {value!r}; choose {choices}"
            ) from exc


@dataclass(frozen=True)
class EffectiveUnembeddingMetadata:
    convention: str
    vocab_size: int
    unembedding_vocab_size: int
    d_model: int
    source_dtype: str
    source_device: str
    norm_type: str | None
    norm_epsilon: float | None
    norm_scale_parameterization: str | None
    model_id: str | None = None
    model_revision: str | None = None
    activation_dependent_scale_omitted: bool = True
    unembedding_bias_omitted: bool = True
    unembedding_bias_present: bool = False
    row_selection: str = "all_unembedding_rows"

    def to_dict(self) -> dict[str, object]:
        return {
            "convention": self.convention,
            "vocab_size": self.vocab_size,
            "unembedding_vocab_size": self.unembedding_vocab_size,
            "d_model": self.d_model,
            "source_dtype": self.source_dtype,
            "source_device": self.source_device,
            "norm_type": self.norm_type,
            "norm_epsilon": self.norm_epsilon,
            "norm_scale_parameterization": self.norm_scale_parameterization,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "activation_dependent_scale_omitted": (
                self.activation_dependent_scale_omitted
            ),
            "unembedding_bias_omitted": self.unembedding_bias_omitted,
            "unembedding_bias_present": self.unembedding_bias_present,
            "row_selection": self.row_selection,
        }


@dataclass(frozen=True)
class EffectiveUnembedding:
    """A row-sliceable effective unembedding without a second ``V x D`` copy."""

    weight: torch.Tensor
    column_scale: torch.Tensor | None
    metadata: EffectiveUnembeddingMetadata

    def __post_init__(self) -> None:
        if self.weight.ndim != 2:
            raise ValueError("unembedding weight must have shape [vocab_size, d_model]")
        if self.column_scale is not None:
            if self.column_scale.ndim != 1:
                raise ValueError("column_scale must be one-dimensional")
            if self.column_scale.shape[0] != self.weight.shape[1]:
                raise ValueError("column_scale length must equal d_model")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.weight.shape[0]), int(self.weight.shape[1])

    @property
    def vocab_size(self) -> int:
        return self.shape[0]

    @property
    def d_model(self) -> int:
        return self.shape[1]

    def rows(
        self,
        start: int,
        stop: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return effective rows ``[start:stop]``, applying RMS scale lazily."""

        if not 0 <= start <= stop <= self.vocab_size:
            raise IndexError(
                f"row slice [{start}:{stop}] is outside vocab size {self.vocab_size}"
            )
        block = self.weight[start:stop]
        target_device = block.device if device is None else torch.device(device)
        target_dtype = block.dtype if dtype is None else dtype
        block = block.to(device=target_device, dtype=target_dtype)
        if self.column_scale is not None:
            scale = self.column_scale.to(device=target_device, dtype=target_dtype)
            block = block * scale.unsqueeze(0)
        return block


def restrict_effective_unembedding(
    effective: EffectiveUnembedding, vocab_size: int
) -> EffectiveUnembedding:
    """Restrict padded model output rows to the tokenizer's valid ID range.

    Some causal-LM checkpoints pad the output head beyond ``len(tokenizer)``.
    The returned weight is a view over rows ``[0, vocab_size)`` rather than a
    copy, so all analyzed rows correspond one-to-one with decodable token IDs.
    """

    size = int(vocab_size)
    if size < 1:
        raise ValueError("tokenizer vocabulary size must be positive")
    if size > effective.vocab_size:
        raise ValueError(
            f"tokenizer has {size} IDs but the model unembedding has only "
            f"{effective.vocab_size} rows"
        )
    if size == effective.vocab_size:
        return effective
    metadata = replace(
        effective.metadata,
        vocab_size=size,
        row_selection=f"tokenizer_ids_[0,{size})",
    )
    return EffectiveUnembedding(
        weight=effective.weight[:size],
        column_scale=effective.column_scale,
        metadata=metadata,
    )


def _resolve_attr_path(obj: object, path: str) -> Any:
    value: Any = obj
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _extract_unembedding(source: Any) -> tuple[torch.Tensor, Any | None]:
    if torch.is_tensor(source):
        return source, None
    # A Linear-like module passed directly.
    direct_weight = getattr(source, "weight", None)
    if torch.is_tensor(direct_weight) and direct_weight.ndim == 2:
        return direct_weight, source
    candidates = ("_lm_head", "lm_head", "embed_out")
    for name in candidates:
        module = getattr(source, name, None)
        weight = getattr(module, "weight", None)
        if torch.is_tensor(weight) and weight.ndim == 2:
            return weight, module
    getter = getattr(source, "get_output_embeddings", None)
    if callable(getter):
        module = getter()
        weight = getattr(module, "weight", None)
        if torch.is_tensor(weight) and weight.ndim == 2:
            return weight, module
    raise ValueError(
        "could not locate a [vocab_size, d_model] unembedding weight; pass a "
        "Tensor, Linear-like output head, official HFLensModel, or HF CausalLM"
    )


def _extract_final_norm(source: Any) -> Any | None:
    for path in (
        "_final_norm",
        "model.norm",
        "model.language_model.norm",
        "language_model.norm",
        "transformer.ln_f",
        "gpt_neox.final_layer_norm",
    ):
        try:
            return _resolve_attr_path(source, path)
        except AttributeError:
            continue
    return None


def _nonempty_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _infer_model_identity(source: Any) -> tuple[str | None, str | None]:
    """Best-effort immutable identity from an HF model or official wrapper."""

    hf_model = getattr(source, "_hf_model", source)
    config = getattr(hf_model, "config", None)
    model_id = _nonempty_text(getattr(hf_model, "name_or_path", None))
    if model_id is None and config is not None:
        model_id = _nonempty_text(
            getattr(config, "_name_or_path", getattr(config, "name_or_path", None))
        )
    model_revision = _nonempty_text(getattr(hf_model, "_commit_hash", None))
    if model_revision is None and config is not None:
        model_revision = _nonempty_text(getattr(config, "_commit_hash", None))
    return model_id, model_revision


def _is_rmsnorm(module: object) -> bool:
    rmsnorm_type = getattr(torch.nn, "RMSNorm", None)
    if rmsnorm_type is not None and isinstance(module, rmsnorm_type):
        return True
    # HF defines architecture-specific RMSNorm classes.  Requiring the class
    # name to say RMSNorm avoids silently treating LayerNorm or an arbitrary
    # learned affine transform as RMSNorm.
    return "rmsnorm" in type(module).__name__.lower().replace("_", "")


def build_effective_unembedding(
    source: Any,
    *,
    convention: UnembeddingConvention | str = UnembeddingConvention.RAW,
    norm: Any | None = None,
    storage_device: torch.device | str | None = None,
    storage_dtype: torch.dtype | None = None,
    rmsnorm_weight_offset: float | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
) -> EffectiveUnembedding:
    """Construct a row-streamable effective unembedding.

    ``storage_device``/``storage_dtype`` explicitly move the underlying raw
    unembedding.  They default to preserving the model weight exactly.
    """

    convention = UnembeddingConvention.parse(convention)
    inferred_model_id, inferred_model_revision = _infer_model_identity(source)
    if model_id is not None and _nonempty_text(model_id) is None:
        raise ValueError("model_id must be a non-empty string when provided")
    if model_revision is not None and _nonempty_text(model_revision) is None:
        raise ValueError("model_revision must be a non-empty string when provided")
    resolved_model_id = _nonempty_text(model_id) or inferred_model_id
    resolved_model_revision = _nonempty_text(model_revision) or inferred_model_revision
    weight, head = _extract_unembedding(source)
    if weight.ndim != 2:
        raise ValueError("unembedding weight must be two-dimensional")
    if storage_device is not None or storage_dtype is not None:
        weight = weight.to(
            device=weight.device if storage_device is None else storage_device,
            dtype=weight.dtype if storage_dtype is None else storage_dtype,
        )
    resolved_norm = norm if norm is not None else _extract_final_norm(source)
    column_scale = None
    norm_scale_parameterization = None
    norm_type = type(resolved_norm).__name__ if resolved_norm is not None else None
    norm_epsilon = None
    if resolved_norm is not None:
        epsilon = getattr(
            resolved_norm,
            "variance_epsilon",
            getattr(resolved_norm, "eps", None),
        )
        if epsilon is not None:
            norm_epsilon = float(epsilon)
    if convention is UnembeddingConvention.RMSNORM_WEIGHTED:
        if resolved_norm is None:
            raise UnsupportedNormalizationError(
                "rmsnorm_weighted requires an explicit or discoverable final RMSNorm"
            )
        if not _is_rmsnorm(resolved_norm):
            raise UnsupportedNormalizationError(
                "rmsnorm_weighted only supports RMSNorm; found "
                f"{type(resolved_norm).__name__}. Use convention='raw' explicitly "
                "or implement a norm-specific linearization."
            )
        scale = getattr(resolved_norm, "weight", None)
        if not torch.is_tensor(scale) or scale.ndim != 1:
            raise UnsupportedNormalizationError(
                "final RMSNorm has no one-dimensional learned weight"
            )
        if int(scale.shape[0]) != int(weight.shape[1]):
            raise UnsupportedNormalizationError(
                "final RMSNorm weight length does not match unembedding d_model"
            )
        if rmsnorm_weight_offset is None:
            # Hugging Face Gemma-family RMSNorm stores a zero-initialized
            # offset and applies (1 + weight); most other RMSNorm modules store
            # the multiplier directly.  Record the choice so spectra from the
            # two conventions cannot be mixed unknowingly.
            rmsnorm_weight_offset = 1.0 if "gemma" in norm_type.lower() else 0.0
        offset = float(rmsnorm_weight_offset)
        column_scale = scale.detach() + offset
        norm_scale_parameterization = (
            "one_plus_weight" if offset == 1.0 else f"weight_plus_{offset:g}"
        )
    bias = getattr(head, "bias", None) if head is not None else None
    metadata = EffectiveUnembeddingMetadata(
        convention=convention.value,
        vocab_size=int(weight.shape[0]),
        unembedding_vocab_size=int(weight.shape[0]),
        d_model=int(weight.shape[1]),
        source_dtype=str(weight.dtype),
        source_device=str(weight.device),
        norm_type=norm_type,
        norm_epsilon=norm_epsilon,
        norm_scale_parameterization=norm_scale_parameterization,
        model_id=resolved_model_id,
        model_revision=resolved_model_revision,
        unembedding_bias_present=bias is not None,
    )
    return EffectiveUnembedding(
        weight=weight.detach(), column_scale=column_scale, metadata=metadata
    )
