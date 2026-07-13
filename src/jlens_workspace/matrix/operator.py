"""Blockwise linear operator for the token frame ``A_l = U_eff J_l``."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from jlens_workspace.jacobian import EffectiveUnembedding
from jlens_workspace.jacobian._optional import torch


class MatrixDeviceError(RuntimeError):
    """A requested compute device is unavailable and fallback is disabled."""


def _is_cuda_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, oom_type) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def _resolve_device(
    requested: torch.device | str | None,
    *,
    preferred: torch.device,
    cpu_fallback: bool,
) -> tuple[torch.device, bool]:
    if requested is None or str(requested) == "auto":
        device = preferred
    else:
        device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        if not cpu_fallback:
            raise MatrixDeviceError(
                f"requested {device}, but CUDA is unavailable and cpu_fallback=False"
            )
        return torch.device("cpu"), True
    return device, False


@dataclass(frozen=True)
class TokenRowBlock:
    start: int
    stop: int
    rows: torch.Tensor
    compute_device: str
    compute_dtype: str
    used_cpu_fallback: bool = False

    def __post_init__(self) -> None:
        if self.rows.ndim != 2 or self.rows.shape[0] != self.stop - self.start:
            raise ValueError("rows do not match TokenRowBlock bounds")


@dataclass(frozen=True)
class TokenFrameOperatorMetadata:
    layer: int | None
    vocab_size: int
    d_model: int
    convention: str
    jacobian_dtype: str
    jacobian_device: str
    unembedding_dtype: str
    unembedding_device: str
    requested_device: str
    resolved_device: str
    compute_dtype: str
    block_size: int
    cpu_fallback_enabled: bool
    initial_cpu_fallback: bool
    effective_unembedding: dict[str, object]
    provenance: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


class TokenFrameOperator:
    """Stream rows of ``A_l = U_eff J_l`` without materializing ``V x D``.

    The ``D x D`` Jacobian is moved to the compute device once per iteration.
    Unembedding rows are transferred and optionally RMSNorm-scaled one block at
    a time.  If a CUDA allocation or multiplication runs out of memory,
    ``cpu_fallback=True`` retries the current and remaining blocks on CPU.
    """

    def __init__(
        self,
        jacobian: torch.Tensor,
        unembedding: EffectiveUnembedding,
        *,
        layer: int | None = None,
        block_size: int = 4096,
        compute_device: torch.device | str | None = None,
        compute_dtype: torch.dtype | None = None,
        cpu_fallback: bool = True,
        provenance: Mapping[str, object] | None = None,
    ) -> None:
        if jacobian.ndim != 2 or jacobian.shape[0] != jacobian.shape[1]:
            raise ValueError("jacobian must have shape [d_model, d_model]")
        d_model = int(jacobian.shape[0])
        if unembedding.d_model != d_model:
            raise ValueError(
                f"unembedding d_model={unembedding.d_model} does not match "
                f"jacobian d_model={d_model}"
            )
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        preferred = jacobian.device
        if preferred.type == "cpu" and unembedding.weight.device.type != "cpu":
            preferred = unembedding.weight.device
        resolved, initial_fallback = _resolve_device(
            compute_device, preferred=preferred, cpu_fallback=cpu_fallback
        )
        if compute_dtype is None:
            if jacobian.dtype == torch.float64 and unembedding.weight.dtype == torch.float64:
                compute_dtype = torch.float64
            else:
                compute_dtype = torch.float32
        if not compute_dtype.is_floating_point:
            raise ValueError("compute_dtype must be floating point")
        self.jacobian = jacobian.detach()
        self.unembedding = unembedding
        self.layer = int(layer) if layer is not None else None
        self.block_size = int(block_size)
        self.compute_device = resolved
        self.compute_dtype = compute_dtype
        self.cpu_fallback = bool(cpu_fallback)
        self._requested_device = (
            "auto" if compute_device is None else str(compute_device)
        )
        self._initial_cpu_fallback = initial_fallback
        self.provenance = dict(provenance or {})

    @classmethod
    def from_lens(
        cls,
        lens: Any,
        layer: int,
        unembedding: EffectiveUnembedding,
        **kwargs: Any,
    ) -> TokenFrameOperator:
        jacobians: Mapping[int, torch.Tensor] = lens.jacobians
        if layer not in jacobians:
            raise KeyError(f"layer {layer} is not present in the fitted lens")
        inferred: dict[str, object] = {}
        lens_metadata = getattr(lens, "metadata", None)
        if lens_metadata is not None:
            serializer = getattr(lens_metadata, "to_dict", None)
            inferred = dict(serializer() if callable(serializer) else lens_metadata)
        supplied = kwargs.pop("provenance", None)
        if supplied is not None:
            inferred.update(dict(supplied))
        return cls(
            jacobians[layer],
            unembedding,
            layer=layer,
            provenance=inferred,
            **kwargs,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.unembedding.vocab_size, int(self.jacobian.shape[0])

    @property
    def vocab_size(self) -> int:
        return self.shape[0]

    @property
    def d_model(self) -> int:
        return self.shape[1]

    @property
    def metadata(self) -> TokenFrameOperatorMetadata:
        return TokenFrameOperatorMetadata(
            layer=self.layer,
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            convention=self.unembedding.metadata.convention,
            jacobian_dtype=str(self.jacobian.dtype),
            jacobian_device=str(self.jacobian.device),
            unembedding_dtype=str(self.unembedding.weight.dtype),
            unembedding_device=str(self.unembedding.weight.device),
            requested_device=self._requested_device,
            resolved_device=str(self.compute_device),
            compute_dtype=str(self.compute_dtype),
            block_size=self.block_size,
            cpu_fallback_enabled=self.cpu_fallback,
            initial_cpu_fallback=self._initial_cpu_fallback,
            effective_unembedding=self.unembedding.metadata.to_dict(),
            provenance=self.provenance,
        )

    def iter_rows(self, *, block_size: int | None = None) -> Iterator[TokenRowBlock]:
        size = self.block_size if block_size is None else int(block_size)
        if size <= 0:
            raise ValueError("block_size must be positive")
        device = self.compute_device
        used_cpu_fallback = self._initial_cpu_fallback
        try:
            jacobian = self.jacobian.to(device=device, dtype=self.compute_dtype)
        except RuntimeError as exc:
            if not (self.cpu_fallback and device.type == "cuda" and _is_cuda_oom(exc)):
                raise
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            used_cpu_fallback = True
            jacobian = self.jacobian.to(device="cpu", dtype=self.compute_dtype)

        for start in range(0, self.vocab_size, size):
            stop = min(start + size, self.vocab_size)
            try:
                rows = self.unembedding.rows(
                    start, stop, device=device, dtype=self.compute_dtype
                )
                rows = rows @ jacobian
            except RuntimeError as exc:
                if not (
                    self.cpu_fallback and device.type == "cuda" and _is_cuda_oom(exc)
                ):
                    raise
                torch.cuda.empty_cache()
                device = torch.device("cpu")
                used_cpu_fallback = True
                jacobian = self.jacobian.to(device="cpu", dtype=self.compute_dtype)
                rows = self.unembedding.rows(
                    start, stop, device="cpu", dtype=self.compute_dtype
                )
                rows = rows @ jacobian
            yield TokenRowBlock(
                start=start,
                stop=stop,
                rows=rows,
                compute_device=str(device),
                compute_dtype=str(self.compute_dtype),
                used_cpu_fallback=used_cpu_fallback,
            )

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        """Compute ``A_l @ vector`` while retaining only a ``V`` output."""

        if vector.ndim != 1 or int(vector.shape[0]) != self.d_model:
            raise ValueError(f"vector must have shape [{self.d_model}]")
        blocks: list[torch.Tensor] = []
        for block in self.iter_rows():
            local = vector.to(device=block.rows.device, dtype=block.rows.dtype)
            blocks.append((block.rows @ local).cpu())
        return torch.cat(blocks, dim=0)

    def rmatvec(self, token_weights: torch.Tensor) -> torch.Tensor:
        """Compute ``A_l.T @ token_weights`` with streaming token rows."""

        if token_weights.ndim != 1 or int(token_weights.shape[0]) != self.vocab_size:
            raise ValueError(f"token_weights must have shape [{self.vocab_size}]")
        result = torch.zeros(self.d_model, dtype=torch.float64, device="cpu")
        for block in self.iter_rows():
            weights = token_weights[block.start : block.stop].to(
                device=block.rows.device, dtype=block.rows.dtype
            )
            result += (block.rows.T @ weights).to(device="cpu", dtype=torch.float64)
        return result


# A concise mathematical alias used by analysis notebooks.
ALOperator = TokenFrameOperator
