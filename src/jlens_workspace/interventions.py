"""Generation-time residual-stream interventions with removable HF hooks."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from jlens_workspace.modeling import (
    hidden_from_block_output,
    model_input_device,
    register_resid_post_hook,
    replace_hidden_in_block_output,
)

PositionPolicy = Literal["last_prompt", "generated", "last_prompt_and_generated", "all"]
InterventionKind = Literal["addition", "project_out"]


@dataclass
class ResidualIntervention:
    """Stateful hook used for both prefill and cached decoding calls."""

    direction: Any
    strength: float
    kind: InterventionKind = "addition"
    position: PositionPolicy = "last_prompt_and_generated"
    residual_norm: float = 1.0
    _calls: int = 0

    def reset(self) -> None:
        self._calls = 0

    def _position_mask(self, hidden: Any) -> Any:
        import torch

        batch, sequence, _ = hidden.shape
        mask = torch.zeros((batch, sequence), dtype=torch.bool, device=hidden.device)
        is_prefill = self._calls == 0
        if self.position == "all":
            mask[:] = True
        elif is_prefill and self.position in {"last_prompt", "last_prompt_and_generated"}:
            mask[:, -1] = True
        elif not is_prefill and self.position in {"generated", "last_prompt_and_generated"}:
            # With KV caching sequence=1. Without caching, the new token is last.
            mask[:, -1] = True
        return mask

    def __call__(self, _module: Any, _inputs: Any, output: Any) -> Any:
        import torch

        hidden = hidden_from_block_output(output)
        if hidden.ndim != 3:
            raise ValueError(f"expected [batch, sequence, d_model], got {hidden.shape}")
        direction = torch.as_tensor(self.direction, device=hidden.device, dtype=hidden.dtype)
        if direction.ndim != 1 or direction.shape[0] != hidden.shape[-1]:
            raise ValueError(
                f"direction shape {tuple(direction.shape)} does not match d_model={hidden.shape[-1]}"
            )
        direction = direction / direction.float().norm().clamp_min(1e-12).to(hidden.dtype)
        mask = self._position_mask(hidden)
        self._calls += 1
        if not mask.any() or self.strength == 0:
            return output
        updated = hidden.clone()
        selected = updated[mask]
        if self.kind == "addition":
            selected = selected + (self.strength * self.residual_norm) * direction
        elif self.kind == "project_out":
            coefficient = selected.float() @ direction.float()
            selected = selected - self.strength * coefficient.to(hidden.dtype).unsqueeze(-1) * direction
        else:  # pragma: no cover - Literal plus runtime protection
            raise ValueError(f"unknown intervention kind: {self.kind}")
        updated[mask] = selected
        return replace_hidden_in_block_output(output, updated)


@contextmanager
def intervention_session(
    model: Any,
    layer: int,
    direction: Any,
    strength: float,
    *,
    kind: InterventionKind = "addition",
    position: PositionPolicy = "last_prompt_and_generated",
    residual_norm: float = 1.0,
) -> Iterator[ResidualIntervention]:
    intervention = ResidualIntervention(
        direction=direction,
        strength=strength,
        kind=kind,
        position=position,
        residual_norm=residual_norm,
    )
    handle = register_resid_post_hook(model, layer, intervention)
    try:
        yield intervention
    finally:
        handle.remove()


def generate_with_intervention(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer: int,
    direction: Any,
    strength: float,
    kind: InterventionKind = "addition",
    position: PositionPolicy = "last_prompt_and_generated",
    residual_norm: float = 1.0,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 1.0,
    seed: int = 42,
) -> str:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(model_input_device(model)) for key, value in encoded.items()}
    with intervention_session(
        model,
        layer,
        direction,
        strength,
        kind=kind,
        position=position,
        residual_norm=residual_norm,
    ):
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
        output = model.generate(**encoded, **generation_kwargs)
    generated = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def matched_random_direction(direction: Any, seed: int = 42) -> Any:
    """Return a deterministic random unit vector orthogonal to ``direction``."""

    import torch

    source = torch.as_tensor(direction)
    generator = torch.Generator(device=source.device)
    generator.manual_seed(seed)
    random = torch.randn(source.shape, generator=generator, device=source.device, dtype=source.dtype)
    unit = source / source.float().norm().clamp_min(1e-12).to(source.dtype)
    random = random - (random.float() @ unit.float()).to(source.dtype) * unit
    return random / random.float().norm().clamp_min(1e-12).to(random.dtype)
