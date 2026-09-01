"""Direct external-source adapter for RAPTOR sequential adaptive steering."""

from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from jlens_workspace.foundation.artifacts import GitIdentityError, git_head_commit
from jlens_workspace.foundation.modeling import hidden_from_block_output, transformer_blocks

RAPTOR_REPOSITORY = "https://github.com/Ziqi-Gao/RAPTOR.git"
RAPTOR_COMMIT = "cf7405899174af39f3970e093e4b86bf0972ff87"


class RaptorError(ValueError):
    """Raised when external RAPTOR source or steering coordinates differ."""


@dataclass
class RaptorInterventionState:
    events: list[dict[str, Any]] = field(default_factory=list)


def verify_raptor_checkout(path: str | Path) -> Path:
    """Require the exact upstream checkout without copying unlicensed source."""

    root = Path(path).resolve()
    if not (root / ".git").exists() or not (root / "src/raptor").is_dir():
        raise RaptorError(f"RAPTOR checkout is incomplete: {root}")
    try:
        observed = git_head_commit(root)
    except GitIdentityError as error:
        raise RaptorError(
            f"cannot resolve RAPTOR checkout commit: {root}"
        ) from error
    if observed != RAPTOR_COMMIT:
        raise RaptorError(
            f"RAPTOR commit mismatch: expected {RAPTOR_COMMIT}, observed {observed}"
        )
    return root


def load_upstream_raptor(path: str | Path) -> Any:
    """Import the author's steering module from the verified external checkout."""

    root = verify_raptor_checkout(path)
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    module = importlib.import_module("raptor.steering.generate")
    module_path = Path(module.__file__).resolve()
    if root not in module_path.parents:
        raise RaptorError(
            f"loaded RAPTOR module from another checkout: {module_path}"
        )
    required = {"compute_adaptive_epsilon", "register_steering_hooks_sequential"}
    missing = sorted(name for name in required if not hasattr(module, name))
    if missing:
        raise RaptorError(f"pinned RAPTOR module lacks functions {missing}")
    return module


def verify_raptor_adaptive_epsilon_parity(
    path: str | Path,
) -> dict[str, Any]:
    """Compare the pinned author's epsilon function with the registered formula."""

    import torch

    upstream = load_upstream_raptor(path)
    direction = torch.tensor([1.0, 0.0], dtype=torch.float64)
    cases = (
        (0.9, 0.0),
        (0.9, 3.0),
        (0.1, 0.0),
        (0.1, -3.0),
    )
    results = []
    for target_probability, projection in cases:
        embedding = torch.tensor(
            [[projection, 0.0]], dtype=torch.float64
        )
        observed = upstream.compute_adaptive_epsilon(
            embedding=embedding,
            concept_vector=direction,
            bias=0.0,
            target_prob=target_probability,
        )
        target_logit = math.log(
            target_probability / (1.0 - target_probability)
        )
        current_probability = 1.0 / (1.0 + math.exp(-projection))
        should_steer = (
            current_probability < target_probability
            if target_probability > 0.5
            else current_probability > target_probability
        )
        expected = target_logit - projection if should_steer else 0.0
        if not torch.allclose(
            observed,
            torch.full_like(observed, expected),
            # The pinned implementation constructs ``logit_target`` with
            # Torch's default float32 dtype before promotion to the embedding
            # dtype.  The resulting ~4e-8 rounding is part of the upstream
            # implementation, not an algorithmic mismatch.
            rtol=1e-6,
            atol=1e-6,
        ):
            raise RaptorError(
                "pinned compute_adaptive_epsilon differs from registered formula"
            )
        results.append(
            {
                "target_probability": target_probability,
                "projection": projection,
                "epsilon": float(observed.item()),
            }
        )
    return {
        "repository": RAPTOR_REPOSITORY,
        "commit": RAPTOR_COMMIT,
        "function": "raptor.steering.generate.compute_adaptive_epsilon",
        "cases": results,
    }


def _unit_directions(directions: Mapping[int, Any]) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for layer, value in directions.items():
        vector = np.asarray(value, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if vector.ndim != 1 or not math.isfinite(norm) or norm <= 0.0:
            raise RaptorError(f"invalid concept vector at layer {layer}")
        output[int(layer)] = vector / norm
    if not output:
        raise RaptorError("RAPTOR requires at least one layer direction")
    return output


@contextmanager
def raptor_intervention_session(
    model: Any,
    *,
    directions: Mapping[int, Any],
    target_probability: float,
    upstream_checkout: str | Path,
) -> Iterator[RaptorInterventionState]:
    """Call the author's sequential hook registration and observe actual deltas."""

    if not 0.0 < target_probability < 1.0:
        raise RaptorError("target_probability must lie in (0,1)")
    import torch

    upstream = load_upstream_raptor(upstream_checkout)
    unit = _unit_directions(directions)
    blocks = transformer_blocks(model)
    layers = tuple(sorted(unit))
    vectors: list[Any | None] = [None] * len(blocks)
    for layer, vector in unit.items():
        if not 0 <= layer < len(blocks):
            raise RaptorError(f"RAPTOR layer {layer} is outside model depth")
        vectors[layer] = torch.as_tensor(vector)

    before: dict[int, tuple[Any, int, int, int]] = {}
    forward_calls = {layer: 0 for layer in layers}
    observer_handles = []
    state = RaptorInterventionState()
    for layer in layers:

        def capture_before(
            _module: Any,
            _inputs: Any,
            output: Any,
            *,
            layer_id: int = layer,
        ) -> None:
            hidden = hidden_from_block_output(output)
            if hidden.shape[0] != 1:
                raise RaptorError(
                    "author RAPTOR hook uses .item(); scoring/generation batch must be 1"
                )
            forward_call = forward_calls[layer_id]
            forward_calls[layer_id] += 1
            before[layer_id] = (
                hidden[:, -1, :].detach().clone(),
                forward_call,
                int(hidden.shape[0]),
                int(hidden.shape[1]),
            )

        observer_handles.append(blocks[layer].register_forward_hook(capture_before))

    # The pinned repository resolves ``model.model.layers`` but Qwen3.5 wraps
    # those same blocks under ``model.model.language_model.layers``.  Supply a
    # module-resolution-only view; the actual modules and all steering math
    # remain the author's objects and code.
    upstream_model_view = SimpleNamespace(
        model=SimpleNamespace(layers=blocks),
    )
    upstream_handles = upstream.register_steering_hooks_sequential(
        upstream_model_view,
        vectors,
        layers,
        target_prob=float(target_probability),
        # The pinned singlelr loader discards the stored per-layer intercept,
        # and the author's CLI passes its global default bias of zero.
        bias=0.0,
        test_accuracies=None,
        use_position_coefficients=False,
        verbose=False,
    )

    for layer in layers:

        def capture_after(
            _module: Any,
            _inputs: Any,
            output: Any,
            *,
            layer_id: int = layer,
            direction: np.ndarray = unit[layer],
        ) -> None:
            hidden = hidden_from_block_output(output)
            prior, forward_call, batch_size, sequence_length = before.pop(
                layer_id
            )
            current = hidden[:, -1, :].detach()
            delta = (current - prior).float()
            tensor_direction = torch.as_tensor(
                direction, device=delta.device, dtype=delta.dtype
            )
            epsilon = float((delta[0] @ tensor_direction).cpu())
            current_logit = float(
                (prior[0].float() @ tensor_direction).cpu()
            )
            state.events.append(
                {
                    "layer": layer_id,
                    "forward_call": forward_call,
                    "generation_step": forward_call,
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "target_probability": float(target_probability),
                    "pre_intervention_logit": current_logit,
                    "pre_intervention_probability": float(
                        torch.sigmoid(torch.tensor(current_logit))
                    ),
                    "epsilon": epsilon,
                    "injected_norm": float(torch.linalg.vector_norm(delta).cpu()),
                    "steered": abs(epsilon) > 0.0,
                }
            )

        observer_handles.append(blocks[layer].register_forward_hook(capture_after))

    try:
        yield state
    finally:
        for handle in reversed(observer_handles):
            handle.remove()
        for handle in upstream_handles:
            handle.remove()


@contextmanager
def no_raptor_intervention() -> Iterator[RaptorInterventionState]:
    """True no-hook baseline."""

    yield RaptorInterventionState()
