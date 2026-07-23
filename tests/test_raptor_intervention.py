from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jlens_workspace.concept_intervention.raptor import intervention as raptor


def test_checkout_verification_requires_pinned_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "src/raptor").mkdir(parents=True)
    monkeypatch.setattr(
        raptor.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=f"{raptor.RAPTOR_COMMIT}\n"),
    )
    assert raptor.verify_raptor_checkout(tmp_path) == tmp_path.resolve()

    monkeypatch.setattr(
        raptor.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=f"{'0' * 40}\n"),
    )
    with pytest.raises(raptor.RaptorError, match="commit mismatch"):
        raptor.verify_raptor_checkout(tmp_path)


def test_adapter_uses_author_hook_math_on_qwen35_wrapper_and_cleans_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    class FakeUpstream:
        @staticmethod
        def register_steering_hooks_sequential(
            model: object,
            vectors: list[object],
            layer_ids: tuple[int, ...],
            *,
            target_prob: float,
            bias: float,
            **kwargs: object,
        ) -> list[object]:
            assert kwargs["test_accuracies"] is None
            assert kwargs["use_position_coefficients"] is False
            blocks = model.model.layers
            handles = []
            target_logit = math.log(target_prob / (1.0 - target_prob))
            for layer in sorted(layer_ids):
                vector = vectors[layer]

                def hook(
                    _module: object,
                    _inputs: object,
                    output: object,
                    *,
                    direction: object = vector,
                ) -> object:
                    hidden = output.clone()
                    last = hidden[:, -1, :]
                    unit = direction.to(device=last.device, dtype=last.dtype)
                    current_logit = bias + torch.sum(last * unit, dim=-1, keepdim=True)
                    current_probability = torch.sigmoid(current_logit)
                    should_steer = (
                        current_probability < target_prob
                        if target_prob > 0.5
                        else current_probability > target_prob
                    )
                    epsilon = (target_logit - current_logit) * should_steer
                    hidden[:, -1, :] = last + epsilon * unit
                    return hidden

                handles.append(blocks[layer].register_forward_hook(hook))
            return handles

    monkeypatch.setattr(raptor, "load_upstream_raptor", lambda _path: FakeUpstream())
    blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    model = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(layers=blocks),
        )
    )
    source = torch.zeros((1, 1, 2))
    with raptor.raptor_intervention_session(
        model,
        directions={0: np.asarray([3.0, 0.0]), 1: np.asarray([0.0, 2.0])},
        target_probability=0.9,
        upstream_checkout="unused",
    ) as state:
        changed = blocks[1](blocks[0](source))

    expected = math.log(0.9 / 0.1)
    torch.testing.assert_close(changed, torch.tensor([[[expected, expected]]]))
    assert [event["layer"] for event in state.events] == [0, 1]
    assert all(event["epsilon"] == pytest.approx(expected) for event in state.events)
    torch.testing.assert_close(blocks[1](blocks[0](source)), source)
