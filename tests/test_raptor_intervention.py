from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jlens_workspace.artifacts import sha256_file
from jlens_workspace.concept_intervention.raptor import intervention as raptor
from jlens_workspace.concept_intervention.raptor.workflow import (
    load_raptor_directions,
)


def test_checkout_verification_requires_pinned_commit(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (tmp_path / "src/raptor").mkdir(parents=True)
    (git_dir / "HEAD").write_text(
        f"{raptor.RAPTOR_COMMIT}\n",
        encoding="utf-8",
    )
    assert raptor.verify_raptor_checkout(tmp_path) == tmp_path.resolve()

    (git_dir / "HEAD").write_text(
        f"{'0' * 40}\n",
        encoding="utf-8",
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
    assert all(event["forward_call"] == 0 for event in state.events)
    assert all(event["generation_step"] == 0 for event in state.events)
    assert all(event["sequence_length"] == 1 for event in state.events)
    torch.testing.assert_close(blocks[1](blocks[0](source)), source)


def test_direction_loader_requires_authoritative_shared_probe_hash(
    tmp_path: Path,
) -> None:
    import json

    row_manifest = tmp_path / "row_manifest.json"
    row_manifest.write_text("{}", encoding="utf-8")
    selection = tmp_path / "layer_selection.json"
    selection.write_text(
        json.dumps(
            {
                "method": "raptor_validation_accuracy",
                "selected_layer_count": 1,
                "row_manifest": "row_manifest.json",
                "row_manifest_sha256": sha256_file(row_manifest),
                "concepts": [
                    {"concept_id": "concept:a", "selected_layers": [3]}
                ],
            }
        ),
        encoding="utf-8",
    )
    probes = tmp_path / "raptor_probes"
    vector = probes / "layer_03/concept_concept%3Aa/probe_vector.npy"
    vector.parent.mkdir(parents=True)
    np.save(vector, np.asarray([1.0, 0.0]))
    (probes / "manifest.json").write_text(
        json.dumps(
            {
                "workflow": "shared_raptor_probe_fitting",
                "layer_selection_sha256": sha256_file(selection),
                "row_manifest": "../row_manifest.json",
                "row_manifest_sha256": sha256_file(row_manifest),
                "probes": [
                    {
                        "layer": 3,
                        "concept_id": "concept:a",
                        "vector_file": (
                            "layer_03/concept_concept%3Aa/probe_vector.npy"
                        ),
                        "vector_sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(raptor.RaptorError, match="hash differs"):
        load_raptor_directions(
            probes_dir=probes,
            selected_layers_path=selection,
            concept_id="concept:a",
        )


def test_pinned_checkout_compute_adaptive_epsilon_direct_parity() -> None:
    checkout = Path(
        os.environ.get(
            "RAPTOR_UPSTREAM_CHECKOUT",
            (
                "/gpfs/projects/p32737/del6500_home/J_lens/"
                "third_party_external/RAPTOR"
            ),
        )
    )
    if not checkout.is_dir():
        pytest.skip("pinned external RAPTOR checkout is not available")

    report = raptor.verify_raptor_adaptive_epsilon_parity(checkout)

    assert report["commit"] == raptor.RAPTOR_COMMIT
    assert report["function"].endswith(".compute_adaptive_epsilon")
    assert len(report["cases"]) == 4
