from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jlens_workspace._vendor.honest_llama_core import (
    get_com_directions,
    get_interventions_dict,
    train_probes,
)
from jlens_workspace.concept_intervention.iti import (
    ITIError,
    ITIHeadShift,
    capture_iti_head_activations,
    fit_iti_concept_directions,
    iti_intervention_session,
    layer_matched_head_order,
    layer_matched_random_head_order,
    load_iti_head_shifts,
    validate_iti_direction_artifact,
)


def test_vendored_iti_core_ranks_and_scales_synthetic_head_signal() -> None:
    rng = np.random.default_rng(7)
    labels = np.asarray([0, 1] * 20, dtype=np.int8)
    activations = rng.normal(scale=0.1, size=(40, 1, 2, 3))
    activations[:, 0, 0, 0] += labels * 3.0 - 1.5
    separated_x = [activations[index : index + 1] for index in range(40)]
    separated_y = [labels[index : index + 1] for index in range(40)]
    train = np.arange(0, 30)
    validation = np.arange(30, 40)

    probes, accuracies = train_probes(
        42, train, validation, separated_x, separated_y, 1, 2
    )
    assert accuracies[0] == 1.0
    assert accuracies[0] > accuracies[1]

    directions = get_com_directions(
        1, 2, train, validation, separated_x, separated_y
    )
    assert directions[0, 0] > 0
    interventions = get_interventions_dict(
        [(0, 0)], probes, activations, 2, True, False, directions, head_dim=3
    )
    head, direction, projection_std = interventions[
        "model.layers.0.self_attn.head_out"
    ][0]
    assert head == 0
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0)
    assert projection_std > 0


def test_layer_matched_order_covers_every_layer_before_global_remainder() -> None:
    accuracies = np.asarray(
        [
            0.99,
            0.98,
            0.60,
            0.59,
            0.70,
            0.69,
        ]
    )
    order = layer_matched_head_order(
        accuracies,
        num_layers=3,
        num_heads=2,
    )

    assert order[:3].tolist() == [0, 4, 2]
    assert {int(value) // 2 for value in order[:3]} == {0, 1, 2}
    assert order[3:].tolist() == [1, 5, 3]


@pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
def test_layer_matched_random_order_covers_all_layers_at_k8(seed: int) -> None:
    np.random.seed(seed)
    native_order = np.random.choice(96, 96, replace=False)

    order = layer_matched_random_head_order(
        native_order,
        num_layers=6,
        num_heads=16,
    )

    assert len(set((order[:8] // 16).tolist())) == 6
    first_by_native_encounter = []
    seen_layers = set()
    for value in native_order:
        layer = int(value) // 16
        if layer not in seen_layers:
            seen_layers.add(layer)
            first_by_native_encounter.append(int(value))
    assert order[:6].tolist() == first_by_native_encounter
    assert sorted(order.tolist()) == list(range(96))


def test_fit_iti_uses_train_validation_only_and_writes_original_modes(
    tmp_path: Path,
) -> None:
    activation_dir = tmp_path / "activations"
    activation_dir.mkdir()
    rows = []
    labels = []
    rng = np.random.default_rng(11)
    activations = []
    for split in ("train", "validation", "test"):
        for index in range(8):
            label = index % 2
            rows.append({"row": len(rows), "split": split, "group_id": f"{split}-{index}"})
            labels.append([label])
            value = rng.normal(scale=0.05, size=(2, 3))
            value[0, 0] += 2.0 * label - 1.0
            activations.append(value)
    np.save(activation_dir / "labels.npy", np.asarray(labels, dtype=np.int8))
    np.save(activation_dir / "layer_03.npy", np.asarray(activations, dtype=np.float32))
    (activation_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (activation_dir / "concepts.json").write_text(
        json.dumps({"concepts": [{"column": 0, "concept_id": "concept:a"}]}),
        encoding="utf-8",
    )
    (activation_dir / "metadata.json").write_text(
        json.dumps(
            {
                "method": "honest_llama_mass_mean_qwen_full_attention_v1",
                "layers": [3],
                "num_heads": 2,
                "head_dim": 3,
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "directions"
    metrics = fit_iti_concept_directions(
        activation_dir=activation_dir,
        output_dir=output,
        concept_id="concept:a",
        max_top_k=2,
        random_seeds=[17, 29],
        seed=42,
    )
    assert metrics["train_examples"] == 8
    assert metrics["validation_examples"] == 8
    assert metrics["test_examples_used"] == 0
    assert set(metrics["files"]) == {
        "mass_mean",
        "probe_weight",
        "random",
        "development_indices",
    }
    shifts = load_iti_head_shifts(
        output, concept_id="concept:a", mode="mass_mean", top_k=1
    )
    assert len(shifts) == 1
    assert shifts[0].direction.shape == (3,)

    direction_path = output / "concept%3Aa" / "mass_mean.npz"
    with np.load(direction_path, allow_pickle=False) as payload:
        changed = {key: np.asarray(payload[key]).copy() for key in payload.files}
    changed["direction"][0] = np.roll(changed["direction"][0], 1)
    np.savez(direction_path, **changed)
    with pytest.raises(ITIError, match="identity"):
        validate_iti_direction_artifact(
            output / "concept%3Aa" / "metrics.json"
        )
    with pytest.raises(ITIError, match="identity"):
        load_iti_head_shifts(
            output, concept_id="concept:a", mode="mass_mean", top_k=1
        )


def test_capture_iti_writes_last_nonpadding_pre_projection_heads(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head_dim = 2
            self.o_proj = torch.nn.Linear(4, 4, bias=False)

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = Attention()

        def forward(self, hidden: object) -> object:
            return self.self_attn.o_proj(hidden)

    class Inner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([Block()])

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Inner()
            self.embedding = torch.nn.Embedding(10, 4)

        def get_input_embeddings(self) -> object:
            return self.embedding

        def forward(self, input_ids: object, **_kwargs: object) -> object:
            hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
            for block in self.model.layers:
                hidden = block(hidden)
            return SimpleNamespace(logits=hidden)

    class Tokenizer:
        padding_side = "left"

        def __call__(self, _texts: object, **_kwargs: object) -> dict[str, object]:
            return {
                "input_ids": torch.asarray([[1, 2, 0], [3, 0, 0]]),
                "attention_mask": torch.asarray([[1, 1, 0], [1, 0, 0]]),
            }

    examples = [
        {
            "concept_id": "concept:a",
            "concept_name": "A",
            "definition": "A concept",
            "label": label,
            "text": f"text {index}",
            "split": "train",
            "group_id": f"group-{index}",
            "source": "synthetic",
            "license": "CC0",
        }
        for index, label in enumerate((0, 1))
    ]
    output = capture_iti_head_activations(
        model=Model(),
        tokenizer=Tokenizer(),
        examples=examples,
        layers=[0],
        output_dir=tmp_path / "capture",
        batch_size=2,
        expected_num_heads=2,
        expected_head_dim=2,
    )
    values = np.load(output / "layer_00.npy", allow_pickle=False)
    assert values.shape == (2, 2, 2)
    np.testing.assert_allclose(values[0], np.full((2, 2), 2.0))
    np.testing.assert_allclose(values[1], np.full((2, 2), 3.0))


def test_iti_hook_changes_only_last_token_and_is_removed() -> None:
    torch = pytest.importorskip("torch")

    class RecordingLinear(torch.nn.Linear):
        def forward(self, value: object) -> object:
            self.observed = value.detach().clone()
            return super().forward(value)

    projection = RecordingLinear(6, 4, bias=False)
    attention = SimpleNamespace(o_proj=projection, head_dim=3)
    block = SimpleNamespace(self_attn=attention)
    model = SimpleNamespace(model=SimpleNamespace(layers=[block]))
    shift = ITIHeadShift(
        rank=0,
        layer=0,
        head=1,
        validation_accuracy=1.0,
        direction=np.asarray([1.0, 0.0, 0.0]),
        projection_std=2.0,
    )
    source = torch.zeros((2, 4, 6))
    with iti_intervention_session(
        model, shifts=[shift], multiplier=3.0, num_heads=2, head_dim=3
    ) as state:
        projection(source)
        changed = projection.observed
    projection(source)
    restored = projection.observed
    expected = torch.zeros_like(source)
    expected[:, -1, 3] = 6.0
    torch.testing.assert_close(changed, expected)
    torch.testing.assert_close(restored, source)
    assert state.events == [
        {
            "layer": 0,
            "forward_call": 0,
            "generation_step": 0,
            "batch_size": 2,
            "sequence_length": 4,
            "active_positions": 2,
            "multiplier": 3.0,
            "injected_norm": pytest.approx(6.0),
        }
    ]
