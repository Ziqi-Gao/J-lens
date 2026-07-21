"""Core ITI routines adapted from ``likenneth/honest_llama``.

Source: https://github.com/likenneth/honest_llama
Pinned commit: 2c6b2179be7b5aa8f0a171688cf9e01b812ca327
Original files: ``legacy/llama_utils.py`` and
``legacy/llama_validate_2fold.py``.

Copyright (c) 2023 Kenneth Li. Licensed under the MIT License; the complete
license is retained at ``third_party/honest_llama/LICENSE``.

The functions below intentionally preserve the original algorithm and naming:
one logistic probe per head, validation-accuracy head ranking, mass-mean
directions, standard-deviation scaling, and a last-token vector addition. The
only generalized argument is ``head_dim`` for random controls because the
original LLaMA implementation hard-coded 128 while Qwen3.5-4B uses 256 for its
full-attention heads.

MIT License

Copyright (c) 2023 Kenneth Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def flattened_idx_to_layer_head(flattened_idx: int, num_heads: int) -> tuple[int, int]:
    return flattened_idx // num_heads, flattened_idx % num_heads


def layer_head_to_flattened_idx(layer: int, head: int, num_heads: int) -> int:
    return layer * num_heads + head


def train_probes(
    seed: int,
    train_set_idxs: np.ndarray,
    val_set_idxs: np.ndarray,
    separated_head_wise_activations: list[np.ndarray],
    separated_labels: list[np.ndarray],
    num_layers: int,
    num_heads: int,
) -> tuple[list[Any], np.ndarray]:
    """Train the original one-logistic-regression-per-head ITI probes."""

    all_head_accs = []
    probes = []

    all_X_train = np.concatenate(
        [separated_head_wise_activations[i] for i in train_set_idxs], axis=0
    )
    all_X_val = np.concatenate(
        [separated_head_wise_activations[i] for i in val_set_idxs], axis=0
    )
    y_train = np.concatenate([separated_labels[i] for i in train_set_idxs], axis=0)
    y_val = np.concatenate([separated_labels[i] for i in val_set_idxs], axis=0)

    for layer in range(num_layers):
        for head in range(num_heads):
            X_train = all_X_train[:, layer, head, :]
            X_val = all_X_val[:, layer, head, :]

            clf = LogisticRegression(random_state=seed, max_iter=1000).fit(
                X_train, y_train
            )
            y_val_pred = clf.predict(X_val)
            all_head_accs.append(accuracy_score(y_val, y_val_pred))
            probes.append(clf)

    return probes, np.array(all_head_accs)


def get_top_heads(
    train_idxs: np.ndarray,
    val_idxs: np.ndarray,
    separated_activations: list[np.ndarray],
    separated_labels: list[np.ndarray],
    num_layers: int,
    num_heads: int,
    seed: int,
    num_to_intervene: int,
    use_random_dir: bool = False,
) -> tuple[list[tuple[int, int]], list[Any]]:
    """Rank heads exactly as the original ITI implementation."""

    probes, all_head_accs_np = train_probes(
        seed,
        train_idxs,
        val_idxs,
        separated_activations,
        separated_labels,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads)

    top_accs = np.argsort(all_head_accs_np.reshape(num_heads * num_layers))[::-1][
        :num_to_intervene
    ]
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]
    if use_random_dir:
        random_idxs = np.random.choice(
            num_heads * num_layers, num_heads * num_layers, replace=False
        )
        top_heads = [
            flattened_idx_to_layer_head(idx, num_heads)
            for idx in random_idxs[:num_to_intervene]
        ]

    return top_heads, probes


def get_interventions_dict(
    top_heads: list[tuple[int, int]],
    probes: list[Any],
    tuning_activations: np.ndarray,
    num_heads: int,
    use_center_of_mass: bool,
    use_random_dir: bool,
    com_directions: np.ndarray | None,
    *,
    head_dim: int = 128,
) -> dict[str, list[tuple[int, np.ndarray, float]]]:
    """Build unit directions and per-head projection standard deviations."""

    interventions: dict[str, list[tuple[int, np.ndarray, float]]] = {}
    for layer, _head in top_heads:
        interventions[f"model.layers.{layer}.self_attn.head_out"] = []

    for layer, head in top_heads:
        if use_center_of_mass:
            assert com_directions is not None
            direction = com_directions[
                layer_head_to_flattened_idx(layer, head, num_heads)
            ]
        elif use_random_dir:
            direction = np.random.normal(size=(head_dim,))
        else:
            direction = probes[
                layer_head_to_flattened_idx(layer, head, num_heads)
            ].coef_
        direction = direction / np.linalg.norm(direction)
        activations = tuning_activations[:, layer, head, :]
        proj_vals = activations @ direction.T
        proj_val_std = np.std(proj_vals)
        interventions[f"model.layers.{layer}.self_attn.head_out"].append(
            (head, direction.squeeze(), float(proj_val_std))
        )
    for layer, _head in top_heads:
        key = f"model.layers.{layer}.self_attn.head_out"
        interventions[key] = sorted(interventions[key], key=lambda value: value[0])
    return interventions


def get_com_directions(
    num_layers: int,
    num_heads: int,
    train_set_idxs: np.ndarray,
    val_set_idxs: np.ndarray,
    separated_head_wise_activations: list[np.ndarray],
    separated_labels: list[np.ndarray],
) -> np.ndarray:
    """Return the original positive-minus-negative mass-mean directions."""

    com_directions = []

    for layer in range(num_layers):
        for head in range(num_heads):
            usable_idxs = np.concatenate([train_set_idxs, val_set_idxs], axis=0)
            usable_head_wise_activations = np.concatenate(
                [
                    separated_head_wise_activations[i][:, layer, head, :]
                    for i in usable_idxs
                ],
                axis=0,
            )
            usable_labels = np.concatenate(
                [separated_labels[i] for i in usable_idxs], axis=0
            )
            true_mass_mean = np.mean(
                usable_head_wise_activations[usable_labels == 1], axis=0
            )
            false_mass_mean = np.mean(
                usable_head_wise_activations[usable_labels == 0], axis=0
            )
            com_directions.append(true_mass_mean - false_mass_mean)
    return np.array(com_directions)


def last_token_modulated_vector_add(
    head_output: Any,
    direction: Any,
    multiplier: float,
) -> Any:
    """Apply the original ITI last-token addition, generalized to batches."""

    updated = head_output.clone()
    updated[:, -1] = updated[:, -1] + direction.to(updated) * multiplier
    return updated
