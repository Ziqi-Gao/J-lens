"""Qwen-compatible adapters around the pinned honest_llama ITI algorithm.

The scientific coordinate in this module is the concatenated full-attention
head output immediately before ``o_proj``. For Qwen3.5-4B each captured tensor
has shape ``[N, H=16, D_head=256]``. The original ITI routines operate on the
same pre-output-projection coordinate; only module resolution and the
GoEmotions row layout are workspace-specific.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from numpy.typing import NDArray

from jlens_workspace._vendor.honest_llama_core import (
    get_com_directions,
    get_interventions_dict,
    last_token_modulated_vector_add,
    train_probes,
)
from jlens_workspace.activations import (
    _capture_forward_kwargs,
    _example_text,
    _last_token_indices,
    _shared_source_examples,
)
from jlens_workspace.artifacts import (
    RunManifest,
    atomic_write_json,
    sha256_file,
    stable_hash,
)
from jlens_workspace.concept_intervention.shared_protocol import (
    load_balanced_indices,
    load_selected_layers,
)
from jlens_workspace.modeling import model_input_device, transformer_blocks

HONEST_LLAMA_REPO = "https://github.com/likenneth/honest_llama"
HONEST_LLAMA_COMMIT = "2c6b2179be7b5aa8f0a171688cf9e01b812ca327"
ITI_METHOD = "honest_llama_mass_mean_qwen_full_attention_v1"
ITI_METHOD_UNVERSIONED = "honest_llama_mass_mean_qwen_full_attention"

FloatArray = NDArray[np.float64]


class ITIError(ValueError):
    """Raised when an ITI input violates the registered coordinate contract."""


@dataclass(frozen=True)
class AttentionHeadSpec:
    """One pre-``o_proj`` head coordinate for a full-attention layer."""

    layer: int
    num_heads: int
    head_dim: int
    projection: Any


@dataclass(frozen=True)
class ITIHeadShift:
    """A unit head direction and its original ITI projection standard deviation."""

    rank: int
    layer: int
    head: int
    validation_accuracy: float
    direction: FloatArray
    projection_std: float


@dataclass
class ITIInterventionState:
    """Telemetry collected without changing the original ITI vector addition."""

    events: list[dict[str, Any]]


def full_attention_head_specs(
    model: Any, layers: Sequence[int]
) -> tuple[AttentionHeadSpec, ...]:
    """Resolve Qwen/LLaMA-style full-attention ``o_proj`` input coordinates."""

    blocks = transformer_blocks(model)
    specs: list[AttentionHeadSpec] = []
    for layer in layers:
        if not 0 <= int(layer) < len(blocks):
            raise ITIError(f"attention layer {layer} outside [0, {len(blocks)})")
        block = blocks[int(layer)]
        attention = getattr(block, "self_attn", None)
        projection = getattr(attention, "o_proj", None)
        head_dim = getattr(attention, "head_dim", None)
        in_features = getattr(projection, "in_features", None)
        if (
            attention is None
            or projection is None
            or head_dim is None
            or in_features is None
        ):
            raise ITIError(
                f"layer {layer} is not a supported full-attention block with o_proj"
            )
        if int(head_dim) <= 0 or int(in_features) % int(head_dim):
            raise ITIError(f"layer {layer} has incompatible o_proj/head dimensions")
        specs.append(
            AttentionHeadSpec(
                layer=int(layer),
                num_heads=int(in_features) // int(head_dim),
                head_dim=int(head_dim),
                projection=projection,
            )
        )
    if not specs:
        raise ITIError("at least one full-attention layer is required")
    layouts = {(spec.num_heads, spec.head_dim) for spec in specs}
    if len(layouts) != 1:
        raise ITIError(f"ITI requires a uniform head layout, observed {sorted(layouts)}")
    return tuple(specs)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def capture_iti_head_activations(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[Any],
    layers: Sequence[int],
    output_dir: str | Path,
    batch_size: int = 8,
    max_length: int = 256,
    add_special_tokens: bool = True,
    expected_num_heads: int | None = None,
    expected_head_dim: int | None = None,
    source_residual_activations: str | Path | None = None,
    manifest: RunManifest | None = None,
    overwrite: bool = False,
) -> Path:
    """Capture final-token pre-``o_proj`` head outputs as ``[N,H,D_head]`` arrays."""

    import torch
    from numpy.lib.format import open_memmap

    if not examples:
        raise ITIError("at least one concept example is required")
    capture_examples, labels, rows, concepts = _shared_source_examples(examples)
    if np.any(labels == -1):
        raise ITIError("ITI requires a complete shared-source concept label matrix")
    specs = full_attention_head_specs(model, sorted(set(int(layer) for layer in layers)))
    if expected_num_heads is not None and specs[0].num_heads != expected_num_heads:
        raise ITIError(
            f"observed {specs[0].num_heads} attention heads, expected {expected_num_heads}"
        )
    if expected_head_dim is not None and specs[0].head_dim != expected_head_dim:
        raise ITIError(
            f"observed head_dim={specs[0].head_dim}, expected {expected_head_dim}"
        )
    destination = Path(output_dir)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"ITI activation artifact already exists: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    source_identity: dict[str, Any] | None = None
    if source_residual_activations is not None:
        source = Path(source_residual_activations)
        source_labels = np.load(source / "labels.npy", allow_pickle=False)
        if not np.array_equal(source_labels, labels):
            raise ITIError(
                "ITI labels differ from the registered residual activation labels"
            )
        source_metadata = json.loads(
            (source / "metadata.json").read_text(encoding="utf-8")
        )
        example_hash = stable_hash(json.dumps(row, sort_keys=True) for row in rows)
        if source_metadata.get("example_hash") != example_hash:
            raise ITIError("ITI example identity differs from residual activations")
        source_manifest = source_metadata.get("manifest")
        current_manifest = None if manifest is None else manifest.__dict__
        if isinstance(source_manifest, Mapping) and isinstance(current_manifest, Mapping):
            identity_fields = (
                "model_id",
                "model_revision",
                "tokenizer_id",
                "tokenizer_revision",
                "dataset_source",
                "dataset_revision",
                "dataset_hash",
            )
            mismatches = [
                field
                for field in identity_fields
                if source_manifest.get(field) != current_manifest.get(field)
            ]
            if mismatches:
                raise ITIError(
                    "ITI model/data identity differs from residual activations: "
                    + ", ".join(mismatches)
                )
        source_identity = {
            "path": str(source),
            "metadata_sha256": sha256_file(source / "metadata.json"),
            "labels_sha256": sha256_file(source / "labels.npy"),
            "example_hash": example_hash,
            "coordinate": source_metadata.get("coordinate"),
        }

    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "right"
    current_last_indices: Any = None
    captured: dict[int, np.ndarray] = {}
    memmaps: dict[int, np.memmap] = {}

    def make_hook(spec: AttentionHeadSpec) -> Any:
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if not inputs:
                raise ITIError(f"layer {spec.layer} o_proj hook received no input")
            head_output = inputs[0]
            if head_output.ndim != 3 or head_output.shape[-1] != (
                spec.num_heads * spec.head_dim
            ):
                raise ITIError(
                    f"layer {spec.layer} expected [B,S,{spec.num_heads * spec.head_dim}], "
                    f"got {tuple(head_output.shape)}"
                )
            indices = current_last_indices.to(device=head_output.device)
            batch_rows = torch.arange(head_output.shape[0], device=head_output.device)
            final = head_output[batch_rows, indices].reshape(
                head_output.shape[0], spec.num_heads, spec.head_dim
            )
            captured[spec.layer] = final.detach().float().cpu().numpy()

        return hook

    handles = [
        spec.projection.register_forward_pre_hook(make_hook(spec)) for spec in specs
    ]
    try:
        with torch.inference_mode():
            for start in range(0, len(capture_examples), batch_size):
                batch = capture_examples[start : start + batch_size]
                encoded = tokenizer(
                    [_example_text(example) for example in batch],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=add_special_tokens,
                    return_tensors="pt",
                )
                current_last_indices = _last_token_indices(encoded["attention_mask"])
                model_inputs = {
                    key: value.to(model_input_device(model))
                    for key, value in encoded.items()
                    if key != "offset_mapping"
                }
                captured.clear()
                model(**model_inputs, **_capture_forward_kwargs(model))
                missing = {spec.layer for spec in specs}.difference(captured)
                if missing:
                    raise ITIError(f"ITI hooks did not capture layers {sorted(missing)}")
                for spec in specs:
                    values = captured[spec.layer]
                    if spec.layer not in memmaps:
                        memmaps[spec.layer] = open_memmap(
                            destination / f"layer_{spec.layer:02d}.npy",
                            mode="w+",
                            dtype=np.float32,
                            shape=(
                                len(capture_examples),
                                spec.num_heads,
                                spec.head_dim,
                            ),
                        )
                    memmaps[spec.layer][start : start + len(batch)] = values
    finally:
        tokenizer.padding_side = original_padding_side
        for handle in handles:
            handle.remove()
        for array in memmaps.values():
            array.flush()

    np.save(destination / "labels.npy", labels, allow_pickle=False)
    _write_rows(destination / "rows.jsonl", rows)
    atomic_write_json(
        destination / "concepts.json", {"schema_version": 1, "concepts": concepts}
    )
    metadata = {
        "schema_version": 1,
        "method": ITI_METHOD,
        "coordinate": "attention_head_output_pre_o_proj",
        "representation": "last_non_padding_token",
        "layers": [spec.layer for spec in specs],
        "num_heads": specs[0].num_heads,
        "head_dim": specs[0].head_dim,
        "n_examples": len(capture_examples),
        "label_shape": list(labels.shape),
        "example_hash": stable_hash(json.dumps(row, sort_keys=True) for row in rows),
        "concept_hash": stable_hash(
            json.dumps(row, sort_keys=True) for row in concepts
        ),
        "source_residual_activations": source_identity,
        "upstream": {
            "repository": HONEST_LLAMA_REPO,
            "commit": HONEST_LLAMA_COMMIT,
            "license": "MIT",
        },
        "compatibility_notes": [
            "TruthfulQA grouped-answer rows are replaced by split-safe GoEmotions source rows.",
            "The upstream 128-dimensional LLaMA head coordinate is generalized to the "
            f"observed {specs[0].head_dim}-dimensional Qwen head coordinate.",
            "Only Qwen full-attention o_proj inputs are included; linear-attention blocks are excluded.",
        ],
        "manifest": None if manifest is None else manifest.__dict__,
    }
    atomic_write_json(destination / "metadata.json", metadata)
    return destination


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _balanced_split_indices(
    labels: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    seed: int,
) -> np.ndarray:
    candidates = np.asarray(
        [index for index, row in enumerate(rows) if row.get("split") == split],
        dtype=np.int64,
    )
    positive = candidates[labels[candidates] == 1]
    negative = candidates[labels[candidates] == 0]
    if not len(positive) or not len(negative):
        raise ITIError(f"split {split!r} must contain both labels")
    count = min(len(positive), len(negative))
    rng = np.random.Generator(np.random.Philox(seed))
    positive = rng.choice(positive, size=count, replace=False)
    negative = rng.choice(negative, size=count, replace=False)
    combined = np.concatenate([positive, negative])
    rng.shuffle(combined)
    return np.asarray(combined, dtype=np.int64)


def _gather_activations(
    root: Path, layers: Sequence[int], indices: np.ndarray
) -> np.ndarray:
    gathered = []
    for layer in layers:
        source = np.load(root / f"layer_{layer:02d}.npy", mmap_mode="r")
        gathered.append(np.asarray(source[indices], dtype=np.float32))
    return np.stack(gathered, axis=1)


def _flatten_interventions(
    interventions: Mapping[str, Sequence[tuple[int, np.ndarray, float]]],
    *,
    actual_layers: Sequence[int],
    accuracies: np.ndarray,
    num_heads: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rank = 0
    for internal_layer, actual_layer in enumerate(actual_layers):
        key = f"model.layers.{internal_layer}.self_attn.head_out"
        for head, direction, projection_std in interventions.get(key, []):
            rows.append(
                {
                    "rank": rank,
                    "layer": int(actual_layer),
                    "head": int(head),
                    "validation_accuracy": float(accuracies[internal_layer, head]),
                    "direction": np.asarray(direction, dtype=np.float64),
                    "projection_std": float(projection_std),
                }
            )
            rank += 1
    return rows


def _save_shift_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    np.savez(
        path,
        rank=np.asarray([row["rank"] for row in rows], dtype=np.int16),
        layer=np.asarray([row["layer"] for row in rows], dtype=np.int16),
        head=np.asarray([row["head"] for row in rows], dtype=np.int16),
        validation_accuracy=np.asarray(
            [row["validation_accuracy"] for row in rows], dtype=np.float64
        ),
        direction=np.stack([np.asarray(row["direction"]) for row in rows]),
        projection_std=np.asarray(
            [row["projection_std"] for row in rows], dtype=np.float64
        ),
    )


def layer_matched_head_order(
    flat_accuracies: np.ndarray,
    *,
    num_layers: int,
    num_heads: int,
) -> np.ndarray:
    """Place one best head per layer first, then preserve native global order."""

    values = np.asarray(flat_accuracies, dtype=np.float64)
    if values.shape != (num_layers * num_heads,):
        raise ITIError("flat accuracies do not match layer/head layout")
    global_order = np.argsort(-values, kind="stable")
    mandatory: list[int] = []
    for layer in range(num_layers):
        start = layer * num_heads
        mandatory.append(start + int(np.argmax(values[start : start + num_heads])))
    mandatory.sort(key=lambda index: (-float(values[index]), int(index)))
    selected = set(mandatory)
    return np.asarray(
        [*mandatory, *[int(value) for value in global_order if int(value) not in selected]],
        dtype=np.int64,
    )


def layer_matched_random_head_order(
    native_random_order: np.ndarray,
    *,
    num_layers: int,
    num_heads: int,
) -> np.ndarray:
    """Adapt an original ITI random permutation to cover every layer first."""

    order = np.asarray(native_random_order, dtype=np.int64)
    total_heads = num_layers * num_heads
    if (
        order.shape != (total_heads,)
        or sorted(order.tolist()) != list(range(total_heads))
    ):
        raise ITIError("random head order must be a complete head permutation")
    mandatory: list[int] = []
    covered_layers: set[int] = set()
    for value in order:
        layer = int(value) // num_heads
        if layer not in covered_layers:
            covered_layers.add(layer)
            mandatory.append(int(value))
        if len(mandatory) == num_layers:
            break
    selected = set(mandatory)
    return np.asarray(
        [*mandatory, *[int(value) for value in order if int(value) not in selected]],
        dtype=np.int64,
    )


def fit_iti_concept_directions(
    *,
    activation_dir: str | Path,
    output_dir: str | Path,
    concept_id: str,
    max_top_k: int,
    random_seeds: Sequence[int],
    seed: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit original ITI head probes and mass-mean directions for one concept."""

    root = Path(activation_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("method") != ITI_METHOD:
        raise ITIError("unsupported ITI activation artifact")
    layers = tuple(int(value) for value in metadata["layers"])
    num_heads = int(metadata["num_heads"])
    head_dim = int(metadata["head_dim"])
    total_heads = len(layers) * num_heads
    if not 0 < max_top_k <= total_heads:
        raise ITIError(f"max_top_k must lie in [1, {total_heads}]")
    concepts_payload = json.loads((root / "concepts.json").read_text(encoding="utf-8"))
    concepts = list(concepts_payload["concepts"])
    concept_by_id = {str(row["concept_id"]): row for row in concepts}
    if concept_id not in concept_by_id:
        raise ITIError(f"concept absent from ITI labels: {concept_id}")
    column = int(concept_by_id[concept_id]["column"])
    labels_matrix = np.load(root / "labels.npy", allow_pickle=False)
    labels = np.asarray(labels_matrix[:, column], dtype=np.int8)
    rows = _read_rows(root / "rows.jsonl")
    concept_seed = seed + column * 1009
    train_source = _balanced_split_indices(
        labels, rows, split="train", seed=concept_seed
    )
    validation_source = _balanced_split_indices(
        labels, rows, split="validation", seed=concept_seed + 1
    )
    source_indices = np.concatenate([train_source, validation_source])
    activations = _gather_activations(root, layers, source_indices)
    development_labels = labels[source_indices]
    separated_activations = [
        activations[index : index + 1] for index in range(len(activations))
    ]
    separated_labels = [
        development_labels[index : index + 1] for index in range(len(activations))
    ]
    train_idxs = np.arange(len(train_source), dtype=np.int64)
    val_idxs = np.arange(len(train_source), len(source_indices), dtype=np.int64)

    probes, flat_accuracies = train_probes(
        seed,
        train_idxs,
        val_idxs,
        separated_activations,
        separated_labels,
        num_layers=len(layers),
        num_heads=num_heads,
    )
    accuracies = flat_accuracies.reshape(len(layers), num_heads)
    ranked_flat = np.argsort(flat_accuracies)[::-1][:max_top_k]
    top_heads = [
        (int(value) // num_heads, int(value) % num_heads) for value in ranked_flat
    ]
    com_directions = get_com_directions(
        len(layers),
        num_heads,
        train_idxs,
        val_idxs,
        separated_activations,
        separated_labels,
    )
    mass_mean = get_interventions_dict(
        top_heads,
        probes,
        activations,
        num_heads,
        True,
        False,
        com_directions,
        head_dim=head_dim,
    )
    probe_weight = get_interventions_dict(
        top_heads,
        probes,
        activations,
        num_heads,
        False,
        False,
        None,
        head_dim=head_dim,
    )
    mass_rows = _flatten_interventions(
        mass_mean,
        actual_layers=layers,
        accuracies=accuracies,
        num_heads=num_heads,
    )
    probe_rows = _flatten_interventions(
        probe_weight,
        actual_layers=layers,
        accuracies=accuracies,
        num_heads=num_heads,
    )
    rank_order = {
        (int(value) // num_heads, int(value) % num_heads): rank
        for rank, value in enumerate(ranked_flat)
    }
    for collection in (mass_rows, probe_rows):
        collection.sort(
            key=lambda row: rank_order[
                (layers.index(int(row["layer"])), int(row["head"]))
            ]
        )
        for rank, row in enumerate(collection):
            row["rank"] = rank

    destination = Path(output_dir) / quote(concept_id, safe="")
    if destination.exists() and not overwrite:
        metrics_path = destination / "metrics.json"
        if metrics_path.is_file():
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        raise FileExistsError(f"incomplete ITI direction artifact exists: {destination}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    np.savez(
        destination / "development_indices.npz",
        train_source_indices=train_source,
        validation_source_indices=validation_source,
    )
    _save_shift_rows(destination / "mass_mean.npz", mass_rows)
    _save_shift_rows(destination / "probe_weight.npz", probe_rows)

    random_files: dict[str, str] = {}
    original_random_state = np.random.get_state()
    try:
        for random_seed in random_seeds:
            np.random.seed(int(random_seed))
            random_flat = np.random.choice(total_heads, total_heads, replace=False)[
                :max_top_k
            ]
            random_heads = [
                (int(value) // num_heads, int(value) % num_heads)
                for value in random_flat
            ]
            random_interventions = get_interventions_dict(
                random_heads,
                probes,
                activations,
                num_heads,
                False,
                True,
                None,
                head_dim=head_dim,
            )
            random_rows = _flatten_interventions(
                random_interventions,
                actual_layers=layers,
                accuracies=accuracies,
                num_heads=num_heads,
            )
            random_rank = {
                (int(value) // num_heads, int(value) % num_heads): rank
                for rank, value in enumerate(random_flat)
            }
            random_rows.sort(
                key=lambda row: random_rank[
                    (layers.index(int(row["layer"])), int(row["head"]))
                ]
            )
            for rank, row in enumerate(random_rows):
                row["rank"] = rank
            filename = f"random_{int(random_seed)}.npz"
            _save_shift_rows(destination / filename, random_rows)
            random_files[str(int(random_seed))] = filename
    finally:
        np.random.set_state(original_random_state)

    metrics = {
        "schema_version": 1,
        "method": ITI_METHOD,
        "concept_id": concept_id,
        "coordinate": "attention_head_output_pre_o_proj",
        "layers": list(layers),
        "num_heads": num_heads,
        "head_dim": head_dim,
        "max_top_k": max_top_k,
        "head_selection_metric": "validation_accuracy_on_seeded_balanced_split",
        "balanced_split_seed": concept_seed,
        "train_examples": len(train_source),
        "validation_examples": len(validation_source),
        "test_examples_used": 0,
        "files": {
            "mass_mean": "mass_mean.npz",
            "probe_weight": "probe_weight.npz",
            "random": random_files,
            "development_indices": "development_indices.npz",
        },
        "development_indices_sha256": sha256_file(
            destination / "development_indices.npz"
        ),
        "activation_metadata_sha256": sha256_file(root / "metadata.json"),
        "labels_sha256": sha256_file(root / "labels.npy"),
        "upstream": {
            "repository": HONEST_LLAMA_REPO,
            "commit": HONEST_LLAMA_COMMIT,
            "license": "MIT",
            "core_functions": [
                "train_probes",
                "get_top_heads/head ranking",
                "get_com_directions",
                "get_interventions_dict",
                "last-token modulated vector add",
            ],
        },
    }
    atomic_write_json(destination / "metrics.json", metrics)
    return metrics


def fit_shared_iti_concept_directions(
    *,
    activation_dir: str | Path,
    output_dir: str | Path,
    concept_id: str,
    selected_layers_path: str | Path,
    row_manifest_path: str | Path,
    max_top_k: int,
    random_seeds: Sequence[int],
    seed: int,
    variants: Sequence[str] = ("native", "layer_matched"),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit ITI on the shared rows/layers and persist both head-order variants."""

    root = Path(activation_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("method") not in {ITI_METHOD, ITI_METHOD_UNVERSIONED}:
        raise ITIError("unsupported ITI activation artifact")
    captured_layers = tuple(int(value) for value in metadata["layers"])
    layers = load_selected_layers(selected_layers_path, concept_id)
    missing = sorted(set(layers) - set(captured_layers))
    if missing:
        raise ITIError(f"selected layers were not captured: {missing}")
    num_heads = int(metadata["num_heads"])
    head_dim = int(metadata["head_dim"])
    total_heads = len(layers) * num_heads
    if not 0 < max_top_k <= total_heads:
        raise ITIError(f"max_top_k must lie in [1, {total_heads}]")
    selected_variants = tuple(str(value) for value in variants)
    if (
        not selected_variants
        or len(set(selected_variants)) != len(selected_variants)
        or set(selected_variants) - {"native", "layer_matched"}
    ):
        raise ITIError("ITI variants must be unique native/layer_matched values")

    concepts_payload = json.loads((root / "concepts.json").read_text(encoding="utf-8"))
    concept_by_id = {
        str(row["concept_id"]): row for row in concepts_payload["concepts"]
    }
    if concept_id not in concept_by_id:
        raise ITIError(f"concept absent from ITI labels: {concept_id}")
    column = int(concept_by_id[concept_id]["column"])
    labels_matrix = np.load(root / "labels.npy", allow_pickle=False)
    labels = np.asarray(labels_matrix[:, column], dtype=np.int8)
    train_source = load_balanced_indices(row_manifest_path, concept_id, "train")
    validation_source = load_balanced_indices(
        row_manifest_path, concept_id, "validation"
    )
    source_indices = np.concatenate((train_source, validation_source))
    activations = _gather_activations(root, layers, source_indices)
    development_labels = labels[source_indices]
    separated_activations = [
        activations[index : index + 1] for index in range(len(activations))
    ]
    separated_labels = [
        development_labels[index : index + 1]
        for index in range(len(development_labels))
    ]
    train_idxs = np.arange(len(train_source), dtype=np.int64)
    val_idxs = np.arange(len(train_source), len(source_indices), dtype=np.int64)
    probes, flat_accuracies = train_probes(
        seed,
        train_idxs,
        val_idxs,
        separated_activations,
        separated_labels,
        num_layers=len(layers),
        num_heads=num_heads,
    )
    accuracies = flat_accuracies.reshape(len(layers), num_heads)
    com_directions = get_com_directions(
        len(layers),
        num_heads,
        train_idxs,
        val_idxs,
        separated_activations,
        separated_labels,
    )
    orders = {
        "native": np.argsort(-flat_accuracies, kind="stable"),
        "layer_matched": layer_matched_head_order(
            flat_accuracies,
            num_layers=len(layers),
            num_heads=num_heads,
        ),
    }

    destination = Path(output_dir) / quote(concept_id, safe="")
    if destination.exists() and not overwrite:
        metrics_path = destination / "metrics.json"
        if metrics_path.is_file():
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        raise FileExistsError(f"incomplete ITI direction artifact exists: {destination}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    np.savez(
        destination / "development_indices.npz",
        train_source_indices=train_source,
        validation_source_indices=validation_source,
    )

    variant_files: dict[str, dict[str, str]] = {}
    for variant in selected_variants:
        order = orders[variant][:max_top_k]
        heads = [
            (int(value) // num_heads, int(value) % num_heads) for value in order
        ]
        rank_order = {
            (int(value) // num_heads, int(value) % num_heads): rank
            for rank, value in enumerate(order)
        }
        files: dict[str, str] = {}
        for mode, use_mass_mean in (("mass_mean", True), ("probe_weight", False)):
            fitted = get_interventions_dict(
                heads,
                probes,
                activations,
                num_heads,
                use_mass_mean,
                False,
                com_directions if use_mass_mean else None,
                head_dim=head_dim,
            )
            shift_rows = _flatten_interventions(
                fitted,
                actual_layers=layers,
                accuracies=accuracies,
                num_heads=num_heads,
            )
            shift_rows.sort(
                key=lambda row: rank_order[
                    (layers.index(int(row["layer"])), int(row["head"]))
                ]
            )
            for rank, row in enumerate(shift_rows):
                row["rank"] = rank
            filename = f"{mode}_{variant}.npz"
            _save_shift_rows(destination / filename, shift_rows)
            files[mode] = filename
        variant_files[variant] = files

    random_files: dict[str, dict[str, str]] = {
        variant: {} for variant in selected_variants
    }
    original_random_state = np.random.get_state()
    try:
        for random_seed in random_seeds:
            for variant in selected_variants:
                # Reset for each variant. The native branch therefore retains
                # the author's exact seeded random permutation/direction path.
                np.random.seed(int(random_seed))
                native_order = np.random.choice(
                    total_heads, total_heads, replace=False
                )
                order = (
                    native_order
                    if variant == "native"
                    else layer_matched_random_head_order(
                        native_order,
                        num_layers=len(layers),
                        num_heads=num_heads,
                    )
                )
                selected_order = order[:max_top_k]
                random_heads = [
                    (int(value) // num_heads, int(value) % num_heads)
                    for value in selected_order
                ]
                random_fitted = get_interventions_dict(
                    random_heads,
                    probes,
                    activations,
                    num_heads,
                    False,
                    True,
                    None,
                    head_dim=head_dim,
                )
                random_rows = _flatten_interventions(
                    random_fitted,
                    actual_layers=layers,
                    accuracies=accuracies,
                    num_heads=num_heads,
                )
                random_rank = {
                    (int(value) // num_heads, int(value) % num_heads): rank
                    for rank, value in enumerate(selected_order)
                }
                random_rows.sort(
                    key=lambda row: random_rank[
                        (layers.index(int(row["layer"])), int(row["head"]))
                    ]
                )
                for rank, row in enumerate(random_rows):
                    row["rank"] = rank
                filename = f"random_{variant}_{int(random_seed)}.npz"
                _save_shift_rows(destination / filename, random_rows)
                random_files[variant][str(int(random_seed))] = filename
    finally:
        np.random.set_state(original_random_state)

    metrics = {
        "schema_version": 1,
        "method": ITI_METHOD_UNVERSIONED,
        "concept_id": concept_id,
        "coordinate": "attention_head_output_pre_o_proj",
        "captured_layers": list(captured_layers),
        "layers": list(layers),
        "num_heads": num_heads,
        "head_dim": head_dim,
        "max_top_k": max_top_k,
        "variants": list(selected_variants),
        "head_selection_metric": "validation_accuracy_on_shared_balanced_rows",
        "train_examples": len(train_source),
        "validation_examples": len(validation_source),
        "test_examples_used": 0,
        "files": {
            "variants": variant_files,
            "random": random_files,
            "development_indices": "development_indices.npz",
        },
        "development_indices_sha256": sha256_file(
            destination / "development_indices.npz"
        ),
        "activation_metadata_sha256": sha256_file(root / "metadata.json"),
        "labels_sha256": sha256_file(root / "labels.npy"),
        "selected_layers_path": str(selected_layers_path),
        "selected_layers_sha256": sha256_file(selected_layers_path),
        "row_manifest_path": str(row_manifest_path),
        "row_manifest_sha256": sha256_file(row_manifest_path),
        "upstream": {
            "repository": HONEST_LLAMA_REPO,
            "commit": HONEST_LLAMA_COMMIT,
            "license": "MIT",
            "native_variant": "original global validation-accuracy head ranking",
            "layer_matched_adaptation": (
                "best head from each shared layer first, then original global order"
            ),
            "native_random_control": (
                "original seeded random head permutation and author random "
                "direction construction"
            ),
            "layer_matched_random_control": (
                "same seeded native permutation, reordered so its first "
                "encountered head from every shared layer precedes the "
                "remaining native order; author random direction construction"
            ),
        },
    }
    atomic_write_json(destination / "metrics.json", metrics)
    return metrics


def load_iti_head_shifts(
    direction_dir: str | Path,
    *,
    concept_id: str,
    mode: str,
    top_k: int,
    random_seed: int | None = None,
    variant: str | None = None,
) -> list[ITIHeadShift]:
    concept = Path(direction_dir) / quote(concept_id, safe="")
    metrics = json.loads((concept / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("method") not in {ITI_METHOD, ITI_METHOD_UNVERSIONED}:
        raise ITIError("unsupported ITI direction artifact")
    if mode == "random":
        if random_seed is None:
            raise ITIError("random ITI shifts require a seed")
        random_files = metrics["files"]["random"]
        if (
            metrics.get("method") == ITI_METHOD_UNVERSIONED
            and variant is not None
            and isinstance(random_files.get(variant), dict)
        ):
            filename = random_files[variant].get(str(int(random_seed)))
        else:
            filename = random_files.get(str(int(random_seed)))
        if filename is None:
            raise ITIError(
                f"random seed/variant absent from ITI artifact: "
                f"{random_seed}/{variant}"
            )
    elif mode in {"mass_mean", "probe_weight"}:
        if variant is None:
            filename = metrics["files"][mode]
        else:
            try:
                filename = metrics["files"]["variants"][variant][mode]
            except KeyError as error:
                raise ITIError(
                    f"ITI variant/mode absent from artifact: {variant}/{mode}"
                ) from error
    else:
        raise ITIError(f"unknown ITI direction mode: {mode}")
    payload = np.load(concept / filename, allow_pickle=False)
    if not 0 < top_k <= len(payload["rank"]):
        raise ITIError(f"top_k={top_k} outside fitted range")
    output = []
    for index in range(top_k):
        output.append(
            ITIHeadShift(
                rank=int(payload["rank"][index]),
                layer=int(payload["layer"][index]),
                head=int(payload["head"][index]),
                validation_accuracy=float(payload["validation_accuracy"][index]),
                direction=np.asarray(payload["direction"][index], dtype=np.float64),
                projection_std=float(payload["projection_std"][index]),
            )
        )
    return output


def layer_shift_vectors(
    shifts: Sequence[ITIHeadShift], *, num_heads: int, head_dim: int
) -> dict[int, FloatArray]:
    """Combine selected ``sigma * theta`` head shifts by layer as in honest_llama."""

    output: dict[int, FloatArray] = {}
    for shift in shifts:
        if not 0 <= shift.head < num_heads or shift.direction.shape != (head_dim,):
            raise ITIError("ITI head shift does not match configured head layout")
        vector = output.setdefault(
            shift.layer, np.zeros(num_heads * head_dim, dtype=np.float64)
        )
        start = shift.head * head_dim
        vector[start : start + head_dim] = shift.projection_std * shift.direction
    return output


@contextmanager
def iti_intervention_session(
    model: Any,
    *,
    shifts: Sequence[ITIHeadShift],
    multiplier: float,
    num_heads: int,
    head_dim: int,
) -> Iterator[ITIInterventionState]:
    """Apply original ITI addition to the last token on every model forward call."""

    vectors = layer_shift_vectors(shifts, num_heads=num_heads, head_dim=head_dim)
    specs = {
        spec.layer: spec
        for spec in full_attention_head_specs(model, sorted(vectors))
    }
    handles = []
    state = ITIInterventionState(events=[])
    forward_calls = {layer: 0 for layer in vectors}
    for layer, vector in vectors.items():
        spec = specs[layer]

        def hook(
            _module: Any,
            inputs: tuple[Any, ...],
            *,
            layer_id: int = layer,
            direction: FloatArray = vector,
            expected: int = spec.num_heads * spec.head_dim,
        ) -> tuple[Any, ...]:
            if not inputs or inputs[0].ndim != 3 or inputs[0].shape[-1] != expected:
                raise ITIError("ITI o_proj input coordinate mismatch")
            import torch

            tensor_direction = torch.as_tensor(
                direction, device=inputs[0].device, dtype=inputs[0].dtype
            )
            changed = last_token_modulated_vector_add(
                inputs[0], tensor_direction, multiplier
            )
            forward_call = forward_calls[layer_id]
            forward_calls[layer_id] += 1
            state.events.append(
                {
                    "layer": int(layer_id),
                    "forward_call": forward_call,
                    "generation_step": forward_call,
                    "batch_size": int(inputs[0].shape[0]),
                    "sequence_length": int(inputs[0].shape[1]),
                    "active_positions": int(inputs[0].shape[0]),
                    "multiplier": float(multiplier),
                    "injected_norm": abs(float(multiplier))
                    * float(np.linalg.norm(direction)),
                }
            )
            return (changed, *inputs[1:])

        handles.append(spec.projection.register_forward_pre_hook(hook))
    try:
        yield state
    finally:
        for handle in handles:
            handle.remove()
