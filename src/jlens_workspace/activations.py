"""Leakage-aware residual activation capture to a simple on-disk artifact."""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jlens_workspace.artifacts import RunManifest, atomic_write_json, stable_hash
from jlens_workspace.modeling import (
    hidden_from_block_output,
    model_input_device,
    register_resid_post_hook,
)


def _field(example: Any, name: str, default: Any = None) -> Any:
    if isinstance(example, Mapping):
        return example.get(name, default)
    return getattr(example, name, default)


def _example_text(example: Any) -> str:
    """Return exactly the labeled text whose final token is probed."""

    text = _field(example, "text")
    if text:
        return str(text)
    raise ValueError("activation example needs non-empty text")


def _last_token_indices(attention_mask: Any) -> Any:
    """Locate the final non-padding token in every padded batch row."""

    import torch

    attention = torch.as_tensor(attention_mask).bool()
    if attention.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if not attention.any(dim=1).all():
        raise ValueError("every activation example must contain at least one token")
    positions = torch.arange(attention.shape[1], device=attention.device)
    return torch.where(attention, positions.unsqueeze(0), -1).amax(dim=1)


def _serializable_example(example: Any, index: int) -> dict[str, Any]:
    fields = (
        "concept_id",
        "concept_name",
        "label",
        "split",
        "group_id",
        "source",
    )
    result = {"row": index}
    for name in fields:
        value = _field(example, name)
        if value is not None:
            result[name] = value
    return result


def _shared_source_examples(
    examples: Sequence[Any],
) -> tuple[list[Any], np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """Collapse concept-expanded rows to one row per global source group.

    Returns representative examples, an ``[N, C]`` int8 label matrix, source-row
    metadata, and a column-ordered concept table. ``-1`` marks a concept label
    unavailable for a source. The formal full GoEmotions artifact is required
    downstream to contain no ``-1`` values; supporting them here keeps the
    capture format usable for balanced controls as well.
    """

    concepts: dict[str, tuple[str, str]] = {}
    groups: dict[str, dict[str, Any]] = {}
    for example in examples:
        concept_id = str(_field(example, "concept_id") or "").strip()
        concept_name = str(_field(example, "concept_name") or "").strip()
        definition = str(_field(example, "definition") or "").strip()
        group_id = str(_field(example, "group_id") or "").strip()
        split = str(_field(example, "split") or "").strip()
        text = _example_text(example)
        source = str(_field(example, "source") or "").strip()
        license_name = str(_field(example, "license") or "").strip()
        label = _field(example, "label")
        if not concept_id or not concept_name or not group_id or not split or not source:
            raise ValueError(
                "shared-source capture requires concept_id, concept_name, group_id, "
                "split, and source"
            )
        if isinstance(label, bool) or not isinstance(label, (int, np.integer)):
            raise ValueError("shared-source labels must be integer 0 or 1")
        if int(label) not in (0, 1):
            raise ValueError("shared-source labels must be integer 0 or 1")
        metadata = (concept_name, definition)
        prior_metadata = concepts.setdefault(concept_id, metadata)
        if prior_metadata != metadata:
            raise ValueError(f"concept {concept_id!r} has inconsistent metadata")

        signature = (text, split, source, license_name)
        group = groups.setdefault(
            group_id,
            {
                "signature": signature,
                "example": example,
                "labels": {},
            },
        )
        if group["signature"] != signature:
            raise ValueError(
                f"source group {group_id!r} has inconsistent text/split/source/license"
            )
        if concept_id in group["labels"]:
            raise ValueError(
                f"source group {group_id!r} contains more than one row for concept "
                f"{concept_id!r}"
            )
        group["labels"][concept_id] = int(label)

    concept_ids = sorted(concepts)
    concept_table = [
        {
            "column": column,
            "concept_id": concept_id,
            "concept_name": concepts[concept_id][0],
            "definition": concepts[concept_id][1],
        }
        for column, concept_id in enumerate(concept_ids)
    ]
    split_order = {"train": 0, "validation": 1, "test": 2}
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            split_order.get(str(_field(item[1]["example"], "split")), 99),
            item[0],
        ),
    )
    labels = np.full((len(ordered_groups), len(concept_ids)), -1, dtype=np.int8)
    representatives: list[Any] = []
    rows: list[dict[str, Any]] = []
    for row_index, (group_id, group) in enumerate(ordered_groups):
        example = group["example"]
        representatives.append(example)
        for column, concept_id in enumerate(concept_ids):
            if concept_id in group["labels"]:
                labels[row_index, column] = group["labels"][concept_id]
        rows.append(
            {
                "row": row_index,
                "split": str(_field(example, "split")),
                "group_id": group_id,
                "source": str(_field(example, "source")),
                "license": str(_field(example, "license")),
                "text_sha256": hashlib.sha256(
                    _example_text(example).encode("utf-8")
                ).hexdigest(),
            }
        )
    return representatives, labels, rows, concept_table


def _capture_forward_kwargs(model: Any) -> dict[str, Any]:
    """Avoid materializing full-sequence vocabulary logits when supported."""

    kwargs: dict[str, Any] = {"use_cache": False}
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return kwargs
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = 1
    elif "num_logits_to_keep" in parameters:
        kwargs["num_logits_to_keep"] = 1
    return kwargs


def capture_residual_activations(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[Any],
    layers: Sequence[int],
    output_dir: str | Path,
    batch_size: int = 8,
    max_length: int = 512,
    add_special_tokens: bool = True,
    manifest: RunManifest | None = None,
    share_examples_by_group: bool = False,
    require_complete_concept_matrix: bool = False,
    overwrite: bool = False,
) -> Path:
    """Capture the final non-padding token's ``[D]`` block output per example.

    The output contains memory-mappable ``layer_XX.npy`` arrays plus labels and
    row metadata. With ``share_examples_by_group=True``, concept-expanded rows
    are collapsed before model inference and ``labels.npy`` has shape ``[N,C]``.
    Hooks are removed even if a forward pass fails.
    """

    import torch
    from numpy.lib.format import open_memmap

    if not examples:
        raise ValueError("at least one example is required")
    unique_layers = sorted(set(int(layer) for layer in layers))
    if not unique_layers:
        raise ValueError("at least one layer is required")
    if share_examples_by_group:
        capture_examples, labels, row_metadata, concept_table = _shared_source_examples(
            examples
        )
        if require_complete_concept_matrix and np.any(labels == -1):
            raise ValueError(
                "shared-source concept matrix is incomplete but complete labels are required"
            )
    else:
        if require_complete_concept_matrix:
            raise ValueError(
                "require_complete_concept_matrix requires share_examples_by_group=True"
            )
        capture_examples = list(examples)
        labels = np.asarray(
            [int(_field(example, "label")) for example in examples], dtype=np.int8
        )
        row_metadata = [
            _serializable_example(example, index)
            for index, example in enumerate(examples)
        ]
        concept_table = []

    destination = Path(output_dir)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"activation artifact already exists: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "right"
    current_last_indices: Any = None
    captured: dict[int, np.ndarray] = {}
    memmaps: dict[int, np.memmap] = {}

    def make_hook(layer: int) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = hidden_from_block_output(output)
            if hidden.ndim != 3:
                raise ValueError(f"expected [batch, sequence, d_model], got {hidden.shape}")
            indices = current_last_indices.to(device=hidden.device)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            final_token = hidden[rows, indices]
            captured[layer] = (
                final_token.detach().to(dtype=torch.float32, device="cpu").numpy()
            )

        return hook

    handles = [register_resid_post_hook(model, layer, make_hook(layer)) for layer in unique_layers]
    try:
        with torch.inference_mode():
            for start in range(0, len(capture_examples), batch_size):
                batch = capture_examples[start : start + batch_size]
                texts = [_example_text(example) for example in batch]
                tokenize_kwargs = dict(
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=add_special_tokens,
                    return_tensors="pt",
                )
                encoded = tokenizer(texts, **tokenize_kwargs)
                current_last_indices = _last_token_indices(encoded["attention_mask"])
                model_inputs = {
                    key: value.to(model_input_device(model))
                    for key, value in encoded.items()
                    if key != "offset_mapping"
                }
                captured.clear()
                model(**model_inputs, **_capture_forward_kwargs(model))
                missing = set(unique_layers).difference(captured)
                if missing:
                    raise RuntimeError(f"hooks did not capture layers: {sorted(missing)}")
                for layer in unique_layers:
                    values = captured[layer]
                    if layer not in memmaps:
                        memmaps[layer] = open_memmap(
                            destination / f"layer_{layer:02d}.npy",
                            mode="w+",
                            dtype=np.float32,
                            shape=(len(capture_examples), values.shape[1]),
                        )
                    memmaps[layer][start : start + len(batch)] = values
    finally:
        tokenizer.padding_side = original_padding_side
        for handle in handles:
            handle.remove()
        for array in memmaps.values():
            array.flush()

    np.save(destination / "labels.npy", labels, allow_pickle=False)
    with (destination / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in row_metadata:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if share_examples_by_group:
        atomic_write_json(
            destination / "concepts.json",
            {"schema_version": 1, "concepts": concept_table},
        )
    metadata = {
        "schema_version": 2 if share_examples_by_group else 1,
        "coordinate": "resid_post",
        "representation": "last_non_padding_token",
        "example_layout": (
            "shared_source_by_group" if share_examples_by_group else "concept_expanded"
        ),
        "add_special_tokens": add_special_tokens,
        "layers": unique_layers,
        "n_examples": len(capture_examples),
        "n_input_task_rows": len(examples),
        "n_concepts": len(concept_table) if share_examples_by_group else None,
        "label_shape": list(labels.shape),
        "missing_label": -1 if share_examples_by_group else None,
        "missing_label_count": int(np.sum(labels == -1)),
        "example_hash": stable_hash(json.dumps(row, sort_keys=True) for row in row_metadata),
        "concept_hash": (
            stable_hash(json.dumps(row, sort_keys=True) for row in concept_table)
            if share_examples_by_group
            else None
        ),
        "manifest": None if manifest is None else manifest.__dict__,
    }
    atomic_write_json(destination / "metadata.json", metadata)
    return destination


def load_activation_layer(path: str | Path, layer: int, mmap_mode: str = "r") -> np.ndarray:
    return np.load(Path(path) / f"layer_{layer:02d}.npy", mmap_mode=mmap_mode, allow_pickle=False)
