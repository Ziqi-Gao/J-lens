"""Deterministic target construction in raw ``resid_post`` coordinates."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from numpy.typing import NDArray

from jlens_workspace.artifacts import atomic_write_json, sha256_file

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

CONCEPT_TEMPLATES = (
    "Tell me about {concept}.",
    "Explain {concept}.",
    "Describe the emotion of {concept}.",
    "What does {concept} feel like?",
    "Give a concise description of {concept}.",
    "Discuss a situation involving {concept}.",
    "Define {concept} in ordinary language.",
    "Write one sentence about {concept}.",
    "How might a person experiencing {concept} think?",
    "How might {concept} affect behavior?",
    "What can cause {concept}?",
    "Give an example of {concept}.",
)

# These descriptions are intentionally label-free.  Validation below rejects
# an accidental occurrence of the corresponding GoEmotions label.
LABEL_FREE_PARAPHRASES: dict[str, tuple[str, ...]] = {
    "goemotions:admiration": (
        "warm respect for another person's qualities or achievements",
        "being impressed by someone and holding them in high regard",
        "an appreciative response to excellence in another person",
        "looking up to someone because of what they are or have done",
    ),
    "goemotions:approval": (
        "a favorable judgment that an action or choice is acceptable",
        "responding positively to what someone decided or did",
        "the sense that a proposal meets one's standards",
        "endorsing a course of action as suitable or good",
    ),
    "goemotions:curiosity": (
        "a pull toward learning more about something not yet understood",
        "wanting to investigate an unanswered question",
        "attention energized by missing information",
        "an urge to explore how or why something works",
    ),
    "goemotions:disapproval": (
        "an unfavorable judgment that conduct or a choice is not acceptable",
        "responding negatively to what someone decided or did",
        "the sense that an action violates one's standards",
        "rejecting a course of action as unsuitable or wrong",
    ),
    "goemotions:gratitude": (
        "warm appreciation after receiving help or kindness",
        "recognizing a benefit and valuing the person who provided it",
        "a thankful response to support that was freely given",
        "appreciating that someone made a positive difference for you",
    ),
    "goemotions:love": (
        "deep affectionate attachment and care for another person",
        "a strong bond expressed through warmth and concern",
        "enduring closeness that makes another person's welfare important",
        "intense fondness joined with commitment and tenderness",
    ),
    "goemotions:optimism": (
        "expecting that future events are likely to turn out well",
        "a hopeful outlook about what will happen next",
        "confidence that difficulties can lead to a favorable result",
        "focusing on plausible positive outcomes in the future",
    ),
}

FOUR_UNRELATED_TOPIC_DESCRIPTIONS = (
    "the process by which green plants convert light into stored chemical energy",
    "the design of buildings and the organization of structural spaces",
    "the movement of tectonic plates beneath the surface of a planet",
    "a method for sorting a list by repeatedly dividing it into smaller parts",
)


class TargetConstructionError(ValueError):
    """A target cannot be constructed without ambiguity or leakage."""


def float64_sha256(values: object, *, ndim: int) -> str:
    """Hash the canonical raw float64 bytes of a finite array."""

    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if array.ndim != ndim or not np.isfinite(array).all():
        raise TargetConstructionError(
            f"registered float64 payload must be finite and {ndim}-dimensional"
        )
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def vector_sha256(vector: object) -> str:
    values = np.ascontiguousarray(np.asarray(vector, dtype=np.float64))
    if values.ndim != 1 or not np.isfinite(values).all():
        raise TargetConstructionError("target vectors must be finite and one-dimensional")
    return float64_sha256(values, ndim=1)


@dataclass(frozen=True)
class TargetRecord:
    """One immutable target and its scientific identity."""

    layer: int
    target_family: str
    target_subtype: str
    source_id: str
    vector: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)
    template_contrasts: FloatArray | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        values = np.asarray(self.vector, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or np.linalg.norm(values) == 0:
            raise TargetConstructionError("target vector must be finite, non-zero, and [D]")
        object.__setattr__(self, "vector", np.ascontiguousarray(values))
        contrasts = self.template_contrasts
        if contrasts is not None:
            matrix = np.ascontiguousarray(np.asarray(contrasts, dtype=np.float64))
            if (
                matrix.ndim != 2
                or matrix.shape[0] < 1
                or matrix.shape[1] != values.size
                or not np.isfinite(matrix).all()
            ):
                raise TargetConstructionError(
                    "per-template contrasts must be finite [T,D] and align with the target"
                )
            if not np.allclose(
                np.mean(matrix, axis=0, dtype=np.float64),
                values,
                rtol=0.0,
                atol=1e-12,
            ):
                raise TargetConstructionError(
                    "target vector is not the mean of its per-template contrasts"
                )
            object.__setattr__(self, "template_contrasts", matrix)

    @property
    def target_id(self) -> str:
        identity = {
            "layer": int(self.layer),
            "target_family": self.target_family,
            "target_subtype": self.target_subtype,
            "source_id": self.source_id,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:20]
        return f"{self.target_family}:{self.target_subtype}:{digest}"

    @property
    def vector_hash(self) -> str:
        return vector_sha256(self.vector)


def cosine_similarity(first: object, second: object) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if a.shape != b.shape or a.ndim != 1 or denominator == 0.0:
        raise TargetConstructionError("cosine inputs must be aligned non-zero vectors")
    return float(a @ b / denominator)


def class_mean_difference(
    activations: object,
    labels: object,
    *,
    train_indices: object,
    validation_indices: object,
) -> FloatArray:
    """Raw positive-minus-negative mean over balanced train+validation rows."""

    values = np.asarray(activations)
    y = np.asarray(labels)
    train = np.asarray(train_indices, dtype=np.int64)
    validation = np.asarray(validation_indices, dtype=np.int64)
    if values.ndim != 2 or y.shape != (values.shape[0],):
        raise TargetConstructionError("activations/labels must have shapes [N,D] and [N]")
    indices = np.concatenate((train, validation))
    if indices.size == 0 or len(set(indices.tolist())) != indices.size:
        raise TargetConstructionError("train+validation indices must be non-empty and disjoint")
    selected_labels = y[indices]
    if set(np.unique(selected_labels).tolist()) != {0, 1}:
        raise TargetConstructionError("train+validation rows must contain both classes")
    positive = np.mean(values[indices[selected_labels == 1]], axis=0, dtype=np.float64)
    negative = np.mean(values[indices[selected_labels == 0]], axis=0, dtype=np.float64)
    return np.asarray(positive - negative, dtype=np.float64)


def paired_template_contrasts(activations: object) -> FloatArray:
    """Apply within-template one-vs-other-concepts contrasts.

    ``activations`` has shape ``[C, T, D]``.  The output has the same shape and
    entry ``[c,t] = h[c,t] - mean(h[c'!=c,t])``.
    """

    values = np.asarray(activations, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 1:
        raise TargetConstructionError("concept-template activations require shape [C,T,D]")
    total = np.sum(values, axis=0, dtype=np.float64)
    return values - (total[None, :, :] - values) / (values.shape[0] - 1)


def concept_mean_vectors(activations: object) -> tuple[FloatArray, FloatArray]:
    contrasts = paired_template_contrasts(activations)
    return contrasts, np.mean(contrasts, axis=1, dtype=np.float64)


def template_bootstrap_stability(
    template_vectors: object,
    *,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Bootstrap templates only; no model forward pass occurs here."""

    values = np.asarray(template_vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or samples < 1000:
        raise TargetConstructionError("template bootstrap requires [T,D] and >=1000 samples")
    reference = np.mean(values, axis=0)
    rng = np.random.Generator(np.random.Philox(seed))
    cosines = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, values.shape[0], size=values.shape[0])
        estimate = np.mean(values[draw], axis=0)
        cosines[index] = cosine_similarity(reference, estimate)
    return {
        "samples": int(samples),
        "seed": int(seed),
        "cosine_median": float(np.median(cosines)),
        "cosine_q05": float(np.quantile(cosines, 0.05)),
        "cosine_q95": float(np.quantile(cosines, 0.95)),
    }


def find_unique_subsequence(sequence: object, subsequence: object) -> tuple[int, int]:
    """Return the unique half-open token span, failing on zero/multiple matches."""

    values = np.asarray(sequence, dtype=np.int64)
    needle = np.asarray(subsequence, dtype=np.int64)
    if values.ndim != 1 or needle.ndim != 1 or needle.size == 0:
        raise TargetConstructionError("token sequence and non-empty subsequence must be 1-D")
    matches = [
        start
        for start in range(0, values.size - needle.size + 1)
        if np.array_equal(values[start : start + needle.size], needle)
    ]
    if len(matches) != 1:
        raise TargetConstructionError(
            f"concept token subsequence has {len(matches)} matches; expected exactly one"
        )
    return matches[0], matches[0] + int(needle.size)


def unique_concept_token_span(
    sequence: object,
    candidate_subsequences: list[object],
) -> tuple[int, int]:
    """Resolve tokenization-context variants to one unambiguous concept span."""

    spans: set[tuple[int, int]] = set()
    for candidate in candidate_subsequences:
        try:
            spans.add(find_unique_subsequence(sequence, candidate))
        except TargetConstructionError as error:
            if "has 0 matches" not in str(error):
                raise
    if len(spans) != 1:
        raise TargetConstructionError(
            f"concept phrase has {len(spans)} distinct token spans; expected one"
        )
    return next(iter(spans))


def raw_activation_positions(
    token_ids: object,
    *,
    attention_mask: object | None = None,
    special_tokens_mask: object | None = None,
    quantiles: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00),
) -> IntArray:
    """Select quantile positions among non-padding, non-special tokens."""

    tokens = np.asarray(token_ids)
    if tokens.ndim != 1 or tokens.size == 0:
        raise TargetConstructionError("token_ids must be a non-empty 1-D sequence")
    allowed = np.ones(tokens.size, dtype=bool)
    if attention_mask is not None:
        attention = np.asarray(attention_mask)
        if attention.shape != tokens.shape:
            raise TargetConstructionError("attention_mask shape mismatch")
        allowed &= attention.astype(bool)
    if special_tokens_mask is not None:
        special = np.asarray(special_tokens_mask)
        if special.shape != tokens.shape:
            raise TargetConstructionError("special_tokens_mask shape mismatch")
        allowed &= ~special.astype(bool)
    eligible = np.flatnonzero(allowed)
    if eligible.size < 1:
        raise TargetConstructionError("no eligible non-special tokens")
    if any(not 0.0 < q <= 1.0 for q in quantiles):
        raise TargetConstructionError("quantiles must lie in (0,1]")
    offsets = [int(np.floor(float(q) * (eligible.size - 1))) for q in quantiles]
    return np.unique(eligible[np.asarray(offsets, dtype=np.int64)]).astype(np.int64)


def _atomic_save_npy(path: Path, values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.asarray(values), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_target_records(
    records: list[TargetRecord], *, artifact_root: str | Path
) -> dict[str, Any]:
    """Persist an exact or additive target set without mutating prior vectors."""

    if not records:
        raise TargetConstructionError("at least one target record is required")
    root = Path(artifact_root)
    target_root = root / "targets"
    index_path = target_root / "index.json"
    existing: dict[str, dict[str, Any]] = {}
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise TargetConstructionError("existing target index schema is unsupported")
        existing = {str(entry["target_id"]): entry for entry in payload["targets"]}
        if len(existing) != len(payload["targets"]):
            raise TargetConstructionError("existing target index has duplicate IDs")

    additions: dict[str, dict[str, Any]] = {}
    for record in sorted(records, key=lambda item: item.target_id):
        suffix = record.target_id.rsplit(":", 1)[-1]
        relative = (
            Path("targets")
            / record.target_family
            / f"layer_{record.layer:02d}"
            / f"{quote(record.target_subtype, safe='')}_{suffix}.npy"
        )
        destination = root / relative
        values = np.asarray(record.vector, dtype=np.float64)
        if destination.is_file():
            observed = np.load(destination, allow_pickle=False)
            if not np.array_equal(observed, values):
                raise TargetConstructionError(
                    f"existing target vector differs from registered target: {destination}"
                )
        else:
            _atomic_save_npy(destination, values)
        entry = {
            "target_id": record.target_id,
            "layer": int(record.layer),
            "target_family": record.target_family,
            "target_subtype": record.target_subtype,
            "source_id": record.source_id,
            "vector_path": relative.as_posix(),
            "file_sha256": sha256_file(destination),
            "raw_float64_sha256": record.vector_hash,
            "dimension": int(values.size),
            "norm": float(np.linalg.norm(values)),
            "metadata": record.metadata,
        }
        if record.template_contrasts is not None:
            template_relative = relative.with_name(
                f"{relative.stem}_per_template_contrasts.npy"
            )
            template_destination = root / template_relative
            template_values = np.asarray(record.template_contrasts, dtype=np.float64)
            if template_destination.is_file():
                observed_templates = np.load(template_destination, allow_pickle=False)
                if not np.array_equal(observed_templates, template_values):
                    raise TargetConstructionError(
                        "existing per-template contrasts differ from registered target: "
                        f"{template_destination}"
                    )
            else:
                _atomic_save_npy(template_destination, template_values)
            entry["per_template_contrasts"] = {
                "path": template_relative.as_posix(),
                "shape": [int(value) for value in template_values.shape],
                "dtype": "float64",
                "raw_float64_sha256": float64_sha256(template_values, ndim=2),
                "file_sha256": sha256_file(template_destination),
            }
        prior = existing.get(record.target_id)
        if prior is not None and json.dumps(prior, sort_keys=True) != json.dumps(
            entry, sort_keys=True
        ):
            raise TargetConstructionError(
                f"target ID collision with different identity: {record.target_id}"
            )
        additions[record.target_id] = entry

    combined = existing | additions
    families = sorted({entry["target_family"] for entry in combined.values()})
    layers = sorted({int(entry["layer"]) for entry in combined.values()})
    payload = {
        "schema_version": 1,
        "complete": True,
        "append_only": True,
        "target_count": len(combined),
        "target_families": families,
        "layers": layers,
        "targets": [combined[key] for key in sorted(combined)],
    }
    atomic_write_json(index_path, payload)
    return payload


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TargetConstructionError(f"cannot read JSON artifact: {path}") from error


def prepare_shared_statistical_targets(
    *,
    shared_root: str | Path,
    layers: list[int],
    concept_ids: list[str],
    probe_settings: Mapping[str, Any] | None = None,
) -> tuple[list[TargetRecord], dict[tuple[int, str], dict[str, Any]]]:
    """Reuse exact logistic probes and derive leakage-free mean differences.

    The returned metadata map carries each real probe's fixed ``C`` and split
    information for the later label-permutation conditional null.
    """

    root = Path(shared_root)
    probe_root = root / "selection" / "raptor_probes"
    probe_manifest = _read_json(probe_root / "manifest.json")
    row_manifest = _read_json(root / "selection" / "row_manifest.json")
    concept_payload = _read_json(root / "activations" / "concepts.json")
    labels = np.load(root / "activations" / "labels.npy", mmap_mode="r")
    row_manifest_path = root / "selection" / "row_manifest.json"
    row_manifest_sha256 = sha256_file(row_manifest_path)
    selection = probe_manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise TargetConstructionError("shared probe manifest lacks pipeline selection metadata")
    if not isinstance(probe_settings, Mapping):
        raise TargetConstructionError(
            "shared statistical targets require the registered probe config"
        )
    registered_pipeline = {
        "standardize": bool(probe_settings.get("standardize")),
        "class_weight": probe_settings.get("class_weight"),
        "solver": str(probe_settings.get("solver")),
        "penalty": str(probe_settings.get("penalty")),
        "max_iter": int(probe_settings.get("max_iter", 0)),
    }
    if registered_pipeline != {
        "standardize": True,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "penalty": "l2",
        "max_iter": 5000,
    }:
        raise TargetConstructionError(
            "k-diagnostic probe config does not match the preregistered pipeline"
        )
    manifest_pipeline = {
        "standardize": selection.get("standardize"),
        "solver": selection.get("solver"),
        "max_iter": selection.get("max_iter"),
    }
    if manifest_pipeline != {
        "standardize": registered_pipeline["standardize"],
        "solver": registered_pipeline["solver"],
        "max_iter": registered_pipeline["max_iter"],
    }:
        raise TargetConstructionError(
            "shared probe manifest does not match the registered pipeline"
        )
    probe_manifest_sha256 = sha256_file(probe_root / "manifest.json")
    columns = {
        str(item["concept_id"]): int(item["column"])
        for item in concept_payload["concepts"]
    }
    entries = {
        (int(item["layer"]), str(item["concept_id"])): item
        for item in probe_manifest["probes"]
    }
    records: list[TargetRecord] = []
    probe_metadata: dict[tuple[int, str], dict[str, Any]] = {}
    for layer in layers:
        activations = np.load(
            root / "activations" / f"layer_{layer:02d}.npy", mmap_mode="r"
        )
        for concept_id in concept_ids:
            key = (int(layer), concept_id)
            if key not in entries or concept_id not in columns:
                raise TargetConstructionError(f"shared artifacts lack probe {key}")
            entry = entries[key]
            vector_path = probe_root / entry["vector_file"]
            observed = sha256_file(vector_path)
            if observed != entry["vector_sha256"]:
                raise TargetConstructionError(f"shared probe hash mismatch: {vector_path}")
            vector = np.asarray(np.load(vector_path, allow_pickle=False), dtype=np.float64)
            metrics_path = probe_root / entry["metrics_file"]
            metrics = _read_json(metrics_path)
            if (
                metrics.get("final_fit_split") != "train+validation"
                or metrics.get("test_use") != "held_out_report_only"
                or metrics.get("probe_vector_sha256") != observed
                or metrics.get("row_manifest_sha256") != row_manifest_sha256
                or metrics.get("standardize") is not True
                or metrics.get("solver") != registered_pipeline["solver"]
                or metrics.get("penalty") != registered_pipeline["penalty"]
            ):
                raise TargetConstructionError(
                    f"shared probe metadata violates split/hash contract: {metrics_path}"
                )
            chosen_c = float(metrics["chosen_C"])
            validation_accuracy = float(metrics["validation_accuracy"])
            test_accuracy = float(metrics["test_accuracy"])
            if (
                not np.isfinite(chosen_c)
                or chosen_c <= 0.0
                or not 0.0 <= validation_accuracy <= 1.0
                or not 0.0 <= test_accuracy <= 1.0
            ):
                raise TargetConstructionError(
                    f"shared probe has invalid C/accuracy telemetry: {metrics_path}"
                )
            common = {
                "concept_id": concept_id,
                "source_vector_path": str(vector_path),
                "source_vector_sha256": observed,
                "source_metrics_path": str(metrics_path),
                "source_metrics_sha256": sha256_file(metrics_path),
                "source_probe_manifest_sha256": probe_manifest_sha256,
                "chosen_C": chosen_c,
                "validation_accuracy": validation_accuracy,
                "test_accuracy": test_accuracy,
                "row_manifest_sha256": row_manifest_sha256,
                "probe_pipeline": registered_pipeline,
                "probe_pipeline_provenance": {
                    "chosen_C": "per-probe metrics",
                    "standardize": "shared probe manifest and per-probe metrics",
                    "class_weight": "registered k-diagnostic config",
                    "solver": "shared probe manifest and per-probe metrics",
                    "penalty": "per-probe metrics and registered config",
                    "max_iter": "shared probe manifest and registered config",
                },
                "fit_split": "train+validation",
                "test_use": "held_out_report_only",
            }
            records.append(
                TargetRecord(
                    layer=layer,
                    target_family="logistic_probe",
                    target_subtype=concept_id,
                    source_id=concept_id,
                    vector=vector,
                    metadata=common,
                )
            )
            concept_rows = row_manifest["concepts"][concept_id]
            train = np.asarray(
                [int(row["activation_index"]) for row in concept_rows["train"]["rows"]],
                dtype=np.int64,
            )
            validation = np.asarray(
                [
                    int(row["activation_index"])
                    for row in concept_rows["validation"]["rows"]
                ],
                dtype=np.int64,
            )
            mean_vector = class_mean_difference(
                activations,
                labels[:, columns[concept_id]],
                train_indices=train,
                validation_indices=validation,
            )
            records.append(
                TargetRecord(
                    layer=layer,
                    target_family="class_mean_difference",
                    target_subtype=concept_id,
                    source_id=concept_id,
                    vector=mean_vector,
                    metadata={
                        "concept_id": concept_id,
                        "fit_split": "train+validation",
                        "test_rows_accessed": False,
                        "train_rows": int(train.size),
                        "validation_rows": int(validation.size),
                        "cosine_with_logistic_probe": cosine_similarity(
                            mean_vector, vector
                        ),
                    },
                )
            )
            probe_metadata[key] = {
                **common,
                "train_indices": train,
                "validation_indices": validation,
                "concept_column": columns[concept_id],
            }
    return records, probe_metadata


def _forward_kwargs(model: Any) -> dict[str, Any]:
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


def _capture_token_positions(
    *,
    model: Any,
    tokenizer: Any,
    formatted_texts: list[str],
    positions: list[list[int]],
    layers: list[int],
    batch_size: int,
    max_length: int,
) -> dict[int, list[FloatArray]]:
    """Capture selected token positions while hooks remain resid_post-only."""

    import torch

    from jlens_workspace.modeling import (
        hidden_from_block_output,
        model_input_device,
        register_resid_post_hook,
    )

    if len(formatted_texts) != len(positions):
        raise TargetConstructionError("texts and token-position rows must align")
    captured: dict[int, Any] = {}
    output: dict[int, list[FloatArray]] = {layer: [] for layer in layers}

    def make_hook(layer: int) -> Any:
        def hook(_module: Any, _inputs: Any, value: Any) -> None:
            hidden = hidden_from_block_output(value)
            captured[layer] = hidden.detach().to(dtype=torch.float32, device="cpu")

        return hook

    handles = [register_resid_post_hook(model, layer, make_hook(layer)) for layer in layers]
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        with torch.inference_mode():
            for start in range(0, len(formatted_texts), batch_size):
                texts = formatted_texts[start : start + batch_size]
                requested = positions[start : start + batch_size]
                encoded = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                lengths = encoded["attention_mask"].sum(dim=1).tolist()
                for row_positions, length in zip(requested, lengths, strict=True):
                    if not row_positions or min(row_positions) < 0 or max(row_positions) >= length:
                        raise TargetConstructionError(
                            "requested token position was truncated or padded"
                        )
                inputs = {
                    key: value.to(model_input_device(model))
                    for key, value in encoded.items()
                }
                captured.clear()
                model(**inputs, **_forward_kwargs(model))
                if set(captured) != set(layers):
                    raise TargetConstructionError("resid_post hooks missed registered layers")
                for layer in layers:
                    hidden = captured[layer]
                    for row, row_positions in enumerate(requested):
                        output[layer].append(
                            np.asarray(hidden[row, row_positions], dtype=np.float64)
                        )
    finally:
        tokenizer.padding_side = original_side
        for handle in handles:
            handle.remove()
    return output


def prepare_seven_emotion_label_contrast_targets(
    *,
    model: Any,
    tokenizer: Any,
    layers: list[int],
    concept_ids: list[str],
    batch_size: int = 8,
    max_length: int = 256,
    bootstrap_samples: int = 1000,
) -> list[TargetRecord]:
    """Build the named seven-emotion estimand with two leakage controls.

    ``label_explicit`` reproduces the seven-label one-vs-six contrast.
    ``label_free_definition`` replaces each label with four descriptions that
    are validated not to contain it. ``four_unrelated_topics_control`` contrasts
    those same descriptions with exactly four non-emotion topics. This is not an Anthropic-style
    concept mean and is never reported under that name.
    """

    if set(concept_ids) != set(LABEL_FREE_PARAPHRASES):
        raise TargetConstructionError("label-free paraphrases must exactly cover concepts")
    if not getattr(tokenizer, "chat_template", None):
        raise TargetConstructionError("native chat template is required")
    formatted: list[str] = []
    positions: list[list[int]] = []
    specifications: list[dict[str, Any]] = []

    def register(
        *,
        concept_id: str,
        arm: str,
        template_index: int,
        role: str,
        prompt: str,
        subject: str,
    ) -> None:
        label = concept_id.split(":", 1)[-1].replace("_", " ")
        if arm != "label_explicit" and label.casefold() in prompt.casefold():
            raise TargetConstructionError(
                f"label-free prompt leaked target label {label!r}"
            )
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(
            rendered,
            add_special_tokens=True,
            truncation=False,
            return_special_tokens_mask=True,
        )
        sequence = np.asarray(encoded["input_ids"], dtype=np.int64)
        candidates = [
            tokenizer.encode(value, add_special_tokens=False)
            for value in (subject, f" {subject}")
        ]
        start, stop = unique_concept_token_span(sequence, candidates)
        boundary = int(sequence.size - 1)
        manifest = {
            "concept_id": concept_id,
            "arm": arm,
            "role": role,
            "template_index": int(template_index),
            "prompt": prompt,
            "rendered_text": rendered,
            "rendered_text_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "token_ids_sha256": hashlib.sha256(
                np.ascontiguousarray(sequence).tobytes(order="C")
            ).hexdigest(),
            "token_count": int(sequence.size),
            "subject_token_span": [int(start), int(stop)],
            "assistant_boundary_token_index": boundary,
        }
        formatted.append(rendered)
        positions.append([boundary, int(stop - 1)])
        specifications.append(manifest)

    for concept_id in concept_ids:
        label = concept_id.split(":", 1)[-1].replace("_", " ")
        for template_index, template in enumerate(CONCEPT_TEMPLATES):
            register(
                concept_id=concept_id,
                arm="label_explicit",
                template_index=template_index,
                role="target",
                prompt=template.format(concept=label),
                subject=label,
            )
        for template_index, definition in enumerate(LABEL_FREE_PARAPHRASES[concept_id]):
            prompt = f"Consider this experience: {definition}. Explain it briefly."
            register(
                concept_id=concept_id,
                arm="label_free_definition",
                template_index=template_index,
                role="target",
                prompt=prompt,
                subject=definition,
            )
            register(
                concept_id=concept_id,
                arm="four_unrelated_topics_control",
                template_index=template_index,
                role="target",
                prompt=prompt,
                subject=definition,
            )
    for template_index, subject in enumerate(FOUR_UNRELATED_TOPIC_DESCRIPTIONS):
        register(
            concept_id="control:four_unrelated_topics",
            arm="four_unrelated_topics_control",
            template_index=template_index,
            role="baseline",
            prompt=f"Consider this topic: {subject}. Explain it briefly.",
            subject=subject,
        )
    captured = _capture_token_positions(
        model=model,
        tokenizer=tokenizer,
        formatted_texts=formatted,
        positions=positions,
        layers=layers,
        batch_size=batch_size,
        max_length=max_length,
    )
    records: list[TargetRecord] = []
    for layer in layers:
        values = np.stack(captured[layer])
        by_key = {
            (
                str(spec["arm"]),
                str(spec["role"]),
                str(spec["concept_id"]),
                int(spec["template_index"]),
            ): values[index]
            for index, spec in enumerate(specifications)
        }
        for arm in (
            "label_explicit",
            "label_free_definition",
            "four_unrelated_topics_control",
        ):
            template_count = (
                len(CONCEPT_TEMPLATES)
                if arm == "label_explicit"
                else len(FOUR_UNRELATED_TOPIC_DESCRIPTIONS)
            )
            for position_index in range(2):
                capture_position = (
                    "assistant_boundary"
                    if position_index == 0
                    else (
                        "concept_token_end"
                        if arm == "label_explicit"
                        else "definition_span_end"
                    )
                )
                for concept_index, concept_id in enumerate(concept_ids):
                    target_values = np.stack(
                        [
                            by_key[(arm, "target", concept_id, template_index)][
                                position_index
                            ]
                            for template_index in range(template_count)
                        ]
                    )
                    if arm == "four_unrelated_topics_control":
                        baseline_values = np.stack(
                            [
                                by_key[
                                    (
                                        arm,
                                        "baseline",
                                        "control:four_unrelated_topics",
                                        template_index,
                                    )
                                ][position_index]
                                for template_index in range(template_count)
                            ]
                        )
                    else:
                        other_concepts = [
                            other for other in concept_ids if other != concept_id
                        ]
                        baseline_values = np.mean(
                            np.stack(
                                [
                                    np.stack(
                                        [
                                            by_key[
                                                (
                                                    arm,
                                                    "target",
                                                    other,
                                                    template_index,
                                                )
                                            ][position_index]
                                            for template_index in range(template_count)
                                        ]
                                    )
                                    for other in other_concepts
                                ]
                            ),
                            axis=0,
                        )
                    contrasts = target_values - baseline_values
                    mean = np.mean(contrasts, axis=0, dtype=np.float64)
                    relevant = [
                        spec
                        for spec in specifications
                        if spec["arm"] == arm
                        and (
                            spec["concept_id"] == concept_id
                            or (
                                arm == "four_unrelated_topics_control"
                                and spec["role"] == "baseline"
                            )
                            or (
                                arm != "four_unrelated_topics_control"
                                and spec["concept_id"] in concept_ids
                            )
                        )
                    ]
                    manifest_hash = hashlib.sha256(
                        json.dumps(
                            relevant, sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")
                    ).hexdigest()
                    stability = template_bootstrap_stability(
                        contrasts,
                        samples=bootstrap_samples,
                        seed=42 + layer * 101 + position_index + concept_index * 17,
                    )
                    records.append(
                        TargetRecord(
                            layer=layer,
                            target_family="seven_emotion_label_contrast",
                            target_subtype=f"{arm}:{capture_position}:{concept_id}",
                            source_id=concept_id,
                            vector=mean,
                            template_contrasts=contrasts,
                            metadata={
                                "concept_id": concept_id,
                                "estimand_name": "seven_emotion_label_contrast",
                                "contrast_arm": arm,
                                "capture_position": capture_position,
                                "coordinate": "resid_post",
                                "template_count": template_count,
                                "paired_template_contrast": True,
                                "baseline_scope": (
                                    "exactly_four_unrelated_topics"
                                    if arm == "four_unrelated_topics_control"
                                    else "other_six_registered_emotions"
                                ),
                                "chat_template": "native",
                                "add_generation_prompt": True,
                                "enable_thinking": False,
                                "prompt_manifest": relevant,
                                "prompt_manifest_sha256": manifest_hash,
                                "bootstrap": stability,
                            },
                        )
                    )
    return records


def _label_blind_test_texts(path: Path) -> list[dict[str, str]]:
    """Read only text/group identity fields; label fields are never accessed."""

    unique: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise TargetConstructionError(
                    f"{path}:{line_number}: invalid JSON"
                ) from error
            text = row.get("text")
            group_id = row.get("group_id")
            if not isinstance(text, str) or not text or not isinstance(group_id, str):
                raise TargetConstructionError(
                    f"{path}:{line_number}: text/group_id are required"
                )
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            candidate = {"text": text, "group_id": group_id, "text_sha256": digest}
            prior = unique.get(digest)
            if prior is not None and prior != candidate:
                raise TargetConstructionError("text hash collision or conflicting group")
            unique[digest] = candidate
    return [unique[key] for key in sorted(unique)]


def prepare_raw_activation_targets(
    *,
    model: Any,
    tokenizer: Any,
    source_test_jsonl: str | Path,
    expected_source_sha256: str,
    layers: list[int],
    targets_per_layer: int,
    selection_rank_start: int = 0,
    batch_size: int = 8,
    max_length: int = 256,
) -> list[TargetRecord]:
    """Select test texts by hash/length only, then capture four raw positions."""

    source = Path(source_test_jsonl)
    observed_source = sha256_file(source)
    if observed_source != expected_source_sha256:
        raise TargetConstructionError(
            f"source test JSONL hash mismatch: expected {expected_source_sha256}, "
            f"observed {observed_source}"
        )
    if targets_per_layer % 4 or selection_rank_start % 4:
        raise TargetConstructionError("raw target counts/rank start must be divisible by four")
    if not 0 <= selection_rank_start < targets_per_layer:
        raise TargetConstructionError("selection_rank_start must precede targets_per_layer")
    needed_texts = targets_per_layer // 4
    selected: list[dict[str, Any]] = []
    for row in _label_blind_test_texts(source):
        encoded = tokenizer(
            row["text"],
            add_special_tokens=True,
            truncation=False,
            return_special_tokens_mask=True,
        )
        token_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        positions = raw_activation_positions(
            token_ids,
            special_tokens_mask=encoded["special_tokens_mask"],
        )
        eligible_count = int(
            np.sum(~np.asarray(encoded["special_tokens_mask"], dtype=bool))
        )
        if eligible_count < 32 or positions.size != 4 or token_ids.size > max_length:
            continue
        selected.append(row | {"token_ids": token_ids, "positions": positions})
        if len(selected) == needed_texts:
            break
    if len(selected) != needed_texts:
        raise TargetConstructionError(
            f"only {len(selected)} label-blind texts meet the token-length contract; "
            f"need {needed_texts}"
        )
    source_rank_start = selection_rank_start // 4
    capture_selected = selected[source_rank_start:]
    captured = _capture_token_positions(
        model=model,
        tokenizer=tokenizer,
        formatted_texts=[row["text"] for row in capture_selected],
        positions=[row["positions"].tolist() for row in capture_selected],
        layers=layers,
        batch_size=batch_size,
        max_length=max_length,
    )
    records: list[TargetRecord] = []
    quantiles = (0.25, 0.50, 0.75, 1.00)
    for layer in layers:
        for source_rank, (row, vectors) in enumerate(
            zip(capture_selected, captured[layer], strict=True),
            start=source_rank_start,
        ):
            for index, (position, quantile) in enumerate(
                zip(row["positions"], quantiles, strict=True)
            ):
                selection_rank = source_rank * len(quantiles) + index
                token_id = int(row["token_ids"][position])
                records.append(
                    TargetRecord(
                        layer=layer,
                        target_family="raw_activation",
                        target_subtype=(
                            f"source={row['text_sha256']}:quantile={quantile:g}"
                        ),
                        source_id=row["text_sha256"],
                        vector=vectors[index],
                        metadata={
                            "source_test_jsonl_sha256": observed_source,
                            "source_text_sha256": row["text_sha256"],
                            "group_id": row["group_id"],
                            "selection_uses_concept_labels": False,
                            "selection_rank": int(selection_rank),
                            "stage_membership": (
                                ["pilot", "raw_metric_full"]
                                if selection_rank < 32
                                else ["raw_metric_full"]
                            ),
                            "quantile": quantile,
                            "token_position": int(position),
                            "token_id": token_id,
                            "decoded_token": tokenizer.decode([token_id]),
                            "non_special_token_count_minimum": 32,
                        },
                    )
                )
    return records
