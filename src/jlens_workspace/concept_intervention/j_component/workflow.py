"""Causal comparison of full concept, sparse-J, non-J, and random directions."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from numpy.typing import NDArray

from jlens_workspace.artifacts import atomic_write_json, sha256_file
from jlens_workspace.concept_intervention.evaluation import (
    PromptRecord,
    atomic_write_jsonl,
    batched,
    candidate_token_ids,
    load_prompt_bank,
)
from jlens_workspace.concept_intervention.j_component.intervention import (
    generate_with_intervention,
    intervention_session,
)
from jlens_workspace.modeling import model_input_device

FloatArray = NDArray[np.float64]


class ConceptInterventionError(ValueError):
    """Raised when intervention inputs violate the registered protocol."""


@dataclass(frozen=True)
class DirectionRecord:
    condition_id: str
    condition: str
    direction: FloatArray
    random_seed: int | None = None


def _median_residual_norm(
    path: str | Path, *, layer: int, chunk_size: int = 4096
) -> float:
    matrix = np.load(Path(path) / f"layer_{layer:02d}.npy", mmap_mode="r")
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ConceptInterventionError("activation layer must have shape [N, D]")
    norms: list[np.ndarray] = []
    for start in range(0, matrix.shape[0], chunk_size):
        chunk = np.asarray(matrix[start : start + chunk_size], dtype=np.float32)
        norms.append(np.linalg.norm(chunk, axis=1))
    value = float(np.median(np.concatenate(norms)))
    if not math.isfinite(value) or value <= 0:
        raise ConceptInterventionError("median residual norm must be finite and positive")
    return value


def _probe_vector_path(probes: Path, layer: int, concept_id: str) -> Path:
    return (
        probes
        / f"layer_{layer:02d}"
        / f"concept_{quote(concept_id, safe='')}"
        / "probe_vector.npy"
    )


def _occupancy_metrics_path(
    occupancy: Path,
    *,
    layer: int,
    concept_id: str,
    convention: str,
) -> Path:
    return (
        occupancy
        / "occupancy"
        / convention
        / "positive_cosine"
        / f"layer_{layer:02d}"
        / quote(concept_id, safe="")
        / "pos"
        / "primary"
        / "metrics.json"
    )


def _matched_random_numpy(direction: FloatArray, seed: int) -> FloatArray:
    unit = direction / np.linalg.norm(direction)
    rng = np.random.Generator(np.random.Philox(seed))
    random = rng.normal(size=direction.size)
    random = random - float(random @ unit) * unit
    norm = float(np.linalg.norm(random))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ConceptInterventionError("failed to construct random direction")
    return np.asarray(random / norm, dtype=np.float64)


def load_registered_directions(
    *,
    occupancy_dir: str | Path,
    probes_dir: str | Path,
    layer: int,
    concept_id: str,
    convention: str,
    random_seeds: Sequence[int],
    expected_occupancy_index_sha256: str,
    expected_occupancy_git_commit: str,
) -> tuple[list[DirectionRecord], dict[str, Any]]:
    """Load and identity-check the registered full/J/non-J/random directions."""

    occupancy = Path(occupancy_dir)
    probes = Path(probes_dir)
    index_path = occupancy / "occupancy" / "index.json"
    if sha256_file(index_path) != expected_occupancy_index_sha256:
        raise ConceptInterventionError("occupancy index SHA-256 mismatch")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not index.get("complete") or index.get("observed_combinations") != 840:
        raise ConceptInterventionError("source occupancy index is incomplete")
    audit_path = occupancy / "provenance_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("source_git_commit") != expected_occupancy_git_commit:
        raise ConceptInterventionError("source occupancy Git commit audit mismatch")

    metrics_path = _occupancy_metrics_path(
        occupancy,
        layer=layer,
        concept_id=concept_id,
        convention=convention,
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    required_identity = {
        "method": "concept_occupancy_method_v2",
        "solver_method": "nonnegative_gradient_pursuit_v2",
        "layer": layer,
        "concept_id": concept_id,
        "sign": "+",
        "replicate_id": "primary",
        "convention": convention,
        "selection_mode": "positive_cosine",
    }
    for field, expected in required_identity.items():
        if metrics.get(field) != expected:
            raise ConceptInterventionError(
                f"{metrics_path}: {field}={metrics.get(field)!r}, expected "
                f"{expected!r}"
            )
    selected_k = int(metrics["primary_occupancy"]["k_selected_before_crossing"])
    if selected_k <= 0:
        raise ConceptInterventionError(
            f"{concept_id}: registered primary occupancy selected K={selected_k}"
        )
    combo = metrics_path.parent
    full_path = _probe_vector_path(probes, layer, concept_id)
    if sha256_file(full_path) != metrics["probe_vector_sha256"]:
        raise ConceptInterventionError(f"probe SHA-256 mismatch: {full_path}")
    full = np.asarray(np.load(full_path, allow_pickle=False), dtype=np.float64)
    j_path = combo / str(metrics["primary_occupancy"]["w_j_file"])
    non_j_path = combo / str(metrics["primary_occupancy"]["w_nonj_file"])
    j_direction = np.asarray(np.load(j_path, allow_pickle=False), dtype=np.float64)
    non_j_direction = np.asarray(
        np.load(non_j_path, allow_pickle=False), dtype=np.float64
    )
    for name, vector in (
        ("full", full),
        ("j", j_direction),
        ("non_j", non_j_direction),
    ):
        if (
            vector.shape != full.shape
            or not np.isfinite(vector).all()
            or float(np.linalg.norm(vector)) <= 0
        ):
            raise ConceptInterventionError(f"{name} direction is invalid")
    reconstruction_error = float(np.max(np.abs(full - j_direction - non_j_direction)))
    if reconstruction_error > 1e-12:
        raise ConceptInterventionError(
            f"full != J + non-J; max error {reconstruction_error}"
        )

    directions = [
        DirectionRecord("full", "full", full),
        DirectionRecord("j", "j", j_direction),
        DirectionRecord("non_j", "non_j", non_j_direction),
    ]
    directions.extend(
        DirectionRecord(
            f"random_{seed}",
            "random",
            _matched_random_numpy(full, seed),
            random_seed=int(seed),
        )
        for seed in random_seeds
    )
    provenance = {
        "occupancy_index": str(index_path),
        "occupancy_index_sha256": sha256_file(index_path),
        "occupancy_metrics": str(metrics_path),
        "occupancy_metrics_sha256": sha256_file(metrics_path),
        "source_git_commit": expected_occupancy_git_commit,
        "probe_vector": str(full_path),
        "probe_vector_sha256": sha256_file(full_path),
        "selected_k": selected_k,
        "selected_tokens": metrics["decoded_tokens"][:selected_k],
        "explained_fraction_at_k": float(
            1.0
            - np.load(combo / "errors.npy", allow_pickle=False)[selected_k]
        ),
        "reconstruction_max_abs_error": reconstruction_error,
        "coordinate": "resid_post",
    }
    return directions, provenance


def _score_condition(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[PromptRecord],
    candidate_token_ids: Mapping[str, int],
    target_concept_id: str,
    layer: int,
    direction: FloatArray,
    strength: float,
    residual_norm: float,
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch

    device = model_input_device(model)
    concept_ids = tuple(candidate_token_ids)
    token_ids = torch.as_tensor(
        [candidate_token_ids[value] for value in concept_ids],
        dtype=torch.long,
        device=device,
    )
    rows: list[dict[str, Any]] = []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for prompt_batch in batched(list(prompts), batch_size):
            encoded = tokenizer(
                [prompt.formatted_text for prompt in prompt_batch],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode(), intervention_session(
                model,
                layer,
                direction,
                strength,
                kind="addition",
                position="last_prompt",
                residual_norm=residual_norm,
            ):
                output = model(**encoded, use_cache=False, logits_to_keep=1)
            logits = output.logits[:, -1, :].float()
            log_probs = torch.log_softmax(logits, dim=-1)[:, token_ids].cpu().numpy()
            candidate_probs = torch.softmax(logits[:, token_ids], dim=-1).cpu().numpy()
            for row_index, prompt in enumerate(prompt_batch):
                log_map = {
                    concept_id: float(log_probs[row_index, column])
                    for column, concept_id in enumerate(concept_ids)
                }
                prob_map = {
                    concept_id: float(candidate_probs[row_index, column])
                    for column, concept_id in enumerate(concept_ids)
                }
                target_log_prob = log_map[target_concept_id]
                off_target = [
                    value
                    for concept_id, value in log_map.items()
                    if concept_id != target_concept_id
                ]
                rank = 1 + sum(value > target_log_prob for value in off_target)
                rows.append(
                    {
                        "prompt_id": prompt.prompt_id,
                        "target_concept_id": target_concept_id,
                        "strength": float(strength),
                        "candidate_log_probabilities": log_map,
                        "candidate_probabilities_normalized": prob_map,
                        "target_log_probability": target_log_prob,
                        "target_candidate_probability": prob_map[target_concept_id],
                        "target_margin": target_log_prob - float(np.mean(off_target)),
                        "target_rank": rank,
                    }
                )
    finally:
        tokenizer.padding_side = original_padding_side
    return rows


def _curve_summary(score_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in score_rows:
        grouped[(str(row["condition_id"]), float(row["strength"]))].append(row)
    curves: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (condition_id, strength), values in sorted(grouped.items()):
        curves[condition_id].append(
            {
                "strength": strength,
                "n_prompts": len(values),
                "mean_target_log_probability": float(
                    np.mean([float(row["target_log_probability"]) for row in values])
                ),
                "mean_target_candidate_probability": float(
                    np.mean(
                        [float(row["target_candidate_probability"]) for row in values]
                    )
                ),
                "mean_target_margin": float(
                    np.mean([float(row["target_margin"]) for row in values])
                ),
                "target_rank1_rate": float(
                    np.mean([int(row["target_rank"]) == 1 for row in values])
                ),
            }
        )
    condition_summaries: dict[str, dict[str, Any]] = {}
    for condition_id, points in curves.items():
        strengths = np.asarray([point["strength"] for point in points])
        margins = np.asarray([point["mean_target_margin"] for point in points])
        slope = float(np.polyfit(strengths, margins, deg=1)[0])
        condition_summaries[condition_id] = {
            "dose_response": points,
            "target_margin_slope": slope,
            "signed_endpoint_effect": float(margins[-1] - margins[0]),
        }
    full_curve = np.asarray(
        [
            point["mean_target_margin"]
            for point in condition_summaries["full"]["dose_response"]
        ]
    )
    j_curve = np.asarray(
        [
            point["mean_target_margin"]
            for point in condition_summaries["j"]["dose_response"]
        ]
    )
    correlation = (
        float(np.corrcoef(full_curve, j_curve)[0, 1])
        if np.std(full_curve) > 0 and np.std(j_curve) > 0
        else None
    )
    full_slope = float(condition_summaries["full"]["target_margin_slope"])
    j_slope = float(condition_summaries["j"]["target_margin_slope"])
    random_slopes = [
        float(value["target_margin_slope"])
        for key, value in condition_summaries.items()
        if key.startswith("random_")
    ]
    return {
        "conditions": condition_summaries,
        "j_full_curve_correlation": correlation,
        "j_to_full_slope_ratio": (
            j_slope / full_slope if abs(full_slope) > 1e-12 else None
        ),
        "random_slope_mean": float(np.mean(random_slopes)),
        "random_slope_std": float(np.std(random_slopes, ddof=1)),
        "j_slope_exceeds_all_random_controls": (
            abs(j_slope) > max(abs(value) for value in random_slopes)
        ),
    }


def run_concept_intervention(
    *,
    output_dir: str | Path,
    model: Any,
    tokenizer: Any,
    target_concept_id: str,
    candidate_labels: Mapping[str, str],
    prompts_path: str | Path,
    occupancy_dir: str | Path,
    probes_dir: str | Path,
    activations_dir: str | Path,
    expected_occupancy_index_sha256: str,
    expected_occupancy_git_commit: str,
    layer: int,
    convention: str,
    strengths: Sequence[float],
    random_seeds: Sequence[int],
    score_batch_size: int,
    generation_prompt_count: int,
    max_new_tokens: int,
    run_metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one target concept and save resumable causal score artifacts."""

    destination = Path(output_dir) / "targets" / quote(target_concept_id, safe="")
    summary_path = destination / "summary.json"
    if summary_path.is_file() and not overwrite:
        return {"status": "already_complete", "summary": str(summary_path)}
    if target_concept_id not in candidate_labels:
        raise ConceptInterventionError(
            f"target concept absent from candidate labels: {target_concept_id}"
        )
    directions, provenance = load_registered_directions(
        occupancy_dir=occupancy_dir,
        probes_dir=probes_dir,
        layer=layer,
        concept_id=target_concept_id,
        convention=convention,
        random_seeds=random_seeds,
        expected_occupancy_index_sha256=expected_occupancy_index_sha256,
        expected_occupancy_git_commit=expected_occupancy_git_commit,
    )
    prompts = load_prompt_bank(
        prompts_path,
        tokenizer=tokenizer,
        candidate_labels=candidate_labels,
    )
    token_ids = candidate_token_ids(tokenizer, candidate_labels)
    residual_norm = _median_residual_norm(activations_dir, layer=layer)

    score_rows: list[dict[str, Any]] = []
    for condition in directions:
        for strength in strengths:
            rows = _score_condition(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                candidate_token_ids=token_ids,
                target_concept_id=target_concept_id,
                layer=layer,
                direction=condition.direction,
                strength=float(strength),
                residual_norm=residual_norm,
                batch_size=score_batch_size,
            )
            for row in rows:
                row.update(
                    {
                        "condition_id": condition.condition_id,
                        "condition": condition.condition,
                        "random_seed": condition.random_seed,
                    }
                )
            score_rows.extend(rows)

    zero_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in score_rows:
        if float(row["strength"]) == 0.0:
            zero_rows[str(row["prompt_id"])].append(row)
    zero_spread = max(
        max(float(row["target_log_probability"]) for row in values)
        - min(float(row["target_log_probability"]) for row in values)
        for values in zero_rows.values()
    )
    if zero_spread > 1e-5:
        raise ConceptInterventionError(
            f"zero-strength conditions disagree by {zero_spread}"
        )

    generation_rows: list[dict[str, Any]] = []
    generation_conditions = [
        record
        for record in directions
        if record.condition != "random"
        or record.random_seed == int(random_seeds[0])
    ]
    generation_strengths = (min(strengths), 0.0, max(strengths))
    for condition in generation_conditions:
        for strength in generation_strengths:
            for prompt in prompts[:generation_prompt_count]:
                text = generate_with_intervention(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt.formatted_text,
                    layer=layer,
                    direction=condition.direction,
                    strength=float(strength),
                    kind="addition",
                    position="last_prompt",
                    residual_norm=residual_norm,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    add_special_tokens=False,
                )
                generation_rows.append(
                    {
                        "prompt_id": prompt.prompt_id,
                        "target_concept_id": target_concept_id,
                        "condition_id": condition.condition_id,
                        "condition": condition.condition,
                        "random_seed": condition.random_seed,
                        "strength": float(strength),
                        "generated_text": text,
                        "character_count": len(text),
                    }
                )

    summary = {
        "schema_version": 1,
        "method": "concept_j_component_intervention_v2",
        "target_concept_id": target_concept_id,
        "layer": layer,
        "convention": convention,
        "coordinate": "resid_post",
        "position": "last_prompt",
        "normalization": "unit_direction_times_median_layer_residual_norm",
        "median_layer_residual_norm": residual_norm,
        "strengths": list(strengths),
        "candidate_labels": dict(candidate_labels),
        "candidate_token_ids": token_ids,
        "n_prompts": len(prompts),
        "n_score_rows": len(score_rows),
        "n_generation_rows": len(generation_rows),
        "zero_strength_max_logprob_spread": zero_spread,
        "source_provenance": provenance,
        "analysis": _curve_summary(score_rows),
        "run_metadata": dict(run_metadata or {}),
    }
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(destination / "scores.jsonl", score_rows)
    atomic_write_jsonl(destination / "generations.jsonl", generation_rows)
    atomic_write_json(summary_path, summary)
    return {"status": "completed", "summary": str(summary_path), **summary}


def rebuild_intervention_index(
    output_dir: str | Path,
    *,
    expected_concepts: Sequence[str],
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate all target summaries and atomically rebuild the shared index."""

    root = Path(output_dir)
    entries: list[dict[str, Any]] = []
    observed: set[str] = set()
    for summary_path in sorted((root / "targets").glob("*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("method") != "concept_j_component_intervention_v2":
            raise ConceptInterventionError(f"unsupported summary: {summary_path}")
        concept_id = str(payload["target_concept_id"])
        if concept_id in observed:
            raise ConceptInterventionError(f"duplicate concept summary: {concept_id}")
        observed.add(concept_id)
        entries.append(
            {
                "target_concept_id": concept_id,
                "summary": str(summary_path.relative_to(root)),
                "summary_sha256": sha256_file(summary_path),
                "selected_k": payload["source_provenance"]["selected_k"],
                "analysis": payload["analysis"],
            }
        )
    expected = set(expected_concepts)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    index = {
        "schema_version": 1,
        "method": "concept_j_component_intervention_v2",
        "complete": not missing and not extra,
        "expected_concepts": list(expected_concepts),
        "observed_concepts": sorted(observed),
        "missing_concepts": missing,
        "extra_concepts": extra,
        "entries": entries,
        "run_metadata": dict(run_metadata or {}),
    }
    atomic_write_json(root / "index.json", index)
    return index
