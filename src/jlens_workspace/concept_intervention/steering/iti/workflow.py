"""Validation-selected ITI intervention and held-out comparison workflow."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import yaml

from jlens_workspace.concept_intervention.evaluation import (
    PromptRecord,
    atomic_write_jsonl,
    batched,
    candidate_token_ids,
    load_prompt_bank,
)
from jlens_workspace.concept_intervention.steering.iti.intervention import (
    ITI_METHOD,
    iti_intervention_session,
    load_iti_head_shifts,
)
from jlens_workspace.foundation.activations import _capture_forward_kwargs
from jlens_workspace.foundation.artifacts import atomic_write_json, sha256_file
from jlens_workspace.foundation.modeling import model_input_device


class ITIWorkflowError(ValueError):
    """Raised when ITI fitting, selection, or held-out evaluation is inconsistent."""


def _prompt_partition(
    prompts: Sequence[PromptRecord], prefixes: Sequence[str]
) -> list[PromptRecord]:
    allowed = set(prefixes)
    selected = [
        prompt for prompt in prompts if prompt.prompt_id.partition("_")[0] in allowed
    ]
    if not selected:
        raise ITIWorkflowError(f"no prompts matched prefixes {sorted(allowed)}")
    return selected


def _score_condition(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[PromptRecord],
    candidate_token_ids: Mapping[str, int],
    target_concept_id: str,
    shifts: Sequence[Any],
    multiplier: float,
    num_heads: int,
    head_dim: int,
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
            with torch.inference_mode(), iti_intervention_session(
                model,
                shifts=shifts,
                multiplier=multiplier,
                num_heads=num_heads,
                head_dim=head_dim,
            ):
                output = model(**encoded, **_capture_forward_kwargs(model))
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
                rows.append(
                    {
                        "prompt_id": prompt.prompt_id,
                        "target_concept_id": target_concept_id,
                        "strength": float(multiplier),
                        "candidate_log_probabilities": log_map,
                        "candidate_probabilities_normalized": prob_map,
                        "target_log_probability": target_log_prob,
                        "target_candidate_probability": prob_map[target_concept_id],
                        "target_margin": target_log_prob - float(np.mean(off_target)),
                        "target_rank": 1
                        + sum(value > target_log_prob for value in off_target),
                    }
                )
    finally:
        tokenizer.padding_side = original_padding_side
    return rows


def _curve_summary(score_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in score_rows:
        grouped[(str(row["condition_id"]), float(row["strength"]))].append(row)
    conditions: dict[str, dict[str, Any]] = {}
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
    for condition_id, points in curves.items():
        strengths = np.asarray([point["strength"] for point in points])
        margins = np.asarray([point["mean_target_margin"] for point in points])
        conditions[condition_id] = {
            "dose_response": points,
            "target_margin_slope": float(np.polyfit(strengths, margins, deg=1)[0]),
            "signed_endpoint_effect": float(margins[-1] - margins[0]),
        }
    primary_slope = float(conditions["mass_mean"]["target_margin_slope"])
    random_slopes = [
        float(value["target_margin_slope"])
        for key, value in conditions.items()
        if key.startswith("random_")
    ]
    return {
        "conditions": conditions,
        "mass_mean_slope_exceeds_all_random_controls": abs(primary_slope)
        > max(abs(value) for value in random_slopes),
        "random_slope_mean": float(np.mean(random_slopes)),
        "random_slope_std": float(np.std(random_slopes, ddof=1)),
    }


def _best_validation_setting(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, float], list[float]] = defaultdict(list)
    for row in rows:
        strength = float(row["strength"])
        if strength > 0:
            grouped[(int(row["top_k"]), strength)].append(float(row["target_margin"]))
    if not grouped:
        raise ITIWorkflowError("ITI validation grid contains no positive strengths")
    candidates = [
        {
            "top_k": top_k,
            "strength": strength,
            "mean_target_margin": float(np.mean(values)),
            "n_prompts": len(values),
        }
        for (top_k, strength), values in grouped.items()
    ]
    selected = max(
        candidates,
        key=lambda row: (
            float(row["mean_target_margin"]),
            -int(row["top_k"]),
            -float(row["strength"]),
        ),
    )
    return {"selected": selected, "candidates": sorted(candidates, key=lambda x: (x["top_k"], x["strength"]))}


def _generate(
    *,
    model: Any,
    tokenizer: Any,
    prompt: PromptRecord,
    shifts: Sequence[Any],
    multiplier: float,
    num_heads: int,
    head_dim: int,
    max_new_tokens: int,
) -> str:
    import torch

    encoded = tokenizer(
        prompt.formatted_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = {
        key: value.to(model_input_device(model)) for key, value in encoded.items()
    }
    with torch.inference_mode(), iti_intervention_session(
        model,
        shifts=shifts,
        multiplier=multiplier,
        num_heads=num_heads,
        head_dim=head_dim,
    ):
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def run_iti_intervention(
    *,
    output_dir: str | Path,
    model: Any,
    tokenizer: Any,
    target_concept_id: str,
    candidate_labels: Mapping[str, str],
    prompts_path: str | Path,
    direction_dir: str | Path,
    top_k_grid: Sequence[int],
    strengths: Sequence[float],
    random_seeds: Sequence[int],
    validation_prompt_prefixes: Sequence[str],
    test_prompt_prefixes: Sequence[str],
    num_heads: int,
    head_dim: int,
    score_batch_size: int,
    generation_prompt_count: int,
    max_new_tokens: int,
    run_metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Select ITI K/alpha on prompt validation templates and score held-out templates."""

    destination = Path(output_dir) / "targets" / quote(target_concept_id, safe="")
    summary_path = destination / "summary.json"
    if summary_path.is_file() and not overwrite:
        return {"status": "already_complete", "summary": str(summary_path)}
    if target_concept_id not in candidate_labels:
        raise ITIWorkflowError(f"unknown target concept {target_concept_id!r}")
    direction_metrics_path = (
        Path(direction_dir) / quote(target_concept_id, safe="") / "metrics.json"
    )
    direction_metrics = json.loads(direction_metrics_path.read_text(encoding="utf-8"))
    if direction_metrics.get("method") != ITI_METHOD:
        raise ITIWorkflowError("direction artifact is not the registered ITI method")
    if (
        int(direction_metrics["num_heads"]) != num_heads
        or int(direction_metrics["head_dim"]) != head_dim
    ):
        raise ITIWorkflowError("configured ITI head layout differs from fitted directions")

    all_prompts = load_prompt_bank(
        prompts_path,
        tokenizer=tokenizer,
        candidate_labels=candidate_labels,
    )
    validation_prompts = _prompt_partition(all_prompts, validation_prompt_prefixes)
    test_prompts = _prompt_partition(all_prompts, test_prompt_prefixes)
    if {p.prompt_id for p in validation_prompts}.intersection(
        p.prompt_id for p in test_prompts
    ):
        raise ITIWorkflowError("validation and test prompt templates overlap")
    token_ids = candidate_token_ids(tokenizer, candidate_labels)

    validation_rows: list[dict[str, Any]] = []
    for top_k in top_k_grid:
        shifts = load_iti_head_shifts(
            direction_dir,
            concept_id=target_concept_id,
            mode="mass_mean",
            top_k=int(top_k),
        )
        for strength in strengths:
            rows = _score_condition(
                model=model,
                tokenizer=tokenizer,
                prompts=validation_prompts,
                candidate_token_ids=token_ids,
                target_concept_id=target_concept_id,
                shifts=shifts,
                multiplier=float(strength),
                num_heads=num_heads,
                head_dim=head_dim,
                batch_size=score_batch_size,
            )
            for row in rows:
                row.update(
                    {
                        "evaluation_split": "validation",
                        "condition_id": "mass_mean",
                        "condition": "mass_mean",
                        "random_seed": None,
                        "top_k": int(top_k),
                    }
                )
            validation_rows.extend(rows)
    selection = _best_validation_setting(validation_rows)
    selected_k = int(selection["selected"]["top_k"])

    conditions = [("mass_mean", "mass_mean", None), ("probe_weight", "probe_weight", None)]
    conditions.extend(
        (f"random_{int(seed)}", "random", int(seed)) for seed in random_seeds
    )
    test_rows: list[dict[str, Any]] = []
    for condition_id, mode, random_seed in conditions:
        shifts = load_iti_head_shifts(
            direction_dir,
            concept_id=target_concept_id,
            mode=mode,
            top_k=selected_k,
            random_seed=random_seed,
        )
        for strength in strengths:
            rows = _score_condition(
                model=model,
                tokenizer=tokenizer,
                prompts=test_prompts,
                candidate_token_ids=token_ids,
                target_concept_id=target_concept_id,
                shifts=shifts,
                multiplier=float(strength),
                num_heads=num_heads,
                head_dim=head_dim,
                batch_size=score_batch_size,
            )
            for row in rows:
                row.update(
                    {
                        "evaluation_split": "test",
                        "condition_id": condition_id,
                        "condition": mode,
                        "random_seed": random_seed,
                        "top_k": selected_k,
                    }
                )
            test_rows.extend(rows)

    zero_rows: dict[str, list[float]] = defaultdict(list)
    for row in test_rows:
        if float(row["strength"]) == 0.0:
            zero_rows[str(row["prompt_id"])].append(
                float(row["target_log_probability"])
            )
    zero_spread = max(max(values) - min(values) for values in zero_rows.values())
    if zero_spread > 1e-5:
        raise ITIWorkflowError(f"zero-strength ITI conditions differ by {zero_spread}")

    generation_rows: list[dict[str, Any]] = []
    generation_conditions = [*conditions[:2], conditions[2]]
    generation_strengths = (min(strengths), 0.0, max(strengths))
    for condition_id, mode, random_seed in generation_conditions:
        shifts = load_iti_head_shifts(
            direction_dir,
            concept_id=target_concept_id,
            mode=mode,
            top_k=selected_k,
            random_seed=random_seed,
        )
        for strength in generation_strengths:
            for prompt in test_prompts[:generation_prompt_count]:
                generated = _generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    shifts=shifts,
                    multiplier=float(strength),
                    num_heads=num_heads,
                    head_dim=head_dim,
                    max_new_tokens=max_new_tokens,
                )
                generation_rows.append(
                    {
                        "prompt_id": prompt.prompt_id,
                        "target_concept_id": target_concept_id,
                        "condition_id": condition_id,
                        "condition": mode,
                        "random_seed": random_seed,
                        "strength": float(strength),
                        "top_k": selected_k,
                        "generated_text": generated,
                        "character_count": len(generated),
                    }
                )

    summary = {
        "schema_version": 1,
        "method": ITI_METHOD,
        "target_concept_id": target_concept_id,
        "coordinate": "attention_head_output_pre_o_proj",
        "position": "last_token_each_forward_call",
        "normalization": "unit_direction_times_projection_std_times_alpha",
        "candidate_labels": dict(candidate_labels),
        "candidate_token_ids": token_ids,
        "top_k_grid": list(top_k_grid),
        "strengths": list(strengths),
        "selection": selection,
        "validation_prompt_ids": [prompt.prompt_id for prompt in validation_prompts],
        "test_prompt_ids": [prompt.prompt_id for prompt in test_prompts],
        "test_examples_used_for_fitting": 0,
        "n_validation_score_rows": len(validation_rows),
        "n_test_score_rows": len(test_rows),
        "n_generation_rows": len(generation_rows),
        "zero_strength_max_logprob_spread": zero_spread,
        "direction_metrics": str(direction_metrics_path),
        "direction_metrics_sha256": sha256_file(direction_metrics_path),
        "analysis": _curve_summary(test_rows),
        "evaluation": {
            "primary": "held-out candidate-label log probability and target margin",
            "llm_as_judge_required": False,
            "reason": "the registered task has deterministic one-token candidate labels",
        },
        "run_metadata": dict(run_metadata or {}),
    }
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(destination / "validation_scores.jsonl", validation_rows)
    atomic_write_jsonl(destination / "test_scores.jsonl", test_rows)
    atomic_write_jsonl(destination / "generations.jsonl", generation_rows)
    atomic_write_json(summary_path, summary)
    return {"status": "completed", "summary": str(summary_path), **summary}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mean_margin(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        raise ITIWorkflowError("cannot summarize an empty score selection")
    return float(np.mean([float(row["target_margin"]) for row in rows]))


def _selected_effect(
    rows: Sequence[Mapping[str, Any]], *, condition_id: str, strength: float
) -> float:
    selected = [
        row
        for row in rows
        if str(row["condition_id"]) == condition_id
        and math.isclose(float(row["strength"]), strength, abs_tol=1e-12)
    ]
    baseline = [
        row
        for row in rows
        if str(row["condition_id"]) == condition_id
        and math.isclose(float(row["strength"]), 0.0, abs_tol=1e-12)
    ]
    return _mean_margin(selected) - _mean_margin(baseline)


def _paired_effect_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition_id: str,
    strength: float,
    seed: int,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    selected: dict[str, float] = {}
    baseline: dict[str, float] = {}
    for row in rows:
        if str(row["condition_id"]) != condition_id:
            continue
        prompt_id = str(row["prompt_id"])
        value = float(row["target_margin"])
        if math.isclose(float(row["strength"]), strength, abs_tol=1e-12):
            selected[prompt_id] = value
        if math.isclose(float(row["strength"]), 0.0, abs_tol=1e-12):
            baseline[prompt_id] = value
    if not selected or set(selected) != set(baseline):
        raise ITIWorkflowError(
            f"paired scores are incomplete for {condition_id} at strength {strength}"
        )
    prompt_ids = sorted(selected)
    deltas = np.asarray(
        [selected[prompt_id] - baseline[prompt_id] for prompt_id in prompt_ids],
        dtype=np.float64,
    )
    rng = np.random.Generator(np.random.Philox(seed))
    bootstrap = deltas[
        rng.integers(0, len(deltas), size=(bootstrap_samples, len(deltas)))
    ].mean(axis=1)
    return {
        "mean": float(np.mean(deltas)),
        "median": float(np.median(deltas)),
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "bootstrap_samples": bootstrap_samples,
        "n_paired_prompts": len(deltas),
        "prompt_ids": prompt_ids,
    }


def _prefix_selected(
    rows: Sequence[Mapping[str, Any]], prefixes: Sequence[str]
) -> list[Mapping[str, Any]]:
    allowed = set(prefixes)
    return [
        row for row in rows if str(row["prompt_id"]).partition("_")[0] in allowed
    ]


def _build_j_comparison(
    *,
    iti_root: Path,
    j_root: Path,
    concepts: Sequence[str],
    validation_prompt_prefixes: Sequence[str],
    test_prompt_prefixes: Sequence[str],
) -> dict[str, Any]:
    j_index = json.loads((j_root / "index.json").read_text(encoding="utf-8"))
    if not j_index.get("complete"):
        raise ITIWorkflowError("reference J-component intervention index is incomplete")
    iti_manifest = json.loads((iti_root / "manifest.json").read_text(encoding="utf-8"))
    j_manifest = json.loads((j_root / "manifest.json").read_text(encoding="utf-8"))
    identity_fields = (
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "dataset_source",
        "dataset_revision",
    )
    mismatch = [
        field
        for field in identity_fields
        if iti_manifest.get(field) != j_manifest.get(field)
    ]
    if mismatch:
        raise ITIWorkflowError(
            "ITI/J reference model or dataset identity differs: " + ", ".join(mismatch)
        )

    j_notes = j_manifest.get("notes", {})
    j_config_path = Path(str(j_notes.get("config_path", "")))
    j_config_sha = j_notes.get("config_sha256")
    if not j_config_path.is_file() and j_config_path.name:
        local_candidate = (
            Path.cwd() / "Concept_intervention" / "configs" / j_config_path.name
        )
        if local_candidate.is_file():
            j_config_path = local_candidate
    if not isinstance(j_config_sha, str) or not j_config_path.is_file() or (
        sha256_file(j_config_path) != j_config_sha
    ):
        raise ITIWorkflowError("reference J intervention config identity is unavailable")
    j_config = yaml.safe_load(j_config_path.read_text(encoding="utf-8"))
    try:
        j_activation_dir = str(j_config["intervention"]["source_activations_dir"])
    except (KeyError, TypeError) as error:
        raise ITIWorkflowError(
            "reference J config does not identify its residual activation artifact"
        ) from error
    iti_notes = iti_manifest.get("notes", {})
    iti_activation_dir = str(iti_notes.get("source_residual_activations_dir", ""))
    if Path(j_activation_dir).resolve() != Path(iti_activation_dir).resolve():
        raise ITIWorkflowError(
            "ITI and J intervention do not reference the same residual activation artifact"
        )
    residual_metadata = Path(iti_activation_dir) / "metadata.json"
    residual_labels = Path(iti_activation_dir) / "labels.npy"
    if (
        sha256_file(residual_metadata)
        != iti_notes.get("source_residual_metadata_sha256")
        or sha256_file(residual_labels)
        != iti_notes.get("source_residual_labels_sha256")
    ):
        raise ITIWorkflowError("shared residual activation artifact content changed")

    def prompt_hash(root: Path, manifest: Mapping[str, Any]) -> str | None:
        value = manifest.get("notes", {}).get("prompts_sha256")
        if value is not None:
            return str(value)
        for shard in sorted((root / "manifests").glob("*.json")):
            payload = json.loads(shard.read_text(encoding="utf-8"))
            value = payload.get("notes", {}).get("prompts_sha256")
            if value is not None:
                return str(value)
        return None

    iti_prompt_hash = prompt_hash(iti_root, iti_manifest)
    j_prompt_hash = prompt_hash(j_root, j_manifest)
    if iti_prompt_hash is None or iti_prompt_hash != j_prompt_hash:
        raise ITIWorkflowError("ITI/J reference prompt banks differ")

    entries: list[dict[str, Any]] = []
    for concept_index, concept_id in enumerate(concepts):
        encoded = quote(concept_id, safe="")
        iti_summary = json.loads(
            (iti_root / "targets" / encoded / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        iti_rows = _read_jsonl(iti_root / "targets" / encoded / "test_scores.jsonl")
        j_rows_all = _read_jsonl(j_root / "targets" / encoded / "scores.jsonl")
        j_validation = _prefix_selected(j_rows_all, validation_prompt_prefixes)
        j_test = _prefix_selected(j_rows_all, test_prompt_prefixes)
        j_positive_strengths = sorted(
            {
                float(row["strength"])
                for row in j_validation
                if str(row["condition_id"]) == "j" and float(row["strength"]) > 0
            }
        )
        j_candidates = [
            {
                "strength": strength,
                "mean_target_margin": _mean_margin(
                    [
                        row
                        for row in j_validation
                        if str(row["condition_id"]) == "j"
                        and math.isclose(
                            float(row["strength"]), strength, abs_tol=1e-12
                        )
                    ]
                ),
            }
            for strength in j_positive_strengths
        ]
        selected_j = max(
            j_candidates,
            key=lambda row: (float(row["mean_target_margin"]), -float(row["strength"])),
        )
        iti_strength = float(iti_summary["selection"]["selected"]["strength"])
        j_strength = float(selected_j["strength"])
        iti_effect_summary = _paired_effect_summary(
            iti_rows,
            condition_id="mass_mean",
            strength=iti_strength,
            seed=10_000 + concept_index,
        )
        j_effect_summary = _paired_effect_summary(
            j_test,
            condition_id="j",
            strength=j_strength,
            seed=20_000 + concept_index,
        )
        iti_effect = float(iti_effect_summary["mean"])
        j_effect = float(j_effect_summary["mean"])
        iti_random = [
            _selected_effect(
                iti_rows, condition_id=f"random_{int(seed)}", strength=iti_strength
            )
            for seed in iti_summary["run_metadata"]["random_control_seeds"]
        ]
        j_random_ids = sorted(
            {
                str(row["condition_id"])
                for row in j_test
                if str(row["condition_id"]).startswith("random_")
            }
        )
        j_random = [
            _selected_effect(j_test, condition_id=condition, strength=j_strength)
            for condition in j_random_ids
        ]
        entries.append(
            {
                "concept_id": concept_id,
                "iti": {
                    "top_k": int(iti_summary["selection"]["selected"]["top_k"]),
                    "strength": iti_strength,
                    "held_out_target_margin_effect": iti_effect,
                    "paired_effect_summary": iti_effect_summary,
                    "random_control_effect_mean": float(np.mean(iti_random)),
                    "exceeds_all_random_controls": iti_effect > max(iti_random),
                },
                "j_component": {
                    "strength": j_strength,
                    "validation_candidates": j_candidates,
                    "held_out_target_margin_effect": j_effect,
                    "paired_effect_summary": j_effect_summary,
                    "random_control_effect_mean": float(np.mean(j_random)),
                    "exceeds_all_random_controls": j_effect > max(j_random),
                },
                "iti_minus_j_held_out_target_margin_effect": iti_effect - j_effect,
            }
        )
    return {
        "schema_version": 1,
        "comparison": "validation_selected_iti_mass_mean_vs_j_component",
        "same_model_tokenizer_dataset": True,
        "same_residual_activation_artifact": True,
        "shared_residual_activation_metadata_sha256": iti_notes[
            "source_residual_metadata_sha256"
        ],
        "shared_residual_activation_labels_sha256": iti_notes[
            "source_residual_labels_sha256"
        ],
        "same_prompt_bank": True,
        "validation_prompt_prefixes": list(validation_prompt_prefixes),
        "test_prompt_prefixes": list(test_prompt_prefixes),
        "primary_metric": "held_out_change_in_target_margin_relative_to_zero_strength",
        "llm_as_judge_required": False,
        "reference_j_index": str(j_root / "index.json"),
        "reference_j_index_sha256": sha256_file(j_root / "index.json"),
        "entries": entries,
        "macro_mean": {
            "iti_effect": float(
                np.mean([row["iti"]["held_out_target_margin_effect"] for row in entries])
            ),
            "j_component_effect": float(
                np.mean(
                    [
                        row["j_component"]["held_out_target_margin_effect"]
                        for row in entries
                    ]
                )
            ),
        },
    }


def rebuild_iti_index(
    output_dir: str | Path,
    *,
    expected_concepts: Sequence[str],
    reference_j_intervention_dir: str | Path,
    validation_prompt_prefixes: Sequence[str],
    test_prompt_prefixes: Sequence[str],
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate all ITI shards and build the identity-checked J comparison."""

    root = Path(output_dir)
    observed: set[str] = set()
    entries: list[dict[str, Any]] = []
    for path in sorted((root / "targets").glob("*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("method") != ITI_METHOD:
            raise ITIWorkflowError(f"unsupported ITI summary: {path}")
        concept_id = str(payload["target_concept_id"])
        if concept_id in observed:
            raise ITIWorkflowError(f"duplicate ITI concept summary: {concept_id}")
        observed.add(concept_id)
        entries.append(
            {
                "target_concept_id": concept_id,
                "summary": str(path.relative_to(root)),
                "summary_sha256": sha256_file(path),
                "selected": payload["selection"]["selected"],
                "analysis": payload["analysis"],
            }
        )
    expected = set(expected_concepts)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    complete = not missing and not extra
    index = {
        "schema_version": 1,
        "method": ITI_METHOD,
        "complete": complete,
        "expected_concepts": list(expected_concepts),
        "observed_concepts": sorted(observed),
        "missing_concepts": missing,
        "extra_concepts": extra,
        "entries": entries,
        "run_metadata": dict(run_metadata or {}),
    }
    atomic_write_json(root / "index.json", index)
    if complete:
        comparison = _build_j_comparison(
            iti_root=root,
            j_root=Path(reference_j_intervention_dir),
            concepts=expected_concepts,
            validation_prompt_prefixes=validation_prompt_prefixes,
            test_prompt_prefixes=test_prompt_prefixes,
        )
        atomic_write_json(root / "comparison.json", comparison)
        index["comparison"] = "comparison.json"
        index["comparison_sha256"] = sha256_file(root / "comparison.json")
        atomic_write_json(root / "index.json", index)
    return index
