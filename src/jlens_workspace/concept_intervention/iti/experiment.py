"""Unversioned exhaustive ITI-native and ITI-layer-matched experiment."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

from jlens_workspace.artifacts import atomic_write_json, sha256_file
from jlens_workspace.concept_intervention.evaluation import (
    atomic_write_jsonl,
    candidate_token_ids,
    load_prompt_bank,
)
from jlens_workspace.concept_intervention.generation import (
    GenerationSettings,
    build_generation_contract,
    generate_full_grid,
    load_open_prompt_bank,
    validate_equivalent_generation_outputs,
    validate_generation_artifacts,
    write_generation_artifacts,
)
from jlens_workspace.concept_intervention.iti.intervention import (
    iti_intervention_session,
    load_iti_head_shifts,
)
from jlens_workspace.concept_intervention.iti.workflow import (
    ITIWorkflowError,
    _best_validation_setting,
    _score_condition,
)


def _settings(value: Any) -> GenerationSettings:
    return GenerationSettings(
        sample_seeds=tuple(value.sample_seeds),
        max_new_tokens=value.max_new_tokens,
        temperature=value.temperature,
        top_p=value.top_p,
        repetition_penalty=value.repetition_penalty,
        no_repeat_ngram_size=value.no_repeat_ngram_size,
    )


def iti_experiment_grid(config: Any) -> list[dict[str, Any]]:
    """Return the deterministic native/layer-matched ITI grid order."""

    grid: list[dict[str, Any]] = []
    conditions = [("mass_mean", None), ("probe_weight", None)]
    conditions.extend(("random", int(seed)) for seed in config.random_control_seeds)
    for variant in config.variants:
        top_k_grid = (
            config.top_k_grid
            if variant == "native"
            else config.layer_matched_top_k_grid
        )
        for mode, random_seed in conditions:
            condition_id = (
                f"iti_{variant}_{mode}"
                if random_seed is None
                else f"iti_{variant}_random_{random_seed}"
            )
            for top_k in top_k_grid:
                for strength in config.strengths:
                    grid.append(
                        {
                            "variant": variant,
                            "mode": mode,
                            "random_seed": random_seed,
                            "condition_id": condition_id,
                            "top_k": int(top_k),
                            "strength": float(strength),
                        }
                    )
    return grid


def run_iti_intervention_experiment(
    *,
    output_dir: str | Path,
    model: Any,
    tokenizer: Any,
    config: Any,
    concept_id: str,
    grid_index: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run every ITI variant/control/K/alpha and all registered decodings."""

    target_root = Path(output_dir) / "targets" / quote(concept_id, safe="")
    grid = iti_experiment_grid(config)
    if grid_index is not None and not 0 <= grid_index < len(grid):
        raise ITIWorkflowError(
            f"ITI grid_index must lie in [0, {len(grid)}), got {grid_index}"
        )
    selected_grid = grid if grid_index is None else [grid[grid_index]]
    destination = (
        target_root
        if grid_index is None
        else target_root / "shards" / f"grid_{grid_index:04d}"
    )
    if destination.exists() and not overwrite:
        summary = destination / "summary.json"
        if summary.is_file():
            return {"status": "already_complete", "summary": str(summary)}
        raise FileExistsError(f"incomplete ITI target exists: {destination}")
    if config.generation is None:
        raise ITIWorkflowError("unversioned ITI requires shared generation settings")
    prompts = load_prompt_bank(
        config.generation.candidate_prompts_path,
        tokenizer=tokenizer,
        candidate_labels=config.candidate_labels,
    )
    validation_prompts = [
        prompt
        for prompt in prompts
        if prompt.prompt_id.partition("_")[0]
        in set(config.validation_prompt_prefixes)
    ]
    test_prompts = [
        prompt
        for prompt in prompts
        if prompt.prompt_id.partition("_")[0] in set(config.test_prompt_prefixes)
    ]
    if not validation_prompts or not test_prompts:
        raise ITIWorkflowError("ITI prompt validation/test partitions are empty")
    all_prompts: list[Any] = [
        *prompts,
        *load_open_prompt_bank(
            config.generation.open_prompts_path, tokenizer=tokenizer
        ),
    ]
    token_ids = candidate_token_ids(tokenizer, config.candidate_labels)
    settings = _settings(config.generation)
    generation_contract = build_generation_contract(all_prompts, settings)
    score_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    validation_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    direction_metrics = (
        Path(config.directions_dir) / quote(concept_id, safe="") / "metrics.json"
    )
    metrics = json.loads(direction_metrics.read_text(encoding="utf-8"))
    expected_layers = set(metrics["layers"])

    for point in selected_grid:
        variant = str(point["variant"])
        mode = str(point["mode"])
        random_seed = point["random_seed"]
        condition_id = str(point["condition_id"])
        top_k = int(point["top_k"])
        strength = float(point["strength"])
        shifts = load_iti_head_shifts(
            config.directions_dir,
            concept_id=concept_id,
            mode=mode,
            top_k=top_k,
            random_seed=random_seed,
            variant=variant,
        )
        active_layers = sorted({int(shift.layer) for shift in shifts})
        if variant == "layer_matched" and set(active_layers) != expected_layers:
            raise ITIWorkflowError(
                "layer-matched ITI did not cover every shared layer"
            )
        for split, split_prompts in (
            ("validation", validation_prompts),
            ("test", test_prompts),
        ):
            rows = _score_condition(
                model=model,
                tokenizer=tokenizer,
                prompts=split_prompts,
                candidate_token_ids=token_ids,
                target_concept_id=concept_id,
                shifts=shifts,
                multiplier=strength,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                batch_size=config.score_batch_size,
            )
            for row in rows:
                row.update(
                    {
                        "method": "iti_intervention",
                        "variant": variant,
                        "condition_id": condition_id,
                        "condition": mode,
                        "random_seed": random_seed,
                        "top_k": top_k,
                        "active_layers": active_layers,
                        "evaluation_split": split,
                    }
                )
            score_rows.extend(rows)
            if split == "validation" and mode == "mass_mean":
                validation_by_variant[variant].extend(rows)
        generation_rows.extend(
            generate_full_grid(
                model=model,
                tokenizer=tokenizer,
                prompts=all_prompts,
                method="iti_intervention",
                concept_id=concept_id,
                condition_id=condition_id,
                grid_point={
                    "variant": variant,
                    "top_k": top_k,
                    "strength": strength,
                },
                intervention_metadata={
                    "selected_layers": list(metrics["layers"]),
                    "active_layers": active_layers,
                    "selected_heads": [
                        {
                            "rank": int(shift.rank),
                            "layer": int(shift.layer),
                            "head": int(shift.head),
                            "validation_accuracy": float(
                                shift.validation_accuracy
                            ),
                            "projection_std": float(shift.projection_std),
                        }
                        for shift in shifts
                    ],
                    "direction_mode": mode,
                    "random_seed": random_seed,
                },
                session_factory=lambda s=shifts, a=strength: (
                    iti_intervention_session(
                        model,
                        shifts=s,
                        multiplier=a,
                        num_heads=config.num_heads,
                        head_dim=config.head_dim,
                    )
                ),
                settings=settings,
            )
        )
    selections = (
        {
            variant: _best_validation_setting(validation_by_variant[variant])
            for variant in config.variants
        }
        if grid_index is None
        else {}
    )

    zero_by_key: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in score_rows:
        if float(row["strength"]) == 0.0:
            zero_by_key[
                (
                    str(row["prompt_id"]),
                    int(row["top_k"]),
                    str(row["variant"]),
                )
            ].append(float(row["target_log_probability"]))
    zero_spread = (
        max(max(values) - min(values) for values in zero_by_key.values())
        if zero_by_key
        else None
    )
    if zero_spread is not None and zero_spread > 1e-5:
        raise ITIWorkflowError(f"zero-strength ITI conditions differ by {zero_spread}")
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(destination / "candidate_scores.jsonl", score_rows)
    generation_files = write_generation_artifacts(destination, generation_rows)
    summary = {
        "schema_version": 1,
        "method": "iti_intervention",
        "target_concept_id": concept_id,
        "variants": list(config.variants),
        "selected_layers": metrics["layers"],
        "coordinate": "attention_head_output_pre_o_proj",
        "position": "last_token_each_forward_call",
        "normalization": "unit_direction_times_projection_std_times_alpha",
        "native_top_k_grid": list(config.top_k_grid),
        "layer_matched_top_k_grid": list(config.layer_matched_top_k_grid),
        "strengths": list(config.strengths),
        "grid_index": grid_index,
        "grid_size": len(grid),
        "grid_condition": None if grid_index is None else selected_grid[0],
        "selection": selections,
        "direction_metrics": str(direction_metrics),
        "direction_metrics_sha256": sha256_file(direction_metrics),
        "candidate_score_rows": len(score_rows),
        "candidate_scores_sha256": sha256_file(
            destination / "candidate_scores.jsonl"
        ),
        "generation_rows": len(generation_rows),
        "generation_files": generation_files,
        "generation_contract": generation_contract,
        "zero_strength_max_logprob_spread": zero_spread,
        "llm_as_judge_run": False,
    }
    atomic_write_json(destination / "summary.json", summary)
    return {
        "status": "completed",
        "summary": str(destination / "summary.json"),
        **summary,
    }


def rebuild_iti_experiment_index(
    output_dir: str | Path,
    *,
    concept_ids: Sequence[str],
    config: Any | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    observed: set[str] = set()
    entries = []
    expected_grid_size = None if config is None else len(iti_experiment_grid(config))
    for concept_id in concept_ids:
        target = root / "targets" / quote(concept_id, safe="")
        path = target / "summary.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            shard_paths = sorted((target / "shards").glob("grid_*/summary.json"))
            shard_payloads = [
                json.loads(shard.read_text(encoding="utf-8"))
                for shard in shard_paths
            ]
            for shard, shard_payload in zip(
                shard_paths, shard_payloads, strict=True
            ):
                candidate_path = shard.parent / "candidate_scores.jsonl"
                if shard_payload.get("candidate_scores_sha256") != sha256_file(
                    candidate_path
                ):
                    raise ITIWorkflowError(
                        f"ITI candidate-score identity mismatch: {candidate_path}"
                    )
                validate_generation_artifacts(
                    shard.parent,
                    shard_payload["generation_files"],
                    contract=shard_payload["generation_contract"],
                )
            indices = {row.get("grid_index") for row in shard_payloads}
            if (
                expected_grid_size is None
                or len(shard_payloads) != expected_grid_size
                or indices != set(range(expected_grid_size))
                or any(
                    int(row.get("grid_size", -1)) != expected_grid_size
                    for row in shard_payloads
                )
            ):
                continue
            layers = {
                tuple(int(value) for value in row["selected_layers"])
                for row in shard_payloads
            }
            if len(layers) != 1:
                raise ITIWorkflowError(
                    f"{concept_id}: ITI shards used different selected layers"
                )
            validation: dict[str, list[dict[str, Any]]] = defaultdict(list)
            zero_by_key: dict[tuple[str, int, str], list[float]] = defaultdict(list)
            zero_generation_paths: list[Path] = []
            for shard, shard_payload in zip(
                shard_paths, shard_payloads, strict=True
            ):
                if float(shard_payload["grid_condition"]["strength"]) == 0.0:
                    zero_generation_paths.append(
                        shard.parent / "generations.jsonl"
                    )
                with (shard.parent / "candidate_scores.jsonl").open(
                    encoding="utf-8"
                ) as handle:
                    for line in handle:
                        row = json.loads(line)
                        if (
                            row["evaluation_split"] == "validation"
                            and row["condition"] == "mass_mean"
                        ):
                            validation[str(row["variant"])].append(row)
                        if float(row["strength"]) == 0.0:
                            zero_by_key[
                                (
                                    str(row["prompt_id"]),
                                    int(row["top_k"]),
                                    str(row["variant"]),
                                )
                            ].append(float(row["target_log_probability"]))
            selections = {
                variant: _best_validation_setting(validation[variant])
                for variant in config.variants
            }
            zero_spread = max(
                max(values) - min(values) for values in zero_by_key.values()
            )
            if zero_spread > 1e-5:
                raise ITIWorkflowError(
                    f"{concept_id}: zero-strength ITI shards differ by {zero_spread}"
                )
            zero_generation_consistency = validate_equivalent_generation_outputs(
                zero_generation_paths
            )
            first = shard_payloads[0]
            payload = {
                "schema_version": 1,
                "method": "iti_intervention",
                "target_concept_id": concept_id,
                "variants": list(config.variants),
                "selected_layers": list(next(iter(layers))),
                "coordinate": first["coordinate"],
                "position": first["position"],
                "normalization": first["normalization"],
                "native_top_k_grid": list(config.top_k_grid),
                "layer_matched_top_k_grid": list(
                    config.layer_matched_top_k_grid
                ),
                "strengths": list(config.strengths),
                "selection": selections,
                "direction_metrics": first["direction_metrics"],
                "direction_metrics_sha256": first[
                    "direction_metrics_sha256"
                ],
                "zero_strength_max_logprob_spread": zero_spread,
                "zero_generation_consistency": zero_generation_consistency,
                "generation_artifacts_sharded": True,
                "generation_contract": first["generation_contract"],
                "grid_size": expected_grid_size,
                "shards": [
                    {
                        "grid_index": row["grid_index"],
                        "grid_condition": row["grid_condition"],
                        "summary": str(shard.relative_to(root)),
                        "summary_sha256": sha256_file(shard),
                    }
                    for shard, row in zip(
                        shard_paths, shard_payloads, strict=True
                    )
                ],
                "llm_as_judge_run": False,
            }
            atomic_write_json(path, payload)
        if payload.get("method") != "iti_intervention":
            raise ITIWorkflowError(f"unsupported ITI summary: {path}")
        if str(payload["target_concept_id"]) != concept_id:
            raise ITIWorkflowError(f"ITI concept identity mismatch: {path}")
        observed.add(concept_id)
        entries.append(
            {
                "concept_id": concept_id,
                "summary": str(path.relative_to(root)),
                "summary_sha256": sha256_file(path),
                "selected_layers": payload["selected_layers"],
                "selection": payload["selection"],
            }
        )
    expected = set(concept_ids)
    index = {
        "schema_version": 1,
        "method": "iti_intervention",
        "complete": observed == expected,
        "expected_concepts": list(concept_ids),
        "observed_concepts": sorted(observed),
        "missing_concepts": sorted(expected - observed),
        "extra_concepts": sorted(observed - expected),
        "manifest_sha256": (
            sha256_file(root / "manifest.json")
            if (root / "manifest.json").is_file()
            else None
        ),
        "entries": entries,
    }
    atomic_write_json(root / "index.json", index)
    return index
