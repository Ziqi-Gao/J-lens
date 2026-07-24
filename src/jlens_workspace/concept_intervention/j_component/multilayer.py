"""Immutable-suite multi-layer J-component intervention experiment."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from jlens_workspace.artifacts import atomic_write_json, sha256_file
from jlens_workspace.concept_intervention.evaluation import (
    PromptRecord,
    atomic_write_jsonl,
    batched,
    candidate_token_ids,
    load_prompt_bank,
)
from jlens_workspace.concept_intervention.generation import (
    GenerationSettings,
    InterventionGenerationError,
    build_generation_contract,
    build_target_artifact_seal,
    generate_full_grid,
    load_open_prompt_bank,
    validate_candidate_score_artifact,
    validate_equivalent_generation_outputs,
    validate_generation_artifacts,
    validate_generation_contract_identity,
    validate_shard_manifest_identity,
    write_generation_artifacts,
)
from jlens_workspace.concept_intervention.j_component.intervention import (
    multilayer_intervention_session,
)
from jlens_workspace.concept_intervention.shared_protocol import (
    load_balanced_indices,
    load_selected_layers,
)
from jlens_workspace.modeling import model_input_device


class MultiLayerJError(ValueError):
    """Raised when multi-layer J inputs are incomplete or identity-mismatched."""


def _occupancy_metrics(
    root: Path, *, layer: int, concept_id: str, replicate_id: str
) -> Path:
    return (
        root
        / "occupancy"
        / "rmsnorm_weighted"
        / "positive_cosine"
        / f"layer_{layer:02d}"
        / quote(concept_id, safe="")
        / "pos"
        / quote(replicate_id, safe="")
        / "metrics.json"
    )


def aggregate_layer_k(
    *,
    occupancy_dir: str | Path,
    concept_id: str,
    layer: int,
    replicate_ids: Sequence[str],
    k_max: int,
) -> dict[str, Any]:
    """Aggregate replicate-supported K and apply the declared K=1 floor."""

    values: list[int] = []
    replicates: list[dict[str, Any]] = []
    for replicate_id in replicate_ids:
        path = _occupancy_metrics(
            Path(occupancy_dir),
            layer=layer,
            concept_id=concept_id,
            replicate_id=replicate_id,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "method": "concept_occupancy_method",
            "solver_method": "nonnegative_gradient_pursuit_standard",
            "layer": layer,
            "concept_id": concept_id,
            "replicate_id": replicate_id,
            "convention": "rmsnorm_weighted",
            "selection_mode": "positive_cosine",
            "sign": "+",
            "k_max": k_max,
        }
        for field, expected in required.items():
            if payload.get(field) != expected:
                raise MultiLayerJError(
                    f"{path}: {field}={payload.get(field)!r}, expected {expected!r}"
                )
        primary = payload["primary_occupancy"]
        measured = int(primary["k_selected_before_crossing"])
        values.append(measured)
        replicates.append(
            {
                "replicate_id": replicate_id,
                "k_supported": measured,
                "crossing_k": primary["crossing_k"],
                "right_censored": bool(primary["right_censored"]),
                "metrics": str(path),
                "metrics_sha256": sha256_file(path),
            }
        )
    measured_k = int(np.median(np.asarray(values, dtype=np.int64)))
    used_k = max(1, measured_k)
    return {
        "layer": layer,
        "replicate_supported_k": values,
        "K_measured": measured_k,
        "K_used": used_k,
        "k_floor_applied": measured_k == 0,
        "statistically_supported": measured_k > 0,
        "right_censored": sum(row["right_censored"] for row in replicates)
        >= math.ceil(len(replicates) / 2),
        "k_max": k_max,
        "replicates": replicates,
    }


def _probe_path(root: Path, layer: int, concept_id: str) -> Path:
    return (
        root
        / f"layer_{layer:02d}"
        / f"concept_{quote(concept_id, safe='')}"
        / "probe_vector.npy"
    )


def load_multilayer_directions(
    *,
    occupancy_dir: str | Path,
    probes_dir: str | Path,
    selected_layers_path: str | Path,
    concept_id: str,
    replicate_ids: Sequence[str],
    k_max: int,
    random_seeds: Sequence[int],
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    """Load per-layer full/J/non-J/random directions at replicate-median K."""

    layers = load_selected_layers(selected_layers_path, concept_id)
    selection_path = Path(selected_layers_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    row_manifest_path = (
        selection_path.parent / str(selection["row_manifest"])
    ).resolve()
    row_manifest_sha256 = sha256_file(row_manifest_path)
    if selection.get("row_manifest_sha256") != row_manifest_sha256:
        raise MultiLayerJError(
            f"shared row-manifest identity mismatch: {row_manifest_path}"
        )
    directions: dict[str, dict[int, np.ndarray]] = {
        "full": {},
        "j": {},
        "non_j": {},
    }
    provenance: list[dict[str, Any]] = []
    for layer in layers:
        k_payload = aggregate_layer_k(
            occupancy_dir=occupancy_dir,
            concept_id=concept_id,
            layer=layer,
            replicate_ids=replicate_ids,
            k_max=k_max,
        )
        full_path = _probe_path(Path(probes_dir), layer, concept_id)
        full = np.asarray(np.load(full_path, allow_pickle=False), dtype=np.float64)
        primary_path = _occupancy_metrics(
            Path(occupancy_dir),
            layer=layer,
            concept_id=concept_id,
            replicate_id="primary",
        )
        primary = json.loads(primary_path.read_text(encoding="utf-8"))
        combo = primary_path.parent
        used_k = int(k_payload["K_used"])
        j_path = combo / f"w_J_k{used_k:02d}.npy"
        non_j_path = combo / f"w_nonJ_k{used_k:02d}.npy"
        if not j_path.is_file() or not non_j_path.is_file():
            raise MultiLayerJError(
                f"primary occupancy must save reconstruction K={used_k}: {combo}"
            )
        j_direction = np.asarray(np.load(j_path, allow_pickle=False), dtype=np.float64)
        non_j = np.asarray(np.load(non_j_path, allow_pickle=False), dtype=np.float64)
        if not np.allclose(full, j_direction + non_j, rtol=0.0, atol=1e-10):
            raise MultiLayerJError(f"full != J + non-J at layer {layer}")
        if primary["probe_vector_sha256"] != sha256_file(full_path):
            raise MultiLayerJError(f"probe vector identity mismatch at layer {layer}")
        directions["full"][layer] = full
        directions["j"][layer] = j_direction
        directions["non_j"][layer] = non_j
        provenance.append(
            {
                **k_payload,
                "probe_vector": str(full_path),
                "probe_vector_sha256": sha256_file(full_path),
                "j_vector": str(j_path),
                "j_vector_sha256": sha256_file(j_path),
                "non_j_vector": str(non_j_path),
                "non_j_vector_sha256": sha256_file(non_j_path),
                "primary_support_token_ids": primary["support_token_ids"][:used_k],
                "primary_decoded_tokens": primary["decoded_tokens"][:used_k],
            }
        )
    for seed in random_seeds:
        condition = f"random_{int(seed)}"
        directions[condition] = {}
        for layer, source in directions["full"].items():
            unit = source / np.linalg.norm(source)
            rng = np.random.Generator(np.random.Philox([int(seed), int(layer)]))
            random = rng.normal(size=source.size)
            random -= float(random @ unit) * unit
            directions[condition][layer] = random / np.linalg.norm(random)
    return directions, {
        "selected_layers": list(layers),
        "selected_layers_path": str(selected_layers_path),
        "selected_layers_sha256": sha256_file(selected_layers_path),
        "row_manifest": str(row_manifest_path),
        "row_manifest_sha256": row_manifest_sha256,
        "layer_k": provenance,
    }


def _mean_residual_norms(
    activation_dir: str | Path,
    *,
    layers: Sequence[int],
    concept_id: str,
    selected_layers_path: str | Path,
) -> dict[int, float]:
    selection_path = Path(selected_layers_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    row_manifest = selection_path.parent / str(selection["row_manifest"])
    fit_indices = np.concatenate(
        [
            load_balanced_indices(row_manifest, concept_id, split)
            for split in ("train", "validation")
        ]
    )
    output: dict[int, float] = {}
    for layer in layers:
        values = np.load(
            Path(activation_dir) / f"layer_{layer:02d}.npy", mmap_mode="r"
        )
        norms = np.linalg.norm(
            np.asarray(values[fit_indices], dtype=np.float64), axis=1
        )
        mean = float(np.mean(norms))
        if not math.isfinite(mean) or mean <= 0:
            raise MultiLayerJError(f"invalid mean residual norm at layer {layer}")
        output[layer] = mean
    return output


def _score(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[PromptRecord],
    candidate_ids: Mapping[str, int],
    concept_id: str,
    directions: Mapping[int, np.ndarray],
    residual_norms: Mapping[int, float],
    strength: float,
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch

    device = model_input_device(model)
    concepts = tuple(candidate_ids)
    tokens = torch.as_tensor(
        [candidate_ids[value] for value in concepts],
        dtype=torch.long,
        device=device,
    )
    rows: list[dict[str, Any]] = []
    old_padding = tokenizer.padding_side
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
            with torch.inference_mode(), multilayer_intervention_session(
                model,
                directions=dict(directions),
                strength=strength,
                residual_norms=dict(residual_norms),
                position="last_prompt_and_generated",
            ):
                output = model(**encoded, use_cache=False, logits_to_keep=1)
            logits = output.logits[:, -1].float()
            selected = torch.log_softmax(logits, dim=-1)[:, tokens].cpu().numpy()
            normalized = torch.softmax(logits[:, tokens], dim=-1).cpu().numpy()
            for row_index, prompt in enumerate(prompt_batch):
                log_map = {
                    value: float(selected[row_index, column])
                    for column, value in enumerate(concepts)
                }
                probability_map = {
                    value: float(normalized[row_index, column])
                    for column, value in enumerate(concepts)
                }
                target = log_map[concept_id]
                off_target = [
                    value for key, value in log_map.items() if key != concept_id
                ]
                rows.append(
                    {
                        "prompt_id": prompt.prompt_id,
                        "evaluation_split": (
                            "validation"
                            if prompt.prompt_id.partition("_")[0]
                            in {"choose", "complete"}
                            else "test"
                        ),
                        "target_concept_id": concept_id,
                        "strength": float(strength),
                        "candidate_log_probabilities": log_map,
                        "candidate_probabilities_normalized": probability_map,
                        "target_log_probability": target,
                        "target_candidate_probability": probability_map[concept_id],
                        "target_margin": target - float(np.mean(off_target)),
                        "target_rank": 1 + sum(value > target for value in off_target),
                    }
                )
    finally:
        tokenizer.padding_side = old_padding
    return rows


def _settings(value: Any) -> GenerationSettings:
    return GenerationSettings(
        sample_seeds=tuple(value.sample_seeds),
        max_new_tokens=value.max_new_tokens,
        temperature=value.temperature,
        top_p=value.top_p,
        repetition_penalty=value.repetition_penalty,
        no_repeat_ngram_size=value.no_repeat_ngram_size,
    )


def run_multilayer_j_intervention(
    *,
    output_dir: str | Path,
    model: Any,
    tokenizer: Any,
    config: Any,
    concept_id: str,
    grid_index: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run every J/full/non-J/random strength and decoding for one concept."""

    target_root = Path(output_dir) / "targets" / quote(concept_id, safe="")
    condition_ids = [
        "full",
        "j",
        "non_j",
        *[f"random_{int(seed)}" for seed in config.random_control_seeds],
    ]
    grid = [
        (condition_id, float(strength))
        for condition_id in condition_ids
        for strength in config.strengths
    ]
    if grid_index is not None and not 0 <= grid_index < len(grid):
        raise MultiLayerJError(
            f"J grid_index must lie in [0, {len(grid)}), got {grid_index}"
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
        raise FileExistsError(f"incomplete J target exists: {destination}")
    directions, provenance = load_multilayer_directions(
        occupancy_dir=config.source_occupancy_dir,
        probes_dir=config.source_probes_dir,
        selected_layers_path=config.selected_layers_path,
        concept_id=concept_id,
        replicate_ids=config.probe_replicates,
        k_max=config.k_max,
        random_seeds=config.random_control_seeds,
    )
    layers = tuple(provenance["selected_layers"])
    residual_norms = _mean_residual_norms(
        config.source_activations_dir,
        layers=layers,
        concept_id=concept_id,
        selected_layers_path=config.selected_layers_path,
    )
    prompts = load_prompt_bank(
        config.generation.candidate_prompts_path,
        tokenizer=tokenizer,
        candidate_labels=config.candidate_labels,
    )
    candidate_ids = candidate_token_ids(tokenizer, config.candidate_labels)
    score_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    all_prompts: list[Any] = [
        *prompts,
        *load_open_prompt_bank(
            config.generation.open_prompts_path, tokenizer=tokenizer
        ),
    ]
    settings = _settings(config.generation)
    generation_contract = build_generation_contract(
        all_prompts,
        settings,
        candidate_labels=config.candidate_labels,
    )
    for condition_id, strength in selected_grid:
        layer_directions = directions[condition_id]
        condition = (
            "random" if condition_id.startswith("random_") else condition_id
        )
        rows = _score(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            candidate_ids=candidate_ids,
            concept_id=concept_id,
            directions=layer_directions,
            residual_norms=residual_norms,
            strength=strength,
            batch_size=config.score_batch_size,
        )
        for row in rows:
            row.update(
                {
                    "method": "j_component_intervention",
                    "condition_id": condition_id,
                    "condition": condition,
                    "selected_layers": list(layers),
                }
            )
        score_rows.extend(rows)
        generation_rows.extend(
            generate_full_grid(
                model=model,
                tokenizer=tokenizer,
                prompts=all_prompts,
                method="j_component_intervention",
                concept_id=concept_id,
                condition_id=condition_id,
                grid_point={"strength": strength},
                intervention_metadata={
                    "selected_layers": list(layers),
                    "K_used_by_layer": {
                        str(row["layer"]): int(row["K_used"])
                        for row in provenance["layer_k"]
                    },
                    "direction_condition": condition,
                    "residual_norms": {
                        str(layer): float(residual_norms[layer])
                        for layer in layers
                    },
                },
                session_factory=lambda d=layer_directions, s=strength: (
                    multilayer_intervention_session(
                        model,
                        directions=dict(d),
                        strength=s,
                        residual_norms=residual_norms,
                        position="last_prompt_and_generated",
                    )
                ),
                settings=settings,
            )
        )
    zero_by_prompt: dict[str, list[float]] = defaultdict(list)
    for row in score_rows:
        if float(row["strength"]) == 0.0:
            zero_by_prompt[str(row["prompt_id"])].append(
                float(row["target_log_probability"])
            )
    zero_spread = (
        max(max(values) - min(values) for values in zero_by_prompt.values())
        if zero_by_prompt
        else None
    )
    if zero_spread is not None and zero_spread > 1e-5:
        raise MultiLayerJError(f"zero-strength conditions differ by {zero_spread}")
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(destination / "candidate_scores.jsonl", score_rows)
    generation_files = write_generation_artifacts(destination, generation_rows)
    summary_payload = {
        "schema_version": 1,
        "method": "j_component_intervention",
        "target_concept_id": concept_id,
        "selected_layers": list(layers),
        "coordinate": "resid_post",
        "position": "last_token_each_forward_call",
        "normalization": (
            "per-layer unit direction times mean development residual norm; "
            "full strength at every layer; no sqrt(layer_count) division"
        ),
        "mean_residual_norms": {
            str(layer): value for layer, value in residual_norms.items()
        },
        "strengths": list(config.strengths),
        "grid_index": grid_index,
        "grid_size": len(grid),
        "grid_condition": (
            None
            if grid_index is None
            else {
                "condition_id": selected_grid[0][0],
                "strength": selected_grid[0][1],
            }
        ),
        "source_provenance": provenance,
        "zero_strength_max_logprob_spread": zero_spread,
        "candidate_score_rows": len(score_rows),
        "candidate_scores_sha256": sha256_file(
            destination / "candidate_scores.jsonl"
        ),
        "generation_rows": len(generation_rows),
        "generation_files": generation_files,
        "generation_contract": generation_contract,
        "llm_as_judge_run": False,
    }
    atomic_write_json(destination / "summary.json", summary_payload)
    return {"status": "completed", **summary_payload}


def rebuild_multilayer_j_index(
    output_dir: str | Path,
    *,
    concept_ids: Sequence[str],
    strengths: Sequence[float] | None = None,
    random_control_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    entries = []
    observed = set()
    expected_grid_size = (
        None
        if strengths is None or random_control_seeds is None
        else (3 + len(random_control_seeds)) * len(strengths)
    )
    expected_grid = (
        None
        if strengths is None or random_control_seeds is None
        else [
            {
                "condition_id": condition_id,
                "strength": float(strength),
            }
            for condition_id in (
                "full",
                "j",
                "non_j",
                *[
                    f"random_{int(seed)}"
                    for seed in random_control_seeds
                ],
            )
            for strength in strengths
        ]
    )
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise MultiLayerJError("J method manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generation_identity = manifest.get("notes", {}).get("generation")
    if not isinstance(generation_identity, Mapping):
        raise MultiLayerJError("J manifest lacks generation identity")
    for concept_id in concept_ids:
        target = root / "targets" / quote(concept_id, safe="")
        path = target / "summary.json"
        if path.is_file() and expected_grid is None:
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
                grid_index = int(shard_payload.get("grid_index", -1))
                if (
                    expected_grid is None
                    or not 0 <= grid_index < len(expected_grid)
                    or shard_payload.get("grid_condition")
                    != expected_grid[grid_index]
                    or shard_payload.get("method")
                    != "j_component_intervention"
                    or shard_payload.get("target_concept_id") != concept_id
                ):
                    raise MultiLayerJError(
                        f"J shard scientific grid identity mismatch: {shard}"
                    )
                shard_manifest_path = (
                    root
                    / "manifests"
                    / quote(concept_id, safe="")
                    / f"grid_{grid_index:04d}.json"
                )
                if not shard_manifest_path.is_file():
                    raise MultiLayerJError(
                        f"J shard manifest is missing: {shard_manifest_path}"
                    )
                try:
                    validate_shard_manifest_identity(
                        manifest,
                        json.loads(
                            shard_manifest_path.read_text(encoding="utf-8")
                        ),
                    )
                except InterventionGenerationError as error:
                    raise MultiLayerJError(
                        f"J shard manifest identity mismatch: {shard_manifest_path}"
                    ) from error
                candidate_path = shard.parent / "candidate_scores.jsonl"
                try:
                    candidate_check = validate_candidate_score_artifact(
                        candidate_path,
                        expected_sha256=str(
                            shard_payload.get("candidate_scores_sha256", "")
                        ),
                        contract=shard_payload["generation_contract"],
                        expected_method="j_component_intervention",
                        expected_concept_id=concept_id,
                        expected_grid_condition=expected_grid[grid_index],
                    )
                    generation_check = validate_generation_artifacts(
                        shard.parent,
                        shard_payload["generation_files"],
                        contract=shard_payload["generation_contract"],
                        expected_selected_layers=shard_payload[
                            "selected_layers"
                        ],
                    )
                except InterventionGenerationError as error:
                    raise MultiLayerJError(
                        f"J shard output contract failed: {shard}"
                    ) from error
                if (
                    int(shard_payload.get("candidate_score_rows", -1))
                    != candidate_check["rows"]
                    or int(shard_payload.get("generation_rows", -1))
                    != generation_check["rows"]
                ):
                    raise MultiLayerJError(
                        f"J shard declared row counts differ: {shard}"
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
            contract_identities = {
                json.dumps(
                    row["generation_contract"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for row in shard_payloads
            }
            provenance_identities = {
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "selected_layers",
                            "coordinate",
                            "position",
                            "normalization",
                            "mean_residual_norms",
                            "source_provenance",
                        )
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for row in shard_payloads
            }
            if len(contract_identities) != 1:
                raise MultiLayerJError(
                    f"{concept_id}: J shards used different generation contracts"
                )
            if len(provenance_identities) != 1:
                raise MultiLayerJError(
                    f"{concept_id}: J shards used different provenance"
                )
            try:
                validate_generation_contract_identity(
                    shard_payloads[0]["generation_contract"],
                    generation_identity,
                )
            except InterventionGenerationError as error:
                raise MultiLayerJError(
                    f"{concept_id}: J shard contract differs from manifest"
                ) from error
            layers = {
                tuple(int(value) for value in row["selected_layers"])
                for row in shard_payloads
            }
            if len(layers) != 1:
                raise MultiLayerJError(
                    f"{concept_id}: J shards used different selected layers"
                )
            zero_by_prompt: dict[str, list[float]] = defaultdict(list)
            zero_generation_paths: list[Path] = []
            for shard, shard_payload in zip(
                shard_paths, shard_payloads, strict=True
            ):
                if float(shard_payload["grid_condition"]["strength"]) != 0.0:
                    continue
                zero_generation_paths.append(shard.parent / "generations.jsonl")
                with (shard.parent / "candidate_scores.jsonl").open(
                    encoding="utf-8"
                ) as handle:
                    for line in handle:
                        row = json.loads(line)
                        zero_by_prompt[str(row["prompt_id"])].append(
                            float(row["target_log_probability"])
                        )
            zero_spread = max(
                max(values) - min(values) for values in zero_by_prompt.values()
            )
            if zero_spread > 1e-5:
                raise MultiLayerJError(
                    f"{concept_id}: zero-strength J shards differ by {zero_spread}"
                )
            zero_generation_consistency = validate_equivalent_generation_outputs(
                zero_generation_paths
            )
            first = shard_payloads[0]
            payload = {
                "schema_version": 1,
                "method": "j_component_intervention",
                "target_concept_id": concept_id,
                "selected_layers": list(next(iter(layers))),
                "coordinate": first["coordinate"],
                "position": first["position"],
                "normalization": first["normalization"],
                "mean_residual_norms": first["mean_residual_norms"],
                "strengths": list(strengths),
                "source_provenance": first["source_provenance"],
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
        if payload.get("method") != "j_component_intervention":
            raise MultiLayerJError(f"unsupported J summary: {path}")
        if str(payload["target_concept_id"]) != concept_id:
            raise MultiLayerJError(f"J concept identity mismatch: {path}")
        observed.add(concept_id)
        entries.append(
            {
                "concept_id": concept_id,
                "summary": str(path.relative_to(root)),
                "summary_sha256": sha256_file(path),
                "selected_layers": payload["selected_layers"],
                "layer_k": payload["source_provenance"]["layer_k"],
                "artifact_seal": build_target_artifact_seal(root, path),
            }
        )
    expected = set(concept_ids)
    index = {
        "schema_version": 1,
        "method": "j_component_intervention",
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
