"""Immutable-suite same-model/data RAPTOR intervention experiment."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from jlens_workspace.activations import _capture_forward_kwargs
from jlens_workspace.artifacts import atomic_write_json, sha256_file
from jlens_workspace.concept_intervention.evaluation import (
    PromptRecord,
    atomic_write_jsonl,
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
    validate_generation_artifacts,
    validate_generation_contract_identity,
    validate_shard_manifest_identity,
    write_generation_artifacts,
)
from jlens_workspace.concept_intervention.raptor.intervention import (
    RAPTOR_COMMIT,
    RAPTOR_REPOSITORY,
    RaptorError,
    no_raptor_intervention,
    raptor_intervention_session,
    verify_raptor_checkout,
)
from jlens_workspace.concept_intervention.shared_protocol import load_selected_layers
from jlens_workspace.modeling import model_input_device


def _probe_path(root: Path, layer: int, concept_id: str) -> Path:
    return (
        root
        / f"layer_{layer:02d}"
        / f"concept_{quote(concept_id, safe='')}"
        / "probe_vector.npy"
    )


def load_raptor_directions(
    *,
    probes_dir: str | Path,
    selected_layers_path: str | Path,
    concept_id: str,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    layers = load_selected_layers(selected_layers_path, concept_id)
    root = Path(probes_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_layers_sha256 = sha256_file(selected_layers_path)
    if (
        manifest.get("workflow") != "shared_raptor_probe_fitting"
        or manifest.get("layer_selection_sha256") != selected_layers_sha256
    ):
        raise RaptorError(
            f"RAPTOR probe manifest does not match layer selection: {manifest_path}"
        )
    row_manifest_path = (root / str(manifest["row_manifest"])).resolve()
    row_manifest_sha256 = sha256_file(row_manifest_path)
    if manifest.get("row_manifest_sha256") != row_manifest_sha256:
        raise RaptorError(
            f"RAPTOR probe row-manifest identity mismatch: {row_manifest_path}"
        )
    authoritative: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in manifest.get("probes", []):
        identity = (int(row["layer"]), str(row["concept_id"]))
        if identity in authoritative:
            raise RaptorError(f"duplicate RAPTOR probe identity: {identity}")
        authoritative[identity] = row
    directions: dict[int, np.ndarray] = {}
    entries = []
    for layer in layers:
        path = _probe_path(root, layer, concept_id)
        registered = authoritative.get((layer, concept_id))
        if registered is None:
            raise RaptorError(
                f"RAPTOR probe is absent from shared manifest: {layer}/{concept_id}"
            )
        registered_path = root / str(registered["vector_file"])
        if registered_path.resolve() != path.resolve():
            raise RaptorError(
                f"RAPTOR probe path differs from shared manifest: {path}"
            )
        observed_hash = sha256_file(path)
        if registered.get("vector_sha256") != observed_hash:
            raise RaptorError(
                f"RAPTOR probe hash differs from shared manifest: {path}"
            )
        vector = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if vector.ndim != 1 or not np.isfinite(vector).all() or norm <= 0:
            raise RaptorError(f"invalid RAPTOR probe vector: {path}")
        directions[layer] = vector / norm
        entries.append(
            {
                "layer": layer,
                "probe_vector": str(path),
                "probe_vector_sha256": observed_hash,
                "source_norm": norm,
                "steering_norm": 1.0,
            }
        )
    return directions, {
        "selected_layers": list(layers),
        "selected_layers_path": str(selected_layers_path),
        "selected_layers_sha256": selected_layers_sha256,
        "row_manifest": str(row_manifest_path),
        "row_manifest_sha256": row_manifest_sha256,
        "probe_manifest": str(manifest_path),
        "probe_manifest_sha256": sha256_file(manifest_path),
        "directions": entries,
        "bias": 0.0,
        "bias_provenance": (
            "pinned singlelr loader discards stored per-layer intercepts; "
            "author steering CLI uses global default bias=0.0"
        ),
        "test_accuracy_filter": False,
    }


def _score(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[PromptRecord],
    candidate_ids: Mapping[str, int],
    concept_id: str,
    directions: Mapping[int, np.ndarray],
    upstream_checkout: str | Path,
    target_probability: float | None,
) -> list[dict[str, Any]]:
    import torch

    device = model_input_device(model)
    concepts = tuple(candidate_ids)
    token_ids = torch.as_tensor(
        [candidate_ids[value] for value in concepts],
        dtype=torch.long,
        device=device,
    )
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    rows: list[dict[str, Any]] = []
    try:
        for prompt in prompts:
            encoded = tokenizer(
                prompt.formatted_text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            session = (
                no_raptor_intervention()
                if target_probability is None
                else raptor_intervention_session(
                    model,
                    directions=directions,
                    target_probability=target_probability,
                    upstream_checkout=upstream_checkout,
                )
            )
            with torch.inference_mode(), session as state:
                output = model(**encoded, **_capture_forward_kwargs(model))
            logits = output.logits[:, -1].float()
            selected = torch.log_softmax(logits, dim=-1)[:, token_ids].cpu().numpy()
            normalized = torch.softmax(logits[:, token_ids], dim=-1).cpu().numpy()
            log_map = {
                value: float(selected[0, column])
                for column, value in enumerate(concepts)
            }
            probability_map = {
                value: float(normalized[0, column])
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
                    "target_probability": target_probability,
                    "candidate_log_probabilities": log_map,
                    "candidate_probabilities_normalized": probability_map,
                    "target_log_probability": target,
                    "target_candidate_probability": probability_map[concept_id],
                    "target_margin": target - float(np.mean(off_target)),
                    "target_rank": 1 + sum(value > target for value in off_target),
                    "telemetry": state.events,
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


def run_raptor_intervention(
    *,
    output_dir: str | Path,
    model: Any,
    tokenizer: Any,
    config: Any,
    concept_id: str,
    grid_index: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run no-hook plus every adaptive target probability and decoding."""

    checkout = verify_raptor_checkout(config.upstream_checkout)
    target_root = Path(output_dir) / "targets" / quote(concept_id, safe="")
    conditions: list[tuple[str, float | None]] = [
        ("no_hook", None),
        *[
            (f"target_probability_{value:g}", float(value))
            for value in config.target_probabilities
        ],
    ]
    if grid_index is not None and not 0 <= grid_index < len(conditions):
        raise RaptorError(
            f"RAPTOR grid_index must lie in [0, {len(conditions)}), got {grid_index}"
        )
    selected_conditions = (
        conditions if grid_index is None else [conditions[grid_index]]
    )
    destination = (
        target_root
        if grid_index is None
        else target_root / "shards" / f"grid_{grid_index:04d}"
    )
    if destination.exists() and not overwrite:
        summary = destination / "summary.json"
        if summary.is_file():
            return {"status": "already_complete", "summary": str(summary)}
        raise FileExistsError(f"incomplete RAPTOR target exists: {destination}")
    directions, provenance = load_raptor_directions(
        probes_dir=config.source_probes_dir,
        selected_layers_path=config.selected_layers_path,
        concept_id=concept_id,
    )
    candidate_prompts = load_prompt_bank(
        config.generation.candidate_prompts_path,
        tokenizer=tokenizer,
        candidate_labels=config.candidate_labels,
    )
    all_prompts: list[Any] = [
        *candidate_prompts,
        *load_open_prompt_bank(
            config.generation.open_prompts_path, tokenizer=tokenizer
        ),
    ]
    candidate_ids = candidate_token_ids(tokenizer, config.candidate_labels)
    settings = _settings(config.generation)
    generation_contract = build_generation_contract(all_prompts, settings)
    scores: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []

    for condition_id, probability in selected_conditions:
        rows = _score(
            model=model,
            tokenizer=tokenizer,
            prompts=candidate_prompts,
            candidate_ids=candidate_ids,
            concept_id=concept_id,
            directions=directions,
            upstream_checkout=checkout,
            target_probability=probability,
        )
        for row in rows:
            row.update(
                {
                    "method": "raptor_intervention",
                    "condition_id": condition_id,
                    "selected_layers": provenance["selected_layers"],
                }
            )
        scores.extend(rows)
        generations.extend(
            generate_full_grid(
                model=model,
                tokenizer=tokenizer,
                prompts=all_prompts,
                method="raptor_intervention",
                concept_id=concept_id,
                condition_id=condition_id,
                grid_point={"target_probability": probability},
                intervention_metadata={
                    "selected_layers": provenance["selected_layers"],
                    "directions": provenance["directions"],
                    "true_no_hook": probability is None,
                },
                session_factory=lambda probability=probability: (
                    no_raptor_intervention()
                    if probability is None
                    else raptor_intervention_session(
                        model,
                        directions=directions,
                        target_probability=probability,
                        upstream_checkout=checkout,
                    )
                ),
                settings=settings,
            )
        )
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(destination / "candidate_scores.jsonl", scores)
    generation_files = write_generation_artifacts(destination, generations)
    summary = {
        "schema_version": 1,
        "method": "raptor_intervention",
        "target_concept_id": concept_id,
        "selected_layers": provenance["selected_layers"],
        "coordinate": "resid_post",
        "position": "last_token_each_forward_call",
        "normalization": "unit RAPTOR probe with source adaptive epsilon",
        "target_probabilities": list(config.target_probabilities),
        "grid_index": grid_index,
        "grid_size": len(conditions),
        "grid_condition": (
            None
            if grid_index is None
            else {
                "condition_id": selected_conditions[0][0],
                "target_probability": selected_conditions[0][1],
            }
        ),
        "true_no_hook_baseline": True,
        "source_provenance": provenance,
        "candidate_score_rows": len(scores),
        "candidate_scores_sha256": sha256_file(
            destination / "candidate_scores.jsonl"
        ),
        "generation_rows": len(generations),
        "generation_files": generation_files,
        "generation_contract": generation_contract,
        "upstream": {
            "repository": RAPTOR_REPOSITORY,
            "commit": RAPTOR_COMMIT,
            "checkout": str(checkout),
            "source_used_directly": [
                "register_steering_hooks_sequential",
                "compute_adaptive_epsilon",
            ],
            "steering_bias": 0.0,
            "steering_bias_provenance": provenance["bias_provenance"],
            "test_accuracy_filter": False,
        },
        "llm_as_judge_run": False,
    }
    atomic_write_json(destination / "summary.json", summary)
    return {"status": "completed", **summary}


def rebuild_raptor_index(
    output_dir: str | Path,
    *,
    concept_ids: Sequence[str],
    target_probabilities: Sequence[float] | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    observed: set[str] = set()
    entries = []
    expected_grid_size = (
        None if target_probabilities is None else 1 + len(target_probabilities)
    )
    expected_grid = (
        None
        if target_probabilities is None
        else [
            {
                "condition_id": "no_hook",
                "target_probability": None,
            },
            *[
                {
                    "condition_id": f"target_probability_{float(value):g}",
                    "target_probability": float(value),
                }
                for value in target_probabilities
            ],
        ]
    )
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RaptorError("RAPTOR method manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generation_identity = manifest.get("notes", {}).get("generation")
    if not isinstance(generation_identity, Mapping):
        raise RaptorError("RAPTOR manifest lacks generation identity")
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
                    or shard_payload.get("method") != "raptor_intervention"
                    or shard_payload.get("target_concept_id") != concept_id
                ):
                    raise RaptorError(
                        f"RAPTOR shard scientific grid identity mismatch: {shard}"
                    )
                shard_manifest_path = (
                    root
                    / "manifests"
                    / quote(concept_id, safe="")
                    / f"grid_{grid_index:04d}.json"
                )
                if not shard_manifest_path.is_file():
                    raise RaptorError(
                        f"RAPTOR shard manifest is missing: {shard_manifest_path}"
                    )
                try:
                    validate_shard_manifest_identity(
                        manifest,
                        json.loads(
                            shard_manifest_path.read_text(encoding="utf-8")
                        ),
                    )
                except InterventionGenerationError as error:
                    raise RaptorError(
                        "RAPTOR shard manifest identity mismatch: "
                        f"{shard_manifest_path}"
                    ) from error
                candidate_path = shard.parent / "candidate_scores.jsonl"
                try:
                    candidate_check = validate_candidate_score_artifact(
                        candidate_path,
                        expected_sha256=str(
                            shard_payload.get("candidate_scores_sha256", "")
                        ),
                        contract=shard_payload["generation_contract"],
                        expected_method="raptor_intervention",
                        expected_concept_id=concept_id,
                        expected_grid_condition=expected_grid[grid_index],
                    )
                    generation_check = validate_generation_artifacts(
                        shard.parent,
                        shard_payload["generation_files"],
                        contract=shard_payload["generation_contract"],
                    )
                except InterventionGenerationError as error:
                    raise RaptorError(
                        f"RAPTOR shard output contract failed: {shard}"
                    ) from error
                if (
                    int(shard_payload.get("candidate_score_rows", -1))
                    != candidate_check["rows"]
                    or int(shard_payload.get("generation_rows", -1))
                    != generation_check["rows"]
                ):
                    raise RaptorError(
                        f"RAPTOR shard declared row counts differ: {shard}"
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
                            "source_provenance",
                            "upstream",
                        )
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for row in shard_payloads
            }
            if len(contract_identities) != 1:
                raise RaptorError(
                    f"{concept_id}: RAPTOR shards used different generation contracts"
                )
            if len(provenance_identities) != 1:
                raise RaptorError(
                    f"{concept_id}: RAPTOR shards used different provenance"
                )
            try:
                validate_generation_contract_identity(
                    shard_payloads[0]["generation_contract"],
                    generation_identity,
                )
            except InterventionGenerationError as error:
                raise RaptorError(
                    f"{concept_id}: RAPTOR shard contract differs from manifest"
                ) from error
            layers = {
                tuple(int(value) for value in row["selected_layers"])
                for row in shard_payloads
            }
            if len(layers) != 1:
                raise RaptorError(
                    f"{concept_id}: RAPTOR shards used different selected layers"
                )
            first = shard_payloads[0]
            payload = {
                "schema_version": 1,
                "method": "raptor_intervention",
                "target_concept_id": concept_id,
                "selected_layers": list(next(iter(layers))),
                "coordinate": first["coordinate"],
                "position": first["position"],
                "normalization": first["normalization"],
                "target_probabilities": list(target_probabilities),
                "true_no_hook_baseline": True,
                "source_provenance": first["source_provenance"],
                "upstream": first["upstream"],
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
        if payload.get("method") != "raptor_intervention":
            raise RaptorError(f"unsupported RAPTOR summary: {path}")
        if str(payload["target_concept_id"]) != concept_id:
            raise RaptorError(f"RAPTOR concept identity mismatch: {path}")
        observed.add(concept_id)
        entries.append(
            {
                "concept_id": concept_id,
                "summary": str(path.relative_to(root)),
                "summary_sha256": sha256_file(path),
                "selected_layers": payload["selected_layers"],
                "artifact_seal": build_target_artifact_seal(root, path),
            }
        )
    expected = set(concept_ids)
    index = {
        "schema_version": 1,
        "method": "raptor_intervention",
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
