"""Method-neutral model execution and artifact writes for steering runs.

The immutable schemas and validators live upstream in
``concept_intervention.protocol.generation``.  This module owns the operations
that execute a model or mutate a steering run directory.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from jlens_workspace.concept_intervention.protocol.contracts import (
    PromptRecord,
    atomic_write_jsonl,
)
from jlens_workspace.concept_intervention.protocol.generation import (
    _INDEX_BUILDER_FIELDS,
    _SCIENTIFIC_MANIFEST_TOP_LEVEL_FIELDS,
    GenerationSettings,
    InterventionGenerationError,
    OpenPromptRecord,
    _canonical_sha256,
    _identifier,
    _manifest_payload,
    _validate_expected_manifest_identity,
    _validate_index_builder,
    build_generation_contract,
    build_target_artifact_seal,
    candidate_prompt_splits,
    load_open_prompt_bank,
    prompt_ids_sha256,
    scientific_shard_identity,
    validate_candidate_score_artifact,
    validate_equivalent_generation_outputs,
    validate_generation_artifacts,
    validate_generation_contract_identity,
    validate_index_builder_provenance,
    validate_method_index_provenance,
    validate_shard_manifest_identity,
    validate_target_artifact_seal,
)
from jlens_workspace.foundation.artifacts import atomic_write_json, sha256_file
from jlens_workspace.foundation.modeling import model_input_device


def _telemetry(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        rows: list[dict[str, Any]] = []
        for layer, item in value.items():
            events = getattr(item, "events", [])
            rows.extend({"layer": int(layer), **dict(event)} for event in events)
        return rows
    events = getattr(value, "events", [])
    output = []
    for event in events:
        if is_dataclass(event):
            output.append(asdict(event))
        else:
            output.append(dict(event))
    return output


def _generate_one(
    *,
    model: Any,
    tokenizer: Any,
    prompt: PromptRecord | OpenPromptRecord,
    session_factory: Callable[[], AbstractContextManager[Any]],
    do_sample: bool,
    seed: int | None,
    settings: GenerationSettings,
) -> tuple[list[int], str, list[float], list[dict[str, Any]]]:
    import torch

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    encoded = tokenizer(
        prompt.formatted_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = {
        key: value.to(model_input_device(model)) for key, value in encoded.items()
    }
    kwargs: dict[str, Any] = {
        "max_new_tokens": settings.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": settings.repetition_penalty,
        "no_repeat_ngram_size": settings.no_repeat_ngram_size,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if do_sample:
        kwargs.update(temperature=settings.temperature, top_p=settings.top_p)
    with torch.inference_mode(), session_factory() as state:
        generated = model.generate(**encoded, **kwargs)
    prompt_length = int(encoded["input_ids"].shape[1])
    token_tensor = generated.sequences[0, prompt_length:]
    token_ids = [int(value) for value in token_tensor.detach().cpu().tolist()]
    log_probabilities: list[float] = []
    for step, scores in enumerate(generated.scores):
        log_probs = torch.log_softmax(scores[0].float(), dim=-1)
        log_probabilities.append(float(log_probs[token_tensor[step]].cpu()))
    return (
        token_ids,
        tokenizer.decode(token_ids, skip_special_tokens=True),
        log_probabilities,
        _telemetry(state),
    )


def generate_full_grid(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[PromptRecord | OpenPromptRecord],
    method: str,
    concept_id: str,
    condition_id: str,
    grid_point: Mapping[str, Any],
    intervention_metadata: Mapping[str, Any] | None = None,
    session_factory: Callable[[], AbstractContextManager[Any]],
    settings: GenerationSettings,
) -> list[dict[str, Any]]:
    """Generate greedy plus all fixed-seed samples for one complete grid point."""

    rows: list[dict[str, Any]] = []
    decodings = [
        ("greedy", None),
        *[("sample", seed) for seed in settings.sample_seeds],
    ]
    for prompt in prompts:
        for decoding, seed in decodings:
            identity = {
                "method": method,
                "concept_id": concept_id,
                "condition_id": condition_id,
                "grid_point": dict(grid_point),
                "intervention_metadata": dict(intervention_metadata or {}),
                "prompt_id": prompt.prompt_id,
                "decoding": decoding,
                "seed": seed,
            }
            token_ids, text, token_logprobs, telemetry = _generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                session_factory=session_factory,
                do_sample=(decoding == "sample"),
                seed=seed,
                settings=settings,
            )
            generation_id = _identifier(identity, prefix="generation")
            injected_by_layer: dict[str, float] = {}
            for event in telemetry:
                norm = float(event.get("injected_norm", 0.0))
                layer = str(event.get("layer", "unscoped"))
                injected_by_layer[layer] = injected_by_layer.get(layer, 0.0) + norm
            rows.append(
                {
                    "generation_id": generation_id,
                    "blind_id": _identifier(
                        {"generation_id": generation_id}, prefix="blind"
                    ),
                    **identity,
                    "prompt_split": (
                        prompt.split
                        if isinstance(prompt, OpenPromptRecord)
                        else (
                            "validation"
                            if prompt.prompt_id.partition("_")[0]
                            in {"choose", "complete"}
                            else "test"
                        )
                    ),
                    "prompt_text": prompt.raw_text,
                    "generated_token_ids": token_ids,
                    "generated_text": text,
                    "token_log_probabilities": token_logprobs,
                    "mean_token_log_probability": (
                        sum(token_logprobs) / len(token_logprobs)
                        if token_logprobs
                        else None
                    ),
                    "telemetry": telemetry,
                    "injected_norm_by_layer": injected_by_layer,
                    "total_injected_norm": float(sum(injected_by_layer.values())),
                    "generation_settings": asdict(settings),
                }
            )
    return rows


def write_generation_artifacts(
    output_dir: str | Path, rows: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    """Write full and method-blind generation artifacts."""

    destination = Path(output_dir)
    full_path = destination / "generations.jsonl"
    blind_path = destination / "judge_blind_generations.jsonl"
    map_path = destination / "judge_blind_map.jsonl"
    atomic_write_jsonl(full_path, rows)
    blind_rows = [
        {
            "blind_id": row["blind_id"],
            "prompt_id": row["prompt_id"],
            "prompt_text": row["prompt_text"],
            "decoding": row["decoding"],
            "generated_text": row["generated_text"],
        }
        for row in rows
    ]
    mapping = [
        {
            "blind_id": row["blind_id"],
            "generation_id": row["generation_id"],
            "method": row["method"],
            "concept_id": row["concept_id"],
            "condition_id": row["condition_id"],
            "grid_point": row["grid_point"],
        }
        for row in rows
    ]
    atomic_write_jsonl(blind_path, blind_rows)
    atomic_write_jsonl(map_path, mapping)
    return {
        "generations": str(full_path),
        "generations_sha256": sha256_file(full_path),
        "blind_generations": str(blind_path),
        "blind_generations_sha256": sha256_file(blind_path),
        "blind_map": str(map_path),
        "blind_map_sha256": sha256_file(map_path),
    }


def build_method_index_provenance(
    root: str | Path,
    manifest_paths: Sequence[str | Path],
    *,
    expected_manifest: Mapping[str, Any] | Any | None = None,
    index_builder: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate shard identities and write an auditable platform registry."""

    method_root = Path(root).resolve()
    paths = sorted({Path(path).resolve() for path in manifest_paths})
    if not paths:
        raise InterventionGenerationError(
            "method index has no shard manifests to validate"
        )
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        if not path.is_relative_to(method_root) or not path.is_file():
            raise InterventionGenerationError(
                f"shard manifest registry path is invalid: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise InterventionGenerationError(
                f"shard manifest must contain a JSON object: {path}"
            )
        payloads.append((path, payload))
    reference = payloads[0][1]
    for path, payload in payloads:
        try:
            validate_shard_manifest_identity(reference, payload)
        except InterventionGenerationError as error:
            raise InterventionGenerationError(
                f"scientific shard manifest identity mismatch: {path}"
            ) from error

    existing_manifest_path = method_root / "manifest.json"
    if expected_manifest is None:
        if not existing_manifest_path.is_file():
            raise InterventionGenerationError("method manifest is missing")
        base_manifest = json.loads(
            existing_manifest_path.read_text(encoding="utf-8")
        )
    else:
        base_manifest = _manifest_payload(expected_manifest)
        _validate_expected_manifest_identity(base_manifest, reference)

    builder_source = index_builder
    if builder_source is None:
        nested_builder = base_manifest.get("index_builder")
        builder_source = (
            nested_builder
            if isinstance(nested_builder, Mapping)
            else {
                field: base_manifest.get(field)
                for field in _INDEX_BUILDER_FIELDS
            }
        )
    builder = _validate_index_builder(builder_source)
    scientific_identity = scientific_shard_identity(reference)
    scientific_identity_sha256 = _canonical_sha256(scientific_identity)

    entries = [
        {
            "path": path.relative_to(method_root).as_posix(),
            "manifest_sha256": sha256_file(path),
            "platform": payload["platform"],
        }
        for path, payload in payloads
    ]
    counts: dict[str, int] = {}
    for entry in entries:
        platform_value = str(entry["platform"])
        counts[platform_value] = counts.get(platform_value, 0) + 1
    platforms = [
        {"platform": platform_value, "shard_count": counts[platform_value]}
        for platform_value in sorted(counts)
    ]
    registry = {
        "schema_version": 1,
        "scientific_shard_identity": scientific_identity,
        "scientific_shard_identity_sha256": scientific_identity_sha256,
        "shard_count": len(entries),
        "platforms": platforms,
        "manifests": entries,
    }
    registry_path = method_root / "shard_manifest_provenance.json"
    atomic_write_json(registry_path, registry)
    registry_summary = {
        "path": registry_path.relative_to(method_root).as_posix(),
        "sha256": sha256_file(registry_path),
        "shard_count": len(entries),
        "platforms": platforms,
    }

    method_manifest = dict(base_manifest)
    for field in _SCIENTIFIC_MANIFEST_TOP_LEVEL_FIELDS:
        method_manifest[field] = reference[field]
    method_manifest["platform"] = reference["platform"]
    method_manifest["scientific_shard_identity"] = scientific_identity
    method_manifest["scientific_shard_identity_sha256"] = (
        scientific_identity_sha256
    )
    method_manifest["shard_manifest_provenance"] = registry_summary
    method_manifest["index_builder"] = builder
    atomic_write_json(existing_manifest_path, method_manifest)
    return {
        "scientific_shard_identity": scientific_identity,
        "scientific_shard_identity_sha256": scientific_identity_sha256,
        "shard_manifest_provenance": registry_summary,
        "index_builder": builder,
    }


__all__ = [
    "GenerationSettings",
    "InterventionGenerationError",
    "OpenPromptRecord",
    "PromptRecord",
    "build_generation_contract",
    "build_method_index_provenance",
    "build_target_artifact_seal",
    "candidate_prompt_splits",
    "generate_full_grid",
    "load_open_prompt_bank",
    "prompt_ids_sha256",
    "scientific_shard_identity",
    "validate_candidate_score_artifact",
    "validate_equivalent_generation_outputs",
    "validate_generation_artifacts",
    "validate_generation_contract_identity",
    "validate_index_builder_provenance",
    "validate_method_index_provenance",
    "validate_shard_manifest_identity",
    "validate_target_artifact_seal",
    "write_generation_artifacts",
]
