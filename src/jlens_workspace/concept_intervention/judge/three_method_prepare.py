"""Prepare and seal method-blind tasks for J-component, RAPTOR, and ITI."""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from jlens_workspace.artifacts import sha256_file
from jlens_workspace.concept_intervention.generation import (
    validate_equivalent_generation_outputs,
    validate_generation_artifacts,
)
from jlens_workspace.concept_intervention.judge.prompts import (
    PROMPT_VERSION,
    rubric_hash,
)
from jlens_workspace.concept_intervention.judge.three_method_config import (
    JudgeEvaluationConfig,
    load_judge_config,
)
from jlens_workspace.concept_intervention.judge.workflow import (
    JudgeWorkflowError,
    _canonical_hash,
    _read_jsonl,
    _registered_resource,
    _registered_write_json,
    _registered_write_jsonl,
)

METHODS = (
    "j_component_intervention",
    "raptor_intervention",
    "iti_intervention",
)
PRIMARY_ROLES = ("common_baseline", "j_component", "raptor", "iti_native")
SECONDARY_ROLES = (
    "j_full",
    "j_non_j",
    "j_matched_random",
    "iti_layer_matched",
    "iti_probe_weight",
    "iti_matched_random",
)
METHOD_PAIRS = (
    ("j_component", "raptor"),
    ("j_component", "iti_native"),
    ("raptor", "iti_native"),
)


@dataclass(frozen=True)
class MethodShard:
    """One sealed generation shard selected for a public judge role."""

    method: str
    role: str
    summary_path: Path
    summary: dict[str, Any]

    @property
    def condition(self) -> Mapping[str, Any]:
        return self.summary["grid_condition"]

    @property
    def generation_path(self) -> Path:
        return self.summary_path.parent / "generations.jsonl"


@dataclass(frozen=True)
class MethodRegistry:
    """Hash-verified method and rescore indexes for one method."""

    method: str
    root: Path
    index_path: Path
    index: dict[str, Any]
    entries: dict[str, dict[str, Any]]
    sealed_files: dict[str, dict[Path, str]]
    rescore_root: Path
    rescore_index_path: Path
    rescore_index: dict[str, Any]
    rescore_entries: dict[str, dict[str, Any]]


def _registered_write_text(path: Path, text: str) -> None:
    """Atomically freeze a text artifact and reject later registration drift."""

    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise JudgeWorkflowError(f"registered text artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise JudgeWorkflowError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise JudgeWorkflowError(f"{label} must be a JSON object: {path}")
    return payload


def _require_inside(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=True)
    root_resolved = root.resolve(strict=True)
    if root_resolved != resolved and root_resolved not in resolved.parents:
        raise JudgeWorkflowError(f"{label} escapes registered root: {resolved}")
    return resolved


def _entry_map(payload: Mapping[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise JudgeWorkflowError(f"{label} entries must be a list")
    output: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise JudgeWorkflowError(f"{label} contains a malformed entry")
        concept_id = str(entry.get("concept_id", ""))
        if not concept_id or concept_id in output:
            raise JudgeWorkflowError(f"{label} contains duplicate/empty concept IDs")
        output[concept_id] = entry
    return output


def _seal_map(
    entry: Mapping[str, Any],
    *,
    root: Path,
) -> dict[Path, str]:
    seal = entry.get("artifact_seal")
    files = seal.get("files") if isinstance(seal, Mapping) else None
    if not isinstance(files, list) or not files:
        raise JudgeWorkflowError("method index entry lacks a non-empty artifact seal")
    output: dict[Path, str] = {}
    for row in files:
        if not isinstance(row, Mapping):
            raise JudgeWorkflowError("malformed method artifact seal row")
        path = _require_inside(root / str(row.get("path", "")), root, label="sealed file")
        digest = str(row.get("sha256", ""))
        if path in output or digest != sha256_file(path):
            raise JudgeWorkflowError(f"method artifact seal mismatch: {path}")
        output[path] = digest
    return output


def _load_source_registry(
    config: JudgeEvaluationConfig,
) -> tuple[dict[str, MethodRegistry], dict[str, Any]]:
    comparison_index_path = Path(config.source.comparison_index)
    if (
        not comparison_index_path.is_file()
        or sha256_file(comparison_index_path) != config.source.comparison_index_sha256
    ):
        raise JudgeWorkflowError("comparison index hash differs from registration")
    outer = _load_json(comparison_index_path, label="comparison index")
    if outer.get("complete") is not True:
        raise JudgeWorkflowError("three-method comparison is not complete")
    comparison_path = comparison_index_path.parent / str(outer.get("comparison", ""))
    if (
        not comparison_path.is_file()
        or outer.get("comparison_sha256") != config.source.comparison_sha256
        or sha256_file(comparison_path) != config.source.comparison_sha256
    ):
        raise JudgeWorkflowError("comparison payload hash differs from registration")
    comparison = _load_json(comparison_path, label="comparison payload")
    if (
        comparison.get("comparison") != "three_method_shared_layer_held_out_target_margin"
        or comparison.get("same_selected_layers_per_concept") is not True
        or comparison.get("llm_as_judge_run") is not False
    ):
        raise JudgeWorkflowError("comparison payload has the wrong scientific identity")
    if outer.get("concept_ids") != config.source.concept_ids:
        raise JudgeWorkflowError("comparison concept order differs from registration")
    roots = {
        "j_component_intervention": Path(config.source.j_component_root),
        "raptor_intervention": Path(config.source.raptor_root),
        "iti_intervention": Path(config.source.iti_root),
    }
    method_indexes = comparison.get("method_indexes")
    rescore_indexes = outer.get("candidate_rescore_indexes")
    if not isinstance(method_indexes, Mapping) or not isinstance(
        rescore_indexes, Mapping
    ):
        raise JudgeWorkflowError("comparison omits method or rescore indexes")
    registries: dict[str, MethodRegistry] = {}
    for method in METHODS:
        root = roots[method].resolve(strict=True)
        index_path = root / "index.json"
        declared_method = method_indexes.get(method)
        declared_rescore = rescore_indexes.get(method)
        if not isinstance(declared_method, Mapping) or not isinstance(
            declared_rescore, Mapping
        ):
            raise JudgeWorkflowError(f"comparison omits {method} index declarations")
        if sha256_file(index_path) != declared_method.get("sha256"):
            raise JudgeWorkflowError(f"{method} index hash differs from comparison")
        index = _load_json(index_path, label=f"{method} index")
        if index.get("complete") is not True or index.get("method") != method:
            raise JudgeWorkflowError(f"{method} index is incomplete or mislabeled")
        entries = _entry_map(index, label=f"{method} index")
        if list(entries) != config.source.concept_ids:
            raise JudgeWorkflowError(f"{method} concept order differs")
        sealed_files = {
            concept_id: _seal_map(entries[concept_id], root=root)
            for concept_id in config.source.concept_ids
        }
        rescore_root = (
            Path(config.source.candidate_rescore_root) / method
        ).resolve(strict=True)
        rescore_index_path = rescore_root / "index.json"
        if (
            str(rescore_index_path) != str(Path(str(declared_rescore.get("path", ""))))
            or sha256_file(rescore_index_path) != declared_rescore.get("sha256")
        ):
            raise JudgeWorkflowError(f"{method} rescore index hash/path mismatch")
        rescore_manifest_path = rescore_root / "manifest.json"
        if (
            sha256_file(rescore_manifest_path)
            != declared_rescore.get("manifest_sha256")
        ):
            raise JudgeWorkflowError(f"{method} rescore manifest hash mismatch")
        rescore_index = _load_json(
            rescore_index_path, label=f"{method} rescore index"
        )
        if (
            rescore_index.get("complete") is not True
            or rescore_index.get("method") != method
            or rescore_index.get("source_method_index_sha256")
            != declared_method.get("sha256")
        ):
            raise JudgeWorkflowError(f"{method} rescore scientific identity mismatch")
        rescore_entries = _entry_map(
            rescore_index, label=f"{method} rescore index"
        )
        registries[method] = MethodRegistry(
            method=method,
            root=root,
            index_path=index_path,
            index=index,
            entries=entries,
            sealed_files=sealed_files,
            rescore_root=rescore_root,
            rescore_index_path=rescore_index_path,
            rescore_index=rescore_index,
            rescore_entries=rescore_entries,
        )
    comparison_entries = {
        str(row["concept_id"]): row for row in comparison.get("entries", [])
    }
    if list(comparison_entries) != config.source.concept_ids:
        raise JudgeWorkflowError("comparison entries differ from concept registration")
    source_record = {
        "comparison_index": str(comparison_index_path),
        "comparison_index_sha256": sha256_file(comparison_index_path),
        "comparison": str(comparison_path),
        "comparison_sha256": sha256_file(comparison_path),
        "comparison_index_builder": outer.get("index_builder"),
        "method_provenance": outer.get("method_provenance"),
        "candidate_rescore_indexes": outer.get("candidate_rescore_indexes"),
        "method_indexes": comparison.get("method_indexes"),
        "comparison_entries": comparison_entries,
    }
    return registries, source_record


def _load_rescore_rows(
    registry: MethodRegistry,
    concept_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    entry = registry.rescore_entries.get(concept_id)
    if not isinstance(entry, Mapping):
        raise JudgeWorkflowError(f"missing rescore entry: {registry.method} {concept_id}")
    summary_path = _require_inside(
        registry.rescore_root / str(entry.get("summary", "")),
        registry.rescore_root,
        label="rescore summary",
    )
    rows_path = _require_inside(
        registry.rescore_root / str(entry.get("candidate_scores", "")),
        registry.rescore_root,
        label="rescore candidate scores",
    )
    if (
        sha256_file(summary_path) != entry.get("summary_sha256")
        or sha256_file(rows_path) != entry.get("candidate_scores_sha256")
    ):
        raise JudgeWorkflowError(
            f"rescore artifact hash mismatch: {registry.method} {concept_id}"
        )
    summary = _load_json(summary_path, label="rescore summary")
    rows = _read_jsonl(rows_path)
    if (
        summary.get("artifact_kind") != "candidate_score_rescore"
        or summary.get("method") != registry.method
        or summary.get("target_concept_id") != concept_id
        or summary.get("candidate_score_rows") != len(rows)
        or len(rows) != entry.get("candidate_score_rows")
    ):
        raise JudgeWorkflowError(
            f"rescore row identity mismatch: {registry.method} {concept_id}"
        )
    return rows, summary, {
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "candidate_scores": str(rows_path),
        "candidate_scores_sha256": sha256_file(rows_path),
        "candidate_score_rows": len(rows),
    }


def _mean_validation(
    rows: Sequence[Mapping[str, Any]],
    query: Mapping[str, Any],
) -> tuple[float, int]:
    values = [
        float(row["target_margin"])
        for row in rows
        if row.get("evaluation_split") == "validation"
        and all(row.get(key) == value for key, value in query.items())
    ]
    if len(values) != 14 or not np.all(np.isfinite(values)):
        raise JudgeWorkflowError(
            f"selection query expected 14 finite validation prompts: {query}"
        )
    return float(np.mean(values)), len(values)


def _select_query(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    config: JudgeEvaluationConfig,
    rescore_summary: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if method == "j_component_intervention":
        strengths = sorted(
            {
                float(row["strength"])
                for row in rows
                if row.get("condition_id") == "j" and float(row["strength"]) > 0
            }
        )
        candidates = []
        for strength in strengths:
            query = {"condition_id": "j", "strength": strength}
            mean, count = _mean_validation(rows, query)
            candidates.append((mean, -strength, query, count))
        if not candidates:
            raise JudgeWorkflowError("J rescore lacks positive selection candidates")
        mean, _, query, count = max(candidates)
        return query, {"mean_target_margin": mean, "n_prompts": count}
    if method == "raptor_intervention":
        probabilities = sorted(
            {
                float(row["target_probability"])
                for row in rows
                if row.get("target_probability") is not None
                and float(row["target_probability"])
                > config.selection.raptor_probability_above
            }
        )
        candidates = []
        for probability in probabilities:
            query = {"target_probability": probability}
            mean, count = _mean_validation(rows, query)
            candidates.append(
                (mean, -abs(probability - 0.5), query, count)
            )
        if not candidates:
            raise JudgeWorkflowError(
                "RAPTOR rescore lacks positive selection candidates"
            )
        mean, _, query, count = max(candidates)
        return query, {"mean_target_margin": mean, "n_prompts": count}
    selection = rescore_summary.get("selection")
    if not isinstance(selection, Mapping):
        raise JudgeWorkflowError("ITI rescore summary lacks frozen selection")
    native = selection.get("native")
    layer = selection.get("layer_matched")
    if not isinstance(native, Mapping) or not isinstance(layer, Mapping):
        raise JudgeWorkflowError("ITI rescore selection lacks both variants")
    native_selected = native.get("selected")
    layer_selected = layer.get("selected")
    if not isinstance(native_selected, Mapping) or not isinstance(
        layer_selected, Mapping
    ):
        raise JudgeWorkflowError("ITI rescore selection is malformed")
    native_query = {
        "variant": "native",
        "condition_id": "iti_native_mass_mean",
        "mode": "mass_mean",
        "strength": float(native_selected["strength"]),
        "top_k": int(native_selected["top_k"]),
        "random_seed": None,
    }
    # Rescore rows preserve condition/variant/K/strength/random-seed fields but
    # omit the generation-only convenience field named mode.
    native_rescore_query = {
        key: value for key, value in native_query.items() if key != "mode"
    }
    observed, count = _mean_validation(rows, native_rescore_query)
    if not np.isclose(
        observed, float(native_selected["mean_target_margin"]), atol=1e-12
    ):
        raise JudgeWorkflowError("ITI native rescore selection does not reproduce")
    layer_query = {
        "variant": "layer_matched",
        "condition_id": "iti_layer_matched_mass_mean",
        "strength": float(layer_selected["strength"]),
        "top_k": int(layer_selected["top_k"]),
        "random_seed": None,
    }
    layer_observed, layer_count = _mean_validation(rows, layer_query)
    if not np.isclose(
        layer_observed,
        float(layer_selected["mean_target_margin"]),
        atol=1e-12,
    ):
        raise JudgeWorkflowError(
            "ITI layer-matched rescore selection does not reproduce"
        )
    return native_query, {
        "mean_target_margin": observed,
        "n_prompts": count,
        "layer_matched": {
            "strength": float(layer_selected["strength"]),
            "top_k": int(layer_selected["top_k"]),
            "mean_target_margin": layer_observed,
            "n_prompts": layer_count,
        },
    }


def _load_shards(
    registry: MethodRegistry,
    concept_id: str,
) -> list[MethodShard]:
    target = registry.root / "targets" / quote(concept_id, safe="") / "shards"
    summaries = sorted(target.glob("grid_*/summary.json"))
    if not summaries:
        raise JudgeWorkflowError(f"no generation shards: {registry.method} {concept_id}")
    output = []
    indexes: set[int] = set()
    for path in summaries:
        summary = _load_json(path, label="generation summary")
        grid_index = int(summary.get("grid_index", -1))
        if (
            summary.get("method") != registry.method
            or summary.get("target_concept_id") != concept_id
            or not isinstance(summary.get("grid_condition"), Mapping)
            or grid_index < 0
            or grid_index in indexes
        ):
            raise JudgeWorkflowError(f"invalid generation shard identity: {path}")
        indexes.add(grid_index)
        output.append(
            MethodShard(
                method=registry.method,
                role="unassigned",
                summary_path=path.resolve(),
                summary=summary,
            )
        )
    return output


def _find_shard(
    shards: Sequence[MethodShard],
    *,
    role: str,
    query: Mapping[str, Any],
) -> MethodShard:
    matches = [
        shard
        for shard in shards
        if all(shard.condition.get(key) == value for key, value in query.items())
    ]
    if len(matches) != 1:
        raise JudgeWorkflowError(
            f"{role} expected one generation shard for {query}, found {len(matches)}"
        )
    shard = matches[0]
    return MethodShard(
        method=shard.method,
        role=role,
        summary_path=shard.summary_path,
        summary=shard.summary,
    )


def _validate_shard(
    shard: MethodShard,
    *,
    registry: MethodRegistry,
    concept_id: str,
) -> dict[str, Any]:
    sealed = registry.sealed_files[concept_id]
    generation_path = shard.generation_path.resolve()
    if (
        shard.summary_path not in sealed
        or generation_path not in sealed
        or sha256_file(shard.summary_path) != sealed[shard.summary_path]
        or sha256_file(generation_path) != sealed[generation_path]
    ):
        raise JudgeWorkflowError(f"selected shard is absent from method seal: {shard.role}")
    checks = validate_generation_artifacts(
        shard.summary_path.parent,
        shard.summary["generation_files"],
        contract=shard.summary["generation_contract"],
        expected_selected_layers=shard.summary["selected_layers"],
    )
    return {
        "method": shard.method,
        "condition": dict(shard.condition),
        "summary": str(shard.summary_path),
        "summary_sha256": sha256_file(shard.summary_path),
        "generations": str(generation_path),
        "generations_sha256": sha256_file(generation_path),
        "validation": checks,
    }


def _eligible_generations(
    shard: MethodShard,
    *,
    config: JudgeEvaluationConfig,
) -> dict[str, dict[tuple[str, str, int | None], dict[str, Any]]]:
    output: dict[str, dict[tuple[str, str, int | None], dict[str, Any]]] = {
        split: {} for split in config.selection.evaluation_splits
    }
    for row in _read_jsonl(shard.generation_path):
        split = str(row.get("prompt_split", ""))
        decoding = str(row.get("decoding", ""))
        prompt_id = str(row.get("prompt_id", ""))
        if (
            not prompt_id.startswith("open_")
            or split not in output
            or decoding not in config.selection.decodings
        ):
            continue
        seed_value = row.get("seed")
        seed = None if seed_value is None else int(seed_value)
        if (decoding == "greedy" and seed is not None) or (
            decoding == "sample" and seed not in config.selection.sample_seeds
        ):
            raise JudgeWorkflowError(
                f"generation seed/decoding contract mismatch: {shard.generation_path}"
            )
        key = (prompt_id, decoding, seed)
        if key in output[split]:
            raise JudgeWorkflowError(f"duplicate generation key: {key}")
        output[split][key] = row
    for split, rows in output.items():
        if len(rows) != 64:
            raise JudgeWorkflowError(
                f"{shard.role} {split} expected 64 open generations, found {len(rows)}"
            )
    return output


def _concept_definitions(
    config: JudgeEvaluationConfig,
    *,
    config_path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    path = _registered_resource(
        config.source.concept_definitions_path, config_path=config_path
    )
    payload = _load_json(path, label="concept definitions")
    definitions: dict[str, dict[str, str]] = {}
    for entry in payload.get("concepts", []):
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("concept_name", ""))
        concept_id = str(entry.get("concept_id") or f"goemotions:{name}")
        definition = str(entry.get("definition", ""))
        if not name or not definition or concept_id in definitions:
            raise JudgeWorkflowError(f"invalid concept definition: {concept_id}")
        definitions[concept_id] = {"name": name, "definition": definition}
    if not all(concept in definitions for concept in config.source.concept_ids):
        raise JudgeWorkflowError("registered concept definitions are incomplete")
    return (
        {concept: definitions[concept] for concept in config.source.concept_ids},
        {"path": str(path), "sha256": sha256_file(path)},
    )


def _selected_roles_for_concept(
    config: JudgeEvaluationConfig,
    registries: Mapping[str, MethodRegistry],
    source_record: Mapping[str, Any],
    concept_id: str,
) -> tuple[dict[str, MethodShard], dict[str, Any]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    rescore_summaries: dict[str, dict[str, Any]] = {}
    rescore_records: dict[str, Any] = {}
    selected_queries: dict[str, dict[str, Any]] = {}
    selection_metrics: dict[str, Any] = {}
    shards: dict[str, list[MethodShard]] = {}
    for method in METHODS:
        rows[method], rescore_summaries[method], rescore_records[method] = (
            _load_rescore_rows(registries[method], concept_id)
        )
        selected, metrics = _select_query(
            rows[method],
            method=method,
            config=config,
            rescore_summary=rescore_summaries[method],
        )
        selected_queries[method] = selected
        selection_metrics[method] = metrics
        shards[method] = _load_shards(registries[method], concept_id)
    j_strength = float(selected_queries["j_component_intervention"]["strength"])
    raptor_probability = float(
        selected_queries["raptor_intervention"]["target_probability"]
    )
    native_query = selected_queries["iti_intervention"]
    layer_selected = selection_metrics["iti_intervention"]["layer_matched"]
    layer_strength = float(layer_selected["strength"])
    layer_top_k = int(layer_selected["top_k"])
    random_seed = config.selection.matched_random_seed
    roles = {
        "common_baseline": _find_shard(
            shards["j_component_intervention"],
            role="common_baseline",
            query={"condition_id": "full", "strength": 0.0},
        ),
        "j_component": _find_shard(
            shards["j_component_intervention"],
            role="j_component",
            query={"condition_id": "j", "strength": j_strength},
        ),
        "raptor": _find_shard(
            shards["raptor_intervention"],
            role="raptor",
            query={"target_probability": raptor_probability},
        ),
        "iti_native": _find_shard(
            shards["iti_intervention"],
            role="iti_native",
            query=native_query,
        ),
        "j_full": _find_shard(
            shards["j_component_intervention"],
            role="j_full",
            query={"condition_id": "full", "strength": j_strength},
        ),
        "j_non_j": _find_shard(
            shards["j_component_intervention"],
            role="j_non_j",
            query={"condition_id": "non_j", "strength": j_strength},
        ),
        "j_matched_random": _find_shard(
            shards["j_component_intervention"],
            role="j_matched_random",
            query={
                "condition_id": f"random_{random_seed}",
                "strength": j_strength,
            },
        ),
        "iti_layer_matched": _find_shard(
            shards["iti_intervention"],
            role="iti_layer_matched",
            query={
                "variant": "layer_matched",
                "condition_id": "iti_layer_matched_mass_mean",
                "mode": "mass_mean",
                "strength": layer_strength,
                "top_k": layer_top_k,
                "random_seed": None,
            },
        ),
        "iti_probe_weight": _find_shard(
            shards["iti_intervention"],
            role="iti_probe_weight",
            query={
                "variant": "native",
                "condition_id": "iti_native_probe_weight",
                "mode": "probe_weight",
                "strength": float(native_query["strength"]),
                "top_k": int(native_query["top_k"]),
                "random_seed": None,
            },
        ),
        "iti_matched_random": _find_shard(
            shards["iti_intervention"],
            role="iti_matched_random",
            query={
                "variant": "layer_matched",
                "condition_id": f"iti_layer_matched_random_{random_seed}",
                "mode": "random",
                "strength": layer_strength,
                "top_k": layer_top_k,
                "random_seed": random_seed,
            },
        ),
    }
    raptor_zero = _find_shard(
        shards["raptor_intervention"],
        role="_raptor_zero",
        query={"condition_id": "no_hook"},
    )
    native_zero_k = min(
        int(value)
        for value in rescore_summaries["iti_intervention"]["native_top_k_grid"]
    )
    iti_zero = _find_shard(
        shards["iti_intervention"],
        role="_iti_zero",
        query={
            "variant": "native",
            "condition_id": "iti_native_mass_mean",
            "mode": "mass_mean",
            "strength": 0.0,
            "top_k": native_zero_k,
            "random_seed": None,
        },
    )
    selected_layers = {
        tuple(int(value) for value in shard.summary["selected_layers"])
        for shard in [*roles.values(), raptor_zero, iti_zero]
    }
    contracts = {
        _canonical_hash(shard.summary["generation_contract"])
        for shard in [*roles.values(), raptor_zero, iti_zero]
    }
    if len(selected_layers) != 1 or len(contracts) != 1:
        raise JudgeWorkflowError(
            f"{concept_id}: selected roles do not share layers/generation contract"
        )
    zero_consistency = validate_equivalent_generation_outputs(
        [
            roles["common_baseline"].generation_path,
            raptor_zero.generation_path,
            iti_zero.generation_path,
        ]
    )
    comparison_entry = source_record["comparison_entries"][concept_id]
    if zero_consistency != comparison_entry.get("zero_generation_consistency"):
        raise JudgeWorkflowError(
            f"{concept_id}: zero-output check differs from sealed comparison"
        )
    expected = {
        "j": comparison_entry["j_component"]["selected"]["strength"],
        "raptor": comparison_entry["raptor"]["selected"]["target_probability"],
        "iti_strength": comparison_entry["iti_native"]["selected"]["strength"],
        "iti_top_k": comparison_entry["iti_native"]["selected"]["top_k"],
    }
    observed = {
        "j": j_strength,
        "raptor": raptor_probability,
        "iti_strength": native_query["strength"],
        "iti_top_k": native_query["top_k"],
    }
    if observed != expected:
        raise JudgeWorkflowError(
            f"{concept_id}: rescore selection differs from sealed comparison"
        )
    source_entries = {
        role: _validate_shard(
            shard,
            registry=registries[shard.method],
            concept_id=concept_id,
        )
        for role, shard in roles.items()
    }
    source_entries["_raptor_zero"] = _validate_shard(
        raptor_zero,
        registry=registries["raptor_intervention"],
        concept_id=concept_id,
    )
    source_entries["_iti_zero"] = _validate_shard(
        iti_zero,
        registry=registries["iti_intervention"],
        concept_id=concept_id,
    )
    return roles, {
        "selection_data": "candidate_score_rescore_v1 validation only",
        "j_component": {
            "query": selected_queries["j_component_intervention"],
            **selection_metrics["j_component_intervention"],
        },
        "raptor": {
            "query": selected_queries["raptor_intervention"],
            **selection_metrics["raptor_intervention"],
        },
        "iti_native": {
            "query": native_query,
            "mean_target_margin": selection_metrics["iti_intervention"][
                "mean_target_margin"
            ],
            "n_prompts": selection_metrics["iti_intervention"]["n_prompts"],
        },
        "iti_layer_matched": dict(layer_selected),
        "zero_generation_consistency": zero_consistency,
        "zero_score_consistency": comparison_entry["zero_score_consistency"],
        "rescore_artifacts": rescore_records,
        "selected_sources": source_entries,
    }


def _reuse_assessment(
    config: JudgeEvaluationConfig,
    *,
    destination: Path,
) -> dict[str, Any]:
    legacy = config.legacy_assessment
    if legacy is None:
        return {
            "assessed": False,
            "reusable_responses": 0,
            "reason": "no legacy source registered",
        }
    source_root = Path(legacy.source_root)
    manifest_path = source_root / "manifest.json"
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != legacy.expected_manifest_sha256
    ):
        raise JudgeWorkflowError("legacy assessment manifest hash mismatch")
    manifest = _load_json(manifest_path, label="legacy judge manifest")
    if manifest.get("experiment_name") != legacy.expected_experiment_name:
        raise JudgeWorkflowError("legacy assessment experiment name mismatch")
    comparisons: dict[str, Any] = {}
    potential_matches = 0
    for split in config.selection.evaluation_splits:
        old_path = source_root / "tasks" / f"pointwise_{split}.jsonl"
        new_path = destination / "tasks" / f"pointwise_{split}.jsonl"
        old_rows = _read_jsonl(old_path)
        new_rows = _read_jsonl(new_path)
        old_by_id = {str(row["task_id"]): row for row in old_rows}
        new_by_id = {str(row["task_id"]): row for row in new_rows}
        individual_matches = sum(
            old_by_id[task_id] == new_by_id.get(task_id)
            for task_id in old_by_id
        )
        potential_matches += individual_matches * 2
        comparisons[split] = {
            "legacy_task_file": str(old_path),
            "legacy_task_file_sha256": sha256_file(old_path),
            "new_task_file": str(new_path),
            "new_task_file_sha256": sha256_file(new_path),
            "public_task_file_byte_identical": (
                sha256_file(old_path) == sha256_file(new_path)
            ),
            "individual_task_matches_per_model": individual_matches,
            "legacy_rows": len(old_rows),
            "new_rows": len(new_rows),
        }
    raw_response_available = False
    reusable = 0
    return {
        "schema_version": 1,
        "assessed": True,
        "source_root": str(source_root),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "rubric_sha256": manifest.get("rubric_sha256"),
        "protocol_version": manifest.get("protocol_version"),
        "models": {
            "primary": config.judges.primary,
            "secondary": config.judges.secondary,
        },
        "task_file_comparisons": comparisons,
        "potential_individual_response_matches": potential_matches,
        "raw_provider_response_available": raw_response_available,
        "reusable_responses": reusable,
        "reused_responses": 0,
        "decision": "no_reuse",
        "reasons": [
            "new three-method public pointwise task files are not byte-identical to v16",
            "v16 did not retain raw provider response objects required by this registration",
            "individual task-ID/content overlap is insufficient for reuse",
        ],
    }


def _validate_adaptation(config: JudgeEvaluationConfig) -> dict[str, Any] | None:
    """Verify the failed parent and prove that its test set remains unopened."""

    adaptation = config.adaptation
    if adaptation is None:
        return None
    artifacts = {
        "manifest": (
            Path(adaptation.parent_manifest),
            adaptation.parent_manifest_sha256,
        ),
        "calibration": (
            Path(adaptation.parent_calibration),
            adaptation.parent_calibration_sha256,
        ),
        "outcome": (
            Path(adaptation.parent_outcome),
            adaptation.parent_outcome_sha256,
        ),
    }
    payloads: dict[str, dict[str, Any]] = {}
    for label, (path, expected_hash) in artifacts.items():
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise JudgeWorkflowError(f"parent {label} seal mismatch: {path}")
        payloads[label] = _load_json(path, label=f"parent {label}")
    if (
        payloads["manifest"].get("experiment_name")
        != adaptation.parent_experiment_name
        or payloads["calibration"].get("passed") is not False
        or payloads["calibration"].get("gates", {}).get(
            "pairwise_order_consistency"
        )
        is not False
        or payloads["calibration"].get("gates", {}).get("pointwise_spearman")
        is not True
        or payloads["calibration"].get("gates", {}).get("pointwise_mae")
        is not True
        or payloads["outcome"].get("pipeline_status") != "blocked_calibration"
        or payloads["outcome"].get("test_unlock_allowed") is not False
    ):
        raise JudgeWorkflowError("parent failure does not justify this adaptation")
    parent_root = Path(adaptation.parent_manifest).parent
    test_responses = sum(
        1
        for path in (parent_root / "responses").glob("*/*_test/*.json")
        if path.name != "index.json"
    )
    if test_responses != adaptation.parent_test_responses_at_registration:
        raise JudgeWorkflowError("parent test response count changed after registration")
    return {
        "reason": adaptation.reason,
        "parent_experiment_name": adaptation.parent_experiment_name,
        "artifacts": {
            label: {"path": str(path), "sha256": expected_hash}
            for label, (path, expected_hash) in artifacts.items()
        },
        "parent_test_responses_at_registration": test_responses,
        "reuse_parent_responses": False,
        "interpretation": "post_hoc_descriptive_only",
    }


def validate_evaluation(config_path: str | Path) -> dict[str, Any]:
    """Validate config and all top-level sealed source identities without writing."""

    registration = Path(config_path)
    config = load_judge_config(registration)
    registries, source_record = _load_source_registry(config)
    definitions, definition_record = _concept_definitions(
        config, config_path=registration
    )
    adaptation = _validate_adaptation(config)
    return {
        "valid": True,
        "schema_version": config.schema_version,
        "protocol_version": config.protocol_version,
        "study_design_version": config.study_design_version,
        "experiment_name": config.experiment_name,
        "descriptive_only": True,
        "task_sets": config.task_sets,
        "adaptation": adaptation,
        "concept_count": len(definitions),
        "methods": list(METHODS),
        "comparison_index_sha256": source_record["comparison_index_sha256"],
        "comparison_sha256": source_record["comparison_sha256"],
        "method_index_sha256": {
            method: sha256_file(registry.index_path)
            for method, registry in registries.items()
        },
        "rescore_index_sha256": {
            method: sha256_file(registry.rescore_index_path)
            for method, registry in registries.items()
        },
        "concept_definitions": definition_record,
        "api_key_stored_in_config": False,
        "formal_remote_run_approved": False,
    }


def prepare_evaluation(config_path: str | Path) -> dict[str, Any]:
    """Select only on rescored validation data and freeze all blinded tasks."""

    registration = Path(config_path)
    config = load_judge_config(registration)
    destination = Path(config.output_dir)
    registries, source_record = _load_source_registry(config)
    definitions, definition_record = _concept_definitions(
        config, config_path=registration
    )
    adaptation = _validate_adaptation(config)
    public_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    private_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    selections: dict[str, Any] = {}
    for concept_id in config.source.concept_ids:
        roles, selection = _selected_roles_for_concept(
            config, registries, source_record, concept_id
        )
        selections[concept_id] = selection
        generations = {
            role: _eligible_generations(shard, config=config)
            for role, shard in roles.items()
        }
        for split in config.selection.evaluation_splits:
            key_sets = {role: set(rows[split]) for role, rows in generations.items()}
            if len({tuple(sorted(values)) for values in key_sets.values()}) != 1:
                raise JudgeWorkflowError(
                    f"{concept_id} {split}: pointwise roles are not exactly paired"
                )
            keys = next(iter(key_sets.values()))
            for role in (*PRIMARY_ROLES, *SECONDARY_ROLES):
                shard = roles[role]
                for key in sorted(
                    keys,
                    key=lambda item: (
                        item[0],
                        item[1],
                        -1 if item[2] is None else item[2],
                    ),
                ):
                    row = generations[role][split][key]
                    task_id = _canonical_hash(
                        {
                            "protocol": config.protocol_version,
                            "rubric": PROMPT_VERSION,
                            "kind": "pointwise",
                            "generation_id": row["generation_id"],
                        }
                    )
                    public_rows[("pointwise", split)].append(
                        {
                            "schema_version": 1,
                            "rubric_version": PROMPT_VERSION,
                            "task_id": task_id,
                            "task_type": "pointwise",
                            "split": split,
                            "concept_id": concept_id,
                            "concept_name": definitions[concept_id]["name"],
                            "concept_definition": definitions[concept_id][
                                "definition"
                            ],
                            "user_prompt": row["prompt_text"],
                            "response": row["generated_text"],
                        }
                    )
                    private_rows[("pointwise", split)].append(
                        {
                            "task_id": task_id,
                            "generation_id": row["generation_id"],
                            "blind_id": row["blind_id"],
                            "analysis_tier": (
                                "primary" if role in PRIMARY_ROLES else "secondary"
                            ),
                            "condition_role": role,
                            "method": shard.method,
                            "condition_id": row["condition_id"],
                            "grid_point": row["grid_point"],
                            "concept_id": concept_id,
                            "prompt_id": row["prompt_id"],
                            "decoding": row["decoding"],
                            "seed": row.get("seed"),
                            "source_summary": str(shard.summary_path),
                            "source_summary_sha256": sha256_file(
                                shard.summary_path
                            ),
                        }
                    )
            for left_role, right_role in METHOD_PAIRS:
                for key in sorted(
                    keys,
                    key=lambda item: (
                        item[0],
                        item[1],
                        -1 if item[2] is None else item[2],
                    ),
                ):
                    left = generations[left_role][split][key]
                    right = generations[right_role][split][key]
                    method_pair = f"{left_role}_vs_{right_role}"
                    group_id = _canonical_hash(
                        {
                            "protocol": config.protocol_version,
                            "kind": "three_method_pairwise_group",
                            "method_pair": method_pair,
                            "left_generation_id": left["generation_id"],
                            "right_generation_id": right["generation_id"],
                        }
                    )
                    first_left = int(group_id[-1], 16) % 2 == 0
                    for order_index in range(2):
                        left_first = first_left if order_index == 0 else not first_left
                        response_a, response_b = (
                            (left, right) if left_first else (right, left)
                        )
                        role_a, role_b = (
                            (left_role, right_role)
                            if left_first
                            else (right_role, left_role)
                        )
                        task_id = _canonical_hash(
                            {
                                "protocol": config.protocol_version,
                                "rubric": PROMPT_VERSION,
                                "kind": "three_method_pairwise",
                                "pair_group_id": group_id,
                                "order_index": order_index,
                            }
                        )
                        public_rows[("pairwise", split)].append(
                            {
                                "schema_version": 1,
                                "rubric_version": PROMPT_VERSION,
                                "task_id": task_id,
                                "task_type": "pairwise",
                                "split": split,
                                "concept_id": concept_id,
                                "concept_name": definitions[concept_id]["name"],
                                "concept_definition": definitions[concept_id][
                                    "definition"
                                ],
                                "user_prompt": left["prompt_text"],
                                "response_a": response_a["generated_text"],
                                "response_b": response_b["generated_text"],
                            }
                        )
                        private_rows[("pairwise", split)].append(
                            {
                                "task_id": task_id,
                                "pair_group_id": group_id,
                                "method_pair": method_pair,
                                "order_index": order_index,
                                "concept_id": concept_id,
                                "prompt_id": left["prompt_id"],
                                "decoding": left["decoding"],
                                "seed": left.get("seed"),
                                "response_a_role": role_a,
                                "response_b_role": role_b,
                                "response_a_generation_id": response_a[
                                    "generation_id"
                                ],
                                "response_b_generation_id": response_b[
                                    "generation_id"
                                ],
                            }
                        )
    expected_per_split = {"pointwise": 4480, "pairwise": 2688}
    task_artifacts: dict[str, Any] = {}
    for (task_type, split), rows in sorted(public_rows.items()):
        if task_type not in config.task_sets:
            continue
        if len(rows) != expected_per_split[task_type]:
            raise JudgeWorkflowError(
                f"{task_type} {split}: expected {expected_per_split[task_type]} "
                f"tasks, found {len(rows)}"
            )
        if len({str(row["task_id"]) for row in rows}) != len(rows):
            raise JudgeWorkflowError(f"duplicate public task IDs: {task_type} {split}")
        private_by_id = {
            str(row["task_id"]): row
            for row in private_rows[(task_type, split)]
        }
        random.Random(
            f"{config.seed}:{task_type}:{split}:{config.study_design_version}"
        ).shuffle(rows)
        ordered_private = [private_by_id[str(row["task_id"])] for row in rows]
        public_path = destination / "tasks" / f"{task_type}_{split}.jsonl"
        private_path = (
            destination / "private" / f"{task_type}_{split}_map.jsonl"
        )
        _registered_write_jsonl(public_path, rows)
        _registered_write_jsonl(private_path, ordered_private)
        private_path.chmod(0o600)
        private_path.parent.chmod(0o700)
        task_artifacts[f"{task_type}_{split}"] = {
            "path": str(public_path),
            "sha256": sha256_file(public_path),
            "rows": len(rows),
            "private_map": str(private_path),
            "private_map_sha256": sha256_file(private_path),
        }
    frozen_config_path = destination / "frozen_config.yaml"
    _registered_write_text(
        frozen_config_path,
        registration.read_text(encoding="utf-8"),
    )
    assessment = _reuse_assessment(config, destination=destination)
    assessment_path = destination / "migration" / "legacy_reuse_assessment.json"
    _registered_write_json(assessment_path, assessment)
    implementation_root = Path(__file__).parent
    implementation_files = {
        path.name: sha256_file(path)
        for path in sorted(implementation_root.glob("*.py"))
    }
    manifest = {
        "schema_version": 1,
        "protocol_version": config.protocol_version,
        "study_design_version": config.study_design_version,
        "experiment_name": config.experiment_name,
        "llm_as_judge_run": False,
        "descriptive_only": True,
        "task_sets": config.task_sets,
        "adaptation": adaptation,
        "implementation": {
            "files": implementation_files,
            "combined_sha256": _canonical_hash(implementation_files),
        },
        "config": str(registration.resolve()),
        "config_sha256": sha256_file(registration),
        "frozen_config": str(frozen_config_path),
        "frozen_config_sha256": sha256_file(frozen_config_path),
        "rubric_version": PROMPT_VERSION,
        "rubric_sha256": rubric_hash(),
        "concept_definitions": definition_record,
        "methods": list(METHODS),
        "primary_pointwise_roles": list(PRIMARY_ROLES),
        "secondary_pointwise_roles": list(SECONDARY_ROLES),
        "primary_method_pairs": (
            [f"{left}_vs_{right}" for left, right in METHOD_PAIRS]
            if "pairwise" in config.task_sets
            else []
        ),
        "primary_contrasts": config.analysis.primary_contrasts,
        "blinding": {
            "hidden_from_judges": [
                "method",
                "condition",
                "strength",
                "top_k",
                "variant",
                "seed",
                "file_path",
                "research_hypothesis",
                "provenance",
            ],
            "target_concept_definition_visible": True,
            "pairwise_both_orders": "pairwise" in config.task_sets,
            "public_task_keys": {
                task_set: sorted(public_rows[(task_set, "validation")][0])
                for task_set in config.task_sets
            },
        },
        "selection": selections,
        "source_registry": source_record,
        "task_artifacts": task_artifacts,
        "legacy_reuse_assessment": {
            "path": str(assessment_path),
            "sha256": sha256_file(assessment_path),
            "reusable_responses": assessment["reusable_responses"],
        },
        "formal_remote_run_approved": False,
    }
    _registered_write_json(destination / "manifest.json", manifest)
    return manifest
