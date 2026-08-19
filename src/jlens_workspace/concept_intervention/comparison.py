"""Identity-checked direct-metric comparison of the three interventions."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from jlens_workspace.artifacts import atomic_write_json, sha256_file
from jlens_workspace.concept_intervention.candidate_rescore import (
    CANONICAL_SCORE_CONTRACT,
)
from jlens_workspace.concept_intervention.generation import (
    InterventionGenerationError,
    validate_candidate_score_artifact,
    validate_equivalent_generation_outputs,
    validate_generation_artifacts,
    validate_generation_contract_identity,
    validate_index_builder_provenance,
    validate_method_index_provenance,
    validate_target_artifact_seal,
)
from jlens_workspace.concept_intervention.iti.intervention import (
    ITIError,
    validate_iti_direction_artifact,
)


class InterventionComparisonError(ValueError):
    """Raised when method outputs are incomplete or not comparable."""


def validate_three_method_smokes(
    *,
    output_path: str | Path,
    j_root: str | Path,
    iti_root: str | Path,
    raptor_root: str | Path,
    concept_id: str = "goemotions:admiration",
) -> dict[str, Any]:
    """Validate the registered GPU smoke shards before full-grid submission."""

    encoded = quote(concept_id, safe="")
    specifications = (
        (
            "j_component_intervention",
            Path(j_root),
            6,
            {"condition_id": "full", "strength": 0.5},
        ),
        (
            "raptor_intervention",
            Path(raptor_root),
            8,
            {
                "condition_id": "target_probability_0.9999",
                "target_probability": 0.9999,
            },
        ),
        (
            "iti_intervention",
            Path(iti_root),
            6,
            {
                "variant": "native",
                "mode": "mass_mean",
                "random_seed": None,
                "condition_id": "iti_native_mass_mean",
                "top_k": 4,
                "strength": 20.0,
            },
        ),
        (
            "iti_intervention",
            Path(iti_root),
            251,
            {
                "variant": "layer_matched",
                "mode": "mass_mean",
                "random_seed": None,
                "condition_id": "iti_layer_matched_mass_mean",
                "top_k": 8,
                "strength": 20.0,
            },
        ),
    )
    entries = []
    for method, root, grid_index, expected_condition in specifications:
        shard = (
            root
            / "targets"
            / encoded
            / "shards"
            / f"grid_{grid_index:04d}"
        )
        summary_path = shard / "summary.json"
        if not summary_path.is_file():
            raise InterventionComparisonError(
                f"registered smoke summary is missing: {summary_path}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        observed_identity = (
            summary.get("method"),
            summary.get("target_concept_id"),
            summary.get("grid_index"),
            summary.get("grid_condition"),
        )
        expected_identity = (
            method,
            concept_id,
            grid_index,
            expected_condition,
        )
        if observed_identity != expected_identity:
            raise InterventionComparisonError(
                f"registered smoke identity mismatch: {summary_path}"
            )
        candidate_path = shard / "candidate_scores.jsonl"
        try:
            validate_candidate_score_artifact(
                candidate_path,
                expected_sha256=str(
                    summary.get("candidate_scores_sha256", "")
                ),
                contract=summary["generation_contract"],
                expected_method=method,
                expected_concept_id=concept_id,
                expected_grid_condition=expected_condition,
            )
        except InterventionGenerationError as error:
            raise InterventionComparisonError(
                f"registered smoke score identity mismatch: {candidate_path}"
            ) from error
        validate_generation_artifacts(
            shard,
            summary["generation_files"],
            contract=summary["generation_contract"],
            expected_selected_layers=summary["selected_layers"],
        )
        entries.append(
            {
                "method": method,
                "grid_index": grid_index,
                "grid_condition": expected_condition,
                "summary": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
            }
        )
    result = {
        "schema_version": 1,
        "complete": True,
        "concept_id": concept_id,
        "contract": "registered_gpu_smokes_before_full_grid_submission",
        "entries": entries,
    }
    atomic_write_json(output_path, result)
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_method_scores(
    target: Path,
    *,
    sealed_paths: set[Path],
) -> list[dict[str, Any]]:
    direct = target / "candidate_scores.jsonl"
    if direct.is_file():
        if direct.resolve() not in sealed_paths:
            raise InterventionComparisonError(
                f"candidate scores are absent from method artifact seal: {direct}"
            )
        return _read_jsonl(direct)
    shards = sorted((target / "shards").glob("grid_*/candidate_scores.jsonl"))
    if not shards:
        raise InterventionComparisonError(
            f"candidate score artifact is missing: {target}"
        )
    unsealed = [path for path in shards if path.resolve() not in sealed_paths]
    if unsealed:
        raise InterventionComparisonError(
            f"candidate scores are absent from method artifact seal: {unsealed[0]}"
        )
    return [row for path in shards for row in _read_jsonl(path)]


def _mean(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        raise InterventionComparisonError("cannot average an empty score set")
    return float(np.mean([float(row["target_margin"]) for row in rows]))


def _baseline_generation_path(
    root: Path,
    summary: Mapping[str, Any],
    *,
    condition: Mapping[str, Any],
) -> Path:
    matches = [
        row
        for row in summary.get("shards", [])
        if all(
            row.get("grid_condition", {}).get(key) == value
            for key, value in condition.items()
        )
    ]
    if len(matches) != 1:
        raise InterventionComparisonError(
            f"expected one baseline generation shard for {condition}, found {len(matches)}"
        )
    return root / str(matches[0]["summary"]).replace(
        "summary.json", "generations.jsonl"
    )


def _validate_sealed_summary_graph(
    root: Path,
    summary_path: Path,
    summary: Mapping[str, Any],
    *,
    sealed_paths: set[Path],
) -> None:
    """Recheck every summary-declared downstream hash at comparison time."""

    method_root = root.resolve()
    if summary_path.resolve() not in sealed_paths:
        raise InterventionComparisonError(
            f"target summary is absent from method artifact seal: {summary_path}"
        )
    for shard in summary.get("shards", []):
        if not isinstance(shard, Mapping):
            raise InterventionComparisonError(
                f"malformed shard entry in target summary: {summary_path}"
            )
        shard_path = (method_root / str(shard.get("summary", ""))).resolve()
        try:
            shard_path.relative_to(method_root)
        except ValueError as error:
            raise InterventionComparisonError(
                f"shard summary escapes method root: {shard_path}"
            ) from error
        if (
            shard_path not in sealed_paths
            or not shard_path.is_file()
            or shard.get("summary_sha256") != sha256_file(shard_path)
        ):
            raise InterventionComparisonError(
                f"shard summary hash-chain mismatch: {shard_path}"
            )
        shard_payload = json.loads(shard_path.read_text(encoding="utf-8"))
        candidate_path = shard_path.parent / "candidate_scores.jsonl"
        candidate_hash = shard_payload.get("candidate_scores_sha256")
        if candidate_hash is not None and (
            candidate_path.resolve() not in sealed_paths
            or not candidate_path.is_file()
            or candidate_hash != sha256_file(candidate_path)
        ):
            raise InterventionComparisonError(
                f"candidate-score hash-chain mismatch: {candidate_path}"
            )
        generation_files = shard_payload.get("generation_files")
        if generation_files is None:
            continue
        if not isinstance(generation_files, Mapping):
            raise InterventionComparisonError(
                f"malformed generation hash declarations: {shard_path}"
            )
        for filename, hash_key in (
            ("generations.jsonl", "generations_sha256"),
            ("judge_blind_generations.jsonl", "blind_generations_sha256"),
            ("judge_blind_map.jsonl", "blind_map_sha256"),
        ):
            artifact = shard_path.parent / filename
            if (
                artifact.resolve() not in sealed_paths
                or not artifact.is_file()
                or generation_files.get(hash_key) != sha256_file(artifact)
            ):
                raise InterventionComparisonError(
                    f"generation hash-chain mismatch: {artifact}"
                )


def _validate_cross_method_zero_scores(
    method_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    queries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    baselines: dict[str, dict[str, Mapping[str, float]]] = {}
    for method, rows in method_rows.items():
        query = queries[method]
        selected = [
            row
            for row in rows
            if all(row.get(key) == value for key, value in query.items())
        ]
        by_prompt = {
            str(row["prompt_id"]): {
                str(key): float(value)
                for key, value in row["candidate_log_probabilities"].items()
            }
            for row in selected
        }
        if not by_prompt or len(by_prompt) != len(selected):
            raise InterventionComparisonError(
                f"{method} zero baseline is empty or has duplicate prompts"
            )
        baselines[method] = by_prompt
    prompt_sets = {tuple(sorted(rows)) for rows in baselines.values()}
    if len(prompt_sets) != 1:
        raise InterventionComparisonError(
            "methods used different prompts for zero/no-hook scoring"
        )
    reference = next(iter(baselines.values()))
    max_difference = 0.0
    for rows in baselines.values():
        for prompt_id, values in rows.items():
            if set(values) != set(reference[prompt_id]):
                raise InterventionComparisonError(
                    "methods used different candidate labels at zero/no-hook"
                )
            max_difference = max(
                max_difference,
                max(
                    abs(values[key] - reference[prompt_id][key])
                    for key in values
                ),
            )
    if max_difference > 1e-5:
        raise InterventionComparisonError(
            f"cross-method zero/no-hook scores differ by {max_difference}"
        )
    return {
        "prompt_count": len(reference),
        "max_candidate_logprob_difference": max_difference,
    }


def _effect(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    def matches(row: Mapping[str, Any], query: Mapping[str, Any]) -> bool:
        return all(row.get(key) == value for key, value in query.items())

    test = [row for row in rows if row.get("evaluation_split") == "test"]
    treatment = [row for row in test if matches(row, selected)]
    reference = [row for row in test if matches(row, baseline)]
    treatment_by_prompt = {
        str(row["prompt_id"]): float(row["target_margin"]) for row in treatment
    }
    baseline_by_prompt = {
        str(row["prompt_id"]): float(row["target_margin"]) for row in reference
    }
    if not treatment_by_prompt or set(treatment_by_prompt) != set(baseline_by_prompt):
        raise InterventionComparisonError(
            f"selected/baseline prompts differ: selected={selected}, baseline={baseline}"
        )
    deltas = np.asarray(
        [
            treatment_by_prompt[prompt] - baseline_by_prompt[prompt]
            for prompt in sorted(treatment_by_prompt)
        ],
        dtype=np.float64,
    )
    return {
        "selected": dict(selected),
        "baseline": dict(baseline),
        "held_out_target_margin_effect": float(np.mean(deltas)),
        "held_out_target_margin_effect_median": float(np.median(deltas)),
        "n_paired_prompts": int(deltas.size),
    }


def _select_j(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("evaluation_split") == "validation"
            and row.get("condition_id") == "j"
            and float(row["strength"]) > 0
        ):
            candidates[float(row["strength"])].append(row)
    strength = max(candidates, key=lambda value: (_mean(candidates[value]), -value))
    return _effect(
        rows,
        selected={"condition_id": "j", "strength": strength},
        baseline={"condition_id": "j", "strength": 0.0},
    )


def _select_raptor(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        probability = row.get("target_probability")
        if (
            row.get("evaluation_split") == "validation"
            and probability is not None
            and float(probability) > 0.5
        ):
            candidates[float(probability)].append(row)
    probability = max(
        candidates,
        key=lambda value: (_mean(candidates[value]), -abs(value - 0.5)),
    )
    return _effect(
        rows,
        selected={"target_probability": probability},
        baseline={"condition_id": "no_hook"},
    )


def _select_iti(
    rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], variant: str
) -> dict[str, Any]:
    selected = summary["selection"][variant]["selected"]
    condition = f"iti_{variant}_mass_mean"
    return _effect(
        rows,
        selected={
            "variant": variant,
            "condition_id": condition,
            "top_k": int(selected["top_k"]),
            "strength": float(selected["strength"]),
        },
        baseline={
            "variant": variant,
            "condition_id": condition,
            "top_k": int(selected["top_k"]),
            "strength": 0.0,
        },
    )


def rebuild_intervention_comparison(
    *,
    output_dir: str | Path,
    shared_layer_selection: str | Path,
    j_root: str | Path,
    iti_root: str | Path,
    raptor_root: str | Path,
    concept_ids: Sequence[str],
    index_builder: Mapping[str, Any] | None = None,
    candidate_rescore_roots: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Build a no-judge held-out comparison after all three indexes complete."""

    roots = {
        "j_component_intervention": Path(j_root),
        "iti_intervention": Path(iti_root),
        "raptor_intervention": Path(raptor_root),
    }
    rescore_roots = (
        None
        if candidate_rescore_roots is None
        else {
            str(method): Path(root)
            for method, root in candidate_rescore_roots.items()
        }
    )
    if rescore_roots is not None and set(rescore_roots) != set(roots):
        raise InterventionComparisonError(
            "candidate rescore roots must cover exactly all three methods"
        )
    candidate_rescore_indexes: dict[str, dict[str, Any]] = {}
    candidate_rescore_entries: dict[
        str, dict[str, Mapping[str, Any]]
    ] = {}
    indexes = {}
    method_provenance: dict[str, dict[str, Any]] = {}
    generation_identities: dict[str, dict[str, Any]] = {}
    method_entries: dict[str, dict[str, Mapping[str, Any]]] = {}
    method_seals: dict[str, dict[str, set[Path]]] = {}
    method_identities: set[tuple[Any, ...]] = set()
    generation_contracts: set[str] = set()
    selection_path = Path(shared_layer_selection)
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    expected_selection_hash = sha256_file(selection_path)
    row_manifest_path = (
        selection_path.parent / str(selection_payload["row_manifest"])
    ).resolve()
    expected_row_manifest_hash = sha256_file(row_manifest_path)
    if (
        selection_payload.get("row_manifest_sha256")
        != expected_row_manifest_hash
    ):
        raise InterventionComparisonError(
            f"shared row-manifest identity mismatch: {row_manifest_path}"
        )
    for method, root in roots.items():
        path = root / "index.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("complete"):
            raise InterventionComparisonError(f"{method} index is incomplete")
        if payload.get("method") != method:
            raise InterventionComparisonError(f"{method} index method mismatch")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise InterventionComparisonError(f"{method} index lacks entries")
        entries_by_concept = {
            str(entry.get("concept_id")): entry for entry in entries
        }
        if (
            len(entries_by_concept) != len(entries)
            or set(entries_by_concept) != set(concept_ids)
            or set(payload.get("expected_concepts", [])) != set(concept_ids)
            or set(payload.get("observed_concepts", [])) != set(concept_ids)
        ):
            raise InterventionComparisonError(
                f"{method} index concept coverage mismatch"
            )
        try:
            method_provenance[method] = validate_method_index_provenance(
                root,
                payload,
            )
        except InterventionGenerationError as error:
            raise InterventionComparisonError(
                f"{method} method provenance validation failed"
            ) from error
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("manifest_sha256") != sha256_file(manifest_path):
            raise InterventionComparisonError(
                f"{method} manifest identity differs from its index"
            )
        notes = manifest.get("notes", {})
        identity = (
            manifest.get("model_id"),
            manifest.get("model_revision"),
            manifest.get("tokenizer_id"),
            manifest.get("tokenizer_revision"),
            manifest.get("lens_source"),
            notes.get("lens_sha256"),
            manifest.get("dataset_source"),
            manifest.get("dataset_revision"),
            manifest.get("dataset_hash"),
            manifest.get("git_commit"),
            manifest.get("python"),
            json.dumps(manifest.get("packages"), sort_keys=True),
        )
        if any(value is None for value in identity):
            raise InterventionComparisonError(
                f"{method} manifest lacks a required model/data/lens identity"
            )
        method_identities.add(identity)
        generation_identity = notes.get("generation")
        if (
            not isinstance(generation_identity, dict)
            or not generation_identity.get("candidate_prompts_sha256")
            or not generation_identity.get("open_prompts_sha256")
        ):
            raise InterventionComparisonError(
                f"{method} manifest lacks prompt content hashes"
            )
        generation_contracts.add(json.dumps(generation_identity, sort_keys=True))
        generation_identities[method] = generation_identity
        if notes.get("selected_layers_sha256") != expected_selection_hash:
            raise InterventionComparisonError(
                f"{method} manifest used another selected-layer artifact"
            )
        if notes.get("row_manifest_sha256") != expected_row_manifest_hash:
            raise InterventionComparisonError(
                f"{method} manifest used another balanced-row artifact"
            )
        seals: dict[str, set[Path]] = {}
        for concept_id, entry in entries_by_concept.items():
            summary_path = (root / str(entry.get("summary", ""))).resolve()
            if (
                not summary_path.is_file()
                or entry.get("summary_sha256") != sha256_file(summary_path)
            ):
                raise InterventionComparisonError(
                    f"{method} target summary artifact mismatch: {summary_path}"
                )
            try:
                sealed = validate_target_artifact_seal(
                    root,
                    entry.get("artifact_seal", {}),
                )
            except InterventionGenerationError as error:
                raise InterventionComparisonError(
                    f"{method} target artifact seal failed for {concept_id}"
                ) from error
            if summary_path not in sealed:
                raise InterventionComparisonError(
                    f"{method} target summary is absent from artifact seal"
                )
            seals[concept_id] = sealed
        method_entries[method] = entries_by_concept
        method_seals[method] = seals
        indexes[method] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    if index_builder is None:
        builder_identities = {
            json.dumps(value["index_builder"], sort_keys=True)
            for value in method_provenance.values()
        }
        if len(builder_identities) != 1:
            raise InterventionComparisonError(
                "method indexes were built by different runtimes"
            )
        comparison_builder = next(
            iter(method_provenance.values())
        )["index_builder"]
    else:
        try:
            comparison_builder = validate_index_builder_provenance(index_builder)
        except InterventionGenerationError as error:
            raise InterventionComparisonError(
                "comparison index-builder provenance is malformed"
            ) from error
    if len(method_identities) != 1 or len(generation_contracts) != 1:
        raise InterventionComparisonError(
            "methods used different model/data/lens or generation identities"
        )
    if rescore_roots is not None:
        for method, rescore_root in rescore_roots.items():
            rescore_root = rescore_root.resolve()
            index_path = rescore_root / "index.json"
            manifest_path = rescore_root / "manifest.json"
            if not index_path.is_file() or not manifest_path.is_file():
                raise InterventionComparisonError(
                    f"{method} candidate rescore index/manifest is missing"
                )
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rescore_entries = payload.get("entries")
            if (
                payload.get("complete") is not True
                or payload.get("artifact_kind")
                != "candidate_score_rescore_index"
                or payload.get("method") != method
                or payload.get("scoring_contract")
                != CANONICAL_SCORE_CONTRACT
                or payload.get("source_method_index_sha256")
                != indexes[method]["sha256"]
                or payload.get("source_method_manifest_sha256")
                != sha256_file(roots[method] / "manifest.json")
                or payload.get("rescore_manifest_sha256")
                != sha256_file(manifest_path)
                or not isinstance(rescore_entries, list)
            ):
                raise InterventionComparisonError(
                    f"{method} candidate rescore index provenance is malformed"
                )
            metadata = manifest.get("notes", {}).get("candidate_rescore")
            if (
                not isinstance(metadata, Mapping)
                or metadata.get("method") != method
                or metadata.get("scoring_contract")
                != CANONICAL_SCORE_CONTRACT
                or metadata.get("source_method_index_sha256")
                != indexes[method]["sha256"]
                or metadata.get("source_method_manifest_sha256")
                != sha256_file(roots[method] / "manifest.json")
            ):
                raise InterventionComparisonError(
                    f"{method} candidate rescore manifest link is malformed"
                )
            try:
                rescore_builder = validate_index_builder_provenance(
                    payload.get("index_builder", {})
                )
            except InterventionGenerationError as error:
                raise InterventionComparisonError(
                    f"{method} candidate rescore builder provenance is malformed"
                ) from error
            entries_by_concept = {
                str(entry.get("concept_id")): entry
                for entry in rescore_entries
            }
            if (
                len(entries_by_concept) != len(rescore_entries)
                or set(entries_by_concept) != set(concept_ids)
                or set(payload.get("expected_concepts", [])) != set(concept_ids)
                or set(payload.get("observed_concepts", [])) != set(concept_ids)
            ):
                raise InterventionComparisonError(
                    f"{method} candidate rescore concept coverage mismatch"
                )
            for concept_id, entry in entries_by_concept.items():
                summary_path = (
                    rescore_root / str(entry.get("summary", ""))
                ).resolve()
                candidate_path = (
                    rescore_root / str(entry.get("candidate_scores", ""))
                ).resolve()
                if (
                    not summary_path.is_relative_to(rescore_root)
                    or not candidate_path.is_relative_to(rescore_root)
                    or not summary_path.is_file()
                    or not candidate_path.is_file()
                    or entry.get("summary_sha256") != sha256_file(summary_path)
                    or entry.get("candidate_scores_sha256")
                    != sha256_file(candidate_path)
                ):
                    raise InterventionComparisonError(
                        f"{method} candidate rescore artifact mismatch: {concept_id}"
                    )
            candidate_rescore_entries[method] = entries_by_concept
            candidate_rescore_indexes[method] = {
                "path": str(index_path),
                "sha256": sha256_file(index_path),
                "index_builder": rescore_builder,
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
            }
    entries = []
    for concept_id in concept_ids:
        encoded = quote(concept_id, safe="")
        summaries = {
            method: json.loads(
                (root / str(method_entries[method][concept_id]["summary"])).read_text(
                    encoding="utf-8"
                )
            )
            for method, root in roots.items()
        }
        for method, summary in summaries.items():
            root = roots[method]
            summary_path = (
                root / str(method_entries[method][concept_id]["summary"])
            ).resolve()
            _validate_sealed_summary_graph(
                root,
                summary_path,
                summary,
                sealed_paths=method_seals[method][concept_id],
            )
        score_summaries = dict(summaries)
        if rescore_roots is not None:
            for method, rescore_root in rescore_roots.items():
                entry = candidate_rescore_entries[method][concept_id]
                summary_path = (
                    rescore_root / str(entry["summary"])
                ).resolve()
                candidate_path = (
                    rescore_root / str(entry["candidate_scores"])
                ).resolve()
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if (
                    summary.get("artifact_kind")
                    != "candidate_score_rescore"
                    or summary.get("method") != method
                    or summary.get("target_concept_id") != concept_id
                    or summary.get("candidate_score_contract")
                    != CANONICAL_SCORE_CONTRACT
                    or summary.get("generation_rows") != 0
                    or summary.get("generation_files") is not None
                    or summary.get("generation_contract")
                    != summaries[method].get("generation_contract")
                    or summary.get("selected_layers")
                    != summaries[method].get("selected_layers")
                    or summary.get("candidate_scores_sha256")
                    != sha256_file(candidate_path)
                ):
                    raise InterventionComparisonError(
                        f"{concept_id}: {method} rescore summary differs from source"
                    )
                score_summaries[method] = summary
        selected_layers = {
            tuple(int(value) for value in summary["selected_layers"])
            for summary in summaries.values()
        }
        if len(selected_layers) != 1:
            raise InterventionComparisonError(
                f"{concept_id}: methods used different selected layers"
            )
        for method, summary in summaries.items():
            generation_contract = summary.get("generation_contract", {})
            generation_identity = generation_identities[method]
            try:
                validate_generation_contract_identity(
                    generation_contract,
                    generation_identity,
                )
            except InterventionGenerationError as error:
                raise InterventionComparisonError(
                    f"{concept_id}: {method} generation contract differs "
                    "from the content-addressed prompt banks"
                ) from error
        iti_direction_metrics = Path(
            summaries["iti_intervention"]["direction_metrics"]
        )
        if (
            summaries["iti_intervention"].get("direction_metrics_sha256")
            != sha256_file(iti_direction_metrics)
        ):
            raise InterventionComparisonError(
                f"{concept_id}: ITI direction metrics identity mismatch"
            )
        try:
            iti_direction_payload = validate_iti_direction_artifact(
                iti_direction_metrics
            )
        except ITIError as error:
            raise InterventionComparisonError(
                f"{concept_id}: ITI direction artifact identity mismatch"
            ) from error
        selection_hashes = {
            str(
                summaries["j_component_intervention"]["source_provenance"][
                    "selected_layers_sha256"
                ]
            ),
            str(
                summaries["raptor_intervention"]["source_provenance"][
                    "selected_layers_sha256"
                ]
            ),
            str(iti_direction_payload["selected_layers_sha256"]),
        }
        if selection_hashes != {expected_selection_hash}:
            raise InterventionComparisonError(
                f"{concept_id}: methods used different selected-layer artifact hashes"
            )
        row_manifest_hashes = {
            str(
                summaries["j_component_intervention"]["source_provenance"][
                    "row_manifest_sha256"
                ]
            ),
            str(
                summaries["raptor_intervention"]["source_provenance"][
                    "row_manifest_sha256"
                ]
            ),
            str(iti_direction_payload["row_manifest_sha256"]),
        }
        if row_manifest_hashes != {expected_row_manifest_hash}:
            raise InterventionComparisonError(
                f"{concept_id}: methods used different balanced-row artifact hashes"
            )
        if rescore_roots is None:
            score_rows = {
                method: _read_method_scores(
                    root / "targets" / encoded,
                    sealed_paths=method_seals[method][concept_id],
                )
                for method, root in roots.items()
            }
        else:
            score_rows = {}
            for method, rescore_root in rescore_roots.items():
                candidate_path = (
                    rescore_root
                    / str(
                        candidate_rescore_entries[method][concept_id][
                            "candidate_scores"
                        ]
                    )
                ).resolve()
                score_rows[method] = _read_method_scores(
                    rescore_root / "targets" / encoded,
                    sealed_paths={candidate_path},
                )
                declared_rows = candidate_rescore_entries[method][concept_id].get(
                    "candidate_score_rows"
                )
                if declared_rows != len(score_rows[method]):
                    raise InterventionComparisonError(
                        f"{concept_id}: {method} rescore row count mismatch"
                    )
        j_rows = score_rows["j_component_intervention"]
        iti_rows = score_rows["iti_intervention"]
        raptor_rows = score_rows["raptor_intervention"]
        native_zero_k = min(
            int(value)
            for value in summaries["iti_intervention"]["native_top_k_grid"]
        )
        baseline_generation_paths = [
            _baseline_generation_path(
                roots["j_component_intervention"],
                summaries["j_component_intervention"],
                condition={"condition_id": "full", "strength": 0.0},
            ),
            _baseline_generation_path(
                roots["raptor_intervention"],
                summaries["raptor_intervention"],
                condition={"condition_id": "no_hook"},
            ),
            _baseline_generation_path(
                roots["iti_intervention"],
                summaries["iti_intervention"],
                condition={
                    "variant": "native",
                    "condition_id": "iti_native_mass_mean",
                    "top_k": native_zero_k,
                    "strength": 0.0,
                },
            ),
        ]
        for method, path in zip(
            (
                "j_component_intervention",
                "raptor_intervention",
                "iti_intervention",
            ),
            baseline_generation_paths,
            strict=True,
        ):
            if path.resolve() not in method_seals[method][concept_id]:
                raise InterventionComparisonError(
                    f"{method} baseline generation is absent from artifact seal"
                )
        zero_generation_consistency = validate_equivalent_generation_outputs(
            baseline_generation_paths
        )
        zero_score_consistency = _validate_cross_method_zero_scores(
            {
                "j_component_intervention": j_rows,
                "raptor_intervention": raptor_rows,
                "iti_intervention": iti_rows,
            },
            {
                "j_component_intervention": {
                    "condition_id": "full",
                    "strength": 0.0,
                },
                "raptor_intervention": {"condition_id": "no_hook"},
                "iti_intervention": {
                    "variant": "native",
                    "condition_id": "iti_native_mass_mean",
                    "top_k": native_zero_k,
                    "strength": 0.0,
                },
            },
        )
        entries.append(
            {
                "concept_id": concept_id,
                "selected_layers": list(next(iter(selected_layers))),
                "zero_generation_consistency": zero_generation_consistency,
                "zero_score_consistency": zero_score_consistency,
                "j_component": _select_j(j_rows),
                "raptor": _select_raptor(raptor_rows),
                "iti_native": _select_iti(
                    iti_rows, score_summaries["iti_intervention"], "native"
                ),
                "iti_layer_matched": _select_iti(
                    iti_rows, score_summaries["iti_intervention"], "layer_matched"
                ),
            }
        )
    comparison = {
        "schema_version": 1,
        "comparison": "three_method_shared_layer_held_out_target_margin",
        "same_selected_layers_per_concept": True,
        "llm_as_judge_run": False,
        "primary_metric": (
            "held-out change in candidate-label target margin relative to each "
            "method/variant's zero or no-hook baseline"
        ),
        "shared_layer_selection": str(shared_layer_selection),
        "shared_layer_selection_sha256": sha256_file(shared_layer_selection),
        "method_indexes": indexes,
        "method_provenance": method_provenance,
        "candidate_rescore_indexes": candidate_rescore_indexes,
        "index_builder": comparison_builder,
        "entries": entries,
    }
    destination = Path(output_dir)
    atomic_write_json(destination / "comparison.json", comparison)
    index = {
        "schema_version": 1,
        "complete": True,
        "comparison": "comparison.json",
        "comparison_sha256": sha256_file(destination / "comparison.json"),
        "methods": list(roots),
        "concept_ids": list(concept_ids),
        "method_provenance": method_provenance,
        "candidate_rescore_indexes": candidate_rescore_indexes,
        "index_builder": comparison_builder,
        "llm_as_judge_run": False,
    }
    atomic_write_json(destination / "index.json", index)
    return index
