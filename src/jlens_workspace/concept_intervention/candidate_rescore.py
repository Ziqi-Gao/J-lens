"""Audited candidate-only rescoring over sealed intervention generations."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

from jlens_workspace.artifacts import atomic_write_json, sha256_file


class CandidateRescoreError(RuntimeError):
    """Raised when a candidate rescore diverges from its sealed source grid."""


CANONICAL_SCORE_CONTRACT = {
    "schema_version": 1,
    "batch_size": 1,
    "batching": "one_prompt_per_forward",
}

_MEASUREMENT_FIELDS = frozenset(
    {
        "candidate_log_probabilities",
        "candidate_probabilities_normalized",
        "target_log_probability",
        "target_candidate_probability",
        "target_margin",
        "target_rank",
        "telemetry",
    }
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateRescoreError(f"invalid JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise CandidateRescoreError(f"JSON artifact must be an object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CandidateRescoreError(f"candidate artifact is missing: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise CandidateRescoreError(
                        f"candidate row must be an object: {path}"
                    )
                rows.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateRescoreError(f"invalid candidate JSONL: {path}") from error
    if not rows:
        raise CandidateRescoreError(f"candidate artifact is empty: {path}")
    return rows


def _row_identity(row: Mapping[str, Any]) -> str:
    identity = {
        str(key): value
        for key, value in row.items()
        if key not in _MEASUREMENT_FIELDS
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _validate_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    concept_id: str,
    generation_contract: Mapping[str, Any],
) -> None:
    labels = generation_contract.get("candidate_labels")
    prompt_splits = generation_contract.get("candidate_prompt_splits")
    if (
        not isinstance(labels, Mapping)
        or len(labels) != 7
        or concept_id not in labels
        or not isinstance(prompt_splits, Mapping)
    ):
        raise CandidateRescoreError("rescore generation contract is malformed")
    for row in rows:
        prompt_id = str(row.get("prompt_id", ""))
        log_probs = row.get("candidate_log_probabilities")
        probabilities = row.get("candidate_probabilities_normalized")
        if (
            row.get("method") != method
            or row.get("target_concept_id") != concept_id
            or prompt_id not in prompt_splits
            or row.get("evaluation_split") != prompt_splits[prompt_id]
            or not isinstance(log_probs, Mapping)
            or not isinstance(probabilities, Mapping)
            or set(log_probs) != set(labels)
            or set(probabilities) != set(labels)
        ):
            raise CandidateRescoreError(
                f"candidate scientific identity mismatch: {concept_id}"
            )
        numeric = [*log_probs.values(), *probabilities.values()]
        numeric.extend(
            row.get(field)
            for field in (
                "target_log_probability",
                "target_candidate_probability",
                "target_margin",
            )
        )
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in numeric
        ) or not isinstance(row.get("target_rank"), int):
            raise CandidateRescoreError(f"candidate metrics are incomplete: {concept_id}")
        target_log_prob = float(log_probs[concept_id])
        off_target = [
            float(value) for key, value in log_probs.items() if key != concept_id
        ]
        expected_margin = target_log_prob - sum(off_target) / len(off_target)
        expected_rank = 1 + sum(value > target_log_prob for value in off_target)
        if (
            not math.isclose(
                float(row["target_log_probability"]),
                target_log_prob,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(row["target_candidate_probability"]),
                float(probabilities[concept_id]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(row["target_margin"]),
                expected_margin,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            or int(row["target_rank"]) != expected_rank
            or not math.isclose(
                sum(float(value) for value in probabilities.values()),
                1.0,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        ):
            raise CandidateRescoreError(
                f"candidate metrics are internally inconsistent: {concept_id}"
            )


def rebuild_candidate_rescore_index(
    output_dir: str | Path,
    *,
    source_method_root: str | Path,
    method: str,
    concept_ids: Sequence[str],
    index_builder: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a canonical rescore against every sealed source grid row."""

    root = Path(output_dir).resolve()
    source_root = Path(source_method_root).resolve()
    if root == source_root:
        raise CandidateRescoreError("rescore output must differ from source method root")
    source_index_path = source_root / "index.json"
    source_manifest_path = source_root / "manifest.json"
    manifest_path = root / "manifest.json"
    source_index = _load_json(source_index_path)
    _load_json(source_manifest_path)
    manifest = _load_json(manifest_path)
    expected = set(concept_ids)
    source_entries = source_index.get("entries")
    if (
        source_index.get("complete") is not True
        or source_index.get("method") != method
        or not isinstance(source_entries, list)
        or source_index.get("manifest_sha256") != sha256_file(source_manifest_path)
    ):
        raise CandidateRescoreError("source method index is incomplete or malformed")
    source_by_concept = {
        str(entry.get("concept_id")): entry for entry in source_entries
    }
    if set(source_by_concept) != expected or len(source_by_concept) != len(source_entries):
        raise CandidateRescoreError("source method concept coverage differs")
    metadata = manifest.get("notes", {}).get("candidate_rescore")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema_version") != 1
        or metadata.get("method") != method
        or metadata.get("scoring_contract") != CANONICAL_SCORE_CONTRACT
        or metadata.get("source_method_index_sha256")
        != sha256_file(source_index_path)
        or metadata.get("source_method_manifest_sha256")
        != sha256_file(source_manifest_path)
    ):
        raise CandidateRescoreError("rescore manifest provenance is malformed")

    entries: list[dict[str, Any]] = []
    observed: set[str] = set()
    for concept_id in concept_ids:
        encoded = quote(concept_id, safe="")
        source_entry = source_by_concept[concept_id]
        source_summary_path = source_root / str(source_entry.get("summary", ""))
        if (
            not source_summary_path.is_file()
            or source_entry.get("summary_sha256") != sha256_file(source_summary_path)
        ):
            raise CandidateRescoreError(
                f"source target summary identity mismatch: {concept_id}"
            )
        source_summary = _load_json(source_summary_path)
        generation_contract = source_summary.get("generation_contract")
        if not isinstance(generation_contract, Mapping):
            raise CandidateRescoreError(
                f"source generation contract is missing: {concept_id}"
            )
        seal = source_entry.get("artifact_seal", {}).get("files")
        if not isinstance(seal, list):
            raise CandidateRescoreError(f"source artifact seal is missing: {concept_id}")
        candidate_entries = sorted(
            (
                entry
                for entry in seal
                if str(entry.get("path", "")).endswith("candidate_scores.jsonl")
            ),
            key=lambda entry: str(entry["path"]),
        )
        if not candidate_entries:
            raise CandidateRescoreError(
                f"source candidate artifacts are absent: {concept_id}"
            )
        source_rows: list[dict[str, Any]] = []
        for sealed in candidate_entries:
            path = source_root / str(sealed["path"])
            if sealed.get("sha256") != sha256_file(path):
                raise CandidateRescoreError(f"source candidate seal mismatch: {path}")
            source_rows.extend(_load_jsonl(path))

        summary_path = root / "targets" / encoded / "summary.json"
        candidate_path = summary_path.parent / "candidate_scores.jsonl"
        summary = _load_json(summary_path)
        if (
            summary.get("artifact_kind") != "candidate_score_rescore"
            or summary.get("method") != method
            or summary.get("target_concept_id") != concept_id
            or summary.get("candidate_score_contract") != CANONICAL_SCORE_CONTRACT
            or summary.get("generation_rows") != 0
            or summary.get("generation_files") is not None
            or summary.get("generation_contract") != generation_contract
            or summary.get("selected_layers") != source_summary.get("selected_layers")
            or summary.get("candidate_scores_sha256") != sha256_file(candidate_path)
        ):
            raise CandidateRescoreError(
                f"rescore target summary identity mismatch: {concept_id}"
            )
        rows = _load_jsonl(candidate_path)
        _validate_metrics(
            rows,
            method=method,
            concept_id=concept_id,
            generation_contract=generation_contract,
        )
        if Counter(map(_row_identity, rows)) != Counter(
            map(_row_identity, source_rows)
        ):
            raise CandidateRescoreError(
                f"rescore grid/prompt identity differs from source: {concept_id}"
            )
        observed.add(concept_id)
        entries.append(
            {
                "concept_id": concept_id,
                "summary": summary_path.relative_to(root).as_posix(),
                "summary_sha256": sha256_file(summary_path),
                "candidate_scores": candidate_path.relative_to(root).as_posix(),
                "candidate_scores_sha256": sha256_file(candidate_path),
                "candidate_score_rows": len(rows),
                "source_candidate_artifacts": len(candidate_entries),
                "selection": summary.get("selection"),
            }
        )

    index = {
        "schema_version": 1,
        "artifact_kind": "candidate_score_rescore_index",
        "method": method,
        "complete": observed == expected,
        "expected_concepts": list(concept_ids),
        "observed_concepts": sorted(observed),
        "missing_concepts": sorted(expected - observed),
        "scoring_contract": CANONICAL_SCORE_CONTRACT,
        "source_method_root": str(source_root),
        "source_method_index_sha256": sha256_file(source_index_path),
        "source_method_manifest_sha256": sha256_file(source_manifest_path),
        "rescore_manifest_sha256": sha256_file(manifest_path),
        "index_builder": dict(index_builder or {}),
        "entries": entries,
    }
    atomic_write_json(root / "index.json", index)
    return index
