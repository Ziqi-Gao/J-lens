"""Scientific completion gates and retry-safe project receipts."""

from __future__ import annotations

import json
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

from jlens_workspace.foundation.artifacts import atomic_write_json, sha256_file
from jlens_workspace.scheduler.catalog import (
    KDIAG_REPORT_REVISION_R1,
    KDIAG_RUN_ID,
    KDIAG_RUN_ROOT,
    KDIAG_STAGE_DIRECTORY,
    THREE_METHOD_CONCEPTS,
    THREE_METHOD_LAYERS,
    THREE_METHOD_REPLICATES,
    THREE_METHOD_REVISION_R2,
    THREE_METHOD_RUN_ID,
    THREE_METHOD_RUN_ROOT,
)
from jlens_workspace.scheduler.errors import ServerSchedulerAdapterError
from jlens_workspace.scheduler.jsonio import integer, strict_json_load, strict_object
from jlens_workspace.scheduler.manifest import RunningJob

_CANONICAL_SCORE_CONTRACT = {
    "schema_version": 1,
    "batch_size": 1,
    "batching": "one_prompt_per_forward",
}


@dataclass(frozen=True)
class CompletionEvidence:
    """Small immutable metadata file proving a task-specific completion gate."""

    path: str
    sha256: str


def derivation_output_root(job: RunningJob, run_root: Path) -> Path:
    """Return the exclusive write-once root for a registered derived task."""

    index = int(job.parameters["shard_index"])
    if job.task == "candidate-rescore":
        methods = (
            "j_component_intervention",
            "iti_intervention",
            "raptor_intervention",
        )
        return (
            run_root
            / THREE_METHOD_REVISION_R2
            / "candidate_score_rescore_v1"
            / methods[index]
        )
    if job.task == "comparison-rescore-index":
        return run_root / THREE_METHOD_REVISION_R2 / "intervention_comparison"
    if job.task == "kdiag-report":
        return run_root / KDIAG_REPORT_REVISION_R1
    raise ServerSchedulerAdapterError(
        f"task {job.task!r} has no registered derivation output root"
    )


def _regular_file(
    path: Path,
    *,
    root: Path,
    label: str,
    maximum: int = 64 * 1024 * 1024,
) -> None:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
        metadata = path.lstat()
    except (OSError, ValueError) as error:
        raise ServerSchedulerAdapterError(
            f"{label} is absent or outside the run root: {path}"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ServerSchedulerAdapterError(f"{label} must be a regular non-symlink file")
    if metadata.st_size <= 0 or metadata.st_size > maximum:
        raise ServerSchedulerAdapterError(f"{label} has an invalid size")


def _json_evidence(
    path: Path,
    *,
    root: Path,
    label: str,
    complete: bool = False,
    expected: Mapping[str, object] | None = None,
    allow_legacy_nonfinite: bool = False,
) -> tuple[Mapping[str, object], CompletionEvidence]:
    _regular_file(path, root=root, label=label)
    if allow_legacy_nonfinite:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ServerSchedulerAdapterError(
                f"cannot read legacy JSON evidence {path}: {error}"
            ) from error
    else:
        payload = strict_json_load(path)
    if not isinstance(payload, Mapping):
        raise ServerSchedulerAdapterError(f"{label} must contain a JSON object")
    if complete and payload.get("complete") is not True:
        raise ServerSchedulerAdapterError(f"{label} is not complete")
    for key, value in (expected or {}).items():
        if payload.get(key) != value:
            raise ServerSchedulerAdapterError(f"{label} has the wrong {key}")
    return payload, CompletionEvidence(str(path.resolve()), sha256_file(path))


def _require_related_files(paths: Sequence[Path], *, root: Path, label: str) -> None:
    for path in paths:
        _regular_file(path, root=root, label=label, maximum=8 * 1024 * 1024 * 1024)


def _descriptor_evidence(
    descriptor: object,
    *,
    root: Path,
    label: str,
) -> CompletionEvidence:
    """Validate one hash/size descriptor from a sealed report manifest."""

    if not isinstance(descriptor, Mapping):
        raise ServerSchedulerAdapterError(f"{label} descriptor is malformed")
    raw_path = descriptor.get("path")
    expected_bytes = descriptor.get("bytes")
    expected_sha256 = descriptor.get("sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or Path(raw_path).is_absolute()
        or ".." in Path(raw_path).parts
        or not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes <= 0
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ServerSchedulerAdapterError(f"{label} descriptor is malformed")
    path = root / raw_path
    _regular_file(path, root=root, label=label, maximum=8 * 1024 * 1024 * 1024)
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha256:
        raise ServerSchedulerAdapterError(f"{label} descriptor no longer matches its file")
    return CompletionEvidence(str(path.resolve()), expected_sha256)


def _relative_hash_evidence(
    raw_path: object,
    expected_sha256: object,
    *,
    base: Path,
    containment_root: Path,
    label: str,
) -> CompletionEvidence:
    """Validate a relative path plus SHA-256 pair from an artifact index."""

    if (
        not isinstance(raw_path, str)
        or not raw_path
        or Path(raw_path).is_absolute()
        or ".." in Path(raw_path).parts
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ServerSchedulerAdapterError(f"{label} path/hash is malformed")
    path = base / raw_path
    _regular_file(path, root=containment_root, label=label, maximum=8 * 1024 * 1024 * 1024)
    if sha256_file(path) != expected_sha256:
        raise ServerSchedulerAdapterError(f"{label} hash no longer matches")
    return CompletionEvidence(str(path.resolve()), expected_sha256)


def _three_method_completion(
    job: RunningJob,
    run_root: Path,
) -> tuple[CompletionEvidence, ...]:
    base = run_root / "artifacts/concept_intervention" / THREE_METHOD_RUN_ID
    shared = base / "shared_intervention_protocol"
    task = job.task
    index = int(job.parameters["shard_index"])
    evidence: list[CompletionEvidence] = []
    if task == "preflight":
        return ()
    if task == "lens":
        lens_root = shared / "lens"
        manifest = lens_root / "qwen35_4b_fp32.checkpoint.pt.manifest.json"
        payload, item = _json_evidence(manifest, root=run_root, label="lens manifest")
        expected_identity = {
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "tokenizer_id": "Qwen/Qwen3.5-4B",
            "tokenizer_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "source_layers": list(THREE_METHOD_LAYERS),
            "target_layer": 31,
            "n_fit_prompts": 1000,
            "force_bos": False,
            "storage_dtype": "float32",
        }
        if any(payload.get(key) != value for key, value in expected_identity.items()):
            raise ServerSchedulerAdapterError("fitted lens identity manifest is wrong")
        final_lens = lens_root / "qwen35_4b_fp32.pt"
        _regular_file(
            final_lens,
            root=run_root,
            label="final fitted lens",
            maximum=8 * 1024 * 1024 * 1024,
        )
        # The adapter saves this path by atomic replace only after fit returns;
        # the checkpoint beside it is resume evidence and is never completion.
        return (item, CompletionEvidence(str(final_lens.resolve()), sha256_file(final_lens)))
    if task in {"capture", "iti-capture"}:
        activation_root = (
            shared / "activations"
            if task == "capture"
            else base / "iti_intervention/head_activations"
        )
        _payload, item = _json_evidence(
            activation_root / "metadata.json",
            root=run_root,
            label=f"{task} metadata",
        )
        related = [activation_root / f"layer_{layer:02d}.npy" for layer in THREE_METHOD_LAYERS]
        related.extend((activation_root / "labels.npy", activation_root / "rows.jsonl"))
        _require_related_files(related, root=run_root, label=f"{task} activation output")
        return (item,)
    if task == "layer-selection":
        payload, item = _json_evidence(
            shared / "selection/layer_selection.json",
            root=run_root,
            label="layer selection",
        )
        selected_count = payload.get("selected_layer_count")
        if (
            not isinstance(selected_count, int)
            or isinstance(selected_count, bool)
            or selected_count <= 0
            or not payload.get("candidate_layers")
        ):
            raise ServerSchedulerAdapterError("layer selection is incomplete")
        return (item,)
    if task == "bootstrap-probes":
        layer = THREE_METHOD_LAYERS[index]
        payload, item = _json_evidence(
            shared / f"selection/probe_replicates/manifests/layers_{layer:02d}.json",
            root=run_root,
            label="bootstrap layer manifest",
        )
        if payload.get("layers") != [layer] or not payload.get("entries"):
            raise ServerSchedulerAdapterError("bootstrap layer manifest is incomplete")
        return (item,)
    if task == "bootstrap-index":
        payload, item = _json_evidence(
            shared / "selection/probe_replicates/manifest.json",
            root=run_root,
            label="bootstrap index",
            complete=True,
        )
        if payload.get("missing_entries") not in ([], 0):
            raise ServerSchedulerAdapterError("bootstrap index contains missing entries")
        return (item,)
    if task == "occupancy":
        layer = THREE_METHOD_LAYERS[index % len(THREE_METHOD_LAYERS)]
        replicate = THREE_METHOD_REPLICATES[index // len(THREE_METHOD_LAYERS)]
        for concept in THREE_METHOD_CONCEPTS:
            payload, item = _json_evidence(
                shared
                / "j_occupancy/occupancy/rmsnorm_weighted/positive_cosine"
                / f"layer_{layer:02d}"
                / quote(concept, safe="")
                / "pos"
                / replicate
                / "metrics.json",
                root=run_root,
                label="occupancy shard metric",
                expected={
                    "layer": layer,
                    "concept_id": concept,
                    "replicate_id": replicate,
                },
                allow_legacy_nonfinite=True,
            )
            if payload.get("method") != "concept_occupancy_method":
                raise ServerSchedulerAdapterError("occupancy shard has the wrong method")
            evidence.append(item)
        return tuple(evidence)
    if task == "occupancy-index":
        _payload, item = _json_evidence(
            shared / "j_occupancy/occupancy/index.json",
            root=run_root,
            label="occupancy index",
            complete=True,
        )
        return (item,)
    if task == "iti-fit":
        for concept in THREE_METHOD_CONCEPTS:
            _payload, item = _json_evidence(
                base / "iti_intervention/directions" / quote(concept, safe="") / "metrics.json",
                root=run_root,
                label="ITI fitted direction",
                expected={"concept_id": concept},
            )
            evidence.append(item)
        return tuple(evidence)
    if task in {"j-grid", "raptor-grid", "iti-grid"}:
        sizes = {"j-grid": 56, "raptor-grid": 9, "iti-grid": 441}
        outputs = {
            "j-grid": "j_component_intervention",
            "raptor-grid": "raptor_intervention",
            "iti-grid": "iti_intervention",
        }
        grid_size = sizes[task]
        concept = THREE_METHOD_CONCEPTS[index // grid_size]
        grid_index = index % grid_size
        shard_root = (
            base
            / outputs[task]
            / "targets"
            / quote(concept, safe="")
            / "shards"
            / f"grid_{grid_index:04d}"
        )
        payload, item = _json_evidence(
            shard_root / "summary.json",
            root=run_root,
            label=f"{task} shard summary",
            expected={"grid_index": grid_index, "target_concept_id": concept},
        )
        if not isinstance(payload.get("generation_rows"), int) or payload["generation_rows"] <= 0:
            raise ServerSchedulerAdapterError(f"{task} shard contains no generations")
        _require_related_files(
            tuple(
                shard_root / name
                for name in (
                    "candidate_scores.jsonl",
                    "generations.jsonl",
                    "judge_blind_generations.jsonl",
                    "judge_blind_map.jsonl",
                )
            ),
            root=run_root,
            label=f"{task} shard output",
        )
        return (item,)
    if task == "smoke-check":
        _payload, item = _json_evidence(
            base / "intervention_comparison/smoke_gate.json",
            root=run_root,
            label="three-method smoke gate",
            complete=True,
        )
        return (item,)
    if task == "method-index":
        method = (
            "j_component_intervention",
            "iti_intervention",
            "raptor_intervention",
        )[index]
        _payload, item = _json_evidence(
            base / method / "index.json",
            root=run_root,
            label="method index",
            complete=True,
        )
        return (item,)
    if task in {"comparison-index", "comparison-rescore-index"}:
        comparison_root = (
            base / "intervention_comparison"
            if task == "comparison-index"
            else run_root / THREE_METHOD_REVISION_R2 / "intervention_comparison"
        )
        payload, item = _json_evidence(
            comparison_root / "index.json",
            root=run_root,
            label=(
                "comparison index"
                if task == "comparison-index"
                else "candidate-rescore comparison index"
            ),
            complete=True,
        )
        comparison_methods = [
            "j_component_intervention",
            "iti_intervention",
            "raptor_intervention",
        ]
        if (
            payload.get("methods") != comparison_methods
            or payload.get("concept_ids") != list(THREE_METHOD_CONCEPTS)
            or payload.get("llm_as_judge_run") is not False
            or not isinstance(payload.get("method_provenance"), Mapping)
            or set(payload["method_provenance"]) != set(comparison_methods)
            or not isinstance(payload.get("index_builder"), Mapping)
        ):
            raise ServerSchedulerAdapterError("comparison index identity is malformed")
        rescore_evidence: list[CompletionEvidence] = []
        if task == "comparison-rescore-index":
            rescore_indexes = payload.get("candidate_rescore_indexes")
            expected_methods = {
                "j_component_intervention",
                "iti_intervention",
                "raptor_intervention",
            }
            if not isinstance(rescore_indexes, Mapping) or set(rescore_indexes) != expected_methods:
                raise ServerSchedulerAdapterError(
                    "comparison rescore index must cover exactly all three methods"
                )
            for method in sorted(expected_methods):
                descriptor = rescore_indexes[method]
                if not isinstance(descriptor, Mapping):
                    raise ServerSchedulerAdapterError(
                        f"comparison rescore descriptor is malformed: {method}"
                    )
                rescore_root = (
                    run_root
                    / THREE_METHOD_REVISION_R2
                    / "candidate_score_rescore_v1"
                    / method
                )
                index_path = rescore_root / "index.json"
                manifest_path = rescore_root / "manifest.json"
                if descriptor.get("path") != str(index_path.resolve(strict=False)):
                    raise ServerSchedulerAdapterError(
                        f"comparison rescore index path is wrong: {method}"
                    )
                index_item = _relative_hash_evidence(
                    "index.json",
                    descriptor.get("sha256"),
                    base=rescore_root,
                    containment_root=run_root,
                    label=f"comparison candidate rescore index {method}",
                )
                if descriptor.get("manifest") != str(manifest_path.resolve(strict=False)):
                    raise ServerSchedulerAdapterError(
                        f"comparison rescore manifest path is wrong: {method}"
                    )
                manifest_item = _relative_hash_evidence(
                    "manifest.json",
                    descriptor.get("manifest_sha256"),
                    base=rescore_root,
                    containment_root=run_root,
                    label=f"comparison candidate rescore manifest {method}",
                )
                rescore_payload = strict_json_load(index_path)
                source_root = base / method
                rescore_entries = (
                    rescore_payload.get("entries")
                    if isinstance(rescore_payload, Mapping)
                    else None
                )
                if (
                    not isinstance(rescore_payload, Mapping)
                    or rescore_payload.get("complete") is not True
                    or rescore_payload.get("artifact_kind")
                    != "candidate_score_rescore_index"
                    or rescore_payload.get("method") != method
                    or rescore_payload.get("scoring_contract")
                    != _CANONICAL_SCORE_CONTRACT
                    or rescore_payload.get("expected_concepts")
                    != list(THREE_METHOD_CONCEPTS)
                    or rescore_payload.get("observed_concepts")
                    != sorted(THREE_METHOD_CONCEPTS)
                    or rescore_payload.get("missing_concepts") != []
                    or not isinstance(rescore_entries, list)
                    or len(rescore_entries) != len(THREE_METHOD_CONCEPTS)
                    or rescore_payload.get("rescore_manifest_sha256")
                    != manifest_item.sha256
                    or rescore_payload.get("source_method_index_sha256")
                    != sha256_file(source_root / "index.json")
                    or rescore_payload.get("source_method_manifest_sha256")
                    != sha256_file(source_root / "manifest.json")
                ):
                    raise ServerSchedulerAdapterError(
                        f"comparison candidate rescore provenance is invalid: {method}"
                    )
                source_index_item = _relative_hash_evidence(
                    "index.json",
                    rescore_payload.get("source_method_index_sha256"),
                    base=source_root,
                    containment_root=run_root,
                    label=f"comparison source method index {method}",
                )
                source_manifest_item = _relative_hash_evidence(
                    "manifest.json",
                    rescore_payload.get("source_method_manifest_sha256"),
                    base=source_root,
                    containment_root=run_root,
                    label=f"comparison source method manifest {method}",
                )
                observed_rescore_concepts: set[str] = set()
                entry_evidence: list[CompletionEvidence] = []
                for entry in rescore_entries:
                    if not isinstance(entry, Mapping):
                        raise ServerSchedulerAdapterError(
                            f"comparison candidate rescore entry is malformed: {method}"
                        )
                    concept_id = entry.get("concept_id")
                    if (
                        not isinstance(concept_id, str)
                        or concept_id in observed_rescore_concepts
                    ):
                        raise ServerSchedulerAdapterError(
                            f"comparison candidate rescore concept is malformed: {method}"
                        )
                    observed_rescore_concepts.add(concept_id)
                    summary_item = _relative_hash_evidence(
                        entry.get("summary"),
                        entry.get("summary_sha256"),
                        base=rescore_root,
                        containment_root=run_root,
                        label=f"comparison rescore summary {method}/{concept_id}",
                    )
                    candidate_item = _relative_hash_evidence(
                        entry.get("candidate_scores"),
                        entry.get("candidate_scores_sha256"),
                        base=rescore_root,
                        containment_root=run_root,
                        label=f"comparison rescore scores {method}/{concept_id}",
                    )
                    summary_payload = strict_json_load(Path(summary_item.path))
                    if (
                        not isinstance(summary_payload, Mapping)
                        or summary_payload.get("artifact_kind")
                        != "candidate_score_rescore"
                        or summary_payload.get("method") != method
                        or summary_payload.get("target_concept_id") != concept_id
                        or summary_payload.get("candidate_score_contract")
                        != _CANONICAL_SCORE_CONTRACT
                        or summary_payload.get("candidate_scores_sha256")
                        != candidate_item.sha256
                    ):
                        raise ServerSchedulerAdapterError(
                            f"comparison rescore summary is malformed: {method}/{concept_id}"
                        )
                    entry_evidence.extend((summary_item, candidate_item))
                if observed_rescore_concepts != set(THREE_METHOD_CONCEPTS):
                    raise ServerSchedulerAdapterError(
                        f"comparison candidate rescore coverage is incomplete: {method}"
                    )
                rescore_evidence.extend(
                    (
                        index_item,
                        manifest_item,
                        source_index_item,
                        source_manifest_item,
                        *entry_evidence,
                    )
                )
        comparison_name = payload.get("comparison")
        comparison_sha256 = payload.get("comparison_sha256")
        if (
            comparison_name != "comparison.json"
            or not isinstance(comparison_sha256, str)
            or len(comparison_sha256) != 64
            or any(character not in "0123456789abcdef" for character in comparison_sha256)
        ):
            raise ServerSchedulerAdapterError("comparison index has malformed companion evidence")
        comparison_path = comparison_root / comparison_name
        _regular_file(
            comparison_path,
            root=run_root,
            label="three-method comparison",
        )
        observed_sha256 = sha256_file(comparison_path)
        if observed_sha256 != comparison_sha256:
            raise ServerSchedulerAdapterError("comparison index companion hash is invalid")
        comparison_payload = strict_json_load(comparison_path)
        comparison_entries = (
            comparison_payload.get("entries")
            if isinstance(comparison_payload, Mapping)
            else None
        )
        if (
            not isinstance(comparison_payload, Mapping)
            or comparison_payload.get("comparison")
            != "three_method_shared_layer_held_out_target_margin"
            or comparison_payload.get("llm_as_judge_run") is not False
            or comparison_payload.get("candidate_rescore_indexes")
            != payload.get("candidate_rescore_indexes")
            or not isinstance(comparison_entries, list)
            or len(comparison_entries) != len(THREE_METHOD_CONCEPTS)
            or {entry.get("concept_id") for entry in comparison_entries if isinstance(entry, Mapping)}
            != set(THREE_METHOD_CONCEPTS)
        ):
            raise ServerSchedulerAdapterError("comparison companion identity is malformed")
        return (
            item,
            CompletionEvidence(str(comparison_path.resolve()), observed_sha256),
            *rescore_evidence,
        )
    if task == "candidate-rescore":
        method = (
            "j_component_intervention",
            "iti_intervention",
            "raptor_intervention",
        )[index]
        rescore_root = (
            run_root
            / THREE_METHOD_REVISION_R2
            / "candidate_score_rescore_v1"
            / method
        )
        payload, item = _json_evidence(
            rescore_root / "index.json",
            root=run_root,
            label="candidate rescore index",
            complete=True,
        )
        expected_concepts = list(THREE_METHOD_CONCEPTS)
        entries = payload.get("entries")
        if (
            payload.get("artifact_kind") != "candidate_score_rescore_index"
            or payload.get("method") != method
            or payload.get("scoring_contract") != _CANONICAL_SCORE_CONTRACT
            or payload.get("expected_concepts") != expected_concepts
            or payload.get("observed_concepts") != sorted(expected_concepts)
            or payload.get("missing_concepts") != []
            or not isinstance(entries, list)
            or len(entries) != len(expected_concepts)
        ):
            raise ServerSchedulerAdapterError("candidate rescore index identity is malformed")

        evidence = [item]
        evidence.append(
            _relative_hash_evidence(
                "manifest.json",
                payload.get("rescore_manifest_sha256"),
                base=rescore_root,
                containment_root=run_root,
                label="candidate rescore manifest",
            )
        )
        source_root = base / method
        try:
            registered_source_root = Path(str(payload.get("source_method_root"))).resolve(
                strict=True
            )
        except OSError as error:
            raise ServerSchedulerAdapterError("candidate rescore source root is absent") from error
        if registered_source_root != source_root.resolve(strict=True):
            raise ServerSchedulerAdapterError("candidate rescore source root is wrong")
        evidence.extend(
            (
                _relative_hash_evidence(
                    "index.json",
                    payload.get("source_method_index_sha256"),
                    base=source_root,
                    containment_root=run_root,
                    label="candidate rescore source index",
                ),
                _relative_hash_evidence(
                    "manifest.json",
                    payload.get("source_method_manifest_sha256"),
                    base=source_root,
                    containment_root=run_root,
                    label="candidate rescore source manifest",
                ),
            )
        )
        observed_concepts: set[str] = set()
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ServerSchedulerAdapterError("candidate rescore entry is malformed")
            concept_id = entry.get("concept_id")
            if not isinstance(concept_id, str) or concept_id in observed_concepts:
                raise ServerSchedulerAdapterError("candidate rescore concept identity is malformed")
            observed_concepts.add(concept_id)
            summary_item = _relative_hash_evidence(
                entry.get("summary"),
                entry.get("summary_sha256"),
                base=rescore_root,
                containment_root=run_root,
                label=f"candidate rescore summary {concept_id}",
            )
            candidate_item = _relative_hash_evidence(
                entry.get("candidate_scores"),
                entry.get("candidate_scores_sha256"),
                base=rescore_root,
                containment_root=run_root,
                label=f"candidate rescore scores {concept_id}",
            )
            summary_payload = strict_json_load(Path(summary_item.path))
            if (
                not isinstance(summary_payload, Mapping)
                or summary_payload.get("artifact_kind") != "candidate_score_rescore"
                or summary_payload.get("method") != method
                or summary_payload.get("target_concept_id") != concept_id
                or summary_payload.get("candidate_score_contract")
                != _CANONICAL_SCORE_CONTRACT
                or summary_payload.get("candidate_scores_sha256") != candidate_item.sha256
            ):
                raise ServerSchedulerAdapterError(
                    f"candidate rescore summary identity is malformed: {concept_id}"
                )
            evidence.extend((summary_item, candidate_item))
        if observed_concepts != set(expected_concepts):
            raise ServerSchedulerAdapterError("candidate rescore concept coverage is incomplete")
        return tuple(evidence)
    raise ServerSchedulerAdapterError(f"task {task!r} has no scientific completion contract")


def _kdiag_completion(job: RunningJob, run_root: Path) -> tuple[CompletionEvidence, ...]:
    root = run_root / "artifacts/concept_intervention" / KDIAG_RUN_ID
    stage_name = str(job.parameters["stage"])
    stage_directory = KDIAG_STAGE_DIRECTORY[stage_name]
    stage = root / stage_directory
    task = job.task
    if task == "kdiag-validate":
        return ()
    if task == "kdiag-prepare-targets":
        _payload, item = _json_evidence(
            root / "targets/index.json",
            root=run_root,
            label="K-diagnostic target index",
            complete=True,
        )
        return (item,)
    if task == "kdiag-build-bases":
        _payload, item = _json_evidence(
            root / "bases/index.json",
            root=run_root,
            label="K-diagnostic basis index",
            complete=True,
        )
        return (item,)
    if task == "kdiag-rotation-cache-preflight":
        _payload, item = _json_evidence(
            stage / "rotation_cache_build_preflight.json",
            root=run_root,
            label="rotation-cache preflight",
            expected={"stage": stage_directory, "approved": True},
        )
        return (item,)
    if task == "kdiag-prepare-rotations":
        _payload, item = _json_evidence(
            stage / "rotation_cache_index.json",
            root=run_root,
            label="rotation-cache index",
            complete=True,
            expected={"stage": stage_directory},
        )
        return (item,)
    if task == "kdiag-resource-preflight":
        _json_evidence(
            stage / "resource_preflight.json",
            root=run_root,
            label="K-diagnostic resource preflight",
            expected={"stage": stage_directory, "approved": True},
        )
        # The controller refreshes this operational gate after every bounded
        # wave.  Validate the current approval every time, but do not bind a
        # permanent job receipt to a deliberately mutable file hash.  The
        # receipt identity itself still binds task, stage, wave instance,
        # shard, commit, profile, and completed attempt.
        return ()
    if task == "kdiag-index":
        _payload, item = _json_evidence(
            stage / "index.json",
            root=run_root,
            label="K-diagnostic stage index",
            complete=True,
            expected={"stage": stage_directory},
        )
        return (item,)
    if task == "kdiag-report":
        report_root = run_root / KDIAG_REPORT_REVISION_R1
        index_payload, index_item = _json_evidence(
            report_root / "index.json",
            root=run_root,
            label="K-diagnostic derived report index",
            complete=True,
            expected={"identity": KDIAG_RUN_ID, "revision": "revision-r1"},
        )
        registered_files = index_payload.get("files")
        expected_files = {
            "decision_table",
            "aggregate_summary",
            "terminal_summary",
            "target_level_summary",
            "target_pair_summary",
            "report",
            "figure_data_manifest",
        }
        if not isinstance(registered_files, Mapping) or set(registered_files) != expected_files:
            raise ServerSchedulerAdapterError("K-diagnostic report index file set is malformed")
        package_evidence = [
            _descriptor_evidence(
                registered_files[name],
                root=report_root,
                label=f"K-diagnostic report {name}",
            )
            for name in sorted(expected_files)
        ]

        terminal_payload = strict_json_load(report_root / "terminal_summary.json")
        decision_payload = strict_json_load(report_root / "decision_table.json")
        if not isinstance(terminal_payload, Mapping) or len(terminal_payload) < 2:
            raise ServerSchedulerAdapterError("K-diagnostic terminal summary is empty")
        if not isinstance(decision_payload, Mapping) or not decision_payload:
            raise ServerSchedulerAdapterError("K-diagnostic decision table is empty")

        figure_payload = strict_json_load(report_root / "data/figure_data_manifest.json")
        if not isinstance(figure_payload, Mapping) or figure_payload.get("identity") != KDIAG_RUN_ID:
            raise ServerSchedulerAdapterError("K-diagnostic figure-data manifest is malformed")
        data_files = figure_payload.get("data_files")
        figure_files = figure_payload.get("figure_files")
        figure_sources = figure_payload.get("figure_sources")
        source_artifacts = figure_payload.get("source_artifacts")
        if (
            not isinstance(data_files, Mapping)
            or not data_files
            or not isinstance(figure_files, Mapping)
            or not figure_files
            or not isinstance(figure_sources, Mapping)
            or set(figure_sources) != set(figure_files)
            or not isinstance(source_artifacts, list)
            or not source_artifacts
        ):
            raise ServerSchedulerAdapterError("K-diagnostic figure-data manifest is incomplete")
        artifact_root = figure_payload.get("artifact_root")
        if not isinstance(artifact_root, str):
            raise ServerSchedulerAdapterError("K-diagnostic source artifact root is malformed")
        try:
            described_root = (report_root / artifact_root).resolve(strict=True)
            expected_root = (
                run_root / "artifacts/concept_intervention" / KDIAG_RUN_ID
            ).resolve(strict=True)
        except OSError as error:
            raise ServerSchedulerAdapterError(
                "K-diagnostic source artifact root is absent"
            ) from error
        if described_root != expected_root:
            raise ServerSchedulerAdapterError("K-diagnostic source artifact root is wrong")

        manifest_evidence: list[CompletionEvidence] = []
        for name in sorted(data_files):
            manifest_evidence.append(
                _descriptor_evidence(
                    data_files[name],
                    root=report_root,
                    label=f"K-diagnostic plot data {name}",
                )
            )
        for figure_name in sorted(figure_files):
            formats = figure_files[figure_name]
            if not isinstance(formats, Mapping) or not formats:
                raise ServerSchedulerAdapterError(
                    f"K-diagnostic figure {figure_name} descriptor is malformed"
                )
            for extension in sorted(formats):
                manifest_evidence.append(
                    _descriptor_evidence(
                        formats[extension],
                        root=report_root,
                        label=f"K-diagnostic figure {figure_name}.{extension}",
                    )
                )
        for position, descriptor in enumerate(source_artifacts):
            manifest_evidence.append(
                _descriptor_evidence(
                    descriptor,
                    root=expected_root,
                    label=f"K-diagnostic source artifact {position}",
                )
            )
        return (index_item, *package_evidence, *manifest_evidence)
    bundles_payload, bundles_item = _json_evidence(
        stage / "bundles.json",
        root=run_root,
        label="K-diagnostic bundle plan",
    )
    bundles = bundles_payload.get("bundles")
    index = int(job.parameters["shard_index"])
    if not isinstance(bundles, list) or index >= len(bundles):
        raise ServerSchedulerAdapterError("K-diagnostic bundle plan is incomplete")
    bundle = bundles[index]
    if not isinstance(bundle, Mapping) or not isinstance(bundle.get("bundle_id"), str):
        raise ServerSchedulerAdapterError("K-diagnostic bundle identity is malformed")
    bundle_id = str(bundle["bundle_id"])
    if task == "kdiag-microbenchmark":
        _payload, item = _json_evidence(
            stage / "microbenchmark.json",
            root=run_root,
            label="K-diagnostic microbenchmark",
            expected={
                "stage": stage_directory,
                "bundle_id": bundle_id,
                "bundle_index": index,
                "approved": True,
            },
        )
        _payload, bundle_item = _json_evidence(
            stage / "bundles" / bundle_id / "complete.json",
            root=run_root,
            label="K-diagnostic microbenchmark bundle completion",
            complete=True,
            expected={"physical_bundle_id": bundle_id},
        )
        return (bundles_item, item, bundle_item)
    if task == "kdiag-bundle":
        _payload, item = _json_evidence(
            stage / "bundles" / bundle_id / "complete.json",
            root=run_root,
            label="K-diagnostic bundle completion",
            complete=True,
            expected={"physical_bundle_id": bundle_id},
        )
        return (bundles_item, item)
    raise ServerSchedulerAdapterError(f"task {task!r} has no scientific completion contract")


def validate_task_completion(
    job: RunningJob,
    *,
    run_root: Path | None = None,
) -> tuple[CompletionEvidence, ...]:
    """Validate task-specific atomic scientific outputs after a zero exit."""

    selected_root = run_root or (
        THREE_METHOD_RUN_ROOT if job.spec.driver == "three-method" else KDIAG_RUN_ROOT
    )
    expected_root = THREE_METHOD_RUN_ROOT if job.spec.driver == "three-method" else KDIAG_RUN_ROOT
    if run_root is None and selected_root != expected_root:
        raise ServerSchedulerAdapterError("task output root is not project-owned")
    if job.spec.driver == "three-method":
        return _three_method_completion(job, selected_root)
    return _kdiag_completion(job, selected_root)


def completion_receipt_path(job: RunningJob, completion_root: Path) -> Path:
    """Return the job-identity-keyed project completion receipt path."""

    return completion_root / f"{job.job_id}.json"


def completion_identity(job: RunningJob) -> dict[str, object]:
    """Return immutable fields that bind a receipt to one scientific shard."""

    return {
        "job_id": job.job_id,
        "task": job.task,
        "run_id": job.parameters["run_id"],
        "git_commit": job.parameters["git_commit"],
        "shard_index": job.parameters["shard_index"],
        "shard_count": job.parameters["shard_count"],
        "stage": job.parameters.get("stage"),
        "instance": job.parameters.get("instance"),
        "execution_profile": job.execution_profile,
    }


def validate_completion_receipt(
    path: Path,
    job: RunningJob,
    evidence: Sequence[CompletionEvidence],
) -> None:
    """Revalidate receipt identity and every current scientific evidence hash."""

    try:
        metadata = path.lstat()
    except OSError as error:
        raise ServerSchedulerAdapterError(f"completion receipt is unreadable: {error}") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size <= 0
        or metadata.st_size > 1024 * 1024
    ):
        raise ServerSchedulerAdapterError(
            "completion receipt must be a small regular non-symlink file"
        )
    payload = strict_object(
        strict_json_load(path),
        {
            "schema_version",
            "marker_version",
            "complete",
            "completed_attempt",
            "identity",
            "evidence",
        },
        label="ServerScheduler completion receipt",
    )
    if (
        payload["schema_version"] != 2
        or payload["marker_version"] != "jlens-server-scheduler-completion-v2"
        or payload["complete"] is not True
        or payload["identity"] != completion_identity(job)
    ):
        raise ServerSchedulerAdapterError("ServerScheduler completion receipt identity is invalid")
    completed_attempt = integer(payload["completed_attempt"], label="completed_attempt", minimum=1)
    if completed_attempt > job.attempt:
        raise ServerSchedulerAdapterError("completion receipt comes from a future attempt")
    expected = [asdict(item) for item in evidence]
    if payload["evidence"] != expected:
        raise ServerSchedulerAdapterError("completion receipt evidence no longer matches outputs")


def write_completion_receipt(
    path: Path,
    job: RunningJob,
    evidence: Sequence[CompletionEvidence],
) -> None:
    """Atomically record and immediately revalidate a successful completion."""

    marker = {
        "schema_version": 2,
        "marker_version": "jlens-server-scheduler-completion-v2",
        "complete": True,
        "completed_attempt": job.attempt,
        "identity": completion_identity(job),
        "evidence": [asdict(item) for item in evidence],
    }
    atomic_write_json(path, marker)
    validate_completion_receipt(path, job, evidence)
