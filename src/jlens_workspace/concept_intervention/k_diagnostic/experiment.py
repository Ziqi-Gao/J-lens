"""Fail-closed scientific grid and shard execution for the K diagnostic.

This module is deliberately independent of model loading.  CLI code supplies
an identity-checked token-frame dictionary and already prepared targets; the
logic below owns the preregistered grid, null construction, persistence, and
completeness audit.  It is therefore fully testable without a model download.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jlens_workspace.artifacts import atomic_write_json, sha256_file
from jlens_workspace.concept_intervention.k_diagnostic.aggregation import (
    summarize_null_calibrated_curve,
)
from jlens_workspace.concept_intervention.k_diagnostic.nulls import (
    SHARED_DIRECTION_MAX_PEAK_BYTES,
    SHARED_DIRECTION_MIN_AVAILABLE_RESERVE_BYTES,
    SHARED_DIRECTION_PROVIDER_VERSION,
    SHARED_DIRECTION_STORAGE,
    SharedDirectionProvider,
    SharedDirectionRandomDictionary,
    inverse_rotate_target,
    load_cached_haar_rotation,
    matched_norm_subset,
    random_prefix_atoms,
    random_prefix_pursuit,
    shared_direction_memory_plan,
)
from jlens_workspace.config import ExperimentConfig
from jlens_workspace.pursuit import (
    MatchedNormRandomDictionary,
    streaming_nonnegative_pursuit,
)


class KDiagnosticExperimentError(RuntimeError):
    """The registered identity, grid, or saved result is inconsistent."""


STAGE_DIRECTORIES = {
    "pilot": "pilot",
    "raw_metric_full": "raw_metric_full",
    "transformed_metric": "transformed_metric",
}
REQUIRED_SHARD_FILES = (
    "summary.json",
    "real_reference.json",
    "null_errors.npy",
    "null_gains.npy",
    "null_metadata.json",
    "manifest.json",
)
REQUIRED_BUNDLE_REAL_FILES = (
    "errors_real.npy",
    "gains_real.npy",
    "support_real.npy",
    "coefficients_real.npz",
    "fixed_budget_real.json",
)
FIXED_BUDGETS = (1, 2, 4, 8, 16, 25, 32, 64)
EXECUTION_HARDWARE_GATE_VERSION = "kdiag_scheduled_single_cuda_v1"
EXECUTION_BACKENDS = frozenset({"fsm_local_scheduler", "slurm"})
NULL_EXECUTION_VERSION = "kdiag_null_batch_v1"
SHARED_DIRECTION_STABLE_KEYS = (
    "provider_version",
    "storage",
    "n_atoms",
    "d_model",
    "chunk_size",
    "float64_direction_storage_bytes",
    "generation_workspace_bytes",
    "peak_bytes",
    "hard_peak_budget_bytes",
    "reserved_available_bytes",
)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _payload_hash(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def validate_k_diagnostic_execution_hardware(payload: object) -> dict[str, object]:
    """Validate and normalize the exact visible CUDA hardware for a v2 run."""

    required_fields = {
        "schema_version",
        "gate_version",
        "execution_backend",
        "cuda_available",
        "gpu_count",
        "gpu_names",
        "visible_device_indices",
        "torch_version",
        "cuda_version",
    }
    if not isinstance(payload, Mapping) or set(payload) != required_fields:
        raise KDiagnosticExperimentError("v2 execution hardware record is missing or malformed")
    count = payload.get("gpu_count")
    names = payload.get("gpu_names")
    indices = payload.get("visible_device_indices")
    backend = payload.get("execution_backend")
    torch_version = payload.get("torch_version")
    cuda_version = payload.get("cuda_version")
    if (
        payload.get("schema_version") != 1
        or payload.get("gate_version") != EXECUTION_HARDWARE_GATE_VERSION
        or backend not in EXECUTION_BACKENDS
        or payload.get("cuda_available") is not True
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count != 1
        or not isinstance(names, list)
        or len(names) != 1
        or not all(isinstance(name, str) and name.strip() for name in names)
        or not isinstance(indices, list)
        or indices != [0]
        or not isinstance(torch_version, str)
        or not torch_version.strip()
        or not isinstance(cuda_version, str)
        or not cuda_version.strip()
    ):
        raise KDiagnosticExperimentError(
            "v2 execution requires one scheduled visible CUDA device and non-empty versions"
        )
    return {
        "schema_version": 1,
        "gate_version": EXECUTION_HARDWARE_GATE_VERSION,
        "execution_backend": backend,
        "cuda_available": True,
        "gpu_count": count,
        "gpu_names": list(names),
        "visible_device_indices": list(indices),
        "torch_version": torch_version,
        "cuda_version": cuda_version,
    }


def execution_hardware_sha256(payload: object) -> str:
    """Hash the normalized v2 hardware record; hostname is deliberately absent."""

    return _payload_hash(validate_k_diagnostic_execution_hardware(payload))


def detect_k_diagnostic_execution_hardware(
    torch_module: object | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Inspect visible CUDA devices and fail before any v2 scientific computation."""

    values = os.environ if env is None else env
    lease_id = values.get("JLENS_LOCAL_LEASE_ID", "").strip()
    slurm_job_id = values.get("SLURM_JOB_ID", "").strip()
    if bool(lease_id) == bool(slurm_job_id):
        raise KDiagnosticExperimentError(
            "v2 execution requires exactly one local-scheduler or Slurm allocation"
        )
    if lease_id:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", lease_id):
            raise KDiagnosticExperimentError("local scheduler lease ID is malformed")
        runtime_root = values.get("JLENS_LOCAL_RUNTIME_ROOT", "").strip()
        if not runtime_root:
            raise KDiagnosticExperimentError("local scheduler runtime root is absent")
        lease_path = Path(runtime_root) / "scheduler" / "leases" / f"{lease_id}.json"
        try:
            lease = _read_json(lease_path)
        except (FileNotFoundError, KDiagnosticExperimentError) as exc:
            raise KDiagnosticExperimentError("local scheduler lease is absent or invalid") from exc
        if (
            lease.get("lease_id") != lease_id
            or lease.get("kind") != "gpu"
            or not isinstance(lease.get("gpu_index"), int)
            or isinstance(lease.get("gpu_index"), bool)
        ):
            raise KDiagnosticExperimentError("local scheduler GPU lease is invalid")
        execution_backend = "fsm_local_scheduler"
    else:
        if not re.fullmatch(r"[0-9]+", slurm_job_id):
            raise KDiagnosticExperimentError("Slurm allocation ID is malformed")
        execution_backend = "slurm"
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise KDiagnosticExperimentError(
                "v2 execution requires an installed CUDA torch"
            ) from exc
    cuda = getattr(torch_module, "cuda", None)
    cuda_available = bool(cuda is not None and cuda.is_available())
    count = int(cuda.device_count()) if cuda_available else 0
    names = [str(cuda.get_device_name(index)) for index in range(count)]
    version = getattr(torch_module, "version", None)
    payload = {
        "schema_version": 1,
        "gate_version": EXECUTION_HARDWARE_GATE_VERSION,
        "execution_backend": execution_backend,
        "cuda_available": cuda_available,
        "gpu_count": count,
        "gpu_names": names,
        "visible_device_indices": list(range(count)),
        "torch_version": str(getattr(torch_module, "__version__", "")),
        "cuda_version": str(getattr(version, "cuda", "") or ""),
    }
    return validate_k_diagnostic_execution_hardware(payload)


def _array_hash(values: object) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _stable_shared_direction_plan(
    plan: Mapping[str, object],
) -> dict[str, object]:
    return {key: plan.get(key) for key in SHARED_DIRECTION_STABLE_KEYS}


def _approved_shared_direction_plan(
    plan: object,
    *,
    n_atoms: int,
    d_model: int,
    chunk_size: int,
) -> bool:
    if (
        not isinstance(plan, Mapping)
        or not isinstance(n_atoms, int)
        or isinstance(n_atoms, bool)
        or not isinstance(d_model, int)
        or isinstance(d_model, bool)
        or not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or n_atoms < 1
        or d_model < 1
        or chunk_size < 1
    ):
        return False
    checks = plan.get("checks")
    observed = plan.get("observed_available_bytes")
    effective = plan.get("effective_available_budget_bytes")
    rows = min(n_atoms, chunk_size)
    float64_bytes = np.dtype(np.float64).itemsize
    expected_storage = n_atoms * d_model * float64_bytes
    expected_workspace = 2 * rows * d_model * float64_bytes + rows * float64_bytes
    expected_peak = expected_storage + expected_workspace
    integer_fields = (
        "n_atoms",
        "d_model",
        "chunk_size",
        "float64_direction_storage_bytes",
        "generation_workspace_bytes",
        "peak_bytes",
        "hard_peak_budget_bytes",
        "observed_available_bytes",
        "reserved_available_bytes",
        "effective_available_budget_bytes",
    )
    if any(
        not isinstance(plan.get(field), int) or isinstance(plan.get(field), bool)
        for field in integer_fields
    ):
        return False
    if (
        plan.get("schema_version") != 1
        or plan.get("provider_version") != SHARED_DIRECTION_PROVIDER_VERSION
        or plan.get("storage") != SHARED_DIRECTION_STORAGE
        or plan.get("n_atoms") != n_atoms
        or plan.get("d_model") != d_model
        or plan.get("chunk_size") != chunk_size
        or plan.get("float64_direction_storage_bytes") != expected_storage
        or plan.get("generation_workspace_bytes") != expected_workspace
        or plan.get("peak_bytes") != expected_peak
        or plan.get("approved") is not True
        or not isinstance(checks, Mapping)
        or checks.get("within_hard_peak_budget") is not True
        or checks.get("preserves_available_ram_reserve") is not True
        or observed <= SHARED_DIRECTION_MIN_AVAILABLE_RESERVE_BYTES
        or effective != observed - SHARED_DIRECTION_MIN_AVAILABLE_RESERVE_BYTES
        or expected_peak > effective
        or expected_peak > SHARED_DIRECTION_MAX_PEAK_BYTES
        or plan.get("hard_peak_budget_bytes") != SHARED_DIRECTION_MAX_PEAK_BYTES
        or plan.get("reserved_available_bytes") != SHARED_DIRECTION_MIN_AVAILABLE_RESERVE_BYTES
    ):
        return False
    return True


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise KDiagnosticExperimentError(f"cannot read valid JSON: {path}") from error


def _atomic_save_npy(path: Path, values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.asarray(values), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_save_npz(path: Path, arrays: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        with Path(temporary).open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_json_exact(path: Path, payload: object) -> None:
    """Create an immutable JSON identity, or verify an exact existing copy."""

    if path.exists():
        observed = _read_json(path)
        if _canonical_json(observed) != _canonical_json(payload):
            raise KDiagnosticExperimentError(
                f"existing artifact identity does not match registered payload: {path}"
            )
        return
    atomic_write_json(path, payload)


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def artifact_root(config: ExperimentConfig, *, run_root: str | Path = ".") -> Path:
    diagnostic = config.k_diagnostic
    if diagnostic is None:
        raise KDiagnosticExperimentError("configuration has no k_diagnostic section")
    return _resolve(Path(run_root), diagnostic.artifact_root)


def stage_root(config: ExperimentConfig, *, run_root: str | Path = ".") -> Path:
    diagnostic = config.k_diagnostic
    if diagnostic is None:
        raise KDiagnosticExperimentError("configuration has no k_diagnostic section")
    return artifact_root(config, run_root=run_root) / STAGE_DIRECTORIES[diagnostic.stage]


def validate_shared_artifacts(
    config: ExperimentConfig, *, run_root: str | Path = "."
) -> dict[str, str]:
    """Hash every registered shared input; never infer or repair a mismatch."""

    diagnostic = config.k_diagnostic
    if diagnostic is None:
        raise KDiagnosticExperimentError("configuration has no k_diagnostic section")
    shared = _resolve(Path(run_root), diagnostic.shared_artifact_root)
    if not shared.is_dir():
        raise KDiagnosticExperimentError(f"shared artifact root is missing: {shared}")
    verified: dict[str, str] = {}
    for relative, expected in sorted(diagnostic.shared_hashes.items()):
        source = shared / relative
        if not source.is_file():
            raise KDiagnosticExperimentError(f"registered shared artifact is missing: {source}")
        observed = sha256_file(source)
        if observed != expected:
            raise KDiagnosticExperimentError(
                f"shared artifact hash mismatch for {source}: "
                f"expected {expected}, observed {observed}"
            )
        verified[relative] = observed
    source_test = _resolve(Path(run_root), diagnostic.source_test_jsonl)
    if not source_test.is_file():
        raise KDiagnosticExperimentError(f"registered source test JSONL is missing: {source_test}")
    observed_test = sha256_file(source_test)
    if observed_test != diagnostic.source_test_sha256:
        raise KDiagnosticExperimentError(
            f"source test JSONL hash mismatch for {source_test}: expected "
            f"{diagnostic.source_test_sha256}, observed {observed_test}"
        )
    verified["source_test_jsonl"] = observed_test
    selection_index = _read_json(shared / "selection" / "index.json")
    if selection_index.get("complete") is not True or selection_index.get("probe_count") != 49:
        raise KDiagnosticExperimentError("shared layer-selection index is incomplete")
    probes = _read_json(shared / "selection" / "raptor_probes" / "manifest.json")
    probe_keys = {
        (int(entry["layer"]), str(entry["concept_id"])) for entry in probes.get("probes", [])
    }
    expected_probe_keys = {
        (layer, concept)
        for layer in diagnostic.candidate_layers
        for concept in diagnostic.concept_ids
    }
    if probe_keys != expected_probe_keys:
        raise KDiagnosticExperimentError("shared primary-probe manifest is not the 7x7 grid")
    activation = _read_json(shared / "activations" / "metadata.json")
    if (
        activation.get("coordinate") != "resid_post"
        or activation.get("representation") != "last_non_padding_token"
        or activation.get("layers") != diagnostic.candidate_layers
        or activation.get("label_shape") != [53861, 7]
        or activation.get("manifest", {}).get("model_revision") != config.model.revision
    ):
        raise KDiagnosticExperimentError("shared activation identity is inconsistent")
    return verified


def _invariant_manifest(config: ExperimentConfig) -> dict[str, Any]:
    diagnostic = config.k_diagnostic
    assert diagnostic is not None
    return {
        "schema_version": 1,
        "identity": diagnostic.identity,
        "artifact_root": diagnostic.artifact_root,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "tokenizer_id": config.model.tokenizer_id,
        "tokenizer_revision": config.model.tokenizer_revision,
        "lens_target_layer": config.lens.target_layer if config.lens else None,
        "candidate_layers": diagnostic.candidate_layers,
        "capture_point": "resid_post",
        "convention": diagnostic.convention,
        "shared_artifact_root": diagnostic.shared_artifact_root,
        "shared_hashes": dict(sorted(diagnostic.shared_hashes.items())),
        "source_test_jsonl": diagnostic.source_test_jsonl,
        "source_test_sha256": diagnostic.source_test_sha256,
        "prohibitions": ["generation", "llm_judge", "shared_artifact_mutation"],
    }


def _stage_manifest(config: ExperimentConfig) -> dict[str, Any]:
    diagnostic = config.k_diagnostic
    assert diagnostic is not None
    payload = diagnostic.model_dump(mode="json")
    return {
        "schema_version": 1,
        "identity": diagnostic.identity,
        "stage": diagnostic.stage,
        "configuration": payload,
        "configuration_sha256": _payload_hash(payload),
    }


def initialize_artifacts(config: ExperimentConfig, *, run_root: str | Path = ".") -> dict[str, Any]:
    """Verify inputs, enforce stage gates, and create immutable manifests."""

    verified = validate_shared_artifacts(config, run_root=run_root)
    root = artifact_root(config, run_root=run_root)
    current_stage = stage_root(config, run_root=run_root)
    diagnostic = config.k_diagnostic
    assert diagnostic is not None
    if diagnostic.stage == "raw_metric_full":
        _require_complete_index(root / "pilot" / "index.json")
    elif diagnostic.stage == "transformed_metric":
        _require_complete_index(root / "raw_metric_full" / "index.json")
    root.mkdir(parents=True, exist_ok=True)
    current_stage.mkdir(parents=True, exist_ok=True)
    _write_json_exact(root / "manifest.json", _invariant_manifest(config))
    _write_json_exact(current_stage / "manifest.json", _stage_manifest(config))
    return {
        "artifact_root": str(root),
        "stage_root": str(current_stage),
        "verified_shared_hashes": verified,
    }


def _require_complete_index(path: Path) -> None:
    if not path.is_file():
        raise KDiagnosticExperimentError(f"required prior-stage index is missing: {path}")
    payload = _read_json(path)
    if payload.get("complete") is not True:
        raise KDiagnosticExperimentError(f"required prior-stage index is incomplete: {path}")


@dataclass(frozen=True)
class ScientificShard:
    """One and only one registered target/metric/null realization."""

    target_id: str
    layer: int
    target_family: str
    target_subtype: str
    metric: str
    null_family: str
    null_seed: int
    dictionary_fraction: float

    @property
    def identity(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def shard_id(self) -> str:
        return _payload_hash(self.identity)[:24]


@dataclass(frozen=True)
class PhysicalBundle:
    """One dictionary/target solve shared by all registered null replicates."""

    target_id: str
    layer: int
    target_family: str
    target_subtype: str
    metric: str
    shard_ids: tuple[str, ...]

    @property
    def scientific_identity(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "layer": self.layer,
            "target_family": self.target_family,
            "target_subtype": self.target_subtype,
            "metric": self.metric,
        }

    @property
    def bundle_id(self) -> str:
        return _payload_hash(self.scientific_identity)[:24]


@dataclass(frozen=True)
class PrecomputedNull:
    """One bundle-computed null result and its immutable execution identity."""

    result: Any
    metadata: Mapping[str, object]
    shared_null_computation_id: str
    source_curve_sha256: str
    null_batch_computation_id: str


def build_physical_bundles(grid: Sequence[ScientificShard]) -> list[PhysicalBundle]:
    """Group logical null replicates without changing their scientific identities."""

    grouped: dict[tuple[object, ...], list[ScientificShard]] = defaultdict(list)
    for shard in grid:
        grouped[
            (
                shard.target_id,
                shard.layer,
                shard.target_family,
                shard.target_subtype,
                shard.metric,
            )
        ].append(shard)
    bundles = [
        PhysicalBundle(
            target_id=str(key[0]),
            layer=int(key[1]),
            target_family=str(key[2]),
            target_subtype=str(key[3]),
            metric=str(key[4]),
            shard_ids=tuple(sorted(shard.shard_id for shard in shards)),
        )
        for key, shards in grouped.items()
    ]
    if sum(len(bundle.shard_ids) for bundle in bundles) != len(grid):
        raise KDiagnosticExperimentError("physical bundle grouping lost logical replicates")
    return sorted(bundles, key=lambda bundle: _canonical_json(bundle.scientific_identity))


def load_target_index(
    config: ExperimentConfig, *, run_root: str | Path = "."
) -> list[dict[str, Any]]:
    """Load prepared targets and re-hash every vector before grid creation."""

    root = artifact_root(config, run_root=run_root)
    path = root / "targets" / "index.json"
    payload = _read_json(path)
    if payload.get("complete") is not True or not isinstance(payload.get("targets"), list):
        raise KDiagnosticExperimentError(f"target index is not complete: {path}")
    seen: set[str] = set()
    targets: list[dict[str, Any]] = []
    for entry in payload["targets"]:
        target_id = entry.get("target_id")
        if not isinstance(target_id, str) or target_id in seen:
            raise KDiagnosticExperimentError("target index contains missing/duplicate target IDs")
        vector_path = root / entry["vector_path"]
        observed = sha256_file(vector_path)
        if observed != entry.get("file_sha256"):
            raise KDiagnosticExperimentError(f"prepared target file hash mismatch: {vector_path}")
        vector = np.load(vector_path, allow_pickle=False)
        if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise KDiagnosticExperimentError(f"invalid prepared target vector: {vector_path}")
        template_payload = entry.get("per_template_contrasts")
        if entry.get("target_family") == "seven_emotion_label_contrast":
            if not isinstance(template_payload, Mapping):
                raise KDiagnosticExperimentError(
                    f"contrast target lacks per-template artifact identity: {target_id}"
                )
            template_path = root / str(template_payload.get("path"))
            if template_payload.get("dtype") != "float64" or template_payload.get(
                "file_sha256"
            ) != sha256_file(template_path):
                raise KDiagnosticExperimentError(
                    f"per-template contrast file hash mismatch: {template_path}"
                )
            templates = np.asarray(np.load(template_path, allow_pickle=False), dtype=np.float64)
            if (
                templates.ndim != 2
                or list(templates.shape) != template_payload.get("shape")
                or templates.shape[1] != vector.size
                or not np.isfinite(templates).all()
                or _array_hash(templates) != template_payload.get("raw_float64_sha256")
                or not np.allclose(
                    np.mean(templates, axis=0, dtype=np.float64),
                    vector,
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                raise KDiagnosticExperimentError(
                    f"invalid per-template contrast matrix: {template_path}"
                )
        elif template_payload is not None:
            raise KDiagnosticExperimentError(
                f"non-contrast target unexpectedly owns template contrasts: {target_id}"
            )
        seen.add(target_id)
        targets.append(dict(entry))
    return targets


def _target_is_registered(entry: Mapping[str, Any], config: ExperimentConfig) -> bool:
    diagnostic = config.k_diagnostic
    assert diagnostic is not None
    layer = int(entry["layer"])
    family = str(entry["target_family"])
    if family not in diagnostic.target_families:
        return False
    if family == "raw_activation":
        return layer in diagnostic.raw_activation_layers
    return layer in diagnostic.analysis_layers


def _selected_target_ids(
    config: ExperimentConfig, targets: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Select a stage cohort using persisted ranks, never global ID ordering."""

    diagnostic = config.k_diagnostic
    assert diagnostic is not None
    selected: list[str] = []
    raw_counts: dict[int, int] = defaultdict(int)
    for entry in targets:
        if not _target_is_registered(entry, config):
            continue
        if entry["target_family"] == "raw_activation":
            metadata = entry.get("metadata")
            rank = metadata.get("selection_rank") if isinstance(metadata, Mapping) else None
            memberships = (
                metadata.get("stage_membership") if isinstance(metadata, Mapping) else None
            )
            if not isinstance(rank, int) or not isinstance(memberships, list):
                raise KDiagnosticExperimentError(
                    "raw activation targets require persisted selection_rank/stage_membership"
                )
            if diagnostic.stage not in memberships:
                continue
            if rank >= diagnostic.raw_activation_targets_per_layer:
                continue
            raw_counts[int(entry["layer"])] += 1
        selected.append(str(entry["target_id"]))
    if "raw_activation" in diagnostic.target_families:
        for layer in diagnostic.raw_activation_layers:
            observed = raw_counts[layer]
            if observed != diagnostic.raw_activation_targets_per_layer:
                raise KDiagnosticExperimentError(
                    f"layer {layer} has {observed} ranked raw targets; expected "
                    f"{diagnostic.raw_activation_targets_per_layer}"
                )
    if len(selected) != len(set(selected)):
        raise KDiagnosticExperimentError("stage target membership contains duplicate IDs")
    return sorted(selected)


def seal_target_membership(
    config: ExperimentConfig,
    targets: Sequence[Mapping[str, Any]],
    *,
    run_root: str | Path = ".",
) -> dict[str, Any]:
    """Create or verify the immutable target cohort for one stage."""

    diagnostic = config.k_diagnostic
    assert diagnostic is not None
    selected_ids = _selected_target_ids(config, targets)
    by_id = {str(entry["target_id"]): entry for entry in targets}
    payload = {
        "schema_version": 2,
        "identity": diagnostic.identity,
        "stage": diagnostic.stage,
        "selection_rule": "persisted selection_rank and explicit stage_membership",
        "target_count": len(selected_ids),
        "targets": [
            {
                "target_id": target_id,
                "file_sha256": by_id[target_id]["file_sha256"],
                "layer": int(by_id[target_id]["layer"]),
                "target_family": str(by_id[target_id]["target_family"]),
                "target_subtype": str(by_id[target_id]["target_subtype"]),
            }
            for target_id in selected_ids
        ],
    }
    path = (
        artifact_root(config, run_root=run_root) / "target_membership" / f"{diagnostic.stage}.json"
    )
    _write_json_exact(path, payload)
    return payload


def build_scientific_grid(
    config: ExperimentConfig,
    targets: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] | Mapping[int, Sequence[str]] | None = None,
) -> list[ScientificShard]:
    """Expand the preregistered grid deterministically and reject empty cells."""

    diagnostic = config.k_diagnostic
    if diagnostic is None:
        raise KDiagnosticExperimentError("configuration has no k_diagnostic section")
    selected_ids = set(_selected_target_ids(config, targets))
    selected = [entry for entry in targets if str(entry["target_id"]) in selected_ids]
    present_families = {str(entry["target_family"]) for entry in selected}
    missing_families = set(diagnostic.target_families) - present_families
    if missing_families:
        raise KDiagnosticExperimentError(
            f"prepared targets lack registered families: {sorted(missing_families)}"
        )
    grid: list[ScientificShard] = []
    for entry in sorted(selected, key=lambda item: str(item["target_id"])):
        base = {
            "target_id": str(entry["target_id"]),
            "layer": int(entry["layer"]),
            "target_family": str(entry["target_family"]),
            "target_subtype": str(entry["target_subtype"]),
        }
        if isinstance(metrics, Mapping):
            entry_metrics = metrics.get(base["layer"])
            if not entry_metrics:
                raise KDiagnosticExperimentError(
                    f"no active metrics registered for layer {base['layer']}"
                )
        else:
            entry_metrics = metrics or diagnostic.metrics
        for metric in entry_metrics:
            for family in diagnostic.null_families:
                if family == "label_permutation_probe":
                    if base["target_family"] not in {
                        "logistic_probe",
                        "class_mean_difference",
                    }:
                        continue
                    count = (
                        diagnostic.permutation_count_layer_27
                        if base["layer"] == 27
                        else diagnostic.permutation_count_other_layers
                    )
                    seed_fractions = [
                        (diagnostic.permutation_seed_start + offset, 1.0) for offset in range(count)
                    ]
                elif family == "orthogonal_rotation":
                    seed_fractions = [(seed, 1.0) for seed in diagnostic.rotation_seeds]
                elif family == "iid_cardinality_sweep":
                    seed_fractions = [
                        (seed, fraction)
                        for fraction in diagnostic.cardinality_fractions
                        for seed in diagnostic.iid_seeds
                    ]
                else:
                    seed_fractions = [(seed, 1.0) for seed in diagnostic.iid_seeds]
                for seed, fraction in seed_fractions:
                    grid.append(
                        ScientificShard(
                            **base,
                            metric=metric,
                            null_family=family,
                            null_seed=int(seed),
                            dictionary_fraction=float(fraction),
                        )
                    )
    identities = [_canonical_json(item.identity) for item in grid]
    if not grid or len(set(identities)) != len(identities):
        raise KDiagnosticExperimentError("scientific grid is empty or contains duplicates")
    return sorted(grid, key=lambda item: _canonical_json(item.identity))


def write_scientific_grid(
    config: ExperimentConfig,
    targets: Sequence[Mapping[str, Any]],
    *,
    run_root: str | Path = ".",
) -> list[ScientificShard]:
    initialize_artifacts(config, run_root=run_root)
    active_metrics: Sequence[str] | Mapping[int, Sequence[str]] | None = None
    metric_dimensions: dict[int, dict[str, int]] = {}
    if config.k_diagnostic and config.k_diagnostic.stage == "transformed_metric":
        basis_index_path = artifact_root(config, run_root=run_root) / "bases" / "index.json"
        basis_index = _read_json(basis_index_path)
        if basis_index.get("complete") is not True:
            raise KDiagnosticExperimentError(f"basis index is incomplete: {basis_index_path}")
        active_metrics = {
            int(layer): [str(value) for value in values]
            for layer, values in basis_index["active_metrics_by_layer"].items()
        }
        raw_dimensions = basis_index.get("metric_dimensions_by_layer")
        if not isinstance(raw_dimensions, Mapping):
            raise KDiagnosticExperimentError(
                "basis index lacks registered metric output dimensions"
            )
        metric_dimensions = {
            int(layer): {str(metric): int(value) for metric, value in values.items()}
            for layer, values in raw_dimensions.items()
        }
    membership = seal_target_membership(config, targets, run_root=run_root)
    member_ids = {str(entry["target_id"]) for entry in membership["targets"]}
    members = [entry for entry in targets if str(entry["target_id"]) in member_ids]
    grid = build_scientific_grid(config, members, metrics=active_metrics)
    bundles = build_physical_bundles(grid)
    diagnostic = config.k_diagnostic
    assert diagnostic is not None
    if len(grid) > diagnostic.max_logical_replicates:
        raise KDiagnosticExperimentError(
            f"logical grid {len(grid)} exceeds hard budget {diagnostic.max_logical_replicates}"
        )
    if len(bundles) > diagnostic.max_physical_bundles:
        raise KDiagnosticExperimentError(
            f"physical bundles {len(bundles)} exceed hard budget {diagnostic.max_physical_bundles}"
        )
    raw_dimension_values = {
        int(
            np.load(
                artifact_root(config, run_root=run_root) / entry["vector_path"],
                mmap_mode="r",
                allow_pickle=False,
            ).shape[0]
        )
        for entry in members
    }
    if len(raw_dimension_values) != 1:
        raise KDiagnosticExperimentError("registered targets do not share one raw dimension")
    raw_dimension = raw_dimension_values.pop()

    def metric_dimension(bundle: PhysicalBundle) -> int:
        if bundle.metric == "raw_euclidean":
            return raw_dimension
        try:
            return metric_dimensions[bundle.layer][bundle.metric]
        except KeyError as error:
            raise KDiagnosticExperimentError(
                f"missing metric dimension for layer={bundle.layer}, metric={bundle.metric}"
            ) from error

    destination = stage_root(config, run_root=run_root) / "grid.json"
    _write_json_exact(
        destination,
        {
            "schema_version": 1,
            "stage": config.k_diagnostic.stage if config.k_diagnostic else None,
            "count": len(grid),
            "shards": [item.identity | {"shard_id": item.shard_id} for item in grid],
        },
    )
    _write_json_exact(
        stage_root(config, run_root=run_root) / "bundles.json",
        {
            "schema_version": 2,
            "stage": config.k_diagnostic.stage if config.k_diagnostic else None,
            "logical_replicate_count": len(grid),
            "physical_bundle_count": len(bundles),
            "bundles": [
                bundle.scientific_identity
                | {
                    "bundle_id": bundle.bundle_id,
                    "logical_shard_count": len(bundle.shard_ids),
                    "metric_dimension": metric_dimension(bundle),
                    "conservative_cost_units": int(
                        diagnostic.k_max * metric_dimension(bundle) * (1 + len(bundle.shard_ids))
                    ),
                    "shard_ids": list(bundle.shard_ids),
                }
                for bundle in bundles
            ],
        },
    )
    _write_json_exact(
        stage_root(config, run_root=run_root) / "resource_plan.json",
        {
            "schema_version": 2,
            "stage": diagnostic.stage,
            "logical_replicates": len(grid),
            "physical_bundles": len(bundles),
            "process_reduction_factor": float(len(grid) / len(bundles)),
            "projected_disk_gib_conservative": None,
            "disk_budget_gib": diagnostic.max_estimated_disk_gib,
            "projected_inodes_conservative": None,
            "inode_budget": diagnostic.max_estimated_inodes,
            "gpu_hours": None,
            "gpu_hours_status": "requires current-stage representative microbenchmark",
            "disk_inode_status": "requires measured bundle and logical-shard artifacts",
            "max_seconds_per_bundle": diagnostic.max_seconds_per_bundle,
            "max_gpu_hours": diagnostic.max_estimated_gpu_hours,
            "slurm_array_chunk_size": diagnostic.slurm_array_chunk_size,
        },
    )
    return grid


REGISTERED_STAGE_RESOURCE_BUDGETS = {
    "pilot": {"gpu_hours": 500.0, "disk_gib": 25.0, "inodes": 500_000},
    "raw_metric_full": {
        "gpu_hours": 2500.0,
        "disk_gib": 75.0,
        "inodes": 2_000_000,
    },
    "transformed_metric": {
        "gpu_hours": 3000.0,
        "disk_gib": 75.0,
        "inodes": 2_000_000,
    },
}


def _effective_rotation_cache_budgets(diagnostic: Any) -> dict[str, float | int]:
    other_stages = [
        value
        for stage, value in REGISTERED_STAGE_RESOURCE_BUDGETS.items()
        if stage != diagnostic.stage
    ]
    other_disk_gib = float(sum(value["disk_gib"] for value in other_stages))
    other_inodes = int(sum(value["inodes"] for value in other_stages))
    full_remaining_disk_gib = float(diagnostic.max_total_estimated_disk_gib - other_disk_gib)
    full_remaining_inodes = int(diagnostic.max_total_estimated_inodes - other_inodes)
    return {
        "other_stages_reserved_disk_gib": other_disk_gib,
        "other_stages_reserved_inodes": other_inodes,
        "full_remaining_disk_gib": full_remaining_disk_gib,
        "full_remaining_inodes": full_remaining_inodes,
        "effective_cache_gib_budget": min(
            float(diagnostic.max_rotation_cache_gib),
            float(diagnostic.max_estimated_disk_gib),
            full_remaining_disk_gib,
        ),
        "effective_cache_inode_budget": min(
            int(diagnostic.max_estimated_inodes), full_remaining_inodes
        ),
    }


def _tree_usage(path: Path) -> dict[str, int]:
    if not path.is_dir():
        raise KDiagnosticExperimentError(f"artifact directory is missing: {path}")
    byte_count = 0
    inode_count = 1
    for directory, directories, files in os.walk(path):
        inode_count += len(directories) + len(files)
        for name in files:
            byte_count += (Path(directory) / name).stat().st_size
    return {"bytes": byte_count, "inodes": inode_count}


def measure_bundle_artifacts(
    config: ExperimentConfig,
    *,
    bundle_id: str,
    run_root: str | Path = ".",
) -> dict[str, Any]:
    """Measure one completed bundle and its logical shards for projection."""

    root = stage_root(config, run_root=run_root)
    bundles = _read_json(root / "bundles.json")
    matches = [entry for entry in bundles["bundles"] if entry["bundle_id"] == bundle_id]
    if len(matches) != 1:
        raise KDiagnosticExperimentError(f"bundle is not registered exactly once: {bundle_id}")
    entry = matches[0]
    bundle_usage = _tree_usage(root / "bundles" / bundle_id)
    shard_usage = [_tree_usage(root / "shards" / str(shard_id)) for shard_id in entry["shard_ids"]]
    if not shard_usage:
        raise KDiagnosticExperimentError("benchmark bundle has no logical shards")
    return {
        "bundle_bytes": bundle_usage["bytes"],
        "bundle_inodes": bundle_usage["inodes"],
        "logical_shard_count": len(shard_usage),
        "max_logical_shard_bytes": max(value["bytes"] for value in shard_usage),
        "max_logical_shard_inodes": max(value["inodes"] for value in shard_usage),
        "measured_logical_shard_bytes": sum(value["bytes"] for value in shard_usage),
        "measured_logical_shard_inodes": sum(value["inodes"] for value in shard_usage),
    }


def record_bundle_runtime(
    config: ExperimentConfig,
    *,
    bundle_id: str,
    wall_seconds: float,
    execution_hardware: object | None = None,
    run_root: str | Path = ".",
) -> dict[str, Any]:
    """Persist immutable observed elapsed time, including rejected over-budget runs."""

    diagnostic = config.k_diagnostic
    if diagnostic is None or not np.isfinite(wall_seconds) or wall_seconds <= 0.0:
        raise KDiagnosticExperimentError("bundle runtime is invalid")
    payload = {
        "schema_version": 1,
        "identity": diagnostic.identity,
        "stage": diagnostic.stage,
        "bundle_id": bundle_id,
        "configuration_sha256": _stage_manifest(config)["configuration_sha256"],
        "wall_seconds": float(wall_seconds),
        "max_seconds_per_bundle": diagnostic.max_seconds_per_bundle,
        "within_hard_limit": bool(wall_seconds <= diagnostic.max_seconds_per_bundle),
    }
    if diagnostic.identity == "qwen35_4b_k_diagnostic_v2":
        hardware = validate_k_diagnostic_execution_hardware(execution_hardware)
        payload["schema_version"] = 3
        payload["null_execution_version"] = NULL_EXECUTION_VERSION
        payload["execution_backend"] = hardware["execution_backend"]
        payload["execution_hardware"] = hardware
        payload["execution_hardware_sha256"] = execution_hardware_sha256(hardware)
    path = stage_root(config, run_root=run_root) / "runtimes" / f"{bundle_id}.json"
    if path.is_file():
        observed = _read_json(path)
        identity_fields = [
            "identity",
            "stage",
            "bundle_id",
            "configuration_sha256",
            "max_seconds_per_bundle",
        ]
        if diagnostic.identity == "qwen35_4b_k_diagnostic_v2":
            identity_fields.extend(
                (
                    "null_execution_version",
                    "execution_backend",
                    "execution_hardware",
                    "execution_hardware_sha256",
                )
            )
        for field in identity_fields:
            if observed.get(field) != payload[field]:
                raise KDiagnosticExperimentError(f"existing runtime identity mismatch: {path}")
        return observed
    _write_json_exact(path, payload)
    return payload


def resource_preflight(config: ExperimentConfig, *, run_root: str | Path = ".") -> dict[str, Any]:
    """Fail closed before sbatch using current-stage measured worst-case costs."""

    diagnostic = config.k_diagnostic
    if diagnostic is None:
        raise KDiagnosticExperimentError("configuration has no k_diagnostic section")
    root = stage_root(config, run_root=run_root)
    bundles = _read_json(root / "bundles.json")
    benchmark = _read_json(root / "microbenchmark.json")
    rotation = _read_json(root / "rotation_cache_index.json")
    rotation_build = _read_json(root / "rotation_cache_build_preflight.json")
    configuration_sha256 = _stage_manifest(config)["configuration_sha256"]
    is_v2 = diagnostic.identity == "qwen35_4b_k_diagnostic_v2"
    registered_bundles = bundles.get("bundles")
    if not isinstance(registered_bundles, list) or not registered_bundles:
        raise KDiagnosticExperimentError("resource preflight has no registered bundles")
    worst = max(
        registered_bundles,
        key=lambda entry: int(entry["conservative_cost_units"]),
    )
    if (
        benchmark.get("approved") is not True
        or (is_v2 and benchmark.get("schema_version") != 5)
        or benchmark.get("identity") != diagnostic.identity
        or benchmark.get("stage") != diagnostic.stage
        or (is_v2 and benchmark.get("null_execution_version") != NULL_EXECUTION_VERSION)
        or benchmark.get("configuration_sha256") != configuration_sha256
        or benchmark.get("bundle_id") != worst["bundle_id"]
        or rotation.get("complete") is not True
        or rotation.get("identity") != diagnostic.identity
        or rotation.get("stage") != diagnostic.stage
        or rotation.get("configuration_sha256") != configuration_sha256
        or rotation_build.get("approved") is not True
        or rotation_build.get("identity") != diagnostic.identity
        or rotation_build.get("stage") != diagnostic.stage
        or rotation_build.get("configuration_sha256") != configuration_sha256
        or rotation.get("binding_plan_sha256") != rotation_build.get("binding_plan_sha256")
        or rotation.get("build_preflight_sha256")
        != sha256_file(root / "rotation_cache_build_preflight.json")
    ):
        raise KDiagnosticExperimentError(
            "current-stage benchmark/rotation index is absent or identity-mismatched"
        )
    memory_plan: Mapping[str, object] | None = None
    execution_hardware: dict[str, object] | None = None
    if is_v2:
        execution_hardware = validate_k_diagnostic_execution_hardware(
            benchmark.get("execution_hardware")
        )
        if (
            benchmark.get("execution_hardware_sha256")
            != execution_hardware_sha256(execution_hardware)
            or benchmark.get("execution_backend") != execution_hardware["execution_backend"]
        ):
            raise KDiagnosticExperimentError(
                "v2 microbenchmark execution hardware hash is absent or mismatched"
            )
        candidate = benchmark.get("shared_direction_memory_plan")
        bundle_manifest = _read_json(root / "bundles" / str(worst["bundle_id"]) / "manifest.json")
        if not isinstance(candidate, Mapping):
            raise KDiagnosticExperimentError("v2 microbenchmark lacks shared-direction memory plan")
        bundle_plan = bundle_manifest.get("shared_direction_memory_plan")
        stable_candidate = _stable_shared_direction_plan(candidate)
        stable_bundle = (
            _stable_shared_direction_plan(bundle_plan) if isinstance(bundle_plan, Mapping) else None
        )
        if (
            bundle_manifest.get("null_execution_version") != NULL_EXECUTION_VERSION
            or stable_candidate != stable_bundle
            or not _approved_shared_direction_plan(
                bundle_plan,
                n_atoms=bundle_manifest.get("dictionary_cardinality"),
                d_model=bundle_manifest.get("dictionary_metric_dimension"),
                chunk_size=diagnostic.vocabulary_chunk_size,
            )
            or not _approved_shared_direction_plan(
                candidate,
                n_atoms=bundle_manifest.get("dictionary_cardinality"),
                d_model=bundle_manifest.get("dictionary_metric_dimension"),
                chunk_size=diagnostic.vocabulary_chunk_size,
            )
        ):
            raise KDiagnosticExperimentError(
                "v2 shared-direction memory plan is stale, mismatched, or rejected"
            )
        memory_plan = candidate
    measured = benchmark.get("artifact_usage")
    required_measures = {
        "bundle_bytes",
        "bundle_inodes",
        "max_logical_shard_bytes",
        "max_logical_shard_inodes",
    }
    if not isinstance(measured, Mapping) or not required_measures.issubset(measured):
        raise KDiagnosticExperimentError("microbenchmark lacks measured artifact usage")
    physical_count = int(bundles["physical_bundle_count"])
    logical_count = int(bundles["logical_replicate_count"])
    stage_gpu_hours = float(benchmark["wall_seconds"]) * physical_count / 3600.0
    stage_bytes = math.ceil(
        1.25
        * (
            physical_count * int(measured["bundle_bytes"])
            + logical_count * int(measured["max_logical_shard_bytes"])
            + int(rotation["matrix_bytes"])
        )
    )
    stage_inodes = math.ceil(
        1.25
        * (
            physical_count * int(measured["bundle_inodes"])
            + logical_count * int(measured["max_logical_shard_inodes"])
            + int(rotation["matrix_inodes"])
        )
    )
    stage_disk_gib = stage_bytes / 2**30
    runtime_files = (
        sorted((root / "runtimes").glob("*.json")) if (root / "runtimes").is_dir() else []
    )
    runtimes = [_read_json(path) for path in runtime_files]
    registered_bundle_ids = {str(entry["bundle_id"]) for entry in registered_bundles}
    for path, row in zip(runtime_files, runtimes, strict=True):
        runtime_hardware = None
        if is_v2:
            runtime_hardware = validate_k_diagnostic_execution_hardware(
                row.get("execution_hardware")
            )
        if (
            row.get("identity") != diagnostic.identity
            or row.get("stage") != diagnostic.stage
            or row.get("configuration_sha256") != configuration_sha256
            or str(row.get("bundle_id")) not in registered_bundle_ids
            or (is_v2 and row.get("null_execution_version") != NULL_EXECUTION_VERSION)
            or (
                is_v2
                and (
                    row.get("execution_backend") != runtime_hardware["execution_backend"]
                    or row.get("execution_hardware_sha256")
                    != execution_hardware_sha256(runtime_hardware)
                    or row.get("execution_hardware_sha256")
                    != benchmark.get("execution_hardware_sha256")
                )
            )
        ):
            raise KDiagnosticExperimentError(f"completed runtime identity mismatch: {path}")
    over_time = sorted(
        str(row.get("bundle_id"))
        for row in runtimes
        if row.get("within_hard_limit") is not True
        or float(row.get("wall_seconds", math.inf)) > diagnostic.max_seconds_per_bundle
    )
    other_budgets = [
        value
        for stage, value in REGISTERED_STAGE_RESOURCE_BUDGETS.items()
        if stage != diagnostic.stage
    ]
    total_gpu_hours = stage_gpu_hours + sum(value["gpu_hours"] for value in other_budgets)
    total_disk_gib = stage_disk_gib + sum(value["disk_gib"] for value in other_budgets)
    total_inodes = stage_inodes + sum(value["inodes"] for value in other_budgets)
    checks = {
        "stage_gpu_hours": stage_gpu_hours <= diagnostic.max_estimated_gpu_hours,
        "stage_disk_gib": stage_disk_gib <= diagnostic.max_estimated_disk_gib,
        "stage_inodes": stage_inodes <= diagnostic.max_estimated_inodes,
        "total_gpu_hours": total_gpu_hours <= diagnostic.max_total_estimated_gpu_hours,
        "total_disk_gib": total_disk_gib <= diagnostic.max_total_estimated_disk_gib,
        "total_inodes": total_inodes <= diagnostic.max_total_estimated_inodes,
        "completed_bundle_runtimes": not over_time,
    }
    if is_v2:
        assert memory_plan is not None
        memory_checks = memory_plan["checks"]
        checks["shared_direction_peak_bytes"] = bool(memory_checks["within_hard_peak_budget"])
        checks["shared_direction_available_ram_reserve"] = bool(
            memory_checks["preserves_available_ram_reserve"]
        )
    payload = {
        "schema_version": 1,
        "identity": diagnostic.identity,
        "stage": diagnostic.stage,
        "configuration_sha256": configuration_sha256,
        "benchmark_bundle_id": worst["bundle_id"],
        "benchmark_cost_units": int(worst["conservative_cost_units"]),
        "projection_formula": (
            "gpu_h=benchmark_wall_seconds*physical_bundles/3600; "
            "bytes/inodes=ceil(1.25*(physical*measured_bundle+logical*measured_max_shard+rotation_cache))"
        ),
        "stage_upper_bound": {
            "gpu_hours": stage_gpu_hours,
            "disk_gib": stage_disk_gib,
            "inodes": stage_inodes,
        },
        "full_experiment_upper_bound": {
            "gpu_hours": total_gpu_hours,
            "disk_gib": total_disk_gib,
            "inodes": total_inodes,
            "unmeasured_stages_reserved_at_registered_hard_budgets": True,
        },
        "hard_budgets": {
            "stage_gpu_hours": diagnostic.max_estimated_gpu_hours,
            "stage_disk_gib": diagnostic.max_estimated_disk_gib,
            "stage_inodes": diagnostic.max_estimated_inodes,
            "total_gpu_hours": diagnostic.max_total_estimated_gpu_hours,
            "total_disk_gib": diagnostic.max_total_estimated_disk_gib,
            "total_inodes": diagnostic.max_total_estimated_inodes,
            "max_seconds_per_bundle": diagnostic.max_seconds_per_bundle,
        },
        "completed_runtime_count": len(runtimes),
        "over_time_bundle_ids": over_time,
        "checks": checks,
        "approved": all(checks.values()),
    }
    if is_v2:
        payload["schema_version"] = 2
        payload["null_execution_version"] = NULL_EXECUTION_VERSION
        payload["shared_direction_memory_plan"] = dict(memory_plan or {})
        payload["execution_hardware"] = dict(execution_hardware or {})
        payload["execution_hardware_sha256"] = execution_hardware_sha256(execution_hardware)
        payload["execution_backend"] = execution_hardware["execution_backend"]
        payload["hard_budgets"]["shared_direction_peak_bytes"] = SHARED_DIRECTION_MAX_PEAK_BYTES
        payload["hard_budgets"]["shared_direction_reserved_available_bytes"] = (
            SHARED_DIRECTION_MIN_AVAILABLE_RESERVE_BYTES
        )
    atomic_write_json(root / "resource_preflight.json", payload)
    if not payload["approved"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise KDiagnosticExperimentError(f"resource preflight rejected: {failed}")
    return payload


def _normalize_rotation_metric_spaces(
    metric_spaces: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    unique: dict[tuple[str, str | None, int], dict[str, object]] = {}
    for raw in metric_spaces:
        metric = str(raw["metric"])
        transform_hash = raw.get("transform_sha256")
        transform_sha256 = None if transform_hash is None else str(transform_hash)
        dimension = int(raw["dimension"])
        key = (metric, transform_sha256, dimension)
        layers = sorted(int(value) for value in raw.get("layers", []))
        prior = unique.get(key)
        if prior is not None and prior.get("layers") != layers:
            raise KDiagnosticExperimentError("rotation metric space is duplicated inconsistently")
        unique[key] = {
            "metric": metric,
            "transform_sha256": transform_sha256,
            "dimension": dimension,
            "layers": layers,
        }
    if not unique:
        raise KDiagnosticExperimentError("rotation cache requires at least one metric space")
    return sorted(unique.values(), key=_canonical_json)


def rotation_metric_spaces_for_grid(
    config: ExperimentConfig,
    grid: Sequence[ScientificShard],
    *,
    run_root: str | Path = ".",
) -> list[dict[str, object]]:
    """Rebuild the binding spaces from current target and transform artifacts."""

    root = artifact_root(config, run_root=run_root)
    target_dimensions = {
        str(entry["target_id"]): int(entry["dimension"])
        for entry in load_target_index(config, run_root=run_root)
    }
    spaces: dict[tuple[str, str | None, int], dict[str, object]] = {}
    for bundle in build_physical_bundles(grid):
        if bundle.target_id not in target_dimensions:
            raise KDiagnosticExperimentError(
                f"rotation grid target is not in the current target index: {bundle.target_id}"
            )
        transform = load_metric_transform(
            artifact_directory=root, layer=bundle.layer, metric=bundle.metric
        )
        if transform is None:
            transform_sha256 = None
            dimension = target_dimensions[bundle.target_id]
        else:
            transform_sha256 = _array_hash(transform)
            dimension = int(transform.shape[0])
        key = (bundle.metric, transform_sha256, dimension)
        payload = spaces.setdefault(
            key,
            {
                "metric": bundle.metric,
                "transform_sha256": transform_sha256,
                "dimension": dimension,
                "layers": [],
            },
        )
        layers = payload["layers"]
        assert isinstance(layers, list)
        if bundle.layer not in layers:
            layers.append(bundle.layer)
    for payload in spaces.values():
        payload["layers"] = sorted(int(value) for value in payload["layers"])
    return _normalize_rotation_metric_spaces(list(spaces.values()))


def rotation_cache_build_preflight(
    config: ExperimentConfig,
    *,
    metric_spaces: Sequence[Mapping[str, object]],
    run_root: str | Path = ".",
    constructor: Any = None,
) -> dict[str, Any]:
    """Time one worst-dimension exact QR and gate the complete cache build."""

    diagnostic = config.k_diagnostic
    if diagnostic is None:
        raise KDiagnosticExperimentError("configuration has no k_diagnostic section")
    spaces = _normalize_rotation_metric_spaces(metric_spaces)
    binding_plan = [
        dict(space) | {"seed": int(seed)} for space in spaces for seed in diagnostic.rotation_seeds
    ]
    binding_plan_sha256 = _payload_hash(binding_plan)
    configuration_sha256 = _stage_manifest(config)["configuration_sha256"]
    stage = stage_root(config, run_root=run_root)
    path = stage / "rotation_cache_build_preflight.json"
    if path.is_file():
        observed = _read_json(path)
        if (
            observed.get("identity") != diagnostic.identity
            or observed.get("stage") != diagnostic.stage
            or observed.get("configuration_sha256") != configuration_sha256
            or observed.get("binding_plan_sha256") != binding_plan_sha256
        ):
            raise KDiagnosticExperimentError(
                "existing rotation cache-build preflight identity mismatch"
            )
        if observed.get("approved") is not True:
            raise KDiagnosticExperimentError("existing rotation cache-build preflight is rejected")
        return observed
    max_dimension = max(int(space["dimension"]) for space in spaces)
    sample_space = next(space for space in spaces if int(space["dimension"]) == max_dimension)
    sample_seed = int(diagnostic.rotation_seeds[0])
    cache_root = artifact_root(config, run_root=run_root) / "rotation_cache"
    matrix, manifest = load_cached_haar_rotation(
        cache_root,
        metric=str(sample_space["metric"]),
        transform_sha256=(
            None
            if sample_space["transform_sha256"] is None
            else str(sample_space["transform_sha256"])
        ),
        dimension=max_dimension,
        seed=sample_seed,
        create=True,
        constructor=constructor,
    )
    sample_wall = float(manifest.get("build_wall_seconds", 0.0))
    if not np.isfinite(sample_wall) or sample_wall <= 0.0:
        raise KDiagnosticExperimentError("Haar QR sample lacks positive measured build wall time")
    unique_matrix_keys = sorted(
        {
            (int(space["dimension"]), int(seed))
            for space in spaces
            for seed in diagnostic.rotation_seeds
        }
    )
    projected_wall = 1.25 * sum(
        sample_wall * (dimension / max_dimension) ** 3 for dimension, _seed in unique_matrix_keys
    )
    matrix_path = cache_root / "matrices" / str(manifest["cache_id"]) / "matrix.npy"
    sample_entry_usage = _tree_usage(matrix_path.parent)
    entry_overhead_bytes = max(0, int(sample_entry_usage["bytes"]) - int(matrix.nbytes))
    bytes_by_dimension = {
        str(dimension): int(8 * dimension * dimension + entry_overhead_bytes)
        for dimension in sorted({key[0] for key in unique_matrix_keys})
    }
    projected_bytes = math.ceil(
        1.25 * sum(bytes_by_dimension[str(dimension)] for dimension, _seed in unique_matrix_keys)
    )
    projected_inodes = math.ceil(1.25 * len(unique_matrix_keys) * int(sample_entry_usage["inodes"]))
    projected_gib = projected_bytes / 2**30
    effective_budgets = _effective_rotation_cache_budgets(diagnostic)
    projected_total_disk_gib = projected_gib + float(
        effective_budgets["other_stages_reserved_disk_gib"]
    )
    projected_total_inodes = projected_inodes + int(
        effective_budgets["other_stages_reserved_inodes"]
    )
    checks = {
        "projected_wall_within_24h": (
            projected_wall <= diagnostic.max_rotation_cache_build_seconds
        ),
        "projected_cache_within_budget": (
            projected_gib <= float(effective_budgets["effective_cache_gib_budget"])
        ),
        "projected_cache_inodes_within_budget": (
            projected_inodes <= int(effective_budgets["effective_cache_inode_budget"])
        ),
        "projected_full_disk_within_budget": (
            projected_total_disk_gib <= diagnostic.max_total_estimated_disk_gib
        ),
        "projected_full_inodes_within_budget": (
            projected_total_inodes <= diagnostic.max_total_estimated_inodes
        ),
    }
    payload = {
        "schema_version": 1,
        "identity": diagnostic.identity,
        "stage": diagnostic.stage,
        "configuration_sha256": configuration_sha256,
        "binding_plan_sha256": binding_plan_sha256,
        "binding_count": len(binding_plan),
        "unique_matrix_count": len(unique_matrix_keys),
        "matrix_identity": ["algorithm_version", "dimension", "seed"],
        "max_dimension": max_dimension,
        "bytes_per_entry_by_dimension": bytes_by_dimension,
        "inodes_per_entry": int(sample_entry_usage["inodes"]),
        "sample": {
            "dimension": max_dimension,
            "seed": sample_seed,
            "cache_id": manifest["cache_id"],
            "matrix_sha256": manifest["matrix_sha256"],
            "matrix_file_bytes": int(matrix_path.stat().st_size),
            "cache_entry_bytes": int(sample_entry_usage["bytes"]),
            "cache_entry_inodes": int(sample_entry_usage["inodes"]),
            "qr_wall_seconds": sample_wall,
        },
        "projection_formula": (
            "1.25*sum(sample_qr_wall*(dimension/max_dimension)^3) over unique "
            "(dimension,seed) matrices; bytes/inodes=ceil(1.25*sum of measured "
            "complete cache-entry payloads)"
        ),
        "projected_cache_build_wall_seconds": float(projected_wall),
        "projected_cache_bytes": int(projected_bytes),
        "projected_cache_gib": float(projected_gib),
        "projected_cache_inodes": int(projected_inodes),
        "projected_full_experiment_disk_gib_with_other_stages_reserved": float(
            projected_total_disk_gib
        ),
        "projected_full_experiment_inodes_with_other_stages_reserved": int(projected_total_inodes),
        "hard_budgets": {
            "seconds": diagnostic.max_rotation_cache_build_seconds,
            "configured_cache_gib": diagnostic.max_rotation_cache_gib,
            "stage_disk_gib": diagnostic.max_estimated_disk_gib,
            "total_disk_gib": diagnostic.max_total_estimated_disk_gib,
            "stage_inodes": diagnostic.max_estimated_inodes,
            "total_inodes": diagnostic.max_total_estimated_inodes,
            **effective_budgets,
        },
        "checks": checks,
        "approved": all(checks.values()),
    }
    _write_json_exact(path, payload)
    if payload["approved"] is not True:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise KDiagnosticExperimentError(f"rotation cache-build preflight rejected: {failed}")
    return payload


def prepare_rotation_cache(
    config: ExperimentConfig,
    *,
    metric_spaces: Sequence[Mapping[str, object]],
    run_root: str | Path = ".",
    require_build_preflight: bool = True,
) -> dict[str, Any]:
    """Precompute exact Haar matrices once per dimension/seed plus use bindings."""

    diagnostic = config.k_diagnostic
    if diagnostic is None:
        raise KDiagnosticExperimentError("configuration has no k_diagnostic section")
    root = artifact_root(config, run_root=run_root)
    cache_root = root / "rotation_cache"
    spaces = _normalize_rotation_metric_spaces(metric_spaces)
    binding_plan = [
        dict(space) | {"seed": int(seed)} for space in spaces for seed in diagnostic.rotation_seeds
    ]
    binding_plan_sha256 = _payload_hash(binding_plan)
    configuration_sha256 = _stage_manifest(config)["configuration_sha256"]
    preflight_path = stage_root(config, run_root=run_root) / "rotation_cache_build_preflight.json"
    if require_build_preflight:
        preflight = _read_json(preflight_path)
        if (
            preflight.get("approved") is not True
            or preflight.get("identity") != diagnostic.identity
            or preflight.get("stage") != diagnostic.stage
            or preflight.get("configuration_sha256") != configuration_sha256
            or preflight.get("binding_plan_sha256") != binding_plan_sha256
        ):
            raise KDiagnosticExperimentError(
                "rotation cache-build preflight is absent, stale, or rejected"
            )

    entries: list[dict[str, object]] = []
    matrices: dict[tuple[int, int], dict[str, object]] = {}
    for space in spaces:
        for seed in diagnostic.rotation_seeds:
            matrix, manifest = load_cached_haar_rotation(
                cache_root,
                metric=str(space["metric"]),
                transform_sha256=(
                    None if space["transform_sha256"] is None else str(space["transform_sha256"])
                ),
                dimension=int(space["dimension"]),
                seed=int(seed),
                create=True,
            )
            matrix_key = (int(space["dimension"]), int(seed))
            matrix_entry = {
                "dimension": matrix_key[0],
                "seed": matrix_key[1],
                "cache_id": str(manifest["cache_id"]),
                "matrix_sha256": str(manifest["matrix_sha256"]),
                "matrix_bytes": int(matrix.nbytes),
            }
            prior_matrix = matrices.get(matrix_key)
            if prior_matrix is not None and prior_matrix != matrix_entry:
                raise KDiagnosticExperimentError(
                    "shared Haar matrix changed across metric bindings"
                )
            matrices[matrix_key] = matrix_entry
            entries.append(dict(space) | matrix_entry)
    binding_identities = [
        _canonical_json(
            {
                key: entry[key]
                for key in (
                    "metric",
                    "transform_sha256",
                    "dimension",
                    "layers",
                    "seed",
                )
            }
        )
        for entry in entries
    ]
    if len(binding_identities) != len(set(binding_identities)):
        raise KDiagnosticExperimentError("rotation cache stage index contains duplicate bindings")
    payload = {
        "schema_version": 3,
        "complete": True,
        "identity": diagnostic.identity,
        "stage": diagnostic.stage,
        "configuration_sha256": configuration_sha256,
        "binding_plan_sha256": binding_plan_sha256,
        "algorithm": "exact Gaussian-QR Haar; cached, never constructed in bundle jobs",
        "metric_space_count": len(spaces),
        "rotation_seed_count": len(diagnostic.rotation_seeds),
        "binding_count": len(entries),
        "cache_entry_count": len(matrices),
        "unique_matrix_count": len(matrices),
        "matrix_bytes": sum(
            _tree_usage(cache_root / "matrices" / str(entry["cache_id"]))["bytes"]
            for entry in matrices.values()
        ),
        "matrix_inodes": sum(
            _tree_usage(cache_root / "matrices" / str(entry["cache_id"]))["inodes"]
            for entry in matrices.values()
        ),
        "build_preflight_sha256": (
            sha256_file(preflight_path) if require_build_preflight else None
        ),
        "entries": entries,
    }
    _write_json_exact(stage_root(config, run_root=run_root) / "rotation_cache_index.json", payload)
    return payload


def audit_rotation_cache_for_grid(
    config: ExperimentConfig,
    grid: Sequence[ScientificShard],
    *,
    run_root: str | Path = ".",
) -> dict[str, Any]:
    """Require the stage cache index to cover exactly every rotation shard space/seed."""

    diagnostic = config.k_diagnostic
    assert diagnostic is not None
    configuration_sha256 = _stage_manifest(config)["configuration_sha256"]
    current_spaces = rotation_metric_spaces_for_grid(config, grid, run_root=run_root)
    current_binding_plan = [
        dict(space) | {"seed": int(seed)}
        for space in current_spaces
        for seed in diagnostic.rotation_seeds
    ]
    binding_plan_sha256 = _payload_hash(current_binding_plan)
    index_path = stage_root(config, run_root=run_root) / "rotation_cache_index.json"
    index = _read_json(index_path)
    if (
        index.get("complete") is not True
        or index.get("identity") != diagnostic.identity
        or index.get("stage") != diagnostic.stage
        or index.get("configuration_sha256") != configuration_sha256
        or index.get("binding_plan_sha256") != binding_plan_sha256
    ):
        raise KDiagnosticExperimentError(f"rotation cache index is incomplete: {index_path}")
    entries = index.get("entries", [])
    if not isinstance(entries, list) or not entries:
        raise KDiagnosticExperimentError("rotation cache binding index is empty")
    binding_identities = {
        _canonical_json(
            {
                key: entry[key]
                for key in (
                    "metric",
                    "transform_sha256",
                    "dimension",
                    "layers",
                    "seed",
                )
            }
        )
        for entry in entries
    }
    if len(binding_identities) != len(entries):
        raise KDiagnosticExperimentError("rotation cache binding index has duplicates")
    required_entry_indices: set[int] = set()
    current_by_metric_layer = {
        (str(space["metric"]), int(layer)): space
        for space in current_spaces
        for layer in space["layers"]
    }
    for shard in grid:
        if shard.null_family != "orthogonal_rotation":
            continue
        current_space = current_by_metric_layer.get((shard.metric, shard.layer))
        if current_space is None:
            raise KDiagnosticExperimentError(
                f"current metric transform does not cover shard {shard.shard_id}"
            )
        matches = [
            (entry_index, entry)
            for entry_index, entry in enumerate(entries)
            if entry["metric"] == shard.metric
            and entry.get("transform_sha256") == current_space.get("transform_sha256")
            and int(entry["dimension"]) == int(current_space["dimension"])
            and int(entry["seed"]) == shard.null_seed
            and shard.layer in [int(value) for value in entry.get("layers", [])]
        ]
        if len(matches) != 1:
            raise KDiagnosticExperimentError(
                f"rotation cache does not uniquely cover shard {shard.shard_id}"
            )
        required_entry_indices.add(int(matches[0][0]))
    if required_entry_indices != set(range(len(entries))):
        raise KDiagnosticExperimentError(
            "rotation cache index does not exactly cover the stage rotation grid"
        )
    cache_root = artifact_root(config, run_root=run_root) / "rotation_cache"
    verified_cache_manifests: dict[str, Mapping[str, object]] = {}
    for entry in entries:
        cache_id = str(entry["cache_id"])
        manifest = verified_cache_manifests.get(cache_id)
        if manifest is None:
            _matrix, manifest = load_cached_haar_rotation(
                cache_root,
                metric=str(entry["metric"]),
                transform_sha256=(
                    None
                    if entry.get("transform_sha256") is None
                    else str(entry["transform_sha256"])
                ),
                dimension=int(entry["dimension"]),
                seed=int(entry["seed"]),
                create=False,
            )
            verified_cache_manifests[cache_id] = manifest
        if manifest["cache_id"] != cache_id or manifest["matrix_sha256"] != entry.get(
            "matrix_sha256"
        ):
            raise KDiagnosticExperimentError("rotation cache manifest ID/hash mismatch")
    if len(verified_cache_manifests) != int(index.get("unique_matrix_count", -1)):
        raise KDiagnosticExperimentError(
            "rotation cache unique matrix count does not match its bindings"
        )
    return index


def _validate_curve(errors: object, *, k_max: int, label: str) -> np.ndarray:
    values = np.asarray(errors, dtype=np.float64)
    if values.shape != (k_max + 1,) or not np.isfinite(values).all():
        raise KDiagnosticExperimentError(f"{label} errors must be finite [K_max+1]")
    if not np.isclose(values[0], 1.0, rtol=0.0, atol=1e-12):
        raise KDiagnosticExperimentError(f"{label} must retain K=0 with error 1")
    if np.any(np.diff(values) > 1e-10):
        raise KDiagnosticExperimentError(f"{label} error curve is not non-increasing")
    return values


def _coefficients_payload(result: Any) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for k, coefficients in enumerate(result.coefficients_per_k):
        values = np.asarray(coefficients, dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < -1e-12):
            raise KDiagnosticExperimentError("real pursuit coefficients must be non-negative")
        arrays[f"k_{k:03d}"] = np.maximum(values, 0.0)
    return arrays


def _fixed_budget_real_payload(
    *,
    result: Any,
    dictionary: Any,
    target: np.ndarray,
    raw_target: object | None,
    token_decoder: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    """Build one hash-pinned bundle artifact for all registered fixed budgets."""

    if token_decoder is None or not callable(token_decoder):
        raise KDiagnosticExperimentError(
            "bundle execution requires a tokenizer-backed selected-token decoder"
        )
    target_norm = float(np.linalg.norm(target))
    raw_vector = None if raw_target is None else np.asarray(raw_target, dtype=np.float64)
    raw_squared_norm = None if raw_vector is None else float(raw_vector @ raw_vector)
    budgets: dict[str, Any] = {}
    raw_reconstructions: dict[str, np.ndarray] | None = (
        {} if hasattr(dictionary, "raw_reconstruction") else None
    )
    if raw_reconstructions is not None and raw_vector is None:
        raise KDiagnosticExperimentError(
            "transformed fixed-budget telemetry requires the original raw target"
        )
    for requested in FIXED_BUDGETS:
        if requested > result.k_max:
            budgets[str(requested)] = {
                "requested_k": requested,
                "available": False,
                "reason": f"K_max={result.k_max}",
                "semantics": "fixed reconstruction budget, not estimated occupancy",
            }
            continue
        effective = min(requested, int(result.support.size))
        token_ids = np.asarray(result.support[:effective], dtype=np.int64)
        coefficients = np.asarray(result.coefficients_at(requested), dtype=np.float64)
        if coefficients.shape != (effective,):
            raise KDiagnosticExperimentError(
                f"fixed-budget coefficients do not align at K={requested}"
            )
        if effective:
            atoms = np.asarray(dictionary.materialize(token_ids), dtype=np.float64)
            reconstruction = coefficients @ atoms
            gram = atoms @ atoms.T
            gram_rank = int(np.linalg.matrix_rank(gram))
            gram_condition = float(np.linalg.cond(gram))
            decoded = token_decoder(token_ids.tolist())
            if isinstance(decoded, str):
                decoded_tokens = [decoded]
            else:
                decoded_tokens = [str(value) for value in decoded]
            if len(decoded_tokens) != effective:
                raise KDiagnosticExperimentError(
                    "selected-token decoder returned the wrong number of tokens"
                )
        else:
            reconstruction = np.zeros_like(target)
            gram_rank = 0
            gram_condition = math.nan
            decoded_tokens = []
        residual = target - reconstruction
        residual_norm = float(np.linalg.norm(residual))
        reconstruction_norm = float(np.linalg.norm(reconstruction))
        cosine = (
            None
            if reconstruction_norm == 0.0
            else float(target @ reconstruction / (target_norm * reconstruction_norm))
        )
        metric_error = float((residual_norm / target_norm) ** 2)
        entry: dict[str, Any] = {
            "requested_k": requested,
            "available": True,
            "effective_support_size": effective,
            "cosine_target_reconstruction": cosine,
            "residual_norm": residual_norm,
            "normalized_residual_norm": float(residual_norm / target_norm),
            "metric_space_error": metric_error,
            "metric_space_explained_fraction": float(1.0 - metric_error),
            "selected_token_ids": token_ids.tolist(),
            "decoded_tokens": decoded_tokens,
            "coefficients": coefficients.tolist(),
            "selected_atom_gram_rank": gram_rank,
            "selected_atom_gram_condition_number": (
                gram_condition if np.isfinite(gram_condition) else None
            ),
            "selected_atom_gram_condition_status": (
                "finite" if np.isfinite(gram_condition) else "singular_or_empty"
            ),
            "semantics": "fixed reconstruction budget, not estimated occupancy",
        }
        if raw_reconstructions is not None:
            assert raw_vector is not None and raw_squared_norm is not None
            raw_reconstruction = (
                dictionary.raw_reconstruction(token_ids, coefficients)
                if effective
                else np.zeros(int(dictionary.raw_d_model), dtype=np.float64)
            )
            raw_reconstructions[f"k_{requested:03d}"] = raw_reconstruction
            raw_error = float(np.sum((raw_vector - raw_reconstruction) ** 2) / raw_squared_norm)
            entry["raw_space_error"] = raw_error
            entry["raw_space_explained_fraction"] = float(1.0 - raw_error)
        budgets[str(requested)] = entry
    payload = {
        "schema_version": 1,
        "fixed_budgets": list(FIXED_BUDGETS),
        "metric_dimension": int(dictionary.d_model),
        "raw_dimension": (
            int(dictionary.raw_d_model)
            if hasattr(dictionary, "raw_d_model")
            else int(dictionary.d_model)
        ),
        "target_norm": target_norm,
        "token_decoding": "pinned tokenizer convert_ids_to_tokens",
        "budgets": budgets,
    }
    return payload, raw_reconstructions


def _null_curve_sha256(errors: object, gains: object) -> str:
    checked_errors = np.ascontiguousarray(np.asarray(errors, dtype=np.float64))
    checked_gains = np.ascontiguousarray(np.asarray(gains, dtype=np.float64))
    return _payload_hash(
        {
            "errors_raw_float64_sha256": hashlib.sha256(
                checked_errors.tobytes(order="C")
            ).hexdigest(),
            "errors_shape": list(checked_errors.shape),
            "gains_raw_float64_sha256": hashlib.sha256(
                checked_gains.tobytes(order="C")
            ).hexdigest(),
            "gains_shape": list(checked_gains.shape),
        }
    )


def _null_computation_id(payload: Mapping[str, object]) -> str:
    return _payload_hash({"null_execution_version": NULL_EXECUTION_VERSION, **dict(payload)})[:24]


def _precomputed_null(
    *,
    result: Any,
    metadata: Mapping[str, object],
    shared_null_computation_id: str,
    null_batch_computation_id: str,
) -> PrecomputedNull:
    errors = np.asarray(result.errors, dtype=np.float64)
    gains = np.asarray(result.gains, dtype=np.float64)
    return PrecomputedNull(
        result=result,
        metadata=dict(metadata),
        shared_null_computation_id=shared_null_computation_id,
        source_curve_sha256=_null_curve_sha256(errors, gains),
        null_batch_computation_id=null_batch_computation_id,
    )


def _existing_v2_shard_complete(
    *,
    directory: Path,
    shard: ScientificShard,
    diagnostic: Any,
    configuration_sha256: str,
    physical_bundle_id: str,
) -> bool:
    if not directory.exists():
        return False
    missing = [name for name in REQUIRED_SHARD_FILES if not (directory / name).is_file()]
    if missing:
        raise KDiagnosticExperimentError(
            f"partial shard exists and is not reusable: {directory}; missing {missing}"
        )
    manifest = _read_json(directory / "manifest.json")
    expected = {
        "experiment_identity": diagnostic.identity,
        "stage": diagnostic.stage,
        "scientific_identity": shard.identity,
        "scientific_identity_sha256": _payload_hash(shard.identity),
        "configuration_sha256": configuration_sha256,
        "physical_bundle_id": physical_bundle_id,
        "null_execution_version": NULL_EXECUTION_VERSION,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise KDiagnosticExperimentError(
                f"existing shard has incompatible {field}: {directory}"
            )
    errors = np.load(directory / "null_errors.npy", allow_pickle=False)
    gains = np.load(directory / "null_gains.npy", allow_pickle=False)
    if errors.shape != (1, int(diagnostic.k_max) + 1) or gains.shape != (1, int(diagnostic.k_max)):
        raise KDiagnosticExperimentError(
            f"existing shard has invalid null curve shapes: {directory}"
        )
    observed_curve_hash = _null_curve_sha256(errors[0], gains[0])
    metadata = _read_json(directory / "null_metadata.json")
    for source in (manifest, metadata):
        if (
            source.get("null_execution_version") != NULL_EXECUTION_VERSION
            or source.get("source_curve_sha256") != observed_curve_hash
            or not isinstance(source.get("shared_null_computation_id"), str)
            or not isinstance(source.get("null_batch_computation_id"), str)
        ):
            raise KDiagnosticExperimentError(
                f"existing shard shared-null hash/version mismatch: {directory}"
            )
    if (
        manifest["shared_null_computation_id"] != metadata["shared_null_computation_id"]
        or manifest["null_batch_computation_id"] != metadata["null_batch_computation_id"]
    ):
        raise KDiagnosticExperimentError(
            f"existing shard shared-null identity mismatch: {directory}"
        )
    return True


def _run_scientific_shard_v2(
    *,
    config: ExperimentConfig,
    shard: ScientificShard,
    dictionary: Any,
    target: object,
    precomputed_null: PrecomputedNull,
    precomputed_real_errors: object,
    real_result_metadata: Mapping[str, object],
    run_root: str | Path = ".",
    null_target: object | None = None,
    raw_target: object | None = None,
    precomputed_norms: object | None = None,
    precomputed_raw_reconstructions: Mapping[str, np.ndarray] | None = None,
    physical_bundle_id: str | None = None,
) -> dict[str, Any]:
    """Persist one logical shard from an audited bundle-level null computation."""

    diagnostic = config.k_diagnostic
    if diagnostic is None:
        raise KDiagnosticExperimentError("configuration has no k_diagnostic section")
    if physical_bundle_id is None:
        raise KDiagnosticExperimentError(
            "scientific shards must be executed inside a physical bundle"
        )
    vector = np.asarray(target, dtype=np.float64)
    if vector.shape != (int(dictionary.d_model),) or not np.isfinite(vector).all():
        raise KDiagnosticExperimentError("target does not match dictionary metric dimension")
    raw_vector: np.ndarray | None = None
    if hasattr(dictionary, "raw_reconstruction"):
        if raw_target is None:
            raise KDiagnosticExperimentError("transformed pursuit requires the original raw target")
        raw_vector = np.asarray(raw_target, dtype=np.float64)
        if raw_vector.shape != (int(dictionary.raw_d_model),) or not np.isfinite(raw_vector).all():
            raise KDiagnosticExperimentError("raw target width/values do not match transform")
        if float(raw_vector @ raw_vector) == 0.0:
            raise KDiagnosticExperimentError("raw target must be non-zero")
    permuted: np.ndarray | None = None
    if shard.null_family == "label_permutation_probe":
        if null_target is None:
            raise KDiagnosticExperimentError("label-permutation shard requires null_target")
        permuted = np.asarray(null_target, dtype=np.float64)
        if permuted.shape != vector.shape or not np.isfinite(permuted).all():
            raise KDiagnosticExperimentError("permutation target has wrong shape or values")
    norms = np.asarray(
        dictionary.atom_norms() if precomputed_norms is None else precomputed_norms,
        dtype=np.float64,
    )
    if norms.shape != (int(dictionary.n_atoms),) or not np.isfinite(norms).all():
        raise KDiagnosticExperimentError("dictionary atom norms are invalid")

    real_errors = _validate_curve(precomputed_real_errors, k_max=diagnostic.k_max, label="real")
    null_errors = _validate_curve(
        precomputed_null.result.errors,
        k_max=diagnostic.k_max,
        label=shard.null_family,
    )
    null_gains = np.asarray(precomputed_null.result.gains, dtype=np.float64)
    if null_gains.shape != (diagnostic.k_max,) or not np.allclose(
        null_gains,
        null_errors[:-1] - null_errors[1:],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise KDiagnosticExperimentError("precomputed null gains do not match errors")
    observed_curve_hash = _null_curve_sha256(null_errors, null_gains)
    if observed_curve_hash != precomputed_null.source_curve_sha256:
        raise KDiagnosticExperimentError("precomputed shared null curve hash mismatch")

    output = stage_root(config, run_root=run_root) / "shards" / shard.shard_id
    manifest = {
        "schema_version": 2,
        "experiment_identity": diagnostic.identity,
        "stage": diagnostic.stage,
        "scientific_identity": shard.identity,
        "scientific_identity_sha256": _payload_hash(shard.identity),
        "solver_method": diagnostic.solver_method,
        "selection_mode": diagnostic.selection_mode,
        "k_max": diagnostic.k_max,
        "configuration_sha256": _stage_manifest(config)["configuration_sha256"],
        "metric_target_sha256": _array_hash(vector),
        "raw_target_sha256": _array_hash(raw_vector) if raw_vector is not None else None,
        "permutation_target_sha256": (_array_hash(permuted) if permuted is not None else None),
        "transform_sha256": getattr(dictionary, "transform_sha256", None),
        "dictionary_cardinality": int(dictionary.n_atoms),
        "dictionary_metric_dimension": int(dictionary.d_model),
        "dictionary_raw_dimension": (
            int(dictionary.raw_d_model)
            if hasattr(dictionary, "raw_d_model")
            else int(dictionary.d_model)
        ),
        "dictionary_atom_norms_sha256": _array_hash(norms),
        "physical_bundle_id": physical_bundle_id,
        "null_execution_version": NULL_EXECUTION_VERSION,
        "shared_null_computation_id": precomputed_null.shared_null_computation_id,
        "source_curve_sha256": observed_curve_hash,
        "null_batch_computation_id": precomputed_null.null_batch_computation_id,
    }
    if output.exists():
        if _canonical_json(_read_json(output / "manifest.json")) != _canonical_json(manifest):
            raise KDiagnosticExperimentError(f"existing shard does not match identity: {output}")
        if all((output / name).is_file() for name in REQUIRED_SHARD_FILES):
            saved_errors = np.load(output / "null_errors.npy", allow_pickle=False)
            saved_gains = np.load(output / "null_gains.npy", allow_pickle=False)
            if (
                saved_errors.shape != (1, diagnostic.k_max + 1)
                or saved_gains.shape != (1, diagnostic.k_max)
                or _null_curve_sha256(saved_errors[0], saved_gains[0]) != observed_curve_hash
            ):
                raise KDiagnosticExperimentError(
                    f"existing shard shared-null payload hash mismatch: {output}"
                )
            return _read_json(output / "summary.json")
        raise KDiagnosticExperimentError(f"partial shard exists and is not reusable: {output}")

    fixed = {
        str(k): {
            "real_explained_fraction": float(1.0 - real_errors[k]),
            "null_explained_fraction": float(1.0 - null_errors[k]),
            "semantics": "fixed reconstruction budget, not estimated occupancy",
        }
        for k in diagnostic.report_grid
    }
    summary = {
        "schema_version": 2,
        **shard.identity,
        "shard_id": shard.shard_id,
        "real_target_norm": float(real_result_metadata["target_norm"]),
        "real_support_size": int(real_result_metadata["support_size"]),
        "real_stopped_early_at": real_result_metadata["stopped_early_at"],
        "real_final_explained_fraction": float(1.0 - real_errors[-1]),
        "null_final_explained_fraction": float(1.0 - null_errors[-1]),
        "fixed_budget": fixed,
    }
    bundle_directory = stage_root(config, run_root=run_root) / "bundles" / physical_bundle_id
    real_files = list(REQUIRED_BUNDLE_REAL_FILES)
    if hasattr(dictionary, "raw_reconstruction"):
        real_files.append("raw_reconstructions_real.npz")
    real_reference = {
        "schema_version": 1,
        "physical_bundle_id": physical_bundle_id,
        "files": {
            name: sha256_file(bundle_directory / name)
            for name in real_files
            if (bundle_directory / name).is_file()
        },
    }
    if set(real_reference["files"]) != set(real_files):
        raise KDiagnosticExperimentError(f"bundle real payload is incomplete: {bundle_directory}")
    if hasattr(dictionary, "raw_reconstruction"):
        assert raw_vector is not None
        if precomputed_raw_reconstructions is None:
            raise KDiagnosticExperimentError(
                "transformed bundle did not supply its shared raw reconstructions"
            )
        squared_raw_norm = float(raw_vector @ raw_vector)
        for k in diagnostic.report_grid:
            reconstruction = np.asarray(
                precomputed_raw_reconstructions[f"k_{k:03d}"], dtype=np.float64
            )
            raw_error = float(np.sum((raw_vector - reconstruction) ** 2) / squared_raw_norm)
            summary["fixed_budget"][str(k)]["raw_space_error"] = raw_error
            summary["fixed_budget"][str(k)]["raw_space_explained_fraction"] = 1.0 - raw_error

    null_metadata = {
        **dict(precomputed_null.metadata),
        "null_family": shard.null_family,
        "null_seed": shard.null_seed,
        "dictionary_fraction": shard.dictionary_fraction,
        "metric_dimension": int(dictionary.d_model),
        "real_dictionary_cardinality": int(dictionary.n_atoms),
        "null_execution_version": NULL_EXECUTION_VERSION,
        "shared_null_computation_id": precomputed_null.shared_null_computation_id,
        "source_curve_sha256": observed_curve_hash,
        "null_batch_computation_id": precomputed_null.null_batch_computation_id,
    }
    staging_root = stage_root(config, run_root=run_root) / "shard_staging" / physical_bundle_id
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(tempfile.mkdtemp(prefix=f".{shard.shard_id}.", dir=staging_root))
    try:
        _atomic_save_npy(staging_directory / "null_errors.npy", null_errors[None, :])
        _atomic_save_npy(staging_directory / "null_gains.npy", null_gains[None, :])
        atomic_write_json(staging_directory / "real_reference.json", real_reference)
        atomic_write_json(staging_directory / "null_metadata.json", null_metadata)
        atomic_write_json(staging_directory / "summary.json", summary)
        atomic_write_json(staging_directory / "manifest.json", manifest)
        if not all((staging_directory / name).is_file() for name in REQUIRED_SHARD_FILES):
            raise KDiagnosticExperimentError("v2 shard staging directory is incomplete")
        staged_errors = np.load(staging_directory / "null_errors.npy", allow_pickle=False)
        staged_gains = np.load(staging_directory / "null_gains.npy", allow_pickle=False)
        if (
            _canonical_json(_read_json(staging_directory / "manifest.json"))
            != _canonical_json(manifest)
            or _null_curve_sha256(staged_errors[0], staged_gains[0]) != observed_curve_hash
        ):
            raise KDiagnosticExperimentError("v2 shard staging verification failed")
        descriptor = os.open(staging_directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_directory, output)
        descriptor = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory)
    return summary


def _bundle_real_state(
    *,
    output: Path,
    bundle_manifest: Mapping[str, object],
    diagnostic: Any,
    transformed: bool,
) -> tuple[np.ndarray, dict[str, object], dict[str, np.ndarray] | None]:
    try:
        if _canonical_json(_read_json(output / "manifest.json")) != _canonical_json(
            bundle_manifest
        ):
            raise KDiagnosticExperimentError("bundle manifest/version mismatch")
        marker_path = output / "real_complete.json"
        marker = _read_json(marker_path)
        real_names = list(REQUIRED_BUNDLE_REAL_FILES)
        if transformed:
            real_names.append("raw_reconstructions_real.npz")
        expected_hashes = {name: sha256_file(output / name) for name in real_names}
        if (
            marker.get("complete") is not True
            or marker.get("null_execution_version") != NULL_EXECUTION_VERSION
            or marker.get("real_files") != expected_hashes
        ):
            raise KDiagnosticExperimentError("bundle real payload hash mismatch")
        errors = _validate_curve(
            np.load(output / "errors_real.npy", allow_pickle=False),
            k_max=int(diagnostic.k_max),
            label="resumed real",
        )
        metadata = marker.get("real_result")
        if not isinstance(metadata, Mapping):
            raise KDiagnosticExperimentError("bundle real result metadata is missing")
        raw_reconstructions = None
        if transformed:
            with np.load(output / "raw_reconstructions_real.npz", allow_pickle=False) as archive:
                raw_reconstructions = {
                    name: np.asarray(archive[name], dtype=np.float64) for name in archive.files
                }
        return errors, dict(metadata), raw_reconstructions
    except KDiagnosticExperimentError as error:
        raise KDiagnosticExperimentError(
            f"partial bundle cache is not reusable: {output}: {error}"
        ) from error
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise KDiagnosticExperimentError(
            f"partial bundle cache is not reusable: {output}"
        ) from error


def _run_scientific_bundle_v2(
    *,
    config: ExperimentConfig,
    bundle: PhysicalBundle,
    shards: Sequence[ScientificShard],
    dictionary: Any,
    target: object,
    run_root: str | Path = ".",
    permutation_targets: Mapping[str, object] | None = None,
    raw_target: object | None = None,
    token_decoder: Any = None,
) -> dict[str, dict[str, Any]]:
    """Run one v2 bundle with exact IID reuse and same-dictionary target batches."""

    diagnostic = config.k_diagnostic
    if diagnostic is None:
        raise KDiagnosticExperimentError("configuration has no k_diagnostic section")
    ordered = sorted(shards, key=lambda shard: _canonical_json(shard.identity))
    if tuple(sorted(shard.shard_id for shard in ordered)) != bundle.shard_ids:
        raise KDiagnosticExperimentError("bundle logical shard membership mismatch")
    for shard in ordered:
        expected = {
            "target_id": shard.target_id,
            "layer": shard.layer,
            "target_family": shard.target_family,
            "target_subtype": shard.target_subtype,
            "metric": shard.metric,
        }
        if expected != bundle.scientific_identity:
            raise KDiagnosticExperimentError("bundle contains a foreign scientific shard")
    vector = np.asarray(target, dtype=np.float64)
    if vector.shape != (int(dictionary.d_model),) or not np.isfinite(vector).all():
        raise KDiagnosticExperimentError("bundle target does not match dictionary")
    norms = np.asarray(dictionary.atom_norms(), dtype=np.float64)
    if norms.shape != (int(dictionary.n_atoms),) or not np.isfinite(norms).all():
        raise KDiagnosticExperimentError("bundle dictionary atom norms are invalid")

    iid_shards = [
        shard
        for shard in ordered
        if shard.null_family in {"iid_full_cardinality", "iid_cardinality_sweep"}
    ]
    direction_plan = (
        shared_direction_memory_plan(
            n_atoms=int(dictionary.n_atoms),
            d_model=int(dictionary.d_model),
            chunk_size=int(diagnostic.vocabulary_chunk_size),
        )
        if iid_shards
        else None
    )
    if direction_plan is not None and direction_plan["approved"] is not True:
        failed = sorted(name for name, passed in direction_plan["checks"].items() if not passed)
        raise KDiagnosticExperimentError(f"shared-direction RAM preflight rejected: {failed}")
    configuration_sha256 = _stage_manifest(config)["configuration_sha256"]
    stage = stage_root(config, run_root=run_root)
    output = stage / "bundles" / bundle.bundle_id
    sealed_direction_plan = direction_plan
    if output.is_dir() and (output / "manifest.json").is_file():
        existing_manifest = _read_json(output / "manifest.json")
        existing_plan = existing_manifest.get("shared_direction_memory_plan")
        if direction_plan is None:
            if existing_plan is not None:
                raise KDiagnosticExperimentError(f"bundle shared-direction plan mismatch: {output}")
        elif (
            existing_manifest.get("null_execution_version") != NULL_EXECUTION_VERSION
            or not isinstance(existing_plan, Mapping)
            or _stable_shared_direction_plan(existing_plan)
            != _stable_shared_direction_plan(direction_plan)
            or not _approved_shared_direction_plan(
                existing_plan,
                n_atoms=int(dictionary.n_atoms),
                d_model=int(dictionary.d_model),
                chunk_size=int(diagnostic.vocabulary_chunk_size),
            )
        ):
            raise KDiagnosticExperimentError(
                f"bundle shared-direction plan/version mismatch: {output}"
            )
        else:
            sealed_direction_plan = dict(existing_plan)
    bundle_manifest = {
        "schema_version": 3,
        "experiment_identity": diagnostic.identity,
        "stage": diagnostic.stage,
        "physical_identity": bundle.scientific_identity,
        "physical_bundle_id": bundle.bundle_id,
        "logical_shard_ids": list(bundle.shard_ids),
        "logical_shard_count": len(bundle.shard_ids),
        "configuration_sha256": configuration_sha256,
        "metric_target_sha256": _array_hash(vector),
        "raw_target_sha256": _array_hash(raw_target) if raw_target is not None else None,
        "transform_sha256": getattr(dictionary, "transform_sha256", None),
        "dictionary_cardinality": int(dictionary.n_atoms),
        "dictionary_metric_dimension": int(dictionary.d_model),
        "dictionary_raw_dimension": (
            int(dictionary.raw_d_model)
            if hasattr(dictionary, "raw_d_model")
            else int(dictionary.d_model)
        ),
        "dictionary_atom_norms_sha256": _array_hash(norms),
        "real_solve_count": 1,
        "atom_norm_pass_count": 1,
        "null_execution_version": NULL_EXECUTION_VERSION,
        "shared_direction_memory_plan": sealed_direction_plan,
    }
    lock_root = stage / "bundle_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock = lock_root / f"{bundle.bundle_id}.lock"
    # Reject old scalar shards before spending a real solve. Complete v2 shards
    # are reusable; partial or version-mismatched shards are fail-closed.
    for shard in ordered:
        shard_directory = stage / "shards" / shard.shard_id
        if shard_directory.exists():
            _existing_v2_shard_complete(
                directory=shard_directory,
                shard=shard,
                diagnostic=diagnostic,
                configuration_sha256=configuration_sha256,
                physical_bundle_id=bundle.bundle_id,
            )

    def completed_result() -> dict[str, dict[str, Any]]:
        if _canonical_json(_read_json(output / "manifest.json")) != _canonical_json(
            bundle_manifest
        ):
            raise KDiagnosticExperimentError(f"bundle cache identity mismatch: {output}")
        complete = _read_json(output / "complete.json")
        if (
            complete.get("complete") is not True
            or complete.get("logical_shard_ids") != list(bundle.shard_ids)
            or complete.get("null_execution_version") != NULL_EXECUTION_VERSION
            or complete.get("real_complete_sha256") != sha256_file(output / "real_complete.json")
        ):
            raise KDiagnosticExperimentError(f"bundle cache is partial: {output}")
        summaries: dict[str, dict[str, Any]] = {}
        verified_real_files: dict[Path, str] = {}
        for shard in ordered:
            shard_directory = stage / "shards" / shard.shard_id
            summary, _real, _null = _load_and_audit_shard(
                shard_directory,
                shard,
                diagnostic=diagnostic,
                configuration_sha256=configuration_sha256,
                verified_real_files=verified_real_files,
                verified_rotation_manifests=None,
            )
            summaries[shard.shard_id] = summary
        return summaries

    if output.exists() and (output / "complete.json").is_file():
        return completed_result()
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise KDiagnosticExperimentError(
            f"bundle {bundle.bundle_id} is already locked by another worker"
        ) from error
    os.close(descriptor)
    try:
        transformed = hasattr(dictionary, "raw_reconstruction")
        if output.exists():
            real_errors, real_result_metadata, raw_reconstructions = _bundle_real_state(
                output=output,
                bundle_manifest=bundle_manifest,
                diagnostic=diagnostic,
                transformed=transformed,
            )
        else:
            output.mkdir(parents=True, exist_ok=False)
            atomic_write_json(output / "manifest.json", bundle_manifest)
            real = streaming_nonnegative_pursuit(
                dictionary,
                vector[None, :],
                k_max=diagnostic.k_max,
                selection_modes=diagnostic.selection_mode,
                solver_method=diagnostic.solver_method,
            )[0]
            real_errors = _validate_curve(real.errors, k_max=diagnostic.k_max, label="real")
            _atomic_save_npy(output / "errors_real.npy", real_errors)
            _atomic_save_npy(output / "gains_real.npy", real.gains)
            _atomic_save_npy(output / "support_real.npy", real.support)
            _atomic_save_npz(output / "coefficients_real.npz", _coefficients_payload(real))
            fixed_budget_payload, raw_reconstructions = _fixed_budget_real_payload(
                result=real,
                dictionary=dictionary,
                target=vector,
                raw_target=raw_target,
                token_decoder=token_decoder,
            )
            atomic_write_json(output / "fixed_budget_real.json", fixed_budget_payload)
            if raw_reconstructions is not None:
                _atomic_save_npz(output / "raw_reconstructions_real.npz", raw_reconstructions)
            real_result_metadata = {
                "target_norm": float(real.target_norm),
                "support_size": int(real.support.size),
                "stopped_early_at": real.stopped_early_at,
            }
            real_names = list(REQUIRED_BUNDLE_REAL_FILES)
            if raw_reconstructions is not None:
                real_names.append("raw_reconstructions_real.npz")
            atomic_write_json(
                output / "real_complete.json",
                {
                    "schema_version": 1,
                    "complete": True,
                    "null_execution_version": NULL_EXECUTION_VERSION,
                    "real_solve_count": 1,
                    "real_result": real_result_metadata,
                    "real_files": {name: sha256_file(output / name) for name in real_names},
                },
            )

        pending = [
            shard
            for shard in ordered
            if not _existing_v2_shard_complete(
                directory=stage / "shards" / shard.shard_id,
                shard=shard,
                diagnostic=diagnostic,
                configuration_sha256=configuration_sha256,
                physical_bundle_id=bundle.bundle_id,
            )
        ]
        completed_ids = {shard.shard_id for shard in ordered} - {
            shard.shard_id for shard in pending
        }

        def persist(
            shard: ScientificShard,
            value: PrecomputedNull,
            null_target: object | None = None,
        ) -> None:
            _run_scientific_shard_v2(
                config=config,
                shard=shard,
                dictionary=dictionary,
                target=vector,
                precomputed_null=value,
                precomputed_real_errors=real_errors,
                real_result_metadata=real_result_metadata,
                run_root=run_root,
                null_target=null_target,
                raw_target=raw_target,
                precomputed_norms=norms,
                precomputed_raw_reconstructions=raw_reconstructions,
                physical_bundle_id=bundle.bundle_id,
            )
            completed_ids.add(shard.shard_id)

        pending_iid = [
            shard
            for shard in pending
            if shard.null_family in {"iid_full_cardinality", "iid_cardinality_sweep"}
        ]
        all_iid_by_seed: dict[int, list[ScientificShard]] = defaultdict(list)
        for shard in iid_shards:
            all_iid_by_seed[shard.null_seed].append(shard)
        pending_iid_by_seed: dict[int, list[ScientificShard]] = defaultdict(list)
        for shard in pending_iid:
            pending_iid_by_seed[shard.null_seed].append(shard)
        for seed, seed_pending in sorted(pending_iid_by_seed.items()):
            all_seed_shards = all_iid_by_seed[seed]
            batch_id = _null_computation_id(
                {
                    "physical_bundle_id": bundle.bundle_id,
                    "kind": "iid_shared_direction_seed_batch",
                    "seed": seed,
                    "logical_shard_ids": sorted(shard.shard_id for shard in all_seed_shards),
                }
            )
            with SharedDirectionProvider(
                n_atoms=int(dictionary.n_atoms),
                d_model=int(dictionary.d_model),
                seed=seed,
                chunk_size=diagnostic.vocabulary_chunk_size,
            ) as provider:
                by_fraction: dict[float, list[ScientificShard]] = defaultdict(list)
                for shard in seed_pending:
                    by_fraction[shard.dictionary_fraction].append(shard)
                for fraction, fraction_shards in sorted(by_fraction.items()):
                    selected_norms, subset_metadata = matched_norm_subset(
                        norms, fraction=fraction, seed=seed
                    )
                    random_dictionary = SharedDirectionRandomDictionary(provider, selected_norms)
                    result = streaming_nonnegative_pursuit(
                        random_dictionary,
                        vector[None, :],
                        k_max=diagnostic.k_max,
                        selection_modes=diagnostic.selection_mode,
                        solver_method=diagnostic.solver_method,
                    )[0]
                    computation_id = _null_computation_id(
                        {
                            "physical_bundle_id": bundle.bundle_id,
                            "kind": "iid_matched_norm_curve",
                            "seed": seed,
                            "fraction": fraction,
                            "norm_subset_sha256": subset_metadata["norm_subset_sha256"],
                            "unit_directions_sha256": provider.raw_float64_sha256,
                        }
                    )
                    value = _precomputed_null(
                        result=result,
                        metadata={
                            **subset_metadata,
                            "search_semantics": ("full best-atom scan at registered cardinality"),
                            "shared_direction_provider_version": (
                                provider.memory_plan["provider_version"]
                            ),
                            "shared_direction_storage": (provider.memory_plan["storage"]),
                            "shared_direction_peak_bytes": provider.memory_plan["peak_bytes"],
                            "shared_direction_generated_chunk_count": (provider.generation_count),
                            "shared_direction_all_chunks_generated_once": all(
                                count == 1 for count in provider.generation_counts.values()
                            ),
                            "unit_directions_raw_float64_sha256": (provider.raw_float64_sha256),
                        },
                        shared_null_computation_id=computation_id,
                        null_batch_computation_id=batch_id,
                    )
                    for shard in fraction_shards:
                        persist(shard, value)

        prefix_shards = [shard for shard in pending if shard.null_family == "random_prefix_same_k"]
        for shard in prefix_shards:
            atoms, metadata = random_prefix_atoms(
                norms,
                k_max=diagnostic.k_max,
                seed=shard.null_seed,
                d_model=int(dictionary.d_model),
            )
            result = random_prefix_pursuit(atoms, vector)
            computation_id = _null_computation_id(
                {
                    "physical_bundle_id": bundle.bundle_id,
                    "kind": "random_prefix",
                    "seed": shard.null_seed,
                }
            )
            persist(
                shard,
                _precomputed_null(
                    result=result,
                    metadata={
                        **metadata,
                        "search_semantics": "fixed nested prefix; no vocabulary search",
                    },
                    shared_null_computation_id=computation_id,
                    null_batch_computation_id=computation_id,
                ),
            )

        rotation_shards = [shard for shard in pending if shard.null_family == "orthogonal_rotation"]
        if rotation_shards:
            all_rotation_ids = sorted(
                shard.shard_id for shard in ordered if shard.null_family == "orthogonal_rotation"
            )
            batch_id = _null_computation_id(
                {
                    "physical_bundle_id": bundle.bundle_id,
                    "kind": "orthogonal_rotation_target_batch",
                    "logical_shard_ids": all_rotation_ids,
                }
            )
            rotated_targets = []
            rotation_metadata = []
            for shard in rotation_shards:
                rotation, rotation_manifest = load_cached_haar_rotation(
                    artifact_root(config, run_root=run_root) / "rotation_cache",
                    metric=shard.metric,
                    transform_sha256=getattr(dictionary, "transform_sha256", None),
                    dimension=int(dictionary.d_model),
                    seed=shard.null_seed,
                    create=False,
                )
                rotated, preservation_error = inverse_rotate_target(vector, rotation)
                rotated_targets.append(rotated)
                rotation_metadata.append(
                    {
                        **rotation_manifest["numerical_validation"],
                        "rotation_cache_id": rotation_manifest["cache_id"],
                        "rotation_matrix_sha256": rotation_manifest["matrix_sha256"],
                        "target_relative_norm_error": preservation_error,
                        "rotation_space": shard.metric,
                    }
                )
            results = streaming_nonnegative_pursuit(
                dictionary,
                np.stack(rotated_targets),
                k_max=diagnostic.k_max,
                selection_modes=diagnostic.selection_mode,
                solver_method=diagnostic.solver_method,
            )
            for shard, rotated, metadata, result in zip(
                rotation_shards,
                rotated_targets,
                rotation_metadata,
                results,
                strict=True,
            ):
                computation_id = _null_computation_id(
                    {
                        "null_batch_computation_id": batch_id,
                        "seed": shard.null_seed,
                        "inverse_rotated_target_sha256": _array_hash(rotated),
                    }
                )
                persist(
                    shard,
                    _precomputed_null(
                        result=result,
                        metadata=metadata,
                        shared_null_computation_id=computation_id,
                        null_batch_computation_id=batch_id,
                    ),
                )

        permutation_shards = [
            shard for shard in pending if shard.null_family == "label_permutation_probe"
        ]
        if permutation_shards:
            if permutation_targets is None:
                raise KDiagnosticExperimentError("bundle lacks registered permutation targets")
            all_permutation_ids = sorted(
                shard.shard_id
                for shard in ordered
                if shard.null_family == "label_permutation_probe"
            )
            batch_id = _null_computation_id(
                {
                    "physical_bundle_id": bundle.bundle_id,
                    "kind": "label_permutation_target_batch",
                    "logical_shard_ids": all_permutation_ids,
                }
            )
            permutation_vectors = []
            permutation_metadata = []
            for shard in permutation_shards:
                if shard.shard_id not in permutation_targets:
                    raise KDiagnosticExperimentError(
                        f"bundle lacks permutation target for {shard.shard_id}"
                    )
                value = permutation_targets[shard.shard_id]
                if isinstance(value, tuple) and len(value) == 2:
                    null_target, metadata = value
                else:
                    null_target, metadata = value, {}
                permuted = np.asarray(null_target, dtype=np.float64)
                if permuted.shape != vector.shape or not np.isfinite(permuted).all():
                    raise KDiagnosticExperimentError("permutation target has wrong shape or values")
                permutation_vectors.append(permuted)
                permutation_metadata.append(dict(metadata))
            results = streaming_nonnegative_pursuit(
                dictionary,
                np.stack(permutation_vectors),
                k_max=diagnostic.k_max,
                selection_modes=diagnostic.selection_mode,
                solver_method=diagnostic.solver_method,
            )
            for shard, permuted, metadata, result in zip(
                permutation_shards,
                permutation_vectors,
                permutation_metadata,
                results,
                strict=True,
            ):
                computation_id = _null_computation_id(
                    {
                        "null_batch_computation_id": batch_id,
                        "seed": shard.null_seed,
                        "permuted_target_sha256": _array_hash(permuted),
                    }
                )
                value = _precomputed_null(
                    result=result,
                    metadata={
                        "conditional_test": "fixed-C conditional null",
                        "test_labels_accessed": False,
                        "permuted_target_sha256": _array_hash(permuted),
                        **metadata,
                    },
                    shared_null_computation_id=computation_id,
                    null_batch_computation_id=batch_id,
                )
                persist(shard, value, permuted)

        if completed_ids != set(bundle.shard_ids):
            raise KDiagnosticExperimentError(
                "bundle null orchestration did not exactly cover logical shards"
            )

        for shard in ordered:
            if not _existing_v2_shard_complete(
                directory=stage / "shards" / shard.shard_id,
                shard=shard,
                diagnostic=diagnostic,
                configuration_sha256=configuration_sha256,
                physical_bundle_id=bundle.bundle_id,
            ):
                raise KDiagnosticExperimentError(
                    f"bundle did not complete logical shard {shard.shard_id}"
                )
        real_file_names = list(REQUIRED_BUNDLE_REAL_FILES)
        if raw_reconstructions is not None:
            real_file_names.append("raw_reconstructions_real.npz")
        atomic_write_json(
            output / "complete.json",
            {
                "schema_version": 3,
                "complete": True,
                "physical_bundle_id": bundle.bundle_id,
                "logical_shard_ids": list(bundle.shard_ids),
                "null_execution_version": NULL_EXECUTION_VERSION,
                "real_complete_sha256": sha256_file(output / "real_complete.json"),
                "real_files": {name: sha256_file(output / name) for name in real_file_names},
            },
        )
        return completed_result()
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _run_scientific_shard_v1_legacy(
    *,
    config: ExperimentConfig,
    shard: ScientificShard,
    dictionary: Any,
    target: object,
    run_root: str | Path = ".",
    null_target: object | None = None,
    raw_target: object | None = None,
    precomputed_real: Any | None = None,
    precomputed_norms: object | None = None,
    precomputed_raw_reconstructions: Mapping[str, np.ndarray] | None = None,
    physical_bundle_id: str | None = None,
    null_target_metadata: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Original scalar v1 writer; its manifests intentionally have no v2 version."""

    diagnostic = config.k_diagnostic
    if diagnostic is None or diagnostic.identity != "qwen35_4b_k_diagnostic_v1":
        raise KDiagnosticExperimentError("legacy shard path requires v1 identity")
    if physical_bundle_id is None:
        raise KDiagnosticExperimentError(
            "scientific shards must be executed inside a physical bundle"
        )
    vector = np.asarray(target, dtype=np.float64)
    if vector.shape != (int(dictionary.d_model),) or not np.isfinite(vector).all():
        raise KDiagnosticExperimentError("target does not match dictionary metric dimension")
    raw_vector = None
    if hasattr(dictionary, "raw_reconstruction"):
        if raw_target is None:
            raise KDiagnosticExperimentError("transformed pursuit requires the original raw target")
        raw_vector = np.asarray(raw_target, dtype=np.float64)
    permuted = None
    if shard.null_family == "label_permutation_probe":
        if null_target is None:
            raise KDiagnosticExperimentError("label-permutation shard requires null_target")
        permuted = np.asarray(null_target, dtype=np.float64)
        if permuted.shape != vector.shape or not np.isfinite(permuted).all():
            raise KDiagnosticExperimentError("permutation target has wrong shape or values")
    norms = np.asarray(
        dictionary.atom_norms() if precomputed_norms is None else precomputed_norms,
        dtype=np.float64,
    )
    output = stage_root(config, run_root=run_root) / "shards" / shard.shard_id
    manifest = {
        "schema_version": 1,
        "experiment_identity": diagnostic.identity,
        "stage": diagnostic.stage,
        "scientific_identity": shard.identity,
        "scientific_identity_sha256": _payload_hash(shard.identity),
        "solver_method": diagnostic.solver_method,
        "selection_mode": diagnostic.selection_mode,
        "k_max": diagnostic.k_max,
        "configuration_sha256": _stage_manifest(config)["configuration_sha256"],
        "metric_target_sha256": _array_hash(vector),
        "raw_target_sha256": _array_hash(raw_vector) if raw_vector is not None else None,
        "permutation_target_sha256": (_array_hash(permuted) if permuted is not None else None),
        "transform_sha256": getattr(dictionary, "transform_sha256", None),
        "dictionary_cardinality": int(dictionary.n_atoms),
        "dictionary_metric_dimension": int(dictionary.d_model),
        "dictionary_raw_dimension": (
            int(dictionary.raw_d_model)
            if hasattr(dictionary, "raw_d_model")
            else int(dictionary.d_model)
        ),
        "dictionary_atom_norms_sha256": _array_hash(norms),
        "physical_bundle_id": physical_bundle_id,
    }
    if output.exists():
        if _canonical_json(_read_json(output / "manifest.json")) != _canonical_json(manifest):
            raise KDiagnosticExperimentError(f"existing shard does not match identity: {output}")
        if all((output / name).is_file() for name in REQUIRED_SHARD_FILES):
            return _read_json(output / "summary.json")
        raise KDiagnosticExperimentError(f"partial shard exists and is not reusable: {output}")

    real = precomputed_real
    if real is None:
        real = streaming_nonnegative_pursuit(
            dictionary,
            vector[None, :],
            k_max=diagnostic.k_max,
            selection_modes=diagnostic.selection_mode,
            solver_method=diagnostic.solver_method,
        )[0]
    real_errors = _validate_curve(real.errors, k_max=diagnostic.k_max, label="real")
    metadata = {
        "null_family": shard.null_family,
        "null_seed": shard.null_seed,
        "dictionary_fraction": shard.dictionary_fraction,
        "metric_dimension": int(dictionary.d_model),
        "real_dictionary_cardinality": int(dictionary.n_atoms),
    }
    if shard.null_family in {"iid_full_cardinality", "iid_cardinality_sweep"}:
        selected_norms, subset = matched_norm_subset(
            norms, fraction=shard.dictionary_fraction, seed=shard.null_seed
        )
        random_dictionary = MatchedNormRandomDictionary(
            selected_norms,
            seed=shard.null_seed,
            d_model=int(dictionary.d_model),
            chunk_size=diagnostic.vocabulary_chunk_size,
        )
        null = streaming_nonnegative_pursuit(
            random_dictionary,
            vector[None, :],
            k_max=diagnostic.k_max,
            selection_modes=diagnostic.selection_mode,
            solver_method=diagnostic.solver_method,
        )[0]
        metadata.update(subset)
        metadata["search_semantics"] = "full best-atom scan at registered cardinality"
    elif shard.null_family == "random_prefix_same_k":
        atoms, prefix = random_prefix_atoms(
            norms,
            k_max=diagnostic.k_max,
            seed=shard.null_seed,
            d_model=int(dictionary.d_model),
        )
        null = random_prefix_pursuit(atoms, vector)
        metadata.update(prefix)
        metadata["search_semantics"] = "fixed nested prefix; no vocabulary search"
    elif shard.null_family == "orthogonal_rotation":
        rotation, rotation_manifest = load_cached_haar_rotation(
            artifact_root(config, run_root=run_root) / "rotation_cache",
            metric=shard.metric,
            transform_sha256=getattr(dictionary, "transform_sha256", None),
            dimension=int(dictionary.d_model),
            seed=shard.null_seed,
            create=False,
        )
        rotated, preservation_error = inverse_rotate_target(vector, rotation)
        null = streaming_nonnegative_pursuit(
            dictionary,
            rotated[None, :],
            k_max=diagnostic.k_max,
            selection_modes=diagnostic.selection_mode,
            solver_method=diagnostic.solver_method,
        )[0]
        metadata.update(rotation_manifest["numerical_validation"])
        metadata["rotation_cache_id"] = rotation_manifest["cache_id"]
        metadata["rotation_matrix_sha256"] = rotation_manifest["matrix_sha256"]
        metadata["target_relative_norm_error"] = preservation_error
        metadata["rotation_space"] = shard.metric
    elif shard.null_family == "label_permutation_probe":
        assert permuted is not None
        null = streaming_nonnegative_pursuit(
            dictionary,
            permuted[None, :],
            k_max=diagnostic.k_max,
            selection_modes=diagnostic.selection_mode,
            solver_method=diagnostic.solver_method,
        )[0]
        metadata.update(
            {
                "conditional_test": "fixed-C conditional null",
                "test_labels_accessed": False,
                "permuted_target_sha256": _array_hash(permuted),
            }
        )
        if null_target_metadata is not None:
            metadata.update(dict(null_target_metadata))
    else:
        raise KDiagnosticExperimentError(f"unknown null family: {shard.null_family}")
    null_errors = _validate_curve(null.errors, k_max=diagnostic.k_max, label=shard.null_family)
    null_gains = null_errors[:-1] - null_errors[1:]
    summary = {
        "schema_version": 1,
        **shard.identity,
        "shard_id": shard.shard_id,
        "real_target_norm": float(real.target_norm),
        "real_support_size": int(real.support.size),
        "real_stopped_early_at": real.stopped_early_at,
        "real_final_explained_fraction": float(1.0 - real_errors[-1]),
        "null_final_explained_fraction": float(1.0 - null_errors[-1]),
        "fixed_budget": {
            str(k): {
                "real_explained_fraction": float(1.0 - real_errors[k]),
                "null_explained_fraction": float(1.0 - null_errors[k]),
                "semantics": "fixed reconstruction budget, not estimated occupancy",
            }
            for k in diagnostic.report_grid
        },
    }
    bundle_directory = stage_root(config, run_root=run_root) / "bundles" / physical_bundle_id
    real_files = list(REQUIRED_BUNDLE_REAL_FILES)
    if hasattr(dictionary, "raw_reconstruction"):
        real_files.append("raw_reconstructions_real.npz")
    real_reference = {
        "schema_version": 1,
        "physical_bundle_id": physical_bundle_id,
        "files": {name: sha256_file(bundle_directory / name) for name in real_files},
    }
    if hasattr(dictionary, "raw_reconstruction"):
        assert raw_vector is not None
        if precomputed_raw_reconstructions is None:
            raise KDiagnosticExperimentError(
                "transformed bundle did not supply its shared raw reconstructions"
            )
        squared_raw_norm = float(raw_vector @ raw_vector)
        for k in diagnostic.report_grid:
            reconstruction = np.asarray(
                precomputed_raw_reconstructions[f"k_{k:03d}"], dtype=np.float64
            )
            raw_error = float(np.sum((raw_vector - reconstruction) ** 2) / squared_raw_norm)
            summary["fixed_budget"][str(k)]["raw_space_error"] = raw_error
            summary["fixed_budget"][str(k)]["raw_space_explained_fraction"] = 1.0 - raw_error
    output.mkdir(parents=True, exist_ok=False)
    _atomic_save_npy(output / "null_errors.npy", null_errors[None, :])
    _atomic_save_npy(output / "null_gains.npy", null_gains[None, :])
    atomic_write_json(output / "real_reference.json", real_reference)
    atomic_write_json(output / "null_metadata.json", metadata)
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(output / "manifest.json", manifest)
    return summary


def _run_scientific_bundle_v1_legacy(
    *,
    config: ExperimentConfig,
    bundle: PhysicalBundle,
    shards: Sequence[ScientificShard],
    dictionary: Any,
    target: object,
    run_root: str | Path = ".",
    permutation_targets: Mapping[str, object] | None = None,
    raw_target: object | None = None,
    token_decoder: Any = None,
) -> dict[str, dict[str, Any]]:
    """Preserve the scalar v1 bundle/resume and schema contract."""

    diagnostic = config.k_diagnostic
    if diagnostic is None or diagnostic.identity != "qwen35_4b_k_diagnostic_v1":
        raise KDiagnosticExperimentError("legacy bundle path requires v1 identity")
    ordered = sorted(shards, key=lambda shard: _canonical_json(shard.identity))
    if tuple(sorted(shard.shard_id for shard in ordered)) != bundle.shard_ids:
        raise KDiagnosticExperimentError("bundle logical shard membership mismatch")
    vector = np.asarray(target, dtype=np.float64)
    norms = np.asarray(dictionary.atom_norms(), dtype=np.float64)
    configuration_sha256 = _stage_manifest(config)["configuration_sha256"]
    bundle_manifest = {
        "schema_version": 2,
        "experiment_identity": diagnostic.identity,
        "stage": diagnostic.stage,
        "physical_identity": bundle.scientific_identity,
        "physical_bundle_id": bundle.bundle_id,
        "logical_shard_ids": list(bundle.shard_ids),
        "logical_shard_count": len(bundle.shard_ids),
        "configuration_sha256": configuration_sha256,
        "metric_target_sha256": _array_hash(vector),
        "raw_target_sha256": _array_hash(raw_target) if raw_target is not None else None,
        "transform_sha256": getattr(dictionary, "transform_sha256", None),
        "dictionary_cardinality": int(dictionary.n_atoms),
        "dictionary_metric_dimension": int(dictionary.d_model),
        "dictionary_raw_dimension": (
            int(dictionary.raw_d_model)
            if hasattr(dictionary, "raw_d_model")
            else int(dictionary.d_model)
        ),
        "dictionary_atom_norms_sha256": _array_hash(norms),
        "real_solve_count": 1,
        "atom_norm_pass_count": 1,
    }
    stage = stage_root(config, run_root=run_root)
    output = stage / "bundles" / bundle.bundle_id
    lock_root = stage / "bundle_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock = lock_root / f"{bundle.bundle_id}.lock"

    def completed_result() -> dict[str, dict[str, Any]]:
        if _canonical_json(_read_json(output / "manifest.json")) != _canonical_json(
            bundle_manifest
        ):
            raise KDiagnosticExperimentError(f"bundle cache identity mismatch: {output}")
        complete = _read_json(output / "complete.json")
        if complete.get("complete") is not True or complete.get("logical_shard_ids") != list(
            bundle.shard_ids
        ):
            raise KDiagnosticExperimentError(f"bundle cache is partial: {output}")
        summaries = {}
        verified_real_files: dict[Path, str] = {}
        for shard in ordered:
            summary, _real, _null = _load_and_audit_shard(
                stage / "shards" / shard.shard_id,
                shard,
                diagnostic=diagnostic,
                configuration_sha256=configuration_sha256,
                verified_real_files=verified_real_files,
                verified_rotation_manifests=None,
            )
            summaries[shard.shard_id] = summary
        return summaries

    if output.exists():
        if not (output / "complete.json").is_file():
            raise KDiagnosticExperimentError(f"partial bundle cache is not reusable: {output}")
        return completed_result()
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise KDiagnosticExperimentError(
            f"bundle {bundle.bundle_id} is already locked by another worker"
        ) from error
    os.close(descriptor)
    try:
        output.mkdir(parents=True, exist_ok=False)
        atomic_write_json(output / "manifest.json", bundle_manifest)
        real = streaming_nonnegative_pursuit(
            dictionary,
            vector[None, :],
            k_max=diagnostic.k_max,
            selection_modes=diagnostic.selection_mode,
            solver_method=diagnostic.solver_method,
        )[0]
        real_errors = _validate_curve(real.errors, k_max=diagnostic.k_max, label="real")
        _atomic_save_npy(output / "errors_real.npy", real_errors)
        _atomic_save_npy(output / "gains_real.npy", real.gains)
        _atomic_save_npy(output / "support_real.npy", real.support)
        _atomic_save_npz(output / "coefficients_real.npz", _coefficients_payload(real))
        fixed_budget_payload, raw_reconstructions = _fixed_budget_real_payload(
            result=real,
            dictionary=dictionary,
            target=vector,
            raw_target=raw_target,
            token_decoder=token_decoder,
        )
        atomic_write_json(output / "fixed_budget_real.json", fixed_budget_payload)
        if raw_reconstructions is not None:
            _atomic_save_npz(output / "raw_reconstructions_real.npz", raw_reconstructions)
        for shard in ordered:
            null_target = None
            null_target_metadata = None
            if shard.null_family == "label_permutation_probe":
                if permutation_targets is None or shard.shard_id not in permutation_targets:
                    raise KDiagnosticExperimentError(
                        f"bundle lacks permutation target for {shard.shard_id}"
                    )
                value = permutation_targets[shard.shard_id]
                if isinstance(value, tuple) and len(value) == 2:
                    null_target, null_target_metadata = value
                else:
                    null_target = value
            _run_scientific_shard_v1_legacy(
                config=config,
                shard=shard,
                dictionary=dictionary,
                target=vector,
                run_root=run_root,
                null_target=null_target,
                raw_target=raw_target,
                precomputed_real=real,
                precomputed_norms=norms,
                precomputed_raw_reconstructions=raw_reconstructions,
                physical_bundle_id=bundle.bundle_id,
                null_target_metadata=null_target_metadata,
            )
        real_names = list(REQUIRED_BUNDLE_REAL_FILES)
        if raw_reconstructions is not None:
            real_names.append("raw_reconstructions_real.npz")
        atomic_write_json(
            output / "complete.json",
            {
                "schema_version": 2,
                "complete": True,
                "physical_bundle_id": bundle.bundle_id,
                "logical_shard_ids": list(bundle.shard_ids),
                "real_files": {name: sha256_file(output / name) for name in real_names},
            },
        )
        return completed_result()
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def run_scientific_bundle(
    *,
    config: ExperimentConfig,
    bundle: PhysicalBundle,
    shards: Sequence[ScientificShard],
    dictionary: Any,
    target: object,
    run_root: str | Path = ".",
    permutation_targets: Mapping[str, object] | None = None,
    raw_target: object | None = None,
    token_decoder: Any = None,
) -> dict[str, dict[str, Any]]:
    """Dispatch without changing the immutable v1 execution/artifact contract."""

    diagnostic = config.k_diagnostic
    if diagnostic is None:
        raise KDiagnosticExperimentError("configuration has no k_diagnostic section")
    kwargs = {
        "config": config,
        "bundle": bundle,
        "shards": shards,
        "dictionary": dictionary,
        "target": target,
        "run_root": run_root,
        "permutation_targets": permutation_targets,
        "raw_target": raw_target,
        "token_decoder": token_decoder,
    }
    if diagnostic.identity == "qwen35_4b_k_diagnostic_v1":
        return _run_scientific_bundle_v1_legacy(**kwargs)
    if diagnostic.identity == "qwen35_4b_k_diagnostic_v2":
        return _run_scientific_bundle_v2(**kwargs)
    raise KDiagnosticExperimentError(f"unsupported K-diagnostic identity: {diagnostic.identity}")


def _load_and_audit_shard(
    directory: Path,
    expected: ScientificShard,
    *,
    diagnostic: Any,
    configuration_sha256: str,
    verified_real_files: dict[Path, str] | None = None,
    verified_rotation_manifests: dict[tuple[object, ...], Mapping[str, object]] | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    k_max = int(diagnostic.k_max)
    missing = [name for name in REQUIRED_SHARD_FILES if not (directory / name).is_file()]
    if missing:
        raise KDiagnosticExperimentError(f"shard {expected.shard_id} misses {missing}")
    manifest = _read_json(directory / "manifest.json")
    expected_manifest_fields = {
        "experiment_identity": diagnostic.identity,
        "stage": diagnostic.stage,
        "scientific_identity": expected.identity,
        "scientific_identity_sha256": _payload_hash(expected.identity),
        "solver_method": diagnostic.solver_method,
        "selection_mode": diagnostic.selection_mode,
        "k_max": k_max,
        "configuration_sha256": configuration_sha256,
    }
    if diagnostic.identity == "qwen35_4b_k_diagnostic_v2":
        expected_manifest_fields["null_execution_version"] = NULL_EXECUTION_VERSION
    for field, value in expected_manifest_fields.items():
        if manifest.get(field) != value:
            raise KDiagnosticExperimentError(f"shard manifest {field} mismatch: {directory}")
    for field in ("metric_target_sha256", "dictionary_atom_norms_sha256"):
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise KDiagnosticExperimentError(f"invalid {field}: {directory}")
    cardinality = manifest.get("dictionary_cardinality")
    dimension = manifest.get("dictionary_metric_dimension")
    if not isinstance(cardinality, int) or cardinality < k_max:
        raise KDiagnosticExperimentError(f"invalid dictionary cardinality: {directory}")
    if not isinstance(dimension, int) or dimension < 1:
        raise KDiagnosticExperimentError(f"invalid dictionary dimension: {directory}")
    physical_bundle_id = manifest.get("physical_bundle_id")
    if not isinstance(physical_bundle_id, str) or not physical_bundle_id:
        raise KDiagnosticExperimentError(f"invalid physical bundle ID: {directory}")
    real_reference = _read_json(directory / "real_reference.json")
    if (
        real_reference.get("schema_version") != 1
        or real_reference.get("physical_bundle_id") != physical_bundle_id
        or not isinstance(real_reference.get("files"), Mapping)
    ):
        raise KDiagnosticExperimentError(f"invalid bundle real reference: {directory}")
    bundle_directory = directory.parent.parent / "bundles" / physical_bundle_id
    bundle_complete = _read_json(bundle_directory / "complete.json")
    bundle_manifest = _read_json(bundle_directory / "manifest.json")
    reference_files = real_reference["files"]
    required_real_files = set(REQUIRED_BUNDLE_REAL_FILES)
    if diagnostic.stage == "transformed_metric":
        required_real_files.add("raw_reconstructions_real.npz")
    if set(reference_files) != required_real_files:
        raise KDiagnosticExperimentError(f"bundle real file set mismatch: {directory}")
    if (
        bundle_complete.get("complete") is not True
        or bundle_complete.get("physical_bundle_id") != physical_bundle_id
        or bundle_complete.get("real_files") != reference_files
    ):
        raise KDiagnosticExperimentError(f"bundle complete hash set mismatch: {directory}")
    if diagnostic.identity == "qwen35_4b_k_diagnostic_v2":
        real_complete_path = bundle_directory / "real_complete.json"
        real_complete = _read_json(real_complete_path)
        if (
            bundle_complete.get("null_execution_version") != NULL_EXECUTION_VERSION
            or bundle_complete.get("real_complete_sha256") != sha256_file(real_complete_path)
            or bundle_manifest.get("null_execution_version") != NULL_EXECUTION_VERSION
            or bundle_manifest.get("configuration_sha256") != configuration_sha256
            or (
                bundle_manifest.get("shared_direction_memory_plan") is not None
                and not _approved_shared_direction_plan(
                    bundle_manifest.get("shared_direction_memory_plan"),
                    n_atoms=bundle_manifest.get("dictionary_cardinality"),
                    d_model=bundle_manifest.get("dictionary_metric_dimension"),
                    chunk_size=diagnostic.vocabulary_chunk_size,
                )
            )
            or real_complete.get("complete") is not True
            or real_complete.get("null_execution_version") != NULL_EXECUTION_VERSION
            or real_complete.get("real_files") != reference_files
        ):
            raise KDiagnosticExperimentError(
                f"v2 bundle execution hash/version mismatch: {directory}"
            )
    for name, expected_hash in reference_files.items():
        source = bundle_directory / str(name)
        observed_hash = verified_real_files.get(source) if verified_real_files is not None else None
        if observed_hash is None and source.is_file():
            observed_hash = sha256_file(source)
            if verified_real_files is not None:
                verified_real_files[source] = observed_hash
        if observed_hash != expected_hash:
            raise KDiagnosticExperimentError(
                f"bundle real payload hash mismatch for {name}: {directory}"
            )
    if diagnostic.stage == "transformed_metric":
        for field in ("raw_target_sha256", "transform_sha256"):
            value = manifest.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise KDiagnosticExperimentError(f"invalid {field}: {directory}")
        raw_dimension = manifest.get("dictionary_raw_dimension")
        raw_archive_path = bundle_directory / "raw_reconstructions_real.npz"
        if not isinstance(raw_dimension, int) or raw_dimension < dimension:
            raise KDiagnosticExperimentError(f"invalid raw dictionary dimension: {directory}")
        if not raw_archive_path.is_file():
            raise KDiagnosticExperimentError(f"missing raw reconstructions: {directory}")
        with np.load(raw_archive_path, allow_pickle=False) as raw_archive:
            expected_raw_names = [f"k_{k:03d}" for k in FIXED_BUDGETS if k <= k_max]
            if raw_archive.files != expected_raw_names:
                raise KDiagnosticExperimentError(
                    f"raw reconstruction budget keys mismatch: {directory}"
                )
            for name in expected_raw_names:
                if (
                    raw_archive[name].shape != (raw_dimension,)
                    or not np.isfinite(raw_archive[name]).all()
                ):
                    raise KDiagnosticExperimentError(
                        f"invalid raw reconstruction {name}: {directory}"
                    )
    if expected.null_family == "label_permutation_probe":
        value = manifest.get("permutation_target_sha256")
        if not isinstance(value, str) or len(value) != 64:
            raise KDiagnosticExperimentError(f"invalid permutation_target_sha256: {directory}")
    summary = _read_json(directory / "summary.json")
    if summary.get("shard_id") != expected.shard_id:
        raise KDiagnosticExperimentError(f"shard summary ID mismatch: {directory}")
    for field, value in expected.identity.items():
        if summary.get(field) != value:
            raise KDiagnosticExperimentError(f"shard summary {field} mismatch: {directory}")
    real = _validate_curve(
        np.load(bundle_directory / "errors_real.npy", allow_pickle=False),
        k_max=k_max,
        label=f"{expected.shard_id} real",
    )
    fixed_payload = _read_json(bundle_directory / "fixed_budget_real.json")
    if (
        fixed_payload.get("schema_version") != 1
        or fixed_payload.get("fixed_budgets") != list(FIXED_BUDGETS)
        or not isinstance(fixed_payload.get("budgets"), Mapping)
        or set(fixed_payload["budgets"]) != {str(k) for k in FIXED_BUDGETS}
    ):
        raise KDiagnosticExperimentError(f"bundle fixed-budget schema mismatch: {directory}")
    for requested in FIXED_BUDGETS:
        budget = fixed_payload["budgets"][str(requested)]
        if requested > k_max:
            if budget.get("available") is not False:
                raise KDiagnosticExperimentError(
                    f"unavailable fixed budget is not marked at K={requested}: {directory}"
                )
            continue
        required_budget_fields = {
            "effective_support_size",
            "cosine_target_reconstruction",
            "residual_norm",
            "normalized_residual_norm",
            "metric_space_error",
            "metric_space_explained_fraction",
            "selected_token_ids",
            "decoded_tokens",
            "coefficients",
            "selected_atom_gram_condition_number",
        }
        if (
            budget.get("available") is not True
            or not required_budget_fields.issubset(budget)
            or len(budget["selected_token_ids"]) != len(budget["decoded_tokens"])
            or len(budget["selected_token_ids"]) != len(budget["coefficients"])
            or not np.isclose(
                float(budget["metric_space_error"]),
                real[requested],
                rtol=0.0,
                atol=1e-10,
            )
        ):
            raise KDiagnosticExperimentError(
                f"invalid fixed-budget telemetry at K={requested}: {directory}"
            )
    null = np.load(directory / "null_errors.npy", allow_pickle=False)
    if null.shape != (1, k_max + 1):
        raise KDiagnosticExperimentError(f"null curve shape mismatch: {directory}")
    checked_null = _validate_curve(null[0], k_max=k_max, label=f"{expected.shard_id} null")
    real_gains = np.load(bundle_directory / "gains_real.npy", allow_pickle=False)
    null_gains = np.load(directory / "null_gains.npy", allow_pickle=False)
    if real_gains.shape != (k_max,) or not np.allclose(
        real_gains, real[:-1] - real[1:], rtol=1e-12, atol=1e-12
    ):
        raise KDiagnosticExperimentError(f"real gains mismatch: {directory}")
    if null_gains.shape != (1, k_max) or not np.allclose(
        null_gains[0], checked_null[:-1] - checked_null[1:], rtol=1e-12, atol=1e-12
    ):
        raise KDiagnosticExperimentError(f"null gains mismatch: {directory}")
    support = np.load(bundle_directory / "support_real.npy", allow_pickle=False)
    if (
        support.ndim != 1
        or support.size > k_max
        or support.dtype.kind not in "iu"
        or np.unique(support).size != support.size
        or np.any(support < 0)
        or np.any(support >= cardinality)
        or summary.get("real_support_size") != int(support.size)
    ):
        raise KDiagnosticExperimentError(f"invalid real support: {directory}")
    with np.load(bundle_directory / "coefficients_real.npz", allow_pickle=False) as archive:
        expected_names = [f"k_{k:03d}" for k in range(support.size + 1)]
        if archive.files != expected_names:
            raise KDiagnosticExperimentError(f"coefficient path keys mismatch: {directory}")
        for k, name in enumerate(expected_names):
            values = archive[name]
            if values.shape != (k,) or not np.isfinite(values).all() or np.any(values < -1e-12):
                raise KDiagnosticExperimentError(f"invalid coefficients {name}: {directory}")
    null_metadata = _read_json(directory / "null_metadata.json")
    if (
        null_metadata.get("null_family") != expected.null_family
        or null_metadata.get("null_seed") != expected.null_seed
        or null_metadata.get("dictionary_fraction") != expected.dictionary_fraction
        or null_metadata.get("metric_dimension") != dimension
        or null_metadata.get("real_dictionary_cardinality") != cardinality
    ):
        raise KDiagnosticExperimentError(f"null metadata mismatch: {directory}")
    if diagnostic.identity == "qwen35_4b_k_diagnostic_v2":
        observed_curve_hash = _null_curve_sha256(checked_null, null_gains[0])
        shared_fields = (
            "shared_null_computation_id",
            "source_curve_sha256",
            "null_batch_computation_id",
        )
        for field in shared_fields:
            manifest_value = manifest.get(field)
            metadata_value = null_metadata.get(field)
            if (
                not isinstance(manifest_value, str)
                or not manifest_value
                or manifest_value != metadata_value
            ):
                raise KDiagnosticExperimentError(f"v2 shared-null {field} mismatch: {directory}")
        if (
            manifest["source_curve_sha256"] != observed_curve_hash
            or null_metadata.get("null_execution_version") != NULL_EXECUTION_VERSION
        ):
            raise KDiagnosticExperimentError(
                f"v2 shared-null curve hash/version mismatch: {directory}"
            )
    if expected.null_family == "orthogonal_rotation":
        orthogonality_error = null_metadata.get("normalized_frobenius_orthogonality_error")
        norm_error = null_metadata.get("target_relative_norm_error")
        if (
            not isinstance(orthogonality_error, int | float)
            or not 0.0 <= orthogonality_error < 1e-5
            or not isinstance(norm_error, int | float)
            or not 0.0 <= norm_error < 1e-6
        ):
            raise KDiagnosticExperimentError(f"rotation numerical validation failed: {directory}")
        rotation_key = (
            expected.metric,
            manifest.get("transform_sha256"),
            dimension,
            expected.null_seed,
        )
        if verified_rotation_manifests is not None:
            rotation_manifest = verified_rotation_manifests.get(rotation_key)
            if rotation_manifest is None:
                raise KDiagnosticExperimentError(
                    "rotation shard key is absent from the audited binding index"
                )
        else:
            _rotation, rotation_manifest = load_cached_haar_rotation(
                directory.parents[2] / "rotation_cache",
                metric=expected.metric,
                transform_sha256=manifest.get("transform_sha256"),
                dimension=dimension,
                seed=expected.null_seed,
                create=False,
            )
        if (
            null_metadata.get("rotation_cache_id") != rotation_manifest["cache_id"]
            or null_metadata.get("rotation_matrix_sha256") != rotation_manifest["matrix_sha256"]
        ):
            raise KDiagnosticExperimentError(
                f"rotation shard is not pinned to its cached matrix: {directory}"
            )
    if expected.null_family == "label_permutation_probe":
        for split in ("train_permutation", "validation_permutation"):
            metadata = null_metadata.get(split)
            if (
                not isinstance(metadata, Mapping)
                or int(metadata.get("hamming_distance", 0)) <= 0
                or float(metadata.get("non_identity_rate", 0.0)) <= 0.0
                or int(metadata.get("effective_permutable_groups", 0)) <= 0
            ):
                raise KDiagnosticExperimentError(
                    f"ineffective label permutation metadata for {split}: {directory}"
                )
    return summary, real, checked_null


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the registered columnar summary; pyarrow is an llm/dev dependency."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - exercised by minimal envs
        raise KDiagnosticExperimentError(
            "summary.parquet requires the registered llm/dev environment (pyarrow)"
        ) from error
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        pq.write_table(pa.Table.from_pylist(list(rows)), temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def index_stage(
    config: ExperimentConfig,
    grid: Sequence[ScientificShard],
    *,
    run_root: str | Path = ".",
) -> dict[str, Any]:
    """Audit the complete registered grid and compute seed-calibrated summaries."""

    diagnostic = config.k_diagnostic
    if diagnostic is None:
        raise KDiagnosticExperimentError("configuration has no k_diagnostic section")
    root = stage_root(config, run_root=run_root)
    verified_rotation_manifests: dict[tuple[object, ...], Mapping[str, object]] = {}
    if diagnostic.identity == "qwen35_4b_k_diagnostic_v2" and any(
        shard.null_family == "orthogonal_rotation" for shard in grid
    ):
        rotation_index = audit_rotation_cache_for_grid(config, grid, run_root=run_root)
        for entry in rotation_index["entries"]:
            key = (
                entry["metric"],
                entry.get("transform_sha256"),
                int(entry["dimension"]),
                int(entry["seed"]),
            )
            verified_rotation_manifests[key] = {
                "cache_id": entry["cache_id"],
                "matrix_sha256": entry["matrix_sha256"],
            }
    shard_root = root / "shards"
    expected_ids = {shard.shard_id for shard in grid}
    observed_ids = (
        {path.name for path in shard_root.iterdir() if path.is_dir()}
        if shard_root.is_dir()
        else set()
    )
    unexpected = sorted(observed_ids - expected_ids)
    if unexpected:
        raise KDiagnosticExperimentError(f"unregistered shard directories found: {unexpected}")
    missing = [shard.shard_id for shard in grid if shard.shard_id not in observed_ids]
    if missing:
        payload = {
            "schema_version": 1,
            "identity": diagnostic.identity,
            "stage": diagnostic.stage,
            "complete": False,
            "expected_shards": len(grid),
            "completed_shards": len(grid) - len(missing),
            "missing_shards": missing,
        }
        atomic_write_json(root / "index.json", payload)
        return payload

    bundles = build_physical_bundles(grid)
    if diagnostic.identity == "qwen35_4b_k_diagnostic_v2":
        expected_bundle_ids = {bundle.bundle_id for bundle in bundles}
        configuration_sha256 = _stage_manifest(config)["configuration_sha256"]
        preflight = _read_json(root / "resource_preflight.json")
        preflight_hardware = validate_k_diagnostic_execution_hardware(
            preflight.get("execution_hardware")
        )
        expected_hardware_sha256 = execution_hardware_sha256(preflight_hardware)
        if (
            preflight.get("approved") is not True
            or preflight.get("identity") != diagnostic.identity
            or preflight.get("stage") != diagnostic.stage
            or preflight.get("configuration_sha256") != configuration_sha256
            or preflight.get("null_execution_version") != NULL_EXECUTION_VERSION
            or preflight.get("execution_hardware_sha256") != expected_hardware_sha256
        ):
            raise KDiagnosticExperimentError(
                "stage index resource preflight hardware identity is stale"
            )
        runtime_root = root / "runtimes"
        runtime_paths = sorted(runtime_root.glob("*.json")) if runtime_root.is_dir() else []
        if {path.stem for path in runtime_paths} != expected_bundle_ids:
            raise KDiagnosticExperimentError(
                "stage index runtimes do not exactly cover physical bundles"
            )
        for runtime_path in runtime_paths:
            runtime = _read_json(runtime_path)
            runtime_hardware = validate_k_diagnostic_execution_hardware(
                runtime.get("execution_hardware")
            )
            if (
                runtime.get("identity") != diagnostic.identity
                or runtime.get("stage") != diagnostic.stage
                or runtime.get("configuration_sha256") != configuration_sha256
                or runtime.get("bundle_id") != runtime_path.stem
                or runtime.get("null_execution_version") != NULL_EXECUTION_VERSION
                or runtime.get("execution_backend") != runtime_hardware["execution_backend"]
                or runtime.get("execution_hardware_sha256")
                != execution_hardware_sha256(runtime_hardware)
                or runtime.get("execution_hardware_sha256") != expected_hardware_sha256
            ):
                raise KDiagnosticExperimentError(
                    f"stage index runtime hardware audit failed: {runtime_path}"
                )
        bundle_root = root / "bundles"
        observed_bundle_ids = (
            {path.name for path in bundle_root.iterdir() if path.is_dir()}
            if bundle_root.is_dir()
            else set()
        )
        if observed_bundle_ids != expected_bundle_ids:
            raise KDiagnosticExperimentError(
                "physical bundle directories do not exactly match the sealed bundle grid"
            )
        grid_by_id = {shard.shard_id: shard for shard in grid}
        for bundle in bundles:
            directory = bundle_root / bundle.bundle_id
            complete = _read_json(directory / "complete.json")
            manifest = _read_json(directory / "manifest.json")
            requires_shared_directions = any(
                grid_by_id[shard_id].null_family
                in {"iid_full_cardinality", "iid_cardinality_sweep"}
                for shard_id in bundle.shard_ids
            )
            bundle_direction_plan = manifest.get("shared_direction_memory_plan")
            if (
                complete.get("complete") is not True
                or complete.get("logical_shard_ids") != list(bundle.shard_ids)
                or complete.get("null_execution_version") != NULL_EXECUTION_VERSION
                or not (directory / "real_complete.json").is_file()
                or complete.get("real_complete_sha256")
                != sha256_file(directory / "real_complete.json")
                or manifest.get("physical_identity") != bundle.scientific_identity
                or manifest.get("null_execution_version") != NULL_EXECUTION_VERSION
                or manifest.get("real_solve_count") != 1
                or manifest.get("atom_norm_pass_count") != 1
                or (
                    requires_shared_directions
                    and not _approved_shared_direction_plan(
                        bundle_direction_plan,
                        n_atoms=manifest.get("dictionary_cardinality"),
                        d_model=manifest.get("dictionary_metric_dimension"),
                        chunk_size=diagnostic.vocabulary_chunk_size,
                    )
                )
                or (not requires_shared_directions and bundle_direction_plan is not None)
            ):
                raise KDiagnosticExperimentError(
                    f"physical bundle audit failed: {bundle.bundle_id}"
                )

    grouped: dict[tuple[object, ...], list[tuple[dict[str, Any], np.ndarray, np.ndarray]]] = (
        defaultdict(list)
    )
    configuration_sha256 = _stage_manifest(config)["configuration_sha256"]
    verified_real_files: dict[Path, str] = {}
    for shard in grid:
        audited = _load_and_audit_shard(
            shard_root / shard.shard_id,
            shard,
            diagnostic=diagnostic,
            configuration_sha256=configuration_sha256,
            verified_real_files=verified_real_files,
            verified_rotation_manifests=verified_rotation_manifests,
        )
        key = (
            shard.target_id,
            shard.layer,
            shard.target_family,
            shard.target_subtype,
            shard.metric,
            shard.null_family,
            shard.dictionary_fraction,
        )
        grouped[key].append(audited)

    if diagnostic.identity == "qwen35_4b_k_diagnostic_v2":
        aliases: dict[tuple[object, ...], dict[str, ScientificShard]] = defaultdict(dict)
        for shard in grid:
            if shard.dictionary_fraction == 1.0 and shard.null_family in {
                "iid_full_cardinality",
                "iid_cardinality_sweep",
            }:
                alias_key = (
                    shard.target_id,
                    shard.layer,
                    shard.target_family,
                    shard.target_subtype,
                    shard.metric,
                    shard.null_seed,
                )
                aliases[alias_key][shard.null_family] = shard
        for alias_key, families in aliases.items():
            if set(families) != {
                "iid_full_cardinality",
                "iid_cardinality_sweep",
            }:
                continue
            full_manifest = _read_json(
                shard_root / families["iid_full_cardinality"].shard_id / "manifest.json"
            )
            sweep_manifest = _read_json(
                shard_root / families["iid_cardinality_sweep"].shard_id / "manifest.json"
            )
            for field in (
                "shared_null_computation_id",
                "source_curve_sha256",
                "null_batch_computation_id",
            ):
                if full_manifest.get(field) != sweep_manifest.get(field):
                    raise KDiagnosticExperimentError(
                        f"IID full/fraction-1 alias {field} mismatch: {alias_key}"
                    )

    rows: list[dict[str, Any]] = []
    for key, results in sorted(grouped.items(), key=lambda item: repr(item[0])):
        reals = np.stack([result[1] for result in results])
        if not np.allclose(reals, reals[0], rtol=1e-12, atol=1e-12):
            raise KDiagnosticExperimentError(f"real curve changed across null seeds: {key}")
        nulls = np.stack([result[2] for result in results])
        calibrated = summarize_null_calibrated_curve(
            reals[0], nulls, report_grid=diagnostic.report_grid
        )
        row = {
            "target_id": key[0],
            "layer": key[1],
            "target_family": key[2],
            "target_subtype": key[3],
            "metric": key[4],
            "null_family": key[5],
            "dictionary_fraction": key[6],
            "null_seeds": sorted(int(result[0]["null_seed"]) for result in results),
            **calibrated,
        }
        rows.append(row)

    _write_parquet(root / "summary.parquet", rows)
    payload = {
        "schema_version": 1,
        "identity": diagnostic.identity,
        "stage": diagnostic.stage,
        "complete": True,
        "expected_shards": len(grid),
        "completed_shards": len(grid),
        "physical_bundle_count": len(bundles),
        "logical_replicate_count": len(grid),
        "duplicate_scientific_identities": 0,
        "missing_shards": [],
        "target_families": sorted({row["target_family"] for row in rows}),
        "layers": sorted({int(row["layer"]) for row in rows}),
        "null_families": sorted({row["null_family"] for row in rows}),
        "metrics": sorted({row["metric"] for row in rows}),
        "summary_rows": len(rows),
        "summary_sha256": sha256_file(root / "summary.parquet"),
        "validation": {
            "all_error_curves_finite": True,
            "all_error_curves_non_increasing": True,
            "all_gain_curves_match_errors": True,
            "all_coefficients_nonnegative": True,
            "all_supports_valid": True,
            "all_manifests_match_registered_configuration": True,
            "all_rotation_tolerances_satisfied": True,
            "all_target_and_source_hashes_verified": True,
        },
    }
    if diagnostic.identity == "qwen35_4b_k_diagnostic_v2":
        payload["schema_version"] = 2
        payload["null_execution_version"] = NULL_EXECUTION_VERSION
    atomic_write_json(root / "index.json", payload)
    return payload


def _save_npy_exact(path: Path, values: object) -> str:
    array = np.asarray(values)
    if path.is_file():
        observed = np.load(path, allow_pickle=False)
        if not np.array_equal(observed, array):
            raise KDiagnosticExperimentError(
                f"existing basis array differs from registered computation: {path}"
            )
    else:
        _atomic_save_npy(path, array)
    return sha256_file(path)


def _energy_rank(eigenvalues: np.ndarray, threshold: float) -> int:
    positive = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    total = float(positive.sum())
    if total <= 0.0:
        raise KDiagnosticExperimentError("basis spectrum has no positive energy")
    return int(np.searchsorted(np.cumsum(positive), threshold * total) + 1)


def _activation_covariance(
    path: Path, indices: np.ndarray, *, chunk_size: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    values = np.load(path, mmap_mode="r")
    if values.ndim != 2 or indices.size < 2:
        raise KDiagnosticExperimentError("activation covariance requires [N,D] and N>=2")
    dimension = int(values.shape[1])
    total = np.zeros(dimension, dtype=np.float64)
    second = np.zeros((dimension, dimension), dtype=np.float64)
    for start in range(0, indices.size, chunk_size):
        block = np.asarray(values[indices[start : start + chunk_size]], dtype=np.float64)
        total += block.sum(axis=0, dtype=np.float64)
        second += block.T @ block
    mean = total / indices.size
    covariance = (second - indices.size * np.outer(mean, mean)) / (indices.size - 1)
    covariance = (covariance + covariance.T) * 0.5
    return covariance, mean


def build_metric_bases(
    config: ExperimentConfig,
    *,
    operators: Mapping[int, Any],
    run_root: str | Path = ".",
) -> dict[str, Any]:
    """Build centered-J PCA and regularized activation-whitening transforms.

    ``operators`` must be identity-checked :class:`TokenFrameOperator` values
    backed by the current exact lens.  Their rows are centered only while
    forming the PCA Gram; pursuit later still receives the uncentered token
    dictionary through :class:`LinearTransformedDictionary`.
    """

    from jlens_workspace.matrix import decompose_gram, streaming_gram

    diagnostic = config.k_diagnostic
    if diagnostic is None or diagnostic.stage != "transformed_metric":
        raise KDiagnosticExperimentError(
            "build-bases requires the transformed_metric configuration"
        )
    initialize_artifacts(config, run_root=run_root)
    if set(operators) != set(diagnostic.analysis_layers):
        raise KDiagnosticExperimentError("operators must exactly cover transformed layers")
    root = artifact_root(config, run_root=run_root)
    shared = _resolve(Path(run_root), diagnostic.shared_artifact_root)
    rows_path = shared / "activations" / "rows.jsonl"
    train_validation_indices: list[int] = []
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") in {"train", "validation"}:
                train_validation_indices.append(int(row["row"]))
    indices = np.asarray(train_validation_indices, dtype=np.int64)
    if indices.size != np.unique(indices).size:
        raise KDiagnosticExperimentError("train+validation activation rows are not unique")

    layer_summaries: list[dict[str, Any]] = []
    active_metrics: set[str] = set()
    active_metrics_by_layer: dict[int, list[str]] = {}
    metric_dimensions_by_layer: dict[int, dict[str, int]] = {}
    for layer in diagnostic.analysis_layers:
        operator = operators[layer]
        if getattr(operator.metadata, "convention", None) != diagnostic.convention:
            raise KDiagnosticExperimentError(
                f"layer {layer} operator convention does not match diagnostic"
            )
        gram = streaming_gram(
            operator,
            centered=True,
            row_normalized=False,
            normalize_by_total_weight=False,
            accumulation_device=diagnostic.device,
            cpu_fallback=True,
        )
        spectrum = decompose_gram(
            gram,
            decomposition_device=diagnostic.device,
            cpu_fallback=True,
            rtol=1e-7,
        )
        eigenvalues = np.asarray(spectrum.eigenvalues.cpu(), dtype=np.float64)
        right_basis = np.asarray(spectrum.right_basis.cpu(), dtype=np.float64)
        k90 = _energy_rank(eigenvalues, 0.90)
        k95 = _energy_rank(eigenvalues, 0.95)
        k99 = _energy_rank(eigenvalues, 0.99)
        pca_dir = root / "bases" / "j_pca" / f"layer_{layer:02d}"
        pca_hashes = {
            "eigenvalues": _save_npy_exact(pca_dir / "eigenvalues.npy", eigenvalues),
            "right_basis": _save_npy_exact(pca_dir / "right_basis.npy", right_basis),
        }
        fixed_ranks = [rank for rank in diagnostic.pca_ranks if rank <= operator.d_model]
        rank_metrics: list[tuple[str, int]] = [(f"j_pca_r{rank}", rank) for rank in fixed_ranks]
        if k90 not in fixed_ranks:
            rank_metrics.append(("j_pca_k90", k90))
        for metric, rank in rank_metrics:
            transform = np.ascontiguousarray(right_basis[:, :rank].T)
            pca_hashes[metric] = _save_npy_exact(
                pca_dir / "transforms" / f"{metric}.npy", transform
            )
            active_metrics.add(metric)
        pca_summary = {
            "centered_for_metric_basis_only": True,
            "pursuit_atoms_remain_uncentered": True,
            "numerical_rank": int(spectrum.numerical_rank),
            "entropy_rank": float(spectrum.entropy_effective_rank),
            "participation_ratio": float(spectrum.participation_ratio),
            "stable_rank": float(spectrum.stable_rank),
            "k90": k90,
            "k95": k95,
            "k99": k99,
            "hashes": pca_hashes,
        }
        _write_json_exact(pca_dir / "metadata.json", pca_summary)

        covariance, activation_mean = _activation_covariance(
            shared / "activations" / f"layer_{layer:02d}.npy", indices
        )
        covariance_eigenvalues, covariance_basis = np.linalg.eigh(covariance)
        covariance_eigenvalues = covariance_eigenvalues[::-1].copy()
        covariance_basis = covariance_basis[:, ::-1].copy()
        scale = max(1.0, float(np.max(np.abs(covariance_eigenvalues))))
        tolerance = 100 * np.finfo(np.float64).eps * covariance.shape[0] * scale
        if float(covariance_eigenvalues.min()) < -tolerance:
            raise KDiagnosticExperimentError("activation covariance is not positive semidefinite")
        covariance_eigenvalues = np.maximum(covariance_eigenvalues, 0.0)
        positive = covariance_eigenvalues[covariance_eigenvalues > tolerance]
        if positive.size == 0:
            raise KDiagnosticExperimentError("activation covariance has no positive spectrum")
        covariance_dir = root / "bases" / "activation_covariance" / f"layer_{layer:02d}"
        whitening_hashes = {
            "eigenvalues": _save_npy_exact(
                covariance_dir / "eigenvalues.npy", covariance_eigenvalues
            ),
            "mean": _save_npy_exact(covariance_dir / "mean.npy", activation_mean),
        }
        whitening: dict[str, Any] = {}
        median_positive = float(np.median(positive))
        for floor in diagnostic.whitening_floors:
            tau = float(floor * median_positive)
            transform = ((covariance_eigenvalues + tau) ** -0.5)[:, None] * covariance_basis.T
            metric = f"activation_whitened_{floor:.2f}"
            active_metrics.add(metric)
            digest = _save_npy_exact(covariance_dir / "transforms" / f"{metric}.npy", transform)
            whitening_hashes[metric] = digest
            whitening[metric] = {
                "c": float(floor),
                "tau": tau,
                "condition_number_regularized": float(
                    (covariance_eigenvalues[0] + tau) / (covariance_eigenvalues[-1] + tau)
                ),
                "transform_sha256": digest,
            }
        active_metrics_by_layer[layer] = sorted(
            [metric for metric, _rank in rank_metrics]
            + [f"activation_whitened_{floor:.2f}" for floor in diagnostic.whitening_floors]
        )
        metric_dimensions_by_layer[layer] = {metric: int(rank) for metric, rank in rank_metrics} | {
            f"activation_whitened_{floor:.2f}": int(operator.d_model)
            for floor in diagnostic.whitening_floors
        }
        whitening_summary = {
            "sample_count": int(indices.size),
            "sample_split": "unique train+validation source activations",
            "concept_labels_accessed": False,
            "median_positive_eigenvalue": median_positive,
            "condition_number_positive_spectrum": float(positive.max() / positive.min()),
            "hashes": whitening_hashes,
            "regularized_whitening": whitening,
        }
        _write_json_exact(covariance_dir / "metadata.json", whitening_summary)
        layer_summaries.append(
            {"layer": layer, "j_pca": pca_summary, "activation": whitening_summary}
        )

    configured = set(diagnostic.metrics)
    if not active_metrics.issubset(configured) or not (configured - {"j_pca_k90"}).issubset(
        active_metrics
    ):
        raise KDiagnosticExperimentError(
            "computed basis metrics do not match the registered transformed grid"
        )
    payload = {
        "schema_version": 1,
        "complete": True,
        "layers": diagnostic.analysis_layers,
        "active_metrics": sorted(active_metrics),
        "active_metrics_by_layer": {
            str(layer): metrics for layer, metrics in sorted(active_metrics_by_layer.items())
        },
        "metric_dimensions_by_layer": {
            str(layer): dimensions
            for layer, dimensions in sorted(metric_dimensions_by_layer.items())
        },
        "duplicate_pca_ranks_removed": "j_pca_k90" not in active_metrics,
        "layer_summaries": layer_summaries,
    }
    atomic_write_json(root / "bases" / "index.json", payload)
    return payload


def load_metric_transform(
    *, artifact_directory: str | Path, layer: int, metric: str
) -> np.ndarray | None:
    """Load one registered metric transform, returning ``None`` for raw space."""

    root = Path(artifact_directory)
    if metric == "raw_euclidean":
        return None
    if metric.startswith("j_pca_"):
        path = root / "bases" / "j_pca" / f"layer_{layer:02d}" / "transforms" / f"{metric}.npy"
    elif metric.startswith("activation_whitened_"):
        path = (
            root
            / "bases"
            / "activation_covariance"
            / f"layer_{layer:02d}"
            / "transforms"
            / f"{metric}.npy"
        )
    else:
        raise KDiagnosticExperimentError(f"unregistered metric: {metric}")
    transform = np.load(path, allow_pickle=False)
    if transform.ndim != 2 or not np.isfinite(transform).all():
        raise KDiagnosticExperimentError(f"invalid metric transform: {path}")
    return np.asarray(transform, dtype=np.float64)
