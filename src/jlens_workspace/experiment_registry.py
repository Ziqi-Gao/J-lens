"""Strict metadata registry for versioned Concept Intervention experiments.

The registry separates scientific design versions from protocol revisions,
operational attempts, and derived artifact revisions.  It intentionally does
not mutate or relocate any scientific artifact.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_EXPERIMENT_ID = re.compile(
    r"[a-z0-9][a-z0-9-]*(?:/[a-z0-9][a-z0-9-]*)*/design-v[1-9][0-9]*(?:-[a-z0-9-]+)?"
)
_DESIGN_VERSION = re.compile(r"design-v[1-9][0-9]*(?:-[a-z0-9-]+)?")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_REVISION_ID = re.compile(r"revision-r[1-9][0-9]*")
_KINDS = {"diagnostic", "evaluation", "scientific_experiment"}
_EXECUTION_STATUSES = {"blocked", "complete", "historical", "registered", "superseded"}
_INTERPRETATION_STATUSES = {
    "descriptive_only",
    "invalid",
    "not_applicable",
    "not_evaluated",
    "pending_review",
    "valid",
}
_MANIFEST_KEYS = {
    "schema_version",
    "id",
    "family",
    "design_version",
    "kind",
    "title",
    "direction",
    "execution_status",
    "interpretation_status",
    "canonical",
    "supersedes",
    "depends_on",
    "repository",
    "artifacts",
    "attempts",
    "notes",
}
_OPTIONAL_MANIFEST_KEYS = {"derivations"}


@dataclass(frozen=True)
class ExperimentRecord:
    """One validated registry entry and its source manifest."""

    experiment_id: str
    manifest_path: Path
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class ExperimentRegistry:
    """Validated registry plus the repository root it describes."""

    path: Path
    repository_root: Path
    taxonomy: Mapping[str, str]
    records: tuple[ExperimentRecord, ...]

    def by_id(self, experiment_id: str) -> ExperimentRecord:
        for record in self.records:
            if record.experiment_id == experiment_id:
                return record
        raise ValueError(f"unknown experiment id: {experiment_id}")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _string_list(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a list of strings")
    result = tuple(value)
    if not all(isinstance(item, str) and item for item in result):
        raise ValueError(f"{label} must contain only non-empty strings")
    return result


def _inside(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} escapes repository root: {path}")
    return resolved


def _validate_repository_paths(
    repository: Mapping[str, Any],
    *,
    repository_root: Path,
    label: str,
) -> None:
    expected = {
        "branch",
        "commit",
        "paths_available",
        "protocol_paths",
        "config_paths",
        "launcher_paths",
    }
    unknown = set(repository) - expected
    missing = expected - set(repository)
    if unknown or missing:
        raise ValueError(f"{label}.repository keys invalid; missing={sorted(missing)}, unknown={sorted(unknown)}")
    branch = repository["branch"]
    commit = repository["commit"]
    if not isinstance(branch, str) or not branch:
        raise ValueError(f"{label}.repository.branch must be a non-empty string")
    if commit is not None and (not isinstance(commit, str) or not _COMMIT.fullmatch(commit)):
        raise ValueError(f"{label}.repository.commit must be null or a 40-character Git SHA")
    if not isinstance(repository["paths_available"], bool):
        raise ValueError(f"{label}.repository.paths_available must be boolean")
    for field in ("protocol_paths", "config_paths", "launcher_paths"):
        paths = _string_list(repository[field], label=f"{label}.repository.{field}")
        for raw_path in paths:
            candidate = Path(raw_path)
            if candidate.is_absolute():
                raise ValueError(f"{label}.repository.{field} must be repository-relative")
            resolved = _inside(repository_root / candidate, repository_root, label=field)
            if repository["paths_available"] and not resolved.is_file():
                raise ValueError(f"registered repository path is missing: {raw_path}")


def _validate_artifacts(artifacts: Mapping[str, Any], *, label: str, check: bool) -> None:
    expected = {"immutable", "roots", "completion_markers", "runtime_roots"}
    unknown = set(artifacts) - expected
    missing = expected - set(artifacts)
    if unknown or missing:
        raise ValueError(f"{label}.artifacts keys invalid; missing={sorted(missing)}, unknown={sorted(unknown)}")
    if artifacts["immutable"] is not True:
        raise ValueError(f"{label}.artifacts.immutable must be true")
    owned_roots = {
        "roots": Path("/data/del6500/J-lens").resolve(strict=False),
        "completion_markers": Path("/data/del6500/J-lens").resolve(strict=False),
        "runtime_roots": Path("/scr/del6500/J-lens").resolve(strict=False),
    }
    for field in ("roots", "completion_markers", "runtime_roots"):
        paths = _string_list(artifacts[field], label=f"{label}.artifacts.{field}")
        for raw_path in paths:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                raise ValueError(f"{label}.artifacts.{field} must contain absolute paths")
            if not candidate.resolve(strict=False).is_relative_to(owned_roots[field]):
                raise ValueError(
                    f"{label}.artifacts.{field} escapes the J-lens owned tree"
                )
            if check and not candidate.exists():
                raise ValueError(f"registered server path is missing: {raw_path}")
            if check and field == "completion_markers":
                try:
                    marker = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"completion marker is not readable JSON: {raw_path}"
                    ) from error
                if not isinstance(marker, Mapping) or marker.get("complete") is not True:
                    raise ValueError(
                        f"completion marker does not declare complete=true: {raw_path}"
                    )


def _validate_attempts(attempts: Mapping[str, Any], *, label: str) -> None:
    expected = {"directory_pattern", "legacy_labels"}
    unknown = set(attempts) - expected
    missing = expected - set(attempts)
    if unknown or missing:
        raise ValueError(f"{label}.attempts keys invalid; missing={sorted(missing)}, unknown={sorted(unknown)}")
    if attempts["directory_pattern"] != "attempt-NNN":
        raise ValueError(f"{label}.attempts.directory_pattern must be attempt-NNN")
    _string_list(attempts["legacy_labels"], label=f"{label}.attempts.legacy_labels")


def _validate_derivations(
    value: Any,
    *,
    expected_id: str,
    repository_root: Path,
    artifacts: Mapping[str, Any],
) -> None:
    """Validate typed revision manifests referenced by one frozen design."""

    paths = _string_list(value, label=f"{expected_id}.derivations")
    if not paths or len(paths) != len(set(paths)):
        raise ValueError(f"{expected_id}.derivations must be a non-empty unique list")
    source_roots = set(_string_list(artifacts["roots"], label=f"{expected_id}.artifacts.roots"))
    source_markers = set(
        _string_list(
            artifacts["completion_markers"],
            label=f"{expected_id}.artifacts.completion_markers",
        )
    )
    data_root = Path("/data/del6500/J-lens")
    seen_revision_ids: set[str] = set()
    for raw_path in paths:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            raise ValueError(f"{expected_id}.derivations must be repository-relative")
        revision_path = _inside(
            repository_root / candidate,
            repository_root,
            label=f"derivation for {expected_id}",
        )
        if not revision_path.is_file():
            raise ValueError(f"registered derivation manifest is missing: {raw_path}")

        import yaml

        revision = _mapping(
            yaml.safe_load(revision_path.read_text(encoding="utf-8")),
            label=f"derivation manifest {raw_path}",
        )
        expected_keys = {
            "schema_version",
            "id",
            "kind",
            "status",
            "source",
            "revision",
            "outputs",
            "scheduler",
        }
        if set(revision) != expected_keys:
            raise ValueError(f"derivation manifest keys are invalid: {raw_path}")
        revision_id = revision["id"]
        revision_name = (
            revision_id.rsplit("/", 1)[-1] if isinstance(revision_id, str) else ""
        )
        if (
            revision["schema_version"] != 1
            or not isinstance(revision_id, str)
            or _REVISION_ID.fullmatch(revision_name) is None
            or revision_id != f"{expected_id}/{revision_name}"
            or revision_id in seen_revision_ids
        ):
            raise ValueError(f"derivation identity is invalid: {raw_path}")
        seen_revision_ids.add(revision_id)
        if revision["kind"] not in {"derived_rescore", "derived_report"}:
            raise ValueError(f"derivation kind is invalid: {raw_path}")
        if revision["status"] not in {"registered_not_executed", "complete"}:
            raise ValueError(f"derivation status is invalid: {raw_path}")

        source = _mapping(revision["source"], label=f"{revision_id}.source")
        if set(source) != {
            "experiment_id",
            "artifact_root",
            "completion_markers",
            "immutable",
        }:
            raise ValueError(f"{revision_id}.source keys are invalid")
        completion_markers = _string_list(
            source["completion_markers"], label=f"{revision_id}.source.completion_markers"
        )
        if (
            source["experiment_id"] != expected_id
            or source["immutable"] is not True
            or source["artifact_root"] not in source_roots
            or not completion_markers
            or not set(completion_markers).issubset(source_markers)
        ):
            raise ValueError(f"{revision_id}.source does not match its registered design")

        revision_contract = _mapping(
            revision["revision"], label=f"{revision_id}.revision"
        )
        if set(revision_contract) != {
            "scientific_samples_regenerated",
            "estimand_changed",
            "description",
        }:
            raise ValueError(f"{revision_id}.revision keys are invalid")
        if (
            revision_contract["scientific_samples_regenerated"] is not False
            or revision_contract["estimand_changed"] is not False
            or not isinstance(revision_contract["description"], str)
            or not revision_contract["description"]
        ):
            raise ValueError(f"{revision_id} is not a compatible derived revision")

        outputs = _mapping(revision["outputs"], label=f"{revision_id}.outputs")
        if set(outputs) != {"root", "layout", "completion_markers", "write_policy"}:
            raise ValueError(f"{revision_id}.outputs keys are invalid")
        raw_output_root = outputs["root"]
        if not isinstance(raw_output_root, str) or not Path(raw_output_root).is_absolute():
            raise ValueError(f"{revision_id}.outputs.root must be absolute")
        output_root = Path(raw_output_root).resolve(strict=False)
        source_root = Path(str(source["artifact_root"])).resolve(strict=False)
        resolved_data_root = data_root.resolve(strict=False)
        if (
            not output_root.is_relative_to(resolved_data_root)
            or output_root.is_relative_to(source_root)
            or ("derivations", revision_name)
            not in tuple(zip(output_root.parts, output_root.parts[1:], strict=False))
        ):
            raise ValueError(f"{revision_id}.outputs.root is outside its revision namespace")
        layout = _mapping(outputs["layout"], label=f"{revision_id}.outputs.layout")
        if not layout or not all(
            isinstance(key, str)
            and key
            and isinstance(item, str)
            and item
            and not Path(item).is_absolute()
            and ".." not in Path(item).parts
            for key, item in layout.items()
        ):
            raise ValueError(f"{revision_id}.outputs.layout is invalid")
        output_markers = _string_list(
            outputs["completion_markers"],
            label=f"{revision_id}.outputs.completion_markers",
        )
        if not output_markers or any(
            Path(item).is_absolute() or ".." in Path(item).parts for item in output_markers
        ):
            raise ValueError(f"{revision_id}.outputs.completion_markers are invalid")
        if outputs["write_policy"] != "write_once":
            raise ValueError(f"{revision_id}.outputs.write_policy must be write_once")

        scheduler = _mapping(revision["scheduler"], label=f"{revision_id}.scheduler")
        if set(scheduler) != {"tasks", "activation"}:
            raise ValueError(f"{revision_id}.scheduler keys are invalid")
        if (
            not _string_list(scheduler["tasks"], label=f"{revision_id}.scheduler.tasks")
            or scheduler["activation"] != "separate_explicit_approval_required"
        ):
            raise ValueError(f"{revision_id}.scheduler activation is invalid")


def _validate_manifest(
    payload: Mapping[str, Any],
    *,
    expected_id: str,
    repository_root: Path,
    check_artifacts: bool,
) -> None:
    unknown = set(payload) - (_MANIFEST_KEYS | _OPTIONAL_MANIFEST_KEYS)
    missing = _MANIFEST_KEYS - set(payload)
    if unknown or missing:
        raise ValueError(
            f"{expected_id} manifest keys invalid; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if payload["schema_version"] != 1:
        raise ValueError(f"{expected_id}.schema_version must be 1")
    experiment_id = payload["id"]
    if experiment_id != expected_id or not _EXPERIMENT_ID.fullmatch(str(experiment_id)):
        raise ValueError(f"invalid or mismatched experiment id: {experiment_id!r}")
    design_version = payload["design_version"]
    if not isinstance(design_version, str) or not _DESIGN_VERSION.fullmatch(design_version):
        raise ValueError(f"invalid design_version for {expected_id}: {design_version!r}")
    if experiment_id.rsplit("/", 1)[-1] != design_version:
        raise ValueError(f"{expected_id} must end with its design_version")
    if payload["family"] != experiment_id.rsplit("/", 1)[0]:
        raise ValueError(f"{expected_id}.family does not match the experiment id")
    if payload["kind"] not in _KINDS:
        raise ValueError(f"invalid experiment kind for {expected_id}: {payload['kind']!r}")
    if payload["direction"] != "concept_intervention":
        raise ValueError(f"{expected_id}.direction must be concept_intervention")
    if payload["execution_status"] not in _EXECUTION_STATUSES:
        raise ValueError(f"invalid execution status for {expected_id}")
    if payload["interpretation_status"] not in _INTERPRETATION_STATUSES:
        raise ValueError(f"invalid interpretation status for {expected_id}")
    if not isinstance(payload["title"], str) or not payload["title"]:
        raise ValueError(f"{expected_id}.title must be a non-empty string")
    if not isinstance(payload["canonical"], bool):
        raise ValueError(f"{expected_id}.canonical must be boolean")
    if payload["execution_status"] == "complete" and not payload["artifacts"]["completion_markers"]:
        raise ValueError(
            f"{expected_id} is complete but has no registered completion marker"
        )
    _string_list(payload["supersedes"], label=f"{expected_id}.supersedes")
    _string_list(payload["depends_on"], label=f"{expected_id}.depends_on")
    _string_list(payload["notes"], label=f"{expected_id}.notes")
    _validate_repository_paths(
        _mapping(payload["repository"], label=f"{expected_id}.repository"),
        repository_root=repository_root,
        label=expected_id,
    )
    _validate_artifacts(
        _mapping(payload["artifacts"], label=f"{expected_id}.artifacts"),
        label=expected_id,
        check=check_artifacts,
    )
    _validate_attempts(
        _mapping(payload["attempts"], label=f"{expected_id}.attempts"),
        label=expected_id,
    )
    if "derivations" in payload:
        _validate_derivations(
            payload["derivations"],
            expected_id=expected_id,
            repository_root=repository_root,
            artifacts=_mapping(payload["artifacts"], label=f"{expected_id}.artifacts"),
        )


def load_experiment_registry(
    path: str | Path = "Concept_intervention/experiments/registry.yaml",
    *,
    check_artifacts: bool = False,
) -> ExperimentRegistry:
    """Load and strictly validate the registry and all referenced manifests."""

    import yaml

    registry_path = Path(path).resolve()
    if not registry_path.is_file():
        raise ValueError(f"experiment registry is missing: {registry_path}")
    repository_root = registry_path.parent.parent.parent.resolve()
    payload = _mapping(yaml.safe_load(registry_path.read_text(encoding="utf-8")), label="registry")
    expected_keys = {"schema_version", "project", "taxonomy", "experiments"}
    unknown = set(payload) - expected_keys
    missing = expected_keys - set(payload)
    if unknown or missing:
        raise ValueError(f"registry keys invalid; missing={sorted(missing)}, unknown={sorted(unknown)}")
    if payload["schema_version"] != 1 or payload["project"] != "J-lens":
        raise ValueError("registry must declare schema_version=1 and project=J-lens")
    taxonomy = _mapping(payload["taxonomy"], label="registry.taxonomy")
    expected_taxonomy = {"design", "protocol", "attempt", "revision"}
    if set(taxonomy) != expected_taxonomy or not all(
        isinstance(value, str) and value for value in taxonomy.values()
    ):
        raise ValueError("registry.taxonomy must define design, protocol, attempt, and revision")
    entries = payload["experiments"]
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError("registry.experiments must be a list")

    records: list[ExperimentRecord] = []
    seen: set[str] = set()
    for position, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, label=f"registry.experiments[{position}]")
        if set(entry) != {"id", "manifest"}:
            raise ValueError(f"registry.experiments[{position}] must contain only id and manifest")
        experiment_id = entry["id"]
        manifest_value = entry["manifest"]
        if not isinstance(experiment_id, str) or experiment_id in seen:
            raise ValueError(f"duplicate or invalid experiment id: {experiment_id!r}")
        if not isinstance(manifest_value, str) or Path(manifest_value).is_absolute():
            raise ValueError(f"manifest path for {experiment_id} must be repository-relative")
        manifest_path = _inside(
            repository_root / manifest_value,
            repository_root,
            label=f"manifest for {experiment_id}",
        )
        if not manifest_path.is_file():
            raise ValueError(f"experiment manifest is missing: {manifest_value}")
        manifest = _mapping(
            yaml.safe_load(manifest_path.read_text(encoding="utf-8")),
            label=f"manifest for {experiment_id}",
        )
        _validate_manifest(
            manifest,
            expected_id=experiment_id,
            repository_root=repository_root,
            check_artifacts=check_artifacts,
        )
        records.append(ExperimentRecord(experiment_id, manifest_path, manifest))
        seen.add(experiment_id)

    for record in records:
        for relation in ("depends_on", "supersedes"):
            unknown_ids = set(record.payload[relation]) - seen
            if unknown_ids:
                raise ValueError(
                    f"{record.experiment_id}.{relation} references unknown ids: {sorted(unknown_ids)}"
                )
    return ExperimentRegistry(
        path=registry_path,
        repository_root=repository_root,
        taxonomy=taxonomy,
        records=tuple(records),
    )


def registry_summary(registry: ExperimentRegistry) -> dict[str, Any]:
    """Return the stable machine-readable summary used by the CLI."""

    return {
        "valid": True,
        "registry": str(registry.path),
        "experiment_count": len(registry.records),
        "experiments": [
            {
                "id": record.experiment_id,
                "title": record.payload["title"],
                "kind": record.payload["kind"],
                "execution_status": record.payload["execution_status"],
                "interpretation_status": record.payload["interpretation_status"],
                "canonical": record.payload["canonical"],
                "manifest": str(record.manifest_path.relative_to(registry.repository_root)),
            }
            for record in registry.records
        ],
    }
