"""Validation for the hash-chained three-method format-repair amendment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jlens_workspace.artifacts import sha256_file
from jlens_workspace.concept_intervention.judge.prompts import rubric_hash

FORMAT_AMENDMENT_ID = "three_method_format_enforcement_v1"
FORMAT_AMENDMENT_RELATIVE_PATH = Path("amendments") / f"{FORMAT_AMENDMENT_ID}.json"


class FormatAmendmentError(ValueError):
    """Raised when the approved protocol amendment is missing or inconsistent."""


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FormatAmendmentError(f"missing or unsafe {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise FormatAmendmentError(f"invalid {label}: {path}")
    return dict(payload)


def _implementation_seal() -> dict[str, Any]:
    implementation_root = Path(__file__).parent
    files = {path.name: sha256_file(path) for path in sorted(implementation_root.glob("*.py"))}
    return {
        "files": files,
        "combined_sha256": _canonical_hash(files),
    }


def _within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError):
        return False
    return True


def _verify_response_seals(
    root: Path,
    response_seals: Mapping[str, Any],
) -> None:
    for label, raw_record in response_seals.items():
        if not isinstance(raw_record, Mapping):
            raise FormatAmendmentError(f"invalid response seal: {label}")
        record = dict(raw_record)
        destination = Path(str(record.get("path", "")))
        if (
            not destination.is_dir()
            or destination.is_symlink()
            or not _within_root(destination, root)
        ):
            raise FormatAmendmentError(f"missing or unsafe response directory: {destination}")
        excluded = record.get("excluded_task_ids", [])
        if (
            not isinstance(excluded, list)
            or any(not isinstance(task_id, str) or not task_id for task_id in excluded)
            or len(set(excluded)) != len(excluded)
        ):
            raise FormatAmendmentError(f"invalid response exclusions: {label}")
        excluded_names = {f"{task_id}.json" for task_id in excluded}
        files = {
            path.name: sha256_file(path)
            for path in sorted(destination.glob("*.json"))
            if path.name != "index.json" and path.name not in excluded_names
        }
        if len(files) != record.get("sealed_responses") or _canonical_hash(files) != record.get(
            "response_file_hashes_sha256"
        ):
            raise FormatAmendmentError(f"pre-amendment response seal mismatch: {label}")


def validate_format_amendment(
    config_path: str | Path,
    *,
    root: str | Path,
    experiment_name: str,
    verify_response_seals: bool = True,
) -> dict[str, Any]:
    """Validate the parent registration, amended code, approval, and response seals."""

    config_path = Path(config_path)
    root = Path(root)
    manifest_path = root / "manifest.json"
    amendment_path = root / FORMAT_AMENDMENT_RELATIVE_PATH
    manifest = _read_mapping(manifest_path, label="parent manifest")
    amendment = _read_mapping(amendment_path, label="format amendment")
    parent = amendment.get("parent_registration")
    amended = amendment.get("amended_registration")
    if not isinstance(parent, Mapping) or not isinstance(amended, Mapping):
        raise FormatAmendmentError("amendment registration seals are missing")
    parent_expected = {
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest_path),
        "rubric_sha256": manifest.get("rubric_sha256"),
        "implementation_combined_sha256": manifest.get("implementation", {}).get("combined_sha256"),
    }
    if (
        amendment.get("schema_version") != 1
        or amendment.get("amendment_id") != FORMAT_AMENDMENT_ID
        or amendment.get("experiment_name") != experiment_name
        or any(parent.get(key) != value for key, value in parent_expected.items())
        or manifest.get("config_sha256") != parent_expected["config_sha256"]
    ):
        raise FormatAmendmentError("format amendment does not chain to the parent registration")
    current_implementation = _implementation_seal()
    if (
        amended.get("rubric_sha256") != rubric_hash()
        or amended.get("implementation") != current_implementation
    ):
        raise FormatAmendmentError("runtime rubric/implementation differs from the amendment")
    repair_task_ids = amendment.get("repair_task_ids")
    if (
        not isinstance(repair_task_ids, list)
        or len(repair_task_ids) != 2
        or len(set(repair_task_ids)) != 2
        or any(not isinstance(task_id, str) or len(task_id) != 64 for task_id in repair_task_ids)
    ):
        raise FormatAmendmentError("format amendment repair task allowlist is invalid")
    approval_path = Path(str(amendment.get("approval_path", "")))
    if not _within_root(approval_path, root):
        raise FormatAmendmentError("format amendment approval path is outside the study root")
    approval = _read_mapping(approval_path, label="format amendment approval")
    amendment_sha256 = sha256_file(amendment_path)
    approval_expected = {
        "approved": True,
        "experiment_name": experiment_name,
        "amendment_id": FORMAT_AMENDMENT_ID,
        "parent_manifest_sha256": parent_expected["manifest_sha256"],
        "amendment_sha256": amendment_sha256,
    }
    if any(approval.get(key) != value for key, value in approval_expected.items()):
        raise FormatAmendmentError("format amendment approval does not match the amendment")
    mode = approval_path.stat().st_mode & 0o777
    if mode != 0o600:
        raise FormatAmendmentError(
            f"format amendment approval must have mode 0600, observed {mode:04o}"
        )
    response_seals = amendment.get("pre_amendment_response_seals")
    if not isinstance(response_seals, Mapping):
        raise FormatAmendmentError("pre-amendment response seals are missing")
    if verify_response_seals:
        _verify_response_seals(root, response_seals)
    return {
        "amendment_id": FORMAT_AMENDMENT_ID,
        "path": str(amendment_path),
        "sha256": amendment_sha256,
        "parent_manifest_sha256": parent_expected["manifest_sha256"],
        "parent_rubric_sha256": parent_expected["rubric_sha256"],
        "amended_rubric_sha256": amended["rubric_sha256"],
        "amended_implementation_combined_sha256": current_implementation["combined_sha256"],
        "approval_path": str(approval_path),
        "approval_sha256": sha256_file(approval_path),
        "repair_task_ids": repair_task_ids,
    }
