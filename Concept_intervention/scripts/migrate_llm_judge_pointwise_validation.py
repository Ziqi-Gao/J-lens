#!/usr/bin/env python3
"""Import only message-equivalent pointwise validation judgments across judge rubrics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

DATA_ROOT = Path("/data/del6500/J-lens")
RUNTIME_ROOT = Path("/scr/del6500/J-lens")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {path}")
    return rows


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _require_tree(path: Path, allowed: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(f"{label} must remain under {allowed}: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a directory: {resolved}")
    return resolved


def _load_prompt_module(bundle: Path, name: str) -> ModuleType:
    path = (
        bundle
        / "src"
        / "jlens_workspace"
        / "concept_intervention"
        / "judge"
        / "prompts.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load prompt module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _schema_without_name(module: ModuleType) -> dict[str, Any]:
    schema = json.loads(json.dumps(module.response_schema("pointwise")))
    del schema["json_schema"]["name"]
    return schema


def _public_payload(task: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in task.items()
        if key not in {"task_id", "rubric_version"}
    }


def _index_source_results(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == "index.json":
            continue
        payload = _read_json(path)
        task_id = str(payload["task_id"])
        if task_id in result:
            raise ValueError(f"duplicate source response task_id: {task_id}")
        result[task_id] = path
    return result


def migrate(
    *,
    source_root: Path,
    target_root: Path,
    source_bundle: Path,
    target_bundle: Path,
    check_only: bool = False,
) -> dict[str, Any]:
    source_root = _require_tree(source_root, DATA_ROOT, "source_root")
    target_root = _require_tree(target_root, DATA_ROOT, "target_root")
    source_bundle = _require_tree(source_bundle, RUNTIME_ROOT, "source_bundle")
    target_bundle = _require_tree(target_bundle, RUNTIME_ROOT, "target_bundle")
    if source_root == target_root:
        raise ValueError("source_root and target_root must differ")

    source_manifest_path = source_root / "manifest.json"
    target_manifest_path = target_root / "manifest.json"
    source_manifest = _read_json(source_manifest_path)
    target_manifest = _read_json(target_manifest_path)
    if source_manifest["protocol_version"] != target_manifest["protocol_version"]:
        raise ValueError("source and target protocol versions differ")
    if source_manifest["rubric_version"] == target_manifest["rubric_version"]:
        raise ValueError("migration requires distinct source and target rubric versions")

    source_tasks = _read_jsonl(source_root / "tasks" / "pointwise_validation.jsonl")
    target_tasks = _read_jsonl(target_root / "tasks" / "pointwise_validation.jsonl")
    source_private = _read_jsonl(
        source_root / "private" / "pointwise_validation_map.jsonl"
    )
    target_private = _read_jsonl(
        target_root / "private" / "pointwise_validation_map.jsonl"
    )
    source_task_by_id = {str(row["task_id"]): row for row in source_tasks}
    target_task_by_id = {str(row["task_id"]): row for row in target_tasks}
    source_by_generation = {
        str(row["generation_id"]): source_task_by_id[str(row["task_id"])]
        for row in source_private
    }
    target_by_generation = {
        str(row["generation_id"]): target_task_by_id[str(row["task_id"])]
        for row in target_private
    }
    if len(source_by_generation) != 672 or set(source_by_generation) != set(
        target_by_generation
    ):
        raise ValueError("pointwise validation generation sets are not identical")

    source_prompts = _load_prompt_module(source_bundle, "jlens_prompts_source_v8")
    target_prompts = _load_prompt_module(target_bundle, "jlens_prompts_target_v9")
    if _schema_without_name(source_prompts) != _schema_without_name(target_prompts):
        raise ValueError("pointwise JSON schemas differ beyond their registered name")

    message_hashes: list[str] = []
    target_by_source_id: dict[str, dict[str, Any]] = {}
    for generation_id in sorted(source_by_generation):
        source_task = source_by_generation[generation_id]
        target_task = target_by_generation[generation_id]
        if _public_payload(source_task) != _public_payload(target_task):
            raise ValueError(f"public pointwise payload changed: {generation_id}")
        source_messages = source_prompts.render_messages(source_task)
        target_messages = target_prompts.render_messages(target_task)
        if source_messages != target_messages:
            raise ValueError(f"pointwise messages changed: {generation_id}")
        message_hashes.append(_canonical_hash(source_messages))
        target_by_source_id[str(source_task["task_id"])] = target_task

    response_sets = sorted(
        directory.parent.name
        for directory in (source_root / "responses").glob(
            "*/pointwise_validation"
        )
        if (directory / "index.json").is_file()
    )
    if len(response_sets) != 2:
        raise ValueError(f"expected two source pointwise response sets: {response_sets}")

    imported_cost = 0.0
    imported: dict[str, int] = {}
    for model_directory in response_sets:
        source_directory = (
            source_root / "responses" / model_directory / "pointwise_validation"
        )
        source_index = _read_json(source_directory / "index.json")
        if (
            source_index.get("complete") is not True
            or source_index.get("completed_tasks") != 672
            or source_index.get("failures") != []
        ):
            raise ValueError(f"incomplete source result set: {source_directory}")
        source_results = _index_source_results(source_directory)
        if set(source_results) != set(target_by_source_id):
            raise ValueError(f"source result IDs differ: {source_directory}")

        target_directory = (
            target_root / "responses" / model_directory / "pointwise_validation"
        )
        if not check_only:
            target_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        count = 0
        for source_task_id, source_path in sorted(source_results.items()):
            source_payload = _read_json(source_path)
            source_task = source_task_by_id[source_task_id]
            if source_payload.get("task_sha256") != _canonical_hash(source_task):
                raise ValueError(f"source task hash mismatch: {source_path}")
            if source_payload.get("rubric_sha256") != source_manifest["rubric_sha256"]:
                raise ValueError(f"source rubric hash mismatch: {source_path}")
            target_task = target_by_source_id[source_task_id]
            target_task_id = str(target_task["task_id"])
            migrated = dict(source_payload)
            migrated["task_id"] = target_task_id
            migrated["task_sha256"] = _canonical_hash(target_task)
            migrated["migration"] = {
                "schema_version": 1,
                "source_root": str(source_root),
                "source_task_id": source_task_id,
                "source_response_sha256": _sha256_file(source_path),
                "source_rubric_version": source_manifest["rubric_version"],
                "target_task_rubric_version": target_manifest["rubric_version"],
                "rendered_messages_byte_identical": True,
                "json_schema_identical_except_name": True,
                "pairwise_judgment_imported": False,
            }
            target_path = target_directory / f"{target_task_id}.json"
            if target_path.exists():
                if _read_json(target_path) != migrated:
                    raise ValueError(f"target response differs: {target_path}")
            elif not check_only:
                _atomic_write_json(target_path, migrated)
            cost = source_payload.get("usage", {}).get("cost")
            if isinstance(cost, bool) or not isinstance(cost, (int, float)):
                raise ValueError(f"source usage.cost is missing: {source_path}")
            if not math.isfinite(float(cost)) or float(cost) < 0:
                raise ValueError(f"source usage.cost is invalid: {source_path}")
            imported_cost += float(cost)
            count += 1
        imported[model_directory] = count

    result = {
        "schema_version": 1,
        "migration_type": "message_equivalent_pointwise_validation",
        "source_root": str(source_root),
        "target_root": str(target_root),
        "source_bundle": str(source_bundle),
        "target_bundle": str(target_bundle),
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "target_manifest_sha256": _sha256_file(target_manifest_path),
        "source_rubric_version": source_manifest["rubric_version"],
        "target_rubric_version": target_manifest["rubric_version"],
        "pointwise_messages_byte_identical": True,
        "pointwise_schema_identical_except_registered_name": True,
        "message_set_sha256": _canonical_hash(message_hashes),
        "generation_count": len(source_by_generation),
        "imported_response_counts": imported,
        "imported_provider_cost_usd": imported_cost,
        "pairwise_judgments_imported": False,
    }
    manifest_path = target_root / "migration" / "v8_pointwise_validation_import.json"
    if manifest_path.exists() and _read_json(manifest_path) != result:
        raise ValueError(f"migration manifest differs: {manifest_path}")
    if not check_only:
        _atomic_write_json(manifest_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--target-bundle", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = migrate(
        source_root=args.source_root,
        target_root=args.target_root,
        source_bundle=args.source_bundle,
        target_bundle=args.target_bundle,
        check_only=args.check_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
