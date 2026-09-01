from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from jlens_workspace import cli
from jlens_workspace.experiment_registry import load_experiment_registry
from jlens_workspace.scheduler.catalog import (
    KDIAG_REPORT_REVISION_R1,
    KDIAG_RUN_ROOT,
    THREE_METHOD_REVISION_R2,
    THREE_METHOD_RUN_ROOT,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPOSITORY_ROOT / "Concept_intervention/experiments/registry.yaml"


def _copy_registry_metadata(tmp_path: Path) -> tuple[Path, Path]:
    registry_copy = tmp_path / "repository"
    source_root = REPOSITORY_ROOT / "Concept_intervention/experiments"
    target_root = registry_copy / "Concept_intervention/experiments"
    target_root.mkdir(parents=True)
    shutil.copytree(source_root, target_root, dirs_exist_ok=True)
    registry_payload = yaml.safe_load(
        (target_root / "registry.yaml").read_text(encoding="utf-8")
    )
    for entry in registry_payload["experiments"]:
        manifest_path = registry_copy / entry["manifest"]
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        manifest["repository"]["paths_available"] = False
        manifest_path.write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )
    return registry_copy, target_root


def test_checked_in_experiment_registry_is_strict_and_complete() -> None:
    registry = load_experiment_registry(REGISTRY_PATH)
    ids = [record.experiment_id for record in registry.records]

    assert len(ids) == len(set(ids)) == 10
    assert "three-method-intervention/design-v1" in ids
    assert "k-diagnostic/design-v2" in ids
    assert "llm-judge/three-method/design-v2-pointwise" in ids

    legacy_judge = registry.by_id("llm-judge/j-raptor/design-v1")
    assert legacy_judge.payload["design_version"] == "design-v1"
    assert legacy_judge.payload["attempts"]["directory_pattern"] == "attempt-NNN"
    assert "llm_judge_pilot_v16" in legacy_judge.payload["attempts"]["legacy_labels"]

    three_method = registry.by_id("three-method-intervention/design-v1")
    kdiag = registry.by_id("k-diagnostic/design-v2")
    assert three_method.payload["derivations"] == [
        "Concept_intervention/experiments/three-method-intervention/design-v1/derivations/revision-r2.yaml"
    ]
    assert kdiag.payload["derivations"] == [
        "Concept_intervention/experiments/k-diagnostic/design-v2/derivations/revision-r1.yaml"
    ]
    three_revision = yaml.safe_load(
        (REPOSITORY_ROOT / three_method.payload["derivations"][0]).read_text(
            encoding="utf-8"
        )
    )
    kdiag_revision = yaml.safe_load(
        (REPOSITORY_ROOT / kdiag.payload["derivations"][0]).read_text(encoding="utf-8")
    )
    assert Path(three_revision["outputs"]["root"]) == (
        THREE_METHOD_RUN_ROOT / THREE_METHOD_REVISION_R2
    )
    assert Path(kdiag_revision["outputs"]["root"]) == (
        KDIAG_RUN_ROOT / KDIAG_REPORT_REVISION_R1
    )


def test_registry_rejects_a_malformed_typed_derivation(
    tmp_path: Path,
) -> None:
    _registry_copy, target_root = _copy_registry_metadata(tmp_path)
    revision_path = target_root / "k-diagnostic/design-v2/derivations/revision-r1.yaml"
    revision = yaml.safe_load(revision_path.read_text(encoding="utf-8"))
    revision["revision"]["estimand_changed"] = True
    revision_path.write_text(yaml.safe_dump(revision, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="compatible derived revision"):
        load_experiment_registry(target_root / "registry.yaml")


def test_registry_resolves_derivation_roots_before_ownership_checks(tmp_path: Path) -> None:
    _registry_copy, target_root = _copy_registry_metadata(tmp_path)
    revision_path = target_root / "k-diagnostic/design-v2/derivations/revision-r1.yaml"
    revision = yaml.safe_load(revision_path.read_text(encoding="utf-8"))
    revision["outputs"]["root"] = (
        "/data/del6500/J-lens/../FedFisher/derivations/revision-r1/report"
    )
    revision_path.write_text(yaml.safe_dump(revision, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="outside its revision namespace"):
        load_experiment_registry(target_root / "registry.yaml")


def test_registry_rejects_duplicate_derivation_identities(tmp_path: Path) -> None:
    registry_copy, target_root = _copy_registry_metadata(tmp_path)
    original = target_root / "k-diagnostic/design-v2/derivations/revision-r1.yaml"
    duplicate = original.with_name("duplicate-revision-r1.yaml")
    shutil.copy2(original, duplicate)
    manifest_path = (
        registry_copy
        / "Concept_intervention/experiments/k-diagnostic/design-v2/experiment.yaml"
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["derivations"].append(
        "Concept_intervention/experiments/k-diagnostic/design-v2/derivations/duplicate-revision-r1.yaml"
    )
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="derivation identity is invalid"):
        load_experiment_registry(target_root / "registry.yaml")


def test_registry_rejects_artifacts_outside_owned_trees(tmp_path: Path) -> None:
    registry_copy, target_root = _copy_registry_metadata(tmp_path)
    manifest_path = (
        registry_copy
        / "Concept_intervention/experiments/three-method-intervention/design-v1/experiment.yaml"
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["roots"] = [
        "/data/del6500/J-lens/../FedFisher/foreign-artifact"
    ]
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes the J-lens owned tree"):
        load_experiment_registry(target_root / "registry.yaml")


def test_experiment_registry_cli_lists_and_shows_designs(capsys: object) -> None:
    assert (
        cli.main(
            [
                "experiments",
                "validate",
                "--registry",
                str(REGISTRY_PATH),
                "--json",
            ]
        )
        == 0
    )
    validate_payload = json.loads(capsys.readouterr().out)
    assert validate_payload["valid"] is True
    assert validate_payload["experiment_count"] == 10
    assert validate_payload["artifact_paths_checked"] is False

    assert (
        cli.main(
            [
                "experiments",
                "show",
                "three-method-intervention/design-v1",
                "--registry",
                str(REGISTRY_PATH),
                "--json",
            ]
        )
        == 0
    )
    show_payload = json.loads(capsys.readouterr().out)
    assert show_payload["execution_status"] == "complete"
    assert show_payload["interpretation_status"] == "valid"


def test_experiment_registry_cli_rejects_unknown_id(capsys: object) -> None:
    assert (
        cli.main(
            [
                "experiments",
                "show",
                "unknown/design-v1",
                "--registry",
                str(REGISTRY_PATH),
            ]
        )
        == 2
    )
    assert "unknown experiment id" in capsys.readouterr().err
