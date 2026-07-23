from __future__ import annotations

import ast
from pathlib import Path


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_j_component_and_iti_workflows_do_not_import_each_other() -> None:
    root = Path(__file__).parents[1] / "src/jlens_workspace/concept_intervention"
    j_imports = set().union(*(_imports(path) for path in (root / "j_component").glob("*.py")))
    iti_imports = set().union(*(_imports(path) for path in (root / "iti").glob("*.py")))

    assert not any(
        module.startswith("jlens_workspace.concept_intervention.iti")
        for module in j_imports
    )
    assert not any(
        module.startswith("jlens_workspace.concept_intervention.j_component")
        for module in iti_imports
    )
    assert "jlens_workspace.concept_intervention.evaluation" in j_imports
    assert "jlens_workspace.concept_intervention.evaluation" in iti_imports


def test_legacy_flat_intervention_modules_are_absent() -> None:
    root = Path(__file__).parents[1] / "src/jlens_workspace"
    legacy_paths = (
        root / "interventions.py",
        root / "iti.py",
        root / "workflows/candidate_evaluation.py",
        root / "workflows/concept_intervention.py",
        root / "workflows/iti.py",
    )
    assert not any(path.exists() for path in legacy_paths)
