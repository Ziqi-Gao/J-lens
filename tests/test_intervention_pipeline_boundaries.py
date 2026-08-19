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


def test_three_intervention_workflows_do_not_import_each_other() -> None:
    root = Path(__file__).parents[1] / "src/jlens_workspace/concept_intervention"
    methods = ("j_component", "iti", "raptor")
    imports = {
        method: set().union(
            *(_imports(path) for path in (root / method).glob("*.py"))
        )
        for method in methods
    }
    for method, modules in imports.items():
        siblings = set(methods) - {method}
        assert not any(
            module.startswith(f"jlens_workspace.concept_intervention.{sibling}")
            for sibling in siblings
            for module in modules
        )

    j_imports = imports["j_component"]
    iti_imports = imports["iti"]
    raptor_imports = imports["raptor"]
    assert "jlens_workspace.concept_intervention.evaluation" in j_imports
    assert "jlens_workspace.concept_intervention.evaluation" in iti_imports
    assert "jlens_workspace.concept_intervention.evaluation" in raptor_imports


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
