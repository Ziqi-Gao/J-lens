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
    root = Path(__file__).parents[1] / "src/jlens_workspace/workflows"
    j_imports = _imports(root / "concept_intervention.py")
    iti_imports = _imports(root / "iti.py")

    assert "jlens_workspace.workflows.iti" not in j_imports
    assert "jlens_workspace.iti" not in j_imports
    assert "jlens_workspace.workflows.concept_intervention" not in iti_imports
    assert "jlens_workspace.interventions" not in iti_imports
    assert "jlens_workspace.workflows.candidate_evaluation" in j_imports
    assert "jlens_workspace.workflows.candidate_evaluation" in iti_imports
