from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = (
    ROOT
    / "src/jlens_workspace/concept_intervention/geometry/k_diagnostic/experiment.py"
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_kdiagnostic_uses_only_the_canonical_scheduler_manifest_contract() -> None:
    imports = _imports(EXPERIMENT)

    assert "jlens_workspace.server_scheduler_adapter" not in imports
    assert "jlens_workspace.scheduler.errors" in imports
    assert "jlens_workspace.scheduler.manifest" in imports
    assert "jlens_workspace.scheduler.runner" not in imports
    assert "jlens_workspace.scheduler.completion" not in imports


def test_kdiagnostic_import_keeps_scheduler_execution_and_gpu_stacks_lazy() -> None:
    environment = os.environ.copy()
    source_root = ROOT / "src"
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source_root)
        if not inherited
        else f"{source_root}{os.pathsep}{inherited}"
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    script = """
import sys
import jlens_workspace.concept_intervention.geometry.k_diagnostic.experiment
forbidden = {
    'jlens_workspace.server_scheduler_adapter',
    'jlens_workspace.scheduler.runner',
    'jlens_workspace.scheduler.completion',
    'torch',
    'transformers',
    'datasets',
    'jlens',
}
raise SystemExit(1 if forbidden.intersection(sys.modules) else 0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
