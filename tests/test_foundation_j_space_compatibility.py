from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_foundation_compatibility_modules_export_canonical_objects() -> None:
    from jlens_workspace import activations as legacy_activations
    from jlens_workspace import artifacts as legacy_artifacts
    from jlens_workspace import config as legacy_config
    from jlens_workspace import jacobian as legacy_jacobian
    from jlens_workspace import modeling as legacy_modeling
    from jlens_workspace.foundation import activations, artifacts, config, jacobian, modeling

    assert legacy_activations.capture_residual_activations is (
        activations.capture_residual_activations
    )
    assert legacy_artifacts.RunManifest is artifacts.RunManifest
    assert legacy_config.ExperimentConfig is config.ExperimentConfig
    assert legacy_modeling.ModelBundle is modeling.ModelBundle
    assert legacy_jacobian.ManagedJacobianLens is jacobian.ManagedJacobianLens
    assert artifacts.RunManifest.__module__ == "jlens_workspace.foundation.artifacts"
    assert config.ExperimentConfig.__module__ == "jlens_workspace.foundation.config"


def test_j_space_compatibility_modules_export_canonical_objects() -> None:
    from jlens_workspace import matrix as legacy_matrix
    from jlens_workspace.j_space import TokenFrameOperator
    from jlens_workspace.j_space.workflow import (
        MatrixWorkflowOptions,
        run_matrix_layers,
    )
    from jlens_workspace.matrix import geometry as legacy_subspace
    from jlens_workspace.matrix import operator as legacy_operator
    from jlens_workspace.matrix import spectrum as legacy_spectrum
    from jlens_workspace.workflows import matrix as legacy_workflow

    assert legacy_matrix.TokenFrameOperator is TokenFrameOperator
    assert legacy_operator.TokenFrameOperator is TokenFrameOperator
    assert legacy_subspace.basis_coverage is legacy_matrix.basis_coverage
    assert legacy_spectrum.streaming_gram is legacy_matrix.streaming_gram
    assert legacy_workflow.MatrixWorkflowOptions is MatrixWorkflowOptions
    assert legacy_workflow.run_matrix_layers is run_matrix_layers
    assert TokenFrameOperator.__module__ == "jlens_workspace.j_space.operator"
    assert run_matrix_layers.__module__ == "jlens_workspace.j_space.workflow"


def test_canonical_j_space_operator_preserves_streamed_coordinate_contract() -> None:
    torch = pytest.importorskip("torch")

    from jlens_workspace.foundation.jacobian import build_effective_unembedding
    from jlens_workspace.j_space import TokenFrameOperator, streaming_gram

    unembedding = torch.tensor(
        [[1.0, 2.0], [-3.0, 4.0], [5.0, -6.0]], dtype=torch.float64
    )
    jacobian = torch.tensor(
        [[2.0, 0.5], [-1.0, 3.0]], dtype=torch.float64
    )
    operator = TokenFrameOperator(
        jacobian,
        build_effective_unembedding(unembedding, convention="raw"),
        block_size=1,
        compute_device="cpu",
        compute_dtype=torch.float64,
    )

    explicit = unembedding @ jacobian
    streamed = torch.cat([block.rows for block in operator.iter_rows()])
    torch.testing.assert_close(streamed, explicit, rtol=0.0, atol=0.0)
    result = streaming_gram(operator)
    torch.testing.assert_close(
        result.gram,
        explicit.T @ explicit,
        rtol=1e-14,
        atol=1e-14,
    )
    assert result.gram.dtype is torch.float64


def test_architecture_packages_import_without_optional_heavy_modules() -> None:
    source_root = Path(__file__).parents[1] / "src"
    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source_root)
        if not current_pythonpath
        else f"{source_root}{os.pathsep}{current_pythonpath}"
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    script = """
import sys
import jlens_workspace.foundation
import jlens_workspace.foundation.activations
import jlens_workspace.foundation.jacobian
import jlens_workspace.j_space
import jlens_workspace.j_space.workflow
import jlens_workspace.numerics
import jlens_workspace.concept_intervention.evaluation
import jlens_workspace.concept_intervention.geometry
import jlens_workspace.concept_intervention.probing
import jlens_workspace.concept_intervention.protocol
import jlens_workspace.concept_intervention.reporting
import jlens_workspace.concept_intervention.steering
import jlens_workspace.scheduler
heavy = {'torch', 'transformers', 'datasets', 'jlens'}.intersection(sys.modules)
raise SystemExit(1 if heavy else 0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
