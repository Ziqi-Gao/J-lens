from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


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
    root = Path(__file__).parents[1] / "src/jlens_workspace/concept_intervention/steering"
    methods = ("j_component", "iti", "raptor")
    imports = {
        method: set().union(*(_imports(path) for path in (root / method).glob("*.py")))
        for method in methods
    }
    for method, modules in imports.items():
        siblings = set(methods) - {method}
        assert not any(
            module.startswith(f"jlens_workspace.concept_intervention.steering.{sibling}")
            for sibling in siblings
            for module in modules
        )

    j_imports = imports["j_component"]
    iti_imports = imports["iti"]
    raptor_imports = imports["raptor"]
    assert "jlens_workspace.concept_intervention.evaluation" in j_imports
    assert "jlens_workspace.concept_intervention.evaluation" in iti_imports
    assert "jlens_workspace.concept_intervention.evaluation" in raptor_imports


def test_probing_does_not_depend_on_steering_or_evaluation() -> None:
    root = Path(__file__).parents[1] / "src/jlens_workspace/concept_intervention/probing"
    imports = set().union(*(_imports(path) for path in root.glob("*.py")))
    forbidden = (
        "jlens_workspace.concept_intervention.steering",
        "jlens_workspace.concept_intervention.evaluation",
    )
    assert not any(module.startswith(forbidden) for module in imports)


def test_protocol_contains_no_probe_training_or_steering_dependencies() -> None:
    root = Path(__file__).parents[1] / "src/jlens_workspace/concept_intervention/protocol"
    imports = set().union(*(_imports(path) for path in root.glob("*.py")))
    forbidden = (
        "jlens_workspace.concept_intervention.probing",
        "jlens_workspace.concept_intervention.steering",
        "jlens_workspace.concept_intervention.evaluation",
        "jlens_workspace.foundation.modeling",
        "sklearn",
        "torch",
    )
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in forbidden
        for module in imports
    )


def test_evaluation_reexports_protocol_contracts_by_identity() -> None:
    evaluation = importlib.import_module(
        "jlens_workspace.concept_intervention.evaluation"
    )
    legacy_contracts = importlib.import_module(
        "jlens_workspace.concept_intervention.evaluation.contracts"
    )
    contracts = importlib.import_module(
        "jlens_workspace.concept_intervention.protocol.contracts"
    )
    assert legacy_contracts is contracts
    assert evaluation.CandidateEvaluationError is contracts.CandidateEvaluationError
    assert evaluation.PromptRecord is contracts.PromptRecord
    assert evaluation.atomic_write_jsonl is contracts.atomic_write_jsonl
    assert evaluation.batched is contracts.batched
    assert evaluation.candidate_token_ids is contracts.candidate_token_ids
    assert evaluation.load_prompt_bank is contracts.load_prompt_bank


def test_generation_execution_is_owned_by_steering() -> None:
    legacy = importlib.import_module(
        "jlens_workspace.concept_intervention.generation"
    )
    protocol = importlib.import_module(
        "jlens_workspace.concept_intervention.protocol.generation"
    )
    execution = importlib.import_module(
        "jlens_workspace.concept_intervention.steering.generation"
    )

    assert legacy is execution
    assert execution.GenerationSettings is protocol.GenerationSettings
    assert (
        execution.InterventionGenerationError
        is protocol.InterventionGenerationError
    )
    assert execution.build_generation_contract is protocol.build_generation_contract
    assert (
        execution.validate_generation_artifacts
        is protocol.validate_generation_artifacts
    )
    for name in (
        "generate_full_grid",
        "write_generation_artifacts",
        "build_method_index_provenance",
    ):
        function = getattr(execution, name)
        assert function.__module__ == execution.__name__
        assert not hasattr(protocol, name)
    assert not hasattr(protocol, "_generate_one")


def test_legacy_shared_protocol_composes_canonical_objects_by_identity() -> None:
    legacy = importlib.import_module("jlens_workspace.concept_intervention.shared_protocol")
    protocol = importlib.import_module("jlens_workspace.concept_intervention.protocol.shared")
    probing = importlib.import_module(
        "jlens_workspace.concept_intervention.probing.layer_selection"
    )
    assert legacy.SharedProtocolError is protocol.SharedProtocolError
    assert legacy.deterministic_balanced_indices is protocol.deterministic_balanced_indices
    assert legacy.load_balanced_indices is protocol.load_balanced_indices
    assert legacy.load_selected_layers is protocol.load_selected_layers
    assert legacy.validate_shared_identity is protocol.validate_shared_identity
    assert legacy._grouped_raptor_c_scores is probing._grouped_raptor_c_scores
    assert legacy.load_upstream_raptor_tuning is probing.load_upstream_raptor_tuning
    assert legacy.run_shared_layer_selection is probing.run_shared_layer_selection


def test_probing_workflow_reexports_neutral_activation_helpers_by_identity() -> None:
    workflow = importlib.import_module("jlens_workspace.concept_intervention.probing.workflow")
    artifact = importlib.import_module(
        "jlens_workspace.concept_intervention.data.activation_artifact"
    )
    assert workflow.ConceptWorkflowError is artifact.ConceptWorkflowError
    assert workflow._ActivationArtifact is artifact.ActivationArtifact
    assert workflow._load_activation_artifact is artifact.load_activation_artifact
    assert workflow._concept_labels is artifact.activation_concept_labels
    assert workflow._groups is artifact.activation_groups
    assert workflow._indices_by_split is artifact.activation_indices_by_split


def test_evaluation_does_not_import_steering_implementations() -> None:
    root = Path(__file__).parents[1] / "src/jlens_workspace/concept_intervention/evaluation"
    imports = set().union(*(_imports(path) for path in root.glob("*.py")))
    assert not any(
        module.startswith("jlens_workspace.concept_intervention.steering") for module in imports
    )


def test_reporting_is_artifact_only() -> None:
    root = Path(__file__).parents[1] / "src/jlens_workspace/concept_intervention/reporting"
    imports = set().union(*(_imports(path) for path in root.glob("*.py")))
    forbidden = (
        "torch",
        "transformers",
        "datasets",
        "jlens",
        "jlens_workspace.foundation.modeling",
        "jlens_workspace.concept_intervention.probing",
        "jlens_workspace.concept_intervention.steering",
    )
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in forbidden
        for module in imports
    )


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    (
        (
            "jlens_workspace.probes.logistic",
            "jlens_workspace.concept_intervention.probing.logistic",
        ),
        (
            "jlens_workspace.concepts.alignment",
            "jlens_workspace.concept_intervention.geometry.alignment.core",
        ),
        (
            "jlens_workspace.pursuit.solver",
            "jlens_workspace.concept_intervention.geometry.sparse_pursuit.solver",
        ),
        (
            "jlens_workspace.pursuit.dictionaries",
            "jlens_workspace.concept_intervention.geometry.sparse_pursuit.dictionaries",
        ),
        (
            "jlens_workspace.pursuit.occupancy",
            "jlens_workspace.concept_intervention.geometry.sparse_pursuit.occupancy",
        ),
        (
            "jlens_workspace.workflows.concept",
            "jlens_workspace.concept_intervention.probing.workflow",
        ),
        (
            "jlens_workspace.workflows.probe_replicates",
            "jlens_workspace.concept_intervention.probing.replicates",
        ),
        (
            "jlens_workspace.workflows.alignment",
            "jlens_workspace.concept_intervention.geometry.alignment.workflow",
        ),
        (
            "jlens_workspace.workflows.occupancy",
            "jlens_workspace.concept_intervention.geometry.sparse_pursuit.workflow",
        ),
        (
            "jlens_workspace.concept_intervention.generation",
            "jlens_workspace.concept_intervention.steering.generation",
        ),
        (
            "jlens_workspace.concept_intervention.candidate_rescore",
            "jlens_workspace.concept_intervention.evaluation.candidate_rescore",
        ),
        (
            "jlens_workspace.concept_intervention.comparison",
            "jlens_workspace.concept_intervention.evaluation.comparison",
        ),
        (
            "jlens_workspace.concept_intervention.j_component.intervention",
            "jlens_workspace.concept_intervention.steering.j_component.intervention",
        ),
        (
            "jlens_workspace.concept_intervention.j_component.multilayer",
            "jlens_workspace.concept_intervention.steering.j_component.multilayer",
        ),
        (
            "jlens_workspace.concept_intervention.j_component.workflow",
            "jlens_workspace.concept_intervention.steering.j_component.workflow",
        ),
        (
            "jlens_workspace.concept_intervention.iti.intervention",
            "jlens_workspace.concept_intervention.steering.iti.intervention",
        ),
        (
            "jlens_workspace.concept_intervention.iti.experiment",
            "jlens_workspace.concept_intervention.steering.iti.experiment",
        ),
        (
            "jlens_workspace.concept_intervention.iti.workflow",
            "jlens_workspace.concept_intervention.steering.iti.workflow",
        ),
        (
            "jlens_workspace.concept_intervention.raptor.intervention",
            "jlens_workspace.concept_intervention.steering.raptor.intervention",
        ),
        (
            "jlens_workspace.concept_intervention.raptor.workflow",
            "jlens_workspace.concept_intervention.steering.raptor.workflow",
        ),
        (
            "jlens_workspace.concept_intervention.k_diagnostic.experiment",
            "jlens_workspace.concept_intervention.geometry.k_diagnostic.experiment",
        ),
        (
            "jlens_workspace.concept_intervention.k_diagnostic.aggregation",
            "jlens_workspace.concept_intervention.geometry.k_diagnostic.aggregation",
        ),
        (
            "jlens_workspace.concept_intervention.k_diagnostic.nulls",
            "jlens_workspace.concept_intervention.geometry.k_diagnostic.nulls",
        ),
        (
            "jlens_workspace.concept_intervention.k_diagnostic.reporting",
            "jlens_workspace.concept_intervention.reporting.k_diagnostic",
        ),
        (
            "jlens_workspace.concept_intervention.geometry.k_diagnostic.reporting",
            "jlens_workspace.concept_intervention.reporting.k_diagnostic",
        ),
        (
            "jlens_workspace.concept_intervention.k_diagnostic.targets",
            "jlens_workspace.concept_intervention.geometry.k_diagnostic.targets",
        ),
        (
            "jlens_workspace.concept_intervention.k_diagnostic.theory",
            "jlens_workspace.concept_intervention.geometry.k_diagnostic.theory",
        ),
        (
            "jlens_workspace.concept_intervention.k_diagnostic.transformed_dictionary",
            "jlens_workspace.concept_intervention.geometry.k_diagnostic.transformed_dictionary",
        ),
    ),
)
def test_legacy_module_is_exact_alias(legacy: str, canonical: str) -> None:
    assert importlib.import_module(legacy) is importlib.import_module(canonical)


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
