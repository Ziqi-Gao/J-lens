from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jlens_workspace.concept_intervention.iti.experiment import iti_experiment_grid
from jlens_workspace.config import (
    AlignmentConfig,
    ExperimentConfig,
    ITIConfig,
    MatrixConfig,
    ProbeConfig,
    load_experiment_config,
)


def _iti_config(**overrides: object) -> ITIConfig:
    values = {
        "attention_layers": [3, 7],
        "num_heads": 2,
        "head_dim": 4,
        "top_k_grid": [1, 4],
        "concept_ids": ["concept:a"],
        "source_residual_activations_dir": "residuals",
        "head_activations_dir": "heads",
        "directions_dir": "directions",
        "reference_j_intervention_dir": "j",
        "prompts_path": "prompts.json",
        "candidate_labels": {"concept:a": "alpha"},
    }
    values.update(overrides)
    return ITIConfig.model_validate(values)


def test_probe_rejects_invalid_penalty_strength() -> None:
    with pytest.raises(ValidationError):
        ProbeConfig(c_grid=[0.1, 0.0])


def test_matrix_accumulation_is_always_float64() -> None:
    with pytest.raises(ValidationError):
        MatrixConfig(accumulation_dtype="float32")


def test_alignment_control_seeds_are_unique() -> None:
    with pytest.raises(ValidationError, match="random_control_seeds"):
        AlignmentConfig(random_control_seeds=[7, 7])
    with pytest.raises(ValidationError, match="non-negative"):
        AlignmentConfig(random_control_seeds=[-1])


def test_iti_grid_and_prompt_splits_are_validation_safe() -> None:
    assert _iti_config().top_k_grid == [1, 4]
    with pytest.raises(ValidationError, match="top_k_grid"):
        _iti_config(top_k_grid=[5])
    with pytest.raises(ValidationError, match="disjoint"):
        _iti_config(validation_prompt_prefixes=["same"], test_prompt_prefixes=["same"])


def test_matrix_rank_sweep_contains_primary_tolerance() -> None:
    with pytest.raises(ValidationError, match="rank_relative_tolerance"):
        MatrixConfig(
            rank_relative_tolerance=1e-7,
            rank_relative_tolerances=[1e-5, 1e-6],
        )


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "schema_version": 1,
                "direction": "j_space",
                "experiment_name": "x",
                "output_dir": "out",
                "model": {"model_id": "tiny"},
                "typo": True,
            }
        )


def test_config_rejects_hybrid_research_directions() -> None:
    with pytest.raises(ValidationError, match="must not define concept fields"):
        ExperimentConfig.model_validate(
            {
                "schema_version": 1,
                "direction": "j_space",
                "experiment_name": "hybrid",
                "output_dir": "out",
                "model": {"model_id": "tiny"},
                "lens": {
                    "source": "local",
                    "path_or_repo": "lens.pt",
                    "layers": [0],
                },
                "dataset": {"source": "jsonl", "path": "data.jsonl"},
                "matrix": {"layers": [0]},
            }
        )


def test_three_method_configs_share_the_immutable_registered_protocol() -> None:
    root = Path(__file__).parents[1]
    names = (
        "j_component_intervention",
        "iti_intervention",
        "raptor_intervention",
    )
    configs = {
        name: load_experiment_config(
            root / "Concept_intervention/configs" / f"qwen35_4b_{name}.yaml"
        )
        for name in names
    }
    for name, config in configs.items():
        assert (
            config.experiment_name
            == "qwen35_4b_three_method_intervention_v1"
        )
        assert (
            "/qwen35_4b_three_method_intervention_v1/"
            in config.output_dir
        )
        assert config.output_dir.endswith(f"/{name}")

    reference = configs["j_component_intervention"]
    for config in configs.values():
        assert config.model == reference.model
        assert config.dataset == reference.dataset
        assert config.lens == reference.lens
        assert config.activations == reference.activations
        assert config.probe == reference.probe

    generations = (
        reference.j_component.generation,
        configs["iti_intervention"].iti.generation,
        configs["raptor_intervention"].raptor.generation,
    )
    assert generations[0] == generations[1] == generations[2]
    layer_paths = {
        reference.j_component.selected_layers_path,
        configs["iti_intervention"].iti.selected_layers_path,
        configs["raptor_intervention"].raptor.selected_layers_path,
    }
    assert len(layer_paths) == 1

    shared = load_experiment_config(
        root
        / "Concept_intervention/configs/qwen35_4b_shared_intervention_protocol.yaml"
    )
    assert (
        shared.experiment_name
        == "qwen35_4b_three_method_intervention_v1"
    )
    assert shared.output_dir.endswith(
        "/qwen35_4b_three_method_intervention_v1/"
        "shared_intervention_protocol"
    )
    assert shared.shared_layer_selection.candidate_layers == [3, 7, 11, 15, 19, 23, 27]
    assert shared.shared_layer_selection.selected_layer_count == 6
    assert len(shared.shared_layer_selection.c_grid) == 100
    assert len(iti_experiment_grid(configs["iti_intervention"].iti)) == 441
