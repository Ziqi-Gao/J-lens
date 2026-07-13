from __future__ import annotations

import pytest
from pydantic import ValidationError

from jlens_workspace.config import (
    AlignmentConfig,
    ExperimentConfig,
    MatrixConfig,
    ProbeConfig,
)


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
