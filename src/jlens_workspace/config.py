"""Strict, versioned configuration shared by command-line experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base class that rejects misspelled configuration keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(StrictModel):
    model_id: str
    revision: str = "main"
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    dtype: Literal["float32", "float16", "bfloat16", "auto"] = "bfloat16"
    device: str = "auto"
    trust_remote_code: bool = False
    force_bos: bool | None = None


class DatasetConfig(StrictModel):
    source: Literal["builtin", "jsonl", "axbench"]
    path: str
    dataset_id: str | None = None
    revision: str | None = None
    allowlist_path: str | None = None
    streaming: bool = True
    min_per_label_per_split: int = Field(default=1, ge=1)
    min_per_concept: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_axbench_fields(self) -> DatasetConfig:
        if self.source == "axbench" and not self.allowlist_path:
            raise ValueError("allowlist_path is required for AxBench")
        return self


class LensConfig(StrictModel):
    source: Literal["fit", "local", "huggingface"]
    path_or_repo: str | None = None
    filename: str | None = None
    revision: str | None = None
    layers: list[int] = Field(min_length=1)
    fit_prompts_path: str | None = None
    fit_output_path: str | None = None
    fit_checkpoint_path: str | None = None
    n_fit_prompts: int = Field(default=128, ge=1)
    target_layer: int | None = None
    dim_batch: int = Field(default=128, ge=1)
    max_seq_len: int = Field(default=128, ge=8)
    skip_first: int = Field(default=16, ge=0)
    compile_blocks: bool = False
    checkpoint_every: int | None = Field(default=25, ge=1)
    resume: bool = True
    storage_dtype: Literal["float32", "float16", "bfloat16"] = "float32"

    @model_validator(mode="after")
    def validate_source_fields(self) -> LensConfig:
        if self.source in {"local", "huggingface"} and not self.path_or_repo:
            raise ValueError("path_or_repo is required for a local or Hugging Face lens")
        if self.source == "huggingface" and not self.filename:
            raise ValueError("filename is required for a Hugging Face lens")
        if self.source == "fit" and not self.fit_prompts_path:
            raise ValueError("fit_prompts_path is required when source='fit'")
        if self.source == "fit" and self.target_layer is None:
            raise ValueError("target_layer is required when source='fit'")
        return self


class ActivationConfig(StrictModel):
    layers: list[int] = Field(min_length=1)
    batch_size: int = Field(default=8, ge=1)
    max_length: int = Field(default=512, ge=8)
    add_special_tokens: bool = True
    share_examples_by_group: bool = False
    require_complete_concept_matrix: bool = False

    @model_validator(mode="after")
    def validate_shared_matrix(self) -> ActivationConfig:
        if self.require_complete_concept_matrix and not self.share_examples_by_group:
            raise ValueError(
                "require_complete_concept_matrix requires share_examples_by_group=true"
            )
        return self


class ProbeConfig(StrictModel):
    penalty: Literal["l2"] = "l2"
    c_grid: list[float] = Field(
        default_factory=lambda: [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0]
    )
    cv_folds: int = Field(default=5, ge=2)
    scoring: Literal["roc_auc", "balanced_accuracy"] = "roc_auc"
    standardize: bool = True
    class_weight: Literal["balanced"] | None = "balanced"
    max_iter: int = Field(default=5000, ge=100)
    seed: int = 42

    @model_validator(mode="after")
    def validate_c_grid(self) -> ProbeConfig:
        if not self.c_grid or any(c <= 0 for c in self.c_grid):
            raise ValueError("c_grid must contain positive inverse-penalty strengths")
        return self


class AlignmentConfig(StrictModel):
    metric: Literal["cosine", "covariance"] = "cosine"
    convention: Literal["raw", "rmsnorm_weighted"] = "rmsnorm_weighted"
    top_k: int = Field(default=50, ge=1)
    vocabulary_chunk_size: int = Field(default=4096, ge=1)
    decompose: bool = False
    sparse_components: int = Field(default=16, ge=1)
    candidate_pool_size: int = Field(default=512, ge=1)
    require_nonnegative: bool = True


class InterventionConfig(StrictModel):
    kind: Literal["addition", "project_out"] = "addition"
    strengths: list[float] = Field(default_factory=lambda: [-2.0, -1.0, 0.0, 1.0, 2.0])
    position: Literal["last_prompt", "generated", "last_prompt_and_generated", "all"] = (
        "last_prompt_and_generated"
    )
    scale_by_residual_norm: bool = True
    max_new_tokens: int = Field(default=64, ge=1)
    do_sample: bool = False
    temperature: float = Field(default=1.0, gt=0)
    seed: int = 42


class MatrixConfig(StrictModel):
    layers: list[int] | None = None
    convention: Literal["raw", "rmsnorm_weighted"] = "rmsnorm_weighted"
    centered: bool = True
    row_normalized: bool = False
    normalize_by_total_weight: bool = False
    zero_row_policy: Literal["error", "skip"] = "error"
    vocabulary_chunk_size: int = Field(default=4096, ge=1)
    operator_compute_dtype: Literal["float32", "float64"] = "float32"
    accumulation_dtype: Literal["float64"] = "float64"
    energy_thresholds: list[float] = Field(default_factory=lambda: [0.9, 0.95, 0.99])
    rank_relative_tolerance: float = Field(default=1e-7, gt=0)
    device: str = "cpu"
    cpu_fallback: bool = True

    @model_validator(mode="after")
    def validate_thresholds(self) -> MatrixConfig:
        if any(not 0 < threshold <= 1 for threshold in self.energy_thresholds):
            raise ValueError("energy_thresholds must lie in (0, 1]")
        return self


class ExperimentConfig(StrictModel):
    schema_version: Literal[1] = 1
    direction: Literal["concept_intervention", "j_space"]
    experiment_name: str
    output_dir: str
    seed: int = 42
    model: ModelConfig
    dataset: DatasetConfig | None = None
    lens: LensConfig | None = None
    activations: ActivationConfig | None = None
    probe: ProbeConfig | None = None
    alignment: AlignmentConfig | None = None
    intervention: InterventionConfig | None = None
    matrix: MatrixConfig | None = None

    @model_validator(mode="after")
    def validate_direction_boundary(self) -> ExperimentConfig:
        if self.lens is None:
            raise ValueError("lens is required for both research directions")
        concept_fields = {
            "dataset": self.dataset,
            "activations": self.activations,
            "probe": self.probe,
            "alignment": self.alignment,
        }
        if self.direction == "concept_intervention":
            missing = [name for name, value in concept_fields.items() if value is None]
            if missing:
                raise ValueError(
                    "concept_intervention requires: " + ", ".join(missing)
                )
            if self.matrix is not None:
                raise ValueError("concept_intervention must not define matrix")
        else:
            if self.matrix is None:
                raise ValueError("j_space requires matrix")
            forbidden = [
                name
                for name, value in concept_fields.items()
                if value is not None
            ]
            if self.intervention is not None:
                forbidden.append("intervention")
            if forbidden:
                raise ValueError(
                    "j_space must not define concept fields: " + ", ".join(forbidden)
                )
        return self


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load YAML and reject unknown fields before expensive model work starts."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    return ExperimentConfig.model_validate(raw)
