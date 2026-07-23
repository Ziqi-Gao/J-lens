"""Strict, versioned configuration shared by command-line experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
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
    fit_prompt_offset: int = Field(default=0, ge=0)
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
        if self.source == "huggingface" and not self.revision:
            raise ValueError("revision is required for a Hugging Face lens")
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
    random_control_seeds: list[int] = Field(
        default_factory=lambda: [101, 202, 303, 404, 505]
    )

    @model_validator(mode="after")
    def validate_control_seeds(self) -> AlignmentConfig:
        if any(seed < 0 for seed in self.random_control_seeds):
            raise ValueError("random_control_seeds must be non-negative")
        if len(set(self.random_control_seeds)) != len(self.random_control_seeds):
            raise ValueError("random_control_seeds must be unique")
        if not self.require_nonnegative:
            raise ValueError("only non-negative sparse J decomposition is supported")
        return self


class GenerationConfig(StrictModel):
    """Shared exhaustive generation contract for all intervention methods."""

    candidate_prompts_path: str
    open_prompts_path: str
    greedy: bool = True
    sample_seeds: list[int] = Field(default_factory=lambda: [1001, 2002, 3003])
    max_new_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.75, gt=0)
    top_p: float = Field(default=0.95, gt=0, le=1)
    repetition_penalty: float = Field(default=1.1, gt=0)
    no_repeat_ngram_size: int = Field(default=3, ge=0)

    @model_validator(mode="after")
    def validate_generation(self) -> GenerationConfig:
        if (
            not self.greedy
            or not self.sample_seeds
            or len(set(self.sample_seeds)) != len(self.sample_seeds)
            or any(seed < 0 for seed in self.sample_seeds)
        ):
            raise ValueError(
                "generation requires greedy output and unique non-negative sample seeds"
            )
        return self


class SharedLayerSelectionConfig(StrictModel):
    """Method-neutral RAPTOR-style residual-probe layer selection."""

    method: Literal["raptor_validation_accuracy"] = "raptor_validation_accuracy"
    upstream_checkout: str = "third_party_external/RAPTOR"
    upstream_commit: Literal["cf7405899174af39f3970e093e4b86bf0972ff87"] = (
        "cf7405899174af39f3970e093e4b86bf0972ff87"
    )
    candidate_layers: list[int] = Field(min_length=2)
    selected_layer_count: int = Field(default=6, ge=1)
    source_activations_dir: str
    output_dir: str
    concept_ids: list[str] = Field(min_length=1)
    c_grid: list[float] = Field(
        default_factory=lambda: np.logspace(-4, 2, 100).tolist()
    )
    cv_folds: int = Field(default=5, ge=2)
    max_iter: int = Field(default=5000, ge=100)
    seed: int = 42

    @model_validator(mode="after")
    def validate_layer_selection(self) -> SharedLayerSelectionConfig:
        if (
            sorted(set(self.candidate_layers)) != self.candidate_layers
            or any(layer < 0 for layer in self.candidate_layers)
        ):
            raise ValueError(
                "shared candidate_layers must be sorted, unique, and non-negative"
            )
        if self.selected_layer_count >= len(self.candidate_layers):
            raise ValueError(
                "selected_layer_count must be smaller than the candidate layer count"
            )
        if len(set(self.concept_ids)) != len(self.concept_ids):
            raise ValueError("shared concept_ids must be unique")
        if (
            len(self.c_grid) != 100
            or any(not math.isfinite(value) or value <= 0 for value in self.c_grid)
            or not math.isclose(self.c_grid[0], 1e-4, rel_tol=1e-9)
            or not math.isclose(self.c_grid[-1], 100.0, rel_tol=1e-9)
        ):
            raise ValueError(
                "shared RAPTOR C grid must contain 100 positive values from 1e-4 to 100"
            )
        return self


class JComponentInterventionConfig(StrictModel):
    """Multi-layer J/full/non-J/random intervention experiment."""

    method: Literal["j_component_intervention"] = "j_component_intervention"
    selected_layers_path: str
    source_occupancy_dir: str
    source_probes_dir: str
    source_activations_dir: str
    concept_ids: list[str] = Field(min_length=1)
    convention: Literal["rmsnorm_weighted"] = "rmsnorm_weighted"
    strengths: list[float] = Field(
        default_factory=lambda: [-0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5]
    )
    random_control_seeds: list[int] = Field(
        default_factory=lambda: [101, 202, 303, 404, 505]
    )
    probe_replicates: list[str] = Field(
        default_factory=lambda: [
            "primary",
            "bootstrap_1101",
            "bootstrap_2202",
            "bootstrap_3303",
            "bootstrap_4404",
        ]
    )
    k_max: int = Field(default=64, ge=1)
    candidate_labels: dict[str, str]
    generation: GenerationConfig
    score_batch_size: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def validate_j_component(self) -> JComponentInterventionConfig:
        if (
            len(set(self.concept_ids)) != len(self.concept_ids)
            or set(self.candidate_labels) != set(self.concept_ids)
        ):
            raise ValueError(
                "J-component candidate_labels must exactly match unique concept_ids"
            )
        _validate_signed_grid(self.strengths, name="J-component strengths")
        _validate_seed_grid(
            self.random_control_seeds, name="J-component random_control_seeds"
        )
        if (
            len(set(self.probe_replicates)) != len(self.probe_replicates)
            or self.probe_replicates[0] != "primary"
        ):
            raise ValueError(
                "J-component probe_replicates must be unique and start with primary"
            )
        return self


class RaptorInterventionConfig(StrictModel):
    """Pinned external RAPTOR adaptive multi-layer intervention."""

    method: Literal["raptor_intervention"] = "raptor_intervention"
    upstream_repository: Literal["https://github.com/Ziqi-Gao/RAPTOR.git"] = (
        "https://github.com/Ziqi-Gao/RAPTOR.git"
    )
    upstream_commit: Literal["cf7405899174af39f3970e093e4b86bf0972ff87"] = (
        "cf7405899174af39f3970e093e4b86bf0972ff87"
    )
    upstream_checkout: str
    selected_layers_path: str
    source_probes_dir: str
    source_activations_dir: str
    concept_ids: list[str] = Field(min_length=1)
    target_probabilities: list[float] = Field(
        default_factory=lambda: [
            0.0001,
            0.001,
            0.01,
            0.1,
            0.9,
            0.99,
            0.999,
            0.9999,
        ]
    )
    candidate_labels: dict[str, str]
    generation: GenerationConfig
    score_batch_size: int = Field(default=1, ge=1, le=1)

    @model_validator(mode="after")
    def validate_raptor(self) -> RaptorInterventionConfig:
        if (
            len(set(self.concept_ids)) != len(self.concept_ids)
            or set(self.candidate_labels) != set(self.concept_ids)
        ):
            raise ValueError(
                "RAPTOR candidate_labels must exactly match unique concept_ids"
            )
        probabilities = self.target_probabilities
        if (
            len(set(probabilities)) != len(probabilities)
            or any(not 0 < value < 1 for value in probabilities)
            or not any(value < 0.5 for value in probabilities)
            or not any(value > 0.5 for value in probabilities)
        ):
            raise ValueError(
                "RAPTOR target_probabilities must be unique in (0,1) and cover both signs"
            )
        return self


def _validate_signed_grid(values: list[float], *, name: str) -> None:
    if (
        not values
        or len(set(values)) != len(values)
        or any(not math.isfinite(value) for value in values)
        or 0.0 not in values
        or not any(value < 0 for value in values)
        or not any(value > 0 for value in values)
    ):
        raise ValueError(f"{name} must be finite, unique, and include zero and both signs")


def _validate_seed_grid(values: list[int], *, name: str) -> None:
    if (
        not values
        or len(set(values)) != len(values)
        or any(value < 0 for value in values)
    ):
        raise ValueError(f"{name} must be non-empty, unique, and non-negative")


class InterventionConfig(StrictModel):
    method: Literal[
        "generic_residual_intervention_v1",
        "concept_j_component_intervention_v2",
    ] = "generic_residual_intervention_v1"
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
    layer: int | None = Field(default=None, ge=0)
    concept_ids: list[str] | None = None
    convention: Literal["raw", "rmsnorm_weighted"] = "rmsnorm_weighted"
    conditions: list[Literal["full", "j", "non_j", "random"]] = Field(
        default_factory=lambda: ["full", "j", "non_j", "random"]
    )
    random_control_seeds: list[int] = Field(
        default_factory=lambda: [101, 202, 303, 404, 505]
    )
    source_occupancy_dir: str | None = None
    source_occupancy_git_commit: str | None = None
    expected_occupancy_index_sha256: str | None = None
    source_probes_dir: str | None = None
    source_activations_dir: str | None = None
    prompts_path: str | None = None
    candidate_labels: dict[str, str] | None = None
    score_batch_size: int = Field(default=8, ge=1)
    generation_prompt_count: int = Field(default=4, ge=0)

    @model_validator(mode="after")
    def validate_intervention(self) -> InterventionConfig:
        if (
            not self.strengths
            or len(set(self.strengths)) != len(self.strengths)
            or any(not math.isfinite(value) for value in self.strengths)
        ):
            raise ValueError("intervention strengths must be finite and unique")
        if 0.0 not in self.strengths:
            raise ValueError("intervention strengths must include zero")
        if not any(value < 0 for value in self.strengths) or not any(
            value > 0 for value in self.strengths
        ):
            raise ValueError("intervention strengths must include both signs")
        if (
            not self.random_control_seeds
            or len(set(self.random_control_seeds))
            != len(self.random_control_seeds)
            or any(seed < 0 for seed in self.random_control_seeds)
        ):
            raise ValueError(
                "random_control_seeds must be non-empty, unique, and non-negative"
            )
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("intervention conditions must be unique")
        if self.method == "concept_j_component_intervention_v2":
            required = {
                "layer": self.layer,
                "concept_ids": self.concept_ids,
                "source_occupancy_dir": self.source_occupancy_dir,
                "source_occupancy_git_commit": self.source_occupancy_git_commit,
                "expected_occupancy_index_sha256": (
                    self.expected_occupancy_index_sha256
                ),
                "source_probes_dir": self.source_probes_dir,
                "source_activations_dir": self.source_activations_dir,
                "prompts_path": self.prompts_path,
                "candidate_labels": self.candidate_labels,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(
                    "concept_j_component_intervention_v2 requires: "
                    + ", ".join(missing)
                )
            if self.kind != "addition" or self.position != "last_prompt":
                raise ValueError(
                    "concept_j_component_intervention_v2 requires addition at "
                    "last_prompt"
                )
            if not self.scale_by_residual_norm:
                raise ValueError(
                    "concept_j_component_intervention_v2 requires "
                    "scale_by_residual_norm=true"
                )
            if set(self.conditions) != {"full", "j", "non_j", "random"}:
                raise ValueError(
                    "concept_j_component_intervention_v2 requires matched full, "
                    "j, non_j, and random conditions"
                )
            assert self.concept_ids is not None
            assert self.candidate_labels is not None
            if (
                len(set(self.concept_ids)) != len(self.concept_ids)
                or set(self.candidate_labels) != set(self.concept_ids)
            ):
                raise ValueError(
                    "candidate_labels keys must exactly match unique concept_ids"
                )
            assert self.source_occupancy_git_commit is not None
            assert self.expected_occupancy_index_sha256 is not None
            for name, value in (
                ("source_occupancy_git_commit", self.source_occupancy_git_commit),
                (
                    "expected_occupancy_index_sha256",
                    self.expected_occupancy_index_sha256,
                ),
            ):
                if len(value) not in {40, 64}:
                    raise ValueError(f"{name} must be a Git/SHA-256 hex digest")
                int(value, 16)
        return self


class ITIConfig(StrictModel):
    """Pinned honest_llama ITI comparison on full-attention head outputs."""

    method: Literal[
        "honest_llama_mass_mean_qwen_full_attention_v1",
        "honest_llama_mass_mean_qwen_full_attention",
    ] = (
        "honest_llama_mass_mean_qwen_full_attention_v1"
    )
    upstream_repository: Literal["https://github.com/likenneth/honest_llama"] = (
        "https://github.com/likenneth/honest_llama"
    )
    upstream_commit: Literal["2c6b2179be7b5aa8f0a171688cf9e01b812ca327"] = (
        "2c6b2179be7b5aa8f0a171688cf9e01b812ca327"
    )
    attention_layers: list[int] = Field(min_length=1)
    num_heads: int = Field(ge=1)
    head_dim: int = Field(ge=1)
    top_k_grid: list[int] = Field(default_factory=lambda: [4, 8, 16, 32, 48])
    strengths: list[float] = Field(
        default_factory=lambda: [-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0]
    )
    random_control_seeds: list[int] = Field(
        default_factory=lambda: [101, 202, 303, 404, 505], min_length=2
    )
    concept_ids: list[str] = Field(min_length=1)
    source_residual_activations_dir: str
    head_activations_dir: str
    directions_dir: str
    reference_j_intervention_dir: str
    prompts_path: str
    candidate_labels: dict[str, str]
    validation_prompt_prefixes: list[str] = Field(
        default_factory=lambda: ["choose", "complete"]
    )
    test_prompt_prefixes: list[str] = Field(
        default_factory=lambda: ["classify", "report"]
    )
    capture_batch_size: int = Field(default=8, ge=1)
    capture_max_length: int = Field(default=256, ge=8)
    score_batch_size: int = Field(default=8, ge=1)
    generation_prompt_count: int = Field(default=4, ge=0)
    max_new_tokens: int = Field(default=4, ge=1)
    selected_layers_path: str | None = None
    variants: list[Literal["native", "layer_matched"]] = Field(
        default_factory=lambda: ["native"]
    )
    layer_matched_top_k_grid: list[int] = Field(
        default_factory=lambda: [8, 16, 32, 48]
    )
    selected_layer_count: int = Field(default=6, ge=1)
    generation: GenerationConfig | None = None

    @model_validator(mode="after")
    def validate_iti(self) -> ITIConfig:
        if (
            sorted(set(self.attention_layers)) != self.attention_layers
            or any(layer < 0 for layer in self.attention_layers)
        ):
            raise ValueError("ITI attention_layers must be sorted, unique, and non-negative")
        total_heads = len(self.attention_layers) * self.num_heads
        if (
            not self.top_k_grid
            or sorted(set(self.top_k_grid)) != self.top_k_grid
            or self.top_k_grid[0] < 1
            or self.top_k_grid[-1] > total_heads
        ):
            raise ValueError(
                f"ITI top_k_grid must be sorted and unique within [1, {total_heads}]"
            )
        if (
            len(set(self.strengths)) != len(self.strengths)
            or any(not math.isfinite(value) for value in self.strengths)
            or 0.0 not in self.strengths
            or not any(value < 0 for value in self.strengths)
            or not any(value > 0 for value in self.strengths)
        ):
            raise ValueError("ITI strengths must be finite, unique, and include zero and both signs")
        if (
            not self.random_control_seeds
            or len(set(self.random_control_seeds)) != len(self.random_control_seeds)
            or any(seed < 0 for seed in self.random_control_seeds)
        ):
            raise ValueError("ITI random_control_seeds must be unique and non-negative")
        if len(set(self.concept_ids)) != len(self.concept_ids):
            raise ValueError("ITI concept_ids must be unique")
        if set(self.candidate_labels) != set(self.concept_ids):
            raise ValueError("ITI candidate_labels keys must exactly match concept_ids")
        if len(set(self.variants)) != len(self.variants):
            raise ValueError("ITI variants must be unique")
        if "layer_matched" in self.variants:
            if self.selected_layers_path is None:
                raise ValueError(
                    "layer-matched ITI requires selected_layers_path"
                )
            matched = self.layer_matched_top_k_grid
            if (
                not matched
                or sorted(set(matched)) != matched
                or matched[0] < self.selected_layer_count
                or matched[-1] > total_heads
            ):
                raise ValueError(
                    "layer_matched_top_k_grid must be sorted, unique, cover every "
                    "selected layer, and not exceed total heads"
                )
        validation = self.validation_prompt_prefixes
        test = self.test_prompt_prefixes
        if (
            not validation
            or not test
            or len(set(validation)) != len(validation)
            or len(set(test)) != len(test)
            or set(validation).intersection(test)
        ):
            raise ValueError("ITI validation/test prompt prefixes must be non-empty and disjoint")
        return self


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
    rank_relative_tolerances: list[float] = Field(
        default_factory=lambda: [1e-5, 1e-6, 1e-7, 1e-8]
    )
    device: str = "cpu"
    cpu_fallback: bool = True

    @model_validator(mode="after")
    def validate_thresholds(self) -> MatrixConfig:
        if any(not 0 < threshold <= 1 for threshold in self.energy_thresholds):
            raise ValueError("energy_thresholds must lie in (0, 1]")
        if (
            not self.rank_relative_tolerances
            or any(
                not math.isfinite(value) or value <= 0
                for value in self.rank_relative_tolerances
            )
            or len(set(self.rank_relative_tolerances))
            != len(self.rank_relative_tolerances)
        ):
            raise ValueError(
                "rank_relative_tolerances must contain unique positive values"
            )
        if self.rank_relative_tolerance not in self.rank_relative_tolerances:
            raise ValueError(
                "rank_relative_tolerances must include rank_relative_tolerance"
            )
        return self


class OccupancyConfig(StrictModel):
    """Concept-vector J-space occupancy with versioned sparse solvers."""

    method: Literal[
        "concept_occupancy_method_v1",
        "concept_occupancy_method_v2",
        "concept_occupancy_method",
    ] = "concept_occupancy_method_v1"
    mode: Literal["exact"] = "exact"
    solver_method: Literal[
        "nnomp_nnls_v1",
        "nonnegative_gradient_pursuit_v2",
        "nonnegative_gradient_pursuit_standard",
    ] = "nnomp_nnls_v1"
    primary_crossing_rule: Literal[
        "first_nonexceed_v1", "consecutive3_nonexceed_v1"
    ] = "first_nonexceed_v1"
    layers: list[int] = Field(min_length=1)
    concept_ids: list[str] | None = None
    signs: list[Literal["+", "-"]] = Field(default_factory=lambda: ["+", "-"])
    conventions: list[Literal["rmsnorm_weighted", "raw"]] = Field(
        default_factory=lambda: ["rmsnorm_weighted", "raw"]
    )
    selection_modes: list[Literal["positive_cosine", "raw_positive_dot"]] = Field(
        default_factory=lambda: ["positive_cosine", "raw_positive_dot"]
    )
    k_max: int = Field(default=64, ge=1)
    report_grid: list[int] = Field(
        default_factory=lambda: [1, 2, 4, 8, 16, 25, 32, 64]
    )
    random_seeds: list[int] = Field(
        default_factory=lambda: [101, 202, 303, 404, 505]
    )
    absolute_thresholds: list[float] = Field(
        default_factory=lambda: [0.01, 0.05, 0.10, 0.20]
    )
    vocabulary_chunk_size: int = Field(default=4096, ge=1)
    device: str = "cpu"
    expected_lens_sha256: str | None = None
    control_atom_fractions: list[float] = Field(default_factory=lambda: [1.0])
    probe_replicates: list[str] | None = None
    bootstrap_seeds: list[int] = Field(
        default_factory=lambda: [1101, 2202, 3303, 4404]
    )
    bootstrap_k_max: int = Field(default=25, ge=1)
    row_manifest_path: str | None = None

    @model_validator(mode="after")
    def validate_occupancy(self) -> OccupancyConfig:
        for name in ("layers", "signs", "conventions", "selection_modes"):
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique")
        if any(layer < 0 for layer in self.layers):
            raise ValueError("layers must be non-negative")
        grid = self.report_grid
        if not grid or sorted(set(grid)) != grid:
            raise ValueError("report_grid must be non-empty, sorted, and unique")
        if grid[0] < 1 or grid[-1] > self.k_max:
            raise ValueError("report_grid must lie in [1, k_max]")
        if not self.random_seeds or len(set(self.random_seeds)) != len(
            self.random_seeds
        ):
            raise ValueError("random_seeds must be non-empty and unique")
        if any(seed < 0 for seed in self.random_seeds):
            raise ValueError("random_seeds must be non-negative")
        if not self.absolute_thresholds or any(
            not 0 < value < 1 for value in self.absolute_thresholds
        ):
            raise ValueError("absolute_thresholds must lie in (0, 1)")
        if len(set(self.absolute_thresholds)) != len(self.absolute_thresholds):
            raise ValueError("absolute_thresholds must be unique")
        if self.expected_lens_sha256 is not None:
            if len(self.expected_lens_sha256) != 64:
                raise ValueError(
                    "expected_lens_sha256 must be a 64-character SHA-256"
                )
            int(self.expected_lens_sha256, 16)
        fractions = self.control_atom_fractions
        if not fractions or len(set(fractions)) != len(fractions):
            raise ValueError("control_atom_fractions must be non-empty and unique")
        if 1.0 not in fractions:
            raise ValueError("control_atom_fractions must include 1.0")
        if any(not 0 < value <= 1 for value in fractions):
            raise ValueError("control_atom_fractions must lie in (0, 1]")
        if self.concept_ids is not None and (
            not self.concept_ids
            or len(set(self.concept_ids)) != len(self.concept_ids)
        ):
            raise ValueError("concept_ids must be non-empty and unique when supplied")
        if self.probe_replicates is not None and (
            not self.probe_replicates
            or len(set(self.probe_replicates)) != len(self.probe_replicates)
        ):
            raise ValueError(
                "probe_replicates must be non-empty and unique when supplied"
            )
        if (
            not self.bootstrap_seeds
            or len(set(self.bootstrap_seeds)) != len(self.bootstrap_seeds)
            or any(seed < 0 for seed in self.bootstrap_seeds)
        ):
            raise ValueError("bootstrap_seeds must be unique and non-negative")
        if self.bootstrap_k_max > self.k_max:
            raise ValueError("bootstrap_k_max must not exceed k_max")
        if self.method == "concept_occupancy_method_v2" and (
            self.solver_method != "nonnegative_gradient_pursuit_v2"
        ):
            raise ValueError(
                "concept_occupancy_method_v2 requires "
                "solver_method=nonnegative_gradient_pursuit_v2"
            )
        if self.method == "concept_occupancy_method" and (
            self.solver_method != "nonnegative_gradient_pursuit_standard"
        ):
            raise ValueError(
                "concept_occupancy_method requires "
                "solver_method=nonnegative_gradient_pursuit_standard"
            )
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
    shared_layer_selection: SharedLayerSelectionConfig | None = None
    j_component: JComponentInterventionConfig | None = None
    raptor: RaptorInterventionConfig | None = None
    iti: ITIConfig | None = None
    occupancy: OccupancyConfig | None = None
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
            if (
                self.occupancy is not None
                and self.lens.source in {"local", "huggingface"}
                and self.occupancy.expected_lens_sha256 is None
            ):
                raise ValueError(
                    "occupancy with a pre-existing lens requires expected_lens_sha256"
                )
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
            if self.shared_layer_selection is not None:
                forbidden.append("shared_layer_selection")
            if self.j_component is not None:
                forbidden.append("j_component")
            if self.raptor is not None:
                forbidden.append("raptor")
            if self.iti is not None:
                forbidden.append("iti")
            if self.occupancy is not None:
                forbidden.append("occupancy")
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
