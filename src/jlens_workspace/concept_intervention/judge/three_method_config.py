"""Strict registration schema for the three-method blinded LLM-judge study."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from jlens_workspace.concept_intervention.judge.config import (
    CalibrationConfig,
    JudgeConfigurationError,
    JudgeModelsConfig,
    OpenRouterConfig,
    PricingConfig,
    ReviewConfig,
    StrictModel,
)


class SourceConfig(StrictModel):
    """Immutable three-method generation, comparison, and rescore sources."""

    comparison_index: str
    comparison_index_sha256: str
    comparison_sha256: str
    j_component_root: str
    raptor_root: str
    iti_root: str
    candidate_rescore_root: str
    concept_definitions_path: str
    concept_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source(self) -> SourceConfig:
        if len(set(self.concept_ids)) != len(self.concept_ids):
            raise ValueError("source concept_ids must be unique")
        for label, value in (
            ("comparison_index_sha256", self.comparison_index_sha256),
            ("comparison_sha256", self.comparison_sha256),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        if len(
            {self.j_component_root, self.raptor_root, self.iti_root}
        ) != 3:
            raise ValueError("the three method roots must be distinct")
        return self


class SelectionConfig(StrictModel):
    """Validation-only selection and method-neutral generation pairing."""

    tuning_split: Literal["validation"] = "validation"
    evaluation_splits: list[Literal["validation", "test"]] = Field(
        default_factory=lambda: ["validation", "test"]
    )
    prompt_family: Literal["open"] = "open"
    decodings: list[Literal["greedy", "sample"]] = Field(
        default_factory=lambda: ["greedy", "sample"]
    )
    sample_seeds: list[int] = Field(default_factory=lambda: [1001, 2002, 3003])
    j_positive_strengths_only: Literal[True] = True
    raptor_probability_above: float = Field(default=0.5, ge=0.5, lt=1.0)
    matched_random_seed: Literal[101] = 101

    @model_validator(mode="after")
    def validate_selection(self) -> SelectionConfig:
        if self.evaluation_splits != ["validation", "test"]:
            raise ValueError("evaluation_splits must be [validation, test]")
        if self.decodings != ["greedy", "sample"]:
            raise ValueError(
                "three-method evaluation requires greedy and sample decodings"
            )
        if self.sample_seeds != [1001, 2002, 3003]:
            raise ValueError(
                "sample_seeds must preserve [1001, 2002, 3003]"
            )
        return self


class BudgetConfig(StrictModel):
    """Fail-closed request, invocation, study, and operator approval limits."""

    require_explicit_limit: Literal[True] = True
    require_live_credit_check: Literal[True] = True
    require_formal_budget_approval: Literal[True] = True
    formal_budget_approval_file: str
    max_tasks_per_invocation: int = Field(default=25, ge=1, le=100)
    max_smoke_tasks_per_invocation: Literal[1] = 1
    max_estimated_prompt_tokens_per_request: int = Field(default=7168, ge=1)
    max_completion_tokens_per_request: int = Field(default=1024, ge=1)
    max_estimated_total_tokens_per_invocation: int = Field(default=300000, ge=1)
    max_cost_usd_per_invocation: float = Field(default=2.0, gt=0)
    max_experiment_cost_usd: float = Field(gt=0)
    reserve_openrouter_credit_usd: float = Field(default=75.0, ge=0)
    pricing_safety_multiplier: float = Field(default=2.0, ge=1.0)

    @model_validator(mode="after")
    def validate_limits(self) -> BudgetConfig:
        if self.max_cost_usd_per_invocation > self.max_experiment_cost_usd:
            raise ValueError(
                "invocation cost cap cannot exceed the experiment cost cap"
            )
        if not Path(self.formal_budget_approval_file).is_absolute():
            raise ValueError("formal_budget_approval_file must be absolute")
        return self


class AnalysisConfig(StrictModel):
    """Prompt-clustered three-method inference and separate guardrails."""

    bootstrap_samples: int = Field(default=10_000, ge=1_000)
    permutation_samples: int = Field(default=10_000, ge=1_000)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1)
    quality_noninferiority_margin: float = Field(default=-10.0, ge=-100, le=0)
    refusal_rate_margin: float = Field(default=0.05, ge=0, le=1)
    primary_contrasts: list[
        Literal[
            "j_minus_raptor",
            "j_minus_iti_native",
            "raptor_minus_iti_native",
        ]
    ] = Field(
        default_factory=lambda: [
            "j_minus_raptor",
            "j_minus_iti_native",
            "raptor_minus_iti_native",
        ]
    )
    multiple_comparison_correction: Literal["benjamini_hochberg"] = (
        "benjamini_hochberg"
    )
    descriptive_only: Literal[True] = True

    @model_validator(mode="after")
    def validate_contrasts(self) -> AnalysisConfig:
        expected = [
            "j_minus_raptor",
            "j_minus_iti_native",
            "raptor_minus_iti_native",
        ]
        if self.primary_contrasts != expected:
            raise ValueError(
                f"primary_contrasts must preserve registered order: {expected}"
            )
        return self


class LegacyAssessmentConfig(StrictModel):
    """Read-only source used only to prove whether old results are reusable."""

    source_root: str
    expected_experiment_name: str
    expected_manifest_sha256: str


class JudgeEvaluationConfig(StrictModel):
    """Complete three-method blinded judge registration."""

    schema_version: Literal[1] = 1
    protocol_version: Literal["concept_intervention_llm_judge_v3"] = (
        "concept_intervention_llm_judge_v3"
    )
    study_design_version: Literal["three_method_llm_judge_v1"] = (
        "three_method_llm_judge_v1"
    )
    experiment_name: Literal["qwen35_4b_three_method_llm_judge_v1"]
    output_dir: str
    seed: Literal[42] = 42
    source: SourceConfig
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    judges: JudgeModelsConfig
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    budget: BudgetConfig
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    legacy_assessment: LegacyAssessmentConfig | None = None
    pricing: PricingConfig = Field(default_factory=PricingConfig)

    @model_validator(mode="after")
    def validate_registration(self) -> JudgeEvaluationConfig:
        models = {
            self.judges.primary,
            self.judges.secondary,
            self.judges.arbitration,
            self.judges.expert_review,
        }
        for table_name, table in (
            ("input_per_million", self.pricing.input_per_million),
            ("output_per_million", self.pricing.output_per_million),
        ):
            if set(table) != models:
                raise ValueError(
                    f"{table_name} must contain exactly the four judge models"
                )
        if self.openrouter.max_tokens > self.budget.max_completion_tokens_per_request:
            raise ValueError("openrouter max_tokens exceeds completion budget")
        policy_models = (
            set(self.openrouter.temperature_unsupported_models)
            | set(self.openrouter.reasoning_effort_by_model)
            | set(self.openrouter.provider_order_by_model)
        )
        if policy_models - models:
            raise ValueError("OpenRouter policy contains an unregistered model")
        if self.review.expert_provider_filter_escalation:
            routes = self.openrouter.provider_order_by_model.get(
                self.judges.expert_review, []
            )
            if len(routes) < self.openrouter.max_retries + 1:
                raise ValueError(
                    "expert filter escalation requires one route per attempt"
                )
        output = Path(self.output_dir)
        if not output.is_absolute() or output.name != "llm_judge_three_method_v1":
            raise ValueError(
                "output_dir must be an isolated llm_judge_three_method_v1 root"
            )
        return self


def load_judge_config(path: str | Path) -> JudgeEvaluationConfig:
    """Load the isolated judge YAML without importing model/GPU packages."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise JudgeConfigurationError(f"judge config must be a mapping: {source}")
    try:
        return JudgeEvaluationConfig.model_validate(payload)
    except ValueError as error:
        raise JudgeConfigurationError(str(error)) from error
