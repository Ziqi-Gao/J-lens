"""Strict registration schema for the blinded LLM-judge study."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class JudgeConfigurationError(ValueError):
    """Raised when the registered judge protocol is unsafe or ambiguous."""


class StrictModel(BaseModel):
    """Reject misspelled fields and prevent mutation after validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceConfig(StrictModel):
    """Immutable J-component and RAPTOR generation sources."""

    j_component_root: str
    raptor_root: str
    concept_definitions_path: str
    concept_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_concepts(self) -> SourceConfig:
        if len(set(self.concept_ids)) != len(self.concept_ids):
            raise ValueError("source concept_ids must be unique")
        return self


class SelectionConfig(StrictModel):
    """Validation-only intervention selection and held-out task sampling."""

    tuning_split: Literal["validation"] = "validation"
    evaluation_splits: list[Literal["validation", "test"]] = Field(
        default_factory=lambda: ["validation", "test"]
    )
    prompt_family: Literal["open"] = "open"
    decodings: list[Literal["greedy", "sample"]] = Field(default_factory=lambda: ["greedy"])
    j_condition: Literal["j"] = "j"
    j_positive_strengths_only: Literal[True] = True
    raptor_probability_above: float = Field(default=0.5, ge=0.5, lt=1.0)
    include_j_controls: list[Literal["full", "j", "non_j", "random"]] = Field(
        default_factory=lambda: ["full", "j", "non_j", "random"]
    )
    random_control_seeds: list[int] = Field(default_factory=lambda: [101])

    @model_validator(mode="after")
    def validate_selection(self) -> SelectionConfig:
        if len(set(self.evaluation_splits)) != len(self.evaluation_splits) or set(
            self.evaluation_splits
        ) != {"validation", "test"}:
            raise ValueError("evaluation_splits must contain validation and test exactly once")
        if not self.decodings or len(set(self.decodings)) != len(self.decodings):
            raise ValueError("decodings must be non-empty and unique")
        if set(self.include_j_controls) != {"full", "j", "non_j", "random"}:
            raise ValueError("matched full, j, non_j, and random controls are required")
        if (
            not self.random_control_seeds
            or len(set(self.random_control_seeds)) != len(self.random_control_seeds)
            or any(value < 0 for value in self.random_control_seeds)
        ):
            raise ValueError("random_control_seeds must be unique and non-negative")
        return self


class JudgeModelsConfig(StrictModel):
    """Exact non-OpenAI model IDs and their pre-registered roles."""

    primary: str
    secondary: str
    arbitration: str
    expert_review: str

    @model_validator(mode="after")
    def validate_models(self) -> JudgeModelsConfig:
        roles = (self.primary, self.secondary, self.arbitration, self.expert_review)
        if len(set(roles)) != len(roles):
            raise ValueError("judge role models must be distinct")
        for model in roles:
            lowered = model.casefold()
            if (
                "/" not in model
                or lowered.startswith("openai/")
                or lowered in {"openrouter/auto", "openrouter/free"}
                or lowered.endswith((":free", ":latest"))
            ):
                raise ValueError("judges require exact, concrete, non-OpenAI model IDs")
        return self

    def for_role(
        self, role: Literal["primary", "secondary", "arbitration", "expert_review"]
    ) -> str:
        return str(getattr(self, role))


class OpenRouterConfig(StrictModel):
    """Privacy-preserving OpenRouter request policy."""

    endpoint: Literal["https://openrouter.ai/api/v1/chat/completions"] = (
        "https://openrouter.ai/api/v1/chat/completions"
    )
    key_status_endpoint: Literal["https://openrouter.ai/api/v1/key"] = (
        "https://openrouter.ai/api/v1/key"
    )
    api_key_env: str = "JLENS_JUDGE_API_KEY"
    timeout_seconds: float = Field(default=180.0, gt=0)
    max_retries: int = Field(default=1, ge=0, le=10)
    max_tokens: int = Field(default=400, ge=128, le=4096)
    concurrency: int = Field(default=2, ge=1, le=32)
    temperature: Literal[0.0] = 0.0
    temperature_unsupported_models: list[str] = Field(default_factory=list)
    reasoning_effort_by_model: dict[str, Literal["minimal", "low", "medium", "high"]] = Field(
        default_factory=dict
    )
    provider_order_by_model: dict[str, list[str]] = Field(default_factory=dict)
    exclude_reasoning: Literal[True] = True
    data_collection: Literal["deny"] = "deny"
    require_parameters: Literal[True] = True
    allow_provider_fallbacks: Literal[False] = False

    @model_validator(mode="after")
    def validate_secret_name(self) -> OpenRouterConfig:
        if not self.api_key_env.isidentifier():
            raise ValueError("api_key_env must be an environment-variable identifier")
        if len(set(self.temperature_unsupported_models)) != len(
            self.temperature_unsupported_models
        ):
            raise ValueError("temperature_unsupported_models must be unique")
        for model, order in self.provider_order_by_model.items():
            if not model.strip() or not order:
                raise ValueError("provider_order_by_model requires non-empty model and order")
            if len(set(order)) != len(order):
                raise ValueError(f"provider order must be unique for {model}")
            if any(
                not provider.strip() or any(character.isspace() for character in provider)
                for provider in order
            ):
                raise ValueError(f"provider order contains an invalid slug for {model}")
        return self


class BudgetConfig(StrictModel):
    """Fail-closed per-request, per-invocation, and experiment spending limits."""

    require_explicit_limit: Literal[True] = True
    require_live_credit_check: Literal[True] = True
    max_tasks_per_invocation: int = Field(default=25, ge=1)
    max_smoke_tasks_per_invocation: int = Field(default=1, ge=1)
    max_estimated_prompt_tokens_per_request: int = Field(default=5_000, ge=1)
    max_completion_tokens_per_request: int = Field(default=400, ge=1)
    max_estimated_total_tokens_per_invocation: int = Field(default=275_000, ge=1)
    max_cost_usd_per_invocation: float = Field(default=1.50, gt=0)
    max_experiment_cost_usd: float = Field(default=15.0, gt=0)
    reserve_openrouter_credit_usd: float = Field(default=75.0, ge=0)
    pricing_safety_multiplier: float = Field(default=2.0, ge=1.0)

    @model_validator(mode="after")
    def validate_limits(self) -> BudgetConfig:
        if self.max_smoke_tasks_per_invocation > self.max_tasks_per_invocation:
            raise ValueError("smoke task cap cannot exceed the general invocation cap")
        if self.max_cost_usd_per_invocation > self.max_experiment_cost_usd:
            raise ValueError("invocation cost cap cannot exceed the experiment cost cap")
        return self


class CalibrationConfig(StrictModel):
    """Pre-registered machine and optional human agreement gates."""

    min_completion_rate: float = Field(default=0.98, ge=0, le=1)
    min_pointwise_spearman: float = Field(default=0.60, ge=-1, le=1)
    max_pointwise_mae: float = Field(default=20.0, ge=0, le=100)
    min_pairwise_order_consistency: float = Field(default=0.80, ge=0, le=1)
    min_pairwise_cross_judge_agreement: float = Field(default=0.70, ge=0, le=1)
    human_annotations_path: str | None = None
    require_human_gate: bool = False
    min_human_spearman: float = Field(default=0.60, ge=-1, le=1)

    @model_validator(mode="after")
    def validate_human_gate(self) -> CalibrationConfig:
        if self.require_human_gate and not self.human_annotations_path:
            raise ValueError("require_human_gate=true requires human_annotations_path")
        return self


class ReviewConfig(StrictModel):
    """Selective disagreement review that never changes the primary estimator."""

    pointwise_disagreement_threshold: int = Field(default=25, ge=1, le=100)
    expert_disagreement_threshold: int = Field(default=20, ge=1, le=100)
    expert_audit_fraction: float = Field(default=0.05, ge=0, le=1)
    expert_provider_filtered_task_ids: list[str] = Field(default_factory=list)
    expert_provider_filter_escalation: bool = False
    human_audit_per_concept: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def validate_provider_filtered_tasks(self) -> ReviewConfig:
        task_ids = self.expert_provider_filtered_task_ids
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("expert_provider_filtered_task_ids must be unique")
        if any(
            len(task_id) != 64
            or task_id != task_id.lower()
            or any(character not in "0123456789abcdef" for character in task_id)
            for task_id in task_ids
        ):
            raise ValueError(
                "expert_provider_filtered_task_ids must contain lowercase SHA-256 task IDs"
            )
        return self


class AnalysisConfig(StrictModel):
    """Prompt-clustered inference and separately reported quality guardrails."""

    bootstrap_samples: int = Field(default=10_000, ge=1_000)
    permutation_samples: int = Field(default=10_000, ge=1_000)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1)
    quality_noninferiority_margin: float = Field(default=-10.0, ge=-100, le=0)
    refusal_rate_margin: float = Field(default=0.05, ge=0, le=1)
    primary_contrast: Literal["j_minus_raptor"] = "j_minus_raptor"
    multiple_comparison_correction: Literal["benjamini_hochberg"] = "benjamini_hochberg"
    report_after_invalid_guardrail_failure: bool = False


class PricingConfig(StrictModel):
    """Registered USD-per-million-token prices used for conservative preflight."""

    input_per_million: dict[str, float] = Field(default_factory=dict)
    output_per_million: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_prices(self) -> PricingConfig:
        for table in (self.input_per_million, self.output_per_million):
            if any(value < 0 for value in table.values()):
                raise ValueError("token prices cannot be negative")
        return self


class JudgeEvaluationConfig(StrictModel):
    """Complete registered protocol for a blinded judge evaluation."""

    schema_version: Literal[1] = 1
    protocol_version: Literal[
        "concept_intervention_llm_judge_v1",
        "concept_intervention_llm_judge_v2",
        "concept_intervention_llm_judge_v3",
    ] = "concept_intervention_llm_judge_v1"
    experiment_name: str
    output_dir: str
    seed: int = 42
    source: SourceConfig
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    judges: JudgeModelsConfig
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
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
            unknown = set(table) - models
            if unknown:
                raise ValueError(f"{table_name} contains unregistered models: {unknown}")
            missing = models - set(table)
            if missing:
                raise ValueError(f"{table_name} lacks registered models: {missing}")
            nonpositive = {model for model in models if table[model] <= 0}
            if nonpositive:
                raise ValueError(f"{table_name} must be positive for: {nonpositive}")
        if self.openrouter.max_tokens > self.budget.max_completion_tokens_per_request:
            raise ValueError("openrouter max_tokens exceeds the registered completion budget")
        unknown_temperature_models = set(self.openrouter.temperature_unsupported_models) - models
        if unknown_temperature_models:
            raise ValueError(
                f"temperature_unsupported_models are not registered: {unknown_temperature_models}"
            )
        unknown_provider_models = set(self.openrouter.provider_order_by_model) - models
        if unknown_provider_models:
            raise ValueError(
                f"provider_order_by_model contains unregistered models: {unknown_provider_models}"
            )
        if self.review.expert_provider_filter_escalation:
            expert_model = self.judges.expert_review
            provider_order = self.openrouter.provider_order_by_model.get(expert_model, [])
            required_routes = self.openrouter.max_retries + 1
            if len(provider_order) < required_routes:
                raise ValueError(
                    "expert_provider_filter_escalation requires at least one distinct "
                    "registered provider per automatic attempt: "
                    f"{len(provider_order)} < {required_routes} for {expert_model}"
                )
        if not self.experiment_name.strip():
            raise ValueError("experiment_name cannot be empty")
        return self


def load_judge_config(path: str | Path) -> JudgeEvaluationConfig:
    """Load an isolated judge YAML without importing the model/GPU stack."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise JudgeConfigurationError(f"judge config must be a mapping: {source}")
    try:
        return JudgeEvaluationConfig.model_validate(payload)
    except ValueError as error:
        raise JudgeConfigurationError(str(error)) from error
