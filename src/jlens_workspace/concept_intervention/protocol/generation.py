"""Method-neutral generation schemas, identities, and pure validators."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from jlens_workspace.concept_intervention.protocol.contracts import PromptRecord
from jlens_workspace.foundation.artifacts import (
    resolve_repository_resource,
    sha256_file,
)


class InterventionGenerationError(ValueError):
    """Raised when a generation grid violates the shared output contract."""


_CANDIDATE_PROMPT_FAMILY_SPLITS = {
    "choose": "validation",
    "complete": "validation",
    "classify": "test",
    "report": "test",
}

_SCIENTIFIC_MANIFEST_TOP_LEVEL_FIELDS = (
    "schema_version",
    "experiment_name",
    "seed",
    "model_id",
    "model_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "lens_source",
    "lens_revision",
    "dataset_source",
    "dataset_revision",
    "dataset_hash",
    "git_commit",
    "python",
    "packages",
)
_SCIENTIFIC_MANIFEST_NOTE_FIELDS = (
    "direction",
    "coordinate",
    "config_sha256",
    "force_bos",
    "workflow",
    "generation",
    "selected_layers_sha256",
    "row_manifest_sha256",
)
_EXPECTED_MANIFEST_RUNTIME_FIELDS = {
    "git_commit",
    "python",
    "packages",
}
_INDEX_BUILDER_FIELDS = ("git_commit", "platform", "python", "packages")


@dataclass(frozen=True)
class OpenPromptRecord:
    prompt_id: str
    split: str
    raw_text: str
    formatted_text: str


@dataclass(frozen=True)
class GenerationSettings:
    sample_seeds: tuple[int, ...] = (1001, 2002, 3003)
    max_new_tokens: int = 128
    temperature: float = 0.75
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3

    def __post_init__(self) -> None:
        if (
            not self.sample_seeds
            or len(set(self.sample_seeds)) != len(self.sample_seeds)
            or any(seed < 0 for seed in self.sample_seeds)
        ):
            raise InterventionGenerationError(
                "sample seeds must be unique and non-negative"
            )
        if self.max_new_tokens < 1:
            raise InterventionGenerationError("max_new_tokens must be positive")
        if not 0 < self.top_p <= 1 or self.temperature <= 0:
            raise InterventionGenerationError(
                "temperature and top_p must define valid sampling"
            )


def prompt_ids_sha256(prompt_ids: Sequence[str]) -> str:
    """Hash an ordered prompt-ID sequence using the shared JSON convention."""

    encoded = json.dumps(
        [str(value) for value in prompt_ids],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def candidate_prompt_splits(prompt_ids: Sequence[str]) -> dict[str, str]:
    """Map every registered candidate prompt family to its frozen split."""

    output: dict[str, str] = {}
    for prompt_id in prompt_ids:
        family = str(prompt_id).partition("_")[0]
        try:
            output[str(prompt_id)] = _CANDIDATE_PROMPT_FAMILY_SPLITS[family]
        except KeyError as error:
            raise InterventionGenerationError(
                f"unknown candidate prompt family for {prompt_id!r}"
            ) from error
    return output


def build_generation_contract(
    prompts: Sequence[PromptRecord | OpenPromptRecord],
    settings: GenerationSettings,
    *,
    candidate_labels: Mapping[str, str],
) -> dict[str, Any]:
    """Freeze the exact prompt-by-decoding cardinality for one grid shard."""

    prompt_ids = [str(prompt.prompt_id) for prompt in prompts]
    if not prompt_ids or len(set(prompt_ids)) != len(prompt_ids):
        raise InterventionGenerationError(
            "generation prompts must be non-empty with unique IDs"
        )
    candidate_prompts = [
        prompt for prompt in prompts if isinstance(prompt, PromptRecord)
    ]
    candidate_label_ids = tuple(str(value) for value in candidate_labels)
    if (
        len(candidate_label_ids) != 7
        or len(set(candidate_label_ids)) != 7
        or any(
            len(prompt.label_order) != 7
            or set(prompt.label_order) != set(candidate_label_ids)
            for prompt in candidate_prompts
        )
    ):
        raise InterventionGenerationError(
            "candidate generation contract requires the same seven labels "
            "on every candidate prompt"
        )
    candidate_ids = [str(prompt.prompt_id) for prompt in candidate_prompts]
    return {
        "schema_version": 1,
        "prompt_ids": prompt_ids,
        "prompt_ids_sha256": prompt_ids_sha256(prompt_ids),
        "prompt_count": len(prompt_ids),
        "candidate_prompt_ids": candidate_ids,
        "candidate_prompt_splits": candidate_prompt_splits(candidate_ids),
        "candidate_labels": {
            str(concept_id): str(label)
            for concept_id, label in candidate_labels.items()
        },
        "open_prompt_ids": [
            str(prompt.prompt_id)
            for prompt in prompts
            if isinstance(prompt, OpenPromptRecord)
        ],
        "sample_seeds": list(settings.sample_seeds),
        "decodings_per_prompt": 1 + len(settings.sample_seeds),
        "expected_rows": len(prompt_ids) * (1 + len(settings.sample_seeds)),
        "generation_settings": asdict(settings),
    }


def _validate_method_telemetry_event(
    *,
    method: str,
    event: Mapping[str, Any],
    grid_point: Mapping[str, Any],
    generation_id: str,
) -> None:
    """Validate method-owned telemetry fields against the scientific grid."""

    required = {
        "j_component_intervention": {
            "active_positions",
            "strength",
            "residual_norm",
            "kind",
        },
        "iti_intervention": {
            "active_positions",
            "multiplier",
        },
        "raptor_intervention": {
            "target_probability",
            "pre_intervention_logit",
            "pre_intervention_probability",
            "epsilon",
            "steered",
        },
    }[method]
    if not required.issubset(event):
        raise InterventionGenerationError(
            f"method-specific telemetry is incomplete for {generation_id}"
        )
    if method == "j_component_intervention":
        strength = float(event["strength"])
        residual_norm = float(event["residual_norm"])
        if (
            not math.isfinite(strength)
            or not math.isclose(
                strength,
                float(grid_point["strength"]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isfinite(residual_norm)
            or residual_norm <= 0.0
            or event["kind"] not in {"addition", "project_out"}
            or int(event["active_positions"]) < 1
        ):
            raise InterventionGenerationError(
                f"method-specific J telemetry is invalid for {generation_id}"
            )
    elif method == "iti_intervention":
        multiplier = float(event["multiplier"])
        if (
            not math.isfinite(multiplier)
            or not math.isclose(
                multiplier,
                float(grid_point["strength"]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or int(event["active_positions"]) < 1
        ):
            raise InterventionGenerationError(
                f"method-specific ITI telemetry is invalid for {generation_id}"
            )
    else:
        target_probability = float(event["target_probability"])
        pre_logit = float(event["pre_intervention_logit"])
        pre_probability = float(event["pre_intervention_probability"])
        epsilon = float(event["epsilon"])
        if (
            not all(
                math.isfinite(value)
                for value in (
                    target_probability,
                    pre_logit,
                    pre_probability,
                    epsilon,
                )
            )
            or not math.isclose(
                target_probability,
                float(grid_point["target_probability"]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not 0.0 <= pre_probability <= 1.0
            or not isinstance(event["steered"], bool)
        ):
            raise InterventionGenerationError(
                f"method-specific RAPTOR telemetry is invalid for {generation_id}"
            )


def _format_chat(tokenizer: Any, text: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return text


def load_open_prompt_bank(
    path: str | Path, *, tokenizer: Any
) -> list[OpenPromptRecord]:
    """Load the frozen neutral open-ended prompt schema."""

    source = resolve_repository_resource(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise InterventionGenerationError(
            f"unsupported open prompt schema: {source}"
        )
    output: list[OpenPromptRecord] = []
    seen: set[str] = set()
    for row in payload.get("prompts", []):
        prompt_id = str(row["prompt_id"])
        split = str(row["split"])
        if prompt_id in seen or split not in {"validation", "test"}:
            raise InterventionGenerationError(
                f"invalid or duplicate open prompt {prompt_id!r}"
            )
        seen.add(prompt_id)
        text = str(row["text"])
        output.append(
            OpenPromptRecord(
                prompt_id=prompt_id,
                split=split,
                raw_text=text,
                formatted_text=_format_chat(tokenizer, text),
            )
        )
    if not output:
        raise InterventionGenerationError("open prompt bank must be non-empty")
    counts = {
        split: sum(prompt.split == split for prompt in output)
        for split in ("validation", "test")
    }
    if counts["validation"] != counts["test"]:
        raise InterventionGenerationError(
            "open prompt bank must have equal validation/test sizes"
        )
    return output


def _identifier(payload: Mapping[str, Any], *, prefix: str) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{prefix}\0{encoded}".encode()).hexdigest()


def validate_generation_artifacts(
    output_dir: str | Path,
    files: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    expected_selected_layers: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Validate hashes, cardinality, IDs, token logprobs, and blind alignment."""

    root = Path(output_dir)
    checks = (
        ("generations.jsonl", "generations_sha256"),
        ("judge_blind_generations.jsonl", "blind_generations_sha256"),
        ("judge_blind_map.jsonl", "blind_map_sha256"),
    )
    for filename, hash_key in checks:
        path = root / filename
        if not path.is_file() or files.get(hash_key) != sha256_file(path):
            raise InterventionGenerationError(
                f"generation artifact identity mismatch: {path}"
            )
    if int(contract.get("schema_version", -1)) != 1:
        raise InterventionGenerationError("unsupported generation contract")
    prompt_ids = [str(value) for value in contract.get("prompt_ids", [])]
    candidate_prompt_ids = [
        str(value) for value in contract.get("candidate_prompt_ids", [])
    ]
    open_prompt_ids = [
        str(value) for value in contract.get("open_prompt_ids", [])
    ]
    sample_seeds = [int(value) for value in contract.get("sample_seeds", [])]
    expected_rows = int(contract.get("expected_rows", -1))
    if (
        not prompt_ids
        or len(set(prompt_ids)) != len(prompt_ids)
        or [*candidate_prompt_ids, *open_prompt_ids] != prompt_ids
        or int(contract.get("prompt_count", -1)) != len(prompt_ids)
        or contract.get("prompt_ids_sha256") != prompt_ids_sha256(prompt_ids)
        or not sample_seeds
        or len(set(sample_seeds)) != len(sample_seeds)
        or int(contract.get("decodings_per_prompt", -1))
        != 1 + len(sample_seeds)
        or expected_rows != len(prompt_ids) * (1 + len(sample_seeds))
    ):
        raise InterventionGenerationError("invalid generation contract cardinality")

    def read_rows(filename: str) -> list[dict[str, Any]]:
        with (root / filename).open(encoding="utf-8") as handle:
            try:
                return [
                    json.loads(line)
                    for line in handle
                    if line.strip()
                ]
            except json.JSONDecodeError as error:
                raise InterventionGenerationError(
                    f"invalid generation JSONL: {root / filename}"
                ) from error

    generations = read_rows("generations.jsonl")
    blind = read_rows("judge_blind_generations.jsonl")
    mapping = read_rows("judge_blind_map.jsonl")
    if {len(generations), len(blind), len(mapping)} != {expected_rows}:
        raise InterventionGenerationError(
            "generation artifact row count differs from prompt-by-decoding contract"
        )

    expected_keys = {
        *[(prompt_id, "greedy", None) for prompt_id in prompt_ids],
        *[
            (prompt_id, "sample", seed)
            for prompt_id in prompt_ids
            for seed in sample_seeds
        ],
    }
    observed_keys: set[tuple[str, str, int | None]] = set()
    generation_ids: set[str] = set()
    blind_ids: set[str] = set()
    by_blind: dict[str, dict[str, Any]] = {}
    scientific_identities: set[str] = set()
    expected_settings = json.dumps(
        contract.get("generation_settings"), sort_keys=True
    )
    for row in generations:
        seed = row.get("seed")
        key = (
            str(row.get("prompt_id")),
            str(row.get("decoding")),
            None if seed is None else int(seed),
        )
        if key in observed_keys:
            raise InterventionGenerationError(f"duplicate generation key: {key}")
        observed_keys.add(key)
        generation_id = str(row.get("generation_id", ""))
        blind_id = str(row.get("blind_id", ""))
        if (
            not generation_id
            or generation_id in generation_ids
            or not blind_id
            or blind_id in blind_ids
        ):
            raise InterventionGenerationError("generation/blind IDs must be unique")
        generation_ids.add(generation_id)
        blind_ids.add(blind_id)
        by_blind[blind_id] = row
        method = str(row.get("method", ""))
        concept_id = str(row.get("concept_id", ""))
        condition_id = str(row.get("condition_id", ""))
        grid_point = row.get("grid_point")
        intervention_metadata = row.get("intervention_metadata")
        if (
            method
            not in {
                "j_component_intervention",
                "iti_intervention",
                "raptor_intervention",
            }
            or not concept_id
            or not condition_id
            or not isinstance(grid_point, Mapping)
            or not isinstance(intervention_metadata, Mapping)
        ):
            raise InterventionGenerationError(
                f"invalid scientific identity for generation {generation_id}"
            )
        scientific_identities.add(
            json.dumps(
                {
                    "method": method,
                    "concept_id": concept_id,
                    "condition_id": condition_id,
                    "grid_point": dict(grid_point),
                    "intervention_metadata": dict(intervention_metadata),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        token_ids = row.get("generated_token_ids")
        token_logprobs = row.get("token_log_probabilities")
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(not isinstance(value, int) for value in token_ids)
            or not isinstance(token_logprobs, list)
            or len(token_ids) != len(token_logprobs)
            or any(
                not isinstance(value, int | float)
                or not math.isfinite(float(value))
                for value in token_logprobs
            )
            or not isinstance(row.get("generated_text"), str)
        ):
            raise InterventionGenerationError(
                f"incomplete token/logprob output for generation {generation_id}"
            )
        telemetry = row.get("telemetry")
        injected_by_layer = row.get("injected_norm_by_layer")
        total_injected_norm = row.get("total_injected_norm")
        if (
            not isinstance(telemetry, list)
            or not isinstance(injected_by_layer, dict)
            or any(
                not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in injected_by_layer.values()
            )
            or not isinstance(total_injected_norm, int | float)
            or not math.isfinite(float(total_injected_norm))
            or float(total_injected_norm) < 0.0
            or not math.isclose(
                float(total_injected_norm),
                sum(float(value) for value in injected_by_layer.values()),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise InterventionGenerationError(
                f"incomplete injection telemetry for generation {generation_id}"
            )
        required_event_fields = {
            "layer",
            "forward_call",
            "generation_step",
            "batch_size",
            "sequence_length",
            "injected_norm",
        }
        recomputed_by_layer: dict[str, float] = {}
        calls_by_layer: dict[str, list[int]] = {}
        observed_event_coordinates: set[tuple[str, int]] = set()
        for event in telemetry:
            if (
                not isinstance(event, Mapping)
                or not required_event_fields.issubset(event)
                or not math.isfinite(float(event["injected_norm"]))
                or float(event["injected_norm"]) < 0.0
            ):
                raise InterventionGenerationError(
                    f"malformed per-forward telemetry for generation {generation_id}"
                )
            layer = str(int(event["layer"]))
            forward_call = int(event["forward_call"])
            generation_step = int(event["generation_step"])
            batch_size = int(event["batch_size"])
            sequence_length = int(event["sequence_length"])
            if (
                forward_call < 0
                or generation_step != forward_call
                or batch_size < 1
                or sequence_length < 1
            ):
                raise InterventionGenerationError(
                    f"invalid telemetry coordinates for generation {generation_id}"
                )
            coordinate = (layer, generation_step)
            if coordinate in observed_event_coordinates:
                raise InterventionGenerationError(
                    f"duplicate telemetry coordinate for generation {generation_id}"
                )
            observed_event_coordinates.add(coordinate)
            _validate_method_telemetry_event(
                method=method,
                event=event,
                grid_point=grid_point,
                generation_id=generation_id,
            )
            recomputed_by_layer[layer] = recomputed_by_layer.get(layer, 0.0) + float(
                event["injected_norm"]
            )
            calls_by_layer.setdefault(layer, []).append(forward_call)
        for layer, calls in calls_by_layer.items():
            if sorted(calls) != list(range(len(calls))):
                raise InterventionGenerationError(
                    f"non-contiguous telemetry calls at layer {layer} "
                    f"for generation {generation_id}"
                )
        if set(recomputed_by_layer) != {
            str(value) for value in injected_by_layer
        } or any(
            not math.isclose(
                recomputed_by_layer[layer],
                float(injected_by_layer[layer]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            for layer in recomputed_by_layer
        ):
            raise InterventionGenerationError(
                f"telemetry/layer norm mismatch for generation {generation_id}"
            )
        strength = grid_point.get("strength")
        target_probability = grid_point.get("target_probability")
        intervention_enabled = (
            target_probability is not None
            if method == "raptor_intervention"
            else strength is not None and float(strength) != 0.0
        )
        if intervention_enabled and not telemetry:
            raise InterventionGenerationError(
                f"nonzero intervention has empty telemetry: {generation_id}"
            )
        expected_layers_key = (
            "active_layers"
            if method == "iti_intervention"
            else "selected_layers"
        )
        selected_layers = {
            str(int(value))
            for value in intervention_metadata.get("selected_layers", [])
        }
        expected_layers = {
            str(int(value))
            for value in intervention_metadata.get(expected_layers_key, [])
        }
        if not selected_layers or (
            expected_selected_layers is not None
            and selected_layers
            != {str(int(value)) for value in expected_selected_layers}
        ):
            raise InterventionGenerationError(
                f"intervention selected-layer identity mismatch: {generation_id}"
            )
        if method == "iti_intervention":
            selected_heads = intervention_metadata.get("selected_heads")
            if not isinstance(selected_heads, list) or not selected_heads:
                raise InterventionGenerationError(
                    f"ITI selected-head metadata is missing: {generation_id}"
                )
            head_layers = {
                str(int(head["layer"]))
                for head in selected_heads
                if isinstance(head, Mapping) and "layer" in head
            }
            if (
                len(head_layers) == 0
                or head_layers != expected_layers
                or not expected_layers.issubset(selected_layers)
            ):
                raise InterventionGenerationError(
                    f"ITI active-layer metadata is invalid: {generation_id}"
                )
        telemetry_required = (
            method in {"j_component_intervention", "iti_intervention"}
            or intervention_enabled
        )
        if telemetry_required and (
            not expected_layers or set(recomputed_by_layer) != expected_layers
        ):
            raise InterventionGenerationError(
                f"intervention telemetry layer coverage mismatch: {generation_id}"
            )
        expected_event_coordinates = {
            (layer, generation_step)
            for layer in expected_layers
            for generation_step in range(len(token_ids))
        }
        if (
            telemetry_required
            and observed_event_coordinates != expected_event_coordinates
        ):
            raise InterventionGenerationError(
                "telemetry generated-token coverage mismatch for "
                f"{generation_id}"
            )
        if (
            intervention_enabled
            and method
            in {"j_component_intervention", "iti_intervention"}
            and float(total_injected_norm) <= 0.0
        ):
            raise InterventionGenerationError(
                f"nonzero intervention has zero injected norm: {generation_id}"
            )
        if not intervention_enabled and float(total_injected_norm) != 0.0:
            raise InterventionGenerationError(
                f"zero/no-hook generation has nonzero injected norm: {generation_id}"
            )
        if (
            json.dumps(row.get("generation_settings"), sort_keys=True)
            != expected_settings
        ):
            raise InterventionGenerationError(
                f"generation settings differ for {generation_id}"
            )
    if observed_keys != expected_keys:
        raise InterventionGenerationError(
            "generation prompt/decoding IDs differ from the contract"
        )
    if len(scientific_identities) != 1:
        raise InterventionGenerationError(
            "generation shard mixes scientific condition identities"
        )

    blind_by_id = {str(row.get("blind_id", "")): row for row in blind}
    map_by_id = {str(row.get("blind_id", "")): row for row in mapping}
    if (
        len(blind_by_id) != expected_rows
        or len(map_by_id) != expected_rows
        or set(blind_by_id) != blind_ids
        or set(map_by_id) != blind_ids
    ):
        raise InterventionGenerationError(
            "blind export/map IDs do not align with full generations"
        )
    for blind_id, full in by_blind.items():
        public = blind_by_id[blind_id]
        private = map_by_id[blind_id]
        if (
            public.get("prompt_id") != full.get("prompt_id")
            or public.get("decoding") != full.get("decoding")
            or public.get("generated_text") != full.get("generated_text")
            or private.get("generation_id") != full.get("generation_id")
            or private.get("method") != full.get("method")
            or private.get("concept_id") != full.get("concept_id")
            or private.get("condition_id") != full.get("condition_id")
            or private.get("grid_point") != full.get("grid_point")
        ):
            raise InterventionGenerationError(
                f"blind export alignment mismatch for {blind_id}"
            )
    return {
        "rows": expected_rows,
        "prompt_count": len(prompt_ids),
        "decodings_per_prompt": 1 + len(sample_seeds),
        "unique_generation_ids": len(generation_ids),
        "unique_blind_ids": len(blind_ids),
    }


def validate_candidate_score_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
    contract: Mapping[str, Any],
    expected_method: str,
    expected_concept_id: str,
    expected_grid_condition: Mapping[str, Any],
) -> dict[str, Any]:
    """Require one complete candidate-score row per registered candidate prompt."""

    source = Path(path)
    if not source.is_file() or sha256_file(source) != expected_sha256:
        raise InterventionGenerationError(
            f"candidate-score artifact identity mismatch: {source}"
        )
    prompt_ids = [
        str(value) for value in contract.get("candidate_prompt_ids", [])
    ]
    prompt_splits = contract.get("candidate_prompt_splits")
    candidate_labels = contract.get("candidate_labels")
    expected_prompt_splits = candidate_prompt_splits(prompt_ids)
    if (
        not prompt_ids
        or len(set(prompt_ids)) != len(prompt_ids)
        or not isinstance(prompt_splits, Mapping)
        or set(prompt_splits) != set(prompt_ids)
        or dict(prompt_splits) != expected_prompt_splits
        or not isinstance(candidate_labels, Mapping)
        or len(candidate_labels) != 7
        or len(set(candidate_labels)) != 7
        or expected_concept_id not in candidate_labels
    ):
        raise InterventionGenerationError(
            "generation contract lacks fixed prompt splits or seven candidate labels"
        )
    with source.open(encoding="utf-8") as handle:
        try:
            rows = [json.loads(line) for line in handle if line.strip()]
        except json.JSONDecodeError as error:
            raise InterventionGenerationError(
                f"invalid candidate-score JSONL: {source}"
            ) from error
    observed_prompt_ids = [str(row.get("prompt_id", "")) for row in rows]
    if (
        len(rows) != len(prompt_ids)
        or len(set(observed_prompt_ids)) != len(observed_prompt_ids)
        or set(observed_prompt_ids) != set(prompt_ids)
    ):
        raise InterventionGenerationError(
            f"candidate-score row count/IDs differ from prompt contract: {source}"
        )
    condition_field_map = {
        "mode": "condition",
    }
    for row in rows:
        if (
            row.get("method") != expected_method
            or row.get("target_concept_id") != expected_concept_id
            or row.get("evaluation_split") not in {"validation", "test"}
        ):
            raise InterventionGenerationError(
                f"candidate-score scientific identity mismatch: {source}"
            )
        if row.get("evaluation_split") != prompt_splits[str(row["prompt_id"])]:
            raise InterventionGenerationError(
                f"candidate prompt family/split mismatch: {source}"
            )
        for key, value in expected_grid_condition.items():
            row_key = condition_field_map.get(key, key)
            if row.get(row_key) != value:
                raise InterventionGenerationError(
                    f"candidate-score grid condition mismatch: {source}"
                )
        log_probabilities = row.get("candidate_log_probabilities")
        probabilities = row.get("candidate_probabilities_normalized")
        target = expected_concept_id
        if (
            not isinstance(log_probabilities, Mapping)
            or not isinstance(probabilities, Mapping)
            or set(log_probabilities) != set(probabilities)
            or set(log_probabilities) != set(candidate_labels)
            or target not in log_probabilities
            or any(
                not isinstance(value, int | float)
                or not math.isfinite(float(value))
                for value in [*log_probabilities.values(), *probabilities.values()]
            )
        ):
            raise InterventionGenerationError(
                f"candidate labels/probabilities differ from frozen contract: {source}"
            )
        scalar_fields = (
            "target_log_probability",
            "target_candidate_probability",
            "target_margin",
        )
        if any(
            not isinstance(row.get(field), int | float)
            or not math.isfinite(float(row[field]))
            for field in scalar_fields
        ) or not isinstance(row.get("target_rank"), int):
            raise InterventionGenerationError(
                f"incomplete candidate metrics: {source}"
            )
        off_target = [
            float(value)
            for key, value in log_probabilities.items()
            if key != target
        ]
        expected_margin = float(log_probabilities[target]) - (
            sum(off_target) / len(off_target)
        )
        expected_rank = 1 + sum(
            value > float(log_probabilities[target]) for value in off_target
        )
        if (
            not math.isclose(
                float(row["target_log_probability"]),
                float(log_probabilities[target]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(row["target_candidate_probability"]),
                float(probabilities[target]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not 1 <= int(row["target_rank"]) <= len(log_probabilities)
            or int(row["target_rank"]) != expected_rank
            or not math.isclose(
                float(row["target_margin"]),
                expected_margin,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            or not math.isclose(
                sum(float(value) for value in probabilities.values()),
                1.0,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        ):
            raise InterventionGenerationError(
                f"inconsistent candidate target metrics: {source}"
            )
    return {
        "rows": len(rows),
        "prompt_ids_sha256": prompt_ids_sha256(prompt_ids),
    }


def validate_generation_contract_identity(
    contract: Mapping[str, Any],
    generation_identity: Mapping[str, Any],
) -> None:
    """Match a shard contract to the prompt banks frozen in the run manifest."""

    candidate_ids = [
        str(value) for value in contract.get("candidate_prompt_ids", [])
    ]
    open_ids = [str(value) for value in contract.get("open_prompt_ids", [])]
    candidate_labels = contract.get("candidate_labels")
    candidate_splits = contract.get("candidate_prompt_splits")
    settings = contract.get("generation_settings")
    if (
        not isinstance(settings, Mapping)
        or not isinstance(candidate_labels, Mapping)
        or not isinstance(candidate_splits, Mapping)
        or dict(candidate_labels)
        != generation_identity.get("candidate_labels")
        or dict(candidate_splits)
        != generation_identity.get("candidate_prompt_splits")
        or contract.get("prompt_ids_sha256")
        != generation_identity.get("prompt_ids_sha256")
        or int(contract.get("prompt_count", -1))
        != int(generation_identity.get("prompt_count", -2))
        or len(candidate_ids)
        != int(generation_identity.get("candidate_prompt_count", -1))
        or len(open_ids)
        != int(generation_identity.get("open_prompt_count", -1))
        or int(contract.get("expected_rows", -1))
        != int(generation_identity.get("expected_rows_per_grid_point", -2))
        or any(
            settings.get(key) != generation_identity.get(key)
            for key in (
                "sample_seeds",
                "max_new_tokens",
                "temperature",
                "top_p",
                "repetition_penalty",
                "no_repeat_ngram_size",
            )
        )
    ):
        raise InterventionGenerationError(
            "generation contract differs from run-manifest prompt identity"
        )


def validate_shard_manifest_identity(
    method_manifest: Mapping[str, Any],
    shard_manifest: Mapping[str, Any],
) -> None:
    """Compare immutable scientific identity while allowing platform changes.

    Kernel/platform provenance is mandatory on both manifests, but it is not a
    scientific identity field. After every shard passes this check, callers may
    use the steering execution layer to build the method index and aggregate
    every observed platform.
    """

    method_identity = scientific_shard_identity(method_manifest)
    shard_identity = scientific_shard_identity(shard_manifest)
    if method_identity != shard_identity:
        differing = sorted(
            field
            for field in _SCIENTIFIC_MANIFEST_TOP_LEVEL_FIELDS
            if method_identity.get(field) != shard_identity.get(field)
        )
        method_notes = method_identity["notes"]
        shard_notes = shard_identity["notes"]
        differing.extend(
            f"notes.{field}"
            for field in _SCIENTIFIC_MANIFEST_NOTE_FIELDS
            if method_notes.get(field) != shard_notes.get(field)
        )
        raise InterventionGenerationError(
            "grid shard scientific identity mismatch: " + ", ".join(differing)
        )


def scientific_shard_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Extract and validate the platform-independent scientific shard identity."""

    missing = [
        field
        for field in (*_SCIENTIFIC_MANIFEST_TOP_LEVEL_FIELDS, "platform", "notes")
        if field not in manifest
    ]
    if missing:
        raise InterventionGenerationError(
            "grid shard manifest lacks required provenance fields: "
            + ", ".join(missing)
        )
    platform_value = manifest.get("platform")
    if not isinstance(platform_value, str) or not platform_value.strip():
        raise InterventionGenerationError(
            "grid shard manifest platform must be a non-empty string"
        )
    if not isinstance(manifest.get("git_commit"), str) or not manifest.get(
        "git_commit"
    ):
        raise InterventionGenerationError(
            "grid shard manifest git_commit must be a non-empty string"
        )
    if not isinstance(manifest.get("python"), str) or not manifest.get("python"):
        raise InterventionGenerationError(
            "grid shard manifest python must be a non-empty string"
        )
    packages = manifest.get("packages")
    if not isinstance(packages, Mapping):
        raise InterventionGenerationError(
            "grid shard manifest packages must be a mapping"
        )
    notes = manifest.get("notes")
    if not isinstance(notes, Mapping):
        raise InterventionGenerationError(
            "grid shard manifest notes must be a mapping"
        )
    missing_notes = [
        field for field in _SCIENTIFIC_MANIFEST_NOTE_FIELDS if field not in notes
    ]
    if missing_notes:
        raise InterventionGenerationError(
            "grid shard manifest lacks required scientific notes: "
            + ", ".join(missing_notes)
        )
    return {
        **{
            field: dict(packages) if field == "packages" else manifest.get(field)
            for field in _SCIENTIFIC_MANIFEST_TOP_LEVEL_FIELDS
        },
        "notes": {
            field: notes.get(field)
            for field in _SCIENTIFIC_MANIFEST_NOTE_FIELDS
        },
    }


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _manifest_payload(value: Any) -> dict[str, Any]:
    payload = asdict(value) if is_dataclass(value) else dict(value)
    return json.loads(json.dumps(payload))


def _validate_expected_manifest_identity(
    expected_manifest: Mapping[str, Any],
    scientific_manifest: Mapping[str, Any],
) -> None:
    """Match current config expectations without conflating builder runtime."""

    expected = scientific_shard_identity(expected_manifest)
    observed = scientific_shard_identity(scientific_manifest)
    differing = [
        field
        for field in _SCIENTIFIC_MANIFEST_TOP_LEVEL_FIELDS
        if field not in _EXPECTED_MANIFEST_RUNTIME_FIELDS
        and expected.get(field) != observed.get(field)
    ]
    differing.extend(
        f"notes.{field}"
        for field in _SCIENTIFIC_MANIFEST_NOTE_FIELDS
        if expected["notes"].get(field) != observed["notes"].get(field)
    )
    if differing:
        raise InterventionGenerationError(
            "scientific shards differ from the configured index identity: "
            + ", ".join(differing)
        )


def _validate_index_builder(value: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in _INDEX_BUILDER_FIELDS if field not in value]
    if missing:
        raise InterventionGenerationError(
            "index-builder provenance lacks fields: " + ", ".join(missing)
        )
    git_commit = value.get("git_commit")
    if not isinstance(git_commit, str) or len(git_commit) not in {40, 64}:
        raise InterventionGenerationError(
            "index-builder git_commit must be a 40- or 64-character digest"
        )
    try:
        int(git_commit, 16)
    except ValueError as error:
        raise InterventionGenerationError(
            "index-builder git_commit must be hexadecimal"
        ) from error
    for field in ("platform", "python"):
        if not isinstance(value.get(field), str) or not value.get(field):
            raise InterventionGenerationError(
                f"index-builder {field} must be a non-empty string"
            )
    if not isinstance(value.get("packages"), Mapping):
        raise InterventionGenerationError(
            "index-builder packages must be a mapping"
        )
    return {
        "git_commit": git_commit,
        "platform": str(value["platform"]),
        "python": str(value["python"]),
        "packages": dict(value["packages"]),
    }


def validate_index_builder_provenance(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize an index-builder runtime identity."""

    return _validate_index_builder(value)


def validate_method_index_provenance(
    root: str | Path,
    index: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate a method index's complete manifest/platform provenance graph."""

    method_root = Path(root).resolve()
    summary = index.get("shard_manifest_provenance")
    if not isinstance(summary, Mapping):
        raise InterventionGenerationError(
            "method index lacks shard-manifest provenance"
        )
    relative_path = summary.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise InterventionGenerationError(
            "method index has malformed shard-manifest provenance path"
        )
    registry_path = (method_root / relative_path).resolve()
    if (
        not registry_path.is_relative_to(method_root)
        or not registry_path.is_file()
        or summary.get("sha256") != sha256_file(registry_path)
    ):
        raise InterventionGenerationError(
            f"shard-manifest provenance registry identity mismatch: {registry_path}"
        )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = registry.get("manifests")
    if not isinstance(entries, list) or not entries:
        raise InterventionGenerationError(
            "shard-manifest provenance registry has no manifests"
        )
    expected_count = len(entries)
    if (
        registry.get("schema_version") != 1
        or registry.get("shard_count") != expected_count
        or summary.get("shard_count") != expected_count
        or registry.get("platforms") != summary.get("platforms")
    ):
        raise InterventionGenerationError(
            "shard-manifest provenance registry cardinality is malformed"
        )
    paths: list[Path] = []
    observed_platforms: dict[str, int] = {}
    previous_path = ""
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise InterventionGenerationError(
                "shard-manifest provenance entry must be a mapping"
            )
        path_value = entry.get("path")
        platform_value = entry.get("platform")
        if (
            not isinstance(path_value, str)
            or not path_value
            or path_value <= previous_path
            or not isinstance(platform_value, str)
            or not platform_value
        ):
            raise InterventionGenerationError(
                "shard-manifest provenance entry is malformed or unsorted"
            )
        path = (method_root / path_value).resolve()
        if (
            not path.is_relative_to(method_root)
            or not path.is_file()
            or entry.get("manifest_sha256") != sha256_file(path)
        ):
            raise InterventionGenerationError(
                f"registered shard manifest identity mismatch: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("platform") != platform_value:
            raise InterventionGenerationError(
                f"registered shard platform mismatch: {path}"
            )
        paths.append(path)
        observed_platforms[platform_value] = (
            observed_platforms.get(platform_value, 0) + 1
        )
        previous_path = path_value
    reference = json.loads(paths[0].read_text(encoding="utf-8"))
    for path in paths[1:]:
        validate_shard_manifest_identity(
            reference,
            json.loads(path.read_text(encoding="utf-8")),
        )
    platforms = [
        {"platform": value, "shard_count": observed_platforms[value]}
        for value in sorted(observed_platforms)
    ]
    identity = scientific_shard_identity(reference)
    identity_sha256 = _canonical_sha256(identity)
    if (
        registry.get("platforms") != platforms
        or registry.get("scientific_shard_identity") != identity
        or registry.get("scientific_shard_identity_sha256") != identity_sha256
        or index.get("scientific_shard_identity") != identity
        or index.get("scientific_shard_identity_sha256") != identity_sha256
    ):
        raise InterventionGenerationError(
            "method index scientific shard provenance is inconsistent"
        )
    builder = _validate_index_builder(index.get("index_builder", {}))
    method_manifest_path = method_root / "manifest.json"
    method_manifest = json.loads(method_manifest_path.read_text(encoding="utf-8"))
    validate_shard_manifest_identity(method_manifest, reference)
    if (
        method_manifest.get("scientific_shard_identity") != identity
        or method_manifest.get("scientific_shard_identity_sha256") != identity_sha256
        or method_manifest.get("shard_manifest_provenance") != dict(summary)
        or method_manifest.get("index_builder") != builder
    ):
        raise InterventionGenerationError(
            "method manifest and index provenance differ"
        )
    return {
        "scientific_shard_identity": identity,
        "scientific_shard_identity_sha256": identity_sha256,
        "shard_count": expected_count,
        "platforms": platforms,
        "shard_manifest_provenance": {
            "path": relative_path,
            "sha256": summary["sha256"],
        },
        "index_builder": builder,
    }


def build_target_artifact_seal(
    root: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    """Hash every method-root artifact reachable from one target summary."""

    method_root = Path(root).resolve()
    target_summary = Path(summary_path).resolve()
    paths: set[Path] = {target_summary}
    payload = json.loads(target_summary.read_text(encoding="utf-8"))
    summary_paths = [target_summary]
    for shard in payload.get("shards", []):
        shard_summary = (method_root / str(shard["summary"])).resolve()
        paths.add(shard_summary)
        summary_paths.append(shard_summary)
    for scientific_summary in summary_paths:
        directory = scientific_summary.parent
        for filename in (
            "candidate_scores.jsonl",
            "generations.jsonl",
            "judge_blind_generations.jsonl",
            "judge_blind_map.jsonl",
        ):
            candidate = directory / filename
            if candidate.is_file():
                paths.add(candidate.resolve())
    rows = []
    for path in sorted(paths, key=str):
        try:
            relative = path.relative_to(method_root)
        except ValueError as error:
            raise InterventionGenerationError(
                f"artifact seal path escapes method root: {path}"
            ) from error
        if not path.is_file():
            raise InterventionGenerationError(
                f"artifact seal input is missing: {path}"
            )
        rows.append(
            {
                "path": str(relative),
                "sha256": sha256_file(path),
            }
        )
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "files": rows,
        "files_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def validate_target_artifact_seal(
    root: str | Path,
    seal: Mapping[str, Any],
) -> set[Path]:
    """Verify an index-time target artifact seal and return its absolute paths."""

    if int(seal.get("schema_version", -1)) != 1:
        raise InterventionGenerationError("unsupported target artifact seal")
    rows = seal.get("files")
    if not isinstance(rows, list) or not rows:
        raise InterventionGenerationError("target artifact seal is empty")
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    if seal.get("files_sha256") != hashlib.sha256(
        encoded.encode("utf-8")
    ).hexdigest():
        raise InterventionGenerationError(
            "target artifact seal list identity mismatch"
        )
    method_root = Path(root).resolve()
    output: set[Path] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise InterventionGenerationError("malformed target artifact seal row")
        path = (method_root / str(row.get("path", ""))).resolve()
        try:
            path.relative_to(method_root)
        except ValueError as error:
            raise InterventionGenerationError(
                f"artifact seal path escapes method root: {path}"
            ) from error
        if (
            path in output
            or not path.is_file()
            or row.get("sha256") != sha256_file(path)
        ):
            raise InterventionGenerationError(
                f"target artifact seal mismatch: {path}"
            )
        output.add(path)
    return output


def validate_equivalent_generation_outputs(
    paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Require zero/no-hook generations to match token-for-token."""

    sources = [Path(path) for path in paths]
    if len(sources) < 2:
        raise InterventionGenerationError(
            "generation equivalence requires at least two artifacts"
        )
    reference: dict[tuple[str, str, int | None], str] | None = None
    for source in sources:
        rows: dict[tuple[str, str, int | None], str] = {}
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                seed = row.get("seed")
                key = (
                    str(row["prompt_id"]),
                    str(row["decoding"]),
                    None if seed is None else int(seed),
                )
                if key in rows:
                    raise InterventionGenerationError(
                        f"duplicate generation key in {source}: {key}"
                    )
                rows[key] = json.dumps(
                    {
                        "generated_token_ids": row["generated_token_ids"],
                        "generated_text": row["generated_text"],
                        "token_log_probabilities": row[
                            "token_log_probabilities"
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
        if not rows:
            raise InterventionGenerationError(
                f"generation equivalence artifact is empty: {source}"
            )
        if reference is None:
            reference = rows
        elif rows != reference:
            raise InterventionGenerationError(
                f"zero/no-hook generation outputs differ: {source}"
            )
    assert reference is not None
    digest = hashlib.sha256()
    for key, value in sorted(reference.items(), key=lambda item: str(item[0])):
        digest.update(repr(key).encode())
        digest.update(value.encode())
    return {
        "artifact_count": len(sources),
        "rows_per_artifact": len(reference),
        "content_sha256": digest.hexdigest(),
    }


__all__ = [
    "GenerationSettings",
    "InterventionGenerationError",
    "OpenPromptRecord",
    "PromptRecord",
    "build_generation_contract",
    "build_target_artifact_seal",
    "candidate_prompt_splits",
    "load_open_prompt_bank",
    "prompt_ids_sha256",
    "scientific_shard_identity",
    "validate_candidate_score_artifact",
    "validate_equivalent_generation_outputs",
    "validate_generation_artifacts",
    "validate_generation_contract_identity",
    "validate_index_builder_provenance",
    "validate_method_index_provenance",
    "validate_shard_manifest_identity",
    "validate_target_artifact_seal",
]
