"""Prepare, execute, calibrate, and review blinded concept-judge tasks."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import numpy as np

from jlens_workspace.artifacts import atomic_write_json, resolve_repository_resource, sha256_file
from jlens_workspace.concept_intervention.evaluation import atomic_write_jsonl
from jlens_workspace.concept_intervention.generation import (
    validate_candidate_score_artifact,
    validate_equivalent_generation_outputs,
    validate_generation_artifacts,
)
from jlens_workspace.concept_intervention.judge.amendment import (
    validate_format_amendment,
)
from jlens_workspace.concept_intervention.judge.client import (
    OpenRouterClient,
    OpenRouterContentFilterError,
)
from jlens_workspace.concept_intervention.judge.config import (
    JudgeConfigurationError,
)
from jlens_workspace.concept_intervention.judge.config import (
    load_judge_config as load_legacy_judge_config,
)
from jlens_workspace.concept_intervention.judge.prompts import (
    PROMPT_VERSION,
    render_messages,
    response_schema,
    rubric_hash,
)
from jlens_workspace.concept_intervention.judge.statistics import agreement_metrics
from jlens_workspace.concept_intervention.judge.three_method_config import (
    JudgeEvaluationConfig,
)
from jlens_workspace.concept_intervention.judge.three_method_config import (
    load_judge_config as load_three_method_judge_config,
)


class JudgeWorkflowError(ValueError):
    """Raised when source identity, blinding, or registered workflow state is invalid."""


def _load_runtime_config(config_path: str | Path) -> Any:
    """Load the new registration while preserving historical two-method workflows."""

    try:
        return load_three_method_judge_config(config_path)
    except JudgeConfigurationError as three_method_error:
        try:
            return load_legacy_judge_config(config_path)
        except JudgeConfigurationError:
            raise three_method_error from None


@dataclass(frozen=True)
class _Shard:
    method: str
    concept_id: str
    summary_path: Path
    summary: dict[str, Any]
    candidate_rows: tuple[dict[str, Any], ...]

    @property
    def condition(self) -> Mapping[str, Any]:
        return self.summary["grid_condition"]

    @property
    def generation_path(self) -> Path:
        return self.summary_path.parent / "generations.jsonl"


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except json.JSONDecodeError as error:
        raise JudgeWorkflowError(f"invalid JSONL: {path}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise JudgeWorkflowError(f"JSONL rows must be objects: {path}")
    return rows


def _registered_resource(path: str, *, config_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    beside_config = config_path.parent / candidate
    if beside_config.exists():
        return beside_config
    return resolve_repository_resource(candidate)


def _load_concept_definitions(
    config: JudgeEvaluationConfig, *, config_path: Path
) -> dict[str, dict[str, str]]:
    path = _registered_resource(config.source.concept_definitions_path, config_path=config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    concepts = payload.get("concepts")
    if not isinstance(concepts, list):
        raise JudgeWorkflowError(f"concept definitions lack concepts: {path}")
    definitions: dict[str, dict[str, str]] = {}
    for entry in concepts:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("concept_name", ""))
        concept_id = str(entry.get("concept_id") or f"goemotions:{name}")
        definition = str(entry.get("definition", ""))
        if concept_id in definitions or not name or not definition:
            raise JudgeWorkflowError(f"invalid concept definition for {concept_id!r}")
        definitions[concept_id] = {"name": name, "definition": definition}
    if set(definitions).intersection(config.source.concept_ids) != set(config.source.concept_ids):
        missing = sorted(set(config.source.concept_ids) - set(definitions))
        raise JudgeWorkflowError(f"missing registered concept definitions: {missing}")
    return {concept: definitions[concept] for concept in config.source.concept_ids}


def _load_shards(root: Path, *, method: str, concept_id: str) -> list[_Shard]:
    target = root / "targets" / quote(concept_id, safe="") / "shards"
    summaries = sorted(target.glob("grid_*/summary.json"))
    if not summaries:
        raise JudgeWorkflowError(f"no {method} shards for {concept_id}: {target}")
    output: list[_Shard] = []
    grid_indexes: set[int] = set()
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        grid_index = int(summary.get("grid_index", -1))
        condition = summary.get("grid_condition")
        if (
            summary.get("method") != method
            or summary.get("target_concept_id") != concept_id
            or grid_index < 0
            or grid_index in grid_indexes
            or not isinstance(condition, Mapping)
        ):
            raise JudgeWorkflowError(f"invalid shard scientific identity: {summary_path}")
        grid_indexes.add(grid_index)
        candidate_path = summary_path.parent / "candidate_scores.jsonl"
        validate_candidate_score_artifact(
            candidate_path,
            expected_sha256=str(summary.get("candidate_scores_sha256", "")),
            contract=summary["generation_contract"],
            expected_method=method,
            expected_concept_id=concept_id,
            expected_grid_condition=condition,
        )
        output.append(
            _Shard(
                method=method,
                concept_id=concept_id,
                summary_path=summary_path,
                summary=summary,
                candidate_rows=tuple(_read_jsonl(candidate_path)),
            )
        )
    return output


def _find_shard(shards: Sequence[_Shard], **condition: Any) -> _Shard:
    matches = [
        shard
        for shard in shards
        if all(shard.condition.get(key) == value for key, value in condition.items())
    ]
    if len(matches) != 1:
        raise JudgeWorkflowError(f"expected one shard for {condition}, found {len(matches)}")
    return matches[0]


def _validation_margin(shard: _Shard) -> float:
    values = [
        float(row["target_margin"])
        for row in shard.candidate_rows
        if row.get("evaluation_split") == "validation"
    ]
    if not values or not np.all(np.isfinite(values)):
        raise JudgeWorkflowError(
            f"validation target margins are missing or invalid: {shard.summary_path}"
        )
    return float(np.mean(values))


def _select_registered_conditions(
    j_shards: Sequence[_Shard],
    raptor_shards: Sequence[_Shard],
    *,
    config: JudgeEvaluationConfig,
) -> tuple[dict[str, _Shard], dict[str, Any]]:
    j_candidates = [
        shard
        for shard in j_shards
        if shard.condition.get("condition_id") == "j"
        and float(shard.condition.get("strength", 0.0)) > 0.0
    ]
    if not j_candidates:
        raise JudgeWorkflowError("J selection lacks positive validation candidates")
    selected_j = max(
        j_candidates,
        key=lambda shard: (
            _validation_margin(shard),
            -float(shard.condition["strength"]),
        ),
    )
    strength = float(selected_j.condition["strength"])
    raptor_candidates = [
        shard
        for shard in raptor_shards
        if shard.condition.get("target_probability") is not None
        and float(shard.condition["target_probability"]) > config.selection.raptor_probability_above
    ]
    if not raptor_candidates:
        raise JudgeWorkflowError("RAPTOR selection lacks positive validation candidates")
    selected_raptor = max(
        raptor_candidates,
        key=lambda shard: (
            _validation_margin(shard),
            -abs(float(shard.condition["target_probability"]) - 0.5),
        ),
    )
    probability = float(selected_raptor.condition["target_probability"])
    selected: dict[str, _Shard] = {
        "baseline": _find_shard(j_shards, condition_id="full", strength=0.0),
        "full": _find_shard(j_shards, condition_id="full", strength=strength),
        "j": selected_j,
        "non_j": _find_shard(j_shards, condition_id="non_j", strength=strength),
        "raptor": selected_raptor,
    }
    for seed in config.selection.random_control_seeds:
        selected[f"random_{seed}"] = _find_shard(
            j_shards, condition_id=f"random_{seed}", strength=strength
        )
    raptor_baseline = _find_shard(raptor_shards, condition_id="no_hook")
    selection = {
        "tuning_split": "validation",
        "selection_metric": "mean candidate-label target_margin",
        "j": {
            "strength": strength,
            "validation_target_margin": _validation_margin(selected_j),
            "tie_break": "smallest positive strength",
        },
        "raptor": {
            "target_probability": probability,
            "validation_target_margin": _validation_margin(selected_raptor),
            "tie_break": "closest probability to 0.5",
        },
        "raptor_baseline_summary": str(raptor_baseline.summary_path),
    }
    return selected | {"_raptor_baseline": raptor_baseline}, selection


def _validate_selected_shards(selected: Mapping[str, _Shard]) -> dict[str, Any]:
    contracts: set[str] = set()
    entries: dict[str, Any] = {}
    for role, shard in selected.items():
        checks = validate_generation_artifacts(
            shard.summary_path.parent,
            shard.summary["generation_files"],
            contract=shard.summary["generation_contract"],
            expected_selected_layers=shard.summary["selected_layers"],
        )
        contracts.add(_canonical_hash(shard.summary["generation_contract"]))
        entries[role] = {
            "method": shard.method,
            "condition": dict(shard.condition),
            "summary": str(shard.summary_path),
            "summary_sha256": sha256_file(shard.summary_path),
            "generations": str(shard.generation_path),
            "generations_sha256": sha256_file(shard.generation_path),
            "validation": checks,
        }
    if len(contracts) != 1:
        raise JudgeWorkflowError("selected shards use different generation contracts")
    baseline = selected["baseline"].generation_path
    raptor_baseline = selected["_raptor_baseline"].generation_path
    zero_check = validate_equivalent_generation_outputs([baseline, raptor_baseline])
    return {
        "generation_contract_sha256": next(iter(contracts)),
        "zero_generation_consistency": zero_check,
        "entries": entries,
    }


def _generation_key(row: Mapping[str, Any]) -> tuple[str, str, int | None]:
    seed = row.get("seed")
    return (
        str(row["prompt_id"]),
        str(row["decoding"]),
        None if seed is None else int(seed),
    )


def _eligible_generations(
    shard: _Shard, *, config: JudgeEvaluationConfig
) -> dict[str, dict[tuple[str, str, int | None], dict[str, Any]]]:
    output: dict[str, dict[tuple[str, str, int | None], dict[str, Any]]] = {
        split: {} for split in config.selection.evaluation_splits
    }
    for row in _read_jsonl(shard.generation_path):
        prompt_id = str(row.get("prompt_id", ""))
        split = str(row.get("prompt_split", ""))
        decoding = str(row.get("decoding", ""))
        if (
            not prompt_id.startswith("open_")
            or split not in output
            or decoding not in config.selection.decodings
        ):
            continue
        key = _generation_key(row)
        if key in output[split]:
            raise JudgeWorkflowError(f"duplicate generation key: {key}")
        output[split][key] = row
    for split, rows in output.items():
        if not rows:
            raise JudgeWorkflowError(
                f"no eligible {split} open generations in {shard.generation_path}"
            )
    return output


def _registered_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    normalized = [dict(row) for row in rows]
    if path.exists():
        if _read_jsonl(path) != normalized:
            raise JudgeWorkflowError(f"registered task artifact changed: {path}")
        return
    atomic_write_jsonl(path, normalized)


def _registered_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    normalized = dict(payload)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise JudgeWorkflowError(f"registered manifest changed: {path}")
        return
    atomic_write_json(path, normalized)


def prepare_evaluation(config_path: str | Path) -> dict[str, Any]:
    """Select on candidate validation data, validate shards, and freeze blind tasks."""

    registration = Path(config_path)
    config = _load_runtime_config(registration)
    destination = Path(config.output_dir)
    definitions = _load_concept_definitions(config, config_path=registration)
    all_public: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    all_private: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    source_entries: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    for concept_id in config.source.concept_ids:
        j_shards = _load_shards(
            Path(config.source.j_component_root),
            method="j_component_intervention",
            concept_id=concept_id,
        )
        raptor_shards = _load_shards(
            Path(config.source.raptor_root),
            method="raptor_intervention",
            concept_id=concept_id,
        )
        selected, selection = _select_registered_conditions(j_shards, raptor_shards, config=config)
        source_entries[concept_id] = _validate_selected_shards(selected)
        selections[concept_id] = selection
        role_rows = {
            role: _eligible_generations(shard, config=config)
            for role, shard in selected.items()
            if not role.startswith("_")
        }
        expected_keys: dict[str, set[tuple[str, str, int | None]]] = {}
        for split in config.selection.evaluation_splits:
            sets = {role: set(rows[split]) for role, rows in role_rows.items()}
            if len({tuple(sorted(value)) for value in sets.values()}) != 1:
                raise JudgeWorkflowError(
                    f"{concept_id} {split}: selected conditions have different prompts"
                )
            expected_keys[split] = next(iter(sets.values()))
        for role, by_split in role_rows.items():
            shard = selected[role]
            for split, rows in by_split.items():
                for key in sorted(rows):
                    row = rows[key]
                    task_id = _canonical_hash(
                        {
                            "protocol": config.protocol_version,
                            "rubric": PROMPT_VERSION,
                            "kind": "pointwise",
                            "generation_id": row["generation_id"],
                        }
                    )
                    public = {
                        "schema_version": 1,
                        "rubric_version": PROMPT_VERSION,
                        "task_id": task_id,
                        "task_type": "pointwise",
                        "split": split,
                        "concept_id": concept_id,
                        "concept_name": definitions[concept_id]["name"],
                        "concept_definition": definitions[concept_id]["definition"],
                        "user_prompt": row["prompt_text"],
                        "response": row["generated_text"],
                    }
                    private = {
                        "task_id": task_id,
                        "generation_id": row["generation_id"],
                        "blind_id": row["blind_id"],
                        "condition_role": role,
                        "method": shard.method,
                        "condition_id": row["condition_id"],
                        "grid_point": row["grid_point"],
                        "concept_id": concept_id,
                        "prompt_id": row["prompt_id"],
                        "decoding": row["decoding"],
                        "seed": row.get("seed"),
                        "source_summary": str(shard.summary_path),
                        "source_summary_sha256": sha256_file(shard.summary_path),
                    }
                    all_public[("pointwise", split)].append(public)
                    all_private[("pointwise", split)].append(private)
        for split, keys in expected_keys.items():
            for key in sorted(keys):
                left = role_rows["j"][split][key]
                right = role_rows["raptor"][split][key]
                pair_group_id = _canonical_hash(
                    {
                        "protocol": config.protocol_version,
                        "kind": "pairwise_group",
                        "j_generation_id": left["generation_id"],
                        "raptor_generation_id": right["generation_id"],
                    }
                )
                initial_j_first = int(pair_group_id[-1], 16) % 2 == 0
                for order_index in range(2):
                    j_first = initial_j_first if order_index == 0 else not initial_j_first
                    response_a, response_b = (left, right) if j_first else (right, left)
                    role_a, role_b = ("j", "raptor") if j_first else ("raptor", "j")
                    task_id = _canonical_hash(
                        {
                            "protocol": config.protocol_version,
                            "rubric": PROMPT_VERSION,
                            "kind": "pairwise",
                            "pair_group_id": pair_group_id,
                            "order_index": order_index,
                        }
                    )
                    public = {
                        "schema_version": 1,
                        "rubric_version": PROMPT_VERSION,
                        "task_id": task_id,
                        "task_type": "pairwise",
                        "split": split,
                        "concept_id": concept_id,
                        "concept_name": definitions[concept_id]["name"],
                        "concept_definition": definitions[concept_id]["definition"],
                        "user_prompt": left["prompt_text"],
                        "response_a": response_a["generated_text"],
                        "response_b": response_b["generated_text"],
                    }
                    private = {
                        "task_id": task_id,
                        "pair_group_id": pair_group_id,
                        "order_index": order_index,
                        "concept_id": concept_id,
                        "prompt_id": left["prompt_id"],
                        "decoding": left["decoding"],
                        "seed": left.get("seed"),
                        "response_a_role": role_a,
                        "response_b_role": role_b,
                        "response_a_generation_id": response_a["generation_id"],
                        "response_b_generation_id": response_b["generation_id"],
                    }
                    all_public[("pairwise", split)].append(public)
                    all_private[("pairwise", split)].append(private)
    task_artifacts: dict[str, Any] = {}
    for (task_type, split), public_rows in sorted(all_public.items()):
        private_rows = all_private[(task_type, split)]
        private_by_id = {row["task_id"]: row for row in private_rows}
        random.Random(f"{config.seed}:{task_type}:{split}").shuffle(public_rows)
        private_rows = [private_by_id[row["task_id"]] for row in public_rows]
        public_path = destination / "tasks" / f"{task_type}_{split}.jsonl"
        private_path = destination / "private" / f"{task_type}_{split}_map.jsonl"
        _registered_write_jsonl(public_path, public_rows)
        _registered_write_jsonl(private_path, private_rows)
        private_path.chmod(0o600)
        private_path.parent.chmod(0o700)
        task_artifacts[f"{task_type}_{split}"] = {
            "path": str(public_path),
            "sha256": sha256_file(public_path),
            "rows": len(public_rows),
            "private_map": str(private_path),
            "private_map_sha256": sha256_file(private_path),
        }
    implementation_root = Path(__file__).parent
    implementation_files = {
        path.name: sha256_file(path) for path in sorted(implementation_root.glob("*.py"))
    }
    manifest = {
        "schema_version": 1,
        "protocol_version": config.protocol_version,
        "experiment_name": config.experiment_name,
        "implementation": {
            "files": implementation_files,
            "combined_sha256": _canonical_hash(implementation_files),
        },
        "config": str(registration),
        "config_sha256": sha256_file(registration),
        "rubric_version": PROMPT_VERSION,
        "rubric_sha256": rubric_hash(),
        "blinding": {
            "hidden_from_judges": [
                "method",
                "condition",
                "strength_or_probability",
                "decoding",
                "seed",
                "research_hypothesis",
            ],
            "target_concept_definition_visible": True,
            "pairwise_order_swapped": True,
        },
        "selection": selections,
        "sources": source_entries,
        "task_artifacts": task_artifacts,
    }
    _registered_write_json(destination / "manifest.json", manifest)
    return manifest


def _task_file(
    root: Path,
    *,
    task_set: Literal["pointwise", "pairwise", "arbitration", "expert_review"],
    split: Literal["validation", "test"] | None,
) -> Path:
    if task_set in {"pointwise", "pairwise"}:
        if split is None:
            raise JudgeWorkflowError(f"{task_set} requires --split")
        return root / "tasks" / f"{task_set}_{split}.jsonl"
    if split not in {None, "test"}:
        raise JudgeWorkflowError(f"{task_set} review tasks are test-only")
    return root / "tasks" / f"{task_set}.jsonl"


def _model_directory(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def _result_directory(
    root: Path,
    *,
    model: str,
    task_set: str,
    split: str | None,
    smoke: bool = False,
) -> Path:
    stem = task_set if split is None else f"{task_set}_{split}"
    top = "smoke_responses" if smoke else "responses"
    return root / top / _model_directory(model) / stem


def _load_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_dir():
        return {}
    output: dict[str, dict[str, Any]] = {}
    for result_path in sorted(path.glob("*.json")):
        if result_path.name == "index.json":
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        task_id = str(payload.get("task_id", ""))
        if not task_id or task_id in output:
            raise JudgeWorkflowError(f"duplicate or invalid result: {result_path}")
        output[task_id] = payload
    return output


def _formal_calibration_passed(root: Path) -> bool:
    path = root / "calibration.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(payload.get("passed"))


def _require_formal_budget_approval(
    config_path: str | Path,
    *,
    config: JudgeEvaluationConfig,
    root: Path,
) -> dict[str, Any]:
    """Require an exact config/manifest-bound operator approval for non-smoke calls."""

    path = Path(config.budget.formal_budget_approval_file)
    if not path.is_file() or path.is_symlink():
        raise JudgeWorkflowError(
            "formal remote judging is locked pending explicit budget approval"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = root / "manifest.json"
    expected = {
        "approved": True,
        "experiment_name": config.experiment_name,
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest_path),
        "approved_max_cost_usd": config.budget.max_experiment_cost_usd,
    }
    if not isinstance(payload, Mapping) or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise JudgeWorkflowError(
            "formal budget approval does not match the frozen config and manifest"
        )
    mode = path.stat().st_mode & 0o777
    if mode != 0o600:
        raise JudgeWorkflowError(
            f"formal budget approval must have mode 0600, observed {mode:04o}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "approved_max_cost_usd": payload["approved_max_cost_usd"],
    }


@contextmanager
def _exclusive_budget_lock(root: Path) -> Iterator[None]:
    """Prevent concurrent invocations from racing the same experiment budget."""

    budget_dir = root / "budget"
    if budget_dir.exists() and (not budget_dir.is_dir() or budget_dir.is_symlink()):
        raise JudgeWorkflowError(f"unsafe budget directory: {budget_dir}")
    budget_dir.mkdir(parents=True, exist_ok=True)
    budget_dir.chmod(0o700)
    lock_path = budget_dir / "run.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        lock_path.chmod(0o600)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise JudgeWorkflowError(
                "another judge invocation holds the experiment budget lock"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JudgeWorkflowError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise JudgeWorkflowError(f"{label} must be a finite number")
    return converted


def _prompt_token_upper_bound(task: Mapping[str, Any]) -> int:
    """Conservatively bound input tokens by serialized UTF-8 bytes plus overhead."""

    task_type = str(task["task_type"])
    request_context = {
        "messages": render_messages(task),
        "response_format": response_schema(task_type),
    }
    encoded = json.dumps(
        request_context,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return len(encoded) + 512


def _recorded_experiment_cost_usd(
    root: Path,
    *,
    config: JudgeEvaluationConfig,
) -> float:
    """Sum provider-reported cost, falling back to registered token prices."""

    total = 0.0
    for top in ("responses", "smoke_responses"):
        response_root = root / top
        if not response_root.is_dir():
            continue
        for result_path in sorted(response_root.rglob("*.json")):
            if result_path.name == "index.json":
                continue
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            usage = payload.get("usage")
            if not isinstance(usage, Mapping):
                raise JudgeWorkflowError(f"result lacks usage accounting: {result_path}")
            reported = usage.get("cost")
            if isinstance(reported, (int, float)) and not isinstance(reported, bool):
                cost = _finite_float(reported, label=f"reported cost in {result_path}")
                if cost < 0:
                    raise JudgeWorkflowError(f"negative reported cost in {result_path}")
                total += cost
                continue
            model = str(payload.get("requested_model", ""))
            input_rate = config.pricing.input_per_million.get(model)
            output_rate = config.pricing.output_per_million.get(model)
            if input_rate is None or output_rate is None:
                raise JudgeWorkflowError(f"result uses unregistered pricing model: {result_path}")
            prompt_tokens = _finite_float(
                usage.get("prompt_tokens", 0),
                label=f"prompt_tokens in {result_path}",
            )
            completion_tokens = _finite_float(
                usage.get("completion_tokens", 0),
                label=f"completion_tokens in {result_path}",
            )
            if prompt_tokens < 0 or completion_tokens < 0:
                raise JudgeWorkflowError(f"negative token usage in {result_path}")
            total += (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000.0
    return total


def _budget_preflight(
    *,
    config: JudgeEvaluationConfig,
    root: Path,
    client: OpenRouterClient,
    model: str,
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail before task transmission unless every registered budget bound passes."""

    bounds = [_prompt_token_upper_bound(task) for task in tasks]
    request_cap = config.budget.max_estimated_prompt_tokens_per_request
    oversized = [
        str(task["task_id"])
        for task, bound in zip(tasks, bounds, strict=True)
        if bound > request_cap
    ]
    if oversized:
        raise JudgeWorkflowError(
            f"{len(oversized)} task(s) exceed the prompt-token upper bound; "
            f"first task: {oversized[0]}"
        )
    attempts = config.openrouter.max_retries + 1
    estimated_total_tokens = (sum(bounds) + len(tasks) * config.openrouter.max_tokens) * attempts
    if estimated_total_tokens > config.budget.max_estimated_total_tokens_per_invocation:
        raise JudgeWorkflowError(
            "invocation exceeds max_estimated_total_tokens_per_invocation: "
            f"{estimated_total_tokens} > "
            f"{config.budget.max_estimated_total_tokens_per_invocation}"
        )
    input_rate = config.pricing.input_per_million[model]
    output_rate = config.pricing.output_per_million[model]
    raw_cost = (
        sum(bounds) * input_rate + len(tasks) * config.openrouter.max_tokens * output_rate
    ) / 1_000_000.0
    planned_cost = raw_cost * attempts * config.budget.pricing_safety_multiplier
    if planned_cost > config.budget.max_cost_usd_per_invocation:
        raise JudgeWorkflowError(
            "invocation exceeds max_cost_usd_per_invocation: "
            f"${planned_cost:.6f} > ${config.budget.max_cost_usd_per_invocation:.6f}"
        )

    key_status = client.key_status()
    remaining = _finite_float(
        key_status.get("limit_remaining"),
        label="OpenRouter limit_remaining",
    )
    if remaining < 0:
        raise JudgeWorkflowError("OpenRouter limit_remaining cannot be negative")
    ledger_path = root / "budget" / "ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if (
            not isinstance(ledger, Mapping)
            or ledger.get("experiment_name") != config.experiment_name
        ):
            raise JudgeWorkflowError(f"invalid experiment budget ledger: {ledger_path}")
        baseline = _finite_float(
            ledger.get("baseline_limit_remaining_usd"),
            label="budget ledger baseline",
        )
    else:
        baseline = remaining
        atomic_write_json(
            ledger_path,
            {
                "schema_version": 1,
                "experiment_name": config.experiment_name,
                "baseline_limit_remaining_usd": baseline,
            },
        )
        ledger_path.chmod(0o600)

    recorded_cost = _recorded_experiment_cost_usd(root, config=config)
    account_spend_since_baseline = max(0.0, baseline - remaining)
    observed_experiment_spend = max(recorded_cost, account_spend_since_baseline)
    projected_experiment_spend = observed_experiment_spend + planned_cost
    if projected_experiment_spend > config.budget.max_experiment_cost_usd:
        raise JudgeWorkflowError(
            "invocation would exceed max_experiment_cost_usd: "
            f"${projected_experiment_spend:.6f} > "
            f"${config.budget.max_experiment_cost_usd:.6f}"
        )
    projected_remaining = remaining - planned_cost
    if projected_remaining < config.budget.reserve_openrouter_credit_usd:
        raise JudgeWorkflowError(
            "invocation would breach reserve_openrouter_credit_usd: "
            f"${projected_remaining:.6f} < "
            f"${config.budget.reserve_openrouter_credit_usd:.6f}"
        )
    return {
        "live_credit_check_passed": True,
        "account_limit_remaining_before_usd": remaining,
        "account_limit_remaining_projected_floor_usd": projected_remaining,
        "experiment_baseline_limit_remaining_usd": baseline,
        "recorded_experiment_cost_before_usd": recorded_cost,
        "observed_experiment_spend_before_usd": observed_experiment_spend,
        "planned_prompt_token_upper_bound": sum(bounds) * attempts,
        "planned_completion_token_limit": (len(tasks) * config.openrouter.max_tokens * attempts),
        "planned_total_token_upper_bound": estimated_total_tokens,
        "planned_cost_upper_bound_usd": planned_cost,
        "automatic_attempts_per_task": attempts,
        "pricing_safety_multiplier": config.budget.pricing_safety_multiplier,
        "reserve_openrouter_credit_usd": config.budget.reserve_openrouter_credit_usd,
        "max_experiment_cost_usd": config.budget.max_experiment_cost_usd,
    }


def _idle_budget_report(
    root: Path,
    *,
    config: JudgeEvaluationConfig,
) -> dict[str, Any]:
    return {
        "live_credit_check_passed": None,
        "reason": "no pending tasks; no remote request attempted",
        "recorded_experiment_cost_before_usd": _recorded_experiment_cost_usd(
            root,
            config=config,
        ),
        "planned_prompt_token_upper_bound": 0,
        "planned_completion_token_limit": 0,
        "planned_total_token_upper_bound": 0,
        "planned_cost_upper_bound_usd": 0.0,
    }


def run_judge_tasks(
    config_path: str | Path,
    *,
    role: Literal["primary", "secondary", "arbitration", "expert_review"],
    task_set: Literal["pointwise", "pairwise", "arbitration", "expert_review"],
    split: Literal["validation", "test"] | None = None,
    limit: int | None = None,
    smoke: bool = False,
    jobs: int | None = None,
    client: OpenRouterClient | None = None,
) -> dict[str, Any]:
    """Run or resume one budgeted batch while holding the experiment lock."""

    config = _load_runtime_config(config_path)
    root = Path(config.output_dir)
    if not (root / "manifest.json").is_file():
        raise JudgeWorkflowError("prepare the registered judge tasks before running")
    with _exclusive_budget_lock(root):
        return _run_judge_tasks_locked(
            config_path,
            role=role,
            task_set=task_set,
            split=split,
            limit=limit,
            smoke=smoke,
            jobs=jobs,
            client=client,
        )


def _run_judge_tasks_locked(
    config_path: str | Path,
    *,
    role: Literal["primary", "secondary", "arbitration", "expert_review"],
    task_set: Literal["pointwise", "pairwise", "arbitration", "expert_review"],
    split: Literal["validation", "test"] | None = None,
    limit: int | None = None,
    smoke: bool = False,
    jobs: int | None = None,
    client: OpenRouterClient | None = None,
) -> dict[str, Any]:
    """Run exact-model tasks with one atomic response file per task and safe resume."""

    config = _load_runtime_config(config_path)
    root = Path(config.output_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise JudgeWorkflowError("prepare the registered judge tasks before running")
    study_design = getattr(config, "study_design_version", None)
    registered_study = study_design in {
        "three_method_llm_judge_v1",
        "three_method_pointwise_judge_v2",
    }
    pointwise_only = study_design == "three_method_pointwise_judge_v2"
    if registered_study and smoke and (
        task_set != "pointwise"
        or split != "validation"
        or role not in {"primary", "secondary"}
    ):
        raise JudgeWorkflowError(
            "authorized smoke scope is one validation pointwise task per base judge"
        )
    approval = (
        _require_formal_budget_approval(config_path, config=config, root=root)
        if registered_study and not smoke
        else None
    )
    protocol_amendment = (
        validate_format_amendment(
            config_path,
            root=root,
            experiment_name=config.experiment_name,
            verify_response_seals=False,
        )
        if study_design == "three_method_llm_judge_v1" and not smoke
        else None
    )
    if pointwise_only and (
        task_set != "pointwise"
        or split not in {"validation", "test"}
        or role not in {"primary", "secondary"}
    ):
        raise JudgeWorkflowError("pointwise v2 rejects pairwise and review tasks")
    if task_set == "arbitration" and role != "arbitration":
        raise JudgeWorkflowError("arbitration tasks require the arbitration model role")
    if task_set == "expert_review" and role != "expert_review":
        raise JudgeWorkflowError("expert_review tasks require the expert_review role")
    if task_set in {"pointwise", "pairwise"} and role not in {"primary", "secondary"}:
        raise JudgeWorkflowError("base tasks require primary or secondary role")
    if split == "test" and not smoke and task_set in {"pointwise", "pairwise"}:
        if not _formal_calibration_passed(root):
            raise JudgeWorkflowError(
                "formal test judging is locked until validation calibration passes"
            )
    task_path = _task_file(root, task_set=task_set, split=split)
    tasks = _read_jsonl(task_path)
    if limit is None:
        raise JudgeWorkflowError("budgeted judge runs require an explicit --limit")
    if limit < 1:
        raise JudgeWorkflowError("limit must be positive")
    if limit > config.budget.max_tasks_per_invocation:
        raise JudgeWorkflowError(
            f"limit exceeds max_tasks_per_invocation: "
            f"{limit} > {config.budget.max_tasks_per_invocation}"
        )
    if smoke and limit > config.budget.max_smoke_tasks_per_invocation:
        raise JudgeWorkflowError(
            f"smoke limit exceeds max_smoke_tasks_per_invocation: "
            f"{limit} > {config.budget.max_smoke_tasks_per_invocation}"
        )
    model = config.judges.for_role(role)
    destination = _result_directory(root, model=model, task_set=task_set, split=split, smoke=smoke)
    destination.mkdir(parents=True, exist_ok=True)
    existing = _load_results(destination)
    registered_task_ids = {str(task["task_id"]) for task in tasks}
    runtime_exclusion_path = root / "private" / "expert_review_runtime_exclusions.jsonl"
    runtime_exclusions: dict[str, dict[str, Any]] = {}
    if task_set == "expert_review" and not smoke and runtime_exclusion_path.is_file():
        runtime_rows = _read_jsonl(runtime_exclusion_path)
        for row in runtime_rows:
            task_id = str(row.get("task_id", ""))
            if (
                task_id not in registered_task_ids
                or task_id in runtime_exclusions
                or row.get("reason") != "provider_content_filter_across_registered_routes"
            ):
                raise JudgeWorkflowError(
                    f"invalid expert runtime exclusion: {runtime_exclusion_path}"
                )
            runtime_exclusions[task_id] = dict(row)
    pending_all = [
        task
        for task in tasks
        if task["task_id"] not in existing and task["task_id"] not in runtime_exclusions
    ]
    pending = pending_all[:limit]
    judge_client = client or OpenRouterClient(config.openrouter)
    budget_report = (
        _budget_preflight(
            config=config,
            root=root,
            client=judge_client,
            model=model,
            tasks=pending,
        )
        if pending
        else _idle_budget_report(root, config=config)
    )

    def execute(task: Mapping[str, Any]) -> dict[str, Any]:
        response = judge_client.judge(model=model, task=task)
        payload = {
            "schema_version": 1,
            "protocol_version": config.protocol_version,
            "rubric_version": PROMPT_VERSION,
            "rubric_sha256": rubric_hash(),
            "protocol_amendment": protocol_amendment,
            "task_id": task["task_id"],
            "task_sha256": _canonical_hash(task),
            "task_type": task["task_type"],
            "split": task["split"],
            "judge_role": role,
            **asdict(response),
        }
        atomic_write_json(destination / f"{task['task_id']}.json", payload)
        return payload

    failure_destination = root / (
        "smoke_errors" if smoke else "errors"
    ) / _model_directory(model) / (
        task_set if split is None else f"{task_set}_{split}"
    )
    failures: list[dict[str, Any]] = []
    provider_filtered: list[dict[str, Any]] = []
    worker_count = jobs or config.openrouter.concurrency
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(execute, task): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
            except Exception as error:
                failure = {
                    "task_id": str(task["task_id"]),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "attempt_records": list(
                        getattr(error, "attempt_records", ())
                    ),
                }
                failure_event = {
                    "schema_version": 1,
                    "experiment_name": config.experiment_name,
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "task_sha256": _canonical_hash(task),
                    "task_type": task["task_type"],
                    "split": task["split"],
                    "judge_role": role,
                    "requested_model": model,
                    **failure,
                }
                failure_destination.mkdir(parents=True, exist_ok=True)
                atomic_write_json(
                    failure_destination / f"{task['task_id']}.json",
                    failure_event,
                )
                if (
                    config.review.expert_provider_filter_escalation
                    and task_set == "expert_review"
                    and not smoke
                    and isinstance(error, OpenRouterContentFilterError)
                ):
                    provider_filtered.append(failure)
                else:
                    failures.append(failure)
    if provider_filtered:
        provider_order = config.openrouter.provider_order_by_model[model]
        for failure in provider_filtered:
            task_id = failure["task_id"]
            runtime_exclusions[task_id] = {
                "task_id": task_id,
                "reason": "provider_content_filter_across_registered_routes",
                "provider_order": provider_order,
                "attempts": config.openrouter.max_retries + 1,
            }
        atomic_write_jsonl(
            runtime_exclusion_path,
            [runtime_exclusions[task_id] for task_id in sorted(runtime_exclusions)],
        )
        runtime_exclusion_path.chmod(0o600)
    observed = _load_results(destination)
    expected_ids = registered_task_ids
    invocation_ids = {str(task["task_id"]) for task in pending}
    machine_completed_ids = expected_ids.intersection(observed)
    resolved_ids = machine_completed_ids.union(runtime_exclusions)
    completed = len(resolved_ids)
    machine_completed = len(machine_completed_ids)
    invocation_completed = len(invocation_ids.intersection(observed))
    invocation_resolved = len(invocation_ids.intersection(resolved_ids))
    usage: dict[str, float] = defaultdict(float)
    completed_results = [observed[task_id] for task_id in machine_completed_ids]
    for result in completed_results:
        for key, value in result.get("usage", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[str(key)] += float(value)
    input_rate = config.pricing.input_per_million[model]
    output_rate = config.pricing.output_per_million[model]
    estimated_cost = (
        usage.get("prompt_tokens", 0.0) * input_rate
        + usage.get("completion_tokens", 0.0) * output_rate
    ) / 1_000_000.0
    all_report_cost = all(
        isinstance(result.get("usage", {}).get("cost"), (int, float))
        and not isinstance(result.get("usage", {}).get("cost"), bool)
        for result in completed_results
    )
    actual_or_estimated_cost = usage.get("cost", 0.0) if all_report_cost else estimated_cost
    invocation_succeeded = not failures and invocation_resolved == len(invocation_ids)
    dataset_complete = completed == len(expected_ids)
    budget_report = {
        **budget_report,
        "recorded_experiment_cost_after_usd": _recorded_experiment_cost_usd(
            root,
            config=config,
        ),
    }
    index = {
        "schema_version": 1,
        "complete": dataset_complete,
        "invocation_succeeded": invocation_succeeded,
        "smoke": smoke,
        "task_set": task_set,
        "split": split,
        "judge_role": role,
        "requested_model": model,
        "task_file": str(task_path),
        "task_file_sha256": sha256_file(task_path),
        "requested_tasks": len(expected_ids),
        "completed_tasks": completed,
        "machine_completed_tasks": machine_completed,
        "provider_filtered_tasks": len(runtime_exclusions),
        "provider_filtered_exclusions_path": (
            str(runtime_exclusion_path) if runtime_exclusions else None
        ),
        "provider_filtered_exclusions_sha256": (
            sha256_file(runtime_exclusion_path) if runtime_exclusions else None
        ),
        "remaining_tasks": len(expected_ids) - completed,
        "invocation_requested_tasks": len(invocation_ids),
        "invocation_completed_tasks": invocation_completed,
        "invocation_provider_filtered_tasks": len(
            invocation_ids.intersection(runtime_exclusions)
        ),
        "new_tasks": len(pending),
        "failures": failures,
        "usage": dict(usage),
        "provider_reported_or_estimated_cost_usd": actual_or_estimated_cost,
        "registered_price_estimate_usd": estimated_cost,
        "budget": budget_report,
        "formal_budget_approval": approval,
    }
    atomic_write_json(destination / "index.json", index)
    return index


def _normalized_pair_preference(
    result: Mapping[str, Any],
    private: Mapping[str, Any],
    field: Literal["target_preference", "quality_preference"] = "target_preference",
) -> str:
    preference = str(result["judgment"][field])
    if preference == "tie":
        return "tie"
    return str(private["response_a_role"] if preference == "A" else private["response_b_role"])


def _pairwise_consistency(
    results: Mapping[str, Mapping[str, Any]],
    private_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[float, dict[str, str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for task_id, private in private_by_id.items():
        if task_id not in results:
            continue
        grouped[str(private["pair_group_id"])].append(
            _normalized_pair_preference(results[task_id], private)
        )
    stable: dict[str, str] = {}
    for group, values in grouped.items():
        if len(values) == 2 and values[0] == values[1]:
            stable[group] = values[0]
    consistency = len(stable) / len(grouped) if grouped else 0.0
    return consistency, stable


def _calibrate_pointwise_only(
    config: JudgeEvaluationConfig,
    *,
    root: Path,
) -> dict[str, Any]:
    """Apply only the preregistered pointwise gates for the v2 adaptation."""

    models = {
        "primary": config.judges.primary,
        "secondary": config.judges.secondary,
    }
    tasks = _read_jsonl(root / "tasks" / "pointwise_validation.jsonl")
    task_ids = {str(task["task_id"]) for task in tasks}
    results = {
        role: _load_results(
            _result_directory(
                root,
                model=model,
                task_set="pointwise",
                split="validation",
            )
        )
        for role, model in models.items()
    }
    completion = {
        role: len(task_ids.intersection(results[role])) / len(task_ids)
        for role in models
    }
    common = sorted(task_ids.intersection(results["primary"], results["secondary"]))
    if not common:
        raise JudgeWorkflowError("pointwise calibration requires completed judgments")
    pointwise = agreement_metrics(
        [
            float(results["primary"][task]["judgment"]["target_expression"])
            for task in common
        ],
        [
            float(results["secondary"][task]["judgment"]["target_expression"])
            for task in common
        ],
    )
    human: dict[str, Any] | None = None
    if config.calibration.human_annotations_path:
        human_path = Path(config.calibration.human_annotations_path)
        annotations = {str(row["task_id"]): row for row in _read_jsonl(human_path)}
        common_human = sorted(
            task_ids.intersection(
                annotations, results["primary"], results["secondary"]
            )
        )
        consensus = [
            (
                float(results["primary"][task]["judgment"]["target_expression"])
                + float(results["secondary"][task]["judgment"]["target_expression"])
            )
            / 2.0
            for task in common_human
        ]
        human = agreement_metrics(
            consensus,
            [float(annotations[task]["target_expression"]) for task in common_human],
        )
        human["path"] = str(human_path)
        human["sha256"] = sha256_file(human_path)
    gates = {
        "completion": min(completion.values())
        >= config.calibration.min_completion_rate,
        "pointwise_spearman": pointwise["spearman"]
        >= config.calibration.min_pointwise_spearman,
        "pointwise_mae": pointwise["mae"]
        <= config.calibration.max_pointwise_mae,
        "human": (
            human is not None
            and human["spearman"] >= config.calibration.min_human_spearman
        )
        if config.calibration.require_human_gate
        else True,
    }
    result = {
        "schema_version": 1,
        "protocol_version": config.protocol_version,
        "study_design_version": config.study_design_version,
        "passed": all(gates.values()),
        "gates": gates,
        "not_applicable_gates": [
            "pairwise_order_consistency",
            "pairwise_cross_judge_agreement",
        ],
        "completion_rate": completion,
        "pointwise_target_expression": pointwise,
        "pairwise_order_consistency": None,
        "pairwise_cross_judge_agreement": None,
        "pairwise_common_stable_groups": 0,
        "human_agreement": human,
        "response_reuse": False,
        "test_unlock_policy": "all registered pointwise gates must pass",
    }
    atomic_write_json(root / "calibration.json", result)
    return result


def calibrate_judges(config_path: str | Path) -> dict[str, Any]:
    """Gate test access on validation completion, inter-judge, and order agreement."""

    config = _load_runtime_config(config_path)
    root = Path(config.output_dir)
    if getattr(config, "study_design_version", None) == "three_method_pointwise_judge_v2":
        return _calibrate_pointwise_only(config, root=root)
    models = {
        "primary": config.judges.primary,
        "secondary": config.judges.secondary,
    }
    point_tasks = _read_jsonl(root / "tasks" / "pointwise_validation.jsonl")
    pair_tasks = _read_jsonl(root / "tasks" / "pairwise_validation.jsonl")
    point_results = {
        role: _load_results(
            _result_directory(
                root,
                model=model,
                task_set="pointwise",
                split="validation",
            )
        )
        for role, model in models.items()
    }
    pair_results = {
        role: _load_results(
            _result_directory(
                root,
                model=model,
                task_set="pairwise",
                split="validation",
            )
        )
        for role, model in models.items()
    }
    point_ids = {str(task["task_id"]) for task in point_tasks}
    pair_ids = {str(task["task_id"]) for task in pair_tasks}
    expected_total = len(point_ids) + len(pair_ids)
    completion = {
        role: (
            len(point_ids.intersection(point_results[role]))
            + len(pair_ids.intersection(pair_results[role]))
        )
        / expected_total
        for role in models
    }
    common_point = sorted(
        point_ids.intersection(point_results["primary"], point_results["secondary"])
    )
    pointwise = agreement_metrics(
        [
            float(point_results["primary"][task]["judgment"]["target_expression"])
            for task in common_point
        ],
        [
            float(point_results["secondary"][task]["judgment"]["target_expression"])
            for task in common_point
        ],
    )
    private_rows = _read_jsonl(root / "private" / "pairwise_validation_map.jsonl")
    private_by_id = {str(row["task_id"]): row for row in private_rows}
    pair_consistency: dict[str, float] = {}
    stable: dict[str, dict[str, str]] = {}
    for role in models:
        pair_consistency[role], stable[role] = _pairwise_consistency(
            pair_results[role], private_by_id
        )
    common_groups = set(stable["primary"]).intersection(stable["secondary"])
    cross_judge = (
        float(
            np.mean(
                [
                    stable["primary"][group] == stable["secondary"][group]
                    for group in sorted(common_groups)
                ]
            )
        )
        if common_groups
        else 0.0
    )
    human: dict[str, Any] | None = None
    if config.calibration.human_annotations_path:
        human_path = Path(config.calibration.human_annotations_path)
        annotations = {str(row["task_id"]): row for row in _read_jsonl(human_path)}
        common_human = sorted(
            point_ids.intersection(
                annotations, point_results["primary"], point_results["secondary"]
            )
        )
        consensus = [
            (
                float(point_results["primary"][task]["judgment"]["target_expression"])
                + float(point_results["secondary"][task]["judgment"]["target_expression"])
            )
            / 2.0
            for task in common_human
        ]
        human = agreement_metrics(
            consensus,
            [float(annotations[task]["target_expression"]) for task in common_human],
        )
        human["path"] = str(human_path)
        human["sha256"] = sha256_file(human_path)
    gates = {
        "completion": min(completion.values()) >= config.calibration.min_completion_rate,
        "pointwise_spearman": pointwise["spearman"] >= config.calibration.min_pointwise_spearman,
        "pointwise_mae": pointwise["mae"] <= config.calibration.max_pointwise_mae,
        "pairwise_order_consistency": min(pair_consistency.values())
        >= config.calibration.min_pairwise_order_consistency,
        "pairwise_cross_judge_agreement": cross_judge
        >= config.calibration.min_pairwise_cross_judge_agreement,
        "human": (human is not None and human["spearman"] >= config.calibration.min_human_spearman)
        if config.calibration.require_human_gate
        else True,
    }
    result = {
        "schema_version": 1,
        "protocol_version": config.protocol_version,
        "passed": all(gates.values()),
        "gates": gates,
        "completion_rate": completion,
        "pointwise_target_expression": pointwise,
        "pairwise_order_consistency": pair_consistency,
        "pairwise_cross_judge_agreement": cross_judge,
        "pairwise_common_stable_groups": len(common_groups),
        "human_agreement": human,
        "test_unlock_policy": "all registered gates must pass",
    }
    atomic_write_json(root / "calibration.json", result)
    return result


def prepare_review_tasks(
    config_path: str | Path,
    *,
    tier: Literal["arbitration", "expert_review"],
) -> dict[str, Any]:
    """Freeze disagreement/audit tasks without changing the primary two-judge score."""

    config = _load_runtime_config(config_path)
    root = Path(config.output_dir)
    base_tasks = {
        task["task_id"]: task
        for kind in ("pointwise", "pairwise")
        for task in _read_jsonl(root / "tasks" / f"{kind}_test.jsonl")
    }
    primary: dict[str, dict[str, Any]] = {}
    secondary: dict[str, dict[str, Any]] = {}
    for kind in ("pointwise", "pairwise"):
        primary.update(
            _load_results(
                _result_directory(
                    root,
                    model=config.judges.primary,
                    task_set=kind,
                    split="test",
                )
            )
        )
        secondary.update(
            _load_results(
                _result_directory(
                    root,
                    model=config.judges.secondary,
                    task_set=kind,
                    split="test",
                )
            )
        )
    common = set(base_tasks).intersection(primary, secondary)
    if common != set(base_tasks):
        raise JudgeWorkflowError(
            "review preparation requires complete primary and secondary test results"
        )
    reasons: dict[str, list[str]] = defaultdict(list)
    for task_id in common:
        task = base_tasks[task_id]
        if task["task_type"] == "pointwise":
            difference = abs(
                int(primary[task_id]["judgment"]["target_expression"])
                - int(secondary[task_id]["judgment"]["target_expression"])
            )
            if difference >= config.review.pointwise_disagreement_threshold:
                reasons[task_id].append(f"pointwise_difference_{difference}")
        elif (
            primary[task_id]["judgment"]["target_preference"]
            != secondary[task_id]["judgment"]["target_preference"]
        ):
            reasons[task_id].append("pairwise_cross_judge_disagreement")
    pair_private = {
        str(row["task_id"]): row
        for row in _read_jsonl(root / "private" / "pairwise_test_map.jsonl")
    }
    pair_groups: dict[str, list[str]] = defaultdict(list)
    for task_id, private in pair_private.items():
        pair_groups[str(private["pair_group_id"])].append(task_id)
    for role, role_results in (("primary", primary), ("secondary", secondary)):
        for task_ids in pair_groups.values():
            preferences = [
                _normalized_pair_preference(role_results[task_id], pair_private[task_id])
                for task_id in task_ids
            ]
            if len(preferences) != 2 or preferences[0] != preferences[1]:
                for task_id in task_ids:
                    reasons[task_id].append(f"pairwise_{role}_order_inconsistency")
    if tier == "expert_review":
        # Expert review is the second tier: include only cases that remain
        # discordant after arbitration plus the preregistered random audit.
        # Carrying first-tier reasons forward would review every arbitration case twice.
        reasons = defaultdict(list)
        arbitration_tasks = _read_jsonl(root / "tasks" / "arbitration.jsonl")
        arbitration = _load_results(
            _result_directory(
                root,
                model=config.judges.arbitration,
                task_set="arbitration",
                split=None,
            )
        )
        if set(arbitration) != {str(task["task_id"]) for task in arbitration_tasks}:
            raise JudgeWorkflowError(
                "expert review requires complete registered arbitration results"
            )
        for task_id, result in arbitration.items():
            if task_id not in common:
                continue
            task = base_tasks[task_id]
            if task["task_type"] == "pointwise":
                midpoint = (
                    int(primary[task_id]["judgment"]["target_expression"])
                    + int(secondary[task_id]["judgment"]["target_expression"])
                ) / 2.0
                difference = abs(int(result["judgment"]["target_expression"]) - midpoint)
                if difference >= config.review.expert_disagreement_threshold:
                    reasons[task_id].append("arbitration_remains_discordant")
            elif result["judgment"]["target_preference"] not in {
                primary[task_id]["judgment"]["target_preference"],
                secondary[task_id]["judgment"]["target_preference"],
            }:
                reasons[task_id].append("arbitration_third_preference")
        audit_limit = round(len(base_tasks) * config.review.expert_audit_fraction)
        audit_ids = sorted(
            base_tasks,
            key=lambda task_id: _canonical_hash(
                {"seed": config.seed, "tier": "expert_audit", "task_id": task_id}
            ),
        )[:audit_limit]
        for task_id in audit_ids:
            reasons[task_id].append("registered_random_audit")
    excluded_ids: list[str] = []
    exclusion_path: Path | None = None
    if tier == "expert_review":
        registered_ids = set(config.review.expert_provider_filtered_task_ids)
        unexpected_ids = registered_ids - set(reasons)
        if unexpected_ids:
            raise JudgeWorkflowError(
                "registered provider-filtered tasks are not selected for expert review: "
                + ", ".join(sorted(unexpected_ids))
            )
        excluded_ids = sorted(registered_ids)
        for task_id in excluded_ids:
            reasons.pop(task_id)
        exclusion_path = root / "private" / "expert_review_exclusions.jsonl"
        _registered_write_jsonl(
            exclusion_path,
            [
                {
                    "task_id": task_id,
                    "reason": "provider_content_filter_across_registered_routes",
                }
                for task_id in excluded_ids
            ],
        )
        exclusion_path.chmod(0o600)
    selected_ids = sorted(reasons)
    tasks = [base_tasks[task_id] for task_id in selected_ids]
    task_path = root / "tasks" / f"{tier}.jsonl"
    private_path = root / "private" / f"{tier}_map.jsonl"
    _registered_write_jsonl(task_path, tasks)
    _registered_write_jsonl(
        private_path,
        [{"task_id": task_id, "review_reasons": reasons[task_id]} for task_id in selected_ids],
    )
    private_path.chmod(0o600)
    result = {
        "schema_version": 1,
        "tier": tier,
        "tasks": len(tasks),
        "task_path": str(task_path),
        "task_sha256": sha256_file(task_path),
        "private_map": str(private_path),
        "private_map_sha256": sha256_file(private_path),
        "changes_primary_estimator": False,
    }
    if tier == "expert_review":
        assert exclusion_path is not None
        result.update(
            {
                "excluded_tasks": len(excluded_ids),
                "exclusions_path": str(exclusion_path),
                "exclusions_sha256": sha256_file(exclusion_path),
            }
        )
    atomic_write_json(root / f"{tier}_manifest.json", result)
    return result


def export_human_audit(config_path: str | Path) -> dict[str, Any]:
    """Export a deterministic method-blind test subset and empty annotation template."""

    config = _load_runtime_config(config_path)
    root = Path(config.output_dir)
    task_sets = getattr(config, "task_sets", ["pointwise", "pairwise"])
    tasks = [
        task
        for kind in task_sets
        for task in _read_jsonl(root / "tasks" / f"{kind}_test.jsonl")
    ]
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_stratum[(str(task["concept_id"]), str(task["task_type"]))].append(task)
    selected: list[dict[str, Any]] = []
    if task_sets == ["pointwise"]:
        strata = (("pointwise", config.review.human_audit_per_concept),)
    else:
        strata = (
            ("pointwise", (config.review.human_audit_per_concept + 1) // 2),
            ("pairwise", config.review.human_audit_per_concept // 2),
        )
    for concept_id in config.source.concept_ids:
        for task_type, count in strata:
            ordered = sorted(
                by_stratum[(concept_id, task_type)],
                key=lambda task: _canonical_hash(
                    {"seed": config.seed, "human_audit": task["task_id"]}
                ),
            )
            if len(ordered) < count:
                raise JudgeWorkflowError(
                    f"human audit stratum is too small: {concept_id} {task_type}"
                )
            selected.extend(ordered[:count])
    task_by_id = {str(task["task_id"]): task for task in tasks}
    runtime_exclusion_path = root / "private" / "expert_review_runtime_exclusions.jsonl"
    runtime_ids: set[str] = set()
    if runtime_exclusion_path.is_file():
        for row in _read_jsonl(runtime_exclusion_path):
            if row.get("reason") != "provider_content_filter_across_registered_routes":
                raise JudgeWorkflowError(f"invalid runtime exclusion: {runtime_exclusion_path}")
            runtime_ids.add(str(row.get("task_id", "")))
    forced_ids = sorted(
        set(config.review.expert_provider_filtered_task_ids).union(runtime_ids)
    )
    missing_ids = set(forced_ids) - set(task_by_id)
    if missing_ids:
        raise JudgeWorkflowError(
            "provider-filtered expert tasks are absent from the test task set: "
            + ", ".join(sorted(missing_ids))
        )
    selected_ids = {str(task["task_id"]) for task in selected}
    selected.extend(task_by_id[task_id] for task_id in forced_ids if task_id not in selected_ids)
    task_path = root / "human_audit" / "tasks.jsonl"
    template_path = root / "human_audit" / "annotations_template.jsonl"
    _registered_write_jsonl(task_path, selected)
    template = []
    for task in selected:
        if task["task_type"] == "pointwise":
            fields = {
                "target_expression": None,
                "coherence": None,
                "relevance": None,
                "refusal": None,
            }
        else:
            fields = {
                "target_preference": None,
                "quality_preference": None,
                "confidence": None,
            }
        template.append({"task_id": task["task_id"], **fields, "annotator_id": None})
    _registered_write_jsonl(template_path, template)
    result = {
        "schema_version": 1,
        "method_blind": True,
        "task_count": len(selected),
        "provider_filtered_expert_tasks_forced": len(forced_ids),
        "tasks": str(task_path),
        "tasks_sha256": sha256_file(task_path),
        "annotations_template": str(template_path),
        "annotations_template_sha256": sha256_file(template_path),
    }
    atomic_write_json(root / "human_audit" / "manifest.json", result)
    return result


def aggregate_evaluation(config_path: str | Path) -> dict[str, Any]:
    """Delegate final inference to the aggregation module."""

    config = _load_runtime_config(config_path)
    study_design = getattr(config, "study_design_version", None)
    if study_design == "three_method_pointwise_judge_v2":
        from jlens_workspace.concept_intervention.judge.pointwise_v2_aggregation import (
            aggregate_evaluation as run,
        )
    elif study_design == "three_method_llm_judge_v1":
        from jlens_workspace.concept_intervention.judge.three_method_aggregation import (
            aggregate_evaluation as run,
        )
    else:
        from jlens_workspace.concept_intervention.judge.aggregation import (
            aggregate_evaluation as run,
        )

    return run(config_path)
