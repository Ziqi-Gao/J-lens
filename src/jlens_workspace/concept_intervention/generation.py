"""Method-neutral exhaustive generation and future blind-judge export."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from jlens_workspace.artifacts import sha256_file
from jlens_workspace.concept_intervention.evaluation import (
    PromptRecord,
    atomic_write_jsonl,
)
from jlens_workspace.modeling import model_input_device


class InterventionGenerationError(ValueError):
    """Raised when a generation grid violates the shared output contract."""


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

    source = Path(path)
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


def _telemetry(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        rows: list[dict[str, Any]] = []
        for layer, item in value.items():
            events = getattr(item, "events", [])
            rows.extend({"layer": int(layer), **dict(event)} for event in events)
        return rows
    events = getattr(value, "events", [])
    output = []
    for event in events:
        if is_dataclass(event):
            output.append(asdict(event))
        else:
            output.append(dict(event))
    return output


def _identifier(payload: Mapping[str, Any], *, prefix: str) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{prefix}\0{encoded}".encode()).hexdigest()


def _generate_one(
    *,
    model: Any,
    tokenizer: Any,
    prompt: PromptRecord | OpenPromptRecord,
    session_factory: Callable[[], AbstractContextManager[Any]],
    do_sample: bool,
    seed: int | None,
    settings: GenerationSettings,
) -> tuple[list[int], str, list[float], list[dict[str, Any]]]:
    import torch

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    encoded = tokenizer(
        prompt.formatted_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = {
        key: value.to(model_input_device(model)) for key, value in encoded.items()
    }
    kwargs: dict[str, Any] = {
        "max_new_tokens": settings.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": settings.repetition_penalty,
        "no_repeat_ngram_size": settings.no_repeat_ngram_size,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if do_sample:
        kwargs.update(temperature=settings.temperature, top_p=settings.top_p)
    with torch.inference_mode(), session_factory() as state:
        generated = model.generate(**encoded, **kwargs)
    prompt_length = int(encoded["input_ids"].shape[1])
    token_tensor = generated.sequences[0, prompt_length:]
    token_ids = [int(value) for value in token_tensor.detach().cpu().tolist()]
    log_probabilities: list[float] = []
    for step, scores in enumerate(generated.scores):
        log_probs = torch.log_softmax(scores[0].float(), dim=-1)
        log_probabilities.append(float(log_probs[token_tensor[step]].cpu()))
    return (
        token_ids,
        tokenizer.decode(token_ids, skip_special_tokens=True),
        log_probabilities,
        _telemetry(state),
    )


def generate_full_grid(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[PromptRecord | OpenPromptRecord],
    method: str,
    concept_id: str,
    condition_id: str,
    grid_point: Mapping[str, Any],
    intervention_metadata: Mapping[str, Any] | None = None,
    session_factory: Callable[[], AbstractContextManager[Any]],
    settings: GenerationSettings,
) -> list[dict[str, Any]]:
    """Generate greedy plus all fixed-seed samples for one complete grid point."""

    rows: list[dict[str, Any]] = []
    decodings = [("greedy", None), *[("sample", seed) for seed in settings.sample_seeds]]
    for prompt in prompts:
        for decoding, seed in decodings:
            identity = {
                "method": method,
                "concept_id": concept_id,
                "condition_id": condition_id,
                "grid_point": dict(grid_point),
                "intervention_metadata": dict(intervention_metadata or {}),
                "prompt_id": prompt.prompt_id,
                "decoding": decoding,
                "seed": seed,
            }
            token_ids, text, token_logprobs, telemetry = _generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                session_factory=session_factory,
                do_sample=(decoding == "sample"),
                seed=seed,
                settings=settings,
            )
            generation_id = _identifier(identity, prefix="generation")
            rows.append(
                {
                    "generation_id": generation_id,
                    "blind_id": _identifier(
                        {"generation_id": generation_id}, prefix="blind"
                    ),
                    **identity,
                    "prompt_split": (
                        prompt.split
                        if isinstance(prompt, OpenPromptRecord)
                        else (
                            "validation"
                            if prompt.prompt_id.partition("_")[0]
                            in {"choose", "complete"}
                            else "test"
                        )
                    ),
                    "prompt_text": prompt.raw_text,
                    "generated_token_ids": token_ids,
                    "generated_text": text,
                    "token_log_probabilities": token_logprobs,
                    "mean_token_log_probability": (
                        sum(token_logprobs) / len(token_logprobs)
                        if token_logprobs
                        else None
                    ),
                    "telemetry": telemetry,
                    "generation_settings": asdict(settings),
                }
            )
    return rows


def write_generation_artifacts(
    output_dir: str | Path, rows: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    """Write full and method-blind generation artifacts."""

    destination = Path(output_dir)
    full_path = destination / "generations.jsonl"
    blind_path = destination / "judge_blind_generations.jsonl"
    map_path = destination / "judge_blind_map.jsonl"
    atomic_write_jsonl(full_path, rows)
    blind_rows = [
        {
            "blind_id": row["blind_id"],
            "prompt_id": row["prompt_id"],
            "prompt_text": row["prompt_text"],
            "decoding": row["decoding"],
            "generated_text": row["generated_text"],
        }
        for row in rows
    ]
    mapping = [
        {
            "blind_id": row["blind_id"],
            "generation_id": row["generation_id"],
            "method": row["method"],
            "concept_id": row["concept_id"],
            "condition_id": row["condition_id"],
            "grid_point": row["grid_point"],
        }
        for row in rows
    ]
    atomic_write_jsonl(blind_path, blind_rows)
    atomic_write_jsonl(map_path, mapping)
    return {
        "generations": str(full_path),
        "generations_sha256": sha256_file(full_path),
        "blind_generations": str(blind_path),
        "blind_generations_sha256": sha256_file(blind_path),
        "blind_map": str(map_path),
        "blind_map_sha256": sha256_file(map_path),
    }


def validate_generation_artifacts(
    output_dir: str | Path, files: Mapping[str, Any]
) -> None:
    """Fail closed when a generation or blind-export shard changed."""

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
