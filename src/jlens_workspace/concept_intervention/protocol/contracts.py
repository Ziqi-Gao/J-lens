"""Method-neutral prompt, label, batching, and JSONL protocol contracts.

This module is the single upstream owner of prompt formatting, candidate-token
validation, deterministic batching, and atomic JSONL output.  It must not
contain evaluation policy, method fitting, or intervention hooks.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from jlens_workspace.foundation.artifacts import resolve_repository_resource

ValueT = TypeVar("ValueT")


class CandidateEvaluationError(ValueError):
    """Raised when the shared prompt or candidate-label contract is invalid."""


@dataclass(frozen=True)
class PromptRecord:
    """One prompt rendered for the common candidate-label evaluation task."""

    prompt_id: str
    raw_text: str
    formatted_text: str
    label_order: tuple[str, ...]


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write a sequence of JSON objects as sorted JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_prompt_bank(
    path: str | Path,
    *,
    tokenizer: Any,
    candidate_labels: Mapping[str, str],
) -> list[PromptRecord]:
    """Load and format the common counterbalanced prompt-bank schema."""

    source = resolve_repository_resource(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise CandidateEvaluationError(f"unsupported prompt schema: {source}")
    concept_ids = tuple(candidate_labels)
    prompts: list[PromptRecord] = []
    seen: set[str] = set()
    for entry in payload.get("prompts", []):
        prompt_id = str(entry["prompt_id"])
        if prompt_id in seen:
            raise CandidateEvaluationError(f"duplicate prompt_id: {prompt_id}")
        seen.add(prompt_id)
        rotation = int(entry["label_rotation"])
        ordered = concept_ids[rotation:] + concept_ids[:rotation]
        labels = ", ".join(candidate_labels[concept_id] for concept_id in ordered)
        raw_text = str(entry["text"]).format(labels=labels)
        if getattr(tokenizer, "chat_template", None):
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            formatted = raw_text
        prompts.append(
            PromptRecord(
                prompt_id=prompt_id,
                raw_text=raw_text,
                formatted_text=formatted,
                label_order=ordered,
            )
        )
    if not prompts:
        raise CandidateEvaluationError("prompt bank must be non-empty")
    return prompts


def candidate_token_ids(
    tokenizer: Any, candidate_labels: Mapping[str, str]
) -> dict[str, int]:
    """Validate and return the unique one-token IDs used by both evaluations."""

    output: dict[str, int] = {}
    for concept_id, label in candidate_labels.items():
        token_ids = tokenizer.encode(f" {label}", add_special_tokens=False)
        if len(token_ids) != 1:
            raise CandidateEvaluationError(
                f"candidate label {concept_id!r}={label!r} must be one token "
                f"with a leading space, got {token_ids}"
            )
        output[concept_id] = int(token_ids[0])
    if len(set(output.values())) != len(output):
        raise CandidateEvaluationError("candidate labels must map to unique tokens")
    return output


def batched(values: Sequence[ValueT], size: int) -> list[Sequence[ValueT]]:
    """Partition a sequence into deterministic contiguous batches."""

    if size <= 0:
        raise ValueError("batch size must be positive")
    return [values[start : start + size] for start in range(0, len(values), size)]
