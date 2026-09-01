"""Strict, dependency-light parsing helpers for scheduler-owned JSON."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Set
from datetime import datetime
from pathlib import Path

from jlens_workspace.scheduler.errors import ServerSchedulerAdapterError

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def strict_json_load(path: Path) -> object:
    """Load RFC-compatible JSON and reject non-finite numeric constants."""

    try:
        raw = path.read_text(encoding="utf-8")
        return json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ServerSchedulerAdapterError(f"cannot read strict JSON {path}: {error}") from error


def strict_object(payload: object, keys: Set[str], *, label: str) -> Mapping[str, object]:
    """Require exactly the registered fields in a JSON object."""

    if not isinstance(payload, Mapping) or set(payload) != set(keys):
        raise ServerSchedulerAdapterError(f"{label} has missing or unknown fields")
    return payload


def integer(
    value: object,
    *,
    label: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    """Validate an integer without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ServerSchedulerAdapterError(f"{label} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ServerSchedulerAdapterError(f"{label} is outside its allowed range")
    return value


def positive_number(value: object, *, label: str) -> float:
    """Validate a finite positive number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ServerSchedulerAdapterError(f"{label} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ServerSchedulerAdapterError(f"{label} must be a positive finite number")
    return result


def identifier(value: object, *, label: str) -> str:
    """Validate a scheduler-safe identifier."""

    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise ServerSchedulerAdapterError(f"{label} is not a valid identifier")
    return value


def timestamp(value: object, *, label: str) -> str:
    """Validate a timezone-aware ISO-8601 timestamp."""

    if not isinstance(value, str):
        raise ServerSchedulerAdapterError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ServerSchedulerAdapterError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ServerSchedulerAdapterError(f"{label} must include a timezone")
    return value


def string_tuple(value: object, *, label: str) -> tuple[str, ...]:
    """Validate a unique JSON array of non-empty strings."""

    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ServerSchedulerAdapterError(f"{label} must be an array of non-empty strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ServerSchedulerAdapterError(f"{label} contains duplicates")
    return result


def integer_tuple(value: object, *, label: str) -> tuple[int, ...]:
    """Validate a unique JSON array of non-negative integers."""

    if not isinstance(value, list):
        raise ServerSchedulerAdapterError(f"{label} must be an array of integers")
    result = tuple(integer(item, label=f"{label} item") for item in value)
    if len(set(result)) != len(result):
        raise ServerSchedulerAdapterError(f"{label} contains duplicates")
    return result
