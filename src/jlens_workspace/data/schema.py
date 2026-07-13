"""Canonical records for abstract-concept datasets.

The schema deliberately uses only the standard library.  Expensive model and
dataset dependencies are kept at the adapter boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

CANONICAL_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class ConceptExample:
    """One binary example for one explicitly abstract concept."""

    concept_id: str
    concept_name: str
    definition: str
    abstractness: str
    label: int
    text: str
    prompt: str
    response: str
    split: str
    group_id: str
    source: str
    license: str

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "concept_id",
        "concept_name",
        "definition",
        "abstractness",
        "label",
        "text",
        "prompt",
        "response",
        "split",
        "group_id",
        "source",
        "license",
    )

    def __post_init__(self) -> None:
        optional_text_fields = {"prompt", "response"}
        for field_name in self.FIELD_NAMES:
            if field_name == "label":
                continue
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
            if field_name not in optional_text_fields and not value.strip():
                raise ValueError(f"{field_name} must be non-empty")

        if isinstance(self.label, bool) or not isinstance(self.label, int):
            raise TypeError("label must be the integer 0 or 1 (booleans are not accepted)")
        if self.label not in (0, 1):
            raise ValueError(f"label must be binary (0 or 1), got {self.label!r}")
        if self.abstractness != "abstract":
            raise ValueError(
                "abstractness must be explicitly set to the exact string 'abstract'"
            )
        if self.split not in CANONICAL_SPLITS:
            raise ValueError(
                f"split must be one of {CANONICAL_SPLITS}, got {self.split!r}"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ConceptExample:
        """Parse a strict mapping, rejecting both missing and unknown fields."""

        if not isinstance(raw, Mapping):
            raise TypeError(f"record must be a mapping, got {type(raw).__name__}")
        supplied = set(raw)
        expected = set(cls.FIELD_NAMES)
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing fields: {', '.join(missing)}")
            if extra:
                details.append(f"unknown fields: {', '.join(extra)}")
            raise ValueError("; ".join(details))
        return cls(**{name: raw[name] for name in cls.FIELD_NAMES})

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-serializable record in schema order."""

        return asdict(self)
