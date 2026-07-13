"""Strict JSONL input/output for concept examples."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from .schema import CANONICAL_SPLITS, ConceptExample
from .validation import DatasetStatistics, DatasetValidationError, validate_examples


def iter_jsonl(path: str | Path) -> Iterator[ConceptExample]:
    """Yield records from one UTF-8 JSONL file with path/line diagnostics."""

    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetValidationError(
                    (f"{source_path}:{line_number}: invalid JSON: {error.msg}",)
                ) from error
            try:
                yield ConceptExample.from_dict(raw)
            except (TypeError, ValueError) as error:
                raise DatasetValidationError(
                    (f"{source_path}:{line_number}: {error}",)
                ) from error


def load_jsonl(
    path: str | Path,
    *,
    validate: bool = True,
    expected_splits: Sequence[str] = CANONICAL_SPLITS,
    min_per_label_per_split: int = 1,
    min_per_concept: int = 1,
) -> list[ConceptExample]:
    records = list(iter_jsonl(path))
    if validate:
        validate_examples(
            records,
            expected_splits=expected_splits,
            min_per_label_per_split=min_per_label_per_split,
            min_per_concept=min_per_concept,
        )
    return records


def load_jsonl_directory(
    path: str | Path,
    *,
    pattern: str = "*.jsonl",
    expected_splits: Sequence[str] = CANONICAL_SPLITS,
    min_per_label_per_split: int = 1,
    min_per_concept: int = 1,
) -> list[ConceptExample]:
    """Load a prepared directory and validate the union of its JSONL files."""

    directory = Path(path)
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise DatasetValidationError(
                (f"{manifest_path}: invalid JSON: {error.msg}",)
            ) from error
        declared = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(declared, dict) or set(declared) != set(expected_splits):
            raise DatasetValidationError(
                (
                    f"{manifest_path}: files must declare exactly "
                    f"{list(expected_splits)}",
                )
            )
        files = []
        for split in expected_splits:
            filename = declared[split]
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise DatasetValidationError(
                    (f"{manifest_path}: unsafe file for split {split!r}: {filename!r}",)
                )
            files.append(directory / filename)
        missing = [str(file_path) for file_path in files if not file_path.is_file()]
        if missing:
            raise DatasetValidationError(
                (f"prepared dataset files do not exist: {missing}",)
            )
    else:
        files = sorted(directory.glob(pattern))
    if not files:
        raise DatasetValidationError((f"no files matching {pattern!r} in {directory}",))
    records = [record for file_path in files for record in iter_jsonl(file_path)]
    validate_examples(
        records,
        expected_splits=expected_splits,
        min_per_label_per_split=min_per_label_per_split,
        min_per_concept=min_per_concept,
    )
    return records


def write_jsonl(path: str | Path, examples: Iterable[ConceptExample]) -> None:
    """Atomically write records as stable, one-object-per-line JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            for example in examples:
                handle.write(
                    json.dumps(example.to_dict(), ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    temporary_path.replace(destination)


def write_statistics(path: str | Path, statistics: DatasetStatistics) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(statistics.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
