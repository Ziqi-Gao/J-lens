#!/usr/bin/env python3
"""Verify pinned Concept500 Parquet files and emit canonical JSONL splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from jlens_workspace.data import (
    Concept500PreparationError,
    load_concept500_allowlist,
    prepare_concept500_rows,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_parquet(path: Path) -> list[dict[str, object]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise Concept500PreparationError(
            "preparing Concept500 requires PyArrow; install the llm dependency extra"
        ) from error
    return pq.read_table(path).to_pylist()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    allowlist = load_concept500_allowlist(args.allowlist)
    observed_hashes = {"train": _sha256(args.train), "test": _sha256(args.test)}
    expected_hashes = {
        "train": allowlist.train_sha256,
        "test": allowlist.test_sha256,
    }
    if observed_hashes != expected_hashes:
        raise Concept500PreparationError(
            f"Parquet hash mismatch: expected {expected_hashes}, observed {observed_hashes}"
        )
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output already exists: {args.output}")
        import shutil

        shutil.rmtree(args.output)
    prepared = prepare_concept500_rows(
        _load_parquet(args.train),
        _load_parquet(args.test),
        allowlist,
        args.output,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    source_manifest = {
        "schema_version": 1,
        "dataset_id": allowlist.dataset_id,
        "revision": allowlist.revision,
        "variant": allowlist.variant,
        "license": allowlist.license,
        "source_files": {
            "train": {"path": str(args.train), "sha256": observed_hashes["train"]},
            "test": {"path": str(args.test), "sha256": observed_hashes["test"]},
        },
        "allowlist": str(args.allowlist),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "dataset_fingerprint": prepared.fingerprint,
    }
    (args.output / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "fingerprint": prepared.fingerprint,
        "statistics": prepared.statistics.to_dict(),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
