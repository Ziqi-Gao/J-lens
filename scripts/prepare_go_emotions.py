#!/usr/bin/env python3
"""Verify pinned GoEmotions Parquet files and emit shared experiment inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from jlens_workspace.data import (
    CANONICAL_SPLITS,
    GoEmotionsPreparationError,
    load_go_emotions_allowlist,
    prepare_go_emotions,
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
        raise GoEmotionsPreparationError(
            "preparing GoEmotions requires PyArrow; install the llm dependency extra"
        ) from error
    return pq.read_table(path, columns=["text", "labels", "id"]).to_pylist()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowlist", type=Path, required=True)
    for split in CANONICAL_SPLITS:
        parser.add_argument(f"--{split}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    allowlist = load_go_emotions_allowlist(args.allowlist)
    source_paths = {split: getattr(args, split) for split in CANONICAL_SPLITS}
    observed_hashes = {split: _sha256(path) for split, path in source_paths.items()}
    expected_hashes = dict(allowlist.file_sha256)
    if observed_hashes != expected_hashes:
        raise GoEmotionsPreparationError(
            f"Parquet hash mismatch: expected {expected_hashes}, observed {observed_hashes}"
        )
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output already exists: {args.output}")
        shutil.rmtree(args.output)
    rows = {split: _load_parquet(path) for split, path in source_paths.items()}
    prepared = prepare_go_emotions(rows, allowlist, args.output, seed=args.seed)
    payload = {
        "output": str(args.output),
        "fingerprint": prepared.dataset.fingerprint,
        "fit_prompts": str(prepared.fit_prompts_path),
        "statistics": prepared.dataset.statistics.to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
