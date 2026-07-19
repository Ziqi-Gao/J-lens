#!/usr/bin/env python3
"""End-to-end tiny-model occupancy smoke: real lens, real streamed dictionary.

Fits a one-layer Jacobian lens on the pinned tiny GPT-2 checkpoint, builds the
UN-CENTERED token-frame dictionary (``raw`` convention only: tiny GPT-2 uses
LayerNorm, so ``rmsnorm_weighted`` must be exercised on Qwen), and runs the
full concept-occupancy workflow against synthetic probe targets, including the
matched-norm random controls and both crossing rules. This validates the
dependency/API path, not any scientific claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

MODEL_ID = "sshleifer/tiny-gpt2"
MODEL_REVISION = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke/occupancy"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jlens_workspace.jacobian import (
        JLensMetadata,
        OfficialJLensAdapter,
        build_effective_unembedding,
        restrict_effective_unembedding,
    )
    from jlens_workspace.pursuit import build_token_frame_dictionary
    from jlens_workspace.workflows.occupancy import (
        ConceptTarget,
        run_concept_occupancy,
    )

    args = parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {args.output}")
        shutil.rmtree(args.output)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model.to(args.device).eval()
    wrapped = OfficialJLensAdapter.from_hf(model, tokenizer, force_bos=False)
    metadata = JLensMetadata(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_id=MODEL_ID,
        tokenizer_revision=MODEL_REVISION,
        d_model=wrapped.d_model,
        source_layers=(0,),
        target_layer=1,
        norm_convention="raw",
        extra={"purpose": "occupancy smoke", "skip_first": 0},
    )
    lens = OfficialJLensAdapter.fit(
        wrapped,
        prompts=[
            "A small deterministic prompt checks that gradients flow through both transformer blocks."
        ],
        metadata=metadata,
        source_layers=[0],
        target_layer=1,
        dim_batch=2,
        max_seq_len=64,
        skip_first=0,
        checkpoint_path=None,
    )

    effective = restrict_effective_unembedding(
        build_effective_unembedding(model, convention="raw"), len(tokenizer)
    )

    rng = np.random.default_rng(42)
    vectors = {
        "smoke:alpha": rng.normal(size=wrapped.d_model),
        "smoke:beta": rng.normal(size=wrapped.d_model),
    }
    targets = [
        ConceptTarget(
            layer=0,
            concept_id=concept_id,
            vector=vector,
            vector_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),
            provenance={"source": "synthetic-smoke"},
        )
        for concept_id, vector in vectors.items()
    ]

    summary = run_concept_occupancy(
        output_dir=args.output,
        dictionary_factory=lambda layer, convention: build_token_frame_dictionary(
            effective,
            lens.jacobians[layer],
            convention=convention,
            chunk_size=1024,
            compute_device=args.device,
        ),
        targets=targets,
        conventions=["raw"],
        selection_modes=["positive_cosine", "raw_positive_dot"],
        signs=["+", "-"],
        k_max=8,
        report_grid=[1, 2, 4, 8],
        random_seeds=[101, 202],
        control_atom_fractions=[1.0, 0.5],
        chunk_size=1024,
        decode_token=lambda token_id: tokenizer.decode([token_id]),
        run_metadata={"purpose": "occupancy smoke", "model": MODEL_ID},
    )

    index = json.loads((args.output / "occupancy" / "index.json").read_text())
    completed = [e for e in index["entries"] if e["status"] == "completed"]
    assert summary["completed"] == 2 * 2 * 2 == len(completed)
    sample = json.loads(
        (
            args.output
            / "occupancy/raw/positive_cosine/layer_00/smoke%3Aalpha/pos/primary/metrics.json"
        ).read_text()
    )
    assert sample["method"] == "concept_occupancy_method_v1"
    assert len(sample["support_token_ids"]) <= 8
    assert sample["decoded_tokens"] is not None
    errors = np.load(
        args.output
        / "occupancy/raw/positive_cosine/layer_00/smoke%3Aalpha/pos/primary/errors.npy",
        allow_pickle=False,
    )
    assert errors[0] == 1.0 and np.all(np.diff(errors) <= 1e-12)
    reduced = (
        args.output
        / "occupancy/raw/positive_cosine/layer_00/smoke%3Aalpha/pos/primary"
    )
    assert (reduced / "control_errors_f0p5.npy").is_file()
    assert "0.5" in sample["occupancy_by_control_fraction"]
    print(
        f"ok model={MODEL_ID}@{MODEL_REVISION} combos={summary['completed']} "
        f"vocab={effective.vocab_size} d_model={wrapped.d_model} "
        f"sample_E8={errors[-1]:.4f} output={args.output}"
    )


if __name__ == "__main__":
    main()
