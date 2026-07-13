#!/usr/bin/env python3
"""Fit a real Jacobian lens on a tiny public GPT-2 checkpoint.

This is an integration check for dependency/API drift, not a scientific lens.
It runs on CPU in well under a minute because the checkpoint has d_model=2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODEL_ID = "sshleifer/tiny-gpt2"
MODEL_REVISION = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke/tiny_gpt2_lens.pt"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jlens_workspace.jacobian import (
        JLensMetadata,
        OfficialJLensAdapter,
        build_effective_unembedding,
    )
    from jlens_workspace.matrix import TokenFrameOperator, analyze_token_frame

    args = parse_args()
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
        extra={"purpose": "API smoke test", "skip_first": 0},
    )
    fitted = OfficialJLensAdapter.fit(
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
    OfficialJLensAdapter.save(fitted, args.output, dtype=torch.float32)
    reloaded = OfficialJLensAdapter.load(
        args.output,
        expected={
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "source_layers": (0,),
            "d_model": wrapped.d_model,
        },
    )
    matrix = reloaded.jacobians[0]
    if matrix.shape != (wrapped.d_model, wrapped.d_model) or not torch.isfinite(matrix).all():
        raise RuntimeError(f"invalid smoke-test Jacobian: shape={tuple(matrix.shape)}")
    frame = TokenFrameOperator(
        matrix,
        build_effective_unembedding(model, convention="raw"),
        layer=0,
        block_size=4096,
        compute_device=args.device,
    )
    _gram, spectrum = analyze_token_frame(frame, centered=True)
    basis = spectrum.minimum_energy_basis(0.95)
    if basis.n_components > wrapped.d_model:
        raise RuntimeError("invalid energy basis dimension")
    print(
        f"ok model={MODEL_ID}@{MODEL_REVISION} layer=0 "
        f"shape={tuple(matrix.shape)} n_prompts={fitted.n_prompts} "
        f"rank={spectrum.numerical_rank} basis95={basis.n_components} output={args.output}"
    )


if __name__ == "__main__":
    main()
