#!/usr/bin/env python3
"""Run data -> activations -> tuned probes -> batched J alignment on tiny GPT-2."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from smoke_jlens import MODEL_ID, MODEL_REVISION

from jlens_workspace.activations import capture_residual_activations
from jlens_workspace.artifacts import atomic_write_json
from jlens_workspace.data import load_jsonl, validate_examples
from jlens_workspace.jacobian import (
    JLensMetadata,
    OfficialJLensAdapter,
    build_effective_unembedding,
)
from jlens_workspace.matrix import TokenFrameOperator
from jlens_workspace.workflows import (
    run_batched_probe_j_alignment,
    run_concept_probe_workflow,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("Concept_intervention/data/builtin_abstract_concepts.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke/concept_pipeline"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {args.output}")
        shutil.rmtree(args.output)
    examples = load_jsonl(args.data)
    statistics = validate_examples(examples, min_per_label_per_split=2)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model.to(args.device).eval()

    activation_dir = args.output / "activations"
    capture_residual_activations(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        layers=[0],
        output_dir=activation_dir,
        batch_size=16,
        max_length=256,
    )
    probe_result = run_concept_probe_workflow(
        activation_dir,
        args.output / "probes",
        layers=[0],
        C_grid=(0.01, 0.1, 1.0, 10.0),
        cv_splits=4,
        standardize=True,
        class_weight="balanced",
        random_state=42,
    )

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
        extra={"purpose": "concept pipeline smoke test", "skip_first": 0},
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
    OfficialJLensAdapter.save(lens, args.output / "tiny_lens.pt", dtype=torch.float32)
    operator = TokenFrameOperator.from_lens(
        lens,
        0,
        build_effective_unembedding(model, convention="raw"),
        block_size=4096,
        compute_device=args.device,
    )
    probe_vectors = {
        probe.concept_id: np.load(probe.probe_vector_path, allow_pickle=False)
        for probe in probe_result.probes
    }
    alignments = run_batched_probe_j_alignment(
        probe_vectors=probe_vectors,
        operator=operator,
        output_dir=args.output / "alignment" / "layer_00",
        tokenizer=tokenizer,
        top_k=5,
        candidate_pool_size=16,
        sparse_components=2,
        metadata={"layer": 0, "convention": "raw", "smoke_only": True},
    )

    summary = {
        "dataset": statistics.to_dict(),
        "n_probes": len(probe_result.probes),
        "concepts": {},
    }
    for probe in probe_result.probes:
        metrics = json.loads(probe.metrics_path.read_text(encoding="utf-8"))
        summary["concepts"][probe.concept_id] = {
            "chosen_C": probe.chosen_C,
            "test_roc_auc": metrics["test"]["roc_auc"],
            "j_explained_energy": alignments[probe.concept_id]["decomposition"][
                "explained_energy"
            ],
        }
    atomic_write_json(args.output / "summary.json", summary)
    print(
        f"ok concepts={statistics.total_concepts} examples={statistics.total_examples} "
        f"probes={len(probe_result.probes)} batched_vocab_passes=1 output={args.output}"
    )


if __name__ == "__main__":
    main()
