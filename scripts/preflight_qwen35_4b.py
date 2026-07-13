#!/usr/bin/env python3
"""GPU/model/lens compatibility preflight for the pinned production runs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from jlens_workspace.activations import _capture_forward_kwargs
from jlens_workspace.cli import _build_effective_unembedding, _load_or_fit_lens
from jlens_workspace.config import load_experiment_config
from jlens_workspace.jacobian import OfficialJLensAdapter, import_official_jlens
from jlens_workspace.matrix import TokenFrameOperator
from jlens_workspace.modeling import load_hf_bundle, model_input_device, transformer_blocks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    probe = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
    if float((probe @ probe).sum().item()) <= 0:
        raise RuntimeError("CUDA bfloat16 matmul preflight failed")

    config = load_experiment_config(args.config)
    bundle = load_hf_bundle(config.model)
    encoded = bundle.tokenizer(
        "A short offline preflight checks the pinned Qwen model and Jacobian lens.",
        return_tensors="pt",
    )
    encoded = {
        key: value.to(model_input_device(bundle.model)) for key, value in encoded.items()
    }
    with torch.inference_mode():
        output = bundle.model(**encoded, **_capture_forward_kwargs(bundle.model))
    logits = output.logits
    if logits.ndim != 3 or logits.shape[0] != 1 or not torch.isfinite(logits).all():
        raise RuntimeError(f"invalid model output shape {tuple(logits.shape)}")

    lens = config.lens
    if lens is None:
        raise RuntimeError("preflight config has no lens section")
    fitted_path = Path(lens.fit_output_path) if lens.fit_output_path else None
    managed_lens = None
    jacobian_benchmark = None
    if lens.source != "fit" or (fitted_path is not None and fitted_path.is_file()):
        managed_lens = _load_or_fit_lens(config, bundle)
    else:
        wrapped = OfficialJLensAdapter.from_hf(
            bundle.model,
            bundle.tokenizer,
            compile=lens.compile_blocks,
            force_bos=config.model.force_bos,
        )
        if wrapped.d_model != bundle.model.config.get_text_config().hidden_size:
            raise RuntimeError("official J-lens wrapper changed the residual width")
        prompt_path = Path(lens.fit_prompts_path)
        first_record = json.loads(prompt_path.read_text(encoding="utf-8").splitlines()[0])
        prompt = first_record["text"] if isinstance(first_record, dict) else first_record
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        jacobians, sequence_length, valid_positions = (
            import_official_jlens().jacobian_for_prompt(
                wrapped,
                prompt,
                lens.layers,
                target_layer=lens.target_layer,
                dim_batch=lens.dim_batch,
                max_seq_len=lens.max_seq_len,
                skip_first=lens.skip_first,
            )
        )
        if set(jacobians) != set(lens.layers):
            raise RuntimeError("single-prompt J-lens benchmark returned the wrong layers")
        jacobian_benchmark = {
            "seconds": time.perf_counter() - started,
            "sequence_length": sequence_length,
            "valid_positions": valid_positions,
            "dim_batch": lens.dim_batch,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        del jacobians
    convention = config.matrix.convention if config.matrix is not None else config.alignment.convention
    effective = _build_effective_unembedding(config, bundle, convention)
    tokenizer_size = len(bundle.tokenizer)
    if tokenizer_size != effective.vocab_size:
        raise RuntimeError(
            f"tokenizer/unembedding vocabulary mismatch: {tokenizer_size} != "
            f"{effective.vocab_size}"
        )
    first = None
    if managed_lens is not None:
        layer = lens.layers[0]
        operator = TokenFrameOperator.from_lens(
            managed_lens,
            layer,
            effective,
            block_size=64,
            compute_device="cuda",
            compute_dtype=torch.float32,
            cpu_fallback=False,
        )
        first = next(operator.iter_rows())
        if first.rows.shape != (64, effective.d_model) or not torch.isfinite(first.rows).all():
            raise RuntimeError(f"invalid A_l block shape {tuple(first.rows.shape)}")

    payload = {
        "ok": True,
        "config": str(args.config),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "model_class": type(bundle.model).__name__,
        "n_layers": len(transformer_blocks(bundle.model)),
        "d_model": effective.d_model,
        "vocab_size": effective.vocab_size,
        "unembedding_vocab_size": effective.metadata.unembedding_vocab_size,
        "unembedding_row_selection": effective.metadata.row_selection,
        "tokenizer_size": tokenizer_size,
        "vocabulary_identity_verified": True,
        "lens_status": "loaded" if managed_lens is not None else "fit_pending",
        "lens_layers": len(lens.layers),
        "lens_n_prompts": None if managed_lens is None else managed_lens.n_prompts,
        "a_block_shape": None if first is None else list(first.rows.shape),
        "single_prompt_jacobian_benchmark": jacobian_benchmark,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
