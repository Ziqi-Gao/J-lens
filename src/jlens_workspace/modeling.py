"""Small Hugging Face compatibility layer with explicit residual coordinates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from jlens_workspace.config import ModelConfig


@dataclass(frozen=True)
class ModelBundle:
    """A loaded model and tokenizer plus the exact requested revisions."""

    model: Any
    tokenizer: Any
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str


def _torch_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def load_hf_bundle(config: ModelConfig) -> ModelBundle:
    """Load a causal LM without importing the optional LLM stack at module import time."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_id = config.tokenizer_id or config.model_id
    tokenizer_revision = config.tokenizer_revision or config.revision
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=tokenizer_revision,
        trust_remote_code=config.trust_remote_code,
    )
    if (
        config.force_bos is not None
        and getattr(tokenizer, "bos_token_id", None) is not None
        and hasattr(tokenizer, "add_bos_token")
    ):
        tokenizer.add_bos_token = config.force_bos
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "revision": config.revision,
        "trust_remote_code": config.trust_remote_code,
        # Transformers 5 renamed the public loading argument from
        # ``torch_dtype`` to ``dtype``; the official J-lens Qwen walkthrough
        # uses this spelling as well.
        "dtype": _torch_dtype(config.dtype),
    }
    if config.device == "auto":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    if config.device != "auto":
        model.to(config.device)
    model.eval()
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_id=config.model_id,
        model_revision=config.revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
    )


def _resolve_path(root: Any, path: str) -> Any | None:
    value = root
    for part in path.split("."):
        if not hasattr(value, part):
            return None
        value = getattr(value, part)
    return value


def transformer_blocks(model: Any) -> Sequence[Any]:
    """Return decoder blocks for common HF causal-LM layouts.

    Hooks are installed on block outputs, i.e. the ``resid_post`` coordinate used
    by the reference Jacobian-lens implementation.
    """

    candidate_paths = (
        "model.layers",  # Llama, Qwen, Mistral, Gemma, Phi-3
        "model.language_model.layers",  # Qwen 3.5 conditional-generation wrapper
        "language_model.layers",
        "transformer.h",  # GPT-2
        "gpt_neox.layers",  # GPT-NeoX / Pythia
        "model.decoder.layers",  # OLMo-style wrappers
        "layers",
    )
    for path in candidate_paths:
        value = _resolve_path(model, path)
        if value is not None and hasattr(value, "__len__") and hasattr(value, "__getitem__"):
            return value
    raise TypeError(
        f"unsupported model layout {type(model).__module__}.{type(model).__name__}; "
        "add an explicit block resolver before using a lens"
    )


def final_norm(model: Any) -> Any:
    for path in (
        "model.norm",
        "model.language_model.norm",
        "language_model.norm",
        "transformer.ln_f",
        "gpt_neox.final_layer_norm",
        "model.decoder.final_layer_norm",
        "norm",
    ):
        value = _resolve_path(model, path)
        if value is not None:
            return value
    raise TypeError(f"cannot resolve final normalization layer for {type(model).__name__}")


def lm_head(model: Any) -> Any:
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if head is not None:
            return head
    for path in ("lm_head", "embed_out"):
        value = _resolve_path(model, path)
        if value is not None:
            return value
    raise TypeError(f"cannot resolve output embedding for {type(model).__name__}")


def hidden_from_block_output(output: Any) -> Any:
    """Extract the residual tensor while preserving the block's output container."""

    if hasattr(output, "shape"):
        return output
    if isinstance(output, (tuple, list)) and output and hasattr(output[0], "shape"):
        return output[0]
    raise TypeError(f"unsupported transformer block output type: {type(output)!r}")


def replace_hidden_in_block_output(output: Any, hidden: Any) -> Any:
    if hasattr(output, "shape"):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"unsupported transformer block output type: {type(output)!r}")


def register_resid_post_hook(model: Any, layer: int, hook: Callable[..., Any]) -> Any:
    blocks = transformer_blocks(model)
    if not 0 <= layer < len(blocks):
        raise IndexError(f"layer {layer} outside [0, {len(blocks)})")
    return blocks[layer].register_forward_hook(hook)


def model_input_device(model: Any) -> Any:
    """Find the embedding device, including models loaded with ``device_map``."""

    if hasattr(model, "get_input_embeddings"):
        embedding = model.get_input_embeddings()
        if embedding is not None and hasattr(embedding, "weight"):
            return embedding.weight.device
    return next(model.parameters()).device
