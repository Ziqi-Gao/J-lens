from __future__ import annotations

import pytest

from jlens_workspace.activations import (
    _capture_forward_kwargs,
    _example_text,
    _last_token_indices,
    _shared_source_examples,
)


def test_labeled_text_is_used_without_prompt_or_response_wrapping() -> None:
    text = _example_text({"text": "answer", "prompt": "question", "response": "answer"})
    assert text == "answer"


def test_last_token_indices_follow_each_rows_attention_mask() -> None:
    torch = pytest.importorskip("torch")
    attention = torch.tensor([[1, 1, 1, 0], [1, 0, 1, 0]], dtype=torch.long)
    assert _last_token_indices(attention).tolist() == [2, 2]


def test_last_token_indices_reject_empty_rows() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="at least one token"):
        _last_token_indices(torch.tensor([[0, 0]], dtype=torch.long))


def test_capture_requests_only_last_logit_when_model_supports_it() -> None:
    class ModernModel:
        def forward(self, *, logits_to_keep: int = 0, use_cache: bool = True) -> None:
            del logits_to_keep, use_cache

    class LegacyModel:
        def forward(self, *, use_cache: bool = True) -> None:
            del use_cache

    assert _capture_forward_kwargs(ModernModel()) == {
        "use_cache": False,
        "logits_to_keep": 1,
    }
    assert _capture_forward_kwargs(LegacyModel()) == {"use_cache": False}


def test_shared_source_examples_build_one_multiconcept_row_per_group() -> None:
    examples = []
    for source_number, labels in enumerate(((1, 0), (0, 1))):
        for concept_number, concept_id in enumerate(("alpha", "beta")):
            examples.append(
                {
                    "concept_id": concept_id,
                    "concept_name": concept_id.title(),
                    "definition": f"Definition of {concept_id}",
                    "label": labels[concept_number],
                    "text": f"source text {source_number}",
                    "split": "train",
                    "group_id": f"source-{source_number}",
                    "source": "synthetic",
                    "license": "CC0-1.0",
                }
            )

    representatives, label_matrix, rows, concepts = _shared_source_examples(examples)
    assert len(representatives) == len(rows) == 2
    assert label_matrix.tolist() == [[1, 0], [0, 1]]
    assert [concept["concept_id"] for concept in concepts] == ["alpha", "beta"]
    assert all(len(row["text_sha256"]) == 64 for row in rows)
