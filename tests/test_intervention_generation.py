from __future__ import annotations

import json
from pathlib import Path

from jlens_workspace.concept_intervention.generation import (
    load_open_prompt_bank,
    write_generation_artifacts,
)


class _Tokenizer:
    chat_template = None


def test_frozen_open_prompts_are_exactly_balanced() -> None:
    root = Path(__file__).parents[1]
    prompts = load_open_prompt_bank(
        root / "Concept_intervention/data/open_intervention_prompts.json",
        tokenizer=_Tokenizer(),
    )
    assert len(prompts) == 32
    assert sum(prompt.split == "validation" for prompt in prompts) == 16
    assert sum(prompt.split == "test" for prompt in prompts) == 16
    assert len({prompt.prompt_id for prompt in prompts}) == 32


def test_blind_export_hides_method_but_keeps_private_mapping(tmp_path: Path) -> None:
    rows = [
        {
            "blind_id": "blind-a",
            "generation_id": "generation-a",
            "method": "j_component_intervention",
            "concept_id": "concept:a",
            "condition_id": "j",
            "grid_point": {"strength": 0.25},
            "prompt_id": "open_test_1",
            "prompt_text": "Write something.",
            "decoding": "greedy",
            "generated_text": "A response.",
        }
    ]
    paths = write_generation_artifacts(tmp_path, rows)
    blind = json.loads(
        Path(paths["blind_generations"]).read_text(encoding="utf-8").strip()
    )
    mapping = json.loads(Path(paths["blind_map"]).read_text(encoding="utf-8").strip())

    assert "method" not in blind
    assert "condition_id" not in blind
    assert mapping["method"] == "j_component_intervention"
    assert mapping["condition_id"] == "j"
