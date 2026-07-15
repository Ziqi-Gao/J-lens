from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_single_canonical_agent_guide() -> None:
    assert (ROOT / "AGENTS.md").is_file()
    assert not (ROOT / "Agent.md").exists()
    assert "[AGENTS.md](AGENTS.md)" in (ROOT / "README.md").read_text()


def test_shared_data_preparation_is_not_nested_in_either_lane() -> None:
    assert (ROOT / "Concept_intervention").is_dir()
    assert (ROOT / "J_space").is_dir()
    assert not (ROOT / "Concept_intervention/J_space").exists()
    assert (ROOT / "scripts/prepare_go_emotions.py").is_file()
    assert not (ROOT / "Concept_intervention/scripts/prepare_go_emotions.py").exists()


def test_formal_lanes_share_the_current_full_fit_prompt_artifact() -> None:
    assert not (ROOT / "Concept_intervention/configs/qwen35_4b_smoke.yaml").exists()
    concept = yaml.safe_load(
        (ROOT / "Concept_intervention/configs/qwen35_4b.yaml").read_text()
    )
    expected = concept["lens"]["fit_prompts_path"]
    assert expected == (
        "artifacts/data/go_emotions_7concept_full_ovr_v1/fit_prompts.jsonl"
    )
    for name in (
        "qwen35_4b.yaml",
        "qwen35_4b_centered.yaml",
        "qwen35_4b_row_normalized.yaml",
    ):
        matrix = yaml.safe_load((ROOT / "J_space/configs" / name).read_text())
        assert matrix["lens"]["fit_prompts_path"] == expected
        assert "goemotions_full" in matrix["lens"]["fit_output_path"]
