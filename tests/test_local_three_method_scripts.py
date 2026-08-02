from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "Concept_intervention/scripts"


def _decode(task: str, index: int) -> dict[str, str]:
    result = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "run_three_method_local_task.sh"),
            "--decode",
            task,
            str(index),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def test_local_grid_decoding_matches_registered_slurm_arrays() -> None:
    assert _decode("j-grid", 6) == {
        "task": "j-grid",
        "task_index": "6",
        "concept_id": "goemotions:admiration",
        "grid_index": "6",
    }
    assert _decode("j-grid", 391)["concept_id"] == "goemotions:optimism"
    assert _decode("j-grid", 391)["grid_index"] == "55"
    assert _decode("raptor-grid", 62)["grid_index"] == "8"
    assert _decode("iti-grid", 3086)["grid_index"] == "440"


def test_local_occupancy_decoding_matches_registered_slurm_array() -> None:
    assert _decode("occupancy", 0) == {
        "task": "occupancy",
        "task_index": "0",
        "layer": "3",
        "replicate_id": "primary",
    }
    assert _decode("occupancy", 34) == {
        "task": "occupancy",
        "task_index": "34",
        "layer": "27",
        "replicate_id": "bootstrap_4404",
    }


def test_local_top_level_plan_has_the_registered_smoke_indices() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_three_method_local.sh"), "plan"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "J[6] + RAPTOR[8] + ITI[6,251] smoke" in result.stdout
    assert "three method indexes" in result.stdout
