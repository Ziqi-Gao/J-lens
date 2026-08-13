from __future__ import annotations

import os
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


def _gpu_policy(task: str, *, slots: str | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    if slots is not None:
        environment["JLENS_LOCAL_OCCUPANCY_GPU_SLOTS_PER_DEVICE"] = slots
    result = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "three_method_local_gpu.sh"),
            "--describe-policy",
            task,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def test_local_gpu_policy_uses_bounded_shared_slots_only_for_occupancy() -> None:
    assert _gpu_policy("occupancy") == {
        "task_class": "occupancy",
        "lock_mode": "shared",
        "slots_per_device": "5",
        "cpu_tokens": "4",
        "host_ram_mib": "4096",
        "gpu_vram_mib": "2048",
        "gpu_memory_reserve_mib": "8192",
        "gpu_utilization_tokens": "20",
    }
    assert _gpu_policy("lens") == {
        "task_class": "standard",
        "lock_mode": "exclusive",
        "slots_per_device": "1",
        "cpu_tokens": "4",
        "host_ram_mib": "24576",
        "gpu_vram_mib": "24576",
        "gpu_memory_reserve_mib": "16384",
        "gpu_utilization_tokens": "40",
    }
    assert _gpu_policy("occupancy", slots="3")["slots_per_device"] == "3"
    gpu_script = (SCRIPTS / "three_method_local_gpu.sh").read_text()
    controller = (SCRIPTS / "run_three_method_local.sh").read_text()
    assert "0,1,2,3" not in gpu_script
    assert 'GPU_WORKERS="${JLENS_LOCAL_GPU_WORKERS:-10}"' in controller


def test_local_gpu_policy_rejects_invalid_occupancy_slot_count() -> None:
    environment = {
        **os.environ,
        "JLENS_LOCAL_OCCUPANCY_GPU_SLOTS_PER_DEVICE": "0",
    }
    result = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "three_method_local_gpu.sh"),
            "--describe-policy",
            "occupancy",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 2
    assert "must be positive" in result.stderr


def test_local_finalize_is_cpu_only_and_independent_of_grid_markers() -> None:
    controller = (SCRIPTS / "run_three_method_local.sh").read_text()
    finalize_body = controller.split("run_finalize() {", 1)[1].split(
        "run_full() {", 1
    )[0]

    assert 'run_range cpu method-index 2 "${CPU_WORKERS}"' in finalize_body
    assert "run_one cpu comparison-index" in finalize_body
    assert " gpu " not in finalize_body
    assert "-grid" not in finalize_body
    assert "smoke-check.done" not in finalize_body
    assert "finalize) run_finalize ;;" in controller
