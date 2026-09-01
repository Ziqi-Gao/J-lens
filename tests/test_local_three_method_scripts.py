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


def test_candidate_rescore_decoding_covers_three_methods() -> None:
    assert _decode("candidate-rescore", 0) == {
        "task": "candidate-rescore",
        "task_index": "0",
    }
    assert _decode("candidate-rescore", 2)["task_index"] == "2"


def test_scheduled_rescore_is_immutable_and_uses_a_derived_comparison_root() -> None:
    task = (SCRIPTS / "server_scheduler_three_method_task.sh").read_text(
        encoding="utf-8"
    )
    rescore_body = task.rsplit("  candidate-rescore)", 1)[1].split(
        "  comparison-rescore-index)", 1
    )[0]
    comparison_body = task.rsplit("  comparison-rescore-index)", 1)[1].split("\nesac", 1)[0]

    assert "--overwrite" not in rescore_body
    assert 'RESCORE_ROOT="derivations/revision-r2/candidate_score_rescore_v1"' in task
    assert 'RESCORE_COMPARISON_ROOT="derivations/revision-r2/intervention_comparison"' in task
    assert '--output "${RESCORE_COMPARISON_ROOT}"' in comparison_body
    assert '--output "${COMPARISON_ROOT}"' not in comparison_body


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
    assert _gpu_policy("candidate-rescore")["task_class"] == "standard"
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


def test_k_diagnostic_profiles_and_local_dag_are_scheduler_only() -> None:
    assert _gpu_policy("kdiag-bundle") == {
        "task_class": "kdiag-bundle",
        "lock_mode": "shared",
        "slots_per_device": "5",
        "cpu_tokens": "4",
        "host_ram_mib": "12288",
        "gpu_vram_mib": "4096",
        "gpu_memory_reserve_mib": "8192",
        "gpu_utilization_tokens": "20",
    }
    assert _gpu_policy("kdiag-targets")["lock_mode"] == "exclusive"
    assert _gpu_policy("kdiag-rotations")["cpu_tokens"] == "16"
    assert _gpu_policy("kdiag-rotations")["host_ram_mib"] == "32768"

    controller_path = SCRIPTS / "run_qwen35_4b_k_diagnostic_v2.sh"
    result = subprocess.run(
        ["bash", str(controller_path), "plan"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "worst pilot bundle only" in result.stdout
    assert "xargs launches scheduler workers, never Python directly" in result.stdout
    controller = controller_path.read_text(encoding="utf-8")
    worker = (SCRIPTS / "run_qwen35_4b_k_diagnostic_v2_local_worker.sh").read_text(encoding="utf-8")
    task = (SCRIPTS / "run_qwen35_4b_k_diagnostic_v2_local_task.sh").read_text(encoding="utf-8")
    assert 'RUN_ROOT="/data/del6500/J-lens/runs/qwen35_4b_k_diagnostic_v2"' in controller
    assert 'JLENS_LOCAL_RUNTIME_ROOT="/scr/del6500/J-lens/runtime/three-method"' in controller
    assert "three_method_local_paths.sh" in controller
    assert "flock -n 8" in controller
    assert 'git -C "${CODE_ROOT}" status --short' in controller
    assert '"${WORKER}" gpu kdiag-bundle bundle' in controller
    assert 'xargs -r -n 1 -P "${WORKERS}"' in controller
    assert "\nsbatch" not in controller
    assert "submit_qwen35_4b_k_diagnostic_v2.sh" not in controller
    assert "three_method_local_gpu.sh" in worker
    assert "three_method_local_resources.sh" in worker
    assert "k-diagnostic-v2/${COMMIT}" in worker
    assert "JLENS_LOCAL_KDIAG_DAG_ACTIVE" in task
    assert "CUDA_VISIBLE_DEVICES" not in controller
    assert "CUDA_VISIBLE_DEVICES" not in worker


def test_k_diagnostic_local_components_reject_direct_execution() -> None:
    for script, arguments in (
        (
            "run_qwen35_4b_k_diagnostic_v2_local_worker.sh",
            ["gpu", "kdiag-bundle", "bundle", "pilot", "0"],
        ),
        ("run_qwen35_4b_k_diagnostic_v2_local_task.sh", ["kdiag-bundle", "bundle", "pilot", "0"]),
    ):
        result = subprocess.run(
            ["bash", str(SCRIPTS / script), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "top-level local DAG" in result.stderr


def test_server_scheduler_task_wrappers_are_separate_and_reject_direct_execution() -> None:
    three_method = (SCRIPTS / "server_scheduler_three_method_task.sh").read_text(
        encoding="utf-8"
    )
    kdiag = (SCRIPTS / "server_scheduler_kdiag_task.sh").read_text(encoding="utf-8")
    legacy_three_method = (SCRIPTS / "run_three_method_local_task.sh").read_text(
        encoding="utf-8"
    )
    legacy_kdiag = (SCRIPTS / "run_qwen35_4b_k_diagnostic_v2_local_task.sh").read_text(
        encoding="utf-8"
    )

    assert "JLENS_SERVER_SCHEDULER_TASK_ACTIVE" in three_method
    assert "JLENS_SERVER_SCHEDULER_TASK_ACTIVE" in kdiag
    assert "JLENS_SERVER_SCHEDULER_TASK_ACTIVE" not in legacy_three_method
    assert "JLENS_SERVER_SCHEDULER_TASK_ACTIVE" not in legacy_kdiag
    assert 'RESCORE_ROOT="derivations/revision-r2/' in three_method
    assert '--output "${RUN_ROOT}/derivations/revision-r1/report"' in kdiag
    validate_body = kdiag.split("  validate)", 1)[1].split("    ;;", 1)[0]
    assert "run_qwen35_4b_k_diagnostic_v2_local_task.sh" not in validate_body
    assert "jlens_workspace.cli k-diagnostic validate" in validate_body

    for script, arguments in (
        ("server_scheduler_three_method_task.sh", ["candidate-rescore", "0"]),
        ("server_scheduler_kdiag_task.sh", ["kdiag-control", "report", "transformed"]),
    ):
        result = subprocess.run(
            ["bash", str(SCRIPTS / script), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "ServerScheduler adapter" in result.stderr


def test_local_finalize_is_cpu_only_and_independent_of_grid_markers() -> None:
    controller = (SCRIPTS / "run_three_method_local.sh").read_text()
    finalize_body = controller.split("run_finalize() {", 1)[1].split("\n}", 1)[0]

    assert 'run_range cpu method-index 2 "${CPU_WORKERS}"' in finalize_body
    assert "run_one cpu comparison-index" in finalize_body
    assert " gpu " not in finalize_body
    assert "-grid" not in finalize_body
    assert "smoke-check.done" not in finalize_body
    assert "finalize) run_finalize ;;" in controller


def test_local_rescore_runs_three_gpu_methods_then_cpu_comparison() -> None:
    controller = (SCRIPTS / "run_three_method_local.sh").read_text()
    rescore_body = controller.split("run_rescore() {", 1)[1].split("run_full() {", 1)[0]

    assert 'run_range gpu candidate-rescore 2 "${GPU_WORKERS}"' in rescore_body
    assert "run_one cpu comparison-rescore-index" in rescore_body
    assert "rescore) run_rescore ;;" in controller
    assert "RESCORE_FINAL_INDEX" not in controller
    legacy_task = (SCRIPTS / "run_three_method_local_task.sh").read_text()
    assert (
        'RESCORE_ROOT="artifacts/concept_intervention/'
        'qwen35_4b_three_method_intervention_v1/candidate_score_rescore_v1"'
        in legacy_task
    )
    assert "--overwrite" in legacy_task
