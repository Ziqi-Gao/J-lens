from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import jlens_workspace.local_scheduler as scheduler


def _lease(*, gpu: int, utilization: int = 20, vram: int = 2048) -> scheduler.Lease:
    return scheduler.Lease(
        lease_id=f"lease-{gpu}-{utilization}-{vram}",
        owner_pid=1,
        child_pid=None,
        task="occupancy",
        kind="gpu",
        gpu_index=gpu,
        cpu_tokens=4,
        host_ram_mib=4096,
        gpu_vram_mib=vram,
        gpu_utilization_tokens=utilization,
        created_at="2026-08-02T00:00:00Z",
    )


def _gpu(
    index: int,
    *,
    free: int = 90_000,
    utilization: int = 0,
    foreign_memory: int = 0,
) -> scheduler.GPUObservation:
    processes = ()
    if foreign_memory:
        processes = (
            scheduler.ComputeProcess(
                pid=9000 + index,
                process_name="/foreign/python",
                used_memory_mib=foreign_memory,
                owner="foreign",
            ),
        )
    return scheduler.GPUObservation(
        index=index,
        total_memory_mib=100_000,
        minimum_free_memory_mib=free,
        maximum_utilization=utilization,
        processes=processes,
    )


def test_profiles_separate_cpu_heavy_occupancy_from_standard_gpu_work() -> None:
    occupancy = scheduler.profile_for_task("occupancy", {})
    assert occupancy.task_class == "occupancy"
    assert occupancy.lock_mode == "shared"
    assert occupancy.slots_per_device == 5
    assert occupancy.cpu_tokens == 4
    assert occupancy.gpu_vram_mib == 2048
    assert occupancy.gpu_memory_reserve_mib == 8192
    assert occupancy.gpu_utilization_tokens == 20

    standard = scheduler.profile_for_task("iti-grid", {})
    assert standard.task_class == "standard"
    assert standard.lock_mode == "exclusive"
    assert standard.slots_per_device == 1
    assert standard.gpu_vram_mib == 24576
    assert standard.gpu_memory_reserve_mib == 16384
    assert standard.gpu_utilization_tokens == 40
    assert scheduler.profile_for_task("candidate-rescore", {}) == standard

    targets = scheduler.profile_for_task("kdiag-targets", {})
    assert targets == standard

    bundle = scheduler.profile_for_task("kdiag-bundle", {})
    assert bundle.task_class == "kdiag-bundle"
    assert bundle.lock_mode == "shared"
    assert bundle.slots_per_device == 5
    assert bundle.cpu_tokens == 4
    assert bundle.host_ram_mib == 12288
    assert bundle.gpu_vram_mib == 4096
    assert bundle.gpu_memory_reserve_mib == 8192
    assert bundle.gpu_utilization_tokens == 20

    overridden = scheduler.profile_for_task(
        "kdiag-bundle",
        {
            "JLENS_LOCAL_KDIAG_CPU_TOKENS": "6",
            "JLENS_LOCAL_KDIAG_HOST_RAM_MIB": "14000",
            "JLENS_LOCAL_KDIAG_GPU_VRAM_MIB": "5000",
            "JLENS_LOCAL_KDIAG_GPU_MEMORY_RESERVE_MIB": "9000",
            "JLENS_LOCAL_KDIAG_GPU_UTILIZATION_TOKENS": "25",
            "JLENS_LOCAL_KDIAG_GPU_SLOTS_PER_DEVICE": "3",
        },
    )
    assert overridden.cpu_tokens == 6
    assert overridden.host_ram_mib == 14000
    assert overridden.gpu_vram_mib == 5000
    assert overridden.gpu_memory_reserve_mib == 9000
    assert overridden.gpu_utilization_tokens == 25
    assert overridden.slots_per_device == 3

    fit = scheduler.profile_for_task("iti-fit", {})
    assert fit.task_class == "cpu"
    assert fit.cpu_tokens == 16
    assert fit.host_ram_mib == 32768
    rotations = scheduler.profile_for_task("kdiag-rotations", {})
    assert rotations.task_class == "cpu"
    assert rotations.cpu_tokens == 16
    assert rotations.host_ram_mib == 32768


def test_kdiag_bundle_uses_the_shared_occupancy_gate_and_slots(
    monkeypatch,
    tmp_path: Path,
) -> None:
    environment = {
        "CODE_ROOT": str(tmp_path / "code"),
        "JLENS_LOCAL_DATA_ROOT": str(tmp_path / "data"),
        "JLENS_LOCAL_SCRATCH_ROOT": str(tmp_path / "scratch"),
        "JLENS_LOCAL_RUNTIME_ROOT": str(tmp_path / "runtime"),
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    calls: list[tuple[Path, bool]] = []

    def fake_lock(path: Path, *, shared: bool):
        calls.append((path, shared))
        return object()

    monkeypatch.setattr(scheduler, "_try_lock", fake_lock)
    local = scheduler.LocalScheduler("kdiag-bundle", "gpu", ["true"])
    assert local._try_gpu_locks(7) is True
    assert calls == [
        (tmp_path / "runtime/gpu-locks/gpu_7.lock", True),
        (tmp_path / "runtime/gpu-locks/gpu_7_occupancy_slot_0.lock", False),
    ]


def test_dynamic_discovery_has_no_hard_coded_device_ids(monkeypatch) -> None:
    monkeypatch.setattr(scheduler, "_run_text", lambda _command: "2\n7\n11\n")
    assert scheduler.discover_gpu_ids({}) == [2, 7, 11]
    assert scheduler.discover_gpu_ids({"JLENS_LOCAL_GPU_IDS": "11,2"}) == [2, 11]


def test_ranking_prefers_clean_gpu_then_uses_bounded_foreign_overlay() -> None:
    profile = scheduler.profile_for_task("occupancy", {})
    observations = {
        2: _gpu(2, utilization=40, foreign_memory=35_000),
        7: _gpu(7),
    }
    ranked = scheduler.rank_candidates(
        observations,
        profile,
        [],
        allow_overlay=True,
        memory_reserve_mib=8192,
        maximum_start_utilization=85,
        target_utilization=95,
    )
    assert [candidate.observation.index for candidate in ranked] == [7, 2]

    clean_gpu_is_full = [_lease(gpu=7) for _ in range(4)]
    ranked = scheduler.rank_candidates(
        observations,
        profile,
        clean_gpu_is_full,
        allow_overlay=True,
        memory_reserve_mib=8192,
        maximum_start_utilization=85,
        target_utilization=95,
    )
    assert [candidate.observation.index for candidate in ranked] == [2]


def test_admission_fails_closed_on_overlay_or_memory_limits() -> None:
    profile = scheduler.profile_for_task("occupancy", {})
    foreign = _gpu(3, free=10_000, utilization=35, foreign_memory=90_000)
    assert (
        scheduler.candidate_for_gpu(
            foreign,
            profile,
            [],
            allow_overlay=True,
            memory_reserve_mib=8192,
            maximum_start_utilization=85,
            target_utilization=95,
        )
        is None
    )
    roomy_foreign = _gpu(3, utilization=35, foreign_memory=30_000)
    assert (
        scheduler.candidate_for_gpu(
            roomy_foreign,
            profile,
            [],
            allow_overlay=False,
            memory_reserve_mib=8192,
            maximum_start_utilization=85,
            target_utilization=95,
        )
        is None
    )


def test_process_classification_uses_jlens_owned_roots() -> None:
    assert (
        scheduler.classify_process(
            999_999_999,
            "/scr/del6500/J-lens/envs/three-method/bin/python",
            code_root=Path("/home/del6500/projects/J-lens"),
            data_root=Path("/data/del6500/J-lens"),
            scratch_root=Path("/scr/del6500/J-lens"),
        )
        == "jlens"
    )
    assert (
        scheduler.classify_process(
            999_999_999,
            "/data/another-project/env/bin/python",
            code_root=Path("/home/del6500/projects/J-lens"),
            data_root=Path("/data/del6500/J-lens"),
            scratch_root=Path("/scr/del6500/J-lens"),
        )
        == "foreign"
    )


def test_scheduler_runs_on_dynamically_discovered_fake_gpu(tmp_path: Path) -> None:
    fake_smi = tmp_path / "nvidia-smi"
    fake_smi.write_text(
        """#!/bin/sh
set -eu
if [ "$1" = "--query-gpu=index" ]; then
  printf '2\\n7\\n'
elif [ "$1" = "--query-gpu=index,memory.free,memory.total,utilization.gpu" ]; then
  printf '2, 90000, 100000, 0\\n7, 80000, 100000, 10\\n'
elif [ "$1" = "-i" ]; then
  exit 0
else
  exit 2
fi
"""
    )
    fake_smi.chmod(0o755)
    runtime_root = tmp_path / "runtime"
    environment = {
        **os.environ,
        "CODE_ROOT": str(tmp_path / "code"),
        "JLENS_LOCAL_DATA_ROOT": str(tmp_path / "data"),
        "JLENS_LOCAL_SCRATCH_ROOT": str(tmp_path / "scratch"),
        "JLENS_LOCAL_RUNTIME_ROOT": str(runtime_root),
        "JLENS_LOCAL_NVIDIA_SMI": str(fake_smi),
        "JLENS_LOCAL_GPU_STABILITY_SAMPLES": "1",
        "JLENS_LOCAL_GPU_SAMPLE_INTERVAL_SECONDS": "0",
        "JLENS_LOCAL_HOST_CPU_TOKENS": "64",
        "JLENS_LOCAL_HOST_RAM_RESERVE_MIB": "1",
    }
    environment.pop("JLENS_LOCAL_GPU_IDS", None)
    for task in ("occupancy", "kdiag-bundle"):
        result = subprocess.run(
            [
                sys.executable,
                str(Path(scheduler.__file__)),
                "run",
                "--kind",
                "gpu",
                "--task",
                task,
                "--",
                sys.executable,
                "-c",
                "import os; assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'; assert os.environ['JLENS_LOCAL_LEASE_ID']",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert "jlens_gpu_selected=2" in result.stdout
        lease_dir = runtime_root / "scheduler" / "leases"
        assert list(lease_dir.glob("*.json")) == []


def test_standard_task_uses_larger_memory_reserve_on_foreign_gpu() -> None:
    profile = scheduler.profile_for_task("iti-grid", {})
    observations = {
        1: _gpu(1, free=36_000, utilization=40, foreign_memory=61_000),
        2: _gpu(2, free=62_000, utilization=46, foreign_memory=35_000),
    }
    ranked = scheduler.rank_candidates(
        observations,
        profile,
        [],
        allow_overlay=True,
        memory_reserve_mib=profile.gpu_memory_reserve_mib,
        maximum_start_utilization=85,
        target_utilization=95,
    )
    assert [candidate.observation.index for candidate in ranked] == [2]
