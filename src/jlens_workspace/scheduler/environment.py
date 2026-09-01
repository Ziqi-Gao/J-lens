"""Protocol-v2 scheduler environment and allocation identity validation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from jlens_workspace.scheduler.allocation import Allocation
from jlens_workspace.scheduler.catalog import PROJECT
from jlens_workspace.scheduler.errors import ServerSchedulerAdapterError
from jlens_workspace.scheduler.jsonio import identifier, integer

SINGULAR_GPU_ENVIRONMENT = {
    "SERVER_SCHEDULER_GPU_INDEX",
    "SERVER_SCHEDULER_GPU_UUID",
    "SERVER_SCHEDULER_GPU_PCI_BUS_ID",
}


def _environment_integer(env: Mapping[str, str], key: str, expected: int) -> None:
    if env.get(key) != str(expected):
        raise ServerSchedulerAdapterError(f"{key} is absent or disagrees with the manifest")


def _environment_csv(env: Mapping[str, str], key: str, expected: Sequence[object]) -> None:
    value = ",".join(str(item) for item in expected)
    if env.get(key) != value:
        raise ServerSchedulerAdapterError(f"{key} is absent or disagrees with the manifest")


def validate_scheduler_environment(
    values: Mapping[str, str],
    *,
    manifest_path: Path,
    job_id: str,
    execution_profile: str,
    estimated_runtime_seconds: float,
    allocation: Allocation,
    cpu_ids: tuple[int, ...],
    numa_node: int | None,
    gpu_indices: tuple[int, ...],
    gpu_uuids: tuple[str, ...],
    gpu_pci_bus_ids: tuple[str, ...],
    stdout_log: object,
    stderr_log: object,
) -> int:
    """Cross-check every scheduler-provided environment field against the manifest."""

    if any(key in values for key in SINGULAR_GPU_ENVIRONMENT):
        raise ServerSchedulerAdapterError(
            "protocol v2 forbids singular ServerScheduler GPU identity variables"
        )

    expected_manifest = values.get("SERVER_SCHEDULER_JOB_MANIFEST", "")
    if not expected_manifest or Path(expected_manifest).resolve() != manifest_path.resolve():
        raise ServerSchedulerAdapterError("SERVER_SCHEDULER_JOB_MANIFEST disagrees with argv")
    if values.get("SERVER_SCHEDULER_JOB_ID") != job_id:
        raise ServerSchedulerAdapterError("SERVER_SCHEDULER_JOB_ID disagrees with the manifest")
    if values.get("SERVER_SCHEDULER_PROJECT") != PROJECT:
        raise ServerSchedulerAdapterError("SERVER_SCHEDULER_PROJECT disagrees with the manifest")
    identifier(values.get("SERVER_SCHEDULER_LEASE_ID"), label="SERVER_SCHEDULER_LEASE_ID")
    attempt = integer(
        int(values.get("SERVER_SCHEDULER_ATTEMPT", "0"))
        if values.get("SERVER_SCHEDULER_ATTEMPT", "").isdigit()
        else values.get("SERVER_SCHEDULER_ATTEMPT"),
        label="SERVER_SCHEDULER_ATTEMPT",
        minimum=1,
    )
    if values.get("SERVER_SCHEDULER_EXECUTION_PROFILE") != execution_profile:
        raise ServerSchedulerAdapterError(
            "SERVER_SCHEDULER_EXECUTION_PROFILE disagrees with the manifest"
        )
    try:
        environment_runtime = float(values.get("SERVER_SCHEDULER_ESTIMATED_RUNTIME_SECONDS", ""))
    except ValueError as error:
        raise ServerSchedulerAdapterError(
            "SERVER_SCHEDULER_ESTIMATED_RUNTIME_SECONDS is malformed"
        ) from error
    if not math.isclose(
        environment_runtime,
        estimated_runtime_seconds,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise ServerSchedulerAdapterError(
            "SERVER_SCHEDULER_ESTIMATED_RUNTIME_SECONDS disagrees with the manifest"
        )
    _environment_integer(values, "SERVER_SCHEDULER_CPU_CORES", allocation.cpu_cores)
    _environment_integer(values, "SERVER_SCHEDULER_MEMORY_MIB", allocation.memory_mib)
    _environment_integer(values, "SERVER_SCHEDULER_GPU_COUNT", allocation.gpu_count)
    _environment_integer(values, "SERVER_SCHEDULER_GPU_MEMORY_MIB", allocation.gpu_memory_mib)
    _environment_integer(
        values,
        "SERVER_SCHEDULER_GPU_UTILIZATION_PCT",
        allocation.gpu_utilization_pct,
    )
    expected_exclusivity = "exclusive" if allocation.exclusive_gpu else "shared"
    if values.get("SERVER_SCHEDULER_GPU_EXCLUSIVITY") != expected_exclusivity:
        raise ServerSchedulerAdapterError(
            "SERVER_SCHEDULER_GPU_EXCLUSIVITY disagrees with the allocation"
        )
    _environment_csv(values, "SERVER_SCHEDULER_CPUSET", cpu_ids)
    _environment_csv(values, "SERVER_SCHEDULER_GPU_INDICES", gpu_indices)
    _environment_csv(values, "SERVER_SCHEDULER_GPU_UUIDS", gpu_uuids)
    _environment_csv(values, "SERVER_SCHEDULER_GPU_PCI_BUS_IDS", gpu_pci_bus_ids)
    if numa_node is None:
        if "SERVER_SCHEDULER_NUMA_NODE" in values:
            raise ServerSchedulerAdapterError(
                "SERVER_SCHEDULER_NUMA_NODE must be absent for a cross-NUMA allocation"
            )
    else:
        _environment_integer(values, "SERVER_SCHEDULER_NUMA_NODE", numa_node)

    visible = values.get("CUDA_VISIBLE_DEVICES", "")
    if allocation.gpu_count == 0:
        if visible.strip():
            raise ServerSchedulerAdapterError("CPU task inherited GPU visibility")
    else:
        expected_visible = ",".join(gpu_uuids or tuple(str(item) for item in gpu_indices))
        if visible != expected_visible:
            raise ServerSchedulerAdapterError(
                "CUDA_VISIBLE_DEVICES disagrees with the assigned device order"
            )

    for log_path, key, field in (
        (stdout_log, "SERVER_SCHEDULER_STDOUT_LOG", "stdout_log"),
        (stderr_log, "SERVER_SCHEDULER_STDERR_LOG", "stderr_log"),
    ):
        if not isinstance(log_path, str) or not Path(log_path).is_absolute():
            raise ServerSchedulerAdapterError(f"{field} must be an absolute scheduler-owned path")
        if values.get(key) != log_path:
            raise ServerSchedulerAdapterError(f"{key} disagrees with the manifest")
    return attempt
