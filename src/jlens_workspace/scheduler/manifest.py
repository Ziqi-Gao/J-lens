"""Strict protocol-v2 running-job manifest parsing."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping, Set
from dataclasses import dataclass
from pathlib import Path

from jlens_workspace.foundation.artifacts import git_head_commit
from jlens_workspace.scheduler.allocation import (
    Allocation,
    parse_cpu_binding,
    parse_gpu_binding,
    parse_resources,
    require_allocation_satisfies_request,
)
from jlens_workspace.scheduler.catalog import CODE_ROOT, PROJECT, TASK_SPECS, TaskSpec
from jlens_workspace.scheduler.environment import validate_scheduler_environment
from jlens_workspace.scheduler.errors import ServerSchedulerAdapterError
from jlens_workspace.scheduler.jsonio import (
    identifier,
    integer,
    positive_number,
    strict_json_load,
    strict_object,
    timestamp,
)

COMMIT = re.compile(r"[0-9a-f]{40}")
INSTANCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
JOB_KEYS = {
    "schema_version",
    "job_id",
    "project",
    "task",
    "resources",
    "allocation",
    "parameters",
    "priority",
    "submitted_at",
    "state",
    "updated_at",
    "requested_profile",
    "execution_profile",
    "estimated_runtime_seconds",
    "gpu_indices",
    "gpu_uuids",
    "gpu_pci_bus_ids",
    "cpu_ids",
    "numa_node",
    "stdout_log",
    "stderr_log",
    "exit_code",
    "failure_reason",
}


@dataclass(frozen=True)
class RunningJob:
    """Validated scheduler job plus its concrete, environment-bound allocation."""

    job_id: str
    task: str
    spec: TaskSpec
    parameters: Mapping[str, object]
    requested_profile: str | None
    execution_profile: str
    estimated_runtime_seconds: float
    attempt: int
    allocation: Allocation
    cpu_ids: tuple[int, ...]
    numa_node: int | None
    gpu_indices: tuple[int, ...]
    gpu_uuids: tuple[str, ...]
    gpu_pci_bus_ids: tuple[str, ...]


def validate_parameters(
    spec: TaskSpec,
    payload: object,
    *,
    observed_commit: str,
) -> Mapping[str, object]:
    """Validate the exact allowlisted parameters for one task."""

    required = {"run_id", "git_commit", "shard_index", "shard_count"}
    if spec.driver == "k-diagnostic":
        required.update({"stage", "instance"})
    values = strict_object(payload, required, label=f"parameters for {spec.name}")
    if values["run_id"] != spec.run_id:
        raise ServerSchedulerAdapterError("parameters.run_id does not match the registered run")
    commit = values["git_commit"]
    if not isinstance(commit, str) or COMMIT.fullmatch(commit) is None:
        raise ServerSchedulerAdapterError("parameters.git_commit must be a lowercase 40-hex commit")
    if commit != observed_commit:
        raise ServerSchedulerAdapterError(
            f"J-lens checkout moved: manifest has {commit}, observed {observed_commit}"
        )
    shard_index = integer(values["shard_index"], label="parameters.shard_index")
    shard_count = integer(values["shard_count"], label="parameters.shard_count", minimum=1)
    if shard_index >= shard_count:
        raise ServerSchedulerAdapterError("parameters.shard_index must be below shard_count")
    if spec.shard_count is not None and (
        shard_count != spec.shard_count or shard_index >= spec.shard_count
    ):
        raise ServerSchedulerAdapterError("parameters shard identity does not match the task")
    if spec.driver == "k-diagnostic":
        stage = values["stage"]
        instance = values["instance"]
        if not isinstance(stage, str) or stage not in spec.stages:
            raise ServerSchedulerAdapterError("parameters.stage is incompatible with the task")
        if not isinstance(instance, str) or INSTANCE.fullmatch(instance) is None:
            raise ServerSchedulerAdapterError("parameters.instance is invalid")
        if spec.shard_count is None and instance != str(shard_index):
            raise ServerSchedulerAdapterError("K-diagnostic bundle instance must equal shard_index")
        if spec.name == "kdiag-resource-preflight":
            if re.fullmatch(r"(?:post_microbenchmark|wave_[0-9]+_[0-9]+)", instance) is None:
                raise ServerSchedulerAdapterError("resource-preflight instance is invalid")
        elif spec.shard_count is not None and instance != "singleton":
            raise ServerSchedulerAdapterError(
                "singleton K-diagnostic task has a non-singleton instance"
            )
    return values


def load_running_job(
    manifest_path: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    affinity: Set[int] | None = None,
    observed_commit: str | None = None,
) -> RunningJob:
    """Validate a protocol-v2 ServerScheduler running manifest and allocation."""

    values = os.environ if env is None else env
    path = Path(manifest_path)
    if not path.is_absolute():
        raise ServerSchedulerAdapterError("job manifest path must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ServerSchedulerAdapterError(f"job manifest is unreadable: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ServerSchedulerAdapterError("job manifest must be a regular non-symlink file")
    if metadata.st_size > 128 * 1024:
        raise ServerSchedulerAdapterError("job manifest is unexpectedly large")
    payload = strict_object(strict_json_load(path), JOB_KEYS, label="running job manifest")
    if payload["schema_version"] != 2 or payload["state"] != "running":
        raise ServerSchedulerAdapterError("job manifest is not a protocol-v2 running job")
    job_id = identifier(payload["job_id"], label="job_id")
    if payload["project"] != PROJECT:
        raise ServerSchedulerAdapterError("job manifest belongs to the wrong project")
    task = identifier(payload["task"], label="task")
    try:
        spec = TASK_SPECS[task]
    except KeyError as error:
        raise ServerSchedulerAdapterError(f"task {task!r} is not allowlisted") from error
    if payload["exit_code"] is not None or payload["failure_reason"] is not None:
        raise ServerSchedulerAdapterError("running job already contains terminal state")
    timestamp(payload["submitted_at"], label="submitted_at")
    timestamp(payload["updated_at"], label="updated_at")
    integer(payload["priority"], label="priority", minimum=-100, maximum=100)

    requested_profile_payload = payload["requested_profile"]
    requested_profile = (
        None
        if requested_profile_payload is None
        else identifier(requested_profile_payload, label="requested_profile")
    )
    execution_profile = identifier(payload["execution_profile"], label="execution_profile")
    if execution_profile != spec.profile.name:
        raise ServerSchedulerAdapterError(
            "selected execution_profile is incompatible with the registered task"
        )
    if requested_profile is not None and requested_profile != execution_profile:
        raise ServerSchedulerAdapterError(
            "requested_profile disagrees with the selected execution_profile"
        )
    estimated_runtime_seconds = positive_number(
        payload["estimated_runtime_seconds"], label="estimated_runtime_seconds"
    )

    resources_payload = payload["resources"]
    requested: Allocation | None = None
    if resources_payload is not None:
        requested = parse_resources(resources_payload, label="requested resources")
        spec.profile.validate(requested, label="requested resources")
    if payload["allocation"] is None:
        raise ServerSchedulerAdapterError("running job is missing its concrete allocation")
    allocation = parse_resources(payload["allocation"], label="allocation")
    spec.profile.validate(allocation, label="allocation")
    if requested is not None:
        require_allocation_satisfies_request(allocation, requested)
    current_commit = observed_commit or git_head_commit(CODE_ROOT)
    parameters = validate_parameters(spec, payload["parameters"], observed_commit=current_commit)

    cpu_ids, numa_node = parse_cpu_binding(
        payload["cpu_ids"], payload["numa_node"], allocation, affinity=affinity
    )
    gpu_indices, gpu_uuids, gpu_pci_bus_ids = parse_gpu_binding(
        payload["gpu_indices"],
        payload["gpu_uuids"],
        payload["gpu_pci_bus_ids"],
        allocation,
    )
    attempt = validate_scheduler_environment(
        values,
        manifest_path=path,
        job_id=job_id,
        execution_profile=execution_profile,
        estimated_runtime_seconds=estimated_runtime_seconds,
        allocation=allocation,
        cpu_ids=cpu_ids,
        numa_node=numa_node,
        gpu_indices=gpu_indices,
        gpu_uuids=gpu_uuids,
        gpu_pci_bus_ids=gpu_pci_bus_ids,
        stdout_log=payload["stdout_log"],
        stderr_log=payload["stderr_log"],
    )

    return RunningJob(
        job_id=job_id,
        task=task,
        spec=spec,
        parameters=parameters,
        requested_profile=requested_profile,
        execution_profile=execution_profile,
        estimated_runtime_seconds=estimated_runtime_seconds,
        attempt=attempt,
        allocation=allocation,
        cpu_ids=cpu_ids,
        numa_node=numa_node,
        gpu_indices=gpu_indices,
        gpu_uuids=gpu_uuids,
        gpu_pci_bus_ids=gpu_pci_bus_ids,
    )
