"""Concrete CPU, NUMA, memory, and GPU allocation validation."""

from __future__ import annotations

import os
from collections.abc import Set
from dataclasses import dataclass

from jlens_workspace.scheduler.errors import ServerSchedulerAdapterError
from jlens_workspace.scheduler.jsonio import integer, integer_tuple, strict_object, string_tuple

RESOURCE_KEYS = {
    "cpu_cores",
    "memory_mib",
    "gpu_count",
    "gpu_memory_mib",
    "gpu_utilization_pct",
    "exclusive_gpu",
}


@dataclass(frozen=True)
class Allocation:
    """Concrete v2 allocation; GPU reservations are per assigned device."""

    cpu_cores: int
    memory_mib: int
    gpu_count: int
    gpu_memory_mib: int
    gpu_utilization_pct: int
    exclusive_gpu: bool


def parse_resources(payload: object, *, label: str) -> Allocation:
    """Parse one strict requested-resource or concrete-allocation object."""

    values = strict_object(payload, RESOURCE_KEYS, label=label)
    exclusive = values["exclusive_gpu"]
    if not isinstance(exclusive, bool):
        raise ServerSchedulerAdapterError(f"{label}.exclusive_gpu must be boolean")
    return Allocation(
        cpu_cores=integer(values["cpu_cores"], label=f"{label}.cpu_cores", minimum=1),
        memory_mib=integer(values["memory_mib"], label=f"{label}.memory_mib", minimum=1),
        gpu_count=integer(values["gpu_count"], label=f"{label}.gpu_count", maximum=1),
        gpu_memory_mib=integer(values["gpu_memory_mib"], label=f"{label}.gpu_memory_mib"),
        gpu_utilization_pct=integer(
            values["gpu_utilization_pct"],
            label=f"{label}.gpu_utilization_pct",
            maximum=100,
        ),
        exclusive_gpu=exclusive,
    )


def require_allocation_satisfies_request(
    allocation: Allocation,
    requested: Allocation,
) -> None:
    """Require a concrete placement to honor every explicit request field."""

    if (
        allocation.cpu_cores != requested.cpu_cores
        or allocation.gpu_count != requested.gpu_count
        or allocation.memory_mib < requested.memory_mib
        or allocation.gpu_memory_mib < requested.gpu_memory_mib
        or allocation.gpu_utilization_pct < requested.gpu_utilization_pct
        or (requested.exclusive_gpu and not allocation.exclusive_gpu)
    ):
        raise ServerSchedulerAdapterError(
            "concrete allocation does not satisfy the requested resource override"
        )


def parse_cpu_binding(
    cpu_ids_payload: object,
    numa_node_payload: object,
    allocation: Allocation,
    *,
    affinity: Set[int] | None = None,
) -> tuple[tuple[int, ...], int | None]:
    """Validate physical CPU coverage, optional NUMA locality, and affinity."""

    cpu_ids = integer_tuple(cpu_ids_payload, label="cpu_ids")
    if len(cpu_ids) != allocation.cpu_cores:
        raise ServerSchedulerAdapterError(
            "cpu_ids must uniquely cover the allocated physical cores"
        )
    numa_node = None if numa_node_payload is None else integer(numa_node_payload, label="numa_node")
    observed_affinity = affinity
    if observed_affinity is None and hasattr(os, "sched_getaffinity"):
        observed_affinity = set(os.sched_getaffinity(0))
    if observed_affinity is not None and not set(cpu_ids).issubset(observed_affinity):
        raise ServerSchedulerAdapterError("process affinity does not include the allocated cpu_ids")
    return cpu_ids, numa_node


def parse_gpu_binding(
    gpu_indices_payload: object,
    gpu_uuids_payload: object,
    gpu_pci_bus_ids_payload: object,
    allocation: Allocation,
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[str, ...]]:
    """Validate plural protocol-v2 GPU identity arrays."""

    gpu_indices = integer_tuple(gpu_indices_payload, label="gpu_indices")
    gpu_uuids = string_tuple(gpu_uuids_payload, label="gpu_uuids")
    gpu_pci_bus_ids = string_tuple(gpu_pci_bus_ids_payload, label="gpu_pci_bus_ids")
    if len(gpu_indices) != allocation.gpu_count:
        raise ServerSchedulerAdapterError("gpu_indices length disagrees with GPU count")
    if gpu_uuids and len(gpu_uuids) != allocation.gpu_count:
        raise ServerSchedulerAdapterError("gpu_uuids length disagrees with GPU count")
    if gpu_pci_bus_ids and len(gpu_pci_bus_ids) != allocation.gpu_count:
        raise ServerSchedulerAdapterError("gpu_pci_bus_ids length disagrees with GPU count")
    return gpu_indices, gpu_uuids, gpu_pci_bus_ids
