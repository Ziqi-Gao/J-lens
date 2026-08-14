"""Capacity-aware local resource scheduler for the three-method DAG.

This module deliberately uses only the Python standard library.  It runs before
the scientific task environment is activated and coordinates J-lens processes
with advisory leases under the J-lens scratch tree.  Foreign processes are
observed but are never signalled or reconfigured.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO

GPU_TASKS = frozenset(
    {
        "lens",
        "capture",
        "occupancy",
        "iti-capture",
        "j-grid",
        "raptor-grid",
        "iti-grid",
        "candidate-rescore",
    }
)


class SchedulerError(RuntimeError):
    """Raised when local resource discovery or admission fails closed."""


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, str(default))
    if not re.fullmatch(r"[1-9][0-9]*", raw):
        raise SchedulerError(f"{name} must be positive integer")
    return int(raw)


def _nonnegative_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, str(default))
    if not re.fullmatch(r"[0-9]+", raw):
        raise SchedulerError(f"{name} must be a non-negative integer")
    return int(raw)


def _boolean(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name, "1" if default else "0")
    if raw not in {"0", "1"}:
        raise SchedulerError(f"{name} must be 0 or 1")
    return raw == "1"


@dataclass(frozen=True)
class TaskProfile:
    task_class: str
    lock_mode: str
    slots_per_device: int
    cpu_tokens: int
    host_ram_mib: int
    gpu_vram_mib: int
    gpu_memory_reserve_mib: int
    gpu_utilization_tokens: int


def profile_for_task(task: str, env: Mapping[str, str] | None = None) -> TaskProfile:
    """Return the conservative resource reservation for one registered task."""

    values = os.environ if env is None else env
    base_gpu_memory_reserve = _nonnegative_int(values, "JLENS_LOCAL_GPU_MEMORY_RESERVE_MIB", 8192)
    if task == "occupancy":
        return TaskProfile(
            task_class="occupancy",
            lock_mode="shared",
            slots_per_device=_positive_int(values, "JLENS_LOCAL_OCCUPANCY_GPU_SLOTS_PER_DEVICE", 5),
            cpu_tokens=_positive_int(values, "JLENS_LOCAL_OCCUPANCY_CPU_TOKENS", 4),
            host_ram_mib=_positive_int(values, "JLENS_LOCAL_OCCUPANCY_HOST_RAM_MIB", 4096),
            gpu_vram_mib=_positive_int(values, "JLENS_LOCAL_OCCUPANCY_GPU_VRAM_MIB", 2048),
            gpu_memory_reserve_mib=_nonnegative_int(
                values,
                "JLENS_LOCAL_OCCUPANCY_GPU_MEMORY_RESERVE_MIB",
                base_gpu_memory_reserve,
            ),
            gpu_utilization_tokens=_positive_int(
                values, "JLENS_LOCAL_OCCUPANCY_GPU_UTILIZATION_TOKENS", 20
            ),
        )
    if task in GPU_TASKS:
        return TaskProfile(
            task_class="standard",
            lock_mode="exclusive",
            slots_per_device=1,
            cpu_tokens=_positive_int(values, "JLENS_LOCAL_STANDARD_CPU_TOKENS", 4),
            host_ram_mib=_positive_int(values, "JLENS_LOCAL_STANDARD_HOST_RAM_MIB", 24576),
            gpu_vram_mib=_positive_int(values, "JLENS_LOCAL_STANDARD_GPU_VRAM_MIB", 24576),
            gpu_memory_reserve_mib=_nonnegative_int(
                values,
                "JLENS_LOCAL_STANDARD_GPU_MEMORY_RESERVE_MIB",
                max(base_gpu_memory_reserve, 16384),
            ),
            gpu_utilization_tokens=_positive_int(
                values, "JLENS_LOCAL_STANDARD_GPU_UTILIZATION_TOKENS", 40
            ),
        )

    cpu_tokens = 4
    host_ram_mib = 8192
    if task in {"layer-selection", "iti-fit"}:
        cpu_tokens = 16
        host_ram_mib = 32768
    elif task == "bootstrap-probes":
        cpu_tokens = 8
        host_ram_mib = 16384
    return TaskProfile(
        task_class="cpu",
        lock_mode="none",
        slots_per_device=0,
        cpu_tokens=_positive_int(values, "JLENS_LOCAL_CPU_TASK_TOKENS", cpu_tokens),
        host_ram_mib=_positive_int(values, "JLENS_LOCAL_CPU_TASK_HOST_RAM_MIB", host_ram_mib),
        gpu_vram_mib=0,
        gpu_memory_reserve_mib=0,
        gpu_utilization_tokens=0,
    )


@dataclass(frozen=True)
class ComputeProcess:
    pid: int
    process_name: str
    used_memory_mib: int
    owner: str


@dataclass(frozen=True)
class GPUObservation:
    index: int
    total_memory_mib: int
    minimum_free_memory_mib: int
    maximum_utilization: int
    processes: tuple[ComputeProcess, ...]

    @property
    def foreign_processes(self) -> tuple[ComputeProcess, ...]:
        return tuple(process for process in self.processes if process.owner == "foreign")

    @property
    def foreign_memory_mib(self) -> int:
        return sum(process.used_memory_mib for process in self.foreign_processes)


@dataclass(frozen=True)
class Lease:
    lease_id: str
    owner_pid: int
    child_pid: int | None
    task: str
    kind: str
    gpu_index: int | None
    cpu_tokens: int
    host_ram_mib: int
    gpu_vram_mib: int
    gpu_utilization_tokens: int
    created_at: str


@dataclass(frozen=True)
class Candidate:
    observation: GPUObservation
    score: tuple[int, int, int, int]
    effective_utilization_after: int
    reserved_gpu_vram_mib_after: int


def _run_text(command: Sequence[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def discover_gpu_ids(
    env: Mapping[str, str] | None = None,
    *,
    nvidia_smi: str = "nvidia-smi",
) -> list[int]:
    """Discover devices dynamically, optionally applying an operator allow-list."""

    values = os.environ if env is None else env
    output = _run_text([nvidia_smi, "--query-gpu=index", "--format=csv,noheader,nounits"])
    discovered = [int(line.strip()) for line in output.splitlines() if line.strip()]
    requested = values.get("JLENS_LOCAL_GPU_IDS", "").strip()
    if not requested:
        return discovered
    allow = []
    for raw in requested.split(","):
        value = raw.strip()
        if not re.fullmatch(r"[0-9]+", value):
            raise SchedulerError(f"invalid GPU id in JLENS_LOCAL_GPU_IDS: {value}")
        allow.append(int(value))
    missing = sorted(set(allow) - set(discovered))
    if missing:
        raise SchedulerError(f"JLENS_LOCAL_GPU_IDS contains unavailable devices: {missing}")
    return [index for index in discovered if index in set(allow)]


def classify_process(
    pid: int,
    process_name: str,
    *,
    code_root: Path,
    data_root: Path,
    scratch_root: Path,
) -> str:
    """Classify a compute PID without modifying or debugging the process."""

    roots = tuple(str(path.resolve()) for path in (code_root, data_root, scratch_root))
    haystacks = [process_name]
    try:
        haystacks.append(Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode())
    except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
        pass
    return "jlens" if any(root in text for root in roots for text in haystacks) else "foreign"


def _parse_gpu_rows(output: str) -> dict[int, tuple[int, int, int]]:
    rows: dict[int, tuple[int, int, int]] = {}
    for row in csv.reader(output.splitlines()):
        if not row:
            continue
        index, free, total, utilization = (int(value.strip()) for value in row)
        rows[index] = (free, total, utilization)
    return rows


def observe_gpus(
    gpu_ids: Sequence[int],
    *,
    samples: int,
    sample_interval_seconds: int,
    nvidia_smi: str,
    code_root: Path,
    data_root: Path,
    scratch_root: Path,
) -> dict[int, GPUObservation]:
    """Take a rolling min-free/max-utilization admission snapshot."""

    aggregate: dict[int, tuple[int, int, int]] = {}
    for sample in range(samples):
        output = _run_text(
            [
                nvidia_smi,
                "--query-gpu=index,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        for index, (free, total, utilization) in _parse_gpu_rows(output).items():
            if index not in gpu_ids:
                continue
            if index not in aggregate:
                aggregate[index] = (free, total, utilization)
            else:
                old_free, old_total, old_utilization = aggregate[index]
                aggregate[index] = (
                    min(old_free, free),
                    old_total,
                    max(old_utilization, utilization),
                )
        if sample + 1 < samples:
            time.sleep(sample_interval_seconds)

    observations: dict[int, GPUObservation] = {}
    for index in gpu_ids:
        if index not in aggregate:
            continue
        process_output = _run_text(
            [
                nvidia_smi,
                "-i",
                str(index),
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        processes = []
        for row in csv.reader(process_output.splitlines()):
            if not row:
                continue
            pid = int(row[0].strip())
            name = row[1].strip()
            used_memory = int(row[2].strip())
            processes.append(
                ComputeProcess(
                    pid=pid,
                    process_name=name,
                    used_memory_mib=used_memory,
                    owner=classify_process(
                        pid,
                        name,
                        code_root=code_root,
                        data_root=data_root,
                        scratch_root=scratch_root,
                    ),
                )
            )
        free, total, utilization = aggregate[index]
        observations[index] = GPUObservation(
            index=index,
            total_memory_mib=total,
            minimum_free_memory_mib=free,
            maximum_utilization=utilization,
            processes=tuple(processes),
        )
    return observations


def candidate_for_gpu(
    observation: GPUObservation,
    profile: TaskProfile,
    leases: Sequence[Lease],
    *,
    allow_overlay: bool,
    memory_reserve_mib: int,
    maximum_start_utilization: int,
    target_utilization: int,
) -> Candidate | None:
    """Apply memory, compute, ownership, and task-class admission rules."""

    active = [lease for lease in leases if lease.gpu_index == observation.index]
    if not allow_overlay and observation.foreign_processes:
        return None
    if observation.maximum_utilization > maximum_start_utilization:
        return None
    if observation.minimum_free_memory_mib < profile.gpu_vram_mib + memory_reserve_mib:
        return None

    reserved_memory = sum(lease.gpu_vram_mib for lease in active)
    reserved_after = reserved_memory + profile.gpu_vram_mib
    projected_free = observation.total_memory_mib - observation.foreign_memory_mib - reserved_after
    if projected_free < memory_reserve_mib:
        return None

    reserved_utilization = sum(lease.gpu_utilization_tokens for lease in active)
    if observation.foreign_processes:
        effective_before = observation.maximum_utilization + reserved_utilization
    else:
        effective_before = max(observation.maximum_utilization, reserved_utilization)
    effective_after = effective_before + profile.gpu_utilization_tokens
    if effective_after > target_utilization:
        return None

    # Prefer clean GPUs, then balance projected compute and memory pressure.
    score = (
        int(bool(observation.foreign_processes)),
        effective_after,
        reserved_after,
        -observation.minimum_free_memory_mib,
    )
    return Candidate(
        observation=observation,
        score=score,
        effective_utilization_after=effective_after,
        reserved_gpu_vram_mib_after=reserved_after,
    )


def rank_candidates(
    observations: Mapping[int, GPUObservation],
    profile: TaskProfile,
    leases: Sequence[Lease],
    *,
    allow_overlay: bool,
    memory_reserve_mib: int,
    maximum_start_utilization: int,
    target_utilization: int,
) -> list[Candidate]:
    candidates = [
        candidate
        for observation in observations.values()
        if (
            candidate := candidate_for_gpu(
                observation,
                profile,
                leases,
                allow_overlay=allow_overlay,
                memory_reserve_mib=memory_reserve_mib,
                maximum_start_utilization=maximum_start_utilization,
                target_utilization=target_utilization,
            )
        )
        is not None
    ]
    return sorted(candidates, key=lambda candidate: candidate.score)


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _load_leases(lease_dir: Path) -> list[Lease]:
    leases = []
    lease_dir.mkdir(parents=True, exist_ok=True)
    for path in lease_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text())
            lease = Lease(**payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            path.unlink(missing_ok=True)
            continue
        if _pid_alive(lease.owner_pid) or _pid_alive(lease.child_pid):
            leases.append(lease)
        else:
            path.unlink(missing_ok=True)
    return leases


def _write_lease(path: Path, lease: Lease) -> None:
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(asdict(lease), sort_keys=True) + "\n")
    os.replace(temporary, path)


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[IO[str]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield handle


def _try_lock(path: Path, *, shared: bool) -> IO[str] | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    try:
        fcntl.flock(handle.fileno(), operation | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _physical_core_count() -> int:
    try:
        output = _run_text(["lscpu", "-p=core,socket"])
        cores = {
            line.strip()
            for line in output.splitlines()
            if line.strip() and not line.startswith("#")
        }
        if cores:
            return len(cores)
    except (OSError, subprocess.CalledProcessError):
        pass
    return os.cpu_count() or 1


def _memory_mib() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = int(raw.strip().split()[0]) // 1024
    return values["MemTotal"], values["MemAvailable"]


def _host_capacity_ok(
    leases: Sequence[Lease],
    profile: TaskProfile,
    env: Mapping[str, str],
) -> bool:
    cpu_capacity = _positive_int(env, "JLENS_LOCAL_HOST_CPU_TOKENS", _physical_core_count())
    memory_reserve = _nonnegative_int(env, "JLENS_LOCAL_HOST_RAM_RESERVE_MIB", 65536)
    total_memory, available_memory = _memory_mib()
    cpu_reserved = sum(lease.cpu_tokens for lease in leases)
    memory_reserved = sum(lease.host_ram_mib for lease in leases)
    return (
        cpu_reserved + profile.cpu_tokens <= cpu_capacity
        and memory_reserved + profile.host_ram_mib <= total_memory - memory_reserve
        and available_memory >= profile.host_ram_mib + memory_reserve
    )


class LocalScheduler:
    def __init__(self, task: str, kind: str, command: Sequence[str]) -> None:
        self.env = os.environ.copy()
        self.task = task
        self.kind = kind
        self.command = list(command)
        self.profile = profile_for_task(task, self.env)
        self.runtime_root = Path(self.env["JLENS_LOCAL_RUNTIME_ROOT"])
        self.code_root = Path(self.env["CODE_ROOT"])
        self.data_root = Path(self.env["JLENS_LOCAL_DATA_ROOT"])
        self.scratch_root = Path(self.env["JLENS_LOCAL_SCRATCH_ROOT"])
        self.scheduler_root = self.runtime_root / "scheduler"
        self.lease_dir = self.scheduler_root / "leases"
        self.broker_lock = self.scheduler_root / "broker.lock"
        self.lease_path: Path | None = None
        self.gate_handle: IO[str] | None = None
        self.slot_handle: IO[str] | None = None

    def _new_lease(self, gpu_index: int | None) -> Lease:
        return Lease(
            lease_id=f"{os.getpid()}-{uuid.uuid4().hex}",
            owner_pid=os.getpid(),
            child_pid=None,
            task=self.task,
            kind=self.kind,
            gpu_index=gpu_index,
            cpu_tokens=self.profile.cpu_tokens,
            host_ram_mib=self.profile.host_ram_mib,
            gpu_vram_mib=self.profile.gpu_vram_mib,
            gpu_utilization_tokens=self.profile.gpu_utilization_tokens,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    def _reserve_host(self, gpu_index: int | None) -> Lease | None:
        with _exclusive_file_lock(self.broker_lock):
            leases = _load_leases(self.lease_dir)
            if not _host_capacity_ok(leases, self.profile, self.env):
                return None
            lease = self._new_lease(gpu_index)
            self.lease_path = self.lease_dir / f"{lease.lease_id}.json"
            _write_lease(self.lease_path, lease)
            return lease

    def _reserve_gpu(
        self,
        observation: GPUObservation,
        *,
        allow_overlay: bool,
        memory_reserve_mib: int,
        maximum_start_utilization: int,
        target_utilization: int,
    ) -> tuple[Lease, Candidate] | None:
        """Atomically recheck host/GPU tokens and create a joint lease."""

        with _exclusive_file_lock(self.broker_lock):
            leases = _load_leases(self.lease_dir)
            admitted = candidate_for_gpu(
                observation,
                self.profile,
                leases,
                allow_overlay=allow_overlay,
                memory_reserve_mib=memory_reserve_mib,
                maximum_start_utilization=maximum_start_utilization,
                target_utilization=target_utilization,
            )
            if admitted is None or not _host_capacity_ok(leases, self.profile, self.env):
                return None
            lease = self._new_lease(observation.index)
            self.lease_path = self.lease_dir / f"{lease.lease_id}.json"
            _write_lease(self.lease_path, lease)
            return lease, admitted

    def _release(self) -> None:
        if self.lease_path is not None:
            with _exclusive_file_lock(self.broker_lock):
                self.lease_path.unlink(missing_ok=True)
            self.lease_path = None
        for handle_name in ("slot_handle", "gate_handle"):
            handle = getattr(self, handle_name)
            if handle is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
                setattr(self, handle_name, None)

    def _run_child(self, lease: Lease, *, gpu_index: int | None) -> int:
        child_environment = self.env.copy()
        child_environment["JLENS_LOCAL_LEASE_ID"] = lease.lease_id
        if gpu_index is not None:
            child_environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            child_environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        lock_descriptors = tuple(
            handle.fileno() for handle in (self.gate_handle, self.slot_handle) if handle is not None
        )
        child = subprocess.Popen(self.command, env=child_environment, pass_fds=lock_descriptors)
        updated = Lease(**{**asdict(lease), "child_pid": child.pid})
        if self.lease_path is not None:
            _write_lease(self.lease_path, updated)
        return child.wait()

    def run_cpu(self) -> int:
        poll = _positive_int(self.env, "JLENS_LOCAL_RESOURCE_POLL_SECONDS", 10)
        while True:
            lease = self._reserve_host(None)
            if lease is not None:
                print(
                    f"jlens_host_reserved cpu_tokens={lease.cpu_tokens} "
                    f"host_ram_mib={lease.host_ram_mib} task_class={self.profile.task_class}",
                    flush=True,
                )
                try:
                    return self._run_child(lease, gpu_index=None)
                finally:
                    self._release()
            print("waiting_for_jlens_host_capacity", file=sys.stderr, flush=True)
            time.sleep(poll)

    def _active_leases(self) -> list[Lease]:
        with _exclusive_file_lock(self.broker_lock):
            return _load_leases(self.lease_dir)

    def _try_gpu_locks(self, gpu_index: int) -> bool:
        lock_root = self.runtime_root / "gpu-locks"
        gate = _try_lock(
            lock_root / f"gpu_{gpu_index}.lock",
            shared=self.profile.lock_mode == "shared",
        )
        if gate is None:
            return False
        self.gate_handle = gate
        if self.profile.task_class != "occupancy":
            return True
        for slot in range(self.profile.slots_per_device):
            handle = _try_lock(
                lock_root / f"gpu_{gpu_index}_occupancy_slot_{slot}.lock",
                shared=False,
            )
            if handle is not None:
                self.slot_handle = handle
                return True
        self._release()
        return False

    def run_gpu(self) -> int:
        nvidia_smi = self.env.get("JLENS_LOCAL_NVIDIA_SMI", "nvidia-smi")
        gpu_ids = discover_gpu_ids(self.env, nvidia_smi=nvidia_smi)
        if not gpu_ids:
            raise SchedulerError("no candidate GPU is visible")
        samples = _positive_int(self.env, "JLENS_LOCAL_GPU_STABILITY_SAMPLES", 3)
        sample_interval = _nonnegative_int(self.env, "JLENS_LOCAL_GPU_SAMPLE_INTERVAL_SECONDS", 2)
        poll = _positive_int(self.env, "JLENS_LOCAL_GPU_POLL_SECONDS", 10)
        timeout = _nonnegative_int(self.env, "JLENS_LOCAL_GPU_WAIT_TIMEOUT_SECONDS", 0)
        allow_overlay = _boolean(self.env, "JLENS_LOCAL_GPU_ALLOW_OVERLAY", True)
        memory_reserve = self.profile.gpu_memory_reserve_mib
        maximum_start = _nonnegative_int(self.env, "JLENS_LOCAL_GPU_MAX_UTILIZATION", 85)
        target_utilization = _positive_int(self.env, "JLENS_LOCAL_GPU_TARGET_UTILIZATION", 95)
        if maximum_start > 100 or target_utilization > 100:
            raise SchedulerError("GPU utilization thresholds must lie in [0, 100]")

        started = time.monotonic()
        while True:
            observations = observe_gpus(
                gpu_ids,
                samples=samples,
                sample_interval_seconds=sample_interval,
                nvidia_smi=nvidia_smi,
                code_root=self.code_root,
                data_root=self.data_root,
                scratch_root=self.scratch_root,
            )
            selected = None
            admission_lock = self.scheduler_root / "gpu-admission.lock"
            with _exclusive_file_lock(admission_lock):
                candidates = rank_candidates(
                    observations,
                    self.profile,
                    self._active_leases(),
                    allow_overlay=allow_overlay,
                    memory_reserve_mib=memory_reserve,
                    maximum_start_utilization=maximum_start,
                    target_utilization=target_utilization,
                )
                for candidate in candidates:
                    gpu_index = candidate.observation.index
                    if not self._try_gpu_locks(gpu_index):
                        continue
                    # Revalidate both host and selected GPU after taking locks.
                    refreshed = observe_gpus(
                        [gpu_index],
                        samples=1,
                        sample_interval_seconds=0,
                        nvidia_smi=nvidia_smi,
                        code_root=self.code_root,
                        data_root=self.data_root,
                        scratch_root=self.scratch_root,
                    )[gpu_index]
                    reservation = self._reserve_gpu(
                        refreshed,
                        allow_overlay=allow_overlay,
                        memory_reserve_mib=memory_reserve,
                        maximum_start_utilization=maximum_start,
                        target_utilization=target_utilization,
                    )
                    if reservation is None:
                        self._release()
                        continue
                    lease, admitted = reservation
                    selected = (gpu_index, refreshed, lease, admitted)
                    break

            if selected is not None:
                gpu_index, refreshed, lease, admitted = selected
                print(
                    f"jlens_gpu_selected={gpu_index} "
                    f"free_mib={refreshed.minimum_free_memory_mib} "
                    f"utilization={refreshed.maximum_utilization} "
                    f"overlay={str(bool(refreshed.foreign_processes)).lower()} "
                    f"task_class={self.profile.task_class} "
                    f"effective_utilization_after={admitted.effective_utilization_after} "
                    f"reserved_gpu_vram_mib_after={admitted.reserved_gpu_vram_mib_after}",
                    flush=True,
                )
                if refreshed.processes:
                    print("preexisting_compute_processes:", flush=True)
                    for process in refreshed.processes:
                        print(
                            f"pid={process.pid} owner={process.owner} "
                            f"used_memory_mib={process.used_memory_mib} "
                            f"process_name={process.process_name}",
                            flush=True,
                        )
                try:
                    return self._run_child(lease, gpu_index=gpu_index)
                finally:
                    self._release()

            if timeout and time.monotonic() - started >= timeout:
                raise SchedulerError("no GPU met J-lens capacity thresholds before timeout")
            print(
                f"waiting_for_jlens_gpu_capacity gpu_ids={','.join(map(str, gpu_ids))} "
                f"memory_reserve_mib={memory_reserve} "
                f"max_utilization={maximum_start}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(poll)

    def run(self) -> int:
        if not self.command:
            raise SchedulerError("scheduled command is empty")
        if self.kind == "gpu":
            if self.task not in GPU_TASKS:
                raise SchedulerError(f"task {self.task!r} is not registered as a GPU task")
            return self.run_gpu()
        if self.kind == "cpu":
            return self.run_cpu()
        raise SchedulerError("kind must be cpu or gpu")


def _describe(task: str) -> int:
    profile = profile_for_task(task)
    for key, value in asdict(profile).items():
        print(f"{key}={value}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    describe = subparsers.add_parser("describe")
    describe.add_argument("task")
    run = subparsers.add_parser("run")
    run.add_argument("--kind", choices=("cpu", "gpu"), required=True)
    run.add_argument("--task", required=True)
    run.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.action == "describe":
            return _describe(arguments.task)
        command = arguments.command
        if command and command[0] == "--":
            command = command[1:]
        return LocalScheduler(arguments.task, arguments.kind, command).run()
    except (OSError, subprocess.CalledProcessError, SchedulerError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
