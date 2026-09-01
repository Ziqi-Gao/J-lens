"""Fixed command planning and foreground ServerScheduler task execution."""

from __future__ import annotations

import os
import stat
import subprocess
from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass
from pathlib import Path

from jlens_workspace.scheduler.catalog import (
    CODE_ROOT,
    COMPLETION_ROOT,
    DATA_ROOT,
    KDIAG_RUN_ID,
    KDIAG_RUN_ROOT,
    KDIAG_STAGE_DIRECTORY,
    PROJECT_PYTHON,
    SCRATCH_ROOT,
    THREE_METHOD_RUN_ROOT,
)
from jlens_workspace.scheduler.completion import (
    CompletionEvidence,
    completion_receipt_path,
    derivation_output_root,
    validate_completion_receipt,
    validate_task_completion,
    write_completion_receipt,
)
from jlens_workspace.scheduler.errors import ServerSchedulerAdapterError
from jlens_workspace.scheduler.jsonio import integer, strict_json_load, strict_object
from jlens_workspace.scheduler.manifest import RunningJob, load_running_job


@dataclass(frozen=True)
class DispatchPlan:
    """Exact fixed command, working directory, environment, and validated job."""

    command: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    job: RunningJob


def validate_owned_directory(
    path: Path,
    *,
    owned_root: Path,
    label: str,
    must_exist: bool,
) -> Path:
    """Reject lexical escapes and symlink components before any task write."""

    try:
        root_metadata = owned_root.lstat()
        root_resolved = owned_root.resolve(strict=True)
    except OSError as error:
        raise ServerSchedulerAdapterError(f"{label} owned root is unavailable") from error
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ServerSchedulerAdapterError(f"{label} owned root must be a non-symlink directory")

    absolute_root = Path(os.path.abspath(owned_root))
    absolute_path = Path(os.path.abspath(path))
    try:
        relative = absolute_path.relative_to(absolute_root)
    except ValueError as error:
        raise ServerSchedulerAdapterError(f"{label} is outside its project-owned root") from error

    current = absolute_root
    missing = False
    for part in relative.parts:
        current /= part
        if missing:
            continue
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            missing = True
            continue
        except OSError as error:
            raise ServerSchedulerAdapterError(f"cannot inspect {label}: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ServerSchedulerAdapterError(f"{label} contains a symlink component")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ServerSchedulerAdapterError(f"{label} must be a directory path")

    if must_exist and missing:
        raise ServerSchedulerAdapterError(f"{label} does not exist")
    try:
        resolved = absolute_path.resolve(strict=must_exist)
        resolved.relative_to(root_resolved)
    except (OSError, ValueError) as error:
        raise ServerSchedulerAdapterError(f"{label} escapes its project-owned root") from error
    return absolute_path


def require_clean_checkout(checkout: Path = CODE_ROOT) -> None:
    """Reject dispatch from tracked or untracked code outside the recorded commit."""

    try:
        completed = subprocess.run(
            [
                "/usr/bin/git",
                "--no-optional-locks",
                "-C",
                str(checkout),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--no-ahead-behind",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise ServerSchedulerAdapterError(
            f"cannot verify the registered J-lens checkout: {error}"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"git status exited {completed.returncode}"
        raise ServerSchedulerAdapterError(f"cannot verify the registered J-lens checkout: {detail}")
    changes = completed.stdout.splitlines()
    if changes:
        preview = ", ".join(line[3:] if len(line) > 3 else line for line in changes[:5])
        suffix = "" if len(changes) <= 5 else f" and {len(changes) - 5} more"
        raise ServerSchedulerAdapterError(
            "registered J-lens checkout is not clean; commit or isolate changes "
            f"before dispatch ({preview}{suffix})"
        )


def validate_kdiag_bundle_plan(
    job: RunningJob,
    *,
    run_root: Path = KDIAG_RUN_ROOT,
) -> None:
    """Bind dynamic K-diagnostic bundle shards to the frozen bundle plan."""

    stage = str(job.parameters["stage"])
    directory = KDIAG_STAGE_DIRECTORY[stage]
    path = run_root / "artifacts/concept_intervention" / KDIAG_RUN_ID / directory / "bundles.json"
    payload = strict_object(
        strict_json_load(path),
        {
            "schema_version",
            "stage",
            "logical_replicate_count",
            "physical_bundle_count",
            "bundles",
        },
        label="K-diagnostic bundle plan",
    )
    if payload["schema_version"] != 2 or payload["stage"] != directory:
        raise ServerSchedulerAdapterError("K-diagnostic bundle plan identity is malformed")
    physical_count = integer(
        payload["physical_bundle_count"],
        label="K-diagnostic physical_bundle_count",
        minimum=1,
    )
    logical_count = integer(
        payload["logical_replicate_count"],
        label="K-diagnostic logical_replicate_count",
        minimum=1,
    )
    bundles = payload["bundles"]
    if not isinstance(bundles, list) or not bundles or len(bundles) != physical_count:
        raise ServerSchedulerAdapterError("K-diagnostic bundle plan is empty")
    bundle_ids: set[str] = set()
    shard_ids: set[str] = set()
    observed_logical_count = 0
    for bundle in bundles:
        if not isinstance(bundle, Mapping):
            raise ServerSchedulerAdapterError("K-diagnostic bundle entry is malformed")
        required = {
            "bundle_id",
            "conservative_cost_units",
            "logical_shard_count",
            "shard_ids",
        }
        if not required.issubset(bundle):
            raise ServerSchedulerAdapterError("K-diagnostic bundle entry is malformed")
        bundle_id = bundle["bundle_id"]
        if not isinstance(bundle_id, str) or not bundle_id or bundle_id in bundle_ids:
            raise ServerSchedulerAdapterError("K-diagnostic bundle identity is malformed")
        bundle_ids.add(bundle_id)
        integer(
            bundle["conservative_cost_units"],
            label="K-diagnostic conservative_cost_units",
            minimum=1,
        )
        registered_shards = bundle["shard_ids"]
        registered_count = integer(
            bundle["logical_shard_count"],
            label="K-diagnostic logical_shard_count",
            minimum=1,
        )
        if (
            not isinstance(registered_shards, list)
            or len(registered_shards) != registered_count
            or any(not isinstance(item, str) or not item for item in registered_shards)
            or len(set(registered_shards)) != registered_count
            or shard_ids.intersection(registered_shards)
        ):
            raise ServerSchedulerAdapterError("K-diagnostic logical shard plan is malformed")
        shard_ids.update(registered_shards)
        observed_logical_count += registered_count
    if observed_logical_count != logical_count:
        raise ServerSchedulerAdapterError("K-diagnostic logical shard count is malformed")
    count = int(job.parameters["shard_count"])
    index = int(job.parameters["shard_index"])
    if count != len(bundles) or index >= len(bundles):
        raise ServerSchedulerAdapterError("K-diagnostic shard identity disagrees with bundles.json")
    if job.task == "kdiag-microbenchmark":
        try:
            costs = [int(item["conservative_cost_units"]) for item in bundles]
        except (KeyError, TypeError, ValueError) as error:
            raise ServerSchedulerAdapterError("K-diagnostic bundle costs are malformed") from error
        if index != max(range(len(costs)), key=costs.__getitem__):
            raise ServerSchedulerAdapterError("microbenchmark is not the registered worst bundle")


def build_dispatch_plan(
    manifest_path: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    affinity: Set[int] | None = None,
    observed_commit: str | None = None,
    verify_inputs: bool = True,
    verify_checkout: bool = True,
) -> DispatchPlan:
    """Return the exact fixed task command after validating a v2 manifest."""

    source_env = dict(os.environ if env is None else env)
    job = load_running_job(
        manifest_path,
        env=source_env,
        affinity=affinity,
        observed_commit=observed_commit,
    )
    if verify_checkout:
        require_clean_checkout()
    if verify_inputs and job.spec.shard_count is None:
        validate_kdiag_bundle_plan(job)

    child_env = source_env.copy()
    scheduling_keys = {
        "JLENS_LOCAL_LEASE_ID",
        "JLENS_LOCAL_NVIDIA_SMI",
        "JLENS_LOCAL_GPU_IDS",
        "JLENS_LOCAL_DAG_ACTIVE",
        "JLENS_LOCAL_KDIAG_DAG_ACTIVE",
        "JLENS_LOCAL_KDIAG_COMMIT",
        "SLURM_JOB_ID",
        "SLURM_CPUS_PER_TASK",
        "SLURM_ARRAY_TASK_ID",
        "OCCUPANCY_OVERWRITE",
        "THREE_METHOD_LOCAL_ENV_LOADED",
        "THREE_METHOD_LOCAL_PATHS_LOADED",
    }
    for key in scheduling_keys:
        child_env.pop(key, None)
    child_env.update(
        {
            "CODE_ROOT": str(CODE_ROOT),
            "JLENS_LOCAL_DATA_ROOT": str(DATA_ROOT),
            "JLENS_LOCAL_SCRATCH_ROOT": str(SCRATCH_ROOT),
            "JLENS_ENV_ROOT": str(PROJECT_PYTHON.parent.parent),
            "JLENS_PYTHON": str(PROJECT_PYTHON),
            "JLENS_GIT_COMMIT": str(job.parameters["git_commit"]),
            "JLENS_CPUS_PER_TASK": str(job.allocation.cpu_cores),
            "JLENS_SERVER_SCHEDULER_TASK_ACTIVE": "1",
            "JLENS_SERVER_SCHEDULER_ATTEMPT": str(job.attempt),
            "HF_HOME": str(SCRATCH_ROOT / "cache/huggingface"),
            "HF_HUB_CACHE": str(SCRATCH_ROOT / "cache/huggingface/hub"),
            "HF_DATASETS_CACHE": str(SCRATCH_ROOT / "cache/huggingface/datasets"),
            "TMPDIR": str(SCRATCH_ROOT / "tmp/three-method"),
            "OMP_NUM_THREADS": str(job.allocation.cpu_cores),
            "MKL_NUM_THREADS": str(job.allocation.cpu_cores),
            "OPENBLAS_NUM_THREADS": str(job.allocation.cpu_cores),
            "NUMEXPR_NUM_THREADS": str(job.allocation.cpu_cores),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": "42",
        }
    )

    shard_index = int(job.parameters["shard_index"])
    if job.spec.driver == "three-method":
        script = (
            CODE_ROOT
            / "Concept_intervention/scripts/server_scheduler_three_method_task.sh"
        )
        command = ["/usr/bin/bash", str(script), job.spec.action]
        if job.spec.shard_count != 1:
            command.append(str(shard_index))
        run_root = THREE_METHOD_RUN_ROOT
    else:
        script = (
            CODE_ROOT / "Concept_intervention/scripts/server_scheduler_kdiag_task.sh"
        )
        command = [
            "/usr/bin/bash",
            str(script),
            str(job.spec.kdiag_profile),
            job.spec.action,
            str(job.parameters["stage"]),
            str(job.parameters["instance"]),
        ]
        run_root = KDIAG_RUN_ROOT
    child_env["RUN_ROOT"] = str(run_root)
    return DispatchPlan(tuple(command), run_root, child_env, job)


def execute_dispatch_plan(
    plan: DispatchPlan,
    *,
    completion_root: Path = COMPLETION_ROOT,
    completion_validator: Callable[[RunningJob], tuple[CompletionEvidence, ...]] = (
        validate_task_completion
    ),
    data_root: Path = DATA_ROOT,
    scratch_root: Path = SCRATCH_ROOT,
) -> int:
    """Run one task in the foreground and enforce retry-safe completion."""

    validate_owned_directory(
        plan.cwd,
        owned_root=data_root,
        label="scientific run root",
        must_exist=True,
    )
    validate_owned_directory(
        completion_root,
        owned_root=scratch_root,
        label="scheduler completion root",
        must_exist=False,
    )
    if plan.job.spec.output_policy == "sealed_source":
        raise ServerSchedulerAdapterError(
            "task targets a registered immutable completed artifact root; "
            "use a separately registered attempt or read-only adoption task"
        )
    receipt = completion_receipt_path(plan.job, completion_root)
    if receipt.exists():
        evidence = completion_validator(plan.job)
        validate_completion_receipt(receipt, plan.job, evidence)
        return 0
    if plan.job.spec.output_policy == "derivation":
        target = derivation_output_root(plan.job, plan.cwd)
        validate_owned_directory(
            target,
            owned_root=data_root,
            label="registered derivation output",
            must_exist=False,
        )
        if target.exists():
            try:
                evidence = completion_validator(plan.job)
            except ServerSchedulerAdapterError as error:
                raise ServerSchedulerAdapterError(
                    "registered derivation output already exists but is incomplete or "
                    "conflicting; preserve it for audit and use a registered "
                    "same-revision attempt/recovery operation"
                ) from error
            write_completion_receipt(receipt, plan.job, evidence)
            return 0
    try:
        completed = subprocess.run(
            plan.command,
            cwd=plan.cwd,
            env=dict(plan.environment),
            check=False,
        )
    except OSError as error:
        raise ServerSchedulerAdapterError(f"cannot start scientific task: {error}") from error
    if completed.returncode != 0:
        return completed.returncode
    evidence = completion_validator(plan.job)
    write_completion_receipt(receipt, plan.job, evidence)
    return 0
