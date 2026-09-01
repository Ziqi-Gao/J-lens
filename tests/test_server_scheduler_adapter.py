from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

import pytest

import jlens_workspace.server_scheduler_adapter as adapter_module
from jlens_workspace.artifacts import git_head_commit
from jlens_workspace.concept_intervention.k_diagnostic.experiment import (
    KDiagnosticExperimentError,
    detect_k_diagnostic_execution_hardware,
)
from jlens_workspace.scheduler.catalog import PILOT_TASKS, THREE_METHOD_CONCEPTS
from jlens_workspace.scheduler.completion import (
    completion_receipt_path,
    validate_completion_receipt,
    write_completion_receipt,
)
from jlens_workspace.scheduler.runner import (
    DispatchPlan,
    validate_kdiag_bundle_plan,
    validate_owned_directory,
)
from jlens_workspace.server_scheduler_adapter import (
    CODE_ROOT,
    GPU_KDIAG_BUNDLE,
    GPU_OCCUPANCY,
    GPU_STANDARD,
    PROJECT,
    TASK_SPECS,
    CompletionEvidence,
    ServerSchedulerAdapterError,
    build_dispatch_plan,
    execute_dispatch_plan,
    load_running_job,
    validate_task_completion,
)


class _FakeCuda:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def get_device_name(self, _index: int) -> str:
        return "NVIDIA RTX PRO 6000 Blackwell Server Edition"


class _FakeTorch:
    __version__ = "2.8.0"
    cuda = _FakeCuda()
    version = type("Version", (), {"cuda": "12.8"})()


def _descriptor(path: Path, *, relative_to: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": adapter_module.sha256_file(path),
    }


def _fixture(
    tmp_path: Path,
    task: str,
    *,
    shard_index: int = 0,
    shard_count: int | None = None,
    stage: str = "pilot",
    instance: str | None = None,
    numa_node: int | None = 0,
) -> tuple[Path, dict[str, str], set[int]]:
    spec = TASK_SPECS[task]
    profile = spec.profile
    count = shard_count if shard_count is not None else spec.shard_count or 5
    parameters: dict[str, object] = {
        "run_id": spec.run_id,
        "git_commit": git_head_commit(CODE_ROOT),
        "shard_index": shard_index,
        "shard_count": count,
    }
    if spec.driver == "k-diagnostic":
        parameters.update(
            {
                "stage": stage,
                "instance": instance
                if instance is not None
                else (str(shard_index) if spec.shard_count is None else "singleton"),
            }
        )
    resources = {
        "cpu_cores": profile.cpu_cores_preferred,
        "memory_mib": profile.memory_mib,
        "gpu_count": profile.gpu_count,
        "gpu_memory_mib": profile.gpu_memory_mib,
        "gpu_utilization_pct": profile.gpu_utilization_pct,
        "exclusive_gpu": profile.gpu_exclusivity == "required",
    }
    available = sorted(os.sched_getaffinity(0))
    cpu_ids = available[: profile.cpu_cores_preferred]
    assert len(cpu_ids) == profile.cpu_cores_preferred
    gpu = profile.gpu_count == 1
    manifest = tmp_path / f"{task}.json"
    payload = {
        "schema_version": 2,
        "job_id": f"fixture-{task}-{shard_index}",
        "project": PROJECT,
        "task": task,
        "resources": None,
        "allocation": dict(resources),
        "parameters": parameters,
        "priority": 0,
        "submitted_at": "2026-08-31T00:00:00Z",
        "state": "running",
        "updated_at": "2026-08-31T00:00:01Z",
        "requested_profile": None,
        "execution_profile": profile.name,
        "estimated_runtime_seconds": float(spec.estimated_runtime_seconds),
        "gpu_indices": [3] if gpu else [],
        "gpu_uuids": ["GPU-fixture"] if gpu else [],
        "gpu_pci_bus_ids": ["00000000:01:00.0"] if gpu else [],
        "cpu_ids": cpu_ids,
        "numa_node": numa_node,
        "stdout_log": str(tmp_path / "stdout.log"),
        "stderr_log": str(tmp_path / "stderr.log"),
        "exit_code": None,
        "failure_reason": None,
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    environment = {
        "SERVER_SCHEDULER_JOB_ID": payload["job_id"],
        "SERVER_SCHEDULER_PROJECT": PROJECT,
        "SERVER_SCHEDULER_JOB_MANIFEST": str(manifest),
        "SERVER_SCHEDULER_LEASE_ID": "lease-fixture",
        "SERVER_SCHEDULER_ATTEMPT": "1",
        "SERVER_SCHEDULER_EXECUTION_PROFILE": profile.name,
        "SERVER_SCHEDULER_ESTIMATED_RUNTIME_SECONDS": (f"{spec.estimated_runtime_seconds:.6f}"),
        "SERVER_SCHEDULER_CPU_CORES": str(profile.cpu_cores_preferred),
        "SERVER_SCHEDULER_CPUSET": ",".join(map(str, cpu_ids)),
        "SERVER_SCHEDULER_MEMORY_MIB": str(profile.memory_mib),
        "SERVER_SCHEDULER_GPU_COUNT": str(profile.gpu_count),
        "SERVER_SCHEDULER_GPU_MEMORY_MIB": str(profile.gpu_memory_mib),
        "SERVER_SCHEDULER_GPU_UTILIZATION_PCT": str(profile.gpu_utilization_pct),
        "SERVER_SCHEDULER_GPU_EXCLUSIVITY": "exclusive" if gpu else "shared",
        "SERVER_SCHEDULER_GPU_INDICES": "3" if gpu else "",
        "SERVER_SCHEDULER_GPU_UUIDS": "GPU-fixture" if gpu else "",
        "SERVER_SCHEDULER_GPU_PCI_BUS_IDS": ("00000000:01:00.0" if gpu else ""),
        "SERVER_SCHEDULER_STDOUT_LOG": payload["stdout_log"],
        "SERVER_SCHEDULER_STDERR_LOG": payload["stderr_log"],
    }
    if numa_node is not None:
        environment["SERVER_SCHEDULER_NUMA_NODE"] = str(numa_node)
    if gpu:
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "GPU-fixture",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            }
        )
    return manifest, environment, set(cpu_ids)


def test_registration_profiles_preserve_reviewed_resource_inputs() -> None:
    assert len(TASK_SPECS) == 28
    assert TASK_SPECS["lens"].profile == GPU_STANDARD
    assert TASK_SPECS["occupancy"].profile == GPU_OCCUPANCY
    assert TASK_SPECS["kdiag-bundle"].profile == GPU_KDIAG_BUNDLE
    assert all(
        spec.profile.gpu_exclusivity == "required"
        for spec in TASK_SPECS.values()
        if spec.profile.gpu_count == 1
    )
    assert all(
        spec.profile.gpu_exclusivity == "shareable"
        for spec in TASK_SPECS.values()
        if spec.profile.gpu_count == 0
    )
    assert GPU_OCCUPANCY.gpu_memory_mib == 2048
    assert GPU_KDIAG_BUNDLE.memory_mib == 12288
    assert GPU_STANDARD.gpu_utilization_pct == 40
    assert TASK_SPECS["preflight"].output_policy == "read_only"
    assert TASK_SPECS["candidate-rescore"].output_policy == "derivation"
    assert TASK_SPECS["kdiag-report"].output_policy == "derivation"
    assert TASK_SPECS["j-grid"].output_policy == "sealed_source"


def test_registration_proposal_matches_adapter_profiles() -> None:
    proposal_path = (
        CODE_ROOT / "Concept_intervention/docs/server_scheduler_registration_proposal.toml"
    )
    with proposal_path.open("rb") as handle:
        proposal = tomllib.load(handle)
    assert proposal["name"] == PROJECT
    assert proposal["enabled"] is False
    assert proposal["integration_status"] == "protocol-v2-bounded-read-only-pilot-ready"
    assert proposal["paths"]["code_root"] == str(CODE_ROOT)
    assert proposal["policy"]["allow_cpu_jobs"] is True
    assert proposal["policy"]["allow_gpu_jobs"] is False
    assert proposal["integration"]["entrypoint"] == str(
        CODE_ROOT / "Concept_intervention/scripts/server_scheduler_entrypoint.sh"
    )
    assert proposal["integration"]["allowed_tasks"] == list(PILOT_TASKS)

    assert [item["name"] for item in proposal["tasks"]] == list(PILOT_TASKS)
    proposed_tasks = {item["name"]: item for item in proposal["tasks"]}
    assert set(proposed_tasks) == set(PILOT_TASKS)
    for name in PILOT_TASKS:
        spec = TASK_SPECS[name]
        task = proposed_tasks[name]
        assert task["selection_policy"] == "earliest_finish"
        assert task["execution_profiles"] == [spec.registration_profile()]
        assert spec.output_policy == "read_only"

    excluded = set(TASK_SPECS) - set(PILOT_TASKS)
    assert len(excluded) == 27
    assert all(
        TASK_SPECS[name].output_policy in {"sealed_source", "derivation", "read_only"}
        for name in excluded
    )


def test_protocol_v2_pilot_request_example_omits_hardware_and_matches_allowlist() -> None:
    examples = (
        CODE_ROOT / "Concept_intervention/docs/server_scheduler_request_kdiag_v2.example.json",
    )
    for path in examples:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert set(payload) == {
            "schema_version",
            "job_id",
            "project",
            "task",
            "priority",
            "parameters",
        }
        assert payload["schema_version"] == 2
        assert payload["project"] == PROJECT
        assert payload["priority"] == 0
        assert payload["task"] in PILOT_TASKS
        assert payload["parameters"]["run_id"] == TASK_SPECS[payload["task"]].run_id
        assert len(payload["parameters"]["git_commit"]) == 40
        assert set(payload["parameters"]["git_commit"]) <= set("0123456789abcdef")
        spec = TASK_SPECS[payload["task"]]
        expected_parameters = {"run_id", "git_commit", "shard_index", "shard_count"}
        if spec.driver == "k-diagnostic":
            expected_parameters.update({"stage", "instance"})
        assert set(payload["parameters"]) == expected_parameters
        assert 0 <= payload["parameters"]["shard_index"] < payload["parameters"]["shard_count"]
        if spec.shard_count is not None:
            assert payload["parameters"]["shard_count"] == spec.shard_count
        if spec.driver == "k-diagnostic":
            assert payload["parameters"]["stage"] in spec.stages


def test_nonpilot_request_example_is_not_in_minimal_registration() -> None:
    path = (
        CODE_ROOT
        / "Concept_intervention/docs/server_scheduler_request_three_method_v2.example.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["task"] in TASK_SPECS
    assert payload["task"] not in PILOT_TASKS
    assert payload["parameters"]["git_commit"] == "0" * 40


def test_gpu_fixture_dispatches_fixed_task_body_without_local_scheduler(tmp_path: Path) -> None:
    manifest, environment, affinity = _fixture(
        tmp_path,
        "occupancy",
        shard_index=34,
        shard_count=35,
    )
    poisoned_environment = dict(environment)
    poisoned_environment.update(
        {
            "SLURM_CPUS_PER_TASK": "99",
            "SLURM_ARRAY_TASK_ID": "99",
            "OCCUPANCY_OVERWRITE": "1",
            "THREE_METHOD_LOCAL_ENV_LOADED": "1",
            "THREE_METHOD_LOCAL_PATHS_LOADED": "1",
        }
    )
    plan = build_dispatch_plan(
        manifest,
        env=poisoned_environment,
        affinity=affinity,
        verify_inputs=False,
        verify_checkout=False,
    )
    assert plan.command[-2:] == ("occupancy", "34")
    assert plan.command[1].endswith("server_scheduler_three_method_task.sh")
    assert "local_scheduler.py" not in " ".join(plan.command)
    assert "three_method_local_gpu.sh" not in " ".join(plan.command)
    assert plan.environment["CUDA_VISIBLE_DEVICES"] == "GPU-fixture"
    assert plan.environment["JLENS_CPUS_PER_TASK"] == "4"
    assert plan.environment["OMP_NUM_THREADS"] == "4"
    assert plan.environment["JLENS_SERVER_SCHEDULER_ATTEMPT"] == "1"
    assert plan.environment["JLENS_SERVER_SCHEDULER_TASK_ACTIVE"] == "1"
    assert "JLENS_LOCAL_LEASE_ID" not in plan.environment
    assert all(
        key not in plan.environment
        for key in (
            "SLURM_CPUS_PER_TASK",
            "SLURM_ARRAY_TASK_ID",
            "OCCUPANCY_OVERWRITE",
            "THREE_METHOD_LOCAL_ENV_LOADED",
            "THREE_METHOD_LOCAL_PATHS_LOADED",
        )
    )


def test_kdiagnostic_dispatch_uses_scheduler_specific_transport(tmp_path: Path) -> None:
    manifest, environment, affinity = _fixture(
        tmp_path,
        "kdiag-report",
        stage="transformed",
    )
    plan = build_dispatch_plan(
        manifest,
        env=environment,
        affinity=affinity,
        verify_inputs=False,
        verify_checkout=False,
    )
    assert plan.command[1].endswith("server_scheduler_kdiag_task.sh")
    assert plan.command[-3:] == ("report", "transformed", "singleton")


def test_dispatch_rejects_dirty_registered_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = type(
        "Completed",
        (),
        {
            "returncode": 0,
            "stdout": "?? src/jlens_workspace/server_scheduler_adapter.py\n",
            "stderr": "",
        },
    )()
    monkeypatch.setattr(adapter_module.subprocess, "run", lambda *_args, **_kwargs: completed)
    with pytest.raises(ServerSchedulerAdapterError, match="checkout is not clean"):
        adapter_module._require_clean_checkout()


def test_cpu_fixture_rejects_gpu_visibility_and_allocation_drift(tmp_path: Path) -> None:
    manifest, environment, affinity = _fixture(tmp_path, "preflight")
    assert load_running_job(manifest, env=environment, affinity=affinity).task == "preflight"

    visible = dict(environment)
    visible["CUDA_VISIBLE_DEVICES"] = "0"
    with pytest.raises(ServerSchedulerAdapterError, match="GPU visibility"):
        load_running_job(manifest, env=visible, affinity=affinity)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["allocation"]["memory_mib"] = 4096
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    environment["SERVER_SCHEDULER_MEMORY_MIB"] = "4096"
    with pytest.raises(ServerSchedulerAdapterError, match="host memory"):
        load_running_job(manifest, env=environment, affinity=affinity)


def test_concrete_allocation_must_satisfy_requested_resource_override(
    tmp_path: Path,
) -> None:
    manifest, environment, affinity = _fixture(tmp_path, "preflight")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["resources"] = dict(payload["allocation"])
    payload["resources"]["memory_mib"] = payload["allocation"]["memory_mib"] * 2
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ServerSchedulerAdapterError, match="does not satisfy"):
        load_running_job(manifest, env=environment, affinity=affinity)


def test_fixture_rejects_v1_singular_gpu_and_wrong_execution_profile(tmp_path: Path) -> None:
    manifest, environment, affinity = _fixture(tmp_path, "lens")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ServerSchedulerAdapterError, match="protocol-v2"):
        load_running_job(manifest, env=environment, affinity=affinity)

    payload["schema_version"] = 2
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    singular = dict(environment)
    singular["SERVER_SCHEDULER_GPU_INDEX"] = "3"
    with pytest.raises(ServerSchedulerAdapterError, match="singular"):
        load_running_job(manifest, env=singular, affinity=affinity)

    payload["execution_profile"] = "gpu-unregistered"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    environment["SERVER_SCHEDULER_EXECUTION_PROFILE"] = "gpu-unregistered"
    with pytest.raises(ServerSchedulerAdapterError, match="execution_profile"):
        load_running_job(manifest, env=environment, affinity=affinity)


def test_cross_numa_fixture_accepts_null_and_requires_omitted_environment(
    tmp_path: Path,
) -> None:
    manifest, environment, affinity = _fixture(
        tmp_path,
        "layer-selection",
        numa_node=None,
    )
    job = load_running_job(manifest, env=environment, affinity=affinity)
    assert job.numa_node is None
    assert "SERVER_SCHEDULER_NUMA_NODE" not in environment

    stale_environment = dict(environment)
    stale_environment["SERVER_SCHEDULER_NUMA_NODE"] = "0"
    with pytest.raises(ServerSchedulerAdapterError, match="cross-NUMA"):
        load_running_job(manifest, env=stale_environment, affinity=affinity)


def test_fixture_rejects_symlink_state_and_parameter_injection(tmp_path: Path) -> None:
    manifest, environment, affinity = _fixture(tmp_path, "j-grid", shard_count=392)
    link = tmp_path / "manifest-link.json"
    link.symlink_to(manifest)
    linked_environment = dict(environment)
    linked_environment["SERVER_SCHEDULER_JOB_MANIFEST"] = str(link)
    with pytest.raises(ServerSchedulerAdapterError, match="non-symlink"):
        load_running_job(link, env=linked_environment, affinity=affinity)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["state"] = "pending"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ServerSchedulerAdapterError, match="running job"):
        load_running_job(manifest, env=environment, affinity=affinity)

    payload["state"] = "running"
    payload["parameters"]["command"] = ["/bin/sh"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ServerSchedulerAdapterError, match="unknown fields"):
        load_running_job(manifest, env=environment, affinity=affinity)


def test_kdiagnostic_accepts_only_matching_server_scheduler_gpu_fixture(tmp_path: Path) -> None:
    _manifest, environment, _affinity = _fixture(
        tmp_path,
        "kdiag-bundle",
        shard_index=2,
        shard_count=5,
        instance="2",
    )
    hardware = detect_k_diagnostic_execution_hardware(_FakeTorch(), env=environment)
    assert hardware["execution_backend"] == "server_scheduler"
    assert hardware["gpu_count"] == 1

    conflicting = dict(environment)
    conflicting["JLENS_LOCAL_LEASE_ID"] = "legacy-lease"
    with pytest.raises(KDiagnosticExperimentError, match="exactly one"):
        detect_k_diagnostic_execution_hardware(_FakeTorch(), env=conflicting)


def test_dynamic_kdiagnostic_dispatch_accepts_the_frozen_v2_bundle_schema(
    tmp_path: Path,
) -> None:
    manifest, environment, affinity = _fixture(
        tmp_path,
        "kdiag-bundle",
        shard_index=1,
        shard_count=2,
        stage="pilot",
        instance="1",
    )
    job = load_running_job(manifest, env=environment, affinity=affinity)
    bundle_path = (
        tmp_path
        / "artifacts/concept_intervention"
        / job.spec.run_id
        / "pilot/bundles.json"
    )
    bundle_path.parent.mkdir(parents=True)
    plan = {
        "schema_version": 2,
        "stage": "pilot",
        "logical_replicate_count": 3,
        "physical_bundle_count": 2,
        "bundles": [
            {
                "bundle_id": "bundle-a",
                "conservative_cost_units": 10,
                "layer": 11,
                "logical_shard_count": 1,
                "metric": "raw_euclidean",
                "metric_dimension": 2560,
                "shard_ids": ["shard-a"],
                "target_family": "class_mean_difference",
                "target_id": "target-a",
                "target_subtype": "concept-a",
            },
            {
                "bundle_id": "bundle-b",
                "conservative_cost_units": 20,
                "layer": 11,
                "logical_shard_count": 2,
                "metric": "raw_euclidean",
                "metric_dimension": 2560,
                "shard_ids": ["shard-b", "shard-c"],
                "target_family": "class_mean_difference",
                "target_id": "target-b",
                "target_subtype": "concept-b",
            },
        ],
    }
    bundle_path.write_text(json.dumps(plan), encoding="utf-8")
    validate_kdiag_bundle_plan(job, run_root=tmp_path)

    plan["physical_bundle_count"] = 1
    bundle_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ServerSchedulerAdapterError, match="bundle plan"):
        validate_kdiag_bundle_plan(job, run_root=tmp_path)


@pytest.mark.parametrize(
    ("stage_name", "stage_directory"),
    (("pilot", "pilot"), ("transformed", "transformed_metric")),
)
def test_kdiagnostic_microbenchmark_completion_includes_computed_bundle(
    tmp_path: Path,
    stage_name: str,
    stage_directory: str,
) -> None:
    manifest, environment, affinity = _fixture(
        tmp_path,
        "kdiag-microbenchmark",
        shard_index=1,
        shard_count=2,
        stage=stage_name,
        instance="1",
    )
    job = load_running_job(manifest, env=environment, affinity=affinity)
    stage = (
        tmp_path
        / "artifacts/concept_intervention"
        / job.spec.run_id
        / stage_directory
    )
    stage.mkdir(parents=True)
    bundles = {
        "schema_version": 2,
        "stage": stage_directory,
        "logical_replicate_count": 2,
        "physical_bundle_count": 2,
        "bundles": [
            {
                "bundle_id": "bundle-a",
                "conservative_cost_units": 1,
                "logical_shard_count": 1,
                "shard_ids": ["shard-a"],
            },
            {
                "bundle_id": "bundle-b",
                "conservative_cost_units": 2,
                "logical_shard_count": 1,
                "shard_ids": ["shard-b"],
            },
        ],
    }
    (stage / "bundles.json").write_text(json.dumps(bundles), encoding="utf-8")
    (stage / "microbenchmark.json").write_text(
        json.dumps(
            {
                "stage": stage_directory,
                "bundle_id": "bundle-b",
                "bundle_index": 1,
                "approved": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ServerSchedulerAdapterError, match="bundle completion"):
        validate_task_completion(job, run_root=tmp_path)

    complete = stage / "bundles/bundle-b/complete.json"
    complete.parent.mkdir(parents=True)
    complete.write_text(
        json.dumps({"complete": True, "physical_bundle_id": "bundle-b"}),
        encoding="utf-8",
    )
    evidence = validate_task_completion(job, run_root=tmp_path)
    assert [Path(item.path).name for item in evidence] == [
        "bundles.json",
        "microbenchmark.json",
        "complete.json",
    ]


@pytest.mark.parametrize(
    ("task", "stage_name", "stage_directory", "filename", "extra"),
    (
        (
            "kdiag-rotation-cache-preflight",
            "full",
            "raw_metric_full",
            "rotation_cache_build_preflight.json",
            {"approved": True},
        ),
        (
            "kdiag-prepare-rotations",
            "transformed",
            "transformed_metric",
            "rotation_cache_index.json",
            {"complete": True},
        ),
        (
            "kdiag-resource-preflight",
            "full",
            "raw_metric_full",
            "resource_preflight.json",
            {"approved": True},
        ),
        (
            "kdiag-index",
            "transformed",
            "transformed_metric",
            "index.json",
            {"complete": True},
        ),
    ),
)
def test_kdiagnostic_completion_uses_canonical_stage_directory_identity(
    tmp_path: Path,
    task: str,
    stage_name: str,
    stage_directory: str,
    filename: str,
    extra: dict[str, object],
) -> None:
    instance = "wave_0_1" if task == "kdiag-resource-preflight" else "singleton"
    manifest, environment, affinity = _fixture(
        tmp_path,
        task,
        stage=stage_name,
        instance=instance,
    )
    job = load_running_job(manifest, env=environment, affinity=affinity)
    marker = (
        tmp_path
        / "artifacts/concept_intervention"
        / job.spec.run_id
        / stage_directory
        / filename
    )
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps({"stage": stage_directory, **extra}), encoding="utf-8"
    )

    evidence = validate_task_completion(job, run_root=tmp_path)
    if task == "kdiag-resource-preflight":
        assert evidence == ()
    else:
        assert [item.path for item in evidence] == [str(marker.resolve())]


def test_profile_payload_is_registration_ready() -> None:
    profile = TASK_SPECS["iti-fit"].registration_profile()
    assert profile == {
        "name": "cpu-16-fixed",
        "kind": "cpu",
        "cpu_cores_min": 16,
        "cpu_cores_preferred": 16,
        "cpu_cores_max": 16,
        "memory_mib": 32768,
        "gpu_count": 0,
        "gpu_memory_mib": 0,
        "gpu_utilization_pct": 0,
        "gpu_exclusivity": "shareable",
        "scheduling_goal": "balanced",
        "resource_mode": "fixed",
        "gpu_models": [],
        "cpu_scaling_efficiency": 0.0,
        "estimated_runtime_seconds": 120,
    }


def test_completion_receipt_is_atomic_and_skips_a_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, environment, affinity = _fixture(tmp_path, "preflight")
    plan = build_dispatch_plan(
        manifest,
        env=environment,
        affinity=affinity,
        verify_inputs=False,
        verify_checkout=False,
    )
    calls: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **_kwargs: object) -> object:
        calls.append(tuple(command))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(adapter_module.subprocess, "run", fake_run)
    completion_root = tmp_path / "completions"
    assert execute_dispatch_plan(plan, completion_root=completion_root) == 0
    receipt = completion_root / f"{plan.job.job_id}.json"
    assert json.loads(receipt.read_text(encoding="utf-8"))["complete"] is True
    assert execute_dispatch_plan(plan, completion_root=completion_root) == 0
    assert len(calls) == 1
    assert list(completion_root.glob(".*.tmp")) == []


def test_nonzero_exit_never_writes_completion_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, environment, affinity = _fixture(tmp_path, "preflight")
    plan = build_dispatch_plan(
        manifest,
        env=environment,
        affinity=affinity,
        verify_inputs=False,
        verify_checkout=False,
    )
    monkeypatch.setattr(
        adapter_module.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Completed", (), {"returncode": 17})(),
    )
    completion_root = tmp_path / "completions"
    assert execute_dispatch_plan(plan, completion_root=completion_root) == 17
    assert not completion_root.exists()


def test_central_dispatch_refuses_to_mutate_a_registered_sealed_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, environment, affinity = _fixture(
        tmp_path,
        "j-grid",
        shard_index=6,
        shard_count=392,
    )
    plan = build_dispatch_plan(
        manifest,
        env=environment,
        affinity=affinity,
        verify_inputs=False,
        verify_checkout=False,
    )
    monkeypatch.setattr(
        adapter_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("sealed task must not start a subprocess"),
    )
    with pytest.raises(ServerSchedulerAdapterError, match="immutable completed artifact root"):
        execute_dispatch_plan(plan, completion_root=tmp_path / "completions")


def test_lens_completion_requires_atomic_final_lens_not_resume_checkpoint(
    tmp_path: Path,
) -> None:
    manifest, environment, affinity = _fixture(tmp_path, "lens")
    job = load_running_job(manifest, env=environment, affinity=affinity)
    lens_root = (
        tmp_path
        / "artifacts/concept_intervention"
        / job.spec.run_id
        / "shared_intervention_protocol/lens"
    )
    lens_root.mkdir(parents=True)
    identity = {
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "tokenizer_id": "Qwen/Qwen3.5-4B",
        "tokenizer_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "source_layers": [3, 7, 11, 15, 19, 23, 27],
        "target_layer": 31,
        "n_fit_prompts": 1000,
        "force_bos": False,
        "storage_dtype": "float32",
    }
    checkpoint_manifest = lens_root / "qwen35_4b_fp32.checkpoint.pt.manifest.json"
    checkpoint_manifest.write_text(json.dumps(identity), encoding="utf-8")
    (lens_root / "qwen35_4b_fp32.checkpoint.pt").write_bytes(b"resume-only")

    with pytest.raises(ServerSchedulerAdapterError, match="final fitted lens"):
        validate_task_completion(job, run_root=tmp_path)

    final_lens = lens_root / "qwen35_4b_fp32.pt"
    final_lens.write_bytes(b"atomic-final-lens")
    evidence = validate_task_completion(job, run_root=tmp_path)
    assert [item.path for item in evidence] == [
        str(checkpoint_manifest.resolve()),
        str(final_lens.resolve()),
    ]


def test_zero_exit_requires_scientific_completion_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, environment, affinity = _fixture(tmp_path, "preflight")
    plan = build_dispatch_plan(
        manifest,
        env=environment,
        affinity=affinity,
        verify_inputs=False,
        verify_checkout=False,
    )
    monkeypatch.setattr(
        adapter_module.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Completed", (), {"returncode": 0})(),
    )
    gate = tmp_path / "test-completion-gate.json"

    def validator(_job: object) -> tuple[CompletionEvidence, ...]:
        if not gate.is_file():
            raise ServerSchedulerAdapterError("test completion gate absent")
        return (CompletionEvidence(str(gate), adapter_module.sha256_file(gate)),)

    with pytest.raises(ServerSchedulerAdapterError, match="test completion gate"):
        execute_dispatch_plan(
            plan,
            completion_root=tmp_path / "completions",
            completion_validator=validator,
        )

    gate.write_text(json.dumps({"complete": True}), encoding="utf-8")
    assert (
        execute_dispatch_plan(
            plan,
            completion_root=tmp_path / "completions",
            completion_validator=validator,
        )
        == 0
    )


def test_rescore_comparison_completion_uses_immutable_revision_root(tmp_path: Path) -> None:
    manifest, environment, affinity = _fixture(tmp_path, "comparison-rescore-index")
    job = load_running_job(manifest, env=environment, affinity=affinity)
    base = tmp_path / "artifacts/concept_intervention" / job.spec.run_id
    legacy_index = base / "intervention_comparison/index.json"
    legacy_index.parent.mkdir(parents=True)
    legacy_comparison = legacy_index.with_name("comparison.json")
    legacy_comparison.write_text(json.dumps({"legacy": True}), encoding="utf-8")
    legacy_index.write_text(
        json.dumps(
            {
                "complete": True,
                "comparison": "comparison.json",
                "comparison_sha256": adapter_module.sha256_file(legacy_comparison),
                "candidate_rescore_indexes": {"legacy": {}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ServerSchedulerAdapterError, match="candidate-rescore comparison"):
        validate_task_completion(job, run_root=tmp_path)

    revision_index = tmp_path / "derivations/revision-r2/intervention_comparison/index.json"
    revision_index.parent.mkdir(parents=True)
    revision_comparison = revision_index.with_name("comparison.json")
    candidate_rescore_indexes: dict[str, dict[str, object]] = {}
    for method in (
        "j_component_intervention",
        "iti_intervention",
        "raptor_intervention",
    ):
        source_root = base / method
        source_root.mkdir(parents=True)
        source_index = source_root / "index.json"
        source_manifest = source_root / "manifest.json"
        source_index.write_text(json.dumps({"complete": True}), encoding="utf-8")
        source_manifest.write_text(json.dumps({"method": method}), encoding="utf-8")
        rescore_root = (
            tmp_path
            / "derivations/revision-r2/candidate_score_rescore_v1"
            / method
        )
        rescore_root.mkdir(parents=True)
        rescore_manifest = rescore_root / "manifest.json"
        rescore_manifest.write_text(json.dumps({"method": method}), encoding="utf-8")
        rescore_entries: list[dict[str, object]] = []
        for concept_id in THREE_METHOD_CONCEPTS:
            concept_root = rescore_root / "targets" / concept_id.replace(":", "%3A")
            concept_root.mkdir(parents=True)
            candidate_path = concept_root / "candidate_scores.jsonl"
            candidate_path.write_text(
                json.dumps({"concept_id": concept_id}) + "\n", encoding="utf-8"
            )
            summary_path = concept_root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "artifact_kind": "candidate_score_rescore",
                        "method": method,
                        "target_concept_id": concept_id,
                        "candidate_score_contract": {
                            "schema_version": 1,
                            "batch_size": 1,
                            "batching": "one_prompt_per_forward",
                        },
                        "candidate_scores_sha256": adapter_module.sha256_file(
                            candidate_path
                        ),
                    }
                ),
                encoding="utf-8",
            )
            rescore_entries.append(
                {
                    "concept_id": concept_id,
                    "summary": summary_path.relative_to(rescore_root).as_posix(),
                    "summary_sha256": adapter_module.sha256_file(summary_path),
                    "candidate_scores": candidate_path.relative_to(
                        rescore_root
                    ).as_posix(),
                    "candidate_scores_sha256": adapter_module.sha256_file(
                        candidate_path
                    ),
                }
            )
        rescore_index = rescore_root / "index.json"
        rescore_index.write_text(
            json.dumps(
                {
                    "complete": True,
                    "artifact_kind": "candidate_score_rescore_index",
                    "method": method,
                    "scoring_contract": {
                        "schema_version": 1,
                        "batch_size": 1,
                        "batching": "one_prompt_per_forward",
                    },
                    "expected_concepts": list(THREE_METHOD_CONCEPTS),
                    "observed_concepts": sorted(THREE_METHOD_CONCEPTS),
                    "missing_concepts": [],
                    "rescore_manifest_sha256": adapter_module.sha256_file(
                        rescore_manifest
                    ),
                    "source_method_index_sha256": adapter_module.sha256_file(
                        source_index
                    ),
                    "source_method_manifest_sha256": adapter_module.sha256_file(
                        source_manifest
                    ),
                    "entries": rescore_entries,
                }
            ),
            encoding="utf-8",
        )
        candidate_rescore_indexes[method] = {
            "path": str(rescore_index.resolve()),
            "sha256": adapter_module.sha256_file(rescore_index),
            "manifest": str(rescore_manifest.resolve()),
            "manifest_sha256": adapter_module.sha256_file(rescore_manifest),
        }
    revision_comparison.write_text(
        json.dumps(
            {
                "comparison": "three_method_shared_layer_held_out_target_margin",
                "llm_as_judge_run": False,
                "candidate_rescore_indexes": candidate_rescore_indexes,
                "entries": [
                    {"concept_id": concept_id} for concept_id in THREE_METHOD_CONCEPTS
                ],
            }
        ),
        encoding="utf-8",
    )
    revision_index.write_text(
        json.dumps(
            {
                "complete": True,
                "comparison": "comparison.json",
                "comparison_sha256": adapter_module.sha256_file(revision_comparison),
                "methods": [
                    "j_component_intervention",
                    "iti_intervention",
                    "raptor_intervention",
                ],
                "concept_ids": list(THREE_METHOD_CONCEPTS),
                "method_provenance": {
                    "j_component_intervention": {},
                    "iti_intervention": {},
                    "raptor_intervention": {},
                },
                "index_builder": {},
                "llm_as_judge_run": False,
                "candidate_rescore_indexes": candidate_rescore_indexes,
            }
        ),
        encoding="utf-8",
    )
    evidence = validate_task_completion(job, run_root=tmp_path)
    assert [item.path for item in evidence[:2]] == [
        str(revision_index.resolve()),
        str(revision_comparison.resolve()),
    ]
    assert len(evidence) == 56
    revision_comparison.write_text(json.dumps({"corrupt": True}), encoding="utf-8")
    with pytest.raises(ServerSchedulerAdapterError, match="companion hash"):
        validate_task_completion(job, run_root=tmp_path)


def test_candidate_rescore_completion_revalidates_all_sealed_companions(
    tmp_path: Path,
) -> None:
    manifest, environment, affinity = _fixture(tmp_path, "candidate-rescore", shard_index=0)
    job = load_running_job(manifest, env=environment, affinity=affinity)
    base = tmp_path / "artifacts/concept_intervention" / job.spec.run_id
    method = "j_component_intervention"
    source_root = base / method
    source_root.mkdir(parents=True)
    source_index = source_root / "index.json"
    source_manifest = source_root / "manifest.json"
    source_index.write_text(json.dumps({"complete": True}), encoding="utf-8")
    source_manifest.write_text(json.dumps({"method": method}), encoding="utf-8")

    rescore_root = (
        tmp_path
        / "derivations/revision-r2/candidate_score_rescore_v1"
        / method
    )
    rescore_root.mkdir(parents=True)
    rescore_manifest = rescore_root / "manifest.json"
    rescore_manifest.write_text(json.dumps({"method": method}), encoding="utf-8")
    entries: list[dict[str, object]] = []
    candidate_paths: list[Path] = []
    for concept_id in THREE_METHOD_CONCEPTS:
        concept_root = rescore_root / "targets" / concept_id.replace(":", "%3A")
        concept_root.mkdir(parents=True)
        candidate_path = concept_root / "candidate_scores.jsonl"
        candidate_path.write_text(json.dumps({"concept_id": concept_id}) + "\n", encoding="utf-8")
        summary_path = concept_root / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "artifact_kind": "candidate_score_rescore",
                    "method": method,
                    "target_concept_id": concept_id,
                    "candidate_score_contract": {
                        "schema_version": 1,
                        "batch_size": 1,
                        "batching": "one_prompt_per_forward",
                    },
                    "candidate_scores_sha256": adapter_module.sha256_file(candidate_path),
                }
            ),
            encoding="utf-8",
        )
        candidate_paths.append(candidate_path)
        entries.append(
            {
                "concept_id": concept_id,
                "summary": summary_path.relative_to(rescore_root).as_posix(),
                "summary_sha256": adapter_module.sha256_file(summary_path),
                "candidate_scores": candidate_path.relative_to(rescore_root).as_posix(),
                "candidate_scores_sha256": adapter_module.sha256_file(candidate_path),
            }
        )
    rescore_index = rescore_root / "index.json"
    rescore_index.write_text(
        json.dumps(
            {
                "artifact_kind": "candidate_score_rescore_index",
                "method": method,
                "complete": True,
                "scoring_contract": {
                    "schema_version": 1,
                    "batch_size": 1,
                    "batching": "one_prompt_per_forward",
                },
                "expected_concepts": list(THREE_METHOD_CONCEPTS),
                "observed_concepts": sorted(THREE_METHOD_CONCEPTS),
                "missing_concepts": [],
                "source_method_root": str(source_root.resolve()),
                "source_method_index_sha256": adapter_module.sha256_file(source_index),
                "source_method_manifest_sha256": adapter_module.sha256_file(source_manifest),
                "rescore_manifest_sha256": adapter_module.sha256_file(rescore_manifest),
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )

    evidence = validate_task_completion(job, run_root=tmp_path)
    assert len(evidence) == 1 + 3 + 2 * len(THREE_METHOD_CONCEPTS)
    candidate_paths[0].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ServerSchedulerAdapterError, match="hash no longer matches"):
        validate_task_completion(job, run_root=tmp_path)


def test_kdiagnostic_report_completion_requires_the_registered_derived_package(
    tmp_path: Path,
) -> None:
    manifest, environment, affinity = _fixture(
        tmp_path,
        "kdiag-report",
        stage="transformed",
    )
    job = load_running_job(manifest, env=environment, affinity=affinity)
    source_root = tmp_path / "artifacts/concept_intervention" / job.spec.run_id
    source_root.mkdir(parents=True)
    source_evidence = source_root / "source.json"
    source_evidence.write_text(json.dumps({"complete": True}), encoding="utf-8")
    report_root = tmp_path / "derivations/revision-r1/report"
    report_root.mkdir(parents=True)
    package_files = {
        "decision_table": report_root / "decision_table.json",
        "aggregate_summary": report_root / "aggregate_summary.json",
        "terminal_summary": report_root / "terminal_summary.json",
        "target_level_summary": report_root / "target_level_summary.parquet",
        "target_pair_summary": report_root / "target_pair_summary.parquet",
        "report": report_root / "report.md",
    }
    package_files["decision_table"].write_text(
        json.dumps({"decision": "inconclusive"}), encoding="utf-8"
    )
    package_files["aggregate_summary"].write_text(
        json.dumps({"summary": True}), encoding="utf-8"
    )
    package_files["terminal_summary"].write_text(
        json.dumps({"A": {}, "B": {}}), encoding="utf-8"
    )
    package_files["target_level_summary"].write_bytes(b"parquet-target")
    package_files["target_pair_summary"].write_bytes(b"parquet-pair")
    package_files["report"].write_text("# report\n", encoding="utf-8")

    data_file = report_root / "data/target_level_summary.parquet"
    data_file.parent.mkdir()
    data_file.write_bytes(b"plot-source")
    figure_pdf = report_root / "figures/figure.pdf"
    figure_png = report_root / "figures/figure.png"
    figure_pdf.parent.mkdir()
    figure_pdf.write_bytes(b"pdf")
    figure_png.write_bytes(b"png")
    figure_manifest = report_root / "data/figure_data_manifest.json"
    figure_manifest.write_text(
        json.dumps(
            {
                "identity": job.spec.run_id,
                "artifact_root": os.path.relpath(source_root, report_root),
                "data_files": {
                    "target_level": _descriptor(data_file, relative_to=report_root)
                },
                "figure_sources": {"figure": {"tables": ["target_level"]}},
                "figure_files": {
                    "figure": {
                        "pdf": _descriptor(figure_pdf, relative_to=report_root),
                        "png": _descriptor(figure_png, relative_to=report_root),
                    }
                },
                "source_artifacts": [
                    _descriptor(source_evidence, relative_to=source_root)
                ],
            }
        ),
        encoding="utf-8",
    )
    package_files["figure_data_manifest"] = figure_manifest
    report_index = report_root / "index.json"
    report_index.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": job.spec.run_id,
                "revision": "revision-r1",
                "complete": True,
                "files": {
                    name: _descriptor(path, relative_to=report_root)
                    for name, path in package_files.items()
                },
            }
        ),
        encoding="utf-8",
    )

    evidence = validate_task_completion(job, run_root=tmp_path)
    assert evidence[0].path == str(report_index.resolve())
    assert {item.path for item in evidence} >= {
        str(package_files["decision_table"].resolve()),
        str(package_files["report"].resolve()),
        str(data_file.resolve()),
        str(figure_pdf.resolve()),
        str(source_evidence.resolve()),
    }

    figure_pdf.write_bytes(b"corrupt")
    with pytest.raises(ServerSchedulerAdapterError, match="no longer matches"):
        validate_task_completion(job, run_root=tmp_path)


def test_derivation_redelivery_adopts_complete_output_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, environment, affinity = _fixture(tmp_path, "candidate-rescore", shard_index=0)
    original = build_dispatch_plan(
        manifest,
        env=environment,
        affinity=affinity,
        verify_inputs=False,
        verify_checkout=False,
    )
    data_root = tmp_path / "owned-data"
    run_root = data_root / "runs" / original.job.spec.run_id
    run_root.mkdir(parents=True)
    plan = DispatchPlan(original.command, run_root, original.environment, original.job)
    target = (
        run_root
        / "derivations/revision-r2/candidate_score_rescore_v1"
        / "j_component_intervention"
    )
    target.mkdir(parents=True)
    (target / "index.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    scratch_root = tmp_path / "owned-scratch"
    scratch_root.mkdir()
    completion_root = scratch_root / "completions"

    monkeypatch.setattr(
        adapter_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("complete derivation must be adopted"),
    )

    def validator(_job: object) -> tuple[CompletionEvidence, ...]:
        index_path = target / "index.json"
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("complete") is not True:
            raise ServerSchedulerAdapterError("test derivation is incomplete")
        return (
            CompletionEvidence(str(index_path), adapter_module.sha256_file(index_path)),
        )

    assert (
        execute_dispatch_plan(
            plan,
            completion_root=completion_root,
            completion_validator=validator,
            data_root=data_root,
            scratch_root=scratch_root,
        )
        == 0
    )
    assert completion_receipt_path(plan.job, completion_root).is_file()


def test_derivation_redelivery_rejects_partial_output_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, environment, affinity = _fixture(tmp_path, "candidate-rescore", shard_index=1)
    original = build_dispatch_plan(
        manifest,
        env=environment,
        affinity=affinity,
        verify_inputs=False,
        verify_checkout=False,
    )
    data_root = tmp_path / "owned-data"
    run_root = data_root / "runs" / original.job.spec.run_id
    target = (
        run_root
        / "derivations/revision-r2/candidate_score_rescore_v1"
        / "iti_intervention"
    )
    target.mkdir(parents=True)
    plan = DispatchPlan(original.command, run_root, original.environment, original.job)
    scratch_root = tmp_path / "owned-scratch"
    scratch_root.mkdir()
    monkeypatch.setattr(
        adapter_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("partial derivation must not run"),
    )

    def validator(_job: object) -> tuple[CompletionEvidence, ...]:
        index_path = target / "index.json"
        if not index_path.is_file():
            raise ServerSchedulerAdapterError("test derivation is incomplete")
        return (
            CompletionEvidence(str(index_path), adapter_module.sha256_file(index_path)),
        )

    with pytest.raises(ServerSchedulerAdapterError, match="incomplete or conflicting"):
        execute_dispatch_plan(
            plan,
            completion_root=scratch_root / "completions",
            completion_validator=validator,
            data_root=data_root,
            scratch_root=scratch_root,
        )


def test_owned_directory_validation_rejects_symlink_components(tmp_path: Path) -> None:
    owned_root = tmp_path / "owned"
    outside = tmp_path / "outside"
    owned_root.mkdir()
    outside.mkdir()
    (owned_root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ServerSchedulerAdapterError, match="symlink component"):
        validate_owned_directory(
            owned_root / "escape/output",
            owned_root=owned_root,
            label="test output",
            must_exist=False,
        )


def test_mutable_resource_preflight_gate_does_not_invalidate_completed_wave_receipt(
    tmp_path: Path,
) -> None:
    manifest, environment, affinity = _fixture(
        tmp_path,
        "kdiag-resource-preflight",
        stage="pilot",
        instance="wave_0_2",
    )
    job = load_running_job(manifest, env=environment, affinity=affinity)
    preflight = (
        tmp_path
        / "artifacts/concept_intervention"
        / job.spec.run_id
        / "pilot/resource_preflight.json"
    )
    preflight.parent.mkdir(parents=True)
    preflight.write_text(
        json.dumps({"stage": "pilot", "approved": True, "wave": 0}),
        encoding="utf-8",
    )
    completion_root = tmp_path / "completions"
    evidence = validate_task_completion(job, run_root=tmp_path)
    assert evidence == ()
    receipt = completion_receipt_path(job, completion_root)
    write_completion_receipt(receipt, job, evidence)
    assert json.loads(receipt.read_text(encoding="utf-8"))["evidence"] == []

    preflight.write_text(
        json.dumps({"stage": "pilot", "approved": True, "wave": 1}),
        encoding="utf-8",
    )
    evidence = validate_task_completion(job, run_root=tmp_path)
    validate_completion_receipt(receipt, job, evidence)

    preflight.write_text(
        json.dumps({"stage": "pilot", "approved": False, "wave": 2}),
        encoding="utf-8",
    )
    with pytest.raises(ServerSchedulerAdapterError, match="wrong approved"):
        validate_task_completion(job, run_root=tmp_path)


def test_receipt_revalidates_evidence_hash_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, environment, affinity = _fixture(tmp_path, "preflight")
    plan = build_dispatch_plan(
        manifest,
        env=environment,
        affinity=affinity,
        verify_inputs=False,
        verify_checkout=False,
    )
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text("{}", encoding="utf-8")

    def validator(_job: object) -> tuple[CompletionEvidence, ...]:
        return (CompletionEvidence(str(evidence_path), adapter_module.sha256_file(evidence_path)),)

    monkeypatch.setattr(
        adapter_module.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Completed", (), {"returncode": 0})(),
    )
    completion_root = tmp_path / "completions"
    assert (
        execute_dispatch_plan(
            plan,
            completion_root=completion_root,
            completion_validator=validator,
        )
        == 0
    )
    evidence_path.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ServerSchedulerAdapterError, match="evidence"):
        execute_dispatch_plan(
            plan,
            completion_root=completion_root,
            completion_validator=validator,
        )
