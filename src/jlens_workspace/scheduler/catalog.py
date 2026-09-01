"""Authoritative J-lens ServerScheduler task and execution-profile catalog."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from jlens_workspace.scheduler.errors import ServerSchedulerAdapterError

if TYPE_CHECKING:
    from jlens_workspace.scheduler.allocation import Allocation

PROJECT = "J-lens"
CODE_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/data/del6500/J-lens")
SCRATCH_ROOT = Path("/scr/del6500/J-lens")
PILOT_TASKS = ("kdiag-validate",)
PROJECT_PYTHON = SCRATCH_ROOT / "envs/three-method/bin/python"
COMPLETION_ROOT = SCRATCH_ROOT / "server_scheduler/completions"
THREE_METHOD_RUN_ID = "qwen35_4b_three_method_intervention_v1"
THREE_METHOD_RUN_ROOT = DATA_ROOT / "runs" / THREE_METHOD_RUN_ID
THREE_METHOD_REVISION_R2 = Path("derivations/revision-r2")
KDIAG_RUN_ID = "qwen35_4b_k_diagnostic_v2"
KDIAG_RUN_ROOT = DATA_ROOT / "runs" / KDIAG_RUN_ID
KDIAG_REPORT_REVISION_R1 = Path("derivations/revision-r1/report")
GPU_MODEL = "NVIDIA RTX PRO 6000 Blackwell Server Edition"

KDIAG_STAGE_DIRECTORY = {
    "pilot": "pilot",
    "full": "raw_metric_full",
    "transformed": "transformed_metric",
}
THREE_METHOD_CONCEPTS = (
    "goemotions:admiration",
    "goemotions:approval",
    "goemotions:curiosity",
    "goemotions:disapproval",
    "goemotions:gratitude",
    "goemotions:love",
    "goemotions:optimism",
)
THREE_METHOD_LAYERS = (3, 7, 11, 15, 19, 23, 27)
THREE_METHOD_REPLICATES = (
    "primary",
    "bootstrap_1101",
    "bootstrap_2202",
    "bootstrap_3303",
    "bootstrap_4404",
)


@dataclass(frozen=True)
class ResourceProfile:
    """Registration-ready v2 execution-profile resource envelope."""

    name: str
    kind: str
    cpu_cores_min: int
    cpu_cores_preferred: int
    cpu_cores_max: int
    memory_mib: int
    gpu_count: int
    gpu_memory_mib: int
    gpu_utilization_pct: int
    gpu_exclusivity: str
    scheduling_goal: str
    resource_mode: str = "fixed"
    gpu_models: tuple[str, ...] = ()
    cpu_scaling_efficiency: float = 0.0

    def validate(self, resources: Allocation, *, label: str) -> None:
        """Require an allocation to stay inside this registered envelope."""

        if not self.cpu_cores_min <= resources.cpu_cores <= self.cpu_cores_max:
            raise ServerSchedulerAdapterError(
                f"{label} CPU cores are outside the registered envelope"
            )
        if self.resource_mode == "fixed" and resources.cpu_cores != self.cpu_cores_preferred:
            raise ServerSchedulerAdapterError(f"{label} does not use the fixed CPU allocation")
        if resources.memory_mib < self.memory_mib:
            raise ServerSchedulerAdapterError(
                f"{label} host memory is below the registered minimum"
            )
        if resources.gpu_count != self.gpu_count:
            raise ServerSchedulerAdapterError(f"{label} GPU count is incompatible with the task")
        if resources.gpu_memory_mib < self.gpu_memory_mib:
            raise ServerSchedulerAdapterError(f"{label} GPU memory is below the registered minimum")
        if resources.gpu_utilization_pct < self.gpu_utilization_pct:
            raise ServerSchedulerAdapterError(
                f"{label} GPU utilization is below the registered minimum"
            )
        if self.gpu_exclusivity == "required" and resources.exclusive_gpu is not True:
            raise ServerSchedulerAdapterError(f"{label} must use an exclusive GPU")
        if self.gpu_count == 0:
            if resources.gpu_memory_mib != 0 or resources.gpu_utilization_pct != 0:
                raise ServerSchedulerAdapterError(f"{label} CPU task declares GPU resources")


@dataclass(frozen=True)
class TaskSpec:
    """Fixed task body, parameter envelope, and scheduler profile."""

    name: str
    driver: str
    action: str
    run_id: str
    shard_count: int | None
    profile: ResourceProfile
    estimated_runtime_seconds: int
    output_policy: str
    stages: frozenset[str] = frozenset()
    kdiag_profile: str | None = None

    def registration_profile(self) -> dict[str, object]:
        """Render the exact ServerScheduler v2 execution-profile payload."""

        payload = asdict(self.profile)
        payload["gpu_models"] = list(self.profile.gpu_models)
        payload["estimated_runtime_seconds"] = self.estimated_runtime_seconds
        return payload


def _profile(
    name: str,
    cpu: int,
    memory: int,
    *,
    gpu_memory: int = 0,
    gpu_utilization: int = 0,
    scheduling_goal: str = "balanced",
) -> ResourceProfile:
    gpu_count = int(gpu_memory > 0)
    return ResourceProfile(
        name=name,
        kind="gpu" if gpu_count else "cpu",
        cpu_cores_min=cpu,
        cpu_cores_preferred=cpu,
        cpu_cores_max=cpu,
        memory_mib=memory,
        gpu_count=gpu_count,
        gpu_memory_mib=gpu_memory,
        gpu_utilization_pct=gpu_utilization,
        gpu_exclusivity="required" if gpu_count else "shareable",
        scheduling_goal=scheduling_goal,
        gpu_models=(GPU_MODEL,) if gpu_count else (),
    )


CPU_DEFAULT = _profile("cpu-4-fixed", 4, 8192)
CPU_BOOTSTRAP = _profile("cpu-8-fixed", 8, 16384, scheduling_goal="throughput")
CPU_HEAVY = _profile("cpu-16-fixed", 16, 32768)
GPU_STANDARD = _profile("gpu-24g-required", 4, 24576, gpu_memory=24576, gpu_utilization=40)
GPU_STANDARD_SHARD = _profile(
    "gpu-24g-required",
    4,
    24576,
    gpu_memory=24576,
    gpu_utilization=40,
    scheduling_goal="throughput",
)
GPU_OCCUPANCY = _profile(
    "gpu-2g-required",
    4,
    4096,
    gpu_memory=2048,
    gpu_utilization=20,
    scheduling_goal="throughput",
)
GPU_KDIAG_BUNDLE = _profile(
    "gpu-4g-required",
    4,
    12288,
    gpu_memory=4096,
    gpu_utilization=20,
    scheduling_goal="throughput",
)


def _three_method_spec(
    name: str,
    count: int,
    profile: ResourceProfile,
    estimated_runtime_seconds: int,
) -> TaskSpec:
    output_policy = (
        "read_only"
        if name == "preflight"
        else (
            "derivation"
            if name in {"candidate-rescore", "comparison-rescore-index"}
            else "sealed_source"
        )
    )
    return TaskSpec(
        name,
        "three-method",
        name,
        THREE_METHOD_RUN_ID,
        count,
        profile,
        estimated_runtime_seconds,
        output_policy,
    )


def _kdiag_spec(
    name: str,
    action: str,
    stages: Sequence[str],
    profile: ResourceProfile,
    kdiag_profile: str,
    estimated_runtime_seconds: int,
    *,
    shard_count: int | None = 1,
) -> TaskSpec:
    output_policy = (
        "read_only"
        if name == "kdiag-validate"
        else ("derivation" if name == "kdiag-report" else "sealed_source")
    )
    return TaskSpec(
        name,
        "k-diagnostic",
        action,
        KDIAG_RUN_ID,
        shard_count,
        profile,
        estimated_runtime_seconds,
        output_policy,
        frozenset(stages),
        kdiag_profile,
    )


TASK_SPECS: Mapping[str, TaskSpec] = {
    spec.name: spec
    for spec in (
        _three_method_spec("preflight", 1, CPU_DEFAULT, 60),
        _three_method_spec("lens", 1, GPU_STANDARD, 21600),
        _three_method_spec("capture", 1, GPU_STANDARD, 900),
        _three_method_spec("layer-selection", 1, CPU_HEAVY, 900),
        _three_method_spec("bootstrap-probes", 7, CPU_BOOTSTRAP, 120),
        _three_method_spec("bootstrap-index", 1, CPU_DEFAULT, 60),
        _three_method_spec("occupancy", 35, GPU_OCCUPANCY, 4200),
        _three_method_spec("occupancy-index", 1, CPU_DEFAULT, 60),
        _three_method_spec("iti-capture", 1, GPU_STANDARD, 900),
        _three_method_spec("iti-fit", 1, CPU_HEAVY, 120),
        _three_method_spec("j-grid", 392, GPU_STANDARD_SHARD, 5400),
        _three_method_spec("raptor-grid", 63, GPU_STANDARD_SHARD, 5400),
        _three_method_spec("iti-grid", 3087, GPU_STANDARD_SHARD, 5400),
        _three_method_spec("smoke-check", 1, CPU_DEFAULT, 60),
        _three_method_spec("method-index", 3, CPU_DEFAULT, 1800),
        _three_method_spec("comparison-index", 1, CPU_DEFAULT, 120),
        _three_method_spec("candidate-rescore", 3, GPU_STANDARD_SHARD, 14400),
        _three_method_spec("comparison-rescore-index", 1, CPU_DEFAULT, 180),
        _kdiag_spec(
            "kdiag-validate",
            "validate",
            ("pilot", "full", "transformed"),
            CPU_DEFAULT,
            "kdiag-control",
            60,
        ),
        _kdiag_spec(
            "kdiag-prepare-targets",
            "prepare-targets",
            ("pilot", "full"),
            GPU_STANDARD,
            "kdiag-targets",
            120,
        ),
        _kdiag_spec(
            "kdiag-build-bases",
            "build-bases",
            ("transformed",),
            GPU_STANDARD,
            "kdiag-targets",
            120,
        ),
        _kdiag_spec(
            "kdiag-rotation-cache-preflight",
            "rotation-cache-preflight",
            ("pilot", "full", "transformed"),
            CPU_HEAVY,
            "kdiag-rotations",
            120,
        ),
        _kdiag_spec(
            "kdiag-prepare-rotations",
            "prepare-rotations",
            ("pilot", "full", "transformed"),
            CPU_HEAVY,
            "kdiag-rotations",
            900,
        ),
        _kdiag_spec(
            "kdiag-resource-preflight",
            "resource-preflight",
            ("pilot", "full", "transformed"),
            CPU_DEFAULT,
            "kdiag-control",
            60,
        ),
        _kdiag_spec(
            "kdiag-index",
            "index",
            ("pilot", "full", "transformed"),
            CPU_DEFAULT,
            "kdiag-control",
            1200,
        ),
        _kdiag_spec(
            "kdiag-report",
            "report",
            ("transformed",),
            CPU_DEFAULT,
            "kdiag-control",
            300,
        ),
        _kdiag_spec(
            "kdiag-microbenchmark",
            "microbenchmark",
            ("pilot", "full", "transformed"),
            GPU_KDIAG_BUNDLE,
            "kdiag-bundle",
            21600,
            shard_count=None,
        ),
        _kdiag_spec(
            "kdiag-bundle",
            "bundle",
            ("pilot", "full", "transformed"),
            GPU_KDIAG_BUNDLE,
            "kdiag-bundle",
            7200,
            shard_count=None,
        ),
    )
}
