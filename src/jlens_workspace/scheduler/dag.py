"""Deterministic scientific DAG plans for ServerScheduler request publishers.

The module is deliberately side-effect free: it does not create an outbox file
or submit a job. It describes exactly which allowlisted invocations may be
released after which scientific gates, so a separately approved publisher can
translate a validated plan into protocol-v2 requests.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from jlens_workspace.scheduler.catalog import TASK_SPECS
from jlens_workspace.scheduler.errors import ServerSchedulerAdapterError

_COMMIT = re.compile(r"[0-9a-f]{40}")
_KDIAG_STAGES = ("pilot", "full", "transformed")


@dataclass(frozen=True)
class TaskInvocation:
    """One fixed task/shard identity that can become a v2 job request."""

    task: str
    shard_index: int
    shard_count: int
    stage: str | None = None
    instance: str | None = None

    def parameters(self, git_commit: str) -> dict[str, object]:
        """Render only the project-owned parameter allowlist."""

        if _COMMIT.fullmatch(git_commit) is None:
            raise ServerSchedulerAdapterError("DAG request commit must be 40 lowercase hex")
        spec = TASK_SPECS[self.task]
        payload: dict[str, object] = {
            "run_id": spec.run_id,
            "git_commit": git_commit,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
        }
        if spec.driver == "k-diagnostic":
            payload.update({"stage": self.stage, "instance": self.instance})
        return payload


@dataclass(frozen=True)
class DAGNode:
    """A fan-out group released atomically after its dependencies pass."""

    node_id: str
    invocations: tuple[TaskInvocation, ...]
    depends_on: tuple[str, ...]
    completion_gate: str


def _invocations(
    task: str,
    indices: Sequence[int],
    *,
    stage: str | None = None,
    shard_count: int | None = None,
    instance_prefix: str | None = None,
) -> tuple[TaskInvocation, ...]:
    spec = TASK_SPECS[task]
    count = spec.shard_count if shard_count is None else shard_count
    if count is None:
        raise ServerSchedulerAdapterError(f"{task} requires a persisted dynamic shard count")
    return tuple(
        TaskInvocation(
            task=task,
            shard_index=index,
            shard_count=count,
            stage=stage,
            instance=(
                None
                if spec.driver != "k-diagnostic"
                else (str(index) if instance_prefix is None else f"{instance_prefix}{index}")
            ),
        )
        for index in indices
    )


def _single(
    task: str,
    *,
    stage: str | None = None,
    instance: str = "singleton",
) -> tuple[TaskInvocation, ...]:
    spec = TASK_SPECS[task]
    return (
        TaskInvocation(
            task=task,
            shard_index=0,
            shard_count=spec.shard_count or 1,
            stage=stage,
            instance=instance if spec.driver == "k-diagnostic" else None,
        ),
    )


def validate_dag(nodes: Sequence[DAGNode]) -> tuple[DAGNode, ...]:
    """Validate catalog membership, shard identities, dependencies, and acyclicity."""

    frozen = tuple(nodes)
    by_id = {node.node_id: node for node in frozen}
    if not frozen or len(by_id) != len(frozen):
        raise ServerSchedulerAdapterError("DAG node IDs must be non-empty and unique")
    observed: set[tuple[object, ...]] = set()
    for node in frozen:
        if not node.invocations or not node.completion_gate.strip():
            raise ServerSchedulerAdapterError(f"DAG node is incomplete: {node.node_id}")
        if node.node_id in node.depends_on or any(dep not in by_id for dep in node.depends_on):
            raise ServerSchedulerAdapterError(f"DAG dependency is invalid: {node.node_id}")
        for invocation in node.invocations:
            spec = TASK_SPECS.get(invocation.task)
            if spec is None:
                raise ServerSchedulerAdapterError(f"DAG task is not registered: {invocation.task}")
            if invocation.shard_count <= 0 or not 0 <= invocation.shard_index < invocation.shard_count:
                raise ServerSchedulerAdapterError("DAG shard identity is invalid")
            if spec.shard_count is not None and invocation.shard_count != spec.shard_count:
                raise ServerSchedulerAdapterError(
                    f"DAG shard count differs from catalog: {invocation.task}"
                )
            if spec.driver == "k-diagnostic":
                if invocation.stage not in spec.stages or not invocation.instance:
                    raise ServerSchedulerAdapterError("K-diagnostic DAG identity is incomplete")
            elif invocation.stage is not None or invocation.instance is not None:
                raise ServerSchedulerAdapterError("three-method DAG received K-diagnostic fields")
            identity = (
                invocation.task,
                invocation.shard_index,
                invocation.shard_count,
                invocation.stage,
                invocation.instance,
            )
            if identity in observed:
                raise ServerSchedulerAdapterError(f"duplicate DAG invocation: {identity}")
            observed.add(identity)

    indegree = {node.node_id: len(set(node.depends_on)) for node in frozen}
    dependants: dict[str, set[str]] = {node.node_id: set() for node in frozen}
    for node in frozen:
        for dependency in set(node.depends_on):
            dependants[dependency].add(node.node_id)
    ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        current = ready.pop(0)
        visited += 1
        for dependant in sorted(dependants[current]):
            indegree[dependant] -= 1
            if indegree[dependant] == 0:
                ready.append(dependant)
                ready.sort()
    if visited != len(frozen):
        raise ServerSchedulerAdapterError("DAG contains a dependency cycle")
    return frozen


def three_method_dag(*, include_rescore: bool = True) -> tuple[DAGNode, ...]:
    """Return the frozen three-method smoke, full, finalize, and rescore DAG."""

    smoke = {"j-grid": (6,), "raptor-grid": (8,), "iti-grid": (6, 251)}
    nodes = [
        DAGNode("preflight", _single("preflight"), (), "validated preflight receipt"),
        DAGNode("lens", _single("lens"), ("preflight",), "validated lens manifest"),
        DAGNode("capture", _single("capture"), ("preflight",), "complete activation metadata"),
        DAGNode(
            "layer-selection",
            _single("layer-selection"),
            ("lens", "capture"),
            "identity-checked selected-layer artifact",
        ),
        DAGNode(
            "bootstrap-probes",
            _invocations("bootstrap-probes", range(7)),
            ("layer-selection",),
            "all seven bootstrap probe artifacts",
        ),
        DAGNode(
            "bootstrap-index",
            _single("bootstrap-index"),
            ("bootstrap-probes",),
            "complete bootstrap index",
        ),
        DAGNode(
            "iti-capture",
            _single("iti-capture"),
            ("bootstrap-index",),
            "complete ITI head-activation artifact",
        ),
        DAGNode(
            "iti-fit",
            _single("iti-fit"),
            ("iti-capture",),
            "content-addressed ITI directions",
        ),
        DAGNode(
            "occupancy",
            _invocations("occupancy", range(35)),
            ("iti-fit", "lens"),
            "all occupancy shards",
        ),
        DAGNode(
            "occupancy-index",
            _single("occupancy-index"),
            ("occupancy",),
            "complete occupancy index",
        ),
        DAGNode(
            "smoke-j",
            _invocations("j-grid", smoke["j-grid"]),
            ("occupancy-index",),
            "J-component smoke shard",
        ),
        DAGNode(
            "smoke-raptor",
            _invocations("raptor-grid", smoke["raptor-grid"]),
            ("occupancy-index",),
            "RAPTOR smoke shard",
        ),
        DAGNode(
            "smoke-iti",
            _invocations("iti-grid", smoke["iti-grid"]),
            ("occupancy-index",),
            "two ITI smoke shards",
        ),
        DAGNode(
            "smoke-check",
            _single("smoke-check"),
            ("smoke-j", "smoke-raptor", "smoke-iti"),
            "validated smoke gate",
        ),
    ]
    for task in ("j-grid", "raptor-grid", "iti-grid"):
        spec = TASK_SPECS[task]
        assert spec.shard_count is not None
        node_id = f"full-{task}"
        indices = tuple(index for index in range(spec.shard_count) if index not in smoke[task])
        nodes.append(
            DAGNode(
                node_id,
                _invocations(task, indices),
                ("smoke-check",),
                f"all non-smoke {task} shards",
            )
        )
    nodes.extend(
        (
            DAGNode(
                "index-j",
                _invocations("method-index", (0,)),
                ("full-j-grid",),
                "complete J-component method index",
            ),
            DAGNode(
                "index-iti",
                _invocations("method-index", (1,)),
                ("full-iti-grid",),
                "complete ITI method index",
            ),
            DAGNode(
                "index-raptor",
                _invocations("method-index", (2,)),
                ("full-raptor-grid",),
                "complete RAPTOR method index",
            ),
            DAGNode(
                "comparison-index",
                _single("comparison-index"),
                ("index-j", "index-iti", "index-raptor"),
                "complete intervention comparison index",
            ),
        )
    )
    if include_rescore:
        nodes.extend(
            (
                DAGNode(
                    "candidate-rescore",
                    _invocations("candidate-rescore", range(3)),
                    ("comparison-index",),
                    "all three immutable candidate rescoring artifacts",
                ),
                DAGNode(
                    "comparison-rescore-index",
                    _single("comparison-rescore-index"),
                    ("candidate-rescore",),
                    "complete comparison index with candidate rescoring",
                ),
            )
        )
    return validate_dag(nodes)


def kdiagnostic_dag(
    bundle_plans: Mapping[str, tuple[int, int]],
    *,
    wave_size: int = 20,
) -> tuple[DAGNode, ...]:
    """Expand the three ordered K-diagnostic stages and mandatory wave gates.

    ``bundle_plans`` maps each stage to ``(bundle_count, worst_bundle_index)``
    read from that stage's immutable ``bundles.json`` after preparation.
    """

    if tuple(bundle_plans) != _KDIAG_STAGES or wave_size <= 0:
        raise ServerSchedulerAdapterError(
            "K-diagnostic plans must provide pilot, full, transformed in order"
        )
    nodes: list[DAGNode] = []
    previous_index: str | None = None
    for stage in _KDIAG_STAGES:
        count, worst = bundle_plans[stage]
        if count <= 0 or not 0 <= worst < count:
            raise ServerSchedulerAdapterError(f"invalid K-diagnostic bundle plan: {stage}")
        prefix = f"kdiag-{stage}"
        prepare_task = "kdiag-build-bases" if stage == "transformed" else "kdiag-prepare-targets"
        validate_dependencies = () if previous_index is None else (previous_index,)
        nodes.extend(
            (
                DAGNode(
                    f"{prefix}-validate",
                    _single("kdiag-validate", stage=stage),
                    validate_dependencies,
                    f"validated {stage} inputs",
                ),
                DAGNode(
                    f"{prefix}-prepare",
                    _single(prepare_task, stage=stage),
                    (f"{prefix}-validate",),
                    f"complete {stage} target/basis index",
                ),
                DAGNode(
                    f"{prefix}-rotation-preflight",
                    _single("kdiag-rotation-cache-preflight", stage=stage),
                    (f"{prefix}-prepare",),
                    "approved rotation-cache preflight",
                ),
                DAGNode(
                    f"{prefix}-rotations",
                    _single("kdiag-prepare-rotations", stage=stage),
                    (f"{prefix}-rotation-preflight",),
                    "complete rotation index",
                ),
                DAGNode(
                    f"{prefix}-microbenchmark",
                    _invocations(
                        "kdiag-microbenchmark",
                        (worst,),
                        stage=stage,
                        shard_count=count,
                    ),
                    (f"{prefix}-rotations",),
                    "complete worst-bundle microbenchmark",
                ),
                DAGNode(
                    f"{prefix}-post-microbenchmark-preflight",
                    _single(
                        "kdiag-resource-preflight",
                        stage=stage,
                        instance="post_microbenchmark",
                    ),
                    (f"{prefix}-microbenchmark",),
                    "approved resource preflight after microbenchmark",
                ),
            )
        )
        prior_gate = f"{prefix}-post-microbenchmark-preflight"
        remaining = [index for index in range(count) if index != worst]
        for offset in range(0, len(remaining), wave_size):
            wave = remaining[offset : offset + wave_size]
            end = offset + len(wave)
            wave_id = f"{prefix}-bundle-wave-{offset}-{end}"
            gate_id = f"{prefix}-resource-preflight-{offset}-{end}"
            nodes.append(
                DAGNode(
                    wave_id,
                    _invocations("kdiag-bundle", wave, stage=stage, shard_count=count),
                    (prior_gate,),
                    f"complete {stage} bundle wave {offset}:{end}",
                )
            )
            nodes.append(
                DAGNode(
                    gate_id,
                    _single(
                        "kdiag-resource-preflight",
                        stage=stage,
                        instance=f"wave_{offset}_{end}",
                    ),
                    (wave_id,),
                    f"approved resource preflight after wave {offset}:{end}",
                )
            )
            prior_gate = gate_id
        index_id = f"{prefix}-index"
        nodes.append(
            DAGNode(
                index_id,
                _single("kdiag-index", stage=stage),
                (prior_gate,),
                f"complete {stage} scientific index",
            )
        )
        previous_index = index_id
    assert previous_index is not None
    nodes.append(
        DAGNode(
            "kdiag-report",
            _single("kdiag-report", stage="transformed"),
            (previous_index,),
            "terminal summary, decision table, figures, and report",
        )
    )
    return validate_dag(nodes)


__all__ = [
    "DAGNode",
    "TaskInvocation",
    "kdiagnostic_dag",
    "three_method_dag",
    "validate_dag",
]
