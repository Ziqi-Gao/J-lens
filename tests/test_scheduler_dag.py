from __future__ import annotations

import pytest

from jlens_workspace.scheduler.dag import (
    DAGNode,
    TaskInvocation,
    kdiagnostic_dag,
    three_method_dag,
    validate_dag,
)
from jlens_workspace.scheduler.errors import ServerSchedulerAdapterError


def _task_indices(nodes: tuple[DAGNode, ...], task: str) -> set[int]:
    return {
        invocation.shard_index
        for node in nodes
        for invocation in node.invocations
        if invocation.task == task
    }


def test_three_method_dag_covers_every_grid_shard_once() -> None:
    nodes = three_method_dag()
    assert _task_indices(nodes, "j-grid") == set(range(392))
    assert _task_indices(nodes, "raptor-grid") == set(range(63))
    assert _task_indices(nodes, "iti-grid") == set(range(3087))
    by_id = {node.node_id: node for node in nodes}
    assert by_id["smoke-check"].depends_on == ("smoke-j", "smoke-raptor", "smoke-iti")
    assert by_id["comparison-index"].depends_on == ("index-j", "index-iti", "index-raptor")
    assert by_id["candidate-rescore"].depends_on == ("comparison-index",)
    assert by_id["comparison-rescore-index"].depends_on == ("candidate-rescore",)


def test_kdiagnostic_dag_preserves_stage_and_wave_gates() -> None:
    nodes = kdiagnostic_dag(
        {"pilot": (5, 2), "full": (4, 1), "transformed": (3, 0)},
        wave_size=2,
    )
    by_id = {node.node_id: node for node in nodes}
    assert by_id["kdiag-full-validate"].depends_on == ("kdiag-pilot-index",)
    assert by_id["kdiag-transformed-validate"].depends_on == ("kdiag-full-index",)
    assert by_id["kdiag-report"].depends_on == ("kdiag-transformed-index",)
    assert by_id["kdiag-pilot-bundle-wave-0-2"].depends_on == (
        "kdiag-pilot-post-microbenchmark-preflight",
    )
    assert by_id["kdiag-pilot-bundle-wave-2-4"].depends_on == (
        "kdiag-pilot-resource-preflight-0-2",
    )
    for stage, count in (("pilot", 5), ("full", 4), ("transformed", 3)):
        indices = {
            invocation.shard_index
            for node in nodes
            for invocation in node.invocations
            if invocation.task in {"kdiag-microbenchmark", "kdiag-bundle"}
            and invocation.stage == stage
        }
        assert indices == set(range(count))


def test_dag_invocation_renders_only_allowlisted_parameters() -> None:
    invocation = TaskInvocation("j-grid", 6, 392)
    assert invocation.parameters("a" * 40) == {
        "run_id": "qwen35_4b_three_method_intervention_v1",
        "git_commit": "a" * 40,
        "shard_index": 6,
        "shard_count": 392,
    }
    with pytest.raises(ServerSchedulerAdapterError, match="40 lowercase hex"):
        invocation.parameters("A" * 40)


def test_dag_rejects_bad_dynamic_plan_and_cycles() -> None:
    with pytest.raises(ServerSchedulerAdapterError, match="bundle plan"):
        kdiagnostic_dag(
            {"pilot": (2, 2), "full": (2, 0), "transformed": (2, 0)}
        )
    cycle = (
        DAGNode("a", (TaskInvocation("preflight", 0, 1),), ("b",), "gate"),
        DAGNode("b", (TaskInvocation("lens", 0, 1),), ("a",), "gate"),
    )
    with pytest.raises(ServerSchedulerAdapterError, match="cycle"):
        validate_dag(cycle)
