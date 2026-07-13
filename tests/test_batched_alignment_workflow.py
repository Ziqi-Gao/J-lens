from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jlens_workspace.jacobian import build_effective_unembedding  # noqa: E402
from jlens_workspace.matrix import TokenFrameOperator  # noqa: E402
from jlens_workspace.workflows import run_batched_probe_j_alignment  # noqa: E402


def test_batched_alignment_scans_once_and_writes_each_probe(tmp_path) -> None:
    unembedding = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]], dtype=torch.float64
    )
    operator = TokenFrameOperator(
        torch.eye(2, dtype=torch.float64),
        build_effective_unembedding(unembedding),
        block_size=2,
        compute_device="cpu",
        compute_dtype=torch.float64,
    )
    results = run_batched_probe_j_alignment(
        probe_vectors={"honesty": np.array([1.0, 0.1]), "uncertainty": np.array([0.1, 1.0])},
        operator=operator,
        output_dir=tmp_path / "batch",
        top_k=2,
        candidate_pool_size=3,
        sparse_components=2,
    )
    assert set(results) == {"honesty", "uncertainty"}
    assert results["honesty"]["top_positive"][0]["token_id"] in {0, 2}
    assert (tmp_path / "batch" / "honesty" / "j_component.npy").is_file()
    assert (tmp_path / "batch" / "uncertainty" / "alignment.json").is_file()


def test_batched_nearest_only_does_not_fit_sparse_components(tmp_path) -> None:
    operator = TokenFrameOperator(
        torch.eye(2, dtype=torch.float64),
        build_effective_unembedding(torch.eye(2, dtype=torch.float64)),
        block_size=1,
        compute_device="cpu",
        compute_dtype=torch.float64,
    )
    output = tmp_path / "nearest"
    results = run_batched_probe_j_alignment(
        probe_vectors={"concept": np.array([1.0, 0.1])},
        operator=operator,
        output_dir=output,
        top_k=1,
        decompose=False,
    )
    assert results["concept"]["decomposition"] is None
    assert not (output / "concept" / "j_component.npy").exists()
    assert len(results["concept"]["top_positive"]) == 1


def test_batched_alignment_reports_deterministic_matched_random_controls(
    tmp_path,
) -> None:
    operator = TokenFrameOperator(
        torch.eye(3, dtype=torch.float64),
        build_effective_unembedding(torch.eye(3, dtype=torch.float64)),
        block_size=2,
        compute_device="cpu",
        compute_dtype=torch.float64,
    )
    kwargs = {
        "probe_vectors": {"concept": np.array([1.0, 0.2, -0.1])},
        "operator": operator,
        "top_k": 1,
        "decompose": False,
        "random_control_seeds": (7, 11, 19),
    }
    first = run_batched_probe_j_alignment(
        **kwargs, output_dir=tmp_path / "first"
    )["concept"]
    second = run_batched_probe_j_alignment(
        **kwargs, output_dir=tmp_path / "second"
    )["concept"]

    assert first["random_controls"] == second["random_controls"]
    assert len(first["random_controls"]) == 3
    assert first["random_control_summary"]["n_controls"] == 3
    assert 0.0 < first["random_control_summary"]["empirical_p_value"] <= 1.0
    assert all(
        abs(control["cosine_with_probe"]) < 1e-12
        for control in first["random_controls"]
    )
    root = __import__("json").loads(
        (tmp_path / "first" / "alignment.json").read_text()
    )
    assert root["vocabulary_passes"] == 1
    assert root["random_control_seeds"] == [7, 11, 19]
