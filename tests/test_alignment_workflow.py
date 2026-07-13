from __future__ import annotations

import json

import numpy as np
import pytest

from jlens_workspace.workflows.alignment import run_probe_j_alignment


def test_alignment_workflow_finds_tokens_and_writes_components(tmp_path) -> None:
    unembedding = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]], dtype=np.float64
    )
    jacobian = np.eye(2, dtype=np.float64)
    payload = run_probe_j_alignment(
        probe_vector=np.asarray([1.0, 0.25]),
        unembedding=unembedding,
        jacobian=jacobian,
        output_dir=tmp_path / "alignment",
        activations=np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        top_k=2,
        candidate_pool_size=3,
        sparse_components=2,
        chunk_size=2,
    )
    assert payload["top_positive"][0]["token_id"] in {0, 2}
    assert payload["top_negative"][0]["score"] <= 0
    assert 0 <= payload["decomposition"]["explained_energy"] <= 1
    assert (tmp_path / "alignment" / "j_component.npy").is_file()
    on_disk = json.loads((tmp_path / "alignment" / "alignment.json").read_text())
    assert on_disk["decomposition"]["selected_count"] <= 2


def test_alignment_workflow_rejects_output_overwrite(tmp_path) -> None:
    output = tmp_path / "alignment"
    output.mkdir()
    (output / "sentinel").write_text("keep")
    with pytest.raises(FileExistsError):
        run_probe_j_alignment(
            probe_vector=np.asarray([1.0, 0.0]),
            unembedding=np.eye(2),
            jacobian=np.eye(2),
            output_dir=output,
            top_k=1,
            candidate_pool_size=1,
            sparse_components=1,
        )
