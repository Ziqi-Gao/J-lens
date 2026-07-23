from __future__ import annotations

import pytest

from jlens_workspace.concept_intervention.j_component import (
    ResidualIntervention,
    matched_random_direction,
    multilayer_intervention_session,
)

torch = pytest.importorskip("torch")


def test_addition_only_changes_last_prefill_position() -> None:
    output = torch.zeros(2, 4, 3)
    hook = ResidualIntervention(
        direction=torch.tensor([1.0, 0.0, 0.0]),
        strength=2.0,
        position="last_prompt",
        residual_norm=3.0,
    )
    changed = hook(None, None, output)
    assert torch.equal(changed[:, :-1], output[:, :-1])
    assert torch.allclose(changed[:, -1, 0], torch.full((2,), 6.0))


def test_project_out_removes_direction() -> None:
    output = torch.tensor([[[2.0, 3.0]]])
    hook = ResidualIntervention(
        direction=torch.tensor([1.0, 0.0]), strength=1.0, kind="project_out", position="all"
    )
    changed = hook(None, None, output)
    assert torch.allclose(changed, torch.tensor([[[0.0, 3.0]]]))


def test_random_control_is_unit_and_orthogonal() -> None:
    direction = torch.tensor([1.0, 2.0, 3.0])
    control = matched_random_direction(direction, seed=7)
    assert torch.allclose(control.norm(), torch.tensor(1.0), atol=1e-6)
    assert abs(float(control @ (direction / direction.norm()))) < 1e-6


def test_multilayer_addition_uses_full_per_layer_norm_and_cleans_hooks() -> None:
    from types import SimpleNamespace

    blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    model = SimpleNamespace(model=SimpleNamespace(layers=blocks))
    source = torch.zeros((1, 1, 2))
    with multilayer_intervention_session(
        model,
        directions={
            0: torch.tensor([1.0, 0.0]),
            1: torch.tensor([0.0, 1.0]),
        },
        strength=0.5,
        residual_norms={0: 2.0, 1: 4.0},
    ) as states:
        changed = blocks[1](blocks[0](source))

    torch.testing.assert_close(changed, torch.tensor([[[1.0, 2.0]]]))
    assert states[0].events[-1]["injected_norm"] == pytest.approx(1.0)
    assert states[1].events[-1]["injected_norm"] == pytest.approx(2.0)
    assert states[0].events[-1]["forward_call"] == 0
    assert states[0].events[-1]["generation_step"] == 0
    assert states[0].events[-1]["strength"] == 0.5
    assert states[0].events[-1]["residual_norm"] == 2.0
    torch.testing.assert_close(blocks[1](blocks[0](source)), source)


def test_manifest_commit_can_be_supplied_without_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    from pathlib import Path

    from jlens_workspace.artifacts import RunManifest

    monkeypatch.setenv("JLENS_GIT_COMMIT", "a" * 40)
    manifest = RunManifest.for_workspace(Path(str(tmp_path)))
    assert manifest.git_commit == "a" * 40
