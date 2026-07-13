from __future__ import annotations

import pytest

from jlens_workspace.interventions import ResidualIntervention, matched_random_direction

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
