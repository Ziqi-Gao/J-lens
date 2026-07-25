from __future__ import annotations

from pathlib import Path

import pytest

from jlens_workspace.artifacts import GitIdentityError, git_head_commit


def test_git_head_commit_reads_detached_checkout_without_git_cli(
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    git_dir = tmp_path / "repo/.git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text(f"{commit}\n", encoding="utf-8")

    assert git_head_commit(tmp_path / "repo") == commit


def test_git_head_commit_resolves_linked_worktree_common_refs(
    tmp_path: Path,
) -> None:
    commit = "b" * 40
    common = tmp_path / "common.git"
    admin = common / "worktrees/experiment"
    checkout = tmp_path / "experiment"
    (common / "refs/heads/codex").mkdir(parents=True)
    admin.mkdir(parents=True)
    checkout.mkdir()
    (checkout / ".git").write_text(
        f"gitdir: {admin}\n",
        encoding="utf-8",
    )
    (admin / "commondir").write_text("../..\n", encoding="utf-8")
    (admin / "HEAD").write_text(
        "ref: refs/heads/codex/experiment\n",
        encoding="utf-8",
    )
    (common / "refs/heads/codex/experiment").write_text(
        f"{commit}\n",
        encoding="utf-8",
    )

    assert git_head_commit(checkout) == commit


def test_git_head_commit_fails_closed_on_missing_ref(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text(
        "ref: refs/heads/missing\n",
        encoding="utf-8",
    )

    with pytest.raises(GitIdentityError, match="cannot resolve"):
        git_head_commit(tmp_path)
