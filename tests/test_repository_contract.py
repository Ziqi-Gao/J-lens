from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _python_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _tree_imports(*roots: Path) -> set[str]:
    files = [path for root in roots for path in root.rglob("*.py")]
    return set().union(*(_python_imports(path) for path in files))


def test_single_canonical_agent_guide() -> None:
    assert (ROOT / "AGENTS.md").is_file()
    assert not (ROOT / "Agent.md").exists()
    assert "[AGENTS.md](AGENTS.md)" in (ROOT / "README.md").read_text()


def test_shared_data_preparation_is_not_nested_in_either_lane() -> None:
    assert (ROOT / "Concept_intervention").is_dir()
    assert (ROOT / "J_space").is_dir()
    assert not (ROOT / "Concept_intervention/J_space").exists()
    assert (ROOT / "scripts/prepare_go_emotions.py").is_file()
    assert not (ROOT / "Concept_intervention/scripts/prepare_go_emotions.py").exists()


def test_direction_roots_remain_scientific_control_surfaces() -> None:
    concept = ROOT / "Concept_intervention"
    j_space = ROOT / "J_space"

    for relative in ("README.md", "experiments", "configs", "data", "docs", "reports", "scripts"):
        assert (concept / relative).exists()
    for relative in ("README.md", "configs", "reports", "scripts"):
        assert (j_space / relative).exists()

    architecture = (ROOT / "docs/architecture.md").read_text(encoding="utf-8")
    assert "authoritative experiment surfaces" in architecture
    assert "what scientific experiment is being run" in architecture
    assert "how that experiment is computed" in architecture


def test_j_space_implementation_does_not_depend_on_concept_intervention() -> None:
    imports = _tree_imports(ROOT / "src/jlens_workspace/j_space")
    assert not any(
        module.startswith("jlens_workspace.concept_intervention") for module in imports
    )


def test_canonical_implementation_does_not_import_compatibility_paths() -> None:
    source = ROOT / "src/jlens_workspace"
    canonical_roots = (
        source / "foundation",
        source / "data",
        source / "j_space",
        source / "scheduler",
        source / "concept_intervention/protocol",
        source / "concept_intervention/probing",
        source / "concept_intervention/geometry",
        source / "concept_intervention/steering",
        source / "concept_intervention/evaluation",
        source / "concept_intervention/reporting",
        source / "concept_intervention/data",
    )
    imports = _tree_imports(*canonical_roots)
    compatibility_prefixes = (
        "jlens_workspace.activations",
        "jlens_workspace.artifacts",
        "jlens_workspace.config",
        "jlens_workspace.jacobian",
        "jlens_workspace.matrix",
        "jlens_workspace.modeling",
        "jlens_workspace.concepts",
        "jlens_workspace.probes",
        "jlens_workspace.pursuit",
        "jlens_workspace.workflows",
        "jlens_workspace.concept_intervention.shared_protocol",
        "jlens_workspace.concept_intervention.generation",
        "jlens_workspace.concept_intervention.j_component",
        "jlens_workspace.concept_intervention.iti",
        "jlens_workspace.concept_intervention.raptor",
        "jlens_workspace.concept_intervention.k_diagnostic",
        "jlens_workspace.concept_intervention.candidate_rescore",
        "jlens_workspace.concept_intervention.comparison",
    )
    assert not any(module.startswith(compatibility_prefixes) for module in imports)


def test_three_method_batch_environment_does_not_require_git_cli() -> None:
    source = (
        ROOT / "Concept_intervention/scripts/three_method_env.sh"
    ).read_text(encoding="utf-8")

    assert 'git -C "${CODE_ROOT}" rev-parse HEAD' not in source
    assert "git_head_commit" in source
    assert 'JLENS_REPOSITORY_ROOT="${CODE_ROOT}"' in source
    assert 'JLENS_PYTHON:-${CODE_ROOT}/.venv/bin/python' in source


def test_three_method_submit_environment_overrides_cluster_mapping() -> None:
    helper = ROOT / "Concept_intervention/scripts/three_method_submit_env.sh"
    environment = os.environ.copy()
    environment.update(
        {
            "SLURM_ACCOUNT": "new_account",
            "SLURM_CPU_PARTITION": "cpu_queue",
            "SLURM_GPU_PARTITION": "gpu_queue",
            "SLURM_GPU_GRES": "gpu:h100:1",
            "SLURM_EXCLUDE": "badnode",
        }
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                "sbatch() { printf '<%s>\\n' \"$@\"; }; "
                f"source {helper}; "
                "three_method_sbatch_cpu --parsable cpu.slurm; "
                "three_method_sbatch_gpu --parsable gpu.slurm"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "<--account=new_account>" in result.stdout
    assert "<--partition=cpu_queue>" in result.stdout
    assert "<--partition=gpu_queue>" in result.stdout
    assert "<--gres=gpu:h100:1>" in result.stdout
    assert "<--exclude=badnode>" in result.stdout


def test_three_method_clean_server_bootstrap_is_content_pinned() -> None:
    bootstrap = (
        ROOT / "Concept_intervention/scripts/bootstrap_three_method_server.sh"
    ).read_text(encoding="utf-8")
    initial = (
        ROOT / "Concept_intervention/scripts/submit_three_method_interventions.sh"
    ).read_text(encoding="utf-8")
    full = (
        ROOT / "Concept_intervention/scripts/submit_three_method_full_grids.sh"
    ).read_text(encoding="utf-8")

    assert "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" in bootstrap
    assert "add492243ff905527e67aeb8b80c082af02207c3" in bootstrap
    assert "cf7405899174af39f3970e093e4b86bf0972ff87" in bootstrap
    assert "snapshot_download" in bootstrap
    assert "prepare_go_emotions.py" in bootstrap
    assert "three_method_sbatch_cpu" in initial
    assert "three_method_sbatch_gpu" in initial
    assert "three_method_sbatch_cpu" in full
    assert "three_method_sbatch_gpu" in full
    assert "SLURM_EXCLUDE THREE_METHOD_SUBMIT_ENV_LOADED" not in (
        ROOT / "Concept_intervention/scripts/three_method_submit_env.sh"
    ).read_text(encoding="utf-8")


def test_shared_occupancy_recovery_requires_explicit_overwrite_opt_in() -> None:
    source = (
        ROOT / "Concept_intervention/scripts/run_shared_j_occupancy.slurm"
    ).read_text(encoding="utf-8")

    assert 'OCCUPANCY_OVERWRITE:-0' in source
    assert 'OVERWRITE_ARGS+=(--overwrite)' in source


def test_formal_lanes_share_the_current_full_fit_prompt_artifact() -> None:
    assert not (ROOT / "Concept_intervention/configs/qwen35_4b_smoke.yaml").exists()
    concept = yaml.safe_load(
        (ROOT / "Concept_intervention/configs/qwen35_4b.yaml").read_text()
    )
    expected = concept["lens"]["fit_prompts_path"]
    assert expected == (
        "artifacts/data/go_emotions_7concept_full_ovr_v1/fit_prompts.jsonl"
    )
    for name in (
        "qwen35_4b.yaml",
        "qwen35_4b_centered.yaml",
        "qwen35_4b_row_normalized.yaml",
    ):
        matrix = yaml.safe_load((ROOT / "J_space/configs" / name).read_text())
        assert matrix["lens"]["fit_prompts_path"] == expected
        assert "goemotions_full" in matrix["lens"]["fit_output_path"]
