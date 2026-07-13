from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from jlens_workspace import cli


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _base_config(tmp_path: Path) -> dict[str, Any]:
    from jlens_workspace.config import ExperimentConfig

    payload = {
        "schema_version": 1,
        "experiment_name": "cli-test",
        "output_dir": str(tmp_path / "artifacts"),
        "model": {
            "model_id": "org/model",
            "revision": "model-commit",
            "tokenizer_id": "org/tokenizer",
            "tokenizer_revision": "tokenizer-commit",
            "dtype": "float32",
            "device": "cpu",
        },
        "lens": {
            "source": "local",
            "path_or_repo": str(tmp_path / "lens.pt"),
            "layers": [0],
            "target_layer": 1,
        },
        "dataset": {
            "source": "builtin",
            "path": str(tmp_path / "concepts.jsonl"),
        },
        "activations": {"layers": [0]},
        "probe": {},
        "alignment": {},
    }
    if "direction" in ExperimentConfig.model_fields:
        payload["direction"] = "concept_intervention"
    return payload


def _concept_record(*, row: int, split: str, label: int) -> dict[str, Any]:
    return {
        "concept_id": "abstract:test",
        "concept_name": "Test concept",
        "definition": "A deliberately abstract test property.",
        "abstractness": "abstract",
        "label": label,
        "text": f"unique example {row}",
        "prompt": f"prompt {row}",
        "response": f"response {row}",
        "split": split,
        # Positive and negative examples deliberately share a source group.
        "group_id": f"paired-{split}",
        "source": "synthetic",
        "license": "CC0-1.0",
    }


def test_parser_exposes_required_commands_without_importing_optional_stack() -> None:
    before = {
        name: name in sys.modules for name in ("torch", "transformers", "jlens")
    }
    parser = cli.build_parser()
    help_text = parser.format_help()
    for command in ("doctor", "config", "data", "concept", "matrix"):
        assert command in help_text
    for argv in (
        ["doctor"],
        ["config", "validate", "--config", "experiment.yaml"],
        ["data", "validate", "data.jsonl"],
        ["data", "validate", "--config", "experiment.yaml"],
        ["concept", "capture", "--config", "experiment.yaml"],
        ["concept", "fit-probes", "--config", "experiment.yaml"],
        ["concept", "align", "--config", "experiment.yaml"],
        ["concept", "run", "--config", "experiment.yaml"],
        ["matrix", "run", "--config", "experiment.yaml"],
    ):
        assert callable(parser.parse_args(argv).handler)
    assert {
        name: name in sys.modules for name in ("torch", "transformers", "jlens")
    } == before

    isolated = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from jlens_workspace import cli; cli.build_parser(); "
                "assert not {'torch', 'transformers', 'jlens'} & set(sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert isolated.returncode == 0, isolated.stderr


def test_top_level_help_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    assert "concept" in capsys.readouterr().out


def test_config_validate_is_json_and_torch_free(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_yaml(tmp_path / "experiment.yaml", _base_config(tmp_path))

    before = {name: name in sys.modules for name in ("torch", "transformers")}
    assert (
        cli.main(
            ["config", "validate", "--config", str(config_path), "--json"]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["model"]["revision"] == "model-commit"
    assert {name: name in sys.modules for name in before} == before


def test_config_validate_reports_concise_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invalid = _base_config(tmp_path) | {"misspelled": True}
    config_path = _write_yaml(tmp_path / "invalid.yaml", invalid)

    assert (
        cli.main(["config", "validate", "--config", str(config_path)]) == 2
    )
    assert "misspelled" in capsys.readouterr().err


def test_data_validate_accepts_matched_label_groups(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [
        _concept_record(row=row, split=split, label=label)
        for row, (split, label) in enumerate(
            (split, label)
            for split in ("train", "validation", "test")
            for label in (0, 1)
        )
    ]
    path = tmp_path / "concepts.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    before = "torch" in sys.modules
    assert cli.main(["data", "validate", "--data", str(path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["statistics"]["total_examples"] == 6
    assert payload["statistics"]["total_groups"] == 3
    assert ("torch" in sys.modules) is before


def test_data_validate_can_resolve_dataset_from_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [
        _concept_record(row=row, split=split, label=label)
        for row, (split, label) in enumerate(
            (split, label)
            for split in ("train", "validation", "test")
            for label in (0, 1)
        )
    ]
    data_path = tmp_path / "concepts.jsonl"
    data_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    raw = _base_config(tmp_path)
    raw["dataset"] = {
        "source": "builtin",
        "path": str(data_path),
        "min_per_label_per_split": 1,
    }
    config_path = _write_yaml(tmp_path / "experiment.yaml", raw)

    assert (
        cli.main(
            ["data", "validate", "--config", str(config_path), "--json"]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["path"] == str(data_path)
    assert payload["config"] == str(config_path)
    assert payload["statistics"]["total_examples"] == 6


def test_fit_probes_maps_probe_config_to_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from jlens_workspace import workflows

    raw = _base_config(tmp_path)
    raw["activations"] = {"layers": [2]}
    raw["probe"] = {
        "c_grid": [0.01, 0.1],
        "cv_folds": 3,
        "standardize": True,
        "class_weight": "balanced",
        "max_iter": 600,
        "seed": 17,
    }
    config_path = _write_yaml(tmp_path / "probe.yaml", raw)
    captured: dict[str, Any] = {}

    def fake_workflow(activation_path: Path, output: Path, **kwargs: Any) -> Any:
        captured.update(
            {"activation_path": activation_path, "output": output, "kwargs": kwargs}
        )
        probe = SimpleNamespace(
            layer=2,
            concept_id="abstract:test",
            chosen_C=0.1,
            probe_vector_path=output / "probe_vector.npy",
            metrics_path=output / "metrics.json",
        )
        return SimpleNamespace(
            output_dir=output,
            artifact_hash="abc123",
            probes=(probe,),
        )

    monkeypatch.setattr(workflows, "run_concept_probe_workflow", fake_workflow)
    activation_path = tmp_path / "activation-artifact"
    output_path = tmp_path / "probe-output"
    exit_code = cli.main(
        [
            "concept",
            "fit-probes",
            "--config",
            str(config_path),
            "--activations",
            str(activation_path),
            "--output",
            str(output_path),
            "--jobs",
            "2",
            "--json",
        ]
    )

    assert exit_code == 0
    assert captured["activation_path"] == activation_path
    assert captured["kwargs"]["layers"] == [2]
    assert captured["kwargs"]["C_grid"] == [0.01, 0.1]
    assert captured["kwargs"]["cv_splits"] == 3
    assert captured["kwargs"]["n_jobs"] == 2
    assert json.loads(capsys.readouterr().out)["n_probes"] == 1


def test_capture_forwards_add_special_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import jlens_workspace.activations as activation_module
    import jlens_workspace.data as data_module

    raw = _base_config(tmp_path)
    raw["dataset"] = {
        "source": "builtin",
        "path": str(tmp_path / "unused.jsonl"),
    }
    raw["activations"] = {
        "layers": [1],
        "add_special_tokens": False,
    }
    config_path = _write_yaml(tmp_path / "capture.yaml", raw)
    example = SimpleNamespace(to_dict=lambda: {"row": "synthetic"})
    bundle = SimpleNamespace(model=object(), tokenizer=object())
    captured: dict[str, Any] = {}
    monkeypatch.setattr(cli, "_load_examples", lambda *_args: [example])
    monkeypatch.setattr(cli, "_load_model_bundle", lambda _config: bundle)
    monkeypatch.setattr(data_module, "dataset_fingerprint", lambda _examples: "sha256:x")

    def fake_capture(**kwargs: Any) -> Path:
        captured.update(kwargs)
        return Path(kwargs["output_dir"])

    monkeypatch.setattr(
        activation_module, "capture_residual_activations", fake_capture
    )
    output = tmp_path / "activations"
    assert (
        cli.main(
            [
                "concept",
                "capture",
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )

    assert captured["model"] is bundle.model
    assert captured["layers"] == [1]
    assert captured["add_special_tokens"] is False
    assert json.loads(capsys.readouterr().out)["output"] == str(output)


def _write_probe_directory(
    path: Path, *, model_revision: str = "model-commit"
) -> dict[str, Any]:
    activation = {
        "artifact_hash": "activation-sha256",
        "coordinate": "resid_post",
        "representation": "last_non_padding_token",
        "add_special_tokens": True,
        "manifest": {
            "model_id": "org/model",
            "model_revision": model_revision,
            "tokenizer_id": "org/tokenizer",
            "tokenizer_revision": "tokenizer-commit",
            "dataset_source": "builtin",
            "dataset_revision": None,
            "notes": {"force_bos": None},
        },
    }
    concept = path / "layer_00" / "concept_abstract%3Atest"
    concept.mkdir(parents=True)
    np.save(concept / "probe_vector.npy", np.ones(4), allow_pickle=False)
    vector_sha256 = hashlib.sha256(
        (concept / "probe_vector.npy").read_bytes()
    ).hexdigest()
    (concept / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "layer": 0,
                "concept_id": "abstract:test",
                "artifact_hash": "activation-sha256",
                "activation": activation,
                "probe": {
                    "coordinate": "resid_post",
                    "dimension": 4,
                    "vector_file": "probe_vector.npy",
                    "vector_sha256": vector_sha256,
                },
            }
        )
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "activation_artifact_hash": "activation-sha256",
                "activation": activation,
            }
        )
    )
    return activation


def test_staged_probe_loading_enforces_end_to_end_identity(tmp_path: Path) -> None:
    probe_path = tmp_path / "probes"
    _write_probe_directory(probe_path)
    expected = {
        "model_id": "org/model",
        "model_revision": "model-commit",
        "tokenizer_id": "org/tokenizer",
        "tokenizer_revision": "tokenizer-commit",
        "dataset_source": "builtin",
        "dataset_revision": None,
        "force_bos": None,
        "coordinate": "resid_post",
        "representation": "last_non_padding_token",
        "add_special_tokens": True,
    }

    vectors = cli._load_probe_vectors(
        probe_path, 0, expected_identity=expected
    )
    assert set(vectors) == {"abstract:test"}
    with pytest.raises(ValueError, match="model_revision"):
        cli._load_probe_vectors(
            probe_path,
            0,
            expected_identity=expected | {"model_revision": "other-commit"},
        )


def test_staged_probe_loading_rejects_missing_provenance(tmp_path: Path) -> None:
    probe_path = tmp_path / "probes"
    _write_probe_directory(probe_path)
    (probe_path / "manifest.json").unlink()

    with pytest.raises(ValueError, match=r"manifest\.json"):
        cli._load_probe_vectors(
            probe_path, 0, expected_identity={"coordinate": "resid_post"}
        )


def test_lens_artifact_provenance_hashes_local_content(tmp_path: Path) -> None:
    from jlens_workspace.config import ExperimentConfig

    lens_path = tmp_path / "lens.pt"
    lens_path.write_bytes(b"exact lens bytes")
    raw = _base_config(tmp_path)
    raw["lens"]["path_or_repo"] = str(lens_path)
    config = ExperimentConfig.model_validate(raw)

    provenance = cli._lens_artifact_provenance(config, None)
    assert provenance["source"] == "local"
    assert provenance["path"] == str(lens_path)
    assert len(provenance["sha256"]) == 64
    assert provenance["identity_status"] == "content_hashed"


def test_lens_artifact_provenance_pins_hub_revision(tmp_path: Path) -> None:
    from jlens_workspace.config import ExperimentConfig

    raw = _base_config(tmp_path)
    raw["lens"] = {
        "source": "huggingface",
        "path_or_repo": "org/lenses",
        "filename": "lens.pt",
        "revision": "lens-commit",
        "layers": [0],
        "target_layer": 1,
    }
    config = ExperimentConfig.model_validate(raw)

    provenance = cli._lens_artifact_provenance(config, None)
    assert provenance == {
        "source": "huggingface",
        "repository": "org/lenses",
        "filename": "lens.pt",
        "revision": "lens-commit",
        "identity_status": "revision_pinned",
    }


def test_huggingface_lens_uses_explicit_revision_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jlens_workspace.config import ExperimentConfig
    from jlens_workspace.jacobian import OfficialJLensAdapter

    raw = _base_config(tmp_path)
    raw["lens"] = {
        "source": "huggingface",
        "path_or_repo": "org/lenses",
        "filename": "qwen/lens.pt",
        "revision": "lens-commit",
        "layers": [0, 1],
        "target_layer": 2,
        "n_fit_prompts": 37,
    }
    config = ExperimentConfig.model_validate(raw)

    def fake_tokenizer(texts: list[str], **_kwargs: Any) -> dict[str, list[list[int]]]:
        return {"input_ids": [[1] * 8 for _text in texts]}

    bundle = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(hidden_size=4)),
        tokenizer=fake_tokenizer,
    )
    sentinel = object()
    captured: dict[str, Any] = {}

    def fake_load(*args: Any, **kwargs: Any) -> object:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(OfficialJLensAdapter, "load", staticmethod(fake_load))

    assert cli._load_or_fit_lens(config, bundle) is sentinel
    assert captured["args"] == ("org/lenses",)
    assert captured["kwargs"]["revision"] == "lens-commit"
    metadata = captured["kwargs"]["metadata"]
    assert metadata.model_revision == "model-commit"
    assert metadata.tokenizer_revision == "tokenizer-commit"
    assert metadata.source_layers == (0, 1)
    assert metadata.target_layer == 2
    assert metadata.n_prompts == 37
    assert captured["kwargs"]["expected"]["d_model"] == 4


def test_local_lens_requires_embedded_metadata_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jlens_workspace.config import ExperimentConfig
    from jlens_workspace.jacobian import OfficialJLensAdapter

    raw = _base_config(tmp_path)
    raw["lens"] = {
        "source": "local",
        "path_or_repo": str(tmp_path / "lens-dir"),
        "filename": "lens.pt",
        "layers": [0],
        "target_layer": 1,
    }
    config = ExperimentConfig.model_validate(raw)
    bundle = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(hidden_size=4)),
        tokenizer=object(),
    )
    captured: dict[str, Any] = {}

    def fake_load(*args: Any, **kwargs: Any) -> object:
        captured.update({"args": args, "kwargs": kwargs})
        return object()

    monkeypatch.setattr(OfficialJLensAdapter, "load", staticmethod(fake_load))
    cli._load_or_fit_lens(config, bundle)

    assert "metadata" not in captured["kwargs"]
    assert captured["kwargs"]["expected"]["model_revision"] == "model-commit"
    assert "n_prompts" not in captured["kwargs"]["expected"]


def test_fit_lens_loads_exact_prompt_count_and_saves_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jlens_workspace.config import ExperimentConfig
    from jlens_workspace.jacobian import OfficialJLensAdapter

    prompts = tmp_path / "prompts.txt"
    prompts.write_text("first prompt\nsecond prompt\nunused prompt\n", encoding="utf-8")
    raw = _base_config(tmp_path)
    raw["lens"] = {
        "source": "fit",
        "layers": [0],
        "target_layer": 1,
        "fit_prompts_path": str(prompts),
        "n_fit_prompts": 2,
        "dim_batch": 7,
        "max_seq_len": 32,
        "skip_first": 3,
        "compile_blocks": True,
        "storage_dtype": "float32",
    }
    config = ExperimentConfig.model_validate(raw)

    def fit_tokenizer(texts: list[str], **_kwargs: Any) -> dict[str, list[list[int]]]:
        return {"input_ids": [[1] * 8 for _text in texts]}

    bundle = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(hidden_size=4)),
        tokenizer=fit_tokenizer,
    )
    wrapped = SimpleNamespace(d_model=4, n_layers=2)
    managed = object()
    captured: dict[str, Any] = {}

    def fake_from_hf(*args: Any, **kwargs: Any) -> object:
        captured["from_hf_args"] = args
        captured["from_hf_kwargs"] = kwargs
        return wrapped

    monkeypatch.setattr(OfficialJLensAdapter, "from_hf", staticmethod(fake_from_hf))

    def fake_fit(*args: Any, **kwargs: Any) -> object:
        captured["fit_args"] = args
        captured["fit_kwargs"] = kwargs
        return managed

    def fake_save(*args: Any, **kwargs: Any) -> Path:
        captured["save_args"] = args
        captured["save_kwargs"] = kwargs
        return Path(args[1])

    monkeypatch.setattr(OfficialJLensAdapter, "fit", staticmethod(fake_fit))
    monkeypatch.setattr(OfficialJLensAdapter, "save", staticmethod(fake_save))
    destination = tmp_path / "fitted.pt"

    assert cli._load_or_fit_lens(config, bundle, lens_output=destination) is managed
    assert captured["fit_args"] == (wrapped, ["first prompt", "second prompt"])
    assert captured["from_hf_args"] == (bundle.model, bundle.tokenizer)
    assert captured["from_hf_kwargs"]["compile"] is True
    assert "compile_blocks" not in captured["fit_kwargs"]
    metadata = captured["fit_kwargs"]["metadata"]
    assert metadata.model_revision == "model-commit"
    assert metadata.source_layers == (0,)
    assert captured["fit_kwargs"]["target_layer"] == 1
    assert captured["fit_kwargs"]["checkpoint_every"] == 25
    assert captured["fit_kwargs"]["resume"] is True
    assert Path(captured["fit_kwargs"]["checkpoint_path"]).name == "fitted.pt.checkpoint"
    assert captured["save_args"] == (managed, destination)
    assert captured["save_kwargs"]["dtype"] == "float32"


def test_fit_prompt_offset_selects_a_disjoint_block(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "".join(json.dumps({"text": f"prompt {index}"}) + "\n" for index in range(6))
    )

    assert cli._load_fit_prompts(prompts, 2, offset=0) == ["prompt 0", "prompt 1"]
    assert cli._load_fit_prompts(prompts, 2, offset=2) == ["prompt 2", "prompt 3"]
    with pytest.raises(ValueError, match="offset=5"):
        cli._load_fit_prompts(prompts, 2, offset=5)


def test_matrix_run_maps_all_numerical_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from jlens_workspace import workflows

    raw = _base_config(tmp_path)
    if "direction" in raw:
        raw["direction"] = "j_space"
    for field in ("dataset", "activations", "probe", "alignment"):
        raw.pop(field, None)
    raw["lens"] = {
        "source": "local",
        "path_or_repo": str(tmp_path / "lens.pt"),
        "layers": [0, 1],
        "target_layer": 2,
    }
    raw["matrix"] = {
        "layers": [1],
        "convention": "raw",
        "centered": False,
        "row_normalized": True,
        "normalize_by_total_weight": True,
        "zero_row_policy": "skip",
        "vocabulary_chunk_size": 23,
        "operator_compute_dtype": "float64",
        "accumulation_dtype": "float64",
        "energy_thresholds": [0.8, 0.95],
        "rank_relative_tolerance": 1e-5,
        "device": "cuda:2",
        "cpu_fallback": False,
    }
    config_path = _write_yaml(tmp_path / "matrix.yaml", raw)
    managed = object()
    effective = object()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(cli, "_load_model_bundle", lambda _config: object())
    monkeypatch.setattr(cli, "_load_or_fit_lens", lambda *_args, **_kwargs: managed)
    monkeypatch.setattr(
        cli, "_build_effective_unembedding", lambda *_args, **_kwargs: effective
    )

    def fake_run(
        received_lens: object,
        received_unembedding: object,
        layers: tuple[int, ...],
        output: Path,
        options: Any,
    ) -> Any:
        captured.update(
            {
                "lens": received_lens,
                "unembedding": received_unembedding,
                "layers": layers,
                "output": output,
                "options": options,
            }
        )
        layer = SimpleNamespace(
            layer=1,
            numerical_rank=3,
            entropy_effective_rank=2.5,
            participation_ratio=2.0,
        )
        return SimpleNamespace(
            output_dir=output, metrics_path=output / "metrics.json", layers=(layer,)
        )

    monkeypatch.setattr(workflows, "run_matrix_layers", fake_run)
    output = tmp_path / "matrix-output"
    assert (
        cli.main(
            [
                "matrix",
                "run",
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )

    options = captured["options"]
    assert captured["layers"] == (1,)
    assert options.centered is False
    assert options.row_normalized is True
    assert options.normalize_by_total_weight is True
    assert options.zero_row_policy == "skip"
    assert options.block_size == 23
    assert options.compute_device == "cuda:2"
    assert options.compute_dtype == "float64"
    assert options.accumulation_device == "cuda:2"
    assert options.decomposition_device == "cuda:2"
    assert options.cpu_fallback is False
    assert options.energy_thresholds == (0.8, 0.95)
    assert options.rank_rtol == pytest.approx(1e-5)
    payload = json.loads(capsys.readouterr().out)
    assert payload["layers"][0]["numerical_rank"] == 3
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_revision"] == "model-commit"
    assert manifest["notes"]["config_sha256"]
    assert manifest["notes"]["workflow"] == "matrix_run"


def test_concept_run_orders_stages_and_reuses_model_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import jlens_workspace.data as data_module

    config_path = _write_yaml(tmp_path / "run.yaml", _base_config(tmp_path))
    output = tmp_path / "run-output"
    examples = [object()]
    bundle = object()
    stages: list[str] = []
    monkeypatch.setattr(cli, "_load_examples", lambda *_args: examples)
    monkeypatch.setattr(cli, "_load_model_bundle", lambda _config: bundle)
    monkeypatch.setattr(
        data_module, "dataset_fingerprint", lambda _examples: "sha256:concepts"
    )

    def fake_capture(args: Any) -> int:
        stages.append("capture")
        assert args._examples is examples
        assert args._bundle is bundle
        args._payload = {"output": str(args.output)}
        return 0

    def fake_fit(args: Any) -> int:
        stages.append("fit_probes")
        assert args.activations == output / "activations"
        args._payload = {"output": str(args.output), "n_probes": 1}
        return 0

    def fake_align(args: Any) -> int:
        stages.append("align")
        assert args._bundle is bundle
        assert args.probes == output / "probes"
        args._payload = {"output": str(args.output)}
        return 0

    monkeypatch.setattr(cli, "_cmd_concept_capture", fake_capture)
    monkeypatch.setattr(cli, "_cmd_concept_fit_probes", fake_fit)
    monkeypatch.setattr(cli, "_cmd_concept_align", fake_align)

    assert (
        cli.main(
            [
                "concept",
                "run",
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )

    assert stages == ["capture", "fit_probes", "align"]
    payload = json.loads(capsys.readouterr().out)
    assert payload["workflow"] == "concept_run"
    assert payload["stages"]["fit_probes"]["n_probes"] == 1
    assert payload["manifest"]["dataset_hash"] == "sha256:concepts"
    assert payload["manifest"]["notes"]["intervention_stage_run"] is False
    assert (output / "manifest.json").is_file()
    assert json.loads((output / "run.json").read_text(encoding="utf-8")) == payload
