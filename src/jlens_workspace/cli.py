"""Command-line entry point for reproducible J-lens experiments.

The module imports only the Python standard library at import time.  Core
validation commands therefore remain usable without PyTorch, Transformers, or
the official ``jlens`` package; optional dependencies are imported only inside
commands that execute model or lens work.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import re
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_LAYER_DIRECTORY = re.compile(r"layer_(\d+)")
_CORE_PACKAGES = (
    ("numpy", "numpy"),
    ("pydantic", "pydantic"),
    ("PyYAML", "yaml"),
    ("scikit-learn", "sklearn"),
    ("scipy", "scipy"),
)
_OPTIONAL_PACKAGES = (
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("huggingface-hub", "huggingface_hub"),
    ("jlens", "jlens"),
)


def _add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")


def _add_overwrite_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing command output"
    )


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Accept the documented flag while retaining unambiguous legacy syntax."""

    parser.add_argument(
        "config_path",
        nargs="?",
        type=Path,
        metavar="CONFIG",
        help="experiment YAML (legacy positional form)",
    )
    parser.add_argument(
        "--config",
        dest="config_option",
        type=Path,
        metavar="CONFIG",
        help="experiment YAML",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the parser without importing any optional model dependency."""

    parser = argparse.ArgumentParser(
        prog="jlens-workspace",
        description="Abstract-concept steering and J-space geometry workflows",
    )
    parser.add_argument(
        "--debug", action="store_true", help="show tracebacks instead of concise errors"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="inspect core and optional dependencies")
    doctor.add_argument(
        "--require-llm",
        action="store_true",
        help="fail unless the complete optional model/J-lens stack is installed",
    )
    _add_json_flag(doctor)
    doctor.set_defaults(handler=_cmd_doctor)

    config = subparsers.add_parser("config", help="configuration operations")
    config_subparsers = config.add_subparsers(dest="config_command", required=True)
    config_validate = config_subparsers.add_parser(
        "validate", help="validate an ExperimentConfig YAML without loading a model"
    )
    _add_config_argument(config_validate)
    _add_json_flag(config_validate)
    config_validate.set_defaults(handler=_cmd_config_validate)

    data = subparsers.add_parser("data", help="concept dataset operations")
    data_subparsers = data.add_subparsers(dest="data_command", required=True)
    data_validate = data_subparsers.add_parser(
        "validate", help="validate concept JSONL coverage and leakage constraints"
    )
    data_validate.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="JSONL file or prepared directory (legacy positional form)",
    )
    data_validate.add_argument(
        "--config",
        dest="data_config",
        type=Path,
        metavar="CONFIG",
        help="experiment YAML whose dataset section should be validated",
    )
    data_validate.add_argument(
        "--data",
        type=Path,
        metavar="PATH",
        help="JSONL/prepared-directory override (or direct input without --config)",
    )
    data_validate.add_argument(
        "--min-per-label-per-split",
        type=int,
        default=None,
        help="minimum examples per concept/label/split cell (default: config value or 1)",
    )
    data_validate.add_argument(
        "--min-per-concept",
        type=int,
        default=None,
        help="minimum total examples for every concept (default: config value or 1)",
    )
    _add_json_flag(data_validate)
    data_validate.set_defaults(handler=_cmd_data_validate)

    lens_parser = subparsers.add_parser("lens", help="Jacobian-lens operations")
    lens_subparsers = lens_parser.add_subparsers(dest="lens_command", required=True)
    lens_fit = lens_subparsers.add_parser(
        "fit", help="fit or strictly reuse the configured local Jacobian lens"
    )
    _add_config_argument(lens_fit)
    lens_fit.add_argument("--output", type=Path, help="fitted lens artifact path")
    _add_json_flag(lens_fit)
    lens_fit.set_defaults(handler=_cmd_lens_fit)

    concept = subparsers.add_parser("concept", help="concept-steering workflows")
    concept_subparsers = concept.add_subparsers(dest="concept_command", required=True)

    capture = concept_subparsers.add_parser(
        "capture", help="capture resid_post activations for the configured dataset"
    )
    _add_config_argument(capture)
    capture.add_argument("--data", type=Path, help="override configured JSONL path")
    capture.add_argument("--output", type=Path, help="activation artifact directory")
    _add_overwrite_flag(capture)
    _add_json_flag(capture)
    capture.set_defaults(handler=_cmd_concept_capture)

    fit_probes = concept_subparsers.add_parser(
        "fit-probes", help="fit train-selected probes from an activation artifact"
    )
    _add_config_argument(fit_probes)
    fit_probes.add_argument("--activations", type=Path, help="activation artifact directory")
    fit_probes.add_argument("--output", type=Path, help="probe output directory")
    fit_probes.add_argument(
        "--layer", type=int, action="append", help="layer to fit; may be repeated"
    )
    fit_probes.add_argument(
        "--concept-id", action="append", help="concept ID to fit; may be repeated"
    )
    fit_probes.add_argument("--jobs", type=int, default=1, help="parallel CV jobs")
    _add_overwrite_flag(fit_probes)
    _add_json_flag(fit_probes)
    fit_probes.set_defaults(handler=_cmd_concept_fit_probes)

    align = concept_subparsers.add_parser(
        "align", help="align fitted probes with token J-directions"
    )
    _add_config_argument(align)
    align.add_argument("--probes", type=Path, help="probe artifact directory")
    align.add_argument("--output", type=Path, help="alignment output directory")
    align.add_argument(
        "--layer", type=int, action="append", help="layer to align; may be repeated"
    )
    align.add_argument(
        "--lens-output", type=Path, help="where a source=fit lens should be saved"
    )
    _add_overwrite_flag(align)
    _add_json_flag(align)
    align.set_defaults(handler=_cmd_concept_align)

    concept_run = concept_subparsers.add_parser(
        "run", help="run capture, probe fitting, and alignment as one workflow"
    )
    _add_config_argument(concept_run)
    concept_run.add_argument("--data", type=Path, help="override configured JSONL path")
    concept_run.add_argument("--output", type=Path, help="workflow root directory")
    concept_run.add_argument(
        "--layer", type=int, action="append", help="layer to fit/align; may be repeated"
    )
    concept_run.add_argument(
        "--concept-id", action="append", help="concept ID to fit; may be repeated"
    )
    concept_run.add_argument("--jobs", type=int, default=1, help="parallel probe CV jobs")
    concept_run.add_argument(
        "--lens-output", type=Path, help="where a source=fit lens should be saved"
    )
    _add_overwrite_flag(concept_run)
    _add_json_flag(concept_run)
    concept_run.set_defaults(handler=_cmd_concept_run)

    matrix = subparsers.add_parser("matrix", help="J-space matrix workflows")
    matrix_subparsers = matrix.add_subparsers(dest="matrix_command", required=True)
    matrix_run = matrix_subparsers.add_parser(
        "run", help="stream A_l^T A_l and save layerwise spectra/bases"
    )
    _add_config_argument(matrix_run)
    matrix_run.add_argument("--output", type=Path, help="matrix output directory")
    matrix_run.add_argument(
        "--layer", type=int, action="append", help="layer to analyze; may be repeated"
    )
    matrix_run.add_argument(
        "--lens-output", type=Path, help="where a source=fit lens should be saved"
    )
    _add_overwrite_flag(matrix_run)
    _add_json_flag(matrix_run)
    matrix_run.set_defaults(handler=_cmd_matrix_run)
    return parser


def _emit(payload: dict[str, Any], *, as_json: bool, message: str) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(message)


def _finish_command(
    args: argparse.Namespace, payload: dict[str, Any], *, message: str
) -> None:
    args._payload = payload
    if not getattr(args, "_quiet", False):
        _emit(payload, as_json=args.json, message=message)


def _package_status(distribution: str, module: str) -> dict[str, Any]:
    available = importlib.util.find_spec(module) is not None
    version = None
    if available:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"available": available, "version": version}


def _cmd_doctor(args: argparse.Namespace) -> int:
    core = {
        name: _package_status(name, module) for name, module in _CORE_PACKAGES
    }
    optional = {
        name: _package_status(name, module) for name, module in _OPTIONAL_PACKAGES
    }
    python_ok = sys.version_info >= (3, 11)
    core_ok = python_ok and all(status["available"] for status in core.values())
    llm_ok = all(status["available"] for status in optional.values())
    ok = core_ok and (llm_ok or not args.require_llm)
    payload = {
        "ok": ok,
        "python": {
            "version": ".".join(str(value) for value in sys.version_info[:3]),
            "supported": python_ok,
        },
        "core": core,
        "optional_llm": optional,
        "optional_llm_complete": llm_ok,
        "require_llm": args.require_llm,
        "expected_jlens_revision": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
    }
    installed_optional = sum(status["available"] for status in optional.values())
    _emit(
        payload,
        as_json=args.json,
        message=(
            f"core={'ok' if core_ok else 'missing'}; optional_llm="
            f"{installed_optional}/{len(optional)} installed"
        ),
    )
    return 0 if ok else 1


def _load_config(path: Path) -> Any:
    from jlens_workspace.config import load_experiment_config

    return load_experiment_config(path)


def _experiment_manifest(
    config: Any,
    config_path: Path,
    *,
    dataset_hash: str | None = None,
    notes: dict[str, Any] | None = None,
) -> Any:
    """Build one provenance record shared by top-level CLI workflows."""

    from jlens_workspace.artifacts import RunManifest, sha256_file

    lens = config.lens
    dataset = config.dataset
    manifest_notes = {
        "direction": config.direction,
        "coordinate": "resid_post/block_output",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "force_bos": config.model.force_bos,
    }
    if lens is not None:
        manifest_notes.update(
            {
                "lens_filename": lens.filename,
                "lens_storage_dtype": lens.storage_dtype,
                "lens_target_layer": lens.target_layer,
                "lens_fit_prompt_offset": lens.fit_prompt_offset,
            }
        )
    if notes:
        manifest_notes.update(notes)
    return RunManifest.for_workspace(
        Path.cwd(),
        experiment_name=config.experiment_name,
        seed=config.seed,
        model_id=config.model.model_id,
        model_revision=config.model.revision,
        tokenizer_id=config.model.tokenizer_id or config.model.model_id,
        tokenizer_revision=(
            config.model.tokenizer_revision or config.model.revision
        ),
        lens_source=(
            None
            if lens is None
            else f"{lens.source}:{lens.path_or_repo or 'runtime-fit'}"
        ),
        lens_revision=None if lens is None else lens.revision,
        dataset_source=(
            None if dataset is None else dataset.dataset_id or dataset.source
        ),
        dataset_revision=None if dataset is None else dataset.revision,
        dataset_hash=dataset_hash,
        notes=manifest_notes,
    )


def _normalize_config_argument(args: argparse.Namespace) -> None:
    """Resolve ``--config`` and the backwards-compatible positional form."""

    if not hasattr(args, "config_option") and not hasattr(args, "config_path"):
        return
    option = getattr(args, "config_option", None)
    positional = getattr(args, "config_path", None)
    if option is not None and positional is not None:
        raise ValueError("pass the experiment YAML via --config or positionally, not both")
    selected = option or positional
    if selected is None:
        raise ValueError("missing experiment YAML; pass --config CONFIG")
    args.config = selected


def _cmd_config_validate(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    sections = [
        name
        for name in ("dataset", "lens", "activations", "probe", "alignment", "matrix")
        if getattr(config, name) is not None
    ]
    payload = {
        "valid": True,
        "schema_version": config.schema_version,
        "experiment_name": config.experiment_name,
        "direction": getattr(config, "direction", None),
        "model": {
            "id": config.model.model_id,
            "revision": config.model.revision,
            "tokenizer_id": config.model.tokenizer_id or config.model.model_id,
            "tokenizer_revision": (
                config.model.tokenizer_revision or config.model.revision
            ),
        },
        "sections": sections,
    }
    _emit(
        payload,
        as_json=args.json,
        message=f"valid config: {config.experiment_name} ({', '.join(sections) or 'model only'})",
    )
    return 0


def _cmd_data_validate(args: argparse.Namespace) -> int:
    from jlens_workspace.data import (
        dataset_fingerprint,
        load_jsonl,
        load_jsonl_directory,
        validate_examples,
    )

    if args.path is not None and args.data is not None:
        raise ValueError("pass a data path via --data or positionally, not both")
    override = args.data or args.path
    config_path = args.data_config
    if config_path is not None:
        config = _load_config(config_path)
        dataset = _require_section(config, "dataset")
        minimum = (
            args.min_per_label_per_split
            if args.min_per_label_per_split is not None
            else dataset.min_per_label_per_split
        )
        minimum_concept = (
            args.min_per_concept
            if args.min_per_concept is not None
            else dataset.min_per_concept
        )
        examples = _load_examples(config, override)
        if override is not None:
            display_path = str(override)
        elif dataset.source == "axbench":
            display_path = f"axbench:{dataset.dataset_id or dataset.path}"
        else:
            display_path = str(dataset.path)
    else:
        if override is None:
            raise ValueError("missing data source; pass --config CONFIG or --data PATH")
        minimum = args.min_per_label_per_split or 1
        minimum_concept = args.min_per_concept or 1
        if override.is_dir():
            examples = load_jsonl_directory(
                override,
                min_per_label_per_split=minimum,
                min_per_concept=minimum_concept,
            )
        else:
            examples = load_jsonl(
                override,
                min_per_label_per_split=minimum,
                min_per_concept=minimum_concept,
            )
        display_path = str(override)
    statistics = validate_examples(
        examples,
        min_per_label_per_split=minimum,
        min_per_concept=minimum_concept,
    )
    payload = {
        "valid": True,
        "path": display_path,
        "config": str(config_path) if config_path is not None else None,
        "fingerprint": dataset_fingerprint(examples),
        "statistics": statistics.to_dict(),
    }
    _emit(
        payload,
        as_json=args.json,
        message=(
            f"valid data: examples={statistics.total_examples} "
            f"concepts={statistics.total_concepts} groups={statistics.total_groups}"
        ),
    )
    return 0


def _require_section(config: Any, name: str) -> Any:
    value = getattr(config, name)
    if value is None:
        raise ValueError(f"configuration requires a {name!r} section for this command")
    return value


def _load_examples(config: Any, override: Path | None = None) -> list[Any]:
    from jlens_workspace.data import (
        load_jsonl,
        load_jsonl_directory,
        prepare_axbench,
    )

    dataset = _require_section(config, "dataset")
    if override is not None:
        path = override
        return (
            load_jsonl_directory(
                path,
                min_per_label_per_split=dataset.min_per_label_per_split,
                min_per_concept=dataset.min_per_concept,
            )
            if path.is_dir()
            else load_jsonl(
                path,
                min_per_label_per_split=dataset.min_per_label_per_split,
                min_per_concept=dataset.min_per_concept,
            )
        )
    if dataset.source == "axbench":
        prepared = prepare_axbench(
            dataset.allowlist_path,
            dataset.path,
            streaming=dataset.streaming,
            seed=config.seed,
        )
        return list(prepared.examples)
    path = Path(dataset.path)
    if path.is_dir():
        return load_jsonl_directory(
            path,
            min_per_label_per_split=dataset.min_per_label_per_split,
            min_per_concept=dataset.min_per_concept,
        )
    return load_jsonl(
        path,
        min_per_label_per_split=dataset.min_per_label_per_split,
        min_per_concept=dataset.min_per_concept,
    )


def _load_model_bundle(config: Any) -> Any:
    from jlens_workspace.modeling import load_hf_bundle

    return load_hf_bundle(config.model)


def _cmd_concept_capture(args: argparse.Namespace) -> int:
    from jlens_workspace.activations import capture_residual_activations
    from jlens_workspace.data import dataset_fingerprint

    config = _load_config(args.config)
    activation = _require_section(config, "activations")
    _require_section(config, "dataset")
    examples = getattr(args, "_examples", None)
    if examples is None:
        examples = _load_examples(config, args.data)
    bundle = getattr(args, "_bundle", None)
    if bundle is None:
        bundle = _load_model_bundle(config)
    destination = args.output or Path(config.output_dir) / "activations"
    manifest = _experiment_manifest(
        config,
        args.config,
        dataset_hash=dataset_fingerprint(examples),
        notes={
            "stage": "activation_capture",
            "representation": "last_non_padding_token",
            "example_layout": (
                "shared_source_by_group"
                if activation.share_examples_by_group
                else "concept_expanded"
            ),
        },
    )
    output = capture_residual_activations(
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        examples=examples,
        layers=activation.layers,
        output_dir=destination,
        batch_size=activation.batch_size,
        max_length=activation.max_length,
        add_special_tokens=activation.add_special_tokens,
        manifest=manifest,
        share_examples_by_group=activation.share_examples_by_group,
        require_complete_concept_matrix=activation.require_complete_concept_matrix,
        overwrite=args.overwrite,
    )
    metadata_path = output / "metadata.json"
    capture_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {"n_examples": len(examples)}
    )
    payload = {
        "output": str(output),
        "n_input_task_rows": len(examples),
        "n_examples": capture_metadata["n_examples"],
        "n_concepts": capture_metadata.get("n_concepts"),
        "label_shape": capture_metadata.get("label_shape"),
        "layers": list(activation.layers),
        "coordinate": "resid_post",
    }
    _finish_command(
        args,
        payload,
        message=(
            f"captured {capture_metadata['n_examples']} source examples x "
            f"{len(activation.layers)} layers -> {output}"
        ),
    )
    return 0


def _cmd_concept_fit_probes(args: argparse.Namespace) -> int:
    from jlens_workspace.workflows import run_concept_probe_workflow

    config = _load_config(args.config)
    probe = _require_section(config, "probe")
    if probe.scoring != "roc_auc":
        raise ValueError("the current probe workflow requires scoring='roc_auc'")
    activation_path = args.activations or Path(config.output_dir) / "activations"
    destination = args.output or Path(config.output_dir) / "probes"
    configured_layers = (
        config.activations.layers if config.activations is not None else None
    )
    result = run_concept_probe_workflow(
        activation_path,
        destination,
        layers=args.layer or configured_layers,
        concept_ids=args.concept_id,
        C_grid=probe.c_grid,
        cv_splits=probe.cv_folds,
        standardize=probe.standardize,
        class_weight=probe.class_weight,
        random_state=probe.seed,
        max_iter=probe.max_iter,
        n_jobs=args.jobs,
        overwrite=args.overwrite,
    )
    payload = {
        "output": str(result.output_dir),
        "artifact_hash": result.artifact_hash,
        "n_probes": len(result.probes),
        "probes": [
            {
                "layer": output.layer,
                "concept_id": output.concept_id,
                "chosen_C": output.chosen_C,
                "vector": str(output.probe_vector_path),
                "metrics": str(output.metrics_path),
            }
            for output in result.probes
        ],
    }
    _finish_command(
        args,
        payload,
        message=f"fit {len(result.probes)} probes -> {result.output_dir}",
    )
    return 0


def _canonical_lens_layers(lens: Any) -> tuple[int, ...]:
    layers = tuple(int(layer) for layer in lens.layers)
    if not layers or layers != tuple(sorted(set(layers))) or any(layer < 0 for layer in layers):
        raise ValueError("lens.layers must be non-negative, sorted, and unique")
    return layers


def _model_d_model(model: Any) -> int:
    model_config = getattr(model, "config", None)
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(model_config, name, None)
        if value is not None and int(value) > 0:
            return int(value)
    getter = getattr(model, "get_output_embeddings", None)
    head = getter() if callable(getter) else getattr(model, "lm_head", None)
    weight = getattr(head, "weight", None)
    shape = getattr(weight, "shape", ())
    if len(shape) == 2 and int(shape[1]) > 0:
        return int(shape[1])
    raise ValueError("cannot determine model residual width for lens validation")


def _lens_convention(config: Any) -> str:
    if config.matrix is not None:
        return config.matrix.convention
    if config.alignment is not None:
        return config.alignment.convention
    return "raw"


def _lens_expected(config: Any, d_model: int) -> dict[str, Any]:
    lens = _require_section(config, "lens")
    expected: dict[str, Any] = {
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "tokenizer_id": config.model.tokenizer_id or config.model.model_id,
        "tokenizer_revision": (
            config.model.tokenizer_revision or config.model.revision
        ),
        "d_model": d_model,
        "source_layers": _canonical_lens_layers(lens),
        "norm_convention": _lens_convention(config),
    }
    if lens.target_layer is not None:
        expected["target_layer"] = int(lens.target_layer)
    if "n_fit_prompts" in lens.model_fields_set:
        expected["n_prompts"] = int(lens.n_fit_prompts)
    return expected


def _default_fitted_lens_path(config: Any) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.experiment_name).strip("._")
    if not slug:
        slug = "fitted_lens"
    return Path(config.output_dir).parent / "lenses" / f"{slug}.pt"


def _fitted_lens_path(config: Any, lens_output: Path | None) -> Path:
    lens = _require_section(config, "lens")
    return lens_output or (
        Path(lens.fit_output_path)
        if lens.fit_output_path is not None
        else _default_fitted_lens_path(config)
    )


def _lens_artifact_provenance(
    config: Any, lens_output: Path | None
) -> dict[str, Any]:
    """Return a content hash for local lenses or an exact Hub identity."""

    from jlens_workspace.artifacts import sha256_file

    lens = _require_section(config, "lens")
    if lens.source == "huggingface":
        return {
            "source": "huggingface",
            "repository": lens.path_or_repo,
            "filename": lens.filename,
            "revision": lens.revision,
            "identity_status": "revision_pinned",
        }
    if lens.source == "fit":
        path = _fitted_lens_path(config, lens_output)
    else:
        path = Path(lens.path_or_repo)
        if path.is_dir():
            path = path / (lens.filename or "lens.pt")
    return {
        "source": lens.source,
        "path": str(path),
        "sha256": sha256_file(path) if path.is_file() else None,
        "identity_status": "content_hashed" if path.is_file() else "missing",
    }


def _fit_checkpoint_path(config: Any, destination: Path) -> Path:
    lens = _require_section(config, "lens")
    if lens.fit_checkpoint_path is not None:
        return Path(lens.fit_checkpoint_path)
    return destination.with_suffix(destination.suffix + ".checkpoint")


def _load_fit_prompts(
    path: str | Path, limit: int, *, offset: int = 0
) -> list[str]:
    source = Path(path)
    if offset < 0:
        raise ValueError("fit prompt offset must be non-negative")
    lines = source.read_text(encoding="utf-8").splitlines()
    prompts: list[str] = []
    usable = 0
    jsonl = source.suffix.casefold() in {".jsonl", ".json"}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if jsonl:
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{source}:{line_number}: invalid prompt JSON: {error.msg}"
                ) from error
            if isinstance(value, str):
                prompt = value
            elif isinstance(value, dict):
                prompt = value.get("text") or value.get("prompt")
            else:
                prompt = None
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"{source}:{line_number}: expected a string or text/prompt object"
                )
        else:
            prompt = line
        if usable < offset:
            usable += 1
            continue
        usable += 1
        prompts.append(prompt.strip())
        if len(prompts) == limit:
            break
    if len(prompts) < limit:
        raise ValueError(
            f"fit prompt file has fewer than offset={offset} + limit={limit} "
            "usable prompts"
        )
    return prompts


def _fit_prompt_token_lengths(
    tokenizer: Any,
    prompts: list[str],
    *,
    max_seq_len: int,
    skip_first: int,
) -> tuple[int, int]:
    encoded = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"]
    lengths = [len(row) for row in input_ids]
    invalid = [index for index, length in enumerate(lengths) if length <= skip_first + 1]
    if invalid:
        raise ValueError(
            f"{len(invalid)} J-lens prompts have no valid positions after skip_first="
            f"{skip_first}; first invalid prompt index is {invalid[0]}"
        )
    return min(lengths), max(lengths)


def _load_or_fit_lens(
    config: Any, bundle: Any, *, lens_output: Path | None = None
) -> Any:
    """Resolve hub/local/fit sources with explicit immutable identity metadata."""

    from jlens_workspace.artifacts import (
        atomic_write_json,
        sha256_file,
        stable_hash,
    )
    from jlens_workspace.jacobian import JLensMetadata, OfficialJLensAdapter

    lens = _require_section(config, "lens")
    layers = _canonical_lens_layers(lens)
    tokenizer_id = config.model.tokenizer_id or config.model.model_id
    tokenizer_revision = config.model.tokenizer_revision or config.model.revision

    if lens.source == "fit":
        destination = _fitted_lens_path(config, lens_output)
        prompt_sha256 = sha256_file(lens.fit_prompts_path)
        prompts = _load_fit_prompts(
            lens.fit_prompts_path,
            lens.n_fit_prompts,
            offset=lens.fit_prompt_offset,
        )
        minimum_tokens, maximum_tokens = _fit_prompt_token_lengths(
            bundle.tokenizer,
            prompts,
            max_seq_len=lens.max_seq_len,
            skip_first=lens.skip_first,
        )
        d_model = _model_d_model(bundle.model)
        target_layer = int(lens.target_layer)
        fit_identity = {
            "model_id": config.model.model_id,
            "model_revision": config.model.revision,
            "tokenizer_id": tokenizer_id,
            "tokenizer_revision": tokenizer_revision,
            "fit_prompts_sha256": prompt_sha256,
            "n_fit_prompts": lens.n_fit_prompts,
            "fit_prompt_offset": lens.fit_prompt_offset,
            "selected_fit_prompts_sha256": stable_hash(prompts),
            "source_layers": list(layers),
            "target_layer": target_layer,
            "dim_batch": lens.dim_batch,
            "max_seq_len": lens.max_seq_len,
            "skip_first": lens.skip_first,
            "compile_blocks": lens.compile_blocks,
            "force_bos": config.model.force_bos,
            "storage_dtype": lens.storage_dtype,
            "fit_prompt_min_tokens": minimum_tokens,
            "fit_prompt_max_tokens": maximum_tokens,
        }
        expected = _lens_expected(config, d_model)
        expected["target_layer"] = target_layer
        expected["n_prompts"] = lens.n_fit_prompts
        if destination.is_file():
            managed = OfficialJLensAdapter.load(destination, expected=expected)
            for key, value in fit_identity.items():
                if managed.metadata.extra.get(key) != value:
                    raise ValueError(
                        f"existing fitted lens metadata {key!r}="
                        f"{managed.metadata.extra.get(key)!r}, expected {value!r}"
                    )
            expected_dtype = f"torch.{lens.storage_dtype}"
            observed_dtypes = {
                str(matrix.dtype) for matrix in managed.lens.jacobians.values()
            }
            if observed_dtypes != {expected_dtype}:
                raise ValueError(
                    f"existing fitted lens dtype is {sorted(observed_dtypes)}, "
                    f"expected {expected_dtype}"
                )
            return managed
        if destination.exists():
            raise FileExistsError(f"fitted lens output is not a file: {destination}")

        wrapper_kwargs = {"compile": lens.compile_blocks}
        if config.model.force_bos is not None:
            wrapper_kwargs["force_bos"] = config.model.force_bos
        wrapped = OfficialJLensAdapter.from_hf(
            bundle.model, bundle.tokenizer, **wrapper_kwargs
        )
        if int(wrapped.d_model) != d_model:
            raise ValueError(
                f"wrapped model d_model={wrapped.d_model} differs from model d_model={d_model}"
            )
        metadata = JLensMetadata(
            model_id=config.model.model_id,
            model_revision=config.model.revision,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            d_model=d_model,
            source_layers=layers,
            target_layer=target_layer,
            norm_convention=_lens_convention(config),
            n_prompts=lens.n_fit_prompts,
            extra={
                "lens_source": "fit",
                "fit_prompts_path": str(lens.fit_prompts_path),
                **fit_identity,
            },
        )
        checkpoint = _fit_checkpoint_path(config, destination)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_manifest = checkpoint.with_suffix(checkpoint.suffix + ".manifest.json")
        if lens.resume and checkpoint.is_file():
            if not checkpoint_manifest.is_file():
                raise FileNotFoundError(
                    f"refusing an unidentified J-lens checkpoint: {checkpoint_manifest}"
                )
            observed_identity = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
            if observed_identity != fit_identity:
                raise ValueError(
                    "J-lens checkpoint identity does not match the current model, "
                    "tokenizer, prompts, or fit settings"
                )
        atomic_write_json(checkpoint_manifest, fit_identity)
        managed = OfficialJLensAdapter.fit(
            wrapped,
            prompts,
            metadata=metadata,
            source_layers=list(layers),
            target_layer=target_layer,
            dim_batch=lens.dim_batch,
            max_seq_len=lens.max_seq_len,
            skip_first=lens.skip_first,
            checkpoint_path=str(checkpoint),
            checkpoint_every=lens.checkpoint_every,
            resume=lens.resume,
        )
        OfficialJLensAdapter.save(managed, destination, dtype=lens.storage_dtype)
        return managed

    d_model = _model_d_model(bundle.model)
    expected = _lens_expected(config, d_model)
    if lens.source == "local":
        return OfficialJLensAdapter.load(
            lens.path_or_repo,
            filename=lens.filename or "lens.pt",
            expected=expected,
        )

    if lens.source != "huggingface":
        raise ValueError(f"unsupported lens source: {lens.source!r}")
    if not lens.revision:
        raise ValueError("a huggingface lens requires an explicit revision")
    metadata = JLensMetadata(
        model_id=config.model.model_id,
        model_revision=config.model.revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        d_model=d_model,
        source_layers=layers,
        target_layer=lens.target_layer,
        norm_convention=_lens_convention(config),
        n_prompts=(
            lens.n_fit_prompts if "n_fit_prompts" in lens.model_fields_set else None
        ),
        extra={
            "lens_source": "huggingface",
            "repository": lens.path_or_repo,
            "filename": lens.filename,
            "revision": lens.revision,
        },
    )
    return OfficialJLensAdapter.load(
        lens.path_or_repo,
        filename=lens.filename,
        revision=lens.revision,
        metadata=metadata,
        expected=expected,
    )


def _cmd_lens_fit(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    lens = _require_section(config, "lens")
    if lens.source != "fit":
        raise ValueError("lens fit requires lens.source='fit'")
    bundle = _load_model_bundle(config)
    managed = _load_or_fit_lens(config, bundle, lens_output=args.output)
    destination = _fitted_lens_path(config, args.output)
    payload = {
        "output": str(destination),
        "layers": list(managed.source_layers),
        "n_prompts": managed.metadata.n_prompts,
        "storage_dtype": lens.storage_dtype,
        "metadata": managed.metadata.to_dict(),
        "lens_artifact": _lens_artifact_provenance(config, args.output),
    }
    _finish_command(
        args,
        payload,
        message=f"fitted/reused J-lens over {managed.metadata.n_prompts} prompts -> {destination}",
    )
    return 0


def _probe_layers(path: Path) -> tuple[int, ...]:
    layers: list[int] = []
    if not path.is_dir():
        raise FileNotFoundError(f"probe directory does not exist: {path}")
    for candidate in sorted(path.iterdir()):
        if not candidate.is_dir():
            continue
        match = _LAYER_DIRECTORY.fullmatch(candidate.name)
        if match is not None:
            layers.append(int(match.group(1)))
    if not layers:
        raise ValueError(f"no layer_XX probe directories found in {path}")
    return tuple(layers)


def _probe_activation_identity(
    activation: Mapping[str, Any], *, source: Path
) -> dict[str, Any]:
    manifest = activation.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"{source}: activation provenance is missing its manifest")
    notes = manifest.get("notes")
    if not isinstance(notes, dict):
        raise ValueError(f"{source}: activation manifest is missing notes")
    return {
        "model_id": manifest.get("model_id"),
        "model_revision": manifest.get("model_revision"),
        "tokenizer_id": manifest.get("tokenizer_id"),
        "tokenizer_revision": manifest.get("tokenizer_revision"),
        "dataset_source": manifest.get("dataset_source"),
        "dataset_revision": manifest.get("dataset_revision"),
        "force_bos": notes.get("force_bos"),
        "config_sha256": notes.get("config_sha256"),
        "coordinate": activation.get("coordinate"),
        "representation": activation.get("representation"),
        "add_special_tokens": activation.get("add_special_tokens"),
    }


def _expected_probe_identity(
    config: Any, config_path: Path | None = None
) -> dict[str, Any]:
    from jlens_workspace.artifacts import sha256_file

    activation = _require_section(config, "activations")
    dataset = _require_section(config, "dataset")
    identity = {
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "tokenizer_id": config.model.tokenizer_id or config.model.model_id,
        "tokenizer_revision": (
            config.model.tokenizer_revision or config.model.revision
        ),
        "dataset_source": dataset.dataset_id or dataset.source,
        "dataset_revision": dataset.revision,
        "force_bos": config.model.force_bos,
        "coordinate": "resid_post",
        "representation": "last_non_padding_token",
        "add_special_tokens": activation.add_special_tokens,
    }
    if config_path is not None:
        identity["config_sha256"] = sha256_file(config_path)
    return identity


def _load_probe_vectors(
    path: Path,
    layer: int,
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import numpy as np

    root_activation: Mapping[str, Any] | None = None
    root_hash: str | None = None
    if expected_identity is not None:
        root_manifest_path = path / "manifest.json"
        if not root_manifest_path.is_file():
            raise ValueError(
                f"probe directory is missing provenance manifest.json: {path}"
            )
        root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        root_activation = root_manifest.get("activation")
        root_hash = root_manifest.get("activation_artifact_hash")
        if not isinstance(root_activation, dict) or not isinstance(root_hash, str):
            raise ValueError(f"{root_manifest_path}: invalid activation provenance")
        observed_identity = _probe_activation_identity(
            root_activation, source=root_manifest_path
        )
        for field, expected in expected_identity.items():
            if observed_identity.get(field) != expected:
                raise ValueError(
                    f"{root_manifest_path}: probe {field}="
                    f"{observed_identity.get(field)!r} differs from configured "
                    f"{field}={expected!r}"
                )

    layer_path = path / f"layer_{layer:02d}"
    vectors: dict[str, Any] = {}
    for metrics_path in sorted(layer_path.glob("concept_*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("layer") != layer:
            raise ValueError(f"{metrics_path}: layer metadata mismatch")
        concept_id = payload.get("concept_id")
        if not isinstance(concept_id, str) or not concept_id:
            raise ValueError(f"{metrics_path}: missing concept_id")
        if concept_id in vectors:
            raise ValueError(f"duplicate probe for concept {concept_id!r} at layer {layer}")
        probe_metadata = payload.get("probe")
        if not isinstance(probe_metadata, dict):
            raise ValueError(f"{metrics_path}: missing probe metadata")
        if probe_metadata.get("coordinate") != "resid_post":
            raise ValueError(f"{metrics_path}: probe coordinate must be resid_post")
        if expected_identity is not None:
            if payload.get("artifact_hash") != root_hash:
                raise ValueError(f"{metrics_path}: activation artifact hash mismatch")
            if payload.get("activation") != root_activation:
                raise ValueError(f"{metrics_path}: activation provenance mismatch")
        vector_file = probe_metadata.get("vector_file", "probe_vector.npy")
        vector_path = metrics_path.parent / vector_file
        expected_vector_hash = probe_metadata.get("vector_sha256")
        if not isinstance(expected_vector_hash, str):
            raise ValueError(f"{metrics_path}: missing probe vector SHA-256")
        from jlens_workspace.artifacts import sha256_file

        if sha256_file(vector_path) != expected_vector_hash:
            raise ValueError(f"{vector_path}: probe vector SHA-256 mismatch")
        vector = np.load(vector_path, allow_pickle=False)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError(f"{metrics_path.parent}: probe vector must be finite and 1D")
        if probe_metadata.get("dimension") != int(vector.shape[0]):
            raise ValueError(f"{metrics_path.parent}: probe dimension metadata mismatch")
        vectors[concept_id] = vector
    if not vectors:
        raise ValueError(f"no concept probes found for layer {layer} in {layer_path}")
    return vectors


def _prepare_fresh_output(path: Path, *, overwrite: bool, label: str) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"{label} output is not a directory: {path}")
        if any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(f"{label} output already exists: {path}")
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _build_effective_unembedding(config: Any, bundle: Any, convention: str) -> Any:
    from jlens_workspace.jacobian import (
        build_effective_unembedding,
        restrict_effective_unembedding,
    )

    effective = build_effective_unembedding(
        bundle.model,
        convention=convention,
        model_id=config.model.model_id,
        model_revision=config.model.revision,
    )
    return restrict_effective_unembedding(effective, len(bundle.tokenizer))


def _cmd_concept_align(args: argparse.Namespace) -> int:
    from jlens_workspace.artifacts import atomic_write_json, sha256_file
    from jlens_workspace.matrix import TokenFrameOperator
    from jlens_workspace.workflows import run_batched_probe_j_alignment

    config = _load_config(args.config)
    alignment = _require_section(config, "alignment")
    _require_section(config, "lens")
    if alignment.metric != "cosine":
        raise ValueError("batched CLI alignment currently requires metric='cosine'")
    probe_path = args.probes or Path(config.output_dir) / "probes"
    layers = tuple(args.layer) if args.layer else _probe_layers(probe_path)
    destination = args.output or Path(config.output_dir) / "alignment"
    _prepare_fresh_output(destination, overwrite=args.overwrite, label="alignment")

    bundle = getattr(args, "_bundle", None)
    if bundle is None:
        bundle = _load_model_bundle(config)
    managed_lens = _load_or_fit_lens(config, bundle, lens_output=args.lens_output)
    lens_artifact = _lens_artifact_provenance(config, args.lens_output)
    probe_manifest_path = probe_path / "manifest.json"
    if not probe_manifest_path.is_file():
        raise ValueError(f"probe directory is missing manifest.json: {probe_path}")
    probe_artifact = {
        "path": str(probe_path),
        "manifest_sha256": sha256_file(probe_manifest_path),
        "manifest": json.loads(probe_manifest_path.read_text(encoding="utf-8")),
    }
    missing = sorted(set(layers).difference(managed_lens.source_layers))
    if missing:
        raise ValueError(
            f"probe layers {missing} are absent from fitted lens {managed_lens.source_layers}"
        )
    convention = _lens_convention(config)
    effective = _build_effective_unembedding(config, bundle, convention)
    tokenizer_size = len(bundle.tokenizer)
    if effective.vocab_size != tokenizer_size:
        raise ValueError(
            f"tokenizer has {tokenizer_size} token IDs but the model unembedding has "
            f"{effective.vocab_size} rows; J-lens token alignment is undefined"
        )
    compute_device = config.matrix.device if config.matrix is not None else config.model.device
    layer_results: list[dict[str, Any]] = []
    for layer in layers:
        probe_vectors = _load_probe_vectors(
            probe_path,
            layer,
            expected_identity=_expected_probe_identity(config, args.config),
        )
        operator = TokenFrameOperator.from_lens(
            managed_lens,
            layer,
            effective,
            block_size=alignment.vocabulary_chunk_size,
            compute_device=compute_device,
        )
        layer_output = destination / f"layer_{layer:02d}"
        run_batched_probe_j_alignment(
            probe_vectors=probe_vectors,
            operator=operator,
            output_dir=layer_output,
            tokenizer=bundle.tokenizer,
            top_k=alignment.top_k,
            candidate_pool_size=alignment.candidate_pool_size,
            sparse_components=alignment.sparse_components,
            decompose=alignment.decompose,
            random_control_seeds=tuple(alignment.random_control_seeds),
            metadata={
                "experiment_name": config.experiment_name,
                "layer": layer,
                "model_id": config.model.model_id,
                "model_revision": config.model.revision,
                "lens": managed_lens.metadata.to_dict(),
                "lens_artifact": lens_artifact,
                "probe_artifact": probe_artifact,
                "convention": convention,
            },
        )
        layer_results.append(
            {
                "layer": layer,
                "n_probes": len(probe_vectors),
                "output": str(layer_output),
            }
        )
    payload = {
        "schema_version": 1,
        "workflow": "concept_alignment",
        "output": str(destination),
        "layers": layer_results,
        "lens_metadata": managed_lens.metadata.to_dict(),
        "lens_artifact": lens_artifact,
        "probe_artifact": probe_artifact,
        "convention": convention,
    }
    atomic_write_json(destination / "alignment.json", payload)
    _finish_command(
        args,
        payload,
        message=f"aligned {sum(row['n_probes'] for row in layer_results)} probes -> {destination}",
    )
    return 0


def _concept_run_inputs(config: Any, args: argparse.Namespace) -> tuple[Path, ...]:
    inputs = [Path(args.config)]
    if args.data is not None:
        inputs.append(args.data)
    elif config.dataset is not None and config.dataset.source != "axbench":
        inputs.append(Path(config.dataset.path))
    if config.dataset is not None and config.dataset.allowlist_path:
        inputs.append(Path(config.dataset.allowlist_path))
    if config.lens is not None:
        if config.lens.source == "local" and config.lens.path_or_repo:
            inputs.append(Path(config.lens.path_or_repo))
        if config.lens.fit_prompts_path:
            inputs.append(Path(config.lens.fit_prompts_path))
    return tuple(inputs)


def _prepare_concept_run_root(
    destination: Path,
    *,
    inputs: tuple[Path, ...],
    overwrite: bool,
) -> None:
    resolved_destination = destination.resolve()
    for input_path in inputs:
        resolved_input = input_path.resolve()
        if resolved_input == resolved_destination or resolved_destination in resolved_input.parents:
            raise ValueError(
                f"workflow output {destination} contains required input {input_path}; "
                "refusing a potentially destructive overwrite"
            )
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(f"concept workflow output is not a directory: {destination}")
        if any(destination.iterdir()):
            if not overwrite:
                raise FileExistsError(f"concept workflow output already exists: {destination}")
            shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)


def _cmd_concept_run(args: argparse.Namespace) -> int:
    from jlens_workspace.artifacts import atomic_write_json
    from jlens_workspace.data import dataset_fingerprint

    config = _load_config(args.config)
    _require_section(config, "dataset")
    _require_section(config, "activations")
    _require_section(config, "probe")
    _require_section(config, "alignment")
    _require_section(config, "lens")
    destination = args.output or Path(config.output_dir)
    _prepare_concept_run_root(
        destination,
        inputs=_concept_run_inputs(config, args),
        overwrite=args.overwrite,
    )

    try:
        examples = _load_examples(config, args.data)
        manifest = _experiment_manifest(
            config,
            args.config,
            dataset_hash=dataset_fingerprint(examples),
            notes={
                "workflow": "concept_run",
                "stages": ["capture", "fit_probes", "align"],
                "intervention_stage_run": False,
            },
        )
        bundle = _load_model_bundle(config)
        capture_args = argparse.Namespace(
            config=args.config,
            data=args.data,
            output=destination / "activations",
            overwrite=False,
            json=False,
            _quiet=True,
            _examples=examples,
            _bundle=bundle,
        )
        _cmd_concept_capture(capture_args)
        probe_args = argparse.Namespace(
            config=args.config,
            activations=destination / "activations",
            output=destination / "probes",
            layer=args.layer,
            concept_id=args.concept_id,
            jobs=args.jobs,
            overwrite=False,
            json=False,
            _quiet=True,
        )
        _cmd_concept_fit_probes(probe_args)
        alignment_args = argparse.Namespace(
            config=args.config,
            probes=destination / "probes",
            output=destination / "alignment",
            layer=args.layer,
            lens_output=args.lens_output,
            overwrite=False,
            json=False,
            _quiet=True,
            _bundle=bundle,
        )
        _cmd_concept_align(alignment_args)
        from dataclasses import replace

        lens_artifact = alignment_args._payload.get("lens_artifact")
        manifest = replace(
            manifest,
            notes=manifest.notes | {"lens_artifact": lens_artifact},
        )
        payload = {
            "schema_version": 1,
            "workflow": "concept_run",
            "output": str(destination),
            "manifest": manifest.__dict__,
            "stages": {
                "capture": capture_args._payload,
                "fit_probes": probe_args._payload,
                "align": alignment_args._payload,
            },
        }
        atomic_write_json(destination / "manifest.json", manifest)
        atomic_write_json(destination / "run.json", payload)
    except BaseException as error:
        atomic_write_json(
            destination / "failure.json",
            {
                "schema_version": 1,
                "workflow": "concept_run",
                "error_type": type(error).__name__,
                "error": str(error),
                "partial_artifacts_preserved": True,
            },
        )
        raise
    _finish_command(
        args,
        payload,
        message=f"completed concept workflow -> {destination}",
    )
    return 0


def _cmd_matrix_run(args: argparse.Namespace) -> int:
    from jlens_workspace.artifacts import atomic_write_json
    from jlens_workspace.workflows import MatrixWorkflowOptions, run_matrix_layers

    config = _load_config(args.config)
    matrix = _require_section(config, "matrix")
    lens = _require_section(config, "lens")
    layers = tuple(args.layer) if args.layer else tuple(matrix.layers or lens.layers)
    destination = args.output or Path(config.output_dir)
    if destination.exists():
        if not args.overwrite:
            raise FileExistsError(f"matrix output already exists: {destination}")
        if not destination.is_dir():
            raise FileExistsError(f"matrix output is not a directory: {destination}")
        shutil.rmtree(destination)

    bundle = _load_model_bundle(config)
    managed_lens = _load_or_fit_lens(config, bundle, lens_output=args.lens_output)
    lens_artifact = _lens_artifact_provenance(config, args.lens_output)
    effective = _build_effective_unembedding(config, bundle, matrix.convention)
    options = MatrixWorkflowOptions(
        centered=matrix.centered,
        row_normalized=matrix.row_normalized,
        normalize_by_total_weight=matrix.normalize_by_total_weight,
        zero_row_policy=matrix.zero_row_policy,
        block_size=matrix.vocabulary_chunk_size,
        compute_device=matrix.device,
        compute_dtype=matrix.operator_compute_dtype,
        accumulation_device=matrix.device,
        decomposition_device=matrix.device,
        cpu_fallback=matrix.cpu_fallback,
        energy_thresholds=tuple(matrix.energy_thresholds),
        rank_rtol=matrix.rank_relative_tolerance,
        rank_rtol_sweep=tuple(matrix.rank_relative_tolerances),
    )
    result = run_matrix_layers(
        managed_lens,
        effective,
        layers,
        destination,
        options,
    )
    try:
        manifest = _experiment_manifest(
            config,
            args.config,
            notes={
                "workflow": "matrix_run",
                "matrix_convention": matrix.convention,
                "centered": matrix.centered,
                "row_normalized": matrix.row_normalized,
                "layers": list(layers),
                "lens_artifact": lens_artifact,
            },
        )
        atomic_write_json(destination / "manifest.json", manifest)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    payload = {
        "output": str(result.output_dir),
        "metrics": str(result.metrics_path),
        "manifest": str(destination / "manifest.json"),
        "lens_artifact": lens_artifact,
        "layers": [
            {
                "layer": output.layer,
                "numerical_rank": output.numerical_rank,
                "entropy_effective_rank": output.entropy_effective_rank,
                "participation_ratio": output.participation_ratio,
            }
            for output in result.layers
        ],
    }
    _emit(
        payload,
        as_json=args.json,
        message=f"analyzed {len(result.layers)} matrix layers -> {result.output_dir}",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _normalize_config_argument(args)
        return int(args.handler(args) or 0)
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        if args.debug:
            raise
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
