"""Artifact-only report and registered figures for the K diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jlens_workspace.concept_intervention.geometry.k_diagnostic.aggregation import (
    classify_registered_effect,
    hierarchical_bootstrap_mean,
    hierarchical_group_contrast,
    holm_adjust,
    observed_crossing_summary,
)
from jlens_workspace.concept_intervention.geometry.k_diagnostic.theory import (
    sphere_max_quantile,
)
from jlens_workspace.foundation.artifacts import atomic_write_json, sha256_file


class KDiagnosticReportingError(RuntimeError):
    """Completed artifacts cannot support the registered report."""


FIGURE_NAMES = (
    "01_k_distribution_by_target_family",
    "02_k_by_layer_and_target_family",
    "03_k_vs_random_dictionary_cardinality",
    "04_real_vs_null_gain_curves",
    "05_cumulative_excess_gain",
    "06_fixed_budget_explained_energy",
    "07_theoretical_vs_empirical_random_max_cosine",
    "08_pca_rank_vs_real_null_separation",
    "09_whitening_real_null_separation",
    "10_target_vector_cosine_and_support_overlap",
)
FIGURE_DATA_SCHEMA_VERSION = 1
RAW_LOGISTIC_COMMON_LAYERS = (11, 19, 27)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise KDiagnosticReportingError(f"cannot read JSON: {path}") from error


def _require_complete(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("complete") is not True:
        raise KDiagnosticReportingError(f"report prerequisite is incomplete: {path}")
    return payload


def _bundle_real_path(shard_directory: Path, name: str) -> Path:
    reference = _read_json(shard_directory / "real_reference.json")
    bundle_id = reference.get("physical_bundle_id")
    files = reference.get("files")
    if not isinstance(bundle_id, str) or not isinstance(files, Mapping):
        raise KDiagnosticReportingError(f"invalid bundle real reference: {shard_directory}")
    expected = files.get(name)
    path = shard_directory.parent.parent / "bundles" / bundle_id / name
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or not path.is_file()
        or sha256_file(path) != expected
    ):
        raise KDiagnosticReportingError(
            f"bundle real payload hash mismatch for {name}: {shard_directory}"
        )
    return path


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - optional environment guard
        raise KDiagnosticReportingError("reporting requires pyarrow") from error
    return pq.read_table(path).to_pylist()


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        pq.write_table(pa.Table.from_pylist(list(rows)), temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _finite(values: Sequence[object]) -> np.ndarray:
    return np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )


def _median(values: Sequence[object]) -> float | None:
    array = _finite(values)
    return None if array.size == 0 else float(np.median(array))


def _family_rows(
    rows: Sequence[Mapping[str, Any]],
    family: str,
    *,
    null_family: str = "iid_full_cardinality",
    metric: str = "raw_euclidean",
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row["target_family"] == family
        and row["null_family"] == null_family
        and row["metric"] == metric
    ]


def _fixed(row: Mapping[str, Any], k: int, field: str) -> float | None:
    payload = row.get("fixed_budget", {}).get(str(k))
    if not isinstance(payload, Mapping) or payload.get(field) is None:
        return None
    return float(payload[field])


def _family_summary(
    rows: Sequence[Mapping[str, Any]], family: str, *, k_max: int
) -> dict[str, Any]:
    selected = _family_rows(rows, family)
    return {
        "target_count": len(selected),
        "K_first_observed_and_censoring": observed_crossing_summary(
            [row.get("K_first") for row in selected],
            [bool(row["K_first_right_censored"]) for row in selected],
            k_max=k_max,
        ),
        "K_consecutive3_observed_and_censoring": observed_crossing_summary(
            [row.get("K_consecutive3") for row in selected],
            [bool(row["K_consecutive3_right_censored"]) for row in selected],
            k_max=k_max,
        ),
        "median_explained_at_4": _median(
            [_fixed(row, 4, "explained_fraction") for row in selected]
        ),
        "median_explained_at_16": _median(
            [_fixed(row, 16, "explained_fraction") for row in selected]
        ),
        "median_explained_at_25": _median(
            [_fixed(row, 25, "explained_fraction") for row in selected]
        ),
    }


def _null_summary(
    rows: Sequence[Mapping[str, Any]], null_family: str, *, k_max: int
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["metric"] == "raw_euclidean" and row["null_family"] == null_family
    ]
    return {
        "row_count": len(selected),
        "K_first_observed_and_censoring": observed_crossing_summary(
            [row.get("K_first") for row in selected],
            [bool(row["K_first_right_censored"]) for row in selected],
            k_max=k_max,
        ),
        "K_consecutive3_observed_and_censoring": observed_crossing_summary(
            [row.get("K_consecutive3") for row in selected],
            [bool(row["K_consecutive3_right_censored"]) for row in selected],
            k_max=k_max,
        ),
        "median_max_cumulative_excess": _median(
            [row.get("max_cumulative_excess") for row in selected]
        ),
    }


def _target_cosines_and_support(
    root: Path,
) -> list[dict[str, Any]]:
    target_index = _require_complete(root / "targets" / "index.json")
    targets = target_index["targets"]
    vectors = {
        entry["target_id"]: np.asarray(
            np.load(root / entry["vector_path"], allow_pickle=False), dtype=np.float64
        )
        for entry in targets
    }
    by_family: dict[str, dict[tuple[int, str], list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for entry in targets:
        by_family[str(entry["target_family"])][
            (int(entry["layer"]), str(entry["source_id"]))
        ].append(entry)
    supports: dict[str, dict[int, set[int]]] = {}
    bundle_index = _read_json(root / "raw_metric_full" / "bundles.json")
    for entry in bundle_index["bundles"]:
        if entry["metric"] != "raw_euclidean":
            continue
        target_id = str(entry["target_id"])
        if target_id in supports:
            raise KDiagnosticReportingError(
                f"raw target has multiple physical bundles: {target_id}"
            )
        directory = root / "raw_metric_full" / "bundles" / entry["bundle_id"]
        complete = _require_complete(directory / "complete.json")
        expected_hash = complete.get("real_files", {}).get("fixed_budget_real.json")
        shard_ids = entry.get("shard_ids")
        if not isinstance(shard_ids, list) or not shard_ids:
            raise KDiagnosticReportingError(
                f"raw bundle lacks explicit logical shard references: {entry['bundle_id']}"
            )
        representative = (
            root
            / "raw_metric_full"
            / "shards"
            / min(str(value) for value in shard_ids)
            / "real_reference.json"
        )
        reference = _read_json(representative)
        path = directory / "fixed_budget_real.json"
        if (
            not isinstance(expected_hash, str)
            or not path.is_file()
            or sha256_file(path) != expected_hash
            or reference.get("physical_bundle_id") != entry["bundle_id"]
            or reference.get("files", {}).get("fixed_budget_real.json") != expected_hash
        ):
            raise KDiagnosticReportingError(
                f"raw bundle fixed-budget hash mismatch: {entry['bundle_id']}"
            )
        fixed = _read_json(path)
        if fixed.get("fixed_budgets") != [1, 2, 4, 8, 16, 25, 32, 64]:
            raise KDiagnosticReportingError(
                f"raw bundle fixed-budget grid mismatch: {entry['bundle_id']}"
            )
        supports[target_id] = {}
        for budget in (4, 16, 25):
            payload = fixed.get("budgets", {}).get(str(budget))
            if not isinstance(payload, Mapping) or payload.get("available") is not True:
                raise KDiagnosticReportingError(
                    f"raw bundle lacks support at K={budget}: {entry['bundle_id']}"
                )
            ids = payload.get("selected_token_ids")
            if not isinstance(ids, list):
                raise KDiagnosticReportingError(f"raw bundle support IDs are invalid at K={budget}")
            supports[target_id][budget] = {int(value) for value in ids}

    def only(family: str, key: tuple[int, str]) -> Mapping[str, Any] | None:
        values = by_family.get(family, {}).get(key, [])
        if len(values) > 1:
            raise KDiagnosticReportingError(f"multiple {family} targets for aligned cell {key}")
        return values[0] if values else None

    def append_pair(
        output: list[dict[str, Any]],
        *,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
        pair_type: str,
        contrast_arm: object = None,
        capture_position: object = None,
    ) -> None:
        left_id = str(left["target_id"])
        right_id = str(right["target_id"])
        if left_id not in supports or right_id not in supports:
            raise KDiagnosticReportingError(
                f"aligned target pair lacks a raw physical bundle: {left_id}, {right_id}"
            )
        first = vectors[left_id]
        second = vectors[right_id]
        cosine = float(first @ second / (np.linalg.norm(first) * np.linalg.norm(second)))
        identity = {
            "pair_type": pair_type,
            "left_target_id": left_id,
            "right_target_id": right_id,
            "layer": int(left["layer"]),
            "source_id": str(left["source_id"]),
            "contrast_arm": contrast_arm,
            "capture_position": capture_position,
        }
        row = {
            "pair_identity": hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            **identity,
            "target_vector_cosine": cosine,
        }
        for budget in (4, 16, 25):
            a = supports[left_id][budget]
            b = supports[right_id][budget]
            row[f"support_jaccard_at_{budget}"] = float(len(a & b) / len(a | b)) if a or b else None
        output.append(row)

    output: list[dict[str, Any]] = []
    aligned_keys = sorted(
        set(by_family.get("logistic_probe", {})) | set(by_family.get("class_mean_difference", {}))
    )
    for key in aligned_keys:
        logistic = only("logistic_probe", key)
        class_mean = only("class_mean_difference", key)
        if logistic is not None and class_mean is not None:
            append_pair(
                output,
                left=logistic,
                right=class_mean,
                pair_type="logistic_vs_class_mean",
            )
        concepts = by_family.get("seven_emotion_label_contrast", {}).get(key, [])
        for concept in sorted(concepts, key=lambda entry: str(entry["target_subtype"])):
            metadata = concept.get("metadata", {})
            arm = metadata.get("contrast_arm")
            position = metadata.get("capture_position")
            if logistic is not None:
                append_pair(
                    output,
                    left=logistic,
                    right=concept,
                    pair_type=f"logistic_vs_concept:{arm}:{position}",
                    contrast_arm=arm,
                    capture_position=position,
                )
            if class_mean is not None:
                append_pair(
                    output,
                    left=class_mean,
                    right=concept,
                    pair_type=f"class_mean_vs_concept:{arm}:{position}",
                    contrast_arm=arm,
                    capture_position=position,
                )
    identities = [row["pair_identity"] for row in output]
    if len(identities) != len(set(identities)):
        raise KDiagnosticReportingError("target-pair support identities are duplicated")
    return output


def _within_target_iid_gain_curves(
    root: Path,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Collapse all iid seeds inside target before any across-target summary."""

    real, null, _ = _within_target_iid_gain_curve_data(root)
    return real, null


def _within_target_iid_gain_curve_data(
    root: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    """Return plotted iid curves and their self-contained target/K table."""

    target_curves: dict[tuple[object, ...], dict[str, Any]] = {}
    for directory in sorted((root / "raw_metric_full" / "shards").iterdir()):
        summary = _read_json(directory / "summary.json")
        if summary["null_family"] != "iid_full_cardinality":
            continue
        key = (summary["target_id"], int(summary["layer"]), summary["metric"])
        real = np.load(_bundle_real_path(directory, "gains_real.npy"), allow_pickle=False)
        null = np.load(directory / "null_gains.npy", allow_pickle=False)[0]
        payload = target_curves.setdefault(key, {"real": real, "nulls": []})
        if not np.array_equal(payload["real"], real):
            raise KDiagnosticReportingError(f"real gain changed across iid seeds: {key}")
        payload["nulls"].append(null)
    real_curves: list[np.ndarray] = []
    null_curves: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for key, payload in sorted(target_curves.items()):
        real = np.asarray(payload["real"], dtype=np.float64)
        nulls = np.stack(payload["nulls"])
        null_median = np.median(nulls, axis=0)
        if null_median.shape != real.shape:
            raise KDiagnosticReportingError(f"real/null iid gain shape mismatch: {key}")
        real_curves.append(real)
        null_curves.append(null_median)
        target_id, layer, metric = key
        for index, (real_gain, null_gain) in enumerate(
            zip(real, null_median, strict=True), start=1
        ):
            rows.append(
                {
                    "target_id": str(target_id),
                    "layer": int(layer),
                    "metric": str(metric),
                    "k": index,
                    "real_marginal_gain": float(real_gain),
                    "within_target_null_seed_median_marginal_gain": float(null_gain),
                    "null_seed_count": int(nulls.shape[0]),
                }
            )
    return real_curves, null_curves, rows


def _cardinality_theory_reference_rows(root: Path) -> list[dict[str, Any]]:
    """Return the registered geometry and exact theory values used by Figure 7."""

    registered_geometry: dict[float, tuple[int, int]] = {}
    for directory in sorted((root / "raw_metric_full" / "shards").iterdir()):
        summary = _read_json(directory / "summary.json")
        if summary["null_family"] != "iid_cardinality_sweep":
            continue
        metadata = _read_json(directory / "null_metadata.json")
        fraction = float(summary["dictionary_fraction"])
        geometry = (
            int(metadata["metric_dimension"]),
            int(metadata["actual_cardinality"]),
        )
        if fraction in registered_geometry and registered_geometry[fraction] != geometry:
            raise KDiagnosticReportingError(
                f"inconsistent dimension/cardinality metadata for fraction {fraction}"
            )
        registered_geometry[fraction] = geometry
    return [
        {
            "dictionary_fraction": fraction,
            "metric_dimension": dimension,
            "actual_cardinality": cardinality,
            "isotropic_median_max_positive_cosine": sphere_max_quantile(
                dimension, cardinality, 0.5
            ),
        }
        for fraction, (dimension, cardinality) in sorted(registered_geometry.items())
    ]


def _artifact_descriptor(path: Path, *, relative_to: Path) -> dict[str, Any]:
    """Describe one report or source artifact without embedding host-specific roots."""

    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_figure_data_package(
    *,
    root: Path,
    report_root: Path,
    identity: str,
    target_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    gain_curve_rows: Sequence[Mapping[str, Any]],
    theory_rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Write a self-contained, hash-pinned package of every plotted data source."""

    data_root = report_root / "data"
    tables: dict[str, tuple[str, Sequence[Mapping[str, Any]]]] = {
        "target_level": ("target_level_summary.parquet", target_rows),
        "target_pair": ("target_pair_summary.parquet", pair_rows),
        "within_target_iid_gain_curves": (
            "within_target_iid_gain_curves.parquet",
            gain_curve_rows,
        ),
        "cardinality_theory_reference": (
            "cardinality_theory_reference.parquet",
            theory_rows,
        ),
    }
    for filename, rows in tables.values():
        _write_parquet(data_root / filename, rows)

    figure_sources = {
        FIGURE_NAMES[0]: {
            "tables": ["target_level"],
            "selection": "metric=raw_euclidean; null_family=iid_full_cardinality",
        },
        FIGURE_NAMES[1]: {
            "tables": ["target_level"],
            "selection": "metric=raw_euclidean; null_family=iid_full_cardinality",
        },
        FIGURE_NAMES[2]: {
            "tables": ["target_level"],
            "selection": "null_family=iid_cardinality_sweep",
        },
        FIGURE_NAMES[3]: {
            "tables": ["within_target_iid_gain_curves"],
            "selection": "all rows; null seeds collapsed within target before plotting",
        },
        FIGURE_NAMES[4]: {
            "tables": ["target_level"],
            "selection": (
                "metric=raw_euclidean; null_family=iid_full_cardinality; cumulative_excess"
            ),
        },
        FIGURE_NAMES[5]: {
            "tables": ["target_level"],
            "selection": (
                "metric=raw_euclidean; null_family=iid_full_cardinality; fixed K=4,16,25"
            ),
        },
        FIGURE_NAMES[6]: {
            "tables": ["target_level", "cardinality_theory_reference"],
            "selection": ("iid_cardinality_sweep fixed K=1 plus registered isotropic reference"),
        },
        FIGURE_NAMES[7]: {
            "tables": ["target_level"],
            "selection": ("iid_full_cardinality; j_pca_* Delta4 paired to raw_euclidean"),
        },
        FIGURE_NAMES[8]: {
            "tables": ["target_level"],
            "selection": (
                "iid_full_cardinality; activation_whitened_* Delta4 paired to raw_euclidean"
            ),
        },
        FIGURE_NAMES[9]: {
            "tables": ["target_pair"],
            "selection": "all unpooled pair cells at K=4,16,25",
        },
    }
    source_paths = [
        root / "targets" / "index.json",
        root / "pilot" / "index.json",
        root / "pilot" / "summary.parquet",
        root / "raw_metric_full" / "index.json",
        root / "raw_metric_full" / "bundles.json",
        root / "raw_metric_full" / "summary.parquet",
        root / "transformed_metric" / "index.json",
        root / "transformed_metric" / "summary.parquet",
    ]
    data_files = {
        name: {
            **_artifact_descriptor(data_root / filename, relative_to=report_root),
            "format": "parquet",
            "row_count": len(rows),
        }
        for name, (filename, rows) in tables.items()
    }
    figure_files = {
        name: {
            extension: _artifact_descriptor(
                report_root / "figures" / f"{name}.{extension}",
                relative_to=report_root,
            )
            for extension in ("pdf", "png")
        }
        for name in FIGURE_NAMES
    }
    manifest = {
        "schema_version": FIGURE_DATA_SCHEMA_VERSION,
        "identity": identity,
        "scope": (
            "self-contained plot-source tables; immutable bundle/shard payloads remain "
            "the single scientific source of truth and are referenced by indexed hashes"
        ),
        "artifact_root": os.path.relpath(root, report_root),
        "data_files": data_files,
        "figure_sources": figure_sources,
        "figure_files": figure_files,
        "source_artifacts": [_artifact_descriptor(path, relative_to=root) for path in source_paths],
    }
    manifest_path = data_root / "figure_data_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def _figure_context() -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, plt


def _save_figure(fig: Any, figure_root: Path, name: str) -> None:
    figure_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_root / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(figure_root / f"{name}.png", dpi=180, bbox_inches="tight")


def _scatter_groups(
    ax: Any,
    groups: Mapping[str, Sequence[object]],
    *,
    ylabel: str,
) -> None:
    for index, (_label, raw) in enumerate(sorted(groups.items())):
        values = _finite(raw)
        if values.size:
            offsets = np.linspace(-0.18, 0.18, values.size)
            ax.scatter(index + offsets, values, alpha=0.35, s=10)
            ax.plot(index, np.median(values), marker="D", color="black")
    ax.set_xticks(range(len(groups)), sorted(groups), rotation=25, ha="right")
    ax.set_ylabel(ylabel)


def _plot_registered_figures(
    *,
    report_root: Path,
    raw_rows: list[dict[str, Any]],
    transformed_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    real_gains: Sequence[np.ndarray],
    null_gains: Sequence[np.ndarray],
    theory_rows: Sequence[Mapping[str, Any]],
) -> None:
    _, plt = _figure_context()
    figures = report_root / "figures"

    primary = [
        row
        for row in raw_rows
        if row["metric"] == "raw_euclidean" and row["null_family"] == "iid_full_cardinality"
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    groups: dict[str, list[object]] = defaultdict(list)
    censored_groups: dict[str, int] = defaultdict(int)
    for row in primary:
        label = str(row["target_family"])
        if label == "seven_emotion_label_contrast":
            label = f"contrast:{row.get('contrast_arm')}:{row.get('capture_position')}"
        groups[label].append(row.get("K_first"))
        censored_groups[label] += int(bool(row["K_first_right_censored"]))
    _scatter_groups(ax, groups, ylabel="K_first (right-censored omitted)")
    for index, label in enumerate(sorted(groups)):
        count = censored_groups[label]
        if count:
            ax.scatter(
                np.full(count, index),
                np.full(count, 65),
                marker="^",
                facecolors="none",
                edgecolors="tab:red",
                label="right-censored at K_max" if index == 0 else None,
            )
    ax.set_title("Target-level occupancy by target family")
    _save_figure(fig, figures, FIGURE_NAMES[0])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    def plot_group(row: Mapping[str, Any]) -> str:
        if row["target_family"] == "seven_emotion_label_contrast":
            return f"contrast:{row.get('contrast_arm')}:{row.get('capture_position')}"
        return str(row["target_family"])

    for family in sorted({plot_group(row) for row in primary}):
        layer_values = []
        medians = []
        for layer in sorted({int(row["layer"]) for row in primary}):
            values = _finite(
                [
                    row.get("K_first")
                    for row in primary
                    if plot_group(row) == family and int(row["layer"]) == layer
                ]
            )
            if values.size:
                ax.scatter(np.full(values.size, layer), values, alpha=0.18, s=8)
                layer_values.append(layer)
                medians.append(float(np.median(values)))
            censored_count = sum(
                bool(row["K_first_right_censored"])
                for row in primary
                if plot_group(row) == family and int(row["layer"]) == layer
            )
            if censored_count:
                ax.scatter(
                    np.full(censored_count, layer),
                    np.full(censored_count, 65),
                    marker="^",
                    facecolors="none",
                    edgecolors="tab:red",
                    alpha=0.5,
                )
        ax.plot(layer_values, medians, marker="o", label=family)
    ax.set_xlabel("layer")
    ax.set_ylabel("K_first")
    ax.legend(fontsize=7)
    ax.set_title("Layer and target-family occupancy (points plus medians)")
    _save_figure(fig, figures, FIGURE_NAMES[1])
    plt.close(fig)

    cardinality = [row for row in raw_rows if row["null_family"] == "iid_cardinality_sweep"]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    fractions = sorted({float(row["dictionary_fraction"]) for row in cardinality})
    for fraction in fractions:
        values = _finite(
            [row.get("K_first") for row in cardinality if row["dictionary_fraction"] == fraction]
        )
        if values.size:
            ax.scatter(np.full(values.size, fraction), values, alpha=0.18, s=8)
            ax.plot(fraction, np.median(values), marker="D", color="black")
    ax.set_xscale("log")
    ax.set_xlabel("random dictionary fraction")
    ax.set_ylabel("K_first")
    ax.set_title("Random-search cardinality sensitivity")
    _save_figure(fig, figures, FIGURE_NAMES[2])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    if real_gains:
        real_array = np.stack(real_gains)
        null_array = np.stack(null_gains)
        k = np.arange(1, real_array.shape[1] + 1)
        ax.plot(k, np.median(real_array, axis=0), label="real J median")
        ax.plot(
            k,
            np.median(null_array, axis=0),
            label="within-target seed median, then across-target median",
        )
        ax.fill_between(
            k,
            np.quantile(real_array, 0.25, axis=0),
            np.quantile(real_array, 0.75, axis=0),
            alpha=0.15,
        )
    ax.set_xlabel("K")
    ax.set_ylabel("marginal gain")
    ax.legend()
    ax.set_title("Unsmoothed real and null gain curves")
    _save_figure(fig, figures, FIGURE_NAMES[3])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    cumulative = [np.asarray(row["cumulative_excess"], dtype=np.float64) for row in primary]
    for curve in cumulative:
        ax.plot(np.arange(1, curve.size + 1), curve, alpha=0.03, color="tab:blue")
    if cumulative:
        matrix = np.stack(cumulative)
        ax.plot(np.arange(1, matrix.shape[1] + 1), np.median(matrix, axis=0), color="black")
    ax.set_xlabel("K")
    ax.set_ylabel("cumulative real-minus-null gain")
    ax.set_title("Cumulative excess gain (target points, no smoothing)")
    _save_figure(fig, figures, FIGURE_NAMES[4])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    budget_groups: dict[str, list[float]] = defaultdict(list)
    for row in primary:
        for budget in (4, 16, 25):
            value = _fixed(row, budget, "explained_fraction")
            if value is not None:
                budget_groups[f"{plot_group(row)}@{budget}"].append(value)
    _scatter_groups(ax, budget_groups, ylabel="explained target energy")
    ax.set_title("Fixed reconstruction budgets; not occupancy estimates")
    _save_figure(fig, figures, FIGURE_NAMES[5])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    theory_x: list[float] = []
    theory_y: list[float] = []
    empirical_x: list[float] = []
    empirical_y: list[float] = []
    for row in theory_rows:
        theory_x.append(float(row["dictionary_fraction"]))
        theory_y.append(float(row["isotropic_median_max_positive_cosine"]))
    for row in cardinality:
        fraction = float(row["dictionary_fraction"])
        fixed_one = _fixed(row, 1, "null_median_explained_fraction")
        if fixed_one is not None:
            empirical_x.append(fraction)
            empirical_y.append(math.sqrt(max(0.0, fixed_one)))
    ax.scatter(empirical_x, empirical_y, alpha=0.15, label="empirical iid max cosine")
    if theory_x:
        ordered = np.argsort(theory_x)
        ax.plot(
            np.asarray(theory_x)[ordered],
            np.asarray(theory_y)[ordered],
            label="isotropic reference",
        )
    ax.set_xscale("log")
    ax.set_xlabel("random dictionary fraction")
    ax.set_ylabel("first-step maximum cosine")
    ax.legend()
    ax.set_title("Theoretical reference versus empirical random maximum")
    _save_figure(fig, figures, FIGURE_NAMES[6])
    plt.close(fig)

    raw_delta4 = {
        (str(row["target_id"]), int(row["layer"])): _fixed(row, 4, "real_minus_null")
        for row in raw_rows
        if row["metric"] == "raw_euclidean" and row["null_family"] == "iid_full_cardinality"
    }

    def transformed_groups(prefix: str) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        values: dict[str, list[float]] = defaultdict(list)
        effects: dict[str, list[float]] = defaultdict(list)
        for row in transformed_rows:
            if row["null_family"] != "iid_full_cardinality" or not str(row["metric"]).startswith(
                prefix
            ):
                continue
            transformed_value = _fixed(row, 4, "real_minus_null")
            raw_value = raw_delta4.get((str(row["target_id"]), int(row["layer"])))
            if transformed_value is None or raw_value is None:
                raise KDiagnosticReportingError(
                    f"transformed/raw Delta4 pair is incomplete: {row['target_id']}"
                )
            values[str(row["metric"])].append(transformed_value)
            effects[str(row["metric"])].append(transformed_value - raw_value)
        return values, effects

    pca_values, pca_effects = transformed_groups("j_pca_")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    _scatter_groups(axes[0], pca_values, ylabel="Delta4 transformed real-null separation")
    _scatter_groups(axes[1], pca_effects, ylabel="paired Delta4 effect (transformed - raw)")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("PCA fixed-K=4 primary estimand")
    axes[1].set_title("Paired raw-vs-PCA fixed-K=4 effect")
    _save_figure(fig, figures, FIGURE_NAMES[7])
    plt.close(fig)

    whitening_values, whitening_effects = transformed_groups("activation_whitened_")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    _scatter_groups(
        axes[0],
        whitening_values,
        ylabel="Delta4 transformed real-null separation",
    )
    _scatter_groups(
        axes[1],
        whitening_effects,
        ylabel="paired Delta4 effect (transformed - raw)",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Whitening fixed-K=4 primary estimand")
    axes[1].set_title("Paired raw-vs-whitening fixed-K=4 effect")
    _save_figure(fig, figures, FIGURE_NAMES[8])
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharex=True, sharey=True)
    pair_types = sorted({str(row["pair_type"]) for row in pair_rows})
    for axis, budget in zip(axes, (4, 16, 25), strict=True):
        for pair_type in pair_types:
            selected = [row for row in pair_rows if row["pair_type"] == pair_type]
            axis.scatter(
                [row["target_vector_cosine"] for row in selected],
                [row[f"support_jaccard_at_{budget}"] for row in selected],
                alpha=0.4,
                s=12,
                label=pair_type,
            )
        axis.set_title(f"Unpooled pair cells at K={budget}")
        axis.set_xlabel("target-vector cosine")
    axes[0].set_ylabel("real-J support Jaccard")
    axes[-1].legend(fontsize=5, bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_figure(fig, figures, FIGURE_NAMES[9])
    plt.close(fig)


def _fmt(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _summary_identity(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        row["target_id"],
        int(row["layer"]),
        row["target_family"],
        row["target_subtype"],
        row["metric"],
        row["null_family"],
        float(row["dictionary_fraction"]),
    )


def _validate_stage_summary(
    root: Path,
    stage: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_null_replicates: int,
) -> None:
    """Require exact grid cells, seed counts, and sealed target membership."""

    stage_root = root / stage
    grid = _read_json(stage_root / "grid.json")
    membership = _read_json(root / "target_membership" / f"{stage}.json")
    expected_targets = {str(entry["target_id"]) for entry in membership["targets"]}
    required_fields = {
        "target_id",
        "layer",
        "target_family",
        "target_subtype",
        "metric",
        "null_family",
        "dictionary_fraction",
        "null_seeds",
        "null_seed_count",
        "K_first",
        "K_first_right_censored",
        "K_consecutive3",
        "K_consecutive3_right_censored",
        "max_cumulative_excess",
        "cumulative_excess",
        "fixed_budget",
        "empirical_diagnostics",
    }
    for row in rows:
        missing = required_fields - set(row)
        if missing:
            raise KDiagnosticReportingError(
                f"{stage} summary row lacks registered fields: {sorted(missing)}"
            )
        seeds = row["null_seeds"]
        curve = np.asarray(row["cumulative_excess"], dtype=np.float64)
        if (
            not isinstance(seeds, list)
            or int(row["null_seed_count"]) != len(seeds)
            or curve.ndim != 1
            or curve.size == 0
            or not np.isfinite(curve).all()
            or not np.isfinite(float(row["max_cumulative_excess"]))
            or not isinstance(row["fixed_budget"], Mapping)
            or not isinstance(row["empirical_diagnostics"], Mapping)
        ):
            raise KDiagnosticReportingError(f"{stage} summary row has invalid schema")
        empirical = row["empirical_diagnostics"]
        marginal = empirical.get("marginal_gain_by_k")
        cumulative_benefit = empirical.get("cumulative_benefit")
        fixed_benefit = empirical.get("fixed_reconstruction_benefit")
        resolution = 1.0 / (len(seeds) + 1)
        if (
            empirical.get("schema_version") != 1
            or int(empirical.get("null_seed_count", -1)) != len(seeds)
            or empirical.get("plus_one_correction") is not True
            or not np.isclose(
                float(empirical.get("resolution", math.nan)),
                resolution,
                rtol=0.0,
                atol=1e-15,
            )
            or not isinstance(marginal, list)
            or len(marginal) != curve.size
            or [cell.get("K") for cell in marginal] != list(range(1, curve.size + 1))
            or not isinstance(cumulative_benefit, Mapping)
            or not isinstance(fixed_benefit, Mapping)
            or "4" not in fixed_benefit
            or not isinstance(empirical.get("fixed_k4"), Mapping)
        ):
            raise KDiagnosticReportingError(f"{stage} empirical diagnostic schema is invalid")
        for diagnostic_cell in [
            *marginal,
            cumulative_benefit,
            empirical["fixed_k4"],
            *fixed_benefit.values(),
        ]:
            if (
                not isinstance(diagnostic_cell, Mapping)
                or not resolution <= float(diagnostic_cell["empirical_upper_tail_p_value"]) <= 1.0
                or not resolution <= float(diagnostic_cell["empirical_percentile"]) <= 1.0
            ):
                raise KDiagnosticReportingError(f"{stage} empirical p-value/percentile is invalid")
        for crossing, flag in (
            (row["K_first"], row["K_first_right_censored"]),
            (row["K_consecutive3"], row["K_consecutive3_right_censored"]),
        ):
            if (crossing is None) != bool(flag):
                raise KDiagnosticReportingError(
                    f"{stage} summary crossing/censoring fields disagree"
                )
    grouped: dict[tuple[object, ...], set[int]] = defaultdict(set)
    for shard in grid.get("shards", []):
        key = (
            shard["target_id"],
            int(shard["layer"]),
            shard["target_family"],
            shard["target_subtype"],
            shard["metric"],
            shard["null_family"],
            float(shard["dictionary_fraction"]),
        )
        grouped[key].add(int(shard["null_seed"]))
    observed = {_summary_identity(row): row for row in rows}
    if len(observed) != len(rows) or set(observed) != set(grouped):
        raise KDiagnosticReportingError(
            f"{stage} summary cells do not exactly match the sealed scientific grid"
        )
    if {str(key[0]) for key in grouped} != expected_targets:
        raise KDiagnosticReportingError(
            f"{stage} summary target IDs do not match sealed membership"
        )
    for key, seeds in grouped.items():
        row = observed[key]
        if sorted(seeds) != sorted(int(value) for value in row["null_seeds"]):
            raise KDiagnosticReportingError(f"{stage} null seed identity mismatch: {key}")
        if (
            row["null_family"]
            in {
                "iid_full_cardinality",
                "iid_cardinality_sweep",
                "random_prefix_same_k",
                "orthogonal_rotation",
                "label_permutation_probe",
            }
            and len(seeds) < minimum_null_replicates
        ):
            raise KDiagnosticReportingError(
                f"{stage} null cell has only {len(seeds)} replicates: {key}"
            )


def _attach_target_metadata(root: Path, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = _require_complete(root / "targets" / "index.json")
    targets = {str(entry["target_id"]): entry for entry in index["targets"]}
    output: list[dict[str, Any]] = []
    for row in rows:
        target = targets.get(str(row["target_id"]))
        if target is None:
            raise KDiagnosticReportingError(f"summary target is not indexed: {row['target_id']}")
        metadata = target.get("metadata", {})
        output.append(
            dict(row)
            | {
                "source_id": str(target["source_id"]),
                "contrast_arm": metadata.get("contrast_arm"),
                "capture_position": metadata.get("capture_position"),
                "resampling_unit": metadata.get("prompt_manifest_sha256", target["target_id"]),
            }
        )
    return output


def _registered_decisions(
    raw_rows: Sequence[Mapping[str, Any]],
    transformed_rows: Sequence[Mapping[str, Any]],
    diagnostic: Any,
) -> list[dict[str, Any]]:
    """Evaluate explicit root-cause contrasts at fixed reconstruction budget K=4."""

    def k4_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for row in rows:
            value = _fixed(row, diagnostic.primary_fixed_k, "real_minus_null")
            if value is None:
                raise KDiagnosticReportingError(
                    f"summary lacks fixed-K primary value: {row['target_id']}"
                )
            output.append(dict(row) | {"k4_separation": value})
        return output

    raw_metric = [row for row in raw_rows if row["metric"] == "raw_euclidean"]
    iid = k4_rows([row for row in raw_metric if row["null_family"] == "iid_full_cardinality"])

    def family(rows: Sequence[Mapping[str, Any]], name: str) -> list[dict[str, Any]]:
        return [dict(row) for row in rows if row["target_family"] == name]

    def contrast_cell(
        rows: Sequence[Mapping[str, Any]], arm: str, position: str
    ) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in rows
            if row["target_family"] == "seven_emotion_label_contrast"
            and row["contrast_arm"] == arm
            and row["capture_position"] == position
        ]

    specs: list[dict[str, Any]] = []

    def add_paired(
        *,
        hypothesis: str,
        numerator: Sequence[Mapping[str, Any]],
        denominator: Sequence[Mapping[str, Any]],
        pair_fields: tuple[str, ...],
        numerator_label: str,
        denominator_label: str,
        direction: str,
    ) -> None:
        def keyed(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[object, ...], Mapping[str, Any]]:
            output: dict[tuple[object, ...], Mapping[str, Any]] = {}
            for row in rows:
                key = tuple(row[field] for field in pair_fields)
                if key in output:
                    raise KDiagnosticReportingError(f"duplicate pair key for {hypothesis}: {key}")
                output[key] = row
            return output

        numerators = keyed(numerator)
        denominators = keyed(denominator)
        if not numerators or set(numerators) != set(denominators):
            raise KDiagnosticReportingError(
                f"paired mechanism cells do not align exactly: {hypothesis}"
            )
        paired = []
        for key in sorted(numerators, key=repr):
            top = numerators[key]
            bottom = denominators[key]
            paired.append(
                dict(top)
                | {
                    "target_id": "|".join(str(value) for value in key),
                    "source_id": str(top["source_id"]),
                    "resampling_unit": "|".join(str(value) for value in key),
                    "mechanism_effect": float(top["k4_separation"])
                    - float(bottom["k4_separation"]),
                }
            )
        specs.append(
            {
                "mode": "paired",
                "hypothesis": hypothesis,
                "rows": paired,
                "numerator": numerator_label,
                "denominator": denominator_label,
                "pair_key": list(pair_fields),
                "n_pairs": len(paired),
                "comparison_direction": direction,
            }
        )

    def add_group(
        *,
        hypothesis: str,
        numerator: Sequence[Mapping[str, Any]],
        denominator: Sequence[Mapping[str, Any]],
        numerator_label: str,
        denominator_label: str,
        direction: str,
    ) -> None:
        if not numerator or not denominator:
            raise KDiagnosticReportingError(f"empty group contrast: {hypothesis}")
        specs.append(
            {
                "mode": "group",
                "hypothesis": hypothesis,
                "numerator_rows": list(numerator),
                "denominator_rows": list(denominator),
                "numerator": numerator_label,
                "denominator": denominator_label,
                "pair_key": None,
                "n_pairs": None,
                "comparison_direction": direction,
            }
        )

    common_layers = set(RAW_LOGISTIC_COMMON_LAYERS)
    add_group(
        hypothesis="target_construction:logistic_minus_raw_activation",
        numerator=[row for row in family(iid, "logistic_probe") if row["layer"] in common_layers],
        denominator=[row for row in family(iid, "raw_activation") if row["layer"] in common_layers],
        numerator_label="logistic_probe K4 separation",
        denominator_label="raw_activation K4 separation",
        direction="positive means the learned logistic construction is more four-atom aligned",
    )
    add_paired(
        hypothesis="target_construction:logistic_minus_class_mean",
        numerator=family(iid, "logistic_probe"),
        denominator=family(iid, "class_mean_difference"),
        pair_fields=("layer", "source_id"),
        numerator_label="logistic_probe K4 separation",
        denominator_label="class_mean_difference K4 separation",
        direction="positive isolates classifier-direction construction beyond class means",
    )
    add_paired(
        hypothesis="lexical_label:explicit_minus_label_free:assistant_boundary",
        numerator=contrast_cell(iid, "label_explicit", "assistant_boundary"),
        denominator=contrast_cell(iid, "label_free_definition", "assistant_boundary"),
        pair_fields=("layer", "source_id", "capture_position"),
        numerator_label="label_explicit assistant-boundary K4 separation",
        denominator_label="label_free assistant-boundary K4 separation",
        direction="positive supports explicit-label lexical leakage",
    )
    add_paired(
        hypothesis="capture_position:explicit_concept_end_minus_assistant_boundary",
        numerator=contrast_cell(iid, "label_explicit", "concept_token_end"),
        denominator=contrast_cell(iid, "label_explicit", "assistant_boundary"),
        pair_fields=("layer", "source_id"),
        numerator_label="label_explicit concept-token-end K4 separation",
        denominator_label="label_explicit assistant-boundary K4 separation",
        direction="positive isolates capture position within the same explicit prompts",
    )
    add_paired(
        hypothesis="capture_position:label_free_definition_end_minus_assistant_boundary",
        numerator=contrast_cell(iid, "label_free_definition", "definition_span_end"),
        denominator=contrast_cell(iid, "label_free_definition", "assistant_boundary"),
        pair_fields=("layer", "source_id"),
        numerator_label="label-free definition-span-end K4 separation",
        denominator_label="label-free assistant-boundary K4 separation",
        direction="positive isolates capture position within the same label-free prompts",
    )
    for position in ("assistant_boundary", "definition_span_end"):
        add_paired(
            hypothesis=f"baseline_control:four_unrelated_minus_one_vs_six:{position}",
            numerator=contrast_cell(iid, "four_unrelated_topics_control", position),
            denominator=contrast_cell(iid, "label_free_definition", position),
            pair_fields=("layer", "source_id", "capture_position"),
            numerator_label=f"four-unrelated-topics {position} K4 separation",
            denominator_label=f"label-free one-vs-six {position} K4 separation",
            direction="positive means the narrow emotion-only baseline was not the source of K<=4",
        )

    prefix = k4_rows([row for row in raw_metric if row["null_family"] == "random_prefix_same_k"])
    add_paired(
        hypothesis="null_search:random_prefix_minus_iid_full",
        numerator=prefix,
        denominator=iid,
        pair_fields=("target_id", "layer", "metric"),
        numerator_label="real-minus-random-prefix K4 separation",
        denominator_label="real-minus-full-iid-search K4 separation",
        direction="positive quantifies the effect of best-atom random dictionary search",
    )

    cardinality = k4_rows(
        [row for row in raw_metric if row["null_family"] == "iid_cardinality_sweep"]
    )
    by_target: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in cardinality:
        by_target[(row["target_id"], int(row["layer"]), row["metric"])].append(row)
    trend_rows = []
    monotonic_target_count = 0
    expected_fractions = sorted(float(value) for value in diagnostic.cardinality_fractions)
    for key, rows in sorted(by_target.items(), key=lambda item: repr(item[0])):
        ordered = sorted(rows, key=lambda row: float(row["dictionary_fraction"]))
        fractions = [float(row["dictionary_fraction"]) for row in ordered]
        if fractions != expected_fractions:
            raise KDiagnosticReportingError(f"cardinality trend grid mismatch: {key}")
        slope = float(
            np.polyfit(
                np.log2(np.asarray(fractions)),
                np.asarray([row["k4_separation"] for row in ordered]),
                deg=1,
            )[0]
        )
        separations = np.asarray([row["k4_separation"] for row in ordered])
        monotonic_target_count += int(np.all(np.diff(separations) <= 0.0))
        trend_rows.append(
            dict(ordered[-1])
            | {
                "target_id": "|".join(str(value) for value in key),
                "resampling_unit": "|".join(str(value) for value in key),
                "mechanism_effect": -slope,
            }
        )
    specs.append(
        {
            "mode": "paired",
            "hypothesis": "null_search:negative_cardinality_slope",
            "rows": trend_rows,
            "numerator": "decrease in K4 separation per log2 dictionary-fraction increase",
            "denominator": "zero cardinality trend",
            "pair_key": ["target_id", "layer", "metric"],
            "n_pairs": len(trend_rows),
            "comparison_direction": "positive means larger iid search erodes apparent K4 separation",
            "effect_units": (
                "fraction of target squared norm per log2 increase in dictionary fraction"
            ),
            "minimum_effect_justification": (
                "0.01 is one target-energy percentage point per doubling of iid search"
            ),
            "monotonic_target_count": monotonic_target_count,
            "monotonic_target_fraction": (
                monotonic_target_count / len(trend_rows) if trend_rows else None
            ),
        }
    )

    for hypothesis, null_family, target_family, numerator, denominator, direction in (
        (
            "dictionary_alignment:real_minus_rotation",
            "orthogonal_rotation",
            None,
            "real J K4 explained fraction",
            "Haar-rotated-target K4 explained fraction",
            "positive supports target-dictionary alignment",
        ),
        (
            "label_structure:logistic_real_minus_permuted",
            "label_permutation_probe",
            "logistic_probe",
            "real logistic K4 explained fraction",
            "group-safe permuted logistic K4 explained fraction",
            "positive supports label structure for the logistic direction",
        ),
        (
            "label_structure:class_mean_real_minus_permuted",
            "label_permutation_probe",
            "class_mean_difference",
            "real class-mean K4 explained fraction",
            "group-safe permuted class-mean K4 explained fraction",
            "positive supports label structure for the class-mean direction",
        ),
    ):
        selected = k4_rows(
            [
                row
                for row in raw_metric
                if row["null_family"] == null_family
                and (target_family is None or row["target_family"] == target_family)
            ]
        )
        specs.append(
            {
                "mode": "paired",
                "hypothesis": hypothesis,
                "rows": [
                    dict(row) | {"mechanism_effect": row["k4_separation"]} for row in selected
                ],
                "numerator": numerator,
                "denominator": denominator,
                "pair_key": ["target_id", "layer", "metric"],
                "n_pairs": len(selected),
                "comparison_direction": direction,
            }
        )

    transformed_layers = {int(row["layer"]) for row in transformed_rows}
    raw_nonactivation = [
        row
        for row in iid
        if row["target_family"] != "raw_activation" and int(row["layer"]) in transformed_layers
    ]
    for hypothesis, metric, numerator in (
        (
            "metric:pca_r256_minus_raw",
            diagnostic.primary_pca_metric,
            f"{diagnostic.primary_pca_metric} K4 separation",
        ),
        (
            "metric:whitening_0.10_minus_raw",
            f"activation_whitened_{diagnostic.primary_whitening_floor:.2f}",
            f"activation_whitened_{diagnostic.primary_whitening_floor:.2f} K4 separation",
        ),
    ):
        transformed = k4_rows(
            [
                row
                for row in transformed_rows
                if row["metric"] == metric and row["null_family"] == "iid_full_cardinality"
            ]
        )
        add_paired(
            hypothesis=hypothesis,
            numerator=transformed,
            denominator=raw_nonactivation,
            pair_fields=("target_id", "layer"),
            numerator_label=numerator,
            denominator_label="raw_euclidean K4 separation",
            direction="positive means the transformed metric increases four-atom separation",
        )

    estimates: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for offset, spec in enumerate(specs):
        seed = diagnostic.hierarchical_bootstrap_seed + offset
        if spec["mode"] == "group":
            estimate = hierarchical_group_contrast(
                spec["numerator_rows"],
                spec["denominator_rows"],
                value_key="k4_separation",
                samples=diagnostic.hierarchical_bootstrap_samples,
                seed=seed,
                confidence_level=diagnostic.confidence_level,
            )
        else:
            if not spec["rows"]:
                raise KDiagnosticReportingError(
                    f"registered hypothesis has no rows: {spec['hypothesis']}"
                )
            estimate = hierarchical_bootstrap_mean(
                spec["rows"],
                value_key="mechanism_effect",
                samples=diagnostic.hierarchical_bootstrap_samples,
                seed=seed,
                confidence_level=diagnostic.confidence_level,
            )
        effects = np.asarray(estimate.pop("bootstrap_effects"), dtype=np.float64)
        p_value = float((1 + np.sum(effects <= diagnostic.minimum_effect)) / (effects.size + 1))
        raw_p_values.append(p_value)
        estimates.append(
            {
                "hypothesis": spec["hypothesis"],
                "primary_estimand": "K4_real_minus_median_null_explained_fraction",
                "effect_units": spec.get(
                    "effect_units", "fraction of target squared norm explained at K=4"
                ),
                "numerator": spec["numerator"],
                "denominator": spec["denominator"],
                "pair_key": spec["pair_key"],
                "n_pairs": spec["n_pairs"],
                "comparison_direction": spec["comparison_direction"],
                "minimum_effect": diagnostic.minimum_effect,
                "minimum_effect_justification": spec.get(
                    "minimum_effect_justification",
                    (
                        "0.01 is the preregistered smallest practically interpretable "
                        "absolute shift: one percentage point of target energy at K=4"
                    ),
                ),
                "equivalence_margin": diagnostic.equivalence_margin,
                "equivalence_justification": "0.0025 is one quarter percentage point of target energy",
                "raw_one_sided_bootstrap_p": p_value,
                **estimate,
                **{
                    key: spec[key]
                    for key in ("monotonic_target_count", "monotonic_target_fraction")
                    if key in spec
                },
            }
        )
    adjusted = holm_adjust(raw_p_values)
    for result, adjusted_p in zip(estimates, adjusted, strict=True):
        result["holm_adjusted_p"] = adjusted_p
        result["multiple_comparison_policy"] = diagnostic.multiple_comparison_policy
        result["conclusion"] = classify_registered_effect(
            ci_low=float(result["ci_low"]),
            ci_high=float(result["ci_high"]),
            minimum_effect=diagnostic.minimum_effect,
            equivalence_margin=diagnostic.equivalence_margin,
            adjusted_p_value=adjusted_p,
        )
        result["cluster_counts"] = {
            key: result[key]
            for key in (
                "layer_n",
                "source_n",
                "target_level_n",
                "numerator_target_n",
                "denominator_target_n",
                "numerator_source_n",
                "denominator_source_n",
            )
            if key in result
        }
    return estimates


def generate_report(
    *,
    artifact_root: str | Path,
    diagnostic: Any,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a derived report only from complete saved experiments.

    ``artifact_root`` is always read-only scientific evidence. ``output_dir``
    must name a new, versioned derivation outside that immutable root. Existing
    or partial output directories are rejected so redelivery cannot overwrite
    a sealed package.
    """

    if output_dir is None:
        raise KDiagnosticReportingError(
            "an explicit versioned derived report output is required"
        )
    root = Path(artifact_root)
    final_report_root = Path(output_dir)
    if final_report_root.exists() or final_report_root.is_symlink():
        raise KDiagnosticReportingError(
            "derived report output already exists; preserve it for audit and use a "
            "registered same-revision attempt/recovery operation"
        )
    try:
        final_report_root.resolve(strict=False).relative_to(root.resolve(strict=True))
    except ValueError:
        pass
    except OSError as error:
        raise KDiagnosticReportingError("source artifact root is unavailable") from error
    else:
        raise KDiagnosticReportingError(
            "derived report output must be outside the immutable source artifact root"
        )
    final_report_root.parent.mkdir(parents=True, exist_ok=True)
    report_root = Path(
        tempfile.mkdtemp(
            prefix=f".{final_report_root.name}.attempt-staging-",
            dir=final_report_root.parent,
        )
    )
    pilot_index = _require_complete(root / "pilot" / "index.json")
    raw_index = _require_complete(root / "raw_metric_full" / "index.json")
    transformed_index = _require_complete(root / "transformed_metric" / "index.json")
    for stage, index in (
        ("pilot", pilot_index),
        ("raw_metric_full", raw_index),
        ("transformed_metric", transformed_index),
    ):
        if index.get("identity") != diagnostic.identity or index.get("stage") != stage:
            raise KDiagnosticReportingError(
                f"{stage} index identity does not match the registered report"
            )
    pilot_rows = _read_parquet(root / "pilot" / "summary.parquet")
    raw_rows = _read_parquet(root / "raw_metric_full" / "summary.parquet")
    transformed_rows = _read_parquet(root / "transformed_metric" / "summary.parquet")
    _validate_stage_summary(
        root,
        "pilot",
        pilot_rows,
        minimum_null_replicates=diagnostic.minimum_null_replicates,
    )
    _validate_stage_summary(
        root,
        "raw_metric_full",
        raw_rows,
        minimum_null_replicates=diagnostic.minimum_null_replicates,
    )
    _validate_stage_summary(
        root,
        "transformed_metric",
        transformed_rows,
        minimum_null_replicates=diagnostic.minimum_null_replicates,
    )
    raw_rows = _attach_target_metadata(root, raw_rows)
    transformed_rows = _attach_target_metadata(root, transformed_rows)
    pairs = _target_cosines_and_support(root)
    real_gains, null_gains, gain_curve_rows = _within_target_iid_gain_curve_data(root)
    theory_rows = _cardinality_theory_reference_rows(root)
    families = (
        "raw_activation",
        "class_mean_difference",
        "logistic_probe",
    )
    family_summary = {family: _family_summary(raw_rows, family, k_max=64) for family in families}
    common_layer_rows = [row for row in raw_rows if int(row["layer"]) in RAW_LOGISTIC_COMMON_LAYERS]
    common_layer_family_summary = {
        family: _family_summary(common_layer_rows, family, k_max=64)
        for family in ("raw_activation", "logistic_probe")
    }
    contrast_rows = [
        row for row in raw_rows if row["target_family"] == "seven_emotion_label_contrast"
    ]
    for arm, position in sorted(
        {(str(row["contrast_arm"]), str(row["capture_position"])) for row in contrast_rows}
    ):
        selected = [
            row
            for row in contrast_rows
            if row["contrast_arm"] == arm and row["capture_position"] == position
        ]
        family_summary[f"seven_emotion_label_contrast:{arm}:{position}"] = _family_summary(
            selected, "seven_emotion_label_contrast", k_max=64
        )
    null_summary = {
        family: _null_summary(raw_rows, family, k_max=64)
        for family in (
            "iid_full_cardinality",
            "random_prefix_same_k",
            "orthogonal_rotation",
        )
    }
    cardinality = {
        str(fraction): observed_crossing_summary(
            [row.get("K_first") for row in selected],
            [bool(row["K_first_right_censored"]) for row in selected],
            k_max=64,
        )
        for fraction in sorted(
            {
                float(row["dictionary_fraction"])
                for row in raw_rows
                if row["null_family"] == "iid_cardinality_sweep"
            }
        )
        for selected in [
            [
                row
                for row in raw_rows
                if row["null_family"] == "iid_cardinality_sweep"
                and float(row["dictionary_fraction"]) == fraction
            ]
        ]
    }
    decisions = _registered_decisions(raw_rows, transformed_rows, diagnostic)
    decisions_by_name = {str(row["hypothesis"]): row for row in decisions}

    def crossing_payload(summary: Mapping[str, Any]) -> dict[str, Any]:
        crossing = summary["K_first_observed_and_censoring"]
        return {
            "observed_crossing_median": crossing["observed_crossing_median"],
            "observed_crossing_count": crossing["observed_crossing_count"],
            "right_censored_count": crossing["right_censored_count"],
            "right_censored_fraction": crossing["right_censored_fraction"],
            "population_median_reported": False,
        }

    def decision_payload(name: str) -> dict[str, Any]:
        row = decisions_by_name[name]
        return {
            "effect": row["effect"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
            "holm_adjusted_p": row["holm_adjusted_p"],
            "conclusion": row["conclusion"],
            "n_pairs": row["n_pairs"],
        }

    terminal_summary = {
        "schema_version": 1,
        "censoring_policy": (
            "observed-crossing medians and censoring fractions are separate; "
            "no population median K is reported"
        ),
        "A_raw_activation_vs_logistic_probe": {
            "common_layers": list(RAW_LOGISTIC_COMMON_LAYERS),
            "comparison_scope": "both families restricted to common layers",
            "raw_activation": crossing_payload(common_layer_family_summary["raw_activation"]),
            "logistic_probe": crossing_payload(common_layer_family_summary["logistic_probe"]),
        },
        "B_concept_contrast_vs_logistic_probe": {
            "concept_cells": {
                key: crossing_payload(value)
                for key, value in family_summary.items()
                if key.startswith("seven_emotion_label_contrast:")
            },
            "logistic_probe": crossing_payload(family_summary["logistic_probe"]),
        },
        "C_class_mean_vs_logistic_probe": {
            "class_mean_difference": crossing_payload(family_summary["class_mean_difference"]),
            "logistic_probe": crossing_payload(family_summary["logistic_probe"]),
        },
        "D_K_under_iid_full_cardinality": crossing_payload(null_summary["iid_full_cardinality"]),
        "E_K_under_random_prefix_same_k": crossing_payload(null_summary["random_prefix_same_k"]),
        "F_K_under_orthogonal_rotation": crossing_payload(null_summary["orthogonal_rotation"]),
        "G_K_by_random_dictionary_cardinality": cardinality,
        "H_PCA_improved_real_null_separation": decision_payload("metric:pca_r256_minus_raw"),
        "I_whitening_improved_real_null_separation": decision_payload(
            "metric:whitening_0.10_minus_raw"
        ),
        "J_fixed_budget_explained_energy": {
            family: {
                "median_explained_at_16": summary["median_explained_at_16"],
                "median_explained_at_25": summary["median_explained_at_25"],
                "semantics": "fixed reconstruction budgets, not occupancy estimates",
            }
            for family, summary in family_summary.items()
        },
        "result_only_decision_table": {
            "target_type_effect": decision_payload(
                "target_construction:logistic_minus_raw_activation"
            ),
            "full_cardinality_null_too_strong": decision_payload(
                "null_search:random_prefix_minus_iid_full"
            ),
            "geometry_preserving_null_changes_K": decision_payload(
                "dictionary_alignment:real_minus_rotation"
            ),
            "pca_improves_j_specific_separation": decision_payload("metric:pca_r256_minus_raw"),
            "whitening_improves_j_specific_separation": decision_payload(
                "metric:whitening_0.10_minus_raw"
            ),
        },
    }
    decision_table = {
        "schema_version": 2,
        "identity": diagnostic.identity,
        "primary_estimand": diagnostic.primary_estimand,
        "primary_estimand_formula": (
            "Delta4(t)=median_b[E_null(t,b,K=4)]-E_real(t,K=4), where E is "
            "normalized residual squared error; mechanism effects are registered "
            "numerator-minus-denominator contrasts of Delta4, except direct "
            "real-minus-rotation/permutation cells where Delta4 itself is the contrast"
        ),
        "secondary_estimand": diagnostic.secondary_estimand,
        "secondary_estimand_policy": (
            "max_cumulative_excess is data-adaptive over K and is exploratory only; "
            "it cannot produce Supported/Not supported mechanism conclusions"
        ),
        "primary_fixed_k": diagnostic.primary_fixed_k,
        "confidence_level": diagnostic.confidence_level,
        "multiple_comparison_policy": diagnostic.multiple_comparison_policy,
        "minimum_effect": diagnostic.minimum_effect,
        "equivalence_margin": diagnostic.equivalence_margin,
        "rules": {
            "Supported": "CI lower bound >= minimum effect and Holm-adjusted p <= 0.05",
            "Not supported": "CI upper bound <= equivalence margin",
            "Inconclusive": "all other outcomes",
        },
        "mechanism_hypotheses": decisions,
    }
    aggregate = {
        "schema_version": 1,
        "pilot_index": pilot_index,
        "raw_index": raw_index,
        "transformed_index": transformed_index,
        "family_summary": family_summary,
        "null_summary": null_summary,
        "cardinality_summary": cardinality,
        "target_pair_summary": pairs,
        "mechanism_decisions": decisions,
    }
    atomic_write_json(report_root / "decision_table.json", decision_table)
    atomic_write_json(report_root / "aggregate_summary.json", aggregate)
    atomic_write_json(report_root / "terminal_summary.json", terminal_summary)
    target_rows = raw_rows + transformed_rows
    _write_parquet(report_root / "target_level_summary.parquet", target_rows)
    _write_parquet(report_root / "target_pair_summary.parquet", pairs)
    _plot_registered_figures(
        report_root=report_root,
        raw_rows=raw_rows,
        transformed_rows=transformed_rows,
        pair_rows=pairs,
        real_gains=real_gains,
        null_gains=null_gains,
        theory_rows=theory_rows,
    )
    figure_data_manifest = _write_figure_data_package(
        root=root,
        report_root=report_root,
        identity=diagnostic.identity,
        target_rows=target_rows,
        pair_rows=pairs,
        gain_curve_rows=gain_curve_rows,
        theory_rows=theory_rows,
    )

    lines = [
        "# Qwen3.5-4B concept-occupancy K diagnostic v2",
        "",
        "## 1. Fixed budgets and occupancy are different quantities",
        "",
        "K=16 and K=25 below are fixed reconstruction budgets. They are not Anthropic-measured occupancy and are not substituted for first-crossing, consecutive-3, or cumulative-excess-peak K.",
        "",
        "| target family/arm/position | observed K_first median | censored fraction | explained@4 | explained@16 | explained@25 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in sorted(family_summary):
        summary = family_summary[family]
        crossing = summary["K_first_observed_and_censoring"]
        lines.append(
            "| "
            + " | ".join(
                [
                    family,
                    _fmt(crossing["observed_crossing_median"]),
                    _fmt(crossing["right_censored_fraction"]),
                    _fmt(summary["median_explained_at_4"]),
                    _fmt(summary["median_explained_at_16"]),
                    _fmt(summary["median_explained_at_25"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 2. Does target family determine K?",
            "",
            "Observed crossings and right-censored targets are shown separately; no population median K is reported. Label-explicit, label-free, four-unrelated-topics, assistant-boundary, concept-token-end, and definition-span-end cells remain separate.",
            "",
            "## 3. Null family and random-search cardinality",
            "",
            "Full-cardinality iid search, fixed random prefix, and orthogonal rotation remain distinct null hypotheses. Cardinality fractions are never silently collapsed. Empirical p-value resolution is limited to 1/(B+1) for B saved seeds.",
            "",
            "## 4. PCA and whitening metrics",
            "",
            "The decision table reports target-level effects, hierarchical-bootstrap confidence intervals, Holm-adjusted p-values, thresholds, and Supported / Not supported / Inconclusive conclusions. PCA and whitening re-estimate transformed atom norms and their matching nulls.",
            "",
            "## 5. Registered mechanism decisions",
            "",
            "The primary target-level estimand is Delta4(t)=median_b[E_null(t,b,4)]-E_real(t,4), with E the normalized residual squared error. Mechanism effects use the registered numerator-minus-denominator contrasts in decision_table.json. The data-adaptive maximum over K is exploratory and cannot support a mechanism conclusion.",
            "",
            *[
                f"- {row['hypothesis']}: numerator={row['numerator']}; denominator={row['denominator']}; pair_key={row['pair_key']}; n_pairs={row['n_pairs']}; clusters={row['cluster_counts']}; effect={row['effect']:.6g}, 95% CI=[{row['ci_low']:.6g}, {row['ci_high']:.6g}], conclusion={row['conclusion']}."
                for row in decisions
            ],
            "",
            "## 6. Limitations",
            "",
            "The exact-sphere curve is a one-sided signed-cosine isotropic reference, not an exact model for the anisotropic J dictionary. sqrt(explained@1) equals the maximum positive cosine only for the registered positive-direction best single-atom step. Fixed-C logistic permutation is conditional; class-mean permutation recomputes its direction. No generation or LLM judge is part of this experiment.",
            "",
            "Mechanism conclusions follow the preregistered thresholds; they are not a recommendation to deploy an intervention method.",
            "",
            "## 7. Self-contained figure data",
            "",
            "The exact plot-source tables for all ten registered figures are packaged under `report/data/`. `figure_data_manifest.json` maps every figure to its table, selection rule, output hashes, and upstream immutable index/summary hashes. The package does not duplicate the bundle/shard tree; those immutable artifacts remain the single scientific source of truth.",
        ]
    )
    (report_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_index = {
        "schema_version": 1,
        "identity": diagnostic.identity,
        "revision": "revision-r1",
        "complete": True,
        "files": {
            name: _artifact_descriptor(report_root / filename, relative_to=report_root)
            for name, filename in (
                ("decision_table", "decision_table.json"),
                ("aggregate_summary", "aggregate_summary.json"),
                ("terminal_summary", "terminal_summary.json"),
                ("target_level_summary", "target_level_summary.parquet"),
                ("target_pair_summary", "target_pair_summary.parquet"),
                ("report", "report.md"),
                ("figure_data_manifest", "data/figure_data_manifest.json"),
            )
        },
    }
    # This is deliberately last: it is the durable atomic completion boundary
    # for the multi-file derived package.
    atomic_write_json(report_root / "index.json", report_index)
    if final_report_root.exists() or final_report_root.is_symlink():
        raise KDiagnosticReportingError(
            "derived report output appeared during generation; staging output is "
            "preserved for same-revision recovery"
        )
    try:
        report_root.rename(final_report_root)
    except OSError as error:
        raise KDiagnosticReportingError(
            "cannot atomically publish the derived report; staging output is preserved "
            "for same-revision recovery"
        ) from error
    return {
        "index": str(final_report_root / "index.json"),
        "report": str(final_report_root / "report.md"),
        "decision_table": str(final_report_root / "decision_table.json"),
        "figures": [
            str(final_report_root / "figures" / f"{name}.pdf") for name in FIGURE_NAMES
        ],
        "figure_data_manifest": str(
            final_report_root / figure_data_manifest.relative_to(report_root)
        ),
        "terminal_summary": terminal_summary,
    }
