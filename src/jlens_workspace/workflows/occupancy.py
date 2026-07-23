"""Concept-vector J-space occupancy workflows.

For every (layer, convention, selection-mode, concept, sign, replicate) this
workflow runs a full-vocabulary streaming pursuit against the token J-direction
dictionary, repeats the identical solve against
``matched_gaussian_atom_norms_v1`` random dictionaries (one per seed), applies
both versioned crossing rules, and persists complete per-integer-k curves plus
reconstruction vectors at the reporting grid and at the primary stopping point.

The workflow deliberately takes ALREADY-VERIFIED inputs (probe vectors with
their SHA-256s, a dictionary factory bound to an identity-checked lens): all
model/tokenizer/lens/probe identity validation lives at the CLI boundary, so
this module stays fully offline-testable with synthetic dictionaries.

``w_nonJ(k) = w - w_J(k)`` is the sparse-reconstruction residual only; it is
NOT an orthogonal complement and every artifact labels it accordingly.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from numpy.typing import NDArray

from jlens_workspace.artifacts import atomic_write_json, sha256_file
from jlens_workspace.pursuit import (
    MatchedNormRandomDictionary,
    PursuitResult,
    absolute_threshold_ks,
    crossing_k,
    k_90_attainable,
    streaming_nonnegative_pursuit,
)
from jlens_workspace.pursuit.dictionaries import (
    RANDOM_CONTROL_METHOD,
    RANDOM_CONTROL_RNG,
)
from jlens_workspace.pursuit.occupancy import CROSSING_RULES

FloatArray = NDArray[np.float64]

SIGN_LABELS = {"+": "pos", "-": "neg"}
OCCUPANCY_METHODS = (
    "concept_occupancy_method_v1",
    "concept_occupancy_method_v2",
    "concept_occupancy_method",
)


class OccupancyWorkflowError(ValueError):
    """Raised when occupancy inputs violate the workflow contract."""


CONTROL_SUBSAMPLE_KEY = 777


def subsample_control_norms(atom_norms: FloatArray, fraction: float) -> FloatArray:
    """Deterministic norm subsample for the low-cardinality control variant.

    ``fraction == 1.0`` returns the input untouched (the primary control).
    Otherwise a Philox permutation keyed by ``(CONTROL_SUBSAMPLE_KEY,
    round(fraction * 1e6))`` selects ``ceil(fraction * V)`` matched norms, so
    the reduced control dictionary is a pure function of (norms, fraction).
    """

    value = float(fraction)
    if not 0.0 < value <= 1.0:
        raise OccupancyWorkflowError("control fraction must lie in (0, 1]")
    if value == 1.0:
        return atom_norms
    count = int(np.ceil(value * atom_norms.size))
    rng = np.random.Generator(
        np.random.Philox(key=[CONTROL_SUBSAMPLE_KEY, round(value * 1_000_000)])
    )
    return atom_norms[rng.permutation(atom_norms.size)[:count]]


def _fraction_slug(fraction: float) -> str:
    return f"{fraction:g}".replace(".", "p")


def verify_lens_artifact_sha256(path: str | Path, expected_sha256: str) -> str:
    """Fail closed unless the lens file hash matches the pinned expectation."""

    expected = expected_sha256.strip().casefold()
    if len(expected) != 64:
        raise OccupancyWorkflowError("expected_lens_sha256 must be a 64-char SHA-256")
    observed = sha256_file(path)
    if observed != expected:
        raise OccupancyWorkflowError(
            f"lens artifact SHA-256 mismatch: expected {expected}, observed "
            f"{observed} at {path}"
        )
    return observed


@dataclass(frozen=True)
class ConceptTarget:
    """One probe-derived decomposition target (sign applied by the workflow)."""

    layer: int
    concept_id: str
    vector: FloatArray
    vector_sha256: str
    replicate_id: str = "primary"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.vector, dtype=np.float64)
        if array.ndim != 1 or array.size == 0:
            raise OccupancyWorkflowError("target vector must be a non-empty [D] vector")
        if not np.isfinite(array).all() or not np.linalg.norm(array):
            raise OccupancyWorkflowError("target vector must be finite and non-zero")
        object.__setattr__(self, "vector", array)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _combo_dir(
    root: Path,
    *,
    convention: str,
    mode: str,
    layer: int,
    concept_id: str,
    sign: str,
    replicate_id: str,
) -> Path:
    return (
        root
        / convention
        / mode
        / f"layer_{layer:02d}"
        / quote(concept_id, safe="")
        / SIGN_LABELS[sign]
        / quote(replicate_id, safe="")
    )


def _summarize(
    result: PursuitResult,
    controls: Sequence[PursuitResult],
    *,
    seeds: Sequence[int],
    thresholds: Sequence[float],
    primary_crossing_rule: str,
    decode_token: Callable[[int], str] | None,
) -> dict[str, Any]:
    control_gains = [control.gains for control in controls]
    occupancy = {
        rule: crossing_k(result.gains, control_gains, rule=rule)
        for rule in CROSSING_RULES
    }
    primary = occupancy[primary_crossing_rule]
    crossing = primary.get("k")
    if crossing is None:
        selected_k = int(result.k_max)
    else:
        selected_k = max(0, int(crossing) - 1)
    primary_occupancy = {
        "rule": primary_crossing_rule,
        "crossing_k": crossing,
        "k_selected_before_crossing": selected_k,
        "right_censored": bool(primary["right_censored"]),
        "semantics": (
            "crossing_k is the first candidate whose marginal gain does not "
            "exceed the median matched-random gain; "
            "k_selected_before_crossing counts the preceding supported atoms"
        ),
    }
    support = result.support.tolist()
    payload: dict[str, Any] = {
        "target_norm": result.target_norm,
        "selection_mode": result.selection_mode,
        "solver_method": result.solver_method,
        "support_token_ids": support,
        "decoded_tokens": (
            [decode_token(token) for token in support] if decode_token else None
        ),
        "selected_atom_norms": result.selected_atom_norms.tolist(),
        "coefficients_per_k": [c.tolist() for c in result.coefficients_per_k],
        "residual_norms": result.residual_norms.tolist(),
        "j_component_norms": result.j_component_norms.tolist(),
        "cosine_full_probe_vs_j_component": result.cosine_with_target.tolist(),
        "support_condition_numbers": [
            None if not np.isfinite(v) else v for v in result.condition_numbers
        ],
        "stopped_early_at": result.stopped_early_at,
        "occupancy": occupancy,
        "primary_occupancy": primary_occupancy,
        "k_90_attainable": k_90_attainable(result.errors),
        "absolute_explained_fraction_ks": {
            key: (value if value is not None else None)
            for key, value in absolute_threshold_ks(
                result.errors, thresholds
            ).items()
        },
        "explained_fraction_final": float(1.0 - result.errors[-1]),
        "random_controls": {
            "method": RANDOM_CONTROL_METHOD,
            "rng": RANDOM_CONTROL_RNG,
            "seeds": list(seeds),
            "final_errors": [float(control.errors[-1]) for control in controls],
            "stopped_early_at": [control.stopped_early_at for control in controls],
        },
    }
    return payload


def run_concept_occupancy(
    *,
    output_dir: str | Path,
    dictionary_factory: Callable[[int, str], Any],
    targets: Sequence[ConceptTarget],
    conventions: Sequence[str],
    selection_modes: Sequence[str],
    signs: Sequence[str] = ("+", "-"),
    k_max: int = 64,
    report_grid: Sequence[int] = (1, 2, 4, 8, 16, 25, 32, 64),
    random_seeds: Sequence[int] = (101, 202, 303, 404, 505),
    control_atom_fractions: Sequence[float] = (1.0,),
    absolute_thresholds: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
    chunk_size: int = 4096,
    decode_token: Callable[[int], str] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    method: str = "concept_occupancy_method_v1",
    solver_method: str = "nnomp_nnls_v1",
    primary_crossing_rule: str = "first_nonexceed_v1",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run every requested combination, resumably, under ``output_dir``.

    A combination whose ``metrics.json`` already exists is skipped untouched
    (unless ``overwrite``), so an interrupted run resumes without changing
    completed outputs. Random-control pursuits share each full-dictionary scan
    across all targets/modes at a layer, which keeps the streamed Gaussian
    control affordable.
    """

    if not targets:
        raise OccupancyWorkflowError("at least one target is required")
    if method not in OCCUPANCY_METHODS:
        raise OccupancyWorkflowError(f"unsupported occupancy method: {method}")
    if primary_crossing_rule not in CROSSING_RULES:
        raise OccupancyWorkflowError(
            f"unsupported primary crossing rule: {primary_crossing_rule}"
        )
    grid = [int(k) for k in report_grid]
    if not grid or sorted(set(grid)) != grid or grid[0] < 1 or grid[-1] > k_max:
        raise OccupancyWorkflowError(
            "report_grid must be sorted, unique, and lie in [1, k_max]"
        )
    unknown_signs = sorted(set(signs) - set(SIGN_LABELS))
    if unknown_signs:
        raise OccupancyWorkflowError(f"unknown signs: {unknown_signs}")
    seeds = [int(seed) for seed in random_seeds]
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise OccupancyWorkflowError("random_seeds must be unique and non-negative")
    fractions = [float(fraction) for fraction in control_atom_fractions]
    if not fractions or len(set(fractions)) != len(fractions):
        raise OccupancyWorkflowError("control_atom_fractions must be non-empty and unique")
    if 1.0 not in fractions:
        raise OccupancyWorkflowError("control_atom_fractions must include 1.0")
    if any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise OccupancyWorkflowError("control_atom_fractions must lie in (0, 1]")

    root = Path(output_dir) / "occupancy"
    root.mkdir(parents=True, exist_ok=True)

    layers = sorted({target.layer for target in targets})
    completed = 0
    skipped = 0

    for layer in layers:
        layer_targets = [target for target in targets if target.layer == layer]
        for convention in conventions:
            # Enumerate the (target, sign, mode) jobs still pending.
            jobs: list[tuple[ConceptTarget, str, str]] = []
            for target in layer_targets:
                for sign in signs:
                    for mode in selection_modes:
                        combo = _combo_dir(
                            root,
                            convention=convention,
                            mode=mode,
                            layer=layer,
                            concept_id=target.concept_id,
                            sign=sign,
                            replicate_id=target.replicate_id,
                        )
                        if (combo / "metrics.json").is_file() and not overwrite:
                            skipped += 1
                            continue
                        jobs.append((target, sign, mode))
            if not jobs:
                continue

            dictionary = dictionary_factory(layer, convention)
            atom_norms = np.asarray(dictionary.atom_norms(), dtype=np.float64)
            norms_path = root / convention / f"layer_{layer:02d}_atom_norms.npy"
            if not norms_path.is_file():
                _atomic_save_npy(norms_path, atom_norms)

            batch = np.stack(
                [
                    (1.0 if sign == "+" else -1.0) * target.vector
                    for target, sign, _ in jobs
                ]
            )
            modes = [mode for _, _, mode in jobs]
            real_results = streaming_nonnegative_pursuit(
                dictionary,
                batch,
                k_max=k_max,
                selection_modes=modes,
                solver_method=solver_method,
            )
            controls_by_fraction: dict[float, dict[int, list[PursuitResult]]] = {}
            for fraction in fractions:
                norms_f = subsample_control_norms(atom_norms, fraction)
                per_seed: dict[int, list[PursuitResult]] = {}
                for seed in seeds:
                    control_dictionary = MatchedNormRandomDictionary(
                        norms_f,
                        seed=seed,
                        d_model=int(dictionary.d_model),
                        chunk_size=chunk_size,
                    )
                    per_seed[seed] = streaming_nonnegative_pursuit(
                        control_dictionary,
                        batch,
                        k_max=k_max,
                        selection_modes=modes,
                        solver_method=solver_method,
                    )
                controls_by_fraction[fraction] = per_seed
            control_results = controls_by_fraction[1.0]

            for job_index, (target, sign, mode) in enumerate(jobs):
                result = real_results[job_index]
                controls = [control_results[seed][job_index] for seed in seeds]
                combo = _combo_dir(
                    root,
                    convention=convention,
                    mode=mode,
                    layer=layer,
                    concept_id=target.concept_id,
                    sign=sign,
                    replicate_id=target.replicate_id,
                )
                combo.mkdir(parents=True, exist_ok=True)
                _atomic_save_npy(combo / "errors.npy", result.errors)
                _atomic_save_npy(combo / "gains.npy", result.gains)
                _atomic_save_npy(
                    combo / "control_errors.npy",
                    np.stack([control.errors for control in controls]),
                )
                _atomic_save_npy(
                    combo / "control_gains.npy",
                    np.stack([control.gains for control in controls]),
                )
                for fraction in fractions:
                    if fraction == 1.0:
                        continue
                    extra = [
                        controls_by_fraction[fraction][seed][job_index]
                        for seed in seeds
                    ]
                    slug = _fraction_slug(fraction)
                    _atomic_save_npy(
                        combo / f"control_errors_f{slug}.npy",
                        np.stack([control.errors for control in extra]),
                    )
                    _atomic_save_npy(
                        combo / f"control_gains_f{slug}.npy",
                        np.stack([control.gains for control in extra]),
                    )
                _atomic_save_npy(
                    combo / "support_token_ids.npy",
                    result.support.astype(np.int64),
                )
                signed_target = (1.0 if sign == "+" else -1.0) * target.vector
                payload = _summarize(
                    result,
                    controls,
                    seeds=seeds,
                    thresholds=absolute_thresholds,
                    primary_crossing_rule=primary_crossing_rule,
                    decode_token=decode_token,
                )
                selected_k = int(
                    payload["primary_occupancy"]["k_selected_before_crossing"]
                )
                reconstruction_ks = sorted(set(grid) | {selected_k})
                for reconstruction_k in reconstruction_ks:
                    effective_k = min(reconstruction_k, result.support.size)
                    if effective_k:
                        atoms = dictionary.materialize(result.support[:effective_k])
                        w_j = result.coefficients_at(effective_k) @ atoms
                    else:
                        w_j = np.zeros_like(signed_target)
                    _atomic_save_npy(
                        combo / f"w_J_k{reconstruction_k:02d}.npy", w_j
                    )
                    _atomic_save_npy(
                        combo / f"w_nonJ_k{reconstruction_k:02d}.npy",
                        signed_target - w_j,
                    )
                payload["primary_occupancy"]["w_j_file"] = (
                    f"w_J_k{selected_k:02d}.npy"
                )
                payload["primary_occupancy"]["w_nonj_file"] = (
                    f"w_nonJ_k{selected_k:02d}.npy"
                )
                if len(fractions) > 1:
                    payload["occupancy_by_control_fraction"] = {
                        f"{fraction:g}": {
                            rule: crossing_k(
                                result.gains,
                                [
                                    controls_by_fraction[fraction][seed][
                                        job_index
                                    ].gains
                                    for seed in seeds
                                ],
                                rule=rule,
                            )
                            for rule in CROSSING_RULES
                        }
                        for fraction in fractions
                    }
                    payload["random_controls"]["fractions"] = [
                        {
                            "fraction": fraction,
                            "method": RANDOM_CONTROL_METHOD
                            if fraction == 1.0
                            else "matched_gaussian_atom_norms_subsampled_v1",
                            "n_atoms": int(
                                subsample_control_norms(atom_norms, fraction).size
                            ),
                            "subsample_key": None
                            if fraction == 1.0
                            else [CONTROL_SUBSAMPLE_KEY, round(fraction * 1_000_000)],
                        }
                        for fraction in fractions
                    ]
                payload.update(
                    {
                        "schema_version": 1,
                        "method": method,
                        "layer": layer,
                        "concept_id": target.concept_id,
                        "sign": sign,
                        "replicate_id": target.replicate_id,
                        "convention": convention,
                        "k_max": k_max,
                        "report_grid": grid,
                        "chunk_size": chunk_size,
                        "probe_vector_sha256": target.vector_sha256,
                        "probe_provenance": dict(target.provenance),
                        "atom_norms_file": str(
                            norms_path.relative_to(root)
                        ),
                        "non_j_note": (
                            "w_nonJ(k) is the sparse-reconstruction residual, "
                            "not an orthogonal complement"
                        ),
                        "run_metadata": dict(run_metadata or {}),
                    }
                )
                atomic_write_json(combo / "metrics.json", payload)
                completed += 1

    return {"completed": completed, "skipped": skipped, "output": str(root)}


def rebuild_occupancy_index(
    output_dir: str | Path,
    *,
    method: str,
    expected_combinations: int | None = None,
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Scan immutable combo metrics and atomically rebuild the shared index."""

    if method not in OCCUPANCY_METHODS:
        raise OccupancyWorkflowError(f"unsupported occupancy method: {method}")
    root = Path(output_dir) / "occupancy"
    entries: list[dict[str, Any]] = []
    identities: set[tuple[object, ...]] = set()
    for metrics_path in sorted(root.rglob("metrics.json")):
        payload = load_combo_metrics(metrics_path)
        if payload["method"] != method:
            raise OccupancyWorkflowError(
                f"{metrics_path}: method {payload['method']!r} != {method!r}"
            )
        identity = (
            payload["layer"],
            payload["concept_id"],
            payload["sign"],
            payload["replicate_id"],
            payload["convention"],
            payload["selection_mode"],
        )
        if identity in identities:
            raise OccupancyWorkflowError(
                f"duplicate occupancy identity at {metrics_path}: {identity}"
            )
        identities.add(identity)
        entries.append(
            {
                "path": str(metrics_path.parent.relative_to(root)),
                "identity": list(identity),
                "primary_occupancy": payload["primary_occupancy"],
                "probe_vector_sha256": payload["probe_vector_sha256"],
            }
        )
    observed = len(entries)
    missing = (
        None
        if expected_combinations is None
        else max(0, int(expected_combinations) - observed)
    )
    if expected_combinations is not None and observed > expected_combinations:
        raise OccupancyWorkflowError(
            f"found {observed} combos but expected at most {expected_combinations}"
        )
    index = {
        "schema_version": 2,
        "method": method,
        "random_control_method": RANDOM_CONTROL_METHOD,
        "crossing_rules": list(CROSSING_RULES),
        "observed_combinations": observed,
        "expected_combinations": expected_combinations,
        "missing_combinations": missing,
        "complete": missing == 0 if missing is not None else None,
        "entries": entries,
        "run_metadata": dict(run_metadata or {}),
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "index.json", index)
    return index


def load_combo_metrics(path: str | Path) -> dict[str, Any]:
    """Load one combination's metrics.json (helper for reports and audits)."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("method") not in OCCUPANCY_METHODS
    ):
        raise OccupancyWorkflowError(f"unsupported occupancy metrics: {path}")
    return payload
