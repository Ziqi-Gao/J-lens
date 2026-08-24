from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import jlens_workspace.concept_intervention.k_diagnostic.experiment as kdiag_experiment
from jlens_workspace.artifacts import sha256_file
from jlens_workspace.cli import _k_diagnostic_v2_benchmark_memory_plan
from jlens_workspace.concept_intervention.k_diagnostic import (
    LinearTransformedDictionary,
    TargetRecord,
    class_mean_difference,
    classify_registered_effect,
    find_unique_subsequence,
    fit_fixed_c_permutation_probe,
    group_safe_permutation,
    haar_orthogonal,
    hierarchical_bootstrap_mean,
    matched_norm_subset,
    observed_crossing_summary,
    paired_template_contrasts,
    permuted_class_mean_difference,
    random_prefix_pursuit,
    raw_activation_positions,
    sphere_max_quantile,
    summarize_null_calibrated_curve,
)
from jlens_workspace.concept_intervention.k_diagnostic.experiment import (
    EXECUTION_HARDWARE_GATE_VERSION,
    NULL_EXECUTION_VERSION,
    REQUIRED_SHARD_FILES,
    KDiagnosticExperimentError,
    PhysicalBundle,
    ScientificShard,
    audit_rotation_cache_for_grid,
    build_physical_bundles,
    build_scientific_grid,
    detect_k_diagnostic_execution_hardware,
    execution_hardware_sha256,
    index_stage,
    load_target_index,
    prepare_rotation_cache,
    record_bundle_runtime,
    resource_preflight,
    rotation_cache_build_preflight,
    run_scientific_bundle,
    seal_target_membership,
    validate_k_diagnostic_execution_hardware,
)
from jlens_workspace.concept_intervention.k_diagnostic.nulls import (
    SHARED_DIRECTION_MAX_PEAK_BYTES,
    SHARED_DIRECTION_MIN_AVAILABLE_RESERVE_BYTES,
    SharedDirectionProvider,
    SharedDirectionRandomDictionary,
    haar_cache_id,
    inverse_rotate_target,
    load_cached_haar_rotation,
    random_prefix_atoms,
    shared_direction_memory_plan,
)
from jlens_workspace.concept_intervention.k_diagnostic.reporting import (
    FIGURE_NAMES,
    KDiagnosticReportingError,
    _validate_stage_summary,
    _within_target_iid_gain_curves,
    generate_report,
)
from jlens_workspace.concept_intervention.k_diagnostic.targets import (
    FOUR_UNRELATED_TOPIC_DESCRIPTIONS,
    LABEL_FREE_PARAPHRASES,
    TargetConstructionError,
    prepare_shared_statistical_targets,
    save_target_records,
)
from jlens_workspace.config import load_experiment_config
from jlens_workspace.pursuit import (
    DenseDictionary,
    MatchedNormRandomDictionary,
    streaming_nonnegative_pursuit,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "Concept_intervention" / "configs"


def _diagnostic_v1_config(stage: str = "pilot"):
    suffix = {
        "pilot": "pilot",
        "raw_metric_full": "full",
        "transformed_metric": "transformed",
    }[stage]
    return load_experiment_config(CONFIG_ROOT / f"qwen35_4b_k_diagnostic_v1_{suffix}.yaml")


def _diagnostic_config(stage: str = "pilot"):
    suffix = {
        "pilot": "pilot",
        "raw_metric_full": "full",
        "transformed_metric": "transformed",
    }[stage]
    return load_experiment_config(CONFIG_ROOT / f"qwen35_4b_k_diagnostic_v2_{suffix}.yaml")


def _scheduled_blackwell_hardware(*, backend: str = "fsm_local_scheduler") -> dict[str, object]:
    return {
        "schema_version": 1,
        "gate_version": EXECUTION_HARDWARE_GATE_VERSION,
        "execution_backend": backend,
        "cuda_available": True,
        "gpu_count": 1,
        "gpu_names": ["NVIDIA RTX PRO 6000 Blackwell"],
        "visible_device_indices": [0],
        "torch_version": "2.8.0",
        "cuda_version": "12.8",
    }


class _FakeCuda:
    def __init__(self, *, available: bool, names: list[str]) -> None:
        self._available = available
        self._names = names

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return len(self._names)

    def get_device_name(self, index: int) -> str:
        return self._names[index]


class _FakeTorch:
    def __init__(
        self,
        *,
        available: bool,
        names: list[str],
        torch_version: str = "2.8.0",
        cuda_version: str = "12.8",
    ) -> None:
        self.__version__ = torch_version
        self.cuda = _FakeCuda(available=available, names=names)
        self.version = type("Version", (), {"cuda": cuda_version})()


def _local_gpu_lease_env(tmp_path: Path) -> dict[str, str]:
    runtime_root = tmp_path / "runtime"
    lease_id = "test-lease"
    lease_path = runtime_root / "scheduler" / "leases" / f"{lease_id}.json"
    lease_path.parent.mkdir(parents=True)
    lease_path.write_text(
        json.dumps({"lease_id": lease_id, "kind": "gpu", "gpu_index": 3}),
        encoding="utf-8",
    )
    return {
        "JLENS_LOCAL_LEASE_ID": lease_id,
        "JLENS_LOCAL_RUNTIME_ROOT": str(runtime_root),
    }


def _seal_v2_index_runtime_hardware(
    config,
    shards: list[ScientificShard],
    *,
    run_root: Path,
) -> None:
    diagnostic = config.k_diagnostic
    assert diagnostic.identity == "qwen35_4b_k_diagnostic_v2"
    hardware = _scheduled_blackwell_hardware()
    for bundle in build_physical_bundles(shards):
        record_bundle_runtime(
            config,
            bundle_id=bundle.bundle_id,
            wall_seconds=1.0,
            execution_hardware=hardware,
            run_root=run_root,
        )
    stage = run_root / diagnostic.artifact_root / diagnostic.stage
    (stage / "resource_preflight.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "approved": True,
                "identity": diagnostic.identity,
                "stage": diagnostic.stage,
                "configuration_sha256": kdiag_experiment._stage_manifest(config)[
                    "configuration_sha256"
                ],
                "null_execution_version": NULL_EXECUTION_VERSION,
                "execution_backend": hardware["execution_backend"],
                "execution_hardware": hardware,
                "execution_hardware_sha256": execution_hardware_sha256(hardware),
            }
        ),
        encoding="utf-8",
    )


def test_v2_execution_hardware_gate_requires_scheduled_single_cuda(
    tmp_path: Path,
) -> None:
    expected = _scheduled_blackwell_hardware()
    local_env = _local_gpu_lease_env(tmp_path)
    assert validate_k_diagnostic_execution_hardware(expected) == expected
    assert len(execution_hardware_sha256(expected)) == 64
    assert (
        detect_k_diagnostic_execution_hardware(
            _FakeTorch(available=True, names=["NVIDIA RTX PRO 6000 Blackwell"]),
            env=local_env,
        )
        == expected
    )

    cpu = dict(expected)
    cpu.update(
        {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_names": [],
            "visible_device_indices": [],
        }
    )
    missing = dict(expected)
    missing.pop("gpu_names")
    with_hostname = dict(expected)
    with_hostname["hostname"] = "not-an-identity-field"
    for rejected in (None, cpu, missing, with_hostname):
        with pytest.raises(KDiagnosticExperimentError):
            validate_k_diagnostic_execution_hardware(rejected)
    with pytest.raises(KDiagnosticExperimentError, match="allocation"):
        detect_k_diagnostic_execution_hardware(
            _FakeTorch(available=True, names=["NVIDIA RTX PRO 6000 Blackwell"]),
            env={},
        )
    for fake in (
        _FakeTorch(available=False, names=[]),
        _FakeTorch(available=True, names=["GPU 0", "GPU 1"]),
        _FakeTorch(available=True, names=["GPU 0"], torch_version=""),
        _FakeTorch(available=True, names=["GPU 0"], cuda_version=""),
    ):
        with pytest.raises(KDiagnosticExperimentError):
            detect_k_diagnostic_execution_hardware(fake, env=local_env)
    assert detect_k_diagnostic_execution_hardware(
        _FakeTorch(available=True, names=["NVIDIA RTX PRO 6000 Blackwell"]),
        env={"SLURM_JOB_ID": "12345"},
    ) == _scheduled_blackwell_hardware(backend="slurm")


def _solve(atoms: np.ndarray, target: np.ndarray, k_max: int = 4):
    return streaming_nonnegative_pursuit(
        DenseDictionary(atoms),
        target[None, :],
        k_max=k_max,
        solver_method="nonnegative_gradient_pursuit_standard",
    )[0]


def _decode_token_ids(token_ids: list[int]) -> list[str]:
    return [f"token:{token_id}" for token_id in token_ids]


def _synthetic_empirical(seed_count: int, *, k_max: int = 4) -> dict[str, object]:
    resolution = 1.0 / (seed_count + 1)
    cell = {
        "empirical_upper_tail_p_value": resolution,
        "empirical_percentile": 1.0,
    }
    return {
        "schema_version": 1,
        "null_seed_count": seed_count,
        "resolution": resolution,
        "plus_one_correction": True,
        "marginal_gain_by_k": [{"K": k, **cell} for k in range(1, k_max + 1)],
        "fixed_k4": dict(cell),
        "fixed_reconstruction_benefit": {"4": dict(cell)},
        "cumulative_benefit": dict(cell),
    }


def test_rotation_is_orthogonal_norm_preserving_and_gram_preserving() -> None:
    rng = np.random.default_rng(8)
    atoms = rng.normal(size=(12, 7))
    target = rng.normal(size=7)
    rotation, metadata = haar_orthogonal(7, seed=1101)
    rotated_atoms = atoms @ rotation.T
    rotated_target = rotation @ target

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(7), atol=1e-12)
    assert metadata["normalized_frobenius_orthogonality_error"] < 1e-5
    assert np.linalg.norm(rotated_target) == pytest.approx(np.linalg.norm(target))
    np.testing.assert_allclose(
        rotated_atoms @ rotated_atoms.T, atoms @ atoms.T, rtol=1e-12, atol=1e-12
    )


def test_rotate_dictionary_equals_inverse_rotate_target_reconstruction_error() -> None:
    rng = np.random.default_rng(19)
    atoms = rng.normal(size=(30, 8))
    target = rng.normal(size=8)
    rotation, _ = haar_orthogonal(8, seed=2202)
    inverse_target, norm_error = inverse_rotate_target(target, rotation)

    explicit = _solve(atoms @ rotation.T, target)
    inverse = _solve(atoms, inverse_target)
    np.testing.assert_allclose(explicit.errors, inverse.errors, atol=1e-12)
    assert norm_error < 1e-6


def test_exact_haar_cache_is_built_once_reused_and_fails_closed(
    tmp_path: Path,
) -> None:
    calls = 0

    def constructor(dimension: int, *, seed: int):
        nonlocal calls
        calls += 1
        return haar_orthogonal(dimension, seed=seed)

    first, first_manifest = load_cached_haar_rotation(
        tmp_path,
        metric="raw_euclidean",
        transform_sha256=None,
        dimension=6,
        seed=1101,
        create=True,
        constructor=constructor,
    )
    second, second_manifest = load_cached_haar_rotation(
        tmp_path,
        metric="j_pca_r6",
        transform_sha256="a" * 64,
        dimension=6,
        seed=1101,
        create=True,
        constructor=constructor,
    )
    assert calls == 1
    assert isinstance(first, np.memmap) and isinstance(second, np.memmap)
    assert first_manifest == second_manifest
    assert first_manifest["identity"] == {
        "algorithm_version": "gaussian_qr_haar_cached_v3_shared_dimension_seed",
        "dimension": 6,
        "seed": 1101,
    }

    matrix = tmp_path / "matrices" / str(first_manifest["cache_id"]) / "matrix.npy"
    with matrix.open("r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_cached_haar_rotation(
            tmp_path,
            metric="raw_euclidean",
            transform_sha256=None,
            dimension=6,
            seed=1101,
        )

    partial_root = tmp_path / "partial"
    partial_id = haar_cache_id(
        metric="raw_euclidean",
        transform_sha256=None,
        dimension=6,
        seed=1102,
    )
    (partial_root / "matrices" / partial_id).mkdir(parents=True)
    with pytest.raises(ValueError, match="partial Haar cache"):
        load_cached_haar_rotation(
            partial_root,
            metric="raw_euclidean",
            transform_sha256=None,
            dimension=6,
            seed=1102,
            create=True,
        )

    locked_root = tmp_path / "locked"
    locked_id = haar_cache_id(
        metric="raw_euclidean",
        transform_sha256=None,
        dimension=6,
        seed=1103,
    )
    lock = locked_root / "locks" / f"{locked_id}.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("competing cache builder", encoding="utf-8")
    with pytest.raises(ValueError, match="already locked"):
        load_cached_haar_rotation(
            locked_root,
            metric="raw_euclidean",
            transform_sha256=None,
            dimension=6,
            seed=1103,
            create=True,
        )


def test_cardinality_increases_isotropic_max_and_first_gain_without_changing_real() -> None:
    rng = np.random.default_rng(23)
    target = rng.normal(size=32)
    directions = rng.normal(size=(4096, 32))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    small = directions[:64]
    small_max = float(np.max(small @ target / np.linalg.norm(target)))
    full_max = float(np.max(directions @ target / np.linalg.norm(target)))
    assert full_max > small_max

    small_result = _solve(small, target, k_max=1)
    full_result = _solve(directions, target, k_max=1)
    assert full_result.gains[0] > small_result.gains[0]

    real_atoms = rng.normal(size=(48, 32))
    before = _solve(real_atoms, target)
    # Constructing reduced null norms must not mutate or replace the real dictionary.
    subset, metadata = matched_norm_subset(
        np.linalg.norm(directions, axis=1), fraction=1 / 16, seed=101
    )
    after = _solve(real_atoms, target)
    assert subset.size == 256 and metadata["actual_cardinality"] == 256
    np.testing.assert_array_equal(before.errors, after.errors)


def test_rotation_cache_build_preflight_shares_dimension_seed_and_gates_budget(
    tmp_path: Path,
) -> None:
    source = _diagnostic_config()
    config = source.model_copy(
        update={
            "k_diagnostic": source.k_diagnostic.model_copy(update={"rotation_seeds": [1101, 1102]})
        }
    )
    spaces = [
        {
            "metric": "raw_euclidean",
            "transform_sha256": None,
            "dimension": 6,
            "layers": [11],
        },
        {
            "metric": "j_pca_r6",
            "transform_sha256": "b" * 64,
            "dimension": 6,
            "layers": [19],
        },
    ]
    preflight = rotation_cache_build_preflight(config, metric_spaces=spaces, run_root=tmp_path)
    assert preflight["approved"] is True
    assert preflight["max_dimension"] == 6
    assert preflight["binding_count"] == 4
    assert preflight["unique_matrix_count"] == 2
    assert preflight["sample"]["qr_wall_seconds"] > 0.0
    assert preflight["sample"]["cache_entry_bytes"] > 6 * 6 * 8
    assert preflight["bytes_per_entry_by_dimension"]["6"] > 6 * 6 * 8
    assert preflight["projected_cache_inodes"] >= 2
    assert preflight["configuration_sha256"]
    assert preflight["binding_plan_sha256"]
    assert preflight["hard_budgets"]["effective_cache_gib_budget"] == min(
        config.k_diagnostic.max_rotation_cache_gib,
        config.k_diagnostic.max_estimated_disk_gib,
        config.k_diagnostic.max_total_estimated_disk_gib - 150.0,
    )
    index = prepare_rotation_cache(config, metric_spaces=spaces, run_root=tmp_path)
    assert index["binding_count"] == 4
    assert index["unique_matrix_count"] == 2
    assert index["configuration_sha256"] == preflight["configuration_sha256"]
    assert index["binding_plan_sha256"] == preflight["binding_plan_sha256"]
    assert len({entry["cache_id"] for entry in index["entries"]}) == 2
    assert index["matrix_bytes"] > 2 * 6 * 6 * 8

    rejected_root = tmp_path / "rejected"
    rejected = config.model_copy(
        update={
            "k_diagnostic": config.k_diagnostic.model_copy(
                update={"max_rotation_cache_build_seconds": 1e-12}
            )
        }
    )
    with pytest.raises(KDiagnosticExperimentError, match="preflight rejected"):
        rotation_cache_build_preflight(rejected, metric_spaces=spaces, run_root=rejected_root)


def test_rotation_cache_preflight_stale_config_and_stage_disk_cap_fail_closed(
    tmp_path: Path,
) -> None:
    source = _diagnostic_config()
    config = source.model_copy(
        update={"k_diagnostic": source.k_diagnostic.model_copy(update={"rotation_seeds": [1101]})}
    )
    spaces = [
        {
            "metric": "raw_euclidean",
            "transform_sha256": None,
            "dimension": 6,
            "layers": [11],
        }
    ]
    preflight = rotation_cache_build_preflight(config, metric_spaces=spaces, run_root=tmp_path)
    prepare_rotation_cache(config, metric_spaces=spaces, run_root=tmp_path)

    for changed_budget in (
        {"max_rotation_cache_build_seconds": 43_200.0},
        {"max_rotation_cache_gib": 24.0},
    ):
        stale = config.model_copy(
            update={"k_diagnostic": config.k_diagnostic.model_copy(update=changed_budget)}
        )
        with pytest.raises(KDiagnosticExperimentError, match="identity mismatch"):
            rotation_cache_build_preflight(stale, metric_spaces=spaces, run_root=tmp_path)
        with pytest.raises(KDiagnosticExperimentError, match="absent, stale, or rejected"):
            prepare_rotation_cache(stale, metric_spaces=spaces, run_root=tmp_path)
    assert preflight["binding_plan_sha256"]

    disk_cap_root = tmp_path / "disk-cap"
    tiny_stage_cap = 1e-10
    disk_capped = config.model_copy(
        update={
            "k_diagnostic": config.k_diagnostic.model_copy(
                update={
                    "max_rotation_cache_gib": 100.0,
                    "max_estimated_disk_gib": tiny_stage_cap,
                }
            )
        }
    )
    with pytest.raises(KDiagnosticExperimentError, match="preflight rejected"):
        rotation_cache_build_preflight(disk_capped, metric_spaces=spaces, run_root=disk_cap_root)
    disk_preflight_path = (
        disk_cap_root
        / disk_capped.k_diagnostic.artifact_root
        / "pilot/rotation_cache_build_preflight.json"
    )
    disk_preflight = json.loads(disk_preflight_path.read_text(encoding="utf-8"))
    assert disk_preflight["approved"] is False
    assert disk_preflight["projected_cache_gib"] > tiny_stage_cap
    assert disk_preflight["hard_budgets"]["effective_cache_gib_budget"] == tiny_stage_cap
    assert disk_preflight["checks"]["projected_cache_within_budget"] is False
    with pytest.raises(KDiagnosticExperimentError, match="absent, stale, or rejected"):
        prepare_rotation_cache(disk_capped, metric_spaces=spaces, run_root=disk_cap_root)
    assert not (
        disk_cap_root / disk_capped.k_diagnostic.artifact_root / "pilot/rotation_cache_index.json"
    ).exists()


def test_random_prefix_is_nested_nonnegative_monotone_and_never_searches_vocab() -> None:
    norms = np.linspace(0.5, 2.0, 200)
    atoms, metadata = random_prefix_atoms(norms, k_max=12, seed=202, d_model=9)
    target = np.arange(1.0, 10.0)
    result = random_prefix_pursuit(atoms, target)

    assert metadata["statement"] == "This is not the full-cardinality random-search null."
    assert atoms.shape == (12, 9)  # only K_max atoms exist in this null
    for k in range(13):
        np.testing.assert_array_equal(result.support_prefixes[k], np.arange(k))
        assert result.coefficients_per_k[k].shape == (k,)
        assert np.all(result.coefficients_per_k[k] >= 0.0)
    assert np.all(np.diff(result.errors) <= 1e-12)


def test_transformed_dictionary_matches_dense_formula_and_preserves_raw_reconstruction() -> None:
    rng = np.random.default_rng(29)
    atoms = rng.normal(size=(17, 6))
    transform = rng.normal(size=(4, 6))
    residuals = rng.normal(size=(3, 4))
    wrapped = LinearTransformedDictionary(DenseDictionary(atoms), transform, norm_chunk_size=5)
    transformed = atoms @ transform.T

    np.testing.assert_allclose(wrapped.dots_batch(residuals), transformed @ residuals.T)
    np.testing.assert_allclose(wrapped.atom_norms(), np.linalg.norm(transformed, axis=1))
    ids = np.asarray([2, 7, 13])
    coefficients = np.asarray([0.4, 1.2, 0.3])
    np.testing.assert_allclose(
        wrapped.raw_reconstruction(ids, coefficients), coefficients @ atoms[ids]
    )


def test_group_safe_label_permutation_preserves_groups_counts_and_seed() -> None:
    labels = np.asarray([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    groups = np.asarray(["a", "a", "b", "b", "c", "d", "e", "e", "f", "f"])
    first = group_safe_permutation(labels, groups, seed=7001)
    second = group_safe_permutation(labels, groups, seed=7001)

    np.testing.assert_array_equal(first, second)
    assert first.sum() == labels.sum()
    for group in np.unique(groups):
        assert np.unique(first[groups == group]).size == 1


def test_fixed_c_permutation_probe_has_no_test_input_and_keeps_real_c() -> None:
    rng = np.random.default_rng(31)
    train_x = rng.normal(size=(40, 5))
    validation_x = rng.normal(size=(20, 5))
    train_y = np.tile([0, 1], 20)
    validation_y = np.tile([0, 1], 10)
    train_groups = np.asarray([f"train-{index}" for index in range(40)])
    validation_groups = np.asarray([f"validation-{index}" for index in range(20)])

    vector, metadata = fit_fixed_c_permutation_probe(
        train_x,
        train_y,
        train_groups,
        validation_x,
        validation_y,
        validation_groups,
        C=0.37,
        seed=7002,
    )
    assert vector.shape == (5,) and np.isfinite(vector).all()
    assert metadata["C"] == pytest.approx(0.37)
    assert metadata["test_labels_accessed"] is False
    assert metadata["fit_split"] == "permuted train+validation"
    assert metadata["standardize"] is True
    assert metadata["class_weight"] == "balanced"
    assert metadata["solver"] == "lbfgs"
    assert metadata["penalty"] == "l2"
    assert metadata["max_iter"] == 5000
    assert metadata["pipeline_parity_validated"] is True
    with pytest.raises(ValueError, match="parity"):
        fit_fixed_c_permutation_probe(
            train_x,
            train_y,
            train_groups,
            validation_x,
            validation_y,
            validation_groups,
            C=0.37,
            seed=7002,
            class_weight=None,
        )


def test_target_construction_contrast_span_positions_and_stable_id() -> None:
    # Two concepts, two templates, scalar residuals.
    values = np.asarray([[[3.0], [7.0]], [[1.0], [2.0]]])
    contrasts = paired_template_contrasts(values)
    np.testing.assert_allclose(contrasts[:, :, 0], [[2.0, 5.0], [-2.0, -5.0]])

    assert find_unique_subsequence([9, 2, 3, 8], [2, 3]) == (1, 3)
    with pytest.raises(TargetConstructionError, match="2 matches"):
        find_unique_subsequence([1, 2, 1, 2], [1, 2])

    positions = raw_activation_positions(
        [101, 5, 6, 7, 8, 0, 0],
        attention_mask=[1, 1, 1, 1, 1, 0, 0],
        special_tokens_mask=[1, 0, 0, 0, 0, 1, 1],
    )
    np.testing.assert_array_equal(positions, [1, 2, 3, 4])

    vector = np.asarray([1.0, 2.0, 3.0])
    first = TargetRecord(11, "raw_activation", "token_position", "row:7:3", vector)
    second = TargetRecord(11, "raw_activation", "token_position", "row:7:3", vector.copy())
    assert first.target_id == second.target_id
    assert first.vector_hash == second.vector_hash


def test_label_free_definitions_exclude_their_target_labels() -> None:
    assert len(FOUR_UNRELATED_TOPIC_DESCRIPTIONS) == 4
    for concept_id, descriptions in LABEL_FREE_PARAPHRASES.items():
        label = concept_id.split(":", 1)[1].replace("_", " ").casefold()
        assert len(descriptions) >= 4
        assert all(label not in description.casefold() for description in descriptions)


def test_class_mean_difference_uses_only_explicit_train_validation_rows() -> None:
    activations = np.asarray([[2.0, 0.0], [0.0, 0.0], [4.0, 2.0], [0.0, 2.0], [999.0, 999.0]])
    labels = np.asarray([1, 0, 1, 0, 1])
    result = class_mean_difference(
        activations,
        labels,
        train_indices=[0, 1],
        validation_indices=[2, 3],
    )
    np.testing.assert_allclose(result, [3.0, 0.0])


def test_shared_probe_metadata_validates_accuracy_hashes_and_pipeline(
    tmp_path: Path,
) -> None:
    concept_id = "goemotions:admiration"
    probe_root = tmp_path / "selection/raptor_probes"
    vector_path = probe_root / "layer_11/probe.npy"
    vector_path.parent.mkdir(parents=True)
    np.save(vector_path, np.asarray([1.0, 2.0, 3.0]))
    row_manifest = {
        "concepts": {
            concept_id: {
                "train": {"rows": [{"activation_index": 0}, {"activation_index": 1}]},
                "validation": {"rows": [{"activation_index": 2}, {"activation_index": 3}]},
            }
        }
    }
    row_path = tmp_path / "selection/row_manifest.json"
    row_path.parent.mkdir(parents=True, exist_ok=True)
    row_path.write_text(json.dumps(row_manifest), encoding="utf-8")
    metrics_path = probe_root / "layer_11/metrics.json"
    metrics = {
        "final_fit_split": "train+validation",
        "test_use": "held_out_report_only",
        "probe_vector_sha256": sha256_file(vector_path),
        "row_manifest_sha256": sha256_file(row_path),
        "standardize": True,
        "solver": "lbfgs",
        "penalty": "l2",
        "chosen_C": 0.1,
        "validation_accuracy": 0.75,
        "test_accuracy": 0.625,
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    (probe_root / "manifest.json").write_text(
        json.dumps(
            {
                "selection": {
                    "standardize": True,
                    "solver": "lbfgs",
                    "max_iter": 5000,
                },
                "probes": [
                    {
                        "layer": 11,
                        "concept_id": concept_id,
                        "vector_file": "layer_11/probe.npy",
                        "vector_sha256": sha256_file(vector_path),
                        "metrics_file": "layer_11/metrics.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    activation_root = tmp_path / "activations"
    activation_root.mkdir()
    (activation_root / "concepts.json").write_text(
        json.dumps({"concepts": [{"concept_id": concept_id, "column": 0}]}),
        encoding="utf-8",
    )
    np.save(activation_root / "labels.npy", np.asarray([[1], [0], [1], [0]]))
    np.save(
        activation_root / "layer_11.npy",
        np.asarray([[2.0, 0.0, 1.0], [0.0, 0.0, 0.0], [4.0, 2.0, 1.0], [0.0, 2.0, 0.0]]),
    )
    probe_settings = {
        "standardize": True,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "penalty": "l2",
        "max_iter": 5000,
    }
    records, permutation_metadata = prepare_shared_statistical_targets(
        shared_root=tmp_path,
        layers=[11],
        concept_ids=[concept_id],
        probe_settings=probe_settings,
    )
    logistic = next(record for record in records if record.target_family == "logistic_probe")
    assert logistic.metadata["chosen_C"] == 0.1
    assert logistic.metadata["validation_accuracy"] == 0.75
    assert logistic.metadata["test_accuracy"] == 0.625
    assert logistic.metadata["source_vector_sha256"] == sha256_file(vector_path)
    assert logistic.metadata["row_manifest_sha256"] == sha256_file(row_path)
    assert logistic.metadata["source_probe_manifest_sha256"] == sha256_file(
        probe_root / "manifest.json"
    )
    assert logistic.metadata["probe_pipeline"]["class_weight"] == "balanced"
    assert permutation_metadata[(11, concept_id)]["chosen_C"] == 0.1

    with pytest.raises(TargetConstructionError, match="preregistered pipeline"):
        prepare_shared_statistical_targets(
            shared_root=tmp_path,
            layers=[11],
            concept_ids=[concept_id],
            probe_settings=probe_settings | {"class_weight": None},
        )

    metrics["row_manifest_sha256"] = "0" * 64
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(TargetConstructionError, match="split/hash contract"):
        prepare_shared_statistical_targets(
            shared_root=tmp_path,
            layers=[11],
            concept_ids=[concept_id],
            probe_settings=probe_settings,
        )


def test_exact_sphere_max_quantile_matches_monte_carlo() -> None:
    dimension = 8
    cardinality = 32
    expected = sphere_max_quantile(dimension, cardinality, 0.5)
    rng = np.random.default_rng(37)
    samples = rng.normal(size=(20_000, cardinality, dimension))
    samples /= np.linalg.norm(samples, axis=2, keepdims=True)
    empirical = float(np.median(np.max(samples[:, :, 0], axis=1)))
    assert empirical == pytest.approx(expected, abs=0.015)


def test_registered_configs_and_pilot_grid_are_strict_and_unique() -> None:
    for stage in ("pilot", "raw_metric_full", "transformed_metric"):
        config = _diagnostic_config(stage)
        assert config.experiment_name == "qwen35_4b_k_diagnostic_v2"
        assert config.k_diagnostic is not None
        assert config.k_diagnostic.stage == stage
        if stage == "transformed_metric":
            source = (CONFIG_ROOT / "qwen35_4b_k_diagnostic_v2_transformed.yaml").read_text(
                encoding="utf-8"
            )
            assert "label_permutation_probe" not in source
            assert "permutation_seed_start" not in source
            assert "permutation_count_" not in source

    config = _diagnostic_config()
    targets: list[dict[str, object]] = []
    concepts = list(config.k_diagnostic.concept_ids)
    for layer in (11, 19, 27):
        for concept in concepts:
            for family in ("logistic_probe", "class_mean_difference"):
                targets.append(
                    {
                        "target_id": f"{family}:{layer}:{concept}",
                        "layer": layer,
                        "target_family": family,
                        "target_subtype": concept,
                    }
                )
            for arm in (
                "label_explicit",
                "label_free_definition",
                "four_unrelated_topics_control",
            ):
                end_position = (
                    "concept_token_end" if arm == "label_explicit" else "definition_span_end"
                )
                for position in ("assistant_boundary", end_position):
                    targets.append(
                        {
                            "target_id": f"contrast:{layer}:{concept}:{arm}:{position}",
                            "layer": layer,
                            "target_family": "seven_emotion_label_contrast",
                            "target_subtype": f"{arm}:{position}:{concept}",
                        }
                    )
        # Supply a strict superset; the pilot must still register exactly 32.
        for index in range(40):
            targets.append(
                {
                    "target_id": f"raw:{layer}:{index:03d}",
                    "layer": layer,
                    "target_family": "raw_activation",
                    "target_subtype": f"source:{index:03d}",
                    "metadata": {
                        "selection_rank": index,
                        "stage_membership": ["pilot", "raw_metric_full"],
                    },
                }
            )
    grid = build_scientific_grid(config, targets)
    identities = [json.dumps(shard.identity, sort_keys=True) for shard in grid]
    assert len(grid) == 66774
    assert len(build_physical_bundles(grid)) == 264
    assert len(set(identities)) == len(identities)
    assert {shard.null_family for shard in grid} == {
        "iid_full_cardinality",
        "iid_cardinality_sweep",
        "random_prefix_same_k",
        "orthogonal_rotation",
        "label_permutation_probe",
    }
    raw_ids = {shard.target_id for shard in grid if shard.target_family == "raw_activation"}
    assert len(raw_ids) == 96
    transformed = _diagnostic_config("transformed_metric")
    transformed_grid = build_scientific_grid(transformed, targets)
    assert len(transformed_grid) == 124_992
    assert len(build_physical_bundles(transformed_grid)) == 1_344
    assert {shard.null_family for shard in transformed_grid} == {
        "iid_full_cardinality",
        "random_prefix_same_k",
        "orthogonal_rotation",
    }


def test_target_store_is_append_only_and_vector_identity_is_immutable(tmp_path: Path) -> None:
    first = TargetRecord(11, "logistic_probe", "a", "a", np.asarray([1.0, 2.0]))
    second = TargetRecord(11, "class_mean_difference", "a", "a", np.asarray([2.0, 1.0]))
    initial = save_target_records([first], artifact_root=tmp_path)
    expanded = save_target_records([first, second], artifact_root=tmp_path)
    assert initial["target_count"] == 1
    assert expanded["target_count"] == 2
    conflicting = TargetRecord(11, "logistic_probe", "a", "a", np.asarray([9.0, 2.0]))
    with pytest.raises(TargetConstructionError, match=r"differs|collision"):
        save_target_records([conflicting], artifact_root=tmp_path)


def test_all_contrast_arms_positions_persist_per_template_vectors_and_resume(
    tmp_path: Path,
) -> None:
    config = _diagnostic_config()
    target_root = tmp_path / str(config.k_diagnostic.artifact_root)
    records = []
    for arm in (
        "label_explicit",
        "label_free_definition",
        "four_unrelated_topics_control",
    ):
        template_count = 12 if arm == "label_explicit" else 4
        end = "concept_token_end" if arm == "label_explicit" else "definition_span_end"
        for position in ("assistant_boundary", end):
            matrix = np.arange(template_count * 3, dtype=np.float64).reshape(template_count, 3)
            matrix[:, 0] += 1.0
            records.append(
                TargetRecord(
                    11,
                    "seven_emotion_label_contrast",
                    f"{arm}:{position}:goemotions:admiration",
                    "goemotions:admiration",
                    np.mean(matrix, axis=0),
                    {
                        "contrast_arm": arm,
                        "capture_position": position,
                        "template_count": template_count,
                    },
                    template_contrasts=matrix,
                )
            )
    first = save_target_records(records, artifact_root=target_root)
    second = save_target_records(records, artifact_root=target_root)
    assert first == second
    assert first["complete"] is True
    assert first["target_count"] == 6
    assert len(load_target_index(config, run_root=tmp_path)) == 6
    for entry in first["targets"]:
        payload = entry["per_template_contrasts"]
        path = target_root / payload["path"]
        values = np.load(path, allow_pickle=False)
        assert list(values.shape) == payload["shape"]
        assert values.dtype == np.float64
        assert sha256_file(path) == payload["file_sha256"]
        assert (
            hashlib.sha256(values.tobytes(order="C")).hexdigest() == payload["raw_float64_sha256"]
        )
    path = target_root / first["targets"][0]["per_template_contrasts"]["path"]
    tampered = np.load(path, allow_pickle=False)
    tampered[0, 0] += 1.0
    np.save(path, tampered)
    with pytest.raises(TargetConstructionError, match="per-template contrasts differ"):
        save_target_records(records, artifact_root=target_root)
    with pytest.raises(KDiagnosticExperimentError, match="per-template contrast file hash"):
        load_target_index(config, run_root=tmp_path)


def test_scientific_bundle_persists_required_files_resumes_and_indexes(
    tmp_path: Path,
) -> None:
    config = _diagnostic_config()
    shard = ScientificShard(
        target_id="synthetic-target",
        layer=11,
        target_family="logistic_probe",
        target_subtype="synthetic",
        metric="raw_euclidean",
        null_family="iid_full_cardinality",
        null_seed=101,
        dictionary_fraction=1.0,
    )
    rng = np.random.default_rng(41)
    dictionary = DenseDictionary(rng.normal(size=(96, 8)))
    target = rng.normal(size=8)
    bundle = PhysicalBundle(
        target_id=shard.target_id,
        layer=shard.layer,
        target_family=shard.target_family,
        target_subtype=shard.target_subtype,
        metric=shard.metric,
        shard_ids=(shard.shard_id,),
    )
    first = run_scientific_bundle(
        config=config,
        bundle=bundle,
        shards=[shard],
        dictionary=dictionary,
        target=target,
        token_decoder=_decode_token_ids,
        run_root=tmp_path,
    )
    second = run_scientific_bundle(
        config=config,
        bundle=bundle,
        shards=[shard],
        dictionary=dictionary,
        target=target,
        token_decoder=_decode_token_ids,
        run_root=tmp_path,
    )
    assert first == second
    shard_root = (
        tmp_path
        / "artifacts/concept_intervention/qwen35_4b_k_diagnostic_v2/pilot/shards"
        / shard.shard_id
    )
    assert all((shard_root / name).is_file() for name in REQUIRED_SHARD_FILES)
    assert not any(
        (shard_root / name).exists()
        for name in (
            "errors_real.npy",
            "gains_real.npy",
            "support_real.npy",
            "coefficients_real.npz",
        )
    )
    reference = json.loads((shard_root / "real_reference.json").read_text(encoding="utf-8"))
    assert reference["physical_bundle_id"] == bundle.bundle_id
    assert "fixed_budget_real.json" in reference["files"]
    assert not (shard_root / "fixed_budget_real.json").exists()
    fixed_path = shard_root.parent.parent / "bundles" / bundle.bundle_id / "fixed_budget_real.json"
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    assert set(fixed["budgets"]) == {
        "1",
        "2",
        "4",
        "8",
        "16",
        "25",
        "32",
        "64",
    }
    assert fixed["budgets"]["64"]["available"] is False
    for budget in (1, 2, 4, 8, 16, 25, 32):
        payload = fixed["budgets"][str(budget)]
        assert payload["available"] is True
        assert len(payload["selected_token_ids"]) == len(payload["decoded_tokens"])
        assert len(payload["selected_token_ids"]) == len(payload["coefficients"])
        assert payload["residual_norm"] >= 0.0
        assert payload["normalized_residual_norm"] >= 0.0
        assert "selected_atom_gram_condition_number" in payload
    _seal_v2_index_runtime_hardware(config, [shard], run_root=tmp_path)
    index = index_stage(config, [shard], run_root=tmp_path)
    assert index["complete"] is True
    assert index["completed_shards"] == 1
    manifest = json.loads((shard_root / "manifest.json").read_text(encoding="utf-8"))
    manifest["k_max"] = 31
    (shard_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(KDiagnosticExperimentError, match=r"identity|manifest"):
        run_scientific_bundle(
            config=config,
            bundle=bundle,
            shards=[shard],
            dictionary=dictionary,
            target=target,
            token_decoder=_decode_token_ids,
            run_root=tmp_path,
        )


def test_inverse_rotation_and_group_permutation_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        inverse_rotate_target(np.zeros(4), np.eye(4))

    labels = np.asarray([0, 0, 1, 1])
    same_label_strata = np.asarray(["negative", "negative", "positive", "positive"])
    with pytest.raises(ValueError, match="identity"):
        group_safe_permutation(labels, same_label_strata, seed=7)


def test_logistic_and_class_mean_permutation_paths_are_nonidentity() -> None:
    rng = np.random.default_rng(43)
    train_x = rng.normal(size=(24, 5))
    validation_x = rng.normal(size=(12, 5))
    train_y = np.tile([0, 1], 12)
    validation_y = np.tile([0, 1], 6)
    train_groups = np.asarray([f"train-{index}" for index in range(24)])
    validation_groups = np.asarray([f"validation-{index}" for index in range(12)])

    logistic, logistic_metadata = fit_fixed_c_permutation_probe(
        train_x,
        train_y,
        train_groups,
        validation_x,
        validation_y,
        validation_groups,
        C=0.5,
        seed=7011,
    )
    class_mean, class_mean_metadata = permuted_class_mean_difference(
        train_x,
        train_y,
        train_groups,
        validation_x,
        validation_y,
        validation_groups,
        seed=7011,
    )
    assert logistic.shape == class_mean.shape == (5,)
    for metadata in (logistic_metadata, class_mean_metadata):
        for split in ("train_permutation", "validation_permutation"):
            assert metadata[split]["hamming_distance"] > 0
            assert metadata[split]["non_identity_rate"] > 0.0
            assert metadata[split]["effective_permutable_groups"] > 0


def test_all_five_null_families_run_as_one_bundle_then_index(tmp_path: Path) -> None:
    config = _diagnostic_config().model_copy(
        update={
            "k_diagnostic": _diagnostic_config().k_diagnostic.model_copy(
                update={
                    "k_max": 4,
                    "report_grid": [1, 2, 4],
                    "rotation_seeds": [1101],
                }
            )
        }
    )
    rng = np.random.default_rng(47)
    target = rng.normal(size=8)
    target_record = TargetRecord(
        layer=11,
        target_family="class_mean_difference",
        target_subtype="synthetic",
        source_id="all-nulls",
        vector=target,
    )
    save_target_records(
        [target_record],
        artifact_root=tmp_path / config.k_diagnostic.artifact_root,
    )
    shards = [
        ScientificShard(
            target_id=target_record.target_id,
            layer=11,
            target_family="class_mean_difference",
            target_subtype="synthetic",
            metric="raw_euclidean",
            null_family=null_family,
            null_seed=seed,
            dictionary_fraction=fraction,
        )
        for null_family, seed, fraction in (
            ("iid_full_cardinality", 101, 1.0),
            ("iid_cardinality_sweep", 102, 0.5),
            ("random_prefix_same_k", 201, 1.0),
            ("orthogonal_rotation", 1101, 1.0),
            ("label_permutation_probe", 7001, 1.0),
        )
    ]
    bundle = build_physical_bundles(shards)[0]
    dictionary = DenseDictionary(rng.normal(size=(96, 8)))
    permutation_metadata = {
        split: {
            "hamming_distance": 8,
            "non_identity_rate": 0.5,
            "effective_permutable_groups": 8,
        }
        for split in ("train_permutation", "validation_permutation")
    }
    permutation_shard = next(
        shard for shard in shards if shard.null_family == "label_permutation_probe"
    )
    prepare_rotation_cache(
        config,
        metric_spaces=[
            {
                "metric": "raw_euclidean",
                "transform_sha256": None,
                "dimension": 8,
                "layers": [11],
            }
        ],
        run_root=tmp_path,
        require_build_preflight=False,
    )
    run_scientific_bundle(
        config=config,
        bundle=bundle,
        shards=shards,
        dictionary=dictionary,
        target=target,
        token_decoder=_decode_token_ids,
        run_root=tmp_path,
        permutation_targets={
            permutation_shard.shard_id: (rng.normal(size=8), permutation_metadata)
        },
    )
    _seal_v2_index_runtime_hardware(config, shards, run_root=tmp_path)
    index = index_stage(config, shards, run_root=tmp_path)
    assert index["complete"] is True
    assert index["physical_bundle_count"] == 1
    assert index["logical_replicate_count"] == 5
    assert set(index["null_families"]) == {shard.null_family for shard in shards}


def test_rotation_binding_audit_rejects_stale_space_hash_and_missing_shard_key(
    tmp_path: Path,
) -> None:
    source = _diagnostic_config()
    config = source.model_copy(
        update={
            "k_diagnostic": source.k_diagnostic.model_copy(
                update={
                    "k_max": 4,
                    "report_grid": [1, 2, 4],
                    "rotation_seeds": [1101],
                }
            )
        }
    )
    rng = np.random.default_rng(4801)
    target = rng.normal(size=4)
    record = TargetRecord(
        layer=11,
        target_family="class_mean_difference",
        target_subtype="synthetic",
        source_id="rotation-binding",
        vector=target,
    )
    save_target_records([record], artifact_root=tmp_path / config.k_diagnostic.artifact_root)
    shard = ScientificShard(
        target_id=record.target_id,
        layer=11,
        target_family=record.target_family,
        target_subtype=record.target_subtype,
        metric="raw_euclidean",
        null_family="orthogonal_rotation",
        null_seed=1101,
        dictionary_fraction=1.0,
    )
    spaces = [
        {
            "metric": "raw_euclidean",
            "transform_sha256": None,
            "dimension": 4,
            "layers": [11],
        }
    ]
    rotation_cache_build_preflight(config, metric_spaces=spaces, run_root=tmp_path)
    prepare_rotation_cache(config, metric_spaces=spaces, run_root=tmp_path)
    run_scientific_bundle(
        config=config,
        bundle=build_physical_bundles([shard])[0],
        shards=[shard],
        dictionary=DenseDictionary(rng.normal(size=(48, 4))),
        target=target,
        token_decoder=_decode_token_ids,
        run_root=tmp_path,
    )
    assert audit_rotation_cache_for_grid(config, [shard], run_root=tmp_path)["complete"]

    stage = tmp_path / config.k_diagnostic.artifact_root / "pilot"
    index_path = stage / "rotation_cache_index.json"
    original_index = json.loads(index_path.read_text(encoding="utf-8"))
    for field, value in (
        ("transform_sha256", "a" * 64),
        ("dimension", 5),
        ("matrix_sha256", "0" * 64),
    ):
        tampered = json.loads(json.dumps(original_index))
        tampered["entries"][0][field] = value
        index_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(KDiagnosticExperimentError):
            audit_rotation_cache_for_grid(config, [shard], run_root=tmp_path)
    index_path.write_text(json.dumps(original_index), encoding="utf-8")

    shard_manifest_path = stage / "shards" / shard.shard_id / "manifest.json"
    shard_manifest = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
    shard_manifest["transform_sha256"] = "f" * 64
    shard_manifest_path.write_text(json.dumps(shard_manifest), encoding="utf-8")
    _seal_v2_index_runtime_hardware(config, [shard], run_root=tmp_path)
    with pytest.raises(
        KDiagnosticExperimentError,
        match="rotation shard key is absent from the audited binding index",
    ):
        index_stage(config, [shard], run_root=tmp_path)


def test_bundle_partial_write_and_concurrent_lock_fail_closed(tmp_path: Path) -> None:
    source = _diagnostic_config()
    config = source.model_copy(
        update={
            "k_diagnostic": source.k_diagnostic.model_copy(
                update={"k_max": 4, "report_grid": [1, 2, 4]}
            )
        }
    )
    shard = ScientificShard(
        target_id="bundle-failure",
        layer=11,
        target_family="logistic_probe",
        target_subtype="synthetic",
        metric="raw_euclidean",
        null_family="iid_full_cardinality",
        null_seed=101,
        dictionary_fraction=1.0,
    )
    bundle = build_physical_bundles([shard])[0]
    rng = np.random.default_rng(49)
    dictionary = DenseDictionary(rng.normal(size=(96, 8)))
    target = rng.normal(size=8)
    relative_stage = Path("artifacts/concept_intervention/qwen35_4b_k_diagnostic_v2/pilot")

    partial_root = tmp_path / "partial"
    (partial_root / relative_stage / "bundles" / bundle.bundle_id).mkdir(parents=True)
    with pytest.raises(KDiagnosticExperimentError, match="partial bundle"):
        run_scientific_bundle(
            config=config,
            bundle=bundle,
            shards=[shard],
            dictionary=dictionary,
            target=target,
            token_decoder=_decode_token_ids,
            run_root=partial_root,
        )

    locked_root = tmp_path / "locked"
    lock = locked_root / relative_stage / "bundle_locks" / f"{bundle.bundle_id}.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("synthetic competing worker", encoding="utf-8")
    with pytest.raises(KDiagnosticExperimentError, match="already locked"):
        run_scientific_bundle(
            config=config,
            bundle=bundle,
            shards=[shard],
            dictionary=dictionary,
            target=target,
            token_decoder=_decode_token_ids,
            run_root=locked_root,
        )

    tampered_root = tmp_path / "tampered"
    run_scientific_bundle(
        config=config,
        bundle=bundle,
        shards=[shard],
        dictionary=dictionary,
        target=target,
        token_decoder=_decode_token_ids,
        run_root=tampered_root,
    )
    gains = tampered_root / relative_stage / "bundles" / bundle.bundle_id / "gains_real.npy"
    with gains.open("r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([byte[0] ^ 1]))
    with pytest.raises(KDiagnosticExperimentError, match="real payload hash mismatch"):
        run_scientific_bundle(
            config=config,
            bundle=bundle,
            shards=[shard],
            dictionary=dictionary,
            target=target,
            token_decoder=_decode_token_ids,
            run_root=tampered_root,
        )


def test_transformed_bundle_writes_and_audits_raw_space_reconstructions(
    tmp_path: Path,
) -> None:
    source = _diagnostic_config("transformed_metric")
    config = source.model_copy(
        update={
            "k_diagnostic": source.k_diagnostic.model_copy(
                update={"k_max": 4, "report_grid": [1, 2, 4]}
            )
        }
    )
    shard = ScientificShard(
        target_id="transformed-target",
        layer=11,
        target_family="logistic_probe",
        target_subtype="synthetic",
        metric="j_pca_r4",
        null_family="iid_full_cardinality",
        null_seed=101,
        dictionary_fraction=1.0,
    )
    rng = np.random.default_rng(53)
    raw_dictionary = DenseDictionary(rng.normal(size=(96, 6)))
    transformed = LinearTransformedDictionary(raw_dictionary, rng.normal(size=(4, 6)))
    raw_target = rng.normal(size=6)
    bundle = build_physical_bundles([shard])[0]
    run_scientific_bundle(
        config=config,
        bundle=bundle,
        shards=[shard],
        dictionary=transformed,
        target=transformed.transform_target(raw_target),
        raw_target=raw_target,
        token_decoder=_decode_token_ids,
        run_root=tmp_path,
    )
    _seal_v2_index_runtime_hardware(config, [shard], run_root=tmp_path)
    index = index_stage(config, [shard], run_root=tmp_path)
    assert index["complete"] is True
    shard_root = (
        tmp_path
        / "artifacts/concept_intervention/qwen35_4b_k_diagnostic_v2/transformed_metric/shards"
        / shard.shard_id
    )
    reference = json.loads((shard_root / "real_reference.json").read_text(encoding="utf-8"))
    bundle_root = shard_root.parent.parent / "bundles" / reference["physical_bundle_id"]
    assert not (shard_root / "raw_reconstructions_real.npz").exists()
    with np.load(bundle_root / "raw_reconstructions_real.npz") as archive:
        assert archive.files == ["k_001", "k_002", "k_004"]
        assert all(archive[name].shape == (6,) for name in archive.files)
    fixed = json.loads((bundle_root / "fixed_budget_real.json").read_text(encoding="utf-8"))
    for k in (1, 2, 4):
        budget = fixed["budgets"][str(k)]
        assert "metric_space_error" in budget
        assert "metric_space_explained_fraction" in budget
        assert "raw_space_error" in budget
        assert "raw_space_explained_fraction" in budget
    summary = json.loads((shard_root / "summary.json").read_text(encoding="utf-8"))
    assert all("raw_space_error" in summary["fixed_budget"][str(k)] for k in (1, 2, 4))


def test_sealed_pilot_membership_is_unchanged_after_full_store_append(
    tmp_path: Path,
) -> None:
    config = _diagnostic_config()
    targets: list[dict[str, object]] = []
    for layer in (11, 19, 27):
        for concept in config.k_diagnostic.concept_ids:
            for family in ("logistic_probe", "class_mean_difference"):
                targets.append(
                    {
                        "target_id": f"{family}:{layer}:{concept}",
                        "file_sha256": "a" * 64,
                        "layer": layer,
                        "target_family": family,
                        "target_subtype": concept,
                    }
                )
            for arm in (
                "label_explicit",
                "label_free_definition",
                "four_unrelated_topics_control",
            ):
                end_position = (
                    "concept_token_end" if arm == "label_explicit" else "definition_span_end"
                )
                for position in ("assistant_boundary", end_position):
                    targets.append(
                        {
                            "target_id": f"contrast:{layer}:{concept}:{arm}:{position}",
                            "file_sha256": "b" * 64,
                            "layer": layer,
                            "target_family": "seven_emotion_label_contrast",
                            "target_subtype": f"{arm}:{position}:{concept}",
                        }
                    )
        for rank in range(32):
            targets.append(
                {
                    "target_id": f"raw:{layer}:{rank:03d}",
                    "file_sha256": "c" * 64,
                    "layer": layer,
                    "target_family": "raw_activation",
                    "target_subtype": f"source:{rank:03d}",
                    "metadata": {
                        "selection_rank": rank,
                        "stage_membership": ["pilot", "raw_metric_full"],
                    },
                }
            )
    first_grid = [shard.identity for shard in build_scientific_grid(config, targets)]
    first_manifest = seal_target_membership(config, targets, run_root=tmp_path)

    expanded = list(targets)
    for layer in (11, 19, 27):
        for rank in range(32, 128):
            expanded.append(
                {
                    "target_id": f"raw:{layer}:{rank:03d}",
                    "file_sha256": "d" * 64,
                    "layer": layer,
                    "target_family": "raw_activation",
                    "target_subtype": f"source:{rank:03d}",
                    "metadata": {
                        "selection_rank": rank,
                        "stage_membership": ["raw_metric_full"],
                    },
                }
            )
    second_manifest = seal_target_membership(config, expanded, run_root=tmp_path)
    second_grid = [shard.identity for shard in build_scientific_grid(config, expanded)]
    assert second_manifest == first_manifest
    assert second_grid == first_grid


def test_censoring_bootstrap_and_registered_decision_thresholds() -> None:
    crossing = observed_crossing_summary([2, None, 4, None], [False, True, False, True], k_max=4)
    assert crossing["observed_crossing_median"] == 3.0
    assert crossing["right_censored_fraction"] == 0.5
    assert crossing["population_median_reported"] is False

    rows = [
        {
            "layer": layer,
            "source_id": source,
            "resampling_unit": template,
            "target_id": f"{layer}:{source}:{template}",
            "effect": value,
        }
        for layer in (11, 19)
        for source, value in (("anger", 0.03), ("joy", 0.04))
        for template in ("definition-1", "definition-2")
    ]
    estimate = hierarchical_bootstrap_mean(rows, value_key="effect", samples=1000, seed=8844)
    assert estimate["target_level_n"] == 8
    assert estimate["resampling_hierarchy"] == [
        "layer",
        "concept_or_source",
        "template_or_target",
    ]
    assert (
        classify_registered_effect(
            ci_low=0.02,
            ci_high=0.05,
            minimum_effect=0.01,
            equivalence_margin=0.0025,
            adjusted_p_value=0.01,
        )
        == "Supported"
    )
    assert (
        classify_registered_effect(
            ci_low=-0.01,
            ci_high=0.002,
            minimum_effect=0.01,
            equivalence_margin=0.0025,
            adjusted_p_value=0.9,
        )
        == "Not supported"
    )


def test_aggregate_crossing_reports_supported_atoms_and_right_censoring() -> None:
    crossed = summarize_null_calibrated_curve(
        [1.0, 0.6, 0.3, 0.2, 0.11, 0.03],
        [[1.0, 0.8, 0.6, 0.4, 0.2, 0.0]],
        report_grid=[1, 4],
    )
    # The first non-exceeding gain is 1-indexed k=3, so supported atoms are k-1=2.
    assert crossed["K_first"] == 2
    assert crossed["K_consecutive3"] == 2
    assert crossed["K_first_right_censored"] is False

    production = summarize_null_calibrated_curve(
        [1.0, 0.6, 0.3, 0.2, 0.11, 0.03],
        np.tile([1.0, 0.8, 0.6, 0.4, 0.2, 0.0], (31, 1)),
        report_grid=[1, 4],
    )
    empirical = production["empirical_diagnostics"]
    assert empirical["null_seed_count"] == 31
    assert empirical["resolution"] == pytest.approx(1 / 32)
    assert len(empirical["marginal_gain_by_k"]) == 5
    assert empirical["fixed_k4"]["empirical_upper_tail_p_value"] >= 1 / 32
    assert empirical["cumulative_benefit"]["empirical_upper_tail_p_value"] >= 1 / 32

    censored = summarize_null_calibrated_curve(
        [1.0, 0.8, 0.6, 0.4, 0.2],
        [[1.0, 0.99, 0.98, 0.97, 0.96]],
        report_grid=[1, 4],
    )
    assert censored["K_first"] is None
    assert censored["K_first_right_censored"] is True


def test_production_summary_rejects_prefix_cell_with_fewer_than_31_seeds(
    tmp_path: Path,
) -> None:
    root = tmp_path / "diagnostic"
    stage = root / "raw_metric_full"
    stage.mkdir(parents=True)
    seeds = list(range(101, 131))
    identity = {
        "target_id": "prefix-target",
        "layer": 11,
        "target_family": "logistic_probe",
        "target_subtype": "anger",
        "metric": "raw_euclidean",
        "null_family": "random_prefix_same_k",
        "dictionary_fraction": 1.0,
    }
    (stage / "grid.json").write_text(
        json.dumps(
            {
                "shards": [identity | {"null_seed": seed} for seed in seeds],
            }
        ),
        encoding="utf-8",
    )
    membership = root / "target_membership"
    membership.mkdir()
    (membership / "raw_metric_full.json").write_text(
        json.dumps({"targets": [{"target_id": "prefix-target"}]}),
        encoding="utf-8",
    )
    row = identity | {
        "null_seeds": seeds,
        "null_seed_count": len(seeds),
        "K_first": 2,
        "K_first_right_censored": False,
        "K_consecutive3": 3,
        "K_consecutive3_right_censored": False,
        "max_cumulative_excess": 0.03,
        "cumulative_excess": [0.01, 0.02, 0.03, 0.025],
        "fixed_budget": {"4": {"real_minus_null": 0.03}},
        "empirical_diagnostics": _synthetic_empirical(30),
    }
    with pytest.raises(KDiagnosticReportingError, match="only 30 replicates"):
        _validate_stage_summary(
            root,
            "raw_metric_full",
            [row],
            minimum_null_replicates=31,
        )

    seeds.append(131)
    (stage / "grid.json").write_text(
        json.dumps({"shards": [identity | {"null_seed": seed} for seed in seeds]}),
        encoding="utf-8",
    )
    production_row = row | {
        "null_seeds": seeds,
        "null_seed_count": 31,
        "empirical_diagnostics": _synthetic_empirical(31),
    }
    _validate_stage_summary(
        root,
        "raw_metric_full",
        [production_row],
        minimum_null_replicates=31,
    )
    invalid_empirical = json.loads(json.dumps(production_row))
    invalid_empirical["empirical_diagnostics"]["fixed_k4"]["empirical_upper_tail_p_value"] = 0.0
    with pytest.raises(KDiagnosticReportingError, match="p-value/percentile"):
        _validate_stage_summary(
            root,
            "raw_metric_full",
            [invalid_empirical],
            minimum_null_replicates=31,
        )


def test_figure_four_collapses_iid_seeds_within_target_first(tmp_path: Path) -> None:
    root = tmp_path / "diagnostic"
    shard_root = root / "raw_metric_full/shards"
    bundle_root = root / "raw_metric_full/bundles"
    expected_nulls = []
    for target_index, seed_curves in enumerate(
        (
            ([0.1, 0.2], [0.3, 0.4]),
            ([0.5, 0.6], [0.7, 0.8]),
        )
    ):
        bundle_id = f"bundle-{target_index}"
        directory = bundle_root / bundle_id
        directory.mkdir(parents=True)
        np.save(directory / "gains_real.npy", np.asarray([0.9, 0.8]))
        digest = sha256_file(directory / "gains_real.npy")
        expected_nulls.append(np.median(np.asarray(seed_curves), axis=0))
        for seed_index, null_curve in enumerate(seed_curves):
            shard = shard_root / f"{target_index}-{seed_index}"
            shard.mkdir(parents=True)
            (shard / "summary.json").write_text(
                json.dumps(
                    {
                        "target_id": f"target-{target_index}",
                        "layer": 11,
                        "metric": "raw_euclidean",
                        "null_family": "iid_full_cardinality",
                    }
                ),
                encoding="utf-8",
            )
            (shard / "real_reference.json").write_text(
                json.dumps(
                    {
                        "physical_bundle_id": bundle_id,
                        "files": {"gains_real.npy": digest},
                    }
                ),
                encoding="utf-8",
            )
            np.save(shard / "null_gains.npy", np.asarray([null_curve]))
    real, null = _within_target_iid_gain_curves(root)
    assert len(real) == len(null) == 2
    for observed, expected in zip(null, expected_nulls, strict=True):
        np.testing.assert_allclose(observed, expected)


def _small_v2_config():
    source = _diagnostic_config()
    return source.model_copy(
        update={
            "k_diagnostic": source.k_diagnostic.model_copy(
                update={"k_max": 4, "report_grid": [1, 2, 4]}
            )
        }
    )


@pytest.mark.parametrize("seed", [101, 102, 103])
def test_shared_direction_provider_is_strict_legacy_parity_and_generated_once(
    seed: int,
) -> None:
    rng = np.random.default_rng(1601 + seed)
    norms = np.abs(rng.normal(size=96)) + 0.1
    residuals = rng.normal(size=(3, 7))
    target = rng.normal(size=7)
    chunk_size = 13
    with SharedDirectionProvider(
        n_atoms=norms.size,
        d_model=target.size,
        seed=seed,
        chunk_size=chunk_size,
        available_bytes=16 * 2**30,
    ) as provider:
        for fraction in (0.25, 0.5, 1.0):
            selected, _metadata = matched_norm_subset(norms, fraction=fraction, seed=seed)
            legacy = MatchedNormRandomDictionary(
                selected,
                seed=seed,
                d_model=target.size,
                chunk_size=chunk_size,
            )
            shared = SharedDirectionRandomDictionary(provider, selected)
            np.testing.assert_array_equal(shared.atom_norms(), legacy.atom_norms())
            np.testing.assert_array_equal(
                shared.dots_batch(residuals), legacy.dots_batch(residuals)
            )
            ids = np.asarray([0, 3, selected.size - 1], dtype=np.int64)
            np.testing.assert_array_equal(shared.materialize(ids), legacy.materialize(ids))
            scalar = streaming_nonnegative_pursuit(
                legacy,
                target[None, :],
                k_max=4,
                solver_method="nonnegative_gradient_pursuit_standard",
            )[0]
            reused = streaming_nonnegative_pursuit(
                shared,
                target[None, :],
                k_max=4,
                solver_method="nonnegative_gradient_pursuit_standard",
            )[0]
            np.testing.assert_array_equal(reused.support, scalar.support)
            np.testing.assert_array_equal(reused.errors, scalar.errors)
            np.testing.assert_array_equal(reused.gains, scalar.gains)
            for observed, expected in zip(
                reused.coefficients_per_k,
                scalar.coefficients_per_k,
                strict=True,
            ):
                np.testing.assert_array_equal(observed, expected)
        assert provider.generation_count == int(np.ceil(norms.size / chunk_size))
        assert set(provider.generation_counts.values()) == {1}


def test_shared_direction_memory_gate_is_fail_closed() -> None:
    too_large = shared_direction_memory_plan(
        n_atoms=96,
        d_model=8,
        chunk_size=16,
        max_peak_bytes=100,
        available_bytes=16 * 2**30,
    )
    assert too_large["approved"] is False
    assert too_large["checks"]["within_hard_peak_budget"] is False
    with pytest.raises(ValueError, match="RAM preflight rejected"):
        SharedDirectionProvider(
            n_atoms=96,
            d_model=8,
            seed=101,
            chunk_size=16,
            max_peak_bytes=100,
            available_bytes=16 * 2**30,
        )

    valid = shared_direction_memory_plan(
        n_atoms=96,
        d_model=8,
        chunk_size=16,
        available_bytes=16 * 2**30,
    )
    reserve_failure = shared_direction_memory_plan(
        n_atoms=96,
        d_model=8,
        chunk_size=16,
        available_bytes=(
            SHARED_DIRECTION_MIN_AVAILABLE_RESERVE_BYTES + int(valid["peak_bytes"]) - 1
        ),
    )
    assert reserve_failure["approved"] is False
    assert reserve_failure["checks"]["preserves_available_ram_reserve"] is False


@pytest.mark.parametrize("gate_case", ["approved", "cap", "reserve"])
def test_production_shared_direction_plan_fits_six_gib_without_allocation(
    gate_case: str,
) -> None:
    n_atoms = 248_077
    d_model = 2_560
    chunk_size = 4_096
    expected_storage = 5_080_616_960
    expected_workspace = 167_804_928
    expected_peak = 5_248_421_888
    max_peak_bytes = expected_peak - 1 if gate_case == "cap" else SHARED_DIRECTION_MAX_PEAK_BYTES
    available_bytes = (
        SHARED_DIRECTION_MIN_AVAILABLE_RESERVE_BYTES + expected_peak - 1
        if gate_case == "reserve"
        else 64 * 2**30
    )

    plan = shared_direction_memory_plan(
        n_atoms=n_atoms,
        d_model=d_model,
        chunk_size=chunk_size,
        max_peak_bytes=max_peak_bytes,
        available_bytes=available_bytes,
    )
    assert SHARED_DIRECTION_MAX_PEAK_BYTES == 6 * 2**30
    assert plan["float64_direction_storage_bytes"] == expected_storage
    assert plan["generation_workspace_bytes"] == expected_workspace
    assert plan["peak_bytes"] == expected_peak
    assert plan["reserved_available_bytes"] == 8 * 2**30
    if gate_case == "approved":
        assert plan["hard_peak_budget_bytes"] == 6 * 2**30
        assert plan["approved"] is True
    elif gate_case == "cap":
        assert plan["hard_peak_budget_bytes"] == expected_peak - 1
        assert plan["approved"] is False
        assert plan["checks"]["within_hard_peak_budget"] is False
    else:
        assert plan["approved"] is False
        assert plan["checks"]["preserves_available_ram_reserve"] is False


def test_batched_rotation_and_permutation_targets_match_31_scalar_scans() -> None:
    rng = np.random.default_rng(1602)
    base = DenseDictionary(rng.normal(size=(96, 8)))
    target = rng.normal(size=8)
    rotation_targets = []
    for seed in range(1101, 1132):
        rotation, _metadata = haar_orthogonal(8, seed=seed)
        rotated, _error = inverse_rotate_target(target, rotation)
        rotation_targets.append(rotated)
    permutation_targets = [rng.normal(size=8) for _ in range(31)]

    class CountingDictionary:
        def __init__(self, source):
            self.source = source
            self.n_atoms = source.n_atoms
            self.d_model = source.d_model
            self.full_scans = 0

        def atom_norms(self):
            return self.source.atom_norms()

        def dots_batch(self, residuals):
            self.full_scans += 1
            return self.source.dots_batch(residuals)

        def materialize(self, ids):
            return self.source.materialize(ids)

    for targets in (rotation_targets, permutation_targets):
        stacked = np.stack(targets)
        batched_dictionary = CountingDictionary(base)
        batched = streaming_nonnegative_pursuit(
            batched_dictionary,
            stacked,
            k_max=4,
            solver_method="nonnegative_gradient_pursuit_standard",
        )
        scalar_dictionary = CountingDictionary(base)
        scalar = [
            streaming_nonnegative_pursuit(
                scalar_dictionary,
                row[None, :],
                k_max=4,
                solver_method="nonnegative_gradient_pursuit_standard",
            )[0]
            for row in stacked
        ]
        assert batched_dictionary.full_scans == 4
        assert scalar_dictionary.full_scans == 31 * 4
        for observed, expected in zip(batched, scalar, strict=True):
            np.testing.assert_array_equal(observed.support, expected.support)
            np.testing.assert_allclose(observed.errors, expected.errors, atol=1e-14)
            np.testing.assert_allclose(observed.gains, expected.gains, atol=1e-14)
            for observed_coefficients, expected_coefficients in zip(
                observed.coefficients_per_k,
                expected.coefficients_per_k,
                strict=True,
            ):
                np.testing.assert_allclose(observed_coefficients, expected_coefficients, atol=1e-14)


def test_v2_iid_full_fraction_one_aliases_one_solve_and_audits_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_v2_config()
    shards = [
        ScientificShard(
            target_id="iid-alias",
            layer=11,
            target_family="logistic_probe",
            target_subtype="synthetic",
            metric="raw_euclidean",
            null_family=family,
            null_seed=101,
            dictionary_fraction=1.0,
        )
        for family in ("iid_full_cardinality", "iid_cardinality_sweep")
    ]
    bundle = build_physical_bundles(shards)[0]
    rng = np.random.default_rng(1603)
    dictionary = DenseDictionary(rng.normal(size=(96, 8)))
    target = rng.normal(size=8)
    calls: list[int] = []
    original = kdiag_experiment.streaming_nonnegative_pursuit

    def counted(dictionary_arg, targets, **kwargs):
        calls.append(int(np.asarray(targets).shape[0]))
        return original(dictionary_arg, targets, **kwargs)

    monkeypatch.setattr(kdiag_experiment, "streaming_nonnegative_pursuit", counted)
    run_scientific_bundle(
        config=config,
        bundle=bundle,
        shards=shards,
        dictionary=dictionary,
        target=target,
        token_decoder=_decode_token_ids,
        run_root=tmp_path,
    )
    assert calls == [1, 1]
    stage = tmp_path / config.k_diagnostic.artifact_root / "pilot"
    manifests = [
        json.loads(
            (stage / "shards" / shard.shard_id / "manifest.json").read_text(encoding="utf-8")
        )
        for shard in shards
    ]
    assert shards[0].shard_id != shards[1].shard_id
    for field in (
        "shared_null_computation_id",
        "source_curve_sha256",
        "null_batch_computation_id",
    ):
        assert manifests[0][field] == manifests[1][field]
    _seal_v2_index_runtime_hardware(config, shards, run_root=tmp_path)
    index = index_stage(config, shards, run_root=tmp_path)
    assert index["null_execution_version"] == NULL_EXECUTION_VERSION

    runtime_path = stage / "runtimes" / f"{bundle.bundle_id}.json"
    original_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    tampered_runtime = json.loads(json.dumps(original_runtime))
    tampered_runtime["execution_hardware_sha256"] = "0" * 64
    runtime_path.write_text(json.dumps(tampered_runtime), encoding="utf-8")
    with pytest.raises(KDiagnosticExperimentError, match="runtime hardware audit"):
        index_stage(config, shards, run_root=tmp_path)
    runtime_path.write_text(json.dumps(original_runtime), encoding="utf-8")

    preflight_path = stage / "resource_preflight.json"
    original_preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    tampered_preflight = json.loads(json.dumps(original_preflight))
    tampered_preflight["execution_hardware"]["gpu_names"] = ["Different scheduled GPU"]
    preflight_path.write_text(json.dumps(tampered_preflight), encoding="utf-8")
    with pytest.raises(KDiagnosticExperimentError, match="preflight hardware identity"):
        index_stage(config, shards, run_root=tmp_path)
    preflight_path.write_text(json.dumps(original_preflight), encoding="utf-8")

    metadata_path = stage / "shards" / shards[0].shard_id / "null_metadata.json"
    original_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = json.loads(json.dumps(original_metadata))
    metadata["source_curve_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(KDiagnosticExperimentError, match="shared-null"):
        index_stage(config, shards, run_root=tmp_path)
    metadata_path.write_text(json.dumps(original_metadata), encoding="utf-8")

    manifest_path = stage / "shards" / shards[0].shard_id / "manifest.json"
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = json.loads(json.dumps(original_manifest))
    manifest["null_batch_computation_id"] = "tampered-batch-id"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(KDiagnosticExperimentError, match="shared-null"):
        index_stage(config, shards, run_root=tmp_path)
    manifest_path.write_text(json.dumps(original_manifest), encoding="utf-8")

    mixed = json.loads(json.dumps(original_manifest))
    mixed["null_execution_version"] = "legacy_scalar_execution"
    manifest_path.write_text(json.dumps(mixed), encoding="utf-8")
    with pytest.raises(KDiagnosticExperimentError, match="null_execution_version"):
        index_stage(config, shards, run_root=tmp_path)


def test_v2_bundle_batches_rotation_and_permutation_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _small_v2_config()
    config = source.model_copy(
        update={
            "k_diagnostic": source.k_diagnostic.model_copy(update={"rotation_seeds": [1101, 1102]})
        }
    )
    common = {
        "target_id": "batch-orchestration",
        "layer": 11,
        "target_family": "class_mean_difference",
        "target_subtype": "synthetic",
        "metric": "raw_euclidean",
        "dictionary_fraction": 1.0,
    }
    shards = [
        ScientificShard(
            **common,
            null_family="orthogonal_rotation",
            null_seed=seed,
        )
        for seed in (1101, 1102)
    ] + [
        ScientificShard(
            **common,
            null_family="label_permutation_probe",
            null_seed=seed,
        )
        for seed in (7001, 7002)
    ]
    spaces = [
        {
            "metric": "raw_euclidean",
            "transform_sha256": None,
            "dimension": 8,
            "layers": [11],
        }
    ]
    prepare_rotation_cache(
        config,
        metric_spaces=spaces,
        run_root=tmp_path,
        require_build_preflight=False,
    )
    rng = np.random.default_rng(1604)
    dictionary = DenseDictionary(rng.normal(size=(96, 8)))
    target = rng.normal(size=8)
    permutation_metadata = {
        split: {
            "hamming_distance": 8,
            "non_identity_rate": 0.5,
            "effective_permutable_groups": 8,
        }
        for split in ("train_permutation", "validation_permutation")
    }
    permutation_targets = {
        shard.shard_id: (rng.normal(size=8), permutation_metadata)
        for shard in shards
        if shard.null_family == "label_permutation_probe"
    }
    calls: list[int] = []
    original = kdiag_experiment.streaming_nonnegative_pursuit

    def counted(dictionary_arg, targets, **kwargs):
        calls.append(int(np.asarray(targets).shape[0]))
        return original(dictionary_arg, targets, **kwargs)

    monkeypatch.setattr(kdiag_experiment, "streaming_nonnegative_pursuit", counted)
    run_scientific_bundle(
        config=config,
        bundle=build_physical_bundles(shards)[0],
        shards=shards,
        dictionary=dictionary,
        target=target,
        permutation_targets=permutation_targets,
        token_decoder=_decode_token_ids,
        run_root=tmp_path,
    )
    assert calls == [1, 2, 2]


@pytest.mark.parametrize("failure_name", REQUIRED_SHARD_FILES)
def test_v2_shard_commit_is_atomic_and_rerun_reuses_real(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_name: str,
) -> None:
    config = _small_v2_config()
    shard = ScientificShard(
        target_id=f"atomic-{failure_name}",
        layer=11,
        target_family="logistic_probe",
        target_subtype="synthetic",
        metric="raw_euclidean",
        null_family="iid_full_cardinality",
        null_seed=101,
        dictionary_fraction=1.0,
    )
    bundle = build_physical_bundles([shard])[0]
    rng = np.random.default_rng(1605)
    dictionary = DenseDictionary(rng.normal(size=(96, 8)))
    target = rng.normal(size=8)
    calls: list[int] = []
    original_solver = kdiag_experiment.streaming_nonnegative_pursuit
    original_json = kdiag_experiment.atomic_write_json
    original_npy = kdiag_experiment._atomic_save_npy

    def counted(dictionary_arg, targets, **kwargs):
        calls.append(int(np.asarray(targets).shape[0]))
        return original_solver(dictionary_arg, targets, **kwargs)

    def failing_json(path, payload):
        candidate = Path(path)
        if "shard_staging" in candidate.parts and candidate.name == failure_name:
            raise RuntimeError(f"injected {failure_name}")
        return original_json(candidate, payload)

    def failing_npy(path, values):
        candidate = Path(path)
        if "shard_staging" in candidate.parts and candidate.name == failure_name:
            raise RuntimeError(f"injected {failure_name}")
        return original_npy(candidate, values)

    monkeypatch.setattr(kdiag_experiment, "streaming_nonnegative_pursuit", counted)
    monkeypatch.setattr(kdiag_experiment, "atomic_write_json", failing_json)
    monkeypatch.setattr(kdiag_experiment, "_atomic_save_npy", failing_npy)
    with pytest.raises(RuntimeError, match="injected"):
        run_scientific_bundle(
            config=config,
            bundle=bundle,
            shards=[shard],
            dictionary=dictionary,
            target=target,
            token_decoder=_decode_token_ids,
            run_root=tmp_path,
        )
    stage = tmp_path / config.k_diagnostic.artifact_root / "pilot"
    final = stage / "shards" / shard.shard_id
    assert not final.exists()
    staging = stage / "shard_staging" / bundle.bundle_id
    assert not staging.exists() or not any(staging.iterdir())
    real_complete = stage / "bundles" / bundle.bundle_id / "real_complete.json"
    real_complete_hash = sha256_file(real_complete)
    real_complete_mtime = real_complete.stat().st_mtime_ns

    calls.clear()
    monkeypatch.setattr(kdiag_experiment, "atomic_write_json", original_json)
    monkeypatch.setattr(kdiag_experiment, "_atomic_save_npy", original_npy)
    run_scientific_bundle(
        config=config,
        bundle=bundle,
        shards=[shard],
        dictionary=dictionary,
        target=target,
        token_decoder=_decode_token_ids,
        run_root=tmp_path,
    )
    assert calls == [1]
    assert final.is_dir()
    assert sha256_file(real_complete) == real_complete_hash
    assert real_complete.stat().st_mtime_ns == real_complete_mtime

    calls.clear()
    run_scientific_bundle(
        config=config,
        bundle=bundle,
        shards=[shard],
        dictionary=dictionary,
        target=target,
        token_decoder=_decode_token_ids,
        run_root=tmp_path,
    )
    assert calls == []


def test_v2_rotation_batch_interruption_persists_completed_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _small_v2_config()
    config = source.model_copy(
        update={
            "k_diagnostic": source.k_diagnostic.model_copy(update={"rotation_seeds": [1101, 1102]})
        }
    )
    shards = [
        ScientificShard(
            target_id="rotation-interruption",
            layer=11,
            target_family="logistic_probe",
            target_subtype="synthetic",
            metric="raw_euclidean",
            null_family="orthogonal_rotation",
            null_seed=seed,
            dictionary_fraction=1.0,
        )
        for seed in (1101, 1102)
    ]
    bundle = build_physical_bundles(shards)[0]
    prepare_rotation_cache(
        config,
        metric_spaces=[
            {
                "metric": "raw_euclidean",
                "transform_sha256": None,
                "dimension": 8,
                "layers": [11],
            }
        ],
        run_root=tmp_path,
        require_build_preflight=False,
    )
    rng = np.random.default_rng(1606)
    dictionary = DenseDictionary(rng.normal(size=(96, 8)))
    target = rng.normal(size=8)
    ordered = sorted(shards, key=lambda value: json.dumps(value.identity, sort_keys=True))
    first_shard, interrupted_shard = ordered
    original_json = kdiag_experiment.atomic_write_json
    original_solver = kdiag_experiment.streaming_nonnegative_pursuit
    calls: list[int] = []

    def counted(dictionary_arg, targets, **kwargs):
        calls.append(int(np.asarray(targets).shape[0]))
        return original_solver(dictionary_arg, targets, **kwargs)

    def fail_second_summary(path, payload):
        candidate = Path(path)
        if (
            "shard_staging" in candidate.parts
            and interrupted_shard.shard_id in str(candidate)
            and candidate.name == "summary.json"
        ):
            raise RuntimeError("interrupt second rotation shard")
        return original_json(candidate, payload)

    monkeypatch.setattr(kdiag_experiment, "streaming_nonnegative_pursuit", counted)
    monkeypatch.setattr(kdiag_experiment, "atomic_write_json", fail_second_summary)
    with pytest.raises(RuntimeError, match="interrupt second rotation"):
        run_scientific_bundle(
            config=config,
            bundle=bundle,
            shards=shards,
            dictionary=dictionary,
            target=target,
            token_decoder=_decode_token_ids,
            run_root=tmp_path,
        )
    assert calls == [1, 2]
    stage = tmp_path / config.k_diagnostic.artifact_root / "pilot"
    first_directory = stage / "shards" / first_shard.shard_id
    second_directory = stage / "shards" / interrupted_shard.shard_id
    assert first_directory.is_dir()
    assert not second_directory.exists()
    first_manifest_hash = sha256_file(first_directory / "manifest.json")
    real_complete = stage / "bundles" / bundle.bundle_id / "real_complete.json"
    real_complete_hash = sha256_file(real_complete)

    calls.clear()
    monkeypatch.setattr(kdiag_experiment, "atomic_write_json", original_json)
    run_scientific_bundle(
        config=config,
        bundle=bundle,
        shards=shards,
        dictionary=dictionary,
        target=target,
        token_decoder=_decode_token_ids,
        run_root=tmp_path,
    )
    assert calls == [1]
    assert sha256_file(first_directory / "manifest.json") == first_manifest_hash
    assert sha256_file(real_complete) == real_complete_hash
    assert second_directory.is_dir()


def test_v1_schema2_bundle_resumes_and_indexes_without_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _diagnostic_v1_config()
    config = source.model_copy(
        update={
            "k_diagnostic": source.k_diagnostic.model_copy(
                update={"k_max": 4, "report_grid": [1, 2, 4]}
            )
        }
    )
    shard = ScientificShard(
        target_id="legacy-v1",
        layer=11,
        target_family="logistic_probe",
        target_subtype="synthetic",
        metric="raw_euclidean",
        null_family="iid_full_cardinality",
        null_seed=101,
        dictionary_fraction=1.0,
    )
    bundle = build_physical_bundles([shard])[0]
    rng = np.random.default_rng(1607)
    dictionary = DenseDictionary(rng.normal(size=(96, 8)))
    target = rng.normal(size=8)
    first = run_scientific_bundle(
        config=config,
        bundle=bundle,
        shards=[shard],
        dictionary=dictionary,
        target=target,
        token_decoder=_decode_token_ids,
        run_root=tmp_path,
    )
    stage = tmp_path / config.k_diagnostic.artifact_root / "pilot"
    bundle_directory = stage / "bundles" / bundle.bundle_id
    shard_directory = stage / "shards" / shard.shard_id
    bundle_manifest = json.loads((bundle_directory / "manifest.json").read_text(encoding="utf-8"))
    shard_manifest = json.loads((shard_directory / "manifest.json").read_text(encoding="utf-8"))
    assert bundle_manifest["schema_version"] == 2
    assert "null_execution_version" not in bundle_manifest
    assert "null_execution_version" not in shard_manifest
    assert not (bundle_directory / "real_complete.json").exists()
    before = {
        path: (sha256_file(path), path.stat().st_mtime_ns)
        for directory in (bundle_directory, shard_directory)
        for path in directory.iterdir()
        if path.is_file()
    }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("v1 completed artifact was recomputed")

    monkeypatch.setattr(kdiag_experiment, "streaming_nonnegative_pursuit", forbidden)
    second = run_scientific_bundle(
        config=config,
        bundle=bundle,
        shards=[shard],
        dictionary=dictionary,
        target=target,
        token_decoder=_decode_token_ids,
        run_root=tmp_path,
    )
    assert second == first
    assert index_stage(config, [shard], run_root=tmp_path)["complete"] is True
    for path, identity in before.items():
        assert (sha256_file(path), path.stat().st_mtime_ns) == identity


def test_v2_bundle_cli_benchmark_resource_contract_and_tamper_rejection(
    tmp_path: Path,
) -> None:
    config = _small_v2_config()
    diagnostic = config.k_diagnostic
    shard = ScientificShard(
        target_id="benchmark-contract",
        layer=11,
        target_family="logistic_probe",
        target_subtype="synthetic",
        metric="raw_euclidean",
        null_family="iid_full_cardinality",
        null_seed=101,
        dictionary_fraction=1.0,
    )
    bundle = build_physical_bundles([shard])[0]
    rng = np.random.default_rng(1608)
    run_scientific_bundle(
        config=config,
        bundle=bundle,
        shards=[shard],
        dictionary=DenseDictionary(rng.normal(size=(96, 8))),
        target=rng.normal(size=8),
        token_decoder=_decode_token_ids,
        run_root=tmp_path,
    )
    stage = tmp_path / diagnostic.artifact_root / "pilot"
    manifest_path = stage / "bundles" / bundle.bundle_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = _k_diagnostic_v2_benchmark_memory_plan(
        manifest,
        expected_chunk_size=diagnostic.vocabulary_chunk_size,
    )
    assert plan["approved"] is True
    assert plan["checks"]["within_hard_peak_budget"] is True
    assert plan["checks"]["preserves_available_ram_reserve"] is True

    for mutation in (
        "version",
        "provider",
        "dimension",
        "storage_bytes",
        "workspace_bytes",
        "hard_cap",
        "peak",
        "checks",
    ):
        tampered = json.loads(json.dumps(manifest))
        tampered_plan = tampered["shared_direction_memory_plan"]
        if mutation == "version":
            tampered["null_execution_version"] = "stale_scalar_execution"
        elif mutation == "provider":
            tampered_plan["provider_version"] = "stale_direction_provider"
        elif mutation == "dimension":
            tampered_plan["d_model"] += 1
        elif mutation == "storage_bytes":
            tampered_plan["float64_direction_storage_bytes"] += 8
        elif mutation == "workspace_bytes":
            tampered_plan["generation_workspace_bytes"] += 8
        elif mutation == "hard_cap":
            tampered_plan["hard_peak_budget_bytes"] = 4 * 2**30
        elif mutation == "peak":
            tampered_plan["peak_bytes"] = SHARED_DIRECTION_MAX_PEAK_BYTES + 1
        else:
            tampered_plan["checks"]["within_hard_peak_budget"] = False
        with pytest.raises(ValueError, match="approved shared-direction"):
            _k_diagnostic_v2_benchmark_memory_plan(
                tampered,
                expected_chunk_size=diagnostic.vocabulary_chunk_size,
            )

    config_sha256 = hashlib.sha256(
        json.dumps(
            diagnostic.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    (stage / "bundles.json").write_text(
        json.dumps(
            {
                "physical_bundle_count": 1,
                "logical_replicate_count": 1,
                "bundles": [
                    {
                        "bundle_id": bundle.bundle_id,
                        "conservative_cost_units": 1,
                        "logical_shard_count": 1,
                        "shard_ids": [shard.shard_id],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    benchmark_path = stage / "microbenchmark.json"
    execution_hardware = _scheduled_blackwell_hardware()
    benchmark = {
        "schema_version": 5,
        "approved": True,
        "identity": diagnostic.identity,
        "stage": diagnostic.stage,
        "configuration_sha256": config_sha256,
        "null_execution_version": NULL_EXECUTION_VERSION,
        "bundle_id": bundle.bundle_id,
        "wall_seconds": 1.0,
        "shared_direction_memory_plan": plan,
        "execution_hardware": execution_hardware,
        "execution_hardware_sha256": execution_hardware_sha256(execution_hardware),
        "execution_backend": execution_hardware["execution_backend"],
        "artifact_usage": {
            "bundle_bytes": 1024,
            "bundle_inodes": 8,
            "max_logical_shard_bytes": 512,
            "max_logical_shard_inodes": 6,
        },
    }
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    rotation_build_path = stage / "rotation_cache_build_preflight.json"
    rotation_build_path.write_text(
        json.dumps(
            {
                "approved": True,
                "identity": diagnostic.identity,
                "stage": diagnostic.stage,
                "configuration_sha256": config_sha256,
                "binding_plan_sha256": "binding-plan",
            }
        ),
        encoding="utf-8",
    )
    (stage / "rotation_cache_index.json").write_text(
        json.dumps(
            {
                "complete": True,
                "identity": diagnostic.identity,
                "stage": diagnostic.stage,
                "configuration_sha256": config_sha256,
                "binding_plan_sha256": "binding-plan",
                "matrix_bytes": 0,
                "matrix_inodes": 0,
                "build_preflight_sha256": sha256_file(rotation_build_path),
            }
        ),
        encoding="utf-8",
    )
    runtime = record_bundle_runtime(
        config,
        bundle_id=bundle.bundle_id,
        wall_seconds=1.0,
        execution_hardware=execution_hardware,
        run_root=tmp_path,
    )
    assert runtime["null_execution_version"] == NULL_EXECUTION_VERSION
    approved = resource_preflight(config, run_root=tmp_path)
    assert approved["approved"] is True
    assert approved["null_execution_version"] == NULL_EXECUTION_VERSION
    assert approved["execution_hardware"] == execution_hardware
    assert approved["execution_backend"] == "fsm_local_scheduler"
    assert approved["execution_hardware_sha256"] == execution_hardware_sha256(execution_hardware)
    assert approved["hard_budgets"]["shared_direction_peak_bytes"] == 6 * 2**30

    for mutation in (
        "version",
        "hard_cap",
        "peak",
        "checks",
        "hardware_missing",
        "hardware_name",
        "hardware_backend",
        "hardware_hash",
    ):
        tampered = json.loads(json.dumps(benchmark))
        if mutation == "version":
            tampered["null_execution_version"] = "stale_scalar_execution"
        elif mutation == "hard_cap":
            tampered["shared_direction_memory_plan"]["hard_peak_budget_bytes"] = 4 * 2**30
        elif mutation == "peak":
            tampered["shared_direction_memory_plan"]["peak_bytes"] = (
                SHARED_DIRECTION_MAX_PEAK_BYTES + 1
            )
        elif mutation == "checks":
            tampered["shared_direction_memory_plan"]["checks"][
                "preserves_available_ram_reserve"
            ] = False
        elif mutation == "hardware_missing":
            tampered.pop("execution_hardware")
        elif mutation == "hardware_name":
            tampered["execution_hardware"]["gpu_names"] = ["Different scheduled GPU"]
        elif mutation == "hardware_backend":
            tampered["execution_backend"] = "slurm"
        else:
            tampered["execution_hardware_sha256"] = "0" * 64
        benchmark_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(KDiagnosticExperimentError):
            resource_preflight(config, run_root=tmp_path)
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")


def test_measured_resource_preflight_and_launcher_wave_gate(tmp_path: Path) -> None:
    config = _diagnostic_config()
    diagnostic = config.k_diagnostic
    config_sha256 = hashlib.sha256(
        json.dumps(
            diagnostic.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    stage = tmp_path / "artifacts/concept_intervention/qwen35_4b_k_diagnostic_v2/pilot"
    stage.mkdir(parents=True)
    memory_plan = shared_direction_memory_plan(
        n_atoms=96,
        d_model=8,
        chunk_size=diagnostic.vocabulary_chunk_size,
        available_bytes=16 * 2**30,
    )
    stable_plan = memory_plan
    execution_hardware = _scheduled_blackwell_hardware()
    bundle_directory = stage / "bundles" / "worst"
    bundle_directory.mkdir(parents=True)
    (bundle_directory / "manifest.json").write_text(
        json.dumps(
            {
                "null_execution_version": NULL_EXECUTION_VERSION,
                "dictionary_cardinality": 96,
                "dictionary_metric_dimension": 8,
                "shared_direction_memory_plan": stable_plan,
            }
        ),
        encoding="utf-8",
    )
    bundle = {
        "bundle_id": "worst",
        "conservative_cost_units": 100,
        "logical_shard_count": 2,
        "shard_ids": ["one", "two"],
    }
    (stage / "bundles.json").write_text(
        json.dumps(
            {
                "physical_bundle_count": 1,
                "logical_replicate_count": 2,
                "bundles": [bundle],
            }
        ),
        encoding="utf-8",
    )
    (stage / "microbenchmark.json").write_text(
        json.dumps(
            {
                "approved": True,
                "schema_version": 5,
                "null_execution_version": NULL_EXECUTION_VERSION,
                "identity": diagnostic.identity,
                "stage": diagnostic.stage,
                "configuration_sha256": config_sha256,
                "bundle_id": "worst",
                "wall_seconds": 1.0,
                "shared_direction_memory_plan": memory_plan,
                "execution_hardware": execution_hardware,
                "execution_hardware_sha256": execution_hardware_sha256(execution_hardware),
                "execution_backend": execution_hardware["execution_backend"],
                "artifact_usage": {
                    "bundle_bytes": 1024,
                    "bundle_inodes": 8,
                    "max_logical_shard_bytes": 512,
                    "max_logical_shard_inodes": 6,
                },
            }
        ),
        encoding="utf-8",
    )
    rotation_build_path = stage / "rotation_cache_build_preflight.json"
    rotation_build_path.write_text(
        json.dumps(
            {
                "approved": True,
                "identity": diagnostic.identity,
                "stage": diagnostic.stage,
                "configuration_sha256": config_sha256,
                "binding_plan_sha256": "binding-plan",
                "max_dimension": 8,
                "unique_matrix_count": 1,
                "projected_cache_build_wall_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (stage / "rotation_cache_index.json").write_text(
        json.dumps(
            {
                "complete": True,
                "identity": diagnostic.identity,
                "stage": diagnostic.stage,
                "configuration_sha256": config_sha256,
                "binding_plan_sha256": "binding-plan",
                "matrix_bytes": 4096,
                "matrix_inodes": 1,
                "build_preflight_sha256": sha256_file(rotation_build_path),
            }
        ),
        encoding="utf-8",
    )
    approved = resource_preflight(config, run_root=tmp_path)
    assert approved["approved"] is True
    assert approved["null_execution_version"] == NULL_EXECUTION_VERSION
    assert approved["shared_direction_memory_plan"] == memory_plan
    assert approved["execution_hardware"] == execution_hardware
    assert approved["checks"]["shared_direction_peak_bytes"] is True
    assert approved["checks"]["shared_direction_available_ram_reserve"] is True
    assert approved["stage_upper_bound"]["gpu_hours"] == pytest.approx(1 / 3600)
    assert "measured_max_shard" in approved["projection_formula"]

    stale = config.model_copy(
        update={"k_diagnostic": diagnostic.model_copy(update={"max_rotation_cache_gib": 24.0})}
    )
    with pytest.raises(KDiagnosticExperimentError, match="identity-mismatched"):
        resource_preflight(stale, run_root=tmp_path)

    runtime_root = stage / "runtimes"
    runtime_root.mkdir()
    runtime_path = runtime_root / "worst.json"
    runtime_payload = {
        "identity": diagnostic.identity,
        "stage": diagnostic.stage,
        "configuration_sha256": config_sha256,
        "bundle_id": "worst",
        "wall_seconds": 1.0,
        "within_hard_limit": True,
        "null_execution_version": NULL_EXECUTION_VERSION,
        "execution_backend": execution_hardware["execution_backend"],
        "execution_hardware": execution_hardware,
        "execution_hardware_sha256": execution_hardware_sha256(execution_hardware),
    }
    for field, value in (
        ("identity", "wrong-identity"),
        ("stage", "raw_metric_full"),
        ("configuration_sha256", "0" * 64),
        ("bundle_id", "unregistered-bundle"),
        ("null_execution_version", "stale_scalar_execution"),
        ("execution_backend", "slurm"),
    ):
        stale_runtime = dict(runtime_payload)
        stale_runtime[field] = value
        runtime_path.write_text(json.dumps(stale_runtime), encoding="utf-8")
        with pytest.raises(KDiagnosticExperimentError, match="runtime identity mismatch"):
            resource_preflight(config, run_root=tmp_path)
    runtime_payload["wall_seconds"] = diagnostic.max_seconds_per_bundle + 1
    runtime_payload["within_hard_limit"] = False
    runtime_path.write_text(json.dumps(runtime_payload), encoding="utf-8")
    with pytest.raises(KDiagnosticExperimentError, match="resource preflight rejected"):
        resource_preflight(config, run_root=tmp_path)

    launcher = (
        ROOT / "Concept_intervention/scripts/submit_qwen35_4b_k_diagnostic_v2.sh"
    ).read_text(encoding="utf-8")
    assert launcher.index("resource-preflight") < launcher.index("for ((OFFSET")
    assert launcher.count("resource-preflight") >= 2
    assert "run_qwen35_4b_k_diagnostic_v2_rotations.slurm" in launcher
    assert "sbatch --wait --parsable --array=" in launcher


def _write_synthetic_report_stage(root: Path, stage: str, rows: list[dict[str, object]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = root / stage
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "index.json").write_text(
        json.dumps(
            {
                "complete": True,
                "identity": "qwen35_4b_k_diagnostic_v2",
                "stage": stage,
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(pa.Table.from_pylist(rows), directory / "summary.parquet")
    (directory / "grid.json").write_text(
        json.dumps(
            {
                "shards": [
                    {
                        key: row[key]
                        for key in (
                            "target_id",
                            "layer",
                            "target_family",
                            "target_subtype",
                            "metric",
                            "null_family",
                            "dictionary_fraction",
                            "null_seed",
                        )
                    }
                    for row in rows
                ]
            }
        ),
        encoding="utf-8",
    )
    unique_targets = {
        str(row["target_id"]): {
            "target_id": row["target_id"],
            "file_sha256": "e" * 64,
            "layer": row["layer"],
            "target_family": row["target_family"],
            "target_subtype": row["target_subtype"],
        }
        for row in rows
    }
    membership = root / "target_membership"
    membership.mkdir(parents=True, exist_ok=True)
    (membership / f"{stage}.json").write_text(
        json.dumps({"targets": list(unique_targets.values())}), encoding="utf-8"
    )


def test_complete_synthetic_artifacts_generate_ten_figures_and_decisions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts/concept_intervention/qwen35_4b_k_diagnostic_v2"
    targets: list[dict[str, object]] = [
        {
            "target_id": "raw",
            "layer": 11,
            "target_family": "raw_activation",
            "target_subtype": "token",
            "source_id": "shared",
            "metadata": {},
        },
        {
            "target_id": "logistic",
            "layer": 11,
            "target_family": "logistic_probe",
            "target_subtype": "anger",
            "source_id": "shared",
            "metadata": {},
        },
        {
            "target_id": "class-mean",
            "layer": 11,
            "target_family": "class_mean_difference",
            "target_subtype": "anger",
            "source_id": "shared",
            "metadata": {},
        },
    ]
    for arm in (
        "label_explicit",
        "label_free_definition",
        "four_unrelated_topics_control",
    ):
        end_position = "concept_token_end" if arm == "label_explicit" else "definition_span_end"
        for position in ("assistant_boundary", end_position):
            targets.append(
                {
                    "target_id": f"contrast-{arm}-{position}",
                    "layer": 11,
                    "target_family": "seven_emotion_label_contrast",
                    "target_subtype": f"{arm}:{position}:anger",
                    "source_id": "shared",
                    "metadata": {
                        "contrast_arm": arm,
                        "capture_position": position,
                        "prompt_manifest_sha256": f"{arm}:{position}",
                    },
                }
            )
    raw_only_layer_targets = [
        dict(target_entry)
        | {
            "target_id": f"{target_entry['target_id']}-layer19",
            "layer": 19,
            "metadata": dict(target_entry["metadata"]),
        }
        for target_entry in targets
    ]
    targets.extend(raw_only_layer_targets)
    targets.extend(
        {
            "target_id": f"{target_family}-layer{layer}",
            "layer": layer,
            "target_family": target_family,
            "target_subtype": "anger",
            "source_id": "shared",
            "metadata": {},
        }
        for layer in (3, 7, 15, 23)
        for target_family in ("logistic_probe", "class_mean_difference")
    )
    vector_root = root / "targets/vectors"
    vector_root.mkdir(parents=True)
    rng = np.random.default_rng(59)
    for target_entry in targets:
        vector_path = Path("targets/vectors") / f"{target_entry['target_id']}.npy"
        np.save(root / vector_path, rng.normal(size=8))
        target_entry["vector_path"] = str(vector_path)
    (root / "targets/index.json").write_text(
        json.dumps({"complete": True, "targets": targets}), encoding="utf-8"
    )

    def summary_row(
        target_entry: dict[str, object],
        null_family: str,
        *,
        metric: str = "raw_euclidean",
        fraction: float = 1.0,
        effect: float = 0.03,
    ) -> dict[str, object]:
        censored = target_entry["target_id"] == "raw" and null_family == "iid_full_cardinality"
        return {
            "target_id": target_entry["target_id"],
            "layer": target_entry["layer"],
            "target_family": target_entry["target_family"],
            "target_subtype": target_entry["target_subtype"],
            "metric": metric,
            "null_family": null_family,
            "dictionary_fraction": fraction,
            "null_seed": 101,
            "null_seeds": [101],
            "null_seed_count": 1,
            "K_first": None if censored else 2,
            "K_first_right_censored": censored,
            "K_consecutive3": None if censored else 3,
            "K_consecutive3_right_censored": censored,
            "K_peak": 2,
            "max_cumulative_excess": effect,
            "cumulative_excess": [effect / 4, effect / 2, effect * 0.75, effect],
            "fixed_budget": {
                str(k): {
                    "explained_fraction": 0.2 + k / 100,
                    "null_median_explained_fraction": 0.1 + k / 200,
                    "real_minus_null": 0.1,
                }
                for k in (1, 4, 16, 25)
            },
            "empirical_diagnostics": _synthetic_empirical(1),
        }

    raw_rows: list[dict[str, object]] = []
    for target_entry in targets:
        for null_family, fraction in (
            ("iid_full_cardinality", 1.0),
            ("random_prefix_same_k", 1.0),
            ("orthogonal_rotation", 1.0),
        ):
            raw_rows.append(summary_row(target_entry, null_family, fraction=fraction))
        for fraction in (1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0):
            raw_rows.append(summary_row(target_entry, "iid_cardinality_sweep", fraction=fraction))
        if target_entry["target_family"] in {"logistic_probe", "class_mean_difference"}:
            raw_rows.append(summary_row(target_entry, "label_permutation_probe"))
    transformed_rows = [
        summary_row(target_entry, "iid_full_cardinality", metric=metric, effect=effect)
        for target_entry in targets
        if target_entry["target_family"] != "raw_activation" and target_entry["layer"] == 11
        for metric, effect in (
            ("j_pca_r256", 0.05),
            ("activation_whitened_0.10", 0.04),
        )
    ]
    _write_synthetic_report_stage(root, "pilot", raw_rows)
    _write_synthetic_report_stage(root, "raw_metric_full", raw_rows)
    _write_synthetic_report_stage(root, "transformed_metric", transformed_rows)

    shard_root = root / "raw_metric_full/shards"
    bundle_ids: dict[str, str] = {}
    bundle_entries = []
    for bundle_index, target_entry in enumerate(targets):
        bundle_id = f"synthetic-real-{bundle_index:03d}"
        bundle_ids[str(target_entry["target_id"])] = bundle_id
        bundle_root = root / "raw_metric_full/bundles" / bundle_id
        bundle_root.mkdir(parents=True)
        np.save(bundle_root / "gains_real.npy", np.asarray([0.2, 0.1, 0.05, 0.025]))
        fixed = {
            "schema_version": 1,
            "fixed_budgets": [1, 2, 4, 8, 16, 25, 32, 64],
            "budgets": {
                str(k): {
                    "requested_k": k,
                    "available": True,
                    "selected_token_ids": list(range(min(k, 4))),
                }
                for k in (1, 2, 4, 8, 16, 25, 32, 64)
            },
        }
        (bundle_root / "fixed_budget_real.json").write_text(json.dumps(fixed), encoding="utf-8")
        real_files = {
            name: sha256_file(bundle_root / name)
            for name in ("gains_real.npy", "fixed_budget_real.json")
        }
        (bundle_root / "complete.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "physical_bundle_id": bundle_id,
                    "real_files": real_files,
                }
            ),
            encoding="utf-8",
        )
        bundle_entries.append(
            {
                "bundle_id": bundle_id,
                "target_id": target_entry["target_id"],
                "metric": "raw_euclidean",
                "shard_ids": [],
            }
        )
    bundle_entry_by_id = {entry["bundle_id"]: entry for entry in bundle_entries}
    for index, row in enumerate(
        entry
        for entry in raw_rows
        if entry["null_family"] in {"iid_full_cardinality", "iid_cardinality_sweep"}
    ):
        directory = shard_root / f"shard-{index:03d}"
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(json.dumps(row), encoding="utf-8")
        np.save(directory / "null_gains.npy", np.asarray([[0.1, 0.05, 0.025, 0.01]]))
        bundle_id = bundle_ids[str(row["target_id"])]
        bundle_entry_by_id[bundle_id]["shard_ids"].append(directory.name)
        referenced_bundle = root / "raw_metric_full/bundles" / bundle_id
        real_files = {
            name: sha256_file(referenced_bundle / name)
            for name in ("gains_real.npy", "fixed_budget_real.json")
        }
        (directory / "real_reference.json").write_text(
            json.dumps(
                {
                    "physical_bundle_id": bundle_id,
                    "files": real_files,
                }
            ),
            encoding="utf-8",
        )
        (directory / "null_metadata.json").write_text(
            json.dumps(
                {
                    "metric_dimension": 8,
                    "actual_cardinality": 64,
                }
            ),
            encoding="utf-8",
        )

    (root / "raw_metric_full/bundles.json").write_text(
        json.dumps({"bundles": bundle_entries}), encoding="utf-8"
    )

    source = _diagnostic_config("raw_metric_full")
    diagnostic = source.k_diagnostic.model_copy(
        update={
            "minimum_null_replicates": 1,
            "hierarchical_bootstrap_samples": 1000,
        }
    )
    result = generate_report(artifact_root=root, diagnostic=diagnostic)
    decisions = json.loads(Path(result["decision_table"]).read_text(encoding="utf-8"))
    assert decisions["primary_estimand"] == "fixed_k4_real_minus_median_null"
    assert "Delta4" in decisions["primary_estimand_formula"]
    assert "exploratory only" in decisions["secondary_estimand_policy"]
    assert len(decisions["mechanism_hypotheses"]) == 14
    assert all(
        {
            "effect",
            "ci_low",
            "ci_high",
            "minimum_effect",
            "conclusion",
            "numerator",
            "denominator",
            "pair_key",
            "n_pairs",
            "cluster_counts",
            "comparison_direction",
        }
        <= set(entry)
        for entry in decisions["mechanism_hypotheses"]
    )
    terminal = result["terminal_summary"]
    assert {key.split("_", 1)[0] for key in terminal if key[:1] in "ABCDEFGHIJ"} == set(
        "ABCDEFGHIJ"
    )
    assert (
        terminal["A_raw_activation_vs_logistic_probe"]["raw_activation"][
            "population_median_reported"
        ]
        is False
    )
    terminal_a = terminal["A_raw_activation_vs_logistic_probe"]
    assert terminal_a["common_layers"] == [11, 19, 27]
    assert terminal_a["comparison_scope"] == ("both families restricted to common layers")
    assert terminal_a["logistic_probe"]["observed_crossing_count"] == 2
    assert set(terminal["result_only_decision_table"]) == {
        "target_type_effect",
        "full_cardinality_null_too_strong",
        "geometry_preserving_null_changes_K",
        "pca_improves_j_specific_separation",
        "whitening_improves_j_specific_separation",
    }
    assert (root / "report/terminal_summary.json").is_file()
    assert (root / "report/target_pair_summary.parquet").is_file()
    aggregate = json.loads((root / "report/aggregate_summary.json").read_text(encoding="utf-8"))
    assert aggregate["family_summary"]["logistic_probe"]["target_count"] == 6
    pair_rows = aggregate["target_pair_summary"]
    assert len({row["pair_type"] for row in pair_rows}) == 13
    assert all(
        {
            "pair_identity",
            "target_vector_cosine",
            "support_jaccard_at_4",
            "support_jaccard_at_16",
            "support_jaccard_at_25",
        }
        <= set(row)
        for row in pair_rows
    )
    for figure in FIGURE_NAMES:
        assert (root / "report/figures" / f"{figure}.pdf").is_file()
        assert (root / "report/figures" / f"{figure}.png").is_file()
