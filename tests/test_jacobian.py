from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from jlens_workspace.jacobian import (  # noqa: E402
    JLensMetadata,
    JLensMetadataMismatchError,
    ManagedJacobianLens,
    OfficialJLensAdapter,
    UnembeddingConvention,
    UnsupportedNormalizationError,
    build_effective_unembedding,
    restrict_effective_unembedding,
)


class _FakeLens:
    def __init__(self, jacobians, *, n_prompts, d_model):
        self.jacobians = {int(k): value.float() for k, value in jacobians.items()}
        self.source_layers = sorted(self.jacobians)
        self.n_prompts = int(n_prompts)
        self.d_model = int(d_model)

    @classmethod
    def load(cls, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            state["J"], n_prompts=state["n_prompts"], d_model=state["d_model"]
        )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # pragma: no cover - local test only
        raise AssertionError("unexpected remote load")


@pytest.fixture
def fake_official(monkeypatch):
    module = types.ModuleType("jlens")
    module.JacobianLens = _FakeLens
    module.__version__ = "test"

    def fit(model, prompts, **kwargs):
        source_layers = kwargs.get("source_layers", range(model.n_layers - 1))
        return _FakeLens(
            {int(layer): torch.eye(model.d_model) for layer in source_layers},
            n_prompts=len(prompts),
            d_model=model.d_model,
        )

    module.fit = fit
    monkeypatch.setitem(sys.modules, "jlens", module)
    return module


def _metadata(**overrides):
    values = {
        "model_id": "org/model",
        "model_revision": "model-commit",
        "tokenizer_id": "org/model",
        "tokenizer_revision": "tokenizer-commit",
        "d_model": 3,
        "source_layers": (0, 2),
        "target_layer": 3,
        "n_prompts": 7,
    }
    values.update(overrides)
    return JLensMetadata(**values)


def test_metadata_validates_layer_shape_and_identity():
    lens = _FakeLens(
        {0: torch.eye(3), 2: torch.eye(3)}, n_prompts=7, d_model=3
    )
    metadata = _metadata()
    metadata.validate_lens(lens)
    metadata.validate_expected(
        {
            "model_id": "org/model",
            "model_revision": "model-commit",
            "tokenizer_revision": "tokenizer-commit",
            "layers": [0, 2],
        }
    )
    with pytest.raises(JLensMetadataMismatchError, match="model_revision"):
        metadata.validate_expected({"model_revision": "moving-main"})
    with pytest.raises(JLensMetadataMismatchError, match="d_model"):
        _metadata(d_model=4).validate_lens(lens)


def test_adapter_round_trip_embeds_metadata_and_is_official_compatible(
    tmp_path, fake_official
):
    raw = _FakeLens(
        {0: torch.eye(3), 2: torch.diag(torch.tensor([1.0, 2.0, 3.0]))},
        n_prompts=7,
        d_model=3,
    )
    managed = ManagedJacobianLens(raw, _metadata())
    path = OfficialJLensAdapter.save(managed, tmp_path / "lens.pt")
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["jlens_workspace_metadata"]["model_revision"] == "model-commit"
    assert state["J"][0].dtype == torch.float32

    loaded = OfficialJLensAdapter.load(
        path,
        expected={
            "model_id": "org/model",
            "model_revision": "model-commit",
            "tokenizer_revision": "tokenizer-commit",
        },
    )
    assert loaded.metadata == managed.metadata
    torch.testing.assert_close(loaded.jacobians[2], raw.jacobians[2])
    with pytest.raises(JLensMetadataMismatchError, match="tokenizer_revision"):
        OfficialJLensAdapter.load(
            path, expected={"tokenizer_revision": "wrong-commit"}
        )


def test_upstream_checkpoint_requires_explicit_identity(tmp_path, fake_official):
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "J": {0: torch.eye(3), 2: torch.eye(3)},
            "n_prompts": 7,
            "source_layers": [0, 2],
            "d_model": 3,
        },
        path,
    )
    with pytest.raises(ValueError, match="no embedded model/tokenizer metadata"):
        OfficialJLensAdapter.load(path)
    loaded = OfficialJLensAdapter.load(path, metadata=_metadata())
    assert loaded.metadata.model_revision == "model-commit"


def test_fit_validates_runtime_revisions_and_resolves_target(fake_official):
    hf_model = types.SimpleNamespace(
        name_or_path="org/model",
        config=types.SimpleNamespace(_commit_hash="model-commit"),
    )
    tokenizer = types.SimpleNamespace(
        name_or_path="org/model", init_kwargs={"_commit_hash": "tokenizer-commit"}
    )
    model = types.SimpleNamespace(
        _hf_model=hf_model,
        tokenizer=tokenizer,
        n_layers=3,
        d_model=2,
    )
    metadata = _metadata(
        d_model=2, source_layers=(0, 1), target_layer=None, n_prompts=None
    )
    fitted = OfficialJLensAdapter.fit(model, ["one", "two"], metadata=metadata)
    assert fitted.metadata.target_layer == 2
    assert fitted.metadata.n_prompts == 2
    with pytest.raises(JLensMetadataMismatchError, match="model_revision"):
        OfficialJLensAdapter.fit(
            model,
            ["one"],
            metadata=_metadata(
                d_model=2,
                source_layers=(0, 1),
                target_layer=None,
                n_prompts=None,
                model_revision="wrong",
            ),
        )


class ToyRMSNorm(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.as_tensor(weight, dtype=torch.float64))
        self.variance_epsilon = 1e-6


class GemmaToyRMSNorm(ToyRMSNorm):
    pass


class ToyModel(torch.nn.Module):
    def __init__(self, weight, gamma, norm_type=ToyRMSNorm):
        super().__init__()
        vocab, width = weight.shape
        self._lm_head = torch.nn.Linear(width, vocab, bias=False, dtype=torch.float64)
        self._lm_head.weight.data.copy_(weight)
        self._final_norm = norm_type(gamma)


def test_effective_unembedding_raw_and_rmsnorm_weighted_are_explicit():
    weight = torch.tensor(
        [[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]], dtype=torch.float64
    )
    gamma = torch.tensor([0.5, 2.0, -1.0], dtype=torch.float64)
    model = ToyModel(weight, gamma)
    raw = build_effective_unembedding(model, convention=UnembeddingConvention.RAW)
    weighted = build_effective_unembedding(
        model, convention=UnembeddingConvention.RMSNORM_WEIGHTED
    )
    torch.testing.assert_close(raw.rows(0, 2), weight)
    torch.testing.assert_close(weighted.rows(0, 2), weight * gamma)
    assert raw.metadata.convention == "raw"
    assert weighted.metadata.norm_type == "ToyRMSNorm"
    assert weighted.metadata.norm_scale_parameterization == "weight_plus_0"
    assert weighted.metadata.activation_dependent_scale_omitted


def test_effective_unembedding_excludes_padded_non_token_rows_without_copy():
    weight = torch.arange(15, dtype=torch.float64).reshape(5, 3)
    effective = build_effective_unembedding(weight)
    restricted = restrict_effective_unembedding(effective, 4)
    assert restricted.shape == (4, 3)
    assert restricted.weight.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
    assert restricted.metadata.vocab_size == 4
    assert restricted.metadata.unembedding_vocab_size == 5
    assert restricted.metadata.row_selection == "tokenizer_ids_[0,4)"
    with pytest.raises(ValueError, match="only 5 rows"):
        restrict_effective_unembedding(effective, 6)


def test_effective_unembedding_records_pinned_or_safely_inferred_model_identity():
    model = ToyModel(torch.eye(2, dtype=torch.float64), torch.ones(2))
    model.name_or_path = "observed/model"
    model.config = types.SimpleNamespace(_commit_hash="observed-commit")

    inferred = build_effective_unembedding(model)
    assert inferred.metadata.model_id == "observed/model"
    assert inferred.metadata.model_revision == "observed-commit"

    explicit = build_effective_unembedding(
        model,
        model_id="pinned/model",
        model_revision="pinned-commit",
    )
    assert explicit.metadata.model_id == "pinned/model"
    assert explicit.metadata.model_revision == "pinned-commit"


def test_gemma_rmsnorm_unit_offset_is_folded_and_recorded():
    weight = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    model = ToyModel(weight, torch.tensor([0.25, -0.5]), norm_type=GemmaToyRMSNorm)
    effective = build_effective_unembedding(model, convention="rmsnorm_weighted")
    torch.testing.assert_close(
        effective.rows(0, 1), weight * torch.tensor([1.25, 0.5])
    )
    assert effective.metadata.norm_scale_parameterization == "one_plus_weight"


def test_rmsnorm_convention_rejects_layernorm_instead_of_silent_fallback():
    weight = torch.eye(3, dtype=torch.float64)
    model = ToyModel(weight, torch.ones(3))
    model._final_norm = torch.nn.LayerNorm(3, dtype=torch.float64)
    with pytest.raises(UnsupportedNormalizationError, match="only supports RMSNorm"):
        build_effective_unembedding(model, convention="rmsnorm_weighted")
