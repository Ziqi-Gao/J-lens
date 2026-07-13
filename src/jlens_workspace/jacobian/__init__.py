"""Official Jacobian-lens integration and artifact conventions."""

from .adapter import (
    ManagedJacobianLens,
    OfficialJLensAdapter,
    OfficialJLensUnavailableError,
    fit_lens,
    import_official_jlens,
    load_lens,
    save_lens,
)
from .metadata import (
    ARTIFACT_FORMAT,
    FORMAT_KEY,
    METADATA_KEY,
    JLensMetadata,
    JLensMetadataError,
    JLensMetadataMismatchError,
)
from .unembedding import (
    EffectiveUnembedding,
    EffectiveUnembeddingMetadata,
    UnembeddingConvention,
    UnsupportedNormalizationError,
    build_effective_unembedding,
    restrict_effective_unembedding,
)

__all__ = [
    "ARTIFACT_FORMAT",
    "FORMAT_KEY",
    "METADATA_KEY",
    "EffectiveUnembedding",
    "EffectiveUnembeddingMetadata",
    "JLensMetadata",
    "JLensMetadataError",
    "JLensMetadataMismatchError",
    "ManagedJacobianLens",
    "OfficialJLensAdapter",
    "OfficialJLensUnavailableError",
    "UnembeddingConvention",
    "UnsupportedNormalizationError",
    "build_effective_unembedding",
    "fit_lens",
    "import_official_jlens",
    "load_lens",
    "restrict_effective_unembedding",
    "save_lens",
]
