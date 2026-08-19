"""Public interface for residual-space J-component intervention."""

from .intervention import (
    ResidualIntervention,
    generate_with_intervention,
    intervention_session,
    matched_random_direction,
    multilayer_intervention_session,
)
from .multilayer import (
    MultiLayerJError,
    aggregate_layer_k,
    load_multilayer_directions,
    rebuild_multilayer_j_index,
    run_multilayer_j_intervention,
)
from .workflow import (
    ConceptInterventionError,
    DirectionRecord,
    load_registered_directions,
    rebuild_intervention_index,
    run_concept_intervention,
)

__all__ = [
    "ConceptInterventionError",
    "DirectionRecord",
    "MultiLayerJError",
    "ResidualIntervention",
    "aggregate_layer_k",
    "generate_with_intervention",
    "intervention_session",
    "load_multilayer_directions",
    "load_registered_directions",
    "matched_random_direction",
    "multilayer_intervention_session",
    "rebuild_intervention_index",
    "rebuild_multilayer_j_index",
    "run_concept_intervention",
    "run_multilayer_j_intervention",
]
