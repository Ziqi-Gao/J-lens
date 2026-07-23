"""Public interface for residual-space J-component intervention."""

from .intervention import (
    ResidualIntervention,
    generate_with_intervention,
    intervention_session,
    matched_random_direction,
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
    "ResidualIntervention",
    "generate_with_intervention",
    "intervention_session",
    "load_registered_directions",
    "matched_random_direction",
    "rebuild_intervention_index",
    "run_concept_intervention",
]
