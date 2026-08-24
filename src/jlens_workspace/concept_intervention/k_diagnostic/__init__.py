"""Registered K-diagnostic for target geometry, nulls, and metric sensitivity.

The package is independent of the immutable occupancy and intervention
workflows.  Its numerical cores are NumPy/SciPy-only; model and Torch imports
remain inside the experiment entry points.
"""

from .aggregation import (
    classify_registered_effect,
    hierarchical_bootstrap_mean,
    hierarchical_group_contrast,
    observed_crossing_summary,
    summarize_null_calibrated_curve,
)
from .nulls import (
    PrefixPursuitResult,
    fit_fixed_c_permutation_probe,
    group_safe_permutation,
    haar_orthogonal,
    matched_norm_subset,
    permuted_class_mean_difference,
    random_prefix_pursuit,
)
from .targets import (
    CONCEPT_TEMPLATES,
    TargetRecord,
    class_mean_difference,
    find_unique_subsequence,
    paired_template_contrasts,
    raw_activation_positions,
)
from .theory import isotropic_sphere_reference, sphere_max_quantile
from .transformed_dictionary import LinearTransformedDictionary

__all__ = [
    "CONCEPT_TEMPLATES",
    "LinearTransformedDictionary",
    "PrefixPursuitResult",
    "TargetRecord",
    "class_mean_difference",
    "classify_registered_effect",
    "find_unique_subsequence",
    "fit_fixed_c_permutation_probe",
    "group_safe_permutation",
    "haar_orthogonal",
    "hierarchical_bootstrap_mean",
    "hierarchical_group_contrast",
    "isotropic_sphere_reference",
    "matched_norm_subset",
    "observed_crossing_summary",
    "paired_template_contrasts",
    "permuted_class_mean_difference",
    "random_prefix_pursuit",
    "raw_activation_positions",
    "sphere_max_quantile",
    "summarize_null_calibrated_curve",
]
