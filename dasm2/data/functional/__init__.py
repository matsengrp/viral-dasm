"""DMS-matrix utilities for the functional-data comparison.

Upstream this package re-exports eight modules' worth of functional-data
analysis -- consensus sequences, replicate averaging, single-mutant filtering,
selection scores, HERMES logits, plotting. `fig:dms` reaches one of them, so
only ``dms_matrix`` is vendored and only what it defines is re-exported here.
See ``dasm2/PROVENANCE.md``.

``functional_formats`` beside this package is the loader -- it reads a
published CSV into the standard long format -- and this package is what turns
that format into the 20 x L matrix a panel correlates.
"""

from dasm2.data.functional.dms_matrix import (
    CorrelationMethod,
    aligned_finite_pairs,
    correlate_dms_matrices,
    covered_position_range,
    exclude_wildtype_entries,
    mask_wt_to_nan,
    melt_dms_matrix,
    restrict_to_shared_cells,
    selection_factors_to_dms_matrix,
    to_dms_matrix,
)

__all__ = [
    "CorrelationMethod",
    "aligned_finite_pairs",
    "correlate_dms_matrices",
    "covered_position_range",
    "exclude_wildtype_entries",
    "mask_wt_to_nan",
    "melt_dms_matrix",
    "restrict_to_shared_cells",
    "selection_factors_to_dms_matrix",
    "to_dms_matrix",
]
