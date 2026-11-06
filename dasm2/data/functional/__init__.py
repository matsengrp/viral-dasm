"""Functional data analysis utilities.

This module provides utilities for analyzing functional data (DMS, FLAb, etc.):
- Consensus sequence computation
- Single-mutant filtering
- DMS-style matrix conversion
- Selection score computation for multi-mutant variants

These utilities complement the format converters in `functional_formats.py`:
- `functional_formats.py`: Data loading and format conversion (input)
- `functional/`: Analysis utilities for working with loaded data

Typical workflow for single-mutant DMS data:
    from dasm2.data.functional_formats import from_flab
    from dasm2.data.functional import (
        compute_consensus,
        filter_single_mutants,
        to_dms_matrix,
    )

    # Load data
    df = from_flab(csv_path, sequence_cols=["heavy"], y_col="fitness")

    # Compute consensus from variants
    consensus = compute_consensus(df, "sequence_heavy")

    # Filter to single mutants and get matrix
    single_mutants = filter_single_mutants(df, "sequence_heavy", consensus)
    matrix = to_dms_matrix(single_mutants, "y_fitness", consensus)

Typical workflow for multi-mutant variants:
    from dasm2.data.functional import (
        compute_consensus,
        add_selection_scores,
    )

    # Compute consensus and add selection scores
    consensus = compute_consensus(df, "sequence_heavy")
    log_sf = torch.log(model.selection_factors_of_aa_str(consensus)[0]).T
    df = add_selection_scores(df, log_sf, consensus, "sequence_heavy")
"""

from dasm2.data.functional.codon_neighbor import (
    backtranslate_human_optimal,
    make_codon_neighbor_indicator,
    split_by_neighbor_status,
)
from dasm2.data.functional.consensus import compute_consensus
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
from dasm2.data.functional.model_scores import hermes_logits_to_dms_matrix
from dasm2.data.functional.replicates import average_replicates, replicate_summary
from dasm2.data.functional.selection_score import (
    add_selection_scores,
    compute_selection_score,
    compute_sequence_differences,
)
from dasm2.data.functional.single_mutants import filter_single_mutants
from dasm2.data.functional.visualization import (
    NeighborComparisonStats,
    neighbor_comparison_stats,
    plot_dms_heatmap,
    plot_neighbor_comparison,
    plot_neighbor_comparison_grid,
)

__all__ = [
    "CorrelationMethod",
    "NeighborComparisonStats",
    "add_selection_scores",
    "aligned_finite_pairs",
    "average_replicates",
    "backtranslate_human_optimal",
    "compute_consensus",
    "compute_selection_score",
    "compute_sequence_differences",
    "correlate_dms_matrices",
    "covered_position_range",
    "exclude_wildtype_entries",
    "filter_single_mutants",
    "hermes_logits_to_dms_matrix",
    "make_codon_neighbor_indicator",
    "mask_wt_to_nan",
    "melt_dms_matrix",
    "neighbor_comparison_stats",
    "plot_dms_heatmap",
    "plot_neighbor_comparison",
    "plot_neighbor_comparison_grid",
    "replicate_summary",
    "restrict_to_shared_cells",
    "selection_factors_to_dms_matrix",
    "split_by_neighbor_status",
    "to_dms_matrix",
]
