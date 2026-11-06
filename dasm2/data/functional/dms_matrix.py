"""DMS matrix format utilities.

Provides functions for converting single-mutant data to and from
DMS-style matrix format (20 amino acids x L positions).
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy import stats

from dasm2.data.functional_formats import (
    MUT_AA_COL,
    POSITION_COL,
    InsufficientDataError,
)
from dasm2.sequence.constants import AA_STR_SORTED

# Minimum data points for meaningful correlation (2 points always give r=±1)
MIN_CORRELATION_PAIRS = 3

#: Every correlation method :func:`correlate_dms_matrices` accepts. Spelled as
#: a ``Literal`` so a typo in a call is a type error at the call site rather
#: than only a ``ValueError`` once that line runs; the runtime check in
#: :func:`correlate_dms_matrices` stays, because CLI strings reach it untyped.
CorrelationMethod = Literal["spearman", "pearson", "kendall"]

#: The accepted methods mapped to the SciPy function computing each.
#: ``"kendall"`` is tau-b (SciPy's default variant), which is the
#: tie-corrected one -- the within-site rank statistics that motivated it feed
#: heavily tied input (every site contributes the same small set of ranks),
#: where tau-a would be attenuated by the ties rather than corrected for them.
CORRELATION_METHODS: Dict[CorrelationMethod, Callable[..., Any]] = {
    "spearman": stats.spearmanr,
    "pearson": stats.pearsonr,
    "kendall": stats.kendalltau,
}


def to_dms_matrix(
    df: pd.DataFrame,
    y_col: str,
    reference_sequence: str,
    position_col: str = POSITION_COL,
    mut_aa_col: str = MUT_AA_COL,
    fill_wt_value: Optional[float] = None,
    aa_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convert single-mutant data to DMS-style matrix format.

    Creates a DataFrame with amino acids as rows and positions as columns,
    where each cell contains the functional measurement for that mutation.

    Args:
        df: DataFrame with single-mutant data (should have position and
            mutant AA columns, e.g., from filter_single_mutants()).
        y_col: Column containing the functional measurement values.
        reference_sequence: Reference sequence (used for determining
            sequence length and WT positions).
        position_col: Column with 0-indexed mutation positions.
        mut_aa_col: Column with mutant amino acid.
        fill_wt_value: Value to use for WT positions (diagonal).
            None leaves them as NaN.
        aa_order: Order of amino acids for rows. Defaults to AA_STR_SORTED
            ("ACDEFGHIKLMNPQRSTVWY").

    Returns:
        DataFrame with shape (20, seq_len) where:
        - Index is amino acids in aa_order
        - Columns are position integers (0 to seq_len-1)
        - Values are functional measurements (NaN for unmeasured)

    Raises:
        ValueError: If required columns are missing.

    Example:
        >>> # After filter_single_mutants():
        >>> matrix = to_dms_matrix(df, "y_fitness", reference_sequence="AAA")
        >>> matrix.shape
        (20, 3)
        >>> matrix.loc["G", 1]  # Fitness of A2G mutation
        -0.5
    """
    if y_col not in df.columns:
        raise ValueError(
            f"Value column '{y_col}' not found. Available columns: {list(df.columns)}"
        )
    if position_col not in df.columns:
        raise ValueError(
            f"Position column '{position_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if mut_aa_col not in df.columns:
        raise ValueError(
            f"Mutant AA column '{mut_aa_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if aa_order is None:
        aa_order = list(AA_STR_SORTED)

    seq_len = len(reference_sequence)

    # Initialize matrix with NaN
    matrix = pd.DataFrame(
        index=aa_order,
        columns=list(range(seq_len)),
        dtype=float,
    )

    # Fill in values from data
    for _, row in df.iterrows():
        pos = int(row[position_col])
        mut_aa = row[mut_aa_col]
        value = row[y_col]

        if mut_aa in aa_order and 0 <= pos < seq_len:
            matrix.loc[mut_aa, pos] = value

    # Optionally fill WT diagonal
    if fill_wt_value is not None:
        for pos, wt_aa in enumerate(reference_sequence):
            if wt_aa in aa_order:
                matrix.loc[wt_aa, pos] = fill_wt_value

    return matrix


def covered_position_range(
    exp_matrix: pd.DataFrame,
    reference: str,
) -> Optional[Tuple[int, int]]:
    """Inclusive ``(first, last)`` position-column range with real coverage.

    "Covered" means the measured matrix has at least one finite entry in a
    *non-WT* row of that column. Only the measured (experimental) matrix
    carries this signal: a model matrix scores every position regardless of
    what was assayed, so its non-WT cells are always populated and can't
    distinguish assayed from unassayed positions. The WT row is always
    dropped before checking for coverage, whether or not it happens to be
    filled: when :func:`to_dms_matrix`'s ``fill_wt_value`` (default ``None``)
    was left unset the row is already all-NaN and dropping it is a no-op,
    but when it *was* filled, a WT-only column would otherwise look covered
    despite nothing having actually been measured there.

    Used to trim both matrices' columns to the assayed region before
    plotting (long ProteinGym reference sequences often carry hundreds of
    entirely-unmeasured flanking positions). Interior all-missing columns
    *inside* the returned range are left in place -- only the leading and
    trailing empty flanks are trimmed.

    Args:
        exp_matrix: Measured (20 x L) DMS matrix, columns 0-indexed
            positions.
        reference: Reference (wild-type) AA sequence, indexed by the same
            0-indexed positions as ``exp_matrix``'s columns.

    Returns:
        ``(first, last)`` inclusive position labels of the covered range, or
        ``None`` if no column has non-WT coverage.
    """
    covered: List[int] = []
    for position in exp_matrix.columns:
        pos = int(position)  # columns are 0-indexed position integers
        column = exp_matrix[position]
        if 0 <= pos < len(reference):
            wt_aa = reference[pos]
            if wt_aa in column.index:
                column = column.drop(index=wt_aa)
        if column.notna().any():
            covered.append(pos)
    if not covered:
        return None
    return min(covered), max(covered)


def melt_dms_matrix(
    dms_df: pd.DataFrame,
    value_name: str = "value",
) -> pd.DataFrame:
    """Convert DMS-style matrix to long format.

    Inverse operation of to_dms_matrix(). Converts from wide format
    (amino acids as rows, positions as columns) to long format with
    one row per (amino acid, position) combination.

    Args:
        dms_df: DMS matrix with amino acid index and position columns.
        value_name: Name for the value column in output.

    Returns:
        Long-format DataFrame with columns:
        - aa: Amino acid (single letter)
        - position: Position (integer, 0-indexed)
        - {value_name}: The measurement value

    Example:
        >>> matrix = pd.DataFrame(
        ...     {"0": [1.0, 2.0], "1": [3.0, 4.0]},
        ...     index=["A", "C"]
        ... )
        >>> melt_dms_matrix(matrix, value_name="fitness")
           aa  position  fitness
        0   A         0      1.0
        1   A         1      3.0
        2   C         0      2.0
        3   C         1      4.0
    """
    # Reset index to make amino acid a column
    df = dms_df.reset_index()
    df = df.rename(columns={"index": "aa"})

    # Melt from wide to long
    melted = df.melt(
        id_vars=["aa"],
        var_name="position",
        value_name=value_name,
    )

    # Convert position to int
    melted["position"] = melted["position"].astype(int)

    # Sort by position, then amino acid
    melted = melted.sort_values(["position", "aa"]).reset_index(drop=True)

    return melted


def selection_factors_to_dms_matrix(
    selection_factors: Union[torch.Tensor, np.ndarray],
    reference_sequence: str,
    log_transform: bool = True,
    set_wt_nan: bool = True,
    aa_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convert model selection factors to DMS-style matrix.

    Takes selection factor output from a DASM model and converts it to the
    standard DMS matrix format (20 amino acids x L positions) for visualization
    and correlation analysis.

    Args:
        selection_factors: Array of shape (seq_len, 20) with selection factors
            for each position and amino acid.
        reference_sequence: Reference amino acid sequence. Used for determining
            sequence length and masking WT positions.
        log_transform: Whether to log-transform the selection factors.
            Defaults to True since selection factors are typically positive
            and log-transform makes them comparable to log-enrichment ratios.
        set_wt_nan: Whether to set wild-type positions to NaN. Defaults to True
            for fair comparison with experimental data where WT is often
            excluded or set to a reference value.
        aa_order: Order of amino acids for rows. Defaults to AA_STR_SORTED
            ("ACDEFGHIKLMNPQRSTVWY").

    Returns:
        DataFrame with shape (20, seq_len) where:
        - Index is amino acids in aa_order
        - Columns are position integers (0 to seq_len-1)
        - Values are (log) selection factors

    Raises:
        ValueError: If selection_factors shape doesn't match reference_sequence.

    Example:
        >>> # After getting selection factors from model:
        >>> heavy_sf, light_sf = model.selection_factors_of_aa_str(
        ...     (heavy_seq, light_seq)
        ... )
        >>> dms_matrix = selection_factors_to_dms_matrix(heavy_sf, heavy_seq)
        >>> dms_matrix.shape
        (20, 120)
    """
    if aa_order is None:
        aa_order = list(AA_STR_SORTED)

    # Convert to numpy if tensor
    if isinstance(selection_factors, torch.Tensor):
        sf_array = selection_factors.detach().cpu().numpy()
    else:
        sf_array = np.asarray(selection_factors)

    # Validate shape
    if sf_array.shape[0] != len(reference_sequence):
        raise ValueError(
            f"selection_factors has {sf_array.shape[0]} positions, "
            f"but reference_sequence has {len(reference_sequence)} positions"
        )
    if sf_array.shape[1] != 20:
        raise ValueError(
            f"selection_factors should have 20 amino acid columns, "
            f"got {sf_array.shape[1]}"
        )

    # Log-transform if requested
    if log_transform:
        # Replace zeros with NaN before log so they become NaN (caught by
        # np.isfinite filter) rather than -inf which would distort correlations.
        sf_array = np.where(sf_array > 0, sf_array, np.nan)
        sf_array = np.log(sf_array)

    # Create DataFrame (transpose to get 20 x L)
    matrix = pd.DataFrame(
        sf_array.T,
        index=aa_order,
        columns=list(range(len(reference_sequence))),
    )

    # Set WT positions to NaN if requested
    if set_wt_nan:
        matrix = mask_wt_to_nan(matrix, reference_sequence)

    return matrix


def exclude_wildtype_entries(
    melted: pd.DataFrame,
    reference_sequence: str,
) -> pd.DataFrame:
    """Remove wild-type amino acid entries from melted DMS data.

    Filters out rows where the amino acid matches the reference sequence at
    that position, so that only mutant entries remain. Useful for codon
    neighbor comparisons where WT entries should be excluded.

    Args:
        melted: Long-format DataFrame with "aa" and "position" columns,
            as produced by ``melt_dms_matrix``.
        reference_sequence: Reference amino acid sequence. Entries where
            ``aa == reference_sequence[position]`` are removed.

    Returns:
        Filtered DataFrame with WT entries removed.
    """
    wt_mask = melted.apply(
        lambda row: (
            row["position"] < len(reference_sequence)
            and row["aa"] == reference_sequence[int(row["position"])]
        ),
        axis=1,
    )
    result: pd.DataFrame = melted[~wt_mask]
    return result


def mask_wt_to_nan(
    matrix: pd.DataFrame,
    reference: str,
    aa_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return a copy of ``matrix`` with wild-type cells set to NaN.

    The wild-type cell holds a *true* ``0`` on both sides of a DMS
    comparison, not a placeholder: ``log(WT/WT) = 0`` experimentally
    (:func:`to_dms_matrix`'s ``fill_wt_value``), and
    ``zap_predictions_along_diagonal(..., fill=0.0)``
    (``dasm2/heads/_selection_factor_forward.py``) makes ``log f = 0`` the
    model's real output under the ``f_parent = 1`` gauge. The two sides
    therefore agree on that cell *a priori, by gauge, not by prediction*
    -- it cannot discriminate between models, and after per-site demeaning
    it becomes a large matched residual that re-imports site-level
    structure into what should be a mutation-level statistic. Masking to
    NaN is how a statistic (a correlation, a color-scale center, a
    median) declares that it excludes the wild-type cell, rather than
    inheriting the exclusion as a side effect of some other matrix's NaN
    pattern; it does not delete the value from the source matrix, which
    keeps it for display and for anything else that legitimately wants
    it.

    This is the cell-level counterpart to :func:`exclude_wildtype_entries`,
    which performs the same exclusion on the long (melted) format; that
    function's parameter is named ``reference_sequence``, while this one
    (matching :func:`covered_position_range`) uses the shorter ``reference``.

    Args:
        matrix: DMS matrix (20 x L) with amino acid index and position
            columns.
        reference: Reference (wild-type) amino-acid sequence, indexed by
            the same 0-indexed positions as ``matrix``'s columns.
        aa_order: Amino acids to check for a wild-type match. ``None``
            (the default) uses ``matrix``'s own index.

    Returns:
        Copy of ``matrix`` with every ``(wt_aa, position)`` cell set to
        NaN.
    """
    masked = matrix.copy()
    aa_set = set(aa_order) if aa_order is not None else set(matrix.index)
    for position, wt_aa in enumerate(reference):
        if wt_aa in aa_set and position in masked.columns:
            masked.loc[wt_aa, position] = np.nan
    return masked


def restrict_to_shared_cells(
    matrix1: pd.DataFrame,
    matrix2: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align two DMS matrices and NaN out every cell not finite in both.

    The matrix-shaped counterpart of :func:`aligned_finite_pairs` (which is
    implemented on top of this), for statistics that must *transform* the
    matrices before pooling them into value pairs -- per-site demeaning and
    per-site ranking (:mod:`dasm2.analysis.within_site_residual`). Those
    transforms are computed per column, so they need the compared cell set
    expressed as a matrix rather than as a flat pair of vectors: a site mean
    taken over cells that the correlation then drops is a mean of something
    other than the compared points, and two sides ranking different-sized
    cell sets produce ranks that are not comparable.

    Both returned frames carry the same shape, the same row/column labels,
    and -- by construction -- the identical NaN pattern, so a subsequent
    :func:`aligned_finite_pairs` call on them (directly or via
    :func:`correlate_dms_matrices`) is a no-op on the cell set.

    Args:
        matrix1: First DMS matrix (20 x L) with amino acid index.
        matrix2: Second DMS matrix (20 x L) with amino acid index.

    Returns:
        Tuple ``(restricted1, restricted2)`` on the intersection of rows and
        columns, with non-shared-finite cells set to NaN.

    Raises:
        ValueError: If the matrices share no rows or no columns at all.
    """
    common_rows = sorted(set(matrix1.index) & set(matrix2.index))
    common_cols = sorted(set(matrix1.columns) & set(matrix2.columns))

    if len(common_rows) == 0 or len(common_cols) == 0:
        raise ValueError(
            "Matrices have no overlapping amino acids or positions. "
            f"Matrix1 has {len(matrix1.index)} rows, {len(matrix1.columns)} cols. "
            f"Matrix2 has {len(matrix2.index)} rows, {len(matrix2.columns)} cols."
        )

    aligned1 = matrix1.loc[common_rows, common_cols]
    aligned2 = matrix2.loc[common_rows, common_cols]

    shared = np.isfinite(aligned1.to_numpy(dtype=float)) & np.isfinite(
        aligned2.to_numpy(dtype=float)
    )
    return aligned1.where(shared), aligned2.where(shared)


def aligned_finite_pairs(
    matrix1: pd.DataFrame,
    matrix2: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten two DMS matrices to their aligned, finite value pairs.

    Aligns on the intersection of rows (amino acids) and columns (positions),
    flattens, and keeps only cell pairs finite (non-NaN, non-inf) in *both*
    matrices. The single source of truth for the plotted/correlated point set,
    shared by :func:`correlate_dms_matrices` and the faceted
    :func:`~dasm2.data.functional.visualization.plot_dms_scatter` (so points and
    the annotated correlation can never drift apart). Unlike
    :func:`correlate_dms_matrices`, it imposes no minimum count -- callers that
    need one (e.g. a correlation) enforce it themselves -- so a sparse facet can
    still be scattered.

    The alignment and shared-finite masking are delegated to
    :func:`restrict_to_shared_cells`, so the flat point set here and the
    matrix-shaped cell set the within-site statistics transform over cannot
    disagree.

    Args:
        matrix1: First DMS matrix (20 x L) with amino acid index.
        matrix2: Second DMS matrix (20 x L) with amino acid index.

    Returns:
        Tuple ``(values1, values2)`` of equal-length 1-D arrays (possibly
        empty).

    Raises:
        ValueError: If the matrices share no rows or no columns at all.
    """
    restricted1, restricted2 = restrict_to_shared_cells(matrix1, matrix2)

    m1 = restricted1.to_numpy(dtype=float).flatten()
    m2 = restricted2.to_numpy(dtype=float).flatten()

    mask = np.isfinite(m1)
    return m1[mask], m2[mask]


def correlate_dms_matrices(
    matrix1: pd.DataFrame,
    matrix2: pd.DataFrame,
    method: CorrelationMethod = "spearman",
    exclude_wt: Optional[str] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute correlation between two DMS-style matrices.

    Aligns matrices by common rows (amino acids) and columns (positions),
    flattens them, removes NaN pairs, and computes correlation.

    Args:
        matrix1: First DMS matrix (20 x L) with amino acid index.
        matrix2: Second DMS matrix (20 x L) with amino acid index.
        method: Correlation method, one of ``"spearman"``, ``"pearson"`` or
            ``"kendall"``. Kendall's tau-b lives here rather than in a
            caller because the row/column alignment, the shared-finite cell
            masking (:func:`aligned_finite_pairs`, the single source of
            truth for the correlated point set) and the
            :data:`MIN_CORRELATION_PAIRS` guard are all method-independent
            -- a second correlation path would have to duplicate every one
            of them. Added for the within-site rank statistics
            (:mod:`dasm2.analysis.within_site_residual`); the one other
            ``kendalltau`` call in the codebase
            (``dasm2/analysis/codon_consistency.py``) is inline on plain
            arrays with no matrices to align.
        exclude_wt: Reference (wild-type) amino-acid sequence. When given,
            :func:`mask_wt_to_nan` is applied to both matrices before
            alignment, so wild-type cells are excluded from the
            correlation on both sides explicitly rather than depending on
            whichever matrix happens to already carry a NaN there (see
            :func:`mask_wt_to_nan` for why that cell can't discriminate
            between models). ``None`` (the default) applies no masking
            and preserves prior behavior exactly.

    Returns:
        Tuple of (correlation, values1, values2) where:
        - correlation: The correlation coefficient
        - values1: Flattened valid values from matrix1
        - values2: Flattened valid values from matrix2

    Raises:
        ValueError: If matrices have no overlapping positions/amino acids,
            fewer than 3 valid pairs, or invalid method.

    Example:
        >>> corr, vals1, vals2 = correlate_dms_matrices(
        ...     log_dasm_matrix, log_expr_matrix
        ... )
        >>> print(f"Spearman r = {corr:.3f}")
    """
    if method not in CORRELATION_METHODS:
        raise ValueError(
            f"method must be one of {sorted(CORRELATION_METHODS)}, got '{method}'"
        )

    if exclude_wt is not None:
        matrix1 = mask_wt_to_nan(matrix1, exclude_wt)
        matrix2 = mask_wt_to_nan(matrix2, exclude_wt)

    m1_valid, m2_valid = aligned_finite_pairs(matrix1, matrix2)

    if len(m1_valid) < MIN_CORRELATION_PAIRS:
        raise InsufficientDataError(
            f"Insufficient valid data points: {len(m1_valid)} < {MIN_CORRELATION_PAIRS} required "
            "for correlation"
        )

    # Compute correlation
    result = CORRELATION_METHODS[method](m1_valid, m2_valid)
    correlation = float(result.statistic if hasattr(result, "statistic") else result[0])

    return correlation, m1_valid, m2_valid
