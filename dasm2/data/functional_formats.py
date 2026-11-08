"""Utilities for converting functional data formats to standardized format.

This module provides conversion utilities for common functional data formats
(FLAb, multidms) to the standardized functional data format defined in
data-central (schemas/functional-data-v1.yaml).

The standardized format uses:
- sequence_* columns for amino acid sequences
- y_* columns for response variables
- x_* columns for per-row covariates (optional)
- group_id for grouping sequences with shared reference
- is_reference for marking reference/wildtype sequences

Example usage:
    # Convert FLAb binding data
    df = from_flab(
        "binding_data.csv",
        sequence_cols=["heavy", "light"],
        y_col="Kd (flow) [M]",
    )

    # Convert multidms single-mutant data
    df = from_multidms("dms_output.csv", wt_sequence="MVLSPAD...")
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Self, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import stats  # type: ignore[import-untyped]

from dasm2.codon_table import CODON_NEIGHBOR_MATRIX
from dasm2.heads.types import HeadType
from dasm2.sequence.constants import AA_STR_SORTED, AA_STR_TO_IDX, CODONS
from dasm2.sequence.conversion import iter_codons

# Minimum number of valid data points required to compute correlation.
# Statistical correlation requires at least 3 points to be meaningful.
MIN_CORRELATION_SAMPLES = 3


class InsufficientDataError(ValueError):
    """Too few valid cells to define a correlation.

    A ``ValueError`` subclass, so existing handlers still catch it. Callers that
    expect this one failure (scoring a sparse domain) catch this class and let every
    other ``ValueError`` propagate.
    """


# The column vocabulary of the standardized functional format (data-central
# schemas/functional-data-v1.yaml), named here in the module that writes it so
# consumers resolve the names rather than re-spelling them.
SEQUENCE_VARIANT_COL = "sequence_variant"
GROUP_ID_COL = "group_id"
IS_REFERENCE_COL = "is_reference"
POSITION_COL = "x_position"
WT_AA_COL = "x_wt_aa"
MUT_AA_COL = "x_mut_aa"

# The y column `from_multidms` writes. multidms CSVs have a fixed header, so
# multidms manifests declare no `y_col` to rename (#989) and consumers resolve
# the output column through this constant instead.
FUNCTIONAL_EFFECT_Y_COL = "y_functional_effect"


class SelectionFactorModel(Protocol):
    """Protocol for models that compute per-position selection factors.

    Implementations expose :py:meth:`eval` (to drop dropout / put modules into
    inference mode) and :py:meth:`selection_factors_of_aa_str` (the per-position
    ``L x 20`` selection-factor tensor).

    The default ``head_type`` matches
    :meth:`dasm2.models.core.base.CoreModel.selection_factors_of_aa_str` so a
    caller that omits the argument gets the same head everywhere.
    """

    def eval(self) -> Self:
        """Set the model to evaluation mode (nn.Module-style return-self)."""
        ...

    def selection_factors_of_aa_str(
        self,
        sequences: str,
        head_type: HeadType = HeadType.SELECTION_FACTOR_HEAD,
        *,
        distance: Optional[float] = None,
    ) -> List[torch.Tensor]:
        """Compute selection factors for amino acid sequences.

        Args:
            sequences: Single sequence, or paired sequences separated by ``"^"``.
            head_type: Which prediction head to use; defaults to
                ``HeadType.SELECTION_FACTOR_HEAD``.

        Returns:
            List of selection factor tensors, one per chain; each has shape
            ``(seq_len, 20)``.
        """
        ...


def from_flab(
    csv_path: "str | Path",
    sequence_cols: Sequence[str],
    y_col: str,
    group_id_col: Optional[str] = None,
    y_col_rename: Optional[str] = None,
) -> pd.DataFrame:
    """Convert FLAb format to standardized functional data format.

    FLAb (Fitness Landscapes for Antibodies) datasets contain antibody sequences
    with associated therapeutic property measurements. This function converts
    them to the standardized format with sequence_* and y_* columns.

    Note: FLAb data typically does not have an explicit reference/wildtype row.
    Reference sequences are usually computed as site-by-site consensus and may
    not exist as actual rows in the data. Therefore, this function does not
    produce an is_reference column.

    Args:
        csv_path: Path to FLAb CSV file.
        sequence_cols: Column names containing amino acid sequences.
            For single-chain: ["sequence"] or ["heavy"]
            For two-chain: ["heavy", "light"]
        y_col: Column name containing the response variable (e.g., "Kd (flow) [M]").
        group_id_col: Optional column name to copy as group_id (e.g., for
            grouping by antibody lineage).
        y_col_rename: Optional name for the output y_ column. Defaults to y_col
            with special characters replaced.

    Returns:
        DataFrame in standardized functional data format with columns:
        - sequence_{name} for each sequence column
        - y_{name} for the response variable
        - group_id (if group_id_col provided)

    Raises:
        ValueError: If required columns are missing or no valid data found.

    Example:
        >>> df = from_flab(
        ...     "kirby_binding.csv",
        ...     sequence_cols=["VH", "VL"],
        ...     y_col="KD",
        ... )
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Validate required columns exist
    missing_seq_cols = [col for col in sequence_cols if col not in df.columns]
    if missing_seq_cols:
        raise ValueError(
            f"Missing sequence columns: {missing_seq_cols}. "
            f"Available columns: {list(df.columns)}"
        )
    if y_col not in df.columns:
        raise ValueError(
            f"Missing response column '{y_col}'. Available columns: {list(df.columns)}"
        )

    # Build output dataframe
    output_data: Dict[str, List] = {}

    # Rename sequence columns
    for seq_col in sequence_cols:
        output_data[f"sequence_{seq_col}"] = df[seq_col].tolist()

    # Rename y column (clean up special characters)
    if y_col_rename is None:
        y_col_rename = _sanitize_column_name(y_col)
    output_data[f"y_{y_col_rename}"] = df[y_col].tolist()

    # Copy group_id column if specified
    if group_id_col is not None:
        if group_id_col not in df.columns:
            raise ValueError(
                f"group_id column '{group_id_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        output_data[GROUP_ID_COL] = df[group_id_col].tolist()

    result = pd.DataFrame(output_data)
    if len(result) == 0:
        raise ValueError(f"No valid data found in {csv_path}")

    return result


# Non-substitution mutation codes that appear in multidms output. A DASM2
# variant is a same-length amino acid string, so neither can be represented:
# `*` truncates the protein and `-` shortens it. The real Haddox and Dadonaite
# CSVs carry both (BA.1: 185 `*`, 101 `-`), and every one of them reached the
# `AA_STR_SORTED` membership check and raised until they were filtered here.
NON_SUBSTITUTION_MUTANT_CODES = ("*", "-")


def _filter_valid_mutations(
    df: pd.DataFrame,
    wt_column: str,
    mutant_column: str,
) -> pd.DataFrame:
    """Filter DataFrame to valid non-WT mutations, excluding non-substitutions.

    Args:
        df: DataFrame with mutation data.
        wt_column: Column containing WT amino acid.
        mutant_column: Column containing mutant amino acid.

    Returns:
        Filtered DataFrame with only single amino acid substitutions: stop
        codons (``*``), deletions (``-``), and no-ops are dropped.
    """
    return df[
        ~df[mutant_column].isin(NON_SUBSTITUTION_MUTANT_CODES)
        & ~df[wt_column].isin(NON_SUBSTITUTION_MUTANT_CODES)
        & (df[mutant_column] != df[wt_column])
    ].copy()


def from_multidms(
    csv_path: "str | Path",
    wt_sequence: str,
    wt_column: str = "wildtype",
    site_column: str = "site",
    mutant_column: str = "mutant",
    func_score_column: str = "functional_effect",
    strict: bool = True,
) -> pd.DataFrame:
    """Convert multidms sparse format to standardized functional data format.

    Multidms output contains single-mutant functional scores in a sparse format
    with (position, wt_aa, mut_aa, score) per row. This function converts to
    the standardized format by constructing full mutant sequences.

    Note: The WT reference row is assigned a functional score of 0.0, which is
    the standard convention in multidms where scores represent log-fold-change
    relative to wildtype.

    Args:
        csv_path: Path to multidms output CSV file.
        wt_sequence: Wild-type amino acid sequence.
        wt_column: Column containing WT amino acid at each position.
        site_column: Column containing 1-indexed position.
        mutant_column: Column containing mutant amino acid.
        func_score_column: Column containing functional scores.
        strict: When True (default), raise if the CSV's WT amino acid disagrees
            with ``wt_sequence`` at a site. When False, emit the row anyway and
            warn with a count. Needed for multidms output, which defines all
            mutations relative to a single reference homolog, so the CSV
            wildtype column does not match the true per-homolog founder
            sequence (#414). Position bounds and amino-acid validity are always
            enforced.

    Returns:
        DataFrame in standardized functional data format with columns:
        - sequence_variant: Full mutant sequence
        - y_functional_effect: Functional score
        - is_reference: True for WT row
        - group_id: Always "wt" (single reference group)
        - x_position: 0-indexed mutation position
        - x_wt_aa: Wild-type amino acid at the mutation site, taken from
          wt_sequence (not the CSV column) so it always describes the
          reconstructed sequence_variant
        - x_mut_aa: Mutant amino acid

    Raises:
        ValueError: If required columns missing or validation fails.

    Example:
        >>> df = from_multidms(
        ...     "dms_output.csv",
        ...     wt_sequence="MVLSPADKTN...",
        ... )
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Validate required columns exist
    required_cols = [wt_column, site_column, mutant_column, func_score_column]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {list(df.columns)}"
        )

    # Filter out stop codons and WT entries
    mutations = _filter_valid_mutations(df, wt_column, mutant_column)

    output_rows = []
    skipped_missing_score = 0
    wt_mismatches = 0
    skipped_noop = 0

    # Add WT as reference row first
    output_rows.append(
        {
            SEQUENCE_VARIANT_COL: wt_sequence,
            FUNCTIONAL_EFFECT_Y_COL: 0.0,  # WT score is 0 by convention
            IS_REFERENCE_COL: True,
            GROUP_ID_COL: "wt",
            POSITION_COL: None,
            WT_AA_COL: None,
            MUT_AA_COL: None,
        }
    )

    # Process mutations
    for _, row in mutations.iterrows():
        wt_aa = str(row[wt_column]).strip()
        mut_aa = str(row[mutant_column]).strip()
        position = int(row[site_column]) - 1  # Convert to 0-indexed

        # Validate position bounds
        if position < 0 or position >= len(wt_sequence):
            raise ValueError(
                f"Position {position + 1} (1-indexed) out of bounds. "
                f"WT sequence has {len(wt_sequence)} positions."
            )

        # Validate amino acids before comparing them: a malformed letter is a
        # hard error whatever `strict` says, and checking it first means the
        # raised error names the root cause instead of a downstream mismatch.
        if wt_aa not in AA_STR_SORTED:
            raise ValueError(f"Unknown WT amino acid '{wt_aa}'")
        if mut_aa not in AA_STR_SORTED:
            raise ValueError(f"Unknown mutant amino acid '{mut_aa}'")

        # Validate WT matches. multidms defines mutations relative to one
        # reference homolog, so a per-homolog founder legitimately disagrees.
        if wt_sequence[position] != wt_aa:
            if strict:
                raise ValueError(
                    f"WT amino acid mismatch at position {position + 1}. "
                    f"CSV has '{wt_aa}', but WT sequence has '{wt_sequence[position]}'"
                )
            wt_mismatches += 1

        # Only reachable when strict=False and the CSV's wildtype disagrees with
        # the reference in a way that happens to match the mutant letter. Such a
        # row would duplicate the reference sequence while carrying a nonzero
        # score. _filter_valid_mutations only drops CSV-self-consistent no-ops
        # (mut_aa == wt_aa), not this reference-relative case.
        if mut_aa == wt_sequence[position]:
            skipped_noop += 1
            continue

        func_score = row[func_score_column]
        if pd.isna(func_score):
            skipped_missing_score += 1
            continue

        # Construct full mutant sequence
        mutant_sequence = wt_sequence[:position] + mut_aa + wt_sequence[position + 1 :]

        output_rows.append(
            {
                SEQUENCE_VARIANT_COL: mutant_sequence,
                FUNCTIONAL_EFFECT_Y_COL: float(func_score),
                IS_REFERENCE_COL: False,
                GROUP_ID_COL: "wt",
                POSITION_COL: position,
                # From the reference, not the CSV: under strict=False these
                # disagree, and this column must describe sequence_variant.
                WT_AA_COL: wt_sequence[position],
                MUT_AA_COL: mut_aa,
            }
        )

    if wt_mismatches > 0:
        warnings.warn(
            f"Relaxed {wt_mismatches} WT mismatch(es) between the CSV "
            f"'{wt_column}' column and the reference sequence in {csv_path.name}. "
            f"Variants were reconstructed from the reference sequence.",
            UserWarning,
            stacklevel=2,
        )

    if skipped_noop > 0:
        warnings.warn(
            f"Skipped {skipped_noop} no-op mutation(s) in {csv_path.name} where "
            f"the mutant amino acid already matches the reference sequence.",
            UserWarning,
            stacklevel=2,
        )

    # Warn about skipped rows
    if skipped_missing_score > 0:
        warnings.warn(
            f"Skipped {skipped_missing_score} rows with missing functional scores "
            f"in {csv_path.name}",
            UserWarning,
            stacklevel=2,
        )

    result = pd.DataFrame(output_rows)
    if len(result) <= 1:  # Only WT row
        raise ValueError(f"No valid mutations found in {csv_path}")

    return result


def predict_for_functional_dataset(
    model: SelectionFactorModel,
    df: pd.DataFrame,
    sequence_col: str,
    head_type: HeadType = HeadType.SELECTION_FACTOR_HEAD,
    output_col: str = "y_predicted_selection",
) -> pd.DataFrame:
    """Add selection factor predictions to functional dataset.

    For each sequence in the dataset, compute the model's predicted selection
    factors and derive a scalar fitness proxy. For single-mutant data (with
    x_position and x_mut_aa columns), returns the log selection factor at the
    specific mutation. For full sequences, returns the sum of log selection
    factors across all positions.

    The sum of log selection factors approximates log-fitness under the assumption
    that selection acts independently at each position. This is a standard
    approximation in mutation-selection models.

    Args:
        model: Trained DASM2 model with selection_factors_of_aa_str method.
        df: DataFrame with sequence column and optionally x_position/x_mut_aa.
        sequence_col: Column name containing amino acid sequences.
        head_type: Head type to use for predictions.
        output_col: Name for the output prediction column.

    Returns:
        DataFrame with added prediction column.

    Raises:
        ValueError: If sequence column not found.

    Example:
        >>> df = predict_for_functional_dataset(
        ...     model, df, sequence_col="sequence_variant"
        ... )
    """
    if sequence_col not in df.columns:
        raise ValueError(
            f"Sequence column '{sequence_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Check if this is single-mutant data
    has_mutation_info = POSITION_COL in df.columns and MUT_AA_COL in df.columns

    predictions = []
    failed_predictions = 0
    model.eval()

    for _, row in df.iterrows():
        seq = row[sequence_col]

        if pd.isna(seq) or not isinstance(seq, str) or len(seq) == 0:
            failed_predictions += 1
            predictions.append(np.nan)
            continue

        # Get selection factors from model
        selection_factors_list = model.selection_factors_of_aa_str(seq, head_type)

        if len(selection_factors_list) == 0:
            failed_predictions += 1
            predictions.append(np.nan)
            continue

        # Extract selection factors for single sequence (multi-sequence not yet supported)
        if len(selection_factors_list) != 1:
            raise ValueError(
                f"Multi-sequence predictions not supported, got {len(selection_factors_list)} sequences"
            )
        selection_factors = selection_factors_list[0]  # [seq_len, 20]

        if has_mutation_info and pd.notna(row.get(POSITION_COL)):
            # Single-mutant: get specific selection factor
            position = int(row[POSITION_COL])
            mut_aa = row[MUT_AA_COL]
            if position < len(selection_factors) and mut_aa in AA_STR_TO_IDX:
                aa_idx = AA_STR_TO_IDX[mut_aa]
                # Return log selection factor for the specific mutation
                sf = selection_factors[position, aa_idx].item()
                predictions.append(np.log(sf) if sf > 0 else -np.inf)
            else:
                failed_predictions += 1
                predictions.append(np.nan)
        else:
            # Full sequence: sum of log selection factors at each position
            # Use the selection factor for the actual amino acid at each position
            log_sf_sum = 0.0
            for pos, aa_char in enumerate(seq):
                if pos < len(selection_factors) and aa_char in AA_STR_TO_IDX:
                    aa_idx = AA_STR_TO_IDX[aa_char]
                    sf = selection_factors[pos, aa_idx].item()
                    log_sf_sum += np.log(sf) if sf > 0 else -np.inf
            predictions.append(log_sf_sum)

    # Warn about failed predictions
    if failed_predictions > 0:
        warnings.warn(
            f"Model failed to predict {failed_predictions}/{len(df)} sequences",
            UserWarning,
            stacklevel=2,
        )

    result = df.copy()
    result[output_col] = predictions
    return result


def _validate_correlation_columns(
    df: pd.DataFrame, response_col: str, prediction_col: str
) -> None:
    """Validate that required columns exist for correlation computation.

    Args:
        df: DataFrame to validate.
        response_col: Column name for actual functional response.
        prediction_col: Column name for model predictions.

    Raises:
        ValueError: If any required column is not found.
    """
    for col in [response_col, prediction_col]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Available columns: {list(df.columns)}"
            )


def _compute_correlation_for_subset(
    subset_df: pd.DataFrame,
    response_col: str,
    prediction_col: str,
    corr_func: Any,
    context: Optional[str] = None,
) -> float:
    """Compute correlation for a DataFrame subset, handling NaN and invalid values.

    Args:
        subset_df: DataFrame subset to compute correlation for.
        response_col: Column name for actual functional response.
        prediction_col: Column name for model predictions.
        corr_func: Scipy correlation function (spearmanr or pearsonr).
        context: Optional context string for error messages (e.g., stratum name).

    Returns:
        Correlation coefficient.

    Raises:
        ValueError: If fewer than MIN_CORRELATION_SAMPLES valid data points.
    """
    valid_mask = (
        pd.notna(subset_df[response_col])
        & pd.notna(subset_df[prediction_col])
        & np.isfinite(subset_df[response_col])
        & np.isfinite(subset_df[prediction_col])
    )
    valid_df = subset_df[valid_mask]

    if len(valid_df) < MIN_CORRELATION_SAMPLES:
        context_msg = f" for {context}" if context else ""
        raise InsufficientDataError(
            f"Insufficient valid data points{context_msg}: "
            f"{len(valid_df)} < {MIN_CORRELATION_SAMPLES} required for correlation"
        )

    result = corr_func(valid_df[response_col], valid_df[prediction_col])
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def _get_correlation_function(method: str) -> Any:
    """Get scipy correlation function for the specified method.

    Args:
        method: Correlation method ("spearman" or "pearson").

    Returns:
        Scipy correlation function.

    Raises:
        ValueError: If method is not recognized.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got '{method}'")
    return stats.spearmanr if method == "spearman" else stats.pearsonr


def compute_functional_correlation(
    df: pd.DataFrame,
    response_col: str,
    prediction_col: str,
    method: str = "spearman",
) -> float:
    """Compute correlation between functional response and predictions.

    Args:
        df: DataFrame with response and prediction columns.
        response_col: Column name for actual functional response.
        prediction_col: Column name for model predictions.
        method: Correlation method ("spearman" or "pearson").

    Returns:
        Correlation coefficient.

    Raises:
        ValueError: If columns not found, invalid method, or fewer than
            MIN_CORRELATION_SAMPLES valid data points.

    Example:
        >>> corr = compute_functional_correlation(
        ...     df, "y_functional_effect", "y_predicted_selection"
        ... )
    """
    _validate_correlation_columns(df, response_col, prediction_col)
    corr_func = _get_correlation_function(method)
    return _compute_correlation_for_subset(df, response_col, prediction_col, corr_func)


def compute_functional_correlation_stratified(
    df: pd.DataFrame,
    response_col: str,
    prediction_col: str,
    stratify_by: str,
    method: str = "spearman",
) -> Dict[Any, float]:
    """Compute stratified correlation between functional response and predictions.

    Computes separate correlations for each unique value of the stratification column.

    Args:
        df: DataFrame with response and prediction columns.
        response_col: Column name for actual functional response.
        prediction_col: Column name for model predictions.
        stratify_by: Column name to stratify correlation by.
        method: Correlation method ("spearman" or "pearson").

    Returns:
        Dict mapping strata values to correlation coefficients.

    Raises:
        ValueError: If columns not found, invalid method, or fewer than
            MIN_CORRELATION_SAMPLES valid data points in any stratum.

    Example:
        >>> corrs = compute_functional_correlation_stratified(
        ...     df, "y_functional_effect", "y_predicted_selection",
        ...     stratify_by="is_codon_neighbor"
        ... )
        >>> print(corrs[True], corrs[False])  # Correlations by codon neighbor status
    """
    _validate_correlation_columns(df, response_col, prediction_col)
    if stratify_by not in df.columns:
        raise ValueError(
            f"Stratification column '{stratify_by}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    corr_func = _get_correlation_function(method)
    correlations: Dict[Any, float] = {}

    for stratum_value, stratum_df in df.groupby(stratify_by):
        correlations[stratum_value] = _compute_correlation_for_subset(
            stratum_df,
            response_col,
            prediction_col,
            corr_func,
            context=f"{stratify_by}={stratum_value}",
        )

    return correlations


def add_codon_neighbor_status(
    df: pd.DataFrame,
    reference_nt_sequence: str,
    position_col: str = POSITION_COL,
    mutant_aa_col: str = MUT_AA_COL,
    output_col: str = "is_codon_neighbor",
) -> pd.DataFrame:
    """Add codon neighbor status column to single-mutant functional data.

    For each mutation, determines whether the mutant amino acid is reachable
    from the reference codon at that position via a single nucleotide
    substitution. This is important for analysis because codon-neighbor
    mutations are more likely under neutral evolution.

    Args:
        df: DataFrame with mutation position and amino acid columns.
        reference_nt_sequence: Nucleotide sequence of the reference (must be
            a multiple of 3 in length).
        position_col: Column containing 0-indexed mutation positions.
        mutant_aa_col: Column containing mutant amino acid (single letter).
        output_col: Name for the output boolean column.

    Returns:
        DataFrame with added codon neighbor status column.

    Raises:
        ValueError: If required columns missing or sequence invalid.

    Example:
        >>> df = add_codon_neighbor_status(
        ...     df,
        ...     reference_nt_sequence="ATGAAACCC...",
        ... )
        >>> # Now df has 'is_codon_neighbor' column
    """
    # Validate inputs
    if position_col not in df.columns:
        raise ValueError(
            f"Position column '{position_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if mutant_aa_col not in df.columns:
        raise ValueError(
            f"Mutant AA column '{mutant_aa_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if len(reference_nt_sequence) % 3 != 0:
        raise ValueError(
            f"Reference nucleotide sequence length ({len(reference_nt_sequence)}) "
            "must be a multiple of 3"
        )

    # Build codon neighbor indicator for the reference sequence
    # Shape: (num_codons, 20) - True if AA is reachable via single nt substitution
    codons = list(iter_codons(reference_nt_sequence))
    num_codons = len(codons)

    # Get codon indices for the reference sequence
    codon_indices = [CODONS.index(codon) if codon in CODONS else -1 for codon in codons]

    is_codon_neighbor: List[Optional[bool]] = []

    for _, row in df.iterrows():
        position_val = row.get(position_col)
        mut_aa = row.get(mutant_aa_col)

        # Handle missing values or reference row
        if position_val is None or pd.isna(position_val) or pd.isna(mut_aa):
            is_codon_neighbor.append(None)
            continue

        position = int(position_val)

        # Validate position
        if position < 0 or position >= num_codons:
            is_codon_neighbor.append(None)
            continue

        # Get codon index at this position
        codon_idx = codon_indices[position]
        if codon_idx < 0:
            # Invalid codon (contains N or other ambiguous nucleotide)
            is_codon_neighbor.append(None)
            continue

        # Check if mutant AA is a codon neighbor
        if mut_aa not in AA_STR_TO_IDX:
            is_codon_neighbor.append(None)
            continue

        aa_idx = AA_STR_TO_IDX[mut_aa]
        is_neighbor = bool(CODON_NEIGHBOR_MATRIX[codon_idx, aa_idx].item())
        is_codon_neighbor.append(is_neighbor)

    result = df.copy()
    result[output_col] = is_codon_neighbor
    return result


def _sanitize_column_name(name: str) -> str:
    """Convert column name to safe format for functional data schema.

    Replaces special characters and spaces with underscores.

    Args:
        name: Original column name.

    Returns:
        Sanitized column name.
    """
    # Replace common special characters with underscores
    for char in " ()[]{}/<>!@#$%^&*+=|\\:;\"',.?":
        name = name.replace(char, "_")
    # Remove consecutive underscores
    while "__" in name:
        name = name.replace("__", "_")
    # Remove leading/trailing underscores
    name = name.strip("_")
    return name.lower()
