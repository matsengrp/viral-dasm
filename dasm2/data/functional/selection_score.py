"""Selection score utilities for multi-mutant functional analysis.

Provides functions for computing selection scores from model predictions
for sequences with multiple mutations relative to a reference. This is
useful for datasets like Shanehsazzadeh where variants have multiple
mutations rather than single-mutant DMS data.

The selection score is the sum of log selection factors at mutated positions,
which approximates log-fitness under the assumption of independent selection
at each position.

Example:
    # Compute selection score for each variant
    log_sf = model.selection_factors_of_aa_str(consensus)[0]
    log_sf_matrix = torch.log(log_sf).T  # Shape: [20, seq_len]

    for seq in variants:
        diffs = compute_sequence_differences(consensus, seq)
        score = compute_selection_score(log_sf_matrix, diffs)
"""

import logging
from typing import TYPE_CHECKING, List, Union

import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd


def compute_sequence_differences(reference: str, variant: str) -> List[str]:
    """Compute differences between reference and variant sequences.

    Returns a list of difference strings in the format "X{pos}Y" where:
    - X is the reference amino acid
    - {pos} is the 1-indexed position
    - Y is the variant amino acid

    Args:
        reference: Reference amino acid sequence.
        variant: Variant amino acid sequence.

    Returns:
        List of difference strings (e.g., ["A105G", "T200S"]).

    Raises:
        ValueError: If sequences have different lengths.

    Example:
        >>> compute_sequence_differences("AAAA", "ABAA")
        ['A2B']
        >>> compute_sequence_differences("MVLSP", "MVKSP")
        ['L3K']
    """
    if len(reference) != len(variant):
        raise ValueError(
            f"Sequences have different lengths: "
            f"reference={len(reference)}, variant={len(variant)}"
        )

    differences = []
    for i, (ref_aa, var_aa) in enumerate(zip(reference, variant)):
        if ref_aa != var_aa:
            # Use 1-indexed position to match common biology convention
            differences.append(f"{ref_aa}{i + 1}{var_aa}")

    return differences


def compute_selection_score(
    log_selection_factors: Union[torch.Tensor, "torch.Tensor"],
    differences: List[str],
    aa_order: str = "ACDEFGHIKLMNPQRSTVWY",
) -> float:
    """Compute selection score from log selection factors and mutation list.

    The selection score is the sum of log selection factors at mutated positions.
    For each mutation X{pos}Y, the score contribution is log_sf[Y_idx, pos-1].

    Args:
        log_selection_factors: Log selection factor matrix with shape [20, seq_len].
            Rows correspond to amino acids in aa_order, columns to positions.
        differences: List of difference strings from compute_sequence_differences().
        aa_order: Amino acid ordering for the selection factor matrix.
            Defaults to alphabetical "ACDEFGHIKLMNPQRSTVWY".

    Returns:
        Sum of log selection factors at mutated positions.

    Raises:
        ValueError: If difference format is invalid or amino acid not in aa_order.

    Example:
        >>> # log_sf has shape [20, 100] for 100-position sequence
        >>> differences = ["A50G", "T75S"]
        >>> score = compute_selection_score(log_sf, differences)
    """
    if len(differences) == 0:
        return 0.0

    score = 0.0
    for diff in differences:
        # Parse difference string: "X{pos}Y"
        if len(diff) < 3:
            raise ValueError(f"Invalid difference format: '{diff}'")

        # First character is reference AA (not used in score)
        # Last character is mutant AA
        # Middle characters are the position
        mut_aa = diff[-1]
        try:
            pos = int(diff[1:-1])
        except ValueError:
            raise ValueError(f"Invalid position in difference: '{diff}'")

        # Validate amino acid
        if mut_aa not in aa_order:
            raise ValueError(f"Unknown amino acid '{mut_aa}' in difference '{diff}'")

        # Get log selection factor for mutant at this position
        aa_idx = aa_order.index(mut_aa)
        pos_idx = pos - 1  # Convert to 0-indexed

        if pos_idx < 0 or pos_idx >= log_selection_factors.shape[1]:
            raise ValueError(
                f"Position {pos} out of bounds for selection factor matrix "
                f"with {log_selection_factors.shape[1]} positions"
            )

        score += log_selection_factors[aa_idx, pos_idx].item()

    return score


def add_selection_scores(
    df: "pd.DataFrame",
    log_selection_factors: torch.Tensor,
    reference_sequence: str,
    sequence_col: str,
    output_col: str = "dasm_score",
    aa_order: str = "ACDEFGHIKLMNPQRSTVWY",
) -> "pd.DataFrame":
    """Add selection score column to DataFrame.

    For each sequence in the DataFrame, computes the selection score relative
    to the reference sequence using the provided log selection factors.

    Args:
        df: DataFrame with sequence column.
        log_selection_factors: Log selection factor matrix [20, seq_len].
        reference_sequence: Reference sequence for computing differences.
        sequence_col: Name of column containing variant sequences.
        output_col: Name for the output score column.
        aa_order: Amino acid ordering for the selection factor matrix.

    Returns:
        DataFrame with added selection score column.

    Raises:
        ValueError: If sequence column not found.

    Example:
        >>> df = add_selection_scores(
        ...     df, log_sf, consensus, sequence_col="heavy"
        ... )
    """

    if sequence_col not in df.columns:
        raise ValueError(
            f"Sequence column '{sequence_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    scores = []
    for seq in df[sequence_col]:
        if not isinstance(seq, str) or len(seq) == 0:
            scores.append(float("nan"))
            continue

        if len(seq) != len(reference_sequence):
            scores.append(float("nan"))
            continue

        try:
            diffs = compute_sequence_differences(reference_sequence, seq)
            score = compute_selection_score(log_selection_factors, diffs, aa_order)
            scores.append(score)
        except (ValueError, IndexError) as e:
            logger.warning("Selection score computation failed for %r: %s", seq, e)
            scores.append(float("nan"))

    result = df.copy()
    result[output_col] = scores
    return result
