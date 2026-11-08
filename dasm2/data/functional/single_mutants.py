"""Single-mutant filtering utilities.

Provides functions for filtering sequence data to single-mutant variants
and extracting mutation metadata.
"""

from typing import Any, List

import pandas as pd

from dasm2.data.functional_formats import MUT_AA_COL, POSITION_COL, WT_AA_COL


def filter_single_mutants(
    df: pd.DataFrame,
    sequence_col: str,
    reference_sequence: str,
    position_col: str = POSITION_COL,
    wt_aa_col: str = WT_AA_COL,
    mut_aa_col: str = MUT_AA_COL,
) -> pd.DataFrame:
    """Filter to single-mutant variants and add mutation metadata.

    Keeps only rows where sequence differs from reference at exactly one position.
    Adds columns identifying the mutation position and amino acids.

    Args:
        df: DataFrame with sequence column.
        sequence_col: Column name containing amino acid sequences.
        reference_sequence: Reference/wildtype sequence to compare against.
        position_col: Output column name for mutation position (0-indexed).
        wt_aa_col: Output column name for wildtype amino acid.
        mut_aa_col: Output column name for mutant amino acid.

    Returns:
        Filtered DataFrame with mutation metadata columns added.
        Original rows are preserved (not copied) where applicable.

    Raises:
        ValueError: If sequence column is missing or sequence lengths
            don't match reference.

    Example:
        >>> df = pd.DataFrame({"seq": ["AAA", "ABA", "ABB"]})
        >>> result = filter_single_mutants(df, "seq", "AAA")
        >>> len(result)  # Only "ABA" is single-mutant
        1
        >>> result["x_position"].iloc[0]
        1
        >>> result["x_wt_aa"].iloc[0]
        'A'
        >>> result["x_mut_aa"].iloc[0]
        'B'
    """
    if sequence_col not in df.columns:
        raise ValueError(
            f"Sequence column '{sequence_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    ref_len = len(reference_sequence)
    # Index type depends on DataFrame; use Any to preserve original index type
    single_mutant_indices: List[Any] = []
    positions: List[int] = []
    wt_aas: List[str] = []
    mut_aas: List[str] = []

    for idx, row in df.iterrows():
        seq = row[sequence_col]

        # Skip invalid sequences
        if not isinstance(seq, str) or len(seq) == 0:
            continue

        if len(seq) != ref_len:
            raise ValueError(
                f"Sequence at index {idx} has length {len(seq)}, "
                f"but reference has length {ref_len}"
            )

        # Find positions where sequence differs from reference
        diff_positions = _find_differences(seq, reference_sequence)

        # Keep only single-mutant sequences
        if len(diff_positions) == 1:
            pos = diff_positions[0]
            single_mutant_indices.append(idx)
            positions.append(pos)
            wt_aas.append(reference_sequence[pos])
            mut_aas.append(seq[pos])

    # Create filtered DataFrame
    result = df.loc[single_mutant_indices].copy()
    result[position_col] = positions
    result[wt_aa_col] = wt_aas
    result[mut_aa_col] = mut_aas

    return result


def _find_differences(seq: str, reference: str) -> List[int]:
    """Find positions where two sequences differ.

    Args:
        seq: First sequence.
        reference: Second sequence (same length as seq).

    Returns:
        List of 0-indexed positions where sequences differ.
    """
    return [i for i in range(len(seq)) if seq[i] != reference[i]]
