"""Consensus sequence computation utilities.

Provides functions for computing site-by-site consensus sequences from
variant data using majority voting.
"""

import logging
from collections import Counter
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def compute_consensus(df: pd.DataFrame, sequence_col: str) -> str:
    """Compute site-by-site consensus sequence using majority voting.

    For each position in the sequences, determines the most common amino acid
    across all sequences. In case of ties, returns the amino acid that
    appears first alphabetically.

    Args:
        df: DataFrame containing sequences.
        sequence_col: Column name with amino acid sequences.

    Returns:
        Consensus sequence string.

    Raises:
        ValueError: If sequences have different lengths, column is missing,
            or DataFrame is empty.

    Example:
        >>> df = pd.DataFrame({"seq": ["AAA", "ABA", "ACA"]})
        >>> compute_consensus(df, "seq")
        'AAA'
    """
    if sequence_col not in df.columns:
        raise ValueError(
            f"Sequence column '{sequence_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if len(df) == 0:
        raise ValueError("Cannot compute consensus from empty DataFrame")

    sequences: List[str] = df[sequence_col].tolist()

    # Filter out any None/NaN values
    sequences = [s for s in sequences if isinstance(s, str) and len(s) > 0]

    if len(sequences) == 0:
        raise ValueError("No valid sequences found in column")

    # Validate all sequences have same length
    lengths = set(len(s) for s in sequences)
    if len(lengths) > 1:
        raise ValueError(
            f"Sequences have different lengths: {sorted(lengths)}. "
            "All sequences must have the same length for consensus computation."
        )

    seq_len = lengths.pop()

    # Compute consensus position-by-position
    consensus_chars: List[str] = []
    for pos in range(seq_len):
        # Count amino acids at this position
        aa_counts = Counter(seq[pos] for seq in sequences)
        # Get max count
        max_count = max(aa_counts.values())
        # Get all AAs with max count (handle ties)
        top_aas = [aa for aa, count in aa_counts.items() if count == max_count]
        # Break ties alphabetically
        consensus_chars.append(sorted(top_aas)[0])

    return "".join(consensus_chars)


def get_reference_sequence(
    group_df: pd.DataFrame,
    reference_col: str,
    sequence_col: str,
    warn_context: Optional[str] = None,
) -> Optional[str]:
    """Extract a group's reference sequence, if its reference rows agree.

    Used by callers that have a true reference (e.g. ProteinGym's
    ``is_reference`` row, or tsuboyama_megascale's several
    biological/technical replicate ``is_reference`` rows per domain) and
    want to use it directly instead of a reconstructed
    :func:`compute_consensus`. Shared by
    :func:`dasm2.analysis.functional_eval.evaluate_grouped_dataset` and the
    ProteinGym family-holdout-eval's consensus-vs-WT Hamming diagnostic so
    the two don't independently duplicate (and risk diverging on) the
    row-extraction and multiplicity-handling logic.

    Reference rows are deduplicated on ``sequence_col`` before checking for
    ambiguity, so a dataset like tsuboyama_megascale -- whose wild-type is
    measured several times as replicate rows, all ``reference_col == True``
    with identical ``sequence_col`` -- resolves to that one sequence rather
    than being flagged ambiguous merely for having more than one reference
    row.

    Args:
        group_df: DataFrame for a single group (e.g. one ProteinGym assay).
        reference_col: Boolean column identifying the reference row(s).
        sequence_col: Column with the amino-acid sequence.
        warn_context: If given, logged (via :func:`logging.warning`) when
            the group's reference rows don't agree on exactly one distinct
            sequence, describing what the caller is about to do about it
            (e.g. "group_id 'X' (falling back to compute_consensus)").
            ``None`` (default) suppresses the warning -- used when the
            caller will raise or handle the anomaly itself.

    Returns:
        The reference rows' shared ``sequence_col`` value, or ``None`` if
        the group has zero reference rows, or its reference rows don't
        agree on exactly one distinct sequence (an ambiguous or missing
        reference is never silently guessed at).
    """
    ref_rows = group_df[group_df[reference_col].fillna(False).astype(bool)]
    unique_seqs = ref_rows[sequence_col].unique()
    if len(unique_seqs) != 1:
        if warn_context is not None:
            logger.warning(
                f"Expected exactly 1 distinct '{sequence_col}' value across "
                f"'{reference_col}' rows for {warn_context}, found "
                f"{len(unique_seqs)} across {len(ref_rows)} rows."
            )
        return None
    return str(unique_seqs[0])
