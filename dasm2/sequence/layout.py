"""Layout tensor construction for segment identification."""

import torch

from dasm2.sequence.constants import AA_AMBIG_IDX, AA_SEPARATOR_IDX


def padding_sentinel_for(num_segments: int) -> int:
    """Return the padding sentinel value for a given number of segments.

    The padding sentinel is ``num_segments + 1``, ensuring it does not
    conflict with any segment index (1..num_segments) or separator
    index (0).

    Args:
        num_segments: Number of protein segments.

    Returns:
        Integer padding sentinel value.
    """
    return num_segments + 1


def layout_of_single_aa_idxs(
    aa_idxs: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Build layout tensor for a single sequence (no batch dimension).

    Single-sequence version of layout_of_aa_idxs. Takes a 1D tensor and
    returns a 1D tensor. Used by per-sample transforms in TransformDataset.

    Args:
        aa_idxs: Token index tensor of shape [seq_len].
        num_segments: Number of expected segments.

    Returns:
        Integer tensor of shape [seq_len] with segment labels.
        See layout_of_aa_idxs for value semantics.
    """
    if aa_idxs.dim() != 1:
        raise ValueError(f"aa_idxs must be 1D [seq_len], got shape {aa_idxs.shape}")
    # Reuse batch version: unsqueeze to [1, seq_len], compute, squeeze back
    return layout_of_aa_idxs(aa_idxs.unsqueeze(0), num_segments).squeeze(0)


def layout_of_aa_idxs(
    aa_idxs: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Build layout tensor from amino acid index tensor.

    Assigns each position a segment label based on separator tokens.
    This is the standalone equivalent of SequenceDataset._create_layout()
    for use outside the dataset pipeline (e.g., during inference).

    Args:
        aa_idxs: Token index tensor of shape [batch_size, seq_len].
        num_segments: Number of expected segments (e.g., 1 for single-segment,
            2 for antibody heavy+light).

    Returns:
        Integer tensor of same shape as aa_idxs, with values:
            0: separator positions
            1..N: segment indices (1-based)
            N+1: padding positions (trailing AA_AMBIG_IDX positions)

    Example:
        >>> # Antibody: "EVQLV^DIQMT" tokenized + 2 padding positions
        >>> # AA_SEPARATOR_IDX = 21 (^), AA_AMBIG_IDX = 20 (padding)
        >>> aa_idxs = torch.tensor([[4, 18, 13, 8, 18, 21, 2, 7, 13, 9, 17, 20, 20]])
        >>> layout_of_aa_idxs(aa_idxs, num_segments=2)
        tensor([[1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 3, 3]])
    """
    if aa_idxs.dim() != 2:
        raise ValueError(
            f"aa_idxs must be 2D [batch_size, seq_len], got shape {aa_idxs.shape}"
        )

    padding_sentinel = padding_sentinel_for(num_segments)

    # Segment index = cumulative separator count + 1
    layout = torch.cumsum(aa_idxs == AA_SEPARATOR_IDX, dim=-1).to(torch.int) + 1

    # Separator positions get value 0
    layout[aa_idxs == AA_SEPARATOR_IDX] = 0

    # Mark trailing padding positions (contiguous AA_AMBIG_IDX at end of each row).
    # Vectorized: a position is trailing padding if it and ALL positions after it
    # are AA_AMBIG_IDX. Computed via reverse cumsum of non-ambiguous positions.
    is_not_ambig = aa_idxs != AA_AMBIG_IDX
    # Reverse cumsum: count non-ambiguous positions from the right.
    # If this count is 0 at position i, then positions i..L-1 are all ambiguous.
    reverse_nonambig_count = torch.flip(
        torch.cumsum(torch.flip(is_not_ambig, dims=[-1]).int(), dim=-1),
        dims=[-1],
    )
    is_trailing_padding = reverse_nonambig_count == 0
    layout[is_trailing_padding] = padding_sentinel

    return layout


def per_segment_positions_of_layout(layout: torch.Tensor) -> torch.Tensor:
    """Compute per-segment positions that reset at each segment boundary.

    Instead of absolute positions that accumulate across the whole
    concatenated sequence, each segment (chain) restarts its position count at
    0 right after the preceding separator. This makes a segment's positional
    encoding independent of the lengths of the segments that precede it — the
    key property motivating per-segment positional encoding (see #334).

    The reset point for a position is the index immediately following the most
    recent separator (``layout == 0``) strictly before it, or 0 when there is
    no preceding separator. A separator token itself gets the position one
    past the last residue of the segment it terminates (e.g. a 5-residue
    segment's separator gets position 5, not 4); padding positions continue
    counting from the last segment's reset point. Both separators and padding
    are masked downstream, so only their being valid, deterministic indices
    matters.

    For a single-segment sequence (no separators) this reduces exactly to
    absolute positions ``0, 1, ..., L-1``.

    Args:
        layout: Integer layout tensor from :func:`layout_of_aa_idxs`, shape
            ``[..., seq_len]``. Values: 0 = separator, 1..N = segment indices,
            N+1 = padding. Positions are computed along the last dimension.

    Returns:
        Long tensor of the same shape as ``layout`` with per-segment positions.

    Example:
        >>> # "EVQLV^DIQMT" + 2 padding, layout = [1,1,1,1,1,0,2,2,2,2,2,3,3]
        >>> layout = torch.tensor([[1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 3, 3]])
        >>> per_segment_positions_of_layout(layout)
        tensor([[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6]])
    """
    seq_len = layout.shape[-1]
    ar = torch.arange(seq_len, device=layout.device)

    # At each separator index j, record the reset point j+1 (position of the
    # first token of the next segment); elsewhere record 0.
    is_sep = layout == 0
    sep_reset = torch.where(is_sep, ar + 1, torch.zeros_like(ar))

    # Reset point for position i = largest recorded reset among separators
    # strictly before i. sep_reset is monotonic in the separator index, so a
    # running max gives the most recent one; shift right by one to make it
    # exclusive of position i (so a separator does not reset itself).
    cummax_inclusive = torch.cummax(sep_reset, dim=-1).values
    reset_point = torch.zeros_like(cummax_inclusive)
    reset_point[..., 1:] = cummax_inclusive[..., :-1]

    return (ar - reset_point).long()
