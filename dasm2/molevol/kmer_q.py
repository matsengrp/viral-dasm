"""Context-dependent Q-matrix construction from a global k-mer lookup table.

QDASM Option 1a (#589) wants each row of ``Q_ℓ`` to carry the mutation rate and
CSP appropriate to *its own* starting codon, not the parent's. This module does
that by evaluating a neutral model's k-mer table at every hypothetical
intermediate codon, for whichever ``k`` the neutral model's receptive field
needs:

- ``k = 13`` — Thrifty, whose 3-mer embeddings plus ``Conv1d(kernel_size=11)``
  give it a 13-nt receptive field. Table is ``[4^13, 5]`` (~1.34 GB).
- ``k = 5`` — S5F, a pure five-mer lookup. Table is ``[4^5, 5]`` (20 KB), and
  the result is *exact*, not an approximation: S5F's rate/CSP at a site is a
  function of the centered five-mer and nothing else.

Everything except the table lookup is shared between the two, so the k-specific
part is confined to :func:`intermediate_kmer_features` and the rest of the
pipeline (edge assembly, diagonal, boundary splice) is written once and is
rank-agnostic — the same code serves ``[L, ...]`` and ``[B, L, ...]`` inputs.

Index decomposition
-------------------
The per-codon nt window is 15 long with the parent codon at positions 6, 7, 8
(see :func:`dasm2.sequence.conversion.build_codon_nt_windows`). Writing
``h = (k - 1) // 2``, the k-mer centered on codon position ``p`` is
``window[6 + p - h : 6 + p - h + k]``, and the codon's nt ``j`` lands at k-mer
position ``j - p + h``. Both stay in range exactly when ``h >= 2``, i.e.
``k >= 5``; below that the codon would not fit inside its own k-mer.

Because the k-mer index is a weighted sum of independent positions, it splits
additively into a *flank* part (parent window only) and a *center* part
(intermediate codon and codon-position only). Adding them with broadcasting
avoids ever materializing the ``[..., L, 61, 3, k]`` k-mer tensor.
"""

from functools import lru_cache
from typing import Optional

import torch
from torch import Tensor

from dasm2.codon_table import (
    NUM_SENSE_CODONS,
    SENSE_CODON_2DIFF_I,
    SENSE_CODON_2DIFF_J,
    SENSE_CODON_2DIFF_POS,
    SENSE_CODON_2DIFF_TARGET_NTS,
    SENSE_CODON_3DIFF_I,
    SENSE_CODON_3DIFF_J,
    SENSE_CODON_3DIFF_POS,
    SENSE_CODON_3DIFF_TARGET_NTS,
    SENSE_CODON_DIFF_I,
    SENSE_CODON_DIFF_J,
    SENSE_CODON_DIFF_POS,
    SENSE_CODON_DIFF_TARGET_NT,
    SENSE_CODON_NT_INDICES,
    SENSE_CODON_TO_AA,
)
from dasm2.molevol._parent_row_q import aa_selection_ratio
from dasm2.neutral.codon_boundary import BOUNDARY_FEATURE_WIDTH

# Width of the per-codon nt window produced by ``build_codon_nt_windows``, and
# the window position of the parent codon's first nt within it.
CODON_NT_WINDOW_LEN = 15
_PARENT_CODON_OFFSET = 6
# Sentinel written by ``build_codon_nt_windows`` for N / separator / padding.
_N_INDEX = 4


@lru_cache(maxsize=None)
def _kmer_constants(
    kmer_len: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Return ``(powers, center_powers, center_part, window_start)`` for a k-mer.

    Cached per ``(kmer_len, device)``: these are compile-time constants of the
    genetic code, and the Q build runs once per training step, so recomputing
    them would show up in the step time.

    Args:
        kmer_len: Odd k >= 5. See the module docstring for why 5 is the floor.
        device: Device to build the constant tensors on.

    Returns:
        powers: ``[k]`` int64 — ``powers[m] = 4 ** (k - 1 - m)``.
        center_powers: ``[3, 3]`` int64 — ``center_powers[p, j]`` is the place
            value of codon nt ``j`` in the k-mer centered on codon position
            ``p``, namely ``4 ** (h - j + p)``.
        center_part: ``[61, 3]`` int64 — ``center_part[c, p]`` is the total
            contribution of intermediate sense codon ``c`` to that k-mer index.
        window_start: Window position of the k-mer for codon position 0, i.e.
            ``6 - h``. Codon position ``p`` starts at ``window_start + p``.
    """
    if kmer_len % 2 == 0 or kmer_len < 5:
        raise ValueError(
            f"kmer_len must be odd and >= 5 (so the codon fits inside its own "
            f"k-mer); got {kmer_len}"
        )
    half = (kmer_len - 1) // 2
    max_kmer_len = 2 * _PARENT_CODON_OFFSET + 1
    if kmer_len > max_kmer_len:
        raise ValueError(
            f"kmer_len={kmer_len} exceeds the largest centered receptive "
            f"field ({max_kmer_len}) that fits around all three nucleotides "
            f"of the parent codon in a {CODON_NT_WINDOW_LEN}-nt window."
        )
    # Place value of codon nt j in the k-mer centered on codon position p:
    # the nt sits at k-mer position (j - p + half), whose weight is
    # 4 ** (kmer_len - 1 - (j - p + half)) = 4 ** (half - j + p).
    j_idx = torch.arange(3, device=device).view(1, 3)
    p_idx = torch.arange(3, device=device).view(3, 1)
    center_powers = torch.pow(4, half - j_idx + p_idx)  # [3, 3]
    center_part = (
        SENSE_CODON_NT_INDICES.to(device).unsqueeze(1) * center_powers.unsqueeze(0)
    ).sum(-1)  # [61, 3]
    powers = torch.pow(4, torch.arange(kmer_len - 1, -1, -1, device=device))
    return powers, center_powers, center_part, _PARENT_CODON_OFFSET - half


def kmer_boundary_mask(codon_parent_nt_window: Tensor, kmer_len: int) -> Tensor:
    """Flag codon sites whose k-mer lookups would read an N / padding base.

    Args:
        codon_parent_nt_window: ``[..., L, 15]`` long, nt indices 0..3 or 4 for
            N / separator / out-of-range.
        kmer_len: Odd k >= 5.

    Returns:
        ``[..., L]`` bool, True where the k-mer footprint contains a 4.
    """
    half = (kmer_len - 1) // 2
    lo = _PARENT_CODON_OFFSET - half
    hi = _PARENT_CODON_OFFSET + 3 + half
    footprint = codon_parent_nt_window[..., lo:hi]
    return (footprint >= _N_INDEX).any(dim=-1)


def narrow_boundary_mask_to_kmer(
    codon_parent_nt_window: Tensor, boundary_mask: Tensor, kmer_len: int
) -> Tensor:
    """Drop boundary flags that only padding *outside* the k-mer footprint caused.

    ``build_codon_nt_windows`` flags a site when any of the 15 window positions
    is non-ACGT. That is exact for ``k = 13``, whose lookups span the whole
    window, but over-conservative below it: a five-mer lookup reads only window
    positions 4..10. The second and second-last codons therefore have complete
    five-mer footprints even though padding elsewhere in their 15-nt windows
    sets the broad boundary flag. Narrowing recovers those two sites for the
    real lookup; the terminal codons still splice because their five-mers
    genuinely contain pad-N bases.

    A flag is dropped only when the window itself explains it *and* the k-mer
    footprint is clean. Callers that flag a site for some other reason — tests
    forcing the splice, or a future caller masking sites deliberately — are
    honored, because their flags are not attributable to window padding.

    For ``k = 13`` the footprint is the whole window, so this is the identity.

    Args:
        codon_parent_nt_window: ``[..., L, 15]`` long.
        boundary_mask: ``[..., L]`` bool, as supplied by the caller.
        kmer_len: Odd k >= 5.

    Returns:
        ``[..., L]`` bool, a subset of ``boundary_mask``.
    """
    window_has_n = (codon_parent_nt_window >= _N_INDEX).any(dim=-1)
    footprint_has_n = kmer_boundary_mask(codon_parent_nt_window, kmer_len)
    return boundary_mask & (footprint_has_n | ~window_has_n)


def intermediate_kmer_features(
    codon_parent_nt_window: Tensor,
    kmer_table: Tensor,
    boundary_mask: Tensor,
    kmer_len: int,
) -> tuple[Tensor, Tensor]:
    """Look up (rate, CSP) for every (site, intermediate codon, codon position).

    For each codon site ``ℓ`` and each of the 61 intermediate sense codons
    ``c'``, forms the k-mer centered on each of the codon's three nt positions —
    taking flanks from the parent window and the center codon from ``c'`` — and
    reads the neutral model's table at that k-mer.

    Args:
        codon_parent_nt_window: ``[..., L, 15]`` long, nt indices 0..3 (A/C/G/T)
            or 4 for N / out-of-range. Positions 6, 7, 8 hold the parent codon.
        kmer_table: ``[4 ** kmer_len, 5]`` float — (rate, csp_A, csp_C, csp_G,
            csp_T), with the wild-type base already zeroed at the k-mer's center
            nt.
        boundary_mask: ``[..., L]`` bool — True at sites where the lookup would
            be invalid. Rates there are returned as 0 and CSPs as 0.25; the
            caller is expected to overwrite them via the boundary splice.
        kmer_len: Odd k >= 5.

    Returns:
        ``(rates_ctx, csps_ctx)`` of shapes ``[..., L, 61, 3]`` and
        ``[..., L, 61, 3, 4]``.
    """
    prefix_shape = codon_parent_nt_window.shape[:-1]
    if codon_parent_nt_window.shape[-1] != CODON_NT_WINDOW_LEN:
        raise ValueError(
            f"codon_parent_nt_window must have last dim = {CODON_NT_WINDOW_LEN}, "
            f"got {tuple(codon_parent_nt_window.shape)}"
        )
    device = codon_parent_nt_window.device
    # Validate k before computing 4**k or accepting a table shape. In
    # particular, k=15 fits the stored window but cannot be centered on all
    # three nucleotides of its parent codon.
    powers, center_powers, center_part, window_start = _kmer_constants(kmer_len, device)
    expected_rows = 4**kmer_len
    if kmer_table.shape != (expected_rows, 5):
        raise ValueError(
            f"kmer_table for kmer_len={kmer_len} must have shape "
            f"({expected_rows}, 5), got {tuple(kmer_table.shape)}"
        )
    if boundary_mask.device != device or kmer_table.device != device:
        raise ValueError(
            f"intermediate_kmer_features inputs must agree on device; got "
            f"codon_parent_nt_window={device}, "
            f"boundary_mask={boundary_mask.device}, "
            f"kmer_table={kmer_table.device}"
        )
    if boundary_mask.shape != prefix_shape:
        raise ValueError(
            f"boundary_mask shape {tuple(boundary_mask.shape)} must match "
            f"codon_parent_nt_window prefix {tuple(prefix_shape)}"
        )

    # Clamp N / out-of-range (4) into range so the gather stays legal; those
    # sites are flagged in boundary_mask and overwritten below.
    wc = codon_parent_nt_window.clamp_max(_N_INDEX - 1)  # [..., L, 15]
    # unfold gives a view whose element t is window[t : t + k]; codon position p
    # needs t = window_start + p.
    windows = wc.unfold(-1, kmer_len, 1)[..., window_start : window_start + 3, :]
    base_index = (windows * powers).sum(dim=-1)  # [..., L, 3]
    # Strip the parent codon's center contribution, leaving flanks only, then
    # broadcast the intermediate codon's contribution back across the 61 dim.
    parent_center = (
        wc[..., _PARENT_CODON_OFFSET : _PARENT_CODON_OFFSET + 3].unsqueeze(-2)
        * center_powers
    ).sum(dim=-1)  # [..., L, 3]
    flank_part = base_index - parent_center  # [..., L, 3]
    kmer_indices = flank_part.unsqueeze(-2) + center_part  # [..., L, 61, 3]

    flat_entries = kmer_table[kmer_indices]  # [..., L, 61, 3, 5]
    rates_ctx = flat_entries[..., 0]
    csps_ctx = flat_entries[..., 1:]

    rate_mask = boundary_mask.unsqueeze(-1).unsqueeze(-1)  # [..., L, 1, 1]
    rates_ctx = rates_ctx.masked_fill(rate_mask, 0.0)
    csps_ctx = csps_ctx.masked_fill(rate_mask.unsqueeze(-1), 0.25)

    return rates_ctx, csps_ctx


def assemble_q_from_context_features(
    rates_ctx: Tensor,
    csps_ctx: Tensor,
    aa_selection_factors: Tensor,
    single_hit_scale: float,
    double_hit_scale: float,
    triple_hit_scale: float,
) -> Tensor:
    """Assemble Q from per-(site, source codon, position) rates and CSPs.

    Rank-agnostic: a ``[L, ...]`` prefix yields ``[L, 61, 61]`` and a
    ``[B, L, ...]`` prefix yields ``[B, L, 61, 61]``. Off-diagonal entries are
    the product of the per-nt rates and CSPs along the substitution path, the
    hit-count attenuator, and the amino-acid selection ratio
    ``f[AA(target)] / f[AA(source)]``. The diagonal is set so rows sum to 0.

    Args:
        rates_ctx: ``[..., L, 61, 3]`` context-dependent rates.
        csps_ctx: ``[..., L, 61, 3, 4]`` context-dependent CSPs.
        aa_selection_factors: ``[..., L, 20]`` float (NOT log).
        single_hit_scale: 1-nt attenuator (γ).
        double_hit_scale: 2-nt attenuator (δ).
        triple_hit_scale: 3-nt attenuator (ψ).

    Returns:
        ``Q [..., L, 61, 61]`` with rows summing to 0.
    """
    device = aa_selection_factors.device
    prefix_shape = rates_ctx.shape[:-2]  # (..., L)
    Q = torch.zeros((*prefix_shape, NUM_SENSE_CODONS, NUM_SENSE_CODONS), device=device)

    sense_to_aa = SENSE_CODON_TO_AA.to(device)

    # Shape legend: N/M/K = number of 1-, 2-, 3-nt-difference codon pairs.
    # Indexing with a leading ellipsis keeps this identical for both ranks.
    diff_i = SENSE_CODON_DIFF_I.to(device)
    diff_j = SENSE_CODON_DIFF_J.to(device)
    diff_pos = SENSE_CODON_DIFF_POS.to(device)
    diff_target_nt = SENSE_CODON_DIFF_TARGET_NT.to(device)

    rates_1 = rates_ctx[..., diff_i, diff_pos]  # [..., L, N]
    csps_1 = csps_ctx[..., diff_i, diff_pos, diff_target_nt]  # [..., L, N]
    sel_1 = aa_selection_ratio(aa_selection_factors, diff_i, diff_j, sense_to_aa)
    Q[..., diff_i, diff_j] = single_hit_scale * rates_1 * csps_1 * sel_1

    if len(SENSE_CODON_2DIFF_I) > 0:
        i2 = SENSE_CODON_2DIFF_I.to(device)
        j2 = SENSE_CODON_2DIFF_J.to(device)
        pos2 = SENSE_CODON_2DIFF_POS.to(device)  # [M, 2]
        nts2 = SENSE_CODON_2DIFF_TARGET_NTS.to(device)  # [M, 2]

        rates_2 = rates_ctx[..., i2, pos2[:, 0]] * rates_ctx[..., i2, pos2[:, 1]]
        csps_2 = (
            csps_ctx[..., i2, pos2[:, 0], nts2[:, 0]]
            * csps_ctx[..., i2, pos2[:, 1], nts2[:, 1]]
        )
        sel_2 = aa_selection_ratio(aa_selection_factors, i2, j2, sense_to_aa)
        Q[..., i2, j2] = double_hit_scale * rates_2 * csps_2 * sel_2

    if len(SENSE_CODON_3DIFF_I) > 0:
        i3 = SENSE_CODON_3DIFF_I.to(device)
        j3 = SENSE_CODON_3DIFF_J.to(device)
        pos3 = SENSE_CODON_3DIFF_POS.to(device)  # [K, 3]
        nts3 = SENSE_CODON_3DIFF_TARGET_NTS.to(device)  # [K, 3]

        rates_3 = (
            rates_ctx[..., i3, pos3[:, 0]]
            * rates_ctx[..., i3, pos3[:, 1]]
            * rates_ctx[..., i3, pos3[:, 2]]
        )
        csps_3 = (
            csps_ctx[..., i3, pos3[:, 0], nts3[:, 0]]
            * csps_ctx[..., i3, pos3[:, 1], nts3[:, 1]]
            * csps_ctx[..., i3, pos3[:, 2], nts3[:, 2]]
        )
        sel_3 = aa_selection_ratio(aa_selection_factors, i3, j3, sense_to_aa)
        Q[..., i3, j3] = triple_hit_scale * rates_3 * csps_3 * sel_3

    Q.diagonal(dim1=-2, dim2=-1).copy_(-Q.sum(dim=-1))
    return Q


def splice_boundary_sites(
    Q: Tensor,
    window: Tensor,
    boundary_mask: Tensor,
    codon_boundary_features: Optional[Tensor],
    aa_selection_factors: Tensor,
    single_hit_scale: float,
    double_hit_scale: float,
    triple_hit_scale: float,
) -> Tensor:
    """Assemble boundary Qs from frozen source-codon features or an explicit oracle.

    Sparse records have shape ``[..., N, BOUNDARY_FEATURE_WIDTH]``: site index followed by
    ``[61, 3, 5]`` rate/CSP features. Records padded with index -1, or beyond
    a caller's truncated Q length, are ignored. Missing features for a valid
    boundary codon are an error: parent-only CSPs forbid returns to the parent.
    """
    device = Q.device
    length = Q.shape[-3]
    flat_boundary = boundary_mask.reshape(-1)
    # Masked ambiguous/stop codons and padding have no sense parent to score.
    parent = window[..., 6:9].reshape(-1, 3)
    valid_parent = (
        (parent.unsqueeze(-2) == SENSE_CODON_NT_INDICES.to(device)).all(-1).any(-1)
    )
    required = flat_boundary & valid_parent
    covered = torch.zeros_like(required)
    if codon_boundary_features is not None:
        expected_prefix = Q.shape[:-3]
        if (
            codon_boundary_features.ndim != len(expected_prefix) + 2
            or codon_boundary_features.shape[:-2] != expected_prefix
            or codon_boundary_features.shape[-1] != BOUNDARY_FEATURE_WIDTH
        ):
            raise ValueError(
                "codon_boundary_features must have shape "
                f"{tuple(expected_prefix)} + (N, {BOUNDARY_FEATURE_WIDTH})"
            )
        records = (
            codon_boundary_features.to(device).reshape(
                -1, codon_boundary_features.shape[-2], BOUNDARY_FEATURE_WIDTH
            )
            if codon_boundary_features.numel()
            else None
        )
        if records is not None:
            sites = records[..., 0].long()
            keep = (sites >= 0) & (sites < length)
            batch, record = keep.nonzero(as_tuple=True)
            indices = batch * length + sites[batch, record]
            use = required[indices]
            indices = indices[use]
            values = records[batch[use], record[use], 1:].reshape(-1, 61, 3, 5)
            covered[indices] = True
            boundary_q = assemble_q_from_context_features(
                values[..., 0],
                values[..., 1:],
                aa_selection_factors.reshape(-1, 20)[indices],
                single_hit_scale,
                double_hit_scale,
                triple_hit_scale,
            )
            Q = (
                Q.reshape(-1, 61, 61)
                .index_copy(0, indices, boundary_q)
                .reshape(Q.shape)
            )
    if (required & ~covered).any():
        raise ValueError(
            "Global k-mer Q requires source-specific codon_boundary_features "
            "at every valid boundary/ambiguous-window site. "
            "Prepare them with ComputeParentNtWindows(neutral_model, segment_handler)."
        )
    return Q


def build_mutsel_q_matrices_from_kmer_lookup(
    codon_parent_nt_window: Tensor,
    codon_parent_boundary_mask: Tensor,
    aa_selection_factors: Tensor,
    kmer_table: Tensor,
    kmer_len: int,
    single_hit_scale: float = 1.0,
    double_hit_scale: float = 0.02,
    triple_hit_scale: float = 0.001,
    codon_rates: Optional[Tensor] = None,
    codon_csps: Optional[Tensor] = None,
    codon_boundary_features: Optional[Tensor] = None,
    codon_rate_adjustments: Optional[Tensor] = None,
) -> Tensor:
    """Build context-dependent Q matrices from a global k-mer table.

    Unlike ``build_mutsel_q_matrices``, whose rates and CSPs depend on
    ``(site, diff_pos, diff_target_nt)`` but *not* on the starting codon (every
    row shares the parent's scalars), this evaluates the neutral model at each
    hypothetical intermediate codon, so each row of ``Q_ℓ`` carries
    context-appropriate rate and CSP information. The amino-acid selection ratio
    is applied identically in both variants.

    Rank-agnostic: pass ``[L, ...]`` or ``[B, L, ...]``.

    Args:
        codon_parent_nt_window: ``[..., L, 15]`` long.
        codon_parent_boundary_mask: ``[..., L]`` bool, as produced by
            ``build_codon_nt_windows`` over the full 15-nt window. It is
            narrowed here to the k-mer's actual footprint (see
            :func:`kmer_boundary_mask`), so smaller k diverts fewer sites to the
            splice. For ``k = 13`` the two coincide.
        aa_selection_factors: ``[..., L, 20]`` float (NOT log).
        kmer_table: ``[4 ** kmer_len, 5]`` float.
        kmer_len: Odd k >= 5.
        single_hit_scale: 1-nt attenuator (γ).
        double_hit_scale: 2-nt attenuator (δ).
        triple_hit_scale: 3-nt attenuator (ψ).
        codon_rates, codon_csps: Legacy parent features, unused by this builder.
            Parent-only CSPs cannot define non-parent Q rows.
        codon_boundary_features: Sparse ``[..., N, 916]`` cached full-segment
            source features. Required at valid boundary sites; see
            :func:`splice_boundary_sites`.
        codon_rate_adjustments: Required ``[..., L]`` segment multipliers
            applied to every source codon's three lookup rates. Pass explicit
            ones for intentionally unadjusted calls. A k-hit edge is
            multiplied by adjustment**k, preserving the per-nucleotide clock
            convention. ``codon_boundary_features`` must already be adjusted;
            the boundary splice does not apply this multiplier again.

    Returns:
        ``Q [..., L, 61, 61]`` with rows summing to 0.
    """
    if codon_rate_adjustments is None:
        raise ValueError(
            "Global k-mer Q requires explicit codon_rate_adjustments; pass the "
            "per-codon segment multipliers from the dataset, or an all-ones "
            "tensor only for intentionally unadjusted rates."
        )
    if codon_rate_adjustments.shape != codon_parent_nt_window.shape[:-1]:
        raise ValueError(
            "codon_rate_adjustments must match the codon-site shape "
            f"{tuple(codon_parent_nt_window.shape[:-1])}; "
            f"got {tuple(codon_rate_adjustments.shape)}"
        )
    device = aa_selection_factors.device
    window = codon_parent_nt_window.to(device)
    boundary = narrow_boundary_mask_to_kmer(
        window, codon_parent_boundary_mask.to(device), kmer_len
    )

    rates_ctx, csps_ctx = intermediate_kmer_features(
        window, kmer_table.to(device), boundary, kmer_len
    )
    rates_ctx = rates_ctx * codon_rate_adjustments.to(device)[..., None, None]
    # The table already has WT-base zeroing applied at the k-mer's center nt.
    # For intermediate codon c' at codon position p the center nt is c'[p], so
    # csps_ctx[..., c', p, c'[p]] is already 0 — no extra zeroing needed.

    Q = assemble_q_from_context_features(
        rates_ctx,
        csps_ctx,
        aa_selection_factors,
        single_hit_scale,
        double_hit_scale,
        triple_hit_scale,
    )

    Q = splice_boundary_sites(
        Q,
        window,
        boundary,
        codon_boundary_features,
        aa_selection_factors,
        single_hit_scale,
        double_hit_scale,
        triple_hit_scale,
    )

    return Q


__all__ = [
    "CODON_NT_WINDOW_LEN",
    "assemble_q_from_context_features",
    "build_mutsel_q_matrices_from_kmer_lookup",
    "intermediate_kmer_features",
    "kmer_boundary_mask",
    "narrow_boundary_mask_to_kmer",
    "splice_boundary_sites",
]
