"""Hybrid model: exact neutral CTMC + post-hoc selection.

Like QDASM, except the matrix exponential runs on a *neutral* Q (unit
selection) and the learned selection factors are applied post-hoc by
``compute_log_adjusted_codon_probs``. Combines QDASM's exact-CTMC neutral
model with DASM's post-hoc selection step.

Extracted from ``mutsel.py``.
"""

from typing import Callable, Optional

import torch
from torch import Tensor

from dasm2.molevol.matrix_exp import _batched_matrix_exp_to_log_probs
from dasm2.molevol.mutsel import compute_log_adjusted_codon_probs
from dasm2.molevol.q_matrix import (
    QVariant,
    _build_q_matrix,
    _build_q_matrix_batched,
    mutsel_codon_transition_probs,
    shared_codon_length,
)


def build_hybrid_neutral_q_all(
    codon_rates: Tensor,
    codon_csps: Tensor,
    length: int,
    *,
    single_hit_scale: float = 1.0,
    double_hit_scale: float = 0.02,
    triple_hit_scale: float = 0.001,
    q_variant: QVariant = "parent_row_degenerate",
    codon_parent_nt_window: Optional[Tensor] = None,
    codon_parent_boundary_mask: Optional[Tensor] = None,
    kmer_table: Optional[Tensor] = None,
    codon_boundary_features: Optional[Tensor] = None,
    codon_rate_adjustments: Optional[Tensor] = None,
) -> Tensor:
    """Build the batched Hybrid *neutral* Q ``[B, length, 61, 61]``.

    Unit selection factors, so this Q depends only on rates/CSPs/hit-scales —
    not on branch length *or* the learned selection. Factored out of
    ``compute_hybrid_log_codon_probs`` so the BL-opt golden-section loop can
    build it once and pass it back via ``precomputed_neutral_q`` instead of
    rebuilding the identical neutral Q for every trial branch length.

    ``codon_rate_adjustments [B, L]`` is required for global k-mer variants.
    See :func:`~dasm2.molevol.kmer_q.build_mutsel_q_matrices_from_kmer_lookup`
    for its scaling contract. It is sliced to ``length`` with the other
    per-site inputs.
    """
    b = codon_rates.shape[0]
    device = codon_rates.device
    unit_selection = torch.ones(b, length, 20, device=device)
    return _build_q_matrix_batched(
        codon_rates[:, :length],
        codon_csps[:, :length],
        unit_selection,
        single_hit_scale=single_hit_scale,
        double_hit_scale=double_hit_scale,
        triple_hit_scale=triple_hit_scale,
        q_variant=q_variant,
        codon_parent_nt_window=(
            codon_parent_nt_window[:, :length]
            if codon_parent_nt_window is not None
            else None
        ),
        codon_parent_boundary_mask=(
            codon_parent_boundary_mask[:, :length]
            if codon_parent_boundary_mask is not None
            else None
        ),
        codon_boundary_features=codon_boundary_features,
        codon_rate_adjustments=(
            codon_rate_adjustments[:, :length]
            if codon_rate_adjustments is not None
            else None
        ),
        kmer_table=kmer_table,
    )


def compute_hybrid_log_codon_probs(
    parent_codon_idxs: Tensor,
    codon_rates: Tensor,
    codon_csps: Tensor,
    log_aa_selection_factors: Tensor,
    branch_lengths: Tensor,
    use_cubic: bool = False,
    single_hit_scale: float = 1.0,
    double_hit_scale: float = 0.02,
    triple_hit_scale: float = 0.001,
    q_variant: QVariant = "parent_row_degenerate",
    codon_parent_nt_window: Optional[Tensor] = None,
    codon_parent_boundary_mask: Optional[Tensor] = None,
    kmer_table: Optional[Tensor] = None,
    precomputed_neutral_q: Optional[Tensor] = None,
    codon_boundary_features: Optional[Tensor] = None,
    codon_rate_adjustments: Optional[Tensor] = None,
) -> Tensor:
    """Compute log codon transition probabilities using Hybrid model.

    The Hybrid model computes neutral transition probabilities via the exact
    CTMC (matrix exponential with unit selection factors), then applies learned
    selection factors post-hoc — combining QDASM's exact neutral model with
    DASM's post-hoc selection application.

    The post-hoc step runs ``compute_log_adjusted_codon_probs`` at its
    ``parent_absorbing`` default, so Hybrid is deliberately untouched by
    issue #1238. Conditioning is equally well-defined on a ``P = exp(tQ)``
    neutral row, but extending it there is a separate decision with its own
    baseline to preserve, not a gap in that issue.

    Vectorized over the batch dimension: builds neutral Q-matrices per sequence
    (fast), then computes all B×L matrix exponentials in a single GPU call.

    Args:
        parent_codon_idxs: Parent codon indices in 64-space [B, L_aa]
        codon_rates: Per-position mutation rates [B, L_codon, 3]
        codon_csps: Per-position CSPs [B, L_codon, 3, 4]
        log_aa_selection_factors: Log AA selection factors [B, L_aa, 20]
        branch_lengths: Branch lengths [B]
        use_cubic: Use cubic Taylor approximation for matrix exponential.
        single_hit_scale: Scale factor (γ) for 1-nucleotide-difference Q
            entries. Attenuates instantaneous single-hit rates.
        double_hit_scale: Scale factor for 2-nucleotide-difference Q entries.
        triple_hit_scale: Scale factor for 3-nucleotide-difference Q entries.
        q_variant: Q-matrix construction variant — one of
            ``"parent_row_degenerate"`` (parent's per-position Thrifty rates,
            broadcast across all 61 rows) or ``"global_13mer"`` (per-row
            context-dependent rates from the 4^13 Thrifty lookup with
            boundary rescans; requires window, mask, cached features, and table).
        codon_parent_nt_window: Required when ``q_variant="global_13mer"``;
            ``[B, L_codon, 15]`` long tensor (positions 6, 7, 8 are the
            parent codon; 6 nt flank on each side). Must agree on device
            with the loss selection factors.
        codon_parent_boundary_mask: Required when ``q_variant="global_13mer"``;
            ``[B, L_codon]`` bool tensor — True at sites where the 15-nt
            window contains any N/out-of-range base; those rows splice
            to cached full-segment source-codon features.
        codon_boundary_features: Sparse cached full-segment source features
            ``[..., N, 916]``; required at valid boundary/ambiguous-window sites.
        codon_rate_adjustments: Required per-site multipliers for global k-mer
            variants; rejected for other variants. See
            :func:`~dasm2.molevol.kmer_q.build_mutsel_q_matrices_from_kmer_lookup`
            and ``docs/concepts/neutral-models.md#heavylight-rate-adjustment``.
        kmer_table: Required when ``q_variant="global_13mer"``;
            ``[4^13, 5]`` float32 lookup tensor (rate, csp_A, csp_C, csp_G,
            csp_T per 13-mer). Lives on the head as a non-persistent buffer.
        precomputed_neutral_q: Optional prebuilt *neutral* ``Q [B, L, 61, 61]``
            (from ``build_hybrid_neutral_q_all``). When given, the neutral
            Q-build is skipped — the BL-opt golden-section loop passes it to
            avoid rebuilding the branch-length-independent neutral Q every
            trial. The Q-construction kwargs are then unused. Named apart from
            QDASM's ``precomputed_q`` because that one is already
            selection-weighted; this one must *not* be — Hybrid applies
            selection after the matrix exponential.

    Returns:
        Log codon transition probabilities [B, L_aa, 64]

    Notes:
        Stop-codon parents (``CODON64_TO_SENSE61[idx] == -1``) must be filtered
        upstream by ``ComputeMasks``. The molevol layer asserts this contract
        at dispatch.
    """
    B, L_aa = parent_codon_idxs.shape
    L = shared_codon_length(L_aa, codon_rates)

    # Build neutral Q matrices (unit selection) for all sequences at once,
    # unless the caller hoisted the build out of a BL search and handed us the
    # branch-length-independent neutral Q via ``precomputed_neutral_q``. Hybrid
    # applies the learned selection post-hoc via compute_log_adjusted_codon_probs
    # after the matrix exponential.
    if precomputed_neutral_q is None:
        Q_all = build_hybrid_neutral_q_all(
            codon_rates,
            codon_csps,
            L,
            single_hit_scale=single_hit_scale,
            double_hit_scale=double_hit_scale,
            triple_hit_scale=triple_hit_scale,
            q_variant=q_variant,
            codon_parent_nt_window=codon_parent_nt_window,
            codon_parent_boundary_mask=codon_parent_boundary_mask,
            codon_boundary_features=codon_boundary_features,
            codon_rate_adjustments=codon_rate_adjustments,
            kmer_table=kmer_table,
        )
    else:
        # Contract: a hoisted neutral Q must already be sliced to the shared
        # codon length ``L`` (see the ``precomputed_neutral_q`` arg). Fail fast
        # here rather than surfacing an opaque shape error deep inside matexp.
        if precomputed_neutral_q.shape[:2] != (B, L):
            raise ValueError(
                "precomputed_neutral_q batch/length "
                f"{tuple(precomputed_neutral_q.shape[:2])} != expected "
                f"(B, L)=({B}, {L}); pass a Q from build_hybrid_neutral_q_all "
                "built with length=shared_codon_length(L_aa, codon_rates)."
            )
        Q_all = precomputed_neutral_q

    # Vectorized matrix exponential → neutral log probs [B, L_aa, 64]
    # No gradients flow through this (unit selection), so cubic just saves compute.
    # apply_stop_zapper=False: compute_log_adjusted_codon_probs applies it.
    neutral_log_probs = _batched_matrix_exp_to_log_probs(
        Q_all,
        parent_codon_idxs[:, :L],
        branch_lengths,
        L_aa,
        use_cubic=use_cubic,
        apply_stop_zapper=False,
    )

    # Apply learned selection post-hoc (already supports batched inputs)
    return compute_log_adjusted_codon_probs(
        parent_codon_idxs,
        neutral_log_probs,
        log_aa_selection_factors,
    )


def hybrid_log_pcp_probability_of(
    sel_matrix: Tensor,
    parent_codon_idxs: Tensor,
    child_codon_idxs: Tensor,
    codon_rates: Tensor,
    codon_csps: Tensor,
    mask: Tensor,
    single_hit_scale: float = 1.0,
    double_hit_scale: float = 0.02,
    triple_hit_scale: float = 0.001,
    use_cubic: bool = False,
    q_variant: QVariant = "parent_row_degenerate",
    codon_parent_nt_window: Optional[Tensor] = None,
    codon_parent_boundary_mask: Optional[Tensor] = None,
    kmer_table: Optional[Tensor] = None,
    codon_boundary_features: Optional[Tensor] = None,
    codon_rate_adjustments: Optional[Tensor] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Construct a closure for Hybrid branch length optimization.

    Builds the neutral Q-matrix once (unit selection factors), then for each
    trial branch length recomputes exp(tQ) and applies selection post-hoc.

    Args:
        sel_matrix: AA selection factors (NOT log) [seq_len, 20]
        parent_codon_idxs: Parent codon indices in 64-space [seq_len]
        child_codon_idxs: Child codon indices in 64-space [seq_len]
        codon_rates: Per-position mutation rates [seq_len, 3]
        codon_csps: Per-position CSPs [seq_len, 3, 4]
        mask: Boolean mask for valid positions [seq_len]
        single_hit_scale: Scale factor (γ) for 1-nucleotide-difference Q
            entries. Attenuates instantaneous single-hit rates.
        double_hit_scale: Scale factor for 2-nucleotide-difference Q entries.
        triple_hit_scale: Scale factor for 3-nucleotide-difference Q entries.
        use_cubic: Use cubic Taylor approximation for matrix exponential.
        q_variant: Q-matrix construction variant; see
            ``compute_hybrid_log_codon_probs`` for the contract on
            ``codon_parent_nt_window``, ``codon_parent_boundary_mask``,
            and ``kmer_table`` when ``q_variant="global_13mer"``.
        codon_parent_nt_window: Required when ``q_variant="global_13mer"``;
            ``[seq_len, 15]`` long tensor.
        codon_parent_boundary_mask: Required when ``q_variant="global_13mer"``;
            ``[seq_len]`` bool tensor.
        codon_boundary_features: Sparse cached full-segment source features
            ``[..., N, 916]``; required at valid boundary/ambiguous-window sites.
        codon_rate_adjustments: Required per-site multipliers for global k-mer
            variants; rejected for other variants. See
            :func:`~dasm2.molevol.kmer_q.build_mutsel_q_matrices_from_kmer_lookup`
            and ``docs/concepts/neutral-models.md#heavylight-rate-adjustment``.
        kmer_table: Required when ``q_variant="global_13mer"``;
            ``[4^13, 5]`` float32 lookup tensor.

    Returns:
        Callable taking log_branch_length (tensor) and returning log probability (tensor)
    """
    # Build neutral Q once — unit selection factors, branch-length-independent
    unit_selection = torch.ones_like(sel_matrix)
    Q_neutral = _build_q_matrix(
        codon_rates,
        codon_csps,
        unit_selection,
        single_hit_scale=single_hit_scale,
        double_hit_scale=double_hit_scale,
        triple_hit_scale=triple_hit_scale,
        q_variant=q_variant,
        codon_parent_nt_window=codon_parent_nt_window,
        codon_parent_boundary_mask=codon_parent_boundary_mask,
        codon_boundary_features=codon_boundary_features,
        codon_rate_adjustments=codon_rate_adjustments,
        kmer_table=kmer_table,
    )

    log_sel = torch.log(sel_matrix)

    # Handle ambiguous codons
    valid_codon_mask = child_codon_idxs < 64
    combined_mask = mask & valid_codon_mask
    clamped_child_idxs = torch.clamp(child_codon_idxs, max=63)
    seq_indices = torch.arange(len(child_codon_idxs), device=child_codon_idxs.device)

    def log_pcp_probability(log_branch_length: torch.Tensor) -> torch.Tensor:
        """Compute log PCP probability for a trial branch length."""
        bl = torch.exp(log_branch_length)
        # Get neutral transition probs (no stop zapper — applied by next step)
        neutral_log_probs = mutsel_codon_transition_probs(
            Q_neutral,
            parent_codon_idxs,
            float(bl.item()),
            use_cubic=use_cubic,
            apply_stop_zapper=False,
        )
        # Apply selection post-hoc (applies stop zapper internally)
        log_probs_64 = compute_log_adjusted_codon_probs(
            parent_codon_idxs, neutral_log_probs, log_sel
        )
        child_log_probs = log_probs_64[seq_indices, clamped_child_idxs]
        result = child_log_probs[combined_mask].sum()
        if not torch.isfinite(result):
            raise RuntimeError(
                f"Non-finite log PCP probability in Hybrid BL closure: "
                f"result={result.item()}, bl={float(bl.item()):.3e}"
            )
        return result

    return log_pcp_probability
