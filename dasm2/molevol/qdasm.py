"""QDASM model: log codon transition probabilities + branch-length closures.

Builds Q with a learned amino-acid selection factor *inside* the rate
matrix (selection-in-Q), then exp(tQ) and reads off codon-transition
log-probabilities. Mirrors ``hybrid.py``, which uses the same Q
construction at unit selection and applies selection post-hoc.

Extracted from ``mutsel.py``.
"""

from typing import Callable, Literal, Optional

import torch
from torch import Tensor

from dasm2.molevol.matrix_exp import _batched_matrix_exp_to_log_probs
from dasm2.molevol.q_matrix import (
    QVariant,
    _build_q_matrix,
    _build_q_matrix_batched,
    mutsel_codon_transition_probs,
    shared_codon_length,
)
from dasm2.molevol.uniformization import (
    DEFAULT_N_TERMS as _UNIFORMIZATION_DEFAULT_N_TERMS,
)


def build_qdasm_q_all(
    codon_rates: Tensor,
    codon_csps: Tensor,
    log_aa_selection_factors: Tensor,
    length: int,
    *,
    single_hit_scale: float = 1.0,
    double_hit_scale: float = 0.02,
    triple_hit_scale: float = 0.001,
    q_variant: QVariant = "parent_row_degenerate",
    codon_parent_nt_window: Optional[Tensor] = None,
    codon_parent_boundary_mask: Optional[Tensor] = None,
    kmer_table: Optional[Tensor] = None,
    neutral_codon_q: Optional[Tensor] = None,
    codon_boundary_features: Optional[Tensor] = None,
    codon_rate_adjustments: Optional[Tensor] = None,
) -> Tensor:
    """Build the batched QDASM Q-matrices ``[B, length, 61, 61]``.

    Factored out of ``compute_qdasm_log_codon_probs`` because ``Q`` does *not*
    depend on branch length: the golden-section BL search rebuilds ``exp(tQ)``
    for ~27 trial branch lengths per chunk, and hoisting this build out of that
    loop (pass the result back via ``precomputed_q``) avoids rebuilding the
    identical ``Q`` — including the 1.34 GB 13-mer gather — every iteration.

    ``length`` is the shared codon length (``min(L_aa, L_codon)``); the caller
    is responsible for computing it and, later, slicing ``parent_codon_idxs``
    to it before the matrix exponential.

    Global k-mer variants require ``codon_parent_nt_window`` and
    ``codon_parent_boundary_mask`` plus their matching lookup:
    one ``kmer_table`` whose width matches the selected global-kmer variant.
    ``codon_rate_adjustments [B, L]`` is required for global k-mer variants.
    See :func:`~dasm2.molevol.kmer_q.build_mutsel_q_matrices_from_kmer_lookup`
    for its scaling contract. ``shm1_codon`` instead requires
    ``neutral_codon_q``.
    """
    aa_selection_factors = torch.exp(log_aa_selection_factors)
    return _build_q_matrix_batched(
        codon_rates[:, :length],
        codon_csps[:, :length],
        aa_selection_factors[:, :length],
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
        neutral_codon_q=neutral_codon_q,
    )


def compute_qdasm_log_codon_probs(
    parent_codon_idxs: Tensor,
    codon_rates: Tensor,
    codon_csps: Tensor,
    log_aa_selection_factors: Tensor,
    branch_lengths: Tensor,
    use_cubic: bool = False,
    single_hit_scale: float = 1.0,
    double_hit_scale: float = 0.02,
    triple_hit_scale: float = 0.001,
    adaptive: bool = False,
    tq_threshold: float = 1.0,
    adaptive_mode: Literal["per_chunk", "per_sequence"] = "per_sequence",
    quadratic_ste: bool = False,
    q_variant: QVariant = "parent_row_degenerate",
    codon_parent_nt_window: Optional[Tensor] = None,
    codon_parent_boundary_mask: Optional[Tensor] = None,
    kmer_table: Optional[Tensor] = None,
    neutral_codon_q: Optional[Tensor] = None,
    long_path: Literal["matrix_exp", "ss_quintic"] = "ss_quintic",
    matexp_method: Literal["adaptive", "uniformization"] = "adaptive",
    n_terms: int = _UNIFORMIZATION_DEFAULT_N_TERMS,
    normalization: Literal["parent_absorbing", "proportional"] = "parent_absorbing",
    precomputed_q: Optional[Tensor] = None,
    codon_boundary_features: Optional[Tensor] = None,
    codon_rate_adjustments: Optional[Tensor] = None,
) -> Tensor:
    """Compute log codon transition probabilities using Q-matrix model.

    Vectorized over the batch dimension: builds Q-matrices per sequence (fast),
    then computes all B×L matrix exponentials in a single GPU call.

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
        adaptive: Use hybrid cubic/exact mode (overrides use_cubic).
        tq_threshold: ``||tQ||_∞`` cutoff for adaptive mode (default 1.0).
            Positions at/below it use the row-only ``_row_expmv`` kernel; the
            hotter tail uses the full ``long_path`` kernel. See
            ``_batched_matrix_exp_to_log_probs``.
        adaptive_mode: Retained for checkpoint/config compat; the parent-row
            adaptive path always routes per-position, so it no longer changes
            results.
        quadratic_ste: Use quadratic straight-through estimator for gradients.
            Exact exp(tQ) in forward, quadratic Taylor surrogate for backward.
            Ignored when use_cubic=True or adaptive=True.
        q_variant: Q-matrix construction variant — one of
            ``"parent_row_degenerate"`` (parent's per-position rates broadcast
            across all 61 rows), ``"global_13mer"`` (per-row Thrifty lookup),
            ``"global_5mer"`` (per-row S5F lookup), or ``"shm1_codon"``
            (context-free codon generator).
        codon_parent_nt_window: Required by both global k-mer variants.
            ``[B, L_codon, 15]`` long tensor of nt indices; positions 6, 7, 8
            are the parent codon. Must share the selection-factor device.
        codon_parent_boundary_mask: Required by both global k-mer variants.
            ``[B, L_codon]`` bool tensor marking sites whose lookup needs the
            source-codon boundary rescans. The five-mer builder narrows flags
            caused only by padding outside its smaller footprint.
        codon_boundary_features: Sparse cached full-segment source features
            ``[..., N, 916]``; required at valid boundary/ambiguous-window sites.
        codon_rate_adjustments: Required per-site multipliers for global k-mer
            variants; rejected for other variants. See
            :func:`~dasm2.molevol.kmer_q.build_mutsel_q_matrices_from_kmer_lookup`
            and ``docs/concepts/neutral-models.md#heavylight-rate-adjustment``.
        kmer_table: Required by global k-mer variants. ``[4^k, 5]`` float32
            lookup tensor with columns (rate, csp_A, csp_C, csp_G, csp_T).
        neutral_codon_q: Required when ``q_variant="shm1_codon"``.
            ``[61, 61]`` context-free neutral codon generator.
        matexp_method: ``"adaptive"`` (default) or ``"uniformization"``
            (issue #838) — see ``_batched_matrix_exp_to_log_probs``.
        n_terms: Uniformization truncation order. Only used when
            ``matexp_method="uniformization"``.
        normalization: Uniformization parent-row normalization convention.
            Only used when ``matexp_method="uniformization"``.
        precomputed_q: Optional prebuilt ``Q [B, L, 61, 61]`` (from
            ``build_qdasm_q_all``). When given, the Q-build is skipped and this
            tensor is used directly — the BL-opt golden-section loop passes it
            to avoid rebuilding the branch-length-independent Q every trial.
            ``codon_rates``/``codon_csps``/``log_aa_selection_factors`` and the
            Q-construction kwargs are then unused. Must match the ``Q`` that the
            same inputs would produce (same ``length`` slice).

    Returns:
        Log codon transition probabilities [B, L_aa, 64]

    Notes:
        Stop-codon parents (``CODON64_TO_SENSE61[idx] == -1``) must be filtered
        upstream by ``ComputeMasks``. The molevol layer asserts this contract
        at dispatch.
    """
    B, L_aa = parent_codon_idxs.shape
    L = shared_codon_length(L_aa, codon_rates)

    # Build Q matrices for all sequences at once (fully vectorized over B×L),
    # unless the caller hoisted the build out of a BL search and handed us the
    # branch-length-independent Q via ``precomputed_q``.
    if precomputed_q is None:
        Q_all = build_qdasm_q_all(
            codon_rates,
            codon_csps,
            log_aa_selection_factors,
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
            neutral_codon_q=neutral_codon_q,
        )
    else:
        # Contract: a hoisted Q must already be sliced to the shared codon
        # length ``L`` (see the ``precomputed_q`` arg). Fail fast here rather
        # than surfacing an opaque shape error deep inside the matrix exp.
        if precomputed_q.shape[:2] != (B, L):
            raise ValueError(
                f"precomputed_q batch/length {tuple(precomputed_q.shape[:2])} != "
                f"expected (B, L)=({B}, {L}); pass a Q from build_qdasm_q_all built "
                "with length=shared_codon_length(L_aa, codon_rates)."
            )
        Q_all = precomputed_q

    # Vectorized matrix exponential: one call for all B×L matrices.
    # Mutual exclusion between quadratic_ste, adaptive, and use_cubic is
    # enforced at QDASMEvolHead.__init__; reaching this call with conflicting
    # flags is a contract violation.
    return _batched_matrix_exp_to_log_probs(
        Q_all,
        parent_codon_idxs[:, :L],
        branch_lengths,
        L_aa,
        quadratic_ste=quadratic_ste,
        use_cubic=use_cubic,
        adaptive=adaptive,
        tq_threshold=tq_threshold,
        adaptive_mode=adaptive_mode,
        long_path=long_path,
        matexp_method=matexp_method,
        n_terms=n_terms,
        normalization=normalization,
    )


def compute_qdasm_log_pcp_probability(
    parent_codon_idxs: Tensor,
    child_codon_idxs: Tensor,
    codon_rates: Tensor,
    codon_csps: Tensor,
    log_aa_selection_factors: Tensor,
    branch_length: float,
    mask: Tensor,
    single_hit_scale: float = 1.0,
    double_hit_scale: float = 0.02,
    triple_hit_scale: float = 0.001,
    use_cubic: bool = False,
    adaptive: bool = False,
    tq_threshold: float = 0.1,
    long_path: Literal["matrix_exp", "ss_quintic"] = "ss_quintic",
    q_variant: QVariant = "parent_row_degenerate",
    codon_parent_nt_window: Optional[Tensor] = None,
    codon_parent_boundary_mask: Optional[Tensor] = None,
    kmer_table: Optional[Tensor] = None,
    neutral_codon_q: Optional[Tensor] = None,
    codon_boundary_features: Optional[Tensor] = None,
    codon_rate_adjustments: Optional[Tensor] = None,
) -> float:
    """Compute log PCP probability for QDASM branch length optimization.

    Unbatched variant for scipy optimization. Parallel to
    compute_dasm_log_pcp_probability().

    Args:
        parent_codon_idxs: Parent codon indices [seq_len]
        child_codon_idxs: Child codon indices [seq_len]
        codon_rates: Per-position mutation rates [seq_len, 3]
        codon_csps: Per-position CSPs [seq_len, 3, 4]
        log_aa_selection_factors: Log AA selection factors [seq_len, 20]
        branch_length: Evolutionary time (scalar)
        mask: Boolean mask indicating valid positions [seq_len]
        single_hit_scale: Scale factor (γ) for 1-nucleotide-difference Q
            entries. Attenuates instantaneous single-hit rates.
        double_hit_scale: Scale factor for 2-nucleotide-difference Q entries.
        triple_hit_scale: Scale factor for 3-nucleotide-difference Q entries.
        use_cubic: Use cubic Taylor approximation for matrix exponential.
        adaptive: Use hybrid cubic/exact mode.
        tq_threshold: ``||tQ||_∞`` threshold for adaptive mode (default 0.1,
            empirical crossover where cubic abs error rises above ss_quintic's).
        long_path: Kernel for the long-||tQ|| adaptive branch — ``"ss_quintic"``
            (default) or ``"matrix_exp"``; matches the batched
            ``compute_qdasm_log_codon_probs`` contract.
        q_variant: Q-matrix construction variant; see
            ``compute_qdasm_log_codon_probs`` for the contract on
            ``codon_parent_nt_window``, ``codon_parent_boundary_mask``,
            and the variant-specific lookup tensor.
        codon_parent_nt_window: Required by both global k-mer variants;
            ``[seq_len, 15]`` long tensor (unbatched variant).
        codon_parent_boundary_mask: Required by both global k-mer variants;
            ``[seq_len]`` bool tensor.
        codon_boundary_features: Sparse cached full-segment source features
            ``[..., N, 916]``; required at valid boundary/ambiguous-window sites.
        codon_rate_adjustments: Required per-site multipliers for global k-mer
            variants; rejected for other variants. See
            :func:`~dasm2.molevol.kmer_q.build_mutsel_q_matrices_from_kmer_lookup`
            and ``docs/concepts/neutral-models.md#heavylight-rate-adjustment``.
        kmer_table: Required by global k-mer variants; ``[4^k, 5]`` float32
            lookup tensor with k=13 or k=5 matching the variant.
        neutral_codon_q: Required when ``q_variant="shm1_codon"``;
            ``[61, 61]`` context-free neutral codon generator.

    Returns:
        Sum of log probabilities for valid masked positions (scalar)
    """
    aa_selection_factors = torch.exp(log_aa_selection_factors)

    Q = _build_q_matrix(
        codon_rates,
        codon_csps,
        aa_selection_factors,
        single_hit_scale=single_hit_scale,
        double_hit_scale=double_hit_scale,
        triple_hit_scale=triple_hit_scale,
        q_variant=q_variant,
        codon_parent_nt_window=codon_parent_nt_window,
        codon_parent_boundary_mask=codon_parent_boundary_mask,
        codon_boundary_features=codon_boundary_features,
        codon_rate_adjustments=codon_rate_adjustments,
        kmer_table=kmer_table,
        neutral_codon_q=neutral_codon_q,
    )
    log_probs_64 = mutsel_codon_transition_probs(
        Q,
        parent_codon_idxs,
        branch_length,
        use_cubic=use_cubic,
        adaptive=adaptive,
        tq_threshold=tq_threshold,
        long_path=long_path,
    )

    # Index child codon probabilities
    seq_len = child_codon_idxs.shape[0]
    seq_indices = torch.arange(seq_len, device=child_codon_idxs.device)

    # Handle ambiguous codons (indices >= 64)
    valid_codon_mask = child_codon_idxs < 64
    clamped_child_idxs = torch.clamp(child_codon_idxs, max=63)

    child_log_probs = log_probs_64[seq_indices, clamped_child_idxs]
    valid_log_probs = child_log_probs[mask & valid_codon_mask]

    return float(valid_log_probs.sum().item())


def qdasm_log_pcp_probability_of(
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
    adaptive: bool = False,
    tq_threshold: float = 0.1,
    long_path: Literal["matrix_exp", "ss_quintic"] = "ss_quintic",
    q_variant: QVariant = "parent_row_degenerate",
    codon_parent_nt_window: Optional[Tensor] = None,
    codon_parent_boundary_mask: Optional[Tensor] = None,
    kmer_table: Optional[Tensor] = None,
    neutral_codon_q: Optional[Tensor] = None,
    codon_boundary_features: Optional[Tensor] = None,
    codon_rate_adjustments: Optional[Tensor] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Construct a closure for QDASM branch length optimization.

    Builds the Q-matrix once from rates/csps/selection, then recomputes only
    exp(tQ) for each trial branch length. Parallel to mutsel_log_pcp_probability_of().

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
        adaptive: Use hybrid cubic/exact mode.
        tq_threshold: ``||tQ||_∞`` threshold for adaptive mode (default 0.1,
            empirical crossover where cubic abs error rises above ss_quintic's).
        long_path: Kernel for the long-||tQ|| adaptive branch — ``"ss_quintic"``
            (default) or ``"matrix_exp"``; matches the batched
            ``compute_qdasm_log_codon_probs`` contract.
        q_variant: Q-matrix construction variant; see
            ``compute_qdasm_log_codon_probs`` for the contract on
            ``codon_parent_nt_window``, ``codon_parent_boundary_mask``,
            and the variant-specific lookup tensor.
        codon_parent_nt_window: Required by both global k-mer variants;
            ``[seq_len, 15]`` long tensor.
        codon_parent_boundary_mask: Required by both global k-mer variants;
            ``[seq_len]`` bool tensor.
        codon_boundary_features: Sparse cached full-segment source features
            ``[..., N, 916]``; required at valid boundary/ambiguous-window sites.
        codon_rate_adjustments: Required per-site multipliers for global k-mer
            variants; rejected for other variants. See
            :func:`~dasm2.molevol.kmer_q.build_mutsel_q_matrices_from_kmer_lookup`
            and ``docs/concepts/neutral-models.md#heavylight-rate-adjustment``.
        kmer_table: Required by global k-mer variants; ``[4^k, 5]`` float32
            lookup tensor with k=13 or k=5 matching the variant.
        neutral_codon_q: Required when ``q_variant="shm1_codon"``;
            ``[61, 61]`` context-free neutral codon generator.

    Returns:
        Callable taking log_branch_length (tensor) and returning log probability (tensor)
    """
    # Build Q once — it doesn't depend on branch length
    Q = _build_q_matrix(
        codon_rates,
        codon_csps,
        sel_matrix,
        single_hit_scale=single_hit_scale,
        double_hit_scale=double_hit_scale,
        triple_hit_scale=triple_hit_scale,
        q_variant=q_variant,
        codon_parent_nt_window=codon_parent_nt_window,
        codon_parent_boundary_mask=codon_parent_boundary_mask,
        codon_boundary_features=codon_boundary_features,
        codon_rate_adjustments=codon_rate_adjustments,
        kmer_table=kmer_table,
        neutral_codon_q=neutral_codon_q,
    )

    # Handle ambiguous codons
    valid_codon_mask = child_codon_idxs < 64
    combined_mask = mask & valid_codon_mask
    clamped_child_idxs = torch.clamp(child_codon_idxs, max=63)
    seq_indices = torch.arange(len(child_codon_idxs), device=child_codon_idxs.device)

    def log_pcp_probability(log_branch_length: torch.Tensor) -> torch.Tensor:
        """Compute log PCP probability for a trial branch length."""
        bl = torch.exp(log_branch_length)
        log_probs_64 = mutsel_codon_transition_probs(
            Q,
            parent_codon_idxs,
            float(bl.item()),
            use_cubic=use_cubic,
            adaptive=adaptive,
            tq_threshold=tq_threshold,
            long_path=long_path,
        )
        child_log_probs = log_probs_64[seq_indices, clamped_child_idxs]
        result = child_log_probs[combined_mask].sum()
        if not torch.isfinite(result):
            raise RuntimeError(
                f"Non-finite log PCP probability in QDASM BL closure: "
                f"result={result.item()}, bl={float(bl.item()):.3e}"
            )
        return result

    return log_pcp_probability
