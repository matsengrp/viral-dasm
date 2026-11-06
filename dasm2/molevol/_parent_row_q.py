"""Parent-row-degenerate Q-matrix builders.

The ``parent_row_degenerate`` Q variant: every row of ``Q_ℓ`` shares the
parent codon's per-position Thrifty rates/CSPs (as opposed to the
``global_13mer`` variant whose rows vary with the starting codon).

Lives in its own module so ``kmer_q`` can import the parent-row
selection-ratio helper without re-introducing the cycle that
arose when both modules sat in ``q_matrix``.
"""

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
    SENSE_CODON_TO_AA,
)


def aa_selection_ratio(
    aa_selection_factors: Tensor,
    source_codons: Tensor,
    target_codons: Tensor,
    sense_to_aa: Tensor,
) -> Tensor:
    """Per-edge selection ratio ``f[AA(target)] / f[AA(source)]``.

    QDASM scales each off-diagonal Q entry by the ratio of the target and
    source amino-acid selection factors (the mutation--selection fitness-ratio
    form), not by the target factor alone. This generalizes the
    wildtype gauge (``f[parent] = 1`` from the diagonal zap) from the parent
    codon to every source codon: synonymous edges are neutral at every AA
    (``f[a]/f[a] = 1``), and edges leaving the parent recover the target-only
    value (``f[AA(target)]/f[parent] = f[AA(target)]``), so the leading
    single-substitution likelihood is unchanged and only multi-step paths are
    corrected.

    Args:
        aa_selection_factors: AA selection factors (NOT log), ``[..., 20]``
            (``[L, 20]`` unbatched or ``[B, L, 20]`` batched).
        source_codons: Source (row) sense-codon indices ``[E]``.
        target_codons: Target (column) sense-codon indices ``[E]``.
        sense_to_aa: Sense-codon → AA index map ``[61]``, on the same device.

    Returns:
        Ratio tensor ``[..., E]`` broadcasting over any leading batch dims.
    """
    source_factors = aa_selection_factors[..., sense_to_aa[source_codons]]
    # Fail fast: float division by zero yields ``inf`` in PyTorch (no exception),
    # which propagates into Q and silently corrupts ``exp(tQ)`` — the exact hazard
    # this ratio form exists to manage. QDASM's head applies ``selection_log_floor``
    # so ``f_source >= e^floor > 0``, but the Q-builders accept raw factors from any
    # caller (tests, simulation, future heads) that may bypass the floor, so guard
    # at the shared division site. Pays one reduction per hit-class per Q-build.
    assert (source_factors > 0).all(), (
        "Source AA selection factors must be strictly positive; a zero/negative "
        "f_source makes 1/f_source non-finite. Apply a floor (e.g. the head's "
        "selection_log_floor) before building the mutation-selection Q."
    )
    return aa_selection_factors[..., sense_to_aa[target_codons]] / source_factors


def build_mutsel_q_matrices(
    codon_rates: Tensor,
    codon_csps: Tensor,
    aa_selection_factors: Tensor,
    single_hit_scale: float = 1.0,
    double_hit_scale: float = 0.02,
    triple_hit_scale: float = 0.001,
) -> Tensor:
    """Build Q-matrix (rate matrix) for mutation-selection codon model.

    Constructs a 61×61 instantaneous rate matrix for each site, where off-diagonal
    entries encode the instantaneous rate of transitioning between sense codons.
    For codons differing at k positions (k=1,2,3), the rate is the product of
    per-position mutation rates × CSPs × the ratio of the target-to-source
    amino acid selection factors (``f[AA(target)]/f[AA(source)]``), attenuated
    by scale factors for multi-hit entries.

    Args:
        codon_rates: Per-position mutation rates [L, 3]
        codon_csps: Per-position conditional substitution probabilities [L, 3, 4]
        aa_selection_factors: Amino acid selection factors (NOT log) [L, 20]
        single_hit_scale: Scale factor (γ) for 1-nucleotide-difference Q
            entries. Attenuates instantaneous single-hit rates; γ<1 pulls
            QDASM toward DASM-Poisson behavior.
        double_hit_scale: Scale factor for 2-nucleotide-difference entries.
            Attenuates instantaneous multi-hit rates to keep departure rates
            compatible with phylogenetic branch lengths.
        triple_hit_scale: Scale factor for 3-nucleotide-difference entries.

    Returns:
        Q matrices [L, 61, 61] with rows summing to 0
    """
    L = codon_rates.shape[0]
    device = codon_rates.device
    Q = torch.zeros((L, NUM_SENSE_CODONS, NUM_SENSE_CODONS), device=device)

    diff_i = SENSE_CODON_DIFF_I.to(device)
    diff_j = SENSE_CODON_DIFF_J.to(device)
    diff_pos = SENSE_CODON_DIFF_POS.to(device)
    diff_target_nt = SENSE_CODON_DIFF_TARGET_NT.to(device)
    sense_to_aa = SENSE_CODON_TO_AA.to(device)

    # --- Single-nucleotide differences ---
    # Rate = rate[pos] * csp[pos, target_nt] * f[AA(target)]/f[AA(source)]
    rates_1 = codon_rates[:, diff_pos]  # [L, N]
    csps_1 = codon_csps[:, diff_pos, diff_target_nt]  # [L, N]
    sel_1 = aa_selection_ratio(
        aa_selection_factors, diff_i, diff_j, sense_to_aa
    )  # [L, N]
    Q[:, diff_i, diff_j] = single_hit_scale * rates_1 * csps_1 * sel_1

    # --- 2-nucleotide differences ---
    if len(SENSE_CODON_2DIFF_I) > 0:
        i2 = SENSE_CODON_2DIFF_I.to(device)
        j2 = SENSE_CODON_2DIFF_J.to(device)
        pos2 = SENSE_CODON_2DIFF_POS.to(device)  # [M, 2]
        nts2 = SENSE_CODON_2DIFF_TARGET_NTS.to(device)  # [M, 2]

        rates_2 = codon_rates[:, pos2[:, 0]] * codon_rates[:, pos2[:, 1]]  # [L, M]
        csps_2 = (
            codon_csps[:, pos2[:, 0], nts2[:, 0]]
            * codon_csps[:, pos2[:, 1], nts2[:, 1]]
        )  # [L, M]
        sel_2 = aa_selection_ratio(aa_selection_factors, i2, j2, sense_to_aa)  # [L, M]
        Q[:, i2, j2] = double_hit_scale * rates_2 * csps_2 * sel_2

    # --- 3-nucleotide differences ---
    if len(SENSE_CODON_3DIFF_I) > 0:
        i3 = SENSE_CODON_3DIFF_I.to(device)
        j3 = SENSE_CODON_3DIFF_J.to(device)
        pos3 = SENSE_CODON_3DIFF_POS.to(device)  # [K, 3]
        nts3 = SENSE_CODON_3DIFF_TARGET_NTS.to(device)  # [K, 3]

        rates_3 = (
            codon_rates[:, pos3[:, 0]]
            * codon_rates[:, pos3[:, 1]]
            * codon_rates[:, pos3[:, 2]]
        )  # [L, K]
        csps_3 = (
            codon_csps[:, pos3[:, 0], nts3[:, 0]]
            * codon_csps[:, pos3[:, 1], nts3[:, 1]]
            * codon_csps[:, pos3[:, 2], nts3[:, 2]]
        )  # [L, K]
        sel_3 = aa_selection_ratio(aa_selection_factors, i3, j3, sense_to_aa)  # [L, K]
        Q[:, i3, j3] = triple_hit_scale * rates_3 * csps_3 * sel_3

    # Set diagonal: Q[i,i] = -sum_{j≠i} Q[i,j]. Float32 cancellation
    # leaves a ~1e-7 residual; the downstream P-row-sum check at
    # mutsel_codon_transition_probs catches anything larger.
    Q.diagonal(dim1=-2, dim2=-1).copy_(-Q.sum(dim=-1))

    return Q


def _build_mutsel_q_matrices_batched(
    codon_rates: Tensor,
    codon_csps: Tensor,
    aa_selection_factors: Tensor,
    single_hit_scale: float = 1.0,
    double_hit_scale: float = 0.02,
    triple_hit_scale: float = 0.001,
) -> Tensor:
    """Build Q-matrices for a batch of sequences, fully vectorized over B×L.

    Same computation as build_mutsel_q_matrices but operates on batched inputs
    to avoid a Python loop over the batch dimension.

    Args:
        codon_rates: Per-position mutation rates [B, L, 3]
        codon_csps: Per-position CSPs [B, L, 3, 4]
        aa_selection_factors: AA selection factors (NOT log) [B, L, 20]
        single_hit_scale: Scale factor (γ) for 1-nucleotide-difference Q
            entries. Attenuates instantaneous single-hit rates.
        double_hit_scale: Scale factor for 2-nucleotide-difference entries.
        triple_hit_scale: Scale factor for 3-nucleotide-difference entries.

    Returns:
        Q matrices [B, L, 61, 61] with rows summing to 0
    """
    B, L = codon_rates.shape[:2]
    device = codon_rates.device
    Q = torch.zeros((B, L, NUM_SENSE_CODONS, NUM_SENSE_CODONS), device=device)

    diff_i = SENSE_CODON_DIFF_I.to(device)
    diff_j = SENSE_CODON_DIFF_J.to(device)
    diff_pos = SENSE_CODON_DIFF_POS.to(device)
    diff_target_nt = SENSE_CODON_DIFF_TARGET_NT.to(device)
    sense_to_aa = SENSE_CODON_TO_AA.to(device)

    rates_1 = codon_rates[:, :, diff_pos]  # [B, L, N]
    csps_1 = codon_csps[:, :, diff_pos, diff_target_nt]  # [B, L, N]
    sel_1 = aa_selection_ratio(
        aa_selection_factors, diff_i, diff_j, sense_to_aa
    )  # [B, L, N]
    Q[:, :, diff_i, diff_j] = single_hit_scale * rates_1 * csps_1 * sel_1

    if len(SENSE_CODON_2DIFF_I) > 0:
        i2 = SENSE_CODON_2DIFF_I.to(device)
        j2 = SENSE_CODON_2DIFF_J.to(device)
        pos2 = SENSE_CODON_2DIFF_POS.to(device)
        nts2 = SENSE_CODON_2DIFF_TARGET_NTS.to(device)

        rates_2 = (
            codon_rates[:, :, pos2[:, 0]] * codon_rates[:, :, pos2[:, 1]]
        )  # [B, L, M]
        csps_2 = (
            codon_csps[:, :, pos2[:, 0], nts2[:, 0]]
            * codon_csps[:, :, pos2[:, 1], nts2[:, 1]]
        )  # [B, L, M]
        sel_2 = aa_selection_ratio(
            aa_selection_factors, i2, j2, sense_to_aa
        )  # [B, L, M]
        Q[:, :, i2, j2] = double_hit_scale * rates_2 * csps_2 * sel_2

    if len(SENSE_CODON_3DIFF_I) > 0:
        i3 = SENSE_CODON_3DIFF_I.to(device)
        j3 = SENSE_CODON_3DIFF_J.to(device)
        pos3 = SENSE_CODON_3DIFF_POS.to(device)
        nts3 = SENSE_CODON_3DIFF_TARGET_NTS.to(device)

        rates_3 = (
            codon_rates[:, :, pos3[:, 0]]
            * codon_rates[:, :, pos3[:, 1]]
            * codon_rates[:, :, pos3[:, 2]]
        )  # [B, L, K]
        csps_3 = (
            codon_csps[:, :, pos3[:, 0], nts3[:, 0]]
            * codon_csps[:, :, pos3[:, 1], nts3[:, 1]]
            * codon_csps[:, :, pos3[:, 2], nts3[:, 2]]
        )  # [B, L, K]
        sel_3 = aa_selection_ratio(
            aa_selection_factors, i3, j3, sense_to_aa
        )  # [B, L, K]
        Q[:, :, i3, j3] = triple_hit_scale * rates_3 * csps_3 * sel_3

    # Set diagonal: Q[i,i] = -sum_{j≠i} Q[i,j]. See build_mutsel_q_matrices
    # for the float32-residual rationale.
    Q.diagonal(dim1=-2, dim2=-1).copy_(-Q.sum(dim=-1))

    return Q


def build_mutsel_q_from_codon_q(
    neutral_codon_q: Tensor,
    aa_selection_factors: Tensor,
    single_hit_scale: float = 1.0,
    double_hit_scale: float = 1.0,
    triple_hit_scale: float = 1.0,
) -> Tensor:
    """Build mutation-selection Q from a full neutral codon generator.

    The ``shm1_codon`` Q variant. Unlike ``build_mutsel_q_matrices`` — which
    *reconstructs* each neutral codon rate from per-position nt rate × CSP
    products — this takes a precomputed 61×61 neutral codon generator
    ``neutral_codon_q`` (Erick's context-free, non-reversible OOF empirical
    SHM-1 matrix) and modulates every off-diagonal edge by the amino-acid
    selection ratio ``f[AA(target)]/f[AA(source)]``:

        Q[i→j] = scale_k · neutral_codon_q[i→j] · f[AA(j)]/f[AA(i)]

    where ``k`` is the number of differing nucleotides (1/2/3). Because
    ``neutral_codon_q`` is itself a CTMC generator that already carries the
    empirical multi-hit rates, the hit-class scales default to 1 — no
    Poisson↔CTMC recalibration is needed, in contrast to the Thrifty-derived
    ``single_hit_scale=0.566`` the parent-row/13-mer variants use to pull the
    reconstructed rates toward DASM-Poisson behavior. The scales are retained
    as knobs for ablation.

    ``neutral_codon_q`` is constant across sites (context-free), so it is
    supplied once (e.g. a loss buffer) rather than per-position like
    ``codon_rates``/``codon_csps``.

    Args:
        neutral_codon_q: Neutral codon generator ``[61, 61]`` in dasm2
            sense-61 codon order — off-diagonals ≥ 0, rows summing to ~0.
        aa_selection_factors: AA selection factors (NOT log), ``[..., 20]``
            (``[L, 20]`` unbatched or ``[B, L, 20]`` batched).
        single_hit_scale: γ, 1-nt-difference multiplier.
        double_hit_scale: δ, 2-nt-difference multiplier.
        triple_hit_scale: ψ, 3-nt-difference multiplier.

    Returns:
        Q matrices ``[..., 61, 61]`` with rows summing to 0.
    """
    device = aa_selection_factors.device
    lead_shape = aa_selection_factors.shape[:-1]
    n_aa = aa_selection_factors.shape[-1]
    # Flatten any leading (batch/length) dims so the sparse edge assignment is
    # a single 2-D scatter regardless of batched vs unbatched callers.
    factors = aa_selection_factors.reshape(-1, n_aa)  # [N, 20]
    n = factors.shape[0]

    nq = neutral_codon_q.to(device)
    sense_to_aa = SENSE_CODON_TO_AA.to(device)
    Q = torch.zeros((n, NUM_SENSE_CODONS, NUM_SENSE_CODONS), device=device)

    # --- Single-nucleotide differences ---
    diff_i = SENSE_CODON_DIFF_I.to(device)
    diff_j = SENSE_CODON_DIFF_J.to(device)
    sel_1 = aa_selection_ratio(factors, diff_i, diff_j, sense_to_aa)  # [N, E1]
    Q[:, diff_i, diff_j] = single_hit_scale * nq[diff_i, diff_j] * sel_1

    # --- 2-nucleotide differences ---
    if len(SENSE_CODON_2DIFF_I) > 0:
        i2 = SENSE_CODON_2DIFF_I.to(device)
        j2 = SENSE_CODON_2DIFF_J.to(device)
        sel_2 = aa_selection_ratio(factors, i2, j2, sense_to_aa)
        Q[:, i2, j2] = double_hit_scale * nq[i2, j2] * sel_2

    # --- 3-nucleotide differences ---
    if len(SENSE_CODON_3DIFF_I) > 0:
        i3 = SENSE_CODON_3DIFF_I.to(device)
        j3 = SENSE_CODON_3DIFF_J.to(device)
        sel_3 = aa_selection_ratio(factors, i3, j3, sense_to_aa)
        Q[:, i3, j3] = triple_hit_scale * nq[i3, j3] * sel_3

    # Diagonal: Q[i,i] = -sum_{j≠i} Q[i,j] (generator form).
    Q.diagonal(dim1=-2, dim2=-1).copy_(-Q.sum(dim=-1))

    return Q.reshape(*lead_shape, NUM_SENSE_CODONS, NUM_SENSE_CODONS)
