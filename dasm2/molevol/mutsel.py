"""Mutation-selection (mutsel) models for molecular evolution with fitness effects.

Functions for calculating evolution probabilities under mutation-selection models, where
amino acid changes have differential fitness effects. These models combine neutral
mutation probabilities with selection factors to model realistic evolution.
"""

import math
from dataclasses import dataclass
from typing import Callable, Literal, Optional, get_args

import torch
from torch import Tensor

import dasm2.tensors_of_seqs
from dasm2.codon_table import (
    CODON_AA_INDICATOR_MATRIX,
    STOP_CODON_ZAPPER,
)
from dasm2.neutral.hit_class import HitClassModel

from .mutation_matrices import (
    build_mutation_matrices,
    codon_probs_of_mutation_matrices,
)
from .tensor_ops import (
    LOG_SUM_CLAMP_MAX,
    mask_ambiguous_codon_rows,
    set_parent_codon_log_prob,
)

# The two DASM endpoint normalizations (issue #1238). Both build the same
# unnormalized sense-restricted log weight row ``log u_y + log f_{a(y)}``;
# they differ only in the final step that turns it into a distribution.
DASMNormalization = Literal["parent_absorbing", "sense_conditional"]
_VALID_NORMALIZATIONS: tuple[str, ...] = get_args(DASMNormalization)

# Maximum non-parent probability sum, matching the canonical path's clamping.
# The canonical path clamps log(sum_other) at LOG_SUM_CLAMP_MAX before
# computing log(1 - sum_other); this is the equivalent linear-space bound.
_MAX_SUM_OTHER = math.exp(LOG_SUM_CLAMP_MAX)


def build_codon_mutsel(
    parent_codon_idxs: Tensor,
    codon_mut_probs: Tensor,
    codon_csps: Tensor,
    aa_sel_matrices: Tensor,
    multihit_model: Optional[HitClassModel] = None,
) -> tuple[Tensor, Optional[Tensor]]:
    """Build a sequence of codon mutation-selection matrices for codons along a
    sequence.

    These will assign zero for the probability of mutating to a stop codon.

    Args:
        parent_codon_idxs (torch.Tensor): The parent codons for each sequence. Shape: (codon_count, 3)
        codon_mut_probs (torch.Tensor): The mutation probabilities for each site in each codon. Shape: (codon_count, 3)
        codon_csps (torch.Tensor): The substitution probabilities for each site in each codon. Shape: (codon_count, 3, 4)
        aa_sel_matrices (torch.Tensor): The amino-acid selection matrices for each site. Shape: (codon_count, 20)

    Returns:
        torch.Tensor: The probability of mutating to each codon, for each sequence. Shape: (codon_count, 4, 4, 4)
    """
    mut_matrices = build_mutation_matrices(
        parent_codon_idxs, codon_mut_probs, codon_csps
    )
    codon_probs = codon_probs_of_mutation_matrices(mut_matrices)

    if multihit_model is not None:
        codon_probs = multihit_model.forward(parent_codon_idxs, codon_probs)

    # Calculate the codon selection matrix for each sequence via Einstein
    # summation, in which we sum over the repeated indices.
    # So, for each site (s) and codon (c), sum over amino acids (a):
    # codon_sel_matrices[s, c] = sum_a(CODON_AA_INDICATOR_MATRIX[c, a] * aa_sel_matrices[s, a])
    # Resulting shape is (S, C) where S is the number of sites and C is the number of codons.
    # Stop codons don't appear in this sum, so columns for stop codons will be zero.
    codon_sel_matrices = torch.einsum(
        "ca,sa->sc", CODON_AA_INDICATOR_MATRIX, aa_sel_matrices
    )

    # Multiply the codon probabilities by the selection matrices
    codon_mutsel = codon_probs * codon_sel_matrices.view(-1, 4, 4, 4)

    # Clamp the codon_mutsel above by 1: these are probabilities.
    codon_mutsel = codon_mutsel.clamp(max=1.0)

    # Now we need to recalculate the probability of staying in the same codon.
    # In our setup, this is the probability of nothing happening.
    # To calculate this, we zero out the probabilities of mutating to the parent
    # codon...
    codon_count = parent_codon_idxs.shape[0]
    codon_mutsel[(torch.arange(codon_count), *parent_codon_idxs.T)] = 0.0
    # sum together their probabilities...
    sums = codon_mutsel.sum(dim=(1, 2, 3))
    # then set the parent codon probabilities to 1 minus the sum.
    codon_mutsel[(torch.arange(codon_count), *parent_codon_idxs.T)] = 1.0 - sums
    codon_mutsel = codon_mutsel.clamp(min=0.0)

    if sums.max() > 1.0:
        sums_too_big = sums.max()
    else:
        sums_too_big = None

    return codon_mutsel, sums_too_big


def _log_codon_selection_with_stop_zap(log_aa_selection_factors: Tensor) -> Tensor:
    """Lift log AA selection factors to codon space and add the stop zapper.

    The branch-length-independent part of the unnormalized log weight row
    ``log w_y = log u_y + log f_{a(y)} + STOP_CODON_ZAPPER`` that both
    :func:`compute_log_adjusted_codon_probs` and
    :func:`precompute_pcp_constants` build.
    """
    device = log_aa_selection_factors.device
    return log_aa_selection_factors @ CODON_AA_INDICATOR_MATRIX.T.to(
        device
    ) + STOP_CODON_ZAPPER.to(device)


def _sense_conditional_log_probs(log_w: Tensor) -> Tensor:
    """The ``sense_conditional`` kernel: one ``log_softmax`` over the row.

    No clamp: ``min=-80`` would lift the zapped stop columns back to
    exp(-80) (the exact artifact this kernel exists to remove), and
    ``max=0`` is meaningless on unnormalized weights. ``log_softmax``
    subtracts the row max internally and is stable on its own. Shared by
    :func:`compute_log_adjusted_codon_probs` and
    :func:`compute_dasm_log_pcp_fast` so the two cannot drift apart.
    """
    return torch.log_softmax(log_w, dim=-1)


def compute_log_adjusted_codon_probs(
    parent_codon_idxs: Tensor,
    log_neutral_codon_probs: Tensor,
    log_aa_selection_factors: Tensor,
    normalization: DASMNormalization = "parent_absorbing",
) -> Tensor:
    """Compute log adjusted codon probabilities combining neutral evolution and selection.

    This is the core DASM calculation that modifies neutral evolutionary probabilities
    with learned amino acid selection factors to capture position-specific fitness effects.

    Handles both batched and unbatched inputs transparently.

    Two normalizations are available (issue #1238). Both start from the
    same unnormalized, stop-zapped log weight row

        log w_y = log u_y + log f_{a(y)} + STOP_CODON_ZAPPER

    and differ only in how that row becomes a distribution:

    ``parent_absorbing`` (default, the classical DASM convention)
        Clamp ``log w`` to ``[-80, 0]``, leave every non-parent entry at
        its (numerically) raw value, and set the exact parent codon to
        ``1 - Σ_{y≠x} p_y`` via :func:`set_parent_codon_log_prob`. All
        stop mass is therefore absorbed into the parent codon, and the
        ``-80`` floor lifts the zapped stop columns back up to
        ``exp(-80) ≈ 1.8e-35`` rather than zero. Normalization is only
        conditional: when ``Σ_{y≠x} p_y > 1`` the parent is floored at
        ``exp(-1e-4)``-complement and the row over-sums (issue #1160).

    ``sense_conditional``
        One ``log_softmax`` over the same row, i.e.

            P(y | x, t, productive) = u_y f_{a(y)} / Σ_{z∈S} u_z f_{a(z)}

        for every sense codon ``y``, the exact parent ``x`` included. Stop
        columns come out **exactly 0** in probability space (``exp(-1e9 -
        logZ)`` underflows), rows sum to 1 unconditionally, and the self
        weight is the *raw* neutral ``u_x`` rather than a residual. With
        all sense factors equal to 1 this reduces to ``ũ_y = u_y / (1 - s)``
        where ``s`` is the raw stop mass.

    **``sense_conditional`` drops the clamp, and the clamp does two jobs.**
    Removing the ``-80`` floor is the point for stop columns, but that
    floor applies to *every* column: under ``sense_conditional`` an
    observed child with a genuinely tiny probability costs its true
    ``-log p`` instead of being capped at 80. That is the more correct
    behavior and is intended, but it is a second behavioral change, not a
    side effect of the stop handling.

    **Stop columns leave ``log_softmax`` at ≈ ``-1e9`` in log space** (the
    probability is exactly 0; only its log is large-negative), where the
    native kernel reports ``-80``. Nothing at this level prevents a stop
    codon from reaching an unmasked *child* position — ``filter_valid_pcps``
    upstream is the only reason it does not happen — so such a position
    would contribute ~1e9 to the NLL instead of 80. Gradients stay finite
    and small either way, but *not* for the same reason, and #1238's body
    gets this wrong. ``CODON_AA_INDICATOR_MATRIX``'s stop rows are all
    zero, so no gradient reaches an AA factor through a stop column's
    *numerator* — which under ``parent_absorbing``, where the column is
    clamped flat to ``-80``, is the whole story and the gradient is
    exactly 0. Under ``sense_conditional`` the column still carries
    ``-log Z``, and ``Z`` depends on every sense factor, so the gradient is
    ``-∂ log Z / ∂ log f`` — nonzero, but bounded by 1 in magnitude per
    factor, so the 1e9 loss value is not a gradient catastrophe.

    The gauge matters under ``sense_conditional``: the conditional kernel
    is invariant to a global rescaling of all 20 selection factors, so the
    parent-AA diagonal zap stops being a convention and becomes the
    identifiability constraint. ``SingleValueHead`` emits nothing *but* a
    global shift and is therefore degenerate under this normalization;
    ``TaskConfig`` rejects that pairing at config load.

    Not to be confused with
    :func:`dasm2.molevol.neutral_evolution.zero_stop_codons_and_renormalize`,
    the pre-existing neutral-side analogue of ``ũ = u / (1 - s)``. That one
    works in linear space and conditions *before* selection is applied,
    which is the wrong order once the denominator has to include the
    selection factors; the two are not redundant.

    IMPORTANT: This function works entirely in log space to avoid gradient explosion.
    The previous implementation did log -> exp -> normalize -> log, which caused
    gradient issues (d/dx log(x) = 1/x explodes for small x). The new implementation
    uses logsumexp and log1mexp to stay in log space throughout.

    Args:
        parent_codon_idxs: Parent codon indices
            Batched: [batch_size, seq_len] or Unbatched: [seq_len]
        log_neutral_codon_probs: Log neutral codon probabilities
            Batched: [batch_size, seq_len, 64] or Unbatched: [seq_len, 64]
        log_aa_selection_factors: Amino acid selection factors in log space
            Batched: [batch_size, seq_len, 20] or Unbatched: [seq_len, 20]
        normalization: Which endpoint kernel to run; see above. Both
            branches use ``parent_codon_idxs`` — ``sense_conditional``
            needs it for ambiguous-row masking even though it never
            singles the parent column out.

    Returns:
        Log adjusted codon probabilities with same shape as log_neutral_codon_probs

    Raises:
        ValueError: If log_neutral_codon_probs is empty, or ``normalization``
            is not one of the two recognized values.

    Steps:
        1. Apply selection factors in log space via CODON_AA_INDICATOR_MATRIX
        2. Add stop codon zapping (sets stop codons to -1e9)
        3. Normalize per ``normalization``
        4. Mask ambiguous-parent rows to -inf
    """
    if len(log_neutral_codon_probs) == 0:
        raise ValueError(
            "Cannot compute adjusted codon probabilities for empty sequence. "
            "log_neutral_codon_probs has length 0."
        )
    if normalization not in _VALID_NORMALIZATIONS:
        raise ValueError(
            f"compute_log_adjusted_codon_probs: normalization must be one of "
            f"{list(_VALID_NORMALIZATIONS)}, got {normalization!r}."
        )

    # Steps 1 & 2: Apply selection factors and stop codon zapping in log
    # space. Shared verbatim by both normalizations — the aa_codon_indicator
    # matrix lifts things up from aa land to codon land.
    log_w = log_neutral_codon_probs + _log_codon_selection_with_stop_zap(
        log_aa_selection_factors
    )

    if normalization == "sense_conditional":
        # Step 3: the whole kernel is one softmax over the sense-restricted
        # row.
        log_preds = _sense_conditional_log_probs(log_w)
        # Step 4: log_softmax over a garbage row at a masked position
        # returns finite junk, so this is load-bearing here, not cosmetic.
        return mask_ambiguous_codon_rows(log_preds, parent_codon_idxs)

    # Step 3: Clamp log probabilities to reasonable range
    # Lower bound: exp(-80) ≈ 1.8e-35, above subnormal threshold
    # Upper bound: 0 (probability <= 1), though individual codons can exceed
    # before normalization. We clamp at 0 to prevent numerical issues.
    log_preds = torch.clamp(log_w, min=-80.0, max=0.0)

    # Step 4: Set parent codon probability in log space (and mask ambiguous
    # rows, which set_parent_codon_log_prob does on the way out).
    # This computes log(p_parent) = log(1 - sum(p_other)) using log1mexp
    # Stays entirely in log space to avoid 1/x gradient explosion
    log_preds = set_parent_codon_log_prob(log_preds, parent_codon_idxs)

    return log_preds


def compute_dasm_log_pcp_probability(
    parent_codon_idxs: Tensor,
    child_codon_idxs: Tensor,
    log_neutral_codon_probs: Tensor,
    log_aa_selection_factors: Tensor,
    mask: Tensor,
    normalization: DASMNormalization = "parent_absorbing",
) -> float:
    """Compute log PCP probability for DASM branch length optimization.

    Canonical implementation: delegates to ``compute_log_adjusted_codon_probs``
    for the full [L, 64] adjusted distribution, then indexes the child codon.
    Use :func:`compute_dasm_log_pcp_fast` for the optimized inner-loop variant.

    Args:
        parent_codon_idxs: Parent codon indices for each position [seq_len]
        child_codon_idxs: Observed child codon indices [seq_len]
        log_neutral_codon_probs: Log neutral codon probabilities at trial branch length [seq_len, 64]
        log_aa_selection_factors: Pre-computed log AA selection factors (branch-length-independent) [seq_len, 20]
        mask: Boolean mask indicating valid positions [seq_len]
        normalization: Endpoint kernel; see
            :func:`compute_log_adjusted_codon_probs`.

    Returns:
        Sum of log probabilities for valid masked positions (scalar to maximize)
    """
    log_adjusted_probs = compute_log_adjusted_codon_probs(
        parent_codon_idxs,
        log_neutral_codon_probs,
        log_aa_selection_factors,
        normalization=normalization,
    )

    child_idx = torch.clamp(child_codon_idxs, max=63).unsqueeze(-1)  # [L, 1]
    child_log_probs = log_adjusted_probs.gather(-1, child_idx).squeeze(-1)  # [L]

    valid = mask & (child_codon_idxs < 64)
    return float(child_log_probs[valid].sum().item())


@dataclass
class PCPConstants:
    """Pre-computed branch-length-independent constants for PCP probability.

    Attributes:
        log_sel_codon: Log selection factors lifted to codon space [seq_len, 64].
        child_idx: Clamped child codon indices [seq_len, 1].
        is_parent: Boolean mask where child == parent [seq_len]. Unused on
            the ``sense_conditional`` branch, where the parent is just
            another column.
        valid: Boolean mask for valid positions [seq_len].
        normalization: Endpoint kernel; see
            :func:`compute_log_adjusted_codon_probs`. Branch-length
            independent, so it rides here rather than being passed on
            every scipy iteration.
    """

    log_sel_codon: Tensor
    child_idx: Tensor
    is_parent: Tensor
    valid: Tensor
    normalization: DASMNormalization = "parent_absorbing"


def precompute_pcp_constants(
    parent_codon_idxs: Tensor,
    child_codon_idxs: Tensor,
    log_aa_selection_factors: Tensor,
    mask: Tensor,
    normalization: DASMNormalization = "parent_absorbing",
) -> PCPConstants:
    """Pre-compute branch-length-independent constants for PCP probability.

    Call once per sequence before the scipy optimization loop. The returned
    ``PCPConstants`` is passed to :func:`compute_dasm_log_pcp_fast` on each
    iteration.

    Args:
        parent_codon_idxs: Parent codon indices [seq_len].
        child_codon_idxs: Observed child codon indices [seq_len].
        log_aa_selection_factors: Log AA selection factors [seq_len, 20].
        mask: Boolean mask [seq_len].
        normalization: Endpoint kernel; see
            :func:`compute_log_adjusted_codon_probs`. The constants
            themselves are identical under both — ``STOP_CODON_ZAPPER`` is
            already folded into ``log_sel_codon`` either way — so this is
            carried through untouched for :func:`compute_dasm_log_pcp_fast`.

    Returns:
        PCPConstants with pre-computed tensors.
    """
    log_sel_codon = _log_codon_selection_with_stop_zap(log_aa_selection_factors)
    child_idx = torch.clamp(child_codon_idxs, max=63).unsqueeze(-1)
    is_parent = child_codon_idxs == parent_codon_idxs
    valid = mask & (child_codon_idxs < 64)
    return PCPConstants(
        log_sel_codon=log_sel_codon,
        child_idx=child_idx,
        is_parent=is_parent,
        valid=valid,
        normalization=normalization,
    )


def compute_dasm_log_pcp_fast(
    log_neutral_codon_probs: Tensor,
    pcp_constants: PCPConstants,
) -> float:
    """Compute log PCP probability using pre-computed constants.

    Under ``parent_absorbing`` this is a fused, linear-space equivalent of
    :func:`compute_dasm_log_pcp_probability` optimized for the scipy inner
    loop: it replaces ``set_parent_codon_log_prob`` (scatter + logsumexp +
    log1mexp) with ``exp`` + ``sum`` + ``gather`` + ``where``, and hoists
    all branch-length-independent work into
    :func:`precompute_pcp_constants`.

    Under ``sense_conditional`` it runs the canonical kernel itself
    (:func:`_sense_conditional_log_probs`) and reads the child column:
    the parent is just another column, so ``PCPConstants.is_parent``, the
    ``_MAX_SUM_OTHER`` cap and the ``1e-35`` floors are all unused on that
    branch. They stay because ``parent_absorbing`` still needs them.

    Must produce identical results to :func:`compute_dasm_log_pcp_probability`
    at the same ``normalization``; this invariant is verified by
    ``test_pcp_fast_matches_canonical``. Note the two paths *do* disagree at
    a position with an ambiguous parent, a valid child and ``mask=True``:
    the canonical path masks such rows to ``-inf`` while neither this
    function nor ``PCPConstants.valid`` inspects parent ambiguity. That is
    pre-existing under both normalizations and invisible only because no
    caller produces the combination.

    Args:
        log_neutral_codon_probs: Log neutral codon probs at trial BL [seq_len, 64].
        pcp_constants: Pre-computed constants from :func:`precompute_pcp_constants`,
            including the ``normalization`` to run.

    Returns:
        Sum of log probabilities (scalar to maximize).
    """
    log_sel_codon = pcp_constants.log_sel_codon
    child_idx = pcp_constants.child_idx
    is_parent = pcp_constants.is_parent
    valid = pcp_constants.valid

    if pcp_constants.normalization == "sense_conditional":
        # No clamp, no parent special-case, no floors: the canonical
        # kernel, read at the child column.
        log_p = _sense_conditional_log_probs(log_neutral_codon_probs + log_sel_codon)
        child_log_p = log_p.gather(-1, child_idx).squeeze(-1)
        return float(child_log_p[valid].sum().item())

    log_preds = log_neutral_codon_probs + log_sel_codon
    log_preds = torch.clamp(log_preds, min=-80.0, max=0.0)

    preds = torch.exp(log_preds)
    sum_all = preds.sum(dim=-1)
    child_p = preds.gather(-1, child_idx).squeeze(-1)

    # Match the canonical path's clamping: cap sum_other so that
    # parent_p >= 1 - _MAX_SUM_OTHER (≈ 1e-4), preventing the huge
    # divergence that occurs when sum_other > 1.
    sum_other = torch.clamp(sum_all - child_p, max=_MAX_SUM_OTHER)
    parent_p = torch.clamp(1.0 - sum_other, min=1e-35)
    child_p = torch.where(is_parent, parent_p, child_p)
    child_p = torch.clamp(child_p, min=1e-35)
    return float(torch.log(child_p[valid]).sum().item())


def mutsel_log_pcp_probability_of(
    sel_matrix: Tensor,
    parent: str,
    child: str,
    nt_rates: Tensor,
    nt_csps: Tensor,
    multihit_model: Optional[HitClassModel] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Constructs the log_pcp_probability function specific to given nt_rates and
    nt_csps.

    This function takes log_branch_length as input and returns the log probability of
    the child sequence. It uses log of branch length to ensure non-negativity.
    """

    assert len(parent) % 3 == 0
    assert sel_matrix.shape == (len(parent) // 3, 20)

    parent_idxs = dasm2.tensors_of_seqs.nt_idx_tensor_of_str(parent)
    child_idxs = dasm2.tensors_of_seqs.nt_idx_tensor_of_str(child)

    def log_pcp_probability(log_branch_length: torch.Tensor) -> torch.Tensor:
        branch_length = torch.exp(log_branch_length)
        nt_mut_probs = -torch.expm1(-branch_length * nt_rates)

        codon_mutsel, sums_too_big = build_codon_mutsel(
            parent_idxs.reshape(-1, 3),
            nt_mut_probs.reshape(-1, 3),
            nt_csps.reshape(-1, 3, 4),
            sel_matrix,
            multihit_model=multihit_model,
        )

        reshaped_child_idxs = child_idxs.reshape(-1, 3)
        child_prob_vector = codon_mutsel[
            torch.arange(len(reshaped_child_idxs)),
            reshaped_child_idxs[:, 0],
            reshaped_child_idxs[:, 1],
            reshaped_child_idxs[:, 2],
        ]

        child_prob_vector = torch.clamp(child_prob_vector, min=1e-10)

        result = torch.sum(torch.log(child_prob_vector))

        assert torch.isfinite(result)

        return result

    return log_pcp_probability
