"""Neutral evolution probability calculations for molecular evolution modeling.

Functions for calculating neutral evolution probabilities at both codon and amino acid
levels. These functions model pure evolutionary drift without selection pressure,
providing the baseline probabilities that selection models modify.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from dasm2.codon_table import (
    CODON_AA_INDICATOR_MATRIX,
    STOP_CODON_ZAPPER,
    aa_idxs_of_codon_idxs,
)
from dasm2.molevol.codon_preference import apply_codon_pref_log
from dasm2.molevol.tandem import (
    TandemTerm,
    apply_tandem_term,
    check_tandem_multihit_compatible,
)
from dasm2.molevol.tensor_ops import flatten_codons, set_parent_codon_prob
from dasm2.neutral.hit_class import HitClassModel
from dasm2.sequence.conversion import flatten_codon_idxs, unflatten_codon_idxs
from dasm2.utils.tensor import clamp_probability

from .mutation_matrices import (
    build_mutation_matrices,
    codon_probs_of_mutation_matrices,
)


def zero_stop_codons_and_renormalize(
    codon_probs: Tensor,
    parent_codon_idxs: Optional[Tensor] = None,
    use_parent_adjustment: bool = False,
) -> Tensor:
    """Zero out stop codon probabilities and renormalize.

    Args:
        codon_probs: Codon probabilities (codon_count, 4, 4, 4) or (codon_count, 64)
        parent_codon_idxs: Nucleotide indices (codon_count, 3). Required if use_parent_adjustment=True
        use_parent_adjustment: If True, use DASM2's parent probability adjustment method.
                             If False (default), use standard normalization (divide by sum).
                             Parent adjustment follows netam compatibility, division is DASM2 style.

    Returns:
        Normalized codon probabilities (codon_count, 64) with zero stop codon probs
    """
    # Flatten to (codon_count, 64) if needed
    if codon_probs.ndim == 4:
        codon_probs = flatten_codons(codon_probs)

    # Zero out stop codon probabilities using STOP_CODON_ZAPPER
    # STOP_CODON_ZAPPER is a large negative value at stop codon positions
    codon_probs = codon_probs * STOP_CODON_ZAPPER.exp().to(codon_probs.device)

    if use_parent_adjustment:
        if parent_codon_idxs is None:
            raise ValueError(
                "parent_codon_idxs required when use_parent_adjustment=True"
            )
        # Convert nucleotide indices to flat codon indices using existing utility
        parent_flat_idxs = flatten_codon_idxs(parent_codon_idxs)

        # Renormalize by adjusting parent codon probability
        # Add batch dimension for set_parent_codon_prob which expects [B, L, 64]
        codon_probs_batched = codon_probs.unsqueeze(0)  # [1, codon_count, 64]
        parent_flat_idxs_batched = parent_flat_idxs.unsqueeze(0)  # [1, codon_count]

        # Renormalize
        codon_probs_normalized = set_parent_codon_prob(
            codon_probs_batched, parent_flat_idxs_batched
        )

        # Remove batch dimension
        return codon_probs_normalized.squeeze(0)  # [codon_count, 64]
    else:
        # Standard normalization: divide by sum (netam-compatible)
        return codon_probs / codon_probs.sum(dim=-1, keepdim=True)


def neutral_codon_probs(
    parent_codon_idxs: Tensor,
    codon_mut_probs: Tensor,
    codon_csps: Tensor,
    multihit_model: Optional[HitClassModel] = None,
    codon_pref_log: Optional[Tensor] = None,
    tandem_pair_probs: Optional[Tensor] = None,
) -> Tensor:
    """For every site, what is the probability that the site will mutate to every
    alternate codon?

    Note: This function returns raw codon probabilities including stop codons.
    For properly normalized probabilities with zero stop codons, use
    zero_stop_codons_and_renormalize() on the output.

    Args:
        parent_codon_idxs (torch.Tensor): The parent codons for each sequence. Shape: (codon_count, 3)
        codon_mut_probs (torch.Tensor): The mutation probabilities for each site in each codon. Shape: (codon_count, 3)
        codon_csps (torch.Tensor): The substitution probabilities for each site in each codon. Shape: (codon_count, 3, 4)
        codon_pref_log: Optional log-space codon preference vector, shape (64,).
            When provided, probabilities are multiplied by exp(codon_pref_log)
            and renormalized per position. Used for codon-optimality tables
            (issue #747). Default None is a no-op.
        tandem_pair_probs: Optional per-site tandem event probabilities τ,
            shape (codon_count, 3), from ``TandemTerm.pair_event_probs``
            (issue #1683). When given, the tandem term is mixed in after the
            product of three and the parent codon is reset to 1 − Σ
            off-parent, before the codon preference. Default None is a no-op.

    Returns:
        torch.Tensor: The probability that each site will change to each codon.
                      Shape: (codon_count, 64)
    """
    check_tandem_multihit_compatible(multihit_model, tandem_pair_probs)

    mut_matrices = build_mutation_matrices(
        parent_codon_idxs, codon_mut_probs, codon_csps
    )
    codon_probs = codon_probs_of_mutation_matrices(mut_matrices)

    if multihit_model is not None:
        codon_probs = multihit_model.forward(parent_codon_idxs, codon_probs)

    if tandem_pair_probs is not None:
        codon_probs = apply_tandem_term(
            codon_probs, parent_codon_idxs, mut_matrices, codon_csps, tandem_pair_probs
        )

    codon_probs = codon_probs.view(-1, 64)

    if tandem_pair_probs is not None:
        codon_probs = set_parent_codon_prob(
            codon_probs, flatten_codon_idxs(parent_codon_idxs)
        )

    # Apply the within-AA log-preference and renormalize, delegating to the
    # shared log-space primitive so the cached neutral path and the (future)
    # differentiable loss path stay in sync.
    #
    # The `.any()` guard is load-bearing, not just an optimization: the raw
    # codon_probs here are unnormalized (they do not sum to 1 over the 64
    # codons), and a within-AA-uniform / between-AA-only / zero-strength table
    # produces an all-zero codon_pref_log. Renormalizing in that case would
    # rescale the raw probs, so the no-op-table identity (issue #747 test 1)
    # requires skipping the renormalization entirely. When no table is
    # configured (`codon_pref_log is None`) the `.any()` short-circuits, so the
    # default training path pays no device-sync cost.
    if codon_pref_log is not None and codon_pref_log.any():
        codon_probs = apply_codon_pref_log(torch.log(codon_probs), codon_pref_log).exp()

    return codon_probs


def neutral_aa_probs(
    parent_codon_idxs: Tensor,
    codon_mut_probs: Tensor,
    codon_csps: Tensor,
    multihit_model: Optional[HitClassModel] = None,
) -> Tensor:
    """For every site, what is the probability that the site will mutate to every
    alternate amino acid?

    Note: Like netam, this function converts codon probs to AA probs without
    zeroing stop codons. For validated conversion with stop codon checks,
    use zero_stop_codons_and_renormalize() followed by aaprobs_of_codon_probs().

    Args:
        parent_codon_idxs (torch.Tensor): The parent codons for each sequence. Shape: (codon_count, 3)
        codon_mut_probs (torch.Tensor): The mutation probabilities for each site in each codon. Shape: (codon_count, 3)
        codon_csps (torch.Tensor): The substitution probabilities for each site in each codon. Shape: (codon_count, 3, 4)

    Returns:
        torch.Tensor: The probability that each site will change to each amino acid.
                      Shape: (codon_count, 20)
    """
    codon_probs = neutral_codon_probs(
        parent_codon_idxs,
        codon_mut_probs,
        codon_csps,
        multihit_model=multihit_model,
    )

    # Get the probability of mutating to each amino acid via direct matrix multiplication
    # This matches netam's approach and doesn't validate stop codon probabilities
    aa_probs = codon_probs @ CODON_AA_INDICATOR_MATRIX.to(codon_probs.device)

    return aa_probs


def mut_probs_of_aa_probs(
    parent_codon_idxs: Tensor,
    aa_probs: Tensor,
) -> Tensor:
    """For every site, what is the probability that the amino acid will have a
    substution or mutate to a stop under neutral evolution?

    Args:
        parent_codon_idxs (torch.Tensor): The parent codons for each sequence. Shape: (codon_count, 3)
        aa_probs (torch.Tensor): The probability that each site will change to each amino acid. Shape: (codon_count, 20)
    """
    parent_aa_idxs = aa_idxs_of_codon_idxs(parent_codon_idxs)
    p_staying_same: Tensor = aa_probs[
        (torch.arange(len(parent_aa_idxs)), parent_aa_idxs)
    ]

    return 1.0 - p_staying_same  # type: ignore[no-any-return]


def neutral_aa_mut_probs(
    parent_codon_idxs: Tensor,
    codon_mut_probs: Tensor,
    codon_csps: Tensor,
    multihit_model: Optional[HitClassModel] = None,
) -> Tensor:
    """For every site, what is the probability that the amino acid will have a
    substution or mutate to a stop under neutral evolution?

    This code computes all the probabilities and then indexes into that tensor
    to get the relevant probabilities. This isn't the most efficient way to do
    this, but it's the cleanest. We could make it faster as needed.

    Args:
        parent_codon_idxs (torch.Tensor): The parent codons for each sequence. Shape: (codon_count, 3)
        codon_mut_probs (torch.Tensor): The mutation probabilities for each site in each codon. Shape: (codon_count, 3)
        codon_csps (torch.Tensor): The conditional substitution probabilities for each site in each codon. Shape: (codon_count, 3, 4)

    Returns:
        torch.Tensor: The probability that each site will change to some other amino acid.
                      Shape: (codon_count,)
    """

    aa_probs = neutral_aa_probs(
        parent_codon_idxs,
        codon_mut_probs,
        codon_csps,
        multihit_model=multihit_model,
    )
    mut_probs = mut_probs_of_aa_probs(parent_codon_idxs, aa_probs)
    return mut_probs


def non_stop_neutral_aa_mut_probs(
    nt_parent_idxs: Tensor,
    nt_rates: Tensor,
    nt_csps: Tensor,
    branch_length: float,
    multihit_model: Optional[HitClassModel] = None,
) -> Tensor:
    """For every site, what is the probability that the amino acid will have a non-stop
    substution under neutral evolution?"""
    mut_probs = -torch.expm1(-branch_length * nt_rates)
    parent_codon_idxs = nt_parent_idxs.reshape(-1, 3)
    aa_probs = neutral_aa_probs(
        parent_codon_idxs,
        mut_probs.reshape(-1, 3),
        nt_csps.reshape(-1, 3, 4),
        multihit_model=multihit_model,
    )

    parent_aa_idxs = aa_idxs_of_codon_idxs(parent_codon_idxs)
    aa_probs[torch.arange(len(parent_aa_idxs)), parent_aa_idxs] = 0.0
    # We exclude stops by summing the probabilities of non-wt aa's, instead of
    # taking 1 - p(wt).
    sums = aa_probs.sum(dim=-1)
    return clamp_probability(sums)


def neutral_codon_probs_of_seq(
    parent_codon_idxs: Tensor,
    mask: Tensor,
    nt_rates: Tensor,
    nt_csps: Tensor,
    branch_length: float,
    multihit_model: Optional[HitClassModel] = None,
    codon_pref_log: Optional[Tensor] = None,
    tandem: Optional[TandemTerm] = None,
) -> Tensor:
    """Compute neutral codon probabilities from codon index tensors.

    This is the main sequence-level interface for neutral evolution calculations,
    taking codon indices and returning codon-level neutral probabilities.

    Args:
        parent_codon_idxs: Codon indices [num_codons] with values 0-63
        mask: Boolean tensor indicating valid codons (True = valid, False = separator/padding/ambiguous)
        nt_rates: Mutation rates per nucleotide (fill values at separator positions) [nt_len]
        nt_csps: Conditional substitution probabilities per nucleotide [nt_len, 4]
        branch_length: Branch length for evolution
        multihit_model: Optional multihit model
        codon_pref_log: Optional log-space codon preference vector, shape (64,).
            Passed through to neutral_codon_probs().
        tandem: Optional tandem term. Its τ comes from each
            site's ``nt_rates`` and ``branch_length``. None, or a term with
            both rates zero, is a no-op.

    Returns:
        Neutral codon probabilities [num_codons, 64] with NaN at masked positions.
        Downstream code will convert NaN to SMALL_PROBABILITY before taking logarithms.
    """
    if len(parent_codon_idxs) == 0:
        return torch.empty((0, 64)).float()

    # Handle invalid codon indices (e.g., AMBIGUOUS_CODON_IDX=64, separators)
    # Replace with 0 (AAA) - will be masked out later anyway
    # This is necessary because unflatten_codon_idxs only handles indices 0-63
    parent_codon_idxs_clamped = torch.clamp(parent_codon_idxs, 0, 63)

    # Convert codon indices to nucleotide indices efficiently using lookup table
    # parent_codon_idxs_clamped: [num_codons] → parent_nt_idxs: [num_codons, 3]
    parent_nt_idxs = unflatten_codon_idxs(parent_codon_idxs_clamped)

    # Flatten to get nucleotide-level indices [num_codons * 3]
    parent_idxs = parent_nt_idxs.reshape(-1)

    parent_len = len(parent_idxs)

    # Compute mutation probabilities (NaN values from separators are okay, they'll be masked)
    mut_probs = -torch.expm1(-branch_length * nt_rates[:parent_len])
    nt_csps_truncated = nt_csps[:parent_len, :]
    tandem_pair_probs = None
    if tandem is not None and tandem.is_active:
        tandem_pair_probs = tandem.pair_event_probs(
            nt_rates[:parent_len].reshape(-1, 3), branch_length
        )

    # Compute neutral probabilities for all positions (including separators with NaN)
    neutral_probs = neutral_codon_probs(
        parent_idxs.reshape(-1, 3),
        mut_probs.reshape(-1, 3),
        nt_csps_truncated.reshape(-1, 3, 4),
        multihit_model=multihit_model,
        codon_pref_log=codon_pref_log,
        tandem_pair_probs=tandem_pair_probs,
    )

    # Validate that non-masked positions are finite
    # Masked positions (separators, ambiguous codons) may have NaN - that's expected and correct
    if len(neutral_probs) > 0 and len(mask) > 0:
        valid_mask = mask[: len(neutral_probs)]
        if valid_mask.any() and not torch.isfinite(neutral_probs[valid_mask]).all():
            raise ValueError(
                f"neutral_probs has non-finite values at valid positions. "
                f"parent_codon_idxs shape: {parent_codon_idxs.shape}, "
                f"mask shape: {mask.shape}, "
                f"nt_rates shape: {nt_rates.shape}, "
                f"nt_csps shape: {nt_csps.shape}, "
                f"branch_length: {branch_length}"
            )

    # Clamp probabilities to ensure positive values before taking log later
    # Only clamp valid positions - set NaN at masked positions (separators, ambiguous codons)
    neutral_probs_clamped = neutral_probs.clone()
    if len(neutral_probs) > 0 and len(mask) > 0:
        valid_mask = mask[: len(neutral_probs)]
        # Clamp valid positions to ensure numerical stability
        neutral_probs_clamped[valid_mask] = clamp_probability(neutral_probs[valid_mask])
        # Set masked positions (separators, ambiguous codons) to NaN
        # Downstream code (e.g., _update_neutral_probs) will convert NaN to SMALL_PROBABILITY
        neutral_probs_clamped[~valid_mask] = float("nan")

    return neutral_probs_clamped


def neutral_codon_probs_with_padding(
    parent_codon_idxs: torch.Tensor,
    mask: torch.Tensor,
    nt_rates: torch.Tensor,
    nt_csps: torch.Tensor,
    branch_length: float,
    multihit_model: Optional[HitClassModel] = None,
    target_length: Optional[int] = None,
    codon_pref_log: Optional[torch.Tensor] = None,
    tandem: Optional[TandemTerm] = None,
) -> torch.Tensor:
    """Compute neutral codon probabilities from codon indices with padding to target length.

    Consolidates the pattern: truncation → neutral_codon_probs_of_seq → padding.
    This function handles all the common logic used in both EvolutionaryDataset._update_neutral_probs
    and branch length optimization.

    Args:
        parent_codon_idxs: Parent codon indices [aa_len] with values 0-63
        mask: Boolean mask [aa_len] indicating valid positions
        nt_rates: Neutral model rates [nt_len]
        nt_csps: Neutral model csps [nt_len, 4]
        branch_length: Branch length for probability computation
        multihit_model: Optional multihit model for computing multi-hit probabilities
        target_length: Expected amino acid sequence length to pad/truncate to.
            This should match the max_aa_seq_len from the dataset to ensure
            uniform tensor shapes for batching. Sequences shorter than this
            are padded with small probabilities (1e-10), longer sequences are truncated.
            If None, returns unpadded probabilities.
        codon_pref_log: Optional log-space codon preference vector, shape (64,).
            Passed through to neutral_codon_probs().
        tandem: Optional tandem term, passed through to
            neutral_codon_probs_of_seq().

    Returns:
        Neutral codon probabilities [target_length, 64] with padding applied if target_length specified,
        otherwise [aa_len, 64]
    """
    # Use the actual length that the neutral model processed
    nt_length = nt_rates.shape[0]
    aa_length = nt_length // 3
    truncated_codon_idxs = parent_codon_idxs[:aa_length]
    truncated_mask = mask[:aa_length]

    # Compute neutral codon probabilities
    neutral_codon_probs = neutral_codon_probs_of_seq(
        truncated_codon_idxs,
        truncated_mask,
        nt_rates,
        nt_csps,
        branch_length,
        multihit_model=multihit_model,
        codon_pref_log=codon_pref_log,
        tandem=tandem,
    )

    # Pad or truncate to match target length if specified
    if target_length is not None:
        current_len = neutral_codon_probs.shape[0]
        if current_len < target_length:
            pad_len = target_length - current_len
            neutral_codon_probs = F.pad(
                neutral_codon_probs, (0, 0, 0, pad_len), value=1e-10
            )
        elif current_len > target_length:
            neutral_codon_probs = neutral_codon_probs[:target_length]

    return neutral_codon_probs
