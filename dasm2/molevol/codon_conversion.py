"""Functions for converting between codon and amino acid probability representations.

Functions for converting mutation probabilities and codon probabilities into
amino acid probability representations, bridging nucleotide-level evolutionary
mechanics with amino acid-level selection modeling.
"""

import torch
from torch import Tensor

from dasm2.codon_table import CODON_AA_INDICATOR_MATRIX

from .mutation_matrices import build_mutation_matrices, codon_probs_of_mutation_matrices
from .tensor_ops import flatten_codons, reshape_for_codons


def codon_idx_to_nt_idxs(codon_indices: Tensor) -> Tensor:
    """Convert flat codon indices (0-64) to nucleotide index triplets.

    Args:
        codon_indices: Tensor of codon indices (...) with values 0-64

    Returns:
        Tensor of nucleotide indices (..., 3) where each position is 0-3

    Example:
        >>> codon_idx = torch.tensor([0, 63])  # AAA and TTT
        >>> nt_idxs = codon_idx_to_nt_idxs(codon_idx)
        >>> nt_idxs
        tensor([[0, 0, 0],  # AAA = (0,0,0)
                [3, 3, 3]]) # TTT = (3,3,3)
    """
    # Codon index is i*16 + j*4 + k where i,j,k are nucleotide indices (0-3)
    # To reverse: k = idx % 4, j = (idx // 4) % 4, i = idx // 16
    k = codon_indices % 4
    j = (codon_indices // 4) % 4
    i = codon_indices // 16

    # Stack along new dimension
    return torch.stack([i, j, k], dim=-1)


def aaprobs_of_codon_probs(codon_probs: Tensor) -> Tensor:
    """Convert codon probabilities to amino acid probabilities via summation.

    Sums the probabilities of all codons encoding each amino acid. Validates that
    inputs are valid probability distributions (non-negative, sum to 1, zero stop
    codon probabilities).

    Args:
        codon_probs: Codon probabilities. Can be:
                    - 4D tensor (codon_count, 4, 4, 4) - will be flattened
                    - ND tensor with last dim 64 - used directly
                    Must be valid probability distributions.

    Returns:
        Tensor with shape (..., 20) containing amino acid probabilities

    Raises:
        ValueError: If codon probabilities are not valid probability distributions:
                   - Values outside [0, 1] range
                   - Don't sum to 1 along codon dimension (tolerance 1e-4)
                   - Non-zero stop codon probabilities (tolerance 1e-6)

    Note:
        The output amino acid probabilities will sum to (1 - P(stop codons)).
        Since stop codons must have zero probability, outputs should be normalized.

    Example:
        >>> codon_probs = torch.randn(10, 4, 4, 4)  # 10 sites
        >>> aa_probs = aaprobs_of_codon_probs(codon_probs)
        >>> aa_probs.shape
        torch.Size([10, 20])
    """
    # Handle 4D codon tensors (codon_count, 4, 4, 4)
    if codon_probs.ndim >= 3 and codon_probs.shape[-3:] == (4, 4, 4):
        codon_probs = flatten_codons(codon_probs)

    # Matrix multiply to sum codon probs by amino acid
    # CODON_AA_INDICATOR_MATRIX is (64, 20)
    aa_probs = codon_probs @ CODON_AA_INDICATOR_MATRIX.to(codon_probs.device)

    return aa_probs


def codon_probs_of_parent_scaled_nt_rates_and_csps(
    parent_idxs: torch.Tensor, scaled_nt_rates: torch.Tensor, nt_csps: torch.Tensor
) -> torch.Tensor:
    """Compute the probabilities of mutating to various codons for a parent sequence.

    This uses the same machinery as we use for fitting the DNSM, but we stay on
    the codon level rather than moving to syn/nonsyn changes.

    Args:
        parent_idxs (torch.Tensor): The parent nucleotide sequence encoded as a
            tensor of length Cx3, where C is the number of codons, containing the nt indices of each site.
        scaled_nt_rates (torch.Tensor): Poisson rates of mutation per site, scaled by branch length.
        nt_csps (torch.Tensor): Conditional substitution probabilities per site: a 2D tensor with shape (site_count, 4).

    Returns:
        torch.Tensor: A 4D tensor with shape (codon_count, 4, 4, 4) where the cijk-th entry is the probability
            of the c'th codon mutating to the codon ijk.
    """
    mut_probs = -torch.expm1(-scaled_nt_rates)
    parent_codon_idxs = reshape_for_codons(parent_idxs)
    codon_mut_probs = reshape_for_codons(mut_probs)
    codon_csps = reshape_for_codons(nt_csps)

    mut_matrices = build_mutation_matrices(
        parent_codon_idxs, codon_mut_probs, codon_csps
    )
    codon_probs = codon_probs_of_mutation_matrices(mut_matrices)

    return codon_probs
