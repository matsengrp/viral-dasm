"""Core mutation matrix construction and codon probability calculations.

Core mechanics for building mutation matrices and calculating codon probabilities
from nucleotide-level mutation parameters. These functions form the foundation
of molecular evolution probability calculations.
"""

import torch
from torch import Tensor


def build_mutation_matrices(
    parent_codon_idxs: Tensor, codon_mut_probs: Tensor, codon_csps: Tensor
) -> Tensor:
    """Generate a sequence of 3x4 mutation matrices for parent codons along a sequence.

    Given indices for parent codons, mutation probabilities, and substitution
    probabilities for each parent codon along the sequence, this function
    constructs a sequence of 3x4 matrices. Each matrix in the sequence
    represents the mutation probabilities for each nucleotide position in a
    parent codon. The ijkth entry of the resulting tensor corresponds to the
    probability of the jth nucleotide in the ith parent codon mutating to the
    kth nucleotide (in indices).

    Args:
        parent_codon_idxs (torch.Tensor): 2D tensor with each row containing indices representing
            the parent codon's nucleotides at each site along the sequence.
            Shape should be (codon_count, 3).
        codon_mut_probs (torch.Tensor): 2D tensor representing the mutation probabilities for each site in the codon,
            for each codon along the sequence. Shape should be (codon_count, 3).
        codon_csps (torch.Tensor): 3D tensor representing conditional substitution probabilities for each NT site of each codon along the
            sequence. Shape should be (codon_count, 3, 4).

    Returns:
        torch.Tensor: A 4D tensor with shape (codon_count, 3, 4) where the ijkth entry is the mutation probability
            of the jth position in the ith parent codon mutating to the kth nucleotide.
    """

    codon_count = parent_codon_idxs.shape[0]
    assert parent_codon_idxs.shape[1] == 3, "Each parent codon must be of length 3"

    result_matrices = torch.empty((codon_count, 3, 4))

    # Create a mask with the shape (codon_count, 3, 4) to identify where
    # nucleotides in the parent codon are the same as the nucleotide positions
    # in the new codon. Each row in the third dimension contains a boolean
    # value, which is True if the nucleotide position matches the parent codon
    # nucleotide. How it works: None adds one more dimension to the tensor, so
    # that the shape of the tensor is (codon_count, 3, 1) instead of
    # (codon_count, 3). Then broadcasting automatically expands dimensions where
    # needed. So the arange(4) tensor is automatically expanded to match the
    # (codon_count, 3, 1) shape by implicitly turning it into a (1, 1, 4) shape
    # tensor, where it is then broadcasted to the shape (codon_count, 3, 4) to
    # match the shape of parent_codon_idxs[:, :, None] for equality
    # testing.
    mask_same_nt = torch.arange(4) == parent_codon_idxs[:, :, None]

    # Find the multi-dimensional indices where the nucleotide in the parent
    # codon is the same as the nucleotide in the mutation outcome (i.e., no
    # mutation occurs).
    same_nt_indices = torch.nonzero(mask_same_nt)

    # Using the multi-dimensional indices obtained from the boolean mask, update
    # the mutation probability in result_matrices to be "1.0 -
    # mutation_probability" at these specific positions. This captures the
    # probability of a given nucleotide not mutating.
    result_matrices[
        same_nt_indices[:, 0], same_nt_indices[:, 1], same_nt_indices[:, 2]
    ] = 1.0 - codon_mut_probs[same_nt_indices[:, 0], same_nt_indices[:, 1]]

    # Assign values where the nucleotide is different via broadcasting.
    mask_diff_nt = ~mask_same_nt
    result_matrices[mask_diff_nt] = (codon_mut_probs[:, :, None] * codon_csps)[
        mask_diff_nt
    ]

    return result_matrices


def codon_probs_of_mutation_matrices(mut_matrix: Tensor) -> Tensor:
    """Compute the probability tensor for mutating to the codon ijk along the entire
    sequence.

    Args:
    mut_matrix (torch.Tensor): A 3D tensor representing the mutation matrix for the entire sequence.
                               The shape should be (n_sites, 3, 4), where n_sites is the number of sites,
                               3 is the number of positions in a codon, and 4 is the number of nucleotides.

    Returns:
    torch.Tensor: A 4D tensor where the first axis represents different sites in the sequence and
                  the ijk-th entry of the remaining 3D tensor is the probability of mutating to the codon ijk.
    """
    assert mut_matrix.shape[1] == 3, (
        "The second dimension of the input mut_matrix should be 3 to represent the 3 positions in a codon"
    )
    assert mut_matrix.shape[2] == 4, (
        "The last dimension of the input mut_matrix should be 4 to represent the 4 nucleotides"
    )

    # The key to understanding how this works is that when these tensors are
    # multiplied, PyTorch broadcasts them into a common shape (n_sites, 4, 4, 4),
    # performing element-wise multiplication for each slice along the first axis
    # (i.e., for each site).
    return (
        mut_matrix[:, 0, :, None, None]
        * mut_matrix[:, 1, None, :, None]
        * mut_matrix[:, 2, None, None, :]
    )
