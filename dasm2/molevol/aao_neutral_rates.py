"""AA-level neutral rate computation for AAOWhichmut."""

import torch
from torch import Tensor

from dasm2.codon_table import CODON_AA_INDICATOR_MATRIX


def marginalize_codon_to_aao_rates(
    codon_neutral_rates: Tensor,
) -> Tensor:
    """Marginalize codon-level neutral rates to AA-level.

    For each position, sums rates over all child codons that translate
    to each target amino acid. This enables AAO Whichmut loss computation
    without tracking individual codon transitions.

    The marginalization is:
        λ_{j, a'} = Σ_{c'} λ_{j, c→c'}  for all c' where translate(c') = a'

    Note: The parent codon identity is already encoded in the codon_neutral_rates,
    which are computed from parent-specific nucleotide context. We only need to sum
    over child codons by their translated amino acid.

    Args:
        codon_neutral_rates: Neutral rates TO each child codon.
            Shape (N, L, 64) where entry [i, j, k] is the rate of
            position j in sequence i mutating to child codon k.

    Returns:
        aao_neutral_rates: Shape (N, L, 20) where entry [i, j, a'] is the
        total neutral rate of position j in sequence i mutating to AA a'.
        This includes the diagonal (synonymous) rate λ_{a→a}.
    """
    device = codon_neutral_rates.device

    # CODON_AA_INDICATOR_MATRIX is (64, 20): indicator[k, a] = 1 if codon k encodes AA a
    # Matrix multiply to sum rates over codons that encode each AA
    indicator = CODON_AA_INDICATOR_MATRIX.to(device)  # (64, 20)

    # (N, L, 64) @ (64, 20) → (N, L, 20)
    aao_neutral_rates = torch.matmul(codon_neutral_rates, indicator)

    return aao_neutral_rates
