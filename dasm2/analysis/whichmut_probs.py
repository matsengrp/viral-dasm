"""Whichmut mutation probabilities on one founder sequence.

A DASM's selection factor says how much a mutation is favoured; whichmut's
probability says how often it is actually *taken*, which is the selection
factor weighted by how often the neutral process offers it::

    p_{j,c->c'} = lambda_{j,c->c'} * f_{j,a->a'} / Z

The partition function ``Z`` runs over every structurally valid position and
child codon in the sequence, so a probability here is conditional on exactly
one mutation happening somewhere in the gene -- not on it happening at site
``j``. Subsetting the mutations afterwards does not renormalize: the plotted
rows keep the probabilities the whole sequence gave them.

``Z`` comes from :func:`compute_per_sequence_partition_functions`, the same
function the training loss calls. Writing a second normalizer here is exactly
how an analysis notebook and a trainer come to disagree about what the model
says, so this module composes the loss's own pieces and asserts their
agreement in its tests rather than reimplementing them.

Two consequences of following the loss worth stating, because neither is this
module's choice:

* **Stop codons are excluded from Z.** The partition function masks children
  whose amino acid index is >= 20, so a mutation to a stop is neither a term
  in the sum nor a probability this module can report.
* **Synonymous children are included.** They carry a gauge-fixed selection
  factor of exactly 1, so they contribute their neutral rate alone -- which is
  why the wildtype amino acid has a non-zero probability at every site.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from dasm2.codon_table import (
    N_CHILDREN_PER_PARENT,
    PARENT_TO_CHILDREN_LOOKUP,
    build_stop_codon_indicator_tensor,
    get_codon_to_aa_lookup_tensor,
)
from dasm2.losses.whichmut_loss import compute_per_sequence_partition_functions
from dasm2.molevol.codon_neutral_rates import (
    combine_nt_rates_and_csps,
    compute_codon_mutation_rates,
)
from dasm2.neutral.base import RatesCSPNeutralModel
from dasm2.sequence.constants import AA_STR_SORTED, AMBIGUOUS_CODON_IDX
from dasm2.sequence.conversion import codon_idx_tensor_of_str_ambig

AA_COUNT = 20


@dataclass(frozen=True)
class FounderRates:
    """One founder sequence's neutral side of the whichmut product.

    Attributes:
        codon_rates: ``[L, 64]`` neutral rate to each child codon.
        codon_parent_idxs: ``[L]`` parent codon indices, 64 where ambiguous.
        mask: ``[L]`` positions the partition function may sum over -- codons
            that are neither ambiguous nor stops. A stop *parent* has no
            whichmut term at all, which is a different exclusion from the stop
            *children* the partition function drops.
    """

    codon_rates: torch.Tensor
    codon_parent_idxs: torch.Tensor
    mask: torch.Tensor


def founder_rates(neutral_model: RatesCSPNeutralModel, founder_nt: str) -> FounderRates:
    """Codon-level neutral rates for a founder, by the training path.

    The sequence is the gene's CDS, which is what the archive's
    ``secondary_struct.csv`` is indexed against (``sequential_site`` is
    gene-relative) and what the training Parquet's prep passed in.

    Args:
        neutral_model: A ``RatesCSPNeutralModel`` -- for SARS-CoV-2, the era's
            ``TabularRatesNeutralModel`` archive.
        founder_nt: The clade founder's coding sequence for one gene.

    Returns:
        The :class:`FounderRates` for that sequence.
    """
    rates_list, csps_list = neutral_model.compute_rates_and_csps(founder_nt)
    per_base = combine_nt_rates_and_csps(rates_list[0], csps_list[0])
    codon_parent_idxs = codon_idx_tensor_of_str_ambig(founder_nt)
    codon_rates = compute_codon_mutation_rates(
        codon_parent_idxs.shape[0], per_base, codon_parent_idxs
    )
    stop = build_stop_codon_indicator_tensor().bool()
    clamped = codon_parent_idxs.clamp(0, 63)
    mask = (codon_parent_idxs != AMBIGUOUS_CODON_IDX) & ~stop[clamped]
    return FounderRates(codon_rates, codon_parent_idxs, mask)


def neutral_rate_to_aa(rates: FounderRates) -> torch.Tensor:
    """``[L, 20]`` neutral rate of reaching each amino acid at each site.

    An amino acid substitution can be reachable by more than one single
    nucleotide change -- two of the six arginine codons sit one base from
    lysine's AAA -- so the children encoding one amino acid have their rates
    summed. Every mask the partition function applies is applied here too, so
    that ``(selection_factors * this).sum()`` *is* the partition function; the
    test suite asserts it.
    """
    device = rates.codon_rates.device
    parent_to_children = PARENT_TO_CHILDREN_LOOKUP.to(device)
    n_children = N_CHILDREN_PER_PARENT.to(device)
    codon_to_aa = get_codon_to_aa_lookup_tensor(device)

    clamped = rates.codon_parent_idxs.clamp(0, 63)
    children = parent_to_children[clamped]
    children_clamped = children.clamp(0, 63)
    gathered = torch.gather(rates.codon_rates, 1, children_clamped)
    child_aa = codon_to_aa[children_clamped]

    max_children = parent_to_children.shape[1]
    within_range = torch.arange(max_children, device=device).view(
        1, max_children
    ) < n_children[clamped].unsqueeze(-1)
    valid = (
        (children >= 0)
        & (child_aa < AA_COUNT)
        & within_range
        & (gathered > 0)
        & rates.mask.unsqueeze(-1)
    )

    out = torch.zeros(
        rates.codon_rates.shape[0], AA_COUNT, dtype=gathered.dtype, device=device
    )
    out.scatter_add_(1, child_aa.clamp(0, AA_COUNT - 1), gathered * valid)
    return out


def aa_mutation_probabilities(
    selection_factors: torch.Tensor, rates: FounderRates
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whichmut probability of every amino acid mutation on one founder.

    Args:
        selection_factors: ``[L, 20]`` in **linear** space, as
            ``selection_factors_of_aa_str`` returns them. A log factor is the
            mistake this is built to catch, since both are float tensors of
            the same shape; the negativity check below is what catches it.
        rates: That founder's :class:`FounderRates`.

    Returns:
        ``(probabilities, rate_to_aa)``, both ``[L, 20]``. The probabilities
        sum to 1 over the whole sequence, not over any site or any subset.

    Raises:
        ValueError: If the shapes disagree, if any factor is negative -- which
            a linear selection factor cannot be, and a log one routinely is --
            or if the partition function is not positive, which means no valid
            mutation at all and makes every probability undefined rather than
            small.
    """
    if selection_factors.shape != (rates.codon_rates.shape[0], AA_COUNT):
        raise ValueError(
            f"selection_factors has shape {tuple(selection_factors.shape)} but "
            f"the founder has {rates.codon_rates.shape[0]} codons"
        )
    # viral-dasm's one divergence from the source commit; see
    # `dasm2/PROVENANCE.md`. A selection factor is a ratio of rates, so it
    # cannot be negative, while a *log* factor is below zero wherever the
    # mutation is deleterious -- which is most of them. Passing logs here is
    # otherwise undetectable: both are float tensors of the same shape, and
    # the product with a rate is a plausible-looking number.
    #
    # It is a guard rather than a conversion on purpose. Exponentiating
    # whatever arrives would make the function accept both and silently pick
    # one reading of its own input.
    negative = int((selection_factors < 0).sum())
    if negative:
        raise ValueError(
            f"{negative} of {selection_factors.numel()} selection factors are "
            f"negative, which a linear factor cannot be. These look like log "
            f"selection factors; `aa_mutation_probabilities` wants linear "
            f"ones, as `selection_factors_of_aa_str` returns them."
        )
    rate_to_aa = neutral_rate_to_aa(rates)
    partition = compute_per_sequence_partition_functions(
        selection_factors.unsqueeze(0),
        rates.codon_rates.unsqueeze(0),
        rates.codon_parent_idxs.unsqueeze(0),
        rates.mask.unsqueeze(0),
    )
    z = float(partition[0])
    if not z > 0:
        raise ValueError(
            f"partition function is {z}, so no mutation has both a positive "
            f"neutral rate and a valid child codon"
        )
    return selection_factors * rate_to_aa / z, rate_to_aa


def build_probability_dataframe(
    probabilities: torch.Tensor,
    rate_to_aa: torch.Tensor,
    founder_aa: str,
) -> pd.DataFrame:
    """One row per (site, amino acid), in the SF frame's shape.

    Columns match :func:`dasm2.analysis.viral.build_sf_dataframe_from_tensor`
    where they mean the same thing -- ``aa_mutation``, ``aa_site`` (1-indexed),
    ``mutant_aa``, ``is_synonymous`` -- so the two frames join on the mutation
    without any renaming.

    ``log10_probability`` is ``-inf`` where the probability is zero, which is
    a mutation no single nucleotide change reaches. Those rows are kept: a
    caller filtering to its own mutation set should see them and drop them
    deliberately, rather than find them missing.
    """
    probs = probabilities.detach().cpu().numpy()
    neutral = rate_to_aa.detach().cpu().numpy()
    if probs.shape[0] != len(founder_aa):
        raise ValueError(
            f"probabilities have {probs.shape[0]} sites but founder_aa has "
            f"{len(founder_aa)} residues"
        )
    # Imported here rather than at module scope: `viral` imports analysis
    # helpers that pull in plotting, and this module is called from scoring
    # code that has no reason to.
    with np.errstate(divide="ignore"):
        log10 = np.log10(probs)
    records = []
    for site_idx, wt_aa in enumerate(founder_aa):
        aa_site = site_idx + 1
        for aa_idx, mut_aa in enumerate(AA_STR_SORTED):
            records.append(
                {
                    "aa_mutation": f"{wt_aa}{aa_site}{mut_aa}",
                    "aa_site": aa_site,
                    "mutant_aa": mut_aa,
                    "whichmut_probability": float(probs[site_idx, aa_idx]),
                    "log10_whichmut_probability": float(log10[site_idx, aa_idx]),
                    "neutral_rate": float(neutral[site_idx, aa_idx]),
                    "is_synonymous": mut_aa == wt_aa,
                }
            )
    return pd.DataFrame(records)
