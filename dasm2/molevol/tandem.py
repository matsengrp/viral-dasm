"""Germline-anchored tandem (hit-class) term for the product-of-three neutral.

Issue #1683. The product of three per-position changes has no instantaneous
multi-nucleotide rate, so a 2-nt codon change is O(t²). The tandem term gives
each unordered pair of codon positions (p, q) a joint change at rate
δ_pq·√(r_p r_q), the two new bases drawn from csp_p·csp_q, where r and csp
are the site's per-base departure rate and CSP (dasm2-codon-tex's
parametrization, phyz#3621). The adjacent pairs 1–2 and 2–3 run at δ; the
outer pair 1–3 at δ_13.

The pairs are competing first-jump events among themselves (Λ sums the
pairs only, not the per-position rates): with λ_pq = δ_pq·√(r_p r_q) and
Λ = Σ_pq λ_pq, pair (p, q) fires first with probability
τ_pq = (λ_pq / Λ)(1 − exp(−Λ t)), so Σ_pq τ_pq < 1 at any δ and t
(dasm2-codon-tex's competing-risks form). If it fires, p and q take the
tandem's bases and the third position changes as the product of three would
have it. So

    P = P₃ + Σ_pq τ_pq (T_pq − P₃),

with T_pq = csp_p ⊗ csp_q ⊗ M_o, M_o the third position's per-position row.
To first order in t this adds δ·w·t·csp_p·csp_q at the 2-nt targets, the
rate-level term's leading order. Rows still sum to one and no entry goes
negative, at any t (``tests/molevol/test_tandem.py``); the linear first-order
form, δ·w·t·csp_p·csp_q added off-parent with the parent reset to 1 − Σ,
drives the parent cell negative at t ≈ 1.

Stop codons are not zeroed: the term is applied to the raw codon
probabilities that production feeds to the loss.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

TANDEM_PAIRS = ((0, 1), (1, 2), (0, 2))
"""Codon-position pairs in the order of ``TandemTerm.pair_event_probs``'s columns."""


@dataclass(frozen=True)
class TandemTerm:
    """Fixed tandem rates per codon-position pair.

    Args:
        delta: δ, the rate of each adjacent pair (1–2, 2–3) in units of
            √(r_p r_q). Besenbacher et al. 2016's germline anchor is 0.003
            (https://doi.org/10.1371/journal.pgen.1006315).
        delta13: δ_13, the rate of the outer pair (1–3).
    """

    delta: float
    delta13: float = 0.0

    def __post_init__(self) -> None:
        if self.delta < 0 or self.delta13 < 0:
            raise ValueError(
                f"Tandem rates must be non-negative, got delta={self.delta}, "
                f"delta13={self.delta13}"
            )

    @property
    def is_active(self) -> bool:
        """Whether any pair has a non-zero rate."""
        return self.delta > 0 or self.delta13 > 0

    def pair_event_probs(self, codon_rates: Tensor, branch_length: float) -> Tensor:
        """τ per site and pair: the probability that the pair's tandem fires
        first, (λ_pq / Λ)(1 − e^{−Λt}); 0 where Λ = 0.

        Args:
            codon_rates: Per-position departure rates, shape (..., 3).
            branch_length: The row's t.

        Returns:
            Shape (..., 3), columns in ``TANDEM_PAIRS`` order.
        """
        sqrt_rates = codon_rates.sqrt()
        deltas = torch.tensor(
            [self.delta, self.delta, self.delta13],
            dtype=codon_rates.dtype,
            device=codon_rates.device,
        )
        weights = torch.stack(
            [sqrt_rates[..., p] * sqrt_rates[..., q] for p, q in TANDEM_PAIRS], dim=-1
        )
        pair_rates = deltas * weights
        total = pair_rates.sum(dim=-1, keepdim=True)
        fired = -torch.expm1(-branch_length * total)
        share = torch.where(total > 0, pair_rates / total, torch.zeros_like(total))
        return share * fired


def check_tandem_multihit_compatible(
    multihit_model: object | None, tandem: TandemTerm | Tensor | None
) -> None:
    """Raise if a multihit correction and the tandem term are both given.

    The tandem mixture pulls the multihit-corrected P₃ toward T_pq, whose
    third-position row is the uncorrected per-position one, so the result is
    neither model. ``tandem`` is a ``TandemTerm`` or its τ.
    """
    if multihit_model is not None and tandem is not None:
        raise ValueError(
            "multihit_model and the tandem term don't compose: the tandem "
            "mixture would pull the multihit-corrected probabilities toward "
            "an uncorrected target. Pass one or the other."
        )


def apply_tandem_term(
    codon_probs: Tensor,
    parent_codon_idxs: Tensor,
    mut_matrices: Tensor,
    codon_csps: Tensor,
    pair_event_probs: Tensor,
) -> Tensor:
    """Mix each pair's tandem outcome into the product-of-three probabilities.

    Args:
        codon_probs: P₃, shape (N, 4, 4, 4).
        parent_codon_idxs: Parent nucleotide indices, shape (N, 3).
        mut_matrices: Per-position rows M (``build_mutation_matrices``), shape
            (N, 3, 4): 1 − m at the parent base, m·csp elsewhere.
        codon_csps: Per-position CSPs, shape (N, 3, 4). The parent base's
            entry is ignored, as in ``build_mutation_matrices``.
        pair_event_probs: τ from ``TandemTerm.pair_event_probs``, shape (N, 3).

    Returns:
        P₃ + Σ_pq τ_pq (T_pq − P₃), shape (N, 4, 4, 4). The mixture keeps rows
        normalized; the caller's parent reset only absorbs float rounding.
    """
    at_parent = (
        torch.arange(4, device=codon_csps.device) == parent_codon_idxs[..., None]
    )
    jump = codon_csps.masked_fill(at_parent, 0.0)
    result = codon_probs
    for k, pair in enumerate(TANDEM_PAIRS):
        # T_pq: the tandem's new bases at p and q, the third position's row.
        r0, r1, r2 = (jump[:, i] if i in pair else mut_matrices[:, i] for i in range(3))
        outcome = r0[:, :, None, None] * r1[:, None, :, None] * r2[:, None, None, :]
        tau = pair_event_probs[:, k, None, None, None]
        result = result + tau * (outcome - codon_probs)
    return result
