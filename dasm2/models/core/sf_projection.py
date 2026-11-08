"""Selection factor projection module.

Projects encoder embeddings to L x 20 log selection factors. This is the
shared projection used by all SF-consuming heads (DASM, Whichmut, etc.)
so that joint training updates a single set of selection factors.

Subclass for alternative projection strategies (e.g., gated mixture, MLP).
"""

from typing import Literal, Optional

import torch
from torch import Tensor, nn

from dasm2.molevol.tensor_ops import log1mexp
from dasm2.sequence.constants import NUM_AMINO_ACIDS

#: Below this ``|S|`` ``log_phi`` uses its Taylor series. The next term,
#: ``S**6 / 181440``, is below 1e-11 here, while the closed form loses
#: float32 precision to cancellation as ``|S|`` shrinks.
LOG_PHI_SERIES_THRESHOLD = 0.1


def wiggle(x: torch.Tensor, beta: float) -> torch.Tensor:
    """A function that when we exp it gives us a function that slopes to 0 at -inf and
    grows sub-linearly as x increases.

    See https://github.com/matsengrp/netam/pull/5#issuecomment-1906665475 for a
    plot.

    Note:
        torch.where evaluates BOTH branches before selecting. To avoid NaN gradients
        from log(x) when x <= 0, we clamp x to min=1.0 for the log branch. This is
        safe because the log branch is only selected when x >= 1, so the clamp has
        no effect on the actual computation - it just prevents inf gradients from
        log(0) that would result in 0 * inf = NaN during backward.
    """
    x_safe_for_log = torch.clamp(x, min=1.0)
    return beta * torch.where(x < 1, x - 1, torch.log(x_safe_for_log))


def log_phi(s: torch.Tensor) -> torch.Tensor:
    """``log phi(S)`` for Halpern-Bruno ``phi(S) = S / (1 - exp(-S))``.

    ``phi`` is 0/0 at ``S = 0``, where ``log phi`` is 0 with slope 1/2.
    Away from 0 this is ``log|S| - log expm1(|S|) + max(S, 0)``, with
    ``log expm1(a) = a + log1mexp(-a)`` so large ``|S|`` cannot overflow; for
    ``|S| < LOG_PHI_SERIES_THRESHOLD`` it is the series
    ``S/2 - S**2/24 + S**4/2880``. The closed form is fed a safe stand-in
    for small ``S`` (the double ``torch.where``), so its unused branch has a
    finite gradient rather than NaN.
    """
    small = s.abs() < LOG_PHI_SERIES_THRESHOLD
    s_safe = torch.where(small, torch.ones_like(s), s)
    abs_s = s_safe.abs()
    log_expm1 = abs_s + log1mexp(-abs_s)
    closed = torch.log(abs_s) - log_expm1 + s_safe.clamp(min=0)
    s2 = s * s
    series = s / 2 - s2 / 24 + s2 * s2 / 2880
    return torch.where(small, series, closed)


def resident_relative_log_phi(F: Tensor, aa_parents_idxs: Tensor) -> Tensor:
    """``log phi(F_y - F_parent)`` per site: the Halpern-Bruno link.

    The parent column is ``log phi(0) = 0``. The clamp only keeps the gather
    in range at X/^/padding sites, whose values no loss reads.

    Args:
        F: ``[B, L, 20]`` float tensor of site fitnesses.
        aa_parents_idxs: ``[B, L]`` parent amino-acid indices.
    """
    gather_index = aa_parents_idxs.clamp(max=NUM_AMINO_ACIDS - 1).unsqueeze(-1)
    return log_phi(F - F.gather(-1, gather_index))


class SelectionFactorProjection(nn.Module):
    """Projects encoder embeddings to L x 20 log selection factors, or to
    site fitnesses F under ``link``.

    Base implementation using a single Linear(D, 20) with optional wiggle
    activation. CoreModel applies gauge zapping after this projection —
    the module returns raw log f (or F under ``link``).

    Args:
        input_dim: Input embedding dimension from sequence encoder.
        num_amino_acids: Number of amino acids to predict (default 20).
        wiggle_beta: Optional beta for wiggle activation. When set, applies
            identically to all loss paths. None disables wiggle (default).
        link: ``"phi"`` makes the output the site fitness F, and
            ``CoreModel`` maps it to ``log phi(F_y - F_parent)`` at the zap,
            where the parent is known. None (default) returns log f directly.
    """

    def __init__(
        self,
        input_dim: int,
        num_amino_acids: int = 20,
        wiggle_beta: Optional[float] = None,
        link: Optional[Literal["phi"]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_amino_acids = num_amino_acids
        self.wiggle_beta = wiggle_beta
        self.link = link
        self.projection = nn.Linear(input_dim, num_amino_acids)

    def forward(self, sequence_embeddings: Tensor) -> Tensor:
        """Compute raw (un-zapped) log selection factors, or F under ``link``.

        Args:
            sequence_embeddings: Encoder output [batch_size, seq_len, input_dim].

        Returns:
            Raw log selection factors [batch_size, seq_len, 20]; under
            ``link="phi"``, the site fitness F that ``CoreModel`` maps to
            log f.
        """
        output: Tensor = self.projection(sequence_embeddings)
        if self.wiggle_beta is not None:
            output = wiggle(output, self.wiggle_beta)
        return output
