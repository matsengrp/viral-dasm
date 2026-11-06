"""Single-matrix prediction head — 20x20 context-free AA-pair selection."""

from typing import TYPE_CHECKING, Dict

import torch
from torch import Tensor, nn

from dasm2.sequence.constants import NUM_AMINO_ACIDS
from dasm2.utils.tensor import zap_predictions_along_diagonal

from .base import PositionAgnosticHead

if TYPE_CHECKING:
    from dasm2.config import SingleMatrixHeadConfig


def parent_row_log_selection(matrix: Tensor, aa_parents_idxs: Tensor) -> Tensor:
    """Row ``matrix[x_j, :]`` per site, parent column zapped to 0; ``[B, L, 20]``.

    X (idx 20) and ^ (idx 21) parents read a clamped in-table row so the
    gather doesn't IndexError; the zap has its own ``aa_parents_idxs < 20``
    guard and leaves those positions untouched, so callers mask them out.

    Args:
        matrix: ``[20, 20]`` log selection factors indexed (parent, child).
        aa_parents_idxs: ``[B, L]`` parent amino-acid indices.
    """
    rows = matrix[aa_parents_idxs.clamp(0, NUM_AMINO_ACIDS - 1)]
    return zap_predictions_along_diagonal(rows, aa_parents_idxs, fill=0.0)


class SingleMatrixHead(PositionAgnosticHead):
    """Context-free 20x20 log-selection-factor table indexed by parent AA.

    Generalizes ``SingleValueHead`` from one scalar to a learnable
    ``S[parent_aa, child_aa]`` table. The forward indexes row
    ``S[aa_parents_idxs[b, l], :]`` at every site, then zaps the
    parent->parent diagonal to 0 (selection factor 1) so synonymous codon
    substitutions stay neutral. With ``S`` initialized to zeros the head
    is bitwise equivalent to ``SingleValueHead(scalar=0)``; with ``S``
    learned it captures off-diagonal amino-acid-pair biochemistry but no
    positional or contextual structure.

    The diagonal zap mirrors the upstream zap that ``CoreModel`` applies
    to ``DASMEvolHead``'s shared selection-factor projection
    (``dasm2/models/core/base.py`` — ``zap_predictions_along_diagonal``
    on the pre-zap log f before the head sees it). DASM's loss multiplies
    ``exp(S[parent_aa, aa(c)])`` onto every codon ``c`` that codes for
    ``aa(c)``, including non-parent codons that are synonymous to the
    parent. Without zapping, ``S[a, a]`` would scale those synonymous
    codons and act as a per-row syn-codon-usage knob — selection on
    synonymous events. Zapping forces ``S[a, a] = 0``, so the only
    thing the parent-codon = ``1 - sum(others)`` constraint couples back
    to is the row's off-diagonal cells, matching the DASMEvolHead
    convention.

    The 20 diagonal cells of the underlying ``log_selection_matrix``
    parameter still exist (param count is 400 not 380), but they are
    overwritten to 0 in every forward pass, so they receive no gradient
    and stay at their initialization. Effective free parameters: 380.

    Routes through the evolutionary (branch-length) data path; the v4
    YAML schema (Task #6) records the matching loss as
    ``LossType.DASM`` for this head.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__(input_dim=input_dim)
        self.log_selection_matrix = nn.Parameter(
            torch.zeros(NUM_AMINO_ACIDS, NUM_AMINO_ACIDS)
        )

    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Return the parent-row of ``log_selection_matrix`` at every site, diagonal-zapped.

        Ignores ``sequence_embeddings``. Indexes ``log_selection_matrix``
        by ``batch["aa_parents_idxs"]`` (clamped to ``[0, 19]`` to handle
        the X / ``^`` tokens that occupy indices 20+; their contributions
        are zeroed by ``batch["mask"]``), then zaps the parent->parent
        diagonal to 0 so synonymous mutations remain neutral.

        Args:
            sequence_embeddings: IGNORED.
            batch: Must contain ``aa_parents_idxs`` of shape ``[B, L]``
                and ``mask`` of shape ``[B, L]``.

        Returns:
            Log selection factors with shape ``[B, L, 20]``, with the
            parent-AA cell at every (b, l) set to 0.
        """
        # Pattern matches dasm2/models/core/mpnn_light_attention.py; the row
        # mask zeros the X/^ positions the clamped gather reads.
        result = parent_row_log_selection(
            self.log_selection_matrix, batch["aa_parents_idxs"]
        )
        return result * batch["mask"].unsqueeze(-1)

    # to_config() inherited from PositionAgnosticHead


# Set config type after class definition to avoid circular import
from dasm2.config import SingleMatrixHeadConfig  # noqa: E402

SingleMatrixHead.head_config_type = SingleMatrixHeadConfig
