"""Site-masked DASM head (issue #1550).

Identity head over the shared selection-factor tensor, like
``SelectionFactorHead``. Its head type makes ``models/factory.py`` build a
``SiteMaskedCoreModel`` (``dasm2/models/core/site_masked_core_model.py``),
which owns the masking and hands this head ``log phi(F_j(y) - F_j(x))``
rather than the zapped single-pass projection. The config fields are the
masking parameters that model reads, and ``pair_table``, which gives the
head a 20x20 table ``S`` that the model adds to that term (issue #1670).
"""

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from dasm2.sequence.constants import NUM_AMINO_ACIDS

from .base import SharedSelectionFactorHead
from .single_matrix import parent_row_log_selection

if TYPE_CHECKING:
    from dasm2.config import BaseHeadConfig


class SiteMaskedSelectionFactorHead(SharedSelectionFactorHead):
    """Identity head that selects ``SiteMaskedCoreModel``.

    Args:
        mask_rate: Fraction of maskable sites masked per row in a training
            forward.
        eval_num_partitions: Encoder passes per evaluation batch; each
            maskable site is masked in exactly one.
        partition_seed: Seed of the evaluation partition.
        pair_table: Hold a learnable ``log_selection_matrix`` ``S``, indexed
            and diagonal-zapped as ``SingleMatrixHead``'s is, which the model
            adds at scored sites.
    """

    requires_site_masked_core_model = True

    def __init__(
        self,
        mask_rate: float = 0.15,
        eval_num_partitions: int = 7,
        partition_seed: int = 0,
        pair_table: bool = False,
    ) -> None:
        super().__init__()
        self.mask_rate = mask_rate
        self.eval_num_partitions = eval_num_partitions
        self.partition_seed = partition_seed
        self.pair_table = pair_table
        if pair_table:
            self.log_selection_matrix = nn.Parameter(
                torch.zeros(NUM_AMINO_ACIDS, NUM_AMINO_ACIDS)
            )

    def pair_log_f(self, aa_parents_idxs: Tensor) -> Tensor:
        """``S[x_j, y]`` per site with the parent column zapped to 0, ``[B, L, 20]``.

        Only for a head built with ``pair_table``. X/^/padding rows read a
        clamped row; the model zeroes every site it does not score.
        """
        if not self.pair_table:
            raise ValueError("pair_log_f needs a head built with pair_table=True")
        return parent_row_log_selection(self.log_selection_matrix, aa_parents_idxs)

    @classmethod
    def from_config(cls, config: "BaseHeadConfig") -> "SiteMaskedSelectionFactorHead":
        """Build from config, keeping the masking fields.

        ``PredictionHead.from_config`` builds shared-SF heads with no
        arguments, which would drop them.
        """
        if not isinstance(config, SiteMaskedSelectionFactorHeadConfig):
            raise TypeError(
                f"{cls.__name__} expects SiteMaskedSelectionFactorHeadConfig, "
                f"got {type(config).__name__}"
            )
        return cls(
            mask_rate=config.mask_rate,
            eval_num_partitions=config.eval_num_partitions,
            partition_seed=config.partition_seed,
            pair_table=config.pair_table,
        )

    def to_config(self) -> "SiteMaskedSelectionFactorHeadConfig":
        """Reconstruct ``SiteMaskedSelectionFactorHeadConfig`` from current state."""
        return SiteMaskedSelectionFactorHeadConfig(
            mask_rate=self.mask_rate,
            eval_num_partitions=self.eval_num_partitions,
            partition_seed=self.partition_seed,
            pair_table=self.pair_table,
        )


# Set config type after class definition to avoid circular import
from dasm2.config import SiteMaskedSelectionFactorHeadConfig  # noqa: E402

SiteMaskedSelectionFactorHead.head_config_type = SiteMaskedSelectionFactorHeadConfig
