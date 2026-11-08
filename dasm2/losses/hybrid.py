"""Hybrid CTMC-neutral / DASM-selection loss — first-class ``nn.Module`` Loss class.

The Hybrid model computes neutral transition probabilities via the exact
CTMC matrix exponential (with unit selection factors), then applies learned
selection factors post-hoc. Combines QDASM's exact neutral model with
DASM's post-hoc selection application.

Pre-#621 the Q-matrix machinery (q_variant, hit scales, thrifty 13-mer
table, cubic flag) lived as plain attributes on ``HybridEvolHead``. After
#621 those fields move here.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from dasm2.losses._kmer_table_loader import resolve_global_kmer_table
from dasm2.losses.loss_utils import (
    EVOLUTIONARY_DATASET_HINT,
    EVOLUTIONARY_LOSS_ONLY_KEYS,
    q_family_global_kmer_inputs,
    q_family_predict_required_keys,
)
from dasm2.losses.q_scale_defaults import (
    DEFAULT_DOUBLE_HIT_SCALE,
    DEFAULT_SINGLE_HIT_SCALE,
    DEFAULT_TRIPLE_HIT_SCALE,
)
from dasm2.molevol.hybrid import compute_hybrid_log_codon_probs
from dasm2.molevol.q_matrix import QVariant
from dasm2.utils.tensor import csps_to_4d
from dasm2.utils.validation import validate_batch_keys


class HybridLoss(nn.Module):
    """Hybrid CTMC-neutral + post-hoc selection codon NLL.

    Owns the loss-side Q-matrix configuration that previously lived on the
    head. Thrifty 13-mer table is reloaded from path on construction
    (non-persistent buffer).
    """

    q_variant: QVariant
    use_cubic: bool
    single_hit_scale: float
    double_hit_scale: float
    triple_hit_scale: float
    kmer_table: Optional[Tensor]
    thrifty_13mer_model_name: Optional[str]
    PREDICT_REQUIRED_KEYS: tuple[str, ...]

    def __init__(
        self,
        use_cubic: bool = False,
        single_hit_scale: float = DEFAULT_SINGLE_HIT_SCALE,
        double_hit_scale: float = DEFAULT_DOUBLE_HIT_SCALE,
        triple_hit_scale: float = DEFAULT_TRIPLE_HIT_SCALE,
        q_variant: QVariant = "parent_row_degenerate",
        thrifty_13mer_table_path: Optional[str] = None,
        thrifty_13mer_model_name: Optional[str] = None,
        kmer_table: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.use_cubic = use_cubic
        self.single_hit_scale = single_hit_scale
        self.double_hit_scale = double_hit_scale
        self.triple_hit_scale = triple_hit_scale
        self.q_variant = q_variant
        self.PREDICT_REQUIRED_KEYS = q_family_predict_required_keys(q_variant)
        self.thrifty_13mer_table_path = thrifty_13mer_table_path
        self.thrifty_13mer_model_name = thrifty_13mer_model_name

        table = resolve_global_kmer_table(
            q_variant,
            kmer_table,
            thrifty_13mer_table_path=thrifty_13mer_table_path,
            expected_model_name=thrifty_13mer_model_name,
        )
        if table is not None:
            self.register_buffer("kmer_table", table, persistent=False)
        else:
            self.kmer_table = None

    def log_codon_probs(self, predictions: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Per-site log P(child codon | parent codon, t), ``[B, L, 64]``.

        The :class:`~dasm2.losses.protocols.CodonSamplingLoss` contract: the
        exact-CTMC neutral rows are per-site normalized and stop-free (61 → 64
        with zeros at the stop columns), and the post-hoc selection factors
        are reapplied and renormalized per site.
        """
        validate_batch_keys(
            batch,
            list(self.PREDICT_REQUIRED_KEYS),
            "HybridLoss.log_codon_probs()",
            EVOLUTIONARY_DATASET_HINT,
        )
        codon_csps = csps_to_4d(batch["codon_csps"])
        (
            codon_parent_nt_window,
            codon_parent_boundary_mask,
            codon_boundary_features,
            codon_rate_adjustments,
        ) = q_family_global_kmer_inputs(
            batch, self.q_variant, "HybridLoss.log_codon_probs()"
        )
        return compute_hybrid_log_codon_probs(
            batch["codon_parents_idxs"],
            batch["codon_rates"],
            codon_csps,
            predictions,
            batch["branch_length"],
            use_cubic=self.use_cubic,
            single_hit_scale=self.single_hit_scale,
            double_hit_scale=self.double_hit_scale,
            triple_hit_scale=self.triple_hit_scale,
            q_variant=self.q_variant,
            codon_parent_nt_window=codon_parent_nt_window,
            codon_parent_boundary_mask=codon_parent_boundary_mask,
            codon_boundary_features=codon_boundary_features,
            codon_rate_adjustments=codon_rate_adjustments,
            kmer_table=self.kmer_table,
        )

    def forward(self, predictions: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        validate_batch_keys(
            batch,
            list(self.PREDICT_REQUIRED_KEYS + EVOLUTIONARY_LOSS_ONLY_KEYS),
            "HybridLoss.forward()",
            EVOLUTIONARY_DATASET_HINT,
        )
        log_codon_probs = self.log_codon_probs(predictions, batch)
        mask = batch["loss_mask"]
        log_codon_probs_masked = log_codon_probs[mask]
        codon_children_masked = batch["codon_children_idxs"][mask]

        assert torch.isnan(log_codon_probs_masked).sum() == 0, (
            "NaN values in log codon probabilities after masking"
        )
        return torch.nn.functional.nll_loss(
            log_codon_probs_masked, codon_children_masked, reduction="sum"
        )

    def predict_codons(self, predictions: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Top-1 child-codon prediction per position (issue #620 contract)."""
        return self.log_codon_probs(predictions, batch).argmax(dim=-1)
