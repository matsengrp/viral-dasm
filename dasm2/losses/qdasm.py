"""QDASM Q-matrix CTMC loss — first-class ``nn.Module`` Loss class.

Pre-#621 the Q-matrix machinery (q_variant, hit scales, thrifty 13-mer table,
adaptive flags, quadratic_ste) lived as plain attributes on
``QDASMEvolHead``. After #621 those fields move here so the head is
forward-only and the loss owns its own configuration.

The thrifty 13-mer lookup table is reloaded from ``thrifty_13mer_table_path``
on construction (matching the head's pre-refactor behavior — the buffer is
non-persistent so it doesn't appear in state_dict).
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

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
from dasm2.molevol.q_matrix import QVariant
from dasm2.molevol.qdasm import compute_qdasm_log_codon_probs
from dasm2.utils.tensor import csps_to_4d
from dasm2.utils.validation import validate_batch_keys


def _resolve_neutral_codon_q(
    q_variant: QVariant,
    neutral_codon_q: Optional[Tensor],
    neutral_codon_q_path: Optional[str],
) -> Optional[Tensor]:
    """Resolve the ``[61, 61]`` SHM-1 codon generator for ``shm1_codon``.

    Mirrors ``resolve_thrifty_13mer_table``: only the ``shm1_codon`` variant
    needs it. An in-memory tensor (test-only) wins; otherwise it is loaded
    from ``neutral_codon_q_path`` (``None`` → the packaged SHM-1 matrix).
    Returns ``None`` for every other variant.
    """
    if q_variant != "shm1_codon":
        return None
    if neutral_codon_q is not None:
        return neutral_codon_q
    from dasm2.neutral.shm_codon import load_shm1_codon_q

    return load_shm1_codon_q(neutral_codon_q_path)


class QDASMLoss(nn.Module):
    """QDASM CTMC codon NLL with configurable Q-matrix construction.

    Owns all the loss-side configuration that previously lived on the head.
    The selected global k-mer lookup is registered as one non-persistent
    ``kmer_table`` buffer and reloaded from its production source on construction.
    """

    q_variant: QVariant
    use_cubic: bool
    single_hit_scale: float
    double_hit_scale: float
    triple_hit_scale: float
    adaptive: bool
    tq_threshold: float
    long_path: Literal["matrix_exp", "ss_quintic"]
    adaptive_mode: Literal["per_chunk", "per_sequence"]
    quadratic_ste: bool
    matexp_method: Literal["adaptive", "uniformization"]
    n_terms: int
    normalization: Literal["parent_absorbing", "proportional"]
    kmer_table: Optional[Tensor]
    thrifty_13mer_model_name: Optional[str]
    neutral_codon_q: Optional[Tensor]
    PREDICT_REQUIRED_KEYS: tuple[str, ...]

    def __init__(
        self,
        use_cubic: bool = False,
        single_hit_scale: float = DEFAULT_SINGLE_HIT_SCALE,
        double_hit_scale: float = DEFAULT_DOUBLE_HIT_SCALE,
        triple_hit_scale: float = DEFAULT_TRIPLE_HIT_SCALE,
        adaptive: bool = False,
        tq_threshold: float = 1.0,
        long_path: Literal["matrix_exp", "ss_quintic"] = "ss_quintic",
        adaptive_mode: Literal["per_chunk", "per_sequence"] = "per_sequence",
        quadratic_ste: bool = False,
        matexp_method: Literal["adaptive", "uniformization"] = "adaptive",
        n_terms: int = 4,
        normalization: Literal["parent_absorbing", "proportional"] = "parent_absorbing",
        q_variant: QVariant = "parent_row_degenerate",
        thrifty_13mer_table_path: Optional[str] = None,
        thrifty_13mer_model_name: Optional[str] = None,
        five_mer_model_name: Optional[str] = None,
        kmer_table: Optional[Tensor] = None,
        neutral_codon_q_path: Optional[str] = None,
        neutral_codon_q: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        if quadratic_ste and (adaptive or use_cubic):
            raise ValueError(
                "quadratic_ste is mutually exclusive with adaptive and use_cubic; "
                f"got quadratic_ste={quadratic_ste}, adaptive={adaptive}, "
                f"use_cubic={use_cubic}."
            )
        if adaptive and use_cubic:
            raise ValueError(
                "adaptive and use_cubic are mutually exclusive; "
                f"got adaptive={adaptive}, use_cubic={use_cubic}."
            )
        if matexp_method == "uniformization" and (
            quadratic_ste or use_cubic or adaptive
        ):
            raise ValueError(
                "matexp_method='uniformization' takes precedence over "
                "quadratic_ste/use_cubic/adaptive and ignores them; got "
                f"quadratic_ste={quadratic_ste}, use_cubic={use_cubic}, "
                f"adaptive={adaptive}. Leave those at their defaults (False) "
                "to avoid a misleading config."
            )
        self.use_cubic = use_cubic
        self.single_hit_scale = single_hit_scale
        self.double_hit_scale = double_hit_scale
        self.triple_hit_scale = triple_hit_scale
        self.adaptive = adaptive
        self.tq_threshold = tq_threshold
        self.long_path = long_path
        self.adaptive_mode = adaptive_mode
        self.quadratic_ste = quadratic_ste
        self.matexp_method = matexp_method
        self.n_terms = n_terms
        self.normalization = normalization
        self.q_variant = q_variant
        self.PREDICT_REQUIRED_KEYS = q_family_predict_required_keys(q_variant)
        self.thrifty_13mer_table_path = thrifty_13mer_table_path
        self.thrifty_13mer_model_name = thrifty_13mer_model_name

        self.five_mer_model_name = five_mer_model_name
        table = resolve_global_kmer_table(
            q_variant,
            kmer_table,
            thrifty_13mer_table_path=thrifty_13mer_table_path,
            expected_model_name=thrifty_13mer_model_name,
            kmer_model_name=five_mer_model_name,
        )
        if table is not None:
            self.register_buffer("kmer_table", table, persistent=False)
        else:
            self.kmer_table = None

        # SHM-1 codon generator for the ``shm1_codon`` variant. Like the
        # 13-mer table it is a non-persistent buffer, reloaded on construction.
        if neutral_codon_q is not None and neutral_codon_q_path is not None:
            raise ValueError(
                "Pass either neutral_codon_q (in-memory, test-only) or "
                "neutral_codon_q_path (config-loaded, production), not both."
            )
        self.neutral_codon_q_path = neutral_codon_q_path
        codon_q = _resolve_neutral_codon_q(
            q_variant, neutral_codon_q, neutral_codon_q_path
        )
        if codon_q is not None:
            self.register_buffer("neutral_codon_q", codon_q, persistent=False)
        else:
            self.neutral_codon_q = None

    def log_codon_probs(self, predictions: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Per-site log P(child codon | parent codon, t), ``[B, L, 64]``.

        The :class:`~dasm2.losses.protocols.CodonSamplingLoss` contract: the
        rows are parent rows of ``P = exp(tQ)`` embedded 61 → 64 with zeros
        at the stop columns, so they are per-site normalized and carry zero
        stop mass by construction. Q is built per codon site, so sites are
        conditionally independent given the parent.
        """
        validate_batch_keys(
            batch,
            list(self.PREDICT_REQUIRED_KEYS),
            "QDASMLoss.log_codon_probs()",
            EVOLUTIONARY_DATASET_HINT,
        )
        codon_csps = csps_to_4d(batch["codon_csps"])
        (
            codon_parent_nt_window,
            codon_parent_boundary_mask,
            codon_boundary_features,
            codon_rate_adjustments,
        ) = q_family_global_kmer_inputs(
            batch, self.q_variant, "QDASMLoss.log_codon_probs()"
        )
        return compute_qdasm_log_codon_probs(
            batch["codon_parents_idxs"],
            batch["codon_rates"],
            codon_csps,
            predictions,
            batch["branch_length"],
            use_cubic=self.use_cubic,
            single_hit_scale=self.single_hit_scale,
            double_hit_scale=self.double_hit_scale,
            triple_hit_scale=self.triple_hit_scale,
            adaptive=self.adaptive,
            tq_threshold=self.tq_threshold,
            long_path=self.long_path,
            adaptive_mode=self.adaptive_mode,
            quadratic_ste=self.quadratic_ste,
            matexp_method=self.matexp_method,
            n_terms=self.n_terms,
            normalization=self.normalization,
            q_variant=self.q_variant,
            codon_parent_nt_window=codon_parent_nt_window,
            codon_parent_boundary_mask=codon_parent_boundary_mask,
            codon_boundary_features=codon_boundary_features,
            codon_rate_adjustments=codon_rate_adjustments,
            kmer_table=self.kmer_table,
            neutral_codon_q=self.neutral_codon_q,
        )

    def forward(self, predictions: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        validate_batch_keys(
            batch,
            list(self.PREDICT_REQUIRED_KEYS + EVOLUTIONARY_LOSS_ONLY_KEYS),
            "QDASMLoss.forward()",
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
