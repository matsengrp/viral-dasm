"""DASM mutation-selection loss — first-class ``nn.Module`` Loss class.

Absorbs the pre-#621 ``DASMLossMixin`` (``dasm2/heads/mixins.py``) and the
``focal_gamma`` branch from the original DASM head's ``compute_loss``. The
pure-function codon-prob math (``compute_log_adjusted_codon_probs``) is
unchanged.

Class-weighted CE (issue #678) is implemented inline: under
``weight_mode='static' | 'inverse_freq'`` the per-position NLL is
multiplied by a binary {conserved, mutated} weight derived so that
``Σ_i w_i == n_valid`` — the trainer's per-loss-mask-count denominator
is preserved and the effective LR is stable across runs.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dasm2.codon_table import get_codon_to_aa_lookup_tensor
from dasm2.losses.loss_utils import (
    EVOLUTIONARY_DATASET_HINT,
    EVOLUTIONARY_LOSS_ONLY_KEYS,
    focal_nll_loss,
)
from dasm2.losses.types import WeightMode
from dasm2.molevol.mutsel import (
    DASMNormalization,
    compute_log_adjusted_codon_probs,
)
from dasm2.utils.validation import validate_batch_keys

# Batch keys read by ``log_codon_probs`` — the CodonSamplingLoss declaration
# (see ``dasm2/losses/protocols.py``). DASM's kernel is static, so unlike
# QDASM's this needs no per-instance widening.
_DASM_PREDICT_REQUIRED = (
    "codon_parents_idxs",
    "log_neutral_codon_probs",
)
_DASM_LOSS_REQUIRED = _DASM_PREDICT_REQUIRED + EVOLUTIONARY_LOSS_ONLY_KEYS


def _class_budget_fractions(
    weight_mode: WeightMode,
    mut_budget: float,
    alpha: float,
    p_cons: Tensor,
    p_mut: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the per-class budget pair ``(w_cons, w_mut)`` summing to 1.

    These are fractions of the total loss budget, *not* per-position
    weights — the caller divides by ``p_b`` to get per-position weights.
    ``p_cons`` and ``p_mut`` are 0-d tensors carrying the per-batch class
    frequencies; keeping the result as tensors lets the per-position
    weighting tensor stay on-device without a host sync.
    """
    if weight_mode == "static":
        w_mut = p_cons.new_tensor(mut_budget)
        w_cons = p_cons.new_tensor(1.0 - mut_budget)
        return w_cons, w_mut
    # inverse_freq: w_b ∝ p_b^(1-alpha), sum-normalized.
    exponent = 1.0 - alpha
    raw_cons = p_cons.pow(exponent)
    raw_mut = p_mut.pow(exponent)
    total = raw_cons + raw_mut
    return raw_cons / total, raw_mut / total


class DASMLoss(nn.Module):
    """Codon-level DASM cross-entropy with optional focal and class weighting.

    With defaults (``focal_gamma == 0`` and ``weight_mode == 'uniform'``),
    reduces to ``F.nll_loss(..., reduction='sum')`` — bit-identical to the
    pre-#678 implementation. When ``focal_gamma > 0`` the masked NLL is
    replaced by ``focal_nll_loss`` (Lin et al., 2017). When
    ``weight_mode != 'uniform'`` the masked NLL is multiplied by a
    per-position binary {conserved, mutated} class weight whose batch sum
    equals the number of valid positions, so the trainer's
    per-mask-count denominator still recovers a true weighted mean.

    Combining ``focal_gamma > 0`` with non-uniform weighting is rejected
    at config-parse time (see ``DASMLossConfig._focal_and_weighting_are_orthogonal_for_now``).
    """

    PREDICT_REQUIRED_KEYS: tuple[str, ...] = _DASM_PREDICT_REQUIRED

    def __init__(
        self,
        focal_gamma: float = 0.0,
        weight_mode: WeightMode = "uniform",
        mut_budget: float = 0.5,
        alpha: float = 0.5,
        syn_loss_mask: bool = False,
        normalization: DASMNormalization = "parent_absorbing",
    ) -> None:
        super().__init__()
        if focal_gamma < 0.0:
            raise ValueError(f"focal_gamma must be >= 0, got {focal_gamma}")
        if not 0.0 < mut_budget < 1.0:
            raise ValueError(
                f"mut_budget must be in the open interval (0, 1), got {mut_budget}"
            )
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if focal_gamma > 0.0 and weight_mode != "uniform":
            # Mirrors the config-level validator so the loss is safe even
            # when constructed directly (tests, scripts).
            raise ValueError(
                "DASMLoss: focal_gamma > 0 combined with "
                f"weight_mode={weight_mode!r} is not supported (issue #678)."
            )
        self.focal_gamma = focal_gamma
        self.weight_mode: WeightMode = weight_mode
        self.mut_budget = mut_budget
        self.alpha = alpha
        self.syn_loss_mask = syn_loss_mask
        # Which endpoint kernel ``log_codon_probs`` runs (issue #1238).
        # Validated inside ``compute_log_adjusted_codon_probs`` rather than
        # here, so a direct construction and a config-built one fail the
        # same way with the same message.
        self.normalization: DASMNormalization = normalization
        # Registered as a non-persistent buffer so DDP / ``.to(device)`` move
        # the codon-to-AA lookup in lock-step with the module's parameters
        # and the cache never leaks into ``state_dict()``. Lazily filled on
        # the first forward call so construction doesn't need a device.
        # The explicit annotation tells mypy that buffer reads return a
        # Tensor (the default ``nn.Module.__getattr__`` stub is ``Any``).
        self._codon_to_aa: Tensor
        self.register_buffer("_codon_to_aa", torch.empty(0), persistent=False)

    def _codon_to_aa_for(self, device: torch.device) -> Tensor:
        cached: Tensor = self._codon_to_aa
        if cached.numel() == 0 or cached.device != device:
            cached = get_codon_to_aa_lookup_tensor(device)
            self._codon_to_aa = cached
        return cached

    def _apply_syn_mask(self, mask: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Return ``mask`` with synonymous-mutation positions zeroed out.

        A position is synonymous-mutated when the parent and child codons
        differ but encode the same AA. Removing these from the loss mask
        leaves the per-position NLL untouched at those sites; per #678 §2
        their gradient is exactly zero already, so the only observable
        change is the mask count (and downstream metric denominators).
        """
        parents = batch["codon_parents_idxs"]
        children = batch["codon_children_idxs"]
        codon_to_aa = self._codon_to_aa_for(mask.device)
        is_syn_mut = (codon_to_aa[parents] == codon_to_aa[children]) & (
            parents != children
        )
        return mask & ~is_syn_mut

    def forward(
        self,
        predictions: Tensor,
        batch: Dict[str, Tensor],
    ) -> Tensor:
        validate_batch_keys(
            batch,
            list(_DASM_LOSS_REQUIRED),
            "DASMLoss.forward()",
            EVOLUTIONARY_DATASET_HINT,
        )
        log_codon_probs = self.log_codon_probs(predictions, batch)
        mask = batch["loss_mask"]
        if self.syn_loss_mask:
            mask = self._apply_syn_mask(mask, batch)
        log_codon_probs_masked = log_codon_probs[mask]
        codon_children_masked = batch["codon_children_idxs"][mask]

        assert torch.isnan(log_codon_probs_masked).sum() == 0, (
            "NaN values in log codon probabilities after masking"
        )

        if self.focal_gamma > 0.0:
            # Focal branch never composes with class weighting (the
            # constructor-level guard makes that combination unreachable).
            return focal_nll_loss(
                log_codon_probs_masked, codon_children_masked, gamma=self.focal_gamma
            )

        if self.weight_mode == "uniform":
            return F.nll_loss(
                log_codon_probs_masked, codon_children_masked, reduction="sum"
            )

        # Class-weighted CE. The trainer does not pre-filter empty-mask
        # batches, so guarding ``n_valid == 0`` is necessary, not defensive.
        n_valid = codon_children_masked.numel()
        if n_valid == 0:
            return log_codon_probs_masked.new_zeros(())

        codon_parents_masked = batch["codon_parents_idxs"][mask]
        is_mut = codon_children_masked != codon_parents_masked
        n_mut = is_mut.sum()
        n_cons = (~is_mut).sum()
        # When the batch happens to be entirely one class, neither inverse
        # freq nor static can derive a finite per-pos weight for the empty
        # class; the budget would divide by zero. Fall back to uniform
        # weights for that batch — the per-class-weight invariant
        # ``Σ w_i = N`` still holds.
        if n_mut == 0 or n_cons == 0:
            return F.nll_loss(
                log_codon_probs_masked, codon_children_masked, reduction="sum"
            )

        n_valid_f = log_codon_probs_masked.new_tensor(float(n_valid))
        p_mut = n_mut.to(log_codon_probs_masked.dtype) / n_valid_f
        p_cons = n_cons.to(log_codon_probs_masked.dtype) / n_valid_f
        w_cons_budget, w_mut_budget = _class_budget_fractions(
            self.weight_mode, self.mut_budget, self.alpha, p_cons, p_mut
        )
        w_per_cons = w_cons_budget / p_cons
        w_per_mut = w_mut_budget / p_mut
        w = torch.where(is_mut, w_per_mut, w_per_cons)

        nll = F.nll_loss(
            log_codon_probs_masked, codon_children_masked, reduction="none"
        )
        return (w * nll).sum()

    def log_codon_probs(self, predictions: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Per-site log P(child codon | parent codon, t), ``[B, L, 64]``.

        The :class:`~dasm2.losses.protocols.CodonSamplingLoss` contract:
        ``compute_log_adjusted_codon_probs`` renormalizes each site over its
        own 64 codons and zeroes stop-codon mass, and DASM's Poisson-product
        neutral model is per-site by construction.

        Which of the two normalizations runs is ``self.normalization``
        (issue #1238). Under the default ``parent_absorbing`` the
        renormalization is only *conditional* — a row whose non-parent sense
        mass exceeds 1 over-sums, with the parent floored (issue #1160).
        ``sense_conditional`` satisfies the contract unconditionally.
        """
        validate_batch_keys(
            batch,
            list(self.PREDICT_REQUIRED_KEYS),
            "DASMLoss.log_codon_probs()",
            EVOLUTIONARY_DATASET_HINT,
        )
        return compute_log_adjusted_codon_probs(
            batch["codon_parents_idxs"],
            batch["log_neutral_codon_probs"],
            predictions,
            normalization=self.normalization,
        )

    def predict_codons(self, predictions: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Top-1 child-codon prediction per position (issue #620 contract).

        Implementing this method opts the loss into validation-time codon
        metrics: the codon
        :class:`~dasm2.training.metric_providers.MetricProvider` resolves
        its extractor by looking up ``predict_codons`` on the task's loss
        (issue #738, ``dasm2/training/codon_metrics.py``).
        """
        return self.log_codon_probs(predictions, batch).argmax(dim=-1)

    def selection_factors(
        self, predictions: Tensor, batch: Dict[str, Tensor]
    ) -> Tensor:
        """Return the per-position selection-factor tensor ``(B, L, 20)``.

        For DASM the ``predictions`` argument *is* the post-parent-AA-zap
        SF tensor (``dasm2/heads/_selection_factor_forward.py``), so this
        is an identity hook. The SF
        :class:`~dasm2.training.metric_providers.MetricProvider` is gated
        on the head's ``SELECTION_FACTOR`` capability (issue #738), not on
        this method: when a loss does not implement it, an identity
        head-level fallback supplies the same tensor. DASM keeps the
        explicit override to document the contract. ``batch`` is unused
        today but is part of the protocol signature so future losses that
        derive SFs from a richer input can customize extraction.
        """
        del batch
        return predictions
