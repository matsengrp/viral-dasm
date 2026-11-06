"""Calibration loss for DASM's dense DMS selection-factor output.

Consumes ``DMSFunctionalScoreHead``'s ``[B, L, 20]`` output directly, aligned
cell-for-cell against a ``DMSMatrixDataset``/``DMSMatrixCollator`` batch's
``measured_matrix``/``valid_mask`` (dense convention) -- a different
consumption convention than ``dasm2/losses/dms.py``'s sparse per-mutation
batch keys.

``PearsonCorrelationLoss`` and ``ListMLELoss`` are ported from MAXWELL's own
current loss code, adapted only to accept this dataset's ``valid_mask``
argument directly rather than assuming a fixed-width, always-fully-measured
list.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

#: Shared denominator-guard epsilon for every masked-reduction below, so a
#: zero-valid-cell row/batch divides by a small constant instead of 0.
_EPS = 1e-7


def masked_mean(x: Tensor, mask: Tensor, eps: float = _EPS) -> Tensor:
    """Per-row mean of ``x`` restricted to ``mask == True`` cells.

    Not interchangeable with ``masked_mean`` in the binding-classifier
    experiment script, which mean-pools tokens (``[B, T, D] -> [B, D]``) and
    does not handle NaN.

    Args:
        x: ``[B, N]`` values. A masked-out cell may hold ``NaN`` (this
            dataset's own sentinel for an unmeasured cell -- see
            ``DMSMatrixDataset``); ``x`` is masked by *selection*
            (``torch.where``), not multiplication, since ``NaN * 0 == NaN``
            in IEEE-754 and would otherwise poison the sum even though the
            cell is excluded.
        mask: ``[B, N]`` boolean validity mask.
        eps: Added to the denominator so an all-masked row (no valid cells)
            returns 0 rather than NaN.

    Returns:
        ``[B]`` per-row masked mean.

    Caution for callers that need a gradient through ``x``: this function's
    own ``torch.where`` only protects *this* reduction. If ``x`` was itself
    built by combining a NaN-containing tensor with something that requires
    grad (e.g. ``(calibrated - measured_matrix) ** 2``), the NaN has already
    poisoned that upstream operation's local derivative, and masking *here*
    is one step too late -- ``0 * NaN == NaN`` reappears when this
    function's (correctly) zeroed incoming gradient reaches that op's own
    backward. Sanitize the NaN-containing operand *before* it combines with
    anything requiring grad, not after (see
    ``DMSMatrixCalibrationLoss.forward``'s MSE term for a worked example of
    getting this right, and its docstring/tests for how getting it wrong
    looks: a finite forward loss but every parameter gradient NaN).
    """
    selected = torch.where(mask, x, torch.zeros_like(x))
    return selected.sum(dim=1) / (mask.float().sum(dim=1) + eps)


class PearsonCorrelationLoss(nn.Module):
    """``1 - masked Pearson correlation`` between predictions and targets.

    Ported from MAXWELL's own ``PearsonCorrelationLoss``. Masked cells are
    zeroed out of the centered tensors before ``nn.CosineSimilarity``, so
    they contribute nothing to either the dot product or the norm -- this
    reproduces exactly the correlation a ragged gather-only-valid-cells
    approach would give, without needing one.
    """

    def __init__(self, dim: int = 1, eps: float = _EPS) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.cos = nn.CosineSimilarity(dim=dim, eps=eps)

    def forward(self, predictions: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        predictions = predictions.reshape(predictions.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
        mask = mask.reshape(mask.shape[0], -1)

        pred_mean = masked_mean(predictions, mask, self.eps).reshape(-1, 1)
        targ_mean = masked_mean(targets, mask, self.eps).reshape(-1, 1)

        pred_centered = predictions - pred_mean
        pred_centered[~mask] = 0
        targ_centered = targets - targ_mean
        targ_centered[~mask] = 0

        correlation = self.cos(pred_centered, targ_centered)
        return torch.mean(1 - correlation)


class ListMLELoss(nn.Module):
    """Masked listwise ranking loss under the Plackett-Luce model.

    Ported from MAXWELL's own ``ListMLELoss``. Takes inputs of shape
    ``[B, L, 20]`` and treats the combined ``L * 20`` cells as a single list
    to rank per batch sample. Only the (gradient-free) targets are sorted;
    predictions are reordered to match via ``gather``, so every
    differentiable quantity stays on ``predictions``, never on an index.
    """

    def __init__(self, eps: float = _EPS) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, predictions: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        # 1. Reshape from [B, L, V] to [B, L * V] -- one list per sample.
        predictions = predictions.reshape(predictions.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
        mask = mask.reshape(mask.shape[0], -1)

        # 2. Push masked targets to -inf so they sort last.
        targ_for_sort = targets.masked_fill(~mask, float("-inf"))

        # 3. Sort targets descending and get the reordering indices.
        _, indices = targ_for_sort.sort(descending=True, dim=-1)

        # 4. Reorder predictions and mask to match the target ranking.
        pred_sorted = torch.gather(predictions, dim=-1, index=indices)
        mask_sorted = torch.gather(mask, dim=-1, index=indices)

        # 5. Masked log-sum-exp over suffixes, with the max trick for
        # numerical stability.
        pred_sorted_masked = pred_sorted.masked_fill(~mask_sorted, float("-inf"))
        max_val, _ = pred_sorted_masked.max(dim=-1, keepdim=True)
        max_val = max_val.masked_fill(max_val == float("-inf"), 0)
        exp_s = torch.exp(pred_sorted_masked - max_val)

        # Reverse cumsum -> suffix sums: sum_{j=i}^{K} exp(s_j).
        sum_exp_s = torch.flip(
            torch.cumsum(torch.flip(exp_s, dims=[-1]), dim=-1), dims=[-1]
        )
        log_sum_exp = max_val + torch.log(sum_exp_s + self.eps)

        # 6. Plackett-Luce per-position loss: -s_i + log(sum_{j=i}^{K} exp(s_j)).
        loss = -pred_sorted + log_sum_exp

        # 7. Normalize by valid-cell count per row, mean over the batch.
        return masked_mean(loss, mask_sorted, self.eps).mean()


class DMSMatrixCalibrationLoss(nn.Module):
    """MSE + Pearson + ListMLE calibration loss for a dense DMS matrix batch.

    Consumes ``DMSFunctionalScoreHead``'s ``[B, L, 20]`` output directly,
    scored cell-for-cell against ``batch["measured_matrix"]`` /
    ``batch["valid_mask"]``. Each term is independently weighable (weight
    ``0.0`` disables a term outright): ``w_mse=1.0`` alone reproduces a
    ThermoMPNN-style pure-regression loss; ``w_pearson>0, w_listmle>0``
    (with ``w_mse=0.0``) reproduces MAXWELL's own current loss shape.

    Pearson and ListMLE are both invariant to an affine transform of the
    predictions, so neither alone can calibrate DASM's raw log-selection-
    factor output against real stability units. A learnable ``scale``/
    ``bias`` (two scalars, shared across the model) is applied to the raw
    predictions before all three terms are computed -- the same pattern
    ``DMSFunctionalScoreLoss`` uses for its own learnable ``scale``.

    Returns ``mean x batch_size`` (sum-over-domains): every term is a
    per-domain quantity (one value per batch row) averaged over the batch,
    so "one domain" -- not "one measured cell" -- is this loss's natural
    position unit, and every term shares that same convention. MSE is no
    exception: it is *not* a flat mean over every valid cell in the whole
    batch (that would make its value depend on how cells are distributed
    across domains, e.g. one domain with 50 measured mutants and three with
    1 each would let the first domain dominate) -- it is each domain's own
    masked mean squared error, averaged over the batch exactly like Pearson
    and ListMLE. This is required, not just tidier: ``SelfNormalizingLoss.
    n_loss_labels`` reports ``batch_size`` (below), so the trainer's
    sum-then-divide-by-``N_t`` gradient-accumulation math is only correct
    if every term this loss returns truly decomposes as a sum over exactly
    that many per-domain contributions -- a flat per-cell MSE pool does
    not (independently verified: with unequal per-domain valid-cell counts,
    a flat-pooled MSE term made accumulated-vs-single-batch training
    diverge by ~4e-2, far above float32 noise; the per-domain-averaged
    version does not).

    Implements :class:`~dasm2.losses.protocols.SelfNormalizingLoss`
    (``n_loss_labels`` returns ``batch_size``) so the trainer divides this
    sum back down by exactly that unit, recovering the mean while
    normalizing gradient-accumulation windows the way every other
    registered loss does -- *regardless* of what other keys the batch
    happens to carry. This loss does not rely on ``Trainer.
    _get_n_loss_positions``'s ``loss_mask``/``mask``-key-absence fallback
    (unlike a first version of this class, and unlike
    ``ContrastiveBindingLoss``'s own convention): ``CoreModel.forward``
    requires a ``"mask"`` key for *every* head's batch (attention masking),
    so any head reached through the standard model path already has one --
    an absence-based fallback would silently flip to per-cell counting the
    moment this loss is wired to a real model, exactly the double-
    normalization bug the sum-vs-mean convention exists to prevent. Verified
    empirically end to end (``tests/training/test_dms_matrix_calibration_accumulation.py``)
    against the real ``Trainer``, both single-task and jointly with a
    second task at equal and unequal per-task batch sizes.
    """

    def __init__(
        self,
        w_mse: float = 1.0,
        w_pearson: float = 0.0,
        w_listmle: float = 0.0,
        scale_init: float = 1.0,
        bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.w_mse = w_mse
        self.w_pearson = w_pearson
        self.w_listmle = w_listmle
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.bias = nn.Parameter(torch.tensor(float(bias_init)))
        self.pearson_loss = PearsonCorrelationLoss()
        self.listmle_loss = ListMLELoss()

    def n_loss_labels(self, batch: Dict[str, Tensor]) -> int:
        """SelfNormalizingLoss: this loss's label unit is "one domain".

        Read during the trainer's mask-only counting pass, before
        ``move_batch_to_device`` -- ``measured_matrix.shape[0]`` needs no
        device and no gradient, matching the protocol's contract.
        """
        return int(batch["measured_matrix"].shape[0])

    def forward(self, predictions: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        for key in ("measured_matrix", "valid_mask"):
            if key not in batch:
                raise ValueError(
                    f"DMSMatrixCalibrationLoss missing required batch key: {key!r}. "
                    f"Got keys: {list(batch.keys())}."
                )
        measured_matrix = batch["measured_matrix"]
        valid_mask = batch["valid_mask"]

        calibrated = self.scale * predictions + self.bias

        total = torch.zeros((), device=predictions.device, dtype=predictions.dtype)
        if self.w_mse != 0.0:
            # Sanitize measured_matrix's NaN (this dataset's own sentinel for
            # "unmeasured" -- see DMSMatrixDataset) *before* it combines with
            # `calibrated` (which requires grad), not after. Masking the
            # *output* of (calibrated - measured_matrix) ** 2 (via
            # multiplication or torch.where) still leaves a NaN gradient: the
            # subtraction's own backward computes
            # d(squared_error)/d(calibrated) = 2*(calibrated-measured_matrix)
            # elementwise for the *whole* tensor (since calibrated requires
            # grad), which is NaN wherever measured_matrix is NaN, and
            # `0 * NaN == NaN` poisons every parameter's gradient the moment
            # that 0-valued incoming gradient (from the mask) meets it one
            # step further back in the chain rule -- even though the
            # *forward* value was already correctly masked. Replacing NaN in
            # measured_matrix itself, before the subtraction, keeps NaN out
            # of the graph entirely, so a plain multiplicative mask is safe
            # for both forward and backward from here on.
            safe_measured_matrix = torch.where(
                valid_mask, measured_matrix, torch.zeros_like(measured_matrix)
            )
            squared_error = (calibrated - safe_measured_matrix) ** 2
            # Per-domain masked mean (each domain's own average over its own
            # valid cells), then averaged over the batch -- exactly Pearson
            # and ListMLE's convention, via the same masked_mean helper. Not
            # a flat mean over every valid cell in the batch: that would
            # weight each domain by its own measured-cell count rather than
            # equally, and would not decompose as a sum over `batch_size`
            # domains the way SelfNormalizingLoss.n_loss_labels assumes (see
            # class docstring).
            squared_error_flat = squared_error.reshape(squared_error.shape[0], -1)
            valid_mask_flat = valid_mask.reshape(valid_mask.shape[0], -1)
            mse = masked_mean(squared_error_flat, valid_mask_flat, _EPS).mean()
            total = total + self.w_mse * mse
        if self.w_pearson != 0.0:
            total = total + self.w_pearson * self.pearson_loss(
                calibrated, measured_matrix, valid_mask
            )
        if self.w_listmle != 0.0:
            total = total + self.w_listmle * self.listmle_loss(
                calibrated, measured_matrix, valid_mask
            )
        return total * predictions.shape[0]
