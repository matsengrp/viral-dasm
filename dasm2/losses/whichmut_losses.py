"""The whichmut loss — a first-class ``nn.Module`` Loss class.

Wraps the pure-function math kernels in ``dasm2.losses.whichmut_loss``. The
AA-level AAO variant went with ``AAHWhichmutHead``, the only head that paired
with it (#4). The former plain-Python wrapper classes are ``nn.Module``
subclasses so loss-owned parameters participate cleanly in the
optimizer's parameter graph via ``model.task_losses``.
"""

from __future__ import annotations

from typing import Dict

from torch import Tensor, nn

from dasm2.losses.whichmut_loss import (
    compute_whichmut_loss_batch,
    predict_whichmut_child_codons,
)


def _require(batch: Dict[str, Tensor], keys: tuple[str, ...], context: str) -> None:
    missing = [k for k in keys if k not in batch]
    if missing:
        raise ValueError(
            f"{context} missing required batch keys: {missing}. "
            f"Got keys: {list(batch.keys())}."
        )


class WhichmutLoss(nn.Module):
    """Whichmut codon-level loss (gauge-anchoring via synonymous mutations)."""

    _REQUIRED = (
        "codon_parents_idxs",
        "codon_children_idxs",
        "codon_neutral_rates",
        "loss_mask",
    )

    def forward(self, predictions: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        _require(batch, self._REQUIRED, "WhichmutLoss")
        return compute_whichmut_loss_batch(
            selection_factors=predictions,
            neutral_rates=batch["codon_neutral_rates"],
            codon_parents_idxs=batch["codon_parents_idxs"],
            codon_children_idxs=batch["codon_children_idxs"],
            mask=batch["loss_mask"],
        )

    def predict_child_codons(
        self, predictions: Tensor, batch: Dict[str, Tensor]
    ) -> Tensor:
        """Top-1 reachable child-codon prediction per position, ``(B, L)``.

        Implementing this opts whichmut into the conditional codon-accuracy
        metric group (``whichmut_codon`` provider): the metric
        provider resolves its extractor by looking up ``predict_child_codons``
        on the task's loss. The name is deliberately distinct from DASM's
        ``predict_codons`` — whichmut predicts *which mutation* conditioned on
        a mutation occurring (argmax over reachable single-NT children), not a
        top-1 over the full codon distribution, so the two are mutually
        exclusive gates that must not co-activate on one task.
        """
        _require(
            batch,
            ("codon_neutral_rates", "codon_parents_idxs"),
            "WhichmutLoss.predict_child_codons",
        )
        return predict_whichmut_child_codons(
            selection_factors=predictions,
            neutral_rates=batch["codon_neutral_rates"],
            codon_parents_idxs=batch["codon_parents_idxs"],
        )
