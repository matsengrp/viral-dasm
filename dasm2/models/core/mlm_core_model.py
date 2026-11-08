"""MLM-specific CoreModel: masked-marginal scoring adapter (issue #923 Phase 2).

``MLMHead`` produces raw, ungauged reconstruction logits (issue #884/#851) —
there is no parent residue and no ``zap_predictions_along_diagonal``-style
anchoring, so the base ``CoreModel.selection_factors_of_aa_str`` (a single
unmasked forward pass) is semantically wrong here: with the true residue
visible at every position, the model trivially reconstructs it and the
resulting logits carry no information about which residue *should* be there
absent that position's own identity. Masked-marginal scoring instead masks
one real position at a time, re-running the encoder on a batch of masked
copies, and reads the model's belief about the masked position from that
row's output — the same principle the DASM paper uses (as pseudo-perplexity)
to score ESM2/AbLang2 in its own FLAb benchmarks.

Reuses the base ``CoreModel.forward()``/``_build_encoder_input`` unchanged
(training already validates this works end to end for MLM — see
``tests/integration/test_mlm_train_step.py``): only ``selection_factors_of_aa_str``
is overridden here. No ``layout``/segment-embedding support, matching
``MaskedTransformCollator``'s own batches, which never populate a "layout"
key either — MLM training has no segment-embedding use case today.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from dasm2.config import HeadConfig
from dasm2.data.masking import DEFAULT_MASK_BATCH_SIZE, masked_marginal_scan
from dasm2.heads.types import HeadType
from dasm2.models.core.base import CoreModel
from dasm2.models.core.sf_projection import SelectionFactorProjection
from dasm2.sequence.mlm_constants import MLM_MASK_IDX, MLM_SEPARATOR_IDX
from dasm2.tokenizers import MLM_TOKENIZER


def _masked_marginal_log_odds(
    logits: torch.Tensor,
    maskable_positions: torch.Tensor,
    ref_residues: torch.Tensor,
) -> torch.Tensor:
    """Convert per-masked-position logits into log-odds relative to the
    reference residue at each position.

    Called once per chunk of the masked-marginal scan, with that chunk's
    slice of the maskable positions and reference residues, so ``n`` below is
    the chunk's number of masked copies, not the sequence's number of sites.
    ``logits`` is ``[n, seq_len, num_amino_acids]`` — row ``i`` is the
    forward pass with position ``maskable_positions[i]`` replaced by
    ``<mask>``; ``maskable_positions`` and ``ref_residues`` are ``[n]``.
    For each row, gathers that row's log-probabilities at its
    own masked position, then subtracts the reference residue's own
    log-probability so the reference column is exactly 0 (``f[ref] = 1``
    after ``exp``) — the same convention ``zap_predictions_along_diagonal``
    already gives gauge-anchored SF heads (``log_f[parent] = 0``), applied
    here to raw MLM reconstruction logits instead of a gauge-anchored head.

    Returns ``[n, num_amino_acids]`` log-odds.
    """
    num_positions = logits.shape[0]
    row_idx = torch.arange(num_positions, device=logits.device)
    per_position_logits = logits[row_idx, maskable_positions, :]
    log_probs = torch.log_softmax(per_position_logits, dim=-1)
    ref_log_probs = log_probs[row_idx, ref_residues].unsqueeze(-1)
    return log_probs - ref_log_probs


class MLMCoreModel(CoreModel):
    """CoreModel specialized for masked-marginal scoring of ``MLMHead`` output.

    Satisfies the ``SelectionFactorModel`` protocol
    (``dasm2/data/functional_formats.py``) so a trained MLM checkpoint plugs
    into the existing FLAb functional-evaluation harness
    (``dasm2/analysis/functional_eval.py``) unchanged.
    """

    def __init__(
        self,
        encoder: nn.Module,
        prediction_heads: Optional[List[HeadConfig]] = None,
        sf_projection: Optional[SelectionFactorProjection] = None,
        mask_batch_size: int = DEFAULT_MASK_BATCH_SIZE,
    ) -> None:
        """Build the model; see ``CoreModel.__init__`` for the shared args.

        Args:
            mask_batch_size: Number of masked copies per encoder forward pass
                in the masked-marginal scan (see
                ``dasm2.data.masking.DEFAULT_MASK_BATCH_SIZE``). Bounds memory
                for long sequences; does not change the scores.

        Raises:
            ValueError: If ``mask_batch_size`` is less than 1.
        """
        if mask_batch_size < 1:
            raise ValueError(
                f"mask_batch_size must be at least 1, got {mask_batch_size}"
            )
        super().__init__(encoder, prediction_heads, sf_projection)
        self.mask_batch_size = mask_batch_size

    def selection_factors_of_aa_str(
        self,
        aa_sequence: str,
        head_type: HeadType = HeadType.MLM_HEAD,
        tdi_tokens: Optional[str] = None,
        distance: Optional[float] = None,
        raw: bool = False,
    ) -> List[torch.Tensor]:
        """Masked-marginal selection factors for a reference sequence.

        For a tokenized reference of length ``seq_len`` (``<cls>``-prefixed,
        ``^``-joined for paired chains), builds ``L`` masked copies — one per
        real (non-special) position, each with exactly that position's token
        replaced by ``<mask>`` — runs them through the encoder + ``MLMHead``,
        ``mask_batch_size`` copies per forward pass, and converts the result
        to log-odds relative to each position's own reference residue.
        ``CLS``, ``^``, ``PAD``, and ``X`` (ambiguous) positions are never
        masked or scored, via the same ``maskable_token_mask`` helper
        (``dasm2/data/masking.py``) ``MaskedTransformCollator`` uses to
        select positions during training — evaluating on exactly the
        position set the model was trained to reconstruct.

        Args:
            aa_sequence: Unified sequence string ("SEQ1^SEQ2" or single
                sequence), already ``^``-joined for paired chains.
            head_type: HeadType enum for head to use (default:
                HeadType.MLM_HEAD).
            tdi_tokens: Not supported — MLM has no 3Di structural input path.
                Must be ``None``. Only checked on the ``HeadType.MLM_HEAD``
                path; ignored (delegated to the base class) otherwise.
            distance: Not supported — the MLM encoder has no distance
                conditioning path (issue #199). Must be ``None`` on the
                ``HeadType.MLM_HEAD`` path; delegated to the base class
                otherwise, which raises if the encoder does need it.
            raw: Not supported — MLM logits have no separate raw/gauge-zapped
                distinction (there is no parent residue to zap against); the
                log-odds computation above is the only "gauge" this head has.
                Must be ``False`` on the ``HeadType.MLM_HEAD`` path; ignored
                (delegated to the base class) otherwise.

        Returns:
            List of selection-factor tensors, one per sequence segment
            (split on ``^``, matching ``split_predictions_by_sequences``'s
            convention). Each tensor has shape ``[segment_len,
            num_amino_acids]`` (``MLMHead.num_amino_acids``, 20 by default)
            with the reference residue's own column equal to 1.0 (``f[ref] =
            1``) by construction. Special/separator positions are excluded
            entirely (never a row in the output), rather than zeroed.

        Raises:
            ValueError: If model doesn't have the specified head, if
                ``tdi_tokens``/``raw`` are set for the ``HeadType.MLM_HEAD``
                path, or if the sequence has no maskable (real amino-acid)
                positions.
        """
        if head_type != HeadType.MLM_HEAD:
            # A model jointly training MLMHead alongside another head on
            # the same shared encoder (D1 head-centric architecture) still
            # builds an MLMCoreModel (models/factory.py dispatches on the
            # presence of MLMHead, not per-head-type) — but masked-marginal
            # scoring only makes sense for MLMHead's own reconstruction
            # logits. Every other head_type gets the base CoreModel's
            # generic single-unmasked-forward-pass behavior unchanged
            # (including its own tdi_tokens/raw support, which MLM itself
            # does not have).
            return super().selection_factors_of_aa_str(
                aa_sequence,
                head_type=head_type,
                tdi_tokens=tdi_tokens,
                distance=distance,
                raw=raw,
            )
        if distance is not None:
            raise ValueError(
                "MLMCoreModel does not support distance conditioning on the "
                "MLM_HEAD path."
            )
        if tdi_tokens is not None:
            raise ValueError(
                "MLMCoreModel does not support tdi_tokens. "
                "3Di structural input is not compatible with the MLM objective."
            )
        if raw:
            raise ValueError(
                "MLMCoreModel does not support raw=True. MLM logits have no "
                "separate raw/gauge-zapped distinction to select between."
            )
        self.require_head(head_type)

        tokens_1d = MLM_TOKENIZER.tokenize([aa_sequence])[0].to(self.device)

        def score_chunk(
            batch_tokens: torch.Tensor,
            positions: torch.Tensor,
            ref_residues: torch.Tensor,
        ) -> torch.Tensor:
            batch = {
                "aa_parents_idxs": batch_tokens,
                # No padding in this single-reference-sequence batch (every
                # row is the same length), so every position is valid.
                "mask": torch.ones_like(batch_tokens, dtype=torch.bool),
            }
            logits = self.forward(batch, head_names=[head_type])[head_type]
            return _masked_marginal_log_odds(logits, positions, ref_residues)

        with torch.no_grad():
            return masked_marginal_scan(
                tokens_1d,
                MLM_MASK_IDX,
                MLM_SEPARATOR_IDX,
                self.mask_batch_size,
                score_chunk,
            )
