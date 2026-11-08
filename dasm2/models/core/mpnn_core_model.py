"""MPNN-specific CoreModel for DASM2.

Provides MPNNCoreModel, a CoreModel subclass that handles MPNN-specific
input processing (passing mpnn_features and structural_logits from the batch
to the encoder). Follows the Open/Closed principle pattern established by
ESM2CoreModel.
"""

from typing import Dict, List, Optional

import torch

from dasm2.heads.types import HeadType
from dasm2.models.core.base import CoreModel
from dasm2.models.core.mpnn_light_attention import (
    MPNNLightAttentionEncoder,
)
from dasm2.sequence.constants import AA_AMBIG_IDX
from dasm2.tokenizers import AA_TOKENIZER


class MPNNCoreModel(CoreModel):
    """CoreModel specialized for MPNNLightAttentionEncoder.

    Handles MPNN-specific input processing:
    - Passes mpnn_features from batch to encoder
    - Uses aa_parents_idxs for both encoder embedding lookup and head diagonal zapping
    - Validates embedding length matches aa_parents_idxs (fails fast on mismatch)

    The MPNNWhichmutDataset provides batches with:
    - mpnn_features: [batch, seq_len, 256] precomputed decoder features
    - structural_logits: [batch, seq_len, 20] frozen log-likelihoods
    - aa_parents_idxs: [batch, seq_len] parent amino acid indices
    - mask: [batch, seq_len] valid position mask
    - Plus all standard Whichmut fields (codon indices, neutral rates, etc.)
    """

    encoder: MPNNLightAttentionEncoder  # Type narrowing for mypy

    def _build_encoder_input(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Build encoder input dict from batch fields.

        Args:
            batch: Input batch containing mpnn_features, aa_parents_idxs, mask.

        Returns:
            Dict with the three fields needed by MPNNLightAttentionEncoder.
        """
        return {
            "mpnn_features": batch["mpnn_features"],
            "aa_parents_idxs": batch["aa_parents_idxs"],
            "mask": batch["mask"],
        }

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        head_names: Optional[List[HeadType]] = None,
    ) -> Dict:
        """Forward pass with MPNN-specific input handling.

        Args:
            batch: Input batch containing:
                - mpnn_features: Precomputed decoder features [batch_size, seq_len, 256]
                - aa_parents_idxs: Parent AA indices [batch_size, seq_len]
                - mask: Position mask [batch_size, seq_len]
                Additional fields depend on head requirements.
            head_names: List of HeadType enums to compute, defaults to all active heads.

        Returns:
            Dict mapping HeadType enums to their outputs, plus "sequence_embeddings" key.

        Raises:
            ValueError: If required fields are missing from batch.
        """
        outputs = {}

        self._validate_mpnn_batch(batch)

        # MPNN-specific input: pass features and AA indices to encoder
        encoder_input = self._build_encoder_input(batch)
        sequence_embeddings = self.encoder(encoder_input)

        # Validate embedding length matches aa_parents_idxs (fail fast)
        self._validate_embedding_length(sequence_embeddings, batch["aa_parents_idxs"])

        outputs["sequence_embeddings"] = sequence_embeddings

        self._compute_head_outputs(outputs, sequence_embeddings, batch, head_names)

        return outputs

    def _validate_mpnn_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """Validate that batch contains required MPNN fields.

        Args:
            batch: Input batch to validate.

        Raises:
            ValueError: If required fields are missing.
        """
        if "aa_parents_idxs" not in batch:
            raise ValueError("Batch must contain 'aa_parents_idxs' field")
        if "mask" not in batch:
            raise ValueError("Batch must contain 'mask' field")
        if "mpnn_features" not in batch:
            raise ValueError(
                "MPNNCoreModel requires 'mpnn_features' field in batch. "
                "Wrap your dataset with MPNNWhichmutDataset."
            )

    def _validate_embedding_length(
        self,
        embeddings: torch.Tensor,
        aa_parents_idxs: torch.Tensor,
    ) -> None:
        """Validate that embedding length matches aa_parents_idxs length.

        Args:
            embeddings: Encoder output [batch, emb_len, d_output]
            aa_parents_idxs: Parent AA indices [batch, seq_len]

        Raises:
            ValueError: If lengths don't match.
        """
        target_length = aa_parents_idxs.shape[1]
        current_length = embeddings.shape[1]

        if current_length != target_length:
            raise ValueError(
                f"Encoder output length ({current_length}) doesn't match "
                f"aa_parents_idxs length ({target_length}). "
                f"This indicates a mismatch between MPNN features and sequence data."
            )

    def selection_factors_of_aa_str(  # type: ignore[override]
        self,
        aa_sequence: str,
        mpnn_features: torch.Tensor,
        structural_logits: torch.Tensor,
        head_type: HeadType = HeadType.WHICHMUT_RESIDUAL_HEAD,
    ) -> List[torch.Tensor]:
        """Compute selection factors from amino acid sequence with MPNN features.

        Routes through ``self.forward()`` so that encoder input construction and
        validation are identical to training.

        Args:
            aa_sequence: Amino acid sequence string (single sequence only).
            mpnn_features: Precomputed decoder features [seq_len, 256].
            structural_logits: Frozen log-likelihoods [seq_len, 20].
            head_type: HeadType enum for head to use.

        Returns:
            List containing a single selection factor tensor [seq_len, 20].

        Raises:
            ValueError: If model doesn't have the specified head.
        """
        self.require_head(head_type)

        aa_parents_idxs = AA_TOKENIZER.tokenize([aa_sequence]).to(self.device)
        mask = aa_parents_idxs != AA_AMBIG_IDX

        # Build batch with all fields that forward() expects
        batch = {
            "mpnn_features": mpnn_features.unsqueeze(0).to(self.device),
            "structural_logits": structural_logits.unsqueeze(0).to(self.device),
            "aa_parents_idxs": aa_parents_idxs,
            "mask": mask,
        }

        with torch.no_grad():
            outputs = self.forward(batch, head_names=[head_type])
            head_output = outputs[head_type]
            result = torch.exp(head_output).squeeze(0)

        return [result.cpu()]
