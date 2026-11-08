"""ESM2-specific CoreModel for DASM2.

This module provides ESM2CoreModel, a subclass of CoreModel that handles
ESM2-specific input processing. Using a subclass (instead of isinstance checks
in the base class) follows the Open/Closed principle and keeps the base
CoreModel clean.
"""

from functools import cached_property
from typing import Dict, List, Optional

import torch

from dasm2.esm2_constants import ESM2_MAX_RESIDUES, esm2_windows
from dasm2.heads.types import HeadType
from dasm2.models.core.base import CoreModel
from dasm2.models.core.esm2 import ESM2Encoder
from dasm2.sequence.constants import AA_AMBIG_IDX, AA_SEPARATOR
from dasm2.tokenizers import AminoAcidTokenizer, ESM2Tokenizer


class ESM2CoreModel(CoreModel):
    """CoreModel specialized for ESM2 encoder.

    Handles ESM2-specific input processing:
    - Uses ESM2-tokenized inputs (token_ids, masks) from ESM2DatasetWrapper
    - Strips <cls>/<eos> tokens in encoder (embeddings align with sequence positions)
    - Validates embedding length matches aa_parents_idxs (fails fast on mismatch)

    The TransformCollator pads ESM2 token_ids/masks at collation time using the "esm2"
    scale (batch_max + 2 for cls/eos). After ESM2Encoder strips these tokens, embeddings
    have shape [batch, max_aa_seq_len, hidden], matching the aa_parents_idxs shape.
    """

    encoder: ESM2Encoder  # Type narrowing for mypy

    @cached_property
    def _esm2_tokenizer(self) -> ESM2Tokenizer:
        """Cached ESM2 tokenizer for inference (avoids re-construction per call)."""
        return ESM2Tokenizer(model_name=self.encoder.model_name)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        head_names: Optional[List[HeadType]] = None,
    ) -> Dict:
        """Forward pass with ESM2-specific input handling.

        Args:
            batch: Input batch containing:
                - token_ids: ESM2-tokenized sequences [batch_size, max_len+2]
                - masks: ESM2 attention mask [batch_size, max_len+2]
                - aa_parents_idxs: DASM parent indices [batch_size, max_len]
                - mask: DASM position mask [batch_size, max_len]
                Additional fields depend on head requirements.
            head_names: List of HeadType enums to compute, defaults to all active heads

        Returns:
            Dict mapping HeadType enums to their outputs, plus "sequence_embeddings" key

        Raises:
            ValueError: If required ESM2 fields are missing from batch
        """
        outputs = {}

        # Validate required fields
        self._validate_esm2_batch(batch)

        # ESM2-specific input: use token_ids and masks from ESM2DatasetWrapper
        encoder_input = {
            "token_ids": batch["token_ids"],
            "masks": batch["masks"],
        }

        # Get embeddings - ESM2Encoder strips <cls>/<eos> internally
        sequence_embeddings = self.encoder(encoder_input)

        # Validate embeddings match aa_parents_idxs length (fail fast on mismatch)
        sequence_embeddings = self._validate_embedding_length(
            sequence_embeddings, batch["aa_parents_idxs"]
        )

        outputs["sequence_embeddings"] = sequence_embeddings

        self._compute_head_outputs(outputs, sequence_embeddings, batch, head_names)

        return outputs

    def _validate_esm2_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """Validate that batch contains required ESM2 fields.

        Args:
            batch: Input batch to validate

        Raises:
            ValueError: If required fields are missing
        """
        if "aa_parents_idxs" not in batch:
            raise ValueError("Batch must contain 'aa_parents_idxs' field")
        if "mask" not in batch:
            raise ValueError("Batch must contain 'mask' field")
        if "token_ids" not in batch:
            raise ValueError(
                "ESM2CoreModel requires 'token_ids' field in batch. "
                "Wrap your dataset with ESM2DatasetWrapper."
            )
        if "masks" not in batch:
            raise ValueError(
                "ESM2CoreModel requires 'masks' field in batch. "
                "Wrap your dataset with ESM2DatasetWrapper."
            )

    def _validate_embedding_length(
        self,
        embeddings: torch.Tensor,
        aa_parents_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """Validate that embedding length matches aa_parents_idxs length.

        ESM2 embeddings after stripping <cls>/<eos> must match the sequence length.
        If they don't match, this indicates a bug in tokenization or data preparation.

        Args:
            embeddings: ESM2 embeddings [batch, emb_len, hidden]
            aa_parents_idxs: Target sequence indices [batch, seq_len]

        Returns:
            embeddings unchanged (validation only)

        Raises:
            ValueError: If embedding length doesn't match sequence length
        """
        target_length = aa_parents_idxs.shape[1]
        current_length = embeddings.shape[1]

        if current_length != target_length:
            raise ValueError(
                f"ESM2 embedding length ({current_length}) doesn't match "
                f"aa_parents_idxs length ({target_length}). "
                f"This indicates a bug in tokenization or ESM2DatasetWrapper. "
                f"ESM2 tokenization (with <cls>/<eos> stripped) must produce "
                f"the same length as DASM tokenization."
            )

        return embeddings

    def selection_factors_of_aa_str(
        self,
        aa_sequence: str,
        head_type: HeadType = HeadType.SELECTION_FACTOR_HEAD,
        tdi_tokens: Optional[str] = None,
        distance: Optional[float] = None,
        raw: bool = False,
    ) -> List[torch.Tensor]:
        """ESM2-specific selection factors from amino acid sequence string.

        Routes through ``self.forward()`` so that encoder masking and embedding
        validation are identical to training. Supports paired sequences
        (e.g., "HEAVY^LIGHT") — the ESM2 tokenizer maps ^ to <unk>, matching
        training behavior where ESM2DatasetWrapper passes the full unified
        sequence to the encoder.

        A single sequence longer than ESM2's 1022 residues is read in windows,
        each site from the window where it is most central
        (:func:`~dasm2.esm2_constants.esm2_windows`), as issue #1356's
        zero-shot ESM2 comparator reads it.

        Args:
            aa_sequence: Amino acid sequence string. Supports both single
                sequences and paired sequences with ^ separator.
            head_type: HeadType enum for head to use (default: HeadType.SELECTION_FACTOR_HEAD)
            distance: ESM2CoreModel does not support distance conditioning; must
                be None. Present in the signature to satisfy the
                SelectionFactorModel protocol.
            raw: If True, return un-zapped selection factors (before gauge
                enforcement). Default False returns zapped SF.

        Returns:
            List of selection factor tensors, one per sequence segment.
            For single sequences: [tensor] with shape [seq_len, 20].
            For paired "HEAVY^LIGHT": [heavy_tensor, light_tensor].

        Raises:
            ValueError: If model doesn't have the specified head, or a paired
                sequence is longer than 1022 residues.
        """
        if tdi_tokens is not None:
            raise ValueError(
                "ESM2CoreModel does not support tdi_tokens. "
                "3Di structural input is not compatible with ESM2 tokenization."
            )
        if distance is not None:
            raise ValueError("ESM2CoreModel does not support distance conditioning.")

        self.require_head(head_type)

        if len(aa_sequence) > ESM2_MAX_RESIDUES:
            if AA_SEPARATOR in aa_sequence:
                raise ValueError(
                    f"a paired sequence of {len(aa_sequence)} residues is over "
                    f"ESM2's {ESM2_MAX_RESIDUES}, and only single sequences are "
                    "windowed"
                )
            return [
                torch.cat(
                    [
                        self._selection_factors_of_window(
                            aa_sequence[start:stop], head_type, raw
                        )[0][own_start - start : own_stop - start]
                        for start, stop, own_start, own_stop in esm2_windows(
                            len(aa_sequence)
                        )
                    ]
                )
            ]
        return self._selection_factors_of_window(aa_sequence, head_type, raw)

    def _selection_factors_of_window(
        self, aa_sequence: str, head_type: HeadType, raw: bool
    ) -> List[torch.Tensor]:
        """Selection factors of a sequence ESM2 reads in one pass."""
        # ESM2 tokenization for encoder — passes the full string including any
        # ^ separator, which the HuggingFace tokenizer maps to <unk>. This
        # matches training where ESM2DatasetWrapper passes unified sequences.
        esm2_batch = self._esm2_tokenizer.create_encoder_batch(
            [aa_sequence], device=str(self.device)
        )

        # DASM tokenization for aa_parents_idxs (needed for diagonal zapping)
        # and for splitting output by segment via ^ separator positions.
        dasm_tokenizer = AminoAcidTokenizer()
        aa_parents_idxs = dasm_tokenizer.tokenize([aa_sequence]).to(self.device)
        tokens_1d = aa_parents_idxs.squeeze(0)

        # Build combined batch with all fields that forward() expects
        batch = {
            "token_ids": esm2_batch["token_ids"],
            "masks": esm2_batch["masks"],
            "aa_parents_idxs": aa_parents_idxs,
            "mask": aa_parents_idxs != AA_AMBIG_IDX,
        }

        return self._run_inference(batch, tokens_1d, head_type, raw=raw)
