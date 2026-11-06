"""ESM2 pretrained encoder for DASM2."""

import logging
from typing import Dict

import torch
from torch import nn
from transformers import EsmModel

from dasm2.config import ESM2DecoderConfig, ESM2EncoderConfig
from dasm2.esm2_constants import ESM2_HIDDEN_DIMS, ESM2_MAX_RESIDUES

from .transformer_block import TransformerStack, make_softmax_attention

logger = logging.getLogger(__name__)


class ESM2Decoder(nn.Module):
    """Trainable transformer layers on ESM2's stripped embeddings (issue #1509).

    Projects ``[B, L, hidden]`` to ``[B, L, d_model]`` with a Linear layer,
    then runs a ``TransformerStack`` with softmax attention over the residue
    positions. It adds no positional encoding: ESM2's embeddings already
    carry position from its rotary attention.
    """

    def __init__(self, input_dim: int, config: ESM2DecoderConfig) -> None:
        """Build the projection and the stack.

        Args:
            input_dim: ESM2's hidden dimension
            config: Decoder width, depth, heads, FFN, dropout and norm placement
        """
        super().__init__()
        self.config = config
        self.projection = nn.Linear(input_dim, config.d_model)
        self.stack = TransformerStack(
            d_model=config.d_model,
            num_layers=config.num_layers,
            attention_factory=self._make_softmax_attention,
            dim_feedforward=config.dim_feedforward,
            norm_first=config.norm_first,
            dropout=config.dropout,
        )

    def _make_softmax_attention(self) -> nn.Module:
        """Create softmax multi-head attention for one layer."""
        return make_softmax_attention(
            self.config.d_model, self.config.nhead, self.config.dropout
        )

    def forward(self, embeddings: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Decode stripped ESM2 embeddings.

        Args:
            embeddings: ESM2 output with <cls>/<eos> stripped [B, L, hidden]
            masks: ESM2 attention mask over <cls>, residues, <eos> and
                padding [B, L + 2]

        Returns:
            Decoded embeddings [B, L, d_model]
        """
        # Stripped position j is token j + 1, a residue iff token j + 2 is
        # still unpadded; masks[:, 1:-1] would leave each shorter row's
        # <eos> open to attention.
        residue_mask = masks[:, 2:].bool()
        decoded: torch.Tensor = self.stack(
            self.projection(embeddings), src_key_padding_mask=~residue_mask
        )
        return decoded


class ESM2Encoder(nn.Module):
    """Pretrained ESM2 encoder for protein sequences.

    Wraps HuggingFace ESM2 models for use as DASM2 encoder backbone.
    Supports frozen (feature extraction) and unfrozen (fine-tuning) modes.

    Unlike TransformerEncoder which uses DASM's 22-token vocabulary,
    ESM2Encoder uses ESM2's native 33-token vocabulary. Input sequences
    must be tokenized using ESM2Tokenizer before passing to this encoder.

    Architecture:
        - Loads pretrained ESM2 model from HuggingFace
        - Optionally freezes all parameters for feature extraction
        - Strips <cls> and <eos> tokens from output embeddings
        - With ``config.decoder`` set, projects them to the decoder's width
          and runs trainable transformer layers over them (issue #1509).
          This ``ESM2Decoder`` sits outside ``self.esm``, so
          ``freeze_parameters``, ``num_frozen_layers`` and
          ``parameters_frozen`` concern ESM2 alone.
          ``CoreModel.freeze_encoder()`` freezes both.
        - Returns [batch, seq_len, d_model] embeddings

    Output Alignment:
        ESM2 naturally outputs [batch, seq_len+2, d_model] due to <cls>/<eos>.
        This encoder strips these tokens to output [batch, seq_len, d_model],
        ensuring positional alignment with:
        - TransformerEncoder output format
        - Prediction heads expecting sequence-aligned embeddings
        - DASM loss computation over sequence positions

    Limitations:
        - Paired sequences are supported: chain separators (^) are mapped to
          <unk> by ESM2Tokenizer, so they pass through without error
        - Requires ESM2Tokenizer for input preparation (not AminoAcidTokenizer)
    """

    # ESM2 vocabulary size (for validation)
    ESM2_VOCAB_SIZE = 33

    def __init__(self, config: ESM2EncoderConfig) -> None:
        """Initialize ESM2 encoder from config.

        Args:
            config: ESM2EncoderConfig with model_name and freeze_parameters settings
        """
        super().__init__()
        self.model_name = config.model_name

        # Load pretrained ESM2 model with PyTorch SDPA for flash attention
        self.esm = EsmModel.from_pretrained(
            config.model_name, attn_implementation="sdpa"
        )

        # Validate and store dimension - fail fast for unknown models
        actual_dim: int = int(self.esm.config.hidden_size)
        if config.model_name not in ESM2_HIDDEN_DIMS:
            available_models = list(ESM2_HIDDEN_DIMS.keys())
            raise ValueError(
                f"Unknown ESM2 model '{config.model_name}'. "
                f"Available models: {available_models}. "
                f"To use a custom model, add it to ESM2_HIDDEN_DIMS in esm2_constants.py."
            )
        expected_dim = ESM2_HIDDEN_DIMS[config.model_name]
        if actual_dim != expected_dim:
            raise ValueError(
                f"Model {config.model_name} has d_model={actual_dim} "
                f"but expected {expected_dim}. Model may be corrupted."
            )
        self._d_model: int = config.d_model

        # Store freezing config for to_config()
        self._num_frozen_layers = config.num_frozen_layers

        self.decoder = (
            ESM2Decoder(actual_dim, config.decoder)
            if config.decoder is not None
            else None
        )

        # Apply freezing strategy
        if config.num_frozen_layers is not None:
            self._freeze_early_layers(config.num_frozen_layers)
        elif config.freeze_parameters:
            self._freeze_parameters()

    def _freeze_parameters(self) -> None:
        """Freeze all ESM2 parameters."""
        for param in self.esm.parameters():
            param.requires_grad = False

    def _unfreeze_parameters(self) -> None:
        """Unfreeze all ESM2 parameters."""
        for param in self.esm.parameters():
            param.requires_grad = True

    def _freeze_early_layers(self, num_layers: int) -> None:
        """Freeze embedding layer and first N transformer layers.

        Args:
            num_layers: Number of transformer layers to freeze (0-indexed).
                Layers 0 to num_layers-1 will be frozen.

        This is a common transfer learning strategy that:
        - Preserves general features learned in early layers
        - Allows task-specific adaptation in later layers
        - Reduces gradient flow issues by limiting trainable parameters
        """
        total_layers = len(self.esm.encoder.layer)
        if num_layers > total_layers:
            raise ValueError(
                f"num_frozen_layers={num_layers} exceeds total layers={total_layers} "
                f"for model {self.model_name}"
            )

        # Always freeze embeddings (most general features)
        for param in self.esm.embeddings.parameters():
            param.requires_grad = False

        # Freeze specified number of transformer layers
        for i in range(num_layers):
            for param in self.esm.encoder.layer[i].parameters():
                param.requires_grad = False

        # Log what's being trained
        trainable_layers = total_layers - num_layers
        logger.info(
            f"ESM2: Froze embeddings + {num_layers} layers, "
            f"training last {trainable_layers} layers"
        )

    @property
    def d_model(self) -> int:
        """Output embedding dimension."""
        return self._d_model

    @property
    def parameters_frozen(self) -> bool:
        """Whether encoder parameters are currently frozen.

        This reflects actual runtime state, not configuration intent.
        Use this to check the current state after potential freeze/unfreeze calls.
        For head-only-freeze intent tracking, see
        ``CoreModel.encoder_frozen_via_freeze_call``.
        """
        return all(not p.requires_grad for p in self.esm.parameters())

    # Special token positions in ESM2 output
    CLS_TOKEN_POSITION = 0  # <cls> is always first
    EOS_TOKEN_OFFSET = -1  # <eos> is always last

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode sequences using ESM2.

        Args:
            batch: Input batch containing:
                - token_ids: ESM2-tokenized sequences [batch, seq_len] including <cls>/<eos>
                - masks: Attention mask [batch, seq_len] (True = valid, False = padding)

        Returns:
            Sequence embeddings [batch, seq_len-2, hidden_dim] with <cls>/<eos> stripped

        Raises:
            ValueError: If batch is empty, token_ids contain values outside ESM2
                vocabulary, shapes are inconsistent, or a sequence is longer
                than ESM2's 1022 residues (issue #1356: ESM2's rotary
                embedding would run on it silently, off its training
                distribution; drop such rows from training data, and score
                long sequences through
                ``ESM2CoreModel.selection_factors_of_aa_str``, which windows).

        Note:
            The returned embeddings have shape [batch, L, hidden_dim] where L is the
            original sequence length (without <cls>/<eos>). This matches the output
            format of TransformerEncoder for compatibility with prediction heads.
        """
        token_ids = batch["token_ids"]
        masks = batch["masks"]

        # Fail-fast on empty batch - indicates data loading issue
        if token_ids.numel() == 0:
            raise ValueError(
                "Received empty batch (token_ids has 0 elements). "
                "Empty batches indicate a data loading issue. "
                "Check that your dataset produces valid sequences."
            )

        # 1. Validate tokenization first (most fundamental - is this ESM2 data?)
        max_token = token_ids.max().item()
        if max_token >= self.ESM2_VOCAB_SIZE:
            raise ValueError(
                f"Token IDs contain values >= {self.ESM2_VOCAB_SIZE} (found max={max_token}). "
                "ESM2Encoder requires ESM2Tokenizer, not AminoAcidTokenizer. "
                "Use ESM2Tokenizer(model_name=...).create_encoder_batch() for input."
            )

        # 2. Validate shape consistency
        if token_ids.shape != masks.shape:
            raise ValueError(
                f"token_ids shape {token_ids.shape} doesn't match masks shape {masks.shape}"
            )

        # 3. ESM2 was trained on at most 1022 residues plus <cls>/<eos>.
        if token_ids.shape[1] > ESM2_MAX_RESIDUES + 2:
            raise ValueError(
                f"ESM2 input of {token_ids.shape[1] - 2} residues is over the "
                f"{ESM2_MAX_RESIDUES} ESM2 was trained on. Drop longer rows from "
                "training data; selection_factors_of_aa_str windows longer "
                "sequences for scoring."
            )

        # ESM2 forward pass
        outputs = self.esm(input_ids=token_ids, attention_mask=masks)

        # Strip special tokens to align with sequence positions
        embeddings = self._strip_special_tokens(outputs.last_hidden_state)

        if self.decoder is not None:
            embeddings = self.decoder(embeddings, masks)

        return embeddings

    def _strip_special_tokens(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Remove <cls> (first) and <eos> (last) tokens from embeddings.

        Args:
            embeddings: Shape [batch, seq_len+2, hidden_dim] with special tokens

        Returns:
            Embeddings with shape [batch, seq_len, hidden_dim] without special tokens
        """
        # Slice from position after <cls> to position before <eos>
        return embeddings[:, self.CLS_TOKEN_POSITION + 1 : self.EOS_TOKEN_OFFSET, :]

    def to_config(self) -> ESM2EncoderConfig:
        """Reconstruct ESM2EncoderConfig from encoder state.

        Returns:
            ESM2EncoderConfig matching current encoder configuration.
            The freeze_parameters field reflects actual parameter state.
        """
        return ESM2EncoderConfig(
            model_name=self.model_name,
            freeze_parameters=self.parameters_frozen,  # Queries actual parameter state
            num_frozen_layers=self._num_frozen_layers,
            decoder=None if self.decoder is None else self.decoder.config,
        )
