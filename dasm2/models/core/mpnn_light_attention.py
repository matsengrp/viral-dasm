"""ThermoMPNN-compatible light-attention/MLP encoder for DASM2.

Processes precomputed ProteinMPNN decoder features (256-dim) concatenated with
a frozen sequence embedding (128-dim) through a per-position light-attention
module and MLP, matching ThermoMPNN's stability prediction architecture
(Dieckhaus et al. 2024).

Architecture per position:
    1. Concat decoder features [256] + frozen seq embedding [128] → [384]
    2. Light attention: two parallel Conv1d(384, 384, k=9, pad=4) paths
       - Feature path with dropout → content features
       - Attention path with softmax → attention weights
       - Element-wise multiply → reweighted features [384]
    3. Encoder MLP: Linear(384, 64) → ReLU → Linear(64, 32) → ReLU → [32]
"""

import logging
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from dasm2.config import MPNNLightAttentionEncoderConfig

logger = logging.getLogger(__name__)

# ProteinMPNN decoder feature dimension (2 decoder layers × 128)
MPNN_DECODER_DIM = 256
# ProteinMPNN sequence embedding dimension
MPNN_SEQ_EMBED_DIM = 128
# Combined input dimension after concatenation
COMBINED_INPUT_DIM = MPNN_DECODER_DIM + MPNN_SEQ_EMBED_DIM  # 384
# Number of standard amino acids (excludes X)
NUM_STANDARD_AAS = 20


class MPNNLightAttentionEncoder(nn.Module):
    """ThermoMPNN-compatible light-attention/MLP encoder.

    Concatenates precomputed ProteinMPNN decoder features (256-dim) with a
    frozen sequence embedding lookup (128-dim), then applies a per-position
    light-attention module and encoder MLP. Each position is processed
    independently (no cross-position mixing).

    The architecture has exact dimensional compatibility with ThermoMPNN,
    enabling optional warm-start from ThermoMPNN pretrained weights.

    Input dict keys:
        mpnn_features: [batch, seq_len, 256] - decoder hidden states
        aa_parents_idxs: [batch, seq_len] - parent AA indices for embedding lookup
        mask: [batch, seq_len] - valid position mask (unused in forward, for API consistency)

    Output: [batch, seq_len, d_output] (default d_output=32)
    """

    def __init__(self, config: MPNNLightAttentionEncoderConfig) -> None:
        """Initialize MPNNLightAttentionEncoder from config.

        Args:
            config: Encoder configuration with architecture params and paths.

        Raises:
            FileNotFoundError: If seq_embedding_path does not exist.
            ValueError: If seq_embedding has unexpected shape.
        """
        super().__init__()
        self._config = config

        # Load frozen sequence embedding from ProteinMPNN W_s matrix
        seq_embedding_weights = np.load(config.seq_embedding_path)
        if seq_embedding_weights.shape != (21, MPNN_SEQ_EMBED_DIM):
            raise ValueError(
                f"Expected seq_embedding shape [21, {MPNN_SEQ_EMBED_DIM}], "
                f"got {list(seq_embedding_weights.shape)}. "
                f"File: {config.seq_embedding_path}"
            )
        # Take first 20 rows (drop X token at index 20)
        self.seq_embedding = nn.Embedding(NUM_STANDARD_AAS, MPNN_SEQ_EMBED_DIM)
        self.seq_embedding.weight.data.copy_(
            torch.from_numpy(seq_embedding_weights[:NUM_STANDARD_AAS])
        )
        self.seq_embedding.weight.requires_grad = False

        # Light attention: two parallel Conv1d paths
        # Feature path: Conv1d + Dropout → content features
        self.feature_conv = nn.Conv1d(
            COMBINED_INPUT_DIM,
            COMBINED_INPUT_DIM,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2,
        )
        self.feature_dropout = nn.Dropout(config.dropout)

        # Attention path: Conv1d + Softmax → attention weights
        self.attention_conv = nn.Conv1d(
            COMBINED_INPUT_DIM,
            COMBINED_INPUT_DIM,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2,
        )

        # Encoder MLP: [384] → hidden_dims → [d_output]
        # ThermoMPNN pattern: Linear(384,64) → ReLU → Linear(64,32) → ReLU
        layer_dims = [COMBINED_INPUT_DIM] + list(config.hidden_dims) + [config.d_output]
        mlp_layers: list[nn.Module] = []
        for i in range(len(layer_dims) - 1):
            mlp_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, encoder_input: Dict[str, Tensor]) -> Tensor:
        """Forward pass: concat → light attention → MLP.

        Args:
            encoder_input: Dict with keys:
                mpnn_features: [batch, seq_len, 256]
                aa_parents_idxs: [batch, seq_len]
                mask: [batch, seq_len] (for API consistency)

        Returns:
            Encoder output [batch, seq_len, d_output]
        """
        mpnn_features = encoder_input["mpnn_features"]  # [B, L, 256]
        aa_parents_idxs = encoder_input["aa_parents_idxs"]  # [B, L]

        # Clamp indices to [0, 19] for safe embedding lookup
        # (handles ambiguous/padding positions >= 20; mask ensures they don't affect loss)
        clamped_idxs = aa_parents_idxs.clamp(0, NUM_STANDARD_AAS - 1)
        seq_embed = self.seq_embedding(clamped_idxs)  # [B, L, 128]

        # Concatenate: [B, L, 256] + [B, L, 128] → [B, L, 384]
        combined = torch.cat([mpnn_features, seq_embed], dim=-1)

        B, L, C = combined.shape

        # Reshape for Conv1d: [B*L, C, 1] (per-position, channel dim = 384, spatial = 1).
        # With spatial dim 1, the Conv1d(k=9, pad=4) degenerates to using only its
        # center weight — this faithfully reproduces ThermoMPNN's per-position
        # light attention which operates over channels, not sequence positions.
        x = combined.reshape(B * L, C, 1)

        # Light attention: two parallel paths
        features = self.feature_dropout(self.feature_conv(x))  # [B*L, 384, 1]
        attention = torch.softmax(self.attention_conv(x), dim=-1)  # [B*L, 384, 1]
        attended = features * attention  # [B*L, 384, 1]

        # Reshape back: [B*L, 384, 1] → [B, L, 384]
        attended = attended.squeeze(-1).reshape(B, L, C)

        # Encoder MLP: [B, L, 384] → [B, L, d_output]
        result: Tensor = self.mlp(attended)
        return result

    def to_config(self) -> MPNNLightAttentionEncoderConfig:
        """Reconstruct MPNNLightAttentionEncoderConfig from encoder state.

        Returns:
            Config matching current encoder architecture.
        """
        return MPNNLightAttentionEncoderConfig(
            d_output=self._config.d_output,
            hidden_dims=list(self._config.hidden_dims),
            kernel_size=self._config.kernel_size,
            dropout=self._config.dropout,
            seq_embedding_path=self._config.seq_embedding_path,
            thermompnn_weights_path=self._config.thermompnn_weights_path,
        )

    def load_thermompnn_weights(self, path: str) -> Tuple[Tensor, Tensor]:
        """Load ThermoMPNN pretrained weights into encoder layers.

        Maps ThermoMPNN checkpoint weights to:
        - Light attention Conv1d layers (exact shape match)
        - Encoder MLP linear layers (384→64, 64→32)

        Returns the final ThermoMPNN output layer weights [21, 32] and bias [21]
        so the caller can warm-start prediction heads.

        Args:
            path: Path to ThermoMPNN checkpoint (.pt file).

        Returns:
            Tuple of (head_weight [21, 32], head_bias [21]) from ThermoMPNN's
            final Linear(32, 21) layer, for head warm-start.

        Raises:
            FileNotFoundError: If checkpoint path doesn't exist.
            KeyError: If checkpoint doesn't contain expected weight keys.
        """
        checkpoint: dict = torch.load(path, map_location="cpu", weights_only=True)

        # Handle different checkpoint formats
        state_dict: dict = checkpoint.get("state_dict", checkpoint)

        # Load light attention Conv1d weights
        self.feature_conv.weight.data.copy_(state_dict["feature_conv.weight"])
        assert self.feature_conv.bias is not None
        self.feature_conv.bias.data.copy_(state_dict["feature_conv.bias"])
        self.attention_conv.weight.data.copy_(state_dict["attention_conv.weight"])
        assert self.attention_conv.bias is not None
        self.attention_conv.bias.data.copy_(state_dict["attention_conv.bias"])

        # Load encoder MLP weights (matches ThermoMPNN's Sequential indexing)
        # mlp.0 = Linear(384, 64), mlp.2 = Linear(64, 32)
        mlp_0 = self.mlp[0]
        mlp_2 = self.mlp[2]
        assert isinstance(mlp_0, nn.Linear) and isinstance(mlp_2, nn.Linear)
        mlp_0.weight.data.copy_(state_dict["mlp.0.weight"])
        assert mlp_0.bias is not None
        mlp_0.bias.data.copy_(state_dict["mlp.0.bias"])
        mlp_2.weight.data.copy_(state_dict["mlp.2.weight"])
        assert mlp_2.bias is not None
        mlp_2.bias.data.copy_(state_dict["mlp.2.bias"])

        # Return head output weights for caller to use
        head_weight: Tensor = state_dict["head.weight"]  # [21, 32]
        head_bias: Tensor = state_dict["head.bias"]  # [21]

        logger.info(
            "Loaded ThermoMPNN weights: Conv1d (feature + attention), "
            "MLP (384→64, 64→32), head (32→21)"
        )

        return head_weight, head_bias
