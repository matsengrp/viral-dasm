"""DASMfit residual prediction head implementation.

Implements the DASMfit residual head g_θ: a small MLP that maps frozen DASM
embeddings to k×20 log-selection modifiers r_i(j, a). These define directions
in selection-factor space that can be fit per clonal family during phylogenetic
inference via alternating optimization over α.

Architecture:
    Linear(input_dim, hidden_dim) → ReLU → Linear(hidden_dim, k×20) → reshape → gauge

Key design:
    - Zero-init last layer so r_i ≈ 0 at training start (α=0 is the base model)
    - Gauge condition: r_i(j, root_aa_j) = 0 (avoids redundancy with DASM baseline)
    - Loss: ``DASMFitResidualLoss`` (negated Poisson log-likelihood
      Σ [M·exp(λ) − D·λ] where λ = Σ_i α_i·r_i)
"""

from typing import Dict

import torch
from torch import Tensor, nn

from .base import PredictionHead


class DASMFitResidualHead(PredictionHead):
    """DASMfit residual head predicting k×20 log-selection modifiers.

    Maps per-site DASM embeddings e_j ∈ ℝ^d to k residual components r_i(j, a),
    each an L×20 matrix of log-selection modifiers. The per-family scalar weights
    α_i are optimized separately in the alternating optimization loop.

    The gauge condition r_i(j, root_aa_j) = 0 is enforced by subtracting the
    root amino acid entry at each site after the MLP forward pass.

    Architecture:
        2-layer MLP: Linear(input_dim, hidden_dim) → ReLU → Linear(hidden_dim, k×20)
        Output reshaped to [batch, seq_len, k, 20], then gauge-subtracted.

    Loss:
        Negated Poisson log-likelihood using per-family α values and per-family
        D_j(a), M_j(a) count arrays from the batch (``D`` = data / observed,
        ``M`` = model / expected under base Q_j).
    """

    def __init__(
        self,
        input_dim: int,
        k: int = 3,
        hidden_dim: int = 128,
        num_amino_acids: int = 20,
    ) -> None:
        """Initialize DASMFitResidualHead.

        Args:
            input_dim: Input embedding dimension from DASM encoder.
            k: Number of residual components (directions in selection-factor space).
            hidden_dim: Hidden dimension of the 2-layer MLP.
            num_amino_acids: Number of amino acids to predict (default 20).
        """
        super().__init__()
        self.input_dim = input_dim
        self.k = k
        self.hidden_dim = hidden_dim
        self.num_amino_acids = num_amino_acids

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k * num_amino_acids),
        )

        # Zero-init last layer so r_i ≈ 0 at start.
        # ReLU(anything)*0 + 0 = 0, so output is exactly zero for any input.
        last_linear = self.mlp[-1]
        assert isinstance(last_linear, nn.Linear)
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, sequence_embeddings: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """Predict k×20 log-selection modifiers with gauge condition.

        Args:
            sequence_embeddings: Per-site embeddings [batch_size, seq_len, input_dim]
                from the frozen DASM encoder.
            batch: Batch data containing:
                - root_aa_idxs: Root amino acid indices [batch_size, seq_len],
                  values in [0, 19]. Used for gauge subtraction.

        Returns:
            Log-selection modifiers r_i(j, a) of shape [batch_size, seq_len, k, 20]
            with gauge condition r_i(j, root_aa_j) = 0 enforced.
        """
        batch_size, seq_len, _ = sequence_embeddings.shape

        # MLP forward: [B, L, D] -> [B, L, k*20]
        raw = self.mlp(sequence_embeddings)

        # Reshape to [B, L, k, num_amino_acids]
        output = raw.view(batch_size, seq_len, self.k, self.num_amino_acids)

        # Gauge subtraction: r_i(j, root_aa_j) = 0
        root_aa_idxs = batch["root_aa_idxs"]  # [B, L]
        # Gather the root AA values: [B, L, k, 1]
        root_idx_expanded = root_aa_idxs[:, :, None, None].expand(
            batch_size, seq_len, self.k, 1
        )
        root_values = torch.gather(
            output, dim=3, index=root_idx_expanded
        )  # [B, L, k, 1]
        result: Tensor = output - root_values  # Broadcast: [B, L, k, 20] - [B, L, k, 1]

        return result

    def to_config(self) -> "DASMFitResidualHeadConfig":
        """Reconstruct DASMFitResidualHeadConfig from head architecture.

        Returns:
            DASMFitResidualHeadConfig matching current head state.
        """
        return DASMFitResidualHeadConfig(
            input_dim=self.input_dim,
            k=self.k,
            hidden_dim=self.hidden_dim,
        )

    # from_config() inherited from PredictionHead base class


# Set config type after class definition to avoid circular import
from dasm2.config import DASMFitResidualHeadConfig  # noqa: E402

DASMFitResidualHead.head_config_type = DASMFitResidualHeadConfig
