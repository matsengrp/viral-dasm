"""Identity (no-op) encoder for encoder-free heads.

DASM2 has several baseline heads that explicitly ignore the encoder output:
``PerSiteParentChildHead``, ``PerSiteChildHead``, ``PerSitePreferenceHead``,
``PerSiteMeanSelectionHead``, ``SingleValueHead``, ``SingleMatrixHead``. Each
head's ``forward()`` documents ``sequence_embeddings: IGNORED`` and they all
inherit from ``PerSiteEncoderFreeHead`` / ``PositionAgnosticHead``.

``CoreModel.forward()`` unconditionally calls ``self.encoder(encoder_input)``
before dispatching to heads (see ``dasm2/models/core/base.py``). Pairing
encoder-free heads with ``TransformerEncoder`` therefore burns compute and
adds ~77K dead parameters to the optimizer for no scientific gain. The
``IdentityEncoder`` is the honest expression of "this model has no
transformer": zero learnable parameters, a single zero-tensor allocation per
forward, output never read.

Reference: netam's ``ParentIndependentBinarySelectionModel`` takes the same
approach (no transformer, position/AA-indexed parameters only).
"""

from typing import TYPE_CHECKING, Dict

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from dasm2.config import IdentityEncoderConfig


class IdentityEncoder(nn.Module):
    """No-op encoder. Returns a zero tensor; contributes no parameters.

    Returns ``torch.zeros([batch, seq_len, d_model])`` on every forward so
    ``CoreModel.forward()``'s unconditional ``self.encoder(encoder_input)``
    call resolves to a tensor of the expected shape. Encoder-free heads
    ignore this tensor; head ``input_dim`` is still auto-set from
    ``self.d_model`` by ``ModelConfig.validate_heads`` and the heads
    accept it for interface compatibility even though they do not use it.

    ``d_model`` defaults to 1 because nothing downstream of the encoder
    reads the tensor's values, and the smallest legal allocation keeps
    the per-batch zero allocation effectively free. The config field is
    kept so that experiments that *do* want a particular ``d_model``
    (e.g., for SF projection compatibility with a future encoder-sharing
    head) can set one explicitly.
    """

    def __init__(self, config: "IdentityEncoderConfig") -> None:
        super().__init__()
        self.d_model = config.d_model

    def forward(self, encoder_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return ``torch.zeros([B, L, d_model])`` on the input device."""
        token_ids = encoder_input["token_ids"]
        batch_size, seq_len = token_ids.shape
        return torch.zeros(
            batch_size,
            seq_len,
            self.d_model,
            device=token_ids.device,
            dtype=torch.float32,
        )

    def to_config(self) -> "IdentityEncoderConfig":
        """Reconstruct ``IdentityEncoderConfig`` from encoder state."""
        from dasm2.config import IdentityEncoderConfig

        return IdentityEncoderConfig(d_model=self.d_model)
