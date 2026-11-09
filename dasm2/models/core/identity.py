"""Identity (no-op) encoder, for the head that has no encoder.

``PerSiteParentChildHead``'s selection factor is a function of (site, parent
amino acid, child amino acid) and nothing else; its ``forward()`` documents
``sequence_embeddings: IGNORED``. But ``CoreModel.forward()`` calls
``self.encoder(encoder_input)`` unconditionally before dispatching to heads
(``dasm2/models/core/base.py``), so a model with no encoder still needs an
object there.

``IdentityEncoder`` is the honest expression of "this model has no
transformer": zero learnable parameters, one zero-tensor allocation per
forward, and an output nothing reads. Pairing an encoder-free head with
``TransformerEncoder`` instead would train a transformer whose output is
discarded -- parameters in the optimizer that cannot affect the loss.
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
    ``self.d_model`` by ``ModelConfig.validate_heads`` and the heads accept
    it for interface compatibility even though they do not use it.

    ``d_model`` defaults to 1 because nothing downstream reads the tensor's
    values, and the smallest legal allocation keeps the per-batch zero
    allocation effectively free.
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
