"""Process-level policy for ``nn.MultiheadAttention``'s fused inference fast path.

Kept as a leaf module (torch only) so ``dasm2/__init__.py`` can apply the policy
without importing the model packages.
"""

import contextlib
from typing import Iterator

import torch


def disable_mha_fastpath() -> None:
    """Turn off PyTorch's fused inference fast path for the whole process.

    The fast path (``torch._native_multi_head_attention``, entered only in eval
    mode under ``no_grad``) materializes the whole ``[B, nhead, L, L]`` attention
    tensor and addresses it with 32-bit element offsets. Once
    ``B * nhead * L**2`` exceeds ``2**31 - 1`` the kernel reads out of bounds and
    the process dies with ``CUDA error: an illegal memory access was
    encountered``. The cutoff is exactly that, measured on an L40S at two
    geometries: B=536/L=1000 and B=2047/L=500 pass, B=537/L=1000
    and B=2148/L=500 fault.

    Disabling it routes attention through ``scaled_dot_product_attention``, which
    never materializes that tensor. That is a strict win at DASM2's scale rather
    than a tradeoff: ``d_model`` is small (32 at the 77k tier), so the fused
    kernel's arithmetic saving never pays for its memory traffic. Measured on an
    L40S, peak GPU memory and time per forward, fast path on -> off:
    B=256/L=1000, 7.80 -> 0.35 GiB and 154 -> 59 ms; B=64/L=200,
    0.095 -> 0.027 GiB and 2.04 -> 1.77 ms.

    Process-wide rather than scoped to ``TransformerBlock._sa_block`` because the
    exposure is process-wide. Every eval-mode forward is at risk, not just
    ``oe-plot`` -- in-training validation runs at ``val_batch_size`` under the
    same conditions -- and netam models driven through ``represent()`` reach
    netam's own ``nn.TransformerEncoder``, which a DASM2 call-site guard would
    never cover. Scoping also means flipping a non-thread-local global on and off
    around every forward, so a concurrent forward could observe the restored
    value; setting it once never flips.

    The cost is that ``import dasm2`` mutates process-wide torch state. Acceptable
    here: DASM2 is an application with no external library users (Constitution
    VIII), and there is no DASM2 geometry where the fast path is the faster
    choice. Use ``mha_fastpath_enabled`` to opt back in for a specific block.

    Note this also disables ``nn.TransformerEncoder``'s NestedTensor path, which
    torch gates on the same flag.
    """
    torch.backends.mha.set_fastpath_enabled(False)


@contextlib.contextmanager
def mha_fastpath_enabled() -> Iterator[None]:
    """Restore PyTorch's fused fast path for the duration of the block.

    The inverse of :func:`disable_mha_fastpath`, for tests that exist to document
    stock PyTorch behavior rather than DASM2's. Restores the previous value on
    exit, including when the body raises, so it composes and nests.

    Not for production code: re-enabling reintroduces the crash for any
    eval-mode forward past the int32 attention cutoff.
    """
    was_enabled = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(True)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(was_enabled)
