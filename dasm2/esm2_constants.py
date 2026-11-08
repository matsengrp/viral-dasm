"""Constants for ESM2 models.

Separated from esm2.py to allow config.py to import without circular dependencies.
"""

from typing import Dict, List, Tuple

#: Residues ESM2 was trained on per sequence: 1024 tokens less <cls> and <eos>.
#: ESM2's rotary position embedding runs on a longer input without error, but
#: off its training distribution (issue #1356).
ESM2_MAX_RESIDUES = 1022

#: Stride between the windows :func:`esm2_windows` tiles a longer sequence with.
ESM2_WINDOW_STRIDE = 511

# ESM2 model name to hidden dimension mapping
ESM2_HIDDEN_DIMS: Dict[str, int] = {
    "facebook/esm2_t6_8M_UR50D": 320,
    "facebook/esm2_t12_35M_UR50D": 480,
    "facebook/esm2_t30_150M_UR50D": 640,
    "facebook/esm2_t33_650M_UR50D": 1280,
    "facebook/esm2_t36_3B_UR50D": 2560,
}


def get_esm2_hidden_dim(model_name: str) -> int:
    """Get hidden dimension for an ESM2 model.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Hidden dimension of the model

    Raises:
        ValueError: If model name not recognized
    """
    if model_name not in ESM2_HIDDEN_DIMS:
        raise ValueError(
            f"Unknown ESM2 model: {model_name}. "
            f"Supported models: {list(ESM2_HIDDEN_DIMS.keys())}"
        )
    return ESM2_HIDDEN_DIMS[model_name]


def esm2_windows(
    length: int,
    window: int = ESM2_MAX_RESIDUES,
    stride: int = ESM2_WINDOW_STRIDE,
) -> List[Tuple[int, int, int, int]]:
    """Tile a sequence into windows ESM2 can read, each site owned by one window.

    A sequence of at most ``window`` residues is one window. A longer one is
    read in ``window``-residue windows starting every ``stride`` residues, plus
    one ending at the last residue, and each site is read from the window in
    which it sits most centrally (the earlier window on a tie). Issue #1356's
    ProteinGym scoring of ESM2 uses this rule, as does scoring a DASM trained on
    an ESM2 encoder.

    Args:
        length: Sequence length in residues.
        window: Residues per window.
        stride: Residues between window starts.

    Returns:
        ``(start, stop, own_start, own_stop)`` per window, in order: the window
        reads residues ``[start, stop)`` and owns sites ``[own_start, own_stop)``.
        The owned ranges partition ``[0, length)``.

    Raises:
        ValueError: If ``length`` is not positive.
    """
    if length < 1:
        raise ValueError(f"length must be positive, got {length}")
    if length <= window:
        return [(0, length, 0, length)]
    starts = sorted({*range(0, length - window, stride), length - window})
    best = [float("inf")] * length
    owner = [0] * length
    for i, start in enumerate(starts):
        center = start + window / 2
        for pos in range(start, start + window):
            distance = abs(pos - center)
            if distance < best[pos]:
                best[pos], owner[pos] = distance, i
    out = []
    for i, start in enumerate(starts):
        owned = [pos for pos in range(length) if owner[pos] == i]
        if owned:
            if owned[-1] - owned[0] + 1 != len(owned):
                raise AssertionError("a window's owned sites are not contiguous")
            out.append((start, start + window, owned[0], owned[-1] + 1))
    return out
