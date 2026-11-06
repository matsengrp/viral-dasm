"""3Di structural token vocabulary and tokenizer.

Foldseek's 3Di alphabet encodes backbone torsion patterns into 20 discrete
structural states (lowercase letters). This module follows SaProt conventions
for masking low-pLDDT regions and padding.

References:
    - Foldseek: van Kempen et al., Nature Biotechnology 2024
    - SaProt: Su et al., ICLR 2024
"""

import torch

# 20 Foldseek structural states (canonical 3Di alphabet, lowercase)
TDI_ALPHABET = "acdefghiklmnpqrstvwy"

# Special tokens following SaProt convention
TDI_MASK_TOKEN = "#"  # Low-pLDDT regions
TDI_PAD_TOKEN = "_"  # Padding for variable-length sequences

# Full 22-token alphabet: 20 states + mask + padding
TDI_TOKEN_STR = TDI_ALPHABET + TDI_MASK_TOKEN + TDI_PAD_TOKEN

# Index constants
TDI_MASK_IDX = TDI_TOKEN_STR.index(TDI_MASK_TOKEN)  # 20
TDI_PAD_IDX = TDI_TOKEN_STR.index(TDI_PAD_TOKEN)  # 21

# Lookup table: character -> index
_TDI_CHAR_TO_IDX = {ch: i for i, ch in enumerate(TDI_TOKEN_STR)}


def tdi_str_to_idxs(s: str) -> torch.Tensor:
    """Tokenize a 3Di string to index tensor.

    Case-insensitive: accepts both Foldseek's uppercase output (e.g. "ACDEF")
    and SaProt's lowercase convention (e.g. "acdef"). Input is normalized to
    lowercase before lookup.

    Args:
        s: String of 3Di tokens (e.g., "acdef#_" or "ACDEF#_").

    Returns:
        Long tensor of shape [len(s)] with token indices.

    Raises:
        ValueError: If any character is not in the 3Di alphabet.
    """
    s = s.lower()
    idxs = []
    for ch in s:
        idx = _TDI_CHAR_TO_IDX.get(ch)
        if idx is None:
            raise ValueError(
                f"Unknown 3Di token '{ch}'. Valid tokens: {TDI_TOKEN_STR!r}"
            )
        idxs.append(idx)
    return torch.tensor(idxs, dtype=torch.long)
