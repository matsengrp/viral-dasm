"""3Di dataset wrapper for Whichmut-compatible datasets.

Wraps any dataset producing Whichmut-compatible __getitem__ output with matched
3Di structural token sequences, adding tdi_token_ids to each batch item.
Follows the decorator pattern from MPNNWhichmutDataset with __getattr__ delegation.

The wrapper reads 3Di token strings from text files named
``{structure_id}_3di.txt`` in a specified directory. Each file contains
a single line with the 3Di token string (one character per residue).
"""

from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from dasm2.sequence.tdi_constants import TDI_PAD_IDX, tdi_str_to_idxs


class TdiWhichmutDataset(Dataset[Dict[str, Any]]):
    """Wraps a Whichmut-compatible dataset with 3Di structural tokens.

    Loads per-structure 3Di text files at construction time, tokenizes them,
    validates lengths against sequence data, and adds tdi_token_ids to each
    batch item.

    Transparent delegation via __getattr__ means all base dataset methods
    (branch_lengths, update_branch_lengths, etc.) work unchanged.

    Args:
        base_dataset: Any dataset whose __getitem__ returns a dict with
            ``aa_parents_idxs`` and that provides ``max_aa_seq_len``.
            Compatible with both eager ``WhichmutDataset`` and unified
            ``TransformDataset``.
        tdi_tokens_dir: Directory containing {structure_id}_3di.txt files.
        structure_id_list: List of structure IDs (one per sequence) mapping each
            sequence to its 3Di file.
        pad_shorter_tokens: If True, 3Di tokens shorter than ``max_aa_seq_len``
            are right-padded with the padding index. Required for multi-structure
            datasets where proteins have different lengths (e.g., flu HA subtypes).
            Default False preserves strict length matching.

    Raises:
        ValueError: If lengths don't match or tokens contain invalid characters.
        FileNotFoundError: If a 3Di file is missing.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        tdi_tokens_dir: Path,
        structure_id_list: List[str],
        *,
        pad_shorter_tokens: bool = False,
    ) -> None:
        self.base_dataset = base_dataset
        self.tdi_tokens_dir = Path(tdi_tokens_dir)
        self._pad_shorter_tokens = pad_shorter_tokens

        # Load and validate 3Di tokens for all sequences
        self._tdi_token_ids_list: list[torch.Tensor] = []
        self._load_and_validate_tokens(structure_id_list)

    def _load_and_validate_tokens(self, structure_ids: List[str]) -> None:
        """Load 3Di token files and validate against sequence lengths.

        Validates once per unique structure_id rather than calling
        ``__getitem__`` per row, avoiding O(N) item construction overhead
        for large datasets.

        Args:
            structure_ids: List of structure IDs, one per sequence.

        Raises:
            FileNotFoundError: If any 3Di file is missing.
            ValueError: If token length doesn't match sequence length,
                or tokens contain invalid characters.
        """
        # Both WhichmutDataset and TransformDataset provide max_aa_seq_len
        seq_len = self.base_dataset.max_aa_seq_len  # type: ignore[attr-defined]

        # Load and validate once per unique structure_id
        cache: Dict[str, torch.Tensor] = {}
        for structure_id in dict.fromkeys(structure_ids):  # unique, preserving order
            tdi_path = self.tdi_tokens_dir / f"{structure_id}_3di.txt"
            if not tdi_path.exists():
                raise FileNotFoundError(
                    f"3Di token file not found: {tdi_path} "
                    f"(structure_id='{structure_id}')"
                )
            tdi_str = tdi_path.read_text().strip()
            tdi_idxs = tdi_str_to_idxs(tdi_str)
            tdi_len = tdi_idxs.shape[0]
            if tdi_len != seq_len:
                if tdi_len < seq_len and self._pad_shorter_tokens:
                    # Pad shorter 3Di tokens to match max_aa_seq_len.
                    # Used for multi-structure datasets where proteins have
                    # different lengths (e.g., flu HA: H3=566, H5=568).
                    pad = torch.full(
                        (seq_len - tdi_len,), TDI_PAD_IDX, dtype=tdi_idxs.dtype
                    )
                    tdi_idxs = torch.cat([tdi_idxs, pad])
                else:
                    raise ValueError(
                        f"3Di token length ({tdi_len}) != sequence length "
                        f"({seq_len}) for structure_id='{structure_id}'. "
                        f"The 3Di tokens must match the evolutionary sequence "
                        f"length. For multi-structure datasets with varying "
                        f"lengths, use pad_shorter_tokens=True."
                    )
            cache[structure_id] = tdi_idxs

        # Build per-row token list from cache
        for structure_id in structure_ids:
            self._tdi_token_ids_list.append(cache[structure_id])

    def __len__(self) -> int:
        """Return number of items in base dataset."""
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        """Get item with 3Di token IDs added to base batch.

        Args:
            idx: Index into the dataset.

        Returns:
            Dict containing all base dataset fields plus:
            - tdi_token_ids: [seq_len] 3Di structural token indices
        """
        item: Dict[str, Any] = self.base_dataset[idx]  # type: ignore[assignment]
        item["tdi_token_ids"] = self._tdi_token_ids_list[idx]
        return item

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base_dataset for transparent decoration.

        Args:
            name: Attribute name to look up.

        Returns:
            Attribute from base_dataset.
        """
        return getattr(self.base_dataset, name)
