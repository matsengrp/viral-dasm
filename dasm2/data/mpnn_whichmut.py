"""MPNN dataset wrapper for Whichmut-compatible datasets.

Wraps any dataset producing Whichmut-compatible __getitem__ output with matched
precomputed ProteinMPNN features and log-likelihoods, adding mpnn_features and
structural_logits to each batch item. Follows the decorator pattern from
ESM2DatasetWrapper with __getattr__ delegation.
"""

from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from dasm2.data.mpnn_validation import (
    load_and_validate_mpnn_npz,
)


class MPNNWhichmutDataset(Dataset[Dict[str, Any]]):
    """Wraps a Whichmut-compatible dataset with ProteinMPNN features.

    Loads per-structure NPZ files at construction time, validates shapes and
    values, and adds mpnn_features and structural_logits to each batch item.

    Transparent delegation via __getattr__ means all base dataset methods
    (branch_lengths, update_branch_lengths, etc.) work unchanged.

    Args:
        base_dataset: Any dataset whose __getitem__ returns a dict with
            ``aa_parents_idxs`` (for length validation). Compatible with both
            eager ``WhichmutDataset`` and unified ``TransformDataset``.
        mpnn_features_dir: Directory containing {structure_id}.npz files.
        structure_id_list: List of structure IDs (one per sequence) mapping each
            sequence to its NPZ file.

    Raises:
        ValueError: If NPZ files are missing, shapes are wrong, or values
            contain NaN/Inf.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        mpnn_features_dir: Path,
        structure_id_list: List[str],
    ) -> None:
        self.base_dataset = base_dataset
        self.mpnn_features_dir = Path(mpnn_features_dir)

        # Load and validate all NPZ files at construction time
        self._features_list: List[torch.Tensor] = []
        self._loglikelihoods_list: List[torch.Tensor] = []
        self._load_and_validate_features(structure_id_list)

    def _load_and_validate_features(self, structure_ids: List[str]) -> None:
        """Load NPZ files for all sequences and validate shapes/values.

        Args:
            structure_ids: List of structure IDs, one per sequence.

        Raises:
            FileNotFoundError: If any NPZ file is missing.
            ValueError: If shapes are wrong or values contain NaN/Inf.
        """
        # Cache loaded NPZ data by structure_id (many sequences share a structure)
        cache: Dict[str, Dict[str, torch.Tensor]] = {}

        for idx, structure_id in enumerate(structure_ids):
            if structure_id not in cache:
                npz_path = self.mpnn_features_dir / f"{structure_id}.npz"
                features, loglikelihoods = load_and_validate_mpnn_npz(
                    npz_path, structure_id
                )

                cache[structure_id] = {
                    "features": features,
                    "loglikelihoods": loglikelihoods,
                }

            cached = cache[structure_id]

            # Validate feature length matches sequence length from base dataset
            base_item = self.base_dataset[idx]
            seq_len = base_item["aa_parents_idxs"].shape[0]
            feat_len = cached["features"].shape[0]
            if feat_len != seq_len:
                raise ValueError(
                    f"MPNN feature length ({feat_len}) != sequence length ({seq_len}) "
                    f"for index {idx}, structure_id='{structure_id}'. "
                    f"The MPNN features must match the evolutionary sequence length."
                )

            self._features_list.append(cached["features"])
            self._loglikelihoods_list.append(cached["loglikelihoods"])

    def __len__(self) -> int:
        """Return number of items in base dataset."""
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        """Get item with MPNN features added to base batch.

        Args:
            idx: Index into the dataset.

        Returns:
            Dict containing all base dataset fields plus:
            - mpnn_features: [seq_len, 256] decoder features
            - structural_logits: [seq_len, 20] frozen log-likelihoods
        """
        item: Dict[str, Any] = self.base_dataset[idx]  # type: ignore[assignment]
        item["mpnn_features"] = self._features_list[idx]
        item["structural_logits"] = self._loglikelihoods_list[idx]
        return item

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base_dataset for transparent decoration.

        Args:
            name: Attribute name to look up.

        Returns:
            Attribute from base_dataset.
        """
        return getattr(self.base_dataset, name)
