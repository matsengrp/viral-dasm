"""MPNN dataset wrapper for DMS datasets.

Wraps a DMS dataset with matched precomputed ProteinMPNN features and
log-likelihoods for the wildtype structure. Since all mutations in a DMS
domain share the same structure, a single set of MPNN features is loaded
and replicated for every item.

Follows the decorator pattern from MPNNWhichmutDataset with __getattr__
delegation.
"""

from pathlib import Path
from typing import Any, Dict

from torch.utils.data import Dataset

from dasm2.data.mpnn_validation import load_and_validate_mpnn_npz


class MPNNDMSDataset(Dataset[Dict[str, Any]]):
    """Wraps a DMS dataset with ProteinMPNN features for the wildtype structure.

    All mutations in the DMS domain share the same wildtype structure, so
    a single set of MPNN features is loaded and returned for every item.
    This is analogous to how DMSDataset shares the WT sequence across items.

    Compatible with both the eager ``DMSDataset`` and the unified
    ``_DMSDatasetAdapter``. The base dataset must provide ``__getitem__``,
    ``__len__``, and a ``seq_len`` property.

    Args:
        base_dataset: DMS dataset with functional score measurements.
            Must expose ``seq_len``, ``__getitem__``, and ``__len__``.
        mpnn_features_dir: Directory containing {structure_id}.npz files.
        structure_id: Identifier for the WT structure NPZ file.

    Raises:
        FileNotFoundError: If NPZ file not found for structure_id.
        ValueError: If shapes are wrong, values contain NaN/Inf, or
            feature length doesn't match WT sequence length.
    """

    def __init__(
        self,
        base_dataset: Any,
        mpnn_features_dir: Path,
        structure_id: str,
    ) -> None:
        self.base_dataset = base_dataset
        self.mpnn_features_dir = Path(mpnn_features_dir)
        self.structure_id = structure_id

        self._load_and_validate_features()

    def _load_and_validate_features(self) -> None:
        """Load and validate MPNN features for the WT structure.

        Raises:
            FileNotFoundError: If NPZ file not found.
            ValueError: If shapes or values are invalid.
        """
        npz_path = self.mpnn_features_dir / f"{self.structure_id}.npz"
        features, loglikelihoods = load_and_validate_mpnn_npz(
            npz_path, self.structure_id
        )

        # Validate feature length matches WT sequence length
        seq_len = self.base_dataset.seq_len
        feat_len = features.shape[0]
        if feat_len != seq_len:
            raise ValueError(
                f"MPNN feature length ({feat_len}) != WT sequence length ({seq_len}) "
                f"for structure_id='{self.structure_id}'. "
                f"The MPNN features must match the wildtype sequence length."
            )

        self.mpnn_features = features
        self.structural_logits = loglikelihoods

    def __len__(self) -> int:
        """Return number of items in base dataset."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item with MPNN features added to base DMS batch.

        Args:
            idx: Index into the dataset.

        Returns:
            Dict containing all base DMSDataset fields plus:
            - mpnn_features: [seq_len, 256] decoder features
            - structural_logits: [seq_len, 20] frozen log-likelihoods
        """
        item: Dict[str, Any] = self.base_dataset[idx]
        item["mpnn_features"] = self.mpnn_features
        item["structural_logits"] = self.structural_logits
        return item

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base_dataset for transparent decoration.

        Args:
            name: Attribute name to look up.

        Returns:
            Attribute from base_dataset.
        """
        return getattr(self.base_dataset, name)
