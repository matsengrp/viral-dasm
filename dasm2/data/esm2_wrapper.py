"""ESM2 dataset wrapper for DASM2 training."""

from typing import Any, Dict

from torch.utils.data import Dataset

from dasm2.data.protocols import SequenceProvider
from dasm2.tokenizers import ESM2Tokenizer


class ESM2DatasetWrapper(Dataset[Dict[str, Any]]):
    """Wraps a DASM2 dataset with ESM2-specific tokenization.

    Takes an existing dataset that provides raw sequences and re-tokenizes
    them using ESM2Tokenizer for use with ESM2Encoder.

    This wrapper implements the Decorator pattern with transparent delegation:
    all attribute/method access not defined on the wrapper is delegated to the
    underlying base_dataset. This allows callers to use the wrapper as if it
    were the base dataset for methods like `update_branch_lengths()`,
    `get_branch_length_optimization_inputs()`, and `branch_lengths`.

    Only `__getitem__` is modified to return ESM2-tokenized items.

    Example:
        >>> from dasm2.data.evolutionary_dataset import create_evolutionary_dataset
        >>> from dasm2.data.esm2_wrapper import ESM2DatasetWrapper
        >>>
        >>> base_dataset = create_evolutionary_dataset(...)
        >>> esm2_dataset = ESM2DatasetWrapper(
        ...     base_dataset,
        ...     model_name="facebook/esm2_t30_150M_UR50D",
        ... )
        >>> item = esm2_dataset[0]  # Returns ESM2-tokenized item
        >>> # Transparent delegation - works like base_dataset:
        >>> esm2_dataset.update_branch_lengths(new_lengths)
        >>> inputs = esm2_dataset.get_branch_length_optimization_inputs()
    """

    def __init__(
        self,
        base_dataset: Dataset[dict[str, Any]],
        model_name: str = "facebook/esm2_t30_150M_UR50D",
    ) -> None:
        """Initialize ESM2 dataset wrapper.

        Args:
            base_dataset: Dataset providing items with 'parent_sequence' or 'sequence'.
            model_name: HuggingFace ESM2 model identifier for tokenization
        """
        self.base_dataset = base_dataset
        self.tokenizer = ESM2Tokenizer(model_name=model_name)

    def __len__(self) -> int:
        """Return number of items in base dataset."""
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get item with ESM2 tokenization.

        When used with TransformCollator (streaming/parquet pipeline), the collator
        handles padding via the "esm2" scale (batch_max + 2). When used without a
        collator, tensors are returned at their natural length (seq_len + 2 for
        <cls>/<eos>).

        Args:
            idx: Index into the dataset

        Returns:
            Dict containing all original fields plus ESM2-tokenized
            'token_ids' and 'masks' tensors

        Raises:
            ValueError: If item doesn't contain 'parent_sequence' or 'sequence'
        """
        item = self.base_dataset[idx]

        # Extract sequence (try parent_sequence first, fall back to sequence)
        sequence = self._extract_sequence(item, idx)

        # Re-tokenize with ESM2
        esm2_batch = self.tokenizer.create_encoder_batch([sequence])
        token_ids = esm2_batch["token_ids"].squeeze(0)
        masks = esm2_batch["masks"].squeeze(0)

        # Merge ESM2 tokenization with original item (replacing token_ids/masks)
        return {
            **item,
            "token_ids": token_ids,
            "masks": masks,
        }

    def _extract_sequence(self, item: Dict[str, Any], idx: int) -> str:
        """Extract sequence from base dataset via SequenceProvider protocol.

        Args:
            item: Dataset item dict (unused, kept for potential future use)
            idx: Item index

        Returns:
            Sequence string

        Raises:
            TypeError: If base_dataset doesn't implement SequenceProvider
        """
        if not isinstance(self.base_dataset, SequenceProvider):
            raise TypeError(
                f"base_dataset must implement SequenceProvider protocol (have aa_parents property). "
                f"Got {type(self.base_dataset).__name__}"
            )
        return str(self.base_dataset.aa_parents[idx])

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base_dataset for transparent decoration.

        This enables the wrapper to be used as a drop-in replacement for the
        base dataset - methods like `update_branch_lengths()` and properties
        like `branch_lengths` are automatically delegated.

        Args:
            name: Attribute name to look up

        Returns:
            Attribute from base_dataset

        Note:
            This is only called for attributes not found on the wrapper itself.
            Attributes like `base_dataset` and `tokenizer` are found directly
            on the wrapper and bypass this method.
        """
        return getattr(self.base_dataset, name)
