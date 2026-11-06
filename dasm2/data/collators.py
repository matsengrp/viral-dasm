"""Custom collators for efficient batching.

Provides ``TransformCollator`` for the unified transform-based pipeline.
It pads variable-length per-sample tensors to the batch maximum.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

TensorScale = Literal["codon", "nt", "scalar", "list", "esm2", "ragged"]
VALID_SCALES: frozenset[str] = frozenset(
    {"codon", "nt", "scalar", "list", "esm2", "ragged"}
)

# Collator specs for the token_ids/masks fields added by ESM2DatasetWrapper.
# pad_value=1 is the ESM2 <pad> token id; pad_value=0 masks out padding positions.
ESM2_COLLATOR_SPECS: dict[str, tuple[float, TensorScale]] = {
    "token_ids": (1.0, "esm2"),
    "masks": (0.0, "esm2"),
}


def pad_and_stack(
    tensors: List[torch.Tensor],
    target_len: int,
    pad_value: Union[float, bool],
) -> torch.Tensor:
    """Pad variable-length tensors to target_len and stack into a batch.

    Supports 1D tensors [L] and 2D tensors [L, D]. Pads along the first
    dimension (sequence length).

    Uses torch.nn.utils.rnn.pad_sequence (C++ loop) instead of per-tensor
    F.pad calls for ~2-5x faster collation on large batches.

    Args:
        tensors: List of tensors with the same dtype and number of dimensions.
            First dimension may vary (variable-length sequences).
        target_len: Length to pad to along dimension 0.
        pad_value: Value to use for padding positions.

    Returns:
        Stacked tensor of shape [batch_size, target_len] (1D input)
        or [batch_size, target_len, D] (2D input).

    Raises:
        ValueError: If any tensor has more than 2 dimensions.
    """
    bad_dims = {i: t.dim() for i, t in enumerate(tensors) if t.dim() > 2}
    if bad_dims:
        raise ValueError(
            f"pad_and_stack only supports 1D and 2D tensors, got >2D at indices: "
            f"{bad_dims}"
        )
    # pad_sequence pads to the max length in the list (C++ loop, no Python iteration)
    stacked = torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=float(pad_value)
    )
    # Extend to target_len if batch max < target_len (e.g. nt = 3 * codon max)
    actual_len = stacked.shape[1]
    if actual_len < target_len:
        pad_size = target_len - actual_len
        if stacked.dim() == 2:
            stacked = F.pad(stacked, (0, pad_size), value=float(pad_value))
        else:
            stacked = F.pad(stacked, (0, 0, 0, pad_size), value=float(pad_value))
    return stacked


def split_tensor_samples(
    samples: List[Dict[str, Any]],
    tensor_specs: Dict[str, Tuple[float, TensorScale]],
    extra_keys: frozenset,
    collator_name: str,
) -> List[Dict[str, Any]]:
    """Validate sample keys and project each sample down to ``tensor_specs``.

    Shared by head-specific collators that wrap ``TransformCollator`` for
    their tensor_specs-covered keys but handle other keys (e.g. per-sample
    strings, batch-level targets) explicitly. Every key on every sample must
    be either declared in ``tensor_specs`` or listed in ``extra_keys``, so a
    transform that starts emitting an unhandled key fails loudly instead of
    being silently dropped.

    Args:
        samples: Per-sample dicts from ``TransformDataset.__getitem__``.
        tensor_specs: Keys handled by the wrapped ``TransformCollator``.
        extra_keys: Keys handled explicitly by the caller, not passed to
            ``TransformCollator``.
        collator_name: Name of the calling collator, for the assertion
            message.

    Returns:
        ``samples`` projected down to only the ``tensor_specs`` keys, ready
        to pass to ``TransformCollator``.

    Raises:
        ValueError: If ``samples`` is empty.
    """
    if not samples:
        raise ValueError("Cannot collate empty sample list.")

    spec_keys = set(tensor_specs)
    unspec_keys = {
        k for s in samples for k in s if k not in spec_keys and k not in extra_keys
    }
    assert not unspec_keys, (
        f"{collator_name} received sample keys with no tensor_specs entry: "
        f"{sorted(unspec_keys)}. Either add them to tensor_specs or drop them "
        f"upstream."
    )
    return [{k: s[k] for k in tensor_specs} for s in samples]


@dataclass
class TransformCollator:
    """Pad variable-length per-sample tensors to batch-max and stack.

    Used with TransformDataset. Each key in tensor_specs declares how
    to collate that tensor:

    tensor_specs maps key name -> (pad_value, scale) where:
        - "codon": pad to batch max sequence length
        - "nt": pad to 3 * batch max (nucleotide-level)
        - "esm2": pad to batch max + 2 (for ESM2 <cls>/<eos> tokens)
        - "ragged": pad to this key's own batch maximum (sparse site records)
        - "scalar": torch.stack directly (pad_value is ignored)

    Args:
        tensor_specs: Mapping of key name to (pad_value, scale) tuples.

    Example:
        >>> collator = TransformCollator(tensor_specs={
        ...     "aa_parents_idxs": (20, "codon"),
        ...     "mask": (False, "codon"),
        ...     "delta_affinity": (0.0, "scalar"),
        ... })
    """

    tensor_specs: Dict[str, Tuple[float, TensorScale]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate tensor_specs and cache derived state at construction time."""
        for key, (_, scale) in self.tensor_specs.items():
            if scale not in VALID_SCALES:
                raise ValueError(
                    f"Invalid scale {scale!r} for key '{key}'. "
                    f"Valid scales: {sorted(VALID_SCALES)}"
                )
        codon_keys = [k for k, (_, s) in self.tensor_specs.items() if s == "codon"]
        non_list_keys = [k for k, (_, s) in self.tensor_specs.items() if s != "list"]
        if non_list_keys and not codon_keys:
            raise ValueError(
                "TransformCollator requires at least one codon-scale key "
                "to determine batch max length when non-list keys are present."
            )
        # Cache the key used to determine batch_max. All codon-scale tensors
        # must have the same per-sample length (they all derive from the same
        # tokenized sequence), so picking the first is sufficient.
        self._batch_max_key: str = codon_keys[0] if codon_keys else ""

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a list of sample dicts into a batched dict.

        Args:
            samples: List of sample dicts from TransformDataset.__getitem__.

        Returns:
            Dict of batched tensors (padded to batch maximum length) and
            lists (for ``"list"``-scale keys, kept as ``List[Tensor]``).

        Raises:
            ValueError: If samples is empty.
        """
        if not samples:
            raise ValueError("Cannot collate empty sample list")

        # Warn about keys present in samples but not in tensor_specs
        sample_keys = set(samples[0].keys())
        spec_keys = set(self.tensor_specs.keys())
        unexpected = sample_keys - spec_keys
        if unexpected:
            logger.warning(
                "TransformCollator: keys %s in samples but not in tensor_specs "
                "(they will be dropped). If a transform produces these keys, "
                "add them to tensor_specs.",
                sorted(unexpected),
            )

        # Determine batch_max from cached codon-scale key
        batch_max = (
            max(s[self._batch_max_key].shape[0] for s in samples)
            if self._batch_max_key
            else 0
        )

        batch: Dict[str, Any] = {}
        for key, (pad_val, scale) in self.tensor_specs.items():
            tensors = [s[key] for s in samples]
            if scale == "list":
                batch[key] = tensors
            elif scale == "scalar":
                batch[key] = torch.stack(tensors)
            elif scale == "nt":
                batch[key] = pad_and_stack(tensors, batch_max * 3, pad_val)
            elif scale == "esm2":
                batch[key] = pad_and_stack(tensors, batch_max + 2, pad_val)
            elif scale == "ragged":
                batch[key] = pad_and_stack(tensors, 0, pad_val)
            elif scale == "codon":
                batch[key] = pad_and_stack(tensors, batch_max, pad_val)
        return batch
