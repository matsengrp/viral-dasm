"""Hit class computation for multihit correction.

The hit class of a codon pair is the number of nucleotide positions that differ
(0, 1, 2, or 3). This module provides:

1. Hit class and position-mask tensors
2. ``position_mask`` for three-bit endpoint masks from codon strings
3. Hit class computation functions (ported from netam.hit_class)
4. HitClassModel class for applying learned hit class corrections

Multi-hit corrections adjust probabilities based on learned factors for each
hit class, accounting for multiple mutations at the same site during a single
evolutionary event.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from dasm2.molevol.tensor_ops import (
    flatten_codons,
    set_parent_codon_prob,
    unflatten_codons,
)
from dasm2.sequence.constants import CODON_TO_IDX
from dasm2.sequence.conversion import flatten_codon_idxs
from dasm2.utils.tensor import clamp_probability, clamp_probability_above_only

# =============================================================================
# Hit Class Tensor (ported from netam.hit_class)
# =============================================================================

_NUM_BASES: int = 4


def _build_pairwise_differences() -> np.ndarray:
    """Build the broadcast parent/child difference indicators."""
    i, j, k = np.indices((_NUM_BASES, _NUM_BASES, _NUM_BASES))
    codon = np.stack((i, j, k), axis=-1)  # (4, 4, 4, 3)

    parent_expanded = codon[:, :, :, np.newaxis, np.newaxis, np.newaxis, :]
    child_expanded = codon[np.newaxis, np.newaxis, np.newaxis, :, :, :, :]
    return np.asarray(parent_expanded != child_expanded, dtype=np.bool_)


def _build_difference_mask_tensor() -> torch.Tensor:
    """Build the 6D tensor mapping codon pairs to three-bit mask codes.

    The first nucleotide position is the high bit and the third is the low
    bit, so masks conventionally written ``110``, ``011``, and ``101`` have
    integer codes 6, 3, and 5 respectively.

    Returns:
        Integer tensor of shape (4, 4, 4, 4, 4, 4) with values 0-7.
    """
    differences = _build_pairwise_differences()
    bit_weights = np.asarray((4, 2, 1), dtype=np.int64)
    return torch.tensor(np.sum(differences * bit_weights, axis=-1)).int()


def _build_hit_class_tensor() -> torch.Tensor:
    """Build the 6D tensor mapping parent/child codons to hit classes.

    Returns:
        Tensor of shape (4, 4, 4, 4, 4, 4) where each element is the
        number of nucleotide differences (0-3) between parent and child codons.
        The first three dimensions represent the parent codon, the last three
        represent the child codon. Codons are identified by triples of
        nucleotide indices: 0=A, 1=C, 2=G, 3=T.
    """
    return torch.tensor(np.sum(_build_pairwise_differences(), axis=-1)).int()


difference_mask_tensor: torch.Tensor = _build_difference_mask_tensor()
hit_class_tensor: torch.Tensor = _build_hit_class_tensor()

# Experiment enumeration loops query these millions of times. Keep a Python
# lookup derived from the shared tensor to avoid scalar tensor indexing per pair.
_POSITION_MASK_STRINGS: tuple[tuple[str, ...], ...] = tuple(
    tuple(f"{code:03b}" for code in row)
    for row in difference_mask_tensor.reshape(64, 64).tolist()
)


def position_mask(parent: str, child: str) -> str:
    """Return a codon pair's three-bit endpoint mask in nucleotide order.

    Both codons must contain exactly three uppercase ACGT bases; stop codons
    are allowed. The first position is the high bit, so adjacent two-hit
    changes are ``110`` or ``011``, gapped changes are ``101``, and identical
    codons give ``000``.

    Raises:
        ValueError: If either input is not a concrete ACGT codon.
    """
    try:
        return _POSITION_MASK_STRINGS[CODON_TO_IDX[parent]][CODON_TO_IDX[child]]
    except KeyError as error:
        raise ValueError(
            f"Expected three-base uppercase ACGT codons, got {parent!r} and {child!r}"
        ) from error


def parent_specific_difference_masks(parent_codon_idxs: torch.Tensor) -> torch.Tensor:
    """Produce position-mask codes for every possible child per parent.

    Args:
        parent_codon_idxs: A (codon_count, 3) shaped tensor containing the
            parent codons' nucleotide indices.

    Returns:
        An integer (codon_count, 4, 4, 4) tensor with mask codes 0-7. The
        first codon position is the high bit.
    """
    return difference_mask_tensor[
        parent_codon_idxs[:, 0], parent_codon_idxs[:, 1], parent_codon_idxs[:, 2]
    ]


def parent_specific_hit_classes(parent_codon_idxs: torch.Tensor) -> torch.Tensor:
    """Produce tensor of hit classes for all possible child codons per parent.

    For each parent codon, returns the hit class (0-3) to every possible child
    codon. The hit class is the number of nucleotide positions that differ
    between parent and child.

    Args:
        parent_codon_idxs: A (codon_count, 3) shaped tensor containing for each
            codon, the indices of the parent codon's nucleotides (0-3 each).

    Returns:
        A (codon_count, 4, 4, 4) shaped tensor containing the hit classes of
        each possible child codon for each parent codon.
    """
    return hit_class_tensor[
        parent_codon_idxs[:, 0], parent_codon_idxs[:, 1], parent_codon_idxs[:, 2]
    ]


def apply_multihit_correction(
    parent_codon_idxs: torch.Tensor,
    codon_probs: torch.Tensor,
    log_hit_class_factors: torch.Tensor,
) -> torch.Tensor:
    """Multiply codon probabilities by their hit class factors.

    This applies learned correction factors based on the number of nucleotide
    differences (hit class) between parent and child codons. Higher hit classes
    typically have different rates than would be predicted by independent
    single-nucleotide mutations.

    Args:
        parent_codon_idxs: A (N, 3) shaped tensor containing for each codon,
            the indices of the parent codon's nucleotides (0-3 each).
        codon_probs: A (N, 4, 4, 4) shaped tensor containing the probabilities
            of mutating to each possible target codon, for each of the N
            parent codons.
        log_hit_class_factors: A tensor of length 3 containing the log hit class
            factors for hit classes 1, 2, and 3. The factor for hit class 0 is
            assumed to be 1 (i.e., 0 in log-space).

    Returns:
        A (N, 4, 4, 4) shaped tensor containing the probabilities of mutating
        to each possible target codon, after applying the hit class factors.
        Note: Does NOT renormalize probabilities (caller's responsibility).
    """
    per_parent_hit_class = parent_specific_hit_classes(parent_codon_idxs)
    # Prepend 0.0 for hit class 0 factor (exp(0) = 1), then exponentiate
    zero_factor = torch.zeros(1, device=log_hit_class_factors.device)
    corrections = torch.cat([zero_factor, log_hit_class_factors]).exp()
    per_codon_corrections = corrections[per_parent_hit_class]
    return codon_probs * per_codon_corrections


# =============================================================================
# HitClassModel (ported from netam.models)
# =============================================================================


class HitClassModel(nn.Module):
    """A model for applying hit class corrections to codon probabilities.

    This model wraps learned log-space factors for hit classes 1, 2, and 3.
    Hit class 0 (no mutation) implicitly has factor 1.0 (log factor 0.0).

    The model can be loaded from pretrained weights or created from scratch.
    It's used to adjust neutral codon probabilities to account for multi-hit
    effects that deviate from the independent mutation assumption.

    Attributes:
        log_hit_class_factors: nn.Parameter of shape (3,) containing log factors
            for hit classes 1, 2, and 3.
    """

    def __init__(self) -> None:
        """Initialize with default zero log factors (no correction)."""
        super().__init__()
        self.reinitialize_weights()

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """Return hyperparameters for model serialization."""
        return {}

    @property
    def values(self) -> nn.Parameter:
        """Alias for log_hit_class_factors for netam compatibility."""
        return self.log_hit_class_factors

    def forward(
        self, parent_codon_idxs: torch.Tensor, uncorrected_codon_probs: torch.Tensor
    ) -> torch.Tensor:
        """Apply hit class corrections with clamping and renormalization.

        Takes a tensor of target codon distributions, for each observed parent
        codon, and adjusts the distributions according to the hit class corrections.

        Inputs should be in unflattened form, with codons represented in shape
        (4, 4, 4).

        Args:
            parent_codon_idxs: A (N, 3) shaped tensor of parent codon nucleotide indices.
            uncorrected_codon_probs: A (N, 4, 4, 4) shaped tensor of codon probabilities.

        Returns:
            Corrected and renormalized codon probabilities of shape (N, 4, 4, 4).
        """
        corrected = self.apply_multihit_correction(
            parent_codon_idxs, uncorrected_codon_probs
        )
        return self._normalize_corrected_probs(corrected, parent_codon_idxs)

    def _normalize_corrected_probs(
        self,
        corrected_probs: torch.Tensor,
        parent_codon_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize and clamp corrected probabilities.

        First clamps above only to prevent small probability accumulation when
        computing parent codon probability, then recomputes parent codon
        probability to ensure probabilities sum to 1, then fully clamps.

        Args:
            corrected_probs: A (N, 4, 4, 4) shaped tensor of corrected probabilities.
            parent_codon_idxs: A (N, 3) shaped tensor of parent codon nucleotide indices.

        Returns:
            Normalized and clamped probabilities of shape (N, 4, 4, 4).
        """
        clamped_above = clamp_probability_above_only(corrected_probs)
        normalized = set_parent_codon_prob(
            flatten_codons(clamped_above),
            flatten_codon_idxs(parent_codon_idxs),
        )
        fully_clamped = clamp_probability(normalized)
        return unflatten_codons(fully_clamped)

    def apply_multihit_correction(
        self, parent_codon_idxs: torch.Tensor, uncorrected_codon_probs: torch.Tensor
    ) -> torch.Tensor:
        """Apply the correction to the uncorrected codon probabilities.

        Unlike `forward` this does not clamp or recompute parent codon probability.
        Otherwise, it is identical to `forward`.

        Args:
            parent_codon_idxs: A (N, 3) shaped tensor of parent codon nucleotide indices.
            uncorrected_codon_probs: A (N, 4, 4, 4) shaped tensor of codon probabilities.

        Returns:
            Corrected codon probabilities of same shape.
        """
        return apply_multihit_correction(
            parent_codon_idxs, uncorrected_codon_probs, self.log_hit_class_factors
        )

    def reinitialize_weights(
        self, weights: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> None:
        """Reinitialize the model weights.

        Args:
            weights: Tuple of 3 floats for log hit class factors.
        """
        self.log_hit_class_factors = nn.Parameter(torch.tensor(weights))

    def to_weights(self) -> tuple[float, float, float]:
        """Export model weights as a tuple.

        Returns:
            Tuple of 3 floats representing log hit class factors.
        """
        values = self.log_hit_class_factors.detach().cpu().numpy()
        return (float(values[0]), float(values[1]), float(values[2]))

    @classmethod
    def from_weights(cls, weights: tuple[float, float, float]) -> "HitClassModel":
        """Create a HitClassModel from weight values.

        Args:
            weights: Tuple of 3 floats for log hit class factors.

        Returns:
            Initialized HitClassModel in eval mode (disables training behavior).

        Raises:
            ValueError: If weights does not contain exactly 3 values.
        """
        if len(weights) != 3:
            raise ValueError(
                f"Expected 3 hit class factors, got {len(weights)}. "
                "Weights must be (factor_1, factor_2, factor_3) for hit classes 1-3."
            )
        model = cls()
        model.reinitialize_weights(weights)
        model.eval()
        return model
