"""Protocols for DASM2 data types.

This module defines protocols (structural typing) for data types used across
DASM2. Using protocols instead of concrete type checks (isinstance) allows
for cleaner decoupling and follows the Open/Closed principle.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class SequenceProvider(Protocol):
    """Protocol for objects that provide sequence access.

    Used to decouple ESM2DatasetWrapper from specific dataset implementations.
    Any dataset that provides amino acid sequences can implement this protocol.
    """

    @property
    def aa_parents(self) -> list[str]:
        """Return list of parent amino acid sequences."""
        ...

    @property
    def max_aa_seq_len(self) -> int:
        """Return maximum amino acid sequence length in dataset."""
        ...
