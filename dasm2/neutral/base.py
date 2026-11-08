"""Base classes for neutral evolutionary models.

Key Design Requirements Implemented:
- Two-level abstraction: AbstractNeutralModel (base) + RatesCSPNeutralModel (rates/CSPs)
- Support for both rates/CSPs decomposition and matrix exponentiation approaches
- Universal sequence + context → probabilities interface

Two Evolutionary Frameworks:
- **Rates/CSPs decomposition** (RatesCSPNeutralModel): Per-nucleotide rates × CSPs,
  Poisson approximation, multihit via MultihitWrapper. Used by ThriftySHM, TabularRates,
  and GTRCodon.
- **Matrix exponentiation**: P(t) = exp(Qt), full 61×61 codon matrix, built-in
  multi-hit handling. GTRCodon extends RatesCSPNeutralModel and additionally supports
  this via ``compute_neutral_codon_probs()`` override.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    NoReturn,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import torch
from typing_extensions import Self

from dasm2.molevol.neutral_evolution import neutral_codon_probs_of_seq
from dasm2.sequence.conversion import codon_idx_tensor_of_str_ambig

if TYPE_CHECKING:
    from dasm2.molevol.tandem import TandemTerm
    from dasm2.neutral.hit_class import HitClassModel


@runtime_checkable
class KmerTableNeutralModel(Protocol):
    """Capability for neutral models with a finite odd k-mer receptive field.

    Q-family losses use this structural interface instead of reaching into a
    concrete neutral-model implementation. The returned table has one row per
    concrete A/C/G/T k-mer and columns ``(rate, csp_A, csp_C, csp_G, csp_T)``.
    """

    @property
    def kmer_length(self) -> int:
        """Odd receptive-field width used to index :meth:`kmer_table`."""
        ...

    def kmer_table(self) -> torch.Tensor:
        """Return a ``[4 ** kmer_length, 5]`` rate/CSP lookup tensor."""
        ...


class SourceCodonFeatureLookup(Protocol):
    """Read-only source features bound to the model defining their semantics."""

    @property
    def neutral_model(self) -> "RatesCSPNeutralModel":
        """Model whose source features this lookup returns."""
        ...

    def get(self, segment: str, start: int) -> torch.Tensor:
        """Return [61, 3, 5] unadjusted source features for one context."""
        ...


class DelegatedSourceCodonLookup:
    """Bind an unchanged base lookup to a nucleotide-preserving wrapper."""

    def __init__(
        self, neutral_model: "RatesCSPNeutralModel", lookup: SourceCodonFeatureLookup
    ) -> None:
        self.neutral_model = neutral_model
        self.lookup = lookup

    def get(self, segment: str, start: int) -> torch.Tensor:
        """Return the base model's source features without copying."""
        return self.lookup.get(segment, start)


class AbstractNeutralModel(ABC):
    """Base class for all neutral models: produces codon probabilities.

    All neutral models must implement compute_neutral_codon_probs() which converts
    a nucleotide sequence to codon transition probabilities.

    This base class is directly used by models that compute probabilities via
    matrix exponentiation (e.g., GTRCodonNeutralModel). Models that use the
    rates/CSPs decomposition should instead inherit from RatesCSPNeutralModel.

    Attributes:
        name: Human-readable identifier for the model (required for all implementations).
        multihit_model: Optional multihit correction model. Subclasses that support
            multihit correction should set this attribute. Default is None.
    """

    # Default: no multihit correction. Subclasses override as needed.
    multihit_model: Optional["HitClassModel"] = None
    # Default: no tandem term. Set only on PerFamilyGTR, from
    # its config; every neutral-probability consumer reads it from here.
    tandem: Optional["TandemTerm"] = None
    # Set by from_catalog() for portable config round-tripping.
    _catalog_name: Optional[str] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this neutral model.

        Used for metadata tracking and logging. Must be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def to_config(self) -> NoReturn:
        """Reconstruct this model's config -- unsupported in this fork.

        The neutral-model config classes went with the prune, because nothing
        here builds one: the corpus carries `nt_rates`/`nt_csps` in the
        Parquet. Implementations raise rather than return.
        """
        pass

    def set_boundary_table_path(self, path: str) -> None:
        """Select a custom source table, rejecting models without this capability."""
        raise ValueError(f"boundary_table_path is unsupported by {type(self).__name__}")

    @classmethod
    def from_catalog(cls, catalog_name: str, archive_path: Union[str, Path]) -> Self:
        """Load from a catalog-resolved archive, recording the catalog name.

        Calls ``from_pretrained(archive_path)`` and sets ``_catalog_name`` so
        that ``to_config()`` can round-trip portably.

        Archive-backed models (``TabularRatesNeutralModel``,
        ``GTRCodonNeutralModel``, ``ThriftySHMModel``) support this. The subclass
        must implement ``from_pretrained(model_path: Union[str, Path]) -> Self``
        as a classmethod taking a single archive-path argument. Models whose
        ``from_pretrained`` has a different contract (e.g. ``MultihitWrapper``,
        which wraps an already-loaded base model) are not catalog-loadable and
        must not be constructed through ``from_catalog``.

        Args:
            catalog_name: The catalog entry name (e.g., "TabularRates-SCoV2-v1").
            archive_path: Path to the zip archive resolved by the catalog.

        Returns:
            Initialized model with ``_catalog_name`` set for config round-tripping.

        Raises:
            NotImplementedError: If the subclass does not implement
                ``from_pretrained(model_path)``.
        """
        if not hasattr(cls, "from_pretrained"):
            raise NotImplementedError(
                f"{cls.__name__} must implement from_pretrained() to use from_catalog()"
            )
        model: Self = cls.from_pretrained(archive_path)  # type: ignore[attr-defined]
        model._catalog_name = catalog_name
        return model

    @abstractmethod
    def compute_neutral_codon_probs(
        self,
        sequence: str,
        mask: torch.Tensor,
        branch_length: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute neutral codon probabilities for a sequence.

        Args:
            sequence: Nucleotide sequence string
            mask: Boolean mask of shape (seq_length // 3,) for valid codons
            branch_length: Evolutionary time for probability calculation
            context: Per-row values keyed by ``required_context``. Models
                with an empty ``required_context`` ignore it.

        Returns:
            torch.Tensor: Codon probabilities of shape (seq_length // 3, 64)
        """
        pass


class RatesCSPNeutralModel(AbstractNeutralModel):
    """Neutral model using per-nucleotide rates and conditional substitution probabilities.

    This class extends AbstractNeutralModel for models that decompose neutral evolution
    into per-nucleotide rates and CSPs. It provides:

    1. Abstract compute_rates_and_csps() method for subclasses to implement
    2. Default compute_neutral_codon_probs() using Poisson approximation

    Models using this approach (ThriftySHM, TabularRates) can use MultihitWrapper
    for multihit corrections.

    The Poisson approximation assumes:
        P(mutation at site) ≈ 1 - exp(-rate × t)

    Attributes:
        computes_rates: False for a model that only carries a multihit
            correction and raises if asked for rates
            (``PrecomputedRatesNeutralModel``). A rate-free Parquet corpus
            needs a model that computes rates, and its loader checks this.
        required_context: Per-row keys the model reads from ``context``. For
            example, ``PerFamilyGTRNeutralModel`` needs ``("sample_id",)``.
            The Parquet loader carries these columns into each sample, and
            ``ComputeNtRatesAndCsps`` passes them as ``context``.
    """

    computes_rates: bool = True
    required_context: Tuple[str, ...] = ()

    @abstractmethod
    def compute_rates_and_csps(
        self,
        sequences: Union[str, List[str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute neutral rates and CSPs for one or more sequences.

        Always returns lists for consistency, even when processing a single sequence.
        This simplifies downstream code by eliminating type checking based on input type.

        Args:
            sequences: Single nucleotide sequence string OR list of sequence strings.
                      For lists, enables batched computation for better performance.
            context: Per-row values keyed by ``required_context``, e.g.
                ``{"sample_id": "fam1"}``. Models with an empty
                ``required_context`` ignore it.

        Returns:
            Tuple of (nt_rates_list, nt_csps_list) where:
                nt_rates_list: List of tensors, one per sequence, each with shape (seq_length,)
                nt_csps_list: List of tensors, one per sequence, each with shape (seq_length, 4)
        """
        pass

    def compute_rates_and_csps_vectorized(
        self,
        sequences: Union[str, List[str]],
        context: Optional[Dict[str, Any]] = None,
        batch_size: int = 4096,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute neutral rates and CSPs using vectorized encoding.

        Drop-in replacement for ``compute_rates_and_csps`` with ~100x faster
        k-mer encoding. Default implementation falls back to ``compute_rates_and_csps``.
        Subclasses with vectorized encoding support should override this method.

        Args:
            sequences: Single sequence string or list of sequences.
            context: Per-row values keyed by ``required_context``. Models
                with an empty ``required_context`` ignore it.
            batch_size: Maximum sequences to process at once.

        Returns:
            Identical to ``compute_rates_and_csps``.
        """
        return self.compute_rates_and_csps(sequences, context)

    def validate_source_coverage(self, parents: Iterable[str], label: str) -> None:
        """Validate all selected parents before global-kmer preprocessing.

        Models without an offline source table require no coverage scan.
        Wrappers must delegate this hook while preserving their rate semantics.
        """

    def source_codon_lookup(self) -> SourceCodonFeatureLookup:
        """Build a bounded full-segment oracle for models without a frozen table."""
        # The cache depends on this interface; defer the import to avoid a cycle.
        from dasm2.neutral.codon_boundary import SourceCodonContextCache

        return SourceCodonContextCache(self)

    def source_codon_context_key(self, sequence: str, codon_start: int) -> Hashable:
        """Identify inputs whose all-source codon features can safely be reused.

        The default retains the entire segment and nucleotide offset, allowing
        sequence-wide and position-dependent models without assuming locality.
        A finite-context model may override this with a smaller sufficient key.
        The cache is owned by one frozen model instance, so model identity and
        segment rate adjustments are handled outside the key.

        Args:
            sequence: Original full segment, without chain separators.
            codon_start: Nucleotide offset of the three-base source codon.

        Returns:
            Hashable key for the unadjusted features of all 61 source codons.
        """
        return sequence, codon_start

    def _unpack_single_sequence(
        self,
        rates_list: List[torch.Tensor],
        csps_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract results for single sequence, validating list length.

        Args:
            rates_list: List of rate tensors (should contain exactly one)
            csps_list: List of CSP tensors (should contain exactly one)

        Returns:
            Tuple of (rates, csps) tensors for the single sequence

        Raises:
            AssertionError: If lists don't contain exactly one element
        """
        assert len(rates_list) == 1 and len(csps_list) == 1, (
            f"Expected single sequence results, got {len(rates_list)} rate tensors "
            f"and {len(csps_list)} CSP tensors"
        )
        return rates_list[0], csps_list[0]

    def compute_neutral_codon_probs(
        self,
        sequence: str,
        mask: torch.Tensor,
        branch_length: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute neutral codon probabilities using rates/CSPs decomposition.

        Default implementation uses Poisson approximation via molevol path.
        Models with multihit corrections can override this method.

        Args:
            sequence: Nucleotide sequence string
            mask: Boolean mask of shape (seq_length // 3,) for valid codons
            branch_length: Evolutionary time for probability calculation
            context: Per-row values keyed by ``required_context``. Models
                with an empty ``required_context`` ignore it.

        Returns:
            torch.Tensor: Codon probabilities of shape (seq_length // 3, 64)
        """
        # Use rates/CSPs interface - always returns lists
        nt_rates_list, nt_csps_list = self.compute_rates_and_csps(sequence, context)
        nt_rates, nt_csps = self._unpack_single_sequence(nt_rates_list, nt_csps_list)

        # Convert nucleotide sequence to codon indices
        codon_idxs = codon_idx_tensor_of_str_ambig(sequence)

        # Use standard netam molecular evolution without multihit
        return neutral_codon_probs_of_seq(
            codon_idxs, mask, nt_rates, nt_csps, branch_length, multihit_model=None
        )
