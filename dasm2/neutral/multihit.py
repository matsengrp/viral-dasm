"""Multihit correction wrapper for neutral models.

This module provides MultihitWrapper, which applies multihit correction to
rates/CSPs-based neutral models by composing a base neutral model with a HitClassModel.

For hit class computation functions and HitClassModel itself, see:
    dasm2.neutral.hit_class
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import torch

from dasm2.molevol.neutral_evolution import neutral_codon_probs_of_seq
from dasm2.neutral.base import (
    DelegatedSourceCodonLookup,
    RatesCSPNeutralModel,
    SourceCodonFeatureLookup,
)
from dasm2.neutral.hit_class import HitClassModel
from dasm2.sequence.conversion import codon_idx_tensor_of_str_ambig

if TYPE_CHECKING:
    from dasm2.config import NeutralModelConfig


class MultihitWrapper(RatesCSPNeutralModel):
    """Wrapper that applies multihit correction to rates/CSPs-based neutral models.

    This wrapper composes a base RatesCSPNeutralModel with a multihit correction model.
    Only models that use the rates/CSPs decomposition (ThriftySHM, TabularRates, S5F,
    PrecomputedRates) can be wrapped - matrix exponentiation models (GTRCodon) handle
    multi-hit internally.

    The wrapper delegates rate/CSP computation to the base model and applies
    multihit correction in compute_neutral_codon_probs().

    Example:
        >>> from dasm2.model_catalog import resolve_model
        >>> base_model = ThriftySHMModel.from_catalog(
        ...     "ThriftyHumV0.2-59", resolve_model("ThriftyHumV0.2-59")
        ... )
        >>> multihit_model = HitClassModel.from_weights((-0.16, 0.07, 0.51))
        >>> model = MultihitWrapper(base_model, multihit_model)
    """

    def __init__(
        self,
        base_model: RatesCSPNeutralModel,
        multihit_model: HitClassModel,
        multihit_model_name: Optional[str] = None,
    ):
        """Wrap a rates/CSPs neutral model with multihit correction.

        Args:
            base_model: The underlying neutral model (must use rates/CSPs decomposition)
            multihit_model: HitClassModel for correction
            multihit_model_name: Name of the multihit model (for serialization)
        """
        self._base_model: RatesCSPNeutralModel = base_model
        # Public attribute to match AbstractNeutralModel.multihit_model interface
        self.multihit_model: Optional[HitClassModel] = multihit_model
        self._multihit_model_name: Optional[str] = multihit_model_name

    @property
    def base_model(self) -> RatesCSPNeutralModel:
        """Underlying model whose nucleotide rates are preserved by this wrapper."""
        return self._base_model

    @property
    def name(self) -> str:
        """Human-readable identifier including multihit status."""
        return f"{self._base_model.name}-multihit"

    @property
    def computes_rates(self) -> bool:  # type: ignore[override]
        """Whether the wrapped base model computes rates (False for the carrier)."""
        return self._base_model.computes_rates

    @property
    def required_context(self) -> Tuple[str, ...]:  # type: ignore[override]
        """The wrapped base model's per-row context keys."""
        return self._base_model.required_context

    def to_config(self) -> "NeutralModelConfig":
        """Reconstruct the base model's config with the multihit model name set.

        Delegates to the base model's ``to_config()`` so catalog-name validation
        lives in one place (it raises if the base model's name is not a
        resolvable catalog entry), then records the multihit model name. Works
        for the base model types this wrapper supports (see class docstring):
        ``ThriftySHMNeutralModelConfig``, ``TabularRatesNeutralModelConfig``,
        ``S5FNeutralModelConfig``, and ``PrecomputedRatesNeutralModelConfig``.

        Returns:
            The base model's config type, with ``multihit_model_name`` set.

        Raises:
            ValueError: If the base model cannot produce a portable config.
            TypeError: If the base model's config type doesn't support
                multihit correction (i.e. isn't a
                ``MultihitWrappableNeutralModelConfig``).
        """
        # Imported here (not at module level) to avoid a circular dependency
        # with dasm2.config, which imports this module for the factory.
        from dasm2.config import MultihitWrappableNeutralModelConfig

        config = self._base_model.to_config()
        if not isinstance(config, MultihitWrappableNeutralModelConfig):
            raise TypeError(
                "MultihitWrapper.to_config() requires a "
                "MultihitWrappableNeutralModelConfig-based model (ThriftySHM, "
                "TabularRates, S5F, or PrecomputedRates), but the base "
                f"produced a {type(config).__name__}."
            )
        config.multihit_model_name = self._multihit_model_name
        # The live model's actual factors, not just the pretrained-registry
        # value named above — takes precedence at reload (issue #501 trap 5),
        # so a refit model round-trips instead of reverting to the pretrained
        # triple.
        if self.multihit_model is not None:
            config.multihit_log_factors = self.multihit_model.to_weights()
        return config

    @classmethod
    def from_pretrained(
        cls,
        base_model: RatesCSPNeutralModel,
        multihit_model_name: str,
    ) -> "MultihitWrapper":
        """Load multihit model from pretrained registry and wrap base model.

        Args:
            base_model: The underlying neutral model (must use rates/CSPs decomposition)
            multihit_model_name: Name of pretrained multihit model
                (e.g., "ThriftyHumV0.2-59-hc-tangshm")

        Returns:
            MultihitWrapper with loaded multihit model

        Raises:
            ValueError: If multihit_model_name not found in registry
        """
        from dasm2.pretrained import load_multihit  # noqa: PLC0415

        loaded_multihit_model = load_multihit(multihit_model_name)
        return cls(base_model, loaded_multihit_model, multihit_model_name)

    def compute_rates_and_csps(
        self,
        sequences: Union[str, List[str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Delegate rate/CSP computation to base model.

        Args:
            sequences: Single sequence string or list of sequences
            context: Reserved for future use. Passed through to base model.

        Returns:
            Tuple of (nt_rates_list, nt_csps_list) from base model
        """
        return self._base_model.compute_rates_and_csps(sequences, context)

    def validate_source_coverage(self, parents: Iterable[str], label: str) -> None:
        """Delegate table coverage; multihit preserves nucleotide features."""
        self.base_model.validate_source_coverage(parents, label)

    def source_codon_lookup(self) -> SourceCodonFeatureLookup:
        """Retain the base lookup's frozen/oracle policy through wrapping."""
        return DelegatedSourceCodonLookup(self, self.base_model.source_codon_lookup())

    def set_boundary_table_path(self, path: str) -> None:
        """Select the wrapped model's custom table before coverage validation."""
        self.base_model.set_boundary_table_path(path)

    def source_codon_context_key(self, sequence: str, codon_start: int) -> Hashable:
        """Reuse the base model's context key; multihit leaves nt features intact."""
        return self._base_model.source_codon_context_key(sequence, codon_start)

    def compute_rates_and_csps_vectorized(
        self,
        sequences: Union[str, List[str]],
        context: Optional[Dict[str, Any]] = None,
        batch_size: int = 4096,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Delegate vectorized rate/CSP computation to base model.

        Without this override, the base class default falls back to the
        non-vectorized ``compute_rates_and_csps``, losing the ~100x encoding
        speedup that ThriftySHMModel provides.

        Args:
            sequences: Single sequence string or list of sequences.
            context: Reserved for future use. Passed through to base model.
            batch_size: Maximum sequences to process at once.

        Returns:
            Tuple of (nt_rates_list, nt_csps_list) from base model.
        """
        return self._base_model.compute_rates_and_csps_vectorized(
            sequences, context, batch_size
        )

    def compute_neutral_codon_probs(
        self,
        sequence: str,
        mask: torch.Tensor,
        branch_length: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute neutral codon probabilities with multihit correction.

        Args:
            sequence: Nucleotide sequence string
            mask: Boolean mask of shape (seq_length // 3,) for valid codons
            branch_length: Evolutionary time for probability calculation
            context: Reserved for future use. Passed through to base model.

        Returns:
            Corrected codon probabilities of shape (seq_length // 3, 64)
        """
        # Get rates and CSPs from base model
        nt_rates_list, nt_csps_list = self.compute_rates_and_csps(sequence, context)
        nt_rates, nt_csps = self._unpack_single_sequence(nt_rates_list, nt_csps_list)

        # Convert nucleotide sequence to codon indices
        codon_idxs = codon_idx_tensor_of_str_ambig(sequence)

        # Use neutral codon probs with multihit model injected
        return neutral_codon_probs_of_seq(
            codon_idxs,
            mask,
            nt_rates,
            nt_csps,
            branch_length,
            multihit_model=self.multihit_model,
        )
