"""Neutral model type enumeration."""

from enum import Enum


class NeutralModelType(str, Enum):
    """Enumeration of available neutral model types.

    This enum serves as the type discriminator for neutral model configurations.
    The enum values match the model class names for clarity.

    Examples:
        >>> # In YAML config
        >>> neutral_model:
        >>>   type: ThriftySHM
        >>>   model_name: ThriftyHumV0.2-59
        >>>
        >>> # In Python code
        >>> config = ThriftySHMNeutralModelConfig(model_name="ThriftyHumV0.2-59")
    """

    THRIFTY_SHM = "ThriftySHM"
    TABULAR_RATES = "TabularRates"
    GTR_CODON = "GTRCodon"
    PER_FAMILY_GTR = "PerFamilyGTR"
    SHM_CODON = "ShmCodon"
    S5F = "S5F"
    # Deliberately not "None"/"Identity": this model computes no rates at all
    # and raises if asked to, rather than standing in for a rate-1 model. See
    # dasm2/neutral/precomputed_rates.py.
    PRECOMPUTED_RATES = "PrecomputedRates"
