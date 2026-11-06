"""Resolve the global k-mer lookup consumed by Q-family losses.

The loss owns one generic ``kmer_table`` buffer. The selected ``q_variant``
determines how that buffer is produced: a provenance-checked 13-mer artifact
for Thrifty, or the finite-table capability exposed by the configured S5F
neutral model. Keeping both routes here prevents the QDASM and Hybrid losses
from growing parallel model-specific resolution code.
"""

from typing import Optional

from torch import Tensor

from dasm2.molevol.q_matrix import GLOBAL_KMER_Q_VARIANTS, QVariant
from dasm2.neutral.base import KmerTableNeutralModel


def resolve_global_kmer_table(
    q_variant: QVariant,
    kmer_table: Optional[Tensor],
    *,
    thrifty_13mer_table_path: Optional[str] = None,
    expected_model_name: Optional[str] = None,
    kmer_model_name: Optional[str] = None,
) -> Optional[Tensor]:
    """Return the lookup table required by a global-k-mer Q variant.

    ``kmer_table`` is an in-memory test input. Production ``global_13mer``
    loads ``thrifty_13mer_table_path`` and verifies both model-name and SHA256
    provenance. Production ``global_5mer`` reloads ``kmer_model_name`` through
    the neutral-model factory and consumes only the
    :class:`~dasm2.neutral.base.KmerTableNeutralModel` capability.
    """
    if q_variant not in GLOBAL_KMER_Q_VARIANTS:
        return None

    production_specs = [
        value
        for value in (
            thrifty_13mer_table_path,
            expected_model_name,
            kmer_model_name,
        )
        if value is not None
    ]
    if kmer_table is not None:
        if production_specs:
            raise ValueError(
                "Pass either kmer_table (in-memory, test-only) or the "
                "production model/table fields, not both."
            )
        return kmer_table

    if q_variant == "global_13mer":
        if kmer_model_name is not None:
            raise ValueError(
                "q_variant='global_13mer' does not use kmer_model_name; pass "
                "thrifty_13mer_table_path and expected_model_name."
            )
        if thrifty_13mer_table_path is None:
            raise ValueError(
                "q_variant='global_13mer' requires thrifty_13mer_table_path "
                "(production) or kmer_table (tests)"
            )
        from dasm2.neutral import thrifty_lookup

        if expected_model_name is not None:
            metadata = thrifty_lookup.load_thrifty_13mer_table_metadata(
                thrifty_13mer_table_path
            )
            source_model_name = metadata["model_name"]
            if source_model_name != expected_model_name:
                raise ValueError(
                    f"Thrifty 13-mer table source model='{source_model_name}' "
                    f"does not match configured neutral model="
                    f"'{expected_model_name}'. The table drives every interior "
                    "row of Q, so this pairing would silently train under a "
                    "different mutation model."
                )
        return thrifty_lookup.load_thrifty_13mer_table(thrifty_13mer_table_path)

    if thrifty_13mer_table_path is not None or expected_model_name is not None:
        raise ValueError(
            "q_variant='global_5mer' does not use Thrifty table fields; pass "
            "kmer_model_name."
        )
    if kmer_model_name is None:
        raise ValueError(
            "q_variant='global_5mer' requires kmer_model_name (production) "
            "or kmer_table (tests). TrainingConfig normally fills the former "
            "from neutral_model.model_name."
        )

    from dasm2.config import S5FNeutralModelConfig, create_neutral_model_from_config

    model = create_neutral_model_from_config(
        S5FNeutralModelConfig(model_name=kmer_model_name)
    )
    if not isinstance(model, KmerTableNeutralModel):
        raise TypeError(
            f"kmer_model_name='{kmer_model_name}' resolved to a "
            f"{type(model).__name__}, which does not implement the "
            "KmerTableNeutralModel capability."
        )
    if model.kmer_length != 5:
        raise ValueError(
            f"q_variant='global_5mer' requires kmer_length=5, but "
            f"'{kmer_model_name}' reports {model.kmer_length}."
        )
    return model.kmer_table()
