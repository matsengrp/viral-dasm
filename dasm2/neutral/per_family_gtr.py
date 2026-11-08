"""Per-family GTR neutral model, keyed on each row's ``sample_id``.

Issue #1299, PR 2. Compara fits one GTR+F model per gene family, and
per-cluster superfamily fits are the same idea. Before this model existed,
the only way to train on those rates was to bake them into each row's
``nt_rates``/``nt_csps`` columns. That stores a (family, parent nucleotide)
lookup at every site, which is 80% of the v10 Compara parquet's bytes.
``PerFamilyGTRNeutralModel`` computes the same rates at load time from a
rate-free parquet: the Parquet loader carries ``sample_id`` into each sample
(``required_context``), and ``ComputeNtRatesAndCsps`` passes it as
``context``.

Each family is stored as its GTR parameters plus the 5×5 lookup table
(rates and CSPs for A/C/G/T/N) that ``GTRCodonNeutralModel`` derives from
them. Both come from the same functions GTR uses
(``build_gtr_nt_rate_matrix``, ``nt_rate_csp_table``), so a family's rates
are identical to its standalone GTR model's. Holding a full
``GTRCodonNeutralModel`` per family would also carry its 61×61 codon
matrices: about 24 KB per family, in every DataLoader worker, which is
gigabytes at 10^5 families. A family's full model is rebuilt on demand only
for the codon-level matrix-exp path.
"""

import io
import logging
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import pandas as pd
import torch
from pydantic import BaseModel, Field

from dasm2.molevol.tandem import TandemTerm
from dasm2.neutral.base import RatesCSPNeutralModel
from dasm2.neutral.gtr_codon import (
    GTRCodonNeutralModel,
    build_gtr_nt_rate_matrix,
    lookup_nt_rates_and_csps,
    nt_rate_csp_table,
    validate_gtr_parameters,
)
from dasm2.serialization.zip_io import (
    data_archive_exists,
    read_data_archive,
    write_data_archive,
)

if TYPE_CHECKING:
    from dasm2.config import PerFamilyGTRNeutralModelConfig

logger = logging.getLogger(__name__)

FAMILY_KEY = "sample_id"
"""The per-row context key a family is looked up by."""

GTR_EXCHANGEABILITY_COLUMNS = ("AC", "AG", "AT", "CG", "CT", "GT")
"""Exchangeability columns of a GTR results table, in ``EXCHANGEABILITY_PAIRS`` order."""

GTR_FREQUENCY_COLUMNS = ("freq_A", "freq_C", "freq_G", "freq_T")
"""Equilibrium-frequency columns of a GTR results table."""

_FAMILIES_FILENAME = "families.tsv"


class PerFamilyGTRArchiveMetadata(BaseModel):
    """Metadata for a ``PerFamilyGTRNeutralModel`` archive (``metadata.yml``)."""

    model_type: str = Field(
        default="PerFamilyGTRNeutralModel",
        description="Model class identifier for validation",
    )
    model_name: str = Field(
        default="PerFamilyGTR", description="Human-readable model name"
    )
    archive_version: int = Field(
        default=1, description="Archive format version for future compatibility"
    )
    num_families: int = Field(description="Number of family models in the archive")


def read_gtr_results_table(source: Union[str, Path, IO[bytes]]) -> pd.DataFrame:
    """Read a GTR results TSV, keeping ``family_id`` exactly as written.

    ``family_id`` is read as a string with no NA parsing, so an ID like
    ``"007"`` or ``"NA"`` survives. Keys must match the packaged
    ``sample_id``, which the Compara packager also reads as a string.

    Args:
        source: A path, or a binary buffer such as an archive member.
    """
    return pd.read_csv(
        source, sep="\t", dtype={"family_id": str}, keep_default_na=False
    )


def _read_parameter_table(
    gtr_df: pd.DataFrame,
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """Read family keys, exchangeabilities (F, 6) and frequencies (F, 4).

    Raises:
        ValueError: If a required column is missing, or a family appears twice.
    """
    required = ["family_id", *GTR_EXCHANGEABILITY_COLUMNS, *GTR_FREQUENCY_COLUMNS]
    missing = [c for c in required if c not in gtr_df.columns]
    if missing:
        raise ValueError(
            f"GTR results table is missing column(s) {missing}; "
            f"columns: {list(gtr_df.columns)}"
        )
    duplicated = gtr_df["family_id"][gtr_df["family_id"].duplicated()].tolist()
    if duplicated:
        raise ValueError(f"GTR results table repeats family_id(s) {duplicated[:5]}")
    exchangeabilities = torch.tensor(
        gtr_df[list(GTR_EXCHANGEABILITY_COLUMNS)].to_numpy(), dtype=torch.float32
    )
    frequencies = torch.tensor(
        gtr_df[list(GTR_FREQUENCY_COLUMNS)].to_numpy(), dtype=torch.float32
    )
    return [str(f) for f in gtr_df["family_id"]], exchangeabilities, frequencies


def gtr_parameters_from_results(
    gtr_df: pd.DataFrame,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Read per-family GTR parameters from a GTR results table.

    Frequencies are renormalized to sum to exactly 1.0, to absorb the
    rounding in the TSV.

    Args:
        gtr_df: One row per family, with a ``family_id`` column plus
            ``GTR_EXCHANGEABILITY_COLUMNS`` and ``GTR_FREQUENCY_COLUMNS``
            (Compara's ``gtr_results_all.tsv`` layout).

    Returns:
        ``family_id`` -> ``(exchangeabilities (6,), frequencies (4,))``.

    Raises:
        ValueError: If a required column is missing, or a family appears twice.
    """
    keys, exchangeabilities, frequencies = _read_parameter_table(gtr_df)
    frequencies = frequencies / frequencies.sum(dim=1, keepdim=True)
    return {key: (exchangeabilities[i], frequencies[i]) for i, key in enumerate(keys)}


class PerFamilyGTRNeutralModel(RatesCSPNeutralModel):
    """One GTR model per family, chosen by each row's ``sample_id``.

    ``compute_rates_and_csps(seq, context)`` uses the family named by
    ``context["sample_id"]``, which must match the packaged ``sample_id``
    exactly, including ``load_pcps``'s ``label::`` prefix on multi-source
    corpora. A missing key raises.

    Not multihit-wrappable, like ``GTRCodonNeutralModel`` (issue #1299,
    Open question 1). It can carry a fixed tandem term (``tandem``, issue
    #1683), set from its config's ``tandem_delta``/``tandem_delta13`` and
    emitted back by ``to_config``.

    Args:
        parameters: ``family key -> (exchangeabilities (6,), frequencies (4,))``.
        name: Human-readable model identifier.

    Raises:
        ValueError: If ``parameters`` is empty or a family's parameters are
            invalid (see ``validate_gtr_parameters``).
    """

    required_context = (FAMILY_KEY,)

    def __init__(
        self,
        parameters: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
        *,
        name: str = "PerFamilyGTR",
    ) -> None:
        if not parameters:
            raise ValueError("PerFamilyGTRNeutralModel needs at least one family")
        self._name = name
        self._keys: List[str] = list(parameters)
        self._index: Dict[str, int] = {key: i for i, key in enumerate(self._keys)}
        exchangeabilities: List[torch.Tensor] = []
        frequencies: List[torch.Tensor] = []
        rates: List[torch.Tensor] = []
        csps: List[torch.Tensor] = []
        for key, (exch, freq) in parameters.items():
            try:
                validate_gtr_parameters(exch, freq)
            except ValueError as e:
                raise ValueError(f"Family {key!r}: {e}") from e
            family_rates, family_csps = nt_rate_csp_table(
                build_gtr_nt_rate_matrix(exch, freq)
            )
            exchangeabilities.append(exch)
            frequencies.append(freq)
            rates.append(family_rates)
            csps.append(family_csps)
        self._exchangeabilities = torch.stack(exchangeabilities)  # (F, 6)
        self._frequencies = torch.stack(frequencies)  # (F, 4)
        self._rates = torch.stack(rates)  # (F, 5)
        self._csps = torch.stack(csps)  # (F, 5, 4)
        self._archive_path: Optional[Path] = None  # Set by from_pretrained()

    @property
    def name(self) -> str:
        """Human-readable identifier for this neutral model."""
        return self._name

    @property
    def family_keys(self) -> List[str]:
        """The family keys, in storage order."""
        return list(self._keys)

    def _family_index(self, context: Optional[Dict[str, Any]]) -> int:
        """Resolve ``context["sample_id"]`` to a family's storage index.

        Raises:
            ValueError: If ``context`` has no ``sample_id``, or no family has
                that key.
        """
        if context is None or FAMILY_KEY not in context:
            raise ValueError(
                f"{self._name} needs context[{FAMILY_KEY!r}] to choose a "
                f"family model, but got context={context!r}. It is wired only "
                "on the Parquet path, whose loader carries sample_id into each "
                "sample; the CSV path does not pass context."
            )
        key = str(context[FAMILY_KEY])
        index = self._index.get(key)
        if index is None:
            raise ValueError(
                f"{self._name} has no model for {FAMILY_KEY} {key!r} "
                f"({len(self._keys)} families, e.g. {self._keys[:3]}). Keys must "
                "match the packaged sample_id exactly, including any "
                "'label::' prefix."
            )
        return index

    def family_model(self, key: str) -> GTRCodonNeutralModel:
        """Build one family's full ``GTRCodonNeutralModel``.

        Args:
            key: The family key.

        Returns:
            A standalone model with that family's parameters.
        """
        index = self._family_index({FAMILY_KEY: key})
        return GTRCodonNeutralModel(
            self._exchangeabilities[index], self._frequencies[index], name=key
        )

    def compute_rates_and_csps(
        self,
        sequences: Union[str, List[str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute per-site rates and CSPs with one family's GTR model.

        Args:
            sequences: A nucleotide sequence or list of them, all from the
                family ``context["sample_id"]`` names.
            context: Must hold ``sample_id``.

        Returns:
            Tuple of (nt_rates_list, nt_csps_list), identical to what that
            family's ``GTRCodonNeutralModel.compute_rates_and_csps`` returns.

        Raises:
            ValueError: If ``context`` does not name a known family.
        """
        index = self._family_index(context)
        return lookup_nt_rates_and_csps(
            sequences, self._rates[index], self._csps[index]
        )

    def compute_neutral_codon_probs(
        self,
        sequence: str,
        mask: torch.Tensor,
        branch_length: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Matrix-exp codon probabilities under the row's family model.

        Delegates to that family's ``GTRCodonNeutralModel``, built on demand.

        Raises:
            ValueError: If ``context`` does not name a known family, or the
                model carries a tandem term, which this path does not apply.
        """
        if self.tandem is not None:
            raise ValueError(
                f"{self._name}'s matrix-exp path does not apply its tandem term "
                "(issue #1683); use the product-of-three path."
            )
        index = self._family_index(context)
        return self.family_model(self._keys[index]).compute_neutral_codon_probs(
            sequence, mask, branch_length
        )

    def set_tandem(self, tandem: Optional[TandemTerm]) -> None:
        """Set the fixed tandem term (issue #1683), logging it when on.

        A term with both rates zero is stored as None, so an off config
        leaves every consumer on the shipped code path.
        """
        self.tandem = tandem if tandem is not None and tandem.is_active else None
        if self.tandem is not None:
            logger.info(
                "%s: tandem term on, delta=%g, delta13=%g",
                self._name,
                self.tandem.delta,
                self.tandem.delta13,
            )

    def to_config(self) -> "PerFamilyGTRNeutralModelConfig":
        """Reconstruct the config: the catalog name if loaded by name, else the path.

        Raises:
            ValueError: If the model was built in memory, since then no config
                can rebuild it. Save it with ``save_pretrained`` and load it
                by path or catalog name.
        """
        from dasm2.config import PerFamilyGTRNeutralModelConfig  # noqa: PLC0415

        tandem = self.tandem or TandemTerm(0.0)
        if self._catalog_name is not None:
            return PerFamilyGTRNeutralModelConfig(
                model_name=self._catalog_name,
                tandem_delta=tandem.delta,
                tandem_delta13=tandem.delta13,
            )
        if self._archive_path is not None:
            logger.info(
                "PerFamilyGTRNeutralModel.to_config() is returning a "
                "non-portable model_path (%s). Use a catalog model_name for "
                "checkpoints that move across machines.",
                self._archive_path,
            )
            return PerFamilyGTRNeutralModelConfig(
                model_path=self._archive_path,
                tandem_delta=tandem.delta,
                tandem_delta13=tandem.delta13,
            )
        raise ValueError(
            f"{self._name} was built in memory, so no config can rebuild it. "
            "Write it with save_pretrained() and configure it by model_path "
            "or catalog model_name."
        )

    @classmethod
    def from_gtr_results(
        cls, gtr_path: Union[str, Path], *, name: str = "PerFamilyGTR"
    ) -> "PerFamilyGTRNeutralModel":
        """Build from a GTR results TSV (Compara's ``gtr_results_all.tsv``).

        Args:
            gtr_path: Tab-separated table, see ``gtr_parameters_from_results``.
            name: Model identifier.

        Returns:
            The model, with one family per row.
        """
        gtr_df = read_gtr_results_table(gtr_path)
        logger.info("Loaded %d GTR results from %s", len(gtr_df), gtr_path)
        return cls(gtr_parameters_from_results(gtr_df), name=name)

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save the per-family parameters to a zip archive.

        The archive holds ``metadata.yml`` and ``families.tsv``, with one row
        per family in the GTR results table layout.

        Args:
            path: Output path for the zip archive.
        """
        table = pd.DataFrame(
            {
                "family_id": self._keys,
                **{
                    col: self._exchangeabilities[:, i].tolist()
                    for i, col in enumerate(GTR_EXCHANGEABILITY_COLUMNS)
                },
                **{
                    col: self._frequencies[:, i].tolist()
                    for i, col in enumerate(GTR_FREQUENCY_COLUMNS)
                },
            }
        )
        write_data_archive(
            path,
            PerFamilyGTRArchiveMetadata(
                model_name=self._name, num_families=len(self._keys)
            ),
            {
                _FAMILIES_FILENAME: table.to_csv(
                    sep="\t", index=False, float_format="%.9g"
                ).encode("utf-8")
            },
        )
        logger.info("Saved %s (%d families) to %s", self._name, len(self._keys), path)

    _EXPECTED_MODEL_TYPE = "PerFamilyGTRNeutralModel"

    @classmethod
    def from_pretrained(
        cls, model_path: Union[str, Path]
    ) -> "PerFamilyGTRNeutralModel":
        """Load from an archive written by ``save_pretrained``.

        Raises:
            FileNotFoundError: If the archive is missing or lacks its table.
            ValueError: If the archive is of another model type.
        """
        path = Path(model_path)
        if not data_archive_exists(path, [_FAMILIES_FILENAME]):
            raise FileNotFoundError(
                f"Invalid or missing archive at {path}. Expected zip with "
                f"metadata.yml and {_FAMILIES_FILENAME}."
            )
        metadata, files = read_data_archive(path, [_FAMILIES_FILENAME])
        if metadata.get("model_type") != cls._EXPECTED_MODEL_TYPE:
            raise ValueError(
                f"Archive has wrong model_type: {metadata.get('model_type')}. "
                f"Expected: {cls._EXPECTED_MODEL_TYPE}."
            )
        gtr_df = read_gtr_results_table(io.BytesIO(files[_FAMILIES_FILENAME]))
        # Already normalized at save time. Renormalizing again could move a
        # frequency by an ulp, and the rates would then differ from the ones
        # a column-carrying twin stored.
        keys, exchangeabilities, frequencies = _read_parameter_table(gtr_df)
        model = cls(
            {key: (exchangeabilities[i], frequencies[i]) for i, key in enumerate(keys)},
            name=metadata.get("model_name", "PerFamilyGTR"),
        )
        model._archive_path = path
        return model
