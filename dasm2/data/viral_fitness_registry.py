"""Registry for viral fitness and PCP data manifests.

This module provides YAML-based registry support for two types of viral data
manifests from the mat-pcp pipeline:

- **Fitness data** (``dasm2_fitness_data_schema: 1``): Per-clade amino acid
  mutation fitness estimates with uncertainty and context columns.
- **Viral PCP data** (``dasm2_viral_pcp_data_schema: 1``): Per-clade
  parent-child pairs, founder sequences, rates tables, and sites to mask.

Both manifest types support ``extends:`` inheritance for sharing base fields
across clade-specific manifests.

Usage::

    from dasm2.data.viral_fitness_registry import (
        get_fitness_config,
        get_viral_pcp_config,
        list_fitness_datasets,
        list_viral_pcp_datasets,
    )

    # Discover available datasets
    for name in list_fitness_datasets():
        config = get_fitness_config(name)
        print(f"{name}: clade={config.clade}")

    # Load fitness data
    config = get_fitness_config("s2SpikeFitness21K-2022-06-01-v1")
    df = config.load()

    # Access mask file
    pcp = get_viral_pcp_config("s2Spike21K-2022-06-01-v1")
    mask_path = pcp.resolve_sites_to_mask().path
"""

from pathlib import Path
from typing import Annotated, Any, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from dasm2.data.discovery import get_manifest_index
from dasm2.data.local import PathWithMD5, VerifiedPath
from dasm2.data.manifests import (
    coerce_date_to_str,
    load_yaml_config,
    resolve_path_with_md5,
)

# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class ViralFitnessConfig(BaseModel):
    """Configuration for a viral fitness dataset (``dasm2_fitness_data_schema: 1``).

    Represents amino acid mutation fitness estimates from the mat-pcp pipeline.
    Base manifests define column schemas; clade-specific manifests add data paths.

    Attributes:
        dasm2_fitness_data_schema: Schema version discriminator.
        description: Human-readable dataset description.
        format: Data format identifier (e.g., ``"mat-pcp-aamut"``).
        virus: Virus name (e.g., ``"sars-cov-2"``).
        pipeline_version: Pipeline version string.
        data_cutoff_date: Data collection cutoff date.
        region: Genomic region (e.g., ``"S"`` for Spike).
        clade: Nextstrain clade identifier (e.g., ``"21K"``).
        mutation_cols: Column names for mutation identification.
        y_col: Column name for fitness values.
        y_uncertainty_col: Column name for uncertainty estimates.
        context_cols: Additional context column names.
        data: Path to fitness CSV with optional MD5 checksum.
        yaml_dir: Directory of source YAML file (set during loading).
    """

    dasm2_fitness_data_schema: Annotated[int, Field(ge=1)]
    description: Optional[str] = None
    format: Optional[str] = None
    virus: Optional[str] = None
    pipeline_version: Optional[str] = None
    data_cutoff_date: Optional[str] = None
    region: Optional[str] = None
    clade: Optional[str] = None
    mutation_cols: Optional[List[str]] = None
    y_col: Optional[str] = None
    y_uncertainty_col: Optional[str] = None
    context_cols: Optional[List[str]] = None
    data: Optional[PathWithMD5] = None
    yaml_dir: Optional[Path] = Field(default=None, exclude=True)

    @field_validator("data_cutoff_date", mode="before")
    @classmethod
    def coerce_date(cls, v: Any) -> Optional[str]:
        """Coerce datetime.date from YAML parsing to string."""
        return coerce_date_to_str(v)

    def resolve_data_path(self) -> VerifiedPath:
        """Resolve the data CSV path relative to the YAML file location.

        Returns:
            VerifiedPath with absolute path and optional MD5 for lazy verification.

        Raises:
            ValueError: If no data path is specified or path cannot be resolved.
        """
        return resolve_path_with_md5(self.data, self.yaml_dir, "data")

    def load(self) -> pd.DataFrame:
        """Load the fitness data as a DataFrame.

        Returns:
            DataFrame from the fitness CSV (triggers MD5 verification on first load).

        Raises:
            ValueError: If data path is not configured.
        """
        return pd.read_csv(self.resolve_data_path().path)


class ViralPCPConfig(BaseModel):
    """Configuration for viral PCP data (``dasm2_viral_pcp_data_schema: 1``).

    Represents parent-child pair data from the mat-pcp pipeline, including
    founder sequences, rates tables, and shared resources like the
    problematic-sites mask file.

    Attributes:
        dasm2_viral_pcp_data_schema: Schema version discriminator.
        description: Human-readable dataset description.
        virus: Virus name (e.g., ``"sars-cov-2"``).
        pipeline_version: Pipeline version string.
        data_cutoff_date: Data collection cutoff date.
        region: Genomic region (e.g., ``"S"`` for Spike).
        clade: Nextstrain clade identifier (e.g., ``"21K"``).
        data: Path to PCP CSV with optional MD5.
        founder_sequence: Path to founder FASTA with optional MD5.
        founder_mutations: Path to founder mutations CSV with optional MD5.
        rates_table: Path to rates table CSV with optional MD5.
        sites_to_mask: Path to problematic sites file with optional MD5. Codon-
            granular; prefer ``masked_nt_sites``.
        masked_nt_sites: Path to the clade-independent nucleotide mask with
            optional MD5.
        master_table: Path to the unfiltered GLM rate table with optional MD5.
        site_annotations: Path to the per-site genome annotation with optional
            MD5.
        censored_mutations: Path to the UShER-masked mutation censor table with
            optional MD5.
        censored_clades: Path to the censor table's clade roster with optional
            MD5.
        yaml_dir: Directory of source YAML file (set during loading).

    Note:
        ``rates_table`` is the rate table as the producing pipeline shipped it.
        For mat-pcp SARS-CoV-2 runs that means after-boundary GLM rows only,
        which is correct for regions downstream of every light-switch boundary
        (``S``) and wrong for regions upstream of them (``ORF1a``). Build a
        neutral model for a non-spike region from ``master_table`` instead.

    Note:
        ``censored_mutations`` and ``censored_clades`` are a pair, and both are
        needed to build a censor table (#1079). The roster is not derivable
        from the CSV: a clade whose censored-mutation set is empty contributes
        no rows, and without the roster it is indistinguishable from a clade
        name that was misspelled.
    """

    dasm2_viral_pcp_data_schema: Annotated[int, Field(ge=1)]
    description: Optional[str] = None
    virus: Optional[str] = None
    pipeline_version: Optional[str] = None
    data_cutoff_date: Optional[str] = None
    region: Optional[str] = None
    clade: Optional[str] = None
    data: Optional[PathWithMD5] = None
    founder_sequence: Optional[PathWithMD5] = None
    founder_mutations: Optional[PathWithMD5] = None
    rates_table: Optional[PathWithMD5] = None
    sites_to_mask: Optional[PathWithMD5] = None
    masked_nt_sites: Optional[PathWithMD5] = None
    master_table: Optional[PathWithMD5] = None
    site_annotations: Optional[PathWithMD5] = None
    censored_mutations: Optional[PathWithMD5] = None
    censored_clades: Optional[PathWithMD5] = None
    yaml_dir: Optional[Path] = Field(default=None, exclude=True)

    @field_validator("data_cutoff_date", mode="before")
    @classmethod
    def coerce_date(cls, v: Any) -> Optional[str]:
        """Coerce datetime.date from YAML parsing to string."""
        return coerce_date_to_str(v)

    def resolve_data_path(self) -> VerifiedPath:
        """Resolve the PCP data CSV path."""
        return resolve_path_with_md5(self.data, self.yaml_dir, "data")

    def resolve_founder_sequence(self) -> VerifiedPath:
        """Resolve the founder sequence FASTA path."""
        return resolve_path_with_md5(
            self.founder_sequence, self.yaml_dir, "founder_sequence"
        )

    def resolve_founder_mutations(self) -> VerifiedPath:
        """Resolve the founder mutations CSV path."""
        return resolve_path_with_md5(
            self.founder_mutations, self.yaml_dir, "founder_mutations"
        )

    def resolve_rates_table(self) -> VerifiedPath:
        """Resolve the rates table CSV path."""
        return resolve_path_with_md5(self.rates_table, self.yaml_dir, "rates_table")

    def resolve_sites_to_mask(self) -> VerifiedPath:
        """Resolve the sites-to-mask file path."""
        return resolve_path_with_md5(self.sites_to_mask, self.yaml_dir, "sites_to_mask")

    def resolve_masked_nt_sites(self) -> VerifiedPath:
        """Resolve the clade-independent nucleotide mask path (mat-pcp#153).

        ``pcps/{gene}/masked_nt_sites.csv``, the nucleotide-granular form of
        :meth:`resolve_sites_to_mask`'s ``{gene}_codon_sites_to_mask.txt``. Both
        express mat-pcp's same three clade-independent exclusion reasons, but the
        codon file widens each flagged nucleotide to its whole codon -- 345 masked
        nucleotides against 121 across the six v12 genes. Present on mat-pcp v12+
        manifests; ``None`` on earlier ones, which shipped only the codon file.
        """
        return resolve_path_with_md5(
            self.masked_nt_sites, self.yaml_dir, "masked_nt_sites"
        )

    def resolve_master_table(self) -> VerifiedPath:
        """Resolve the unfiltered GLM rate table CSV path.

        Carries both before- and after-boundary rows plus the
        ``nt_site_before_boundary`` flag, so a caller can select the rows that
        are correct for its region. Present on mat-pcp v8+ manifests.
        """
        return resolve_path_with_md5(self.master_table, self.yaml_dir, "master_table")

    def resolve_site_annotations(self) -> VerifiedPath:
        """Resolve the per-site genome annotation CSV path.

        Source of the ``unpaired`` secondary-structure call used to build a
        neutral model's ``secondary_struct`` table. Present on mat-pcp v8+
        manifests.
        """
        return resolve_path_with_md5(
            self.site_annotations, self.yaml_dir, "site_annotations"
        )

    @model_validator(mode="after")
    def check_censor_pair(self) -> "ViralPCPConfig":
        """Require the censor table and its roster to be declared together.

        Rejected at load rather than at use because the half-declared case is
        the one that fails quietly: with a CSV and no roster a caller falls
        back to deriving the clade set from the rows, and every clade that
        censors nothing then looks like a clade that does not exist.
        """
        present = [
            name
            for name in ("censored_mutations", "censored_clades")
            if getattr(self, name) is not None
        ]
        if len(present) == 1:
            missing = (
                "censored_clades"
                if present[0] == "censored_mutations"
                else ("censored_mutations")
            )
            raise ValueError(
                f"{present[0]} is set but {missing} is not. A censor table and "
                "its clade roster are a pair; declare both or neither."
            )
        return self

    def resolve_censored_mutations(self) -> VerifiedPath:
        """Resolve the UShER-masked mutation censor table CSV path.

        Rows are the censored mutations of one clade, as emitted by mat-pcp's
        ``censor_table`` rule: site masks carry blank ``from_nt``/``to_nt``,
        directional masks name both. Consumed with
        :func:`dasm2.sequence.masking.load_censor_table`, which needs
        :meth:`resolve_censored_clades` as well. Present on mat-pcp v11+
        manifests.
        """
        return resolve_path_with_md5(
            self.censored_mutations, self.yaml_dir, "censored_mutations"
        )

    def resolve_censored_clades(self) -> VerifiedPath:
        """Resolve the censor table's clade roster path.

        One clade name per line, covering every clade in the corpus including
        those with no censored mutations at all -- which is the whole point of
        shipping it separately from the CSV.
        """
        return resolve_path_with_md5(
            self.censored_clades, self.yaml_dir, "censored_clades"
        )


# ---------------------------------------------------------------------------
# Registry (delegates to unified ManifestIndex)
# ---------------------------------------------------------------------------

_FITNESS_DISCRIMINATOR = "dasm2_fitness_data_schema"
_PCP_DISCRIMINATOR = "dasm2_viral_pcp_data_schema"


def load_fitness_config(yaml_path: Union[str, Path]) -> ViralFitnessConfig:
    """Load a viral fitness configuration from a YAML file.

    Handles ``extends:`` inheritance if present.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Parsed ViralFitnessConfig with yaml_dir set for path resolution.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        ValueError: If the YAML is invalid or missing the discriminator.
    """
    return load_yaml_config(
        Path(yaml_path).resolve(),
        _FITNESS_DISCRIMINATOR,
        ViralFitnessConfig,
    )


def load_viral_pcp_config(yaml_path: Union[str, Path]) -> ViralPCPConfig:
    """Load a viral PCP configuration from a YAML file.

    Handles ``extends:`` inheritance if present.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Parsed ViralPCPConfig with yaml_dir set for path resolution.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        ValueError: If the YAML is invalid or missing the discriminator.
    """
    return load_yaml_config(
        Path(yaml_path).resolve(),
        _PCP_DISCRIMINATOR,
        ViralPCPConfig,
    )


def get_fitness_config(name: str) -> ViralFitnessConfig:
    """Get a registered fitness dataset configuration by name.

    Args:
        name: Dataset name (YAML filename stem, e.g.,
            ``"s2SpikeFitness21K-2022-06-01-v1"``).

    Returns:
        The fitness dataset configuration.

    Raises:
        KeyError: If no dataset with this name is registered.
    """
    config: ViralFitnessConfig = get_manifest_index().get_typed(
        name, _FITNESS_DISCRIMINATOR
    )
    return config


def get_viral_pcp_config(name: str) -> ViralPCPConfig:
    """Get a registered viral PCP dataset configuration by name.

    Args:
        name: Dataset name (YAML filename stem, e.g.,
            ``"s2Spike21K-2022-06-01-v1"``).

    Returns:
        The viral PCP configuration.

    Raises:
        KeyError: If no dataset with this name is registered.
    """
    config: ViralPCPConfig = get_manifest_index().get_typed(name, _PCP_DISCRIMINATOR)
    return config


def list_fitness_datasets() -> List[str]:
    """List all registered fitness dataset names.

    Returns:
        Sorted list of dataset names.
    """
    return get_manifest_index().list_names(_FITNESS_DISCRIMINATOR)


def list_viral_pcp_datasets() -> List[str]:
    """List all registered viral PCP dataset names.

    Returns:
        Sorted list of dataset names.
    """
    return get_manifest_index().list_names(_PCP_DISCRIMINATOR)
