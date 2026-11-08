"""Pydantic models for YAML-based dataset registry schema.

This module defines the schema for dataset YAML files. Each dataset has its own
YAML file containing all dataset-specific metadata, making datasets portable
and self-contained.

File type discrimination uses explicit keys:
- Dataset YAMLs: `dasm2_pcp_data_schema: 1`
- Simulation YAMLs: `dasm2_simulation_schema: 1`

This makes file types unambiguous and enables fail-fast validation when
a discriminator is present.

Example YAML file:
    ```yaml
    dasm2_pcp_data_schema: 1

    description: >
      Tang et al. heavy chain BCR data.

    data:
      path: v3/tang-data.csv.gz
      md5: 9bca0b9fe5d4e1ef63c234158f141c82

    holdout_samples:
      - B11
      - B20

    anarci:
      imgt:
        heavy:
          path: anarci/tang_imgt.csv
          md5: 5c35e91d1fe730717c66538d2cae8724
    ```
"""

from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from dasm2.data.local import PathWithMD5, VerifiedPath, resolve_path

# Type alias for ANARCI structure: {numbering_scheme: {segment: PathWithMD5}}
AnarciDict = Dict[str, Dict[str, PathWithMD5]]


class NaughtyFamily(BaseModel):
    """A problematic clonal family to filter during loading.

    Attributes:
        sample_id: The sample identifier.
        family: The family identifier.
    """

    sample_id: str
    family: str

    @classmethod
    def from_list(cls, data: List[str]) -> "NaughtyFamily":
        """Create from [sample_id, family] list format used in YAML."""
        if len(data) != 2:
            raise ValueError(
                f"Naughty family must be [sample_id, family], got {len(data)} elements"
            )
        return cls(sample_id=data[0], family=str(data[1]))


def _parse_naughty_families_list(
    v: Optional[List[Any]],
) -> Optional[List[NaughtyFamily]]:
    """Parse naughty families from YAML list format [[sample_id, family], ...].

    Shared validator logic for DatasetConfig and SimulationConfig.

    Args:
        v: Raw list from YAML (lists, dicts, or NaughtyFamily instances).

    Returns:
        Parsed list of NaughtyFamily objects, or None if input is None.
    """
    if v is None:
        return None
    result = []
    for item in v:
        if isinstance(item, list):
            result.append(NaughtyFamily.from_list(item))
        elif isinstance(item, dict):
            result.append(NaughtyFamily(**item))
        elif isinstance(item, NaughtyFamily):
            result.append(item)
        else:
            raise ValueError(f"Invalid naughty family format: {item}")
    return result


class DatasetConfig(BaseModel):
    """Configuration for a single dataset loaded from YAML.

    All path fields use relative paths from the YAML file's directory.

    Attributes:
        dasm2_pcp_data_schema: Discriminator key and schema version (currently 1).
        extends: Optional path to base YAML file for inheritance.
        description: Human-readable dataset description.
        data: Main data file path with optional MD5.
        holdout_samples: Sample IDs held out for validation.
        naughty_clonal_families: Problematic (sample_id, family) pairs to filter.
        anarci: ANARCI annotation paths by numbering scheme and segment.
        clonal_family_trees: Optional path to clonal family trees (for simulation).
        unfiltered_data: Optional path to unfiltered data (for simulation).
    """

    dasm2_pcp_data_schema: Annotated[int, Field(ge=1)]
    extends: Optional[str] = None
    description: Optional[str] = None
    data: PathWithMD5
    holdout_samples: Optional[List[str]] = None
    naughty_clonal_families: Optional[List[NaughtyFamily]] = None
    anarci: Optional[AnarciDict] = None
    clonal_family_trees: Optional[PathWithMD5] = None
    unfiltered_data: Optional[PathWithMD5] = None

    @field_validator("naughty_clonal_families", mode="before")
    @classmethod
    def parse_naughty_families(
        cls, v: Optional[List[Any]]
    ) -> Optional[List[NaughtyFamily]]:
        """Parse naughty families from YAML list format [[sample_id, family], ...]."""
        return _parse_naughty_families_list(v)

    def get_naughty_families_as_tuples(self) -> List[Tuple[str, str]]:
        """Get naughty families as list of (sample_id, family) tuples.

        Returns:
            List of tuples for compatibility with existing filtering code.
        """
        if self.naughty_clonal_families is None:
            return []
        return [(nf.sample_id, nf.family) for nf in self.naughty_clonal_families]


class ResolvedConfigBase(BaseModel):
    """Base class for resolved configurations with verified paths.

    This class provides common fields shared between dataset and simulation
    configurations. Paths are wrapped in VerifiedPath for lazy MD5 verification.

    Attributes:
        yaml_path: Absolute path to the source YAML file.
        discriminator_schema: Schema version from discriminator key.
        description: Human-readable description.
        data: Main data file with lazy MD5 verification.
        clonal_family_trees: Optional trees file (required in simulation).
        unfiltered_data: Optional unfiltered data file (required in simulation).
        anarci: ANARCI annotation files by numbering scheme and segment.
        holdout_samples: Sample IDs for validation holdout.
        naughty_clonal_families: Problematic families to filter.
    """

    yaml_path: Path
    discriminator_schema: int
    description: Optional[str] = None
    data: VerifiedPath
    clonal_family_trees: Optional[VerifiedPath] = None
    unfiltered_data: Optional[VerifiedPath] = None
    anarci: Dict[str, Dict[str, VerifiedPath]] = Field(default_factory=dict)
    holdout_samples: List[str] = Field(default_factory=list)
    naughty_clonal_families: List[Tuple[str, str]] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class ResolvedDatasetConfig(ResolvedConfigBase):
    """A DatasetConfig with all paths resolved to absolute paths.

    This is the result of loading and resolving a DatasetConfig, with
    paths converted from relative (to YAML file) to absolute. Paths are
    wrapped in VerifiedPath for lazy MD5 verification on first access.

    Attributes:
        name: Dataset name (derived from YAML filename).
    """

    name: str

    @classmethod
    def from_config(
        cls,
        config: DatasetConfig,
        name: str,
        yaml_path: Path,
    ) -> "ResolvedDatasetConfig":
        """Create a ResolvedDatasetConfig from a DatasetConfig.

        Resolves all relative paths to absolute paths based on YAML file location.

        Args:
            config: The parsed DatasetConfig.
            name: Dataset name (typically from YAML filename).
            yaml_path: Absolute path to the YAML file.

        Returns:
            ResolvedDatasetConfig with absolute paths.
        """
        yaml_dir = yaml_path.parent

        # Resolve main data path
        data = VerifiedPath(
            resolve_path(yaml_dir, config.data.path),
            config.data.md5,
        )

        # Resolve ANARCI paths
        resolved_anarci: Dict[str, Dict[str, VerifiedPath]] = {}
        if config.anarci:
            for scheme, segments in config.anarci.items():
                resolved_anarci[scheme] = {}
                for segment, path_with_md5 in segments.items():
                    resolved_anarci[scheme][segment] = VerifiedPath(
                        resolve_path(yaml_dir, path_with_md5.path),
                        path_with_md5.md5,
                    )

        # Resolve optional paths
        trees = None
        if config.clonal_family_trees:
            trees = VerifiedPath(
                resolve_path(yaml_dir, config.clonal_family_trees.path),
                config.clonal_family_trees.md5,
            )

        unfiltered = None
        if config.unfiltered_data:
            unfiltered = VerifiedPath(
                resolve_path(yaml_dir, config.unfiltered_data.path),
                config.unfiltered_data.md5,
            )

        return cls(
            name=name,
            yaml_path=yaml_path,
            discriminator_schema=config.dasm2_pcp_data_schema,
            description=config.description,
            data=data,
            holdout_samples=config.holdout_samples or [],
            naughty_clonal_families=config.get_naughty_families_as_tuples(),
            anarci=resolved_anarci,
            clonal_family_trees=trees,
            unfiltered_data=unfiltered,
        )


# =============================================================================
# Simulation Configuration
# =============================================================================


class SimulationConfig(BaseModel):
    """Configuration for a simulation loaded from YAML.

    Simulation configs can either reference a dataset by name OR be fully
    self-contained with all required fields inline. Dataset fields can be
    overridden when referencing a dataset.

    Example YAML file (referencing a dataset):
        ```yaml
        dasm2_simulation_schema: 1
        dataset: v1tang

        checkpoint_path: checkpoints/model.zip
        seed: 42
        replicates: 10
        ```

    Example YAML file (fully self-contained):
        ```yaml
        dasm2_simulation_schema: 1

        checkpoint_path: checkpoints/model.zip
        seed: 42

        data:
          path: my_data.csv.gz
        unfiltered_data:
          path: unfiltered.csv.gz
        clonal_family_trees:
          path: trees.csv.gz
        ```

    Attributes:
        dasm2_simulation_schema: Discriminator key and schema version (currently 1).
        dataset: Optional name of dataset to reference from registry.
        checkpoint_path: Path to DASM2 model checkpoint.
        seed: Random seed for reproducibility.
        replicates: Number of simulation replicates (default 1).
        description: Optional description of this simulation.
        data: Data file path (required if no dataset reference).
        unfiltered_data: Unfiltered data path (required, from dataset or inline).
        clonal_family_trees: Trees path (required, from dataset or inline).
        anarci: ANARCI annotation paths.
        holdout_samples: Sample IDs for validation holdout.
        naughty_clonal_families: Problematic families to filter.
    """

    dasm2_simulation_schema: Annotated[int, Field(ge=1)]
    dataset: Optional[str] = None
    checkpoint_path: str
    seed: int
    replicates: int = 1
    description: Optional[str] = None

    # Data fields (required if no dataset reference, or as overrides)
    data: Optional[PathWithMD5] = None
    unfiltered_data: Optional[PathWithMD5] = None
    clonal_family_trees: Optional[PathWithMD5] = None
    anarci: Optional[AnarciDict] = None
    holdout_samples: Optional[List[str]] = None
    naughty_clonal_families: Optional[List[NaughtyFamily]] = None

    # Optional indel-aware simulation: NPZ file with per-node presence masks
    presence_mask_file: Optional[str] = None

    @field_validator("dataset")
    @classmethod
    def dataset_not_empty(cls, v: Optional[str]) -> Optional[str]:
        """Validate that dataset name is not empty if provided."""
        if v is not None and not v.strip():
            raise ValueError("dataset name cannot be empty")
        return v

    @field_validator("checkpoint_path")
    @classmethod
    def checkpoint_path_not_empty(cls, v: str) -> str:
        """Validate that checkpoint_path is not empty."""
        if not v or not v.strip():
            raise ValueError("checkpoint_path cannot be empty")
        return v

    @field_validator("naughty_clonal_families", mode="before")
    @classmethod
    def parse_naughty_families(
        cls, v: Optional[List[Any]]
    ) -> Optional[List[NaughtyFamily]]:
        """Parse naughty families from YAML list format [[sample_id, family], ...]."""
        return _parse_naughty_families_list(v)


class ResolvedSimulationConfig(ResolvedConfigBase):
    """A SimulationConfig with all paths resolved to absolute paths.

    This is the result of loading a SimulationConfig, resolving the dataset
    reference (if any), and merging/resolving all paths. Paths are wrapped in
    VerifiedPath for lazy MD5 verification on first access.

    Note: Unlike ResolvedDatasetConfig, simulation configs require that
    clonal_family_trees and unfiltered_data are present.

    Attributes:
        dataset_name: Name of the referenced dataset (None if self-contained).
        checkpoint: Model checkpoint file with lazy MD5 verification.
        seed: Random seed.
        replicates: Number of replicates.
        clonal_family_trees: Required trees file (overrides Optional from base).
        unfiltered_data: Required unfiltered data file (overrides Optional from base).
    """

    dataset_name: Optional[str]
    checkpoint: VerifiedPath
    seed: int
    replicates: int

    # Optional indel-aware simulation: resolved path to NPZ presence mask file
    presence_mask_file: Optional[Path] = None

    # Override Optional types from base - these are required for simulation
    clonal_family_trees: VerifiedPath  # type: ignore[assignment]
    unfiltered_data: VerifiedPath  # type: ignore[assignment]

    @classmethod
    def from_config(
        cls,
        config: SimulationConfig,
        dataset: Optional["ResolvedDatasetConfig"],
        yaml_path: Path,
    ) -> "ResolvedSimulationConfig":
        """Create a ResolvedSimulationConfig by merging simulation config with dataset.

        Resolves all paths relative to the simulation YAML file location.
        If a dataset is provided, its fields are used as defaults and simulation
        config fields override. If no dataset is provided, all required fields
        must be in the simulation config.

        Args:
            config: The parsed SimulationConfig.
            dataset: The resolved dataset config to inherit from, or None if
                the simulation config is self-contained.
            yaml_path: Absolute path to the simulation YAML file.

        Returns:
            ResolvedSimulationConfig with all paths resolved.

        Raises:
            ValueError: If required fields (data, unfiltered_data, clonal_family_trees)
                are not available from dataset or config.
        """
        yaml_dir = yaml_path.parent

        # Resolve checkpoint path
        checkpoint = VerifiedPath(
            resolve_path(yaml_dir, config.checkpoint_path),
            md5=None,  # Checkpoint doesn't have MD5 in config
        )

        # Resolve data (config or dataset, required)
        if config.data:
            data = VerifiedPath(
                resolve_path(yaml_dir, config.data.path),
                config.data.md5,
            )
        elif dataset:
            data = dataset.data
        else:
            raise ValueError(
                "Simulation requires 'data' field but no dataset reference provided "
                "and no 'data' field in simulation config."
            )

        # Resolve unfiltered_data (config or dataset, required)
        if config.unfiltered_data:
            unfiltered = VerifiedPath(
                resolve_path(yaml_dir, config.unfiltered_data.path),
                config.unfiltered_data.md5,
            )
        elif dataset and dataset.unfiltered_data:
            unfiltered = dataset.unfiltered_data
        else:
            source = f"dataset '{config.dataset}'" if config.dataset else "config"
            raise ValueError(
                f"Simulation requires unfiltered_data but {source} "
                f"does not have it defined and no override provided."
            )

        # Resolve clonal_family_trees (config or dataset, required)
        if config.clonal_family_trees:
            trees = VerifiedPath(
                resolve_path(yaml_dir, config.clonal_family_trees.path),
                config.clonal_family_trees.md5,
            )
        elif dataset and dataset.clonal_family_trees:
            trees = dataset.clonal_family_trees
        else:
            source = f"dataset '{config.dataset}'" if config.dataset else "config"
            raise ValueError(
                f"Simulation requires clonal_family_trees but {source} "
                f"does not have it defined and no override provided."
            )

        # Resolve ANARCI (config or dataset)
        if config.anarci:
            resolved_anarci: Dict[str, Dict[str, VerifiedPath]] = {}
            for scheme, segments in config.anarci.items():
                resolved_anarci[scheme] = {}
                for segment, path_with_md5 in segments.items():
                    resolved_anarci[scheme][segment] = VerifiedPath(
                        resolve_path(yaml_dir, path_with_md5.path),
                        path_with_md5.md5,
                    )
        elif dataset:
            resolved_anarci = dataset.anarci
        else:
            resolved_anarci = {}

        # Use config or dataset for holdout_samples
        if config.holdout_samples is not None:
            holdout_samples = config.holdout_samples
        elif dataset:
            holdout_samples = dataset.holdout_samples
        else:
            holdout_samples = []

        # Use config or dataset for naughty_clonal_families
        if config.naughty_clonal_families is not None:
            naughty_families = [
                (nf.sample_id, nf.family) for nf in config.naughty_clonal_families
            ]
        elif dataset:
            naughty_families = dataset.naughty_clonal_families
        else:
            naughty_families = []

        # Resolve presence_mask_file (optional, for indel-aware simulation)
        resolved_mask_file: Optional[Path] = None
        if config.presence_mask_file:
            resolved_mask_file = resolve_path(yaml_dir, config.presence_mask_file)

        return cls(
            yaml_path=yaml_path,
            discriminator_schema=config.dasm2_simulation_schema,
            dataset_name=config.dataset,
            checkpoint=checkpoint,
            seed=config.seed,
            replicates=config.replicates,
            description=config.description,
            data=data,
            unfiltered_data=unfiltered,
            clonal_family_trees=trees,
            anarci=resolved_anarci,
            holdout_samples=holdout_samples,
            naughty_clonal_families=naughty_families,
            presence_mask_file=resolved_mask_file,
        )
