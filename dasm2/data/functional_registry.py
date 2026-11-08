"""Registry schema and loading for functional data (FLAb, DMS, etc.).

This module defines a YAML-based registry for functional datasets that is
separate from the evolutionary/PCP data registry. Functional data includes:
- FLAb (Fitness Landscape for Antibodies) binding/expression data
- DMS (Deep Mutational Scanning) data
- Any sequence-to-function mapping data

Uses discriminator `dasm2_functional_data_schema: 1` to distinguish from
evolutionary data (`dasm2_pcp_data_schema: 1`).

Example YAML:
    ```yaml
    dasm2_functional_data_schema: 1

    description: >
      Koenig 2017 g6 binding data from FLAb.

    format: flab
    data:
      path: FLAb/data/binding/Koenig2017_g6_Kd.csv

    sequence_cols:
      - heavy
      - light

    y_col: fitness
    y_col_rename: binding
    ```

Usage:
    from dasm2.data.functional_registry import load_functional_dataset

    df = load_functional_dataset("koenig_binding")
"""

from enum import Enum
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from dasm2.data.discovery import get_manifest_index
from dasm2.data.functional_formats import (
    FUNCTIONAL_EFFECT_Y_COL,
    _sanitize_column_name,
    from_flab,
    from_multidms,
)
from dasm2.data.local import PathWithMD5, VerifiedPath, resolve_path
from dasm2.data.manifests import load_yaml_config


class MutationType(str, Enum):
    """Whether variants have single or multiple mutations.

    Used to determine which correlation function to use during evaluation:
    - SINGLE: DMS-style data with one mutation per variant → use DMS matrix correlation
    - MULTIPLE: Multi-mutant data → use selection score correlation
    """

    SINGLE = "single"
    MULTIPLE = "multiple"


# Type alias for y_type field values
# After transform, higher values = better (stronger binding, higher fitness)
YType = Literal[
    "enrichment_ratio",  # Raw ratio centered at 1 → log(y)
    "kd_nm",  # Kd in nM → -log10(y * 1e-9)
    "kd_m",  # Kd in M → -log10(y)
    "log_kd_nm",  # log10(Kd in nM) → 9 - y (converts to pKd)
    "fitness",  # Already in higher=better orientation → identity
]


def _validate_positive(values: pd.Series, transform_name: str) -> None:
    """Raise ValueError if any values are non-positive or NaN."""
    invalid = (values <= 0) | values.isna()
    if invalid.any():
        raise ValueError(
            f"{transform_name}: {invalid.sum()} values are non-positive or NaN "
            f"(valid range: (0, inf)). Cannot apply log transform."
        )


def _transform_enrichment_ratio(values: pd.Series) -> pd.Series:
    """Transform enrichment ratio to log-enrichment.

    Args:
        values: Enrichment ratios centered at 1 (WT = 1.0).

    Returns:
        Log-transformed values where WT = 0.0 and higher = better.

    Raises:
        ValueError: If any values are <= 0.
    """
    _validate_positive(values, "enrichment_ratio")
    result: pd.Series = pd.Series(np.log(values), index=values.index)
    return result


def _transform_kd_nm(values: pd.Series) -> pd.Series:
    """Transform KD in nM to -log10(KD in M).

    Higher values = better binding.

    Args:
        values: KD values in nanomolar.

    Returns:
        -log10(KD in molar), so 1 nM → 9, 10 nM → 8, etc.

    Raises:
        ValueError: If any values are <= 0.
    """
    _validate_positive(values, "kd_nm")
    result: pd.Series = pd.Series(-np.log10(values * 1e-9), index=values.index)
    return result


def _transform_kd_m(values: pd.Series) -> pd.Series:
    """Transform KD in M to -log10(KD).

    Higher values = better binding.

    Args:
        values: KD values in molar.

    Returns:
        -log10(KD), so 1e-9 M → 9, 1e-8 M → 8, etc.

    Raises:
        ValueError: If any values are <= 0.
    """
    _validate_positive(values, "kd_m")
    result: pd.Series = pd.Series(-np.log10(values), index=values.index)
    return result


def _transform_log_kd_nm(values: pd.Series) -> pd.Series:
    """Transform log10(Kd in nM) to pKd (-log10(Kd in M)).

    Higher values = better binding.

    Args:
        values: log10(Kd in nM) values.

    Returns:
        pKd values: 9 - y, so log10(1 nM) = 0 → pKd 9.
    """
    return pd.Series(9 - values, index=values.index)


def _identity(values: pd.Series) -> pd.Series:
    """Identity transform for data already in higher=better orientation.

    Args:
        values: Pre-transformed values where higher = better.

    Returns:
        Values unchanged.
    """
    return values


# Transform function type: takes a Series, returns a Series
_TransformFunc = Callable[[pd.Series], pd.Series]

Y_TYPE_TRANSFORMS: Dict[YType, _TransformFunc] = {
    "enrichment_ratio": _transform_enrichment_ratio,
    "kd_nm": _transform_kd_nm,
    "kd_m": _transform_kd_m,
    "log_kd_nm": _transform_log_kd_nm,
    "fitness": _identity,
}


def apply_y_transform_array(values: np.ndarray, y_type: YType) -> np.ndarray:
    """Numpy-friendly wrapper around :data:`Y_TYPE_TRANSFORMS`.

    Existing transforms operate on ``pd.Series``; experiment scripts
    typically hold ``np.ndarray`` y values, so this round-trips through
    a Series and returns an ndarray. Higher = better orientation by
    construction (see :data:`YType`).

    Args:
        values: y values as a 1-D numpy array.
        y_type: One of the :data:`YType` literals.

    Returns:
        Transformed values as a numpy array of the same length.
    """
    return Y_TYPE_TRANSFORMS[y_type](pd.Series(values)).to_numpy()


def make_paired_seq(heavy: str, light: Optional[str]) -> str:
    """Build a paired heavy/light AA string in the canonical encoder format.

    The DASM2 encoder is trained with ``^`` as a real vocabulary token
    separating heavy and light chains (see
    :func:`dasm2.analysis.functional_eval` and
    :data:`dasm2.tokenizers.AA_TOKENIZER`). Empty / ``None`` light chain
    falls back to heavy-only.
    """
    return f"{heavy}^{light}" if light else heavy


class FunctionalDatasetConfig(BaseModel):
    """Configuration for a functional dataset loaded from YAML.

    Attributes:
        dasm2_functional_data_schema: Discriminator key and schema version.
        description: Human-readable dataset description.
        format: Data format - "flab", "multidms", or "standard".
        data: Path to data file as PathWithMD5 (accepts string or {path, md5} dict).
            Paths are relative to the YAML file location.
        sequence_cols: Column names containing amino acid sequences. Required
            except for format "multidms", whose CSVs have the fixed header
            wildtype,site,mutant,functional_effect.
        y_col: Column name for response variable in source data. Required
            except for format "multidms", which writes the constant column
            y_functional_effect.
        y_col_rename: Optional rename for output y_ column.
        y_type: Type of y values determining transformation to apply.
            After transformation, all y values follow the convention:
            higher values = better fitness/binding.
            Options: "enrichment_ratio", "kd_nm", "kd_m", "log_kd_nm",
            "fitness". Default is "fitness" (no transform, data already
            in higher=better orientation).
        group_id_col: Optional column to use as group_id.
        reference_sequence: For multidms format, the wildtype sequence. Accepts
            either a bare string or the data-central provenance mapping
            {source, clade, length, sequence}, which is flattened to the
            whitespace-normalized residues during validation.
        reference_nt_sequence: Optional nucleotide sequence for codon analysis.
        relax_wt_mismatch: Allow the CSV wildtype column to disagree with
            reference_sequence. multidms defines all mutations relative to a
            single reference homolog, so per-homolog founder sequences
            legitimately differ (#414). Unset means True for "multidms" and
            False for every other format; set it explicitly to override. See
            relax_wt_mismatch_resolved.
        model_checkpoint: Optional path to model for predictions.
        segments: Chain segments present in the dataset, e.g. ("heavy",) or
            ("heavy", "light"). Documents what sequences exist in the data.
        variable_segments: Chain segments that actually vary in the dataset.
            Used for model compatibility checking. If not specified, falls back
            to segments. For datasets where one chain is constant (e.g., Adams
            with constant light chain), set variable_segments to only the
            varying chain to allow heavy-only models to evaluate.
        mutations: Whether variants have single or multiple mutations.
            Required for unified functional evaluation.
        mask_positions: Optional tuple of 0-indexed positions to exclude from
            correlation analysis (e.g., positions not mutated in library design).
        mask_explanation: Human-readable explanation for why positions are masked.
            Must be present if mask_positions is present.
        yaml_dir: Directory containing the source YAML file (set during loading).
    """

    model_config = ConfigDict(extra="ignore")

    dasm2_functional_data_schema: Annotated[int, Field(ge=1)]
    description: Optional[str] = None
    format: Literal["flab", "multidms", "standard"]
    data: PathWithMD5
    sequence_cols: Optional[List[str]] = None
    y_col: Optional[str] = None
    y_col_rename: Optional[str] = None
    y_type: YType = "fitness"
    group_id_col: Optional[str] = None
    reference_sequence: Optional[str] = None
    reference_nt_sequence: Optional[str] = None
    relax_wt_mismatch: Optional[bool] = None
    model_checkpoint: Optional[str] = None

    # Evaluation metadata (required for unified functional evaluation)
    segments: Tuple[str, ...] = ()
    variable_segments: Optional[Tuple[str, ...]] = None
    mutations: Optional[MutationType] = None

    # Optional mask for positions to exclude from correlation
    mask_positions: Optional[Tuple[int, ...]] = None
    mask_explanation: Optional[str] = None

    yaml_dir: Optional[Path] = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def flatten_reference_sequence(cls, data: Any) -> Any:
        """Accept the manifests' {source, clade, length, sequence} mapping.

        data-central writes ``reference_sequence`` as provenance wrapping a
        nested ``sequence:`` block scalar. Consuming code (``from_multidms``)
        indexes the residues positionally, so flatten to a bare string here.

        The ``>-`` folded scalars fold newlines into spaces, making the raw
        value longer than the protein (1290-1291 chars for a 1273-mer), so all
        whitespace is stripped and the declared ``length`` is cross-checked.
        """
        if not isinstance(data, dict):
            return data
        ref = data.get("reference_sequence")
        if not isinstance(ref, dict):
            return data

        if "sequence" not in ref:
            raise ValueError(
                "reference_sequence mapping has no 'sequence' key; got "
                f"{sorted(ref)}. Provenance alone is not a sequence."
            )
        sequence = "".join(str(ref["sequence"]).split())
        declared = ref.get("length")
        if declared is not None and len(sequence) != int(declared):
            raise ValueError(
                f"reference_sequence length mismatch: declared {declared}, got "
                f"{len(sequence)} residues after whitespace normalization."
            )

        data = dict(data)
        data["reference_sequence"] = sequence
        return data

    @field_validator("y_col")
    @classmethod
    def validate_y_col_not_empty(cls, v: Optional[str]) -> Optional[str]:
        """Validate that y_col, when declared, is not empty.

        ``None`` passes through: whether omitting it is legal depends on the
        format, which :meth:`validate_format_required_fields` decides.
        """
        if v is None:
            return v
        if not v.strip():
            raise ValueError("y_col cannot be empty")
        return v

    @field_validator("sequence_cols")
    @classmethod
    def validate_sequence_cols_not_empty(
        cls, v: Optional[List[str]]
    ) -> Optional[List[str]]:
        """Validate that sequence_cols, when declared, has at least one column.

        ``None`` passes through; see :meth:`validate_y_col_not_empty`.
        """
        if v is None:
            return v
        if not v:
            raise ValueError("sequence_cols must have at least one column")
        return v

    @property
    def output_y_col(self) -> str:
        """Column name that :meth:`load` writes the target under.

        Standard-format CSVs expose the column as-is; flab / multidms
        converters rename it via ``y_col_rename`` (falling back to the
        sanitized ``y_col``). Either way, the loaded DataFrame's y column
        is ``y_<output_y_col_suffix>``.

        Some ``format: standard`` manifests write ``y_col`` already in its
        final, ``y_``-prefixed form (e.g. ``y_col: y_fitness``, matching the
        literal CSV column) rather than the pre-rename raw name flab/multidms
        expect. Re-prepending ``y_`` in that case would look up a
        nonexistent ``y_y_fitness`` column, so an already-prefixed
        ``y_col`` is returned as-is.

        ``multidms`` manifests declare no ``y_col`` (#989) because
        ``from_multidms`` writes the fixed column ``y_functional_effect``;
        that name is returned for them. Only for them --
        :meth:`validate_format_required_fields` requires ``y_col`` for every
        other format, so a ``None`` here would be a validator hole rather than
        a manifest this constant could describe.
        """
        if self.y_col_rename:
            return f"y_{self.y_col_rename}"
        if self.format == "multidms":
            return FUNCTIONAL_EFFECT_Y_COL
        if self.y_col is None:
            raise ValueError(
                f"format '{self.format}' has no y_col; output_y_col is undefined. "
                "validate_format_required_fields should have rejected this."
            )
        sanitized = _sanitize_column_name(self.y_col)
        return sanitized if sanitized.startswith("y_") else f"y_{sanitized}"

    @property
    def relax_wt_mismatch_resolved(self) -> bool:
        """Whether to tolerate CSV/reference wildtype disagreement.

        Defaults to True for ``format: multidms`` and False otherwise.
        multidms defines every mutation relative to a single reference homolog,
        so a per-homolog founder sequence *legitimately* disagrees with the CSV
        wildtype column -- that is intrinsic to the format, not a data defect.
        All seven shipped multidms manifests diverge (2 sites for BA.1, 29 for
        Delta, 13 for BA.2), so strict-by-default made every one of them
        unloadable (#414).

        The other formats keep strict matching, and an explicit
        ``relax_wt_mismatch: false`` still forces it for multidms too.
        """
        if self.relax_wt_mismatch is not None:
            return self.relax_wt_mismatch
        return self.format == "multidms"

    @model_validator(mode="after")
    def validate_format_required_fields(self) -> Self:
        """Require sequence_cols/y_col for the formats that read them.

        Exempts only ``multidms``, whose CSVs have the fixed header
        wildtype,site,mutant,functional_effect and whose output column is the
        constant ``y_functional_effect``. Requiring them unconditionally made
        all seven multidms manifests fail validation and vanish from the
        registry (#989); exempting ``standard`` as well went too far -- it reads
        ``y_col`` through :attr:`output_y_col`, so omitting it there yields a
        column name that is not in the CSV and fails far from the cause.
        """
        if self.format == "multidms":
            return self
        missing = [
            name
            for name, value in (
                ("sequence_cols", self.sequence_cols),
                ("y_col", self.y_col),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"format '{self.format}' requires {', '.join(missing)}")
        return self

    @model_validator(mode="after")
    def validate_mask_fields(self) -> Self:
        """Ensure mask_positions and mask_explanation appear together."""
        has_positions = self.mask_positions is not None
        has_explanation = self.mask_explanation is not None
        if has_positions != has_explanation:
            raise ValueError(
                "mask_positions and mask_explanation must both be present or both absent"
            )
        return self

    def resolve_data_path(self) -> VerifiedPath:
        """Resolve the data path relative to the YAML file location.

        If the path is absolute, it is used directly. If relative, yaml_dir
        must be set (typically via load_functional_config()).

        Returns:
            VerifiedPath with absolute path and optional MD5 for lazy verification.

        Raises:
            ValueError: If path is relative and yaml_dir is not set.
        """
        data_path = Path(self.data.path)
        if data_path.is_absolute():
            return VerifiedPath(data_path.resolve(), self.data.md5)
        if self.yaml_dir is None:
            raise ValueError(
                f"Cannot resolve relative path '{self.data.path}': yaml_dir not set. "
                "Use absolute paths or load config via load_functional_config()."
            )
        return self.data.resolve(self.yaml_dir)

    def resolve_model_checkpoint(self) -> Optional[Path]:
        """Resolve the model checkpoint path if specified.

        If the path is absolute, it is used directly. If relative, yaml_dir
        must be set (typically via load_functional_config()).

        Returns:
            Absolute path to checkpoint, or None if not specified.

        Raises:
            FileNotFoundError: If the resolved path doesn't exist.
            ValueError: If path is relative and yaml_dir is not set.
        """
        if self.model_checkpoint is None:
            return None
        checkpoint_path = Path(self.model_checkpoint)
        if checkpoint_path.is_absolute():
            resolved = checkpoint_path.resolve()
        elif self.yaml_dir is not None:
            resolved = resolve_path(self.yaml_dir, self.model_checkpoint)
        else:
            raise ValueError(
                f"Cannot resolve relative path '{self.model_checkpoint}': yaml_dir not set. "
                "Use absolute paths or load config via load_functional_config()."
            )
        if not resolved.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {resolved}")
        return resolved

    def load(self) -> pd.DataFrame:
        """Load the dataset as a standardized DataFrame.

        Uses the appropriate format converter based on self.format, then applies
        the y_type transformation to convert y values to the standard convention:
        - Log scale (additive effects)
        - Higher values = better fitness/binding

        Returns:
            DataFrame in standardized functional data format with y values
            transformed according to y_type.

        Raises:
            ValueError: If format is unknown or required fields are missing.
        """
        # .path triggers MD5 verification if specified
        data_path = self.resolve_data_path().path

        if self.format == "flab":
            # Narrowing for the type checker, which cannot see that
            # validate_format_required_fields already rejected None here. Not a
            # defensive check -- if it ever fires, that validator has a hole.
            if self.sequence_cols is None or self.y_col is None:
                raise ValueError("flab format requires sequence_cols and y_col")
            df = from_flab(
                data_path,
                sequence_cols=self.sequence_cols,
                y_col=self.y_col,
                group_id_col=self.group_id_col,
                y_col_rename=self.y_col_rename,
            )
        elif self.format == "multidms":
            if self.reference_sequence is None:
                raise ValueError("multidms format requires reference_sequence")
            df = from_multidms(
                data_path,
                wt_sequence=self.reference_sequence,
                strict=not self.relax_wt_mismatch_resolved,
            )
        elif self.format == "standard":
            # For standard format, just read CSV directly
            # Data already uses sequence_*, y_* column prefixes
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unknown format: {self.format}")

        # Apply y_type transform to y column
        y_col_name = self.output_y_col
        if self.y_type != "fitness":
            if y_col_name not in df.columns:
                raise ValueError(
                    f"y column '{y_col_name}' not found in loaded data. "
                    f"Available columns: {list(df.columns)}"
                )
            transform = Y_TYPE_TRANSFORMS[self.y_type]
            df[y_col_name] = transform(df[y_col_name])

        return df


# ---------------------------------------------------------------------------
# Registry (delegates to unified ManifestIndex)
# ---------------------------------------------------------------------------


def load_functional_config(yaml_path: Union[str, Path]) -> FunctionalDatasetConfig:
    """Load a functional dataset configuration from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Parsed FunctionalDatasetConfig with yaml_dir set for path resolution.

    Raises:
        ValueError: If YAML is invalid or missing required fields.
        FileNotFoundError: If YAML file doesn't exist.
    """
    return load_yaml_config(
        Path(yaml_path).resolve(),
        "dasm2_functional_data_schema",
        FunctionalDatasetConfig,
    )


_DISCRIMINATOR = "dasm2_functional_data_schema"


def register_functional_dataset(name: str, config: FunctionalDatasetConfig) -> None:
    """Register a functional dataset configuration.

    Args:
        name: Unique name for the dataset.
        config: The dataset configuration.

    Raises:
        ValueError: If a dataset with this name is already registered.
    """
    get_manifest_index().register(name, _DISCRIMINATOR, config)


def get_functional_config(name: str) -> FunctionalDatasetConfig:
    """Get a registered functional dataset configuration.

    Args:
        name: Name of the registered dataset.

    Returns:
        The dataset configuration.

    Raises:
        KeyError: If no dataset with this name is registered.
    """
    config: FunctionalDatasetConfig = get_manifest_index().get_typed(
        name, _DISCRIMINATOR
    )
    return config


def load_functional_dataset(name: str) -> pd.DataFrame:
    """Load a functional dataset by name.

    Args:
        name: Name of the registered dataset.

    Returns:
        DataFrame in standardized functional data format.

    Raises:
        KeyError: If no dataset with this name is registered.
    """
    config = get_functional_config(name)
    return config.load()


def list_functional_datasets() -> List[str]:
    """List all registered functional datasets.

    Returns:
        List of dataset names.
    """
    return get_manifest_index().list_names(_DISCRIMINATOR)
