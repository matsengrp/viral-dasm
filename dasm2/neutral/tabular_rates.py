"""Tabular rates neutral mutation model.

Implements context-dependent (3-mer motif) neutral mutation rates from tabular data,
with optional secondary structure effects (paired/unpaired status).

Supports two data formats:
1. **With secondary structure** (e.g., SARS-CoV-2): Rates vary by motif AND paired/unpaired
   status. Requires both rates_table.csv (with `unpaired` column) and secondary_struct.csv.
2. **Without secondary structure** (e.g., influenza): Rates vary by motif only.
   Requires only rates_table.csv (no `unpaired` column), no secondary structure file.

The neutral rate computation follows:
1. Build lookup table from rates CSV:
   - With structure: {unpaired: {motif: {wt: {mut: rate}}}}
   - Without structure: {motif: {wt: {mut: rate}}}
2. For each position, get 3-mer context (and optionally secondary structure status)
3. Look up raw substitution rates for each possible mutation
4. Normalize to get conditional substitution probabilities (CSPs)
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field

from dasm2.neutral.base import RatesCSPNeutralModel
from dasm2.sequence.conversion import _NT_CHAR_TO_IDX
from dasm2.serialization.zip_io import (
    data_archive_exists,
    read_data_archive,
    write_data_archive,
)

if TYPE_CHECKING:
    from dasm2.config import TabularRatesNeutralModelConfig

logger = logging.getLogger(__name__)

BASES = ["A", "C", "G", "T"]

# Type aliases for the nested motif rates structures
# Simple map (without secondary structure): {motif: {wt: {mut: rate}}}
SimpleMotifRatesMap = Dict[str, Dict[str, Dict[str, float]]]

# Structured map (with secondary structure): {unpaired: {motif: {wt: {mut: rate}}}}
StructuredMotifRatesMap = Dict[bool, Dict[str, Dict[str, Dict[str, float]]]]

# Union type for either format
MotifRatesMap = Union[SimpleMotifRatesMap, StructuredMotifRatesMap]

# File names in the archive
_RATES_TABLE_FILENAME = "rates_table.csv"
_SECONDARY_STRUCT_FILENAME = "secondary_struct.csv"


class TabularRatesArchiveMetadata(BaseModel):
    """Metadata for TabularRatesNeutralModel archive.

    Stored as metadata.yml in the zip archive alongside the CSV data files.
    """

    model_type: str = Field(
        default="TabularRatesNeutralModel",
        description="Model class identifier for validation",
    )
    model_name: str = Field(
        default="TabularRatesNeutral",
        description="Human-readable model name",
    )
    archive_version: int = Field(
        default=1,
        description="Archive format version for future compatibility",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the model and its provenance",
    )

    has_secondary_structure: bool = Field(
        default=True,
        description="Whether model uses secondary structure (unpaired) data. "
        "If True, archive contains secondary_struct.csv and rates_table has "
        "'unpaired' column. If False, no secondary structure file.",
    )


def _build_rates_for_motif(motif_group_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Build mutation rates dictionary for a single motif.

    Args:
        motif_group_df: DataFrame rows for a single motif, with columns: mut_type, rate

    Returns:
        Dictionary mapping wildtype base -> mutant base -> rate
    """
    wt_to_mut_rates: Dict[str, Dict[str, float]] = {}
    for _, row in motif_group_df.iterrows():
        wt: str = row["mut_type"][0]
        mut: str = row["mut_type"][1]
        if wt not in wt_to_mut_rates:
            wt_to_mut_rates[wt] = {}
        wt_to_mut_rates[wt][mut] = float(row["rate"])
    return wt_to_mut_rates


def _build_motif_rates_from_df(
    rates_df: pd.DataFrame,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build motif rates dictionary from a rates DataFrame.

    Used for both single-unpaired-status groups (structured) and entire
    tables (simple, no secondary structure).

    Args:
        rates_df: DataFrame with columns: mut_type, motif, rate

    Returns:
        Dictionary mapping motif -> wildtype base -> mutant base -> rate
    """
    motif_rates: Dict[str, Dict[str, Dict[str, float]]] = {}
    for motif_key, motif_group in rates_df.groupby("motif"):
        motif = str(motif_key)
        motif_rates[motif] = _build_rates_for_motif(motif_group)
    return motif_rates


def has_secondary_structure_column(rates_table_df: pd.DataFrame) -> bool:
    """Check if rates table has secondary structure (unpaired) column.

    Args:
        rates_table_df: DataFrame with mutation rates data

    Returns:
        True if 'unpaired' column is present, False otherwise
    """
    return "unpaired" in rates_table_df.columns


def create_motif_sub_rates_map(
    rates_table_df: pd.DataFrame,
) -> MotifRatesMap:
    """Create a mapping of mutation rates based on motif and optionally unpaired status.

    Detects whether the rates table has an 'unpaired' column and creates the
    appropriate map structure:
    - With 'unpaired' column: {unpaired: {motif: {wt: {mut: rate}}}}
    - Without 'unpaired' column: {motif: {wt: {mut: rate}}}

    Args:
        rates_table_df: DataFrame with columns: mut_type, motif, rate
                        and optionally: unpaired

    Returns:
        A nested dictionary with raw, un-normalized rates.
        Structure depends on whether 'unpaired' column is present.
    """
    if has_secondary_structure_column(rates_table_df):
        # With secondary structure: {unpaired: {motif: {wt: {mut: rate}}}}
        motif_rates_map: StructuredMotifRatesMap = {}
        for unpaired_key, unpaired_group in rates_table_df.groupby("unpaired"):
            unpaired = bool(unpaired_key)
            motif_rates_map[unpaired] = _build_motif_rates_from_df(unpaired_group)
        return motif_rates_map
    else:
        # Without secondary structure: {motif: {wt: {mut: rate}}}
        return _build_motif_rates_from_df(rates_table_df)


def is_structured_motif_map(motif_rates_map: MotifRatesMap) -> bool:
    """Check if motif rates map has secondary structure (unpaired keyed).

    Args:
        motif_rates_map: Either SimpleMotifRatesMap or StructuredMotifRatesMap

    Returns:
        True if map is structured (has bool keys), False if simple (has string keys)
    """
    if not motif_rates_map:
        raise ValueError("Cannot determine map type from empty motif_rates_map")
    first_key = next(iter(motif_rates_map.keys()))
    return isinstance(first_key, bool)


def _get_motif_rates(
    motif_rates_map: MotifRatesMap,
    motif: str,
    unpaired_status: Optional[bool] = None,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Get mutation rates for a motif, handling both map types.

    Args:
        motif_rates_map: Either SimpleMotifRatesMap or StructuredMotifRatesMap
        motif: The 3-mer motif to look up
        unpaired_status: Required for structured maps, ignored for simple maps

    Returns:
        Rates dict {wt: {mut: rate}} if motif found, None otherwise
    """
    if is_structured_motif_map(motif_rates_map):
        # Structured map: {unpaired: {motif: {wt: {mut: rate}}}}
        # Cast to proper type for mypy
        structured_map: StructuredMotifRatesMap = motif_rates_map  # type: ignore[assignment]
        if unpaired_status is None:
            raise ValueError(
                "unpaired_status required for structured motif map with secondary structure"
            )
        if (
            unpaired_status in structured_map
            and motif in structured_map[unpaired_status]
        ):
            return structured_map[unpaired_status][motif]
        return None
    else:
        # Simple map: {motif: {wt: {mut: rate}}}
        # Cast to proper type for mypy
        simple_map: SimpleMotifRatesMap = motif_rates_map  # type: ignore[assignment]
        if motif in simple_map:
            return simple_map[motif]
        return None


def _motif_not_found_error(
    motif: str, idx: int, curr_base: str, unpaired_status: Optional[bool]
) -> ValueError:
    """Create a ValueError for a motif not found in the rates table."""
    if unpaired_status is not None:
        return ValueError(
            f"Motif '{motif}' with unpaired status {unpaired_status} "
            f"not found in rates table. "
            f"Sequence position: {idx}, current base: {curr_base}. "
            f"This indicates incomplete rate data coverage."
        )
    return ValueError(
        f"Motif '{motif}' not found in rates table. "
        f"Sequence position: {idx}, current base: {curr_base}. "
        f"This indicates incomplete rate data coverage."
    )


def sub_rates_of_seq_site(
    seq: str,
    idx: int,
    motif_rates_map: MotifRatesMap,
    unpaired_status: Optional[bool] = None,
    has_secondary_structure: Optional[bool] = None,
) -> torch.Tensor:
    """Calculate substitution rates for a specific site based on context.

    For maps with secondary structure, unpaired_status is required to look up
    the appropriate rate. For simple maps (no secondary structure), unpaired_status
    is ignored.

    Args:
        seq: The nucleotide sequence
        idx: Index of the site in the sequence
        motif_rates_map: Mapping from motif to mutation rates (with or without
                         unpaired structure)
        unpaired_status: Whether the site is unpaired in RNA secondary structure.
                        Required for structured maps, optional for simple maps.
        has_secondary_structure: Whether the map has secondary structure. If None,
                                detected from map keys at runtime.

    Returns:
        Substitution rates tensor of shape (4,) for this site
    """
    curr_base = seq[idx]
    if has_secondary_structure is None:
        has_secondary_structure = is_structured_motif_map(motif_rates_map)

    # For first and last position, average over possible contexts
    if idx == 0 or idx == len(seq) - 1:
        sub_rates = torch.zeros(4, dtype=torch.float32)
        if idx == 0:
            possible_motifs = [prev + curr_base + seq[idx + 1] for prev in BASES]
        else:
            possible_motifs = [
                seq[idx - 1] + curr_base + next_base for next_base in BASES
            ]

        valid_motifs = 0
        for motif in possible_motifs:
            motif_data = _get_motif_rates(motif_rates_map, motif, unpaired_status)
            if motif_data is not None and curr_base in motif_data:
                valid_motifs += 1
                for i, mut_base in enumerate(BASES):
                    if mut_base in motif_data[curr_base]:
                        sub_rates[i] += motif_data[curr_base][mut_base]
            else:
                raise _motif_not_found_error(motif, idx, curr_base, unpaired_status)

        sub_rates = sub_rates / valid_motifs
    else:
        motif = seq[idx - 1 : idx + 2]
        sub_rates = torch.zeros(4, dtype=torch.float32)

        motif_data = _get_motif_rates(motif_rates_map, motif, unpaired_status)
        if motif_data is not None and curr_base in motif_data:
            for i, mut_base in enumerate(BASES):
                if mut_base in motif_data[curr_base]:
                    sub_rates[i] = motif_data[curr_base][mut_base]
        else:
            raise _motif_not_found_error(motif, idx, curr_base, unpaired_status)
    return sub_rates


def sub_rates_of_seq(
    seq: str,
    motif_rates_map: MotifRatesMap,
    unpaired_map: Optional[Dict[int, bool]] = None,
    has_secondary_structure: Optional[bool] = None,
) -> torch.Tensor:
    """Calculate substitution rates for an entire sequence.

    For maps with secondary structure, unpaired_map is required to look up
    the appropriate rate per position. For simple maps (no secondary structure),
    unpaired_map should be None.

    Args:
        seq: The nucleotide sequence
        motif_rates_map: Mapping from motif to mutation rates (with or without
                         unpaired structure)
        unpaired_map: Mapping from position (0-indexed) to unpaired status.
                     Required for structured maps, should be None for simple maps.
        has_secondary_structure: Whether the map has secondary structure. If None,
                                detected from map keys at runtime.

    Returns:
        2D tensor of substitution rates, shape (sequence_length, 4)
    """
    seq_len = len(seq)
    all_sub_rates = torch.zeros((seq_len, 4), dtype=torch.float32)
    if has_secondary_structure is None:
        has_secondary_structure = is_structured_motif_map(motif_rates_map)

    for i in range(seq_len):
        if has_secondary_structure:
            if unpaired_map is None:
                raise ValueError(
                    "unpaired_map required for motif rates map with secondary structure"
                )
            if i not in unpaired_map:
                raise ValueError(
                    f"Position {i} not found in secondary structure data (unpaired_map). "
                    f"Sequence length is {seq_len} but structure data may be incomplete."
                )
            unpaired_status: Optional[bool] = unpaired_map[i]
        else:
            unpaired_status = None

        site_sub_rates = sub_rates_of_seq_site(
            seq, i, motif_rates_map, unpaired_status, has_secondary_structure
        )
        all_sub_rates[i] = site_sub_rates

    return all_sub_rates


def site_rates_of_sub_rates(sub_rates: torch.Tensor) -> torch.Tensor:
    """Compute site-specific rates by summing the substitution rates for each site.

    Args:
        sub_rates: 2D tensor of substitution rates, shape (sequence_length, 4)

    Returns:
        1D tensor of site-specific rates, shape (sequence_length,)
    """
    return torch.sum(sub_rates, dim=1)


def csps_of_sub_rates_and_site_rates(
    sub_rates: torch.Tensor,
    site_rates: torch.Tensor,
) -> torch.Tensor:
    """Normalize substitution rates to get conditional substitution probabilities (CSPs).

    Args:
        sub_rates: 2D tensor of substitution rates, shape (sequence_length, 4)
        site_rates: 1D tensor of site-specific rates, shape (sequence_length,)

    Returns:
        2D tensor of normalized CSPs, shape (sequence_length, 4)

    Raises:
        ValueError: If any site has zero total rate
    """
    if not torch.all(site_rates > 0):
        zero_positions = torch.where(site_rates == 0)[0].tolist()
        raise ValueError(
            f"Site rates contain zero values at positions: {zero_positions}. "
            f"This indicates invalid mutation rate data - all sites must have non-zero rates."
        )

    reshaped_rates = site_rates.reshape(-1, 1)
    csps = sub_rates / reshaped_rates

    return csps


# ---------------------------------------------------------------------------
# Vectorized lookup tables for batch computation
# ---------------------------------------------------------------------------

# ASCII byte → base index: A=0, C=1, G=2, T=3, else=-1.
# Derived from _NT_CHAR_TO_IDX but with N mapped to -1 (not 4), because
# motif indices must stay in 0..63 (4^3 - 1) for the rate lookup tables.
_BASE_TO_IDX = _NT_CHAR_TO_IDX.copy()
_BASE_TO_IDX[ord("N")] = -1


def _build_rate_lookup(
    motif_rates_map: MotifRatesMap,
) -> np.ndarray:
    """Build a numpy lookup array from the motif rates map.

    For simple maps: returns shape (64, 4) — motif_idx → sub_rates for wt base.
    For structured maps: returns shape (2, 64, 4) — (unpaired, motif_idx) → sub_rates.

    The motif index encodes the 3-mer as ``base_idx[0]*16 + base_idx[1]*4 + base_idx[2]``
    (same encoding as codon indices). Each entry stores the 4 substitution rates
    for the center base of that motif.

    Args:
        motif_rates_map: Motif rates mapping (simple or structured).

    Returns:
        Numpy array of substitution rates.
    """
    structured = is_structured_motif_map(motif_rates_map)

    if structured:
        # Shape: (2, 64, 4) — unpaired_idx × motif_idx × mut_base
        lookup = np.zeros((2, 64, 4), dtype=np.float32)
        structured_map: StructuredMotifRatesMap = motif_rates_map  # type: ignore[assignment]
        for unpaired_val, motif_dict in structured_map.items():
            ui = 1 if unpaired_val else 0
            for motif, wt_dict in motif_dict.items():
                b = [_BASE_TO_IDX[ord(c)] for c in motif]
                if any(x < 0 for x in b):
                    continue
                midx = int(b[0]) * 16 + int(b[1]) * 4 + int(b[2])
                center = motif[1]
                if center in wt_dict:
                    for mi, mut_base in enumerate(BASES):
                        if mut_base in wt_dict[center]:
                            lookup[ui, midx, mi] = wt_dict[center][mut_base]
        return lookup
    else:
        # Shape: (64, 4) — motif_idx × mut_base
        lookup = np.zeros((64, 4), dtype=np.float32)
        simple_map: SimpleMotifRatesMap = motif_rates_map  # type: ignore[assignment]
        for motif, wt_dict in simple_map.items():
            b = [_BASE_TO_IDX[ord(c)] for c in motif]
            if any(x < 0 for x in b):
                continue
            midx = int(b[0]) * 16 + int(b[1]) * 4 + int(b[2])
            center = motif[1]
            if center in wt_dict:
                for mi, mut_base in enumerate(BASES):
                    if mut_base in wt_dict[center]:
                        lookup[midx, mi] = wt_dict[center][mut_base]
        return lookup


def _vectorized_sub_rates(
    seq: str,
    rate_lookup: np.ndarray,
    unpaired_arr: Optional[np.ndarray],
) -> np.ndarray:
    """Compute substitution rates for a sequence using vectorized numpy lookups.

    Args:
        seq: Nucleotide sequence string.
        rate_lookup: Precomputed rate array from ``_build_rate_lookup``.
            Shape (64, 4) for simple or (2, 64, 4) for structured.
        unpaired_arr: Boolean array of unpaired status per position, or None
            for non-structured models.

    Returns:
        Array of shape (seq_len, 4) with substitution rates per position.
    """
    seq_len = len(seq)
    if seq_len == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if seq_len < 3:
        raise ValueError(
            f"Sequence length must be >= 3 for motif-based rate computation, "
            f"got {seq_len}. A 3-mer context requires at least 3 bases."
        )
    arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    base_idxs = _BASE_TO_IDX[arr]  # shape (seq_len,)

    # Compute motif indices for interior positions (1..seq_len-2)
    # motif_idx = prev*16 + center*4 + next
    motif_idxs = (
        base_idxs[:-2].astype(np.int32) * 16
        + base_idxs[1:-1].astype(np.int32) * 4
        + base_idxs[2:].astype(np.int32)
    )

    structured = rate_lookup.ndim == 3
    sub_rates = np.zeros((seq_len, 4), dtype=np.float32)

    if structured:
        if unpaired_arr is None:
            raise ValueError(
                "_vectorized_sub_rates called with structured rate_lookup "
                "but unpaired_arr is None"
            )
        # Interior positions: vectorized lookup
        ui = unpaired_arr[1:-1].astype(np.intp)
        sub_rates[1:-1] = rate_lookup[ui, motif_idxs]

        # Boundary: position 0 — average over 4 possible prev bases
        ui0 = int(unpaired_arr[0])
        for prev in range(4):
            midx = prev * 16 + int(base_idxs[0]) * 4 + int(base_idxs[1])
            sub_rates[0] += rate_lookup[ui0, midx]
        sub_rates[0] /= 4.0

        # Boundary: position seq_len-1 — average over 4 possible next bases
        ui_last = int(unpaired_arr[-1])
        for nxt in range(4):
            midx = int(base_idxs[-2]) * 16 + int(base_idxs[-1]) * 4 + nxt
            sub_rates[-1] += rate_lookup[ui_last, midx]
        sub_rates[-1] /= 4.0
    else:
        # Interior positions: vectorized lookup
        sub_rates[1:-1] = rate_lookup[motif_idxs]

        # Boundary: position 0
        for prev in range(4):
            midx = prev * 16 + int(base_idxs[0]) * 4 + int(base_idxs[1])
            sub_rates[0] += rate_lookup[midx]
        sub_rates[0] /= 4.0

        # Boundary: position seq_len-1
        for nxt in range(4):
            midx = int(base_idxs[-2]) * 16 + int(base_idxs[-1]) * 4 + nxt
            sub_rates[-1] += rate_lookup[midx]
        sub_rates[-1] /= 4.0

    return sub_rates


class TabularRatesNeutralModel(RatesCSPNeutralModel):
    """Tabular rates neutral mutation model.

    Computes neutral mutation rates based on 3-mer motif context from tabular rate
    data, with optional secondary structure (paired/unpaired status) effects.

    Supports two modes:
    1. **With secondary structure** (e.g., SARS-CoV-2): Requires rates_table with
       'unpaired' column and separate secondary_struct file mapping positions to
       unpaired status.
    2. **Without secondary structure** (e.g., influenza): Requires only rates_table
       without 'unpaired' column. No secondary structure file needed.

    Args:
        rates_table_path: Path to CSV with columns: mut_type, motif, rate
                         and optionally: unpaired
        secondary_struct_path: Path to CSV with columns: site, unpaired, sequential_site.
                              Required only if rates_table has 'unpaired' column.
        model_name: Optional name for this model instance

    Example:
        # With secondary structure (SARS-CoV-2 style)
        model = TabularRatesNeutralModel(
            rates_table_path="data/rates_table.csv",
            secondary_struct_path="data/secondary_struct.csv",
        )

        # Without secondary structure (flu style)
        model = TabularRatesNeutralModel(
            rates_table_path="data/flu_rates_table.csv",
        )

        rates_list, csps_list = model.compute_rates_and_csps(sequence)
    """

    def __init__(
        self,
        rates_table_path: Union[str, Path],
        secondary_struct_path: Optional[Union[str, Path]] = None,
        model_name: str = "TabularRatesNeutral",
    ) -> None:
        self._model_name = model_name
        self._archive_path: Optional[Path] = None  # Set by from_pretrained()

        # Load rates table
        self._rates_table_df = pd.read_csv(rates_table_path)

        # Detect if rates table has secondary structure column
        self._has_secondary_structure = has_secondary_structure_column(
            self._rates_table_df
        )

        # Load secondary structure if needed
        if self._has_secondary_structure:
            if secondary_struct_path is None:
                raise ValueError(
                    "secondary_struct_path is required when rates_table has 'unpaired' "
                    "column. Provide a secondary structure file or use a rates table "
                    "without the 'unpaired' column."
                )
            self._secondary_struct_df: Optional[pd.DataFrame] = pd.read_csv(
                secondary_struct_path
            )
            self._unpaired_map: Optional[Dict[int, bool]] = dict(
                zip(
                    self._secondary_struct_df.sequential_site,
                    self._secondary_struct_df.unpaired,
                )
            )
            # Store max covered position for validation
            self._max_struct_position: Optional[int] = self._secondary_struct_df[
                "sequential_site"
            ].max()
        else:
            if secondary_struct_path is not None:
                logger.warning(
                    "secondary_struct_path provided but rates_table has no 'unpaired' "
                    "column. Secondary structure file will be ignored."
                )
            self._secondary_struct_df = None
            self._unpaired_map = None
            self._max_struct_position = None

        # Build lookup table
        self._motif_rates_map = create_motif_sub_rates_map(self._rates_table_df)

        # Build vectorized numpy lookup arrays
        self._build_vectorized_lookups()

    def _build_vectorized_lookups(self) -> None:
        """Build vectorized numpy lookup arrays for batch computation.

        Populates ``_rate_lookup`` from the motif rates map and, for structured
        models, ``_unpaired_np`` from ``_unpaired_map``.
        """
        self._rate_lookup = _build_rate_lookup(self._motif_rates_map)

        self._unpaired_np: Optional[np.ndarray] = None
        if self._has_secondary_structure and self._unpaired_map is not None:
            max_pos = max(self._unpaired_map.keys())
            self._unpaired_np = np.zeros(max_pos + 1, dtype=np.bool_)
            for pos, val in self._unpaired_map.items():
                self._unpaired_np[pos] = val

    @property
    def name(self) -> str:
        """Human-readable identifier for this neutral model."""
        return self._model_name

    def to_config(self) -> "TabularRatesNeutralModelConfig":
        """Reconstruct TabularRatesNeutralModelConfig from neutral model state.

        If the model was loaded via catalog name, returns a config with
        ``model_name`` (portable). Otherwise returns ``model_path``.

        Returns:
            TabularRatesNeutralModelConfig with model_name or model_path set.

        Raises:
            ValueError: If model was not loaded from an archive or catalog.
        """
        from dasm2.config import TabularRatesNeutralModelConfig

        if self._catalog_name is not None:
            return TabularRatesNeutralModelConfig(model_name=self._catalog_name)

        if self._archive_path is None:
            raise ValueError(
                "Cannot create config: model was not loaded from an archive. "
                "Use save_pretrained() first to create an archive, then load it "
                "with from_pretrained() to get a model that can produce a config."
            )

        logger.info(
            "TabularRatesNeutralModel.to_config() is returning a non-portable "
            "model_path (%s). Checkpoints using this config will not be portable "
            "across machines. Use model_name (catalog-based) configs instead.",
            self._archive_path,
        )
        return TabularRatesNeutralModelConfig(model_path=self._archive_path)

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save model data to a portable zip archive.

        Creates a zip archive containing:
        - metadata.yml: Model metadata (name, version, type, has_secondary_structure)
        - rates_table.csv: Mutation rates by motif (and optionally structure)
        - secondary_struct.csv: Secondary structure data (only if has_secondary_structure)

        Args:
            path: Output path for the zip archive (should end in .zip)

        Example:
            model.save_pretrained("tabular_rates_v1.zip")
        """
        path = Path(path)
        logger.info(f"Saving TabularRatesNeutralModel to {path}")

        # Create metadata
        metadata = TabularRatesArchiveMetadata(
            model_name=self._model_name,
            has_secondary_structure=self._has_secondary_structure,
        )

        # Convert dataframes to CSV bytes
        rates_buffer = BytesIO()
        self._rates_table_df.to_csv(rates_buffer, index=False)
        rates_bytes = rates_buffer.getvalue()

        files: Dict[str, bytes] = {
            _RATES_TABLE_FILENAME: rates_bytes,
        }

        # Only include secondary structure if present
        if self._has_secondary_structure and self._secondary_struct_df is not None:
            struct_buffer = BytesIO()
            self._secondary_struct_df.to_csv(struct_buffer, index=False)
            struct_bytes = struct_buffer.getvalue()
            files[_SECONDARY_STRUCT_FILENAME] = struct_bytes

        write_data_archive(path, metadata, files)

        logger.info(f"Saved TabularRatesNeutralModel '{self._model_name}' to {path}")

    _EXPECTED_MODEL_TYPE = "TabularRatesNeutralModel"

    @classmethod
    def from_pretrained(
        cls, model_path: Union[str, Path]
    ) -> "TabularRatesNeutralModel":
        """Load model from a pretrained zip archive.

        Supports both archive formats:
        - With secondary structure: contains rates_table.csv and secondary_struct.csv
        - Without secondary structure: contains only rates_table.csv

        Args:
            model_path: Path to zip archive created by save_pretrained()

        Returns:
            Initialized TabularRatesNeutralModel with loaded data

        Raises:
            FileNotFoundError: If archive doesn't exist
            ValueError: If archive format is invalid

        Example:
            model = TabularRatesNeutralModel.from_pretrained("tabular_rates_v1.zip")
        """
        path = Path(model_path)
        logger.info(f"Loading TabularRatesNeutralModel from {path}")

        # First check if archive has rates_table (minimum requirement)
        if not data_archive_exists(path, [_RATES_TABLE_FILENAME]):
            raise FileNotFoundError(
                f"Invalid or missing archive at {path}. "
                f"Expected zip with metadata.yml and {_RATES_TABLE_FILENAME}."
            )

        # Read metadata to determine archive format, then load all needed files
        metadata, files = read_data_archive(path, [_RATES_TABLE_FILENAME])

        # Validate required metadata fields (fail fast, no silent defaults)
        model_type = metadata["model_type"]
        if model_type != cls._EXPECTED_MODEL_TYPE:
            raise ValueError(
                f"Archive has wrong model_type: {model_type}. "
                f"Expected: {cls._EXPECTED_MODEL_TYPE}."
            )

        model_name = metadata["model_name"]

        # Parse rates table
        rates_df = pd.read_csv(BytesIO(files[_RATES_TABLE_FILENAME]))

        # Determine secondary structure from metadata if present,
        # otherwise detect from the rates table columns
        if "has_secondary_structure" in metadata:
            has_secondary_structure = metadata["has_secondary_structure"]
        else:
            has_secondary_structure = has_secondary_structure_column(rates_df)

        # Load secondary structure if present
        struct_df: Optional[pd.DataFrame] = None
        if has_secondary_structure:
            try:
                _, files_with_struct = read_data_archive(
                    path, [_RATES_TABLE_FILENAME, _SECONDARY_STRUCT_FILENAME]
                )
            except KeyError:
                raise FileNotFoundError(
                    f"Archive at {path} has has_secondary_structure=True "
                    f"but is missing {_SECONDARY_STRUCT_FILENAME}."
                )
            struct_df = pd.read_csv(
                BytesIO(files_with_struct[_SECONDARY_STRUCT_FILENAME])
            )

        # Create model instance by setting internal state directly
        # (avoiding the file-based constructor)
        model = cls.__new__(cls)
        model._model_name = model_name
        model._archive_path = path  # Track source archive for to_config()
        model._has_secondary_structure = has_secondary_structure

        # Initialize from loaded data
        model._rates_table_df = rates_df
        model._motif_rates_map = create_motif_sub_rates_map(rates_df)

        if has_secondary_structure and struct_df is not None:
            model._secondary_struct_df = struct_df
            model._unpaired_map = dict(
                zip(struct_df.sequential_site, struct_df.unpaired)
            )
            model._max_struct_position = struct_df["sequential_site"].max()
        else:
            model._secondary_struct_df = None
            model._unpaired_map = None
            model._max_struct_position = None

        # Build vectorized numpy lookup arrays
        model._build_vectorized_lookups()

        logger.info(f"Loaded TabularRatesNeutralModel '{model_name}' from {path}")
        return model

    def compute_rates_and_csps(
        self,
        sequences: Union[str, List[str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute neutral rates and CSPs for one or more sequences.

        Uses vectorized numpy array lookups for motif rates instead of
        per-position Python dict lookups (~270x faster for spike-length
        sequences).

        Args:
            sequences: Single nucleotide sequence string OR list of sequences
            context: Optional context data (unused for this model)

        Returns:
            Tuple of (nt_rates_list, nt_csps_list) where each list contains
            one tensor per input sequence:
                - nt_rates_list: List of tensors, each shape (seq_length,)
                - nt_csps_list: List of tensors, each shape (seq_length, 4)

        Raises:
            ValueError: If sequence length exceeds secondary structure coverage
                       (only applies when model has secondary structure)
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        nt_rates_list: List[torch.Tensor] = []
        nt_csps_list: List[torch.Tensor] = []

        for seq in sequences:
            if self._has_secondary_structure and self._max_struct_position is not None:
                if len(seq) > self._max_struct_position + 1:
                    raise ValueError(
                        f"Sequence length ({len(seq)}) exceeds secondary structure "
                        f"coverage ({self._max_struct_position + 1} positions). "
                        f"All sequence positions must have corresponding structure data."
                    )

            unpaired_arr = (
                self._unpaired_np[: len(seq)] if self._unpaired_np is not None else None
            )
            sub_rates_np = _vectorized_sub_rates(seq, self._rate_lookup, unpaired_arr)

            # site_rates = sum over mutations, csps = normalized
            site_rates_np = sub_rates_np.sum(axis=1)
            zero_mask = site_rates_np == 0
            if zero_mask.any():
                zero_positions = np.where(zero_mask)[0].tolist()
                raise ValueError(
                    f"Site rates contain zero values at positions: {zero_positions}. "
                    f"This indicates invalid mutation rate data - all sites must "
                    f"have non-zero rates."
                )
            csps_np = sub_rates_np / site_rates_np[:, np.newaxis]

            nt_rates_list.append(torch.from_numpy(site_rates_np))
            nt_csps_list.append(torch.from_numpy(csps_np))

        return nt_rates_list, nt_csps_list
