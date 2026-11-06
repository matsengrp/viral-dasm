"""Segment handlers for different protein types.

Key Design Requirements:
- Separation of concerns: Segment logic separated from dataset logic
- Composition over inheritance: Handlers are attributes, not base classes
- Extensibility: Easy to add new segment configurations
- Reusability: Same handler works for both parent-only and evolutionary datasets
"""

import warnings
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import pandas as pd
import torch

from dasm2.molevol.branch_length_init import (
    BranchLengthInitStrategy,
    ProportionalStrategy,
)
from dasm2.neutral.base import RatesCSPNeutralModel
from dasm2.sequence.constants import NT_SEPARATOR
from dasm2.utils.tensor import stack_heterogeneous


class SegmentHandler:
    """Base handler for single-segment proteins.

    Handles segment-specific operations:
    - Converting segment columns to unified parent_sequence/child_sequence format
    - Splitting predictions back into per-segment results
    - Providing segment-specific rate adjustments

    The segment_suffixes attribute defines column name suffixes for each segment.
    For single-segment proteins (default), use an empty string ("",) - no special
    prefix or suffix is needed, just use the base column names like "parent_sequence".
    For multi-segment proteins, suffixes include the leading underscore (e.g.,
    "_heavy", "_light") and are appended to base names: parent_heavy, parent_light.
    Column names are formed as: {prefix}{suffix}
    """

    # Segment configuration: default is single segment with no special suffix
    segment_suffixes: Tuple[str, ...] = ("",)

    # Pluggable strategy for initializing branch lengths from mutation frequency.
    # None indicates no calibrated strategy for this data type.
    # When available, enables fast branch length initialization from mutation frequency.
    # Can be set per-dataset via parquet metadata key "bl_init_strategy" (JSON)
    # or legacy "branch_length_slope" (float).
    # NOTE: Class attribute — replace via assignment (handler.bl_init_strategy = ...),
    # never mutate the strategy object in-place (it is shared across instances).
    bl_init_strategy: Optional[BranchLengthInitStrategy] = None

    def get_segment_names(self) -> Tuple[str, ...]:
        """Get display names for segments (strips leading underscores).

        Returns:
            Tuple of segment display names
        """
        # Strip leading underscores for display
        return tuple(
            seg.lstrip("_") if seg else "sequence" for seg in self.segment_suffixes
        )

    def get_num_segments(self) -> int:
        """Return number of segments.

        Returns:
            Number of segments (e.g., 2 for antibodies, 1 for single-segment)
        """
        return len(self.segment_suffixes)

    def get_segment_index(self, name: str) -> int:
        """Map segment name to 1-based index in layout tensor.

        Args:
            name: Segment display name (e.g., "heavy", "light", "sequence")

        Returns:
            1-based segment index for use with layout tensor

        Raises:
            ValueError: If segment name not recognized
        """
        segment_names = self.get_segment_names()
        try:
            zero_based_idx = segment_names.index(name)
        except ValueError:
            raise ValueError(
                f"Unknown segment name '{name}'. "
                f"Valid segment names: {list(segment_names)}"
            )
        return zero_based_idx + 1  # Layout uses 1-based indexing

    def validate_segment_index(self, idx: int) -> None:
        """Validate that a segment index is within bounds.

        Args:
            idx: 1-based segment index to validate

        Raises:
            ValueError: If segment index out of bounds
        """
        num_segments = self.get_num_segments()
        if idx < 1 or idx > num_segments:
            raise ValueError(
                f"Segment index {idx} out of bounds. "
                f"Valid range: 1 to {num_segments} "
                f"(segments: {list(self.get_segment_names())})"
            )

    def get_rate_adjustments(self) -> torch.Tensor:
        """Get per-segment rate multipliers for neutral model.

        Returns:
            Tensor of rate multipliers per segment (default: [1.0] for single segment)
        """
        return torch.tensor([1.0])

    def map_to_segments(
        self,
        sequence: str,
        compute_fn: Callable[[str], Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ) -> List[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        """Apply computation function to each segment.

        Splits sequence by NT_SEPARATOR (if present) and applies compute_fn to each
        non-empty segment. No special handling for single-segment sequences.

        Args:
            sequence: Input sequence that may contain NT_SEPARATOR (^^^)
            compute_fn: Function mapping segment string -> tensor or tuple of tensors

        Returns:
            List of results, one per segment
        """
        segments = sequence.split(NT_SEPARATOR)
        segment_results = [compute_fn(segment) for segment in segments]
        return segment_results

    def extract_segment(self, sequence: str, segment_idx: int) -> str:
        """Extract a single segment from a combined sequence.

        Splits sequence by NT_SEPARATOR and returns the segment at the given index.
        Returns empty string if the index is out of bounds.

        Args:
            sequence: Input sequence that may contain NT_SEPARATOR (^^^)
            segment_idx: Zero-based index of the segment to extract

        Returns:
            The segment string at the given index, or empty string if not found
        """
        segments = sequence.split(NT_SEPARATOR)
        return segments[segment_idx] if segment_idx < len(segments) else ""

    def get_segment_column_name(self, base_name: str, segment_idx: int) -> str:
        """Get segment-specific column name for a base column name.

        Combines base column name with segment suffix to form the full column name.
        For single-segment handlers with empty suffix, returns the base name unchanged.

        Args:
            base_name: Base column name (e.g., "parent", "child", "nt_rates")
            segment_idx: Zero-based segment index

        Returns:
            Segment-specific column name (e.g., "parent_heavy", "nt_rates_light")

        Example:
            >>> handler = AntibodySegmentHandler()  # suffixes: ("_heavy", "_light")
            >>> handler.get_segment_column_name("parent", 0)
            'parent_heavy'
            >>> handler.get_segment_column_name("parent", 1)
            'parent_light'
        """
        return f"{base_name}{self.segment_suffixes[segment_idx]}"

    def strip_segment_suffix(self, column_name: str, segment_idx: int) -> str:
        """Strip segment suffix from column name to get generic name.

        Removes the segment-specific suffix from a column name. If the suffix is empty
        (single-segment handler) or the column doesn't end with the suffix, returns
        the original column name unchanged.

        Args:
            column_name: Segment-specific column name (e.g., "parent_heavy", "cdr1_codon_start_light")
            segment_idx: Zero-based segment index

        Returns:
            Generic column name (e.g., "parent", "cdr1_codon_start")

        Example:
            >>> handler = AntibodySegmentHandler()  # suffixes: ("_heavy", "_light")
            >>> handler.strip_segment_suffix("parent_heavy", 0)
            'parent'
            >>> handler.strip_segment_suffix("cdr1_codon_start_light", 1)
            'cdr1_codon_start'
        """
        suffix = self.segment_suffixes[segment_idx]
        if suffix and column_name.endswith(suffix):
            return column_name[: -len(suffix)]
        return column_name

    def prepare_neutral_rates_and_csps(
        self,
        sequences_df: pd.DataFrame,
        batch_size: int = 4096,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate and combine precomputed neutral rates and CSPs from DataFrame columns.

        Reads from segment-specific columns (e.g., nt_rates_heavy, nt_csps_heavy for
        AntibodySegmentHandler) and combines them with separator values.

        Only processes segments that have both rates and csps columns. This supports
        partial segment data (e.g., heavy-only antibody datasets).

        Args:
            sequences_df: DataFrame with segment-specific nt_rates and nt_csps columns.
                         For single-segment: nt_rates, nt_csps (lists of tensors)
                         For antibodies: nt_rates_heavy, nt_csps_heavy (and optionally
                         nt_rates_light, nt_csps_light for paired data)
            batch_size: Maximum sequences to process at once (default: 4096).
                Reduces peak memory for large datasets by processing in chunks.

        Returns:
            Tuple of (rates_batch, csps_batch) as stacked tensors with separator values:
                rates_batch: Tensor of shape (batch_size, max_nt_len) with zero-padding
                csps_batch: Tensor of shape (batch_size, max_nt_len, 4) with zero-padding

        Raises:
            ValueError: If no precomputed columns are found for any segment

        Example:
            >>> # For antibody data with precomputed columns:
            >>> rates_batch, csps_batch = handler.prepare_neutral_rates_and_csps(df)
        """
        # 1. Find which segments have precomputed columns
        available_segments = []
        for seg_idx, suffix in enumerate(self.segment_suffixes):
            rates_col = f"nt_rates{suffix}"
            csps_col = f"nt_csps{suffix}"
            if rates_col in sequences_df.columns and csps_col in sequences_df.columns:
                available_segments.append((seg_idx, suffix))

        if not available_segments:
            expected_cols = [f"nt_rates{s}, nt_csps{s}" for s in self.segment_suffixes]
            raise ValueError(
                f"No precomputed neutral model columns found. "
                f"Expected at least one pair of columns from: {expected_cols}. "
                f"Available columns: {sorted(sequences_df.columns)}"
            )

        # 2. Extract segment tensors from columns
        segment_rates = []
        segment_csps = []

        for seg_idx, suffix in available_segments:
            rates_col = f"nt_rates{suffix}"
            csps_col = f"nt_csps{suffix}"

            # Get lists of tensors from DataFrame columns (already rate-adjusted)
            rates_list = sequences_df[rates_col].tolist()
            csps_list = sequences_df[csps_col].tolist()

            segment_rates.append(rates_list)
            segment_csps.append(csps_list)

        # 3. Re-combine per sequence with separator values, processing in batches
        # to avoid memory explosion with large datasets
        num_sequences = len(sequences_df)
        rates_batches = []
        csps_batches = []

        for batch_start in range(0, num_sequences, batch_size):
            batch_end = min(batch_start + batch_size, num_sequences)
            rates_per_seq = []
            csps_per_seq = []

            for seq_idx in range(batch_start, batch_end):
                # Collect rates/csps for this sequence across available segments
                seq_segment_rates = [
                    segment_rates[i][seq_idx] for i in range(len(available_segments))
                ]
                seq_segment_csps = [
                    segment_csps[i][seq_idx] for i in range(len(available_segments))
                ]

                # Combine with separators
                combined_rates = self._combine_segments_with_separators(
                    seq_segment_rates, fill=1.0
                )
                combined_csps = self._combine_segments_with_separators(
                    seq_segment_csps, fill=0.0
                )

                rates_per_seq.append(combined_rates)
                csps_per_seq.append(combined_csps)

            # Stack this batch and add to batch lists
            rates_batches.append(stack_heterogeneous(rates_per_seq, pad_value=0.0))
            csps_batches.append(stack_heterogeneous(csps_per_seq, pad_value=0.0))
            del rates_per_seq, csps_per_seq  # Explicit cleanup

        # 4. Concatenate all batches (they may have different max lengths per batch,
        # so we need to pad to the global max)
        if len(rates_batches) == 1:
            return rates_batches[0], csps_batches[0]

        # Find global max lengths
        max_rates_len = max(b.shape[1] for b in rates_batches)
        max_csps_len = max(b.shape[1] for b in csps_batches)

        # Pad each batch to global max and concatenate
        padded_rates = []
        padded_csps = []
        for rates_b, csps_b in zip(rates_batches, csps_batches):
            if rates_b.shape[1] < max_rates_len:
                pad_size = max_rates_len - rates_b.shape[1]
                rates_b = torch.nn.functional.pad(rates_b, (0, pad_size), value=0.0)
            if csps_b.shape[1] < max_csps_len:
                pad_size = max_csps_len - csps_b.shape[1]
                csps_b = torch.nn.functional.pad(csps_b, (0, 0, 0, pad_size), value=0.0)
            padded_rates.append(rates_b)
            padded_csps.append(csps_b)

        rates_batch = torch.cat(padded_rates, dim=0)
        csps_batch = torch.cat(padded_csps, dim=0)
        del padded_rates, padded_csps, rates_batches, csps_batches  # Cleanup

        return rates_batch, csps_batch

    def _combine_segments_with_separators(
        self, segment_tensors: List[torch.Tensor], fill: float
    ) -> torch.Tensor:
        """Combine segment tensors with separator values between them.

        Args:
            segment_tensors: List of tensors, one per segment
            fill: Value to use for separator positions (1.0 for rates, 0.0 for csps)

        Returns:
            Combined tensor with separator values inserted between segments
        """
        if len(segment_tensors) == 1:
            # Single segment - no separators needed
            return segment_tensors[0]

        # Multi-segment - need separators
        num_separator_rows = len(NT_SEPARATOR)  # 3 for "^^^"

        # Build result by interleaving segments with separators
        result_parts = []
        for i, segment_tensor in enumerate(segment_tensors):
            result_parts.append(segment_tensor)
            if i < len(segment_tensors) - 1:
                # Add separator (except after last segment)
                separator_shape = list(segment_tensor.shape)
                separator_shape[0] = num_separator_rows
                separator = torch.full(separator_shape, fill)
                result_parts.append(separator)

        return torch.cat(result_parts, dim=0)

    def is_format_compatible(
        self,
        columns: Set[str],
        base_column_names: Tuple[str, ...],
    ) -> bool:
        """Check if DataFrame columns match this handler's segment format.

        Uses the same logic as validate_segment_format but returns bool
        instead of raising. Used for format auto-detection in factory.

        Args:
            columns: Set of column names from DataFrame
            base_column_names: Expected base column names (e.g., ("parent", "child"))

        Returns:
            True if columns contain at least one complete suffix set

        Example:
            >>> handler = AntibodySegmentHandler()  # suffixes: ("_heavy", "_light")
            >>> handler.is_format_compatible(
            ...     {"parent_heavy", "child_heavy", "parent_light", "child_light"},
            ...     ("parent", "child")
            ... )
            True
        """
        # Find which segment indices have the first base column present
        found_segment_indices = [
            seg_idx
            for seg_idx in range(len(self.segment_suffixes))
            if self.get_segment_column_name(base_column_names[0], seg_idx) in columns
        ]

        if not found_segment_indices:
            return False

        # Check all base column names exist for found segments
        for seg_idx in found_segment_indices:
            for base_name in base_column_names:
                if self.get_segment_column_name(base_name, seg_idx) not in columns:
                    return False

        return True

    def validate_segment_columns(
        self,
        available_columns: Set[str],
        base_column_names: Tuple[str, ...] = ("parent", "child"),
    ) -> Set[str]:
        """Validate segment columns and return the names of any missing ones.

        Checks that at least one complete segment is present (all base
        columns for that segment exist).  Segments that are entirely absent
        are acceptable — their column names are returned so callers can fill
        defaults (empty strings for DataFrames, ``None`` for row dicts).

        This is the shared validation logic used by both the eager CSV path
        (``validate_segment_format``) and the streaming Parquet path
        (``make_parquet_row_converter``).

        Args:
            available_columns: Column names present in the data source.
            base_column_names: Base column names to check
                (e.g., ``("parent", "child")``).

        Returns:
            Set of column names that are absent from *available_columns*
            but belong to absent segments.

        Raises:
            ValueError: If no complete segment is found, or if a partially
                present segment is missing some base columns.
        """
        num_segments = len(self.segment_suffixes)

        if not self.is_format_compatible(available_columns, base_column_names):
            # Provide detailed error message
            found_segment_indices = [
                seg_idx
                for seg_idx in range(num_segments)
                if self.get_segment_column_name(base_column_names[0], seg_idx)
                in available_columns
            ]

            if not found_segment_indices:
                required_cols = ", ".join(
                    self.get_segment_column_name(base_column_names[0], seg_idx)
                    for seg_idx in range(num_segments)
                )
                raise ValueError(
                    f"Dataset validation failed: missing required columns.\n"
                    f"Expected at least one of: {required_cols}\n"
                    f"Found columns: {sorted(available_columns)}\n"
                    f"Check that your data has the correct column naming convention."
                )
            else:
                for seg_idx in found_segment_indices:
                    for base_name in base_column_names:
                        col_name = self.get_segment_column_name(base_name, seg_idx)
                        if col_name not in available_columns:
                            suffix = self.segment_suffixes[seg_idx]
                            raise ValueError(
                                f"Dataset expected {col_name} column because another "
                                f"column with suffix {suffix} was observed.\n"
                                f"Found columns: {sorted(available_columns)}"
                            )

        # Identify missing columns from absent segments
        missing = set()
        for base_name in base_column_names:
            for seg_idx in range(len(self.segment_suffixes)):
                col_name = self.get_segment_column_name(base_name, seg_idx)
                if col_name not in available_columns:
                    missing.add(col_name)

        return missing

    def validate_segment_format(
        self,
        pcp_df: pd.DataFrame,
        base_column_names: Tuple[str, ...] = ("parent", "child"),
    ) -> pd.DataFrame:
        """Validate and prepare dataframe with segment columns.

        Column names are formed as {base_name}{suffix} where suffixes come from segment_suffixes.
        For example, with base_column_names=("parent", "child") and segment_suffixes=("_heavy", "_light"),
        expects columns: parent_heavy, child_heavy, parent_light, child_light.

        Args:
            pcp_df: DataFrame with segment-specific columns
            base_column_names: Tuple of base column names (e.g., ("parent", "child"))

        Returns:
            DataFrame with all segment columns present (missing ones filled with "")

        Raises:
            ValueError: If required columns missing or sequences too short
        """
        # Make a copy to avoid modifying the original
        df = pcp_df.copy()

        missing_cols = self.validate_segment_columns(set(df.columns), base_column_names)

        # Add missing columns with empty strings; fill NaN in existing ones
        for base_name in base_column_names:
            for seg_idx in range(len(self.segment_suffixes)):
                col_name = self.get_segment_column_name(base_name, seg_idx)
                if col_name in missing_cols:
                    df[col_name] = ""
                else:
                    # Fill NaN values in existing columns with empty strings
                    # This happens when concatenating dataframes with different columns
                    df[col_name] = df[col_name].fillna("")

        # Validate that all base column names have equal length for each segment
        # For evolutionary datasets this ensures parent/child pairs match
        # For multi-sequence datasets this ensures all sequences in each row have equal length
        if len(base_column_names) >= 2:
            for seg_idx in range(len(self.segment_suffixes)):
                # Check all base columns against the first one (transitivity)
                reference_col = self.get_segment_column_name(
                    base_column_names[0], seg_idx
                )
                reference_lengths = df[reference_col].str.len().fillna(0)

                for base_name in base_column_names[1:]:
                    compare_col = self.get_segment_column_name(base_name, seg_idx)
                    compare_lengths = df[compare_col].str.len().fillna(0)

                    if not (reference_lengths == compare_lengths).all():
                        mismatched = df[
                            reference_lengths != compare_lengths
                        ].index.tolist()
                        suffix = self.segment_suffixes[seg_idx]
                        raise ValueError(
                            f"All sequences with suffix '{suffix}' must have equal length across base columns. "
                            f"Mismatch between {reference_col} and {compare_col}. "
                            f"Mismatched rows: {mismatched}"
                        )

        # Validate that each row has at least 3 nucleotides total across all segments
        first_base_name = base_column_names[0]
        total_lengths: pd.Series[int] = pd.Series([0] * len(df), dtype=int)
        for seg_idx in range(len(self.segment_suffixes)):
            col_name = self.get_segment_column_name(first_base_name, seg_idx)
            total_lengths = total_lengths + df[col_name].str.len().fillna(0).astype(int)

        # Check minimum length (at least one codon = 3 nucleotides)
        min_length = total_lengths.min()

        if min_length < 3:
            raise ValueError(
                f"At least one sequence has fewer than 3 nucleotides (minimum: {min_length}). "
                "Each sequence must contain at least one codon."
            )

        return df

    def get_column_names(self, basename: str) -> List[str]:
        """Return column names for each segment, with basename"""
        return [f"{basename}{suffix}" for suffix in self.segment_suffixes]

    def prepare_sequences(
        self,
        pcp_df: pd.DataFrame,
        base_column_names: Tuple[str, ...] = ("parent", "child"),
    ) -> pd.DataFrame:
        """Convert segment-specific columns to combined sequence columns.

        Reads from input columns ({base_name}{suffix}) and creates output columns named by
        the base_column_names, combining segments with ^^^ separators. For single-segment datasets
        with empty suffix, copies columns directly without combining.

        Args:
            pcp_df: DataFrame with segment-specific columns (e.g., parent_heavy, child_heavy)
            base_column_names: Tuple of base column names that become output column names
                              (e.g., ("parent", "child") creates columns "parent" and "child")

        Returns:
            DataFrame with combined sequence columns named by base_column_names, preserving all
            original metadata columns
        """
        # Validate first
        validated_df = self.validate_segment_format(
            pcp_df,
            base_column_names=base_column_names,
        )

        # Start with copy of original DataFrame to preserve all metadata
        sequences_df = validated_df.copy()

        # Process each base column name (e.g., parent, child)
        for base_name in base_column_names:
            combined_sequences = []

            for _, row in validated_df.iterrows():
                # Extract all segments for this base column name
                segments = []
                for suffix in self.segment_suffixes:
                    col_name = f"{base_name}{suffix}"
                    segments.append(row.get(col_name, ""))

                # Combine using separator
                combined_seq = self._combine_segments(*segments)
                combined_sequences.append(combined_seq)

            # Add combined column to DataFrame
            sequences_df[base_name] = combined_sequences

        return sequences_df

    def _combine_segments(self, *segment_seqs: Optional[str]) -> str:
        """Combine segments with ^^^ separator.

        For single-segment datasets, returns the sequence as-is.
        For multi-segment datasets, joins with ^^^ separators:
        - Two segments: segment1^^^segment2
        - Three segments: segment1^^^segment2^^^segment3

        Args:
            *segment_seqs: Segment sequences in order (may be None, which will be interpreted
                as empty strings)

        Returns:
            Combined sequence with ^^^ separators (multi-segment) or plain sequence (single-segment)
        """
        # Normalize empty/None values to empty strings
        normalized = []
        for seq in segment_seqs:
            if seq is None or pd.isna(seq):
                normalized.append("")
            else:
                normalized.append(str(seq))

        return NT_SEPARATOR.join(normalized)


class AntibodySegmentHandler(SegmentHandler):
    """Handler for antibody heavy/light chain sequences.

    Provides antibody-specific functionality:
    - Heavy and light chain segment handling
    - Antibody rate adjustments (heavy=1.0, light=0.63)
    - V gene family column creation and validation
    - Factory methods for heavy/light format

    Uses segment suffixes with leading underscores: "_heavy", "_light"
    Combined with prefixes like "parent" or "child", forms column names:
    parent_heavy, child_heavy, parent_light, child_light
    """

    # Define segment suffixes for antibody datasets (with leading underscores)
    segment_suffixes = ("_heavy", "_light")

    # Empirical slope from trained DASM on v1jaffeCC+v1tangCC (n=742,377, R²=0.974).
    # This slope applies to segments with rate adjustment factor 1.0 (heavy chains).
    # For other segments, the effective slope is slope / rate_adjustment.
    # See issue #140 for derivation details.
    bl_init_strategy: BranchLengthInitStrategy = ProportionalStrategy(slope=1.76)  # type: ignore[assignment]

    # Expected V gene family prefixes per segment (first 3 chars of v_gene before '-')
    V_GENE_FAMILY_PREFIXES: Dict[str, FrozenSet[str]] = {
        "_heavy": frozenset({"IGH"}),
        "_light": frozenset({"IGK", "IGL"}),
    }

    def get_rate_adjustments(self) -> torch.Tensor:
        """Antibody-specific rate adjustments.

        Returns:
            Tensor [1.0, 0.63] for heavy and light chains
        """
        return torch.tensor([1.0, 0.63])

    def add_v_family_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create v_family_{suffix} columns from v_gene_{suffix} by splitting on '-'.

        Only processes segments that have v_gene columns present in the DataFrame.
        The v_family is extracted as the portion before the first '-' in the v_gene.

        Args:
            df: DataFrame with v_gene_{suffix} columns (e.g., v_gene_heavy, v_gene_light)

        Returns:
            DataFrame with added v_family_{suffix} columns

        Example:
            >>> handler = AntibodySegmentHandler()
            >>> df = pd.DataFrame({"v_gene_heavy": ["IGHV1-2*01", "IGHV3-11*02"]})
            >>> result = handler.add_v_family_columns(df)
            >>> result["v_family_heavy"].tolist()
            ['IGHV1', 'IGHV3']
        """
        df = df.copy()

        for suffix in self.segment_suffixes:
            v_gene_col = f"v_gene{suffix}"
            v_family_col = f"v_family{suffix}"

            if v_gene_col in df.columns:
                # Extract family by splitting on '-' and taking the first part
                df[v_family_col] = df[v_gene_col].str.split("-").str[0]

        return df

    def validate_v_gene_families(self, df: pd.DataFrame) -> None:
        """Validate V gene family prefixes match expected families per segment.

        Checks that the first 3 characters of each v_family column match the
        expected prefixes for that segment (IGH for heavy, IGK/IGL for light).

        Only validates segments that have v_family columns present.

        Args:
            df: DataFrame with v_family_{suffix} columns

        Raises:
            ValueError: If unexpected V gene families found in any segment
        """
        for suffix in self.segment_suffixes:
            v_family_col = f"v_family{suffix}"

            if v_family_col not in df.columns:
                continue

            expected_prefixes = self.V_GENE_FAMILY_PREFIXES[suffix]

            # Get first 3 characters of each v_family value
            family_prefixes = df[v_family_col].str[:3]

            # Find rows with unexpected prefixes (excluding empty/NaN values)
            valid_mask = (
                family_prefixes.isin(expected_prefixes)
                | family_prefixes.isna()
                | (family_prefixes == "")
            )
            invalid_mask = ~valid_mask

            if invalid_mask.any():
                unexpected_families = df.loc[invalid_mask, v_family_col].unique()
                segment_name = suffix.lstrip("_")
                raise ValueError(
                    f"Unexpected {segment_name} chain V gene families: {unexpected_families}. "
                    f"Expected prefixes: {sorted(expected_prefixes)}"
                )


#: The antibody layout's segment display names, in slot order: heavy first,
#: light second. Derived from :attr:`AntibodySegmentHandler.segment_suffixes`
#: rather than restated, so the ordering convention has exactly one home --
#: the handler that already encodes it for column naming, rate adjustments and
#: sequence combination.
#:
#: Note this is the *layout*, not a training record. The antibody layout
#: always has both slots: training data fills an absent chain with ``""``
#: rather than omitting its separator (see
#: :meth:`SegmentHandler._combine_segments`), so a model trained on the heavy
#: chain alone still has a light slot, and one trained on the light chain
#: alone still receives its heavy sequence in slot 0.
ANTIBODY_SEGMENT_NAMES: Tuple[str, ...] = AntibodySegmentHandler().get_segment_names()


def is_antibody_segment_layout(names: Sequence[str]) -> bool:
    """Whether these segment display names belong to the antibody layout.

    Args:
        names: Segment display names, e.g. ``("heavy", "light")`` or
            ``("protein",)``.

    Returns:
        ``True`` if any name is an antibody chain name.
    """
    return any(name in ANTIBODY_SEGMENT_NAMES for name in names)


def canonical_segment_order(names: Sequence[str]) -> Tuple[str, ...]:
    """Order segment display names as their handler lays the segments out.

    By convention the heavy chain is the first segment and the light chain the
    second, so ``("light", "heavy")`` and ``("heavy", "light")`` name the same
    paired model and both canonicalize to the latter. Ordering is a spelling
    variant, not an error, so it is normalized rather than rejected; a set of
    names that could not describe any layout -- a repeat, or an antibody chain
    mixed with a non-antibody segment -- is rejected instead.

    Args:
        names: Segment display names in any order.

    Returns:
        The same names in the layout's slot order.

    Raises:
        ValueError: If ``names`` is empty, repeats a segment, mixes antibody
            chains with non-antibody segments, or names more than one
            non-antibody segment (the generic handler has a single slot).
    """
    ordered = tuple(names)
    if not ordered:
        raise ValueError(
            "No segments given, so there is no layout to order them in. Pass "
            "the segments the model was trained on, e.g. ('heavy', 'light') "
            "or ('protein',)."
        )
    duplicates = sorted({name for name in ordered if ordered.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"Segments {list(ordered)} repeat {duplicates}. Each segment "
            "occupies one slot of the layout, so a repeat has no meaning -- "
            "an antibody pairs a heavy chain with a light chain, never two "
            "of the same."
        )
    if not is_antibody_segment_layout(ordered):
        if len(ordered) > 1:
            raise ValueError(
                f"Segments {list(ordered)} name no known multi-segment "
                f"layout. The antibody layout is {list(ANTIBODY_SEGMENT_NAMES)}; "
                "every other handler is single-segment."
            )
        return ordered
    unknown = [name for name in ordered if name not in ANTIBODY_SEGMENT_NAMES]
    if unknown:
        raise ValueError(
            f"Segments {list(ordered)} mix antibody chains with {unknown}. "
            f"The antibody layout holds exactly {list(ANTIBODY_SEGMENT_NAMES)}."
        )
    return tuple(name for name in ANTIBODY_SEGMENT_NAMES if name in ordered)


def layout_segment_names(names: Sequence[str]) -> Tuple[str, ...]:
    """Every slot of the layout ``names`` belong to, in slot order.

    ``names`` records which segments a *model* was trained on; this returns
    which slots that model's layout *has*. For the antibody layout the answer
    is always both chains, whether or not ``names`` mentions both: a
    heavy-only model is still handed a light slot, and a paired dataset still
    varies in it.

    The distinction matters to any caller that has to hold the non-scored
    segments fixed. Deriving the held set from ``names`` makes a heavy-only
    model blind to a light-chain substitution -- it would see nothing to hold
    -- and admit doubly-mutated rows into a single-mutant analysis.

    Args:
        names: Segment display names, in any order.

    Returns:
        The layout's segment display names in slot order.

    Raises:
        ValueError: If ``names`` describes no layout (see
            :func:`canonical_segment_order`).
    """
    canonical_segment_order(names)
    if is_antibody_segment_layout(names):
        return ANTIBODY_SEGMENT_NAMES
    return tuple(names)


def primary_segment_of(names: Sequence[str]) -> str:
    """The segment occupying slot 0 of the layout ``names`` belong to.

    The single definition of "primary segment", so every caller that names
    slot 0 agrees on it. For the antibody layout this is always the heavy
    chain, *whether or not* ``names`` contains it: a light-only-trained model
    still receives its heavy sequence in slot 0, so reading the first element
    of ``names`` would label that slot "light" and collide with the real light
    slot.

    Args:
        names: Segment display names, in any order.

    Returns:
        The primary segment's display name.

    Raises:
        ValueError: If ``names`` describes no layout (see
            :func:`canonical_segment_order`).
    """
    return layout_segment_names(names)[0]


def segment_sequence_column(segment: str) -> str:
    """Column holding ``segment``'s amino-acid sequence in a functional dataset.

    Routed through the owning handler for antibody chains, so the suffix comes
    from :attr:`AntibodySegmentHandler.segment_suffixes` rather than from a
    second copy of ``"_heavy"``/``"_light"`` spelled out here.

    Functional datasets suffix every segment, including single-segment ones
    (``sequence_protein``), which is why the generic handler's empty suffix is
    not used for the non-antibody case -- that suffix is the parent/child
    training-column convention, a different naming scheme.

    Args:
        segment: Segment display name, e.g. ``"heavy"`` or ``"protein"``.

    Returns:
        The column name, e.g. ``"sequence_light"``.
    """
    if segment in ANTIBODY_SEGMENT_NAMES:
        handler = AntibodySegmentHandler()
        index = handler.get_segment_index(segment) - 1
        return handler.get_segment_column_name("sequence", index)
    return f"sequence_{segment}"


# Standard segment handlers for auto-detection
# Order matters: most specific handlers first
STANDARD_SEGMENT_HANDLERS = [
    AntibodySegmentHandler(),  # Checks for _heavy/_light suffixes
    SegmentHandler(),  # Generic single-segment (empty suffix)
]


def add_neutral_model_outputs_to_pcp_df(
    pcp_df: pd.DataFrame,
    neutral_model: RatesCSPNeutralModel,
    segment_handler: SegmentHandler,
    base_column_name: str = "parent",
) -> pd.DataFrame:
    """Add neutral model rate and CSP columns to a parent-child pair DataFrame.

    This is a data preparation function that computes neutral model outputs
    and adds them as columns to the DataFrame. The resulting DataFrame can
    then be used to create an EvolutionaryDataset.

    Column naming follows the pattern nt_rates{suffix} and nt_csps{suffix}, where
    suffix values come from segment_handler.segment_suffixes. Only adds columns
    for segments that exist in the input DataFrame.

    Follows the netam pattern of `add_shm_model_outputs_to_pcp_df`.

    Note:
        This function requires a RatesCSPNeutralModel (not a general AbstractNeutralModel)
        because it calls compute_rates_and_csps(). GTRCodonNeutralModel inherits from
        RatesCSPNeutralModel and implements this method, so it is fully supported here.

    Args:
        pcp_df: DataFrame with segment-specific sequence columns
               (e.g., parent_heavy, parent_light for antibodies)
        neutral_model: Model with compute_rates_and_csps method (ThriftySHM, TabularRates, etc.)
        segment_handler: Handler defining segment suffixes and rate adjustments
        base_column_name: Base column name containing sequences (default: "parent")

    Returns:
        DataFrame with added nt_rates{suffix} and nt_csps{suffix} columns
        for each segment in segment_handler.segment_suffixes that exists in the input

    Example:
        >>> from dasm2.data.segment_handlers import (
        ...     AntibodySegmentHandler, add_neutral_model_outputs_to_pcp_df
        ... )
        >>> from dasm2.model_catalog import resolve_model
        >>> from dasm2.neutral.thrifty import ThriftySHMModel
        >>>
        >>> handler = AntibodySegmentHandler()
        >>> neutral_model = ThriftySHMModel.from_catalog(
        ...     "ThriftyHumV0.2-59", resolve_model("ThriftyHumV0.2-59")
        ... )
        >>> df_with_neutral_outputs = add_neutral_model_outputs_to_pcp_df(
        ...     pcp_df, neutral_model, handler
        ... )
        >>> # df_with_neutral_outputs now has nt_rates_heavy, nt_csps_heavy,
        >>> # nt_rates_light, nt_csps_light columns (if light chain exists)
    """
    df = pcp_df.copy()

    # Get rate adjustments for each segment
    rate_adjustments = segment_handler.get_rate_adjustments()

    # Compute and add columns only for segments that exist in the DataFrame
    for seg_idx, suffix in enumerate(segment_handler.segment_suffixes):
        col_name = f"{base_column_name}{suffix}"

        # Skip segments that don't exist in the input DataFrame
        if col_name not in df.columns:
            continue

        rates_col = f"nt_rates{suffix}"
        csps_col = f"nt_csps{suffix}"

        # Get sequences for this segment, filling NaN with empty string
        # This handles combined datasets where some rows lack certain segments
        sequences = df[col_name].fillna("").tolist()

        # Compute rates and CSPs
        rates_list, csps_list = neutral_model.compute_rates_and_csps(sequences)

        # Apply rate adjustment for this segment
        adjusted_rates = [r * rate_adjustments[seg_idx] for r in rates_list]

        # Add columns to DataFrame (cast to Any to satisfy mypy with list[Tensor])
        df[rates_col] = adjusted_rates  # type: ignore[assignment]
        df[csps_col] = csps_list  # type: ignore[assignment]

    return df


def detect_compatible_handler(
    columns: Set[str],
    base_column_names: Tuple[str, ...],
) -> SegmentHandler:
    """Detect which segment handler is compatible with the given columns.

    Iterates through STANDARD_SEGMENT_HANDLERS and returns the first handler
    whose required columns are present for the given base column names.

    Args:
        columns: Set of DataFrame column names.
        base_column_names: Base column names to check for (e.g., ("parent", "child")
            for evolutionary datasets, ("sequence",) for parent-only).

    Returns:
        First compatible handler from STANDARD_SEGMENT_HANDLERS.

    Raises:
        ValueError: If no compatible handler found.
    """
    compatible_handlers = [
        handler
        for handler in STANDARD_SEGMENT_HANDLERS
        if handler.is_format_compatible(columns, base_column_names)
    ]

    if not compatible_handlers:
        example_columns = []
        for handler in STANDARD_SEGMENT_HANDLERS:
            for base_name in base_column_names:
                for suffix in handler.segment_suffixes:
                    example_columns.append(f"{base_name}{suffix}")

        raise ValueError(
            f"No compatible segment handler found.\n"
            f"Expected columns matching one of these patterns:\n"
            f"  - {example_columns[:4]} (examples)\n"
            f"Found columns: {sorted(columns)}\n"
            f"Check that your DataFrame has the correct column naming convention."
        )

    if len(compatible_handlers) > 1:
        handler_names = [type(h).__name__ for h in compatible_handlers]
        warnings.warn(
            f"Multiple segment handlers compatible: {handler_names}. "
            f"Using first match: {handler_names[0]}",
            UserWarning,
        )

    return compatible_handlers[0]


def compute_segment_masks(
    layout_batch: torch.Tensor,
    segment_handler: SegmentHandler,
) -> Dict[str, torch.Tensor]:
    """Compute per-segment boolean masks from a layout tensor.

    Extracts the segment mask logic previously in
    SequenceDataset.get_all_segment_positions() as a standalone function.

    Args:
        layout_batch: Layout tensor of shape ``[N, L]`` or ``[L]`` where values
            encode segment membership (1-indexed by segment).
        segment_handler: Handler providing segment names.

    Returns:
        Dict mapping segment name to boolean mask tensor with same shape
        as layout_batch.
    """
    return {
        name: (layout_batch == (idx + 1))
        for idx, name in enumerate(segment_handler.get_segment_names())
    }


def validate_parent_columns(columns: Iterable[str], handler: SegmentHandler) -> None:
    """Require parents for every present child, allowing wholly absent segments."""
    available = set(columns)
    handler.validate_segment_columns(available, ("parent",))
    for parent, child in zip(
        handler.get_column_names("parent"), handler.get_column_names("child")
    ):
        if child in available and parent not in available:
            raise ValueError(
                f"Missing required parent column {parent!r} for {child!r}; received {sorted(available)!r}"
            )


def parent_sequences_from_frame(
    frame: pd.DataFrame, handler: Optional[SegmentHandler] = None
) -> Iterator[str]:
    """Yield parents using the dataset's validated segment assembly.

    Entirely absent segments (heavy-only/light-only data) follow the dataset
    handler's policy. A missing parent for a present child, or no parent column
    at all, raises before a coverage scan can silently see zero contexts.
    """
    if handler is None:
        handler = detect_compatible_handler(set(frame.columns), ("parent",))
    validate_parent_columns(frame.columns, handler)
    validated = handler.validate_segment_format(frame, ("parent",))
    columns = handler.get_column_names("parent")
    for values in validated[columns].itertuples(index=False, name=None):
        yield handler._combine_segments(*values)
