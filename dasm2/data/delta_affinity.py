"""Delta-affinity dataset for binding affinity change prediction.

Loads standardized functional data (output of FunctionalDatasetConfig.load())
and prepares it for training with DeltaAffinityHead. Each item is a single
sequence with its associated delta-affinity target.

Data Format:
    Expects a DataFrame in the data-central functional-data-v1 schema:
    - sequence_* columns: amino acid sequences for each chain
    - y_* column: target response variable (binding affinity, fitness, etc.)
    - Optional: group_id, is_reference columns (stored but not used in forward)

    Multi-chain sequences (e.g., heavy + light) are concatenated with
    AA_SEPARATOR ('^') to form a single tokenized sequence.

Usage:
    # Old eager interface (pre-tokenizes everything in __init__):
    from dasm2.data.delta_affinity import DeltaAffinityDataset
    dataset = DeltaAffinityDataset(df=df, sequence_cols=[...], y_col="y_binding")

    # Unified interface (returns DatasetBundle):
    from dasm2.data.delta_affinity import create_delta_affinity_dataset
    bundle = create_delta_affinity_dataset(df, [...], "y_binding")
    loader = DataLoader(bundle.dataset, batch_size=32, collate_fn=bundle.collator)
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

from dasm2.data.collators import TransformCollator
from dasm2.data.dataset_utils import (
    clean_chain_series,
    concat_chains_with_separator,
    split_df_train_val,
)
from dasm2.data.evolutionary_dataset import (
    DatasetBundle,
    TransformDataset,
    batched_layout_masks_for_samples,
    make_worker_init_fn,
)
from dasm2.data.functional_formats import GROUP_ID_COL, IS_REFERENCE_COL
from dasm2.data.functional_registry import get_functional_config
from dasm2.sequence.constants import AA_AMBIG_IDX
from dasm2.sequence.layout import padding_sentinel_for
from dasm2.tensors_of_seqs import tokenize_aa_sequences, tokenize_single_aa_sequence

DELTA_AFFINITY_KEY = "delta_affinity"
BINARY_TARGET_KEY = "binary_target"
CLASS_LABEL_KEY = "class_label"


def normalize_sequence_cols(cols: Optional[List[str]]) -> List[str]:
    """Ensure sequence column names have the 'sequence_' prefix.

    Functional data manifests are inconsistent: some use bare names
    (e.g. "heavy") while others already include the "sequence_" prefix
    (e.g. "sequence_heavy"). This normalizes both forms.

    Args:
        cols: Raw sequence column names from a manifest config. ``None`` for
            ``format: multidms`` and ``standard`` manifests, which declare no
            ``sequence_cols`` (#989) -- delta-affinity needs them, so that is
            an error here rather than an empty list.

    Returns:
        Column names with "sequence_" prefix guaranteed.

    Raises:
        ValueError: If *cols* is None or empty.
    """
    if not cols:
        raise ValueError(
            "sequence_cols is required for delta-affinity data but the manifest "
            "declares none; only 'format: flab' manifests carry it."
        )
    return [col if col.startswith("sequence_") else f"sequence_{col}" for col in cols]


class DeltaAffinityDataset(torch.utils.data.Dataset):
    """Dataset for delta-affinity prediction from sequence embeddings.

    .. deprecated::
        This eager dataset pre-tokenizes all sequences in __init__. Use
        :func:`create_delta_affinity_dataset` (unified pipeline) instead, which
        tokenizes per-sample and pads to batch-max. This class will be removed
        once all heads are migrated to the unified pipeline.

    Each item represents a single sequence (possibly multi-chain) with its
    associated delta-affinity target value. Sequences are tokenized and
    padded to uniform length. A mask excludes padding, separator, and
    ambiguous tokens from pooling.

    Multi-chain sequences are concatenated with AA_SEPARATOR ('^'),
    e.g., "EVQL...^DIQM..." for heavy+light chains.

    Memory Model:
        All sequences are tokenized upfront into a single tensor.
        __getitem__ returns slices (views) of precomputed tensors.
    """

    row_filter_digest: Optional[str] = None
    """Never row-filtered; part of the ``CacheableDataset`` protocol (#1672)."""

    def __init__(
        self,
        df: pd.DataFrame,
        sequence_cols: List[str],
        y_col: str,
        nickname: str = "unnamed_delta_affinity",
    ) -> None:
        """Initialize from a standardized functional DataFrame.

        Args:
            df: DataFrame with sequence_* and y_* columns (output of
                FunctionalDatasetConfig.load() or equivalent).
            sequence_cols: Column names containing amino acid sequences to
                concatenate with AA_SEPARATOR (e.g., ["sequence_heavy", "sequence_light"]).
            y_col: Column name for the target variable (e.g., "y_binding").
            nickname: Human-readable identifier for this dataset.

        Raises:
            ValueError: If required columns are missing, y_col contains NaN,
                or any sequence is empty.
        """
        self.nickname = nickname

        _validate_delta_affinity_df(df, sequence_cols, y_col)

        sequences = concat_chains_with_separator(df, sequence_cols)

        # Compute max length for padding
        max_len = max(len(s) for s in sequences)

        # Tokenize all sequences
        # Named aa_parents_idxs for compatibility with CoreModel.forward()
        # which expects this key for encoder input
        self.aa_parents_idxs = tokenize_aa_sequences(sequences, max_len)  # [N, max_len]

        # Mask: True where token is a standard amino acid (< AA_AMBIG_IDX)
        # This excludes X (ambiguous), ^ (separator), and padding
        self.masks = self.aa_parents_idxs < AA_AMBIG_IDX  # [N, max_len]

        # Store all N targets (sliced to scalar per item in __getitem__)
        self.delta_affinities = torch.tensor(df[y_col].values, dtype=torch.float32)

        # Store optional metadata columns (not used in forward pass)
        self.group_ids: Optional[pd.Series] = (
            df[GROUP_ID_COL].copy() if GROUP_ID_COL in df.columns else None
        )
        self.is_reference: Optional[pd.Series] = (
            df[IS_REFERENCE_COL].copy() if IS_REFERENCE_COL in df.columns else None
        )

        self.num_sequences = len(df)

    def __len__(self) -> int:
        """Return number of sequences in the dataset."""
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single sequence with its delta-affinity target.

        Args:
            idx: Index of the sequence (0 to num_sequences - 1).

        Returns:
            Dict containing:
            - aa_parents_idxs: LongTensor[seq_len] - tokenized amino acid indices
            - mask: BoolTensor[seq_len] - True for valid AA positions
            - loss_mask: BoolTensor[seq_len] - same as mask (all valid positions)
            - delta_affinity: FloatTensor (scalar) - target value
        """
        mask = self.masks[idx]
        return {
            "aa_parents_idxs": self.aa_parents_idxs[idx],
            "mask": mask,
            # loss_mask mirrors mask for interface compatibility with the unified
            # pipeline collator; cloned to avoid aliasing.
            "loss_mask": mask.clone(),
            "delta_affinity": self.delta_affinities[idx],
        }

    @classmethod
    def from_manifest(
        cls,
        name: str,
        y_col: str,
        nickname: Optional[str] = None,
    ) -> "DeltaAffinityDataset":
        """Create dataset from a registered functional data manifest.

        Convenience method that loads the data via the functional registry
        and uses the manifest's sequence_cols configuration.

        Args:
            name: Registered dataset name (e.g., "koenig_binding").
            y_col: Target column name (e.g., "y_binding").
            nickname: Optional override for dataset nickname (defaults to name).

        Returns:
            DeltaAffinityDataset loaded from the manifest.

        Raises:
            KeyError: If no dataset with this name is registered.
        """
        config = get_functional_config(name)
        df = config.load()

        sequence_cols = normalize_sequence_cols(config.sequence_cols)

        missing = [c for c in sequence_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Expected columns {sequence_cols} after loading manifest '{name}', "
                f"but {missing} not found. Available: {list(df.columns)}"
            )

        return cls(
            df=df,
            sequence_cols=sequence_cols,
            y_col=y_col,
            nickname=nickname or name,
        )


# ---------------------------------------------------------------------------
# Unified pipeline factory functions
# ---------------------------------------------------------------------------


def _validate_delta_affinity_df(
    df: pd.DataFrame,
    sequence_cols: List[str],
    y_col: str,
) -> None:
    """Validate DataFrame for DeltaAffinity dataset creation.

    Same validations as DeltaAffinityDataset.__init__: missing columns,
    NaN targets, empty sequences, empty DataFrame.

    Raises:
        ValueError: If validation fails.
    """
    missing_seq = [c for c in sequence_cols if c not in df.columns]
    if missing_seq:
        raise ValueError(
            f"Missing sequence columns: {missing_seq}. "
            f"Available columns: {list(df.columns)}"
        )
    if y_col not in df.columns:
        raise ValueError(
            f"Missing target column '{y_col}'. Available columns: {list(df.columns)}"
        )

    nan_count = df[y_col].isna().sum()
    if nan_count > 0:
        raise ValueError(
            f"Target column '{y_col}' has {nan_count} NaN values. "
            f"Filter these before creating the dataset."
        )

    for col in sequence_cols:
        empty_mask = clean_chain_series(df, col).str.len() == 0
        if empty_mask.any():
            empty_rows = df.index[empty_mask].tolist()
            raise ValueError(
                f"Empty sequence found in column '{col}' at rows: {empty_rows}"
            )

    if len(df) == 0:
        raise ValueError("DataFrame is empty, cannot create dataset.")


def create_delta_affinity_dataset(
    df: pd.DataFrame,
    sequence_cols: List[str],
    y_col: str,
    nickname: str = "unnamed_delta_affinity",
    target_key: str = DELTA_AFFINITY_KEY,
    target_dtype: torch.dtype = torch.float32,
) -> DatasetBundle:
    """Create a dataset bundle for DeltaAffinity / pooled-head prediction.

    Returns a DatasetBundle containing the dataset, collator, and worker_init_fn.

    Args:
        df: DataFrame with sequence_* and y_* columns.
        sequence_cols: Column names containing amino acid sequences
            (e.g., ["sequence_heavy", "sequence_light"]).
        y_col: Column name for the target variable (e.g., "y_binding").
        nickname: Human-readable identifier for this dataset.
        target_key: Batch-dict key to store the target under. Defaults to
            ``"delta_affinity"`` (DeltaAffinityHead / PooledRegressionHead).
            Use ``"binary_target"`` for PooledMultiLabelHead and
            ``"class_label"`` for PooledClassificationHead.
        target_dtype: dtype for the target tensor. Use ``torch.float32`` for
            regression / BCE, ``torch.long`` for multi-class classification.

    Returns:
        DatasetBundle with dataset, collator, and worker_init_fn.

    Raises:
        ValueError: If required columns are missing, y_col contains NaN,
            any sequence is empty, or DataFrame is empty.
    """
    _validate_delta_affinity_df(df, sequence_cols, y_col)

    sequences = concat_chains_with_separator(df, sequence_cols)
    targets = df[y_col].tolist()

    num_segments = len(sequence_cols)
    padding_sentinel = padding_sentinel_for(num_segments)

    # Fused tokenize + batched layout/masks (concat→compute→split pattern)
    aa_tokens = [tokenize_single_aa_sequence(seq) for seq in sequences]

    output_keys = frozenset(
        {"aa_parents_idxs", "layout", "mask", "loss_mask", target_key}
    )

    # Build sample dicts with pre-tokenized tensors
    samples: List[Dict[str, Any]] = []
    for i in range(len(sequences)):
        samples.append({"aa_parents_idxs": aa_tokens[i]})

    # Batched layout + masks (replaces per-sample ComputeLayout + ComputeMasks)
    batched_layout_masks_for_samples(samples, num_segments)

    # Add targets
    for i in range(len(sequences)):
        samples[i][target_key] = torch.tensor(targets[i], dtype=target_dtype)

    # Filter to output keys
    samples = [{k: v for k, v in s.items() if k in output_keys} for s in samples]

    target_pad = 0 if target_dtype in (torch.long, torch.int32, torch.int64) else 0.0
    collator = TransformCollator(
        tensor_specs={
            "aa_parents_idxs": (AA_AMBIG_IDX, "codon"),
            "layout": (padding_sentinel, "codon"),
            "mask": (False, "codon"),
            "loss_mask": (False, "codon"),
            target_key: (target_pad, "scalar"),
        }
    )
    dataset = TransformDataset(
        aa_parents=sequences,
        precomputed_samples=samples,
        nickname=nickname,
    )
    return DatasetBundle(
        dataset=dataset,
        collator=collator,
        worker_init_fn=make_worker_init_fn(),
    )


def create_delta_affinity_dataset_from_manifest(
    name: str,
    y_col: str,
    nickname: Optional[str] = None,
    target_key: str = DELTA_AFFINITY_KEY,
    target_dtype: torch.dtype = torch.float32,
) -> DatasetBundle:
    """Create a DeltaAffinity / pooled-head dataset bundle from a registered manifest.

    Convenience function that loads data via the functional registry
    and uses the manifest's sequence_cols configuration.

    Args:
        name: Registered dataset name (e.g., "koenig_binding").
        y_col: Target column name (e.g., "y_binding").
        nickname: Optional override for dataset nickname (defaults to name).
        target_key: Batch-dict key for the target. See
            :func:`create_delta_affinity_dataset`.
        target_dtype: dtype for the target tensor. See
            :func:`create_delta_affinity_dataset`.

    Returns:
        DatasetBundle with dataset, collator, and worker_init_fn.

    Raises:
        KeyError: If no dataset with this name is registered.
    """
    config = get_functional_config(name)
    df = config.load()
    sequence_cols = normalize_sequence_cols(config.sequence_cols)
    return create_delta_affinity_dataset(
        df=df,
        sequence_cols=sequence_cols,
        y_col=y_col,
        nickname=nickname or name,
        target_key=target_key,
        target_dtype=target_dtype,
    )


def load_delta_affinity_train_val(
    data_path: str,
    seed: int,
    train_fraction: float = 0.8,
    target_key: str = DELTA_AFFINITY_KEY,
    target_dtype: torch.dtype = torch.float32,
) -> Tuple[DatasetBundle, DatasetBundle]:
    """Load DeltaAffinity dataset with train/val split.

    Uses group-aware splitting when the DataFrame contains a ``group_id``
    column (splits on whole groups to avoid data leakage). Falls back to
    random sample-level splitting when no groups are present.

    Args:
        data_path: One of:

            - ``"path/to/file.csv:y_col"`` — direct CSV; ``y_col`` is required.
            - ``"path/to/manifest.yaml"`` — manifest declares the y column
              via ``y_col`` / ``y_col_rename``; no suffix needed.
              A ``:y_col`` suffix is still accepted and overrides the
              manifest default.
            - ``"manifest_nickname:y_col"`` — registered manifest with
              explicit y column.
        seed: Random seed for reproducible split.
        train_fraction: Fraction for training set (default: 0.8).

    Returns:
        Tuple of (train_bundle, val_bundle) DatasetBundles.

    Raises:
        ValueError: If data_path format is invalid.
    """
    # Split "source:y_col" only if there's a colon. Manifest YAML paths may
    # omit the suffix and let the manifest provide the y column.
    if ":" in data_path:
        manifest_name, y_col = data_path.rsplit(":", 1)
    else:
        manifest_name, y_col = data_path, None

    source_path = Path(manifest_name)
    is_csv_path = source_path.suffix == ".csv" or manifest_name.endswith(".csv.gz")
    is_yaml_path = source_path.suffix in (".yaml", ".yml")
    if is_csv_path:
        if y_col is None:
            raise ValueError(
                f"CSV data paths require a ':y_col' suffix, got: {data_path}"
            )
        if not source_path.exists():
            raise FileNotFoundError(f"CSV file not found: {source_path}")
        df = pd.read_csv(source_path)
        sequence_cols = [c for c in df.columns if c.startswith("sequence_")]
        if not sequence_cols:
            raise ValueError(
                f"No sequence_* columns found in {source_path}. "
                f"Expected columns like sequence_heavy, sequence_light."
            )
    elif is_yaml_path:
        from dasm2.data.functional_registry import load_functional_config

        if not source_path.exists():
            raise FileNotFoundError(f"Manifest YAML not found: {source_path}")
        func_config = load_functional_config(source_path)
        df = func_config.load()
        sequence_cols = normalize_sequence_cols(func_config.sequence_cols)
        if y_col is None:
            y_col = func_config.output_y_col
    else:
        if y_col is None:
            raise ValueError(
                f"Registered-manifest data paths require a ':y_col' suffix, "
                f"got: {data_path}"
            )
        func_config = get_functional_config(manifest_name)
        df = func_config.load()
        sequence_cols = normalize_sequence_cols(func_config.sequence_cols)

    # Drop NaN targets before splitting.
    nan_count = df[y_col].isna().sum()
    if nan_count > 0:
        warnings.warn(
            f"Dropped {nan_count}/{len(df)} rows with NaN in '{y_col}' "
            f"from {manifest_name}.",
            stacklevel=2,
        )
        df = df.dropna(subset=[y_col]).reset_index(drop=True)

    train_df, val_df = split_df_train_val(df, seed=seed, train_fraction=train_fraction)

    train_bundle = create_delta_affinity_dataset(
        df=train_df,
        sequence_cols=sequence_cols,
        y_col=y_col,
        nickname=f"{manifest_name}_train",
        target_key=target_key,
        target_dtype=target_dtype,
    )
    val_bundle = create_delta_affinity_dataset(
        df=val_df,
        sequence_cols=sequence_cols,
        y_col=y_col,
        nickname=f"{manifest_name}_val",
        target_key=target_key,
        target_dtype=target_dtype,
    )
    return train_bundle, val_bundle
