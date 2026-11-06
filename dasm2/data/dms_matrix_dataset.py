"""Per-domain DMS stability dataset, in ``to_dms_matrix``-style ``[L, 20]`` shape.

Named after the shape it produces (one ``to_dms_matrix``-style ``L x 20``
matrix per item), matching this repo's existing shape/task-based dataset
naming (``DeltaAffinityDataset``, ``MLMDataset``, ``MPNNWhichmutDataset``),
and distinct from the existing mutation-level ``_DMSDatasetAdapter``
(``dasm2/data/evolutionary_dataset.py``).

Each item is one wild-type domain: the WT sequence, its tokenized parent
indices, an ``[L, 20]`` matrix of measured stability values, and a validity
mask -- built from a per-domain-group functional DataFrame (columns
``sequence_aa``, ``y_stability``, ``group_id``, ``is_reference``,
``cluster_id``; the last is an assumption this module consumes but does not
compute -- see issue #1295).

Wildtype cells (the ``[L, 20]`` cell whose amino acid equals the WT residue
at that position) are NaN in ``measured_matrix`` and ``False`` in
``valid_mask``, so the loss never scores them -- exactly like unmeasured
cells. This is deliberate: the WT cell is a gauge-fixed 0 on both the
experimental and model side (DESIGN.md D13, issue #1110), so it cannot
discriminate between models. The exclusion is declared here rather than
inherited from ``filter_single_mutants`` dropping the WT row. The model side
stays finite because a NaN there would poison gradients.

Deliberately an eager ``Dataset`` subclass (matching ``DeltaAffinityDataset``'s
pattern), not the newer unified ``TransformDataset``/``DatasetBundle``
pipeline ``pairwise_binding.py`` uses: that pipeline's per-sample transform
chain assumes one raw parent sequence maps to one sample, while each item
here aggregates *multiple* DataFrame rows (a domain's reference replicates
plus all its single-mutant variants) into one domain -- a groupby-then-
aggregate shape that doesn't fit the per-sample transform-chain model.
``aa_parents_idxs`` is tokenized via ``tokenize_aa_sequences([wt], seq_len)[0]``
rather than the newer, no-padding ``tokenize_single_aa_sequence`` -- matching
``create_dms_dataset``'s own call for this exact purpose, per issue #1295.

Each item also carries ``mask`` (``BoolTensor[L]``, ``aa_parents_idxs != AA_AMBIG_IDX``:
True at every position whose wild-type amino acid is known, False at a wild-type ``X``), the
attention mask ``CoreModel.forward`` requires. ``DMSMatrixCollator`` pads it with False,
and its ``mask_rate`` never changes it: ``mask_rate`` only drops cells from
``valid_mask`` (which measured cells the loss scores), never positions the encoder sees.

The eager ``Dataset`` is wrapped in a ``DatasetBundle`` only at the end of this module,
by :func:`load_dms_matrix_train_val`, which is what ``dasm2 train`` calls for a
``dms_matrix_calibration`` task (issue #1330). That loader and
:func:`load_dms_matrix_splits` take the train/val/test split as an input (a file or a
``split`` column, checked by :func:`resolve_split_table`), unlike
:meth:`DMSMatrixDataset.split_dataset`, which computes one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast, get_args

import numpy as np
import pandas as pd
import torch

from dasm2.config import CvSplitConfig
from dasm2.data.collators import TensorScale, TransformCollator, split_tensor_samples
from dasm2.data.dataset_utils import group_kfold_in_train
from dasm2.data.evolutionary_dataset import DatasetBundle, make_worker_init_fn
from dasm2.data.functional.consensus import get_reference_sequence
from dasm2.data.functional.dms_matrix import mask_wt_to_nan, to_dms_matrix
from dasm2.data.functional.single_mutants import filter_single_mutants
from dasm2.data.functional_formats import GROUP_ID_COL, IS_REFERENCE_COL
from dasm2.data.functional_registry import load_functional_dataset
from dasm2.data.masking import MASK_KEY, TOKEN_IDS_KEY
from dasm2.sequence.constants import AA_AMBIG_IDX
from dasm2.tensors_of_seqs import tokenize_aa_sequences

logger = logging.getLogger(__name__)

SEQUENCE_COL = "sequence_aa"
CLUSTER_ID_COL = "cluster_id"
Y_STABILITY_COL = "y_stability"
SPLIT_COL = "split"
#: Batch keys of a domain item, shared with ``metric_providers._update_dms_matrix``.
#: ``TOKEN_IDS_KEY`` and ``MASK_KEY`` are the keys of ``dasm2.data.masking``.
MEASURED_MATRIX_KEY = "measured_matrix"
VALID_MASK_KEY = "valid_mask"

CvSplit = Literal["train", "val", "test"]
SPLIT_NAMES: Tuple[CvSplit, ...] = get_args(CvSplit)
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = SPLIT_NAMES


class DMSMatrixDataset(torch.utils.data.Dataset):
    """One item per wild-type domain, in dense ``[L, 20]`` matrix form.

    Construction validates every domain up front (at ``__init__`` time, not
    lazily in ``__getitem__``) and stores only the domains that resolved
    successfully; ``self.skipped`` is a list of ``(group_id, reason)`` for
    every domain excluded because its WT reference was ambiguous/missing or
    its ``cluster_id`` was ambiguous.
    """

    row_filter_digest: Optional[str] = None
    """Never row-filtered; part of the ``CacheableDataset`` protocol (#1672)."""

    def __init__(self, df: pd.DataFrame, nickname: str = "unnamed_dms_matrix") -> None:
        """Build one item per ``group_id`` in a standardized functional DataFrame.

        Args:
            df: DataFrame with ``sequence_aa``, ``y_stability``, ``group_id``,
                ``is_reference``, ``cluster_id`` columns (the output of
                :func:`dasm2.data.functional_registry.load_functional_dataset`
                for a dataset carrying a ``cluster_id`` column).
            nickname: Human-readable identifier for this dataset.

        Raises:
            ValueError: If a required column is missing.
        """
        self.nickname = nickname
        self.samples: List[Dict[str, Any]] = []
        self.skipped: List[Tuple[str, str]] = []

        required_cols = {GROUP_ID_COL, IS_REFERENCE_COL, SEQUENCE_COL, CLUSTER_ID_COL}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}. "
                f"Available columns: {list(df.columns)}"
            )

        for group_id, group_df in df.groupby(GROUP_ID_COL, sort=False):
            group_id = str(group_id)
            sample, skip_reason = _build_domain_sample(group_id, group_df)
            if skip_reason is not None:
                self.skipped.append((group_id, skip_reason))
                logger.warning("Skipping domain %r: %s", group_id, skip_reason)
                continue
            assert sample is not None  # guaranteed by _build_domain_sample's contract
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    @classmethod
    def from_dataset_name(cls, dataset_name: str) -> "DMSMatrixDataset":
        """Build from a registered functional-data manifest name.

        Args:
            dataset_name: Name registered via
                :func:`dasm2.data.functional_registry.register_functional_dataset`
                (or a manifest YAML's own name), for a dataset carrying a
                ``cluster_id`` column.

        Raises:
            KeyError: If no dataset with this name is registered.
        """
        df = load_functional_dataset(dataset_name)
        return cls(df, nickname=dataset_name)

    @classmethod
    def _from_samples(
        cls,
        samples: List[Dict[str, Any]],
        skipped: List[Tuple[str, str]],
        nickname: str,
    ) -> "DMSMatrixDataset":
        """Build an instance directly from already-validated samples.

        Bypasses ``__init__``'s DataFrame validation -- used by
        :meth:`split_dataset` to carve a subset of domains out of an
        already-built dataset without re-running domain resolution.
        """
        instance = cls.__new__(cls)
        instance.nickname = nickname
        instance.samples = samples
        instance.skipped = skipped
        return instance

    @classmethod
    def split_dataset(
        cls,
        full: "DMSMatrixDataset",
        split: CvSplit,
        cv_split: CvSplitConfig,
        test_num_folds: int = 5,
        test_fold_seed: int = 0,
    ) -> "DMSMatrixDataset":
        """Cluster-safe train/val/test split of an already-built dataset.

        Two-stage split, both keyed on ``cluster_id``, reusing
        :func:`dasm2.data.dataset_utils.group_kfold_in_train` (plain
        ``GroupKFold``, no reimplementation):

        1. A one-time split carves off a fixed held-out ``test`` set
           (``test_num_folds`` folds, fold 0), regardless of ``cv_split``.
        2. ``cv_split`` splits the remaining pool into ``train``/``val``.

        Validates that no ``cluster_id`` appears in more than one of
        ``{train, val, test}`` before returning.

        Args:
            full: The complete, unsplit dataset (e.g. from
                :meth:`from_dataset_name`).
            split: Which split to return.
            cv_split: ``num_folds``/``fold_index`` for the train/val stage.
                Its own ``group_col`` is ignored; this method always splits
                on ``cluster_id``.
            test_num_folds: Fold count for the one-time test-set split
                (fixed across every training run so results stay comparable
                across ``cv_split.fold_index`` values).
            test_fold_seed: Accepted for interface symmetry with other
                seeded splits; unused, since ``GroupKFold`` is itself
                deterministic given ``(group_col, num_folds, fold_index,
                {(group_label, group_size)})``.

        Raises:
            ValueError: If ``split`` is not one of ``"train"``/``"val"``/``"test"``.
            AssertionError: If a ``cluster_id`` ends up in more than one split
                (would indicate a bug in the split logic, not bad input data).
        """
        del test_fold_seed  # accepted for interface symmetry only; see docstring.
        if split not in SPLIT_NAMES:
            raise ValueError(
                f"split must be one of 'train'/'val'/'test', got {split!r}"
            )

        cluster_ids = np.array([s["cluster_id"] for s in full.samples])

        test_cv = CvSplitConfig(
            num_folds=test_num_folds, fold_index=0, group_col=CLUSTER_ID_COL
        )
        in_pool = group_kfold_in_train(cluster_ids, test_cv)
        test_mask = ~in_pool

        pool_idx = np.where(in_pool)[0]
        in_train_within_pool = group_kfold_in_train(cluster_ids[pool_idx], cv_split)

        train_mask = np.zeros(len(cluster_ids), dtype=bool)
        train_mask[pool_idx[in_train_within_pool]] = True
        val_mask = np.zeros(len(cluster_ids), dtype=bool)
        val_mask[pool_idx[~in_train_within_pool]] = True

        train_clusters = set(cluster_ids[train_mask])
        val_clusters = set(cluster_ids[val_mask])
        test_clusters = set(cluster_ids[test_mask])
        leaked = (
            (train_clusters & val_clusters)
            | (train_clusters & test_clusters)
            | (val_clusters & test_clusters)
        )
        if leaked:
            raise AssertionError(
                f"cluster_id leakage across train/val/test split: {sorted(leaked)}"
            )

        mask = {"train": train_mask, "val": val_mask, "test": test_mask}[split]
        selected = [s for s, keep in zip(full.samples, mask) if keep]
        return cls._from_samples(
            selected, full.skipped, nickname=f"{full.nickname}_{split}"
        )

    @classmethod
    def with_cluster_split(
        cls,
        dataset_name: str,
        split: CvSplit,
        cv_split: CvSplitConfig,
        test_num_folds: int = 5,
        test_fold_seed: int = 0,
    ) -> "DMSMatrixDataset":
        """Load ``dataset_name`` and return its cluster-safe train/val/test split.

        See :meth:`split_dataset` for the split algorithm.
        """
        full = cls.from_dataset_name(dataset_name)
        return cls.split_dataset(
            full,
            split,
            cv_split,
            test_num_folds=test_num_folds,
            test_fold_seed=test_fold_seed,
        )


def _build_domain_sample(
    group_id: str, group_df: pd.DataFrame
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Build one domain's sample dict, or return a skip reason.

    Returns:
        ``(sample, None)`` on success, or ``(None, reason)`` when the domain
        must be skipped (ambiguous/missing WT reference, or ambiguous
        ``cluster_id``).
    """
    ref_mask = group_df[IS_REFERENCE_COL].fillna(False).astype(bool)
    ref_rows = group_df[ref_mask]
    ref_rows_dedup = ref_rows.drop_duplicates(subset=[SEQUENCE_COL])
    wt = get_reference_sequence(ref_rows_dedup, IS_REFERENCE_COL, SEQUENCE_COL)
    if wt is None:
        return None, (
            f"expected exactly 1 distinct WT reference sequence, found "
            f"{len(ref_rows_dedup)}"
        )

    cluster_ids = ref_rows[CLUSTER_ID_COL].unique()
    if len(cluster_ids) != 1:
        return None, (
            f"expected exactly 1 distinct cluster_id on reference rows, found "
            f"{sorted(str(c) for c in cluster_ids)}"
        )
    cluster_id = str(cluster_ids[0])

    wt_len = len(wt)
    length_ok = group_df[SEQUENCE_COL].str.len() == wt_len
    variants_df = group_df[length_ok]

    single_mutants = filter_single_mutants(variants_df, SEQUENCE_COL, wt)
    matrix_df = to_dms_matrix(
        single_mutants, y_col=Y_STABILITY_COL, reference_sequence=wt
    )

    # Wildtype cells are excluded *explicitly* (see module docstring), not as
    # a side effect of filter_single_mutants dropping the WT row.
    matrix_df = mask_wt_to_nan(matrix_df, wt)

    # to_dms_matrix returns (20 AAs, L positions); this dataset's convention
    # is [L, 20].
    measured_matrix = torch.tensor(matrix_df.T.to_numpy(), dtype=torch.float32)
    valid_mask = torch.isfinite(measured_matrix)

    aa_parents_idxs = tokenize_aa_sequences([wt], wt_len)[0]

    sample: Dict[str, Any] = {
        "group_id": group_id,
        "cluster_id": cluster_id,
        "wt_aa_sequence": wt,
        TOKEN_IDS_KEY: aa_parents_idxs,
        MEASURED_MATRIX_KEY: measured_matrix,
        VALID_MASK_KEY: valid_mask,
        # Attention mask for CoreModel.forward: True where the wild-type amino acid is
        # known (a wild-type ``X`` tokenizes to AA_AMBIG_IDX and gives False).
        MASK_KEY: aa_parents_idxs != AA_AMBIG_IDX,
    }
    return sample, None


#: Default ``tensor_specs`` for this dataset's fields -- passed to
#: :class:`DMSMatrixCollator` unless a caller overrides it. Padded positions
#: get ``valid_mask=False`` automatically (the pad value declared here), so
#: a padded cell and a real-but-unmeasured cell are excluded from the loss
#: identically. The pad value is ``None`` for ``"list"``-scale keys
#: (``group_id``/``cluster_id``/``wt_aa_sequence``): ``TransformCollator``
#: passes list-scale values through unpadded, so no pad value is ever
#: consumed for them. ``wt_aa_sequence`` rides along as "list"-scale too
#: (not just ``group_id``/``cluster_id``) so it survives batching for
#: per-domain bookkeeping/debugging rather than being silently dropped.
DMS_MATRIX_TENSOR_SPECS: Dict[str, Tuple[Optional[float], TensorScale]] = {
    TOKEN_IDS_KEY: (AA_AMBIG_IDX, "codon"),
    MEASURED_MATRIX_KEY: (float("nan"), "codon"),
    VALID_MASK_KEY: (False, "codon"),
    MASK_KEY: (False, "codon"),
    "group_id": (None, "list"),
    "cluster_id": (None, "list"),
    "wt_aa_sequence": (None, "list"),
}


class DMSMatrixCollator:
    """Pads a batch of domains to ``L_max`` and optionally corrupts ``valid_mask``.

    Wraps a :class:`~dasm2.data.collators.TransformCollator` (unmodified,
    padding delegated to it entirely) with a ``mask_rate`` knob: this
    dataset has only ~300 unique domains total, split further across
    train/val/test, so a fine-tuning run risks memorizing each domain's
    exact measured cells rather than learning a generalizable pattern.
    ``mask_rate > 0.0`` independently drops each already-``True`` cell of
    the padded ``valid_mask`` with that probability on every call -- so the
    same domain never presents the identical supervised cell set twice.
    Never turns a ``False`` cell ``True``: padded and never-measured cells
    stay excluded regardless of ``mask_rate``.

    ``mask_rate=0.0`` (the default) makes this collator byte-identical to
    calling the wrapped ``TransformCollator`` directly.
    """

    def __init__(
        self,
        tensor_specs: Optional[Dict[str, Tuple[Optional[float], TensorScale]]] = None,
        mask_rate: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        Args:
            tensor_specs: ``TransformCollator``-style spec for this batch's
                fields. Defaults to :data:`DMS_MATRIX_TENSOR_SPECS`; a caller
                overrides it only to change padding conventions, not to add
                unrelated keys (this collator always looks for a
                :data:`VALID_MASK_KEY` key when ``mask_rate > 0.0``).
            mask_rate: Fraction of already-valid cells to randomly drop per
                call. ``0.0`` (default) is a no-op.
            generator: Optional ``torch.Generator`` for reproducible
                corruption draws. ``None`` draws from the ambient RNG.
        """
        if not 0.0 <= mask_rate <= 1.0:
            raise ValueError(f"mask_rate must be in [0, 1], got {mask_rate}")
        self.mask_rate = mask_rate
        self.generator = generator
        resolved_specs = (
            tensor_specs if tensor_specs is not None else DMS_MATRIX_TENSOR_SPECS
        )
        # TransformCollator's own tensor_specs type is float-only (it never
        # reads the pad value for "list"-scale keys); cast rather than
        # widening that shared class's type for this module's None sentinel.
        self._tensor_specs = cast(Dict[str, Tuple[float, TensorScale]], resolved_specs)
        self._pad_collator = TransformCollator(tensor_specs=self._tensor_specs)

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Loud failure on a sample key with no tensor_specs entry, matching
        # every sibling collator (ContrastiveBindingCollator,
        # PairwiseBindingCollator, MaskedTransformCollator) rather than
        # TransformCollator's own quieter drop-with-a-log-warning default.
        samples = split_tensor_samples(
            samples, self._tensor_specs, frozenset(), "DMSMatrixCollator"
        )
        batch = self._pad_collator(samples)
        if self.mask_rate > 0.0:
            if VALID_MASK_KEY not in batch:
                raise ValueError(
                    f"DMSMatrixCollator needs a {VALID_MASK_KEY!r} key to apply "
                    f"mask_rate={self.mask_rate}, but the configured "
                    f"tensor_specs produced none. Got keys: {list(batch.keys())}."
                )
            valid_mask = batch[VALID_MASK_KEY]
            keep = (
                torch.rand(valid_mask.shape, generator=self.generator) >= self.mask_rate
            )
            batch[VALID_MASK_KEY] = valid_mask & keep
        return batch


SPLIT_COLUMNS = (GROUP_ID_COL, CLUSTER_ID_COL, SPLIT_COL)


def read_dataset_csv(dataset_csv: Path) -> pd.DataFrame:
    """Read a dataset CSV with ``group_id`` and ``cluster_id`` kept as text.

    Without this a numeric-looking id such as ``001`` would load as ``1`` and then
    disagree with the same id read from a split file.

    Raises:
        ValueError: If any row has no ``group_id`` or no ``cluster_id``.
    """
    df = pd.read_csv(dataset_csv, dtype={GROUP_ID_COL: str, CLUSTER_ID_COL: str})
    return _require_text_ids(df, str(dataset_csv))


def _require_text_ids(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """``df`` with ``group_id`` and ``cluster_id`` as text, refusing missing ids.

    The check runs before the text conversion, because ``astype(str)`` turns a
    missing id into the string ``"nan"`` and every unclustered domain would then
    form one giant cluster that passes the leakage check.

    Raises:
        ValueError: If any row has no ``group_id`` or no ``cluster_id``.
    """
    for column in (GROUP_ID_COL, CLUSTER_ID_COL):
        if column in df.columns and df[column].isna().any():
            raise ValueError(
                f"{source} has {int(df[column].isna().sum())} rows with no {column}."
            )
    df = df.copy()
    for column in (GROUP_ID_COL, CLUSTER_ID_COL):
        if column in df.columns:
            df[column] = df[column].astype(str)
    return df


def load_dataset_table(dataset: Union[str, Path]) -> pd.DataFrame:
    """The dataset table for a CSV path or a registered functional-data nickname.

    An existing file is read with :func:`read_dataset_csv`; anything else is looked up
    in the functional-data registry (for example ``tsuboyama_megascale``), whose table
    carries its own ``cluster_id`` and ``split`` columns.

    Raises:
        KeyError: If ``dataset`` is neither an existing file nor a registered name.
    """
    if Path(dataset).is_file():
        return read_dataset_csv(Path(dataset))
    return _require_text_ids(load_functional_dataset(str(dataset)), str(dataset))


def _split_table_from_column(df: pd.DataFrame) -> pd.DataFrame:
    """The ``group_id, cluster_id, split`` table carried by ``df``'s own ``split`` column."""
    if SPLIT_COL not in df.columns:
        raise ValueError(
            f"No split file given and the dataset has no {SPLIT_COL!r} column. "
            f"Pass '<dataset>:<split_csv>' or add the column. "
            f"Dataset columns: {list(df.columns)}."
        )
    per_group = df[list(SPLIT_COLUMNS)].astype(str).drop_duplicates()
    multi = per_group.loc[per_group[GROUP_ID_COL].duplicated(), GROUP_ID_COL]
    if len(multi):
        raise ValueError(
            f"dataset column {SPLIT_COL!r}: group_ids with more than one split or "
            f"cluster_id value: {sorted(set(multi))[:5]}."
        )
    return per_group.reset_index(drop=True)


def _split_table_from_file(split_csv: Path) -> pd.DataFrame:
    """Read a ``group_id, cluster_id, split`` file, one row per ``group_id``."""
    split_df = pd.read_csv(split_csv, dtype=str)
    missing = set(SPLIT_COLUMNS) - set(split_df.columns)
    if missing:
        raise ValueError(
            f"split file {split_csv} is missing columns {sorted(missing)}. "
            f"Expected {list(SPLIT_COLUMNS)}, got {list(split_df.columns)}."
        )
    empty_rows = split_df[list(SPLIT_COLUMNS)].isna().any(axis=1)
    if empty_rows.any():
        raise ValueError(
            f"split file {split_csv} has {int(empty_rows.sum())} rows with an empty "
            f"{'/'.join(SPLIT_COLUMNS)} cell."
        )
    duplicated = split_df.loc[split_df[GROUP_ID_COL].duplicated(), GROUP_ID_COL]
    if len(duplicated):
        raise ValueError(
            f"split file {split_csv} lists group_ids more than once: "
            f"{sorted(set(duplicated))[:5]}."
        )
    return split_df


def _validate_split_table(
    split_df: pd.DataFrame, df: pd.DataFrame, source: str
) -> pd.DataFrame:
    """Check a split table against the dataset ``df``; return it unchanged."""
    bad_labels = sorted(set(split_df[SPLIT_COL]) - set(SPLIT_NAMES))
    if bad_labels:
        raise ValueError(
            f"{source} has unknown split labels {bad_labels}; "
            f"expected only {list(SPLIT_NAMES)}."
        )
    dataset_clusters = (
        df[[GROUP_ID_COL, CLUSTER_ID_COL]]
        .astype(str)
        .drop_duplicates()
        .set_index(GROUP_ID_COL)[CLUSTER_ID_COL]
    )
    ambiguous = dataset_clusters.index[dataset_clusters.index.duplicated()]
    if len(ambiguous):
        raise ValueError(
            f"The dataset gives {len(set(ambiguous))} group_ids more than one "
            f"cluster_id, e.g. {sorted(set(ambiguous))[:5]}."
        )
    unknown = sorted(set(split_df[GROUP_ID_COL]) - set(dataset_clusters.index))
    if unknown:
        raise ValueError(
            f"{source} has {len(unknown)} group_ids not in the dataset, "
            f"e.g. {unknown[:5]}."
        )
    disagree = split_df[
        split_df[GROUP_ID_COL].map(dataset_clusters) != split_df[CLUSTER_ID_COL]
    ]
    if len(disagree):
        raise ValueError(
            f"{source} disagrees with the dataset's cluster_id for "
            f"{len(disagree)} domains, e.g. {disagree[GROUP_ID_COL].tolist()[:5]}."
        )
    splits_per_cluster = split_df.groupby(CLUSTER_ID_COL)[SPLIT_COL].nunique()
    leaked = sorted(splits_per_cluster[splits_per_cluster > 1].index)
    if leaked:
        raise ValueError(
            f"{source} puts {len(leaked)} clusters in more than one split, "
            f"e.g. {leaked[:5]}."
        )
    return split_df


def resolve_split_table(df: pd.DataFrame, split_csv: Optional[Path]) -> pd.DataFrame:
    """Return a validated ``group_id, cluster_id, split`` table for ``df``.

    The split is an input prepared outside the experiment. It comes either from
    ``split_csv`` or, when that is ``None``, from a ``split`` column on ``df``
    itself (constant within each ``group_id``). Either way it is trusted for
    *which* domains go where, but not for cluster safety: this checks that every
    ``cluster_id`` sits entirely in one split, that every ``group_id`` is in the
    dataset, and that the table's ``cluster_id`` agrees with the dataset's.

    Raises:
        ValueError: On a missing column, an unknown split label, a duplicated
            ``group_id``, a ``group_id`` absent from ``df``, a ``cluster_id``
            that disagrees with ``df``'s, a ``cluster_id`` shared by two splits,
            or (column form) a ``group_id`` with more than one split label.
    """
    if split_csv is None:
        return _validate_split_table(
            _split_table_from_column(df), df, f"dataset column {SPLIT_COL!r}"
        )
    return _validate_split_table(
        _split_table_from_file(split_csv), df, f"split file {split_csv}"
    )


def load_dms_matrix_splits(
    dataset: Union[str, Path], split_csv: Optional[Path] = None
) -> Dict[CvSplit, DMSMatrixDataset]:
    """Build one :class:`DMSMatrixDataset` per split from a dataset plus a split.

    Args:
        dataset: A functional-format CSV path, or a registered functional-data
            nickname such as ``tsuboyama_megascale`` (see :func:`load_dataset_table`).
            Either carries ``sequence_aa``, ``y_stability``, ``group_id``,
            ``is_reference``, ``cluster_id`` (and ``split`` when ``split_csv`` is
            ``None``).
        split_csv: Optional ``group_id, cluster_id, split`` file; see
            :func:`resolve_split_table`.

    Raises:
        ValueError: If a split ends up with no domains, or a listed domain was
            skipped by :class:`DMSMatrixDataset` (ambiguous WT or cluster).
    """
    df = load_dataset_table(dataset)
    split_df = resolve_split_table(df, split_csv)
    listed = set(split_df[GROUP_ID_COL])
    unlisted = set(df[GROUP_ID_COL]) - listed
    if unlisted:
        logger.warning(
            "%d dataset domains are in no split and are left out, e.g. %s",
            len(unlisted),
            sorted(unlisted)[:3],
        )
    # Resolve every listed domain once, then carve the per-split subsets out of it.
    full = DMSMatrixDataset(
        df[df[GROUP_ID_COL].isin(listed)], nickname=Path(dataset).stem
    )
    result: Dict[CvSplit, DMSMatrixDataset] = {}
    for split in SPLIT_NAMES:
        group_ids = set(split_df.loc[split_df[SPLIT_COL] == split, GROUP_ID_COL])
        if not group_ids:
            raise ValueError(f"No domains in split {split!r}.")
        skipped = [item for item in full.skipped if item[0] in group_ids]
        if skipped:
            raise ValueError(
                f"{len(skipped)} {split} domains were skipped: {skipped[:3]}."
            )
        samples = [s for s in full.samples if s["group_id"] in group_ids]
        result[split] = DMSMatrixDataset._from_samples(
            samples, [], nickname=f"{Path(dataset).stem}_{split}"
        )
    return result


def load_dms_matrix_train_val(
    data_spec: str, mask_rate: float
) -> Tuple[DatasetBundle, DatasetBundle]:
    """``dasm2 train`` loader: ``"<dataset>[:<split_csv>]"`` to train/val bundles.

    ``<dataset>`` is a CSV path or a registered nickname (``tsuboyama_megascale``).
    Without ``:<split_csv>`` the dataset must carry a ``split`` column. Neither
    value may contain a colon. ``mask_rate`` applies to the train collator only;
    validation always uses ``mask_rate=0.0`` so validation loss is deterministic.
    The test split is built by :func:`load_dms_matrix_splits` too, which checks
    that it is non-empty and well-formed, but ``dasm2 train`` does not use it.

    Raises:
        ValueError: If ``data_spec`` has more than one colon, or on any error
            :func:`load_dms_matrix_splits` raises.
    """
    dataset, colon, split_csv = data_spec.partition(":")
    if data_spec.count(":") > 1 or (colon and not split_csv) or not dataset:
        raise ValueError(
            "DMS matrix data must be '<dataset>' or '<dataset>:<split_csv>' "
            f"with no other colons in the values, got: {data_spec!r}"
        )
    splits = load_dms_matrix_splits(dataset, Path(split_csv) if split_csv else None)
    worker_init_fn = make_worker_init_fn()
    train_bundle = DatasetBundle(
        splits[TRAIN_SPLIT],
        cast(Any, DMSMatrixCollator(mask_rate=mask_rate)),
        worker_init_fn,
    )
    val_bundle = DatasetBundle(
        splits[VAL_SPLIT], cast(Any, DMSMatrixCollator(mask_rate=0.0)), worker_init_fn
    )
    return train_bundle, val_bundle
