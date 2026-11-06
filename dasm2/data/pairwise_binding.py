"""Pairwise binding-classification dataset + collator (issue #802).

Follows the project-wide data pattern: a single ``TransformDataset`` in
streaming mode backed by a list of per-sample transforms, wrapped by a
head-specific collator. Mirrors ``dasm2/data/contrastive_binding.py``'s file
layout, but the classifier consumes an already-labeled CSV (negative
sampling / group assignment happen upstream) and needs the full per-residue
entity-B tensor for cross-attention rather than a pre-pooled vector.

Entity-A path (per-sample transforms, run in ``__getitem__``):
    TokenizeParent → ComputeLayout → ComputeMasks → LoadFullContextEmbedding

Unlike ``create_contrastive_binding_dataset``, ``loss_mask`` is *not*
renamed: ``BindingClassificationLoss`` reads an explicit per-pair label
(``BINARY_TARGET_KEY``) rather than relying on the trainer's
``_get_n_loss_positions`` batch-size fallback, so there is no reason to hide
``loss_mask`` from that fallback.

Entity-B path: the identifier column (``antigen_key``, reusing the existing
contrastive-binding convention) is injected into the seed dict via
``TransformDataset.extra_string_fields``, then ``LoadFullContextEmbedding``
resolves it to ``{context_dir}/{key}.npz`` and returns the *full*
per-residue tensor (no pooling), unmodified, plus a mask that is ``False``
at the leading/trailing ESM-2 ``<cls>``/``<eos>`` special-token positions
(never stripped from the tensor itself, only excluded from the head's
cross-attention — see ``LoadFullContextEmbedding``'s docstring). The
loader/cache (``load_residue_embedding`` / ``cached_residue_tensor`` /
``resolve_embedding_path``) lives in ``dasm2.data.embedding_io``, shared
with ``contrastive_binding.py`` so the two datasets use one LRU cache, not
two with the same bound. This module's own vocabulary is not fully
domain-neutral yet — it still reuses ``contrastive_binding.py``'s
antibody/antigen-specific ``detect_ab_sequence_columns`` and
``ANTIGEN_KEY_FIELD`` rather than entity-A/entity-B equivalents; only the
embedding I/O layer has been generalized so far.

Collator: ``TransformCollator`` handles entity-A padding (its ``batch_max``
is derived from the first ``"codon"``-scale key). Entity-B has its own,
independently-lengthed per-residue tensor, so ``PairwiseBindingCollator``
pads it to its own batch-max separately, via ``pad_and_stack`` — the one
piece of collation logic with no existing template (see module docstring
of ``collators.py``: no ``VALID_SCALES`` entry pads a second sequence to an
independent batch-max).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from dasm2.data.collators import (
    TensorScale,
    TransformCollator,
    pad_and_stack,
    split_tensor_samples,
)
from dasm2.data.contrastive_binding import ANTIGEN_KEY_FIELD, detect_ab_sequence_columns
from dasm2.data.dataset_utils import (
    clean_chain_series,
    concat_chains_with_separator,
    split_df_train_val,
)
from dasm2.data.delta_affinity import BINARY_TARGET_KEY
from dasm2.data.embedding_io import cached_residue_tensor, resolve_embedding_path
from dasm2.data.evolutionary_dataset import (
    DatasetBundle,
    TransformDataset,
    make_worker_init_fn,
)
from dasm2.data.transforms import ComputeLayout, ComputeMasks, TokenizeParent
from dasm2.sequence.constants import AA_AMBIG_IDX
from dasm2.sequence.layout import padding_sentinel_for

if TYPE_CHECKING:
    from dasm2.config import CvSplitConfig
    from dasm2.data.binding_corruption import BindingCorruptionConfig

logger = logging.getLogger(__name__)
cv_split_logger = logger.getChild("cv_split")

# Antigen clustering column used for this dataset's CV split — a fixed
# property of ``classification_dataset_hard.csv``'s schema, decoupled from
# the ``group_col`` a caller may set on ``CvSplitConfig`` for a co-trained
# dataset with a different (and differently-meaning) grouping column, e.g.
# the evolutionary side's ``sample_id``.
DEFAULT_BINDING_CV_GROUP_COL = "antigen_cluster_40"

# ``random_state=0`` mirrors the exact call in this dataset's own build
# script (experiments/2026-06-07-binding-classification-sabdab/scripts/
# build_classification_dataset.py:346), so re-running the split here
# reproduces that script's published per-fold counts bit-for-bit rather
# than an independently valid (but different) partition.
_BINDING_CV_RANDOM_STATE = 0

# Key of the full per-residue entity-B embedding (shape ``[L_b, D_b]``, no
# pooling — the classification head needs the whole sequence for
# cross-attention, unlike the contrastive head's pre-pooled vector).
BINDING_CONTEXT_EMBEDDING_KEY = "binding_context_embedding"
# All-``True`` ``[L_b]`` mask paired with the embedding above. Kept even
# though entity-B NPZs carry no padding themselves — after batch-level
# padding to a shared ``context_max``, this mask is what tells the head's
# cross-attention which entity-B positions are real vs. padding.
BINDING_CONTEXT_MASK_KEY = "binding_context_mask"
# Explicit binary label column in the input CSV.
LABEL_COLUMN = "label"

_PAIRWISE_OUTPUT_KEYS = frozenset(
    {
        "aa_parents_idxs",
        "layout",
        "mask",
        "loss_mask",
        BINDING_CONTEXT_EMBEDDING_KEY,
        BINDING_CONTEXT_MASK_KEY,
        BINARY_TARGET_KEY,
    }
)


def entity_a_mask_from_batch(batch: Dict[str, Any], context: str) -> torch.Tensor:
    """Real-vs-padding/separator mask for entity-A, ``loss_mask`` preferred
    over ``mask`` when both are present.

    The one place this precedent is defined; both
    ``BindingClassificationHead.forward()`` (the query mask for
    cross-attention) and ``CorruptedPairwiseBindingCollator`` (issue #1221,
    which positions are eligible for corruption) need the same real-vs-not
    distinction and must stay in sync.

    Args:
        batch: The batch dict to read the mask from.
        context: Caller identity, folded into the error message so a
            missing key is attributable to its call site (mirrors
            ``dasm2.utils.validation.validate_batch_keys``'s ``context``
            parameter).

    Raises:
        ValueError: Neither ``"loss_mask"`` nor ``"mask"`` is in ``batch``.
    """
    entity_a_mask = batch.get("loss_mask")
    if entity_a_mask is None:
        entity_a_mask = batch.get("mask")
    if entity_a_mask is None:
        raise ValueError(
            f"{context} missing required batch key: one of 'loss_mask' or "
            f"'mask'. Got keys: {list(batch.keys())}."
        )
    return cast(torch.Tensor, entity_a_mask)


# ---------------------------------------------------------------------------
# Transform: resolve antigen_key → full per-residue entity-B embedding + mask
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadFullContextEmbedding:
    """Resolve ``sample[ANTIGEN_KEY_FIELD]`` to its full per-residue tensor.

    Reads ``{context_dir}/{key}.npz`` via the shared LRU cache in
    ``dasm2.data.embedding_io`` (also used by ``contrastive_binding``, so
    repeated keys across both datasets are deserialized exactly once per
    worker), but performs no pooling —
    ``BindingClassificationHead`` needs the whole ``[L_b, D_b]`` tensor for
    cross-attention. Populates:
        - ``binding_context_embedding``: ``[L_b, D_b]`` float32 tensor,
          unmodified — including the leading ``<cls>``/trailing ``<eos>``
          special-token embeddings ESM-2 prepends/appends to every sequence
          (verified against ``pairs_clustered.csv``'s raw antigen sequence
          lengths: every checked NPZ is exactly ``len(sequence) + 2``).
        - ``binding_context_mask``: ``[L_b]`` bool tensor, ``True`` at real
          residue positions and ``False`` at the ``<cls>``/``<eos>``
          positions (indices ``0`` and ``-1``) — those two special-token
          embeddings are never removed from the tensor (so real-residue
          positions keep their original index), but are excluded from the
          cross-attention softmax via ``BindingClassificationHead``'s
          ``context_mask``, the same way entity-A's ``^`` separator is
          excluded from that head's computation via ``loss_mask``. Padding
          is added later, at the batch level, by ``PairwiseBindingCollator``.

    ``max_context_len``, if set, truncates the leading ``max_context_len``
    residues of any longer tensor before the mask is built (mirrors the
    grid script's ``max_ag_len`` precedent). Truncation keeps ``<cls>`` at
    index 0 (always masked) but drops the trailing ``<eos>`` entirely, so
    the *last* position is only masked when the tensor was not truncated
    (i.e. it's still the genuine ``<eos>``, not an arbitrary residue that
    happened to land at the end of the truncated window). Slicing the
    LRU-cached tensor returns a view, never mutating the cached full-length
    tensor, so other callers/instances with a different ``max_context_len``
    are unaffected.
    """

    context_dir: Path
    max_context_len: Optional[int] = None

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        key = sample[ANTIGEN_KEY_FIELD]
        path = resolve_embedding_path(self.context_dir, key)
        residues = cached_residue_tensor(str(path))  # [L_b, D_b]: <cls> + AAs + <eos>
        full_len = residues.shape[0]
        truncated = self.max_context_len is not None and full_len > self.max_context_len
        if truncated:
            residues = residues[: self.max_context_len]
        sample[BINDING_CONTEXT_EMBEDDING_KEY] = residues

        mask = torch.ones(residues.shape[0], dtype=torch.bool)
        mask[0] = False  # <cls>, always at position 0 regardless of truncation
        if not truncated:
            mask[-1] = False  # <eos>, only present when nothing was truncated away
        sample[BINDING_CONTEXT_MASK_KEY] = mask
        return sample


AgPoolMode = Literal["full", "mean", "cls"]


def pool_antigen_context(
    embedding: torch.Tensor, mask: torch.Tensor, mode: AgPoolMode
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Reduce one ``LoadFullContextEmbedding`` output to a given ``ag_pool`` mode.

    A pure per-sample function of the ``(embedding, mask)`` pair every
    ``ag_pool`` value shares (issue #1187), applied before collation so a
    ``"mean"``/``"cls"`` sample never needs the batch-level padding
    ``PairwiseBindingCollator`` builds for ``"full"``:

        - ``"full"``: returns ``(embedding, mask)`` unchanged — the
          ``xattn_*`` heads consume the untouched per-residue sequence and
          mask via cross-attention.
        - ``"mean"``: masked mean over real residues only (``mask`` excludes
          ``<cls>``/``<eos>``), collapsing to ``[D]``.
        - ``"cls"``: the ``<cls>`` embedding itself, ``embedding[0]``,
          ignoring ``mask`` by design (``<cls>`` is always real position 0
          regardless of truncation).

    Args:
        embedding: ``[L, D]`` single-sample entity-B tensor (no batch dim),
            as produced by ``LoadFullContextEmbedding``.
        mask: ``[L]`` bool, ``True`` at real residue positions.
        mode: Which reduction to apply.

    Returns:
        ``(pooled_embedding, pooled_mask)``. For ``"full"``, ``pooled_mask``
        is the input ``mask``. For ``"mean"``/``"cls"``, the result is
        already a fixed-length ``[D]`` vector with no padding concept, so
        ``pooled_mask`` is ``None``.

    Raises:
        ValueError: Unknown ``mode``; ``mode="mean"`` with no valid
            (``mask``-True) residue to average; or ``mode="cls"`` when
            ``mask[0]`` is ``True`` (position 0 isn't a special ``<cls>``
            token per ``LoadFullContextEmbedding``'s convention, so
            ``embedding[0]`` would be an arbitrary real residue instead).
    """
    if mode == "full":
        return embedding, mask
    if mode == "mean":
        n_valid = int(mask.sum())
        if n_valid == 0:
            raise ValueError(
                "pool_antigen_context(mode='mean') requires at least one "
                "True position in mask, got none."
            )
        mask_f = mask.to(embedding.dtype).unsqueeze(-1)  # [L, 1]
        pooled = (embedding * mask_f).sum(dim=0) / n_valid
        return pooled, None
    if mode == "cls":
        if bool(mask[0]):
            raise ValueError(
                "pool_antigen_context(mode='cls') expects position 0 to be a "
                "special (mask=False) <cls> token, per LoadFullContextEmbedding's "
                "convention, but mask[0] is True here -- embedding[0] would be an "
                "ordinary residue, not the antigen's <cls> summary."
            )
        return embedding[0], None
    raise ValueError(f"Unknown ag_pool mode: {mode!r}. Expected 'full'/'mean'/'cls'.")


# ---------------------------------------------------------------------------
# Collator — entity-A padding via TransformCollator, entity-B padded to its
# own independent batch-max, plus the binary target stack.
# ---------------------------------------------------------------------------


@dataclass
class PairwiseBindingCollator:
    """Pads entity-A and entity-B tensors to their own, independent lengths.

    ``tensor_specs`` covers entity-A keys only (``aa_parents_idxs``,
    ``layout``, ``mask``, ``loss_mask``); entity-B's per-residue embedding
    and mask are handled explicitly below since they have their own length
    axis (``context_max``), unrelated to entity-A's ``batch_max``.
    """

    tensor_specs: Dict[str, Tuple[float, TensorScale]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._inner = TransformCollator(tensor_specs=self.tensor_specs)

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        known_extra_keys = frozenset(
            {
                BINDING_CONTEXT_EMBEDDING_KEY,
                BINDING_CONTEXT_MASK_KEY,
                BINARY_TARGET_KEY,
            }
        )
        tensor_samples = split_tensor_samples(
            samples, self.tensor_specs, known_extra_keys, "PairwiseBindingCollator"
        )
        batch = self._inner(tensor_samples)

        context_tensors = [s[BINDING_CONTEXT_EMBEDDING_KEY] for s in samples]
        context_max = max(t.shape[0] for t in context_tensors)
        batch[BINDING_CONTEXT_EMBEDDING_KEY] = pad_and_stack(
            context_tensors, context_max, 0.0
        )
        mask_tensors = [s[BINDING_CONTEXT_MASK_KEY] for s in samples]
        batch[BINDING_CONTEXT_MASK_KEY] = pad_and_stack(
            mask_tensors, context_max, False
        )
        batch[BINARY_TARGET_KEY] = torch.stack([s[BINARY_TARGET_KEY] for s in samples])
        return batch


# ---------------------------------------------------------------------------
# Factory — builds TransformDataset + collator
# ---------------------------------------------------------------------------


def _validate_pairwise_binding_df(
    df: pd.DataFrame,
    context_dir: Path,
    sequence_cols: Sequence[str],
) -> None:
    if len(df) == 0:
        raise ValueError("Pairwise binding DataFrame is empty.")
    if ANTIGEN_KEY_FIELD not in df.columns:
        raise ValueError(
            f"Missing required column '{ANTIGEN_KEY_FIELD}'. Got: {list(df.columns)}"
        )
    missing_cols = [c for c in sequence_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing entity-A sequence columns: {missing_cols}. "
            f"Available: {list(df.columns)}"
        )
    # Each row must carry at least one non-empty chain (reject rows that
    # would tokenize to nothing but a separator).
    per_chain_lens = np.stack(
        [clean_chain_series(df, c).str.len().to_numpy() for c in sequence_cols]
    )
    empty_rows = per_chain_lens.sum(axis=0) == 0
    if empty_rows.any():
        idxs = df.index[empty_rows].tolist()
        raise ValueError(
            f"Rows with all entity-A chain columns empty at indices: {idxs}. "
            f"Every row must have at least one non-empty chain."
        )

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Missing required label column '{LABEL_COLUMN}'. Got: {list(df.columns)}"
        )
    nan_mask = df[LABEL_COLUMN].isna()
    if nan_mask.any():
        idxs = df.index[nan_mask].tolist()
        raise ValueError(f"Label column '{LABEL_COLUMN}' has NaN at indices: {idxs}")
    bad_values = sorted(set(df[LABEL_COLUMN].tolist()) - {0, 1})
    if bad_values:
        raise ValueError(
            f"Label column '{LABEL_COLUMN}' must be binary (0/1); found other "
            f"values: {bad_values}"
        )

    # Filesystem check last: schema problems (columns, empty chains, label
    # values) should all surface together before a caller goes fix a path.
    if not context_dir.is_dir():
        raise FileNotFoundError(
            f"Entity-B embedding directory not found: {context_dir}"
        )


def create_pairwise_binding_dataset(
    df: pd.DataFrame,
    context_dir: Union[str, Path],
    nickname: str = "unnamed_pairwise_binding",
    sequence_cols: Optional[Sequence[str]] = None,
    max_context_len: Optional[int] = None,
    corruption_config: Optional["BindingCorruptionConfig"] = None,
) -> DatasetBundle:
    """Build a ``DatasetBundle`` for pairwise binding-classification training.

    Uses ``TransformDataset`` in streaming mode with the project's standard
    entity-A transforms (``TokenizeParent`` → ``ComputeLayout`` →
    ``ComputeMasks``), leaves ``loss_mask`` under its normal name (read
    directly by ``BindingClassificationLoss``, unlike the contrastive
    head), and ends with ``LoadFullContextEmbedding`` which pulls the frozen,
    unpooled entity-B tensor from disk via an LRU cache.

    Args:
        df: Entity-A DataFrame. Must contain at least one of
            ``sequence_heavy`` / ``sequence_light``, an ``antigen_key``
            column, and an explicit binary ``label`` column. Rows may be
            heavy-only, light-only, or both; NaN is treated as empty.
        context_dir: Directory containing ``{antigen_key}.npz`` files with
            per-residue embeddings under the key ``"embeddings"``.
        nickname: Human-readable dataset identifier.
        sequence_cols: Override for the entity-A chain columns used in the
            per-row concatenation. If ``None``, auto-detect from
            ``df.columns`` (preferring the heavy-before-light order).
        max_context_len: Optional cap on entity-B residue count; longer
            embeddings are truncated to the leading ``max_context_len``
            residues. ``None`` (default) keeps the full, untruncated tensor.
        corruption_config: Optional random input-corruption recipe (issue
            #1221, ``dasm2.data.binding_corruption.BindingCorruptionConfig``).
            When set with a nonzero ``mask_rate``, the resulting collator is
            a ``CorruptedPairwiseBindingCollator`` instead of a plain
            ``PairwiseBindingCollator``. Callers building a validation
            bundle should leave this ``None`` — see
            ``load_pairwise_binding_train_val``.

    Raises:
        ValueError: Empty DataFrame, missing columns, all-empty chains, or
            a label column with non-binary/NaN values.
        FileNotFoundError: Missing context directory or a referenced
            entity-B embedding file.
    """
    context_dir = Path(context_dir)
    if sequence_cols is None:
        sequence_cols = detect_ab_sequence_columns(df)
    df = df.reset_index(drop=True)
    # Fill any canonical sequence column missing from the schema with empty
    # strings so the separator is anchored at the same position in every
    # row of the dataset (matches create_contrastive_binding_dataset).
    for col in sequence_cols:
        if col not in df.columns:
            df[col] = ""
    _validate_pairwise_binding_df(df, context_dir, sequence_cols)

    sequences = concat_chains_with_separator(df, sequence_cols)
    context_keys = df[ANTIGEN_KEY_FIELD].astype(str).tolist()
    # Fail fast on any missing entity-B file before training starts.
    for key in set(context_keys):
        resolve_embedding_path(context_dir, key)

    labels = df[LABEL_COLUMN].astype(float).tolist()

    num_segments = len(sequence_cols)
    transforms: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = [
        TokenizeParent(),
        ComputeLayout(num_segments=num_segments),
        ComputeMasks(num_segments=num_segments),
        LoadFullContextEmbedding(
            context_dir=context_dir, max_context_len=max_context_len
        ),
    ]

    dataset = TransformDataset(
        aa_parents=sequences,
        raw_parents=sequences,
        transforms=transforms,
        output_keys=_PAIRWISE_OUTPUT_KEYS,
        extra_string_fields={ANTIGEN_KEY_FIELD: context_keys},
        extra_fields={BINARY_TARGET_KEY: labels},
        nickname=nickname,
    )

    padding_sentinel = padding_sentinel_for(num_segments)
    tensor_specs: Dict[str, Tuple[float, TensorScale]] = {
        "aa_parents_idxs": (AA_AMBIG_IDX, "codon"),
        "layout": (padding_sentinel, "codon"),
        "mask": (False, "codon"),
        "loss_mask": (False, "codon"),
    }
    collator: Union[PairwiseBindingCollator, "CorruptedPairwiseBindingCollator"]
    if corruption_config is not None and corruption_config.mask_rate > 0.0:
        # Inline to avoid a circular import: binding_corruption.py imports
        # PairwiseBindingCollator/BINDING_CONTEXT_MASK_KEY from this module.
        from dasm2.data.binding_corruption import CorruptedPairwiseBindingCollator

        collator = CorruptedPairwiseBindingCollator(
            tensor_specs=tensor_specs, corruption_config=corruption_config
        )
    else:
        collator = PairwiseBindingCollator(tensor_specs=tensor_specs)
    return DatasetBundle(
        dataset=dataset,
        collator=collator,  # type: ignore[arg-type]
        worker_init_fn=make_worker_init_fn(),
    )


def _parse_pairwise_data_spec(data_path: str) -> Tuple[Path, Path]:
    """Parse a pairwise-binding data spec into ``(csv, context_dir)``.

    Accepts ``<csv_path>:<context_dir>`` — direct paths only. The context
    dir is resolved relative to the CSV's parent when not absolute (keeps
    CSVs and their embedding directory co-located in experiment artifact
    trees, matching ``contrastive_binding``'s direct-path spec shape).

    Unlike ``contrastive_binding``, there is no registered-nickname spec
    shape here: this factory only consumes an already-built, labeled CSV
    (negative sampling and manifest wiring are out of scope per issue
    #802), so inventing a manifest schema for it would be a premature
    abstraction (Constitution IX).
    """
    if ":" not in data_path:
        raise ValueError(
            f"Pairwise-binding data spec must be '<csv_path>:<context_dir>', "
            f"got: {data_path}"
        )
    csv_path_str, context_dir_str = data_path.rsplit(":", 1)
    csv_path = Path(csv_path_str)
    context_dir = Path(context_dir_str)
    if not context_dir.is_absolute():
        context_dir = (csv_path.parent / context_dir).resolve()
    return csv_path, context_dir


def split_pairwise_binding_cv_fold(
    df: pd.DataFrame,
    num_folds: int,
    fold_index: int,
    group_col: str = DEFAULT_BINDING_CV_GROUP_COL,
    label_col: str = LABEL_COLUMN,
    strategy: Literal["cluster", "random"] = "cluster",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split via ``StratifiedGroupKFold`` (``strategy="cluster"``) or
    ``StratifiedKFold`` on ``(group_col, label_col)`` (``strategy="random"``),
    both ``shuffle=True, random_state=0``.

    ``strategy="cluster"`` (the default) deliberately does *not* reuse
    ``dataset_utils.group_kfold_in_train`` (plain ``GroupKFold``, no
    stratification): this dataset's own build script assigns folds with
    ``StratifiedGroupKFold`` so each fold's label balance stays close to
    the full dataset's, and reproducing that exact algorithm here is what
    makes this loader's fold-0 match the published per-fold counts in
    ``classification_dataset_hard.json`` bit-for-bit, rather than landing
    on an independently valid but different partition. Every fold's
    groups are disjoint from every other fold's under this strategy.

    ``strategy="random"`` stratifies on the joint ``(group_col, label_col)``
    key, keeping folds balanced on family mix and label rate while letting
    a group appear in both train and val — a broader read on how well the
    classifier ranks binding across the full antigen diversity present in
    training, not only the families withheld entirely under ``"cluster"``.
    This mirrors the *stratification key* ``experiments/2026-06-07-binding-
    classification-sabdab/scripts/run_grid.py::make_splits``'s own "random"
    branch uses, but is a flat, single-level split (same shape as
    ``"cluster"`` above) rather than ``make_splits``'s nested outer-test +
    inner-val structure — this function returns the one train/val partition
    a training run needs, not a held-out test fold on top of it.

    A real ``group_col`` (e.g. ``antigen_cluster_40``) is typically
    long-tailed, so most ``(group, label)`` strata have far fewer than
    ``num_folds`` members. ``StratifiedKFold`` tolerates this: it only
    raises when *every* stratum falls short of ``num_folds`` (an
    unsatisfiable request), and merely warns when some but not all do,
    folding the shortfall into whichever folds it can (sklearn
    ``_split.py::StratifiedKFold._make_test_folds``). Do not add a
    pre-check that raises whenever *any* stratum is sparse — that is
    stricter than sklearn's own behavior and was tried once (issue #1144
    follow-up): it broke real runs (e.g. the SAbDab binding CSV's
    long-tailed antigen families) that sklearn itself handles fine via a
    warning.

    ``groups`` is cast to ``str`` before splitting, matching
    ``build_classification_dataset.py``'s own defensive cast (that script
    casts every use of a cluster/family column the same way) — harmless
    when the column is already string-typed, but avoids a real risk if a
    future dataset variant's grouping column were numeric:
    ``StratifiedGroupKFold(shuffle=True, ...)`` orders unique groups via
    ``np.unique`` before shuffling, and numeric vs. string dtypes of the
    same values can sort differently, silently changing the fold
    assignment.
    """
    if strategy not in ("cluster", "random"):
        raise ValueError(f"strategy must be 'cluster' or 'random', got {strategy!r}")
    if group_col not in df.columns:
        raise ValueError(
            f"group_col='{group_col}' not found in DataFrame columns: "
            f"{sorted(df.columns)[:20]}"
        )
    groups = df[group_col].astype(str).to_numpy()
    labels = df[label_col].to_numpy()
    if strategy == "cluster":
        sgkf = StratifiedGroupKFold(
            n_splits=num_folds, shuffle=True, random_state=_BINDING_CV_RANDOM_STATE
        )
        folds = list(sgkf.split(np.arange(len(df)), y=labels, groups=groups))
    else:
        stratum = np.array([f"{g}||{lab}" for g, lab in zip(groups, labels)])
        skf = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=_BINDING_CV_RANDOM_STATE
        )
        folds = list(skf.split(np.arange(len(df)), y=stratum))
    train_idx, val_idx = folds[fold_index]
    if len(val_idx) == 0:
        raise ValueError(
            f"CV fold {fold_index} of {num_folds} on '{group_col}' "
            f"(strategy={strategy!r}) produced an empty validation set on "
            f"{len(df)} rows. Use fewer folds or a larger dataset."
        )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    cv_split_logger.info(
        "Binding CV split (strategy=%s): fold %d of %d on '%s' "
        "(%d train, %d val, %d unique groups)",
        strategy,
        fold_index,
        num_folds,
        group_col,
        len(train_df),
        len(val_df),
        len(set(groups)),
    )
    return train_df, val_df


def load_pairwise_binding_train_val(
    data_path: str,
    seed: int,
    train_fraction: float = 0.8,
    cv_split: Optional["CvSplitConfig"] = None,
    max_context_len: Optional[int] = None,
    corruption_config: Optional["BindingCorruptionConfig"] = None,
) -> Tuple[DatasetBundle, DatasetBundle]:
    """Load CSV + context dir and return ``(train_bundle, val_bundle)``.

    ``data_path`` is ``<csv_path>:<context_dir>``; the context dir is
    resolved relative to the CSV's parent if not absolute.

    ``corruption_config`` (issue #1221), if given with a nonzero
    ``mask_rate``, is applied to the *training* bundle only — the
    validation bundle always uses a plain, uncorrupted collator, since
    ``OverfitGuardHook``'s freeze decision and this feature's whole point
    (does corruption delay overfitting on *held-out* pairs?) both depend on
    genuine, uncorrupted validation loss.

    When ``cv_split`` is set, the split is via
    :func:`split_pairwise_binding_cv_fold` (``DEFAULT_BINDING_CV_GROUP_COL``,
    stratified by ``label``, matching this dataset's own build script
    exactly for the default ``strategy``). ``num_folds``/``fold_index``/
    ``binding_split_strategy`` are taken from ``cv_split``; ``train_fraction``,
    ``cv_split.group_col``, and ``cv_split.seed`` are all ignored in this branch —
    ``DEFAULT_BINDING_CV_GROUP_COL`` is a fixed property of this CSV's own
    schema, not the same knob as a co-trained dataset's ``cv_split.group_col``,
    which may need to mean something entirely different, e.g. ``sample_id``
    for an evolutionary dataset trained jointly with this one; and the
    split's randomness is pinned to this dataset's own build-script seed,
    not the caller's.

    Otherwise, falls back to :func:`split_df_train_val`: group-aware (via
    a ``group_id`` column) if present, otherwise a random row-level split.
    """
    csv_path, context_dir = _parse_pairwise_data_spec(data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Pairwise-binding CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    sequence_cols = detect_ab_sequence_columns(df)

    if cv_split is not None:
        # split_pairwise_binding_cv_fold already raises a precise
        # ValueError if the fold produces an empty val set, so no
        # generic empty-split guard is needed for this branch.
        train_df, val_df = split_pairwise_binding_cv_fold(
            df,
            num_folds=cv_split.num_folds,
            fold_index=cv_split.fold_index,
            strategy=cv_split.binding_split_strategy,
        )
    else:
        train_df, val_df = split_df_train_val(
            df, seed=seed, train_fraction=train_fraction
        )
        if len(train_df) == 0 or len(val_df) == 0:
            n_groups = len(set(df["group_id"])) if "group_id" in df.columns else "n/a"
            raise ValueError(
                f"Pairwise-binding split produced empty side: "
                f"train={len(train_df)}, val={len(val_df)}. "
                f"Check train_fraction={train_fraction} vs dataset size "
                f"({len(df)} rows, {n_groups} groups)."
            )

    train_bundle = create_pairwise_binding_dataset(
        df=train_df,
        context_dir=context_dir,
        nickname=f"{csv_path.stem}_train",
        sequence_cols=sequence_cols,
        max_context_len=max_context_len,
        corruption_config=corruption_config,
    )
    val_bundle = create_pairwise_binding_dataset(
        df=val_df,
        context_dir=context_dir,
        nickname=f"{csv_path.stem}_val",
        sequence_cols=sequence_cols,
        max_context_len=max_context_len,
        # No corruption_config here — see this function's own docstring.
    )
    return train_bundle, val_bundle


__all__ = [
    "AgPoolMode",
    "BINDING_CONTEXT_EMBEDDING_KEY",
    "BINDING_CONTEXT_MASK_KEY",
    "DEFAULT_BINDING_CV_GROUP_COL",
    "LABEL_COLUMN",
    "LoadFullContextEmbedding",
    "PairwiseBindingCollator",
    "create_pairwise_binding_dataset",
    "entity_a_mask_from_batch",
    "load_pairwise_binding_train_val",
    "pool_antigen_context",
    "split_pairwise_binding_cv_fold",
]
