"""Minimal MLM dataset + data-loading bridge (issue #884 / #851, Phase 4).

``dasm2/data/evolutionary_dataset.py::TransformDataset`` is not a generic
sequence container — its constructor is structurally a parent/child
NT-mutation engine (paired ``raw_parents``/``raw_children``, translated
``aa_parents``, branch lengths, neutral-model tensors). MLM training
examples are individual observed sequences with no parent/child pairing, no
branch length, and no neutral model anywhere, so forcing them through
``TransformDataset`` would mean inventing a fake ``raw_child`` purely to
satisfy a constructor built for mutation-selection pairs. ``MLMDataset``
below is a dedicated, minimal ``torch.utils.data.Dataset`` instead: a flat
list of AA sequence strings in, one tokenized (CLS-prefixed) sample out.

Data source for this phase only: evolutionary/PCP datasets are the only
resource in this repo that currently expose productive BCR sequences at
all, so ``load_mlm_train_val`` sources sequences by reading the existing PCP
CSV manifest, filtering to ``child_is_leaf == True`` (an actual sequenced
read, not a tree-inferred internal node), and extracting only the child
sequence(s) from each surviving row — everything else the PCP schema
carries (parent, branch_length, v_gene, ...) is discarded immediately.
Column detection and chain-joining go through
``dasm2.data.segment_handlers.SegmentHandler`` (the same machinery
``dasm2/data/evolutionary_dataset.py`` uses for DASM training data) — see
``_translated_child_sequences`` below. This is a deliberate, temporary
bridge, not a permanent data pipeline: it will be replaced once a dedicated
productive-sequence manifest exists (tracked against #851).

Multiname resolution (issue #923): ``data_path`` is resolved component-wise
via ``dasm2.data.registry.resolve_component_specs``/``load_dataframe`` — the
same multiname ("a+b") parsing every other loader uses — then concatenated
before the ``child_is_leaf`` filter. This deliberately does *not* go through
``sequence_df_from_spec``/``_load_component_dataframes``: those apply
``detect_compatible_handler(columns, ("parent", "child"))`` unconditionally,
which requires parent-segment columns to exist — a requirement MLM's
genuinely parent-agnostic data path (see above) must not inherit, since MLM
discards the parent side entirely and some callers (including this module's
own tests) legitimately pass child-only data. Bypassing that pipeline does
*not*, however, mean skipping its row filtering: as of issue #1125 this
module applies the registry's naughty-clonal-family filtering through the
shared ``dasm2.data.registry.preprocess_rows_for_split`` seam, calling the
same ``filter_naughty_clonal_families`` DASM calls rather than
re-implementing it (the filter reads only ``sample_id``/``family`` plus the
nickname, so it was only ever *bundled* into ``_load_component_dataframes``,
never coupled to the parent/child handler detection). Naughty clonal
families are therefore dropped from MLM's training data too — the same
family universe DASM trains on. This reverses the pre-#1125 behavior, which
kept them.

Cross-validation (issues #923, #1125): ``create_mlm_dataset``/
``load_mlm_train_val`` accept an optional ``cv_split``
(``dasm2.config.CvSplitConfig``). Only group-aware CV is supported here
(``cv_split.group_col`` must be set, e.g. ``"sample_id"``) — MLM has no
row-level CV use case today, so that mode (supported by the
evolutionary/whichmut CV path) is deliberately not built here per
Constitution IX (Simplicity). The GroupKFold partitioning itself reuses
``dasm2.data.dataset_utils.group_kfold_in_train``, the same helper
``dasm2/data/evolutionary_dataset.py`` uses for its own CV path.

**Ordering matters for fold parity with DASM.** ``load_mlm_train_val``
computes that partition over the *full, pre-``child_is_leaf``* row set and
only then restricts the resulting per-row mask to the leaf rows it trains
on. sklearn's ``GroupKFold`` assigns each group to the lightest fold so far,
keyed on per-group row counts, so splitting the ``child_is_leaf``-filtered
rows (as this module did before #1125) yields a *different*
``sample_id -> fold`` map than the DASM/whichmut path, which splits the full
frame in ``evolutionary_dataset._load_and_split``. That silently mis-paired
fold-for-fold MLM-vs-DASM comparisons (see #1125's audit of PR #963).
Splitting the full frame — with the same row-dropping preprocessing applied
— makes the two maps identical by construction.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dasm2.data.dataset_utils import group_kfold_in_train, split_df_train_val
from dasm2.data.evolutionary_dataset import (
    DatasetBundle,
    make_worker_init_fn,
)
from dasm2.data.masking import TOKEN_IDS_KEY, MaskedTransformCollator, MaskingConfig
from dasm2.data.registry import (
    load_dataframe,
    preprocess_rows_for_split,
    resolve_component_specs,
)
from dasm2.data.segment_handlers import detect_compatible_handler
from dasm2.sequence.conversion import translate_sequence
from dasm2.sequence.mlm_constants import MLM_PAD_IDX
from dasm2.tokenizers import MLM_TOKENIZER

if TYPE_CHECKING:
    from dasm2.config import CvSplitConfig


def _translated_child_sequences(df: pd.DataFrame) -> List[str]:
    """Combine and translate the child sequence(s) in ``df`` to AA.

    ``detect_compatible_handler`` picks ``AntibodySegmentHandler``
    (``child_heavy`` [+ ``child_light``]) or the generic single-segment
    ``SegmentHandler`` (bare ``child``) from column presence, and
    ``handler.prepare_sequences`` combines per-segment NT columns with
    ``NT_SEPARATOR`` (``"^^^"``) before translation. ``translate_sequence``'s
    codon-level ``TOKEN_CODON_MAPPING`` turns each ``NT_SEPARATOR`` codon
    into a single ``AA_SEPARATOR`` (``"^"``) as part of the same translation
    pass, so no separate AA-level join step is needed.

    Whether ``^`` appears in the output is decided once, by which segment
    handler matches the DataFrame's columns — never per-row. A heavy-only
    schema (only ``child_heavy`` present) still matches
    ``AntibodySegmentHandler``; ``validate_segment_format`` fills the
    missing light segment with ``""``, which still gets joined with the
    separator, so every sequence from a heavy-only source carries a
    trailing ``^``.
    """
    handler = detect_compatible_handler(set(df.columns), ("child",))
    prepared_df = handler.prepare_sequences(df, base_column_names=("child",))
    return [translate_sequence(nt) for nt in prepared_df["child"]]


class MLMDataset(Dataset):
    """Minimal in-memory dataset for the standalone MLM training objective.

    Constructed directly from a flat list of AA sequence strings — no
    parent/child pairing, no branch lengths, no neutral model, and no
    dependence on ``TransformDataset``. Tokenization (including the CLS
    prefix, and the ``^`` chain separator for already-joined paired
    sequences) happens per-sample in ``__getitem__`` via ``MLM_TOKENIZER``.
    Masking happens later, per batch, in ``MaskedTransformCollator`` — never
    here, so every epoch draws a fresh mask.
    """

    row_filter_digest: Optional[str] = None
    """Never row-filtered; part of the ``CacheableDataset`` protocol (#1672)."""

    def __init__(self, sequences: List[str], nickname: str = "unnamed_mlm") -> None:
        if not sequences:
            raise ValueError("MLMDataset requires at least one sequence.")
        self.sequences = list(sequences)
        self.nickname = nickname

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = MLM_TOKENIZER.tokenize([self.sequences[idx]])[0]
        return {TOKEN_IDS_KEY: tokens}


def _build_dataset_bundle(
    sequences: List[str], nickname: str, masking_config: Optional[MaskingConfig]
) -> DatasetBundle:
    dataset = MLMDataset(sequences, nickname=nickname)
    collator = MaskedTransformCollator(
        tensor_specs={TOKEN_IDS_KEY: (MLM_PAD_IDX, "codon")},
        masking_config=masking_config or MaskingConfig(),
    )
    return DatasetBundle(
        dataset=dataset,
        collator=collator,  # type: ignore[arg-type]
        worker_init_fn=make_worker_init_fn(),
    )


def require_group_aware_cv_split(cv_split: "CvSplitConfig") -> str:
    """Return ``cv_split.group_col``, raising unless it is set.

    MLM training only supports group-aware CV — row-level (ungrouped) CV,
    which the evolutionary/whichmut CV path does support, is deliberately
    not built here per Constitution IX (Simplicity; no current MLM use case
    needs it). Shared by ``create_mlm_dataset``, ``load_mlm_train_val``, and
    ``train_command.py``'s pre-flight CV validation so the three call sites
    agree on one wording and one place to change it. Returns the validated
    ``group_col`` so a caller that needs it as a plain ``str`` gets it already
    narrowed; the validation-only call sites ignore the return.
    """
    if cv_split.group_col is None:
        raise ValueError(
            "MLM cv_split requires group_col to be set (e.g. 'sample_id'); "
            "row-level (ungrouped) CV is not supported for MLM training."
        )
    return cv_split.group_col


def create_mlm_dataset(
    sequences: List[str],
    group_ids: List[Any],
    seed: int,
    train_fraction: float = 0.8,
    masking_config: Optional[MaskingConfig] = None,
    cv_split: Optional["CvSplitConfig"] = None,
    in_train_mask: Optional[Union[Sequence[bool], np.ndarray]] = None,
) -> Tuple[DatasetBundle, DatasetBundle]:
    """Build train/val ``DatasetBundle``s from a flat list of productive sequences.

    ``group_ids`` is whichever grouping key the active split strategy needs:
    clonal ``family`` for the default random split, or the CV fold spec's
    ``group_col`` (e.g. ``sample_id``) when ``cv_split`` is set. Either way,
    every sequence sharing a group id lands on the same side of the
    train/val boundary — the same convention
    ``create_contrastive_binding_dataset``/``create_pairwise_binding_dataset``
    already use for the same leakage-avoidance reason.

    When ``cv_split`` is ``None`` (default), splits via ``split_df_train_val``
    (random group-aware 80/20 by default). When ``cv_split`` is set, splits
    via ``dasm2.data.dataset_utils.group_kfold_in_train`` instead — this
    requires ``cv_split.group_col`` to be set; row-level (ungrouped) CV is
    not supported for MLM training (see module docstring).

    ``in_train_mask`` lets a caller supply the realized split per sequence;
    when given it is the split, and ``seed``/``train_fraction`` are unused.
    ``load_mlm_train_val`` passes either a ``cv_split`` partition computed
    over the *full* pre-``child_is_leaf`` row set and restricted to the leaf
    subset — the fold parity with DASM described in the module docstring
    (issue #1125) — or the data's own ``in_train`` column (issue #1386).
    ``masking_config`` defaults to the standard BERT recipe (``MaskingConfig()``)
    if not provided.
    """
    if len(sequences) != len(group_ids):
        raise ValueError(
            f"sequences and group_ids must have the same length, got "
            f"{len(sequences)} and {len(group_ids)}."
        )
    if in_train_mask is not None:
        if len(in_train_mask) != len(sequences):
            raise ValueError(
                f"in_train_mask and sequences must have the same length, got "
                f"{len(in_train_mask)} and {len(sequences)}."
            )
    df = pd.DataFrame({"sequence": sequences, "group_id": group_ids})

    if cv_split is not None:
        require_group_aware_cv_split(cv_split)
    in_train: Optional["np.ndarray"] = None
    if in_train_mask is not None:
        in_train = np.asarray(in_train_mask, dtype=bool)
    elif cv_split is not None:
        in_train = group_kfold_in_train(df["group_id"].to_numpy(), cv_split)
    if in_train is not None:
        train_df = df[in_train].reset_index(drop=True)
        val_df = df[~in_train].reset_index(drop=True)
    else:
        train_df, val_df = split_df_train_val(
            df, seed=seed, train_fraction=train_fraction
        )

    if len(train_df) == 0 or len(val_df) == 0:
        # Only the random split reads train_fraction and group ids; a supplied
        # or cv_split mask is the split as given (the pre-split path's group
        # ids are all None).
        hint = (
            "Check the supplied in_train_mask / cv_split assignment."
            if in_train is not None
            else f"Check train_fraction={train_fraction} vs dataset size "
            f"({len(df)} rows, {df['group_id'].nunique()} groups)."
        )
        raise ValueError(
            f"MLM split produced empty side: train={len(train_df)}, "
            f"val={len(val_df)}. {hint}"
        )
    train_bundle = _build_dataset_bundle(
        train_df["sequence"].tolist(), "mlm_train", masking_config
    )
    val_bundle = _build_dataset_bundle(
        val_df["sequence"].tolist(), "mlm_val", masking_config
    )
    return train_bundle, val_bundle


def _leaf_in_train_mask(
    split_frames: List[pd.DataFrame],
    group_col: str,
    cv_split: "CvSplitConfig",
) -> "np.ndarray":
    """Realize ``cv_split``'s GroupKFold over the full pre-leaf rows, then
    restrict the per-row mask to the ``child_is_leaf`` rows (issue #1125).

    ``split_frames`` are the per-component frames as they stand *after*
    ``registry.preprocess_rows_for_split`` and *before* the
    ``child_is_leaf`` filter — the row set the DASM path splits — each
    carrying ``group_col`` and ``child_is_leaf``. They are concatenated and
    split in a single pass (the DASM path likewise splits the concatenated
    frame, so a group spanning two components carries its summed weight),
    then the resulting mask is indexed *positionally* by the leaf mask read
    off that same combined frame. Positional indexing — rather than a
    ``group -> side`` dict — stays exact even when components disagree on the
    group column's dtype, since it reads the same concatenated labels
    ``GroupKFold`` saw. Group-leakage safety comes for free: ``GroupKFold``
    puts a whole group on one side, and taking a subset of rows cannot split
    one.
    """
    split_df = pd.concat(split_frames, ignore_index=True)
    in_train_full = group_kfold_in_train(split_df[group_col].to_numpy(), cv_split)
    leaf_in_train: "np.ndarray" = in_train_full[
        split_df["child_is_leaf"].to_numpy(dtype=bool)
    ]
    return leaf_in_train


def load_mlm_train_val(
    data_path: str,
    seed: int,
    train_fraction: float = 0.8,
    masking_config: Optional[MaskingConfig] = None,
    cv_split: Optional["CvSplitConfig"] = None,
) -> Tuple[DatasetBundle, DatasetBundle]:
    """Load train/val MLM ``DatasetBundle``s from a PCP-schema CSV/multiname.

    ``data_path`` is resolved component-wise (see module docstring for why
    this doesn't go through ``sequence_df_from_spec``), supporting the same
    ``"a+b"`` multiname syntax the DASM/Whichmut loaders use. See the module
    docstring for why this reads the PCP schema directly rather than
    through ``TransformDataset``/the evolutionary-dataset loading path.
    ``masking_config`` defaults to the standard BERT recipe if not provided
    (``train_command.py`` builds one from ``MLMLossConfig``'s YAML fields).

    ``cv_split``, when set, must have ``group_col`` set (e.g. ``"sample_id"``)
    — see ``create_mlm_dataset``. Its GroupKFold partition is computed over
    the full pre-``child_is_leaf`` row set and then restricted to the leaf
    rows, so the realized ``group_col -> fold`` map matches the DASM/whichmut
    one (issue #1125; see the module docstring for why the ordering decides
    this). When ``cv_split`` is ``None`` (default), falls back to the
    original random 80/20 split grouped on clonal ``family`` — unless every
    component already carries an ``in_train`` column, in which case that
    column is the split, ``family`` is not required and ``seed``/
    ``train_fraction`` are unused (issue #1386). This shares the DASM
    path's pre-split semantics (``registry.add_train_val_markers``: an
    existing ``in_train`` is used as-is) but not its validation: this path
    rejects a non-boolean or incomplete ``in_train``, which the registry path
    does not check. Components must agree: either all carry ``in_train`` or
    none does.
    """

    # Validate and translate each multiname component individually, before
    # concatenating anything. Two reasons:
    # - pd.concat unions columns, so a component missing the required group
    #   column would otherwise pass a post-concat check (the other
    #   components' column masks the gap) while its own rows silently get
    #   NaN group ids, surfacing later as a confusing type error deep inside
    #   the group-split machinery instead of a clear message naming the
    #   actual bad component.
    # - _translated_child_sequences's detect_compatible_handler call picks
    #   a chain schema (single-chain vs. paired heavy/light) from column
    #   presence. Concatenating raw components first would union their
    #   columns too, so a single-chain component (only `child`) combined
    #   with a paired component (`child_heavy`/`child_light`) would resolve
    #   one handler for both, silently mistranslating whichever component
    #   doesn't match it. Translating per component (each against only its
    #   own columns) keeps that resolution correct.
    component_specs = resolve_component_specs(data_path)
    sequences: List[str] = []
    group_ids: List[Any] = []
    # The full (pre-leaf) rows each component contributes to the CV split,
    # narrowed to the two columns _leaf_in_train_mask reads, so the partition
    # can be computed on the DASM row set and then restricted to the leaf
    # rows (issue #1125). Only populated when cv_split is set; the
    # random-split path never reads them.
    split_frames: List[pd.DataFrame] = []
    # Each component's leaf-row in_train values, on the pre-split path only.
    presplit_parts: List["np.ndarray"] = []
    # Decided by the first component; every later one must agree.
    presplit: Optional[bool] = None
    cv_group_col = (
        require_group_aware_cv_split(cv_split) if cv_split is not None else None
    )
    for nickname, path in component_specs:
        component_df = load_dataframe(path)
        has_in_train = cv_split is None and "in_train" in component_df.columns
        if presplit is None:
            presplit = has_in_train
        elif has_in_train != presplit:
            raise ValueError(
                f"MLM data source {data_path!r} mixes components with and "
                f"without an 'in_train' column (at {nickname!r}); either every "
                f"component is pre-split or none is."
            )
        required_group_col = cv_group_col or (None if presplit else "family")
        if "child_is_leaf" not in component_df.columns:
            raise ValueError(
                f"MLM data component {nickname!r} ({path}) has no "
                f"'child_is_leaf' column; cannot filter to observed leaf "
                f"sequences."
            )
        if (
            required_group_col is not None
            and required_group_col not in component_df.columns
        ):
            reason = (
                "needed for cv_split's GroupKFold"
                if cv_split is not None
                else "needed for a group-aware train/val split when there is "
                "no 'in_train' column"
            )
            raise ValueError(
                f"MLM data component {nickname!r} ({path}) has no "
                f"'{required_group_col}' column, {reason}."
            )

        # The parity seam (issue #1125): drop exactly the rows DASM drops
        # before *it* splits, on the full pre-leaf frame. This both removes
        # naughty clonal families from MLM's training data (so MLM trains on
        # the same family universe as DASM) and keeps the per-group row
        # counts GroupKFold packs on identical between the two paths.
        component_df = preprocess_rows_for_split(component_df, nickname)

        # A pickled frame can carry pandas' nullable "boolean" dtype, which
        # passes is_bool_dtype even with missing values, so check NA too.
        if presplit and (
            not pd.api.types.is_bool_dtype(component_df["in_train"])
            or component_df["in_train"].isna().any()
        ):
            raise ValueError(
                f"MLM data component {nickname!r} ({path}) has a non-boolean "
                f"or incomplete 'in_train' column "
                f"(dtype {component_df['in_train'].dtype})."
            )

        if cv_group_col is not None:
            # Accumulated before the leaf filter, and for every component —
            # even one contributing no leaf sequences at all — since such a
            # component still carries rows (and hence per-group weight) into
            # the split frame, exactly as it does on the DASM path. Both the
            # split rows and the training rows below are read off this one
            # preprocessed frame, in order, so they cannot drift apart.
            split_frames.append(component_df[[cv_group_col, "child_is_leaf"]])

        component_leaf_mask = component_df["child_is_leaf"].to_numpy(dtype=bool)
        leaf_df = component_df[component_leaf_mask].reset_index(drop=True)
        if len(leaf_df) == 0:
            continue

        sequences.extend(_translated_child_sequences(leaf_df))
        group_ids.extend(
            leaf_df[required_group_col].tolist()
            if required_group_col is not None
            else [None] * len(leaf_df)
        )
        if presplit:
            presplit_parts.append(leaf_df["in_train"].to_numpy(dtype=bool))

    if not sequences:
        raise ValueError(f"MLM data source {data_path!r} has no child_is_leaf rows.")

    # One entry per leaf row, in the same order `sequences` was collected in.
    # create_mlm_dataset validates the two lengths against each other.
    in_train_mask: Optional["np.ndarray"] = None
    if cv_group_col is not None:
        assert cv_split is not None
        in_train_mask = _leaf_in_train_mask(split_frames, cv_group_col, cv_split)
    elif presplit:
        in_train_mask = np.concatenate(presplit_parts)

    return create_mlm_dataset(
        sequences,
        group_ids,
        seed=seed,
        train_fraction=train_fraction,
        masking_config=masking_config,
        cv_split=cv_split,
        in_train_mask=in_train_mask,
    )


__all__ = [
    "MLMDataset",
    "create_mlm_dataset",
    "load_mlm_train_val",
    "require_group_aware_cv_split",
]
