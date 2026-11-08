"""Contrastive binding dataset + collator (issue #362).

Follows the project-wide data pattern: a single ``TransformDataset`` in
streaming mode backed by a list of per-sample transforms, wrapped by a
head-specific collator.

Ab path (per-sample transforms, run in ``__getitem__``):
    TokenizeParent → ComputeLayout → ComputeMasks
    → RenameKey(loss_mask, ab_pool_mask)
    → LoadAntigenEmbedding(antigen_dir)

Antibody heavy/light handling matches the evolutionary datasets: per-row,
non-empty chains are joined with ``AA_SEPARATOR`` (``^``), so the shared
``ComputeLayout`` / ``ComputeMasks`` transforms produce the correct
per-segment layout and pooling mask. Rows may carry heavy-only, light-only,
or both chains; the schema (which columns exist in the DataFrame)
determines ``num_segments`` for the whole dataset.

Ag path: ``antigen_key`` is injected into the seed dict via
``TransformDataset.extra_string_fields``, then ``LoadAntigenEmbedding``
resolves it to ``antigen_embeddings/{key}.npz`` through a process-local
LRU cache AND mean-pools the residue dimension on the spot. This keeps
per-residue NPZ layout flexible on disk while ensuring the per-sample
output is a fixed ``[D_esm]`` vector — which means no per-residue padding,
no Ag-side mask, and no pooling call in the head.

Collator: delegates Ab-side padding plus Ag-side stacking to
``TransformCollator`` (the pre-pooled antigen uses ``scale="scalar"``,
which ``torch.stack``s 1-D vectors into ``[B, D_esm]``). It then adds the
``positive_pair_matrix`` built from antigen-key equality.

``loss_mask`` is renamed to ``ab_pool_mask`` because it is the antibody pooling
mask here, not a loss-position mask. Per-pair normalization comes from
``ContrastiveBindingLoss.n_loss_labels`` (the batch also carries the padding
``mask``, which the trainer would otherwise count residue by residue).
"""

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
    get_args,
)

import numpy as np
import pandas as pd
import torch
import yaml

from dasm2.config import PositiveRule
from dasm2.data.collators import TensorScale, TransformCollator, split_tensor_samples
from dasm2.data.dataset_utils import (
    clean_chain_series,
    concat_chains_with_separator,
    group_kfold_in_train,
    nested_group_kfold_split,
    split_df_train_val,
)
from dasm2.data.embedding_io import cached_residue_tensor, resolve_embedding_path
from dasm2.data.evolutionary_dataset import (
    DatasetBundle,
    TransformDataset,
    make_worker_init_fn,
)
from dasm2.data.structural_embedding_registry import get_structural_embedding_config
from dasm2.data.transforms import (
    ComputeLayout,
    ComputeMasks,
    RenameKey,
    TokenizeParent,
)
from dasm2.sequence.constants import AA_AMBIG_IDX
from dasm2.sequence.layout import padding_sentinel_for

if TYPE_CHECKING:
    from dasm2.config import CvSplitConfig

# Reduction used by ``LoadAntigenEmbedding`` to collapse the residue axis of
# a ``[L_ag, D_esm]`` antigen embedding to a fixed ``[D_esm]`` vector.
# ``"mean"`` is the CALM-style masked mean; ``"first"`` / ``"last"`` select
# a single residue, which is useful when the source model prepends a CLS
# token (ESM-2) or uses an end-of-sequence aggregator.
AgPoolingMode = Literal["mean", "first", "last"]

# Key of the per-sample, mean-pooled antigen vector (shape ``[D_esm]``).
ANTIGEN_EMBEDDING_KEY = "antigen_embedding"
# ``[B, B]`` bool produced by the collator.
POSITIVE_PAIR_MATRIX_KEY = "positive_pair_matrix"
# Per-sample string carried through ``extra_string_fields`` so downstream
# transforms and the collator can read it.
ANTIGEN_KEY_FIELD = "antigen_key"
# Per-sample identity strings, carried only under the "sequence" positive rule.
ANTIGEN_SEQUENCE_FIELD = "antigen_sequence"
ANTIBODY_SEQUENCE_FIELD = "antibody_sequence"
# CSV column holding the antigen's amino-acid sequence.
ANTIGEN_SEQUENCE_COLUMN = "sequence_antigen"
# Which pairs of a batch count as positives (``PositiveRule``, defined in
# ``dasm2.config`` next to the loss config that selects it):
# - ``"antigen_key"``: rows with the same ``antigen_key``.
# - ``"sequence"``: rows with an identical antigen sequence OR an identical
#   antibody (heavy + light) sequence. Two antigen ids with the same
#   sequence are positives; two antigens never are merely because they share
#   a cluster.
_VALID_POSITIVE_RULES: Tuple[str, ...] = get_args(PositiveRule)
_SEQUENCE_FIELDS = frozenset({ANTIGEN_SEQUENCE_FIELD, ANTIBODY_SEQUENCE_FIELD})
# Renamed ``loss_mask`` used as the Ab-side pooling mask. Not ``loss_mask``
# because it selects residues to pool, not loss positions.
AB_POOL_MASK_KEY = "ab_pool_mask"

# Candidate antibody-sequence columns. Order is significant only for
# concatenation — heavy comes before light in the combined sequence.
_AB_SEQUENCE_COLUMNS: Tuple[str, ...] = ("sequence_heavy", "sequence_light")

_CONTRASTIVE_OUTPUT_KEYS = frozenset(
    {
        "aa_parents_idxs",
        "layout",
        "mask",
        AB_POOL_MASK_KEY,
        ANTIGEN_EMBEDDING_KEY,
        ANTIGEN_KEY_FIELD,
    }
)


# ---------------------------------------------------------------------------
# Sequence-column handling (reuses the evolutionary per-row concatenation
# pattern: AA_SEPARATOR joins whichever chain columns are present).
# ---------------------------------------------------------------------------


def detect_ab_sequence_columns(df: pd.DataFrame) -> List[str]:
    """Return the canonical antibody sequence columns when at least one is
    present.

    Mirrors ``AntibodySegmentHandler``: heavy/light is a structurally
    2-segment dataset, so the canonical column list is always
    ``("sequence_heavy", "sequence_light")`` whenever the schema declares an
    antibody. Rows missing one chain are tolerated (the absent column is
    filled with empty strings before concatenation), which keeps
    ``num_segments = 2`` and the ``^`` separator anchored at the boundary
    between heavy and light positions for every row in the dataset.

    Shared by both ``contrastive_binding.py`` and ``pairwise_binding.py``,
    so the message names the DataFrame generically rather than either
    module's dataset flavor.
    """
    has_any = any(c in df.columns for c in _AB_SEQUENCE_COLUMNS)
    if not has_any:
        raise ValueError(
            f"DataFrame has no antibody sequence columns. "
            f"Expected at least one of {_AB_SEQUENCE_COLUMNS}. "
            f"Got: {list(df.columns)}"
        )
    return list(_AB_SEQUENCE_COLUMNS)


# ---------------------------------------------------------------------------
# Transform: resolve antigen_key → antigen_embeddings + antigen_mask
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadAntigenEmbedding:
    """Resolve ``sample["antigen_key"]`` to a pre-pooled ESM-2 vector.

    Reads ``{antigen_dir}/{antigen_key}.npz`` via a module-level LRU cache so
    repeated antigens across samples are deserialized exactly once per
    worker. Then collapses the per-residue tensor along its length axis via
    ``ag_pooling_mode`` so downstream code sees a fixed-shape ``[D_esm]``
    vector regardless of antigen length. Populates:
        - ``antigen_embedding``: ``[D_esm]`` float32 tensor.

    Supported modes:
        - ``"mean"`` (default): unweighted mean over all residues — matches
          CALM.
        - ``"first"``: select residue 0. Natural choice when the NPZ was
          produced by a model that prepends a CLS aggregator (ESM-2 with
          the ``<cls>`` token included).
        - ``"last"``: select residue ``L_ag - 1``. Natural choice for
          end-of-sequence aggregators (e.g. autoregressive LMs).

    Residue-level NPZs are preserved on disk so pooling can change without
    re-running the frozen ESM-2 encoder.
    """

    antigen_dir: Path
    ag_pooling_mode: AgPoolingMode = "mean"

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        key = sample[ANTIGEN_KEY_FIELD]
        path = resolve_embedding_path(self.antigen_dir, key)
        residues = cached_residue_tensor(str(path))  # [L_ag, D_esm]
        if self.ag_pooling_mode == "mean":
            pooled = residues.mean(dim=0)
        elif self.ag_pooling_mode == "first":
            pooled = residues[0]
        elif self.ag_pooling_mode == "last":
            pooled = residues[-1]
        else:  # guarded by Literal but keep a runtime fail-fast
            raise ValueError(
                f"ag_pooling_mode must be one of 'mean', 'first', 'last'; "
                f"got {self.ag_pooling_mode!r}."
            )
        sample[ANTIGEN_EMBEDDING_KEY] = pooled  # [D_esm]
        return sample


# ---------------------------------------------------------------------------
# Collator — wraps TransformCollator with antigen padding + pair-matrix build
# ---------------------------------------------------------------------------


@dataclass
class ContrastiveBindingCollator:
    """Thin wrapper over ``TransformCollator`` that adds the in-batch
    ``positive_pair_matrix``.

    ``TransformCollator`` handles all per-sample tensors (Ab-side codon-scale
    tensors + the pre-pooled ``antigen_embedding`` via ``scale="scalar"``).
    The only batch-level concern left for this class is building
    ``positive_pair_matrix`` from per-sample strings, per ``positive_rule``:

    - ``"antigen_key"``: ``[i, j] = (keys[i] == keys[j])``.
    - ``"sequence"``: ``[i, j] = (antigen_seq[i] == antigen_seq[j]) or
      (antibody_seq[i] == antibody_seq[j])``.
    """

    tensor_specs: Dict[str, Tuple[float, TensorScale]] = field(default_factory=dict)
    positive_rule: PositiveRule = "antigen_key"

    def __post_init__(self) -> None:
        _check_positive_rule(self.positive_rule)
        self._inner = TransformCollator(tensor_specs=self.tensor_specs)

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # TransformCollator pads/stacks only the tensor keys; the per-sample
        # strings are handled explicitly below.
        extra_keys = {ANTIGEN_KEY_FIELD}
        if self.positive_rule == "sequence":
            extra_keys |= _SEQUENCE_FIELDS
        tensor_samples = split_tensor_samples(
            samples,
            self.tensor_specs,
            frozenset(extra_keys),
            "ContrastiveBindingCollator",
        )
        batch = self._inner(tensor_samples)

        antigen_keys = [s[ANTIGEN_KEY_FIELD] for s in samples]
        if self.positive_rule == "sequence":
            positives = positive_pair_matrix(
                [s[ANTIBODY_SEQUENCE_FIELD] for s in samples],
                [s[ANTIGEN_SEQUENCE_FIELD] for s in samples],
            )
        else:
            positives = _equality_matrix(antigen_keys)
        batch[POSITIVE_PAIR_MATRIX_KEY] = torch.from_numpy(positives)
        batch[ANTIGEN_KEY_FIELD] = antigen_keys
        return batch


def _check_positive_rule(rule: str) -> None:
    if rule not in _VALID_POSITIVE_RULES:
        raise ValueError(
            f"positive_rule must be one of {_VALID_POSITIVE_RULES}; got {rule!r}."
        )


def _equality_matrix(values: Sequence[str]) -> np.ndarray:
    """``[N, N]`` bool: ``[i, j] = values[i] == values[j]``."""
    array = np.asarray(values)
    equal: np.ndarray = array[:, None] == array[None, :]
    return equal


def positive_pair_matrix(
    antibody_ids: Sequence[str], antigen_ids: Sequence[str]
) -> np.ndarray:
    """``[N, N]`` bool for the ``"sequence"`` rule: same antibody OR same antigen.

    The single definition of a sequence positive: the collator uses it to
    build training batches and ``dasm2 contrastive-eval`` to score retrieval.
    Symmetric, with a True diagonal.
    """
    positives: np.ndarray = _equality_matrix(antibody_ids) | _equality_matrix(
        antigen_ids
    )
    return positives


def pair_identities(
    df: pd.DataFrame, sequence_cols: Optional[Sequence[str]] = None
) -> Tuple[List[str], List[str]]:
    """Per-pair ``(antibody_ids, antigen_ids)`` for the ``"sequence"`` rule.

    The antibody identity is the heavy^light concatenation the model sees; the
    antigen identity is the ``sequence_antigen`` column. Fails when that column
    is missing or any antigen sequence is empty, since all empty sequences
    would compare equal and become one giant positive group.
    """
    if ANTIGEN_SEQUENCE_COLUMN not in df.columns:
        raise ValueError(
            f"positive_rule='sequence' needs a '{ANTIGEN_SEQUENCE_COLUMN}' "
            f"column. Got: {list(df.columns)}"
        )
    antigen_ids = clean_chain_series(df, ANTIGEN_SEQUENCE_COLUMN)
    if (antigen_ids == "").any():
        raise ValueError(
            f"positive_rule='sequence' needs a non-empty "
            f"'{ANTIGEN_SEQUENCE_COLUMN}' in every row: empty sequences "
            f"would all count as one antigen."
        )
    if sequence_cols is None:
        sequence_cols = detect_ab_sequence_columns(df)
    return concat_chains_with_separator(df, sequence_cols), antigen_ids.tolist()


# ---------------------------------------------------------------------------
# Factory — builds TransformDataset + collator
# ---------------------------------------------------------------------------


def _validate_contrastive_df(
    df: pd.DataFrame,
    antigen_dir: Path,
    sequence_cols: Sequence[str],
) -> None:
    if len(df) == 0:
        raise ValueError("Contrastive binding DataFrame is empty.")
    if ANTIGEN_KEY_FIELD not in df.columns:
        raise ValueError(
            f"Missing required column '{ANTIGEN_KEY_FIELD}'. Got: {list(df.columns)}"
        )
    missing_cols = [c for c in sequence_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing antibody sequence columns: {missing_cols}. "
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
            f"Rows with all antibody chain columns empty at indices: {idxs}. "
            f"Every row must have at least one non-empty chain."
        )
    if not antigen_dir.is_dir():
        raise FileNotFoundError(f"Antigen embedding directory not found: {antigen_dir}")


def create_contrastive_binding_dataset(
    df: pd.DataFrame,
    antigen_dir: Union[str, Path],
    nickname: str = "unnamed_contrastive_binding",
    sequence_cols: Optional[Sequence[str]] = None,
    ag_pooling_mode: AgPoolingMode = "mean",
    positive_rule: PositiveRule = "antigen_key",
) -> DatasetBundle:
    """Build a ``DatasetBundle`` for contrastive Ab-Ag training.

    Uses ``TransformDataset`` in streaming mode with the project's standard
    AA-side transforms (``TokenizeParent`` → ``ComputeLayout`` →
    ``ComputeMasks``), renames the strict pooling mask to ``ab_pool_mask``,
    and ends with ``LoadAntigenEmbedding`` which pulls frozen ESM-2 features
    from disk via an LRU cache.

    Args:
        df: Antibody DataFrame. Must contain at least one of
            ``sequence_heavy`` / ``sequence_light`` plus ``antigen_key``.
            Rows may be heavy-only, light-only, or both; NaN is treated as
            empty. ``num_segments`` is determined by which columns exist in
            the DataFrame (not by per-row content).
        antigen_dir: Directory containing ``{antigen_key}.npz`` files with
            per-residue ESM-2 embeddings under the key ``"embeddings"``.
        nickname: Human-readable dataset identifier.
        sequence_cols: Override for the antibody chain columns used in the
            per-row concatenation. If ``None``, auto-detect from
            ``df.columns`` (preferring the heavy-before-light order).
        ag_pooling_mode: How ``LoadAntigenEmbedding`` collapses the
            per-residue antigen tensor to ``[D_esm]``. ``"mean"`` (default,
            matches CALM), ``"first"`` (ESM-style CLS at position 0), or
            ``"last"``.
        positive_rule: Which in-batch pairs are positives; see
            :data:`~dasm2.config.PositiveRule`. ``"sequence"`` also requires
            a ``sequence_antigen`` column in ``df``. An unknown rule is
            rejected before any data is read.
    """
    _check_positive_rule(positive_rule)
    antigen_dir = Path(antigen_dir)
    if sequence_cols is None:
        sequence_cols = detect_ab_sequence_columns(df)
    df = df.reset_index(drop=True)
    # Fill any canonical antibody column missing from the schema with empty
    # strings so the heavy/light separator is anchored at the same position
    # in every row of the dataset (matches AntibodySegmentHandler: heavy/light
    # is structurally a 2-segment dataset; absence of one chain is per-row
    # data, not a different dataset shape).
    for col in sequence_cols:
        if col not in df.columns:
            df[col] = ""
    _validate_contrastive_df(df, antigen_dir, sequence_cols)

    sequences = concat_chains_with_separator(df, sequence_cols)
    antigen_keys = df[ANTIGEN_KEY_FIELD].astype(str).tolist()
    # Fail fast on any missing antigen file before training starts.
    for key in set(antigen_keys):
        resolve_embedding_path(antigen_dir, key)

    extra_string_fields = {ANTIGEN_KEY_FIELD: antigen_keys}
    output_keys = _CONTRASTIVE_OUTPUT_KEYS
    if positive_rule == "sequence":
        antibody_ids, antigen_ids = pair_identities(df, sequence_cols)
        extra_string_fields[ANTIBODY_SEQUENCE_FIELD] = antibody_ids
        extra_string_fields[ANTIGEN_SEQUENCE_FIELD] = antigen_ids
        output_keys = output_keys | _SEQUENCE_FIELDS

    num_segments = len(sequence_cols)
    transforms: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = [
        TokenizeParent(),
        ComputeLayout(num_segments=num_segments),
        ComputeMasks(num_segments=num_segments),
        RenameKey(src="loss_mask", dst=AB_POOL_MASK_KEY),
        LoadAntigenEmbedding(antigen_dir=antigen_dir, ag_pooling_mode=ag_pooling_mode),
    ]

    dataset = TransformDataset(
        aa_parents=sequences,
        raw_parents=sequences,
        transforms=transforms,
        output_keys=output_keys,
        extra_string_fields=extra_string_fields,
        nickname=nickname,
    )

    padding_sentinel = padding_sentinel_for(num_segments)
    collator = ContrastiveBindingCollator(
        tensor_specs={
            "aa_parents_idxs": (AA_AMBIG_IDX, "codon"),
            "layout": (padding_sentinel, "codon"),
            "mask": (False, "codon"),
            AB_POOL_MASK_KEY: (False, "codon"),
            # "scalar" scale ⇒ torch.stack → [B, D_esm]
            ANTIGEN_EMBEDDING_KEY: (0.0, "scalar"),
        },
        positive_rule=positive_rule,
    )
    return DatasetBundle(
        dataset=dataset,
        collator=collator,  # type: ignore[arg-type]
        worker_init_fn=make_worker_init_fn(),
    )


_AG_POOLING_MODE_PREFIX = "mode="
# Derived from the Literal so the runtime check stays in lockstep with the
# type alias — adding a mode in one place makes both sites pick it up.
_VALID_AG_POOLING_MODES: Tuple[str, ...] = get_args(AgPoolingMode)


def _resolve_nickname_to_paths(nickname: str) -> Tuple[Path, Path]:
    """Resolve an antigen-embeddings nickname to ``(pairs_csv, antigen_dir)``.

    The nickname manifest (``dasm2_structural_embedding_data_schema``) yields
    the antigen directory via ``data.path``. The paired CSV is read from the
    upstream ``source_manifest`` at ``products.pairs.path`` — both locations
    are explicit declared fields of their respective manifests, so no
    sibling-filename convention is needed.

    Relative paths are resolved against their own manifest file's directory.
    """
    try:
        config = get_structural_embedding_config(nickname)
    except KeyError as exc:
        raise FileNotFoundError(
            f"Contrastive-binding spec {nickname!r} is neither a "
            f"'<csv>:<antigen_dir>' pair nor a registered antigen-embeddings "
            f"nickname."
        ) from exc

    antigen_dir = config.resolve_data_path().raw_path
    if config.source_manifest is None:
        raise ValueError(
            f"Nickname {nickname!r} has no source_manifest; cannot locate the "
            f"paired CSV. Either add source_manifest to the nickname YAML, or "
            f"use a direct '<csv>:<antigen_dir>' spec."
        )
    if config.yaml_dir is None:
        raise ValueError(
            f"Nickname {nickname!r} config has no yaml_dir (registry did not "
            f"attach one); cannot resolve relative source_manifest path."
        )
    source_path = (config.yaml_dir / config.source_manifest).resolve()
    if not source_path.exists():
        raise FileNotFoundError(
            f"source_manifest for nickname {nickname!r} not found: {source_path}"
        )
    with source_path.open() as fh:
        source = yaml.safe_load(fh) or {}
    pairs = (source.get("products") or {}).get("pairs") or {}
    pairs_rel = pairs.get("path")
    if pairs_rel is None:
        raise ValueError(
            f"Source manifest {source_path} has no products.pairs.path; cannot "
            f"locate the paired CSV for nickname {nickname!r}."
        )
    csv_path = Path(pairs_rel)
    if not csv_path.is_absolute():
        csv_path = (source_path.parent / csv_path).resolve()
    return csv_path, antigen_dir


def _parse_contrastive_data_spec(
    data_path: str,
) -> Tuple[Path, Path, Optional[AgPoolingMode]]:
    """Parse a contrastive-binding data spec into ``(csv, antigen_dir, mode)``.

    Accepts three shapes:

    - ``<csv_path>:<antigen_dir>`` — direct paths, mode unspecified.
    - ``<csv_path>:<antigen_dir>:mode=<mean|first|last>`` — direct paths with
      explicit Ag pooling mode.
    - ``<nickname>[:mode=<...>]`` — registered antigen-embeddings nickname;
      the paired CSV is pulled from ``source_manifest.products.pairs.path``.

    Using an explicit ``mode=`` prefix (rather than a bare third segment)
    keeps the grammar unambiguous when a real path happens to end in the
    string ``"first"``/``"last"`` and also survives Windows-style drive
    letters on the CSV path.

    For direct paths, the antigen dir is resolved relative to the CSV's
    parent when not absolute — keeps CSVs and their embedding directory
    co-located in experiment artifact trees.

    Returns ``mode = None`` when the spec carries no ``mode=`` suffix so
    callers can fall back to their own default (e.g. a kwarg) without
    re-examining the raw string.
    """
    ag_pooling_mode: Optional[AgPoolingMode] = None
    head, sep, tail = data_path.rpartition(":")
    if sep and tail.startswith(_AG_POOLING_MODE_PREFIX):
        mode_str = tail[len(_AG_POOLING_MODE_PREFIX) :]
        if mode_str not in _VALID_AG_POOLING_MODES:
            raise ValueError(
                f"Invalid ag_pooling_mode {mode_str!r} in data spec; must be "
                f"one of {_VALID_AG_POOLING_MODES}."
            )
        ag_pooling_mode = mode_str  # type: ignore[assignment]
        remainder = head
    else:
        remainder = data_path

    if not remainder:
        raise ValueError(
            f"Contrastive-binding data spec must be "
            f"'<csv_path>:<antigen_dir>[:mode=<mean|first|last>]' or "
            f"'<nickname>[:mode=<...>]', got empty spec."
        )

    if ":" in remainder:
        # Direct ``<csv>:<antigen_dir>`` form.
        csv_path_str, antigen_dir_str = remainder.rsplit(":", 1)
        csv_path = Path(csv_path_str)
        antigen_dir = Path(antigen_dir_str)
        if not antigen_dir.is_absolute():
            antigen_dir = (csv_path.parent / antigen_dir).resolve()
    else:
        # Bare token → registry lookup.
        csv_path, antigen_dir = _resolve_nickname_to_paths(remainder)
    return csv_path, antigen_dir, ag_pooling_mode


def _read_contrastive_pairs(
    data_path: str, ag_pooling_mode: AgPoolingMode
) -> Tuple[pd.DataFrame, Path, Path, AgPoolingMode]:
    """Read the pairs CSV named by a contrastive data spec.

    Returns ``(df, csv_path, antigen_dir, ag_pooling_mode)``. A ``mode=`` in
    the spec wins over the ``ag_pooling_mode`` kwarg (so CLI callers have a
    way to set the mode without adding a dedicated flag to the training API).
    """
    csv_path, antigen_dir, spec_mode = _parse_contrastive_data_spec(data_path)
    if spec_mode is not None:
        ag_pooling_mode = spec_mode
    if not csv_path.exists():
        raise FileNotFoundError(f"Contrastive-binding CSV not found: {csv_path}")
    return pd.read_csv(csv_path), csv_path, antigen_dir, ag_pooling_mode


def require_group_aware_contrastive_cv_split(cv_split: "CvSplitConfig") -> str:
    """Return ``cv_split.group_col``, raising unless it is set.

    Antigen families must stay whole across folds, so contrastive-binding CV is
    always group-aware. Shared by the loaders and ``train_command.py``'s
    pre-flight CV validation so both agree on one wording and one place to
    change it.
    """
    if cv_split.group_col is None:
        raise ValueError(
            "ContrastiveBindingHead cv_split requires group_col (e.g. "
            "'antigen_cluster_80'): antigen families must stay whole across "
            "folds. Set it in the YAML cv_split block or pass --cv-group-col."
        )
    return cv_split.group_col


def _cluster_labels(df: pd.DataFrame, cv_split: "CvSplitConfig") -> np.ndarray:
    """Validated ``cv_split.group_col`` labels of ``df``, one string per row."""
    group_col = require_group_aware_contrastive_cv_split(cv_split)
    if group_col not in df.columns:
        raise ValueError(
            f"cv_split.group_col={group_col!r} not found in the pairs CSV. "
            f"Available: {list(df.columns)}"
        )
    if df[group_col].isna().any():
        raise ValueError(
            f"Column {group_col!r} has {int(df[group_col].isna().sum())} missing "
            f"values; drop or label those rows before splitting."
        )
    labels: np.ndarray = df[group_col].astype(str).to_numpy()
    return labels


def split_contrastive_pairs(
    df: pd.DataFrame, cv_split: "CvSplitConfig"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split pairs into group-disjoint ``(train_df, val_df)`` by ``cv_split``.

    - Without ``cv_split.val_num_folds``: sklearn ``GroupKFold`` on
      ``group_col``; fold ``fold_index`` is the validation set, as in every
      other loader.
    - With it: nested GroupKFold (see :func:`nested_group_kfold_split`); fold
      ``fold_index`` is a test set that is left out of both frames, and
      :func:`select_contrastive_test_pairs` returns it.

    Capping large groups is a data-preparation step: this function splits
    the rows it is given.
    """
    groups = _cluster_labels(df, cv_split)
    if cv_split.val_num_folds is None:
        in_train = group_kfold_in_train(groups, cv_split)
        return df[in_train].reset_index(drop=True), df[~in_train].reset_index(drop=True)
    train_idx, val_idx, _ = nested_group_kfold_split(groups, cv_split)
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
    )


def select_contrastive_test_pairs(
    df: pd.DataFrame, cv_split: "CvSplitConfig"
) -> pd.DataFrame:
    """Return the held-out test rows of ``df`` for a nested ``cv_split``.

    The complement of :func:`split_contrastive_pairs` under the same
    ``cv_split``, so these are exactly the rows training never loaded.
    """
    if cv_split.val_num_folds is None:
        raise ValueError(
            "A held-out test fold exists only when cv_split.val_num_folds is "
            "set; with a plain train/val cv_split there is no test set."
        )
    _, _, test_idx = nested_group_kfold_split(_cluster_labels(df, cv_split), cv_split)
    return df.iloc[test_idx].reset_index(drop=True)


def load_contrastive_binding_train_val(
    data_path: str,
    seed: int,
    train_fraction: float = 0.8,
    ag_pooling_mode: AgPoolingMode = "mean",
    cv_split: Optional["CvSplitConfig"] = None,
    positive_rule: PositiveRule = "antigen_key",
) -> Tuple[DatasetBundle, DatasetBundle]:
    """Load CSV + antigen dir and return ``(train_bundle, val_bundle)``.

    ``data_path`` is one of:

    - ``<csv_path>:<antigen_dir>[:mode=<mean|first|last>]`` — direct paths.
      The antigen dir is resolved relative to the CSV's parent if not
      absolute.
    - ``<nickname>[:mode=<...>]`` — a registered antigen-embeddings
      nickname (``dasm2_structural_embedding_data_schema`` manifest); the
      paired CSV is taken from the upstream
      ``source_manifest.products.pairs.path``.

    Without ``cv_split``, and when ``group_id`` is present, the split is
    group-aware (groups split wholly); otherwise a random sample-level split
    is used. With ``cv_split`` the split is a GroupKFold on
    ``cv_split.group_col``; see :func:`split_contrastive_pairs`. When
    ``cv_split.val_num_folds`` is set, the test fold is left out of both
    bundles; :func:`load_contrastive_binding_test` returns it.

    ``ag_pooling_mode`` kwarg is used when the spec omits ``mode=...``;
    when the spec carries a mode, the spec wins. ``positive_rule`` sets
    which in-batch pairs are positives (see :data:`PositiveRule`).
    """
    df, csv_path, antigen_dir, ag_pooling_mode = _read_contrastive_pairs(
        data_path, ag_pooling_mode
    )
    # Detect sequence columns once; pass through to both split bundles so
    # ``create_contrastive_binding_dataset`` doesn't re-run column detection
    # per split. Full per-split validation happens inside that factory.
    sequence_cols = detect_ab_sequence_columns(df)

    if cv_split is None:
        train_df, val_df = split_df_train_val(
            df, seed=seed, train_fraction=train_fraction
        )
    else:
        train_df, val_df = split_contrastive_pairs(df, cv_split)

    if len(train_df) == 0 or len(val_df) == 0:
        n_groups = len(set(df["group_id"])) if "group_id" in df.columns else "n/a"
        raise ValueError(
            f"Contrastive split produced empty side: "
            f"train={len(train_df)}, val={len(val_df)}. "
            f"Check train_fraction={train_fraction} vs dataset size ({len(df)} rows, "
            f"{n_groups} groups)."
        )

    train_bundle = create_contrastive_binding_dataset(
        df=train_df,
        antigen_dir=antigen_dir,
        nickname=f"{csv_path.stem}_train",
        sequence_cols=sequence_cols,
        ag_pooling_mode=ag_pooling_mode,
        positive_rule=positive_rule,
    )
    val_bundle = create_contrastive_binding_dataset(
        df=val_df,
        antigen_dir=antigen_dir,
        nickname=f"{csv_path.stem}_val",
        sequence_cols=sequence_cols,
        ag_pooling_mode=ag_pooling_mode,
        positive_rule=positive_rule,
    )
    return train_bundle, val_bundle


def load_contrastive_binding_test(
    data_path: str,
    cv_split: "CvSplitConfig",
    ag_pooling_mode: AgPoolingMode = "mean",
    positive_rule: PositiveRule = "antigen_key",
) -> Tuple[DatasetBundle, pd.DataFrame]:
    """Return the held-out test fold as ``(bundle, test_df)``.

    Applies the same split as :func:`load_contrastive_binding_train_val` for
    the same ``data_path`` and ``cv_split``, so the rows are exactly the ones
    training never saw. ``cv_split.val_num_folds`` must be set. Pass the
    ``positive_rule`` used in training so the bundle's positive matrix is
    built the same way.
    """
    df, csv_path, antigen_dir, ag_pooling_mode = _read_contrastive_pairs(
        data_path, ag_pooling_mode
    )
    sequence_cols = detect_ab_sequence_columns(df)
    test_df = select_contrastive_test_pairs(df, cv_split)
    bundle = create_contrastive_binding_dataset(
        df=test_df,
        antigen_dir=antigen_dir,
        nickname=f"{csv_path.stem}_test",
        sequence_cols=sequence_cols,
        ag_pooling_mode=ag_pooling_mode,
        positive_rule=positive_rule,
    )
    return bundle, test_df


__all__ = [
    "ANTIGEN_EMBEDDING_KEY",
    "POSITIVE_PAIR_MATRIX_KEY",
    "ANTIGEN_KEY_FIELD",
    "ANTIGEN_SEQUENCE_FIELD",
    "ANTIBODY_SEQUENCE_FIELD",
    "ANTIGEN_SEQUENCE_COLUMN",
    "AB_POOL_MASK_KEY",
    "AgPoolingMode",
    "LoadAntigenEmbedding",
    "ContrastiveBindingCollator",
    "create_contrastive_binding_dataset",
    "detect_ab_sequence_columns",
    "load_contrastive_binding_test",
    "load_contrastive_binding_train_val",
    "pair_identities",
    "positive_pair_matrix",
    "require_group_aware_contrastive_cv_split",
    "select_contrastive_test_pairs",
    "split_contrastive_pairs",
]
