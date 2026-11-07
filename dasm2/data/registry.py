"""Unified dataset registry for DASM2 experiments.

This module provides:
- YAML-based dataset configuration with automatic discovery
- Dataset nickname → path resolution
- Holdout sample ID sets for deterministic train/val splitting
- ANARCI annotation paths for structural data
- Sequential loading workflow for datasets (including multiname composition)
- Head type categorization for data loader dispatch
- Pickle/Parquet DataFrame support for precomputed neutral rates

Loading Workflow:
1. Resolve data_spec to component paths via YAML registry
2. Load each component into DataFrame (CSV, Parquet, or pickle)
3. Add train/val markers (optional)
4. Combine components into single DataFrame
5. Split into train/val subsets (optional)

High-Level API:
- load_dataframe_with_split_markers() - Get DataFrame with train/val markers
- resolve_data_spec() - Resolve dataset nickname to component paths
- split_dataframe_by_markers() - Split DataFrame into train/val subsets
"""

import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from dasm2.data.local import get_artifacts_dir as _get_artifacts_dir_from_config
from dasm2.data.local import resolve_path_with_fallback
from dasm2.data.segment_handlers import (
    detect_compatible_handler,
)
from dasm2.heads.types import HeadType
from dasm2.sequence.constants import NUM_BASES

if TYPE_CHECKING:
    from dasm2.data.segment_handlers import SegmentHandler

# Head types categorized by their data requirements.  After #621 Task #4
# the architecture-determined sets stay head-keyed (one loss per head
# shape); the loss-keyed dispatch (``EVOLUTIONARY_LOSS_TYPES``,
# ``WHICHMUT_LOSS_TYPES``) lives in ``dasm2/losses/types.py``.


STRUCTURAL_HEAD_TYPES: FrozenSet[HeadType] = frozenset(
    {
        HeadType.STRUCTURAL_GUIDANCE_HEAD,
    }
)

# MPNN Whichmut heads use MPNNWhichmutDataset (WhichmutDataset + MPNN features)
MPNN_WHICHMUT_HEAD_TYPES: FrozenSet[HeadType] = frozenset(
    {
        HeadType.WHICHMUT_RESIDUAL_HEAD,
    }
)

# DMS heads use DMSDataset (with optional MPNN wrapping)
DMS_HEAD_TYPES: FrozenSet[HeadType] = frozenset(
    {
        HeadType.DMS_FUNCTIONAL_SCORE_HEAD,
    }
)

# Delta-affinity / pooled-prediction heads share the supervised-regression
# data loader (no neutral model needed). Each head type dispatches its target
# batch key and dtype via _pooled_target_spec() in train_command.py.
DELTA_AFFINITY_HEAD_TYPES: FrozenSet[HeadType] = frozenset(
    {
        HeadType.DELTA_AFFINITY_HEAD,
        HeadType.POOLED_REGRESSION_HEAD,
        HeadType.POOLED_MULTI_LABEL_HEAD,
        HeadType.POOLED_CLASSIFICATION_HEAD,
    }
)

# Contrastive binding heads use ContrastiveBindingDataset (Ab encoder + frozen
# pre-computed antigen embeddings; no neutral model needed)
CONTRASTIVE_BINDING_HEAD_TYPES: FrozenSet[HeadType] = frozenset(
    {
        HeadType.CONTRASTIVE_BINDING_HEAD,
    }
)

# Binding-classification heads use the pairwise-binding dataset (entity-A
# encoder output + full per-residue frozen entity-B embeddings for
# cross-attention; no neutral model needed)
BINDING_CLASSIFICATION_HEAD_TYPES: FrozenSet[HeadType] = frozenset(
    {
        HeadType.BINDING_CLASSIFICATION_HEAD,
    }
)

# MLM heads use MLMDataset (flat list of observed productive sequences; no
# parent/child pairing, no branch lengths, no neutral model needed)
MLM_HEAD_TYPES: FrozenSet[HeadType] = frozenset(
    {
        HeadType.MLM_HEAD,
    }
)


def get_artifacts_dir() -> Path:
    """Get configured artifacts directory.

    Returns:
        Path to artifacts directory from config file.

    Raises:
        ValueError: If artifacts_dir not configured
    """
    artifacts_dir = _get_artifacts_dir_from_config()
    if artifacts_dir is None:
        raise ValueError(
            "artifacts_dir not configured. Add it to ~/.config/dasm2/config.yaml:\n"
            "  artifacts_dir: ~/dasm2-artifacts"
        )
    return artifacts_dir


def resolve_data_path(data_spec: str) -> Path:
    """Resolve a dataset specification to a file path.

    Upstream this fell back to a manifest index when the spec was a
    registered nickname rather than a path. This repository names every
    input explicitly in ``config.yaml`` -- that is the whole bargain it
    offers a reader who brings their own GISAID access -- so there is no
    manifest to consult and the fallback is not vendored. A spec that is
    not a path that exists is an error here, not a nickname to look up.

    Args:
        data_spec: Path to a dataset file.

    Returns:
        Resolved Path to the dataset file.

    Raises:
        ValueError: If data_spec is empty or whitespace-only.
        FileNotFoundError: If the path does not exist.
    """
    return resolve_path_with_fallback(
        spec=data_spec,
        registry_lookup=lambda _spec: None,
        item_type="Dataset",
        list_available=list,
    )












def _generate_nickname_from_path(file_path: str) -> str:
    """Generate consistent hash-based nickname from filepath.

    Uses MD5 hash of the full, expanded, resolved path to create a consistent
    identifier that remains stable across Python sessions.

    Args:
        file_path: Path to the dataset file

    Returns:
        8-character hash prefix as nickname
    """
    resolved_path = Path(file_path).expanduser().resolve()
    path_hash = hashlib.md5(str(resolved_path).encode()).hexdigest()
    return f"path_{path_hash[:8]}"


def _determine_nickname_from_spec_and_path(data_spec: str, csv_path: Path) -> str:
    """Derive a nickname for a dataset from the path it resolved to.

    Upstream this returned ``data_spec`` unchanged when it named a registered
    dataset, and fell back to a path hash otherwise. No dataset is registered
    here, so only the fallback remains -- every corpus gets a ``path_<hash>``
    nickname, which is what the seam stamp and the log lines already carry.

    Args:
        data_spec: Original specification. Unused; kept so the call sites read
            the same as upstream's and a future registry would slot back in.
        csv_path: Already-resolved path corresponding to data_spec.

    Returns:
        Hash-based dataset nickname.
    """
    del data_spec
    return _generate_nickname_from_path(str(csv_path))


def determine_nickname_from_data_spec(data_spec: str) -> str:
    """Determine dataset nickname from a data specification.

    This is the single source of truth for interpreting data_spec and
    determining the appropriate nickname. Handles:
    - Multinames (e.g., "v1jaffe+v1tang") - resolves each component, rejoins with +
    - Registered nicknames - uses the nickname directly
    - File paths - generates hash-based nickname

    Path-based multinames (e.g., "/abs/path1.parquet+/abs/path2.parquet")
    need per-component resolution: returning the raw multiname embeds the
    component paths' ``/`` characters into downstream filenames (checkpoint
    names, BL artifact cache names), which the OS interprets as directory
    separators, scattering artifacts across a deeply nested tree the
    Snakefile's flat ``ls *.zip`` glob cannot find. Hashing each component
    yields a filename-safe identifier while preserving the multiname
    structure for the join (``path_a3b2c1d4+path_e5f6g7h8``).

    Args:
        data_spec: File path, registered nickname, or multiname

    Returns:
        Dataset nickname
    """
    if "+" in data_spec:
        return "+".join(nick for nick, _ in resolve_component_specs(data_spec))
    csv_path = resolve_data_path(data_spec)
    return _determine_nickname_from_spec_and_path(data_spec, csv_path)


def parse_multiname(multiname: str) -> List[str]:
    """Parse a multiname string into component dataset nicknames.

    Args:
        multiname: Dataset specification, e.g., "v1jaffe+v1tang"
                  Must not contain spaces around '+' separator.

    Returns:
        List of dataset nicknames

    Raises:
        ValueError: If multiname is empty, contains empty components,
                   or has spaces around the '+' separator
    """
    if not multiname:
        raise ValueError("Multiname cannot be empty")

    components = multiname.split("+")

    for component in components:
        if component != component.strip():
            raise ValueError(
                f"Invalid multiname '{multiname}': spaces around '+' separator are not allowed. "
                f"Use 'component1+component2' format without spaces."
            )

    if any(not c for c in components):
        raise ValueError(f"Invalid multiname '{multiname}': empty component")

    return components


def resolve_component_specs(data_spec: str) -> List[Tuple[str, Path]]:
    """Parse data_spec into component specifications.

    Args:
        data_spec: File path, registered nickname, or multiname (e.g., "v1jaffe+v1tang")

    Returns:
        List of (component_nickname, csv_path) tuples, one per component

    Raises:
        FileNotFoundError: If any component path doesn't exist
        ValueError: If data_spec is empty or has empty components
    """
    component_specs = parse_multiname(data_spec)
    result = []
    for component_spec in component_specs:
        csv_path = resolve_data_path(component_spec)
        nickname = _determine_nickname_from_spec_and_path(component_spec, csv_path)
        result.append((nickname, csv_path))
    return result


_PICKLE_EXTENSIONS = (".pkl", ".pickle")
_PARQUET_EXTENSIONS = (".parquet",)


_INDEX_ARTIFACT_COLUMN = re.compile(r"Unnamed: \d+(\.\d+)*")
"""What a pandas index saved into a CSV reads back as: ``Unnamed: 0``, with
``.1``/``.2`` suffixes when a frame carrying one was saved with its index
again (as in ``tstPaired.csv.gz``)."""


def drop_index_artifact_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the ``Unnamed: N`` columns a saved pandas index reads back as.

    PCP CSVs are written with their index, so every one read without
    ``index_col`` gains such a column, and a Parquet file built from that
    frame keeps it. They are never meant as data.
    """
    artifacts = [
        column
        for column in df.columns
        if isinstance(column, str) and _INDEX_ARTIFACT_COLUMN.fullmatch(column)
    ]
    return df.drop(columns=artifacts)


def is_parquet_path(path: Path) -> bool:
    """Whether ``path`` names a Parquet file, by its (case-insensitive) suffix."""
    return path.name.lower().endswith(_PARQUET_EXTENSIONS)


def arrow_to_dataframe(data: Union[pa.Table, pa.RecordBatch]) -> pd.DataFrame:
    """Convert Arrow data read from a Parquet file to a ``RangeIndex`` frame.

    A pandas index stored in the file comes back as the index. An unnamed
    level is the same row-number artifact as a CSV's ``Unnamed: 0`` and is
    dropped; a named level is data and becomes a column. ``Unnamed: N``
    columns are dropped too (:func:`drop_index_artifact_columns`).
    """
    df = data.to_pandas()
    unnamed_levels = [i for i, name in enumerate(df.index.names) if name is None]
    if len(unnamed_levels) < df.index.nlevels:
        # Drop only the unnamed levels; resetting them would add
        # ``level_N`` columns, which drop_index_artifact_columns does not catch.
        if unnamed_levels:
            df = df.reset_index(level=unnamed_levels, drop=True)
        df = df.reset_index()
    else:
        df = df.reset_index(drop=True)
    return drop_index_artifact_columns(df)


def read_parquet_dataframe(
    path: Path, columns: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Read one whole Parquet file into a DataFrame (:func:`arrow_to_dataframe`).

    Args:
        path: A single ``.parquet`` file.
        columns: Read only these columns; all of them if None. A pandas
            index stored in the file is then not read, so a named index
            level is dropped rather than becoming a column.

    Raises:
        ValueError: If ``path`` is a directory. A directory of Parquet files
            is read only by the training path's ``ParquetBackend``.
    """
    if path.is_dir():
        raise ValueError(
            f"{path} is a directory. Only a single .parquet file can be read "
            "into a DataFrame; a directory of them is read only by the "
            "training path."
        )
    return arrow_to_dataframe(pq.read_table(path, columns=columns))


def load_dataframe(file_path: Path) -> pd.DataFrame:
    """Load DataFrame from CSV, Parquet or pickle based on file extension.

    Supports:
    - CSV: .csv, .csv.gz - Standard parent-child sequence DataFrames
    - Parquet: .parquet - A single file, e.g. pcp-pipeline's PCP output
    - Pickle: .pkl, .pickle, .pkl.gz, .pickle.gz - DataFrames optionally containing
      precomputed nt_rates/nt_csps tensor columns from neutral model evaluation

    CSV and Parquet frames lose their ``Unnamed: N`` index columns
    (:func:`drop_index_artifact_columns`).

    Args:
        file_path: Path to data file

    Returns:
        Loaded DataFrame

    Raises:
        ValueError: If file format is unsupported, or a Parquet path is a
            directory
        TypeError: If pickle file doesn't contain a DataFrame
    """
    name_lower = file_path.name.lower()

    if name_lower.endswith((".pkl", ".pickle", ".pkl.gz", ".pickle.gz")):
        result = pd.read_pickle(file_path)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"Pickle file {file_path} contains {type(result).__name__}, "
                f"expected pandas DataFrame"
            )
        return result

    if name_lower.endswith((".csv", ".csv.gz")):
        return drop_index_artifact_columns(pd.read_csv(file_path))

    if is_parquet_path(file_path):
        return read_parquet_dataframe(file_path)

    raise ValueError(
        f"Unsupported file format: {file_path.name}. "
        f"Supported formats: .csv, .csv.gz, .parquet, .pkl, .pickle, .pkl.gz, "
        f".pickle.gz"
    )


def is_parquet_data_spec(data_spec: str) -> bool:
    """Check if data_spec resolves to Parquet file(s) or a directory of them.

    For multinames, all components must be the same format.
    Raises ValueError if formats are mixed.

    Args:
        data_spec: File path, registered nickname, or multiname

    Returns:
        True if data resolves to Parquet file(s), False otherwise

    Raises:
        ValueError: If formats are mixed in a multiname
    """
    component_specs = parse_multiname(data_spec)
    component_is_parquet_flags = []

    for component in component_specs:
        resolved_path = resolve_data_path(component)
        if resolved_path.is_dir():
            has_parquet = any(resolved_path.glob("*.parquet"))
            if not has_parquet:
                raise ValueError(
                    f"Directory '{resolved_path}' (from component '{component}') "
                    f"contains no .parquet files"
                )
            component_is_parquet_flags.append(True)
        else:
            component_is_parquet_flags.append(is_parquet_path(resolved_path))

    if len(set(component_is_parquet_flags)) > 1:
        raise ValueError(
            f"Mixed file formats in multiname '{data_spec}' not supported. "
            f"All components must be the same format."
        )

    return component_is_parquet_flags[0] if component_is_parquet_flags else False


def is_pickle_data_spec(data_spec: str) -> bool:
    """Check if data_spec resolves to pickle file(s).

    For multinames, all components must be the same format.
    Raises ValueError if formats are mixed.

    Args:
        data_spec: File path, registered nickname, or multiname

    Returns:
        True if data resolves to pickle file(s), False if CSV

    Raises:
        ValueError: If formats are mixed in a multiname
    """
    component_specs = parse_multiname(data_spec)
    component_is_pickle_flags = []

    for component in component_specs:
        resolved_path = resolve_data_path(component)
        name_lower = resolved_path.name.lower()
        is_pickle = any(
            name_lower.endswith(ext) or name_lower.endswith(f"{ext}.gz")
            for ext in _PICKLE_EXTENSIONS
        )
        component_is_pickle_flags.append(is_pickle)

    if len(set(component_is_pickle_flags)) > 1:
        raise ValueError(
            f"Mixed file formats in multiname '{data_spec}' not supported. "
            f"All components must be either CSV or pickle."
        )

    return component_is_pickle_flags[0] if component_is_pickle_flags else False




def _apply_segment_specific_preprocessing(
    df: pd.DataFrame,
    segment_handler: "SegmentHandler",
) -> pd.DataFrame:
    """Apply segment-specific preprocessing to loaded DataFrame.

    For antibody data, adds v_family columns extracted from v_gene columns
    and validates v_gene families.

    Args:
        df: Loaded DataFrame
        segment_handler: Detected segment handler

    Returns:
        DataFrame with segment-specific preprocessing applied
    """
    from dasm2.data.segment_handlers import AntibodySegmentHandler

    if isinstance(segment_handler, AntibodySegmentHandler):
        df = segment_handler.add_v_family_columns(df)
        segment_handler.validate_v_gene_families(df)

    return df


ROW_PREPROCESSING_VERSION = "1"
"""Identifies the *behaviour* of :func:`preprocess_rows_for_split`.

Bump this whenever the seam starts or stops dropping a row it used to keep
(a new filtering step, a changed predicate, a removed one). Parquet files
stamped under an older version then fail
:func:`verify_parquet_seam_stamp` instead of silently splitting a row set
the CSV/MLM paths no longer produce.

``tests/test_data_registry.py`` pins the seam's *behaviour* -- that it drops
naughty-list rows and nothing else -- so a change that moves the row set fails
loudly there with a pointer back to this constant, rather than being forgotten
quietly.
"""




def seam_identity(nickname: Optional[str]) -> Dict[str, str]:
    """Identify the seam as it applies to one source dataset.

    Two row sets preprocessed under the same identity had the same rows
    dropped, so ``GroupKFold`` packs them into the same folds. Stamping this
    into a Parquet file at write time and recomputing it at load time is what
    lets the Parquet CV path (which splits rows *as stored*, never calling
    :func:`preprocess_rows_for_split`) prove it is splitting the row set the
    CSV and MLM paths split -- see issue #1131.

    Args:
        nickname: Registered dataset nickname whose manifest drives the seam,
            or ``None`` for data that never passes through a manifest (the
            viral PCP corpora), whose naughty list is vacuously empty.

    Returns:
        ``{"row_preprocessing_version": ..., "row_preprocessing_inputs": ...}``,
        the latter a sha256 over the sorted naughty list.

    Raises:
        ValueError: If ``nickname`` is not ``None`` and names no registered
            dataset. Deliberately *not*
            ``_get_naughty_clonal_families``'s "unknown nickname -> empty
            list": a stamp naming a since-renamed dataset would then recompute
            as "no naughty families" and match any file written while the list
            was empty (Constitution IV). ``None`` is the explicit way to ask
            for the vacuous identity.
    """
    if nickname is not None:
        raise ValueError(
            f"Cannot compute the row-preprocessing seam identity for "
            f"'{nickname}': this repository has no manifest registry to look "
            f"a nickname up in, so None -- the vacuous identity -- is the only "
            f"answer it can give. See registered_nickname_or_none."
        )
    naughty: List[Tuple[str, str]] = []

    payload = json.dumps(sorted([sample, family] for sample, family in naughty))
    return {
        "row_preprocessing_version": ROW_PREPROCESSING_VERSION,
        "row_preprocessing_inputs": hashlib.sha256(payload.encode()).hexdigest(),
    }


def registered_nickname_or_none(nickname: Optional[str]) -> Optional[str]:
    """Always ``None``: no dataset in this repository is manifest-backed.

    Upstream this asked a YAML registry whether a manifest drove the
    nickname, because :func:`seam_identity` refuses an unregistered one by
    design and the write and load sides had to agree on which specs were
    manifest-backed. Here the answer is the same for every spec -- the
    corpora are named by path in ``config.yaml`` and no manifest exists --
    so the function is kept as the one place that records the decision,
    rather than being inlined at its two call sites where the next reader
    would have to rediscover it.

    Keeping it also keeps the stamp stable: ``seam_identity(None)`` hashes
    an empty naughty list, which is what every Parquet file this pipeline
    reads was stamped with.
    """
    return None


def _seam_problem_lines(
    problems: Sequence[Tuple[Optional[Path], str]],
) -> List[str]:
    """One bullet per *distinct* problem, naming the files it was found in.

    A directory-backed component is written by one writer run, so when its
    stamp is wrong every shard is usually wrong the same way. Emitting the
    problem once per shard buries the problems that differ under hundreds of
    identical lines, and none of them says which file to look at. Grouping by
    problem text keeps each failure mode to one line; the paths are carried
    because the offender is often only a subset of the component (one shard
    appended by a later, unstamped run).

    Problems whose path is ``None`` are component-level -- no single file owns
    them -- and get a bare bullet.
    """
    paths_by_problem: Dict[str, List[Path]] = {}
    for path, problem in problems:
        paths = paths_by_problem.setdefault(problem, [])
        if path is not None:
            paths.append(path)

    lines = []
    for problem, paths in paths_by_problem.items():
        if not paths:
            lines.append(f"  - {problem}")
        elif len(paths) == 1:
            lines.append(f"  - {paths[0]}: {problem}")
        else:
            others = len(paths) - 1
            plural = "file" if others == 1 else "files"
            lines.append(f"  - {paths[0]} and {others} more {plural}: {problem}")
    return lines


def verify_parquet_seam_stamp(
    component_spec: str,
    file_metadata: Sequence[Tuple[Path, Mapping[str, str]]],
    actual_num_rows: int,
    *,
    component_nickname: Optional[str],
    strict: bool,
) -> None:
    """Check a Parquet component's seam stamp against the current registry.

    Catches a file built before a row-dropping step entered the seam (version
    mismatch), a naughty list edited after the file was built (input-hash
    mismatch), a file no stamping writer produced (no stamp), a file resliced
    out of a stamped one -- Arrow copies schema metadata through
    ``pq.read_table``/``pa.concat_tables``, so the derivative carries a
    valid-looking stamp and only ``num_rows`` gives it away -- and a file
    head-truncated by ``write_parquet.py --max-rows``.

    It does *not* catch a writer that stamps without filtering;
    ``prepare_viral_parquet.py`` stamps an assertion about rows the seam never
    touched. The stamp therefore certifies "the writer asserted seam
    compatibility", not "these rows went through the seam".

    Args:
        component_spec: The data-spec component being verified, for messages.
        file_metadata: One ``(path, decoded schema metadata)`` pair per
            Parquet file in the component. Every file is checked: the stamp is
            written per file, so a directory-backed component is only as
            verified as its least-checked shard. The paths are what the
            failure message points the reader at, so they must be the ones a
            reader can open, not display labels.
        actual_num_rows: Rows actually present, summed over the same files.
        component_nickname: The registered nickname this component is being
            loaded *as*, or ``None`` for a spec no manifest backs (pass
            :func:`registered_nickname_or_none`'s result). Checked against
            each file's self-declared ``source_dataset``: a file can carry a
            perfectly self-consistent stamp and still be the wrong row set for
            the manifest it is loaded under. Required rather than defaulted so
            a new call site cannot silently drop the cross-check.
        strict: Raise on any problem rather than warn. True when a
            ``cv_split`` is in play, where per-group row counts *are* the
            split; False on the ``in_train``/holdout path, where they are not.

    Raises:
        ValueError: If ``strict`` and the stamp is absent or disagrees.
    """
    problems: List[Tuple[Optional[Path], str]] = []
    stamped_num_rows = 0
    rows_comparable = True
    stamped_sources: Set[str] = set()
    subsampled_from: Optional[str] = None

    # The identity the *loaded* spec implies, as against the one each file
    # self-declares. Without this, a file stamped vacuously -- built from a
    # bare CSV path before that path was registered -- verifies against its
    # own empty naughty list forever, even after the manifest it is now loaded
    # under grows a real one (PR #1139 review).
    loaded_identity = (
        seam_identity(component_nickname) if component_nickname is not None else None
    )

    for path, meta in file_metadata:
        if "row_preprocessing_version" not in meta:
            problems.append(
                (
                    path,
                    "no seam stamp (written by a writer that predates issue "
                    "#1131, or by no known writer)",
                )
            )
            rows_comparable = False
            continue

        source_dataset = meta.get("source_dataset")
        if source_dataset is None:
            problems.append((path, "stamp is missing 'source_dataset'"))
            rows_comparable = False
            continue

        if source_dataset:
            stamped_sources.add(source_dataset)

        try:
            expected = seam_identity(source_dataset or None)
        except ValueError:
            problems.append(
                (
                    path,
                    f"stamped source_dataset '{source_dataset}' is no longer a "
                    f"registered nickname, so the seam identity it names "
                    f"cannot be recomputed",
                )
            )
            rows_comparable = False
            continue

        for key in ("row_preprocessing_version", "row_preprocessing_inputs"):
            # Presence is checked before indexing, the same way ``num_rows``
            # is below. ``row_preprocessing_version`` is already known present
            # (the no-stamp branch above returns on it), but
            # ``row_preprocessing_inputs`` is not: a hand-edited or
            # partially-copied stamp can carry one key without the other, and
            # indexing the missing one raised ``KeyError`` in both strict and
            # warn modes instead of the ``ValueError`` with rebuild
            # instructions this function promises (PR #1139 review).
            if key not in meta:
                problems.append((path, f"stamp is missing '{key}'"))
            elif meta[key] != expected[key]:
                problems.append(
                    (
                        path,
                        f"{key} is '{meta[key]}' but '{source_dataset}' "
                        f"currently yields '{expected[key]}'",
                    )
                )

        if (
            loaded_identity is not None
            and source_dataset != component_nickname
            and meta.get("row_preprocessing_inputs")
            != loaded_identity["row_preprocessing_inputs"]
        ):
            problems.append(
                (
                    path,
                    f"stamp is self-consistent for source_dataset "
                    f"'{source_dataset or '(none)'}', but this file is being "
                    f"loaded as '{component_nickname}', whose naughty list "
                    f"currently yields "
                    f"'{loaded_identity['row_preprocessing_inputs']}' -- the "
                    f"rows that manifest drops are still in this file",
                )
            )

        if "subsampled_from" in meta:
            subsampled_from = meta["subsampled_from"]
            problems.append(
                (
                    path,
                    f"subsampled from '{subsampled_from}' by "
                    f"scripts/subsample_parquet.py, so it holds a random "
                    f"subset of that file's rows rather than the row set the "
                    f"CSV path splits",
                )
            )
            rows_comparable = False

        if "max_rows" in meta:
            problems.append(
                (
                    path,
                    f"written with --max-rows {meta['max_rows']}, so it holds "
                    f"a head-truncated prefix of '{source_dataset}' rather "
                    f"than the row set the CSV path splits",
                )
            )

        if "num_rows" not in meta:
            # A subsample deliberately carries no stamped count (the writer
            # strips it, since the original's no longer applies). Its real
            # problem is already reported above; a second line about the
            # missing count would just point away from it.
            if "subsampled_from" not in meta:
                problems.append((path, "stamp is missing 'num_rows'"))
            rows_comparable = False
        else:
            stamped_num_rows += int(meta["num_rows"])

    if rows_comparable and stamped_num_rows != actual_num_rows:
        # Component-level: the counts are summed over every file, so no single
        # path owns the discrepancy.
        problems.append(
            (
                None,
                f"holds {actual_num_rows} rows but was stamped at "
                f"{stamped_num_rows}, so rows were added or removed after the "
                f"writer ran (a resliced derivative carries the original "
                f"stamp)",
            )
        )

    if not problems:
        return

    if subsampled_from is not None:
        # "Rebuild with write_parquet.py" is actively wrong advice here: it
        # would produce the full dataset, not this subsample (PR #1139
        # review). A subsample has no correct rebuild -- it is the wrong input
        # for a row-count-driven split, whatever it was built from.
        remedy = (
            f"\nA subsample holds a random subset of the rows in "
            f"'{subsampled_from}', so it cannot realize that file's fold "
            f"map. Point the cv_split at the source Parquet and subsample "
            f"after the split, or load this file without a cv_split."
        )
    else:
        # Name the stamped source dataset in the remedy when the files agree
        # on one; otherwise leave the placeholder. `component_spec` is
        # deliberately not spliced in as the output path -- it is usually a
        # nickname, not the .parquet path write_parquet.py takes as its third
        # positional.
        dataset_arg = (
            next(iter(stamped_sources))
            if len(stamped_sources) == 1
            else "<source-dataset>"
        )
        remedy = (
            f"\nIts rows may therefore differ from the ones the CSV and MLM "
            f"paths split, which changes the realized fold map. Rebuild it "
            f"with scripts/write_parquet.py, writing over the .parquet path "
            f"'{component_spec}' resolves to:\n"
            f"  pixi run python scripts/write_parquet.py "
            f"{dataset_arg} <neutral-model> <output.parquet> --force\n"
            f"(scripts/prepare_viral_parquet.py for the viral PCP corpora, "
            f"scripts/prepare_compara_parquet.py for compara, "
            f"scripts/prepare_superfamily_parquet.py for superfamily PCPs)."
        )
    message = (
        f"Parquet component '{component_spec}' does not carry a seam stamp "
        f"matching the current registry:\n"
        + "\n".join(_seam_problem_lines(problems))
        + remedy
    )
    if strict:
        raise ValueError(message)
    warnings.warn(message, stacklevel=2)


def _preprocess_component(
    df: pd.DataFrame,
    nickname: str,
    segment_handler: "SegmentHandler",
) -> pd.DataFrame:
    """Apply all preprocessing to a loaded DataFrame.

    Upstream this also ran a row-dropping pass, ``preprocess_rows_for_split``,
    whose only step was dropping rows belonging to a manifest's naughty clonal
    families. That list came from a YAML registry this fork does not have, and
    the concept is a BCR one with no viral counterpart, so the pass is gone and
    the seam it defined is vacuous -- which is exactly what ``seam_identity``
    stamps and ``verify_parquet_seam_stamp`` checks.

    Args:
        df: Loaded DataFrame.
        nickname: Dataset nickname. Unused now the row-dropping pass is gone;
            kept so the seam's two sides keep naming the same thing.
        segment_handler: Detected segment handler.

    Returns:
        Preprocessed DataFrame.
    """
    del nickname
    return _apply_segment_specific_preprocessing(df, segment_handler)


def _load_component_dataframes(
    component_specs: List[Tuple[str, Path]],
) -> List[Tuple[str, pd.DataFrame, "SegmentHandler"]]:
    """Load each component specification into a DataFrame with preprocessing.

    Args:
        component_specs: List of (nickname, file_path) tuples

    Returns:
        List of (nickname, dataframe, segment_handler) tuples
    """
    components = []
    for nickname, file_path in component_specs:
        print(f"Loading {file_path}")
        df = load_dataframe(file_path)

        segment_handler = detect_compatible_handler(
            set(df.columns), ("parent", "child")
        )
        df = _preprocess_component(df, nickname, segment_handler)

        components.append((nickname, df, segment_handler))
    return components


def _add_split_markers_to_components(
    components: List[Tuple[str, pd.DataFrame]],
) -> List[Tuple[str, pd.DataFrame]]:
    """Add train/val markers to each component DataFrame.

    Args:
        components: List of (nickname, dataframe) tuples

    Returns:
        List of (nickname, dataframe_with_markers) tuples
    """
    marked_components = []
    for nickname, df in components:
        marked_df = add_train_val_markers(df, nickname)
        marked_components.append((nickname, marked_df))
    return marked_components


def _combine_component_dataframes(
    components: List[Tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Combine component DataFrames into a single DataFrame.

    Args:
        components: List of (nickname, dataframe) tuples

    Returns:
        Combined DataFrame

    Raises:
        ValueError: If components list is empty
    """
    if not components:
        raise ValueError(
            "No datasets available to combine. "
            "Check that all dataset nicknames in the multiname specification are valid."
        )

    dfs = [df for _, df in components]
    return pd.concat(dfs, ignore_index=True)


def validate_dataset_nickname_for_filesystem_safety(dataset_nickname: str) -> None:
    """Validate that dataset nickname is safe for use in filenames and paths.

    Args:
        dataset_nickname: The dataset nickname to validate

    Raises:
        ValueError: If the dataset nickname contains problematic characters
    """
    problematic_chars = ["/", "\\", ":", "*", "?", "<", ">", "|", '"', "'"]
    for char in problematic_chars:
        if char in dataset_nickname:
            raise ValueError(
                f"Dataset nickname '{dataset_nickname}' contains problematic character '{char}'. "
                f"Dataset nicknames should be simple identifiers without path separators or special characters."
            )


def add_train_val_markers(
    sequences_df: pd.DataFrame,
    dataset_nickname: str,
) -> pd.DataFrame:
    """Add in_train column to sequences dataframe based on holdout strategy.

    If the DataFrame already has an 'in_train' column, it is used as-is and
    no holdout lookup is performed. This allows loading pre-split datasets
    (e.g., viral data with random splits) without requiring registry entries.

    Args:
        sequences_df: DataFrame with at minimum parent_sequence and child_sequence.
                     Must include sample_id column for holdout-based splitting
                     unless in_train column is already present.
        dataset_nickname: Dataset identifier to look up holdouts

    Returns:
        DataFrame with in_train column (True for training, False for validation)

    Raises:
        ValueError: If dataset has no holdouts defined or sample_id column missing
                    (only when in_train column is not already present)
    """
    sequences_df = sequences_df.copy()

    # If in_train column already exists, use it as-is (for pre-split datasets)
    if "in_train" in sequences_df.columns:
        num_train = sequences_df["in_train"].sum()
        num_val = (~sequences_df["in_train"]).sum()
        print(f"Using pre-existing train/val split: {num_train} train, {num_val} val")
        return sequences_df

    # Everything past here derived the split from a YAML registry's
    # `holdout_samples`. This fork has no registry, and a split that lived
    # outside the corpus would not travel with it to another reader, so a
    # frame that arrives without `in_train` is an error rather than something
    # to reconstruct.
    raise ValueError(
        f"Dataset '{dataset_nickname}' has no 'in_train' column. The split is "
        f"a property of the corpus and is written by the corpus-building "
        f"pipeline; it cannot be derived here."
    )


def load_dataframe_with_split_markers(
    data_spec: str,
) -> pd.DataFrame:
    """Load dataset into DataFrame with in_train column marking train/val split.

    Supports multiname format (e.g., "v1jaffe+v1tang") for combining datasets.

    Args:
        data_spec: File path, registered nickname, or multiname (e.g., "v1jaffe+v1tang")

    Returns:
        DataFrame with in_train column marking train/val split

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If dataset has no holdouts defined or lacks sample_id column
    """
    component_specs = resolve_component_specs(data_spec)
    loaded_components = _load_component_dataframes(component_specs)
    components = [(nickname, df) for nickname, df, _ in loaded_components]
    marked_components = _add_split_markers_to_components(components)
    return _combine_component_dataframes(marked_components)


def split_dataframe(
    sequences_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train and validation subsets by in_train column.

    Args:
        sequences_df: DataFrame with in_train column

    Returns:
        Tuple of (train_df, val_df)

    Raises:
        ValueError: If in_train column not present
    """
    if "in_train" not in sequences_df.columns:
        raise ValueError(
            "DataFrame must have 'in_train' column. "
            "Use load_dataframe_with_split_markers() or add_train_val_markers() first."
        )

    train_df = sequences_df[sequences_df["in_train"]].copy()
    val_df = sequences_df[~sequences_df["in_train"]].copy()

    return train_df, val_df


def has_precomputed_neutral_columns(df: pd.DataFrame) -> bool:
    """Check if DataFrame has precomputed neutral model output columns.

    Args:
        df: DataFrame to check for precomputed columns

    Returns:
        True if precomputed neutral outputs exist, False otherwise
    """
    return any(col.startswith(("nt_rates", "nt_csps")) for col in df.columns)


def validate_precomputed_columns(df: pd.DataFrame) -> None:
    """Validate precomputed neutral columns contain properly formatted tensors.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If precomputed columns are missing or empty
        TypeError: If columns contain non-tensor data
        ValueError: If tensors have incorrect shape
    """
    if not has_precomputed_neutral_columns(df):
        raise ValueError(
            "DataFrame lacks precomputed neutral outputs (nt_rates/nt_csps "
            "columns), which this fork's Parquet corpora always carry."
        )

    for col in df.columns:
        if not col.startswith(("nt_rates", "nt_csps")):
            continue

        non_null = df[col].dropna()
        if len(non_null) == 0:
            raise ValueError(f"Column {col} contains only null values")

        sample = non_null.iloc[0]
        if not isinstance(sample, torch.Tensor):
            raise TypeError(
                f"Column {col} contains {type(sample).__name__}, expected torch.Tensor"
            )

        is_rates = col.startswith("nt_rates")
        if is_rates:
            if sample.dim() != 1:
                raise ValueError(
                    f"Column {col} has shape {list(sample.shape)}, "
                    f"expected [L] (1D) for nucleotide rates"
                )
        else:
            if sample.dim() != 2 or sample.shape[1] != NUM_BASES:
                raise ValueError(
                    f"Column {col} has shape {list(sample.shape)}, "
                    f"expected [L, {NUM_BASES}] for nucleotide CSPs"
                )

    print("Using precomputed neutral model outputs from DataFrame")


def sequence_df_from_spec(data_spec: str) -> pd.DataFrame:
    """Load sequences DataFrame from data specification without split markers.

    Args:
        data_spec: File path, registered nickname, or multiname

    Returns:
        Combined DataFrame without train/val markers
    """
    sequences_df, _ = sequence_df_and_handler_from_spec(data_spec)
    return sequences_df


def sequence_df_and_handler_from_spec(
    data_spec: str,
) -> Tuple[pd.DataFrame, "SegmentHandler"]:
    """Load sequences DataFrame and detected segment handler from data spec.

    Like ``sequence_df_from_spec`` but also returns the segment handler
    detected during loading, avoiding redundant re-detection.

    Args:
        data_spec: File path, registered nickname, or multiname.

    Returns:
        Tuple of (sequences_df, segment_handler).
    """
    component_specs = resolve_component_specs(data_spec)
    loaded_components = _load_component_dataframes(component_specs)
    components = [(nickname, df) for nickname, df, _ in loaded_components]
    sequences_df = _combine_component_dataframes(components)
    # Use handler from last component (all components share the same format)
    _, _, handler = loaded_components[-1]
    return sequences_df, handler


def preprocess_sequences_dataframe_columns(
    sequences_df: pd.DataFrame,
    segment_handler: "SegmentHandler",
) -> pd.DataFrame:
    """Preprocess sequences DataFrame to add derived columns.

    Currently adds v_family columns by extracting from v_gene columns.

    Args:
        sequences_df: DataFrame with sequence data.
        segment_handler: Handler for segment-specific column resolution.

    Returns:
        DataFrame with added derived columns.
    """
    for newname, colname in zip(
        segment_handler.get_column_names("v_family"),
        segment_handler.get_column_names("v_gene"),
    ):
        if colname in sequences_df.columns:
            sequences_df[newname] = sequences_df[colname].str.split("-").str[0]
    return sequences_df
