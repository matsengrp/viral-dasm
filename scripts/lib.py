"""config.yaml, read once, with the invariants checked at import.

Every path and every pinned number the pipeline uses comes through here.
Import-time rather than rule-time validation: the alternative is discovering a
missing corpus or a contradictory split partway into a multi-day DAG.

Nothing in this module resolves anything. If a path is not in config.yaml it
does not exist, which is what makes config.yaml a complete description of the
pipeline's inputs for a reader who has their own mat-pcp run.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

#: Repository root -- the directory holding config.yaml.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Put this repository ahead of anything installed, so `import dasm2` is the
# vendored fork sitting next to this file.
#
# Not belt-and-braces. The venv carries an editable install of this project
# whose finder maps `dasm2` to the path it was installed from, and that is a
# *different checkout* whenever the pipeline runs from a `git worktree` -- the
# arrangement a long training run forces, because the main checkout is busy.
# `_EditableFinder` sits after `PathFinder` in `sys.meta_path`, so the
# repository wins only while its root is on `sys.path`; a script run puts
# `scripts/` there, never the root. Without this line a rule imports the other
# checkout's code and runs against it, which is a silent failure in the worst
# direction: the tree that produced the result was never the tree on disk
# here. `assert_dasm2_is_this_repo` below is the check that makes it loud.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: The vendored package itself, which is what a `dasm2` import has to resolve
#: inside. `REPO_ROOT` is too loose: `.venv/` sits under it.
VENDORED_DASM2 = REPO_ROOT / "dasm2"


def assert_dasm2_is_this_repo() -> None:
    """Fail unless every loaded `dasm2` module came from this repository.

    Called by every script that imports `dasm2`, right after it does.

    Checking the package root is not enough, and the difference matters. The
    editable install's finder maps the name `dasm2` to the directory it was
    installed from and resolves submodules against *that*, not against the
    parent package's `__path__`. So `import dasm2` can land here while
    `import dasm2.foo` lands in the other checkout -- which is exactly what
    happens to a module this repository has deleted: `PathFinder` finds
    nothing, and the editable finder then serves the copy that still exists
    over there.

    Resolution is checked against ``REPO_ROOT / "dasm2"``, not ``REPO_ROOT``:
    ``.venv/`` lives under the repository root, so a ``dasm2`` installed into
    this environment's site-packages would satisfy the looser test.
    """
    import dasm2

    leaked = sorted(
        f"{name} -> {Path(module.__file__).resolve()}"
        for name, module in list(sys.modules.items())
        if (name == "dasm2" or name.startswith("dasm2."))
        and getattr(module, "__file__", None)
        and not Path(module.__file__).resolve().is_relative_to(VENDORED_DASM2)
    )
    if leaked:
        listing = "\n  ".join(leaked)
        raise RuntimeError(
            f"{len(leaked)} `dasm2` module(s) resolved outside "
            f"{VENDORED_DASM2}:\n"
            f"  {listing}\n"
            "An editable install pointing at another checkout is shadowing "
            "the vendored fork, so this run would measure code this "
            "repository does not contain -- most likely a module deleted "
            "here that still exists there."
        )
    assert dasm2  # the import above is the point; keep linters quiet


CONFIG_PATH = REPO_ROOT / "config.yaml"

with CONFIG_PATH.open() as handle:
    CONFIG = yaml.safe_load(handle)

#: Where the pipeline writes. ``config.yaml`` is the answer unless
#: ``VIRAL_DASM_ARTIFACT_ROOT`` overrides it.
#:
#: The override exists for one case: a second checkout -- a ``git worktree``
#: used to change the code while a multi-day run is still going in the main
#: one. Both read the same ``config.yaml``, so without it a forced rerun in
#: the worktree writes into the live run's directories and can overwrite its
#: logs and checkpoints. It is deliberately not a config key: a value that
#: belongs to one checkout for a few days should not be committed, and the
#: default must stay the one written down.
ARTIFACT_ROOT = Path(
    os.environ.get("VIRAL_DASM_ARTIFACT_ROOT", CONFIG["artifact_root"])
)
GENE = CONFIG["mat_pcp"]["gene"]

#: The mat-pcp run's axes. They appear in every neutral archive's stem and in
#: the era map's ``neutral_model_name``, so they are read from ``config.yaml``
#: rather than spelled a second time in the scripts that build those names.
CUTOFF = str(CONFIG["mat_pcp"]["cutoff"])
VERSION = str(CONFIG["mat_pcp"]["version"])

TRAIN_SEED = int(CONFIG["train_seed"])

#: The three numbers every arm's split shares. The resulting fold *sizes* are
#: per corpus -- the draw sizes itself from that corpus's ``n_rows`` -- and
#: live on :class:`Corpus` rather than here.
CV_NUM_FOLDS = int(CONFIG["cv_split"]["num_folds"])
CV_FOLD_INDEX = int(CONFIG["cv_split"]["fold_index"])
CV_SEED = int(CONFIG["cv_split"]["seed"])

#: One evaluation per this many steps. The cadence the paper's reference runs
#: used, kept so a curve recorded here lines up row for row with one recorded
#: in ``dasm2-experiments``.
EVAL_EVERY_STEPS = 2000

#: Evals between intermediate checkpoint saves, matching the reference run.
CHECKPOINT_EVERY_N_EVALS = 10

#: Cycles per run. These heads need no branch-length re-optimization, so
#: ``global_step`` and ``cycle_step`` agree and a step budget is a step budget.
NUM_CYCLES = 1

#: DataLoader workers. The per-sample whichmut transform chain runs in the
#: worker, so at 0 a scoring pass is dataloader-bound and roughly 4x slower for
#: an identical answer.
DATALOADER_WORKERS = 6

#: The mat-pcp files the training config names. A typo in one of these renders
#: a config with a path that silently censors nothing, which is why the lookup
#: is an allowlist rather than a dictionary get.
TRAINING_CONFIG_FILES = frozenset(
    {
        "censored_mutations_file",
        "censored_clades_file",
        "masked_nt_sites_file",
        "reference_sequence_file",
    }
)

#: The mat-pcp files the neutral-model build reads. Separate from the set
#: above because they are inputs to a *stage* rather than fields of a rendered
#: training config, and because only this set is era-partitioned.
NEUTRAL_BUILD_FILES = frozenset(
    {
        "master_table_file",
        "master_table_pre_omicron_file",
        "master_table_omicron_file",
        "site_annotations_file",
        "era_fit_summary_file",
    }
)


def mat_pcp_path(key: str) -> Path:
    """One of the mat-pcp files ``config.yaml`` names.

    Raises:
        KeyError: If ``key`` is not one of them -- so a typo fails here rather
            than rendering a config with a path that silently censors nothing,
            or building a neutral model from a table nobody meant.
    """
    block = CONFIG["mat_pcp"]
    allowed = TRAINING_CONFIG_FILES | NEUTRAL_BUILD_FILES
    if key not in allowed:
        raise KeyError(
            f"{key!r} is not a mat-pcp file; expected one of {sorted(allowed)}"
        )
    return Path(block["root"]) / block[key]


def master_table_path(era: str | None) -> Path:
    """The fitted rate table for one era, or the pooled fit when ``era`` is None.

    The era names are the era map's, not mat-pcp's: mat-pcp fits the two
    partitions and this pipeline decides which rows belong to each. Keeping the
    mapping here means ``data/era_map.yaml`` and the neutral build cannot
    disagree about which table an era's name refers to.

    Raises:
        KeyError: If ``era`` is neither era nor None.
    """
    by_era = {
        None: "master_table_file",
        "pre_omicron": "master_table_pre_omicron_file",
        "omicron": "master_table_omicron_file",
    }
    if era not in by_era:
        raise KeyError(
            f"{era!r} is not an era; expected one of "
            f"{sorted(k for k in by_era if k is not None)}, or None for the "
            f"pooled fit."
        )
    return mat_pcp_path(by_era[era])


@dataclass(frozen=True)
class Corpus:
    """A prepared Parquet, with the things that identify it.

    ``source_pcp_md5`` is the pin that matters: the Parquet inherits its row
    order verbatim from that mat-pcp file, and the paper's fold 0 is a
    function of row order. ``md5`` is the Parquet's own, recorded so a
    rebuild's effect is visible, but not a gate -- a rebuild recomputes
    ``nt_rates``/``nt_csps`` and picks its own row-group boundaries, so the
    bytes can move while every row stays where it was. It is ``None`` for a
    corpus this pipeline builds, which has no pinned file to be compared
    against.

    ``n_train``/``n_val`` are per corpus rather than shared, because the fold
    draw sizes itself from this corpus's ``n_rows``.

    ``source_pcp_manifest`` is a provenance label only. Nothing resolves it:
    it records which data-central manifest describes this file, for a reader
    who has that repository and wants to cross-check the md5.
    """

    name: str
    path: Path
    md5: str | None
    n_rows: int
    n_train: int
    n_val: int
    row_order_digest: str
    parquet_cache_budget_gb: float
    source_pcp: Path
    source_pcp_md5: str
    source_pcp_manifest: str

    @property
    def is_built_here(self) -> bool:
        """Whether this pipeline produces the file, rather than reading a pin."""
        return self.path.is_relative_to(ARTIFACT_ROOT)


def corpus(name: str) -> Corpus:
    """The named corpus from ``config.yaml``.

    A path under the configured ``artifact_root`` is re-rooted at
    :data:`ARTIFACT_ROOT`, so a checkout running with
    ``VIRAL_DASM_ARTIFACT_ROOT`` set builds its corpora into its own tree
    instead of overwriting the live run's. Paths outside that root -- a pin in
    another project's tree -- are returned verbatim, which is what makes the
    rewrite safe: it moves outputs and never inputs.

    Raises:
        KeyError: If ``name`` is not a corpus in ``config.yaml``.
    """
    try:
        entry = CONFIG["corpora"][name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not a corpus; expected one of {sorted(CONFIG['corpora'])}"
        ) from None
    path = Path(entry["path"])
    configured_root = Path(CONFIG["artifact_root"])
    if path.is_relative_to(configured_root):
        path = ARTIFACT_ROOT / path.relative_to(configured_root)
    return Corpus(
        name=name,
        path=path,
        md5=entry.get("md5"),
        n_rows=int(entry["n_rows"]),
        n_train=int(entry["n_train"]),
        n_val=int(entry["n_val"]),
        row_order_digest=entry["row_order_digest"],
        parquet_cache_budget_gb=float(entry["parquet_cache_budget_gb"]),
        source_pcp=Path(entry["source_pcp"]),
        source_pcp_md5=entry["source_pcp_md5"],
        source_pcp_manifest=entry["source_pcp_manifest"],
    )


#: Every corpus, in ``config.yaml`` order: ``full_depth`` then the thinned arms
#: from deepest to shallowest. Used for the ``{corpus}`` wildcard constraint, so
#: a typo fails at DAG build rather than as a bare ``KeyError`` at run time.
CORPUS_NAMES = tuple(CONFIG["corpora"])

#: The corpora this pipeline builds, as opposed to the ones it reads pinned.
BUILT_CORPUS_NAMES = tuple(n for n in CORPUS_NAMES if corpus(n).is_built_here)

FULL_DEPTH = corpus("full_depth")

#: sha256 of the ``rates_table.csv`` inside each neutral archive the pipeline
#: builds, keyed by era (``"pooled"`` for the unstratified fit). Pinned to the
#: archives the paper trained against; see ``config.yaml`` for why it is the
#: member's digest rather than the zip's.
NEUTRAL_ARCHIVE_DIGESTS = dict(CONFIG["neutral_archives"])


def file_md5(path: Path, chunk_bytes: int = 1 << 24) -> str:
    """md5 of a file, read in chunks so a 1.8 GB Parquet does not go in memory."""
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


#: Row groups the Parquet row-group cache must hold to stay off the thrash
#: cliff, and the floor every corpus's budget is sized against.
#:
#: The number is a property of the geometry, not of a corpus. Training draws
#: rows through a shuffle buffer, so at any moment the reader is touching a
#: working set of row groups scattered across the file; a cache smaller than
#: that set evicts a row group it is about to need again, and every miss is a
#: decode of the whole row group to serve one row. The working set follows from
#: the sequence length and the writer's rows-per-row-group, which this pipeline
#: does not vary, so one floor covers every arm.
#:
#: A thinned arm has *fewer* row groups, not smaller ones, so the floor is if
#: anything more generous there. :func:`parquet_cache_budget_gb` still reads
#: each file, because converting a count into bytes needs that file's own
#: row-group sizes.
PARQUET_CACHE_ROW_GROUPS = 40

#: Peak RSS for one corpus build, in MB. Snakemake co-schedules on this, and it
#: enforces a budget only when the invocation supplies one -- so a run that
#: builds more than one arm needs ``--resources mem_mb=``.
#:
#: The same for every arm: the peak is set by ``rows_per_chunk`` times the gene
#: length, not by how many rows the corpus has.
PREP_MEM_MB = 16_000


def parquet_cache_row_groups(rg_byte_sizes: Sequence[int]) -> int:
    """Row groups the cache must hold: at least :data:`PARQUET_CACHE_ROW_GROUPS`.

    Split from :func:`parquet_cache_budget_gb` so the sizing can be checked
    against real geometry without a GISAID corpus to read. ``rg_byte_sizes`` are
    the row groups' parquet ``total_byte_size``.

    Never below the library's own default (``default_cache_row_groups``: as many
    row groups as fit 2 GiB of ``total_byte_size``). Shrinking the cache for a
    small arm would be a behaviour change with nothing measured behind it.

    Raises:
        ValueError: If there are no row groups, which a prepared Parquet never
            has -- reaching it means the file is not one.
    """
    from dasm2.data.parquet_backend import default_cache_row_groups

    if not rg_byte_sizes:
        raise ValueError(
            "no row groups, so there is no geometry to size a cache budget "
            "from. A prepared Parquet always has at least one."
        )
    return max(PARQUET_CACHE_ROW_GROUPS, default_cache_row_groups(rg_byte_sizes))


def parquet_cache_budget_gb(parquet_path: str | Path) -> float:
    """``DASM2_PARQUET_CACHE_BUDGET_GB`` for a corpus, from its own file.

    Read from the file's row-group metadata rather than a per-arm table, so
    re-preparing an arm at a different ``rows_per_chunk`` cannot leave a stale
    number behind. The number is only right relative to the geometry it was
    measured against, and that geometry lives in the file.

    The budget counts **decoded** bytes, so the row-group count from
    :func:`parquet_cache_row_groups` is converted at this file's median-size
    decoded row group -- which decodes that one row group, and is why the
    conversion cannot be done from row-group counts alone. Rounded up to the
    hundredth ``config.yaml`` records, so the budget never holds one row group
    fewer than asked.
    """
    import pyarrow.parquet as pq

    from dasm2.data.parquet_backend import typical_decoded_row_group_bytes

    parquet_file = pq.ParquetFile(parquet_path)
    metadata = parquet_file.metadata
    sizes = [
        metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups)
    ]
    decoded = typical_decoded_row_group_bytes([parquet_file], sizes)
    return math.ceil(parquet_cache_row_groups(sizes) * decoded / 1e7) / 100


def _check_config() -> None:
    """Fail at import on a config that cannot describe a run.

    Only invariants internal to the file are checked. Whether a path's contents
    are what the file says they are is the corpus guard's job
    (``scripts/check_corpus.py``), because that costs a read of 1.8 GB.
    """
    if not 0 <= CV_FOLD_INDEX < CV_NUM_FOLDS:
        raise ValueError(
            f"cv_split holds out fold {CV_FOLD_INDEX} of {CV_NUM_FOLDS}; "
            f"CvSplitConfig rejects a fold_index >= num_folds, hours into training."
        )
    for name in CORPUS_NAMES:
        spec = corpus(name)
        if spec.n_train + spec.n_val != spec.n_rows:
            raise ValueError(
                f"{name}'s split halves sum to {spec.n_train + spec.n_val}, but "
                f"it has {spec.n_rows} rows. The split is a partition of every "
                f"row, so these must agree: one of the three numbers is stale."
            )
        # The held-out share has to be 1/num_folds to within the draw's own
        # randomness. A fold that is 20% of the corpus means num_folds moved
        # and the recorded sizes did not.
        share = spec.n_val / spec.n_rows
        expected = 1.0 / CV_NUM_FOLDS
        if abs(share - expected) > 0.01:
            raise ValueError(
                f"fold {CV_FOLD_INDEX} is {share:.4f} of {name}, but a "
                f"{CV_NUM_FOLDS}-fold split draws about {expected:.4f}. The "
                f"recorded split sizes do not come from this num_folds."
            )
        if len(spec.row_order_digest) != 32:
            raise ValueError(
                f"{name}'s row_order_digest is {len(spec.row_order_digest)} hex "
                f"characters; blake2b-128 gives 32."
            )
        # A corpus this pipeline builds has no pinned md5 to compare against,
        # and one it reads pinned is unidentifiable without it.
        if spec.is_built_here and spec.md5 is not None:
            raise ValueError(
                f"{name} is built into the artifact root but records an md5. "
                f"A built file's bytes move with the writer's row-group "
                f"boundaries while every row stays put, so that md5 would fail "
                f"a legitimate rebuild while saying nothing about the split."
            )
        if not spec.is_built_here and spec.md5 is None:
            raise ValueError(
                f"{name} is read pinned from {spec.path} but records no md5, "
                f"so nothing identifies the file this pipeline was run against."
            )


_check_config()
