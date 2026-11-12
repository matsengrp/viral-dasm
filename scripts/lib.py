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
import re
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

#: Where every process that takes a card writes its lock. One name, because the
#: guarantee is that a training run and a scoring pass can never both hold one
#: GPU, and that guarantee is only as strong as the agreement about the
#: directory. ``scripts/train.py`` and ``scripts/train_eval.py`` still spell it
#: themselves; they are frozen for the duration of the training campaign and
#: adopt this on the first edit after it.
GPU_LOCK_DIR = ARTIFACT_ROOT / ".gpu_locks"

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

#: Evals between intermediate checkpoint saves, matching the reference run.
CHECKPOINT_EVERY_N_EVALS = 10

#: Cycles per run. These heads need no branch-length re-optimization, so
#: ``global_step`` and ``cycle_step`` agree and a step budget is a step budget.
NUM_CYCLES = 1

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

#: The mat-pcp inputs the scoring stage reads. Two of the three are
#: directories: the clade roster is read from their contents rather than
#: restated in `config.yaml`, so adding a clade to the mat-pcp run cannot leave
#: this pipeline scoring 56 of 57.
SCORING_INPUT_FILES = frozenset(
    {
        "aamut_fitness_dir",
        "aamut_fitness_pooled_file",
        "clade_founders_dir",
    }
)

#: The mat-pcp inputs a panel rule reads. Separate from
#: :data:`SCORING_INPUT_FILES` because no scoring job opens either: they are
#: measurements *of the tree*, read by `fig:subsampling`'s panels A and B,
#: which take no model and no checkpoint.
PANEL_INPUT_FILES = frozenset(
    {
        "clade_counts_file",
        "ntmut_fitness_dir",
    }
)

#: The mat-pcp inputs `fig:dms`'s predictor arms read. Its own set for the
#: reason :data:`PANEL_INPUT_FILES` is separate from :data:`SCORING_INPUT_FILES`:
#: no scoring job and no `fig:subsampling` panel opens this one. It holds the
#: per-leaf mutation paths the leaf log-frequency predictor reconstructs each
#: leaf's spike from, which nothing else in the pipeline reads.
DMS_INPUT_FILES = frozenset(
    {
        "sample_paths_dir",
    }
)

#: Suffix mat-pcp gives each per-clade amino-acid fitness table. Here rather
#: than in the two scripts that glob for it, because ``check_clades.py``
#: validates a roster that ``score_run.py`` then reads back by this suffix: two
#: spellings could drift and leave the guard passing while the scorer opens a
#: file that does not exist.
FITNESS_SUFFIX = "_aamut_fitness.csv"

#: Suffix mat-pcp gives each per-clade *nucleotide* fitness table, the grid
#: `fig:subsampling`'s panel B is measured over. Here for the same reason as
#: :data:`FITNESS_SUFFIX`: `workflow/figures.smk` declares these files as a
#: rule's inputs and `scripts/panels.py` opens them, so two spellings could
#: drift and leave the rule declaring a path nothing reads.
NTMUT_SUFFIX = "_ntmut_fitness.csv"

#: Suffix mat-pcp gives each per-clade *leaf path* file -- one row per leaf,
#: holding that leaf's whole root-to-tip list of nucleotide mutations. Here for
#: the same reason as the two above: `workflow/figures.smk` declares these files
#: as a rule's inputs and `scripts/dms_leaf_freq.py` opens them.
SAMPLE_PATHS_SUFFIX = "_sample_paths.tsv"


def mat_pcp_path(key: str) -> Path:
    """One of the mat-pcp files ``config.yaml`` names.

    Raises:
        KeyError: If ``key`` is not one of them -- so a typo fails here rather
            than rendering a config with a path that silently censors nothing,
            or building a neutral model from a table nobody meant.
    """
    block = CONFIG["mat_pcp"]
    allowed = (
        TRAINING_CONFIG_FILES
        | NEUTRAL_BUILD_FILES
        | SCORING_INPUT_FILES
        | PANEL_INPUT_FILES
        | DMS_INPUT_FILES
    )
    if key not in allowed:
        raise KeyError(
            f"{key!r} is not a mat-pcp file; expected one of {sorted(allowed)}"
        )
    return Path(block["root"]) / block[key]


def founder_clades(founders_dir: Path) -> set[str]:
    """Clades with a founder FASTA, from ``{clade}.fa`` basenames."""
    return {path.stem for path in founders_dir.glob("*.fa")}


def clade_roster() -> list[str]:
    """The clades to score, read from mat-pcp's founder FASTAs.

    One function, because ``scripts/check_clades.py`` requires this roster to
    match the fitness tables and the censor roster before any scoring job runs,
    and a guard that validated a different reading of the directory from the one
    the scorer uses would be no guard at all.
    """
    return sorted(founder_clades(mat_pcp_path("clade_founders_dir")))


def fitness_clades(fitness_dir: Path) -> set[str]:
    """Clades with a per-clade fitness table, excluding the pooled one.

    Globs the directory's own level rather than walking it, so the per-arm refit
    subdirectories are not reached. The pooled table has to be excluded by name:
    ``all_aamut_fitness.csv`` ends in :data:`FITNESS_SUFFIX` like any clade's,
    and its column order differs -- field 2 is ``clade_founder_aa`` where the
    per-clade files carry ``gene`` -- so reading it as a clade reads a
    single-letter residue as a gene name.
    """
    pooled = mat_pcp_path("aamut_fitness_pooled_file").name
    return {
        path.name[: -len(FITNESS_SUFFIX)]
        for path in fitness_dir.glob(f"*{FITNESS_SUFFIX}")
        if path.name != pooled
    }


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


@dataclass(frozen=True)
class DmsDataset:
    """One homolog's deep mutational scan: its file, and how to read it.

    ``md5`` is a gate rather than a label, which is the one way this differs
    from :class:`Corpus`'s. The file is one this pipeline receives rather than
    builds, so a digest is what makes a changed dataset fail loudly instead of
    quietly moving every number this figure reports. Since #30 the file it
    pins is the study's own, committed under ``data/dms/``, so a reader's
    checkout either matches it or does not.

    ``clade`` is the Nextstrain label this pipeline keys its tables on. The
    papers name their homologs by Pango alias, and the two are not
    interchangeable in any join here.

    The rest declares the conversion `scripts/dms_published.py` runs, so that
    it branches on what a dataset says rather than on its name.
    ``effect_column`` is the measured phenotype. The four that follow are
    empty for a file that already carries one row per mutation:

    - ``mutation_column`` names a column of ``A1015D``-style strings to parse
      into the three join keys, for a file carrying no site, wildtype or
      mutant column of its own.
    - ``replicate_column`` names the replicates ``effect_column`` is averaged
      over.
    - ``fixed`` pins a column to one value, for a file holding a sweep.
    - ``min_times_seen`` is a ``(column, floor)`` pair applied as a row filter.
    """

    homolog: str
    clade: str
    path: Path
    md5: str
    effect_column: str
    mutation_column: str | None = None
    replicate_column: str | None = None
    fixed: tuple[tuple[str, float], ...] = ()
    min_times_seen: tuple[str, int] | None = None


def dms_homologs() -> list[str]:
    """The homologs ``fig:dms`` scores, in the order ``config.yaml`` lists them."""
    return list(CONFIG["dms"]["homologs"])


def dms_dataset(homolog: str) -> DmsDataset:
    """The named homolog's DMS dataset from ``config.yaml``.

    ``root`` is anchored to :data:`REPO_ROOT` rather than resolved as given.
    It is the one input path that lives inside the checkout, and a relative
    path would otherwise resolve against the working directory -- correct
    under snakemake and pytest, which both start at the repository root, and
    silently wrong for anything else.

    Raises:
        KeyError: If ``homolog`` is not one of them. An allowlist for the same
            reason :func:`mat_pcp_path` is one: these names are written into
            panel filenames and read back as join keys, so a typo that resolved
            to nothing would produce an empty panel rather than an error.
    """
    block = CONFIG["dms"]
    try:
        entry = block["homologs"][homolog]
    except KeyError:
        raise KeyError(
            f"{homolog!r} is not a DMS homolog; expected one of "
            f"{sorted(block['homologs'])}"
        ) from None
    times_seen = entry.get("min_times_seen")
    return DmsDataset(
        homolog=homolog,
        clade=str(entry["clade"]),
        path=REPO_ROOT / block["root"] / entry["file"],
        md5=str(entry["md5"]),
        effect_column=str(entry["effect_column"]),
        mutation_column=entry.get("mutation_column"),
        replicate_column=entry.get("replicate_column"),
        fixed=tuple((str(k), v) for k, v in (entry.get("fixed") or {}).items()),
        min_times_seen=(
            None
            if times_seen is None
            else (str(times_seen["column"]), int(times_seen["value"]))
        ),
    )


def cap_of(corpus_name: str) -> int:
    """The per-clade cap a thinned corpus's name records.

    ``cap_50000_0`` is a cap of 50,000. Read off the name rather than restated
    in a table, so a cap cannot drift from the arm that was built -- and in one
    place, because panel A draws a rule per cap and panel B labels a curve per
    cap from the same number.
    """
    parts = corpus_name.split("_")
    if len(parts) != 3 or parts[0] != "cap" or not parts[1].isdigit():
        raise ValueError(
            f"{corpus_name!r} is not a thinned corpus: a cap arm is named "
            f"`cap_<n>_<fold>`, as `config.yaml`'s corpora block spells them."
        )
    return int(parts[1])


#: The wiring smoke's corpus, and its md5. Not a :class:`Corpus`: it records
#: no fold sizes and no row-order digest, because no result depends on its
#: split. The smoke asks whether the whole path runs, not what it answers.
SMOKE_CORPUS = Path(CONFIG["smoke_corpus"]["path"])
SMOKE_CORPUS_MD5 = str(CONFIG["smoke_corpus"]["md5"])


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


def corpus_max_aa_seq_len(parquet_path: str | Path) -> int:
    """The corpus's longest amino-acid sequence, from its own Parquet footer.

    ``ci-484k``'s head sizes a ``[max_aa_seq_len, 20, 20]`` parameter from
    this number, so it is read from the file the run trains against rather
    than written down anywhere. A table shorter than the sequence raises
    ``IndexError`` on the first batch; a longer one allocates rows that can
    never receive a gradient and quietly changes the effective parameter
    count the model is named for.

    The *writer* measured it, at the length it actually wrote, so this cannot
    drift from the corpus the way a constant would.

    Raises:
        KeyError: If the file carries no ``max_aa_seq_len``, which a Parquet
            prepared by this pipeline always does.
    """
    from dasm2.data.parquet_backend import read_parquet_file_metadata

    metadata = read_parquet_file_metadata(Path(parquet_path))
    try:
        return int(metadata["max_aa_seq_len"])
    except KeyError:
        raise KeyError(
            f"{parquet_path} carries no `max_aa_seq_len` in its footer "
            f"metadata, so there is nothing to size a per-site head's "
            f"parameter table from. Every Parquet "
            f"`scripts/prepare_viral_parquet.py` writes has one."
        ) from None


#: The checkpoint a finished run is selected by. At :data:`NUM_CYCLES` 1 the
#: trainer writes exactly one per output directory.
BEST_CHECKPOINT_GLOB = f"*_cycle{NUM_CYCLES - 1}_best_*.zip"


def select_checkpoint(output_dir: str | Path) -> Path:
    """The one best-validation checkpoint a finished run left behind.

    A function rather than a rule: ``scripts/read_run.py`` needs the path for
    its checksum column, so a separate rule would run the same glob in two
    processes and let them disagree.

    Deliberately not a pattern match on the filename. Checkpoint names carry
    a config-metadata hash and a run id, so a directory holding two attempts
    offers no lexical order that means anything -- the reason
    ``train.py::latest_resumable`` sorts by mtime. Here the answer is that
    there should only ever be one, so two is reported rather than resolved.

    A resumed run still leaves one: the trainer writes a cycle's best only
    when that cycle ends, and an attempt killed partway through never gets
    there. Two means two attempts each ran a cycle to completion in the same
    directory, which is a different mistake.

    Raises:
        FileNotFoundError: If the directory holds no best checkpoint, which
            means the run never finished a cycle.
        RuntimeError: If it holds more than one, which means two attempts
            shared an output directory and the selected weights would depend
            on which this glob returned first.
    """
    found = sorted(Path(output_dir).glob(BEST_CHECKPOINT_GLOB))
    if not found:
        raise FileNotFoundError(
            f"no {BEST_CHECKPOINT_GLOB} under {output_dir}. The trainer writes "
            f"one at the end of each cycle, so a run that reached its stop has "
            f"one; this directory holds a run that did not."
        )
    if len(found) > 1:
        listing = "\n  ".join(p.name for p in found)
        raise RuntimeError(
            f"{len(found)} best checkpoints under {output_dir}:\n  {listing}\n"
            f"Two attempts shared this directory. Which one is 'the' selected "
            f"checkpoint is then a question about glob order, so it is raised "
            f"rather than answered: move or remove the stale attempt."
        )
    return found[0]


#: A well-formed md5, as this file must spell one. Length alone would accept a
#: digest YAML parsed as a number and stringified, which is the way a pinned
#: digest silently stops being one.
_MD5_HEX = re.compile(r"[0-9a-f]{32}")


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
    if len(SMOKE_CORPUS_MD5) != 32:
        raise ValueError(
            f"smoke_corpus.md5 is {len(SMOKE_CORPUS_MD5)} hex characters; md5 gives 32."
        )
    seen_clades: dict[str, str] = {}
    for homolog in dms_homologs():
        dataset = dms_dataset(homolog)
        if not _MD5_HEX.fullmatch(dataset.md5):
            raise ValueError(
                f"dms.homologs.{homolog}.md5 is {dataset.md5!r}, which is not "
                f"32 lowercase hex characters. It is a gate here, not a label "
                f"-- the file it pins is replaced in place when a dataset is "
                f"repointed -- so a digest YAML coerced to a number, or one "
                f"with the right length and the wrong alphabet, must fail "
                f"here rather than at the comparison."
            )
        # A `fixed` value is compared against a column parsed out of a CSV, so
        # it has to arrive as a number. `4e-05` in YAML is the *string*
        # `'4e-05'` -- PyYAML's float resolver wants a decimal point, so only
        # `4.0e-05` parses -- and a string compares unequal to every row,
        # leaving an empty frame rather than an error at the point of use.
        for column, value in dataset.fixed:
            # ValueError, not the TypeError ruff's TRY004 would prefer: every
            # other failure in this function is a ValueError, and a caller
            # distinguishing a mistyped config key from a mistyped config value
            # would be drawing a line `config.yaml`'s reader does not have.
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(  # noqa: TRY004
                    f"dms.homologs.{homolog}.fixed.{column} is {value!r}, which "
                    f"is {type(value).__name__} rather than a number. YAML reads "
                    f"`4e-05` as a string and `4.0e-05` as a float; written the "
                    f"first way this would match no row and the figure would be "
                    f"drawn on an empty frame."
                )
        if dataset.clade in seen_clades:
            raise ValueError(
                f"{homolog} and {seen_clades[dataset.clade]} both name clade "
                f"{dataset.clade}. A clade's founder carries one set of "
                f"selection factors, so two homologs on it would be scored "
                f"against identical predictions and read as agreement."
            )
        seen_clades[dataset.clade] = homolog


_check_config()
