"""Score one run's best checkpoint on parent haplotypes that are not founders.

`fig:within_clade_shifts` compares a mutation's effect between two genetic
backgrounds of the same clade. The DASM side of that comparison is a forward
pass on each background's amino-acid haplotype, which
:mod:`score_run` deliberately does not do: it scores clade founders and writes
``background`` empty for every row.

**Its own table, not rows in `selection_factors.parquet`.** That table is the
agreement figure's input, ``tests/test_score_run.py`` pins ``background`` empty
across all of it, and ``panels.base_frames`` filters on ``(model, clade)``
alone -- so a haplotype row sharing it would be pooled into every agreement
panel with nothing raising. The two tables join downstream on
``(clade, aa_site, parent_aa, mutant_aa, background)``.

**The cache is keyed on the haplotype, which is the whole reason this file
exists.** :class:`SelectionFactorSource` keys ``(gene, run_id, clade)`` with no
sequence in it, so a loop that passed the same clade twice would get the first
haplotype's table for the second one, silently and in the right shape. Here the
clade argument is a composite label, ``f"{clade}/{background}"``, which is what
that class's own docstring asks a multi-haplotype caller to do. The class is
vendored and is left code-identical to the source commit.

**No counts join.** The pooled counts side is a separate, CPU-only producer:
the pool a haplotype stands for is a set of branches, and summing their events
needs the corpus rather than a checkpoint. Keeping the two apart is also what
lets the counts side be re-run while a measurement is being checked without
holding a card.

**This table holds its haplotypes, and stays on the cluster for that reason.**
Dropping the ``aa_sequence`` column does not make the table sequence-free:
``parent_aa`` is the haplotype's residue at every site and ``aa_mutation``
embeds it, so one row per ``aa_site`` within a ``(clade, background)`` group
rebuilds the whole Spike haplotype. That is not a defect to engineer around --
``parent_aa`` is the key the counts side joins on, and
:mod:`score_run`'s founder table has exactly the same property for exactly the
same reason -- but it does mean this output is GISAID-derived data and belongs
under the artifact root, which is where it is written.

The place a background must appear as a *label* and nothing else is the tracked
side: the per-panel CSVs under ``results/``. The producer is free to carry what
it needs; the published tables are not.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install.
from lib import (
    GENE,
    GPU_LOCK_DIR,
    assert_dasm2_is_this_repo,
    mat_pcp_path,
    select_checkpoint,
)

# Then `dasm2`, before pandas or pyarrow: `dasm2/arrow_pool.py` moves Arrow off
# the bundled mimalloc, and can only do so if pyarrow has not been imported
# yet. This script writes a Parquet, so the allocator is not academic.
import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import argparse

import pandas as pd
from gpu_slot import claim_card
from score_run import codon_sites_of, globally_masked_sites

from dasm2.analysis.viral_fitness_eval import SelectionFactorSource
from dasm2.neutral.scov2_rates import GENE_NT_LENGTH
from dasm2.sequence.masking import censored_codon_sites, load_censor_table

#: Columns the backgrounds table must carry. ``aa_sequence`` is the haplotype
#: itself: it is read, scored, and not carried through as a column -- though
#: ``parent_aa`` reconstructs it, as the module docstring says.
BACKGROUND_COLUMNS = ("clade", "background", "aa_sequence")

#: The residues a haplotype may be spelled with. An ambiguity code or a gap is
#: the realistic failure for a reconstructed parent, and it reaches the model
#: as a silently wrong input rather than as an error.
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")

#: The columns written, in order. ``parent_aa`` is the *haplotype's* residue
#: rather than the founder's, which is the one column whose meaning differs
#: from the founder table's column of the same name.
OUTPUT_COLUMNS = (
    "gene",
    "clade",
    "background",
    "arm",
    "model",
    "aa_site",
    "parent_aa",
    "mutant_aa",
    "aa_mutation",
    "is_synonymous",
    "log_selection_factor",
    "globally_masked",
    "censored",
    "censored_directional",
)


def cache_label(clade: str, background: str) -> str:
    """The composite key one haplotype's pass is cached under.

    One spelling, because the failure this guards is two callers disagreeing
    about it: the second would then be handed the first's table. ``/`` cannot
    occur in a clade name or in a background label, so the composite cannot
    collide with a bare clade key.
    """
    if "/" in clade or "/" in background:
        raise ValueError(
            f"neither a clade nor a background label may contain '/': "
            f"got clade={clade!r}, background={background!r}."
        )
    return f"{clade}/{background}"


def read_backgrounds(path: Path, *, sites: int) -> pd.DataFrame:
    """The haplotypes to score, validated before any weight is loaded.

    Parquet only. The sequences are GISAID-derived and the table lives under
    the artifact root, which is where this pipeline keeps Parquet; a second
    accepted format would be a branch nothing produces.

    Everything that could fail is checked here rather than inside the loop,
    because the loop runs after a checkpoint is resident and a card is claimed.
    No error message echoes a sequence.
    """
    frame = pd.read_parquet(path)
    missing = [name for name in BACKGROUND_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(
            f"{path}: backgrounds table is missing {missing}; it must carry "
            f"{list(BACKGROUND_COLUMNS)}."
        )
    if frame.empty:
        raise ValueError(
            f"{path}: backgrounds table is empty, so the pass would do nothing."
        )

    for column in ("clade", "background"):
        blank = frame[column].isna() | frame[column].astype(str).str.strip().eq("")
        if blank.any():
            raise ValueError(
                f"{path}: {int(blank.sum())} row(s) have an empty {column!r}. "
                f"An empty background would also collide with the founder "
                f"table's convention, where empty means 'this is a founder'."
            )

    duplicated = frame.duplicated(["clade", "background"])
    if duplicated.any():
        pairs = frame.loc[duplicated, ["clade", "background"]].to_records(index=False)
        raise ValueError(f"{path}: duplicate (clade, background) rows: {list(pairs)}.")

    for row in frame.itertuples(index=False):
        label = f"{row.clade}/{row.background}"
        sequence = row.aa_sequence
        if not isinstance(sequence, str) or len(sequence) != sites:
            length = "null" if not isinstance(sequence, str) else len(sequence)
            raise ValueError(
                f"{path}: {label} has a {length}-residue haplotype, but the "
                f"gene has {sites} sites. A wrong length would otherwise raise "
                f"deep inside the grid builder, after inference."
            )
        unknown = sorted(set(sequence) - AA_ALPHABET)
        if unknown:
            raise ValueError(
                f"{path}: {label}'s haplotype contains {unknown}, which is not "
                f"an amino acid. An ambiguity code or a gap reaches the model "
                f"as a silently wrong input."
            )
    return frame


def score_haplotype(
    source: SelectionFactorSource,
    *,
    gene: str,
    run_id: str,
    clade: str,
    background: str,
    aa_sequence: str,
    model: str,
    corpus: str,
    masked: set[int],
    censored: set[int],
    censored_directional: set[int],
) -> pd.DataFrame:
    """One haplotype's full ``[L, 20]`` grid.

    ``aa_mutation`` and ``is_synonymous`` come out of the builder keyed on this
    sequence, so they describe a substitution *from the haplotype* rather than
    from the clade founder.
    """
    grid = source.log_sf_frame(
        gene, run_id, cache_label(clade, background), aa_sequence
    ).copy()
    grid["parent_aa"] = [aa_sequence[site - 1] for site in grid["aa_site"]]

    grid.insert(0, "gene", gene)
    grid.insert(1, "clade", clade)
    grid.insert(2, "background", background)
    grid.insert(3, "arm", corpus)
    grid.insert(4, "model", model)
    grid["globally_masked"] = grid["aa_site"].isin(masked)
    grid["censored"] = grid["aa_site"].isin(censored)
    grid["censored_directional"] = grid["aa_site"].isin(censored_directional)
    return grid[list(OUTPUT_COLUMNS)]


def score_backgrounds(
    backgrounds: pd.DataFrame,
    *,
    checkpoint: Path,
    gene: str,
    run_id: str,
    model: str,
    corpus: str,
    masked: set[int],
    censor_table,
    device: str,
) -> pd.DataFrame:
    """Every haplotype's grid, concatenated, on one resident checkpoint."""
    source = SelectionFactorSource({(gene, run_id): checkpoint}, device)
    try:
        frames = []
        for row in backgrounds.itertuples(index=False):
            spec = censor_table[row.clade]
            frames.append(
                score_haplotype(
                    source,
                    gene=gene,
                    run_id=run_id,
                    clade=row.clade,
                    background=row.background,
                    aa_sequence=row.aa_sequence,
                    model=model,
                    corpus=corpus,
                    masked=masked,
                    censored=censored_codon_sites(spec),
                    censored_directional=codon_sites_of(spec.directional[:, 0]),
                )
            )
        return pd.concat(frames, ignore_index=True)
    finally:
        source.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--backgrounds",
        required=True,
        type=Path,
        help="Parquet of (clade, background, aa_sequence) rows to score",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()

    gene = GENE
    sites = GENE_NT_LENGTH[gene] // 3
    checkpoint = select_checkpoint(args.run_dir)
    run_id = f"{args.model}_{args.corpus}"
    backgrounds = read_backgrounds(args.backgrounds, sites=sites)

    masked = globally_masked_sites()
    censor_table = load_censor_table(
        mat_pcp_path("censored_mutations_file"),
        mat_pcp_path("censored_clades_file"),
    )
    # Before the weights load and the card is claimed: a clade the censor table
    # does not know would otherwise raise a bare KeyError partway through the
    # loop, with a GPU held and the finished haplotypes thrown away.
    unknown = sorted({c for c in backgrounds["clade"] if c not in censor_table})
    if unknown:
        raise KeyError(
            f"{args.backgrounds}: the censor table has no entry for {unknown}; "
            f"it covers {len(censor_table)} clades."
        )

    print(
        f"{run_id}: scoring {checkpoint.name} on {len(backgrounds)} {gene} "
        f"haplotypes across {backgrounds['clade'].nunique()} clades, "
        f"device {args.device}",
        flush=True,
    )

    def score_all(device: str) -> pd.DataFrame:
        return score_backgrounds(
            backgrounds,
            checkpoint=checkpoint,
            gene=gene,
            run_id=run_id,
            model=args.model,
            corpus=args.corpus,
            masked=masked,
            censor_table=censor_table,
            device=device,
        )

    if args.device == "cuda":
        with claim_card(GPU_LOCK_DIR, f"score_haplotypes_{run_id}") as card:
            table = score_all(f"cuda:{card}")
    else:
        table = score_all(args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False)

    print(f"  {len(table):,} rows over {len(backgrounds)} haplotypes")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
