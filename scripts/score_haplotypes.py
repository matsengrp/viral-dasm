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
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from gpu_slot import claim_card
from panel_spec import candidate_cells
from score_run import codon_sites_of, globally_masked_sites

from dasm2.analysis.viral_fitness_eval import SelectionFactorSource
from dasm2.neutral.scov2_rates import GENE_NT_LENGTH
from dasm2.sequence.constants import AA_STR_SORTED
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
    cells: Sequence[tuple[int, str]],
    masked: set[int],
    censored: set[int],
    censored_directional: set[int],
) -> list[dict]:
    """One haplotype's rows, for the cells asked for and no others.

    ``cells`` is ``(aa_site, mutant_aa)``, matched **on the site and the
    mutant residue** rather than on a mutation string. A candidate is named
    against the clade founder, ``D574V``, while ``aa_mutation`` on a haplotype
    spells that haplotype's own parent residue -- so a string match would drop
    exactly the rows where the haplotype differs from the founder, which is
    the comparison the panel exists to draw.

    **The ``[L, 20]`` tensor, not the frame.** ``log_sf_frame`` builds 25,460
    rows through a Python append loop, caches them under the composite label,
    and never evicts: at 8,693 haplotypes that is 8.7 GB resident and about
    three minutes of pandas, for the handful of cells a panel reads. The
    tensor behind it is 100 KB and is what gets indexed here. It is cached
    too, so the cache is cleared by the caller between haplotypes.

    ``parent_aa`` is the *haplotype's* residue, which is what the counts side
    joins on and the one column whose meaning differs from the founder
    table's column of the same name.
    """
    log_sf = np.log(
        source.selection_factors(
            gene, run_id, cache_label(clade, background), aa_sequence
        ).numpy()
    )
    rows = []
    for aa_site, mutant_aa in cells:
        parent_aa = aa_sequence[aa_site - 1]
        rows.append(
            {
                "gene": gene,
                "clade": clade,
                "background": background,
                "arm": corpus,
                "model": model,
                "aa_site": aa_site,
                "parent_aa": parent_aa,
                "mutant_aa": mutant_aa,
                "aa_mutation": f"{parent_aa}{aa_site}{mutant_aa}",
                "is_synonymous": mutant_aa == parent_aa,
                "log_selection_factor": float(
                    log_sf[aa_site - 1, AA_STR_SORTED.index(mutant_aa)]
                ),
                "globally_masked": aa_site in masked,
                "censored": aa_site in censored,
                "censored_directional": aa_site in censored_directional,
            }
        )
    return rows


def score_backgrounds(
    backgrounds: pd.DataFrame,
    cells_by_clade: Mapping[str, Sequence[tuple[int, str]]],
    *,
    checkpoint: Path,
    gene: str,
    run_id: str,
    model: str,
    corpus: str,
    masked: set[int],
    censor_table,
    device: str,
    progress_every: int = 500,
) -> pd.DataFrame:
    """Every haplotype's cells, on one resident checkpoint.

    The source's per-haplotype caches are cleared as the loop goes. They are
    keyed on the composite label and never evicted, and this loop runs over
    thousands of haplotypes rather than the 56 founders the class was written
    for, so holding them would cost gigabytes to no purpose: nothing asks for
    a haplotype twice.
    """
    source = SelectionFactorSource({(gene, run_id): checkpoint}, device)
    try:
        rows = []
        for index, row in enumerate(backgrounds.itertuples(index=False), start=1):
            spec = censor_table[row.clade]
            rows.extend(
                score_haplotype(
                    source,
                    gene=gene,
                    run_id=run_id,
                    clade=row.clade,
                    background=row.background,
                    aa_sequence=row.aa_sequence,
                    model=model,
                    corpus=corpus,
                    cells=cells_by_clade[row.clade],
                    masked=masked,
                    censored=censored_codon_sites(spec),
                    censored_directional=codon_sites_of(spec.directional[:, 0]),
                )
            )
            source._factors.clear()
            source._frames.clear()
            if index % progress_every == 0:
                print(f"  {index:,}/{len(backgrounds):,} haplotypes", flush=True)
        return pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))
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
    # CPU by default, where `score_run.py` defaults to cuda. Measured on
    # `dasm-1077k_full_depth`: one 1,273-residue haplotype is 28.5 ms on CPU
    # against 4.7 ms on an idle A40, so the whole 8,693-haplotype pass is 4.1
    # minutes rather than 0.7. A 3.4-minute saving does not justify holding a
    # card that a training run could be using, and it is what keeps every
    # rule of `fig:within_clade_shifts` off the GPU.
    parser.add_argument("--device", default="cpu", choices=("cuda", "cpu"))
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

    # The cells each clade's haplotypes are scored at: its candidate
    # mutations, named against the clade founder. Every candidate, not only
    # the ones on the figure, so the designed null is measured too -- it is a
    # handful of extra cells per haplotype and the whole point of a null is
    # that it goes through the same pipeline.
    cells_by_clade = candidate_cells()
    missing = sorted(set(backgrounds["clade"]) - set(cells_by_clade))
    if missing:
        raise KeyError(
            f"{args.backgrounds}: no candidate mutations for clade(s) "
            f"{missing}. Scoring their haplotypes would write no rows, so the "
            f"pool average would have nothing to average."
        )

    print(
        f"{run_id}: scoring {checkpoint.name} on {len(backgrounds):,} {gene} "
        f"haplotypes across {backgrounds['clade'].nunique()} clades at "
        f"{sum(len(c) for c in cells_by_clade.values())} cells, "
        f"device {args.device}",
        flush=True,
    )

    def score_all(device: str) -> pd.DataFrame:
        return score_backgrounds(
            backgrounds,
            cells_by_clade,
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
