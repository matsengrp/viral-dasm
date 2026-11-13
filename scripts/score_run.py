"""Score one run's best checkpoint against mat-pcp's fitness effects.

One forward pass per clade founder, joined to the counts-based amino-acid
effects for the same mutations, written raw: unclipped, no count gate, no clade
roster applied. Those three are swept axes downstream, so applying any of them
here would force a rescore to move it.

**The join is left, from the model grid.** A model scores every cell: 1,273
sites x 20 amino acids per clade, 56 clades, so 1,425,760 rows per run. mat-pcp
estimates only mutations it observed or expected, so most of those cells carry
null fitness columns. The grid is kept rather than the intersection because the
DMS comparison scores cells mat-pcp never estimated, and because an inner join
is recoverable from a left join by dropping nulls while the reverse is not.
``merge_fitness_and_sf`` is an inner join and is deliberately not used here.

**Full depth for every arm, the capped ones included.** For the subsampling
figure the full-depth counts estimate is the yardstick a thinned-corpus model is
measured against, so the per-arm refits under ``aamut_fitness/cap_*_0/`` are a
different comparison and are not read.

**One forward pass per clade, not per mutation.** The selection factor is a
property of the founder sequence, so a clade's whole ``[L, 20]`` table comes out
of a single pass. ``background`` is written empty for every row: this script
scores founders. Scoring other parent haplotypes needs a per-haplotype pass, and
:class:`SelectionFactorSource`'s cache is keyed ``(gene, run_id, clade)`` with no
sequence in it, so a haplotype loop built on this class would return the first
haplotype's table for every one of them, silently.

The output goes to the artifact root on the cluster because of its size -- 1.4M
rows per run -- not because of the data rules. Counts, fitness effects and
selection factors are publishable; the per-panel CSVs M5 draws from this table
are what land in tracked ``results/``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install.
from lib import (
    FITNESS_SUFFIX,
    GENE,
    GPU_LOCK_DIR,
    assert_dasm2_is_this_repo,
    clade_roster,
    mat_pcp_path,
    select_checkpoint,
)

# Then `dasm2`, before pandas or pyarrow: `dasm2/arrow_pool.py` moves Arrow off
# the bundled mimalloc, and can only do so if pyarrow has not been imported
# yet. This script writes a 1.4M-row Parquet, so the allocator is not academic.
import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import argparse

import pandas as pd
from founders import founder_aa_sequence
from gpu_slot import claim_card

from dasm2.analysis.viral import (
    normalize_fitness_columns,
    slice_fitness_for_gene,
)
from dasm2.analysis.viral_fitness_eval import SelectionFactorSource
from dasm2.sequence.masking import (
    censored_codon_sites,
    load_censor_table,
    parse_masked_nt_sites_file,
)

#: Columns carried over from the per-clade fitness table.
PER_CLADE_VALUE_COLUMNS = [
    "actual_count",
    "predicted_count",
    "delta_fitness_ratio",
    "delta_fitness_bayesian",
    "uncertainty",
]

#: Columns carried over from the clade-pooled table. The four that collide with
#: :data:`PER_CLADE_VALUE_COLUMNS` arrive suffixed ``_all``; ``source_clade``
#: has no per-clade counterpart, so it arrives under its own name.
POOLED_VALUE_COLUMNS = [
    "actual_count",
    "predicted_count",
    "delta_fitness_ratio",
    "delta_fitness_bayesian",
    "source_clade",
]


def codon_sites_of(nt_indices: Iterable[int]) -> set[int]:
    """The 1-indexed codon sites a set of 0-indexed nucleotide indices touches.

    mat-pcp flags nucleotides; this table is keyed by ``aa_site``. One flagged
    nucleotide makes its whole codon unreadable as an amino-acid substitution,
    so the mask is widened to the codon -- correct at this granularity even
    though it would be too coarse for filtering PCPs. The ``+ 1`` is the
    0-indexed-to-1-indexed step, and it lives here once because an off-by-one
    applied to one of the two masks and not the other would shift a whole
    column's worth of flags by a site with nothing raising.
    """
    return {int(index) // 3 + 1 for index in nt_indices}


def globally_masked_sites() -> set[int]:
    """Codon sites the global nucleotide mask covers."""
    return codon_sites_of(
        parse_masked_nt_sites_file(mat_pcp_path("masked_nt_sites_file"))
    )


def per_clade_fitness(clade: str, gene: str, founder_aa: str) -> pd.DataFrame:
    """mat-pcp's per-clade effects for one clade, founder-verified.

    ``slice_fitness_for_gene`` checks every row's ``clade_founder_aa`` against
    the founder residue at its site and raises on any disagreement, which is
    what makes a founder from the wrong clade fail loudly rather than mis-pair
    mutations.
    """
    path = mat_pcp_path("aamut_fitness_dir") / f"{clade}{FITNESS_SUFFIX}"
    frame = normalize_fitness_columns(pd.read_csv(path))
    return slice_fitness_for_gene(frame, gene, founder_aa=founder_aa)


def pooled_fitness(gene: str) -> pd.DataFrame:
    """mat-pcp's clade-pooled effects, keyed for a founder-residue join.

    ``founder_aa`` is deliberately None. The pooled table carries more than one
    ``clade_founder_aa`` at some sites -- it is pooled over clades whose
    founders differ there -- so the founder-agreement check cannot run on it and
    would raise. The residue is instead carried into the join key, so a pooled
    row only reaches a grid row whose parent residue it actually describes.
    """
    frame = normalize_fitness_columns(
        pd.read_csv(mat_pcp_path("aamut_fitness_pooled_file"))
    )
    sliced = slice_fitness_for_gene(frame, gene, founder_aa=None)
    return sliced.rename(columns={"clade_founder_aa": "parent_aa"})


def score_clade(
    source: SelectionFactorSource,
    *,
    gene: str,
    run_id: str,
    clade: str,
    model: str,
    corpus: str,
    pooled: pd.DataFrame,
    masked: set[int],
    censored: set[int],
    censored_directional: set[int],
) -> pd.DataFrame:
    """One clade's full grid, with fitness left-joined onto it."""
    founder_aa = founder_aa_sequence(clade, gene)
    grid = source.log_sf_frame(gene, run_id, clade, founder_aa).copy()
    grid["parent_aa"] = [founder_aa[site - 1] for site in grid["aa_site"]]

    fitness = per_clade_fitness(clade, gene, founder_aa)
    merged = grid.merge(
        fitness[["aa_site", "mutant_aa", *PER_CLADE_VALUE_COLUMNS]],
        on=["aa_site", "mutant_aa"],
        how="left",
        validate="one_to_one",
    )

    pooled_keyed = pooled[["aa_site", "parent_aa", "mutant_aa", *POOLED_VALUE_COLUMNS]]
    merged = merged.merge(
        pooled_keyed,
        on=["aa_site", "parent_aa", "mutant_aa"],
        how="left",
        suffixes=("", "_all"),
        validate="one_to_one",
    )

    merged.insert(0, "gene", gene)
    merged.insert(1, "clade", clade)
    # Reserved, and written empty: this script scores founders. See the module
    # docstring for why a haplotype loop cannot be built on this class's cache.
    merged.insert(2, "background", "")
    merged.insert(3, "arm", corpus)
    merged.insert(4, "model", model)
    merged["globally_masked"] = merged["aa_site"].isin(masked)
    merged["censored"] = merged["aa_site"].isin(censored)
    merged["censored_directional"] = merged["aa_site"].isin(censored_directional)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()

    # No `--clades`: there is no partial-table mode. The whole roster or
    # nothing, because a subset written to the rule's declared output is a
    # figure input that looks complete and is not. The tests that score two
    # clades call `score_clade` directly.
    gene = GENE
    checkpoint = select_checkpoint(args.run_dir)
    run_id = f"{args.model}_{args.corpus}"
    clades = clade_roster()

    masked = globally_masked_sites()
    censor_table = load_censor_table(
        mat_pcp_path("censored_mutations_file"),
        mat_pcp_path("censored_clades_file"),
    )
    pooled = pooled_fitness(gene)

    print(
        f"{run_id}: scoring {checkpoint.name} on {len(clades)} {gene} clade "
        f"founders, device {args.device}",
        flush=True,
    )

    def score_all(device: str) -> pd.DataFrame:
        source = SelectionFactorSource({(gene, run_id): checkpoint}, device)
        try:
            frames = []
            for clade in clades:
                spec = censor_table[clade]
                frames.append(
                    score_clade(
                        source,
                        gene=gene,
                        run_id=run_id,
                        clade=clade,
                        model=args.model,
                        corpus=args.corpus,
                        pooled=pooled,
                        masked=masked,
                        censored=censored_codon_sites(spec),
                        censored_directional=codon_sites_of(spec.directional[:, 0]),
                    )
                )
            # The source's per-clade caches are left alone. They hold a
            # [1273, 20] tensor and its frame per clade, about 6 MB over the
            # whole roster against a 1.3 GB pass, and each clade is visited
            # once, so there is nothing to free and nothing to invalidate.
            # `release()` drops the resident weights, which are the part worth
            # freeing.
            return pd.concat(frames, ignore_index=True)
        finally:
            source.release()

    if args.device == "cuda":
        with claim_card(GPU_LOCK_DIR, f"score_{run_id}") as card:
            table = score_all(f"cuda:{card}")
    else:
        table = score_all(args.device)

    table["actual_count"] = table["actual_count"].astype("Int64")
    table["actual_count_all"] = table["actual_count_all"].astype("Int64")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False)

    matched = int(table["delta_fitness_ratio"].notna().sum())
    pooled_matched = int(table["delta_fitness_ratio_all"].notna().sum())
    print(
        f"  {len(table):,} rows   {matched:,} with per-clade fitness   "
        f"{pooled_matched:,} with pooled fitness"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
