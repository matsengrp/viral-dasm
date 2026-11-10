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

GISAID-derived: the output is a per-site slice of GISAID counts joined to model
predictions. It goes to the artifact root on the cluster, never to tracked
``data/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install.
from lib import (
    ARTIFACT_ROOT,
    GENE,
    assert_dasm2_is_this_repo,
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
from gpu_slot import claim_card

from dasm2.analysis.viral import (
    normalize_fitness_columns,
    slice_fitness_for_gene,
)
from dasm2.analysis.viral_fitness_eval import SelectionFactorSource
from dasm2.io import read_fasta_first_sequence
from dasm2.neutral.scov2_rates import GENE_NT_LENGTH
from dasm2.sequence.conversion import translate_sequence
from dasm2.sequence.masking import (
    censored_codon_sites,
    load_censor_table,
    parse_masked_nt_sites_file,
)

#: The same directory `scripts/train.py` and `scripts/train_eval.py` lock, so a
#: scoring pass and a training run can never both take one card.
GPU_LOCK_DIR = Path(ARTIFACT_ROOT) / ".gpu_locks"

#: Columns carried over from the per-clade fitness table.
PER_CLADE_VALUE_COLUMNS = [
    "actual_count",
    "predicted_count",
    "delta_fitness_ratio",
    "delta_fitness_bayesian",
    "uncertainty",
]

#: Columns carried over from the clade-pooled table, suffixed on arrival.
POOLED_VALUE_COLUMNS = [
    "actual_count",
    "predicted_count",
    "delta_fitness_ratio",
    "delta_fitness_bayesian",
    "source_clade",
]

#: Suffix mat-pcp gives each per-clade fitness table.
FITNESS_SUFFIX = "_aamut_fitness.csv"


def founder_aa_sequence(clade: str, gene: str) -> str:
    """One clade's founder protein sequence, length-checked.

    Read from mat-pcp's own ``clade_founders/{gene}/{clade}.fa``, *not* derived
    from the fitness table. Taking the table's distinct ``aa_site`` values
    instead drops sites that happen to carry no fitness rows, and the resulting
    sequence is frame-shifted from the first gap onward with nothing raising.

    Raises:
        ValueError: If the translation's length disagrees with
            ``GENE_NT_LENGTH``, which is how a founder for the wrong gene, or a
            truncated FASTA, fails here rather than at the join.
    """
    path = mat_pcp_path("clade_founders_dir") / f"{clade}.fa"
    aa = translate_sequence(read_fasta_first_sequence(path))
    if len(aa) * 3 != GENE_NT_LENGTH[gene]:
        raise ValueError(
            f"{gene}/{clade}: founder translates to {len(aa)} codons but "
            f"GENE_NT_LENGTH says {GENE_NT_LENGTH[gene] // 3}"
        )
    return aa


def globally_masked_sites() -> set[int]:
    """Codon sites the global nucleotide mask covers.

    ``parse_masked_nt_sites_file`` returns 0-indexed nucleotide indices;
    ``aa_site`` is 1-indexed. One flagged nucleotide makes its whole codon
    unreadable as an amino-acid substitution, so the mask is widened to the
    codon -- correct at this granularity even though it would be too coarse for
    filtering PCPs.
    """
    nt_indices = parse_masked_nt_sites_file(mat_pcp_path("masked_nt_sites_file"))
    return {index // 3 + 1 for index in nt_indices}


def clade_roster() -> list[str]:
    """Clades to score, from the founder FASTAs.

    ``scripts/check_clades.py`` has already required this roster to match the
    fitness tables and the censor roster, so reading any one of the three gives
    the same answer.
    """
    founders = mat_pcp_path("clade_founders_dir")
    return sorted(path.stem for path in founders.glob("*.fa"))


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
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--clades",
        default=None,
        help=(
            "Comma-separated clades to score instead of the whole roster. For "
            "the fast test only; a partial table is not the figure input."
        ),
    )
    args = parser.parse_args()

    gene = GENE
    checkpoint = select_checkpoint(args.run_dir)
    run_id = f"{args.model}_{args.corpus}"
    clades = args.clades.split(",") if args.clades else clade_roster()

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
                        censored_directional={
                            int(idx) // 3 + 1 for idx in spec.directional[:, 0]
                        },
                    )
                )
                # The per-clade frames are kept but the weights are not needed
                # between clades; the cache would otherwise hold every clade's
                # [L, 20] tensor for the whole pass.
                source._factors.clear()
                source._frames.clear()
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
