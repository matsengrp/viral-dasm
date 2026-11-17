"""The DASM collapsed to one number per site, as a predictor of the spike DMS.

The model read with its within-site resolution removed. At each site this takes
the average of the DASM's own log selection factors over the amino-acid
mutations the whichmut loss is defined on -- those one nucleotide change to the
founder's codon reaches -- and hands that single number to **every** measured
cell at the site, whichever stratum the cell is in.

**What it is for.** `main.tex` reports that the DASM's estimates for
multi-nucleotide amino-acid mutations still correlate with the DMS although it
never trains on them. This arm is the null that says what that correlation is
made of: a predictor with no residue-level information at all, over exactly the
cells `dms_dasm`'s own rows are scored on, so the two correlations in
`results/panels/dms/predictor_correlations.csv` can be read against each other
per stratum. Issue #60 carries the measurement and the argument.

**Both strata, which is the whole point.** A one-stratum arm would answer half
the question. Scored on the single-nucleotide cells it says what the model's
residue-level resolution is worth where the model is trained; scored on the
multi-nucleotide cells it says what survives the extrapolation off that
stratum. The arm therefore declares ``dms.STRATA``, like `dms_dasm` and unlike
`dms_prob`.

**The average is taken over the model's whole grid at the site, not over the
measured cells there.** Restricting it to measured cells would leave the
multi-nucleotide cells at a site the assay sampled only multi-nucleotide
mutations at with no value at all, and `within_clade_metrics` would drop
them, so this arm and
`dms_dasm`'s would stop being correlated over the same population. That
identical population is the one property the comparison rests on.
`dms_prob.founder_frame` chooses the same way for a related reason.

**Synonymous cells are excluded** because `selection_factors.parquet` is
gauge-fixed and every one of them is exactly ``log f = 0``. Including them
would pull each site's average toward zero in proportion to how many
synonymous neighbours its codon has, which is a fact about the genetic code
rather than about the model. **Censored and masked mutations are excluded**
because the model was told to ignore them, and because every other population
in this pipeline carries that mask.

**A single-nucleotide cell is one of the terms in its own site average.** That
is what the plain average means, and it is the one definition both strata can
share; a reader of this arm's ``single_nt`` row should know it is reading one.

**The mean rather than the median**, because Pearson *r* is what the table
reports and a mean is the aggregate a linear statistic is about.

**No inference and no checkpoint.** ``rule score`` did the forward passes and
`selection_factors.parquet` holds the result, so this is a column-pruned
Parquet read, a filter and a groupby.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install.
import lib
from lib import assert_dasm2_is_this_repo, dms_dataset, dms_homologs

import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import dms
import dms_censor
import dms_dasm
import dms_models
import pandas as pd

#: This predictor's id, as `results/panels/dms/` carries it and
#: `scripts/panel_spec.py` lists in :data:`panel_spec.DMS_PREDICTORS`.
DASM_SITE_MEAN = "dasm_site_mean"


def site_means(
    grid: pd.DataFrame, reachable: pd.DataFrame, censored: frozenset
) -> pd.Series:
    """One value per site: the mean ``log f`` over the trained mutations there.

    Takes frames and a set and returns a Series, reading no file, so the
    statistic can be tested without the artifact root -- the same split
    `dms_prob.linear_selection_factors` is written for.

    Args:
        grid: :func:`dms_dasm.selection_factor_grid`'s rows for one clade, the
            whole ``[L, 20]`` of them.
        reachable: ``dms.single_nt_indicator``'s frame for that clade.
        censored: ``dms_censor.censored_mutations``' set for that clade.

    Returns:
        A Series indexed by ``aa_site``, covering only the sites that have at
        least one surviving mutation. A site missing from it is not an error
        here -- :func:`assign_site_means` is where a *measured* cell at such a
        site becomes one.
    """
    trained = dms_models.trained_single_nt_cells(grid, reachable, censored)
    return trained.groupby("aa_site")[dms_dasm.SCORE_COLUMN].mean()


def assign_site_means(cells: pd.DataFrame, means: pd.Series) -> pd.DataFrame:
    """Every measured cell carrying its site's value.

    Both strata deliberately: two cells at one site get the same number
    whether one nucleotide change reaches them or three. That equality is the
    arm's defining property.

    Raises:
        ValueError: If any measured cell sits at a site :func:`site_means` has
            no value for, which happens only when every single-nucleotide
            mutation at that site is synonymous or censored. Raised rather
            than dropped: ``validate_scores`` would refuse the NaN and
            ``dms_summary``'s coverage check would report a short ``n``, but
            both would name the symptom instead of this cause.
    """
    scored = cells.assign(score=cells["aa_site"].map(means))
    unscored = scored[scored["score"].isna()]
    if len(unscored):
        # The homolog comes off the frame, as `homolog_cells` puts it there,
        # so this reads like every other predictor's failure.
        who = sorted(set(unscored["homolog"]))
        sample = sorted({int(site) for site in unscored["aa_site"]})[:5]
        raise ValueError(
            f"{'/'.join(who)}: {len(unscored)} of {len(scored)} measured cells "
            f"sit at sites with no value to take, either because the scored "
            f"grid does not cover the site or because every single-nucleotide "
            f"mutation there is synonymous or censored. First few sites: "
            f"{sample}."
        )
    return scored


def site_mean_scores(
    homolog: str, table_path: str | Path
) -> tuple[pd.DataFrame, dms_models.Manifest, str]:
    """One homolog's cells, each carrying its site's mean ``log f``."""
    cells, founder = dms_models.homolog_cells(homolog)
    clade = dms_dataset(homolog).clade
    grid = dms_dasm.selection_factor_grid(clade, table_path)

    # The grid's own parent residues, checked against the founder the cells
    # were built from. `dms_dasm` can skip this because it never reads
    # `parent_aa`; this arm takes its synonymous test from that column, so a
    # grid read for another clade would quietly average a different set --
    # `dms_prob._check_grid_founder` guards the same thing for the same
    # reason, and is not imported here because `dms_prob` pulls in torch.
    declared = [founder[site - 1] for site in grid["aa_site"]]
    disagree = grid[[a != b for a, b in zip(declared, grid["parent_aa"])]]
    if len(disagree):
        row = disagree.iloc[0]
        raise ValueError(
            f"{homolog}: the scored grid's parent_aa disagrees with the "
            f"{clade} founder at {len(disagree)} of {len(grid)} rows, e.g. "
            f"site {int(row['aa_site'])}: the table says {row['parent_aa']!r} "
            f"and the founder has {founder[int(row['aa_site']) - 1]!r}. The "
            f"table was read for a different clade than the founder FASTA."
        )

    means = site_means(
        grid,
        dms.single_nt_indicator(clade, lib.GENE),
        dms_censor.censored_mutations(clade),
    )
    frame = assign_site_means(cells, means).assign(model=DASM_SITE_MEAN)

    manifest = dms_models.Manifest(
        model=DASM_SITE_MEAN,
        strata=dms.STRATA,
        frame=(
            f"clade founder {clade} {lib.GENE}, from mat-pcp's own "
            f"clade_founders FASTA; 1-indexed, so no site mapping. Score is the "
            f"mean log selection factor over the non-synonymous, uncensored, "
            f"single-nucleotide-reachable mutations at the site, taken over the "
            f"model's whole grid rather than over the measured cells"
        ),
        input_sequence_sha256=dms_models.sequence_sha256(founder),
    )
    return frame[list(dms_models.PREDICTOR_COLUMNS)], manifest, founder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--homolog", required=True, choices=dms_homologs())
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    frame, manifest, sequence = site_mean_scores(args.homolog, args.table)
    dms_models.write_scores(frame, manifest, sequence, args.output)
    print(f"wrote {args.output}: {len(frame)} cells for {args.homolog}")


if __name__ == "__main__":
    main()
