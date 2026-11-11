"""How conserved a site is among the clade's own leaves, as a `fig:dms` baseline.

The score for a substitution is ``log((mutant leaf count + p) / (parent leaf
count + p))`` at its site, where the counts come from
`scripts/dms_leaf_freq.py`'s tally. The log-odds is against the clade founder's
own count at the same site rather than against a background composition, which
is what puts it on the same axis as BLOSUM62's log-odds and the DASM's log
selection factor; a conventional PSSM's ``log(p_obs / p_background)`` would be
comparable to neither.

**What it measures is the site, not the substitution.** Measured on all three
homologs: ``-log(parent + 1)`` alone -- one minus the site's deviation fraction,
which carries no information at all about which residue the substitution goes to
-- scores Spearman +0.381 / +0.384 / +0.508 against the DMS, matching or beating
the full score's +0.338 / +0.404 / +0.526. The mutant term adds almost nothing,
and the pseudocount is inert over five orders of magnitude for the same reason.
Centre both the score and the measurement on their per-site means and the leaf
predictor falls to +0.19 / +0.17 / +0.10 while BLOSUM62 -- which knows nothing
about the site -- rises to +0.35 / +0.27 / +0.31.

So this is a **per-site conservation statistic**, and the pooled correlation a
panel would print is dominated by between-site variance. That is a real and
strong signal: how variable a spike site is among sampled leaves predicts that
site's mean DMS effect. It is not a statement about which substitution at a site
is tolerated, and `scripts/dms_summary.py` reports the within-site correlation
beside the pooled one so the two cannot be confused.

**It is also a standing-frequency measure, not a count of evidence.** A clade's
leaves are not independent observations: one old substitution with 50,000
descendants outweighs fifty independent ones. The tally weights a substitution by
its descendant count rather than by the number of times it arose, which is
deliberately not mat-pcp's per-branch event count.
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
import dms_leaf_freq
import dms_models
import numpy as np
import pandas as pd

#: This predictor's id, as `results/panels/dms/` carries it and
#: `scripts/panel_charts.py` maps to a label.
LEAF_LOGFREQ = "leaf_logfreq"

#: The pseudocount in the log frequency ratio. **Not**
#: ``viral_fitness_eval.RATIO_PSEUDOCOUNT``, which regularises a count ratio
#: rather than a residue frequency; they hold different numbers for different
#: reasons and must not be assumed to move together.
#:
#: Its value barely matters, which is itself the finding. Every unobserved
#: mutant lands at ``-log(parent + p)``, which is monotone in the parent count
#: for any ``p``, so the ranking of the 77% of cells whose mutant residue the
#: clade has never shown is the site statistic regardless: sweeping
#: ``p`` over 0.01 to 1000 moves KP.3.1.1's pooled Spearman from +0.5262 to
#: +0.5315. One is kept because it is the plain Laplace choice and nothing is
#: gained by tuning a parameter the result does not depend on.
LEAF_PSEUDOCOUNT = 1.0


def leaf_logfreq_scores(
    homolog: str, counts_path: Path
) -> tuple[pd.DataFrame, dms_models.Manifest, str]:
    """The log frequency ratio for every cell one homolog's assay measured."""
    dataset = dms_dataset(homolog)
    counts = dms_leaf_freq.read_counts(counts_path)
    cells, founder = dms_models.homolog_cells(homolog)

    sites = cells["aa_site"].to_numpy()
    mutant_counts = dms_leaf_freq.counts_at(counts, sites, cells["mutant_aa"])
    parent_counts = dms_leaf_freq.counts_at(counts, sites, cells["parent_aa"])
    scores = np.log(
        (mutant_counts + LEAF_PSEUDOCOUNT) / (parent_counts + LEAF_PSEUDOCOUNT)
    )

    frame = cells.assign(model=LEAF_LOGFREQ, score=scores)
    manifest = dms_models.Manifest(
        model=LEAF_LOGFREQ,
        strata=dms.STRATA,
        frame=(
            f"clade founder {dataset.clade} {lib.GENE} for the scored parent; the "
            f"tally behind it is each leaf's own path applied to the "
            f"{dms_leaf_freq.PATH_ROOT_CLADE} founder, 1-indexed, so no site mapping"
        ),
        input_sequence_sha256=dms_models.sequence_sha256(founder),
    )
    return frame[list(dms_models.PREDICTOR_COLUMNS)], manifest, founder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--homolog", required=True, choices=dms_homologs())
    parser.add_argument("--counts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    frame, manifest, sequence = leaf_logfreq_scores(args.homolog, args.counts)
    dms_models.write_scores(frame, manifest, sequence, args.output)
    print(f"wrote {args.output}: {len(frame)} cells for {args.homolog}")


if __name__ == "__main__":
    main()
