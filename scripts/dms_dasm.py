"""The DASM as a predictor of the spike DMS, through the same join as the rest.

The model the paper is about, scored against the four assays on the same cells
and through the same contract as BLOSUM62, the leaf log-frequency ratio and
antiGen. What it contributes that they cannot is the point of the comparison;
what this module contributes is only that it arrives in the same shape.

**No inference here.** ``rule score`` already did the forward passes, and
``selection_factors.parquet`` holds the result: one row per
``(clade, model, arm, aa_site, mutant_aa)``, left-joined from the model's own
20 x L grid, so a selection factor exists for every cell an assay can measure --
the multi-nucleotide ones included. This module is a reader and a rename.

**Why a reader rather than ``dms.dms_frame``.** That function is `fig:dms`'s and
goes through ``panels.base_frames``, whose ``dropna`` on the counts metric is
what restricts those panels to the single-nucleotide stratum: mat-pcp estimates
only what the tree could observe. Reusing it here would empty the
multi-nucleotide half, which is the half this predictor exists to be compared on.
The population is instead ``dms_models.homolog_cells``, shared with every other
predictor, and the training mask is applied there.

**The score is ``log f``, and the figure has to say so.** ``scripts/dms_antigen``
records that antiGen's column is ``log p`` while this one is ``log f``, and that
nothing converts between them "because no conversion exists that does not assume
a mutation model". The two are predictors of the same measurement on different
scales, so they are never put on a shared axis.

``scripts/dms_prob`` is the third reading: the DASM's *own* ``log p``, which is
this module's ``log f`` weighted by how often the neutral process offers the
mutation. That one does convert from this one -- it is built by exponentiating
:func:`selection_factor_grid`'s rows -- but only on the single-nucleotide
stratum, where the neutral rate is non-zero. So there are two ``log p`` columns
in the table and they are not the same quantity: ``antigen``'s is another
model's, ``dasm_prob``'s is this model's.
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
import dms_models
import pandas as pd
import panels
from panel_spec import FIGURE_MODEL

#: This predictor's id, as `results/panels/dms/` carries it and
#: `scripts/panel_spec.py` lists in :data:`panel_spec.DMS_PREDICTORS`.
DASM = "dasm"

#: The scored table's column this predictor is. Named rather than inlined
#: because the rename to ``score`` is the only transformation in this module,
#: and a reader should be able to see what was renamed.
SCORE_COLUMN = "log_selection_factor"

#: The slice of :data:`panel_spec.TABLE_COLUMNS` this reader needs, named apart
#: from it so a grep for ``TABLE_COLUMNS`` does not conflate the two. The table
#: is ~1.4M rows per run and the filter is pushed into Parquet, so asking for
#: seven columns rather than fifteen is most of what makes this cheap.
#:
#: ``parent_aa`` is here although this module never reads it:
#: ``panels.read_scored_table`` copies it to ``clade_founder_aa`` on the way
#: out, and raises on a frame that does not carry it. The founder residue this
#: module does use comes from ``homolog_cells``, which reads the FASTA.
SCORE_TABLE_COLUMNS: tuple[str, ...] = (
    "clade",
    "model",
    "arm",
    "aa_site",
    "parent_aa",
    "mutant_aa",
    SCORE_COLUMN,
)


def selection_factor_grid(clade: str, table_path: str | Path) -> pd.DataFrame:
    """:data:`FIGURE_MODEL`'s whole ``[L, 20]`` grid for one clade, long.

    One function rather than two spellings of the same read, because
    `scripts/dms_prob.py` needs these rows too: it exponentiates this grid to
    recover the linear selection factors whichmut's probability is defined
    against. A second copy would also be a second copy of
    :data:`SCORE_TABLE_COLUMNS`, carrying the ``parent_aa`` requirement
    documented above as a trap for whoever trimmed it.

    The grid comes back whole -- every site and every residue, the cells no
    assay measured included -- because a partition function sums over the
    whole sequence. Filtering to measured cells is a caller's step.
    """
    return panels.read_scored_table(
        table_path,
        models=[FIGURE_MODEL],
        clades=[clade],
        columns=SCORE_TABLE_COLUMNS,
    )


def dasm_scores(
    homolog: str, table_path: str | Path
) -> tuple[pd.DataFrame, dms_models.Manifest, str]:
    """One homolog's cells, each carrying the DASM's log selection factor.

    Raises:
        ValueError: If the scored table does not cover every measured cell. A
            missing cell would otherwise arrive as a NaN score, which
            ``dms_models.validate_scores`` refuses and ``dms_summary``'s
            coverage check would report as a short ``n`` -- but both would name
            the symptom rather than this cause.
    """
    cells, founder = dms_models.homolog_cells(homolog)
    clade = dms_dataset(homolog).clade
    table = selection_factor_grid(clade, table_path)

    scores = table[["aa_site", "mutant_aa", SCORE_COLUMN]].drop_duplicates(
        subset=list(dms_models.JOIN_KEY)
    )
    merged = cells.merge(
        scores, on=list(dms_models.JOIN_KEY), how="left", validate="one_to_one"
    )

    missing = merged[merged[SCORE_COLUMN].isna()]
    if not missing.empty:
        sample = sorted(
            zip(missing["aa_site"], missing["mutant_aa"]),
        )[:5]
        raise ValueError(
            f"{homolog}: {len(missing)} of {len(merged)} measured cells have no "
            f"{FIGURE_MODEL} selection factor for clade {clade} in "
            f"{table_path}. The scored table is left-joined from the model's "
            f"own 20 x L grid, so every cell should be present: a gap means the "
            f"table was built for a different clade roster or gene. First few: "
            f"{sample}."
        )

    frame = merged.assign(model=DASM, score=merged[SCORE_COLUMN])
    manifest = dms_models.Manifest(
        model=DASM,
        strata=dms.STRATA,
        frame=(
            f"clade founder {clade} {lib.GENE}, from mat-pcp's own "
            f"clade_founders FASTA; 1-indexed, so no site mapping"
        ),
        input_sequence_sha256=dms_models.sequence_sha256(founder),
    )
    return frame[list(dms_models.PREDICTOR_COLUMNS)], manifest, founder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--homolog", required=True, choices=dms_homologs())
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    frame, manifest, sequence = dasm_scores(args.homolog, args.table)
    dms_models.write_scores(frame, manifest, sequence, args.output)
    print(f"wrote {args.output}: {len(frame)} cells for {args.homolog}")


if __name__ == "__main__":
    main()
