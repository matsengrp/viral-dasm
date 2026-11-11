"""BLOSUM62 as a predictor of the spike DMS: `fig:dms` panel C's substitution floor.

How much of a deep mutational scan does a general-purpose substitution matrix
already explain? It knows nothing about SARS-CoV-2, nothing about the site and
nothing about the clade, so whatever it scores is what a reader should subtract
from any model's correlation before being impressed by it.

The table is already a log-odds, so nothing is transformed -- that is what puts
it on the same axis as the DASM's log selection factor and the leaf frequency's
log ratio.
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
import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices

from dasm2.sequence.constants import AA_STR_SORTED, AA_STR_TO_IDX

#: This predictor's id, as `results/panels/dms/` carries it and
#: `scripts/panel_charts.py` maps to a label.
BLOSUM62 = "blosum62"


def blosum62_log_odds() -> np.ndarray:
    """BLOSUM62's 20 x 20 log-odds, rows and columns in ``AA_STR_SORTED``.

    Biopython's table also carries the ambiguity codes and the stop; this keeps
    the twenty standard residues, in the order every selection-factor row in
    this pipeline is laid out in.

    Taken from ``dasm2-experiments`` ``dasm2/analysis/pam_baseline.py``. It is
    here and not in ``dasm2/`` on purpose: that tree is a one-way fork whose
    value is that a diff against its source commit is meaningful, so a module at
    a name no upstream commit uses does not belong in it -- and ``ruff`` excludes
    it, so a file there would go unchecked.
    """
    blosum = substitution_matrices.load("BLOSUM62")
    return np.array(
        [[blosum[a, b] for b in AA_STR_SORTED] for a in AA_STR_SORTED],
        dtype=np.float64,
    )


def blosum62_scores(homolog: str) -> tuple[pd.DataFrame, dms_models.Manifest, str]:
    """BLOSUM62's log-odds for every cell one homolog's assay measured.

    The score is ``B[parent_aa, mutant_aa]`` at the clade founder's residue. It
    is already a log-odds, so nothing is transformed: that is what puts it on the
    same axis as the DASM's log selection factor.
    """
    cells, founder = dms_models.homolog_cells(homolog)
    matrix = blosum62_log_odds()
    scores = [
        matrix[AA_STR_TO_IDX[parent], AA_STR_TO_IDX[mutant]]
        for parent, mutant in zip(cells["parent_aa"], cells["mutant_aa"])
    ]
    frame = cells.assign(model=BLOSUM62, score=scores)
    manifest = dms_models.Manifest(
        model=BLOSUM62,
        strata=dms.STRATA,
        frame=(
            f"clade founder {dms_dataset(homolog).clade} {lib.GENE}, from "
            f"mat-pcp's own clade_founders FASTA; 1-indexed, so no site mapping"
        ),
        input_sequence_sha256=dms_models.sequence_sha256(founder),
    )
    return frame[list(dms_models.PREDICTOR_COLUMNS)], manifest, founder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--homolog", required=True, choices=dms_homologs())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    frame, manifest, sequence = blosum62_scores(args.homolog)
    dms_models.write_scores(frame, manifest, sequence, args.output)
    print(f"wrote {args.output}: {len(frame)} cells for {args.homolog}")


if __name__ == "__main__":
    main()
