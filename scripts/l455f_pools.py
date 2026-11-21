"""The counts and DASM halves of `fig:l455f`'s panel B, one clade at a time.

For every XBB clade in :data:`panel_spec.L455F_CLADES`, this measures L455F's
fitness effect on the parents that carry the clade founder's 456F and on the
parents that carry 456L, takes the shift between them, and tests that shift by
randomization. It is the within-clade form of the FLip claim: L455F and F456L
are individually costly to the XBB spike and jointly beneficial, so L455F's
effect should be much higher on a 456L background than on a 456F one.

**Why this is its own rule and not `rule pool_counts`.** Everything downstream
of the pool definition is shared -- :func:`pool_counts.clade_counts` is what
that rule calls, so the two estimates here are computed exactly as every other
`fig:within_clade_shifts` cell is. What differs is the *split*, and only the
split:

* `rule pool_counts` takes each clade's two backgrounds from
  ``clade_backgrounds.csv``, which names them from branch frequencies alone.
  Here the split is specified by hand as site 456 and nothing else.
* Its roster is :data:`panel_spec.CANDIDATE_CLADES` and its candidates are
  ``data/shift_candidates.csv``. Neither carries these clades or this
  mutation, and adding them there would add rows to `fig:within_clade_shifts`.

**The hand-specified split is a deviation and is reported as one.** For 23A
the frequency-derived marker *is* F456L, but it arrives with L455F itself at
59% of marker branches as a co-traveller -- which puts the mutation under test
inside the background that defines the contrast, and leaves the cell
unscoreable. For 22F the frequency-derived marker is not site 456 at all but
S486P. For 23B it is F456L with no co-travellers, so there the hand
specification confirms rather than overrides. Measured, not assumed: see the
PR body and ``_ignore/step_zero_clade_backgrounds.csv``.

**No training, and no card.** This scores haplotypes under the published
checkpoint on CPU, which is inference: the forward passes cost minutes and a
card would save a few of them. `rule score_haplotypes` makes the same choice
for the same reason.

GISAID: reads GISAID-derived PCPs and mat-pcp count tables. The per-cell
tables it writes stay under the artifact root; the tracked panel CSV names a
background by its residues and carries no sequence.
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
# yet. This script writes a Parquet, so the allocator is not academic.
import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import argparse
import json

import pandas as pd
from clade_haplotypes import clade_haplotypes
from founders import founder_aa_sequence
from panel_spec import (
    ARM,
    FIGURE_MODEL,
    L455F_CLADES,
    L455F_HAPLOTYPE_TABLE,
    L455F_MARKER_AA,
    L455F_MARKER_SITE,
    L455F_MUTATION,
    L455F_POOL_CELLS,
    L455F_POOL_SHIFTS,
    L455F_POOL_SUMMARY,
    L455F_SITE,
    PCP_CHUNK_ROWS,
    parse_mutation,
)
from pool_counts import clade_counts
from runs import run_id
from score_haplotypes import score_backgrounds
from score_run import globally_masked_sites

from dasm2.analysis.within_clade_pools import DEFAULT_PERMUTATIONS
from dasm2.sequence.masking import load_censor_table

#: The split, as :func:`pool_counts.clade_specs` takes it: the derived pool
#: requires 456L, the founder pool requires whatever the clade founder carries
#: there, and no other site is named on either side.
SPLIT_DERIVED = {L455F_MARKER_SITE: L455F_MARKER_AA}


def measurable_clades(clades: list[str]) -> tuple[list[str], dict[str, str]]:
    """Which clades have two distinct pools, and why the others do not.

    Two founder checks, at the two sites that define the contrast:

    **At** :data:`panel_spec.L455F_SITE`. The founder must carry the residue
    L455F mutates away from, or the clade is not at risk for the mutation at
    all and both its pools would be empty. A clade failing this is a roster
    error rather than a finding, so it raises.

    **At** :data:`panel_spec.L455F_MARKER_SITE`. The founder must *not*
    already carry the derived residue. 23F (EG.5.1) does: it is the one XBB
    subclade whose founder carries 456L, so ``split_derived`` would equal the
    founder's own residues there and :func:`pool_counts.clade_specs` would
    build two pools with identical predicates -- the same branches twice, a
    shift of exactly zero, and nothing in the output saying so.
    :mod:`exploratory.l455f_456.shift_per_clade` guards only the first site,
    which is why 23F is absent from its roster by hand; here it is refused by
    the check, with the reason recorded.

    Returns:
        ``(measured, refused)`` -- the clades to build pools for, and a reason
        per clade for those left out.
    """
    founder_residue, site, _ = parse_mutation(L455F_MUTATION)
    if site != L455F_SITE:
        raise ValueError(
            f"{L455F_MUTATION} is at site {site} but `L455F_SITE` is "
            f"{L455F_SITE}. The two spellings of the figure's site disagree."
        )
    measured, refused = [], {}
    for clade in clades:
        founder = founder_aa_sequence(clade, GENE)
        if founder[site - 1] != founder_residue:
            raise ValueError(
                f"{clade}'s founder carries {founder[site - 1]!r} at site "
                f"{site}, not {founder_residue!r}: it is not at risk for "
                f"{L455F_MUTATION}, so neither pool would hold a branch. The "
                f"roster names a clade this mutation cannot be measured in."
            )
        if founder[L455F_MARKER_SITE - 1] == L455F_MARKER_AA:
            refused[clade] = (
                f"founder already carries {L455F_MARKER_AA}{L455F_MARKER_SITE}, "
                f"so the derived pool's predicate matches the founder pool's "
                f"and the two would be the same branches"
            )
            continue
        measured.append(clade)
    if not measured:
        raise ValueError(
            f"no clade of {clades} can be measured: {refused}. The panel would "
            f"be empty."
        )
    return measured, refused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcp", required=True, help="the PCP corpus to read")
    parser.add_argument("--clades", nargs="+", default=list(L455F_CLADES))
    parser.add_argument("--cells", default=str(L455F_POOL_CELLS), type=Path)
    parser.add_argument("--shifts", default=str(L455F_POOL_SHIFTS), type=Path)
    parser.add_argument("--summary", default=str(L455F_POOL_SUMMARY), type=Path)
    parser.add_argument(
        "--haplotype-scores", default=str(L455F_HAPLOTYPE_TABLE), type=Path
    )
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "where the forward passes run. CPU by default and in the rule: "
            "the saving a card buys here is a few minutes, and holding one a "
            "training run could use for that is not a trade this pipeline "
            "makes."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help=(
            "stop each corpus pass after this many chunks. Wiring check only: "
            "every count it produces is a partial tally."
        ),
    )
    args = parser.parse_args()

    measured, refused = measurable_clades(list(args.clades))
    for clade, reason in refused.items():
        print(f"{clade}: refused -- {reason}", flush=True)
    print(f"measuring {L455F_MUTATION} in {measured}", flush=True)

    print("\nenumerating haplotypes ...", flush=True)
    haplotypes = clade_haplotypes(measured, args.pcp, chunk_rows=PCP_CHUNK_ROWS)
    print(haplotypes.groupby("clade").size().to_string(), flush=True)

    _, site, mutant = parse_mutation(L455F_MUTATION)
    run = run_id(FIGURE_MODEL, ARM)
    checkpoint = select_checkpoint(Path(ARTIFACT_ROOT) / "runs" / run)
    print(
        f"\nscoring {L455F_MUTATION} on {len(haplotypes):,} haplotypes, "
        f"{checkpoint.name}, {args.device}",
        flush=True,
    )
    scores = score_backgrounds(
        haplotypes,
        {clade: ((site, mutant),) for clade in measured},
        checkpoint=checkpoint,
        gene=GENE,
        run_id=run,
        model=FIGURE_MODEL,
        corpus=ARM,
        masked=globally_masked_sites(),
        censor_table=load_censor_table(
            mat_pcp_path("censored_mutations_file"),
            mat_pcp_path("censored_clades_file"),
        ),
        device=args.device,
        progress_every=1000,
    )
    args.haplotype_scores.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(args.haplotype_scores, index=False)
    print(f"wrote {len(scores):,} rows to {args.haplotype_scores}", flush=True)

    annotations_path = mat_pcp_path("site_annotations_file")
    cells, shifts, summary = [], [], {}
    for clade in measured:
        print(f"\n=== {clade} ===", flush=True)
        clade_cells, clade_shifts, clade_summary = clade_counts(
            clade,
            [L455F_MUTATION],
            pcp_path=args.pcp,
            annotations_path=annotations_path,
            haplotype_scores=scores[scores["clade"] == clade],
            split_derived=SPLIT_DERIVED,
            permutations=args.permutations,
            chunk_rows=PCP_CHUNK_ROWS,
            max_chunks=args.max_chunks,
        )
        # Both frames already carry the clade as `cluster` -- `pool_cell_table`
        # is handed it and `shift_table` keys on it -- so nothing is inserted
        # here. That column is what the panel groups its sub-panels on, and it
        # is constant within each call: the concatenation below is what makes
        # it vary.
        cells.append(clade_cells)
        shifts.append(clade_shifts)
        summary[clade] = clade_summary
    summary["__refused"] = refused

    for path, frame in (
        (args.cells, pd.concat(cells, ignore_index=True)),
        (args.shifts, pd.concat(shifts, ignore_index=True)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        print(f"wrote {len(frame):,} rows to {path}")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
