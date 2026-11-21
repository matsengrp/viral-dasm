"""The counts and DASM halves of `fig:l455f`, one clade at a time.

For every XBB clade in :data:`panel_spec.L455F_CLADES`, this measures L455F's
fitness effect on the parents that carry the clade founder's 456F and on the
parents that carry 456L, takes the shift between them, and tests that shift by
randomization. The exception is 23F, whose founder already carries 456L: it
has no founder-side pool, and is measured on its derived pool alone. It is the within-clade form of the FLip claim: L455F and F456L
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
    master_table_path,
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
    era_of_clade,
    parse_mutation,
)
from pool_counts import (
    DERIVED_POOL,
    _pool_average,
    background_label,
    clade_counts,
    pool_name,
)
from runs import run_id
from score_haplotypes import score_backgrounds
from score_run import globally_masked_sites

from dasm2.analysis.within_clade_pools import (
    DEFAULT_PERMUTATIONS,
    PoolSpec,
    join_dasm_effects,
    join_rates,
    load_annotations,
    pool_cell_table,
    tally_pools,
)
from dasm2.sequence.masking import load_censor_table

#: The split, as :func:`pool_counts.clade_specs` takes it: the derived pool
#: requires 456L, the founder pool requires whatever the clade founder carries
#: there, and no other site is named on either side.
SPLIT_DERIVED = {L455F_MARKER_SITE: L455F_MARKER_AA}


#: What a clade gets measured as. A clade whose founder carries the *founder*
#: residue at the marker site has two pools and a shift between them;
#: a clade whose founder already carries the *derived* residue has only the
#: derived one.
PAIRED, SINGLE = "paired", "single"


def clade_roles(clades: list[str]) -> dict[str, str]:
    """How each clade can be measured: as a contrast, or as one pool.

    Two founder checks, at the two sites that define the contrast:

    **At** :data:`panel_spec.L455F_SITE`. The founder must carry the residue
    L455F mutates away from, or the clade is not at risk for the mutation at
    all and no pool of it would hold a branch. A clade failing this is a
    roster error rather than a finding, so it raises.

    **At** :data:`panel_spec.L455F_MARKER_SITE`. If the founder already
    carries the derived residue -- 23F (EG.5.1) is the one XBB subclade where
    it does -- then :func:`pool_counts.clade_specs` would build two pools with
    identical predicates: the same branches twice, a shift of exactly zero,
    and nothing in the output saying so.
    :mod:`exploratory.l455f_456.shift_per_clade` guards only the first site,
    which is why 23F is absent from its roster by hand.

    Such a clade is measured on its derived pool alone rather than dropped.
    It has no founder-side pool *by construction*, which is a different thing
    from a pool that was measured and found thin, and the figure draws it as
    a named tick with no point. It is the clade that makes the figure's
    point: a lineage whose founder already carries 456L, where L455F is
    strongly favoured from the start.

    The founder-side pool is deliberately **not** reconstructed from the
    branches that reverted to 456F. There are a few hundred of them against
    tens of thousands on the derived side, and they are revertants rather
    than the clade's own background -- a different object from the founder
    pools of the other clades, drawn in the same position and read as the
    same thing.

    Returns:
        ``{clade: PAIRED | SINGLE}``, in the order given.
    """
    founder_residue, site, _ = parse_mutation(L455F_MUTATION)
    if site != L455F_SITE:
        raise ValueError(
            f"{L455F_MUTATION} is at site {site} but `L455F_SITE` is "
            f"{L455F_SITE}. The two spellings of the figure's site disagree."
        )
    roles = {}
    for clade in clades:
        founder = founder_aa_sequence(clade, GENE)
        if founder[site - 1] != founder_residue:
            raise ValueError(
                f"{clade}'s founder carries {founder[site - 1]!r} at site "
                f"{site}, not {founder_residue!r}: it is not at risk for "
                f"{L455F_MUTATION}, so no pool of it would hold a branch. The "
                f"roster names a clade this mutation cannot be measured in."
            )
        roles[clade] = (
            SINGLE if founder[L455F_MARKER_SITE - 1] == L455F_MARKER_AA else PAIRED
        )
    return roles


def derived_pool_only(
    clade: str,
    *,
    pcp_path: str,
    annotations_path,
    haplotype_scores: pd.DataFrame,
    chunk_rows: int,
    max_chunks: int | None,
) -> tuple[pd.DataFrame, dict]:
    """One clade's derived pool, for a clade that has no founder-side one.

    The same quantities :func:`pool_counts.clade_counts` computes for a
    contrast, for one pool: the counts effect over the branches whose parent
    carries the derived residue, and the DASM pool average over the
    haplotypes those branches hold. No shift and no randomization test, which
    both need two pools.

    :func:`pool_cell_table` is written for a contrast and takes two specs, so
    it is handed the same spec twice and the duplicate half discarded. That
    is cheaper and plainer than a second code path through
    :func:`aggregate_to_aa_cells`, and the tallies are keyed by pool name --
    one name, tallied once -- so nothing is counted twice.
    """
    founder_aa = founder_aa_sequence(clade, GENE)
    _, site, mutant = parse_mutation(L455F_MUTATION)
    annotations = load_annotations(annotations_path, [clade], GENE)
    era = era_of_clade(clade)
    annotations = join_rates(annotations, pd.read_csv(master_table_path(era)))
    print(f"{clade}: era {era}, {len(annotations):,} annotated rows", flush=True)

    spec = PoolSpec(
        name=pool_name(L455F_MUTATION, DERIVED_POOL),
        at_risk={site: founder_aa[site - 1]},
        marker=SPLIT_DERIVED,
        hold={},
    )
    print(f"{clade}: streaming the corpus for 1 pool ...", flush=True)
    tallies = tally_pools(
        pcp_path,
        [spec],
        annotations,
        clade=clade,
        chunk_rows=chunk_rows,
        max_chunks=max_chunks,
    )
    cells = pool_cell_table(
        tallies, annotations, cluster=clade, gene=GENE, pool0=spec, pool1=spec
    )
    cells = cells[cells["pool"] == 1].copy()

    average = _pool_average(
        tallies,
        spec.name,
        haplotype_scores,
        clade=clade,
        founder_aa=founder_aa,
        aa_site=site,
        mutant_aa=mutant,
    )
    cells = join_dasm_effects(
        cells,
        {spec.name: average.log_sf},
        # As the paired clades are measured; see `clade_counts(spread=True)`.
        spread_by_pool={spec.name: average.spread},
    )
    cells.insert(0, "candidate", L455F_MUTATION)
    cells["background"] = background_label(SPLIT_DERIVED)

    focal = cells[(cells["aa_site"] == site) & (cells["mutant_aa"] == mutant)]
    if len(focal) != 1:
        raise ValueError(
            f"{clade}/{L455F_MUTATION}: the focal cell appears {len(focal)} "
            f"times in the pool's cell table. It was dropped by the "
            f"exclude/mask restriction."
        )
    summary = {
        "clade": clade,
        "backgrounds": {DERIVED_POOL: background_label(SPLIT_DERIVED)},
        "founder_pool": (
            f"none: the clade founder already carries "
            f"{L455F_MARKER_AA}{L455F_MARKER_SITE}"
        ),
        "pools": {
            DERIVED_POOL: {
                "label": background_label(SPLIT_DERIVED),
                "n_pcps": int(tallies.n_pcps[spec.name]),
                "synonymous": int(tallies.synonymous_observed[spec.name]),
                "n_haplotypes": average.n_haplotypes,
                "counts_effect": float(focal.iloc[0]["counts_effect"]),
                "dasm_effect": average.log_sf,
            }
        },
    }
    print(
        f"  {L455F_MUTATION:>7}: one pool, counts "
        f"{float(focal.iloc[0]['counts_effect']):+.2f}, DASM "
        f"{average.log_sf:+.2f}",
        flush=True,
    )
    return cells, summary


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

    roles = clade_roles(list(args.clades))
    measured = list(roles)
    for clade, role in roles.items():
        print(f"{clade}: {role}", flush=True)

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
    for clade, role in roles.items():
        print(f"\n=== {clade} ({role}) ===", flush=True)
        if role == SINGLE:
            clade_cells, clade_summary = derived_pool_only(
                clade,
                pcp_path=args.pcp,
                annotations_path=annotations_path,
                haplotype_scores=scores[scores["clade"] == clade],
                chunk_rows=PCP_CHUNK_ROWS,
                max_chunks=args.max_chunks,
            )
            cells.append(clade_cells)
            summary[clade] = {L455F_MUTATION: clade_summary}
            continue
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
            # The whiskers the panel draws: how far the model's estimate moves
            # between the parents of one pool, at the weight-weighted 5th and
            # 95th percentiles.
            spread=True,
        )
        # Both frames already carry the clade as `cluster` -- `pool_cell_table`
        # is handed it and `shift_table` keys on it -- so nothing is inserted
        # here. That column is what the panel groups its sub-panels on, and it
        # is constant within each call: the concatenation below is what makes
        # it vary.
        cells.append(clade_cells)
        shifts.append(clade_shifts)
        summary[clade] = clade_summary

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
