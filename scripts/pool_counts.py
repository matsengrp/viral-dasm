"""The counts half of `fig:within_clade_shifts`' panels B, D and F.

Each clade of the figure splits into two genetic backgrounds, named by
``rule clade_backgrounds`` from branch frequencies alone. For every candidate
mutation this script measures that mutation's effect in each background by
counts, takes the shift between them, and tests the shift by randomization.

**No checkpoint, no GPU, no model.** The DASM side is a separate rule, joined
downstream on ``(clade, background, aa_site, mutant_aa)``. Keeping the two
apart is what lets the counts side be re-run while a measurement is being
checked, and it is what makes the comparison a test rather than a
restatement: nothing the model produces reaches the split, the pools or the
p-value.

**Two pools, not three.** The source discovered a marker first and its
co-travellers afterwards, so "how specifically is the alternative background
named" was a live question and it built a third, marker-only pool.
``clade_backgrounds.csv`` answers it up front -- the marker and its
co-travellers are one set of sites, and the two backgrounds differ at all of
them -- so there is one contrast per candidate and no pool ordering to
reason about.

**Both sides are pinned at every split site**, one at the founder residues
and one at the derived residues. Leaving the founder side free at those
sites is what let 70% of one candidate's founder pool carry the very
substitutions the opposite pool was named for, which pulls the two pools
together and shrinks the measured shift.

**The fixed substitutions are not pinned.** A substitution carried by more
than 95% of the clade's branch mass is in both pools whether or not it is
required, so pinning it would cost both pools a twentieth of their branches
and could not change the contrast.

GISAID: reads GISAID-derived PCPs and mat-pcp count tables. The per-branch
and per-cell tables it writes stay under the artifact root; the tracked
per-panel CSVs name a background by its residues and carry no sequence.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install.
from lib import (
    GENE,
    assert_dasm2_is_this_repo,
    era_of_clade,
    master_table_path,
    mat_pcp_path,
)

# Then `dasm2`, before pandas or pyarrow: `dasm2/arrow_pool.py` moves Arrow off
# the bundled mimalloc, and can only do so if pyarrow has not been imported
# yet. This script writes a Parquet, so the allocator is not academic.
import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import argparse
import json

import pandas as pd
from clade_haplotypes import FOUNDER_LABEL
from founders import founder_aa_sequence
from panel_spec import (
    BACKGROUND_CSV,
    BACKGROUND_SPLIT_ROLES,
    CANDIDATE_CLADES,
    HAPLOTYPE_TABLE,
    PCP_CHUNK_ROWS,
    POOL_CELLS,
    POOL_SHIFTS,
    POOL_SUMMARY,
    figure_candidates,
    off_figure_candidates,
    parse_mutation,
)

from dasm2.analysis.within_clade_pools import (
    DEFAULT_PERMUTATIONS,
    PoolSpec,
    branch_fraction_table,
    join_dasm_effects,
    join_rates,
    load_annotations,
    permutation_shift_test,
    pool_average_log_sf,
    pool_cell_table,
    shift_table,
    tally_pcps,
    tally_pools,
)
from dasm2.sequence.conversion import translate_sequences

#: Haldane-Anscombe correction on the focal counts inside the randomization
#: test. Named rather than left as the source's bare ``0.5`` default: it is a
#: third role for that number, distinct from
#: ``within_clade_pools.ZERO_COUNT_PSEUDOCOUNT`` (which moves a zero-count
#: cell's point estimate) and from ``VARIANCE_STABILIZER`` (which never
#: enters a point estimate at all).
PERMUTATION_PSEUDOCOUNT = 0.5

#: Suffixes the two pools of a contrast and its denominator are named with.
#: The pool name carries the candidate, because one streaming pass serves
#: every candidate of a clade and a name shared between two of them would
#: merge their branches.
FOUNDER_POOL = "founder"
DERIVED_POOL = "derived"
ELIGIBLE_POOL = "__eligible"


def pool_name(mutation: str, role: str) -> str:
    """``"D574V|derived"``: one pool of one candidate's contrast."""
    return f"{mutation}|{role}"


def background_label(sites: dict[int, str]) -> str:
    """``"R346, K356, F490"``: what a background carries, site-ordered.

    The residues themselves, in the one-letter-then-site form the literature
    names a position in, rather than the substitutions that reach them. Both
    pools of a contrast are defined by their residues at the same sites, so
    naming them this way makes the two x ticks a like-for-like pair -- where
    naming them by substitution leaves the founder side, which substitutes
    nothing, with no name at all beyond "none".
    """
    return ", ".join(f"{residue}{site}" for site, residue in sorted(sites.items()))


def clade_specs(
    clade: str,
    candidates: list[str],
    split_derived: dict[int, str],
    founder_aa: str,
) -> tuple[dict[str, tuple[PoolSpec, PoolSpec]], dict[str, PoolSpec]]:
    """Every pool of one clade: two per candidate, plus each one's denominator.

    Returns ``({mutation: (founder_pool, derived_pool)}, {mutation:
    eligible_pool})``. The eligible pool requires the at-risk residue and
    nothing else, so it holds every branch whose parent can still make the
    mutation whatever its background -- the denominator the contrast's share
    of the clade is measured against, which no sum over the two pools can
    supply.
    """
    split_founder = {site: founder_aa[site - 1] for site in split_derived}
    contrasts, eligible = {}, {}
    for mutation in candidates:
        founder_residue, site, _ = parse_mutation(mutation)
        if founder_aa[site - 1] != founder_residue:
            raise ValueError(
                f"{clade}'s founder carries {founder_aa[site - 1]!r} at site "
                f"{site}, not {founder_residue!r} as {mutation} claims. The "
                f"candidate is named against the wrong founder, so both pools "
                f"would be defined on the wrong residue."
            )
        if site in split_derived:
            raise ValueError(
                f"{clade}/{mutation} sits at site {site}, which is one of the "
                f"sites that define the clade's two backgrounds. Its effect "
                f"cannot be contrasted between them: the at-risk requirement "
                f"and the background requirement would disagree on one side."
            )
        at_risk = {site: founder_residue}
        contrasts[mutation] = (
            PoolSpec(
                name=pool_name(mutation, FOUNDER_POOL),
                at_risk=at_risk,
                marker=split_founder,
                hold={},
            ),
            PoolSpec(
                name=pool_name(mutation, DERIVED_POOL),
                at_risk=at_risk,
                marker=split_derived,
                hold={},
            ),
        )
        eligible[mutation] = PoolSpec(
            name=pool_name(mutation, ELIGIBLE_POOL),
            at_risk=at_risk,
            marker={},
            hold={},
        )
    return contrasts, eligible


def clade_counts(
    clade: str,
    candidates: list[str],
    *,
    pcp_path: str | Path,
    annotations_path: Path,
    haplotype_scores: pd.DataFrame,
    split_derived: dict[int, str],
    permutations: int,
    chunk_rows: int,
    max_chunks: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """One clade's cell table, shift table and summary.

    Two streaming passes over the corpus, not two per candidate:
    :func:`tally_pools` serves every spec at once and :func:`tally_pcps` every
    candidate at once. The second pass exists because the point estimate sums
    the branches away and the randomization test has to keep them apart.
    """
    founder_aa = founder_aa_sequence(clade, GENE)
    annotations = load_annotations(annotations_path, [clade], GENE)
    era = era_of_clade(clade)
    annotations = join_rates(annotations, pd.read_csv(master_table_path(era)))
    print(f"{clade}: era {era}, {len(annotations):,} annotated rows", flush=True)

    contrasts, eligible = clade_specs(clade, candidates, split_derived, founder_aa)
    specs = [spec for pair in contrasts.values() for spec in pair] + list(
        eligible.values()
    )
    print(f"{clade}: streaming the corpus for {len(specs)} pools ...", flush=True)
    tallies = tally_pools(
        pcp_path,
        specs,
        annotations,
        clade=clade,
        chunk_rows=chunk_rows,
        max_chunks=max_chunks,
    )

    print(f"{clade}: streaming again for the randomization test ...", flush=True)
    branches = tally_pcps(
        pcp_path,
        {mutation: pair for mutation, pair in contrasts.items()},
        annotations,
        clade=clade,
        focal_by_candidate={
            mutation: (parse_mutation(mutation)[1], parse_mutation(mutation)[2])
            for mutation in candidates
        },
        chunk_rows=chunk_rows,
        max_chunks=max_chunks,
    )

    labels = {
        FOUNDER_POOL: background_label(
            {site: founder_aa[site - 1] for site in split_derived}
        ),
        DERIVED_POOL: background_label(split_derived),
    }

    cell_frames, shift_frames, summaries = [], [], {}
    for mutation in candidates:
        pool0, pool1 = contrasts[mutation]
        _, site, mutant = parse_mutation(mutation)
        cells = pool_cell_table(
            tallies, annotations, cluster=clade, gene=GENE, pool0=pool0, pool1=pool1
        )

        averages = {
            spec.name: _pool_average(
                tallies,
                spec.name,
                haplotype_scores,
                clade=clade,
                founder_aa=founder_aa,
                aa_site=site,
                mutant_aa=mutant,
            )
            for spec in (pool0, pool1)
        }
        cells = join_dasm_effects(
            cells, {name: value.log_sf for name, value in averages.items()}
        )
        cells.insert(0, "candidate", mutation)
        cells["background"] = cells["pool_name"].map(
            {pool0.name: labels[FOUNDER_POOL], pool1.name: labels[DERIVED_POOL]}
        )

        shifts = shift_table(cells)
        shifts.insert(0, "candidate", mutation)

        test = permutation_shift_test(
            branches[branches["candidate"] == mutation],
            pool0=pool0.name,
            pool1=pool1.name,
            n_permutations=permutations,
            pseudocount=PERMUTATION_PSEUDOCOUNT,
            seed=0,
        )
        shifts["p_value"] = test.p_value
        shifts["p_at_floor"] = test.n_extreme == 0

        fractions = branch_fraction_table(
            tallies,
            founder_pool=pool0.name,
            marker_pools=[pool1.name],
            eligible_pool=eligible[mutation].name,
        )
        focal = shifts[(shifts["aa_site"] == site) & (shifts["mutant_aa"] == mutant)]
        if len(focal) != 1:
            raise ValueError(
                f"{clade}/{mutation}: the focal cell appears {len(focal)} times "
                f"in the shift table. It was dropped by the exclude/mask "
                f"restriction, or the two pools disagree about it."
            )
        summaries[mutation] = {
            "clade": clade,
            "backgrounds": labels,
            "counts_shift": float(focal.iloc[0]["counts_shift"]),
            "dasm_shift": float(focal.iloc[0]["dasm_shift"]),
            "p_value": test.p_value,
            "p_at_floor": bool(test.n_extreme == 0),
            "n_permutations": test.n_permutations,
            "pools": {
                role: {
                    "label": labels[role],
                    "n_pcps": int(tallies.n_pcps[spec.name]),
                    "synonymous": int(tallies.synonymous_observed[spec.name]),
                    "n_haplotypes": averages[spec.name].n_haplotypes,
                    "counts_effect": float(
                        cells[
                            (cells["pool_name"] == spec.name)
                            & (cells["aa_site"] == site)
                            & (cells["mutant_aa"] == mutant)
                        ].iloc[0]["counts_effect"]
                    ),
                    "dasm_effect": averages[spec.name].log_sf,
                }
                for role, spec in ((FOUNDER_POOL, pool0), (DERIVED_POOL, pool1))
            },
            "branch_fractions": fractions.to_dict(orient="records"),
        }
        cell_frames.append(cells)
        shift_frames.append(shifts)
        head = summaries[mutation]
        print(
            f"  {mutation:>7}: "
            f"counts {head['pools'][FOUNDER_POOL]['counts_effect']:+.2f} -> "
            f"{head['pools'][DERIVED_POOL]['counts_effect']:+.2f} "
            f"(shift {head['counts_shift']:+.2f}), "
            f"DASM shift {head['dasm_shift']:+.2f}, "
            f"p {'<' if head['p_at_floor'] else '='} {test.p_value:.2g}",
            flush=True,
        )

    return (
        pd.concat(cell_frames, ignore_index=True),
        pd.concat(shift_frames, ignore_index=True),
        summaries,
    )


def _pool_average(
    tallies,
    pool: str,
    haplotype_scores: pd.DataFrame,
    *,
    clade: str,
    founder_aa: str,
    aa_site: int,
    mutant_aa: str,
):
    """The DASM effect for one cell on one pool: its haplotypes' weighted mean.

    The weight is the haplotype's synonymous event count, which is
    proportional to its expected count, so this is the quantity that is
    apples-to-apples with the counts effect -- the counts side divides the
    pool's summed events by its summed expected count.

    ``clade_haplotypes.py`` writes the founder-identical haplotype under
    :data:`FOUNDER_LABEL` because an empty ``background`` is the founder
    table's convention for "this row is a founder". Here the label is turned
    back into the empty string that :func:`substitution_key` produces, which
    is the key the weights are built on.
    """
    cell = haplotype_scores[
        (haplotype_scores["clade"] == clade)
        & (haplotype_scores["aa_site"] == aa_site)
        & (haplotype_scores["mutant_aa"] == mutant_aa)
    ]
    if cell.empty:
        raise ValueError(
            f"no DASM score for {clade} site {aa_site} -> {mutant_aa}. The "
            f"scored haplotype table and the candidate list disagree about "
            f"which cells were scored."
        )
    log_sf_by_haplotype = {
        ("" if background == FOUNDER_LABEL else background): float(value)
        for background, value in zip(cell["background"], cell["log_selection_factor"])
    }
    by_parent = tallies.synonymous_by_parent[pool]
    parents = list(by_parent)
    return pool_average_log_sf(
        by_parent,
        dict(zip(parents, translate_sequences(parents))),
        log_sf_by_haplotype,
        founder_aa=founder_aa,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcp", required=True, help="the PCP corpus to read")
    parser.add_argument("--backgrounds", default=str(BACKGROUND_CSV), type=Path)
    parser.add_argument("--haplotype-scores", default=str(HAPLOTYPE_TABLE), type=Path)
    parser.add_argument("--cells", default=str(POOL_CELLS), type=Path)
    parser.add_argument("--shifts", default=str(POOL_SHIFTS), type=Path)
    parser.add_argument("--summary", default=str(POOL_SUMMARY), type=Path)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
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

    backgrounds = pd.read_csv(args.backgrounds)
    haplotype_scores = pd.read_parquet(args.haplotype_scores)
    annotations_path = mat_pcp_path("site_annotations_file")

    by_clade: dict[str, list[str]] = {}
    for clade, mutation in [
        *figure_candidates(),
        *((clade, mutation) for clade, mutation, _ in off_figure_candidates()),
    ]:
        by_clade.setdefault(clade, []).append(mutation)

    cells, shifts, summary = [], [], {}
    for clade in CANDIDATE_CLADES:
        split = backgrounds[
            (backgrounds["clade"] == clade)
            & (backgrounds["role"].isin(BACKGROUND_SPLIT_ROLES))
        ]
        if split.empty:
            raise ValueError(
                f"{args.backgrounds} gives {clade} no marker or co-traveller, "
                f"so it has no two backgrounds to contrast."
            )
        clade_cells, clade_shifts, clade_summary = clade_counts(
            clade,
            sorted(by_clade[clade]),
            pcp_path=args.pcp,
            annotations_path=annotations_path,
            haplotype_scores=haplotype_scores,
            split_derived=dict(zip(split["aa_site"].astype(int), split["mutant_aa"])),
            permutations=args.permutations,
            chunk_rows=PCP_CHUNK_ROWS,
            max_chunks=args.max_chunks,
        )
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
