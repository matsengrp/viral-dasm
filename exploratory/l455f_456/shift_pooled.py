"""Pooled L455F contrast across the XBB subclades, split on the 456 marker.

viral-dasm#64. Each clade on its own gives a within-clade contrast with a
handful of events on the 456L side. Pooling puts every XBB subclade's branches
into two pools by the parent's residue at 456 and asks the same question of
the combined counts.

Counts side: sum `actual` and `expected` over clades within each 456 residue,
then effect = log(sum actual / sum expected). Summing is the right operation --
`expected` is a total evolutionary time times a rate, and total time is
additive over disjoint branch sets, which is what makes a pooled expected count
mean anything at all.

DASM side: each clade-pool's effect is already an expected-count-weighted mean
over its haplotypes (`pool_average_log_sf`), so pooling across clades weights
each clade-pool by its own expected count -- through `pooled_log_sf`, in the
LINEAR domain, which is the same operation that function performs over
haplotypes within a pool. A log-space weighted mean here would be the geometric
mean, which is not what the counts aggregate computes: that divides summed
events by summed expected counts, a ratio of sums, and matching it forces the
arithmetic mean of the selection factors. The two differ by about sigma**2 / 2
(Jensen), which on the 456L pool is 0.10 nats.

Both pools require the founder residue at 455 (the at-risk requirement), so
`actual` and `expected` describe the same branches on both sides and the only
declared difference between the pools is the residue at 456.

Not a pipeline rule: no snakemake, no writes under the artifact root, no
training. Reads the published checkpoint to score.

Usage (from a checkout with its virtual environment):

    python exploratory/l455f_456/shift_pooled.py --out _ignore/issue64_pooled
"""

import sys
from pathlib import Path

# `exploratory/` is not a package and these scripts are run as files, so
# nothing has put `scripts/` on the path yet. Resolved from `__file__` rather
# than hard-coded, so a worktree runs its own `scripts/` and its own vendored
# `dasm2` -- which is what `assert_dasm2_is_this_repo` then checks.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install.
from lib import (
    ARTIFACT_ROOT,
    GENE,
    assert_dasm2_is_this_repo,
    corpus,
    master_table_path,
    mat_pcp_path,
    select_checkpoint,
)

# Then `dasm2`, before pandas or pyarrow: `dasm2/arrow_pool.py` moves Arrow off
# the bundled mimalloc, and can only do so if pyarrow has not been imported
# yet. These scripts read Parquet, so the allocator is not academic.
import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import argparse
import json

import numpy as np
import pandas as pd
from clade_haplotypes import FOUNDER_LABEL, clade_haplotypes
from founders import founder_aa_sequence
from panel_spec import ARM, FIGURE_MODEL, PCP_CHUNK_ROWS, era_of_clade
from runs import run_id
from score_haplotypes import score_backgrounds
from score_run import globally_masked_sites

from dasm2.analysis.within_clade_pools import (
    PoolSpec,
    join_rates,
    load_annotations,
    pool_average_log_sf,
    pool_cell_table,
    pooled_log_sf,
    tally_pools,
)
from dasm2.sequence.conversion import translate_sequences
from dasm2.sequence.masking import load_censor_table

PCP = corpus(ARM).source_pcp
RUN_DIR = Path(ARTIFACT_ROOT) / "runs" / run_id(FIGURE_MODEL, ARM)

SITE, MUT, MARKER = 455, "F", 456

#: The XBB subclades in this run's roster, by Nextstrain clade: 22F is XBB
#: itself, then XBB.1.5, XBB.1.16, XBB.1.9, XBB.2.3 and EG.5.1. 23C (CH.1.1),
#: 23I (BA.2.86) and 24A (JN.1) are not XBB and are deliberately absent.
DEFAULT_CLADES = ("22F", "23A", "23B", "23D", "23E", "23F")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--clades", nargs="+", default=list(DEFAULT_CLADES))
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

print(f"XBB subclades: {args.clades}\nenumerating haplotypes ...", flush=True)
haps = clade_haplotypes(args.clades, PCP, chunk_rows=PCP_CHUNK_ROWS)

cache = args.out / "haplotype_scores.parquet"
if cache.exists():
    print(f"\nreusing cached scores: {cache}", flush=True)
    scores = pd.read_parquet(cache)
else:
    print(
        f"\nscoring L455F on {len(haps):,} haplotypes (cpu, published checkpoint) ...",
        flush=True,
    )
    scores = score_backgrounds(
        haps,
        {clade: ((SITE, MUT),) for clade in args.clades},
        checkpoint=select_checkpoint(RUN_DIR),
        gene=GENE,
        run_id=run_id(FIGURE_MODEL, ARM),
        model=FIGURE_MODEL,
        corpus=ARM,
        masked=globally_masked_sites(),
        censor_table=load_censor_table(
            mat_pcp_path("censored_mutations_file"),
            mat_pcp_path("censored_clades_file"),
        ),
        device="cpu",
        progress_every=2000,
    )
    scores.to_parquet(cache, index=False)

rows = []
for clade in args.clades:
    founder = founder_aa_sequence(clade, GENE)
    annotations = join_rates(
        load_annotations(mat_pcp_path("site_annotations_file"), [clade], GENE),
        pd.read_csv(master_table_path(era_of_clade(clade))),
    )
    specs = [
        PoolSpec(
            name=f"456{residue}",
            at_risk={SITE: founder[SITE - 1]},
            marker={MARKER: residue},
            hold={},
        )
        for residue in ("F", "L")
    ]
    tallies = tally_pools(
        PCP, specs, annotations, clade=clade, chunk_rows=PCP_CHUNK_ROWS
    )
    cells = pool_cell_table(
        tallies, annotations, cluster=clade, gene=GENE, pool0=specs[0], pool1=specs[1]
    )
    cell = cells[(cells.aa_site == SITE) & (cells.mutant_aa == MUT)]
    clade_scores = scores[scores.clade == clade]
    for spec in specs:
        got = cell[cell.pool_name == spec.name]
        if got.empty:
            print(
                f"  {clade}/{spec.name}: no cell -- {MUT} is not nucleotide-accessible here"
            )
            continue
        got = got.iloc[0]
        by_parent = tallies.synonymous_by_parent[spec.name]
        if not by_parent:
            print(f"  {clade}/{spec.name}: no branches -- skipped")
            continue
        lut = {
            ("" if background == FOUNDER_LABEL else background): float(value)
            for background, value in zip(
                clade_scores["background"], clade_scores["log_selection_factor"]
            )
        }
        parents = list(by_parent)
        average = pool_average_log_sf(
            by_parent,
            dict(zip(parents, translate_sequences(parents))),
            lut,
            founder_aa=founder,
        )
        rows.append(
            {
                "clade": clade,
                "marker": spec.name[-1],
                "n_pcps": int(got.n_pcps),
                "actual": float(got.actual),
                "expected": float(got.expected),
                "counts_effect": float(got.counts_effect),
                "dasm_effect": float(average.log_sf),
                "n_haplotypes": int(average.n_haplotypes),
                "synonymous_weight": float(average.synonymous_weight),
            }
        )
        print(
            f"  {clade}/{spec.name}: actual={got.actual:>6.0f} expected={got.expected:>8.2f} "
            f"counts={got.counts_effect:+.2f} dasm={average.log_sf:+.2f}",
            flush=True,
        )

per_clade = pd.DataFrame(rows)
per_clade.to_csv(args.out / "per_clade_pools.csv", index=False)

print("\n" + "=" * 74 + "\nPOOLED ACROSS XBB SUBCLADES\n" + "=" * 74)
pooled = {}
for marker, sub in per_clade.groupby("marker"):
    actual, expected = sub.actual.sum(), sub.expected.sum()
    pooled[marker] = {
        "clades": sorted(sub.clade),
        "n_clades": len(sub),
        "actual": float(actual),
        "expected": float(expected),
        "counts_effect": float(np.log(actual / expected)),
        "dasm_effect": float(
            pooled_log_sf(sub.expected.tolist(), sub.dasm_effect.tolist())
        ),
        "branches": int(sub.n_pcps.sum()),
    }
    value = pooled[marker]
    print(
        f"  456{marker}: {value['n_clades']} clades, {value['branches']:,} branches, "
        f"actual={actual:.0f} expected={expected:.2f}"
    )
    print(
        f"         counts effect {value['counts_effect']:+.2f}   DASM {value['dasm_effect']:+.2f}"
    )

counts_shift = pooled["L"]["counts_effect"] - pooled["F"]["counts_effect"]
dasm_shift = pooled["L"]["dasm_effect"] - pooled["F"]["dasm_effect"]
print(
    f"\n  POOLED SHIFT (456L - 456F):  counts {counts_shift:+.2f}   DASM {dasm_shift:+.2f}"
)
print(f"  DASM / counts amplitude: {dasm_shift / counts_shift:.2f}")
(args.out / "pooled.json").write_text(
    json.dumps(
        {"pools": pooled, "counts_shift": counts_shift, "dasm_shift": dasm_shift},
        indent=2,
    )
)
print(f"\nwrote {args.out}")
