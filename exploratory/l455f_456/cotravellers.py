"""What differs between the 456F and 456L pools, besides 456.

viral-dasm#64. The contrast `shift_pooled.py` measures declares one difference
between its two pools -- the parent's residue at 456 -- but the pools are
observational, so anything that happened to ride along with F456L is also a
difference between them. This script enumerates those.

It walks the same branch sets the contrast uses: the parent must carry the
founder residue at 455 (the at-risk requirement), then the branches split on
the parent's residue at 456. For every other site, it reports the share of each
pool's BRANCH MASS -- branch length, the same weight the counts side integrates
to get `expected` -- carrying a non-founder residue. A site with a large gap
between the pools is a co-traveller: the two pools differ there as well as at
456, so it rides along in the measured shift.

A caveat on the pooled table at the bottom: the six clades have six different
founders, so "a substitution" at a site is a change from a different reference
in each. Site 486 is founder-P in 23A but founder-S in 22F, and the pooled row
spells the founder residue of the first clade only. Read the per-clade rows
above it, which are unambiguous; the pooled row is a magnitude, not a label.

One pass over the full PCP corpus, reading parent sequences. No model, no
snakemake, no writes.

Usage (from a checkout with its virtual environment):

    python exploratory/l455f_456/cotravellers.py | tee _ignore/issue64_cotravellers.txt
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
from lib import GENE, assert_dasm2_is_this_repo, corpus

# Then `dasm2`, before pandas or pyarrow: `dasm2/arrow_pool.py` moves Arrow off
# the bundled mimalloc, and can only do so if pyarrow has not been imported
# yet. These scripts read Parquet, so the allocator is not academic.
import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from founders import founder_nt_sequence
from panel_spec import ARM, PCP_CHUNK_ROWS

from dasm2.sequence.conversion import translate_sequence

PCP = corpus(ARM).source_pcp
SITE, MARKER = 455, 456

#: Same roster as `shift_pooled.py`: the XBB subclades, and only those.
DEFAULT_CLADES = ("22F", "23A", "23B", "23D", "23E", "23F")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--clades", nargs="+", default=list(DEFAULT_CLADES))
parser.add_argument(
    "--gap",
    type=float,
    default=0.05,
    help=(
        "report a site in the per-clade lines only when the two pools' shares "
        "differ by more than this. At 0 every site that ever differs is listed, "
        "which on this corpus is some 80,000 characters of mostly 0%%->0%%."
    ),
)
args = parser.parse_args()
clades = list(args.clades)

founder_nt = {clade: founder_nt_sequence(clade, GENE) for clade in clades}
n_codons = len(next(iter(founder_nt.values()))) // 3
founder_aa = {
    clade: [translate_sequence(seq[3 * i : 3 * i + 3]) for i in range(n_codons)]
    for clade, seq in founder_nt.items()
}
founder_bytes = {
    clade: np.frombuffer(seq.encode(), dtype="S1") for clade, seq in founder_nt.items()
}

mass = {clade: {"F": defaultdict(float), "L": defaultdict(float)} for clade in clades}
denominator = {clade: {"F": 0.0, "L": 0.0} for clade in clades}

for chunk in pd.read_csv(
    PCP, usecols=["clade", "parent", "branch_length"], chunksize=PCP_CHUNK_ROWS
):
    chunk = chunk[chunk["clade"].isin(clades)]
    if chunk.empty:
        continue
    for clade, part in chunk.groupby("clade"):
        founder_at_site = founder_aa[clade][SITE - 1]
        for parent, length in zip(part["parent"], part["branch_length"].astype(float)):
            if translate_sequence(parent[3 * (SITE - 1) : 3 * SITE]) != founder_at_site:
                continue  # not at risk for the mutation under test
            marker_residue = translate_sequence(parent[3 * (MARKER - 1) : 3 * MARKER])
            if marker_residue not in ("F", "L"):
                continue
            denominator[clade][marker_residue] += length
            block = np.frombuffer(parent.encode(), dtype="S1")
            differs = (block != founder_bytes[clade]).reshape(n_codons, 3).any(axis=1)
            for codon in np.nonzero(differs)[0].tolist():
                residue = translate_sequence(parent[3 * codon : 3 * codon + 3])
                if residue == founder_aa[clade][codon] or residue == "X":
                    continue  # synonymous, or an ambiguous codon
                mass[clade][marker_residue][(codon + 1, residue)] += length

print(
    f"{'clade':<6} {'456F mass':>10} {'456L mass':>10}   co-travellers with 456L "
    f"(|gap| > {args.gap:.0%}, {MARKER} excluded)"
)
pool_total = {"F": 0.0, "L": 0.0}
pool_mass = {"F": defaultdict(float), "L": defaultdict(float)}
for clade in clades:
    mass_f, mass_l = denominator[clade]["F"], denominator[clade]["L"]
    pool_total["F"] += mass_f
    pool_total["L"] += mass_l
    for residue in ("F", "L"):
        for key, value in mass[clade][residue].items():
            pool_mass[residue][key] += value
    hits = []
    for site, residue in sorted(set(mass[clade]["F"]) | set(mass[clade]["L"])):
        if site == MARKER:
            continue
        share_f = mass[clade]["F"].get((site, residue), 0.0) / mass_f if mass_f else 0.0
        share_l = mass[clade]["L"].get((site, residue), 0.0) / mass_l if mass_l else 0.0
        if abs(share_l - share_f) > args.gap:
            hits.append(
                f"{founder_aa[clade][site - 1]}{site}{residue} ({share_f:.0%}->{share_l:.0%})"
            )
    print(
        f"{clade:<6} {mass_f:>10.0f} {mass_l:>10.0f}   {', '.join(hits) if hits else '(none)'}"
    )

print(
    f"\n{'=' * 74}\nPOOLED over the {len(clades)} XBB clades "
    f"(456F mass {pool_total['F']:.0f}, 456L mass {pool_total['L']:.0f})\n{'=' * 74}"
)
ranked = []
for site, residue in set(pool_mass["F"]) | set(pool_mass["L"]):
    if site == MARKER:
        continue
    share_f = pool_mass["F"][(site, residue)] / pool_total["F"]
    share_l = pool_mass["L"][(site, residue)] / pool_total["L"]
    ranked.append((abs(share_l - share_f), site, residue, share_f, share_l))
print(f"  {'substitution':<14} {'in 456F pool':>13} {'in 456L pool':>13} {'gap':>8}")
for _gap, site, residue, share_f, share_l in sorted(ranked, reverse=True)[:12]:
    # The founder residue of the FIRST clade only -- see the module docstring.
    label = f"{founder_aa[clades[0]][site - 1]}{site}{residue}"
    print(f"  {label:<14} {share_f:>12.1%} {share_l:>12.1%} {share_l - share_f:>+8.1%}")
