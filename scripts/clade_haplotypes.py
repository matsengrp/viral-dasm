"""Every distinct amino-acid parent haplotype in each of the figure's clades.

`fig:within_clade_shifts`' panels B, D and F put a mutation's effect on two
genetic backgrounds of one clade beside each other. The DASM side of that
comparison is the **pool average**: the mean log selection factor over the
haplotypes a background actually contains, weighted by each one's synonymous
events. So the model has to score every haplotype a pool might hold, and this
script is what says which those are.

**A superset of the pools, on purpose.**
:func:`dasm2.analysis.within_clade_pools.pool_average_log_sf` raises when a
haplotype in a pool has no score, rather than averaging over the intersection
and quietly dropping that haplotype's evolutionary time. Satisfying it means
scoring a set that is guaranteed to cover both pools, and the clade's whole
distinct-parent set is the only such set known before the counts pass runs.
Narrowing to pool members would save 13% of the passes -- 7,549 of 8,693 sit
in a pool -- and would make the scoring rule depend on the counts rule, which
is the ordering this DAG is arranged to avoid.

**Two haplotypes the scorer would otherwise refuse.**
``scripts/score_haplotypes.py`` rejects a sequence containing anything but the
twenty residues, and rejects an empty background label. Both would arrive
without this script's filtering:

* ``translate_sequence`` yields :data:`UNKNOWN_AA` for a stop or an unreadable
  codon. Such a parent is dropped here and counted in the printed summary. It
  is consistent with how the pools treat it -- :meth:`PoolSpec.matches` already
  refuses a parent whose codon at a required site will not translate -- and it
  is better to lose it here, loudly, than inside a scoring loop.
* A parent whose haplotype *is* the founder has no substitutions, so
  :func:`substitution_key` gives it the empty string. It is labelled
  :data:`FOUNDER_LABEL` instead. Dropping it is not an option: it is usually
  the heaviest haplotype of the founder-side pool.

GISAID: reads GISAID-derived parent sequences and writes amino-acid
haplotypes, so the output goes under the artifact root rather than into
tracked ``results/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install.
from lib import GENE, assert_dasm2_is_this_repo

# Then `dasm2`, before pandas or pyarrow: `dasm2/arrow_pool.py` moves Arrow off
# the bundled mimalloc, and can only do so if pyarrow has not been imported
# yet. This script writes a Parquet, so the allocator is not academic.
import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import argparse
from collections import Counter

import pandas as pd
from founders import founder_aa_sequence
from panel_spec import (
    CANDIDATE_CLADES,
    CLADE_HAPLOTYPE_TABLE,
    PCP_CHUNK_ROWS,
)

from dasm2.analysis.within_clade_pools import substitution_key
from dasm2.sequence.conversion import translate_sequence

#: The residues a scorable haplotype may be spelled with. The same twenty
#: `score_haplotypes.AA_ALPHABET` enforces; named again here rather than
#: imported because that module loads torch and this one must not.
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")

#: What a haplotype identical to the clade founder is called.
#: :func:`substitution_key` returns the empty string for it, which
#: ``score_haplotypes.read_backgrounds`` refuses -- an empty ``background`` is
#: the founder table's convention for "this row is a founder", and the two
#: tables must not collide on it.
FOUNDER_LABEL = "founder"


def clade_haplotypes(
    clades: list[str], pcp_path: str | Path, *, chunk_rows: int
) -> pd.DataFrame:
    """One row per distinct amino-acid parent, per clade.

    One streaming pass. Parents are collected as nucleotide strings and
    translated afterwards, because a clade's parents repeat heavily -- 21K's
    148,927 branches carry 8,729 distinct nucleotide parents, which collapse
    to 4,633 amino-acid haplotypes -- so translating per branch would do
    twenty times the work for the same answer.

    Returns:
        ``(clade, background, aa_sequence, n_pcps, n_nt_parents)``, sorted
        heaviest first within a clade. ``background`` is the haplotype's
        substitutions against the clade founder, or :data:`FOUNDER_LABEL`.
    """
    wanted = set(clades)
    pcps: dict[str, Counter] = {clade: Counter() for clade in clades}
    for chunk in pd.read_csv(
        pcp_path, usecols=["clade", "parent"], chunksize=chunk_rows
    ):
        chunk = chunk[chunk["clade"].isin(wanted)]
        if chunk.empty:
            continue
        for clade, part in chunk.groupby("clade"):
            pcps[clade].update(part["parent"])

    rows = []
    for clade in clades:
        if not pcps[clade]:
            raise ValueError(
                f"no branches for clade {clade!r} in {pcp_path}. Every pool of "
                f"this clade would be empty, and the panel would be drawn on "
                f"nothing."
            )
        founder_aa = founder_aa_sequence(clade, GENE)
        by_haplotype: dict[str, dict[str, int]] = {}
        unreadable = 0
        for parent, count in pcps[clade].items():
            aa_parent = translate_sequence(parent)
            if not set(aa_parent) <= AA_ALPHABET:
                unreadable += count
                continue
            entry = by_haplotype.setdefault(aa_parent, {"n_pcps": 0, "n_nt_parents": 0})
            entry["n_pcps"] += count
            entry["n_nt_parents"] += 1
        if not by_haplotype:
            raise ValueError(
                f"every parent of clade {clade!r} carries an unreadable codon."
            )
        for aa_parent, entry in by_haplotype.items():
            rows.append(
                {
                    "clade": clade,
                    "background": substitution_key(aa_parent, founder_aa)
                    or FOUNDER_LABEL,
                    "aa_sequence": aa_parent,
                    "n_pcps": entry["n_pcps"],
                    "n_nt_parents": entry["n_nt_parents"],
                }
            )
        print(
            f"{clade:>5}: {sum(pcps[clade].values()):>7,} branches, "
            f"{len(pcps[clade]):>6,} nucleotide parents, "
            f"{len(by_haplotype):>6,} haplotypes"
            + (
                f", {unreadable:,} branches dropped (unreadable codon)"
                if unreadable
                else ""
            ),
            flush=True,
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["clade", "n_pcps"], ascending=[True, False])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcp", required=True, help="the PCP corpus to read")
    parser.add_argument("--output", default=str(CLADE_HAPLOTYPE_TABLE), type=Path)
    args = parser.parse_args()

    frame = clade_haplotypes(
        list(CANDIDATE_CLADES), args.pcp, chunk_rows=PCP_CHUNK_ROWS
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(
        f"wrote {len(frame):,} haplotypes over "
        f"{frame['clade'].nunique()} clades to {args.output}"
    )


if __name__ == "__main__":
    main()
