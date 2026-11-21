"""Within-clade L455F contrast, 456F parents against 456L parents, per clade.

viral-dasm#64. The marker is site 456 alone, fixed by hand: no
`clade_backgrounds` call, no co-travellers, no majority rule. That is the whole
reason this is a script and not a pipeline rule -- `rule clade_backgrounds`
picks 23A's marker unaided and pulls L455F itself in as a co-traveller at 59%,
which leaves the cell we want to measure unscoreable.

Everything downstream of the pool definition is the pipeline's own machinery:
`clade_counts` is what `rule pool_counts` calls, so the counts estimate and the
DASM estimate here are the two `fig:within_clade_shifts` reports for every
other candidate, computed the same way on different pools.

This script never invokes snakemake, writes nothing under the artifact root,
and touches no tracked pipeline output. It reads the published checkpoint to
score; it never trains.

Usage (from a checkout with its virtual environment):

    python exploratory/l455f_456/shift_per_clade.py --out _ignore/issue64
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

import pandas as pd
from clade_haplotypes import clade_haplotypes
from founders import founder_aa_sequence
from panel_spec import ARM, FIGURE_MODEL, PCP_CHUNK_ROWS
from pool_counts import clade_counts
from runs import run_id
from score_haplotypes import score_backgrounds
from score_run import globally_masked_sites

from dasm2.sequence.masking import load_censor_table

PCP = corpus(ARM).source_pcp
RUN_DIR = Path(ARTIFACT_ROOT) / "runs" / run_id(FIGURE_MODEL, ARM)

CANDIDATE = "L455F"
SITE, MUT, MARKER = 455, "F", 456
SPLIT = {MARKER: "L"}  # the marker, alone

#: 22F, 23A, 23B and 23D are the XBB subclades with both pools populated; 23C
#: is not XBB at all and rides along as a negative control.
DEFAULT_CLADES = ("23D", "23A", "23B", "22F", "23C")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--clades", nargs="+", default=list(DEFAULT_CLADES))
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--permutations", type=int, default=10_000)
parser.add_argument("--max-chunks", type=int, default=None)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

print(f"enumerating haplotypes for {args.clades} ...", flush=True)
haps = clade_haplotypes(args.clades, PCP, chunk_rows=PCP_CHUNK_ROWS)
print(haps.groupby("clade").size().to_string(), flush=True)

for clade in args.clades:
    founder = founder_aa_sequence(clade, GENE)
    if founder[SITE - 1] != "L":
        raise SystemExit(
            f"{clade}'s founder carries {founder[SITE - 1]} at {SITE}, not L: "
            f"{CANDIDATE} is not the mutation this clade is at risk for."
        )

checkpoint = select_checkpoint(RUN_DIR)
print(
    f"\nscoring {CANDIDATE} on {len(haps):,} haplotypes, {checkpoint.name}, cpu",
    flush=True,
)
scores = score_backgrounds(
    haps,
    {clade: ((SITE, MUT),) for clade in args.clades},
    checkpoint=checkpoint,
    gene=GENE,
    run_id=run_id(FIGURE_MODEL, ARM),
    model=FIGURE_MODEL,
    corpus=ARM,
    masked=globally_masked_sites(),
    censor_table=load_censor_table(
        mat_pcp_path("censored_mutations_file"), mat_pcp_path("censored_clades_file")
    ),
    device="cpu",
    progress_every=1000,
)
scores.to_parquet(args.out / "haplotype_scores.parquet", index=False)

annotations_path = mat_pcp_path("site_annotations_file")
all_cells, all_shifts, summaries = [], [], {}
for clade in args.clades:
    print(f"\n=== {clade} ===", flush=True)
    cells, shifts, summary = clade_counts(
        clade,
        [CANDIDATE],
        pcp_path=PCP,
        annotations_path=annotations_path,
        haplotype_scores=scores[scores["clade"] == clade],
        split_derived=SPLIT,
        permutations=args.permutations,
        chunk_rows=PCP_CHUNK_ROWS,
        max_chunks=args.max_chunks,
    )
    cells.insert(0, "clade", clade)
    shifts.insert(0, "clade", clade)
    all_cells.append(cells)
    all_shifts.append(shifts)
    summaries[clade] = summary

pd.concat(all_cells).to_parquet(args.out / "pool_cells.parquet", index=False)
pd.concat(all_shifts).to_parquet(args.out / "pool_shifts.parquet", index=False)
(args.out / "summary.json").write_text(json.dumps(summaries, indent=2))
print(f"\nwrote {args.out}")
