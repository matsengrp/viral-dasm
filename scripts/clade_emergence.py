"""When each clade emerged, for `fig:subsampling`'s panel A.

Panel A colours every clade's dot by the date that clade appeared, which is
what turns a ranked list of 56 counts into a picture of the pandemic: the
clades with millions of sequences are the early ones, and the modern clades
sit at the bottom. The height alone does not say that.

**Emergence is the 1% date**, the earliest collection date by which one
hundredth of a clade's samples have been seen. Not the first sample: one
mislabelled or back-dated sequence would move it by months, and with 56 clades
on a colour ramp a single outlier rescales every other clade's hue. One per
cent of a clade with four million samples is forty thousand of them, which no
handful of stragglers can shift.

Carried across from `_clade_size_and_emergence` in
`notebooks/cap_arm_scatters.py` of the mat-pcp run this paper is built on --
the notebook that drew the published panel A, named in `main.tex`'s provenance
comment -- so the dots are coloured by the same measurement and not a second
reading of the same files.

**Its own rule, because it is a pass over 2.7 GB.** The 56 sample files hold
about 17 million lines between them. The panel is redrawn often and this answer
does not change unless the tree does, so the walk happens once and the panel
reads the small table it writes, exactly as `clade_backgrounds` does for
`fig:within_clade_shifts`.

GISAID-derived: reads sample *names* and their collection dates, and writes one
count and one date per clade. No sequence passes through this module.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install -- a
# script run puts `scripts/` on `sys.path[0]`, never the repository root.
from lib import assert_dasm2_is_this_repo

# Then `dasm2`, before pandas or pyarrow: `dasm2/arrow_pool.py` steers Arrow
# off the bundled mimalloc and can only do so if pyarrow has not been imported
# yet, and pandas pulls pyarrow in.
import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import pandas as pd

#: The share of a clade's samples that has to be seen before it counts as
#: emerged. See the module docstring on why this is not the first sample.
EMERGENCE_SHARE = 0.01

#: A sample line ends in its collection date, and the ones that do not are
#: undated rather than malformed -- mat-pcp keeps a sample whose date it could
#: not parse, so the tree it emits is the tree this pipeline scores. They are
#: counted in the clade's size and left out of its date.
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")

#: What the sample file for one clade is called, under `mat/clades`.
SAMPLES_SUFFIX = "_samples.txt"


def samples_path(clades_dir: Path, clade: str) -> Path:
    """Where one clade's sample list lives."""
    return clades_dir / f"{clade}{SAMPLES_SUFFIX}"


def clade_size_and_emergence(path: Path) -> tuple[int, str | None]:
    """One clade's sample count and its emergence date.

    Streamed a line at a time rather than read into a frame: 21J alone is 4.3
    million lines, and all that is wanted from each is the last ten characters.

    Returns:
        ``(n_samples, emergence)``, where the date is ``None`` for a clade with
        no dated sample at all. The count is every line, dated or not, because
        that is the clade's size -- it is what `mat/clade_counts.csv` reports
        and what panel A plots as the dot's height.
    """
    total_lines = 0
    seen: dict[str, int] = {}
    with open(path) as handle:
        for line in handle:
            total_lines += 1
            tail = line.rstrip("\n")[-10:]
            if DATE_PATTERN.fullmatch(tail):
                seen[tail] = seen.get(tail, 0) + 1
    dated = sum(seen.values())
    if not dated:
        return total_lines, None
    # At least one sample, so a clade whose 1% rounds to zero still emerges on
    # the day its first dated sample was collected rather than not at all.
    want = max(1, int(EMERGENCE_SHARE * dated))
    running = 0
    for day in sorted(seen):
        running += seen[day]
        if running >= want:
            return total_lines, day
    return total_lines, None


def clade_emergence_frame(clades_dir: Path, clades: list[str]) -> pd.DataFrame:
    """``clade,n_samples,emergence`` for ``clades``, in the order given.

    Raises:
        FileNotFoundError: If a clade in the roster has no sample file. The
            roster is what this pipeline scores, so a scored clade with no
            sample list means the founders and the tree disagree.
    """
    rows = []
    for clade in clades:
        path = samples_path(clades_dir, clade)
        if not path.exists():
            raise FileNotFoundError(
                f"{path}: no sample list for {clade}, which this pipeline "
                f"scores. The clade founders and mat-pcp's `mat/clades` "
                f"describe different trees."
            )
        n_samples, emergence = clade_size_and_emergence(path)
        rows.append({"clade": clade, "n_samples": n_samples, "emergence": emergence})
    return pd.DataFrame(rows)


def main() -> None:
    """Write the emergence table panel A colours its dots by."""
    import argparse

    from lib import clade_roster
    from panel_spec import EMERGENCE_CSV

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clades-dir", required=True, help="mat-pcp's `mat/clades` directory"
    )
    parser.add_argument("--out", default=str(EMERGENCE_CSV), help="where to write")
    args = parser.parse_args()

    frame = clade_emergence_frame(Path(args.clades_dir), clade_roster())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    undated = frame["emergence"].isna().sum()
    print(frame.to_string(index=False))
    print(f"{len(frame)} clades, {undated} with no dated sample")


if __name__ == "__main__":
    main()
