"""Verify the per-era TabularRates archives before anything trains.

A neutral model holding the wrong rates trains without error and yields wrong
selection factors, so these checks run against the built archives rather than
trusting the builder. Four of them, each catching a different way the build
could be wrong while still producing a well-formed archive:

1. **Every archive reproduces the one the paper trained against**, compared on
   a digest of its ``rates_table.csv``. ``config.yaml``'s ``neutral_archives``
   holds those digests, and they were computed from the archives the paper
   trained against rather than from this pipeline's own output -- so the
   equality is a reproduction and not a restatement.

   The digest covers a *canonical rendering* of the table -- rows sorted by key,
   each rate formatted to :data:`RATES_DIGEST_SIG` significant digits -- rather
   than the member's raw bytes, because the bytes record how a pandas version
   chose to print a float rather than what the model says.

   No rate is recomputed between the two builds. :func:`select_gene_rates` does
   no arithmetic on rates at all -- it masks rows out of the GLM's master table
   and sorts them -- so both builds ship the same doubles. What differs is the
   rendering: the pinned archives wrote about 16 significant digits, while this
   environment writes Python's shortest round-trip repr, and this build's rate
   strings are the master table's own strings. Parse the shorter text back and
   it lands up to 4.43e-14 relative away, worst case the omicron ``TG/GTA``
   unpaired=0 rate. Nothing about the model changed; a byte comparison fails
   anyway, and would fail again after every ``uv lock --upgrade``. A gate that
   fires on its own toolchain teaches the reader to re-pin it, which is how a
   real failure gets waved through.

   Eight significant digits sits beside check 3's ``rtol=1e-9`` so the two
   checks test one claim at one strength, and it is far tighter than any
   rate-selection change, which moves rates by whole percent: taking a
   before-boundary row where an after-boundary row belongs changes a rate in
   its first digits, not its eighth. The closest of these 1,152 rates sits
   1.3e-11 relative from an 8-digit rounding boundary, roughly 300x the
   rendering spread. Treat that as a measured property of this rate table
   rather than a designed margin -- it is a minimum-gap order statistic, and it
   would move if the GLM were refit. Six through nine digits all clear the
   spread; at eleven and twelve the nearest rate is within one part in 1e15 of
   a boundary, so agreement there would be luck.

   The member rather than the zip, because a zip records mtimes -- two
   identical rate tables packed a minute apart give different archive digests,
   and the check would fail on a legitimate rebuild.

   What this cannot do is show the rates were right to begin with: the
   comparand records what was built, not what should have been. Check 3 carries
   that weight, by rebuilding the table from the GLM input. See
   ``dasm2/PROVENANCE.md``.
2. **The two eras differ, in the direction mat-pcp measured.** G->T rates must
   be higher pre-Omicron by mat-pcp's own measured contrast. An archive built
   from the wrong era's master table shows ~0 here.
3. **The stored rates round-trip**, for every variant *including pooled*: the
   archive's table equals ``select_gene_rates`` applied to that variant's master
   table, so the zip holds what we believe it holds and not, say, the pooled
   selection under an era name. This one is independent of the pinned digests --
   it rebuilds the expected table from the GLM input rather than comparing to a
   recorded answer -- which is what keeps check 1's circularity from being the
   whole story.
4. **The three variants share one site annotation.** ``annotated_counts.csv``
   carries the per-site ``unpaired`` call, which the GLM does not refit per era.
   Here that holds by construction -- ``config.yaml`` names one
   ``site_annotations_file`` -- so what is asserted is that the builder still
   reads one path for every variant, which is the thing an edit could break.

**On checks that can pass without running.** Each guard is written so that
"this check did not execute" is a failure rather than a silent pass, which is
the failure mode a verification script is most prone to:

* Check 2 compares a mean over the G->T rows. If those rows are absent -- an
  upstream relabelling of ``mut_type`` would do it, while leaving the 384-row
  count and the 1:1 join intact -- ``.mean()`` returns ``nan``, and
  ``abs(nan - x) > tol`` is ``False``. The comparison would *pass*. So the row
  count and finiteness are asserted first.
* Check 1 fails rather than skips when ``config.yaml`` records no digest for a
  variant. "The bytes match" is precisely the claim a reader would take on
  trust, so a run that could not make it must not report success.
* The variant list comes from ``BUILD_VARIANTS`` rather than being restated, so
  adding a third era cannot leave the builder writing archives the verifier
  silently ignores.

GISAID: reads GLM tables and built archives on the cluster, and prints only
aggregate contrasts and digests.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first (it puts this repository's root on `sys.path`), then `dasm2`
# before pandas -- which pulls pyarrow in, and Arrow fixes its memory pool at
# first import. `tests/test_import_order.py` enforces both.
from lib import (
    GENE,
    NEUTRAL_ARCHIVE_DIGESTS,
    assert_dasm2_is_this_repo,
    master_table_path,
)

import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import argparse
import hashlib
import logging
from io import BytesIO

import numpy as np
import pandas as pd
from build_era_neutral_models import BUILD_VARIANTS, archive_name, build_inputs

from dasm2.neutral.scov2_rates import RATES_COLUMNS, select_gene_rates
from dasm2.serialization.zip_io import read_data_archive

logger = logging.getLogger(__name__)

#: The rate table's key columns: ``RATES_COLUMNS`` is this plus ``rate``.
RATE_KEY = [c for c in RATES_COLUMNS if c != "rate"]

#: mat-pcp's own pre-Omicron/Omicron log rate ratio for G->T. Asserted rather
#: than described, so a change in the upstream fit surfaces as a failure here
#: instead of as a stale comment. G->T because it is the largest era effect.
MATPCP_GT_LOG_RATIO = 0.6158
GT_TOL = 0.05

#: Rows the G->T contrast is averaged over: 32 (motif, unpaired) combinations.
#: Asserted so an upstream relabelling cannot quietly empty the selection.
EXPECTED_GT_ROWS = 32

#: The archive member carrying the fitted rates, and so the one whose digest is
#: pinned. See check 1.
RATES_MEMBER = "rates_table.csv"

#: Significant digits each rate is rendered to before hashing. Eight: see
#: check 1 for why the digest cannot be taken over the member's raw bytes, and
#: why more digits would pin how a pandas version prints a float rather than
#: what the model says.
RATES_DIGEST_SIG = 8


def variant_key(era: str | None) -> str:
    """``config.yaml``'s key for a build variant."""
    return era or "pooled"


def read_members(archive: Path, members: list[str]) -> dict[str, bytes]:
    """Named members of a TabularRates archive, as raw bytes.

    Via :func:`~dasm2.serialization.zip_io.read_data_archive`, the canonical
    reader for this archive family -- it matches member names exactly and raises
    a ``KeyError`` listing what the archive does contain, where a hand-rolled
    ``endswith`` scan matches ambiguously and raises a bare ``StopIteration``.
    """
    _, files = read_data_archive(archive, members)
    return files


def sorted_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` in a canonical row order, so two tables can be compared."""
    return frame.sort_values(RATE_KEY, kind="stable").reset_index(drop=True)


def canonical_rates_digest(frame: pd.DataFrame) -> str:
    """sha256 of ``frame`` rendered canonically. See check 1.

    Row order comes from :func:`sorted_rates` rather than from the file, so a
    rate table written in a different order still digests the same -- the
    archive's row order is a property of the writer, not of the model.

    The rendering formats a Python float itself rather than going through
    pandas, because a ``to_csv`` float format is a pandas version's business
    and this digest exists precisely to not depend on one.
    """
    rendered = "\n".join(
        ",".join(
            [
                *(str(row[column]) for column in RATE_KEY),
                f"{float(row['rate']):.{RATES_DIGEST_SIG}g}",
            ]
        )
        for _, row in sorted_rates(frame).iterrows()
    )
    return hashlib.sha256(rendered.encode()).hexdigest()


def check_rates_match(want: pd.DataFrame, got: pd.DataFrame, label: str) -> list[str]:
    """Compare two rate tables on keys *and* values, not just values.

    ``np.allclose`` on the rate column alone would broadcast a degenerate
    one-row table against all 384 expected values, and would not notice that the
    keys differ at all.
    """
    try:
        pd.testing.assert_frame_equal(
            sorted_rates(want), sorted_rates(got), check_exact=False, rtol=1e-9
        )
    except AssertionError as exc:
        first = str(exc).strip().splitlines()[0]
        return [f"{label}: {first}"]
    return []


def check_gene(gene: str, archive_dir: Path) -> tuple[dict, list[str]]:
    """Run all four checks for one gene. Returns (summary row, failures)."""
    failures: list[str] = []
    archives = {
        era: archive_dir / f"{archive_name(gene, era)}.zip" for era in BUILD_VARIANTS
    }
    absent = sorted(str(p) for p in archives.values() if not p.exists())
    if absent:
        return {"gene": gene, "status": "not built"}, [
            f"{gene}: archive(s) not built: {absent}"
        ]

    members = {era: read_members(p, [RATES_MEMBER]) for era, p in archives.items()}
    rates = {
        era: pd.read_csv(BytesIO(blob[RATES_MEMBER])) for era, blob in members.items()
    }

    # 1. every archive must reproduce the paper's, by its rate table's digest.
    digests_match = True
    for era in BUILD_VARIANTS:
        key = variant_key(era)
        got = canonical_rates_digest(rates[era])
        want = NEUTRAL_ARCHIVE_DIGESTS.get(key)
        if want is None:
            digests_match = False
            failures.append(
                f"{gene}/{key}: config.yaml's neutral_archives records no "
                f"digest, so the reproduction check did not run. Its digest is "
                f"{got}. A missing comparand is a failure and not a skip."
            )
        elif got != want:
            digests_match = False
            failures.append(
                f"{gene}/{key}: {RATES_MEMBER} digests to {got}, not the "
                f"recorded {want}. The digest is taken at "
                f"{RATES_DIGEST_SIG} significant digits, orders of magnitude "
                f"coarser than any float-rendering or library-version "
                f"difference -- so some rate moved by more than "
                f"1e-{RATES_DIGEST_SIG} relative. Look first at the rate "
                f"selection in dasm2.neutral.scov2_rates, then at the GLM "
                f"tables themselves: config.yaml pins md5s for the source "
                f"PCPs and the corpora, but NOT for glm/master_table*.csv, "
                f"and mat-pcp rebuilds its tree in place. Lambda is pinned "
                f"across every corpus, so this is not a difference the "
                f"pipeline can absorb."
            )

    # 3. every variant's stored rates round-trip against its own master table,
    #    pooled included -- otherwise the pooled archive is unchecked.
    for era in BUILD_VARIANTS:
        master = pd.read_csv(master_table_path(era))
        failures.extend(
            check_rates_match(
                select_gene_rates(master, gene),
                rates[era],
                f"{gene}/{variant_key(era)}: archive vs {master_table_path(era).name}",
            )
        )

    # 4. the variants must have been built from one site annotation.
    annotations = {
        era: build_inputs(gene, era).site_annotations for era in BUILD_VARIANTS
    }
    if len(set(annotations.values())) != 1:
        failures.append(
            f"{gene}: the variants were built from different site annotations "
            f"({ {variant_key(e): str(p) for e, p in annotations.items()} }); the "
            "structure table is only era-invariant because they read one file."
        )

    # 2. the eras must differ, in mat-pcp's measured direction
    merged = rates["pre_omicron"].merge(
        rates["omicron"], on=RATE_KEY, suffixes=("_pre", "_omi")
    )
    if len(merged) != len(rates["pre_omicron"]):
        failures.append(f"{gene}: the two era rate tables do not join 1:1")
    merged["dlog"] = np.log(merged["rate_pre"] / merged["rate_omi"])
    gt_rows = merged.loc[merged["mut_type"] == "GT", "dlog"]
    gt_dlog = float(gt_rows.mean()) if len(gt_rows) else float("nan")
    median_abs = float(merged["dlog"].abs().median())

    # Guard before comparing: `abs(nan - x) > tol` is False, so an empty or
    # non-finite selection would pass check 2 rather than failing it.
    if len(gt_rows) != EXPECTED_GT_ROWS:
        failures.append(
            f"{gene}: found {len(gt_rows)} G->T rows, expected "
            f"{EXPECTED_GT_ROWS}; check 2 did not run"
        )
    elif not (np.isfinite(gt_dlog) and np.isfinite(median_abs)):
        failures.append(
            f"{gene}: G->T contrast is not finite (gt_dlog={gt_dlog}, "
            f"median={median_abs}); check 2 did not run"
        )
    elif np.isclose(median_abs, 0.0, atol=1e-9):
        failures.append(
            f"{gene}: the two era archives hold identical rates, so "
            "'the eras differ' proves nothing"
        )
    elif abs(gt_dlog - MATPCP_GT_LOG_RATIO) > GT_TOL:
        failures.append(
            f"{gene}: G->T era contrast {gt_dlog:+.4f} does not match mat-pcp's "
            f"{MATPCP_GT_LOG_RATIO:+.4f} within {GT_TOL}"
        )

    return (
        {
            "gene": gene,
            "status": "ok" if not failures else "FAIL",
            "rate_rows": len(rates["pre_omicron"]),
            "digests_match": digests_match,
            "GT_dlog": gt_dlog,
            "median_|dlog|": median_abs,
        },
        failures,
    )


def main() -> int:
    """Verify the archives; non-zero exit on any failure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--genes", nargs="+", default=[GENE])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows, failures = [], []
    for gene in args.genes:
        row, gene_failures = check_gene(gene, args.archive_dir)
        rows.append(row)
        failures.extend(gene_failures)

    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\nmat-pcp's own per-era G->T contrast: {MATPCP_GT_LOG_RATIO:+.4f}")
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for failure in failures:
            print("  -", failure)
        return 1
    print(f"\nAll checks passed for {len(args.genes)} gene(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
