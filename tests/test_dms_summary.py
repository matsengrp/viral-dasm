"""What a wrong predictor table would report, and what catches it.

`scripts/dms_summary.py` turns joined score frames into the tracked CSV. Its two
guards are the kind that fail quietly: a table that is short by one arm still
looks complete, and a correlation computed over fewer cells than the assay
measured still prints a confident four-decimal coefficient. Both raise here
instead, and these are the checks that say so.

Everything runs on hand-built frames and needs no data, so it runs off-cluster.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

pd = pytest.importorskip("pandas")
pearsonr = pytest.importorskip("scipy.stats").pearsonr

import dms
import dms_summary
from dms_summary import DMS_COLUMN
from panel_spec import DMS_PREDICTORS

#: Enough cells per stratum for `within_clade_metrics`, which refuses below three.
_PER_STRATUM = 6


def _cells() -> pd.DataFrame:
    """One homolog's joined frame: both strata, every predictor scored.

    The score columns are built per predictor rather than spelled out, so an arm
    added to :data:`DMS_PREDICTORS` extends this file's coverage instead of
    breaking its fixture. No two of them are the same function of the
    measurement, which is the property that makes them tell each other apart:
    columns that were affine copies of one vector would share every rank and
    every coefficient, and a summary that swapped or mislabelled two predictors
    would still pass. :func:`test_each_predictor_is_scored_from_its_own_column`
    is what holds that property in place.
    """
    rows = []
    for index in range(_PER_STRATUM):
        for stratum in (dms.SINGLE_NT, dms.MULTI_NT):
            site = index + 1 + (0 if stratum == dms.SINGLE_NT else 100)
            row = {
                "aa_site": site,
                "mutant_aa": "ACDEFG"[index],
                "parent_aa": "M",
                "dms_cell_entry": float(index) - 2.0,
                "stratum": stratum,
            }
            for offset, predictor in enumerate(DMS_PREDICTORS):
                # Built from the loop's own `index`, not from the frame's row
                # number: the rows interleave the two strata, so a positional
                # expression would score a cell against another cell's index.
                #
                # Sign alternates and odd cells are nudged by an amount that
                # grows with the predictor, which is what makes every column
                # score differently. Sign alone stops being enough at three
                # arms: the first and third would share it, and a swap between
                # them would go unseen.
                row[predictor] = (-1.0) ** offset * (
                    float(index) + offset * (index % 2) * 0.9
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _manifests(strata=(dms.ALL,)) -> dict:
    return {
        "BA.1": {predictor: {"strata": list(strata)} for predictor in DMS_PREDICTORS}
    }


def test_all_is_the_union_of_the_two_strata_and_not_a_third_label():
    """`all` must partition, not nest.

    ``melt_dms`` labels every cell ``single_nt`` or ``multi_nt``; ``all`` is a
    union label no cell carries. A summary that filtered on ``stratum == "all"``
    would emit three empty rows per predictor rather than raising, so what this
    pins is that the two strata's counts sum to the third's.
    """
    table = dms_summary.summarize({"BA.1": _cells()}, _manifests())

    assert len(table) == len(DMS_PREDICTORS) * len(dms.STRATA)
    for predictor, rows in table.groupby("predictor"):
        counts = rows.set_index("stratum")["n_mutations"]
        assert counts[dms.ALL] == counts[dms.SINGLE_NT] + counts[dms.MULTI_NT]
        assert counts[dms.ALL] == 2 * _PER_STRATUM, predictor


def test_each_predictor_is_scored_from_its_own_column():
    """A summary that read one predictor's scores under another's label.

    Every row is well formed under a column mixup -- the keys, the counts and
    the column set are all unchanged -- so nothing structural catches it.

    **Asserting only that the coefficients differ is not enough**, which is the
    trap this test was rewritten to escape: rotating the labels permutes a set
    of distinct values into another set of distinct values, so a distinctness
    check passes. Each predictor's coefficient is therefore compared against one
    computed independently from the column that bears its name.
    """
    cells = _cells()
    table = dms_summary.summarize({"BA.1": cells}, _manifests())
    reported = table[table["stratum"] == dms.ALL].set_index("predictor")["pearson_r"]

    for predictor in DMS_PREDICTORS:
        expected = pearsonr(cells[predictor], cells[DMS_COLUMN]).statistic
        assert reported[predictor] == pytest.approx(expected, abs=1e-12), (
            f"{predictor} is reported at {reported[predictor]:+.6f} but its own "
            f"column correlates at {expected:+.6f}: the table is carrying "
            f"another predictor's score under this label"
        )
    # And the fixture must keep the labels telling apart, or the loop above
    # would hold under a rotation that happened to be the identity on values.
    assert len(set(reported.round(9))) == len(DMS_PREDICTORS), (
        f"the fixture no longer distinguishes predictors: {dict(reported)}"
    )


def test_a_predictor_that_scored_nothing_fails_rather_than_shortening_the_table():
    """A missing arm is the failure that looks most like success.

    The CSV's shape is one row per `(homolog, predictor, stratum)`, so a table
    built from one predictor is a well-formed table reporting a comparison that
    was never run.
    """
    cells = _cells().drop(columns=[DMS_PREDICTORS[0]])
    with pytest.raises(ValueError, match="no scores joined"):
        dms_summary.summarize({"BA.1": cells}, _manifests())


def test_an_unscored_cell_fails_rather_than_lowering_n():
    """`within_clade_metrics` drops NaN rows, so a short `n` is the only trace.

    A cell that failed to join, or whose score came back NaN, leaves the
    coefficient looking ordinary over a smaller population. The check compares
    the reported `n` against the cells in the strata the predictor *declares*.
    """
    cells = _cells()
    cells.loc[0, DMS_PREDICTORS[0]] = float("nan")
    with pytest.raises(ValueError, match="reports .* mutations against"):
        dms_summary.summarize({"BA.1": cells}, _manifests())


def test_a_predictor_may_declare_one_stratum_without_failing_the_coverage_check():
    """Partial coverage is an absence, not a defect.

    `join_scores` allows a predictor that speaks on a subset, so the coverage
    check is against the strata the manifest declares rather than against every
    measured cell -- otherwise the first partial-coverage arm would be refused.
    """
    cells = _cells()
    single_only = cells["stratum"] == dms.SINGLE_NT
    cells.loc[~single_only, DMS_PREDICTORS[0]] = float("nan")

    manifests = _manifests()
    manifests["BA.1"][DMS_PREDICTORS[0]] = {"strata": [dms.SINGLE_NT]}

    table = dms_summary.summarize({"BA.1": cells}, manifests)
    reported = table.set_index(["predictor", "stratum"])["n_mutations"]
    assert reported[(DMS_PREDICTORS[0], dms.ALL)] == _PER_STRATUM
    assert reported[(DMS_PREDICTORS[1], dms.ALL)] == 2 * _PER_STRATUM


def test_the_written_columns_are_the_ones_the_manuscript_reports():
    """A column added here lands in a tracked CSV, so the set is pinned.

    The table carries what ``within_clade_metrics`` returns, the three keys and
    ``median_score``, and nothing else: the site-centred and mutant-observed
    readouts were cut as outside the manuscript's scope, and a reintroduction
    should be a decision rather than a diff nobody notices. ``median_score`` is
    here because a claim needs it -- `_median_score` records which.
    """
    table = dms_summary.summarize({"BA.1": _cells()}, _manifests())
    assert tuple(table.columns) == dms_summary.SUMMARY_COLUMNS
    assert tuple(table.columns) == (
        "homolog",
        "predictor",
        "stratum",
        "pearson_r",
        "spearman_rho",
        "regression_slope",
        "median_score",
        "n_mutations",
    )


def test_the_median_is_taken_over_the_rows_the_correlation_used():
    """`median_score` and `n_mutations` describe one population, or neither does.

    ``within_clade_metrics`` drops a row where either column is NaN. A median
    over the raw subset would then be over more rows than the `n` printed
    beside it -- a drift that nothing downstream could see, since both numbers
    land in the same CSV row and read as if they came from the same cells.

    Unit-level, because `summarize`'s coverage guard refuses a frame with an
    unscored cell in a declared stratum: the drop this is about happens inside
    the helper, which is where it can be shown.
    """
    frame = pd.DataFrame(
        {
            dms_summary.DMS_COLUMN: [-6.0, -3.0, 0.0, float("nan")],
            dms_summary.METRIC_COLUMN: [-5.0, -2.0, float("nan"), 100.0],
        }
    )
    # Surviving rows are the first two, so the median is -3.5. Either of the
    # dropped cells would move it: keeping the NaN score makes it NaN, and
    # keeping the unmeasured 100.0 makes it -2.0.
    assert dms_summary._median_score(frame) == pytest.approx(-3.5)
    assert frame[dms_summary.METRIC_COLUMN].median() == pytest.approx(-2.0)

    # Empty is NaN rather than a raise: `dasm_prob` declares no
    # multi-nucleotide stratum, so that row is empty by construction.
    assert pd.isna(dms_summary._median_score(frame.iloc[:0]))


def test_the_median_lands_in_the_table_per_group():
    """And it is the group's median, not the homolog's: the strata differ.

    The claim it exists for is a *difference between strata* -- antiGen scores
    the single-nucleotide cells higher than the multi-nucleotide ones -- so a
    median computed once per predictor would report the one number the
    sentence cannot use.
    """
    cells = _cells()
    predictor = DMS_PREDICTORS[0]
    table = dms_summary.summarize({"BA.1": cells}, _manifests()).set_index(
        ["predictor", "stratum"]
    )
    for stratum in (dms.SINGLE_NT, dms.MULTI_NT):
        expected = cells.loc[cells["stratum"] == stratum, predictor].median()
        assert table.loc[(predictor, stratum), "median_score"] == pytest.approx(
            expected
        )
    assert table.loc[(predictor, dms.ALL), "median_score"] == pytest.approx(
        cells[predictor].median()
    )
