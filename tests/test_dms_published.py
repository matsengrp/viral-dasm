"""The conversion from each study's published file to this pipeline's columns.

The failure these tests exist for is silent: a conversion that reads the wrong
column, or applies a filter one study wants to another that does not, still
returns a plausible four-column table and still draws a figure. So the checks
here are against the studies themselves -- one against a file XBB.1.5's authors
publish, one against the filtered column KP.3.1.1's authors publish -- rather
than against numbers this repository chose.

Everything but the two "against the study" tests runs on fixtures and needs no
data, so the conversion stays tested off the cluster.
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

import lib
from dms_published import OUTPUT_COLUMNS, normalise_published


def test_the_published_files_are_in_the_repository():
    """Every input this figure reads is a file a reader has after cloning.

    The point of carrying them: `config.yaml` used to name a prepared CSV
    under `/fh/fast`, built by a private repository, so `fig:dms` could not be
    rebuilt off the cluster however public its sources were.
    """
    for homolog in lib.dms_homologs():
        path = lib.dms_dataset(homolog).path
        assert path.is_file(), f"{homolog}: {path} is missing"
        assert REPO_ROOT in path.parents, f"{homolog}: {path} is outside the repo"


@pytest.mark.parametrize("homolog", ["BA.1", "XBB.1.5", "KP.3.1.1"])
def test_each_conversion_emits_the_four_columns_and_nothing_else(homolog):
    """The loader reads four columns by name; a fifth would ride along unseen."""
    dataset = lib.dms_dataset(homolog)
    out = normalise_published(pd.read_csv(dataset.path), dataset)
    assert tuple(out.columns) == OUTPUT_COLUMNS
    assert len(out), homolog
    assert out["functional_effect"].notna().all()
    assert (out["wildtype"] != out["mutant"]).all()
    assert out["site"].dtype.kind == "i"
    assert out.equals(out.sort_values(["site", "mutant"], kind="stable"))


def test_the_xbb_conversion_reproduces_the_summary_its_authors_publish():
    """XBB.1.5's released summary is already filtered, so nothing may move.

    This is the control for the other two: the same code path, on a study that
    needs no threshold from us, must return that study's own numbers. It also
    pins the column choice -- `spike mediated entry` rather than the escape or
    ACE2-binding columns beside it in the same file.
    """
    dataset = lib.dms_dataset("XBB.1.5")
    raw = pd.read_csv(dataset.path)
    out = normalise_published(raw, dataset)

    expected = raw.rename(columns={"spike mediated entry": "functional_effect"})
    expected = expected[expected["functional_effect"].notna()]
    expected = expected[expected["wildtype"] != expected["mutant"]]
    assert len(out) == len(expected)
    merged = out.merge(
        expected[["site", "wildtype", "mutant", "functional_effect"]].astype(
            {"site": int}
        ),
        on=["site", "wildtype", "mutant"],
        suffixes=("", "_published"),
    )
    assert len(merged) == len(out)
    assert (merged["functional_effect"] == merged["functional_effect_published"]).all()


def test_the_kp_filter_is_the_one_its_own_authors_apply():
    """KP.3.1.1's thresholds come from that repository, and are checked against it.

    Its `func_effects/averages` table is released unfiltered, so this pipeline
    applies `times_seen >= 2` and `effect_std <= 2.5`, which
    `data/func_effects_config.yml` and `data/summaries_config.yml` declare.
    That those are the right two was established by reproducing the
    `spike mediated entry` column of the summaries that repository publishes
    -- 8,263 substitutions, identical values, recorded in
    `data/dms/PROVENANCE.md`. Those summary files are not carried here, so
    what this test pins is the predicate itself, in both directions: every
    kept substitution clears both thresholds, and every dropped one fails at
    least one. A threshold borrowed from XBB.1.5 (seen in at least three
    variants) or from BA.1 keeps a different set and fails here, which is the
    mistake this test exists for.
    """
    dataset = lib.dms_dataset("KP.3.1.1")
    raw = pd.read_csv(dataset.path)
    out = normalise_published(raw, dataset)

    assert dataset.min_times_seen == 2
    assert dataset.max_effect_std == 2.5

    candidates = raw[
        (raw["wildtype"] != raw["mutant"])
        & raw["effect"].notna()
        & raw["site"].astype(str).str.fullmatch(r"-?\d+")
    ].astype({"site": int})
    kept_keys = set(zip(out["site"], out["wildtype"], out["mutant"]))
    clears = (candidates["times_seen"] >= 2) & (candidates["effect_std"] <= 2.5)
    in_output = [
        key in kept_keys
        for key in zip(candidates["site"], candidates["wildtype"], candidates["mutant"])
    ]
    assert list(clears) == in_output, (
        "the rows kept are exactly the rows clearing the study's thresholds"
    )
    assert len(kept_keys) == int(clears.sum()) > 0


def test_ba1_keeps_exactly_the_mutations_its_study_would_correlate():
    """BA.1's threshold is its own study's, and only the one it declares.

    That repository publishes `muteffects_observed.csv` over every mutation it
    fit and leaves the threshold to the consumer, exactly as KP.3.1.1 does.
    Its own `notebooks/compare_muteffects.ipynb` -- which correlates these
    effects against other datasets, this figure's job -- reads
    ``config["muteffects_plot_kwargs"]["addtl_slider_stats"]["times_seen"]``
    and keeps ``times_seen >= 3``. It declares no bound on `effect_std`, and
    the largest on the current file is 3.54, so inventing one would drop rows
    the authors keep.

    Pinned as the predicate in both directions, like KP.3.1.1's above: a
    threshold borrowed from KP.3.1.1 (seen in at least two variants) keeps a
    different set and fails here.
    """
    dataset = lib.dms_dataset("BA.1")
    raw = pd.read_csv(dataset.path)
    out = normalise_published(raw, dataset)

    assert dataset.min_times_seen == 3
    assert dataset.max_effect_std is None, (
        "that study declares no effect_std bound; one here would drop rows it keeps"
    )

    candidates = raw[
        (raw["wildtype"] != raw["mutant"])
        & raw["effect"].notna()
        & raw["reference_site"].astype(str).str.fullmatch(r"-?\d+")
    ].astype({"reference_site": int})
    kept_keys = set(zip(out["site"], out["wildtype"], out["mutant"]))
    clears = candidates["times_seen"] >= 3
    in_output = [
        key in kept_keys
        for key in zip(
            candidates["reference_site"], candidates["wildtype"], candidates["mutant"]
        )
    ]
    assert list(clears) == in_output, (
        "the rows kept are exactly the rows clearing the study's threshold"
    )
    assert len(kept_keys) == int(clears.sum()) > 0


def test_a_study_without_thresholds_carries_none():
    """A threshold in the spec means "the authors apply this and their file does not".

    XBB.1.5 releases a filtered summary, so a `min_times_seen` appearing on it
    would silently filter a second time, on a column whose meaning in that
    study is not the one BA.1 and KP.3.1.1 give it. It is the only one of the
    three that still releases its own filtering.
    """
    dataset = lib.dms_dataset("XBB.1.5")
    assert dataset.min_times_seen is None
    assert dataset.max_effect_std is None


def test_a_renamed_column_raises_rather_than_reading_the_wrong_quantity():
    """A re-released file with different column names must not be guessed at.

    These studies publish several phenotypes in one table. A conversion that
    fell back to a positional read, or to the first numeric column, would
    correlate the model against antibody escape and report it as cell entry.
    """
    dataset = lib.dms_dataset("XBB.1.5")
    raw = pd.read_csv(dataset.path).rename(
        columns={"spike mediated entry": "spike-mediated entry"}
    )
    with pytest.raises(KeyError, match="spike mediated entry"):
        normalise_published(raw, dataset)


def test_the_conversion_drops_what_the_matrix_loader_cannot_place():
    """Synonymous rows, missing effects and insertion sites, on a fixture.

    All three reach `from_multidms` otherwise, which warns and drops them, and
    `load_dms`'s balance check then raises on the difference -- so this is the
    boundary that keeps that check meaningful.

    A row whose *wildtype* is a stop is kept, deliberately: it is the
    construct's own terminator, a real disagreement with the clade founder,
    and `load_dms` is where that is counted.
    """
    dataset = lib.dms_dataset("KP.3.1.1")
    raw = pd.DataFrame(
        {
            "site": ["10", "10", "16a", "1253", "11"],
            "wildtype": ["A", "A", "C", "*", "D"],
            "mutant": ["V", "A", "V", "Q", "E"],
            "effect": [-1.0, 0.0, -2.0, -3.0, None],
            "effect_std": [0.1, 0.1, 0.1, 0.1, 0.1],
            "times_seen": [5, 5, 5, 5, 5],
        }
    )
    out = normalise_published(raw, dataset)
    assert list(out["site"]) == [10, 1253]
    assert list(out["mutant"]) == ["V", "Q"]


def test_the_threshold_keeps_the_boundary_and_drops_just_below_it():
    """`>=` and `<=`, on the fixture that tells them from `>` and `<`.

    An off-by-one here moves the retained-mutation count by a few hundred and
    nothing else, which no other test in this file would notice.
    """
    dataset = lib.dms_dataset("KP.3.1.1")
    raw = pd.DataFrame(
        {
            "site": ["1", "2", "3", "4"],
            "wildtype": ["A", "A", "A", "A"],
            "mutant": ["C", "D", "E", "F"],
            "effect": [1.0, 2.0, 3.0, 4.0],
            "effect_std": [2.5, 2.6, 0.1, 0.1],
            "times_seen": [2, 2, 1.5, 2],
        }
    )
    out = normalise_published(raw, dataset)
    assert list(out["mutant"]) == ["C", "F"], (
        "boundary values are kept, just-past are not"
    )
