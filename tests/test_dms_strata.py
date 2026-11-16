"""What `fig:dms_strata` claims about itself, checked against the modules.

`scripts/panel_spec.py` is stdlib-plus-`lib` so `workflow/figures.smk` stays
cheap to parse, which means the figure's predictor ids and stratum names are
hand-copied there. These tie them back to the modules that define them.

The failures here are the quiet kind. A column order that is not a permutation
of the strata draws a figure that is missing a population or shows one twice,
and either reads as a finding. A predictor the table does not carry raises in
the notebook, but only once the rule has run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import lib

import dasm2  # noqa: F401

pd = pytest.importorskip("pandas")

import dms
import panel_charts as charts
import panel_spec


def test_the_columns_are_the_strata_reordered():
    """A permutation, not a different set.

    The figure reads the population before the cut through it, which is the
    reverse of how the table reads, so the order is its own constant. What it
    must not become is a different *set*: a column dropped here is a population
    silently missing from the figure while the table still reports it.
    """
    assert set(panel_spec.DMS_STRATA_COLUMNS) == set(dms.STRATA)
    assert len(panel_spec.DMS_STRATA_COLUMNS) == len(dms.STRATA)
    assert panel_spec.DMS_STRATA_COLUMNS[0] == dms.ALL, (
        "the union is drawn first, so a reader meets the whole cloud before "
        "the split; see DMS_STRATA_COLUMNS"
    )


def test_every_drawn_model_is_a_predictor_the_table_carries():
    """The figure cannot name a predictor `dms_summary` does not compute.

    It reads `r` and `n` from `predictor_correlations.csv` rather than
    recomputing them, so a model missing from `DMS_PREDICTORS` would draw a
    cloud with an empty corner rather than fail.
    """
    drawn = [model for model, _ in panel_spec.DMS_STRATA_MODELS]
    missing = set(drawn) - set(panel_spec.DMS_PREDICTORS)
    assert not missing, f"{sorted(missing)} are drawn but not scored"


def test_every_stratum_has_a_label():
    """`dms_stratum_label` raises on an unlabelled stratum; this finds it here.

    The raise is deliberate -- printing ``single_nt`` on a paper figure is the
    one failure that looks intentional -- but it would land inside the rule.
    """
    for stratum in dms.STRATA:
        assert charts.dms_stratum_label(stratum)


def test_both_halves_have_a_colour_and_the_union_does_not():
    """Hue means accessibility, and only the two halves have one.

    ``all`` is drawn as the two halves layered rather than as a third colour,
    which is what keeps every point under Altair's per-dataset row limit. A
    colour appearing for it would mean the union had become its own layer.
    """
    assert set(charts.DMS_STRATUM_COLORS) == {dms.SINGLE_NT, dms.MULTI_NT}
    assert len(set(charts.DMS_STRATUM_COLORS.values())) == 2


#: The three slugs, spelled out rather than computed. These are what the
#: manuscript's filenames and `\\label`s would carry, so a `homolog_slug` that
#: quietly started returning something else would rename three figures, and a
#: test that derived its expectation from the function could not see it.
EXPECTED_SLUGS = {"BA.1": "ba1", "XBB.1.5": "xbb15", "KP.3.1.1": "kp311"}


def test_the_slugs_are_what_the_manuscript_would_carry():
    """A figure stem leaves the pipeline, so its spelling is pinned here."""
    for homolog, slug in EXPECTED_SLUGS.items():
        produced = panel_spec.homolog_slug(homolog)
        assert produced == slug
        # Asserted on the function's own output, not on the literal above: a
        # dot in a composed stem reaches a filename and a `\\label`.
        assert "." not in produced
    assert set(lib.dms_homologs()) == set(EXPECTED_SLUGS), (
        "config.yaml's homologs moved; the slugs above are what the "
        "manuscript's figure filenames would become"
    )


def test_every_slug_resolves_back_to_its_homolog():
    """`workflow/figures.smk` holds only the wildcard and needs the homolog."""
    roster = lib.dms_homologs()
    for homolog in roster:
        assert (
            panel_spec.homolog_for_slug(panel_spec.homolog_slug(homolog), roster)
            == homolog
        )
    with pytest.raises(ValueError, match="no homolog's slug"):
        panel_spec.homolog_for_slug("not-a-slug", roster)


def test_the_two_panel_stems_match_the_two_models():
    """One image per model row, and the letters follow the stems."""
    for slug in EXPECTED_SLUGS.values():
        panels = panel_spec.dms_strata_panels(slug)
        letters = panel_spec.dms_strata_panel_letters(slug)
        assert len(panels) == len(panel_spec.DMS_STRATA_MODELS)
        assert set(letters.values()) == set(panels)
        assert [
            letter for row in panel_spec.dms_strata_figure_rows(slug) for letter in row
        ] == list(letters)


def test_the_roster_has_no_two_homologs_that_slug_alike():
    """`homolog_slug` strips dots and case, so it is not injective.

    ``KP.3.1.1`` and ``KP.3.11`` both come out ``kp311``. Two such homologs
    would write one figure stem and the second job would overwrite the first,
    with the `slug` wildcard constraint holding the same token twice and
    nothing raising. The slugs are taken from the **roster** here rather than
    from `EXPECTED_SLUGS`, because that is the set that can collide.
    """
    roster = lib.dms_homologs()
    assert set(panel_spec.check_slugs_are_distinct(roster).values()) == set(roster)

    with pytest.raises(ValueError, match="both slug to"):
        panel_spec.check_slugs_are_distinct(["KP.3.1.1", "KP.3.11"])


def test_no_two_assays_share_a_stem():
    """Three figures in one directory, each drawn from a different assay.

    What this catches is a stem builder that **ignores its slug** -- dropping
    the ``_{slug}`` suffix would have three rules write one pair of PNGs, and
    the last to finish would be the figure all three compositions read. The
    distinctness of the slugs themselves is the test above; this one is handed
    three that already differ.
    """
    stems = [
        stem
        for slug in EXPECTED_SLUGS.values()
        for stem in panel_spec.dms_strata_panels(slug)
    ]
    assert len(set(stems)) == len(stems), f"panel stems collide: {stems}"


def test_the_stems_are_the_tracked_tables_beside_them():
    """The one check that pins the stem *format* rather than its shape.

    Every other assertion here is structural -- the letters follow the stems,
    the rows follow the letters -- and all of them would pass if the format
    string changed, while the six committed CSVs became orphans named for a
    spelling nothing produces any more. These files are tracked, so this reads
    them.
    """
    for slug in EXPECTED_SLUGS.values():
        for stem in panel_spec.dms_strata_panels(slug):
            path = panel_spec.panel_csv_path(panel_spec.DMS_SELECTION, stem)
            assert path.exists(), (
                f"{path} is not in the tree, so `dms_strata_panels` no longer "
                f"names the tables `rule dms_strata_panels` wrote"
            )

    figures = [
        panel_spec.dms_strata_figure_stem(slug) for slug in EXPECTED_SLUGS.values()
    ]
    assert len(set(figures)) == len(figures), f"figure stems collide: {figures}"


def _row_spec():
    """One row drawn over a frame small enough to build in the test."""
    frame = pd.DataFrame(
        {
            "dms_cell_entry": [-6.0, -3.0, 0.0, -1.0],
            "dasm": [-5.0, -2.0, 0.5, -1.5],
            "stratum": [dms.SINGLE_NT, dms.SINGLE_NT, dms.MULTI_NT, dms.MULTI_NT],
        }
    )
    return charts.dms_stratum_row(
        frame,
        score_column="dasm",
        score_title=["DASM", "fitness effect"],
        stratum_order=panel_spec.DMS_STRATA_COLUMNS,
        x_domain=[-7.0, 1.0],
    ).to_dict()


def test_the_leftmost_cell_reserves_a_fixed_y_label_width():
    """What makes the two rows' plots start at the same x.

    Vega-Lite sizes a y axis to the widest label it draws, so rows whose
    labels differ in length get different axis bands and their plots do not
    line up -- antiGen's run to ``-16`` where the DASM's run to ``-5``. The
    fix has to be in the chart: padding the images at composition time leaves
    the panel PNGs and the notebook misaligned.

    Pinned on the spec rather than on a rendered width, which would need a
    browser. `tests/test_dms_censor.py`'s style: check the mechanism, not a
    number that moves with the data.
    """
    spec = _row_spec()
    first = spec["hconcat"][0]
    y_axis = first["layer"][0]["encoding"]["y"]["axis"]
    assert "labelExpr" in y_axis, (
        "the leftmost cell draws no labelExpr, so its axis is as wide as its "
        "own labels and the rows will not line up"
    )
    assert f"pad(datum.label, {charts.DMS_Y_LABEL_CHARS}" in y_axis["labelExpr"]
    assert charts.DMS_LABEL_PAD in y_axis["labelExpr"], (
        "padding with an ordinary space leaves the bands unequal by a "
        "fraction of a character; see DMS_LABEL_PAD"
    )


def test_only_the_leftmost_cell_draws_y_labels():
    """The other two share its scale, so repeating them is noise.

    Also the reason the reserved width above is needed once per row rather
    than once per cell.
    """
    spec = _row_spec()
    rest = spec["hconcat"][1:]
    assert rest, "a row with one cell would make this vacuous"
    for cell in rest:
        axis = cell["layer"][0]["encoding"]["y"]["axis"]
        assert axis.get("labels") is False
        assert axis.get("title") is None


def test_the_figure_does_not_share_fig_dms_panel_stems():
    """Two figures in one directory, and neither may claim the other's panels.

    `DMS_SELECTION` is shared, so a stem collision would have one rule
    overwrite the other's PNG and `notebooks/dms.py`'s completeness check pass
    on the wrong picture.
    """
    for slug in EXPECTED_SLUGS.values():
        assert not set(panel_spec.dms_strata_panels(slug)) & set(panel_spec.DMS_PANELS)
        assert panel_spec.dms_strata_figure_stem(slug) != panel_spec.DMS_FIGURE_STEM
