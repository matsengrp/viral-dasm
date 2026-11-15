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

import lib  # noqa: F401

import dasm2  # noqa: F401

pytest.importorskip("pandas")

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


def test_the_two_panel_stems_match_the_two_models():
    """One image per model row, and the letters follow the stems."""
    assert len(panel_spec.DMS_STRATA_PANELS) == len(panel_spec.DMS_STRATA_MODELS)
    assert set(panel_spec.DMS_STRATA_PANEL_LETTERS.values()) == set(
        panel_spec.DMS_STRATA_PANELS
    )
    assert [letter for row in panel_spec.DMS_STRATA_FIGURE_ROWS for letter in row] == (
        list(panel_spec.DMS_STRATA_PANEL_LETTERS)
    )


def test_the_figure_does_not_share_fig_dms_panel_stems():
    """Two figures in one directory, and neither may claim the other's panels.

    `DMS_SELECTION` is shared, so a stem collision would have one rule
    overwrite the other's PNG and `notebooks/dms.py`'s completeness check pass
    on the wrong picture.
    """
    assert not set(panel_spec.DMS_STRATA_PANELS) & set(panel_spec.DMS_PANELS)
    assert panel_spec.DMS_STRATA_FIGURE_STEM != panel_spec.DMS_FIGURE_STEM
