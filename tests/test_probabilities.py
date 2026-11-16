"""`fig:probabilities`: the displacement panel, and the bars beside it.

The figure's two claims are checkable without rendering anything. Panel A
asserts an identity, so the test is that the identity holds on the cells it
draws. Panel B asserts an ordering, so the test is that the ordering holds in
the table it reads -- and that it reads rather than recomputes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import dms_summary
import lib
import numpy as np
import pandas as pd
import panel_charts as charts
import panel_spec
from test_dms_models import needs_dms

#: Turns a skip into a failure on the host that builds the figures, as every
#: other data-gated module in this suite does.
REQUIRE_DATA = os.environ.get("VIRAL_DASM_REQUIRE_DATA") == "1"

#: The scores both panels are built from, one file per (predictor, homolog).
SCORE_PATHS = sorted(panel_spec.DMS_SCORE_DIR.glob("*.parquet"))

#: The table panel B reads its bars from.
CORRELATIONS = panel_spec.panel_csv_path(
    panel_spec.DMS_SELECTION, panel_spec.DMS_PREDICTOR_CORRELATIONS
)


def _absent() -> list[str]:
    try:
        missing = [] if SCORE_PATHS else [str(panel_spec.DMS_SCORE_DIR)]
        if not CORRELATIONS.is_file():
            missing.append(str(CORRELATIONS))
        return missing
    except OSError as exc:
        return [f"{panel_spec.DMS_SCORE_DIR} ({exc})"]


_ABSENT = _absent()

needs_artifacts = pytest.mark.skipif(
    bool(_ABSENT), reason=f"not readable here: {', '.join(_ABSENT)}"
)

if REQUIRE_DATA and _ABSENT:
    raise RuntimeError(
        "VIRAL_DASM_REQUIRE_DATA=1 but these are not readable: " + ", ".join(_ABSENT)
    )


@pytest.fixture(scope="module")
def scatter() -> pd.DataFrame:
    """Panel A's rows, built the way the notebook builds them."""
    merged, _ = dms_summary.merged_frames(SCORE_PATHS)
    return merged[panel_spec.PROBABILITIES_SCATTER_HOMOLOG].dropna(
        subset=["dasm", "dasm_prob", "neutral_rate"]
    )


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    """Panel B's rows, filtered the way the notebook filters them."""
    table = pd.read_csv(CORRELATIONS)
    return table[
        (table["stratum"] == "single_nt")
        & (table["predictor"].isin(panel_spec.PROBABILITIES_BAR_ARMS))
    ]


@needs_dms
@needs_artifacts
def test_panel_a_draws_the_identity_it_claims(scatter: pd.DataFrame):
    """``log p - log f - log lambda`` is one constant over the drawn cells.

    This is the whole content of panel A: the vertical displacement from a
    slope-1 line is the neutral rate, less a per-founder constant. A wrong
    column, a mis-join or a swapped axis all break the constancy.

    ``log Z`` is written nowhere, so it is derived here as the mean residual.
    The tolerance is 1e-5 against an observed spread of ~2.6e-07: the neutral
    rates are float32 and the scores make a Parquet round trip, so 1e-9 would
    fail on arithmetic rather than on a defect.
    """
    residual = scatter["dasm_prob"] - scatter["dasm"] - scatter["neutral_rate"]
    assert np.ptp(residual) < 1e-5

    # And in the base the panel is actually drawn in. Converting all three
    # together is what keeps the guide a 45-degree line: a mixed-base panel
    # would have slope 1/ln(10), and nothing on the picture would say so.
    log10 = residual / np.log(10)
    assert np.ptp(log10) < 1e-5

    # A real `-log Z` and not zero, which is what a column accidentally written
    # as `log f + log lambda` with no normalizer would give.
    assert abs(residual.mean()) > 1.0


@needs_dms
@needs_artifacts
def test_the_colour_is_an_exact_reading_of_the_displacement(
    scatter: pd.DataFrame,
):
    """The gradient panel A promises cannot fail to appear.

    Since ``log Z`` is constant, the vertical residual from *any* slope-1 line
    is an affine function of ``log lambda``, so the iso-colour contours are
    exactly those lines. If this ever stops holding, the panel is drawing
    something other than the identity its caption describes.
    """
    residual = scatter["dasm_prob"] - scatter["dasm"]
    assert residual.corr(scatter["neutral_rate"]) == pytest.approx(1.0, abs=1e-9)


@needs_dms
@needs_artifacts
def test_panel_a_draws_only_cells_one_base_away(scatter: pd.DataFrame):
    """The drawn population is the single-nucleotide stratum, and all of it.

    `dasm_prob` is non-null exactly where the predictor declares itself able to
    speak, so dropping on it should select the same cells the stratum column
    labels -- checked against that column rather than assumed from it.
    """
    merged, _ = dms_summary.merged_frames(SCORE_PATHS)
    whole = merged[panel_spec.PROBABILITIES_SCATTER_HOMOLOG]
    expected = whole[whole["stratum"] == "single_nt"]
    assert len(scatter) == len(expected)
    assert set(zip(scatter["aa_site"], scatter["mutant_aa"])) == set(
        zip(expected["aa_site"], expected["mutant_aa"])
    )


@needs_dms
@needs_artifacts
def test_panel_b_reads_every_arm_on_one_population(bars: pd.DataFrame):
    """Three arms per homolog, each over the same cells.

    A difference between two bars in a group is then the method rather than
    the cell set, which is the only reading that makes the panel honest.
    """
    for homolog in lib.dms_homologs():
        group = bars[bars["homolog"] == homolog]
        assert set(group["predictor"]) == set(panel_spec.PROBABILITIES_BAR_ARMS)
        assert group["n_mutations"].nunique() == 1


@needs_dms
@needs_artifacts
def test_panel_b_shows_the_ordering_the_figure_claims(bars: pd.DataFrame):
    """`dasm` over `dasm_prob` over `antigen`, in every dataset.

    The figure's claim, tested rather than asserted in a caption. Note the
    second comparison is a near-tie on BA.1 (+0.011), so this holds but should
    not be described as a clean sweep.
    """
    for homolog in lib.dms_homologs():
        group = bars[bars["homolog"] == homolog].set_index("predictor")["pearson_r"]
        assert group["dasm"] > group["dasm_prob"], homolog
        assert group["dasm_prob"] >= group["antigen"], homolog


def test_the_bar_format_gives_two_significant_figures():
    """``.2f`` is two significant figures only while ``0.1 <= r < 1``.

    Every value the figure draws is in that range, which is what makes the
    format honest. A correlation below 0.1 would render as ``0.07`` -- one
    significant figure -- so this pins the assumption rather than the values.
    """
    assert charts.PROBABILITIES_BAR_FORMAT == ".2f"
    assert f"{0.6810401:{charts.PROBABILITIES_BAR_FORMAT}}" == "0.68"
    assert f"{0.4785118:{charts.PROBABILITIES_BAR_FORMAT}}" == "0.48"


@needs_dms
@needs_artifacts
def test_every_drawn_correlation_is_in_the_two_figure_range(bars: pd.DataFrame):
    """The range the `.2f` label assumes, checked against the real values."""
    assert bars["pearson_r"].between(0.1, 1.0).all()


def test_the_two_axis_domains_are_equal_in_width():
    """Panel A's 45-degree reading depends on this and nothing else.

    `dms_padded_domain`, the repository's usual helper, pads each axis
    independently and so returns *unequal* spans -- which would quietly make
    the slope-1 guide not a 45-degree line. This is why panel A has a helper
    of its own, and this is the property that helper exists for.
    """
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [-10.0, -4.0, -3.0]})
    domains = charts.probabilities_equal_span_domains(frame, "x", "y")
    spans = [high - low for low, high in domains.values()]
    assert spans[0] == pytest.approx(spans[1])

    # Each centred on its own column, not shared: a shared domain would put one
    # cloud off the panel entirely.
    assert domains["x"][0] > domains["y"][1] - spans[1]


def test_the_figure_does_not_share_fig_dms_panel_stems():
    """`DMS_SELECTION` is one directory, so a clash would overwrite a panel."""
    others = set(panel_spec.DMS_PANELS) | set(panel_spec.DMS_STRATA_PANELS)
    assert not others & set(panel_spec.PROBABILITIES_PANELS)
    assert (
        panel_spec.PROBABILITIES_FIGURE_STEM != panel_spec.DMS_FIGURE_STEM
        and panel_spec.PROBABILITIES_FIGURE_STEM != panel_spec.DMS_STRATA_FIGURE_STEM
    )


def test_each_experiment_has_its_own_colour_and_each_arm_its_own_label():
    """Colour carries the experiment; the labels carry the arm.

    Panel B groups by DMS experiment and names the three arms under every bar,
    so the colour channel is spent on the grouping rather than repeating a
    distinction the labels already make. That needs one hue per homolog, and
    one label per arm.
    """
    assert len(set(charts.PROBABILITIES_HOMOLOG_COLORS)) == len(
        charts.PROBABILITIES_HOMOLOG_COLORS
    )
    assert len(charts.PROBABILITIES_HOMOLOG_COLORS) >= len(lib.dms_homologs())
    assert set(panel_spec.PROBABILITIES_BAR_ARMS) == set(
        charts.PROBABILITIES_ARM_LABELS
    )


def test_the_scatter_mark_matches_the_other_dms_figures():
    """One dot size and one opacity across the paper's scatters.

    Upstream's version of this panel used a smaller mark; these figures are a
    set first. Pinned so the panel cannot drift away from `fig:dms_strata`
    without someone deciding to.
    """
    assert charts.PROBABILITIES_POINT_SIZE == charts.DMS_POINT_SIZE
    assert charts.PROBABILITIES_POINT_OPACITY == charts.DMS_POINT_OPACITY
