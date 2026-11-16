"""`fig:probabilities`: the displacement panel, and the bars beside it.

The figure's two claims are checkable without rendering anything. Panel A
asserts an identity, so the test is that the identity holds on the cells it
draws. Panel B asserts an ordering, so the test is that the ordering holds in
the table it reads -- and that it reads rather than recomputes.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import dms
import dms_models
import dms_prob
import dms_summary
import lib
import numpy as np
import pandas as pd
import panel_charts as charts
import panel_spec
from build_era_neutral_models import archive_name
from test_dms_models import _DMS_PATHS, needs_dms

#: Turns a skip into a failure on the host that builds the figures, as every
#: other data-gated module in this suite does.
REQUIRE_DATA = os.environ.get("VIRAL_DASM_REQUIRE_DATA") == "1"

#: The scores both panels are built from, one file per (predictor, homolog).
SCORE_PATHS = sorted(panel_spec.DMS_SCORE_DIR.glob("*.parquet"))

#: The era archive panel A rebuilds its probabilities from.
_SCATTER_CLADE = lib.dms_dataset(panel_spec.PROBABILITIES_SCATTER_HOMOLOG).clade
ARCHIVE = (
    lib.ARTIFACT_ROOT
    / "neutral_models"
    / f"{archive_name(lib.GENE, panel_spec.era_of_clade(_SCATTER_CLADE))}.zip"
)

#: The scored grid panel A rebuilds its probabilities from.
SCORED_TABLE = panel_spec.SCORED_DIR / "selection_factors.parquet"

#: The table panel B reads its bars from.
CORRELATIONS = panel_spec.panel_csv_path(
    panel_spec.DMS_SELECTION, panel_spec.DMS_PREDICTOR_CORRELATIONS
)


def _absent() -> list[str]:
    try:
        missing = [] if SCORE_PATHS else [str(panel_spec.DMS_SCORE_DIR)]
        if not CORRELATIONS.is_file():
            missing.append(str(CORRELATIONS))
        # The DMS CSVs too: most checks here are also gated on `needs_dms`, so
        # without them `REQUIRE_DATA=1` would leave the suite green with the
        # data tests skipped -- the exact failure that flag exists to prevent.
        if not all(path.is_file() for path in _DMS_PATHS):
            missing.append("the DMS CSVs")
        for path in (SCORED_TABLE, ARCHIVE):
            if not path.is_file():
                missing.append(str(path))
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
    """Panel A's rows, built the way the notebook builds them.

    Founder-wide, not assay-wide: the panel characterises the model over the
    whole protein, so it draws every single-nucleotide mutation the model was
    trained on rather than the subset some assay measured.
    """
    homolog = panel_spec.PROBABILITIES_SCATTER_HOMOLOG
    _cells, founder = dms_models.homolog_cells(homolog)
    return dms_prob.founder_frame(
        lib.dms_dataset(homolog).clade, founder, SCORED_TABLE, ARCHIVE
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
def test_panel_a_draws_the_whole_protein_not_the_measured_part(
    scatter: pd.DataFrame,
):
    """Founder-wide, and a strict superset of what any assay measured.

    This is the property that makes panel A a statement about the model rather
    than about an assay's coverage, and it is the one most likely to be
    undone by accident -- reverting to `merged_frames` would still draw a
    valid-looking cloud, just of a different population.
    """
    homolog = panel_spec.PROBABILITIES_SCATTER_HOMOLOG
    merged, _ = dms_summary.merged_frames(SCORE_PATHS)
    whole = merged[homolog]
    measured = whole[whole["stratum"] == "single_nt"]

    drawn = set(zip(scatter["aa_site"], scatter["mutant_aa"]))
    assert drawn > set(zip(measured["aa_site"], measured["mutant_aa"]))
    assert len(scatter) > len(measured)

    # Every drawn cell is one base from the founder's codon, and none is
    # synonymous -- the two things "single-nucleotide amino-acid mutation"
    # means, checked against the codon indicator rather than a score's nulls.
    clade = lib.dms_dataset(homolog).clade
    indicator = dms.single_nt_indicator(clade, lib.GENE)
    for site, mutant, parent in zip(
        scatter["aa_site"], scatter["mutant_aa"], scatter["parent_aa"]
    ):
        assert mutant != parent
        assert bool(indicator.loc[mutant, site - 1])


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

    The figure's claim, tested rather than left to a caption. The second
    comparison is a near-tie on BA.1 -- 0.49 against 0.48 -- so the figure
    should not be described as a clean sweep, but the ordering is what the
    committed table says and this pins it.
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
    """The range the `.2f` label assumes, checked against the real values.

    The upper bound is strict and below 1: `.2f` renders 0.996 as "1.00",
    which is three significant figures and reads as a perfect correlation.
    """
    assert (bars["pearson_r"] >= 0.1).all()
    assert (bars["pearson_r"] < 0.995).all()


def test_the_two_axis_domains_are_equal_in_width():
    """Panel A's 45-degree reading depends on this and nothing else.

    `dms_padded_domain`, the repository's usual helper, pads each axis
    independently and so returns *unequal* spans -- which would quietly stop
    the cloud's slope-1 structure reading at 45 degrees, and with it the claim
    that the iso-colour contours run parallel to the diagonal. This is why
    panel A has a helper of its own, and this is what that helper is for.
    """
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [-10.0, -4.0, -3.0]})
    domains = charts.probabilities_equal_span_domains(frame, "x", "y")
    spans = [high - low for low, high in domains.values()]
    assert spans[0] == pytest.approx(spans[1])

    # Each centred on its own column's midrange, not on a shared one: `y` runs
    # far below `x` here, as `log p` does below `log f`, and a shared domain
    # would push one cloud off the panel entirely.
    for column, (low, high) in domains.items():
        midrange = (frame[column].min() + frame[column].max()) / 2
        assert (low + high) / 2 == pytest.approx(midrange)

    # The span is the wider column's, padded -- not each column's own.
    widest = max(frame[column].max() - frame[column].min() for column in ("x", "y"))
    assert spans[0] == pytest.approx(widest * charts.PROBABILITIES_DOMAIN_PAD)

    # Every point fits inside its own domain.
    for column, (low, high) in domains.items():
        assert low <= frame[column].min() and frame[column].max() <= high


def test_the_figure_does_not_share_fig_dms_panel_stems():
    """`DMS_SELECTION` is one directory, so a clash would overwrite a panel.

    `fig:dms_strata` is three figures now, one per assay, so its stems are
    checked against all three rather than against one.
    """
    _slugs = [panel_spec.homolog_slug(homolog) for homolog in lib.dms_homologs()]
    others = set(panel_spec.DMS_PANELS) | {
        stem for slug in _slugs for stem in panel_spec.dms_strata_panels(slug)
    }
    assert not others & set(panel_spec.PROBABILITIES_PANELS)
    assert panel_spec.PROBABILITIES_FIGURE_STEM != panel_spec.DMS_FIGURE_STEM
    assert panel_spec.PROBABILITIES_FIGURE_STEM not in {
        panel_spec.dms_strata_figure_stem(slug) for slug in _slugs
    }


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

    Asserted against the chart the builder actually produces, not against the
    constants it reads: the upstream version of this panel used a smaller mark
    on the argument that a colour encoding needs one, so this is a live choice
    that could be made again without anyone noticing it had diverged from
    `fig:dms_strata`.
    """
    spec = charts.probability_against_fitness(
        _toy_scatter(),
        x="dasm",
        y="dasm_prob",
        colour="neutral_rate",
        homolog="X",
        r=0.5,
    ).to_dict()
    mark = _scatter_layer(spec)["mark"]
    assert mark["size"] == charts.DMS_POINT_SIZE
    assert mark["opacity"] == charts.DMS_POINT_OPACITY


# ---------------------------------------------------------------------------
# Chart-level checks
# ---------------------------------------------------------------------------
#
# The checks above read the columns the panels are built from. These build the
# panels. Without them a figure could silently revert: this one has already
# been drawn in log10 and with a dashed guide line, and both were taken out
# again -- neither of which a check on the underlying columns can see, because
# the columns never change.


def _toy_scatter() -> pd.DataFrame:
    """A frame in the shape panel A draws, obeying the model's identity."""
    f = np.linspace(-4.0, 2.0, 40)
    lam = np.linspace(-3.0, 1.0, 40)
    return pd.DataFrame({"dasm": f, "neutral_rate": lam, "dasm_prob": f + lam - 8.5})


def _toy_bars() -> pd.DataFrame:
    """Three arms over two homologs, in the correlation table's columns."""
    rows = []
    for homolog, base in (("BA.1", 0.60), ("XBB.1.5", 0.55)):
        for offset, predictor in enumerate(panel_spec.PROBABILITIES_BAR_ARMS):
            rows.append(
                {
                    "homolog": homolog,
                    "predictor": predictor,
                    "stratum": "single_nt",
                    "pearson_r": base - 0.05 * offset,
                    "n_mutations": 4000,
                }
            )
    return pd.DataFrame(rows)


def _scatter_layer(spec: dict) -> dict:
    """The points layer of panel A's spec."""
    layers = spec["layer"]
    return next(
        layer
        for layer in layers
        if "encoding" in layer and "color" in layer["encoding"]
    )


def test_panel_a_draws_the_columns_it_says_it_does():
    """x, y and colour are the three natural-log columns, not a transform.

    This is what stops a silent return to log10: the columns are natural logs,
    so a panel drawing anything else is drawing something the identity test
    above never sees.
    """
    spec = charts.probability_against_fitness(
        _toy_scatter(),
        x="dasm",
        y="dasm_prob",
        colour="neutral_rate",
        homolog="X",
        r=0.5,
    ).to_dict()
    encoding = _scatter_layer(spec)["encoding"]
    assert encoding["x"]["field"] == "dasm"
    assert encoding["y"]["field"] == "dasm_prob"
    assert encoding["color"]["field"] == "neutral_rate"
    assert "transform" not in spec


def test_panel_a_has_no_guide_line_and_keeps_unrounded_equal_spans():
    """Two things a redraw could quietly undo.

    The dashed slope-1 guide was removed deliberately. And `nice=False` is
    load-bearing: without it Vega re-rounds the domains, the two spans stop
    being equal, and the panel's 45-degree reading becomes false while still
    looking fine.
    """
    spec = charts.probability_against_fitness(
        _toy_scatter(),
        x="dasm",
        y="dasm_prob",
        colour="neutral_rate",
        homolog="X",
        r=0.5,
    ).to_dict()

    assert not any(
        layer.get("mark", {}).get("type") == "line" for layer in spec["layer"]
    )

    scales = [_scatter_layer(spec)["encoding"][axis]["scale"] for axis in ("x", "y")]
    spans = []
    for scale in scales:
        assert scale["nice"] is False
        assert scale["zero"] is False
        low, high = scale["domain"]
        spans.append(high - low)
    assert spans[0] == pytest.approx(spans[1])


def test_panel_b_anchors_every_group_at_zero_on_one_domain():
    """Bars encode length, so a nonzero baseline misstates every ratio.

    A shared domain is also what makes a bar in one group comparable to the
    bar beside it in another -- the reason the groups are drawn side by side
    at all.
    """
    frame = _toy_bars()
    homologs = ["BA.1", "XBB.1.5"]
    spec = charts.dms_correlation_bars(
        frame, homologs=homologs, arms=panel_spec.PROBABILITIES_BAR_ARMS, bar_height=100
    ).to_dict()

    domains = [
        layer["encoding"]["y"]["scale"]["domain"]
        for group in spec["hconcat"]
        for layer in group["vconcat"][1]["layer"]
        if "y" in layer.get("encoding", {})
    ]
    assert domains, "no bar y scales found in the spec"
    for domain in domains:
        assert domain[0] == 0.0
    assert len({tuple(domain) for domain in domains}) == 1


def test_panel_b_labels_every_bar_to_two_significant_figures():
    """The labels are drawn, and drawn in the format the figure promises."""
    frame = _toy_bars()
    spec = charts.dms_correlation_bars(
        frame,
        homologs=["BA.1", "XBB.1.5"],
        arms=panel_spec.PROBABILITIES_BAR_ARMS,
        bar_height=100,
    ).to_dict()
    # Altair hoists each cell's frame into `spec["datasets"]` and leaves the
    # cell referring to it by name, so the labels have to be resolved rather
    # than read off the layer.
    datasets = spec["datasets"]
    drawn = set()
    for group in spec["hconcat"]:
        cell = group["vconcat"][1]
        text_layers = [
            layer
            for layer in cell["layer"]
            if layer.get("mark", {}).get("type") == "text"
        ]
        assert text_layers, "the bars carry no value labels"
        for layer in text_layers:
            assert layer["encoding"]["text"]["field"] == "_value"
        drawn |= {
            row["_value"] for row in datasets[cell["data"]["name"]] if "_value" in row
        }

    assert drawn, "no bar value labels in the spec"
    assert all(re.fullmatch(r"0\.\d\d", label) for label in drawn), sorted(drawn)


def test_panel_b_refuses_more_homologs_than_it_has_colours():
    """`zip` would otherwise drop the last group without saying so."""
    extra = len(charts.PROBABILITIES_HOMOLOG_COLORS) + 1
    frame = pd.concat(
        [_toy_bars().assign(homolog=f"H{index}") for index in range(extra)],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="only"):
        charts.dms_correlation_bars(
            frame,
            homologs=[f"H{index}" for index in range(extra)],
            arms=panel_spec.PROBABILITIES_BAR_ARMS,
            bar_height=100,
        )
