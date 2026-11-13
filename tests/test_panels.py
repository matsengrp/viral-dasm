"""What a wrong agreement panel would show, and what catches it.

Every failure mode here returns a full-looking panel. A null fitness column
read as a zero effect, a shift join that compares D614G against E614G, a
two-layer site exclusion believed to be three: each draws a picture with the
right shape and the wrong numbers, and none of them raises.

The checks over a hand-built table run always. The ones that need this
repository's own censor audit -- the reverting exclusion, which is the layer
the scored table does not carry -- skip when it is absent, and
``VIRAL_DASM_REQUIRE_AUDIT=1`` turns that skip into a failure.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import lib

import dasm2  # noqa: F401

pd = pytest.importorskip("pandas")
numpy = pytest.importorskip("numpy")

import panels
from runs import MODELS, RUNS

from dasm2.analysis.viral_fitness_eval import COUNT_THRESHOLD_LADDER
from dasm2.sequence.masking import censored_codon_sites, load_censor_table

#: The canary clade for the reverting exclusion. Its reverting sites are the
#: largest set that neither of the two layers the table *does* carry covers, so
#: it is where substituting `censored_directional` for the audit would go
#: furthest wrong while still producing a panel.
CANARY_CLADE = "21L"

#: Two models and three clades is the smallest table that exercises everything
#: the panels do: a model axis to group by, a clade the shift panel can pair,
#: and a third clade to drop.
FAKE_MODELS = ("dasm-1077k", "ci-484k")
FAKE_CLADES = ("21J", "21L", "23A")

#: Enough mutations per clade to clear the adequacy floor at every rung, so a
#: test that means to measure the filter is not measuring the floor.
FAKE_SITES = 60


def fake_table(seed: int = 0) -> pd.DataFrame:
    """A scored table in the shape ``rule score`` writes.

    A left join from the model grid, so most cells carry a null fitness column:
    that is the state the null drop exists for, and a table without it would
    let the drop be a no-op and still pass.
    """
    rng = numpy.random.default_rng(seed)
    rows = []
    for model in FAKE_MODELS:
        for clade_index, clade in enumerate(FAKE_CLADES):
            for site in range(1, FAKE_SITES + 1):
                for mutant in ("A", "G", "V"):
                    parent = "A" if site % 7 else "E"
                    estimated = (site + clade_index) % 4 != 3
                    log_sf = float(rng.normal(-1.0, 1.5))
                    rows.append(
                        {
                            "clade": clade,
                            "model": model,
                            "arm": "full_depth",
                            "aa_mutation": f"{parent}{site}{mutant}",
                            "aa_site": site,
                            "parent_aa": parent,
                            "mutant_aa": mutant,
                            "is_synonymous": mutant == parent,
                            "log_selection_factor": log_sf,
                            "actual_count": (
                                float(rng.integers(0, 80)) if estimated else None
                            ),
                            "predicted_count": (
                                float(rng.integers(0, 80)) if estimated else None
                            ),
                            "delta_fitness_ratio": (
                                log_sf + float(rng.normal(0, 0.5))
                                if estimated
                                else None
                            ),
                            "delta_fitness_bayesian": (log_sf if estimated else None),
                            "uncertainty": 0.1 if estimated else None,
                            "globally_masked": site == 1,
                            "censored": site == 2,
                        }
                    )
    return pd.DataFrame(rows)


def written_table(directory: Path, frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` through Parquet and back through the real reader.

    Every test reads its table this way rather than taking ``fake_table``
    directly, so :func:`panels.read_scored_table` is on the tested path. It is
    what renames ``parent_aa`` to ``clade_founder_aa``, and that rename is a
    silent one to lose: the shift join's wildtype guard skips itself when the
    column is absent rather than raising.
    """
    path = directory / "selection_factors.parquet"
    frame.to_parquet(path, index=False)
    return panels.read_scored_table(path, models=FAKE_MODELS)


@pytest.fixture(scope="module")
def table(tmp_path_factory) -> pd.DataFrame:
    return written_table(tmp_path_factory.mktemp("scored"), fake_table())


@pytest.fixture(scope="module")
def reverting() -> dict[str, set[int]]:
    """A different reverting site per clade.

    Different per clade on purpose: with one shared set, a lookup hoisted out of
    `base_frames`' clade loop -- which would score every clade against the first
    clade's sites -- produces exactly the same frames, and nothing here would
    notice.
    """
    return {clade: {3 + index} for index, clade in enumerate(FAKE_CLADES)}


@pytest.fixture(scope="module")
def frames(table, reverting) -> dict[tuple[str, str], pd.DataFrame]:
    return panels.base_frames(
        table,
        reverting_by_clade=reverting,
        models=FAKE_MODELS,
        clades=FAKE_CLADES,
    )


def test_null_fitness_rows_are_dropped(table, frames):
    """The null drop is the inner join, counted independently.

    `rule score` writes a left join from the model grid, so a cell mat-pcp never
    estimated carries a null fitness column. Those rows are not zero-effect
    mutations -- counted as such they would pull every correlation toward the
    model's own mean -- and a correlation over them is undefined.

    Counted from the table rather than from the chain that produced the frames:
    a drop that silently did nothing would otherwise pass.
    """
    rows = table[
        (table["model"] == FAKE_MODELS[0])
        & table["delta_fitness_ratio"].notna()
        & ~table["is_synonymous"]
        & ~table["aa_site"].isin({1, 2})
    ]
    expected = sum(
        len(frame[frame["aa_site"] != 3 + index])
        for index, (_, frame) in enumerate(rows.groupby("clade", sort=False))
    )
    produced = sum(
        len(frame) for (model, _), frame in frames.items() if model == FAKE_MODELS[0]
    )
    assert produced == expected
    assert expected > 0, "the fixture produced no estimated rows to correlate"


def test_every_layer_of_the_site_exclusion_fires(frames, reverting):
    """All three site classes come out, and each is a different site.

    Site 1 is globally masked, site 2 is censored, and one site per clade
    reverts. The first two are columns of the table and the third is not, so a
    panel that read only the table would keep the reverting site -- a two-layer
    exclusion believing it ran three.

    Checked per clade against that clade's own set, and the other clades'
    reverting sites are asserted *present*: a lookup hoisted out of
    `base_frames`' loop would drop the first clade's site everywhere, which
    reads as a stricter filter rather than as a wrong one.
    """
    for (_, clade), frame in frames.items():
        assert not frame["globally_masked"].any()
        assert not frame["censored"].any()
        sites = set(frame["aa_site"])
        assert not (sites & reverting[clade])
        elsewhere = set.union(*(reverting[other] for other in FAKE_CLADES))
        assert elsewhere - reverting[clade] <= sites, (
            f"{clade} lost a site that reverts in another clade, which is what "
            f"a reverting lookup hoisted out of the clade loop would do."
        )


def test_synonymous_rows_are_excluded(frames):
    """Their log SF is a structural zero rather than a prediction."""
    for frame in frames.values():
        assert not frame["is_synonymous"].any()


def test_within_clade_metrics_emits_every_published_column(frames):
    """The five columns of the published within-clade table.

    `viral.py` carries a same-named sibling, `compute_within_clade_r_summary`,
    which emits `clade, pearson_r, n_mutations` and nothing else. It is the
    function a reader would reach for and it cannot reproduce the published
    table, so the column set is pinned here rather than assumed.
    """
    metrics = panels.within_clade_metrics(
        panels.by_clade(frames, FAKE_MODELS[0]), "delta_fitness_ratio"
    )
    assert list(metrics.columns) == [
        "clade",
        "pearson_r",
        "spearman_rho",
        "regression_slope",
        "n_mutations",
    ]
    assert len(metrics) == len(FAKE_CLADES)


def test_the_cutoff_and_the_matching_rung_agree(frames):
    """The two notebooks' readouts, where they overlap.

    The per-mutation panels apply one cutoff to the base frames; the ladder
    panels apply each rung to the same frames. At the rung that equals the
    cutoff they must be the same rows, which is what makes a dot in panel C and
    the cloud behind it in panel B the same mutations.

    ``n_mutations`` exactly, because a different number of mutations is a bug
    however small the gap; the correlations to 1e-10, which is float noise in
    two orderings of the same sum rather than a tolerance for disagreement.
    """
    # The two invariants behind the claim, which the comparison below assumes
    # rather than shows: the per-mutation panels' cutoff is a rung of the
    # ladder, and the base frames carry no count filter for either to inherit.
    assert panels.PANEL_MIN_COUNT in panels.COUNT_LADDER
    assert panels.BASE_SPEC.min_count == 0

    base = panels.by_clade(frames, FAKE_MODELS[0])
    at_cutoff = {
        clade: panels.apply_count_threshold(frame, panels.PANEL_MIN_COUNT)
        for clade, frame in base.items()
    }
    single = panels.within_clade_metrics(at_cutoff, "delta_fitness_ratio")
    ladder = panels.within_clade_count_ladder(
        base,
        "delta_fitness_ratio",
        sorted(base),
        thresholds=[panels.PANEL_MIN_COUNT],
    )
    merged = single.merge(ladder, on="clade", suffixes=("_single", "_ladder"))
    assert len(merged) == len(FAKE_CLADES)
    assert (merged["n_mutations_single"] == merged["n_mutations_ladder"]).all()
    for column in ("pearson_r", "spearman_rho", "regression_slope"):
        assert (
            (merged[f"{column}_single"] - merged[f"{column}_ladder"]).abs() < 1e-10
        ).all()


def test_the_shift_join_refuses_a_mismatched_wildtype(tmp_path, reverting):
    """D614G in one clade is never compared with E614G in another.

    The guard keys on ``clade_founder_aa`` and **silently does nothing when that
    column is absent**. The scored table spells the founder residue
    ``parent_aa``, so a reader that forgot the rename would pair every mutation
    at a site whose founder differs between the clades, report more shared
    mutations than exist, and raise nothing.
    """
    moved = fake_table()
    at_site_seven = (moved["clade"] == "21L") & (moved["aa_site"] == 7)
    moved.loc[at_site_seven, "parent_aa"] = "W"
    # Through the reader, which is what supplies `clade_founder_aa` at all. A
    # hand-assigned column here would test the guard and not the rename.
    read = written_table(tmp_path, moved)
    assert (read["clade_founder_aa"] == read["parent_aa"]).all()
    frames = panels.base_frames(
        read, reverting_by_clade=reverting, models=[FAKE_MODELS[0]], clades=FAKE_CLADES
    )
    shift = panels.shift_frame(
        frames[(FAKE_MODELS[0], "21J")],
        frames[(FAKE_MODELS[0], "21L")],
        "21J",
        "21L",
        "delta_fitness_ratio",
    )
    assert 7 not in set(shift["aa_site"])
    assert len(shift) > 0, "the fixture paired nothing at all, so this proves nothing"


def test_identical_founders_drop_the_pair_rather_than_scoring_it(frames):
    """A pair the model cannot answer is omitted, not entered at r = 0.

    Two clades with the same founder are the same input to the model twice, so
    its ΔSF is identically zero while mat-pcp's Δfitness still moves with each
    clade's own tree counts. Scored, such a pair enters at zero and drags the
    median toward the baseline's floor for a reason that has nothing to do with
    the model.
    """
    founder = "A" * 10
    identical = {clade: founder for clade in FAKE_CLADES}
    assert panels.identical_founder_pairs(identical, list(FAKE_CLADES)) == [
        ("21J", "21L"),
        ("21J", "23A"),
        ("21L", "23A"),
    ]
    scored = panels.shift_pair_metrics(
        panels.by_clade(frames, FAKE_MODELS[0]),
        list(FAKE_CLADES),
        "delta_fitness_ratio",
        founders=identical,
    )
    assert scored.empty

    # And the mixed case, which an empty frame cannot distinguish from a drop
    # that fired for some other reason: one pair shares a founder, the rest do
    # not, and only that pair is missing.
    mixed = dict.fromkeys(FAKE_CLADES, founder)
    mixed["23A"] = "C" + founder[1:]
    partial = panels.shift_pair_metrics(
        panels.by_clade(frames, FAKE_MODELS[0]),
        list(FAKE_CLADES),
        "delta_fitness_ratio",
        founders=mixed,
    )
    assert set(partial["pair_label"]) == {"21J\u219223A", "21L\u219223A"}


def test_reverting_sites_raise_for_an_unmeasured_clade(reverting):
    """Absent means unmeasured, not "reverts nowhere".

    An empty set would read as the second, and would quietly widen the row set
    of every clade the audit never saw.
    """
    with pytest.raises(KeyError, match="unmeasured rather than empty"):
        panels.reverting_sites(reverting, "99Z")


def test_a_stale_audit_is_refused(tmp_path):
    """An audit without the key fails rather than excluding nothing."""
    path = tmp_path / "S.json"
    path.write_text(json.dumps({"something_else": {}}))
    with pytest.raises(ValueError, match="reverting_codon_sites_by_clade"):
        panels.reverting_sites_by_clade(path)
    with pytest.raises(FileNotFoundError):
        panels.reverting_sites_by_clade(tmp_path / "absent.json")


def audit_or_skip() -> dict[str, set[int]]:
    """This repository's own censor audit and the censor table, or skip.

    Both, not just the audit: the canary compares the two, and a host with the
    audit but no mat-pcp mount would otherwise error in the middle of the test
    rather than say what is missing.

    ``VIRAL_DASM_REQUIRE_AUDIT=1`` turns the skip into a failure, as
    ``VIRAL_DASM_REQUIRE_CHECKPOINT`` does in `tests/test_score_run.py`. Set it
    at merge time: the skip is right while a campaign is mid-flight, and after
    it lands a silent ``s`` is the one outcome that looks like a pass.
    """
    missing = [
        str(path)
        for path in (
            panels.AUDIT_JSON,
            lib.mat_pcp_path("censored_mutations_file"),
            lib.mat_pcp_path("censored_clades_file"),
        )
        if not path.is_file()
    ]
    if missing:
        message = f"not readable here: {', '.join(missing)}"
        if os.environ.get("VIRAL_DASM_REQUIRE_AUDIT") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return panels.reverting_sites_by_clade()


def test_censored_directional_is_not_the_reverting_exclusion():
    """The canary: what substituting the table's column would wrongly admit.

    ``censored`` is ``codon_sites_of(site_nt_indices | directional[:, 0])`` and
    ``censored_directional`` is ``codon_sites_of(directional[:, 0])``, so the
    second is a subset of the first **by construction** -- substituting it adds
    exactly zero exclusions. What that wrongly admits in this clade is the
    reverting sites that neither layer in the table covers, and the number is
    pinned so a change to the audit or the censor table fails here rather than
    moving a published row set silently.
    """
    by_clade = audit_or_skip()
    censor_table = load_censor_table(
        lib.mat_pcp_path("censored_mutations_file"),
        lib.mat_pcp_path("censored_clades_file"),
    )
    spec = censor_table[CANARY_CLADE]
    censored = censored_codon_sites(spec)
    directional = {int(index) // 3 + 1 for index in spec.directional[:, 0]}
    assert directional <= censored, (
        "censored_directional is meant to be a subset of censored by "
        "construction; if it is not, the substitution argument has changed."
    )
    admitted = by_clade[CANARY_CLADE] - censored
    assert len(admitted) == 24, (
        f"{CANARY_CLADE} has {len(admitted)} reverting sites that `censored` "
        f"does not cover, against the 24 measured when #19 was written. The "
        f"audit or the censor table moved; re-measure rather than re-pinning."
    )


def test_the_ladder_is_the_library_ladder():
    """`scripts/panel_spec.py`'s rungs against the ones the filter applies.

    The spec module is stdlib-only on purpose -- `workflow/figures.smk` imports
    it to declare outputs, and a `.smk` is parsed on every snakemake invocation
    -- so it restates the rungs as a literal rather than importing them. This is
    what stops that literal drifting from the ladder `apply_count_threshold`
    actually filters on, which would sweep one axis and label it another.
    """
    assert panels.COUNT_LADDER == list(
        COUNT_THRESHOLD_LADDER[: len(panels.COUNT_LADDER)]
    )
    assert panels.PANEL_MIN_COUNT == panels.COUNT_LADDER[0]


def test_panel_selections_name_runs_the_pipeline_trains():
    """A selection cannot name a model, or an arm, that has no run.

    `run_id` raises on an unknown pair, but nothing reaches it from a panel
    rule: the selections are read straight out of the table, so a typo would
    produce an empty frame and an empty panel rather than an error.
    """
    for selection, models in panels.SELECTIONS.items():
        assert models, f"selection {selection!r} names no models"
        for model in models:
            assert model in MODELS, f"{model!r} is not one of {sorted(MODELS)}"
            assert (model, panels.ARM) in RUNS, (
                f"({model!r}, {panels.ARM!r}) is not one of the seven runs"
            )
    assert panels.DEFAULT_SELECTION in panels.SELECTIONS
    # The published figure is one model, not a pooled median: panels C and E
    # pool whatever their selection names, so a second model here would change
    # what `fig:agreement` shows without changing any code.
    assert len(panels.SELECTIONS[panels.DEFAULT_SELECTION]) == 1


# The composition. A wrong layout here is a figure that looks finished: a
# panel silently not placed, letters that no longer match the caption, or
# type that drifted from the panels it labels. None of it raises.


def test_the_layout_places_every_panel_exactly_once():
    """A letter missing from the rows is a panel silently not drawn.

    ``figure_layout.compose`` takes the panels and the rows separately and
    draws whatever the rows name, so a letter left out of ``FIGURE_ROWS``
    produces a complete-looking figure with one panel gone -- and the rule
    still succeeds, because its outputs are the composed files rather than
    the count of panels in them.
    """
    import panel_spec as spec

    placed = [letter for row in spec.FIGURE_ROWS for letter in row]
    assert sorted(placed) == sorted(spec.PANEL_LETTERS), (
        f"the rows place {sorted(placed)} but the figure has panels "
        f"{sorted(spec.PANEL_LETTERS)}. A letter in one and not the other is "
        f"either a panel drawn and not placed, or a placement with no panel."
    )
    assert len(placed) == len(set(placed)), (
        f"a letter is placed twice: {placed}. The same panel would be drawn "
        f"in two cells of the figure."
    )


def test_the_composed_letters_are_the_panels_the_rules_write():
    """The caption's letters and the rules' outputs are one set.

    These are two spellings of the same five panels -- ``MUTATION_PANELS``
    and ``LADDER_PANELS`` are what the two rules declare as outputs, and
    ``PANEL_LETTERS`` is what the composition reads. A stem added to one and
    not the other is a panel that is drawn, declared and never composed.
    """
    import panel_spec as spec

    assert set(spec.PANEL_LETTERS.values()) == set(
        spec.MUTATION_PANELS + spec.LADDER_PANELS
    ), (
        f"the composition reads {sorted(spec.PANEL_LETTERS.values())} and the "
        f"rules write {sorted(spec.MUTATION_PANELS + spec.LADDER_PANELS)}."
    )


def test_the_panel_letters_are_sized_against_the_type_the_panels_draw():
    """The letters are sized from the panels' own font constants.

    ``panel_spec`` is stdlib plus ``lib`` on purpose -- ``workflow/
    figures.smk`` imports it on every invocation -- so it cannot import
    ``panel_charts`` to read these, and restates them. Restated numbers
    drift: a panel font bumped in ``panel_charts`` alone would leave the
    letters sized against type the figure no longer carries, which is a
    figure whose labels are quietly the wrong size rather than one that
    fails.
    """
    import figure_layout as fl
    import panel_charts
    import panel_spec as spec

    assert spec.PANEL_TITLE_PX == panel_charts.PANEL_FONT_SIZE, (
        f"panel_spec says the panels' titles are {spec.PANEL_TITLE_PX} px and "
        f"panel_charts draws them at {panel_charts.PANEL_FONT_SIZE}."
    )
    assert fl.VEGA_WRITE_PPI == panel_charts.VEGA_WRITE_PPI, (
        f"figure_layout converts spec px to disk px at "
        f"{fl.VEGA_WRITE_PPI} ppi and panel_charts saves at "
        f"{panel_charts.VEGA_WRITE_PPI}. The letters would be sized against "
        f"a panel rendered at a different resolution."
    )


def test_the_composition_fits_the_page_box_and_keeps_the_panels_in_proportion():
    """The solved dpi is a fit, not a setting, and it resamples nothing.

    ``compose`` picks the largest dpi whose figure still fits the box, so the
    two things worth asserting are that the result *is* inside the box and
    that the panels kept their relative sizes. A layout that quietly stretched
    one panel to fill its row would pass every file-level check in the rule.
    """
    import figure_layout as fl
    import panel_spec as spec

    # Panels of deliberately unequal shape, so a stretch is visible as a
    # changed ratio rather than hidden by squares.
    shapes = {
        "A": (400, 700),
        "B": (400, 500),
        "C": (400, 500),
        "D": (600, 900),
        "E": (300, 600),
    }
    panels_in = {
        letter: numpy.zeros((h, w, 4), dtype=float) for letter, (h, w) in shapes.items()
    }
    composed = fl.compose(
        panels_in,
        [list(row) for row in spec.FIGURE_ROWS],
        box_in=spec.BOX_IN,
        scale=spec.FIGURE_SCALE,
        title_px_on_disk=fl.vega_title_px_on_disk(title_px=spec.PANEL_TITLE_PX),
    )
    box_w, box_h = (
        spec.BOX_IN[0] * spec.FIGURE_SCALE,
        spec.BOX_IN[1] * spec.FIGURE_SCALE,
    )
    assert composed.width_in <= box_w + 1e-9 and composed.height_in <= box_h + 1e-9, (
        f"the composition is {composed.width_in:.3f} x {composed.height_in:.3f} in, "
        f"outside the {box_w:.3f} x {box_h:.3f} in box it solved its dpi against."
    )
    # Every panel lands at the one solved dpi, so a panel's inches are its
    # pixels over that dpi and the ratio between any two is their pixel ratio.
    placed = {
        letter: (w / composed.dpi, h / composed.dpi)
        for letter, (h, w) in shapes.items()
    }
    assert placed["D"][0] / placed["A"][0] == pytest.approx(900 / 700), (
        "D and A no longer stand in their pixel ratio, so a panel was resized "
        "relative to another rather than the whole figure being scaled."
    )


def test_save_figure_writes_every_format_the_rule_declares(tmp_path):
    """The declared outputs and the written files are one list.

    A format in ``COMPOSED_FORMATS`` that ``save_figure`` does not write
    fails the rule outright, which is fine. The quiet direction is the other
    one: a file written but not declared is one snakemake will not clean up
    after a failed job, so a half-written PDF would survive the rerun that
    replaced its PNG -- and that PDF is what the manuscript includes.
    """
    import figure_layout as fl
    import panel_spec as spec

    composed = fl.compose(
        {
            letter: numpy.zeros((40, 60, 4), dtype=float)
            for letter in spec.PANEL_LETTERS
        },
        [list(row) for row in spec.FIGURE_ROWS],
        box_in=spec.BOX_IN,
        scale=spec.FIGURE_SCALE,
    )
    written = fl.save_figure(composed, tmp_path, spec.FIGURE_STEM, tex_figure_dir=None)
    assert sorted(p.suffix.lstrip(".") for p in written) == sorted(
        spec.COMPOSED_FORMATS
    ), (
        f"save_figure wrote {[p.name for p in written]} and the rule declares "
        f"{sorted(spec.COMPOSED_FORMATS)}."
    )
    for path in written:
        assert path.stat().st_size > 0, f"{path.name} was written empty."


def test_the_manuscript_copy_is_skipped_rather_than_invented(tmp_path):
    """An absent manuscript is reported, not created.

    ``save_figure`` keys on the *parent* of ``tex_figure_dir`` existing, so a
    clone that is not checked out is a skip and a clone that is gets its
    ``figures/`` created. Were it to key on the directory itself, a first copy
    into a fresh clone would be silently skipped; were it to mkdir the parent,
    a typo in ``config.yaml`` would grow a stray directory beside the
    repository and report a copy that no reader will find.
    """
    import figure_layout as fl
    import panel_spec as spec

    composed = fl.compose(
        {
            letter: numpy.zeros((40, 60, 4), dtype=float)
            for letter in spec.PANEL_LETTERS
        },
        [list(row) for row in spec.FIGURE_ROWS],
        box_in=spec.BOX_IN,
        scale=spec.FIGURE_SCALE,
    )
    absent = tmp_path / "no-such-clone" / "figures"
    written = fl.save_figure(composed, tmp_path / "out", "x", tex_figure_dir=absent)
    assert not absent.exists(), "a copy was made into a manuscript that is not here"
    assert all(p.parent == tmp_path / "out" for p in written)

    present = tmp_path / "clone" / "figures"
    present.parent.mkdir()
    written = fl.save_figure(composed, tmp_path / "out2", "x", tex_figure_dir=present)
    assert (present / "x.pdf").is_file(), (
        "the manuscript is checked out and its figures/ did not get the PDF"
    )


# Three ways this port drifted from the figure in the manuscript. Each drew a
# complete, plausible panel, and the first two were only visible once the
# whole figure was composed. They are pinned here against the notebooks
# `main.tex`'s provenance block names as the published figure's source.


def _marks(spec) -> set[str]:
    """Every mark type in a Vega-Lite spec, at any depth."""
    found: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            mark = node.get("mark")
            if isinstance(mark, str):
                found.add(mark)
            elif isinstance(mark, dict) and "type" in mark:
                found.add(mark["type"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(spec)
    return found


def test_panel_a_does_not_shade_the_count_noise_interval():
    """The published panel draws no band, and the machinery for one exists.

    `with_count_noise_interval` computes the interval and the panel's CSV
    carries it, so passing `interval_columns` is one keyword away and draws a
    grey band that looks like a considered choice. It is not: the band widens
    the y domain to cover the thinnest clade's Poisson tail, which costs the
    two series the contrast the panel exists to show, and makes panel A
    taller than B and C beside it in the composed row. Upstream's checkbox
    for it defaults to off and the manuscript figure was saved with it off.
    """
    import panel_charts as charts

    frame = pd.DataFrame(
        {
            "clade": ["21J", "21L", "23A"],
            "delta_fitness": [0.8, -1.2, 2.6],
            "log_sf": [0.78, -2.5, 2.1],
            "count_noise_lo": [0.5, -1.7, 2.1],
            "count_noise_hi": [1.0, -0.6, 3.0],
        }
    )
    banded = charts.mutation_lookup_line(
        frame,
        "F456L",
        interval_columns=("count_noise_lo", "count_noise_hi"),
    ).to_dict()
    assert "area" in _marks(banded), (
        "the band is meant to be one keyword away -- if this fails, "
        "`interval_columns` no longer draws one and the test below proves "
        "nothing."
    )

    plain = charts.mutation_lookup_line(frame, "F456L").to_dict()
    assert "area" not in _marks(plain), (
        f"panel A drew a shaded band; marks are {sorted(_marks(plain))}. The "
        f"figure in the manuscript has none."
    )


def test_panel_a_is_sized_as_the_published_panel_is():
    """Panel A's height is a property of the whole page, not of panel A.

    `FIGURE_ROWS` puts A beside B and C, and `compose` solves one dpi for the
    figure, so a taller panel A does not make itself bigger -- it pushes the
    row down, forces a smaller dpi, and shrinks the type on *every* panel of
    the figure. That is a layout bug with no local symptom: panel A alone
    looks fine, panel A's rule succeeds, and only the composed figure shows
    it. This port had it, at 420x350 against the published 0.8x336x232.

    Pinned as the two constants rather than as a rendered size, because what
    a panel measures on disk also depends on how many clades it places and
    how wide its tick labels are, neither of which is a decision.
    """
    import panel_charts as charts
    import panel_spec as spec

    row = next(row for row in spec.FIGURE_ROWS if "A" in row)
    assert set(row) == {"A", "B", "C"}, (
        f"panel A's row is {row}; this test's reasoning is about the row it "
        f"shares, so it needs rewriting for the new layout."
    )
    assert (charts.LOOKUP_PANEL_SCALE, charts.LOOKUP_PANEL_HEIGHT) == (0.8, 290), (
        f"panel A is sized at {charts.LOOKUP_PANEL_SCALE} x "
        f"{charts.LOOKUP_PANEL_HEIGHT}, where the published figure uses "
        f"0.8 x 290."
    )
    assert charts.lookup_panel_sizing(n_clades=15) == {"width": 336, "height": 232}


def test_the_rendered_panel_a_is_no_taller_than_panel_b():
    """The invariant the constants above are standing in for.

    Runs against the panels this repository last built, so it is the check
    that would actually have caught the drift; it skips where they have not
    been built. A difference of a tick row is data -- a clade whose estimate
    needs one more label -- so this allows the margin one row of type costs
    and fails on the 500 px the band added.
    """
    import panel_spec as spec

    directory = spec.figure_dir(spec.DEFAULT_SELECTION)
    paths = {letter: directory / f"{spec.PANEL_LETTERS[letter]}.png" for letter in "AB"}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        pytest.skip(f"panels not built here: {', '.join(missing)}")

    import matplotlib.image as mpimg

    heights = {letter: mpimg.imread(path).shape[0] for letter, path in paths.items()}
    # One row of 18 px type at 300 ppi is 75 px on disk; allow one.
    assert heights["A"] <= heights["B"] + 75, (
        f"panel A is {heights['A']} px tall and panel B is {heights['B']}. "
        f"Panel A sets the height of the row, so the whole figure composes "
        f"at a smaller dpi and every panel's type shrinks."
    )


def test_each_cell_of_the_shift_grid_names_the_pair_it_draws():
    """Panel D's cells are titled, and with their *own* pair.

    The grid is lower-triangular, so without a title a reader works out which
    pair a cell holds by counting rows and columns against a layout whose
    axes are labelled only on the bottom row and the first column. A title
    taken from the wrong cell is worse than none: a correct-looking panel
    attributing a correlation to the wrong clades, which no file-level check
    can see.

    Driven through `shift_grid` rather than `shift_panel`, because the
    wiring is the thing that can be wrong -- a `shift_panel` called directly
    with a title is given the answer.
    """
    import panel_charts as charts

    clades = ["21J", "21L", "23A"]
    pairs = ["21J\u219221L", "21J\u219223A", "21L\u219223A"]
    shift_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "pair": pair,
                    "sf_shift": [0.1, -0.4, 1.2, 0.7],
                    "fitness_shift": [0.2, -0.3, 1.0, 0.9],
                    "mutation_label": ["A1C", "D2E", "F3G", "H4I"],
                }
            )
            for pair in pairs
        ],
        ignore_index=True,
    )
    metrics = pd.DataFrame(
        {
            "pair_label": pairs,
            "pearson_r": [0.9, 0.8, 0.7],
            "n_mutations": [4, 4, 4],
            "sf_shift_is_constant": [False, False, False],
        }
    )
    grid = charts.shift_grid(shift_df, metrics, clades).to_dict()

    # Row r holds clades[1:][r] as its later clade and clades[:r+1] as its
    # columns, which is the layout the docstring promises and what a reader
    # reads the grid by.
    expected = [
        [f"{earlier}\u2192{later}" for earlier in clades[:row]]
        for row, later in enumerate(clades[1:], start=1)
    ]
    drawn = [
        [cell.get("title", {}).get("text") for cell in row["hconcat"]]
        for row in grid["vconcat"]
    ]
    assert drawn == expected, (
        f"the grid's cells are titled {drawn}, and their positions say they "
        f"hold {expected}. A cell titled with another cell's pair attributes "
        f"its correlation to the wrong clades."
    )
