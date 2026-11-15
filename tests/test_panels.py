"""What a wrong agreement panel would show, and what catches it.

Every failure mode here returns a full-looking panel. A null fitness column
read as a zero effect, a shift join that compares D614G against E614G, an
exclusion widened from the mutation to its whole codon: each draws a picture
with the right shape and the wrong numbers, and none of them raises.

The checks over a hand-built table run always. The two that read real data --
this repository's own scored table, and mat-pcp's count tables -- skip when it
is absent, and
``VIRAL_DASM_REQUIRE_DATA=1`` turns those skips into failures.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


import dasm2  # noqa: F401

pd = pytest.importorskip("pandas")
numpy = pytest.importorskip("numpy")

import lib
import panel_spec
import panels
from lib import GENE
from runs import MODELS, RUNS

from dasm2.analysis.viral_fitness_eval import COUNT_THRESHOLD_LADDER

#: Two models and three clades is the smallest table that exercises everything
#: the panels do: a model axis to group by, a clade the shift panel can pair,
#: and a third clade to drop.
FAKE_MODELS = ("dasm-1077k", "ci-484k")
FAKE_CLADES = ("21J", "21L", "23A")

#: Enough mutations per clade to clear the adequacy floor at every rung, so a
#: test that means to measure the filter is not measuring the floor.
FAKE_SITES = 60


def read_founder_nt(clade: str) -> str:
    """One clade's founder nucleotide sequence for the gene, as a string."""
    path = lib.mat_pcp_path("clade_founders_dir") / f"{clade}.fa"
    lines = path.read_text().splitlines()
    return "".join(line.strip() for line in lines if not line.startswith(">"))


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
def frames(table) -> dict[tuple[str, str], pd.DataFrame]:
    return panels.base_frames(
        table,
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
    ]
    expected = len(rows)
    produced = sum(
        len(frame) for (model, _), frame in frames.items() if model == FAKE_MODELS[0]
    )
    assert produced == expected
    assert expected > 0, "the fixture produced no estimated rows to correlate"


def test_a_flagged_site_keeps_the_mutations_mat_pcp_estimated(table, frames):
    """The exclusion is per mutation, so a flag on the codon is not a filter.

    Site 1 is globally masked and site 2 is censored in the fixture, and both
    still carry mutations mat-pcp estimated. Those estimates are computed on the
    clade founder's background and are not built from whatever was censored at
    the codon, so they belong in the comparison; widening the flag to the site
    is what #36 removed, and it cost 1,067 mutations across the figure's clades.

    Asserted in both directions: the flagged rows that have an estimate are
    present, and the rows with no estimate are absent however they are flagged.
    """
    for (model, clade), frame in frames.items():
        present = set(zip(frame["aa_site"], frame["mutant_aa"]))
        rows = table[(table["model"] == model) & (table["clade"] == clade)]
        flagged = rows[rows["globally_masked"] | rows["censored"]]
        estimated = flagged[flagged["delta_fitness_ratio"].notna()]
        estimated = estimated[~estimated["is_synonymous"]]
        assert len(estimated), "the fixture flagged no site that mat-pcp estimated"
        for site, mutant in zip(estimated["aa_site"], estimated["mutant_aa"]):
            assert (site, mutant) in present, (
                f"{clade} dropped site {site} because the codon carries a flag, "
                f"though mat-pcp estimated this mutation."
            )
        absent = rows[rows["delta_fitness_ratio"].isna()]
        for site, mutant in zip(absent["aa_site"], absent["mutant_aa"]):
            assert (site, mutant) not in present


def test_the_published_table_still_compares_21k_r346k():
    """The worked case of #36, against the table the manuscript cites.

    21K's codon 346 carries no flag at all in the scored table -- the founder
    residue is R and the reference codon is identical -- yet the site rule
    dropped every mutation there, because the audit called the site reverting.
    Its rule is the *modal parent* base rather than the founder: most of 21K's
    parents carry A at the codon's second nucleotide where the reference carries
    G, so the site is flagged though the founder matches the reference. The
    reversions behind that modal base are K->R on those parents, a different
    mutation on a different background, contributing nothing to these counts.

    R346K is the most strongly observed mutation it cost, at 753 actual against
    22.2 expected, and the two approaches agree closely on it. Pinned against
    the tracked table rather than a fixture, because the fixture cannot express
    "the audit flags a site the table does not".
    """
    model = "dasm-1077k"
    read = panels.read_scored_table(
        published_table_or_skip(), models=[model], clades=["21K"]
    )
    frames = panels.base_frames(read, models=[model], clades=["21K"])
    frame = frames[(model, "21K")]
    row = frame[frame["aa_mutation"] == "R346K"]
    assert len(row) == 1, "R346K is absent from 21K's comparison frame"
    assert float(row["actual_count"].iloc[0]) == 753.0
    assert float(row["predicted_count"].iloc[0]) == pytest.approx(22.238403, abs=1e-5)


def published_table_or_skip():
    """The tracked scored table, or a skip, honouring ``VIRAL_DASM_REQUIRE_DATA``."""
    table_path = (
        REPO_ROOT / "results" / "selection_factors" / "dasm-1077k_full_depth.parquet"
    )
    if not table_path.exists():
        if os.environ.get("VIRAL_DASM_REQUIRE_DATA") == "1":
            pytest.fail(f"{table_path} is missing")
        pytest.skip(f"{table_path} is not present")
    return table_path


def test_a_codon_the_tree_cannot_represent_is_dropped_and_its_neighbours_are_not():
    """#39, in both directions, against the table the manuscript cites.

    21J's codon 142 carries six routes with an actual count of zero against
    259 expected in total, because Delta's founder codon ``GAT`` differs from
    the reference ``GGT`` at the tree-masked nt 21987 and so matches no
    observation mat-pcp counted. Those rows are not measurements and leave.

    The other direction is the one that makes this different from the rule #36
    removed: the ``globally_masked`` mutations mat-pcp *did* estimate must stay.
    Both sit on the same flag, and a predicate that read the flag rather than
    the mask file would take the second with the first.
    """
    model = "dasm-1077k"
    clades = ["21J", "21L", "22B", "24A"]
    read = panels.read_scored_table(
        published_table_or_skip(), models=[model], clades=clades
    )
    frames = panels.base_frames(read, models=[model], clades=clades)

    for clade in clades:
        frame = frames[(model, clade)]
        gone = panels.unobservable_founder_codons(clade)
        assert not frame["aa_site"].isin(gone).any(), (
            f"{clade} still compares a codon whose founder background the tree "
            f"cannot represent: {sorted(gone)}"
        )
        kept = frame[frame["globally_masked"].fillna(False)]
        assert not kept.empty, (
            f"{clade} dropped every globally masked mutation, which is the "
            f"codon widening #36 removed rather than #39's predicate"
        )

    assert panels.unobservable_founder_codons("21J") == frozenset({142, 950})
    assert panels.unobservable_founder_codons("21L") == frozenset({142})


def test_a_config_excluded_nucleotide_does_not_drop_its_codon():
    """24A's F939L: the case the reason column decides.

    24A's founder differs from the reference at nt 24378, which is masked --
    but with reason ``sites_to_exclude``, mat-pcp's own exclusion, not the
    tree's. The base is still observable, so the founder codon still matches
    what matUtils read, and F939L carries 14 actual counts against 5.70
    expected: an ordinary comparison. A predicate that read
    ``masked_nt_sites.csv`` without filtering on ``reason`` would discard it,
    which is why this is pinned rather than left to the measurement above.
    """
    model = "dasm-1077k"
    read = panels.read_scored_table(
        published_table_or_skip(), models=[model], clades=["24A"]
    )
    frame = panels.base_frames(read, models=[model], clades=["24A"])[(model, "24A")]
    row = frame[frame["aa_mutation"] == "F939L"]
    assert len(row) == 1, "F939L left 24A's comparison frame"
    assert float(row["actual_count"].iloc[0]) == 14.0
    assert float(row["predicted_count"].iloc[0]) == pytest.approx(5.698881, abs=1e-5)
    assert 939 not in panels.unobservable_founder_codons("24A")


def test_the_unobservable_codons_are_derived_rather_than_listed():
    """The predicate is rebuilt from mat-pcp's files, so a mask change moves it.

    Asserted by deriving the same answer a second way, from the founder and
    reference FASTAs against the mask file's own rows, rather than by comparing
    to a literal: a literal would pass if the function returned a hardcoded
    {142, 950} and stopped reading its inputs.
    """
    masked = pd.read_csv(lib.mat_pcp_path("masked_nt_sites_file"))
    tree_masked = sorted(
        {
            int(index)
            for index, reason in zip(masked["nt_index"], masked["reason"])
            if reason == panels.TREE_MASK_REASON
        }
    )
    assert tree_masked, "no nucleotide carries the tree-mask reason"
    reference = read_founder_nt("19A")
    for clade in ("21J", "21L", "22B", "24A", "19A", "20A"):
        founder = read_founder_nt(clade)
        expected = frozenset(
            index // 3 + 1
            for index in tree_masked
            if founder[index] != reference[index]
        )
        assert panels.unobservable_founder_codons(clade) == expected


def test_training_censors_exactly_what_mat_pcp_excludes():
    """The premise that makes the comparison fair, re-derived from the files.

    #36 takes the comparison's exclusion from mat-pcp alone: a mutation with no
    counts estimate leaves at the null drop, and nothing else is dropped. That
    is fair only if the model was trained on what is left -- if training censored
    a route mat-pcp kept, the DASM would be scored on a mutation it was taught
    to ignore, and the panel would report a disagreement the pipeline created.

    Nothing in the code enforces it: ``censor_table.py`` partitions mat-pcp's
    five exclusion reasons across the two files a training run reads, and the
    agreement between them is upstream's invariant rather than this
    repository's. So this check re-derives both sides here. Training's censored
    routes are rebuilt from the three configured inputs -- the global mask, the
    per-clade censor table, and founder->reference reversions against the
    reference sequence -- and mat-pcp's excluded routes are the grid of
    single-nucleotide routes off the clade founder minus the routes its own
    per-clade count table carries. The two sets must be equal.

    A drift either way is silent in every other test, because both sides would
    still produce full-looking panels.
    """
    root = Path(lib.CONFIG["mat_pcp"]["root"])
    if not root.exists():
        if os.environ.get("VIRAL_DASM_REQUIRE_DATA") == "1":
            pytest.fail(f"{root} is missing")
        pytest.skip(f"{root} is not mounted")

    masked = pd.read_csv(lib.mat_pcp_path("masked_nt_sites_file"))
    # The gene's first nucleotide site, from the file's own two spellings of
    # one position. Taken rather than hardcoded: a gene change would move it,
    # and a stale constant would shift every route by a frame.
    starts = set(masked["nt_site"] - masked["nt_index"])
    assert len(starts) == 1, f"masked_nt_sites.csv disagrees on where {GENE} starts"
    start = int(starts.pop())
    masked_sites = {int(site) for site in masked["nt_site"]}

    censor = pd.read_csv(lib.mat_pcp_path("censored_mutations_file"))
    reference = read_founder_nt("19A")

    for clade in ("21K", "24A"):
        counts_path = root / "fitness" / "ntmut_counts" / f"{clade}_ntmut_counts.csv"
        if not counts_path.exists():
            pytest.skip(f"{counts_path} is not present")
        founder = read_founder_nt(clade)
        grid = {
            (start + index, base)
            for index, parent in enumerate(founder)
            for base in "ACGT"
            if base != parent
        }
        counts = pd.read_csv(counts_path, usecols=["nt_site", "nt_mutation", "gene"])
        counts = counts[counts["gene"] == GENE]
        kept = {
            (int(site), mutation[-1])
            for site, mutation in zip(counts["nt_site"], counts["nt_mutation"])
        }
        assert not kept - grid, f"{clade}'s count table carries a route off the grid"
        excluded = grid - kept

        rows = censor[censor["clade"] == clade]
        whole_site = {int(site) for site in rows.loc[rows["from_nt"].isna(), "nt_site"]}
        directional = rows[rows["from_nt"].notna()]
        one_route = {
            (int(site), to_nt)
            for site, to_nt in zip(directional["nt_site"], directional["to_nt"])
        }
        censored = {
            (site, base)
            for site, base in grid
            if site in masked_sites
            or site in whole_site
            or (site, base) in one_route
            or (
                base == reference[site - start]
                and founder[site - start] != reference[site - start]
            )
        }

        assert censored, f"{clade} censors nothing, so the check proves nothing"
        assert censored == excluded, (
            f"{clade}: training censors {len(censored - excluded)} routes mat-pcp "
            f"kept and misses {len(excluded - censored)} routes mat-pcp dropped, "
            f"so the two sides of fig:agreement do not see the same mutations."
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


def test_the_shift_join_refuses_a_mismatched_wildtype(tmp_path):
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
    frames = panels.base_frames(read, models=[FAKE_MODELS[0]], clades=FAKE_CLADES)
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


def test_only_the_published_selection_reaches_the_manuscript():
    """Every selection composes to one file name, so only one may be copied.

    ``snakemake figures`` builds both, and both write ``FIGURE_STEM``. A
    ``fidelity`` composition that carried a ``tex_figure_dir`` would land on
    the manuscript's ``agreement.pdf`` with a figure drawn from two other
    models -- same layout, same size, different numbers, and no later stage
    that could tell. This is the only thing standing between those two facts.
    """
    import panel_spec as spec

    for selection in spec.SELECTIONS:
        destination = spec.tex_figure_dir(selection)
        if selection == spec.DEFAULT_SELECTION:
            assert destination == spec.TEX_FIGURE_DIR
        else:
            assert destination is None, (
                f"{selection} would copy over the manuscript's "
                f"{spec.FIGURE_STEM}.pdf, which the paper draws from "
                f"{spec.DEFAULT_SELECTION}."
            )


def test_the_subsampling_figure_reaches_the_manuscript():
    """`fig:subsampling` is published, so its composition is copied.

    The guard above keys on a *selection*, and ``SUBSAMPLING_SELECTION`` is
    not one -- it is this figure's directory, with one panel set and a stem of
    its own. Nothing else composes to ``subsampling.pdf``, so there is no
    second composition for it to be confused with, and it must not be skipped.
    """
    import panel_spec as spec

    assert spec.tex_figure_dir(spec.SUBSAMPLING_SELECTION) == spec.TEX_FIGURE_DIR
    assert spec.SUBSAMPLING_SELECTION not in spec.SELECTIONS


def test_the_subsampling_figure_places_every_panel_once():
    """The rows and the letters are two spellings of one panel set.

    ``rule compose_subsampling`` declares its inputs from
    ``SUBSAMPLING_PANEL_LETTERS`` and the notebook lays out
    ``SUBSAMPLING_FIGURE_ROWS``, so a letter in one and not the other is
    either a panel the rule fetches and the layout drops, or a ``KeyError``
    inside the composition -- and the two panel rules' stems are what both
    have to name.
    """
    import panel_spec as spec

    assert set(spec.SUBSAMPLING_PANEL_LETTERS.values()) == set(
        spec.DEPTH_PANELS + spec.LADDER_ARM_PANELS
    )
    # A cell is a letter or a column of them, which is how B sits over C.
    placed = [
        letter
        for row in spec.SUBSAMPLING_FIGURE_ROWS
        for cell in row
        for letter in ((cell,) if isinstance(cell, str) else cell)
    ]
    assert sorted(placed) == sorted(spec.SUBSAMPLING_PANEL_LETTERS)
    assert len(placed) == len(set(placed)), "a panel is placed twice"


def test_the_subsampling_panels_are_set_in_the_published_type_size():
    """`panel_spec` states this figure's type size and `panel_charts` draws it.

    Two modules have to agree because the first is stdlib-only -- it is parsed
    on every snakemake invocation -- and the second is what the panels are
    rendered from. `SUBSAMPLING_TITLE_PX` is also what the composition sizes
    the panel letters against, so a drift here moves the letters as well as
    the titles.
    """
    import panel_charts as charts
    import panel_spec as spec

    assert spec.SUBSAMPLING_TITLE_PX == charts.SUBSAMPLING_TITLE_PX
    assert spec.SUBSAMPLING_LABEL_PX == charts.SUBSAMPLING_LABEL_PX
    # The one figure not drawn at the module's own size; see either docstring.
    assert spec.SUBSAMPLING_TITLE_PX != charts.PANEL_FONT_SIZE


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


def test_the_candidate_clades_match_the_candidate_file():
    """`CANDIDATE_CLADES` is a hand list beside a file it must agree with.

    The file exists so panels A and B cannot disagree about their subjects; a
    candidate added for a third clade would otherwise sit silently outside the
    clade tuple and be dropped by every panel that reads it. The tuple keeps
    its own order, because that order is the figure's row order and not a
    property of the data.
    """
    from_file = {clade for clade, _ in panel_spec.figure_candidates()}
    from_file |= {clade for clade, _, _ in panel_spec.off_figure_candidates()}
    assert set(panel_spec.CANDIDATE_CLADES) == from_file


def test_every_clade_row_has_both_of_its_panels():
    """A clade of the figure gets a scatter and a background panel, or neither.

    The composed figure is three rows of two, so a clade present in one stem
    tuple and absent from the other would leave a row half drawn -- and the
    letters in the caption would stop matching the images.
    """
    scatter = {stem.rsplit("_", 1)[-1] for stem in panel_spec.OUTLIER_PANELS}
    backgrounds = {stem.rsplit("_", 1)[-1] for stem in panel_spec.BACKGROUND_PANELS}
    assert scatter == backgrounds == set(panel_spec.CANDIDATE_CLADES)


#: The smallest frame `background_panel` draws: two clades of unequal width,
#: so the shared slot geometry has something to line up, and two backgrounds
#: per candidate, which is the contrast the panel exists to show.
def _background_frame():
    rows = []
    for clade, candidates in (
        ("22D", ["D574V", "K444T", "L452R"]),
        ("20B", ["D1259H"]),
    ):
        founder, derived = f"R346, K356 ({clade})", f"T346, T356 ({clade})"
        for index, candidate in enumerate(candidates):
            for order, (background, level) in enumerate(
                ((founder, 6.0 - index), (derived, 1.0 + index))
            ):
                rows.append(
                    {
                        "cluster": clade,
                        "candidate": candidate,
                        "background": background,
                        "background_order": order,
                        "counts_effect": level,
                        "dasm_effect": level + 0.2,
                        "p_label": "p < 1e-4",
                    }
                )
    return pd.DataFrame(rows)


def test_the_background_rows_come_out_the_same_size():
    """B, D and F are three images that have to read as one exhibit.

    They are stacked left justified, each beside its own clade's scatter, so
    a reader compares their slopes by eye across rows. That only works if a
    sub-panel sits at the same x and the same width in every row -- and the
    rows do not have the same number of candidates, so nothing makes that
    true on its own. Two things do: every row is drawn the same number of
    slots wide, and every row after the first is given the first's margins.

    This is the failure the Vega-Lite version had and had no symptom for.
    Each row rendered correctly on its own; only the composed figure showed
    that 20B's panels were wider than 22D's and started further left.
    """
    import panel_charts as charts

    frame = _background_frame()
    domain = charts.background_y_domain(frame)
    slots = 4

    sizes, geometries, h_params = [], [], None
    for clade in ("22D", "20B"):
        figure = charts.background_panel(
            frame[frame["cluster"] == clade],
            clade,
            y_domain=domain,
            columns=slots,
            h_params=h_params,
        )
        if h_params is None:
            pars = figure.subplotpars
            h_params = (pars.left, pars.right, pars.wspace)
        sizes.append(tuple(figure.get_size_inches()))
        first = figure.axes[0].get_position()
        geometries.append((round(first.x0, 6), round(first.width, 6)))

    assert sizes[0] == sizes[1], (
        f"the rows are {sizes[0]} and {sizes[1]} inches. Rows of different "
        f"sizes cannot line up however they are composed."
    )
    assert geometries[0] == geometries[1], (
        f"the first sub-panel sits at x0/width {geometries[0]} in 22D's row "
        f"and {geometries[1]} in 20B's. `h_params` is what carries the "
        f"widest row's margins onto the narrower ones; without it "
        f"`tight_layout` measures each row's margins from its own tick "
        f"labels and the narrower rows come out wider."
    )


def test_every_background_sub_panel_draws_its_grid():
    """The grid is how a reader carries a level across a row.

    `sharey` puts the tick *labels* on the leftmost sub-panel only, which is
    right -- every cell shares one domain and the rest would repeat the same
    numbers. It does not follow that the rest should lose their grid lines:
    without them a reader comparing F486V's level against D574V's has
    nothing to carry the value across four panels. The Vega-Lite version
    dropped the axis and the grid together, which is how this came up.
    """
    import panel_charts as charts

    frame = _background_frame()
    figure = charts.background_panel(
        frame[frame["cluster"] == "22D"],
        "22D",
        y_domain=charts.background_y_domain(frame),
        columns=4,
    )
    drawn = [axis for axis in figure.axes if axis.get_title()]
    assert len(drawn) == 3, f"expected three titled sub-panels, got {len(drawn)}."
    for axis in drawn:
        assert any(line.get_visible() for line in axis.get_ygridlines()), (
            f"the {axis.get_title()} sub-panel has no horizontal grid. Only "
            f"the leftmost cell carries tick labels, so the grid is the only "
            f"thing the other cells give a reader to read a level against."
        )


def test_the_background_rows_share_one_value_axis():
    """A steep line has to mean a large shift, not a narrow axis.

    The three rows are the figure's claim that the DASM tracks a shift the
    counts see, and their slopes are only commensurable on one scale. The
    domain is computed over every clade at once for that reason, so this
    checks the renderer honours what it is handed rather than rescaling to
    the clade in front of it.
    """
    import panel_charts as charts

    frame = _background_frame()
    domain = charts.background_y_domain(frame)
    for clade in ("22D", "20B"):
        figure = charts.background_panel(
            frame[frame["cluster"] == clade], clade, y_domain=domain, columns=4
        )
        for axis in figure.axes:
            if axis.get_title():
                assert axis.get_ylim() == pytest.approx(domain), (
                    f"{clade}'s {axis.get_title()} panel is scaled to "
                    f"{axis.get_ylim()}, not the figure's {domain}."
                )


def test_a_background_row_refuses_to_crowd_out_its_own_legend():
    """Every row carries its own key, in a slot the padding leaves empty.

    B, D and F are separately lettered panels and a reader may meet any one
    of them first, so none can borrow its key from the row above. The key
    lives in the first slot past that clade's sub-panels, which exists only
    because every row is drawn one slot wider than the widest clade needs.
    Lose that slot -- by sizing a row to its own candidates, say -- and the
    legend is drawn over the last panel with nothing saying so.
    """
    import panel_charts as charts

    frame = _background_frame()
    with pytest.raises(ValueError, match="no slot for the legend"):
        charts.background_panel(
            frame[frame["cluster"] == "22D"],
            "22D",
            y_domain=charts.background_y_domain(frame),
            columns=3,
        )


def test_a_background_panel_refuses_a_third_background():
    """The panel draws a contrast, and a contrast has two sides.

    A third pool reaching here would be plotted as a third x position with
    nothing in the panel saying which pair of them the p-value above it
    tested.
    """
    import panel_charts as charts

    frame = _background_frame()
    extra = frame[frame["cluster"] == "22D"].copy()
    third = extra.head(1).copy()
    third["background"] = "Q346, K356 (22D)"
    third["background_order"] = 2
    with pytest.raises(ValueError, match="not two"):
        charts.background_panel(
            pd.concat([extra, third]),
            "22D",
            y_domain=(0.0, 1.0),
            columns=4,
        )


# ---------------------------------------------------------------------------
# `fig:subsampling` panels A and B
# ---------------------------------------------------------------------------
#
# These read mat-pcp's count tables rather than a scored table, so they run
# against the real tree or not at all -- there is no hand-built stand-in for
# "how unevenly the clades are sampled". The eCDF reads eight per-clade tables,
# so it is built once for the module.

#: The counts measured from this mat-pcp run on 2026-11-14, which is what the
#: panel claims. Pinned rather than recomputed: a changed share means the tree
#: or the gene filter moved, and either should fail here rather than quietly
#: redraw the paper's figure with different numbers.
ECDF_SHARE_AT_THRESHOLD = {
    ("21J", "full_depth"): 0.6579,
    ("21J", "cap_50000_0"): 0.0705,
    ("21J", "cap_20000_0"): 0.0478,
    ("21J", "cap_5000_0"): 0.0004,
    ("25C", "full_depth"): 0.0680,
    ("25C", "cap_50000_0"): 0.0416,
    ("25C", "cap_20000_0"): 0.0148,
    ("25C", "cap_5000_0"): 0.0009,
}

#: Each clade's spike amino-acid grid: the non-synonymous rows of the
#: per-clade table, 7,901 of 21J's 9,140 spike rows. This is the population the
#: paper's "7,901 amino-acid mutations to its founder spike sequence" names.
SPIKE_GRID_ROWS = {"21J": 7901, "25C": 7795}


def _aamut_dir() -> Path:
    return lib.mat_pcp_path("aamut_fitness_dir")


@pytest.fixture(scope="module")
def ecdf():
    """Panel B's frame, built once: eight per-clade tables."""
    if not _aamut_dir().exists():
        pytest.skip(f"{_aamut_dir()} is not mounted")
    return panels.count_ecdf_frame(_aamut_dir())


def test_the_amino_acid_grid_does_not_move_with_sampling_depth():
    """A thinned arm counts the same mutations, or the founder moved.

    The grid is enumerated from the clade founder, so its size cannot depend on
    how many sequences were sampled. If it does, the two curves in a panel are
    drawn over different mutations and the panel still looks fine -- which is
    why `count_ecdf_frame` raises rather than warns, and why the row counts are
    pinned here as well.
    """
    directory = _aamut_dir()
    if not directory.exists():
        pytest.skip(f"{directory} is not mounted")
    for clade, expected in SPIKE_GRID_ROWS.items():
        sizes = {
            arm: len(panels._aa_grid(directory, clade, arm)) for arm in lib.CORPUS_NAMES
        }
        assert set(sizes.values()) == {expected}, (
            f"{clade}'s spike grid is {sizes}, not {expected} at every depth. "
            f"The clade founder moved, so the depth curves would be drawn over "
            f"different mutations."
        )


def test_the_ecdf_reproduces_the_measured_shares(ecdf):
    """Panel B's headline, against the tree it is measured from."""
    at_threshold = ecdf[ecdf["x"] == float(panels.PANEL_MIN_COUNT)]
    assert not at_threshold.empty, (
        f"the eCDF grid does not land on {panels.PANEL_MIN_COUNT}, so the "
        f"panel's dotted threshold sits between two sampled points and the "
        f"share a reader takes off the curve is interpolated."
    )
    measured = {
        (row.clade, row.arm): row.share for row in at_threshold.itertuples(index=False)
    }
    assert set(measured) == set(ECDF_SHARE_AT_THRESHOLD)
    for key, expected in ECDF_SHARE_AT_THRESHOLD.items():
        # 1e-4 is about one mutation in a 7,901-row grid. At the 1e-3 this
        # started at, the two cap_5000 pins -- 0.0004 and 0.0009 -- accepted
        # zero, so an arm that silently lost every count still passed.
        assert measured[key] == pytest.approx(expected, abs=1e-4), (
            f"{key[0]}/{key[1]} is {measured[key]:.4f} against the {expected} "
            f"measured from this run. Re-measure rather than re-pinning."
        )


def test_no_cap_sits_above_full_depth(ecdf):
    """Thinning cannot add counts.

    Checked at every sampled x rather than at the threshold alone: a sign error
    in the arm lookup, or a cap directory read for the wrong arm, would leave
    the threshold's ordering intact while inverting the curves either side of
    it.
    """
    for clade, rows in ecdf.groupby("clade"):
        full = rows[rows["arm"] == "full_depth"].set_index("x")["share"]
        for arm, arm_rows in rows[rows["arm"] != "full_depth"].groupby("arm"):
            capped = arm_rows.set_index("x")["share"]
            above = capped[capped > full + 1e-12]
            assert above.empty, (
                f"{clade}/{arm} is above full depth at x={list(above.index)}, "
                f"so thinning the tree added counts."
            )


def test_panel_a_plots_the_roster_and_not_the_counts_table():
    """The two clades the counts table carries and this pipeline does not score.

    `mat/clade_counts.csv` has 58 rows; `21A` and `25I` have a sequence count,
    no founder FASTA and no row in any fitness table. A panel drawn from the
    file as it stands would put two bars beside 56 that every other table in
    this pipeline omits.
    """
    counts_csv = lib.mat_pcp_path("clade_counts_file")
    if not counts_csv.exists():
        pytest.skip(f"{counts_csv} is not mounted")
    assert counts_csv.name == "clade_counts.csv", (
        "panel A must read `clade_counts.csv`. "
        "`clade_counts_by_cutoff.csv` sits beside it with a `cutoff` column "
        "holding the cutoff-date ladder and never this run's cutoff, so "
        "reading it returns an empty frame rather than raising."
    )
    emergence_csv = panel_spec.EMERGENCE_CSV
    if not emergence_csv.exists():
        pytest.skip(f"{emergence_csv} has not been built")
    sizes = panels.clade_size_frame(counts_csv, emergence_csv)
    # `year` is the emergence date panel A colours its dots by, as a decimal
    # year; see `panels._emergence_years`.
    assert list(sizes.columns) == ["clade", "n_sequences", "year"]
    assert sorted(sizes["clade"]) == lib.clade_roster()
    raw = pd.read_csv(counts_csv)
    extra = sorted(set(raw["Nextstrain_clade_usher"]) - set(sizes["clade"]))
    assert extra == ["21A", "25I"], (
        f"the counts table's clades beyond the roster are {extra}, not the "
        f"['21A', '25I'] measured when #24 was written. Re-measure: the "
        f"roster and the counts table moved relative to each other."
    )
    assert sizes["n_sequences"].is_monotonic_decreasing


def test_the_subsampling_directory_is_not_a_model_selection():
    """`subsampling` names a directory, and must not name a wildcard value.

    `SELECTIONS` is `rule panels`' wildcard domain. An entry there would let the
    agreement rules resolve `selection=subsampling` and draw `fig:agreement`'s
    panels into this figure's directory, which is a DAG that builds and a figure
    that is wrong.
    """
    assert panel_spec.SUBSAMPLING_SELECTION not in panel_spec.SELECTIONS


def test_the_subsampling_arms_are_the_corpora_the_pipeline_builds():
    """Panel B's depth axis against the corpora, and the caps against their names.

    `panel_spec` restates nothing here -- the arms *are* `lib.CORPUS_NAMES` and
    the caps are read off those names -- so what this pins is that the reading
    still works: a corpus renamed without `cap_<n>_<fold>` in it would drop
    silently out of `subsampling_caps()` and take a rule off panel A.
    """
    assert panel_spec.subsampling_caps() == (50_000, 20_000, 5_000)
    assert len(panel_spec.subsampling_caps()) == len(lib.CORPUS_NAMES) - 1
    assert lib.CORPUS_NAMES[0] == lib.FULL_DEPTH.name
    assert set(panel_spec.ECDF_CLADES) <= set(lib.clade_roster())


# ---------------------------------------------------------------------------
# `fig:subsampling` panels C and D
# ---------------------------------------------------------------------------
#
# Built on a synthetic four-arm table rather than on the scored one: every
# property below is a property of the construction, and a 10 GB Parquet read
# would make the suite too slow to run on every change without testing
# anything these frames do not.

#: The clades the synthetic ladder is built over. Two, because the roster
#: guard needs one clade that a cap thins below the floor and one that does
#: not.
LADDER_CLADES = ("21J", "21L")


def fake_arm_table(seed: int = 3) -> pd.DataFrame:
    """A scored table carrying every arm, in the shape `rule score` writes.

    The full-depth counts estimate is the same on every arm's rows, which is
    what `rule score` does and what makes it the shared target; only
    `log_selection_factor` moves, and it moves *towards noise* as the cap
    tightens, so a test that confuses the arms sees it.
    """
    rng = numpy.random.default_rng(seed)
    rows = []
    for clade in LADDER_CLADES:
        for site in range(1, FAKE_SITES + 1):
            for mutant in ("A", "G", "V"):
                parent = "A" if site % 7 else "E"
                target = float(rng.normal(-1.0, 1.5))
                for arm_index, arm in enumerate(lib.CORPUS_NAMES):
                    rows.append(
                        {
                            "clade": clade,
                            "model": panel_spec.subsampling_model(arm),
                            "arm": arm,
                            "aa_mutation": f"{parent}{site}{mutant}",
                            "aa_site": site,
                            "parent_aa": parent,
                            "mutant_aa": mutant,
                            "is_synonymous": mutant == parent,
                            "log_selection_factor": target
                            + float(rng.normal(0, 0.3 * arm_index)),
                            "actual_count": float(rng.integers(10, 80)),
                            "predicted_count": float(rng.integers(10, 80)),
                            "delta_fitness_ratio": target,
                            "delta_fitness_bayesian": target,
                            "uncertainty": 0.1,
                            "globally_masked": False,
                            "censored": False,
                        }
                    )
    return pd.DataFrame(rows)


def write_fake_arm_fitness(directory: Path, table: pd.DataFrame) -> Path:
    """mat-pcp's per-arm amino-acid tables, in the layout the figure reads.

    Full depth at the top level and each cap arm in a subdirectory named for
    its corpus, which is `panel_spec.aamut_grid_path`'s layout. The full-depth
    file carries the target itself, so the full-depth counts cell is the
    target against itself -- r = 1 by construction, which is the panel's
    stated baseline and the test below.
    """
    for arm in lib.CORPUS_NAMES:
        root = directory if arm == lib.FULL_DEPTH.name else directory / arm
        root.mkdir(parents=True, exist_ok=True)
        rows = table[table["arm"] == arm]
        for clade in LADDER_CLADES:
            frame = rows[rows["clade"] == clade]
            noise = numpy.random.default_rng(len(arm)).normal(
                0, 0.4 * lib.CORPUS_NAMES.index(arm), len(frame)
            )
            pd.DataFrame(
                {
                    "gene": lib.GENE,
                    "aa_mutation": frame["aa_mutation"].to_numpy(),
                    "aa_site": frame["aa_site"].to_numpy(),
                    "clade_founder_aa": frame["parent_aa"].to_numpy(),
                    "mutant_aa": frame["mutant_aa"].to_numpy(),
                    "actual_count": frame["actual_count"].to_numpy(),
                    "expected_count": frame["predicted_count"].to_numpy(),
                    "delta_fitness_ratio": frame["delta_fitness_ratio"].to_numpy()
                    + noise,
                }
            ).to_csv(root / f"{clade}{lib.FITNESS_SUFFIX}", index=False)
    return directory


@pytest.fixture(scope="module")
def ladder(tmp_path_factory):
    """Panel C's frames and panel D's statistics, over the synthetic arms."""
    directory = tmp_path_factory.mktemp("ladder")
    frame = fake_arm_table()
    table_path = directory / "selection_factors.parquet"
    frame.to_parquet(table_path, index=False)
    aamut = write_fake_arm_fitness(directory / "aamut_fitness", frame)
    frames = panels.ladder_frames(table_path, aamut, clades=LADDER_CLADES)
    return frames, panels.ladder_metrics(frames), table_path, aamut


def test_the_population_is_fixed_on_the_full_depth_arm(ladder):
    """The survivorship-bias guard, and the reason the figure is built this way.

    Filtering each arm on its own counts would score a thinned arm only over
    the mutations it still happened to observe -- the subset it is best at --
    and every curve would be drawn over a different population. The bug it
    produces is a plausible figure rather than an error, so what is asserted
    is equality across arms rather than any particular count.
    """
    _frames, metrics, _table, _aamut = ladder
    for (clade, method), rows in metrics.groupby(["clade", "method"]):
        counts = set(rows["n_mutations"])
        assert len(counts) == 1, (
            f"{clade}/{method} is scored over {dict(zip(rows['arm'], rows['n_mutations']))}. "
            f"Every arm must be scored over the mutations the full-depth arm "
            f"selected, or the arms are not comparable."
        )
        assert counts != {0}


def test_the_full_depth_counts_cell_is_the_target_against_itself(ladder):
    """Panel C's baseline, which the caption calls one by construction."""
    _frames, metrics, _table, _aamut = ladder
    full = metrics[
        (metrics["arm"] == lib.FULL_DEPTH.name) & (metrics["method"] == "counts")
    ]
    assert not full.empty
    for row in full.itertuples(index=False):
        assert row.pearson_r == pytest.approx(1.0, abs=1e-12)
        assert row.regression_slope == pytest.approx(1.0, abs=1e-12)


def test_a_clade_a_cap_thins_keeps_its_place(ladder):
    """The roster is fixed on full depth too, one level up from the population.

    A clade a cap thinned to four mutations is the ladder's question, not a
    clade to drop, so every clade appears at every arm.
    """
    _frames, metrics, _table, _aamut = ladder
    for arm in lib.CORPUS_NAMES:
        drawn = set(metrics[metrics["arm"] == arm]["clade"])
        assert drawn == set(LADDER_CLADES), (
            f"arm {arm} draws {sorted(drawn)}, not {sorted(LADDER_CLADES)}."
        )


def test_the_clip_is_a_control_and_not_a_filter(ladder):
    """Clipping moves where a point is drawn, never which rows are scored."""
    _frames, metrics, table_path, aamut = ladder
    unclipped = panels.ladder_metrics(
        panels.ladder_frames(
            table_path,
            aamut,
            clades=LADDER_CLADES,
            spec=panels.FilterSpec(min_count=0.0, clip_lower=None),
        )
    )
    merged = metrics.merge(
        unclipped, on=["arm", "clade", "method"], suffixes=("", "_unclipped")
    )
    assert (merged["n_mutations"] == merged["n_mutations_unclipped"]).all(), (
        "the clip changed the row set, so it is filtering rather than "
        "clamping and `n_mutations` no longer means what the panel says."
    )


def test_the_drawn_fit_comes_from_the_reported_slope(ladder):
    """The line and the corner label cannot disagree.

    Vega's `transform_regression` fits the rows it is handed, so under a point
    budget the drawn line would come from a sample and the label from the
    whole cloud. `arm_fit_line` takes the reported slope and the full frame's
    means instead, so the line is a statement about the same numbers the CSV
    carries.
    """
    import panel_charts as charts

    frames, metrics, _table, _aamut = ladder
    arm, clade, method = lib.CORPUS_NAMES[-1], LADDER_CLADES[0], "DASM"
    frame = frames[(arm, clade, method)]
    row = metrics[
        (metrics["arm"] == arm)
        & (metrics["clade"] == clade)
        & (metrics["method"] == method)
    ].iloc[0]
    line = charts.arm_fit_line(
        frame,
        x_field="delta_fitness_ratio",
        y_field="log_selection_factor",
        slope=float(row["regression_slope"]),
        domain=[-6.0, 2.0],
    )
    drawn_slope = (
        line["log_selection_factor"].iloc[1] - line["log_selection_factor"].iloc[0]
    ) / (line["delta_fitness_ratio"].iloc[1] - line["delta_fitness_ratio"].iloc[0])
    assert drawn_slope == pytest.approx(float(row["regression_slope"]), rel=1e-12)


def test_the_subsampling_clip_is_not_the_library_clip():
    """-6.0 here, -4.0 in the library, and a sibling of that name is a trap."""
    from dasm2.analysis.viral_fitness_eval import CLIP_LOWER

    assert panel_spec.SUBSAMPLING_CLIP_LOWER == -6.0
    assert panel_spec.SUBSAMPLING_CLIP_LOWER != CLIP_LOWER
    assert set(panel_spec.SUBSAMPLING_MODELS) <= set(MODELS)
    assert panel_spec.SUBSAMPLING_SCATTER_CLADE in panel_spec.SUBSAMPLING_LADDER_CLADES
