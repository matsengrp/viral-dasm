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

import panel_spec
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


def test_the_scan_row_does_not_overlap_the_candidate_clades():
    """The scan excludes the two the candidate panels cover, by construction."""
    assert not set(panel_spec.SCAN_CLADES) & set(panel_spec.CANDIDATE_CLADES)
