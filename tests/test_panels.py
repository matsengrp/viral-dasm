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
    frame = pd.DataFrame(rows)
    return frame.assign(clade_founder_aa=frame["parent_aa"])


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return fake_table()


@pytest.fixture(scope="module")
def reverting() -> dict[str, set[int]]:
    """Site 3 reverts in every clade, so each layer drops a different site."""
    return {clade: {3} for clade in FAKE_CLADES}


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
    expected = len(
        table[
            (table["model"] == FAKE_MODELS[0])
            & table["delta_fitness_ratio"].notna()
            & ~table["is_synonymous"]
            & ~table["aa_site"].isin({1, 2, 3})
        ]
    )
    produced = sum(
        len(frame) for (model, _), frame in frames.items() if model == FAKE_MODELS[0]
    )
    assert produced == expected
    assert expected > 0, "the fixture produced no estimated rows to correlate"


def test_every_layer_of_the_site_exclusion_fires(frames):
    """All three site classes come out, and each is a different site.

    Site 1 is globally masked, site 2 is censored and site 3 reverts. The first
    two are columns of the table and the third is not, so a panel that read only
    the table would keep site 3 -- a two-layer exclusion believing it ran three.
    """
    for frame in frames.values():
        assert not frame["globally_masked"].any()
        assert not frame["censored"].any()
        assert 3 not in set(frame["aa_site"])


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


def test_the_shift_join_refuses_a_mismatched_wildtype(table, reverting):
    """D614G in one clade is never compared with E614G in another.

    The guard keys on ``clade_founder_aa`` and **silently does nothing when that
    column is absent**. The scored table spells the founder residue
    ``parent_aa``, so a reader that forgot the rename would pair every mutation
    at a site whose founder differs between the clades, report more shared
    mutations than exist, and raise nothing.
    """
    moved = table.copy()
    at_site_five = (moved["clade"] == "21L") & (moved["aa_site"] == 5)
    moved.loc[at_site_five, ["parent_aa", "clade_founder_aa"]] = "W"
    frames = panels.base_frames(
        moved, reverting_by_clade=reverting, models=[FAKE_MODELS[0]], clades=FAKE_CLADES
    )
    shift = panels.shift_frame(
        frames[(FAKE_MODELS[0], "21J")],
        frames[(FAKE_MODELS[0], "21L")],
        "21J",
        "21L",
        "delta_fitness_ratio",
    )
    assert 5 not in set(shift["aa_site"])
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
    """This repository's own censor audit, or skip."""
    if not panels.AUDIT_JSON.is_file():
        message = f"{panels.AUDIT_JSON} does not exist; run `rule audit_censor`"
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
    assert panels.SELECTIONS[panels.DEFAULT_SELECTION] == (panels.FIGURE_MODEL,)
