"""What a wrong scoring pass would produce, and what catches it.

The failure mode this file exists for is that none of the plausible mistakes
raise. A frame-shifted founder, a count column under the wrong name, a cache
that answers for the wrong clade: each returns a full-looking table whose rows
are joined to the wrong thing, in the right shape, with no error.

The checks that need a trained checkpoint skip when none is present, so the
file is runnable before the campaign finishes and meaningful after it. The
ones that do not -- the count gate, the roster readers -- run always.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import lib

import dasm2  # noqa: F401

pd = pytest.importorskip("pandas")

import check_clades
import score_run

from dasm2.analysis.viral import normalize_fitness_columns
from dasm2.analysis.viral_fitness_eval import apply_count_threshold
from dasm2.neutral.scov2_rates import GENE_NT_LENGTH

#: The run whose checkpoint these checks read when one exists. The
#: context-independent baseline, because its head is a function of
#: ``(aa_site, parent_aa, mutant_aa)`` alone, which is a property no
#: transformer has and which a wrong clade key would break.
CI_RUN = "ci-484k_full_depth"

#: Clades in the mat-pcp run ``config.yaml`` points at. Pinned rather than
#: asserted to be "more than one": the number is what every row count in this
#: file is derived from, and a roster that silently shrank would still agree
#: across the three readings.
EXPECTED_CLADES = 56

#: Codon sites in the gene being scored, which is the grid's first axis.
SITES = GENE_NT_LENGTH[lib.GENE] // 3


def ci_run_dir() -> Path:
    """The baseline's run directory, or skip.

    ``VIRAL_DASM_REQUIRE_CHECKPOINT=1`` turns the skip into a failure. Set it at
    merge time: the skip is right while the campaign is mid-flight, and after it
    lands a silent ``s`` is the one outcome that looks like a pass. There are
    four ways to get it once the run exists -- a ``VIRAL_DASM_ARTIFACT_ROOT``
    override, a different ``artifact_root`` in ``config.yaml``, a renamed run,
    or no torch -- and none of them says so.
    """
    run_dir = Path(lib.ARTIFACT_ROOT) / "runs" / CI_RUN
    if not (run_dir / "training_complete.done").is_file():
        message = f"{CI_RUN} has not finished training (looked in {run_dir})"
        if os.environ.get("VIRAL_DASM_REQUIRE_CHECKPOINT") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return run_dir


@pytest.fixture(scope="module")
def scored_two_clades() -> pd.DataFrame:
    """Two clades of the baseline's grid, scored on the CPU.

    Two rather than one: the invariant that matters most here is a comparison
    *across* clades, and one clade cannot fail it.
    """
    torch = pytest.importorskip("torch")
    run_dir = ci_run_dir()
    source = score_run.SelectionFactorSource(
        {(lib.GENE, CI_RUN): lib.select_checkpoint(run_dir)}, "cpu"
    )
    pooled = score_run.pooled_fitness(lib.GENE)
    masked = score_run.globally_masked_sites()
    frames = []
    del torch  # only imported to skip the module when it is absent
    try:
        for clade in ("21J", "21L"):
            frames.append(
                score_run.score_clade(
                    source,
                    gene=lib.GENE,
                    run_id=CI_RUN,
                    clade=clade,
                    model="ci-484k",
                    corpus="full_depth",
                    pooled=pooled,
                    masked=masked,
                    censored=set(),
                    censored_directional=set(),
                )
            )
    finally:
        source.release()
    return pd.concat(frames, ignore_index=True)


def test_context_independent_head_has_no_clade_spread(scored_two_clades):
    """``ci-484k``'s selection factor cannot depend on which clade it scored.

    Its head is a lookup on ``(aa_site, parent_aa, mutant_aa)``, so two clades
    whose founders agree at a site must get identical numbers there. Any spread
    means clade identity reached the forward pass.

    It does **not** catch an SF cache that answered for the wrong clade. The
    triples this compares are by construction the ones both clades have, which
    are the sites where the founders agree, and a context-independent head
    gives those sites the same numbers in either clade whether the frame came
    from the right one or not. The wrong rows are at the founder-differing
    sites, which are not shared and so are not compared here.
    :func:`test_synonymous_rows_are_the_gauge_zero` is what catches that:
    ``parent_aa`` is rebuilt from the clade's own founder while
    ``is_synonymous`` rides along inside the cached frame, so a frame from the
    wrong clade breaks their agreement at exactly those sites.
    """
    founders = {
        clade: score_run.founder_aa_sequence(clade, lib.GENE)
        for clade in ("21J", "21L")
    }
    assert founders["21J"] != founders["21L"], (
        "21J and 21L have identical founders in this mat-pcp run, which makes "
        "both this test and the wrong-clade coverage in "
        "test_synonymous_rows_are_the_gauge_zero vacuous; pick two clades that "
        "differ"
    )

    grouped = scored_two_clades.groupby(["aa_site", "parent_aa", "mutant_aa"])[
        "log_selection_factor"
    ]
    shared = grouped.count() > 1
    assert shared.sum() > 10_000, (
        "the two clades share too few (site, parent, child) triples for this "
        "to be a test; their founders would have to differ almost everywhere"
    )
    spread = (grouped.max() - grouped.min())[shared]
    assert spread.max() == 0.0, (
        f"ci-484k gave the same mutation different selection factors in "
        f"different clades, by up to {spread.max():.3g}. Its head reads only "
        f"(aa_site, parent_aa, mutant_aa), so the clade reached the forward "
        f"pass."
    )


def test_synonymous_rows_are_the_gauge_zero(scored_two_clades):
    """Wildtype-to-wildtype is exactly 0, not approximately.

    ``selection_factors_of_aa_str`` zaps the diagonal, fixing the factor at
    exactly 1.0 at the wildtype. A synonymous row that is merely small means
    the grid's ``parent_aa`` and the sequence the model conditioned on have
    drifted apart -- the one failure the row count cannot show.
    """
    syn = scored_two_clades[scored_two_clades["is_synonymous"]]
    assert len(syn) == 2 * SITES
    assert (syn["parent_aa"] == syn["mutant_aa"]).all()
    assert syn["log_selection_factor"].abs().max() == 0.0


def test_grid_is_the_whole_grid_and_unique(scored_two_clades):
    """Every (site, amino acid) cell, once, with a prediction in it."""
    assert len(scored_two_clades) == 2 * SITES * 20
    key = ["model", "arm", "clade", "aa_site", "mutant_aa"]
    assert not scored_two_clades.duplicated(key).any()
    assert scored_two_clades["log_selection_factor"].notna().all()
    assert (scored_two_clades["background"] == "").all()


def test_fitness_is_left_joined_not_inner(scored_two_clades):
    """Cells mat-pcp never estimated survive, carrying null fitness.

    An inner join would drop them, and the DMS comparison scores them. This is
    the one structural difference between this table and what
    ``merge_and_filter`` returns.
    """
    missing = scored_two_clades["delta_fitness_ratio"].isna()
    assert missing.any(), "no unmatched cells: this looks like an inner join"
    assert scored_two_clades.loc[missing, "log_selection_factor"].notna().all()


def test_pooled_columns_arrive_under_the_names_the_schema_promises(scored_two_clades):
    """The ``_all`` half of the join is present, suffixed, and keyed on parent.

    Four pooled columns collide with their per-clade counterparts and arrive as
    ``_all``; ``source_clade`` has no counterpart and arrives bare. A rename
    that moved any of them would leave the figure code reading a column of
    nulls, which is indistinguishable from a mutation mat-pcp never estimated.

    ``source_clade`` is also the one check on the pooled join's third key. The
    pooled table is pooled over clades whose founders differ, so a row only
    belongs to a grid row whose ``parent_aa`` it describes; a matched row with
    no ``source_clade`` would mean the parent residue was dropped from the key.
    """
    for column in (
        "actual_count_all",
        "predicted_count_all",
        "delta_fitness_ratio_all",
        "delta_fitness_bayesian_all",
        "source_clade",
    ):
        assert column in scored_two_clades.columns, column
    assert "source_clade_all" not in scored_two_clades.columns

    matched = scored_two_clades[scored_two_clades["delta_fitness_ratio_all"].notna()]
    assert len(matched) > 10_000
    assert matched["source_clade"].notna().all()


def test_the_count_column_survives_under_the_canonical_name(scored_two_clades):
    """``predicted_count`` reaches the table, from mat-pcp's ``expected_count``.

    The gate in ``filter_merged_fitness`` fires only ``if "predicted_count" in
    df.columns``, so a frame that reached the figures still spelling mat-pcp's
    ``expected_count`` would be filtered on ``actual_count`` alone: no error, a
    plausible row count, and a wrong x-axis. Asserted on the scored output
    rather than on a hand-built frame, because that is where the spelling has to
    be right.
    """
    for column in ("predicted_count", "predicted_count_all"):
        assert column in scored_two_clades.columns, column
    assert "expected_count" not in scored_two_clades.columns

    matched = scored_two_clades[scored_two_clades["delta_fitness_ratio"].notna()]
    assert matched["predicted_count"].notna().all()
    assert matched["predicted_count"].gt(0).any()


def test_founder_length_is_checked():
    """A founder of the wrong length raises instead of shifting every site.

    This is the frame-shift guard. ``clade_founders_dir`` points at one gene's
    founders, so asking for ``E``'s 225 nucleotides gets ``S``'s 3,819 and the
    comparison fails -- which is what a truncated FASTA, or a directory
    repointed at another gene, would look like. Without the check the sequence
    translates fine and every ``aa_site`` shifts from the first missing codon
    onward, mis-pairing mutations in a table that still has the right number of
    rows.

    ``match="codons"`` pins it to the length branch. Asking instead for a gene
    ``GENE_NT_LENGTH`` does not know raises ``KeyError`` from the lookup inside
    the condition, so the comparison never runs and the check could be deleted
    with the test still passing.
    """
    assert GENE_NT_LENGTH["E"] != GENE_NT_LENGTH[lib.GENE]
    with pytest.raises(ValueError, match="codons"):
        score_run.founder_aa_sequence("21J", "E")


def test_count_gate_actually_drops_rows():
    """The library's count filter bites, and skips half its gate un-normalized.

    A contract test on the vendored code, not on scoring: what the scored table
    actually carries is
    :func:`test_the_count_column_survives_under_the_canonical_name`. This one
    establishes that the hazard is real, by showing the same four rows filter
    differently either side of ``normalize_fitness_columns``.

    The un-normalized expectation of 2 is the quiet failure written down, not a
    behaviour to preserve: a library change that made
    :func:`apply_count_threshold` raise on an unrecognised count column instead
    would be a fix, and this line is where it would surface.
    """
    raw = pd.DataFrame(
        {
            "aa_site": [1, 2, 3, 4],
            "mutant_aa": list("ACDE"),
            "actual_count": [0, 0, 100, 100],
            "expected_count": [0.0, 100.0, 0.0, 100.0],
            "log_selection_factor": [0.0, 0.0, 0.0, 0.0],
            "delta_fitness_ratio": [0.0, 0.0, 0.0, 0.0],
        }
    )
    normalized = normalize_fitness_columns(raw)
    assert "predicted_count" in normalized.columns

    kept = apply_count_threshold(normalized, 10)
    assert len(kept) == 3, (
        "an `or` gate at threshold 10 keeps a row with either count above it, "
        "so only the (0, 0) row should go"
    )

    # Without normalizing, the predicted-count half of the gate is skipped and
    # the (0, 100) row is dropped on `actual_count` alone -- the silent wrong
    # answer this spelling exists to prevent.
    unnormalized = apply_count_threshold(raw, 10)
    assert len(unnormalized) == 2


def test_clade_rosters_agree():
    """The three readings of mat-pcp's clade set name the same clades.

    And the scorer reads the same set the guard validated. ``clade_roster`` is
    what ``score_run`` loops over, so a guard that checked a different reading
    of the same directory would pass while the scorer covered a different set of
    clades -- the one failure a full-looking table cannot show.
    """
    founders = check_clades.founder_clades(lib.mat_pcp_path("clade_founders_dir"))
    fitness = check_clades.fitness_clades(lib.mat_pcp_path("aamut_fitness_dir"))
    censored = check_clades.censor_clades(lib.mat_pcp_path("censored_clades_file"))
    assert founders == fitness == censored
    assert sorted(founders) == lib.clade_roster()
    assert len(founders) == EXPECTED_CLADES


def test_pooled_table_is_excluded_from_the_clade_roster(tmp_path):
    """The pooled table and the per-arm refits are not clades.

    Both live inside ``aamut_fitness_dir``. The pooled table's column order
    differs -- its field 2 is ``clade_founder_aa`` where the per-clade files
    carry ``gene`` -- so sweeping it in reads a single-letter residue as a gene
    name, and the per-arm refits would double-count every clade.
    """
    (tmp_path / "21J_aamut_fitness.csv").write_text("header\n")
    (tmp_path / "all_aamut_fitness.csv").write_text("header\n")
    refit = tmp_path / "cap_5000_0"
    refit.mkdir()
    (refit / "21L_aamut_fitness.csv").write_text("header\n")
    assert check_clades.fitness_clades(tmp_path) == {"21J"}
