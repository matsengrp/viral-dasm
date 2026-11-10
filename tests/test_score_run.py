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

#: The run whose checkpoint these checks read when one exists. The
#: context-independent baseline, because its head is a function of
#: ``(aa_site, parent_aa, mutant_aa)`` alone, which is a property no
#: transformer has and which a wrong clade key would break.
CI_RUN = "ci-484k_full_depth"


def ci_run_dir() -> Path:
    """The baseline's run directory, or skip."""
    run_dir = Path(lib.ARTIFACT_ROOT) / "runs" / CI_RUN
    if not (run_dir / "training_complete.done").is_file():
        pytest.skip(f"{CI_RUN} has not finished training")
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
        del torch
    return pd.concat(frames, ignore_index=True)


def test_context_independent_head_has_no_clade_spread(scored_two_clades):
    """``ci-484k``'s selection factor cannot depend on which clade it scored.

    Its head is a lookup on ``(aa_site, parent_aa, mutant_aa)``, so two clades
    whose founders agree at a site must get identical numbers there. Any spread
    means clade identity reached the forward pass -- or, more likely, that a
    cache keyed on the clade handed back another clade's table.
    """
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
        f"pass or the SF cache answered for the wrong one."
    )


def test_synonymous_rows_are_the_gauge_zero(scored_two_clades):
    """Wildtype-to-wildtype is exactly 0, not approximately.

    ``selection_factors_of_aa_str`` zaps the diagonal, fixing the factor at
    exactly 1.0 at the wildtype. A synonymous row that is merely small means
    the grid's ``parent_aa`` and the sequence the model conditioned on have
    drifted apart -- the one failure the row count cannot show.
    """
    syn = scored_two_clades[scored_two_clades["is_synonymous"]]
    assert len(syn) == 2 * 1273
    assert (syn["parent_aa"] == syn["mutant_aa"]).all()
    assert syn["log_selection_factor"].abs().max() == 0.0


def test_grid_is_the_whole_grid_and_unique(scored_two_clades):
    """Every (site, amino acid) cell, once, with a prediction in it."""
    assert len(scored_two_clades) == 2 * 1273 * 20
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


def test_founder_length_is_checked():
    """A founder for the wrong gene fails at the length check.

    ``GENE_NT_LENGTH`` is what makes a truncated FASTA or a mismatched gene
    raise here rather than at the join, where it would instead shift every
    ``aa_site`` and silently mis-pair mutations.
    """
    with pytest.raises(KeyError):
        score_run.founder_aa_sequence("21J", "NOT_A_GENE")


def test_count_gate_actually_drops_rows():
    """The count filter bites, under the name the library canonicalizes to.

    A positive control for the quiet failure: ``filter_merged_fitness`` gates
    on a count column only ``if ... in df.columns``, so a frame still spelling
    mat-pcp's ``expected_count`` would gate on ``actual_count`` alone and
    report a wrong population with no error.
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
    """The three readings of mat-pcp's clade set name the same clades."""
    founders = check_clades.founder_clades(lib.mat_pcp_path("clade_founders_dir"))
    fitness = check_clades.fitness_clades(lib.mat_pcp_path("aamut_fitness_dir"))
    censored = check_clades.censor_clades(lib.mat_pcp_path("censored_clades_file"))
    assert founders == fitness == censored
    assert len(founders) > 1


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


def test_founder_sentinel_is_excluded_from_the_clade_roster(tmp_path):
    """``founders.done`` is mat-pcp's completion marker, not a clade."""
    (tmp_path / "21J.fa").write_text(">x\nATG\n")
    (tmp_path / "founders.done").write_text("")
    assert check_clades.founder_clades(tmp_path) == {"21J"}
