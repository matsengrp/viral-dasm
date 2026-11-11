"""What a wrong per-haplotype pass would produce, and what catches it.

The failure this file exists for does not raise and does not look wrong.
:class:`SelectionFactorSource` caches on ``(gene, run_id, clade)`` with no
sequence in the key, so a haplotype loop that passed the same clade twice would
be handed the first haplotype's ``[L, 20]`` table for the second one -- a
full-looking grid, correct shape, every column populated, describing the wrong
background. Every comparison downstream would then read a shift of exactly zero
and call it a model that predicts no shift.

The check that needs a trained checkpoint skips when none is present; the
label, validation and schema checks run always.
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

import score_haplotypes

#: The run whose checkpoint this reads when one exists. The
#: context-independent baseline, whose head is a function of
#: ``(aa_site, parent_aa, mutant_aa)`` alone -- so a one-residue change to the
#: parent must move that site's row and leave every other site alone, which is
#: a sharper assertion than "the tables differ somewhere".
CI_RUN = "ci-484k_full_depth"

#: The clade the two synthetic backgrounds are built from.
CLADE = "21J"

#: The site the two backgrounds differ at, and what the second carries there.
#: A site well inside the gene, so neither end's padding is involved.
SPLIT_SITE = 484
SPLIT_AA = "W"


def ci_run_dir() -> Path:
    """The baseline's run directory, or skip.

    ``VIRAL_DASM_REQUIRE_CHECKPOINT=1`` turns the skip into a failure, for
    merge time: while the campaign is mid-flight the skip is right, and after
    it lands a silent ``s`` is the one outcome that looks like a pass.
    """
    run_dir = Path(lib.ARTIFACT_ROOT) / "runs" / CI_RUN
    if not (run_dir / "training_complete.done").is_file():
        message = f"{CI_RUN} has not finished training (looked in {run_dir})"
        if os.environ.get("VIRAL_DASM_REQUIRE_CHECKPOINT") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return run_dir


@pytest.fixture(scope="module")
def two_backgrounds() -> tuple[str, str]:
    """One clade's founder, and the same sequence with one residue replaced."""
    from founders import founder_aa_sequence

    founder = founder_aa_sequence(CLADE, lib.GENE)
    if founder[SPLIT_SITE - 1] == SPLIT_AA:
        raise AssertionError(
            f"{CLADE} already carries {SPLIT_AA} at site {SPLIT_SITE}, so the "
            f"two backgrounds would be identical and the test would pass "
            f"vacuously."
        )
    alternative = founder[: SPLIT_SITE - 1] + SPLIT_AA + founder[SPLIT_SITE:]
    assert len(alternative) == len(founder)
    return founder, alternative


@pytest.fixture(scope="module")
def scored_two_backgrounds(two_backgrounds) -> pd.DataFrame:
    """Both backgrounds scored through **one** source, which is the point.

    Two sources would cache separately and could not fail this test however
    the key was spelled.
    """
    pytest.importorskip("torch")
    run_dir = ci_run_dir()
    founder, alternative = two_backgrounds
    source = score_haplotypes.SelectionFactorSource(
        {(lib.GENE, CI_RUN): lib.select_checkpoint(run_dir)}, "cpu"
    )
    try:
        frames = [
            score_haplotypes.score_haplotype(
                source,
                gene=lib.GENE,
                run_id=CI_RUN,
                clade=CLADE,
                background=label,
                aa_sequence=sequence,
                model="ci-484k",
                corpus="full_depth",
                masked=set(),
                censored=set(),
                censored_directional=set(),
            )
            for label, sequence in (("founder", founder), ("alternative", alternative))
        ]
    finally:
        source.release()
    return pd.concat(frames, ignore_index=True)


def test_two_backgrounds_of_one_clade_do_not_share_a_cache_entry(
    scored_two_backgrounds,
):
    """The regression this module exists for.

    Delete the composite key in ``score_haplotypes.cache_label`` and the two
    halves become identical, which is the silent failure.
    """
    halves = {
        label: half.set_index(["aa_site", "mutant_aa"])["log_selection_factor"]
        for label, half in scored_two_backgrounds.groupby("background")
    }
    assert set(halves) == {"founder", "alternative"}
    assert not halves["founder"].equals(halves["alternative"])


def test_the_split_site_is_what_moved(scored_two_backgrounds):
    """And nothing else did.

    The baseline's head reads only ``(aa_site, parent_aa, mutant_aa)``, so
    replacing one residue must move that one site and leave the rest untouched.
    A cache that keyed on the sequence but mis-indexed it would differ in the
    wrong places and still pass the test above.
    """
    halves = {
        label: half.set_index(["aa_site", "mutant_aa"])["log_selection_factor"]
        for label, half in scored_two_backgrounds.groupby("background")
    }
    differs = halves["founder"].ne(halves["alternative"])
    moved = sorted({site for site, _ in differs[differs].index})
    assert moved == [SPLIT_SITE]


def test_parent_aa_is_the_haplotypes_residue_not_the_founders(
    scored_two_backgrounds,
):
    """The one column whose meaning differs from the founder table's.

    Carrying the founder's residue here would make every pool's rows join to
    the founder's cells and report no shift at the site that defines the pool.
    """
    alternative = scored_two_backgrounds[
        scored_two_backgrounds["background"] == "alternative"
    ]
    at_split = alternative[alternative["aa_site"] == SPLIT_SITE]
    assert set(at_split["parent_aa"]) == {SPLIT_AA}


def test_background_is_written_for_every_row(scored_two_backgrounds):
    """The column `score_run.py` reserves and leaves empty is filled here."""
    assert (scored_two_backgrounds["background"].str.len() > 0).all()


def test_no_haplotype_sequence_reaches_the_table(scored_two_backgrounds):
    """A background is identified by label; its sequence is GISAID-derived.

    `aa_sequence` is read, scored and dropped. The guard is on length rather
    than on the column name, so re-adding it under any name fails here.
    """
    residues = set("ACDEFGHIKLMNPQRSTVWY")
    for column in scored_two_backgrounds.columns:
        values = scored_two_backgrounds[column]
        if values.dtype != object:
            continue
        longest = values.astype(str).str.len().max()
        assert longest <= 40, f"{column} holds a {longest}-character string"
        assert not any(
            len(value) > 40 and set(value) <= residues
            for value in values.astype(str).unique()
        )


def test_cache_label_refuses_a_separator_it_cannot_undo():
    """A clade or label carrying `/` would make the composite ambiguous."""
    assert score_haplotypes.cache_label("21J", "founder") == "21J/founder"
    with pytest.raises(ValueError, match="'/'"):
        score_haplotypes.cache_label("21J/x", "founder")
    with pytest.raises(ValueError, match="'/'"):
        score_haplotypes.cache_label("21J", "founder/x")


def test_a_duplicate_background_is_refused(tmp_path):
    """Two rows under one label would double the table from one pass."""
    path = tmp_path / "backgrounds.csv"
    pd.DataFrame(
        {
            "clade": ["21J", "21J"],
            "background": ["founder", "founder"],
            "aa_sequence": ["MA", "MC"],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        score_haplotypes.read_backgrounds(path)


def test_a_backgrounds_table_missing_a_column_is_refused(tmp_path):
    """Named rather than positional, so an added column cannot shift a read."""
    path = tmp_path / "backgrounds.csv"
    pd.DataFrame({"clade": ["21J"], "background": ["founder"]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="aa_sequence"):
        score_haplotypes.read_backgrounds(path)
