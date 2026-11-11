"""What a wrong per-haplotype pass would produce, and what catches it.

The failure this file exists for does not raise and does not look wrong.
:class:`SelectionFactorSource` caches on ``(gene, run_id, clade)`` with no
sequence in the key, so a haplotype loop that passed the same clade twice would
be handed the first haplotype's ``[L, 20]`` table for the second one -- a
full-looking grid, correct shape, every column populated, describing the wrong
background. Every comparison downstream would then read a shift of exactly zero
and call it a model that predicts no shift.

Three tests catch that by different routes: the two halves must differ at all;
the one replaced residue must be the only site that moved, which holds for this
baseline's head; and each row's columns must agree with each other, which holds
for any model and is what a stale cached frame breaks.

The checks needing a trained checkpoint skip when none is present. The reader's
validation checks need pandas but no weights.
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
#: ``(aa_site, parent_aa, mutant_aa)`` alone, which is what lets
#: :func:`test_the_split_site_is_what_moved` be exact.
CI_RUN = "ci-484k_full_depth"

#: The clade the two synthetic backgrounds are built from.
CLADE = "21J"

#: The site the two backgrounds differ at, and what the second carries there.
#: A site well inside the gene, so neither end's padding is involved.
SPLIT_SITE = 484
SPLIT_AA = "W"

#: Three distinct one-site mask sets, so a pass that crossed two of the three
#: flag columns fails rather than agreeing with itself.
MASKED_SITE = 7
CENSORED_SITE = 11
DIRECTIONAL_SITE = 13


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


def backgrounds_parquet(path: Path, rows: list[dict]) -> Path:
    """A backgrounds table on disk, for the reader's validation checks."""
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


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

    Two sources would cache separately and could not fail these tests however
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
                masked={MASKED_SITE},
                censored={CENSORED_SITE},
                censored_directional={DIRECTIONAL_SITE},
            )
            for label, sequence in (("founder", founder), ("alternative", alternative))
        ]
    finally:
        source.release()
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def halves(scored_two_backgrounds) -> dict[str, pd.Series]:
    """Each background's selection factors, keyed for comparison."""
    return {
        label: half.set_index(["aa_site", "mutant_aa"])["log_selection_factor"]
        for label, half in scored_two_backgrounds.groupby("background")
    }


def test_two_backgrounds_of_one_clade_do_not_share_a_cache_entry(halves):
    """The regression this module exists for.

    Delete the composite key in ``score_haplotypes.cache_label`` and the two
    halves become identical, which is the silent failure.
    """
    assert set(halves) == {"founder", "alternative"}
    assert not halves["founder"].equals(halves["alternative"])


def test_the_split_site_is_what_moved(halves):
    """And nothing else did.

    This baseline's head reads only ``(aa_site, parent_aa, mutant_aa)``, so
    replacing one residue must move that one site and leave the rest untouched.
    A cache that keyed on the sequence but mis-indexed it would differ in the
    wrong places and still pass the test above.

    **The exactness is a property of this checkpoint, not of the producer.** A
    transformer's one-residue change legitimately moves many sites, so this
    assertion must not be carried onto a transformer's table.
    """
    differs = halves["founder"].ne(halves["alternative"])
    moved = sorted({site for site, _ in differs[differs].index})
    assert moved == [SPLIT_SITE]


def test_each_row_agrees_with_its_own_parent_residue(scored_two_backgrounds):
    """The model-independent version of the cache check.

    ``parent_aa`` is rebuilt from the sequence this call was given, while
    ``aa_mutation``, ``is_synonymous`` and the gauge zero ride in the frame the
    cache returned. A stale frame therefore disagrees with ``parent_aa`` row by
    row -- which catches the bug without relying on this baseline's head being
    context-independent, so it keeps holding once the producer runs on a
    transformer.
    """
    frame = scored_two_backgrounds
    expected = frame["parent_aa"] + frame["aa_site"].astype(str) + frame["mutant_aa"]
    assert list(frame["aa_mutation"]) == list(expected)
    assert list(frame["is_synonymous"]) == list(
        frame["mutant_aa"] == frame["parent_aa"]
    )

    synonymous = frame[frame["is_synonymous"]]
    assert len(synonymous) == 2 * frame["aa_site"].nunique()
    assert synonymous["log_selection_factor"].abs().max() == pytest.approx(
        0.0, abs=1e-9
    )


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


def test_the_three_mask_flags_land_on_their_own_sites(scored_two_backgrounds):
    """Three distinct sets, so crossing two of the three columns fails.

    With one shared set -- or with all three empty -- swapping `censored` and
    `censored_directional` in the producer would agree with itself.
    """
    frame = scored_two_backgrounds
    for column, site in (
        ("globally_masked", MASKED_SITE),
        ("censored", CENSORED_SITE),
        ("censored_directional", DIRECTIONAL_SITE),
    ):
        flagged = sorted(frame.loc[frame[column], "aa_site"].unique())
        assert flagged == [site], f"{column} flagged {flagged}, expected [{site}]"


def test_the_written_columns_are_exactly_what_is_declared(scored_two_backgrounds):
    """A column added to the output fails here.

    This is the guard that is worth having. The haplotype *is* reconstructable
    from ``parent_aa`` by design and the module docstring says so, so what must
    not happen is a new column appearing in a table whose schema the counts
    side joins against. An equality on the declared tuple catches that
    directly, where scanning values for long strings did not: under pandas 3 a
    string column's dtype is `str` rather than `object`, so a scan written
    against `object` inspects nothing at all and passes whatever is added.
    """
    assert list(scored_two_backgrounds.columns) == list(score_haplotypes.OUTPUT_COLUMNS)
    assert "aa_sequence" not in scored_two_backgrounds.columns

    residues = set("ACDEFGHIKLMNPQRSTVWY")
    inspected = 0
    for column in scored_two_backgrounds.columns:
        values = scored_two_backgrounds[column]
        if not pd.api.types.is_string_dtype(values):
            continue
        inspected += 1
        longest = int(values.str.len().max())
        assert longest <= 16, f"{column} holds a {longest}-character string"
        assert not any(
            len(value) > 16 and set(value) <= residues
            for value in values.unique().tolist()
        )
    # Without this the loop can go quiet again the next time pandas changes how
    # it spells a string column, and the test would still pass.
    assert inspected >= 5


def test_cache_label_refuses_a_separator_it_cannot_undo():
    """A clade or label carrying `/` would make the composite ambiguous."""
    assert score_haplotypes.cache_label("21J", "founder") == "21J/founder"
    with pytest.raises(ValueError, match="'/'"):
        score_haplotypes.cache_label("21J/x", "founder")
    with pytest.raises(ValueError, match="'/'"):
        score_haplotypes.cache_label("21J", "founder/x")


def test_a_duplicate_background_is_refused(tmp_path):
    """Two rows under one label would double the table from one pass."""
    path = backgrounds_parquet(
        tmp_path / "backgrounds.parquet",
        [
            {"clade": "21J", "background": "founder", "aa_sequence": "MA"},
            {"clade": "21J", "background": "founder", "aa_sequence": "MC"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate"):
        score_haplotypes.read_backgrounds(path, sites=2)


def test_a_backgrounds_table_missing_a_column_is_refused(tmp_path):
    """Named rather than positional, so an added column cannot shift a read.

    The match names the reported column, not just the required list, which the
    message also carries.
    """
    path = backgrounds_parquet(
        tmp_path / "backgrounds.parquet",
        [{"clade": "21J", "background": "founder"}],
    )
    with pytest.raises(ValueError, match=r"missing \['aa_sequence'\]"):
        score_haplotypes.read_backgrounds(path, sites=2)


def test_an_empty_background_label_is_refused(tmp_path):
    """Empty is the founder table's own spelling of "this is a founder"."""
    path = backgrounds_parquet(
        tmp_path / "backgrounds.parquet",
        [{"clade": "21J", "background": "", "aa_sequence": "MA"}],
    )
    with pytest.raises(ValueError, match="empty 'background'"):
        score_haplotypes.read_backgrounds(path, sites=2)


def test_a_haplotype_of_the_wrong_length_is_refused_before_inference(tmp_path):
    """Otherwise it raises inside the grid builder, after the forward pass."""
    path = backgrounds_parquet(
        tmp_path / "backgrounds.parquet",
        [{"clade": "21J", "background": "founder", "aa_sequence": "MA"}],
    )
    with pytest.raises(ValueError, match="2-residue haplotype"):
        score_haplotypes.read_backgrounds(path, sites=3)


def test_an_ambiguity_code_is_refused_without_echoing_the_sequence(tmp_path):
    """`X` or a gap reaches the model as a silently wrong input."""
    path = backgrounds_parquet(
        tmp_path / "backgrounds.parquet",
        [{"clade": "21J", "background": "founder", "aa_sequence": "MX"}],
    )
    with pytest.raises(ValueError, match="which is not") as raised:
        score_haplotypes.read_backgrounds(path, sites=2)
    assert "MX" not in str(raised.value)


def test_extra_columns_are_accepted(tmp_path):
    """Found by name has to mean a wider table is still readable."""
    path = backgrounds_parquet(
        tmp_path / "backgrounds.parquet",
        [
            {
                "clade": "21J",
                "background": "founder",
                "aa_sequence": "MA",
                "n_branches": 1234,
            }
        ],
    )
    frame = score_haplotypes.read_backgrounds(path, sites=2)
    assert list(frame["background"]) == ["founder"]


def test_an_empty_backgrounds_table_is_refused(tmp_path):
    """A pass over nothing writes an empty table that looks like a result."""
    path = tmp_path / "backgrounds.parquet"
    pd.DataFrame(
        {"clade": [], "background": [], "aa_sequence": []}, dtype="object"
    ).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="empty"):
        score_haplotypes.read_backgrounds(path, sites=2)
