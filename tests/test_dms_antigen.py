"""What a wrong antiGen column would report, and what catches it.

Three failure modes here reach panel C as a correlation of the wrong magnitude
and raise nothing. The ``(position, amino-acid)`` axes can be transposed, or
the alphabet misordered, and every row count, key check and hash in
`scripts/dms_models.py`'s contract still passes. The model can be fed the clade
founder instead of the construct the assay expressed, and **nothing in that
contract can tell**: ``homolog_cells`` takes ``parent_aa`` from the founder and
every column where the two disagree has already been dropped, so the declared
sequence agrees with ``parent_aa`` either way. And the probability map can be
un-normalised, which moves every score by a constant and so is invisible to a
correlation but not to a reported ``log p``.

So the guards are a value assertion on the lookup, the query's gap count and
sha256, and the map's sum. The last is also what makes the choice of background
safe rather than lucky: the invariant that no measured cell sits at a column
where founder and construct differ is asserted here, per homolog, rather than
taken from the issue that found it.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import lib

import dasm2  # noqa: F401

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import dms_antigen
import dms_models

#: The construct projections `config.yaml` points at, as
#: ``dasm2-experiments`` #1962 records them: the number of deletion columns in
#: each, and the sha256 of the whole string. **This is the only live guard on
#: which background was scored** -- the clade founders carry no gaps at all, so
#: a rule quietly repointed at them fails the first number here, and a
#: projection rebuilt on a different deletion convention fails the second.
EXPECTED_QUERIES = {
    "BA.1": (
        6,
        "5f3398d44f945d0b62bed6f8bb602243007a02f0e666696f2aa2ed5cba053140",
    ),
    "XBB.1.5": (
        4,
        "92622b76bd6b5fb70b81e26e0d0778b48e9ce8e7bd173cf7c9757f20f3d134ae",
    ),
    "KP.3.1.1": (
        9,
        "de144a5e5047addeb4d788aae2dc3fffc07cf66a767106d61c12b7dc2c01818d",
    ),
}

#: The homolog the slow checks run on -- the one `main.tex`'s panel-C caption
#: names, and so the one whose number the paper quotes. One forward pass rather
#: than three: the lookup, the alphabet and the normalisation are per-model
#: facts, not per-homolog ones, and the per-homolog fact is the query, which
#: every homolog is checked on above.
SCORED_HOMOLOG = "KP.3.1.1"

#: Panel C's antiGen bar at :data:`SCORED_HOMOLOG`, stratum ``all``, as issue
#: #32's success criterion states it. Wide enough to survive a pandas or torch
#: version bump and narrow enough that a transposed axis or a founder
#: background -- which score around zero -- fails it.
EXPECTED_SPEARMAN = (0.34, 0.44)

_SPEC = lib.antigen_model()

needs_dms = pytest.mark.skipif(
    not all(lib.dms_dataset(h).path.is_file() for h in lib.dms_homologs()),
    reason="the DMS CSVs are not readable here",
)

needs_antigen = pytest.mark.skipif(
    not (_SPEC.checkout / "antiGen").is_dir(),
    reason=f"the antiGen checkout is not readable at {_SPEC.checkout}",
)


@pytest.fixture(scope="module")
def scored():
    """One real forward pass, shared. Loading the model costs about 20 seconds."""
    query = dms_antigen.read_parents(_SPEC.parents)[SCORED_HOMOLOG]
    probs, alphabet = dms_antigen.substitution_probabilities(
        _SPEC.checkout, _SPEC.checkpoint, query
    )
    return probs, alphabet, query


def test_the_lookup_reads_the_cell_it_says_it_does():
    """The assertion the contract cannot make: the score's own value.

    Built by hand so the expected number is arithmetic rather than another run
    of the code under test. A transposed axis, a misordered alphabet, or a
    lookup that dropped the mutant and read the wild-type column would satisfy
    every row count, key and hash `scripts/dms_models.py` checks, and would
    pass the normalisation check below as well.
    """
    alphabet = "ACDEFGHIKLMNPQRSTVWY*-"
    columns = dms_antigen.antigen_columns(alphabet)
    probs = np.full((5, len(alphabet)), 1e-9)
    # Site 3 (index 2) to W, and site 1 (index 0) to C, at values no other cell
    # carries. The transpose of each is a different cell, so a swapped axis
    # reads 1e-9 and fails.
    probs[2, alphabet.index("W")] = 0.25
    probs[0, alphabet.index("C")] = 0.5
    scores = dms_antigen.cell_scores(
        probs, columns, np.array([3, 1]), pd.Series(["W", "C"])
    )
    assert scores == pytest.approx([np.log(0.25), np.log(0.5)])


def test_the_alphabet_conversion_is_written_down_rather_than_assumed():
    """A reordered alphabet on either side transposes every lookup silently."""
    identity = dms_antigen.antigen_columns("ACDEFGHIKLMNPQRSTVWY*-")
    assert identity == tuple(range(20))
    # Reversed, so every column moves and the mapping has to follow it.
    reversed_20 = "YWVTSRQPNMLKIHGFEDCA"
    assert dms_antigen.antigen_columns(reversed_20) == tuple(range(19, -1, -1))
    with pytest.raises(ValueError, match="no column for"):
        dms_antigen.antigen_columns("ACDEFGHIKLMNPQRSTVW")


def test_a_zero_probability_is_refused_rather_than_logged():
    """``log(0)`` is ``-inf``, which `validate_scores` would report as a NaN count."""
    alphabet = "ACDEFGHIKLMNPQRSTVWY*-"
    columns = dms_antigen.antigen_columns(alphabet)
    probs = np.full((3, len(alphabet)), 0.1)
    probs[1, alphabet.index("K")] = 0.0
    with pytest.raises(ValueError, match="probability 0"):
        dms_antigen.cell_scores(probs, columns, np.array([2]), pd.Series(["K"]))


def test_the_committed_queries_are_the_construct_projections_they_claim():
    """Trap 1's only live guard: the gap count and the hash of what is scored.

    The founders carry no gaps, so a rule repointed at them fails the first
    number. Checked on the committed FASTA rather than on the issue that
    recorded the hashes, so a re-exported file has to still be the same string.
    """
    parents = dms_antigen.read_parents(_SPEC.parents)
    assert sorted(parents) == sorted(lib.dms_homologs())
    for homolog, (gaps, digest) in EXPECTED_QUERIES.items():
        query = parents[homolog]
        assert len(query) == 1273
        assert query.count("-") == gaps
        assert hashlib.sha256(query.encode()).hexdigest() == digest
        # And the manifest declares that same string, not the founder.
        assert dms_models.sequence_sha256(query) == digest


@needs_dms
def test_no_measured_cell_sits_where_founder_and_construct_disagree():
    """Why feeding the construct cannot move which cells are compared.

    ``homolog_cells`` supplies ``parent_aa`` from the clade founder while the
    model reads the construct projection, so a measured cell at a column where
    the two differ would be scored against one parent and reported against
    another -- and at a deletion column `write_scores` would refuse it, since
    the declared sequence carries ``-`` there and ``parent_aa`` cannot. This
    asserts the premise instead of trusting it: the disagreement is real (the
    counts below are not zero) and disjoint from the cells.
    """
    parents = dms_antigen.read_parents(_SPEC.parents)
    for homolog in lib.dms_homologs():
        cells, founder = dms_models.homolog_cells(homolog)
        query = parents[homolog]
        differing = {
            site
            for site in range(1, len(founder) + 1)
            if founder[site - 1] != query[site - 1]
        }
        assert differing, (
            f"{homolog}: the construct projection and the clade founder agree "
            f"everywhere, so this test has stopped distinguishing them"
        )
        assert not differing & set(cells["aa_site"]), (
            f"{homolog}: measured cells sit at "
            f"{sorted(differing & set(cells['aa_site']))}, where the founder "
            f"and the construct disagree. The background is no longer a free "
            f"choice: those cells are scored against one parent and reported "
            f"against another."
        )


@needs_antigen
def test_the_probability_map_is_normalised_before_the_log(scored):
    """An un-normalised map shifts every ``log p`` by a constant.

    Invisible to a correlation and visible in a reported number, which is why
    `scripts/dms_antigen.py` pins ``--drop-gap-stop``: that flag renormalises,
    and a reader comparing this column against another run's has to know which
    normalisation it is on.
    """
    probs, alphabet, _ = scored
    assert probs.shape == (1273, len(alphabet))
    assert probs.sum() == pytest.approx(1.0, abs=2e-7)
    # The stop and gap columns carry no mass, which is what the flag does.
    for token in ("*", "-"):
        assert probs[:, alphabet.index(token)].sum() == 0.0


@needs_antigen
@needs_dms
def test_antigen_clears_the_band_the_issue_predicted(scored):
    """The canary, on the pipeline's own cells rather than on the raw CSV.

    A transposed axis or a founder background both score near zero here while
    passing every other check in this file's contract, so this is the one
    assertion that holds the whole path to an answer rather than to a shape.
    """
    stats = pytest.importorskip("scipy.stats")
    frame, manifest, query = dms_antigen.antigen_scores(SCORED_HOMOLOG)
    cells, _ = dms_models.homolog_cells(SCORED_HOMOLOG)
    merged, _ = dms_models.join_scores(cells, [(frame, manifest)])

    assert merged[dms_antigen.ANTIGEN].notna().all()
    low, high = EXPECTED_SPEARMAN
    rho = stats.spearmanr(
        merged[dms_antigen.ANTIGEN], merged["dms_cell_entry"]
    ).statistic
    assert low <= rho <= high, (
        f"antiGen's Spearman against {SCORED_HOMOLOG} is {rho:+.4f}, outside "
        f"[{low}, {high}]. A transposed axis or the clade founder as the "
        f"background both land near zero; a real change in the model or the "
        f"cells moves it by less than this band's width."
    )
    # And the manifest declares the construct, not the founder.
    assert manifest.input_sequence_sha256 == EXPECTED_QUERIES[SCORED_HOMOLOG][1]
    assert query.count("-") == EXPECTED_QUERIES[SCORED_HOMOLOG][0]


@pytest.mark.parametrize("homolog", sorted(EXPECTED_QUERIES))
def test_each_scored_parent_is_in_antigens_training_split(homolog: str):
    """The reason this checkpoint is the grid's last cutoff and not another.

    antiGen's grid trains one model per time cutoff on the branches whose
    parent node predates it, so an earlier cutoff would score a DMS parent the
    model never saw. The last rung was chosen because all three are in sample,
    and in the *training* split rather than the held-out one. Read out of the
    authors' own dumped examples, which are not carried here, so this skips
    where they are unreadable -- and the sequence it looks for is the one
    :data:`EXPECTED_QUERIES` pins, so it cannot pass against a different string.
    """
    examples = (
        _SPEC.checkout.parent
        / "examples/output_sc2_grid/cutoff_5.939/examples_cutoff_5.93893.txt"
    )
    if not examples.is_file():
        pytest.skip(f"antiGen's dumped examples are not readable at {examples}")

    query = dms_antigen.read_parents(_SPEC.parents)[homolog]
    blocks = examples.read_text().split("EXAMPLE_ID: ")
    hits = [block for block in blocks if f"\n{query}\n" in block]
    assert len(hits) == 1, (
        f"{homolog}'s scored parent occurs {len(hits)} times in antiGen's "
        f"training examples; it must occur exactly once, or the node this "
        f"model saw is not the node being scored"
    )
    block = hits[0]
    assert "\nTRAINING\n" in block, (
        f"{homolog}'s scored parent is in the examples but not in the "
        f"training split, so panel C would report a held-out sequence"
    )
    children = block.split("MUTATIONS_TO_CHILDREN:\n")[1].strip()
    assert children, (
        f"{homolog}'s scored parent carries no mutations to children, so "
        f"antiGen's `ignore_dead_ends` drops it and the node contributes no "
        f"gradient however it is labelled"
    )
