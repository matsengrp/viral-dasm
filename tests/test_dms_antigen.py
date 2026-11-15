"""What a wrong antiGen column would report, and what catches it.

Three failure modes here reach the predictor table as a correlation of the
wrong magnitude and raise nothing. The ``(position, amino-acid)`` axes can be transposed, or
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
import os
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
import dms_published
from founders import founder_aa_sequence
from lib import GENE

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

#: The homolog the slow checks run on, and the one `scripts/panel_spec.py`
#: names as :data:`panel_spec.SCATTER_HOMOLOG`. One forward pass rather
#: than three: the lookup, the alphabet and the normalisation are per-model
#: facts, not per-homolog ones, and the per-homolog fact is the query, which
#: every homolog is checked on above.
SCORED_HOMOLOG = "KP.3.1.1"

#: The floor the antiGen row at :data:`SCORED_HOMOLOG` must clear, stratum
#: ``all``. **One-sided, deliberately.** The defects this guards against all
#: push the coefficient *down*: a transposed axis and a founder background
#: score near zero, and a one-site misjoin lands around +0.24. An upper bound
#: would guard nothing and would fail on a better model or a cleaner release,
#: which says nothing about correctness. Measured at +0.4095 when this landed,
#: so the floor sits well below that and well above the misjoin.
MINIMUM_SPEARMAN = 0.30

_SPEC = lib.antigen_model()

#: The house convention: on the host that builds the figure a skip is itself
#: the failure, because every check below that reaches the model or the assays
#: is gated on one of these. Raised at collection so the reason is the missing
#: path, not a pile of skip reasons. `tests/test_dms.py` sets the precedent.
REQUIRE_DATA = os.environ.get("VIRAL_DASM_REQUIRE_DATA") == "1"

needs_dms = pytest.mark.skipif(
    not all(lib.dms_dataset(h).path.is_file() for h in lib.dms_homologs()),
    reason="the DMS CSVs are not readable here",
)

needs_antigen = pytest.mark.skipif(
    not (_SPEC.checkout / "antiGen").is_dir(),
    reason=f"the antiGen checkout is not readable at {_SPEC.checkout}",
)


if REQUIRE_DATA:
    _absent = []
    if not all(lib.dms_dataset(h).path.is_file() for h in lib.dms_homologs()):
        _absent.append("the three DMS CSVs")
    if not (_SPEC.checkout / "antiGen").is_dir():
        _absent.append(f"the antiGen checkout at {_SPEC.checkout}")
    if _absent:
        raise RuntimeError(
            f"VIRAL_DASM_REQUIRE_DATA=1 but {' and '.join(_absent)} "
            f"cannot be read, so every check that reaches the model or the "
            f"assays would skip and the suite would pass having tested none "
            f"of them."
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
def test_the_zeroed_cell_at_each_position_is_the_querys_own_residue(scored):
    """The model's own output traces the query, which fixes the row axis.

    ``score_batch`` zeroes the wild-type cell at every position. So the pattern
    of zeros in the map *is* the query sequence, read out of the checkpoint's
    own forward pass rather than asserted against the string that was fed in.
    That makes this the check for an off-by-one in the row index and for a
    mis-ordered alphabet at the same time: shift the rows by one, or permute
    the columns, and the zeroed residue stops matching the query almost
    everywhere.

    The gap columns are excluded because `DROP_GAP_STOP` zeroes ``*`` and ``-``
    at every position, so at a deletion column there is no wild-type zero to
    find -- which is itself asserted, as the alternative.
    """
    probs, alphabet, query = scored
    for position, residue in enumerate(query):
        zeroed = {
            alphabet[column]
            for column in range(len(alphabet))
            if probs[position, column] == 0.0
        } - {"*", "-"}
        if residue == "-":
            assert not zeroed, (
                f"position {position + 1} is a deletion in the query but the "
                f"map zeroes {sorted(zeroed)} there"
            )
        else:
            assert zeroed == {residue}, (
                f"position {position + 1}: the map zeroes {sorted(zeroed)} but "
                f"the query carries {residue!r}. The row axis and the query "
                f"have come apart, which no row count or hash would catch."
            )


@needs_dms
def test_the_published_wildtype_column_pins_the_site_numbering():
    """What actually holds the DMS frame to the founder's, measured both ways.

    It is tempting to check this by comparing the scored query against
    ``parent_aa``. That proves nothing: `dms_models.homolog_cells` builds
    ``parent_aa`` as ``founder[site - 1]``, so both sides are indexed by the
    same ``aa_site`` and a shift in the study's own numbering moves them
    together. `test_no_measured_cell_sits_where_founder_and_construct_disagree`
    already covers what such a comparison does establish.

    The genuinely independent source is the study's own ``wildtype`` column,
    which `dms.load_dms` checks against the clade founder and drops on
    mismatch. This pins that check's discriminating power rather than assuming
    it: in frame the two agree almost everywhere, and one residue either way
    they agree at close to the background rate a random protein would give, so
    a shifted release loses nearly every measurement and `load_dms`'s
    accounting cannot miss it.
    """
    for homolog in lib.dms_homologs():
        dataset = lib.dms_dataset(homolog)
        founder = founder_aa_sequence(dataset.clade, GENE)
        published = dms_published.normalise_published(
            pd.read_csv(dataset.path), dataset
        )
        rates = {}
        for shift in (-1, 0, 1):
            pairs = [
                (founder[site - 1 + shift], wildtype)
                for site, wildtype in zip(published["site"], published["wildtype"])
                if 0 <= site - 1 + shift < len(founder)
            ]
            rates[shift] = sum(a == b for a, b in pairs) / len(pairs)
        assert rates[0] > 0.95, (
            f"{homolog}: the published wildtype column agrees with the "
            f"{dataset.clade} founder at only {rates[0]:.1%} of sites, so the "
            f"two are not in the same frame"
        )
        for shift in (-1, 1):
            assert rates[shift] < 0.25, (
                f"{homolog}: shifting the site index by {shift:+d} still agrees "
                f"at {rates[shift]:.1%}, so this check cannot tell a correct "
                f"frame from a shifted one"
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
def test_antigen_clears_the_band_the_issue_predicted():
    """The canary, on the pipeline's own cells rather than on the raw CSV.

    A transposed axis or a founder background both score near zero here while
    passing every other check in this file's contract, so this is the one
    assertion that holds the whole path to an answer rather than to a shape.
    """
    stats = pytest.importorskip("scipy.stats")
    frame, manifest, _ = dms_antigen.antigen_scores(SCORED_HOMOLOG)
    cells, _ = dms_models.homolog_cells(SCORED_HOMOLOG)
    merged, _ = dms_models.join_scores(cells, [(frame, manifest)])

    assert merged[dms_antigen.ANTIGEN].notna().all()
    rho = stats.spearmanr(
        merged[dms_antigen.ANTIGEN], merged["dms_cell_entry"]
    ).statistic
    assert rho >= MINIMUM_SPEARMAN, (
        f"antiGen's Spearman against {SCORED_HOMOLOG} is {rho:+.4f}, below "
        f"{MINIMUM_SPEARMAN}. A transposed axis or the clade founder as the "
        f"background both land near zero, and joining each score to the next "
        f"site's measurement lands around +0.24."
    )


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
        f"training split, so the table would report a held-out sequence"
    )
    children = block.split("MUTATIONS_TO_CHILDREN:\n")[1].strip()
    assert children, (
        f"{homolog}'s scored parent carries no mutations to children, so "
        f"antiGen's `ignore_dead_ends` drops it and the node contributes no "
        f"gradient however it is labelled"
    )
