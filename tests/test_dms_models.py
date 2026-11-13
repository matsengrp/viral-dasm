"""What a wrong predictor join would report, and what catches it.

Three failure modes here produce a full-looking panel. A predictor scored
against a sequence other than the one it declares correlates the right numbers
against the wrong cells; a join whose key is not unique multiplies rows and
raises every ``n`` without raising anything; and a predictor with no colour or
no label reaches a chart that recycles a hue or prints ``leaf_logfreq`` on a
paper figure. None of them raises on its own.

The checks over hand-built frames run always. The ones that read this
repository's own mat-pcp run or the three DMS CSVs skip when those are not
readable, so the suite runs off-cluster.
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

import dms
import dms_antigen
import dms_blosum
import dms_leaf_scores
import dms_models
import founders
import panel_charts
import panel_spec

from dasm2.sequence.constants import AA_STR_SORTED

#: The cells each homolog's assay measures, after `load_dms`'s filters. The same
#: three numbers `tests/test_dms.py` pins as retained substitutions, restated
#: here because this file asserts the *join* preserves them and a join that
#: silently dropped a homolog would otherwise pass.
EXPECTED_CELLS = {"BA.1": 7202, "XBB.1.5": 6204, "KP.3.1.1": 9803}

_DMS_PATHS = tuple(lib.dms_dataset(homolog).path for homolog in lib.dms_homologs())

# Asserted at import, not probed: an empty roster would make every per-homolog
# loop below a pass over zero homologs, and a roster that did not match the pins
# would leave a new homolog unchecked and a removed one still asserted. Neither
# is a reason to skip -- both are config bugs.
assert _DMS_PATHS, "config.yaml names no DMS homologs"
assert set(EXPECTED_CELLS) == set(lib.dms_homologs()), (
    f"EXPECTED_CELLS covers {sorted(EXPECTED_CELLS)} against config.yaml's "
    f"{sorted(lib.dms_homologs())}"
)


def _dms_readable() -> bool:
    try:
        return all(path.is_file() for path in _DMS_PATHS)
    except OSError:
        return False


needs_dms = pytest.mark.skipif(
    not _dms_readable(), reason="the DMS CSVs are not readable here"
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    """A score frame in the contract's own columns."""
    return pd.DataFrame(rows, columns=list(dms_models.PREDICTOR_COLUMNS))


def _manifest(model: str, sequence: str) -> dms_models.Manifest:
    return dms_models.Manifest(
        model=model,
        strata=dms.STRATA,
        frame="hand-built",
        input_sequence_sha256=dms_models.sequence_sha256(sequence),
    )


def test_blosum62_is_biopythons_table_in_this_repositorys_residue_order():
    """A reindexed matrix scores every substitution as some other substitution.

    BLOSUM62 is symmetric, so a transposed read would pass unnoticed; what this
    catches is a row or column order that is not ``AA_STR_SORTED``, which is the
    layout every selection-factor row in this pipeline uses.
    """
    from Bio.Align import substitution_matrices

    blosum = substitution_matrices.load("BLOSUM62")
    matrix = dms_blosum.blosum62_log_odds()
    assert matrix.shape == (20, 20)
    for i, parent in enumerate(AA_STR_SORTED):
        for j, mutant in enumerate(AA_STR_SORTED):
            assert matrix[i, j] == blosum[parent, mutant]


def test_a_declared_hash_that_is_not_the_sequences_is_refused(tmp_path):
    """The one guard against a predictor scored on something else entirely."""
    sequence = "MFV"
    frame = _frame(
        [
            {
                "homolog": lib.dms_homologs()[0],
                "aa_site": 1,
                "parent_aa": "M",
                "mutant_aa": "A",
                "model": "hand",
                "score": 0.0,
            }
        ]
    )
    wrong = dms_models.Manifest(
        model="hand",
        strata=dms.STRATA,
        frame="hand-built",
        input_sequence_sha256=dms_models.sequence_sha256("MFA"),
    )
    with pytest.raises(ValueError, match="not the hash of the sequence"):
        dms_models.write_scores(frame, wrong, sequence, tmp_path / "scores.parquet")

    # And the same frame with the right hash writes and round-trips, so the
    # refusal above is the hash and not the frame.
    dms_models.write_scores(
        frame, _manifest("hand", sequence), sequence, tmp_path / "scores.parquet"
    )
    read, manifest = dms_models.read_scores(tmp_path / "scores.parquet")
    assert manifest == _manifest("hand", sequence)
    assert list(read.columns) == list(dms_models.PREDICTOR_COLUMNS)


def test_a_parent_residue_the_declared_sequence_does_not_have_is_refused(tmp_path):
    """A frame-shifted predictor still declares a sequence; this is what notices."""
    sequence = "MFV"
    frame = _frame(
        [
            {
                "homolog": lib.dms_homologs()[0],
                "aa_site": 2,
                "parent_aa": "M",  # the sequence has F at site 2
                "mutant_aa": "A",
                "model": "hand",
                "score": 0.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="parent_aa disagrees"):
        dms_models.write_scores(
            frame, _manifest("hand", sequence), sequence, tmp_path / "s.parquet"
        )


def test_a_repeated_key_is_refused_rather_than_multiplying_rows():
    """A duplicate key raises every `n` without raising anything by itself."""
    homolog = lib.dms_homologs()[0]
    rows = [
        {
            "homolog": homolog,
            "aa_site": 1,
            "parent_aa": "M",
            "mutant_aa": "A",
            "model": "hand",
            "score": value,
        }
        for value in (0.0, 1.0)
    ]
    with pytest.raises(ValueError, match="duplicate"):
        dms_models.validate_scores(_frame(rows), _manifest("hand", "M"))


def test_a_predictor_declaring_no_stratum_says_nothing():
    """An empty `strata` reads downstream as absence, so it must not be writable."""
    homolog = lib.dms_homologs()[0]
    frame = _frame(
        [
            {
                "homolog": homolog,
                "aa_site": 1,
                "parent_aa": "M",
                "mutant_aa": "A",
                "model": "hand",
                "score": 0.0,
            }
        ]
    )
    empty = dms_models.Manifest(
        model="hand",
        strata=(),
        frame="hand-built",
        input_sequence_sha256=dms_models.sequence_sha256("M"),
    )
    with pytest.raises(ValueError, match="declares no strata"):
        dms_models.validate_scores(frame, empty)


def test_the_predictor_ids_the_rules_resolve_are_the_ones_the_modules_score():
    """`panel_spec` spells these without importing the modules that define them.

    It is stdlib-plus-`lib` so `workflow/figures.smk` stays cheap to parse, so
    the ids are a hand-copied tuple there and this is what ties them to the
    predictors that actually write them.
    """
    assert tuple(panel_spec.DMS_PREDICTORS) == (
        dms_blosum.BLOSUM62,
        dms_leaf_scores.LEAF_LOGFREQ,
        dms_antigen.ANTIGEN,
    )


def test_every_predictor_has_a_label_and_a_colour():
    """A predictor missing either reaches a chart that recycles a hue or prints an id."""
    for predictor in panel_spec.DMS_PREDICTORS:
        label = panel_charts.predictor_label(predictor)
        assert label in panel_charts.DMS_PREDICTOR_COLORS
    with pytest.raises(KeyError):
        panel_charts.predictor_label("no-such-predictor")


@needs_dms
def test_the_join_preserves_every_measured_cell_and_scores_each_once():
    """A join that dropped or duplicated a cell would report its own key as coverage."""
    for homolog, expected in EXPECTED_CELLS.items():
        cells, cells_founder = dms_models.homolog_cells(homolog)
        assert len(cells) == expected
        frame, manifest, founder = dms_blosum.blosum62_scores(homolog)
        merged, manifests = dms_models.join_scores(cells, [(frame, manifest)])
        assert len(merged) == expected
        assert merged[dms_blosum.BLOSUM62].notna().all()
        assert set(manifests) == {dms_blosum.BLOSUM62}
        assert manifest.input_sequence_sha256 == dms_models.sequence_sha256(founder)
        # One founder read, so the cells and the declared sequence cannot be two
        # different readings of the same FASTA.
        assert cells_founder == founder
        # And every score's parent is that founder's residue at that site,
        # compared against the FASTA rather than against the column's own source,
        # which would be a tautology.
        clade = lib.dms_dataset(homolog).clade
        expected_parents = founders.founder_aa_sequence(clade, lib.GENE)
        assert all(
            expected_parents[site - 1] == parent
            for site, parent in zip(merged["aa_site"], merged["parent_aa"])
        )


@needs_dms
def test_a_predictor_that_scores_a_subset_leaves_the_rest_missing():
    """An absent cell must be NaN, not a zero a correlation would read as a value."""
    homolog = lib.dms_homologs()[0]
    cells, _ = dms_models.homolog_cells(homolog)
    frame, manifest, _ = dms_blosum.blosum62_scores(homolog)
    half = frame.iloc[: len(frame) // 2]
    merged, _ = dms_models.join_scores(cells, [(half, manifest)])
    assert len(merged) == len(cells)
    assert merged[dms_blosum.BLOSUM62].notna().sum() == len(half)


@needs_dms
def test_blosum62_scores_the_cell_it_says_it_does():
    """The assertion the rest of this file does not make: the score's own value.

    Every other check here is a row count, a key or a hash, all of which a
    predictor that scored ``matrix[parent, parent]``, or scored the transpose, or
    dropped the lookup entirely would satisfy. A wrong value reaches panel C as a
    correlation of the wrong magnitude and nothing else notices.
    """
    from Bio.Align import substitution_matrices

    blosum = substitution_matrices.load("BLOSUM62")
    frame, _, _ = dms_blosum.blosum62_scores(panel_spec.SCATTER_HOMOLOG)
    for row in frame.head(200).itertuples():
        assert row.score == blosum[row.parent_aa, row.mutant_aa]
    # And not all one value, which a dropped lookup would be.
    assert frame["score"].nunique() > 1


def test_a_non_finite_score_is_refused():
    """`within_clade_metrics` drops NaN rows, so a bad predictor shrinks `n`."""
    homolog = lib.dms_homologs()[0]
    rows = [
        {
            "homolog": homolog,
            "aa_site": site,
            "parent_aa": "M",
            "mutant_aa": "A",
            "model": "hand",
            "score": score,
        }
        for site, score in ((1, 0.0), (2, float("nan")))
    ]
    with pytest.raises(ValueError, match="NaN or"):
        dms_models.validate_scores(_frame(rows), _manifest("hand", "MM"))


@needs_dms
def test_a_predictor_may_not_score_a_stratum_it_did_not_declare():
    """Otherwise the declaration says nothing and the extra rows read as coverage."""
    homolog = panel_spec.SCATTER_HOMOLOG
    cells, _ = dms_models.homolog_cells(homolog)
    frame, manifest, _ = dms_blosum.blosum62_scores(homolog)
    narrowed = dms_models.Manifest(
        model=manifest.model,
        strata=(dms.SINGLE_NT,),
        frame=manifest.frame,
        input_sequence_sha256=manifest.input_sequence_sha256,
    )
    with pytest.raises(ValueError, match="does not declare"):
        dms_models.join_scores(cells, [(frame, narrowed)])


def test_a_predictor_declaring_all_strata_is_not_refused():
    """``all`` is a union label no cell carries, so it has to mean both.

    It is the natural way for a predictor that scores everything to say so, and
    before the expansion it passed `validate_scores` and then raised in the join
    for scoring the two strata its own declaration covered.
    """
    assert dms_models.declared_cell_strata((dms.ALL,)) == {dms.SINGLE_NT, dms.MULTI_NT}
    assert dms_models.declared_cell_strata((dms.SINGLE_NT,)) == {dms.SINGLE_NT}
    assert dms_models.declared_cell_strata(dms.STRATA) == {dms.SINGLE_NT, dms.MULTI_NT}


@needs_dms
def test_a_predictor_scoring_everything_may_declare_all(tmp_path):
    """The end-to-end form of the above, through the join."""
    homolog = panel_spec.SCATTER_HOMOLOG
    cells, _ = dms_models.homolog_cells(homolog)
    frame, manifest, _ = dms_blosum.blosum62_scores(homolog)
    all_only = dms_models.Manifest(
        model=manifest.model,
        strata=(dms.ALL,),
        frame=manifest.frame,
        input_sequence_sha256=manifest.input_sequence_sha256,
    )
    merged, _ = dms_models.join_scores(cells, [(frame, all_only)])
    assert merged[dms_blosum.BLOSUM62].notna().all()


def test_the_bar_panel_refuses_a_metric_it_has_no_axis_title_for():
    """Otherwise a correlation is drawn under a title naming a different one."""
    frame = pd.DataFrame(
        {
            "homolog": ["BA.1"],
            "stratum": [dms.SINGLE_NT],
            "predictor": ["BLOSUM62"],
            "pearson_r": [0.2],
        }
    )
    with pytest.raises(KeyError, match="no axis title"):
        panel_charts.homolog_correlation_panel(
            frame,
            homolog_order=["BA.1"],
            stratum_order=[dms.SINGLE_NT],
            predictor_order=["BLOSUM62"],
            metric="kendall_tau",
        )
    # And the two it does know are both drawable.
    for metric in ("pearson_r", "spearman_rho"):
        panel_charts.homolog_correlation_panel(
            frame.assign(**{metric: 0.2}),
            homolog_order=["BA.1"],
            stratum_order=[dms.SINGLE_NT],
            predictor_order=["BLOSUM62"],
            metric=metric,
        )
