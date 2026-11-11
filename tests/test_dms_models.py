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
import dms_models
import panel_charts
import panel_spec

from dasm2.sequence.constants import AA_STR_SORTED

#: The cells each homolog's assay measures, after `load_dms`'s filters. The same
#: three numbers `tests/test_dms.py` pins as retained substitutions, restated
#: here because this file asserts the *join* preserves them and a join that
#: silently dropped a homolog would otherwise pass.
EXPECTED_CELLS = {"BA.1": 7202, "XBB.1.5": 6204, "KP.3.1.1": 9803}

_DMS_PATHS = tuple(lib.dms_dataset(homolog).path for homolog in lib.dms_homologs())


def _dms_readable() -> bool:
    if not _DMS_PATHS:
        return False
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
    """A reindexed matrix scores every substitution as some other substitution."""
    substitution_matrices = pytest.importorskip("Bio.Align").substitution_matrices
    blosum = substitution_matrices.load("BLOSUM62")
    matrix = dms_models.blosum62_log_odds()
    assert matrix.shape == (20, 20)
    for i, parent in enumerate(AA_STR_SORTED):
        for j, mutant in enumerate(AA_STR_SORTED):
            assert matrix[i, j] == blosum[parent, mutant]
    # Asymmetric indexing would pass a symmetric-matrix check, so pin a cell
    # whose residues sit far apart in `AA_STR_SORTED`: W is last, A first.
    assert (
        matrix[AA_STR_SORTED.index("A"), AA_STR_SORTED.index("W")] == blosum["A", "W"]
    )


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


def test_the_predictor_ids_the_rules_resolve_are_the_ones_the_module_scores():
    """`panel_spec` spells these without importing the module that defines them."""
    assert tuple(panel_spec.DMS_PREDICTORS) == dms_models.PREDICTORS


def test_every_predictor_has_a_label_and_a_colour():
    """A predictor missing either reaches a chart that recycles a hue or prints an id."""
    for predictor in dms_models.PREDICTORS:
        label = panel_charts.predictor_label(predictor)
        assert label in panel_charts.DMS_PREDICTOR_COLORS
    with pytest.raises(KeyError):
        panel_charts.predictor_label("no-such-predictor")


@needs_dms
def test_the_join_preserves_every_measured_cell_and_scores_each_once():
    """A join that dropped or duplicated a cell would report its own key as coverage."""
    for homolog, expected in EXPECTED_CELLS.items():
        cells = dms_models.homolog_cells(homolog)
        assert len(cells) == expected
        frame, manifest, founder = dms_models.blosum62_scores(homolog)
        merged, manifests = dms_models.join_scores(cells, [(frame, manifest)])
        assert len(merged) == expected
        assert merged[dms_models.BLOSUM62].notna().all()
        assert set(manifests) == {dms_models.BLOSUM62}
        # The parent the score was taken away from is the clade founder's
        # residue, which is what makes this predictor comparable to the DASM's.
        assert (merged["parent_aa"] == frame["parent_aa"]).all()
        assert manifest.input_sequence_sha256 == dms_models.sequence_sha256(founder)


@needs_dms
def test_a_predictor_that_scores_a_subset_leaves_the_rest_missing():
    """An absent cell must be NaN, not a zero a correlation would read as a value."""
    homolog = lib.dms_homologs()[0]
    cells = dms_models.homolog_cells(homolog)
    frame, manifest, _ = dms_models.blosum62_scores(homolog)
    half = frame.iloc[: len(frame) // 2]
    merged, _ = dms_models.join_scores(cells, [(half, manifest)])
    assert len(merged) == len(cells)
    assert merged[dms_models.BLOSUM62].notna().sum() == len(half)
