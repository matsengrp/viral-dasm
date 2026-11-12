"""What the conversion from a study's own file has to get right.

The byte-for-byte check below is the one that matters most. Each of these
three datasets was prepared outside this repository until #30, and the
prepared tables are what every number `fig:dms` reports was computed from, so
reproducing them exactly is what makes this a port rather than a reanalysis.
The digests are written here rather than read from ``config.yaml``, which
pins the *inputs* now: after #30 nothing in the checkout records what the
conversion should produce except this file.

The rest are the mistakes a reimplementation actually makes. Each one was
written by breaking the implementation and checking the test caught it.
"""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

pytest.importorskip("pandas")

import dms
import lib
import pandas as pd
from dms_published import (
    DMS_EFFECT_COL,
    DMS_MUT_COL,
    DMS_SITE_COL,
    DMS_WT_COL,
    OUTPUT_COLUMNS,
    normalise_published,
    parse_aa_mutation,
    read_published,
)

#: md5 of ``normalise_published(...).to_csv(index=False)``, per homolog.
#:
#: These are the digests of the tables `dasm2-data-prep` published, carried
#: over from the `dms` block `config.yaml` held before #30.
EXPECTED_CONVERSION_MD5 = {
    "BA.1": "9f21c3ee991c1e7ea7a4362e510bc3c2",
    "XBB.1.5": "18c630548aa0e5d3b3a6e341bc8bbd26",
    "KP.3.1.1": "6cb4e58c8e61831fac6f470325b68a7c",
}

#: Rows each conversion emits.
#:
#: Not an independent check of the whole conversion, and it would be wrong to
#: read it as one: all nine of BA.1's regularization strengths yield exactly
#: 7,820 rows under nine different digests, so only the digest above catches a
#: wrong ``fixed`` value. What the row count does separate is the
#: ``times_seen`` floor, where the plausible wrong readings give 6,848 or
#: 10,123 instead.
EXPECTED_ROWS = {"BA.1": 7820, "XBB.1.5": 7099, "KP.3.1.1": 11841}


def _dataset(**overrides) -> lib.DmsDataset:
    """A dataset for a fixture frame; ``path`` is unused by the conversion."""
    fields = {
        "homolog": "fixture",
        "clade": "21K",
        "path": Path("/nonexistent/fixture.csv"),
        "md5": "0" * 32,
        "effect_column": DMS_EFFECT_COL,
    }
    fields.update(overrides)
    return lib.DmsDataset(**fields)


@pytest.mark.parametrize("homolog", sorted(EXPECTED_CONVERSION_MD5))
def test_the_conversion_reproduces_the_table_the_figure_was_built_from(homolog):
    dataset = lib.dms_dataset(homolog)
    converted = normalise_published(read_published(dataset), dataset)
    assert list(converted.columns) == list(OUTPUT_COLUMNS)
    assert len(converted) == EXPECTED_ROWS[homolog]
    # The docstring promises a reset index, and `to_csv(index=False)` cannot
    # see one: dropping `.reset_index(drop=True)` otherwise passes everything.
    assert converted.index.equals(pd.RangeIndex(len(converted)))
    digest = hashlib.md5(converted.to_csv(index=False).encode()).hexdigest()
    assert digest == EXPECTED_CONVERSION_MD5[homolog]


@pytest.mark.parametrize("homolog", sorted(EXPECTED_CONVERSION_MD5))
def test_a_changed_input_file_fails_the_pin(homolog, tmp_path, monkeypatch):
    """The gate fires, which `tests/test_dms.py` cannot show.

    That file asserts the file on disk matches its pin and that
    ``verify_dms_md5`` agrees -- the passing half, which a gate returning the
    pinned value unconditionally would also satisfy. This is the failing half,
    and it has to go through ``verify_dms_md5`` rather than compare digests
    itself, or it would hold for any file and any digest function.

    The copy goes under ``tmp_path`` and ``dms.root`` is repointed at it, so
    no committed file is touched: ``lib.dms_dataset`` joins ``root`` onto
    ``REPO_ROOT``, and an absolute root discards that prefix.
    """
    real = lib.dms_dataset(homolog)
    mutated = bytearray(real.path.read_bytes())
    mutated[-1] ^= 0x01
    (tmp_path / real.path.name).write_bytes(bytes(mutated))

    doctored = copy.deepcopy(lib.CONFIG)
    doctored["dms"]["root"] = str(tmp_path)
    doctored["dms"]["homologs"][homolog]["file"] = real.path.name
    monkeypatch.setattr(lib, "CONFIG", doctored)

    with pytest.raises(ValueError, match="has md5"):
        dms.verify_dms_md5(homolog)


def test_an_insertion_label_is_dropped_and_its_base_site_is_kept():
    """KP.3.1.1's ``16a``-``16d`` have no Wuhan-Hu-1 coordinate.

    Dropping them by position -- everything at site 16, say -- would pass a
    test that only counted rows, so this checks site 16's own measurements
    survive. The real file has eight of them.
    """
    dataset = lib.dms_dataset("KP.3.1.1")
    converted = normalise_published(read_published(dataset), dataset)
    assert (converted[DMS_SITE_COL] == 16).sum() == 8

    frame = pd.DataFrame(
        {
            DMS_SITE_COL: ["16", "16a", "17"],
            DMS_WT_COL: ["Q", "Q", "C"],
            DMS_MUT_COL: ["A", "A", "A"],
            DMS_EFFECT_COL: [-1.0, -2.0, -3.0],
        }
    )
    kept = normalise_published(frame, _dataset())
    assert list(kept[DMS_SITE_COL]) == [16, 17]


def test_the_times_seen_floor_is_a_row_filter_not_a_per_mutation_one():
    """The mistake most likely to be made, and it is silent.

    A mutation seen often enough in one replicate and not the other keeps the
    replicate that passed and averages over that alone. Requiring both
    replicates instead returns 6,848 rows rather than 7,820 -- a different
    file, with no error anywhere.

    The fixture carries that case (``C20E``), a mutation sitting exactly on
    the floor (``D30F``, which a ``>`` would wrongly drop), and a
    three-replicate mutation whose mean and median differ (``E40G``), since
    with two replicates ``.median()`` passes everything ``.mean()`` does.
    """
    frame = pd.DataFrame(
        {
            "dataset_name": ["rep-1", "rep-2"] * 2
            + ["rep-1"]
            + ["rep-1", "rep-2", "rep-3"],
            "scale_coeff_lasso_shift": [4.0e-05] * 8,
            "mutation": [
                "A10D",
                "A10D",
                "C20E",
                "C20E",
                "D30F",
                "E40G",
                "E40G",
                "E40G",
            ],
            "score": [1.0, 3.0, 10.0, 99.0, 7.0, 0.0, 1.0, 20.0],
            "seen": [5, 5, 5, 1, 3, 5, 5, 5],
        }
    )
    dataset = _dataset(
        effect_column="score",
        mutation_column="mutation",
        replicate_column="dataset_name",
        fixed=(("scale_coeff_lasso_shift", 4.0e-05),),
        min_times_seen=("seen", 3),
    )
    converted = normalise_published(frame, dataset)
    effects = dict(zip(converted[DMS_SITE_COL], converted[DMS_EFFECT_COL]))
    # Stated first, so a row-set mistake reads as one rather than as a KeyError.
    assert sorted(effects) == [10, 20, 30, 40]
    # A10D passes in both replicates, so it is their mean.
    assert effects[10] == pytest.approx(2.0)
    # C20E passes in rep-1 only, so it is rep-1's value, not the mean of both
    # (54.5) and not absent.
    assert effects[20] == pytest.approx(10.0)
    # D30F sits on the floor, which `>=` keeps and `>` would drop.
    assert effects[30] == pytest.approx(7.0)
    # E40G: mean 7.0, median 1.0.
    assert effects[40] == pytest.approx(7.0)


def test_an_unparsable_mutation_raises_rather_than_scoring_at_site_nan():
    with pytest.raises(ValueError, match="cannot parse mutation"):
        parse_aa_mutation("not-a-mutation")

    frame = pd.DataFrame({"mutation": ["A10D", "???"], "score": [1.0, 2.0]})
    dataset = _dataset(effect_column="score", mutation_column="mutation")
    with pytest.raises(ValueError, match="cannot parse mutation"):
        normalise_published(frame, dataset)


def test_a_column_named_in_the_config_and_absent_from_the_file_is_named():
    """Including ``replicate_column``, which nothing else would ever read.

    The averaging groups on ``mutation_column``, so a misspelled
    ``replicate_column`` would change no output and raise nothing. This is the
    only thing that reads it.
    """
    frame = pd.DataFrame({"mutation": ["A10D"], "score": [1.0]})
    dataset = _dataset(
        effect_column="score",
        mutation_column="mutation",
        replicate_column="typo_name",
    )
    with pytest.raises(ValueError, match="typo_name.*named in config.yaml"):
        normalise_published(frame, dataset)


def test_a_pinned_value_absent_from_the_file_says_what_is_present():
    frame = pd.DataFrame(
        {
            "scale_coeff_lasso_shift": [4.0e-05, 8.0e-05],
            "mutation": ["A10D", "A10E"],
            "score": [1.0, 2.0],
        }
    )
    dataset = _dataset(
        effect_column="score",
        mutation_column="mutation",
        fixed=(("scale_coeff_lasso_shift", 1.0),),
    )
    with pytest.raises(ValueError, match=r"The file's values are \[4e-05, 8e-05\]"):
        normalise_published(frame, dataset)


def test_a_yaml_scalar_without_its_decimal_point_still_means_a_number():
    """`4e-05` loads as the string ``'4e-05'``; coercion makes both spellings one.

    `lib.dms_dataset` applies `float` at the read site, as it does for every
    other typed field, so the spelling cannot reach the comparison and leave
    an empty frame.
    """
    doctored = copy.deepcopy(lib.CONFIG)
    doctored["dms"]["homologs"]["BA.1"]["fixed"] = {"scale_coeff_lasso_shift": "4e-05"}
    original = lib.CONFIG
    try:
        lib.CONFIG = doctored
        assert lib.dms_dataset("BA.1").fixed == (("scale_coeff_lasso_shift", 4.0e-05),)
    finally:
        lib.CONFIG = original


def test_the_conversion_drops_what_the_accounting_never_sees():
    """Missing effects and no-op mutations go before ``raw_rows`` is taken.

    They are the dataset's own preparation rather than this figure's filters,
    and `load_dms`'s balance check requires ``raw_rows`` to be the count the
    conversion emits -- so a conversion that left them in would push rows into
    ``from_multidms``, which warns and drops them, and the balance check would
    raise.
    """
    frame = pd.DataFrame(
        {
            DMS_SITE_COL: ["10", "11", "12"],
            DMS_WT_COL: ["A", "C", "D"],
            DMS_MUT_COL: ["G", "C", "G"],
            DMS_EFFECT_COL: [1.0, 2.0, None],
        }
    )
    converted = normalise_published(frame, _dataset())
    assert list(converted[DMS_SITE_COL]) == [10]
