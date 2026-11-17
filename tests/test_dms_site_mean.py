"""`scripts/dms_site_mean.py`: the DASM collapsed to one value per site.

The statistic is a groupby, so what is worth testing is not the mean itself
but the three things around it: which cells it is taken over, that both strata
receive the same value at a site, and that a site with nothing to average over
raises instead of arriving as a NaN. The first three tests here do that on
hand-built frames, off the artifact root, which is why
`scripts/dms_site_mean.py` splits its pure part out at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import dms
import dms_censor
import dms_dasm
import dms_models
import dms_site_mean
import lib
import pandas as pd
import panel_spec
from lib import dms_dataset, dms_homologs
from test_dms_models import needs_dms

from dasm2.sequence.constants import AA_STR_SORTED

#: The table the data-gated test reads, spelled as `tests/test_dms_prob.py`
#: spells it. Never rebuilt here.
SCORED_TABLE = panel_spec.SCORED_DIR / "selection_factors.parquet"

#: `needs_dms` gates on the published DMS CSVs alone, and the last test here
#: also reads the scored table off the artifact root. Gated separately for
#: `tests/test_dms_prob.py`'s reason: without this the test errors rather
#: than skipping on a host that has the CSVs and not the mount.
needs_artifacts = pytest.mark.skipif(
    not SCORED_TABLE.is_file(), reason=f"not readable here: {SCORED_TABLE}"
)

#: A two-residue founder: site 1 is M, site 2 is A.
_TOY_FOUNDER = "MA"

#: Which mutations one base change reaches in the toy, as
#: ``dms.single_nt_indicator`` lays it out: residues down, 0-indexed positions
#: across. ``F`` at site 1 is deliberately out of reach, so the filter has
#: something to exclude that is neither synonymous nor censored.
_REACHABLE = pd.DataFrame(
    {
        0: [aa in {"C", "D", "E", "H", "M"} for aa in AA_STR_SORTED],
        1: [aa in {"G", "S"} for aa in AA_STR_SORTED],
    },
    index=list(AA_STR_SORTED),
)

#: One censored mutation, at a site that has others, so the exclusion shows up
#: in the mean rather than in the coverage check.
_CENSORED = frozenset({(1, "E")})


def _toy_grid() -> pd.DataFrame:
    """The model's grid over the toy founder, one row per cell drawn on."""
    rows = [
        (1, "M", "C", -1.0),  # trained
        (1, "M", "D", -3.0),  # trained
        (1, "M", "H", -8.0),  # trained, and skews the mean away from the median
        (1, "M", "M", 0.0),  # synonymous, and gauge-fixed to exactly zero
        (1, "M", "E", -50.0),  # censored
        (1, "M", "F", 10.0),  # two base changes away
        (2, "A", "G", 5.0),  # trained
        (2, "A", "S", 1.0),  # trained, so the second site averages too
    ]
    return pd.DataFrame(
        rows, columns=["aa_site", "parent_aa", "mutant_aa", dms_dasm.SCORE_COLUMN]
    )


def test_the_site_value_averages_the_trained_mutations_and_only_those():
    """Site 1 holds four non-trained cells that would each move the mean.

    The synonymous cell is the one that matters most: it is exactly 0.0 by the
    gauge, so including it would pull every site toward zero in proportion to
    its codon's synonymous neighbours, which is a fact about the genetic code
    rather than about the model.
    """
    means = dms_site_mean.site_means(_toy_grid(), _REACHABLE, _CENSORED)
    assert means.loc[1] == pytest.approx(-4.0)  # (-1 + -3 + -8) / 3
    assert means.loc[2] == pytest.approx(3.0)  # (5 + 1) / 2
    assert set(means.index) == {1, 2}
    # Site 1's three values have a median of -3, so this also pins the mean
    # as the aggregate, which the module docstring states as a decision.
    assert means.loc[1] != pytest.approx(-3.0)


def test_a_site_whose_mutations_are_all_untrained_gets_no_value():
    """It is absent from the series rather than NaN in it.

    Whether that is an error depends on something this function cannot see --
    whether the assay measured a cell there -- so the decision belongs to
    :func:`assign_site_means` and this one stays quiet.
    """
    grid = _toy_grid()
    grid = grid[grid["aa_site"] == 1]
    means = dms_site_mean.site_means(
        grid, _REACHABLE, frozenset({(1, "C"), (1, "D"), (1, "E"), (1, "H")})
    )
    assert 1 not in means.index


def _toy_cells() -> pd.DataFrame:
    """Measured cells at one site, one per stratum."""
    return pd.DataFrame(
        {
            "homolog": ["toy", "toy"],
            "aa_site": [1, 1],
            "parent_aa": ["M", "M"],
            "mutant_aa": ["C", "W"],
            "stratum": [dms.SINGLE_NT, dms.MULTI_NT],
        }
    )


def test_both_strata_take_the_same_value_at_a_site():
    """The arm's defining property: no residue-level information at all.

    A single-nucleotide cell and a multi-nucleotide one at the same site score
    identically, which is what makes this a null for the DASM's own column
    rather than a second reading of it. Nothing else in the suite would catch
    the loss of this.
    """
    scored = dms_site_mean.assign_site_means(
        _toy_cells(), dms_site_mean.site_means(_toy_grid(), _REACHABLE, _CENSORED)
    )
    assert list(scored["score"]) == [-4.0, -4.0]


def test_a_measured_cell_with_no_site_value_raises_and_names_the_site():
    """Raised rather than dropped, so the message names the cause.

    ``validate_scores`` would refuse the NaN and `dms_summary`'s coverage
    check would report a short ``n``; both would name the symptom.
    """
    means = dms_site_mean.site_means(_toy_grid(), _REACHABLE, _CENSORED)
    orphan = _toy_cells().assign(aa_site=[1, 7])
    with pytest.raises(ValueError, match="no value to take") as raised:
        dms_site_mean.assign_site_means(orphan, means)
    # The site and the homolog both, since a message that named neither would
    # send a reader to the wrong assay.
    assert "[7]" in str(raised.value)
    assert str(raised.value).startswith("toy:")


@needs_dms
@needs_artifacts
def test_the_arm_scores_every_cell_the_dasm_s_own_column_does():
    """The identical-population property the whole comparison rests on.

    Per homolog and per stratum, this arm's cell count is `dasm`'s
    ``n_mutations`` in the tracked table. If it were not, the two correlations
    in that table would be over different populations and reading one against
    the other -- which is what #60 exists to do -- would be meaningless.
    """
    table = pd.read_csv(panel_spec.panel_csv_path("dms", "predictor_correlations"))
    reference = table[table["predictor"] == "dasm"].set_index(["homolog", "stratum"])

    for homolog in dms_homologs():
        frame, manifest, _ = dms_site_mean.site_mean_scores(homolog, SCORED_TABLE)
        assert manifest.strata == dms.STRATA
        assert not frame["score"].isna().any()

        cells, _ = dms_models.homolog_cells(homolog)
        assert len(frame) == int(reference.loc[(homolog, dms.ALL), "n_mutations"])
        for stratum in (dms.SINGLE_NT, dms.MULTI_NT):
            measured = int((cells["stratum"] == stratum).sum())
            expected = int(reference.loc[(homolog, stratum), "n_mutations"])
            assert measured == expected, (
                f"{homolog}/{stratum}: the join population moved since the "
                f"table was written"
            )

        # One value as well as the population, recomputed here from the real
        # grid: the counts above would all pass against a column of constants.
        clade = dms_dataset(homolog).clade
        grid = dms_dasm.selection_factor_grid(clade, SCORED_TABLE)
        trained = dms_models.trained_single_nt_cells(
            grid,
            dms.single_nt_indicator(clade, lib.GENE),
            dms_censor.censored_mutations(clade),
        )
        site = int(frame["aa_site"].iloc[0])
        at_site = trained[trained["aa_site"] == site][dms_dasm.SCORE_COLUMN]
        assert frame["score"].iloc[0] == pytest.approx(float(at_site.mean()))
