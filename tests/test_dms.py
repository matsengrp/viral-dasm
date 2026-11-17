"""What a wrong DMS port would report, and what catches it.

Four failure modes here produce a full-looking figure. A mask step dropped or
reordered leaves an accounting that calls masking a construct disagreement; a
stratum read off mat-pcp's coverage instead of the founder's codons turns the
split into a statement about sampling depth; a log selection factor passed
where a linear one belongs makes a probability that is wrong by a factor of
nothing visible; and a second spelling of the codon mask drifts from the one
the scored table was built with. None of them raises on its own.

The checks over a hand-built founder run always. The ones that read this
repository's own mat-pcp run -- the mask sizes, the DMS files -- skip
when it is not readable, so the suite runs off-cluster.
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
torch = pytest.importorskip("torch")

import dms
import panel_charts
import panel_spec

#: The global codon mask and the per-clade censored codon sites this run
#: carries, measured with the functions that produce them rather than pinned
#: from the experiment this comparison was ported from.
#:
#: That experiment pins 61 / 61 / 86 clade-masked sites for 21K / 23A / 24E and
#: 17 global codons, read from `counts_all.csv`'s `exclude` and
#: `masked_in_usher` columns. `config.yaml` does not name that file, so what is
#: pinned here is what this pipeline can produce from the files it does read.
#: The global count agrees; the per-clade ones do not.
#:
#: **The difference is which mutation-level file is widened, not the
#: granularity.** `exclude` is per nucleotide mutation too -- `counts_all.csv`
#: carries 9 rows per codon site per clade, and on 21K/S 51 of the 61 flagged
#: sites are flagged on only some of their 9 rows -- so both masks collapse
#: mutation-level flags to codon sites by the same `idx // 3 + 1`. They widen
#: different files.
#:
#: The censor table is the better substrate here, and this is the reason:
#: `scripts/render_config.py` passes exactly `censored_mutations_file`,
#: `censored_clades_file` and `masked_nt_sites_file` into a `TrainingConfig`,
#: and nothing passes `counts_all.csv`. So these are the layers the models were
#: actually trained without, which is what a DMS comparison has to exclude.
EXPECTED_GLOBAL_MASK_SITES = 17
EXPECTED_CENSORED_SITES = {"21K": 20, "23A": 33, "24E": 49}


#: The mat-pcp files these tests read. Resolved OUTSIDE the probe's `try`:
#: `lib.mat_pcp_path` raises `KeyError` on a key this pipeline does not have,
#: which is a config bug and must not come back as a skip.
_MAT_PCP_INPUTS = (
    lib.mat_pcp_path("clade_founders_dir"),
    lib.mat_pcp_path("masked_nt_sites_file"),
    lib.mat_pcp_path("censored_mutations_file"),
    lib.mat_pcp_path("censored_clades_file"),
)


#: Set on a machine where these inputs are expected -- the cluster the figure
#: is built on -- so that a moved or renamed file fails loudly instead of
#: turning this file into green skips. `tests/test_panels.py` and
#: `tests/test_clade_haplotypes.py` read the same variable.
REQUIRE_DATA = os.environ.get("VIRAL_DASM_REQUIRE_DATA") == "1"


def _mat_pcp_readable() -> bool:
    try:
        return all(path.exists() for path in _MAT_PCP_INPUTS)
    except OSError:
        return False


needs_mat_pcp = pytest.mark.skipif(
    not _mat_pcp_readable(), reason="this run's mat-pcp outputs are not readable here"
)


#: Likewise resolved outside the probe, and non-empty: `all()` over an empty
#: roster is True, which would turn every per-homolog test below into a pass
#: over zero homologs.
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


@needs_mat_pcp
def test_the_mask_sizes_this_run_produces_have_not_moved():
    """A repointed mat-pcp run must fail here, not score cells unseen in it."""
    assert len(dms.globally_masked_sites()) == EXPECTED_GLOBAL_MASK_SITES
    measured = {
        clade: len(dms.censored_sites(clade)) for clade in EXPECTED_CENSORED_SITES
    }
    assert measured == EXPECTED_CENSORED_SITES


@needs_mat_pcp
def test_the_codon_mask_agrees_with_the_one_the_scored_table_was_built_with():
    """`scripts/score_run.py` spells this widening too; the two must agree.

    `dms.codon_sites_of` is deliberately a second definition -- importing
    `score_run` would pull the model stack and the GPU slot into the panel path
    -- so this is what keeps the duplication honest. An off-by-one in one of
    them and not the other shifts a whole column of flags by a site, with
    nothing raising.
    """
    import score_run

    assert dms.codon_sites_of([0, 1, 2, 3, 5, 3818]) == score_run.codon_sites_of(
        [0, 1, 2, 3, 5, 3818]
    )
    assert dms.globally_masked_sites() == score_run.globally_masked_sites()


@needs_mat_pcp
def test_the_loaders_mask_is_exactly_what_from_multidms_needs():
    """Two layers here, and the figure's population is not built from them.

    `dms.unreadable_sites` exists so `from_multidms` does not raise on a
    wildtype disagreement, and those disagreements sit at masked sites. It is
    **not** what `fig:dms` excludes. Since #36 the figure's exclusion is
    mat-pcp's own per-mutation `exclude`, applied before any fitness effect is
    computed and so arriving as the nulls :func:`panels.base_frames` drops,
    together with :func:`panels.drop_unobservable_codons`. A third layer
    folded in here would quietly make the loader's mask the population, which
    is the mistake this pins.
    """
    for clade in EXPECTED_CENSORED_SITES:
        assert dms.unreadable_sites(clade) == (
            dms.globally_masked_sites() | dms.censored_sites(clade)
        )


@needs_mat_pcp
def test_the_strata_come_from_the_founders_codons():
    """Not from which mutations mat-pcp happened to emit a count for.

    A stratification read off the counts table passes every other check here
    and makes the single-nucleotide stratum a statement about sampling depth.
    The discriminating case is a mutation one nucleotide away whose count is
    absent: the indicator must still call it single-nucleotide, because the
    indicator never sees a count at all.
    """
    clade = "24E"
    indicator = dms.single_nt_indicator(clade)
    founder_nt = lib_founder_nt(clade)
    from dasm2.sequence.conversion import iter_codons

    codons = list(iter_codons(founder_nt))
    assert len(indicator.columns) == len(codons)
    assert "*" not in set(indicator.index)

    # Every position, against an independent source (Biopython). Position 0
    # alone would not discriminate: it is ATG, and methionine has one codon, so
    # a codon-aware indicator and an amino-acid-only one agree there.
    for position, codon in enumerate(codons):
        reachable = set(indicator.index[indicator[position]])
        assert reachable == _one_change_away(codon), f"position {position + 1}"

    # The discriminating case, stated explicitly: at a six-codon residue the
    # founder's actual codon reaches strictly fewer amino acids than the best
    # synonymous codon would. An indicator built from the best case over
    # synonymous realizations -- or from mat-pcp's coverage -- passes the loop
    # above only if it happens to agree everywhere, and it does not here.
    best_case_is_wider = False
    for position, codon in enumerate(codons):
        synonymous = _synonymous_codons(codon)
        if len(synonymous) < 2:
            continue
        actual = _one_change_away(codon)
        best_case = set().union(*(_one_change_away(alt) for alt in synonymous))
        if best_case > actual:
            assert set(indicator.index[indicator[position]]) == actual
            best_case_is_wider = True
            break
    assert best_case_is_wider, (
        "no site on this founder distinguishes the actual codon from the best "
        "synonymous one, so this test cannot see that difference"
    )


def _synonymous_codons(codon: str) -> list[str]:
    """Every codon encoding the same amino acid as ``codon``."""
    from Bio.Seq import Seq

    from dasm2.sequence.constants import CODONS

    residue = str(Seq(codon).translate())
    return [other for other in CODONS if str(Seq(other).translate()) == residue]


def _one_change_away(codon: str) -> set[str]:
    """Every amino acid one nucleotide substitution in ``codon`` encodes."""
    from Bio.Seq import Seq

    out: set[str] = set()
    for position in range(3):
        for base in "ACGT":
            if base == codon[position]:
                continue
            residue = str(
                Seq(codon[:position] + base + codon[position + 1 :]).translate()
            )
            if residue != "*":
                out.add(residue)
    return out


def lib_founder_nt(clade: str) -> str:
    from founders import founder_nt_sequence

    return founder_nt_sequence(clade, lib.GENE)


@needs_mat_pcp
@needs_dms
def test_masking_runs_before_the_loader_that_raises_on_a_mismatch():
    """Dropping the mask step must raise, which is why the order is fixed.

    ``from_multidms`` raises on a wildtype/reference disagreement, and the
    disagreements sit at masked sites, so the mask has to be applied first.
    This is the failure the order prevents -- not a reordering of steps 2 and
    3, which does not raise because step 3 clears the frame either way.
    """
    homolog = "KP.3.1.1"
    dataset = lib.dms_dataset(homolog)
    from founders import founder_aa_sequence

    founder = founder_aa_sequence(dataset.clade, lib.GENE)
    # Through the published-file conversion, as `load_dms` reads it: the file
    # carries the study's own column names, and the mask step is the only one
    # removed here.
    from dms_published import normalise_published

    raw = normalise_published(pd.read_csv(dataset.path), dataset)
    unmasked = raw[~raw[dms.DMS_MUT_COL].isin(["*", "-"])]

    import tempfile

    from dasm2.data.functional_formats import from_multidms

    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "unmasked.csv"
        unmasked.to_csv(path, index=False)
        with pytest.raises(ValueError, match="WT amino acid mismatch"):
            from_multidms(path, wt_sequence=founder)


@needs_mat_pcp
@needs_dms
def test_the_accounting_separates_masking_from_a_construct_disagreement():
    """Reversing steps 2 and 3 does not raise; it moves the count.

    So the accounting is the thing that catches it. KP.3.1.1 carries exactly
    one disagreeing *site* outside every mask -- 1253, the construct's stop
    codon -- and two measured rows at it, which is the one place the two
    fields of the accounting are not interchangeable: ``mismatch_sites`` is
    sites and ``dropped_founder_mismatch`` is rows. Masked disagreements
    belong to ``dropped_masked_site``, and reversing the two steps would move
    them here.
    """
    _, accounting = dms.load_dms("KP.3.1.1")
    assert accounting.mismatch_sites == (1253,)
    # Measured on the population the study's own thresholds leave, which is
    # what `scripts/dms_published.py` applies: 8,342 converted rows, against
    # the 11,841 of the unfiltered table this pipeline read before. One of the
    # two rows at site 1253 survives that filter -- the other was seen once --
    # so the mismatch is one row at one site rather than two.
    assert accounting.raw_rows == 8342
    assert accounting.dropped_masked_site == 865
    assert accounting.dropped_founder_mismatch == 1


@needs_mat_pcp
@needs_dms
def test_every_measured_cell_lands_in_exactly_one_stratum():
    """A melted frame that lost or duplicated rows reports its own padding."""
    for homolog in lib.dms_homologs():
        matrix, accounting = dms.load_dms(homolog)
        long = dms.melt_dms(matrix, lib.dms_dataset(homolog).clade)
        assert len(long) == accounting.retained_substitutions
        assert set(long["stratum"]) == {dms.SINGLE_NT, dms.MULTI_NT}
        assert long["stratum"].notna().all()
        # Both strata non-empty: an indicator that came back all-False would
        # satisfy the subset check above with everything in one stratum.
        assert long["stratum"].eq(dms.SINGLE_NT).sum() > 0
        assert long["stratum"].eq(dms.MULTI_NT).sum() > 0


@needs_dms
def test_a_changed_dms_file_fails_rather_than_moving_every_number():
    """The md5 is a gate: these files are replaced in place when repointed."""
    import hashlib

    assert lib.dms_homologs(), "no homologs configured, so this loop proves nothing"
    for homolog in lib.dms_homologs():
        dataset = lib.dms_dataset(homolog)
        # Computed here rather than read back from `verify_dms_md5`, which
        # returns the digest it just compared: a gate that returned the pinned
        # value unconditionally would satisfy that comparison.
        measured = hashlib.md5(dataset.path.read_bytes()).hexdigest()
        assert measured == dataset.md5
        assert dms.verify_dms_md5(homolog) == measured


def test_a_log_selection_factor_is_refused_where_a_linear_one_belongs():
    """viral-dasm's one divergence from the vendored source; see PROVENANCE.

    Upstream the only check is on shape, and its docstring says a log factor
    "would be silently wrong rather than an error". Panel D reads
    ``log_selection_factor`` out of a table whose every other consumer wants it
    logged, so this is the mistake the arm is actually exposed to.

    The guard has to both fire and pass, so linear factors are asserted through
    the same call.
    """
    from dasm2.analysis.whichmut_probs import FounderRates, aa_mutation_probabilities
    from dasm2.sequence.constants import AA_STR_SORTED

    sites, n_aa = 4, len(AA_STR_SORTED)
    # `codon_rates` is [L, 64] -- the rate to each *child codon*, which
    # `neutral_rate_to_aa` gathers through `PARENT_TO_CHILDREN_LOOKUP` before
    # summing the synonymous children. A [L, 9] fixture indexes out of bounds.
    rates = FounderRates(
        codon_rates=torch.full((sites, 64), 0.1),
        codon_parent_idxs=torch.zeros(sites, dtype=torch.long),
        mask=torch.ones(sites, dtype=torch.bool),
    )
    linear = torch.full((sites, n_aa), 0.5)
    # A zero factor is a legitimate linear input -- a mutation the model calls
    # lethal -- so the guard must be `< 0` and not `<= 0`. Without this cell a
    # guard strengthened to `<= 0` would pass every test here.
    linear[0, 0] = 0.0
    probabilities, _ = aa_mutation_probabilities(linear, rates)
    assert torch.isfinite(probabilities).all()
    # The probabilities are normalised over the whole sequence, so this is the
    # cheap check that the passing leg reached the real computation rather
    # than a degenerate path.
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0), atol=1e-6)

    with pytest.raises(ValueError, match="negative"):
        aa_mutation_probabilities(torch.log(linear.clamp(min=1e-6)), rates)


def test_the_scatter_homolog_is_one_this_pipeline_reads():
    """`panel_spec` stays stdlib-and-`lib` only, so it cannot check this itself."""
    assert panel_spec.SCATTER_HOMOLOG in lib.dms_homologs()


def test_every_method_the_frames_carry_has_an_axis_title():
    """A method with no axis title raises at draw time, after the whole pass.

    `panel_charts` spells the method names as literals because it must not
    import `dms`: a chart module that reached for the frame layer would pull
    the DMS loader into the declarations of every panel rule. This is what
    keeps the two spellings in step.
    """
    assert set(panel_charts.DMS_AXIS_TITLES) == {
        label for _column, label in dms.METHODS
    }
    for title in panel_charts.DMS_AXIS_TITLES.values():
        assert len(title) == 2, "two lines, so all three sit on one baseline"


def test_every_threshold_the_figure_draws_has_a_row_colour():
    """A row with no colour is a `KeyError` in the middle of the draw."""
    assert set(panel_charts.DMS_ROW_COLORS) == set(dms.MIN_COUNTS)


def test_the_count_label_names_the_unfiltered_row_as_unfiltered():
    """At 0 the floor admits every mutation carrying an estimate at all.

    Printing it as "≥ 0 counts" sends a reader looking for a cut that is not
    there, which is the source notebook's own reason for the special case.
    """
    assert panel_charts.dms_count_label(0) == "all mutations"
    assert panel_charts.dms_count_label(10) == "≥ 10 counts"


SCORED_TABLE = panel_spec.SCORED_DIR / "selection_factors.parquet"

needs_scored_table = pytest.mark.skipif(
    not SCORED_TABLE.is_file(), reason=f"no scored table at {SCORED_TABLE}"
)


if REQUIRE_DATA:
    # On the host that builds the figure, a skip is the failure: every test
    # below that checks what a panel draws -- the count floor, the clip, the
    # join -- is gated on these three, so a moved file would leave the suite
    # green with nothing of the figure tested. Raised at collection, so the
    # reason is the missing path rather than a pile of skip reasons.
    _absent = [
        name
        for name, present in (
            ("mat-pcp outputs", _mat_pcp_readable()),
            ("the DMS CSVs", _dms_readable()),
            (str(SCORED_TABLE), SCORED_TABLE.is_file()),
        )
        if not present
    ]
    if _absent:
        raise RuntimeError(
            "VIRAL_DASM_REQUIRE_DATA=1 but these are not readable: "
            + ", ".join(_absent)
        )


@pytest.fixture(scope="module")
def figure_frames():
    import panels

    table = panels.read_scored_table(
        SCORED_TABLE,
        models=[panel_spec.FIGURE_MODEL],
        clades=[lib.dms_dataset(homolog).clade for homolog in lib.dms_homologs()],
        columns=dms.TABLE_COLUMNS,
    )
    return dms.dms_frames(table, lib.dms_homologs())


@needs_mat_pcp
@needs_dms
@needs_scored_table
def test_the_drawn_frame_holds_estimated_mutations_at_their_reported_values(
    figure_frames,
):
    """What a panel draws is what the model reported, unbounded.

    Two ways this goes wrong without raising. A left join leaves rows the tree
    never estimated, which a correlation then averages in as though they were
    measurements of zero -- the null check below. And a clip reintroduced
    anywhere on this path, whether through `dms_spec` or a stray `.clip` in
    `dms_frame`, would move a point onto a bound and leave the correlation
    unable to show it: on this run exactly one predictor value sits outside
    the source's old [-8, 6], so a bounds check would pass while the figure
    silently stopped reporting what the model said. Equality against the
    ``*_raw`` columns `clip_axes` writes either way is what catches that.
    """
    assert dms.CLIP_LOWER is None, "this figure plots unclipped values"
    for (homolog, _min_count), frame in figure_frames.items():
        assert len(frame), homolog
        for column, _label in dms.METHODS:
            assert frame[column].notna().all(), f"{homolog}: null {column}"
            pd.testing.assert_series_equal(
                frame[column],
                frame[f"{column}_raw"],
                check_names=False,
                obj=f"{homolog}: {column} against its pre-clip original",
            )
        # The assay is never bounded either: it has a floor of its own, and
        # touching it would be this figure measuring its own axis.
        assert "dms_cell_entry_raw" not in frame


@needs_mat_pcp
@needs_dms
@needs_scored_table
def test_the_count_floor_keeps_a_subset_and_nothing_else(figure_frames):
    """The bottom row is the top row's well-observed part, mutation for mutation.

    A floor applied to the wrong column -- or with `and` where mat-pcp's own
    convention is `or` -- still returns fewer rows, which is all a count check
    would notice. So the kept and the dropped sets are both checked against
    the rule itself, on the two count columns mat-pcp fills: every survivor
    clears the floor on at least one of them, and every mutation dropped from
    the top row fails it on both. A floor that was never applied, read off the
    wrong column, or applied with the wrong combinator fails one of these; so
    does swapping the two rows, since the subset check runs in one direction.
    """
    floor = max(dms.MIN_COUNTS)
    assert floor > 0, "the strict row must actually restrict something"
    counts = ["actual_count", "predicted_count"]
    for homolog in lib.dms_homologs():
        loose_frame = figure_frames[(homolog, 0)]
        strict_frame = figure_frames[(homolog, floor)]
        loose = set(loose_frame["aa_mutation"])
        strict = set(strict_frame["aa_mutation"])
        assert strict <= loose, homolog
        assert strict < loose, f"{homolog}: the floor kept everything"

        # `or`, as `COUNT_FILTER_MODE` has it: the two counts fail in opposite
        # directions, so a mutation survives on real observations *or* on a
        # meaningful denominator.
        kept = strict_frame[counts].max(axis=1)
        assert (kept >= floor).all(), f"{homolog}: a row below the floor survived"

        dropped = loose_frame[~loose_frame["aa_mutation"].isin(strict)]
        assert len(dropped), homolog
        assert (dropped[counts].max(axis=1) < floor).all(), (
            f"{homolog}: a row clearing the floor was dropped"
        )
        assert strict, homolog


def _synthetic_frame(n: int = 40, *, slope: float = 1.0) -> pd.DataFrame:
    """A frame shaped like `dms_frame`'s output, with a correlation we choose.

    Ungated on purpose: everything below it is arithmetic and chart structure,
    which no assay, founder or scored table is needed to check.
    """
    x = pd.Series(range(n), dtype=float)
    return pd.DataFrame(
        {
            "log_selection_factor": x,
            "delta_fitness_ratio": -x,
            dms.POOLED_METRIC: x,
            dms.TARGET: slope * x,
            "mutation": [f"A{i + 1}C" for i in range(n)],
            "site": list(range(1, n + 1)),
        }
    )


def test_a_perfect_line_comes_back_as_a_coefficient_of_one():
    """`dms_correlations` reports the correlation of the column it names.

    Each predictor is checked against a target built to a known sign: a
    transposed column or a method read off the wrong one would come back with
    the wrong sign rather than merely a different number, and `n` is the
    length the pair had after the nulls were dropped, which is what the bar's
    header prints.
    """
    frame = _synthetic_frame()
    frame.loc[0, "log_selection_factor"] = float("nan")

    table = dms.dms_correlations({("H", 0): frame})

    assert set(table["method"]) == {label for _column, label in dms.METHODS}
    by_column = table.set_index("column")
    assert by_column.loc["log_selection_factor", "pearson"] == pytest.approx(1.0)
    assert by_column.loc[dms.POOLED_METRIC, "pearson"] == pytest.approx(1.0)
    # The one predictor built to run the other way: a sign lost here is a bar
    # drawn in the wrong direction.
    assert by_column.loc["delta_fitness_ratio", "pearson"] == pytest.approx(-1.0)
    # The null row is dropped from the column that carries it, and only there.
    assert int(by_column.loc["log_selection_factor", "n"]) == len(frame) - 1
    assert int(by_column.loc[dms.POOLED_METRIC, "n"]) == len(frame)


def test_a_single_point_has_no_correlation_rather_than_an_exception():
    """One row is the degenerate rung: `nan`, not a raise and not a zero.

    A bar of 0 would read as "this method does not predict", which is a claim;
    a `nan` draws nothing, which is the truth about a subset of one.
    """
    table = dms.dms_correlations({("H", 10): _synthetic_frame(n=1)})
    assert table["pearson"].isna().all()
    assert (table["n"] == 1).all()


def test_the_two_rows_are_drawn_on_one_pair_of_axes():
    """Panels A and C share domains, so a point sits in the same place in both.

    The rows are separate images composed side by side, so nothing in the
    layout can align them: if the scatter row stopped forwarding the domains
    it is handed -- or recomputed them per row, which is what Altair does
    unasked -- the two rows would silently acquire different axes and a
    reader would compare positions that are not comparable.
    """
    frame = _synthetic_frame()
    x_domain, y_domain = [-9.0, 7.0], [-8.0, 1.0]
    rows = {
        min_count: panel_charts.dms_scatter_row(
            frame,
            min_count=min_count,
            clade="24E",
            methods=dms.METHODS,
            x_domain=x_domain,
            y_domain=y_domain,
            coefficients={column: 0.5 for column, _label in dms.METHODS},
        ).to_dict()
        for min_count in dms.MIN_COUNTS
    }
    for min_count, spec in rows.items():
        scatters = [
            layer
            for panel in spec["hconcat"]
            for layer in panel["layer"]
            if "circle" in str(layer.get("mark", ""))
        ]
        assert len(scatters) == len(dms.METHODS), min_count
        for scatter in scatters:
            assert scatter["encoding"]["x"]["scale"]["domain"] == x_domain
            assert scatter["encoding"]["y"]["scale"]["domain"] == y_domain


def test_the_corner_coefficient_is_the_one_the_bars_report():
    """The `r` in a scatter is read from the correlations frame, not recomputed.

    The frame here correlates at 1.0, so a panel that recomputed its own
    number would print `r = 1.00` and disagree with the bar beside it and the
    CSV beneath it, which both come from `dms_correlations`.
    """
    frame = _synthetic_frame()
    spec = panel_charts.dms_scatter_row(
        frame,
        min_count=0,
        clade="24E",
        methods=dms.METHODS,
        x_domain=[0.0, 40.0],
        y_domain=[0.0, 40.0],
        coefficients={column: 0.42 for column, _label in dms.METHODS},
    ).to_dict()
    # Altair hoists each layer's frame into a named dataset, so the text is
    # reached through the name the layer carries rather than inline.
    datasets = spec["datasets"]
    labels = [
        datum["_label"]
        for panel in spec["hconcat"]
        for layer in panel["layer"]
        for datum in datasets.get(layer["data"]["name"], [])
        if "_label" in datum
    ]
    assert labels == ["r = 0.42"] * len(dms.METHODS)


def test_a_padded_domain_covers_its_data_and_survives_a_constant_column():
    """`dms_padded_domain` is what puts a point inside the plot rather than on it."""
    domain = panel_charts.dms_padded_domain(pd.Series([-2.0, 5.0]))
    assert domain[0] < -2.0 and domain[1] > 5.0
    flat = panel_charts.dms_padded_domain(pd.Series([3.0, 3.0]))
    assert flat[0] < flat[1], "a constant column must still give a drawable axis"
    grouped = panel_charts.dms_group_domain(
        _synthetic_frame(), [column for column, _label in dms.METHODS]
    )
    assert grouped[0] < -39.0 and grouped[1] > 39.0


def test_the_figure_lays_out_exactly_the_rung_it_declares():
    """`fig:dms` draws one threshold, and the letters follow from which one.

    Two constants have to agree or the figure quietly mixes rungs: the rung
    it draws, and the letters it lays out. Spelling the letters out here from
    `DMS_FIGURE_MIN_COUNT` rather than from the table means that moving the
    figure to the thresholded rung moves the layout with it, instead of
    leaving ``("A", "B")`` pointing at the other row's panels.
    """
    laid_out = [letter for row in panel_spec.DMS_FIGURE_ROWS for letter in row]
    stems = [panel_spec.DMS_PANEL_LETTERS[letter] for letter in laid_out]
    assert sorted(stems) == sorted(
        stem
        for stem in panel_spec.DMS_PANELS
        if stem.endswith(f"min{panel_spec.DMS_FIGURE_MIN_COUNT}")
    )
    # And the pair that is not laid out is still written, so its CSVs and
    # PNGs stay available as the check they are.
    assert set(stems) < set(panel_spec.DMS_PANELS)


def _row_title(spec: dict) -> dict | None:
    """A concatenated chart's own title, as Altair serialises it."""
    title = spec.get("title")
    return title if isinstance(title, dict) else None


def test_the_laid_out_scatter_row_is_headed_by_its_homolog_in_black():
    """Panel A names the assay, not the count floor it shares with nothing.

    The default heading is the floor, which exists to tell two stacked rows
    apart. `fig:dms` draws one row, so a heading that reads "all mutations
    with estimated fitness effects" answers a question the figure no longer
    poses -- and a coloured one reads as an encoding when there is no second
    hue to contrast with.
    """
    frame = _synthetic_frame()
    heading = panel_charts.dms_homolog_heading("KP.3.1.1", "24E", len(frame))
    assert heading == f"KP.3.1.1 (clade 24E), n = {len(frame):,}"
    spec = panel_charts.dms_scatter_row(
        frame,
        min_count=panel_spec.DMS_FIGURE_MIN_COUNT,
        clade="24E",
        methods=dms.METHODS,
        x_domain=[0.0, 40.0],
        y_domain=[0.0, 40.0],
        coefficients={column: 0.42 for column, _label in dms.METHODS},
        heading=heading,
        heading_color=panel_charts.INK,
    ).to_dict()
    title = _row_title(spec)
    assert title is not None and title["text"] == heading
    assert title["color"] == panel_charts.INK

    # Unasked, it still heads itself with the floor, in the row's colour:
    # that is what the panel left off the page needs.
    default = _row_title(
        panel_charts.dms_scatter_row(
            frame,
            min_count=10,
            clade="24E",
            methods=dms.METHODS,
            x_domain=[0.0, 40.0],
            y_domain=[0.0, 40.0],
            coefficients={column: 0.42 for column, _label in dms.METHODS},
        ).to_dict()
    )
    assert default is not None
    assert "10 actual or expected counts" in default["text"]
    assert default["color"] == panel_charts.DMS_ROW_COLORS[10]


def test_the_laid_out_bar_row_carries_no_heading_of_its_own():
    """Panel B's groups already say which homolog and how many mutations.

    An empty heading has to mean "draw none" rather than "draw the default",
    which is why the parameter distinguishes ``""`` from ``None``: the rung
    that is written but not laid out keeps its own heading, and the two go
    through the same call.
    """
    correlations = pd.DataFrame(
        {
            "homolog": ["KP.3.1.1"] * len(dms.METHODS),
            "min_count": [panel_spec.DMS_FIGURE_MIN_COUNT] * len(dms.METHODS),
            "method": [label for _column, label in dms.METHODS],
            "pearson": [0.6, 0.4, 0.5],
            "n": [100, 100, 100],
        }
    )
    kwargs = dict(
        homologs=["KP.3.1.1"],
        clades={"KP.3.1.1": "24E"},
        methods=dms.METHODS,
        y_domain=[0.0, 1.0],
    )
    bare = panel_charts.dms_bar_row(
        correlations,
        min_count=panel_spec.DMS_FIGURE_MIN_COUNT,
        heading="",
        **kwargs,
    ).to_dict()
    assert _row_title(bare) is None

    default = panel_charts.dms_bar_row(
        correlations,
        min_count=panel_spec.DMS_FIGURE_MIN_COUNT,
        **kwargs,
    ).to_dict()
    title = _row_title(default)
    assert title is not None and "all mutations" in title["text"]
