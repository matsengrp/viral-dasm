"""What a wrong DMS port would report, and what catches it.

Four failure modes here produce a full-looking figure. A mask step dropped or
reordered leaves an accounting that calls masking a construct disagreement; a
stratum read off mat-pcp's coverage instead of the founder's codons turns the
split into a statement about sampling depth; a log selection factor passed
where a linear one belongs makes a probability that is wrong by a factor of
nothing visible; and a second spelling of the codon mask drifts from the one
the scored table was built with. None of them raises on its own.

The checks over a hand-built founder run always. The ones that read this
repository's own mat-pcp run -- the mask sizes, the three DMS files -- skip
when it is not readable, so the suite runs off-cluster.
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
#: `masked_in_usher` columns. `config.yaml` does not name that file -- it is
#: 563 MB and would be a second definition of "masked" beside the one the
#: scored table carries -- so what is pinned here is what this pipeline can
#: produce from the files it does read. The global count agrees; the per-clade
#: ones do not, because a censor table is per mutation where `exclude` is per
#: site.
EXPECTED_GLOBAL_MASK_SITES = 17
EXPECTED_CENSORED_SITES = {"21K": 20, "23A": 33, "24E": 49}


def _mat_pcp_readable() -> bool:
    try:
        return lib.mat_pcp_path("clade_founders_dir").is_dir()
    except (KeyError, OSError):
        return False


needs_mat_pcp = pytest.mark.skipif(
    not _mat_pcp_readable(), reason="this run's mat-pcp outputs are not readable here"
)


def _dms_readable() -> bool:
    try:
        return all(
            lib.dms_dataset(homolog).path.is_file() for homolog in lib.dms_homologs()
        )
    except (KeyError, OSError):
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
    score_run = pytest.importorskip("score_run")
    assert dms.codon_sites_of([0, 1, 2, 3, 5, 3818]) == score_run.codon_sites_of(
        [0, 1, 2, 3, 5, 3818]
    )
    assert dms.globally_masked_sites() == score_run.globally_masked_sites()


@needs_mat_pcp
def test_the_figures_exclusion_is_wider_than_the_loaders_mask():
    """The reversion layer is additional, and belongs to the panels, not here.

    `dms.unreadable_sites` is two layers on purpose: it exists so
    `from_multidms` does not raise on a wildtype disagreement, and the
    disagreements sit at masked sites. The figure's own exclusion is
    `panels.excluded_sites`, which adds the per-clade reversion rule. If the
    reversion sites were a subset of the two layers here, a panel built on this
    set alone would look right, so the test is that they are not.
    """
    panels = pytest.importorskip("panels")
    audit = panel_spec.AUDIT_JSON
    if not audit.is_file():
        pytest.skip(f"no censor audit at {audit}")
    reverting = panels.reverting_sites_by_clade(audit)
    for clade in EXPECTED_CENSORED_SITES:
        extra = panels.reverting_sites(reverting, clade) - dms.unreadable_sites(clade)
        assert extra, (
            f"{clade}: every reverting site is already in the loader's mask, so "
            f"a panel built on that mask alone would be indistinguishable from "
            f"one built on all three layers."
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

    # Position 0's codon is ATG (Met). One change reaches exactly the amino
    # acids its nine single-nucleotide neighbours encode, which is a property
    # of the codon table and of nothing else.
    reachable = set(indicator.index[indicator[0]])
    assert reachable == _one_change_away(codons[0])


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
    raw = pd.read_csv(dataset.path)
    unmasked = raw[~raw[dms.DMS_MUT_COL].isin(["*", "-"])]

    import tempfile

    from dasm2.data.functional_formats import from_multidms

    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "unmasked.csv"
        unmasked.to_csv(path, index=False)
        with pytest.raises(ValueError):
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
    assert accounting.dropped_masked_site > 0
    assert accounting.dropped_founder_mismatch == 2


@needs_mat_pcp
@needs_dms
def test_every_measured_cell_lands_in_exactly_one_stratum():
    """A melted frame that lost or duplicated rows reports its own padding."""
    for homolog in lib.dms_homologs():
        matrix, accounting = dms.load_dms(homolog)
        long = dms.melt_dms(matrix, lib.dms_dataset(homolog).clade)
        assert len(long) == accounting.retained_substitutions
        assert set(long["stratum"]) <= {dms.SINGLE_NT, dms.MULTI_NT}
        assert long["stratum"].notna().all()


@needs_dms
def test_a_changed_dms_file_fails_rather_than_moving_every_number():
    """The md5 is a gate: these files are replaced in place when repointed."""
    for homolog in lib.dms_homologs():
        assert dms.verify_dms_md5(homolog) == lib.dms_dataset(homolog).md5


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
    probabilities, _ = aa_mutation_probabilities(linear, rates)
    assert torch.isfinite(probabilities).all()

    with pytest.raises(ValueError, match="negative"):
        aa_mutation_probabilities(torch.log(linear), rates)


def test_the_scatter_homolog_is_one_this_pipeline_reads():
    """`panel_spec` stays stdlib-and-`lib` only, so it cannot check this itself."""
    assert panel_spec.SCATTER_HOMOLOG in lib.dms_homologs()


def test_every_stratum_has_both_label_forms():
    """A missing short form overlaps its neighbour in panel B's facet header.

    Vega-Lite neither wraps nor elides a header label, so two long labels over
    150 px columns simply overlap and the row becomes unreadable with nothing
    raising. :func:`panel_charts.stratum_label` falls back to the raw key, which
    is why the coverage is asserted rather than left to the fallback.
    """
    for stratum in dms.STRATA:
        assert stratum in panel_charts.STRATUM_LABELS
        long_form, short_form = panel_charts.STRATUM_LABELS[stratum]
        assert long_form and short_form
        assert len(short_form) <= len(long_form)
