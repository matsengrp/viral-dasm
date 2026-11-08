"""Gene-aware SARS-CoV-2 TabularRates neutral model construction.

Builds one gene's :class:`~dasm2.neutral.tabular_rates.TabularRatesNeutralModel`
archive from a mat-pcp run's GLM outputs. Nothing in it is specific to a
pipeline version or to a GLM stratification, so an era is only ever a different
``master_table`` argument. The run being consumed is named by the caller --
this module holds no paths.

mat-pcp v9 emits PCPs for six genes (``pcp_genes: [S, ORF1a, ORF1b, M, N, E]``),
and two of its outputs stop being usable as-is the moment the gene is not one
of the four sitting in spike's regime.

**The shipped rates table is after-boundary only.** mat-pcp's
``get_rates_table.py`` exports only GLM rows with
``nt_site_before_boundary == False``. The light-switch boundaries sit at
``CT: 13467`` and ``AT/CG/GC: 21562``, so S (21563-25381), E (26245-26469), M
(26523-27188) and N (28274-29530) are downstream of every one of them and the
shipped table is right for all four. ORF1a (266-13480) is upstream of the
AT/CG/GC boundary entirely and upstream of C->T bar a 13 nt tail. ORF1b
(13442-21552) is the awkward one: upstream of AT/CG/GC but *downstream* of C->T
for 99.7% of its length, so it needs opposite treatment from ORF1a in the
mutation type that matters most. :func:`before_boundary_types` derives all of
this from the gene's coordinates and the GLM's own boundaries rather than
branching on gene name -- the v8 version of this module raised for anything but
S and ORF1a, and ORF1b is the counterexample that retired that design.

Median ratio of the correct ORF1a rate to the shipped one, per mutation type::

    AT 0.317    CG 0.402    GC 0.398    CT 1.543

Note the sign flip on C->T. The pooled median over all 128 differing rows is
0.399, but quoting that alone is misleading: C->T -- the dominant SARS-CoV-2
substitution -- is 1.54x *higher* in ORF1a, not lower. Since ``p ∝ λ·f``, an
understated λ is absorbed by an inflated ``f``, so using the shipped table for
ORF1a would inflate selection factors at C->T by ~1.5x while deflating them
~2.5x at A->T, C->G and G->C. Nothing errors; the distortion just runs in
opposite directions across mutation types, which no single scale factor would
reproduce. :func:`select_gene_rates` reselects from the unfiltered
``glm/master_table.csv`` instead.

**The reusable secondary-structure table is spike-only.** The committed
``TabularRates-SCoV2-v1.zip`` covers nt 21563-25381 exclusively.
:func:`build_secondary_struct` rebuilds it per gene from the genome-wide
``glm/annotated_counts.csv``, whose spike values are identical to the
published ones.

Callers pass already-resolved paths in :class:`NeutralBuildInputs`. In this
fork that is the only way in: the manifest-resolving helpers are not vendored,
because viral-dasm names every path in ``config.yaml``. See the comment where
they were.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dasm2.io import read_fasta_first_sequence
from dasm2.neutral.tabular_rates import TabularRatesNeutralModel

logger = logging.getLogger(__name__)

# 1-based inclusive nucleotide CDS blocks of each gene in the SARS-CoV-2 reference
# (Wuhan-Hu-1, 29,903 nt). Selecting on coordinates rather than the annotation's
# `gene` column is deliberate: that column holds `ORF1a;ORF1ab` (13,214 sites)
# and `ORF1a;ORF1ab;ORF1ab` (1 site) rather than a bare `ORF1a`, so an equality
# test silently returns zero rows and a substring test silently returns 21,287
# rows spanning nt 266-21552.
#
# Blocks rather than one span, because ORF1b is two blocks sharing nt 13468 --
# which is how the -1 ribosomal frameshift stays in frame. Its length is
# therefore the *sum* of its blocks (8112 nt = 2704 codons) and not their span
# (8111), and nt 13468 legitimately appears twice in its structure table. A
# single (start, end) cannot express that; an envelope-based length check
# rejects the correct coordinates.
#
# ORF1b is nsp12-nsp16, not the post-slippage product: nsp12 begins at ORF1ab
# codon 4393 but slippage is at 4402, so its first 9 residues are translated in
# ORF1a's frame. Defining it as the single post-slippage block (13468, 21552)
# gives 2695 codons and silently clips them. These coordinates mirror
# mat-pcp's `pcp_gene_cds_blocks` and are checked against the shipped founder by
# `check_against_founder`.
#
# Every entry stops before the stop codon, which is what mat-pcp's GTF does and
# what its founders reflect. E is where that is easiest to get wrong from an
# outside source: most annotations give it as 26245-26472 (228 nt, TAA included),
# and taking that yields a plausible 76-codon gene whose structure table is one
# codon longer than every founder in the run. `check_against_founder` catches it,
# but the coordinate is right here to begin with.
GENE_CDS_BLOCKS: dict[str, list[tuple[int, int]]] = {
    "S": [(21563, 25381)],
    "ORF1a": [(266, 13480)],
    "ORF1b": [(13442, 13468), (13468, 21552)],
    "M": [(26523, 27188)],
    "N": [(28274, 29530)],
    "E": [(26245, 26469)],
}

# Expected nucleotide length of each gene, summed over its blocks and asserted
# against the founder FASTA. S 1273 codons, ORF1a 4405, ORF1b 2704, M 222, N 419,
# E 75.
GENE_NT_LENGTH: dict[str, int] = {
    gene: sum(end - start + 1 for start, end in blocks)
    for gene, blocks in GENE_CDS_BLOCKS.items()
}


def gene_nt_sites(gene: str) -> list[int]:
    """The gene's nucleotide sites in translation order, one entry per position.

    Repeats are real: ORF1b reads nt 13468 twice, so this is 8112 long while the
    gene spans 8111 positions. Callers must index structure tables by position in
    this list, never by `site - start`.
    """
    return [
        site for start, end in GENE_CDS_BLOCKS[gene] for site in range(start, end + 1)
    ]


def gene_nt_span(gene: str) -> tuple[int, int]:
    """1-based inclusive envelope of the gene's blocks.

    The envelope, *not* the length: ORF1b spans 8111 positions but is 8112 nt.
    Use :data:`GENE_NT_LENGTH` for length and :func:`gene_nt_sites` for order.
    """
    sites = gene_nt_sites(gene)
    return min(sites), max(sites)


# Genes that are internal slices of a larger ORF and so carry no start codon of
# their own. ORF1b is nsp12-nsp16, beginning at ORF1ab codon 4393 -- mid
# polyprotein -- so its founder starts TCAGCTGAT (Ser-Ala-Asp, the `SADAQSFLN`
# N-terminus), not ATG. Requiring ATG would reject the
# correct coordinates, so the check is skipped for these and the frame is pinned
# instead by the block reconstruction in `check_against_founder`'s callers.
INTERNAL_SLICE_GENES = frozenset({"ORF1b"})

# The four mutation types the GLM fits with a light switch, i.e. the only ones
# whose rate depends on which side of a boundary the site falls. The other
# eight are boundary-invariant and appear only in the after-boundary rows.
LIGHT_SWITCH_MUT_TYPES = frozenset({"AT", "CG", "CT", "GC"})

# 12 mutation types x 32 (motif, unpaired) combinations. Both the shipped
# rates table and every per-gene selection must land on exactly this many rows;
# anything else means the boundary filter selected overlapping or missing rows.
EXPECTED_RATES_ROWS = 384

# Columns TabularRatesNeutralModel reads from a rates table, in archive order.
RATES_COLUMNS = ["mut_type", "motif", "unpaired", "rate"]

# Columns of the companion secondary-structure table. Note that `site` is
# carried for provenance only -- the model indexes on `sequential_site` and
# reads `unpaired`, and never looks at `site` at runtime.
SECONDARY_STRUCT_COLUMNS = ["site", "unpaired", "sequential_site"]


class GeneRatesError(ValueError):
    """Raised when a per-gene table fails a shape or content invariant.

    Distinct from a bare ValueError so callers can tell "this build is wrong"
    apart from an unrelated pandas failure. Every raise site here guards an
    invariant whose violation would otherwise produce a well-formed archive
    holding the wrong numbers.
    """


@dataclass(frozen=True)
class NeutralBuildInputs:
    """Resolved, manifest-tracked inputs for one gene's neutral model."""

    gene: str
    master_table: Path
    site_annotations: Path
    founder_fasta: Path


# Three functions are not vendored: `resolve_run_root`, `resolve_build_inputs`
# and `_region_checked_config`. All three existed to turn a data-central
# manifest nickname into paths, and viral-dasm resolves nothing -- every path
# this module needs is named in `config.yaml` and handed in as
# `NeutralBuildInputs`. Vendoring them would mean vendoring
# `data/viral_fitness_registry.py` and the `discovery`/`manifests`/
# `registry_schema` modules it imports, which #4 removed.
#
# `_region_checked_config` was a real guard, and it loses its input here: it
# refused to pair one region's founder with another's rates, by comparing the
# two manifests' `region` fields. There are no manifests to compare. What
# replaces it is weaker and structural rather than checked -- `config.yaml`
# names one `mat_pcp.gene` and one `mat_pcp.root`, so the founder and the rate
# tables come from the same run by construction. `validate_gene` below still
# rejects a gene with no coordinate blocks, which is the other half of what the
# old guard caught.

def validate_gene(gene: str) -> None:
    """Reject genes with no coordinate blocks rather than guessing them."""
    if gene not in GENE_CDS_BLOCKS:
        raise GeneRatesError(
            f"Unknown gene {gene!r}; expected one of {sorted(GENE_CDS_BLOCKS)}. "
            "Add its 1-based inclusive nt blocks to GENE_CDS_BLOCKS first -- "
            "there is no safe default, and wrong blocks yield a "
            "plausible-looking archive with misaligned structure."
        )


def light_switch_boundaries(master_table: pd.DataFrame) -> dict[str, int]:
    """Each light-switch mutation type's boundary position, from the GLM itself.

    Read from the table rather than hardcoded so that a run whose GLM moved a
    boundary re-derives its own selection instead of silently reusing this run's
    coordinates.
    """
    if "nt_site_boundary" not in master_table.columns:
        raise GeneRatesError(
            "master_table has no `nt_site_boundary` column, so the boundary "
            "positions cannot be derived. Refusing to guess them."
        )
    rows = master_table.loc[master_table["nt_site_boundary"].astype(int) > 0]
    return {
        str(mut_type): int(group["nt_site_boundary"].iloc[0])
        for mut_type, group in rows.groupby("mut_type", sort=True)
    }


def before_boundary_types(master_table: pd.DataFrame, gene: str) -> frozenset[str]:
    """Which light-switch mutation types `gene` takes *before*-boundary rates for.

    Replaces the per-gene ``if gene == "S" ... elif gene == "ORF1a"`` branch this
    module used through v8, which raised for anything else. That was the right
    call while the rule was "one uniform side per gene" -- but it is not, and
    ORF1b is the counterexample that forces the issue.

    The side is decided per mutation type, by counting the gene's own
    nucleotides on each side of that type's boundary:

    - **S, E, M, N** are downstream of every boundary (21563+, 26245+, 26523+,
      28274+ against a last boundary at 21562), so no type takes before-boundary
      rows and the shipped after-boundary ``rates_table.csv`` is already correct
      for them.
    - **ORF1a** (266-13480) is upstream of all four, so every light-switch type
      takes before-boundary rows -- reproducing the v8 branch exactly.
    - **ORF1b** (13442-21552) is upstream of AT/CG/GC but **downstream of C->T**
      for 8086 of its 8112 nt. It is the case the old code refused: it needs
      before-boundary AT/CG/GC and after-boundary C->T, in the same selection.

    Counting is over :func:`gene_nt_sites`, not the (min, max) envelope, because
    ORF1b's shared nt 13468 makes those differ by one.

    Where a boundary cuts the gene, the majority side wins and the minority is a
    documented approximation -- 13 of 13215 nt for ORF1a (0.098%), 26 of 8112 for
    ORF1b (0.32%). Both are logged. Serving one uniform rate per gene is what
    makes this an approximation at all; splitting the table per site would remove
    it, at a complexity cost neither gene justifies.
    """
    validate_gene(gene)
    sites = gene_nt_sites(gene)
    takes_before = set()
    for mut_type, boundary in light_switch_boundaries(master_table).items():
        if mut_type not in LIGHT_SWITCH_MUT_TYPES:
            continue
        n_before = sum(site <= boundary for site in sites)
        n_after = len(sites) - n_before
        if n_before > n_after:
            takes_before.add(mut_type)
        if n_before and n_after:
            minority = min(n_before, n_after)
            logger.warning(
                "%s: the %s boundary at %d falls inside the gene; serving the "
                "%s-boundary rate uniformly misspecifies %d of %d nt (%.3f%%).",
                gene,
                mut_type,
                boundary,
                "before" if n_before > n_after else "after",
                minority,
                len(sites),
                100 * minority / len(sites),
            )
    return frozenset(takes_before)


def select_gene_rates(master_table: pd.DataFrame, gene: str) -> pd.DataFrame:
    """Select the boundary-correct rate rows for `gene` from the GLM master table.

    Which rows a gene takes is decided per mutation type by
    :func:`before_boundary_types`, from the gene's own coordinates and the GLM's
    own boundaries. For the six v9 genes that works out to:

    - **S, E, M, N** -- downstream of every boundary, so they take the
      after-boundary rows wholesale. That is exactly what mat-pcp ships as
      ``rates_table.csv``, and validation 1 pins all four against it.
    - **ORF1a** -- upstream of every boundary bar a 13 nt tail, so before-boundary
      rows for the four light-switch types and after-boundary rows for the eight
      boundary-invariant ones (which have no before-boundary rows to take).
    - **ORF1b** -- upstream of AT/CG/GC but *downstream* of C->T for 8086 of its
      8112 nt, so it mixes the two within one selection.

    Known approximation, for the two genes a boundary cuts. One uniform rate is
    served per (gene, mutation type), so the minority side receives the majority's
    rate: 13 of 13215 nt for ORF1a (0.098%, past the C->T boundary at 13467) and
    26 of 8112 for ORF1b (0.32%, before it). The mask is what is wrong at those
    sites, not the boundary. Both are logged at build time. Splitting the table
    per site would remove the approximation at a complexity cost neither gene
    justifies.

    v8 had to quote each of those windows as two numbers, because mat-pcp
    disagreed with itself at the boundary site: the GLM partitioned training data
    with ``nt_site > boundary`` while ``annotate_counts.py`` labelled with
    ``site < boundary``, so 13467 trained as before-boundary and was served the
    after-boundary rate. mat-pcp fixed that before the run this paper uses, and
    the counts above are now unambiguous.

    Args:
        master_table: ``glm/master_table.csv`` as a DataFrame.
        gene: Gene identifier; must be a key of :data:`GENE_CDS_BLOCKS`.

    Returns:
        A 384-row frame with :data:`RATES_COLUMNS`, sorted for stable output.

    Raises:
        GeneRatesError: If the gene is unknown, required columns are missing,
            or the selection does not land on exactly 384 unique rows.
    """
    validate_gene(gene)
    required = {"mut_type", "motif", "unpaired", "rate", "nt_site_before_boundary"}
    if missing := required - set(master_table.columns):
        raise GeneRatesError(f"master_table is missing columns: {sorted(missing)}")

    before = master_table["nt_site_before_boundary"].astype(bool)
    takes_before = master_table["mut_type"].isin(
        before_boundary_types(master_table, gene)
    )
    keep = (takes_before & before) | (~takes_before & ~before)

    rates = (
        master_table.loc[keep, RATES_COLUMNS]
        .sort_values(["mut_type", "motif", "unpaired"], kind="stable")
        .reset_index(drop=True)
    )

    if len(rates) != EXPECTED_RATES_ROWS:
        raise GeneRatesError(
            f"{gene}: selected {len(rates)} rate rows, expected "
            f"{EXPECTED_RATES_ROWS}. The boundary filter is over- or "
            "under-selecting; do not write this archive."
        )
    duplicated = rates.duplicated(["mut_type", "motif", "unpaired"]).sum()
    if duplicated:
        raise GeneRatesError(
            f"{gene}: {duplicated} duplicate (mut_type, motif, unpaired) keys in "
            "the selected rates. A key present on both sides of a boundary "
            "would make the lookup order-dependent."
        )
    return rates


def build_secondary_struct(site_annotations: pd.DataFrame, gene: str) -> pd.DataFrame:
    """Build `gene`'s secondary-structure table from the genome-wide annotation.

    The annotation has one row per (clade, subset, nt_site, nt_mutation), so it
    is deduplicated to one row per site first, rejecting any site whose repeats
    disagree on ``unpaired``. ``unpaired`` is stored as 0/1 upstream and emitted
    as bool here to match the committed ``TabularRates-SCoV2-v1.zip``
    representation. (The companion rates table keeps the upstream 0/1, matching
    mat-pcp's own ``rates_table.csv``; the model coerces both through ``bool()``
    when it builds its lookup, so the two representations agree at runtime.)

    Args:
        site_annotations: ``glm/annotated_counts.csv`` as a DataFrame; needs at
            least ``nt_site`` and ``unpaired``.
        gene: Gene identifier; must be a key of :data:`GENE_CDS_BLOCKS`.

    Returns:
        A frame with :data:`SECONDARY_STRUCT_COLUMNS`, one row per nucleotide
        of the gene, ``sequential_site`` 0-based and contiguous.

    Raises:
        GeneRatesError: If the gene is unknown, columns are missing, a site's
            repeats disagree on ``unpaired``, the slice has the wrong length,
            or ``unpaired`` has nulls.
    """
    validate_gene(gene)
    required = {"nt_site", "unpaired"}
    if missing := required - set(site_annotations.columns):
        raise GeneRatesError(f"site_annotations is missing columns: {sorted(missing)}")

    sites = gene_nt_sites(gene)
    span_lo, span_hi = min(sites), max(sites)
    # Deduplicate on both columns rather than on `nt_site` alone. The annotation
    # carries one row per (clade, subset, nt_site, nt_mutation), so a site
    # legitimately repeats -- 4.58M rows over 29,903 sites in v8 -- but every
    # repeat must agree on `unpaired`. Deduplicating on `nt_site` alone would
    # silently keep whichever row happened to sort first if a future mat-pcp
    # made the structure call per clade, which is exactly the well-formed-
    # archive-with-wrong-numbers failure this module exists to prevent. In v8
    # no site disagrees, so this is a guard, not a fix.
    per_site = (
        site_annotations.loc[:, ["nt_site", "unpaired"]]
        .drop_duplicates(["nt_site", "unpaired"])
        .sort_values("nt_site", kind="stable")
    )
    sliced = per_site[
        (per_site["nt_site"] >= span_lo) & (per_site["nt_site"] <= span_hi)
    ]

    if conflicting := int(sliced["nt_site"].duplicated().sum()):
        raise GeneRatesError(
            f"{gene}: {conflicting} site(s) in nt {span_lo}-{span_hi} carry more "
            "than one distinct `unpaired` value, so the structure call is "
            "ambiguous. Decide which one applies -- do not let row order pick."
        )
    if sliced["unpaired"].isna().any():
        raise GeneRatesError(
            f"{gene}: null `unpaired` values in nt {span_lo}-{span_hi}."
        )

    # Index by site and reindex onto the gene's own site list rather than
    # slicing a contiguous range. This is what lets a multi-block gene work:
    # ORF1b reads nt 13468 twice, so its structure table legitimately carries
    # that site on two consecutive rows, and any approach that assumes
    # `sequential_site == site - start` puts every downstream residue one
    # position out of frame.
    unpaired_by_site = sliced.set_index("nt_site")["unpaired"]
    missing_sites = [
        site for site in dict.fromkeys(sites) if site not in unpaired_by_site.index
    ]
    if missing_sites:
        raise GeneRatesError(
            f"{gene}: annotation is missing {len(missing_sites)} of "
            f"{len(set(sites))} distinct sites (first: {missing_sites[:5]}). A gap here "
            "silently becomes unpaired=False in the model's vectorized lookup."
        )

    values = unpaired_by_site.reindex(sites)
    if len(values) != GENE_NT_LENGTH[gene]:
        raise GeneRatesError(
            f"{gene}: built {len(values)} structure rows, expected "
            f"{GENE_NT_LENGTH[gene]} from its CDS blocks."
        )

    return pd.DataFrame(
        {
            "site": sites,
            "unpaired": values.to_numpy().astype(bool),
            "sequential_site": range(len(sites)),
        }
    )[SECONDARY_STRUCT_COLUMNS]


def check_against_founder(
    secondary_struct: pd.DataFrame, founder_nt: str, gene: str
) -> None:
    """Assert the structure table lines up with the sequence it will be used on.

    ``TabularRatesNeutralModel`` raises when a sequence is *longer* than its
    structure coverage but silently truncates the table when it is shorter, so
    an over-long table produces wrong-but-plausible rates. This pins the two to
    equal length.
    """
    if len(founder_nt) != len(secondary_struct):
        raise GeneRatesError(
            f"{gene}: founder is {len(founder_nt)} nt but the structure table has "
            f"{len(secondary_struct)} rows. The model would silently use the "
            "first len(seq) rows."
        )
    if len(founder_nt) % 3 != 0:
        raise GeneRatesError(
            f"{gene}: founder length {len(founder_nt)} is not a multiple of 3."
        )
    if gene not in INTERNAL_SLICE_GENES and not founder_nt.startswith("ATG"):
        raise GeneRatesError(f"{gene}: founder does not start with ATG.")


def build_neutral_model(
    inputs: NeutralBuildInputs,
    output: Path,
    model_name: str,
) -> None:
    """Build and save one gene's TabularRates neutral model archive.

    Every invariant is checked before ``save_pretrained``: a neutral model with
    the wrong rates trains without error and yields wrong selection factors, so
    failing loudly here is strictly better than shipping the archive.
    """
    gene = inputs.gene
    validate_gene(gene)

    # master_table is 512 rows, so read it whole; site_annotations is 682 MB
    # with 29 columns, so take only the two that matter.
    rates = select_gene_rates(pd.read_csv(inputs.master_table), gene)
    secondary_struct = build_secondary_struct(
        pd.read_csv(inputs.site_annotations, usecols=["nt_site", "unpaired"]), gene
    )
    founder_nt = read_fasta_first_sequence(inputs.founder_fasta)
    check_against_founder(secondary_struct, founder_nt, gene)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        rates_csv = tmp_dir / "rates_table.csv"
        struct_csv = tmp_dir / "secondary_struct.csv"
        rates.to_csv(rates_csv, index=False)
        secondary_struct.to_csv(struct_csv, index=False)
        model = TabularRatesNeutralModel(
            rates_table_path=rates_csv,
            secondary_struct_path=struct_csv,
            model_name=model_name,
        )
        model.save_pretrained(output)
    logger.info(
        f"Wrote {output} (gene={gene}, {len(rates)} rate rows, "
        f"{len(secondary_struct)} structure sites)"
    )
