"""Measure a mutation's effect separately in two genetic backgrounds inside one clade.

mat-pcp gives a mutation's fitness effect as one number per ``(clade,
mutation)``, averaged over every branch in the clade. A DASM gives its effect on
the clade's *founder sequence*. These agree only if the effect does not depend on
what else the parent carries. When they disagree, the disagreement is a pointer
to a **shift**, and the way to check it is to split the clade into backgrounds and
measure the effect separately in each one. This module is the counts half of that
measurement; it draws panels B, D and F of ``fig:within_clade_shifts``.

Expected counts
---------------
``expected = T_pool * rate(condition)``, exactly as mat-pcp computes it
clade-wide. ``rate`` is keyed on ``(mut_type, motif, unpaired,
nt_site_before_boundary)`` and read off the era's fitted GLM. It is a property of
the mutation and the *clade founder's* sequence context, not of which PCPs are
being looked at, so it carries over to a sub-pool untouched.

``T_pool`` is the pool's evolutionary opportunity, and the only pool-specific
quantity::

    T_pool = (synonymous events observed in the pool) / (summed synonymous rate)

The numerator is pool-specific; **the denominator is not** -- it is a property of
the clade, identical for every pool cut from it. That is what makes ``T_pool`` a
*total* time rather than a mean branch length, and therefore additive when a pool
is subdivided. Numerator and denominator are restricted identically
(``mut_class == "synonymous"``, ``~exclude``, ``~masked_in_usher``); a different
restriction on one side rescales every expected count in the pool.

The at-risk requirement
-----------------------
A branch can only produce ``D574V`` if its parent still carries ``574D``. Every
branch in a pool contributes to ``T_pool``, but only at-risk branches can
contribute to ``actual``, so both pools *require* the founder residue at the
mutated site. The alternative -- computing ``expected`` on the full pool and
scaling by ``n_at_risk / n_pool`` -- weights by PCP count, which assumes at-risk
and not-at-risk branches have the same average length. They do not: a
not-at-risk branch is by construction further along its lineage. Requiring the
residue instead weights by evolutionary time, and deletes both the correction
step and the zero-at-risk edge case.

The effect is ``log(actual / expected)``, with no pseudocount on ``expected``.
In a sub-pool the expected counts run 0.01-1, so ``e + 1/2 ~ 1/2`` and
``log((a + 1/2)/(e + 1/2))`` collapses onto ``log((a + 1/2)/(1/2))`` -- a
function of the event count alone, with the expected count edited out. A pool
that observed *nothing* is the one case the plain ratio cannot express, and it
takes :data:`ZERO_COUNT_PSEUDOCOUNT` on both counts -- the only row in the table
that does.

Provenance
----------
Vendored from ``matsengrp/dasm2-experiments`` at ``f255e9361``,
``dasm2/analysis/within_clade_pools.py``, 2,338 lines reduced to the symbols
three callers here import. See ``dasm2/PROVENANCE.md``. The marker-nomination
half of the source is **not** here: this pipeline derives each clade's two
backgrounds from branch frequencies in ``scripts/backgrounds.py``, so
``marker_enrichment``, ``marker_effect_size``, ``cooccurrence``,
``majority_cotravellers`` and ``fixed_substitutions`` have no caller. Neither is
the systematic scan -- ``permutation_scan_test``, ``PermutationScan``,
``benjamini_hochberg`` -- which drew a figure panel this project has retired.

**Three deliberate divergences from the source**, so this module does not diff
clean against it:

1. :meth:`PoolSpec.matches` translates with ``dasm2.sequence.conversion``'s
   ``translate_codon`` rather than carrying its own exact-64-codon table. The
   source's own note on :func:`pool_average_log_sf` warns that a second table
   "would disagree with the shared one on N-containing codons, which BioPython
   resolves when the ambiguity is synonymous"; every other consumer in this
   repository uses the shared one, and two conventions inside one figure is the
   failure that warning describes.
2. ``chunk_rows`` is a required argument rather than defaulting to the source's
   ``DEFAULT_CHUNK_ROWS = 50_000``, which existed to match a screen script this
   repository does not have. Callers pass ``panel_spec.PCP_CHUNK_ROWS``.
3. :func:`permutation_shift_test` is **two-sided**. The source takes the tail on
   whichever side of zero the observed statistic fell, which is a tail chosen
   after the fact; see that function.
4. :func:`tally_pcps` tallies every candidate of a clade in one pass. The
   source took a single focal cell, so a clade with nine candidates paid nine
   streaming reads of a 112 MB corpus to collect nine columns over the same
   branches. The synonymous count does not depend on the candidate, so the
   shared work is done once.

GISAID restriction: reads GISAID-derived PCPs, founders and mat-pcp count
tables. Aggregate statistics may leave the cluster; the per-cell tables built
here may not.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dasm2.sequence.conversion import translate_codon

#: Rows read per chunk when streaming the big annotation table. Its own value
#: rather than the corpus chunk size: this reads a 748 MB CSV of narrow rows,
#: where the corpus pass reads wide sequence rows.
ANNOTATION_CHUNK_ROWS = 500_000


#: A pool is informative when ``max(actual, expected)`` reaches this. Either
#: count alone carries information -- 50 events against an expected 0.06 says
#: strongly favoured, 0 events against an expected 100 says strongly disfavoured
#: -- so the gate takes the larger of the two. Only both being small kills a
#: pool.
#:
#: Set to 1 rather than 3 deliberately: a single event against an expected
#: 0.0004 is a real observation with a wide error bar, and ``se`` already says
#: how wide. Excluding it discards the measurement instead of reporting its
#: uncertainty, which on the specific-background pools is most of the data --
#: 22D's ``346T+490S`` pool holds 77% of the marker pool's PCPs and 1 of its 5
#: events. The cost is that a passing pool no longer implies a tight interval,
#: so read ``se``, not ``passes_gate`` alone.
MIN_POOL_EVIDENCE = 1

#: Pseudocount applied to a cell whose ``actual`` is zero, on *both* the
#: observed and the expected count: ``log((0 + 1/2) / (expected + 1/2))``.
#: ``log(0 / expected)`` is negative infinity, so a zero needs some
#: convention, and one half is the one mat-pcp already uses -- it is also the
#: Jeffreys posterior mean for a Poisson rate observed zero times, under the
#: prior ``lambda ** -0.5``.
#:
#: Both sides, so a zero-count cell is read on the same scale as
#: :data:`MATPCP_PSEUDOCOUNT` puts every cell of the genome-wide scan on, and
#: the two approaches do not report different numbers for the same pool.
#: Numerator-only would leave ``expected`` untouched and give a systematically
#: higher effect -- on 22D's ``D574V`` the two differ by 0.67 nats -- which is
#: a difference of convention, not of evidence.
#:
#: Only a zero is touched. A cell with events is estimated from them, so the
#: rest of the table stays ``log(actual / expected)`` and a pseudocount is not
#: quietly shrinking effects that did not need it.
ZERO_COUNT_PSEUDOCOUNT = 0.5

#: Anscombe-style term in the *variance* of a log count ratio. It stabilizes the
#: standard error at small counts and never enters the point estimate.
VARIANCE_STABILIZER = 0.5

#: Rate-table join key. Every one of these is a property of the clade founder's
#: sequence context, which is why ``rate`` is pool-independent.
RATE_KEY = ["mut_type", "motif", "unpaired", "nt_site_before_boundary"]

#: Columns of ``glm/annotated_counts.csv`` this module needs. ``mut_class`` and
#: the four GLM context columns exist only in this table, not in
#: ``counts_all.csv``, which is why the larger file has to be read.
ANNOTATION_COLUMNS = [
    "clade",
    "nt_site",
    "nt_mutation",
    "mut_type",
    "exclude",
    "masked_in_usher",
    "gene",
    "clade_founder_aa",
    "mutant_aa",
    "codon_position",
    "codon_site",
    "mut_class",
    "nt_site_before_boundary",
    "unpaired",
    "motif",
]

#: Nucleotides a mutation may run between. Anything else in a corpus column is
#: an alignment artifact and is skipped rather than counted.
TALLY_BASES = frozenset("ACGT")


@dataclass(frozen=True)
class PoolSpec:
    """The parent residues a PCP must carry to join one pool.

    Three roles, kept separate because they are chosen by different arguments and
    a reader has to be able to tell which is which. A bare ``{site: residue}``
    mapping cannot express them.

    Attributes:
        name: Identifier carried into the output table.
        at_risk: ``{mutated_site: founder_residue}``. Required in *both* pools of
            a contrast, so that ``expected`` and ``actual`` describe the same set
            of branches.
        marker: The background marker, and the only thing that differs between
            the two pools of a contrast.
        hold: Sites required at their founder residue in both pools so the
            contrast changes one thing at a time. Explicit rather than defaulted:
            an omitted ``hold`` silently produces a confounded contrast, which is
            the failure this argument exists to prevent, so pass ``{}`` to say
            "checked, nothing to hold".
    """

    name: str
    at_risk: Mapping[int, str]
    marker: Mapping[int, str]
    hold: Mapping[int, str]

    def __post_init__(self) -> None:
        for role in ("at_risk", "marker", "hold"):
            mapping = getattr(self, role)
            if mapping is None:
                raise ValueError(
                    f"{self.name}: {role} must be a mapping, not None. Pass an "
                    "empty mapping to state that the role is deliberately empty."
                )
            for site, residue in mapping.items():
                if int(site) < 1:
                    raise ValueError(
                        f"{self.name}: {role} site {site} is not a 1-based amino "
                        "acid site."
                    )
                if len(residue) != 1:
                    raise ValueError(
                        f"{self.name}: {role} residue {residue!r} at site {site} "
                        "is not a single residue."
                    )

    @property
    def requirements(self) -> dict[int, str]:
        """Every required ``site -> residue``, merged across the three roles.

        Raises when two roles name the same site with different residues. That
        would be unsatisfiable and would silently produce an empty pool, which
        reads as "this background never occurs" rather than as a spec error.
        """
        merged: dict[int, str] = {}
        for role in ("at_risk", "marker", "hold"):
            for site, residue in getattr(self, role).items():
                site = int(site)
                previous = merged.get(site)
                if previous is not None and previous != residue:
                    raise ValueError(
                        f"{self.name}: site {site} is required as both "
                        f"{previous!r} and {residue!r}. No parent can satisfy "
                        "both, so the pool would come out empty."
                    )
                merged[site] = residue
        return merged

    def matches(self, parent_nt: str) -> bool:
        """Does this parent's gene-local nucleotide sequence satisfy every requirement?

        Codons are read straight out of the sequence rather than translating it
        whole: a spec names a handful of sites, and the corpus holds hundreds of
        thousands of parents.

        ``translate_codon`` is the repository's shared translation, and is
        ``@cache``'d, so the per-parent cost is a dict lookup after the first
        sighting of a codon. It returns ``"X"`` for a stop or an unreadable
        codon, and a requirement is always one of the twenty residues, so an
        unreadable parent fails the match rather than landing in a pool by
        string accident. Where the source carried its own exact-64-codon table,
        this is the divergence the module docstring records: BioPython resolves
        an N-containing codon whose ambiguity is synonymous, and every other
        consumer in this repository sees that resolution.
        """
        for site, residue in self.requirements.items():
            if translate_codon(parent_nt[3 * (site - 1) : 3 * site]) != residue:
                return False
        return True


@dataclass(frozen=True)
class PoolTallies:
    """What one streaming pass over the corpus counted, for every spec at once.

    Attributes:
        nt_observed: ``pool name -> {(nt_site, from_nt, to_nt): events}``. Keyed
            on the nucleotide change because that is what ``rate`` is keyed on;
            aggregation to amino acids happens later, by summing over routes.
        synonymous_observed: ``pool name -> events``, the numerator of ``T``.
        n_pcps: ``pool name -> PCPs``. Reported, never used in an estimate.
        n_clade_pcps: Every PCP of the clade the pass saw, pool membership
            ignored. The denominator for "what fraction of the clade is in
            this pool", which no sum over pools can supply: pools overlap
            where one names a subset of another's background, and together
            they leave out every branch carrying a mixture of it.
        synonymous_by_parent: ``pool name -> {parent nucleotide sequence: events}``.
            Supports the per-haplotype expected counts the pool average is
            weighted by, and the leave-dominant-haplotype-out check.
        pcps_by_parent: ``pool name -> {parent nucleotide sequence: PCPs}``.
    """

    nt_observed: Mapping[str, Mapping[tuple[int, str, str], int]]
    synonymous_observed: Mapping[str, int]
    n_pcps: Mapping[str, int]
    synonymous_by_parent: Mapping[str, Mapping[str, int]]
    pcps_by_parent: Mapping[str, Mapping[str, int]]
    n_clade_pcps: int


def load_annotations(
    annotated_counts_path: str | Path,
    clades: Iterable[str],
    gene: str,
    *,
    chunk_rows: int = ANNOTATION_CHUNK_ROWS,
) -> pd.DataFrame:
    """Per-clade annotated nucleotide-mutation rows, in one pass over the big table."""
    wanted = set(clades)
    parts = []
    for chunk in pd.read_csv(
        annotated_counts_path,
        usecols=ANNOTATION_COLUMNS,
        chunksize=chunk_rows,
        dtype={"codon_site": "object", "codon_position": "object"},
    ):
        chunk = chunk[chunk["clade"].isin(wanted) & (chunk["gene"] == gene)]
        if len(chunk):
            parts.append(chunk)
    if not parts:
        raise ValueError(
            f"no annotated rows for clades {sorted(wanted)} in gene {gene}. "
            "Without them every expected count would be zero."
        )
    out: pd.DataFrame = pd.concat(parts, ignore_index=True)
    out["codon_site"] = out["codon_site"].astype(int)
    out["codon_position"] = out["codon_position"].astype(int)
    return out


def join_rates(annotations: pd.DataFrame, master_table: pd.DataFrame) -> pd.DataFrame:
    """Join the era's fitted GLM rate onto the annotations.

    Raises on any row that finds no rate. A missing rate becomes a missing
    expected count, which reads downstream as a mutation with no neutral
    opportunity rather than as a failed join.
    """
    # ``validate="many_to_one"`` because a master table with two rows sharing a
    # RATE_KEY would otherwise duplicate the annotation row, and
    # ``aggregate_to_aa_cells``' groupby-sum would then double-count that route's
    # contribution to ``expected``. A merge whose whole job is producing a
    # scientific denominator should fail rather than inflate.
    merged = annotations.merge(
        master_table[RATE_KEY + ["rate"]],
        on=RATE_KEY,
        how="left",
        validate="many_to_one",
    )
    missing = int(merged["rate"].isna().sum())
    if missing:
        raise ValueError(
            f"{missing} of {len(merged)} annotated rows found no rate in the "
            "master table. A missing rate silently becomes a missing expected "
            "count, so the join is required to be total."
        )
    return merged


@dataclass(frozen=True)
class EvolutionaryTime:
    """``T`` for one pool, with the two quantities it was built from.

    Attributes:
        time: ``synonymous_observed / rate_sum``.
        rate_sum: Summed synonymous rate. A property of the **clade**, identical
            for every pool cut from it, which is what makes ``time`` additive
            under subdivision.
        n_synonymous_sites: Rows the sum ran over. Reported so a restriction
            applied to one side and not the other is visible.
        synonymous_observed: The pool-specific numerator.
    """

    time: float
    rate_sum: float
    n_synonymous_sites: int
    synonymous_observed: int


def pool_evolutionary_time(
    annotations: pd.DataFrame, synonymous_observed: int
) -> EvolutionaryTime:
    """mat-pcp's ``T``: observed synonymous events over summed synonymous rate.

    ``annotations`` is the whole clade's rate-joined frame, not the pool's. Only
    the numerator is pool-specific; see the module docstring.
    """
    usable = ~annotations["exclude"] & ~annotations["masked_in_usher"]
    synonymous = annotations[usable & (annotations["mut_class"] == "synonymous")]
    rate_sum = float(synonymous["rate"].sum())
    if rate_sum <= 0:
        raise ValueError(
            "summed synonymous rate is zero, so T is undefined. Either the rate "
            "join produced zeros or the exclude/mask restriction removed every "
            "synonymous site."
        )
    if synonymous_observed < 0:
        raise ValueError(f"synonymous_observed is {synonymous_observed}, not a count.")
    return EvolutionaryTime(
        time=synonymous_observed / rate_sum,
        rate_sum=rate_sum,
        n_synonymous_sites=len(synonymous),
        synonymous_observed=synonymous_observed,
    )


def tally_pools(
    pcp_path: str | Path,
    specs: Sequence[PoolSpec],
    annotations: pd.DataFrame,
    *,
    clade: str,
    chunk_rows: int,
    max_chunks: int | None = None,
) -> PoolTallies:
    """One streaming pass over the corpus, serving every spec at once.

    Every spec is tallied in the same pass because a pass costs the whole corpus
    however few rows are kept -- the aggregate file holds all of the tree's
    clades. A per-spec signature would read the corpus once per candidate.

    ``annotations`` supplies two things that cannot be derived from the corpus:
    the map from a gene-local codon position to a genome ``nt_site``, and which
    ``(nt_site, nt_mutation)`` pairs count toward ``T``.

    ``max_chunks`` stops the pass early. It exists for the wiring check only:
    every count it returns is a partial tally, so nothing it produces is a
    measurement. Callers that set it must say so in their output.
    """
    if not specs:
        raise ValueError("no pool specs given; there is nothing to tally.")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"pool names must be unique, got {names}.")

    clade_rows = annotations[annotations["clade"] == clade]
    if clade_rows.empty:
        raise ValueError(
            f"no annotated rows for clade {clade!r}; the site map would be empty "
            "and every mutation would be silently skipped."
        )
    site_map = (
        clade_rows.groupby(["codon_site", "codon_position"])["nt_site"]
        .first()
        .to_dict()
    )
    synonymous_rows = clade_rows[
        (clade_rows["mut_class"] == "synonymous")
        & ~clade_rows["exclude"]
        & ~clade_rows["masked_in_usher"]
    ]
    synonymous_keys = set(
        zip(synonymous_rows["nt_site"], synonymous_rows["nt_mutation"])
    )

    nt_observed: dict[str, Counter] = {name: Counter() for name in names}
    synonymous_observed: dict[str, int] = {name: 0 for name in names}
    n_pcps: dict[str, int] = {name: 0 for name in names}
    synonymous_by_parent: dict[str, Counter] = {name: Counter() for name in names}
    pcps_by_parent: dict[str, Counter] = {name: Counter() for name in names}

    # Spec matching depends only on the parent, and a clade's parents repeat
    # heavily -- 22D has 16,576 PCPs over 2,410 distinct parents. Memoizing the
    # match turns a per-PCP codon lookup into a per-parent one.
    membership: dict[str, tuple[str, ...]] = {}
    n_clade_pcps = 0

    for chunk_index, chunk in enumerate(
        pd.read_csv(
            pcp_path, usecols=["clade", "parent", "child"], chunksize=chunk_rows
        )
    ):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        chunk = chunk[chunk["clade"] == clade]
        if chunk.empty:
            continue
        n_clade_pcps += len(chunk)
        for parent, child in zip(chunk["parent"], chunk["child"]):
            pools = membership.get(parent)
            if pools is None:
                pools = tuple(spec.name for spec in specs if spec.matches(parent))
                membership[parent] = pools
            if not pools:
                continue
            for name in pools:
                n_pcps[name] += 1
                pcps_by_parent[name][parent] += 1
            for position, (source, target) in enumerate(zip(parent, child)):
                if source == target:
                    continue
                if source not in TALLY_BASES or target not in TALLY_BASES:
                    continue
                genome_site = site_map.get((position // 3 + 1, position % 3 + 1))
                if genome_site is None:
                    continue
                is_synonymous = (
                    genome_site,
                    f"{source}{genome_site}{target}",
                ) in synonymous_keys
                for name in pools:
                    nt_observed[name][(genome_site, source, target)] += 1
                    if is_synonymous:
                        synonymous_observed[name] += 1
                        synonymous_by_parent[name][parent] += 1

    return PoolTallies(
        nt_observed=nt_observed,
        synonymous_observed=synonymous_observed,
        n_pcps=n_pcps,
        synonymous_by_parent=synonymous_by_parent,
        pcps_by_parent=pcps_by_parent,
        n_clade_pcps=n_clade_pcps,
    )


#: Stands in for a pool name on the remainder row of
#: :func:`branch_fraction_table`. A name rather than a null, so the row
#: survives a round trip through JSON and reads as itself in a printed table.
UNASSIGNED_POOL = "(neither)"


def branch_fraction_table(
    tallies: PoolTallies,
    *,
    founder_pool: str,
    marker_pools: Sequence[str],
    eligible_pool: str,
) -> pd.DataFrame:
    """How a clade's branches divide between the two pools of each contrast.

    Two denominators, because the pools are carved out of the clade in two
    steps and a single percentage hides which one did the work. ``of_clade``
    is against every branch in the clade; ``of_eligible`` is against the
    branches whose parent can still make the focal mutation, which is the set
    the contrast is actually drawn from. ``eligible_pool`` names the spec that
    requires the at-risk residue and nothing else.

    One row per ``(contrast, pool)``, plus an ``unassigned`` row per contrast
    -- named ``(neither)``, since it is a remainder and not a pool -- for the
    eligible branches in neither side. Those carry part of the background and
    not the rest, and their share is the cost of naming the background
    symmetrically, invisible in the pool sizes alone.

    Args:
        tallies: One pass's counts, including ``eligible_pool`` and every
            named pool.
        founder_pool: The pool both contrasts are measured against.
        marker_pools: One marker pool per contrast, each paired with
            ``founder_pool``.
        eligible_pool: The at-risk-only spec, supplying the eligible
            denominator.

    Returns:
        Columns ``contrast``, ``pool_name``, ``role`` (``founder`` /
        ``marker`` / ``unassigned``), ``n_pcps``, ``of_eligible``,
        ``of_clade``, ``n_eligible`` and ``n_clade``.

    Raises:
        ValueError: If a named pool is missing from ``tallies``, if the clade
            or eligible count is zero, or if a contrast's two pools together
            hold more branches than are eligible -- which would mean they
            overlap, and the contrast is not a split.
    """
    missing = [
        name
        for name in (eligible_pool, founder_pool, *marker_pools)
        if name not in tallies.n_pcps
    ]
    if missing:
        raise ValueError(
            f"pools {missing} were never tallied, so their share of the clade "
            "cannot be computed. Pass them to `tally_pools` in the same pass."
        )
    n_clade = tallies.n_clade_pcps
    n_eligible = tallies.n_pcps[eligible_pool]
    if not n_clade or not n_eligible:
        raise ValueError(
            f"clade holds {n_clade} PCPs of which {n_eligible} are eligible; "
            "a fraction of either would divide by zero."
        )
    rows = []
    for marker_pool in marker_pools:
        n_founder = tallies.n_pcps[founder_pool]
        n_marker = tallies.n_pcps[marker_pool]
        unassigned = n_eligible - n_founder - n_marker
        if unassigned < 0:
            raise ValueError(
                f"{founder_pool} and {marker_pool} hold "
                f"{n_founder + n_marker} eligible PCPs between them but only "
                f"{n_eligible} are eligible, so they overlap. The two sides "
                "of a contrast must be disjoint."
            )
        for name, role, count in (
            (founder_pool, "founder", n_founder),
            (marker_pool, "marker", n_marker),
            (UNASSIGNED_POOL, "unassigned", unassigned),
        ):
            rows.append(
                {
                    "contrast": marker_pool,
                    "pool_name": name,
                    "role": role,
                    "n_pcps": int(count),
                    "of_eligible": count / n_eligible,
                    "of_clade": count / n_clade,
                    "n_eligible": int(n_eligible),
                    "n_clade": int(n_clade),
                }
            )
    return pd.DataFrame(rows)


def aggregate_to_aa_cells(
    annotations: pd.DataFrame,
    nt_observed: Mapping[str, Mapping[tuple[int, str, str], int]],
    times: Mapping[str, EvolutionaryTime],
) -> pd.DataFrame:
    """Sum nucleotide mutations onto amino-acid cells, per pool.

    Summing rather than averaging is what makes a pool that mixes parent codons
    correct: two codons for the same residue are different ``mut_type`` rows with
    their own rates, and each contributes its own expected count.
    """
    usable = (
        ~annotations["exclude"]
        & ~annotations["masked_in_usher"]
        & (annotations["mut_class"] == "nonsynonymous")
    )
    candidates = annotations[usable].copy()
    if candidates.empty:
        raise ValueError(
            "no nonsynonymous candidate rows survived the exclude/mask "
            "restriction; there are no cells to report."
        )
    candidates["from_nt"] = candidates["nt_mutation"].str[0]
    candidates["to_nt"] = candidates["nt_mutation"].str[-1]

    value_columns = []
    for name, time in times.items():
        observed = nt_observed[name]
        candidates[f"expected__{name}"] = time.time * candidates["rate"]
        candidates[f"actual__{name}"] = [
            observed.get((site, source, target), 0)
            for site, source, target in zip(
                candidates["nt_site"], candidates["from_nt"], candidates["to_nt"]
            )
        ]
        value_columns += [f"expected__{name}", f"actual__{name}"]

    grouped = (
        candidates.groupby(["codon_site", "clade_founder_aa", "mutant_aa"])[
            value_columns
        ]
        .sum()
        .reset_index()
        .rename(columns={"codon_site": "aa_site"})
    )
    return grouped[grouped["mutant_aa"] != "*"].reset_index(drop=True)


def counts_effect(
    actual: np.ndarray | pd.Series,
    expected: np.ndarray | pd.Series,
    *,
    pseudocount: float = 0.0,
) -> pd.DataFrame:
    """The per-pool effect and its standard error.

    * ``effect`` -- ``log(actual / expected)``, except where ``actual`` is
      zero, which is read as ``log((0 + 1/2) / (expected + 1/2))`` under
      :data:`ZERO_COUNT_PSEUDOCOUNT`. Every pool therefore gets a point
      estimate; none gets a bound.
    * ``se`` -- ``sqrt(1 / (actual + 1/2))``, the Poisson delta-method standard
      error of a log count ratio, defined for a zero count too. The ``1/2`` is
      a variance stabilizer and does not enter ``effect``.

    A zero count carries little information, and ``se`` is where that shows:
    at ``actual = 0`` it is 1.41 nats, the widest any row can be. Read the
    estimate with it, not on its own.

    ``pseudocount`` exists for one caller only -- reproducing mat-pcp's
    ``delta_fitness_ratio``, which *is* ``log((a + 1/2)/(e + 1/2))``. It must be
    left at 0 for per-pool work; see the module docstring on why adding it to
    *both* counts saturates in a sub-pool.
    """
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if np.any(expected <= 0):
        raise ValueError(
            "expected count of zero or less: every cell needs neutral "
            "opportunity before its effect means anything."
        )
    se = np.sqrt(1.0 / (actual + VARIANCE_STABILIZER))
    if pseudocount:
        return pd.DataFrame(
            {
                "effect": np.log((actual + pseudocount) / (expected + pseudocount)),
                "se": se,
            }
        )
    # Only the zeros move, and they move on both sides of the ratio; see
    # `ZERO_COUNT_PSEUDOCOUNT`.
    saw_nothing = actual == 0
    correction = np.where(saw_nothing, float(ZERO_COUNT_PSEUDOCOUNT), 0.0)
    return pd.DataFrame(
        {"effect": np.log((actual + correction) / (expected + correction)), "se": se}
    )


def passes_evidence_gate(
    actual: np.ndarray | pd.Series,
    expected: np.ndarray | pd.Series,
    *,
    minimum: int = MIN_POOL_EVIDENCE,
) -> np.ndarray:
    """``max(actual, expected) >= minimum``, elementwise.

    See :data:`MIN_POOL_EVIDENCE` for why the gate takes the larger of the two
    rather than requiring both.
    """
    larger = np.maximum(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    )
    return np.asarray(larger >= minimum)


def pool_cell_table(
    tallies: PoolTallies,
    annotations: pd.DataFrame,
    *,
    cluster: str,
    gene: str,
    pool0: PoolSpec,
    pool1: PoolSpec,
    pseudocount: float = 0.0,
) -> pd.DataFrame:
    """Counts-side cell table for one contrast. Pure: no IO, no model, no checkpoint.

    ``annotations`` must already carry ``rate`` (see :func:`join_rates`) and be
    restricted to ``cluster`` and ``gene``. One row per amino-acid cell per pool,
    long rather than wide so that a third pool -- the at-risk-free spec the
    count-ratio comparison needs -- does not change the schema.

    Column names follow ``<clade>_aamut_fitness.csv``: the clade is ``cluster``
    and the residues are ``clade_founder_aa`` and ``mutant_aa``.
    """
    clades = annotations["clade"].unique().tolist()
    if clades != [cluster]:
        raise ValueError(
            f"annotations carry clade(s) {clades} but the table is being built "
            f"for {cluster!r}. `T` is fitted from the summed synonymous rate over "
            "this frame, so a second clade's rows would inflate the denominator "
            "for every pool, and the amino-acid groupby would merge two clades' "
            "cells that happen to share a site and residues."
        )
    names = [pool0.name, pool1.name]
    for name in names:
        if name not in tallies.synonymous_observed:
            raise ValueError(
                f"pool {name!r} was not tallied; tally_pools must be given every "
                f"spec that pool_cell_table is asked about. Tallied: "
                f"{sorted(tallies.synonymous_observed)}."
            )
    times = {
        name: pool_evolutionary_time(annotations, tallies.synonymous_observed[name])
        for name in names
    }
    cells = aggregate_to_aa_cells(annotations, tallies.nt_observed, times)

    frames = []
    for index, spec in enumerate((pool0, pool1)):
        name = spec.name
        frame = cells[["aa_site", "clade_founder_aa", "mutant_aa"]].copy()
        frame.insert(0, "cluster", cluster)
        frame.insert(1, "gene", gene)
        frame["pool"] = index
        frame["pool_name"] = name
        frame["marker"] = ";".join(
            f"{site}{residue}" for site, residue in sorted(spec.marker.items())
        )
        frame["hold"] = ";".join(
            f"{site}{residue}" for site, residue in sorted(spec.hold.items())
        )
        frame["n_pcps"] = tallies.n_pcps[name]
        frame["synonymous_observed"] = times[name].synonymous_observed
        frame["T_pool"] = times[name].time
        frame["synonymous_rate_sum"] = times[name].rate_sum
        frame["n_synonymous_sites"] = times[name].n_synonymous_sites
        frame["actual"] = cells[f"actual__{name}"].to_numpy()
        frame["expected"] = cells[f"expected__{name}"].to_numpy()
        effect = counts_effect(
            frame["actual"], frame["expected"], pseudocount=pseudocount
        )
        frame["counts_effect"] = effect["effect"].to_numpy()
        frame["se"] = effect["se"].to_numpy()
        frame["passes_gate"] = passes_evidence_gate(frame["actual"], frame["expected"])
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def pooled_log_sf(expected_weights: Sequence[float], log_sfs: Sequence[float]) -> float:
    """The pool's mean selection factor, in the **linear** domain.

    The counts effect is a ratio of sums over the pool: haplotype ``h``
    contributes ``e_h * exp(log_sf_h)`` events, and the counts side divides
    ``sum(actual_h)`` by ``sum(e_h)``. Matching that forces an arithmetic mean of
    the selection factors::

        log_sf_pool = log( sum_h e_h * exp(log_sf_h) / sum_h e_h )

    A log-space average would give the geometric mean, which answers "what does a
    typical haplotype do" rather than "what does the pool produce", and by
    Jensen's inequality sits below this by about ``sigma**2 / 2``.

    The weight is ``e_h``, the haplotype's *expected count*, not its PCP count.
    ``rate`` is common across the haplotypes of a pool, so ``e_h`` is
    proportional to ``T_h`` -- the haplotype's own evolutionary time, which is
    what ``expected`` is built from. PCP count is only a proxy for branch length,
    the same approximation the at-risk requirement removes.
    """
    weights = np.asarray(expected_weights, dtype=float)
    values = np.asarray(log_sfs, dtype=float)
    if weights.shape != values.shape:
        raise ValueError(
            f"got {weights.size} weights for {values.size} selection factors."
        )
    if np.any(weights < 0):
        raise ValueError("expected-count weights cannot be negative.")
    total = weights.sum()
    if total <= 0:
        raise ValueError(
            "every haplotype has zero expected count, so the pool average is "
            "undefined. A haplotype with no synonymous events has no fitted "
            "evolutionary time; report the discarded PCP mass instead of "
            "averaging over nothing."
        )
    # Shifted before exponentiating: a log selection factor of +7 is ordinary
    # here and exp(7) is fine, but the shift costs nothing and keeps a +700
    # pathology from silently becoming inf.
    offset = values.max()
    return float(offset + np.log((weights * np.exp(values - offset)).sum() / total))


def substitution_key(aa_sequence: str, founder_aa: str) -> str:
    """``D142G;R346T``: a haplotype's differences from the founder, site-ordered.

    The join key between the DASM side's per-haplotype scores and the counts
    side's per-parent tallies. It lives here, rather than being spelled out in
    each script, because the two sides must agree character for character: a
    haplotype written two ways does not collide, it goes *missing*, and a
    missing haplotype silently shrinks the pool average's support instead of
    raising.

    Lengths must match. ``zip`` would stop at the shorter sequence and drop
    every difference past its end, so a truncated parent would report as
    identical to the founder over the tail -- the same silent-truncation bug
    :func:`_stack_haplotypes` guards against on the array path.
    """
    if len(aa_sequence) != len(founder_aa):
        raise ValueError(
            f"haplotype has {len(aa_sequence)} residues and the founder has "
            f"{len(founder_aa)}. Differences past the shorter sequence cannot "
            "be read, and truncating silently would call the tail identical."
        )
    return ";".join(
        f"{founder}{site + 1}{observed}"
        for site, (observed, founder) in enumerate(zip(aa_sequence, founder_aa))
        if observed != founder
    )


@dataclass(frozen=True)
class PoolAverage:
    """The expected-count-weighted mean effect over a pool's own haplotypes.

    Attributes:
        log_sf: the pool average, :func:`pooled_log_sf` of the joined table.
        n_haplotypes: distinct amino-acid haplotypes contributing weight.
        synonymous_weight: total synonymous events behind the average. This is
            the pool's ``T`` in un-normalized units, so it is the support the
            number rests on.
        zero_weight_haplotypes: haplotypes in the pool that recorded no
            synonymous event. They have no fitted evolutionary time, so they
            carry no weight; reported because a pool where most haplotypes land
            here is being represented by a handful of parents.
    """

    log_sf: float
    n_haplotypes: int
    synonymous_weight: float
    zero_weight_haplotypes: tuple[str, ...]


def pool_average_log_sf(
    synonymous_by_parent: Mapping[str, int],
    aa_by_parent: Mapping[str, str],
    log_sf_by_haplotype: Mapping[str, float],
    *,
    founder_aa: str,
) -> PoolAverage:
    """Average a pool's per-haplotype DASM effects by expected count.

    The weight is the haplotype's **synonymous event count**, which is
    proportional to its expected count. ``expected = T_h * rate(condition)``,
    and ``rate`` is read off the clade founder, so it is common to every
    haplotype of the clade for a given cell; ``T_h`` is the haplotype's
    synonymous events over a clade-level rate sum, whose denominator is also
    common. Both constants cancel in a weighted mean, so synonymous events are
    the expected-count weights up to a factor that does not matter.

    This is the quantity that is apples-to-apples with the counts effect. The
    counts side divides the pool's summed events by its summed expected count;
    matching that requires the arithmetic mean of the selection factors over
    the same weights, which is what :func:`pooled_log_sf` computes.

    ``aa_by_parent`` is supplied by the caller rather than translated here so
    that this module stays free of the translation dependency *and* so that
    both scripts translate with one implementation. Reimplementing translation
    against the 64 exact codons would disagree with the shared one on
    N-containing codons, which BioPython resolves when the ambiguity is
    synonymous -- a disagreement that would surface as unmatched haplotypes.

    Raises:
        ValueError: if a parent carrying synonymous events has no translation,
            or if its haplotype has no DASM score. Either would drop
            evolutionary time out of the average without saying so.
    """
    weights: dict[str, float] = {}
    for parent, events in synonymous_by_parent.items():
        aa_parent = aa_by_parent.get(parent)
        if aa_parent is None:
            if events:
                raise ValueError(
                    f"a parent carrying {events} synonymous event(s) has no "
                    "translation in aa_by_parent, so its evolutionary time "
                    "cannot be assigned to a haplotype."
                )
            continue
        key = substitution_key(aa_parent, founder_aa)
        weights[key] = weights.get(key, 0.0) + float(events)

    # Every haplotype of the pool, including the ones with no synonymous event,
    # so that a pool represented by a few parents is visible rather than
    # implied. Zero-weight haplotypes still have to be *scored*: an unscored
    # haplotype means the marker discovery and the tally disagree about which
    # parents the pool contains, which is a bug, not a weighting question.
    all_keys = {
        substitution_key(aa_by_parent[parent], founder_aa)
        for parent in synonymous_by_parent
        if parent in aa_by_parent
    }
    unscored = sorted(all_keys - set(log_sf_by_haplotype))
    if unscored:
        raise ValueError(
            f"{len(unscored)} of the pool's {len(all_keys)} haplotypes have no "
            f"DASM score, e.g. {unscored[:3]}. The score table and the tally "
            "disagree about the pool's membership; averaging over the "
            "intersection would quietly drop evolutionary time."
        )

    carrying = {key: weight for key, weight in weights.items() if weight > 0}
    if not carrying:
        raise ValueError(
            "no haplotype in the pool recorded a synonymous event, so the pool "
            "has no fitted evolutionary time to weight by."
        )
    keys = sorted(carrying)
    return PoolAverage(
        log_sf=pooled_log_sf(
            [carrying[key] for key in keys],
            [log_sf_by_haplotype[key] for key in keys],
        ),
        n_haplotypes=len(keys),
        synonymous_weight=float(sum(carrying.values())),
        zero_weight_haplotypes=tuple(sorted(all_keys - set(carrying))),
    )


def join_dasm_effects(
    cells: pd.DataFrame,
    log_sf_by_pool: Mapping[str, float],
    *,
    pooled_by_pool: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Add the DASM columns to a cell table built by :func:`pool_cell_table`.

    ``log_sf_by_pool`` is the constructed two-background contrast: the model's
    score for this cell on a sequence that differs from the other pool's only by
    the marker. ``pooled_by_pool`` is the optional expected-count-weighted pool
    mean from :func:`pooled_log_sf`, which checks it.
    """
    missing = set(cells["pool_name"]) - set(log_sf_by_pool)
    if missing:
        raise ValueError(
            f"no DASM score for pool(s) {sorted(missing)}. A missing score would "
            "join as NaN and drop the cell from the shift without saying so."
        )
    out = cells.copy()
    out["dasm_effect"] = out["pool_name"].map(log_sf_by_pool).astype(float)
    if pooled_by_pool is not None:
        out["dasm_effect_pooled"] = out["pool_name"].map(pooled_by_pool).astype(float)
    return out


def shift_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Collapse a two-pool cell table to one shift row per cell.

    Every shift is **pool1 minus pool0** -- alternative minus founder. ``se``
    and ``z`` are defined for every row, a pool with no events included: that
    pool's effect is an estimate under :data:`ZERO_COUNT_PSEUDOCOUNT` rather
    than a bound, and its ``se`` of 1.41 nats is what says how little a zero
    count settles.

    .. warning::

       :func:`permutation_scan_test`'s ``statistic`` uses the **opposite**
       convention, pool0 minus pool1. Both are deliberate and tested, and
       nothing joins them today. A caller that does must negate one side.
    """
    key = ["cluster", "gene", "aa_site", "clade_founder_aa", "mutant_aa"]
    pools = {pool: frame.set_index(key) for pool, frame in cells.groupby("pool")}
    if set(pools) != {0, 1}:
        raise ValueError(
            f"shift_table needs exactly pools 0 and 1, got {sorted(map(str, pools))}."
        )
    for pool, frame in pools.items():
        if not frame.index.is_unique:
            duplicated = frame.index[frame.index.duplicated()].tolist()[:3]
            raise ValueError(
                f"pool {pool!r} has more than one row for cell(s) {duplicated!r}. "
                "A cell key must be unique within a pool; duplicates would "
                "misalign the two pools against each other."
            )
    low, high = pools[0], pools[1]
    common = low.index.intersection(high.index)
    low, high = low.loc[common], high.loc[common]

    se_shift = np.sqrt(
        1.0 / (low["actual"].to_numpy() + VARIANCE_STABILIZER)
        + 1.0 / (high["actual"].to_numpy() + VARIANCE_STABILIZER)
    )
    counts_shift = high["counts_effect"].to_numpy() - low["counts_effect"].to_numpy()

    out = pd.DataFrame(
        {
            "marker": low["marker"].to_numpy() + " -> " + high["marker"].to_numpy(),
            "counts_shift": counts_shift,
            "se": se_shift,
            "z": counts_shift / se_shift,
            "passes_gate": low["passes_gate"].to_numpy()
            & high["passes_gate"].to_numpy(),
        },
        index=common,
    )
    for column in ("dasm_effect", "dasm_effect_pooled"):
        if column in low.columns:
            out[f"{column.replace('dasm_effect', 'dasm_shift')}"] = (
                high[column].to_numpy() - low[column].to_numpy()
            )
    return out.reset_index()


#: Permutations drawn by default. The attainable floor on a permutation p-value
#: is ``1 / (X + 1)``, so this one reports down to 1e-4 -- enough to separate a
#: strong shift from the gate without claiming a precision the draw count cannot
#: support. Report ``p < 1/(X+1)``, never ``p = 0``.
DEFAULT_PERMUTATIONS = 10_000


@dataclass(frozen=True)
class PermutationTest:
    """A randomization test of one contrast's counts shift.

    Attributes:
        observed: The test statistic, ``log((a1/s1) / (a0/s0))`` with
            ``pseudocount`` added to both focal counts.
        observed_uncorrected: The same ratio at ``pseudocount = 0``, which is
            the shift the pipeline reports. Shown beside ``observed`` so the
            correction's size is visible rather than implied.
        p_value: ``(1 + extreme) / (1 + n_permutations)``, **two-sided**. The
            ``1 +`` counts the observed labelling as one of the draws, which is
            what keeps the test valid at small counts, and it floors the
            p-value at ``1 / (n_permutations + 1)`` -- report a run that
            reached the floor as an inequality, never as zero.
        n_extreme: Permutations at least as far from zero as ``observed``, on
            either side: ``abs(null) >= abs(observed)``.
        n_undefined: Permutations whose statistic came out non-finite. Zero
            whenever ``pseudocount`` is positive; counted as extreme if not.
        pseudocount: What was added to both focal counts.
        null: Every permuted statistic, for plotting.
    """

    observed: float
    observed_uncorrected: float
    p_value: float
    n_permutations: int
    n_extreme: int
    n_undefined: int
    pseudocount: float
    null: np.ndarray


def log_rate_ratio(
    focal0: float,
    synonymous0: float,
    focal1: float,
    synonymous1: float,
    *,
    pseudocount: float = 0.0,
) -> float:
    """The counts shift, written as what it algebraically is.

    ``effect = log(actual / expected)`` and ``expected = (s / R) * rate``, where
    the summed synonymous rate ``R`` and the cell's ``rate`` are properties of
    the *clade* -- identical in both pools. Both therefore cancel out of the
    difference::

        shift = log(a1 / e1) - log(a0 / e0) = log((a1 / s1) / (a0 / s0))

    So the contrast is a 2x2 log rate ratio of focal events against synonymous
    events, and needs no rate table to permute.

    ``pseudocount`` is added to both focal counts only, the Haldane-Anscombe
    correction for a log ratio with an empty cell. At 0 -- the default, and
    what the pipeline reports -- a pool with no focal events gives an infinite
    ratio, which the caller has to handle rather than clip. A permutation test
    should set it to 0.5: it applies the same shrinkage to the observed
    statistic and to every draw, so the test stays exact, and it stops a
    relabelling that empties a pool from registering as infinitely extreme.
    """
    if pseudocount < 0:
        raise ValueError(f"pseudocount is {pseudocount}; it cannot be negative.")
    for name, value in (
        ("focal0", focal0),
        ("synonymous0", synonymous0),
        ("focal1", focal1),
        ("synonymous1", synonymous1),
    ):
        if value < 0:
            raise ValueError(f"{name} is {value}, not a count.")
    if synonymous0 <= 0 or synonymous1 <= 0:
        raise ValueError(
            "a pool with no synonymous events has no fitted evolutionary time, "
            "so its effect is undefined rather than infinite."
        )
    focal0, focal1 = focal0 + pseudocount, focal1 + pseudocount
    if focal0 == 0 and focal1 == 0:
        return float("nan")
    if focal0 == 0:
        return float("inf")
    if focal1 == 0:
        return float("-inf")
    return float(np.log((focal1 / synonymous1) / (focal0 / synonymous0)))


def permutation_shift_test(
    pcps: pd.DataFrame,
    *,
    pool0: str,
    pool1: str,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    pseudocount: float = 0.5,
    seed: int = 0,
) -> PermutationTest:
    """Shuffle branches between the pools and ask how often the shift gets this large.

    ``pcps`` is :func:`tally_pcps`'s frame: one row per branch, carrying its
    pool, its focal events and its synonymous events. Each draw
    reassigns branches to the two pools, keeping each pool's branch count
    fixed, and recomputes the shift. ``expected`` moves with the labels the way
    ``actual`` does -- it is ``(synonymous / R) * rate``, and only the pool's
    synonymous count is pool-specific -- so both sides of the ratio are
    recomputed from the shuffled pool, not held at their observed values.

    The branch is the unit because mat-pcp counts events per branch: each is a
    distinct evolutionary opportunity rather than a copy of its ancestor. Focal
    and synonymous events move together because they are properties of the same
    branch, which is what this buys over a Fisher exact test on the four
    totals.

    ``pseudocount`` defaults to 0.5, the Haldane-Anscombe correction. A draw
    that leaves a pool with no focal events has an infinite statistic and is
    therefore infinitely extreme, which puts a floor under the p-value that no
    amount of signal can get below; on a thin candidate that floor can be the
    whole answer. The correction is applied identically to the observed
    statistic and to every draw, so the test remains exact -- it changes what
    is being ranked, not how.

    The p-value is **two-sided**, and this is a divergence from the source.

    The question is whether a counts shift this large is likely to arise by
    chance -- signal against noise in the counts alone. It is not about whether
    the counts and the DASM agree, and the model takes no part in it. Nothing
    fixes a direction in advance, so the tail is taken on both sides.

    The source took the tail on whichever side of zero the *observed* statistic
    fell, justified by the marker having been chosen by the model, "so the
    direction is fixed before the counts are consulted". Two things are wrong
    with that. The implementation never did what the argument describes: the
    model fixed the marker, not the tail, which was read off the sign of the
    observed counts statistic -- a tail chosen after the fact, worth about a
    factor of two. And here the premise is gone as well, because
    ``scripts/backgrounds.py`` derives the split from branch frequencies with
    no model involved at any point.

    Branches sharing a parent are **not** pseudo-replicates, and must not be
    collapsed to one observation: mat-pcp counts each branch's events on that
    branch, and descendants do not re-count, so two branches off one parent
    that both show the focal mutation are two independent occurrences of it.
    Aggregating them would discard real evidence, not correct for double
    counting.

    What this cannot do is speak to cause. The marker is inherited, so the
    marker pool is a clade, and everything else about that clade travels with
    the label. A small p-value says the pool labels are not exchangeable --
    which is what the contrast was built to be. Only replication across clades
    that carry the marker on different backgrounds says the marker is why.
    """
    if n_permutations < 1:
        raise ValueError(f"n_permutations is {n_permutations}; draw at least one.")
    missing = {"pool", "focal", "synonymous"} - set(pcps.columns)
    if missing:
        raise ValueError(f"pcps is missing column(s) {sorted(missing)}.")
    frame = pcps[pcps["pool"].isin([pool0, pool1])]
    if frame.empty:
        raise ValueError(
            f"no rows in pools {pool0!r} or {pool1!r}; nothing to permute."
        )
    for name in (pool0, pool1):
        if not (frame["pool"] == name).any():
            raise ValueError(f"pool {name!r} holds no branches.")

    focal = frame["focal"].to_numpy(dtype=np.int64)
    synonymous = frame["synonymous"].to_numpy(dtype=np.int64)
    in_pool1 = (frame["pool"] == pool1).to_numpy()
    total_focal, total_synonymous = int(focal.sum()), int(synonymous.sum())
    counted = (
        total_focal - int(focal[in_pool1].sum()),
        total_synonymous - int(synonymous[in_pool1].sum()),
        int(focal[in_pool1].sum()),
        int(synonymous[in_pool1].sum()),
    )
    observed = log_rate_ratio(*counted, pseudocount=pseudocount)
    if not np.isfinite(observed):
        raise ValueError(
            f"the observed statistic is {observed}: a pool holds no focal "
            "events and no pseudocount was applied, so there is no finite "
            "statistic to compare a null against."
        )

    rng = np.random.default_rng(seed)
    order = np.arange(len(frame))
    n1 = int(in_pool1.sum())
    null = np.empty(n_permutations, dtype=float)
    for draw in range(n_permutations):
        rng.shuffle(order)
        chosen = order[:n1]
        focal1 = int(focal[chosen].sum())
        synonymous1 = int(synonymous[chosen].sum())
        synonymous0 = total_synonymous - synonymous1
        if synonymous1 <= 0 or synonymous0 <= 0:
            # `log_rate_ratio` *raises* on an empty synonymous pool, which is
            # right for the observed statistic -- there is no fitted
            # evolutionary time, so the effect is undefined rather than large.
            # Inside the loop it would abort the whole test on one unlucky
            # draw. A draw that strands every synonymous event on one side is
            # recorded as undefined instead, which `n_extreme` below treats as
            # extreme; that is the conservative direction. It cannot happen on
            # the pools in this PR (721 and 7,051 synonymous events), but
            # `Outstanding` proposes the thinner candidates where it could.
            null[draw] = float("nan")
            continue
        null[draw] = log_rate_ratio(
            total_focal - focal1,
            synonymous0,
            focal1,
            synonymous1,
            pseudocount=pseudocount,
        )
    n_undefined = int((~np.isfinite(null)).sum())
    # An undefined draw has no finite statistic to rank, so it counts as
    # extreme rather than being silently dropped from the numerator -- a
    # `nan >= x` comparison is False, which would have made undefined draws
    # count as *evidence* instead. With a positive pseudocount and non-empty
    # synonymous pools there are none.
    # Two-sided: a draw is extreme when it sits at least as far from zero as
    # the observation, on either side. NaN fails the comparison, so the
    # `~isfinite` term is what keeps an undefined draw counted as extreme
    # rather than as evidence.
    reaches = np.abs(null) >= abs(observed)
    n_extreme = int((~np.isfinite(null) | reaches).sum())
    return PermutationTest(
        observed=observed,
        observed_uncorrected=log_rate_ratio(*counted),
        p_value=(1 + n_extreme) / (1 + n_permutations),
        n_permutations=n_permutations,
        n_extreme=n_extreme,
        n_undefined=n_undefined,
        pseudocount=pseudocount,
        null=null,
    )


def tally_pcps(
    pcp_path: str | Path,
    pools_by_candidate: Mapping[str, Sequence[PoolSpec]],
    annotations: pd.DataFrame,
    *,
    clade: str,
    focal_by_candidate: Mapping[str, tuple[int, str]],
    chunk_rows: int,
    max_chunks: int | None = None,
) -> pd.DataFrame:
    """One row per branch per candidate: its pool, and its event counts.

    :func:`tally_pools` sums these away, which is all a point estimate needs.
    The permutation test needs them kept apart, because what it resamples is
    the branch -- with its focal and synonymous counts bound together.

    A branch matching several of a candidate's specs appears once per spec, so
    nested pools do not silently merge. ``focal`` counts nucleotide routes
    that reach the candidate's mutant residue at its site, filtered exactly as
    :func:`aggregate_to_aa_cells` filters them, so summing this frame by pool
    reproduces that function's ``actual``.

    **Every candidate of the clade in one pass**, which is a divergence from
    the source: there this took a single ``focal_site`` and ``focal_mutant``,
    so a clade with nine candidates cost nine streaming reads of a 112 MB
    corpus to collect nine columns of the same branches. The focal keys are
    per candidate and the synonymous count is not, so the inner loop does the
    shared work once.

    Returns:
        ``(candidate, pool, focal, synonymous)``.
    """
    if not pools_by_candidate:
        raise ValueError("no pool specs given; there is nothing to tally.")
    if set(pools_by_candidate) != set(focal_by_candidate):
        raise ValueError(
            f"pools are given for {sorted(pools_by_candidate)} and focal cells "
            f"for {sorted(focal_by_candidate)}. A candidate with pools and no "
            f"focal cell would tally every branch and count no events."
        )
    clade_rows = annotations[annotations["clade"] == clade]
    if clade_rows.empty:
        raise ValueError(f"no annotated rows for clade {clade!r}.")
    site_map = (
        clade_rows.groupby(["codon_site", "codon_position"])["nt_site"]
        .first()
        .to_dict()
    )
    usable = ~clade_rows["exclude"] & ~clade_rows["masked_in_usher"]
    synonymous_keys = set(
        zip(
            *(
                clade_rows[usable & (clade_rows["mut_class"] == "synonymous")][column]
                for column in ("nt_site", "nt_mutation")
            )
        )
    )

    focal_keys: dict[str, set[tuple[int, str, str]]] = {}
    for candidate, (focal_site, focal_mutant) in focal_by_candidate.items():
        if len(focal_mutant) != 1:
            raise ValueError(
                f"{candidate}: focal_mutant {focal_mutant!r} is not a single residue."
            )
        focal_rows = clade_rows[
            usable
            & (clade_rows["mut_class"] == "nonsynonymous")
            & (clade_rows["codon_site"] == focal_site)
            & (clade_rows["mutant_aa"] == focal_mutant)
        ]
        if focal_rows.empty:
            raise ValueError(
                f"no usable nonsynonymous route reaches {focal_mutant} at site "
                f"{focal_site} in clade {clade!r}, so {candidate}'s focal count "
                "would be zero by construction rather than by observation."
            )
        focal_keys[candidate] = {
            (site, mutation[0], mutation[-1])
            for site, mutation in zip(focal_rows["nt_site"], focal_rows["nt_mutation"])
        }

    specs = [spec for pools in pools_by_candidate.values() for spec in pools]
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"pool names must be unique across candidates, got {names}.")
    candidate_of = {
        spec.name: candidate
        for candidate, pools in pools_by_candidate.items()
        for spec in pools
    }

    membership: dict[str, tuple[str, ...]] = {}
    rows: list[tuple[str, str, int, int]] = []
    for chunk_index, chunk in enumerate(
        pd.read_csv(
            pcp_path, usecols=["clade", "parent", "child"], chunksize=chunk_rows
        )
    ):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        chunk = chunk[chunk["clade"] == clade]
        if chunk.empty:
            continue
        for parent, child in zip(chunk["parent"], chunk["child"]):
            pools = membership.get(parent)
            if pools is None:
                pools = tuple(spec.name for spec in specs if spec.matches(parent))
                membership[parent] = pools
            if not pools:
                continue
            wanted = {candidate_of[name] for name in pools}
            synonymous = 0
            focal = dict.fromkeys(wanted, 0)
            for position, (source, target) in enumerate(zip(parent, child)):
                if source == target:
                    continue
                if source not in TALLY_BASES or target not in TALLY_BASES:
                    continue
                genome_site = site_map.get((position // 3 + 1, position % 3 + 1))
                if genome_site is None:
                    continue
                if (genome_site, f"{source}{genome_site}{target}") in synonymous_keys:
                    synonymous += 1
                for candidate in wanted:
                    if (genome_site, source, target) in focal_keys[candidate]:
                        focal[candidate] += 1
            for name in pools:
                candidate = candidate_of[name]
                rows.append((candidate, name, focal[candidate], synonymous))
    return pd.DataFrame(rows, columns=["candidate", "pool", "focal", "synonymous"])
