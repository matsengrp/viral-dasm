"""Measure what a gene's censor table actually removes, and write it as JSON.

This is the ``O(corpus)`` half of the censoring accounting, deliberately kept
out of dataset construction. The cheap ``O(table)`` invariants -- the
``nt_index`` range check against the corpus, the both-directions clade
cross-check, the ``num_segments`` guard -- stay in
:class:`CensorMaskedMutations` and are paid on every training start. A full
``(clade, parent, child)`` scan of the corpus is not, and a versioned JSON
artifact serves the guard better than a log line that scrolls past.

Two numbers, because the mechanism has two halves and they are not the same
size.

**The over-masking rate** is the numerator half: the fraction of *observed*
parent->child mutations that censoring drops from the whichmut numerator. It is
the number that accounting needs and never had. Two failure modes it separates:

* a rate of **zero** means censoring cannot change the loss at all -- the
  ``loss_mask`` half never fires, so a censored run and an uncensored one are
  bit-identical. A gene can land here; it is the first thing to check before
  reading anything else in this file.
* a rate that is *large* means censoring is discarding real signal, and the
  fix's cost needs weighing against the bias it removes.

**The censored-candidate rate** is the denominator half: the fraction of the
nine-per-codon candidates that censoring removes from Z. Compare it against the
over-masking rate and nothing else -- both are fractions of what was on offer.
Comparing the raw counts -- on Spike, 94,118,717 candidates against 543
observations -- compares a candidate to an observation and reads as five orders
of magnitude when the rate ratio is 16x: 0.519% of candidates against 0.032% of
observations. The asymmetry is a property of the gene and its table, not a law;
this script is how it is measured rather than assumed.

Why the numerator half is so small is worth stating, because it is close to
tautological: the PCPs are extracted from the same MAT the masking deleted
these mutations from, so a censored mutation is largely absent from the corpus
by construction. Under a null where censoring were independent of observation,
Spike would show ~8,800 censored observations rather than 543. The residual is the
interesting quantity, not the near-zero.

Also reported, because each is a distinct way the table can be wrong while
still producing a plausible run:

* ``censored_cells_declared`` against
  ``censored_cells_matched_by_an_observation`` -- how much of what the table
  declares any PCP ever hits. Counted in ``(site, from, to)`` *cells* rather
  than table rows, because one site-mask row declares twelve of them. Most
  cells are expected to go unmatched: the table names what *could* have been
  masked, not what was observed. This is a magnitude to eyeball, not a failure.
* ``distinct_fully_censored_codons`` and its per-clade companion
  ``fully_censored_codons`` -- codons where every off-diagonal transition is
  censored at all three positions, so the codon's row of
  ``codon_neutral_rates`` goes entirely to zero whatever the parent is. Report
  the union: Spike has **23** such codons, recurring across 37 clades for a
  per-clade sum of 236, and they are the known Omicron NTD/RBD deletion regions
  (aa 17-26, 69, 143-144, 156-157, 212-218, 243-244, 483), not a discovery.
  The table's own reasoning calls these unreachable, but it reasons about
  entries; 1,118 of Spike's 1,892 rows are site masks, and three in one
  codon zero it by construction. A codon here contributes nothing to Z, which is
  why ``WhichmutLoss`` clamps the partition function.
* ``pcps_in_clades_absent_from_roster`` -- corpus a roster gap would leave
  uncensored. Expected to be empty; the training path refuses outright, so this
  reports the size of the problem rather than restating that it exists.

Both derivations of the per-clade counts are independent: this script reads the
PCP corpus, and mat-pcp's ``censor_table`` rule reports its own numbers from the
MAT. Agreement between them is meaningful; disagreement is a version skew.

GISAID restriction: reads GISAID-derived PCPs. The JSON it writes holds only
aggregate counts -- no sequences, no strain names -- but it is still derived and
belongs under the artifact root with group permissions.

Usage::

    python scripts/audit_censor.py --output <artifact root>/audit/S.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first (it puts this repository's root on `sys.path`), then `dasm2`
# before pandas -- which pulls pyarrow in, and Arrow fixes its memory pool at
# first import. `tests/test_import_order.py` enforces both.
from lib import GENE, assert_dasm2_is_this_repo, corpus, mat_pcp_path

import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import argparse
import json
from collections import Counter
from collections.abc import Iterator

import numpy as np
import pandas as pd

from dasm2.io import read_fasta_first_sequence

# `BASES` order is the index convention `CensorSpec` stores, so the lookup table
# is built from it rather than from a second spelling of "ACGT".
from dasm2.sequence.constants import BASES
from dasm2.sequence.masking import (
    CensorSpec,
    load_censor_table,
    parse_masked_nt_sites_file,
)

# ACGT -> 0..3, everything else (N, gaps, ambiguity codes) -> AMBIGUOUS. A
# mutation touching an ambiguous base is not an observation of anything, and
# `loss_mask` already excludes it, so it must not count in the denominator of
# the over-masking rate either.
AMBIGUOUS = np.uint8(255)
_BASE_LUT = np.full(256, AMBIGUOUS, dtype=np.uint8)
for _i, _b in enumerate(BASES):
    _BASE_LUT[ord(_b)] = _i
    _BASE_LUT[ord(_b.lower())] = _i

# PCP rows per chunk. Sized so the largest gene (ORF1a, 13,215 nt) holds well
# under a gigabyte of unpacked uint8 per chunk: 50k x 13,215 x 2 strands is
# ~1.3 GB, which is why this is not larger.
DEFAULT_CHUNK_ROWS = 20_000


def encode_sequences(sequences: pd.Series, expected_length: int) -> np.ndarray:
    """Pack a column of equal-length nucleotide strings into ``(n, L)`` uint8.

    Raises on a length mismatch rather than truncating. A short sequence would
    otherwise silently shift every index after it, and every censored position
    downstream of the ragged row would be compared against the wrong base --
    producing a plausible, wrong over-masking rate.
    """
    joined = "".join(sequences)
    if len(joined) != len(sequences) * expected_length:
        lengths = Counter(len(s) for s in sequences)
        raise ValueError(
            f"expected every sequence to be {expected_length} nt, saw lengths "
            f"{dict(lengths)}. The censor table is indexed by nucleotide "
            "position, so a ragged corpus cannot be audited."
        )
    raw = np.frombuffer(joined.encode("ascii"), dtype=np.uint8)
    return _BASE_LUT[raw].reshape(len(sequences), expected_length)


def censored_lookup(spec: CensorSpec, seq_length: int) -> np.ndarray:
    """A ``(L, 4, 4)`` bool array: is ``from_base -> to_base`` censored at this site?

    Dense rather than a set of tuples so the per-PCP lookup is a single fancy
    index instead of a Python loop over mutations. At S's 3,819 nt this is 61 kB.

    Site masks fill a whole ``(4, 4)`` slice: masking deleted every mutation at
    that site, in every direction. Directional masks set one cell.
    """
    table = np.zeros((seq_length, len(BASES), len(BASES)), dtype=bool)
    site_indices = spec.site_nt_indices.numpy()
    if site_indices.size:
        table[site_indices, :, :] = True
    directional = spec.directional.numpy()
    if directional.size:
        table[directional[:, 0], directional[:, 1], directional[:, 2]] = True
    return table


def fully_censored_codons(table: np.ndarray) -> np.ndarray:
    """Indices of codons whose every single-nt neighbour is censored.

    A codon has nine single-nucleotide neighbours: three positions times three
    alternative bases. Counting them off the ``(L, 4, 4)`` table means asking,
    for each codon, whether all three of its positions are censored *out of the
    parent's actual base* -- but the parent base varies by PCP, so this uses the
    site-mask-style question instead: is every off-diagonal transition censored
    at all three positions? That is exactly the case where the codon's row of
    `codon_neutral_rates` goes entirely to zero regardless of the parent.

    Returns the indices rather than a count because the counts are per-clade and
    the same codon is typically fully censored in many clades at once: summing
    the per-clade counts multiplies one deletion region by its clade count.
    Reporting both the union and the per-clade counts is what keeps the two
    apart -- see ``distinct_fully_censored_codons`` in the emitted JSON.
    """
    off_diagonal = ~np.eye(len(BASES), dtype=bool)
    # (L,) -- is every from->to change censored at this site?
    site_fully = table[:, off_diagonal].all(axis=1)
    n_codons = len(site_fully) // 3
    if n_codons == 0:
        return np.empty(0, dtype=np.int64)
    per_codon = site_fully[: n_codons * 3].reshape(n_codons, 3).all(axis=1)
    return np.flatnonzero(per_codon)


def iter_pcp_chunks(path: Path, chunk_rows: int) -> Iterator[pd.DataFrame]:
    """Stream the aggregate PCP file in chunks, keeping only what is needed.

    Three columns of the seven: the audit does not care about branch lengths,
    node names, or the train/val split. Reading only these is what keeps ORF1a's
    855 MB payload inside a laptop-sized working set.
    """
    yield from pd.read_csv(
        path,
        usecols=["clade", "parent", "child"],
        dtype=str,
        chunksize=chunk_rows,
    )


def audit_gene(gene: str, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> dict:
    """Measure the censor table against the full-depth PCP corpus.

    Every path comes from ``config.yaml``, and from the same keys
    ``scripts/render_config.py`` writes into a training config -- so the audit
    measures the table and mask that runs actually censor with, rather than a
    separately named pair that happens to agree.
    """
    full_depth = corpus("full_depth")
    censor_table = load_censor_table(
        mat_pcp_path("censored_mutations_file"),
        mat_pcp_path("censored_clades_file"),
    )

    pcp_path = full_depth.source_pcp
    # Length is taken from the corpus rather than from the founder FASTA: the
    # founder is a full-genome slice and the PCPs are the gene, and it is the
    # PCPs the censor indices have to line up with.
    first = next(iter_pcp_chunks(pcp_path, 1))
    seq_length = len(first["parent"].iloc[0])

    max_index = max(
        (int(spec.site_nt_indices.max()) if spec.site_nt_indices.numel() else -1)
        for spec in censor_table.values()
    )
    max_index = max(
        max_index,
        max(
            (int(spec.directional[:, 0].max()) if spec.directional.numel() else -1)
            for spec in censor_table.values()
        ),
    )
    if max_index >= seq_length:
        raise ValueError(
            f"{gene}: censor table names nt_index {max_index} but the corpus is "
            f"{seq_length} nt. This table belongs to a different gene."
        )

    lookups = {
        clade: censored_lookup(spec, seq_length) for clade, spec in censor_table.items()
    }
    # Per site, how many of the three alternative bases are censored *out of*
    # each possible parent base. `(L, 4)` int8: index by the parent's actual
    # base to get the number of candidates that leave Z at that site. This is
    # the denominator half, and it needs no observed mutation to fire.
    off_diagonal = ~np.eye(len(BASES), dtype=bool)
    candidate_counts = {
        clade: (table & off_diagonal).sum(axis=2).astype(np.int16)
        for clade, table in lookups.items()
    }

    per_clade: dict[str, dict[str, int]] = {
        clade: {
            "pcps": 0,
            "mutations": 0,
            "censored": 0,
            "censored_candidates": 0,
            "pcps_with_a_censored_candidate": 0,
            "table_entries": spec.num_entries,
        }
        for clade, spec in censor_table.items()
    }
    # Which (site, from, to) cells any observation actually landed on. Compared
    # against the declared cells afterwards to count entries that match nothing.
    hit_cells: dict[str, set[tuple[int, int, int]]] = {
        clade: set() for clade in censor_table
    }
    unknown_clades: Counter[str] = Counter()

    # The global artifact mask, accumulated in this same pass. Its counters sit
    # outside the clade groupby below, for two reasons: the mask is
    # clade-independent, so grouping would be pure overhead, and the groupby skips
    # PCPs whose clade is absent from the roster -- so a roster gap would silently
    # shrink the global mask's denominators too, when the mask applies to every
    # PCP regardless of whether its clade is in the censor table. Nothing else
    # cross-checks these counters, so roster-independence is the whole reason
    # they sit outside the groupby.
    # Nucleotide granularity, because that is what `config.yaml`'s
    # `masked_nt_sites_file` holds and what every training config names. The
    # codon-granularity branch upstream carried is not vendored: it read a file
    # this pipeline does not configure, and auditing a mask no run uses would
    # over-report the footprint while looking like a passing gate.
    global_nt = np.zeros(seq_length, dtype=bool)
    global_nt_indices = parse_masked_nt_sites_file(mat_pcp_path("masked_nt_sites_file"))
    if global_nt_indices and max(global_nt_indices) >= seq_length:
        raise ValueError(
            f"{gene}: global mask names nt index {max(global_nt_indices)} "
            f"but the corpus is {seq_length} nt. This mask file belongs to "
            f"a different gene. Nothing else ties a masked_nt_sites.csv to "
            f"a corpus, so this check is the whole guard."
        )
    global_nt[global_nt_indices] = True
    glob = dict.fromkeys(
        (
            "censored_observations",
            "censored_candidates",
            "pcps_with_a_censored_observation",
            "single_mutation_pcps",
        ),
        0,
    )

    # Parent -> reference reversions, accumulated in the same pass and
    # outside the clade groupby for the same two reasons the global mask is: the
    # rule is clade-independent, so grouping is pure overhead, and the groupby
    # skips PCPs whose clade is absent from the roster -- which would make these
    # numbers disagree with the corpus over exactly the rows a roster gap covers.
    #
    # The reference is `config.yaml`'s `reference_sequence_file`, gene-local and
    # in frame (Spike, 3,819 nt). The comment above about the founder being a
    # full-genome slice is about a different field.
    reference = np.frombuffer(
        read_fasta_first_sequence(mat_pcp_path("reference_sequence_file"))
        .strip()
        .upper()
        .encode(),
        dtype=np.uint8,
    )
    reference_nt = np.full(len(reference), AMBIGUOUS, dtype=np.uint8)
    for idx, base in enumerate(BASES):
        reference_nt[reference == ord(base)] = idx
    if len(reference_nt) != seq_length:
        raise ValueError(
            f"{gene}: reference FASTA is {len(reference_nt)} nt but the corpus "
            f"is {seq_length} nt. Nothing else ties a clade_founders/{gene}/19A.fa "
            f"to a corpus, so this check is the whole guard."
        )
    if int((reference_nt == AMBIGUOUS).sum()):
        raise ValueError(
            f"{gene}: reference FASTA holds "
            f"{int((reference_nt == AMBIGUOUS).sum())} non-ACGT positions. The "
            f"rule needs a concrete destination base everywhere: an ambiguous "
            f"one censors nothing there and reports nothing."
        )
    rev = dict.fromkeys(
        (
            "censored_observations",
            "censored_candidates",
            "pcps_with_a_censored_observation",
            "differing_positions",
        ),
        0,
    )

    # Parent-base counts per (clade, position), for the per-(site, clade)
    # reverting split the readout needs.
    #
    # Derived from the corpus rather than from a per-clade founder, because
    # mat-pcp publishes none: every clade's `founder_sequence` resolves to 19A,
    # the reference itself. The modal parent base is the closest stand-in
    # and is arguably the better one -- it is what the model actually saw, where
    # a founder is what the tree says it should have seen.
    #
    # `(L, 4)` per clade: 56 x 3819 x 4 int32 is 3.4 MB for S, so this is
    # accumulated in the same pass rather than costing a second one.
    parent_base_counts = {
        clade: np.zeros((seq_length, len(BASES)), dtype=np.int32)
        for clade in censor_table
    }

    for chunk in iter_pcp_chunks(pcp_path, chunk_rows):
        all_parents = encode_sequences(chunk["parent"], seq_length)
        all_children = encode_sequences(chunk["child"], seq_length)
        called = (
            (all_parents != AMBIGUOUS)
            & (all_children != AMBIGUOUS)
            & (all_parents != all_children)
        )
        per_pcp_mutations = called.sum(axis=1)
        per_pcp_global = (called & global_nt).sum(axis=1)
        glob["censored_observations"] += int(per_pcp_global.sum())
        glob["pcps_with_a_censored_observation"] += int((per_pcp_global > 0).sum())
        glob["single_mutation_pcps"] += int((per_pcp_mutations == 1).sum())
        # Three alternative bases per known parent nucleotide inside the mask.
        # Counted rather than closed-form `9 * sites * pcps`, because an ambiguous
        # parent base offers no candidates and would inflate the rate.
        glob["censored_candidates"] += 3 * int(
            ((all_parents != AMBIGUOUS) & global_nt).sum()
        )

        # One candidate per differing nucleotide, not three: the rule is
        # directional, so it removes only the destination that restores the
        # reference. Counting three here -- the shape the global mask's line
        # above has -- would overstate the candidate rate by exactly 3x, and
        # 0.6% is as plausible a number as 0.2%.
        parent_differs = (all_parents != AMBIGUOUS) & (all_parents != reference_nt)
        rev["differing_positions"] += int(parent_differs.sum())
        rev["censored_candidates"] += int(parent_differs.sum())
        # An observed reversion: the child carries the reference base at a
        # position where a called substitution happened. `called` already
        # requires parent != child and both known, so parent != reference
        # follows and does not need its own term.
        reverting = called & (all_children == reference_nt)
        per_pcp_reverting = reverting.sum(axis=1)
        rev["censored_observations"] += int(per_pcp_reverting.sum())
        rev["pcps_with_a_censored_observation"] += int((per_pcp_reverting > 0).sum())

        for clade, rows in chunk.groupby("clade", sort=False):
            if clade not in lookups:
                # Counted rather than raised: the training path already
                # cross-checks the clade sets both ways and refuses. Here the
                # useful output is *how much* corpus a roster gap would cover.
                unknown_clades[str(clade)] += len(rows)
                continue
            parents = encode_sequences(rows["parent"], seq_length)
            children = encode_sequences(rows["child"], seq_length)
            differs = (
                (parents != children) & (parents != AMBIGUOUS) & (children != AMBIGUOUS)
            )
            row_idx, col_idx = np.nonzero(differs)
            from_b = parents[row_idx, col_idx]
            to_b = children[row_idx, col_idx]
            censored = lookups[clade][col_idx, from_b, to_b]

            # Denominator half. `parents` may hold AMBIGUOUS (255), which is
            # out of range for the (L, 4) table, so those positions are zeroed
            # out rather than indexed -- an ambiguous parent base has no
            # candidates in Z either way.
            known = parents != AMBIGUOUS
            safe_parents = np.where(known, parents, 0)
            per_site_removed = candidate_counts[str(clade)][
                np.arange(seq_length)[None, :], safe_parents
            ]
            per_site_removed = np.where(known, per_site_removed, 0)
            per_pcp_removed = per_site_removed.sum(axis=1)

            # One bincount per base rather than a (rows, L, 4) one-hot, which
            # would be 20000 x 3819 x 4 bytes per chunk for S.
            counts = parent_base_counts[str(clade)]
            for base_idx in range(len(BASES)):
                counts[:, base_idx] += (parents == base_idx).sum(axis=0)

            stats = per_clade[str(clade)]
            stats["pcps"] += len(rows)
            stats["mutations"] += int(row_idx.size)
            stats["censored"] += int(censored.sum())
            stats["censored_candidates"] += int(per_pcp_removed.sum())
            stats["pcps_with_a_censored_candidate"] += int((per_pcp_removed > 0).sum())
            if censored.any():
                hit = np.stack(
                    [col_idx[censored], from_b[censored], to_b[censored]], axis=1
                )
                hit_cells[str(clade)].update(map(tuple, np.unique(hit, axis=0)))

    total_pcps = sum(s["pcps"] for s in per_clade.values())
    total_mutations = sum(s["mutations"] for s in per_clade.values())
    total_censored = sum(s["censored"] for s in per_clade.values())
    total_candidates = sum(s["censored_candidates"] for s in per_clade.values())
    pcps_touched = sum(s["pcps_with_a_censored_candidate"] for s in per_clade.values())
    # The floor under pcps_touched: a site mask censors candidates in every PCP
    # of its clade regardless of what the parent carries, so every PCP in one of
    # these clades is touched by construction. Anything above this floor is the
    # sequence-dependent contribution of the directional masks.
    site_mask_clade_pcps = sum(
        stats["pcps"]
        for clade, stats in per_clade.items()
        if clade in censor_table and censor_table[clade].site_nt_indices.numel()
    )
    # Off-diagonal only: a base does not mutate to itself, so the four diagonal
    # cells of each site's (4, 4) slice are not censorable events and counting
    # them inflated a site mask from its true 12 cells to 16.
    off_diagonal = ~np.eye(len(BASES), dtype=bool)
    declared_cells = sum(
        int(lookups[clade][:, off_diagonal].sum()) for clade in censor_table
    )
    matched_cells = sum(len(cells) for cells in hit_cells.values())

    # The denominator the removed-candidate count belongs over. Every PCP offers
    # nine single-nt candidates per codon, so this is what makes the denominator
    # half commensurable with the over-masking rate -- which is likewise a
    # fraction, of observations rather than of candidates. Comparing the raw
    # counts instead (94M against 543) compares a candidate to an observation.
    total_offered_candidates = total_pcps * (seq_length // 3) * 9

    codons_by_clade = {
        clade: fully_censored_codons(table) for clade, table in lookups.items()
    }
    distinct_fully_censored = sorted(
        {int(codon) for codons in codons_by_clade.values() for codon in codons}
    )

    return {
        "gene": gene,
        "source_pcp": str(full_depth.source_pcp),
        "source_pcp_manifest": full_depth.source_pcp_manifest,
        "sequence_length_nt": seq_length,
        "clades": len(censor_table),
        "table_entries": sum(spec.num_entries for spec in censor_table.values()),
        "clades_with_no_entries": sum(
            1 for spec in censor_table.values() if spec.is_empty
        ),
        "pcps": total_pcps,
        # --- numerator half: what leaves the loss_mask ---
        "observed_mutations": total_mutations,
        "censored_observations": total_censored,
        # Zero here means baseline and loss_mask_only would be bit-identical.
        "over_masking_rate": (
            total_censored / total_mutations if total_mutations else 0.0
        ),
        # --- denominator half: what leaves Z ---
        "censored_candidates": total_candidates,
        "offered_candidates": total_offered_candidates,
        # THIS is the figure comparable to over_masking_rate. Both are fractions
        # of what was on offer. The two raw counts are not comparable.
        "censored_candidate_rate": (
            total_candidates / total_offered_candidates
            if total_offered_candidates
            else 0.0
        ),
        "pcps_with_a_censored_candidate": pcps_touched,
        # Read this against its floor, below, not on its own. A site mask removes
        # candidates from every PCP of its clade whatever the parent carries, so
        # for a table that is mostly site masks this figure is a restatement of
        # which clades are big -- it is only informative to the extent it exceeds
        # the floor, which is where the directional masks show up.
        "fraction_of_pcps_with_a_censored_candidate": (
            pcps_touched / total_pcps if total_pcps else 0.0
        ),
        "fraction_of_pcps_in_a_clade_carrying_a_site_mask": (
            site_mask_clade_pcps / total_pcps if total_pcps else 0.0
        ),
        "mean_censored_candidates_per_pcp": (
            total_candidates / total_pcps if total_pcps else 0.0
        ),
        # Cells, not table rows: a site mask declares 12 off-diagonal cells at
        # one site, so this is the denominator the "matched nothing" figure
        # belongs over.
        "censored_cells_declared": declared_cells,
        "censored_cells_matched_by_an_observation": matched_cells,
        # Per clade, and separately the union. The union is the honest answer to
        # "how many codons does censoring silence?": the same Omicron deletion
        # region recurs in clade after clade, so the per-clade sum overstates it
        # by roughly the number of clades that inherited the deletion.
        "distinct_fully_censored_codons": distinct_fully_censored,
        "fully_censored_codons": {
            clade: len(codons)
            for clade, codons in codons_by_clade.items()
            if len(codons)
        },
        "pcps_in_clades_absent_from_roster": dict(unknown_clades),
        "per_clade": {
            clade: stats for clade, stats in sorted(per_clade.items()) if stats["pcps"]
        },
        # --- the other mask file ---
        #
        # Reported as its own block rather than folded into the totals above, and
        # `rule train` gates on the two separately. They are different axes: a
        # gene can have an empty per-clade table and a non-empty global mask (E
        # does), so a merged count would say a gene is censorable without saying
        # which mechanism makes it so -- and after merging, no clade is empty, so
        # `clades_with_no_entries` would read as healthy for a table that had gone
        # missing entirely.
        "global": _global_block(
            gene,
            "nucleotide",
            [int(i) for i in np.flatnonzero(global_nt)],
            glob,
            total_pcps,
            total_mutations,
            seq_length,
        ),
        "reversions": _reversion_block(
            gene, rev, total_pcps, total_mutations, seq_length
        ),
        "reverting_codon_sites_by_clade": _reverting_sites_by_clade(
            parent_base_counts, reference_nt, per_clade
        ),
    }


def _reverting_sites_by_clade(
    parent_base_counts: dict[str, np.ndarray],
    reference_nt: np.ndarray,
    per_clade: dict[str, dict[str, int]],
) -> dict[str, list[int]]:
    """Per-clade **1-indexed codon sites** where the reversion rule fires.

    The unit the readout splits on, stated explicitly because it is easy to
    assume otherwise. A reversion censored set is per-PCP, not per-site: it
    depends on the parent, so a clade-defining site is reverting in the clades
    that carry the derived allele and not in the ones that do not. Per
    `(site, clade)` is therefore the coarsest unit that does not mix the two, and
    it is the one `fitness_eval` can join against, since its correlations are
    already computed per `(gene, clade)`.

    A site counts as reverting for a clade when the clade's **modal parent base**
    at any of the codon's three nucleotides differs from the reference. Modal
    rather than "any parent differs": the latter would take in every position with
    sporadic within-clade variation, which on S is most of the gene, and the class
    would stop distinguishing anything. Modal picks out the clade-defining
    differences, which on S is 73.7% of its reversions.

    The 26.2% that undo within-clade mutations therefore land in the
    *non*-reverting class. That is a known and deliberate impurity, not an
    oversight: those reversions have no clade-level signature to key on, so no
    per-(site, clade) split can separate them. It makes the non-reverting class a
    conservative comparison -- if censoring hurt it, some of that is reversion
    effect leaking in -- which is the right direction for a "must not decrease"
    criterion.

    Clades with no PCPs are omitted rather than emitted empty: an empty list would
    read as "this clade reverts nowhere" when the truth is that nothing was
    measured.
    """
    reverting: dict[str, list[int]] = {}
    for clade, counts in parent_base_counts.items():
        if not per_clade[clade]["pcps"]:
            continue
        seen = counts.sum(axis=1) > 0
        modal = counts.argmax(axis=1)
        differs = seen & (modal != reference_nt)
        # Codon-site granularity, 1-indexed, which is what `CensoredSites.sites`
        # returns and what `merge_and_filter` expects.
        codon_sites = np.unique(np.nonzero(differs)[0] // 3) + 1
        reverting[clade] = [int(site) for site in codon_sites]
    return reverting


def _reversion_block(
    gene: str,
    counts: dict[str, int],
    total_pcps: int,
    total_mutations: int,
    seq_length: int,
) -> dict:
    """The reversion rule's half of the audit.

    Reported over the same two denominators as the other two sources, because the
    whole point of this block is that they disagree. On Spike this rule is the
    *smallest* of the three on candidates (0.212%, against the global mask's
    1.335% and the table's 0.519%) and roughly 45x the largest on observations
    (7.22%, against 0.161% and 0.032%). So it is numerator-dominated where the
    other two are denominator-dominated, which is what makes `rates_only` and
    `loss_mask_only` more informative here than for the other two sources -- the loss
    shift comes from lost numerator terms rather than from Z shrinking.

    No uniform null here, unlike the global mask's block. There is nothing to
    compare against: the global mask asks "are these sites unusually quiet?",
    which needs an expectation, whereas this rule's claim is about a *direction*
    being artifactually amplified and the null for that would be the true
    reversion rate -- which is exactly what is unmeasurable. What this block
    reports is the size of what censoring removes, not evidence that it should be.

    Deliberately does **not** decompose reversions by what they undo (a
    clade-defining difference against a within-clade mutation). That split
    needs per-clade founder sequences, and mat-pcp publishes none: every clade's
    `founder_sequence` resolves to 19A, the reference itself.
    """
    offered = total_pcps * (seq_length // 3) * 9
    observed = counts["censored_observations"]
    return {
        "gene": gene,
        "censored_observations": observed,
        "over_masking_rate": observed / total_mutations if total_mutations else 0.0,
        "censored_candidates": counts["censored_candidates"],
        "censored_candidate_rate": (
            counts["censored_candidates"] / offered if offered else 0.0
        ),
        # Measured equal to `censored_observations` on Spike. Not one PCP of its
        # 1,582,521 carries two reverting substitutions, though 99,055 carry two
        # mutations and 6,624 carry three; independence at a 7.22% per-mutation
        # reversion rate would predict roughly 550.
        #
        # **This is an upstream filter, not a property of the tree.** mat-pcp
        # caps reversions to the reference at one per branch, and PCP branch QC
        # reuses that setting, so a branch carrying two is dropped before it can
        # become a PCP. It matters which of the two it is: a change to mat-pcp's
        # configuration -- not a change to the tree -- is what would move this
        # number.
        #
        # Reported anyway, and worth keeping: if these two ever diverge, either
        # that cap was raised or the reconstruction changed.
        "pcps_with_a_censored_observation": counts["pcps_with_a_censored_observation"],
        # Equal to `censored_candidates` by construction -- one candidate per
        # differing nucleotide -- and reported anyway, because that identity is
        # the definition of the rule and a divergence would mean the candidate
        # count had picked up the global mask's 3x.
        "parent_differs_from_reference_positions": counts["differing_positions"],
        "mean_censored_candidates_per_pcp": (
            counts["censored_candidates"] / total_pcps if total_pcps else 0.0
        ),
        "offered_candidates": offered,
        "observations": total_mutations,
    }


def _global_block(
    gene: str,
    granularity: str,
    masked_nt: list[int],
    counts: dict[str, int],
    total_pcps: int,
    total_mutations: int,
    seq_length: int,
) -> dict:
    """The global mask's half of the audit, with both of its nulls.

    Takes the **applied** mask as 0-indexed nucleotide positions rather than a
    codon list, because the two granularities disagree on the one number this
    block is mostly about. `3 * len(codons)` is the codon file's footprint and
    over-states the nucleotide file's, which would put that factor straight
    into `fraction_of_gene_masked` and both nulls. Counting the mask that
    `audit_gene` actually built is correct at either granularity and needs no
    branch here.

    Two nulls, and the gap between them is the finding. The raw uniform-per-nt
    null is computed on the *post*-filter corpus, so it measures mat-pcp's
    `curate_pcps_for_dasm.py` drop filter rather than the sites: that filter
    removed every PCP whose only mutation was at a masked site, cutting the
    numerator and leaving the denominator. Correcting it by the fraction of
    mutations sitting in multi-mutation PCPs -- the only ones whose masked-site
    mutations could have survived -- takes S from 0.12x to 0.93x, i.e.
    indistinguishable from no effect.

    So the raw ratio is not evidence about these sites' biology. What it *is* is
    what the model sees: offered candidate mass is flat across sites (nine
    candidates per codon in every PCP), so observations per unit offered mass
    reduces to observations per site, and there the deficit is real and uncorrected
    -- which is the bias censoring removes.
    """
    footprint = len(masked_nt)
    # Codons touched, because censoring acts on a codon: one masked nucleotide makes its whole codon
    # unreadable as an amino-acid substitution. Derived rather than passed, so it
    # cannot disagree with the footprint above.
    codon_sites = sorted({index // 3 for index in masked_nt})
    offered = total_pcps * (seq_length // 3) * 9
    uniform_null = total_mutations * footprint / seq_length if seq_length else 0.0
    # Over *mutations*, not PCPs: a single-mutation PCP contributes exactly one
    # mutation, so mutations in multi-mutation PCPs are the difference. Using the
    # PCP fraction instead understates the expectation by about a third.
    in_multi = total_mutations - counts["single_mutation_pcps"]
    frac_in_multi = in_multi / total_mutations if total_mutations else 0.0
    filtered_null = uniform_null * frac_in_multi
    observed = counts["censored_observations"]
    return {
        "gene": gene,
        "granularity": granularity,
        "masked_nt": footprint,
        "codon_sites": len(codon_sites),
        "codon_sites_1indexed": [site + 1 for site in codon_sites],
        "footprint_nt": footprint,
        # What the codon file would have masked, for the same sites. Recorded so
        # the granularity change is visible in the audit rather than only in the
        # On Spike this is 51 against `masked_nt`'s 18.
        "footprint_nt_at_codon_granularity": 3 * len(codon_sites),
        "fraction_of_gene_masked": footprint / seq_length if seq_length else 0.0,
        # Same two rates as the per-clade half, over the same denominators, so the
        # two mechanisms are directly comparable.
        "censored_observations": observed,
        "over_masking_rate": observed / total_mutations if total_mutations else 0.0,
        "censored_candidates": counts["censored_candidates"],
        "censored_candidate_rate": (
            counts["censored_candidates"] / offered if offered else 0.0
        ),
        "pcps_with_a_censored_observation": counts["pcps_with_a_censored_observation"],
        # The attribution. The drop filter can only remove a masked mutation from a
        # PCP that had no other one, so if it is what emptied these sites then
        # every surviving PCP here is multi-mutation. On Spike all 2,734 are.
        "single_mutation_pcps": counts["single_mutation_pcps"],
        "fraction_of_mutations_in_multi_mutation_pcps": frac_in_multi,
        "expected_observations_uniform_null": uniform_null,
        "observed_over_expected": observed / uniform_null if uniform_null else None,
        "expected_observations_after_drop_filter": filtered_null,
        "observed_over_expected_after_drop_filter": (
            observed / filtered_null if filtered_null else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene", default=GENE, choices=[GENE])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=DEFAULT_CHUNK_ROWS,
        help="PCP rows read at a time; lower it if the largest genes swap.",
    )
    args = parser.parse_args()

    result = audit_gene(args.gene, args.chunk_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(
        f"{args.gene}: {result['pcps']:,} PCPs, {result['clades']} clades, "
        f"{result['table_entries']:,} table entries"
    )
    print(
        f"  numerator:   {result['censored_observations']:,} of "
        f"{result['observed_mutations']:,} observed mutations censored "
        f"({result['over_masking_rate']:.4%})"
    )
    print(
        f"  denominator: {result['censored_candidates']:,} of "
        f"{result['offered_candidates']:,} candidates removed from Z "
        f"({result['censored_candidate_rate']:.4%})"
    )
    ratio = (
        result["censored_candidate_rate"] / result["over_masking_rate"]
        if result["over_masking_rate"]
        else float("inf")
    )
    print(
        f"  the two halves, as comparable rates: denominator/numerator = {ratio:.1f}x"
    )
    print(
        f"  PCPs touched: {result['fraction_of_pcps_with_a_censored_candidate']:.2%} "
        f"against a site-mask floor of "
        f"{result['fraction_of_pcps_in_a_clade_carrying_a_site_mask']:.2%} "
        f"({result['mean_censored_candidates_per_pcp']:.2f} candidates per PCP)"
    )
    print(
        f"  fully-censored codons: {len(result['distinct_fully_censored_codons'])} "
        f"distinct, {sum(result['fully_censored_codons'].values())} summed over "
        f"{len(result['fully_censored_codons'])} clades"
    )
    if result["censored_observations"] == 0:
        print(
            f"  WARNING: {args.gene} censors no observation at all. The "
            "loss_mask half of the mechanism never fires here, so the "
            "baseline and loss_mask_only arms would be bit-identical, as "
            "would rates_only and both. This gene cannot serve as a pilot.",
            file=sys.stderr,
        )
    if result["censored_candidates"] == 0:
        print(
            f"  WARNING: {args.gene} removes nothing from Z. All four arms "
            "would be bit-identical.",
            file=sys.stderr,
        )
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
