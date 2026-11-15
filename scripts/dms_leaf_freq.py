"""Per-site residue counts over a clade's own tree leaves.

Panel C asks how much of a deep mutational scan a model explains that something
cheaper does not. One cheap answer is the clade's standing variation: a
substitution the tree has never shown in this clade is probably costly, and one
carried by a tenth of its leaves is probably not. This module measures that --
every leaf's spike reconstructed from its own root-to-tip mutation path, tallied
per site, and read out as a log frequency ratio against the clade founder's
residue.

**It is a standing-frequency measure, not a count of evidence.** 124,359 leaves
are not 124,359 independent observations: one old substitution with 50,000
descendants outweighs fifty independent ones. The tally weights a substitution by
its descendant count rather than by the number of times it arose, which is the
right quantity for "how common is this residue in the sampled population" and is
deliberately not mat-pcp's per-branch event count.

This module builds and reads the tally. `scripts/dms_leaf_scores.py` turns it
into a predictor.

**Masked sites are reconstructed wrong, and it does not reach a scored cell.**
mat-pcp's branch-mutation extraction drops masked nucleotide sites while the
founder FASTA does not, so the reconstruction keeps 19A's base wherever a clade
founder differs at one. A codon all of whose nucleotides are masked is never
visited at all; a codon with one masked nucleotide and another that is mutated
*is* visited, and is rebuilt with 19A's base at the masked position -- a wrong
residue either way. ``dms.codon_sites_of`` puts the whole codon in
``globally_masked_sites`` and ``load_dms`` drops every such site, so none of them
is among the cells a score is written for, and the tests assert that rather than
assume it. The rows stay in the written tally, which is why its docstring says
what they are.

GISAID-derived input. The per-site tally this writes is a derived per-site slice
of the tree, so it goes under the artifact root and is not committed; see
``README.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install.
from lib import GENE, assert_dasm2_is_this_repo, mat_pcp_path

import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import dms
import numpy as np
import pandas as pd
import panel_spec
from founders import founder_aa_sequence, founder_nt_sequence

from dasm2.neutral.scov2_rates import gene_nt_span
from dasm2.sequence.constants import AA_STR_SORTED, AA_STR_TO_IDX
from dasm2.sequence.conversion import translate_codon, translate_sequence

#: The clade whose founder every path's mutations are expressed against. The
#: paths are rooted above the clade founders, and 19A's is the state they start
#: from -- which `tests/test_dms_leaf_freq.py` checks by requiring every
#: mutation's own reference base to equal the sequence's current base there,
#: rather than by asserting anything about the tree's topology.
PATH_ROOT_CLADE = "19A"

#: The count table's site column. The twenty residue columns are
#: ``AA_STR_SORTED`` itself, so a reader of the CSV gets the same column order as
#: every selection-factor row in this pipeline.
SITE_COLUMN = "aa_site"


def sample_paths_path(clade: str) -> Path:
    """One clade's per-leaf mutation paths.

    Resolved through `scripts/panel_spec.py`, which is where
    `workflow/figures.smk` declares the rule's input from. The suffix is spelled
    once, there: a second spelling here would be exactly the drift that module
    exists to prevent.
    """
    return panel_spec.sample_paths_path(mat_pcp_path("sample_paths_dir"), clade)


def parse_mutation(token: str) -> tuple[str, int, str]:
    """``C14408T`` as ``("C", 14408, "T")``.

    Raises:
        ValueError: If the token is not a reference base, a 1-based genome site
            and a mutant base. A silently unparsed token is a dropped mutation,
            which is a leaf reconstructed as something it is not.
    """
    if len(token) < 3 or not token[1:-1].isdigit():
        raise ValueError(f"{token!r} is not a nucleotide mutation")
    return token[0], int(token[1:-1]), token[-1]


def iter_leaf_mutations(line: str):
    """Every mutation on one leaf's path, in path order.

    The file gives ``node_id:MUT[,MUT...]`` tokens separated by spaces, root
    first. Order matters and is preserved: a leaf's base at a site is set by the
    last mutation on its path, so a reversion has to be applied after the
    mutation it reverses.
    """
    _, _, rest = line.rstrip("\n").partition("\t")
    for token in rest.split():
        _, _, mutations = token.partition(":")
        for mutation in mutations.split(","):
            if mutation:
                yield parse_mutation(mutation)


class LeafTally:
    """Per-site residue counts over a clade's leaves, as deviations.

    Counting all 1,273 sites for all of a clade's leaves is tens of millions of
    increments for a table almost all of whose mass sits on one residue per site.
    So only the sites a leaf actually differs at are tallied, and the reference's
    own count is the leaf total minus what deviated and what was unreadable. That
    is an optimisation, not a different quantity: :meth:`frame` reconstructs the
    full table.

    **The reference is 19A throughout, not the clade founder.** Both the codons
    built from nucleotides and the untouched ones a leaf is assumed to carry have
    to come from the same sequence, and the reconstruction's starting state is
    19A's. Mixing the two -- building touched codons from 19A and assuming the
    clade founder elsewhere -- disagrees at exactly the masked sites, where no
    path carries a mutation and so no codon is ever visited, and it reports the
    founder's residue at a site where nothing was observed. The clade founder
    enters at scoring time instead, as the residue a substitution is scored away
    from.
    """

    def __init__(self, clade: str, gene: str = GENE) -> None:
        self.clade = clade
        self.gene = gene
        self.root_nt = founder_nt_sequence(PATH_ROOT_CLADE, gene)
        self.root_aa = translate_sequence(self.root_nt)
        self.span_start = gene_nt_span(gene)[0]
        self.n_sites = len(self.root_aa)
        self.deviations = np.zeros((self.n_sites, len(AA_STR_SORTED)), dtype=np.int64)
        #: Leaves whose codon at a site translates outside ``AA_STR_SORTED``.
        #: Dropped from that site rather than counted as a residue, so the site's
        #: tally sums to fewer than the leaf count.
        self.unreadable = np.zeros(self.n_sites, dtype=np.int64)
        self.n_leaves = 0
        #: Mutations whose own reference base disagreed with the sequence's base
        #: at that offset. Expected to be zero; counted rather than raised on so
        #: one bad row does not hide how many there are.
        self.reference_mismatches = 0
        self.mismatch_examples: list[str] = []
        #: Mutations outside the gene's span, which are simply not this gene's.
        self.off_gene = 0

    def add_leaf(self, line: str) -> None:
        """Apply one leaf's path and tally where it ends up."""
        applied: dict[int, str] = {}
        for reference, site, mutant in iter_leaf_mutations(line):
            offset = site - self.span_start
            if offset < 0 or offset >= len(self.root_nt):
                self.off_gene += 1
                continue
            current = applied.get(offset, self.root_nt[offset])
            if current != reference:
                self.reference_mismatches += 1
                if len(self.mismatch_examples) < 5:
                    self.mismatch_examples.append(
                        f"genome {site}: token says {reference} over {current}"
                    )
            applied[offset] = mutant

        self.n_leaves += 1
        for codon_site in {offset // 3 for offset in applied}:
            start = 3 * codon_site
            codon = "".join(
                applied.get(start + k, self.root_nt[start + k]) for k in range(3)
            )
            residue = translate_codon(codon)
            if residue == self.root_aa[codon_site]:
                continue
            if residue not in AA_STR_TO_IDX:
                self.unreadable[codon_site] += 1
                continue
            self.deviations[codon_site, AA_STR_TO_IDX[residue]] += 1

    def frame(self) -> pd.DataFrame:
        """The full 20 x L tally, as ``aa_site`` plus one column per residue."""
        counts = self.deviations.copy()
        deviated = counts.sum(axis=1) + self.unreadable
        for site, residue in enumerate(self.root_aa):
            counts[site, AA_STR_TO_IDX[residue]] += self.n_leaves - deviated[site]
        # No runtime check here. Every site's residues sum to the leaf count less
        # its unreadable codons -- but `counts` is *built* to satisfy that, so
        # asserting it is algebra rather than a test of anything, exactly as a
        # negative-count guard would be: each leaf adds at most one increment per
        # site because the codon sites it touches are a set. What can be checked
        # is the optimisation itself, against a tally that counts every site of
        # every leaf, and `tests/test_dms_leaf_freq.py` does that on a sample.
        table = pd.DataFrame(counts, columns=list(AA_STR_SORTED))
        table.insert(0, SITE_COLUMN, np.arange(1, self.n_sites + 1))
        return table


def build_tally(clade: str, gene: str = GENE) -> LeafTally:
    """Read one clade's leaf paths and tally every leaf's spike."""
    tally = LeafTally(clade, gene)
    with sample_paths_path(clade).open() as handle:
        for line in handle:
            if line.strip():
                tally.add_leaf(line)
    if tally.reference_mismatches:
        raise ValueError(
            f"{clade}: {tally.reference_mismatches} mutations disagree with the "
            f"sequence's base at their own offset, e.g. "
            f"{tally.mismatch_examples}. Either the paths are not expressed "
            f"against {PATH_ROOT_CLADE}'s founder or the gene offset is wrong; "
            f"every residue tallied past the first is then unreliable."
        )
    if not tally.n_leaves:
        raise ValueError(
            f"{sample_paths_path(clade)} holds no leaves. An empty tally scores "
            f"every substitution at log(1/1) = 0, which correlates with nothing "
            f"and raises nothing."
        )
    return tally


def read_counts(path: Path) -> pd.DataFrame:
    """A written tally, indexed by ``aa_site``."""
    table = pd.read_csv(path)
    missing = [column for column in AA_STR_SORTED if column not in table.columns]
    if missing or SITE_COLUMN not in table.columns:
        raise ValueError(
            f"{path}: expected {SITE_COLUMN} and the 20 residues of AA_STR_SORTED; "
            f"missing {missing}"
        )
    # :func:`counts_at` indexes this positionally, so the rows must be exactly
    # 1..L in order. A truncated, reordered or hand-edited tally would otherwise
    # return some other site's counts, or raise an IndexError much later.
    expected = np.arange(1, len(table) + 1)
    if not (table[SITE_COLUMN].to_numpy() == expected).all():
        raise ValueError(
            f"{path}: {SITE_COLUMN} is not 1..{len(table)} in order, which is what "
            f"the per-cell lookup indexes by position"
        )
    # Projected, not just checked: `counts_at` indexes both axes positionally, so
    # a CSV carrying the right headers in the wrong order -- or an extra column --
    # would return some other residue's counts without raising.
    return table[[SITE_COLUMN, *AA_STR_SORTED]].set_index(SITE_COLUMN)


def counts_at(counts: pd.DataFrame, sites, residues):
    """The tally's value for each ``(site, residue)`` pair, as an array.

    One helper rather than the same fancy-index in each caller: the scorer needs
    it for the mutant residue and for the parent, and `scripts/dms_summary.py`
    needs it for the mutant alone.
    """
    return counts.to_numpy()[
        np.asarray(sites) - 1,
        [AA_STR_TO_IDX[residue] for residue in residues],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clade", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    tally = build_tally(args.clade)
    table = tally.frame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    # What the table says against the clade's own founder, which is what a
    # reader checks it by: the modal residue should be the founder's everywhere
    # the reconstruction can see, and the sites where it is not are the masked
    # ones, reported rather than asserted away.
    founder = founder_aa_sequence(args.clade, GENE)
    modal = table[list(AA_STR_SORTED)].to_numpy().argmax(axis=1)
    disagree = sorted(
        site + 1
        for site in range(tally.n_sites)
        if AA_STR_SORTED[modal[site]] != founder[site]
    )
    masked = dms.globally_masked_sites()
    print(
        f"wrote {args.output}: {tally.n_leaves} leaves, {tally.n_sites} sites, "
        f"{int(tally.deviations.sum())} leaf-site deviations from "
        f"{PATH_ROOT_CLADE}, {int(tally.unreadable.sum())} unreadable codons, "
        f"{tally.off_gene} off-gene mutations"
    )
    print(
        f"  modal residue equals the {args.clade} founder at "
        f"{tally.n_sites - len(disagree)} of {tally.n_sites} sites; "
        f"disagreements {disagree}, of which "
        f"{sorted(set(disagree) & masked)} are globally masked"
    )


if __name__ == "__main__":
    main()
