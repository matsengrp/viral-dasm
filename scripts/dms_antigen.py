"""antiGen as a predictor of the spike DMS: the comparison's published model.

The paper's Results say the DASM beats antiGen, and this is the arm that has to
earn it. antiGen is a transformer trained on the UShER tree's own branches --
the same evolutionary signal the DASM reads -- so unlike BLOSUM62 and the leaf
frequency it is not a floor to subtract but a competitor at the same task.

**What it reports is a probability, not a fitness effect.** ``score_batch``
returns one softmax over the whole (position x amino-acid) map with the
wild-type cell removed, renormalised over the sequence's candidate
substitutions: a whole-sequence distribution over mutation events, not an
independent per-position categorical. So this column is ``log p``, while the
DASM's is ``log f``. Whatever reports the two together has to say which is
which; nothing here converts between them, because no conversion exists that
does not assume a mutation model.

**The background is the construct, not the clade founder.** The other
predictors score against the clade founder, which is the frame they were
trained in. antiGen is fed the WH1 projection of the sequence each assay
actually expressed, carrying the deletions the lineage really has -- BA.1's
founder presents ``H69 V70``, ``G142 V143 Y144`` and ``N211`` where the virus
has none. The two declarations differ at 8 / 6 / 11 columns and ``load_dms``
has already dropped every one, so this moves no measured cell: it moves only
what the transformer conditions on. Nothing in the predictor contract can tell
the two apart, which is why the gap count is asserted here and the sequence's
sha256 is what the manifest declares.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install.
import lib
from lib import antigen_model, assert_dasm2_is_this_repo, dms_dataset, dms_homologs

import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import dms
import dms_models
import numpy as np
import pandas as pd

from dasm2.sequence.constants import AA_STR_SORTED, AA_STR_TO_IDX

#: This predictor's id, as `results/panels/dms/` carries it and
#: `scripts/panel_spec.py` lists in :data:`panel_spec.DMS_PREDICTORS`.
ANTIGEN = "antigen"

#: The device the forward pass runs on. **Not a default and not an option.**
#: `antiGen/infer.py` takes CUDA whenever a GPU is visible, and this rule runs
#: beside training jobs that own the cards -- `scripts/gpu_slot.py` hands those
#: out and the figure rules hold no slot. The whole pass is a few seconds of CPU
#: work on three 1,273-residue queries, so there is nothing to win by asking.
DEVICE = "cpu"

#: Zero the stop and gap columns and renormalise, as `antiGen/infer.py`'s
#: ``--drop-gap-stop`` does. **A renormalisation, not a filter**: it cannot
#: change which cells are scored, because no measured cell asks for ``*`` or
#: ``-``, but it shifts every ``log p`` by a constant per query (+0.040 /
#: +0.026 / +0.041 nats on BA.1 / XBB.1.5 / KP.3.1.1). A correlation cannot see
#: that; a reported ``log p`` can, so the flag is pinned rather than left to a
#: caller. On is the right side of the choice: a DMS measures substitutions, and
#: leaving probability mass on deletions and stops would put this column on a
#: different normalisation from the quantity it is compared against.
DROP_GAP_STOP = True


def antigen_columns(alphabet: str) -> tuple[int, ...]:
    """For each residue of ``AA_STR_SORTED``, its column in antiGen's alphabet.

    Spelled out rather than assumed to be the identity. It *is* the identity
    today -- antiGen's alphabet is ``ACDEFGHIKLMNPQRSTVWY*-`` and this
    repository's twenty are its first twenty -- and that is exactly why it is
    written down: a reordered alphabet on either side would transpose every
    lookup while passing every row count, key check and hash the contract
    makes, and arrive as a correlation of the wrong magnitude.

    Raises:
        ValueError: If the model's alphabet omits a standard residue, which
            would leave a measured cell unscorable rather than mis-scored.
    """
    index = {residue: column for column, residue in enumerate(alphabet)}
    missing = [residue for residue in AA_STR_SORTED if residue not in index]
    if missing:
        raise ValueError(
            f"antiGen's alphabet {alphabet!r} carries no column for {missing}, "
            f"which this assay measures substitutions to"
        )
    return tuple(index[residue] for residue in AA_STR_SORTED)


def read_parents(path: Path) -> dict[str, str]:
    """The query sequence per homolog, from the committed FASTA.

    Its own reader rather than ``founders.py``'s: that one length-checks
    against the gene's codon count and rejects a gap, and these sequences carry
    gaps on purpose -- they are projections of a construct onto WH1 coordinates
    and a deletion is a column, not a missing residue.
    """
    sequences: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                sequences[name] = "".join(chunks)
            name = line[1:].strip()
            chunks = []
        elif line.strip():
            chunks.append(line.strip())
    if name is not None:
        sequences[name] = "".join(chunks)
    return sequences


def substitution_probabilities(checkout: Path, checkpoint: Path, query: str):
    """antiGen's ``(L, E)`` probability map for one query, and its alphabet.

    The checkout goes on ``sys.path`` here rather than at module import: it is a
    `config.yaml` path, so it is not known until :func:`lib.antigen_model` has
    been read, and a module-level import would make every test in this file --
    including the ones that check the alphabet conversion with no model at all
    -- depend on a tree outside the checkout.

    Returns:
        ``(probs, alphabet)`` -- the map as a ``(L, E)`` NumPy array and the
        alphabet its second axis is indexed by, taken from the model rather than
        restated so the conversion is checked against the checkpoint that was
        loaded.
    """
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    from antiGen.infer import score_batch
    from antiGen.models.factory import load_model

    model, _ = load_model(str(checkpoint), device=DEVICE)
    model.eval()
    alphabet = model.config.alphabet.amino_acids
    if len(query) != model.config.seq_len:
        raise ValueError(
            f"the query is {len(query)} residues and this model predicts on a "
            f"fixed frame of {model.config.seq_len}; the committed parents are "
            f"WH1 projections, so a length mismatch means the projection moved"
        )
    batch = score_batch(model, [query], alphabet, DEVICE, DROP_GAP_STOP)
    return batch[0].numpy().astype(np.float64), alphabet


def cell_scores(
    probs: np.ndarray,
    columns: tuple[int, ...],
    sites: np.ndarray,
    mutants: pd.Series,
) -> np.ndarray:
    """``log p`` for each measured cell, looked out of the dense map.

    A per-cell lookup, as both reference predictors are: ``load_dms`` has
    already removed every cell that would ask for an insertion site, a ``-``, a
    ``*`` or a non-standard residue, so indexing the map by
    ``(aa_site - 1, column)`` makes all of those impossible by construction
    rather than by a filter this would have to keep in step with
    `scripts/dms.py`.

    Raises:
        ValueError: If a cell's probability is zero, which ``log`` would turn
            into ``-inf``. ``score_batch`` zeroes the wild-type residue at each
            position, and the ``*`` and ``-`` columns under
            :data:`DROP_GAP_STOP`, and nothing else -- so this fires only if a
            measured cell's mutant residue is the query's own residue there,
            which would mean the parent the model read and the parent the
            contract supplies have come apart.
    """
    rows = sites - 1
    cols = np.array([columns[AA_STR_TO_IDX[aa]] for aa in mutants])
    picked = probs[rows, cols]
    zero = np.flatnonzero(picked == 0.0)
    if zero.size:
        first = int(zero[0])
        raise ValueError(
            f"{zero.size} measured cells carry probability 0, e.g. site "
            f"{int(sites[first])} to {mutants.iloc[first]!r}. `score_batch` "
            f"zeroes the wild-type cell at each position, so this means the "
            f"query's residue there is the mutant the assay measured"
        )
    return np.log(picked)


def antigen_scores(homolog: str) -> tuple[pd.DataFrame, dms_models.Manifest, str]:
    """antiGen's ``log p`` for every cell one homolog's assay measured."""
    spec = antigen_model()
    for label, path, pinned in (
        ("checkpoint", spec.checkpoint, spec.checkpoint_md5),
        ("parents", spec.parents, spec.parents_md5),
    ):
        found = lib.file_md5(path)
        if found != pinned:
            raise ValueError(
                f"antigen.{label} at {path} has md5 {found}, but `config.yaml` "
                f"pins {pinned}. The digest is a gate: a changed file moves "
                f"the antiGen row of the predictor table and nothing else "
                f"would notice."
            )

    parents = read_parents(spec.parents)
    try:
        query = parents[homolog]
    except KeyError:
        raise KeyError(
            f"{spec.parents} carries no record for {homolog!r}; it holds "
            f"{sorted(parents)}"
        ) from None

    cells, _ = dms_models.homolog_cells(homolog)
    probs, alphabet = substitution_probabilities(spec.checkout, spec.checkpoint, query)
    columns = antigen_columns(alphabet)
    scores = cell_scores(
        probs, columns, cells["aa_site"].to_numpy(), cells["mutant_aa"]
    )

    frame = cells.assign(model=ANTIGEN, score=scores)
    manifest = dms_models.Manifest(
        model=ANTIGEN,
        strata=dms.STRATA,
        frame=(
            f"WH1 projection of the construct {homolog}'s assay expressed, "
            f"carrying {query.count('-')} deletion columns, **not** the "
            f"{dms_dataset(homolog).clade} clade founder the other "
            f"predictors score against; scored by antiGen "
            f"{spec.checkpoint.name} (md5 {spec.checkpoint_md5}) at the time "
            f"cutoff {spec.cutoff}, --drop-gap-stop on, on {DEVICE}; log of a "
            f"whole-sequence probability over mutation events, not a selection "
            f"factor"
        ),
        input_sequence_sha256=dms_models.sequence_sha256(query),
    )
    return frame[list(dms_models.PREDICTOR_COLUMNS)], manifest, query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--homolog", required=True, choices=dms_homologs())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    frame, manifest, sequence = antigen_scores(args.homolog)
    dms_models.write_scores(frame, manifest, sequence, args.output)
    print(f"wrote {args.output}: {len(frame)} cells for {args.homolog}")


if __name__ == "__main__":
    main()
