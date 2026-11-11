"""The contract a non-DASM predictor scores the spike DMS through, and BLOSUM62.

`fig:dms`'s panel C sets the DASM against published models. Each of those reads
a different parent -- the DASM arms read the clade founder, which is the UShER
tree's WH1-aligned frame they were trained in, while antiGen and CoVFit must
read the construct the assay was measured on -- so a predictor declares the
sequence it scored against and this module checks the declaration against the
scores before either reaches a table. Nothing here knows which model it is
handling, which is what lets a later arm land as a predictor rather than as an
edit to this file.

The join target is ``scripts/dms.py``'s melted frame, not a dense 20 x L table:
``melt_dms`` has already dropped the cells the assay did not measure, so the
invariant a join preserves here is that the row count does not move and no key
repeats, rather than a fixed row count.

What is deliberately absent is the scored-parent/founder difference check that
``dasm2-experiments`` #1913 carries. Both predictors here read the clade founder
itself, and ``load_dms`` has already dropped every founder-mismatched cell, so
that machinery would have no live case and only synthetic tests. It arrives with
the first predictor that reads a different parent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install.
from lib import GENE, assert_dasm2_is_this_repo, dms_dataset, dms_homologs

import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import dms
import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
from founders import founder_aa_sequence

from dasm2.neutral.scov2_rates import GENE_NT_LENGTH
from dasm2.sequence.constants import AA_STR_SORTED

#: One predictor's scores, in column order. ``parent_aa`` is the residue the
#: predictor actually scored against, spelled as `selection_factors.parquet` and
#: ``score_haplotypes.OUTPUT_COLUMNS`` spell it rather than as a third name for
#: the same thing.
PREDICTOR_COLUMNS = (
    "homolog",
    "aa_site",
    "parent_aa",
    "mutant_aa",
    "model",
    "score",
)

#: The join key, as `scripts/dms.py::melt_dms` and `selection_factors.parquet`
#: both spell it, so neither side of the join renames anything.
JOIN_KEY = ("aa_site", "mutant_aa")

#: Spike's codon count, from the gene's own CDS blocks rather than a literal.
#: `scripts/founders.py` length-checks every founder against the same number.
GENE_AA_LENGTH = GENE_NT_LENGTH[GENE] // 3

#: The twenty standard residues, from the repository's own alphabet.
STANDARD_AA = frozenset(AA_STR_SORTED)

#: The predictor ids this module can score. A rule's ``{model}`` wildcard is
#: constrained to these, so a typo fails at DAG construction rather than
#: writing a table named for a predictor that does not exist.
BLOSUM62 = "blosum62"
LEAF_LOGFREQ = "leaf_logfreq"
PREDICTORS: tuple[str, ...] = (BLOSUM62, LEAF_LOGFREQ)


@dataclass(frozen=True)
class Manifest:
    """What a predictor answers before its scores are allowed into the table.

    ``strata`` is a declaration of what the predictor can speak on, and a subset
    is an *absence* downstream rather than a zero: a predictor that scores only
    single-nucleotide-reachable cells must have its other rows missing, not
    present with ``n = 0`` and a NaN correlation.

    ``input_sequence_sha256`` is the hash and not the sequence because nothing
    downstream of the predictor has the sequence. :func:`write_scores` is the one
    point where it exists, and so the one point where the claim is checkable.
    """

    model: str
    strata: tuple[str, ...]
    frame: str
    input_sequence_sha256: str

    def as_dict(self) -> dict[str, object]:
        """A plain dict for the Parquet footer.

        A dataclass in ``attrs`` fails at write time, which is the same
        constraint `scripts/prepare_viral_parquet.py` records for its era map.
        """
        row = asdict(self)
        row["strata"] = list(self.strata)
        return row

    @classmethod
    def from_dict(cls, raw: dict) -> Manifest:
        missing = {f.name for f in fields(cls)} - set(raw)
        if missing:
            raise ValueError(f"manifest is missing {sorted(missing)}")
        return cls(
            model=str(raw["model"]),
            strata=tuple(raw["strata"]),
            frame=str(raw["frame"]),
            input_sequence_sha256=str(raw["input_sequence_sha256"]),
        )


def sequence_sha256(sequence: str) -> str:
    """The identity a predictor declares for the sequence it read."""
    return hashlib.sha256(sequence.encode()).hexdigest()


def blosum62_log_odds() -> np.ndarray:
    """BLOSUM62's 20 x 20 log-odds, rows and columns in ``AA_STR_SORTED``.

    Biopython's table also carries the ambiguity codes and the stop; this keeps
    the twenty standard residues, in the order every selection-factor row in
    this pipeline is laid out in.

    Taken from ``dasm2-experiments`` ``dasm2/analysis/pam_baseline.py``. It is
    here and not in ``dasm2/`` on purpose: that tree is a one-way fork whose
    value is that a diff against its source commit is meaningful, so a module at
    a name no upstream commit uses does not belong in it -- and ``ruff`` excludes
    it, so a file there would go unchecked.
    """
    blosum = substitution_matrices.load("BLOSUM62")
    return np.array(
        [[blosum[a, b] for b in AA_STR_SORTED] for a in AA_STR_SORTED],
        dtype=np.float64,
    )


def homolog_cells(homolog: str) -> pd.DataFrame:
    """One homolog's measured cells, with the two columns the join needs added.

    ``melt_dms`` returns ``aa_site``, ``mutant_aa``, ``dms_cell_entry`` and
    ``stratum`` and nothing else, while a predictor needs the residue it is
    scoring away from and the join needs to know which homolog it is holding.
    Both are added here rather than in ``melt_dms``: #26's panels read its four
    columns and ``tests/test_dms.py`` pins its row count and stratum set against
    them.
    """
    dataset = dms_dataset(homolog)
    matrix, _ = dms.load_dms(homolog)
    cells = dms.melt_dms(matrix, dataset.clade)
    founder = founder_aa_sequence(dataset.clade, GENE)
    cells = cells.copy()
    cells["homolog"] = homolog
    cells["parent_aa"] = [founder[site - 1] for site in cells["aa_site"]]
    return cells


def validate_scores(frame: pd.DataFrame, manifest: Manifest) -> None:
    """Every invariant the contract can check without the sequence.

    Raises:
        ValueError: On the first invariant violated, naming what was expected
            and what was found.
    """
    missing = set(PREDICTOR_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{manifest.model}: scores are missing {sorted(missing)}")

    unknown_strata = set(manifest.strata) - set(dms.STRATA)
    if unknown_strata:
        raise ValueError(
            f"{manifest.model}: declares unknown strata {sorted(unknown_strata)}; "
            f"known strata are {list(dms.STRATA)}"
        )
    if not manifest.strata:
        raise ValueError(f"{manifest.model}: declares no strata, so it says nothing")

    models = set(frame["model"].unique())
    if models != {manifest.model}:
        raise ValueError(
            f"{manifest.model}: the model column holds {sorted(models)}, which "
            f"disagrees with the manifest"
        )

    for column in ("parent_aa", "mutant_aa"):
        bad = sorted(set(frame[column].unique()) - STANDARD_AA)
        if bad:
            raise ValueError(
                f"{manifest.model}: {column} holds {bad}; the contract carries "
                f"amino-acid substitutions only, so no stop and no deletion"
            )

    sites = frame["aa_site"]
    off = sites[(sites < 1) | (sites > GENE_AA_LENGTH)]
    if len(off):
        raise ValueError(
            f"{manifest.model}: {len(off)} sites outside 1..{GENE_AA_LENGTH}; "
            f"sites are 1-indexed into the clade founder, as `melt_dms` emits them"
        )

    duplicated = frame.duplicated(subset=["homolog", *JOIN_KEY])
    if duplicated.any():
        example = frame.loc[duplicated, ["homolog", *JOIN_KEY]].iloc[0].to_dict()
        raise ValueError(
            f"{manifest.model}: {int(duplicated.sum())} duplicate "
            f"(homolog, aa_site, mutant_aa) keys, e.g. {example}"
        )

    unknown_homologs = sorted(set(frame["homolog"].unique()) - set(dms_homologs()))
    if unknown_homologs:
        raise ValueError(
            f"{manifest.model}: unknown homologs {unknown_homologs}; "
            f"`config.yaml` names {dms_homologs()}"
        )


def write_scores(
    frame: pd.DataFrame, manifest: Manifest, sequence: str, path: Path
) -> None:
    """Validate one predictor's scores against the sequence it read, then write.

    The sequence is checked here because nothing downstream has it: the manifest
    carries only its hash. A predictor whose ``parent_aa`` disagrees with the
    sequence it declares has scored against something other than what it says it
    did, and that is the failure this catches.
    """
    if sequence_sha256(sequence) != manifest.input_sequence_sha256:
        raise ValueError(
            f"{manifest.model}: input_sequence_sha256 is not the hash of the "
            f"sequence passed beside it"
        )
    validate_scores(frame, manifest)

    beyond = frame[frame["aa_site"] > len(sequence)]
    if len(beyond):
        raise ValueError(
            f"{manifest.model}: scores {len(beyond)} sites past the end of the "
            f"declared sequence, which is {len(sequence)} residues long, e.g. "
            f"site {int(beyond['aa_site'].iloc[0])}"
        )

    declared = frame["aa_site"].map(lambda site: sequence[site - 1])
    disagree = frame[declared != frame["parent_aa"]]
    if len(disagree):
        row = disagree.iloc[0]
        raise ValueError(
            f"{manifest.model}: parent_aa disagrees with the declared sequence at "
            f"{len(disagree)} rows, e.g. site {row['aa_site']} carries "
            f"{row['parent_aa']!r} but the sequence has "
            f"{declared.loc[disagree.index[0]]!r}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame[list(PREDICTOR_COLUMNS)].reset_index(drop=True)
    out.attrs["manifest"] = json.dumps(manifest.as_dict())
    out.to_parquet(path, index=False)
    # Mirrored for humans. The Parquet footer is what `read_scores` reads; this
    # is a copy, as `prepare_viral_parquet.py` writes its own summary beside the
    # table it put the same facts into.
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest.as_dict(), indent=1, sort_keys=True) + "\n"
    )


def read_scores(path: Path) -> tuple[pd.DataFrame, Manifest]:
    """One predictor's scores and the manifest its footer carries."""
    frame = pd.read_parquet(path)
    raw = frame.attrs.get("manifest")
    if raw is None:
        raise ValueError(f"{path}: no manifest in the Parquet footer")
    return frame, Manifest.from_dict(json.loads(raw))


def join_scores(
    cells: pd.DataFrame, scored: list[tuple[pd.DataFrame, Manifest]]
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Put each predictor's score onto the melted frame as its own column.

    A left join on ``(aa_site, mutant_aa)`` validated ``one_to_one``, which is
    how `scripts/score_run.py` already states this invariant: a cell a predictor
    does not score stays NaN, and a key that repeats on either side raises rather
    than silently multiplying rows.

    Masking needs no code here. ``load_dms`` has already dropped every masked
    site, so a masked mutation is not among the cells at all, and a second mask
    path would be a second thing to keep in step with `scripts/dms.py`.

    Returns:
        ``(merged, manifests)`` -- the frame, and each joined predictor's
        manifest keyed by model name.
    """
    homologs = set(cells["homolog"].unique())
    if len(homologs) != 1:
        raise ValueError(
            f"join_scores takes one homolog's cells at a time; got {sorted(homologs)}"
        )
    homolog = homologs.pop()

    names = [manifest.model for _, manifest in scored]
    if len(set(names)) != len(names):
        raise ValueError(
            f"two predictors claim the same model name in {sorted(names)}; each "
            f"contributes one column, and a manifest is per (model, homolog) "
            f"because the founder it declares differs by clade"
        )

    merged = cells.copy()
    manifests: dict[str, dict] = {}
    for frame, manifest in scored:
        if manifest.model in merged.columns:
            raise ValueError(
                f"{manifest.model}: already a column of the melted frame; a "
                f"predictor name may not collide with one of its own columns"
            )
        validate_scores(frame, manifest)
        part = frame[frame["homolog"] == homolog]
        if part.empty:
            raise ValueError(f"{manifest.model}: no rows for homolog {homolog}")
        merged = merged.merge(
            part[[*JOIN_KEY, "score"]].rename(columns={"score": manifest.model}),
            on=list(JOIN_KEY),
            how="left",
            validate="one_to_one",
        )
        manifests[manifest.model] = manifest.as_dict()

    return merged, manifests


def blosum62_scores(homolog: str) -> tuple[pd.DataFrame, Manifest, str]:
    """BLOSUM62's log-odds for every cell one homolog's assay measured.

    The score is ``B[parent_aa, mutant_aa]`` at the clade founder's residue. It
    is already a log-odds, so nothing is transformed: that is what puts it on the
    same axis as the DASM's log selection factor.
    """
    cells = homolog_cells(homolog)
    founder = founder_aa_sequence(dms_dataset(homolog).clade, GENE)
    matrix = blosum62_log_odds()
    index = {aa: i for i, aa in enumerate(AA_STR_SORTED)}
    scores = [
        matrix[index[parent], index[mutant]]
        for parent, mutant in zip(cells["parent_aa"], cells["mutant_aa"])
    ]
    frame = cells.assign(model=BLOSUM62, score=scores)
    manifest = Manifest(
        model=BLOSUM62,
        strata=dms.STRATA,
        frame=(
            f"clade founder {dms_dataset(homolog).clade} {GENE}, from mat-pcp's "
            f"own clade_founders FASTA; 1-indexed, so no site mapping"
        ),
        input_sequence_sha256=sequence_sha256(founder),
    )
    return frame[list(PREDICTOR_COLUMNS)], manifest, founder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=PREDICTORS)
    parser.add_argument("--homolog", required=True, choices=dms_homologs())
    parser.add_argument("--counts", type=Path, help=f"required for {LEAF_LOGFREQ}")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.model == BLOSUM62:
        frame, manifest, sequence = blosum62_scores(args.homolog)
    else:
        # Imported here, not at module scope: `dms_leaf_freq` imports this
        # module for the contract, and a top-level import back would be a cycle.
        import dms_leaf_freq

        if args.counts is None:
            raise SystemExit(f"--counts is required for {LEAF_LOGFREQ}")
        frame, manifest, sequence = dms_leaf_freq.leaf_logfreq_scores(
            args.homolog, args.counts
        )

    write_scores(frame, manifest, sequence, args.output)
    print(f"wrote {args.output}: {len(frame)} cells for {args.homolog}")


if __name__ == "__main__":
    main()
