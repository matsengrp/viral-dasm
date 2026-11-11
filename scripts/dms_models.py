"""The contract a non-DASM predictor scores the spike DMS through.

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
``dasm2-experiments`` #1913 carries. Both predictors landing with this read the
clade founder itself, and ``load_dms`` has already dropped every
founder-mismatched cell, so that machinery would have no live case and only
synthetic tests. It arrives with the first predictor that reads a different
parent.

**The sequence checks in :func:`write_scores` are for those arms, not for these
two.** A predictor that reads the founder declares the founder, so its hash and
its ``parent_aa`` agree by construction and the checks are tautologies today.
They are four lines, they are the only thing that will catch antiGen or CoVFit
scored against the wrong construct, and they are written now so that arm cannot
land without answering them. Said plainly here rather than left to look like
live validation.
"""

from __future__ import annotations

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
import pyarrow as pa
import pyarrow.parquet as pq
from founders import founder_aa_sequence

from dasm2.data.parquet_backend import read_parquet_file_metadata
from dasm2.neutral.scov2_rates import GENE_NT_LENGTH
from dasm2.sequence.constants import AA_STR_TO_IDX

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


def declared_cell_strata(strata: tuple[str, ...]) -> set[str]:
    """The per-cell strata a manifest's declaration covers.

    ``dms.ALL`` is a union label that no cell carries -- ``melt_dms`` labels every
    cell ``single_nt`` or ``multi_nt`` -- so a predictor declaring it means both,
    and comparing the declaration against a cell's own label needs the expansion.
    Without it the natural way to say "I score everything" is refused by
    :func:`join_scores`.
    """
    expanded = set(strata) - {dms.ALL}
    if dms.ALL in strata:
        expanded |= {dms.SINGLE_NT, dms.MULTI_NT}
    return expanded


#: The footer key one predictor's manifest is stored under. A top-level
#: key-value entry, which is what `dasm2/data/parquet_backend.py` says every
#: metadata reader goes through -- not pandas' ``DataFrame.attrs``, whose
#: survival across a Parquet round trip is a pandas implementation detail and
#: which `read_parquet_file_metadata` would not see.
MANIFEST_KEY = "manifest"

#: The twenty standard residues, from the repository's own alphabet.
STANDARD_AA = frozenset(AA_STR_TO_IDX)


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


def homolog_cells(homolog: str) -> tuple[pd.DataFrame, str]:
    """One homolog's measured cells, with the two columns the join needs added.

    ``melt_dms`` returns ``aa_site``, ``mutant_aa``, ``dms_cell_entry`` and
    ``stratum`` and nothing else, while a predictor needs the residue it is
    scoring away from and the join needs to know which homolog it is holding.
    Both are added here rather than in ``melt_dms``: #26's panels read its four
    columns and ``tests/test_dms.py`` pins its row count and stratum set against
    them.

    Returns:
        ``(cells, founder)``. The founder comes back rather than being read again
        by each caller: every predictor needs it for the sequence its manifest
        declares, and two reads of one FASTA are two chances to disagree.
    """
    dataset = dms_dataset(homolog)
    matrix, _ = dms.load_dms(homolog)
    cells = dms.melt_dms(matrix, dataset.clade)
    founder = founder_aa_sequence(dataset.clade, GENE)
    cells = cells.copy()
    cells["homolog"] = homolog
    cells["parent_aa"] = [founder[site - 1] for site in cells["aa_site"]]
    return cells, founder


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

    nonfinite = int((~np.isfinite(frame["score"])).sum())
    if nonfinite:
        raise ValueError(
            f"{manifest.model}: {nonfinite} of {len(frame)} scores are NaN or "
            f"infinite. `within_clade_metrics` drops those rows, so they would "
            f"shrink this predictor's `n` rather than fail anything."
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
    table = pa.Table.from_pandas(out, preserve_index=False)
    table = table.replace_schema_metadata(
        {
            **(table.schema.metadata or {}),
            MANIFEST_KEY.encode(): json.dumps(manifest.as_dict()).encode(),
        }
    )
    pq.write_table(table, path)
    # Mirrored for humans. The footer entry above is what `read_scores` reads;
    # this is a copy, as `prepare_viral_parquet.py` writes its own summary beside
    # the table it put the same facts into.
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest.as_dict(), indent=1, sort_keys=True) + "\n"
    )


def read_scores(path: Path) -> tuple[pd.DataFrame, Manifest]:
    """One predictor's scores and the manifest its footer carries."""
    raw = read_parquet_file_metadata(path).get(MANIFEST_KEY)
    if raw is None:
        raise ValueError(f"{path}: no {MANIFEST_KEY!r} key in the Parquet footer")
    return pd.read_parquet(path), Manifest.from_dict(json.loads(raw))


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
        # `strata` is a claim about what the predictor can speak on, and this is
        # where it is held to. Scoring a stratum it did not declare makes the
        # declaration meaningless, and a reader of the summary would take those
        # rows for coverage. Declaring more than it scores is allowed: that is
        # absence, which the summary reports as a smaller `n`.
        scored_strata = set(merged.loc[merged[manifest.model].notna(), "stratum"])
        undeclared = sorted(scored_strata - declared_cell_strata(manifest.strata))
        if undeclared:
            raise ValueError(
                f"{manifest.model}: scores cells in {undeclared}, which its "
                f"manifest does not declare; it declares {sorted(manifest.strata)}"
            )
        manifests[manifest.model] = manifest.as_dict()

    return merged, manifests
