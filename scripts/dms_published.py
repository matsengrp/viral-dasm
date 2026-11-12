"""Turn a study's own file into the four columns `fig:dms` reads.

Each of the three deep mutational scans is carried in `data/dms/` as the file
its authors wrote, and this is the one place that converts one into the
``wildtype,site,mutant,functional_effect`` frame the rest of the figure
consumes. Before #30 the conversion ran outside this repository and only its
output arrived, so the choices below -- which effect column, which
regularization, how thin a measurement to keep -- were not readable here.

**Its own module, not part of `scripts/dms.py`.** `dms.py` is a declared input
of `rule dms_leaf_counts`, and `tests/test_rule_inputs.py` compares a rule's
declared inputs by content, so an edit to `dms.py` reruns an 18-minute,
2.6 GB per-leaf tally that reads no DMS file at all. The predictor modules
were split for the same reason in #29. That shields the tally from edits to
*this* module and nothing else: `dms.py` and `config.yaml` are both still
declared inputs of it, so an edit to either still reruns it.

**What this drops, and why the accounting does not see it.** `load_dms` takes
``raw_rows`` from what this returns, so the rows dropped here -- the
regularization strengths the figure does not read, measurements below the
``times_seen`` floor, replicate duplicates, insertion labels, rows with no
measured effect, and mutations equal to their own wildtype -- are outside
:class:`~dms.DmsAccounting` entirely. ``raw_rows`` is therefore the dataset as
this conversion emits it, **not** the row count of the file in `data/dms/`:
that file is carried pre-preparation, so counting its lines will not match.
Every filter *after* ``raw_rows`` is named in the accounting and balanced
against it.

The row counts are unchanged by #30 -- the conversion reproduces, byte for
byte, the tables that used to arrive prepared -- so what moved is where the
dropping is written down, not which rows survive.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `lib` first, so this repository's root is on `sys.path` before anything
# imports `dasm2` and the vendored fork wins over an editable install.
from lib import DmsDataset, assert_dasm2_is_this_repo

import dasm2  # noqa: F401  (imported first for its side effects)

assert_dasm2_is_this_repo()

import pandas as pd

#: The four columns every consumer downstream of here expects, in this order.
#:
#: They are passed into ``from_multidms`` explicitly rather than left to its
#: defaults, which happen to be the same strings today: a file whose columns
#: were renamed would otherwise be pre-filtered on one set of names and parsed
#: on another.
DMS_SITE_COL = "site"
DMS_WT_COL = "wildtype"
DMS_MUT_COL = "mutant"
DMS_EFFECT_COL = "functional_effect"

#: The output frame's column order, which the byte-for-byte check depends on.
OUTPUT_COLUMNS = (DMS_WT_COL, DMS_SITE_COL, DMS_MUT_COL, DMS_EFFECT_COL)

#: A multidms mutation string: ``A1015D``, ``A1020*``, ``A222-``.
#:
#: BA.1's file carries no site, wildtype or mutant column -- its unit is this
#: string -- so for that homolog the three join keys are parsed out of it.
AA_MUTATION_PATTERN = re.compile(r"^([A-Z*])(\d+)([A-Z*-])$")


def parse_aa_mutation(mutation: str) -> tuple[str, int, str]:
    """``"A1015D"`` to ``("A", 1015, "D")``.

    Named for amino acids because `scripts/dms_leaf_freq.py` has its own
    ``parse_mutation`` for a *nucleotide* mutation off a tree branch, which
    takes the same type and returns the same one while meaning something else.

    Raises:
        ValueError: If the string is not a wildtype residue, a site and a
            mutant residue. Raising rather than coercing to NaN is the point:
            an unparsed row would otherwise reach the matrix as a measurement
            at site NaN, and a silent `.notna()` later would drop it.
    """
    matched = AA_MUTATION_PATTERN.match(mutation)
    if matched is None:
        raise ValueError(
            f"cannot parse mutation {mutation!r} as wildtype, site and mutant. "
            f"A row this loader cannot place is a row the figure would score "
            f"at the wrong site or silently lose."
        )
    return matched.group(1), int(matched.group(2)), matched.group(3)


def declared_columns(dataset: DmsDataset) -> tuple[str, ...]:
    """Every column of the source file this dataset's conversion names.

    Checked against the frame before the conversion runs, so a typo in
    `config.yaml` fails saying which key is wrong rather than as a `KeyError`
    inside a groupby -- or, for ``replicate_column``, not at all. That one is
    the reason this exists: the averaging groups on ``mutation_column``, so
    nothing else would ever read it, and a misspelling would change no output.
    """
    named = [dataset.effect_column, dataset.mutation_column, dataset.replicate_column]
    named += [column for column, _ in dataset.fixed]
    if dataset.min_times_seen is not None:
        named.append(dataset.min_times_seen[0])
    return tuple(column for column in named if column is not None)


def _check_columns(frame: pd.DataFrame, dataset: DmsDataset) -> None:
    missing = [column for column in declared_columns(dataset) if column not in frame]
    if missing:
        raise ValueError(
            f"{dataset.homolog}: {sorted(missing)} named in config.yaml but not "
            f"in {dataset.path.name}, which carries {sorted(frame.columns)}."
        )


def _apply_fixed(
    frame: pd.DataFrame, fixed: tuple[tuple[str, float], ...]
) -> pd.DataFrame:
    """Keep the rows whose named columns hold the pinned values."""
    for column, value in fixed:
        # `.tolist()` so the message carries `4e-05` rather than numpy's
        # `np.float64(4e-05)`, which buries the value this is diagnosing.
        present = sorted(frame[column].unique().tolist())
        frame = frame[frame[column] == value]
        if frame.empty:
            raise ValueError(
                f"no row has {column} == {value}. The file's values are {present[:10]}."
            )
    return frame


def normalise_published(raw: pd.DataFrame, dataset: DmsDataset) -> pd.DataFrame:
    """One study's file as the four columns, in the order they are written.

    Two shapes, chosen by whether the dataset names a ``mutation_column``: a
    *long* file whose unit is a mutation string, with a regularization sweep
    and replicates to collapse, and a *tidy* file that already carries one row
    per measured substitution. They meet at a shared tail.

    The steps, in the order they run -- each skipped when the dataset declares
    nothing for it:

    1. Keep the rows matching ``fixed``. BA.1's file holds nine regularization
       strengths; the figure reads one.
    2. Keep the rows whose ``min_times_seen`` column is at least its value.
       This is a **row** filter, not a per-mutation one: a mutation seen often
       enough in one replicate and not the other keeps the replicate that
       passed, and averages over that alone. 972 of BA.1's 7,820 output
       mutations are that case.
    3. Average ``effect_column`` over the replicates, grouping on the mutation
       key.
    4. Parse ``mutation_column`` into the three join keys, or, for a tidy
       file, drop the sites that are not integers. Those are insertion labels
       like ``16a``, numbered relative to a Wuhan-Hu-1 site and so having no
       coordinate in the frame this pipeline indexes on.
    5. The shared tail: drop rows with no measured effect and rows whose
       mutant equals its wildtype, rename the effect column, and sort by
       ``(site, mutant)``.

    Step 5's two drops are the dataset's own preparation rather than this
    figure's filters; see the module docstring on where the accounting
    boundary sits and what it no longer catches.

    Returns:
        A frame with exactly :data:`OUTPUT_COLUMNS` and a reset index.
    """
    _check_columns(raw, dataset)
    frame = _apply_fixed(raw.copy(), dataset.fixed)

    if dataset.min_times_seen is not None:
        column, floor = dataset.min_times_seen
        frame = frame[frame[column] >= floor]

    if dataset.replicate_column is not None:
        frame = (
            frame.groupby(dataset.mutation_column)[dataset.effect_column]
            .mean()
            .reset_index()
        )

    if dataset.mutation_column is not None:
        parsed = frame[dataset.mutation_column].apply(parse_aa_mutation)
        frame[DMS_WT_COL] = parsed.apply(lambda fields: fields[0])
        frame[DMS_SITE_COL] = parsed.apply(lambda fields: fields[1])
        frame[DMS_MUT_COL] = parsed.apply(lambda fields: fields[2])
    else:
        frame[DMS_SITE_COL] = pd.to_numeric(frame[DMS_SITE_COL], errors="coerce")
        frame = frame[frame[DMS_SITE_COL].notna()]
        frame[DMS_SITE_COL] = frame[DMS_SITE_COL].astype(int)

    frame = frame[frame[dataset.effect_column].notna()]
    frame = frame[frame[DMS_WT_COL] != frame[DMS_MUT_COL]]
    frame = frame.rename(columns={dataset.effect_column: DMS_EFFECT_COL})

    return (
        frame[list(OUTPUT_COLUMNS)]
        .sort_values([DMS_SITE_COL, DMS_MUT_COL])
        .reset_index(drop=True)
    )


def read_published(dataset: DmsDataset) -> pd.DataFrame:
    """One study's file as it was written, before any conversion.

    ``site`` is read as a string so an insertion label survives to be dropped
    by name in :func:`normalise_published`; read as a number it would already
    be NaN, and indistinguishable from a site that failed to parse for some
    other reason. In practice KP.3.1.1's column is read as a string anyway,
    because of those labels, so this is defensive rather than load-bearing --
    the conversion's output is byte-identical with and without it. A ``.gz``
    suffix is decompressed by pandas.
    """
    return pd.read_csv(dataset.path, dtype={DMS_SITE_COL: str})
