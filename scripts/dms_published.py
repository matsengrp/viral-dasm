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
were split for the same reason in #29. Only the rules that read a DMS file
declare this one.

**The conversion runs before the accounting starts.** `load_dms` takes
``raw_rows`` from what this returns, so a row this drops is not a row that
figure's accounting loses track of: it is a row the dataset's own preparation
never published. That is the same boundary as before #30, when the dropping
happened in another repository -- what changed is that it is now visible.
Every filter *after* ``raw_rows`` is named in
:class:`~dms.DmsAccounting` and balanced against it.
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
MUTATION_PATTERN = re.compile(r"^([A-Z*])(\d+)([A-Z*-])$")


def parse_mutation(mutation: str) -> tuple[str, int, str]:
    """``"A1015D"`` to ``("A", 1015, "D")``.

    Raises:
        ValueError: If the string is not a wildtype residue, a site and a
            mutant residue. Raising rather than coercing to NaN is the point:
            an unparsed row would otherwise reach the matrix as a measurement
            at site NaN, and a silent `.notna()` later would drop it.
    """
    matched = MUTATION_PATTERN.match(mutation)
    if matched is None:
        raise ValueError(
            f"cannot parse mutation {mutation!r} as wildtype, site and mutant. "
            f"A row this loader cannot place is a row the figure would score "
            f"at the wrong site or silently lose."
        )
    return matched.group(1), int(matched.group(2)), matched.group(3)


def _apply_fixed(frame: pd.DataFrame, fixed: tuple[tuple[str, float], ...]):
    """Keep the rows whose named columns hold the pinned values."""
    for column, value in fixed:
        if column not in frame.columns:
            raise ValueError(
                f"{column!r} is pinned to {value} but the file has no such "
                f"column; it carries {sorted(frame.columns)}."
            )
        frame = frame[frame[column] == value]
        if frame.empty:
            raise ValueError(
                f"no row has {column} == {value}. A YAML scalar written "
                f"without a decimal point, such as `4e-05`, loads as a string "
                f"and matches nothing."
            )
    return frame


def normalise_published(raw: pd.DataFrame, dataset: DmsDataset) -> pd.DataFrame:
    """One study's file as the four columns, in the order they are written.

    The steps, in the order they run -- each one skipped when the dataset
    declares nothing for it:

    1. Keep the rows matching ``fixed``. BA.1's file holds nine
       regularization strengths; the figure reads one.
    2. Keep the rows whose ``min_times_seen`` column is at least its value.
       This is a **row** filter, not a per-mutation one: a mutation seen often
       enough in one replicate and not the other keeps the replicate that
       passed, and averages over that alone.
    3. Average ``effect_column`` over ``replicate_column``, grouping on the
       mutation key.
    4. Parse ``mutation_column`` into the three join keys, or, when the file
       already names them, drop the sites that are not integers. Those are
       insertion labels like ``16a``, numbered relative to a Wuhan-Hu-1 site
       and so having no coordinate in the frame this pipeline indexes on.
    5. Drop rows with no measured effect and rows whose mutant equals its
       wildtype, rename the effect column, and sort by ``(site, mutant)``.

    Step 5's two drops are the dataset's own preparation, not this figure's:
    see the module docstring on where the accounting boundary sits.

    Returns:
        A frame with exactly :data:`OUTPUT_COLUMNS`, its index reset.
    """
    frame = _apply_fixed(raw.copy(), dataset.fixed)

    if dataset.min_times_seen is not None:
        column, floor = dataset.min_times_seen
        frame = frame[frame[column] >= floor]

    if dataset.replicate_column is not None:
        if dataset.mutation_column is None:
            raise ValueError(
                f"{dataset.homolog} averages over {dataset.replicate_column!r} "
                f"but names no mutation_column to group on."
            )
        frame = (
            frame.groupby(dataset.mutation_column)[dataset.effect_column]
            .mean()
            .reset_index()
        )

    if dataset.mutation_column is not None:
        parsed = frame[dataset.mutation_column].apply(parse_mutation)
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
    other reason. A ``.gz`` suffix is decompressed by pandas.
    """
    return pd.read_csv(dataset.path, dtype={DMS_SITE_COL: str})
