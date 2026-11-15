"""Each published DMS file, as its own study releases it, in one shape.

`fig:dms` reads three deep mutational scans. Each study publishes its
measurements in its own format -- two columns of identifiers and a phenotype
column among several, or a per-mutation table of fitted parameters -- and each
applies its own filtering before releasing them. This module converts all
three to the four columns the loader wants, ``wildtype,site,mutant,
functional_effect``, and applies whatever filter the study applies that its
released file does not already carry.

**The filtering standard is the study's own.** The three experiments set their
own thresholds and there is no house rule here: a mutation is kept when the
authors would keep it, which is read from their configuration and their
analysis code rather than borrowed from whichever homolog is to hand. What
each one does, and where it was read:

- **BA.1** -- `matsengrp/SARS-CoV-2_spike_multidms`' ``results/mutations_df.csv``
  is already filtered. ``spike-analysis.ipynb`` writes it from
  ``combine_replicate_muts(..., times_seen_threshold=times_seen_threshold,
  how="inner")`` at ``times_seen_threshold = 1`` -- "the number of times a
  mutation must be seen in each condition to be included in downstream
  analysis" -- so every mutation it carries was seen in each of the three
  homolog conditions and was fit in both replicates. The fit is the one that
  notebook calls "the chosen lasso strength for our final spike model
  results", ``chosen_lasso_strength = 5e-5``. Nothing is filtered here.
- **XBB.1.5** -- `dms-vep/SARS-CoV-2_XBB.1.5_spike_DMS` publishes
  ``results/summaries/summary.csv``, the filtered summary across its assays;
  the ``spike mediated entry`` column is the phenotype this figure reads.
  Nothing is filtered here.
- **KP.3.1.1** -- `dms-vep/SARS-CoV-2_KP.3.1.1_spike_DMS` publishes the
  averaged per-phenotype table unfiltered, leaving the thresholds to the
  consumer, so this module applies the ones that repository declares:
  ``times_seen: 2`` and ``effect_std: 2.5`` as a maximum, from
  ``data/func_effects_config.yml``'s ``avg_func_effects_default`` and
  ``data/summaries_config.yml``'s ``min_times_seen``. Applying exactly those
  reproduces that study's own published filtered column -- see
  `tests/test_dms_published.py`.

**The upper bound on ``effect_std`` is inert on KP.3.1.1's current file**, whose
largest value is 1.879. It is applied anyway, because it is the authors'
stated threshold and a later release of the same file may reach it; a filter
that is a no-op today and load-bearing tomorrow is worth keeping rather than
rediscovering.

Published datasets, not GISAID-derived: their measurements, and anything
derived from them here, may be committed.
"""

from __future__ import annotations

import pandas as pd

#: The four columns every study is converted to, in the order they are written.
#: They are `scripts/dms.py`'s ``DMS_*_COL`` constants; this module takes them
#: as arguments rather than importing that module, so that a rule which
#: converts a CSV need not declare the figure's loader as an input.
OUTPUT_COLUMNS = ("wildtype", "site", "mutant", "functional_effect")


def normalise_published(raw: pd.DataFrame, spec) -> pd.DataFrame:
    """One study's published file, as the four columns the loader reads.

    The steps are shared by all three studies and every one of them is a
    property of the released file rather than a choice made here:

    1. Rename the study's identifier and phenotype columns.
    2. Drop rows with no measured effect. A summary table carries one row per
       mutation per assay, so a mutation measured in another assay and not in
       this one arrives with a null phenotype.
    3. Drop synonymous rows, where the wildtype is also the mutant. They are
       not substitutions, and ``from_multidms`` cannot place them.
    4. Drop non-integer sites. These are insertion labels -- KP.3.1.1's
       ``16a`` to ``16d``, numbered relative to Wuhan-Hu-1 site 16 -- which
       have no coordinate in the frame this pipeline indexes on.
    5. Apply the study's thresholds, where its released file does not already
       carry them: see the module docstring for which study needs which.
    6. Sort by ``(site, mutant)``, so the output is a function of the input
       and not of the reader's row order.

    Steps 2 to 4 are not optional even where they bite on nothing. Rows left
    in reach ``from_multidms``, which warns and drops them, and
    :func:`dms.load_dms`'s balance check then raises on the difference.

    Args:
        raw: The study's published file, read with no conversion.
        spec: A :class:`lib.DmsDataset`, carrying the column names and
            whatever thresholds the study declares.

    Returns:
        A frame of :data:`OUTPUT_COLUMNS`, sorted, with a fresh index.

    Raises:
        KeyError: If a column the spec names is not in the file, which is what
            a renamed or repointed release looks like from here.
    """
    missing = [
        column
        for column in (
            spec.wt_column,
            spec.site_column,
            spec.mutant_column,
            spec.effect_column,
        )
        if column not in raw.columns
    ]
    if missing:
        raise KeyError(
            f"{spec.homolog}: {missing} not in {list(raw.columns)}. The "
            f"published file this names has changed shape, so the conversion "
            f"would silently read a different quantity."
        )

    frame = raw.rename(
        columns={
            spec.wt_column: "wildtype",
            spec.site_column: "site",
            spec.mutant_column: "mutant",
            spec.effect_column: "functional_effect",
        }
    )

    frame = frame[frame["functional_effect"].notna()]
    frame = frame[frame["wildtype"] != frame["mutant"]]
    frame = frame[frame["site"].astype(str).str.fullmatch(r"-?\d+")]

    # A row whose *wildtype* is a stop is the construct's own terminator --
    # KP.3.1.1's site 1253 -- and is deliberately left in. It is a genuine
    # construct-versus-founder disagreement, and `dms.load_dms` is where that
    # is counted: dropping it here would empty `mismatch_sites` and leave the
    # accounting reporting that the two agree everywhere.

    if spec.min_times_seen is not None:
        frame = frame[frame[spec.times_seen_column] >= spec.min_times_seen]
    if spec.max_effect_std is not None:
        frame = frame[frame[spec.effect_std_column] <= spec.max_effect_std]

    frame = frame.astype({"site": int})
    frame = frame.sort_values(["site", "mutant"], kind="stable")
    return frame[list(OUTPUT_COLUMNS)].reset_index(drop=True)
