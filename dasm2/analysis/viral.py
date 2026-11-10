"""Viral fitness analysis primitives.

Per-mutation selection factor DataFrame construction, fitness merging, delta
fitness computation, cross-clade shift analysis, concordance metrics, and
failure mode classification.

Nothing here reads a path or loads a model. Every function takes frames and
returns frames, which is why it is the layer both the scoring rule and the
figure readouts are built on.
"""

from typing import Literal, NamedTuple, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats

from dasm2.sequence.constants import AA_STR_SORTED


class ShiftMetrics(BaseModel):
    """Correlation and concordance metrics for a cross-clade shift comparison.

    Attributes:
        pearson_r: Pearson correlation between SF shift and fitness shift.
        pearson_p: P-value for the Pearson correlation.
        spearman_rho: Spearman rank correlation between SF shift and fitness shift.
        spearman_p: P-value for the Spearman correlation.
        directional_concordance: Fraction of non-zero mutations where the sign
            of SF shift matches the sign of fitness shift. None if no non-zero
            mutations exist.
        n_mutations: Number of mutations used for the metrics.
    """

    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    directional_concordance: Optional[float] = None
    n_mutations: int


def compute_concordance(sf_shift: pd.Series, fitness_shift: pd.Series) -> float:
    """Compute directional concordance between SF shift and fitness shift.

    Concordance is the fraction of mutations where the sign of the SF shift
    matches the sign of the fitness shift, excluding mutations where either
    shift is effectively zero (|value| <= 1e-10).

    Args:
        sf_shift: Selection factor shift values (B - A).
        fitness_shift: Fitness shift values (B - A).

    Returns:
        Fraction of non-zero mutations with matching signs, or NaN if no
        non-zero mutations exist.
    """
    nonzero = (sf_shift.abs() > 1e-10) & (fitness_shift.abs() > 1e-10)
    if nonzero.sum() == 0:
        return np.nan
    return float((np.sign(sf_shift[nonzero]) == np.sign(fitness_shift[nonzero])).mean())


def build_sf_dataframe_from_tensor(
    log_sf: np.ndarray,
    founder_sequence: str,
) -> pd.DataFrame:
    """Build a per-mutation selection factor DataFrame from a log SF tensor.

    This is the low-level pure function that takes pre-computed log selection
    factors and constructs a DataFrame with one row per (site, amino acid) pair.

    Args:
        log_sf: Log selection factors with shape ``[L, 20]`` where L is the
            sequence length and 20 corresponds to ``AA_STR_SORTED``.
        founder_sequence: Founder/wild-type amino acid sequence of length L.

    Returns:
        DataFrame with columns: ``aa_mutation``, ``aa_site`` (1-indexed),
        ``mutant_aa``, ``log_selection_factor``, ``is_synonymous``.
        Has ``L * 20`` rows.

    Raises:
        ValueError: If ``log_sf`` shape is inconsistent with
            ``founder_sequence`` length.
    """
    if log_sf.shape[0] != len(founder_sequence):
        raise ValueError(
            f"log_sf has {log_sf.shape[0]} sites but founder_sequence has "
            f"{len(founder_sequence)} residues"
        )
    if log_sf.shape[1] != len(AA_STR_SORTED):
        raise ValueError(
            f"log_sf has {log_sf.shape[1]} amino acids but expected "
            f"{len(AA_STR_SORTED)}"
        )

    records = []
    for site_idx in range(len(founder_sequence)):
        wt_aa = founder_sequence[site_idx]
        aa_site = site_idx + 1  # 1-indexed
        for aa_idx, mut_aa in enumerate(AA_STR_SORTED):
            records.append(
                {
                    "aa_mutation": f"{wt_aa}{aa_site}{mut_aa}",
                    "aa_site": aa_site,
                    "mutant_aa": mut_aa,
                    "log_selection_factor": float(log_sf[site_idx, aa_idx]),
                    "is_synonymous": mut_aa == wt_aa,
                }
            )

    return pd.DataFrame(records)


def merge_fitness_and_sf(
    fitness_df: pd.DataFrame,
    sf_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge fitness data with selection factor predictions.

    Performs an inner join on ``(aa_mutation, aa_site, mutant_aa)``.

    Args:
        fitness_df: Fitness DataFrame with columns ``aa_mutation``,
            ``aa_site``, ``mutant_aa``, and fitness-related columns.
        sf_df: Selection factor DataFrame (from :func:`build_sf_dataframe_from_tensor`)
            with columns ``aa_mutation``, ``aa_site``, ``mutant_aa``,
            ``log_selection_factor``, ``is_synonymous``.

    Returns:
        Inner-joined DataFrame with columns from both inputs.
    """
    return fitness_df.merge(
        sf_df, on=["aa_mutation", "aa_site", "mutant_aa"], how="inner"
    )


def compute_delta_fitness(
    merged_df: pd.DataFrame,
    pseudocount: float = 0.5,
) -> pd.DataFrame:
    """Compute per-mutation delta fitness with pseudocount smoothing.

    Adds a ``computed_delta_fitness_pseudo`` column as
    ``log((actual_count + pc) / (predicted_count + pc))``.

    When ``pseudocount=0``, mutations with zero actual or predicted count
    get NaN instead of -inf.

    Also computes ``sitewise_delta_fitness`` by aggregating actual/predicted
    counts per site.

    Args:
        merged_df: DataFrame with ``actual_count`` and ``predicted_count``
            columns.
        pseudocount: Smoothing pseudocount added to numerator and denominator.

    Returns:
        Copy of ``merged_df`` with added ``computed_delta_fitness_pseudo``
        and ``sitewise_delta_fitness`` columns.
    """
    df = merged_df.copy()

    if len(df) == 0:
        df["computed_delta_fitness_pseudo"] = pd.Series(dtype=float)
        df["sitewise_delta_fitness"] = pd.Series(dtype=float)
        return df

    df["computed_delta_fitness_pseudo"] = _log_ratio_with_pseudocount(
        df["actual_count"], df["predicted_count"], pseudocount
    )

    site_agg = (
        df.groupby("aa_site")
        .agg({"actual_count": "sum", "predicted_count": "sum"})
        .reset_index()
    )
    site_agg["sitewise_delta_fitness"] = _log_ratio_with_pseudocount(
        site_agg["actual_count"], site_agg["predicted_count"], pseudocount
    )
    df = df.merge(
        site_agg[["aa_site", "sitewise_delta_fitness"]], on="aa_site", how="left"
    )

    return df


def _log_ratio_with_pseudocount(
    actual: pd.Series,
    predicted: pd.Series,
    pseudocount: float,
) -> pd.Series:
    """Compute log(actual/predicted) with pseudocount smoothing.

    Args:
        actual: Actual counts.
        predicted: Predicted (expected) counts.
        pseudocount: Added to both numerator and denominator. When 0,
            returns NaN for zero-count entries.

    Returns:
        Log ratio series.
    """
    if pseudocount == 0:
        safe = (actual > 0) & (predicted > 0) & actual.notna() & predicted.notna()
        result = pd.Series(np.nan, index=actual.index)
        result[safe] = np.log(actual[safe] / predicted[safe])
        return result
    values: np.ndarray = np.log((actual + pseudocount) / (predicted + pseudocount))
    return pd.Series(values, index=actual.index)


def filter_merged_fitness(
    merged_df: pd.DataFrame,
    *,
    min_predicted_count: float = 0,
    min_actual_count: float = 0,
    count_filter_mode: Literal["and", "or"] = "and",
    exclude_synonymous: bool = False,
    exclude_sites: Optional[set[int]] = None,
) -> pd.DataFrame:
    """Apply quality filters to merged fitness/SF DataFrame.

    Args:
        merged_df: DataFrame from :func:`merge_fitness_and_sf` (possibly
            with delta fitness columns from :func:`compute_delta_fitness`).
        min_predicted_count: Exclude mutations with predicted count below
            this threshold.
        min_actual_count: Exclude mutations with actual count below this
            threshold.
        count_filter_mode: How to combine the two count thresholds.
            ``"and"`` (default): a mutation passes only if it meets both
            thresholds. ``"or"``: a mutation passes if it meets *either*
            threshold — useful for keeping mutations with high actual counts
            but low predicted counts (e.g. antibody escape candidates). Only
            thresholds greater than zero and columns that exist in
            ``merged_df`` contribute to the condition.
        exclude_synonymous: If True, exclude synonymous (WT->WT) mutations.
        exclude_sites: Set of 1-indexed site numbers to exclude (e.g.
            problematic sites).

    Returns:
        Filtered copy of the DataFrame.

    Raises:
        ValueError: If ``count_filter_mode`` is not ``"and"`` or ``"or"``.
    """
    if count_filter_mode not in {"and", "or"}:
        raise ValueError(
            f"count_filter_mode must be 'and' or 'or', got {count_filter_mode!r}"
        )

    df = merged_df.copy()

    if exclude_synonymous and "is_synonymous" in df.columns:
        df = df[~df["is_synonymous"]]
    if exclude_sites:
        df = df[~df["aa_site"].isin(exclude_sites)]

    count_conds = []
    if min_actual_count > 0 and "actual_count" in df.columns:
        count_conds.append(df["actual_count"] >= min_actual_count)
    if min_predicted_count > 0 and "predicted_count" in df.columns:
        count_conds.append(df["predicted_count"] >= min_predicted_count)

    if count_conds:
        if count_filter_mode == "and":
            mask = count_conds[0]
            for cond in count_conds[1:]:
                mask = mask & cond
        else:  # "or"
            mask = count_conds[0]
            for cond in count_conds[1:]:
                mask = mask | cond
        df = df[mask]

    return df


def compute_cross_clade_shift(
    merged_a: pd.DataFrame,
    merged_b: pd.DataFrame,
    fitness_col: str = "delta_fitness",
) -> pd.DataFrame:
    """Compute per-mutation shift between two clades.

    Joins two clade-specific merged DataFrames on ``(aa_site, mutant_aa)``
    and computes ``sf_shift = log_sf_b - log_sf_a`` and
    ``fitness_shift = delta_fitness_b - delta_fitness_a``.

    Args:
        merged_a: Merged fitness+SF DataFrame for clade A. Must contain
            columns: ``aa_site``, ``mutant_aa``, ``aa_mutation``,
            ``log_selection_factor``, and ``fitness_col``.
        merged_b: Merged fitness+SF DataFrame for clade B. Same required
            columns as ``merged_a``.
        fitness_col: Which column carries the fitness effect. Default
            ``"delta_fitness"``, which :func:`normalize_fitness_columns` sets to
            the chosen mat-pcp estimate. Pass ``"delta_fitness_ratio"``,
            ``"delta_fitness_bayesian"`` or ``"computed_delta_fitness_pseudo"``
            to name one directly. The output column names do not change.

    Returns:
        DataFrame with shift columns. Contains: ``aa_site``, ``mutant_aa``,
        ``sf_shift``, ``fitness_shift``, ``mutation_label``, plus suffixed
        columns from each clade (``log_sf_a``, ``log_sf_b``,
        ``delta_fitness_a``, ``delta_fitness_b``, etc.).

        Rows are restricted to mutations where the wildtype amino acid matches
        between the two clades (i.e. same ``(wt, site, mut)`` triplet). When
        ``clade_founder_aa`` is present in both inputs, mismatched-WT pairs are
        dropped so that e.g. D614G in clade A is not compared to E614G in clade
        B.

    Raises:
        ValueError: If either input lacks ``fitness_col``. Letting it through
            would raise a bare ``KeyError`` naming the column but not the side.
    """

    def _suffix_map(side: str) -> dict[str, str]:
        # `fitness_col` is mapped last so that naming it `delta_fitness` (the
        # default) collapses onto the same single entry rather than adding one.
        return {
            "log_selection_factor": f"log_sf_{side}",
            "expected_count": f"expected_count_{side}",
            "predicted_count": f"predicted_count_{side}",
            "actual_count": f"actual_count_{side}",
            "uncertainty": f"uncertainty_{side}",
            "clade_founder_aa": f"wt_aa_{side}",
            "aa_mutation": f"aa_mutation_{side}",
            fitness_col: f"delta_fitness_{side}",
        }

    _SUFFIX_A = _suffix_map("a")
    _SUFFIX_B = _suffix_map("b")

    for name, frame in (("merged_a", merged_a), ("merged_b", merged_b)):
        if fitness_col not in frame.columns:
            raise ValueError(
                f"{name} has no {fitness_col!r} column; columns are "
                f"{sorted(frame.columns)}. Run normalize_fitness_columns on the "
                "fitness frame before merging, or name a column that exists."
            )

    renamed_a = merged_a.rename(columns=_SUFFIX_A)
    renamed_b = merged_b.rename(columns=_SUFFIX_B)

    # Select columns that actually exist after renaming
    a_join_cols = ["aa_site", "mutant_aa"]
    a_data_cols = [c for c in _SUFFIX_A.values() if c in renamed_a.columns]
    b_data_cols = [c for c in _SUFFIX_B.values() if c in renamed_b.columns]

    paired = renamed_a[a_join_cols + a_data_cols].merge(
        renamed_b[["aa_site", "mutant_aa"] + b_data_cols],
        on=["aa_site", "mutant_aa"],
        how="inner",
    )

    # Enforce matching wildtype: only compare (wt, site, mut) triplets that
    # agree between clades. D614G in clade A and E614G in clade B are
    # biologically distinct mutations.
    if "wt_aa_a" in paired.columns and "wt_aa_b" in paired.columns:
        paired = paired[paired["wt_aa_a"] == paired["wt_aa_b"]].reset_index(drop=True)

    # Compute shifts
    paired["sf_shift"] = paired["log_sf_b"] - paired["log_sf_a"]
    paired["fitness_shift"] = paired["delta_fitness_b"] - paired["delta_fitness_a"]

    # Unified mutation label using clade A's WT (by construction identical to B's).
    if "wt_aa_a" in paired.columns:
        paired["mutation_label"] = (
            paired["wt_aa_a"] + paired["aa_site"].astype(str) + paired["mutant_aa"]
        )

    return paired


def compute_shift_metrics(shift_df: pd.DataFrame) -> ShiftMetrics:
    """Compute correlation and concordance metrics for a shift DataFrame.

    Args:
        shift_df: DataFrame from :func:`compute_cross_clade_shift` with
            ``sf_shift`` and ``fitness_shift`` columns.

    Returns:
        ShiftMetrics with Pearson, Spearman, concordance, and mutation count.

    Raises:
        ValueError: If fewer than 3 non-NaN mutations are available.
    """
    valid = shift_df.dropna(subset=["sf_shift", "fitness_shift"])
    if len(valid) < 3:
        raise ValueError(
            f"Need at least 3 non-NaN mutations for shift metrics, got {len(valid)}"
        )

    pearson_r, pearson_p = stats.pearsonr(valid["sf_shift"], valid["fitness_shift"])
    spearman_rho, spearman_p = stats.spearmanr(
        valid["sf_shift"], valid["fitness_shift"]
    )
    concordance = compute_concordance(valid["sf_shift"], valid["fitness_shift"])

    dir_concordance: Optional[float] = None if np.isnan(concordance) else concordance

    return ShiftMetrics(
        pearson_r=float(pearson_r),
        pearson_p=float(pearson_p),
        spearman_rho=float(spearman_rho),
        spearman_p=float(spearman_p),
        directional_concordance=dir_concordance,
        n_mutations=len(valid),
    )


def compute_within_clade_r_summary(
    merged_by_clade: dict[str, pd.DataFrame],
    fitness_col: str = "computed_delta_fitness_pseudo",
) -> pd.DataFrame:
    """Compute per-clade Pearson r between log SF and delta fitness.

    For each clade, computes ``scipy.stats.pearsonr`` between
    ``log_selection_factor`` and ``fitness_col`` over the rows in the provided
    DataFrame. Filtering (pseudocount, OR-mode min counts, exclude-synonymous,
    exclude-sites) must be applied by the caller before constructing
    ``merged_by_clade``; this helper does not filter.

    Args:
        merged_by_clade: Mapping from clade name to a pre-filtered merged
            DataFrame with columns ``log_selection_factor`` and ``fitness_col``.
        fitness_col: Which column carries the fitness effect. The default is the
            locally recomputed ``computed_delta_fitness_pseudo``, kept so that
            existing callers keep their meaning; pass
            ``"delta_fitness_ratio"`` to correlate against mat-pcp's own
            published estimate instead.

    Non-finite values are dropped per row, not per clade. An infinite log SF is
    a value the model really produced and cannot be correlated, so the row goes
    and the clade's remaining rows still report; leaving it in makes
    ``pearsonr`` return NaN and costs the whole clade over one entry.
    :func:`~dasm2.analysis.viral_fitness_eval.within_clade_metrics` is the same
    computation with a Spearman column, a settable ``min_mutations`` and clades
    sorted, and the two agree on every input.

    Returns:
        DataFrame with columns ``clade``, ``pearson_r``, ``n_mutations``.
        One row per clade, where ``n_mutations`` counts the rows actually
        correlated. Clades with fewer than 3 finite rows get ``pearson_r=NaN``,
        as do clades whose fitness column is constant. Empty input returns an
        empty DataFrame with those columns.

    Raises:
        ValueError: If any clade DataFrame is missing a required column.
    """
    required_cols = ("log_selection_factor", fitness_col)

    if not merged_by_clade:
        return pd.DataFrame(
            {
                "clade": pd.Series(dtype=str),
                "pearson_r": pd.Series(dtype=float),
                "n_mutations": pd.Series(dtype=int),
            }
        )

    rows = []
    for clade, df in merged_by_clade.items():
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Clade {clade!r} missing required column {col!r}")

        valid = df.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required_cols))
        n_mutations = len(valid)
        if n_mutations < 3:
            pearson_r = float("nan")
        else:
            pearson_r = float(
                stats.pearsonr(
                    valid["log_selection_factor"],
                    valid[fitness_col],
                ).statistic
            )
        rows.append(
            {"clade": clade, "pearson_r": pearson_r, "n_mutations": n_mutations}
        )

    return pd.DataFrame(rows, columns=["clade", "pearson_r", "n_mutations"])


def compute_shift_r_from_reference(
    merged_by_clade: dict[str, pd.DataFrame],
    reference_clade: str,
    other_clades: list[str],
    fitness_col: str = "delta_fitness",
) -> pd.DataFrame:
    """Compute shift Pearson r from a reference clade to each other clade.

    For each clade in ``other_clades``, computes the cross-clade shift with
    :func:`compute_cross_clade_shift` and extracts all
    :class:`ShiftMetrics` fields via :func:`compute_shift_metrics`. This
    helper is single-model; call once per selected model.

    Args:
        merged_by_clade: Mapping from clade name to a merged fitness+SF
            DataFrame (with the columns required by
            :func:`compute_cross_clade_shift`).
        reference_clade: Clade to use as the reference (A) in all pairs.
        other_clades: Clades to pair against the reference. An empty list is
            valid and yields an empty DataFrame with the documented columns.
        fitness_col: Passed through to :func:`compute_cross_clade_shift`.

    Returns:
        DataFrame with columns ``other_clade``, ``pearson_r``, ``pearson_p``,
        ``spearman_rho``, ``spearman_p``, ``directional_concordance``,
        ``n_mutations``. Clade pairs with fewer than 3 non-NaN shared
        mutations get ``pearson_r`` and the other metric columns set to NaN
        rather than raising.

    Raises:
        ValueError: If ``reference_clade`` or any entry of ``other_clades``
            is not a key in ``merged_by_clade``.
    """
    if reference_clade not in merged_by_clade:
        raise ValueError(f"reference_clade {reference_clade!r} not in merged_by_clade")

    columns = [
        "other_clade",
        "pearson_r",
        "pearson_p",
        "spearman_rho",
        "spearman_p",
        "directional_concordance",
        "n_mutations",
    ]

    rows = []
    reference_df = merged_by_clade[reference_clade]
    for other in other_clades:
        if other not in merged_by_clade:
            raise ValueError(f"other_clade {other!r} not in merged_by_clade")
        shift_df = compute_cross_clade_shift(
            reference_df, merged_by_clade[other], fitness_col=fitness_col
        )
        try:
            metrics = compute_shift_metrics(shift_df)
        except ValueError:
            n_valid = len(shift_df.dropna(subset=["sf_shift", "fitness_shift"]))
            rows.append(
                {
                    "other_clade": other,
                    "pearson_r": float("nan"),
                    "pearson_p": float("nan"),
                    "spearman_rho": float("nan"),
                    "spearman_p": float("nan"),
                    "directional_concordance": float("nan"),
                    "n_mutations": n_valid,
                }
            )
            continue

        dir_conc = (
            float("nan")
            if metrics.directional_concordance is None
            else metrics.directional_concordance
        )
        rows.append(
            {
                "other_clade": other,
                "pearson_r": metrics.pearson_r,
                "pearson_p": metrics.pearson_p,
                "spearman_rho": metrics.spearman_rho,
                "spearman_p": metrics.spearman_p,
                "directional_concordance": dir_conc,
                "n_mutations": metrics.n_mutations,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def classify_shift_failure_modes(
    shift_df: pd.DataFrame,
    neutral_threshold: float = 0.5,
    effect_threshold: float = 1.0,
) -> pd.DataFrame:
    """Classify mutations by their cross-clade shift pattern.

    Categories:
    - **switch**: Neutral in one clade (|delta_fitness| < neutral_threshold)
      but large effect in the other (|delta_fitness| >= effect_threshold).
    - **large_shift**: Large absolute fitness shift (>= effect_threshold)
      that doesn't meet the switch pattern.
    - **small_shift**: Everything else.

    Args:
        shift_df: DataFrame with ``delta_fitness_a``, ``delta_fitness_b``,
            and ``fitness_shift`` columns.
        neutral_threshold: Maximum |delta_fitness| to be considered neutral.
        effect_threshold: Minimum |delta_fitness| to be considered a large
            effect.

    Returns:
        Copy of ``shift_df`` with an added ``shift_category`` column.
    """
    df = shift_df.copy()

    def _classify(row: pd.Series) -> str:
        fa = abs(row["delta_fitness_a"])
        fb = abs(row["delta_fitness_b"])
        if (fa < neutral_threshold and fb >= effect_threshold) or (
            fb < neutral_threshold and fa >= effect_threshold
        ):
            return "switch"
        elif abs(row["fitness_shift"]) >= effect_threshold:
            return "large_shift"
        else:
            return "small_shift"

    df["shift_category"] = df.apply(_classify, axis=1)
    return df


def lookup_mutation_across_clades(
    mutation_str: str,
    fitness_data: dict[str, pd.DataFrame],
    sf_data: dict[str, pd.DataFrame],
    founders: dict[str, str],
    fitness_col: str = "delta_fitness",
) -> pd.DataFrame:
    """Look up a single mutation's fitness and SF across all clades.

    Args:
        mutation_str: Mutation string like ``"D614G"`` (first char = WT AA,
            last char = mutant AA, middle = 1-indexed site number).
        fitness_data: ``{clade: fitness_df}`` where each DataFrame has
            ``aa_site``, ``mutant_aa``, ``clade_founder_aa``, and
            ``fitness_col`` columns.
        sf_data: ``{clade: sf_df}`` where each DataFrame has ``aa_site``,
            ``mutant_aa``, ``log_selection_factor`` columns.
        founders: ``{clade: founder_sequence}`` for each clade.
        fitness_col: Which column to read the fitness effect from. The output
            column is named ``delta_fitness`` either way.

    Returns:
        DataFrame with columns ``clade``, ``delta_fitness``, ``log_sf``,
        ``expected_count``. One row per clade where the mutation is relevant
        (founder AA at the site matches the mutation's WT AA). Empty DataFrame
        if mutation not found in any clade.

    Raises:
        ValueError: If ``mutation_str`` is too short to parse or contains a
            non-numeric site.
    """
    if len(mutation_str) < 3:
        raise ValueError(
            f"Mutation string must be at least 3 characters, got '{mutation_str}'"
        )

    wt_aa = mutation_str[0]
    mutant_aa = mutation_str[-1]
    site_str = mutation_str[1:-1]

    try:
        aa_site = int(site_str)
    except ValueError:
        raise ValueError(
            f"Could not parse site number from mutation '{mutation_str}': "
            f"'{site_str}' is not an integer"
        ) from None

    site_idx = aa_site - 1  # 0-indexed

    records = []
    for clade in sorted(fitness_data.keys()):
        founder = founders.get(clade, "")

        # Only include if founder AA at this site matches WT
        if site_idx >= len(founder) or founder[site_idx] != wt_aa:
            continue

        fit_df = fitness_data[clade]
        fit_match = fit_df[
            (fit_df["aa_site"] == aa_site)
            & (fit_df["mutant_aa"] == mutant_aa)
            & (fit_df["clade_founder_aa"] == wt_aa)
        ]
        delta_fitness = (
            float(fit_match[fitness_col].iloc[0]) if len(fit_match) > 0 else np.nan
        )
        expected_count = (
            float(fit_match["expected_count"].iloc[0])
            if len(fit_match) > 0 and "expected_count" in fit_match.columns
            else np.nan
        )

        sf_df = sf_data[clade]
        sf_match = sf_df[
            (sf_df["aa_site"] == aa_site) & (sf_df["mutant_aa"] == mutant_aa)
        ]
        log_sf = (
            float(sf_match["log_selection_factor"].iloc[0])
            if len(sf_match) > 0
            else np.nan
        )

        if not (np.isnan(delta_fitness) and np.isnan(log_sf)):
            records.append(
                {
                    "clade": clade,
                    "delta_fitness": delta_fitness,
                    "log_sf": log_sf,
                    "expected_count": expected_count,
                }
            )

    return pd.DataFrame(records)


# Canonical name -> the mat-pcp spellings of the same quantity. The canonical
# name is what every function below reads, and it is not always the name on
# disk: the v12 tables this repository reads carry `expected_count`, while
# `compute_delta_fitness` and `filter_merged_fitness` read `predicted_count`.
#
# Normalizing is therefore mandatory, not a convenience, and the count column
# is the one that fails quietly. `filter_merged_fitness` gates on a count
# column only `if ... in df.columns`, so a frame that reached it still spelled
# `expected_count` would pass the count filter on `actual_count` alone -- no
# error, a plausible row count, and a wrong x-axis wherever the count
# threshold is swept.
#
# The two estimate aliases cover mat-pcp's earlier spellings of the same two
# columns (`delta_fitness` for the shrunk estimate, `naive_delta_fitness` for
# the raw log ratio). The v12 tables carry the canonical names already.
FITNESS_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "predicted_count": ("expected_count",),
    "delta_fitness_ratio": ("naive_delta_fitness",),
    "delta_fitness_bayesian": ("delta_fitness",),
}

# The two fitness estimates mat-pcp publishes per amino-acid mutation.
#
# `ratio` is the raw log count ratio, and on this repository's v12 tables it is
# *exactly* `log((actual_count + 0.5) / (expected_count + 0.5))`: checked
# against all 9,140 clade-21J Spike rows, max absolute deviation 2.22e-16. So
# it equals what :func:`compute_delta_fitness` recomputes at its default
# pseudocount, and reading the shipped column instead removes the chance of a
# pseudocount setting quietly redefining the quantity being correlated against.
#
# `bayesian` shrinks that ratio toward zero using a fitted prior. It is the
# better *estimate* of a single mutation's effect, but the shrinkage is itself
# count-dependent, so it mixes an estimator choice into any correlation against
# model predictions. `ratio` is the default here for that reason. On clade 21J
# Spike the two correlate at 0.9825, so the choice moves a correlation very
# little -- it is made for what the quantity *is*, not for what it scores.
FitnessEstimate = Literal["ratio", "bayesian"]

FITNESS_ESTIMATE_COLUMNS: dict[str, str] = {
    "ratio": "delta_fitness_ratio",
    "bayesian": "delta_fitness_bayesian",
}

DEFAULT_FITNESS_ESTIMATE: FitnessEstimate = "ratio"


def normalize_fitness_columns(
    fitness_df: pd.DataFrame,
    estimate: FitnessEstimate = "ratio",
) -> pd.DataFrame:
    """Give a mat-pcp fitness frame canonical column names, whatever its version.

    Does two things. First, copies each column in :data:`FITNESS_COLUMN_ALIASES`
    to its canonical name when only an older spelling is present, so that any
    mat-pcp frame comes out with ``predicted_count``, ``delta_fitness_ratio``
    and ``delta_fitness_bayesian``. Originals are left in place, so nothing is
    lost and the call is idempotent.

    Second, sets ``delta_fitness`` to the chosen estimate. That column is what
    :func:`compute_cross_clade_shift` and :func:`lookup_mutation_across_clades`
    read by default, so ``estimate`` decides which of mat-pcp's two published
    estimates the rest of the pipeline means by "fitness". On a frame that
    already carries its own ``delta_fitness`` this **overwrites** it; the
    original is still reachable as ``delta_fitness_bayesian``.

    **Calling this is not optional.** The v12 tables spell the count column
    ``expected_count``, so without this :func:`compute_delta_fitness` raises
    ``KeyError: 'predicted_count'`` and :func:`filter_merged_fitness` silently
    skips the predicted-count half of its gate -- see
    :data:`FITNESS_COLUMN_ALIASES`.

    Args:
        fitness_df: A mat-pcp ``aamut_fitness`` frame, any pipeline version.
        estimate: Which published estimate ``delta_fitness`` should carry.
            ``"ratio"`` (default) is the raw log count ratio; ``"bayesian"`` is
            the shrunk estimate. See :data:`FITNESS_ESTIMATE_COLUMNS`.

    Returns:
        A copy with canonical columns filled in and ``delta_fitness`` set.

    Raises:
        ValueError: If ``estimate`` is not a known key, or if a required column
            is absent under every known spelling. Guessing would produce a
            plausible correlation against the wrong quantity, so this fails
            instead.
    """
    if estimate not in FITNESS_ESTIMATE_COLUMNS:
        raise ValueError(
            f"estimate must be one of {sorted(FITNESS_ESTIMATE_COLUMNS)}, "
            f"got {estimate!r}"
        )

    df = fitness_df.copy()
    for canonical, aliases in FITNESS_COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        source = next((a for a in aliases if a in df.columns), None)
        if source is not None:
            df[canonical] = df[source]

    estimate_col = FITNESS_ESTIMATE_COLUMNS[estimate]
    for required in ("predicted_count", estimate_col):
        if required not in df.columns:
            raise ValueError(
                f"fitness frame has no {required!r} column and none of its known "
                f"aliases {FITNESS_COLUMN_ALIASES.get(required, ())}; columns "
                f"are {sorted(df.columns)}. Add the new mat-pcp spelling to "
                "FITNESS_COLUMN_ALIASES rather than letting a downstream "
                "KeyError name the wrong cause."
            )
    df["delta_fitness"] = df[estimate_col]
    return df


class FitnessGeneMapping(NamedTuple):
    """How a PCP gene is recovered from the genome-wide fitness table.

    Attributes:
        gene: The ``gene`` value to select on in ``aamut_fitness``.
        codon_offset: Subtracted from ``aa_site`` to reach the PCP gene's own
            1-based numbering.
        max_codon: Last codon of the PCP gene that shares ``gene``'s reading
            frame, in the PCP gene's numbering, or None for "all of it".
    """

    gene: str
    codon_offset: int
    max_codon: Optional[int] = None


# PCP gene -> how to recover it from `aamut_fitness`, mirroring mat-pcp's
# `pcp_gene_to_fitness_gene`. Needed only where a gene is absent from the fitness
# table under its own name and numbering.
#
# S, E, M and N reach the fitness table under their own names at their own codon
# numbering, so they are absent here and slice by name alone. ORF1a and ORF1b do
# not: `aamutfit.get_coding` drops ORF1a entirely, and ORF1b is an internal slice
# of ORF1ab. Both are recovered from `ORF1ab` rows, with ORF1b's codon 1 sitting
# at ORF1ab codon 4393 -- the -1 ribosomal frameshift.
#
# `max_codon` exists for ORF1a and is the subtle one. ORF1a is 4405 codons, but
# the -1 slippage is at ORF1ab codon 4402, so from there on `ORF1ab` is
# translated in ORF1b's frame while ORF1a's own founder continues in ORF1a's.
# The same four nucleotide triplets therefore give different residues, and only
# codons 1-4401 are comparable. ORF1b needs no cap: its founder is post-slippage
# throughout, and its first 9 residues (ORF1ab 4393-4401) are translated in
# ORF1a's frame in both the founder and the fitness table, so they agree.
#
# Getting any of this wrong does not error on its own: it returns a full-looking
# frame whose rows are compared against the wrong residue. So
# :func:`slice_fitness_for_gene` checks founder agreement before returning.
#
# This repository reads `S` only, which slices by name alone, so neither entry
# below is reachable from `config.yaml`. They are kept because
# `slice_fitness_for_gene` consults the mapping unconditionally.
PCP_GENE_TO_FITNESS_GENE: dict[str, FitnessGeneMapping] = {
    "ORF1a": FitnessGeneMapping(gene="ORF1ab", codon_offset=0, max_codon=4401),
    "ORF1b": FitnessGeneMapping(gene="ORF1ab", codon_offset=4392),
}


def slice_fitness_for_gene(
    fitness_df: pd.DataFrame,
    gene: str,
    founder_aa: Optional[str] = None,
) -> pd.DataFrame:
    """Select one gene's rows from a genome-wide fitness frame, renumbered.

    mat-pcp's ``aamut_fitness`` is genome-wide with a ``gene`` column. Most
    genes, ``S`` among them, appear under their own name and numbering; ORF1a
    and ORF1b have to be recovered from ``ORF1ab`` and renumbered by
    :data:`PCP_GENE_TO_FITNESS_GENE`.

    Args:
        fitness_df: Genome-wide fitness frame with ``gene`` and ``aa_site``.
        gene: PCP gene name, e.g. ``"S"``, ``"E"``, ``"ORF1b"``.
        founder_aa: That gene's founder amino acid sequence, **for the same clade
            as** ``fitness_df``. When given, every returned row's
            ``clade_founder_aa`` must equal the founder residue at its
            (renumbered) site. This is what makes a wrong offset, or a founder
            from the wrong clade, fail loudly instead of silently mis-pairing
            mutations.

    Returns:
        A copy of the gene's rows with ``aa_site`` in the gene's own 1-based
        numbering, sorted by ``(aa_site, mutant_aa)``. For ORF1a this stops at
        codon 4401; see :data:`PCP_GENE_TO_FITNESS_GENE`.

        ``aa_mutation`` is rebuilt as ``clade_founder_aa + aa_site + mutant_aa``
        from the renumbered site, matching what
        :func:`build_sf_dataframe_from_tensor` produces. This is not cosmetic:
        :func:`merge_fitness_and_sf` joins on ``aa_mutation``, so leaving the
        source spelling in place made ORF1b -- the only gene with a nonzero offset
        -- merge to **zero rows**, silently and with no error, because its labels
        still read ``S4393A`` where the model's read ``S1A``. At offset 0 the
        rebuild reproduces the source string exactly.

    Raises:
        ValueError: If ``gene`` has no rows, if no row survives the codon range,
            or if any ``clade_founder_aa`` disagrees with ``founder_aa``.
    """
    if "gene" not in fitness_df.columns:
        raise ValueError(
            f"fitness frame has no 'gene' column; columns are "
            f"{sorted(fitness_df.columns)}"
        )

    mapping = PCP_GENE_TO_FITNESS_GENE.get(gene, FitnessGeneMapping(gene, 0))
    df = fitness_df[fitness_df["gene"] == mapping.gene].copy()
    if df.empty:
        available = sorted(fitness_df["gene"].dropna().unique())
        raise ValueError(
            f"no fitness rows for gene {gene!r} (looked for {mapping.gene!r}); "
            f"available genes are {available}"
        )

    offset = mapping.codon_offset
    if offset:
        df["aa_site"] = df["aa_site"] - offset
    if mapping.max_codon is not None:
        df = df[df["aa_site"] <= mapping.max_codon].copy()

    if founder_aa is not None:
        df = df[(df["aa_site"] >= 1) & (df["aa_site"] <= len(founder_aa))].copy()
        if df.empty:
            raise ValueError(
                f"{gene}: no fitness rows fall within codons 1-{len(founder_aa)} "
                f"after applying offset {offset}. The offset is almost certainly "
                "wrong for this gene."
            )
        expected = df["aa_site"].map(lambda s: founder_aa[int(s) - 1])
        mismatched = df["clade_founder_aa"] != expected
        if mismatched.any():
            sample = df.loc[mismatched, ["aa_site", "clade_founder_aa"]].head(3)
            n_bad = int(mismatched.sum())
            shown = ", ".join(
                f"{int(site)}:{wt}!={founder_aa[int(site) - 1]}"
                for site, wt in zip(sample["aa_site"], sample["clade_founder_aa"])
            )
            raise ValueError(
                f"{gene}: {n_bad} of {len(df)} fitness rows disagree with the "
                f"founder on the wildtype residue (offset {offset}). First few, "
                f"as site/fitness-WT against founder-WT: {shown}. A wrong codon "
                "offset, or a founder from a different clade, produces exactly this."
            )

    # Relabel to the renumbered site. `merge_fitness_and_sf` joins on
    # `aa_mutation`, so a label still carrying the source numbering matches
    # nothing -- see the note in this function's Returns section.
    if {"clade_founder_aa", "mutant_aa"} <= set(df.columns):
        df["aa_mutation"] = (
            df["clade_founder_aa"] + df["aa_site"].astype(str) + df["mutant_aa"]
        )

    return df.sort_values(["aa_site", "mutant_aa"]).reset_index(drop=True)


def compute_shift_r_all_pairs(
    merged_by_clade: dict[str, pd.DataFrame],
    clades: list[str],
    fitness_col: str = "delta_fitness",
) -> pd.DataFrame:
    """Shift metrics for every unordered pair drawn from `clades`.

    :func:`compute_shift_r_from_reference` fixes one clade as A and varies B,
    which answers "how well does the model track change *away from this clade*".
    This instead covers all ``n*(n-1)/2`` pairs, so no clade is privileged and the
    spread reflects the clade set rather than the choice of reference.

    Pair order is ``(a, b)`` with ``a < b`` in the order given by ``clades``, and
    the shift is ``b - a`` in both quantities -- so the sign convention matches
    :func:`compute_cross_clade_shift` and flipping a pair would only flip the sign
    of both shifts, leaving Pearson r unchanged.

    Args:
        merged_by_clade: ``{clade: merged fitness+SF frame}``, as required by
            :func:`compute_cross_clade_shift`.
        clades: Clades to draw pairs from. Fewer than two yields an empty frame.
        fitness_col: Passed through to :func:`compute_cross_clade_shift`.

    A **constant** ``sf_shift`` -- what a context-independent baseline produces
    by construction, since its selection factor depends only on (site, parent,
    child) and the shift cancels -- makes Pearson r undefined, and this reports
    it as NaN.
    :func:`~dasm2.analysis.viral_fitness_eval.shift_pair_metrics` deliberately
    reports **0** for that case instead, flagged in ``sf_shift_is_constant``:
    NaN reads as missing data and blocks its margin arithmetic, while 0 says
    what is true, that the model explains none of the cross-clade variation.
    Both conventions are defensible and they are not interchangeable, so do not
    pool the two functions' outputs into one column.

    Returns:
        Frame with ``clade_a``, ``clade_b``, ``pair_label``, and every
        :class:`ShiftMetrics` field. Pairs sharing fewer than three usable
        mutations get NaN metrics rather than raising, so one sparse pair does not
        lose the whole comparison.

    Raises:
        ValueError: If any entry of ``clades`` is absent from ``merged_by_clade``,
            or if ``clades`` contains duplicates (which would silently emit a
            self-pair with r == 1).
    """
    if len(set(clades)) != len(clades):
        duplicated = sorted({c for c in clades if clades.count(c) > 1})
        raise ValueError(
            f"clades contains duplicates {duplicated}; a repeated clade would "
            "pair with itself and report a perfect correlation."
        )
    missing = [c for c in clades if c not in merged_by_clade]
    if missing:
        raise ValueError(f"clades not in merged_by_clade: {missing}")

    columns = [
        "clade_a",
        "clade_b",
        "pair_label",
        "pearson_r",
        "pearson_p",
        "spearman_rho",
        "spearman_p",
        "directional_concordance",
        "n_mutations",
    ]

    rows = []
    for i, clade_a in enumerate(clades):
        for clade_b in clades[i + 1 :]:
            shift_df = compute_cross_clade_shift(
                merged_by_clade[clade_a],
                merged_by_clade[clade_b],
                fitness_col=fitness_col,
            )
            base = {
                "clade_a": clade_a,
                "clade_b": clade_b,
                "pair_label": f"{clade_a}→{clade_b}",
            }
            try:
                metrics = compute_shift_metrics(shift_df)
            except ValueError:
                n_valid = len(shift_df.dropna(subset=["sf_shift", "fitness_shift"]))
                rows.append(
                    {
                        **base,
                        "pearson_r": float("nan"),
                        "pearson_p": float("nan"),
                        "spearman_rho": float("nan"),
                        "spearman_p": float("nan"),
                        "directional_concordance": float("nan"),
                        "n_mutations": n_valid,
                    }
                )
                continue
            rows.append(
                {
                    **base,
                    "pearson_r": metrics.pearson_r,
                    "pearson_p": metrics.pearson_p,
                    "spearman_rho": metrics.spearman_rho,
                    "spearman_p": metrics.spearman_p,
                    "directional_concordance": (
                        float("nan")
                        if metrics.directional_concordance is None
                        else metrics.directional_concordance
                    ),
                    "n_mutations": metrics.n_mutations,
                }
            )

    return pd.DataFrame(rows, columns=columns)
