"""Visualization utilities for DMS-style matrices.

Provides functions for visualizing DMS heatmaps comparing experimental
data with model predictions, codon neighbor comparison boxplots, and
their multi-panel grid analogue used by the compara functional-eval
experiment.
"""

from typing import (
    TYPE_CHECKING,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from pydantic import BaseModel, ConfigDict
from scipy import stats

from dasm2.data.functional.codon_neighbor import split_by_neighbor_status
from dasm2.data.functional.dms_matrix import (
    aligned_finite_pairs,
    correlate_dms_matrices,
    exclude_wildtype_entries,
    mask_wt_to_nan,
    melt_dms_matrix,
)
from dasm2.sequence.constants import AA_HADDOX_GROUPS, AA_STR_HADDOX

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: Diverging red -> white -> light-purple colormap, ported from
#: ``dnsmex.dasm_viz.dms_style_heatmap`` (matsengrp/dasm-experiments). NaN
#: cells (genuinely missing data, and WT cells, which are masked to NaN and
#: marked with ``x`` rather than colored) render as light gray.
DMS_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "dms_cmap_lin", ["red", "white", "#C8C6FF"]
)
DMS_DIVERGING_CMAP.set_bad(color="lightgray")


class DmsPanel(NamedTuple):
    """One heatmap panel in a :func:`plot_dms_comparison` stack.

    Attributes:
        title: Panel title (e.g. ``"Predicted"``, ``"Measured"``,
            ``"ESM2 zero-shot"``).
        matrix: (20 x L) DMS matrix for this panel.
        cb_min_center_max: Optional ``(vmin, vcenter, vmax)`` diverging color
            scale. ``None`` auto-computes a median-centered scale from this
            panel's own finite values (WT cells excluded -- see
            :func:`plot_dms_comparison`).
        cbar_label: Optional colorbar label (e.g. ``"log selection factor"``
            for a predicted panel, or the dataset's ``y_type`` for the
            measured panel). ``None`` leaves the colorbar unlabeled.
    """

    title: str
    matrix: "pd.DataFrame"
    cb_min_center_max: Optional[Tuple[float, float, float]] = None
    cbar_label: Optional[str] = None


class NeighborComparisonStats(BaseModel):
    """Codon-neighbor single-vs-multiple summary statistics.

    Attributes:
        delta_median: ``median_single - median_multiple``. Codon-table
            bias headline number (positive = single-nt-step mutations
            score higher).
        mw_pvalue: Two-sided Mann-Whitney U p-value comparing single
            and multiple distributions.
        n_single: Count of single-nt-neighbor entries (after WT
            exclusion and NaN drop).
        n_multiple: Count of multiple-nt-step entries.
        median_single: Median of the single-neighbor distribution.
        median_multiple: Median of the multiple-neighbor distribution.
    """

    model_config = ConfigDict(extra="forbid")

    delta_median: float
    mw_pvalue: float
    n_single: int
    n_multiple: int
    median_single: float
    median_multiple: float


def plot_dms_heatmap(
    ax: "Axes",
    matrix: "pd.DataFrame",
    title: str,
    aa_order: List[str],
    *,
    show_xlabel: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "coolwarm",
    center: float = 0,
) -> None:
    """Plot DMS-style heatmap with consistent formatting.

    Creates a heatmap visualization of a DMS matrix (20 amino acids x L positions)
    with amino acids on the y-axis and positions on the x-axis.

    Args:
        ax: Matplotlib axes to plot on.
        matrix: DMS matrix with amino acid index and position columns.
        title: Title for the heatmap.
        aa_order: Order of amino acids for y-axis labels.
        show_xlabel: Whether to show x-axis label ("Position").
        vmin: Minimum value for colormap. None uses data min.
        vmax: Maximum value for colormap. None uses data max.
        cmap: Colormap name. Defaults to "coolwarm".
        center: Center value for diverging colormap. Defaults to 0.

    Example:
        >>> fig, ax = plt.subplots()
        >>> plot_dms_heatmap(ax, matrix, "Binding", list("YWFLIVAMKRHDESTNQPGC"))
    """
    import seaborn as sns

    sns.heatmap(
        matrix,
        cmap=cmap,
        center=center,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        yticklabels=True,
    )
    ax.set_title(title)
    ax.set_xlabel("Position (1-indexed)" if show_xlabel else "")
    ax.set_yticklabels(aa_order, rotation=0)


def _mark_wt_cells(
    ax: "Axes",
    matrix_columns: "pd.Index",
    aa_order: List[str],
    reference: str,
    *,
    marker_size: float = 6.0,
) -> None:
    """Draw a small dot at the center of each wild-type cell of a heatmap.

    WT cells are identified from ``reference`` directly, not from NaN --
    the model matrix NaNs its WT cells (``set_wt_nan=True``) while the
    measured matrix does not, so NaN is not a reliable WT signal. A filled
    dot (rather than a glyph like ``*`` or ``x``) reads as unambiguously
    centered and carries no accidental resemblance to an amino-acid letter;
    it's sized well below the cell so the underlying gray WT cell stays
    visible around it.

    Args:
        ax: Axes the heatmap was drawn on (cell (row, col) centers at
            ``(col + 0.5, row + 0.5)``, matching seaborn's ``heatmap``).
        matrix_columns: The plotted matrix's columns, in on-screen order --
            i.e. post-mask 0-indexed positions, so ``enumerate`` gives the
            correct on-screen column index even when some positions were
            dropped.
        aa_order: Amino acid row order as plotted (top-to-bottom).
        reference: Reference (wild-type) amino-acid sequence.
        marker_size: Dot area in ``points**2`` (matplotlib ``scatter`` ``s``);
            kept small so the dot is clearly smaller than a cell.
    """
    aa_row = {aa: idx for idx, aa in enumerate(aa_order)}
    xs, ys = [], []
    for col_idx, position in enumerate(matrix_columns):
        if 0 <= position < len(reference):
            wt_aa = reference[position]
            if wt_aa in aa_row:
                xs.append(col_idx + 0.5)
                ys.append(aa_row[wt_aa] + 0.5)
    ax.scatter(xs, ys, s=marker_size, c="black", marker="o", linewidths=0)


def _guard_degenerate_scale(
    vmin: float, vcenter: float, vmax: float
) -> Tuple[float, float, float]:
    """Nudge ``vmin``/``vmax`` off ``vcenter`` if degenerate, ported from
    ``dnsmex.dasm_viz``'s guard: ``TwoSlopeNorm`` requires ``vmin < vcenter
    < vmax``, so all-one-sided or all-equal data would otherwise raise.
    Applied to both auto-computed and caller-supplied scales -- a caller
    passing an explicit but degenerate ``cb_min_center_max`` should get the
    same nudge as the default path, not a ``TwoSlopeNorm`` error."""
    if vmin >= vcenter:
        vmin = vcenter - 1e-6
    if vmax <= vcenter:
        vmax = vcenter + 1e-6
    return vmin, vcenter, vmax


def _median_centered_cb_min_center_max(
    matrix: "pd.DataFrame",
) -> Tuple[float, float, float]:
    """Independent ``(vmin, median, vmax)`` diverging scale from one panel's
    own finite values.

    Centers the diverging colormap on the panel's own **median** rather
    than a hardcoded ``0.0``. Hardcoded-zero centering only makes sense for
    log-ratio-style ``y_type``s (``enrichment_ratio``/``kd_nm``/
    ``log_kd_nm``), where ``0`` means "no change from WT" by construction.
    For a raw-magnitude ``y_type`` like ProteinGym stability's ``"fitness"``
    (values entirely positive, e.g. 40-56, never near 0), zero is not a
    meaningful reference: forcing ``vcenter=0`` crushes the entire
    distribution into one arm of the colormap. The median splits each
    panel's real values into two comparably-sized arms regardless of
    ``y_type``, and for genuinely-balanced log-ratio data the median sits
    near 0 anyway, so this does not regress the zero-centered case. WT
    cells should be masked out (see
    :func:`~dasm2.data.functional.dms_matrix.mask_wt_to_nan` for why) before
    calling so they neither pull the median nor set ``vmin``/``vmax``. No
    percentile clipping, so no real data point is silently saturated to an
    endpoint color."""
    values = matrix.to_numpy(dtype=float).ravel()
    finite = values[np.isfinite(values)]
    if not finite.size:
        return _guard_degenerate_scale(-1.0, 0.0, 1.0)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    vcenter = float(np.median(finite))
    return _guard_degenerate_scale(vmin, vcenter, vmax)


def _group_boundaries(groups: "Sequence[str]") -> List[int]:
    """Interior row boundaries for a grouped AA ordering: the cumulative group
    sizes, excluding the trailing full-length one (which is just the bottom
    edge). For :data:`AA_HADDOX_GROUPS` (9/5/3/3) this is ``[9, 14, 17]`` -- the
    y positions where :func:`plot_dms_comparison` draws its group-separator
    lines."""
    boundaries = []
    running = 0
    for group in groups[:-1]:
        running += len(group)
        boundaries.append(running)
    return boundaries


def plot_dms_comparison(
    panels: "Sequence[DmsPanel]",
    *,
    reference: str,
    aa_order: Optional[List[str]] = None,
    group_boundaries: Optional[List[int]] = None,
    col_width: float = 0.11,
    row_height: float = 0.18,
    ytick_fontsize: float = 7.0,
    wt_marker_size: float = 6.0,
) -> "Figure":
    """Render a vertical stack of DMS heatmap panels sharing the x (position) axis.

    Ported from ``dnsmex.dasm_viz.dms_style_heatmap`` (matsengrp/dasm-
    experiments): a diverging red/white/light-purple colormap
    (:data:`DMS_DIVERGING_CMAP`), with WT cells marked by a small dot
    (identified from ``reference``, not NaN -- see :func:`_mark_wt_cells`),
    masked to light gray
    (:func:`~dasm2.data.functional.dms_matrix.mask_wt_to_nan`), and
    genuinely-NaN cells also rendered light gray. Each panel gets its own
    independently-scaled color axis and colorbar rather than one scale
    shared across all -- predicted
    (e.g. log selection factors), zero-shot ESM2 log-probabilities, and
    measured (e.g. raw fitness) values routinely live on very different
    dynamic ranges, and a shared scale crushes whichever panel has the
    smaller range to a uniform blob. Each panel's default scale is
    median-centered over its own real values (:func:`_median_centered_cb_min_center_max`).

    Takes an ordered list of :class:`DmsPanel` rather than a fixed
    ``(model, exp)`` pair so callers can add rows (e.g. an ESM2 zero-shot
    panel between predicted and measured) without a signature change.
    Panels stack vertically and share the position (x) axis, as in
    ``koenig.ipynb``'s own comparisons -- unlike side-by-side panels, this
    keeps corresponding positions in the same screen column, so scrolling
    the combined figure left-to-right (it's typically wider than a page for
    long sequences) moves every panel together as one unit. Cells are *not*
    forced square (no ``aspect="equal"``): row height (``row_height``) is
    decoupled from column width (``col_width``) so the y-axis AA labels have
    room to sit legibly one-per-row even when columns are packed tight for a
    long sequence.

    Args:
        panels: Ordered :class:`DmsPanel` list, rendered top-to-bottom. Must
            be non-empty; every panel's matrix must share the same columns
            (position set) so the shared x-axis lines up.
        reference: Reference (wild-type) amino-acid sequence used to mark and
            mask WT cells in every panel. Indexed by the *original* 0-indexed
            position (each matrix's column labels), so it stays full-length
            even when the matrices have been column-sliced to a coverage
            range -- do not pre-slice it.
        aa_order: Row order for all panels. Defaults to ``AA_STR_HADDOX``,
            the biochemically-grouped ordering (charged/polar first, then
            aliphatic, aromatic, and the G/P/C odd-balls) proposed by Hugh
            Haddox for these DMS heatmaps.
        group_boundaries: Row indices at which to draw a thin black dotted
            group-separator line (e.g. ``[9, 14, 17]`` between the four
            ``AA_HADDOX_GROUPS``). When left ``None`` *and* ``aa_order`` also
            defaults, the Haddox group boundaries are drawn; pass ``[]`` to
            suppress them, or a matching list when supplying a custom
            ``aa_order``.
        col_width: Per-position column width in inches (panel width is
            ``col_width * n_positions``). Kept small so long sequences don't
            render impractically wide.
        row_height: Per-AA-row height in inches (panel height is
            ``row_height * len(aa_order)``). Larger than ``col_width`` so the
            y-tick AA labels fit one per row.
        ytick_fontsize: Font size for the y-axis AA labels, set explicitly to
            fit ``row_height`` rather than inheriting the rcParam default
            (which overflows a short row).
        wt_marker_size: Dot area in ``points**2`` for the WT-cell markers
            (see :func:`_mark_wt_cells`).

    Returns:
        Matplotlib ``Figure`` with one heatmap panel per entry in ``panels``,
        each with its own colorbar.

    Raises:
        ValueError: If ``panels`` is empty.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not panels:
        raise ValueError("plot_dms_comparison requires at least one panel")

    if aa_order is None:
        aa_order = list(AA_STR_HADDOX)
        # Default order is grouped, so draw its group-separator lines unless the
        # caller opted out with an explicit []. A custom aa_order gets no lines
        # unless the caller supplies matching boundaries (the Haddox groups
        # wouldn't line up).
        if group_boundaries is None:
            group_boundaries = _group_boundaries(AA_HADDOX_GROUPS)

    n_positions = panels[0].matrix.shape[1]
    fig_width = col_width * n_positions
    panel_height = row_height * len(aa_order)
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(fig_width, panel_height * len(panels)),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for ax, panel in zip(axes, panels):
        plot_matrix = mask_wt_to_nan(
            panel.matrix.reindex(aa_order), reference, aa_order
        )
        cb_min_center_max = panel.cb_min_center_max
        if cb_min_center_max is None:
            cb_min_center_max = _median_centered_cb_min_center_max(plot_matrix)
        vmin, vcenter, vmax = _guard_degenerate_scale(*cb_min_center_max)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        sns.heatmap(
            plot_matrix,
            cmap=DMS_DIVERGING_CMAP,
            norm=norm,
            cbar=True,
            cbar_kws={"label": panel.cbar_label} if panel.cbar_label else None,
            ax=ax,
            yticklabels=True,
        )
        ax.set_title(panel.title)
        ax.set_yticklabels(aa_order, rotation=0, fontsize=ytick_fontsize)
        ax.set_xlabel("")
        _mark_wt_cells(
            ax, plot_matrix.columns, aa_order, reference, marker_size=wt_marker_size
        )
        # Thin black dotted lines demarcating the AA biochemical groups (drawn
        # on top of the heatmap; seaborn's row boundaries fall at integer y).
        for boundary in group_boundaries or []:
            ax.axhline(boundary, color="black", linewidth=0.8, linestyle=":")

    axes[-1].set_xlabel("Position (0-indexed)")
    fig.subplots_adjust(hspace=0.6)
    return fig


class ScatterSeries(NamedTuple):
    """One predicted-vs-measured scatter series in a :func:`plot_dms_scatter`
    figure.

    Attributes:
        label: Panel/row label naming the model (e.g. ``"Predicted (DASM2)"``,
            ``"ESM2 zero-shot"``). In the non-faceted layout it is the panel's
            x-axis label; in the faceted grid it is the row's y-axis label.
        model_matrix: Predicted (20 x L) DMS matrix for this series,
            correlated against the shared measured matrix.
        color: Point/annotation color, used only in the non-faceted layout
            (the faceted grid colors by :class:`ScatterFacet`). ``None`` uses
            matplotlib's default.
    """

    label: str
    model_matrix: "pd.DataFrame"
    color: Optional[str] = None


class ScatterFacet(NamedTuple):
    """One accessibility-class column of a faceted :func:`plot_dms_scatter`
    grid.

    Attributes:
        label: Column title, e.g. ``"single-nt reachable"`` /
            ``">=2 nt required"``.
        cell_mask: Boolean (20 x L) DataFrame; ``True`` keeps that
            ``(amino acid, position)`` cell in this facet. Reindexed onto each
            series' matrix before use (missing cells treated as ``False``), so
            its own row order / column labels need not match exactly. A cell
            belonging to no facet (e.g. a wild-type self-cell, minimum step 0)
            is simply absent from every column.
        color: Point/annotation color for this column, held constant across
            model rows so a reader compares the same class down a column.
    """

    label: str
    cell_mask: "pd.DataFrame"
    color: Optional[str] = None


def plot_dms_scatter(
    series: "Sequence[ScatterSeries]",
    exp_matrix: "pd.DataFrame",
    *,
    method: Literal["spearman", "pearson"] = "spearman",
    exp_label: str = "Measured",
    facets: "Optional[Sequence[ScatterFacet]]" = None,
    reference: Optional[str] = None,
) -> "Figure":
    """Scatter predicted vs measured DMS values, optionally faceted by class.

    Ported from ``dnsmex.koenig_helper.plot_variant_comparison``
    (matsengrp/dasm-experiments), which used Pearson; default here is
    Spearman to match PR #903's ProteinGym reporting metric. Every panel's
    plotted points and its annotated correlation come from the same aligned,
    finite-only point set (:func:`aligned_finite_pairs`) -- any cell that is
    NaN in that series' prediction or in the measured matrix is dropped from
    both, rather than re-deriving alignment via a separate melt/merge that
    could drift out of sync.

    Two layouts:

    - **No ``facets`` (default):** one panel per series, left-to-right, colored
      by :attr:`ScatterSeries.color`, sharing the measured (y) axis.
    - **With ``facets``:** an ``len(series)`` x ``len(facets)`` grid -- one row
      per model (series), one column per accessibility class (facet). Color
      encodes the *facet* (constant down a column); the model is named by the
      row's y-axis label. All panels share the measured (y) axis; each model
      row shares its predicted (x) axis across facets so single- vs
      multiple-nt spreads are directly comparable.

    Args:
        series: Ordered :class:`ScatterSeries` list. Must be non-empty. In the
            faceted layout these are the grid rows (models).
        exp_matrix: Measured (20 x L) DMS matrix, shared across all panels
            (the y-axis).
        method: Correlation method, ``"pearson"`` or ``"spearman"`` --
            narrower than
            :data:`~dasm2.data.functional.dms_matrix.CorrelationMethod`
            because each panel's annotation names the method, and that
            label only knows those two.
        exp_label: Shared measured-axis label.
        facets: Optional ordered :class:`ScatterFacet` list defining the grid
            columns. ``None`` uses the non-faceted layout.
        reference: Reference (wild-type) amino-acid sequence. When given,
            passed through to :func:`correlate_dms_matrices` as
            ``exclude_wt`` so the annotated correlation explicitly excludes
            wild-type cells rather than depending on whichever matrix
            happens to already carry a NaN there. ``None`` (the default)
            preserves prior behavior exactly.

    Returns:
        Matplotlib ``Figure``.

    Raises:
        ValueError: If ``series`` is empty, or ``facets`` is an empty list.
    """
    import matplotlib.pyplot as plt

    if not series:
        raise ValueError("plot_dms_scatter requires at least one series")
    corr_label = "Spearman" if method == "spearman" else "Pearson"

    if facets is None:
        fig, axes = plt.subplots(
            1,
            len(series),
            figsize=(5 * len(series), 5),
            sharey=True,
            squeeze=False,
        )
        for ax, s in zip(axes[0, :], series):
            correlation, model_values, exp_values = correlate_dms_matrices(
                s.model_matrix, exp_matrix, method=method, exclude_wt=reference
            )
            ax.scatter(
                model_values, exp_values, alpha=0.3, s=10, linewidth=0, color=s.color
            )
            ax.set_xlabel(s.label)
            ax.text(
                0.05,
                0.95,
                f"{corr_label} r = {correlation:.3f} (n={len(model_values)})",
                transform=ax.transAxes,
                va="top",
                fontsize=11,
                color=s.color,
            )
        axes[0, 0].set_ylabel(exp_label)
        fig.tight_layout()
        return fig

    if not facets:
        raise ValueError("facets, when given, must be a non-empty list")

    n_rows, n_cols = len(series), len(facets)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        sharex="row",
        sharey=True,
        squeeze=False,
    )
    for i, s in enumerate(series):
        for j, facet in enumerate(facets):
            ax = axes[i, j]
            # Restrict this model's predictions to the facet's cells, then let
            # aligned_finite_pairs drop anything NaN in either matrix -- so the
            # facet's points are exactly (facet cells) intersect (finite pairs).
            mask = (
                facet.cell_mask.reindex(
                    index=s.model_matrix.index, columns=s.model_matrix.columns
                )
                .fillna(False)
                .astype(bool)
            )
            masked_model = s.model_matrix.where(mask)
            if reference is not None:
                # Mask before aligned_finite_pairs, not just in the
                # correlate_dms_matrices call below -- otherwise the
                # scattered points and the annotated correlation could
                # silently diverge on which cells they exclude.
                masked_model = mask_wt_to_nan(masked_model, reference)
            model_values, exp_values = aligned_finite_pairs(masked_model, exp_matrix)
            try:
                correlation, _, _ = correlate_dms_matrices(
                    masked_model, exp_matrix, method=method, exclude_wt=reference
                )
                r_text = f"{corr_label} r = {correlation:.3f}"
            except ValueError:
                # Too few finite pairs in this facet to define a correlation;
                # still scatter whatever points it has.
                r_text = f"{corr_label} r = n/a"
            ax.scatter(
                model_values,
                exp_values,
                alpha=0.3,
                s=10,
                linewidth=0,
                color=facet.color,
            )
            ax.text(
                0.05,
                0.95,
                f"{r_text} (n={len(model_values)})",
                transform=ax.transAxes,
                va="top",
                fontsize=11,
                color=facet.color,
            )
            if i == 0:
                ax.set_title(facet.label, color=facet.color)
            if i == n_rows - 1:
                ax.set_xlabel("Predicted")
        axes[i, 0].set_ylabel(f"{s.label}\n{exp_label}")
    fig.tight_layout()
    return fig


def neighbor_comparison_stats(
    dms_matrix: pd.DataFrame,
    neighbor_indicator: pd.DataFrame,
    reference_sequence: Optional[str] = None,
    *,
    value_name: str = "score",
) -> NeighborComparisonStats:
    """Compute single-vs-multiple-nt-step Δmedian and Mann-Whitney U p-value.

    Splits a DMS-style score matrix by whether each (aa, position) cell
    is reachable from the reference codon via a single nucleotide
    substitution ("single") or requires multiple changes ("multiple"),
    then returns the median difference plus a Mann-Whitney U p-value
    and the underlying per-group counts/medians.

    Lifted from ``experiments/2026-02-20-codon-table-bias/notebooks/
    codon_bias_analysis.py`` (lines 457-471 on main), with four reviewed
    changes from the inline form:

    1. ``value_name`` is now a parameter (was hard-coded ``"score"``).
    2. ``melted.loc[melted["is_codon_neighbor"], ...]`` replaces the
       ``== True`` comparison (and the ``# noqa: E712`` markers).
    3. Returns a :class:`NeighborComparisonStats` with Mann-Whitney
       p-value, per-group counts, and component medians, rather than a
       bare float.
    4. Adds full type hints.

    Args:
        dms_matrix: (20, L) DataFrame of scores (amino acid rows,
            position columns).
        neighbor_indicator: (20, L) boolean indicator from
            ``make_codon_neighbor_indicator``.
        reference_sequence: Reference amino acid sequence used to
            exclude WT cells via ``exclude_wildtype_entries``. ``None``
            skips WT exclusion (use when the input already has WT cells
            NaN-masked, e.g. from
            ``selection_factors_to_dms_matrix(..., set_wt_nan=True)``).
        value_name: Column name used when melting the DMS matrix.

    Returns:
        :class:`NeighborComparisonStats` with delta_median,
        ``mw_pvalue``, single/multiple sample sizes, and per-group
        medians.
    """
    melted = melt_dms_matrix(dms_matrix, value_name=value_name)
    if reference_sequence is not None:
        melted = exclude_wildtype_entries(melted, reference_sequence)
    melted = split_by_neighbor_status(melted, neighbor_indicator)
    melted = melted.dropna(subset=[value_name, "is_codon_neighbor"])

    single_vals = melted.loc[melted["is_codon_neighbor"], value_name].to_numpy()
    multiple_vals = melted.loc[~melted["is_codon_neighbor"], value_name].to_numpy()

    median_single = (
        float(np.nanmedian(single_vals)) if single_vals.size else float("nan")
    )
    median_multiple = (
        float(np.nanmedian(multiple_vals)) if multiple_vals.size else float("nan")
    )
    delta = median_single - median_multiple

    if single_vals.size and multiple_vals.size:
        _, p = stats.mannwhitneyu(single_vals, multiple_vals, alternative="two-sided")
        mw_pvalue = float(p)
    else:
        mw_pvalue = float("nan")

    return NeighborComparisonStats(
        delta_median=float(delta),
        mw_pvalue=mw_pvalue,
        n_single=int(single_vals.size),
        n_multiple=int(multiple_vals.size),
        median_single=median_single,
        median_multiple=median_multiple,
    )


def plot_neighbor_comparison(
    dms_matrix: "pd.DataFrame",
    neighbor_indicator: "pd.DataFrame",
    title: str,
    ax: "Axes",
    *,
    value_name: str = "score",
    reference_sequence: Optional[str] = None,
) -> None:
    """Create boxplot comparing single vs multiple nucleotide mutation scores.

    Splits DMS matrix entries by whether the amino acid is reachable from
    the reference codon via a single nucleotide substitution ("single") or
    requires multiple changes ("multiple"). Produces side-by-side boxplots.

    Args:
        dms_matrix: (20, L) DataFrame of scores (amino acid rows, position columns).
        neighbor_indicator: (20, L) boolean DataFrame from
            ``make_codon_neighbor_indicator``.
        title: Plot title (e.g., "DASM2", "HERMES", "Expression").
        ax: Matplotlib axes to plot on.
        value_name: Label for the y-axis.
        reference_sequence: If provided, exclude WT amino acid at each position
            (entries where the amino acid equals the reference).

    Example:
        >>> fig, ax = plt.subplots()
        >>> plot_neighbor_comparison(dms_matrix, indicator, "DASM2", ax)
    """
    # Stats from the canonical pipeline (medians, counts, Δmedian).
    summary = neighbor_comparison_stats(
        dms_matrix,
        neighbor_indicator,
        reference_sequence,
        value_name=value_name,
    )

    # Melt into long form for seaborn's boxplot. Mirrors the inner pipeline of
    # neighbor_comparison_stats; kept here because seaborn requires the long
    # frame and reconstructing it from the stats object would be lossy.
    melted = melt_dms_matrix(dms_matrix, value_name=value_name)
    if reference_sequence is not None:
        melted = exclude_wildtype_entries(melted, reference_sequence)
    melted = split_by_neighbor_status(melted, neighbor_indicator)
    melted = melted.dropna(subset=[value_name, "is_codon_neighbor"])
    melted["mutation_type"] = melted["is_codon_neighbor"].map(
        {True: "single", False: "multiple"}
    )

    import seaborn as sns

    sns.boxplot(
        data=melted,
        x="mutation_type",
        y=value_name,
        order=["single", "multiple"],
        ax=ax,
        showfliers=False,
        width=0.5,
    )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(value_name)

    # Annotate with counts from the stats summary.
    counts = {"single": summary.n_single, "multiple": summary.n_multiple}
    for i, label in enumerate(["single", "multiple"]):
        y_pos = ax.get_ylim()[1]
        ax.text(
            i,
            y_pos,
            f"n={counts[label]}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="gray",
        )

    ax.text(
        0.95,
        0.05,
        f"Δmedian={summary.delta_median:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
    )


def plot_neighbor_comparison_grid(
    matrices: Mapping[Tuple[str, str], pd.DataFrame],
    neighbor_indicators: Mapping[str, pd.DataFrame],
    reference_sequences: Mapping[str, str],
    *,
    sharey_rows: bool = True,
    panel_width: float = 3.0,
    panel_height: float = 3.5,
) -> "Figure":
    """Render a (model x dataset) grid of single-vs-multiple boxplots.

    Each row is a model, each column is a dataset; cells reuse the
    single-panel :func:`plot_neighbor_comparison` renderer so the
    annotations (n=, Δmedian=) are identical to the existing
    one-off figure produced by the codon-bias notebook.

    Args:
        matrices: Mapping ``(model_name, dataset_name) -> DMS matrix``.
            Missing keys leave their panel blank.
        neighbor_indicators: Mapping ``dataset_name -> neighbor
            indicator DataFrame``. Shared across models for a given
            dataset.
        reference_sequences: Mapping ``dataset_name -> reference AA
            string``. Shared across models for a given dataset.
        sharey_rows: If True, each row shares the same y-axis range.
            Default True so a single model's panels are visually
            comparable across datasets.
        panel_width: Per-panel width in inches.
        panel_height: Per-panel height in inches.

    Returns:
        Matplotlib ``Figure`` containing the grid.
    """
    import matplotlib.pyplot as plt

    models = sorted({m for m, _ in matrices.keys()})
    datasets = sorted({d for _, d in matrices.keys()})

    n_rows = max(len(models), 1)
    n_cols = max(len(datasets), 1)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_width * n_cols, panel_height * n_rows),
        sharey="row" if sharey_rows else False,
        squeeze=False,
    )

    for row_idx, model in enumerate(models):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            matrix = matrices.get((model, dataset))
            if matrix is None:
                ax.text(
                    0.5,
                    0.5,
                    "n/a",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="gray",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            plot_neighbor_comparison(
                matrix,
                neighbor_indicators[dataset],
                title=dataset if row_idx == 0 else "",
                ax=ax,
                reference_sequence=reference_sequences[dataset],
            )
            if col_idx == 0:
                ax.set_ylabel(model)
            else:
                ax.set_ylabel("")

    fig.tight_layout()
    return fig
