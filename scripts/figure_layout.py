"""Place pre-rendered panels on a page, for the paper's composed figures.

A fork of the module of the same name in the private repository's
`2026-09-09-viral-dasm-figure-drafts`, carried across unchanged except for
`save_figure`'s vector write: that tree reaches it through
`dasm2.analysis.plotting`, and this one prunes `dasm2/analysis/` to the single
chart builder `fig:agreement` needs.

Every figure in the paper is built the same way: the rule that owns a panel
writes it as a PNG at its own settings, and a composition notebook reads those
PNGs and arranges them. This module is the arranging part, shared so that two
figures cannot drift into two different ideas of what a panel gap or a panel
letter is.

The one real decision here is that **the dpi is solved for, not chosen**.
Panels arrive as pixels; inches are pixels divided by a dpi; so picking the
smallest dpi whose figure still fits the page box is the same as making the
figure as large as it can legibly be. Nothing is resampled and no panel
changes size relative to any other -- they all move together.

`scale` then shrinks the whole composition uniformly, box and spacing
alike, for a figure that should sit smaller on the page than the box
allows.

GISAID-derived: the panels this places are rendered from mat-pcp counts and
live under the artifact root. Nothing here reads or writes data.
"""

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.figure
import matplotlib.image as mpimg
import matplotlib.lines
import matplotlib.text
import numpy as np

# A panel's type is sized in whatever units its renderer uses, and what the
# label has to match is the size that type ends up on the *page*. Both
# renderers we use come down to the same conversion: a panel's title occupies
# some number of pixels on disk, and those pixels land at the solved dpi.
#
#   Vega-Lite, saved by `save_panel` at `ppi`: a title of N "px" in the spec
#   is N * ppi/72 pixels on disk.
#   matplotlib, saved by `savefig(dpi=D)`: a title of N pt is N * D/72 pixels
#   on disk.
#
# so both are `title_px_on_disk`, and the physical size follows from the dpi.
VEGA_PANEL_TITLE_PX = 14
VEGA_WRITE_PPI = 300

#: The vector formats every composed figure is written in, in the order
#: `save_figure` returns them. The PDF is what `main.tex` includes.
VECTOR_FORMATS = ("pdf", "svg")

# What the composed figure is rendered at for display in the notebook. The
# files on disk are written at the solved layout dpi instead; see `compose`.
SCREEN_DPI = 150

#: Stroke width of a `row_groups` bracket. A hairline: it is a grouping mark,
#: not data, and at the sizes these figures print a heavier rule reads as an
#: axis.
BRACKET_LW_PT = 0.8


def vega_title_px_on_disk(
    title_px: float = VEGA_PANEL_TITLE_PX, write_ppi: float = VEGA_WRITE_PPI
) -> float:
    """Disk pixels taken by a Vega-Lite title of `title_px` at `write_ppi`."""
    return title_px * write_ppi / 72.0


def mpl_title_px_on_disk(title_pt: float, write_dpi: float) -> float:
    """Disk pixels taken by a matplotlib title of `title_pt` at `write_dpi`."""
    return title_pt * write_dpi / 72.0


@dataclass(frozen=True)
class Spacing:
    """Margins and gaps, in inches.

    Inches rather than figure fractions, so a panel changing size upstream
    moves its neighbours without changing the gaps between them.

    Both gaps are about one line of text. Each panel already carries its own
    title and axis labels, so a wider gutter does not separate them any
    better -- it just reads as two figures that happened to be printed
    together.
    """

    # The strip reserved to the left of each panel for its letter. Narrow
    # enough that the letter sits tight against the panel it names and
    # overlaps that panel's own left margin rather than standing clear of it
    # -- the whitespace inside a panel's axes area is free space. Still wide
    # enough that a letter cannot reach the panel before it, which is what
    # once put "C" on top of panel B's title.
    label_w_in: float = 0.16
    # Added to `label_w_in` to make the left margin, so the letters have the
    # same clearance from the page edge that the right side has.
    margin_pad_in: float = 0.06
    margin_right_in: float = 0.06
    margin_top_in: float = 0.06
    margin_bottom_in: float = 0.06
    col_gap_in: float = 0.18
    row_gap_in: float = 0.18
    # Where a letter goes relative to its panel's top-left corner: inside the
    # reserved strip, a hair below the top edge.
    label_dy_in: float = 0.02

    # The gutter `compose`'s `row_groups` draws into, right of the panels: a
    # gap, the bracket's own depth, a gap, then the group's name. Only
    # `group_text_w_in` is a guess about type rather than a measured length,
    # so it is the one `compose` checks against what it actually drew.
    group_gap_in: float = 0.10
    group_arm_in: float = 0.07
    group_text_gap_in: float = 0.06
    group_text_w_in: float = 0.60

    @property
    def group_gutter_in(self) -> float:
        """Total width reserved right of the panels for one bracket."""
        return (
            self.group_gap_in
            + self.group_arm_in
            + self.group_text_gap_in
            + self.group_text_w_in
        )

    @property
    def margin_left_in(self) -> float:
        return self.margin_pad_in + self.label_w_in

    def scaled(self, scale: float) -> "Spacing":
        """Every length multiplied by `scale`, so a shrink stays uniform."""
        return replace(
            self,
            label_w_in=self.label_w_in * scale,
            margin_pad_in=self.margin_pad_in * scale,
            margin_right_in=self.margin_right_in * scale,
            margin_top_in=self.margin_top_in * scale,
            margin_bottom_in=self.margin_bottom_in * scale,
            col_gap_in=self.col_gap_in * scale,
            row_gap_in=self.row_gap_in * scale,
            label_dy_in=self.label_dy_in * scale,
            group_gap_in=self.group_gap_in * scale,
            group_arm_in=self.group_arm_in * scale,
            group_text_gap_in=self.group_text_gap_in * scale,
            group_text_w_in=self.group_text_w_in * scale,
        )


@dataclass(frozen=True)
class Composed:
    """A placed figure and the numbers worth printing above it."""

    figure: matplotlib.figure.Figure
    dpi: float
    width_in: float
    height_in: float
    title_pt: float
    label_pt: float

    def summary(self) -> str:
        return (
            f"figure: {self.width_in:.2f} x {self.height_in:.2f} in, "
            f"panels at {self.dpi:.0f} dpi; panel titles ~{self.title_pt:.1f} pt, "
            f"labels {self.label_pt:.1f} pt"
        )


def missing_panels(panel_dir: Path, stems: Sequence[str]) -> list[str]:
    """Which of `stems` has no PNG in `panel_dir` yet."""
    return [stem for stem in stems if not (panel_dir / f"{stem}.png").exists()]


def load_panels(
    panel_dir: Path, panel_files: Mapping[str, str]
) -> dict[str, np.ndarray]:
    """Read `{letter: stem}` as `{letter: image}` from `panel_dir`."""
    return {
        letter: mpimg.imread(panel_dir / f"{stem}.png")
        for letter, stem in panel_files.items()
    }


def pad_to_common_width(
    panels: MutableMapping[str, np.ndarray], keys: Sequence[str]
) -> int:
    """Widen each of `keys` to the widest of them, with transparent pixels.

    For a figure whose rows are placed left justified. A row's second panel
    starts wherever its first one ends, so panels in the *first* column that
    are not the same width put every panel in the second column at a
    different x -- and a column of panels that do not share a left edge does
    not read as a column.

    The widths differ for reasons that belong to the panel and not to the
    figure: `fig:within_clade_shifts`' three scatters draw their callout
    labels outside the plot where they fit and inside it where they do not,
    so 22D's is the widest of the three by the width of a label column.
    Padding is the fix that leaves both panels alone -- the pixels added are
    transparent, so the panel moves without anything being drawn, and
    nothing in either renderer has to know what it is being composed beside.

    Mutates `panels` in place and returns the common width, in pixels.
    """
    width = max(panels[key].shape[1] for key in keys)
    for key in keys:
        image = panels[key]
        short = width - image.shape[1]
        if short > 0:
            panels[key] = np.concatenate(
                [
                    image,
                    np.zeros((image.shape[0], short, image.shape[2]), image.dtype),
                ],
                axis=1,
            )
    return width


#: The spacing every composed figure uses unless a caller passes its own.
#: A module-level singleton rather than a `Spacing()` default argument:
#: `Spacing` is frozen so the call would be harmless, but a mutable default
#: built at import is the bug ruff's B008 is written against, and the rule
#: does not read the dataclass to find out.
DEFAULT_SPACING = Spacing()


def compose(
    panels: Mapping[str, np.ndarray],
    rows: Sequence[Sequence[str | Sequence[str]]],
    *,
    box_in: tuple[float, float] = (7.5, 8.0),
    scale: float = 1.0,
    spacing: Spacing = DEFAULT_SPACING,
    title_px_on_disk: float = vega_title_px_on_disk(),
    label_title_ratio: float = 1.7,
    labels: Mapping[str, str] | None = None,
    row_groups: Sequence[tuple[str, Sequence[str]]] | None = None,
) -> Composed:
    """Lay `panels` out row by row and return the figure.

    Rows are left justified and their panels top aligned, so the left edge
    and each row's top edge are the lines the eye follows. A row narrower
    than the widest one leaves whitespace on the right rather than being
    stretched to meet it.

    Args:
        panels: `{letter: image}`, as `load_panels` returns.
        rows: Cells per row, top to bottom and left to right. A cell is
            a panel letter, or a list of letters to stack vertically in
            one column.
        box_in: `(width, height)` the composition has to fit inside.
        scale: Uniform shrink, applied to the box and to every margin and
            gap alike. The figure fits the box exactly in one dimension, so
            this is the only way to make it smaller on the page.
        spacing: Margins and gaps at `scale = 1`.
        title_px_on_disk: What one panel's title measures on disk, which is
            what the panel letters are sized against. See the module header
            for the two renderers' conversions.
        label_title_ratio: Letter size as a multiple of the title's size on
            the page. Derived rather than hardcoded because a title's
            physical size moves whenever the layout does, and a fixed label
            size silently changes its relationship to the titles each time.
        labels: Optional `{key: letter}` overriding what is drawn beside each
            panel, where `panels`' keys are otherwise used directly. An empty
            string draws nothing, which is what a panel continuing the one
            above it needs: a figure whose panel A is two rows of scatters
            is still one panel, and lettering the second row would make it
            two.
        row_groups: Optional `[(name, letters), ...]`, drawn as a bracket
            right of the panels spanning those panels, with `name` beside
            it. For a figure whose rows come in groups that share something
            the panel letters cannot say -- three DMS experiments, two rows
            each. The bracket spans from the top of the group's highest
            panel to the bottom of its lowest, so the letters it covers
            must be contiguous down the page; `compose` does not reorder
            them. Groups reserve `Spacing.group_gutter_in` on the right
            whether or not the names are short enough to need it, so a
            figure with one group is the same width as a figure with four.

    Returns:
        The figure, with the dpi and type sizes it was placed at.
    """
    import matplotlib.pyplot as plt

    gaps = spacing.scaled(scale)
    box_w_in, box_h_in = box_in[0] * scale, box_in[1] * scale

    # A row holds cells, and a cell is either one panel or a short column of
    # them: `[["A"], [["B", "C"], "D"]]` is A across the top, then B above C
    # with D beside them. A column exists so that two panels of one width can
    # sit against one taller panel without the taller one pushing whatever
    # comes after it down the page.
    grid = [
        [list(cell) if isinstance(cell, (list, tuple)) else [cell] for cell in row]
        for row in rows
    ]

    # Everything that does not scale with the dpi: margins and gaps are in
    # inches by choice, so they stay the same thickness whatever size the
    # panels end up. A column's internal gaps are counted against the row
    # that holds the deepest column, which is the row they are spent in.
    groups = [(name, list(letters)) for name, letters in (row_groups or ())]
    _grouped: list[str] = [letter for _name, letters in groups for letter in letters]
    _unknown = [letter for letter in _grouped if letter not in panels]
    if _unknown:
        raise KeyError(f"row_groups name panels that were not passed: {_unknown}")
    if len(set(_grouped)) != len(_grouped):
        raise ValueError(f"row_groups overlap: {_grouped}")

    # Reserved whether or not a name needs the whole width, so that the
    # panels sit at one size regardless of what the groups are called.
    gutter_in = gaps.group_gutter_in if groups else 0.0

    widest = max(len(row) for row in grid)
    fixed_w_in = (
        gaps.margin_left_in
        + gaps.margin_right_in
        + gutter_in
        + (gaps.col_gap_in + gaps.label_w_in) * (widest - 1)
    )
    stacked_gaps = sum(max(len(cell) - 1 for cell in row) for row in grid)
    fixed_h_in = (
        gaps.margin_top_in
        + gaps.margin_bottom_in
        + gaps.row_gap_in * (len(grid) - 1 + stacked_gaps)
    )
    # The panel pixels that have to fit in what is left. The smallest dpi
    # whose figure still fits, i.e. the largest the figure can be; whichever
    # of width and height binds, sets it.
    panel_w_px = max(
        sum(max(panels[c].shape[1] for c in cell) for cell in row) for row in grid
    )
    panel_h_px = sum(
        max(sum(panels[c].shape[0] for c in cell) for cell in row) for row in grid
    )
    dpi = max(
        panel_w_px / (box_w_in - fixed_w_in),
        panel_h_px / (box_h_in - fixed_h_in),
    )

    title_pt = title_px_on_disk * 72.0 / dpi
    label_pt = label_title_ratio * title_pt

    def panel_size_in(letter: str) -> tuple[float, float]:
        image = panels[letter]
        return image.shape[1] / dpi, image.shape[0] / dpi

    def cell_size_in(cell: Sequence[str]) -> tuple[float, float]:
        sizes = [panel_size_in(letter) for letter in cell]
        return (
            max(w for w, _ in sizes),
            sum(h for _, h in sizes) + gaps.row_gap_in * (len(cell) - 1),
        )

    def row_extent_in(row) -> tuple[float, float]:
        sizes = [cell_size_in(cell) for cell in row]
        width = sum(w for w, _ in sizes) + (gaps.col_gap_in + gaps.label_w_in) * (
            len(row) - 1
        )
        return width, max(h for _, h in sizes)

    extents = [row_extent_in(row) for row in grid]
    content_w_in = max(w for w, _ in extents)
    content_h_in = sum(h for _, h in extents) + gaps.row_gap_in * (len(grid) - 1)
    fig_w_in = gaps.margin_left_in + content_w_in + gutter_in + gaps.margin_right_in
    fig_h_in = gaps.margin_top_in + content_h_in + gaps.margin_bottom_in
    assert fig_w_in <= box_w_in + 1e-6 and fig_h_in <= box_h_in + 1e-6, (
        f"{fig_w_in:.2f} x {fig_h_in:.2f} in does not fit "
        f"{box_w_in:.2f} x {box_h_in:.2f}"
    )

    # The figure's own dpi is for the screen only; `save_figure` passes the
    # solved one explicitly. They have to be separate: at 930 dpi an 8-inch
    # figure renders to a 60-megapixel PNG, which marimo refuses to embed
    # ("your output is too large"), and nothing about the layout needs it --
    # type is in points and the axes are placed in figure fractions, so the
    # screen copy is the same picture at fewer pixels.
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=SCREEN_DPI)

    drawn_label = dict(labels or {})
    # Where each panel ended up, for the group brackets to span. Filled by
    # `place`, which is the only thing that knows a panel's final box.
    placed: dict[str, tuple[float, float, float]] = {}

    def place(letter: str, x_in: float, top_in: float) -> float:
        """Draw one panel with its letter; return the height it took."""
        w_in, h_in = panel_size_in(letter)
        placed[letter] = (x_in + w_in, top_in, top_in - h_in)
        bottom_in = top_in - h_in
        ax = fig.add_axes(
            (
                x_in / fig_w_in,
                bottom_in / fig_h_in,
                w_in / fig_w_in,
                h_in / fig_h_in,
            )
        )
        ax.imshow(panels[letter])
        ax.axis("off")
        text = drawn_label.get(letter, letter)
        if text:
            ax.text(
                (x_in - gaps.label_w_in) / fig_w_in,
                (top_in + gaps.label_dy_in) / fig_h_in,
                text,
                transform=fig.transFigure,
                fontsize=label_pt,
                fontweight="bold",
                va="top",
                ha="left",
            )
        return h_in

    # Walk the rows top-down, each row's cells left to right and top aligned,
    # so every cell in a row starts at the row's top edge and its first
    # panel's title and letter sit on one line across the row.
    y_top_in = fig_h_in - gaps.margin_top_in
    for row, (_row_w_in, row_h_in) in zip(grid, extents):
        x_in = gaps.margin_left_in
        for cell in row:
            cell_top_in = y_top_in
            for letter in cell:
                cell_top_in -= place(letter, x_in, cell_top_in) + gaps.row_gap_in
            x_in += cell_size_in(cell)[0] + gaps.col_gap_in + gaps.label_w_in
        y_top_in -= row_h_in + gaps.row_gap_in

    # The brackets, once every panel has a box. Each spans its group from the
    # top of the highest panel to the bottom of the lowest, and is drawn in
    # figure fractions like the panel letters are. The bracket's left edge is
    # the same for every group -- the right edge of the widest panel -- so the
    # names line up in a column rather than following each row's ragged end.
    if groups:
        drawn: list[matplotlib.text.Text] = []
        content_right_in = max(right for right, _top, _bottom in placed.values())
        arm_x_in = content_right_in + gaps.group_gap_in
        tip_x_in = arm_x_in + gaps.group_arm_in
        text_x_in = tip_x_in + gaps.group_text_gap_in
        for name, letters in groups:
            boxes = [placed[letter] for letter in letters]
            top_in = max(top for _right, top, _bottom in boxes)
            bottom_in = min(bottom for _right, _top, bottom in boxes)
            fig.add_artist(
                matplotlib.lines.Line2D(
                    [
                        arm_x_in / fig_w_in,
                        tip_x_in / fig_w_in,
                        tip_x_in / fig_w_in,
                        arm_x_in / fig_w_in,
                    ],
                    [
                        top_in / fig_h_in,
                        top_in / fig_h_in,
                        bottom_in / fig_h_in,
                        bottom_in / fig_h_in,
                    ],
                    transform=fig.transFigure,
                    color="black",
                    linewidth=BRACKET_LW_PT,
                    solid_capstyle="projecting",
                    solid_joinstyle="miter",
                )
            )
            drawn.append(
                fig.text(
                    text_x_in / fig_w_in,
                    (top_in + bottom_in) / 2.0 / fig_h_in,
                    name,
                    fontsize=title_pt,
                    fontweight="bold",
                    va="center",
                    ha="left",
                )
            )

        # **Checked, not corrected**, as the panel widths are: the gutter was
        # reserved from `group_text_w_in`, a guess about type, and a name that
        # outgrows it runs off the page rather than wrapping. Widening it here
        # would move every panel to fit one long name, silently, so this says
        # which name and by how much and leaves the choice to the caller.
        _renderer = fig.canvas.get_renderer()
        _over = {
            text.get_text(): text.get_window_extent(_renderer).width / SCREEN_DPI
            for text in drawn
            if text.get_window_extent(_renderer).width / SCREEN_DPI
            > gaps.group_text_w_in + 1e-6
        }
        if _over:
            raise ValueError(
                f"group names do not fit the {gaps.group_text_w_in:.2f} in "
                f"reserved by `Spacing.group_text_w_in` (scaled): "
                + ", ".join(f"{k!r} needs {v:.2f} in" for k, v in _over.items())
            )

    return Composed(
        figure=fig,
        dpi=dpi,
        width_in=fig_w_in,
        height_in=fig_h_in,
        title_pt=title_pt,
        label_pt=label_pt,
    )


def save_figure(
    composed: Composed,
    out_dir: Path,
    stem: str,
    *,
    tex_figure_dir: Path | None = None,
) -> list[Path]:
    """Write PNG and vector copies, and put the PDF in the manuscript.

    Copying the PDF is part of saving rather than a step someone remembers:
    a stale `figures/<stem>.pdf` in the paper is the one failure mode this
    pipeline exists to prevent. A manuscript that is not checked out beside
    this repo is reported, not an error.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    composed.figure.savefig(png, dpi=composed.dpi, bbox_inches=None)
    # PDF and SVG beside the PNG. The composition is raster -- the panels
    # arrive as pixels -- but the chrome this module draws, the panel letters,
    # stays vector, and the PDF is what the manuscript includes.
    written = [png]
    for extension in VECTOR_FORMATS:
        vector = out_dir / f"{stem}.{extension}"
        composed.figure.savefig(vector, dpi=composed.dpi, bbox_inches=None)
        written.append(vector)

    if tex_figure_dir is not None and tex_figure_dir.parent.is_dir():
        import shutil

        pdf = next(p for p in written if p.suffix == ".pdf")
        tex_figure_dir.mkdir(exist_ok=True)
        shutil.copy2(pdf, tex_figure_dir / pdf.name)
        written.append(tex_figure_dir / pdf.name)
    return written
