"""Codon-optimality tables for sub-protein selection in neutral codon probabilities.

This module provides infrastructure to load a measured codon-optimality table
(e.g. CSC — Codon Stability Coefficient), split it into within-AA and
between-AA components, and produce the log-space tensor that multiplies the
neutral codon probabilities.

The within-AA component is the identifiable part (Proposition 1(i) of
dasm2-tex): it reshuffles synonymous codons and is identified by synonymous
substitutions. The between-AA levels are unidentifiable from sequence data
(Proposition 1(ii)) and are kept as metadata for optional post-hoc
attribution only.

Two measured-optimality axes are supported (issue #747):

- **CSC** (codon stability coefficient) via :func:`load_csc_table`.
- **tAI** (tRNA adaptation index) via :func:`load_tai_table`.

:func:`build_codon_pref_log` combines one or more axes at per-axis strengths
``β_k`` into the 64-dim log-preference ``Σ_k β_k · ξ̃ᵏ``. The strengths can be
caller-supplied constants (e.g. the ``β=1`` smoke test) or fit jointly with the
model by the ``CodonStrengthOptimizer`` between-cycle hook
(:mod:`dasm2.training.hooks.codon_strength`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import torch
from torch import Tensor

from dasm2.codon_table import (
    AA_TO_SENSE_CODONS,
    CODON64_TO_SENSE61,
    NUM_AA,
    NUM_SENSE_CODONS,
    SENSE_CODON_DIFF_I,
    SENSE_CODON_DIFF_J,
    SENSE_CODON_LIST_TENSOR,
)
from dasm2.sequence.constants import CODON_TO_IDX

#: The two ways a codon preference vector is injected into the neutral codon
#: distribution (issue #930). See :func:`apply_codon_pref_log`.
CodonPrefForm = Literal["divide_z", "ratio"]


@dataclass(frozen=True)
class CodonPreferenceTable:
    """A single codon-optimality axis with within-AA and between-AA components.

    Attributes:
        name: Human-readable axis name (e.g. "CSC", "tAI").
        source: Provenance string (URL, DOI, file path).
        within_aa_log: Within-AA log-optimality reference ξ̃ᵏ for this axis
            k, shape (61,). Normalized so that within each amino acid the
            normalized optimality values have geometric mean 1 (i.e. the log
            values sum to 0 within each AA's synonymous family).
        between_aa_log: Between-AA log-optimality ξ̄ᵏ, shape (20,).
            Per-AA mean of log(raw_optimality) — stripped by the within-AA
            normalization. Kept for attribution only; never enters the
            likelihood.
        raw_sense61: Original raw values in sense-61 space, shape (61,).
    """

    name: str
    source: str
    within_aa_log: Tensor  # (61,)
    between_aa_log: Tensor  # (20,)
    raw_sense61: Tensor  # (61,)


def _to_positive_optimality(raw: Tensor) -> Tensor:
    """Convert raw optimality values (which may be negative) to positive weights.

    CSC values are Pearson correlations in roughly [-0.15, +0.13] (the
    checked-in six-assay mean). We use exp(raw) as the natural map to positive
    reals: it preserves ordering, avoids arbitrary shifts, and produces moderate
    within-AA ratios (e.g. across the six Leu codons exp(0.116)/exp(-0.076) ≈ 1.2).
    """
    return torch.exp(raw)


def _within_aa_normalize(positive_vals: Tensor) -> tuple[Tensor, Tensor]:
    """Split positive optimality values into within-AA and between-AA parts.

    For each amino acid, normalize the synonymous codons to have geometric
    mean 1 (i.e. mean-0 in log space). This separates:
    - within_aa_log: the identifiable within-AA shape (mean-0 per AA in log)
    - between_aa_log: the per-AA geometric mean (unidentifiable level)

    Args:
        positive_vals: Strictly positive optimality values, shape (61,).

    Returns:
        (within_aa_log, between_aa_log) where:
        - within_aa_log is shape (61,), sums to 0 within each AA family
        - between_aa_log is shape (20,), the per-AA log-mean
    """
    log_vals = torch.log(positive_vals)
    within_aa_log = torch.zeros_like(log_vals)
    between_aa_log = torch.zeros(NUM_AA, dtype=log_vals.dtype)

    for aa_idx in range(NUM_AA):
        sense_indices = AA_TO_SENSE_CODONS[aa_idx]
        if len(sense_indices) == 0:
            continue
        idx = torch.tensor(sense_indices, dtype=torch.long)
        aa_log_vals = log_vals[idx]
        mean_log = aa_log_vals.mean()
        between_aa_log[aa_idx] = mean_log
        within_aa_log[idx] = aa_log_vals - mean_log

    return within_aa_log, between_aa_log


def _read_sense61_column(path: Path, value_col: str) -> Tensor:
    """Read a 'codon' + ``value_col`` TSV into a sense-61-indexed tensor.

    The file must list exactly the 61 sense codons (stop codons excluded).
    Lines starting with '#' are comments.

    Raises:
        ValueError: On missing columns, unknown/stop/duplicate codons, or a
            codon count != 61.
    """
    import pandas as pd  # heavy optional dependency; imported lazily

    df = pd.read_csv(path, sep="\t", comment="#")
    if "codon" not in df.columns or value_col not in df.columns:
        raise ValueError(
            f"TSV must have 'codon' and '{value_col}' columns, got {list(df.columns)}"
        )

    raw_sense61 = torch.zeros(NUM_SENSE_CODONS, dtype=torch.float64)
    seen = set()
    for _, row in df.iterrows():
        codon = row["codon"].upper().strip()
        value = float(row[value_col])
        if codon not in CODON_TO_IDX:
            raise ValueError(f"Unknown codon: {codon}")
        codon64_idx = CODON_TO_IDX[codon]
        sense61_idx = CODON64_TO_SENSE61[codon64_idx]
        if sense61_idx == -1:
            raise ValueError(f"Stop codon {codon} should not appear in the table")
        if sense61_idx in seen:
            raise ValueError(f"Duplicate codon: {codon}")
        seen.add(sense61_idx)
        raw_sense61[sense61_idx] = value

    if len(seen) != NUM_SENSE_CODONS:
        missing_count = NUM_SENSE_CODONS - len(seen)
        raise ValueError(
            f"Expected {NUM_SENSE_CODONS} sense codons, got {len(seen)} "
            f"({missing_count} missing)"
        )
    return raw_sense61


def _load_optimality_table(
    path: Path,
    *,
    name: str,
    source: str,
    value_col: str,
    to_positive: Callable[[Tensor], Tensor],
) -> CodonPreferenceTable:
    """Load a measured codon-optimality TSV into a :class:`CodonPreferenceTable`.

    Args:
        path: Path to the TSV (``codon`` + ``value_col`` columns).
        name: Axis name (e.g. ``"CSC"``, ``"tAI"``).
        source: Provenance string recorded on the table.
        value_col: Name of the optimality-value column.
        to_positive: Map from the raw (61,) values to strictly positive
            optimality weights, before the within-AA log decomposition.
    """
    raw_sense61 = _read_sense61_column(path, value_col)
    positive = to_positive(raw_sense61)
    within_aa_log, between_aa_log = _within_aa_normalize(positive)
    return CodonPreferenceTable(
        name=name,
        source=source,
        within_aa_log=within_aa_log.float(),
        between_aa_log=between_aa_log.float(),
        raw_sense61=raw_sense61.float(),
    )


def load_csc_table(path: Path) -> CodonPreferenceTable:
    """Load a CSC (Codon Stability Coefficient) table from a TSV file.

    The file must have columns 'codon' and 'csc', with 61 sense codons
    (stop codons excluded). Lines starting with '#' are comments. CSC values
    are Pearson correlations (the checked-in six-assay mean runs roughly
    [-0.15, +0.13]); they are mapped to positive optimality weights via ``exp``
    (see :func:`_to_positive_optimality`).

    Args:
        path: Path to the TSV file.

    Returns:
        A CodonPreferenceTable with within-AA and between-AA decomposition.

    Raises:
        ValueError: If the file doesn't contain exactly the 61 sense codons.
    """
    return _load_optimality_table(
        path,
        name="CSC",
        source=f"Wu et al. 2019, eLife, doi:10.7554/eLife.45396; loaded from {path}",
        value_col="csc",
        to_positive=_to_positive_optimality,
    )


def load_tai_table(path: Path) -> CodonPreferenceTable:
    """Load a tAI (tRNA adaptation index) table from a TSV file.

    The file must have columns 'codon' and 'tai', with 61 sense codons. tAI
    values are relative-adaptiveness weights ``w_c`` that are already strictly
    positive (in (0, 1]), so the within-AA decomposition takes ``log(w_c)``
    directly — no ``exp`` map (contrast :func:`load_csc_table`).

    The checked-in human table
    (``data/codon_optimality/human_tai_dosreis2004.tsv``) is built by
    ``scripts/build_human_tai_table.py`` from GtRNAdb hg38 tRNA gene copy
    numbers and the dos Reis 2004 optimized wobble s-values; the exact source
    and version are recorded in the TSV header.

    Args:
        path: Path to the TSV file.

    Returns:
        A CodonPreferenceTable with within-AA and between-AA decomposition.

    Raises:
        ValueError: If the file doesn't contain exactly the 61 sense codons.
    """
    return _load_optimality_table(
        path,
        name="tAI",
        source=(
            "dos Reis et al. 2004, NAR, doi:10.1093/nar/gkh834; "
            f"GtRNAdb hg38 tRNA gene copy numbers; loaded from {path}"
        ),
        value_col="tai",
        to_positive=lambda raw: raw,  # tAI weights are already positive
    )


def build_codon_pref_log(
    tables: list[CodonPreferenceTable],
    strengths: Optional[list[float]] = None,
) -> Tensor:
    """Build the 64-dim log codon preference vector from one or more axes.

    Computes: codon_pref_log[c] = Σ_k β_k · ξ̃ᵏ_c

    where ξ̃ᵏ_c is the within-AA log-optimality of axis k for codon c,
    and β_k is the strength (default 1.0 for all axes).

    The result is a (64,) tensor in log-space: stop codon entries are 0.0
    (neutral in log-space, i.e. multiplier = 1.0; stops are separately
    zapped downstream by STOP_CODON_ZAPPER).

    Args:
        tables: List of CodonPreferenceTable objects.
        strengths: Per-axis strengths β_k. If None, all strengths are 1.0.

    Returns:
        Tensor of shape (64,) with log-space within-AA codon preferences.
    """
    if not tables:
        raise ValueError("At least one table is required")
    if strengths is not None and len(strengths) != len(tables):
        raise ValueError(f"Got {len(strengths)} strengths for {len(tables)} tables")

    # Accumulate in sense-61 space
    combined = torch.zeros(NUM_SENSE_CODONS, dtype=torch.float32)
    for k, table in enumerate(tables):
        beta = 1.0 if strengths is None else strengths[k]
        combined = combined + beta * table.within_aa_log

    return lift_sense61_to_64(combined)


def lift_sense61_to_64(vec_sense61: Tensor) -> Tensor:
    """Lift a (61,) sense-codon vector to a (64,) codon vector.

    Stop-codon entries are set to 0.0 (neutral in log-space). Preserves the
    input dtype/device and autograd graph, so it can lift a tensor that
    depends on free strength parameters.

    Args:
        vec_sense61: Shape (61,) sense-codon values (e.g. ``Σ_k β_k·ξ̃ᵏ``).

    Returns:
        Shape (64,) tensor with sense entries scattered to their 64-indices.
    """
    if vec_sense61.shape != (NUM_SENSE_CODONS,):
        raise ValueError(
            f"Expected shape ({NUM_SENSE_CODONS},), got {tuple(vec_sense61.shape)}"
        )
    log64 = torch.zeros(64, dtype=vec_sense61.dtype, device=vec_sense61.device)
    idx = SENSE_CODON_LIST_TENSOR.to(vec_sense61.device)
    return log64.index_copy(0, idx, vec_sense61)


def single_nt_synonymous_reachable_mask() -> Tensor:
    """Boolean (61,) mask of sense codons reachable from a synonymous codon by
    one nucleotide substitution.

    These are the codons whose within-AA optimality enters the synonymous
    likelihood signal that point-identifies the strengths (Proposition 1(i)).
    Single-codon amino acids (Met, Trp) are False.
    """
    aa_of = _sense61_aa_indices()
    syn = aa_of[SENSE_CODON_DIFF_I] == aa_of[SENSE_CODON_DIFF_J]
    mask = torch.zeros(NUM_SENSE_CODONS, dtype=torch.bool)
    mask[SENSE_CODON_DIFF_I[syn]] = True
    return mask


def _sense61_aa_indices() -> Tensor:
    """Return (61,) amino-acid index for each sense codon."""
    aa_of = torch.full((NUM_SENSE_CODONS,), -1, dtype=torch.long)
    for aa_idx in range(NUM_AA):
        for sense_idx in AA_TO_SENSE_CODONS[aa_idx]:
            aa_of[sense_idx] = aa_idx
    return aa_of


@dataclass(frozen=True)
class DesignConditioning:
    """Conditioning report for the multi-axis strength regression.

    Restricted to single-nt-reachable synonymous codons — the subset that
    point-identifies the strengths (Proposition 1(i)).

    Attributes:
        n_codons: Number of codons in the restricted design.
        correlations: Pairwise Pearson correlations between axes,
            ``{(name_i, name_j): r}``. Empty for a single axis.
        condition_number: σ_max / σ_min of the (n_codons, n_axes) design
            matrix. 1.0 for a single (nonzero) axis; large when axes are
            collinear.
        axis_names: Axis names in column order.
    """

    n_codons: int
    correlations: dict
    condition_number: float
    axis_names: list


def design_conditioning(tables: list[CodonPreferenceTable]) -> DesignConditioning:
    """Report collinearity and condition number of the strength design matrix.

    The design columns are the within-AA references ``ξ̃ᵏ`` of each axis,
    restricted to single-nt-reachable synonymous codons (the subset that
    actually pins the strengths). Collinearity between CSC and tAI does not
    block the aggregate fit, but it destabilizes the per-axis split, so it is
    surfaced explicitly rather than hidden (issue #747).

    Args:
        tables: Codon-optimality axes (in strength-vector column order).

    Returns:
        A :class:`DesignConditioning` report.
    """
    if not tables:
        raise ValueError("At least one table is required")
    mask = single_nt_synonymous_reachable_mask()
    cols = [t.within_aa_log[mask].double() for t in tables]
    design = torch.stack(cols, dim=1)  # (M, K)

    correlations: dict = {}
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            xi, xj = cols[i], cols[j]
            xi_c = xi - xi.mean()
            xj_c = xj - xj.mean()
            denom = xi_c.norm() * xj_c.norm()
            r = float((xi_c @ xj_c) / denom) if denom > 0 else float("nan")
            correlations[(tables[i].name, tables[j].name)] = r

    svals = torch.linalg.svdvals(design)
    smin = float(svals.min())
    cond = float(svals.max() / smin) if smin > 0 else float("inf")

    return DesignConditioning(
        n_codons=int(mask.sum()),
        correlations=correlations,
        condition_number=cond,
        axis_names=[t.name for t in tables],
    )


def apply_codon_pref_log(
    log_neutral_codon_probs: Tensor,
    codon_pref_log: Tensor,
    *,
    parent_codon_idxs: Optional[Tensor] = None,
    form: CodonPrefForm = "divide_z",
) -> Tensor:
    """Apply codon preference to log neutral codon probabilities.

    Two forms (issue #930):

    - ``"divide_z"`` (default, the #749/#767 merged form): add the log
      preference, then renormalize with a softmax over all 64 codons
      (``÷ Z``, ``Z = Σ_k p_k·g_k``). Not a cycle-consistent potential — ``Z``
      depends on the neutral row (branch length, parent) — but agrees with the
      ratio form on the identifiable within-box synonymous signal.
    - ``"ratio"`` (issue #930 Rung 1): add the log preference gauged by the
      *parent codon's own* preference, i.e. ``g_{c'}/g_{x̄_j}`` for child ``c'``
      and parent codon ``x̄_j``. This is the cycle-consistent form the DASM
      likelihood uses (``ℓ = p·(g_{c'}/g_{x̄_j})·f``): a global shift of
      ``codon_pref_log`` is an exactly flat direction, and the output is
      **not** renormalized to sum to 1 — the 64-way normalization comes from
      the caller's ``compute_log_adjusted_codon_probs`` step, not from this
      primitive. That step is configurable since issue #1238 (parent-absorb
      or a sense-restricted ``log_softmax``); the flat-direction claim holds
      under either, since both denominators scale with a global shift.
      Requires ``parent_codon_idxs``.

    Args:
        log_neutral_codon_probs: Shape (..., 64) log neutral codon probabilities.
        codon_pref_log: Shape (64,) log codon preference from build_codon_pref_log.
        parent_codon_idxs: Shape (...,) flat codon index (0-63) of the parent at
            each position, matching ``log_neutral_codon_probs``'s leading dims.
            Required (and only used) when ``form="ratio"``.
        form: ``"divide_z"`` (default) or ``"ratio"``.

    Returns:
        Adjusted log neutral codon probabilities, shape (..., 64). Renormalized
        per position for ``"divide_z"``; **not** renormalized for ``"ratio"``.

    Raises:
        ValueError: If ``form`` is unrecognized, or ``form="ratio"`` is given
            without ``parent_codon_idxs``.
    """
    pref = codon_pref_log.to(log_neutral_codon_probs.device)

    if form == "ratio":
        if parent_codon_idxs is None:
            raise ValueError("form='ratio' requires parent_codon_idxs")
        parent_pref = pref[parent_codon_idxs].unsqueeze(-1)
        return log_neutral_codon_probs + pref - parent_pref
    elif form == "divide_z":
        # Add preference in log-space
        adjusted = log_neutral_codon_probs + pref

        # Renormalize: log_softmax over the codon dimension
        # But we need to handle -inf entries (masked/stop codons) carefully.
        # Use logsumexp for renormalization, masking out -inf entries.
        log_sum = torch.logsumexp(adjusted, dim=-1, keepdim=True)
        return adjusted - log_sum
    else:
        raise ValueError(f"Unknown form: {form!r}. Use 'divide_z' or 'ratio'.")
