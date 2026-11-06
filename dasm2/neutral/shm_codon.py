"""SHM-1: context-free, non-reversible codon-level CTMC neutral model.

``ShmCodonNeutralModel`` wraps Erick's **SHM-1** matrix — a context-free
(no flanking dependence) but codon-level, *non-reversible* 61×61 empirical
codon rate matrix estimated from BCR out-of-frame (selection-neutral)
parent–child pairs. It is the ``phyz`` ``EmpCodon``/``SHM1`` preset expressed
as a dasm2 neutral model.

Why a codon CTMC rather than Thrifty? Thrifty was fit as a *Poisson* per-site
rate model, so reusing its rates as instantaneous CTMC generator entries is a
semantic mismatch that QDASM patches with the ``single_hit_scale=0.566``
fudge. SHM-1 *is* a CTMC generator, so it co-calibrates with QDASM's
``exp(tQ)`` Q construction natively — and being context-free, it drops the
1.3 GB Thrifty 13-mer lookup entirely.

The matrix file (``shm1_codon_q.dat``, phyz ``nonrev_codon`` format) is 61
whitespace-separated Q rows (row-major, dasm2 sense-61 codon order) followed
by 61 stationary frequencies, with ``#`` comment lines. It ships inside the
package under ``dasm2/neutral/data/``.

The QDASM ``shm1_codon`` Q variant consumes ``Q_codon`` *directly* (a constant
loss buffer); this class additionally implements the ``RatesCSPNeutralModel``
rate/CSP interface (derived from the single-nt codon neighbors) so the data
pipeline, branch-length, and eval paths behave like any other neutral model.
"""

from __future__ import annotations

import logging
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from dasm2.codon_table import (
    CODON64_TO_SENSE61,
    NUM_SENSE_CODONS,
    SENSE_CODON_DIFF_I,
    SENSE_CODON_DIFF_J,
    SENSE_CODON_DIFF_POS,
    SENSE_CODON_DIFF_TARGET_NT,
    SENSE_CODON_LIST,
)
from dasm2.neutral.base import RatesCSPNeutralModel
from dasm2.sequence.conversion import codon_idx_tensor_of_str_ambig

logger = logging.getLogger(__name__)

# Default SHM-1 matrix shipped with the package (see module docstring).
_SHM1_DAT_RESOURCE = "neutral/data/shm1_codon_q.dat"
SHM1_MODEL_NAME = "SHM1"

# Tolerances for validating the loaded matrix as a CTMC generator.
_ROW_SUM_ATOL = 1e-3
_OFF_DIAG_ATOL = 1e-6


def default_shm1_codon_q_path() -> Path:
    """Absolute path to the packaged ``shm1_codon_q.dat``."""
    return Path(str(files("dasm2").joinpath(_SHM1_DAT_RESOURCE)))


def resolve_shm1_codon_q_path(spec: Optional[Union[str, Path]]) -> Path:
    """Resolve a SHM-1 matrix spec to a concrete path.

    ``None`` → the packaged default. Otherwise the spec is used as-is; a
    ``FileNotFoundError`` is raised if it does not exist (fail fast rather
    than silently falling back to the default).
    """
    if spec is None:
        return default_shm1_codon_q_path()
    path = Path(spec).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"SHM-1 codon Q matrix not found at {path}. Provide a path to a "
            "phyz nonrev_codon .dat file, or omit to use the packaged default."
        )
    return path


def parse_shm1_codon_q_dat(
    path: Union[str, Path],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parse a phyz ``nonrev_codon`` .dat into ``(Q [61,61], freqs [61])``.

    Format: ``#``-prefixed comment lines are ignored; the remaining 62
    whitespace-separated numeric lines are the 61 Q rows (row-major) followed
    by the 61 stationary frequencies.
    """
    rows: List[List[float]] = []
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append([float(tok) for tok in stripped.split()])

    expected_lines = NUM_SENSE_CODONS + 1
    if len(rows) != expected_lines:
        raise ValueError(
            f"Expected {NUM_SENSE_CODONS} Q rows + 1 frequency row "
            f"({expected_lines} numeric lines) in {path}, got {len(rows)}."
        )
    widths = {len(r) for r in rows}
    if widths != {NUM_SENSE_CODONS}:
        raise ValueError(
            f"Every numeric line in {path} must have {NUM_SENSE_CODONS} "
            f"values; got line widths {sorted(widths)}."
        )

    data = torch.tensor(rows, dtype=torch.float32)
    q_codon = data[:NUM_SENSE_CODONS].contiguous()
    frequencies = data[NUM_SENSE_CODONS].contiguous()
    return q_codon, frequencies


def load_shm1_codon_q(spec: Optional[Union[str, Path]] = None) -> torch.Tensor:
    """Load just the ``[61, 61]`` SHM-1 generator (for the QDASM loss buffer)."""
    q_codon, _ = parse_shm1_codon_q_dat(resolve_shm1_codon_q_path(spec))
    return q_codon


class ShmCodonNeutralModel(RatesCSPNeutralModel):
    """Context-free, non-reversible codon CTMC neutral model (SHM-1).

    Holds a raw 61×61 codon generator ``Q_codon`` (dasm2 sense-61 order) and
    its stationary distribution. Neutral codon probabilities are the exact
    ``P(t) = exp(Q_codon · t)``; per-nt rates/CSPs are derived from the
    single-nt codon neighbours so the model also satisfies the
    ``RatesCSPNeutralModel`` contract used by the data pipeline.

    Example:
        >>> model = ShmCodonNeutralModel.from_dat()  # packaged SHM-1
        >>> Q = model.Q_codon                        # [61, 61]
        >>> P = model.compute_transition_matrix(0.1)  # [61, 61]
    """

    _EXPECTED_MODEL_TYPE = "ShmCodonNeutralModel"

    def __init__(
        self,
        q_codon: torch.Tensor,
        frequencies: torch.Tensor,
        *,
        name: str = SHM1_MODEL_NAME,
    ) -> None:
        """Initialize from a codon generator and stationary frequencies.

        Args:
            q_codon: ``[61, 61]`` codon rate matrix in dasm2 sense-61 order —
                off-diagonals ≥ 0, rows summing to ~0.
            frequencies: ``[61]`` stationary codon frequencies (sum to 1).
            name: Human-readable identifier.

        Raises:
            ValueError: If shapes or the generator/frequency constraints fail.
        """
        super().__init__()
        if q_codon.shape != (NUM_SENSE_CODONS, NUM_SENSE_CODONS):
            raise ValueError(
                f"q_codon must be ({NUM_SENSE_CODONS}, {NUM_SENSE_CODONS}), "
                f"got {tuple(q_codon.shape)}."
            )
        if frequencies.shape != (NUM_SENSE_CODONS,):
            raise ValueError(
                f"frequencies must be ({NUM_SENSE_CODONS},), "
                f"got {tuple(frequencies.shape)}."
            )
        q_codon = q_codon.to(torch.float32)
        row_sums = q_codon.sum(dim=-1)
        if row_sums.abs().max().item() > _ROW_SUM_ATOL:
            raise ValueError(
                "q_codon rows must sum to ~0 (valid CTMC generator); "
                f"max |row sum| = {row_sums.abs().max().item():.2e}."
            )
        off_diag = q_codon.clone()
        off_diag.fill_diagonal_(0.0)
        if off_diag.min().item() < -_OFF_DIAG_ATOL:
            raise ValueError(
                "q_codon off-diagonals must be non-negative; "
                f"min = {off_diag.min().item():.2e}."
            )

        self._q_codon = q_codon.clone()
        self._frequencies = frequencies.to(torch.float32).clone()
        self._name = name
        # Provenance, mirrors GTRCodonNeutralModel (set by loaders / from_catalog).
        self._dat_path: Optional[Path] = None
        self._catalog_name: Optional[str] = None
        self._archive_path: Optional[Path] = None

        self._build_rate_csp_lookups()
        # 64-space scatter target for compute_neutral_codon_probs.
        self._sense_codon_list = torch.tensor(SENSE_CODON_LIST, dtype=torch.long)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dat(
        cls,
        path: Optional[Union[str, Path]] = None,
        *,
        name: str = SHM1_MODEL_NAME,
    ) -> "ShmCodonNeutralModel":
        """Load from a phyz ``nonrev_codon`` .dat (``None`` → packaged SHM-1)."""
        resolved = resolve_shm1_codon_q_path(path)
        q_codon, frequencies = parse_shm1_codon_q_dat(resolved)
        model = cls(q_codon, frequencies, name=name)
        model._dat_path = resolved
        return model

    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path]) -> "ShmCodonNeutralModel":
        """Load from a .dat file path (portable artifact for this model)."""
        return cls.from_dat(model_path)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        """Human-readable identifier for this neutral model."""
        return self._name

    @property
    def Q_codon(self) -> torch.Tensor:
        """61×61 codon rate matrix (sense codons, dasm2 order)."""
        return self._q_codon

    @property
    def frequencies(self) -> torch.Tensor:
        """Stationary codon frequencies (61,)."""
        return self._frequencies

    # ------------------------------------------------------------------ #
    # CTMC transition probabilities
    # ------------------------------------------------------------------ #

    def compute_transition_matrix(self, branch_length: float) -> torch.Tensor:
        """``P(t) = exp(Q_codon · t)`` — a 61×61 codon transition matrix."""
        return torch.matrix_exp(self._q_codon * float(branch_length))

    def compute_neutral_codon_probs(
        self,
        sequence: str,
        mask: torch.Tensor,
        branch_length: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Per-codon distribution over 64 codons via ``exp(tQ)``.

        Overrides the Poisson default of ``RatesCSPNeutralModel`` with the
        exact codon matrix exponential (mirrors ``GTRCodonNeutralModel``).

        Returns:
            ``(num_codons, 64)`` tensor; stop codons have probability 0.
        """
        parent_codon64 = codon_idx_tensor_of_str_ambig(sequence)
        num_codons = parent_codon64.shape[0]

        transition = self.compute_transition_matrix(branch_length)
        result = torch.zeros((num_codons, 64), dtype=torch.float32)
        sense_targets = self._sense_codon_list

        for pos in range(num_codons):
            if not mask[pos]:
                continue
            codon64_idx = int(parent_codon64[pos].item())
            sense_parent = CODON64_TO_SENSE61.get(codon64_idx, -1)
            if sense_parent == -1:
                raise ValueError(
                    f"Non-sense parent codon (64-index {codon64_idx}) at "
                    f"position {pos}; parents must be sense codons."
                )
            result[pos, sense_targets] = transition[sense_parent, :]

        return result

    # ------------------------------------------------------------------ #
    # RatesCSPNeutralModel interface (per-nt rates + CSPs)
    # ------------------------------------------------------------------ #

    def _build_rate_csp_lookups(self) -> None:
        """Precompute per-parent-codon ``rate[61,3]`` and ``csp[61,3,4]``.

        For parent codon ``i``, position ``p``, target nt ``t`` reachable by a
        single-nt change to sense codon ``j``, the neutral single-hit rate is
        ``Q_codon[i, j]``. The per-position departure rate is the sum over the
        (≤3) sense neighbours at that position, and the CSP is the normalized
        share. Multi-nt (2/3-hit) codon rates in ``Q_codon`` are *not* folded
        in here — they are used directly by the ``shm1_codon`` Q builder; this
        decomposition backs only the rate/CSP consumers.
        """
        csp_numerator = torch.zeros(NUM_SENSE_CODONS, 3, 4)
        # Each (i, pos, target_nt) is a unique single-nt edge; scatter its rate.
        csp_numerator[
            SENSE_CODON_DIFF_I, SENSE_CODON_DIFF_POS, SENSE_CODON_DIFF_TARGET_NT
        ] = self._q_codon[SENSE_CODON_DIFF_I, SENSE_CODON_DIFF_J]
        rate_lookup = csp_numerator.sum(dim=-1)  # [61, 3]
        csp_lookup = csp_numerator / rate_lookup.clamp_min(1e-12).unsqueeze(-1)

        # Pad row 61 = "non-sense / ambiguous" → rate 0, uniform CSP. These
        # positions are masked out of the loss downstream.
        self._rate_lookup = torch.cat(
            [rate_lookup, torch.zeros(1, 3)], dim=0
        )  # [62, 3]
        self._csp_lookup = torch.cat(
            [csp_lookup, torch.full((1, 3, 4), 0.25)], dim=0
        )  # [62, 3, 4]
        # Vectorized codon64 → sense61 map (-1 for stops → pad index 61).
        table = torch.full((64,), -1, dtype=torch.long)
        for c64 in range(64):
            table[c64] = CODON64_TO_SENSE61.get(c64, -1)
        self._codon64_to_sense = table

    def compute_rates_and_csps(
        self,
        sequences: Union[str, List[str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Per-nucleotide rates ``(L,)`` and CSPs ``(L, 4)`` per sequence.

        Derived from the single-nt codon-neighbour decomposition of
        ``Q_codon`` (see ``_build_rate_csp_lookups``). Context-free: identical
        for every occurrence of a given parent codon.
        """
        seq_list = [sequences] if isinstance(sequences, str) else list(sequences)

        rates_out: List[torch.Tensor] = []
        csps_out: List[torch.Tensor] = []
        for seq in seq_list:
            nt_len = len(seq)
            num_codons = nt_len // 3
            rates = torch.zeros(nt_len)
            csps = torch.full((nt_len, 4), 0.25)
            if num_codons > 0:
                codon64 = codon_idx_tensor_of_str_ambig(seq[: num_codons * 3])
                ambiguous = codon64 >= 64
                sense = self._codon64_to_sense[codon64.clamp(max=63)]
                sense = torch.where(ambiguous, torch.tensor(-1), sense)
                # -1 (stop/ambiguous) → pad row 61 (rate 0, uniform CSP).
                sense_idx = torch.where(
                    sense < 0, torch.tensor(NUM_SENSE_CODONS), sense
                )
                codon_rates = self._rate_lookup[sense_idx]  # [num_codons, 3]
                codon_csps = self._csp_lookup[sense_idx]  # [num_codons, 3, 4]
                rates[: num_codons * 3] = codon_rates.reshape(-1)
                csps[: num_codons * 3] = codon_csps.reshape(-1, 4)
            rates_out.append(rates)
            csps_out.append(csps)

        return rates_out, csps_out

    # ------------------------------------------------------------------ #
    # Config round-trip
    # ------------------------------------------------------------------ #

    def to_config(self) -> "Any":
        """Reconstruct the ``ShmCodonNeutralModelConfig`` for this model."""
        from dasm2.config import ShmCodonNeutralModelConfig

        # ``dat_path=None`` means "the packaged default"; only record a path
        # when a non-default matrix was loaded (keeps checkpoints portable).
        dat_path: Optional[str] = None
        if self._dat_path is not None and self._dat_path != default_shm1_codon_q_path():
            dat_path = str(self._dat_path)
        return ShmCodonNeutralModelConfig(dat_path=dat_path)
