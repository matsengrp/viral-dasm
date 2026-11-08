"""GTR codon neutral mutation model with matrix exponentiation.

Implements a context-free neutral model where codon transition probabilities
are computed via direct matrix exponentiation P(t) = exp(Qt), enabling clean
amino acid marginalization for AA-only PCPs.

The model builds a 61×61 codon rate matrix (excluding stop codons) from GTR
nucleotide parameters. This differs from the Poisson approximation used by
other neutral models (rates + CSPs decomposition) and provides accurate
multi-hit handling for any branch length.

Mathematical Background:
    This implements the GY94 codon model (Goldman & Yang 1994) parameterized
    with GTR nucleotide exchangeabilities.

    GTR (General Time Reversible) nucleotide model with 10 parameters:
    - 6 exchangeability parameters: r_AC, r_AG, r_AT, r_CG, r_CT, r_GT
      (symmetric: r_ij = r_ji, required by time-reversibility)
    - 4 equilibrium frequencies: π_A, π_C, π_G, π_T (sum to 1)

    Codon equilibrium frequencies (product of nucleotide frequencies):
        π(c) = π_{c_1} × π_{c_2} × π_{c_3}  (normalized over sense codons)

    Codon rate matrix Q (GY94 neutral model):
        Q[c,c'] = r_{c_k,c'_k} × π(c')  if codons differ at exactly position k
        Q[c,c'] = 0                      if codons differ at 2+ positions
        Q[c,c] = -Σ_{c'≠c} Q[c,c']

    Note: The rate is proportional to the full TARGET CODON frequency π(c'),
    not just the nucleotide frequency at the changing position. This follows
    the GY94 formulation. The alternative MG94 model (Muse & Gaut 1994) uses
    nucleotide frequency instead, which they describe as "a matter of convenience"
    compared to the "more realistic" codon frequency approach.

References:
    - Goldman N, Yang Z. 1994. A codon-based model of nucleotide substitution
      for protein-coding DNA sequences. Mol Biol Evol 11(5):725-736.
      https://doi.org/10.1093/oxfordjournals.molbev.a040153
    - Muse SV, Gaut BS. 1994. A likelihood approach for comparing synonymous
      and nonsynonymous nucleotide substitution rates. Mol Biol Evol 11(5):715-724.
      https://doi.org/10.1093/oxfordjournals.molbev.a040152
    - Bio++ GY94 implementation:
      https://gensoft.pasteur.fr/docs/bpplib/2.2.0/bpp-phyl/classbpp_1_1GY94.html
    - Issue #234: True amino-acid-only PCPs via parent codon marginalization

Testing:
    This implementation is validated against Pyvolve (Spielman & Wilke 2015),
    an established sequence simulation package. Tests verify that nucleotide
    and codon rate matrices match Pyvolve's GTR and GY-style implementations
    exactly, including matrix exponentiation for transition probabilities.
    See tests/neutral/test_pyvolve_compatibility.py.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
import yaml
from pydantic import BaseModel, Field

from dasm2.codon_table import (
    CODON64_TO_SENSE61,
    NUM_SENSE_CODONS,
    SENSE_CODON_AA_INDICATOR,
    SENSE_CODON_DIFF_I,
    SENSE_CODON_DIFF_J,
    SENSE_CODON_LIST,
    SENSE_CODON_NT_INDICES,
    SENSE_CODON_NT_PAIRS,
)
from dasm2.molevol.tensor_ops import normalize_rate_matrix
from dasm2.neutral.base import RatesCSPNeutralModel
from dasm2.sequence.constants import BASES_AND_N_TO_INDEX, CODONS
from dasm2.sequence.conversion import codon_idx_tensor_of_str_ambig
from dasm2.serialization.zip_io import (
    data_archive_exists,
    read_data_archive,
    write_data_archive,
)

if TYPE_CHECKING:
    from dasm2.config import GTRCodonNeutralModelConfig

logger = logging.getLogger(__name__)

# Number of nucleotides
NUM_NT = 4

# Alias for internal use in this module (shorter name for vectorized operations)
AA_INDICATOR = SENSE_CODON_AA_INDICATOR

# Exchangeability parameter order: AC, AG, AT, CG, CT, GT
# This follows the standard convention for GTR parameterization
EXCHANGEABILITY_PAIRS = [
    (0, 1),  # A-C
    (0, 2),  # A-G
    (0, 3),  # A-T
    (1, 2),  # C-G
    (1, 3),  # C-T
    (2, 3),  # G-T
]

# File names in the archive
_PARAMETERS_FILENAME = "parameters.yml"

# Tolerance for frequency normalization validation
_FREQUENCY_SUM_TOLERANCE = 1e-6


class GTRCodonArchiveMetadata(BaseModel):
    """Metadata for GTRCodonNeutralModel archive.

    Stored as metadata.yml in the zip archive alongside parameters.yml.
    """

    model_type: str = Field(
        default="GTRCodonNeutralModel",
        description="Model class identifier for validation",
    )
    model_name: str = Field(
        default="GTRCodon",
        description="Human-readable model name",
    )
    archive_version: int = Field(
        default=1,
        description="Archive format version for future compatibility",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the model and its provenance",
    )


# Aliases for internal use (shorter names)
CODON_DIFF_I = SENSE_CODON_DIFF_I
CODON_DIFF_J = SENSE_CODON_DIFF_J
CODON_NT_PAIRS = SENSE_CODON_NT_PAIRS
CODON_NT_INDICES = SENSE_CODON_NT_INDICES


def validate_gtr_parameters(
    exchangeabilities: torch.Tensor, frequencies: torch.Tensor
) -> None:
    """Check GTR parameters' shapes and values.

    Args:
        exchangeabilities: Shape (6,): r_AC, r_AG, r_AT, r_CG, r_CT, r_GT.
        frequencies: Shape (4,): π_A, π_C, π_G, π_T.

    Raises:
        ValueError: If a shape is wrong, an exchangeability is not positive, a
            frequency is negative, or the frequencies don't sum to 1.
    """
    if exchangeabilities.shape != (6,):
        raise ValueError(
            f"exchangeabilities must have shape (6,), got {exchangeabilities.shape}"
        )
    if (exchangeabilities <= 0).any():
        raise ValueError(
            f"All exchangeabilities must be positive, got min={exchangeabilities.min().item()}"
        )
    if frequencies.shape != (4,):
        raise ValueError(f"frequencies must have shape (4,), got {frequencies.shape}")
    if (frequencies < 0).any():
        raise ValueError(
            f"Frequencies must be non-negative, got min={frequencies.min().item()}"
        )
    freq_sum = frequencies.sum()
    if not torch.isclose(freq_sum, torch.tensor(1.0), atol=_FREQUENCY_SUM_TOLERANCE):
        raise ValueError(f"Frequencies must sum to 1.0, got {freq_sum.item()}")


def build_gtr_nt_rate_matrix(
    exchangeabilities: torch.Tensor, frequencies: torch.Tensor
) -> torch.Tensor:
    """Build the normalized 4×4 nucleotide rate matrix from GTR parameters.

    The GTR rate matrix Q has off-diagonal entries:
        Q[i,j] = r_ij × π_j

    Diagonal entries make rows sum to 0:
        Q[i,i] = -Σ_{j≠i} Q[i,j]

    The matrix is normalized so -Σ_i Q[i,i] × π[i] = 1, making branch
    lengths represent expected nucleotide substitutions per site.

    Args:
        exchangeabilities: Shape (6,), in ``EXCHANGEABILITY_PAIRS`` order.
        frequencies: Shape (4,), equilibrium nucleotide frequencies.

    Returns:
        Normalized 4×4 rate matrix (rows sum to 0, mean rate = 1)
    """
    Q = torch.zeros((NUM_NT, NUM_NT), dtype=torch.float32)

    # Fill off-diagonal entries: Q[i,j] = r_ij × π_j for all nucleotide pairs
    for param_idx, (i, j) in enumerate(EXCHANGEABILITY_PAIRS):
        r_ij = exchangeabilities[param_idx]
        Q[i, j] = r_ij * frequencies[j]
        Q[j, i] = r_ij * frequencies[i]

    # Set diagonal to make rows sum to 0 (vectorized)
    Q.diagonal().copy_(-Q.sum(dim=1))

    # Normalize so mean substitution rate = 1
    return normalize_rate_matrix(Q, frequencies)


def nt_rate_csp_table(Q_nt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-nucleotide rates and CSPs of a context-free 4×4 rate matrix.

    For nucleotide n:

        rate[n] = -Q_nt[n, n]   (total departure rate)
        csp[n, j] = Q_nt[n, j] / rate[n]   for j != n

    Row 4 is the N entry: rate 0 and uniform CSPs. N positions come from
    gap→N conversion and are excluded from the loss via AMBIGUOUS_CODON_IDX
    during tokenization.

    Args:
        Q_nt: A 4×4 nucleotide rate matrix.

    Returns:
        ``(rates, csps)`` of shapes (5,) and (5, 4), indexed by
        ``BASES_AND_N_TO_INDEX``.
    """
    nt_departure_rates = -Q_nt.diag()  # shape (4,)

    # CSPs: Q[n, j] / rate[n] for j != n, 0 for j == n
    csp_matrix = Q_nt / nt_departure_rates.unsqueeze(1)  # (4, 4)
    csp_matrix.fill_diagonal_(0.0)

    rates = torch.cat([nt_departure_rates, torch.zeros(1, device=Q_nt.device)])
    csps = torch.cat([csp_matrix, torch.full((1, 4), 0.25, device=Q_nt.device)])
    return rates, csps


_BYTE_TO_NT_INDEX = torch.full((256,), -1, dtype=torch.long)
"""ASCII byte -> ``BASES_AND_N_TO_INDEX`` index; -1 for any other character."""
_BYTE_TO_NT_INDEX[[ord(base) for base in BASES_AND_N_TO_INDEX]] = torch.tensor(
    list(BASES_AND_N_TO_INDEX.values())
)


def lookup_nt_rates_and_csps(
    sequences: Union[str, List[str]],
    rates: torch.Tensor,
    csps: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Look up each site's rate and CSPs in a context-free table.

    Args:
        sequences: A nucleotide sequence or a list of them (A/C/G/T/N).
        rates: Shape (5,) from :func:`nt_rate_csp_table`.
        csps: Shape (5, 4) from :func:`nt_rate_csp_table`.

    Returns:
        ``(nt_rates_list, nt_csps_list)``, one (L,) and one (L, 4) tensor
        per sequence.

    Raises:
        ValueError: If a sequence has a character other than A/C/G/T/N.
    """
    seq_list = [sequences] if isinstance(sequences, str) else list(sequences)
    nt_rates_list: List[torch.Tensor] = []
    nt_csps_list: List[torch.Tensor] = []
    for seq in seq_list:
        # A byte table rather than a per-character dict lookup: this runs per
        # sample in the dataloader on rate-free Parquet (issue #1299).
        # torch.frombuffer rejects an empty buffer, and empty segments are
        # real: a heavy-only row under the paired handler has an empty light.
        if seq:
            seq_bytes = torch.frombuffer(
                bytearray(seq.encode("ascii")), dtype=torch.uint8
            )
            nt_indices = _BYTE_TO_NT_INDEX[seq_bytes.long()]
        else:
            nt_indices = torch.empty(0, dtype=torch.long)
        if (nt_indices < 0).any():
            bad = sorted({c for c in seq if c not in BASES_AND_N_TO_INDEX})
            raise ValueError(
                f"Sequence has character(s) {bad}; expected only A, C, G, T or N."
            )
        nt_rates_list.append(rates[nt_indices])
        nt_csps_list.append(csps[nt_indices])
    return nt_rates_list, nt_csps_list


class GTRCodonNeutralModel(RatesCSPNeutralModel):
    """Context-free neutral model using GTR + codon matrix exponentiation.

    Computes neutral codon probabilities via P(t) = exp(Qt) where Q is a 61×61
    codon rate matrix derived from GTR nucleotide parameters. This enables:

    1. Accurate multi-hit handling (multiple substitutions over time)
    2. Clean amino acid marginalization for AA-only datasets
    3. Context-free rates (no dependence on flanking nucleotides)

    Attributes:
        name: Human-readable identifier for the model.

    Example:
        >>> # Create JC69 model (simplest case: all rates equal)
        >>> model = GTRCodonNeutralModel.from_jc69()
        >>> P = model.compute_transition_matrix(0.1)  # P(t) at t=0.1
        >>> P_aa = model.compute_aa_transition_matrix(0.1)  # 20×20 AA matrix
    """

    def __init__(
        self,
        exchangeabilities: torch.Tensor,
        frequencies: torch.Tensor,
        *,
        name: str = "GTRCodon",
    ):
        """Initialize GTRCodonNeutralModel with GTR parameters.

        Args:
            exchangeabilities: Shape (6,) tensor of symmetric rate ratios.
                Order: r_AC, r_AG, r_AT, r_CG, r_CT, r_GT.
                All values must be positive.
            frequencies: Shape (4,) tensor of equilibrium nucleotide frequencies.
                Order: π_A, π_C, π_G, π_T.
                Must be non-negative and sum to 1.
            name: Human-readable model identifier for metadata/logging.

        Raises:
            ValueError: If input shapes are wrong, values are invalid, or
                frequencies don't sum to 1.
        """
        validate_gtr_parameters(exchangeabilities, frequencies)

        self._exchangeabilities = exchangeabilities.clone()
        self._frequencies = frequencies.clone()
        self._name = name
        self._archive_path: Optional[Path] = None  # Set by from_pretrained()

        # Compute equilibrium codon frequencies first (needed for rate matrix)
        self._codon_equilibrium = self._compute_codon_equilibrium()

        # Validate all amino acids are reachable (fail-fast on degenerate params)
        aa_freq_sums = AA_INDICATOR @ self._codon_equilibrium
        if (aa_freq_sums <= 0).any():
            raise ValueError(
                "Some amino acids have zero equilibrium frequency. "
                "This indicates invalid GTR parameters that exclude amino acids. "
                f"Min AA frequency: {aa_freq_sums.min().item()}"
            )

        # Build rate matrices and derived quantities (computed once at init)
        self._Q_nt = self._build_nt_rate_matrix()
        self._nt_rates_table, self._nt_csps_table = nt_rate_csp_table(self._Q_nt)
        self._Q_codon = self._build_codon_rate_matrix()
        self._codon_given_aa = self._compute_codon_given_aa()
        self._neutral_aa_rates = self._compute_neutral_aa_rates()

    @property
    def name(self) -> str:
        """Human-readable identifier for this neutral model."""
        return self._name

    @property
    def exchangeabilities(self) -> torch.Tensor:
        """GTR exchangeability parameters (6,): AC, AG, AT, CG, CT, GT."""
        return self._exchangeabilities

    @property
    def frequencies(self) -> torch.Tensor:
        """Equilibrium nucleotide frequencies (4,): A, C, G, T."""
        return self._frequencies

    @property
    def Q_nt(self) -> torch.Tensor:
        """4×4 GTR nucleotide rate matrix."""
        return self._Q_nt

    @property
    def Q_codon(self) -> torch.Tensor:
        """61×61 codon rate matrix (sense codons only)."""
        return self._Q_codon

    # =========================================================================
    # Factory methods for common models
    # =========================================================================

    @classmethod
    def from_hky(
        cls,
        kappa: float,
        frequencies: torch.Tensor,
        *,
        name: str = "HKY-Codon",
    ) -> "GTRCodonNeutralModel":
        """Create HKY model (transition/transversion ratio).

        HKY is a constrained GTR where:
        - Transitions (A↔G, C↔T) have rate κ
        - Transversions (all others) have rate 1

        Args:
            kappa: Transition/transversion ratio (typically 2-10 for coding DNA)
            frequencies: Shape (4,) equilibrium frequencies (A, C, G, T)
            name: Model identifier

        Returns:
            GTRCodonNeutralModel with HKY exchangeabilities
        """
        if kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        # Exchangeabilities: AC=1, AG=κ, AT=1, CG=1, CT=κ, GT=1
        exchangeabilities = torch.tensor([1.0, kappa, 1.0, 1.0, kappa, 1.0])
        return cls(exchangeabilities, frequencies, name=name)

    @classmethod
    def from_k80(
        cls, kappa: float, *, name: str = "K80-Codon"
    ) -> "GTRCodonNeutralModel":
        """Create Kimura 1980 (K80/K2P) model.

        K80 is HKY with equal nucleotide frequencies (0.25 each).

        Args:
            kappa: Transition/transversion ratio
            name: Model identifier

        Returns:
            GTRCodonNeutralModel with K80 parameters
        """
        return cls.from_hky(kappa, torch.full((4,), 0.25), name=name)

    @classmethod
    def from_jc69(cls, *, name: str = "JC69-Codon") -> "GTRCodonNeutralModel":
        """Create Jukes-Cantor 1969 (JC69) model.

        JC69 is the simplest model: all exchangeabilities equal, all
        frequencies equal. All substitutions have equal rate.

        Args:
            name: Model identifier

        Returns:
            GTRCodonNeutralModel with JC69 parameters
        """
        return cls(torch.ones(6), torch.full((4,), 0.25), name=name)

    # =========================================================================
    # Rate matrix construction
    # =========================================================================

    def _build_nt_rate_matrix(self) -> torch.Tensor:
        """Build normalized 4×4 nucleotide rate matrix from GTR parameters.

        See :func:`build_gtr_nt_rate_matrix`.
        """
        return build_gtr_nt_rate_matrix(self._exchangeabilities, self._frequencies)

    def _build_exchangeability_matrix(self) -> torch.Tensor:
        """Build 4×4 symmetric exchangeability matrix from 6 parameters.

        Returns:
            4×4 matrix R where R[i,j] = r_ij (symmetric, diagonal = 0)
        """
        R = torch.zeros((NUM_NT, NUM_NT), dtype=torch.float32)
        for param_idx, (i, j) in enumerate(EXCHANGEABILITY_PAIRS):
            R[i, j] = self._exchangeabilities[param_idx]
            R[j, i] = self._exchangeabilities[param_idx]
        return R

    def _build_codon_rate_matrix(self) -> torch.Tensor:
        """Build normalized 61×61 codon rate matrix following GY94 formulation.

        Implements the GY94 neutral codon model (Goldman & Yang 1994):
            Q[c,c'] = r_{c_k,c'_k} × π(c')  if codons differ at exactly position k
            Q[c,c'] = 0                      if codons differ at 2+ positions

        where r is the nucleotide exchangeability and π(c') is the full TARGET
        CODON equilibrium frequency (not just the nucleotide frequency).

        The matrix is normalized so -Σ_c Q[c,c] × π(c) = 1, making branch
        lengths represent expected codon substitutions per site.

        Uses precomputed indices (CODON_DIFF_I, CODON_DIFF_J, CODON_NT_PAIRS)
        for vectorized construction.

        Returns:
            Normalized 61×61 rate matrix (rows sum to 0, mean rate = 1)
        """
        Q = torch.zeros((NUM_SENSE_CODONS, NUM_SENSE_CODONS), dtype=torch.float32)

        # Build exchangeability matrix R (symmetric, 4×4)
        R = self._build_exchangeability_matrix()

        # Get exchangeabilities for each single-nucleotide codon change
        nt_from = CODON_NT_PAIRS[:, 0]
        nt_to = CODON_NT_PAIRS[:, 1]
        exchangeabilities = R[nt_from, nt_to]

        # Get target codon equilibrium frequencies (GY94: rate ∝ π(target codon))
        target_codon_freqs = self._codon_equilibrium[CODON_DIFF_J]

        # GY94 rate: r_{nt} × π(target_codon)
        rates = exchangeabilities * target_codon_freqs

        # Assign rates to the codon matrix
        Q[CODON_DIFF_I, CODON_DIFF_J] = rates

        # Set diagonal to make rows sum to 0 (vectorized)
        Q.diagonal().copy_(-Q.sum(dim=1))

        # Normalize so mean substitution rate = 1
        return normalize_rate_matrix(Q, self._codon_equilibrium)

    def _compute_codon_equilibrium(self) -> torch.Tensor:
        """Compute equilibrium codon frequencies from nucleotide frequencies.

        Under the M0 model, codon frequencies are products of nucleotide frequencies:
            π(c) = π(c_1) × π(c_2) × π(c_3)

        These are then normalized to sum to 1 over sense codons only.

        Uses precomputed CODON_NT_INDICES for vectorized computation.

        Returns:
            Shape (61,) tensor of equilibrium codon frequencies (float32)
        """
        # Gather nucleotide frequencies for each position of each codon: (61, 3)
        nt_freqs = self._frequencies[CODON_NT_INDICES]

        # Product across the 3 positions: (61,)
        codon_freq = nt_freqs.prod(dim=1)

        # Normalize to account for excluded stop codons
        codon_freq = codon_freq / codon_freq.sum()

        return codon_freq

    # =========================================================================
    # Transition probability computation
    # =========================================================================

    def compute_transition_matrix(self, branch_length: float) -> torch.Tensor:
        """Compute P(t) = exp(Qt) for given branch length.

        Args:
            branch_length: Evolutionary time t (non-negative)

        Returns:
            61×61 transition probability matrix where P[i,j] is the probability
            of codon i transitioning to codon j over time t (float32).

        Raises:
            ValueError: If branch_length is negative
        """
        if branch_length < 0:
            raise ValueError(f"branch_length must be non-negative, got {branch_length}")

        # P(t) = exp(Q × t) via matrix exponential
        # Use float64 for matrix_exp numerical stability, then convert back
        P = torch.matrix_exp(self._Q_codon.double() * branch_length)

        return P.float()

    def compute_aa_transition_matrix(self, branch_length: float) -> torch.Tensor:
        """Marginalize codon matrix to 20×20 AA transition matrix.

        Computes:
            P(a → a') = Σ_{c ∈ codons(a)} Σ_{c' ∈ codons(a')} π(c|a) × P(c → c')

        where π(c|a) is the conditional codon probability given amino acid,
        from the cached ``codon_given_aa`` matrix.

        Uses vectorized matrix operations with precomputed AA_INDICATOR matrix.

        Args:
            branch_length: Evolutionary time t (non-negative)

        Returns:
            20×20 amino acid transition probability matrix (float32)
        """
        P_codon = self.compute_transition_matrix(branch_length)

        # Marginalize to AA space via matrix multiplication:
        # P_aa[a, a'] = Σ_c Σ_c' π(c|a) × P(c→c') × I[c'→a']
        P_aa = self._codon_given_aa @ P_codon @ AA_INDICATOR.T

        return P_aa

    # =========================================================================
    # RatesCSPNeutralModel interface (rates/CSPs decomposition)
    # =========================================================================

    def compute_rates_and_csps(
        self,
        sequences: Union[str, List[str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute per-nucleotide rates and CSPs from the GTR rate matrix.

        Derives position-specific rates and conditional substitution probabilities
        from the 4x4 nucleotide rate matrix Q_nt. For nucleotide n at position i:

            rate_i = -Q_nt[n, n]   (total departure rate)
            csp_i[j] = Q_nt[n, j] / rate_i   for j != n

        Since GTR is context-free, all positions with the same nucleotide get
        identical rates and CSPs.

        Note: GTRCodonNeutralModel overrides compute_neutral_codon_probs() with
        matrix exponentiation, so the default Poisson approximation inherited from
        RatesCSPNeutralModel is NOT used. This method exists to support code paths
        that need rates/CSPs directly (e.g., Whichmut data loading).

        Args:
            sequences: Single nucleotide sequence string or list of sequences.
            context: Unused by this model.

        Returns:
            Tuple of (nt_rates_list, nt_csps_list) where each list contains
            one tensor per input sequence:
                - nt_rates_list: Each tensor has shape (seq_length,)
                - nt_csps_list: Each tensor has shape (seq_length, 4)
        """
        return lookup_nt_rates_and_csps(
            sequences, self._nt_rates_table, self._nt_csps_table
        )

    # =========================================================================
    # AA-level marginalization properties (for AA-only PCPs)
    # =========================================================================

    def _compute_codon_given_aa(self) -> torch.Tensor:
        """Compute π(c|a) matrix once at initialization.

        Returns:
            Tensor of shape (20, 61) with conditional codon probabilities.
        """
        aa_freq_sums = AA_INDICATOR @ self._codon_equilibrium
        codon_given_aa = AA_INDICATOR * self._codon_equilibrium.unsqueeze(0)
        codon_given_aa = codon_given_aa / aa_freq_sums.unsqueeze(1)
        return codon_given_aa

    @property
    def codon_equilibrium(self) -> torch.Tensor:
        """Equilibrium codon frequencies (61,) over sense codons.

        Computed as the product of nucleotide frequencies at each codon position:
            π(c) = π(c₁) × π(c₂) × π(c₃)

        Normalized to sum to 1 over sense codons only (stop codons excluded).

        Returns:
            Tensor of shape (61,) with equilibrium codon frequencies.
        """
        return self._codon_equilibrium

    @property
    def codon_given_aa(self) -> torch.Tensor:
        """Conditional codon probability given amino acid: π(c|a).

        Shape (20, 61) where row a contains π(c|a) for each codon c.
        Computed from equilibrium codon frequencies:
            π(c|a) = π(c) / Σ_{c' ∈ codons(a)} π(c')

        This matrix is useful for marginalizing codon-level quantities to
        amino acid level, e.g., P_aa = codon_given_aa @ P_codon @ AA_INDICATOR.T

        Returns:
            Tensor of shape (20, 61) with conditional codon probabilities.
        """
        return self._codon_given_aa

    def _compute_neutral_aa_rates(self) -> torch.Tensor:
        """Compute 20×20 AA neutral rate matrix once at initialization.

        Returns:
            Tensor of shape (20, 20) with AA-level neutral rates.
        """
        # Zero out diagonal of Q_codon to get only off-diagonal (substitution) rates
        Q_off_diag = self._Q_codon.clone()
        Q_off_diag.fill_diagonal_(0)

        # Marginalize: neutral_aa = codon_given_aa @ Q_offdiag @ AA_INDICATOR.T
        neutral_aa = self._codon_given_aa @ Q_off_diag @ AA_INDICATOR.T
        return neutral_aa

    @property
    def neutral_aa_rates(self) -> torch.Tensor:
        """20×20 AA neutral rates marginalized from Q_codon off-diagonal.

        **Uses off-diagonal entries only** (diagonal zeroed) because the AAO loss
        expects positive diagonal entries for synonymous rates in the gauge term.
        Using the full Q_codon (with negative diagonal) would give incorrect signs.

        Computes amino-acid-level neutral substitution rates by marginalizing
        the codon rate matrix Q_codon:

            neutral_aa[a, a'] = Σ_{c: aa(c)=a} π(c|a) × Σ_{c': aa(c')=a'} Q_offdiag[c, c']

        By marginalizing from Q_codon with diagonal zeroed:
        - **Off-diagonal** (a ≠ a'): nonsynonymous rates (positive)
        - **Diagonal** (a = a'): synonymous codon substitution rates (non-negative;
          zero for Met/Trp which have single codons)

        This matches what the existing codon-level pipeline produces via
        ``marginalize_codon_to_aao_rates``, where single-codon amino acids also
        get zero diagonal since ``CODON_SINGLE_MUTATIONS`` only enumerates changes
        to different codons.

        **Key properties:**
        - Cached at model initialization (no recomputation on access)
        - Context-free (same rates apply to all positions)
        - These are instantaneous rates (Q-derived), not transition probabilities P(t)

        Returns:
            Tensor of shape (20, 20) with AA-level neutral rates.

        Example:
            >>> model = GTRCodonNeutralModel.from_jc69()
            >>> rates = model.neutral_aa_rates
            >>> rates.shape
            torch.Size([20, 20])
            >>> # Single-codon AAs (Met, Trp) have zero diagonal
            >>> AA_STR_SORTED.index('M')  # Methionine
            12
            >>> rates[12, 12].item()  # Zero synonymous rate
            0.0
        """
        return self._neutral_aa_rates

    # =========================================================================
    # AbstractNeutralModel interface
    # =========================================================================

    def to_config(self) -> "GTRCodonNeutralModelConfig":
        """Reconstruct GTRCodonNeutralModelConfig from neutral model state.

        If the model was loaded via catalog name, returns a config with
        ``model_name`` (portable). Otherwise returns ``model_path`` or
        inline GTR parameters.

        Returns:
            GTRCodonNeutralModelConfig with model_name, model_path, or
            inline GTR parameters.
        """
        from dasm2.config import GTRCodonNeutralModelConfig

        if self._catalog_name is not None:
            return GTRCodonNeutralModelConfig(model_name=self._catalog_name)

        if self._archive_path is not None:
            logger.info(
                "GTRCodonNeutralModel.to_config() is returning a non-portable "
                "model_path (%s). Checkpoints using this config will not be portable "
                "across machines. Use model_name (catalog-based) configs instead.",
                self._archive_path,
            )
            return GTRCodonNeutralModelConfig(model_path=self._archive_path)

        # Reconstruct inline config from stored parameters
        return GTRCodonNeutralModelConfig(
            variant="gtr",
            exchangeabilities=self._exchangeabilities.tolist(),
            frequencies=self._frequencies.tolist(),
        )

    def compute_neutral_codon_probs(
        self,
        sequence: str,
        mask: torch.Tensor,
        branch_length: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute neutral codon probabilities using matrix exponentiation.

        For each codon position in the sequence, returns the probability
        distribution over all 64 codons (including stop codons with prob 0).

        Args:
            sequence: Nucleotide sequence (length must be multiple of 3)
            mask: Boolean mask of shape (num_codons,) for valid positions
            branch_length: Evolutionary time for probability calculation
            context: Interface compatibility parameter, unused by this model.

        Returns:
            Codon probabilities of shape (num_codons, 64) where the probability
            of transitioning from each parent codon to each possible child codon
            is given. Stop codons have probability 0.
        """
        # Get parent codon indices (flat tensor of shape (num_codons,))
        parent_codon64 = codon_idx_tensor_of_str_ambig(sequence)
        num_codons = parent_codon64.shape[0]

        # Compute transition matrix: (61, 61)
        P = self.compute_transition_matrix(branch_length)

        # Output in 64-codon space
        result = torch.zeros((num_codons, 64), dtype=torch.float32)

        # Build mapping from 64-space to 61-space as tensor for vectorized lookup
        # SENSE_CODON_LIST[i] gives the 64-space index for sense codon i
        sense_codon_list_tensor = torch.tensor(SENSE_CODON_LIST, dtype=torch.long)

        # Process each position
        for pos in range(num_codons):
            if not mask[pos]:
                continue

            codon64_idx = parent_codon64[pos].item()
            sense_parent = CODON64_TO_SENSE61.get(int(codon64_idx), -1)

            if sense_parent == -1:
                codon_str = CODONS[int(codon64_idx)]
                raise ValueError(
                    f"Stop codon '{codon_str}' at position {pos} in parent sequence. "
                    f"Stop codons are not valid in parent sequences for neutral model."
                )

            # Vectorized: copy transition probs from 61-space to 64-space
            result[pos, sense_codon_list_tensor] = P[sense_parent, :]

        return result

    # =========================================================================
    # Serialization (archive format)
    # =========================================================================

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save model to a portable zip archive.

        Creates a zip archive containing:
        - metadata.yml: Model metadata (name, version, type)
        - parameters.yml: GTR exchangeabilities and frequencies

        Args:
            path: Output path for the zip archive (should end in .zip)

        Example:
            >>> model = GTRCodonNeutralModel.from_jc69()
            >>> model.save_pretrained("jc69_codon_v1.zip")
        """
        path = Path(path)
        logger.info(f"Saving GTRCodonNeutralModel to {path}")

        # Create metadata
        metadata = GTRCodonArchiveMetadata(model_name=self._name)

        # Create parameters YAML
        parameters = {
            "exchangeabilities": self._exchangeabilities.tolist(),
            "frequencies": self._frequencies.tolist(),
        }
        params_yaml = yaml.dump(parameters, default_flow_style=False)

        # Write archive
        files = {
            _PARAMETERS_FILENAME: params_yaml.encode("utf-8"),
        }
        write_data_archive(path, metadata, files)

        logger.info(f"Saved GTRCodonNeutralModel '{self._name}' to {path}")

    _EXPECTED_MODEL_TYPE = "GTRCodonNeutralModel"

    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path]) -> "GTRCodonNeutralModel":
        """Load model from a pretrained zip archive.

        Args:
            model_path: Path to zip archive created by save_pretrained()

        Returns:
            Initialized GTRCodonNeutralModel with loaded parameters

        Raises:
            FileNotFoundError: If archive doesn't exist
            ValueError: If archive format is invalid

        Example:
            >>> model = GTRCodonNeutralModel.from_pretrained("jc69_codon_v1.zip")
        """
        path = Path(model_path)
        logger.info(f"Loading GTRCodonNeutralModel from {path}")

        # Validate archive exists and has expected files
        if not data_archive_exists(path, [_PARAMETERS_FILENAME]):
            raise FileNotFoundError(
                f"Invalid or missing archive at {path}. "
                f"Expected zip with metadata.yml and {_PARAMETERS_FILENAME}."
            )

        # Read archive
        metadata, files = read_data_archive(path, [_PARAMETERS_FILENAME])

        # Validate metadata
        model_type = metadata.get("model_type")
        if model_type != cls._EXPECTED_MODEL_TYPE:
            raise ValueError(
                f"Archive has wrong model_type: {model_type}. "
                f"Expected: {cls._EXPECTED_MODEL_TYPE}."
            )

        # Parse parameters
        params = yaml.safe_load(files[_PARAMETERS_FILENAME])
        exchangeabilities = torch.tensor(
            params["exchangeabilities"], dtype=torch.float32
        )
        frequencies = torch.tensor(params["frequencies"], dtype=torch.float32)
        model_name = metadata.get("model_name", "GTRCodon")

        # Create model instance
        model = cls(exchangeabilities, frequencies, name=model_name)
        model._archive_path = path  # Track source for to_config()

        logger.info(f"Loaded GTRCodonNeutralModel '{model_name}' from {path}")
        return model
