"""Native DASM2 implementation of the S5F five-mer SHM neutral model.

S5F is netam's ``RSFivemerModel`` (``netam/models.py``): a pure lookup table
over the 4^5 five-mers, holding one log-rate and four CSP logits per five-mer.
Unlike Thrifty (a CNN over the whole sequence), there is no learned interaction
between sites, so the rate and CSP at a site depend only on the five-mer window
centered on it. This module reimplements that lookup natively so netam stays off
the inference path (same rationale ``thrifty.py`` uses for the CNN).

Runtime pieces:
    - ``RSFivemerModule``: two ``nn.Embedding`` tables (log-rate, CSP logits),
      the native equivalent of netam's ``RSFivemerModel``.
    - ``S5FNeutralModel``: a ``RatesCSPNeutralModel`` that serves rates/CSPs by
      *direct table lookup per site* (not the Thrifty encoder path).

Rate scale convention
---------------------
S5F mutabilities are *relative* targeting rates. SHazaM normalizes the published
tables to sum to 1 over the 1024 N-free five-mers, so the raw mean rate is
exactly 1/1024 and carries no information about the absolute substitution rate.
DASM2's branch-length machinery assumes rates are O(1): branch length means
"expected substitutions per site", the antibody warm start
(``ProportionalStrategy(slope=1.76)``) never consults the neutral model, and
``LOG_BL_UPPER`` caps branch length at ``exp(4)``. A model whose rates are ~1000x
too small therefore does *not* merely rescale the inferred branch lengths — the
optimizer pins against its bounds and returns wrong branch lengths with no
warning, because the near-bound check only fires at the absolute bounds.

So on load we divide the table by its mean over the 1024 concrete five-mers,
putting S5F on the same "mean substitution rate = 1" convention as
``gtr_codon._normalize_rate_matrix``, and matching what netam establishes for
Thrifty via ``SHMBurrito.standardize_model_rates`` /
``RSFivemerModel.adjust_rate_bias_by``. This is a pure global scale factor: it
preserves the relative ordering across five-mers exactly, leaves CSPs untouched
(a softmax is invariant to a constant log-rate offset), and is idempotent, so
re-standardizing an already-standardized table is a no-op.

The uniform mean over all 1024 five-mers is used rather than a mean over some
reference sequence because it is data-free and reproducible. It is also a good
proxy for the latter: standardized S5F-HH's mean rate over VRC01 is 0.857 versus
Thrifty's 0.887 (3.4%), so the Thrifty-calibrated warm-start slope and the 0.63
light-chain rate multiplier carry over to S5F unchanged. Note netam's own anchor
(rate 1 at VRC01 site 14) is deliberately *not* used: shipped Thrifty measures
0.605 there rather than 1.0, and that site's five-mer is cold in HH but hot in
HKL (an 8x spread), so anchoring there would manufacture a heavy/light scale
difference out of a single arbitrary site.

N handling and near-boundary sites
-----------------------------------
The Thrifty encoder collapses every five-mer that contains an N — whether from
boundary padding or a real ambiguous flanking base — to a single shared
"ambiguous" row (index 0). That would give the first/second and last/second-last
sites of every sequence (which are scored) one global fallback rate/CSP with no
dependence on their known partial context. S5F does **not** do this.

Instead, for any *scored* site (center base in ACGT) whose five-mer window
contains one or more N's, S5F **marginalizes**: it enumerates every concrete
resolution of the N positions (each N → {A, C, G, T}), looks up each concrete
five-mer's rate and CSP, and combines them under a uniform prior on the
unknown bases:

    - rate = mean of the concrete five-mers' rates (scalar) — a linear
      expectation, so the plain mean is the exact marginal rate;
    - CSP  = the *rate-weighted* mean of the concrete five-mers' CSP
      distributions (each already softmaxed with the wild-type/center base
      zeroed). The downstream Poisson approximation reads a substitution
      probability as ``rate * t * csp(b)``, so the exact marginal substitution
      probability is ``mean(rate_X * csp_X(b))``; dividing by the marginal
      rate ``mean(rate_X)`` recovers the CSP in that decomposition. A plain
      ``mean(csp_X)`` would misweight resolutions whose rate and CSP
      preference co-vary.

Because S5F is a pure lookup table, these are the *exact* marginals over a
uniform prior on the unknown bases. The center base of a scored site is never
N, so the wild-type position is consistent across all resolutions and the
rate-weighted CSP is still a valid distribution with 0 self-substitution.
Concretely:

    - First site (i=0): 2 leading pad-N's → mean over 16 five-mers.
    - Second site (i=1): 1 leading pad-N → mean over 4.
    - Second-last / last sites: symmetric (4 and 16).
    - Interior site with a real N in a flank: mean over that N's resolutions.
    - Center-N sites: masked/unscored — rate 1.0, uniform CSP (matching the
      ``exp(log_rate * mask)`` convention the rest of DASM2 uses for masked
      positions); no resolution needed.

Resolved ``(rate, csp)`` values are memoized per (possibly N-containing)
five-mer string on the instance, so each distinct ambiguous pattern is resolved
once and reused across all sites and sequences.
"""

import itertools
import logging
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field, ValidationError

from dasm2.neutral.base import RatesCSPNeutralModel
from dasm2.neutral.kmer_encoding import BASES, generate_kmers
from dasm2.sequence.conversion import _NT_CHAR_TO_IDX
from dasm2.serialization.zip_io import (
    data_archive_exists,
    read_data_archive,
    write_data_archive,
)

if TYPE_CHECKING:
    from dasm2.config import S5FNeutralModelConfig

logger = logging.getLogger(__name__)

# Name of the torch state-dict file stored inside S5F zip archives.
_WEIGHTS_FILENAME = "weights.pth"

# S5F is defined over five-mers; this is fixed, not a hyperparameter.
_KMER_LENGTH = 5
_HALF_KMER = _KMER_LENGTH // 2

# Base-4 place values for a concrete (N-free) five-mer, matching the order of
# ``generate_kmers(5)``: index 0 is the "N" placeholder, then five-mers in
# lexicographic ACGT order, so idx = 1 + Σ base_j * 4^(4-j) with A=0..T=3.
_POWERS = [4 ** (_KMER_LENGTH - 1 - j) for j in range(_KMER_LENGTH)]
_BASE_TO_IDX = {base: idx for idx, base in enumerate(BASES)}

# ASCII byte → base index (A=0, C=1, G=2, T=3), -1 for N or any other
# ambiguous character. Derived from ``_NT_CHAR_TO_IDX`` (which maps N→4) with N
# folded into -1 so the vectorized concrete-window path can select fully
# ACGT windows with a single ``>= 0`` mask, mirroring ``tabular_rates``.
_ASCII_TO_BASE_IDX = _NT_CHAR_TO_IDX.astype(np.int64).copy()
_ASCII_TO_BASE_IDX[_ASCII_TO_BASE_IDX == 4] = -1


class S5FArchiveMetadata(BaseModel):
    """Metadata for an ``S5FNeutralModel`` zip archive.

    Stored as ``metadata.yml`` inside the archive alongside ``weights.pth`` (the
    torch state dict of the two embedding tables). Unlike ThriftySHM there are no
    ``filter_count`` / ``kernel_size`` fields — S5F is a lookup table, not a CNN.
    """

    model_type: Literal["S5FNeutralModel"] = Field(
        default="S5FNeutralModel",
        description="Model class identifier for validation",
    )
    model_name: str = Field(description="Human-readable model name")
    archive_version: int = Field(
        default=1,
        description="Archive format version for future compatibility",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the model and its provenance",
    )
    kmer_length: int = Field(
        default=_KMER_LENGTH,
        description="K-mer length (always 5 for S5F)",
    )
    site_count: int = Field(
        gt=0,
        description=(
            "Reference sequence length recorded for provenance. The lookup path "
            "is context-only and length-independent, so this is not enforced."
        ),
    )


class RSFivemerModule(nn.Module):
    """Two embedding tables indexed by five-mer: log-rate and CSP logits.

    Native DASM2 equivalent of netam's ``RSFivemerModel``. Table row ``i``
    corresponds to ``generate_kmers(5)[i]`` (row 0 is the "N" placeholder).
    """

    def __init__(self) -> None:
        super().__init__()
        self.kmer_length = _KMER_LENGTH
        self.all_kmers = generate_kmers(_KMER_LENGTH)
        self.kmer_count = len(self.all_kmers)  # 1 + 4**5 = 1025

        self.r_kmer_embedding = nn.Embedding(self.kmer_count, 1)
        self.s_kmer_embedding = nn.Embedding(self.kmer_count, 4)

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """Return model hyperparameters."""
        return {"kmer_length": self.kmer_length}


def concrete_kmer_index(kmer: str) -> int:
    """Return the ``generate_kmers(5)`` row index for an N-free five-mer.

    Public utility: the S5F build script and tests reuse it to align an
    ``RSFivemerModule`` embedding row with a specific five-mer string.
    """
    offset = sum(_BASE_TO_IDX[base] * power for base, power in zip(kmer, _POWERS))
    return 1 + int(offset)


def _concrete_mean_rate(rate_table: torch.Tensor) -> torch.Tensor:
    """Return the mean rate over the 1024 concrete five-mers of ``rate_table``.

    Row 0 (the "N" placeholder) is excluded: no scored site ever reads it, so
    letting it into the mean would let an arbitrary placeholder value shift the
    whole model's rate scale.
    """
    scale = rate_table[1:].mean()
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"S5F rate table has a non-finite or non-positive mean rate ({scale}); "
            "cannot standardize it to mean rate 1. The source mutability table "
            "likely contains NaN, inf, or all-zero entries."
        )
    return scale


class S5FNeutralModel(RatesCSPNeutralModel):
    """Native S5F five-mer neutral model backed by a per-site lookup table.

    Serves rates/CSPs by looking up each site's five-mer window directly, with
    exact marginalization over N's (boundary padding or ambiguous flanking
    bases); see the module docstring for the full N-handling contract.

    Example:
        >>> from dasm2.model_catalog import resolve_model
        >>> path = resolve_model("S5F-HH-classic-v1")
        >>> model = S5FNeutralModel.from_catalog("S5F-HH-classic-v1", path)
        >>> rates, csps = model.compute_rates_and_csps(["ACGTACGTACGT"])
    """

    def __init__(
        self,
        module: RSFivemerModule,
        site_count: int,
        model_name: str = "unnamed",
    ) -> None:
        """Initialize from a loaded ``RSFivemerModule``.

        Use ``from_catalog()`` / ``from_pretrained()`` to load from an archive.

        Args:
            module: RSFivemerModule with loaded embedding weights.
            site_count: Reference sequence length (provenance only, not enforced).
            model_name: Human-readable model identifier.
        """
        self._module = module
        self._module.eval()
        self._site_count = site_count
        self._model_name = model_name
        # Set by from_pretrained(); None for in-memory models built via __init__.
        self._archive_path: Optional[Path] = None

        # Precompute the flat lookup tables once. rate_table[i] is the concrete
        # rate exp(log_rate) for five-mer i; csp_table[i] is the softmaxed CSP
        # with the center (wild-type) base zeroed. Row 0 ("N" placeholder) is
        # never used for scored sites; set it to uniform for safety.
        with torch.no_grad():
            r_weight = module.r_kmer_embedding.weight.detach().squeeze(-1)
            rate_table = torch.exp(r_weight)
            # See "Rate scale convention" in the module docstring: S5F tables
            # carry relative targeting only, so the absolute scale is set here.
            self._rate_table = rate_table / _concrete_mean_rate(rate_table)

            s_logits = module.s_kmer_embedding.weight.detach().clone()
            # Zero the wild-type (center) base of each concrete five-mer so it
            # has probability 0 after softmax, exactly as netam does via the
            # wt_base_modifier. The center base is determined by the five-mer.
            # all_kmers is ordered by row index (row 0 is the "N" placeholder),
            # so enumerate gives the table row directly — no need to re-derive it.
            wt_modifier = torch.zeros_like(s_logits)
            for idx, kmer in enumerate(module.all_kmers[1:], start=1):
                center_base_idx = _BASE_TO_IDX[kmer[_HALF_KMER]]
                wt_modifier[idx, center_base_idx] = float("-inf")
            self._csp_table = torch.softmax(s_logits + wt_modifier, dim=-1)
            # Placeholder row: uniform (unused by scored sites).
            self._csp_table[0] = 0.25

        # Per-instance memoization of resolved five-mers → (rate, csp). Distinct
        # ambiguous patterns are bounded and small, so this stays tiny. A plain
        # dict is fine under the per-worker deep-copy in streaming datasets.
        self._lookup_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def name(self) -> str:
        """Human-readable identifier for this neutral model."""
        return self._model_name

    @property
    def site_count(self) -> int:
        """Reference sequence length (provenance only; not enforced)."""
        return self._site_count

    @property
    def kmer_length(self) -> int:
        """S5F's bounded neutral-model receptive field."""
        return _KMER_LENGTH

    def kmer_table(self) -> torch.Tensor:
        """Return the concrete five-mer lookup as a ``[4^5, 5]`` tensor.

        Columns are ``(rate, csp_A, csp_C, csp_G, csp_T)``, with the rate
        already standardized to mean 1 and the wild-type (center) base zeroed in
        the CSP — i.e. exactly the layout the k-mer Q builder
        (:mod:`dasm2.molevol.kmer_q`) consumes, and the same layout as the
        precomputed Thrifty 13-mer table.

        Row ``i`` is the five-mer with ``i = Σ base_j * 4^(4-j)`` (A=0..T=3);
        the internal row-0 "N" placeholder is dropped, so this is
        index-compatible with the builder's base-4 decomposition. Because S5F's
        rate and CSP at a site are a function of the centered five-mer and
        nothing else, a Q built from this table is *exact* rather than an
        approximation.

        Returns:
            ``[1024, 5]`` float tensor.
        """
        # Rows 1..1024 are already the normalized, WT-zeroed concrete k-mers
        # in base-4 order; row 0 is the internal ambiguous-token placeholder.
        return torch.cat(
            [self._rate_table[1:].unsqueeze(-1), self._csp_table[1:]], dim=-1
        )

    def _lookup_5mer(self, kmer: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(rate, csp)`` for a five-mer, marginalizing over any N's.

        The caller guarantees the center base (index 2) is in ACGT. Any flank
        position not in ACGT is treated as fully ambiguous and marginalized over
        {A, C, G, T}. Results are memoized keyed by the five-mer string.
        """
        cached = self._lookup_cache.get(kmer)
        if cached is not None:
            return cached

        ambiguous_positions = [
            pos for pos, base in enumerate(kmer) if base not in _BASE_TO_IDX
        ]
        if not ambiguous_positions:
            idx = concrete_kmer_index(kmer)
            rate = self._rate_table[idx]
            csp = self._csp_table[idx]
        else:
            indices = []
            for resolution in itertools.product(BASES, repeat=len(ambiguous_positions)):
                chars = list(kmer)
                for pos, base in zip(ambiguous_positions, resolution):
                    chars[pos] = base
                indices.append(concrete_kmer_index("".join(chars)))
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            resolution_rates = self._rate_table[idx_tensor]
            rate = resolution_rates.mean()
            # CSP is the *rate-weighted* mean over resolutions, not a plain mean.
            # The downstream Poisson approximation reads a substitution
            # probability as rate * t * csp(b), so the exact marginal
            # substitution probability under a uniform prior is
            # mean(rate_X * csp_X(b)); dividing by the marginal rate mean(rate_X)
            # recovers the CSP in that decomposition. A plain mean(csp_X) would
            # misweight resolutions whose rate and CSP preference co-vary.
            csp = (resolution_rates.unsqueeze(-1) * self._csp_table[idx_tensor]).sum(
                dim=0
            ) / resolution_rates.sum()

        result = (rate, csp)
        self._lookup_cache[kmer] = result
        return result

    def _rates_and_csps_for_sequence(
        self, sequence: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-site rates and CSPs for a single sequence.

        The common case — an interior site whose full five-mer window is
        concrete (all ACGT) — is served by a single vectorized table lookup.
        The remaining sites (the two nearest each boundary, any window
        containing a real N, and center-N masked sites) fall back to the
        per-site ``_lookup_5mer`` marginalization path. Output is identical to
        a pure per-site loop; only the concrete interior lookups are batched.
        """
        seq_len = len(sequence)
        rates = torch.ones(seq_len, dtype=torch.float32)
        # Masked/unscored default is a uniform CSP, matching netam's behavior for
        # positions where the wild-type base is N (csp_logits zeroed → uniform).
        csps = torch.full((seq_len, 4), 0.25, dtype=torch.float32)
        if seq_len == 0:
            return rates, csps

        base_idxs = _ASCII_TO_BASE_IDX[
            np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        ]

        # Interior centers (indices _HALF_KMER .. seq_len-1-_HALF_KMER) whose
        # full five-mer window is concrete (no N in any of the five positions)
        # are looked up in one vectorized batch. windows[j] is the j-th window
        # position across all windows, so column w is the five-mer centered at
        # w + _HALF_KMER.
        handled = np.zeros(seq_len, dtype=bool)
        if seq_len >= _KMER_LENGTH:
            windows = np.stack(
                [
                    base_idxs[j : seq_len - (_KMER_LENGTH - 1 - j)]
                    for j in range(_KMER_LENGTH)
                ]
            )
            concrete = (windows >= 0).all(axis=0)
            # Base-4 index of each window; garbage where non-concrete (bases are
            # -1 there) but those columns are masked out before any lookup.
            fivemer_idx = 1 + sum(windows[j] * _POWERS[j] for j in range(_KMER_LENGTH))
            concrete_centers = np.nonzero(concrete)[0] + _HALF_KMER
            if concrete_centers.size:
                rows = torch.from_numpy(fivemer_idx[concrete].astype(np.int64))
                rates[concrete_centers] = self._rate_table[rows]
                csps[concrete_centers] = self._csp_table[rows]
                handled[concrete_centers] = True

        # Everything the vectorized path did not cover: boundary sites, interior
        # windows containing a real N, and center-N masked sites. Resolved per
        # site so the N-marginalization contract (see module docstring) applies.
        for i in np.nonzero(~handled)[0]:
            center = sequence[i]
            if center not in _BASE_TO_IDX:
                # Center-N (or other ambiguous) site: masked/unscored. Leave the
                # rate=1.0 / uniform-CSP defaults set above.
                continue

            # Five-mer window centered on i; out-of-range positions are N pads.
            window_chars = [
                sequence[j] if 0 <= j < seq_len else "N"
                for j in range(i - _HALF_KMER, i + _HALF_KMER + 1)
            ]
            rate, csp = self._lookup_5mer("".join(window_chars))
            rates[i] = rate
            csps[i] = csp

        return rates, csps

    def compute_rates_and_csps(
        self,
        sequences: Union[str, List[str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute neutral rates and CSPs for one or more sequences.

        Concrete interior five-mer windows are resolved with a single vectorized
        numpy/torch table lookup per sequence (only boundary and N-containing
        windows use the per-site path), so this method is already fast; like
        ``TabularRatesNeutralModel`` it therefore relies on the base-class
        ``compute_rates_and_csps_vectorized`` fallback, which delegates here,
        rather than reimplementing a separate vectorized entry point.

        Args:
            sequences: Single sequence string or list of sequences.
            context: Reserved for future use (interface compatibility); unused.

        Returns:
            Tuple of (nt_rates_list, nt_csps_list), one tensor per input
            sequence: rates of shape (seq_length,), CSPs of shape (seq_length, 4).
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        nt_rates_list: List[torch.Tensor] = []
        nt_csps_list: List[torch.Tensor] = []
        for seq in sequences:
            rates, csps = self._rates_and_csps_for_sequence(seq)
            nt_rates_list.append(rates)
            nt_csps_list.append(csps)
        return nt_rates_list, nt_csps_list

    def to_config(self) -> "S5FNeutralModelConfig":
        """Reconstruct ``S5FNeutralModelConfig`` from neutral model state.

        Prefers the catalog name recorded by ``from_catalog()`` (portable).
        Falls back to the archive ``model_path`` (non-portable) recorded by
        ``from_pretrained()``, mirroring ``TabularRatesNeutralModel``.

        Raises:
            ValueError: If the model was loaded neither via catalog nor archive.
        """
        from dasm2.config import S5FNeutralModelConfig  # noqa: PLC0415

        if self._catalog_name is not None:
            return S5FNeutralModelConfig(model_name=self._catalog_name)

        if self._archive_path is None:
            raise ValueError(
                "Cannot create config: S5FNeutralModel was not loaded from an "
                "archive or catalog. Use save_pretrained() then from_pretrained()."
            )

        logger.info(
            "S5FNeutralModel.to_config() is returning a non-portable model_path "
            "(%s); checkpoints using this config will not be portable across "
            "machines. Prefer catalog (model_name) configs.",
            self._archive_path,
        )
        return S5FNeutralModelConfig(model_path=self._archive_path)

    def save_pretrained(
        self,
        path: Union[str, Path],
        description: str = "",
        extra_files: Optional[Dict[str, bytes]] = None,
    ) -> None:
        """Save the model to a portable zip archive.

        Creates a zip archive containing ``metadata.yml`` (an
        ``S5FArchiveMetadata``) and ``weights.pth`` (the two embedding tables).

        Args:
            path: Output path for the zip archive (should end in .zip).
            description: Optional human-readable provenance description.
            extra_files: Optional mapping of archive-relative filename to raw
                bytes for provenance artifacts bundled alongside the model (e.g.
                the source S5F CSVs under ``source/``). These are ignored by
                ``from_pretrained``, which only reads ``weights.pth``.

        Raises:
            ValueError: If an ``extra_files`` name collides with a reserved
                archive filename (``weights.pth`` / ``metadata.yml``).
        """
        path = Path(path)
        hyperparameters = self._module.hyperparameters
        metadata = S5FArchiveMetadata(
            model_name=self._model_name,
            description=description,
            kmer_length=hyperparameters["kmer_length"],
            site_count=self._site_count,
        )

        weights_buffer = BytesIO()
        torch.save(self._module.state_dict(), weights_buffer)

        files = {_WEIGHTS_FILENAME: weights_buffer.getvalue()}
        for name, content in (extra_files or {}).items():
            if name in (_WEIGHTS_FILENAME, "metadata.yml"):
                raise ValueError(
                    f"extra_files name '{name}' collides with a reserved archive "
                    f"filename ({_WEIGHTS_FILENAME} / metadata.yml)."
                )
            files[name] = content

        write_data_archive(path, metadata, files)
        logger.info(f"Saved S5FNeutralModel '{self._model_name}' to {path}")

    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path]) -> "S5FNeutralModel":
        """Load an S5F model from a pretrained zip archive.

        Args:
            model_path: Path to zip archive created by ``save_pretrained()``.

        Returns:
            Initialized S5FNeutralModel with loaded weights.

        Raises:
            FileNotFoundError: If the archive doesn't exist or is missing files.
            ValueError: If the archive metadata is invalid.
        """
        path = Path(model_path)
        logger.info(f"Loading S5FNeutralModel from {path}")

        if not data_archive_exists(path, [_WEIGHTS_FILENAME]):
            raise FileNotFoundError(
                f"Invalid or missing archive at {path}. "
                f"Expected zip with metadata.yml and {_WEIGHTS_FILENAME}."
            )

        raw_metadata, files = read_data_archive(path, [_WEIGHTS_FILENAME])
        try:
            metadata = S5FArchiveMetadata.model_validate(raw_metadata)
        except ValidationError as exc:
            raise ValueError(f"Archive at {path} has invalid metadata: {exc}") from exc

        module = RSFivemerModule()
        state_dict = torch.load(
            BytesIO(files[_WEIGHTS_FILENAME]), map_location="cpu", weights_only=True
        )
        module.load_state_dict(state_dict)

        model = cls(module, metadata.site_count, metadata.model_name)
        model._archive_path = path
        return model
