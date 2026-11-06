"""General sequence masking utilities for netam compatibility.

Key Requirements:
- ✅ ^ separator masking (works for any number of sequences)
- ✅ Exclude ^ separator tokens from outputs (netam-compatible)
- ✅ Standardized on single ^ separator (simplified from multiple types)
- ✅ Maintain sequence order and structure
- ✅ User-specified problematic sites masking for viral sequences

Note:
    Separator masking is implemented at the dataset level via the layout tensor,
    where separator positions have layout value 0 (see SequenceDataset.get_mask).
    This is simpler and more direct than token-based masking approaches.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    get_args,
)

import torch
from torch import Tensor

if TYPE_CHECKING:
    import pandas as pd

from dasm2.io import read_fasta_first_sequence
from dasm2.sequence.constants import BASES, TOKEN_STR_SORTED


def create_problematic_sites_mask(
    problematic_sites: List[int],
    max_seq_len: int,
) -> torch.Tensor:
    """Create boolean mask for problematic sites.

    Args:
        problematic_sites: List of 0-indexed site positions to mask
        max_seq_len: Maximum sequence length (must match dataset max_aa_seq_len).
                    Must be positive.

    Returns:
        Boolean tensor [max_seq_len] where True=valid, False=masked

    Raises:
        ValueError: If any site is out of bounds or max_seq_len <= 0

    Example:
        >>> mask = create_problematic_sites_mask([2, 5], max_seq_len=10)
        >>> mask  # tensor([True, True, False, True, True, False, True, True, True, True])
    """
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

    mask = torch.ones(max_seq_len, dtype=torch.bool)

    for site in problematic_sites:
        if not (0 <= site < max_seq_len):
            raise ValueError(f"Site {site} out of bounds for max_seq_len={max_seq_len}")
        mask[site] = False

    return mask


def parse_problematic_sites_file(
    filepath: str,
    zero_indexed: bool = False,
) -> List[int]:
    """Parse problematic sites from file (always returns 0-indexed).

    File format: One site number per line. Lines starting with '#' are comments.
    Empty lines are ignored.

    Args:
        filepath: Path to file with one site number per line
        zero_indexed: If True, sites in file are 0-indexed;
                     if False, convert from 1-indexed (common in biology)

    Returns:
        Sorted list of unique site indices (0-indexed)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file contains invalid site numbers (with line numbers in error message)
    """
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"Problematic sites file not found: {filepath}")

    sites = []
    invalid_lines = []

    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse site number - separate int conversion from domain validation
            try:
                site = int(line)
            except ValueError:
                invalid_lines.append((line_num, line, "Not a valid integer"))
                continue

            if site < 0:
                invalid_lines.append((line_num, line, f"Negative site index: {site}"))
                continue

            # Convert to 0-indexed if needed
            sites.append(site if zero_indexed else site - 1)

    # Fail if any invalid lines found
    if invalid_lines:
        error_msg = "Invalid lines in problematic sites file:\n"
        for line_num, line, error in invalid_lines:
            error_msg += f"  Line {line_num}: '{line}' - {error}\n"
        raise ValueError(error_msg)

    return sorted(set(sites))


def split_predictions_by_sequences(
    predictions: Tensor, token_ids: Tensor
) -> List[Tensor]:
    """Split predictions into components corresponding to individual sequences.

    Separates predictions for multi-segment sequences (e.g., "HEAVY^LIGHT") into
    individual segment predictions, excluding separator token positions. For
    single-segment sequences without separators, returns the predictions as-is.

    Args:
        predictions: Model predictions tensor of shape [seq_len, num_classes]
        token_ids: Tokenized sequence tensor of shape [seq_len]

    Returns:
        List of prediction tensors, one for each sequence component.
        ^ separator positions are excluded from all components.

    Example:
        For "SEQA^SEQB" input, returns [predictions_for_SEQA, predictions_for_SEQB]
        where ^ separator position predictions are excluded.

    Note:
        This implements the same logic as netam's split_heavy_light_model_outputs
        but generalized to work with any number of segments separated by ^.
    """
    # Find separator token index
    separator_idx = TOKEN_STR_SORTED.index("^")

    # Create mask for non-separator positions
    non_separator_mask = token_ids != separator_idx

    # If no separators, return predictions as single list
    if non_separator_mask.all():
        return [predictions]

    # Find separator positions
    separator_positions = torch.where(~non_separator_mask)[0].tolist()

    # Split into segments
    segments = []
    start_idx = 0

    for sep_pos in separator_positions:
        # Add segment from start to separator (excluding separator)
        if start_idx < sep_pos:
            segments.append(predictions[start_idx:sep_pos])
        start_idx = sep_pos + 1  # Skip separator

    # Add final segment after last separator
    if start_idx < len(predictions):
        segments.append(predictions[start_idx:])

    return segments


def has_unmasked_codon_mutation(
    parent_nt: str,
    child_nt: str,
    mask: torch.Tensor,
) -> bool:
    """Check if a PCP has at least one mutation in unmasked positions.

    When problematic sites are masked, some PCPs may have all their mutations
    in masked positions, leaving no valid evolutionary signal. This function
    identifies such PCPs for filtering.

    Args:
        parent_nt: Parent nucleotide sequence (must be divisible by 3)
        child_nt: Child nucleotide sequence (same length as parent)
        mask: Boolean mask [num_codons] where True=valid, False=masked

    Returns:
        True if at least one codon-level mutation exists in unmasked positions

    Raises:
        ValueError: If sequences differ in length, aren't divisible by 3,
            or mask is shorter than the number of codons

    Example:
        >>> import torch
        >>> mask = torch.tensor([True, False, True])  # middle codon masked
        >>> has_unmasked_codon_mutation("ATGATGATG", "ATCATGATG", mask)  # mut at pos 0
        True
        >>> has_unmasked_codon_mutation("ATGATGATG", "ATGAAGATG", mask)  # mut at pos 1 (masked)
        False
    """
    if len(parent_nt) != len(child_nt):
        raise ValueError(
            f"Parent length ({len(parent_nt)}) != child length ({len(child_nt)})"
        )
    if len(parent_nt) % 3 != 0:
        raise ValueError(f"Sequence length ({len(parent_nt)}) not divisible by 3")

    num_codons = len(parent_nt) // 3

    if len(mask) < num_codons:
        raise ValueError(
            f"Mask length ({len(mask)}) must be >= number of codons ({num_codons})"
        )

    for codon_idx in range(num_codons):
        # Skip masked positions
        if not mask[codon_idx]:
            continue

        start = codon_idx * 3
        end = start + 3
        parent_codon = parent_nt[start:end]
        child_codon = child_nt[start:end]

        # Check for mutation at codon level
        if parent_codon != child_codon:
            return True

    return False


def filter_pcps_with_only_masked_mutations(
    pcp_df: "pd.DataFrame",
    mask: torch.Tensor,
    parent_col: str = "parent",
    child_col: str = "child",
) -> "pd.DataFrame":
    """Filter out PCPs where all mutations fall in masked positions.

    Args:
        pcp_df: DataFrame with parent/child nucleotide sequence columns
        mask: Boolean mask [num_codons] where True=valid, False=masked
        parent_col: Column name for parent sequences
        child_col: Column name for child sequences

    Returns:
        Filtered DataFrame containing only PCPs with at least one unmasked mutation

    Example:
        >>> import torch
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "parent": ["ATGATGATG", "ATGATGATG"],
        ...     "child": ["ATCATGATG", "ATGAAGATG"],  # mut at pos 0, 1
        ... })
        >>> mask = torch.tensor([True, False, True])  # mask pos 1
        >>> filtered = filter_pcps_with_only_masked_mutations(df, mask)
        >>> len(filtered)
        1
    """
    from dasm2.data.validation import _report_filtered

    valid_mask = pcp_df.apply(
        lambda row: has_unmasked_codon_mutation(row[parent_col], row[child_col], mask),
        axis=1,
    )

    return _report_filtered(pcp_df, valid_mask, "PCPs with only masked mutations")


# ---------------------------------------------------------------------------
# UShER censor tables (#1079)
# ---------------------------------------------------------------------------
#
# The functions above express a *codon-granular, clade-independent* mask, which
# is all `{gene}_codon_sites_to_mask.txt` can carry. A censor table needs more:
# UShER masking deletes mutations from the MAT rather than marking sites
# ambiguous, so a masked site is censored, not ambiguous -- the state is known,
# the *event* was destroyed. Expressing that needs single-nucleotide and
# single-direction granularity, per clade, which is why this is a table of
# coordinates rather than a boolean mask over positions.
#
# Produced by mat-pcp's `censor_table` rule (matsengrp/mat-pcp#144) as
# `pcps/{gene}/censored_mutations.csv` plus a `.clades.txt` roster sidecar.
#
# The coordinate table is the *more general* representation, though, so the codon
# mask is expressible in it rather than being a parallel mechanism: a global codon
# site is a clade-independent site mask, which is exactly what
# `CensorSpec.site_nt_indices` already means. `with_global_codon_sites` (#1083)
# does that merge, so both mask files reach the loss through one code path.

CENSOR_TABLE_COLUMNS = ("clade", "nt_site", "nt_index", "from_nt", "to_nt")

# mat-pcp's `pcps/{gene}/masked_nt_sites.csv` (mat-pcp#153) -- the *clade-
# independent* half of the same coordinate contract, at nucleotide granularity.
# `censored_mutations.csv` carries the two clade-specific reasons of
# `masking.compute_exclusions`' five; this file carries the three global ones
# (`sites_to_exclude`, the ProblematicSites VCF, outside `site_include_range`).
# Together they give the complete nucleotide mask with no codon widening
# anywhere. It supersedes `{gene}_codon_sites_to_mask.txt`, which expresses the
# same three reasons but can only name whole codons.
#
# No `clade` column, because there is nothing clade-specific in it, and no
# `from_nt`/`to_nt`, because every row is a whole-site mask -- which is exactly
# `CensorSpec.site_nt_indices`. `reason` is carried for provenance and is not
# read: censoring is the same act whichever of the three flagged the site.
MASKED_NT_SITES_COLUMNS = ("nt_site", "nt_index", "reason")

# Base -> index into BASES, which is the encoding
# `molevol.codon_neutral_rates._MUTATION_BASE_INDICES` is built with. Spelled as
# `BASES.index` rather than a fourth `BASE_TO_IDX` dict (there are already three
# in the tree, in `neutral.s5f`, `neutral.tabular_rates` and
# `molevol.codon_neutral_rates`) and rather than importing the molevol one, which
# would invert the layering -- `sequence` is below `molevol`. The two encodings
# agreeing is load-bearing, so a test pins it.


# The censoring ablation's arms (#1079). "both" is the only coherent objective;
# the two single-half modes exist to demonstrate that the halves do different
# things, since each on its own still trains and produces a plausible curve.
#
# Defined here, beside `CensorSpec`, rather than in `data.transforms` where the
# transform lives: `TrainingConfig.censor_mode` needs the same alias, and
# `dasm2.config` importing `dasm2.data.transforms` would point a low-level module
# at a high-level one. `sequence` is already below both.
CensorMode = Literal["both", "rates_only", "loss_mask_only"]

# Derived from the alias rather than re-spelled, so the two cannot disagree.
CENSOR_MODES = frozenset(get_args(CensorMode))


# `eq=False` because the generated `__eq__` compares the two tensor fields with
# `==`, which returns a tensor and then raises RuntimeError in the `bool()` the
# tuple comparison does. Identity is what the callers want anyway: a censor spec
# is built once per clade at load time and thereafter passed around by reference.
# `CensorReferenceReversions` and `LookupAAONeutralRates` carry this for the same
# reason.
@dataclass(frozen=True, eq=False)
class CensorSpec:
    """One clade's censored nucleotide positions and transitions.

    Two tensors rather than one, because the two mask kinds censor different
    amounts and the transform applies them differently:

    * ``site_nt_indices`` ``(K,)`` -- whole-site masks. Censors every mutation
      away from that nucleotide, i.e. all three candidate codons differing at
      its codon offset.
    * ``directional`` ``(M, 3)`` -- ``[nt_index, from_base_idx, to_base_idx]``,
      base indices into :data:`~dasm2.sequence.constants.BASES`. Censors exactly
      one transition, and only where the parent actually carries
      ``from_base_idx``. The reverse direction stays live.

    Index tensors rather than a DataFrame so instances pickle cheaply for
    DataLoader workers (the viral configs run ``num_dataloader_workers: 8``, and
    each worker gets its own copy of the whole table).

    ``nt_index`` is 0-based into the *gene's* PCP sequence, not the genome: the
    consumer has no CDS knowledge, so the PCP sequence is the coordinate system.
    """

    site_nt_indices: Tensor
    directional: Tensor

    def __post_init__(self) -> None:
        if self.site_nt_indices.dtype != torch.int64:
            raise ValueError(
                f"site_nt_indices must be int64, got {self.site_nt_indices.dtype}. "
                "It indexes tensors directly."
            )
        if self.site_nt_indices.ndim != 1:
            raise ValueError(
                f"site_nt_indices must be 1-D, got shape "
                f"{tuple(self.site_nt_indices.shape)}."
            )
        if self.directional.dtype != torch.int64:
            raise ValueError(
                f"directional must be int64, got {self.directional.dtype}."
            )
        if self.directional.ndim != 2 or self.directional.shape[1] != 3:
            raise ValueError(
                f"directional must have shape (M, 3) for "
                f"[nt_index, from_base_idx, to_base_idx], got "
                f"{tuple(self.directional.shape)}."
            )

    @property
    def is_empty(self) -> bool:
        """True when this clade censors nothing.

        Legitimate and common: on the v11 tree only 2 of 56 clades carry any E
        mask at all. Distinguishing this from a mistyped clade name is what the
        roster exists for -- see :func:`load_censor_table`.
        """
        return self.site_nt_indices.numel() == 0 and self.directional.numel() == 0

    @property
    def num_entries(self) -> int:
        """Total censored entries this clade declares, for reporting.

        Counts *table rows*, not censored candidates and not matched positions: a
        site entry removes three candidates and a directional entry one, and
        either may match no position in a given corpus. Measuring what the corpus
        actually loses needs a pass over the PCPs, which is the ``audit_censor``
        step's job rather than this property's.
        """
        return int(self.site_nt_indices.shape[0] + self.directional.shape[0])


def empty_censor_spec() -> CensorSpec:
    """A :class:`CensorSpec` that censors nothing.

    A function rather than a module constant so no two clades share a tensor.
    The transform never writes to these, but a shared mutable default is the
    kind of thing that stops being true later.
    """
    return CensorSpec(
        site_nt_indices=torch.zeros(0, dtype=torch.int64),
        directional=torch.zeros((0, 3), dtype=torch.int64),
    )


def censored_codon_sites(spec: CensorSpec) -> Set[int]:
    """1-indexed codon sites this clade censors at least one candidate in.

    The site class the fitness comparison splits on (#1079). A model-external
    metric is computed separately over these sites and over the rest, because
    the two are where censoring predicts an effect and where it predicts at most
    an indirect one.

    Both mask kinds count, and they are not equal in weight: a whole-site mask
    removes three of the codon's nine candidates while a directional mask
    removes one. Site masks dominate in practice -- 1,118 of S's 1,892 v11 rows
    -- so a site here is usually one that lost a third of its candidates. Union
    rather than a split, because the prediction ("selection factors biased low
    where candidates were removed and nothing opposed them") has the same sign
    for both and the counts are too thin in the directional-only class to
    correlate separately.

    Args:
        spec: One clade's censored positions.

    Returns:
        Codon sites, 1-indexed to match ``aa_site`` in the fitness tables.
        Empty for a clade that censors nothing.
    """
    nt_indices = torch.cat([spec.site_nt_indices, spec.directional[:, 0]])
    return {int(idx) // 3 + 1 for idx in nt_indices}


def _parse_censor_row(
    row: Dict[str, str], line_num: int, table_path: Path
) -> tuple[str, int, str, str]:
    """Validate one censor-table row into ``(clade, nt_index, from_nt, to_nt)``.

    Every failure raises rather than skipping. A row this function cannot read is
    a row that silently censors nothing, and in the loaded table an unparseable
    entry and a legitimately-absent one look identical -- which is the failure
    mode the whole censoring change exists to avoid reintroducing.
    """
    where = f"{table_path}:{line_num}"
    clade = (row["clade"] or "").strip()
    if not clade:
        raise ValueError(f"{where}: blank clade.")

    try:
        nt_index = int(row["nt_index"])
    except (TypeError, ValueError):
        raise ValueError(
            f"{where}: nt_index {row['nt_index']!r} is not an integer."
        ) from None
    if nt_index < 0:
        raise ValueError(
            f"{where}: nt_index is {nt_index}, but it is a 0-based index into "
            f"the gene's PCP sequence and cannot be negative. A 1-based genomic "
            f"site belongs in nt_site ({row.get('nt_site')!r})."
        )

    from_nt = (row["from_nt"] or "").strip().upper()
    to_nt = (row["to_nt"] or "").strip().upper()

    # Both blank is a site mask; both populated is a directional mask. One of
    # each is neither, and guessing which was intended would either over-censor
    # (treat it as a site mask) or censor nothing (drop it).
    if bool(from_nt) != bool(to_nt):
        raise ValueError(
            f"{where}: from_nt={from_nt!r} and to_nt={to_nt!r}. A site mask "
            f"leaves both blank and a directional mask populates both; one of "
            f"each is neither, so refusing to guess."
        )
    for name, base in (("from_nt", from_nt), ("to_nt", to_nt)):
        if base and base not in BASES:
            raise ValueError(
                f"{where}: {name}={base!r} is not one of {list(BASES)}. "
                f"Ambiguity codes cannot be censored -- a censored mutation is a "
                f"known event that was deleted, not an unknown state."
            )
    if from_nt and from_nt == to_nt:
        raise ValueError(
            f"{where}: from_nt and to_nt are both {from_nt!r}. That is not a "
            f"mutation, so it can censor nothing."
        )

    return clade, nt_index, from_nt, to_nt


def load_censor_table(
    table_path: str | Path,
    roster_path: str | Path,
) -> Dict[str, CensorSpec]:
    """Load a gene's per-clade censor table, keyed by clade.

    Args:
        table_path: ``pcps/{gene}/censored_mutations.csv`` -- one row per
            censored (clade, nucleotide) or (clade, nucleotide, direction).
            Site masks leave ``from_nt``/``to_nt`` blank; directional masks
            populate both. ``nt_index`` is *one-to-many* against ``nt_site``: a
            site read twice by ``slice_cds`` at a ribosomal frameshift occupies
            two positions of the PCP sequence, in different codons, and gets a
            row for each.
        roster_path: ``pcps/{gene}/censored_mutations.clades.txt`` -- every clade
            the table was built over, one per line.

    Returns:
        ``{clade: CensorSpec}`` covering **every** clade in the roster, with an
        empty spec for those censoring nothing.

    Raises:
        FileNotFoundError: If either file is missing.
        ValueError: On a malformed row, a missing column, an empty roster, a
            duplicated roster entry, or a table clade absent from the roster.

    Why the roster is a separate input rather than derived from the rows: a clade
    with zero censored mutations contributes zero rows, so in the CSV alone it is
    indistinguishable from a mistyped clade name (``22B`` vs ``22B.1``). Keying
    off the rows alone forces a choice between ``table[clade]``, which
    ``KeyError``s on every legitimately-empty clade, and ``table.get(clade, [])``,
    which silently censors nothing for a typo. Materializing the roster first
    makes absence-from-roster and absence-from-rows different things. On the v11
    tree this is the common case, not the corner case: 54 of 56 clades have no E
    mask at all.
    """
    table_path, roster_path = Path(table_path), Path(roster_path)
    if not table_path.exists():
        raise FileNotFoundError(f"Censor table not found: {table_path}")
    if not roster_path.exists():
        raise FileNotFoundError(
            f"Censor clade roster not found: {roster_path}. It is emitted "
            f"alongside {table_path.name} and is required -- without it an empty "
            f"clade cannot be told from a mistyped one."
        )

    roster: List[str] = []
    for line in roster_path.read_text().splitlines():
        clade = line.strip()
        if clade:
            roster.append(clade)
    if not roster:
        raise ValueError(
            f"{roster_path} lists no clades. An empty roster would make every "
            f"clade in the corpus unknown."
        )
    if duplicated := sorted({c for c in roster if roster.count(c) > 1}):
        raise ValueError(f"{roster_path} lists {duplicated} more than once.")

    sites: Dict[str, List[int]] = {clade: [] for clade in roster}
    directional: Dict[str, List[tuple[int, int, int]]] = {clade: [] for clade in roster}

    with open(table_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if missing := [
            c for c in CENSOR_TABLE_COLUMNS if c not in (reader.fieldnames or [])
        ]:
            raise ValueError(
                f"{table_path} is missing column(s) {missing}. Found "
                f"{list(reader.fieldnames or [])}; expected "
                f"{list(CENSOR_TABLE_COLUMNS)}."
            )
        for line_num, row in enumerate(reader, start=2):
            clade, nt_index, from_nt, to_nt = _parse_censor_row(
                row, line_num, table_path
            )
            if clade not in sites:
                raise ValueError(
                    f"{table_path}:{line_num}: clade {clade!r} is not in the "
                    f"roster {roster_path.name}. The two are written by the same "
                    f"rule, so this means they have been decoupled. Roster "
                    f"clades: {roster}."
                )
            if from_nt:
                directional[clade].append(
                    (nt_index, BASES.index(from_nt), BASES.index(to_nt))
                )
            else:
                sites[clade].append(nt_index)

    # Deduplicated and sorted so the loaded table is a deterministic function of
    # the file's *content*, not its row order. Duplicates are legitimate rather
    # than suspicious: a site mask and a directional mask can name the same
    # nt_index, and censoring the same entry twice is idempotent.
    return {
        clade: CensorSpec(
            site_nt_indices=torch.tensor(
                sorted(set(sites[clade])), dtype=torch.int64
            ).reshape(-1),
            directional=torch.tensor(
                sorted(set(directional[clade])), dtype=torch.int64
            ).reshape(-1, 3),
        )
        for clade in roster
    }


def parse_masked_nt_sites_file(table_path: str | Path) -> List[int]:
    """Load a gene's clade-independent nucleotide mask (mat-pcp#153).

    The nucleotide-granular replacement for
    :func:`parse_problematic_sites_file`'s ``{gene}_codon_sites_to_mask.txt``.
    Both name the same three clade-independent reasons; this one names the
    nucleotides they actually flagged rather than the codons containing them,
    which on the six v12 production genes is 121 sites against 345.

    Args:
        table_path: ``pcps/{gene}/masked_nt_sites.csv``, columns
            :data:`MASKED_NT_SITES_COLUMNS`.

    Returns:
        Sorted, deduplicated **0-indexed** indices into this gene's PCP
        nucleotide sequence, ready for :func:`with_global_nt_sites`.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: On a missing column, a non-integer ``nt_index``, a negative
            ``nt_index``, or a header-only file.

    Note:
        ``nt_index`` is read and ``nt_site`` is not, and the two are *not*
        interchangeable. ``nt_site`` is the 1-based genomic coordinate and is
        one-to-many against ``nt_index``: at ORF1b's -1 ribosomal frameshift a
        single genomic site is read twice by ``slice_cds`` and occupies two
        positions of the PCP sequence, in different codons, so it emits two rows
        sharing an ``nt_site``. Deduplicating on ``nt_site`` would drop one of
        them and leave half the frameshift uncensored. This matches
        :func:`load_censor_table`, which keys on ``nt_index`` for the same
        reason.
    """
    table_path = Path(table_path)
    if not table_path.exists():
        raise FileNotFoundError(f"Global nucleotide mask not found: {table_path}")

    indices: List[int] = []
    with open(table_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if missing := [
            c for c in MASKED_NT_SITES_COLUMNS if c not in (reader.fieldnames or [])
        ]:
            raise ValueError(
                f"{table_path} is missing column(s) {missing}. Found "
                f"{list(reader.fieldnames or [])}; expected "
                f"{list(MASKED_NT_SITES_COLUMNS)}."
            )
        for line_num, row in enumerate(reader, start=2):
            raw = (row["nt_index"] or "").strip()
            try:
                nt_index = int(raw)
            except ValueError:
                raise ValueError(
                    f"{table_path}:{line_num}: nt_index {raw!r} is not an integer."
                ) from None
            if nt_index < 0:
                raise ValueError(
                    f"{table_path}:{line_num}: nt_index {nt_index} is negative. "
                    f"The column is a 0-based index into the gene's PCP "
                    f"nucleotide sequence."
                )
            indices.append(nt_index)

    # A header-only file is rejected here rather than at the merge, so the error
    # names the file. mat-pcp emits one legitimately when a gene's CDS carries no
    # flagged nucleotide, but for these six genes it never does, and an empty
    # global mask silently reproduces the per-clade table -- the same reasoning
    # `with_global_nt_sites` applies to an empty `nt_indices`.
    if not indices:
        raise ValueError(
            f"{table_path} is header-only, so it censors nothing. That "
            f"reproduces the per-clade table exactly. Leave "
            f"TrainingConfig.masked_nt_sites_file unset to skip the global mask "
            f"deliberately."
        )
    return sorted(set(indices))


def _require_clade_roster(table: Dict[str, CensorSpec], what: str) -> None:
    """Raise unless ``table`` carries the corpus's clade roster.

    A global mask carries no clade information of its own, but the transform is
    keyed by clade, so it can only be expressed by merging into a table that
    already knows which clades exist. That is why global-only censoring is
    unrepresentable rather than merely unsupported.
    """
    if not table:
        raise ValueError(
            f"table covers no clades, so there is nothing to merge {what} into. "
            f"Load a censor table first: a global mask carries no clade "
            f"information, but the transform is keyed by clade and needs the "
            f"roster to know which clades exist."
        )


def with_global_codon_sites(
    table: Dict[str, CensorSpec],
    codon_sites: Sequence[int],
    num_codons: Optional[int],
) -> Dict[str, CensorSpec]:
    """Return ``table`` with every candidate at ``codon_sites`` censored in every clade.

    The second of mat-pcp's two mask files, folded into the first's representation
    (#1083). ``{gene}_codon_sites_to_mask.txt`` marks codon sites where *every*
    clade excludes every nucleotide -- the ProblematicSites VCF, the config's
    ``sites_to_exclude``, and ``site_include_range``. Those are artifact lists, so
    the claim they support is not that these sites mutate often but that
    observations at them are untrustworthy. That is the same argument the per-clade
    table rests on, and it wants the same treatment: an observation nobody trusts
    should not have its *absence* charged to selection.

    A merge rather than a second mechanism, because a global codon site *is* a
    clade-independent site mask and :attr:`CensorSpec.site_nt_indices` already
    means "censor every mutation away from this nucleotide". Everything downstream
    -- the scatter in ``CensorMaskedMutations._censored_entries_mask``, the
    ``mode`` handling, the tie between numerator and denominator, the per-sample
    bounds guard -- is inherited unchanged, and there is one code path to test
    rather than two to keep in agreement.

    Why this is not folded into :func:`load_censor_table`: under ``mode="both"``
    the surviving observations at these sites leave the numerator too, and on the
    v11 S corpus that is 2,738 real mutations. Discarding them is right if they
    are artifacts, but it is a choice, so it sits behind an explicit config field
    (``TrainingConfig.codon_sites_to_mask_file``) rather than arriving with the
    per-clade table.

    Args:
        table: Per-clade table from :func:`load_censor_table`, covering every
            clade of the corpus.
        codon_sites: **0-indexed** codon sites, as
            :func:`parse_problematic_sites_file` returns them. May contain
            duplicates and need not be sorted.
        num_codons: Codon count of the corpus, for the bounds check, or ``None``
            when the caller does not know it. ``None`` is a real case rather than
            laziness: only heads that size a per-site table from it set
            ``max_aa_seq_len``, and the transformer configs deliberately leave it
            unset because ``create_aa_tensors`` truncates to it silently. When it
            is ``None`` the bound is not checked here and
            ``CensorMaskedMutations``'s per-sample ``_max_nt_index`` check is what
            catches a gene mismatch, on the first batch.

    Returns:
        A new table, same keys. Never mutates ``table`` or its tensors.

    Raises:
        ValueError: If ``codon_sites`` is empty, if ``table`` is empty, or if a
            site falls outside the corpus.

    Note:
        The bounds check catches a mask file naming codons the corpus does not
        have, but *not* the reverse -- a short gene's mask against a long gene's
        corpus is entirely in range and would silently censor that gene's first
        codons. Nothing ties a mask file to a corpus; the same hole exists for the
        per-clade table. ``_resolve_censor_table`` logs the site count and maximum
        so a mismatch is visible, and the ``audit_censor`` rule, which reads the
        corpus, checks it.
    """
    _require_clade_roster(table, "global codon sites")
    if not codon_sites:
        raise ValueError(
            "codon_sites is empty. Merging no global sites reproduces the "
            "per-clade table exactly, so requesting it is almost certainly a "
            "mistake -- most likely the wrong gene's "
            "{gene}_codon_sites_to_mask.txt. Leave "
            "TrainingConfig.codon_sites_to_mask_file unset to censor only the "
            "per-clade table deliberately."
        )
    if num_codons is not None and (
        out_of_range := sorted({site for site in codon_sites if site >= num_codons})
    ):
        raise ValueError(
            f"global codon site(s) {[site + 1 for site in out_of_range]} "
            f"(1-indexed) fall outside a corpus of {num_codons} codons "
            f"({3 * num_codons} nt). This mask file belongs to a different gene "
            f"-- nothing else ties a {{gene}}_codon_sites_to_mask.txt to a corpus."
        )
    if negative := sorted({site for site in codon_sites if site < 0}):
        raise ValueError(
            f"global codon site(s) {negative} are negative. codon_sites is "
            f"0-indexed, as parse_problematic_sites_file returns it; a 1-based "
            f"site of 0 is not a site."
        )

    # All three nucleotides, not just the codon's first: the mask is codon-granular
    # and a site mask censors mutations *away from* the named nucleotide, so
    # covering the codon takes one entry per offset. Getting this wrong censors a
    # third of what was intended and still trains, which is why V-A pins each
    # offset separately.
    global_nt = {3 * site + offset for site in codon_sites for offset in range(3)}

    return _merge_global_nt_indices(table, global_nt)


def with_global_nt_sites(
    table: Dict[str, CensorSpec],
    nt_indices: Sequence[int],
    num_codons: Optional[int],
) -> Dict[str, CensorSpec]:
    """Return ``table`` with every candidate at ``nt_indices`` censored in every clade.

    The nucleotide-granular form of :func:`with_global_codon_sites`, and the one
    to prefer. Both express the same three clade-independent reasons from
    mat-pcp's ``masking.compute_exclusions`` -- the config's ``sites_to_exclude``,
    the ProblematicSites VCF, and everything outside ``site_include_range`` --
    but ``{gene}_codon_sites_to_mask.txt`` can only name whole codons, so one
    suspect nucleotide takes its two innocent neighbours with it. Across the six
    v12 production genes that is a **2.85x over-mask**: 121 nucleotide sites
    become 345 masked nucleotides. On S it is 18 against 51.

    Reading ``pcps/{gene}/masked_nt_sites.csv`` instead (mat-pcp#153) censors
    exactly the flagged nucleotides. Nothing else changes: a clade-independent
    site mask is what :attr:`CensorSpec.site_nt_indices` already means, so the
    scatter in ``CensorMaskedMutations._censored_entries_mask``, the ``mode``
    handling, the numerator/denominator tie and the per-sample bounds guard are
    all inherited unchanged -- the codon version's argument for merging rather
    than adding a mechanism, with the widening step removed.

    The two are mutually exclusive at the config layer rather than additive. The
    codon set contains the nucleotide set by construction, so applying both would
    let the wider mask win silently: the run would read as nucleotide-granular in
    its config and be codon-granular in its loss.

    Args:
        table: Per-clade table from :func:`load_censor_table`, covering every
            clade of the corpus.
        nt_indices: **0-indexed** indices into the gene's PCP nucleotide
            sequence, as :func:`parse_masked_nt_sites_file` returns them. May
            contain duplicates and need not be sorted.
        num_codons: Codon count of the corpus, for the bounds check, or ``None``
            when the caller does not know it -- the same case
            :func:`with_global_codon_sites` documents, where the transformer
            configs leave ``max_aa_seq_len`` unset. When it is ``None`` the bound
            is not checked here and ``CensorMaskedMutations``'s per-sample
            ``_max_nt_index`` check is what catches a gene mismatch, on the first
            batch.

    Returns:
        A new table, same keys. Never mutates ``table`` or its tensors.

    Raises:
        ValueError: If ``nt_indices`` is empty, if ``table`` is empty, or if an
            index falls outside the corpus.

    Note:
        The bounds check catches a mask file naming nucleotides the corpus does
        not have, but not the reverse -- a short gene's mask against a long
        gene's corpus is entirely in range. That hole is the same one
        :func:`with_global_codon_sites` documents, and the same
        ``audit_censor`` rule closes it by reading the corpus.
    """
    _require_clade_roster(table, "global nucleotide sites")
    if not nt_indices:
        raise ValueError(
            "nt_indices is empty. Merging no global sites reproduces the "
            "per-clade table exactly, so requesting it is almost certainly a "
            "mistake -- most likely the wrong gene's masked_nt_sites.csv. Leave "
            "TrainingConfig.masked_nt_sites_file unset to censor only the "
            "per-clade table deliberately."
        )
    if num_codons is not None and (
        out_of_range := sorted({idx for idx in nt_indices if idx >= 3 * num_codons})
    ):
        raise ValueError(
            f"global nucleotide index/indices {out_of_range} fall outside a "
            f"corpus of {num_codons} codons ({3 * num_codons} nt). This mask "
            f"file belongs to a different gene -- nothing else ties a "
            f"masked_nt_sites.csv to a corpus."
        )
    if negative := sorted({idx for idx in nt_indices if idx < 0}):
        raise ValueError(
            f"global nucleotide index/indices {negative} are negative. "
            f"nt_indices is 0-indexed, as parse_masked_nt_sites_file returns it."
        )

    return _merge_global_nt_indices(table, set(nt_indices))


def _merge_global_nt_indices(
    table: Dict[str, CensorSpec],
    global_nt: Set[int],
) -> Dict[str, CensorSpec]:
    """Union ``global_nt`` into every clade's ``site_nt_indices``.

    The shared tail of :func:`with_global_codon_sites` and
    :func:`with_global_nt_sites`, which differ only in how they reach this set of
    nucleotide indices. Sharing it is what makes "the two granularities are the
    same mechanism" true rather than asserted.

    Rebuilt from a set union per clade rather than concatenated, so the result is
    a deterministic function of content and idempotent under re-application --
    the same contract :func:`load_censor_table` establishes, and one a
    ``torch.cat`` would break the moment a global site coincided with a clade
    site mask (2 of S's 17 codon sites do).
    """
    return {
        clade: CensorSpec(
            site_nt_indices=torch.tensor(
                sorted(set(spec.site_nt_indices.tolist()) | global_nt),
                dtype=torch.int64,
            ).reshape(-1),
            directional=spec.directional.clone(),
        )
        for clade, spec in table.items()
    }


def load_reference_base_idxs(
    fasta_path: Path, num_codons: Optional[int] = None
) -> Tensor:
    """Base indices of a gene's reference sequence, for reversion censoring (#1099).

    The input to :class:`~dasm2.data.transforms.CensorReferenceReversions`. Read
    from the aggregate manifest's ``founder_sequence``, which for v11 resolves to
    ``clade_founders/{gene}/19A.fa`` -- Wuhan-Hu-1, the reference every reversion
    in that rule reverts *to*. Resolved through the registry rather than as a raw
    path so a run against a different tree cannot silently pick up this tree's
    file, the same reason ``global_mask_path`` takes that route (#1083).

    Args:
        fasta_path: Gene-local reference FASTA. Only the first record is read;
            these files carry one sequence.
        num_codons: Codon count of the corpus, for the length check, or ``None``
            when the caller does not know it -- the same case
            :func:`with_global_codon_sites` documents, where the transformer
            configs leave ``max_aa_seq_len`` unset. When ``None`` the length is
            not checked here and ``CensorReferenceReversions``'s per-sample check
            is what catches a gene mismatch, on the first batch.

    Returns:
        ``(3 * num_codons,)`` int64 tensor of indices into
        :data:`~dasm2.sequence.constants.BASES`.

    Raises:
        ValueError: If its length is not a multiple of 3, if it carries any base
            outside ``ACGT``, or if its length disagrees with ``num_codons``.
            An empty or missing record raises from
            :func:`~dasm2.io.read_fasta_first_sequence`.

    Note:
        Ambiguity is rejected rather than tolerated, and that is the whole reason
        this is a loader instead of two lines at the call site. An ``N`` in the
        reference would make ``parent != reference`` true at that position while
        the destination base index ``4`` matches nothing in the ``(64, 9)``
        candidate tables -- so the rule would censor *nothing* there and report no
        error. Silently censoring less is exactly the failure mode #1099's
        fail-fast guard exists to catch, and it is cheaper to catch it here.
    """
    # `read_fasta_first_sequence` already raises on a missing or empty record,
    # so there is no empty-sequence branch here to be dead code.
    sequence = read_fasta_first_sequence(fasta_path).strip().upper()
    if len(sequence) % 3 != 0:
        raise ValueError(
            f"reference FASTA {fasta_path} is {len(sequence)} nt, which is not a "
            f"multiple of 3. This is a gene-local coding sequence and the rule "
            f"indexes it as {len(sequence) // 3} codons of 3 nucleotides."
        )

    index_of = {base: idx for idx, base in enumerate(BASES)}
    unknown = sorted(set(sequence) - index_of.keys())
    if unknown:
        raise ValueError(
            f"reference FASTA {fasta_path} carries {unknown} outside "
            f"{list(BASES)}. Reversion censoring needs a concrete destination "
            f"base at every position: an ambiguous one compares unequal to every "
            f"parent base and then matches no candidate, so it would censor "
            f"nothing there and raise nothing."
        )

    if num_codons is not None and len(sequence) != 3 * num_codons:
        raise ValueError(
            f"reference FASTA {fasta_path} is {len(sequence)} nt "
            f"({len(sequence) // 3} codons), but the corpus has {num_codons} "
            f"codons ({3 * num_codons} nt). The reference almost certainly "
            f"belongs to a different gene."
        )

    return torch.tensor([index_of[base] for base in sequence], dtype=torch.int64)
