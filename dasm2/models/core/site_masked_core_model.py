"""Site-masked DASM: Halpern-Bruno scoring of a context-predicted F (issue #1550).

For each site ``j`` the model reads a fitness vector ``F_j`` (20 entries)
from the shared ``sf_projection`` applied to an encoder pass in which site
``j`` holds the mask token, and returns

    log f_j(y | x) = log phi(F_j(y) - F_j(x)),   phi(S) = S / (1 - exp(-S)).

The resident ``x`` never reaches the encoder at ``j``, so the model cannot
learn a free wt-to-mut table. The parent column is ``log phi(0) = 0``, the
same gauge as ``CoreModel``'s zapped output, so ``DASMLoss`` runs unchanged.

With the head's ``pair_table`` on (issue #1670), a context-free table is
added at scored sites: ``log f_j(y | x) = S[x, y] + log phi(F_j(y) - F_j(x))``,
with ``S``'s diagonal zapped, so the parent column stays 0.

The model owns its masking, so every caller of ``forward`` or
``selection_factors_of_aa_str`` gets the masked path:

- **Training** (``model.training``): one pass per batch with a uniformly
  random ``mask_rate`` of each row's maskable sites masked, always with the
  mask token. ``prepare_training_batch`` draws the sites, stores them under
  ``SITE_MASK_KEY`` and replaces ``batch["loss_mask"]`` with its AND with
  them, so the loss scores masked sites only, changed and unchanged alike.
  The trainer calls it before counting ``loss_mask`` for loss
  normalization, so that count is the masked sites too; a training-mode
  ``forward`` on an unprepared batch calls it itself.
- **Evaluation**: ``eval_num_partitions`` passes. Each maskable site is
  masked in exactly one pass, chosen by a partition that depends only on
  the row's tokens and ``partition_seed``, and takes its ``F_j`` from it.
- **Scan** (``selection_factors_of_aa_str``): one pass per maskable site of
  the reference, returning ``exp(F_j(y) - F_j(wt))``, ESM-1v's masked
  marginal, or ``exp(S[wt, y]) phi(F_j(y) - F_j(wt))`` with ``pair_table``.

This is DESIGN.md D2's exception: the encoder runs more than once per
evaluation batch.
"""

import hashlib
from typing import Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn

from dasm2.config import HeadConfig
from dasm2.data.masking import (
    DEFAULT_MASK_BATCH_SIZE,
    MaskingConfig,
    maskable_token_mask,
    masked_marginal_scan,
    select_and_corrupt,
)
from dasm2.heads.site_masked import SiteMaskedSelectionFactorHead
from dasm2.heads.types import HeadType
from dasm2.models.core.base import CoreModel
from dasm2.models.core.sf_projection import (
    SelectionFactorProjection,
    resident_relative_log_phi,
)
from dasm2.sequence.constants import (
    AA_PADDING_TOKEN,
    AA_SEPARATOR_IDX,
    NUM_AMINO_ACIDS,
)
from dasm2.sequence.mlm_constants import MLM_MASK_IDX

#: Batch key of the training forward's masked sites, ``[B, L]`` bool.
SITE_MASK_KEY = "site_masked_sites"


def _row_partition_seed(row_tokens: torch.Tensor, partition_seed: int) -> int:
    """Generator seed for one row's partition, from its tokens and the seed.

    Trailing padding is stripped, so a row hashes the same alone and inside
    a padded batch.
    """
    is_real = row_tokens != AA_PADDING_TOKEN
    length = int(is_real.nonzero().max()) + 1 if bool(is_real.any()) else 0
    digest = hashlib.blake2b(
        row_tokens[:length].to(torch.int8).numpy().tobytes(),
        digest_size=8,
        key=partition_seed.to_bytes(8, "little", signed=True),
    ).digest()
    return int.from_bytes(digest, "little") & (2**63 - 1)


def partition_blocks(
    tokens: torch.Tensor, num_partitions: int, partition_seed: int
) -> torch.Tensor:
    """Assign each maskable site to one of ``num_partitions`` passes.

    Within a row, a seeded random permutation of the maskable sites is dealt
    round-robin, so block sizes differ by at most one. The assignment
    depends only on the row's tokens and ``partition_seed``.

    The per-row hash and permutation run on CPU, once per evaluation batch;
    the ``num_partitions`` encoder passes that follow dominate its cost.

    Args:
        tokens: ``[B, L]`` parent token ids, padded with ``AA_PADDING_TOKEN``.
        num_partitions: Number of blocks.
        partition_seed: Seed shared by every row.

    Returns:
        ``[B, L]`` long block ids on ``tokens.device``; -1 where a site is
        not maskable.
    """
    tokens_cpu = tokens.cpu()
    maskable = maskable_token_mask(tokens_cpu)
    blocks = torch.full_like(tokens_cpu, -1, dtype=torch.long)
    for row in range(tokens_cpu.shape[0]):
        positions = maskable[row].nonzero().squeeze(-1)
        generator = torch.Generator().manual_seed(
            _row_partition_seed(tokens_cpu[row], partition_seed)
        )
        order = torch.randperm(positions.shape[0], generator=generator)
        blocks[row, positions] = order % num_partitions
    return blocks.to(tokens.device)


class SiteMaskedCoreModel(CoreModel):
    """``CoreModel`` whose ``F_j`` comes from a pass with site ``j`` masked.

    Built by ``models/factory.py`` for a ``SiteMaskedSelectionFactorHead``,
    which must be the model's only head. Reads the masking parameters off
    that head.
    """

    def __init__(
        self,
        encoder: nn.Module,
        prediction_heads: Optional[List[HeadConfig]] = None,
        sf_projection: Optional[SelectionFactorProjection] = None,
        mask_batch_size: int = DEFAULT_MASK_BATCH_SIZE,
    ) -> None:
        """Build the model; see ``CoreModel.__init__`` for the shared args.

        Args:
            mask_batch_size: Masked copies per encoder pass in the one-site
                scan. Bounds memory; does not change the scores.

        Raises:
            ValueError: If ``mask_batch_size`` is less than 1, or the heads
                are anything but a single ``SiteMaskedSelectionFactorHead``.
        """
        if mask_batch_size < 1:
            raise ValueError(
                f"mask_batch_size must be at least 1, got {mask_batch_size}"
            )
        super().__init__(encoder, prediction_heads, sf_projection)
        head_keys = list(self.prediction_heads.keys())
        if head_keys != [HeadType.SITE_MASKED_SELECTION_FACTOR_HEAD.value]:
            raise ValueError(
                "SiteMaskedCoreModel takes exactly one head, "
                f"{HeadType.SITE_MASKED_SELECTION_FACTOR_HEAD.value}; got "
                f"{head_keys}. Its masked passes produce F, not embeddings "
                "another head could share."
            )
        self.mask_batch_size = mask_batch_size

    @property
    def _masked_head(self) -> SiteMaskedSelectionFactorHead:
        return cast(
            SiteMaskedSelectionFactorHead,
            self.prediction_heads[HeadType.SITE_MASKED_SELECTION_FACTOR_HEAD.value],
        )

    def _encode(
        self, batch: Dict[str, torch.Tensor], encoder_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One encoder pass on ``encoder_tokens``; returns embeddings and F.

        The attention mask and every other encoder field come from ``batch``,
        whose ``aa_parents_idxs`` stay the true parents.
        """
        encoder_input = self._build_encoder_input(batch)
        encoder_input["token_ids"] = encoder_tokens
        embeddings = self.encoder(encoder_input)
        assert self.sf_projection is not None  # guaranteed by _validate_sf_projection
        return embeddings, self.sf_projection(embeddings)

    def prepare_training_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """Draw this training batch's masked sites and narrow ``loss_mask``.

        Stores the ``[B, L]`` selection under ``SITE_MASK_KEY`` and replaces
        ``batch["loss_mask"]`` with ``loss_mask & selection``. A batch that
        already carries a selection keeps it, so a forward replayed on the
        same batch masks the same sites. Does nothing in eval mode.
        """
        if not self.training or SITE_MASK_KEY in batch:
            return
        parents = batch["aa_parents_idxs"]
        masking_config = MaskingConfig(
            mask_rate=self._masked_head.mask_rate,
            mask_token_prob=1.0,
            random_token_prob=0.0,
        )
        selected, _ = select_and_corrupt(
            parents, maskable_token_mask(parents), masking_config, MLM_MASK_IDX
        )
        batch[SITE_MASK_KEY] = selected
        if "loss_mask" in batch:
            batch["loss_mask"] = batch["loss_mask"] & selected

    def _partitioned_F(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """F from the evaluation partition: each site's F from its own pass."""
        parents = batch["aa_parents_idxs"]
        head = self._masked_head
        blocks = partition_blocks(
            parents, head.eval_num_partitions, head.partition_seed
        )
        # Sites in no block (X, ^, padding) keep 0, the gauge of an unscored site.
        F = parents.new_zeros((*parents.shape, NUM_AMINO_ACIDS), dtype=torch.float32)
        for block in range(head.eval_num_partitions):
            in_block = blocks == block
            if not bool(in_block.any()):
                continue
            _, F_block = self._encode(
                batch, torch.where(in_block, MLM_MASK_IDX, parents)
            )
            F = torch.where(in_block[..., None], F_block.float(), F)
        return F

    def _log_f(self, F: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
        """``log phi(F(y) - F(x))``, plus ``S[x, y]`` with ``pair_table``."""
        log_f = resident_relative_log_phi(F, parents)
        if self._masked_head.pair_table:
            log_f = log_f + self._masked_head.pair_log_f(parents)
        return log_f

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        head_names: Optional[List[HeadType]] = None,
    ) -> Dict:
        """Masked forward; see the module docstring for the three modes.

        Returns:
            Dict with the head's ``log phi(F_j(y) - F_j(x))`` (plus
            ``S[x, y]`` with ``pair_table``) under its ``HeadType``, 0 at
            every site that was not scored; ``"raw_selection_factors"``, the
            same log f before the head; and ``"site_fitness"``, the F behind
            it (0 at unscored sites). In training mode it also carries
            ``"sequence_embeddings"`` from the one pass; evaluation runs
            several passes and has no single embedding to return.
        """
        if "aa_parents_idxs" not in batch:
            raise ValueError("Batch must contain 'aa_parents_idxs' field")
        if "mask" not in batch:
            raise ValueError("Batch must contain 'mask' field")
        head_type = HeadType.SITE_MASKED_SELECTION_FACTOR_HEAD
        for requested in head_names or [head_type]:
            self.require_head(requested)

        parents = batch["aa_parents_idxs"]
        outputs: Dict = {}
        if self.training:
            # A no-op when the trainer has already prepared this batch.
            self.prepare_training_batch(batch)
            scored = batch[SITE_MASK_KEY]
            embeddings, F = self._encode(
                batch, torch.where(scored, MLM_MASK_IDX, parents)
            )
            outputs["sequence_embeddings"] = embeddings
        else:
            scored = maskable_token_mask(parents)
            F = self._partitioned_F(batch)

        F = F.float()
        scored_3d = scored.unsqueeze(-1)
        log_f = torch.where(scored_3d, self._log_f(F, parents), 0.0)
        outputs["site_fitness"] = torch.where(scored_3d, F, 0.0)
        outputs["raw_selection_factors"] = log_f
        outputs[head_type] = self._masked_head(log_f, batch)
        return outputs

    def selection_factors_of_aa_str(
        self,
        aa_sequence: str,
        head_type: HeadType = HeadType.SITE_MASKED_SELECTION_FACTOR_HEAD,
        tdi_tokens: Optional[str] = None,
        distance: Optional[float] = None,
        raw: bool = False,
    ) -> List[torch.Tensor]:
        """One-site masked scan: ``exp(F_j(y) - F_j(wt))`` per maskable site.

        Each maskable position of the reference is masked in its own pass,
        ``mask_batch_size`` copies per encoder call. ``functional_eval``
        sums the log over a variant's mutated sites, which gives ESM-1v's
        masked marginal. This is ``exp(S)``, not ``phi(S)``: ``phi`` is
        increasing, so the ranks within a site are the same.

        With ``pair_table`` the table term breaks that rank equivalence, so
        the scan returns the forward's own ``f_j(y | wt) = exp(S[wt, y])
        phi(F_j(y) - F_j(wt))`` instead. The two forms differ in scale, so a
        model with ``S = 0`` does not score as one built without the table.

        Args:
            aa_sequence: Unified sequence string ("SEQ1^SEQ2" or single).
            head_type: Must be ``SITE_MASKED_SELECTION_FACTOR_HEAD``.
            tdi_tokens: Optional 3Di token string, as for ``CoreModel``.
            distance: Maturation distance; required when the encoder is
                distance-aware, as for ``CoreModel``.
            raw: Not supported; the scan has no zapped/raw distinction.

        Returns:
            One ``[segment_len, 20]`` tensor per segment, with 1 in the WT
            column. ``^`` and ``X`` positions get no row.

        Raises:
            ValueError: If ``raw`` is set, the head is missing, the encoder
                needs a distance that was not given, or the sequence has no
                maskable position.
        """
        if raw:
            raise ValueError(
                "SiteMaskedCoreModel does not support raw=True: the scan "
                "returns masked-marginal scores, with no zapped/raw distinction."
            )
        self.require_head(head_type)
        batch, tokens_1d = self._build_inference_batch(
            aa_sequence, tdi_tokens=tdi_tokens
        )
        self._add_inference_distance(batch, distance)

        def score_chunk(
            batch_tokens: torch.Tensor,
            positions: torch.Tensor,
            ref_residues: torch.Tensor,
        ) -> torch.Tensor:
            num_copies = batch_tokens.shape[0]
            chunk_batch = {
                key: value.expand(num_copies, *value.shape[1:])
                for key, value in batch.items()
            }
            _, F = self._encode(chunk_batch, batch_tokens)
            F_site = F[torch.arange(num_copies, device=F.device), positions].float()
            if not self._masked_head.pair_table:
                return F_site - F_site.gather(-1, ref_residues.unsqueeze(-1))
            # One site per copy: the forward's composition on [N, 1] sites.
            return self._log_f(F_site[:, None], ref_residues[:, None])[:, 0]

        with torch.no_grad():
            return masked_marginal_scan(
                tokens_1d,
                MLM_MASK_IDX,
                AA_SEPARATOR_IDX,
                self.mask_batch_size,
                score_chunk,
            )
