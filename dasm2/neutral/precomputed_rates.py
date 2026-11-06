"""Carrier neutral model for datasets whose rates/CSPs are already in the data.

``PrecomputedRatesNeutralModel`` exists to make a multihit correction
*configurable* on Parquet corpora that have no neutral model at all — Compara
being the motivating case: its per-family GTR rates are baked into the
``nt_rates``/``nt_csps`` columns, so ``compara-training-v10_dasm_full`` trains
with ``neutral_model: null``.

The problem that creates: ``multihit_model_name`` and ``multihit_log_factors``
live on ``MultihitWrappableNeutralModelConfig``, i.e. only *inside* a
neutral-model block. With no neutral model there is no config that expresses
"rates come from the data, but do apply this multihit correction" — the Parquet
dispatch in ``train_command`` reads the correction off the configured neutral
model, so it stays ``None``, and ``fit_multihit_factors: true`` refuses to start.
This class fills exactly that hole: a ``RatesCSPNeutralModel`` carrying nothing
but the ``MultihitWrapper`` that ``_maybe_wrap_multihit`` puts around it, which
reuses the whole existing multihit path (the ``isinstance`` check in the Parquet
dispatch, ``MultihitWrapper.to_config()``'s round-trip of the live factors into
checkpoint metadata, and the ``getattr(neutral_model, "multihit_model", None)``
lookup that gates the refit hook) rather than duplicating it at
``TrainingConfig`` level.

Why the rate methods raise
--------------------------
Being a ``RatesCSPNeutralModel`` that cannot produce rates is the point, and it
has to fail loudly rather than return something plausible. A carrier that
*returned* rates would let some path silently fit branch lengths or multihit
factors against rates it invented, instead of against the ones the forward
pass consumed.

Today every consumer reads the stored columns. On a Parquet bundle,
``BranchLengthDatasetBundle._resolve_nt_rates_and_csps`` streams them through
``create_bl_optimization_dataloader``, and ``collect_fit_inputs`` (the shared
entry point for ``CodonStrengthOptimizer`` and ``MultihitOptimizer``) goes
through it too. The carrier is therefore never asked for rates on a
column-carrying file, and the raise is the guard for any future path that
tries. It is also what makes other misuses fail at once:

- **Rate-free Parquet file:** the loader checks ``computes_rates`` (False
  here) and refuses the carrier, since nothing could supply rates
  (issue #1299).
- **CSV corpus:** rejected by the raise itself, since rates are not
  precomputed there.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

from dasm2.neutral.base import RatesCSPNeutralModel

if TYPE_CHECKING:
    from dasm2.config import PrecomputedRatesNeutralModelConfig

PRECOMPUTED_RATES_MODEL_NAME = "PrecomputedRates"

_NO_RATES_MESSAGE = (
    "PrecomputedRatesNeutralModel cannot compute rates or CSPs: on a "
    "precomputed-rates dataset they come from the data (the Parquet "
    "nt_rates/nt_csps columns), and this model exists only to carry a "
    "multihit correction for corpora that have no neutral model. Reaching "
    "this call means some code path tried to compute neutral rates anyway — "
    "either the dataset is not actually rate-precomputed (a CSV corpus needs "
    "a real neutral model, e.g. ThriftySHM), or the caller took "
    "BranchLengthDatasetBundle.get_branch_length_optimization_inputs()'s "
    "compute-on-the-fly fallback, which a Parquet bundle cannot satisfy."
)


class PrecomputedRatesNeutralModel(RatesCSPNeutralModel):
    """Parameter-free neutral model for datasets that already carry their rates.

    Carries no rate model of its own — both rate methods raise
    ``NotImplementedError`` (see the module docstring for why that is
    load-bearing rather than a stub). Its only purpose is to be wrapped by
    ``MultihitWrapper`` so a multihit correction can be configured on a corpus
    with ``neutral_model: null``:

    .. code-block:: yaml

        neutral_model:
          type: PrecomputedRates
          multihit_model_name: ThriftyHumV0.2-59-hc-tangshm
    """

    computes_rates = False

    @property
    def name(self) -> str:
        """Human-readable identifier for this neutral model."""
        return PRECOMPUTED_RATES_MODEL_NAME

    def to_config(self) -> "PrecomputedRatesNeutralModelConfig":
        """Reconstruct the (parameter-free) config for this model.

        Portable by construction: there is no archive, path, or parameter to
        record. When wrapped, ``MultihitWrapper.to_config()`` fills in
        ``multihit_model_name``/``multihit_log_factors`` on top of this.
        """
        from dasm2.config import PrecomputedRatesNeutralModelConfig  # noqa: PLC0415

        return PrecomputedRatesNeutralModelConfig()

    def compute_rates_and_csps(
        self,
        sequences: Union[str, List[str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Always raises: rates come from the data, not from this model.

        Raises:
            NotImplementedError: Unconditionally.
        """
        raise NotImplementedError(_NO_RATES_MESSAGE)

    def compute_rates_and_csps_vectorized(
        self,
        sequences: Union[str, List[str]],
        context: Optional[Dict[str, Any]] = None,
        batch_size: int = 4096,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Always raises, like ``compute_rates_and_csps``.

        Overridden explicitly rather than left to the base class's delegation,
        because this is the method the dataset precompute path actually calls
        (``_precompute_neutral_outputs_vectorized``) — an override keeps the
        raise from depending on that delegation staying in place.

        Raises:
            NotImplementedError: Unconditionally.
        """
        raise NotImplementedError(_NO_RATES_MESSAGE)
