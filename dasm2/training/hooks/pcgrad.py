"""PCGrad: Projecting Conflicting Gradients (issue #1144, Phase 2).

Implements Algorithm 1 of Yu et al., "Gradient Surgery for Multi-Task
Learning" (NeurIPS 2020, https://arxiv.org/abs/2001.06782), as a concrete
:class:`~dasm2.training.hooks.base.GradientCombineHook`. See that module's
docstring for how this plugs into ``Trainer`` — this file only implements
the combination rule itself.
"""

from typing import TYPE_CHECKING, Dict, List

import torch

from dasm2.config import HookConfig
from dasm2.training.hooks.base import GradientCombineHook

if TYPE_CHECKING:
    from dasm2.training.trainer import Trainer

# Guards only the *diagnostic* cosine-similarity computation below (a
# zero-norm task gradient would otherwise divide by zero there). The
# projection formula itself needs no such guard: a task gradient that is
# exactly zero has zero dot product with everything, which is never
# negative, so the projection branch (guarded by ``dot < 0``) can never
# reach a zero-normed denominator in the first place — adding an epsilon
# to that formula would only cost the exact-cancellation invariant this
# hook's own unit tests check for (torch.equal, not allclose).
_DIAGNOSTIC_NORM_EPS = 1e-12


class PCGradHook(GradientCombineHook):
    """Projects each task's gradient onto the normal plane of every other
    conflicting task's gradient, on shared parameters only, before summing.

    For each task ``i``'s (flattened) gradient ``g_i``, iterate over every
    other task ``j`` present in the current micro-batch, in random order:
    if ``g_i . g_j < 0`` (conflicting), update ``g_i`` in place —
    ``g_i <- g_i - (g_i . g_j / ||g_j||^2) * g_j`` — so a third task's
    projection sees the already-updated ``g_i``, matching the paper's own
    Algorithm 1. ``g_j`` is always the *original*, unmodified gradient of
    task ``j`` (a snapshot read, taken before any task's own projection
    loop runs) — never task ``j``'s own evolving projected value, even
    after task ``j``'s loop has run. Non-conflicting gradients
    (``g_i . g_j >= 0``) are returned bit-identical to their input, with no
    floating-point operation performed on them at all.

    No tunable hyperparameters (matching the paper's core Algorithm 1, not
    its optional gradient-magnitude-normalization variants) — the random
    task order draws from the ambient global RNG (``torch.randperm``),
    which is already seeded once per run via ``dasm2.utils.torch_utils.
    set_seed``, so this reproduces identically run-to-run at fixed seed
    with no extra plumbing. For exactly 2 tasks (this repo's only real
    joint-training configuration today), there is only ever one "other"
    task to check, so no permutation is drawn at all and the result is
    order-independent by construction.

    Diagnostics (``pop_diagnostics``) are computed separately from the
    projection algorithm above, over the *original* (pre-projection)
    gradients for every unordered task pair — decoupled from whichever
    directed order the projection loop happens to visit, so "conflict
    fraction" and "cosine mean" describe the tasks' raw relationship, not
    an artifact of iteration order.
    """

    def __init__(self) -> None:
        self._reset_diagnostics()

    def _reset_diagnostics(self) -> None:
        self._diag_calls = 0
        self._diag_pair_count = 0
        self._diag_conflict_count = 0
        self._diag_cosine_sum = 0.0
        self._diag_norm_sum: Dict[str, float] = {}
        self._diag_norm_count: Dict[str, int] = {}
        self._diag_ratio_sum = 0.0
        self._diag_ratio_count = 0

    def combine_gradients(
        self,
        trainer: "Trainer",
        per_task_grads: Dict[str, List[torch.Tensor]],
        shared_params: List[torch.nn.Parameter],
    ) -> Dict[str, List[torch.Tensor]]:
        del trainer  # unused: no per-run seed plumbing needed, see class docstring
        keys = list(per_task_grads.keys())
        flat: Dict[str, torch.Tensor] = {
            k: torch.cat([g.reshape(-1) for g in per_task_grads[k]]) for k in keys
        }
        self._record_diagnostics(keys, flat)

        pc_flat: Dict[str, torch.Tensor] = {k: flat[k].clone() for k in keys}
        for i in keys:
            others = [k for k in keys if k != i]
            if len(others) > 1:
                perm = torch.randperm(len(others)).tolist()
                others = [others[idx] for idx in perm]
            for j in others:
                dot = torch.dot(pc_flat[i], flat[j])
                if dot < 0:
                    pc_flat[i] = pc_flat[i] - (dot / flat[j].pow(2).sum()) * flat[j]

        shapes = [p.shape for p in shared_params]
        result: Dict[str, List[torch.Tensor]] = {}
        for k in keys:
            parts: List[torch.Tensor] = []
            offset = 0
            for shape in shapes:
                n = shape.numel()
                parts.append(pc_flat[k][offset : offset + n].reshape(shape))
                offset += n
            result[k] = parts
        return result

    def _record_diagnostics(
        self, keys: List[str], flat: Dict[str, torch.Tensor]
    ) -> None:
        self._diag_calls += 1
        norms = {k: flat[k].norm().item() for k in keys}
        for k in keys:
            self._diag_norm_sum[k] = self._diag_norm_sum.get(k, 0.0) + norms[k]
            self._diag_norm_count[k] = self._diag_norm_count.get(k, 0) + 1
        for idx_i in range(len(keys)):
            for idx_j in range(idx_i + 1, len(keys)):
                i, j = keys[idx_i], keys[idx_j]
                cosine = torch.dot(flat[i], flat[j]).item() / (
                    norms[i] * norms[j] + _DIAGNOSTIC_NORM_EPS
                )
                self._diag_cosine_sum += cosine
                self._diag_pair_count += 1
                if cosine < 0:
                    self._diag_conflict_count += 1
        if len(keys) == 2:
            lo, hi = sorted(norms.values())
            self._diag_ratio_sum += hi / (lo + _DIAGNOSTIC_NORM_EPS)
            self._diag_ratio_count += 1

    def pop_diagnostics(self) -> Dict[str, float]:
        """Per-eval-window summary of the *unordered* task-pair relationship
        (conflict fraction, mean cosine similarity), each task's own mean
        raw gradient norm (flattened key ``pcgrad_grad_norm_<task_log_key>``),
        and — only when exactly 2 tasks are ever present in a call — the
        mean ratio between the larger and smaller task gradient norm.

        All of the above are computed on the *pre-projection* gradients
        (see class docstring); this never reflects the actual combined
        gradient PCGrad applied, only the raw relationship between tasks.
        Resets the accumulator (matching ``OptimizerStepMetrics.compute()``
        + ``.reset()``'s own per-eval-window contract, which
        ``Trainer._run_eval_boundary`` already calls this same way).
        ``pcgrad_calls`` distinguishes "this window's stats are real" (>0)
        from "this window never had a 2+-task micro-batch" (all zeros).
        """
        out: Dict[str, float] = {"pcgrad_calls": float(self._diag_calls)}
        if self._diag_pair_count > 0:
            out["pcgrad_conflict_fraction"] = (
                self._diag_conflict_count / self._diag_pair_count
            )
            out["pcgrad_cosine_mean"] = self._diag_cosine_sum / self._diag_pair_count
        for key, total in self._diag_norm_sum.items():
            out[f"pcgrad_grad_norm_{key}"] = total / self._diag_norm_count[key]
        if self._diag_ratio_count > 0:
            out["pcgrad_grad_norm_ratio_mean"] = (
                self._diag_ratio_sum / self._diag_ratio_count
            )
        self._reset_diagnostics()
        return out

    def to_config(self) -> HookConfig:
        return HookConfig(type="PCGradHook")
