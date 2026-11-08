"""DASMfit alpha optimization hook for alternating optimization.

Implements the α-step of DASMfit alternating optimization: for each clonal
family, solve a Poisson GLM for the per-family α coefficients that weight the
k residual directions r_i(j, a) predicted by DASMFitResidualHead.

The optimization problem per family f:

    α^(f) = argmax_α  Σ_j Σ_a [ N_j(a) · λ - M_j(a) · exp(λ) ] - (σ/2) ‖α‖²

where λ_j(a) = Σ_i α_i · r_i(j, a). This is concave (Poisson GLM with L2
penalty), so Newton's method with analytical gradient and Hessian converges
reliably in 3-5 iterations for k ≤ 5.
"""

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from dasm2.heads.types import HeadType
from dasm2.training.hooks.base import TrainingHook
from dasm2.utils.batch_operations import move_batch_to_device

if TYPE_CHECKING:
    from dasm2.config import HookConfig
    from dasm2.training.trainer import Trainer

logger = logging.getLogger(__name__)

# Default batch size for model inference during alpha optimization.
DEFAULT_INFERENCE_BATCH_SIZE = 32


def poisson_objective(
    alpha: Tensor,
    r: Tensor,
    D: Tensor,
    M: Tensor,
    sigma: float,
) -> Tensor:
    """Compute the penalized Poisson log-likelihood objective.

    Objective = Σ_j Σ_a [ D_j(a) · λ - M_j(a) · exp(λ) ] - (σ/2) ‖α‖²

    where λ_j(a) = Σ_i α_i · r_i(j, a).

    Args:
        alpha: Per-family coefficients [k].
        r: Residual directions [L, k, 20] from DASMFitResidualHead.
        D: Observed mutation counts [L, 20] ("data").
        M: Expected mutation counts [L, 20] under the base Q_j ("model").
        sigma: L2 penalty strength.

    Returns:
        Scalar objective value (to be maximized).
    """
    # λ_j(a) = Σ_i α_i · r_i(j, a): [L, 20]
    lam = torch.einsum("i,lia->la", alpha, r)
    obj = (D * lam - M * torch.exp(lam)).sum()
    obj = obj - 0.5 * sigma * (alpha * alpha).sum()
    return obj


def _compute_gradient_and_hessian(
    alpha: Tensor,
    r: Tensor,
    D: Tensor,
    M: Tensor,
    sigma: float,
) -> Tuple[Tensor, Tensor]:
    """Compute analytical gradient and Hessian for the Poisson GLM.

    Gradient:
        g_i = Σ_j Σ_a [ (D_j(a) - M_j(a)·exp(λ)) · r_i(j,a) ] - σ·α_i

    Hessian:
        H_ij = -Σ_s Σ_a [ M_s(a)·exp(λ) · r_i(s,a) · r_j(s,a) ] - σ·δ_ij

    Args:
        alpha: Current coefficients [k].
        r: Residual directions [L, k, 20].
        D: Observed counts [L, 20] ("data").
        M: Expected counts [L, 20] under the base Q_j ("model").
        sigma: L2 penalty strength.

    Returns:
        Tuple of (gradient [k], Hessian [k, k]).
    """
    # λ_j(a) = Σ_i α_i · r_i(j, a): [L, 20]
    lam = torch.einsum("i,lia->la", alpha, r)
    exp_lam = torch.exp(lam)  # [L, 20]

    # Gradient: g_i = Σ_{j,a} (D - M·exp(λ)) · r_i - σ·α_i
    residual = D - M * exp_lam  # [L, 20]
    grad = torch.einsum("la,lia->i", residual, r) - sigma * alpha

    # Hessian: H_ij = -Σ_{j,a} M·exp(λ) · r_i · r_j - σ·δ_ij
    # Weight each (j,a) entry: w = M·exp(λ)  [L, 20]
    w = M * exp_lam  # [L, 20]
    # H_ij = -Σ_{j,a} w_{j,a} · r_i(j,a) · r_j(j,a)
    # Flatten L,20 -> n, then it's -r_flat^T @ diag(w_flat) @ r_flat
    weighted_r = w.unsqueeze(1) * r  # [L, k, 20]
    hess = -torch.einsum("lia,lja->ij", weighted_r, r)
    # Add penalty: -σ·I
    hess.diagonal().sub_(sigma)

    return grad, hess


def solve_alpha_newton(
    r: Tensor,
    D: Tensor,
    M: Tensor,
    sigma: float,
    alpha_init: Optional[Tensor] = None,
    max_iter: int = 10,
    tol: float = 1e-6,
) -> Tuple[Tensor, int]:
    """Solve per-family Poisson GLM for α via Newton's method.

    Maximizes the penalized Poisson log-likelihood:
        Σ_j Σ_a [ D_j(a) · λ - M_j(a) · exp(λ) ] - (σ/2) ‖α‖²
    where λ_j(a) = Σ_i α_i · r_i(j, a).

    The problem is strictly concave (negative definite Hessian), guaranteeing
    convergence. For k ≤ 5, typically converges in 3-5 iterations.

    Args:
        r: Residual directions [L, k, 20] from DASMFitResidualHead.
        D: Observed mutation counts [L, 20] ("data").
        M: Expected mutation counts [L, 20] under the base Q_j ("model").
        sigma: L2 penalty strength (must be > 0).
        alpha_init: Initial α values [k]. Defaults to zeros.
        max_iter: Maximum Newton iterations.
        tol: Convergence tolerance on gradient norm.

    Returns:
        Tuple of (optimal alpha [k], number of iterations used).

    Raises:
        ValueError: If sigma is not positive.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    k = r.shape[1]
    if alpha_init is not None:
        alpha = alpha_init.clone()
    else:
        alpha = torch.zeros(k, dtype=r.dtype, device=r.device)

    for iteration in range(max_iter):
        grad, hess = _compute_gradient_and_hessian(alpha, r, D, M, sigma)

        if grad.norm() < tol:
            return alpha, iteration

        # Newton step: delta = -H^{-1} g
        # H is negative definite, so -H is positive definite -> use Cholesky
        neg_hess = -hess
        try:
            L_chol = torch.linalg.cholesky(neg_hess)
            delta = torch.cholesky_solve(grad.unsqueeze(1), L_chol).squeeze(1)
        except torch.linalg.LinAlgError:
            # Fallback to general solve if Cholesky fails (shouldn't happen
            # for strictly concave problems, but guard against numerical edge cases)
            delta = torch.linalg.solve(neg_hess, grad)

        alpha = alpha + delta

    return alpha, max_iter


class DASMFitAlphaOptimizer(TrainingHook):
    """Pre-cycle hook that solves per-family Poisson GLMs for α coefficients.

    Follows the BranchLengthOptimizer pattern: runs model inference in pre_cycle,
    solves per-family optimization, stores results for the θ-step.

    The hook:
    1. Runs the frozen encoder + DASMFitResidualHead to get r_i(j, a) per family
    2. Solves per-family Poisson GLMs for optimal α using Newton's method
    3. Writes α values directly into the dataset items as an ``"alpha"`` field

    Because α is stored on the dataset, it flows through the standard DataLoader
    collator into each batch, where ``DASMFitResidualHead.compute_loss()`` reads
    it from ``targets["alpha"]``. No trainer modifications are needed.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        max_newton_iter: int = 10,
        newton_tol: float = 1e-6,
        inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    ) -> None:
        """Initialize DASMFitAlphaOptimizer.

        Args:
            sigma: L2 penalty strength for the Poisson GLM. Must be positive.
            max_newton_iter: Maximum Newton iterations per family.
            newton_tol: Convergence tolerance on gradient norm.
            inference_batch_size: Batch size for model inference.
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = sigma
        self.max_newton_iter = max_newton_iter
        self.newton_tol = newton_tol
        self.inference_batch_size = inference_batch_size

    def pre_cycle(self, trainer: "Trainer", **context: Any) -> None:
        """Run the α-step: model inference + per-family Newton solve.

        Computes r_i(j, a) for all families using the current model, then
        solves per-family Poisson GLMs for optimal α. Results are written
        directly into the dataset items as an ``"alpha"`` field so they flow
        through the collator into each training batch.

        Args:
            trainer: Trainer instance with model and training state.
            **context: Must include ``train_data`` and ``cycle_count``.
        """
        required = ("train_data", "cycle_count")
        missing = [k for k in required if k not in context]
        if missing:
            raise ValueError(f"Hook context missing required keys: {missing}")

        train_data = context["train_data"]
        cycle_count = context["cycle_count"]

        if trainer.is_main_process:
            logger.info(
                "Alpha optimizer: solving per-family α (cycle %d, σ=%.4f)",
                cycle_count,
                self.sigma,
            )

        start_time = time.perf_counter()

        # Run inference and solve per-family alpha
        alpha_tensor, newton_stats = self._solve_all_families(trainer, train_data)

        # Write alpha into dataset items so it flows through the collator.
        dataset = _unwrap_bundle(train_data)
        _write_alpha_to_dataset(dataset, alpha_tensor)

        duration = time.perf_counter() - start_time
        self._log_diagnostics(alpha_tensor, newton_stats, duration, trainer)

    def post_cycle(self, trainer: "Trainer", **context: Any) -> None:
        """No post-cycle action needed for alpha optimization."""

    def to_config(self) -> "HookConfig":
        """Reconstruct HookConfig from hook state.

        Returns:
            HookConfig matching current hook configuration.
        """
        from dasm2.config import HookConfig

        return HookConfig(  # type: ignore[call-arg]
            type="DASMFitAlphaOptimizer",
            sigma=self.sigma,
            max_newton_iter=self.max_newton_iter,
            newton_tol=self.newton_tol,
            inference_batch_size=self.inference_batch_size,
        )

    def _solve_all_families(
        self,
        trainer: "Trainer",
        train_data: Any,
    ) -> Tuple[Tensor, Dict[str, List[int]]]:
        """Run model inference and solve alpha for all families in the dataset.

        Args:
            trainer: Trainer instance.
            train_data: DatasetBundle or raw Dataset. If a bundle, its collator
                is used for DataLoader construction (matching the
                BranchLengthOptimizer pattern).

        Returns:
            Tuple of (alpha tensor [N, k], newton iteration stats dict).
        """
        from dasm2.data.evolutionary_dataset import DatasetBundle

        model = trainer.unwrapped_model
        device = str(model.device)
        model.eval()

        model.require_head(HeadType.DASM_FIT_RESIDUAL_HEAD)

        all_alphas: List[Tensor] = []
        newton_iters: List[int] = []

        # Extract dataset and collator from bundle (following BranchLengthOptimizer)
        if isinstance(train_data, DatasetBundle):
            dataset = train_data.dataset
            collate_fn = train_data.collator
        else:
            dataset = train_data
            collate_fn = None

        dl_kwargs: Dict[str, Any] = {
            "batch_size": self.inference_batch_size,
            "shuffle": False,
        }
        if collate_fn is not None:
            dl_kwargs["collate_fn"] = collate_fn
        dataloader = DataLoader(dataset, **dl_kwargs)

        with torch.no_grad():
            for batch in dataloader:
                batch = move_batch_to_device(batch, device)
                outputs = model(batch, head_names=[HeadType.DASM_FIT_RESIDUAL_HEAD])
                # r: [B, L, k, 20]
                r_batch = outputs[HeadType.DASM_FIT_RESIDUAL_HEAD].cpu()
                D_batch = batch["D"].cpu()  # [B, L, 20]
                M_batch = batch["M"].cpu()  # [B, L, 20]
                batch_size = r_batch.shape[0]

                # Warm-start from previous cycle's alpha if present in batch
                prev_alpha_batch = batch.get("alpha")

                for i in range(batch_size):
                    r_i = r_batch[i]  # [L, k, 20]
                    D_i = D_batch[i]  # [L, 20]
                    M_i = M_batch[i]  # [L, 20]

                    alpha_init = None
                    if prev_alpha_batch is not None:
                        alpha_init = prev_alpha_batch[i].cpu()

                    alpha, n_iter = solve_alpha_newton(
                        r_i,
                        D_i,
                        M_i,
                        sigma=self.sigma,
                        alpha_init=alpha_init,
                        max_iter=self.max_newton_iter,
                        tol=self.newton_tol,
                    )
                    all_alphas.append(alpha)
                    newton_iters.append(n_iter)

        alpha_tensor = torch.stack(all_alphas)  # [N, k]

        stats: Dict[str, List[int]] = {"newton_iters": newton_iters}
        return alpha_tensor, stats

    def _log_diagnostics(
        self,
        alpha: Tensor,
        stats: Dict[str, List[int]],
        duration: float,
        trainer: "Trainer",
    ) -> None:
        """Log per-cycle alpha optimization diagnostics.

        Args:
            alpha: Optimized alpha tensor [N, k].
            stats: Newton iteration statistics.
            duration: Wall time for the alpha step.
            trainer: Trainer instance for rank checking.
        """
        if not trainer.is_main_process:
            return

        iters = stats["newton_iters"]
        alpha_abs = alpha.abs()
        logger.info(
            "Alpha step: %.2fs | families=%d | "
            "|α| mean=%.4f std=%.4f max=%.4f | "
            "Newton iters mean=%.1f max=%d",
            duration,
            alpha.shape[0],
            alpha_abs.mean().item(),
            alpha_abs.std().item(),
            alpha_abs.max().item(),
            sum(iters) / len(iters) if iters else 0,
            max(iters) if iters else 0,
        )


def _unwrap_bundle(data: Any) -> Any:
    """Unwrap a DatasetBundle to its underlying dataset.

    Args:
        data: DatasetBundle or raw Dataset.

    Returns:
        The underlying dataset.
    """
    from dasm2.data.evolutionary_dataset import DatasetBundle

    if isinstance(data, DatasetBundle):
        return data.dataset
    return data


def _write_alpha_to_dataset(dataset: Any, alpha: Tensor) -> None:
    """Write per-family alpha values into dataset items.

    Supports datasets that store items as a mutable list of dicts:
    - ``items`` attribute (e.g. test fixtures)
    - ``_precomputed`` attribute (TransformDataset in precomputed mode)

    Args:
        dataset: Dataset to write alpha into. Must have a list-of-dicts
            attribute (``items`` or ``_precomputed``) with length matching
            ``alpha.shape[0]``.
        alpha: Alpha tensor [N, k] aligned with dataset ordering.

    Raises:
        RuntimeError: If the dataset type is not supported (no known
            list-of-dicts attribute found).
    """
    items = getattr(dataset, "items", None) or getattr(dataset, "_precomputed", None)
    if items is not None and len(items) == alpha.shape[0]:
        for i, item in enumerate(items):
            item["alpha"] = alpha[i]
        return

    raise RuntimeError(
        f"Cannot write alpha to dataset of type {type(dataset).__name__}: "
        f"no 'items' or '_precomputed' attribute with {alpha.shape[0]} entries. "
        "Add alpha support for this dataset type."
    )
