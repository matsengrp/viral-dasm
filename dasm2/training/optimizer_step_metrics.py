"""Per-eval-window optimizer-step metrics: gradient norm + clip-fire rate.

Issue #678 §7 added these to make the effect of class-weighted CE visible
in the training-time JSONL: weighting inflates per-position gradient
variance (~1.2× at α=0.5, ~2.4× at α=1.0), so the existing fixed
``gradient_clip_val = 1.0`` will fire more often. The two metrics surfaced
here let the trajectory plot tell apart "weighting is working" from
"weighting is being clip-uniformed".

The accumulator is reset at every eval boundary so the values track the
window each ``eval_end`` event covers. It is a plain dataclass — the
trainer drives ``record_step()`` from ``_clip_and_step`` and
``compute()`` from ``_run_eval_boundary``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class OptimizerStepMetrics:
    """Mean pre-clip grad norm + fraction of steps where clipping fired."""

    sum_pre_clip_norm: float = 0.0
    n_steps: int = 0
    n_clipped: int = 0

    def record_step(
        self,
        pre_clip_norm: float,
        clip_threshold: Optional[float],
    ) -> None:
        """Record one optimizer step.

        ``pre_clip_norm`` is the total grad norm prior to any clipping;
        when ``clip_threshold is None`` the call still increments
        ``n_steps`` and the running norm sum so the mean stays defined,
        but the clip counter is not incremented (clipping cannot have
        fired). Non-finite norms (rare NaN-retry path) are skipped — they
        would otherwise blow up the running sum.
        """
        if not math.isfinite(pre_clip_norm):
            return
        self.sum_pre_clip_norm += pre_clip_norm
        self.n_steps += 1
        if clip_threshold is not None and pre_clip_norm > clip_threshold:
            self.n_clipped += 1

    def compute(self) -> Dict[str, float]:
        """Return the per-window summary; defaults preserve schema on empty windows."""
        if self.n_steps == 0:
            return {"grad_norm_pre_clip_mean": 0.0, "clip_fraction": 0.0}
        return {
            "grad_norm_pre_clip_mean": self.sum_pre_clip_norm / self.n_steps,
            "clip_fraction": self.n_clipped / self.n_steps,
        }

    def reset(self) -> None:
        self.sum_pre_clip_norm = 0.0
        self.n_steps = 0
        self.n_clipped = 0
