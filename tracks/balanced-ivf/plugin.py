"""ClassicAlgoPlugin implementation for Balanced Inverted-File Indexing (Balanced IVF)."""

from __future__ import annotations

from typing import Any
import numpy as np

from core.plugin import ClassicAlgoPlugin, EMARunResult, run_ema_engine
from core.revival import RevivalEvent, revive_dead_units
from core.schedule import StepSchedule
from core.stats import BatchStats, EMAStats
try:
    from .balanced_ivf import (
        BalancedIVFIndex,
        SinkhornBalancedAssigner,
        pairwise_sq_dists,
    )
except ImportError:
    from balanced_ivf import (
        BalancedIVFIndex,
        SinkhornBalancedAssigner,
        pairwise_sq_dists,
    )


class BalancedIVFPlugin(ClassicAlgoPlugin):
    """e-means plugin integrating mini-batch Sinkhorn-Knopp balanced assignment.

    Maintains coarse quantizer centroids in EMAStats, using optimal transport to balance
    mini-batch assignments and eliminate IVF posting list length skew.
    """

    def __init__(
        self,
        d: int,
        nlist: int,
        epsilon: float = 0.05,
        num_iters: int = 3,
        slack: float = 0.0,
        tau: float = 0.1,
        delta: float = 1.0 / 128.0,
        seed: int = 42,
    ) -> None:
        self.d = int(d)
        self.nlist = int(nlist)
        self.epsilon = float(epsilon)
        self.num_iters = int(num_iters)
        self.slack = float(slack)
        self.tau = float(tau)
        self.delta = float(delta)
        self.seed = int(seed)

        self.assigner = SinkhornBalancedAssigner(
            epsilon=self.epsilon,
            num_iters=self.num_iters,
            slack=self.slack,
        )
        self.current_centroids: np.ndarray = np.zeros((self.nlist, self.d), dtype=np.float32)
        self._step: int = 0

    def init(self, x0: np.ndarray) -> EMAStats:
        """Initialize coarse centroids from subset x0 and return initial EMAStats."""
        rng = np.random.default_rng(self.seed)
        n = x0.shape[0]

        if n >= self.nlist:
            chosen = rng.choice(n, size=self.nlist, replace=False)
            self.current_centroids = np.ascontiguousarray(x0[chosen], dtype=np.float32)
        else:
            chosen = rng.choice(n, size=self.nlist, replace=True)
            self.current_centroids = np.ascontiguousarray(x0[chosen], dtype=np.float32)

        stats = EMAStats.from_params(self.current_centroids)
        self._step = 0
        return stats

    def assign(self, batch: np.ndarray) -> BatchStats:
        """Assign batch vectors via Sinkhorn-Knopp transport and emit BatchStats."""
        b_size = batch.shape[0]
        batch_f32 = np.ascontiguousarray(batch, dtype=np.float32)

        labels, _, _ = self.assigner.assign(
            batch_f32,
            self.current_centroids,
            step=self._step,
        )
        self._step += 1

        b_counts = np.bincount(labels, minlength=self.nlist).astype(np.float64)
        b_sums = np.zeros((self.nlist, self.d), dtype=np.float64)
        np.add.at(b_sums, labels, batch_f32.astype(np.float64))

        return BatchStats(counts=b_counts, sums=b_sums)

    def params_from_stats(self, stats: EMAStats) -> np.ndarray:
        """Extract updated centroids from EMAStats and store locally."""
        self.current_centroids = stats.means(out_dtype=np.float32)
        return self.current_centroids

    def revive(
        self,
        stats: EMAStats,
        step: int,
        rng: np.random.Generator | None = None,
    ) -> RevivalEvent:
        """Revive dead clusters with Faiss-style donor splitting."""
        event = revive_dead_units(stats, tau=self.tau, delta=self.delta, rng=rng)
        self.current_centroids = stats.means(out_dtype=np.float32)
        return event

    def loss(self, x: np.ndarray) -> float:
        """Compute mean squared Euclidean quantization distortion."""
        x_f32 = np.ascontiguousarray(x, dtype=np.float32)
        # Compute in batches if x is large
        total_sq_err = 0.0
        n = x_f32.shape[0]
        batch_sz = 2048

        for start in range(0, n, batch_sz):
            end = min(start + batch_sz, n)
            chunk = x_f32[start:end]
            M = pairwise_sq_dists(chunk, self.current_centroids)
            min_dists = np.min(M, axis=1)
            total_sq_err += float(np.sum(min_dists))

        return float(total_sq_err / max(float(n), 1.0))


def fit_balanced_ivf_plugin(
    plugin: BalancedIVFPlugin,
    x_train: np.ndarray,
    epochs: float = 1.0,
    batch_size: int = 1024,
    schedule: StepSchedule | None = None,
    revival_every: int = 5,
    seed: int = 42,
) -> EMARunResult:
    """Convenience driver to train BalancedIVFPlugin using core.plugin.run_ema_engine."""
    from core.schedule import CosineSchedule

    n = x_train.shape[0]
    b = min(max(int(batch_size), 1), n)
    steps_per_epoch = max(n // b, 1)
    total_steps = max(int(round(epochs * steps_per_epoch)), 1)

    sched = schedule if schedule is not None else CosineSchedule(alpha0=0.1, total_steps=total_steps, eps=0.1)

    return run_ema_engine(
        plugin=plugin,
        x=x_train,
        epochs=epochs,
        batch_size=b,
        schedule=sched,
        revival_every=revival_every,
        seed=seed,
    )
