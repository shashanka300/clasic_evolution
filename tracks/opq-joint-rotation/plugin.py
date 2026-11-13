"""ClassicAlgoPlugin implementation for Optimized Product Quantization (OPQ)."""

from __future__ import annotations

from typing import Any
import numpy as np

from core.plugin import ClassicAlgoPlugin, EMARunResult, run_ema_engine
from core.revival import RevivalEvent, revive_dead_units
from core.schedule import CosineSchedule, StepSchedule
from core.stats import BatchStats, EMAStats
from opq import OPQQuantizer, solve_orthogonal_procrustes


class OPQPlugin(ClassicAlgoPlugin):
    """e-means plugin integrating joint online OPQ with core.plugin engine.

    Maintains subspace codebooks in EMAStats and cross-covariance in EMAStats extras.
    """

    def __init__(
        self,
        d: int,
        M: int,
        k_sub: int = 256,
        alpha_r: float = 0.05,
        refresh_m: int = 10,
        tau: float = 0.1,
        delta: float = 1.0 / 128.0,
        seed: int = 42,
    ) -> None:
        self.d = int(d)
        self.M = int(M)
        self.k_sub = int(k_sub)
        self.d_s = self.d // self.M
        self.alpha_r = float(alpha_r)
        self.refresh_m = max(int(refresh_m), 1)
        self.tau = float(tau)
        self.delta = float(delta)
        self.seed = int(seed)

        self.quantizer = OPQQuantizer(
            d=self.d,
            M=self.M,
            k_sub=self.k_sub,
            alpha_r=self.alpha_r,
            refresh_m=self.refresh_m,
            tau=self.tau,
            delta=self.delta,
            seed=self.seed,
        )
        self._step: int = 0

    def init(self, x0: np.ndarray) -> EMAStats:
        """Initialize codebooks and stats from x0."""
        rng = np.random.default_rng(self.seed)
        self.quantizer.init_codebooks(x0, rng=rng)
        init_c_flat = self.quantizer.codebooks.reshape(self.M * self.k_sub, self.d_s)
        init_h = np.eye(self.d, dtype=np.float64) / float(self.d)

        stats = EMAStats.from_params(
            init_c_flat,
            extra_inits={"cross_cov": init_h},
        )
        self._step = 0
        return stats

    def assign(self, batch: np.ndarray) -> BatchStats:
        """Assign batch to current rotated subspace centroids and emit BatchStats."""
        b_size = batch.shape[0]
        batch_f32 = np.ascontiguousarray(batch, dtype=np.float32)

        # Rotate batch
        batch_y = batch_f32 @ self.quantizer.R.T  # (B, d)

        all_counts = np.zeros(self.M * self.k_sub, dtype=np.float64)
        all_sums = np.zeros((self.M * self.k_sub, self.d_s), dtype=np.float64)
        y_hat = np.zeros((b_size, self.d), dtype=np.float32)

        for m in range(self.M):
            sub_y = batch_y[:, m * self.d_s : (m + 1) * self.d_s]
            c_m = self.quantizer.codebooks[m]
            c_norms_sq = np.sum(c_m**2, axis=1)
            dots = sub_y @ c_m.T
            dists = c_norms_sq[None, :] - 2.0 * dots
            labels = np.argmin(dists, axis=1).astype(np.int32)

            offset = m * self.k_sub
            all_counts[offset : offset + self.k_sub] = np.bincount(
                labels, minlength=self.k_sub
            )
            sums_m = np.zeros((self.k_sub, self.d_s), dtype=np.float64)
            np.add.at(sums_m, labels, sub_y.astype(np.float64))
            all_sums[offset : offset + self.k_sub] = sums_m

            y_hat[:, m * self.d_s : (m + 1) * self.d_s] = c_m[labels]

        # Batch cross-covariance
        h_batch = (batch_f32.T @ y_hat).astype(np.float64) / float(b_size)

        return BatchStats(
            counts=all_counts,
            sums=all_sums,
            extras={"cross_cov": h_batch},
        )

    def params_from_stats(self, stats: EMAStats) -> np.ndarray:
        """Extract codebooks and update orthogonal rotation R from stats."""
        c_flat = stats.means(out_dtype=np.float32)
        self.quantizer.codebooks = c_flat.reshape(self.M, self.k_sub, self.d_s)

        if "cross_cov" in stats.extras:
            self._step += 1
            if self._step % self.refresh_m == 0:
                h = stats.extras["cross_cov"]
                self.quantizer.R = solve_orthogonal_procrustes(h)

        return c_flat

    def set_params(self, params: np.ndarray) -> None:
        """Set codebook parameters from external array."""
        c_flat = np.asarray(params, dtype=np.float32)
        self.quantizer.codebooks = c_flat.reshape(self.M, self.k_sub, self.d_s)

    def revive(
        self, stats: EMAStats, step: int, rng: np.random.Generator | None = None
    ) -> RevivalEvent:
        """Revive dead codebook units across all subspaces."""
        return revive_dead_units(stats, tau=self.tau, delta=self.delta, rng=rng)

    def loss(self, x: np.ndarray) -> float:
        """Evaluate reconstruction MSE on dataset x."""
        return self.quantizer.reconstruction_mse(x)


def fit_opq_plugin(
    x: np.ndarray,
    d: int,
    M: int,
    k_sub: int = 256,
    epochs: float = 1.0,
    batch_size: int = 1024,
    alpha0: float = 0.1,
    alpha_r: float = 0.05,
    refresh_m: int = 10,
    seed: int = 42,
) -> tuple[OPQPlugin, EMARunResult]:
    """Train OPQ using the standard core.plugin.run_ema_engine runner."""
    n = x.shape[0]
    b = min(max(int(batch_size), 1), n)
    steps_per_epoch = max(n // b, 1)
    total_steps = max(int(round(epochs * steps_per_epoch)), 1)
    schedule = CosineSchedule(alpha0=alpha0, total_steps=total_steps, eps=0.1)

    plugin = OPQPlugin(
        d=d,
        M=M,
        k_sub=k_sub,
        alpha_r=alpha_r,
        refresh_m=refresh_m,
        seed=seed,
    )

    res = run_ema_engine(
        plugin=plugin,
        x=x,
        epochs=epochs,
        batch_size=b,
        schedule=schedule,
        revival_every=5,
        finalize_mode="last_epoch",
        seed=seed,
    )

    # Perform final Procrustes update
    if "cross_cov" in res.stats.extras:
        plugin.quantizer.R = solve_orthogonal_procrustes(res.stats.extras["cross_cov"])

    return plugin, res
