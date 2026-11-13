"""ClassicAlgoPlugin integration for Online Dictionary Learning and NMF."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any
import numpy as np

_track_dir = Path(__file__).resolve().parent
if str(_track_dir) not in sys.path:
    sys.path.insert(0, str(_track_dir))

from core.plugin import ClassicAlgoPlugin, EMARunResult, run_ema_engine
from core.revival import RevivalEvent
from core.schedule import CosineSchedule, StepSchedule, make_schedule
from core.scaling import emeans_alpha0, emeans_batch_size
from core.stats import BatchStats, EMAStats
from dictlearn import (
    block_coordinate_descent_update,
    fista_sparse_code,
    revive_dead_atoms,
)


class DictionaryLearningPlugin(ClassicAlgoPlugin):
    """e-means plugin integrating online dictionary learning with core.plugin.run_ema_engine."""

    def __init__(
        self,
        n_components: int = 64,
        alpha_reg: float = 0.1,
        non_negative: bool = False,
        tau: float = 0.1,
        delta: float = 1.0 / 128.0,
        fista_max_iter: int = 50,
        fista_tol: float = 1e-4,
        seed: int = 0,
    ) -> None:
        self.n_components = int(n_components)
        self.alpha_reg = float(alpha_reg)
        self.non_negative = bool(non_negative)
        self.tau = float(tau)
        self.delta = float(delta)
        self.fista_max_iter = int(fista_max_iter)
        self.fista_tol = float(fista_tol)
        self.seed = int(seed)

        self.D: np.ndarray | None = None
        self.d: int | None = None
        self._last_batch: np.ndarray | None = None
        self._last_alpha: np.ndarray | None = None

    def init(self, x0: np.ndarray) -> EMAStats:
        """Initialize dictionary atoms and EMA sufficient statistics from x0."""
        x_arr = np.asarray(x0, dtype=np.float64)
        if x_arr.ndim != 2:
            raise ValueError(f"x0 must be 2D, got shape {x_arr.shape}")

        n_samples, d_dim = x_arr.shape
        self.d = d_dim
        k = self.n_components

        rng = np.random.default_rng(self.seed)

        # Initialize dictionary D of shape (K, d)
        if n_samples >= k:
            sub_indices = rng.choice(n_samples, size=k, replace=False)
            d_init = x_arr[sub_indices].copy()
        else:
            d_init = rng.standard_normal((k, d_dim))

        if self.non_negative:
            d_init = np.abs(d_init)

        # Normalize atoms to unit L2 norm
        norms = np.linalg.norm(d_init, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        d_init /= norms

        self.D = d_init.astype(np.float32)

        # Pseudo-counts and equilibrium initial statistics
        n0 = 1e-10
        counts = np.ones(k, dtype=np.float64) * n0
        sums = (d_init * n0).astype(np.float64)
        extras = {"A": (np.eye(k, dtype=np.float64) * n0)}

        self._last_batch = None
        self._last_alpha = None

        return EMAStats(counts=counts, sums=sums, extras=extras)

    def assign(self, batch: np.ndarray) -> BatchStats:
        """Compute sparse codes for batch and emit sufficient statistics A_batch and B_batch."""
        b_arr = np.asarray(batch, dtype=np.float64)
        if b_arr.ndim != 2:
            raise ValueError(f"Batch must be 2D, got shape {b_arr.shape}")

        if self.d is not None and b_arr.shape[1] != self.d:
            raise ValueError(f"Input dimension d={b_arr.shape[1]} does not match dictionary d={self.d}")

        if self.non_negative and np.any(b_arr < 0):
            raise ValueError("NMF requires non-negative inputs (found negative entries in batch)")

        if self.D is None:
            raise RuntimeError("Plugin not initialized. Call init() first.")

        b_size = max(b_arr.shape[0], 1)

        # Solve sparse coding via FISTA
        alpha = fista_sparse_code(
            x=b_arr,
            d=self.D,
            alpha_reg=self.alpha_reg,
            max_iter=self.fista_max_iter,
            tol=self.fista_tol,
            non_negative=self.non_negative,
        )

        self._last_batch = b_arr
        self._last_alpha = alpha

        # Sufficient statistics:
        # A_batch = (1/b) * alpha^T @ alpha
        # B_batch = (1/b) * alpha^T @ batch
        # counts_batch = (1/b) * sum |alpha|
        alpha_f64 = alpha.astype(np.float64)
        a_batch = (alpha_f64.T @ alpha_f64) / float(b_size)
        b_batch = (alpha_f64.T @ b_arr) / float(b_size)
        c_batch = np.sum(np.abs(alpha_f64), axis=0) / float(b_size)

        return BatchStats(
            counts=c_batch.astype(np.float64),
            sums=b_batch.astype(np.float64),
            extras={"A": a_batch.astype(np.float64)},
        )

    def params_from_stats(self, stats: EMAStats) -> np.ndarray:
        """Update dictionary D via Block Coordinate Descent from EMA sufficient statistics."""
        if self.D is None:
            raise RuntimeError("Dictionary not initialized")

        a_mat = stats.extras["A"]
        b_mat = stats.sums

        updated_d = block_coordinate_descent_update(
            d_init=self.D,
            a=a_mat,
            b=b_mat,
            non_negative=self.non_negative,
            n_iter=1,
        )
        self.D = updated_d.astype(np.float32)
        return self.D

    def set_params(self, params: np.ndarray) -> None:
        """Set active dictionary parameters."""
        self.D = np.asarray(params, dtype=np.float32)

    def revive(self, stats: EMAStats, step: int, rng: np.random.Generator | None = None) -> RevivalEvent:
        """Revive inactive dictionary atoms and reset sufficient statistics."""
        if self.D is None:
            empty = np.empty(0, dtype=np.int64)
            return RevivalEvent(dead_indices=empty, donor_indices=empty, delta=self.delta)

        counts = stats.counts
        mean_n = float(np.mean(counts))
        if mean_n <= 0.0:
            empty = np.empty(0, dtype=np.int64)
            return RevivalEvent(dead_indices=empty, donor_indices=empty, delta=self.delta)

        thresh = self.tau * mean_n
        dead_indices = np.flatnonzero(counts < thresh)
        if len(dead_indices) == 0:
            empty = np.empty(0, dtype=np.int64)
            return RevivalEvent(dead_indices=empty, donor_indices=empty, delta=self.delta)

        # Compute residuals from last batch if available
        residuals = None
        if self._last_batch is not None and self._last_alpha is not None and len(self._last_batch) > 0:
            residuals = self._last_batch - self._last_alpha.astype(np.float64) @ self.D.astype(np.float64)

        # Select live donors for fallback
        live_indices = np.flatnonzero(counts >= thresh)
        if len(live_indices) == 0:
            live_indices = np.flatnonzero(counts > 0)
        if len(live_indices) == 0:
            empty = np.empty(0, dtype=np.int64)
            return RevivalEvent(dead_indices=empty, donor_indices=empty, delta=self.delta)

        rng = rng if rng is not None else np.random.default_rng(self.seed + step)
        donor_indices = rng.choice(live_indices, size=len(dead_indices), replace=True)

        revive_dead_atoms(
            d=self.D,
            stats_a=stats.extras["A"],
            stats_b=stats.sums,
            counts=stats.counts,
            dead_indices=dead_indices,
            donor_indices=donor_indices,
            delta=self.delta,
            non_negative=self.non_negative,
            residuals=residuals,
        )

        return RevivalEvent(
            dead_indices=dead_indices,
            donor_indices=donor_indices,
            delta=self.delta,
        )

    def loss(self, x: np.ndarray) -> float:
        """Evaluate penalized reconstruction loss on dataset x."""
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {x_arr.shape}")

        if self.d is not None and x_arr.shape[1] != self.d:
            raise ValueError(f"Input dimension d={x_arr.shape[1]} does not match dictionary d={self.d}")

        if self.D is None:
            raise RuntimeError("Dictionary not initialized")

        n = x_arr.shape[0]
        if n == 0:
            return 0.0

        alpha = fista_sparse_code(
            x=x_arr,
            d=self.D,
            alpha_reg=self.alpha_reg,
            max_iter=self.fista_max_iter,
            tol=self.fista_tol,
            non_negative=self.non_negative,
        )
        recon_err = 0.5 * float(np.sum((x_arr - alpha.astype(np.float64) @ self.D.astype(np.float64)) ** 2)) / float(n)
        sparsity = self.alpha_reg * float(np.sum(np.abs(alpha))) / float(n)
        return float(recon_err + sparsity)


class NMFPlugin(DictionaryLearningPlugin):
    """Plugin specialized for Non-negative Matrix Factorization."""

    def __init__(
        self,
        n_components: int = 64,
        alpha_reg: float = 0.0,
        tau: float = 0.1,
        delta: float = 1.0 / 128.0,
        fista_max_iter: int = 50,
        fista_tol: float = 1e-4,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("non_negative", None)
        super().__init__(
            n_components=n_components,
            alpha_reg=alpha_reg,
            non_negative=True,
            tau=tau,
            delta=delta,
            fista_max_iter=fista_max_iter,
            fista_tol=fista_tol,
            seed=seed,
        )


def fit_dictlearn_emeans(
    x: np.ndarray,
    n_components: int = 64,
    alpha_reg: float = 0.1,
    epochs: float = 5.0,
    batch_size: int = 128,
    alpha0: float | None = None,
    schedule: str = "cosine",
    tau: float = 0.1,
    delta: float = 1.0 / 128.0,
    revival_every: int = 5,
    seed: int = 0,
    **kwargs: Any,
) -> EMARunResult:
    """Fit online dictionary learning using e-means EMA engine."""
    x_arr = np.asarray(x, dtype=np.float32)
    n, d = x_arr.shape

    plugin = DictionaryLearningPlugin(
        n_components=n_components,
        alpha_reg=alpha_reg,
        non_negative=False,
        tau=tau,
        delta=delta,
        seed=seed,
    )

    b = min(batch_size, n)
    steps_per_epoch = max(1, math.ceil(n / b))
    total_steps = max(1, int(math.ceil(float(epochs) * steps_per_epoch)))

    if alpha0 is None:
        alpha0_val = emeans_alpha0(n=n, k=n_components, epochs=int(math.ceil(epochs)), batch_size=b)
    else:
        alpha0_val = float(alpha0)

    sched = make_schedule(schedule, alpha0=alpha0_val, total_steps=total_steps)

    res = run_ema_engine(
        plugin=plugin,
        x=x_arr,
        epochs=epochs,
        batch_size=b,
        schedule=sched,
        revival_every=revival_every,
        finalize_mode=kwargs.get("finalize_mode", None),
        seed=seed,
    )
    return res


def fit_nmf_emeans(
    x: np.ndarray,
    n_components: int = 64,
    alpha_reg: float = 0.0,
    epochs: float = 5.0,
    batch_size: int = 128,
    alpha0: float | None = None,
    schedule: str = "cosine",
    tau: float = 0.1,
    delta: float = 1.0 / 128.0,
    revival_every: int = 5,
    seed: int = 0,
    **kwargs: Any,
) -> EMARunResult:
    """Fit online NMF using e-means EMA engine."""
    x_arr = np.asarray(x, dtype=np.float32)
    n, d = x_arr.shape

    plugin = NMFPlugin(
        n_components=n_components,
        alpha_reg=alpha_reg,
        tau=tau,
        delta=delta,
        seed=seed,
    )

    b = min(batch_size, n)
    steps_per_epoch = max(1, math.ceil(n / b))
    total_steps = max(1, int(math.ceil(float(epochs) * steps_per_epoch)))

    if alpha0 is None:
        alpha0_val = emeans_alpha0(n=n, k=n_components, epochs=int(math.ceil(epochs)), batch_size=b)
    else:
        alpha0_val = float(alpha0)

    sched = make_schedule(schedule, alpha0=alpha0_val, total_steps=total_steps)

    res = run_ema_engine(
        plugin=plugin,
        x=x_arr,
        epochs=epochs,
        batch_size=b,
        schedule=sched,
        revival_every=revival_every,
        finalize_mode=kwargs.get("finalize_mode", None),
        seed=seed,
    )
    return res
