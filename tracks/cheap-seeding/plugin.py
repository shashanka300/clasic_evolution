"""ClassicAlgoPlugin implementations for cheap seeding and split-as-you-go cluster growth."""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import torch

from core.emeans import _accumulate_counts_and_sums, assign_nearest, eval_kmeans_mse
from core.plugin import ClassicAlgoPlugin, EMARunResult, run_ema_engine
from core.revival import RevivalEvent, apply_donor_splits, revive_dead_units
from core.schedule import CosineSchedule, StepSchedule
from core.scaling import emeans_alpha0, emeans_batch_size
from core.stats import BatchStats, EMAStats
from accounting import BudgetPlan


class SeededEMeansPlugin(ClassicAlgoPlugin):
    """e-means plugin initialized with explicit, pre-seeded centroids."""

    def __init__(
        self,
        initial_centroids: np.ndarray,
        tau: float = 0.1,
        delta: float = 1.0 / 128.0,
        device: str = "auto",
        seed: int = 0,
    ) -> None:
        self.k = int(initial_centroids.shape[0])
        self.d = int(initial_centroids.shape[1])
        self.initial_centroids = np.ascontiguousarray(initial_centroids, dtype=np.float32)
        self.tau = float(tau)
        self.delta = float(delta)
        self.device = device
        self.seed = int(seed)
        self._centroids: np.ndarray | None = None

    def init(self, x0: np.ndarray) -> EMAStats:
        self._centroids = self.initial_centroids.copy()
        return EMAStats.from_params(self._centroids)

    def assign(self, batch: np.ndarray) -> BatchStats:
        assert self._centroids is not None
        labels, _ = assign_nearest(batch, self._centroids, device=self.device)
        counts, sums = _accumulate_counts_and_sums(
            np.ascontiguousarray(batch, dtype=np.float32),
            labels,
            self.k,
        )
        return BatchStats(counts=counts, sums=sums)

    def params_from_stats(self, stats: EMAStats) -> np.ndarray:
        self._centroids = stats.means(out_dtype=np.float32)
        return self._centroids

    def set_params(self, params: np.ndarray) -> None:
        self._centroids = np.asarray(params, dtype=np.float32)

    def revive(self, stats: EMAStats, step: int, rng: np.random.Generator | None = None) -> RevivalEvent:
        return revive_dead_units(stats, tau=self.tau, delta=self.delta, rng=rng)

    def loss(self, x: np.ndarray) -> float:
        assert self._centroids is not None
        return eval_kmeans_mse(x, self._centroids, device=self.device)


class SplitAsYouGoPlugin(ClassicAlgoPlugin):
    """Dynamic cluster growth plugin expanding clusters during epoch 1 via donor splitting.

    Starts with k0 << k centroids (s = 0.0 upfront distance evaluations).
    Dynamically splits highest-weight donor clusters during early mini-batches until
    the full set of k clusters is reached.
    """

    def __init__(
        self,
        k: int,
        k0: int | None = None,
        growth_steps: int = 20,
        tau: float = 0.1,
        delta: float = 1.0 / 128.0,
        device: str = "auto",
        seed: int = 0,
    ) -> None:
        self.k = int(k)
        self.k0 = int(k0) if k0 is not None else max(2, min(self.k, 4))
        self.k_active = self.k0
        self.growth_steps = max(int(growth_steps), self.k - self.k0)
        self.tau = float(tau)
        self.delta = float(delta)
        self.device = device
        self.seed = int(seed)
        self._centroids: np.ndarray | None = None
        self._step = 0
        self.dist_evals_during_training = 0

    def init(self, x0: np.ndarray) -> EMAStats:
        rng = np.random.default_rng(self.seed)
        n, d = x0.shape
        self.k_active = min(self.k0, n)
        idx0 = rng.choice(n, size=self.k_active, replace=False)

        # Pre-allocate k centroid slots in float32
        c_full = np.zeros((self.k, d), dtype=np.float32)
        c_full[:self.k_active] = x0[idx0]
        # Inactive slots duplicate random active points
        for i in range(self.k_active, self.k):
            c_full[i] = c_full[i % self.k_active]

        self._centroids = c_full
        stats = EMAStats.from_params(c_full)
        # Zero out counts of inactive slots so they are dormant
        stats.counts[self.k_active:] = 0.0
        stats.sums[self.k_active:] = 0.0
        self._step = 0
        self.dist_evals_during_training = 0
        return stats

    def assign(self, batch: np.ndarray) -> BatchStats:
        assert self._centroids is not None
        b_len = batch.shape[0]

        # Points only compete among active centroids
        active_centroids = np.ascontiguousarray(self._centroids[:self.k_active], dtype=np.float32)
        labels, _ = assign_nearest(batch, active_centroids, device=self.device)
        self.dist_evals_during_training += b_len * self.k_active

        counts_active, sums_active = _accumulate_counts_and_sums(
            np.ascontiguousarray(batch, dtype=np.float32),
            labels,
            self.k_active,
        )

        # Place into full k-size stats
        full_counts = np.zeros(self.k, dtype=np.float64)
        full_sums = np.zeros((self.k, batch.shape[1]), dtype=np.float64)
        full_counts[:self.k_active] = counts_active
        full_sums[:self.k_active] = sums_active

        return BatchStats(counts=full_counts, sums=full_sums)

    def params_from_stats(self, stats: EMAStats) -> np.ndarray:
        self._centroids = stats.means(out_dtype=np.float32)
        return self._centroids

    def set_params(self, params: np.ndarray) -> None:
        self._centroids = np.asarray(params, dtype=np.float32)

    def revive(self, stats: EMAStats, step: int, rng: np.random.Generator | None = None) -> RevivalEvent:
        self._step = step

        # Check if dynamic cluster growth is still in progress
        if self.k_active < self.k:
            # Check if this step should trigger a split
            # Distribute remaining splits across remaining growth steps
            needed_splits = self.k - self.k_active
            split_interval = max(1, self.growth_steps // max(self.k - self.k0, 1))

            if (step + 1) % split_interval == 0 or step >= self.growth_steps - needed_splits:
                # Find donor with highest count
                donor_idx = int(np.argmax(stats.counts[:self.k_active]))
                dead_slot = self.k_active
                apply_donor_splits(
                    stats,
                    dead_indices=np.array([dead_slot], dtype=np.int64),
                    donor_indices=np.array([donor_idx], dtype=np.int64),
                    delta=self.delta,
                )
                self.k_active += 1
                self.params_from_stats(stats)
                return RevivalEvent(
                    dead_indices=np.array([dead_slot], dtype=np.int64),
                    donor_indices=np.array([donor_idx], dtype=np.int64),
                    delta=self.delta,
                )
            empty = np.empty(0, dtype=np.int64)
            return RevivalEvent(dead_indices=empty, donor_indices=empty, delta=self.delta)

        # Once all k clusters are active, run standard dead unit revival
        return revive_dead_units(stats, tau=self.tau, delta=self.delta, rng=rng)

    def loss(self, x: np.ndarray) -> float:
        assert self._centroids is not None
        # Evaluate using all active centroids (or full k if complete)
        return eval_kmeans_mse(x, self._centroids[:self.k_active], device=self.device)


def fit_budgeted_seeded_emeans(
    x: np.ndarray,
    initial_centroids: np.ndarray,
    budget_plan: BudgetPlan,
    alpha0: float | None = None,
    tau: float = 0.1,
    delta: float = 1.0 / 128.0,
    revival_every: int = 5,
    finalize_mode: str = "last_epoch",
    slow_half_life: float = 50.0,
    device: str = "auto",
    seed: int = 0,
) -> tuple[np.ndarray, int, int]:
    """Fit e-means initialized with pre-seeded centroids under strict budget allocation.

    Parameters
    ----------
    x : np.ndarray, shape (n, d)
        Input training dataset.
    initial_centroids : np.ndarray, shape (k, d)
        Initial centroids from seeding phase.
    budget_plan : BudgetPlan
        Budget allocation plan specifying clustering_steps.
    alpha0 : float, optional
        Base learning rate. Defaults to emeans_alpha0 scaling rule.
    tau, delta, revival_every, finalize_mode, slow_half_life, device, seed:
        Standard e-means hyperparameters.

    Returns
    -------
    final_centroids : np.ndarray, shape (k, d)
    steps_run : int
    revivals_count : int
    """
    n, d = x.shape
    k = initial_centroids.shape[0]

    # If no clustering steps are allocated, return initial centroids directly
    if budget_plan.clustering_steps <= 0:
        return np.ascontiguousarray(initial_centroids, dtype=np.float32), 0, 0

    b = budget_plan.batch_size
    a0 = float(alpha0) if alpha0 is not None else emeans_alpha0(n, k, budget_plan.total_budget_epochs, b)

    plugin = SeededEMeansPlugin(
        initial_centroids=initial_centroids,
        tau=tau,
        delta=delta,
        device=device,
        seed=seed,
    )

    sched = CosineSchedule(alpha0=a0, total_steps=budget_plan.clustering_steps, eps=0.1)

    effective_epochs = float(budget_plan.clustering_steps) / float(budget_plan.steps_per_epoch)

    res: EMARunResult = run_ema_engine(
        plugin=plugin,
        x=x,
        epochs=effective_epochs,
        batch_size=b,
        schedule=sched,
        revival_every=revival_every,
        finalize_mode=finalize_mode,
        slow_half_life=slow_half_life,
        seed=seed,
    )

    return res.params, res.steps, res.revivals


def fit_budgeted_split_as_you_go(
    x: np.ndarray,
    k: int,
    budget_plan: BudgetPlan,
    k0: int | None = None,
    alpha0: float | None = None,
    tau: float = 0.1,
    delta: float = 1.0 / 128.0,
    revival_every: int = 5,
    finalize_mode: str = "last_epoch",
    slow_half_life: float = 50.0,
    device: str = "auto",
    seed: int = 0,
) -> tuple[np.ndarray, int, int, int]:
    """Fit e-means using Split-as-you-go cluster growth under strict budget allocation.

    Returns
    -------
    final_centroids : np.ndarray, shape (k, d)
    steps_run : int
    revivals_count : int
    training_dist_evals : int
    """
    n, d = x.shape
    if budget_plan.clustering_steps <= 0:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=k, replace=k > n)
        return np.ascontiguousarray(x[idx], dtype=np.float32), 0, 0, 0

    b = budget_plan.batch_size
    a0 = float(alpha0) if alpha0 is not None else emeans_alpha0(n, k, budget_plan.total_budget_epochs, b)

    growth_steps = max(k - (k0 or 4), int(round(0.1 * budget_plan.clustering_steps)))

    plugin = SplitAsYouGoPlugin(
        k=k,
        k0=k0,
        growth_steps=growth_steps,
        tau=tau,
        delta=delta,
        device=device,
        seed=seed,
    )

    sched = CosineSchedule(alpha0=a0, total_steps=budget_plan.clustering_steps, eps=0.1)

    effective_epochs = float(budget_plan.clustering_steps) / float(budget_plan.steps_per_epoch)

    res: EMARunResult = run_ema_engine(
        plugin=plugin,
        x=x,
        epochs=effective_epochs,
        batch_size=b,
        schedule=sched,
        revival_every=revival_every,
        finalize_mode=finalize_mode,
        slow_half_life=slow_half_life,
        seed=seed,
    )

    return res.params, res.steps, res.revivals, plugin.dist_evals_during_training
