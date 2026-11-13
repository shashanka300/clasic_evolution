"""Online KMeans with Drift Detection and Warm-Restart Cosine Schedules (Track T08)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from core.schedule import WarmRestartCosineSchedule, ConstantSchedule, CosineSchedule, StepSchedule
from core.stats import EMAStats
from core.revival import revive_dead_units
try:
    from .detector import PageHinkley, CUSUM
except ImportError:
    from detector import PageHinkley, CUSUM


def assign_and_loss(batch: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Assign points to nearest centroid and return labels, distances, and MSE."""
    # (n, d), (k, d)
    dists = np.sum((batch[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    labels = np.argmin(dists, axis=1)
    min_dists = dists[np.arange(len(batch)), labels]
    mse = float(np.mean(min_dists))
    return labels, min_dists, mse


@dataclass
class DriftRewarmKMeans:
    """e-means clustering with online Page-Hinkley drift detection and warm restart.

    When concept drift causes the batch MSE to spike, the Page-Hinkley test fires,
    triggering a warm restart that raises the step size back to alpha0 and anneals
    over a restart period.
    """

    k: int
    dim: int
    alpha0: float = 0.5
    restart_period: int = 30
    min_eps: float = 0.05
    revival_interval: int = 5
    revival_threshold: float = 0.1
    ph_delta: float = 0.02
    ph_threshold: float = 5.0
    seed: int = 42

    # Internal state
    centroids: np.ndarray = field(init=False)
    counts: np.ndarray = field(init=False)
    sums: np.ndarray = field(init=False)
    schedule: WarmRestartCosineSchedule = field(init=False)
    detector: PageHinkley = field(init=False)
    step: int = 0
    alarms: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.centroids = rng.normal(0.0, 1.0, size=(self.k, self.dim)).astype(np.float32)
        self.counts = np.ones(self.k, dtype=np.float32)
        self.sums = self.centroids.copy()
        self.schedule = WarmRestartCosineSchedule(
            alpha0=self.alpha0,
            period=self.restart_period,
            eps=self.min_eps,
        )
        self.detector = PageHinkley(delta=self.ph_delta, threshold=self.ph_threshold)
        self.step = 0
        self.alarms = []

    def update_batch(self, batch: np.ndarray) -> float:
        """Process one streaming batch, detect drift, and update centroids."""
        labels, _, batch_mse = assign_and_loss(batch, self.centroids)

        # Check for drift
        if self.step > 5 and self.detector.update(batch_mse):
            self.schedule.trigger_restart(self.step, new_period=self.restart_period)
            self.alarms.append(self.step)

        alpha = self.schedule(self.step)

        # Batch sufficient statistics
        batch_counts = np.bincount(labels, minlength=self.k).astype(np.float32)
        batch_sums = np.zeros((self.k, self.dim), dtype=np.float32)
        for i in range(self.k):
            mask = labels == i
            if np.any(mask):
                batch_sums[i] = np.sum(batch[mask], axis=0)

        # EMA updates of sufficient statistics
        self.counts = (1.0 - alpha) * self.counts + alpha * batch_counts
        self.sums = (1.0 - alpha) * self.sums + alpha * batch_sums
        self.centroids = self.sums / np.maximum(self.counts[:, None], 1e-8)

        # Revival of dead / starved units
        if self.step % self.revival_interval == 0:
            stats = EMAStats(counts=self.counts, sums=self.sums)
            revive_dead_units(
                stats=stats,
                tau=self.revival_threshold,
                rng=np.random.default_rng(self.seed + self.step),
            )
            self.counts = stats.counts
            self.sums = stats.sums
            self.centroids = self.sums / np.maximum(self.counts[:, None], 1e-8)

        self.step += 1
        return batch_mse

    def evaluate(self, test_data: np.ndarray) -> float:
        """Compute MSE loss on test dataset."""
        _, _, mse = assign_and_loss(test_data, self.centroids)
        return mse


@dataclass
class FrozenEMeans:
    """Standard e-means that anneals over initial_period and freezes at eps * alpha0."""

    k: int
    dim: int
    alpha0: float = 0.5
    total_steps: int = 60
    min_eps: float = 0.05
    seed: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.centroids = rng.normal(0.0, 1.0, size=(self.k, self.dim)).astype(np.float32)
        self.counts = np.ones(self.k, dtype=np.float32)
        self.sums = self.centroids.copy()
        self.schedule = CosineSchedule(alpha0=self.alpha0, total_steps=self.total_steps, eps=self.min_eps)
        self.step = 0

    def update_batch(self, batch: np.ndarray) -> float:
        labels, _, batch_mse = assign_and_loss(batch, self.centroids)
        alpha = self.schedule(self.step)

        batch_counts = np.bincount(labels, minlength=self.k).astype(np.float32)
        batch_sums = np.zeros((self.k, self.dim), dtype=np.float32)
        for i in range(self.k):
            mask = labels == i
            if np.any(mask):
                batch_sums[i] = np.sum(batch[mask], axis=0)

        self.counts = (1.0 - alpha) * self.counts + alpha * batch_counts
        self.sums = (1.0 - alpha) * self.sums + alpha * batch_sums
        self.centroids = self.sums / np.maximum(self.counts[:, None], 1e-8)
        self.step += 1
        return batch_mse

    def evaluate(self, test_data: np.ndarray) -> float:
        _, _, mse = assign_and_loss(test_data, self.centroids)
        return mse


@dataclass
class ConstantEMAKMeans:
    """Constant step-size EMA k-means (VQ-VAE style)."""

    k: int
    dim: int
    alpha: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.centroids = rng.normal(0.0, 1.0, size=(self.k, self.dim)).astype(np.float32)
        self.counts = np.ones(self.k, dtype=np.float32)
        self.sums = self.centroids.copy()
        self.step = 0

    def update_batch(self, batch: np.ndarray) -> float:
        labels, _, batch_mse = assign_and_loss(batch, self.centroids)
        alpha = self.alpha

        batch_counts = np.bincount(labels, minlength=self.k).astype(np.float32)
        batch_sums = np.zeros((self.k, self.dim), dtype=np.float32)
        for i in range(self.k):
            mask = labels == i
            if np.any(mask):
                batch_sums[i] = np.sum(batch[mask], axis=0)

        self.counts = (1.0 - alpha) * self.counts + alpha * batch_counts
        self.sums = (1.0 - alpha) * self.sums + alpha * batch_sums
        self.centroids = self.sums / np.maximum(self.counts[:, None], 1e-8)
        self.step += 1
        return batch_mse

    def evaluate(self, test_data: np.ndarray) -> float:
        _, _, mse = assign_and_loss(test_data, self.centroids)
        return mse
