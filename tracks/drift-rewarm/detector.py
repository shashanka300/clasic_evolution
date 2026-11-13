"""Drift detectors and streaming benchmark generators for Track T08."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Tuple
import numpy as np


class PageHinkley:
    """Page-Hinkley test for concept drift detection in streaming loss signals.

    Tracks whether the cumulative deviation of loss from its running mean
    exceeds an alarm threshold `threshold` with minimum change tolerance `delta`.
    """

    def __init__(self, delta: float = 0.05, threshold: float = 15.0, alpha: float = 0.05) -> None:
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.reset()

    def reset(self) -> None:
        self.running_mean: float = 0.0
        self.cumulative_sum: float = 0.0
        self.min_cumulative_sum: float = 0.0
        self.count: int = 0

    def update(self, value: float) -> bool:
        """Update detector with observed value (e.g. batch MSE). Returns True on alarm."""
        self.count += 1
        if self.count == 1:
            self.running_mean = float(value)
        else:
            self.running_mean = (1.0 - self.alpha) * self.running_mean + self.alpha * float(value)

        # Cumulative difference with tolerance delta
        diff = float(value) - self.running_mean - self.delta
        self.cumulative_sum += diff
        if self.cumulative_sum < self.min_cumulative_sum:
            self.min_cumulative_sum = self.cumulative_sum

        ph_stat = self.cumulative_sum - self.min_cumulative_sum
        if ph_stat > self.threshold:
            self.reset()
            return True
        return False


class CUSUM:
    """Two-sided cumulative sum drift detector."""

    def __init__(self, threshold: float = 10.0, drift_tol: float = 0.05) -> None:
        self.threshold = float(threshold)
        self.drift_tol = float(drift_tol)
        self.reset()

    def reset(self) -> None:
        self.mean: float = 0.0
        self.s_pos: float = 0.0
        self.s_neg: float = 0.0
        self.count: int = 0

    def update(self, value: float) -> bool:
        self.count += 1
        if self.count == 1:
            self.mean = float(value)
            return False
        # Running mean
        self.mean += (value - self.mean) / self.count
        self.s_pos = max(0.0, self.s_pos + (value - self.mean - self.drift_tol))
        self.s_neg = max(0.0, self.s_neg - (value - self.mean + self.drift_tol))
        if self.s_pos > self.threshold or self.s_neg > self.threshold:
            self.reset()
            return True
        return False


@dataclass
class DriftScenario:
    """Configurable synthetic data generator that streams batches under 4 drift regimes."""

    n_batches: int = 200
    batch_size: int = 500
    dim: int = 32
    k: int = 16
    drift_type: str = "abrupt"  # "abrupt", "gradual", "seasonal", "encoder_upgrade"
    seed: int = 42

    def stream(self) -> Generator[Tuple[np.ndarray, np.ndarray, bool], None, None]:
        """Yield (train_batch, test_window, is_drift_point) for each step."""
        rng = np.random.default_rng(self.seed)
        # Initial cluster centers
        centers = rng.normal(0.0, 1.0, size=(self.k, self.dim)).astype(np.float32)

        # Fixed drift points
        drift_point_1 = int(self.n_batches * 0.35)
        drift_point_2 = int(self.n_batches * 0.70)

        for step in range(self.n_batches):
            is_drift = False

            if self.drift_type == "abrupt":
                if step == drift_point_1:
                    # Half of clusters jump to new locations
                    centers[: self.k // 2] += rng.normal(3.0, 0.5, size=(self.k // 2, self.dim))
                    is_drift = True
                elif step == drift_point_2:
                    # Remaining clusters jump
                    centers[self.k // 2 :] += rng.normal(-3.0, 0.5, size=(self.k - self.k // 2, self.dim))
                    is_drift = True

            elif self.drift_type == "gradual":
                # Continuous smooth drift of centers
                if drift_point_1 <= step < drift_point_2:
                    velocity = 0.05
                    centers += velocity * rng.normal(0.0, 0.2, size=(self.k, self.dim)).astype(np.float32)
                    if step == drift_point_1:
                        is_drift = True

            elif self.drift_type == "seasonal":
                # Periodic sinusoidal oscillation of centers
                omega = 2.0 * np.pi / 40.0
                centers += (0.15 * np.sin(omega * step)).astype(np.float32)
                if step % 40 == 0 and step > 0:
                    is_drift = True

            elif self.drift_type == "encoder_upgrade":
                if step == drift_point_1:
                    # Random orthogonal rotation of the entire space simulating model update
                    q, _ = np.linalg.qr(rng.normal(size=(self.dim, self.dim)))
                    centers = (centers @ q).astype(np.float32)
                    is_drift = True

            # Generate batch points from current centers with std 0.3
            cluster_assignments = rng.integers(0, self.k, size=self.batch_size)
            noise = rng.normal(0.0, 0.3, size=(self.batch_size, self.dim)).astype(np.float32)
            batch = (centers[cluster_assignments] + noise).astype(np.float32)

            # Test window from ground truth centers
            test_assignments = rng.integers(0, self.k, size=1000)
            test_noise = rng.normal(0.0, 0.3, size=(1000, self.dim)).astype(np.float32)
            test_window = (centers[test_assignments] + test_noise).astype(np.float32)

            yield batch, test_window, is_drift
