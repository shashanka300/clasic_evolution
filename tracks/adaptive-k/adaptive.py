"""Adaptive-k e-means clustering via EMA-driven BIC split and merge rules (Track T10)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np

from core.schedule import CosineSchedule


def compute_bic(
    counts: np.ndarray,
    sums: np.ndarray,
    sum_sq_errs: np.ndarray,
    dim: int,
) -> float:
    """Compute Bayesian Information Criterion (BIC) for current clustering state.

    BIC = -2 * log_likelihood + num_params * log(N)
    Under a spherical Gaussian mixture model with shared or per-cluster variance.
    """
    valid = counts > 1.0
    k = int(np.sum(valid))
    if k == 0:
        return float("inf")

    n_total = float(np.sum(counts[valid]))
    total_sse = float(np.sum(sum_sq_errs[valid]))
    # Estimated pooled variance
    sigma2 = max(total_sse / max(n_total * dim, 1.0), 1e-8)

    # Log likelihood
    log_lik = -0.5 * n_total * dim * (np.log(2.0 * np.pi * sigma2) + 1.0)
    # Mixture weights log likelihood
    log_lik += float(np.sum(counts[valid] * np.log(np.maximum(counts[valid] / n_total, 1e-8))))

    # Free parameters: (k - 1) weights + k * d coordinates + 1 variance
    num_params = (k - 1) + k * dim + 1
    bic = -2.0 * log_lik + num_params * np.log(max(n_total, 2.0))
    return float(bic)


@dataclass
class AdaptiveKMeans:
    """e-means clustering that dynamically adapts k via BIC split-merge operations."""

    dim: int
    init_k: int = 8
    min_k: int = 2
    max_k: int = 64
    alpha0: float = 0.5
    total_steps: int = 60
    adapt_interval: int = 5
    split_threshold: float = 1.3
    merge_threshold: float = 0.8
    seed: int = 42

    centroids: np.ndarray = field(init=False)
    counts: np.ndarray = field(init=False)
    sums: np.ndarray = field(init=False)
    sum_sq: np.ndarray = field(init=False)  # Sum of squared errors per cluster
    schedule: CosineSchedule = field(init=False)
    step: int = 0
    history_k: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.centroids = rng.normal(0.0, 1.0, size=(self.init_k, self.dim)).astype(np.float32)
        self.counts = np.ones(self.init_k, dtype=np.float32)
        self.sums = self.centroids.copy()
        self.sum_sq = np.ones(self.init_k, dtype=np.float32)
        self.schedule = CosineSchedule(alpha0=self.alpha0, total_steps=self.total_steps, eps=0.05)
        self.step = 0
        self.history_k = [self.init_k]

    @property
    def current_k(self) -> int:
        return len(self.centroids)

    def _assign(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        dists = np.sum((batch[:, None, :] - self.centroids[None, :, :]) ** 2, axis=-1)
        labels = np.argmin(dists, axis=1)
        min_dists = dists[np.arange(len(batch)), labels]
        mse = float(np.mean(min_dists))
        return labels, min_dists, mse

    def update_batch(self, batch: np.ndarray) -> float:
        """Process one batch, update statistics, and perform split/merge if scheduled."""
        labels, min_dists, batch_mse = self._assign(batch)
        alpha = self.schedule(self.step)
        k = self.current_k

        # Compute batch statistics
        batch_counts = np.bincount(labels, minlength=k).astype(np.float32)
        batch_sums = np.zeros((k, self.dim), dtype=np.float32)
        batch_sq = np.zeros(k, dtype=np.float32)

        for i in range(k):
            mask = labels == i
            if np.any(mask):
                batch_sums[i] = np.sum(batch[mask], axis=0)
                batch_sq[i] = float(np.sum(min_dists[mask]))

        # EMA update
        self.counts = (1.0 - alpha) * self.counts + alpha * batch_counts
        self.sums = (1.0 - alpha) * self.sums + alpha * batch_sums
        self.sum_sq = (1.0 - alpha) * self.sum_sq + alpha * batch_sq
        self.centroids = self.sums / np.maximum(self.counts[:, None], 1e-8)

        # Periodic split and merge decisions
        if self.step > 0 and self.step % self.adapt_interval == 0 and self.step < int(self.total_steps * 0.85):
            self._split_and_merge()

        self.step += 1
        self.history_k.append(self.current_k)
        return batch_mse

    def _split_and_merge(self) -> None:
        """Evaluate candidate splits and merges based on cluster variances and distances."""
        # 1. Merge test: check if any pair of centroids are excessively close
        if self.current_k > self.min_k:
            k = self.current_k
            mean_var = float(np.mean(self.sum_sq / np.maximum(self.counts * self.dim, 1e-8)))
            dist_matrix = np.sum((self.centroids[:, None, :] - self.centroids[None, :, :]) ** 2, axis=-1)
            np.fill_diagonal(dist_matrix, np.inf)

            i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            min_dist = dist_matrix[i, j]

            if min_dist < self.merge_threshold * mean_var:
                # Merge cluster j into cluster i
                new_count = self.counts[i] + self.counts[j]
                new_sum = self.sums[i] + self.sums[j]
                new_sq = self.sum_sq[i] + self.sum_sq[j] + (self.counts[i] * self.counts[j] / max(new_count, 1e-8)) * min_dist

                self.counts[i] = new_count
                self.sums[i] = new_sum
                self.sum_sq[i] = new_sq
                self.centroids[i] = new_sum / max(new_count, 1e-8)

                # Delete j
                keep = [idx for idx in range(k) if idx != j]
                self.counts = self.counts[keep]
                self.sums = self.sums[keep]
                self.sum_sq = self.sum_sq[keep]
                self.centroids = self.centroids[keep]
                return

        # 2. Split test: check if high-variance clusters should split
        if self.current_k < self.max_k:
            variances = self.sum_sq / np.maximum(self.counts * self.dim, 1e-8)
            mean_var = float(np.mean(variances))
            candidates = np.flatnonzero((variances > self.split_threshold * mean_var) & (self.counts > 5.0))

            if len(candidates) > 0:
                # Sort by variance descending and split top candidate(s)
                candidates = candidates[np.argsort(-variances[candidates])]
                max_splits = min(len(candidates), max(1, (self.max_k - self.current_k) // 4))

                for heavy_idx in candidates[:max_splits]:
                    if self.current_k >= self.max_k:
                        break
                    c_donor = self.centroids[heavy_idx]
                    n_donor = self.counts[heavy_idx]

                    d = self.dim
                    sign = np.where(np.arange(d) % 2 == 0, 1.0, -1.0).astype(np.float32)
                    delta = float(np.sqrt(mean_var) * 0.5)

                    c1 = c_donor + delta * sign
                    c2 = c_donor - delta * sign
                    half_n = 0.5 * n_donor
                    half_sq = 0.5 * self.sum_sq[heavy_idx]

                    # Update donor in-place
                    self.counts[heavy_idx] = half_n
                    self.centroids[heavy_idx] = c1
                    self.sums[heavy_idx] = half_n * c1
                    self.sum_sq[heavy_idx] = half_sq

                    # Append new split unit
                    self.counts = np.append(self.counts, half_n)
                    self.centroids = np.vstack([self.centroids, c2[None, :]])
                    self.sums = np.vstack([self.sums, (half_n * c2)[None, :]])
                    self.sum_sq = np.append(self.sum_sq, half_sq)

    def evaluate(self, test_data: np.ndarray) -> float:
        """Compute MSE loss on test dataset."""
        _, _, mse = self._assign(test_data)
        return mse
