"""Finalization passes: last-epoch-mean (train) and slow-EMA (out-of-sample)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from core.revival import RevivalEvent, apply_donor_splits
from core.stats import BatchStats, EMAStats


@dataclass
class LastEpochMeanFinalizer:
    """Exact unweighted average over assignments observed during the final epoch."""

    k: int
    shape: tuple[int, ...]
    counts: np.ndarray | None = None
    sums: np.ndarray | None = None

    def reset(self) -> None:
        self.counts = np.zeros((self.k,), dtype=np.float64)
        self.sums = np.zeros((self.k, *self.shape), dtype=np.float64)

    def observe(self, batch: BatchStats) -> None:
        if self.counts is None or self.sums is None:
            self.reset()
        self.counts += np.asarray(batch.counts, dtype=np.float64)
        self.sums += np.asarray(batch.sums, dtype=np.float64)

    def finalize(self, fallback_params: np.ndarray, out_dtype: np.dtype | type = np.float32) -> np.ndarray:
        """Return exact last-epoch means, keeping ``fallback_params`` for empty units."""
        fallback = np.asarray(fallback_params, dtype=np.float64)
        if self.counts is None or self.sums is None:
            return fallback.astype(out_dtype, copy=False)
        active = self.counts > 0.0
        out = fallback.copy()
        broadcast = (int(active.sum()),) + (1,) * (self.sums.ndim - 1)
        if np.any(active):
            out[active] = self.sums[active] / self.counts[active].reshape(broadcast)
        return out.astype(out_dtype, copy=False)


class SlowEMAFinalizer:
    """Shadow EMA of sufficient statistics for lower out-of-sample error.

    Controlled by half-life ``half_life`` (in batches): ``alpha_slow = 1 - 2^(-1 / h)``.
    Observes assignments and mirrors donor splits without altering training assignments.
    """

    def __init__(self, init_params: np.ndarray, half_life: float = 50.0, alpha_slow: float | None = None) -> None:
        if alpha_slow is None:
            h = max(float(half_life), 1.0)
            alpha_slow = 1.0 - (2.0 ** (-1.0 / h))
        self.alpha_slow = float(alpha_slow)
        self.stats = EMAStats.from_params(init_params)

    def observe(self, batch: BatchStats) -> None:
        self.stats.update(batch, alpha=self.alpha_slow)

    def mirror_revival(self, event: RevivalEvent) -> None:
        apply_donor_splits(self.stats, event.dead_indices, event.donor_indices, delta=event.delta)

    def finalize(self, out_dtype: np.dtype | type = np.float32) -> np.ndarray:
        return self.stats.means(out_dtype=out_dtype)

