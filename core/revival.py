"""Dead-unit detection and Faiss-style donor-split revival."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from core.stats import EMAStats


@dataclass(frozen=True)
class RevivalEvent:
    """Record of revived dead slots and their donor units."""

    dead_indices: np.ndarray
    donor_indices: np.ndarray
    delta: float


def apply_donor_splits(stats: EMAStats, dead_indices: np.ndarray, donor_indices: np.ndarray, delta: float) -> None:
    """Split each donor in ``stats`` into ``(donor, dead)`` child pairs in-place."""
    if len(dead_indices) == 0:
        return
    d = stats.sums.shape[1] if stats.sums.ndim >= 2 else 1
    sign = np.where(np.arange(d) % 2 == 0, 1.0, -1.0)
    if stats.sums.ndim > 2:
        sign = sign.reshape((d,) + (1,) * (stats.sums.ndim - 2))

    for dead_i, donor_i in zip(dead_indices, donor_indices):
        n_donor = float(stats.counts[donor_i])
        if n_donor <= 0.0:
            continue
        c_donor = stats.sums[donor_i] / n_donor
        half_n = 0.5 * n_donor
        stats.counts[dead_i] = half_n
        stats.counts[donor_i] = half_n
        stats.sums[dead_i] = half_n * c_donor * (1.0 + delta * sign)
        stats.sums[donor_i] = half_n * c_donor * (1.0 - delta * sign)
        for name, arr in stats.extras.items():
            if arr.shape[0] == stats.counts.shape[0]:
                half_extra = 0.5 * arr[donor_i]
                arr[dead_i] = half_extra
                arr[donor_i] = half_extra


def revive_dead_units(
    stats: EMAStats,
    tau: float = 0.1,
    delta: float = 1.0 / 128.0,
    rng: np.random.Generator | None = None,
) -> RevivalEvent:
    """Detect units with ``N_j < tau * mean(N)`` and split weighted live donors."""
    counts = stats.counts
    mean_n = float(counts.mean())
    if mean_n <= 0.0:
        empty = np.empty(0, dtype=np.int64)
        return RevivalEvent(dead_indices=empty, donor_indices=empty, delta=delta)

    thresh = tau * mean_n
    dead_idx = np.flatnonzero(counts < thresh)
    if len(dead_idx) == 0:
        empty = np.empty(0, dtype=np.int64)
        return RevivalEvent(dead_indices=empty, donor_indices=empty, delta=delta)

    live_idx = np.flatnonzero(counts >= thresh)
    if len(live_idx) == 0:
        live_idx = np.flatnonzero(counts > 0.0)
    if len(live_idx) == 0:
        empty = np.empty(0, dtype=np.int64)
        return RevivalEvent(dead_indices=empty, donor_indices=empty, delta=delta)

    rng = rng if rng is not None else np.random.default_rng()
    weights = counts[live_idx]
    w_sum = float(weights.sum())
    probs = weights / w_sum if w_sum > 0.0 else None
    replace = len(dead_idx) > len(live_idx)
    donor_idx = rng.choice(live_idx, size=len(dead_idx), replace=replace, p=probs)

    apply_donor_splits(stats, dead_idx, donor_idx, delta=delta)
    return RevivalEvent(dead_indices=dead_idx, donor_indices=donor_idx, delta=delta)

