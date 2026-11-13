"""EMA accumulator for arbitrary sufficient statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class BatchStats:
    """Batch sufficient statistics emitted by assignment or E-step."""

    counts: np.ndarray
    sums: np.ndarray
    extras: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class EMAStats:
    """Exponential moving average of per-unit sufficient statistics.

    Initialized with ``counts = N0`` (default ``1e-10``) and ``sums = N0 * init_params``
    so ``means()`` returns the exact initial parameters before updates while letting
    initial priors decay immediately once real batches arrive.
    """

    counts: np.ndarray
    sums: np.ndarray
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_params(
        cls,
        init_params: np.ndarray,
        n0: float = 1e-10,
        extra_inits: dict[str, np.ndarray] | None = None,
    ) -> EMAStats:
        """Create EMAStats initialized at ``init_params`` with pseudo-count ``n0``."""
        params64 = np.asarray(init_params, dtype=np.float64)
        k = params64.shape[0]
        counts = np.full((k,), float(n0), dtype=np.float64)
        broadcast = (k,) + (1,) * (params64.ndim - 1)
        sums = counts.reshape(broadcast) * params64
        extras: dict[str, np.ndarray] = {}
        if extra_inits:
            for name, val in extra_inits.items():
                arr = np.asarray(val, dtype=np.float64)
                eb = (k,) + (1,) * (arr.ndim - 1) if arr.shape[0] == k else (1,) * arr.ndim
                extras[name] = (counts.reshape(eb) * arr) if arr.shape[0] == k else (float(n0) * arr)
        return cls(counts=counts, sums=sums, extras=extras)

    def update(self, batch: BatchStats, alpha: float) -> None:
        """Apply ``X <- (1 - alpha) * X + alpha * batch_X`` to all statistics."""
        gamma = 1.0 - float(alpha)
        self.counts *= gamma
        self.counts += alpha * np.asarray(batch.counts, dtype=np.float64)
        self.sums *= gamma
        self.sums += alpha * np.asarray(batch.sums, dtype=np.float64)
        for name, bval in batch.extras.items():
            bval64 = np.asarray(bval, dtype=np.float64)
            if name not in self.extras:
                self.extras[name] = np.zeros_like(bval64)
            self.extras[name] *= gamma
            self.extras[name] += alpha * bval64

    def means(self, out_dtype: np.dtype | type = np.float32) -> np.ndarray:
        """Return ``sums / counts`` with safe division for zero-count units."""
        k = self.counts.shape[0]
        broadcast = (k,) + (1,) * (self.sums.ndim - 1)
        denom = np.where(self.counts > 0.0, self.counts, 1.0).reshape(broadcast)
        out = self.sums / denom
        return out.astype(out_dtype, copy=False)

    def extra_means(self, name: str, out_dtype: np.dtype | type = np.float32) -> np.ndarray:
        """Return ``extras[name] / counts`` for unit-aligned second-moment/matrix stats."""
        arr = self.extras[name]
        k = self.counts.shape[0]
        if arr.shape[0] != k:
            total = max(float(self.counts.sum()), 1e-30)
            return (arr / total).astype(out_dtype, copy=False)
        broadcast = (k,) + (1,) * (arr.ndim - 1)
        denom = np.where(self.counts > 0.0, self.counts, 1.0).reshape(broadcast)
        return (arr / denom).astype(out_dtype, copy=False)

    def copy(self) -> EMAStats:
        """Return a deep copy of all accumulated statistics."""
        return EMAStats(
            counts=self.counts.copy(),
            sums=self.sums.copy(),
            extras={k: v.copy() for k, v in self.extras.items()},
        )

