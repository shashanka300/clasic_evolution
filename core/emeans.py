"""Reference e-means implementation (Bessa et al., arXiv:2611.09501) and toy k-means plugin."""

from __future__ import annotations

import numba
import numpy as np
import torch

from core.plugin import ClassicAlgoPlugin, EMARunResult, run_ema_engine
from core.revival import RevivalEvent, revive_dead_units
from core.scaling import emeans_alpha0, emeans_batch_size
from core.schedule import CosineSchedule, StepSchedule
from core.stats import BatchStats, EMAStats


@numba.njit(cache=True)
def _accumulate_counts_and_sums(batch: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate cluster counts and vector sums in float64."""
    n, d = batch.shape
    counts = np.zeros(k, dtype=np.float64)
    sums = np.zeros((k, d), dtype=np.float64)
    for i in range(n):
        c = labels[i]
        counts[c] += 1.0
        for j in range(d):
            sums[c, j] += batch[i, j]
    return counts, sums


def resolve_device(device: str = "auto", n_batch: int = 0, k: int = 0) -> str:
    """Select ``cuda``/``mps`` or ``cpu``; the accelerator wins for larger dense matrix multiplies."""
    from core.device import accel
    if device in ("cpu", "mps", "cuda"):
        if device != "cpu" and accel() != device:
            return "cpu"
        return device
    if accel() != "cpu" and n_batch * k >= 262_144:
        return accel()
    return "cpu"


def assign_nearest(
    batch: np.ndarray,
    centroids: np.ndarray,
    device: str = "auto",
    chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each row in ``batch`` to its nearest centroid (returns ``labels, min_sq_dists``)."""
    n, d = batch.shape
    k = centroids.shape[0]
    backend = resolve_device(device, n, k)

    if backend in ("mps", "cuda"):
        c_t = torch.from_numpy(np.array(centroids, dtype=np.float32, copy=True)).to(backend)
        c_norm2 = (c_t * c_t).sum(dim=1)
        labels = np.empty(n, dtype=np.int64)
        min_dists = np.empty(n, dtype=np.float32)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            x_t = torch.from_numpy(np.array(batch[start:end], dtype=np.float32, copy=True)).to(backend)
            x_norm2 = (x_t * x_t).sum(dim=1, keepdim=True)
            dists = x_norm2 + c_norm2.unsqueeze(0) - 2.0 * (x_t @ c_t.T)
            vals, idx = dists.min(dim=1)
            labels[start:end] = idx.cpu().numpy()
            min_dists[start:end] = torch.clamp(vals, min=0.0).cpu().numpy()
        return labels, min_dists

    import faiss

    index = faiss.IndexFlatL2(d)
    index.add(np.ascontiguousarray(centroids, dtype=np.float32))
    dists, idx = index.search(np.ascontiguousarray(batch, dtype=np.float32), 1)
    return idx[:, 0].astype(np.int64, copy=False), np.maximum(dists[:, 0], 0.0)


def eval_kmeans_mse(
    x: np.ndarray,
    centroids: np.ndarray,
    batch_size: int = 65536,
    device: str = "auto",
) -> float:
    """Compute exact mean squared error ``(1/n) * sum ||x - c_{a(x)}||^2``."""
    n = x.shape[0]
    total = 0.0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = np.asarray(x[start:end], dtype=np.float32)
        _, sq_dists = assign_nearest(xb, centroids, device=device)
        total += float(np.sum(sq_dists, dtype=np.float64))
    return total / float(n)


def sample_initial_centroids(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``k`` distinct initial centroids uniformly without replacement."""
    n = x.shape[0]
    idx = rng.choice(n, size=k, replace=k > n)
    return np.ascontiguousarray(x[idx], dtype=np.float32)


class EMeans:
    """Reference e-means clustering implementation matching Algorithm 1 of arXiv:2611.09501."""

    def __init__(
        self,
        k: int,
        epochs: float = 2.0,
        batch_size: int | None = None,
        alpha0: float | None = None,
        eps: float = 0.1,
        tau: float = 0.1,
        delta: float = 1.0 / 128.0,
        revival_every: int = 5,
        finalize_mode: str = "last_epoch",
        slow_half_life: float = 50.0,
        device: str = "auto",
        seed: int = 0,
    ) -> None:
        self.k = int(k)
        self.epochs = float(epochs)
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.eps = float(eps)
        self.tau = float(tau)
        self.delta = float(delta)
        self.revival_every = int(revival_every)
        self.finalize_mode = finalize_mode
        self.slow_half_life = float(slow_half_life)
        self.device = device
        self.seed = int(seed)
        self.centroids_: np.ndarray | None = None
        self.slow_centroids_: np.ndarray | None = None
        self.revivals_: int = 0

    def fit(self, x: np.ndarray, schedule: StepSchedule | None = None) -> EMeans:
        n = x.shape[0]
        b = int(self.batch_size) if self.batch_size is not None else emeans_batch_size(n, self.k, self.epochs)
        b = min(max(b, 1), n)
        a0 = float(self.alpha0) if self.alpha0 is not None else emeans_alpha0(n, self.k, self.epochs, b)

        steps_per_epoch = max(n // b, 1)
        total_steps = max(int(round(self.epochs * steps_per_epoch)), 1)
        sched = schedule if schedule is not None else CosineSchedule(alpha0=a0, total_steps=total_steps, eps=self.eps)

        plugin = ToyKMeansPlugin(
            k=self.k,
            tau=self.tau,
            delta=self.delta,
            device=self.device,
            seed=self.seed,
        )
        res: EMARunResult = run_ema_engine(
            plugin=plugin,
            x=x,
            epochs=self.epochs,
            batch_size=b,
            schedule=sched,
            revival_every=self.revival_every,
            finalize_mode=self.finalize_mode,
            slow_half_life=self.slow_half_life,
            seed=self.seed,
        )
        self.batch_size = b
        self.alpha0 = a0
        self.centroids_ = res.params
        self.slow_centroids_ = res.slow_params
        self.revivals_ = res.revivals
        return self

    def score(self, x: np.ndarray) -> float:
        if self.centroids_ is None:
            raise RuntimeError("EMeans must be fitted before calling score()")
        return eval_kmeans_mse(x, self.centroids_, device=self.device)


class ToyKMeansPlugin(ClassicAlgoPlugin):
    """Plain k-means implemented via ``ClassicAlgoPlugin`` for equivalence testing."""

    def __init__(
        self,
        k: int,
        tau: float = 0.1,
        delta: float = 1.0 / 128.0,
        device: str = "auto",
        seed: int = 0,
    ) -> None:
        self.k = int(k)
        self.tau = float(tau)
        self.delta = float(delta)
        self.device = device
        self.seed = int(seed)
        self._centroids: np.ndarray | None = None

    def init(self, x0: np.ndarray) -> EMAStats:
        rng = np.random.default_rng(self.seed)
        c0 = sample_initial_centroids(x0, self.k, rng)
        self._centroids = c0
        return EMAStats.from_params(c0)

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

