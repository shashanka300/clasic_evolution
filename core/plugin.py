"""Plugin interface and shared EMA runner for classic algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np

from core.finalize import LastEpochMeanFinalizer, SlowEMAFinalizer
from core.revival import RevivalEvent, revive_dead_units
from core.schedule import StepSchedule
from core.stats import BatchStats, EMAStats


class ClassicAlgoPlugin(ABC):
    """Interface implemented by every track plugging into the e-means engine."""

    @abstractmethod
    def init(self, x0: np.ndarray) -> EMAStats:
        """Initialize sufficient statistics from an initial dataset or subset ``x0``."""

    @abstractmethod
    def assign(self, batch: np.ndarray) -> BatchStats:
        """Run one mini-batch E-step / assignment and emit sufficient statistics."""

    @abstractmethod
    def params_from_stats(self, stats: EMAStats) -> np.ndarray:
        """Compute current model parameters from EMA sufficient statistics."""

    def revive(self, stats: EMAStats, step: int, rng: np.random.Generator | None = None) -> RevivalEvent:
        """Revive dead units in ``stats`` (defaults to Faiss-style donor splitting)."""
        return revive_dead_units(stats, rng=rng)

    @abstractmethod
    def loss(self, x: np.ndarray) -> float:
        """Evaluate objective/loss on dataset ``x`` using current parameters."""


@dataclass
class EMARunResult:
    """Outputs produced by ``run_ema_engine``."""

    params: np.ndarray
    stats: EMAStats
    steps: int
    revivals: int
    slow_params: np.ndarray | None = None


def iter_epoch_batches(
    n: int,
    batch_size: int,
    epochs: float,
    rng: np.random.Generator,
):
    """Yield slice indices for ``epochs`` epochs (supports integer and fractional epochs)."""
    b = min(max(int(batch_size), 1), n)
    steps_per_epoch = max(n // b, 1)
    total_steps = max(int(round(epochs * steps_per_epoch)), 1)
    step = 0
    while step < total_steps:
        perm = rng.permutation(n)
        for start in range(0, steps_per_epoch * b, b):
            if step >= total_steps:
                break
            yield step, total_steps, steps_per_epoch, perm[start : start + b]
            step += 1


def run_ema_engine(
    plugin: ClassicAlgoPlugin,
    x: np.ndarray,
    epochs: float,
    batch_size: int,
    schedule: StepSchedule,
    revival_every: int = 5,
    finalize_mode: str = "last_epoch",
    slow_half_life: float = 50.0,
    seed: int = 0,
    on_epoch_end: Callable[[int, np.ndarray], None] | None = None,
) -> EMARunResult:
    """Execute the generic e-means recipe on any ``ClassicAlgoPlugin``."""
    rng = np.random.default_rng(seed)
    stats = plugin.init(x)
    params = plugin.params_from_stats(stats)

    n = x.shape[0]
    b = min(max(int(batch_size), 1), n)
    steps_per_epoch = max(n // b, 1)
    total_steps = max(int(round(epochs * steps_per_epoch)), 1)
    final_epoch_start = max(0, total_steps - steps_per_epoch)

    last_epoch_finalizer = (
        LastEpochMeanFinalizer(k=params.shape[0], shape=params.shape[1:])
        if finalize_mode in ("last_epoch", "both")
        else None
    )
    slow_finalizer = (
        SlowEMAFinalizer(params, half_life=slow_half_life)
        if finalize_mode in ("slow_ema", "both")
        else None
    )

    revivals = 0
    for step, _, spe, batch_idx in iter_epoch_batches(n, b, epochs, rng):
        batch = np.asarray(x[batch_idx], dtype=np.float32)
        bstats = plugin.assign(batch)
        alpha = schedule(step)
        stats.update(bstats, alpha=alpha)

        if last_epoch_finalizer is not None and step >= final_epoch_start:
            last_epoch_finalizer.observe(bstats)
        if slow_finalizer is not None:
            slow_finalizer.observe(bstats)

        if revival_every > 0 and (step + 1) % revival_every == 0:
            event = plugin.revive(stats, step=step, rng=rng)
            revivals += len(event.dead_indices)
            if slow_finalizer is not None and len(event.dead_indices) > 0:
                slow_finalizer.mirror_revival(event)

        params = plugin.params_from_stats(stats)

        if on_epoch_end is not None and (step + 1) % spe == 0:
            on_epoch_end((step + 1) // spe, params)

    slow_params = slow_finalizer.finalize() if slow_finalizer is not None else None
    if last_epoch_finalizer is not None:
        final_params = last_epoch_finalizer.finalize(params)
        if hasattr(plugin, "set_params"):
            plugin.set_params(final_params)
    elif slow_params is not None:
        final_params = slow_params
        if hasattr(plugin, "set_params"):
            plugin.set_params(final_params)
    else:
        final_params = params

    return EMARunResult(
        params=final_params,
        stats=stats,
        steps=total_steps,
        revivals=revivals,
        slow_params=slow_params,
    )

