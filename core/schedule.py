"""Step-size schedules alpha(t) = 1 - gamma(t) for EMA optimization."""

from __future__ import annotations

from dataclasses import dataclass
import math


class StepSchedule:
    """Base interface for step-size schedules."""

    def __call__(self, step: int) -> float:
        raise NotImplementedError


@dataclass
class CosineSchedule(StepSchedule):
    """e-means cosine schedule: ``alpha0 * (eps + (1 - eps) * 0.5 * (1 + cos(pi * t / M)))``."""

    alpha0: float
    total_steps: int
    eps: float = 0.1

    def __call__(self, step: int) -> float:
        m = max(int(self.total_steps), 1)
        t = min(max(int(step), 0), m)
        cos_term = 0.5 * (1.0 + math.cos(math.pi * t / m))
        return float(self.alpha0 * (self.eps + (1.0 - self.eps) * cos_term))


@dataclass
class ConstantSchedule(StepSchedule):
    """Constant step size ``alpha(t) = alpha0`` (VQ-VAE style)."""

    alpha0: float

    def __call__(self, step: int) -> float:
        return float(self.alpha0)


@dataclass
class InverseTimeSchedule(StepSchedule):
    """Sculley / Robbins-Monro inverse-time schedule ``alpha0 * (tau0 + t)^(-kappa)``."""

    alpha0: float
    tau0: float = 1.0
    kappa: float = 1.0

    def __call__(self, step: int) -> float:
        t = max(int(step), 0)
        return float(min(1.0, self.alpha0 * ((self.tau0 + t) ** (-self.kappa))))


@dataclass
class WarmRestartCosineSchedule(StepSchedule):
    """Cosine schedule supporting periodic or drift-triggered warm restarts."""

    alpha0: float
    period: int
    eps: float = 0.1
    t_mult: float = 1.0
    _cycle_start: int = 0
    _cycle_len: int = 0

    def __post_init__(self) -> None:
        self._cycle_len = max(int(self.period), 1)

    def trigger_restart(self, step: int, new_period: int | None = None) -> None:
        """Reset cosine cycle at ``step`` (used by drift detectors in T08)."""
        self._cycle_start = max(int(step), 0)
        if new_period is not None:
            self._cycle_len = max(int(new_period), 1)

    def __call__(self, step: int) -> float:
        t_rel = max(int(step) - self._cycle_start, 0)
        while self.t_mult > 0 and t_rel >= self._cycle_len and self._cycle_len > 0:
            self._cycle_start += self._cycle_len
            self._cycle_len = max(int(round(self._cycle_len * self.t_mult)), 1)
            t_rel = int(step) - self._cycle_start
        cos_term = 0.5 * (1.0 + math.cos(math.pi * min(t_rel, self._cycle_len) / self._cycle_len))
        return float(self.alpha0 * (self.eps + (1.0 - self.eps) * cos_term))


_SCHEDULE_BUILDERS = {
    "cosine": lambda alpha0, total_steps, **kw: CosineSchedule(alpha0, total_steps, eps=kw.get("eps", 0.1)),
    "constant": lambda alpha0, total_steps, **kw: ConstantSchedule(alpha0),
    "inverse_time": lambda alpha0, total_steps, **kw: InverseTimeSchedule(
        alpha0, tau0=kw.get("tau0", 1.0), kappa=kw.get("kappa", 1.0)
    ),
    "warm_restart": lambda alpha0, total_steps, **kw: WarmRestartCosineSchedule(
        alpha0,
        period=kw.get("period", max(total_steps, 1)),
        eps=kw.get("eps", 0.1),
        t_mult=kw.get("t_mult", 1.0),
    ),
}


def make_schedule(kind: str, alpha0: float, total_steps: int, **kwargs: float) -> StepSchedule:
    """Create a step schedule by name (``cosine``, ``constant``, ``inverse_time``, ``warm_restart``)."""
    if kind not in _SCHEDULE_BUILDERS:
        raise ValueError(f"Unknown schedule {kind!r}; expected one of {tuple(_SCHEDULE_BUILDERS)}")
    return _SCHEDULE_BUILDERS[kind](alpha0, total_steps, **kwargs)

