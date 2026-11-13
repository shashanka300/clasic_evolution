"""ClassicAlgoPlugin implementation for Gaussian Mixture Models with EMA updates."""

from __future__ import annotations

from typing import Any, Literal
import numpy as np
from sklearn.mixture import GaussianMixture

from core.plugin import ClassicAlgoPlugin, EMARunResult, run_ema_engine
from core.revival import RevivalEvent
from core.schedule import (
    CosineSchedule,
    InverseTimeSchedule,
    StepSchedule,
    make_schedule,
)
from core.stats import BatchStats, EMAStats
try:
    from .gmm import (
        CovarianceType,
        GMMParameters,
        compute_batch_stats,
        compute_responsibilities,
        evaluate_gmm_nll,
        init_gmm_parameters,
        m_step_from_stats,
        revive_gmm_components,
    )
except ImportError:
    from gmm import (
        CovarianceType,
        GMMParameters,
        compute_batch_stats,
        compute_responsibilities,
        evaluate_gmm_nll,
        init_gmm_parameters,
        m_step_from_stats,
        revive_gmm_components,
    )


class GMMEmaPlugin(ClassicAlgoPlugin):
    """Gaussian Mixture Model plugin for the e-means engine.

    Maintains running sufficient statistics (counts, linear sums, and
    second-moment / scatter matrices) via EMA with cosine annealing.
    Supports diagonal and full covariance models, Hard CEM, and
    covariance-aware principal axis component revival.
    """

    def __init__(
        self,
        k: int,
        d: int,
        cov_type: CovarianceType = "diag",
        hard: bool = False,
        tau: float = 0.1,
        delta: float = 1.0 / 128.0,
        eps_cov: float = 1e-6,
        seed: int = 0,
    ) -> None:
        self.k = int(k)
        self.d = int(d)
        self.cov_type: CovarianceType = cov_type
        self.hard = bool(hard)
        self.tau = float(tau)
        self.delta = float(delta)
        self.eps_cov = float(eps_cov)
        self.seed = int(seed)
        self.params: GMMParameters | None = None

    def init(self, x0: np.ndarray) -> EMAStats:
        """Initialize GMM parameters from data x0 and return starting EMAStats."""
        rng = np.random.default_rng(self.seed)
        self.params = init_gmm_parameters(
            x0,
            k=self.k,
            cov_type=self.cov_type,
            eps_cov=self.eps_cov,
            rng=rng,
        )

        # Synchronize effective k with initialized components (handles k > N)
        self.k = self.params.means.shape[0]
        k, d = self.k, self.d
        extra_inits: dict[str, np.ndarray] = {}
        if self.cov_type == "diag":
            # E[X^2] = Var + Mean^2
            sq_inits = self.params.covars + (self.params.means * self.params.means)
            extra_inits["sq_sums"] = sq_inits
        elif self.cov_type == "full":
            # E[X X^T] = Cov + Mean Mean^T
            cov_inits = np.empty((k, d, d), dtype=np.float64)
            for j in range(k):
                m_j = self.params.means[j]
                cov_inits[j] = self.params.covars[j] + np.outer(m_j, m_j)
            extra_inits["cov_sums"] = cov_inits

        return EMAStats.from_params(
            init_params=self.params.means,
            n0=1e-10,
            extra_inits=extra_inits,
        )

    def assign(self, batch: np.ndarray) -> BatchStats:
        """Execute E-step on mini-batch and emit sufficient statistics."""
        assert self.params is not None, "Plugin must be initialized before assign()"
        gamma, _ = compute_responsibilities(
            batch,
            weights=self.params.weights,
            means=self.params.means,
            covars=self.params.covars,
            cov_type=self.cov_type,
            hard=self.hard,
            eps_cov=self.eps_cov,
        )
        return compute_batch_stats(batch, gamma, cov_type=self.cov_type)

    def params_from_stats(self, stats: EMAStats) -> np.ndarray:
        """Derive updated GMM parameters from accumulated EMA statistics."""
        self.params = m_step_from_stats(stats, cov_type=self.cov_type, eps_cov=self.eps_cov)
        return self.params.means

    def set_params(self, params: np.ndarray) -> None:
        """Set current centroids from external finalizer (e.g. LastEpochMeanFinalizer)."""
        if self.params is not None:
            self.params.means = np.asarray(params, dtype=np.float64)

    def revive(self, stats: EMAStats, step: int, rng: np.random.Generator | None = None) -> RevivalEvent:
        """Revive dead components by splitting live donor along top covariance eigenvector."""
        return revive_gmm_components(
            stats,
            cov_type=self.cov_type,
            tau=self.tau,
            delta=self.delta,
            eps_cov=self.eps_cov,
            rng=rng,
        )

    def loss(self, x: np.ndarray) -> float:
        """Compute negative log-likelihood per sample on dataset x."""
        assert self.params is not None, "Plugin must be initialized before loss()"
        return evaluate_gmm_nll(x, self.params)


def fit_gmm_ema(
    x: np.ndarray,
    k: int,
    epochs: float = 3.0,
    batch_size: int = 512,
    alpha0: float = 0.05,
    cov_type: CovarianceType = "diag",
    hard: bool = False,
    tau: float = 0.1,
    delta: float = 1.0 / 128.0,
    eps_cov: float = 1e-6,
    revival_every: int = 5,
    schedule: StepSchedule | None = None,
    seed: int = 0,
) -> tuple[GMMEmaPlugin, EMARunResult]:
    """Fit GMM via mini-batch EMA sufficient statistics and cosine annealing."""
    n, d = x.shape
    b = min(max(int(batch_size), 1), n)
    steps_per_epoch = max(n // b, 1)
    total_steps = max(int(round(epochs * steps_per_epoch)), 1)

    if schedule is None:
        schedule = CosineSchedule(alpha0=alpha0, total_steps=total_steps, eps=0.1)

    plugin = GMMEmaPlugin(
        k=k,
        d=d,
        cov_type=cov_type,
        hard=hard,
        tau=tau,
        delta=delta,
        eps_cov=eps_cov,
        seed=seed,
    )

    result = run_ema_engine(
        plugin=plugin,
        x=x,
        epochs=epochs,
        batch_size=batch_size,
        schedule=schedule,
        revival_every=revival_every,
        finalize_mode="none",  # GMM preserves both means and covariances from stats
        seed=seed,
    )

    return plugin, result


def fit_full_batch_em(
    x: np.ndarray,
    k: int,
    cov_type: CovarianceType = "diag",
    max_iter: int = 25,
    eps_cov: float = 1e-6,
    seed: int = 0,
) -> GMMParameters:
    """Fit baseline full-batch EM via scikit-learn GaussianMixture."""
    k_eff = min(k, x.shape[0])
    gm = GaussianMixture(
        n_components=k_eff,
        covariance_type=cov_type,
        max_iter=max_iter,
        reg_covar=eps_cov,
        random_state=seed,
    )
    gm.fit(x)
    return GMMParameters(
        weights=gm.weights_.astype(np.float64),
        means=gm.means_.astype(np.float64),
        covars=gm.covariances_.astype(np.float64),
        cov_type=cov_type,
        eps_cov=eps_cov,
    )


def fit_online_em(
    x: np.ndarray,
    k: int,
    epochs: float = 3.0,
    batch_size: int = 512,
    kappa: float = 0.6,
    tau0: float = 100.0,
    cov_type: CovarianceType = "diag",
    eps_cov: float = 1e-6,
    seed: int = 0,
) -> tuple[GMMEmaPlugin, EMARunResult]:
    """Fit baseline Online EM (Cappé & Moulines 2009) with step size t^-kappa."""
    n, d = x.shape
    b = min(max(int(batch_size), 1), n)
    steps_per_epoch = max(n // b, 1)
    total_steps = max(int(round(epochs * steps_per_epoch)), 1)

    # Online EM schedule: alpha_t = (tau0 + t)^(-kappa)
    schedule = InverseTimeSchedule(alpha0=1.0, tau0=tau0, kappa=kappa)

    plugin = GMMEmaPlugin(
        k=k,
        d=d,
        cov_type=cov_type,
        hard=False,
        tau=0.0,  # No component revival in classic Cappé & Moulines 2009
        delta=0.0,
        eps_cov=eps_cov,
        seed=seed,
    )

    result = run_ema_engine(
        plugin=plugin,
        x=x,
        epochs=epochs,
        batch_size=batch_size,
        schedule=schedule,
        revival_every=0,  # No revival
        finalize_mode="none",
        seed=seed,
    )

    return plugin, result
