"""Pure algorithmic implementations for Gaussian Mixture Models with EMA updates.

Provides:
- GMMParameters: Dataclass holding weights, means, and covariances (diag/full).
- Log-sum-exp numerically stable responsibility computation.
- Batch sufficient statistics accumulation.
- M-step parameter extraction from EMA statistics.
- Covariance-aware principal axis revival for dead components.
- Pure GMM algorithms: Full-batch EM, Online EM, and Mini-batch EMA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import numpy as np
import scipy.linalg
from scipy.special import logsumexp

from core.revival import RevivalEvent
from core.stats import BatchStats, EMAStats

CovarianceType = Literal["diag", "full"]


@dataclass
class GMMParameters:
    """Parameters representing a Gaussian Mixture Model."""

    weights: np.ndarray  # Shape: (K,)
    means: np.ndarray  # Shape: (K, d)
    covars: np.ndarray  # Shape: (K, d) for diag, (K, d, d) for full
    cov_type: CovarianceType = "diag"
    eps_cov: float = 1e-6

    def copy(self) -> GMMParameters:
        return GMMParameters(
            weights=self.weights.copy(),
            means=self.means.copy(),
            covars=self.covars.copy(),
            cov_type=self.cov_type,
            eps_cov=self.eps_cov,
        )


def compute_log_responsibilities(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    cov_type: CovarianceType = "diag",
    eps_cov: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute log responsibilities and per-sample log marginal likelihood.

    Parameters
    ----------
    x : np.ndarray, shape (B, d)
        Input batch of vectors.
    weights : np.ndarray, shape (K,)
        Mixture component weights (must sum to 1).
    means : np.ndarray, shape (K, d)
        Component centroids.
    covars : np.ndarray, shape (K, d) or (K, d, d)
        Component covariances.
    cov_type : {"diag", "full"}
        Covariance structure.
    eps_cov : float
        Regularization floor for covariance.

    Returns
    -------
    log_resp : np.ndarray, shape (B, K)
        Normalized log-responsibilities log gamma_ik.
    log_prob_x : np.ndarray, shape (B,)
        Log marginal likelihood per sample log p(x_i).
    """
    x = np.asarray(x, dtype=np.float64)
    b, d = x.shape
    k = means.shape[0]

    # Guard weights against zero
    safe_weights = np.maximum(weights, 1e-30)
    safe_weights /= safe_weights.sum()
    log_weights = np.log(safe_weights)  # (K,)

    log_2pi = np.log(2.0 * np.pi)
    log_gauss = np.empty((b, k), dtype=np.float64)

    if cov_type == "diag":
        safe_covars = np.maximum(covars, eps_cov)  # (K, d)
        log_det = np.sum(np.log(safe_covars), axis=1)  # (K,)
        inv_covars = 1.0 / safe_covars  # (K, d)

        for j in range(k):
            diff = x - means[j]  # (B, d)
            quad = np.sum((diff * diff) * inv_covars[j], axis=1)  # (B,)
            log_gauss[:, j] = -0.5 * (d * log_2pi + log_det[j] + quad)
    elif cov_type == "full":
        eye = np.eye(d, dtype=np.float64)
        for j in range(k):
            cov_j = 0.5 * (covars[j] + covars[j].T) + eps_cov * eye
            try:
                chol = scipy.linalg.cholesky(cov_j, lower=True)
            except scipy.linalg.LinAlgError:
                # Regularize with eigenvalue floor if non-PD
                w, v = scipy.linalg.eigh(cov_j)
                w = np.maximum(w, eps_cov)
                cov_j = (v * w) @ v.T
                chol = scipy.linalg.cholesky(cov_j, lower=True)

            log_det = 2.0 * np.sum(np.log(np.diag(chol)))
            diff = (x - means[j]).T  # (d, B)
            sol = scipy.linalg.solve_triangular(chol, diff, lower=True)  # (d, B)
            quad = np.sum(sol * sol, axis=0)  # (B,)
            log_gauss[:, j] = -0.5 * (d * log_2pi + log_det + quad)
    else:
        raise ValueError(f"Unknown cov_type: {cov_type}")

    # Unnormalized log joint: log a_ik = log pi_k + log N(x_i | mu_k, Sigma_k)
    log_joint = log_gauss + log_weights.reshape(1, k)  # (B, K)

    # Log normalizer log p(x_i) via logsumexp
    log_prob_x = logsumexp(log_joint, axis=1)  # (B,)

    # Normalized log responsibilities
    log_resp = log_joint - log_prob_x.reshape(b, 1)
    return log_resp, log_prob_x


def compute_responsibilities(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    cov_type: CovarianceType = "diag",
    hard: bool = False,
    eps_cov: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute responsibilities gamma_ik and per-sample log likelihood.

    Parameters
    ----------
    x : np.ndarray, shape (B, d)
    weights : np.ndarray, shape (K,)
    means : np.ndarray, shape (K, d)
    covars : np.ndarray, shape (K, d) or (K, d, d)
    cov_type : {"diag", "full"}
    hard : bool, default False
        If True, assigns hard binary responsibilities (CEM).
    eps_cov : float

    Returns
    -------
    gamma : np.ndarray, shape (B, K)
    log_prob_x : np.ndarray, shape (B,)
    """
    log_resp, log_prob_x = compute_log_responsibilities(
        x, weights, means, covars, cov_type=cov_type, eps_cov=eps_cov
    )
    if hard:
        b, k = log_resp.shape
        gamma = np.zeros((b, k), dtype=np.float64)
        best_k = np.argmax(log_resp, axis=1)
        gamma[np.arange(b), best_k] = 1.0
    else:
        gamma = np.exp(log_resp)
        # Ensure row sums are strictly normalized
        row_sums = gamma.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
        gamma /= row_sums
    return gamma, log_prob_x


def compute_batch_stats(
    x: np.ndarray,
    gamma: np.ndarray,
    cov_type: CovarianceType = "diag",
) -> BatchStats:
    """Emit BatchStats from mini-batch vectors and responsibilities.

    Parameters
    ----------
    x : np.ndarray, shape (B, d)
    gamma : np.ndarray, shape (B, K)
    cov_type : {"diag", "full"}

    Returns
    -------
    BatchStats with counts, sums, and extras.
    """
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    b, d = x.shape
    k = gamma.shape[1]

    # Effective counts: N_k = sum_i gamma_ik
    counts = gamma.sum(axis=0)  # (K,)

    # First moments: S_k = sum_i gamma_ik * x_i
    sums = gamma.T @ x  # (K, d)

    extras: dict[str, np.ndarray] = {}
    if cov_type == "diag":
        # Second moments: S^(2)_k = sum_i gamma_ik * (x_i ** 2)
        sq_sums = gamma.T @ (x * x)  # (K, d)
        extras["sq_sums"] = sq_sums
    elif cov_type == "full":
        # Scatter matrix: C_k = sum_i gamma_ik * (x_i x_i.T)
        cov_sums = np.empty((k, d, d), dtype=np.float64)
        for j in range(k):
            w = np.sqrt(np.maximum(gamma[:, j : j + 1], 0.0))  # (B, 1)
            xw = x * w  # (B, d)
            cov_sums[j] = xw.T @ xw
        extras["cov_sums"] = cov_sums
    else:
        raise ValueError(f"Unknown cov_type: {cov_type}")

    return BatchStats(counts=counts, sums=sums, extras=extras)


def m_step_from_stats(
    stats: EMAStats,
    cov_type: CovarianceType = "diag",
    eps_cov: float = 1e-6,
) -> GMMParameters:
    """Extract GMMParameters from accumulated EMAStats.

    Parameters
    ----------
    stats : EMAStats
    cov_type : {"diag", "full"}
    eps_cov : float

    Returns
    -------
    GMMParameters
    """
    counts = np.maximum(stats.counts, 1e-12)  # (K,)
    k = counts.shape[0]
    d = stats.sums.shape[1]

    # Weights
    total_count = counts.sum()
    weights = counts / max(total_count, 1e-12)
    weights = np.maximum(weights, 1e-12)
    weights /= weights.sum()

    # Means
    means = stats.sums / counts.reshape(k, 1)

    # Covariances
    if cov_type == "diag":
        sq_sums = stats.extras.get("sq_sums")
        if sq_sums is None:
            covars = np.full((k, d), float(eps_cov), dtype=np.float64)
        else:
            raw_var = (sq_sums / counts.reshape(k, 1)) - (means * means)
            covars = np.maximum(raw_var, eps_cov)
    elif cov_type == "full":
        cov_sums = stats.extras.get("cov_sums")
        eye = np.eye(d, dtype=np.float64)
        covars = np.empty((k, d, d), dtype=np.float64)
        if cov_sums is None:
            for j in range(k):
                covars[j] = eye * eps_cov
        else:
            for j in range(k):
                raw_cov = (cov_sums[j] / counts[j]) - np.outer(means[j], means[j])
                sym_cov = 0.5 * (raw_cov + raw_cov.T) + eps_cov * eye
                # Ensure positive definiteness via eigenvalue clipping
                w, v = scipy.linalg.eigh(sym_cov)
                w = np.maximum(w, eps_cov)
                covars[j] = (v * w) @ v.T
    else:
        raise ValueError(f"Unknown cov_type: {cov_type}")

    return GMMParameters(
        weights=weights,
        means=means,
        covars=covars,
        cov_type=cov_type,
        eps_cov=eps_cov,
    )


def revive_gmm_components(
    stats: EMAStats,
    cov_type: CovarianceType = "diag",
    tau: float = 0.1,
    delta: float = 1.0 / 128.0,
    eps_cov: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> RevivalEvent:
    """Revive dead GMM components by splitting donors along principal covariance axis.

    When component count falls below tau * mean(N), donor cluster is chosen
    proportionally to count, and split along its top eigenvector.

    Parameters
    ----------
    stats : EMAStats
        Statistics to update in-place.
    cov_type : {"diag", "full"}
    tau : float
        Threshold fraction of mean count below which a component is considered dead.
    delta : float
        Perturbation scale along principal axis.
    eps_cov : float
        Covariance floor.
    rng : np.random.Generator | None

    Returns
    -------
    RevivalEvent containing dead_indices, donor_indices, and delta.
    """
    counts = stats.counts
    k = counts.shape[0]
    mean_n = float(counts.mean())
    if mean_n <= 0.0 or k <= 1:
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
    donor_weights = counts[live_idx]
    w_sum = float(donor_weights.sum())
    probs = donor_weights / w_sum if w_sum > 0.0 else None
    replace = len(dead_idx) > len(live_idx)
    donor_idx = rng.choice(live_idx, size=len(dead_idx), replace=replace, p=probs)

    # Extract current model parameters to compute principal axis
    params = m_step_from_stats(stats, cov_type=cov_type, eps_cov=eps_cov)
    means = params.means
    covars = params.covars
    d = means.shape[1]

    for dead_i, donor_i in zip(dead_idx, donor_idx):
        n_donor = float(stats.counts[donor_i])
        if n_donor <= 0.0:
            continue

        c_donor = means[donor_i]  # Centroid of donor

        # Find principal axis of donor covariance
        if cov_type == "diag":
            # Diagonal: top eigenvalue is max variance coordinate
            var_donor = covars[donor_i]  # (d,)
            top_dim = int(np.argmax(var_donor))
            lambda1 = float(var_donor[top_dim])
            v1 = np.zeros(d, dtype=np.float64)
            v1[top_dim] = 1.0
        else:
            # Full: compute top eigenpair of Sigma_donor
            cov_donor = covars[donor_i]  # (d, d)
            w, v = scipy.linalg.eigh(cov_donor)
            lambda1 = float(max(w[-1], eps_cov))
            v1 = v[:, -1]
            norm_v = np.linalg.norm(v1)
            if norm_v > 0.0:
                v1 /= norm_v

        # Perturbation along principal axis: delta * sqrt(lambda1) * v1
        shift = delta * np.sqrt(max(lambda1, 1e-12)) * v1
        if np.all(shift == 0.0):
            shift = delta * np.ones(d, dtype=np.float64)

        mean_dead = c_donor + shift
        mean_donor_new = c_donor - shift

        half_n = 0.5 * n_donor
        stats.counts[dead_i] = half_n
        stats.counts[donor_i] = half_n

        stats.sums[dead_i] = half_n * mean_dead
        stats.sums[donor_i] = half_n * mean_donor_new

        # Update second-moment statistics
        if cov_type == "diag":
            var_base = covars[donor_i]
            if "sq_sums" in stats.extras:
                stats.extras["sq_sums"][dead_i] = half_n * (var_base + mean_dead * mean_dead)
                stats.extras["sq_sums"][donor_i] = half_n * (var_base + mean_donor_new * mean_donor_new)
        elif cov_type == "full":
            cov_base = covars[donor_i]
            if "cov_sums" in stats.extras:
                stats.extras["cov_sums"][dead_i] = half_n * (cov_base + np.outer(mean_dead, mean_dead))
                stats.extras["cov_sums"][donor_i] = half_n * (cov_base + np.outer(mean_donor_new, mean_donor_new))

    return RevivalEvent(dead_indices=dead_idx, donor_indices=donor_idx, delta=delta)


def init_gmm_parameters(
    x: np.ndarray,
    k: int,
    cov_type: CovarianceType = "diag",
    eps_cov: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> GMMParameters:
    """Initialize GMM parameters from data using k-means++ centroids.

    Parameters
    ----------
    x : np.ndarray, shape (N, d)
    k : int
    cov_type : {"diag", "full"}
    eps_cov : float
    rng : np.random.Generator | None

    Returns
    -------
    GMMParameters
    """
    rng = rng if rng is not None else np.random.default_rng()
    n, d = x.shape
    k = min(k, n)

    # Initial uniform weights
    weights = np.full(k, 1.0 / k, dtype=np.float64)

    # K-means++ initialization for means
    means = np.empty((k, d), dtype=np.float64)
    first_idx = rng.integers(0, n)
    means[0] = x[first_idx]

    dist_sq = np.sum((x - means[0]) ** 2, axis=1)
    for j in range(1, k):
        dist_sum = float(dist_sq.sum())
        if dist_sum <= 0.0 or not np.isfinite(dist_sum):
            probs = None  # Fallback to uniform random choice when all points coincide
        else:
            probs = dist_sq / dist_sum
        next_idx = rng.choice(n, p=probs)
        means[j] = x[next_idx]
        new_dist_sq = np.sum((x - means[j]) ** 2, axis=1)
        dist_sq = np.minimum(dist_sq, new_dist_sq)

    # Global variance floor from data
    var_data = np.var(x, axis=0) + eps_cov  # (d,)
    if cov_type == "diag":
        covars = np.tile(var_data, (k, 1))  # (K, d)
    elif cov_type == "full":
        covars = np.empty((k, d, d), dtype=np.float64)
        diag_mat = np.diag(var_data)
        for j in range(k):
            covars[j] = diag_mat.copy()
    else:
        raise ValueError(f"Unknown cov_type: {cov_type}")

    return GMMParameters(
        weights=weights,
        means=means,
        covars=covars,
        cov_type=cov_type,
        eps_cov=eps_cov,
    )


def evaluate_gmm_nll(
    x: np.ndarray,
    params: GMMParameters,
    batch_size: int = 4096,
) -> float:
    """Evaluate negative log-likelihood per sample on dataset x.

    Parameters
    ----------
    x : np.ndarray, shape (N, d)
    params : GMMParameters
    batch_size : int

    Returns
    -------
    float: Negative log-likelihood per sample.
    """
    n = x.shape[0]
    total_log_lik = 0.0
    for start in range(0, n, batch_size):
        batch = x[start : start + batch_size]
        _, log_prob = compute_log_responsibilities(
            batch,
            params.weights,
            params.means,
            params.covars,
            cov_type=params.cov_type,
            eps_cov=params.eps_cov,
        )
        total_log_lik += float(np.sum(log_prob))
    return -total_log_lik / max(n, 1)
