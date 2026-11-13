"""Algorithmic core for Online Dictionary Learning and NMF with EMA sufficient statistics."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any
import numpy as np

_track_dir = Path(__file__).resolve().parent
if str(_track_dir) not in sys.path:
    sys.path.insert(0, str(_track_dir))


def fista_sparse_code(
    x: np.ndarray,
    d: np.ndarray,
    alpha_reg: float,
    max_iter: int = 50,
    tol: float = 1e-4,
    non_negative: bool = False,
) -> np.ndarray:
    """Solve min_alpha 0.5 * ||x - alpha @ d||_2^2 + alpha_reg * ||alpha||_1 via FISTA.

    Parameters
    ----------
    x : np.ndarray of shape (b, d)
        Mini-batch of input data vectors.
    d : np.ndarray of shape (k, d)
        Current dictionary atoms.
    alpha_reg : float
        L1 regularization parameter (lambda).
    max_iter : int
        Maximum number of accelerated proximal gradient iterations.
    tol : float
        Convergence tolerance on relative code change.
    non_negative : bool
        If True, enforces alpha >= 0 (for NMF).

    Returns
    -------
    alpha : np.ndarray of shape (b, k)
        Sparse code representation.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    d_arr = np.asarray(d, dtype=np.float64)

    if x_arr.ndim != 2 or d_arr.ndim != 2:
        raise ValueError(f"x and d must be 2D, got x: {x_arr.shape}, d: {d_arr.shape}")

    b, d_dim = x_arr.shape
    k, d_dict = d_arr.shape

    if d_dim != d_dict:
        raise ValueError(f"Input dimension d={d_dim} does not match dictionary d={d_dict}")

    if b == 0:
        return np.zeros((0, k), dtype=np.float32)

    # Check for all-zero or near-zero inputs
    x_norm = np.linalg.norm(x_arr)
    if x_norm < 1e-12:
        return np.zeros((b, k), dtype=np.float32)

    # Gram matrix G = D @ D.T in (k, k), cross-correlation C = X @ D.T in (b, k)
    g = d_arr @ d_arr.T
    c = x_arr @ d_arr.T

    # Lipschitz constant L = ||G||_2 = sigma_max(G)
    # Using spectral norm (2-norm) with fallback guard
    try:
        lipschitz = float(np.linalg.norm(g, 2))
    except Exception:
        lipschitz = float(np.trace(g))

    if lipschitz < 1e-8:
        lipschitz = 1e-8

    step_size = 1.0 / (lipschitz + 1e-8)
    reg_step = float(alpha_reg) * step_size

    # Initialize codes
    alpha = np.zeros((b, k), dtype=np.float64)
    y = alpha.copy()
    t = 1.0

    for it in range(max(1, int(max_iter))):
        # Gradient of 0.5 * ||x - y @ d||_2^2 is (y @ g - c)
        grad = y @ g - c
        v = y - step_size * grad

        if non_negative:
            # Non-negative proximal thresholding: max(0, v - step_size * lambda)
            alpha_next = np.maximum(0.0, v - reg_step)
        else:
            # Standard soft-thresholding: sign(v) * max(0, |v| - step_size * lambda)
            alpha_next = np.sign(v) * np.maximum(0.0, np.abs(v) - reg_step)

        # Relative change convergence check
        code_norm = float(np.linalg.norm(alpha_next))
        diff_norm = float(np.linalg.norm(alpha_next - alpha))
        if it > 0 and diff_norm / (code_norm + 1e-10) < tol:
            alpha = alpha_next
            break

        # Nesterov momentum acceleration
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        momentum = (t - 1.0) / t_next
        y = alpha_next + momentum * (alpha_next - alpha)
        t = t_next
        alpha = alpha_next

    return alpha.astype(np.float32)


def block_coordinate_descent_update(
    d_init: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    non_negative: bool = False,
    n_iter: int = 2,
    eps: float = 1e-12,
) -> np.ndarray:
    """Mairal-style Block Coordinate Descent dictionary atom update.

    Solves: min_D 0.5 * Tr(D.T @ A @ D) - Tr(D.T @ B)
    Subject to ||d_k||_2 <= 1 (and d_k >= 0 if non_negative).

    Update rule per atom k:
        u_k = d_k + (B_k - A_k @ D) / A_kk
        d_k = u_k / max(1.0, ||u_k||_2)

    Parameters
    ----------
    d_init : np.ndarray of shape (k, d)
        Initial dictionary atoms.
    a : np.ndarray of shape (k, k)
        Accumulated code autocorrelation matrix (sum alpha_i alpha_i^T).
    b : np.ndarray of shape (k, d)
        Accumulated cross-correlation matrix (sum alpha_i x_i^T).
    non_negative : bool
        If True, projects atoms to non-negative orthant (for NMF).
    n_iter : int
        Number of coordinate descent sweeps across all k atoms.
    eps : float
        Guard threshold for A_kk to prevent division by zero.

    Returns
    -------
    d_updated : np.ndarray of shape (k, d)
        Updated dictionary with unit-ball constrained atoms.
    """
    d = np.array(d_init, dtype=np.float64, copy=True)
    a_mat = np.array(a, dtype=np.float64, copy=True)
    b_mat = np.array(b, dtype=np.float64, copy=True)

    # Enforce symmetry on A
    a_mat = 0.5 * (a_mat + a_mat.T)

    k, d_dim = d.shape

    for _ in range(max(1, int(n_iter))):
        for j in range(k):
            a_jj = float(a_mat[j, j])
            if a_jj <= eps:
                # Inactive atom: skip BCD step to avoid division by zero
                continue

            # u_j = d_j + (B_j - A_j @ D) / A_jj
            # Equivalent to: u_j = (B_j - sum_{l!=j} A_jl * D_l) / A_jj
            diff = b_mat[j] - a_mat[j] @ d
            u_j = d[j] + diff / a_jj

            if non_negative:
                u_j = np.maximum(u_j, 0.0)

            norm_u = float(np.linalg.norm(u_j))
            if norm_u < eps:
                # Guard zero candidate vector
                if non_negative:
                    u_j = np.ones(d_dim, dtype=np.float64) / np.sqrt(float(d_dim))
                else:
                    u_j = np.random.default_rng(j).standard_normal(d_dim)
                    u_j /= float(np.linalg.norm(u_j))
                d[j] = u_j
            else:
                # Unit ball projection: ||d_j||_2 <= 1.0
                d[j] = u_j / max(1.0, norm_u)

    return d.astype(np.float32)


def revive_dead_atoms(
    d: np.ndarray,
    stats_a: np.ndarray,
    stats_b: np.ndarray,
    counts: np.ndarray,
    dead_indices: np.ndarray,
    donor_indices: np.ndarray | None = None,
    delta: float = 1.0 / 128.0,
    non_negative: bool = False,
    residuals: np.ndarray | None = None,
) -> None:
    """Revive inactive dictionary atoms and reset sufficient statistics to equilibrium.

    For each dead atom:
    1. If residuals are provided, select the sample with highest residual reconstruction error.
    2. Revive atom direction as the normalized residual.
    3. Reset sufficient statistics:
       N_k = mean(N_live), A_kk = mean(A_live), A_kj = 0, B_k = A_kk * d_k.

    Parameters
    ----------
    d : np.ndarray of shape (k, d)
        Dictionary atoms (modified in-place).
    stats_a : np.ndarray of shape (k, k)
        EMA code autocorrelation matrix (modified in-place).
    stats_b : np.ndarray of shape (k, d)
        EMA cross-correlation matrix (modified in-place).
    counts : np.ndarray of shape (k,)
        EMA atom usage counts (modified in-place).
    dead_indices : np.ndarray of shape (n_dead,)
        Indices of dead atoms to revive.
    donor_indices : np.ndarray | None
        Optional donor atom indices for fallback perturbation.
    delta : float
        Perturbation scale for fallback donor splitting.
    non_negative : bool
        If True, clips revived atoms to non-negative orthant.
    residuals : np.ndarray | None
        Mini-batch reconstruction residuals (x_i - alpha_i @ D) of shape (b, d).
    """
    if len(dead_indices) == 0:
        return

    k, d_dim = d.shape
    live_mask = np.ones(k, dtype=bool)
    live_mask[dead_indices] = False

    # Equilibrium statistic targets from live units
    if np.any(live_mask) and np.any(counts[live_mask] > 0):
        mean_n = float(np.mean(counts[live_mask]))
        diag_a = np.diag(stats_a)
        mean_a = float(np.mean(diag_a[live_mask]))
    else:
        mean_n = 1.0
        mean_a = 1.0

    if mean_n <= 0.0:
        mean_n = 1.0
    if mean_a <= 0.0:
        mean_a = 1.0

    # Sort residuals by descending reconstruction error
    has_residuals = residuals is not None and len(residuals) > 0
    if has_residuals:
        res_arr = np.asarray(residuals, dtype=np.float64)
        res_norms = np.linalg.norm(res_arr, axis=1)
        sort_order = np.argsort(-res_norms)
    else:
        res_arr = None
        sort_order = None

    for i, dead_k in enumerate(dead_indices):
        revived = False

        if has_residuals and sort_order is not None and len(sort_order) > 0:
            sample_idx = sort_order[i % len(sort_order)]
            cand = res_arr[sample_idx].copy()
            if non_negative:
                cand = np.maximum(cand, 0.0)
            norm_cand = float(np.linalg.norm(cand))
            if norm_cand > 1e-8:
                d[dead_k] = (cand / norm_cand).astype(d.dtype)
                revived = True

        if not revived:
            # Fallback to donor perturbation or random sphere vector
            if donor_indices is not None and len(donor_indices) > i:
                donor_k = donor_indices[i]
                cand = np.array(d[donor_k], dtype=np.float64) * (1.0 + delta)
            else:
                rng = np.random.default_rng(dead_k)
                cand = rng.standard_normal(d_dim)
            if non_negative:
                cand = np.maximum(cand, 0.0)
            norm_cand = float(np.linalg.norm(cand))
            if norm_cand < 1e-8:
                cand = np.ones(d_dim, dtype=np.float64)
                norm_cand = float(np.linalg.norm(cand))
            d[dead_k] = (cand / norm_cand).astype(d.dtype)

        # Reset sufficient statistics to equilibrium
        counts[dead_k] = mean_n
        stats_a[dead_k, :] = 0.0
        stats_a[:, dead_k] = 0.0
        stats_a[dead_k, dead_k] = mean_a
        stats_b[dead_k, :] = mean_a * d[dead_k]


class OnlineDictionaryLearning:
    """Scikit-learn compatible estimator for Online Dictionary Learning via e-means.

    Parameters
    ----------
    n_components : int
        Number of dictionary atoms (K).
    alpha_reg : float
        L1 sparsity penalty parameter (lambda).
    max_iter : float
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    alpha0 : float | None
        Initial peak EMA step size. If None, uses scaling law heuristic.
    schedule : str
        Schedule kind ("cosine", "step", "constant").
    tau : float
        Dead atom threshold factor (Nk < tau * mean(N)).
    delta : float
        Revival perturbation scale.
    revival_every : int
        Revival check frequency in mini-batch steps.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 64,
        alpha_reg: float = 0.1,
        max_iter: float = 10.0,
        batch_size: int = 128,
        alpha0: float | None = None,
        schedule: str = "cosine",
        tau: float = 0.1,
        delta: float = 1.0 / 128.0,
        revival_every: int = 5,
        random_state: int = 0,
    ) -> None:
        self.n_components = int(n_components)
        self.alpha_reg = float(alpha_reg)
        self.max_iter = float(max_iter)
        self.batch_size = int(batch_size)
        self.alpha0 = float(alpha0) if alpha0 is not None else None
        self.schedule = str(schedule)
        self.tau = float(tau)
        self.delta = float(delta)
        self.revival_every = int(revival_every)
        self.random_state = int(random_state)

        self.components_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.n_iter_: int = 0

    def fit(self, x: np.ndarray) -> OnlineDictionaryLearning:
        """Fit dictionary from training data X using e-means."""
        from plugin import fit_dictlearn_emeans

        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {x_arr.shape}")

        self.n_features_in_ = x_arr.shape[1]
        res = fit_dictlearn_emeans(
            x=x_arr,
            n_components=self.n_components,
            alpha_reg=self.alpha_reg,
            epochs=self.max_iter,
            batch_size=self.batch_size,
            alpha0=self.alpha0,
            schedule=self.schedule,
            tau=self.tau,
            delta=self.delta,
            revival_every=self.revival_every,
            seed=self.random_state,
        )
        self.components_ = np.asarray(res.params, dtype=np.float32)
        self.n_iter_ = res.steps
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Encode data X into sparse code representation alpha."""
        if self.components_ is None:
            raise ValueError("OnlineDictionaryLearning is not fitted yet. Call fit() first.")
        x_arr = np.asarray(x, dtype=np.float32)
        return fista_sparse_code(x_arr, self.components_, self.alpha_reg, non_negative=False)

    def loss(self, x: np.ndarray) -> float:
        """Compute penalized LASSO reconstruction loss: 0.5/n * ||X - alpha @ D||_F^2 + alpha_reg/n * ||alpha||_1."""
        if self.components_ is None:
            raise ValueError("OnlineDictionaryLearning is not fitted yet. Call fit() first.")
        x_arr = np.asarray(x, dtype=np.float32)
        n = x_arr.shape[0]
        if n == 0:
            return 0.0
        alpha = self.transform(x_arr)
        recon_err = 0.5 * float(np.sum((x_arr - alpha @ self.components_) ** 2)) / float(n)
        sparsity = self.alpha_reg * float(np.sum(np.abs(alpha))) / float(n)
        return recon_err + sparsity

    def score(self, x: np.ndarray) -> float:
        """Return opposite of loss (higher is better, per scikit-learn convention)."""
        return -self.loss(x)


class OnlineNMF(OnlineDictionaryLearning):
    """Non-negative Matrix Factorization via e-means with non-negative constraints."""

    def __init__(
        self,
        n_components: int = 64,
        alpha_reg: float = 0.0,
        max_iter: float = 10.0,
        batch_size: int = 128,
        alpha0: float | None = None,
        schedule: str = "cosine",
        tau: float = 0.1,
        delta: float = 1.0 / 128.0,
        revival_every: int = 5,
        random_state: int = 0,
    ) -> None:
        super().__init__(
            n_components=n_components,
            alpha_reg=alpha_reg,
            max_iter=max_iter,
            batch_size=batch_size,
            alpha0=alpha0,
            schedule=schedule,
            tau=tau,
            delta=delta,
            revival_every=revival_every,
            random_state=random_state,
        )

    def fit(self, x: np.ndarray) -> OnlineNMF:
        """Fit NMF dictionary from non-negative data X."""
        from plugin import fit_nmf_emeans

        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {x_arr.shape}")
        if np.any(x_arr < 0):
            raise ValueError("NMF requires non-negative inputs (found negative entries in x)")

        self.n_features_in_ = x_arr.shape[1]
        res = fit_nmf_emeans(
            x=x_arr,
            n_components=self.n_components,
            alpha_reg=self.alpha_reg,
            epochs=self.max_iter,
            batch_size=self.batch_size,
            alpha0=self.alpha0,
            schedule=self.schedule,
            tau=self.tau,
            delta=self.delta,
            revival_every=self.revival_every,
            seed=self.random_state,
        )
        self.components_ = np.asarray(res.params, dtype=np.float32)
        self.n_iter_ = res.steps
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Encode non-negative data X into non-negative sparse codes alpha >= 0."""
        if self.components_ is None:
            raise ValueError("OnlineNMF is not fitted yet. Call fit() first.")
        x_arr = np.asarray(x, dtype=np.float32)
        if np.any(x_arr < 0):
            raise ValueError("NMF requires non-negative inputs (found negative entries in x)")
        return fista_sparse_code(x_arr, self.components_, self.alpha_reg, non_negative=True)
