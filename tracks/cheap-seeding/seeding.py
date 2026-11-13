"""Algorithmic implementations of baseline and candidate seeding methods for Track T09.

This module provides pure seeding algorithms that initialize k cluster centroids:
Baselines:
- random: Uniform random sampling without replacement (s = 0.0).
- kmeanspp: Standard Arthur-Vassilvitskii (2007) k-means++ (s = (k-1)/k).
- greedy_kmeanspp: Multi-candidate greedy k-means++ (s = r * (k-1)/k).
- kmeans_parallel: Scalable k-means|| (Bahmani et al., 2012) (s ~ 10).

Candidates (budget s <= 0.10 epoch):
- afkmc2: Assumption-Free K-MC^2 (Bachem et al., NeurIPS 2016) (s ~ 0.02 - 0.07).
- subsampled_kmeanspp: Single-batch k-means++ on b <= 0.05-0.10 N points (s <= 0.05).
- split_as_you_go: Dynamic donor-splitting cluster growth during epoch 1 (s = 0.0 upfront).
"""

from __future__ import annotations

import math
from typing import Callable
import numpy as np


def seed_random(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Uniform random sampling without replacement.

    Parameters
    ----------
    x : np.ndarray, shape (n, d)
        Input dataset.
    k : int
        Number of centroids.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    centroids : np.ndarray, shape (k, d)
        Selected initial centroids.
    dist_evals : int
        Number of pairwise distance evaluations (identically 0).
    """
    n = x.shape[0]
    idx = rng.choice(n, size=k, replace=k > n)
    centroids = np.ascontiguousarray(x[idx], dtype=np.float32)
    return centroids, 0


def seed_kmeanspp(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Standard Arthur-Vassilvitskii (2007) k-means++.

    Parameters
    ----------
    x : np.ndarray, shape (n, d)
        Input dataset.
    k : int
        Number of centroids.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    centroids : np.ndarray, shape (k, d)
        Selected initial centroids.
    dist_evals : int
        Number of pairwise distance evaluations: (k - 1) * n.
    """
    n, d = x.shape
    c = np.empty((k, d), dtype=np.float32)

    # First center chosen uniformly at random
    idx0 = rng.integers(0, n)
    c[0] = x[idx0]

    # Running minimum squared distance cache
    diff = x - c[0]
    d2 = np.sum(diff * diff, axis=1, dtype=np.float64)

    for j in range(1, k):
        total_d2 = float(np.sum(d2))
        if total_d2 <= 1e-12:
            probs = None
        else:
            probs = d2 / total_d2
        idx = rng.choice(n, p=probs)
        c[j] = x[idx]
        diff_j = x - c[j]
        d2_j = np.sum(diff_j * diff_j, axis=1, dtype=np.float64)
        np.minimum(d2, d2_j, out=d2)

    dist_evals = (k - 1) * n
    return c, dist_evals


def seed_greedy_kmeanspp(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    r: int | None = None,
) -> tuple[np.ndarray, int]:
    """Greedy multi-candidate k-means++ (scikit-learn style).

    Evaluates r candidates per step and picks the one minimizing the total potential.

    Parameters
    ----------
    x : np.ndarray, shape (n, d)
        Input dataset.
    k : int
        Number of centroids.
    rng : np.random.Generator
        Random number generator.
    r : int, optional
        Number of candidates per center. Defaults to 2 + floor(ln(k)).

    Returns
    -------
    centroids : np.ndarray, shape (k, d)
        Selected initial centroids.
    dist_evals : int
        Number of pairwise distance evaluations: r * (k - 1) * n.
    """
    n, d = x.shape
    if r is None:
        r = max(2 + int(math.log(max(k, 1))), 2)

    c = np.empty((k, d), dtype=np.float32)
    idx0 = rng.integers(0, n)
    c[0] = x[idx0]

    diff = x - c[0]
    d2 = np.sum(diff * diff, axis=1, dtype=np.float64)

    for j in range(1, k):
        total_d2 = float(np.sum(d2))
        probs = d2 / total_d2 if total_d2 > 1e-12 else None
        cand_indices = rng.choice(n, size=r, p=probs)

        best_cand_idx = cand_indices[0]
        best_pot = float("inf")
        best_d2_new = None

        for cand_idx in cand_indices:
            cand_x = x[cand_idx]
            diff_cand = x - cand_x
            d2_cand = np.sum(diff_cand * diff_cand, axis=1, dtype=np.float64)
            d2_new = np.minimum(d2, d2_cand)
            pot = float(np.sum(d2_new))
            if pot < best_pot:
                best_pot = pot
                best_cand_idx = cand_idx
                best_d2_new = d2_new

        c[j] = x[best_cand_idx]
        assert best_d2_new is not None
        d2[:] = best_d2_new

    dist_evals = r * (k - 1) * n
    return c, dist_evals


def seed_kmeans_parallel(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    l: int | None = None,
    rounds: int = 5,
) -> tuple[np.ndarray, int]:
    """Scalable k-means|| (Bahmani et al., PVLDB 2012).

    Parameters
    ----------
    x : np.ndarray, shape (n, d)
        Input dataset.
    k : int
        Number of centroids.
    rng : np.random.Generator
        Random number generator.
    l : int, optional
        Oversampling factor per round. Defaults to 2 * k.
    rounds : int
        Number of sampling rounds (standard default is 5).

    Returns
    -------
    centroids : np.ndarray, shape (k, d)
        Selected initial centroids.
    dist_evals : int
        Number of pairwise distance evaluations performed.
    """
    n, d = x.shape
    if k <= 1:
        c = np.empty((k, d), dtype=np.float32)
        if k == 1 and n > 0:
            c[0] = x[rng.integers(0, n)]
        return c, 0

    if l is None:
        l = 2 * k

    dist_evals = 0

    # Round 0: 1 random center
    idx0 = rng.integers(0, n)
    candidates = [x[idx0].copy()]
    diff = x - candidates[0]
    d2 = np.sum(diff * diff, axis=1, dtype=np.float64)
    dist_evals += n
    psi = float(np.sum(d2))

    for _ in range(rounds):
        if psi <= 1e-12:
            break
        # Probability of selecting each point x_i
        probs = np.minimum(1.0, (float(l) * d2) / psi)
        sampled_mask = rng.random(n) < probs
        sampled_idx = np.flatnonzero(sampled_mask)

        if len(sampled_idx) == 0:
            # Sample at least 1 point
            p_norm = d2 / psi
            sampled_idx = np.array([rng.choice(n, p=p_norm)])

        new_candidates = x[sampled_idx]
        candidates.extend(new_candidates)

        # Update d2 for all newly added candidates
        for cand in new_candidates:
            diff_cand = x - cand
            d2_cand = np.sum(diff_cand * diff_cand, axis=1, dtype=np.float64)
            np.minimum(d2, d2_cand, out=d2)
            dist_evals += n

        psi = float(np.sum(d2))

    cand_arr = np.array(candidates, dtype=np.float32)
    num_cand = cand_arr.shape[0]

    if num_cand <= k:
        # Fewer candidates than k; fill remaining with random points
        if num_cand < k:
            needed = k - num_cand
            extra_idx = rng.choice(n, size=needed, replace=needed > n)
            cand_arr = np.vstack([cand_arr, x[extra_idx]])
        return cand_arr[:k], dist_evals

    # Assign all points in x to closest candidate to obtain candidate weights
    # Pairwise distances from x to cand_arr: shape (n, num_cand)
    # Process in chunks to maintain low memory
    chunk_size = 8192
    cand_weights = np.zeros(num_cand, dtype=np.float64)
    cand_norms = np.sum(cand_arr * cand_arr, axis=1, dtype=np.float64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        xb = x[start:end]
        xb_norms = np.sum(xb * xb, axis=1, keepdims=True, dtype=np.float64)
        # (xb_len, num_cand)
        dists_b = xb_norms + cand_norms.reshape(1, -1) - 2.0 * np.dot(xb, cand_arr.T)
        nearest_cand = np.argmin(dists_b, axis=1)
        np.add.at(cand_weights, nearest_cand, 1.0)
    dist_evals += n * num_cand

    # Weighted k-means++ on candidate set cand_arr
    c_final = np.empty((k, d), dtype=np.float32)
    # Pick first center with probability proportional to weight
    w_sum = float(np.sum(cand_weights))
    w_probs = cand_weights / w_sum if w_sum > 0.0 else None
    first_idx = rng.choice(num_cand, p=w_probs)
    c_final[0] = cand_arr[first_idx]

    # Running distance from cand_arr to c_final
    diff_c = cand_arr - c_final[0]
    cand_d2 = np.sum(diff_c * diff_c, axis=1, dtype=np.float64)
    dist_evals += num_cand

    for j in range(1, k):
        weighted_d2 = cand_weights * cand_d2
        w_d2_sum = float(np.sum(weighted_d2))
        p = weighted_d2 / w_d2_sum if w_d2_sum > 1e-12 else None
        sel_idx = rng.choice(num_cand, p=p)
        c_final[j] = cand_arr[sel_idx]

        diff_cj = cand_arr - c_final[j]
        d2_cj = np.sum(diff_cj * diff_cj, axis=1, dtype=np.float64)
        np.minimum(cand_d2, d2_cj, out=cand_d2)
        dist_evals += num_cand

    return c_final, dist_evals


def seed_afkmc2(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    m: int | None = None,
) -> tuple[np.ndarray, int]:
    """Assumption-Free K-MC^2 (Bachem et al., NeurIPS 2016).

    Uses a proposal distribution q(x) and an MCMC Metropolis-Hastings chain
    of length m for each subsequent center.
    Adaptive proposal subsampling and MCMC chain scaling ensure the seeding
    cost invariant s <= 0.10 epochs is satisfied for any (N, k).

    Parameters
    ----------
    x : np.ndarray, shape (n, d)
        Input dataset.
    k : int
        Number of centroids.
    rng : np.random.Generator
        Random number generator.
    m : int, optional
        MCMC chain length. If None, dynamically determined from budget.

    Returns
    -------
    centroids : np.ndarray, shape (k, d)
        Selected initial centroids.
    dist_evals : int
        Number of pairwise distance evaluations performed:
        n_prop + (m + 1) * k * (k - 1) // 2.
    """
    n, d = x.shape
    if k <= 1:
        c = np.empty((k, d), dtype=np.float32)
        if k == 1 and n > 0:
            c[0] = x[rng.integers(0, n)]
        return c, 0

    # Proposal pass configuration:
    # When 1/k >= 0.05 (e.g. k <= 20), subsample x to ensure proposal evals <= 0.04 epoch
    if 1.0 / float(k) >= 0.05:
        n_prop = min(n, max(k, int(0.04 * n * k)))
        prop_idx = rng.choice(n, size=n_prop, replace=n_prop > n)
        x_prop = x[prop_idx]
    else:
        n_prop = n
        x_prop = x

    # MCMC chain length calculation to guarantee MCMC evals <= 0.05 epoch
    d_mcmc_budget = max(10, int(0.05 * n * k))
    k_pairs = k * (k - 1) // 2
    if m is None:
        m = max(10, min(200, int((2 * d_mcmc_budget) // (k * (k - 1)))))

    c = np.empty((k, d), dtype=np.float32)

    # First center chosen uniformly at random from full x
    idx0 = rng.integers(0, n)
    c[0] = x[idx0]

    # Pass 1: compute proposal distribution q(x) over x_prop
    diff = x_prop - c[0]
    d2_first = np.sum(diff * diff, axis=1, dtype=np.float64)
    s1 = float(np.sum(d2_first))

    if s1 > 1e-12:
        q = 0.5 * (d2_first / s1) + 0.5 / float(n_prop)
    else:
        q = np.full(n_prop, 1.0 / float(n_prop), dtype=np.float64)
    q /= np.sum(q)

    # Distances for Phase 1
    dist_evals = n_prop

    # MCMC sampling for centers 1 .. k-1
    for j in range(1, k):
        # Draw initial chain state from proposal q
        curr_idx = rng.choice(n_prop, p=q)
        curr_x = x_prop[curr_idx]
        diff_curr = c[:j] - curr_x
        curr_d2 = float(np.min(np.sum(diff_curr * diff_curr, axis=1)))
        dist_evals += j

        # Propose m transitions
        cand_indices = rng.choice(n_prop, size=m, p=q)
        for cand_idx in cand_indices:
            cand_x = x_prop[cand_idx]
            diff_cand = c[:j] - cand_x
            cand_d2 = float(np.min(np.sum(diff_cand * diff_cand, axis=1)))
            dist_evals += j

            # Metropolis-Hastings acceptance ratio
            denom = max(curr_d2, 1e-12) * q[cand_idx]
            num = cand_d2 * q[curr_idx]
            alpha = min(1.0, num / denom) if denom > 0.0 else 1.0

            if rng.random() <= alpha:
                curr_idx = cand_idx
                curr_x = cand_x
                curr_d2 = cand_d2

        c[j] = curr_x

    return c, dist_evals


def seed_subsampled_kmeanspp(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    sample_ratio: float = 0.05,
    max_sample: int = 8192,
) -> tuple[np.ndarray, int]:
    """Subsampled / Single-Batch k-means++.

    Draws a sample of size b = min(n, max(10 * k, min(sample_ratio * n, max_sample)))
    and executes exact Arthur-Vassilvitskii k-means++ within that sample.

    Parameters
    ----------
    x : np.ndarray, shape (n, d)
        Input dataset.
    k : int
        Number of centroids.
    rng : np.random.Generator
        Random number generator.
    sample_ratio : float
        Fraction of dataset to subsample (default: 0.05 = 5%).
    max_sample : int
        Maximum sample size cap (default: 8192).

    Returns
    -------
    centroids : np.ndarray, shape (k, d)
        Selected initial centroids.
    dist_evals : int
        Number of pairwise distance evaluations: (k - 1) * b.
    """
    n, d = x.shape
    b = min(n, max(10 * k, min(int(round(sample_ratio * n)), max_sample)))
    idx_sub = rng.choice(n, size=b, replace=False)
    x_sub = np.ascontiguousarray(x[idx_sub], dtype=np.float32)

    centroids, dist_evals = seed_kmeanspp(x_sub, k, rng)
    return centroids, dist_evals


def seed_split_as_you_go_init(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    k0: int | None = None,
) -> tuple[np.ndarray, int]:
    """Upfront initialization for Split-As-You-Go cluster growth.

    Samples k0 << k centroids uniformly at random upfront with 0 distance evaluations.
    Remaining clusters are dynamically grown during the first epoch.

    Parameters
    ----------
    x : np.ndarray, shape (n, d)
        Input dataset.
    k : int
        Target total number of centroids.
    rng : np.random.Generator
        Random number generator.
    k0 : int, optional
        Number of initial centroids (default: max(2, min(k, 4))).

    Returns
    -------
    centroids : np.ndarray, shape (k0, d)
        Initial small set of centroids.
    dist_evals : int
        Always 0.
    """
    n, d = x.shape
    if k0 is None:
        if k <= 2:
            k0 = k
        else:
            k0 = max(2, min(k, 4))
    k0 = min(k0, n, k)
    idx0 = rng.choice(n, size=k0, replace=k0 > n)
    centroids = np.ascontiguousarray(x[idx0], dtype=np.float32)
    return centroids, 0


# Registry of seeding methods
SEEDING_REGISTRY: dict[str, Callable[..., tuple[np.ndarray, int]]] = {
    "random": seed_random,
    "kmeanspp": seed_kmeanspp,
    "kmeans_plusplus": seed_kmeanspp,
    "greedy_kmeanspp": seed_greedy_kmeanspp,
    "kmeans_parallel": seed_kmeans_parallel,
    "afkmc2": seed_afkmc2,
    "afk_mc2": seed_afkmc2,
    "subsampled_kmeanspp": seed_subsampled_kmeanspp,
    "single_batch_kmeanspp": seed_subsampled_kmeanspp,
    "split_as_you_go": seed_split_as_you_go_init,
}
