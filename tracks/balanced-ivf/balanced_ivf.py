r"""Balanced Inverted-File Indexing (Balanced IVF) via Mini-Batch Optimal Transport.

Implements:
- Vectorized pairwise squared Euclidean distance matrix computation.
- Numerically stabilized mini-batch Sinkhorn-Knopp optimal transport solver:
  * Entropic regularization parameter $\epsilon$.
  * Uniform marginal targets $r = \frac{1}{B} \mathbf{1}_B$ and $c = \frac{1}{K} \mathbf{1}_K$.
  * Log-space normalization with logsumexp to avoid underflow/overflow.
  * Dual variable interpretability (cluster capacity penalty $-\epsilon \beta_j$).
  * Hard discretization $\hat{y}_i = \arg\max_j P_{ij}$.
- Coarse quantizer centroid EMA updates via `EMAStats` and `CosineSchedule`.
- Dead-unit detection and donor-split revival (`core.revival`).
- IVF index construction, posting list storage, and list length imbalance factor:
  $I = \frac{K}{N^2} \sum_{j=1}^K |L_j|^2 \quad (I = 1.0 \text{ is perfect balance})$.
- Single-threaded query routing and latency profiling (Recall@10, p50, p95, p99 tail latency).
- Baselines:
  * Unbalanced e-means IVF (`EMeansIVFIndex`).
  * Faiss `IndexIVFFlat` reference (`FaissIVFBenchmark`).
  * Size-penalized balanced k-means (`BalancedKMeansIVFIndex`).
- Synthetic skewed mixture dataset generator for controlled benchmark studies.
"""

from __future__ import annotations

import math
import time
from typing import Any, Sequence
import numpy as np
import scipy.special

from core.revival import revive_dead_units
from core.schedule import CosineSchedule, StepSchedule
from core.stats import BatchStats, EMAStats


def pairwise_sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances between X and C.

    Parameters
    ----------
    X : np.ndarray
        Sample matrix of shape (B, d).
    C : np.ndarray
        Centroid matrix of shape (K, d).

    Returns
    -------
    np.ndarray
        Distance matrix M of shape (B, K) where M[i, j] = ||X[i] - C[j]||^2 >= 0.
    """
    X_f32 = np.ascontiguousarray(X, dtype=np.float32)
    C_f32 = np.ascontiguousarray(C, dtype=np.float32)

    x_norms_sq = np.sum(X_f32**2, axis=1, keepdims=True)  # (B, 1)
    c_norms_sq = np.sum(C_f32**2, axis=1, keepdims=True).T  # (1, K)
    dots = X_f32 @ C_f32.T  # (B, K)

    M = x_norms_sq + c_norms_sq - 2.0 * dots
    np.maximum(M, 0.0, out=M)
    return M


def sinkhorn_knopp_transport(
    M: np.ndarray,
    epsilon: float = 0.05,
    num_iters: int = 3,
    r: np.ndarray | None = None,
    c: np.ndarray | None = None,
    slack: float = 0.0,
    log_space: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Solve regularized optimal transport via mini-batch Sinkhorn-Knopp iterations.

    Solves:
        \min_{P \in U(r, c)} \langle P, M \rangle - \epsilon H(P)
    where:
        r = (1/B) 1_B,  c = (1/K) 1_K.

    In log-space:
        cost = -M / \epsilon
        \alpha \leftarrow \log r - \mathrm{logsumexp}(\mathrm{cost} + \beta, \mathrm{axis}=1)
        \beta \leftarrow \log c - \mathrm{logsumexp}(\mathrm{cost} + \alpha, \mathrm{axis}=0)
        \log P = \alpha + \mathrm{cost} + \beta

    Parameters
    ----------
    M : np.ndarray
        Distance matrix of shape (B, K).
    epsilon : float
        Entropic regularization coefficient > 0.
    num_iters : int
        Number of Sinkhorn scaling iterations (typically 1, 3, or 10).
    r : np.ndarray | None
        Target marginals for samples (shape (B,)), default uniform 1/B.
    c : np.ndarray | None
        Target marginals for clusters (shape (K,)), default uniform 1/K.
    slack : float
        Allowed fractional slack on marginals (default 0.0).
    log_space : bool
        Whether to perform iterations in log-space (unconditionally stable).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (P, alpha_or_u, beta_or_v) where P has shape (B, K).
    """
    B, K = M.shape
    if B == 0 or K == 0:
        return np.empty((B, K), dtype=np.float32), np.empty(B), np.empty(K)

    if K == 1:
        # Trivial single-cluster assignment
        P = np.full((B, 1), 1.0 / float(B), dtype=np.float32)
        return P, np.zeros(B, dtype=np.float32), np.zeros(1, dtype=np.float32)

    eps = max(float(epsilon), 1e-6)
    iters = max(int(num_iters), 1)

    r_target = (
        np.full(B, 1.0 / float(B), dtype=np.float64)
        if r is None
        else np.asarray(r, dtype=np.float64)
    )
    c_target = (
        np.full(K, 1.0 / float(K), dtype=np.float64)
        if c is None
        else np.asarray(c, dtype=np.float64)
    )

    if log_space:
        log_r = np.log(np.maximum(r_target, 1e-15))
        log_c = np.log(np.maximum(c_target, 1e-15))
        cost = -M.astype(np.float64) / eps

        beta = np.zeros(K, dtype=np.float64)
        alpha = np.zeros(B, dtype=np.float64)

        for _ in range(iters):
            # Row normalization: P 1_K = r
            alpha = log_r - scipy.special.logsumexp(cost + beta[None, :], axis=1)
            # Column normalization: P^T 1_B = c
            beta = log_c - scipy.special.logsumexp(cost + alpha[:, None], axis=0)

        log_P = alpha[:, None] + cost + beta[None, :]
        P = np.exp(log_P).astype(np.float32)

        # Ensure exact row marginals
        row_sums = np.sum(P, axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
        P = (P / row_sums) * (1.0 / float(B))

        return P, alpha.astype(np.float32), beta.astype(np.float32)

    else:
        # Stabilized multiplicative Sinkhorn scaling
        M_shift = M.astype(np.float64) - np.min(M, axis=1, keepdims=True)
        K_mat = np.exp(-M_shift / eps)

        v = np.ones(K, dtype=np.float64) / float(K)
        u = np.ones(B, dtype=np.float64) / float(B)

        for _ in range(iters):
            kv = K_mat @ v
            np.maximum(kv, 1e-15, out=kv)
            u = r_target / kv

            ktu = K_mat.T @ u
            np.maximum(ktu, 1e-15, out=ktu)
            v = c_target / ktu

        P = (u[:, None] * K_mat * v[None, :]).astype(np.float32)
        row_sums = np.sum(P, axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
        P = (P / row_sums) * (1.0 / float(B))

        return P, u.astype(np.float32), v.astype(np.float32)


def compute_imbalance_factor(
    list_lengths: Sequence[int] | np.ndarray,
    num_vectors: int | None = None,
    num_clusters: int | None = None,
) -> float:
    r"""Compute posting list length imbalance factor:

    .. math::
        I = \frac{K}{N^2} \sum_{j=1}^K |L_j|^2

    Properties:
    - I = 1.0 indicates perfect balance (|L_j| = N/K for all j).
    - I > 1.0 indicates imbalance (I = K in the worst-case single giant list).

    Parameters
    ----------
    list_lengths : Sequence[int] | np.ndarray
        Array or sequence of list lengths |L_j| for each posting list j.
    num_vectors : int | None
        Total number of vectors N (inferred from sum(list_lengths) if None).
    num_clusters : int | None
        Total number of clusters K (inferred from len(list_lengths) if None).

    Returns
    -------
    float
        Imbalance factor I >= 1.0.
    """
    lengths = np.asarray(list_lengths, dtype=np.float64)
    K = len(lengths) if num_clusters is None else int(num_clusters)
    N = float(np.sum(lengths)) if num_vectors is None else float(num_vectors)

    if K <= 1 or N <= 0.0:
        return 1.0

    sum_sq = float(np.sum(lengths**2))
    factor = (float(K) / (N * N)) * sum_sq
    return float(max(factor, 1.0))


class SinkhornBalancedAssigner:
    """Mini-batch balanced assignment using Sinkhorn-Knopp optimal transport."""

    def __init__(
        self,
        epsilon: float = 0.05,
        num_iters: int = 3,
        slack: float = 0.0,
        mode: str = "always",
        early_ratio: float = 0.5,
    ) -> None:
        self.epsilon = float(epsilon)
        self.num_iters = int(num_iters)
        self.slack = float(slack)
        self.mode = str(mode)
        self.early_ratio = float(early_ratio)

    def assign(
        self,
        batch: np.ndarray,
        centroids: np.ndarray,
        step: int = 0,
        total_steps: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Assign batch vectors to centroids, returning (hard_labels, soft_P, diagnostics)."""
        B = batch.shape[0]
        K = centroids.shape[0]

        if K == 1:
            return np.zeros(B, dtype=np.int32), np.ones((B, 1), dtype=np.float32), {}

        # Distance matrix
        M = pairwise_sq_dists(batch, centroids)

        use_sinkhorn = True
        if self.mode == "early" and total_steps > 0:
            if float(step) / float(total_steps) > self.early_ratio:
                use_sinkhorn = False

        if use_sinkhorn:
            P, alpha, beta = sinkhorn_knopp_transport(
                M,
                epsilon=self.epsilon,
                num_iters=self.num_iters,
                slack=self.slack,
                log_space=True,
            )
            # Hard discretization argmax P_{ij}
            labels = np.argmax(P, axis=1).astype(np.int32)

            # Diagnostics
            row_err = float(np.max(np.abs(np.sum(P, axis=1) - 1.0 / B)))
            col_err = float(np.max(np.abs(np.sum(P, axis=0) - 1.0 / K)))
            counts = np.bincount(labels, minlength=K)
            batch_imbalance = compute_imbalance_factor(counts, num_vectors=B, num_clusters=K)

            diag = {
                "row_marginal_err": row_err,
                "col_marginal_err": col_err,
                "batch_imbalance": batch_imbalance,
                "beta_min": float(np.min(beta)),
                "beta_max": float(np.max(beta)),
            }
            return labels, P, diag
        else:
            # Fallback to standard nearest centroid
            labels = np.argmin(M, axis=1).astype(np.int32)
            counts = np.bincount(labels, minlength=K)
            batch_imbalance = compute_imbalance_factor(counts, num_vectors=B, num_clusters=K)
            return labels, np.empty((0, 0), dtype=np.float32), {"batch_imbalance": batch_imbalance}


def generate_skewed_data(
    n_train: int = 20_000,
    n_test: int = 2_000,
    d: int = 32,
    k_clusters: int = 16,
    skew_factor: float = 2.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Generate synthetic Gaussian mixture data with power-law cluster sizes.

    Cluster mixture weights follow a Zipf distribution:
        \pi_k \propto (k + 1)^{-\text{skew\_factor}}
    This creates an inherently skewed data distribution where standard Lloyd k-means
    produces unbalanced posting lists, highlighting the advantage of Balanced IVF.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (x_train, x_test, queries, groundtruth_top10)
    """
    rng = np.random.default_rng(seed)

    # Component weights
    weights = (np.arange(k_clusters) + 1.0) ** (-float(skew_factor))
    weights /= np.sum(weights)

    # Cluster means separated in d-dimensional space
    means = rng.standard_normal((k_clusters, d)).astype(np.float32)
    means *= 3.0 / np.sqrt(d)

    # Cluster scales
    scales = rng.uniform(0.3, 0.8, size=k_clusters).astype(np.float32)

    def sample_mixture(n_samples: int) -> np.ndarray:
        c_choices = rng.choice(k_clusters, size=n_samples, p=weights)
        counts = np.bincount(c_choices, minlength=k_clusters)
        samples = np.empty((n_samples, d), dtype=np.float32)
        idx = 0
        for k in range(k_clusters):
            cnt = counts[k]
            if cnt > 0:
                samples[idx : idx + cnt] = (
                    means[k] + scales[k] * rng.standard_normal((cnt, d)).astype(np.float32)
                )
                idx += cnt
        rng.shuffle(samples)
        return samples

    x_train = sample_mixture(n_train)
    x_test = sample_mixture(n_test)
    queries = sample_mixture(min(n_test, 1_000))

    # Compute exact ground truth top-10 on x_train database
    groundtruth = np.zeros((queries.shape[0], 10), dtype=np.int32)
    q_norms = np.sum(queries**2, axis=1, keepdims=True)
    db_norms = np.sum(x_train**2, axis=1)

    batch_size = 200
    for q_start in range(0, queries.shape[0], batch_size):
        q_end = min(q_start + batch_size, queries.shape[0])
        dots = queries[q_start:q_end] @ x_train.T
        dists = q_norms[q_start:q_end] + db_norms[None, :] - 2.0 * dots
        groundtruth[q_start:q_end] = np.argsort(dists, axis=1)[:, :10]

    return x_train, x_test, queries, groundtruth


class BalancedIVFIndex:
    """Balanced Inverted-File Index using mini-batch Sinkhorn-Knopp optimal transport.

    Maintains K coarse centroids trained with uniform marginal regularized transport,
    stores posting lists of database vector IDs and vector embeddings,
    and supports single-threaded multi-probe nearest neighbor queries.
    """

    def __init__(
        self,
        d: int,
        nlist: int,
        epsilon: float = 0.05,
        num_iters: int = 3,
        slack: float = 0.0,
        tau: float = 0.1,
        delta: float = 1.0 / 128.0,
        seed: int = 42,
    ) -> None:
        self.d = int(d)
        self.nlist = int(nlist)
        self.epsilon = float(epsilon)
        self.num_iters = int(num_iters)
        self.slack = float(slack)
        self.tau = float(tau)
        self.delta = float(delta)
        self.seed = int(seed)

        self.centroids: np.ndarray = np.zeros((self.nlist, self.d), dtype=np.float32)
        self.stats: EMAStats | None = None
        self.assigner = SinkhornBalancedAssigner(
            epsilon=self.epsilon,
            num_iters=self.num_iters,
            slack=self.slack,
        )

        # Inverted index posting lists
        self.posting_lists: list[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(self.nlist)]
        self.posting_vectors: list[np.ndarray] = [np.empty((0, self.d), dtype=np.float32) for _ in range(self.nlist)]
        self.database_size: int = 0
        self._is_fitted: bool = False
        self._is_built: bool = False

    def init_centroids(self, x: np.ndarray, rng: np.random.Generator | None = None) -> None:
        """Initialize coarse centroids from a random subset or k-means++ of x."""
        n = x.shape[0]
        rng = rng if rng is not None else np.random.default_rng(self.seed)

        if n >= self.nlist:
            chosen = rng.choice(n, size=self.nlist, replace=False)
            self.centroids = np.ascontiguousarray(x[chosen], dtype=np.float32)
        else:
            chosen = rng.choice(n, size=self.nlist, replace=True)
            self.centroids = np.ascontiguousarray(x[chosen], dtype=np.float32)

        self.stats = EMAStats.from_params(self.centroids)
        self._is_fitted = True

    def fit(
        self,
        x: np.ndarray,
        epochs: float = 1.0,
        batch_size: int = 1024,
        alpha0: float = 0.1,
        revival_every: int = 5,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Train coarse centroids using mini-batch Sinkhorn-Knopp balanced assignment."""
        rng_seed = self.seed if seed is None else int(seed)
        rng = np.random.default_rng(rng_seed)
        n = x.shape[0]
        x_f32 = np.ascontiguousarray(x, dtype=np.float32)

        if not self._is_fitted or self.stats is None:
            self.init_centroids(x_f32, rng=rng)

        b_size = min(max(int(batch_size), 1), n)
        steps_per_epoch = max(n // b_size, 1)
        total_steps = max(int(round(epochs * steps_per_epoch)), 1)
        schedule = CosineSchedule(alpha0=alpha0, total_steps=total_steps, eps=0.1)

        total_revivals = 0
        step = 0
        while step < total_steps:
            perm = rng.permutation(n)
            for start in range(0, steps_per_epoch * b_size, b_size):
                if step >= total_steps:
                    break

                batch = x_f32[perm[start : start + b_size]]
                current_c = self.stats.means()

                labels, _, _ = self.assigner.assign(
                    batch,
                    current_c,
                    step=step,
                    total_steps=total_steps,
                )

                # Batch statistics
                b_counts = np.bincount(labels, minlength=self.nlist).astype(np.float64)
                b_sums = np.zeros((self.nlist, self.d), dtype=np.float64)
                np.add.at(b_sums, labels, batch.astype(np.float64))

                batch_stats = BatchStats(counts=b_counts, sums=b_sums)
                alpha = schedule(step)
                self.stats.update(batch_stats, alpha)

                # Dead unit revival
                if revival_every > 0 and step % revival_every == 0:
                    event = revive_dead_units(
                        self.stats,
                        tau=self.tau,
                        delta=self.delta,
                        rng=rng,
                    )
                    total_revivals += len(event.dead_indices)

                step += 1

        self.centroids = self.stats.means()
        self._is_fitted = True
        return {"steps": step, "revivals": total_revivals}

    def build(
        self,
        x: np.ndarray,
        assign_mode: str = "nearest",
        batch_size: int = 4096,
    ) -> None:
        """Populate inverted posting lists for database vectors x.

        Parameters
        ----------
        x : np.ndarray
            Database vectors of shape (N, d).
        assign_mode : str
            Assignment strategy: "nearest" (standard Voronoi) or "balanced" (mini-batch Sinkhorn).
        batch_size : int
            Chunk size for processing assignments.
        """
        assert self._is_fitted, "Centroids must be trained before building index."
        N = x.shape[0]
        x_f32 = np.ascontiguousarray(x, dtype=np.float32)
        self.database_size = N

        assigned_clusters = np.empty(N, dtype=np.int32)
        b_size = min(max(int(batch_size), 1), N)

        for start in range(0, N, b_size):
            end = min(start + b_size, N)
            batch = x_f32[start:end]

            if assign_mode == "balanced":
                labels, _, _ = self.assigner.assign(batch, self.centroids)
            else:
                # Nearest centroid Voronoi assignment
                M = pairwise_sq_dists(batch, self.centroids)
                labels = np.argmin(M, axis=1).astype(np.int32)

            assigned_clusters[start:end] = labels

        # Group database indices and vectors into posting lists
        self.posting_lists = [np.empty(0, dtype=np.int32) for _ in range(self.nlist)]
        self.posting_vectors = [np.empty((0, self.d), dtype=np.float32) for _ in range(self.nlist)]

        for k in range(self.nlist):
            members = np.flatnonzero(assigned_clusters == k).astype(np.int32)
            self.posting_lists[k] = members
            self.posting_vectors[k] = x_f32[members]

        self._is_built = True

    def imbalance_factor(self) -> float:
        """Compute imbalance factor I = (K / N^2) * sum(|L_j|^2)."""
        lengths = [len(l) for l in self.posting_lists]
        return compute_imbalance_factor(lengths, num_vectors=self.database_size, num_clusters=self.nlist)

    def list_stats(self) -> dict[str, float]:
        """Compute detailed posting list length distribution statistics."""
        lengths = np.array([len(l) for l in self.posting_lists], dtype=np.float64)
        mean_len = float(np.mean(lengths)) if len(lengths) > 0 else 0.0
        max_len = float(np.max(lengths)) if len(lengths) > 0 else 0.0
        min_len = float(np.min(lengths)) if len(lengths) > 0 else 0.0
        std_len = float(np.std(lengths)) if len(lengths) > 0 else 0.0
        p50 = float(np.percentile(lengths, 50)) if len(lengths) > 0 else 0.0
        p95 = float(np.percentile(lengths, 95)) if len(lengths) > 0 else 0.0
        p99 = float(np.percentile(lengths, 99)) if len(lengths) > 0 else 0.0

        return {
            "imbalance_factor": self.imbalance_factor(),
            "mean_length": mean_len,
            "min_length": min_len,
            "max_length": max_len,
            "std_length": std_len,
            "max_to_mean_ratio": max_len / max(mean_len, 1e-12),
            "p50_length": p50,
            "p95_length": p95,
            "p99_length": p99,
        }

    def search_single_query(
        self,
        query: np.ndarray,
        nprobe: int = 8,
        top_k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform exact IVF search for a single query vector q.

        Parameters
        ----------
        query : np.ndarray
            Single query vector of shape (d,).
        nprobe : int
            Number of coarse centroid lists to scan.
        top_k : int
            Number of nearest neighbors to retrieve.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (top_k_distances, top_k_indices)
        """
        q = np.ascontiguousarray(query.reshape(1, self.d), dtype=np.float32)
        nprobe_capped = min(max(int(nprobe), 1), self.nlist)

        # Distance to centroids
        c_dists = pairwise_sq_dists(q, self.centroids)[0]  # (K,)
        probed_clusters = np.argsort(c_dists)[:nprobe_capped]

        # Gather candidates from probed posting lists
        cand_ids_list: list[np.ndarray] = []
        cand_vecs_list: list[np.ndarray] = []
        for c_idx in probed_clusters:
            cand_ids_list.append(self.posting_lists[c_idx])
            cand_vecs_list.append(self.posting_vectors[c_idx])

        if cand_ids_list and sum(len(a) for a in cand_ids_list) > 0:
            cand_ids = np.concatenate(cand_ids_list)
            cand_vecs = np.concatenate(cand_vecs_list, axis=0)

            # Compute exact Euclidean distances
            q_sq = np.sum(q**2)
            c_sq = np.sum(cand_vecs**2, axis=1)
            dots = cand_vecs @ q.T
            cand_dists = q_sq + c_sq - 2.0 * dots.ravel()
            np.maximum(cand_dists, 0.0, out=cand_dists)

            k_ret = min(int(top_k), len(cand_dists))
            top_order = np.argpartition(cand_dists, k_ret - 1)[:k_ret]
            top_order = top_order[np.argsort(cand_dists[top_order])]

            out_dists = cand_dists[top_order]
            out_ids = cand_ids[top_order]

            if len(out_ids) < top_k:
                pad = top_k - len(out_ids)
                out_ids = np.pad(out_ids, (0, pad), constant_values=-1)
                out_dists = np.pad(out_dists, (0, pad), constant_values=np.inf)

            return out_dists.astype(np.float32), out_ids.astype(np.int32)
        else:
            return np.full(top_k, np.inf, dtype=np.float32), np.full(top_k, -1, dtype=np.int32)

    def evaluate_queries(
        self,
        queries: np.ndarray,
        groundtruth: np.ndarray,
        nprobe_list: Sequence[int] = (1, 2, 4, 8, 16, 32),
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Evaluate single-threaded query routing over a suite of nprobe values.

        Measures Recall@top_k, p50 latency, p95 latency, and p99 tail latency.
        """
        assert self._is_built, "Index must be built with database vectors."
        num_q = queries.shape[0]
        results: list[dict[str, Any]] = []

        for nprobe in nprobe_list:
            latencies_ms = np.empty(num_q, dtype=np.float64)
            retrieved_ids = np.empty((num_q, top_k), dtype=np.int32)

            for i in range(num_q):
                t0 = time.perf_counter()
                _, top_ids = self.search_single_query(queries[i], nprobe=nprobe, top_k=top_k)
                t1 = time.perf_counter()
                latencies_ms[i] = (t1 - t0) * 1000.0
                retrieved_ids[i] = top_ids

            # Compute Recall@top_k
            recalls = np.zeros(num_q, dtype=np.float64)
            for i in range(num_q):
                gt_set = set(groundtruth[i, :top_k])
                ret_set = set(retrieved_ids[i])
                recalls[i] = len(gt_set.intersection(ret_set)) / float(top_k)

            mean_recall = float(np.mean(recalls))
            p50_lat = float(np.percentile(latencies_ms, 50))
            p95_lat = float(np.percentile(latencies_ms, 95))
            p99_lat = float(np.percentile(latencies_ms, 99))
            mean_lat = float(np.mean(latencies_ms))
            qps = float(1000.0 / mean_lat) if mean_lat > 0.0 else 0.0

            results.append({
                "nprobe": int(nprobe),
                "recall_at_10": mean_recall,
                "mean_latency_ms": mean_lat,
                "p50_latency_ms": p50_lat,
                "p95_latency_ms": p95_lat,
                "p99_latency_ms": p99_lat,
                "qps": qps,
            })

        return results


class EMeansIVFIndex(BalancedIVFIndex):
    """Standard unbalanced e-means IVF index (nearest-centroid assignment baseline)."""

    def fit(
        self,
        x: np.ndarray,
        epochs: float = 1.0,
        batch_size: int = 1024,
        alpha0: float = 0.1,
        revival_every: int = 5,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Fit centroids via standard nearest-centroid e-means (no Sinkhorn balancing)."""
        rng_seed = self.seed if seed is None else int(seed)
        rng = np.random.default_rng(rng_seed)
        n = x.shape[0]
        x_f32 = np.ascontiguousarray(x, dtype=np.float32)

        if not self._is_fitted or self.stats is None:
            self.init_centroids(x_f32, rng=rng)

        b_size = min(max(int(batch_size), 1), n)
        steps_per_epoch = max(n // b_size, 1)
        total_steps = max(int(round(epochs * steps_per_epoch)), 1)
        schedule = CosineSchedule(alpha0=alpha0, total_steps=total_steps, eps=0.1)

        total_revivals = 0
        step = 0
        while step < total_steps:
            perm = rng.permutation(n)
            for start in range(0, steps_per_epoch * b_size, b_size):
                if step >= total_steps:
                    break

                batch = x_f32[perm[start : start + b_size]]
                current_c = self.stats.means()

                M = pairwise_sq_dists(batch, current_c)
                labels = np.argmin(M, axis=1).astype(np.int32)

                b_counts = np.bincount(labels, minlength=self.nlist).astype(np.float64)
                b_sums = np.zeros((self.nlist, self.d), dtype=np.float64)
                np.add.at(b_sums, labels, batch.astype(np.float64))

                batch_stats = BatchStats(counts=b_counts, sums=b_sums)
                alpha = schedule(step)
                self.stats.update(batch_stats, alpha)

                if revival_every > 0 and step % revival_every == 0:
                    event = revive_dead_units(
                        self.stats,
                        tau=self.tau,
                        delta=self.delta,
                        rng=rng,
                    )
                    total_revivals += len(event.dead_indices)

                step += 1

        self.centroids = self.stats.means()
        self._is_fitted = True
        return {"steps": step, "revivals": total_revivals}


class BalancedKMeansIVFIndex(BalancedIVFIndex):
    """Size-penalized balanced k-means IVF index (Malinen & Fränti heuristic baseline)."""

    def __init__(
        self,
        d: int,
        nlist: int,
        penalty_lambda: float = 0.01,
        tau: float = 0.1,
        delta: float = 1.0 / 128.0,
        seed: int = 42,
    ) -> None:
        super().__init__(d=d, nlist=nlist, tau=tau, delta=delta, seed=seed)
        self.penalty_lambda = float(penalty_lambda)

    def fit(
        self,
        x: np.ndarray,
        epochs: float = 1.0,
        batch_size: int = 1024,
        alpha0: float = 0.1,
        revival_every: int = 5,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Fit centroids via size-penalized k-means."""
        rng_seed = self.seed if seed is None else int(seed)
        rng = np.random.default_rng(rng_seed)
        n = x.shape[0]
        x_f32 = np.ascontiguousarray(x, dtype=np.float32)

        if not self._is_fitted or self.stats is None:
            self.init_centroids(x_f32, rng=rng)

        b_size = min(max(int(batch_size), 1), n)
        steps_per_epoch = max(n // b_size, 1)
        total_steps = max(int(round(epochs * steps_per_epoch)), 1)
        schedule = CosineSchedule(alpha0=alpha0, total_steps=total_steps, eps=0.1)

        total_revivals = 0
        step = 0
        while step < total_steps:
            perm = rng.permutation(n)
            for start in range(0, steps_per_epoch * b_size, b_size):
                if step >= total_steps:
                    break

                batch = x_f32[perm[start : start + b_size]]
                current_c = self.stats.means()
                counts_norm = self.stats.counts / max(float(np.mean(self.stats.counts)), 1e-12)

                M = pairwise_sq_dists(batch, current_c)
                # Size penalty: dist + lambda * (count / mean_count)
                penalized_cost = M + self.penalty_lambda * counts_norm[None, :]
                labels = np.argmin(penalized_cost, axis=1).astype(np.int32)

                b_counts = np.bincount(labels, minlength=self.nlist).astype(np.float64)
                b_sums = np.zeros((self.nlist, self.d), dtype=np.float64)
                np.add.at(b_sums, labels, batch.astype(np.float64))

                batch_stats = BatchStats(counts=b_counts, sums=b_sums)
                alpha = schedule(step)
                self.stats.update(batch_stats, alpha)

                if revival_every > 0 and step % revival_every == 0:
                    event = revive_dead_units(
                        self.stats,
                        tau=self.tau,
                        delta=self.delta,
                        rng=rng,
                    )
                    total_revivals += len(event.dead_indices)

                step += 1

        self.centroids = self.stats.means()
        self._is_fitted = True
        return {"steps": step, "revivals": total_revivals}


class FaissIVFBenchmark:
    """Faiss IndexIVFFlat reference benchmark wrapper."""

    def __init__(self, d: int, nlist: int, seed: int = 42) -> None:
        self.d = int(d)
        self.nlist = int(nlist)
        self.seed = int(seed)
        self.index: Any = None
        self.database_size: int = 0

    def fit_and_build(self, x_train: np.ndarray, x_database: np.ndarray) -> None:
        """Train Faiss IndexIVFFlat on x_train and add x_database."""
        import faiss

        quantizer = faiss.IndexFlatL2(self.d)
        self.index = faiss.IndexIVFFlat(quantizer, self.d, self.nlist, faiss.METRIC_L2)
        x_tr = np.ascontiguousarray(x_train, dtype=np.float32)
        x_db = np.ascontiguousarray(x_database, dtype=np.float32)

        self.index.train(x_tr)
        self.index.add(x_db)
        self.database_size = x_db.shape[0]

    def imbalance_factor(self) -> float:
        """Extract posting list sizes from Faiss inverted lists and compute imbalance factor."""
        assert self.index is not None
        invlists = self.index.invlists
        lengths = [invlists.list_size(i) for i in range(self.nlist)]
        return compute_imbalance_factor(lengths, num_vectors=self.database_size, num_clusters=self.nlist)

    def evaluate_queries(
        self,
        queries: np.ndarray,
        groundtruth: np.ndarray,
        nprobe_list: Sequence[int] = (1, 2, 4, 8, 16, 32),
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Evaluate Faiss IndexIVFFlat single-threaded query routing over nprobe suite."""
        import faiss

        faiss.omp_set_num_threads(1)
        q_f32 = np.ascontiguousarray(queries, dtype=np.float32)
        num_q = q_f32.shape[0]
        results: list[dict[str, Any]] = []

        for nprobe in nprobe_list:
            self.index.nprobe = min(max(int(nprobe), 1), self.nlist)
            latencies_ms = np.empty(num_q, dtype=np.float64)
            retrieved_ids = np.empty((num_q, top_k), dtype=np.int32)

            for i in range(num_q):
                single_q = q_f32[i : i + 1]
                t0 = time.perf_counter()
                _, I_out = self.index.search(single_q, top_k)
                t1 = time.perf_counter()
                latencies_ms[i] = (t1 - t0) * 1000.0
                retrieved_ids[i] = I_out[0]

            recalls = np.zeros(num_q, dtype=np.float64)
            for i in range(num_q):
                gt_set = set(groundtruth[i, :top_k])
                ret_set = set(retrieved_ids[i])
                recalls[i] = len(gt_set.intersection(ret_set)) / float(top_k)

            mean_recall = float(np.mean(recalls))
            p50_lat = float(np.percentile(latencies_ms, 50))
            p95_lat = float(np.percentile(latencies_ms, 95))
            p99_lat = float(np.percentile(latencies_ms, 99))
            mean_lat = float(np.mean(latencies_ms))
            qps = float(1000.0 / mean_lat) if mean_lat > 0.0 else 0.0

            results.append({
                "nprobe": int(nprobe),
                "recall_at_10": mean_recall,
                "mean_latency_ms": mean_lat,
                "p50_latency_ms": p50_lat,
                "p95_latency_ms": p95_lat,
                "p99_latency_ms": p99_lat,
                "qps": qps,
            })

        return results
