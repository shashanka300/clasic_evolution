r"""Optimized Product Quantization (OPQ) with Joint Online Orthogonal Procrustes Rotation.

Implements:
- Orthogonal Procrustes rotation solver with reflection guard (det(R) > 0).
- Subspace partitioning and codebook quantization.
- Online cross-covariance EMA accumulation and periodic Procrustes SVD updates.
- Asymmetric Distance Computation (ADC) tables and fast top-k nearest neighbor search.
- Baselines: Standard PQ (R = I), Post-hoc OPQ (PQ + 1-step SVD rotation refit), Faiss reference.
- Synthetic correlated Gaussian generator for benchmarking OPQ rotation advantage.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Generator, Sequence
import numpy as np

from core.schedule import CosineSchedule, StepSchedule
from core.stats import BatchStats, EMAStats
from core.revival import revive_dead_units


def solve_orthogonal_procrustes(H: np.ndarray) -> np.ndarray:
    """Solve the orthogonal Procrustes problem: max_{R in SO(d)} Tr(R H).

    Given cross-covariance H = X^T \\hat{Y} in R^{d x d}, computes the optimal
    orthogonal rotation R in R^{d x d} such that R^T R = I and det(R) = +1.

    Parameters
    ----------
    H : np.ndarray
        Cross-covariance matrix of shape (d, d).

    Returns
    -------
    np.ndarray
        Orthogonal rotation matrix R of shape (d, d) with det(R) > 0.
    """
    H64 = np.asarray(H, dtype=np.float64)
    d = H64.shape[0]
    if d == 1:
        return np.ones((1, 1), dtype=np.float32)

    # SVD: H = U S V^T. In numpy svd: H = U @ np.diag(S) @ Vt, so V = Vt.T
    U, S, Vt = np.linalg.svd(H64, full_matrices=True)

    # Det of candidate R = V @ U^T = Vt.T @ U^T
    # det(V @ U^T) = det(Vt.T) * det(U^T) = det(U @ Vt)
    det = np.linalg.det(U @ Vt)
    if det < 0.0:
        # Reflection guard: flip sign of singular vector for smallest singular value
        Vt_adj = Vt.copy()
        Vt_adj[-1, :] *= -1.0
        R = Vt_adj.T @ U.T
    else:
        R = Vt.T @ U.T

    # Numerical cleanup to ensure strict orthogonality
    # R_clean = R @ (R^T R)^(-1/2) via SVD if necessary
    u_r, _, vt_r = np.linalg.svd(R, full_matrices=True)
    R = u_r @ vt_r
    if np.linalg.det(R) < 0.0:
        u_r[:, -1] *= -1.0
        R = u_r @ vt_r

    return np.ascontiguousarray(R, dtype=np.float32)


def generate_correlated_blobs(
    n_train: int = 50_000,
    n_test: int = 5_000,
    d: int = 32,
    k_clusters: int = 16,
    correlation: float = 0.85,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic Gaussian mixture data with correlated coordinate axes.

    Because dimensions are rotated / correlated across canonical coordinate axes,
    standard axis-aligned PQ suffers higher quantization distortion compared to
    OPQ which discovers the optimal rotation R aligning data with subspaces.

    Returns
    -------
    (x_train, x_test, queries, groundtruth_top10)
    """
    rng = np.random.default_rng(seed)

    # Create random orthogonal rotation to induce axis misalignment
    gaussian_mat = rng.standard_normal((d, d))
    q_mat, _ = np.linalg.qr(gaussian_mat)
    if np.linalg.det(q_mat) < 0.0:
        q_mat[:, -1] *= -1.0

    # Create anisotropic covariance eigenvalues decaying exponentially
    scales = np.exp(-np.linspace(0, 3.0, d)).astype(np.float32)
    cov_diag = np.diag(scales)
    # Apply rotation to covariance
    sigma = (q_mat @ cov_diag @ q_mat.T).astype(np.float32)
    cholesky_l = np.linalg.cholesky(sigma + 1e-5 * np.eye(d, dtype=np.float32))

    # Cluster centers
    centers = rng.uniform(-5.0, 5.0, size=(k_clusters, d)).astype(np.float32)

    def sample_points(n: int) -> np.ndarray:
        c_idx = rng.integers(0, k_clusters, size=n)
        noise = rng.standard_normal((n, d)).astype(np.float32) @ cholesky_l.T
        pts = centers[c_idx] + noise
        return pts.astype(np.float32)

    x_train = sample_points(n_train)
    x_test = sample_points(n_test)
    queries = x_test[: min(1000, n_test)].copy()

    # Exact Euclidean ground truth for queries vs x_test
    # Compute in blocks to stay memory-efficient
    gt_neighbors = np.zeros((queries.shape[0], 10), dtype=np.int32)
    q_norms = np.sum(queries**2, axis=1, keepdims=True)
    t_norms = np.sum(x_test**2, axis=1)

    block_size = 250
    for q_start in range(0, queries.shape[0], block_size):
        q_end = min(q_start + block_size, queries.shape[0])
        dots = queries[q_start:q_end] @ x_test.T
        dists = q_norms[q_start:q_end] + t_norms[None, :] - 2.0 * dots
        gt_neighbors[q_start:q_end] = np.argsort(dists, axis=1)[:, :10]

    return x_train, x_test, queries, gt_neighbors


class OPQQuantizer:
    """Optimized Product Quantization with joint online orthogonal Procrustes rotation.

    Parameters
    ----------
    d : int
        Full vector dimension. Must be divisible by M.
    M : int
        Number of subquantizers (subspaces). Subspace dimension d_s = d / M.
    k_sub : int
        Number of centroids per subquantizer codebook (default 256 for 8-bit codes).
    alpha_r : float
        EMA step size for cross-covariance accumulator H. If negative, defaults to
        tracking the codebook schedule alpha_t.
    refresh_m : int
        Procrustes refresh period: SVD rotation R is recomputed every m batches.
    tau : float
        Dead cluster count revival threshold relative to mean cluster count.
    delta : float
        Donor splitting perturbation magnitude for reviving dead codebook entries.
    seed : int
        Random seed for codebook initialization.
    """

    def __init__(
        self,
        d: int,
        M: int,
        k_sub: int = 256,
        alpha_r: float = 0.05,
        refresh_m: int = 10,
        tau: float = 0.1,
        delta: float = 1.0 / 128.0,
        seed: int = 42,
    ) -> None:
        if d % M != 0:
            raise ValueError(f"Dimension d={d} must be divisible by M={M}")
        self.d = int(d)
        self.M = int(M)
        self.d_s = self.d // self.M
        self.k_sub = int(k_sub)
        self.alpha_r = float(alpha_r)
        self.refresh_m = max(int(refresh_m), 1)
        self.tau = float(tau)
        self.delta = float(delta)
        self.seed = int(seed)

        # Rotation matrix R in SO(d), initialized to Identity
        self.R: np.ndarray = np.eye(self.d, dtype=np.float32)

        # Subspace codebooks: shape (M, k_sub, d_s)
        self.codebooks: np.ndarray = np.zeros(
            (self.M, self.k_sub, self.d_s), dtype=np.float32
        )

        # EMA accumulator for cross-covariance H = X^T \hat{Y}
        self.H: np.ndarray = np.eye(self.d, dtype=np.float64) / float(self.d)

        # Codebook EMA stats per subspace
        self._stats: list[EMAStats] = []
        self._is_fitted: bool = False
        self._batch_counter: int = 0

    def init_codebooks(self, x0: np.ndarray, rng: np.random.Generator | None = None) -> None:
        """Initialize codebooks and rotation from an initial data slice x0."""
        if rng is None:
            rng = np.random.default_rng(self.seed)

        n = x0.shape[0]
        # Rotate x0 with initial R
        y0 = x0 @ self.R.T

        self._stats = []
        for m in range(self.M):
            sub_y = y0[:, m * self.d_s : (m + 1) * self.d_s]
            if n >= self.k_sub:
                sample_idx = rng.choice(n, size=self.k_sub, replace=False)
                c_m = np.ascontiguousarray(sub_y[sample_idx], dtype=np.float32)
            else:
                repeats = math.ceil(self.k_sub / n)
                tiled = np.tile(sub_y, (repeats, 1))[: self.k_sub]
                jitter = rng.standard_normal(tiled.shape).astype(np.float32) * 1e-4
                c_m = np.ascontiguousarray(tiled + jitter, dtype=np.float32)

            self.codebooks[m] = c_m
            self._stats.append(EMAStats.from_params(c_m))

        self.H = np.eye(self.d, dtype=np.float64) / float(self.d)
        self._batch_counter = 0
        self._is_fitted = True

    def _assign_subspace(
        self, sub_y: np.ndarray, m: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assign subvectors to nearest centroid in subspace m.

        Returns
        -------
        (labels, counts, sums)
        """
        c_m = self.codebooks[m]  # (k_sub, d_s)
        # ||y - c||^2 = ||c||^2 - 2 y c^T (+ ||y||^2 omitted for argmin)
        c_norms_sq = np.sum(c_m**2, axis=1)  # (k_sub,)
        dots = sub_y @ c_m.T  # (B, k_sub)
        dists = c_norms_sq[None, :] - 2.0 * dots
        labels = np.argmin(dists, axis=1).astype(np.int32)

        counts = np.bincount(labels, minlength=self.k_sub).astype(np.float64)
        sums = np.zeros((self.k_sub, self.d_s), dtype=np.float64)
        np.add.at(sums, labels, sub_y.astype(np.float64))

        return labels, counts, sums

    def update_batch(
        self,
        batch_x: np.ndarray,
        alpha_codebook: float,
        alpha_rot: float | None = None,
        update_rotation: bool = True,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Process one mini-batch of vectors.

        1. Rotate batch: Y = X @ R^T.
        2. Assign each subspace to nearest centroid and update EMAStats.
        3. Form quantized rotated vector \hat{Y}.
        4. Accumulate cross-covariance H and periodically update R via Procrustes SVD.

        Returns
        -------
        (codes, y_hat)
        """
        b_size = batch_x.shape[0]
        batch_x_f32 = np.ascontiguousarray(batch_x, dtype=np.float32)

        # 1. Rotate batch
        batch_y = batch_x_f32 @ self.R.T  # (B, d)

        # 2. Subspace assignment and EMA accumulation
        codes = np.zeros((b_size, self.M), dtype=np.int32)
        y_hat = np.zeros((b_size, self.d), dtype=np.float32)

        for m in range(self.M):
            sub_y = batch_y[:, m * self.d_s : (m + 1) * self.d_s]
            labels, counts, sums = self._assign_subspace(sub_y, m)
            codes[:, m] = labels
            y_hat[:, m * self.d_s : (m + 1) * self.d_s] = self.codebooks[m][labels]

            # Update subspace EMAStats
            bstats = BatchStats(counts=counts, sums=sums)
            self._stats[m].update(bstats, alpha=alpha_codebook)
            self.codebooks[m] = self._stats[m].means(out_dtype=np.float32)

            # Revival of dead centroids within subspace
            if rng is not None:
                revive_dead_units(
                    self._stats[m], tau=self.tau, delta=self.delta, rng=rng
                )
                self.codebooks[m] = self._stats[m].means(out_dtype=np.float32)

        # 3. Cross-covariance accumulation H = X^T \hat{Y} / B
        if update_rotation:
            h_batch = (batch_x_f32.T @ y_hat).astype(np.float64) / float(b_size)
            a_r = self.alpha_r if self.alpha_r > 0.0 else (alpha_rot or alpha_codebook)
            self.H = (1.0 - a_r) * self.H + a_r * h_batch

            self._batch_counter += 1
            if self._batch_counter % self.refresh_m == 0:
                self.R = solve_orthogonal_procrustes(self.H)

        return codes, y_hat

    def fit(
        self,
        x: np.ndarray,
        epochs: float = 1.0,
        batch_size: int = 1024,
        schedule: StepSchedule | None = None,
        update_rotation: bool = True,
        seed: int | None = None,
    ) -> OPQQuantizer:
        """Fit codebooks and joint rotation on training set x."""
        use_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(use_seed)
        n = x.shape[0]
        b = min(max(int(batch_size), 1), n)

        if not self._is_fitted:
            self.init_codebooks(x[: min(n, max(b, self.k_sub * 4))], rng=rng)

        steps_per_epoch = max(n // b, 1)
        total_steps = max(int(round(epochs * steps_per_epoch)), 1)

        if schedule is None:
            schedule = CosineSchedule(
                alpha0=0.1, total_steps=total_steps, eps=0.1
            )

        step = 0
        while step < total_steps:
            perm = rng.permutation(n)
            for start in range(0, steps_per_epoch * b, b):
                if step >= total_steps:
                    break
                batch_idx = perm[start : start + b]
                batch_x = np.asarray(x[batch_idx], dtype=np.float32)
                alpha_t = schedule(step)
                self.update_batch(
                    batch_x,
                    alpha_codebook=alpha_t,
                    alpha_rot=alpha_t,
                    update_rotation=update_rotation,
                    rng=rng,
                )
                step += 1

        # Final Procrustes rotation update at end of training
        if update_rotation:
            self.R = solve_orthogonal_procrustes(self.H)

        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input vectors into discrete cluster indices.

        Returns
        -------
        np.ndarray
            Array of shape (N, M) of dtype np.uint8 if k_sub <= 256 else np.int32.
        """
        x_f32 = np.ascontiguousarray(x, dtype=np.float32)
        n = x_f32.shape[0]
        y = x_f32 @ self.R.T

        out_dtype = np.uint8 if self.k_sub <= 256 else np.int32
        codes = np.zeros((n, self.M), dtype=out_dtype)

        # Process in chunks to prevent large intermediate distance tensors
        chunk_size = 8192
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_y = y[start:end]
            for m in range(self.M):
                sub_y = chunk_y[:, m * self.d_s : (m + 1) * self.d_s]
                c_m = self.codebooks[m]
                c_norms_sq = np.sum(c_m**2, axis=1)
                dots = sub_y @ c_m.T
                dists = c_norms_sq[None, :] - 2.0 * dots
                codes[start:end, m] = np.argmin(dists, axis=1).astype(out_dtype)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct vectors in the original space from discrete codes.

        \\hat{X} = \\hat{Y} @ R.

        Returns
        -------
        np.ndarray
            Reconstructed vectors of shape (N, d) and dtype np.float32.
        """
        n = codes.shape[0]
        y_hat = np.zeros((n, self.d), dtype=np.float32)

        for m in range(self.M):
            sub_codes = codes[:, m]
            y_hat[:, m * self.d_s : (m + 1) * self.d_s] = self.codebooks[m][sub_codes]

        # Invert rotation: x_hat = y_hat @ R (since y = x @ R^T => x = y @ R)
        x_hat = y_hat @ self.R
        return np.ascontiguousarray(x_hat, dtype=np.float32)

    def reconstruction_mse(self, x: np.ndarray) -> float:
        """Compute mean squared reconstruction error per dimension."""
        codes = self.encode(x)
        x_hat = self.decode(codes)
        x_f32 = np.ascontiguousarray(x, dtype=np.float32)
        diff = x_f32 - x_hat
        mse = float(np.mean(diff**2))
        return mse

    def compute_adc_table(self, queries: np.ndarray) -> np.ndarray:
        """Precompute Asymmetric Distance Computation (ADC) lookup table for queries.

        For each query q, precomputes squared Euclidean distance to all centroids
        across all M subspaces:
        D[q, m, k] = ||(R q)^m - C_{m, k}||^2.

        Returns
        -------
        np.ndarray
            Distance table of shape (Q, M, k_sub) and dtype np.float32.
        """
        q_f32 = np.ascontiguousarray(queries, dtype=np.float32)
        num_q = q_f32.shape[0]
        # Rotate queries
        y_q = q_f32 @ self.R.T  # (Q, d)

        table = np.zeros((num_q, self.M, self.k_sub), dtype=np.float32)
        for m in range(self.M):
            sub_q = y_q[:, m * self.d_s : (m + 1) * self.d_s]  # (Q, d_s)
            c_m = self.codebooks[m]  # (k_sub, d_s)
            q_norms = np.sum(sub_q**2, axis=1, keepdims=True)  # (Q, 1)
            c_norms = np.sum(c_m**2, axis=1)  # (k_sub,)
            dots = sub_q @ c_m.T  # (Q, k_sub)
            sub_dists = q_norms + c_norms[None, :] - 2.0 * dots
            # Guard numerical clipping below 0.0
            table[:, m, :] = np.maximum(sub_dists, 0.0)

        return table

    def adc_distances(self, queries: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Compute ADC distances from queries to encoded database items.

        Parameters
        ----------
        queries : np.ndarray
            Query vectors of shape (Q, d).
        codes : np.ndarray
            Quantized codes of shape (N, M).

        Returns
        -------
        np.ndarray
            Distance matrix of shape (Q, N).
        """
        adc_table = self.compute_adc_table(queries)  # (Q, M, k_sub)
        num_q = queries.shape[0]
        num_items = codes.shape[0]

        dists = np.zeros((num_q, num_items), dtype=np.float32)
        for m in range(self.M):
            sub_codes = codes[:, m]  # (N,)
            sub_table = adc_table[:, m, :]  # (Q, k_sub)
            dists += sub_table[:, sub_codes]

        return dists

    def search_adc(
        self, queries: np.ndarray, codes: np.ndarray, top_k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform top-k nearest neighbor search via ADC table lookup.

        Returns
        -------
        (top_k_distances, top_k_indices) of shape (Q, top_k).
        """
        dists = self.adc_distances(queries, codes)
        top_k = min(top_k, dists.shape[1])
        top_indices = np.argsort(dists, axis=1)[:, :top_k]
        top_dists = np.take_along_axis(dists, top_indices, axis=1)
        return top_dists, top_indices

    def evaluate_recall(
        self,
        queries: np.ndarray,
        codes: np.ndarray,
        groundtruth: np.ndarray,
        top_k: int = 10,
    ) -> float:
        """Compute top-k Recall against exact Euclidean ground truth."""
        _, pred_indices = self.search_adc(queries, codes, top_k=top_k)
        num_q = queries.shape[0]
        recall_sum = 0.0
        for i in range(num_q):
            gt_set = set(groundtruth[i, :top_k])
            pred_set = set(pred_indices[i, :top_k])
            recall_sum += len(gt_set.intersection(pred_set)) / float(top_k)
        return float(recall_sum / num_q)


class StandardPQ(OPQQuantizer):
    """Standard Product Quantization baseline with fixed Identity rotation (R = I)."""

    def __init__(
        self,
        d: int,
        M: int,
        k_sub: int = 256,
        tau: float = 0.1,
        delta: float = 1.0 / 128.0,
        seed: int = 42,
    ) -> None:
        super().__init__(
            d=d,
            M=M,
            k_sub=k_sub,
            alpha_r=0.0,
            refresh_m=10_000_000,
            tau=tau,
            delta=delta,
            seed=seed,
        )

    def fit(
        self,
        x: np.ndarray,
        epochs: float = 1.0,
        batch_size: int = 1024,
        schedule: StepSchedule | None = None,
        update_rotation: bool = False,
        seed: int | None = None,
    ) -> StandardPQ:
        # Guarantee R remains strictly Identity
        self.R = np.eye(self.d, dtype=np.float32)
        super().fit(
            x=x,
            epochs=epochs,
            batch_size=batch_size,
            schedule=schedule,
            update_rotation=False,
            seed=seed,
        )
        self.R = np.eye(self.d, dtype=np.float32)
        return self


class PostHocOPQ:
    """Post-hoc OPQ: Train Standard PQ (R=I), then perform one SVD rotation refit.

    This serves as the primary competitor / kill-criterion baseline:
    tests whether online continuous Procrustes rotation during training
    beats training unrotated PQ and doing one final SVD rotation refit.
    """

    def __init__(
        self,
        d: int,
        M: int,
        k_sub: int = 256,
        refit_epochs: float = 1.0,
        tau: float = 0.1,
        delta: float = 1.0 / 128.0,
        seed: int = 42,
    ) -> None:
        self.d = d
        self.M = M
        self.k_sub = k_sub
        self.refit_epochs = float(refit_epochs)
        self.tau = tau
        self.delta = delta
        self.seed = seed
        self.quantizer = OPQQuantizer(
            d=d,
            M=M,
            k_sub=k_sub,
            alpha_r=0.0,
            refresh_m=10_000_000,
            tau=tau,
            delta=delta,
            seed=seed,
        )

    def fit(
        self,
        x: np.ndarray,
        epochs: float = 1.0,
        batch_size: int = 1024,
        schedule: StepSchedule | None = None,
        seed: int | None = None,
    ) -> PostHocOPQ:
        # Phase 1: Train Standard PQ with R = I
        self.quantizer.R = np.eye(self.d, dtype=np.float32)
        self.quantizer.fit(
            x,
            epochs=epochs,
            batch_size=batch_size,
            schedule=schedule,
            update_rotation=False,
            seed=seed,
        )

        # Phase 2: Compute full cross-covariance on a sample and solve SVD rotation
        sample_n = min(x.shape[0], 20_000)
        x_sample = np.ascontiguousarray(x[:sample_n], dtype=np.float32)
        codes = self.quantizer.encode(x_sample)
        y_hat = np.zeros((sample_n, self.d), dtype=np.float32)
        for m in range(self.M):
            sub_codes = codes[:, m]
            y_hat[:, m * self.quantizer.d_s : (m + 1) * self.quantizer.d_s] = (
                self.quantizer.codebooks[m][sub_codes]
            )

        H = (x_sample.T @ y_hat).astype(np.float64) / float(sample_n)
        R_post = solve_orthogonal_procrustes(H)
        self.quantizer.R = R_post

        # Phase 3: One pass of codebook refit on newly rotated data
        if self.refit_epochs > 0:
            self.quantizer.fit(
                x,
                epochs=self.refit_epochs,
                batch_size=batch_size,
                schedule=schedule,
                update_rotation=False,
                seed=seed,
            )
            # Re-ensure R remains R_post
            self.quantizer.R = R_post

        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        return self.quantizer.encode(x)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return self.quantizer.decode(codes)

    def reconstruction_mse(self, x: np.ndarray) -> float:
        return self.quantizer.reconstruction_mse(x)

    def compute_adc_table(self, queries: np.ndarray) -> np.ndarray:
        return self.quantizer.compute_adc_table(queries)

    def adc_distances(self, queries: np.ndarray, codes: np.ndarray) -> np.ndarray:
        return self.quantizer.adc_distances(queries, codes)

    def search_adc(
        self, queries: np.ndarray, codes: np.ndarray, top_k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.quantizer.search_adc(queries, codes, top_k=top_k)

    def evaluate_recall(
        self,
        queries: np.ndarray,
        codes: np.ndarray,
        groundtruth: np.ndarray,
        top_k: int = 10,
    ) -> float:
        return self.quantizer.evaluate_recall(queries, codes, groundtruth, top_k=top_k)


class FaissOPQBenchmark:
    """Faiss OPQMatrix + IndexPQ benchmark wrapper."""

    def __init__(self, d: int, M: int, nbits: int = 8) -> None:
        self.d = d
        self.M = M
        self.nbits = nbits
        self.index = None
        self.opq = None

    def fit(self, x_train: np.ndarray) -> FaissOPQBenchmark:
        import faiss

        x_f32 = np.ascontiguousarray(x_train, dtype=np.float32)
        self.opq = faiss.OPQMatrix(self.d, self.M)
        pq = faiss.IndexPQ(self.d, self.M, self.nbits)
        self.index = faiss.IndexPreTransform(self.opq, pq)
        self.index.train(x_f32)
        return self

    def reconstruction_mse(self, x_eval: np.ndarray) -> float:
        x_f32 = np.ascontiguousarray(x_eval, dtype=np.float32)
        # Reconstruct through faiss
        import faiss

        # Pre-transform x
        x_rot = self.opq.apply(x_f32)
        pq_idx = faiss.downcast_index(self.index.index)
        pq = pq_idx.pq
        codes = pq.compute_codes(x_rot)
        y_recon = pq.decode(codes)
        # Reverse OPQ rotation: R is orthogonal, so reverse is R^T
        # In Faiss OPQMatrix: A is the rotation matrix, x_rot = x @ A^T
        # Reverse is y_recon @ A
        a_mat = faiss.vector_to_array(self.opq.A).reshape(self.d, self.d)
        x_recon = y_recon @ a_mat
        mse = float(np.mean((x_f32 - x_recon) ** 2))
        return mse

    def evaluate_recall(
        self,
        queries: np.ndarray,
        database: np.ndarray,
        groundtruth: np.ndarray,
        top_k: int = 10,
    ) -> float:
        import faiss

        q_f32 = np.ascontiguousarray(queries, dtype=np.float32)
        db_f32 = np.ascontiguousarray(database, dtype=np.float32)

        # Create temporary index with database added
        self.index.reset()
        self.index.add(db_f32)
        dists, indices = self.index.search(q_f32, top_k)

        num_q = q_f32.shape[0]
        recall_sum = 0.0
        for i in range(num_q):
            gt_set = set(groundtruth[i, :top_k])
            pred_set = set(indices[i, :top_k])
            recall_sum += len(gt_set.intersection(pred_set)) / float(top_k)
        return float(recall_sum / num_q)
