"""Comprehensive unit test suite for Track T03: balanced-ivf."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure tracks/balanced-ivf is on sys.path
TRACK_DIR = Path(__file__).resolve().parent.parent / "tracks" / "balanced-ivf"
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

from balanced_ivf import (
    BalancedIVFIndex,
    BalancedKMeansIVFIndex,
    EMeansIVFIndex,
    FaissIVFBenchmark,
    SinkhornBalancedAssigner,
    compute_imbalance_factor,
    generate_skewed_data,
    pairwise_sq_dists,
    sinkhorn_knopp_transport,
)
from plugin import BalancedIVFPlugin, fit_balanced_ivf_plugin
from core.schedule import CosineSchedule
from core.stats import EMAStats


@pytest.fixture
def synthetic_skewed_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic skewed Gaussian blobs fixture."""
    return generate_skewed_data(
        n_train=2500,
        n_test=500,
        d=16,
        k_clusters=8,
        skew_factor=2.0,
        seed=101,
    )


def test_pairwise_sq_dists() -> None:
    """Test pairwise squared Euclidean distance matrix properties."""
    rng = np.random.default_rng(42)
    B, K, d = 20, 10, 8
    X = rng.standard_normal((B, d)).astype(np.float32)
    C = rng.standard_normal((K, d)).astype(np.float32)

    M = pairwise_sq_dists(X, C)
    assert M.shape == (B, K)
    assert np.all(M >= 0.0)

    # Spot-check against manual computation
    for i in range(min(B, 5)):
        for j in range(min(K, 5)):
            diff = X[i] - C[j]
            expected = np.sum(diff**2)
            assert np.isclose(M[i, j], expected, rtol=1e-4, atol=1e-5)


def test_sinkhorn_marginals_log_space() -> None:
    """Test that log-space Sinkhorn-Knopp satisfies row and column marginal targets."""
    rng = np.random.default_rng(42)
    B, K = 128, 16
    M = rng.uniform(0.5, 5.0, size=(B, K)).astype(np.float32)

    P, alpha, beta = sinkhorn_knopp_transport(
        M,
        epsilon=0.05,
        num_iters=10,
        log_space=True,
    )

    assert P.shape == (B, K)
    assert np.all(P >= 0.0)

    # Row marginals: sum_j P_{ij} should equal 1/B
    row_sums = np.sum(P, axis=1)
    assert np.allclose(row_sums, 1.0 / B, atol=1e-5)

    # Column marginals: sum_i P_{ij} should be close to 1/K
    col_sums = np.sum(P, axis=0)
    assert np.allclose(col_sums, 1.0 / K, atol=1e-2)


def test_sinkhorn_marginals_stabilized_scaling() -> None:
    """Test stabilized multiplicative Sinkhorn scaling mode."""
    rng = np.random.default_rng(42)
    B, K = 64, 8
    M = rng.uniform(0.1, 4.0, size=(B, K)).astype(np.float32)

    P, u, v = sinkhorn_knopp_transport(
        M,
        epsilon=0.1,
        num_iters=10,
        log_space=False,
    )

    assert P.shape == (B, K)
    assert np.all(P >= 0.0)

    row_sums = np.sum(P, axis=1)
    assert np.allclose(row_sums, 1.0 / B, atol=1e-5)

    col_sums = np.sum(P, axis=0)
    assert np.allclose(col_sums, 1.0 / K, atol=2e-2)


def test_sinkhorn_boundary_conditions() -> None:
    """Test boundary conditions: K=1, B < K, and empty matrices."""
    rng = np.random.default_rng(42)

    # Boundary 1: K = 1
    M_single = rng.uniform(1.0, 2.0, size=(10, 1)).astype(np.float32)
    P_single, _, _ = sinkhorn_knopp_transport(M_single, epsilon=0.05, num_iters=3)
    assert P_single.shape == (10, 1)
    assert np.allclose(np.sum(P_single, axis=1), 1.0 / 10.0)
    assert np.allclose(np.sum(P_single, axis=0), 1.0)

    # Boundary 2: B < K (more clusters than samples, e.g. B=4, K=16)
    M_underdetermined = rng.uniform(0.5, 3.0, size=(4, 16)).astype(np.float32)
    P_und, _, _ = sinkhorn_knopp_transport(M_underdetermined, epsilon=0.05, num_iters=3)
    assert P_und.shape == (4, 16)
    assert not np.any(np.isnan(P_und))
    assert np.all(P_und >= 0.0)
    assert np.allclose(np.sum(P_und, axis=1), 1.0 / 4.0, atol=1e-5)

    # Boundary 3: Empty inputs
    P_empty, _, _ = sinkhorn_knopp_transport(np.empty((0, 5), dtype=np.float32))
    assert P_empty.shape == (0, 5)


def test_imbalance_factor_math() -> None:
    """Test posting list imbalance factor I = (K / N^2) * sum(|L_j|^2)."""
    # Case 1: Perfectly balanced lists
    lengths_uniform = [100, 100, 100, 100]
    I_uniform = compute_imbalance_factor(lengths_uniform)
    assert np.isclose(I_uniform, 1.0, atol=1e-7)

    # Case 2: Completely imbalanced (single list holds everything)
    lengths_worst = [400, 0, 0, 0]
    I_worst = compute_imbalance_factor(lengths_worst)
    assert np.isclose(I_worst, 4.0, atol=1e-7)

    # Case 3: Moderate skew
    lengths_skew = [250, 100, 30, 20]
    N = sum(lengths_skew)
    K = 4
    expected_I = (K / (N**2)) * sum(l**2 for l in lengths_skew)
    assert np.isclose(compute_imbalance_factor(lengths_skew), expected_I)

    # Case 4: K = 1 boundary
    assert compute_imbalance_factor([500]) == 1.0
    # Case 5: Empty
    assert compute_imbalance_factor([]) == 1.0


def test_assigner_hard_discretization() -> None:
    """Test SinkhornBalancedAssigner produces valid hard assignments and diagnostics."""
    rng = np.random.default_rng(42)
    B, K, d = 64, 8, 4
    batch = rng.standard_normal((B, d)).astype(np.float32)
    centroids = rng.standard_normal((K, d)).astype(np.float32)

    assigner = SinkhornBalancedAssigner(epsilon=0.05, num_iters=3)
    labels, P, diag = assigner.assign(batch, centroids)

    assert labels.shape == (B,)
    assert np.all(labels >= 0) and np.all(labels < K)
    assert P.shape == (B, K)
    assert "row_marginal_err" in diag
    assert "batch_imbalance" in diag
    assert diag["row_marginal_err"] < 1e-4


def test_balanced_ivf_index_lifecycle(synthetic_skewed_dataset) -> None:
    """Test BalancedIVFIndex fitting, building, and searching lifecycle."""
    x_train, _, queries, groundtruth = synthetic_skewed_dataset
    d = 16
    nlist = 8

    index = BalancedIVFIndex(
        d=d,
        nlist=nlist,
        epsilon=0.05,
        num_iters=3,
        seed=42,
    )
    assert not index._is_fitted
    assert not index._is_built

    # 1. Fit centroids
    fit_res = index.fit(x_train, epochs=1.0, batch_size=256, alpha0=0.1)
    assert index._is_fitted
    assert fit_res["steps"] > 0
    assert index.centroids.shape == (nlist, d)

    # 2. Build index
    index.build(x_train, assign_mode="nearest")
    assert index._is_built
    assert len(index.posting_lists) == nlist
    assert len(index.posting_vectors) == nlist
    total_stored = sum(len(l) for l in index.posting_lists)
    assert total_stored == x_train.shape[0]

    # 3. Check stats
    stats = index.list_stats()
    assert stats["imbalance_factor"] >= 1.0
    assert stats["mean_length"] == float(x_train.shape[0]) / float(nlist)
    assert stats["max_length"] >= stats["min_length"]

    # 4. Search single query
    q0 = queries[0]
    dists, ids = index.search_single_query(q0, nprobe=4, top_k=10)
    assert len(dists) == 10
    assert len(ids) == 10
    assert np.all(dists[:-1] <= dists[1:])  # sorted ascending

    # 5. Evaluate queries over probe list
    eval_res = index.evaluate_queries(queries[:50], groundtruth[:50], nprobe_list=[1, 2, 4, 8])
    assert len(eval_res) == 4
    # Recall should be monotonically non-decreasing with nprobe
    recalls = [r["recall_at_10"] for r in eval_res]
    assert recalls[-1] >= recalls[0]
    for r in eval_res:
        assert 0.0 <= r["recall_at_10"] <= 1.0
        assert r["p50_latency_ms"] >= 0.0
        assert r["p99_latency_ms"] >= r["p50_latency_ms"]


def test_balanced_vs_unbalanced_skew_reduction(synthetic_skewed_dataset) -> None:
    """Test that Balanced IVF achieves significantly lower imbalance factor than e-means."""
    x_train, _, _, _ = synthetic_skewed_dataset
    d = 16
    nlist = 8

    # Balanced IVF
    b_index = BalancedIVFIndex(d=d, nlist=nlist, epsilon=0.05, num_iters=3, seed=42)
    b_index.fit(x_train, epochs=2.0, batch_size=256, alpha0=0.1)
    b_index.build(x_train, assign_mode="balanced")
    b_imbalance = b_index.imbalance_factor()

    # Unbalanced e-means IVF
    u_index = EMeansIVFIndex(d=d, nlist=nlist, seed=42)
    u_index.fit(x_train, epochs=2.0, batch_size=256, alpha0=0.1)
    u_index.build(x_train, assign_mode="nearest")
    u_imbalance = u_index.imbalance_factor()

    assert b_imbalance < u_imbalance, (
        f"Balanced IVF imbalance ({b_imbalance:.3f}) should be strictly lower than "
        f"unbalanced e-means imbalance ({u_imbalance:.3f})"
    )


def test_plugin_lifecycle_and_engine_integration(synthetic_skewed_dataset) -> None:
    """Test BalancedIVFPlugin conforms to ClassicAlgoPlugin and runs via run_ema_engine."""
    x_train, _, _, _ = synthetic_skewed_dataset
    d = 16
    nlist = 8

    plugin = BalancedIVFPlugin(
        d=d,
        nlist=nlist,
        epsilon=0.05,
        num_iters=3,
        seed=42,
    )

    stats = plugin.init(x_train)
    assert isinstance(stats, EMAStats)
    assert stats.counts.shape == (nlist,)
    assert stats.sums.shape == (nlist, d)

    batch = x_train[:128]
    b_stats = plugin.assign(batch)
    assert b_stats.counts.shape == (nlist,)
    assert b_stats.sums.shape == (nlist, d)
    assert np.isclose(np.sum(b_stats.counts), 128.0)

    params = plugin.params_from_stats(stats)
    assert params.shape == (nlist, d)

    loss_init = plugin.loss(x_train[:500])
    assert loss_init > 0.0

    # Run full engine via fit_balanced_ivf_plugin
    res = fit_balanced_ivf_plugin(
        plugin,
        x_train,
        epochs=1.0,
        batch_size=256,
        seed=42,
    )
    assert res.steps > 0
    loss_fitted = plugin.loss(x_train[:500])
    # Quantization loss after training should be lower than initial random centroids
    assert loss_fitted < loss_init


def test_balanced_kmeans_baseline(synthetic_skewed_dataset) -> None:
    """Test size-penalized BalancedKMeansIVFIndex baseline."""
    x_train, _, queries, groundtruth = synthetic_skewed_dataset
    d = 16
    nlist = 8

    bk_index = BalancedKMeansIVFIndex(d=d, nlist=nlist, penalty_lambda=0.02, seed=42)
    bk_index.fit(x_train, epochs=1.0, batch_size=256, alpha0=0.1)
    bk_index.build(x_train)

    assert bk_index._is_built
    assert bk_index.imbalance_factor() >= 1.0
    eval_res = bk_index.evaluate_queries(queries[:20], groundtruth[:20], nprobe_list=[2, 4])
    assert len(eval_res) == 2
    assert eval_res[1]["recall_at_10"] >= eval_res[0]["recall_at_10"]


def test_faiss_ivf_benchmark(synthetic_skewed_dataset) -> None:
    """Test FaissIVFBenchmark wrapper matches interface."""
    x_train, _, queries, groundtruth = synthetic_skewed_dataset
    d = 16
    nlist = 8

    faiss_bench = FaissIVFBenchmark(d=d, nlist=nlist, seed=42)
    faiss_bench.fit_and_build(x_train, x_train)

    imbalance = faiss_bench.imbalance_factor()
    assert imbalance >= 1.0

    eval_res = faiss_bench.evaluate_queries(queries[:20], groundtruth[:20], nprobe_list=[2, 4])
    assert len(eval_res) == 2
    assert 0.0 <= eval_res[0]["recall_at_10"] <= 1.0
    assert eval_res[0]["p50_latency_ms"] >= 0.0
