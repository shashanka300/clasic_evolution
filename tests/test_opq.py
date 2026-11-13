"""Comprehensive unit test suite for Track T02: opq-joint-rotation."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure tracks/opq-joint-rotation is on sys.path
TRACK_DIR = Path(__file__).resolve().parent.parent / "tracks" / "opq-joint-rotation"
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

from opq import (
    OPQQuantizer,
    StandardPQ,
    PostHocOPQ,
    FaissOPQBenchmark,
    generate_correlated_blobs,
    solve_orthogonal_procrustes,
)
from plugin import OPQPlugin, fit_opq_plugin
from core.schedule import CosineSchedule


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic correlated Gaussian blobs."""
    return generate_correlated_blobs(
        n_train=2000,
        n_test=500,
        d=16,
        k_clusters=8,
        correlation=0.8,
        seed=101,
    )


def test_procrustes_solver_orthogonality_and_reflection() -> None:
    """Test orthogonal Procrustes solver preserves R^T R = I and det(R) > 0."""
    rng = np.random.default_rng(42)

    for d in (2, 4, 16, 32):
        # Case 1: Random matrix
        H = rng.standard_normal((d, d))
        R = solve_orthogonal_procrustes(H)

        assert R.shape == (d, d)
        eye = np.eye(d, dtype=np.float32)
        assert np.allclose(R.T @ R, eye, atol=1e-5), f"Failed orthogonality for d={d}"
        assert np.allclose(R @ R.T, eye, atol=1e-5), f"Failed orthogonality for d={d}"
        det = float(np.linalg.det(R))
        assert det > 0.99, f"Reflection guard failed: det(R)={det} for d={d}"

        # Case 2: Matrix explicitly constructed with negative determinant
        u, _, vt = np.linalg.svd(H)
        # Force det(u @ vt) to be negative
        if np.linalg.det(u @ vt) > 0:
            vt[-1, :] *= -1.0
        H_neg = u @ vt
        assert np.linalg.det(H_neg) < 0.0

        R_fixed = solve_orthogonal_procrustes(H_neg)
        assert np.allclose(R_fixed.T @ R_fixed, eye, atol=1e-5)
        assert float(np.linalg.det(R_fixed)) > 0.99


def test_opq_init_and_shapes(synthetic_data) -> None:
    """Test OPQQuantizer initialization, shapes, and types."""
    x_train, _, _, _ = synthetic_data
    d = 16
    M = 4
    k_sub = 32

    quantizer = OPQQuantizer(d=d, M=M, k_sub=k_sub, seed=42)
    assert quantizer.d == 16
    assert quantizer.M == 4
    assert quantizer.d_s == 4
    assert quantizer.codebooks.shape == (4, 32, 4)
    assert np.allclose(quantizer.R, np.eye(16, dtype=np.float32))

    quantizer.init_codebooks(x_train)
    assert quantizer._is_fitted
    assert not np.allclose(quantizer.codebooks, 0.0)


def test_opq_encode_decode_roundtrip(synthetic_data) -> None:
    """Test quantization encoding and decoding shapes and value constraints."""
    x_train, x_test, _, _ = synthetic_data
    d = 16
    M = 4
    k_sub = 16

    quantizer = OPQQuantizer(d=d, M=M, k_sub=k_sub, seed=42)
    quantizer.fit(x_train, epochs=1.0, batch_size=256)

    codes = quantizer.encode(x_test)
    assert codes.shape == (x_test.shape[0], M)
    assert codes.dtype == np.uint8
    assert np.all(codes >= 0)
    assert np.all(codes < k_sub)

    x_hat = quantizer.decode(codes)
    assert x_hat.shape == x_test.shape
    assert x_hat.dtype == np.float32

    # Reconstruction MSE must be finite and positive
    mse = quantizer.reconstruction_mse(x_test)
    assert isinstance(mse, float)
    assert 0.0 < mse < 100.0


def test_adc_table_and_distance_correctness(synthetic_data) -> None:
    """Verify ADC distance computation matches exact sum of subvector distances."""
    x_train, x_test, queries, _ = synthetic_data
    d = 16
    M = 4
    k_sub = 16

    quantizer = OPQQuantizer(d=d, M=M, k_sub=k_sub, seed=42)
    quantizer.fit(x_train, epochs=1.0, batch_size=256)

    test_queries = queries[:5]
    test_db = x_test[:20]
    codes = quantizer.encode(test_db)

    # Fast ADC distances
    adc_dists = quantizer.adc_distances(test_queries, codes)
    assert adc_dists.shape == (5, 20)

    # Manual ground-truth distance in rotated space
    y_q = test_queries @ quantizer.R.T
    d_s = quantizer.d_s

    for q_idx in range(5):
        for db_idx in range(20):
            expected_dist = 0.0
            for m in range(M):
                sub_q = y_q[q_idx, m * d_s : (m + 1) * d_s]
                centroid = quantizer.codebooks[m, codes[db_idx, m]]
                expected_dist += float(np.sum((sub_q - centroid) ** 2))

            np.testing.assert_allclose(
                adc_dists[q_idx, db_idx],
                expected_dist,
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"ADC distance mismatch at q={q_idx}, db={db_idx}",
            )


def test_adc_search_and_recall_computation(synthetic_data) -> None:
    """Test top-k nearest neighbor search and recall evaluation."""
    x_train, x_test, queries, gt = synthetic_data
    d = 16
    M = 4
    k_sub = 32

    quantizer = OPQQuantizer(d=d, M=M, k_sub=k_sub, seed=42)
    quantizer.fit(x_train, epochs=1.0, batch_size=256)

    codes = quantizer.encode(x_test)
    top_dists, top_indices = quantizer.search_adc(queries[:50], codes, top_k=10)

    assert top_dists.shape == (50, 10)
    assert top_indices.shape == (50, 10)
    # Check monotonicity of returned distances
    for i in range(50):
        assert np.all(np.diff(top_dists[i]) >= -1e-5), "Top-k distances must be sorted"

    recall = quantizer.evaluate_recall(queries[:50], codes, gt[:50], top_k=10)
    assert 0.0 <= recall <= 1.0


def test_standard_pq_baseline(synthetic_data) -> None:
    """Verify StandardPQ maintains R = Identity throughout training."""
    x_train, x_test, _, _ = synthetic_data
    d = 16
    M = 4
    k_sub = 16

    pq = StandardPQ(d=d, M=M, k_sub=k_sub, seed=42)
    pq.fit(x_train, epochs=1.0, batch_size=256)

    eye = np.eye(d, dtype=np.float32)
    assert np.allclose(pq.R, eye, atol=1e-6), "StandardPQ must keep R = Identity"

    mse = pq.reconstruction_mse(x_test)
    assert mse > 0.0


def test_post_hoc_opq_baseline(synthetic_data) -> None:
    """Verify PostHocOPQ fits initial PQ, finds rotation R, and refits."""
    x_train, x_test, _, _ = synthetic_data
    d = 16
    M = 4
    k_sub = 16

    post_hoc = PostHocOPQ(d=d, M=M, k_sub=k_sub, refit_epochs=0.5, seed=42)
    post_hoc.fit(x_train, epochs=1.0, batch_size=256)

    eye = np.eye(d, dtype=np.float32)
    R = post_hoc.quantizer.R
    # R must be orthogonal and non-identity
    assert np.allclose(R.T @ R, eye, atol=1e-5)
    assert float(np.linalg.det(R)) > 0.99
    assert not np.allclose(R, eye, atol=1e-3)

    mse = post_hoc.reconstruction_mse(x_test)
    assert mse > 0.0


def test_faiss_opq_wrapper(synthetic_data) -> None:
    """Verify FaissOPQBenchmark runs when faiss is available."""
    x_train, x_test, queries, gt = synthetic_data
    d = 16
    M = 4

    try:
        import faiss
    except ImportError:
        pytest.skip("Faiss not installed")

    faiss_bench = FaissOPQBenchmark(d=d, M=M, nbits=8)
    faiss_bench.fit(x_train)

    mse = faiss_bench.reconstruction_mse(x_test)
    assert 0.0 < mse < 100.0

    recall = faiss_bench.evaluate_recall(queries[:20], x_test, gt[:20], top_k=10)
    assert 0.0 <= recall <= 1.0


def test_opq_plugin_lifecycle(synthetic_data) -> None:
    """Test OPQPlugin satisfies ClassicAlgoPlugin interface lifecycle."""
    x_train, x_test, _, _ = synthetic_data
    d = 16
    M = 4
    k_sub = 16

    plugin = OPQPlugin(d=d, M=M, k_sub=k_sub, seed=42)
    stats = plugin.init(x_train[:500])

    assert stats.counts.shape == (M * k_sub,)
    assert stats.sums.shape == (M * k_sub, d // M)
    assert "cross_cov" in stats.extras

    batch = x_train[:128]
    bstats = plugin.assign(batch)
    assert bstats.counts.shape == (M * k_sub,)
    assert bstats.sums.shape == (M * k_sub, d // M)
    assert "cross_cov" in bstats.extras

    # Update stats
    stats.update(bstats, alpha=0.1)
    params = plugin.params_from_stats(stats)
    assert params.shape == (M * k_sub, d // M)

    # Test revival
    event = plugin.revive(stats, step=10)
    assert hasattr(event, "dead_indices")

    # Test loss
    loss_val = plugin.loss(x_test[:100])
    assert isinstance(loss_val, float)
    assert loss_val > 0.0


def test_fit_opq_plugin_with_ema_engine(synthetic_data) -> None:
    """Test full integration with core.plugin.run_ema_engine."""
    x_train, x_test, _, _ = synthetic_data
    d = 16
    M = 4
    k_sub = 16

    plugin, res = fit_opq_plugin(
        x=x_train,
        d=d,
        M=M,
        k_sub=k_sub,
        epochs=1.0,
        batch_size=256,
        alpha0=0.1,
        seed=42,
    )

    assert res.steps > 0
    assert res.params.shape == (M * k_sub, d // M)
    # Check that rotation matrix R is orthogonal
    eye = np.eye(d, dtype=np.float32)
    assert np.allclose(plugin.quantizer.R.T @ plugin.quantizer.R, eye, atol=1e-5)
    assert float(np.linalg.det(plugin.quantizer.R)) > 0.99

    loss = plugin.loss(x_test)
    assert 0.0 < loss < 100.0


def test_boundary_cases() -> None:
    """Test boundary cases: M=1 (full vector VQ), M=d (scalar quantization), small N."""
    rng = np.random.default_rng(999)

    # Boundary 1: M = 1 (single subspace = full vector quantization)
    d = 8
    M = 1
    k_sub = 8
    x = rng.standard_normal((100, d)).astype(np.float32)

    q1 = OPQQuantizer(d=d, M=M, k_sub=k_sub, refresh_m=1, seed=42)
    q1.fit(x, epochs=1.0, batch_size=32)
    codes1 = q1.encode(x[:10])
    assert codes1.shape == (10, 1)
    recon1 = q1.decode(codes1)
    assert recon1.shape == (10, d)
    mse1 = q1.reconstruction_mse(x[:10])
    assert mse1 > 0.0

    # Boundary 2: M = d (subspace dimension d_s = 1 = scalar quantization)
    M_d = d
    q_d = OPQQuantizer(d=d, M=M_d, k_sub=4, refresh_m=1, seed=42)
    q_d.fit(x, epochs=1.0, batch_size=32)
    codes_d = q_d.encode(x[:10])
    assert codes_d.shape == (10, d)
    recon_d = q_d.decode(codes_d)
    assert recon_d.shape == (10, d)
    mse_d = q_d.reconstruction_mse(x[:10])
    assert mse_d > 0.0

    # Boundary 3: Small N (N < k_sub)
    small_x = rng.standard_normal((10, d)).astype(np.float32)
    q_small = OPQQuantizer(d=d, M=2, k_sub=16, seed=42)
    q_small.fit(small_x, epochs=1.0, batch_size=5)
    codes_small = q_small.encode(small_x)
    assert codes_small.shape == (10, 2)
    recon_small = q_small.decode(codes_small)
    assert recon_small.shape == (10, d)


def test_deterministic_reproducibility(synthetic_data) -> None:
    """Verify identical seeds produce identical codebooks and rotations."""
    x_train, _, _, _ = synthetic_data
    d = 16
    M = 4
    k_sub = 16

    q_a = OPQQuantizer(d=d, M=M, k_sub=k_sub, seed=777)
    q_a.fit(x_train, epochs=1.0, batch_size=256)

    q_b = OPQQuantizer(d=d, M=M, k_sub=k_sub, seed=777)
    q_b.fit(x_train, epochs=1.0, batch_size=256)

    np.testing.assert_array_equal(q_a.codebooks, q_b.codebooks)
    np.testing.assert_array_equal(q_a.R, q_b.R)
