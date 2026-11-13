"""Unit and integration tests for Track T01: gmm-ema."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest
import scipy.linalg
from scipy.optimize import linear_sum_assignment

# Ensure tracks/gmm-ema is in sys.path
TRACK_DIR = Path(__file__).resolve().parent.parent / "tracks" / "gmm-ema"
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

from gmm import (
    GMMParameters,
    compute_batch_stats,
    compute_log_responsibilities,
    compute_responsibilities,
    evaluate_gmm_nll,
    init_gmm_parameters,
    m_step_from_stats,
    revive_gmm_components,
)
from plugin import (
    GMMEmaPlugin,
    fit_full_batch_em,
    fit_gmm_ema,
    fit_online_em,
)
from core.plugin import run_ema_engine
from core.schedule import CosineSchedule
from core.stats import EMAStats


@pytest.fixture
def synthetic_gmm_2d() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate 3 well-separated 2D Gaussian clusters with ground truth."""
    rng = np.random.default_rng(42)
    k = 3
    d = 2
    n_per_k = 400
    true_means = np.array([[-6.0, -6.0], [0.0, 6.0], [6.0, -6.0]], dtype=np.float64)
    true_covars = np.array([[0.5, 0.5], [0.8, 0.3], [0.4, 0.9]], dtype=np.float64)
    true_weights = np.array([0.33, 0.33, 0.34], dtype=np.float64)

    samples = []
    for j in range(k):
        std = np.sqrt(true_covars[j])
        s = rng.normal(loc=true_means[j], scale=std, size=(n_per_k, d))
        samples.append(s)
    x = np.vstack(samples).astype(np.float32)
    rng.shuffle(x)
    return x, true_weights, true_means, true_covars


def test_logsumexp_numerical_stability() -> None:
    """TC: Numerical stability of log-sum-exp under extreme inputs and near-zero weights."""
    k = 4
    d = 8
    b = 32
    rng = np.random.default_rng(101)

    # Test extreme feature magnitudes
    x_extreme = rng.standard_normal((b, d)) * 1e4
    means = rng.standard_normal((k, d)) * 1e4
    covars_diag = np.full((k, d), 1e-4, dtype=np.float64)
    weights = np.array([1e-25, 0.5, 0.5, 1e-30], dtype=np.float64)

    log_resp, log_prob = compute_log_responsibilities(
        x_extreme,
        weights=weights,
        means=means,
        covars=covars_diag,
        cov_type="diag",
        eps_cov=1e-6,
    )

    # No NaN or Inf
    assert np.all(np.isfinite(log_resp)), "Log responsibilities contain non-finite values"
    assert np.all(np.isfinite(log_prob)), "Log marginal likelihood contains non-finite values"

    gamma, _ = compute_responsibilities(
        x_extreme,
        weights=weights,
        means=means,
        covars=covars_diag,
        cov_type="diag",
        hard=False,
    )
    assert np.all(np.isfinite(gamma)), "Responsibilities contain non-finite values"
    assert np.all(gamma >= 0.0)
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-6), "Responsibilities do not sum to 1"


def test_covariance_positive_definiteness_and_regularization() -> None:
    """TC: Covariance floor enforcement and Cholesky positive definiteness."""
    k = 2
    d = 4
    eps_cov = 1e-4

    # Create degenerate singular stats where all observations are identical
    identical_point = np.array([[2.0, 3.0, -1.0, 5.0]], dtype=np.float64)
    x = np.tile(identical_point, (20, 1))
    gamma = np.full((20, k), 0.5, dtype=np.float64)

    # Diagonal mode
    bstats_diag = compute_batch_stats(x, gamma, cov_type="diag")
    stats_diag = EMAStats.from_params(
        init_params=np.zeros((k, d)),
        n0=1e-10,
        extra_inits={"sq_sums": np.zeros((k, d))},
    )
    stats_diag.update(bstats_diag, alpha=1.0)
    params_diag = m_step_from_stats(stats_diag, cov_type="diag", eps_cov=eps_cov)

    assert np.all(params_diag.covars >= eps_cov), "Diagonal variance fell below eps_cov floor"

    # Full covariance mode
    bstats_full = compute_batch_stats(x, gamma, cov_type="full")
    stats_full = EMAStats.from_params(
        init_params=np.zeros((k, d)),
        n0=1e-10,
        extra_inits={"cov_sums": np.zeros((k, d, d))},
    )
    stats_full.update(bstats_full, alpha=1.0)
    params_full = m_step_from_stats(stats_full, cov_type="full", eps_cov=eps_cov)

    for j in range(k):
        cov = params_full.covars[j]
        # Check symmetry
        assert np.allclose(cov, cov.T, atol=1e-8), f"Covariance matrix {j} is not symmetric"
        # Check Cholesky factorization succeeds (strictly positive definite)
        chol = scipy.linalg.cholesky(cov, lower=True)
        assert np.all(np.diag(chol) > 0.0), f"Covariance matrix {j} has non-positive pivots"


def test_parameter_recovery_synthetic(synthetic_gmm_2d) -> None:
    """TC: Parameter recovery on synthetic GMM with known ground truth."""
    x, true_weights, true_means, true_covars = synthetic_gmm_2d
    k = true_means.shape[0]

    # Fit GMM via EMA
    plugin, result = fit_gmm_ema(
        x=x,
        k=k,
        epochs=5.0,
        batch_size=128,
        alpha0=0.1,
        cov_type="diag",
        seed=42,
    )

    fitted_means = plugin.params.means  # (3, 2)
    # Match fitted means to true means via Hungarian algorithm
    cost_matrix = np.linalg.norm(fitted_means[:, None, :] - true_means[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mean_err = cost_matrix[row_ind, col_ind].mean()
    assert mean_err < 0.5, f"Parameter recovery mean error too high: {mean_err:.4f}"

    # Verify log likelihood improves over initial state
    init_loss = evaluate_gmm_nll(x, init_gmm_parameters(x, k=k, rng=np.random.default_rng(42)))
    final_loss = plugin.loss(x)
    assert final_loss < init_loss, f"Loss did not decrease: init={init_loss:.3f}, final={final_loss:.3f}"


def test_dead_component_revival_principal_axis() -> None:
    """TC: Covariance-aware principal axis component revival."""
    k = 2
    d = 2
    eps_cov = 1e-6
    tau = 0.2
    delta = 0.1

    # Donor cluster is elongated along axis 1 (y-axis): var_x = 1.0, var_y = 100.0
    donor_mean = np.array([10.0, 20.0])
    donor_var = np.array([1.0, 100.0])
    donor_count = 100.0

    # Dead cluster has 0 count
    dead_mean = np.array([0.0, 0.0])
    dead_var = np.array([1.0, 1.0])
    dead_count = 0.0

    init_means = np.stack([donor_mean, dead_mean])
    stats = EMAStats.from_params(
        init_params=init_means,
        n0=1e-10,
        extra_inits={"sq_sums": np.stack([donor_var + donor_mean**2, dead_var + dead_mean**2])},
    )
    stats.counts = np.array([donor_count, dead_count], dtype=np.float64)
    stats.sums = np.stack([donor_count * donor_mean, dead_count * dead_mean])
    stats.extras["sq_sums"] = np.stack([
        donor_count * (donor_var + donor_mean**2),
        dead_count * (dead_var + dead_mean**2),
    ])

    # Trigger revival
    event = revive_gmm_components(
        stats=stats,
        cov_type="diag",
        tau=tau,
        delta=delta,
        eps_cov=eps_cov,
    )

    assert len(event.dead_indices) == 1
    assert event.dead_indices[0] == 1  # Component 1 was dead
    assert event.donor_indices[0] == 0  # Component 0 was donor

    # Counts should be halved
    assert np.isclose(stats.counts[0], 50.0)
    assert np.isclose(stats.counts[1], 50.0)

    # Means should be split along axis 1 (the principal axis with var=100 => sqrt(100) = 10)
    # Expected shift: delta * 10 = 0.1 * 10 = 1.0 along axis 1, 0.0 along axis 0
    params = m_step_from_stats(stats, cov_type="diag", eps_cov=eps_cov)
    split_dead = params.means[1]
    split_donor = params.means[0]

    assert np.isclose(split_dead[0], donor_mean[0], atol=1e-4), "Shift occurred along minor axis"
    assert np.isclose(split_donor[0], donor_mean[0], atol=1e-4), "Shift occurred along minor axis"
    assert np.isclose(split_dead[1], donor_mean[1] + 1.0, atol=1e-3)
    assert np.isclose(split_donor[1], donor_mean[1] - 1.0, atol=1e-3)


def test_full_covariance_eigenvector_revival() -> None:
    """TC: Full covariance revival along non-axis-aligned principal eigenvector."""
    k = 2
    d = 2
    eps_cov = 1e-6

    # Create covariance with principal axis along [1, 1] (eigenvalue 10.0) and [1, -1] (eigenvalue 1.0)
    v1 = np.array([1.0, 1.0]) / np.sqrt(2.0)
    v2 = np.array([1.0, -1.0]) / np.sqrt(2.0)
    lambda1 = 16.0  # sqrt = 4.0
    lambda2 = 1.0
    cov_donor = lambda1 * np.outer(v1, v1) + lambda2 * np.outer(v2, v2)
    mean_donor = np.array([5.0, 5.0])

    stats = EMAStats.from_params(
        init_params=np.stack([mean_donor, np.zeros(2)]),
        n0=1e-10,
        extra_inits={"cov_sums": np.stack([cov_donor + np.outer(mean_donor, mean_donor), np.eye(2)])},
    )
    stats.counts = np.array([200.0, 0.0], dtype=np.float64)
    stats.sums = np.stack([200.0 * mean_donor, np.zeros(2)])
    stats.extras["cov_sums"] = np.stack([
        200.0 * (cov_donor + np.outer(mean_donor, mean_donor)),
        np.zeros((2, 2)),
    ])

    delta = 0.25  # Shift = delta * sqrt(lambda1) * v1 = 0.25 * 4 * v1 = 1.0 * v1
    event = revive_gmm_components(stats, cov_type="full", tau=0.1, delta=delta, eps_cov=eps_cov)

    assert len(event.dead_indices) == 1
    params = m_step_from_stats(stats, cov_type="full", eps_cov=eps_cov)

    diff_dead = params.means[1] - mean_donor
    unit_diff = diff_dead / np.linalg.norm(diff_dead)
    # Should align with v1 ([1/sqrt(2), 1/sqrt(2)]) up to sign
    cos_sim = abs(float(np.dot(unit_diff, v1)))
    assert np.isclose(cos_sim, 1.0, atol=1e-3), f"Split not aligned with top eigenvector: cos_sim={cos_sim}"


def test_plugin_lifecycle_and_run_ema_engine() -> None:
    """TC: GMMEmaPlugin lifecycle integrated with run_ema_engine."""
    rng = np.random.default_rng(99)
    x = rng.standard_normal((300, 4)).astype(np.float32)
    k = 4
    d = 4

    # Diagonal mode
    plugin_diag = GMMEmaPlugin(k=k, d=d, cov_type="diag", seed=1)
    sched = CosineSchedule(alpha0=0.08, total_steps=10, eps=0.1)
    res_diag = run_ema_engine(
        plugin=plugin_diag,
        x=x,
        epochs=2.0,
        batch_size=64,
        schedule=sched,
        revival_every=3,
        finalize_mode="none",
        seed=1,
    )
    assert res_diag.steps > 0
    assert res_diag.params.shape == (k, d)
    loss_diag = plugin_diag.loss(x)
    assert np.isfinite(loss_diag)

    # Full covariance mode
    plugin_full = GMMEmaPlugin(k=k, d=d, cov_type="full", seed=2)
    res_full = run_ema_engine(
        plugin=plugin_full,
        x=x,
        epochs=2.0,
        batch_size=64,
        schedule=sched,
        revival_every=3,
        finalize_mode="none",
        seed=2,
    )
    assert res_full.steps > 0
    assert res_full.params.shape == (k, d)
    loss_full = plugin_full.loss(x)
    assert np.isfinite(loss_full)


def test_hard_cem_mode() -> None:
    """TC: Classification EM (Hard CEM) mode."""
    rng = np.random.default_rng(77)
    x = rng.standard_normal((200, 3)).astype(np.float32)
    k = 3

    plugin_cem, res = fit_gmm_ema(
        x=x,
        k=k,
        epochs=2.0,
        batch_size=64,
        alpha0=0.05,
        cov_type="diag",
        hard=True,
        seed=77,
    )
    loss_cem = plugin_cem.loss(x)
    assert np.isfinite(loss_cem)
    assert plugin_cem.hard is True


def test_boundary_conditions() -> None:
    """TC: Edge cases: k=1, small dataset, small batch."""
    rng = np.random.default_rng(55)
    x = rng.standard_normal((30, 2)).astype(np.float32)

    # k=1 boundary
    plugin_k1, res_k1 = fit_gmm_ema(x=x, k=1, epochs=1.0, batch_size=10, seed=55)
    assert plugin_k1.params.means.shape == (1, 2)
    assert np.isfinite(plugin_k1.loss(x))

    # Batch size larger than dataset
    plugin_b_large, _ = fit_gmm_ema(x=x, k=2, epochs=1.0, batch_size=100, seed=55)
    assert np.isfinite(plugin_b_large.loss(x))


def test_baselines_sanity() -> None:
    """TC: Baseline runners (full-batch EM and Online EM) execute cleanly."""
    rng = np.random.default_rng(12)
    x = rng.standard_normal((150, 2)).astype(np.float32)
    k = 2

    # Full batch EM
    params_fb = fit_full_batch_em(x, k=k, cov_type="diag", max_iter=10, seed=12)
    loss_fb = evaluate_gmm_nll(x, params_fb)
    assert np.isfinite(loss_fb)

    # Online EM
    plugin_on, res_on = fit_online_em(x, k=k, epochs=2.0, batch_size=32, seed=12)
    loss_on = plugin_on.loss(x)
    assert np.isfinite(loss_on)


def test_zero_variance_and_duplicate_points() -> None:
    """TC: Robustness to zero-variance (identical points) and few unique points."""
    # 1. Zero-variance data (all identical points)
    x_zero = np.ones((50, 4), dtype=np.float64) * 3.14159
    plugin_diag, res_diag = fit_gmm_ema(x_zero, k=4, cov_type="diag", seed=42)
    assert np.isfinite(plugin_diag.loss(x_zero))
    assert np.all(np.isfinite(plugin_diag.params.means))
    assert np.all(np.isfinite(plugin_diag.params.covars))

    plugin_full, res_full = fit_gmm_ema(x_zero, k=4, cov_type="full", seed=42)
    assert np.isfinite(plugin_full.loss(x_zero))

    # 2. Fewer unique points than k (U = 2 < k = 5)
    x_dup = np.array([[1.0, 2.0], [3.0, 4.0]] * 25, dtype=np.float64)
    params = init_gmm_parameters(x_dup, k=5, rng=np.random.default_rng(42))
    assert params.means.shape == (5, 2)
    assert np.all(np.isfinite(params.means))


def test_k_greater_than_n() -> None:
    """TC: k > N edge case with both diagonal and full covariance."""
    rng = np.random.default_rng(42)
    x_small = rng.standard_normal((3, 4)).astype(np.float64)

    # Full covariance with k > N
    plugin_full, res_full = fit_gmm_ema(x_small, k=10, cov_type="full", seed=42)
    assert plugin_full.k == 3
    assert plugin_full.params.means.shape == (3, 4)
    assert plugin_full.params.covars.shape == (3, 4, 4)
    assert np.isfinite(plugin_full.loss(x_small))

    # Diagonal covariance with k > N
    plugin_diag, res_diag = fit_gmm_ema(x_small, k=10, cov_type="diag", seed=42)
    assert plugin_diag.k == 3
    assert plugin_diag.params.means.shape == (3, 4)
    assert plugin_diag.params.covars.shape == (3, 4)
    assert np.isfinite(plugin_diag.loss(x_small))

    # Baseline fit_full_batch_em with k > N
    params_fb = fit_full_batch_em(x_small, k=10, seed=42)
    assert params_fb.means.shape[0] == 3
    assert np.isfinite(evaluate_gmm_nll(x_small, params_fb))

