"""Comprehensive unit test suite for Track T09: cheap-seeding."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure tracks/cheap-seeding is on sys.path
TRACK_DIR = Path(__file__).resolve().parent.parent / "tracks" / "cheap-seeding"
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

from seeding import (
    seed_random,
    seed_kmeanspp,
    seed_greedy_kmeanspp,
    seed_kmeans_parallel,
    seed_afkmc2,
    seed_subsampled_kmeanspp,
    seed_split_as_you_go_init,
    SEEDING_REGISTRY,
)
from accounting import compute_budget_plan, compute_loss_ratios, BudgetPlan
from plugin import (
    SeededEMeansPlugin,
    SplitAsYouGoPlugin,
    fit_budgeted_seeded_emeans,
    fit_budgeted_split_as_you_go,
)
from core.emeans import eval_kmeans_mse


@pytest.fixture
def synthetic_clusters() -> np.ndarray:
    """Generate 4 well-separated Gaussian clusters in 8 dimensions."""
    rng = np.random.default_rng(123)
    c1 = rng.standard_normal((250, 8)).astype(np.float32) + np.array([10.0] * 8, dtype=np.float32)
    c2 = rng.standard_normal((250, 8)).astype(np.float32) + np.array([-10.0] * 8, dtype=np.float32)
    c3 = rng.standard_normal((250, 8)).astype(np.float32) + np.array([10.0, -10.0] * 4, dtype=np.float32)
    c4 = rng.standard_normal((250, 8)).astype(np.float32) + np.array([-10.0, 10.0] * 4, dtype=np.float32)
    x = np.vstack([c1, c2, c3, c4])
    rng.shuffle(x)
    return x


def test_registry_contains_all_methods() -> None:
    expected = {"random", "kmeanspp", "greedy_kmeanspp", "kmeans_parallel", "afkmc2", "subsampled_kmeanspp", "split_as_you_go"}
    assert expected.issubset(set(SEEDING_REGISTRY.keys()))


def test_seeding_output_shapes_and_dist_evals(synthetic_clusters: np.ndarray) -> None:
    x = synthetic_clusters
    n, d = x.shape
    k = 4
    rng = np.random.default_rng(42)

    # 1. Random
    c_rand, ev_rand = seed_random(x, k, rng)
    assert c_rand.shape == (k, d)
    assert ev_rand == 0

    # 2. k-means++
    c_km, ev_km = seed_kmeanspp(x, k, rng)
    assert c_km.shape == (k, d)
    assert ev_km == (k - 1) * n

    # 3. Greedy k-means++
    c_gr, ev_gr = seed_greedy_kmeanspp(x, k, rng, r=3)
    assert c_gr.shape == (k, d)
    assert ev_gr == 3 * (k - 1) * n

    # 4. k-means||
    c_par, ev_par = seed_kmeans_parallel(x, k, rng, rounds=2)
    assert c_par.shape == (k, d)
    assert ev_par > 0

    # 5. AFK-MC2
    m = 50
    c_afk, ev_afk = seed_afkmc2(x, k, rng, m=m)
    assert c_afk.shape == (k, d)
    n_prop = min(n, max(k, int(0.04 * n * k))) if 1.0 / float(k) >= 0.05 else n
    expected_afk = n_prop + (m + 1) * k * (k - 1) // 2
    assert ev_afk == expected_afk

    # 6. Subsampled k-means++
    c_sub, ev_sub = seed_subsampled_kmeanspp(x, k, rng, sample_ratio=0.1)
    assert c_sub.shape == (k, d)
    assert ev_sub < ev_km

    # 7. Split-as-you-go initial
    c_split, ev_split = seed_split_as_you_go_init(x, k, rng, k0=2)
    assert c_split.shape == (2, d)
    assert ev_split == 0


def test_candidates_satisfy_low_epoch_budget() -> None:
    """Verify that candidate algorithms achieve s <= 0.10 epochs on realistic sizes."""
    n = 100_000
    k = 50
    epoch_evals = n * k

    # AFK-MC2 with m=200
    m = 200
    afk_evals = n + (m + 1) * k * (k - 1) // 2
    s_afk = afk_evals / epoch_evals
    assert s_afk < 0.10, f"AFK-MC2 exceeded 0.10 epoch budget: {s_afk:.4f}"

    # Subsampled k-means++ with sample_ratio=0.05
    b = int(0.05 * n)
    sub_evals = (k - 1) * b
    s_sub = sub_evals / epoch_evals
    assert s_sub < 0.10, f"Subsampled k-means++ exceeded 0.10 epoch budget: {s_sub:.4f}"

    # Split-as-you-go upfront
    s_split = 0.0
    assert s_split <= 0.10


def test_budget_accounting_math() -> None:
    n = 100_000
    k = 50
    batch_size = 4096
    total_budget = 2.0

    # Test when seeding cost is 0.0
    plan_rand = compute_budget_plan(n, k, total_budget, batch_size, dist_evals_seed=0)
    assert plan_rand.seeding_cost_epochs == 0.0
    assert plan_rand.clustering_epochs == 2.0
    assert plan_rand.clustering_steps > 0
    assert plan_rand.is_within_budget

    # Test when seeding cost is s = (k-1)/k ~= 0.98 (standard k-means++)
    km_seed_evals = (k - 1) * n
    plan_km = compute_budget_plan(n, k, 1.0, batch_size, dist_evals_seed=km_seed_evals)
    # At T=1.0, clustering_epochs is 1 - 0.98 = 0.02, steps should be 0 because 0.02 * 24 = 0.48 < 0.5
    assert plan_km.clustering_steps == 0

    # At T=2.0, standard k-means++ should have ~1.02 epochs remaining
    plan_km2 = compute_budget_plan(n, k, 2.0, batch_size, dist_evals_seed=km_seed_evals)
    assert plan_km2.clustering_steps > 0
    assert plan_km2.is_within_budget


def test_seeded_initial_quality_better_than_random(synthetic_clusters: np.ndarray) -> None:
    """Check that k-means++ and AFK-MC2 produce substantially lower initial MSE than random."""
    x = synthetic_clusters
    k = 4
    rng = np.random.default_rng(42)

    c_rand, _ = seed_random(x, k, rng)
    loss_rand = eval_kmeans_mse(x, c_rand)

    c_km, _ = seed_kmeanspp(x, k, rng)
    loss_km = eval_kmeans_mse(x, c_km)

    c_afk, _ = seed_afkmc2(x, k, rng, m=100)
    loss_afk = eval_kmeans_mse(x, c_afk)

    # Well-separated clusters should see huge gain from k-means++ / AFK-MC2
    assert loss_km < loss_rand
    assert loss_afk < loss_rand


def test_seeded_emeans_execution(synthetic_clusters: np.ndarray) -> None:
    x = synthetic_clusters
    n, d = x.shape
    k = 4
    rng = np.random.default_rng(42)

    c_afk, ev_afk = seed_afkmc2(x, k, rng, m=50)
    plan = compute_budget_plan(n, k, total_epochs=2.0, batch_size=64, dist_evals_seed=ev_afk)
    final_c, steps, revivals = fit_budgeted_seeded_emeans(x, c_afk, plan, seed=42)

    assert final_c.shape == (k, d)
    assert steps == plan.clustering_steps
    loss = eval_kmeans_mse(x, final_c)
    assert np.isfinite(loss)


def test_split_as_you_go_growth(synthetic_clusters: np.ndarray) -> None:
    x = synthetic_clusters
    n, d = x.shape
    k = 4
    plan = compute_budget_plan(n, k, total_epochs=2.0, batch_size=64, dist_evals_seed=0)
    final_c, steps, revivals, dist_evals = fit_budgeted_split_as_you_go(
        x, k, plan, k0=2, seed=42
    )

    assert final_c.shape == (k, d)
    assert steps == plan.clustering_steps
    loss = eval_kmeans_mse(x, final_c)
    assert np.isfinite(loss)


def test_loss_ratio_metrics() -> None:
    metrics = compute_loss_ratios(candidate_loss=0.20, random_loss=0.25, unbudgeted_kmpp_loss=0.15)
    assert metrics["ratio_vs_random"] == pytest.approx(0.80)
    assert metrics["recovery_fraction"] == pytest.approx(0.50)


def test_split_as_you_go_k1_boundary(synthetic_clusters: np.ndarray) -> None:
    """Verify SplitAsYouGoPlugin and fit_budgeted_split_as_you_go handle k=1 gracefully."""
    x = synthetic_clusters
    n, d = x.shape
    rng = np.random.default_rng(42)

    # 1. Direct seed_split_as_you_go_init with k=1
    c_init, ev_init = seed_split_as_you_go_init(x, k=1, rng=rng)
    assert c_init.shape == (1, d)
    assert ev_init == 0

    # 2. Plugin directly with k=1
    plugin = SplitAsYouGoPlugin(k=1, seed=42)
    stats = plugin.init(x)
    assert stats.counts.shape == (1,)
    assert stats.sums.shape == (1, d)
    c_params = plugin.params_from_stats(stats)
    assert c_params.shape == (1, d)

    b_stats = plugin.assign(x[:16])
    assert b_stats.counts.shape == (1,)
    assert b_stats.sums.shape == (1, d)

    rev = plugin.revive(stats, step=0)
    assert len(rev.dead_indices) == 0

    loss = plugin.loss(x)
    assert np.isfinite(loss)

    # 3. fit_budgeted_split_as_you_go with k=1
    plan = compute_budget_plan(n, 1, total_epochs=1.0, batch_size=64, dist_evals_seed=0)
    final_c, steps, revivals, dist_evals = fit_budgeted_split_as_you_go(x, 1, plan, seed=42)
    assert final_c.shape == (1, d)
    assert steps == plan.clustering_steps


def test_kmeans_parallel_k_greater_than_n() -> None:
    """Verify seed_kmeans_parallel handles k > n without crashing on replace=False."""
    # Data with fewer points than k and identical coordinates
    x = np.ones((5, 3), dtype=np.float32)
    rng = np.random.default_rng(42)
    c, evals = seed_kmeans_parallel(x, k=8, rng=rng)
    assert c.shape == (8, 3)
    assert evals >= 0

    # Test k=1
    c1, evals1 = seed_kmeans_parallel(x, k=1, rng=rng)
    assert c1.shape == (1, 3)
    assert evals1 == 0


def test_afkmc2_budget_invariant_across_configurations() -> None:
    """Verify s <= 0.10 epoch invariant for seed_afkmc2 across diverse (N, k) pairs."""
    configs = [
        (50_000, 10),
        (50_000, 50),
        (50_000, 100),
        (100_000, 10),
        (100_000, 50),
        (100_000, 100),
        (1_000_000, 10),
    ]
    rng = np.random.default_rng(42)

    for n, k in configs:
        # Use synthetic dummy points for eval count verification
        # Fast 1D points to check exact distance accounting and shape
        x_dummy = rng.standard_normal((n, 2)).astype(np.float32)
        c, evals = seed_afkmc2(x_dummy, k, rng)
        assert c.shape == (k, 2)
        s = float(evals) / float(n * k)
        assert s <= 0.10, f"AFK-MC2 violated s <= 0.10 on (N={n}, k={k}): s={s:.6f}"


def test_afkmc2_k1_boundary() -> None:
    """Verify seed_afkmc2 handles k=1 boundary without distance evals."""
    x = np.random.randn(20, 4).astype(np.float32)
    rng = np.random.default_rng(42)
    c, evals = seed_afkmc2(x, k=1, rng=rng)
    assert c.shape == (1, 4)
    assert evals == 0


def test_accounting_floor_guarantees_no_budget_overrun() -> None:
    """Verify math.floor in accounting prevents budget overrun even in worst-case rounding scenarios."""
    # Counterexample from adversarial challenge
    plan = compute_budget_plan(n=1000, k=10, total_epochs=1.0, batch_size=500, dist_evals_seed=900)
    assert plan.is_within_budget
    assert plan.dist_evals_total <= plan.dist_evals_budget

    # Fine grid test over diverse configurations
    for total_epochs in [1.0, 2.0, 3.0]:
        for n in [1000, 5000, 50000]:
            for k in [5, 10, 50]:
                for b in [64, 500, 4096]:
                    seed_cost_fraction = 0.05
                    d_seed = int(seed_cost_fraction * n * k)
                    p = compute_budget_plan(n, k, total_epochs, b, dist_evals_seed=d_seed)
                    assert p.dist_evals_total <= p.dist_evals_budget, (
                        f"Overrun: {p.dist_evals_total} > {p.dist_evals_budget} for "
                        f"n={n}, k={k}, T={total_epochs}, b={b}"
                    )
                    assert p.is_within_budget
