"""Unit and equivalence tests for the core e-means engine."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from core.bench import REQUIRED_RESULT_KEYS, Timer, read_results, write_result
from core.emeans import EMeans, ToyKMeansPlugin, eval_kmeans_mse
from core.finalize import LastEpochMeanFinalizer, SlowEMAFinalizer
from core.plugin import run_ema_engine
from core.revival import revive_dead_units
from core.scaling import (
    emeans_alpha0,
    emeans_batch_size,
    find_batch_knee,
    fit_log_linear,
    run_sweep,
)
from core.schedule import (
    CosineSchedule,
    WarmRestartCosineSchedule,
    make_schedule,
)
from core.stats import BatchStats, EMAStats


def test_stats_exact_initialization_and_update() -> None:
    c0 = np.array([[1.0, -2.0], [3.5, 4.5]], dtype=np.float32)
    stats = EMAStats.from_params(c0, n0=1e-10)
    np.testing.assert_allclose(stats.means(), c0, rtol=1e-6, atol=1e-6)

    batch = BatchStats(
        counts=np.array([2.0, 0.0]),
        sums=np.array([[4.0, 6.0], [0.0, 0.0]]),
    )
    stats.update(batch, alpha=0.5)
    means = stats.means(out_dtype=np.float64)
    np.testing.assert_allclose(means[0], [2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(means[1], [3.5, 4.5], atol=1e-6)


def test_schedules() -> None:
    cos = CosineSchedule(alpha0=0.4, total_steps=100, eps=0.1)
    assert abs(cos(0) - 0.4) < 1e-12
    assert abs(cos(100) - 0.04) < 1e-12
    assert abs(cos(50) - 0.4 * 0.55) < 1e-12

    const = make_schedule("constant", alpha0=0.2, total_steps=10)
    assert const(0) == 0.2 and const(9) == 0.2

    inv = make_schedule("inverse_time", alpha0=0.5, total_steps=10, tau0=1.0, kappa=1.0)
    assert abs(inv(0) - 0.5) < 1e-12
    assert abs(inv(1) - 0.25) < 1e-12

    warm = WarmRestartCosineSchedule(alpha0=0.5, period=20, eps=0.1)
    assert abs(warm(20) - 0.5) < 1e-12
    warm.trigger_restart(step=35, new_period=10)
    assert abs(warm(35) - 0.5) < 1e-12


def test_revival_splits_donor_and_halves_donor_count() -> None:
    c0 = np.array([[10.0, 10.0, 10.0, 10.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
    stats = EMAStats(
        counts=np.array([100.0, 1.0]),
        sums=np.array([100.0 * c0[0], 1.0 * c0[1]]),
    )
    rng = np.random.default_rng(7)
    ev = revive_dead_units(stats, tau=0.1, delta=0.01, rng=rng)
    assert list(ev.dead_indices) == [1]
    assert list(ev.donor_indices) == [0]
    assert abs(stats.counts.sum() - 100.0) < 1e-9
    assert abs(stats.counts[0] - 50.0) < 1e-9
    assert abs(stats.counts[1] - 50.0) < 1e-9
    means = stats.means(out_dtype=np.float64)
    np.testing.assert_allclose(means[0], [9.9, 10.1, 9.9, 10.1], atol=1e-9)
    np.testing.assert_allclose(means[1], [10.1, 9.9, 10.1, 9.9], atol=1e-9)


def test_finalizers() -> None:
    c0 = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float32)
    fin = LastEpochMeanFinalizer(k=2, shape=(2,))
    fin.observe(BatchStats(counts=np.array([2.0, 0.0]), sums=np.array([[2.0, 4.0], [0.0, 0.0]])))
    out = fin.finalize(c0)
    np.testing.assert_allclose(out[0], [1.0, 2.0])
    np.testing.assert_allclose(out[1], [5.0, 5.0])

    slow = SlowEMAFinalizer(c0, half_life=10.0)
    assert 0.06 < slow.alpha_slow < 0.07
    slow.observe(BatchStats(counts=np.array([10.0, 10.0]), sums=np.array([[20.0, 20.0], [50.0, 50.0]])))
    assert np.all(np.isfinite(slow.finalize()))


def test_scaling_laws_and_fitter() -> None:
    b = emeans_batch_size(n=1_000_000, k=1024, epochs=2.0)
    a0 = emeans_alpha0(n=1_000_000, k=1024, epochs=2.0, batch_size=b)
    assert 1 <= b <= 1_000_000 and (b & (b - 1)) == 0
    assert 0.0 < a0 <= 1.0

    # Synthetic exact power law: y = 2.0 * n^0.5 * k^-0.25 * T^-0.1 * B^0.3
    rng = np.random.default_rng(0)
    recs = []
    for _ in range(30):
        n = float(rng.uniform(1e3, 1e6))
        k = float(rng.uniform(16, 1024))
        t = float(rng.uniform(1, 20))
        b_val = float(rng.uniform(64, 4096))
        y = 2.0 * (n ** 0.5) * (k ** -0.25) * (t ** -0.1) * (b_val ** 0.3)
        recs.append({"n": n, "k": k, "T": t, "B": b_val, "alpha0": y})

    law = fit_log_linear(recs, target="alpha0", features=("n", "k", "T", "B"))
    assert law.r2 > 0.999999
    assert abs(law.constant - 2.0) < 1e-6
    assert abs(law.coeffs["n"] - 0.5) < 1e-6
    assert abs(law.coeffs["k"] + 0.25) < 1e-6

    knee = find_batch_knee([64, 128, 256, 512], [1.0, 1.001, 1.005, 1.05], rel_tol=0.002)
    assert 128 <= knee <= 256

    swept = run_sweep(lambda c: {"x": c["x"] * 2}, [{"x": 1}, {"x": 3}])
    assert swept == [{"x": 2}, {"x": 6}]


def test_emeans_matches_toy_plugin_identically() -> None:
    rng = np.random.default_rng(123)
    centers = rng.standard_normal((16, 24)).astype(np.float32) * 3.0
    x = np.repeat(centers, 200, axis=0) + 0.35 * rng.standard_normal((3200, 24)).astype(np.float32)

    em = EMeans(k=16, epochs=3.0, batch_size=256, alpha0=0.25, device="cpu", seed=42).fit(x)

    plugin = ToyKMeansPlugin(k=16, device="cpu", seed=42)
    steps = int(round(3.0 * (len(x) // 256)))
    sched = CosineSchedule(alpha0=0.25, total_steps=steps, eps=0.1)
    res = run_ema_engine(plugin, x, epochs=3.0, batch_size=256, schedule=sched, seed=42)

    np.testing.assert_allclose(em.centroids_, res.params, rtol=0.0, atol=0.0)
    assert em.score(x) == plugin.loss(x)
    assert em.revivals_ == res.revivals

    # Also verify synthetic clustering beats random init substantially.
    init_mse = eval_kmeans_mse(x, x[:16], device="cpu")
    assert em.score(x) < init_mse * 0.2


def test_jsonl_writer_and_timer() -> None:
    with Timer(use_mps=False) as t:
        _ = np.ones((100, 100)) @ np.ones((100, 100))
    assert t.wall_s >= 0.0

    rec = {
        "track": "T00",
        "method": "e-means",
        "dataset": "synthetic",
        "n": 1000,
        "d": 16,
        "k_or_units": 8,
        "epochs": 2,
        "batch": 128,
        "alpha0": 0.1,
        "seed": 0,
        "train_loss": 0.5,
        "test_loss": 0.55,
        "wall_s": t.wall_s,
        "gpu_s": 0.0,
        "extra": {"device": "cpu"},
    }
    assert tuple(rec.keys()) == REQUIRED_RESULT_KEYS
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "res.jsonl"
        write_result(out, rec)
        loaded = read_results(out)
        assert loaded == [rec]
