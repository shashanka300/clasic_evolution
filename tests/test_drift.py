"""Unit and integration tests for Track T08: drift-rewarm."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure tracks/drift-rewarm is in sys.path
TRACK_DIR = Path(__file__).resolve().parent.parent / "tracks" / "drift-rewarm"
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

from detector import PageHinkley, CUSUM, DriftScenario
from plugin import DriftRewarmKMeans, FrozenEMeans, ConstantEMAKMeans, assign_and_loss
from core.schedule import WarmRestartCosineSchedule


def test_page_hinkley_stationary():
    """Verify Page-Hinkley does not trigger false alarms on stationary gaussian noise."""
    ph = PageHinkley(delta=0.05, threshold=15.0)
    rng = np.random.default_rng(42)
    alarms = 0
    for _ in range(100):
        val = float(rng.normal(1.0, 0.1))
        if ph.update(val):
            alarms += 1
    assert alarms == 0


def test_page_hinkley_detects_abrupt_jump():
    """Verify Page-Hinkley triggers alarm upon abrupt upward shift in loss."""
    ph = PageHinkley(delta=0.02, threshold=2.0)
    rng = np.random.default_rng(42)
    # Warm up with low loss
    for _ in range(30):
        ph.update(float(rng.normal(0.5, 0.05)))
    
    # Inject large jump in loss
    alarm_fired = False
    for _ in range(20):
        if ph.update(float(rng.normal(3.0, 0.1))):
            alarm_fired = True
            break
    assert alarm_fired is True


def test_cusum_detector():
    """Verify CUSUM detector detects shift."""
    cusum = CUSUM(threshold=5.0, drift_tol=0.05)
    rng = np.random.default_rng(42)
    for _ in range(25):
        cusum.update(float(rng.normal(1.0, 0.05)))
    detected = False
    for _ in range(25):
        if cusum.update(float(rng.normal(3.5, 0.1))):
            detected = True
            break
    assert detected is True


@pytest.mark.parametrize("drift_type", ["abrupt", "gradual", "seasonal", "encoder_upgrade"])
def test_drift_scenarios_generate_correct_shapes(drift_type: str):
    """Verify all 4 drift scenarios stream valid numpy arrays."""
    scenario = DriftScenario(n_batches=10, batch_size=50, dim=8, k=4, drift_type=drift_type, seed=42)
    count = 0
    for batch, test_win, is_drift in scenario.stream():
        assert batch.shape == (50, 8)
        assert test_win.shape == (1000, 8)
        assert isinstance(is_drift, bool)
        count += 1
    assert count == 10


def test_warm_restart_schedule_behavior():
    """Verify schedule anneals and warm restart raises alpha back up."""
    sched = WarmRestartCosineSchedule(alpha0=0.5, period=20, eps=0.1)
    a0 = sched(0)
    assert abs(a0 - 0.5) < 1e-4

    # Halfway through period
    a_mid = sched(10)
    assert a_mid < a0

    # Near end of period (step 19 out of 20)
    a_end = sched(19)
    assert a_end < a_mid
    assert a_end < 0.1

    # Trigger explicit restart at step 15
    sched.trigger_restart(step=15, new_period=20)
    a_restart = sched(15)
    assert abs(a_restart - 0.5) < 1e-4


def test_drift_rewarm_kmeans_initialization():
    """Verify DriftRewarmKMeans initializes properly."""
    model = DriftRewarmKMeans(k=8, dim=16, alpha0=0.4, seed=42)
    assert model.centroids.shape == (8, 16)
    assert model.counts.shape == (8,)
    assert model.sums.shape == (8, 16)
    assert model.step == 0


def test_drift_rewarm_kmeans_batch_update():
    """Verify updating with a batch updates centroids and counts."""
    model = DriftRewarmKMeans(k=4, dim=8, alpha0=0.5, seed=42)
    rng = np.random.default_rng(42)
    batch = rng.normal(0.0, 1.0, size=(100, 8)).astype(np.float32)
    
    old_centroids = model.centroids.copy()
    mse = model.update_batch(batch)
    
    assert mse > 0.0
    assert not np.allclose(model.centroids, old_centroids)
    assert model.step == 1


def test_drift_rewarm_outperforms_frozen_on_drift():
    """Verify drift-rewarm achieves lower test loss than frozen e-means after concept drift."""
    scenario = DriftScenario(n_batches=50, batch_size=200, dim=16, k=8, drift_type="abrupt", seed=42)
    
    model_rewarm = DriftRewarmKMeans(k=8, dim=16, alpha0=0.6, restart_period=15, seed=42)
    model_frozen = FrozenEMeans(k=8, dim=16, alpha0=0.6, total_steps=50, seed=42)

    last_test_rewarm = 0.0
    last_test_frozen = 0.0

    for step, (batch, test_window, _) in enumerate(scenario.stream()):
        model_rewarm.update_batch(batch)
        model_frozen.update_batch(batch)
        if step >= 40:
            last_test_rewarm = model_rewarm.evaluate(test_window)
            last_test_frozen = model_frozen.evaluate(test_window)

    assert last_test_rewarm < last_test_frozen


def test_assign_and_loss_math():
    """Verify assign_and_loss returns correct labels and MSE."""
    batch = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    centroids = np.array([[0.1, 0.0], [9.9, 10.0]], dtype=np.float32)
    labels, dists, mse = assign_and_loss(batch, centroids)
    assert list(labels) == [0, 1]
    assert mse < 0.05
