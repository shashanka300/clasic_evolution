"""Unit and integration tests for Track T06: als-ema."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure tracks/als-ema is in sys.path
TRACK_DIR = Path(__file__).resolve().parent.parent / "tracks" / "als-ema"
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

from als import EMAAlternatingLeastSquares, FullBatchALS


def test_ema_als_initialization():
    """Verify factor matrices and sufficient statistics shapes."""
    model = EMAAlternatingLeastSquares(n_users=100, n_items=50, n_factors=8, seed=42)
    assert model.user_factors.shape == (100, 8)
    assert model.item_factors.shape == (50, 8)
    assert model.item_A.shape == (50, 8, 8)
    assert model.item_b.shape == (50, 8)
    assert model.step == 0


def test_update_user_batch_updates_factors():
    """Verify batch updates modify user and item factors."""
    model = EMAAlternatingLeastSquares(n_users=20, n_items=10, n_factors=4, seed=42)
    user_ratings = {
        0: [(1, 4.0), (2, 5.0)],
        1: [(0, 3.0), (3, 2.0)],
    }
    old_u0 = model.user_factors[0].copy()
    old_i1 = model.item_factors[1].copy()

    batch_users = np.array([0, 1])
    mse = model.update_user_batch(batch_users, user_ratings)

    assert mse >= 0.0
    assert not np.allclose(model.user_factors[0], old_u0)
    assert not np.allclose(model.item_factors[1], old_i1)
    assert model.step == 1


def test_compute_rmse_calculation():
    """Verify RMSE returns exact zero for perfectly reconstructed factors."""
    model = EMAAlternatingLeastSquares(n_users=2, n_items=2, n_factors=2, seed=42)
    model.user_factors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    model.item_factors = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    
    test_triplets = [(0, 0, 2.0), (1, 1, 3.0)]
    rmse = model.compute_rmse(test_triplets)
    assert abs(rmse) < 1e-5


def test_full_batch_als_epoch():
    """Verify FullBatchALS runs an epoch and updates factors."""
    model = FullBatchALS(n_users=15, n_items=10, n_factors=4, seed=42)
    user_ratings = {u: [(i, 3.0)] for u in range(15) for i in range(2)}
    item_ratings = {i: [(u, 3.0)] for i in range(10) for u in range(2)}

    old_u = model.user_factors.copy()
    model.run_epoch(user_ratings, item_ratings)
    assert not np.allclose(model.user_factors, old_u)


def test_cold_item_handles_gracefully():
    """Verify items with no ratings do not cause NaN or inversion failures."""
    model = EMAAlternatingLeastSquares(n_users=10, n_items=5, n_factors=4, seed=42)
    # Item 4 has zero ratings
    user_ratings = {u: [(0, 1.0)] for u in range(10)}
    model.update_user_batch(np.arange(10), user_ratings)

    assert not np.any(np.isnan(model.item_factors))
    assert not np.any(np.isinf(model.item_factors))


def test_ema_als_reduces_loss():
    """Verify multiple updates systematically improve test RMSE."""
    n_users = 50
    n_items = 20
    model = EMAAlternatingLeastSquares(n_users=n_users, n_items=n_items, n_factors=4, total_steps=20, seed=42)
    
    # Simple consistent ratings
    user_ratings = {u: [(i, float((u + i) % 5 + 1)) for i in range(5)] for u in range(n_users)}
    test_triplets = [(u, 0, float((u + 0) % 5 + 1)) for u in range(10)]

    init_rmse = model.compute_rmse(test_triplets)
    for _ in range(15):
        batch = np.random.choice(n_users, size=10, replace=False)
        model.update_user_batch(batch, user_ratings)
    
    final_rmse = model.compute_rmse(test_triplets)
    assert final_rmse < init_rmse

