"""Unit tests for Track T13: drift-balanced-ivf."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure tracks/drift-balanced-ivf is in sys.path
TRACK_DIR = Path(__file__).resolve().parent.parent / "tracks" / "drift-balanced-ivf"
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

from drift_balanced_ivf import LiveDriftBalancedIVF, IndexStats


def test_initialization_and_insertion():
    """Verify live balanced IVF index creation, batch insertion, and list balance."""
    rng = np.random.default_rng(42)
    index = LiveDriftBalancedIVF(d=8, k=4, sinkhorn_reg=0.1)

    batch1 = rng.normal(size=(40, 8)).astype(np.float32)
    ids1 = np.arange(40)
    index.insert_batch(batch1, ids1, adapt=True)

    assert index.is_initialized
    total_docs = sum(len(l) for l in index.invlists)
    assert total_docs == 40

    list_sizes = [len(l) for l in index.invlists]
    assert max(list_sizes) <= 25  # Substantially more balanced than hard nearest-neighbor collapse


def test_query_and_retrieval():
    """Verify querying retrieves valid nearest neighbors."""
    rng = np.random.default_rng(123)
    index = LiveDriftBalancedIVF(d=4, k=4, nlist_probe=4)

    data = rng.normal(size=(50, 4)).astype(np.float32)
    ids = np.arange(50)
    index.insert_batch(data, ids, adapt=False)

    # Query one of the indexed vectors
    res_ids, lat = index.query(data[0], top_k=5)
    assert len(res_ids) == 5
    assert 0 in res_ids  # The exact vector should be retrieved
    assert lat > 0.0


def test_drift_detection_and_warm_restart():
    """Verify that severe distribution shift triggers CUSUM alarm and restarts schedule."""
    rng = np.random.default_rng(999)
    index = LiveDriftBalancedIVF(d=4, k=4, drift_threshold=2.0, alpha0=0.6)

    # 1. Stationary batches
    for b in range(5):
        batch = rng.normal(0.0, 0.5, size=(30, 4)).astype(np.float32)
        index.insert_batch(batch, np.arange(b * 30, (b + 1) * 30), adapt=True)

    restarts_before = index.restarts_triggered

    # 2. Sudden drift batch (shifted mean + 10.0)
    drifted_batch = rng.normal(10.0, 0.5, size=(30, 4)).astype(np.float32)
    alarm = index.insert_batch(drifted_batch, np.arange(150, 180), adapt=True)

    assert alarm is True
    assert index.restarts_triggered > restarts_before


def test_evaluate_stats():
    """Verify evaluate produces valid IndexStats."""
    rng = np.random.default_rng(42)
    index = LiveDriftBalancedIVF(d=4, k=4, nlist_probe=2)

    data = rng.normal(size=(40, 4)).astype(np.float32)
    index.insert_batch(data, np.arange(40), adapt=False)

    queries = rng.normal(size=(10, 4)).astype(np.float32)
    dummy_gt = np.tile(np.arange(5), (10, 1))

    stats = index.evaluate(queries, dummy_gt, top_k=5)
    assert isinstance(stats, IndexStats)
    assert stats.imbalance_ratio >= 1.0
    assert stats.p99_latency_ms >= 0.0
    assert 0.0 <= stats.recall_at_10 <= 1.0
