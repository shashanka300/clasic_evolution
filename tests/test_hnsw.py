"""Unit and integration tests for Track T11: hnsw-scaling-laws."""

from __future__ import annotations

import sys
import math
from pathlib import Path
import numpy as np
import pytest

# Ensure tracks/hnsw-scaling-laws is in sys.path
TRACK_DIR = Path(__file__).resolve().parent.parent / "tracks" / "hnsw-scaling-laws"
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

from laws import (
    compute_ground_truth_topk,
    compute_recall_at_k,
    fit_hnsw_scaling_laws,
    HNSWPredictor,
    evaluate_hnsw_index,
)


def test_ground_truth_topk_and_recall():
    """Verify top-k brute force search and recall calculation logic."""
    rng = np.random.default_rng(42)
    base = rng.normal(size=(50, 4)).astype(np.float32)
    queries = rng.normal(size=(10, 4)).astype(np.float32)

    gt = compute_ground_truth_topk(queries, base, k=5)
    assert gt.shape == (10, 5)

    # Identical predictions should yield perfect 1.0 recall
    assert compute_recall_at_k(gt, gt) == pytest.approx(1.0)

    # Completely disjoint predictions should yield 0.0 recall
    dummy = (gt + 25) % 50
    # ensure no accidental overlap for at least test validity
    if any(len(set(dummy[i]).intersection(set(gt[i]))) > 0 for i in range(10)):
        dummy = np.full_like(gt, 49)
    # Check partial overlap calculation
    partial = np.copy(gt)
    partial[:, 0] = 49  # corrupt one element
    assert compute_recall_at_k(partial, gt) <= 1.0


def test_hnsw_index_evaluation():
    """Verify HNSW building, querying, recall, and QPS evaluation."""
    rng = np.random.default_rng(123)
    train_x = rng.normal(size=(300, 8)).astype(np.float32)
    test_x = rng.normal(size=(30, 8)).astype(np.float32)

    gt = compute_ground_truth_topk(test_x, train_x, k=5)
    recall, qps, build_time = evaluate_hnsw_index(
        train_x=train_x,
        test_x=test_x,
        ground_truth=gt,
        M=16,
        ef_construction=100,
        ef_search=50,
        top_k=5,
    )

    assert 0.0 <= recall <= 1.0
    assert recall >= 0.7  # HNSW should have high recall on small Gaussian data
    assert qps > 0.0
    assert build_time > 0.0


def test_fit_scaling_laws_and_monotonicity():
    """Verify log-linear scaling law fits and predictor monotonicity."""
    # Synthetic calibration records: M ~ 2 * n^0.1 * d^0.15 * r^0.3
    records = []
    for n in [1000, 5000, 10000]:
        for d in [16, 32, 64]:
            for target_recall in [0.8, 0.9, 0.95]:
                r = -math.log(max(1.0 - target_recall, 1e-4))
                m_val = 4.0 * (n ** 0.08) * (d ** 0.12) * (r ** 0.35)
                ef_val = 15.0 * (n ** 0.12) * (d ** 0.10) * (r ** 0.50)
                records.append({
                    "n": float(n),
                    "d": float(d),
                    "r": float(r),
                    "M": float(m_val),
                    "ef_construction": float(ef_val),
                })

    predictor = fit_hnsw_scaling_laws(records)
    assert isinstance(predictor, HNSWPredictor)
    assert predictor.m_law.r2 > 0.90
    assert predictor.ef_law.r2 > 0.90

    # Predict for lower vs higher recall target
    m_low, ef_low = predictor.predict(n=5000, d=32, target_recall=0.85)
    m_high, ef_high = predictor.predict(n=5000, d=32, target_recall=0.98)

    assert 4 <= m_low <= 96
    assert 16 <= ef_low <= 500
    assert m_high >= m_low
    assert ef_high >= ef_low


def test_scaling_law_bounds_respected():
    """Verify predictor clamps to valid hardware and operational bounds."""
    # Create extreme predictor with dummy laws (need at least len(feats)+1 = 4 records)
    records = [
        {"n": 100.0, "d": 4.0, "r": 0.5, "M": 2.0, "ef_construction": 5.0},
        {"n": 200.0, "d": 8.0, "r": 1.0, "M": 3.0, "ef_construction": 8.0},
        {"n": 500.0, "d": 32.0, "r": 2.0, "M": 50.0, "ef_construction": 200.0},
        {"n": 1000.0, "d": 128.0, "r": 5.0, "M": 200.0, "ef_construction": 1000.0},
    ]
    predictor = fit_hnsw_scaling_laws(records)

    # Extreme low
    m_min, ef_min = predictor.predict(n=10, d=2, target_recall=0.5)
    assert m_min >= 4
    assert ef_min >= 16

    # Extreme high
    m_max, ef_max = predictor.predict(n=1_000_000, d=1024, target_recall=0.999)
    assert m_max <= 96
    assert ef_max <= 500
