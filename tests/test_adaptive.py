"""Unit and integration tests for Track T10: adaptive-k."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure tracks/adaptive-k is in sys.path
TRACK_DIR = Path(__file__).resolve().parent.parent / "tracks" / "adaptive-k"
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

from adaptive import AdaptiveKMeans, compute_bic


def test_bic_computation_monotonicity():
    """Verify compute_bic produces valid finite scores and penalizes huge SSE."""
    counts = np.array([100.0, 100.0], dtype=np.float32)
    sums = np.zeros((2, 4), dtype=np.float32)
    low_sse = np.array([10.0, 10.0], dtype=np.float32)
    high_sse = np.array([1000.0, 1000.0], dtype=np.float32)

    bic_good = compute_bic(counts, sums, low_sse, dim=4)
    bic_bad = compute_bic(counts, sums, high_sse, dim=4)

    assert np.isfinite(bic_good)
    assert np.isfinite(bic_bad)
    assert bic_good < bic_bad


def test_adaptive_kmeans_init():
    """Verify initialization shapes and properties."""
    model = AdaptiveKMeans(dim=8, init_k=6, min_k=2, max_k=20, seed=42)
    assert model.current_k == 6
    assert model.centroids.shape == (6, 8)
    assert model.counts.shape == (6,)
    assert model.sums.shape == (6, 8)
    assert model.sum_sq.shape == (6,)


def test_adaptive_kmeans_single_batch():
    """Verify single batch updates state without errors."""
    model = AdaptiveKMeans(dim=8, init_k=4, seed=42)
    rng = np.random.default_rng(42)
    batch = rng.normal(0.0, 1.0, size=(100, 8)).astype(np.float32)

    loss = model.update_batch(batch)
    assert loss > 0.0
    assert model.step == 1


def test_split_operation_increases_k():
    """Verify high-variance cluster triggers split and increases k."""
    model = AdaptiveKMeans(
        dim=4,
        init_k=2,
        min_k=2,
        max_k=10,
        adapt_interval=1,
        split_threshold=0.5,
        merge_threshold=0.0,
        seed=42,
    )
    # Artificially inject high variance on cluster 0
    model.counts = np.array([50.0, 50.0], dtype=np.float32)
    model.sum_sq = np.array([1000.0, 1.0], dtype=np.float32)

    initial_k = model.current_k
    model._split_and_merge()
    assert model.current_k == initial_k + 1


def test_merge_operation_decreases_k():
    """Verify closely overlapping centroids merge and decrease k."""
    model = AdaptiveKMeans(
        dim=4,
        init_k=3,
        min_k=1,
        max_k=10,
        adapt_interval=1,
        split_threshold=100.0,
        merge_threshold=10.0,
        seed=42,
    )
    # Place cluster 0 and cluster 1 at almost identical positions
    model.centroids[0] = np.zeros(4, dtype=np.float32)
    model.centroids[1] = np.zeros(4, dtype=np.float32) + 1e-4
    model.centroids[2] = np.ones(4, dtype=np.float32) * 10.0
    model.counts = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    model.sum_sq = np.array([5.0, 5.0, 5.0], dtype=np.float32)

    initial_k = model.current_k
    model._split_and_merge()
    assert model.current_k == initial_k - 1


def test_bounds_respected():
    """Verify split and merge respect min_k and max_k."""
    model_min = AdaptiveKMeans(dim=2, init_k=2, min_k=2, max_k=4, seed=42)
    model_min.centroids[0] = model_min.centroids[1]  # Force merge condition
    model_min.merge_threshold = 1000.0
    model_min.split_threshold = 1000.0
    model_min._split_and_merge()
    assert model_min.current_k == 2  # Cannot drop below min_k=2

    model_max = AdaptiveKMeans(dim=2, init_k=4, min_k=2, max_k=4, seed=42)
    model_max.sum_sq[0] = 1e6  # Force split condition
    model_max.split_threshold = 0.01
    model_max.merge_threshold = 0.0
    model_max._split_and_merge()
    assert model_max.current_k == 4  # Cannot exceed max_k=4


def test_adaptive_grows_towards_target_k():
    """Verify model initialized with k=2 grows when trained on 8 well-separated clusters."""
    rng = np.random.default_rng(42)
    k_true = 8
    dim = 8
    centers = rng.normal(0.0, 5.0, size=(k_true, dim)).astype(np.float32)

    model = AdaptiveKMeans(
        dim=dim,
        init_k=2,
        min_k=2,
        max_k=16,
        total_steps=40,
        adapt_interval=2,
        split_threshold=1.1,
        merge_threshold=0.5,
        seed=42,
    )

    for _ in range(40):
        labels = rng.integers(0, k_true, size=200)
        batch = centers[labels] + rng.normal(0.0, 0.2, size=(200, dim)).astype(np.float32)
        model.update_batch(batch)

    assert model.current_k > 2
    assert model.current_k <= 16


def test_stability_across_seeds():
    """Verify chosen k has low coefficient of variation across seeds (< 20%)."""
    ks = []
    dim = 6
    k_true = 6

    for seed in [101, 102, 103]:
        rng = np.random.default_rng(seed)
        centers = rng.normal(0.0, 5.0, size=(k_true, dim)).astype(np.float32)
        model = AdaptiveKMeans(
            dim=dim,
            init_k=3,
            min_k=2,
            max_k=12,
            total_steps=50,
            adapt_interval=4,
            split_threshold=1.15,
            merge_threshold=0.6,
            seed=seed,
        )
        for _ in range(50):
            labels = rng.integers(0, k_true, size=200)
            batch = centers[labels] + rng.normal(0.0, 0.2, size=(200, dim)).astype(np.float32)
            model.update_batch(batch)
        ks.append(model.current_k)

    cv = float(np.std(ks) / np.mean(ks))
    assert cv < 0.20, f"CV was {cv*100:.1f}%, expected < 20%"
