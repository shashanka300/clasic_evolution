"""Unit and integration tests for Track T05: lda-scaling."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure tracks/lda-scaling is in sys.path
TRACK_DIR = Path(__file__).resolve().parent.parent / "tracks" / "lda-scaling"
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

from lda import OnlineLDA, dirichlet_expectation, predict_lda_alpha0


def test_dirichlet_expectation_math():
    """Verify dirichlet_expectation properties."""
    alpha = np.array([1.0, 2.0, 5.0], dtype=np.float32)
    e_log = dirichlet_expectation(alpha)
    assert e_log.shape == (3,)
    assert np.all(e_log < 0.0)  # log of probabilities in (0, 1) is negative
    assert e_log[0] < e_log[1] < e_log[2]  # monotonic in alpha


def test_predict_lda_alpha0_bounds():
    """Verify predicted alpha0 stays within [0.05, 0.95] across regimes."""
    a1 = predict_lda_alpha0(n_docs=1000, n_topics=10, vocab_size=500, batch_size=32)
    a2 = predict_lda_alpha0(n_docs=100000, n_topics=50, vocab_size=10000, batch_size=256)
    assert 0.05 <= a1 <= 0.95
    assert 0.05 <= a2 <= 0.95
    assert a1 > a2  # smaller corpus can afford larger initial step size


def test_online_lda_initialization():
    """Verify OnlineLDA initialization parameters and shapes."""
    model = OnlineLDA(n_topics=8, vocab_size=100, n_docs_total=500, seed=42)
    assert model.lambda_mat.shape == (8, 100)
    assert np.all(model.lambda_mat > 0.0)
    assert model.step == 0
    assert model.revivals == 0


def test_e_step_shapes_and_validity():
    """Verify variational E-step outputs valid probabilities and statistics."""
    model = OnlineLDA(n_topics=5, vocab_size=50, n_docs_total=200, seed=42)
    batch_docs = [
        {0: 2.0, 5: 1.0, 12: 3.0},
        {1: 4.0, 2: 1.0},
        {5: 2.0, 10: 1.0, 25: 1.0},
    ]
    gamma, sstats = model._e_step(batch_docs)
    assert gamma.shape == (3, 5)
    assert sstats.shape == (5, 50)
    assert np.all(gamma > 0.0)
    assert np.all(sstats >= 0.0)


def test_update_batch_updates_lambda():
    """Verify update_batch modifies lambda_mat and advances step counter."""
    model = OnlineLDA(n_topics=4, vocab_size=40, n_docs_total=100, seed=42)
    old_lambda = model.lambda_mat.copy()
    batch_docs = [{0: 3.0, 1: 2.0}, {2: 1.0, 3: 4.0}]
    
    g_mean = model.update_batch(batch_docs)
    assert g_mean > 0.0
    assert not np.allclose(model.lambda_mat, old_lambda)
    assert model.step == 1


def test_topic_revival_splits_donor():
    """Verify _revive_dead_topics detects low-mass topics and splits live donors."""
    model = OnlineLDA(n_topics=4, vocab_size=20, n_docs_total=100, seed=42)
    # Force topic 0 to be dead
    model.lambda_mat[0] = 1e-5
    # Force topic 1 to be heavy donor
    model.lambda_mat[1] = 100.0

    initial_revivals = model.revivals
    model._revive_dead_topics()

    assert model.revivals > initial_revivals
    assert np.sum(model.lambda_mat[0]) > 1.0  # Revived


def test_compute_perplexity_finite():
    """Verify held-out perplexity computation produces finite positive number."""
    model = OnlineLDA(n_topics=5, vocab_size=30, n_docs_total=100, seed=42)
    test_docs = [{0: 2.0, 1: 1.0}, {2: 3.0, 5: 1.0}]
    perp = model.compute_perplexity(test_docs)
    assert np.isfinite(perp)
    assert perp > 1.0

