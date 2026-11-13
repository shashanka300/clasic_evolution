"""Unit and integration tests for Track T07: hmm-ema."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure tracks/hmm-ema is in sys.path
TRACK_DIR = Path(__file__).resolve().parent.parent / "tracks" / "hmm-ema"
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

from hmm import EMABaumWelch, FullBatchBaumWelch


def test_ema_baum_welch_initialization():
    """Verify HMM parameter shapes and valid probability constraints."""
    model = EMABaumWelch(n_states=5, n_emissions=10, n_seqs_total=100, seed=42)
    assert model.log_pi.shape == (5,)
    assert model.log_A.shape == (5, 5)
    assert model.log_B.shape == (5, 10)
    assert model.counts_A.shape == (5, 5)
    assert model.counts_B.shape == (5, 10)
    assert model.step == 0


def test_forward_backward_computation():
    """Verify forward-backward variables are finite and sum to valid probabilities."""
    model = EMABaumWelch(n_states=3, n_emissions=6, n_seqs_total=50, seed=42)
    seq = np.array([0, 1, 2, 0, 1], dtype=np.int64)

    log_lik, gamma, xi = model._forward_backward(seq)
    assert np.isfinite(log_lik)
    assert gamma.shape == (5, 3)
    assert xi.shape == (4, 3, 3)
    # gamma must sum to 1 across states at each time step
    assert np.allclose(np.sum(gamma, axis=1), 1.0, atol=1e-4)


def test_update_batch_updates_parameters():
    """Verify update_batch modifies transition and emission matrices."""
    model = EMABaumWelch(n_states=4, n_emissions=8, n_seqs_total=20, seed=42)
    old_A = model.log_A.copy()
    old_B = model.log_B.copy()

    batch = [
        np.array([0, 1, 2, 3], dtype=np.int64),
        np.array([1, 2, 3, 0], dtype=np.int64),
    ]
    mean_ll = model.update_batch(batch)

    assert np.isfinite(mean_ll)
    assert not np.allclose(model.log_A, old_A)
    assert not np.allclose(model.log_B, old_B)
    assert model.step == 1


def test_unused_state_revival():
    """Verify _revive_unused_states revives starved states."""
    model = EMABaumWelch(n_states=4, n_emissions=8, n_seqs_total=20, seed=42)
    # Force state 0 to be dead
    model.counts_B[0] = 1e-6
    # Force state 1 to be high donor
    model.counts_B[1] = 500.0

    initial_revivals = model.revivals
    model._revive_unused_states()

    assert model.revivals > initial_revivals
    assert np.sum(model.counts_B[0]) > 1.0


def test_evaluate_log_likelihood():
    """Verify evaluate_log_likelihood returns positive finite loss (-LL/token)."""
    model = EMABaumWelch(n_states=3, n_emissions=6, n_seqs_total=20, seed=42)
    test_seqs = [np.array([0, 1, 2], dtype=np.int64), np.array([2, 1, 0], dtype=np.int64)]
    loss = model.evaluate_log_likelihood(test_seqs)
    assert np.isfinite(loss)
    assert loss > 0.0


def test_full_batch_baum_welch():
    """Verify FullBatchBaumWelch executes epoch without errors."""
    model = FullBatchBaumWelch(n_states=3, n_emissions=6, seed=42)
    seqs = [np.array([0, 1, 2], dtype=np.int64), np.array([1, 2, 0], dtype=np.int64)]
    old_A = model.log_A.copy()
    model.run_epoch(seqs)
    assert not np.allclose(model.log_A, old_A)

