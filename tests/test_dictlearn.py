"""Comprehensive unit test suite for Track T04: dictlearn-scaling.

Tests cover:
1. test_fista_convergence_and_reconstruction
2. test_fista_sparsity_spectrum
3. test_sufficient_stats_accumulation
4. test_dictionary_update_unit_norm_invariant
5. test_nmf_non_negativity_invariants
6. test_dead_atom_detection_and_revival
7. test_dead_atom_revival_nmf_clipping
8. test_scaling_law_fitting
9. test_edge_case_zero_diagonal_akk
10. test_edge_case_all_zero_batch
11. test_edge_case_dimension_mismatch
12. test_cli_smoke_execution_and_jsonl_schema
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

# Ensure tracks/dictlearn-scaling is on sys.path
TEST_DIR = Path(__file__).resolve().parent
WT_DIR = TEST_DIR.parent
TRACK_DIR = WT_DIR / "tracks" / "dictlearn-scaling"
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))
if str(WT_DIR) not in sys.path:
    sys.path.insert(0, str(WT_DIR))

from core.bench import REQUIRED_RESULT_KEYS
from core.scaling import fit_log_linear
from core.stats import BatchStats, EMAStats
from dictlearn import (
    OnlineDictionaryLearning,
    OnlineNMF,
    block_coordinate_descent_update,
    fista_sparse_code,
    revive_dead_atoms,
)
from plugin import (
    DictionaryLearningPlugin,
    NMFPlugin,
    fit_dictlearn_emeans,
    fit_nmf_emeans,
)


def test_fista_convergence_and_reconstruction():
    """Test Case 1: FISTA solves LASSO sparse coding and reconstructs ground-truth signal."""
    rng = np.random.default_rng(42)
    b, d_dim, k = 32, 16, 8

    # Ground truth dictionary and sparse codes
    d_gt = rng.standard_normal((k, d_dim))
    d_gt /= np.linalg.norm(d_gt, axis=1, keepdims=True)

    mask = rng.uniform(0.0, 1.0, (b, k)) < 0.25
    alpha_gt = rng.standard_normal((b, k)) * mask
    x = alpha_gt @ d_gt + 0.01 * rng.standard_normal((b, d_dim))

    alpha = fista_sparse_code(x, d_gt, alpha_reg=0.01, max_iter=50, tol=1e-5)

    assert alpha.shape == (b, k)
    assert not np.any(np.isnan(alpha))
    assert not np.any(np.isinf(alpha))

    # Initial loss at alpha = 0 vs converged loss
    loss_0 = 0.5 * np.sum(x**2)
    x_hat = alpha @ d_gt
    loss_final = 0.5 * np.sum((x - x_hat) ** 2) + 0.01 * np.sum(np.abs(alpha))
    assert loss_final < loss_0

    # Relative reconstruction error on clean signal
    rel_err = np.linalg.norm(x - x_hat) / np.linalg.norm(x)
    assert rel_err < 0.25


def test_fista_sparsity_spectrum():
    """Test Case 2: Increasing regularization lambda strictly increases sparsity."""
    rng = np.random.default_rng(123)
    b, d_dim, k = 40, 20, 10

    d = rng.standard_normal((k, d_dim))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    x = rng.standard_normal((b, d_dim))

    lambdas = [0.001, 0.05, 0.2, 5.0]
    sparsity_levels = []

    for lam in lambdas:
        alpha = fista_sparse_code(x, d, alpha_reg=lam, max_iter=50)
        nnz_ratio = np.mean(alpha != 0.0)
        sparsity_levels.append(nnz_ratio)

    # Monotonically non-increasing non-zero fraction
    for i in range(len(sparsity_levels) - 1):
        assert sparsity_levels[i] >= sparsity_levels[i + 1]

    # Extreme lambda shrinks all codes to exact zeros
    alpha_extreme = fista_sparse_code(x, d, alpha_reg=100.0, max_iter=30)
    assert np.all(alpha_extreme == 0.0)


def test_sufficient_stats_accumulation():
    """Test Case 3: EMA accumulator preserves symmetry, PSD, and exact update rule."""
    rng = np.random.default_rng(456)
    k, d_dim, b = 8, 16, 32

    alpha = rng.standard_normal((b, k))
    x = rng.standard_normal((b, d_dim))

    a_batch = (alpha.T @ alpha) / b
    b_batch = (alpha.T @ x) / b
    c_batch = np.sum(np.abs(alpha), axis=0) / b

    stats = EMAStats(
        counts=np.ones(k, dtype=np.float64) * 0.1,
        sums=rng.standard_normal((k, d_dim)).astype(np.float64),
        extras={"A": np.eye(k, dtype=np.float64) * 0.5},
    )

    a_prev = stats.extras["A"].copy()
    alpha_step = 0.2

    bstats = BatchStats(
        counts=c_batch.astype(np.float64),
        sums=b_batch.astype(np.float64),
        extras={"A": a_batch.astype(np.float64)},
    )
    stats.update(bstats, alpha=alpha_step)

    # Check exact EMA linear combination
    expected_a = (1.0 - alpha_step) * a_prev + alpha_step * a_batch
    np.testing.assert_allclose(stats.extras["A"], expected_a, rtol=1e-6)

    # Check symmetry and positive semi-definiteness
    a_mat = 0.5 * (stats.extras["A"] + stats.extras["A"].T)
    np.testing.assert_allclose(stats.extras["A"], a_mat, atol=1e-7)
    min_eig = np.min(np.linalg.eigvalsh(a_mat))
    assert min_eig >= -1e-7


def test_dictionary_update_unit_norm_invariant():
    """Test Case 4: BCD atom update strictly enforces unit ball constraint ||d_k||_2 <= 1."""
    rng = np.random.default_rng(789)
    k, d_dim = 12, 24

    d_init = rng.standard_normal((k, d_dim))
    d_init /= np.linalg.norm(d_init, axis=1, keepdims=True)

    # Generate positive definite A and arbitrary B
    w = rng.standard_normal((k, k))
    a = w.T @ w + 0.1 * np.eye(k)
    b = rng.standard_normal((k, d_dim))

    def surrogate(dict_mat):
        return 0.5 * np.trace(dict_mat.T @ a @ dict_mat) - np.trace(dict_mat.T @ b)

    surr_before = surrogate(d_init)
    d_up = block_coordinate_descent_update(d_init, a, b, n_iter=5)

    # Assert unit ball constraint
    norms = np.linalg.norm(d_up, axis=1)
    assert np.all(norms <= 1.0 + 1e-6)

    # Assert surrogate objective did not increase
    surr_after = surrogate(d_up)
    assert surr_after <= surr_before + 1e-6


def test_nmf_non_negativity_invariants():
    """Test Case 5: OnlineNMF guarantees alpha >= 0 and D >= 0."""
    rng = np.random.default_rng(321)
    n, d_dim, k = 100, 16, 6

    # Non-negative synthetic data
    x = np.abs(rng.standard_normal((n, d_dim))).astype(np.float32)

    nmf = OnlineNMF(n_components=k, alpha_reg=0.01, max_iter=3.0, batch_size=32, random_state=42)
    nmf.fit(x)

    # Dictionary atoms non-negative and bounded
    assert nmf.components_ is not None
    assert np.all(nmf.components_ >= 0.0)
    norms = np.linalg.norm(nmf.components_, axis=1)
    assert np.all(norms <= 1.0 + 1e-6)

    # Transform produces non-negative codes
    codes = nmf.transform(x[:20])
    assert np.all(codes >= 0.0)
    assert nmf.loss(x) >= 0.0


def test_dead_atom_detection_and_revival():
    """Test Case 6: Dead atoms are revived using residual of highest-error sample."""
    rng = np.random.default_rng(654)
    k, d_dim, b = 6, 12, 16

    d = rng.standard_normal((k, d_dim))
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    # Set atom 2 as completely dead
    counts = np.array([10.0, 12.0, 0.0, 11.0, 9.0, 15.0], dtype=np.float64)
    stats_a = np.eye(k, dtype=np.float64) * 5.0
    stats_a[2, :] = 0.0
    stats_a[:, 2] = 0.0
    stats_b = stats_a @ d

    # Artificial batch where sample 5 has huge residual
    residuals = rng.standard_normal((b, d_dim)) * 0.1
    target_sample = 5
    residuals[target_sample] = np.ones(d_dim) * 2.0  # Dominant residual

    dead_indices = np.array([2], dtype=np.int64)
    revive_dead_atoms(
        d=d,
        stats_a=stats_a,
        stats_b=stats_b,
        counts=counts,
        dead_indices=dead_indices,
        residuals=residuals,
    )

    # Atom 2 revived in the direction of dominant residual
    expected_dir = residuals[target_sample] / np.linalg.norm(residuals[target_sample])
    np.testing.assert_allclose(d[2], expected_dir, atol=1e-5)

    # Statistics reset to equilibrium
    assert counts[2] > 0.0
    assert stats_a[2, 2] > 0.0
    assert np.all(stats_a[2, :2] == 0.0)
    assert np.all(stats_a[2, 3:] == 0.0)
    np.testing.assert_allclose(stats_b[2], stats_a[2, 2] * d[2], atol=1e-5)


def test_dead_atom_revival_nmf_clipping():
    """Test Case 7: Dead atom revival in NMF preserves non-negativity."""
    k, d_dim = 4, 8
    d = np.ones((k, d_dim), dtype=np.float32) / np.sqrt(d_dim)
    stats_a = np.eye(k, dtype=np.float64)
    stats_b = d.copy().astype(np.float64)
    counts = np.array([5.0, 0.0, 4.0, 6.0], dtype=np.float64)

    # Residuals with negative components
    residuals = np.array([[-1.0, 2.0, -3.0, 4.0, -0.5, 1.5, -2.0, 0.0]])
    dead_indices = np.array([1], dtype=np.int64)

    revive_dead_atoms(
        d=d,
        stats_a=stats_a,
        stats_b=stats_b,
        counts=counts,
        dead_indices=dead_indices,
        non_negative=True,
        residuals=residuals,
    )

    assert np.all(d[1] >= 0.0)
    assert np.isclose(np.linalg.norm(d[1]), 1.0, atol=1e-5)


def test_scaling_law_fitting():
    """Test Case 8: fit_log_linear calibrates empirical scaling law with R2 >= 0.90."""
    records = []
    # Synthetic power-law scaling: alpha0 = 0.5 * n^(-0.2) * k^(0.1) * T^(-0.3) * B^(0.2)
    for n in [200, 500, 1000]:
        for k in [8, 16, 32]:
            for t in [1.0, 3.0]:
                for b in [32, 64]:
                    alpha0_true = 0.5 * (n ** -0.2) * (k ** 0.1) * (t ** -0.3) * (b ** 0.2)
                    records.append({
                        "n": n,
                        "k_or_units": k,
                        "epochs": t,
                        "batch": b,
                        "alpha0": alpha0_true,
                    })

    law = fit_log_linear(records, target="alpha0", features=("n", "k_or_units", "epochs", "batch"))

    assert law.target == "alpha0"
    assert law.r2 >= 0.95
    assert -0.25 <= law.coeffs["n"] <= -0.15
    assert 0.05 <= law.coeffs["k_or_units"] <= 0.15

    pred = law.predict(n=500, k_or_units=16, epochs=2.0, batch=64)
    assert 0.01 <= pred <= 1.0


def test_edge_case_zero_diagonal_akk():
    """Test Case 9: Inactive atom with Akk = 0 does not divide by zero or emit NaN."""
    k, d_dim = 5, 10
    d_init = np.random.standard_normal((k, d_dim)).astype(np.float32)
    d_init /= np.linalg.norm(d_init, axis=1, keepdims=True)

    a = np.eye(k, dtype=np.float64)
    a[3, 3] = 0.0  # Zero diagonal element
    b = np.random.standard_normal((k, d_dim)).astype(np.float64)

    d_up = block_coordinate_descent_update(d_init, a, b, eps=1e-12)

    assert not np.any(np.isnan(d_up))
    assert not np.any(np.isinf(d_up))
    assert np.all(np.linalg.norm(d_up, axis=1) <= 1.0 + 1e-6)


def test_edge_case_all_zero_batch():
    """Test Case 10: All-zero data batch handles cleanly without NaN."""
    plugin = DictionaryLearningPlugin(n_components=4, alpha_reg=0.1)
    x0 = np.random.standard_normal((20, 8))
    stats = plugin.init(x0)

    # All-zero batch
    zero_batch = np.zeros((10, 8), dtype=np.float32)
    bstats = plugin.assign(zero_batch)

    assert np.all(bstats.counts == 0.0)
    assert np.all(bstats.sums == 0.0)
    assert np.all(bstats.extras["A"] == 0.0)

    # Update does not crash
    stats.update(bstats, alpha=0.1)
    params = plugin.params_from_stats(stats)
    assert not np.any(np.isnan(params))


def test_edge_case_dimension_mismatch():
    """Test Case 11: Dimension mismatch raises explicit ValueError."""
    plugin = DictionaryLearningPlugin(n_components=4, alpha_reg=0.1)
    x0 = np.random.standard_normal((20, 8))
    plugin.init(x0)

    # Pass wrong dimension d=12 instead of d=8
    wrong_batch = np.random.standard_normal((5, 12))
    with pytest.raises(ValueError, match="Input dimension"):
        plugin.assign(wrong_batch)

    with pytest.raises(ValueError, match="Input dimension"):
        plugin.loss(wrong_batch)


def test_cli_smoke_execution_and_jsonl_schema():
    """Test Case 12: CLI entry point executes with smoke arguments and writes valid 15-key JSONL."""
    bench_script = TRACK_DIR / "bench.py"
    assert bench_script.is_file(), f"bench.py missing at {bench_script}"

    cmd = [sys.executable, str(bench_script), "--smoke", "--seeds", "1", "--epochs", "1"]
    res = subprocess.run(cmd, cwd=str(WT_DIR), capture_output=True, text=True, timeout=15)

    assert res.returncode == 0, f"bench.py failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"

    # Verify JSONL records
    jsonl_path = WT_DIR / "results" / "dictlearn-scaling" / "benchmarks.jsonl"
    assert jsonl_path.is_file(), f"benchmarks.jsonl missing at {jsonl_path}"

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) > 0, "benchmarks.jsonl is empty"
    last_record = json.loads(lines[-1])

    # Validate strict 15-key schema
    for key in REQUIRED_RESULT_KEYS:
        assert key in last_record, f"Missing required key: {key}"
    assert len(last_record) == 15, f"Expected 15 keys, got {len(last_record)}: {list(last_record.keys())}"

    assert last_record["track"] == "dictlearn-scaling"
    assert isinstance(last_record["n"], int) and last_record["n"] > 0
    assert isinstance(last_record["d"], int) and last_record["d"] > 0
    assert isinstance(last_record["k_or_units"], int) and last_record["k_or_units"] > 0
    assert isinstance(last_record["epochs"], (int, float)) and last_record["epochs"] > 0
    assert isinstance(last_record["batch"], int) and last_record["batch"] > 0
    assert isinstance(last_record["seed"], int) and last_record["seed"] >= 0
    assert isinstance(last_record["train_loss"], (int, float)) and last_record["train_loss"] >= 0
    assert isinstance(last_record["test_loss"], (int, float)) and last_record["test_loss"] >= 0
    assert isinstance(last_record["wall_s"], (int, float)) and last_record["wall_s"] >= 0
    assert isinstance(last_record["extra"], dict)

    # Verify LOG.md header
    log_path = WT_DIR / "results" / "dictlearn-scaling" / "LOG.md"
    assert log_path.is_file(), f"LOG.md missing at {log_path}"
    with open(log_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    assert first_line == "# Track T04 dictlearn-scaling log"
